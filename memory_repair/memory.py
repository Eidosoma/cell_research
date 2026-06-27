"""Finite local-memory wrappers for E04 S01.

The E03 policy interface already stores mutable per-cell policy state. This
module adds an explicit, bounded memory contract on top of that interface
without changing the validated no-memory semantics. Memory is carried by the
cell, moves with the cell during swaps, and is stored under a reserved key so it
does not collide with E03 policy state such as Selection's ``ideal_position`` or
DSL ``target_position``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from morphospace import (
    LocalObservation,
    LocalRulePolicy,
    PolicyCell,
    PolicyEventSimulator,
    PolicySpec,
    ProposedAction,
    policy_from_spec,
)
from e02_deterministic_simulator.simulator import StepOutcome


MEMORY_REPAIR_VERSION = "e04_s01_memory_extension.v1"
MEMORY_STATE_KEY = "__e04_memory__"
MEMORY_VARIANTS = ("no_memory", "one_bit", "bounded_counter", "neighbor_memory")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


@dataclass(frozen=True)
class MemoryConfig:
    """Bounded local-memory configuration for one policy wrapper."""

    variant: str = "no_memory"
    counter_max: int = 7
    neighbor_history: int = 4

    def __post_init__(self) -> None:
        if self.variant not in MEMORY_VARIANTS:
            raise ValueError(f"memory variant must be one of {MEMORY_VARIANTS}, got {self.variant!r}")
        if int(self.counter_max) < 1:
            raise ValueError("counter_max must be positive")
        if int(self.neighbor_history) < 1:
            raise ValueError("neighbor_history must be positive")

    @classmethod
    def from_spec(cls, payload: Mapping[str, Any] | str | "MemoryConfig") -> "MemoryConfig":
        if isinstance(payload, MemoryConfig):
            return payload
        if isinstance(payload, str):
            return cls(variant=payload)
        return cls(
            variant=str(payload.get("variant", "no_memory")),
            counter_max=int(payload.get("counterMax", payload.get("counter_max", 7))),
            neighbor_history=int(payload.get("neighborHistory", payload.get("neighbor_history", 4))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "counterMax": int(self.counter_max),
            "neighborHistory": int(self.neighbor_history),
            "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        }


def initial_memory_state(config: MemoryConfig | Mapping[str, Any] | str) -> dict[str, Any]:
    """Return the reset value for a bounded local-memory variant."""

    config = MemoryConfig.from_spec(config)
    if config.variant == "no_memory":
        return {}
    if config.variant == "one_bit":
        return {"last_move_success": False}

    state: dict[str, Any] = {
        "last_move_success": False,
        "recent_failed_swaps": 0,
        "time_since_movement": 0,
        "local_frustration": 0,
        "last_failed_target_cell_id": None,
        "last_failed_target_position": None,
    }
    if config.variant == "neighbor_memory":
        state.update(
            {
                "last_left_neighbor_id": None,
                "last_right_neighbor_id": None,
                "recent_neighbor_ids": [],
            }
        )
    return state


def serialize_memory_state(memory_state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-ready copy of a memory state."""

    return _json_ready(dict(memory_state))


def memory_state_for_trace(policy_state: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only the reserved E04 memory state from a policy state."""

    value = policy_state.get(MEMORY_STATE_KEY, {})
    if isinstance(value, Mapping):
        return serialize_memory_state(value)
    return {}


def reset_memory_state(
    policy_state: dict[str, Any],
    config: MemoryConfig | Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Reset the reserved memory key in a mutable per-cell policy state."""

    config = MemoryConfig.from_spec(config)
    if config.variant == "no_memory":
        policy_state.pop(MEMORY_STATE_KEY, None)
    else:
        policy_state[MEMORY_STATE_KEY] = initial_memory_state(config)
    return policy_state


def _base_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in state.items() if key != MEMORY_STATE_KEY}


def _target_cell_id(observation: LocalObservation, target_position: int | None) -> int | None:
    if target_position is None:
        return None
    if target_position == observation.actor_position - 1 and observation.left is not None:
        return observation.left.cell_id
    if target_position == observation.actor_position + 1 and observation.right is not None:
        return observation.right.cell_id
    if observation.target_position == target_position and observation.target is not None:
        return observation.target.cell_id
    return None


def _updated_memory_state(
    config: MemoryConfig,
    current: Mapping[str, Any],
    observation: LocalObservation,
    action: ProposedAction,
    outcome: StepOutcome,
) -> dict[str, Any]:
    if config.variant == "no_memory":
        return {}

    if config.variant == "one_bit":
        return {"last_move_success": bool(outcome.swapped)}

    previous = dict(initial_memory_state(config))
    previous.update(dict(current))
    attempted_swap = action.action == "swap" and action.target_position is not None
    swapped = bool(outcome.swapped)

    if swapped:
        failed_swaps = 0
        time_since_movement = 0
        frustration = _clamp(int(previous.get("local_frustration", 0)) - 1, 0, config.counter_max)
        failed_target_cell_id = None
        failed_target_position = None
    else:
        failed_swaps = _clamp(
            int(previous.get("recent_failed_swaps", 0)) + (1 if attempted_swap else 0),
            0,
            config.counter_max,
        )
        time_since_movement = _clamp(
            int(previous.get("time_since_movement", 0)) + 1,
            0,
            config.counter_max,
        )
        frustration = _clamp(
            int(previous.get("local_frustration", 0)) + (1 if attempted_swap or outcome.activated else 0),
            0,
            config.counter_max,
        )
        failed_target_cell_id = _target_cell_id(observation, action.target_position) if attempted_swap else None
        failed_target_position = action.target_position if attempted_swap else None

    updated: dict[str, Any] = {
        "last_move_success": swapped,
        "recent_failed_swaps": failed_swaps,
        "time_since_movement": time_since_movement,
        "local_frustration": frustration,
        "last_failed_target_cell_id": failed_target_cell_id,
        "last_failed_target_position": failed_target_position,
    }
    if config.variant == "neighbor_memory":
        current_neighbors = [
            None if observation.left is None else observation.left.cell_id,
            None if observation.right is None else observation.right.cell_id,
        ]
        history = list(previous.get("recent_neighbor_ids", []))
        history.extend(cell_id for cell_id in current_neighbors if cell_id is not None)
        updated.update(
            {
                "last_left_neighbor_id": current_neighbors[0],
                "last_right_neighbor_id": current_neighbors[1],
                "recent_neighbor_ids": [int(cell_id) for cell_id in history[-int(config.neighbor_history) :]],
            }
        )
    return serialize_memory_state(updated)


class MemoryPolicyWrapper(LocalRulePolicy):
    """LocalRulePolicy wrapper that adds finite cell-local memory."""

    family = "memory_repair"
    version = MEMORY_REPAIR_VERSION

    def __init__(
        self,
        base_policy: str | LocalRulePolicy | PolicySpec | Mapping[str, Any],
        memory_config: MemoryConfig | Mapping[str, Any] | str = "no_memory",
    ) -> None:
        self.base_policy = policy_from_spec(base_policy)
        self.memory_config = MemoryConfig.from_spec(memory_config)
        self.algotype = self.base_policy.algotype
        self.policy_id = f"e04_{self.memory_config.variant}__{self.base_policy.policy_id}"
        super().__init__(
            base_policy=self.base_policy.to_spec().to_dict(),
            memory_config=self.memory_config.to_dict(),
        )

    def to_spec(self) -> PolicySpec:
        return PolicySpec(
            policy_id=self.policy_id,
            family=self.family,
            algotype=self.algotype,
            version=self.version,
            parameters={
                "basePolicy": self.base_policy.to_spec().to_dict(),
                "memoryConfig": self.memory_config.to_dict(),
            },
        )

    def initial_state(
        self,
        *,
        cell_id: int,
        position: int,
        value: int,
        n: int,
        reverse_direction: bool = False,
    ) -> dict[str, Any]:
        state = self.base_policy.initial_state(
            cell_id=cell_id,
            position=position,
            value=value,
            n=n,
            reverse_direction=reverse_direction,
        )
        return reset_memory_state(dict(state), self.memory_config)

    def observe(
        self,
        cells: Sequence[PolicyCell],
        actor_position: int,
        state: Mapping[str, Any],
        frozen_variant: str,
    ) -> LocalObservation:
        return self.base_policy.observe(cells, actor_position, _base_state(state), frozen_variant)

    def propose_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        forced_direction: int | None = None,
    ) -> ProposedAction:
        return self.base_policy.propose_action(
            observation,
            _base_state(state),
            rng,
            forced_direction=forced_direction,
        )

    def constrain_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        action: ProposedAction,
    ) -> ProposedAction:
        return self.base_policy.constrain_action(observation, _base_state(state), action)

    def update_state(
        self,
        state: dict[str, Any],
        observation: LocalObservation,
        action: ProposedAction,
        outcome: StepOutcome,
    ) -> None:
        self.base_policy.update_state(state, observation, action, outcome)
        if self.memory_config.variant == "no_memory":
            state.pop(MEMORY_STATE_KEY, None)
            return
        state[MEMORY_STATE_KEY] = _updated_memory_state(
            self.memory_config,
            memory_state_for_trace(state),
            observation,
            action,
            outcome,
        )

    def legal_action_exists(self, observation: LocalObservation, state: Mapping[str, Any]) -> bool:
        return self.base_policy.legal_action_exists(observation, _base_state(state))


def memory_policy_from_spec(spec: PolicySpec | Mapping[str, Any] | str | LocalRulePolicy) -> LocalRulePolicy:
    """Instantiate a memory wrapper from a serialized spec or return policies unchanged."""

    if isinstance(spec, MemoryPolicyWrapper):
        return spec
    if isinstance(spec, str):
        return MemoryPolicyWrapper(policy_from_spec(spec), "no_memory")
    if isinstance(spec, PolicySpec):
        payload = spec.to_dict()
    elif isinstance(spec, LocalRulePolicy):
        return spec
    else:
        payload = dict(spec)

    if payload.get("family") != "memory_repair":
        return policy_from_spec(payload)
    parameters = dict(payload.get("parameters", {}))
    base_policy = parameters.get("basePolicy") or parameters.get("base_policy")
    memory_config = parameters.get("memoryConfig") or parameters.get("memory_config") or "no_memory"
    if base_policy is None:
        raise ValueError("memory policy specs require parameters.basePolicy")
    return MemoryPolicyWrapper(base_policy, memory_config)


def memory_policy_to_json(policy: LocalRulePolicy) -> str:
    return _compact_json(policy.to_spec().to_dict())


def memory_policy_from_json(payload: str) -> LocalRulePolicy:
    return memory_policy_from_spec(json.loads(payload))


def build_memory_variants(
    base_policy: str | LocalRulePolicy | PolicySpec | Mapping[str, Any],
    variants: Iterable[str] = MEMORY_VARIANTS,
    *,
    counter_max: int = 7,
    neighbor_history: int = 4,
) -> tuple[MemoryPolicyWrapper, ...]:
    return tuple(
        MemoryPolicyWrapper(
            base_policy,
            MemoryConfig(variant=variant, counter_max=counter_max, neighbor_history=neighbor_history),
        )
        for variant in variants
    )


def reset_all_memory(cells: Iterable[PolicyCell]) -> None:
    """Reset memory on all cells whose policies are memory wrappers."""

    for cell in cells:
        if isinstance(cell.policy, MemoryPolicyWrapper):
            reset_memory_state(cell.state, cell.policy.memory_config)


class MemoryEventSimulator(PolicyEventSimulator):
    """PolicyEventSimulator with E04 memory-state trace fields."""

    def __init__(
        self,
        initial_values: Sequence[int],
        policies: str | LocalRulePolicy | PolicySpec | Sequence[str | LocalRulePolicy | PolicySpec | Mapping[str, Any]],
        *,
        trace_memory_activations: bool = True,
        implementation: str = "memory_repair_interface",
        research_step_id: str = "S01",
        **kwargs: Any,
    ) -> None:
        self.trace_memory_activations = bool(trace_memory_activations)
        super().__init__(
            initial_values,
            policies,
            implementation=implementation,
            research_step_id=research_step_id,
            **kwargs,
        )

    def _memory_trace_payload(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for position, cell in enumerate(self.cells):
            policy = cell.policy
            if isinstance(policy, MemoryPolicyWrapper):
                variant = policy.memory_config.variant
                policy_id = policy.policy_id
            else:
                variant = "unwrapped"
                policy_id = policy.policy_id
            rows.append(
                {
                    "position": int(position),
                    "cell_id": int(cell.cell_id),
                    "policy_id": policy_id,
                    "memory_variant": variant,
                    "memory_state": memory_state_for_trace(cell.state),
                }
            )
        return rows

    def _actor_memory_trace(self, actor_cell_id: int | None) -> dict[str, Any]:
        if actor_cell_id is None:
            return {}
        position = self.positions_by_id.get(int(actor_cell_id))
        if position is None:
            return {}
        cell = self.cells[position]
        return {
            "position": int(position),
            "cell_id": int(cell.cell_id),
            "policy_id": cell.policy.policy_id,
            "memory_state": memory_state_for_trace(cell.state),
        }

    def _append_trace_row(
        self,
        *,
        event_kind: str,
        activation_index: int,
        actor_cell_id: int | None,
        actor_algotype: str | None,
        target_position: int | None,
    ) -> None:
        super()._append_trace_row(
            event_kind=event_kind,
            activation_index=activation_index,
            actor_cell_id=actor_cell_id,
            actor_algotype=actor_algotype,
            target_position=target_position,
        )
        row = self.trace_rows[-1]
        payload = self._memory_trace_payload()
        row["memory_schema_version"] = MEMORY_REPAIR_VERSION
        row["memory_states_json"] = _compact_json(payload)
        row["actor_memory_state_json"] = _compact_json(self._actor_memory_trace(actor_cell_id))
        row["memory_variant_counts_json"] = _compact_json(
            Counter(item["memory_variant"] for item in payload)
        )

    def step(
        self,
        *,
        forced_cell_id: int | None = None,
        forced_direction: int | None = None,
    ) -> StepOutcome:
        trace_count_before = len(self.trace_rows)
        outcome = super().step(forced_cell_id=forced_cell_id, forced_direction=forced_direction)
        if (
            self.trace_memory_activations
            and outcome.activated
            and len(self.trace_rows) == trace_count_before
        ):
            actor_algotype = None
            if outcome.actor_cell_id is not None and outcome.actor_cell_id in self.positions_by_id:
                actor_algotype = self.cells[self.positions_by_id[outcome.actor_cell_id]].algotype
            self._append_trace_row(
                event_kind="memory_update",
                activation_index=self.activation_count,
                actor_cell_id=outcome.actor_cell_id,
                actor_algotype=actor_algotype,
                target_position=outcome.target_position,
            )
        return outcome
