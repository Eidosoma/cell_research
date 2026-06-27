"""Local-rule policy interface and E02-compatible policy simulator.

S01's goal is to expose the classic cell-view algorithms through a common
interface while preserving the deterministic E02 reference behavior. The
interface separates five policy responsibilities:

1. observe the local world available to the actor cell,
2. keep per-cell internal state,
3. propose an action,
4. respect action constraints supplied by the world,
5. update state after the attempted action.

The simulator below intentionally mirrors ``e02_deterministic_simulator`` for
Bubble, Insertion, and Selection so that later morphospace work can use this
module as a reusable policy boundary without changing baseline semantics.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from e02_deterministic_simulator.metrics import (
    aggregation,
    monotonicity_error,
    sortedness_percent,
    sortedness_raw,
    state_hash,
)
from e02_deterministic_simulator.simulator import SimulationResult, StepOutcome


FROZEN_VARIANTS = {"none", "passive", "stuck"}
CLASSIC_ALGOTYPES = {"bubble", "insertion", "selection"}
POLICY_INTERFACE_VERSION = "e03_s01_policy_interface.v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


@dataclass(frozen=True)
class PolicySpec:
    """Serializable description of a local-rule policy."""

    policy_id: str
    family: str
    algotype: str
    version: str = POLICY_INTERFACE_VERSION
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "family": self.family,
            "algotype": self.algotype,
            "version": self.version,
            "parameters": _jsonable(self.parameters),
        }


@dataclass(frozen=True)
class CellSnapshot:
    """Read-only cell view exposed through observations."""

    cell_id: int
    value: int
    algotype: str
    label: int = 0
    reverse_direction: bool = False
    frozen: bool = False
    policy_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalObservation:
    """Observation made available to one policy invocation."""

    actor: CellSnapshot
    actor_position: int
    n: int
    left: CellSnapshot | None
    right: CellSnapshot | None
    left_context: tuple[CellSnapshot, ...]
    target_position: int | None = None
    target: CellSnapshot | None = None
    frozen_variant: str = "none"


@dataclass(frozen=True)
class ProposedAction:
    """A policy action proposal before world constraints are applied."""

    action: str
    target_position: int | None = None
    comparison_delta: int = 0
    reason: str = "wait"
    state_updates: dict[str, Any] = field(default_factory=dict)
    swapped_reason: str = "swapped"
    blocked_reason: str = "swap_blocked"

    @classmethod
    def wait(
        cls,
        reason: str,
        *,
        target_position: int | None = None,
        comparison_delta: int = 0,
        state_updates: Mapping[str, Any] | None = None,
    ) -> "ProposedAction":
        return cls(
            "wait",
            target_position=target_position,
            comparison_delta=comparison_delta,
            reason=reason,
            state_updates=dict(state_updates or {}),
        )

    @classmethod
    def swap(
        cls,
        target_position: int,
        *,
        comparison_delta: int = 0,
        state_updates: Mapping[str, Any] | None = None,
        swapped_reason: str = "swapped",
        blocked_reason: str = "swap_blocked",
    ) -> "ProposedAction":
        return cls(
            "swap",
            target_position=target_position,
            comparison_delta=comparison_delta,
            reason="swap",
            state_updates=dict(state_updates or {}),
            swapped_reason=swapped_reason,
            blocked_reason=blocked_reason,
        )


@dataclass
class PolicyCell:
    """Mutable simulator cell whose policy state moves with the cell."""

    cell_id: int
    value: int
    policy: "LocalRulePolicy"
    label: int = 0
    reverse_direction: bool = False
    frozen: bool = False
    state: dict[str, Any] = field(default_factory=dict)
    tried_to_swap_with_frozen: bool = False

    @property
    def algotype(self) -> str:
        return self.policy.algotype

    def snapshot(self) -> CellSnapshot:
        return CellSnapshot(
            cell_id=self.cell_id,
            value=self.value,
            algotype=self.algotype,
            label=self.label,
            reverse_direction=self.reverse_direction,
            frozen=self.frozen,
            policy_state=dict(self.state),
        )


class LocalRulePolicy(ABC):
    """Base class for local policies used by the morphospace simulator."""

    family = "abstract"
    algotype = "abstract"
    policy_id = "abstract"
    version = POLICY_INTERFACE_VERSION

    def __init__(self, **parameters: Any) -> None:
        self.parameters = dict(parameters)

    def to_spec(self) -> PolicySpec:
        return PolicySpec(
            policy_id=self.policy_id,
            family=self.family,
            algotype=self.algotype,
            version=self.version,
            parameters=dict(self.parameters),
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
        return {}

    def observe(
        self,
        cells: Sequence[PolicyCell],
        actor_position: int,
        state: Mapping[str, Any],
        frozen_variant: str,
    ) -> LocalObservation:
        left = cells[actor_position - 1].snapshot() if actor_position > 0 else None
        right = cells[actor_position + 1].snapshot() if actor_position < len(cells) - 1 else None
        return LocalObservation(
            actor=cells[actor_position].snapshot(),
            actor_position=actor_position,
            n=len(cells),
            left=left,
            right=right,
            left_context=tuple(cell.snapshot() for cell in cells[:actor_position]),
            frozen_variant=frozen_variant,
        )

    @abstractmethod
    def propose_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        forced_direction: int | None = None,
    ) -> ProposedAction:
        """Return the next policy action before world-level constraints."""

    def constrain_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        action: ProposedAction,
    ) -> ProposedAction:
        if action.action not in {"swap", "wait"}:
            return ProposedAction.wait("invalid_action", target_position=action.target_position)
        return action

    def update_state(
        self,
        state: dict[str, Any],
        observation: LocalObservation,
        action: ProposedAction,
        outcome: StepOutcome,
    ) -> None:
        state.update(action.state_updates)

    def legal_action_exists(self, observation: LocalObservation, state: Mapping[str, Any]) -> bool:
        return False


def _next_selection_ideal(ideal_position: int, reverse_direction: bool) -> int:
    return ideal_position - 1 if reverse_direction else ideal_position + 1


def _insertion_prefix_enabled(observation: LocalObservation) -> bool:
    previous = 100000 if observation.actor.reverse_direction else -1
    for cell in observation.left_context:
        if cell.frozen:
            previous = -1
            continue
        if observation.actor.reverse_direction and cell.value > previous:
            return False
        if not observation.actor.reverse_direction and cell.value < previous:
            return False
        previous = cell.value
    return True


def _target_allowed_by_frozen_variant(observation: LocalObservation, target: CellSnapshot | None) -> bool:
    return target is not None and (observation.frozen_variant != "stuck" or not target.frozen)


class BubblePolicy(LocalRulePolicy):
    """Cell-view Bubble policy from E01/E02."""

    family = "classic"
    algotype = "bubble"
    policy_id = "classic_bubble"

    def propose_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        forced_direction: int | None = None,
    ) -> ProposedAction:
        if forced_direction is None:
            direction = 1 if rng.random() < 0.5 else -1
        else:
            direction = 1 if forced_direction > 0 else -1
        target_position = observation.actor_position + direction
        target = observation.right if direction == 1 else observation.left
        if target is None:
            return ProposedAction.wait("target_oob", target_position=target_position)
        if observation.actor.reverse_direction:
            should_swap = (
                observation.actor.value < target.value
                if direction == 1
                else observation.actor.value > target.value
            )
        else:
            should_swap = (
                observation.actor.value > target.value
                if direction == 1
                else observation.actor.value < target.value
            )
        if not should_swap:
            return ProposedAction.wait("no_policy_swap", target_position=target_position, comparison_delta=1)
        return ProposedAction.swap(target_position, comparison_delta=1)

    def legal_action_exists(self, observation: LocalObservation, state: Mapping[str, Any]) -> bool:
        actor = observation.actor
        for direction, target in [(-1, observation.left), (1, observation.right)]:
            if target is None or not _target_allowed_by_frozen_variant(observation, target):
                continue
            if actor.reverse_direction:
                should_swap = actor.value < target.value if direction == 1 else actor.value > target.value
            else:
                should_swap = actor.value > target.value if direction == 1 else actor.value < target.value
            if should_swap:
                return True
        return False


class InsertionPolicy(LocalRulePolicy):
    """Cell-view Insertion policy from E01/E02."""

    family = "classic"
    algotype = "insertion"
    policy_id = "classic_insertion"

    def propose_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        forced_direction: int | None = None,
    ) -> ProposedAction:
        if not _insertion_prefix_enabled(observation):
            return ProposedAction.wait("prefix_not_enabled")
        target_position = observation.actor_position - 1
        target = observation.left
        if target is None:
            return ProposedAction.wait("target_oob", target_position=target_position)
        should_swap = (
            observation.actor.value > target.value
            if observation.actor.reverse_direction
            else observation.actor.value < target.value
        )
        if not should_swap:
            return ProposedAction.wait("no_policy_swap", target_position=target_position, comparison_delta=1)
        return ProposedAction.swap(target_position, comparison_delta=1)

    def legal_action_exists(self, observation: LocalObservation, state: Mapping[str, Any]) -> bool:
        target = observation.left
        actor = observation.actor
        if target is None or not _insertion_prefix_enabled(observation):
            return False
        if not _target_allowed_by_frozen_variant(observation, target):
            return False
        if actor.reverse_direction:
            return actor.value > target.value
        return actor.value < target.value


class SelectionPolicy(LocalRulePolicy):
    """Cell-view Selection policy from E01/E02.

    The comparison rule intentionally preserves E02's exact target-position
    semantics, including its reverse-direction asymmetry, because S01 is a
    behavior-preserving wrapper rather than a redesigned Selection policy.
    """

    family = "classic"
    algotype = "selection"
    policy_id = "classic_selection"

    def initial_state(
        self,
        *,
        cell_id: int,
        position: int,
        value: int,
        n: int,
        reverse_direction: bool = False,
    ) -> dict[str, Any]:
        return {"ideal_position": n - 1 if reverse_direction else 0}

    def observe(
        self,
        cells: Sequence[PolicyCell],
        actor_position: int,
        state: Mapping[str, Any],
        frozen_variant: str,
    ) -> LocalObservation:
        base = super().observe(cells, actor_position, state, frozen_variant)
        ideal = state.get("ideal_position")
        target_position = int(ideal) if ideal is not None else None
        target = None
        if target_position is not None and 0 <= target_position < len(cells):
            target = cells[target_position].snapshot()
        return LocalObservation(
            actor=base.actor,
            actor_position=base.actor_position,
            n=base.n,
            left=base.left,
            right=base.right,
            left_context=base.left_context,
            target_position=target_position,
            target=target,
            frozen_variant=frozen_variant,
        )

    def propose_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        forced_direction: int | None = None,
    ) -> ProposedAction:
        ideal = observation.target_position
        if ideal is None or ideal < 0 or ideal >= observation.n:
            return ProposedAction.wait("ideal_oob", target_position=ideal)
        if observation.actor_position == ideal:
            return ProposedAction.wait("already_at_ideal", target_position=ideal)
        target = observation.target
        if target is None:
            return ProposedAction.wait("ideal_oob", target_position=ideal)

        next_ideal = _next_selection_ideal(ideal, observation.actor.reverse_direction)
        if target.frozen:
            updates = {"ideal_position": next_ideal}
            attempted_swap = observation.actor.value < target.value
            if attempted_swap:
                return ProposedAction.swap(
                    ideal,
                    comparison_delta=1,
                    state_updates=updates,
                    swapped_reason="swapped_frozen_target",
                    blocked_reason="frozen_target",
                )
            return ProposedAction.wait(
                "frozen_target",
                target_position=ideal,
                comparison_delta=1,
                state_updates=updates,
            )

        if observation.actor.value >= target.value:
            return ProposedAction.wait(
                "advance_ideal",
                target_position=ideal,
                comparison_delta=1,
                state_updates={"ideal_position": next_ideal},
            )
        return ProposedAction.swap(ideal, comparison_delta=1)

    def legal_action_exists(self, observation: LocalObservation, state: Mapping[str, Any]) -> bool:
        ideal = observation.target_position
        return ideal is not None and 0 <= ideal < observation.n and observation.actor_position != ideal


class NullPolicy(LocalRulePolicy):
    """No-op policy for negative controls and future dummy Algotype tests."""

    family = "null"
    algotype = "null"
    policy_id = "null_wait"

    def propose_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        forced_direction: int | None = None,
    ) -> ProposedAction:
        return ProposedAction.wait("null_wait")


class RandomWalkPolicy(LocalRulePolicy):
    """Random adjacent-swap policy used as a minimal stochastic wrapper."""

    family = "randomized"
    algotype = "random_walk"
    policy_id = "random_walk_adjacent"

    def __init__(self, swap_probability: float = 1.0) -> None:
        super().__init__(swap_probability=float(swap_probability))
        if not 0.0 <= float(swap_probability) <= 1.0:
            raise ValueError("swap_probability must be between 0 and 1")
        self.swap_probability = float(swap_probability)

    def propose_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        forced_direction: int | None = None,
    ) -> ProposedAction:
        if forced_direction is None:
            direction = 1 if rng.random() < 0.5 else -1
        else:
            direction = 1 if forced_direction > 0 else -1
        target_position = observation.actor_position + direction
        target = observation.right if direction == 1 else observation.left
        if target is None:
            return ProposedAction.wait("target_oob", target_position=target_position)
        if rng.random() > self.swap_probability:
            return ProposedAction.wait("random_wait", target_position=target_position)
        return ProposedAction.swap(target_position)

    def legal_action_exists(self, observation: LocalObservation, state: Mapping[str, Any]) -> bool:
        return _target_allowed_by_frozen_variant(observation, observation.left) or _target_allowed_by_frozen_variant(
            observation, observation.right
        )


def policy_from_spec(spec: PolicySpec | Mapping[str, Any] | str | LocalRulePolicy) -> LocalRulePolicy:
    """Instantiate a policy from a spec, dict, string ID, or existing policy."""

    if isinstance(spec, LocalRulePolicy):
        return spec
    if isinstance(spec, str):
        key = spec
        payload: Mapping[str, Any] = {"algotype": key, "parameters": {}}
    elif isinstance(spec, PolicySpec):
        payload = spec.to_dict()
        key = payload.get("algotype", payload.get("policy_id"))
    else:
        payload = spec
        key = str(payload.get("algotype", payload.get("policy_id", "")))
    parameters = dict(payload.get("parameters", {}))
    family = str(payload.get("family", ""))

    if key in {"bubble", "classic_bubble"}:
        return BubblePolicy(**parameters)
    if key in {"insertion", "classic_insertion"}:
        return InsertionPolicy(**parameters)
    if key in {"selection", "classic_selection"}:
        return SelectionPolicy(**parameters)
    if key in {"null", "null_wait"} or family == "null":
        return NullPolicy(**parameters)
    if key in {"random_walk", "random_walk_adjacent"} or family == "randomized":
        return RandomWalkPolicy(**parameters)
    raise ValueError(f"unknown policy spec: {spec!r}")


def policy_to_json(policy: LocalRulePolicy) -> str:
    return json.dumps(policy.to_spec().to_dict(), sort_keys=True, separators=(",", ":"))


def policy_from_json(payload: str) -> LocalRulePolicy:
    return policy_from_spec(json.loads(payload))


class PolicyEventSimulator:
    """Deterministic event simulator driven by LocalRulePolicy objects."""

    def __init__(
        self,
        initial_values: Sequence[int],
        policies: str | LocalRulePolicy | PolicySpec | Sequence[str | LocalRulePolicy | PolicySpec | Mapping[str, Any]],
        *,
        labels: Sequence[int] | None = None,
        reverse_directions: Sequence[bool] | None = None,
        frozen_positions: Iterable[int] = (),
        frozen_variant: str = "none",
        scheduler_seed: int = 0,
        tie_breaker_seed: int = 0,
        condition_id: str = "manual",
        implementation: str = "policy_interface",
        research_step_id: str = "S01",
    ) -> None:
        if frozen_variant not in FROZEN_VARIANTS:
            raise ValueError(f"unknown frozen_variant: {frozen_variant}")
        self.initial_values = [int(value) for value in initial_values]
        if isinstance(policies, (str, LocalRulePolicy, PolicySpec)) or isinstance(policies, Mapping):
            policy_list = [policy_from_spec(policies)] * len(self.initial_values)
        else:
            policy_list = [policy_from_spec(policy) for policy in policies]
        if len(policy_list) != len(self.initial_values):
            raise ValueError("policies length must match initial_values")

        label_list = [0] * len(self.initial_values) if labels is None else [int(value) for value in labels]
        reverse_list = (
            [False] * len(self.initial_values)
            if reverse_directions is None
            else [bool(value) for value in reverse_directions]
        )
        if len(label_list) != len(self.initial_values) or len(reverse_list) != len(self.initial_values):
            raise ValueError("labels and reverse_directions must match initial_values length")

        frozen_set = set(int(pos) for pos in frozen_positions)
        n = len(self.initial_values)
        if min(frozen_set, default=0) < 0 or max(frozen_set, default=-1) >= n:
            raise ValueError("frozen position out of bounds")
        if frozen_variant == "none" and frozen_set:
            raise ValueError("frozen positions require frozen_variant passive or stuck")

        self.condition_id = condition_id
        self.implementation = implementation
        self.research_step_id = research_step_id
        self.frozen_variant = frozen_variant
        self.scheduler_seed = int(scheduler_seed)
        self.tie_breaker_seed = int(tie_breaker_seed)
        self.scheduler_rng = np.random.default_rng(self.scheduler_seed)
        self.tie_rng = np.random.default_rng(self.tie_breaker_seed)
        self.cells: list[PolicyCell] = []
        for pos, (value, policy, label, reverse) in enumerate(zip(self.initial_values, policy_list, label_list, reverse_list)):
            self.cells.append(
                PolicyCell(
                    cell_id=pos,
                    value=value,
                    policy=policy,
                    label=label,
                    reverse_direction=reverse,
                    frozen=pos in frozen_set,
                    state=policy.initial_state(
                        cell_id=pos,
                        position=pos,
                        value=int(value),
                        n=n,
                        reverse_direction=reverse,
                    ),
                )
            )

        self.positions_by_id = {cell.cell_id: pos for pos, cell in enumerate(self.cells)}
        self._eligible_cell_ids = [cell.cell_id for cell in self.cells if not cell.frozen]
        self.initial_algotypes = self.current_algotypes()
        self.initial_frozen_positions = self.current_frozen_positions()
        self.initial_state_hash = state_hash(self.initial_values)
        self.swap_count = 0
        self.comparison_count = 0
        self.archived_compare_and_swap_count = 0
        self.blocked_move_attempts = 0
        self.frozen_swap_attempts = 0
        self.activation_count = 0
        self.trace_rows: list[dict[str, Any]] = []
        self._append_trace_row(
            event_kind="initial",
            activation_index=0,
            actor_cell_id=None,
            actor_algotype=None,
            target_position=None,
        )

    def current_values(self) -> list[int]:
        return [cell.value for cell in self.cells]

    def current_algotypes(self) -> list[str]:
        return [cell.algotype for cell in self.cells]

    def current_policy_states(self) -> list[dict[str, Any]]:
        return [dict(cell.state) for cell in self.cells]

    def current_frozen_positions(self) -> list[int]:
        return [pos for pos, cell in enumerate(self.cells) if cell.frozen]

    def position_by_cell_id(self) -> dict[int, int]:
        return dict(self.positions_by_id)

    def is_sorted(self) -> bool:
        return sortedness_raw(self.current_values()) == len(self.cells) - 1

    def value_counts_conserved(self) -> bool:
        return Counter(self.initial_values) == Counter(self.current_values())

    def algotype_counts_conserved(self) -> bool:
        return Counter(self.initial_algotypes) == Counter(self.current_algotypes())

    def frozen_cell_ids(self) -> list[int]:
        return sorted(cell.cell_id for cell in self.cells if cell.frozen)

    def eligible_cell_ids(self) -> list[int]:
        return list(self._eligible_cell_ids)

    def _target_in_bounds(self, target_pos: int) -> bool:
        return 0 <= target_pos < len(self.cells)

    def _target_status_allows_policy_check(self, target_pos: int) -> bool:
        return self._target_in_bounds(target_pos)

    def _target_is_active(self, target_pos: int) -> bool:
        return self._target_in_bounds(target_pos) and not self.cells[target_pos].frozen

    def _count_frozen_attempt(self, actor_pos: int) -> None:
        actor = self.cells[actor_pos]
        if not actor.tried_to_swap_with_frozen:
            self.frozen_swap_attempts += 1
            actor.tried_to_swap_with_frozen = True

    def _can_swap(self, actor_pos: int, target_pos: int) -> bool:
        if not self._target_in_bounds(target_pos):
            return False
        if self.cells[actor_pos].frozen:
            self._count_frozen_attempt(actor_pos)
            return False
        if self.frozen_variant == "stuck" and self.cells[target_pos].frozen:
            self.blocked_move_attempts += 1
            return False
        return True

    def _swap(self, actor_pos: int, target_pos: int) -> None:
        actor = self.cells[actor_pos]
        target = self.cells[target_pos]
        actor.tried_to_swap_with_frozen = False
        target.tried_to_swap_with_frozen = False
        self.cells[actor_pos], self.cells[target_pos] = target, actor
        self.positions_by_id[actor.cell_id] = target_pos
        self.positions_by_id[target.cell_id] = actor_pos
        self.swap_count += 1

    def _append_trace_row(
        self,
        *,
        event_kind: str,
        activation_index: int,
        actor_cell_id: int | None,
        actor_algotype: str | None,
        target_position: int | None,
    ) -> None:
        values = self.current_values()
        raw = sortedness_raw(values)
        self.trace_rows.append(
            {
                "research_step_id": self.research_step_id,
                "condition_id": self.condition_id,
                "implementation": self.implementation,
                "algorithm": self.algorithm_label(),
                "event_index": len(self.trace_rows),
                "event_kind": event_kind,
                "activation_index": int(activation_index),
                "actor_cell_id": actor_cell_id,
                "actor_algotype": actor_algotype,
                "target_position": target_position,
                "swap_count": int(self.swap_count),
                "comparison_count": int(self.comparison_count),
                "archived_compare_and_swap_count": int(self.archived_compare_and_swap_count),
                "sortedness_raw_count": int(raw),
                "sortedness_percent": 100.0 * raw / (len(values) - 1) if len(values) > 1 else 100.0,
                "monotonicity_error": max(0, len(values) - 1) - raw,
                "state_hash": state_hash(values),
                "initial_state_hash": self.initial_state_hash,
                "frozen_positions_json": json.dumps(self.current_frozen_positions(), separators=(",", ":")),
                "algotypes_json": json.dumps(self.current_algotypes(), separators=(",", ":")),
                "scheduler_seed": self.scheduler_seed,
                "tie_breaker_seed": self.tie_breaker_seed,
            }
        )

    def algorithm_label(self) -> str:
        algotypes = sorted(set(self.current_algotypes()))
        if len(algotypes) == 1:
            return algotypes[0]
        return "+".join(algotypes)

    def _insertion_prefix_enabled_by_position(self, actor_pos: int, reverse_direction: bool) -> bool:
        previous = 100000 if reverse_direction else -1
        for pos in range(actor_pos):
            cell = self.cells[pos]
            if cell.frozen:
                previous = -1
                continue
            if reverse_direction and cell.value > previous:
                return False
            if not reverse_direction and cell.value < previous:
                return False
            previous = cell.value
        return True

    def _original_style_move_opportunity(self, actor_pos: int) -> bool:
        actor = self.cells[actor_pos]
        if actor.frozen or actor.algotype not in CLASSIC_ALGOTYPES:
            return False
        if actor.algotype == "bubble":
            if actor.reverse_direction:
                bigger_than_left = (
                    actor_pos > 0
                    and actor.value > self.cells[actor_pos - 1].value
                    and self._target_is_active(actor_pos - 1)
                )
                smaller_than_right = (
                    actor_pos < len(self.cells) - 1
                    and actor.value < self.cells[actor_pos + 1].value
                    and self._target_is_active(actor_pos + 1)
                )
                return bigger_than_left or smaller_than_right
            smaller_than_left = (
                actor_pos > 0
                and actor.value < self.cells[actor_pos - 1].value
                and self._target_is_active(actor_pos - 1)
            )
            bigger_than_right = (
                actor_pos < len(self.cells) - 1
                and actor.value > self.cells[actor_pos + 1].value
                and self._target_is_active(actor_pos + 1)
            )
            return smaller_than_left or bigger_than_right
        if actor.algotype == "insertion":
            if not self._insertion_prefix_enabled_by_position(actor_pos, actor.reverse_direction):
                return False
            if actor_pos == 0 or not self._target_is_active(actor_pos - 1):
                return False
            if actor.reverse_direction:
                return actor.value > self.cells[actor_pos - 1].value
            return actor.value < self.cells[actor_pos - 1].value
        if actor.algotype == "selection":
            ideal = actor.state.get("ideal_position")
            return ideal is not None and self._target_in_bounds(int(ideal)) and actor_pos != int(ideal)
        return False

    def step(
        self,
        *,
        forced_cell_id: int | None = None,
        forced_direction: int | None = None,
    ) -> StepOutcome:
        eligible = self.eligible_cell_ids()
        if not eligible:
            return StepOutcome(False, None, None, None, None, False, 0, 0, False, "no_eligible_cells")
        if forced_cell_id is None:
            actor_cell_id = int(self.scheduler_rng.choice(eligible))
        else:
            actor_cell_id = int(forced_cell_id)
            if actor_cell_id not in eligible:
                return StepOutcome(False, actor_cell_id, None, None, None, False, 0, 0, False, "forced_cell_not_eligible")

        actor_pos = self.positions_by_id[actor_cell_id]
        actor = self.cells[actor_pos]
        self.activation_count += 1
        archived_delta = 1 if self._original_style_move_opportunity(actor_pos) else 0
        self.archived_compare_and_swap_count += archived_delta
        observation = actor.policy.observe(self.cells, actor_pos, actor.state, self.frozen_variant)
        proposed = actor.policy.propose_action(observation, actor.state, self.tie_rng, forced_direction=forced_direction)
        proposed = actor.policy.constrain_action(observation, actor.state, proposed)
        self.comparison_count += int(proposed.comparison_delta)
        actor.state.update(proposed.state_updates)

        if proposed.action != "swap" or proposed.target_position is None:
            outcome = StepOutcome(
                True,
                actor.cell_id,
                actor_pos,
                actor_pos,
                proposed.target_position,
                False,
                int(proposed.comparison_delta),
                archived_delta,
                False,
                proposed.reason,
            )
            actor.policy.update_state(actor.state, observation, proposed, outcome)
            return outcome

        target_pos = int(proposed.target_position)
        if not self._target_status_allows_policy_check(target_pos):
            outcome = StepOutcome(
                True,
                actor.cell_id,
                actor_pos,
                actor_pos,
                target_pos,
                False,
                int(proposed.comparison_delta),
                archived_delta,
                False,
                "target_oob",
            )
            actor.policy.update_state(actor.state, observation, proposed, outcome)
            return outcome
        if not self._can_swap(actor_pos, target_pos):
            outcome = StepOutcome(
                True,
                actor.cell_id,
                actor_pos,
                actor_pos,
                target_pos,
                False,
                int(proposed.comparison_delta),
                archived_delta,
                True,
                proposed.blocked_reason,
            )
            actor.policy.update_state(actor.state, observation, proposed, outcome)
            return outcome

        self._swap(actor_pos, target_pos)
        outcome = StepOutcome(
            True,
            actor.cell_id,
            actor_pos,
            target_pos,
            target_pos,
            True,
            int(proposed.comparison_delta),
            archived_delta,
            False,
            proposed.swapped_reason,
        )
        actor.policy.update_state(actor.state, observation, proposed, outcome)
        self._append_trace_row(
            event_kind="swap",
            activation_index=self.activation_count,
            actor_cell_id=outcome.actor_cell_id,
            actor_algotype=actor.algotype,
            target_position=outcome.target_position,
        )
        return outcome

    def legal_action_exists(self) -> bool:
        for cell in self.cells:
            if cell.frozen:
                continue
            pos = self.positions_by_id[cell.cell_id]
            observation = cell.policy.observe(self.cells, pos, cell.state, self.frozen_variant)
            if cell.policy.legal_action_exists(observation, cell.state):
                return True
        return False

    def run(
        self,
        *,
        max_activations: int = 2_000_000,
        max_swaps: int = 500_000,
        max_comparisons: int = 4_000_000,
        no_move_checks_required: int = 2,
        no_move_check_interval: int | None = None,
    ) -> SimulationResult:
        import time

        started_at = time.perf_counter()
        stop_reason = "sorted"
        no_move_checks = 0
        interval = no_move_check_interval or max(1, len(self.cells))

        while True:
            if self.is_sorted():
                stop_reason = "sorted"
                break
            if self.activation_count >= max_activations:
                stop_reason = "max_activation_cap"
                break
            if self.swap_count >= max_swaps:
                stop_reason = "max_step_cap"
                break
            if self.comparison_count >= max_comparisons:
                stop_reason = "max_comparison_cap"
                break
            if self.activation_count % interval == 0:
                if not self.legal_action_exists():
                    no_move_checks += 1
                    if no_move_checks >= no_move_checks_required:
                        stop_reason = "no_cell_can_move_after_two_checks"
                        break
                else:
                    no_move_checks = 0

            self.step()

        final_values = self.current_values()
        return SimulationResult(
            condition_id=self.condition_id,
            implementation=self.implementation,
            algorithm=self.algorithm_label(),
            completed=self.is_sorted(),
            stop_reason=stop_reason,
            initial_values=list(self.initial_values),
            final_values=final_values,
            initial_algotypes=list(self.initial_algotypes),
            final_algotypes=self.current_algotypes(),
            initial_frozen_positions=list(self.initial_frozen_positions),
            final_frozen_positions=self.current_frozen_positions(),
            swap_count=self.swap_count,
            comparison_count=self.comparison_count,
            archived_compare_and_swap_count=self.archived_compare_and_swap_count,
            blocked_move_attempts=self.blocked_move_attempts,
            frozen_swap_attempts=self.frozen_swap_attempts,
            activation_count=self.activation_count,
            event_count=len(self.trace_rows),
            final_sortedness_raw_count=sortedness_raw(final_values),
            final_sortedness_percent=sortedness_percent(final_values),
            final_monotonicity_error=monotonicity_error(final_values),
            final_aggregation=aggregation(self.current_algotypes()),
            wall_time_seconds=time.perf_counter() - started_at,
            trace_rows=list(self.trace_rows),
        )
