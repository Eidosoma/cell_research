"""Local and diffusive communication fields for E04 S02."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from e02_deterministic_simulator.simulator import StepOutcome
from morphospace import (
    LocalObservation,
    LocalRulePolicy,
    PolicyCell,
    PolicySpec,
    ProposedAction,
    policy_from_spec,
)

from .memory import MemoryEventSimulator, MemoryPolicyWrapper, memory_policy_from_spec, memory_state_for_trace


SIGNAL_REPAIR_VERSION = "e04_s02_local_signals.v1"
SIGNAL_STATE_KEY = "__e04_signals__"
SIGNAL_VARIANTS = ("no_signal", "nearest_neighbor", "diffusive", "randomized", "inert")
SIGNAL_CHANNELS = ("blocked", "sorted", "frustrated", "target_seeking", "morphogen")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float):
        return round(float(value), 12)
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SignalConfig:
    """Bounded local signal-field configuration."""

    variant: str = "no_signal"
    channels: tuple[str, ...] = SIGNAL_CHANNELS
    signal_range: int = 1
    diffusion_rate: float = 0.25
    decay: float = 0.0
    noise_std: float = 0.0
    emission_scale: float = 1.0
    random_seed: int = 0
    update_order: str = "sense_act_emit_diffuse"

    def __post_init__(self) -> None:
        if self.variant not in SIGNAL_VARIANTS:
            raise ValueError(f"signal variant must be one of {SIGNAL_VARIANTS}, got {self.variant!r}")
        channels = tuple(str(channel) for channel in self.channels)
        unknown = sorted(set(channels) - set(SIGNAL_CHANNELS))
        if unknown:
            raise ValueError(f"unknown signal channels: {unknown}")
        object.__setattr__(self, "channels", channels)
        if int(self.signal_range) < 0:
            raise ValueError("signal_range must be non-negative")
        if not 0.0 <= float(self.diffusion_rate) <= 0.5:
            raise ValueError("diffusion_rate must be between 0 and 0.5")
        if not 0.0 <= float(self.decay) <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        if float(self.noise_std) < 0.0:
            raise ValueError("noise_std must be non-negative")
        if float(self.emission_scale) < 0.0:
            raise ValueError("emission_scale must be non-negative")
        if self.update_order != "sense_act_emit_diffuse":
            raise ValueError("only sense_act_emit_diffuse update order is implemented in S02")

    @classmethod
    def from_spec(cls, payload: Mapping[str, Any] | str | "SignalConfig") -> "SignalConfig":
        if isinstance(payload, SignalConfig):
            return payload
        if isinstance(payload, str):
            return cls(variant=payload)
        return cls(
            variant=str(payload.get("variant", "no_signal")),
            channels=tuple(payload.get("channels", SIGNAL_CHANNELS)),
            signal_range=int(payload.get("signalRange", payload.get("signal_range", 1))),
            diffusion_rate=float(payload.get("diffusionRate", payload.get("diffusion_rate", 0.25))),
            decay=float(payload.get("decay", 0.0)),
            noise_std=float(payload.get("noiseStd", payload.get("noise_std", 0.0))),
            emission_scale=float(payload.get("emissionScale", payload.get("emission_scale", 1.0))),
            random_seed=int(payload.get("randomSeed", payload.get("random_seed", 0))),
            update_order=str(payload.get("updateOrder", payload.get("update_order", "sense_act_emit_diffuse"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "channels": list(self.channels),
            "signalRange": int(self.signal_range),
            "diffusionRate": float(self.diffusion_rate),
            "decay": float(self.decay),
            "noiseStd": float(self.noise_std),
            "emissionScale": float(self.emission_scale),
            "randomSeed": int(self.random_seed),
            "updateOrder": self.update_order,
            "signalRepairVersion": SIGNAL_REPAIR_VERSION,
        }


def empty_signal_fields(n: int, channels: Iterable[str] = SIGNAL_CHANNELS) -> dict[str, np.ndarray]:
    return {str(channel): np.zeros(int(n), dtype=float) for channel in channels}


def serialize_signal_fields(fields: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {str(channel): [round(float(value), 12) for value in values.tolist()] for channel, values in fields.items()}


def local_window_indices(n: int, position: int, signal_range: int) -> list[int]:
    lo = max(0, int(position) - int(signal_range))
    hi = min(int(n), int(position) + int(signal_range) + 1)
    return list(range(lo, hi))


def decay_signal_fields(fields: Mapping[str, np.ndarray], config: SignalConfig | Mapping[str, Any] | str) -> dict[str, np.ndarray]:
    config = SignalConfig.from_spec(config)
    return {channel: np.maximum(0.0, np.asarray(values, dtype=float) * (1.0 - config.decay)) for channel, values in fields.items()}


def diffuse_signal_fields(
    fields: Mapping[str, np.ndarray],
    config: SignalConfig | Mapping[str, Any] | str,
) -> dict[str, np.ndarray]:
    """Apply one nearest-neighbor diffusion/decay tick with no-flux boundaries."""

    config = SignalConfig.from_spec(config)
    rate = float(config.diffusion_rate)
    out: dict[str, np.ndarray] = {}
    for channel, values in fields.items():
        current = np.asarray(values, dtype=float)
        if current.size == 0:
            out[channel] = current.copy()
            continue
        updated = current.copy()
        for index in range(current.size):
            delta = 0.0
            if index > 0:
                delta += current[index - 1] - current[index]
            if index < current.size - 1:
                delta += current[index + 1] - current[index]
            updated[index] = current[index] + rate * delta
        updated = np.maximum(0.0, updated * (1.0 - config.decay))
        out[channel] = updated
    return out


def signal_state_for_trace(policy_state: Mapping[str, Any]) -> dict[str, Any]:
    value = policy_state.get(SIGNAL_STATE_KEY, {})
    if isinstance(value, Mapping):
        return _json_ready(dict(value))
    return {}


def flatten_signal_state(signal_state: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    channels = signal_state.get("channels", {})
    if isinstance(channels, Mapping):
        for channel, metrics in channels.items():
            if not isinstance(metrics, Mapping):
                continue
            for key, value in metrics.items():
                if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
                    flattened[f"signal_{channel}_{key}"] = _json_ready(value)
    return flattened


def reset_signal_state(
    policy_state: dict[str, Any],
    config: SignalConfig | Mapping[str, Any] | str,
) -> dict[str, Any]:
    config = SignalConfig.from_spec(config)
    if config.variant == "no_signal":
        policy_state.pop(SIGNAL_STATE_KEY, None)
    else:
        policy_state[SIGNAL_STATE_KEY] = {
            "signal_schema_version": SIGNAL_REPAIR_VERSION,
            "variant": config.variant,
            "visible_positions": [],
            "channels": {},
        }
    return policy_state


def _state_for_base_policy(state: Mapping[str, Any]) -> dict[str, Any]:
    base = {str(key): value for key, value in state.items() if key != SIGNAL_STATE_KEY}
    base.update(flatten_signal_state(signal_state_for_trace(state)))
    return base


def sense_signal_state(
    fields: Mapping[str, np.ndarray],
    position: int,
    config: SignalConfig | Mapping[str, Any] | str,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Return the local signal observation visible from one position."""

    config = SignalConfig.from_spec(config)
    if config.variant == "no_signal":
        return {}
    n = len(next(iter(fields.values()))) if fields else 0
    visible_positions = local_window_indices(n, position, config.signal_range)
    channels: dict[str, dict[str, float]] = {}

    if config.variant == "randomized":
        rng = rng or np.random.default_rng(config.random_seed)
        for channel in config.channels:
            samples = rng.random(max(1, len(visible_positions)))
            center_value = float(rng.random())
            channels[channel] = {
                "center": center_value,
                "left": float(samples[0]) if int(position) > 0 else 0.0,
                "right": float(samples[-1]) if int(position) < n - 1 else 0.0,
                "local_sum": float(samples.sum()),
                "local_mean": float(samples.mean()),
                "local_max": float(samples.max()),
            }
    else:
        for channel in config.channels:
            values = np.asarray(fields.get(channel, np.zeros(n, dtype=float)), dtype=float)
            local_values = values[visible_positions] if visible_positions else np.asarray([], dtype=float)
            left_value = values[position - 1] if position > 0 and values.size else 0.0
            right_value = values[position + 1] if position < n - 1 and values.size else 0.0
            channels[channel] = {
                "center": float(values[position]) if 0 <= position < values.size else 0.0,
                "left": float(left_value),
                "right": float(right_value),
                "local_sum": float(local_values.sum()) if local_values.size else 0.0,
                "local_mean": float(local_values.mean()) if local_values.size else 0.0,
                "local_max": float(local_values.max()) if local_values.size else 0.0,
            }

    return {
        "signal_schema_version": SIGNAL_REPAIR_VERSION,
        "variant": config.variant,
        "range": int(config.signal_range),
        "actor_position": int(position),
        "visible_positions": visible_positions,
        "channels": _json_ready(channels),
    }


def _is_local_ordered(cells: Sequence[PolicyCell], position: int) -> bool:
    actor = cells[position]
    if position > 0 and cells[position - 1].value > actor.value:
        return False
    if position < len(cells) - 1 and actor.value > cells[position + 1].value:
        return False
    return True


class SignalPolicyWrapper(LocalRulePolicy):
    """LocalRulePolicy wrapper that exposes bounded local signal observations."""

    family = "signal_repair"
    version = SIGNAL_REPAIR_VERSION

    def __init__(
        self,
        base_policy: str | LocalRulePolicy | PolicySpec | Mapping[str, Any],
        signal_config: SignalConfig | Mapping[str, Any] | str = "no_signal",
    ) -> None:
        if isinstance(base_policy, Mapping) and base_policy.get("family") == "memory_repair":
            self.base_policy = memory_policy_from_spec(base_policy)
        else:
            self.base_policy = base_policy if isinstance(base_policy, LocalRulePolicy) else policy_from_spec(base_policy)
        self.signal_config = SignalConfig.from_spec(signal_config)
        self.algotype = self.base_policy.algotype
        self.policy_id = f"e04_signal_{self.signal_config.variant}__{self.base_policy.policy_id}"
        super().__init__(
            base_policy=self.base_policy.to_spec().to_dict(),
            signal_config=self.signal_config.to_dict(),
        )

    def to_spec(self) -> PolicySpec:
        return PolicySpec(
            policy_id=self.policy_id,
            family=self.family,
            algotype=self.algotype,
            version=self.version,
            parameters={
                "basePolicy": self.base_policy.to_spec().to_dict(),
                "signalConfig": self.signal_config.to_dict(),
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
        return reset_signal_state(dict(state), self.signal_config)

    def observe(
        self,
        cells: Sequence[PolicyCell],
        actor_position: int,
        state: Mapping[str, Any],
        frozen_variant: str,
    ) -> LocalObservation:
        return self.base_policy.observe(cells, actor_position, _state_for_base_policy(state), frozen_variant)

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
            _state_for_base_policy(state),
            rng,
            forced_direction=forced_direction,
        )

    def constrain_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        action: ProposedAction,
    ) -> ProposedAction:
        return self.base_policy.constrain_action(observation, _state_for_base_policy(state), action)

    def update_state(
        self,
        state: dict[str, Any],
        observation: LocalObservation,
        action: ProposedAction,
        outcome: StepOutcome,
    ) -> None:
        self.base_policy.update_state(state, observation, action, outcome)
        if self.signal_config.variant == "no_signal":
            state.pop(SIGNAL_STATE_KEY, None)

    def legal_action_exists(self, observation: LocalObservation, state: Mapping[str, Any]) -> bool:
        return self.base_policy.legal_action_exists(observation, _state_for_base_policy(state))


def signal_policy_from_spec(spec: PolicySpec | Mapping[str, Any] | str | LocalRulePolicy) -> LocalRulePolicy:
    if isinstance(spec, SignalPolicyWrapper):
        return spec
    if isinstance(spec, str):
        return SignalPolicyWrapper(policy_from_spec(spec), "no_signal")
    if isinstance(spec, PolicySpec):
        payload = spec.to_dict()
    elif isinstance(spec, LocalRulePolicy):
        return spec
    else:
        payload = dict(spec)

    if payload.get("family") != "signal_repair":
        return memory_policy_from_spec(payload) if payload.get("family") == "memory_repair" else policy_from_spec(payload)
    parameters = dict(payload.get("parameters", {}))
    base_policy = parameters.get("basePolicy") or parameters.get("base_policy")
    signal_config = parameters.get("signalConfig") or parameters.get("signal_config") or "no_signal"
    if base_policy is None:
        raise ValueError("signal policy specs require parameters.basePolicy")
    return SignalPolicyWrapper(base_policy, signal_config)


def signal_policy_to_json(policy: LocalRulePolicy) -> str:
    return _compact_json(policy.to_spec().to_dict())


def signal_policy_from_json(payload: str) -> LocalRulePolicy:
    return signal_policy_from_spec(json.loads(payload))


def build_signal_variants(
    base_policy: str | LocalRulePolicy | PolicySpec | Mapping[str, Any],
    variants: Iterable[str] = SIGNAL_VARIANTS,
    *,
    signal_range: int = 1,
    diffusion_rate: float = 0.25,
    decay: float = 0.0,
    noise_std: float = 0.0,
    random_seed: int = 0,
) -> tuple[SignalPolicyWrapper, ...]:
    return tuple(
        SignalPolicyWrapper(
            base_policy,
            SignalConfig(
                variant=variant,
                signal_range=signal_range,
                diffusion_rate=diffusion_rate,
                decay=decay,
                noise_std=noise_std,
                random_seed=random_seed,
            ),
        )
        for variant in variants
    )


class SignalEventSimulator(MemoryEventSimulator):
    """MemoryEventSimulator with local and diffusive signal fields."""

    def __init__(
        self,
        initial_values: Sequence[int],
        policies: str | LocalRulePolicy | PolicySpec | Sequence[str | LocalRulePolicy | PolicySpec | Mapping[str, Any]],
        *,
        signal_config: SignalConfig | Mapping[str, Any] | str = "no_signal",
        trace_signal_activations: bool = True,
        auto_wrap_policies: bool = True,
        implementation: str = "signal_repair_interface",
        research_step_id: str = "S02",
        **kwargs: Any,
    ) -> None:
        self.signal_config = SignalConfig.from_spec(signal_config)
        self.trace_signal_activations = bool(trace_signal_activations)
        self.signal_rng = np.random.default_rng(self.signal_config.random_seed)
        self.signal_fields = empty_signal_fields(len(initial_values), self.signal_config.channels)
        wrapped = self._wrap_policies(policies) if auto_wrap_policies else policies
        super().__init__(
            initial_values,
            wrapped,
            implementation=implementation,
            research_step_id=research_step_id,
            **kwargs,
        )

    def _wrap_one(self, policy: str | LocalRulePolicy | PolicySpec | Mapping[str, Any]) -> LocalRulePolicy:
        parsed = signal_policy_from_spec(policy)
        if isinstance(parsed, SignalPolicyWrapper):
            return parsed
        return SignalPolicyWrapper(parsed, self.signal_config)

    def _wrap_policies(
        self,
        policies: str | LocalRulePolicy | PolicySpec | Sequence[str | LocalRulePolicy | PolicySpec | Mapping[str, Any]],
    ):
        if isinstance(policies, (str, LocalRulePolicy, PolicySpec)) or isinstance(policies, Mapping):
            return self._wrap_one(policies)
        return [self._wrap_one(policy) for policy in policies]

    def _set_actor_signal_state(self, actor_pos: int) -> None:
        actor = self.cells[actor_pos]
        sensed = sense_signal_state(self.signal_fields, actor_pos, self.signal_config, rng=self.signal_rng)
        if self.signal_config.variant == "no_signal":
            actor.state.pop(SIGNAL_STATE_KEY, None)
        else:
            actor.state[SIGNAL_STATE_KEY] = sensed

    def sense_at_position(self, position: int) -> dict[str, Any]:
        return sense_signal_state(self.signal_fields, int(position), self.signal_config, rng=self.signal_rng)

    def _emission_from_outcome(
        self,
        actor_pos_after: int | None,
        observation: LocalObservation,
        action: ProposedAction,
        outcome: StepOutcome,
    ) -> dict[str, float]:
        if actor_pos_after is None or not 0 <= int(actor_pos_after) < len(self.cells):
            return {channel: 0.0 for channel in self.signal_config.channels}
        actor = self.cells[int(actor_pos_after)]
        memory = memory_state_for_trace(actor.state)
        attempted_swap = action.action == "swap" and action.target_position is not None
        local_frustration = float(memory.get("local_frustration", 0.0)) if isinstance(memory, Mapping) else 0.0
        return {
            "blocked": 1.0 if outcome.blocked_move_attempt else 0.0,
            "sorted": 1.0 if _is_local_ordered(self.cells, int(actor_pos_after)) else 0.0,
            "frustrated": max(local_frustration, 1.0 if attempted_swap and not outcome.swapped else 0.0),
            "target_seeking": 1.0 if attempted_swap and not outcome.swapped else 0.0,
            "morphogen": 1.0 if outcome.activated else 0.0,
        }

    def _apply_signal_update(
        self,
        actor_pos_after: int | None,
        observation: LocalObservation,
        action: ProposedAction,
        outcome: StepOutcome,
    ) -> None:
        if self.signal_config.variant in {"no_signal", "inert", "randomized"}:
            return
        if actor_pos_after is None or not 0 <= int(actor_pos_after) < len(self.cells):
            return
        emission = self._emission_from_outcome(actor_pos_after, observation, action, outcome)
        for channel in self.signal_config.channels:
            self.signal_fields[channel][int(actor_pos_after)] += float(emission.get(channel, 0.0)) * float(
                self.signal_config.emission_scale
            )
        if self.signal_config.noise_std > 0:
            for channel in self.signal_config.channels:
                noise = self.signal_rng.normal(0.0, self.signal_config.noise_std, size=self.signal_fields[channel].shape)
                self.signal_fields[channel] = np.maximum(0.0, self.signal_fields[channel] + noise)
        if self.signal_config.variant == "diffusive":
            self.signal_fields = diffuse_signal_fields(self.signal_fields, self.signal_config)
        elif self.signal_config.variant == "nearest_neighbor":
            self.signal_fields = decay_signal_fields(self.signal_fields, self.signal_config)

    def _signal_trace_payload(self) -> dict[str, Any]:
        return {
            "config": self.signal_config.to_dict(),
            "fields": serialize_signal_fields(self.signal_fields),
        }

    def _actor_signal_trace(self, actor_cell_id: int | None) -> dict[str, Any]:
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
            "signal_state": signal_state_for_trace(cell.state),
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
        row["signal_schema_version"] = SIGNAL_REPAIR_VERSION
        row["signal_config_json"] = _compact_json(self.signal_config.to_dict())
        row["signal_fields_json"] = _compact_json(self._signal_trace_payload()["fields"])
        row["actor_signal_state_json"] = _compact_json(self._actor_signal_trace(actor_cell_id))
        row["signal_variant_counts_json"] = _compact_json(
            Counter(
                cell.policy.signal_config.variant if isinstance(cell.policy, SignalPolicyWrapper) else "unwrapped"
                for cell in self.cells
            )
        )

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

        trace_count_before = len(self.trace_rows)
        actor_pos = self.positions_by_id[actor_cell_id]
        self._set_actor_signal_state(actor_pos)
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
            self._apply_signal_update(actor_pos, observation, proposed, outcome)
            if self.trace_signal_activations and len(self.trace_rows) == trace_count_before:
                self._append_trace_row(
                    event_kind="signal_update",
                    activation_index=self.activation_count,
                    actor_cell_id=outcome.actor_cell_id,
                    actor_algotype=actor.algotype,
                    target_position=outcome.target_position,
                )
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
            self._apply_signal_update(actor_pos, observation, proposed, outcome)
            if self.trace_signal_activations and len(self.trace_rows) == trace_count_before:
                self._append_trace_row(
                    event_kind="signal_update",
                    activation_index=self.activation_count,
                    actor_cell_id=outcome.actor_cell_id,
                    actor_algotype=actor.algotype,
                    target_position=outcome.target_position,
                )
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
            self._apply_signal_update(actor_pos, observation, proposed, outcome)
            if self.trace_signal_activations and len(self.trace_rows) == trace_count_before:
                self._append_trace_row(
                    event_kind="signal_update",
                    activation_index=self.activation_count,
                    actor_cell_id=outcome.actor_cell_id,
                    actor_algotype=actor.algotype,
                    target_position=outcome.target_position,
                )
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
        self._apply_signal_update(target_pos, observation, proposed, outcome)
        self._append_trace_row(
            event_kind="swap",
            activation_index=self.activation_count,
            actor_cell_id=outcome.actor_cell_id,
            actor_algotype=actor.algotype,
            target_position=outcome.target_position,
        )
        return outcome
