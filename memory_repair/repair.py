"""Repairable Frozen Cell dynamics for E04 S03."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from e02_deterministic_simulator.simulator import StepOutcome
from morphospace import LocalRulePolicy, PolicySpec

from .signals import (
    SIGNAL_REPAIR_VERSION,
    SignalConfig,
    SignalEventSimulator,
    SignalPolicyWrapper,
    sense_signal_state,
)


REPAIR_REPAIR_VERSION = "e04_s03_repairable_frozen.v1"
REPAIR_RULE_VARIANTS = ("permanent", "nudge_count", "signal_threshold", "elapsed_time", "direction_contact")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float):
        return round(float(value), 12)
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RepairRuleConfig:
    """Local-only recovery rule for initially frozen cells."""

    variant: str = "permanent"
    nudge_threshold: int = 2
    signal_channel: str = "blocked"
    signal_threshold: float = 1.0
    elapsed_activations: int = 3
    approach_direction: str = "from_left"

    def __post_init__(self) -> None:
        if self.variant not in REPAIR_RULE_VARIANTS:
            raise ValueError(f"repair variant must be one of {REPAIR_RULE_VARIANTS}, got {self.variant!r}")
        if int(self.nudge_threshold) < 1:
            raise ValueError("nudge_threshold must be positive")
        if float(self.signal_threshold) <= 0:
            raise ValueError("signal_threshold must be positive")
        if int(self.elapsed_activations) < 1:
            raise ValueError("elapsed_activations must be positive")
        if self.approach_direction not in {"from_left", "from_right", "either"}:
            raise ValueError("approach_direction must be from_left, from_right, or either")

    @classmethod
    def from_spec(cls, payload: Mapping[str, Any] | str | "RepairRuleConfig") -> "RepairRuleConfig":
        if isinstance(payload, RepairRuleConfig):
            return payload
        if isinstance(payload, str):
            return cls(variant=payload)
        return cls(
            variant=str(payload.get("variant", "permanent")),
            nudge_threshold=int(payload.get("nudgeThreshold", payload.get("nudge_threshold", 2))),
            signal_channel=str(payload.get("signalChannel", payload.get("signal_channel", "blocked"))),
            signal_threshold=float(payload.get("signalThreshold", payload.get("signal_threshold", 1.0))),
            elapsed_activations=int(payload.get("elapsedActivations", payload.get("elapsed_activations", 3))),
            approach_direction=str(payload.get("approachDirection", payload.get("approach_direction", "from_left"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "nudgeThreshold": int(self.nudge_threshold),
            "signalChannel": self.signal_channel,
            "signalThreshold": float(self.signal_threshold),
            "elapsedActivations": int(self.elapsed_activations),
            "approachDirection": self.approach_direction,
            "repairRepairVersion": REPAIR_REPAIR_VERSION,
        }


def initial_repair_state(cell_id: int, position: int) -> dict[str, Any]:
    return {
        "cell_id": int(cell_id),
        "initial_position": int(position),
        "current_position": int(position),
        "frozen": True,
        "recovered": False,
        "recovery_reason": None,
        "recovered_at_activation": None,
        "nudge_count": 0,
        "contacts_from_left": 0,
        "contacts_from_right": 0,
        "signal_exposure": 0.0,
        "last_signal_local_sum": 0.0,
        "frozen_age_activations": 0,
        "last_actor_cell_id": None,
        "last_contact_direction": None,
    }


def serialize_repair_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return _json_ready(dict(state))


class RepairableFrozenEventSimulator(SignalEventSimulator):
    """SignalEventSimulator with locally repairable Frozen Cells."""

    def __init__(
        self,
        initial_values: Sequence[int],
        policies: str | LocalRulePolicy | PolicySpec | Sequence[str | LocalRulePolicy | PolicySpec | Mapping[str, Any]],
        *,
        repair_config: RepairRuleConfig | Mapping[str, Any] | str = "permanent",
        signal_config: SignalConfig | Mapping[str, Any] | str = "no_signal",
        implementation: str = "repairable_frozen_interface",
        research_step_id: str = "S03",
        **kwargs: Any,
    ) -> None:
        self.repair_config = RepairRuleConfig.from_spec(repair_config)
        self.repair_states: dict[int, dict[str, Any]] = {}
        self.recovery_events: list[dict[str, Any]] = []
        super().__init__(
            initial_values,
            policies,
            signal_config=signal_config,
            implementation=implementation,
            research_step_id=research_step_id,
            **kwargs,
        )
        self._initialize_repair_states()
        if self.trace_rows:
            self._decorate_repair_trace_row(self.trace_rows[-1])

    def _initialize_repair_states(self) -> None:
        self.repair_states = {}
        for position, cell in enumerate(self.cells):
            if cell.frozen:
                self.repair_states[cell.cell_id] = initial_repair_state(cell.cell_id, position)

    def _repair_state_payload(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cell_id, state in sorted(self.repair_states.items()):
            row = dict(state)
            row["current_position"] = int(self.positions_by_id.get(int(cell_id), row.get("current_position", -1)))
            if int(cell_id) in self.positions_by_id:
                row["frozen"] = bool(self.cells[self.positions_by_id[int(cell_id)]].frozen)
            rows.append(serialize_repair_state(row))
        return rows

    def _decorate_repair_trace_row(self, row: dict[str, Any]) -> None:
        row["repair_schema_version"] = REPAIR_REPAIR_VERSION
        row["repair_config_json"] = _compact_json(self.repair_config.to_dict())
        row["repair_states_json"] = _compact_json(self._repair_state_payload())
        row["recovery_events_json"] = _compact_json(self.recovery_events)

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
        self._decorate_repair_trace_row(self.trace_rows[-1])

    def _unfreeze_cell(self, cell_id: int, reason: str) -> bool:
        if int(cell_id) not in self.positions_by_id:
            return False
        position = self.positions_by_id[int(cell_id)]
        cell = self.cells[position]
        if not cell.frozen:
            return False
        cell.frozen = False
        if cell.cell_id not in self._eligible_cell_ids:
            self._eligible_cell_ids.append(cell.cell_id)
            self._eligible_cell_ids = sorted(set(self._eligible_cell_ids))
        state = self.repair_states.setdefault(cell.cell_id, initial_repair_state(cell.cell_id, position))
        state["frozen"] = False
        state["recovered"] = True
        state["recovery_reason"] = reason
        state["recovered_at_activation"] = int(self.activation_count)
        state["current_position"] = int(position)
        event = {
            "activation_index": int(self.activation_count),
            "cell_id": int(cell.cell_id),
            "position": int(position),
            "repair_variant": self.repair_config.variant,
            "recovery_reason": reason,
            "nudge_count": int(state.get("nudge_count", 0)),
            "signal_exposure": float(state.get("signal_exposure", 0.0)),
            "frozen_age_activations": int(state.get("frozen_age_activations", 0)),
        }
        self.recovery_events.append(_json_ready(event))
        return True

    def _contact_direction(self, actor_pos: int, target_pos: int) -> str:
        if int(actor_pos) < int(target_pos):
            return "from_left"
        if int(actor_pos) > int(target_pos):
            return "from_right"
        return "same_position"

    def _direction_matches(self, direction: str) -> bool:
        return self.repair_config.approach_direction == "either" or direction == self.repair_config.approach_direction

    def _matching_direction_contact_count(self, state: Mapping[str, Any]) -> int:
        if self.repair_config.approach_direction == "from_left":
            return int(state.get("contacts_from_left", 0))
        if self.repair_config.approach_direction == "from_right":
            return int(state.get("contacts_from_right", 0))
        return int(state.get("nudge_count", 0))

    def _record_local_contact(self, actor_pos: int, target_pos: int) -> None:
        if not self._target_in_bounds(target_pos):
            return
        target = self.cells[target_pos]
        if not target.frozen:
            return
        state = self.repair_states.setdefault(target.cell_id, initial_repair_state(target.cell_id, target_pos))
        direction = self._contact_direction(actor_pos, target_pos)
        state["nudge_count"] = int(state.get("nudge_count", 0)) + 1
        if direction == "from_left":
            state["contacts_from_left"] = int(state.get("contacts_from_left", 0)) + 1
        elif direction == "from_right":
            state["contacts_from_right"] = int(state.get("contacts_from_right", 0)) + 1
        state["last_actor_cell_id"] = int(self.cells[actor_pos].cell_id)
        state["last_contact_direction"] = direction
        state["current_position"] = int(target_pos)

        if self.repair_config.variant == "nudge_count" and state["nudge_count"] >= self.repair_config.nudge_threshold:
            self._unfreeze_cell(target.cell_id, "nudge_threshold_met")
        elif (
            self.repair_config.variant == "direction_contact"
            and self._direction_matches(direction)
            and self._matching_direction_contact_count(state) >= self.repair_config.nudge_threshold
        ):
            self._unfreeze_cell(target.cell_id, f"direction_contact_{direction}")

    def _can_swap(self, actor_pos: int, target_pos: int) -> bool:
        allowed = super()._can_swap(actor_pos, target_pos)
        if not allowed and self._target_in_bounds(target_pos) and self.cells[target_pos].frozen:
            self._record_local_contact(actor_pos, target_pos)
        return allowed

    def _advance_frozen_ages(self) -> None:
        for cell_id, state in self.repair_states.items():
            if int(cell_id) not in self.positions_by_id:
                continue
            cell = self.cells[self.positions_by_id[int(cell_id)]]
            if not cell.frozen:
                continue
            state["frozen_age_activations"] = int(state.get("frozen_age_activations", 0)) + 1
            state["current_position"] = int(self.positions_by_id[int(cell_id)])
            if (
                self.repair_config.variant == "elapsed_time"
                and state["frozen_age_activations"] >= self.repair_config.elapsed_activations
            ):
                self._unfreeze_cell(int(cell_id), "elapsed_time_threshold_met")

    def _check_signal_thresholds(self) -> None:
        if self.repair_config.variant != "signal_threshold":
            return
        for cell_id, state in list(self.repair_states.items()):
            if int(cell_id) not in self.positions_by_id:
                continue
            position = self.positions_by_id[int(cell_id)]
            cell = self.cells[position]
            if not cell.frozen:
                continue
            sensed = sense_signal_state(self.signal_fields, position, self.signal_config, rng=self.signal_rng)
            channel = sensed.get("channels", {}).get(self.repair_config.signal_channel, {})
            local_sum = float(channel.get("local_sum", 0.0)) if isinstance(channel, Mapping) else 0.0
            state["last_signal_local_sum"] = local_sum
            state["signal_exposure"] = float(state.get("signal_exposure", 0.0)) + local_sum
            state["current_position"] = int(position)
            if state["signal_exposure"] >= self.repair_config.signal_threshold:
                self._unfreeze_cell(int(cell_id), "signal_threshold_met")

    def _apply_signal_update(self, actor_pos_after, observation, action, outcome) -> None:
        super()._apply_signal_update(actor_pos_after, observation, action, outcome)
        self._check_signal_thresholds()

    def step(
        self,
        *,
        forced_cell_id: int | None = None,
        forced_direction: int | None = None,
    ) -> StepOutcome:
        self._advance_frozen_ages()
        return super().step(forced_cell_id=forced_cell_id, forced_direction=forced_direction)


def repair_summary_record(simulator: RepairableFrozenEventSimulator) -> dict[str, Any]:
    return {
        "repairVariant": simulator.repair_config.variant,
        "initialFrozenCellCount": len(simulator.repair_states),
        "recoveredCellCount": len(simulator.recovery_events),
        "remainingFrozenCellCount": len(simulator.current_frozen_positions()),
        "recoveryEventsJson": _compact_json(simulator.recovery_events),
        "repairStatesJson": _compact_json(simulator._repair_state_payload()),
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
    }
