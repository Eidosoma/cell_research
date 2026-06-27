"""Fatigue and damage dynamics for E04 S04."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from e02_deterministic_simulator.simulator import StepOutcome
from morphospace import LocalRulePolicy, PolicySpec

from .repair import (
    REPAIR_REPAIR_VERSION,
    RepairRuleConfig,
    RepairableFrozenEventSimulator,
)
from .signals import SIGNAL_REPAIR_VERSION, SignalConfig


FATIGUE_DAMAGE_VERSION = "e04_s04_fatigue_damage.v1"
FATIGUE_DAMAGE_VARIANTS = ("none", "movement", "failed_swap", "frustration", "directional", "stochastic_damage", "combined")


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
class FatigueDamageConfig:
    """Local dynamic unreliability configuration."""

    variant: str = "none"
    movement_threshold: int = 3
    failed_swap_threshold: int = 2
    frustration_threshold: int = 2
    recovery_activations: int = 2
    damage_probability: float = 0.0
    damage_cooldown_activations: int = 2
    impaired_direction: str = "either"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.variant not in FATIGUE_DAMAGE_VARIANTS:
            raise ValueError(f"fatigue/damage variant must be one of {FATIGUE_DAMAGE_VARIANTS}, got {self.variant!r}")
        if int(self.movement_threshold) < 1:
            raise ValueError("movement_threshold must be positive")
        if int(self.failed_swap_threshold) < 1:
            raise ValueError("failed_swap_threshold must be positive")
        if int(self.frustration_threshold) < 1:
            raise ValueError("frustration_threshold must be positive")
        if int(self.recovery_activations) < 1:
            raise ValueError("recovery_activations must be positive")
        if int(self.damage_cooldown_activations) < 1:
            raise ValueError("damage_cooldown_activations must be positive")
        if not 0.0 <= float(self.damage_probability) <= 1.0:
            raise ValueError("damage_probability must be between 0 and 1")
        if self.impaired_direction not in {"left", "right", "either", "attempted"}:
            raise ValueError("impaired_direction must be left, right, either, or attempted")

    @classmethod
    def from_spec(cls, payload: Mapping[str, Any] | str | "FatigueDamageConfig") -> "FatigueDamageConfig":
        if isinstance(payload, FatigueDamageConfig):
            return payload
        if isinstance(payload, str):
            return cls(variant=payload)
        return cls(
            variant=str(payload.get("variant", "none")),
            movement_threshold=int(payload.get("movementThreshold", payload.get("movement_threshold", 3))),
            failed_swap_threshold=int(payload.get("failedSwapThreshold", payload.get("failed_swap_threshold", 2))),
            frustration_threshold=int(payload.get("frustrationThreshold", payload.get("frustration_threshold", 2))),
            recovery_activations=int(payload.get("recoveryActivations", payload.get("recovery_activations", 2))),
            damage_probability=float(payload.get("damageProbability", payload.get("damage_probability", 0.0))),
            damage_cooldown_activations=int(
                payload.get("damageCooldownActivations", payload.get("damage_cooldown_activations", 2))
            ),
            impaired_direction=str(payload.get("impairedDirection", payload.get("impaired_direction", "either"))),
            random_seed=int(payload.get("randomSeed", payload.get("random_seed", 0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "movementThreshold": int(self.movement_threshold),
            "failedSwapThreshold": int(self.failed_swap_threshold),
            "frustrationThreshold": int(self.frustration_threshold),
            "recoveryActivations": int(self.recovery_activations),
            "damageProbability": float(self.damage_probability),
            "damageCooldownActivations": int(self.damage_cooldown_activations),
            "impairedDirection": self.impaired_direction,
            "randomSeed": int(self.random_seed),
            "fatigueDamageVersion": FATIGUE_DAMAGE_VERSION,
        }


def initial_fatigue_state(cell_id: int, position: int) -> dict[str, Any]:
    return {
        "cell_id": int(cell_id),
        "initial_position": int(position),
        "current_position": int(position),
        "movement_count": 0,
        "failed_swap_count": 0,
        "frustration_count": 0,
        "fatigue_cooldown_remaining": 0,
        "damage_cooldown_remaining": 0,
        "fatigued": False,
        "damaged": False,
        "impaired_direction": "none",
        "last_attempt_direction": None,
        "fatigue_event_count": 0,
        "damage_event_count": 0,
        "recovery_event_count": 0,
        "last_transition": None,
        "last_transition_activation": None,
        "last_actor_position_before": None,
        "last_actor_position_after": None,
    }


def serialize_fatigue_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return _json_ready(dict(state))


class FatigueDamageEventSimulator(RepairableFrozenEventSimulator):
    """Repairable Frozen Cell simulator with local fatigue and damage state."""

    def __init__(
        self,
        initial_values: Sequence[int],
        policies: str | LocalRulePolicy | PolicySpec | Sequence[str | LocalRulePolicy | PolicySpec | Mapping[str, Any]],
        *,
        fatigue_config: FatigueDamageConfig | Mapping[str, Any] | str = "none",
        repair_config: RepairRuleConfig | Mapping[str, Any] | str = "permanent",
        signal_config: SignalConfig | Mapping[str, Any] | str = "no_signal",
        implementation: str = "fatigue_damage_interface",
        research_step_id: str = "S04",
        **kwargs: Any,
    ) -> None:
        self.fatigue_config = FatigueDamageConfig.from_spec(fatigue_config)
        self.fatigue_states: dict[int, dict[str, Any]] = {}
        self.fatigue_events: list[dict[str, Any]] = []
        self._newly_impaired_cell_ids: set[int] = set()
        self._pending_impairment_block: dict[str, Any] | None = None
        self.damage_rng = np.random.default_rng(self.fatigue_config.random_seed)
        super().__init__(
            initial_values,
            policies,
            repair_config=repair_config,
            signal_config=signal_config,
            implementation=implementation,
            research_step_id=research_step_id,
            **kwargs,
        )
        self._initialize_fatigue_states()
        if self.trace_rows and self.fatigue_config.variant != "none":
            self._decorate_fatigue_trace_row(self.trace_rows[-1])

    def _initialize_fatigue_states(self) -> None:
        self.fatigue_states = {
            cell.cell_id: initial_fatigue_state(cell.cell_id, position) for position, cell in enumerate(self.cells)
        }

    def _fatigue_state_for_cell(self, cell_id: int) -> dict[str, Any]:
        position = self.positions_by_id.get(int(cell_id), -1)
        return self.fatigue_states.setdefault(int(cell_id), initial_fatigue_state(int(cell_id), int(position)))

    def _fatigue_state_payload(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cell_id, state in sorted(self.fatigue_states.items()):
            row = dict(state)
            row["current_position"] = int(self.positions_by_id.get(int(cell_id), row.get("current_position", -1)))
            row["fatigued"] = int(row.get("fatigue_cooldown_remaining", 0)) > 0
            row["damaged"] = int(row.get("damage_cooldown_remaining", 0)) > 0
            rows.append(serialize_fatigue_state(row))
        return rows

    def _decorate_fatigue_trace_row(self, row: dict[str, Any]) -> None:
        row["fatigue_schema_version"] = FATIGUE_DAMAGE_VERSION
        row["fatigue_config_json"] = _compact_json(self.fatigue_config.to_dict())
        row["fatigue_states_json"] = _compact_json(self._fatigue_state_payload())
        row["fatigue_events_json"] = _compact_json(self.fatigue_events)

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
        if self.fatigue_config.variant != "none" and hasattr(self, "fatigue_states"):
            self._decorate_fatigue_trace_row(self.trace_rows[-1])

    def _direction_from_positions(self, actor_pos: int | None, target_pos: int | None) -> str | None:
        if actor_pos is None or target_pos is None:
            return None
        if int(target_pos) > int(actor_pos):
            return "right"
        if int(target_pos) < int(actor_pos):
            return "left"
        return "same"

    def _configured_impairment_direction(self, attempted_direction: str | None) -> str:
        if self.fatigue_config.impaired_direction == "attempted":
            return attempted_direction or "either"
        return self.fatigue_config.impaired_direction

    def _event_position(self, cell_id: int) -> int:
        return int(self.positions_by_id.get(int(cell_id), -1))

    def _record_event(self, cell_id: int, event_type: str, reason: str, *, direction: str | None = None) -> None:
        state = self._fatigue_state_for_cell(cell_id)
        state["last_transition"] = event_type
        state["last_transition_activation"] = int(self.activation_count)
        if direction is not None:
            state["last_attempt_direction"] = direction
        if event_type.startswith("fatigue"):
            state["fatigue_event_count"] = int(state.get("fatigue_event_count", 0)) + 1
        elif event_type.startswith("damage"):
            state["damage_event_count"] = int(state.get("damage_event_count", 0)) + 1
        elif event_type.endswith("recovered"):
            state["recovery_event_count"] = int(state.get("recovery_event_count", 0)) + 1
        self.fatigue_events.append(
            _json_ready(
                {
                    "activation_index": int(self.activation_count),
                    "cell_id": int(cell_id),
                    "position": self._event_position(cell_id),
                    "event_type": event_type,
                    "reason": reason,
                    "variant": self.fatigue_config.variant,
                    "impaired_direction": state.get("impaired_direction", "none"),
                    "fatigue_cooldown_remaining": int(state.get("fatigue_cooldown_remaining", 0)),
                    "damage_cooldown_remaining": int(state.get("damage_cooldown_remaining", 0)),
                    "attempt_direction": direction,
                }
            )
        )

    def _induce_fatigue(self, cell_id: int, reason: str, *, attempted_direction: str | None = None) -> None:
        state = self._fatigue_state_for_cell(cell_id)
        if int(state.get("fatigue_cooldown_remaining", 0)) > 0:
            return
        state["fatigue_cooldown_remaining"] = int(self.fatigue_config.recovery_activations)
        state["fatigued"] = True
        state["impaired_direction"] = self._configured_impairment_direction(attempted_direction)
        self._newly_impaired_cell_ids.add(int(cell_id))
        self._record_event(cell_id, "fatigue_induced", reason, direction=attempted_direction)

    def _induce_damage(self, cell_id: int, reason: str, *, attempted_direction: str | None = None) -> None:
        state = self._fatigue_state_for_cell(cell_id)
        if int(state.get("damage_cooldown_remaining", 0)) > 0:
            return
        state["damage_cooldown_remaining"] = int(self.fatigue_config.damage_cooldown_activations)
        state["damaged"] = True
        self._newly_impaired_cell_ids.add(int(cell_id))
        self._record_event(cell_id, "damage_induced", reason, direction=attempted_direction)

    def _impairment_reason_for_swap(self, actor_pos: int, target_pos: int) -> str | None:
        if self.fatigue_config.variant == "none":
            return None
        actor = self.cells[actor_pos]
        state = self._fatigue_state_for_cell(actor.cell_id)
        direction = self._direction_from_positions(actor_pos, target_pos)
        if int(state.get("damage_cooldown_remaining", 0)) > 0:
            return "damage_impaired"
        if int(state.get("fatigue_cooldown_remaining", 0)) <= 0:
            return None
        impaired_direction = str(state.get("impaired_direction", "either"))
        if impaired_direction == "either" or direction == impaired_direction:
            return "fatigue_impaired"
        return None

    def _can_swap(self, actor_pos: int, target_pos: int) -> bool:
        reason = self._impairment_reason_for_swap(actor_pos, target_pos)
        if reason is not None:
            actor = self.cells[actor_pos]
            direction = self._direction_from_positions(actor_pos, target_pos)
            self.blocked_move_attempts += 1
            self._pending_impairment_block = {
                "actor_cell_id": int(actor.cell_id),
                "target_position": int(target_pos),
                "reason": reason,
                "direction": direction,
            }
            return False
        return super()._can_swap(actor_pos, target_pos)

    def _is_frustrating_wait(self, outcome: StepOutcome) -> bool:
        return bool(outcome.activated and not outcome.swapped and outcome.reason in {"target_oob", "no_policy_swap"})

    def _maybe_apply_damage_draw(self, cell_id: int, direction: str | None) -> None:
        if self.fatigue_config.variant not in {"stochastic_damage", "combined"}:
            return
        if self.fatigue_config.damage_probability <= 0.0:
            return
        state = self._fatigue_state_for_cell(cell_id)
        if int(state.get("damage_cooldown_remaining", 0)) > 0:
            return
        if float(self.damage_rng.random()) < float(self.fatigue_config.damage_probability):
            self._induce_damage(cell_id, "damage_probability_met", attempted_direction=direction)

    def _record_outcome_for_fatigue(self, outcome: StepOutcome) -> None:
        if self.fatigue_config.variant == "none" or not outcome.activated or outcome.actor_cell_id is None:
            return
        cell_id = int(outcome.actor_cell_id)
        state = self._fatigue_state_for_cell(cell_id)
        direction = self._direction_from_positions(outcome.actor_position_before, outcome.target_position)
        state["current_position"] = self._event_position(cell_id)
        state["last_actor_position_before"] = outcome.actor_position_before
        state["last_actor_position_after"] = outcome.actor_position_after
        if direction is not None:
            state["last_attempt_direction"] = direction

        blocked_by_impairment = (
            self._pending_impairment_block is not None
            and self._pending_impairment_block.get("actor_cell_id") == cell_id
            and self._pending_impairment_block.get("target_position") == outcome.target_position
        )
        if blocked_by_impairment:
            outcome.reason = str(self._pending_impairment_block.get("reason", outcome.reason))
            self._record_event(cell_id, "impairment_blocked_swap", outcome.reason, direction=direction)
            return

        if outcome.swapped:
            state["movement_count"] = int(state.get("movement_count", 0)) + 1
            state["failed_swap_count"] = 0
            state["frustration_count"] = 0
            if self.fatigue_config.variant in {"movement", "combined"} and state["movement_count"] >= self.fatigue_config.movement_threshold:
                state["movement_count"] = 0
                self._induce_fatigue(cell_id, "movement_threshold_met", attempted_direction=direction)
        elif outcome.blocked_move_attempt:
            state["failed_swap_count"] = int(state.get("failed_swap_count", 0)) + 1
            state["frustration_count"] = int(state.get("frustration_count", 0)) + 1
            if self.fatigue_config.variant == "directional" and state["failed_swap_count"] >= self.fatigue_config.failed_swap_threshold:
                state["failed_swap_count"] = 0
                self._induce_fatigue(cell_id, "directional_failed_swap_threshold_met", attempted_direction=direction)
            elif self.fatigue_config.variant in {"failed_swap", "combined"} and state["failed_swap_count"] >= self.fatigue_config.failed_swap_threshold:
                state["failed_swap_count"] = 0
                self._induce_fatigue(cell_id, "failed_swap_threshold_met", attempted_direction=direction)
            elif self.fatigue_config.variant in {"frustration", "combined"} and state["frustration_count"] >= self.fatigue_config.frustration_threshold:
                state["frustration_count"] = 0
                self._induce_fatigue(cell_id, "frustration_threshold_met", attempted_direction=direction)
        elif self._is_frustrating_wait(outcome):
            state["frustration_count"] = int(state.get("frustration_count", 0)) + 1
            if self.fatigue_config.variant in {"frustration", "combined"} and state["frustration_count"] >= self.fatigue_config.frustration_threshold:
                state["frustration_count"] = 0
                self._induce_fatigue(cell_id, "frustration_threshold_met", attempted_direction=direction)

        self._maybe_apply_damage_draw(cell_id, direction)

    def _tick_recovery(self) -> None:
        if self.fatigue_config.variant == "none":
            return
        for cell_id, state in sorted(self.fatigue_states.items()):
            state["current_position"] = int(self.positions_by_id.get(int(cell_id), state.get("current_position", -1)))
            if int(cell_id) in self._newly_impaired_cell_ids:
                continue
            fatigue_cooldown = int(state.get("fatigue_cooldown_remaining", 0))
            if fatigue_cooldown > 0:
                state["fatigue_cooldown_remaining"] = fatigue_cooldown - 1
                if state["fatigue_cooldown_remaining"] == 0:
                    state["fatigued"] = False
                    state["impaired_direction"] = "none"
                    self._record_event(int(cell_id), "fatigue_recovered", "recovery_period_elapsed")
            damage_cooldown = int(state.get("damage_cooldown_remaining", 0))
            if damage_cooldown > 0:
                state["damage_cooldown_remaining"] = damage_cooldown - 1
                if state["damage_cooldown_remaining"] == 0:
                    state["damaged"] = False
                    self._record_event(int(cell_id), "damage_recovered", "damage_recovery_period_elapsed")
        self._newly_impaired_cell_ids.clear()

    def step(
        self,
        *,
        forced_cell_id: int | None = None,
        forced_direction: int | None = None,
    ) -> StepOutcome:
        if self.fatigue_config.variant == "none":
            return super().step(forced_cell_id=forced_cell_id, forced_direction=forced_direction)
        self._pending_impairment_block = None
        outcome = super().step(forced_cell_id=forced_cell_id, forced_direction=forced_direction)
        self._record_outcome_for_fatigue(outcome)
        self._tick_recovery()
        if self.trace_rows:
            self._decorate_fatigue_trace_row(self.trace_rows[-1])
        self._pending_impairment_block = None
        return outcome


def fatigue_summary_record(simulator: FatigueDamageEventSimulator) -> dict[str, Any]:
    fatigue_states = simulator._fatigue_state_payload()
    return {
        "fatigueVariant": simulator.fatigue_config.variant,
        "fatigueEventCount": len(simulator.fatigue_events),
        "currentlyFatiguedCellCount": sum(1 for state in fatigue_states if int(state.get("fatigue_cooldown_remaining", 0)) > 0),
        "currentlyDamagedCellCount": sum(1 for state in fatigue_states if int(state.get("damage_cooldown_remaining", 0)) > 0),
        "fatigueEventsJson": _compact_json(simulator.fatigue_events),
        "fatigueStatesJson": _compact_json(fatigue_states),
        "fatigueDamageVersion": FATIGUE_DAMAGE_VERSION,
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
    }
