"""Bounded per-cell memory wrappers for E04 S01.

This module extends the E03 policy interface without changing the public
cell-view sorting classes.  Memory is stored in an external bank keyed by the
stable public ``threadID`` identity, so it follows a cell across swaps while
remaining ablatable.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.e03.policy_interface import (
    ORIGINAL_BEHAVIORS,
    OriginalCellPolicyWrapper,
    PolicyAction,
    PolicyObservation,
    PolicyState,
    PolicyStepResult,
    cells_signature,
)


@dataclass(frozen=True)
class FailedSwapMemory:
    """One bounded record of a recent failed local swap attempt."""

    event_step: int | None
    actor_thread_id: int
    actor_position: int
    actor_value: int
    target_index: int | None
    target_thread_id: int | None
    target_value: int | None
    target_label: str | None
    target_status: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_step": self.event_step,
            "actor_thread_id": self.actor_thread_id,
            "actor_position": self.actor_position,
            "actor_value": self.actor_value,
            "target_index": self.target_index,
            "target_thread_id": self.target_thread_id,
            "target_value": self.target_value,
            "target_label": self.target_label,
            "target_status": self.target_status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class NeighborIdentityMemory:
    """One bounded record of a recently sensed immediate neighbor."""

    event_step: int | None
    side: str
    position: int
    thread_id: int | None
    value: int
    label: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_step": self.event_step,
            "side": self.side,
            "position": self.position,
            "thread_id": self.thread_id,
            "value": self.value,
            "label": self.label,
            "status": self.status,
        }


@dataclass(frozen=True)
class CellMemoryConfig:
    """Configuration for finite local cell memory."""

    enabled: bool = True
    failed_swap_capacity: int = 4
    neighbor_capacity: int = 4
    max_time_since_movement: int = 1_000_000
    max_frustration: int = 255
    frustration_increment: int = 1
    frustration_recovery: int = 1
    record_neighbor_identities: bool = True

    def __post_init__(self) -> None:
        for name in (
            "failed_swap_capacity",
            "neighbor_capacity",
            "max_time_since_movement",
            "max_frustration",
            "frustration_increment",
            "frustration_recovery",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "failed_swap_capacity": int(self.failed_swap_capacity),
            "neighbor_capacity": int(self.neighbor_capacity),
            "max_time_since_movement": int(self.max_time_since_movement),
            "max_frustration": int(self.max_frustration),
            "frustration_increment": int(self.frustration_increment),
            "frustration_recovery": int(self.frustration_recovery),
            "record_neighbor_identities": bool(self.record_neighbor_identities),
        }


@dataclass(frozen=True)
class CellMemoryState:
    """Finite memory state stored for one cell identity."""

    last_move_success: bool | None = None
    last_action_type: str | None = None
    last_target_index: int | None = None
    time_since_movement: int = 0
    local_frustration: int = 0
    recent_failed_swaps: tuple[FailedSwapMemory, ...] = ()
    recent_neighbor_identities: tuple[NeighborIdentityMemory, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_move_success": self.last_move_success,
            "last_action_type": self.last_action_type,
            "last_target_index": self.last_target_index,
            "time_since_movement": int(self.time_since_movement),
            "local_frustration": int(self.local_frustration),
            "recent_failed_swaps": [entry.to_dict() for entry in self.recent_failed_swaps],
            "recent_neighbor_identities": [entry.to_dict() for entry in self.recent_neighbor_identities],
        }

    def compact_dict(self) -> dict[str, Any]:
        return {
            "last_move_success": self.last_move_success,
            "last_action_type": self.last_action_type,
            "last_target_index": self.last_target_index,
            "time_since_movement": int(self.time_since_movement),
            "local_frustration": int(self.local_frustration),
            "failed_swap_count": len(self.recent_failed_swaps),
            "neighbor_identity_count": len(self.recent_neighbor_identities),
        }


@dataclass(frozen=True)
class MemoryPolicyObservation:
    """Policy-visible observation bundled with local memory only."""

    local: PolicyObservation
    memory: CellMemoryState
    memory_config: CellMemoryConfig

    @property
    def state(self) -> PolicyState:
        return PolicyState(
            ideal_position=self.local.ideal_position,
            reverse_direction=self.local.reverse_direction,
            memory={"e04_cell_memory": self.memory.to_dict()},
        )


@dataclass(frozen=True)
class MemoryStepResult:
    """Result of executing one policy step and updating local memory."""

    base_result: PolicyStepResult
    memory_before: CellMemoryState
    memory_after: CellMemoryState
    memory_enabled: bool

    @property
    def behavior(self) -> str:
        return self.base_result.behavior

    @property
    def applied_action(self) -> PolicyAction:
        return self.base_result.applied_action

    @property
    def proposed_action(self) -> PolicyAction:
        return self.base_result.proposed_action


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(int(lower), min(int(upper), int(value)))


def _bounded_append(entries: Sequence[Any], entry: Any, capacity: int) -> tuple[Any, ...]:
    if capacity <= 0:
        return ()
    updated = tuple(entries) + (entry,)
    return tuple(updated[-int(capacity) :])


def _signature_row(signature: Sequence[Sequence[Any]], position: int | None) -> Sequence[Any] | None:
    if position is None:
        return None
    if position < 0 or position >= len(signature):
        return None
    return signature[position]


def _thread_id_at(signature: Sequence[Sequence[Any]], position: int | None) -> int | None:
    row = _signature_row(signature, position)
    if row is None:
        return None
    return int(row[1])


def _neighbor_entries(
    observation: PolicyObservation,
    signature_before: Sequence[Sequence[Any]],
    event_step: int | None,
) -> tuple[NeighborIdentityMemory, ...]:
    entries: list[NeighborIdentityMemory] = []
    for side, position in (("left", observation.actor_index - 1), ("right", observation.actor_index + 1)):
        if position < observation.left_boundary or position > observation.right_boundary:
            continue
        entries.append(
            NeighborIdentityMemory(
                event_step=event_step,
                side=side,
                position=int(position),
                thread_id=_thread_id_at(signature_before, position),
                value=int(observation.values[position]),
                label=str(observation.labels[position]),
                status=str(observation.statuses[position]),
            )
        )
    return tuple(entries)


def _failure_reason(result: PolicyStepResult) -> str:
    if result.applied_action.action_type == "blocked_swap_attempt" or result.frozen_attempt_delta > 0:
        return "blocked_swap_attempt"
    if result.proposed_action.action_type == "swap" and result.applied_action.action_type != "swap":
        failed_constraints = [check.name for check in result.proposed_action.constraints if not check.passed]
        if failed_constraints:
            return "proposal_constraints_failed:" + ",".join(failed_constraints)
        return "proposed_swap_not_applied"
    return "no_failed_swap"


def _failed_swap_entry(result: PolicyStepResult, event_step: int | None) -> FailedSwapMemory:
    target_index = result.proposed_action.target_index
    if target_index is None:
        target_index = result.applied_action.target_index
    target_value: int | None = None
    target_label: str | None = None
    target_status: str | None = None
    if target_index is not None and 0 <= int(target_index) < len(result.observation.values):
        target_value = int(result.observation.values[int(target_index)])
        target_label = str(result.observation.labels[int(target_index)])
        target_status = str(result.observation.statuses[int(target_index)])
    return FailedSwapMemory(
        event_step=event_step,
        actor_thread_id=int(result.observation.actor_thread_id),
        actor_position=int(result.observation.actor_index),
        actor_value=int(result.observation.actor_value),
        target_index=None if target_index is None else int(target_index),
        target_thread_id=_thread_id_at(result.signature_before, None if target_index is None else int(target_index)),
        target_value=target_value,
        target_label=target_label,
        target_status=target_status,
        reason=_failure_reason(result),
    )


class BoundedCellMemoryBank:
    """External finite memory store keyed by stable cell identity."""

    def __init__(self, config: CellMemoryConfig | None = None) -> None:
        self.config = config or CellMemoryConfig()
        self._states: dict[int, CellMemoryState] = {}

    def clear(self) -> None:
        self._states.clear()

    def get(self, thread_id: int) -> CellMemoryState:
        if not self.config.enabled:
            return CellMemoryState()
        return self._states.get(int(thread_id), CellMemoryState())

    def for_cell(self, cell: Any) -> CellMemoryState:
        return self.get(int(cell.threadID))

    def observe(self, cell: Any, label_to_behavior: Mapping[str, str] | None = None) -> MemoryPolicyObservation:
        wrapper = OriginalCellPolicyWrapper.from_cell(cell, label_to_behavior)
        local = wrapper.observe(cell)
        return MemoryPolicyObservation(local=local, memory=self.get(local.actor_thread_id), memory_config=self.config)

    def policy_state_for_cell(self, cell: Any, label_to_behavior: Mapping[str, str] | None = None) -> PolicyState:
        observation = self.observe(cell, label_to_behavior)
        return observation.state

    def update_from_step(self, result: PolicyStepResult, event_step: int | None = None) -> CellMemoryState:
        thread_id = int(result.observation.actor_thread_id)
        if not self.config.enabled:
            return CellMemoryState()

        before = self.get(thread_id)
        swap_succeeded = result.applied_action.action_type == "swap" and result.swap_delta > 0
        attempted_swap = (
            result.proposed_action.action_type == "swap"
            or result.applied_action.action_type in {"swap", "blocked_swap_attempt"}
            or result.frozen_attempt_delta > 0
        )
        failed_swap = attempted_swap and not swap_succeeded
        last_move_success = True if swap_succeeded else False if failed_swap else before.last_move_success
        time_since_movement = 0 if swap_succeeded else before.time_since_movement + 1
        time_since_movement = _clamp(time_since_movement, 0, self.config.max_time_since_movement)
        if failed_swap:
            local_frustration = before.local_frustration + int(self.config.frustration_increment)
        elif swap_succeeded:
            local_frustration = before.local_frustration - int(self.config.frustration_recovery)
        else:
            local_frustration = before.local_frustration
        local_frustration = _clamp(local_frustration, 0, self.config.max_frustration)

        recent_failed_swaps = before.recent_failed_swaps
        if failed_swap:
            recent_failed_swaps = _bounded_append(
                recent_failed_swaps,
                _failed_swap_entry(result, event_step),
                self.config.failed_swap_capacity,
            )

        recent_neighbors = before.recent_neighbor_identities
        if self.config.record_neighbor_identities:
            for entry in _neighbor_entries(result.observation, result.signature_before, event_step):
                recent_neighbors = _bounded_append(recent_neighbors, entry, self.config.neighbor_capacity)

        state = CellMemoryState(
            last_move_success=last_move_success,
            last_action_type=str(result.applied_action.action_type),
            last_target_index=result.proposed_action.target_index
            if result.proposed_action.target_index is not None
            else result.applied_action.target_index,
            time_since_movement=time_since_movement,
            local_frustration=local_frustration,
            recent_failed_swaps=tuple(recent_failed_swaps),
            recent_neighbor_identities=tuple(recent_neighbors),
        )
        self._states[thread_id] = state
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "states": {str(thread_id): state.to_dict() for thread_id, state in sorted(self._states.items())},
        }

    def compact_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "states": {str(thread_id): state.compact_dict() for thread_id, state in sorted(self._states.items())},
        }

    def stable_digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        import hashlib

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MemoryEnabledPolicyWrapper:
    """Behavior-preserving original-policy wrapper with ablatable memory."""

    def __init__(
        self,
        behavior: str,
        *,
        memory_bank: BoundedCellMemoryBank | None = None,
        memory_config: CellMemoryConfig | None = None,
        label_to_behavior: Mapping[str, str] | None = None,
    ) -> None:
        if behavior not in ORIGINAL_BEHAVIORS:
            raise ValueError(f"Unsupported original behavior: {behavior!r}")
        self.behavior = behavior
        self.label_to_behavior = dict(label_to_behavior or {})
        self.memory_bank = memory_bank or BoundedCellMemoryBank(memory_config)
        self.original = OriginalCellPolicyWrapper(behavior, self.label_to_behavior)

    @classmethod
    def from_cell(
        cls,
        cell: Any,
        *,
        memory_bank: BoundedCellMemoryBank | None = None,
        memory_config: CellMemoryConfig | None = None,
        label_to_behavior: Mapping[str, str] | None = None,
    ) -> "MemoryEnabledPolicyWrapper":
        original = OriginalCellPolicyWrapper.from_cell(cell, label_to_behavior)
        return cls(
            original.behavior,
            memory_bank=memory_bank,
            memory_config=memory_config,
            label_to_behavior=label_to_behavior,
        )

    def observe(self, cell: Any) -> MemoryPolicyObservation:
        return self.memory_bank.observe(cell, self.label_to_behavior)

    def state(self, cell: Any) -> PolicyState:
        return self.memory_bank.policy_state_for_cell(cell, self.label_to_behavior)

    def propose_action(self, observation: PolicyObservation | MemoryPolicyObservation, rng: random.Random | None = None) -> PolicyAction:
        local = observation.local if isinstance(observation, MemoryPolicyObservation) else observation
        return self.original.propose_action(local, rng)

    def step(self, cell: Any, event_step: int | None = None) -> MemoryStepResult:
        memory_before = self.memory_bank.for_cell(cell)
        result = self.original.step(cell)
        memory_after = self.memory_bank.update_from_step(result, event_step=event_step)
        return MemoryStepResult(
            base_result=result,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_enabled=bool(self.memory_bank.config.enabled),
        )


def memory_policy_for_cell(
    cell: Any,
    memory_bank: BoundedCellMemoryBank | None = None,
    memory_config: CellMemoryConfig | None = None,
    label_to_behavior: Mapping[str, str] | None = None,
) -> MemoryEnabledPolicyWrapper:
    """Return a memory-enabled wrapper for a public cell instance."""

    return MemoryEnabledPolicyWrapper.from_cell(
        cell,
        memory_bank=memory_bank,
        memory_config=memory_config,
        label_to_behavior=label_to_behavior,
    )


def memory_signature(bank: BoundedCellMemoryBank) -> tuple[tuple[str, Any], ...]:
    """Return a compact deterministic signature for unit tests."""

    return tuple(sorted(bank.compact_dict()["states"].items()))


def public_signature(cells: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    """Expose E03 public-state signature through the E04 module."""

    return cells_signature(cells)
