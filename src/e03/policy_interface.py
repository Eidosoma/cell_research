"""Local-rule policy interface for E03 morphospace work.

The E02 simulator preserves the public repository's cell-view behavior by
calling each original ``move()`` method.  This module adds a small interface
around those same public methods: callers can inspect an observation, expose
the cell's internal policy state, preview the action that the public rule would
try, and then execute the original method without replacing its behavior.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


ORIGINAL_BEHAVIORS = ("bubble", "insertion", "selection")


@dataclass(frozen=True)
class ConstraintCheck:
    """One boolean gate used by a local-rule action proposal."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class PolicyState:
    """Internal state exposed by a local-rule policy wrapper."""

    ideal_position: int | None
    reverse_direction: bool
    memory: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyObservation:
    """Cell-local observation plus minimal global context used by originals."""

    actor_index: int
    actor_thread_id: int
    actor_value: int
    actor_label: str
    actor_status: str
    behavior: str
    values: tuple[int, ...]
    labels: tuple[str, ...]
    statuses: tuple[str, ...]
    left_boundary: int
    right_boundary: int
    reverse_direction: bool
    ideal_position: int | None
    group_status: str

    @property
    def state(self) -> PolicyState:
        return PolicyState(
            ideal_position=self.ideal_position,
            reverse_direction=self.reverse_direction,
        )


@dataclass(frozen=True)
class PolicyAction:
    """A proposed or applied local action."""

    action_type: str
    target_index: int | None = None
    compare_counted: bool = False
    constraints: tuple[ConstraintCheck, ...] = ()
    state_update: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return all(check.passed for check in self.constraints)


@dataclass(frozen=True)
class PolicyStepResult:
    """Result of executing one public cell method through the interface."""

    behavior: str
    observation: PolicyObservation
    state_before: PolicyState
    proposed_action: PolicyAction
    applied_action: PolicyAction
    signature_before: tuple[tuple[Any, ...], ...]
    signature_after: tuple[tuple[Any, ...], ...]
    comparison_delta: int
    swap_delta: int
    frozen_attempt_delta: int
    state_changed: bool


class LocalRulePolicy(Protocol):
    """Protocol implemented by local-rule policies and public wrappers."""

    behavior: str

    def observe(self, cell: Any) -> PolicyObservation:
        """Return the policy-visible observation for ``cell``."""

    def state(self, cell: Any) -> PolicyState:
        """Return internal policy state for ``cell``."""

    def propose_action(self, observation: PolicyObservation, rng: random.Random | None = None) -> PolicyAction:
        """Return the next local action proposal without mutating ``cell``."""

    def step(self, cell: Any) -> PolicyStepResult:
        """Execute one local policy step against the live cell object."""


def _status_name(status: Any) -> str:
    return getattr(status, "name", str(status))


def _cell_behavior(cell: Any, label_to_behavior: Mapping[str, str] | None = None) -> str:
    label = str(getattr(cell, "label", ""))
    behavior_map = dict(label_to_behavior or {})
    if label in behavior_map:
        behavior = str(behavior_map[label])
    else:
        cell_type = str(getattr(cell, "cell_type", "")).lower()
        behavior = behavior_map.get(cell_type, cell_type)
    if behavior not in ORIGINAL_BEHAVIORS:
        raise ValueError(f"Unsupported original behavior: {behavior!r}")
    return behavior


def _active_status_name(cell: Any) -> str:
    return _status_name(cell.status.__class__.ACTIVE)


def _freeze_status_name(cell: Any) -> str:
    return _status_name(cell.status.__class__.FREEZE)


def _within_boundary(observation: PolicyObservation, target_index: int | None) -> bool:
    return target_index is not None and observation.left_boundary <= target_index <= observation.right_boundary


def _target_status(observation: PolicyObservation, target_index: int | None) -> str | None:
    if target_index is None or target_index < 0 or target_index >= len(observation.statuses):
        return None
    return observation.statuses[target_index]


def _target_active_or_frozen(observation: PolicyObservation, target_index: int | None) -> bool:
    status = _target_status(observation, target_index)
    return status in {"ACTIVE", "FREEZE", "CellStatus.ACTIVE", "CellStatus.FREEZE"}


def _target_active(observation: PolicyObservation, target_index: int | None) -> bool:
    status = _target_status(observation, target_index)
    return status in {"ACTIVE", "CellStatus.ACTIVE"}


def _actor_active(observation: PolicyObservation) -> bool:
    return observation.actor_status in {"ACTIVE", "CellStatus.ACTIVE"}


def _selection_next_ideal(observation: PolicyObservation) -> int | None:
    if observation.ideal_position is None:
        return None
    if observation.reverse_direction:
        return observation.ideal_position - 1
    return observation.ideal_position + 1


def _insertion_enable_flag(observation: PolicyObservation) -> bool:
    """Match public InsertionSortCell.is_enable_to_move()."""

    prev = 100000 if observation.reverse_direction else -1
    for idx in range(observation.left_boundary, observation.actor_index):
        if observation.statuses[idx] in {"FREEZE", "CellStatus.FREEZE"}:
            # The public implementation resets to -1 for both directions.
            prev = -1
            continue
        value = observation.values[idx]
        if observation.reverse_direction and value > prev:
            return False
        if not observation.reverse_direction and value < prev:
            return False
        prev = value
    return True


def _bubble_compare_counted(observation: PolicyObservation) -> bool:
    left_idx = observation.actor_index - 1
    right_idx = observation.actor_index + 1
    if observation.reverse_direction:
        bigger_than_left = (
            left_idx >= observation.left_boundary
            and _target_active(observation, left_idx)
            and observation.actor_value > observation.values[left_idx]
        )
        smaller_than_right = (
            right_idx <= observation.right_boundary
            and _target_active(observation, right_idx)
            and observation.actor_value < observation.values[right_idx]
        )
        return bigger_than_left or smaller_than_right
    smaller_than_left = (
        left_idx >= observation.left_boundary
        and _target_active(observation, left_idx)
        and observation.actor_value < observation.values[left_idx]
    )
    bigger_than_right = (
        right_idx <= observation.right_boundary
        and _target_active(observation, right_idx)
        and observation.actor_value > observation.values[right_idx]
    )
    return smaller_than_left or bigger_than_right


def _insertion_compare_counted(observation: PolicyObservation) -> bool:
    left_idx = observation.actor_index - 1
    if left_idx < observation.left_boundary:
        return False
    if not _target_active(observation, left_idx):
        return False
    if not _insertion_enable_flag(observation):
        return False
    if observation.reverse_direction:
        return observation.actor_value > observation.values[left_idx]
    return observation.actor_value < observation.values[left_idx]


def _selection_compare_counted(observation: PolicyObservation) -> bool:
    return (
        observation.ideal_position is not None
        and observation.actor_index != observation.ideal_position
        and _within_boundary(observation, observation.ideal_position)
    )


def _constraints(*items: tuple[str, bool, str]) -> tuple[ConstraintCheck, ...]:
    return tuple(ConstraintCheck(name=name, passed=bool(passed), detail=detail) for name, passed, detail in items)


def cells_signature(cells: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    """Return a compact state signature for decision-level parity tests."""

    return tuple(
        (
            position,
            int(cell.threadID),
            int(cell.value),
            str(getattr(cell, "label", "")),
            str(getattr(cell, "cell_type", "")),
            _status_name(cell.status),
            int(cell.current_position[0]),
            None if cell.ideal_position is None else int(cell.ideal_position[0]),
            bool(cell.reverse_direction),
        )
        for position, cell in enumerate(cells)
    )


def observe_cell(cell: Any, label_to_behavior: Mapping[str, str] | None = None) -> PolicyObservation:
    """Build the local-rule observation for a public cell object."""

    cells = tuple(cell.cells)
    return PolicyObservation(
        actor_index=int(cell.current_position[0]),
        actor_thread_id=int(cell.threadID),
        actor_value=int(cell.value),
        actor_label=str(getattr(cell, "label", "")),
        actor_status=_status_name(cell.status),
        behavior=_cell_behavior(cell, label_to_behavior),
        values=tuple(int(item.value) for item in cells),
        labels=tuple(str(getattr(item, "label", "")) for item in cells),
        statuses=tuple(_status_name(item.status) for item in cells),
        left_boundary=int(cell.left_boundary[0]),
        right_boundary=int(cell.right_boundary[0]),
        reverse_direction=bool(cell.reverse_direction),
        ideal_position=None if cell.ideal_position is None else int(cell.ideal_position[0]),
        group_status=_status_name(cell.group.status),
    )


def extract_policy_state(cell: Any) -> PolicyState:
    """Extract current internal policy state from a public cell object."""

    return PolicyState(
        ideal_position=None if cell.ideal_position is None else int(cell.ideal_position[0]),
        reverse_direction=bool(cell.reverse_direction),
        memory={
            "tried_to_swap_with_frozen": bool(getattr(cell, "tried_to_swap_with_frozen", False)),
            "with_lock": bool(getattr(cell, "with_lock", False)),
        },
    )


def _clone_global_rng() -> random.Random:
    clone = random.Random()
    clone.setstate(random.getstate())
    return clone


class OriginalCellPolicyWrapper:
    """Adapter exposing the public Bubble, Insertion, and Selection methods."""

    def __init__(self, behavior: str, label_to_behavior: Mapping[str, str] | None = None) -> None:
        if behavior not in ORIGINAL_BEHAVIORS:
            raise ValueError(f"Unsupported original behavior: {behavior!r}")
        self.behavior = behavior
        self.label_to_behavior = dict(label_to_behavior or {})

    @classmethod
    def from_cell(cls, cell: Any, label_to_behavior: Mapping[str, str] | None = None) -> "OriginalCellPolicyWrapper":
        return cls(_cell_behavior(cell, label_to_behavior), label_to_behavior)

    def observe(self, cell: Any) -> PolicyObservation:
        return observe_cell(cell, self.label_to_behavior)

    def state(self, cell: Any) -> PolicyState:
        return extract_policy_state(cell)

    def propose_action(self, observation: PolicyObservation, rng: random.Random | None = None) -> PolicyAction:
        rng = rng or _clone_global_rng()
        if self.behavior == "bubble":
            return self._propose_bubble(observation, rng)
        if self.behavior == "insertion":
            return self._propose_insertion(observation, rng)
        if self.behavior == "selection":
            return self._propose_selection(observation)
        raise ValueError(f"Unsupported original behavior: {self.behavior!r}")

    def _propose_bubble(self, observation: PolicyObservation, rng: random.Random) -> PolicyAction:
        compare_counted = _bubble_compare_counted(observation)
        check_right = rng.random() < 0.5
        target_index = observation.actor_index + 1 if check_right else observation.actor_index - 1
        constraints = _constraints(
            ("actor_active", _actor_active(observation), observation.actor_status),
            ("within_boundary", _within_boundary(observation, target_index), str(target_index)),
            ("target_active_or_frozen", _target_active_or_frozen(observation, target_index), str(_target_status(observation, target_index))),
        )
        if all(check.passed for check in constraints):
            # The public method consumes this random draw, although the branch is
            # impossible because the threshold is zero.
            rng.random()
            target_value = observation.values[target_index]
            if observation.reverse_direction:
                value_relation = observation.actor_value < target_value if check_right else observation.actor_value > target_value
            else:
                value_relation = observation.actor_value > target_value if check_right else observation.actor_value < target_value
            constraints = (*constraints, ConstraintCheck("value_relation", bool(value_relation), f"target_value={target_value}"))
            action_type = "swap" if value_relation else "wait"
        else:
            action_type = "wait"
        return PolicyAction(
            action_type=action_type,
            target_index=target_index,
            compare_counted=compare_counted,
            constraints=constraints,
            metadata={"check_right": check_right, "source_method": "BubbleSortCell.move"},
        )

    def _propose_insertion(self, observation: PolicyObservation, rng: random.Random) -> PolicyAction:
        target_index = observation.actor_index - 1
        enabled = _insertion_enable_flag(observation)
        compare_counted = _insertion_compare_counted(observation)
        constraints = _constraints(
            ("insertion_prefix_enabled", enabled, "matches is_enable_to_move"),
            ("actor_active_or_frozen", observation.actor_status in {"ACTIVE", "FREEZE"}, observation.actor_status),
            ("within_boundary", _within_boundary(observation, target_index), str(target_index)),
            ("target_active_or_frozen", _target_active_or_frozen(observation, target_index), str(_target_status(observation, target_index))),
        )
        if enabled and all(check.passed for check in constraints[1:]):
            rng.random()
            target_value = observation.values[target_index]
            if observation.reverse_direction:
                value_relation = observation.actor_value > target_value
            else:
                value_relation = observation.actor_value < target_value
            constraints = (*constraints, ConstraintCheck("value_relation", bool(value_relation), f"target_value={target_value}"))
            action_type = "swap" if value_relation else "wait"
        else:
            action_type = "wait"
        return PolicyAction(
            action_type=action_type,
            target_index=target_index,
            compare_counted=compare_counted,
            constraints=constraints,
            metadata={"source_method": "InsertionSortCell.move"},
        )

    def _propose_selection(self, observation: PolicyObservation) -> PolicyAction:
        target_index = observation.ideal_position
        compare_counted = _selection_compare_counted(observation)
        constraints = _constraints(
            ("actor_active", _actor_active(observation), observation.actor_status),
            ("has_ideal_position", target_index is not None, str(target_index)),
            ("within_boundary", _within_boundary(observation, target_index), str(target_index)),
            ("not_at_ideal_position", target_index is not None and observation.actor_index != target_index, str(observation.actor_index)),
        )
        if not all(check.passed for check in constraints):
            return PolicyAction(
                action_type="wait",
                target_index=target_index,
                compare_counted=compare_counted,
                constraints=constraints,
                metadata={"source_method": "SelectionSortCell.move"},
            )
        target_status = _target_status(observation, target_index)
        target_value = observation.values[target_index]
        if target_status in {"FREEZE", "CellStatus.FREEZE"}:
            next_ideal = _selection_next_ideal(observation)
            action_type = "blocked_swap_attempt" if observation.actor_value < target_value else "update_state"
            return PolicyAction(
                action_type=action_type,
                target_index=target_index,
                compare_counted=compare_counted,
                constraints=(*constraints, ConstraintCheck("target_frozen", True, str(target_status))),
                state_update={"ideal_position": next_ideal},
                metadata={"source_method": "SelectionSortCell.should_move_to"},
            )
        constraints = (*constraints, ConstraintCheck("target_active", _target_active(observation, target_index), str(target_status)))
        if not _target_active(observation, target_index):
            return PolicyAction(
                action_type="wait",
                target_index=target_index,
                compare_counted=compare_counted,
                constraints=constraints,
                metadata={"source_method": "SelectionSortCell.move"},
            )
        if observation.actor_value >= target_value:
            return PolicyAction(
                action_type="update_state",
                target_index=target_index,
                compare_counted=compare_counted,
                constraints=(*constraints, ConstraintCheck("value_relation_swap", False, f"target_value={target_value}")),
                state_update={"ideal_position": _selection_next_ideal(observation)},
                metadata={"source_method": "SelectionSortCell.should_move_to"},
            )
        return PolicyAction(
            action_type="swap",
            target_index=target_index,
            compare_counted=compare_counted,
            constraints=(*constraints, ConstraintCheck("value_relation_swap", True, f"target_value={target_value}")),
            metadata={"source_method": "SelectionSortCell.move"},
        )

    def step(self, cell: Any) -> PolicyStepResult:
        observation = self.observe(cell)
        state_before = self.state(cell)
        proposal = self.propose_action(observation)
        signature_before = cells_signature(cell.cells)
        actor_thread_id = int(cell.threadID)
        probe = cell.status_probe
        before_comparisons = int(probe.compare_and_swap_count)
        before_swaps = int(probe.swap_count)
        before_frozen_attempts = int(probe.frozen_swap_attempts)
        cell.move()
        signature_after = cells_signature(cell.cells)
        comparison_delta = int(probe.compare_and_swap_count) - before_comparisons
        swap_delta = int(probe.swap_count) - before_swaps
        frozen_attempt_delta = int(probe.frozen_swap_attempts) - before_frozen_attempts
        applied_action = infer_applied_action(
            signature_before,
            signature_after,
            actor_thread_id=actor_thread_id,
            comparison_delta=comparison_delta,
            swap_delta=swap_delta,
            frozen_attempt_delta=frozen_attempt_delta,
        )
        return PolicyStepResult(
            behavior=self.behavior,
            observation=observation,
            state_before=state_before,
            proposed_action=proposal,
            applied_action=applied_action,
            signature_before=signature_before,
            signature_after=signature_after,
            comparison_delta=comparison_delta,
            swap_delta=swap_delta,
            frozen_attempt_delta=frozen_attempt_delta,
            state_changed=signature_before != signature_after,
        )


def infer_applied_action(
    signature_before: Sequence[Sequence[Any]],
    signature_after: Sequence[Sequence[Any]],
    *,
    actor_thread_id: int,
    comparison_delta: int,
    swap_delta: int,
    frozen_attempt_delta: int,
) -> PolicyAction:
    """Infer the observable action applied by one public ``move()`` call."""

    before_by_thread = {int(row[1]): row for row in signature_before}
    after_by_thread = {int(row[1]): row for row in signature_after}
    actor_before = before_by_thread[actor_thread_id]
    actor_after = after_by_thread[actor_thread_id]
    compare_counted = comparison_delta > 0
    if swap_delta > 0:
        return PolicyAction(
            action_type="swap",
            target_index=int(actor_after[0]),
            compare_counted=compare_counted,
            metadata={"swap_delta": swap_delta, "frozen_attempt_delta": frozen_attempt_delta},
        )
    if frozen_attempt_delta > 0:
        return PolicyAction(
            action_type="blocked_swap_attempt",
            target_index=int(actor_before[0]),
            compare_counted=compare_counted,
            metadata={"swap_delta": swap_delta, "frozen_attempt_delta": frozen_attempt_delta},
        )
    if actor_before[7] != actor_after[7]:
        return PolicyAction(
            action_type="update_state",
            target_index=None if actor_before[7] is None else int(actor_before[7]),
            compare_counted=compare_counted,
            state_update={"ideal_position": actor_after[7]},
        )
    return PolicyAction(action_type="wait", compare_counted=compare_counted)


def policy_for_cell(cell: Any, label_to_behavior: Mapping[str, str] | None = None) -> OriginalCellPolicyWrapper:
    """Return an original-policy wrapper for a public cell instance."""

    return OriginalCellPolicyWrapper.from_cell(cell, label_to_behavior)
