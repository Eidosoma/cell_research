"""Deterministic event simulator for public cell-view sorting policies.

The public repository implements each cell as a Python thread.  This module
keeps the public Bubble, Insertion, and Selection ``move()`` methods as the
policy layer, but replaces OS thread interleaving with an explicit event loop
that activates one active cell at a time from a seeded distribution.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


ALGORITHMS = ("bubble", "insertion", "selection")
DEFAULT_LABEL_TO_BEHAVIOR = {
    "bubble": "bubble",
    "insertion": "insertion",
    "selection": "selection",
    "bubble_label_a": "bubble",
    "bubble_label_b": "bubble",
}
DEFAULT_LABEL_TO_CODE = {
    "bubble": "B",
    "insertion": "I",
    "selection": "S",
    "bubble_label_a": "A",
    "bubble_label_b": "C",
}
SUPPORTED_ACTIVATION_DISTRIBUTIONS = (
    "uniform_active",
    "left_to_right_active",
    "right_to_left_active",
)


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration for one deterministic cell-view run."""

    values: tuple[int, ...]
    algorithm: str | None = None
    algotypes: tuple[str, ...] | None = None
    label_to_behavior: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_LABEL_TO_BEHAVIOR))
    reverse_directions: Mapping[str, bool] | tuple[bool, ...] | None = None
    frozen_indices: tuple[int, ...] = ()
    frozen_semantics: str = "none"
    frozen_behavior: Mapping[str, Any] | None = None
    activation_seed: int = 0
    policy_seed: int | None = None
    activation_distribution: str = "uniform_active"
    max_events: int = 100_000
    max_successful_swaps: int | None = 50_000
    stall_events: int = 1_000
    stop_when_sorted: bool = True
    sort_direction: str = "increasing"
    repo_dir: Path | None = None


@dataclass(frozen=True)
class SimulatorResult:
    """In-memory result from one deterministic event simulation."""

    config: SimulatorConfig
    records: tuple[dict[str, Any], ...]
    activation_log: tuple[dict[str, Any], ...]
    final_values: tuple[int, ...]
    final_labels: tuple[str, ...]
    final_frozen_positions: tuple[int, ...]
    final_frozen_values: tuple[int, ...]
    stop_reason: str
    max_guard_hit: bool
    event_count: int
    swap_count: int
    comparison_count: int
    frozen_attempt_count: int
    compare_plus_swap_count: int
    frozen_behavior_transition_log: tuple[dict[str, Any], ...] = ()
    frozen_behavior_attempt_log: tuple[dict[str, Any], ...] = ()
    frozen_behavior_summary: Mapping[str, Any] = field(default_factory=dict)
    wrapper_name: str = "deterministic_single_event_public_cell_methods"


class EventTracingStatusProbe:
    """StatusProbe-compatible recorder with event-level metadata."""

    def __init__(self) -> None:
        self.sorting_steps: list[list[int]] = []
        self.swap_count = 0
        self.cell_types: list[list[list[Any]]] = []
        self.frozen_swap_attempts = 0
        self.compare_and_swap_count = 0
        self.comparison_counts_at_step: list[int] = []
        self.frozen_attempt_counts_at_step: list[int] = []
        self.swap_counts_at_step: list[int] = []
        self.event_steps_at_step: list[int] = []
        self.current_event_step = 0

    def record_swap(self) -> None:
        self.swap_count += 1

    def record_compare_and_swap(self) -> None:
        self.compare_and_swap_count += 1

    def record_sorting_step(self, snapshot: list[int]) -> None:
        self.sorting_steps.append(list(snapshot))
        self.comparison_counts_at_step.append(int(self.compare_and_swap_count))
        self.frozen_attempt_counts_at_step.append(int(self.frozen_swap_attempts))
        self.swap_counts_at_step.append(int(self.swap_count))
        self.event_steps_at_step.append(int(self.current_event_step))

    def record_cell_type(self, snapshot: list[list[Any]]) -> None:
        self.cell_types.append(snapshot)

    def count_frozen_cell_attempt(self) -> None:
        self.frozen_swap_attempts += 1


def default_repo_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def stable_json_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def import_cell_modules(repo_dir: Path | None = None) -> dict[str, Any]:
    resolved = repo_dir or default_repo_dir()
    repo_str = str(resolved)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from modules.multithread.BubbleSortCell import BubbleSortCell
    from modules.multithread.CellGroup import CellGroup, GroupStatus
    from modules.multithread.InsertionSortCell import InsertionSortCell
    from modules.multithread.MultiThreadCell import CellStatus
    from modules.multithread.SelectionSortCell import SelectionSortCell

    return {
        "BubbleSortCell": BubbleSortCell,
        "CellGroup": CellGroup,
        "CellStatus": CellStatus,
        "GroupStatus": GroupStatus,
        "InsertionSortCell": InsertionSortCell,
        "SelectionSortCell": SelectionSortCell,
    }


def sortedness_percent(values: Sequence[int], direction: str = "increasing") -> float:
    if not values:
        return 100.0
    if direction in {"increasing", "nondecreasing", "all_increasing"}:
        ordered_pairs = sum(1 for idx in range(1, len(values)) if values[idx - 1] <= values[idx])
    elif direction in {"decreasing", "nonincreasing", "all_decreasing"}:
        ordered_pairs = sum(1 for idx in range(1, len(values)) if values[idx - 1] >= values[idx])
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return 100.0 * (1 + ordered_pairs) / len(values)


def monotonicity_error_count(values: Sequence[int], direction: str = "increasing") -> int:
    if direction in {"increasing", "nondecreasing", "all_increasing"}:
        return sum(1 for idx in range(1, len(values)) if values[idx] < values[idx - 1])
    if direction in {"decreasing", "nonincreasing", "all_decreasing"}:
        return sum(1 for idx in range(1, len(values)) if values[idx] > values[idx - 1])
    raise ValueError(f"Unsupported direction: {direction}")


def is_sorted(values: Sequence[int], direction: str = "increasing") -> bool:
    return monotonicity_error_count(values, direction) == 0


def aggregation_left_neighbor_percent(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    same_left = sum(1 for idx in range(1, len(labels)) if labels[idx] == labels[idx - 1])
    return 100.0 * same_left / len(labels)


def aggregation_right_neighbor_legacy_percent(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    same_right = sum(1 for idx in range(len(labels) - 1) if labels[idx] == labels[idx + 1])
    return 100.0 * same_right / len(labels)


def label_code(labels: Sequence[str]) -> str:
    return "".join(DEFAULT_LABEL_TO_CODE.get(label, str(label)[:1].upper() or "?") for label in labels)


def count_json(labels: Sequence[str]) -> str:
    return json.dumps(dict(sorted(Counter(labels).items())), separators=(",", ":"))


def cell_values(cells: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(cell.value) for cell in cells)


def cell_labels(cells: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(cell.label) for cell in cells)


def cell_directions(cells: Sequence[Any]) -> tuple[str, ...]:
    return tuple("decreasing" if bool(cell.reverse_direction) else "increasing" for cell in cells)


def frozen_positions(cells: Sequence[Any], cell_status: Any) -> tuple[int, ...]:
    return tuple(sorted(int(cell.current_position[0]) for cell in cells if cell.status == cell_status.FREEZE))


def frozen_values(cells: Sequence[Any], cell_status: Any) -> tuple[int, ...]:
    return tuple(sorted(int(cell.value) for cell in cells if cell.status == cell_status.FREEZE))


def cell_state_signature(cells: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            int(cell.threadID),
            int(cell.value),
            str(cell.label),
            bool(cell.reverse_direction),
            str(cell.status),
            int(cell.current_position[0]),
            int(cell.ideal_position[0]) if cell.ideal_position is not None else None,
        )
        for cell in cells
    )


def _validate_config(config: SimulatorConfig) -> None:
    if config.algorithm is None and config.algotypes is None:
        raise ValueError("Either algorithm or algotypes must be provided.")
    if config.algorithm is not None and config.algorithm not in ALGORITHMS and config.algotypes is None:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    if config.algotypes is not None and len(config.algotypes) != len(config.values):
        raise ValueError("algotypes length must match values length.")
    if config.frozen_semantics not in {"none", "passive", "stuck", "dynamic"}:
        raise ValueError(f"Unsupported frozen_semantics: {config.frozen_semantics}")
    if config.frozen_semantics == "none" and config.frozen_indices:
        raise ValueError("frozen_indices require passive, stuck, or dynamic frozen_semantics.")
    if config.frozen_semantics == "dynamic" and config.frozen_behavior is None:
        raise ValueError("dynamic frozen_semantics requires frozen_behavior.")
    if config.frozen_behavior is not None and config.frozen_semantics != "dynamic":
        raise ValueError("frozen_behavior is only supported with dynamic frozen_semantics.")
    if config.activation_distribution not in SUPPORTED_ACTIVATION_DISTRIBUTIONS:
        raise ValueError(f"Unsupported activation_distribution: {config.activation_distribution}")
    if config.max_events <= 0:
        raise ValueError("max_events must be positive.")
    if config.stall_events <= 0:
        raise ValueError("stall_events must be positive.")
    frozen_set = set(config.frozen_indices)
    if len(frozen_set) != len(config.frozen_indices):
        raise ValueError("frozen_indices must be unique.")
    if any(idx < 0 or idx >= len(config.values) for idx in frozen_set):
        raise ValueError("frozen_indices out of range.")


def _labels_for_config(config: SimulatorConfig) -> tuple[str, ...]:
    if config.algotypes is not None:
        return tuple(map(str, config.algotypes))
    if config.algorithm is None:
        raise ValueError("algorithm is required when algotypes are absent.")
    return tuple(config.algorithm for _ in config.values)


def _reverse_flags_for_config(config: SimulatorConfig, labels: Sequence[str]) -> tuple[bool, ...]:
    if config.reverse_directions is None:
        return tuple(False for _ in labels)
    if isinstance(config.reverse_directions, Mapping):
        return tuple(bool(config.reverse_directions.get(label, False)) for label in labels)
    flags = tuple(bool(flag) for flag in config.reverse_directions)
    if len(flags) != len(labels):
        raise ValueError("reverse_directions sequence length must match values length.")
    return flags


def _behavior_for_label(config: SimulatorConfig, label: str) -> str:
    if label in config.label_to_behavior:
        behavior = config.label_to_behavior[label]
    elif label in ALGORITHMS:
        behavior = label
    elif config.algorithm in ALGORITHMS:
        behavior = str(config.algorithm)
    else:
        raise ValueError(f"No behavior mapping for Algotype label {label!r}.")
    if behavior not in ALGORITHMS:
        raise ValueError(f"Unsupported behavior {behavior!r} for label {label!r}.")
    return behavior


def _patch_stuck_swap(cells: Sequence[Any], frozen_thread_ids: set[int]) -> None:
    cell_status_enum = cells[0].status.__class__

    def make_swap(cell: Any) -> Any:
        original_swap = cell.swap

        def stuck_swap(target_position: tuple[int, int], skip_stats: bool = False) -> Any:
            if cell.status == cell_status_enum.FREEZE:
                if not cell.tried_to_swap_with_frozen:
                    cell.status_probe.count_frozen_cell_attempt()
                    cell.tried_to_swap_with_frozen = True
                return None
            target = cell.cells[int(target_position[0])]
            if target.threadID in frozen_thread_ids or target.status == cell_status_enum.FREEZE:
                if not cell.tried_to_swap_with_frozen:
                    cell.status_probe.count_frozen_cell_attempt()
                    cell.tried_to_swap_with_frozen = True
                return None
            cell.tried_to_swap_with_frozen = False
            target.tried_to_swap_with_frozen = False
            return original_swap(target_position, skip_stats)

        return stuck_swap

    for cell in cells:
        cell.swap = make_swap(cell)


class FrozenBehaviorController:
    """Runtime policy for S12 dynamic Frozen Cell target behavior."""

    MAX_LOG_ENTRIES = 512

    def __init__(
        self,
        cells: Sequence[Any],
        cell_status: Any,
        behavior: Mapping[str, Any],
        *,
        seed: int,
    ) -> None:
        self.cell_status = cell_status
        self.behavior = dict(behavior)
        self.behavior_type = str(self.behavior.get("type", "passive_limit"))
        self.behavior_label = str(self.behavior.get("label", self.behavior_type))
        self.frozen_thread_ids = {
            int(cell.threadID)
            for cell in cells
            if cell.status == cell_status.FREEZE
        }
        if not self.frozen_thread_ids:
            raise ValueError("dynamic frozen behavior requires at least one frozen cell.")
        self.seed = int(self.behavior.get("seed", seed))
        self.rng = random.Random(self.seed)
        self.current_event_step = 0
        self.window_events = max(1, int(self.behavior.get("window_events", 250)))
        self.block_probability = float(self.behavior.get("block_probability", 0.5))
        if not 0.0 <= self.block_probability <= 1.0:
            raise ValueError("block_probability must be in [0, 1].")
        self.fatigue_threshold = max(1, int(self.behavior.get("fatigue_threshold", 4)))
        self.recovery_events = max(1, int(self.behavior.get("recovery_events", 250)))
        self.sticky_blocked_from = str(self.behavior.get("blocked_from", "left"))
        if self.sticky_blocked_from not in {"left", "right"}:
            raise ValueError("directional sticky blocked_from must be left or right.")
        self.state_by_thread: dict[int, str] = {}
        self.fatigue_by_thread: dict[int, int] = {thread_id: 0 for thread_id in self.frozen_thread_ids}
        self.recover_until_by_thread: dict[int, int] = {thread_id: 0 for thread_id in self.frozen_thread_ids}
        self.transition_log: list[dict[str, Any]] = []
        self.attempt_log: list[dict[str, Any]] = []
        self.transition_count = 0
        self.attempt_counts: Counter[tuple[str, str, str]] = Counter()
        self.allowed_attempt_count = 0
        self.blocked_attempt_count = 0
        for cell in cells:
            if int(cell.threadID) in self.frozen_thread_ids:
                state = self._initial_state()
                self.state_by_thread[int(cell.threadID)] = state
                self._apply_frozen_status(cell)
                self._log_transition(cell, None, state, "initial_state")

    def _initial_state(self) -> str:
        if self.behavior_type == "passive_limit":
            return "passive"
        if self.behavior_type == "stuck_limit":
            return "stuck"
        if self.behavior_type == "probabilistic_stuck":
            return "probabilistic"
        if self.behavior_type == "time_varying":
            return self._time_state(0)
        if self.behavior_type == "fatigue_recovery":
            return "passive"
        if self.behavior_type == "directional_sticky":
            return "directional_sticky"
        raise ValueError(f"Unsupported frozen behavior type: {self.behavior_type}")

    def _time_state(self, event_step: int) -> str:
        phase = (int(event_step) // self.window_events) % 2
        return "stuck" if phase == 0 else "passive"

    def _apply_frozen_status(self, cell: Any) -> None:
        cell.status = self.cell_status.FREEZE
        cell.previous_status = self.cell_status.FREEZE

    def _log_transition(self, cell: Any, from_state: str | None, to_state: str, reason: str) -> None:
        self.transition_count += 1
        if len(self.transition_log) >= self.MAX_LOG_ENTRIES:
            return
        self.transition_log.append(
            {
                "event_step": int(self.current_event_step),
                "thread_id": int(cell.threadID),
                "position": int(cell.current_position[0]),
                "value": int(cell.value),
                "from_state": from_state,
                "to_state": str(to_state),
                "reason": str(reason),
                "behavior_label": self.behavior_label,
            }
        )

    def _set_state(self, cell: Any, new_state: str, reason: str) -> None:
        thread_id = int(cell.threadID)
        old_state = self.state_by_thread.get(thread_id)
        if old_state != new_state:
            self.state_by_thread[thread_id] = new_state
            self._log_transition(cell, old_state, new_state, reason)
        self._apply_frozen_status(cell)

    def before_event(self, event_step: int, cells: Sequence[Any]) -> None:
        self.current_event_step = int(event_step)
        for cell in cells:
            if int(cell.threadID) not in self.frozen_thread_ids:
                continue
            if self.behavior_type == "time_varying":
                self._set_state(cell, self._time_state(event_step), "periodic_time_phase")
            elif self.behavior_type == "fatigue_recovery":
                thread_id = int(cell.threadID)
                if (
                    self.state_by_thread.get(thread_id) == "recovering_stuck"
                    and event_step >= self.recover_until_by_thread.get(thread_id, 0)
                ):
                    self.fatigue_by_thread[thread_id] = 0
                    self._set_state(cell, "passive", "recovery_window_elapsed")
                else:
                    self._apply_frozen_status(cell)
            else:
                self._apply_frozen_status(cell)

    def is_frozen_identity(self, cell: Any) -> bool:
        return int(cell.threadID) in self.frozen_thread_ids

    def _direction_from_actor(self, actor: Any, target: Any) -> str:
        return "left" if int(actor.current_position[0]) < int(target.current_position[0]) else "right"

    def _direction_is_blocked(self, actor: Any, target: Any) -> bool:
        return self._direction_from_actor(actor, target) == self.sticky_blocked_from

    def target_potentially_movable(self, actor: Any, target: Any) -> bool:
        if target.status == self.cell_status.ACTIVE:
            return True
        if not self.is_frozen_identity(target):
            return False
        state = self.state_by_thread.get(int(target.threadID), "stuck")
        if state == "passive":
            return True
        if state in {"stuck", "recovering_stuck"}:
            return False
        if state == "probabilistic":
            return self.block_probability < 1.0
        if state == "directional_sticky":
            return not self._direction_is_blocked(actor, target)
        return False

    def _record_attempt(
        self,
        actor: Any,
        target: Any,
        *,
        state: str,
        direction: str,
        decision: str,
        roll: float | None,
    ) -> None:
        self.attempt_counts[(state, direction, decision)] += 1
        if decision == "allowed":
            self.allowed_attempt_count += 1
        else:
            self.blocked_attempt_count += 1
        if len(self.attempt_log) >= self.MAX_LOG_ENTRIES:
            return
        self.attempt_log.append(
            {
                "event_step": int(self.current_event_step),
                "actor_thread_id": int(actor.threadID),
                "actor_position": int(actor.current_position[0]),
                "target_thread_id": int(target.threadID),
                "target_position": int(target.current_position[0]),
                "target_value": int(target.value),
                "state": str(state),
                "direction_from_actor": str(direction),
                "decision": str(decision),
                "roll": None if roll is None else float(roll),
                "behavior_label": self.behavior_label,
            }
        )

    def allow_target_swap(self, actor: Any, target: Any) -> bool:
        if not self.is_frozen_identity(target):
            return True
        state = self.state_by_thread.get(int(target.threadID), "stuck")
        direction = self._direction_from_actor(actor, target)
        roll: float | None = None
        if state == "passive":
            allowed = True
        elif state in {"stuck", "recovering_stuck"}:
            allowed = False
        elif state == "probabilistic":
            roll = self.rng.random()
            allowed = roll >= self.block_probability
        elif state == "directional_sticky":
            allowed = not self._direction_is_blocked(actor, target)
        else:
            allowed = False
        self._record_attempt(
            actor,
            target,
            state=state,
            direction=direction,
            decision="allowed" if allowed else "blocked",
            roll=roll,
        )
        return bool(allowed)

    def after_allowed_target_swap(self, target: Any) -> None:
        if not self.is_frozen_identity(target):
            return
        if self.behavior_type != "fatigue_recovery":
            return
        thread_id = int(target.threadID)
        if self.state_by_thread.get(thread_id) != "passive":
            return
        self.fatigue_by_thread[thread_id] = self.fatigue_by_thread.get(thread_id, 0) + 1
        if self.fatigue_by_thread[thread_id] >= self.fatigue_threshold:
            self.recover_until_by_thread[thread_id] = int(self.current_event_step + self.recovery_events)
            self._set_state(target, "recovering_stuck", "fatigue_threshold_reached")

    def state_signature(self) -> tuple[Any, ...]:
        return (
            tuple(sorted(self.state_by_thread.items())),
            tuple(sorted(self.fatigue_by_thread.items())),
            tuple(sorted(self.recover_until_by_thread.items())),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "behavior_type": self.behavior_type,
            "behavior_label": self.behavior_label,
            "seed": self.seed,
            "transition_count": int(self.transition_count),
            "transition_log_truncated": bool(self.transition_count > len(self.transition_log)),
            "attempt_count": int(self.allowed_attempt_count + self.blocked_attempt_count),
            "allowed_attempt_count": int(self.allowed_attempt_count),
            "blocked_attempt_count": int(self.blocked_attempt_count),
            "attempt_log_truncated": bool(sum(self.attempt_counts.values()) > len(self.attempt_log)),
            "attempt_counts": {
                f"{state}|{direction}|{decision}": int(count)
                for (state, direction, decision), count in sorted(self.attempt_counts.items())
            },
            "final_state_by_thread": {str(thread_id): state for thread_id, state in sorted(self.state_by_thread.items())},
            "fatigue_by_thread": {str(thread_id): int(value) for thread_id, value in sorted(self.fatigue_by_thread.items())},
            "recover_until_by_thread": {str(thread_id): int(value) for thread_id, value in sorted(self.recover_until_by_thread.items())},
            "parameters": self.behavior,
        }


def _patch_dynamic_frozen_swap(cells: Sequence[Any], controller: FrozenBehaviorController) -> None:
    def count_frozen_attempt_once(cell: Any) -> None:
        if not cell.tried_to_swap_with_frozen:
            cell.status_probe.count_frozen_cell_attempt()
            cell.tried_to_swap_with_frozen = True

    def make_swap(cell: Any) -> Any:
        original_swap = cell.swap

        def dynamic_swap(target_position: tuple[int, int], skip_stats: bool = False) -> Any:
            if controller.is_frozen_identity(cell) or cell.status == controller.cell_status.FREEZE:
                count_frozen_attempt_once(cell)
                return None
            target = cell.cells[int(target_position[0])]
            if controller.is_frozen_identity(target) and not controller.allow_target_swap(cell, target):
                count_frozen_attempt_once(cell)
                return None
            cell.tried_to_swap_with_frozen = False
            target.tried_to_swap_with_frozen = False
            original_target_thread_id = int(target.threadID)
            result = original_swap(target_position, skip_stats)
            if original_target_thread_id in controller.frozen_thread_ids:
                controller.after_allowed_target_swap(target)
            return result

        return dynamic_swap

    for cell in cells:
        cell.swap = make_swap(cell)


def _build_cells(config: SimulatorConfig, probe: EventTracingStatusProbe) -> tuple[list[Any], Any]:
    modules = import_cell_modules(config.repo_dir)
    cls_by_behavior = {
        "bubble": modules["BubbleSortCell"],
        "insertion": modules["InsertionSortCell"],
        "selection": modules["SelectionSortCell"],
    }
    labels = _labels_for_config(config)
    reverse_flags = _reverse_flags_for_config(config, labels)
    lock = threading.RLock()
    left_boundary = (0, 1)
    right_boundary = (len(config.values) - 1, 1)
    cells: list[Any] = []
    for idx, (value, label, reverse_direction) in enumerate(zip(config.values, labels, reverse_flags, strict=True)):
        behavior = _behavior_for_label(config, label)
        cls = cls_by_behavior[behavior]
        cell = cls(
            idx + 1,
            int(value),
            lock,
            (idx, 1),
            cells,
            left_boundary,
            right_boundary,
            probe,
            disable_visualization=True,
            label=str(label),
            reverse_direction=bool(reverse_direction),
        )
        cells.append(cell)
    group = modules["CellGroup"](
        cells,
        cells,
        0,
        left_boundary,
        right_boundary,
        modules["GroupStatus"].ACTIVE,
        lock,
        100_000_000,
        100_000_000,
    )
    for cell in cells:
        cell.group = group
    for idx in config.frozen_indices:
        cells[idx].set_cell_to_freeze()
    if config.frozen_semantics == "stuck":
        _patch_stuck_swap(cells, {cells[idx].threadID for idx in config.frozen_indices})
    return cells, modules["CellStatus"]


def _target_is_movable(
    actor: Any,
    target: Any,
    semantics: str,
    cell_status: Any,
    frozen_behavior_controller: FrozenBehaviorController | None = None,
) -> bool:
    if frozen_behavior_controller is not None:
        return frozen_behavior_controller.target_potentially_movable(actor, target)
    return target.status == cell_status.ACTIVE or (semantics == "passive" and target.status == cell_status.FREEZE)


def _insertion_enable_flags(cells: Sequence[Any], cell_status: Any, reverse: bool) -> tuple[bool, ...]:
    """Vectorized equivalent of public InsertionSortCell.is_enable_to_move()."""

    flags: list[bool] = []
    enabled = True
    prev = 100000 if reverse else -1
    for cell in cells:
        flags.append(enabled)
        if cell.status == cell_status.FREEZE:
            # The public method resets to -1 even for reverse-direction cells.
            prev = -1
            enabled = True
            continue
        if reverse:
            if cell.value > prev:
                enabled = False
        else:
            if cell.value < prev:
                enabled = False
        prev = cell.value
    return tuple(flags)


def has_public_legal_action(
    cells: Sequence[Any],
    cell_status: Any,
    label_to_behavior: Mapping[str, str] | None = None,
    frozen_semantics: str = "none",
    frozen_behavior_controller: FrozenBehaviorController | None = None,
) -> bool:
    """Return whether some active cell could swap or update local target state."""

    behavior_map = label_to_behavior or DEFAULT_LABEL_TO_BEHAVIOR
    insertion_enable_cache: dict[bool, tuple[bool, ...]] = {}
    for cell in cells:
        if cell.status != cell_status.ACTIVE:
            continue
        label = str(cell.label)
        behavior = behavior_map.get(label, label if label in ALGORITHMS else str(cell.cell_type).lower())
        current_idx = int(cell.current_position[0])
        reverse = bool(cell.reverse_direction)
        if behavior == "bubble":
            left_idx = current_idx - 1
            right_idx = current_idx + 1
            if left_idx >= 0 and _target_is_movable(cell, cells[left_idx], frozen_semantics, cell_status, frozen_behavior_controller):
                if (reverse and cell.value > cells[left_idx].value) or ((not reverse) and cell.value < cells[left_idx].value):
                    return True
            if right_idx < len(cells) and _target_is_movable(cell, cells[right_idx], frozen_semantics, cell_status, frozen_behavior_controller):
                if (reverse and cell.value < cells[right_idx].value) or ((not reverse) and cell.value > cells[right_idx].value):
                    return True
        elif behavior == "insertion":
            left_idx = current_idx - 1
            if left_idx < 0:
                continue
            if not _target_is_movable(cell, cells[left_idx], frozen_semantics, cell_status, frozen_behavior_controller):
                continue
            if reverse not in insertion_enable_cache:
                insertion_enable_cache[reverse] = _insertion_enable_flags(cells, cell_status, reverse)
            if not insertion_enable_cache[reverse][current_idx]:
                continue
            if (reverse and cell.value > cells[left_idx].value) or ((not reverse) and cell.value < cells[left_idx].value):
                return True
        elif behavior == "selection":
            if cell.current_position == cell.ideal_position:
                continue
            target_idx = int(cell.ideal_position[0])
            if 0 <= target_idx < len(cells) and _target_is_movable(cell, cells[target_idx], frozen_semantics, cell_status, frozen_behavior_controller):
                return True
        else:
            raise ValueError(f"Unsupported behavior: {behavior}")
    return False


def _active_positions(cells: Sequence[Any], cell_status: Any) -> list[int]:
    return [idx for idx, cell in enumerate(cells) if cell.status == cell_status.ACTIVE]


def _choose_active_position(
    active_positions: Sequence[int],
    distribution: str,
    rng: random.Random,
    cursor: int,
) -> tuple[int, int]:
    if not active_positions:
        raise ValueError("No active positions to choose from.")
    ordered = sorted(active_positions)
    if distribution == "uniform_active":
        return int(rng.choice(ordered)), cursor
    if distribution == "left_to_right_active":
        for pos in ordered:
            if pos >= cursor:
                return int(pos), int(pos + 1)
        return int(ordered[0]), int(ordered[0] + 1)
    if distribution == "right_to_left_active":
        descending = sorted(active_positions, reverse=True)
        effective_cursor = cursor if cursor >= 0 else descending[0]
        for pos in descending:
            if pos <= effective_cursor:
                return int(pos), int(pos - 1)
        return int(descending[0]), int(descending[0] - 1)
    raise ValueError(f"Unsupported activation distribution: {distribution}")


def _labels_from_snapshot(snapshot: Sequence[Sequence[Any]]) -> tuple[str, ...]:
    return tuple(str(item[1]) for item in snapshot)


def _make_metric_record(
    values: Sequence[int],
    labels: Sequence[str],
    event_step: int,
    swap_step: int,
    comparison_count: int,
    frozen_attempt_count: int,
    direction: str,
) -> dict[str, Any]:
    values_tuple = tuple(map(int, values))
    labels_tuple = tuple(map(str, labels))
    return {
        "event_step": int(event_step),
        "swap_step": int(swap_step),
        "comparison_count": int(comparison_count),
        "frozen_attempt_count": int(frozen_attempt_count),
        "values": list(values_tuple),
        "labels": list(labels_tuple),
        "sortedness_percent": float(sortedness_percent(values_tuple, direction)),
        "monotonicity_error_count": int(monotonicity_error_count(values_tuple, direction)),
        "aggregation_left_neighbor_percent": float(aggregation_left_neighbor_percent(labels_tuple)),
        "aggregation_right_neighbor_legacy_percent": float(aggregation_right_neighbor_legacy_percent(labels_tuple)),
        "algotype_positions_code": label_code(labels_tuple),
        "values_sha256": stable_json_sha256(list(values_tuple)),
        "labels_sha256": stable_json_sha256(list(labels_tuple)),
    }


def _records_from_probe(
    config: SimulatorConfig,
    probe: EventTracingStatusProbe,
    initial_values: Sequence[int],
    initial_labels: Sequence[str],
    final_values: Sequence[int],
    final_labels: Sequence[str],
    event_count: int,
) -> tuple[dict[str, Any], ...]:
    records = [
        _make_metric_record(
            initial_values,
            initial_labels,
            0,
            0,
            0,
            0,
            config.sort_direction,
        )
    ]
    lengths = {
        len(probe.sorting_steps),
        len(probe.cell_types),
        len(probe.comparison_counts_at_step),
        len(probe.frozen_attempt_counts_at_step),
        len(probe.swap_counts_at_step),
        len(probe.event_steps_at_step),
    }
    if len(lengths) != 1:
        raise RuntimeError("StatusProbe snapshot vectors diverged.")
    for values_snapshot, type_snapshot, comparison_count, frozen_attempt_count, swap_count, event_step in zip(
        probe.sorting_steps,
        probe.cell_types,
        probe.comparison_counts_at_step,
        probe.frozen_attempt_counts_at_step,
        probe.swap_counts_at_step,
        probe.event_steps_at_step,
        strict=True,
    ):
        records.append(
            _make_metric_record(
                values_snapshot,
                _labels_from_snapshot(type_snapshot),
                int(event_step),
                int(swap_count),
                int(comparison_count),
                int(frozen_attempt_count),
                config.sort_direction,
            )
        )
    if records[-1]["values"] != list(final_values) or records[-1]["labels"] != list(final_labels):
        records.append(
            _make_metric_record(
                final_values,
                final_labels,
                event_count,
                int(probe.swap_count),
                int(probe.compare_and_swap_count),
                int(probe.frozen_swap_attempts),
                config.sort_direction,
            )
        )
    else:
        records[-1]["comparison_count"] = int(probe.compare_and_swap_count)
        records[-1]["frozen_attempt_count"] = int(probe.frozen_swap_attempts)
    return tuple(records)


def simulate(config: SimulatorConfig) -> SimulatorResult:
    """Run one deterministic single-event cell-view simulation."""

    _validate_config(config)
    probe = EventTracingStatusProbe()
    cells, cell_status = _build_cells(config, probe)
    frozen_behavior_controller: FrozenBehaviorController | None = None
    if config.frozen_behavior is not None:
        frozen_behavior_controller = FrozenBehaviorController(
            cells,
            cell_status,
            config.frozen_behavior,
            seed=int(config.policy_seed if config.policy_seed is not None else config.activation_seed),
        )
        _patch_dynamic_frozen_swap(cells, frozen_behavior_controller)
    initial_values = cell_values(cells)
    initial_labels = cell_labels(cells)
    activation_rng = random.Random(config.activation_seed)
    random.seed(config.policy_seed if config.policy_seed is not None else config.activation_seed)
    activation_log: list[dict[str, Any]] = []
    event_count = 0
    no_progress_events = 0
    cached_legal_signature: tuple[Any, ...] | None = None
    cached_legal_action_exists: bool | None = None
    cursor = 0 if config.activation_distribution != "right_to_left_active" else len(cells) - 1
    stop_reason = "max_events_exceeded"
    max_guard_hit = False

    while event_count < config.max_events:
        if frozen_behavior_controller is not None:
            frozen_behavior_controller.before_event(event_count + 1, cells)
        current_values = cell_values(cells)
        if config.stop_when_sorted and is_sorted(current_values, config.sort_direction):
            stop_reason = "sorted"
            break
        if config.max_successful_swaps is not None and probe.swap_count >= config.max_successful_swaps:
            stop_reason = "max_successful_swaps_exceeded"
            max_guard_hit = True
            break
        active_positions = _active_positions(cells, cell_status)
        if not active_positions:
            stop_reason = "no_active_cells"
            break
        position, cursor = _choose_active_position(
            active_positions,
            config.activation_distribution,
            activation_rng,
            cursor,
        )
        cell = cells[position]
        before_signature = cell_state_signature(cells)
        before_swap_count = int(probe.swap_count)
        before_comparison_count = int(probe.compare_and_swap_count)
        event_step = event_count + 1
        probe.current_event_step = event_step
        cell.move()
        after_signature = cell_state_signature(cells)
        after_swap_count = int(probe.swap_count)
        state_changed = after_signature != before_signature
        made_progress = after_swap_count != before_swap_count or state_changed
        legal_action_exists = True
        if not made_progress:
            legal_signature: tuple[Any, ...] = after_signature
            if frozen_behavior_controller is not None:
                legal_signature = (*after_signature, ("frozen_behavior", frozen_behavior_controller.state_signature()))
            if cached_legal_signature == legal_signature and cached_legal_action_exists is not None:
                legal_action_exists = cached_legal_action_exists
            else:
                legal_action_exists = has_public_legal_action(
                    cells,
                    cell_status,
                    config.label_to_behavior,
                    config.frozen_semantics,
                    frozen_behavior_controller,
                )
                cached_legal_signature = legal_signature
                cached_legal_action_exists = legal_action_exists
        else:
            cached_legal_signature = None
            cached_legal_action_exists = None
        activation_log.append(
            {
                "event_step": event_step,
                "activated_position": int(position),
                "activated_thread_id": int(cell.threadID),
                "activated_label": str(cell.label),
                "activated_behavior": _behavior_for_label(config, str(cell.label)),
                "activated_value_before": int(before_signature[position][1]),
                "swap_count_before": before_swap_count,
                "swap_count_after": after_swap_count,
                "comparison_count_before": before_comparison_count,
                "comparison_count_after": int(probe.compare_and_swap_count),
                "state_changed": bool(state_changed),
                "legal_action_existed_before_event": bool(legal_action_exists),
            }
        )
        event_count = event_step
        if not made_progress and not legal_action_exists:
            no_progress_events += 1
            if no_progress_events >= config.stall_events:
                stop_reason = "no_legal_action_window"
                break
        else:
            no_progress_events = 0
    else:
        max_guard_hit = True

    final_values = cell_values(cells)
    final_labels = cell_labels(cells)
    if config.stop_when_sorted and is_sorted(final_values, config.sort_direction):
        stop_reason = "sorted"
        max_guard_hit = False
    records = _records_from_probe(
        config,
        probe,
        initial_values,
        initial_labels,
        final_values,
        final_labels,
        event_count,
    )
    return SimulatorResult(
        config=config,
        records=records,
        activation_log=tuple(activation_log),
        final_values=tuple(final_values),
        final_labels=tuple(final_labels),
        final_frozen_positions=frozen_positions(cells, cell_status),
        final_frozen_values=frozen_values(cells, cell_status),
        stop_reason=stop_reason,
        max_guard_hit=bool(max_guard_hit),
        event_count=int(event_count),
        swap_count=int(probe.swap_count),
        comparison_count=int(probe.compare_and_swap_count),
        frozen_attempt_count=int(probe.frozen_swap_attempts),
        compare_plus_swap_count=int(probe.swap_count + probe.compare_and_swap_count),
        frozen_behavior_transition_log=tuple(frozen_behavior_controller.transition_log) if frozen_behavior_controller is not None else (),
        frozen_behavior_attempt_log=tuple(frozen_behavior_controller.attempt_log) if frozen_behavior_controller is not None else (),
        frozen_behavior_summary=frozen_behavior_controller.summary() if frozen_behavior_controller is not None else {},
    )


def result_summary(result: SimulatorResult) -> dict[str, Any]:
    """Return a compact JSON-serializable run summary."""

    return {
        "wrapper_name": result.wrapper_name,
        "algorithm": result.config.algorithm,
        "algotypes": list(result.config.algotypes) if result.config.algotypes is not None else None,
        "activation_seed": int(result.config.activation_seed),
        "policy_seed": int(result.config.policy_seed if result.config.policy_seed is not None else result.config.activation_seed),
        "activation_distribution": result.config.activation_distribution,
        "frozen_semantics": result.config.frozen_semantics,
        "frozen_behavior": dict(result.config.frozen_behavior) if result.config.frozen_behavior is not None else None,
        "frozen_indices": list(result.config.frozen_indices),
        "stop_reason": result.stop_reason,
        "max_guard_hit": bool(result.max_guard_hit),
        "event_count": int(result.event_count),
        "swap_count": int(result.swap_count),
        "comparison_count": int(result.comparison_count),
        "frozen_attempt_count": int(result.frozen_attempt_count),
        "compare_plus_swap_count": int(result.compare_plus_swap_count),
        "final_values": list(result.final_values),
        "final_labels": list(result.final_labels),
        "final_frozen_positions": list(result.final_frozen_positions),
        "final_sortedness_percent": sortedness_percent(result.final_values, result.config.sort_direction),
        "final_monotonicity_error_count": monotonicity_error_count(result.final_values, result.config.sort_direction),
        "trace_record_count": len(result.records),
        "activation_event_count": len(result.activation_log),
        "trace_sha256": stable_json_sha256(result.records),
        "activation_log_sha256": stable_json_sha256(result.activation_log),
        "frozen_behavior_summary": dict(result.frozen_behavior_summary),
        "frozen_behavior_transition_log_sha256": stable_json_sha256(result.frozen_behavior_transition_log),
        "frozen_behavior_attempt_log_sha256": stable_json_sha256(result.frozen_behavior_attempt_log),
    }


def trace_rows(result: SimulatorResult, metadata: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Convert metric records to E01-compatible long trace rows."""

    meta = dict(metadata or {})
    rows: list[dict[str, Any]] = []
    initial_hash = stable_json_sha256(list(result.config.values))
    final_hash = stable_json_sha256(list(result.final_values))
    initial_counts = count_json(_labels_for_config(result.config))
    final_counts = count_json(result.final_labels)
    last_idx = len(result.records) - 1
    for idx, record in enumerate(result.records):
        row = {
            **meta,
            "mode": "cell_view",
            "algorithm": result.config.algorithm or "mixed",
            "baseline_source": "public_repository_cell_view_classes",
            "scheduler": "deterministic_single_event",
            "activation_distribution": result.config.activation_distribution,
            "initial_array_seed": None,
            "scheduler_seed": int(result.config.activation_seed),
            "policy_seed": int(result.config.policy_seed if result.config.policy_seed is not None else result.config.activation_seed),
            "frozen_semantics": result.config.frozen_semantics,
            "frozen_behavior_json": json.dumps(result.config.frozen_behavior, sort_keys=True, separators=(",", ":"), default=str)
            if result.config.frozen_behavior is not None
            else None,
            "frozen_count": len(result.config.frozen_indices),
            "initial_frozen_indices_json": json.dumps(list(result.config.frozen_indices), separators=(",", ":")),
            "initial_algotype_counts_json": initial_counts,
            "final_algotype_counts_json": final_counts,
            "event_step": int(record["event_step"]),
            "swap_step": int(record["swap_step"]),
            "comparison_count_at_step": int(record["comparison_count"]),
            "frozen_attempt_count_at_step": int(record["frozen_attempt_count"]),
            "sortedness_percent": float(record["sortedness_percent"]),
            "monotonicity_error_count": int(record["monotonicity_error_count"]),
            "aggregation_left_neighbor_percent": float(record["aggregation_left_neighbor_percent"]),
            "aggregation_right_neighbor_legacy_percent": float(record["aggregation_right_neighbor_legacy_percent"]),
            "algotype_positions_code": record["algotype_positions_code"],
            "is_initial": idx == 0,
            "is_final": idx == last_idx,
            "stop_reason": result.stop_reason,
            "max_guard_hit": bool(result.max_guard_hit),
            "final_values_json": json.dumps(list(result.final_values), separators=(",", ":")) if idx == last_idx else None,
            "initial_array_sha256": initial_hash,
            "final_array_sha256": final_hash,
        }
        rows.append(row)
    return rows
