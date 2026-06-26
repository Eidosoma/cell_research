"""Single-event asynchronous simulator for cell-view sorting policies.

The implementation mirrors the policy logic in the original threaded
BubbleSortCell, InsertionSortCell, and SelectionSortCell classes, but replaces
thread scheduling with explicit seeded activation of one cell identity at a
time. Trace rows keep the E01 core schema fields and add scheduler diagnostics.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import aggregation, monotonicity_error, sortedness_percent, sortedness_raw, state_hash


ALGOTYPES = {"bubble", "insertion", "selection"}
FROZEN_VARIANTS = {"none", "passive", "stuck"}


@dataclass
class Cell:
    cell_id: int
    value: int
    algotype: str
    label: int = 0
    reverse_direction: bool = False
    frozen: bool = False
    ideal_position: int | None = None
    tried_to_swap_with_frozen: bool = False


@dataclass
class StepOutcome:
    activated: bool
    actor_cell_id: int | None
    actor_position_before: int | None
    actor_position_after: int | None
    target_position: int | None
    swapped: bool
    comparison_delta: int
    archived_compare_delta: int
    blocked_move_attempt: bool
    reason: str


@dataclass
class SimulationResult:
    condition_id: str
    implementation: str
    algorithm: str
    completed: bool
    stop_reason: str
    initial_values: list[int]
    final_values: list[int]
    initial_algotypes: list[str]
    final_algotypes: list[str]
    initial_frozen_positions: list[int]
    final_frozen_positions: list[int]
    swap_count: int
    comparison_count: int
    archived_compare_and_swap_count: int
    blocked_move_attempts: int
    frozen_swap_attempts: int
    activation_count: int
    event_count: int
    final_sortedness_raw_count: int
    final_sortedness_percent: float
    final_monotonicity_error: int
    final_aggregation: float
    wall_time_seconds: float
    trace_rows: list[dict[str, Any]]

    def summary_record(
        self,
        *,
        research_step_id: str,
        replicate_index: int,
        replicate_number: int,
        input_permutation_seed: int | None,
        scheduler_seed: int,
        tie_breaker_seed: int,
        frozen_position_seed: int | None,
        input_profile: str,
        frozen_variant: str,
        frozen_count: int,
    ) -> dict[str, Any]:
        return {
            "research_step_id": research_step_id,
            "condition_id": self.condition_id,
            "implementation": self.implementation,
            "algorithm": self.algorithm,
            "replicate_index": int(replicate_index),
            "replicate_number": int(replicate_number),
            "input_profile": input_profile,
            "n": len(self.initial_values),
            "frozen_variant": frozen_variant,
            "frozen_count": int(frozen_count),
            "input_permutation_seed": input_permutation_seed,
            "scheduler_seed": int(scheduler_seed),
            "tie_breaker_seed": int(tie_breaker_seed),
            "frozen_position_seed": frozen_position_seed,
            "initial_state_hash": state_hash(self.initial_values),
            "final_state_hash": state_hash(self.final_values),
            "initial_values_json": json.dumps(self.initial_values, separators=(",", ":")),
            "final_values_json": json.dumps(self.final_values, separators=(",", ":")),
            "initial_algotypes_json": json.dumps(self.initial_algotypes, separators=(",", ":")),
            "final_algotypes_json": json.dumps(self.final_algotypes, separators=(",", ":")),
            "initial_frozen_positions_json": json.dumps(self.initial_frozen_positions, separators=(",", ":")),
            "final_frozen_positions_json": json.dumps(self.final_frozen_positions, separators=(",", ":")),
            "completed": bool(self.completed),
            "stop_reason": self.stop_reason,
            "swap_count": int(self.swap_count),
            "comparison_count": int(self.comparison_count),
            "archived_compare_and_swap_count": int(self.archived_compare_and_swap_count),
            "blocked_move_attempts": int(self.blocked_move_attempts),
            "frozen_swap_attempts": int(self.frozen_swap_attempts),
            "activation_count": int(self.activation_count),
            "event_count": int(self.event_count),
            "final_sortedness_raw_count": int(self.final_sortedness_raw_count),
            "final_sortedness_percent": float(self.final_sortedness_percent),
            "final_monotonicity_error": int(self.final_monotonicity_error),
            "final_aggregation": float(self.final_aggregation),
            "wall_time_seconds": float(self.wall_time_seconds),
        }


class DeterministicEventSimulator:
    """Reference simulator with explicit single-cell activation events."""

    def __init__(
        self,
        initial_values: Sequence[int],
        algotypes: str | Sequence[str],
        *,
        labels: Sequence[int] | None = None,
        reverse_directions: Sequence[bool] | None = None,
        frozen_positions: Iterable[int] = (),
        frozen_variant: str = "none",
        scheduler_seed: int = 0,
        tie_breaker_seed: int = 0,
        condition_id: str = "manual",
        implementation: str = "cell_view",
        research_step_id: str = "S01",
    ) -> None:
        if frozen_variant not in FROZEN_VARIANTS:
            raise ValueError(f"unknown frozen_variant: {frozen_variant}")
        self.initial_values = [int(value) for value in initial_values]
        if isinstance(algotypes, str):
            algotype_list = [algotypes] * len(self.initial_values)
        else:
            algotype_list = [str(value) for value in algotypes]
        if len(algotype_list) != len(self.initial_values):
            raise ValueError("algotypes length must match initial_values")
        unknown = sorted(set(algotype_list) - ALGOTYPES)
        if unknown:
            raise ValueError(f"unknown algotypes: {unknown}")
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
        self.cells: list[Cell] = []
        for pos, (value, algotype, label, reverse) in enumerate(
            zip(self.initial_values, algotype_list, label_list, reverse_list)
        ):
            ideal_position = None
            if algotype == "selection":
                ideal_position = n - 1 if reverse else 0
            self.cells.append(
                Cell(
                    cell_id=pos,
                    value=value,
                    algotype=algotype,
                    label=label,
                    reverse_direction=reverse,
                    frozen=pos in frozen_set,
                    ideal_position=ideal_position,
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

    def _count_frozen_attempt(self, actor_pos: int) -> None:
        actor = self.cells[actor_pos]
        if not actor.tried_to_swap_with_frozen:
            self.frozen_swap_attempts += 1
            actor.tried_to_swap_with_frozen = True

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

    def _original_style_move_opportunity(self, actor_pos: int) -> bool:
        actor = self.cells[actor_pos]
        if actor.frozen:
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
            if not self._insertion_prefix_enabled(actor_pos, actor.reverse_direction):
                return False
            if actor_pos == 0 or not self._target_is_active(actor_pos - 1):
                return False
            if actor.reverse_direction:
                return actor.value > self.cells[actor_pos - 1].value
            return actor.value < self.cells[actor_pos - 1].value
        if actor.algotype == "selection":
            ideal = actor.ideal_position
            return ideal is not None and self._target_in_bounds(ideal) and actor_pos != ideal
        raise ValueError(actor.algotype)

    def _insertion_prefix_enabled(self, actor_pos: int, reverse_direction: bool) -> bool:
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

    def _bubble_step(self, actor_pos: int, forced_direction: int | None) -> StepOutcome:
        actor = self.cells[actor_pos]
        archived_delta = 1 if self._original_style_move_opportunity(actor_pos) else 0
        self.archived_compare_and_swap_count += archived_delta

        if forced_direction is None:
            direction = 1 if self.tie_rng.random() < 0.5 else -1
        else:
            direction = 1 if forced_direction > 0 else -1
        target_pos = actor_pos + direction
        if not self._target_status_allows_policy_check(target_pos):
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, target_pos, False, 0, archived_delta, False, "target_oob")

        self.comparison_count += 1
        if actor.reverse_direction:
            should_swap = actor.value < self.cells[target_pos].value if direction == 1 else actor.value > self.cells[target_pos].value
        else:
            should_swap = actor.value > self.cells[target_pos].value if direction == 1 else actor.value < self.cells[target_pos].value
        if not should_swap:
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, target_pos, False, 1, archived_delta, False, "no_policy_swap")
        if not self._can_swap(actor_pos, target_pos):
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, target_pos, False, 1, archived_delta, True, "swap_blocked")
        self._swap(actor_pos, target_pos)
        return StepOutcome(True, actor.cell_id, actor_pos, target_pos, target_pos, True, 1, archived_delta, False, "swapped")

    def _insertion_step(self, actor_pos: int) -> StepOutcome:
        actor = self.cells[actor_pos]
        if not self._insertion_prefix_enabled(actor_pos, actor.reverse_direction):
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, None, False, 0, 0, False, "prefix_not_enabled")
        archived_delta = 1 if self._original_style_move_opportunity(actor_pos) else 0
        self.archived_compare_and_swap_count += archived_delta
        target_pos = actor_pos - 1
        if not self._target_status_allows_policy_check(target_pos):
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, target_pos, False, 0, archived_delta, False, "target_oob")
        self.comparison_count += 1
        should_swap = actor.value > self.cells[target_pos].value if actor.reverse_direction else actor.value < self.cells[target_pos].value
        if not should_swap:
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, target_pos, False, 1, archived_delta, False, "no_policy_swap")
        if not self._can_swap(actor_pos, target_pos):
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, target_pos, False, 1, archived_delta, True, "swap_blocked")
        self._swap(actor_pos, target_pos)
        return StepOutcome(True, actor.cell_id, actor_pos, target_pos, target_pos, True, 1, archived_delta, False, "swapped")

    def _selection_step(self, actor_pos: int) -> StepOutcome:
        actor = self.cells[actor_pos]
        archived_delta = 1 if self._original_style_move_opportunity(actor_pos) else 0
        self.archived_compare_and_swap_count += archived_delta
        ideal = actor.ideal_position
        if ideal is None or not self._target_in_bounds(ideal):
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, ideal, False, 0, archived_delta, False, "ideal_oob")
        if actor_pos == ideal:
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, ideal, False, 0, archived_delta, False, "already_at_ideal")

        self.comparison_count += 1
        target = self.cells[ideal]
        if target.frozen:
            actor.ideal_position = ideal - 1 if actor.reverse_direction else ideal + 1
            attempted_swap = actor.value < target.value
            if attempted_swap and self._can_swap(actor_pos, ideal):
                self._swap(actor_pos, ideal)
                return StepOutcome(True, actor.cell_id, actor_pos, ideal, ideal, True, 1, archived_delta, False, "swapped_frozen_target")
            blocked = attempted_swap and self.frozen_variant == "stuck" and target.frozen
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, ideal, False, 1, archived_delta, blocked, "frozen_target")

        if actor.value >= target.value:
            actor.ideal_position = ideal - 1 if actor.reverse_direction else ideal + 1
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, ideal, False, 1, archived_delta, False, "advance_ideal")
        if not self._can_swap(actor_pos, ideal):
            return StepOutcome(True, actor.cell_id, actor_pos, actor_pos, ideal, False, 1, archived_delta, True, "swap_blocked")
        self._swap(actor_pos, ideal)
        return StepOutcome(True, actor.cell_id, actor_pos, ideal, ideal, True, 1, archived_delta, False, "swapped")

    def step(
        self,
        *,
        forced_cell_id: int | None = None,
        forced_direction: int | None = None,
    ) -> StepOutcome:
        """Activate exactly one eligible cell identity."""
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
        if actor.algotype == "bubble":
            outcome = self._bubble_step(actor_pos, forced_direction)
        elif actor.algotype == "insertion":
            outcome = self._insertion_step(actor_pos)
        elif actor.algotype == "selection":
            outcome = self._selection_step(actor_pos)
        else:
            raise ValueError(actor.algotype)

        if outcome.swapped:
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
            if cell.algotype == "bubble":
                for direction in (-1, 1):
                    target = pos + direction
                    if not self._target_in_bounds(target):
                        continue
                    if cell.reverse_direction:
                        should_swap = cell.value < self.cells[target].value if direction == 1 else cell.value > self.cells[target].value
                    else:
                        should_swap = cell.value > self.cells[target].value if direction == 1 else cell.value < self.cells[target].value
                    if should_swap and (self.frozen_variant != "stuck" or not self.cells[target].frozen):
                        return True
            elif cell.algotype == "insertion":
                target = pos - 1
                if (
                    self._target_in_bounds(target)
                    and self._insertion_prefix_enabled(pos, cell.reverse_direction)
                    and (self.frozen_variant != "stuck" or not self.cells[target].frozen)
                ):
                    if cell.reverse_direction and cell.value > self.cells[target].value:
                        return True
                    if not cell.reverse_direction and cell.value < self.cells[target].value:
                        return True
            elif cell.algotype == "selection":
                ideal = cell.ideal_position
                if ideal is not None and self._target_in_bounds(ideal) and pos != ideal:
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
