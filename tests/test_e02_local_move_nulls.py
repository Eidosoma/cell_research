from __future__ import annotations

import unittest
from collections import Counter

from scripts.e02_s07_local_move_nulls import (
    SourceRun,
    labels_from_seed,
    run_local_move_null,
    sha256_json,
    sortedness_percent,
    state_hash,
)


def unit_source(
    *,
    frozen_variant: str = "none",
    initial_frozen_positions: list[int] | None = None,
    swap_count: int = 6,
) -> SourceRun:
    values = [4, 1, 3, 2]
    labels = ["bubble", "insertion", "bubble", "selection"]
    frozen_positions = [] if initial_frozen_positions is None else initial_frozen_positions
    return SourceRun(
        source_run_id=f"unit_{frozen_variant}",
        source_context="unit",
        source_research_step_id="unit",
        source_condition_id="unit_condition",
        mixture_id="bubble_insertion_selection",
        algorithm="unit",
        label_source="unit_labels",
        n=len(values),
        replicate_index=0,
        replicate_number=1,
        input_permutation_seed=1,
        algotype_assignment_seed=2,
        scheduler_seed=3,
        tie_breaker_seed=4,
        frozen_position_seed=5 if frozen_positions else None,
        frozen_variant=frozen_variant,
        frozen_count=len(frozen_positions),
        initial_frozen_positions=frozen_positions,
        final_frozen_positions=frozen_positions,
        initial_values=values,
        initial_policy_algotypes=labels,
        initial_labels_by_cell_id=labels,
        final_values=sorted(values),
        final_labels_by_position=labels,
        completed=True,
        stop_reason="sorted",
        swap_count=swap_count,
        comparison_count=10,
        archived_compare_and_swap_count=10,
        activation_count=20,
        event_count=swap_count + 1,
        real_initial_aggregation=0.0,
        real_peak_aggregation=0.5,
        real_final_aggregation=0.5,
        real_auc_aggregation=0.25,
        real_final_sortedness_percent=100.0,
        real_final_monotonicity_error=0,
        real_dg_max_drop_percent=0.0,
        real_dg_decrease_count=0,
        real_path_curvature_sign_changes=0,
        real_sortedness_total_variation=50.0,
        source_trajectory_hash="unit_hash",
        wall_time_seconds=0.0,
    )


class TestE02LocalMoveNulls(unittest.TestCase):
    def test_random_adjacent_matches_budget_and_preserves_counts(self) -> None:
        source = unit_source(swap_count=8)
        run = run_local_move_null(
            source=source,
            null_model="random_adjacent",
            null_replicate_index=0,
            null_seed=11,
            attempt_multiplier=20,
            metropolis_temperature=2.0,
            metropolis_min_acceptance=0.1,
        )
        self.assertTrue(run.completed)
        self.assertEqual(run.accepted_swap_count, source.swap_count)
        self.assertEqual(run.local_move_violations, 0)
        self.assertEqual(run.invalid_frozen_move_count, 0)
        self.assertEqual(Counter(run.final_values), Counter(source.initial_values))
        self.assertEqual(Counter(run.final_labels_by_position), Counter(source.initial_labels_by_cell_id))
        self.assertTrue(run.matched_initial_values)
        self.assertTrue(run.matched_initial_labels)

    def test_stuck_frozen_cell_never_moves(self) -> None:
        source = unit_source(frozen_variant="stuck", initial_frozen_positions=[1], swap_count=10)
        run = run_local_move_null(
            source=source,
            null_model="random_adjacent",
            null_replicate_index=0,
            null_seed=12,
            attempt_multiplier=50,
            metropolis_temperature=2.0,
            metropolis_min_acceptance=0.1,
        )
        self.assertTrue(run.completed)
        self.assertEqual(run.invalid_frozen_move_count, 0)
        for row in run.trace_rows:
            self.assertEqual(row["frozenPositions"], "[1]")

    def test_passive_frozen_cell_can_be_moved_by_neighbor(self) -> None:
        source = unit_source(frozen_variant="passive", initial_frozen_positions=[1], swap_count=1)
        run = run_local_move_null(
            source=source,
            null_model="random_adjacent",
            null_replicate_index=0,
            null_seed=1,
            attempt_multiplier=10,
            metropolis_temperature=2.0,
            metropolis_min_acceptance=0.1,
        )
        self.assertTrue(run.completed)
        self.assertEqual(run.invalid_frozen_move_count, 0)
        self.assertEqual(run.accepted_swap_count, 1)
        self.assertEqual(len(run.final_frozen_positions), 1)

    def test_metropolis_local_respects_budget_and_trace_distance(self) -> None:
        source = unit_source(swap_count=5)
        run = run_local_move_null(
            source=source,
            null_model="metropolis_local",
            null_replicate_index=0,
            null_seed=13,
            attempt_multiplier=100,
            metropolis_temperature=2.0,
            metropolis_min_acceptance=0.2,
        )
        self.assertTrue(run.completed)
        self.assertEqual(run.accepted_swap_count, 5)
        self.assertTrue(all(row["localMoveDistance"] in (None, 1) for row in run.trace_rows))

    def test_diagnostic_labels_are_seeded_and_count_preserving(self) -> None:
        first = labels_from_seed("bubble_insertion_selection", 30, 123)
        second = labels_from_seed("bubble_insertion_selection", 30, 123)
        self.assertEqual(first, second)
        self.assertEqual(Counter(first), {"bubble": 10, "insertion": 10, "selection": 10})
        self.assertEqual(sha256_json(first), sha256_json(second))
        self.assertEqual(state_hash([3, 1, 2]), state_hash([3, 1, 2]))
        self.assertAlmostEqual(sortedness_percent([1, 2, 3]), 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
