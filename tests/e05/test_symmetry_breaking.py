"""E05 S10 symmetry-breaking benchmark tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.symmetry_breaking import (
    AXIS_CHOICES,
    DEFAULT_EVENT_MULTIPLIER,
    SymmetryInitialCondition,
    axis_choice_metrics,
    axis_label_symmetry_error,
    construct_axis_symmetric_initial_state,
    default_initial_conditions,
    default_symmetry_target,
    initial_condition_rows,
    policy_parameter_signature,
    run_symmetry_breaking_benchmark,
    simulate_symmetry_breaking_run,
    source_guardrail_rows,
)


class SymmetryBreakingTests(unittest.TestCase):
    def test_exact_axis_initial_state_has_zero_construction_axis_label_error(self) -> None:
        target = default_symmetry_target()
        condition = next(item for item in default_initial_conditions() if item.condition_id == "exact_horizontal_mirror")
        state = construct_axis_symmetric_initial_state(target, condition, 4101)

        self.assertEqual(axis_label_symmetry_error(target, state, condition.construction_axis), 0.0)
        self.assertGreater(target.target_error(state), 0.0)

    def test_near_axis_initial_state_breaks_but_preserves_low_symmetry_error(self) -> None:
        target = default_symmetry_target()
        exact = next(item for item in default_initial_conditions() if item.condition_id == "exact_vertical_mirror")
        near = next(item for item in default_initial_conditions() if item.condition_id == "near_vertical_mirror")
        exact_state = construct_axis_symmetric_initial_state(target, exact, 4102)
        near_state = construct_axis_symmetric_initial_state(target, near, 4102)
        exact_error = axis_label_symmetry_error(target, exact_state, "vertical")
        near_error = axis_label_symmetry_error(target, near_state, "vertical")

        self.assertEqual(exact_error, 0.0)
        self.assertGreater(near_error, exact_error)
        self.assertLessEqual(near_error, 0.25)

    def test_axis_choice_metrics_are_defined_for_vertical_and_horizontal_axes(self) -> None:
        target = default_symmetry_target()
        condition = next(item for item in default_initial_conditions() if item.condition_id == "exact_horizontal_mirror")
        state = construct_axis_symmetric_initial_state(target, condition, 4103)
        metrics = axis_choice_metrics(target, state, "initial")

        self.assertIn(metrics["chosen_axis"], AXIS_CHOICES)
        self.assertIn("vertical_orientation_error", metrics)
        self.assertIn("horizontal_orientation_error", metrics)
        self.assertGreaterEqual(metrics["axis_margin"], 0.0)

    def test_initial_condition_rows_validate_expected_conditions(self) -> None:
        rows = initial_condition_rows(default_symmetry_target(), default_initial_conditions(), seed=4101)

        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["organizer_cue_added"] is False for row in rows))
        self.assertTrue(all(row["initial_target_error"] > 0.0 for row in rows))
        exact_rows = [row for row in rows if row["symmetry_class"] == "exact_axis_mirror"]
        self.assertTrue(all(row["construction_axis_label_symmetry_error"] == 0.0 for row in exact_rows))

    def test_source_guardrails_do_not_add_organizer_or_axis_cues(self) -> None:
        rows = source_guardrail_rows()
        signatures = [
            policy_parameter_signature(row["policy_id"], DEFAULT_EVENT_MULTIPLIER)
            for row in rows
        ]

        self.assertTrue(rows)
        self.assertTrue(all(row["organizer_cue_added"] is False for row in rows))
        self.assertTrue(all(row["global_axis_observation_added"] is False for row in rows))
        self.assertTrue(all("organizer_cue_added" in signature for signature in signatures))

    def test_simulate_symmetry_breaking_run_records_axis_metrics_and_trace(self) -> None:
        target = default_symmetry_target()
        condition = SymmetryInitialCondition(
            "test_exact_horizontal",
            "horizontal",
            "exact_axis_mirror",
            0,
            "test condition",
        )
        result = simulate_symmetry_breaking_run(
            target=target,
            condition=condition,
            policy_id="s07_local_target_neighbor_descent",
            seed=4104,
            event_multiplier=5,
            records_per_run=4,
        )

        self.assertEqual(result.summary_row["research_step_id"], "S10")
        self.assertIn(result.summary_row["final_chosen_axis"], AXIS_CHOICES)
        self.assertIn("axis_switch_count", result.summary_row)
        self.assertTrue(result.trace_rows)
        self.assertEqual(result.summary_row["population_delta_total"], 0)
        self.assertEqual(len(result.metric_rows), 16)

    def test_compact_symmetry_breaking_benchmark_matrix(self) -> None:
        target = default_symmetry_target()
        conditions = default_initial_conditions()[:2]
        summaries, traces, metrics, _runs = run_symmetry_breaking_benchmark(
            target=target,
            conditions=conditions,
            policy_ids=("s07_local_target_neighbor_descent", "s07_random_adjacent_swap_control"),
            seeds=(4101, 4102),
            event_multiplier=3,
            records_per_run=2,
        )

        self.assertEqual(len(summaries), len(conditions) * 2 * 2)
        self.assertTrue(traces)
        self.assertEqual(len(metrics), len(summaries) * 2 * 8)
        self.assertTrue(all(row["organizer_cue_added"] is False for row in summaries))


if __name__ == "__main__":
    unittest.main()
