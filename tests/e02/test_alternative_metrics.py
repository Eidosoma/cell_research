"""Focused tests for E02 S09 alternative distance metrics."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from scripts.e02_s09_alternative_metrics import (
    alternative_metrics,
    edit_distance_to_target,
    earth_mover_position_distance,
    inversion_count,
    kendall_tau_distance,
    metric_run_rows,
    metric_sanity_checks,
    occurrence_tokens,
    scalar_path_curvature,
    spearman_position_distance,
    state_space_curvature,
    summarize_metric_results,
)


class AlternativeMetricTests(unittest.TestCase):
    def test_metric_sanity_checks_pass(self) -> None:
        result = metric_sanity_checks()
        self.assertTrue(result["metricSanityChecksPassed"])
        self.assertTrue(result["duplicateHandlingValidationPassed"])

    def test_sorted_and_reversed_unique_arrays(self) -> None:
        sorted_metrics = alternative_metrics([1, 2, 3])
        self.assertEqual(sorted_metrics["inversion_count"], 0)
        self.assertEqual(sorted_metrics["kendall_tau_distance"], 0.0)
        self.assertEqual(sorted_metrics["spearman_position_distance"], 0.0)
        self.assertEqual(sorted_metrics["earth_mover_position_distance"], 0.0)
        self.assertEqual(sorted_metrics["edit_distance_to_target"], 0.0)

        reversed_metrics = alternative_metrics([3, 2, 1])
        self.assertEqual(reversed_metrics["inversion_count"], 3)
        self.assertEqual(reversed_metrics["kendall_tau_distance"], 1.0)
        self.assertEqual(reversed_metrics["spearman_position_distance"], 1.0)
        self.assertEqual(reversed_metrics["earth_mover_position_distance"], 1.0)
        self.assertAlmostEqual(reversed_metrics["edit_distance_to_target"], 2.0 / 3.0)

    def test_duplicate_ties_are_explicit(self) -> None:
        values = [2, 1, 2, 1]
        self.assertEqual(occurrence_tokens(values), ((2, 0), (1, 0), (2, 1), (1, 1)))
        self.assertEqual(inversion_count(values), 3)
        self.assertAlmostEqual(kendall_tau_distance(values), 0.75)
        self.assertAlmostEqual(earth_mover_position_distance(values), 0.75)

        all_ties = [1, 1, 1]
        self.assertEqual(inversion_count(all_ties), 0)
        self.assertEqual(kendall_tau_distance(all_ties), 0.0)
        self.assertEqual(edit_distance_to_target(all_ties), 0.0)

    def test_directional_metrics_support_decreasing_targets(self) -> None:
        self.assertEqual(inversion_count([3, 2, 1], direction="decreasing"), 0)
        self.assertEqual(kendall_tau_distance([3, 2, 1], direction="decreasing"), 0.0)
        self.assertEqual(spearman_position_distance([3, 2, 1], direction="decreasing"), 0.0)
        self.assertEqual(earth_mover_position_distance([3, 2, 1], direction="decreasing"), 0.0)

    def test_curvature_metrics(self) -> None:
        self.assertTrue(math.isclose(scalar_path_curvature([1.0, 0.5, 0.0]), 1.0))
        self.assertGreater(scalar_path_curvature([1.0, 0.2, 0.8, 0.0]), 1.0)
        self.assertGreater(state_space_curvature([[0, 0], [1, 0], [0, 0], [2, 0]]), 1.0)

    def test_repair_metric_summary_classifies_synthetic_trace(self) -> None:
        trace = pd.DataFrame(
            [
                {
                    "condition_id": "T",
                    "repeat_index": 0,
                    "scheduler_regime": "deterministic_single_event_uniform_active",
                    "run_family": "synthetic",
                    "algorithm": "bubble",
                    "algotype_mix": "bubble",
                    "value_distribution": "unique",
                    "frozen_semantics": "none",
                    "frozen_count": 0,
                    "trace_record_index": idx,
                    "duplicate_handling_policy": "test",
                    "sortedness_adjacency_distance": sortedness_distance,
                    "kendall_tau_distance": kendall,
                    "inversion_fraction": kendall,
                    "spearman_position_distance": spearman,
                    "edit_distance_to_target": edit,
                    "earth_mover_position_distance": emd,
                    "inversion_count": inv,
                    "stop_reason": "sorted",
                    "max_guard_hit": False,
                    "initial_array_sha256": "a",
                    "final_array_sha256": "b",
                    "target_values_sha256": "c",
                }
                for idx, (sortedness_distance, kendall, spearman, edit, emd, inv) in enumerate(
                    [(0.5, 1.0, 1.0, 0.5, 1.0, 3), (0.0, 0.0, 0.0, 0.0, 0.0, 0)]
                )
            ]
        )
        run_df = pd.DataFrame(
            [
                {
                    "condition_id": "T",
                    "repeat_index": 0,
                    "scheduler_regime": "deterministic_single_event_uniform_active",
                    "value_state_path_curvature_ratio": 1.0,
                }
            ]
        )
        metric_df = metric_run_rows(trace, run_df)
        self.assertIn("value_state_space_curvature", set(metric_df["metric_name"]))
        summary, classified = summarize_metric_results(metric_df)
        self.assertFalse(summary.empty)
        alt_classes = set(classified[classified["metric_name"] == "kendall_tau_distance"]["metric_stability_class"])
        self.assertEqual(alt_classes, {"metric_stable"})


if __name__ == "__main__":
    unittest.main()
