"""Focused tests for E02 S09 alternative distance metrics."""

from __future__ import annotations

import math
import unittest

from scripts.e02_s09_alternative_metrics import (
    alternative_metrics,
    edit_distance_to_target,
    earth_mover_position_distance,
    inversion_count,
    kendall_tau_distance,
    metric_sanity_checks,
    occurrence_tokens,
    scalar_path_curvature,
    spearman_position_distance,
    state_space_curvature,
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


if __name__ == "__main__":
    unittest.main()
