from __future__ import annotations

import math
import unittest

from scripts.e02_s09_alternative_metrics import (
    longest_nondecreasing_subsequence_length,
    state_distance_metrics,
    target_position_assignment,
    trajectory_curvature_metrics,
)


class TestE02AlternativeMetrics(unittest.TestCase):
    def test_sorted_unique_has_zero_distance(self) -> None:
        metrics = state_distance_metrics([1, 2, 3, 4])
        self.assertEqual(metrics["inversionCount"], 0)
        self.assertAlmostEqual(metrics["kendallTauDistanceNormalized"], 0.0)
        self.assertEqual(metrics["spearmanFootruleDistance"], 0)
        self.assertAlmostEqual(metrics["earthMoverPositionDistance"], 0.0)
        self.assertEqual(metrics["editDistanceToTargetOrder"], 0)

    def test_reverse_unique_has_max_kendall_and_expected_edit(self) -> None:
        metrics = state_distance_metrics([4, 3, 2, 1])
        self.assertEqual(metrics["inversionCount"], 6)
        self.assertEqual(metrics["comparablePairCount"], 6)
        self.assertAlmostEqual(metrics["kendallTauDistanceNormalized"], 1.0)
        self.assertAlmostEqual(metrics["spearmanFootruleDistanceNormalized"], 1.0)
        self.assertEqual(metrics["editDistanceToTargetOrder"], 3)

    def test_duplicate_sorted_arrays_are_zero_distance(self) -> None:
        metrics = state_distance_metrics([1, 1, 2, 2])
        self.assertTrue(metrics["hasDuplicates"])
        self.assertEqual(metrics["inversionCount"], 0)
        self.assertEqual(metrics["spearmanFootruleDistance"], 0)
        self.assertEqual(metrics["editDistanceToTargetOrder"], 0)
        self.assertEqual(target_position_assignment([1, 1, 2, 2]).tolist(), [0, 1, 2, 3])

    def test_duplicate_unsorted_arrays_are_tie_aware(self) -> None:
        metrics = state_distance_metrics([2, 1, 2, 1])
        self.assertTrue(metrics["hasDuplicates"])
        self.assertEqual(metrics["inversionCount"], 3)
        self.assertEqual(metrics["comparablePairCount"], 4)
        self.assertAlmostEqual(metrics["kendallTauDistanceNormalized"], 0.75)
        self.assertEqual(metrics["spearmanFootruleDistance"], 6)
        self.assertEqual(metrics["editDistanceToTargetOrder"], 2)

    def test_lnds_allows_equal_ties(self) -> None:
        self.assertEqual(longest_nondecreasing_subsequence_length([1, 1, 1]), 3)
        self.assertEqual(longest_nondecreasing_subsequence_length([2, 1, 2, 1]), 2)

    def test_state_space_curvature_detects_turns(self) -> None:
        curve = trajectory_curvature_metrics([[3, 1, 2], [1, 3, 2], [1, 2, 3]])
        self.assertEqual(curve["trajectoryStateCount"], 3)
        self.assertEqual(curve["trajectoryTurnCount"], 1)
        self.assertGreater(curve["stateSpaceCurvatureTotalTurnRadians"], 0.0)
        self.assertTrue(math.isfinite(curve["trajectoryExcessPathRatio"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
