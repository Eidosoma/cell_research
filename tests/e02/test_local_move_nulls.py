"""Focused tests for E02 S07 local-move null models."""

from __future__ import annotations

import unittest

from scripts.e02_s07_local_move_nulls import (
    NULL_POLICIES,
    hidden_label_access_check,
    simulate_null_path,
    trajectory_shape_metrics,
)


class LocalMoveNullTests(unittest.TestCase):
    def test_null_policies_match_exact_swap_count_and_locality(self) -> None:
        values = (4, 1, 3, 2)
        labels = ("bubble", "insertion", "bubble", "insertion")
        for idx, policy in enumerate(NULL_POLICIES):
            with self.subTest(policy=policy):
                result = simulate_null_path(values, labels, policy, seed=100 + idx, target_swap_count=12)
                self.assertEqual(result["observed_swap_count"], 12)
                self.assertEqual(result["max_abs_swap_distance"], 1)
                self.assertEqual(result["nonlocal_swap_count"], 0)
                self.assertEqual(len(result["sortedness_values"]), 13)
                self.assertEqual(len(result["aggregation_values"]), 13)

    def test_decision_path_is_label_permutation_invariant(self) -> None:
        values = (5, 1, 4, 2, 3)
        labels = ("a", "b", "a", "b", "a")
        for idx, policy in enumerate(NULL_POLICIES):
            with self.subTest(policy=policy):
                self.assertTrue(hidden_label_access_check(values, labels, policy, seed=200 + idx, target_swap_count=20))

    def test_decision_path_hash_is_unchanged_when_labels_change(self) -> None:
        values = (5, 1, 4, 2, 3)
        labels_a = ("a", "b", "a", "b", "a")
        labels_b = ("b", "a", "b", "a", "b")
        first = simulate_null_path(values, labels_a, "inversion_biased_swap", seed=77, target_swap_count=15)
        second = simulate_null_path(values, labels_b, "inversion_biased_swap", seed=77, target_swap_count=15)
        self.assertEqual(first["decision_path_sha256"], second["decision_path_sha256"])

    def test_path_curvature_is_one_for_monotone_progress_and_above_one_for_backtracking(self) -> None:
        monotone = trajectory_shape_metrics([50, 60, 70, 80])
        backtracking = trajectory_shape_metrics([50, 70, 60, 80])
        self.assertEqual(monotone["path_curvature_ratio"], 1.0)
        self.assertGreater(backtracking["path_curvature_ratio"], 1.0)
        self.assertEqual(backtracking["path_backtracking_drop_total"], 10.0)


if __name__ == "__main__":
    unittest.main()
