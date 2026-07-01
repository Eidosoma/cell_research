"""Focused tests for the E02 S03 activation-rate intervention."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from scripts.e02_s03_activation_rates import (
    ActivationRateTask,
    activation_rate_deviation,
    make_rate_profiles,
    run_activation_task,
)


class ActivationRateTests(unittest.TestCase):
    def test_make_rate_profiles_includes_equal_and_each_fast_label(self) -> None:
        profiles = make_rate_profiles(("bubble", "insertion", "selection"))
        regimes = [profile[0] for profile in profiles]
        self.assertEqual(regimes, ["equal_per_cell", "fast_bubble_3x", "fast_insertion_3x", "fast_selection_3x"])
        self.assertEqual(profiles[0][2], {"bubble": 1, "insertion": 1, "selection": 1})
        self.assertEqual(profiles[1][2], {"bubble": 3, "insertion": 1, "selection": 1})

    def test_activation_rate_deviation_is_exact_for_complete_weighted_cycles(self) -> None:
        label_counts = Counter({"a": 2, "b": 1})
        weights = {"a": 3, "b": 1}
        activation_counts = Counter({"a": 12, "b": 2})
        max_share_error, max_ratio_error, passed = activation_rate_deviation(label_counts, weights, activation_counts)
        self.assertEqual(max_share_error, 0.0)
        self.assertEqual(max_ratio_error, 0.0)
        self.assertTrue(passed)

    def test_small_same_code_control_run_validates_activation_rates(self) -> None:
        task = ActivationRateTask(
            condition={
                "condition_id": "TEST",
                "run_family": "same_goal_chimera",
                "mode": "cell_view",
                "algorithm": "mixed",
                "algotype_mix": "same_algorithm_bubble_label_control",
                "value_bank_id": "test_values",
                "algotype_assignment_bank_id": "test_assignments",
                "frozen_semantics": "none",
                "frozen_count": 0,
                "frozen_index_bank_id": "frozen_0",
            },
            repeat_index=0,
            values=(4, 1, 3, 2),
            initial_array_seed=1,
            assignments=("bubble_label_a", "bubble_label_b", "bubble_label_a", "bubble_label_b"),
            assignment_seed=2,
            frozen_indices=(),
            frozen_index_seed=None,
            activation_rate_regime="fast_bubble_label_a_3x",
            fast_label="bubble_label_a",
            label_weights={"bubble_label_a": 3, "bubble_label_b": 1},
            scheduler_seed=101,
            repo_dir=str(Path(__file__).resolve().parents[2]),
            max_cycles=500,
            max_successful_swaps=1000,
            stall_cycles=2,
        )
        row = run_activation_task(task)
        self.assertEqual(row["stop_reason"], "sorted")
        self.assertEqual(row["final_sortedness_percent"], 100.0)
        self.assertTrue(row["activation_rate_validation_passed"])
        self.assertEqual(row["activation_rate_max_abs_share_error"], 0.0)
        self.assertTrue(row["activation_log_sha256"])


if __name__ == "__main__":
    unittest.main()
