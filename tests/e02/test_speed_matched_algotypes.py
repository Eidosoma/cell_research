"""Focused tests for E02 S06 speed-matched Algotype controls."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from scripts.e02_s06_speed_matched_algotypes import (
    SPEED_MATCH_RATE_REGIME,
    SpeedMatchedTask,
    choose_speed_match_weights,
    run_speed_matched_task,
    speed_match_validation_table,
)


class SpeedMatchedAlgotypeTests(unittest.TestCase):
    def test_integer_weight_search_matches_expected_step_burden(self) -> None:
        pure_means = {"bubble": 7181.4, "insertion": 4991.1, "selection": 8120.58}
        weights, expected_costs, ratio, cv = choose_speed_match_weights(
            ("bubble", "insertion", "selection"),
            pure_means,
            max_weight=8,
        )
        self.assertEqual(weights, {"bubble": 7, "insertion": 5, "selection": 8})
        self.assertLessEqual(ratio, 1.05)
        self.assertLess(cv, 0.02)
        self.assertLess(max(expected_costs.values()) / min(expected_costs.values()), 1.05)

    def test_validation_table_records_weighted_pure_distribution(self) -> None:
        pure_stats = pd.DataFrame(
            [
                {
                    "algorithm": "bubble",
                    "n_runs": 2,
                    "mean_compare_plus_swap_steps": 100.0,
                    "std_compare_plus_swap_steps": 10.0,
                    "q10_compare_plus_swap_steps": 90.0,
                    "q50_compare_plus_swap_steps": 100.0,
                    "q90_compare_plus_swap_steps": 110.0,
                },
                {
                    "algorithm": "insertion",
                    "n_runs": 2,
                    "mean_compare_plus_swap_steps": 50.0,
                    "std_compare_plus_swap_steps": 5.0,
                    "q10_compare_plus_swap_steps": 45.0,
                    "q50_compare_plus_swap_steps": 50.0,
                    "q90_compare_plus_swap_steps": 55.0,
                },
            ]
        )
        rows = [
            {
                "condition_id": "TEST",
                "algotype_mix": "bubble_insertion",
                "activation_rate_regime": SPEED_MATCH_RATE_REGIME,
                "configured_label_weights_json": json.dumps({"bubble": 2, "insertion": 1}, sort_keys=True),
                "expected_cycle_cost_by_label_json": json.dumps({"bubble": 50.0, "insertion": 50.0}, sort_keys=True),
            }
        ]
        detail = speed_match_validation_table(rows, pure_stats, threshold=1.05)
        self.assertEqual(len(detail), 2)
        self.assertTrue(detail["passes_speed_ratio_threshold"].all())
        bubble = detail[detail["label"] == "bubble"].iloc[0]
        self.assertEqual(bubble["weighted_expected_cycle_mean_steps"], 50.0)
        self.assertEqual(bubble["weighted_expected_cycle_q90_steps"], 55.0)

    def test_small_same_code_run_preserves_dispatch_and_activation_distribution(self) -> None:
        task = SpeedMatchedTask(
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
            activation_rate_regime=SPEED_MATCH_RATE_REGIME,
            label_weights={"bubble_label_a": 2, "bubble_label_b": 1},
            speed_match_basis="test_basis",
            pure_efficiency_means={"bubble_label_a": 100.0, "bubble_label_b": 100.0},
            expected_cycle_cost_by_label={"bubble_label_a": 50.0, "bubble_label_b": 100.0},
            expected_cycle_cost_ratio=2.0,
            expected_cycle_cost_cv=0.333,
            scheduler_seed=321,
            repo_dir=str(Path(__file__).resolve().parents[2]),
            max_cycles=200,
            max_successful_swaps=1000,
            stall_cycles=2,
        )
        row = run_speed_matched_task(task)
        self.assertTrue(row["policy_dispatch_preserved"])
        self.assertTrue(row["activation_rate_validation_passed"])
        self.assertEqual(row["activation_rate_max_abs_share_error"], 0.0)
        self.assertEqual(row["activation_per_weight_ratio"], 1.0)
        self.assertEqual(row["stop_reason"], "sorted")
        self.assertEqual(row["final_sortedness_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
