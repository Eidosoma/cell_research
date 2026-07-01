"""Focused tests for E02 S10 input-distribution stress tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.e02_s10_input_distributions import (
    INPUT_DISTRIBUTIONS,
    InputTask,
    build_tasks,
    generate_distribution_values,
    run_input_task,
    summarize_results,
    validate_distribution_values,
)


def fake_config() -> dict:
    unique = list(range(1, 101))
    duplicate = [value for value in range(1, 11) for _ in range(10)]
    return {
        "globalDefaults": {
            "arrayLength": 100,
            "baseSeed": 12345,
            "repeatCount": 2,
        },
        "seedBanks": {
            "valueBanks": {
                "unique_1_to_100": {
                    "initialArrays": [unique, list(reversed(unique))],
                    "seeds": [101, 102],
                },
                "duplicate_1_to_10_x10": {
                    "initialArrays": [duplicate, list(reversed(duplicate))],
                    "seeds": [201, 202],
                },
            },
            "algotypeAssignmentBanks": {
                "bubble_insertion_50_50": {
                    "assignments": [["bubble"] * 50 + ["insertion"] * 50, ["insertion"] * 50 + ["bubble"] * 50],
                    "seeds": [301, 302],
                }
            },
            "frozenIndexBanks": {
                "frozen_0": {"indices": [[], []], "seeds": []},
            },
        },
    }


class InputDistributionTests(unittest.TestCase):
    def test_distribution_generators_satisfy_constraints(self) -> None:
        cfg = fake_config()
        for name in INPUT_DISTRIBUTIONS:
            generated = generate_distribution_values(cfg, name, 0, 100)
            self.assertTrue(generated.validation["constraints_passed"], name)
            regenerated = generate_distribution_values(cfg, name, 0, 100)
            self.assertEqual(generated.values, regenerated.values)

    def test_explicit_validation_rejects_bad_duplicate_distribution(self) -> None:
        bad = [1] * 100
        validation = validate_distribution_values("duplicate_heavy", bad, 100, 1)
        self.assertFalse(validation["constraints_passed"])
        self.assertIn("duplicate_values_not_1_to_10", validation["constraint_failures"])

    def test_build_tasks_counts_condition_distribution_repeats(self) -> None:
        cfg = fake_config()
        condition = {
            "condition_id": "T001",
            "run_family": "synthetic",
            "mode": "cell_view",
            "algorithm": "mixed",
            "algotype_mix": "bubble_insertion",
            "algotype_assignment_bank_id": "bubble_insertion_50_50",
            "frozen_semantics": "none",
            "frozen_count": 0,
            "frozen_index_bank_id": "frozen_0",
            "direction_profile": "all_increasing",
            "algotype_roles": "bubble:increasing;insertion:increasing",
        }
        tasks = build_tasks(
            cfg,
            [condition],
            Path.cwd(),
            max_repeats=2,
            max_events=100,
            max_successful_swaps=50,
            stall_events=10,
            activation_distribution="uniform_active",
            input_distributions=("random_permutation", "duplicate_heavy"),
        )
        self.assertEqual(len(tasks), 4)
        self.assertEqual({task.input_distribution for task in tasks}, {"random_permutation", "duplicate_heavy"})
        self.assertEqual({task.repeat_index for task in tasks}, {0, 1})

    def test_small_simulator_task_sorts_and_reports_metrics(self) -> None:
        condition = {
            "condition_id": "T002",
            "run_family": "synthetic",
            "algorithm": "bubble",
            "algotype_mix": "bubble",
            "algotype_assignment_bank_id": "not_applicable",
            "frozen_semantics": "none",
            "frozen_count": 0,
            "frozen_index_bank_id": "frozen_0",
        }
        validation = validate_distribution_values("random_permutation", [3, 1, 2], 3, 11)
        task = InputTask(
            condition=condition,
            repeat_index=0,
            input_distribution="random_permutation",
            values=(3, 1, 2),
            assignments=("bubble", "bubble", "bubble"),
            reverse_directions=(False, False, False),
            frozen_indices=(),
            input_distribution_seed=11,
            source_value_bank_id="synthetic",
            generation_policy="unit test",
            generation_validation=validation | {"constraints_passed": True},
            algotype_assignment_seed=None,
            frozen_index_seed=None,
            activation_seed=7,
            repo_dir=str(Path.cwd()),
            max_events=500,
            max_successful_swaps=100,
            stall_events=50,
            activation_distribution="uniform_active",
            sort_direction="increasing",
        )
        row = run_input_task(task)
        self.assertEqual(row["final_monotonicity_error_count"], 0)
        self.assertEqual(row["final_sorted_non_decreasing"], True)
        self.assertIn("kendall_tau_distance_final_progress_percent", row)

    def test_summary_classifies_synthetic_sensitivity(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "condition_id": "C",
                    "run_family": "synthetic",
                    "algorithm": "bubble",
                    "algotype_mix": "bubble",
                    "frozen_semantics": "none",
                    "frozen_count": 0,
                    "input_distribution": "random_permutation",
                    "repeat_index": 0,
                    "final_sorted_non_decreasing": True,
                    "final_sortedness_percent": 100.0,
                    "peak_sortedness_percent": 100.0,
                    "curve_mean_sortedness_percent": 80.0,
                    "final_monotonicity_error_count": 0,
                    "compare_plus_swap_steps": 100.0,
                    "event_count": 100.0,
                    "swap_only_steps": 50.0,
                    "dg_primary": 0.0,
                    "dg_event_count": 0,
                    "peak_aggregation_left_neighbor_percent": 0.0,
                    "final_aggregation_left_neighbor_percent": 0.0,
                    "sortedness_path_curvature_ratio": 1.0,
                    "value_state_path_curvature_ratio": 1.0,
                    "kendall_tau_distance_final_progress_percent": 100.0,
                    "earth_mover_position_distance_final_progress_percent": 100.0,
                    "max_guard_hit": False,
                },
                {
                    "condition_id": "C",
                    "run_family": "synthetic",
                    "algorithm": "bubble",
                    "algotype_mix": "bubble",
                    "frozen_semantics": "none",
                    "frozen_count": 0,
                    "input_distribution": "reverse_sorted",
                    "repeat_index": 0,
                    "final_sorted_non_decreasing": True,
                    "final_sortedness_percent": 100.0,
                    "peak_sortedness_percent": 100.0,
                    "curve_mean_sortedness_percent": 50.0,
                    "final_monotonicity_error_count": 0,
                    "compare_plus_swap_steps": 300.0,
                    "event_count": 300.0,
                    "swap_only_steps": 150.0,
                    "dg_primary": 0.0,
                    "dg_event_count": 0,
                    "peak_aggregation_left_neighbor_percent": 0.0,
                    "final_aggregation_left_neighbor_percent": 0.0,
                    "sortedness_path_curvature_ratio": 1.0,
                    "value_state_path_curvature_ratio": 1.0,
                    "kendall_tau_distance_final_progress_percent": 100.0,
                    "earth_mover_position_distance_final_progress_percent": 100.0,
                    "max_guard_hit": False,
                },
            ]
        )
        _, classified = summarize_results(frame)
        cls = classified[classified["input_distribution"] == "reverse_sorted"]["input_generalization_class"].iloc[0]
        self.assertEqual(cls, "input_sensitive_efficiency_or_path")


if __name__ == "__main__":
    unittest.main()
