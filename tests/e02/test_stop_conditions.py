"""Focused tests for E02 S13 stop-condition sensitivity runner."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.e02_s13_stop_conditions import (
    StopTask,
    build_tasks,
    run_stop_task,
    selected_stop_regimes,
    stop_config_for_regime,
    summarize_results,
)


def minimal_e01_config() -> dict:
    return {
        "globalDefaults": {"repeatCount": 2, "baseSeed": 7000},
        "seedBanks": {
            "valueBanks": {
                "unique_1_to_100": {
                    "seeds": [101, 102],
                    "initialArrays": [[4, 1, 3, 2], [3, 2, 4, 1]],
                }
            },
            "frozenIndexBanks": {
                "frozen3_random": {
                    "seeds": [201, 202],
                    "indices": [[0, 2, 3], [0, 1, 3]],
                }
            },
        },
        "conditions": [
            {
                "condition_id": "E01C004",
                "run_family": "unperturbed",
                "mode": "cell_view",
                "algorithm": "bubble",
                "algotype_mix": "pure",
                "value_bank_id": "unique_1_to_100",
                "value_distribution": "random_permutation_unique",
                "direction_profile": "all_increasing",
                "frozen_semantics": "none",
                "frozen_count": 0,
                "frozen_index_bank_id": None,
            },
            {
                "condition_id": "E01C040",
                "run_family": "frozen_stuck",
                "mode": "cell_view",
                "algorithm": "bubble",
                "algotype_mix": "pure",
                "value_bank_id": "unique_1_to_100",
                "value_distribution": "random_permutation_unique",
                "direction_profile": "all_increasing",
                "frozen_semantics": "stuck",
                "frozen_count": 3,
                "frozen_index_bank_id": "frozen3_random",
            },
        ],
    }


class StopConditionRunnerTests(unittest.TestCase):
    def test_selected_stop_regimes_preserves_order_and_rejects_unknown(self) -> None:
        regimes = selected_stop_regimes(["low_event_cap_25000", "reference_no_legal_2000"])
        self.assertEqual([row["name"] for row in regimes], ["low_event_cap_25000", "reference_no_legal_2000"])
        with self.assertRaisesRegex(RuntimeError, "Unsupported S13 stop regimes"):
            selected_stop_regimes(["missing_regime"])

    def test_build_tasks_reuses_activation_seed_across_stop_regimes(self) -> None:
        cfg = minimal_e01_config()
        regimes = selected_stop_regimes(["reference_no_legal_2000", "low_event_cap_25000"])
        tasks = build_tasks(
            cfg,
            [cfg["conditions"][0]],
            regimes,
            Path("/workspace/cell-research"),
            max_repeats=2,
            activation_distribution="uniform_active",
        )
        self.assertEqual(len(tasks), 4)
        grouped: dict[tuple[str, int], set[int]] = {}
        for task in tasks:
            key = (task.condition["condition_id"], task.repeat_index)
            grouped.setdefault(key, set()).add(task.activation_seed)
            encoded = stop_config_for_regime(task.stop_regime)
            self.assertIn("max_events", encoded)
            self.assertIn("stall_events", encoded)
            self.assertIn("convergence_criterion", encoded)
        self.assertTrue(grouped)
        self.assertTrue(all(len(seeds) == 1 for seeds in grouped.values()))

    def test_run_stop_task_logs_caps_reason_and_frozen_identity(self) -> None:
        condition = minimal_e01_config()["conditions"][1]
        task = StopTask(
            condition=condition,
            repeat_index=0,
            stop_regime={
                "name": "unit_no_convergence_cap",
                "family": "convergence_criterion",
                "max_events": 5,
                "max_successful_swaps": None,
                "stall_events": 3,
                "stop_when_sorted": False,
                "stop_sortedness_threshold": None,
                "convergence_criterion": "none",
                "description": "unit-test cap regime",
            },
            values=(4, 1, 3, 2),
            assignments=("bubble", "bubble", "bubble", "bubble"),
            reverse_directions=(False, False, False, False),
            frozen_indices=(0, 2, 3),
            initial_frozen_values=(4, 3, 2),
            initial_array_seed=101,
            frozen_bank_seed=201,
            activation_seed=7101000,
            repo_dir="/workspace/cell-research",
            activation_distribution="uniform_active",
            sort_direction="increasing",
        )
        row = run_stop_task(task)
        self.assertEqual(row["stop_reason"], "max_events_exceeded")
        self.assertTrue(row["stop_reason_logged"])
        self.assertEqual(row["max_events_config"], 5)
        self.assertEqual(row["stall_events_config"], 3)
        self.assertEqual(row["convergence_criterion_config"], "none")
        self.assertTrue(row["frozen_identity_validation_passed"])

    def test_summarize_results_classifies_final_state_sensitivity(self) -> None:
        common = {
            "condition_id": "E01C004",
            "run_family": "unperturbed",
            "algorithm": "bubble",
            "algotype_mix": "pure",
            "frozen_semantics": "none",
            "frozen_count": 0,
            "repeat_index": 0,
            "mean_marker": 1,
            "final_monotonicity_error_count": 0,
            "compare_plus_swap_steps": 100,
            "event_count": 100,
            "swap_only_steps": 50,
            "frozen_attempt_count": 0,
            "dg_primary": 0.0,
            "sortedness_path_curvature_ratio": 1.0,
            "value_state_path_curvature_ratio": 1.0,
            "max_guard_hit": False,
            "frozen_identity_validation_passed": True,
        }
        df = pd.DataFrame(
            [
                {
                    **common,
                    "stop_regime": "reference_no_legal_2000",
                    "stop_regime_family": "reference",
                    "final_sorted_non_decreasing": True,
                    "final_sortedness_percent": 100.0,
                    "stop_reason": "sorted",
                },
                {
                    **common,
                    "stop_regime": "low_event_cap_25000",
                    "stop_regime_family": "max_step_cap",
                    "final_sorted_non_decreasing": False,
                    "final_sortedness_percent": 50.0,
                    "final_monotonicity_error_count": 3,
                    "event_count": 25,
                    "stop_reason": "max_events",
                    "max_guard_hit": True,
                },
            ]
        )
        _summary, classified = summarize_results(df)
        classes = dict(zip(classified["stop_regime"], classified["stop_sensitivity_class"], strict=True))
        self.assertEqual(classes["reference_no_legal_2000"], "reference_stop_rule")
        self.assertEqual(classes["low_event_cap_25000"], "stop_sensitive_final_state")


if __name__ == "__main__":
    unittest.main()
