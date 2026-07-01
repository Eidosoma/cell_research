"""Focused tests for E02 S12 Frozen Cell behavior variants."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.e02_s12_frozen_behaviors import (
    BEHAVIOR_VARIANTS,
    build_tasks,
    micro_behavior_validation,
    selected_behavior_variants,
    selected_condition_pairs,
    summarize_results,
)


def fake_config() -> dict:
    values0 = [5, 1, 4, 2, 3, 6]
    values1 = [6, 5, 4, 3, 2, 1]
    return {
        "globalDefaults": {
            "arrayLength": 6,
            "baseSeed": 12345,
            "repeatCount": 2,
        },
        "seedBanks": {
            "valueBanks": {
                "unique_1_to_100": {
                    "initialArrays": [values0, values1],
                    "seeds": [101, 102],
                },
            },
            "algotypeAssignmentBanks": {},
            "frozenIndexBanks": {
                "frozen_1": {"indices": [[4], [2]], "seeds": [201, 202]},
            },
        },
        "conditions": [
            {
                "condition_id": "E01C010",
                "run_family": "synthetic",
                "mode": "cell_view",
                "algorithm": "bubble",
                "algotype_mix": "bubble",
                "algotype_assignment_bank_id": "not_applicable",
                "frozen_semantics": "passive",
                "frozen_count": 1,
                "frozen_index_bank_id": "frozen_1",
                "value_bank_id": "unique_1_to_100",
                "value_distribution": "unique_1_to_100",
                "direction_profile": "all_increasing",
                "algotype_roles": "bubble:increasing",
            },
            {
                "condition_id": "E01C028",
                "run_family": "synthetic",
                "mode": "cell_view",
                "algorithm": "bubble",
                "algotype_mix": "bubble",
                "algotype_assignment_bank_id": "not_applicable",
                "frozen_semantics": "stuck",
                "frozen_count": 1,
                "frozen_index_bank_id": "frozen_1",
                "value_bank_id": "unique_1_to_100",
                "value_distribution": "unique_1_to_100",
                "direction_profile": "all_increasing",
                "algotype_roles": "bubble:increasing",
            },
        ],
    }


class FrozenBehaviorTests(unittest.TestCase):
    def test_selected_variants_reject_unknown_names(self) -> None:
        variants = selected_behavior_variants(["original_passive", "fatigue_recovery"])
        self.assertEqual([row["name"] for row in variants], ["original_passive", "fatigue_recovery"])
        with self.assertRaises(RuntimeError):
            selected_behavior_variants(["missing_variant"])

    def test_build_tasks_pairs_original_and_dynamic_variants(self) -> None:
        cfg = fake_config()
        pairs = selected_condition_pairs(cfg, ["E01C010"])
        variants = selected_behavior_variants(["original_passive", "dynamic_passive_limit", "fatigue_recovery"])
        tasks = build_tasks(
            cfg,
            pairs,
            variants,
            Path.cwd(),
            max_repeats=2,
            max_events=500,
            max_successful_swaps=100,
            stall_events=50,
            activation_distribution="uniform_active",
        )
        self.assertEqual(len(tasks), 6)
        self.assertEqual({task.pair.stuck_condition["condition_id"] for task in tasks}, {"E01C028"})
        dynamic = [task for task in tasks if task.frozen_semantics == "dynamic"]
        self.assertTrue(all(task.behavior_spec is not None for task in dynamic))
        self.assertTrue(all(task.behavior_seed is not None for task in dynamic))
        seeds_by_repeat = {}
        for task in tasks:
            seeds_by_repeat.setdefault(task.repeat_index, set()).add(task.activation_seed)
        self.assertTrue(all(len(seeds) == 1 for seeds in seeds_by_repeat.values()))

    def test_summary_classifies_behavior_sensitivity(self) -> None:
        base = {
            "condition_pair_id": "bubble_f1",
            "passive_condition_id": "E01C010",
            "stuck_condition_id": "E01C028",
            "run_family": "synthetic",
            "algorithm": "bubble",
            "algotype_mix": "bubble",
            "frozen_count": 1,
            "frozen_semantics": "dynamic",
            "repeat_index": 0,
            "final_sorted_non_decreasing": True,
            "final_sortedness_percent": 100.0,
            "peak_sortedness_percent": 100.0,
            "curve_mean_sortedness_percent": 80.0,
            "final_monotonicity_error_count": 0,
            "curve_mean_monotonicity_error_count": 1.0,
            "compare_plus_swap_steps": 100.0,
            "event_count": 100.0,
            "swap_only_steps": 50.0,
            "frozen_attempt_count": 10.0,
            "dg_primary": 0.0,
            "dg_event_count": 0,
            "peak_aggregation_left_neighbor_percent": 0.0,
            "final_aggregation_left_neighbor_percent": 0.0,
            "sortedness_path_curvature_ratio": 1.0,
            "value_state_path_curvature_ratio": 1.0,
            "behavior_transition_count": 1,
            "behavior_attempt_count": 1,
            "behavior_allowed_attempt_count": 1,
            "behavior_blocked_attempt_count": 0,
            "max_guard_hit": False,
            "frozen_identity_validation_passed": True,
        }
        frame = pd.DataFrame(
            [
                {**base, "behavior_variant": "original_passive", "behavior_family": "original_limit", "frozen_semantics": "passive"},
                {**base, "behavior_variant": "original_stuck", "behavior_family": "original_limit", "frozen_semantics": "stuck", "final_sortedness_percent": 90.0},
                {
                    **base,
                    "behavior_variant": "probabilistic_stuck_p75",
                    "behavior_family": "probabilistic",
                    "final_sorted_non_decreasing": False,
                    "final_sortedness_percent": 70.0,
                    "final_monotonicity_error_count": 8,
                },
            ]
        )
        _, classified = summarize_results(frame)
        cls = classified[classified["behavior_variant"] == "probabilistic_stuck_p75"]["behavior_sensitivity_class"].iloc[0]
        self.assertEqual(cls, "behavior_sensitive_final_state")

    def test_micro_behavior_validation_passes(self) -> None:
        validation = micro_behavior_validation(Path.cwd())
        self.assertTrue(validation["microBehaviorValidationPassed"])
        self.assertEqual(len(validation["microBehaviorValidationChecks"]), 5)

    def test_behavior_variant_names_are_unique(self) -> None:
        names = [row["name"] for row in BEHAVIOR_VARIANTS]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
