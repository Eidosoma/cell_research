"""Focused tests for E02 S11 Frozen Cell placement stress tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.e02_s11_frozen_placement import (
    PLACEMENT_RULES,
    PlacementTask,
    center_indices,
    evenly_spaced_indices,
    high_value_indices,
    low_value_indices,
    make_placement,
    run_placement_task,
    summarize_results,
    validate_placement_rule,
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
                "frozen_3": {"indices": [[0, 2, 5], [1, 3, 5]], "seeds": [301, 302]},
            },
        },
    }


def condition(frozen_count: int = 1, frozen_semantics: str = "stuck") -> dict:
    return {
        "condition_id": "E01C999",
        "run_family": "synthetic",
        "mode": "cell_view",
        "algorithm": "bubble",
        "algotype_mix": "bubble",
        "algotype_assignment_bank_id": "not_applicable",
        "frozen_semantics": frozen_semantics,
        "frozen_count": frozen_count,
        "frozen_index_bank_id": f"frozen_{frozen_count}",
        "value_bank_id": "unique_1_to_100",
        "value_distribution": "unique_1_to_100",
        "direction_profile": "all_increasing",
        "algotype_roles": "bubble:increasing",
    }


class FrozenPlacementTests(unittest.TestCase):
    def test_rule_indices_are_expected(self) -> None:
        values = [5, 1, 4, 2, 3, 6]
        self.assertEqual(center_indices(6, 1), (2,))
        self.assertEqual(center_indices(6, 3), (1, 2, 3))
        self.assertEqual(evenly_spaced_indices(6, 3), (1, 2, 4))
        self.assertEqual(high_value_indices(values, 2), (0, 5))
        self.assertEqual(low_value_indices(values, 2), (1, 3))

    def test_make_placement_validates_all_rules(self) -> None:
        cfg = fake_config()
        row = condition(frozen_count=3)
        values = cfg["seedBanks"]["valueBanks"]["unique_1_to_100"]["initialArrays"][0]
        bank_indices = cfg["seedBanks"]["frozenIndexBanks"]["frozen_3"]["indices"][0]
        bank_seed = cfg["seedBanks"]["frozenIndexBanks"]["frozen_3"]["seeds"][0]
        for rule in PLACEMENT_RULES:
            placement = make_placement(cfg, row, values, 0, rule, bank_indices, bank_seed)
            self.assertTrue(placement.validation["placement_rule_valid"], rule)
            self.assertEqual(len(placement.indices), 3)
            self.assertEqual(len(set(placement.indices)), 3)
            self.assertTrue(all(0 <= idx < len(values) for idx in placement.indices))

    def test_validation_rejects_duplicate_or_ambiguous_indices(self) -> None:
        duplicate_indices = validate_placement_rule(
            "left_end",
            [1, 2, 3],
            [0, 0],
            2,
            random_bank_indices=[0, 1],
            seed=None,
        )
        self.assertFalse(duplicate_indices["placement_rule_valid"])
        self.assertIn("duplicate_frozen_indices", duplicate_indices["placement_rule_failures"])

        ambiguous = validate_placement_rule(
            "high_value",
            [1, 2, 2, 3],
            [3],
            1,
            random_bank_indices=[3],
            seed=None,
        )
        self.assertFalse(ambiguous["placement_rule_valid"])
        self.assertIn("high_low_value_target_ambiguous_under_duplicates", ambiguous["placement_rule_failures"])

    def test_small_simulator_task_preserves_stuck_frozen_identity(self) -> None:
        row = condition(frozen_count=1, frozen_semantics="stuck")
        validation = validate_placement_rule(
            "center",
            [3, 1, 2],
            [1],
            1,
            random_bank_indices=[1],
            seed=None,
        )
        task = PlacementTask(
            condition=row,
            repeat_index=0,
            placement_rule="center",
            values=(3, 1, 2),
            assignments=("bubble", "bubble", "bubble"),
            reverse_directions=(False, False, False),
            frozen_indices=(1,),
            initial_frozen_values=(1,),
            placement_seed=None,
            placement_source="unit_test",
            placement_policy="unit test center",
            placement_validation=validation,
            initial_array_seed=11,
            original_frozen_bank_indices=(1,),
            original_frozen_bank_seed=12,
            activation_seed=7,
            repo_dir=str(Path.cwd()),
            max_events=300,
            max_successful_swaps=100,
            stall_events=50,
            activation_distribution="uniform_active",
            sort_direction="increasing",
        )
        row = run_placement_task(task)
        self.assertTrue(row["placement_rule_valid"])
        self.assertTrue(row["frozen_identity_count_preserved"])
        self.assertTrue(row["frozen_identity_values_preserved"])
        self.assertTrue(row["stuck_frozen_positions_preserved"])
        self.assertTrue(row["frozen_identity_validation_passed"])

    def test_summary_classifies_synthetic_placement_sensitivity(self) -> None:
        base = {
            "condition_id": "C",
            "run_family": "synthetic",
            "algorithm": "bubble",
            "algotype_mix": "bubble",
            "frozen_semantics": "stuck",
            "frozen_count": 1,
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
            "initial_frozen_position_mean": 1.0,
            "initial_frozen_value_mean": 2.0,
            "max_guard_hit": False,
            "placement_rule_valid": True,
            "frozen_identity_validation_passed": True,
        }
        frame = pd.DataFrame(
            [
                {**base, "placement_rule": "random_bank"},
                {
                    **base,
                    "placement_rule": "left_end",
                    "final_sortedness_percent": 80.0,
                    "final_sorted_non_decreasing": False,
                    "final_monotonicity_error_count": 7,
                },
            ]
        )
        _, classified = summarize_results(frame)
        cls = classified[classified["placement_rule"] == "left_end"]["placement_sensitivity_class"].iloc[0]
        self.assertEqual(cls, "placement_sensitive_final_state")


if __name__ == "__main__":
    unittest.main()
