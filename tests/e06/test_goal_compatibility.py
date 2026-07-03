"""E06 S04 goal-compatibility tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.algotype_library import builtin_control_records
from src.e06.goal_compatibility import (
    DEFAULT_GOAL_PROFILES,
    S04Config,
    evaluate_goal_state,
    goal_assignments_for_profile,
    goal_sortedness,
    rank_values,
    run_s04_sweep,
    select_s04_candidates,
    validate_s04_outputs,
)


class GoalCompatibilityTests(unittest.TestCase):
    def test_goal_rankings_score_expected_orders(self) -> None:
        values = (1, 2, 3, 4)
        self.assertEqual(goal_sortedness(values, "increasing", 4)["inversion_sortedness"], 1.0)
        self.assertEqual(goal_sortedness(tuple(reversed(values)), "decreasing", 4)["inversion_sortedness"], 1.0)
        self.assertEqual(rank_values((2, 1, 4, 3), "parity_even_first", 4), (2, 5, 4, 7))
        self.assertLess(goal_sortedness(values, "decreasing", 4)["inversion_sortedness"], 1.0)

    def test_evaluate_goal_state_contains_relevant_scores(self) -> None:
        profile = DEFAULT_GOAL_PROFILES[1]
        policy_ids = ("a", "b")
        assignments = goal_assignments_for_profile(policy_ids, profile)
        evaluation = evaluate_goal_state(
            values=(1, 2, 4, 3),
            labels=("a", "a", "b", "b"),
            goal_assignments=assignments,
            profile=profile,
            array_size=4,
            aggregation_delta_percent=12.0,
        )
        scores = json.loads(evaluation["goal_scores_json"])
        self.assertIn("increasing", scores)
        self.assertIn("decreasing", scores)
        self.assertTrue(0.0 <= evaluation["assigned_policy_mean_sortedness"] <= 1.0)
        self.assertTrue(evaluation["goal_state_class"])

    def test_select_s04_candidates_uses_sensitive_and_memory_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "s02_candidate_id": f"c{i}",
                    "s02_candidate_rank": i,
                    "panel": "memory_vs_original" if i == 4 else "frontier_vs_original",
                    "candidate_reason": "memory_repair" if i == 4 else "minority_effect",
                    "policy_ids_json": json.dumps([f"p{i}", f"q{i}"], separators=(",", ":")),
                    "display_names_json": json.dumps(["original_bubble", "e04_memory_repair_01" if i == 4 else f"frontier_{i}"], separators=(",", ":")),
                    "ratio_targets_json": "[0.5,0.5]",
                    "sortedness_range_across_arrangements": 0.20 - i * 0.02,
                    "aggregation_delta_range_across_arrangements": 5.0 + i,
                    "best_sortedness_arrangement": "contiguous_patch",
                    "worst_sortedness_arrangement": "alternating",
                    "arrangement_sensitive_candidate": i < 3,
                }
                for i in range(5)
            ]
        )
        selected = select_s04_candidates(frame, S04Config(max_sensitive_rows=2, include_memory_contrast=True))
        self.assertEqual(len(selected), 3)
        self.assertTrue((selected["selection_reason"] == "memory_repair_contrast").any())

    def test_small_s04_sweep_runs_and_validates(self) -> None:
        records = builtin_control_records()
        sensitivity = pd.DataFrame(
            [
                {
                    "s02_candidate_id": "unit",
                    "s02_candidate_rank": 1,
                    "panel": "unit",
                    "candidate_reason": "unit",
                    "policy_ids_json": json.dumps([records[0]["algotypeId"], records[2]["algotypeId"]], separators=(",", ":")),
                    "display_names_json": json.dumps([records[0]["displayName"], records[2]["displayName"]], separators=(",", ":")),
                    "ratio_targets_json": "[0.25,0.75]",
                    "sortedness_range_across_arrangements": 0.2,
                    "aggregation_delta_range_across_arrangements": 12.0,
                    "best_sortedness_arrangement": "contiguous_patch",
                    "worst_sortedness_arrangement": "alternating",
                    "arrangement_sensitive_candidate": True,
                }
            ]
        )
        config = S04Config(
            array_size=20,
            event_cap=20,
            seeds=(1, 2),
            max_sensitive_rows=1,
            include_memory_contrast=False,
            goal_profiles=DEFAULT_GOAL_PROFILES[:2],
        )
        run_df, condition_df, selected = run_s04_sweep(records, sensitivity, config)
        validation = validate_s04_outputs(run_df, condition_df, selected, config, figure_written=True, unit_tests_success=True)
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertEqual(len(run_df), 8)
        self.assertTrue(cases["per_algotype_goal_assignments_valid"])
        self.assertTrue(cases["final_state_evaluated_under_relevant_goals"])
        self.assertTrue(cases["matched_seed_sets_by_condition"])
        self.assertTrue(cases["matched_goal_profiles_by_seed"])


if __name__ == "__main__":
    unittest.main()
