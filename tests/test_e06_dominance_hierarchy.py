from __future__ import annotations

import unittest

import pandas as pd

from chimera.dominance import (
    UNRESOLVED_WINNER,
    build_dominance_condition_matrix,
    load_s05_metrics,
    predefined_winner_criteria,
    score_dominance_results,
    select_s06_pairs,
    winner_criteria_table,
)
from chimera.goals import reverse_directions_for_assignment
from chimera.mixtures import compact_json, initial_values, load_ready_panel, parse_json_maybe


class TestE06DominanceHierarchy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ready = load_ready_panel()
        cls.s05 = load_s05_metrics()
        cls.criteria = predefined_winner_criteria(pd.DataFrame())
        cls.selected = select_s06_pairs(cls.s05, cls.ready, max_pairs=5)

    def test_winner_criteria_are_explicit_and_ordered(self) -> None:
        table = winner_criteria_table(self.criteria)
        self.assertEqual(self.criteria["researchStepId"], "S06")
        self.assertEqual(table["criterionId"].tolist(), ["policy_goal_margin", "global_direction_margin", "balanced_or_unresolved"])
        self.assertGreater(self.criteria["thresholds"]["minimumPolicyGoalMarginPct"], 0)

    def test_initial_value_profiles_are_deterministic(self) -> None:
        random_a = initial_values(123, 100, "random_unique")
        random_b = initial_values(123, 100, "random_unique")
        reversed_values = initial_values(123, 5, "reversed_unique")
        duplicate_values = initial_values(123, 100, "duplicate_1_10_x10")
        self.assertEqual(random_a, random_b)
        self.assertEqual(reversed_values, [5, 4, 3, 2, 1])
        self.assertEqual(len(duplicate_values), 100)
        self.assertEqual(len(set(duplicate_values)), 10)

    def test_condition_matrix_crosses_context_axes_and_goal_labels(self) -> None:
        conditions, condition_meta, policy_meta, arrangements = build_dominance_condition_matrix(
            self.selected.head(2),
            self.ready,
            self.criteria,
            seed_count=1,
            ratios=((50, 50),),
            arrangements=("random", "graft_like"),
            value_profiles=("random_unique", "reversed_unique"),
        )
        self.assertEqual(len(conditions), 2 * 2 * 1 * 2 * 2)
        self.assertEqual(set(conditions["goalMode"]), {"opposite_goal"})
        self.assertEqual(conditions["oppositeGoalOrientation"].nunique(), 2)
        self.assertEqual(set(conditions["valueProfile"]), {"random_unique", "reversed_unique"})
        self.assertEqual(set(conditions["perturbationType"]), {"none", "input_order_reversal_stress"})
        self.assertEqual(len(condition_meta), len(conditions))
        self.assertEqual(len(policy_meta), len(conditions) * 2)
        self.assertEqual(len(arrangements), len(conditions))
        first = conditions.iloc[0]
        assigned = [str(item) for item in parse_json_maybe(first["arrangementPolicyIdsJson"], [])]
        specs = parse_json_maybe(first["policyGoalMetadataJson"], {})
        reverse = [bool(item) for item in parse_json_maybe(first["goalReverseDirectionsJson"], [])]
        self.assertEqual(reverse, reverse_directions_for_assignment(assigned, specs))

    def test_score_dominance_results_uses_predeclared_policy_margin_first(self) -> None:
        left = "left_policy"
        right = "right_policy"
        row = {
            "conditionId": "unit_s06",
            "runSucceeded": True,
            "completed": True,
            "stopReason": "no_cell_can_move_after_two_checks",
            "policyIdsJson": compact_json([left, right]),
            "leftPanelPolicyId": left,
            "rightPanelPolicyId": right,
            "policyGoalMetadataJson": compact_json(
                {
                    left: {"targetDirection": "increasing"},
                    right: {"targetDirection": "decreasing"},
                }
            ),
            "policyGoalScoresJson": compact_json(
                {
                    left: 82.0,
                    right: 60.0,
                    "__global_increasing__": 70.0,
                    "__global_decreasing__": 68.0,
                }
            ),
            "meanPolicyGoalScore": 71.0,
            "minPolicyGoalScore": 60.0,
            "finalIncreasingSortednessPercent": 70.0,
            "finalDecreasingSortednessPercent": 68.0,
            "valueProfile": "random_unique",
            "paperExpectedWinnerPolicyId": "",
            "paperExpectedWinnerAlgotype": "",
        }
        scored = score_dominance_results(pd.DataFrame([row]), self.criteria)
        self.assertEqual(scored.iloc[0]["winnerPolicyId"], left)
        self.assertEqual(scored.iloc[0]["winnerCriterion"], "policy_goal_margin")
        self.assertGreater(scored.iloc[0]["dominanceProxyScore"], 0.20)

    def test_score_dominance_results_can_leave_weak_context_unresolved(self) -> None:
        row = {
            "conditionId": "unit_s06_unresolved",
            "runSucceeded": True,
            "completed": True,
            "stopReason": "no_cell_can_move_after_two_checks",
            "policyIdsJson": compact_json(["a", "b"]),
            "leftPanelPolicyId": "a",
            "rightPanelPolicyId": "b",
            "policyGoalMetadataJson": compact_json({"a": {"targetDirection": "increasing"}, "b": {"targetDirection": "decreasing"}}),
            "policyGoalScoresJson": compact_json({"a": 51.0, "b": 49.0, "__global_increasing__": 50.0, "__global_decreasing__": 49.0}),
            "meanPolicyGoalScore": 50.0,
            "minPolicyGoalScore": 49.0,
            "finalIncreasingSortednessPercent": 50.0,
            "finalDecreasingSortednessPercent": 49.0,
            "valueProfile": "random_unique",
            "paperExpectedWinnerPolicyId": "",
            "paperExpectedWinnerAlgotype": "",
        }
        scored = score_dominance_results(pd.DataFrame([row]), self.criteria)
        self.assertEqual(scored.iloc[0]["winnerPolicyId"], UNRESOLVED_WINNER)
        self.assertEqual(scored.iloc[0]["winnerConfidenceClass"], "unresolved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
