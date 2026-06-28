from __future__ import annotations

import unittest

import pandas as pd

from chimera.goals import (
    GOAL_MODES,
    augment_goal_results,
    build_goal_condition_matrix,
    load_s03_results,
    policy_goal_specs,
    reverse_directions_for_assignment,
)
from chimera.mixtures import compact_json, load_ready_panel, run_mixture_condition


class TestE06GoalCompatibility(unittest.TestCase):
    def test_policy_goal_specs_cover_required_modes(self) -> None:
        policy_ids = ["left", "right"]
        for mode in GOAL_MODES:
            specs = policy_goal_specs(policy_ids, mode)
            self.assertEqual(set(specs), set(policy_ids))
            self.assertTrue({"goalType", "targetDirection", "behaviorDirection", "role"} <= set(specs["left"]))
        opposite = policy_goal_specs(policy_ids, "opposite_goal")
        self.assertEqual(opposite["left"]["behaviorDirection"], "increasing")
        self.assertEqual(opposite["right"]["behaviorDirection"], "decreasing")
        assigned = ["left", "right", "right", "left"]
        self.assertEqual(reverse_directions_for_assignment(assigned, opposite), [False, True, True, False])

    def test_goal_condition_matrix_has_all_modes_per_template(self) -> None:
        ready = load_ready_panel()
        s03 = load_s03_results()
        conditions, selected, condition_meta, policy_meta = build_goal_condition_matrix(
            s03,
            ready,
            max_conditions_per_category=1,
            max_total_conditions=2,
        )
        self.assertFalse(selected.empty)
        self.assertEqual(len(conditions), len(selected) * len(GOAL_MODES))
        self.assertEqual(len(condition_meta), len(conditions))
        self.assertEqual(len(policy_meta), len(conditions) * 2)
        for _, group in conditions.groupby("s03SourceConditionId"):
            self.assertEqual(set(group["goalMode"]), set(GOAL_MODES))
            self.assertEqual(group["policyIdsJson"].nunique(), 1)
            self.assertEqual(group["policyCountsJson"].nunique(), 1)
            self.assertEqual(group["arrangementPolicyIdsJson"].nunique(), 1)
            self.assertEqual(group["valueSeed"].nunique(), 1)
            self.assertEqual(group["schedulerSeed"].nunique(), 1)

    def test_opposite_goal_smoke_replay_and_scores(self) -> None:
        records = [
            {
                "panelPolicyId": "left",
                "constructorKind": "morphospace_json",
                "constructorPayloadJson": '{"algotype":"bubble","family":"classic","parameters":{},"policy_id":"classic_bubble","version":"e03_s01_policy_interface.v1"}',
                "simulatorKind": "policy_event",
            },
            {
                "panelPolicyId": "right",
                "constructorKind": "morphospace_json",
                "constructorPayloadJson": '{"algotype":"insertion","family":"classic","parameters":{},"policy_id":"classic_insertion","version":"e03_s01_policy_interface.v1"}',
                "simulatorKind": "policy_event",
            },
        ]
        assigned = ["left", "left", "right", "right", "left", "right", "left", "right"]
        specs = policy_goal_specs(["left", "right"], "opposite_goal")
        reverse = reverse_directions_for_assignment(assigned, specs)
        condition = {
            "conditionId": "unit_s04_opposite",
            "researchStepId": "S04",
            "implementationPrefix": "e06_s04",
            "mixtureKind": "pair",
            "leftPanelPolicyId": "left",
            "rightPanelPolicyId": "right",
            "thirdPanelPolicyId": "",
            "pairCategory": "unit",
            "priorityReason": "unit",
            "ratioLabel": "50:50",
            "targetCountsJson": '{"left":4,"right":4}',
            "targetProportionsJson": '{"left":0.5,"right":0.5}',
            "n": 8,
            "targetPolicyCount": 2,
            "policyIdsJson": '["left","right"]',
            "policyCountsJson": "[4,4]",
            "arrangementType": "unit",
            "arrangementPolicyIdsJson": compact_json(assigned),
            "arrangementNumericLabelsJson": "[0,0,1,1,0,1,0,1]",
            "arrangementDescriptorJson": "{}",
            "arrangementHash": "unit",
            "goalMode": "opposite_goal",
            "goalCompatibilityClass": "opposite",
            "policyGoalMetadataJson": compact_json(specs),
            "goalReverseDirectionsJson": compact_json(reverse),
            "goalDirectionByPolicyJson": compact_json({"left": "increasing", "right": "decreasing"}),
            "conditionHash": "unit",
            "s04Version": "unit",
            "seedIndex": 0,
            "valueSeed": 1,
            "arrangementSeed": 2,
            "schedulerSeed": 3,
            "tieBreakerSeed": 4,
        }
        lookup = {row["panelPolicyId"]: row for row in records}
        first = run_mixture_condition(condition, lookup, max_activations=400, max_swaps=200, max_comparisons=1000)
        second = run_mixture_condition(condition, lookup, max_activations=400, max_swaps=200, max_comparisons=1000)
        self.assertTrue(first["runSucceeded"])
        self.assertEqual(first["finalStateHash"], second["finalStateHash"])
        scored = augment_goal_results(pd.DataFrame([first]))
        self.assertIn("meanPolicyGoalScore", scored.columns)
        self.assertIn("dominantDirectionProxy", scored.columns)
        self.assertEqual(scored.iloc[0]["goalMode"], "opposite_goal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
