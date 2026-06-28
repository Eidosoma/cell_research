from __future__ import annotations

from collections import Counter
import unittest

import pandas as pd

from chimera.arrangements import (
    ARRANGEMENT_TYPES,
    augment_arrangement_results,
    build_arrangement_condition_matrix,
    generate_arrangement,
    load_s02_pair_results,
)
from chimera.mixtures import compact_json, load_ready_panel, run_mixture_condition


class TestE06InitialArrangements(unittest.TestCase):
    def test_arrangement_generators_preserve_counts(self) -> None:
        policy_ids = ["left", "right"]
        counts = [25, 75]
        for arrangement_type in ARRANGEMENT_TYPES:
            assigned, numeric_labels, descriptor = generate_arrangement(
                policy_ids,
                counts,
                arrangement_type,
                seed=123,
            )
            self.assertEqual(len(assigned), 100)
            self.assertEqual(len(numeric_labels), 100)
            self.assertEqual(Counter(assigned), {"left": 25, "right": 75})
            self.assertEqual(Counter(numeric_labels), {0: 25, 1: 75})
            self.assertEqual(descriptor["arrangementType"], arrangement_type)
            self.assertIn("interfaceCount", descriptor)
            self.assertIn("arrangementHash", descriptor)

    def test_condition_matrix_is_matched_by_base(self) -> None:
        ready = load_ready_panel()
        s02 = load_s02_pair_results()
        conditions, selected, descriptors = build_arrangement_condition_matrix(
            s02,
            ready,
            max_groups_per_category=1,
            max_total_groups=2,
        )
        self.assertFalse(selected.empty)
        self.assertEqual(len(conditions), 2 * 2 * len(ARRANGEMENT_TYPES))
        self.assertEqual(len(descriptors), len(conditions))
        for _, group in conditions.groupby("baseMatchId"):
            self.assertEqual(set(group["arrangementType"]), set(ARRANGEMENT_TYPES))
            self.assertEqual(group["policyIdsJson"].nunique(), 1)
            self.assertEqual(group["policyCountsJson"].nunique(), 1)
            self.assertEqual(group["valueSeed"].nunique(), 1)
            self.assertEqual(group["schedulerSeed"].nunique(), 1)
            self.assertEqual(group["tieBreakerSeed"].nunique(), 1)
            self.assertEqual(group["goalMode"].nunique(), 1)

    def test_explicit_arrangement_smoke_replay(self) -> None:
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
        assigned, labels, descriptor = generate_arrangement(["left", "right"], [4, 4], "graft_like", seed=7)
        condition = {
            "conditionId": "unit_s03_graft",
            "researchStepId": "S03",
            "implementationPrefix": "e06_s03",
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
            "arrangementType": "graft_like",
            "arrangementPolicyIdsJson": compact_json(assigned),
            "arrangementNumericLabelsJson": compact_json(labels),
            "arrangementDescriptorJson": compact_json(descriptor),
            "conditionHash": "unit",
            "s03Version": "unit",
            "seedIndex": 0,
            "valueSeed": 1,
            "arrangementSeed": 7,
            "schedulerSeed": 3,
            "tieBreakerSeed": 4,
            "goalMode": "same_goal_increasing",
        }
        lookup = {row["panelPolicyId"]: row for row in records}
        first = run_mixture_condition(condition, lookup, max_activations=400, max_swaps=200, max_comparisons=1000)
        second = run_mixture_condition(condition, lookup, max_activations=400, max_swaps=200, max_comparisons=1000)
        self.assertTrue(first["runSucceeded"])
        self.assertEqual(first["initialPolicyAssignmentHash"], second["initialPolicyAssignmentHash"])
        self.assertEqual(first["finalStateHash"], second["finalStateHash"])
        self.assertTrue(first["actualRatiosMatchTarget"])
        augmented = augment_arrangement_results(pd.DataFrame([first]))
        self.assertIn("finalMosaicClass", augmented.columns)
        self.assertEqual(int(augmented.iloc[0]["initialInterfaceCount"]), int(descriptor["interfaceCount"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
