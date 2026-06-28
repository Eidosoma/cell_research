from __future__ import annotations

import unittest

import pandas as pd

from chimera.mixtures import (
    PAIR_RATIOS,
    build_condition_matrix,
    load_ready_panel,
    policy_assignment,
    run_mixture_condition,
    run_smoke_replays,
    select_priority_policy_ids,
)


class TestE06MixtureRatios(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ready = load_ready_panel().head(12).copy()

    def test_priority_selection_uses_ready_ids(self) -> None:
        priority = select_priority_policy_ids(load_ready_panel())
        ready_ids = set(load_ready_panel()["panelPolicyId"])
        selected = {pid for ids in priority.values() for pid in ids}
        self.assertTrue(selected)
        self.assertTrue(selected <= ready_ids)

    def test_condition_matrix_ratios_and_seeds(self) -> None:
        ready = load_ready_panel()
        conditions, selection = build_condition_matrix(ready, seed_count=1)
        self.assertFalse(conditions.empty)
        self.assertTrue(set(f"{a}:{b}" for a, b in PAIR_RATIOS) <= set(conditions["ratioLabel"]))
        pair = conditions[conditions["mixtureKind"] == "pair"].iloc[0].to_dict()
        assigned, labels, _ = policy_assignment(pair)
        self.assertEqual(len(assigned), 100)
        self.assertEqual(len(labels), 100)
        self.assertEqual(sum(pair["ratioLabel"] == label for label in conditions["ratioLabel"]), len(conditions[conditions["ratioLabel"] == pair["ratioLabel"]]))
        self.assertTrue({"valueSeed", "arrangementSeed", "schedulerSeed", "tieBreakerSeed"}.issubset(conditions.columns))
        self.assertIn("full_ready_panel", set(selection["priorityGroup"]))

    def test_mixture_smoke_replay(self) -> None:
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
        lookup = {row["panelPolicyId"]: row for row in records}
        condition = {
            "conditionId": "unit_pair",
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
            "conditionHash": "unit",
            "s02Version": "unit",
            "seedIndex": 0,
            "valueSeed": 1,
            "arrangementSeed": 2,
            "schedulerSeed": 3,
            "tieBreakerSeed": 4,
        }
        first = run_mixture_condition(condition, lookup, max_activations=400, max_swaps=200, max_comparisons=1000)
        second = run_mixture_condition(condition, lookup, max_activations=400, max_swaps=200, max_comparisons=1000)
        self.assertTrue(first["runSucceeded"])
        self.assertEqual(first["finalStateHash"], second["finalStateHash"])
        self.assertTrue(first["actualRatiosMatchTarget"])
        self.assertTrue(first["valueCountsConserved"])
        self.assertTrue(first["policyCountsConserved"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
