from __future__ import annotations

import unittest

import pandas as pd

from chimera.interfaces import (
    INTERFACE_RULE_VARIANTS,
    InterfaceRuleConfig,
    InterfaceRuleEventSimulator,
    build_interface_condition_matrix,
    evaluate_interface_rule,
    interface_rule_configs,
    interface_rule_parameter_table,
    select_s08_baseline_contexts,
)
from morphospace import BubblePolicy


class TestE06InterfaceRules(unittest.TestCase):
    def test_disabled_interface_rule_allows_any_in_bounds_swap(self) -> None:
        decision = evaluate_interface_rule(
            [0, 1, 0, 1],
            1,
            2,
            InterfaceRuleConfig("disabled_baseline", False, "disabled"),
        )
        self.assertTrue(decision["allowed"])
        self.assertEqual(decision["blockReason"], "disabled")

    def test_enabled_rules_block_expected_boundary_cases(self) -> None:
        labels = [0, 0, 1, 1]
        adhesion = evaluate_interface_rule(labels, 1, 2, "adhesion_homotypic")
        repulsion = evaluate_interface_rule(labels, 1, 2, "repulsion_heterotypic")
        recognition = evaluate_interface_rule(labels, 1, 2, "recognition_self_nonself")
        self.assertTrue(adhesion["blocked"])
        self.assertTrue(repulsion["blocked"])
        self.assertTrue(recognition["blocked"])
        self.assertIn("adhesion", adhesion["blockReason"])
        self.assertIn("repulsion", repulsion["blockReason"])
        self.assertIn("recognition", recognition["blockReason"])

    def test_disabled_simulator_reproduces_signal_path_without_blocks(self) -> None:
        values = [3, 1, 2, 4]
        policies = [BubblePolicy()] * len(values)
        common = {
            "labels": [0, 0, 1, 1],
            "scheduler_seed": 123,
            "tie_breaker_seed": 456,
            "condition_id": "unit_disabled",
            "research_step_id": "S08",
            "trace_signal_activations": False,
            "trace_memory_activations": False,
            "auto_wrap_policies": False,
            "signal_config": "no_signal",
        }
        sim_a = InterfaceRuleEventSimulator(
            values,
            policies,
            interface_rule_config="disabled_baseline",
            implementation="unit_disabled",
            **common,
        )
        sim_b = InterfaceRuleEventSimulator(
            values,
            policies,
            interface_rule_config="disabled_baseline",
            implementation="unit_disabled_2",
            **common,
        )
        result_a = sim_a.run(max_activations=200, max_swaps=100, max_comparisons=400)
        result_b = sim_b.run(max_activations=200, max_swaps=100, max_comparisons=400)
        self.assertEqual(result_a.final_values, result_b.final_values)
        self.assertEqual([cell.label for cell in sim_a.cells], [cell.label for cell in sim_b.cells])
        self.assertEqual(sim_a.interface_rule_blocked_swaps, 0)
        self.assertEqual(sim_a.interface_rule_evaluations, 0)

    def test_parameter_table_logs_all_variants(self) -> None:
        params = interface_rule_parameter_table(interface_rule_configs())
        self.assertEqual(set(params["interfaceRuleVariant"]), set(INTERFACE_RULE_VARIANTS))
        self.assertTrue(params["interfaceRuleConfigJson"].str.contains("e06_s08_interface_rules.v1").all())
        self.assertTrue(params["description"].str.len().gt(0).all())

    def test_condition_matrix_crosses_selected_contexts_and_rules(self) -> None:
        s07 = pd.DataFrame(
            [
                {
                    "conditionId": f"c_{klass}",
                    "s07MosaicClass": klass,
                    "dominanceProxyScore": 0.5,
                    "policyIdsJson": "[\"a\",\"b\"]",
                    "policyCountsJson": "[2,2]",
                    "targetCountsJson": "{\"a\":2,\"b\":2}",
                    "arrangementPolicyIdsJson": "[\"a\",\"a\",\"b\",\"b\"]",
                    "goalReverseDirectionsJson": "[false,false,true,true]",
                    "valueSeed": 1,
                    "schedulerSeed": 2,
                    "tieBreakerSeed": 3,
                    "n": 4,
                    "finalStateHash": "hash",
                    "finalValuesJson": "[1,2,3,4]",
                    "finalPanelPolicyIdsJson": "[\"a\",\"a\",\"b\",\"b\"]",
                    "stopReason": "sorted",
                    "swapCount": 1,
                    "activationCount": 2,
                }
                for klass in ["homogeneous", "patchy", "layered", "polarized", "oscillatory", "frozen_conflict", "mixed"]
            ]
        )
        priority = pd.DataFrame(
            {
                "conditionId": s07["conditionId"],
                "s07PriorityRank": range(1, len(s07) + 1),
                "s07PriorityScore": [10.0] * len(s07),
                "s07PriorityTagsJson": ["[]"] * len(s07),
            }
        )
        selected = select_s08_baseline_contexts(s07, priority, max_contexts=7)
        matrix, params = build_interface_condition_matrix(selected)
        self.assertEqual(len(selected), 7)
        self.assertEqual(len(matrix), 7 * len(INTERFACE_RULE_VARIANTS))
        self.assertEqual(set(matrix["interfaceRuleVariant"]), set(INTERFACE_RULE_VARIANTS))
        self.assertEqual(set(params["interfaceRuleVariant"]), set(INTERFACE_RULE_VARIANTS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
