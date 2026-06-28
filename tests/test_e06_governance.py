from __future__ import annotations

import unittest

import pandas as pd

from chimera.governance import (
    GOVERNANCE_GOAL_MODES,
    GOVERNANCE_VARIANTS,
    GovernanceConfig,
    GovernanceEventSimulator,
    build_governance_condition_matrix,
    governance_configs,
    governance_information_audit,
    governance_parameter_table,
    select_s09_contexts,
)
from chimera.interfaces import InterfaceRuleEventSimulator
from morphospace import BubblePolicy


def _s08_row(index: int, variant: str = "disabled_baseline") -> dict[str, object]:
    return {
        "s08ContextIndex": index,
        "conditionId": f"s08_c{index:03d}_{variant}",
        "s08SourceConditionId": f"s06_source_{index}",
        "interfaceRuleVariant": variant,
        "policyIdsJson": '["a","b"]',
        "policyCountsJson": "[2,2]",
        "targetCountsJson": '{"a":2,"b":2}',
        "targetProportionsJson": '{"a":0.5,"b":0.5}',
        "arrangementPolicyIdsJson": '["a","a","b","b"]',
        "arrangementNumericLabelsJson": "[0,0,1,1]",
        "arrangementDescriptorJson": "{}",
        "arrangementHash": "arr_hash",
        "goalReverseDirectionsJson": "[false,false,true,true]",
        "goalMode": "opposite_goal",
        "leftPanelPolicyId": "a",
        "rightPanelPolicyId": "b",
        "thirdPanelPolicyId": "",
        "ratioLabel": "50:50",
        "n": 4,
        "targetPolicyCount": 2,
        "arrangementType": "contiguous_patch",
        "arrangementTypeIndex": 0,
        "pairCategory": "unit",
        "valueProfile": "random_unique",
        "perturbationType": "none",
        "seedIndex": 0,
        "valueSeed": 1,
        "arrangementSeed": 2,
        "schedulerSeed": 3,
        "tieBreakerSeed": 4,
        "finalStateHash": "hash",
        "finalValuesJson": "[1,2,3,4]",
        "finalPanelPolicyIdsJson": '["a","a","b","b"]',
        "stopReason": "sorted",
        "swapCount": 1,
        "activationCount": 2,
        "s07MosaicClass": "layered",
        "finalInterfaceDensityS07": 0.33,
        "finalTargetQualityScore": 0.8,
        "dominanceProxyScore": 0.2,
        "runSucceeded": True,
        "s08ClassMatchesBaseline": True,
        "deltaFinalInterfaceDensityS07": 0.0,
        "deltaFinalTargetQualityScore": 0.0,
        "interfaceRuleBlockedSwaps": 0,
    }


class TestE06Governance(unittest.TestCase):
    def test_parameter_audit_flags_broad_organizer(self) -> None:
        params = governance_parameter_table(governance_configs())
        audit = governance_information_audit(params)
        self.assertEqual(set(params["governanceVariant"]), set(GOVERNANCE_VARIANTS))
        broad = audit[audit["governanceVariant"] == "organizer_global_upper_bound"].iloc[0]
        local_vote = audit[audit["governanceVariant"] == "local_voting_range1"].iloc[0]
        self.assertTrue(bool(broad["globalControlFlag"]))
        self.assertFalse(bool(local_vote["globalControlFlag"]))
        self.assertTrue(params["governanceConfigJson"].str.contains("e06_s09_governance_mechanisms.v1").all())

    def test_condition_matrix_includes_no_governance_per_goal(self) -> None:
        rows = [_s08_row(0)]
        rows.extend(
            {
                **_s08_row(0, variant),
                "s08ClassMatchesBaseline": False,
                "deltaFinalInterfaceDensityS07": 0.1,
                "deltaFinalTargetQualityScore": 0.03,
                "interfaceRuleBlockedSwaps": 1000,
            }
            for variant in ["adhesion_homotypic", "repulsion_heterotypic"]
        )
        selected = select_s09_contexts(pd.DataFrame(rows), max_contexts=1)
        matrix, params, goals = build_governance_condition_matrix(selected)
        self.assertEqual(len(matrix), len(GOVERNANCE_VARIANTS) * len(GOVERNANCE_GOAL_MODES))
        no_gov = matrix[matrix["governanceVariant"] == "no_governance_control"]
        self.assertEqual(set(no_gov["goalMode"]), set(GOVERNANCE_GOAL_MODES))
        self.assertEqual(set(params["governanceVariant"]), set(GOVERNANCE_VARIANTS))
        self.assertEqual(set(goals["goalMode"]), set(GOVERNANCE_GOAL_MODES))

    def test_no_governance_reproduces_interface_disabled_path(self) -> None:
        values = [3, 1, 2, 4]
        policies = [BubblePolicy()] * len(values)
        common = {
            "labels": [0, 0, 1, 1],
            "reverse_directions": [False, False, True, True],
            "scheduler_seed": 123,
            "tie_breaker_seed": 456,
            "condition_id": "unit_no_governance",
            "research_step_id": "S09",
            "trace_signal_activations": False,
            "trace_memory_activations": False,
            "auto_wrap_policies": False,
            "signal_config": "no_signal",
            "interface_rule_config": "disabled_baseline",
        }
        baseline = InterfaceRuleEventSimulator(
            values,
            policies,
            implementation="unit_interface_disabled",
            **common,
        )
        governed = GovernanceEventSimulator(
            values,
            policies,
            governance_config="no_governance_control",
            implementation="unit_no_governance",
            **common,
        )
        result_a = baseline.run(max_activations=200, max_swaps=100, max_comparisons=400)
        result_b = governed.run(max_activations=200, max_swaps=100, max_comparisons=400)
        self.assertEqual(result_a.final_values, result_b.final_values)
        self.assertEqual([cell.label for cell in baseline.cells], [cell.label for cell in governed.cells])
        self.assertEqual(result_a.swap_count, result_b.swap_count)
        self.assertEqual(governed.governance_interventions, 0)

    def test_governance_config_from_variant(self) -> None:
        config = GovernanceConfig.from_payload("local_voting_range1")
        self.assertTrue(config.enabled)
        self.assertEqual(config.influence_range, 1)
        self.assertTrue(config.local_information_only)


if __name__ == "__main__":
    unittest.main(verbosity=2)
