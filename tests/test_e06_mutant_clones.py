from __future__ import annotations

import unittest

import pandas as pd

from chimera.governance import GovernanceEventSimulator
from chimera.mutants import (
    MUTANT_BEHAVIOR_VARIANTS,
    MUTANT_CLONE_SIZES,
    MUTANT_POSITION_NAMES,
    MutantCloneEventSimulator,
    build_mutant_condition_matrix,
    mutant_clone_positions,
    select_s11_contexts,
)
from morphospace import BubblePolicy


def _s10_control_row(index: int) -> dict[str, object]:
    host = f"host_{index}"
    donor = f"donor_{index}"
    values = "[" + ",".join(str(i) for i in range(100, 0, -1)) + "]"
    host_assignment = "[" + ",".join(f'"{host}"' for _ in range(100)) + "]"
    return {
        "conditionId": f"s10_control_{index}",
        "matchedNoGraftConditionId": f"s10_control_{index}",
        "s10ConditionKind": "matched_no_graft_control",
        "graftApplied": False,
        "graftOutcomeClass": "matched_no_graft_control",
        "hostPolicyId": host,
        "donorPolicyId": donor,
        "leftPanelPolicyId": host,
        "rightPanelPolicyId": donor,
        "governanceVariant": "no_governance_control",
        "interfaceRuleVariant": "disabled_baseline",
        "goalMode": "same_goal",
        "goalCompatibilityClass": "same",
        "policyIdsJson": f'["{host}","{donor}"]',
        "preGraftFinalValuesJson": values,
        "preGraftFinalPanelPolicyIdsJson": host_assignment,
        "postGraftInitialPanelPolicyIdsJson": host_assignment,
        "initialValuesJson": values,
        "finalValuesJson": values,
        "initialPanelPolicyIdsJson": host_assignment,
        "finalPanelPolicyIdsJson": host_assignment,
        "targetCountsJson": f'{{"{host}":100}}',
        "targetProportionsJson": f'{{"{host}":1.0}}',
        "finalStateHash": f"hash_{index}",
        "finalTargetQualityScore": 0.8,
        "minPolicyGoalScore": 80.0,
        "meanPolicyGoalScore": 80.0,
        "finalSortednessPercent": 0.0,
        "s07MosaicClass": "layered",
        "runSucceeded": True,
        "metricBoundary": "unit metric boundary",
        "n": 100,
        "schedulerSeed": 100 + index,
        "tieBreakerSeed": 200 + index,
        "graftStartIndex": 10,
        "broadControlLike": False,
        "usesTargetMap": False,
        "preGraftStateId": f"pre_{index}",
        "preGraftTimingLabel": "early",
        "ratioLabel": "100:0",
        "pairCategory": "unit",
        "valueProfile": "random_unique",
        "perturbationType": "none",
        "seedIndex": 0,
        "stopReason": "max_step_cap",
        "swapCount": 1,
        "activationCount": 2,
    }


def _s10_graft_row(index: int, outcome: str, governance: str, interface: str) -> dict[str, object]:
    base = _s10_control_row(index)
    host = str(base["hostPolicyId"])
    donor = str(base["donorPolicyId"])
    graft_assignment = "[" + ",".join([f'"{donor}"' if i < 10 else f'"{host}"' for i in range(100)]) + "]"
    return {
        **base,
        "conditionId": f"s10_graft_{index}_{outcome}_{governance}_{interface}",
        "matchedNoGraftConditionId": f"s10_control_{index}",
        "s10ConditionKind": "graft",
        "graftApplied": True,
        "graftOutcomeClass": outcome,
        "governanceVariant": governance,
        "interfaceRuleVariant": interface,
        "postGraftInitialPanelPolicyIdsJson": graft_assignment,
        "initialPanelPolicyIdsJson": graft_assignment,
        "finalPanelPolicyIdsJson": graft_assignment,
        "targetCountsJson": f'{{"{donor}":10,"{host}":90}}',
        "targetProportionsJson": f'{{"{donor}":0.1,"{host}":0.9}}',
        "graftSize": 10,
        "graftPositionName": "left_edge_patch",
        "graftStartIndex": 10,
        "graftChangedMosaicClass": outcome != "segregated_patch",
        "deltaVsNoGraftFinalTargetQualityScore": -0.1 if outcome == "absorbed_or_mixed" else 0.02,
        "deltaVsNoGraftMinPolicyGoalScore": -10.0 if outcome == "absorbed_or_mixed" else 2.0,
        "graftDonorFinalAggregation": 0.9,
        "graftDonorFinalSpreadFraction": 0.2,
        "broadControlLike": governance == "organizer_global_upper_bound",
        "usesTargetMap": governance == "organizer_global_upper_bound",
    }


class TestE06MutantClones(unittest.TestCase):
    def test_mutant_clone_positions_are_contiguous(self) -> None:
        self.assertEqual(mutant_clone_positions(10, 3, "left_edge_clone"), [0, 1, 2])
        self.assertEqual(mutant_clone_positions(10, 4, "center_clone"), [3, 4, 5, 6])
        self.assertEqual(mutant_clone_positions(10, 2, "s10_graft_site", source_start=7), [7, 8])
        with self.assertRaises(ValueError):
            mutant_clone_positions(10, 0, "center_clone")

    def test_select_contexts_and_condition_matrix_use_s10_controls(self) -> None:
        rows: list[dict[str, object]] = []
        for index in range(4):
            rows.append(_s10_control_row(index))
        rows.extend(
            [
                _s10_graft_row(0, "segregated_patch", "no_governance_control", "disabled_baseline"),
                _s10_graft_row(1, "absorbed_or_mixed", "conflict_resolution_range1", "combined_self_boundary"),
                _s10_graft_row(2, "equilibrated_mosaic", "local_voting_range1", "disabled_baseline"),
                _s10_graft_row(3, "segregated_patch", "organizer_global_upper_bound", "combined_self_boundary"),
            ]
        )
        selected = select_s11_contexts(pd.DataFrame(rows), max_contexts=4)
        matrix, params, lineage = build_mutant_condition_matrix(selected)
        expected_mutants = len(selected) * len(MUTANT_CLONE_SIZES) * len(MUTANT_POSITION_NAMES) * len(MUTANT_BEHAVIOR_VARIANTS)
        self.assertEqual(len(matrix), expected_mutants + len(selected))
        self.assertEqual(len(lineage), expected_mutants)
        self.assertEqual(set(MUTANT_BEHAVIOR_VARIANTS), set(params["mutantBehaviorVariant"]))
        controls = matrix[matrix["s11ConditionKind"] == "matched_no_mutant_control"]
        self.assertTrue(controls["simulationConditionId"].astype(str).str.startswith("s10_control_").all())
        self.assertTrue(matrix["analogyCaveat"].astype(str).str.contains("computational analogy").all())

    def test_mutant_hook_changes_selfish_clone_but_neutral_matches_governance(self) -> None:
        values = [1, 2, 3, 4, 5]
        policies = [BubblePolicy()] * len(values)
        common = {
            "labels": [0] * len(values),
            "reverse_directions": [False] * len(values),
            "scheduler_seed": 1,
            "tie_breaker_seed": 2,
            "condition_id": "unit",
            "research_step_id": "S11",
            "trace_signal_activations": False,
            "trace_memory_activations": False,
            "auto_wrap_policies": False,
            "signal_config": "no_signal",
            "interface_rule_config": "disabled_baseline",
            "governance_config": "no_governance_control",
        }
        neutral = GovernanceEventSimulator(values, policies, implementation="neutral", **common)
        mutant = MutantCloneEventSimulator(
            values,
            policies,
            implementation="mutant",
            mutant_clone_config="selfish_position_center",
            mutant_cell_ids=[0],
            **common,
        )
        neutral.step(forced_cell_id=0)
        mutant.step(forced_cell_id=0)
        self.assertEqual(mutant.positions_by_id[0], 1)
        self.assertGreater(mutant.mutant_selfish_proposals, 0)
        self.assertEqual(neutral.positions_by_id[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
