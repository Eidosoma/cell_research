from __future__ import annotations

import unittest

import pandas as pd

from chimera.grafts import (
    GRAFT_INTERFACE_VARIANTS,
    GRAFT_POSITIONS,
    GRAFT_SIZES,
    GRAFT_TIMINGS,
    build_graft_condition_matrix,
    graft_positions,
    select_s10_base_contexts,
    selected_governance_variants,
)


def _s09_no_governance_row(index: int, goal_mode: str, score: float = 0.5) -> dict[str, object]:
    return {
        "s09ContextIndex": index,
        "conditionId": f"s09_c{index:03d}_{goal_mode}_no_governance_control",
        "governanceVariant": "no_governance_control",
        "goalMode": goal_mode,
        "goalCompatibilityClass": goal_mode.replace("_goal", ""),
        "leftPanelPolicyId": f"host_{index}",
        "rightPanelPolicyId": f"donor_{index}",
        "policyIdsJson": f'["host_{index}","donor_{index}"]',
        "policyCountsJson": "[50,50]",
        "targetCountsJson": f'{{"host_{index}":50,"donor_{index}":50}}',
        "targetProportionsJson": f'{{"host_{index}":0.5,"donor_{index}":0.5}}',
        "arrangementPolicyIdsJson": f'["host_{index}","donor_{index}"]',
        "arrangementNumericLabelsJson": "[0,1]",
        "arrangementDescriptorJson": "{}",
        "arrangementHash": "arr",
        "n": 100,
        "targetPolicyCount": 2,
        "ratioLabel": "50:50",
        "arrangementType": "contiguous_patch",
        "arrangementTypeIndex": 0,
        "pairCategory": "unit_pair",
        "valueProfile": "random_unique",
        "perturbationType": "none",
        "valueSeed": 100 + index,
        "schedulerSeed": 200 + index,
        "tieBreakerSeed": 300 + index,
        "seedIndex": 0,
        "runSucceeded": True,
        "metricBoundary": "unit metric boundary",
        "s07MosaicClass": "mixed",
        "finalTargetQualityScore": score,
        "minPolicyGoalScore": 100.0 * score,
        "initialValuesJson": "[" + ",".join(str(i) for i in range(100, 0, -1)) + "]",
        "policyGoalMetadataJson": "{}",
        "conditionGoalMetadataJson": "{}",
        "goalReverseDirectionsJson": "[false,true]",
        "s09InterfaceSensitiveScore": 1.0 + index,
    }


class TestE06Grafts(unittest.TestCase):
    def test_graft_positions_are_contiguous_and_named(self) -> None:
        self.assertEqual(graft_positions(10, 3, "left_edge_patch"), [0, 1, 2])
        self.assertEqual(graft_positions(10, 4, "center_patch"), [3, 4, 5, 6])
        self.assertEqual(graft_positions(10, 2, "right_edge_patch"), [8, 9])
        with self.assertRaises(ValueError):
            graft_positions(10, 11, "center_patch")

    def test_selected_governance_variants_include_controls_and_broad_baseline(self) -> None:
        effect = pd.DataFrame(
            [
                {"governanceVariant": "local_voting_range1", "rankScore": 0.2, "broadControlLike": False},
                {"governanceVariant": "conflict_resolution_range1", "rankScore": 0.4, "broadControlLike": False},
                {"governanceVariant": "organizer_global_upper_bound", "rankScore": 1.0, "broadControlLike": True},
            ]
        )
        variants = selected_governance_variants(effect, max_local=2)
        self.assertEqual(variants[0], "no_governance_control")
        self.assertIn("conflict_resolution_range1", variants)
        self.assertIn("local_voting_range1", variants)
        self.assertEqual(variants[-1], "organizer_global_upper_bound")

    def test_condition_matrix_writes_matched_controls_and_insertions(self) -> None:
        s09 = pd.DataFrame(
            [
                _s09_no_governance_row(0, "same_goal", 0.7),
                _s09_no_governance_row(1, "opposite_goal", 0.3),
                _s09_no_governance_row(2, "partially_compatible", 0.5),
            ]
        )
        effect = pd.DataFrame(
            [
                {"goalMode": "same_goal", "governanceVariant": "local_voting_range1", "rankScore": 0.1, "rescueSuccessProxyRate": 0.2},
                {"goalMode": "opposite_goal", "governanceVariant": "conflict_resolution_range1", "rankScore": 0.3, "rescueSuccessProxyRate": 0.4},
                {"goalMode": "partially_compatible", "governanceVariant": "leader_cells_range2", "rankScore": 0.2, "rescueSuccessProxyRate": 0.3},
            ]
        )
        selected = select_s10_base_contexts(s09, effect, max_contexts=3)
        variants = ("no_governance_control", "conflict_resolution_range1", "organizer_global_upper_bound")
        matrix, pre, insertions = build_graft_condition_matrix(selected, governance_variants=variants)
        expected_pre = len(selected) * len(GRAFT_TIMINGS) * len(variants) * len(GRAFT_INTERFACE_VARIANTS)
        expected_graft = expected_pre * len(GRAFT_SIZES) * len(GRAFT_POSITIONS)
        expected_controls = expected_pre
        self.assertEqual(len(pre), expected_pre)
        self.assertEqual(len(insertions), expected_graft)
        self.assertEqual(len(matrix), expected_graft + expected_controls)
        graft_rows = matrix[matrix["graftApplied"].map(bool)]
        control_ids = set(matrix[~matrix["graftApplied"].map(bool)]["conditionId"])
        self.assertTrue(set(graft_rows["matchedNoGraftConditionId"]) <= control_ids)
        self.assertTrue(graft_rows["graftInsertionDescriptorJson"].str.contains("policy-identity patch").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
