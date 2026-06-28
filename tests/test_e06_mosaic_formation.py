from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from chimera.mixtures import compact_json
from chimera.mosaics import (
    REQUESTED_MOSAIC_CLASSES,
    build_hand_labeled_exemplars,
    build_priority_candidates,
    classify_mosaic_formations,
    mosaic_rule_definitions,
)


def _row(condition_id: str, final_ids: list[str], **overrides: object) -> dict[str, object]:
    policy_ids = ["a", "b"]
    base = {
        "conditionId": condition_id,
        "s06PairKey": "a||b",
        "pairCategory": "unit_pair",
        "ratioLabel": "50:50",
        "arrangementType": "random",
        "valueProfile": "random_unique",
        "perturbationType": "none",
        "policyIdsJson": compact_json(policy_ids),
        "initialPanelPolicyIdsJson": compact_json(["a", "b"] * (len(final_ids) // 2)),
        "finalPanelPolicyIdsJson": compact_json(final_ids),
        "runSucceeded": True,
        "completed": False,
        "stopReason": "max_activation_cap",
        "finalStateClass": "resource_capped_partial",
        "finalAggregation": 0.5,
        "finalInterfaceDensity": 0.5,
        "finalTargetQualityScore": 0.5,
        "dominanceProxyScore": 0.5,
        "winnerPolicyId": "a",
        "winnerConfidenceClass": "strong",
        "oscillationRiskProxy": 1.0,
        "metricBoundary": "unit-test dominance is a proxy",
        "metricBoundaryCaveatFlagsJson": compact_json(["opposite_goal_reverse_behavior"]),
    }
    base.update(overrides)
    return base


class TestE06MosaicFormation(unittest.TestCase):
    def test_rule_table_covers_requested_classes_in_order(self) -> None:
        rules = mosaic_rule_definitions()
        self.assertEqual(set(rules["s07MosaicClass"]), set(REQUESTED_MOSAIC_CLASSES))
        self.assertEqual(rules["ruleOrder"].tolist(), list(range(1, len(REQUESTED_MOSAIC_CLASSES) + 1)))
        self.assertTrue(rules["metricBoundary"].str.len().gt(0).all())

    def test_classifier_separates_mosaic_classes(self) -> None:
        rows = [
            _row(
                "homogeneous",
                ["a", "b"] * 5,
                completed=True,
                stopReason="sorted",
                finalStateClass="sorted",
                finalTargetQualityScore=0.9,
                dominanceProxyScore=0.2,
                oscillationRiskProxy=0.0,
            ),
            _row(
                "oscillatory",
                ["a", "b"] * 5,
                completed=False,
                stopReason="max_activation_cap",
                finalTargetQualityScore=0.4,
                dominanceProxyScore=0.9,
                oscillationRiskProxy=1.0,
            ),
            _row(
                "frozen_conflict",
                ["a", "b"] * 5,
                completed=False,
                stopReason="no_cell_can_move_after_two_checks",
                finalTargetQualityScore=0.5,
                dominanceProxyScore=0.8,
                oscillationRiskProxy=0.2,
            ),
            _row(
                "layered",
                ["a"] * 25 + ["b"] * 75,
                completed=False,
                stopReason="max_activation_cap",
                finalTargetQualityScore=0.8,
                dominanceProxyScore=0.8,
                oscillationRiskProxy=1.0,
            ),
            _row(
                "polarized",
                ["a"] * 40 + ["b"] * 10 + ["a"] * 10 + ["b"] * 40,
                completed=False,
                stopReason="max_activation_cap",
                finalTargetQualityScore=0.8,
                dominanceProxyScore=0.8,
                oscillationRiskProxy=1.0,
            ),
            _row(
                "patchy",
                ["a"] * 40 + ["b"] * 40 + ["a"] * 20,
                completed=False,
                stopReason="max_activation_cap",
                finalTargetQualityScore=0.8,
                dominanceProxyScore=0.8,
                oscillationRiskProxy=1.0,
            ),
            _row(
                "mixed",
                ["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"],
                completed=True,
                stopReason="sorted",
                finalStateClass="sorted",
                finalTargetQualityScore=0.5,
                dominanceProxyScore=0.1,
                oscillationRiskProxy=0.0,
            ),
        ]
        classified = classify_mosaic_formations(pd.DataFrame(rows))
        observed = dict(zip(classified["conditionId"], classified["s07MosaicClass"], strict=True))
        self.assertEqual(observed["homogeneous"], "homogeneous")
        self.assertEqual(observed["oscillatory"], "oscillatory")
        self.assertEqual(observed["frozen_conflict"], "frozen_conflict")
        self.assertEqual(observed["layered"], "layered")
        self.assertEqual(observed["polarized"], "polarized")
        self.assertEqual(observed["patchy"], "patchy")
        self.assertEqual(observed["mixed"], "mixed")
        self.assertTrue(classified["s07MosaicCaveatsJson"].str.contains("metric_boundary_caveats_inherited_from_s06").all())

    def test_priority_candidates_include_context_and_unresolved_tags(self) -> None:
        classified = classify_mosaic_formations(
            pd.DataFrame(
                [
                    _row("context", ["a", "b"] * 5, winnerPolicyId="a", winnerConfidenceClass="strong"),
                    _row("unresolved", ["a", "b"] * 5, winnerPolicyId="balanced_or_unresolved", winnerConfidenceClass="unresolved"),
                ]
            ),
            pair_summary=pd.DataFrame(
                [
                    {
                        "s06PairKey": "a||b",
                        "contextDependencyClass": "context_dependent_dominance_proxy",
                        "modalWinnerPolicyId": "a",
                        "modalWinnerShare": 0.5,
                        "unresolvedCount": 2,
                        "meanDominanceProxyScore": 0.6,
                        "meanOscillationRiskProxy": 0.8,
                    }
                ]
            ),
        )
        priority = build_priority_candidates(classified)
        tags = " ".join(priority["s07PriorityTagsJson"].astype(str))
        self.assertIn("context_dependent_pair", tags)
        self.assertIn("unresolved_winner", tags)

    @unittest.skipUnless(Path("/artifacts/results/e06_dominance_mosaics.parquet").exists(), "S06 artifacts unavailable")
    def test_hand_labeled_exemplars_match_s06_artifacts(self) -> None:
        s06 = pd.read_parquet("/artifacts/results/e06_dominance_mosaics.parquet")
        pair_summary = pd.read_parquet("/artifacts/research_steps/S06/dominance_pair_summary.parquet")
        classified = classify_mosaic_formations(s06, pair_summary)
        exemplars = build_hand_labeled_exemplars(classified)
        self.assertTrue(exemplars["conditionFound"].all())
        self.assertGreaterEqual(float(exemplars["classifierAgreement"].mean()), 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
