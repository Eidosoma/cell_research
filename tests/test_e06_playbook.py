from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from chimera.panel import CLAIM_BOUNDARY
from chimera.playbook import (
    build_caveat_register,
    build_evidence_traceability,
    build_phase_diagram_summary,
    build_recommendations,
    validate_s15_outputs,
)


def _tables() -> dict[str, pd.DataFrame]:
    return {
        "mixture_ratio_pair_summary": pd.DataFrame(
            [
                {
                    "pairCategory": "original_pair",
                    "ratioLabel": "50:50",
                    "meanFinalSortednessPercent": 91.0,
                    "meanFinalAggregation": 0.7,
                    "completionRate": 0.5,
                }
            ]
        ),
        "arrangement_summary": pd.DataFrame(
            [
                {
                    "arrangementType": "random",
                    "conditionCount": 2,
                    "meanFinalAggregation": 0.6,
                    "meanInterfaceStabilityScore": 0.8,
                    "meanPolicyPositionChangeFraction": 0.2,
                }
            ]
        ),
        "compatibility_mode_summary": pd.DataFrame(
            [
                {
                    "goalMode": "same_goal",
                    "goalCompatibilityClass": "same",
                    "conditionCount": 4,
                    "meanCompatibilityCompositeScore": 0.82,
                    "meanFinalTargetQualityScore": 0.78,
                    "meanMinTargetQualityScore": 0.68,
                    "meanMutualInterferenceScore": 0.0,
                    "highCompatibilityRate": 0.75,
                },
                {
                    "goalMode": "partially_compatible",
                    "goalCompatibilityClass": "partial",
                    "conditionCount": 4,
                    "meanCompatibilityCompositeScore": 0.84,
                    "meanFinalTargetQualityScore": 0.82,
                    "meanMinTargetQualityScore": 0.74,
                    "meanMutualInterferenceScore": 0.01,
                    "highCompatibilityRate": 0.95,
                },
                {
                    "goalMode": "opposite_goal",
                    "goalCompatibilityClass": "opposite",
                    "conditionCount": 4,
                    "meanCompatibilityCompositeScore": 0.64,
                    "meanFinalTargetQualityScore": 0.60,
                    "meanMinTargetQualityScore": 0.36,
                    "meanMutualInterferenceScore": 0.27,
                    "highCompatibilityRate": 0.37,
                },
            ]
        ),
        "dominance_pair_summary": pd.DataFrame(
            [
                {
                    "s06PairKey": "a||b",
                    "conditionCount": 8,
                    "modalWinnerPolicyId": "a",
                    "modalWinnerShare": 0.75,
                    "contextStableWinner": True,
                    "contextDependencyClass": "stable_dominance_proxy",
                    "meanDominanceProxyScore": 0.7,
                },
                {
                    "s06PairKey": "c||d",
                    "conditionCount": 8,
                    "modalWinnerPolicyId": "c",
                    "modalWinnerShare": 0.5,
                    "contextStableWinner": False,
                    "contextDependencyClass": "context_dependent_dominance_proxy",
                    "meanDominanceProxyScore": 0.5,
                },
            ]
        ),
        "mosaic_class_summary": pd.DataFrame(
            [
                {
                    "s07MosaicClass": "mixed",
                    "conditionCount": 10,
                    "meanFinalAggregation": 0.65,
                    "meanFinalInterfaceDensity": 0.35,
                    "meanDominanceProxyScore": 0.4,
                    "resourceCappedRate": 0.8,
                }
            ]
        ),
        "interface_effect_summary": pd.DataFrame(
            [
                {
                    "interfaceRuleVariant": "disabled_baseline",
                    "conditionCount": 2,
                    "classMatchBaselineRate": 1.0,
                    "meanDeltaFinalAggregation": 0.0,
                    "meanDeltaFinalTargetQualityScore": 0.0,
                },
                {
                    "interfaceRuleVariant": "adhesion_homotypic",
                    "conditionCount": 2,
                    "classMatchBaselineRate": 0.25,
                    "meanDeltaFinalAggregation": 0.1,
                    "meanDeltaFinalTargetQualityScore": 0.02,
                },
            ]
        ),
        "governance_effect_summary": pd.DataFrame(
            [
                {
                    "goalMode": "opposite_goal",
                    "governanceVariant": "local_voting_range1",
                    "conditionCount": 2,
                    "rescueSuccessProxyRate": 0.2,
                    "meanDeltaFinalTargetQualityScore": 0.01,
                    "meanGovernanceCostProxy": 0.08,
                    "rankScore": 0.19,
                    "broadControlLike": False,
                },
                {
                    "goalMode": "opposite_goal",
                    "governanceVariant": "broad_organizer",
                    "conditionCount": 2,
                    "rescueSuccessProxyRate": 0.5,
                    "meanDeltaFinalTargetQualityScore": 0.04,
                    "meanGovernanceCostProxy": 0.3,
                    "rankScore": 0.3,
                    "broadControlLike": True,
                },
            ]
        ),
        "graft_outcome_summary": pd.DataFrame(
            [{"graftOutcomeClass": "absorbed_or_mixed", "conditionCount": 3, "meanDeltaTargetQualityVsNoGraft": -0.1, "meanDonorSpreadFraction": 0.8}]
        ),
        "mutant_outcome_summary": pd.DataFrame(
            [{"mutantOutcomeClass": "disruptive_takeover_proxy", "conditionCount": 3, "meanDeltaTargetQualityVsNoMutant": -0.4, "meanCloneSpreadFraction": 0.3}]
        ),
        "history_outcome_summary": pd.DataFrame(
            [
                {
                    "historyProtocol": "late_staged_introduction_reset",
                    "conditionCount": 3,
                    "meanHistoryEffectMagnitudeScore": 0.3,
                    "meanHistoryDisruptionProxyScore": 0.5,
                    "meanDeltaTargetQualityVsSimultaneous": 0.1,
                    "mosaicShiftRateVsSimultaneous": 0.2,
                }
            ]
        ),
        "heldout_intervention_validation": pd.DataFrame(
            [
                {
                    "historyProtocol": "late_staged_introduction_reset",
                    "trainConditionCount": 9,
                    "heldoutConditionCount": 3,
                    "heldoutRescueRate": 0.67,
                    "heldoutMeanMinimalInterventionScore": 0.54,
                    "recommendedForS15PlaybookCandidate": False,
                    "heldoutSeedValidationStatus": "not_feasible_single_s12_seed",
                    "heldoutRatioValidationStatus": "not_feasible_single_s12_ratio",
                }
            ]
        ),
    }


class TestE06Playbook(unittest.TestCase):
    def test_recommendations_include_s14_restricted_caveat(self) -> None:
        recommendations = build_recommendations(_tables())
        s14 = recommendations[recommendations["recommendationId"] == "R09_s14_interventions_are_constrained_candidates"].iloc[0]
        self.assertTrue(bool(s14["s14RestrictedCandidate"]))
        caveats = str(s14["caveatsJson"]).lower()
        self.assertIn("seed", caveats)
        self.assertIn("ratio", caveats)
        self.assertIn("held-out", caveats)
        self.assertGreater(len(json.loads(s14["evidenceArtifactsJson"])), 0)

    def test_phase_summary_covers_synthesis_axes(self) -> None:
        phase = build_phase_diagram_summary(_tables())
        axes = set(phase["phaseAxis"])
        self.assertIn("minimal_intervention", axes)
        self.assertIn("governance", axes)
        self.assertIn("goal_compatibility", axes)
        self.assertTrue(phase["claimBoundary"].astype(str).str.contains("Computational").all())

    def test_validation_passes_for_linked_recommendations(self) -> None:
        tables = _tables()
        recommendations = build_recommendations(tables)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evidence = tmp_path / "evidence.parquet"
            evidence.write_text("evidence", encoding="utf-8")
            recommendations = recommendations.copy()
            recommendations["evidenceArtifactsJson"] = json.dumps([str(evidence)])
            recommendations["evidenceStepIdsJson"] = json.dumps(["S14"])
            traceability = build_evidence_traceability(recommendations)
            playbook = tmp_path / "playbook.md"
            playbook.write_text(CLAIM_BOUNDARY, encoding="utf-8")
            figure = tmp_path / "figure.png"
            figure.write_text("figure", encoding="utf-8")
            bundle = tmp_path / "manifest.json"
            bundle.write_text("{}", encoding="utf-8")
            validation = validate_s15_outputs(
                source_index=pd.DataFrame({"exists": [True, True]}),
                status_index=pd.DataFrame(
                    {
                        "success": [True] * 14,
                        "validationResult": ["passed"] * 14,
                    }
                ),
                recommendations=recommendations,
                traceability=traceability,
                caveats=build_caveat_register(),
                phase_summary=build_phase_diagram_summary(tables),
                playbook_path=playbook,
                figure_paths=[figure],
                bundle_paths=[bundle],
            )
            self.assertTrue(validation["success"].all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
