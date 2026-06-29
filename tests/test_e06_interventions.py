from __future__ import annotations

import unittest

import pandas as pd

from chimera.interventions import (
    NO_INTERVENTION_PROTOCOL,
    build_intervention_recipes,
    build_intervention_search_results,
    summarize_intervention_effects,
    validation_checks,
)


def _s12_row(group: str, protocol: str, *, split_score: float = 0.0, rescue: bool = False) -> dict[str, object]:
    target = 0.5 + split_score
    baseline_target = 0.5
    min_goal = 40.0 + 100.0 * max(split_score, 0.0)
    return {
        "conditionId": f"{group}_{protocol}",
        "s12SourceS11ConditionId": group,
        "historyProtocol": protocol,
        "historyOutcomeClass": "history_rescue_proxy" if rescue else "history_neutral_within_threshold",
        "finalTargetQualityScore": target,
        "minPolicyGoalScore": min_goal,
        "finalAggregation": 0.5 + split_score / 10.0,
        "dominanceProxyScore": 0.2,
        "historyEffectMagnitudeScore": 0.2 if rescue else 0.05,
        "historyDisruptionProxyScore": 0.0,
        "simultaneousFinalTargetQualityScore": baseline_target,
        "simultaneousMinPolicyGoalScore": 40.0,
        "simultaneousFinalAggregation": 0.5,
        "simultaneousDominanceProxyScore": 0.2,
        "simultaneousMutantCloneSpreadFraction": 0.2,
        "mutantCloneSpreadFraction": 0.2,
        "deltaVsSimultaneousFinalTargetQualityScore": target - baseline_target,
        "deltaVsSimultaneousMinPolicyGoalScore": min_goal - 40.0,
        "deltaVsSimultaneousFinalAggregation": split_score / 10.0,
        "historyChangedMosaicClassVsSimultaneous": protocol != NO_INTERVENTION_PROTOCOL and rescue,
        "s07MosaicClass": "layered" if protocol == NO_INTERVENTION_PROTOCOL else ("patchy" if rescue else "layered"),
        "finalStateHash": f"hash_{group}_{protocol}",
        "activationCount": 3500,
        "historyTotalActivationCap": 3500,
        "preExposureActivationCap": 500 if "staged" in protocol else 0,
        "transientPerturbationStartActivation": 0 if "transient" in protocol else -1,
        "transientPerturbationEndActivation": 500 if "transient" in protocol else -1,
        "governanceInformationAccessScore": 0.0,
        "broadControlLike": False,
        "ratioLabel": "75:25",
        "seedIndex": 0,
        "claimBoundary": "Computational local-policy validation only",
    }


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for group in ["g0", "g1", "g2", "g3"]:
        rows.append(_s12_row(group, NO_INTERVENTION_PROTOCOL))
        rows.append(_s12_row(group, "matched_no_mutant_replay", split_score=0.4))
        rows.append(_s12_row(group, "late_staged_introduction_reset", split_score=0.2, rescue=True))
        rows.append(_s12_row(group, "early_transient_rule_release_reset", split_score=0.05, rescue=group in {"g0", "g1"}))
    hypotheses = pd.DataFrame(
        [
            {
                "hypothesisId": "h1",
                "hypothesis": "late_staged_introduction_reset may increase rescueSuccessProxyTarget",
                "causalCaveat": "not biological causal proof",
            }
        ]
    )
    contrasts = pd.DataFrame(
        [
            {
                "comparisonName": "late_staged_reset_minus_simultaneous",
                "treatmentProtocol": "late_staged_introduction_reset",
                "metricName": "rescueSuccessProxyTarget",
                "meanTreatmentMinusReference": 0.5,
            },
            {
                "comparisonName": "late_staged_reset_minus_simultaneous",
                "treatmentProtocol": "late_staged_introduction_reset",
                "metricName": "finalTargetQualityScore",
                "meanTreatmentMinusReference": 0.2,
            },
        ]
    )
    split = pd.DataFrame(
        [
            {"conditionId": f"{group}_{protocol}", "s12SourceS11ConditionId": group, "split": "test" if group == "g3" else "train"}
            for group in ["g0", "g1", "g2", "g3"]
            for protocol in [
                NO_INTERVENTION_PROTOCOL,
                "matched_no_mutant_replay",
                "late_staged_introduction_reset",
                "early_transient_rule_release_reset",
            ]
        ]
    )
    importance = pd.DataFrame(
        [
            {"sourceFeature": "historyProtocol", "totalImportance": 1.0},
            {"sourceFeature": "cloneIntroductionActivation", "totalImportance": 0.5},
            {"sourceFeature": "memoryCarryoverMode", "totalImportance": 0.2},
        ]
    )
    return pd.DataFrame(rows), hypotheses, contrasts, split, importance


class TestE06Interventions(unittest.TestCase):
    def test_recipes_include_s13_evidence_and_costs(self) -> None:
        s12, hypotheses, contrasts, _, importance = _inputs()
        recipes = build_intervention_recipes(s12, hypotheses, contrasts, importance)
        staged = recipes[recipes["historyProtocol"] == "late_staged_introduction_reset"].iloc[0]
        self.assertTrue(bool(staged["searchEligible"]))
        self.assertGreater(float(staged["interventionCostScore"]), 0.0)
        self.assertGreater(float(staged["s13PredictorSupportScore"]), 0.0)
        self.assertIn("late_staged_reset_minus_simultaneous", str(staged["sourceS13ComparisonNamesJson"]))

    def test_search_results_have_paired_controls_and_side_effects(self) -> None:
        s12, hypotheses, contrasts, split, importance = _inputs()
        recipes = build_intervention_recipes(s12, hypotheses, contrasts, importance)
        search, paired = build_intervention_search_results(s12, recipes, split)
        interventions = search[search["interventionApplied"]]
        self.assertFalse(paired.empty)
        self.assertTrue(interventions["pairedNoInterventionControlPresent"].all())
        self.assertTrue(interventions["totalInterventionCostScore"].notna().all())
        self.assertTrue(interventions["sideEffectPenaltyScore"].notna().all())
        self.assertFalse(interventions["heldoutSeedValidated"].any())
        self.assertFalse(interventions["heldoutRatioValidated"].any())

    def test_heldout_summary_and_validation_checks(self) -> None:
        s12, hypotheses, contrasts, split, importance = _inputs()
        recipes = build_intervention_recipes(s12, hypotheses, contrasts, importance)
        search, paired = build_intervention_search_results(s12, recipes, split)
        effect, _, heldout, _ = summarize_intervention_effects(search)
        checks = validation_checks(s12, hypotheses, recipes, search, paired, effect, heldout)
        self.assertIn("not_feasible_single_s12_seed", set(heldout["heldoutSeedValidationStatus"]))
        check_lookup = dict(zip(checks["checkId"], checks["success"], strict=True))
        self.assertTrue(check_lookup["paired_no_intervention_controls"])
        self.assertTrue(check_lookup["cost_and_side_effect_accounting"])
        self.assertTrue(check_lookup["analogy_caveats_retained"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
