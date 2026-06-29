from __future__ import annotations

import unittest

import pandas as pd

from chimera.causal_models import (
    DIRECT_TARGET_COLUMNS,
    build_feature_matrix,
    make_group_holdout_split,
    paired_protocol_contrasts,
    target_definitions,
)


def _row(group: str, protocol: str, *, carryover: str, mosaic: str, target: float, rescue: bool) -> dict[str, object]:
    return {
        "conditionId": f"{group}_{protocol}",
        "s12SourceS11ConditionId": group,
        "runSucceeded": True,
        "pairCategory": "frontier_pair",
        "ratioLabel": "75:25",
        "arrangementType": "contiguous_patch",
        "valueProfile": "random_unique",
        "perturbationType": "none",
        "goalMode": "opposite_goal",
        "goalCompatibilityClass": "opposite",
        "hostPolicyId": "host",
        "donorPolicyId": "donor",
        "mutantBehaviorVariant": "selfish_aggregation",
        "mutantObjectiveFamily": "selfish",
        "mutantClonePositionName": "center_clone",
        "s11MutantOutcomeClass": "disruptive_takeover_proxy",
        "graftOutcomeClass": "equilibrated_mosaic",
        "governanceVariant": "no_governance_control",
        "governanceFamily": "none",
        "interfaceRuleVariant": "disabled_baseline",
        "interfaceRuleFamily": "none",
        "historyProtocol": protocol,
        "historyConditionKind": "unit",
        "historyTimingLabel": "early",
        "memoryCarryoverMode": carryover,
        "policyStateCarryoverMode": carryover,
        "signalFieldCarryoverMode": carryover,
        "baselineGovernanceVariant": "no_governance_control",
        "baselineInterfaceRuleVariant": "disabled_baseline",
        "n": 100,
        "targetPolicyCount": 2,
        "policyCount": 2,
        "influenceRange": 1,
        "mutantCloneSize": 5,
        "preExposureActivationCap": 500,
        "cloneIntroductionActivation": 500,
        "transientPerturbationStartActivation": -1,
        "transientPerturbationEndActivation": -1,
        "historyTotalActivationCap": 3500,
        "historyActualStageCount": 2,
        "containmentScore": 0.3,
        "takeoverProxyScore": 0.7,
        "s10StressScore": 0.5,
        "s11ContextPriorityScore": 1.0,
        "s12PriorityScore": 1.5,
        "governanceEnabled": False,
        "localInformationOnly": True,
        "usesGlobalState": False,
        "usesTargetMap": False,
        "usesOrganizer": False,
        "broadControlLike": False,
        "upperBoundControlProxy": False,
        "interfaceRuleEnabled": False,
        "clonePresentAtStart": True,
        "cloneIntroducedDuringRun": False,
        "divisionEnabled": False,
        "memoryEligiblePolicyPresent": False,
        "simultaneousReplayMatchesS11Reference": True,
        "noHistoryReplayMatchesS11NoMutant": True,
        "graftApplied": False,
        "s07MosaicClass": mosaic,
        "finalTargetQualityScore": target,
        "dominanceProxyScore": 0.2 + target,
        "finalAggregation": 0.4 + target / 10,
        "historyOutcomeClass": "history_rescue_proxy" if rescue else "history_neutral_within_threshold",
        "historyEffectMagnitudeScore": 0.2 if rescue else 0.05,
        "historyDisruptionProxyScore": 0.0,
        "finalStateHash": "same_row_outcome_hash",
        "deltaVsSimultaneousFinalTargetQualityScore": target - 0.5,
    }


class TestE06CausalModels(unittest.TestCase):
    def test_target_definitions_have_required_caveats(self) -> None:
        targets = target_definitions()
        self.assertEqual(set(targets["targetName"]), {
            "final_morphology_class",
            "rescue_success_proxy",
            "history_sensitive_proxy",
            "final_target_quality",
            "dominance_proxy",
            "final_aggregation",
        })
        self.assertTrue(targets["causalCaveat"].str.contains("not biological causal proof").all())

    def test_feature_matrix_includes_history_flags_and_excludes_targets(self) -> None:
        df = pd.DataFrame(
            [
                _row("g0", "simultaneous_clone_reference", carryover="single_stage_reference", mosaic="patchy", target=0.4, rescue=False),
                _row("g0", "early_staged_introduction_reset", carryover="state_reset_at_clone_insertion", mosaic="layered", target=0.6, rescue=True),
            ]
        )
        matrix, _, audit, features, _, _, _ = build_feature_matrix(df)
        self.assertIn("isResetProtocol", features)
        self.assertIn("simultaneousReplayMatchesS11Reference", features)
        self.assertTrue(matrix["exactReplayIntegrityAllMatched"].all())
        included = set(audit.loc[audit["includedInPrimaryModel"], "featureName"])
        self.assertTrue(set(DIRECT_TARGET_COLUMNS).isdisjoint(included))
        high_included = audit[audit["includedInPrimaryModel"] & audit["leakageRisk"].str.startswith("high_")]
        self.assertTrue(high_included.empty)

    def test_group_split_keeps_source_contexts_separate(self) -> None:
        rows = []
        for group_index in range(8):
            rows.append(
                _row(
                    f"g{group_index}",
                    "simultaneous_clone_reference",
                    carryover="single_stage_reference",
                    mosaic=["patchy", "layered", "polarized"][group_index % 3],
                    target=0.1 * group_index,
                    rescue=group_index % 2 == 0,
                )
            )
        matrix, *_ = build_feature_matrix(pd.DataFrame(rows))
        split = make_group_holdout_split(matrix, random_state=13013)
        train_groups = set(split.loc[split["split"] == "train", "s12SourceS11ConditionId"])
        test_groups = set(split.loc[split["split"] == "test", "s12SourceS11ConditionId"])
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertGreaterEqual(len(test_groups), 1)

    def test_paired_protocol_contrasts_use_matched_source_contexts(self) -> None:
        rows = []
        for group in ["g0", "g1"]:
            rows.extend(
                [
                    _row(group, "early_staged_introduction_reset", carryover="state_reset_at_clone_insertion", mosaic="patchy", target=0.2, rescue=False),
                    _row(group, "early_staged_introduction_carryover", carryover="policy_state_carryover_within_simulator", mosaic="layered", target=0.5, rescue=True),
                ]
            )
        contrasts = paired_protocol_contrasts(pd.DataFrame(rows))
        target = contrasts[
            (contrasts["comparisonName"] == "carryover_minus_reset_early_staged")
            & (contrasts["metricName"] == "finalTargetQualityScore")
        ].iloc[0]
        self.assertEqual(int(target["nPairs"]), 2)
        self.assertAlmostEqual(float(target["meanTreatmentMinusReference"]), 0.3)
        self.assertTrue(contrasts["causalCaveat"].str.contains("not biological causal proof").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
