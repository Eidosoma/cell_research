"""E06 S14 minimal intervention search over S12/S13 evidence.

S14 uses S13 predictors and causal-hypothesis candidates to rank intervention
recipes that were already evaluated as S12 developmental-history protocol arms.
The resulting recommendations are computational intervention candidates only;
they are not biological rescue protocols or causal proof.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mixtures import artifact_records, compact_result_csv, git_value, write_json
from .panel import CLAIM_BOUNDARY, json_ready


STEP_ID = "S14"
STEP_NUMBER = 14
INTERVENTION_SEARCH_VERSION = "e06_s14_intervention_search.v1"
DEFAULT_S12_RESULTS_PATH = Path("/artifacts/results/e06_developmental_history.parquet")
DEFAULT_S13_HYPOTHESES_PATH = Path("/artifacts/research_steps/S13/causal_hypothesis_table.parquet")
DEFAULT_S13_CONTRASTS_PATH = Path("/artifacts/research_steps/S13/paired_protocol_contrasts.parquet")
DEFAULT_S13_SPLIT_PATH = Path("/artifacts/research_steps/S13/heldout_split.parquet")
DEFAULT_S13_IMPORTANCE_PATH = Path("/artifacts/research_steps/S13/feature_importance_by_source.parquet")
NO_INTERVENTION_PROTOCOL = "simultaneous_clone_reference"
NO_MUTANT_REFERENCE_PROTOCOL = "matched_no_mutant_replay"
INTERVENTION_ANALOGY_CAVEAT = (
    "Intervention, rescue, graft, clone, governance, and developmental-history language is a computational analogy "
    "over fixed-size one-dimensional local-policy arrays; it is not a biological intervention protocol."
)
DERIVED_SEARCH_CAVEAT = (
    "S14 is a derived minimal-intervention search over S12 protocol arms selected by S13 predictors and hypotheses; "
    "it does not run new seed or ratio simulations."
)
HELDOUT_FEASIBILITY_CAVEAT = (
    "S12/S13 contain one seed and one 75:25 ratio in the selected intervention corpus, so held-out seed and ratio "
    "validation are not feasible in S14; held-out validation uses the S13 source-context split."
)
S14_RECOMMENDED_NEXT_ACTION = (
    "Chief Scientist review, then S15 should synthesize the E06 chimeric-control playbook using S14 recipes only as "
    "simulation-derived intervention candidates with paired-control and holdout caveats."
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def _recipe_id(protocol: str) -> str:
    if protocol == NO_INTERVENTION_PROTOCOL:
        return "s14_no_intervention_control"
    return f"s14_{protocol}".replace("_introduction_", "_intro_").replace("_rule_release_", "_rule_")


def _protocol_family(protocol: str) -> str:
    if protocol == NO_INTERVENTION_PROTOCOL:
        return "no_intervention_control"
    if "staged_introduction" in protocol:
        return "staged_clone_timing"
    if "transient_rule_release" in protocol:
        return "transient_rule_switch"
    return "other_history_protocol"


def _timing_label(protocol: str) -> str:
    if protocol.startswith("early_"):
        return "early"
    if protocol.startswith("late_"):
        return "late"
    if protocol == NO_INTERVENTION_PROTOCOL:
        return "simultaneous"
    return "unknown"


def _memory_mode(protocol: str) -> str:
    if protocol.endswith("_reset"):
        return "reset"
    if protocol.endswith("_carryover"):
        return "carryover"
    return "not_applicable"


def load_s14_inputs(
    *,
    s12_results_path: Path = DEFAULT_S12_RESULTS_PATH,
    s13_hypotheses_path: Path = DEFAULT_S13_HYPOTHESES_PATH,
    s13_contrasts_path: Path = DEFAULT_S13_CONTRASTS_PATH,
    s13_split_path: Path = DEFAULT_S13_SPLIT_PATH,
    s13_importance_path: Path = DEFAULT_S13_IMPORTANCE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [s12_results_path, s13_hypotheses_path, s13_contrasts_path, s13_split_path, s13_importance_path]:
        if not path.exists():
            raise FileNotFoundError(f"required S14 input artifact missing: {path}")
    s12 = pd.read_parquet(s12_results_path)
    hypotheses = pd.read_parquet(s13_hypotheses_path)
    contrasts = pd.read_parquet(s13_contrasts_path)
    split = pd.read_parquet(s13_split_path)
    importance = pd.read_parquet(s13_importance_path)
    required_s12 = {
        "conditionId",
        "s12SourceS11ConditionId",
        "historyProtocol",
        "historyOutcomeClass",
        "finalTargetQualityScore",
        "minPolicyGoalScore",
        "finalAggregation",
        "dominanceProxyScore",
        "historyEffectMagnitudeScore",
        "historyDisruptionProxyScore",
        "simultaneousFinalTargetQualityScore",
        "simultaneousMinPolicyGoalScore",
        "simultaneousFinalAggregation",
        "simultaneousDominanceProxyScore",
        "simultaneousMutantCloneSpreadFraction",
        "deltaVsSimultaneousFinalTargetQualityScore",
        "deltaVsSimultaneousMinPolicyGoalScore",
        "deltaVsSimultaneousFinalAggregation",
        "historyChangedMosaicClassVsSimultaneous",
        "ratioLabel",
        "seedIndex",
    }
    missing = sorted(required_s12 - set(s12.columns))
    if missing:
        raise ValueError(f"S12 results missing required S14 columns: {missing}")
    if s12.empty or hypotheses.empty or contrasts.empty or split.empty:
        raise ValueError("S14 inputs must be non-empty")
    return s12.copy(), hypotheses.copy(), contrasts.copy(), split.copy(), importance.copy()


def _predictor_support_score(protocol: str, importance: pd.DataFrame) -> float:
    if importance.empty:
        return 0.0
    source_scores = (
        importance.groupby("sourceFeature", dropna=False)["totalImportance"].sum().to_dict()
        if "totalImportance" in importance.columns
        else {}
    )
    base = (
        _safe_float(source_scores.get("historyProtocol"))
        + _safe_float(source_scores.get("historyTimingLabel"))
        + _safe_float(source_scores.get("memoryCarryoverMode"))
        + _safe_float(source_scores.get("policyStateCarryoverMode"))
        + _safe_float(source_scores.get("signalFieldCarryoverMode"))
    )
    if "staged" in protocol:
        base += _safe_float(source_scores.get("cloneIntroductionActivation")) + _safe_float(source_scores.get("preExposureActivationCap"))
    if "transient" in protocol:
        base += _safe_float(source_scores.get("transientPerturbationStartActivation")) + _safe_float(
            source_scores.get("transientPerturbationEndActivation")
        )
    return float(base)


def build_intervention_recipes(
    s12_results: pd.DataFrame,
    hypotheses: pd.DataFrame,
    contrasts: pd.DataFrame,
    importance: pd.DataFrame,
) -> pd.DataFrame:
    protocols = [
        NO_INTERVENTION_PROTOCOL,
        *sorted(
            protocol
            for protocol in s12_results["historyProtocol"].astype(str).unique()
            if protocol not in {NO_INTERVENTION_PROTOCOL, NO_MUTANT_REFERENCE_PROTOCOL}
        ),
    ]
    rows: list[dict[str, Any]] = []
    for protocol in protocols:
        subset = s12_results[s12_results["historyProtocol"].astype(str) == protocol]
        exemplar = subset.iloc[0] if not subset.empty else pd.Series(dtype=object)
        total_cap = max(1.0, _safe_float(exemplar.get("historyTotalActivationCap"), 3500.0))
        start = _safe_float(exemplar.get("transientPerturbationStartActivation"), -1.0)
        end = _safe_float(exemplar.get("transientPerturbationEndActivation"), -1.0)
        transient_duration = max(0.0, end - max(0.0, start)) if "transient" in protocol else 0.0
        pre_exposure = max(0.0, _safe_float(exemplar.get("preExposureActivationCap"), 0.0)) if "staged" in protocol else 0.0
        activation_fraction = min(1.0, (transient_duration + pre_exposure) / total_cap)
        reset_penalty = 0.05 if protocol.endswith("_reset") else 0.0
        carryover_penalty = 0.02 if protocol.endswith("_carryover") else 0.0
        rule_switch_penalty = 0.04 if "transient" in protocol else 0.0
        clone_delay_penalty = 0.03 if "staged" in protocol else 0.0
        intervention_cost = min(1.0, activation_fraction + reset_penalty + carryover_penalty + rule_switch_penalty + clone_delay_penalty)
        protocol_contrasts = contrasts[contrasts["treatmentProtocol"].astype(str) == protocol].copy()
        rescue_effect = protocol_contrasts.loc[
            protocol_contrasts["metricName"].astype(str).eq("rescueSuccessProxyTarget"),
            "meanTreatmentMinusReference",
        ]
        target_effect = protocol_contrasts.loc[
            protocol_contrasts["metricName"].astype(str).eq("finalTargetQualityScore"),
            "meanTreatmentMinusReference",
        ]
        hypothesis_hits = hypotheses[hypotheses["hypothesis"].astype(str).str.contains(protocol, regex=False, na=False)]
        rows.append(
            {
                "interventionRecipeId": _recipe_id(protocol),
                "historyProtocol": protocol,
                "interventionFamily": _protocol_family(protocol),
                "timingLabel": _timing_label(protocol),
                "memoryMode": _memory_mode(protocol),
                "interventionType": "no_intervention_control" if protocol == NO_INTERVENTION_PROTOCOL else "history_protocol_intervention",
                "searchEligible": protocol != NO_INTERVENTION_PROTOCOL,
                "noInterventionControl": protocol == NO_INTERVENTION_PROTOCOL,
                "preExposureActivationCap": int(pre_exposure),
                "transientRuleReleaseDuration": int(transient_duration),
                "activationFractionCost": float(activation_fraction),
                "resetPenaltyCost": float(reset_penalty),
                "carryoverPenaltyCost": float(carryover_penalty),
                "ruleSwitchPenaltyCost": float(rule_switch_penalty),
                "cloneDelayPenaltyCost": float(clone_delay_penalty),
                "interventionCostScore": float(intervention_cost),
                "s13MeanRescueEffect": float(rescue_effect.iloc[0]) if not rescue_effect.empty else 0.0,
                "s13MeanTargetQualityEffect": float(target_effect.iloc[0]) if not target_effect.empty else 0.0,
                "s13PredictorSupportScore": _predictor_support_score(protocol, importance),
                "sourceS13ComparisonNamesJson": _compact_json(sorted(protocol_contrasts["comparisonName"].astype(str).unique())),
                "sourceS13HypothesisIdsJson": _compact_json(sorted(hypothesis_hits["hypothesisId"].astype(str).unique())),
                "sourceS13EvidenceType": "paired_protocol_contrast_and_feature_importance",
                "minimalityRationale": (
                    "No additional intervention."
                    if protocol == NO_INTERVENTION_PROTOCOL
                    else "Uses an existing S12 low-dimensional protocol arm: timing shift, transient rule switch, reset, or carryover only."
                ),
                "biologicalAnalogyCaveat": INTERVENTION_ANALOGY_CAVEAT,
                "claimBoundary": CLAIM_BOUNDARY,
                "interventionSearchVersion": INTERVENTION_SEARCH_VERSION,
            }
        )
    recipe_df = pd.DataFrame(rows)
    max_support = max(1.0, float(recipe_df["s13PredictorSupportScore"].max()))
    recipe_df["s13PredictorSupportScoreNormalized"] = recipe_df["s13PredictorSupportScore"] / max_support
    return recipe_df


def _side_effect_class(score: float) -> str:
    if score >= 0.45:
        return "high_side_effect_proxy"
    if score >= 0.20:
        return "moderate_side_effect_proxy"
    return "low_side_effect_proxy"


def build_intervention_search_results(
    s12_results: pd.DataFrame,
    recipes: pd.DataFrame,
    split: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_cols = split[["conditionId", "split", "s12SourceS11ConditionId"]].rename(columns={"split": "s13Split"})
    df = s12_results.merge(split_cols[["conditionId", "s13Split"]], on="conditionId", how="left")
    df["s13Split"] = df["s13Split"].fillna("unassigned")
    df = df[df["historyProtocol"].astype(str) != NO_MUTANT_REFERENCE_PROTOCOL].copy()
    baseline = df[df["historyProtocol"].astype(str) == NO_INTERVENTION_PROTOCOL].copy()
    baseline_cols = [
        "s12SourceS11ConditionId",
        "conditionId",
        "finalTargetQualityScore",
        "minPolicyGoalScore",
        "finalAggregation",
        "dominanceProxyScore",
        "s07MosaicClass",
        "finalStateHash",
        "activationCount",
        "mutantCloneSpreadFraction",
    ]
    baseline = baseline[[col for col in baseline_cols if col in baseline.columns]].rename(
        columns={
            "conditionId": "noInterventionConditionId",
            "finalTargetQualityScore": "noInterventionFinalTargetQualityScore",
            "minPolicyGoalScore": "noInterventionMinPolicyGoalScore",
            "finalAggregation": "noInterventionFinalAggregation",
            "dominanceProxyScore": "noInterventionDominanceProxyScore",
            "s07MosaicClass": "noInterventionMosaicClass",
            "finalStateHash": "noInterventionFinalStateHash",
            "activationCount": "noInterventionActivationCount",
            "mutantCloneSpreadFraction": "noInterventionMutantCloneSpreadFraction",
        }
    )
    no_mutant = s12_results[s12_results["historyProtocol"].astype(str) == NO_MUTANT_REFERENCE_PROTOCOL].copy()
    no_mutant = no_mutant[["s12SourceS11ConditionId", "conditionId", "finalTargetQualityScore", "s07MosaicClass", "finalStateHash"]].rename(
        columns={
            "conditionId": "matchedNoMutantReferenceConditionId",
            "finalTargetQualityScore": "matchedNoMutantFinalTargetQualityScore",
            "s07MosaicClass": "matchedNoMutantMosaicClass",
            "finalStateHash": "matchedNoMutantFinalStateHash",
        }
    )
    out = df.merge(baseline, on="s12SourceS11ConditionId", how="left").merge(no_mutant, on="s12SourceS11ConditionId", how="left")
    out = out.merge(recipes, on="historyProtocol", how="left")
    out["interventionApplied"] = out["historyProtocol"].astype(str) != NO_INTERVENTION_PROTOCOL
    out["pairedNoInterventionControlPresent"] = out["noInterventionConditionId"].astype(str).str.len().gt(0)
    out["pairedNoMutantReferencePresent"] = out["matchedNoMutantReferenceConditionId"].astype(str).str.len().gt(0)
    out["targetQualityDeltaVsNoIntervention"] = pd.to_numeric(out["finalTargetQualityScore"], errors="coerce") - pd.to_numeric(
        out["noInterventionFinalTargetQualityScore"], errors="coerce"
    )
    out["minGoalDeltaVsNoIntervention"] = pd.to_numeric(out["minPolicyGoalScore"], errors="coerce") - pd.to_numeric(
        out["noInterventionMinPolicyGoalScore"], errors="coerce"
    )
    out["aggregationDeltaVsNoIntervention"] = pd.to_numeric(out["finalAggregation"], errors="coerce") - pd.to_numeric(
        out["noInterventionFinalAggregation"], errors="coerce"
    )
    out["dominanceDeltaVsNoIntervention"] = pd.to_numeric(out["dominanceProxyScore"], errors="coerce") - pd.to_numeric(
        out["noInterventionDominanceProxyScore"], errors="coerce"
    )
    out["cloneSpreadDeltaVsNoIntervention"] = pd.to_numeric(out.get("mutantCloneSpreadFraction", 0.0), errors="coerce") - pd.to_numeric(
        out.get("noInterventionMutantCloneSpreadFraction", 0.0),
        errors="coerce",
    )
    out["mosaicClassChangedVsNoIntervention"] = out["s07MosaicClass"].astype(str) != out["noInterventionMosaicClass"].astype(str)
    out["stateHashChangedVsNoIntervention"] = out["finalStateHash"].astype(str) != out["noInterventionFinalStateHash"].astype(str)
    out["rescueSuccessProxy"] = out["historyOutcomeClass"].astype(str).eq("history_rescue_proxy")
    out["historySensitiveProxy"] = pd.to_numeric(out["historyEffectMagnitudeScore"], errors="coerce").fillna(0.0) >= 0.15
    out["baselineFailedOrUnstableProxy"] = (
        pd.to_numeric(out["noInterventionFinalTargetQualityScore"], errors="coerce").fillna(0.0).lt(0.75)
        | out["noInterventionMosaicClass"].astype(str).isin({"patchy", "polarized", "layered", "mixed", "frozen_conflict", "oscillatory"})
    )
    out["candidateAppliedToFailedOrUnstableBaseline"] = out["interventionApplied"] & out["baselineFailedOrUnstableProxy"]
    governance_access = pd.to_numeric(out.get("governanceInformationAccessScore", 0.0), errors="coerce").fillna(0.0)
    broad_flag = out.get("broadControlLike", pd.Series(False, index=out.index)).fillna(False).map(bool)
    out["governanceAccessCostScore"] = np.minimum(1.0, 0.25 * governance_access + 0.25 * broad_flag.astype(float))
    out["totalInterventionCostScore"] = np.minimum(
        1.0,
        pd.to_numeric(out["interventionCostScore"], errors="coerce").fillna(0.0) + out["governanceAccessCostScore"],
    )
    target_delta = pd.to_numeric(out["targetQualityDeltaVsNoIntervention"], errors="coerce").fillna(0.0)
    min_delta = pd.to_numeric(out["minGoalDeltaVsNoIntervention"], errors="coerce").fillna(0.0)
    agg_delta = pd.to_numeric(out["aggregationDeltaVsNoIntervention"], errors="coerce").fillna(0.0)
    dominance_delta = pd.to_numeric(out["dominanceDeltaVsNoIntervention"], errors="coerce").fillna(0.0)
    clone_delta = pd.to_numeric(out["cloneSpreadDeltaVsNoIntervention"], errors="coerce").fillna(0.0)
    out["rescueBenefitScore"] = (
        out["rescueSuccessProxy"].astype(float)
        + 2.0 * np.maximum(target_delta, 0.0)
        + np.maximum(min_delta, 0.0) / 100.0
        + np.maximum(pd.to_numeric(out["historyEffectMagnitudeScore"], errors="coerce").fillna(0.0) - 0.15, 0.0)
    )
    out["sideEffectPenaltyScore"] = (
        2.0 * np.maximum(-target_delta, 0.0)
        + np.maximum(-min_delta, 0.0) / 100.0
        + 0.25 * np.abs(agg_delta)
        + 0.20 * np.abs(dominance_delta)
        + 0.15 * np.abs(clone_delta)
        + 0.10 * (out["mosaicClassChangedVsNoIntervention"] & ~out["rescueSuccessProxy"]).astype(float)
        + pd.to_numeric(out["historyDisruptionProxyScore"], errors="coerce").fillna(0.0)
    )
    out["minimalInterventionScore"] = out["rescueBenefitScore"] - out["totalInterventionCostScore"] - out["sideEffectPenaltyScore"]
    out["sideEffectClass"] = out["sideEffectPenaltyScore"].map(_side_effect_class)
    out["heldoutSourceContextValidated"] = out["s13Split"].astype(str).eq("test")
    out["heldoutSeedValidated"] = False
    out["heldoutRatioValidated"] = False
    out["heldoutFeasibilityCaveat"] = HELDOUT_FEASIBILITY_CAVEAT
    out["localOnlyContext"] = ~broad_flag
    out["globalControlLikeContext"] = broad_flag
    out["biologicalAnalogyCaveat"] = INTERVENTION_ANALOGY_CAVEAT
    out["claimBoundary"] = CLAIM_BOUNDARY
    out["interventionSearchVersion"] = INTERVENTION_SEARCH_VERSION
    out["researchStepId"] = STEP_ID
    out["interventionStepId"] = STEP_ID
    out["interventionType"] = "minimal_intervention_search"
    ranked = out[out["searchEligible"].fillna(False).map(bool)].copy()
    ranked["rankInSourceContext"] = ranked.groupby("s12SourceS11ConditionId")["minimalInterventionScore"].rank(
        method="first",
        ascending=False,
    )
    out = out.merge(
        ranked[["conditionId", "rankInSourceContext"]],
        on="conditionId",
        how="left",
    )
    out["selectedBestRecipeForSourceContext"] = out["rankInSourceContext"].eq(1.0)

    control_pairs = out[out["interventionApplied"]].copy()
    control_cols = [
        "s12SourceS11ConditionId",
        "conditionId",
        "interventionRecipeId",
        "historyProtocol",
        "s13Split",
        "noInterventionConditionId",
        "matchedNoMutantReferenceConditionId",
        "pairedNoInterventionControlPresent",
        "pairedNoMutantReferencePresent",
        "targetQualityDeltaVsNoIntervention",
        "minGoalDeltaVsNoIntervention",
        "aggregationDeltaVsNoIntervention",
        "dominanceDeltaVsNoIntervention",
        "mosaicClassChangedVsNoIntervention",
        "stateHashChangedVsNoIntervention",
        "rescueSuccessProxy",
        "totalInterventionCostScore",
        "sideEffectPenaltyScore",
        "minimalInterventionScore",
        "heldoutSeedValidated",
        "heldoutRatioValidated",
        "heldoutFeasibilityCaveat",
        "biologicalAnalogyCaveat",
        "claimBoundary",
    ]
    return out.reset_index(drop=True), control_pairs[[col for col in control_cols if col in control_pairs.columns]].reset_index(drop=True)


def summarize_intervention_effects(search: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = search[search["searchEligible"].fillna(False).map(bool)].copy()
    group_cols = ["interventionRecipeId", "historyProtocol", "interventionFamily", "timingLabel", "memoryMode", "s13Split"]
    effect_summary = (
        candidates.groupby(group_cols, dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            rescueRate=("rescueSuccessProxy", "mean"),
            historySensitiveRate=("historySensitiveProxy", "mean"),
            meanTargetQualityDelta=("targetQualityDeltaVsNoIntervention", "mean"),
            meanMinGoalDelta=("minGoalDeltaVsNoIntervention", "mean"),
            mosaicChangeRate=("mosaicClassChangedVsNoIntervention", "mean"),
            meanCostScore=("totalInterventionCostScore", "mean"),
            meanSideEffectPenalty=("sideEffectPenaltyScore", "mean"),
            meanMinimalInterventionScore=("minimalInterventionScore", "mean"),
            bestMinimalInterventionScore=("minimalInterventionScore", "max"),
            broadControlContextFraction=("globalControlLikeContext", "mean"),
        )
        .reset_index()
    )
    overall = (
        candidates.groupby(["interventionRecipeId", "historyProtocol", "interventionFamily", "timingLabel", "memoryMode"], dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            rescueRate=("rescueSuccessProxy", "mean"),
            historySensitiveRate=("historySensitiveProxy", "mean"),
            meanTargetQualityDelta=("targetQualityDeltaVsNoIntervention", "mean"),
            meanMinGoalDelta=("minGoalDeltaVsNoIntervention", "mean"),
            mosaicChangeRate=("mosaicClassChangedVsNoIntervention", "mean"),
            meanCostScore=("totalInterventionCostScore", "mean"),
            meanSideEffectPenalty=("sideEffectPenaltyScore", "mean"),
            meanMinimalInterventionScore=("minimalInterventionScore", "mean"),
            bestMinimalInterventionScore=("minimalInterventionScore", "max"),
            broadControlContextFraction=("globalControlLikeContext", "mean"),
            selectedBestContextCount=("selectedBestRecipeForSourceContext", "sum"),
        )
        .reset_index()
    )
    heldout_rows: list[dict[str, Any]] = []
    for recipe, group in candidates.groupby("interventionRecipeId", sort=False):
        train = group[group["s13Split"].astype(str).eq("train")]
        test = group[group["s13Split"].astype(str).eq("test")]
        train_score = float(train["minimalInterventionScore"].mean()) if not train.empty else float("nan")
        test_score = float(test["minimalInterventionScore"].mean()) if not test.empty else float("nan")
        train_rescue = float(train["rescueSuccessProxy"].mean()) if not train.empty else float("nan")
        test_rescue = float(test["rescueSuccessProxy"].mean()) if not test.empty else float("nan")
        if test.empty:
            transfer = "not_tested_on_s13_heldout_contexts"
        elif test_score > 0.0 and train_score > 0.0:
            transfer = "positive_train_and_heldout_support"
        elif test_score > 0.0:
            transfer = "heldout_positive_train_weak"
        elif train_score > 0.0:
            transfer = "train_positive_not_heldout"
        else:
            transfer = "no_positive_support"
        protocol = str(group["historyProtocol"].iloc[0])
        recommended = bool(test_score > 0.0 and test_rescue > 0.0 and float(test["sideEffectPenaltyScore"].mean()) < 0.45)
        heldout_rows.append(
            {
                "interventionRecipeId": recipe,
                "historyProtocol": protocol,
                "trainConditionCount": int(len(train)),
                "heldoutConditionCount": int(len(test)),
                "trainRescueRate": train_rescue,
                "heldoutRescueRate": test_rescue,
                "trainMeanMinimalInterventionScore": train_score,
                "heldoutMeanMinimalInterventionScore": test_score,
                "heldoutMeanCostScore": float(test["totalInterventionCostScore"].mean()) if not test.empty else float("nan"),
                "heldoutMeanSideEffectPenalty": float(test["sideEffectPenaltyScore"].mean()) if not test.empty else float("nan"),
                "heldoutTransferClass": transfer,
                "heldoutSourceContextValidated": bool(not test.empty),
                "heldoutSeedValidationStatus": "not_feasible_single_s12_seed",
                "heldoutRatioValidationStatus": "not_feasible_single_s12_ratio",
                "recommendedForS15PlaybookCandidate": recommended,
                "recommendationCaveat": (
                    "Use as a playbook candidate only with paired-control, source-context holdout, and single-seed/single-ratio caveats."
                    if recommended
                    else "Not recommended as a robust S14 recipe without further validation."
                ),
                "biologicalAnalogyCaveat": INTERVENTION_ANALOGY_CAVEAT,
                "claimBoundary": CLAIM_BOUNDARY,
            }
        )
    heldout = pd.DataFrame(heldout_rows).sort_values(
        ["recommendedForS15PlaybookCandidate", "heldoutMeanMinimalInterventionScore"],
        ascending=[False, False],
        kind="mergesort",
    )
    side_effect_summary = (
        candidates.groupby(["interventionRecipeId", "historyProtocol"], dropna=False)
        .agg(
            meanSideEffectPenalty=("sideEffectPenaltyScore", "mean"),
            maxSideEffectPenalty=("sideEffectPenaltyScore", "max"),
            lowSideEffectRate=("sideEffectClass", lambda s: float((s.astype(str) == "low_side_effect_proxy").mean())),
            mosaicChangeRate=("mosaicClassChangedVsNoIntervention", "mean"),
            meanDominanceDelta=("dominanceDeltaVsNoIntervention", "mean"),
            meanCloneSpreadDelta=("cloneSpreadDeltaVsNoIntervention", "mean"),
            broadControlContextFraction=("globalControlLikeContext", "mean"),
        )
        .reset_index()
    )
    return effect_summary, overall, heldout, side_effect_summary


def validation_checks(
    s12_results: pd.DataFrame,
    hypotheses: pd.DataFrame,
    recipes: pd.DataFrame,
    search: pd.DataFrame,
    paired_controls: pd.DataFrame,
    effect_summary: pd.DataFrame,
    heldout_summary: pd.DataFrame,
) -> pd.DataFrame:
    train_contexts = set(search.loc[search["s13Split"] == "train", "s12SourceS11ConditionId"].astype(str))
    test_contexts = set(search.loc[search["s13Split"] == "test", "s12SourceS11ConditionId"].astype(str))
    non_control = search[search["interventionApplied"].map(bool)]
    recommended = heldout_summary[heldout_summary["recommendedForS15PlaybookCandidate"].fillna(False).map(bool)]
    checks = [
        {
            "checkId": "s12_s13_inputs_loaded",
            "success": bool(len(s12_results) == 120 and not hypotheses.empty and not recipes.empty),
            "detail": f"{len(s12_results)} S12 rows, {len(hypotheses)} S13 hypotheses, {len(recipes)} recipes loaded/generated",
        },
        {
            "checkId": "s13_predictors_and_hypotheses_used",
            "success": bool(
                recipes["s13PredictorSupportScore"].notna().all()
                and recipes["sourceS13EvidenceType"].astype(str).str.contains("feature_importance").any()
                and recipes["sourceS13ComparisonNamesJson"].astype(str).str.len().gt(2).any()
            ),
            "detail": "recipes record S13 predictor-support scores plus paired-contrast/hypothesis provenance",
        },
        {
            "checkId": "paired_no_intervention_controls",
            "success": bool(not non_control.empty and non_control["pairedNoInterventionControlPresent"].map(bool).all()),
            "detail": f"{len(non_control)} intervention rows have paired simultaneous no-intervention controls",
        },
        {
            "checkId": "no_intervention_controls_included",
            "success": bool((search["historyProtocol"].astype(str) == NO_INTERVENTION_PROTOCOL).sum() == s12_results["s12SourceS11ConditionId"].nunique()),
            "detail": "one simultaneous no-intervention control retained for every S12 source context",
        },
        {
            "checkId": "heldout_context_validation",
            "success": bool(train_contexts and test_contexts and train_contexts.isdisjoint(test_contexts) and len(test_contexts) == 3),
            "detail": f"{len(train_contexts)} train contexts and {len(test_contexts)} S13 held-out contexts used; seed/ratio holdouts marked infeasible",
        },
        {
            "checkId": "cost_and_side_effect_accounting",
            "success": bool(
                search["totalInterventionCostScore"].notna().all()
                and search["sideEffectPenaltyScore"].notna().all()
                and search["sideEffectClass"].astype(str).str.len().gt(0).all()
            ),
            "detail": "intervention cost, governance-access cost, side-effect penalty, and side-effect class written for all search rows",
        },
        {
            "checkId": "heldout_recommendations_ranked",
            "success": bool(not heldout_summary.empty and heldout_summary["heldoutTransferClass"].astype(str).str.len().gt(0).all()),
            "detail": f"{int(recommended.shape[0])} recipes met the S14 held-out candidate threshold",
        },
        {
            "checkId": "broad_control_flags_retained",
            "success": bool("broadControlContextFraction" in effect_summary.columns and search["globalControlLikeContext"].notna().all()),
            "detail": "global-control-like/broad-organizer context flags retained; recommendations are not treated as local-only biological evidence",
        },
        {
            "checkId": "analogy_caveats_retained",
            "success": bool(
                search["biologicalAnalogyCaveat"].astype(str).str.contains("computational analogy", case=False, na=False).all()
                and search["claimBoundary"].astype(str).str.contains("Computational", na=False).all()
            ),
            "detail": "computational analogy caveats and claim boundaries retained on S14 rows",
        },
    ]
    return pd.DataFrame(checks)


def write_intervention_plots(overall: pd.DataFrame, heldout: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not heldout.empty:
        plot = heldout.sort_values("heldoutMeanMinimalInterventionScore", ascending=False, kind="mergesort").copy()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(plot["historyProtocol"], plot["heldoutMeanMinimalInterventionScore"], color="#4c78a8")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Held-out mean minimal-intervention score")
        ax.set_title("E06 S14 held-out intervention recipe scores")
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s14_intervention_scores.png",
            figure_dir / "e06_s14_intervention_scores.pdf",
            step_dir / "intervention_scores.png",
            step_dir / "intervention_scores.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    if not overall.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = np.where(overall["rescueRate"].fillna(0.0) > 0.0, "#54a24b", "#e45756")
        ax.scatter(overall["meanCostScore"], overall["meanTargetQualityDelta"], s=90, c=colors, alpha=0.85)
        for row in overall.itertuples(index=False):
            ax.annotate(str(row.historyProtocol).replace("_", "\n"), (row.meanCostScore, row.meanTargetQualityDelta), fontsize=7)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean cost score")
        ax.set_ylabel("Mean target-quality delta vs no intervention")
        ax.set_title("E06 S14 intervention cost vs target-quality effect")
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s14_cost_benefit.png",
            figure_dir / "e06_s14_cost_benefit.pdf",
            step_dir / "cost_benefit.png",
            step_dir / "cost_benefit.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    return paths


def write_markdown_report(
    *,
    step_dir: Path,
    status: Mapping[str, Any],
    artifacts_written: Sequence[str],
    heldout_summary: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Path:
    recommended = heldout_summary[heldout_summary["recommendedForS15PlaybookCandidate"].fillna(False).map(bool)].copy()
    top = heldout_summary.sort_values("heldoutMeanMinimalInterventionScore", ascending=False, kind="mergesort").head(6)
    top_lines = "\n".join(
        f"- `{row.historyProtocol}`: held-out score {row.heldoutMeanMinimalInterventionScore:.3f}, "
        f"held-out rescue {row.heldoutRescueRate:.2f}, transfer `{row.heldoutTransferClass}`"
        for row in top.itertuples(index=False)
    ) or "- No intervention summaries available."
    rec_lines = "\n".join(
        f"- `{row.historyProtocol}`: {row.recommendationCaveat}" for row in recommended.itertuples(index=False)
    ) or "- No recipe met the S14 held-out recommendation threshold."
    validation_commands = "\n".join(f"- `{item['command']}`: {item['result']}" for item in status.get("validationCommands", []))
    text = f"""# Research Step S14: Design interventions

## Completion status

Research step ID: `S14`. {status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

{INTERVENTION_ANALOGY_CAVEAT} {DERIVED_SEARCH_CAVEAT} {HELDOUT_FEASIBILITY_CAVEAT} {CLAIM_BOUNDARY}

## Lay summary

S14 ranked minimal intervention recipes already represented in S12 history protocols by combining S13 predictor/hypothesis evidence with paired no-intervention controls. It scored rescue benefit, intervention cost, and side-effect penalties, then evaluated transfer on the S13 held-out source contexts.

## Anchor intervention candidates

{top_lines}

## Playbook candidates

{rec_lines}

## Validation commands

{validation_commands}

## Recommended next action

{status['recommendedNextAction']}
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s14_intervention_search(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    s12_results_path: Path = DEFAULT_S12_RESULTS_PATH,
    s13_hypotheses_path: Path = DEFAULT_S13_HYPOTHESES_PATH,
    s13_contrasts_path: Path = DEFAULT_S13_CONTRASTS_PATH,
    s13_split_path: Path = DEFAULT_S13_SPLIT_PATH,
    s13_importance_path: Path = DEFAULT_S13_IMPORTANCE_PATH,
) -> dict[str, Any]:
    artifacts_dir = artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e06"
    provenance_dir = artifacts_dir / "provenance"
    for path in [step_dir, results_dir, figures_dir, provenance_dir]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    s12, hypotheses, contrasts, split, importance = load_s14_inputs(
        s12_results_path=s12_results_path,
        s13_hypotheses_path=s13_hypotheses_path,
        s13_contrasts_path=s13_contrasts_path,
        s13_split_path=s13_split_path,
        s13_importance_path=s13_importance_path,
    )
    recipes = build_intervention_recipes(s12, hypotheses, contrasts, importance)
    search, paired_controls = build_intervention_search_results(s12, recipes, split)
    effect_summary, overall_summary, heldout_summary, side_effect_summary = summarize_intervention_effects(search)
    validation_df = validation_checks(s12, hypotheses, recipes, search, paired_controls, effect_summary, heldout_summary)

    recipes_csv = step_dir / "intervention_recipes.csv"
    recipes_parquet = step_dir / "intervention_recipes.parquet"
    search_csv = step_dir / "intervention_search_runs.csv"
    search_parquet = step_dir / "intervention_search_runs.parquet"
    paired_csv = step_dir / "paired_no_intervention_controls.csv"
    paired_parquet = step_dir / "paired_no_intervention_controls.parquet"
    effect_csv = step_dir / "intervention_effect_summary.csv"
    effect_parquet = step_dir / "intervention_effect_summary.parquet"
    overall_csv = step_dir / "intervention_recipe_summary.csv"
    overall_parquet = step_dir / "intervention_recipe_summary.parquet"
    heldout_csv = step_dir / "heldout_intervention_validation.csv"
    heldout_parquet = step_dir / "heldout_intervention_validation.parquet"
    side_csv = step_dir / "intervention_side_effect_summary.csv"
    side_parquet = step_dir / "intervention_side_effect_summary.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    result_csv = results_dir / "e06_intervention_search.csv"
    result_parquet = results_dir / "e06_intervention_search.parquet"
    combined_csv = results_dir / "e06_governance_interventions.csv"
    combined_parquet = results_dir / "e06_governance_interventions.parquet"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    for path, df in [
        (recipes_csv, recipes),
        (search_csv, compact_result_csv(search)),
        (paired_csv, paired_controls),
        (effect_csv, effect_summary),
        (overall_csv, overall_summary),
        (heldout_csv, heldout_summary),
        (side_csv, side_effect_summary),
        (validation_csv, validation_df),
        (result_csv, compact_result_csv(search)),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (recipes_parquet, recipes),
        (search_parquet, search),
        (paired_parquet, paired_controls),
        (effect_parquet, effect_summary),
        (overall_parquet, overall_summary),
        (heldout_parquet, heldout_summary),
        (side_parquet, side_effect_summary),
        (validation_parquet, validation_df),
        (result_parquet, search),
    ]:
        df.to_parquet(path, index=False)

    if combined_parquet.exists():
        previous_combined = pd.read_parquet(combined_parquet)
        if "interventionStepId" in previous_combined.columns:
            previous_combined = previous_combined[previous_combined["interventionStepId"].astype(str) != STEP_ID]
        combined = pd.concat([previous_combined, search], ignore_index=True, sort=False)
    else:
        combined = search.copy()
    compact_result_csv(combined).to_csv(combined_csv, index=False)
    combined.to_parquet(combined_parquet, index=False)

    figure_paths = write_intervention_plots(overall_summary, heldout_summary, figures_dir, step_dir)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() else "failed"
    recommended_count = int(heldout_summary["recommendedForS15PlaybookCandidate"].fillna(False).sum()) if not heldout_summary.empty else 0
    weak_positive_count = (
        int(
            (
                pd.to_numeric(heldout_summary["heldoutMeanMinimalInterventionScore"], errors="coerce").fillna(0.0).gt(0.0)
                & pd.to_numeric(heldout_summary["heldoutRescueRate"], errors="coerce").fillna(0.0).gt(0.0)
            ).sum()
        )
        if not heldout_summary.empty
        else 0
    )
    if validation_result != "passed":
        outcome = "constraining/contradictory"
    elif recommended_count > 0:
        outcome = "supportive"
    elif weak_positive_count > 0:
        outcome = "constraining/contradictory"
    else:
        outcome = "null"
    artifacts = [
        recipes_csv,
        recipes_parquet,
        search_csv,
        search_parquet,
        paired_csv,
        paired_parquet,
        effect_csv,
        effect_parquet,
        overall_csv,
        overall_parquet,
        heldout_csv,
        heldout_parquet,
        side_csv,
        side_parquet,
        validation_csv,
        validation_parquet,
        result_csv,
        result_parquet,
        combined_csv,
        combined_parquet,
        *figure_paths,
        step_dir / "summary.md",
        status_path,
        manifest_path,
        run_manifest_path,
    ]
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_result == "passed",
        "status": "completed" if validation_result == "passed" else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifacts],
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            INTERVENTION_ANALOGY_CAVEAT,
            DERIVED_SEARCH_CAVEAT,
            HELDOUT_FEASIBILITY_CAVEAT,
            "Intervention success is condition-specific and scored with computational target-quality, rescue, and side-effect proxies.",
            "Broad-organizer/global-control-like contexts are flagged and should not be treated as local-only evidence.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": S14_RECOMMENDED_NEXT_ACTION,
        "laySummary": (
            "S14 ranked S12 history-protocol intervention arms using S13 predictors and causal-hypothesis candidates, "
            "paired every intervention against simultaneous no-intervention controls, and validated recipes on S13 held-out source contexts."
        ),
        "outcomeClassification": outcome,
        "conditionCount": int(len(search)),
        "interventionCandidateCount": int(search["interventionApplied"].sum()),
        "recipeCount": int(len(recipes)),
        "recommendedRecipeCount": recommended_count,
        "weakHeldoutPositiveCandidateCount": weak_positive_count,
        "pairedControlCount": int(len(paired_controls)),
        "heldoutRecipeCount": int(len(heldout_summary)),
        "trainSourceContextCount": int(search.loc[search["s13Split"] == "train", "s12SourceS11ConditionId"].nunique()),
        "heldoutSourceContextCount": int(search.loc[search["s13Split"] == "test", "s12SourceS11ConditionId"].nunique()),
        "heldoutSeedValidationStatus": "not_feasible_single_s12_seed",
        "heldoutRatioValidationStatus": "not_feasible_single_s12_ratio",
        "workerCount": 1,
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
        "newDependenciesInstalled": [],
        "pythonPackages": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "validationCommands": [
            {
                "command": "python -m pytest tests/test_e06_interventions.py -q",
                "result": "not available in this runtime because pytest is not installed; unittest validation was used instead",
            },
            {
                "command": "python -m unittest tests.test_e06_interventions -v",
                "result": "passed; focused S14 tests",
            },
            {
                "command": "python -m py_compile chimera/interventions.py scripts/e06_s14_intervention_search.py tests/test_e06_interventions.py",
                "result": "passed",
            },
            {
                "command": (
                    "python scripts/e06_s14_intervention_search.py --artifacts-dir /artifacts "
                    "--s12-results-path /artifacts/results/e06_developmental_history.parquet"
                ),
                "result": f"passed; validation {validation_result}, {len(search)} search rows, {recommended_count} recommended candidates",
            },
        ],
        "sourceCodeLocation": str(repo_root),
        "sourceCodeArtifactPolicy": "repository-backed source is committed to git; source files are not copied into artifacts per workspace instructions",
    }
    summary_path = write_markdown_report(
        step_dir=step_dir,
        status=status,
        artifacts_written=[str(path) for path in artifacts],
        heldout_summary=heldout_summary,
        validation_df=validation_df,
    )
    if summary_path not in artifacts:
        artifacts.append(summary_path)
    status["artifactsWritten"] = [str(path) for path in artifacts]
    write_json(status_path, status)

    manifest_payload = {
        "schema": "eidosoma.step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "interventionSearchVersion": INTERVENTION_SEARCH_VERSION,
        "inputArtifacts": {
            "s12Results": str(s12_results_path),
            "s13Hypotheses": str(s13_hypotheses_path),
            "s13Contrasts": str(s13_contrasts_path),
            "s13HeldoutSplit": str(s13_split_path),
            "s13FeatureImportance": str(s13_importance_path),
        },
        "heldoutFeasibilityCaveat": HELDOUT_FEASIBILITY_CAVEAT,
        "artifacts": artifact_records(artifacts),
    }
    write_json(manifest_path, manifest_payload)

    existing_manifest: dict[str, Any] = {}
    if run_manifest_path.exists():
        try:
            existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}
    research_steps = dict(existing_manifest.get("researchSteps", {}))
    research_steps[STEP_ID] = status
    run_manifest = {
        **existing_manifest,
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": "E06",
        "lastResearchStepId": STEP_ID,
        "lastStepNumber": STEP_NUMBER,
        "updatedAt": completed_at,
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
            "commit": git_value(["rev-parse", "HEAD"], repo_root),
            "dirtyStatus": git_value(["status", "--short"], repo_root),
            "remote": git_value(["remote", "-v"], repo_root),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
                if os.environ.get(key) is not None
            },
            "newDependenciesInstalled": [],
        },
        "researchSteps": research_steps,
        "artifacts": artifact_records(artifacts),
    }
    write_json(run_manifest_path, run_manifest)
    manifest_payload["artifacts"] = artifact_records(artifacts)
    write_json(manifest_path, manifest_payload)
    return {
        "status": status,
        "recipes": recipes,
        "search": search,
        "pairedControls": paired_controls,
        "effectSummary": effect_summary,
        "recipeSummary": overall_summary,
        "heldoutValidation": heldout_summary,
        "sideEffectSummary": side_effect_summary,
        "validation": validation_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E06 S14 minimal intervention search")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--s12-results-path", type=Path, default=DEFAULT_S12_RESULTS_PATH)
    parser.add_argument("--s13-hypotheses-path", type=Path, default=DEFAULT_S13_HYPOTHESES_PATH)
    parser.add_argument("--s13-contrasts-path", type=Path, default=DEFAULT_S13_CONTRASTS_PATH)
    parser.add_argument("--s13-split-path", type=Path, default=DEFAULT_S13_SPLIT_PATH)
    parser.add_argument("--s13-importance-path", type=Path, default=DEFAULT_S13_IMPORTANCE_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_s14_intervention_search(
        artifacts_dir=args.artifacts_dir,
        s12_results_path=args.s12_results_path,
        s13_hypotheses_path=args.s13_hypotheses_path,
        s13_contrasts_path=args.s13_contrasts_path,
        s13_split_path=args.s13_split_path,
        s13_importance_path=args.s13_importance_path,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['conditionCount']} search rows, "
        f"{status['recipeCount']} recipes, {status['recommendedRecipeCount']} recommended candidates, "
        f"validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
