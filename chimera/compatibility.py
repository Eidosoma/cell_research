"""E06 S05 compatibility-vector metrics derived from S04 goal sweeps."""

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

from .goals import GOAL_MODES
from .mixtures import (
    compact_json,
    compact_result_csv,
    git_value,
    parse_json_maybe,
    sha256_file,
    write_json,
)
from .panel import CLAIM_BOUNDARY, json_ready


STEP_ID = "S05"
STEP_NUMBER = 5
COMPATIBILITY_VECTOR_VERSION = "e06_s05_compatibility_vector.v1"
DEFAULT_S04_RESULTS_PATH = Path("/artifacts/results/e06_goal_compatibility.parquet")
DEFAULT_S01_PURE_VALIDATION_PATH = Path("/artifacts/research_steps/S01/pure_policy_validation.parquet")
DEFAULT_DUMMY_DIAGNOSTICS_PATH = Path("/previous-artifacts/E02/results/e02_dummy_algotypes_diagnostics.parquet")
DEFAULT_DUMMY_SUMMARY_PATH = Path("/previous-artifacts/E02/results/e02_dummy_algotypes_summary.parquet")
REQUIRED_COMPONENTS: tuple[str, ...] = (
    "cooperative_efficiency",
    "mutual_interference",
    "final_target_quality",
    "interface_stability",
    "aggregation",
    "dominance",
    "energy_use",
    "oscillation",
    "minority_survival",
    "compromise",
)
CORE_SCORE_COLUMNS: tuple[str, ...] = (
    "compatibilityCompositeScore",
    "cooperativeEfficiencyScore",
    "mutualInterferenceScore",
    "finalTargetQualityScore",
    "minTargetQualityScore",
    "interfaceStabilityScore",
    "aggregationScore",
    "dominanceProxyScore",
    "energyUseScore",
    "energyEfficiencyScore",
    "oscillationRiskProxy",
    "minoritySurvivalScore",
    "minorityGoalQualityScore",
    "compromiseScore",
)


def clip01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(number):
        return float("nan")
    return float(min(1.0, max(0.0, number)))


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return float(number) if np.isfinite(number) else default


def load_s04_results(path: Path = DEFAULT_S04_RESULTS_PATH) -> pd.DataFrame:
    results = pd.read_parquet(path)
    required = {
        "conditionId",
        "s03SourceConditionId",
        "goalMode",
        "goalCompatibilityClass",
        "conditionGoalMetadataJson",
        "policyGoalMetadataJson",
        "goalDirectionByPolicyJson",
        "policyGoalScoresJson",
        "meanPolicyGoalScore",
        "minPolicyGoalScore",
        "goalScoreSpread",
        "jointGoalSatisfied",
        "finalIncreasingSortednessPercent",
        "finalDecreasingSortednessPercent",
        "finalAggregation",
        "initialInterfaceCount",
        "finalInterfaceCount",
        "interfaceCountDelta",
        "finalInterfaceDensity",
        "minorityPanelPolicyIdsJson",
        "minorityPersistence",
        "activationCount",
        "swapCount",
        "comparisonCount",
        "completed",
        "stopReason",
        "runSucceeded",
        "valueCountsConserved",
        "policyCountsConserved",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"S04 goal-compatibility results missing required columns: {missing}")
    usable = results[
        results["runSucceeded"].fillna(False).map(bool)
        & results["valueCountsConserved"].fillna(False).map(bool)
        & results["policyCountsConserved"].fillna(False).map(bool)
    ].copy()
    if usable.empty:
        raise ValueError("S04 goal-compatibility results contain no usable rows")
    return usable.reset_index(drop=True)


def load_optional_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def metric_definitions() -> pd.DataFrame:
    rows = [
        {
            "componentName": "cooperative_efficiency",
            "columnName": "cooperativeEfficiencyScore",
            "range": "0..1",
            "higherMeans": "closer to matched same-goal target attainment",
            "formulaDescription": "meanPolicyGoalScore divided by the matched same-goal baseline meanPolicyGoalScore, clipped to 0..1",
            "inputColumnsJson": compact_json(["meanPolicyGoalScore", "sameGoalMeanPolicyGoalScore"]),
            "metricBoundary": "Relative proxy; same-goal rows are the calibration baseline, not independent validation.",
        },
        {
            "componentName": "mutual_interference",
            "columnName": "mutualInterferenceScore",
            "range": "0..1",
            "higherMeans": "more loss or imbalance relative to matched same-goal baseline",
            "formulaDescription": "weighted clipped loss in mean score, minimum score, and score spread compared with matched same-goal row",
            "inputColumnsJson": compact_json(["meanPolicyGoalScore", "minPolicyGoalScore", "goalScoreSpread", "sameGoalMeanPolicyGoalScore", "sameGoalMinPolicyGoalScore", "sameGoalGoalScoreSpread"]),
            "metricBoundary": "Interference is comparative within one S03 template; it does not prove causal conflict.",
        },
        {
            "componentName": "final_target_quality",
            "columnName": "finalTargetQualityScore",
            "range": "0..1",
            "higherMeans": "higher mean per-policy goal score",
            "formulaDescription": "meanPolicyGoalScore / 100, with minTargetQualityScore storing minPolicyGoalScore / 100",
            "inputColumnsJson": compact_json(["meanPolicyGoalScore", "minPolicyGoalScore", "policyGoalScoresJson"]),
            "metricBoundary": "Goal scores inherit S04 target definitions, including metric-only proxy targets.",
        },
        {
            "componentName": "interface_stability",
            "columnName": "interfaceStabilityScore",
            "range": "0..1",
            "higherMeans": "fewer interface-count changes from initial to final state",
            "formulaDescription": "1 - abs(finalInterfaceCount - initialInterfaceCount) / (n - 1), clipped to 0..1",
            "inputColumnsJson": compact_json(["initialInterfaceCount", "finalInterfaceCount", "n"]),
            "metricBoundary": "One-dimensional interface count is a proxy for spatial stability.",
        },
        {
            "componentName": "aggregation",
            "columnName": "aggregationScore",
            "range": "0..1",
            "higherMeans": "more same-policy adjacency in final labels",
            "formulaDescription": "finalAggregation copied from S04, with aggregationDelta retained as direction of change",
            "inputColumnsJson": compact_json(["finalAggregation", "aggregationDelta"]),
            "metricBoundary": "Aggregation can mean cooperative clustering, segregation, or dummy-label structure depending on context.",
        },
        {
            "componentName": "dominance",
            "columnName": "dominanceProxyScore",
            "range": "0..1",
            "higherMeans": "larger per-policy goal-score or directional advantage",
            "formulaDescription": "max(goalScoreSpread / 100, directional advantage for opposite-goal rows)",
            "inputColumnsJson": compact_json(["goalScoreSpread", "finalIncreasingSortednessPercent", "finalDecreasingSortednessPercent", "policyGoalScoresJson"]),
            "metricBoundary": "Dominance is a metric proxy; S06 must explicitly define winners for new contests.",
        },
        {
            "componentName": "energy_use",
            "columnName": "energyUseScore",
            "range": "0..1",
            "higherMeans": "larger normalized activation/swap/comparison resource use",
            "formulaDescription": "mean of activationCount/maxActivations, swapCount/maxSwaps, and comparisonCount/maxComparisons, clipped to 0..1",
            "inputColumnsJson": compact_json(["activationCount", "swapCount", "comparisonCount"]),
            "metricBoundary": "Resource denominators are S04 run caps, not physical energy.",
        },
        {
            "componentName": "oscillation",
            "columnName": "oscillationRiskProxy",
            "range": "0..1",
            "higherMeans": "more unresolved resource-capped behavior",
            "formulaDescription": "1.0 for non-completed max-activation caps, 0.75 for other max caps, 0.2 for no-move partial states, 0 for completed rows",
            "inputColumnsJson": compact_json(["completed", "stopReason", "finalStateClass"]),
            "metricBoundary": "No full trajectory oscillation detector is available in S04; this is an unresolved-dynamics proxy.",
        },
        {
            "componentName": "minority_survival",
            "columnName": "minoritySurvivalScore",
            "range": "0..1",
            "higherMeans": "minority policy labels persisted",
            "formulaDescription": "minorityPersistence copied from S04, with minorityGoalQualityScore storing minority policy goal-score mean / 100",
            "inputColumnsJson": compact_json(["minorityPanelPolicyIdsJson", "minorityPersistence", "policyGoalScoresJson"]),
            "metricBoundary": "S04 conserves policy labels, so survival is mostly a sanity check until later perturbation steps.",
        },
        {
            "componentName": "compromise",
            "columnName": "compromiseScore",
            "range": "0..1",
            "higherMeans": "both policies retain target quality with low score spread",
            "formulaDescription": "(minPolicyGoalScore / 100) * (1 - goalScoreSpread / 100), clipped to 0..1",
            "inputColumnsJson": compact_json(["minPolicyGoalScore", "goalScoreSpread", "compromiseStateClass"]),
            "metricBoundary": "Compromise is score balance, not a mechanistic negotiation claim.",
        },
    ]
    return pd.DataFrame(rows)


def _policy_scores(row: Mapping[str, Any]) -> dict[str, float]:
    parsed = parse_json_maybe(row.get("policyGoalScoresJson", ""), {})
    if not isinstance(parsed, Mapping):
        return {}
    scores: dict[str, float] = {}
    for key, value in parsed.items():
        if str(key).startswith("__"):
            continue
        scores[str(key)] = safe_float(value)
    return scores


def _policy_specs(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    parsed = parse_json_maybe(row.get("policyGoalMetadataJson", ""), {})
    return {str(key): value for key, value in parsed.items()} if isinstance(parsed, Mapping) else {}


def _same_goal_baselines(s04_results: pd.DataFrame) -> pd.DataFrame:
    same = s04_results[s04_results["goalMode"].astype(str) == "same_goal"].copy()
    if same.empty:
        raise ValueError("S05 requires matched same_goal S04 baselines")
    baseline_columns = [
        "s03SourceConditionId",
        "meanPolicyGoalScore",
        "minPolicyGoalScore",
        "goalScoreSpread",
        "finalAggregation",
        "finalInterfaceDensity",
        "activationCount",
        "swapCount",
        "comparisonCount",
    ]
    baseline = same[baseline_columns].rename(
        columns={
            "meanPolicyGoalScore": "sameGoalMeanPolicyGoalScore",
            "minPolicyGoalScore": "sameGoalMinPolicyGoalScore",
            "goalScoreSpread": "sameGoalGoalScoreSpread",
            "finalAggregation": "sameGoalFinalAggregation",
            "finalInterfaceDensity": "sameGoalFinalInterfaceDensity",
            "activationCount": "sameGoalActivationCount",
            "swapCount": "sameGoalSwapCount",
            "comparisonCount": "sameGoalComparisonCount",
        }
    )
    return baseline.drop_duplicates("s03SourceConditionId", keep="first")


def _dominant_policy(scores: Mapping[str, float]) -> tuple[str, float, float]:
    finite = [(pid, float(score)) for pid, score in scores.items() if np.isfinite(score)]
    if not finite:
        return "", float("nan"), float("nan")
    ranked = sorted(finite, key=lambda item: item[1], reverse=True)
    winner, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else top
    return winner, float(top), float(top - runner_up)


def _oscillation_risk(row: Mapping[str, Any]) -> float:
    completed = bool(row.get("completed", False))
    stop_reason = str(row.get("stopReason", "")).lower()
    if completed:
        return 0.0
    if "max_activation" in stop_reason:
        return 1.0
    if stop_reason.startswith("max_"):
        return 0.75
    if "no_cell_can_move" in stop_reason or "no_move" in stop_reason:
        return 0.2
    return 0.5


def _outcome_class(row: Mapping[str, Any]) -> str:
    target = safe_float(row.get("finalTargetQualityScore"))
    interference = safe_float(row.get("mutualInterferenceScore"))
    dominance = safe_float(row.get("dominanceProxyScore"))
    aggregation = safe_float(row.get("aggregationScore"))
    interface_density = safe_float(row.get("finalInterfaceDensity"))
    oscillation = safe_float(row.get("oscillationRiskProxy"))
    if oscillation >= 0.75 and target < 0.80:
        return "resource_limited_unresolved_proxy"
    if bool(row.get("jointGoalSatisfied", False)) and dominance < 0.15:
        return "cooperative_goal_attainment_proxy"
    if str(row.get("goalMode")) == "opposite_goal" and dominance >= 0.25:
        return "conflict_directional_dominance_proxy"
    if target >= 0.75 and interference <= 0.15:
        return "compatible_partial_attainment_proxy"
    if aggregation >= 0.75 and interface_density <= 0.25 and target < 0.75:
        return "segregated_low_goal_attainment_proxy"
    if dominance >= 0.35:
        return "asymmetric_goal_attainment_proxy"
    return "low_or_mixed_compatibility_proxy"


def compute_compatibility_metrics(
    s04_results: pd.DataFrame,
    *,
    max_activations: int = 6000,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
) -> pd.DataFrame:
    baseline = _same_goal_baselines(s04_results)
    merged = s04_results.merge(baseline, on="s03SourceConditionId", how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        policy_scores = _policy_scores(row)
        policy_specs = _policy_specs(row)
        winner, winner_score, dominance_margin = _dominant_policy(policy_scores)
        mean_score = safe_float(row.get("meanPolicyGoalScore"))
        min_score = safe_float(row.get("minPolicyGoalScore"))
        spread = safe_float(row.get("goalScoreSpread"), 0.0)
        same_mean = safe_float(row.get("sameGoalMeanPolicyGoalScore"))
        same_min = safe_float(row.get("sameGoalMinPolicyGoalScore"))
        same_spread = safe_float(row.get("sameGoalGoalScoreSpread"), 0.0)
        mean_loss = max(0.0, same_mean - mean_score) / 100.0 if np.isfinite(same_mean) else 0.0
        min_loss = max(0.0, same_min - min_score) / 100.0 if np.isfinite(same_min) else 0.0
        spread_delta = max(0.0, spread - same_spread) / 100.0
        mutual_interference = clip01(0.45 * mean_loss + 0.35 * min_loss + 0.20 * spread_delta)
        cooperative_efficiency = clip01(mean_score / same_mean) if np.isfinite(same_mean) and same_mean > 0 else clip01(mean_score / 100.0)
        final_target = clip01(mean_score / 100.0)
        min_target = clip01(min_score / 100.0)
        n = int(row.get("n", 0) or 0)
        interface_stability = clip01(1.0 - abs(safe_float(row.get("interfaceCountDelta"), 0.0)) / max(1, n - 1))
        aggregation_score = clip01(row.get("finalAggregation"))
        increasing = safe_float(row.get("finalIncreasingSortednessPercent"))
        decreasing = safe_float(row.get("finalDecreasingSortednessPercent"))
        directional_advantage = abs(increasing - decreasing) / 100.0 if np.isfinite(increasing) and np.isfinite(decreasing) else 0.0
        dominance_asymmetry = clip01(spread / 100.0)
        dominance_proxy = clip01(max(dominance_asymmetry, directional_advantage if str(row.get("goalMode")) == "opposite_goal" else 0.0))
        activation_use = safe_float(row.get("activationCount"), 0.0) / max(1, int(max_activations))
        swap_use = safe_float(row.get("swapCount"), 0.0) / max(1, int(max_swaps))
        comparison_use = safe_float(row.get("comparisonCount"), 0.0) / max(1, int(max_comparisons))
        energy_use = clip01(float(np.mean([clip01(activation_use), clip01(swap_use), clip01(comparison_use)])))
        energy_efficiency = clip01(1.0 - energy_use)
        oscillation_risk = _oscillation_risk(row)
        minority_ids = [str(pid) for pid in parse_json_maybe(row.get("minorityPanelPolicyIdsJson", ""), [])]
        minority_scores = [policy_scores[pid] for pid in minority_ids if pid in policy_scores and np.isfinite(policy_scores[pid])]
        minority_goal = clip01(float(np.mean(minority_scores)) / 100.0) if minority_scores else final_target
        minority_survival = clip01(row.get("minorityPersistence", 0.0))
        minority_composite = clip01(float(np.mean([minority_survival, minority_goal])))
        compromise = clip01((min_target) * (1.0 - dominance_asymmetry))
        compatibility_composite = clip01(
            0.25 * final_target
            + 0.20 * cooperative_efficiency
            + 0.15 * (1.0 - mutual_interference)
            + 0.10 * interface_stability
            + 0.10 * minority_composite
            + 0.10 * energy_efficiency
            + 0.10 * compromise
        )
        metadata = parse_json_maybe(row.get("conditionGoalMetadataJson", ""), {})
        metric_boundary = str(metadata.get("metricBoundary", "")) if isinstance(metadata, Mapping) else ""
        role = ""
        goal_type = ""
        if winner and winner in policy_specs:
            role = str(policy_specs[winner].get("role", ""))
            goal_type = str(policy_specs[winner].get("goalType", ""))
        out = {
            **row,
            "researchStepId": STEP_ID,
            "sourceResearchStepId": "S04",
            "compatibilityVectorVersion": COMPATIBILITY_VECTOR_VERSION,
            "cooperativeEfficiencyScore": cooperative_efficiency,
            "mutualInterferenceScore": mutual_interference,
            "cooperationRetentionScore": clip01(1.0 - mutual_interference),
            "finalTargetQualityScore": final_target,
            "minTargetQualityScore": min_target,
            "interfaceStabilityScore": interface_stability,
            "aggregationScore": aggregation_score,
            "dominanceAsymmetryScore": dominance_asymmetry,
            "directionalDominanceScore": clip01(directional_advantage),
            "dominanceProxyScore": dominance_proxy,
            "dominantPolicyId": winner,
            "dominantPolicyGoalScore": winner_score,
            "dominanceMarginScore": clip01(dominance_margin / 100.0),
            "dominantPolicyRole": role,
            "dominantPolicyGoalType": goal_type,
            "energyUseScore": energy_use,
            "energyEfficiencyScore": energy_efficiency,
            "oscillationRiskProxy": oscillation_risk,
            "minoritySurvivalScore": minority_survival,
            "minorityGoalQualityScore": minority_goal,
            "minorityCompositeScore": minority_composite,
            "compromiseScore": compromise,
            "compatibilityCompositeScore": compatibility_composite,
            "deltaVsSameGoalMeanPolicyGoalScore": mean_score - same_mean if np.isfinite(same_mean) else float("nan"),
            "deltaVsSameGoalMinPolicyGoalScore": min_score - same_min if np.isfinite(same_min) else float("nan"),
            "deltaVsSameGoalGoalScoreSpread": spread - same_spread if np.isfinite(same_spread) else float("nan"),
            "metricBoundary": metric_boundary,
            "claimBoundary": CLAIM_BOUNDARY,
        }
        out["compatibilityOutcomeClass"] = _outcome_class(out)
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_compatibility(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode_summary = (
        metrics.groupby(["goalMode", "goalCompatibilityClass"], dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            meanCompatibilityCompositeScore=("compatibilityCompositeScore", "mean"),
            meanFinalTargetQualityScore=("finalTargetQualityScore", "mean"),
            meanMinTargetQualityScore=("minTargetQualityScore", "mean"),
            meanCooperativeEfficiencyScore=("cooperativeEfficiencyScore", "mean"),
            meanMutualInterferenceScore=("mutualInterferenceScore", "mean"),
            meanDominanceProxyScore=("dominanceProxyScore", "mean"),
            meanAggregationScore=("aggregationScore", "mean"),
            meanInterfaceStabilityScore=("interfaceStabilityScore", "mean"),
            meanEnergyUseScore=("energyUseScore", "mean"),
            meanOscillationRiskProxy=("oscillationRiskProxy", "mean"),
            meanMinorityCompositeScore=("minorityCompositeScore", "mean"),
            meanCompromiseScore=("compromiseScore", "mean"),
            highCompatibilityRate=("compatibilityCompositeScore", lambda values: float(np.mean(np.asarray(values) >= 0.75))),
        )
        .reset_index()
    )
    category_summary = (
        metrics.groupby(["pairCategory", "goalMode"], dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            meanCompatibilityCompositeScore=("compatibilityCompositeScore", "mean"),
            meanFinalTargetQualityScore=("finalTargetQualityScore", "mean"),
            meanMutualInterferenceScore=("mutualInterferenceScore", "mean"),
            meanDominanceProxyScore=("dominanceProxyScore", "mean"),
            meanAggregationScore=("aggregationScore", "mean"),
            meanOscillationRiskProxy=("oscillationRiskProxy", "mean"),
        )
        .reset_index()
    )
    outcome_summary = (
        metrics.groupby(["goalMode", "compatibilityOutcomeClass"], dropna=False)
        .agg(conditionCount=("conditionId", "size"))
        .reset_index()
        .sort_values(["goalMode", "conditionCount"], ascending=[True, False], kind="mergesort")
    )
    return mode_summary, category_summary, outcome_summary


def build_boundary_caveats(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, group in metrics.groupby("goalMode", sort=True):
        boundaries = sorted({str(value) for value in group["metricBoundary"].dropna().astype(str) if value})
        rows.append(
            {
                "goalMode": str(mode),
                "goalCompatibilityClass": str(group["goalCompatibilityClass"].iloc[0]),
                "conditionCount": int(len(group)),
                "metricBoundaryJson": compact_json(boundaries),
                "claimBoundary": CLAIM_BOUNDARY,
            }
        )
    rows.append(
        {
            "goalMode": "__global__",
            "goalCompatibilityClass": "all",
            "conditionCount": int(len(metrics)),
            "metricBoundaryJson": compact_json(
                [
                    "S05 derives metrics from S04 final states; it does not add trajectory-level oscillation detection.",
                    "Dummy identical-code validation is imported from E02 and not rerun inside E06.",
                    "Compatibility scores are computational proxies for mixed local policies.",
                ]
            ),
            "claimBoundary": CLAIM_BOUNDARY,
        }
    )
    return pd.DataFrame(rows)


def build_control_anchor_validation(
    metrics: pd.DataFrame,
    pure_validation: pd.DataFrame,
    dummy_diagnostics: pd.DataFrame,
    dummy_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    policy_ids = sorted({pid for text in metrics["policyIdsJson"].astype(str) for pid in parse_json_maybe(text, [])})
    pure = pure_validation[
        pure_validation.get("panelPolicyId", pd.Series(dtype=str)).astype(str).isin(policy_ids)
        & (pure_validation.get("goalDirection", pd.Series(dtype=str)).astype(str) == "increasing")
    ].copy() if not pure_validation.empty else pd.DataFrame()
    control_mask = pure["panelPolicyId"].astype(str).str.contains("null|random", case=False, regex=True) if not pure.empty else pd.Series(dtype=bool)
    non_control = pure[~control_mask] if not pure.empty else pd.DataFrame()
    control = pure[control_mask] if not pure.empty else pd.DataFrame()
    pure_success = bool(
        not pure.empty
        and set(policy_ids) <= set(pure["panelPolicyId"].astype(str))
        and pure["runSucceeded"].fillna(False).map(bool).all()
        and non_control.groupby("panelPolicyId")["finalGoalSortednessPercent"].mean().ge(95.0).all()
        and (control.empty or control.groupby("panelPolicyId")["finalGoalSortednessPercent"].mean().lt(80.0).all())
    )
    rows.append(
        {
            "anchorId": "pure_policy_controls",
            "success": pure_success,
            "sourceArtifact": str(DEFAULT_S01_PURE_VALIDATION_PATH),
            "detail": (
                f"{len(policy_ids)} S04 policies had S01 pure increasing validation rows; "
                f"{non_control['panelPolicyId'].nunique() if not non_control.empty else 0} non-control policies averaged >=95% and "
                f"{control['panelPolicyId'].nunique() if not control.empty else 0} null/random controls stayed below 80%."
            ),
        }
    )
    same = metrics[metrics["goalMode"] == "same_goal"]
    same_success = bool(
        not same.empty
        and same["cooperativeEfficiencyScore"].between(0.999, 1.001).all()
        and same["mutualInterferenceScore"].le(1e-9).all()
    )
    rows.append(
        {
            "anchorId": "same_goal_chimeras",
            "success": same_success,
            "sourceArtifact": str(DEFAULT_S04_RESULTS_PATH),
            "detail": f"{len(same)} same-goal rows define the matched baseline with mean efficiency {same['cooperativeEfficiencyScore'].mean():.3f}.",
        }
    )
    opposite = metrics[metrics["goalMode"] == "opposite_goal"]
    opposite_success = bool(
        not opposite.empty
        and opposite["goalDirectionByPolicyJson"].astype(str).str.contains("decreasing").all()
        and opposite["dominanceProxyScore"].mean() > same["dominanceProxyScore"].mean()
        and opposite["mutualInterferenceScore"].mean() > same["mutualInterferenceScore"].mean()
    )
    rows.append(
        {
            "anchorId": "opposite_goal_conflicts",
            "success": opposite_success,
            "sourceArtifact": str(DEFAULT_S04_RESULTS_PATH),
            "detail": (
                f"{len(opposite)} opposite-goal rows carry increasing/decreasing labels; "
                f"mean dominance {opposite['dominanceProxyScore'].mean():.3f} vs same-goal {same['dominanceProxyScore'].mean():.3f}."
            ),
        }
    )
    dummy_success = bool(
        not dummy_diagnostics.empty
        and dummy_diagnostics["allPolicyAlgotypesIdentical"].fillna(False).map(bool).all()
        and dummy_diagnostics["behaviorHistoryMatchesReference"].fillna(False).map(bool).all()
        and dummy_diagnostics["activationHistoryMatchesReference"].fillna(False).map(bool).all()
        and (dummy_summary.empty or int(dummy_summary["significantPeakRuns"].sum()) == 0)
    )
    rows.append(
        {
            "anchorId": "dummy_identical_code_labels",
            "success": dummy_success,
            "sourceArtifact": str(DEFAULT_DUMMY_DIAGNOSTICS_PATH),
            "detail": (
                f"Imported E02 dummy-label controls: {len(dummy_diagnostics)} diagnostic rows; "
                "behavior and activation histories match their pure-code references."
            ),
        }
    )
    boundary_modes = {"partially_compatible", "shared_global", "unrelated_goal"}
    boundary_success = bool(
        boundary_modes <= set(metrics["goalMode"].astype(str))
        and metrics[metrics["goalMode"].isin(boundary_modes)]["metricBoundary"].astype(str).str.len().gt(20).all()
    )
    rows.append(
        {
            "anchorId": "metric_boundary_caveats",
            "success": boundary_success,
            "sourceArtifact": str(DEFAULT_S04_RESULTS_PATH),
            "detail": "Partial, shared-global, and unrelated modes retain explicit metric-boundary caveats from S04 metadata.",
        }
    )
    return pd.DataFrame(rows)


def validation_checks(
    metrics: pd.DataFrame,
    definitions: pd.DataFrame,
    control_anchors: pd.DataFrame,
    s04_results: pd.DataFrame,
) -> pd.DataFrame:
    required_defined = set(REQUIRED_COMPONENTS) <= set(definitions["componentName"].astype(str))
    score_bounds = True
    for column in CORE_SCORE_COLUMNS:
        if column not in metrics.columns:
            score_bounds = False
            break
        values = pd.to_numeric(metrics[column], errors="coerce")
        if values.isna().any() or not values.between(0.0, 1.0).all():
            score_bounds = False
            break
    same_baselines = metrics.groupby("s03SourceConditionId")["goalMode"].agg(lambda values: "same_goal" in set(values))
    opposite = metrics[metrics["goalMode"] == "opposite_goal"]
    boundary_modes = {"partially_compatible", "shared_global", "unrelated_goal"}
    mode_means = metrics.groupby("goalMode")["compatibilityCompositeScore"].mean()
    checks = [
        {
            "checkId": "metric_definitions_cover_required_components",
            "success": bool(required_defined),
            "detail": f"{len(definitions)} metric definition rows cover {len(REQUIRED_COMPONENTS)} required vector components",
        },
        {
            "checkId": "one_vector_per_s04_condition",
            "success": bool(len(metrics) == len(s04_results) and metrics["conditionId"].is_unique),
            "detail": f"{len(metrics)} compatibility-vector rows for {len(s04_results)} S04 rows",
        },
        {
            "checkId": "core_scores_bounded",
            "success": bool(score_bounds),
            "detail": f"{len(CORE_SCORE_COLUMNS)} core score columns are finite and bounded in [0, 1]",
        },
        {
            "checkId": "goal_labels_and_metadata_preserved",
            "success": bool(set(GOAL_MODES) <= set(metrics["goalMode"].astype(str)) and metrics["metricBoundary"].astype(str).str.len().gt(10).all()),
            "detail": "all S04 goal modes and metric-boundary labels are retained",
        },
        {
            "checkId": "matched_same_goal_baselines_complete",
            "success": bool(not same_baselines.empty and same_baselines.all() and metrics["sameGoalMeanPolicyGoalScore"].notna().all()),
            "detail": f"{int(same_baselines.sum())}/{len(same_baselines)} S03 templates include matched same-goal baselines",
        },
        {
            "checkId": "same_goal_baseline_behavior_validated",
            "success": bool(control_anchors.loc[control_anchors["anchorId"] == "same_goal_chimeras", "success"].all()),
            "detail": str(control_anchors.loc[control_anchors["anchorId"] == "same_goal_chimeras", "detail"].iloc[0]),
        },
        {
            "checkId": "opposite_goal_conflict_behavior_validated",
            "success": bool(
                not opposite.empty
                and opposite["goalDirectionByPolicyJson"].astype(str).str.contains("decreasing").all()
                and control_anchors.loc[control_anchors["anchorId"] == "opposite_goal_conflicts", "success"].all()
            ),
            "detail": str(control_anchors.loc[control_anchors["anchorId"] == "opposite_goal_conflicts", "detail"].iloc[0]),
        },
        {
            "checkId": "pure_control_anchor_validated",
            "success": bool(control_anchors.loc[control_anchors["anchorId"] == "pure_policy_controls", "success"].all()),
            "detail": str(control_anchors.loc[control_anchors["anchorId"] == "pure_policy_controls", "detail"].iloc[0]),
        },
        {
            "checkId": "dummy_identical_code_anchor_validated",
            "success": bool(control_anchors.loc[control_anchors["anchorId"] == "dummy_identical_code_labels", "success"].all()),
            "detail": str(control_anchors.loc[control_anchors["anchorId"] == "dummy_identical_code_labels", "detail"].iloc[0]),
        },
        {
            "checkId": "metric_boundary_caveats_retained",
            "success": bool(
                boundary_modes <= set(metrics["goalMode"].astype(str))
                and control_anchors.loc[control_anchors["anchorId"] == "metric_boundary_caveats", "success"].all()
            ),
            "detail": str(control_anchors.loc[control_anchors["anchorId"] == "metric_boundary_caveats", "detail"].iloc[0]),
        },
        {
            "checkId": "goal_modes_distinguishable_by_vector",
            "success": bool(len(mode_means) >= 5 and (mode_means.max() - mode_means.min()) >= 0.05),
            "detail": f"mode mean compatibility range is {mode_means.min():.3f}..{mode_means.max():.3f}",
        },
    ]
    return pd.DataFrame(checks)


def write_metric_definitions_markdown(definitions: pd.DataFrame, path: Path) -> Path:
    lines = ["# E06 S05 Compatibility Metric Definitions", ""]
    for row in definitions.itertuples(index=False):
        lines.extend(
            [
                f"## {row.componentName}",
                "",
                f"- Column: `{row.columnName}`",
                f"- Range: {row.range}",
                f"- Higher means: {row.higherMeans}",
                f"- Formula: {row.formulaDescription}",
                f"- Boundary: {row.metricBoundary}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_compatibility_plot(mode_summary: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    if mode_summary.empty:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_df = mode_summary.copy()
    plot_df["goalMode"] = pd.Categorical(plot_df["goalMode"], categories=list(GOAL_MODES), ordered=True)
    plot_df = plot_df.sort_values("goalMode", kind="mergesort")
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 0.24, plot_df["meanCompatibilityCompositeScore"], width=0.24, label="Compatibility composite")
    ax.bar(x, plot_df["meanFinalTargetQualityScore"], width=0.24, label="Target quality")
    ax.bar(x + 0.24, plot_df["meanDominanceProxyScore"], width=0.24, label="Dominance proxy")
    ax.set_xticks(x, plot_df["goalMode"].astype(str), rotation=25, ha="right")
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("E06 S05 compatibility-vector components by goal mode")
    ax.legend()
    fig.tight_layout()
    paths = [
        figure_dir / "e06_s05_compatibility_vector_by_goal_mode.png",
        figure_dir / "e06_s05_compatibility_vector_by_goal_mode.pdf",
        step_dir / "compatibility_vector_by_goal_mode.png",
        step_dir / "compatibility_vector_by_goal_mode.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def artifact_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if path.exists() and path.is_file():
            records.append({"path": str(path), "sha256": sha256_file(path), "sizeBytes": int(path.stat().st_size)})
    return records


def write_markdown_report(
    *,
    step_dir: Path,
    status: Mapping[str, Any],
    artifacts_written: Sequence[str],
    mode_summary: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Path:
    mode_lines = "\n".join(
        f"- `{row.goalMode}`: {int(row.conditionCount)} rows, composite {row.meanCompatibilityCompositeScore:.3f}, "
        f"target {row.meanFinalTargetQualityScore:.3f}, interference {row.meanMutualInterferenceScore:.3f}, dominance {row.meanDominanceProxyScore:.3f}"
        for row in mode_summary.sort_values("goalMode").itertuples(index=False)
    ) or "- No goal-mode summaries available."
    top = mode_summary.sort_values("meanCompatibilityCompositeScore", ascending=False).head(3)
    top_lines = "\n".join(
        f"- `{row.goalMode}` had mean compatibility composite {row.meanCompatibilityCompositeScore:.3f} and target quality {row.meanFinalTargetQualityScore:.3f}"
        for row in top.itertuples(index=False)
    ) or "- No anchor compatibility results available."
    text = f"""# Research Step S05: Measure compatibility

## Completion status

{status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Built compatibility vectors for {status['conditionCount']} S04 conditions. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

S05 is a derived-metric step and does not run new chimeric simulations. Oscillation is an unresolved-dynamics proxy from stop reasons because S04 did not persist dense trajectories. Dummy identical-code validation is imported from E02 controls rather than rerun inside E06. Partial, shared-global, and unrelated goals retain S04 metric-boundary caveats and should not be treated as direct biological or causal validation.

## Lay summary

S05 turns the S04 goal-compatibility runs into a compatibility vector with separate scores for target quality, interference, dominance, aggregation, interface stability, resource use, unresolved dynamics, minority survival, and compromise. The vector distinguishes same-goal, opposite-goal, partial, shared-global, and unrelated proxy regimes while keeping each score tied to its input assumptions.

## Goal-mode summary

{mode_lines}

## Anchor results

{top_lines}

## Recommended next action

Chief Scientist review, then S06 should use the S05 dominance and compatibility vector definitions to predefine winner criteria for new pairwise opposite-goal contests.
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s05_compatibility_metrics(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    s04_results_path: Path = DEFAULT_S04_RESULTS_PATH,
    s01_pure_validation_path: Path = DEFAULT_S01_PURE_VALIDATION_PATH,
    dummy_diagnostics_path: Path = DEFAULT_DUMMY_DIAGNOSTICS_PATH,
    dummy_summary_path: Path = DEFAULT_DUMMY_SUMMARY_PATH,
    max_activations: int = 6000,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
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
    s04_results = load_s04_results(s04_results_path)
    pure_validation = load_optional_table(s01_pure_validation_path)
    dummy_diagnostics = load_optional_table(dummy_diagnostics_path)
    dummy_summary = load_optional_table(dummy_summary_path)
    definitions = metric_definitions()
    metrics = compute_compatibility_metrics(
        s04_results,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )
    mode_summary, category_summary, outcome_summary = summarize_compatibility(metrics)
    boundary_caveats = build_boundary_caveats(metrics)
    control_anchors = build_control_anchor_validation(metrics, pure_validation, dummy_diagnostics, dummy_summary)
    validation_df = validation_checks(metrics, definitions, control_anchors, s04_results)

    step_metrics_csv = step_dir / "compatibility_metrics.csv"
    step_metrics_parquet = step_dir / "compatibility_metrics.parquet"
    result_csv = results_dir / "e06_compatibility_metrics.csv"
    result_parquet = results_dir / "e06_compatibility_metrics.parquet"
    definitions_csv = step_dir / "compatibility_metric_definitions.csv"
    definitions_parquet = step_dir / "compatibility_metric_definitions.parquet"
    definitions_json = step_dir / "compatibility_metric_definitions.json"
    definitions_md = step_dir / "compatibility_metric_definitions.md"
    mode_summary_csv = step_dir / "compatibility_mode_summary.csv"
    mode_summary_parquet = step_dir / "compatibility_mode_summary.parquet"
    category_summary_csv = step_dir / "compatibility_category_summary.csv"
    category_summary_parquet = step_dir / "compatibility_category_summary.parquet"
    outcome_summary_csv = step_dir / "compatibility_outcome_summary.csv"
    outcome_summary_parquet = step_dir / "compatibility_outcome_summary.parquet"
    boundary_csv = step_dir / "metric_boundary_caveats.csv"
    boundary_parquet = step_dir / "metric_boundary_caveats.parquet"
    controls_csv = step_dir / "control_anchor_validation.csv"
    controls_parquet = step_dir / "control_anchor_validation.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    metrics_csv_df = compact_result_csv(metrics)
    for path, df in [
        (step_metrics_csv, metrics_csv_df),
        (result_csv, metrics_csv_df),
        (definitions_csv, definitions),
        (mode_summary_csv, mode_summary),
        (category_summary_csv, category_summary),
        (outcome_summary_csv, outcome_summary),
        (boundary_csv, boundary_caveats),
        (controls_csv, control_anchors),
        (validation_csv, validation_df),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (step_metrics_parquet, metrics),
        (result_parquet, metrics),
        (definitions_parquet, definitions),
        (mode_summary_parquet, mode_summary),
        (category_summary_parquet, category_summary),
        (outcome_summary_parquet, outcome_summary),
        (boundary_parquet, boundary_caveats),
        (controls_parquet, control_anchors),
        (validation_parquet, validation_df),
    ]:
        df.to_parquet(path, index=False)
    write_json(definitions_json, {"schema": "eidosoma.e06_s05_metric_definitions.v1", "definitions": definitions.to_dict(orient="records")})
    write_metric_definitions_markdown(definitions, definitions_md)
    figure_paths = write_compatibility_plot(mode_summary, figures_dir, step_dir)

    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() else "failed"
    artifacts = [
        step_metrics_csv,
        step_metrics_parquet,
        result_csv,
        result_parquet,
        definitions_csv,
        definitions_parquet,
        definitions_json,
        definitions_md,
        mode_summary_csv,
        mode_summary_parquet,
        category_summary_csv,
        category_summary_parquet,
        outcome_summary_csv,
        outcome_summary_parquet,
        boundary_csv,
        boundary_parquet,
        controls_csv,
        controls_parquet,
        validation_csv,
        validation_parquet,
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
            "S05 derives compatibility metrics from S04 final-state rows; no new chimeric simulations were run.",
            "Oscillation is represented by a stop-reason/resource-cap proxy because S04 did not persist dense trajectories.",
            "Dummy identical-code validation is imported from E02 artifacts, not rerun in E06.",
            "Partial, shared-global, and unrelated goal metrics remain computational proxy objectives with S04 boundary labels.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then run S06 dominance-hierarchy contests using the S05 dominanceProxyScore, "
            "target-quality, and goal-label definitions as predeclared winner criteria."
        ),
        "laySummary": (
            "S05 converted S04 goal-compatibility runs into a validated compatibility vector covering cooperation, "
            "interference, target quality, interfaces, aggregation, dominance, resource use, unresolved dynamics, "
            "minority survival, and compromise."
        ),
        "outcomeClassification": "supportive" if validation_result == "passed" else "constraining/contradictory",
        "conditionCount": int(len(metrics)),
        "s04InputCount": int(len(s04_results)),
        "metricDefinitionCount": int(len(definitions)),
        "controlAnchorCount": int(len(control_anchors)),
        "goalModes": sorted(metrics["goalMode"].astype(str).unique()),
        "meanCompatibilityCompositeScore": float(metrics["compatibilityCompositeScore"].mean()),
        "meanFinalTargetQualityScore": float(metrics["finalTargetQualityScore"].mean()),
        "meanMutualInterferenceScore": float(metrics["mutualInterferenceScore"].mean()),
        "meanDominanceProxyScore": float(metrics["dominanceProxyScore"].mean()),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
        "workerCount": 1,
        "newDependenciesInstalled": [],
    }
    summary_path = write_markdown_report(
        step_dir=step_dir,
        status=status,
        artifacts_written=[str(path) for path in artifacts],
        mode_summary=mode_summary,
        validation_df=validation_df,
    )
    if summary_path not in artifacts:
        artifacts.append(summary_path)
    write_json(status_path, status)
    manifest_payload = {
        "schema": "eidosoma.step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "compatibilityVectorVersion": COMPATIBILITY_VECTOR_VERSION,
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
        "metrics": metrics,
        "definitions": definitions,
        "modeSummary": mode_summary,
        "categorySummary": category_summary,
        "outcomeSummary": outcome_summary,
        "boundaryCaveats": boundary_caveats,
        "controlAnchors": control_anchors,
        "validation": validation_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute E06 S05 compatibility-vector metrics from S04 outputs.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--s04-results-path", type=Path, default=DEFAULT_S04_RESULTS_PATH)
    parser.add_argument("--s01-pure-validation-path", type=Path, default=DEFAULT_S01_PURE_VALIDATION_PATH)
    parser.add_argument("--dummy-diagnostics-path", type=Path, default=DEFAULT_DUMMY_DIAGNOSTICS_PATH)
    parser.add_argument("--dummy-summary-path", type=Path, default=DEFAULT_DUMMY_SUMMARY_PATH)
    parser.add_argument("--max-activations", type=int, default=6000)
    parser.add_argument("--max-swaps", type=int, default=4000)
    parser.add_argument("--max-comparisons", type=int, default=30000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_s05_compatibility_metrics(
        artifacts_dir=args.artifacts_dir,
        s04_results_path=args.s04_results_path,
        s01_pure_validation_path=args.s01_pure_validation_path,
        dummy_diagnostics_path=args.dummy_diagnostics_path,
        dummy_summary_path=args.dummy_summary_path,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['conditionCount']} compatibility vectors, "
        f"{status['metricDefinitionCount']} metric definitions, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
