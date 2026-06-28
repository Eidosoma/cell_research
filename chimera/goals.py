"""E06 S04 goal-compatibility sweeps."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e02_deterministic_simulator.metrics import aggregation, sortedness_percent

from .arrangements import (
    classify_mosaic,
    interface_count,
    mean_positions,
)
from .mixtures import (
    DEFAULT_PANEL_PATH,
    compact_json,
    compact_result_csv,
    git_value,
    load_ready_panel,
    parse_json_maybe,
    run_conditions,
    run_mixture_condition,
    sha256_file,
    sha256_text,
    write_json,
)
from .panel import CLAIM_BOUNDARY, json_ready


STEP_ID = "S04"
STEP_NUMBER = 4
GOAL_SWEEP_VERSION = "e06_s04_goal_compatibility.v1"
DEFAULT_S03_RESULTS_PATH = Path("/artifacts/results/e06_initial_arrangements.parquet")
GOAL_MODES: tuple[str, ...] = (
    "same_goal",
    "opposite_goal",
    "partially_compatible",
    "shared_global",
    "unrelated_goal",
)
PAIR_CATEGORY_ORDER: tuple[str, ...] = (
    "original_pair",
    "frontier_pair",
    "handoff_vs_frontier",
    "memory_signal_handoff_vs_original",
    "e03_vs_original",
    "control_pair",
)
GOAL_MODE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "same_goal": {
        "label": "same increasing goal",
        "compatibilityClass": "same",
        "conditionTarget": "all policies evaluate strict increasing global value order",
        "metricBoundary": "direct directional Sortedness proxy",
    },
    "opposite_goal": {
        "label": "opposite increasing/decreasing goals",
        "compatibilityClass": "opposite",
        "conditionTarget": "left policy evaluates increasing order, right policy evaluates decreasing order",
        "metricBoundary": "per-cell reverse directions change behavior; dominance is a proxy from directional scores",
    },
    "partially_compatible": {
        "label": "strict plus coarse compatible goals",
        "compatibilityClass": "partial",
        "conditionTarget": "left policy evaluates strict increasing order, right policy evaluates coarse increasing quartile order",
        "metricBoundary": "right-policy target is a metric-level coarse-order proxy",
    },
    "shared_global": {
        "label": "shared global increasing goal",
        "compatibilityClass": "shared_global",
        "conditionTarget": "both policies receive credit for the same global increasing sortedness target",
        "metricBoundary": "shared global target is evaluated as whole-array Sortedness",
    },
    "unrelated_goal": {
        "label": "unrelated ordering and identity-aggregation goals",
        "compatibilityClass": "unrelated",
        "conditionTarget": "left policy evaluates strict increasing order, right policy evaluates policy-identity aggregation",
        "metricBoundary": "identity aggregation is a proxy objective; no new aggregation-seeking controller is introduced",
    },
}


def load_s03_results(path: Path = DEFAULT_S03_RESULTS_PATH) -> pd.DataFrame:
    results = pd.read_parquet(path)
    required = {
        "conditionId",
        "pairCategory",
        "leftPanelPolicyId",
        "rightPanelPolicyId",
        "ratioLabel",
        "arrangementType",
        "policyIdsJson",
        "policyCountsJson",
        "targetCountsJson",
        "targetProportionsJson",
        "arrangementPolicyIdsJson",
        "arrangementNumericLabelsJson",
        "arrangementDescriptorJson",
        "valueSeed",
        "arrangementSeed",
        "schedulerSeed",
        "tieBreakerSeed",
        "seedIndex",
        "runSucceeded",
        "actualRatiosMatchTarget",
        "finalSortednessPercent",
        "finalAggregation",
        "interfaceStabilityScore",
        "policyPositionChangeFraction",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"S03 results missing required columns: {missing}")
    usable = results[results["runSucceeded"].map(bool) & results["actualRatiosMatchTarget"].map(bool)].copy()
    if usable.empty:
        raise ValueError("S03 results contain no usable rows")
    return usable.reset_index(drop=True)


def _s03_sensitivity_table(s03_results: pd.DataFrame) -> pd.DataFrame:
    df = s03_results.copy()
    random = df[df["arrangementType"] == "random"][
        ["baseMatchId", "finalSortednessPercent", "finalAggregation", "interfaceStabilityScore"]
    ].rename(
        columns={
            "finalSortednessPercent": "randomFinalSortednessPercent",
            "finalAggregation": "randomFinalAggregation",
            "interfaceStabilityScore": "randomInterfaceStabilityScore",
        }
    )
    out = df.merge(random, on="baseMatchId", how="left")
    out["deltaVsRandomFinalSortednessPercent"] = out["finalSortednessPercent"] - out["randomFinalSortednessPercent"]
    out["deltaVsRandomFinalAggregation"] = out["finalAggregation"] - out["randomFinalAggregation"]
    out["deltaVsRandomInterfaceStabilityScore"] = out["interfaceStabilityScore"] - out["randomInterfaceStabilityScore"]
    out["s03SensitivityScore"] = (
        out["deltaVsRandomFinalSortednessPercent"].abs().fillna(0.0)
        + 20.0 * out["deltaVsRandomFinalAggregation"].abs().fillna(0.0)
        + 10.0 * (1.0 - out["interfaceStabilityScore"].fillna(1.0))
        + 5.0 * out["policyPositionChangeFraction"].fillna(0.0)
    )
    category_rank = {category: rank for rank, category in enumerate(PAIR_CATEGORY_ORDER)}
    out["categoryRank"] = out["pairCategory"].map(category_rank).fillna(99).astype(int)
    return out


def select_s04_base_conditions(
    s03_results: pd.DataFrame,
    *,
    max_conditions_per_category: int = 4,
    max_total_conditions: int = 24,
) -> pd.DataFrame:
    sensitive = _s03_sensitivity_table(s03_results)
    selected: dict[str, dict[str, Any]] = {}

    def add(rows: pd.DataFrame, reason: str) -> None:
        for row in rows.to_dict(orient="records"):
            key = str(row["conditionId"])
            existing = selected.get(key)
            if existing is None:
                row["s04SelectionReason"] = reason
                selected[key] = row
            elif reason not in str(existing["s04SelectionReason"]).split("+"):
                existing["s04SelectionReason"] = f"{existing['s04SelectionReason']}+{reason}"

    for category in PAIR_CATEGORY_ORDER:
        cat = sensitive[sensitive["pairCategory"] == category]
        if cat.empty:
            continue
        add(cat.sort_values(["s03SensitivityScore"], ascending=False, kind="mergesort").head(1), "max_s03_sensitivity")
        add(cat.sort_values(["finalSortednessPercent"], ascending=False, kind="mergesort").head(1), "high_s03_sortedness")
        add(cat.sort_values(["finalSortednessPercent"], ascending=True, kind="mergesort").head(1), "low_s03_sortedness_stress")
        chosen = [row for row in selected.values() if row["pairCategory"] == category]
        if len(chosen) < max_conditions_per_category:
            chosen_ids = {str(row["conditionId"]) for row in chosen}
            remaining = cat[~cat["conditionId"].astype(str).isin(chosen_ids)]
            add(
                remaining.sort_values(["s03SensitivityScore", "policyPositionChangeFraction"], ascending=False, kind="mergesort").head(
                    max_conditions_per_category - len(chosen)
                ),
                "category_fill",
            )

    # Ensure the original 50:50 condition is available for the opposite-direction baseline check.
    original_50 = sensitive[
        (sensitive["pairCategory"] == "original_pair")
        & (sensitive["ratioLabel"].astype(str) == "50:50")
    ].sort_values(["s03SensitivityScore"], ascending=False, kind="mergesort").head(1)
    add(original_50, "mandatory_original_50_50_opposite_baseline")

    selected_df = pd.DataFrame(selected.values())
    if selected_df.empty:
        raise ValueError("S04 selected no S03 base conditions")
    selected_df = selected_df.sort_values(
        ["categoryRank", "s03SensitivityScore", "finalSortednessPercent"],
        ascending=[True, False, False],
        kind="mergesort",
    ).head(int(max_total_conditions))
    selected_df = selected_df.reset_index(drop=True)
    selected_df.insert(0, "s04BaseConditionIndex", np.arange(len(selected_df), dtype=int))
    selected_df["s04Version"] = GOAL_SWEEP_VERSION
    return selected_df


def policy_goal_specs(policy_ids: Sequence[str], goal_mode: str) -> dict[str, dict[str, Any]]:
    ids = [str(pid) for pid in policy_ids]
    if len(ids) != 2:
        raise ValueError("S04 currently expects pairwise S03 conditions")
    left, right = ids
    if goal_mode == "same_goal":
        return {
            left: {"goalType": "strict_global_order", "targetDirection": "increasing", "behaviorDirection": "increasing", "role": "same_left"},
            right: {"goalType": "strict_global_order", "targetDirection": "increasing", "behaviorDirection": "increasing", "role": "same_right"},
        }
    if goal_mode == "opposite_goal":
        return {
            left: {"goalType": "strict_global_order", "targetDirection": "increasing", "behaviorDirection": "increasing", "role": "opposes_right"},
            right: {"goalType": "strict_global_order", "targetDirection": "decreasing", "behaviorDirection": "decreasing", "role": "opposes_left"},
        }
    if goal_mode == "partially_compatible":
        return {
            left: {"goalType": "strict_global_order", "targetDirection": "increasing", "behaviorDirection": "increasing", "role": "strict_component"},
            right: {"goalType": "coarse_global_order", "targetDirection": "increasing", "behaviorDirection": "increasing", "role": "coarse_component"},
        }
    if goal_mode == "shared_global":
        return {
            left: {"goalType": "shared_global_order", "targetDirection": "increasing", "behaviorDirection": "increasing", "role": "shared_contributor"},
            right: {"goalType": "shared_global_order", "targetDirection": "increasing", "behaviorDirection": "increasing", "role": "shared_contributor"},
        }
    if goal_mode == "unrelated_goal":
        return {
            left: {"goalType": "strict_global_order", "targetDirection": "increasing", "behaviorDirection": "increasing", "role": "ordering_component"},
            right: {"goalType": "policy_identity_aggregation", "targetDirection": "identity", "behaviorDirection": "increasing", "role": "aggregation_component"},
        }
    raise ValueError(f"unknown goal mode: {goal_mode}")


def condition_goal_metadata(policy_ids: Sequence[str], goal_mode: str) -> dict[str, Any]:
    definition = GOAL_MODE_DEFINITIONS[goal_mode]
    return {
        "goalMode": goal_mode,
        "goalModeLabel": definition["label"],
        "compatibilityClass": definition["compatibilityClass"],
        "conditionTarget": definition["conditionTarget"],
        "metricBoundary": definition["metricBoundary"],
        "policyGoalSpecs": policy_goal_specs(policy_ids, goal_mode),
        "claimBoundary": CLAIM_BOUNDARY,
    }


def reverse_directions_for_assignment(assigned_policy_ids: Sequence[str], goal_specs: Mapping[str, Mapping[str, Any]]) -> list[bool]:
    return [str(goal_specs[str(pid)]["behaviorDirection"]) == "decreasing" for pid in assigned_policy_ids]


def build_goal_condition_matrix(
    s03_results: pd.DataFrame,
    ready_panel: pd.DataFrame,
    *,
    max_conditions_per_category: int = 4,
    max_total_conditions: int = 24,
    goal_modes: Sequence[str] = GOAL_MODES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = select_s04_base_conditions(
        s03_results,
        max_conditions_per_category=max_conditions_per_category,
        max_total_conditions=max_total_conditions,
    )
    ready_ids = set(ready_panel["panelPolicyId"].astype(str))
    rows: list[dict[str, Any]] = []
    condition_metadata: list[dict[str, Any]] = []
    policy_metadata: list[dict[str, Any]] = []

    for base in selected.to_dict(orient="records"):
        policy_ids = [str(item) for item in parse_json_maybe(base["policyIdsJson"], [])]
        assigned = [str(item) for item in parse_json_maybe(base["arrangementPolicyIdsJson"], [])]
        numeric_labels = [int(item) for item in parse_json_maybe(base["arrangementNumericLabelsJson"], [])]
        if set(policy_ids) - ready_ids:
            raise ValueError("S04 base row includes policy not marked ready")
        if not assigned or len(assigned) != int(base["n"]):
            raise ValueError("S04 base row lacks explicit S03 arrangement assignment")
        for goal_index, goal_mode in enumerate(goal_modes):
            specs = policy_goal_specs(policy_ids, str(goal_mode))
            reverse = reverse_directions_for_assignment(assigned, specs)
            goal_meta = condition_goal_metadata(policy_ids, str(goal_mode))
            condition_id = (
                f"s04_b{int(base['s04BaseConditionIndex']):03d}_"
                f"{str(base['ratioLabel']).replace(':', '_')}_seed{int(base['seedIndex'])}_"
                f"{str(base['arrangementType'])}_{goal_mode}"
            )
            condition_payload = {
                "s03ConditionId": str(base["conditionId"]),
                "goalMode": str(goal_mode),
                "policyGoalSpecs": specs,
                "arrangementHash": str(base["arrangementHash"]),
                "s04Version": GOAL_SWEEP_VERSION,
            }
            row = {
                "conditionId": condition_id,
                "researchStepId": STEP_ID,
                "implementationPrefix": "e06_s04",
                "s04Version": GOAL_SWEEP_VERSION,
                "s03SourceConditionId": str(base["conditionId"]),
                "s03SourceConditionHash": str(base.get("conditionHash", "")),
                "s04BaseConditionIndex": int(base["s04BaseConditionIndex"]),
                "s03BaseMatchId": str(base["baseMatchId"]),
                "s04SelectionReason": str(base["s04SelectionReason"]),
                "mixtureKind": "pair",
                "pairCategory": str(base["pairCategory"]),
                "priorityReason": str(base.get("priorityReason", "")),
                "leftPanelPolicyId": str(base["leftPanelPolicyId"]),
                "rightPanelPolicyId": str(base["rightPanelPolicyId"]),
                "thirdPanelPolicyId": "",
                "ratioLabel": str(base["ratioLabel"]),
                "targetCountsJson": str(base["targetCountsJson"]),
                "targetProportionsJson": str(base["targetProportionsJson"]),
                "n": int(base["n"]),
                "targetPolicyCount": len(policy_ids),
                "policyIdsJson": compact_json(policy_ids),
                "policyCountsJson": str(base["policyCountsJson"]),
                "arrangementType": str(base["arrangementType"]),
                "arrangementTypeIndex": int(base.get("arrangementTypeIndex", 0)),
                "arrangementPolicyIdsJson": compact_json(assigned),
                "arrangementNumericLabelsJson": compact_json(numeric_labels),
                "arrangementDescriptorJson": str(base["arrangementDescriptorJson"]),
                "arrangementHash": str(base["arrangementHash"]),
                "goalMode": str(goal_mode),
                "goalModeIndex": int(goal_index),
                "goalCompatibilityClass": str(goal_meta["compatibilityClass"]),
                "conditionGoalMetadataJson": compact_json(goal_meta),
                "policyGoalMetadataJson": compact_json(specs),
                "goalReverseDirectionsJson": compact_json(reverse),
                "goalDirectionByPolicyJson": compact_json({pid: specs[pid]["targetDirection"] for pid in policy_ids}),
                "goalBehaviorDirectionByPolicyJson": compact_json({pid: specs[pid]["behaviorDirection"] for pid in policy_ids}),
                "conditionHash": sha256_text(compact_json(condition_payload))[:16],
                "seedIndex": int(base["seedIndex"]),
                "valueSeed": int(base["valueSeed"]),
                "arrangementSeed": int(base["arrangementSeed"]),
                "schedulerSeed": int(base["schedulerSeed"]),
                "tieBreakerSeed": int(base["tieBreakerSeed"]),
                "s03FinalSortednessPercent": float(base["finalSortednessPercent"]),
                "s03Completed": bool(base["completed"]),
                "s03FinalAggregation": float(base["finalAggregation"]),
                "s03InterfaceStabilityScore": float(base["interfaceStabilityScore"]),
                "s03SensitivityScore": float(base["s03SensitivityScore"]),
            }
            rows.append(row)
            condition_metadata.append(
                {
                    "conditionId": condition_id,
                    "s03SourceConditionId": str(base["conditionId"]),
                    "goalMode": str(goal_mode),
                    "goalCompatibilityClass": str(goal_meta["compatibilityClass"]),
                    "conditionGoalMetadataJson": compact_json(goal_meta),
                    "goalReverseDirectionsJson": compact_json(reverse),
                }
            )
            counts = Counter(assigned)
            for pid in policy_ids:
                spec = specs[pid]
                policy_metadata.append(
                    {
                        "conditionId": condition_id,
                        "panelPolicyId": pid,
                        "goalMode": str(goal_mode),
                        "goalType": str(spec["goalType"]),
                        "targetDirection": str(spec["targetDirection"]),
                        "behaviorDirection": str(spec["behaviorDirection"]),
                        "role": str(spec["role"]),
                        "cellCount": int(counts[pid]),
                        "policyGoalSpecJson": compact_json(spec),
                    }
                )

    return pd.DataFrame(rows), selected, pd.DataFrame(condition_metadata), pd.DataFrame(policy_metadata)


def _coarse_values(values: Sequence[int], bins: int = 4) -> list[int]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return []
    quantiles = np.quantile(arr, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return [int(np.searchsorted(quantiles, value, side="right")) for value in arr]


def _subsequence_score(values: Sequence[int], labels: Sequence[str], panel_policy_id: str, direction: str, *, coarse: bool = False) -> float:
    subset = [int(value) for value, label in zip(values, labels, strict=True) if str(label) == str(panel_policy_id)]
    if len(subset) < 2:
        return 100.0
    scored = _coarse_values(subset) if coarse else subset
    return float(sortedness_percent(scored, direction=direction))


def _policy_local_aggregation_score(labels: Sequence[str], panel_policy_id: str) -> float:
    touching = 0
    same = 0
    pid = str(panel_policy_id)
    for left, right in zip(labels, labels[1:]):
        if left == pid or right == pid:
            touching += 1
            same += int(left == right)
    if touching == 0:
        return 100.0
    return 100.0 * same / touching


def _policy_goal_scores(row: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, bool]]:
    final_values = [int(item) for item in parse_json_maybe(row.get("finalValuesJson", ""), [])]
    final_ids = [str(item) for item in parse_json_maybe(row.get("finalPanelPolicyIdsJson", ""), [])]
    policy_ids = [str(item) for item in parse_json_maybe(row.get("policyIdsJson", ""), [])]
    specs = parse_json_maybe(row.get("policyGoalMetadataJson", ""), {})
    increasing = float(sortedness_percent(final_values, direction="increasing")) if final_values else float("nan")
    decreasing = float(sortedness_percent(final_values, direction="decreasing")) if final_values else float("nan")
    scores: dict[str, float] = {}
    satisfied: dict[str, bool] = {}
    for pid in policy_ids:
        spec = specs.get(pid, {}) if isinstance(specs, Mapping) else {}
        goal_type = str(spec.get("goalType", "strict_global_order"))
        direction = str(spec.get("targetDirection", "increasing"))
        if goal_type == "strict_global_order":
            score = _subsequence_score(final_values, final_ids, pid, direction)
        elif goal_type == "coarse_global_order":
            score = _subsequence_score(final_values, final_ids, pid, "increasing", coarse=True)
        elif goal_type == "shared_global_order":
            score = increasing
        elif goal_type == "policy_identity_aggregation":
            score = _policy_local_aggregation_score(final_ids, pid)
        else:
            score = float("nan")
        scores[pid] = float(score)
        satisfied[pid] = bool(np.isfinite(score) and score >= 95.0)
    # Directional scores are included under reserved pseudo-keys for transparent downstream use.
    scores["__global_increasing__"] = increasing
    scores["__global_decreasing__"] = decreasing
    return scores, satisfied


def _dominant_direction(increasing: float, decreasing: float) -> str:
    if not np.isfinite(increasing) or not np.isfinite(decreasing):
        return "unknown"
    if abs(increasing - decreasing) < 5.0:
        return "balanced"
    return "increasing" if increasing > decreasing else "decreasing"


def _compromise_class(row: Mapping[str, Any], scores: Mapping[str, float]) -> str:
    policy_scores = [value for key, value in scores.items() if not str(key).startswith("__") and np.isfinite(value)]
    if not policy_scores:
        return "unscored"
    mean_score = float(np.mean(policy_scores))
    spread = float(max(policy_scores) - min(policy_scores)) if len(policy_scores) > 1 else 0.0
    increasing = float(scores.get("__global_increasing__", np.nan))
    decreasing = float(scores.get("__global_decreasing__", np.nan))
    if mean_score >= 95.0 and spread <= 10.0:
        return "mutual_goal_attainment_proxy"
    if mean_score >= 75.0 and spread <= 20.0:
        return "partial_compromise_proxy"
    if str(row.get("goalMode")) == "opposite_goal" and _dominant_direction(increasing, decreasing) != "balanced":
        return f"directional_dominance_proxy_{_dominant_direction(increasing, decreasing)}"
    if spread > 35.0:
        return "asymmetric_goal_attainment_proxy"
    return "low_joint_attainment_proxy"


def augment_goal_results(result_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in result_df.to_dict(orient="records"):
        initial_values = [int(item) for item in parse_json_maybe(row.get("initialValuesJson", ""), [])]
        final_values = [int(item) for item in parse_json_maybe(row.get("finalValuesJson", ""), [])]
        initial_ids = [str(item) for item in parse_json_maybe(row.get("initialPanelPolicyIdsJson", ""), [])]
        final_ids = [str(item) for item in parse_json_maybe(row.get("finalPanelPolicyIdsJson", ""), [])]
        policy_ids = [str(item) for item in parse_json_maybe(row.get("policyIdsJson", ""), [])]
        scores, satisfied = _policy_goal_scores(row)
        policy_scores = [value for key, value in scores.items() if not str(key).startswith("__") and np.isfinite(value)]
        initial_interfaces = interface_count(initial_ids)
        final_interfaces = interface_count(final_ids)
        initial_means = mean_positions(initial_ids, policy_ids)
        final_means = mean_positions(final_ids, policy_ids)
        row.update(
            {
                "initialIncreasingSortednessPercent": float(sortedness_percent(initial_values, direction="increasing")) if initial_values else float("nan"),
                "finalIncreasingSortednessPercent": float(scores.get("__global_increasing__", np.nan)),
                "initialDecreasingSortednessPercent": float(sortedness_percent(initial_values, direction="decreasing")) if initial_values else float("nan"),
                "finalDecreasingSortednessPercent": float(scores.get("__global_decreasing__", np.nan)),
                "dominantDirectionProxy": _dominant_direction(float(scores.get("__global_increasing__", np.nan)), float(scores.get("__global_decreasing__", np.nan))),
                "policyGoalScoresJson": compact_json(scores),
                "policyGoalSatisfiedJson": compact_json(satisfied),
                "meanPolicyGoalScore": float(np.mean(policy_scores)) if policy_scores else float("nan"),
                "minPolicyGoalScore": float(np.min(policy_scores)) if policy_scores else float("nan"),
                "goalScoreSpread": float(np.max(policy_scores) - np.min(policy_scores)) if len(policy_scores) > 1 else 0.0,
                "jointGoalSatisfied": bool(policy_scores and min(policy_scores) >= 95.0),
                "compromiseStateClass": _compromise_class(row, scores),
                "initialInterfaceCount": int(initial_interfaces),
                "finalInterfaceCount": int(final_interfaces),
                "interfaceCountDelta": int(final_interfaces - initial_interfaces),
                "finalInterfaceDensity": float(final_interfaces / max(1, len(final_ids) - 1)) if final_ids else float("nan"),
                "finalMosaicClass": classify_mosaic(final_ids, bool(row.get("completed", False))) if final_ids else "run_failed",
                "initialPolicyMeanPositionsJson": compact_json(initial_means),
                "finalPolicyMeanPositionsJson": compact_json(final_means),
                "claimBoundary": CLAIM_BOUNDARY,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_goal_results(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode_summary = (
        result_df.groupby(["goalMode", "goalCompatibilityClass"], dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            runSuccessRate=("runSucceeded", "mean"),
            increasingCompletionRate=("completed", "mean"),
            jointGoalSatisfiedRate=("jointGoalSatisfied", "mean"),
            meanPolicyGoalScore=("meanPolicyGoalScore", "mean"),
            meanMinPolicyGoalScore=("minPolicyGoalScore", "mean"),
            meanGoalScoreSpread=("goalScoreSpread", "mean"),
            meanFinalIncreasingSortednessPercent=("finalIncreasingSortednessPercent", "mean"),
            meanFinalDecreasingSortednessPercent=("finalDecreasingSortednessPercent", "mean"),
            meanFinalAggregation=("finalAggregation", "mean"),
            meanFinalInterfaceCount=("finalInterfaceCount", "mean"),
            meanActivationCount=("activationCount", "mean"),
            meanRuntimeSeconds=("runtimeSeconds", "mean"),
        )
        .reset_index()
    )
    category_summary = (
        result_df.groupby(["pairCategory", "goalMode"], dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            jointGoalSatisfiedRate=("jointGoalSatisfied", "mean"),
            meanPolicyGoalScore=("meanPolicyGoalScore", "mean"),
            meanGoalScoreSpread=("goalScoreSpread", "mean"),
            meanFinalIncreasingSortednessPercent=("finalIncreasingSortednessPercent", "mean"),
            meanFinalDecreasingSortednessPercent=("finalDecreasingSortednessPercent", "mean"),
            meanFinalAggregation=("finalAggregation", "mean"),
        )
        .reset_index()
    )
    same = result_df[result_df["goalMode"] == "same_goal"][
        ["s03SourceConditionId", "meanPolicyGoalScore", "minPolicyGoalScore", "finalIncreasingSortednessPercent"]
    ].rename(
        columns={
            "meanPolicyGoalScore": "sameGoalMeanPolicyGoalScore",
            "minPolicyGoalScore": "sameGoalMinPolicyGoalScore",
            "finalIncreasingSortednessPercent": "sameGoalFinalIncreasingSortednessPercent",
        }
    )
    deltas = result_df.merge(same, on="s03SourceConditionId", how="left")
    deltas["deltaVsSameMeanPolicyGoalScore"] = deltas["meanPolicyGoalScore"] - deltas["sameGoalMeanPolicyGoalScore"]
    deltas["deltaVsSameMinPolicyGoalScore"] = deltas["minPolicyGoalScore"] - deltas["sameGoalMinPolicyGoalScore"]
    deltas["deltaVsSameFinalIncreasingSortednessPercent"] = (
        deltas["finalIncreasingSortednessPercent"] - deltas["sameGoalFinalIncreasingSortednessPercent"]
    )
    return mode_summary, category_summary, deltas


def validation_checks(
    condition_df: pd.DataFrame,
    result_df: pd.DataFrame,
    ready_panel: pd.DataFrame,
    condition_goal_df: pd.DataFrame,
    policy_goal_df: pd.DataFrame,
    smoke_df: pd.DataFrame,
    *,
    goal_modes: Sequence[str] = GOAL_MODES,
) -> pd.DataFrame:
    ready_ids = set(ready_panel["panelPolicyId"].astype(str))
    condition_policy_ids = {
        pid for text in condition_df["policyIdsJson"].astype(str) for pid in parse_json_maybe(text, [])
    }
    observed_by_base = condition_df.groupby("s03SourceConditionId")["goalMode"].agg(lambda values: set(values))
    required_modes = set(goal_modes)
    reverse_ok = True
    for row in condition_df.to_dict(orient="records"):
        assigned = [str(item) for item in parse_json_maybe(row["arrangementPolicyIdsJson"], [])]
        reverse = [bool(item) for item in parse_json_maybe(row["goalReverseDirectionsJson"], [])]
        specs = parse_json_maybe(row["policyGoalMetadataJson"], {})
        if len(reverse) != int(row["n"]):
            reverse_ok = False
            break
        expected = reverse_directions_for_assignment(assigned, specs)
        if reverse != expected:
            reverse_ok = False
            break
    goal_label_modes = set(condition_goal_df["goalMode"].astype(str)) if not condition_goal_df.empty else set()
    policy_rows_expected = int(sum(len(parse_json_maybe(text, [])) for text in condition_df["policyIdsJson"].astype(str)))
    original_opposite = result_df[
        (result_df["pairCategory"] == "original_pair")
        & (result_df["goalMode"] == "opposite_goal")
    ].copy()
    original_opposite_ok = bool(
        not original_opposite.empty
        and original_opposite["runSucceeded"].map(bool).all()
        and original_opposite["finalIncreasingSortednessPercent"].notna().all()
        and original_opposite["finalDecreasingSortednessPercent"].notna().all()
        and original_opposite["goalDirectionByPolicyJson"].astype(str).str.contains("decreasing").all()
        and original_opposite["goalDirectionByPolicyJson"].astype(str).str.contains("increasing").all()
    )
    checks = [
        {
            "checkId": "only_ready_policies",
            "success": bool(condition_policy_ids <= ready_ids),
            "detail": f"{len(condition_policy_ids)} S04 policies all came from readyForS02Mixing=true rows",
        },
        {
            "checkId": "all_goal_modes_present_per_s03_template",
            "success": bool(not observed_by_base.empty and observed_by_base.map(lambda values: required_modes <= values).all()),
            "detail": f"each selected S03 source condition includes {len(required_modes)} requested goal modes",
        },
        {
            "checkId": "condition_goal_metadata_complete",
            "success": bool(len(condition_goal_df) == len(condition_df) and required_modes <= goal_label_modes),
            "detail": f"{len(condition_goal_df)} condition goal metadata rows written",
        },
        {
            "checkId": "policy_goal_metadata_complete",
            "success": bool(len(policy_goal_df) == policy_rows_expected and {"goalType", "targetDirection", "behaviorDirection"} <= set(policy_goal_df.columns)),
            "detail": f"{len(policy_goal_df)} policy goal metadata rows for {len(condition_df)} conditions",
        },
        {
            "checkId": "reverse_directions_match_goal_labels",
            "success": bool(reverse_ok),
            "detail": "per-cell reverse direction vectors match behaviorDirection in policy goal metadata",
        },
        {
            "checkId": "opposite_direction_baseline_validated",
            "success": original_opposite_ok,
            "detail": f"{len(original_opposite)} original-pair opposite-goal rows carry increasing and decreasing labels with finite metrics",
        },
        {
            "checkId": "one_result_per_condition",
            "success": bool(len(result_df) == len(condition_df) and result_df["conditionId"].is_unique),
            "detail": f"{len(result_df)} result rows for {len(condition_df)} conditions",
        },
        {
            "checkId": "runs_succeeded_and_counts_conserved",
            "success": bool(
                result_df["runSucceeded"].fillna(False).map(bool).all()
                and result_df["valueCountsConserved"].fillna(False).map(bool).all()
                and result_df["policyCountsConserved"].fillna(False).map(bool).all()
            ),
            "detail": "all S04 runs succeeded and conserved values/policy labels",
        },
        {
            "checkId": "goal_scores_populated",
            "success": bool(result_df["meanPolicyGoalScore"].notna().all() and result_df["policyGoalScoresJson"].astype(str).str.len().gt(2).all()),
            "detail": "per-policy goal score JSON and aggregate goal scores are populated",
        },
        {
            "checkId": "smoke_replay_deterministic",
            "success": bool(not smoke_df.empty and smoke_df["replayMatch"].map(bool).all()),
            "detail": f"{len(smoke_df)} smoke replay checks matched",
        },
    ]
    return pd.DataFrame(checks)


def run_smoke_replays(
    condition_df: pd.DataFrame,
    ready_panel: pd.DataFrame,
    *,
    sample_size: int = 6,
    max_activations: int = 2000,
) -> pd.DataFrame:
    sample = condition_df.head(int(sample_size))
    lookup = {str(row["panelPolicyId"]): dict(row) for row in ready_panel.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for condition in sample.to_dict(orient="records"):
        first = run_mixture_condition(condition, lookup, max_activations=max_activations, max_swaps=1000, max_comparisons=8000)
        second = run_mixture_condition(condition, lookup, max_activations=max_activations, max_swaps=1000, max_comparisons=8000)
        keys = ["runSucceeded", "finalStateHash", "finalPanelPolicyIdsJson", "swapCount", "activationCount", "stopReason"]
        match = all(first.get(key) == second.get(key) for key in keys)
        rows.append(
            {
                "conditionId": condition["conditionId"],
                "goalMode": condition["goalMode"],
                "replayMatch": bool(match),
                "firstFinalStateHash": first.get("finalStateHash"),
                "secondFinalStateHash": second.get("finalStateHash"),
                "firstStopReason": first.get("stopReason"),
                "secondStopReason": second.get("stopReason"),
                "checkedKeysJson": compact_json(keys),
            }
        )
    return pd.DataFrame(rows)


def write_goal_plot(mode_summary: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    if mode_summary.empty:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_df = mode_summary.copy()
    plot_df["goalMode"] = pd.Categorical(plot_df["goalMode"], categories=list(GOAL_MODES), ordered=True)
    plot_df = plot_df.sort_values("goalMode", kind="mergesort")
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, plot_df["meanPolicyGoalScore"], width=0.36, label="Mean policy-goal score")
    ax.bar(x + 0.18, plot_df["meanMinPolicyGoalScore"], width=0.36, label="Mean min policy-goal score")
    ax.set_xticks(x, plot_df["goalMode"].astype(str), rotation=25, ha="right")
    ax.set_ylabel("Goal score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("E06 S04 goal-compatibility proxy scores")
    ax.legend()
    fig.tight_layout()
    paths = [
        figure_dir / "e06_s04_goal_compatibility_scores.png",
        figure_dir / "e06_s04_goal_compatibility_scores.pdf",
        step_dir / "goal_compatibility_scores.png",
        step_dir / "goal_compatibility_scores.pdf",
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
        f"- `{row.goalMode}`: {int(row.conditionCount)} conditions, joint-goal satisfaction {row.jointGoalSatisfiedRate:.2f}, "
        f"mean policy-goal score {row.meanPolicyGoalScore:.1f}%, mean score spread {row.meanGoalScoreSpread:.1f}"
        for row in mode_summary.sort_values("goalMode").itertuples(index=False)
    ) or "- No goal-mode summaries available."
    top = mode_summary.sort_values(["meanPolicyGoalScore", "jointGoalSatisfiedRate"], ascending=[False, False]).head(3)
    top_lines = "\n".join(
        f"- `{row.goalMode}` retained mean policy-goal score {row.meanPolicyGoalScore:.1f}% with spread {row.meanGoalScoreSpread:.1f}"
        for row in top.itertuples(index=False)
    ) or "- No anchor goal-mode results available."
    text = f"""# Research Step S04: Vary goal compatibility

## Completion status

{status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Ran {status['conditionCount']} goal-compatibility conditions from {status['selectedS03ConditionCount']} S03-sensitive templates. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

S04 uses selected S03-sensitive pair/ratio/arrangement templates rather than the full S03 matrix. Opposite goals alter per-cell reverse-direction behavior where the simulator supports it. Partially compatible, shared-global, and unrelated goals are explicitly labeled computational proxy objectives; the unrelated identity-aggregation target is metric-level and does not introduce a new aggregation-seeking controller.

## Lay summary

S04 asked whether the same mixed collectives look cooperative, conflicting, partially compatible, globally shared, or unrelated when their goals are labeled and scored differently. It records both behavior-level direction choices and evaluation-level proxy targets so S05 can formalize compatibility metrics.

## Goal-mode summary

{mode_lines}

## Anchor results

{top_lines}

## Recommended next action

Chief Scientist review, then S05 should define the compatibility vector using S04 goal labels, per-policy goal scores, dominance proxies, and the caveats on metric-only goals.
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s04_goal_compatibility(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    panel_path: Path = DEFAULT_PANEL_PATH,
    s03_results_path: Path = DEFAULT_S03_RESULTS_PATH,
    workers: int | None = None,
    max_conditions_per_category: int = 4,
    max_total_conditions: int = 24,
    max_activations: int = 6000,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
) -> dict[str, Any]:
    artifacts_dir = artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    workers = int(workers if workers is not None else min(8, os.cpu_count() or 1))
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e06"
    provenance_dir = artifacts_dir / "provenance"
    for path in [step_dir, results_dir, figures_dir, provenance_dir]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ready_panel = load_ready_panel(panel_path)
    s03_results = load_s03_results(s03_results_path)
    condition_df, selected_df, condition_goal_df, policy_goal_df = build_goal_condition_matrix(
        s03_results,
        ready_panel,
        max_conditions_per_category=max_conditions_per_category,
        max_total_conditions=max_total_conditions,
    )
    smoke_df = run_smoke_replays(condition_df, ready_panel)
    raw_result_df = run_conditions(
        condition_df,
        ready_panel,
        workers=workers,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )
    result_df = augment_goal_results(raw_result_df)
    mode_summary, category_summary, matched_deltas = summarize_goal_results(result_df)
    validation_df = validation_checks(condition_df, result_df, ready_panel, condition_goal_df, policy_goal_df, smoke_df)

    condition_csv = step_dir / "goal_condition_matrix.csv"
    condition_parquet = step_dir / "goal_condition_matrix.parquet"
    selected_csv = step_dir / "selected_s03_sensitive_conditions.csv"
    selected_parquet = step_dir / "selected_s03_sensitive_conditions.parquet"
    condition_goal_csv = step_dir / "condition_goal_metadata.csv"
    condition_goal_parquet = step_dir / "condition_goal_metadata.parquet"
    policy_goal_csv = step_dir / "policy_goal_metadata.csv"
    policy_goal_parquet = step_dir / "policy_goal_metadata.parquet"
    step_result_csv = step_dir / "goal_compatibility_runs.csv"
    step_result_parquet = step_dir / "goal_compatibility_runs.parquet"
    result_csv = results_dir / "e06_goal_compatibility.csv"
    result_parquet = results_dir / "e06_goal_compatibility.parquet"
    sweep_csv = results_dir / "e06_mixture_sweeps.csv"
    sweep_parquet = results_dir / "e06_mixture_sweeps.parquet"
    mode_summary_csv = step_dir / "goal_mode_summary.csv"
    mode_summary_parquet = step_dir / "goal_mode_summary.parquet"
    category_summary_csv = step_dir / "goal_category_summary.csv"
    category_summary_parquet = step_dir / "goal_category_summary.parquet"
    deltas_csv = step_dir / "matched_same_goal_deltas.csv"
    deltas_parquet = step_dir / "matched_same_goal_deltas.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    smoke_csv = step_dir / "smoke_replay_checks.csv"
    smoke_parquet = step_dir / "smoke_replay_checks.parquet"

    result_csv_df = compact_result_csv(result_df)
    for path, df in [
        (condition_csv, condition_df),
        (selected_csv, selected_df),
        (condition_goal_csv, condition_goal_df),
        (policy_goal_csv, policy_goal_df),
        (step_result_csv, result_csv_df),
        (result_csv, result_csv_df),
        (mode_summary_csv, mode_summary),
        (category_summary_csv, category_summary),
        (deltas_csv, compact_result_csv(matched_deltas)),
        (validation_csv, validation_df),
        (smoke_csv, smoke_df),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (condition_parquet, condition_df),
        (selected_parquet, selected_df),
        (condition_goal_parquet, condition_goal_df),
        (policy_goal_parquet, policy_goal_df),
        (step_result_parquet, result_df),
        (result_parquet, result_df),
        (mode_summary_parquet, mode_summary),
        (category_summary_parquet, category_summary),
        (deltas_parquet, matched_deltas),
        (validation_parquet, validation_df),
        (smoke_parquet, smoke_df),
    ]:
        df.to_parquet(path, index=False)

    previous_sweeps = pd.read_parquet(sweep_parquet) if sweep_parquet.exists() else pd.DataFrame()
    if not previous_sweeps.empty and "researchStepId" in previous_sweeps.columns:
        previous_sweeps = previous_sweeps[previous_sweeps["researchStepId"] != STEP_ID].copy()
    cumulative = pd.concat([previous_sweeps, result_df], ignore_index=True, sort=False)
    compact_result_csv(cumulative).to_csv(sweep_csv, index=False)
    cumulative.to_parquet(sweep_parquet, index=False)

    figure_paths = write_goal_plot(mode_summary, figures_dir, step_dir)
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() and result_df["runSucceeded"].fillna(False).map(bool).all() else "failed"
    artifacts = [
        condition_csv,
        condition_parquet,
        selected_csv,
        selected_parquet,
        condition_goal_csv,
        condition_goal_parquet,
        policy_goal_csv,
        policy_goal_parquet,
        step_result_csv,
        step_result_parquet,
        result_csv,
        result_parquet,
        sweep_csv,
        sweep_parquet,
        mode_summary_csv,
        mode_summary_parquet,
        category_summary_csv,
        category_summary_parquet,
        deltas_csv,
        deltas_parquet,
        validation_csv,
        validation_parquet,
        smoke_csv,
        smoke_parquet,
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
            "S04 used selected S03-sensitive templates rather than the full S03 matrix.",
            "Opposite goals use per-cell reverse-direction behavior, but the simulator stop condition remains increasing-order based; goal-specific metrics are reported separately.",
            "Partially compatible, shared-global, and unrelated goals are explicitly labeled computational proxy objectives.",
            "The unrelated identity-aggregation goal is metric-level and does not add an aggregation-seeking controller.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then run S05 compatibility-metric design using S04 goal labels, per-policy "
            "goal scores, dominance proxies, and metric-boundary caveats."
        ),
        "laySummary": (
            "S04 varied goal labels and per-cell goal directions over S03-sensitive mixed collectives, then scored "
            "same, opposite, partially compatible, shared-global, and unrelated proxy objectives."
        ),
        "outcomeClassification": "supportive" if validation_result == "passed" else "constraining/contradictory",
        "conditionCount": int(len(condition_df)),
        "resultCount": int(len(result_df)),
        "selectedS03ConditionCount": int(selected_df.shape[0]),
        "selectedPolicyCount": int(len({pid for text in condition_df["policyIdsJson"].astype(str) for pid in parse_json_maybe(text, [])})),
        "goalModes": list(GOAL_MODES),
        "meanPolicyGoalScore": float(result_df["meanPolicyGoalScore"].mean()),
        "jointGoalSatisfiedRate": float(result_df["jointGoalSatisfied"].mean()),
        "workerCount": int(workers),
        "maxActivations": int(max_activations),
        "maxSwaps": int(max_swaps),
        "maxComparisons": int(max_comparisons),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
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
        "goalSweepVersion": GOAL_SWEEP_VERSION,
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
            "workerCount": int(workers),
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
        "conditions": condition_df,
        "selected": selected_df,
        "conditionGoalMetadata": condition_goal_df,
        "policyGoalMetadata": policy_goal_df,
        "results": result_df,
        "modeSummary": mode_summary,
        "categorySummary": category_summary,
        "matchedDeltas": matched_deltas,
        "validation": validation_df,
        "smoke": smoke_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E06 S04 goal-compatibility sweeps.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--s03-results-path", type=Path, default=DEFAULT_S03_RESULTS_PATH)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-conditions-per-category", type=int, default=4)
    parser.add_argument("--max-total-conditions", type=int, default=24)
    parser.add_argument("--max-activations", type=int, default=6000)
    parser.add_argument("--max-swaps", type=int, default=4000)
    parser.add_argument("--max-comparisons", type=int, default=30000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_s04_goal_compatibility(
        artifacts_dir=args.artifacts_dir,
        panel_path=args.panel_path,
        s03_results_path=args.s03_results_path,
        workers=args.workers,
        max_conditions_per_category=args.max_conditions_per_category,
        max_total_conditions=args.max_total_conditions,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['conditionCount']} conditions, "
        f"{status['selectedS03ConditionCount']} S03 templates, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
