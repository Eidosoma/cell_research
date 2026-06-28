"""E06 S06 opposite-goal dominance hierarchy sweeps."""

from __future__ import annotations

import argparse
import json
import math
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

from .arrangements import generate_arrangement
from .goals import (
    augment_goal_results,
    condition_goal_metadata,
    policy_goal_specs,
    reverse_directions_for_assignment,
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
from .panel import CLAIM_BOUNDARY


STEP_ID = "S06"
STEP_NUMBER = 6
DOMINANCE_SWEEP_VERSION = "e06_s06_dominance_hierarchy.v1"
WINNER_CRITERIA_VERSION = "e06_s06_winner_criteria.v1"
DEFAULT_S05_METRICS_PATH = Path("/artifacts/results/e06_compatibility_metrics.parquet")
DEFAULT_S05_BOUNDARY_CAVEATS_PATH = Path("/artifacts/research_steps/S05/metric_boundary_caveats.parquet")
DEFAULT_E01_OPPOSITE_SUMMARY_PATH = Path("/previous-artifacts/E01/results/e01_opposite_direction_chimera_summary.parquet")
DOMINANCE_RATIOS: tuple[tuple[int, int], ...] = ((25, 75), (50, 50), (75, 25))
DOMINANCE_ARRANGEMENTS: tuple[str, ...] = ("random", "contiguous_patch", "alternating", "graft_like")
DOMINANCE_VALUE_PROFILES: tuple[str, ...] = ("random_unique", "reversed_unique", "duplicate_1_10_x10")
PAPER_ORIGINAL_ORDER: dict[frozenset[str], str] = {
    frozenset(("bubble", "insertion")): "bubble",
    frozenset(("bubble", "selection")): "bubble",
    frozenset(("selection", "insertion")): "selection",
}
UNRESOLVED_WINNER = "balanced_or_unresolved"


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return float(number) if np.isfinite(number) else default


def clip01(value: Any) -> float:
    number = safe_float(value)
    if not np.isfinite(number):
        return float("nan")
    return float(min(1.0, max(0.0, number)))


def load_s05_metrics(path: Path = DEFAULT_S05_METRICS_PATH) -> pd.DataFrame:
    metrics = pd.read_parquet(path)
    required = {
        "conditionId",
        "goalMode",
        "pairCategory",
        "leftPanelPolicyId",
        "rightPanelPolicyId",
        "policyIdsJson",
        "ratioLabel",
        "arrangementType",
        "runSucceeded",
        "dominanceProxyScore",
        "finalTargetQualityScore",
        "minTargetQualityScore",
        "dominantPolicyId",
        "metricBoundary",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"S05 compatibility metrics missing required columns: {missing}")
    usable = metrics[
        (metrics["goalMode"].astype(str) == "opposite_goal")
        & metrics["runSucceeded"].fillna(False).map(bool)
    ].copy()
    if usable.empty:
        raise ValueError("S05 compatibility metrics contain no usable opposite-goal rows")
    return usable.reset_index(drop=True)


def load_boundary_caveats(path: Path = DEFAULT_S05_BOUNDARY_CAVEATS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_e01_paper_anchors(path: Path = DEFAULT_E01_OPPOSITE_SUMMARY_PATH) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def canonical_pair_key(left: str, right: str) -> str:
    return "||".join(sorted([str(left), str(right)]))


def _policy_class_label(row: Mapping[str, Any]) -> str:
    raw_label = row.get("classLabel", "")
    label = "" if raw_label is None else str(raw_label)
    if label and label.lower() not in {"none", "nan"}:
        return label
    pid = str(row.get("panelPolicyId", ""))
    for algotype in ["bubble", "insertion", "selection"]:
        if algotype in pid:
            return f"classic_{algotype}"
    return ""


def _policy_algotype(label: str) -> str:
    text = str(label)
    if text.startswith("classic_"):
        return text.replace("classic_", "", 1)
    return text


def _panel_lookup(ready_panel: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["panelPolicyId"]): dict(row) for row in ready_panel.to_dict(orient="records")}


def _class_by_policy(ready_panel: pd.DataFrame) -> dict[str, str]:
    return {pid: _policy_class_label(row) for pid, row in _panel_lookup(ready_panel).items()}


def _group_by_policy(ready_panel: pd.DataFrame) -> dict[str, str]:
    return {pid: str(row.get("panelGroup", "")) for pid, row in _panel_lookup(ready_panel).items()}


def original_policy_ids(ready_panel: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in ready_panel.to_dict(orient="records"):
        if str(row.get("panelGroup")) != "original_classic":
            continue
        algotype = _policy_algotype(_policy_class_label(row))
        if algotype:
            out[algotype] = str(row["panelPolicyId"])
    return out


def paper_expected_winner(left: str, right: str, class_by_policy: Mapping[str, str]) -> tuple[str, str]:
    left_alg = _policy_algotype(class_by_policy.get(str(left), ""))
    right_alg = _policy_algotype(class_by_policy.get(str(right), ""))
    expected_alg = PAPER_ORIGINAL_ORDER.get(frozenset((left_alg, right_alg)), "")
    if not expected_alg:
        return "", ""
    for policy_id, label in class_by_policy.items():
        if _policy_algotype(label) == expected_alg and policy_id in {str(left), str(right)}:
            return policy_id, expected_alg
    return "", expected_alg


def _s05_pair_stats(s05_metrics: pd.DataFrame, ready_panel: pd.DataFrame) -> pd.DataFrame:
    class_by_id = _class_by_policy(ready_panel)
    group_by_id = _group_by_policy(ready_panel)
    df = s05_metrics.copy()
    df["policyAId"] = df.apply(lambda row: min(str(row["leftPanelPolicyId"]), str(row["rightPanelPolicyId"])), axis=1)
    df["policyBId"] = df.apply(lambda row: max(str(row["leftPanelPolicyId"]), str(row["rightPanelPolicyId"])), axis=1)
    df["s06PairKey"] = df.apply(lambda row: canonical_pair_key(row["policyAId"], row["policyBId"]), axis=1)
    rows: list[dict[str, Any]] = []
    for pair_key, group in df.groupby("s06PairKey", sort=False):
        policy_a = str(group["policyAId"].iloc[0])
        policy_b = str(group["policyBId"].iloc[0])
        winners = Counter(str(value) for value in group["dominantPolicyId"].fillna("").astype(str) if value)
        modal_winner, modal_count = ("", 0)
        if winners:
            modal_winner, modal_count = winners.most_common(1)[0]
        categories = Counter(str(value) for value in group["pairCategory"].astype(str))
        category = categories.most_common(1)[0][0] if categories else ""
        expected_policy, expected_alg = paper_expected_winner(policy_a, policy_b, class_by_id)
        if group_by_id.get(policy_a, "") != "original_classic" or group_by_id.get(policy_b, "") != "original_classic":
            expected_policy, expected_alg = "", ""
        rows.append(
            {
                "s06PairKey": pair_key,
                "policyAPanelPolicyId": policy_a,
                "policyBPanelPolicyId": policy_b,
                "pairCategory": category,
                "s05RowCount": int(len(group)),
                "meanS05DominanceProxyScore": float(group["dominanceProxyScore"].mean()),
                "maxS05DominanceProxyScore": float(group["dominanceProxyScore"].max()),
                "meanS05FinalTargetQualityScore": float(group["finalTargetQualityScore"].mean()),
                "meanS05MinTargetQualityScore": float(group["minTargetQualityScore"].mean()),
                "modalS05DominantPolicyId": modal_winner,
                "modalS05DominantShare": float(modal_count / max(1, len(group))),
                "policyAClassLabel": class_by_id.get(policy_a, ""),
                "policyBClassLabel": class_by_id.get(policy_b, ""),
                "policyAGroup": group_by_id.get(policy_a, ""),
                "policyBGroup": group_by_id.get(policy_b, ""),
                "paperExpectedWinnerPolicyId": expected_policy,
                "paperExpectedWinnerAlgotype": expected_alg,
                "s05SourceConditionIdsJson": compact_json(sorted(group["conditionId"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def _append_selected_pair(
    selected: dict[str, dict[str, Any]],
    candidate: Mapping[str, Any],
    reason: str,
) -> None:
    key = str(candidate["s06PairKey"])
    row = dict(candidate)
    existing = selected.get(key)
    if existing is None:
        row["s06SelectionReason"] = reason
        selected[key] = row
        return
    reasons = set(str(existing.get("s06SelectionReason", "")).split("+"))
    if reason not in reasons:
        existing["s06SelectionReason"] = f"{existing['s06SelectionReason']}+{reason}"


def select_s06_pairs(
    s05_metrics: pd.DataFrame,
    ready_panel: pd.DataFrame,
    *,
    max_pairs: int = 8,
) -> pd.DataFrame:
    stats = _s05_pair_stats(s05_metrics, ready_panel)
    if stats.empty:
        raise ValueError("S06 pair selection found no S05 pair statistics")
    class_by_id = _class_by_policy(ready_panel)
    group_by_id = _group_by_policy(ready_panel)
    selected: dict[str, dict[str, Any]] = {}
    originals = original_policy_ids(ready_panel)

    def original_candidate(left: str, right: str) -> dict[str, Any]:
        policy_a, policy_b = sorted([str(left), str(right)])
        expected_policy, expected_alg = paper_expected_winner(policy_a, policy_b, class_by_id)
        return {
            "s06PairKey": canonical_pair_key(policy_a, policy_b),
            "policyAPanelPolicyId": policy_a,
            "policyBPanelPolicyId": policy_b,
            "pairCategory": "original_pair",
            "s05RowCount": 0,
            "meanS05DominanceProxyScore": 0.0,
            "maxS05DominanceProxyScore": 0.0,
            "meanS05FinalTargetQualityScore": 0.0,
            "meanS05MinTargetQualityScore": 0.0,
            "modalS05DominantPolicyId": "",
            "modalS05DominantShare": 0.0,
            "policyAClassLabel": class_by_id.get(policy_a, ""),
            "policyBClassLabel": class_by_id.get(policy_b, ""),
            "policyAGroup": group_by_id.get(policy_a, ""),
            "policyBGroup": group_by_id.get(policy_b, ""),
            "paperExpectedWinnerPolicyId": expected_policy,
            "paperExpectedWinnerAlgotype": expected_alg,
            "s05SourceConditionIdsJson": compact_json([]),
        }

    for left_alg, right_alg in [("bubble", "insertion"), ("bubble", "selection"), ("selection", "insertion")]:
        if left_alg not in originals or right_alg not in originals:
            continue
        key = canonical_pair_key(originals[left_alg], originals[right_alg])
        hits = stats[stats["s06PairKey"] == key]
        candidate = original_candidate(originals[left_alg], originals[right_alg]) if hits.empty else hits.iloc[0].to_dict()
        _append_selected_pair(selected, candidate, "paper_original_order_anchor")

    priority_slices = [
        ("handoff_vs_frontier", 2, "high_s05_handoff_frontier_dominance"),
        ("frontier_pair", 2, "high_s05_frontier_pair_dominance"),
        ("memory_signal_handoff_vs_original", 2, "memory_signal_handoff_original_dominance"),
        ("e03_vs_original", 1, "frontier_or_discovered_original_dominance"),
    ]
    ranked = stats.sort_values(
        ["maxS05DominanceProxyScore", "meanS05FinalTargetQualityScore", "modalS05DominantShare"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    for category, limit, reason in priority_slices:
        for row in ranked[ranked["pairCategory"].astype(str) == category].head(limit).to_dict(orient="records"):
            if len(selected) >= int(max_pairs):
                break
            _append_selected_pair(selected, row, reason)
    for row in ranked.to_dict(orient="records"):
        if len(selected) >= int(max_pairs):
            break
        _append_selected_pair(selected, row, "fill_by_s05_dominance_proxy")

    selected_df = pd.DataFrame(selected.values())
    selected_df = selected_df.sort_values(
        ["pairCategory", "maxS05DominanceProxyScore", "s06PairKey"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_df.insert(0, "s06PairIndex", np.arange(len(selected_df), dtype=int))
    selected_df["s06Version"] = DOMINANCE_SWEEP_VERSION
    return selected_df


def predefined_winner_criteria(boundary_caveats: pd.DataFrame | None = None) -> dict[str, Any]:
    caveats: list[str] = [
        "Opposite-goal contests use per-cell reverse-direction behavior; dominance remains a computational proxy.",
        "Winner calls use final-state target scores and global-direction scores, not causal proof or biological validation.",
        "Resource-capped runs are retained but flagged because unresolved dynamics can inflate or obscure dominance.",
    ]
    if boundary_caveats is not None and not boundary_caveats.empty:
        for text in boundary_caveats.get("metricBoundaryJson", pd.Series(dtype=str)).dropna().astype(str):
            parsed = parse_json_maybe(text, [])
            if isinstance(parsed, Sequence) and not isinstance(parsed, str):
                caveats.extend(str(item) for item in parsed)
    return {
        "schema": "eidosoma.e06_s06_winner_criteria.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "criteriaVersion": WINNER_CRITERIA_VERSION,
        "sourceCompatibilityVectorVersion": "e06_s05_compatibility_vector.v1",
        "winnerRuleOrder": [
            "policy_goal_margin",
            "global_direction_margin",
            "balanced_or_unresolved",
        ],
        "thresholds": {
            "minimumPolicyGoalMarginPct": 10.0,
            "minimumWinnerTargetQualityPct": 50.0,
            "minimumGlobalDirectionMarginPct": 15.0,
            "weakDominanceProxyScore": 0.10,
            "moderateDominanceProxyScore": 0.20,
            "strongDominanceProxyScore": 0.35,
            "stablePairWinnerShare": 0.67,
            "wilsonZ95": 1.96,
        },
        "dominanceProxyFormula": (
            "max(abs(leftPolicyGoalScore - rightPolicyGoalScore) / 100, "
            "abs(finalIncreasingSortednessPercent - finalDecreasingSortednessPercent) / 100)"
        ),
        "winnerDefinition": (
            "A policy wins a context when its own target-quality score exceeds the opponent by at least "
            "minimumPolicyGoalMarginPct and reaches minimumWinnerTargetQualityPct. If that is not met, "
            "the policy aligned to the globally dominant direction wins when the increasing-vs-decreasing "
            "directional margin reaches minimumGlobalDirectionMarginPct. Otherwise the context is unresolved."
        ),
        "stablePairDefinition": (
            "Pair-level dominance is stable when the modal context winner share is at least "
            "stablePairWinnerShare; Wilson 95% intervals are reported as uncertainty proxies over tested contexts."
        ),
        "metricBoundaryCaveats": sorted(set(caveats)),
        "claimBoundary": CLAIM_BOUNDARY,
    }


def winner_criteria_table(criteria: Mapping[str, Any]) -> pd.DataFrame:
    thresholds = dict(criteria.get("thresholds", {}))
    rows = [
        {
            "criterionId": "policy_goal_margin",
            "order": 1,
            "rule": "winner is the policy with the higher per-policy target score",
            "requiredThresholdsJson": compact_json(
                {
                    "minimumPolicyGoalMarginPct": thresholds["minimumPolicyGoalMarginPct"],
                    "minimumWinnerTargetQualityPct": thresholds["minimumWinnerTargetQualityPct"],
                }
            ),
            "metricBoundary": "Uses S04/S05 per-policy goal scores under explicit opposite-goal labels.",
        },
        {
            "criterionId": "global_direction_margin",
            "order": 2,
            "rule": "winner is the policy whose targetDirection matches the stronger global direction",
            "requiredThresholdsJson": compact_json({"minimumGlobalDirectionMarginPct": thresholds["minimumGlobalDirectionMarginPct"]}),
            "metricBoundary": "Used only when per-policy target-score margin is too small; still a directional proxy.",
        },
        {
            "criterionId": "balanced_or_unresolved",
            "order": 3,
            "rule": "no winner is called when neither margin clears its threshold",
            "requiredThresholdsJson": compact_json({}),
            "metricBoundary": "Avoids overcalling weak or resource-limited differences.",
        },
    ]
    return pd.DataFrame(rows)


def write_winner_criteria_markdown(criteria: Mapping[str, Any], path: Path) -> Path:
    thresholds = criteria["thresholds"]
    caveat_lines = "\n".join(f"- {item}" for item in criteria.get("metricBoundaryCaveats", []))
    text = f"""# E06 S06 Winner Criteria

- Research step ID: `{criteria['researchStepId']}`
- Step number: {criteria['stepNumber']}
- Criteria version: `{criteria['criteriaVersion']}`

## Explicit Winner Rule

{criteria['winnerDefinition']}

## Dominance Proxy Formula

`{criteria['dominanceProxyFormula']}`

## Thresholds

- Policy-goal margin: {thresholds['minimumPolicyGoalMarginPct']:.1f} percentage points
- Minimum winner target quality: {thresholds['minimumWinnerTargetQualityPct']:.1f}%
- Global direction margin: {thresholds['minimumGlobalDirectionMarginPct']:.1f} percentage points
- Weak/moderate/strong dominance proxy: {thresholds['weakDominanceProxyScore']:.2f}, {thresholds['moderateDominanceProxyScore']:.2f}, {thresholds['strongDominanceProxyScore']:.2f}
- Stable pair modal-winner share: {thresholds['stablePairWinnerShare']:.2f}

## Metric-Boundary Caveats

{caveat_lines}
"""
    path.write_text(text, encoding="utf-8")
    return path


def _seed_bundle(
    pair_index: int,
    orientation_index: int,
    ratio_index: int,
    arrangement_index: int,
    profile_index: int,
    seed_index: int,
) -> dict[str, int]:
    base = (
        861_000
        + int(pair_index) * 10_000
        + int(orientation_index) * 4_000
        + int(ratio_index) * 500
        + int(arrangement_index) * 100
        + int(profile_index) * 25
        + int(seed_index) * 7
    )
    return {
        "seedIndex": int(seed_index),
        "valueSeed": int(base + 11),
        "arrangementSeed": int(base + 23),
        "schedulerSeed": int(base + 37),
        "tieBreakerSeed": int(base + 53),
    }


def _perturbation_type(value_profile: str) -> str:
    if value_profile == "random_unique":
        return "none"
    if value_profile == "reversed_unique":
        return "input_order_reversal_stress"
    if value_profile == "duplicate_1_10_x10":
        return "duplicate_value_distribution"
    return "other_value_profile"


def build_dominance_condition_matrix(
    selected_pairs: pd.DataFrame,
    ready_panel: pd.DataFrame,
    criteria: Mapping[str, Any],
    *,
    n: int = 100,
    seed_count: int = 1,
    ratios: Sequence[tuple[int, int]] = DOMINANCE_RATIOS,
    arrangements: Sequence[str] = DOMINANCE_ARRANGEMENTS,
    value_profiles: Sequence[str] = DOMINANCE_VALUE_PROFILES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ready_ids = set(ready_panel["panelPolicyId"].astype(str))
    criteria_hash = sha256_text(compact_json(criteria))[:16]
    rows: list[dict[str, Any]] = []
    condition_meta: list[dict[str, Any]] = []
    policy_meta: list[dict[str, Any]] = []
    arrangement_rows: list[dict[str, Any]] = []
    arrangement_offsets = {name: index for index, name in enumerate(arrangements)}

    for pair in selected_pairs.to_dict(orient="records"):
        policy_a = str(pair["policyAPanelPolicyId"])
        policy_b = str(pair["policyBPanelPolicyId"])
        if {policy_a, policy_b} - ready_ids:
            raise ValueError("S06 selected pair includes policy not marked ready")
        orientations = [(policy_a, policy_b, "A_increasing_B_decreasing"), (policy_b, policy_a, "B_increasing_A_decreasing")]
        for orientation_index, (left_id, right_id, orientation_label) in enumerate(orientations):
            policy_ids = [left_id, right_id]
            for ratio_index, (left_count, right_count) in enumerate(ratios):
                if int(left_count) + int(right_count) != int(n):
                    raise ValueError("S06 dominance ratios require counts that sum to n")
                counts = [int(left_count), int(right_count)]
                ratio_label = f"{left_count}:{right_count}"
                target_counts = dict(zip(policy_ids, counts, strict=True))
                for arrangement_index, arrangement_type in enumerate(arrangements):
                    for profile_index, value_profile in enumerate(value_profiles):
                        for seed_index in range(int(seed_count)):
                            seeds = _seed_bundle(
                                int(pair["s06PairIndex"]),
                                orientation_index,
                                ratio_index,
                                arrangement_index,
                                profile_index,
                                seed_index,
                            )
                            assigned, numeric_labels, descriptor = generate_arrangement(
                                policy_ids,
                                counts,
                                str(arrangement_type),
                                seed=int(seeds["arrangementSeed"]),
                            )
                            specs = policy_goal_specs(policy_ids, "opposite_goal")
                            reverse = reverse_directions_for_assignment(assigned, specs)
                            goal_meta = condition_goal_metadata(policy_ids, "opposite_goal")
                            goal_meta = {
                                **goal_meta,
                                "winnerCriteriaVersion": WINNER_CRITERIA_VERSION,
                                "winnerCriteriaHash": criteria_hash,
                            }
                            condition_payload = {
                                "pairKey": pair["s06PairKey"],
                                "policyIds": policy_ids,
                                "counts": counts,
                                "ratioLabel": ratio_label,
                                "arrangementType": arrangement_type,
                                "valueProfile": value_profile,
                                "orientation": orientation_label,
                                "seeds": seeds,
                                "criteriaHash": criteria_hash,
                            }
                            condition_id = (
                                f"s06_p{int(pair['s06PairIndex']):03d}_o{orientation_index}_"
                                f"{ratio_label.replace(':', '_')}_{arrangement_type}_"
                                f"{value_profile}_seed{seed_index}"
                            )
                            row = {
                                "conditionId": condition_id,
                                "researchStepId": STEP_ID,
                                "implementationPrefix": "e06_s06",
                                "s06Version": DOMINANCE_SWEEP_VERSION,
                                "winnerCriteriaVersion": WINNER_CRITERIA_VERSION,
                                "winnerCriteriaHash": criteria_hash,
                                "s06PairIndex": int(pair["s06PairIndex"]),
                                "s06PairKey": str(pair["s06PairKey"]),
                                "s06SelectionReason": str(pair["s06SelectionReason"]),
                                "policyAPanelPolicyId": policy_a,
                                "policyBPanelPolicyId": policy_b,
                                "mixtureKind": "pair",
                                "pairCategory": str(pair["pairCategory"]),
                                "priorityReason": str(pair.get("s06SelectionReason", "")),
                                "leftPanelPolicyId": left_id,
                                "rightPanelPolicyId": right_id,
                                "thirdPanelPolicyId": "",
                                "oppositeGoalOrientation": orientation_label,
                                "ratioLabel": ratio_label,
                                "targetCountsJson": compact_json(target_counts),
                                "targetProportionsJson": compact_json({pid: count / int(n) for pid, count in target_counts.items()}),
                                "n": int(n),
                                "targetPolicyCount": 2,
                                "policyIdsJson": compact_json(policy_ids),
                                "policyCountsJson": compact_json(counts),
                                "arrangementType": str(arrangement_type),
                                "arrangementTypeIndex": int(arrangement_offsets[str(arrangement_type)]),
                                "arrangementPolicyIdsJson": compact_json(assigned),
                                "arrangementNumericLabelsJson": compact_json(numeric_labels),
                                "arrangementDescriptorJson": compact_json(descriptor),
                                "arrangementHash": str(descriptor["arrangementHash"]),
                                "valueProfile": str(value_profile),
                                "perturbationType": _perturbation_type(str(value_profile)),
                                "goalMode": "opposite_goal",
                                "goalModeIndex": 0,
                                "goalCompatibilityClass": "opposite",
                                "conditionGoalMetadataJson": compact_json(goal_meta),
                                "policyGoalMetadataJson": compact_json(specs),
                                "goalReverseDirectionsJson": compact_json(reverse),
                                "goalDirectionByPolicyJson": compact_json({pid: specs[pid]["targetDirection"] for pid in policy_ids}),
                                "goalBehaviorDirectionByPolicyJson": compact_json({pid: specs[pid]["behaviorDirection"] for pid in policy_ids}),
                                "conditionHash": sha256_text(compact_json(condition_payload))[:16],
                                "paperExpectedWinnerPolicyId": str(pair.get("paperExpectedWinnerPolicyId", "")),
                                "paperExpectedWinnerAlgotype": str(pair.get("paperExpectedWinnerAlgotype", "")),
                                "metricBoundary": str(goal_meta["metricBoundary"]),
                                **seeds,
                            }
                            rows.append(row)
                            condition_meta.append(
                                {
                                    "conditionId": condition_id,
                                    "s06PairKey": str(pair["s06PairKey"]),
                                    "goalMode": "opposite_goal",
                                    "goalCompatibilityClass": "opposite",
                                    "conditionGoalMetadataJson": compact_json(goal_meta),
                                    "winnerCriteriaVersion": WINNER_CRITERIA_VERSION,
                                    "winnerCriteriaHash": criteria_hash,
                                    "goalReverseDirectionsJson": compact_json(reverse),
                                    "metricBoundary": str(goal_meta["metricBoundary"]),
                                }
                            )
                            counts_by_policy = Counter(assigned)
                            for pid in policy_ids:
                                spec = specs[pid]
                                policy_meta.append(
                                    {
                                        "conditionId": condition_id,
                                        "s06PairKey": str(pair["s06PairKey"]),
                                        "panelPolicyId": pid,
                                        "goalMode": "opposite_goal",
                                        "goalType": str(spec["goalType"]),
                                        "targetDirection": str(spec["targetDirection"]),
                                        "behaviorDirection": str(spec["behaviorDirection"]),
                                        "role": str(spec["role"]),
                                        "cellCount": int(counts_by_policy[pid]),
                                        "policyGoalSpecJson": compact_json(spec),
                                    }
                                )
                            arrangement_rows.append(
                                {
                                    "conditionId": condition_id,
                                    "s06PairKey": str(pair["s06PairKey"]),
                                    "ratioLabel": ratio_label,
                                    "arrangementType": str(arrangement_type),
                                    "valueProfile": str(value_profile),
                                    "perturbationType": _perturbation_type(str(value_profile)),
                                    "policyIdsJson": compact_json(policy_ids),
                                    "policyCountsJson": compact_json(counts),
                                    "arrangementSeed": int(seeds["arrangementSeed"]),
                                    "arrangementHash": str(descriptor["arrangementHash"]),
                                    "arrangementDescriptorJson": compact_json(descriptor),
                                    "interfaceCount": int(descriptor["interfaceCount"]),
                                    "interfaceDensity": float(descriptor["interfaceDensity"]),
                                    "runCount": int(descriptor["runCount"]),
                                    "longestRun": int(descriptor["longestRun"]),
                                    "initialAggregation": float(descriptor["initialAggregation"]),
                                }
                            )

    return pd.DataFrame(rows), pd.DataFrame(condition_meta), pd.DataFrame(policy_meta), pd.DataFrame(arrangement_rows)


def _oscillation_risk(row: Mapping[str, Any]) -> float:
    if bool(row.get("completed", False)):
        return 0.0
    stop_reason = str(row.get("stopReason", "")).lower()
    if "max_activation" in stop_reason:
        return 1.0
    if stop_reason.startswith("max_"):
        return 0.75
    if "no_cell_can_move" in stop_reason or "no_move" in stop_reason:
        return 0.2
    return 0.5


def _direction_winner(
    policy_ids: Sequence[str],
    specs: Mapping[str, Mapping[str, Any]],
    direction: str,
) -> str:
    for pid in policy_ids:
        spec = specs.get(str(pid), {})
        if str(spec.get("targetDirection", "")) == str(direction):
            return str(pid)
    return ""


def score_dominance_results(result_df: pd.DataFrame, criteria: Mapping[str, Any]) -> pd.DataFrame:
    thresholds = dict(criteria["thresholds"])
    scored_rows: list[dict[str, Any]] = []
    for row in result_df.to_dict(orient="records"):
        policy_ids = [str(item) for item in parse_json_maybe(row.get("policyIdsJson", ""), [])]
        left_id = str(row.get("leftPanelPolicyId", ""))
        right_id = str(row.get("rightPanelPolicyId", ""))
        scores = parse_json_maybe(row.get("policyGoalScoresJson", ""), {})
        specs = parse_json_maybe(row.get("policyGoalMetadataJson", ""), {})
        scores = scores if isinstance(scores, Mapping) else {}
        specs = specs if isinstance(specs, Mapping) else {}
        left_score = safe_float(scores.get(left_id))
        right_score = safe_float(scores.get(right_id))
        inc = safe_float(row.get("finalIncreasingSortednessPercent"))
        dec = safe_float(row.get("finalDecreasingSortednessPercent"))
        goal_margin = left_score - right_score if np.isfinite(left_score) and np.isfinite(right_score) else float("nan")
        direction_margin = inc - dec if np.isfinite(inc) and np.isfinite(dec) else float("nan")
        dominance_asymmetry = abs(goal_margin) / 100.0 if np.isfinite(goal_margin) else 0.0
        directional = abs(direction_margin) / 100.0 if np.isfinite(direction_margin) else 0.0
        dominance_proxy = clip01(max(dominance_asymmetry, directional))
        final_target = clip01(safe_float(row.get("meanPolicyGoalScore")) / 100.0)
        min_target = clip01(safe_float(row.get("minPolicyGoalScore")) / 100.0)
        winner = UNRESOLVED_WINNER
        criterion = "balanced_or_unresolved"
        criterion_detail = "neither target-score nor global-direction margin cleared the predeclared threshold"
        if (
            bool(row.get("runSucceeded", False))
            and np.isfinite(goal_margin)
            and abs(goal_margin) >= float(thresholds["minimumPolicyGoalMarginPct"])
            and max(left_score, right_score) >= float(thresholds["minimumWinnerTargetQualityPct"])
        ):
            winner = left_id if goal_margin > 0 else right_id
            criterion = "policy_goal_margin"
            criterion_detail = (
                f"absolute per-policy goal-score margin {abs(goal_margin):.2f} pp with winner target score "
                f"{max(left_score, right_score):.2f}%"
            )
        elif bool(row.get("runSucceeded", False)) and np.isfinite(direction_margin) and abs(direction_margin) >= float(
            thresholds["minimumGlobalDirectionMarginPct"]
        ):
            direction = "increasing" if direction_margin > 0 else "decreasing"
            direction_policy = _direction_winner(policy_ids, specs, direction)
            if direction_policy:
                winner = direction_policy
                criterion = "global_direction_margin"
                criterion_detail = f"global {direction} direction margin {abs(direction_margin):.2f} pp"
        loser = ""
        if winner != UNRESOLVED_WINNER:
            loser_candidates = [pid for pid in policy_ids if pid != winner]
            loser = loser_candidates[0] if loser_candidates else ""
        if winner == UNRESOLVED_WINNER:
            confidence = "unresolved"
        elif dominance_proxy >= float(thresholds["strongDominanceProxyScore"]):
            confidence = "strong"
        elif dominance_proxy >= float(thresholds["moderateDominanceProxyScore"]):
            confidence = "moderate"
        else:
            confidence = "weak"
        oscillation = _oscillation_risk(row)
        caveat_flags = ["opposite_goal_reverse_behavior"]
        if oscillation >= 0.75:
            caveat_flags.append("resource_cap_unresolved_dynamics")
        if str(row.get("valueProfile", "")) == "duplicate_1_10_x10":
            caveat_flags.append("duplicate_value_target_ties")
        expected = str(row.get("paperExpectedWinnerPolicyId", ""))
        if not expected:
            paper_eval = "not_applicable"
            paper_match: bool | None = None
        elif winner == UNRESOLVED_WINNER:
            paper_eval = "unresolved"
            paper_match = None
        elif winner == expected:
            paper_eval = "match"
            paper_match = True
        else:
            paper_eval = "mismatch"
            paper_match = False
        out = {
            **row,
            "researchStepId": STEP_ID,
            "sourceResearchStepId": "S06",
            "dominanceSweepVersion": DOMINANCE_SWEEP_VERSION,
            "winnerCriteriaVersion": WINNER_CRITERIA_VERSION,
            "leftPolicyGoalScore": left_score,
            "rightPolicyGoalScore": right_score,
            "leftMinusRightGoalScoreMarginPct": goal_margin,
            "absoluteGoalScoreMarginPct": abs(goal_margin) if np.isfinite(goal_margin) else float("nan"),
            "increasingMinusDecreasingMarginPct": direction_margin,
            "absoluteDirectionMarginPct": abs(direction_margin) if np.isfinite(direction_margin) else float("nan"),
            "dominanceAsymmetryScore": clip01(dominance_asymmetry),
            "directionalDominanceScore": clip01(directional),
            "dominanceProxyScore": dominance_proxy,
            "finalTargetQualityScore": final_target,
            "minTargetQualityScore": min_target,
            "winnerPolicyId": winner,
            "loserPolicyId": loser,
            "winnerCriterion": criterion,
            "winnerCriterionDetail": criterion_detail,
            "winnerConfidenceClass": confidence,
            "winnerTargetDirection": str(specs.get(winner, {}).get("targetDirection", "")) if winner != UNRESOLVED_WINNER else "",
            "winnerGoalScore": safe_float(scores.get(winner)) if winner != UNRESOLVED_WINNER else float("nan"),
            "loserGoalScore": safe_float(scores.get(loser)) if loser else float("nan"),
            "oscillationRiskProxy": oscillation,
            "metricBoundaryCaveatFlagsJson": compact_json(caveat_flags),
            "paperAnchorEvaluation": paper_eval,
            "paperAnchorMatch": paper_match,
            "claimBoundary": CLAIM_BOUNDARY,
        }
        scored_rows.append(out)
    return pd.DataFrame(scored_rows)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def summarize_pair_dominance(scored: pd.DataFrame, criteria: Mapping[str, Any]) -> pd.DataFrame:
    thresholds = dict(criteria["thresholds"])
    rows: list[dict[str, Any]] = []
    for pair_key, group in scored.groupby("s06PairKey", sort=False):
        winners = [str(value) for value in group["winnerPolicyId"].astype(str)]
        resolved = [value for value in winners if value != UNRESOLVED_WINNER]
        winner_counts = Counter(resolved)
        modal_winner, modal_count = ("", 0)
        if winner_counts:
            modal_winner, modal_count = winner_counts.most_common(1)[0]
        total = int(len(group))
        lower, upper = wilson_interval(int(modal_count), total, float(thresholds["wilsonZ95"]))
        modal_share = modal_count / max(1, total)
        stable = bool(modal_winner and modal_share >= float(thresholds["stablePairWinnerShare"]))
        if not modal_winner:
            dependency = "mostly_unresolved_proxy"
        elif stable and lower >= 0.50:
            dependency = "stable_dominance_proxy"
        elif stable:
            dependency = "modal_but_uncertain_dominance_proxy"
        else:
            dependency = "context_dependent_dominance_proxy"
        paper_rows = group[group["paperAnchorEvaluation"].isin(["match", "mismatch"])]
        paper_match_rate = float(paper_rows["paperAnchorMatch"].map(bool).mean()) if not paper_rows.empty else float("nan")
        rows.append(
            {
                "s06PairKey": str(pair_key),
                "policyAPanelPolicyId": str(group["policyAPanelPolicyId"].iloc[0]),
                "policyBPanelPolicyId": str(group["policyBPanelPolicyId"].iloc[0]),
                "pairCategory": str(group["pairCategory"].iloc[0]),
                "conditionCount": total,
                "resolvedWinnerCount": int(len(resolved)),
                "unresolvedCount": int(total - len(resolved)),
                "winnerCountsJson": compact_json(dict(winner_counts)),
                "modalWinnerPolicyId": modal_winner,
                "modalWinnerCount": int(modal_count),
                "modalWinnerShare": float(modal_share),
                "modalWinnerWilson95Lower": lower,
                "modalWinnerWilson95Upper": upper,
                "contextStableWinner": stable,
                "contextDependencyClass": dependency,
                "meanDominanceProxyScore": float(group["dominanceProxyScore"].mean()),
                "meanFinalTargetQualityScore": float(group["finalTargetQualityScore"].mean()),
                "meanMinTargetQualityScore": float(group["minTargetQualityScore"].mean()),
                "strongOrModerateWinRate": float(group["winnerConfidenceClass"].isin(["strong", "moderate"]).mean()),
                "meanOscillationRiskProxy": float(group["oscillationRiskProxy"].mean()),
                "paperExpectedWinnerPolicyId": str(group["paperExpectedWinnerPolicyId"].iloc[0]),
                "paperExpectedWinnerAlgotype": str(group["paperExpectedWinnerAlgotype"].iloc[0]),
                "paperAnchorMatchRate": paper_match_rate,
                "ratioLabelsJson": compact_json(sorted(group["ratioLabel"].astype(str).unique())),
                "arrangementTypesJson": compact_json(sorted(group["arrangementType"].astype(str).unique())),
                "valueProfilesJson": compact_json(sorted(group["valueProfile"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def build_dominance_edges(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in pair_summary.to_dict(orient="records"):
        winner = str(row.get("modalWinnerPolicyId", ""))
        if not winner:
            continue
        policies = [str(row["policyAPanelPolicyId"]), str(row["policyBPanelPolicyId"])]
        loser_candidates = [pid for pid in policies if pid != winner]
        if not loser_candidates:
            continue
        loser = loser_candidates[0]
        rows.append(
            {
                "winnerPolicyId": winner,
                "loserPolicyId": loser,
                "s06PairKey": str(row["s06PairKey"]),
                "pairCategory": str(row["pairCategory"]),
                "edgeWeight": float(row["modalWinnerShare"]) * float(row["meanDominanceProxyScore"]),
                "modalWinnerShare": float(row["modalWinnerShare"]),
                "modalWinnerWilson95Lower": float(row["modalWinnerWilson95Lower"]),
                "modalWinnerWilson95Upper": float(row["modalWinnerWilson95Upper"]),
                "conditionCount": int(row["conditionCount"]),
                "contextDependencyClass": str(row["contextDependencyClass"]),
            }
        )
    return pd.DataFrame(rows)


def build_policy_hierarchy(scored: pd.DataFrame, pair_summary: pd.DataFrame, ready_panel: pd.DataFrame) -> pd.DataFrame:
    confidence_weight = {"strong": 1.0, "moderate": 0.67, "weak": 0.33, "unresolved": 0.0}
    class_by_id = _class_by_policy(ready_panel)
    group_by_id = _group_by_policy(ready_panel)
    policy_ids = sorted({pid for text in scored["policyIdsJson"].astype(str) for pid in parse_json_maybe(text, [])})
    rows: list[dict[str, Any]] = []
    for pid in policy_ids:
        involved = scored[
            (scored["leftPanelPolicyId"].astype(str) == pid)
            | (scored["rightPanelPolicyId"].astype(str) == pid)
        ].copy()
        wins = involved[involved["winnerPolicyId"].astype(str) == pid]
        resolved_losses = involved[
            (involved["winnerPolicyId"].astype(str) != UNRESOLVED_WINNER)
            & (involved["winnerPolicyId"].astype(str) != pid)
        ]
        unresolved = involved[involved["winnerPolicyId"].astype(str) == UNRESOLVED_WINNER]
        win_weights = wins["winnerConfidenceClass"].map(confidence_weight).fillna(0.0)
        outgoing = Counter(wins["loserPolicyId"].dropna().astype(str))
        incoming = Counter(resolved_losses["winnerPolicyId"].dropna().astype(str))
        stable_pairs = pair_summary[
            (pair_summary["modalWinnerPolicyId"].astype(str) == pid)
            & pair_summary["contextStableWinner"].map(bool)
        ]
        rows.append(
            {
                "panelPolicyId": pid,
                "panelGroup": group_by_id.get(pid, ""),
                "classLabel": class_by_id.get(pid, ""),
                "contestContextCount": int(len(involved)),
                "winCount": int(len(wins)),
                "lossCount": int(len(resolved_losses)),
                "unresolvedCount": int(len(unresolved)),
                "winRate": float(len(wins) / max(1, len(involved))),
                "resolvedWinRate": float(len(wins) / max(1, len(wins) + len(resolved_losses))),
                "weightedWinScore": float(win_weights.sum() / max(1, len(involved))),
                "stablePairWinCount": int(len(stable_pairs)),
                "meanDominanceProxyWhenWinning": float(wins["dominanceProxyScore"].mean()) if not wins.empty else float("nan"),
                "meanTargetQualityWhenWinning": float(wins["finalTargetQualityScore"].mean()) if not wins.empty else float("nan"),
                "outgoingWinCountsJson": compact_json(dict(outgoing)),
                "incomingLossCountsJson": compact_json(dict(incoming)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["stablePairWinCount", "weightedWinScore", "resolvedWinRate", "winRate", "panelPolicyId"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_contexts(scored: pd.DataFrame) -> pd.DataFrame:
    grouped = scored.groupby(["ratioLabel", "arrangementType", "valueProfile", "perturbationType"], dropna=False)
    return grouped.agg(
        conditionCount=("conditionId", "size"),
        meanDominanceProxyScore=("dominanceProxyScore", "mean"),
        meanFinalTargetQualityScore=("finalTargetQualityScore", "mean"),
        meanMinTargetQualityScore=("minTargetQualityScore", "mean"),
        unresolvedRate=("winnerPolicyId", lambda values: float(np.mean(np.asarray(values, dtype=str) == UNRESOLVED_WINNER))),
        strongOrModerateWinRate=("winnerConfidenceClass", lambda values: float(np.mean(pd.Series(values).isin(["strong", "moderate"])))),
        meanOscillationRiskProxy=("oscillationRiskProxy", "mean"),
    ).reset_index()


def build_paper_order_comparison(scored: pd.DataFrame, e01_anchors: pd.DataFrame) -> pd.DataFrame:
    paper = scored[scored["paperExpectedWinnerPolicyId"].astype(str).str.len() > 0].copy()
    rows: list[dict[str, Any]] = []
    if paper.empty:
        return pd.DataFrame(
            columns=[
                "paperComparisonPair",
                "expectedWinnerPolicyId",
                "expectedWinnerAlgotype",
                "conditionCount",
                "matchCount",
                "mismatchCount",
                "unresolvedCount",
                "matchRateResolved",
                "e01DominanceMatchesPaperRate",
            ]
        )
    for pair_key, group in paper.groupby("s06PairKey", sort=False):
        expected_alg = str(group["paperExpectedWinnerAlgotype"].iloc[0])
        policy_labels = sorted(
            [
                _policy_algotype(str(group["policyAPanelPolicyId"].iloc[0])),
                _policy_algotype(str(group["policyBPanelPolicyId"].iloc[0])),
            ]
        )
        pair_name = "|".join(policy_labels)
        resolved = group[group["paperAnchorEvaluation"].isin(["match", "mismatch"])]
        match_count = int((resolved["paperAnchorEvaluation"] == "match").sum())
        mismatch_count = int((resolved["paperAnchorEvaluation"] == "mismatch").sum())
        unresolved_count = int((group["paperAnchorEvaluation"] == "unresolved").sum())
        e01_rate = float("nan")
        if not e01_anchors.empty and "expectedDominantAlgotype" in e01_anchors.columns:
            hits = e01_anchors[e01_anchors["expectedDominantAlgotype"].astype(str) == expected_alg]
            if not hits.empty:
                e01_rate = float(hits["dominanceMatchesPaperRate"].mean())
        rows.append(
            {
                "paperComparisonPair": pair_name,
                "s06PairKey": str(pair_key),
                "expectedWinnerPolicyId": str(group["paperExpectedWinnerPolicyId"].iloc[0]),
                "expectedWinnerAlgotype": expected_alg,
                "conditionCount": int(len(group)),
                "matchCount": match_count,
                "mismatchCount": mismatch_count,
                "unresolvedCount": unresolved_count,
                "matchRateResolved": float(match_count / max(1, match_count + mismatch_count)),
                "e01DominanceMatchesPaperRate": e01_rate,
                "paperAnchorEvaluationCountsJson": compact_json(dict(Counter(group["paperAnchorEvaluation"].astype(str)))),
            }
        )
    return pd.DataFrame(rows)


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
        rows.append(
            {
                "conditionId": condition["conditionId"],
                "valueProfile": condition["valueProfile"],
                "replayMatch": bool(all(first.get(key) == second.get(key) for key in keys)),
                "firstFinalStateHash": first.get("finalStateHash"),
                "secondFinalStateHash": second.get("finalStateHash"),
                "firstStopReason": first.get("stopReason"),
                "secondStopReason": second.get("stopReason"),
                "checkedKeysJson": compact_json(keys),
            }
        )
    return pd.DataFrame(rows)


def validation_checks(
    condition_df: pd.DataFrame,
    scored: pd.DataFrame,
    ready_panel: pd.DataFrame,
    criteria: Mapping[str, Any],
    criteria_df: pd.DataFrame,
    policy_goal_df: pd.DataFrame,
    pair_summary: pd.DataFrame,
    paper_comparison: pd.DataFrame,
    smoke_df: pd.DataFrame,
) -> pd.DataFrame:
    ready_ids = set(ready_panel["panelPolicyId"].astype(str))
    condition_policy_ids = {
        str(pid)
        for text in condition_df["policyIdsJson"].astype(str)
        for pid in parse_json_maybe(text, [])
    }
    reverse_ok = True
    for row in condition_df.to_dict(orient="records"):
        assigned = [str(item) for item in parse_json_maybe(row["arrangementPolicyIdsJson"], [])]
        reverse = [bool(item) for item in parse_json_maybe(row["goalReverseDirectionsJson"], [])]
        specs = parse_json_maybe(row["policyGoalMetadataJson"], {})
        if len(reverse) != int(row["n"]) or reverse != reverse_directions_for_assignment(assigned, specs):
            reverse_ok = False
            break
    score_bounds = (
        scored["dominanceProxyScore"].between(0.0, 1.0).all()
        and scored["finalTargetQualityScore"].between(0.0, 1.0).all()
        and scored["minTargetQualityScore"].between(0.0, 1.0).all()
    )
    paper_complete = bool(
        not paper_comparison.empty
        and set(PAPER_ORIGINAL_ORDER.values()) <= set(paper_comparison["expectedWinnerAlgotype"].astype(str))
        and paper_comparison["conditionCount"].ge(1).all()
    )
    interval_ok = bool(
        not pair_summary.empty
        and pair_summary["modalWinnerWilson95Lower"].dropna().between(0.0, 1.0).all()
        and pair_summary["modalWinnerWilson95Upper"].dropna().between(0.0, 1.0).all()
    )
    checks = [
        {
            "checkId": "winner_criteria_predeclared",
            "success": bool(criteria.get("criteriaVersion") == WINNER_CRITERIA_VERSION and len(criteria_df) == 3),
            "detail": f"{WINNER_CRITERIA_VERSION} written before scoring with {len(criteria_df)} ordered criteria rows",
        },
        {
            "checkId": "only_ready_policies",
            "success": bool(condition_policy_ids <= ready_ids),
            "detail": f"{len(condition_policy_ids)} S06 policies all came from readyForS02Mixing=true rows",
        },
        {
            "checkId": "opposite_goal_labels_complete",
            "success": bool(
                set(condition_df["goalMode"].astype(str)) == {"opposite_goal"}
                and reverse_ok
                and {"increasing", "decreasing"} <= set(policy_goal_df["targetDirection"].astype(str))
            ),
            "detail": "all conditions carry opposite-goal metadata and per-cell reverse vectors matching policy behavior directions",
        },
        {
            "checkId": "context_axes_present",
            "success": bool(
                condition_df["ratioLabel"].nunique() >= 3
                and condition_df["arrangementType"].nunique() >= 4
                and condition_df["valueProfile"].nunique() >= 2
                and condition_df["perturbationType"].nunique() >= 2
                and condition_df["oppositeGoalOrientation"].nunique() == 2
            ),
            "detail": "condition matrix crosses ratios, placements, value profiles, perturbation labels, and both opposite-goal orientations",
        },
        {
            "checkId": "one_result_per_condition",
            "success": bool(len(scored) == len(condition_df) and scored["conditionId"].is_unique),
            "detail": f"{len(scored)} result rows for {len(condition_df)} S06 conditions",
        },
        {
            "checkId": "runs_succeeded_and_counts_conserved",
            "success": bool(
                scored["runSucceeded"].fillna(False).map(bool).all()
                and scored["valueCountsConserved"].fillna(False).map(bool).all()
                and scored["policyCountsConserved"].fillna(False).map(bool).all()
            ),
            "detail": "all S06 runs succeeded and conserved values/policy labels",
        },
        {
            "checkId": "winner_scores_populated_and_bounded",
            "success": bool(score_bounds and scored["winnerCriterion"].astype(str).str.len().gt(0).all()),
            "detail": "dominance, target-quality, direction, and winner-criterion columns are populated and bounded",
        },
        {
            "checkId": "uncertainty_intervals_populated",
            "success": interval_ok,
            "detail": f"Wilson 95% intervals written for {len(pair_summary)} pair summaries",
        },
        {
            "checkId": "paper_order_comparison_populated",
            "success": paper_complete,
            "detail": f"{len(paper_comparison)} original-classic pair comparisons scored against E01 paper-order anchors",
        },
        {
            "checkId": "metric_boundary_caveats_retained",
            "success": bool(
                scored["metricBoundary"].astype(str).str.contains("dominance is a proxy").all()
                and scored["metricBoundaryCaveatFlagsJson"].astype(str).str.contains("opposite_goal_reverse_behavior").all()
            ),
            "detail": "opposite-goal metric-boundary caveats and per-row caveat flags are retained",
        },
        {
            "checkId": "smoke_replay_deterministic",
            "success": bool(not smoke_df.empty and smoke_df["replayMatch"].map(bool).all()),
            "detail": f"{len(smoke_df)} smoke replay checks matched",
        },
    ]
    return pd.DataFrame(checks)


def write_dominance_plots(
    edges: pd.DataFrame,
    context_summary: pd.DataFrame,
    figure_dir: Path,
    step_dir: Path,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not edges.empty:
        plot_df = edges.sort_values("edgeWeight", ascending=True).tail(12)
        labels = [f"{row.winnerPolicyId[:24]} -> {row.loserPolicyId[:24]}" for row in plot_df.itertuples(index=False)]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df) + 1.5)))
        ax.barh(labels, plot_df["edgeWeight"])
        ax.set_xlabel("Modal support x mean dominance proxy")
        ax.set_title("E06 S06 dominance hierarchy edges")
        fig.tight_layout()
        paths.extend(
            [
                figure_dir / "e06_s06_dominance_network.png",
                figure_dir / "e06_s06_dominance_network.pdf",
                step_dir / "dominance_network.png",
                step_dir / "dominance_network.pdf",
            ]
        )
        for path in paths[-4:]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
        plt.close(fig)
    if not context_summary.empty:
        pivot = (
            context_summary.groupby(["arrangementType", "valueProfile"], dropna=False)["meanDominanceProxyScore"]
            .mean()
            .reset_index()
        )
        pivot["label"] = pivot["arrangementType"].astype(str) + "\n" + pivot["valueProfile"].astype(str)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(pivot["label"], pivot["meanDominanceProxyScore"])
        ax.set_ylabel("Mean dominance proxy")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=35)
        ax.set_title("E06 S06 dominance by arrangement and value profile")
        fig.tight_layout()
        context_paths = [
            figure_dir / "e06_s06_context_dependency.png",
            figure_dir / "e06_s06_context_dependency.pdf",
            step_dir / "context_dependency.png",
            step_dir / "context_dependency.pdf",
        ]
        for path in context_paths:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
        plt.close(fig)
        paths.extend(context_paths)
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
    pair_summary: pd.DataFrame,
    hierarchy: pd.DataFrame,
    paper_comparison: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Path:
    top_pairs = pair_summary.sort_values(
        ["contextStableWinner", "modalWinnerShare", "meanDominanceProxyScore"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(5)
    pair_lines = "\n".join(
        f"- `{row.s06PairKey}`: modal winner `{row.modalWinnerPolicyId or 'unresolved'}` in "
        f"{row.modalWinnerShare:.2f} of contexts, dominance {row.meanDominanceProxyScore:.3f}, "
        f"{row.contextDependencyClass}"
        for row in top_pairs.itertuples(index=False)
    ) or "- No pair summaries available."
    top_policies = hierarchy.head(5)
    policy_lines = "\n".join(
        f"- `{row.panelPolicyId}`: weighted win score {row.weightedWinScore:.3f}, "
        f"resolved win rate {row.resolvedWinRate:.2f}, stable pair wins {int(row.stablePairWinCount)}"
        for row in top_policies.itertuples(index=False)
    ) or "- No policy hierarchy rows available."
    if paper_comparison.empty:
        paper_lines = "- No original paper-order comparisons were available."
    else:
        paper_lines = "\n".join(
            f"- `{row.paperComparisonPair}` expected `{row.expectedWinnerAlgotype}`: "
            f"resolved match rate {row.matchRateResolved:.2f}, unresolved {int(row.unresolvedCount)}/{int(row.conditionCount)}"
            for row in paper_comparison.itertuples(index=False)
        )
    text = f"""# Research Step S06: Map dominance hierarchies

## Completion status

Research step ID: `S06`. {status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Ran {status['conditionCount']} opposite-goal dominance contests across {status['selectedPairCount']} selected policy pairs. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

S06 winner calls are final-state computational proxies based on predeclared S05-compatible dominance, target-quality, and direction margins. Opposite goals use per-cell reverse-direction behavior while simulator resource caps remain generic; resource-capped rows are retained with caveat flags. Value-profile perturbations include reversed order and duplicate-value targets, so paper-order comparisons are contextual rather than exact reruns of E01.

## Lay summary

S06 tested selected pairs under opposing goals and asked which policy's target was better satisfied across ratios, placements, and value profiles. It maps stable winners, context-dependent contests, and how original Bubble, Selection, and Insertion comparisons line up with the earlier paper-order anchor.

## Pair dominance summary

{pair_lines}

## Policy hierarchy summary

{policy_lines}

## Paper-order comparison

{paper_lines}

## Recommended next action

{status['recommendedNextAction']}
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s06_dominance_hierarchy(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    panel_path: Path = DEFAULT_PANEL_PATH,
    s05_metrics_path: Path = DEFAULT_S05_METRICS_PATH,
    s05_boundary_caveats_path: Path = DEFAULT_S05_BOUNDARY_CAVEATS_PATH,
    e01_anchor_path: Path = DEFAULT_E01_OPPOSITE_SUMMARY_PATH,
    workers: int | None = None,
    n: int = 100,
    seed_count: int = 1,
    max_pairs: int = 8,
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
    s05_metrics = load_s05_metrics(s05_metrics_path)
    boundary_caveats = load_boundary_caveats(s05_boundary_caveats_path)
    e01_anchors = load_e01_paper_anchors(e01_anchor_path)
    criteria = predefined_winner_criteria(boundary_caveats)
    criteria_df = winner_criteria_table(criteria)
    selected_pairs = select_s06_pairs(s05_metrics, ready_panel, max_pairs=max_pairs)
    condition_df, condition_goal_df, policy_goal_df, arrangement_df = build_dominance_condition_matrix(
        selected_pairs,
        ready_panel,
        criteria,
        n=n,
        seed_count=seed_count,
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
    goal_result_df = augment_goal_results(raw_result_df)
    scored = score_dominance_results(goal_result_df, criteria)
    pair_summary = summarize_pair_dominance(scored, criteria)
    edges = build_dominance_edges(pair_summary)
    hierarchy = build_policy_hierarchy(scored, pair_summary, ready_panel)
    context_summary = summarize_contexts(scored)
    paper_comparison = build_paper_order_comparison(scored, e01_anchors)
    validation_df = validation_checks(
        condition_df,
        scored,
        ready_panel,
        criteria,
        criteria_df,
        policy_goal_df,
        pair_summary,
        paper_comparison,
        smoke_df,
    )

    criteria_json = step_dir / "dominance_winner_criteria.json"
    criteria_md = step_dir / "dominance_winner_criteria.md"
    criteria_csv = step_dir / "dominance_winner_criteria.csv"
    criteria_parquet = step_dir / "dominance_winner_criteria.parquet"
    condition_csv = step_dir / "dominance_condition_matrix.csv"
    condition_parquet = step_dir / "dominance_condition_matrix.parquet"
    selected_csv = step_dir / "selected_dominance_pairs.csv"
    selected_parquet = step_dir / "selected_dominance_pairs.parquet"
    condition_goal_csv = step_dir / "condition_goal_metadata.csv"
    condition_goal_parquet = step_dir / "condition_goal_metadata.parquet"
    policy_goal_csv = step_dir / "policy_goal_metadata.csv"
    policy_goal_parquet = step_dir / "policy_goal_metadata.parquet"
    arrangement_csv = step_dir / "arrangement_descriptors.csv"
    arrangement_parquet = step_dir / "arrangement_descriptors.parquet"
    step_runs_csv = step_dir / "dominance_runs.csv"
    step_runs_parquet = step_dir / "dominance_runs.parquet"
    result_contest_csv = results_dir / "e06_dominance_contests.csv"
    result_contest_parquet = results_dir / "e06_dominance_contests.parquet"
    result_mosaic_csv = results_dir / "e06_dominance_mosaics.csv"
    result_mosaic_parquet = results_dir / "e06_dominance_mosaics.parquet"
    hierarchy_csv = results_dir / "e06_dominance_hierarchy.csv"
    hierarchy_parquet = results_dir / "e06_dominance_hierarchy.parquet"
    step_hierarchy_csv = step_dir / "dominance_policy_hierarchy.csv"
    step_hierarchy_parquet = step_dir / "dominance_policy_hierarchy.parquet"
    pair_summary_csv = step_dir / "dominance_pair_summary.csv"
    pair_summary_parquet = step_dir / "dominance_pair_summary.parquet"
    edges_csv = step_dir / "dominance_network_edges.csv"
    edges_parquet = step_dir / "dominance_network_edges.parquet"
    context_csv = step_dir / "dominance_context_summary.csv"
    context_parquet = step_dir / "dominance_context_summary.parquet"
    paper_csv = step_dir / "paper_order_comparison.csv"
    paper_parquet = step_dir / "paper_order_comparison.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    smoke_csv = step_dir / "smoke_replay_checks.csv"
    smoke_parquet = step_dir / "smoke_replay_checks.parquet"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    write_json(criteria_json, criteria)
    write_winner_criteria_markdown(criteria, criteria_md)
    for path, df in [
        (criteria_csv, criteria_df),
        (condition_csv, condition_df),
        (selected_csv, selected_pairs),
        (condition_goal_csv, condition_goal_df),
        (policy_goal_csv, policy_goal_df),
        (arrangement_csv, arrangement_df),
        (step_runs_csv, compact_result_csv(scored)),
        (result_contest_csv, compact_result_csv(scored)),
        (result_mosaic_csv, compact_result_csv(scored)),
        (hierarchy_csv, hierarchy),
        (step_hierarchy_csv, hierarchy),
        (pair_summary_csv, pair_summary),
        (edges_csv, edges),
        (context_csv, context_summary),
        (paper_csv, paper_comparison),
        (validation_csv, validation_df),
        (smoke_csv, smoke_df),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (criteria_parquet, criteria_df),
        (condition_parquet, condition_df),
        (selected_parquet, selected_pairs),
        (condition_goal_parquet, condition_goal_df),
        (policy_goal_parquet, policy_goal_df),
        (arrangement_parquet, arrangement_df),
        (step_runs_parquet, scored),
        (result_contest_parquet, scored),
        (result_mosaic_parquet, scored),
        (hierarchy_parquet, hierarchy),
        (step_hierarchy_parquet, hierarchy),
        (pair_summary_parquet, pair_summary),
        (edges_parquet, edges),
        (context_parquet, context_summary),
        (paper_parquet, paper_comparison),
        (validation_parquet, validation_df),
        (smoke_parquet, smoke_df),
    ]:
        df.to_parquet(path, index=False)

    figure_paths = write_dominance_plots(edges, context_summary, figures_dir, step_dir)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() and scored["runSucceeded"].fillna(False).map(bool).all() else "failed"
    paper_match_rate = float(paper_comparison["matchRateResolved"].mean()) if not paper_comparison.empty else float("nan")
    stable_pair_count = int(pair_summary["contextStableWinner"].map(bool).sum()) if not pair_summary.empty else 0
    outcome = "supportive" if validation_result == "passed" and stable_pair_count > 0 else "constraining/contradictory"
    artifacts = [
        criteria_json,
        criteria_md,
        criteria_csv,
        criteria_parquet,
        condition_csv,
        condition_parquet,
        selected_csv,
        selected_parquet,
        condition_goal_csv,
        condition_goal_parquet,
        policy_goal_csv,
        policy_goal_parquet,
        arrangement_csv,
        arrangement_parquet,
        step_runs_csv,
        step_runs_parquet,
        result_contest_csv,
        result_contest_parquet,
        result_mosaic_csv,
        result_mosaic_parquet,
        hierarchy_csv,
        hierarchy_parquet,
        step_hierarchy_csv,
        step_hierarchy_parquet,
        pair_summary_csv,
        pair_summary_parquet,
        edges_csv,
        edges_parquet,
        context_csv,
        context_parquet,
        paper_csv,
        paper_parquet,
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
            "Winner calls are final-state computational proxies from S05-compatible dominance and target-quality scores.",
            "Opposite-goal rows use per-cell reverse-direction behavior, while simulator resource caps remain generic.",
            "Value-profile perturbations include reversed unique order and duplicate values; these are stress contexts, not exact paper reruns.",
            "Paper-order comparisons are contextual checks against E01 anchors, not required to match in every S06 context.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then proceed to S07 mosaic formation using S06 dominance-sensitive pairs, "
            "arrangements, value profiles, and unresolved or context-dependent contests."
        ),
        "laySummary": (
            "S06 mapped which policies win under opposing goals across selected ratios, placements, value profiles, "
            "and perturbation contexts, then summarized stable and context-dependent dominance edges."
        ),
        "outcomeClassification": outcome,
        "conditionCount": int(len(condition_df)),
        "resultCount": int(len(scored)),
        "selectedPairCount": int(len(selected_pairs)),
        "selectedPolicyCount": int(len({pid for text in condition_df["policyIdsJson"].astype(str) for pid in parse_json_maybe(text, [])})),
        "stablePairCount": stable_pair_count,
        "contextDependentPairCount": int((pair_summary["contextDependencyClass"] == "context_dependent_dominance_proxy").sum()) if not pair_summary.empty else 0,
        "paperOrderMeanResolvedMatchRate": paper_match_rate,
        "meanDominanceProxyScore": float(scored["dominanceProxyScore"].mean()),
        "meanFinalTargetQualityScore": float(scored["finalTargetQualityScore"].mean()),
        "unresolvedContextRate": float((scored["winnerPolicyId"].astype(str) == UNRESOLVED_WINNER).mean()),
        "workerCount": int(workers),
        "maxActivations": int(max_activations),
        "maxSwaps": int(max_swaps),
        "maxComparisons": int(max_comparisons),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
        "newDependenciesInstalled": [],
    }
    summary_path = write_markdown_report(
        step_dir=step_dir,
        status=status,
        artifacts_written=[str(path) for path in artifacts],
        pair_summary=pair_summary,
        hierarchy=hierarchy,
        paper_comparison=paper_comparison,
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
        "dominanceSweepVersion": DOMINANCE_SWEEP_VERSION,
        "winnerCriteriaVersion": WINNER_CRITERIA_VERSION,
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
        "criteria": criteria,
        "criteriaTable": criteria_df,
        "selectedPairs": selected_pairs,
        "conditions": condition_df,
        "conditionGoalMetadata": condition_goal_df,
        "policyGoalMetadata": policy_goal_df,
        "arrangementDescriptors": arrangement_df,
        "results": scored,
        "pairSummary": pair_summary,
        "edges": edges,
        "hierarchy": hierarchy,
        "contextSummary": context_summary,
        "paperComparison": paper_comparison,
        "validation": validation_df,
        "smoke": smoke_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E06 S06 opposite-goal dominance hierarchy sweeps.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--s05-metrics-path", type=Path, default=DEFAULT_S05_METRICS_PATH)
    parser.add_argument("--s05-boundary-caveats-path", type=Path, default=DEFAULT_S05_BOUNDARY_CAVEATS_PATH)
    parser.add_argument("--e01-anchor-path", type=Path, default=DEFAULT_E01_OPPOSITE_SUMMARY_PATH)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--max-activations", type=int, default=6000)
    parser.add_argument("--max-swaps", type=int, default=4000)
    parser.add_argument("--max-comparisons", type=int, default=30000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_s06_dominance_hierarchy(
        artifacts_dir=args.artifacts_dir,
        panel_path=args.panel_path,
        s05_metrics_path=args.s05_metrics_path,
        s05_boundary_caveats_path=args.s05_boundary_caveats_path,
        e01_anchor_path=args.e01_anchor_path,
        workers=args.workers,
        n=args.n,
        seed_count=args.seed_count,
        max_pairs=args.max_pairs,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['conditionCount']} dominance contests, "
        f"{status['selectedPairCount']} pairs, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
