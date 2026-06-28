"""E06 S03 initial spatial arrangement sweeps."""

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

from e02_deterministic_simulator.metrics import aggregation

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


STEP_ID = "S03"
STEP_NUMBER = 3
ARRANGEMENT_SWEEP_VERSION = "e06_s03_initial_arrangements.v1"
DEFAULT_S02_RESULTS_PATH = Path("/artifacts/results/e06_mixture_ratios.parquet")
ARRANGEMENT_TYPES: tuple[str, ...] = (
    "random",
    "contiguous_patch",
    "alternating",
    "clustered_island",
    "gradient",
    "graft_like",
)
PAIR_CATEGORY_ORDER: tuple[str, ...] = (
    "original_pair",
    "frontier_pair",
    "handoff_vs_frontier",
    "memory_signal_handoff_vs_original",
    "e03_vs_original",
    "control_pair",
)
PAIR_KEY_COLUMNS: tuple[str, ...] = (
    "pairCategory",
    "leftPanelPolicyId",
    "rightPanelPolicyId",
    "ratioLabel",
)


def load_s02_pair_results(path: Path = DEFAULT_S02_RESULTS_PATH) -> pd.DataFrame:
    results = pd.read_parquet(path)
    required = {
        "mixtureKind",
        "pairCategory",
        "leftPanelPolicyId",
        "rightPanelPolicyId",
        "ratioLabel",
        "policyIdsJson",
        "policyCountsJson",
        "targetCountsJson",
        "valueSeed",
        "arrangementSeed",
        "schedulerSeed",
        "tieBreakerSeed",
        "seedIndex",
        "runSucceeded",
        "actualRatiosMatchTarget",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"S02 results missing required columns: {missing}")
    pair = results[
        (results["mixtureKind"].astype(str) == "pair")
        & results["runSucceeded"].map(bool)
        & results["actualRatiosMatchTarget"].map(bool)
    ].copy()
    if pair.empty:
        raise ValueError("S02 results contain no usable pair rows")
    return pair.reset_index(drop=True)


def summarize_s02_pair_groups(s02_pair_results: pd.DataFrame) -> pd.DataFrame:
    df = s02_pair_results.copy()
    for column in [
        "completed",
        "finalSortednessPercent",
        "aggregationDelta",
        "finalAggregation",
        "activationCount",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    summary = (
        df.groupby(list(PAIR_KEY_COLUMNS), dropna=False)
        .agg(
            s02RunCount=("conditionId", "size"),
            s02CompletionRate=("completed", "mean"),
            s02MeanFinalSortednessPercent=("finalSortednessPercent", "mean"),
            s02MinFinalSortednessPercent=("finalSortednessPercent", "min"),
            s02MeanAggregationDelta=("aggregationDelta", "mean"),
            s02MeanFinalAggregation=("finalAggregation", "mean"),
            s02MeanActivationCount=("activationCount", "mean"),
        )
        .reset_index()
    )
    category_rank = {category: rank for rank, category in enumerate(PAIR_CATEGORY_ORDER)}
    summary["categoryRank"] = summary["pairCategory"].map(category_rank).fillna(99).astype(int)
    summary["absAggregationDelta"] = summary["s02MeanAggregationDelta"].abs()
    summary["s02HighContrastScore"] = (
        summary["s02CompletionRate"].fillna(0.0) * 100.0
        + summary["s02MeanFinalSortednessPercent"].fillna(0.0)
        + summary["absAggregationDelta"].fillna(0.0) * 50.0
    )
    return summary.sort_values(
        ["categoryRank", "s02HighContrastScore", "s02MeanFinalSortednessPercent"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def select_s03_base_groups(
    s02_pair_results: pd.DataFrame,
    *,
    max_groups_per_category: int = 3,
    max_total_groups: int = 18,
) -> pd.DataFrame:
    summary = summarize_s02_pair_groups(s02_pair_results)
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add_rows(rows: pd.DataFrame, reason: str) -> None:
        for row in rows.to_dict(orient="records"):
            key = tuple(row[column] for column in PAIR_KEY_COLUMNS)
            existing = selected.get(key)
            if existing is None:
                row["selectionReason"] = reason
                selected[key] = row
            elif reason not in str(existing["selectionReason"]).split("+"):
                existing["selectionReason"] = f"{existing['selectionReason']}+{reason}"

    for category in PAIR_CATEGORY_ORDER:
        cat = summary[summary["pairCategory"] == category]
        if cat.empty:
            continue
        add_rows(
            cat.sort_values(
                ["s02CompletionRate", "s02MeanFinalSortednessPercent", "s02HighContrastScore"],
                ascending=[False, False, False],
                kind="mergesort",
            ).head(1),
            "high_completion_sortedness",
        )
        add_rows(
            cat.sort_values(
                ["absAggregationDelta", "s02MeanFinalSortednessPercent"],
                ascending=[False, False],
                kind="mergesort",
            ).head(1),
            "high_aggregation_shift",
        )
        add_rows(
            cat.sort_values(
                ["s02MeanFinalSortednessPercent", "s02CompletionRate"],
                ascending=[True, True],
                kind="mergesort",
            ).head(1),
            "low_sortedness_stress",
        )
        category_selected = [
            value for value in selected.values() if value["pairCategory"] == category
        ]
        if len(category_selected) < max_groups_per_category:
            chosen_keys = {tuple(value[column] for column in PAIR_KEY_COLUMNS) for value in category_selected}
            remaining = cat[
                ~cat.apply(lambda row: tuple(row[column] for column in PAIR_KEY_COLUMNS) in chosen_keys, axis=1)
            ]
            add_rows(
                remaining.sort_values(
                    ["s02HighContrastScore", "s02MeanFinalSortednessPercent"],
                    ascending=[False, False],
                    kind="mergesort",
                ).head(max_groups_per_category - len(category_selected)),
                "category_fill",
            )

    selected_df = pd.DataFrame(selected.values())
    if selected_df.empty:
        raise ValueError("S03 selected no S02 base groups")
    selected_df = selected_df.sort_values(
        ["categoryRank", "s02HighContrastScore", "s02MeanFinalSortednessPercent"],
        ascending=[True, False, False],
        kind="mergesort",
    ).head(int(max_total_groups))
    selected_df = selected_df.reset_index(drop=True)
    selected_df.insert(0, "s03BaseGroupIndex", np.arange(len(selected_df), dtype=int))
    selected_df["s03Version"] = ARRANGEMENT_SWEEP_VERSION
    return selected_df


def interface_count(labels: Sequence[Any]) -> int:
    return int(sum(1 for left, right in zip(labels, labels[1:]) if left != right))


def run_lengths(labels: Sequence[Any]) -> list[int]:
    if not labels:
        return []
    lengths: list[int] = []
    current = labels[0]
    length = 1
    for label in labels[1:]:
        if label == current:
            length += 1
        else:
            lengths.append(length)
            current = label
            length = 1
    lengths.append(length)
    return lengths


def mean_positions(labels: Sequence[str], policy_ids: Sequence[str]) -> dict[str, float]:
    n = len(labels)
    positions = np.linspace(0.0, 1.0, num=n) if n > 1 else np.asarray([0.0])
    out: dict[str, float] = {}
    for pid in policy_ids:
        hits = [positions[index] for index, label in enumerate(labels) if label == pid]
        out[str(pid)] = float(np.mean(hits)) if hits else float("nan")
    return out


def _labels_to_ids(labels: Sequence[int], policy_ids: Sequence[str]) -> list[str]:
    return [str(policy_ids[int(label)]) for label in labels]


def _low_discrepancy_labels(counts: Sequence[int]) -> list[int]:
    slots: list[tuple[float, int, int]] = []
    for label, count in enumerate(counts):
        for index in range(int(count)):
            slots.append(((index + 0.5) / max(1, int(count)), label, index))
    return [label for _, label, _ in sorted(slots, key=lambda item: (item[0], item[1], item[2]))]


def _clustered_labels(counts: Sequence[int], n: int, seed: int) -> list[int]:
    if len(counts) != 2:
        return _low_discrepancy_labels(counts)
    rng = np.random.default_rng(int(seed))
    left_count, right_count = [int(value) for value in counts]
    cluster_label = 0 if left_count <= right_count else 1
    cluster_total = int(counts[cluster_label])
    background_label = 1 - cluster_label
    labels = [background_label] * int(n)
    cluster_count = max(1, min(3, cluster_total, max(1, cluster_total // 8 + 1)))
    base_sizes = [cluster_total // cluster_count] * cluster_count
    for index in range(cluster_total % cluster_count):
        base_sizes[index] += 1
    centers = np.linspace(0.20, 0.80, num=cluster_count)
    jitter = rng.uniform(-0.05, 0.05, size=cluster_count)
    centers = np.clip((centers + jitter) * (n - 1), 0, n - 1)
    occupied: set[int] = set()
    for center, size in zip(centers, base_sizes, strict=True):
        ordered_positions = sorted(range(n), key=lambda pos: (abs(pos - float(center)), pos))
        placed = 0
        for pos in ordered_positions:
            if pos in occupied:
                continue
            labels[pos] = cluster_label
            occupied.add(pos)
            placed += 1
            if placed >= size:
                break
    return labels


def _gradient_labels(counts: Sequence[int], n: int, seed: int) -> list[int]:
    if len(counts) != 2:
        return _low_discrepancy_labels(counts)
    rng = np.random.default_rng(int(seed))
    scores = np.linspace(0.0, 1.0, num=n) + rng.normal(0.0, 0.08, size=n)
    left_count = int(counts[0])
    left_positions = set(int(pos) for pos in np.argsort(scores, kind="mergesort")[:left_count])
    return [0 if pos in left_positions else 1 for pos in range(n)]


def _graft_labels(counts: Sequence[int], n: int, seed: int) -> list[int]:
    if len(counts) != 2:
        return _low_discrepancy_labels(counts)
    rng = np.random.default_rng(int(seed))
    left_count, right_count = [int(value) for value in counts]
    if left_count == right_count:
        graft_label = 1
    else:
        graft_label = 0 if left_count < right_count else 1
    graft_count = int(counts[graft_label])
    background_label = 1 - graft_label
    labels = [background_label] * int(n)
    if graft_count >= n:
        return [graft_label] * int(n)
    low = max(0, int(n * 0.20) - graft_count // 2)
    high = min(n - graft_count, int(n * 0.80) - graft_count // 2)
    if high < low:
        low, high = 0, n - graft_count
    start = int(rng.integers(low, high + 1)) if high >= low else 0
    for pos in range(start, start + graft_count):
        labels[pos] = graft_label
    return labels


def generate_arrangement(
    policy_ids: Sequence[str],
    counts: Sequence[int],
    arrangement_type: str,
    *,
    seed: int,
) -> tuple[list[str], list[int], dict[str, Any]]:
    if arrangement_type not in ARRANGEMENT_TYPES:
        raise ValueError(f"unknown arrangement type: {arrangement_type}")
    ids = [str(pid) for pid in policy_ids]
    counts = [int(count) for count in counts]
    n = int(sum(counts))
    if n <= 0:
        raise ValueError("arrangement counts must sum to a positive n")
    if len(ids) != len(counts):
        raise ValueError("policy_ids and counts length mismatch")

    base_labels: list[int] = []
    for label, count in enumerate(counts):
        base_labels.extend([label] * int(count))

    if arrangement_type == "random":
        rng = np.random.default_rng(int(seed))
        labels = list(rng.permutation(np.asarray(base_labels, dtype=np.int16)).astype(int))
    elif arrangement_type == "contiguous_patch":
        labels = list(base_labels)
    elif arrangement_type == "alternating":
        labels = _low_discrepancy_labels(counts)
    elif arrangement_type == "clustered_island":
        labels = _clustered_labels(counts, n, seed)
    elif arrangement_type == "gradient":
        labels = _gradient_labels(counts, n, seed)
    elif arrangement_type == "graft_like":
        labels = _graft_labels(counts, n, seed)
    else:  # pragma: no cover - guarded above
        raise ValueError(f"unknown arrangement type: {arrangement_type}")

    if Counter(labels) != Counter(base_labels):
        raise ValueError(f"arrangement {arrangement_type} did not preserve counts")
    assigned = _labels_to_ids(labels, ids)
    lengths = run_lengths(labels)
    descriptor = {
        "arrangementType": arrangement_type,
        "n": n,
        "policyIds": ids,
        "policyCounts": dict(zip(ids, counts, strict=True)),
        "interfaceCount": interface_count(labels),
        "interfaceDensity": interface_count(labels) / max(1, n - 1),
        "runCount": len(lengths),
        "longestRun": max(lengths) if lengths else 0,
        "meanRunLength": float(np.mean(lengths)) if lengths else 0.0,
        "initialAggregation": aggregation(assigned),
        "policyMeanPositions": mean_positions(assigned, ids),
        "arrangementHash": sha256_text(compact_json(assigned)),
    }
    return assigned, labels, descriptor


def build_arrangement_condition_matrix(
    s02_pair_results: pd.DataFrame,
    ready_panel: pd.DataFrame,
    *,
    max_groups_per_category: int = 3,
    max_total_groups: int = 18,
    arrangement_types: Sequence[str] = ARRANGEMENT_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_groups = select_s03_base_groups(
        s02_pair_results,
        max_groups_per_category=max_groups_per_category,
        max_total_groups=max_total_groups,
    )
    selected_keys = selected_groups[list(PAIR_KEY_COLUMNS) + ["s03BaseGroupIndex", "selectionReason"]]
    base_rows = s02_pair_results.merge(selected_keys, on=list(PAIR_KEY_COLUMNS), how="inner")
    base_rows = base_rows.sort_values(
        ["s03BaseGroupIndex", "seedIndex", "conditionId"],
        kind="mergesort",
    ).reset_index(drop=True)
    ready_ids = set(ready_panel["panelPolicyId"].astype(str))
    rows: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    arrangement_offsets = {name: index for index, name in enumerate(arrangement_types)}

    for base in base_rows.to_dict(orient="records"):
        policy_ids = [str(item) for item in parse_json_maybe(base["policyIdsJson"], [])]
        counts = [int(item) for item in parse_json_maybe(base["policyCountsJson"], [])]
        if set(policy_ids) - ready_ids:
            raise ValueError("S03 base row includes policy not marked ready")
        base_match_payload = {
            "s02ConditionId": base["conditionId"],
            "policyIds": policy_ids,
            "counts": counts,
            "ratioLabel": base["ratioLabel"],
            "seedIndex": int(base["seedIndex"]),
        }
        base_match_id = sha256_text(compact_json(base_match_payload))[:16]
        for arrangement_type in arrangement_types:
            arrangement_seed = int(base["arrangementSeed"])
            assigned, numeric_labels, descriptor = generate_arrangement(
                policy_ids,
                counts,
                arrangement_type,
                seed=arrangement_seed,
            )
            condition_payload = {
                "baseMatchId": base_match_id,
                "arrangementType": arrangement_type,
                "arrangementHash": descriptor["arrangementHash"],
                "s03Version": ARRANGEMENT_SWEEP_VERSION,
            }
            condition_id = (
                f"s03_g{int(base['s03BaseGroupIndex']):03d}_"
                f"{str(base['ratioLabel']).replace(':', '_')}_seed{int(base['seedIndex'])}_"
                f"{arrangement_type}"
            )
            row = {
                "conditionId": condition_id,
                "researchStepId": STEP_ID,
                "implementationPrefix": "e06_s03",
                "s03Version": ARRANGEMENT_SWEEP_VERSION,
                "s02SourceConditionId": str(base["conditionId"]),
                "s02SourceConditionHash": str(base.get("conditionHash", "")),
                "s03BaseGroupIndex": int(base["s03BaseGroupIndex"]),
                "baseMatchId": base_match_id,
                "selectionReason": str(base["selectionReason"]),
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
                "policyCountsJson": compact_json(counts),
                "arrangementType": arrangement_type,
                "arrangementTypeIndex": int(arrangement_offsets[arrangement_type]),
                "arrangementPolicyIdsJson": compact_json(assigned),
                "arrangementNumericLabelsJson": compact_json(numeric_labels),
                "arrangementDescriptorJson": compact_json(descriptor),
                "arrangementHash": descriptor["arrangementHash"],
                "conditionHash": sha256_text(compact_json(condition_payload))[:16],
                "seedIndex": int(base["seedIndex"]),
                "valueSeed": int(base["valueSeed"]),
                "arrangementSeed": arrangement_seed,
                "schedulerSeed": int(base["schedulerSeed"]),
                "tieBreakerSeed": int(base["tieBreakerSeed"]),
                "goalMode": "same_goal_increasing",
                "s02FinalSortednessPercent": float(base["finalSortednessPercent"]),
                "s02Completed": bool(base["completed"]),
                "s02FinalAggregation": float(base["finalAggregation"]),
                "s02AggregationDelta": float(base["aggregationDelta"]),
            }
            rows.append(row)
            descriptors.append(
                {
                    "conditionId": condition_id,
                    "baseMatchId": base_match_id,
                    "arrangementType": arrangement_type,
                    "arrangementSeed": arrangement_seed,
                    "policyIdsJson": compact_json(policy_ids),
                    "policyCountsJson": compact_json(counts),
                    "n": int(descriptor["n"]),
                    "interfaceCount": int(descriptor["interfaceCount"]),
                    "interfaceDensity": float(descriptor["interfaceDensity"]),
                    "runCount": int(descriptor["runCount"]),
                    "longestRun": int(descriptor["longestRun"]),
                    "meanRunLength": float(descriptor["meanRunLength"]),
                    "initialAggregation": float(descriptor["initialAggregation"]),
                    "arrangementHash": str(descriptor["arrangementHash"]),
                    "arrangementDescriptorJson": compact_json(descriptor),
                }
            )

    condition_df = pd.DataFrame(rows)
    descriptor_df = pd.DataFrame(descriptors)
    return condition_df, selected_groups, descriptor_df


def _position_shift(initial_ids: Sequence[str], final_ids: Sequence[str], policy_ids: Sequence[str]) -> float:
    initial_means = mean_positions(initial_ids, policy_ids)
    final_means = mean_positions(final_ids, policy_ids)
    shifts = [
        abs(float(final_means[pid]) - float(initial_means[pid]))
        for pid in policy_ids
        if np.isfinite(initial_means.get(pid, np.nan)) and np.isfinite(final_means.get(pid, np.nan))
    ]
    return float(np.mean(shifts)) if shifts else float("nan")


def classify_mosaic(final_ids: Sequence[str], completed: bool) -> str:
    final_agg = aggregation(final_ids)
    final_interfaces = interface_count(final_ids)
    density = final_interfaces / max(1, len(final_ids) - 1)
    if completed:
        return "sorted_value_order"
    if final_agg >= 0.95 and final_interfaces <= 2:
        return "segregated_low_interface"
    if final_agg >= 0.80:
        return "patchy_mosaic"
    if density >= 0.45:
        return "intermixed_high_interface"
    return "partial_mosaic"


def augment_arrangement_results(result_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in result_df.to_dict(orient="records"):
        initial_ids = [str(item) for item in parse_json_maybe(row.get("initialPanelPolicyIdsJson", ""), [])]
        final_ids = [str(item) for item in parse_json_maybe(row.get("finalPanelPolicyIdsJson", ""), [])]
        policy_ids = [str(item) for item in parse_json_maybe(row.get("policyIdsJson", ""), [])]
        initial_interfaces = interface_count(initial_ids)
        final_interfaces = interface_count(final_ids)
        n = len(initial_ids)
        changed_fraction = (
            float(sum(1 for left, right in zip(initial_ids, final_ids) if left != right) / max(1, n))
            if final_ids
            else float("nan")
        )
        interface_stability = 1.0 - min(1.0, abs(final_interfaces - initial_interfaces) / max(1, n - 1))
        row.update(
            {
                "initialInterfaceCount": int(initial_interfaces),
                "finalInterfaceCount": int(final_interfaces),
                "interfaceCountDelta": int(final_interfaces - initial_interfaces),
                "finalInterfaceDensity": float(final_interfaces / max(1, n - 1)),
                "interfaceStabilityScore": float(interface_stability),
                "policyPositionChangeFraction": changed_fraction,
                "meanPolicyPositionShift": _position_shift(initial_ids, final_ids, policy_ids),
                "finalMosaicClass": classify_mosaic(final_ids, bool(row.get("completed", False))) if final_ids else "run_failed",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_arrangement_results(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["pairCategory", "ratioLabel", "arrangementType"]
    arrangement_summary = (
        result_df.groupby(group_cols, dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            completionRate=("completed", "mean"),
            meanFinalSortednessPercent=("finalSortednessPercent", "mean"),
            meanSortednessGain=("sortednessGain", "mean"),
            meanInitialAggregation=("initialAggregation", "mean"),
            meanFinalAggregation=("finalAggregation", "mean"),
            meanAggregationDelta=("aggregationDelta", "mean"),
            meanInitialInterfaceCount=("initialInterfaceCount", "mean"),
            meanFinalInterfaceCount=("finalInterfaceCount", "mean"),
            meanInterfaceCountDelta=("interfaceCountDelta", "mean"),
            meanInterfaceStabilityScore=("interfaceStabilityScore", "mean"),
            meanPolicyPositionChangeFraction=("policyPositionChangeFraction", "mean"),
            meanMeanPolicyPositionShift=("meanPolicyPositionShift", "mean"),
            meanActivationCount=("activationCount", "mean"),
            meanRuntimeSeconds=("runtimeSeconds", "mean"),
        )
        .reset_index()
    )
    random_baseline = result_df[result_df["arrangementType"] == "random"][
        [
            "baseMatchId",
            "finalSortednessPercent",
            "completed",
            "finalAggregation",
            "finalInterfaceCount",
            "policyPositionChangeFraction",
        ]
    ].rename(
        columns={
            "finalSortednessPercent": "randomFinalSortednessPercent",
            "completed": "randomCompleted",
            "finalAggregation": "randomFinalAggregation",
            "finalInterfaceCount": "randomFinalInterfaceCount",
            "policyPositionChangeFraction": "randomPolicyPositionChangeFraction",
        }
    )
    deltas = result_df.merge(random_baseline, on="baseMatchId", how="left")
    deltas["deltaVsRandomFinalSortednessPercent"] = (
        deltas["finalSortednessPercent"] - deltas["randomFinalSortednessPercent"]
    )
    deltas["deltaVsRandomCompleted"] = deltas["completed"].astype(float) - deltas["randomCompleted"].astype(float)
    deltas["deltaVsRandomFinalAggregation"] = deltas["finalAggregation"] - deltas["randomFinalAggregation"]
    deltas["deltaVsRandomFinalInterfaceCount"] = deltas["finalInterfaceCount"] - deltas["randomFinalInterfaceCount"]
    deltas["deltaVsRandomPolicyPositionChangeFraction"] = (
        deltas["policyPositionChangeFraction"] - deltas["randomPolicyPositionChangeFraction"]
    )
    return arrangement_summary, deltas


def validation_checks(
    condition_df: pd.DataFrame,
    result_df: pd.DataFrame,
    ready_panel: pd.DataFrame,
    descriptor_df: pd.DataFrame,
    smoke_df: pd.DataFrame,
    *,
    arrangement_types: Sequence[str] = ARRANGEMENT_TYPES,
) -> pd.DataFrame:
    ready_ids = set(ready_panel["panelPolicyId"].astype(str))
    condition_ids = {
        pid for text in condition_df["policyIdsJson"].astype(str) for pid in parse_json_maybe(text, [])
    }
    required_arrangements = set(arrangement_types)
    observed_by_base = condition_df.groupby("baseMatchId")["arrangementType"].agg(lambda values: set(values))
    counts_match = []
    for row in condition_df.to_dict(orient="records"):
        assigned = [str(item) for item in parse_json_maybe(row["arrangementPolicyIdsJson"], [])]
        target = parse_json_maybe(row["targetCountsJson"], {})
        counts_match.append(dict(Counter(assigned)) == dict(target))
    matched_seed_ok = True
    for _, group in condition_df.groupby("baseMatchId"):
        for col in ["policyIdsJson", "policyCountsJson", "valueSeed", "schedulerSeed", "tieBreakerSeed", "goalMode"]:
            if group[col].nunique(dropna=False) != 1:
                matched_seed_ok = False
    checks = [
        {
            "checkId": "only_ready_policies",
            "success": bool(condition_ids <= ready_ids),
            "detail": f"{len(condition_ids)} S03 policies all came from readyForS02Mixing=true rows",
        },
        {
            "checkId": "all_required_arrangements_present",
            "success": bool(not observed_by_base.empty and observed_by_base.map(lambda values: required_arrangements <= values).all()),
            "detail": f"each base match includes {len(required_arrangements)} requested arrangements",
        },
        {
            "checkId": "target_ratios_match_arrangements",
            "success": bool(all(counts_match)),
            "detail": "every explicit arrangement assignment matches targetCountsJson",
        },
        {
            "checkId": "matched_random_controls_hold_ratio_goals_values",
            "success": bool(matched_seed_ok),
            "detail": "within each base match, policies, counts, value seed, scheduler seed, tie-breaker seed, and goal mode are constant",
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
            "detail": "all S03 runs succeeded and conserved values/policy labels",
        },
        {
            "checkId": "arrangement_descriptors_saved",
            "success": bool(
                len(descriptor_df) == len(condition_df)
                and {"arrangementType", "interfaceCount", "initialAggregation", "arrangementHash"} <= set(descriptor_df.columns)
            ),
            "detail": f"{len(descriptor_df)} arrangement descriptors written",
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
                "arrangementType": condition["arrangementType"],
                "replayMatch": bool(match),
                "firstFinalStateHash": first.get("finalStateHash"),
                "secondFinalStateHash": second.get("finalStateHash"),
                "firstStopReason": first.get("stopReason"),
                "secondStopReason": second.get("stopReason"),
                "checkedKeysJson": compact_json(keys),
            }
        )
    return pd.DataFrame(rows)


def write_arrangement_plot(arrangement_summary: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    if arrangement_summary.empty:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = list(ARRANGEMENT_TYPES)
    plot_df = arrangement_summary.copy()
    plot_df["arrangementType"] = pd.Categorical(plot_df["arrangementType"], categories=order, ordered=True)
    plot_df = plot_df.sort_values(["pairCategory", "arrangementType"], kind="mergesort")
    fig, ax = plt.subplots(figsize=(11, 6))
    for category, group in plot_df.groupby("pairCategory", observed=False):
        if group.empty:
            continue
        by_type = group.groupby("arrangementType", observed=False)["meanFinalSortednessPercent"].mean().reset_index()
        ax.plot(by_type["arrangementType"].astype(str), by_type["meanFinalSortednessPercent"], marker="o", linewidth=1.5, label=str(category))
    ax.set_xlabel("Initial arrangement")
    ax.set_ylabel("Mean final Sortedness (%)")
    ax.set_title("E06 S03 initial-arrangement sweep by S02 priority category")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    paths = [
        figure_dir / "e06_s03_arrangement_sortedness_by_category.png",
        figure_dir / "e06_s03_arrangement_sortedness_by_category.pdf",
        step_dir / "arrangement_sortedness_by_category.png",
        step_dir / "arrangement_sortedness_by_category.pdf",
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
    arrangement_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Path:
    top_delta = delta_summary[delta_summary["arrangementType"] != "random"].sort_values(
        ["deltaVsRandomFinalSortednessPercent", "deltaVsRandomFinalAggregation"],
        ascending=[False, False],
        kind="mergesort",
    ).head(5)
    top_lines = "\n".join(
        f"- `{row.arrangementType}` in {row.pairCategory} `{row.ratioLabel}`: "
        f"mean Sortedness delta vs random {row.deltaVsRandomFinalSortednessPercent:.2f} points"
        for row in top_delta.itertuples(index=False)
    ) or "- No non-random arrangement deltas available."
    arrangement_lines = "\n".join(
        f"- `{row.arrangementType}`: {int(row.conditionCount)} conditions, completion {row.completionRate:.2f}, "
        f"mean Sortedness {row.meanFinalSortednessPercent:.1f}%"
        for row in arrangement_summary.groupby("arrangementType", dropna=False)
        .agg(
            conditionCount=("conditionCount", "sum"),
            completionRate=("completionRate", "mean"),
            meanFinalSortednessPercent=("meanFinalSortednessPercent", "mean"),
        )
        .reset_index()
        .sort_values("arrangementType")
        .itertuples(index=False)
    ) or "- No arrangement summaries available."
    text = f"""# Research Step S03: Vary initial spatial arrangement

## Completion status

{status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Ran {status['conditionCount']} matched S03 arrangement conditions from {status['selectedBaseGroupCount']} S02 high-contrast pair-ratio groups. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

The S03 sweep is a bounded follow-up over selected high-contrast S02 pair-ratio groups, not the full S02 condition matrix. All conditions hold same-goal increasing sorting constant; goal compatibility and opposite-direction conflict remain deferred to S04 and later steps. Interface stability, boundary movement, and mosaic labels are computational proxy summaries from one-dimensional policy-label positions.

## Lay summary

S03 asks whether the same policies and ratios behave differently when their initial cell identities start mixed randomly, as patches, alternating, clustered islands, gradients, or graft-like inserts. The result is an arrangement sensitivity map for choosing S04 goal-compatibility tests.

## Arrangement summary

{arrangement_lines}

## Anchor results

{top_lines}

## Recommended next action

Chief Scientist review, then S04 should vary goal compatibility using S03-sensitive pair/ratio/arrangement combinations from `$ARTIFACTS_DIR/results/e06_initial_arrangements.parquet`.
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s03_initial_arrangements(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    panel_path: Path = DEFAULT_PANEL_PATH,
    s02_results_path: Path = DEFAULT_S02_RESULTS_PATH,
    workers: int | None = None,
    max_groups_per_category: int = 3,
    max_total_groups: int = 18,
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
    s02_pair_results = load_s02_pair_results(s02_results_path)
    condition_df, selected_groups, descriptor_df = build_arrangement_condition_matrix(
        s02_pair_results,
        ready_panel,
        max_groups_per_category=max_groups_per_category,
        max_total_groups=max_total_groups,
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
    result_df = augment_arrangement_results(raw_result_df)
    arrangement_summary, matched_deltas = summarize_arrangement_results(result_df)
    delta_summary = (
        matched_deltas.groupby(["pairCategory", "ratioLabel", "arrangementType"], dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            deltaVsRandomFinalSortednessPercent=("deltaVsRandomFinalSortednessPercent", "mean"),
            deltaVsRandomCompleted=("deltaVsRandomCompleted", "mean"),
            deltaVsRandomFinalAggregation=("deltaVsRandomFinalAggregation", "mean"),
            deltaVsRandomFinalInterfaceCount=("deltaVsRandomFinalInterfaceCount", "mean"),
            deltaVsRandomPolicyPositionChangeFraction=("deltaVsRandomPolicyPositionChangeFraction", "mean"),
        )
        .reset_index()
    )
    validation_df = validation_checks(condition_df, result_df, ready_panel, descriptor_df, smoke_df)

    condition_csv = step_dir / "arrangement_condition_matrix.csv"
    condition_parquet = step_dir / "arrangement_condition_matrix.parquet"
    descriptor_csv = step_dir / "arrangement_descriptors.csv"
    descriptor_parquet = step_dir / "arrangement_descriptors.parquet"
    selected_csv = step_dir / "selected_s02_high_contrast_groups.csv"
    selected_parquet = step_dir / "selected_s02_high_contrast_groups.parquet"
    step_result_csv = step_dir / "initial_arrangement_runs.csv"
    step_result_parquet = step_dir / "initial_arrangement_runs.parquet"
    result_csv = results_dir / "e06_initial_arrangements.csv"
    result_parquet = results_dir / "e06_initial_arrangements.parquet"
    sweep_csv = results_dir / "e06_mixture_sweeps.csv"
    sweep_parquet = results_dir / "e06_mixture_sweeps.parquet"
    arrangement_summary_csv = step_dir / "arrangement_summary.csv"
    arrangement_summary_parquet = step_dir / "arrangement_summary.parquet"
    matched_deltas_csv = step_dir / "matched_random_deltas.csv"
    matched_deltas_parquet = step_dir / "matched_random_deltas.parquet"
    delta_summary_csv = step_dir / "matched_random_delta_summary.csv"
    delta_summary_parquet = step_dir / "matched_random_delta_summary.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    smoke_csv = step_dir / "smoke_replay_checks.csv"
    smoke_parquet = step_dir / "smoke_replay_checks.parquet"

    result_csv_df = compact_result_csv(result_df)
    for path, df in [
        (condition_csv, condition_df),
        (descriptor_csv, descriptor_df),
        (selected_csv, selected_groups),
        (step_result_csv, result_csv_df),
        (result_csv, result_csv_df),
        (arrangement_summary_csv, arrangement_summary),
        (matched_deltas_csv, compact_result_csv(matched_deltas)),
        (delta_summary_csv, delta_summary),
        (validation_csv, validation_df),
        (smoke_csv, smoke_df),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (condition_parquet, condition_df),
        (descriptor_parquet, descriptor_df),
        (selected_parquet, selected_groups),
        (step_result_parquet, result_df),
        (result_parquet, result_df),
        (arrangement_summary_parquet, arrangement_summary),
        (matched_deltas_parquet, matched_deltas),
        (delta_summary_parquet, delta_summary),
        (validation_parquet, validation_df),
        (smoke_parquet, smoke_df),
    ]:
        df.to_parquet(path, index=False)

    s02_for_cumulative = pd.read_parquet(s02_results_path).copy()
    if "researchStepId" not in s02_for_cumulative.columns:
        s02_for_cumulative["researchStepId"] = "S02"
    cumulative = pd.concat([s02_for_cumulative, result_df], ignore_index=True, sort=False)
    compact_result_csv(cumulative).to_csv(sweep_csv, index=False)
    cumulative.to_parquet(sweep_parquet, index=False)

    figure_paths = write_arrangement_plot(arrangement_summary, figures_dir, step_dir)
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() and result_df["runSucceeded"].fillna(False).map(bool).all() else "failed"
    artifacts = [
        condition_csv,
        condition_parquet,
        descriptor_csv,
        descriptor_parquet,
        selected_csv,
        selected_parquet,
        step_result_csv,
        step_result_parquet,
        result_csv,
        result_parquet,
        sweep_csv,
        sweep_parquet,
        arrangement_summary_csv,
        arrangement_summary_parquet,
        matched_deltas_csv,
        matched_deltas_parquet,
        delta_summary_csv,
        delta_summary_parquet,
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
            "S03 used selected high-contrast S02 pair-ratio groups rather than the full S02 matrix.",
            "All rows use same-goal increasing sorting; goal compatibility is deferred to S04.",
            "Interface stability, boundary movement, and mosaic classes are one-dimensional computational proxies.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then run S04 goal-compatibility sweeps using S03-sensitive pair/ratio/"
            "arrangement combinations."
        ),
        "laySummary": (
            "S03 compared matched random, patch, alternating, clustered, gradient, and graft-like initial "
            "policy-label layouts while holding ratio, goals, values, and scheduler seeds fixed within each match."
        ),
        "outcomeClassification": "supportive" if validation_result == "passed" else "constraining/contradictory",
        "conditionCount": int(len(condition_df)),
        "resultCount": int(len(result_df)),
        "selectedBaseGroupCount": int(selected_groups.shape[0]),
        "selectedPolicyCount": int(len({pid for text in condition_df["policyIdsJson"].astype(str) for pid in parse_json_maybe(text, [])})),
        "arrangementTypes": list(ARRANGEMENT_TYPES),
        "completionRate": float(result_df["completed"].mean()),
        "meanFinalSortednessPercent": float(result_df["finalSortednessPercent"].mean()),
        "meanInterfaceStabilityScore": float(result_df["interfaceStabilityScore"].mean()),
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
        arrangement_summary=arrangement_summary,
        delta_summary=delta_summary,
        validation_df=validation_df,
    )
    if summary_path not in artifacts:
        artifacts.append(summary_path)
    write_json(status_path, status)
    manifest_payload = {
        "schema": "eidosoma.step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "arrangementSweepVersion": ARRANGEMENT_SWEEP_VERSION,
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
        "results": result_df,
        "selectedGroups": selected_groups,
        "descriptors": descriptor_df,
        "arrangementSummary": arrangement_summary,
        "matchedDeltas": matched_deltas,
        "deltaSummary": delta_summary,
        "validation": validation_df,
        "smoke": smoke_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E06 S03 initial spatial arrangement sweeps.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--s02-results-path", type=Path, default=DEFAULT_S02_RESULTS_PATH)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-groups-per-category", type=int, default=3)
    parser.add_argument("--max-total-groups", type=int, default=18)
    parser.add_argument("--max-activations", type=int, default=6000)
    parser.add_argument("--max-swaps", type=int, default=4000)
    parser.add_argument("--max-comparisons", type=int, default=30000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_s03_initial_arrangements(
        artifacts_dir=args.artifacts_dir,
        panel_path=args.panel_path,
        s02_results_path=args.s02_results_path,
        workers=args.workers,
        max_groups_per_category=args.max_groups_per_category,
        max_total_groups=args.max_total_groups,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['conditionCount']} conditions, "
        f"{status['selectedBaseGroupCount']} S02 base groups, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
