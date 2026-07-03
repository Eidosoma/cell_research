"""E06 S03 spatial-arrangement sweeps for selected chimeric mixtures."""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.coarse_sweep import actor_schedule, initial_values, sortedness_metrics
from src.e06.mixture_ratios import (
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    MixtureRuntime,
    aggregation_metrics,
    build_initial_policy_ids,
    classify_final_state,
    counts_for_ratios,
    dominance_metrics,
    stable_hash,
)


STEP_ID = "S03"
SPATIAL_SCHEMA = "eidosoma.e06.s03_spatial_arrangement_run.v1"
SUMMARY_SCHEMA = "eidosoma.e06.s03_spatial_arrangement_summary.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s03_validation.v1"
DEFAULT_ARRANGEMENTS = (
    "random_permutation",
    "contiguous_patch",
    "alternating",
    "clustered_islands",
    "gradient",
    "graft_like_insertions",
)


@dataclass(frozen=True)
class S03Config:
    """Configuration for the bounded S03 arrangement sweep."""

    array_size: int = 100
    event_cap: int = 4_000
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    arrangements: tuple[str, ...] = DEFAULT_ARRANGEMENTS
    max_candidate_pairs: int = 9
    top_sortedness_pairs: int = 6
    top_aggregation_pairs: int = 4
    scheduler: str = "cyclic_scan_seed_offset"


@dataclass(frozen=True)
class ArrangementCondition:
    """One S03 condition before expansion over seeds."""

    condition_id: str
    s02_candidate_id: str
    s02_candidate_rank: int
    panel: str
    policy_ids: tuple[str, ...]
    display_names: tuple[str, ...]
    ratio_targets: tuple[float, ...]
    arrangement: str
    candidate_reason: str
    s02_threshold_between_ratios: tuple[float, ...]


def _json_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _condition_id(policy_ids: Sequence[str], ratios: Sequence[float], arrangement: str) -> str:
    payload = {
        "step": STEP_ID,
        "policyIds": list(policy_ids),
        "ratios": [round(float(value), 6) for value in ratios],
        "arrangement": arrangement,
    }
    return f"s03_{stable_hash(payload)[:16]}"


def _candidate_id(policy_ids: Sequence[str]) -> str:
    return f"s03_pair_{stable_hash({'policyIds': list(policy_ids)})[:14]}"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def select_s03_candidates(threshold_candidates: pd.DataFrame, config: S03Config) -> pd.DataFrame:
    """Select a bounded pair panel from S02 threshold and aggregation candidates."""

    if threshold_candidates.empty:
        raise ValueError("S02 threshold candidate table is empty")
    required = {
        "policy_ids_json",
        "display_names_json",
        "panel",
        "sortedness_range",
        "aggregation_delta_range",
        "largest_sortedness_jump",
        "candidate_threshold_between_ratios_json",
        "minority_effect_candidate",
        "aggregation_sensitive_candidate",
    }
    missing = required - set(threshold_candidates.columns)
    if missing:
        raise ValueError(f"S02 threshold candidate table is missing columns: {sorted(missing)}")

    candidates = threshold_candidates.copy()
    for column in ("sortedness_range", "aggregation_delta_range", "largest_sortedness_jump"):
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(0.0)
    candidates["minority_effect_candidate"] = candidates["minority_effect_candidate"].map(_truthy)
    candidates["aggregation_sensitive_candidate"] = candidates["aggregation_sensitive_candidate"].map(_truthy)
    candidates["memory_candidate"] = candidates.apply(
        lambda row: "memory" in str(row["panel"]).lower() or "memory" in str(row["display_names_json"]).lower(),
        axis=1,
    )

    selected_indices: list[int] = []

    def add(indexes: Sequence[int]) -> None:
        for idx in indexes:
            if idx not in selected_indices and len(selected_indices) < config.max_candidate_pairs:
                selected_indices.append(int(idx))

    jump_ranked = candidates.sort_values(["largest_sortedness_jump", "sortedness_range"], ascending=[False, False], kind="mergesort")
    add(jump_ranked.head(config.top_sortedness_pairs).index.tolist())
    agg_ranked = candidates[candidates["aggregation_sensitive_candidate"]].sort_values(
        ["aggregation_delta_range", "largest_sortedness_jump"],
        ascending=[False, False],
        kind="mergesort",
    )
    add(agg_ranked.head(config.top_aggregation_pairs).index.tolist())
    memory_ranked = candidates[candidates["memory_candidate"]].sort_values(
        ["aggregation_delta_range", "largest_sortedness_jump"],
        ascending=[False, False],
        kind="mergesort",
    )
    add(memory_ranked.head(1).index.tolist())
    add(jump_ranked.index.tolist())

    selected = candidates.loc[selected_indices].copy()
    selected["s02_candidate_rank"] = np.arange(1, len(selected) + 1)
    selected["s02_candidate_id"] = selected["policy_ids_json"].map(lambda text: _candidate_id(tuple(json.loads(text))))

    reasons: list[str] = []
    for _, row in selected.iterrows():
        tags = []
        if bool(row["minority_effect_candidate"]):
            tags.append("minority_effect")
        if bool(row["aggregation_sensitive_candidate"]):
            tags.append("aggregation_sensitive")
        if bool(row["memory_candidate"]):
            tags.append("memory_repair")
        if not tags:
            tags.append("threshold_jump")
        reasons.append(",".join(tags))
    selected["candidate_reason"] = reasons
    return selected.reset_index(drop=True)


def _repeated_labels(policy_ids: Sequence[str], counts: Sequence[int]) -> list[str]:
    labels: list[str] = []
    for policy_id, count in zip(policy_ids, counts, strict=True):
        labels.extend([str(policy_id)] * int(count))
    return labels


def _alternating_labels(policy_ids: Sequence[str], counts: Sequence[int]) -> list[str]:
    remaining = [int(count) for count in counts]
    scores = [0 for _ in counts]
    total = sum(remaining)
    labels: list[str] = []
    while len(labels) < total:
        for idx, count in enumerate(counts):
            if remaining[idx] > 0:
                scores[idx] += int(count)
        choice = max((idx for idx in range(len(counts)) if remaining[idx] > 0), key=lambda idx: (scores[idx], -idx))
        labels.append(str(policy_ids[choice]))
        scores[choice] -= total
        remaining[choice] -= 1
    return labels


def _fill_nearest_empty(labels: list[str | None], center: int, policy_id: str, count: int) -> None:
    placed = 0
    radius = 0
    n = len(labels)
    while placed < count and radius <= n:
        for pos in (center - radius, center + radius):
            if placed >= count:
                break
            if 0 <= pos < n and labels[pos] is None:
                labels[pos] = policy_id
                placed += 1
        radius += 1
    if placed != count:
        raise RuntimeError("could not place requested label count")


def _clustered_islands(policy_ids: Sequence[str], counts: Sequence[int]) -> list[str]:
    n = sum(counts)
    host_idx = max(range(len(counts)), key=lambda idx: (counts[idx], idx))
    labels: list[str | None] = [None] * n
    for idx, (policy_id, count) in enumerate(zip(policy_ids, counts, strict=True)):
        if idx == host_idx:
            continue
        islands = min(4, max(1, int(round(math.sqrt(max(1, count)) / 2))))
        base = count // islands
        remainder = count % islands
        for island_idx in range(islands):
            size = base + (1 if island_idx < remainder else 0)
            center = int(round((island_idx + 1) * n / (islands + 1)))
            _fill_nearest_empty(labels, center, str(policy_id), size)
    host_id = str(policy_ids[host_idx])
    return [label if label is not None else host_id for label in labels]


def _gradient_labels(policy_ids: Sequence[str], counts: Sequence[int], *, seed: int, condition_id: str) -> list[str]:
    n = sum(counts)
    if len(policy_ids) == 1:
        return _repeated_labels(policy_ids, counts)
    rng = random.Random(seed ^ int(stable_hash({"gradient": condition_id})[:12], 16))
    remaining = [int(count) for count in counts]
    labels: list[str | None] = [None] * n
    scores: list[tuple[float, int, int]] = []
    for pos in range(n):
        x = pos / max(1, n - 1)
        for idx in range(len(policy_ids)):
            center = idx / max(1, len(policy_ids) - 1)
            scores.append((abs(x - center) + rng.random() * 0.12, idx, pos))
    for _, idx, pos in sorted(scores, key=lambda item: item[0]):
        if labels[pos] is None and remaining[idx] > 0:
            labels[pos] = str(policy_ids[idx])
            remaining[idx] -= 1
    leftovers = [str(policy_ids[idx]) for idx, count in enumerate(remaining) for _ in range(count)]
    for pos in range(n):
        if labels[pos] is None:
            labels[pos] = leftovers.pop()
    if leftovers:
        raise RuntimeError("gradient arrangement left labels unplaced")
    return [str(label) for label in labels]


def _graft_like(policy_ids: Sequence[str], counts: Sequence[int]) -> list[str]:
    n = sum(counts)
    host_idx = max(range(len(counts)), key=lambda idx: (counts[idx], idx))
    labels: list[str | None] = [str(policy_ids[host_idx])] * n
    insertions = [(idx, count) for idx, count in enumerate(counts) if idx != host_idx and count > 0]
    total_insert = sum(count for _, count in insertions)
    start = max(0, (n - total_insert) // 2)
    cursor = start
    for idx, count in insertions:
        for pos in range(cursor, min(n, cursor + count)):
            labels[pos] = str(policy_ids[idx])
        cursor += count
    return [str(label) for label in labels]


def labels_for_arrangement(
    policy_ids: Sequence[str],
    counts: Sequence[int],
    *,
    seed: int,
    condition_id: str,
    arrangement: str,
) -> tuple[str, ...]:
    """Construct exact-count initial labels for one arrangement family."""

    arrangement = str(arrangement)
    if arrangement == "random_permutation":
        labels = list(build_initial_policy_ids(policy_ids, counts, seed=seed, condition_id=condition_id))
    elif arrangement == "contiguous_patch":
        labels = _repeated_labels(policy_ids, counts)
    elif arrangement == "alternating":
        labels = _alternating_labels(policy_ids, counts)
    elif arrangement == "clustered_islands":
        labels = _clustered_islands(policy_ids, counts)
    elif arrangement == "gradient":
        labels = _gradient_labels(policy_ids, counts, seed=seed, condition_id=condition_id)
    elif arrangement == "graft_like_insertions":
        labels = _graft_like(policy_ids, counts)
    else:
        raise ValueError(f"unknown arrangement: {arrangement}")
    if len(labels) != sum(counts):
        raise RuntimeError("arrangement generator returned wrong length")
    expected = Counter(dict(zip(policy_ids, counts, strict=True)))
    observed = Counter(labels)
    if observed != expected:
        raise RuntimeError(f"arrangement generator returned wrong counts: {observed} != {expected}")
    return tuple(labels)


def arrangement_metric_row(labels: Sequence[str], policy_ids: Sequence[str]) -> dict[str, Any]:
    metrics = aggregation_metrics(labels)
    dominance = dominance_metrics(labels, policy_ids)
    return {
        "initial_aggregation_left_neighbor_percent": float(metrics["aggregation_left_neighbor_percent"]),
        "initial_expected_random_left_neighbor_percent": float(metrics["expected_random_left_neighbor_percent"]),
        "initial_aggregation_delta_percent": float(metrics["aggregation_delta_percent"]),
        "initial_interface_count": int(metrics["interface_count"]),
        "initial_contiguous_run_count": int(metrics["contiguous_run_count"]),
        "initial_largest_block_fraction": float(metrics["largest_block_fraction"]),
        "initial_label_entropy": float(metrics["label_entropy"]),
        "initial_leftmost_policy_id": dominance["leftmost_policy_id"],
        "initial_rightmost_policy_id": dominance["rightmost_policy_id"],
        "initial_position_bias_margin": float(dominance["position_bias_margin"]),
        "initial_mean_position_by_policy_json": dominance["mean_position_by_policy_json"],
    }


def build_s03_conditions(selected_candidates: pd.DataFrame, config: S03Config) -> list[ArrangementCondition]:
    conditions: list[ArrangementCondition] = []
    for row in selected_candidates.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in json.loads(str(row["policy_ids_json"])))
        display_names = tuple(str(item) for item in json.loads(str(row["display_names_json"])))
        threshold_ratios = tuple(float(item) for item in json.loads(str(row["candidate_threshold_between_ratios_json"])))
        if len(policy_ids) != 2 or len(threshold_ratios) < 1:
            continue
        for ratio in threshold_ratios:
            ratios = (float(ratio), float(1.0 - ratio))
            for arrangement in config.arrangements:
                conditions.append(
                    ArrangementCondition(
                        condition_id=_condition_id(policy_ids, ratios, arrangement),
                        s02_candidate_id=str(row["s02_candidate_id"]),
                        s02_candidate_rank=int(row["s02_candidate_rank"]),
                        panel=str(row["panel"]),
                        policy_ids=policy_ids,
                        display_names=display_names,
                        ratio_targets=ratios,
                        arrangement=str(arrangement),
                        candidate_reason=str(row["candidate_reason"]),
                        s02_threshold_between_ratios=threshold_ratios,
                    )
                )
    return conditions


def simulate_arrangement_condition(
    condition: ArrangementCondition,
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: S03Config,
) -> dict[str, Any]:
    counts = counts_for_ratios(condition.ratio_targets, config.array_size)
    values = list(initial_values(config.array_size, seed))
    labels = list(
        labels_for_arrangement(
            condition.policy_ids,
            counts,
            seed=seed,
            condition_id=condition.condition_id,
            arrangement=condition.arrangement,
        )
    )
    cell_ids = list(range(config.array_size))
    memory: dict[int, dict[str, Any]] = {}
    runtime = MixtureRuntime([records_by_id[policy_id] for policy_id in condition.policy_ids], seed)
    schedule = actor_schedule(config.array_size, config.event_cap, seed)
    rng = random.Random(seed ^ int(stable_hash(condition.condition_id)[:12], 16))
    compare_count = 0
    swap_count = 0
    update_count = 0
    wait_count = 0
    invalid_action_count = 0
    started = time.perf_counter()
    initial_metrics = sortedness_metrics(values)
    initial_counts = Counter(labels)
    initial_arrangement = arrangement_metric_row(labels, condition.policy_ids)

    for actor_index in schedule:
        actor_index = int(actor_index)
        policy_id = labels[actor_index]
        cell_id = cell_ids[actor_index]
        action = runtime.action_for(policy_id, values, labels, actor_index, cell_id, memory, rng)
        compare_count += int(bool(action.compare_counted))
        if action.action_type == "swap" and action.target_index is not None:
            target = int(action.target_index)
            if 0 <= target < len(values):
                values[actor_index], values[target] = values[target], values[actor_index]
                labels[actor_index], labels[target] = labels[target], labels[actor_index]
                cell_ids[actor_index], cell_ids[target] = cell_ids[target], cell_ids[actor_index]
                swap_count += 1
                state = memory.setdefault(cell_id, {})
                state["last_move_success"] = True
                state["last_action_type"] = "swap"
                state["time_since_movement"] = 0
            else:
                invalid_action_count += 1
                wait_count += 1
                state = memory.setdefault(cell_id, {})
                state["last_move_success"] = False
                state["last_action_type"] = "invalid_swap"
                state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)
        elif action.action_type == "update_state":
            update_count += 1
            state = memory.setdefault(cell_id, {})
            if "ideal_position" in action.state_update:
                updated = action.state_update["ideal_position"]
                state["ideal_position"] = None if updated is None else int(updated)
            state["last_action_type"] = "update_state"
        else:
            wait_count += 1
            state = memory.setdefault(cell_id, {})
            state["last_move_success"] = None
            state["last_action_type"] = "wait"
            state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)

    elapsed = time.perf_counter() - started
    final_metrics = sortedness_metrics(values)
    final_arrangement = aggregation_metrics(labels)
    final_counts = Counter(labels)
    dominance = dominance_metrics(labels, condition.policy_ids)
    realized_ratios = tuple(final_counts.get(policy_id, 0) / config.array_size for policy_id in condition.policy_ids)
    expected_counts = dict(zip(condition.policy_ids, counts, strict=True))
    final_state_class = classify_final_state(
        float(final_metrics["inversion_sortedness"]),
        float(final_arrangement["aggregation_delta_percent"]),
        float(final_arrangement["largest_block_fraction"]),
        int(final_arrangement["interface_count"]),
    )
    categories = [str(records_by_id[policy_id]["sourceCategory"]) for policy_id in condition.policy_ids]

    row = {
        "schema": SPATIAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "condition_id": condition.condition_id,
        "s02_candidate_id": condition.s02_candidate_id,
        "s02_candidate_rank": int(condition.s02_candidate_rank),
        "panel": condition.panel,
        "candidate_reason": condition.candidate_reason,
        "arrangement": condition.arrangement,
        "array_size": int(config.array_size),
        "event_cap": int(config.event_cap),
        "events_executed": int(config.event_cap),
        "seed": int(seed),
        "scheduler": config.scheduler,
        "policy_ids_json": _json_list(condition.policy_ids),
        "display_names_json": _json_list(condition.display_names),
        "source_categories_json": _json_list(categories),
        "ratio_targets_json": _json_list(condition.ratio_targets),
        "s02_threshold_between_ratios_json": _json_list(condition.s02_threshold_between_ratios),
        "expected_counts_json": json.dumps(expected_counts, sort_keys=True, separators=(",", ":")),
        "initial_counts_json": json.dumps(dict(sorted(initial_counts.items())), sort_keys=True, separators=(",", ":")),
        "final_counts_json": json.dumps(dict(sorted(final_counts.items())), sort_keys=True, separators=(",", ":")),
        "realized_ratios_json": _json_list(realized_ratios),
        "realized_ratio_max_abs_error": float(max(abs(realized_ratios[idx] - (counts[idx] / config.array_size)) for idx in range(len(counts)))),
        "count_preservation_success": dict(initial_counts) == dict(final_counts) == expected_counts,
        "compare_count": int(compare_count),
        "swap_count": int(swap_count),
        "update_count": int(update_count),
        "wait_count": int(wait_count),
        "work_count": int(compare_count + swap_count + update_count),
        "invalid_action_count": int(invalid_action_count),
        "initial_inversion_count": int(initial_metrics["inversion_count"]),
        "final_inversion_count": int(final_metrics["inversion_count"]),
        "initial_inversion_sortedness": float(initial_metrics["inversion_sortedness"]),
        "final_inversion_sortedness": float(final_metrics["inversion_sortedness"]),
        "inversion_sortedness_delta": float(final_metrics["inversion_sortedness"] - initial_metrics["inversion_sortedness"]),
        "initial_adjacent_sortedness": float(initial_metrics["adjacent_sortedness"]),
        "final_adjacent_sortedness": float(final_metrics["adjacent_sortedness"]),
        "final_is_sorted": bool(final_metrics["is_sorted"]),
        "final_aggregation_left_neighbor_percent": float(final_arrangement["aggregation_left_neighbor_percent"]),
        "expected_random_left_neighbor_percent": float(final_arrangement["expected_random_left_neighbor_percent"]),
        "aggregation_delta_percent": float(final_arrangement["aggregation_delta_percent"]),
        "interface_count": int(final_arrangement["interface_count"]),
        "contiguous_run_count": int(final_arrangement["contiguous_run_count"]),
        "largest_block_fraction": float(final_arrangement["largest_block_fraction"]),
        "label_entropy": float(final_arrangement["label_entropy"]),
        "leftmost_policy_id": dominance["leftmost_policy_id"],
        "rightmost_policy_id": dominance["rightmost_policy_id"],
        "position_bias_margin": float(dominance["position_bias_margin"]),
        "mean_position_by_policy_json": dominance["mean_position_by_policy_json"],
        "final_state_class": final_state_class,
        "initial_labels_head_json": _json_list(labels[:20]),
        "initial_labels_tail_json": _json_list(labels[-20:]),
        "final_values_head_json": _json_list(values[:20]),
        "final_values_tail_json": _json_list(values[-20:]),
        "final_labels_head_json": _json_list(labels[:20]),
        "final_labels_tail_json": _json_list(labels[-20:]),
        "elapsed_seconds": float(elapsed),
    }
    row.update(initial_arrangement)
    return row


def run_s03_sweep(
    records: Sequence[Mapping[str, Any]],
    threshold_candidates: pd.DataFrame,
    config: S03Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or S03Config()
    selected = select_s03_candidates(threshold_candidates, config)
    conditions = build_s03_conditions(selected, config)
    records_by_id = {str(record["algotypeId"]): dict(record) for record in records}
    missing = sorted({policy_id for condition in conditions for policy_id in condition.policy_ids} - set(records_by_id))
    if missing:
        raise ValueError(f"S03 selected policies missing from S01 library: {missing}")
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in config.seeds:
            rows.append(simulate_arrangement_condition(condition, records_by_id, seed=int(seed), config=config))
    run_df = pd.DataFrame(rows)
    condition_rows = [
        {
            "condition_id": condition.condition_id,
            "s02_candidate_id": condition.s02_candidate_id,
            "s02_candidate_rank": int(condition.s02_candidate_rank),
            "panel": condition.panel,
            "candidate_reason": condition.candidate_reason,
            "arrangement": condition.arrangement,
            "policy_ids_json": _json_list(condition.policy_ids),
            "display_names_json": _json_list(condition.display_names),
            "source_categories_json": _json_list([records_by_id[pid]["sourceCategory"] for pid in condition.policy_ids]),
            "ratio_targets_json": _json_list(condition.ratio_targets),
            "s02_threshold_between_ratios_json": _json_list(condition.s02_threshold_between_ratios),
            "expected_counts_json": json.dumps(
                dict(zip(condition.policy_ids, counts_for_ratios(condition.ratio_targets, config.array_size), strict=True)),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for condition in conditions
    ]
    return run_df, pd.DataFrame(condition_rows), selected


def summarize_s03_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["s02_candidate_id", "ratio_targets_json", "arrangement"]
    for _, group in run_df.groupby(group_columns, sort=False):
        first = group.iloc[0]
        class_counts = group["final_state_class"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "s02_candidate_id": first["s02_candidate_id"],
                "s02_candidate_rank": int(first["s02_candidate_rank"]),
                "panel": first["panel"],
                "candidate_reason": first["candidate_reason"],
                "arrangement": first["arrangement"],
                "policy_ids_json": first["policy_ids_json"],
                "display_names_json": first["display_names_json"],
                "source_categories_json": first["source_categories_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "expected_counts_json": first["expected_counts_json"],
                "seed_count": int(group["seed"].nunique()),
                "run_count": int(len(group)),
                "mean_initial_interface_count": float(group["initial_interface_count"].mean()),
                "mean_initial_contiguous_run_count": float(group["initial_contiguous_run_count"].mean()),
                "mean_initial_largest_block_fraction": float(group["initial_largest_block_fraction"].mean()),
                "mean_initial_aggregation_delta_percent": float(group["initial_aggregation_delta_percent"].mean()),
                "mean_final_inversion_sortedness": float(group["final_inversion_sortedness"].mean()),
                "sd_final_inversion_sortedness": float(group["final_inversion_sortedness"].std(ddof=0)),
                "mean_inversion_sortedness_delta": float(group["inversion_sortedness_delta"].mean()),
                "sorted_run_fraction": float(group["final_is_sorted"].mean()),
                "mean_aggregation_delta_percent": float(group["aggregation_delta_percent"].mean()),
                "mean_interface_count": float(group["interface_count"].mean()),
                "mean_largest_block_fraction": float(group["largest_block_fraction"].mean()),
                "mean_position_bias_margin": float(group["position_bias_margin"].mean()),
                "mean_work_count": float(group["work_count"].mean()),
                "invalid_action_count": int(group["invalid_action_count"].sum()),
                "final_state_class_mode": str(group["final_state_class"].mode().iloc[0]),
                "final_state_class_counts_json": json.dumps(class_counts, sort_keys=True, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def arrangement_sensitivity(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary_df.empty:
        return pd.DataFrame()
    group_columns = ["s02_candidate_id", "ratio_targets_json"]
    for _, group in summary_df.groupby(group_columns, sort=False):
        first = group.iloc[0]
        sortedness = group["mean_final_inversion_sortedness"].astype(float)
        aggregation = group["mean_aggregation_delta_percent"].astype(float)
        initial_interfaces = group["mean_initial_interface_count"].astype(float)
        best_idx = sortedness.idxmax()
        worst_idx = sortedness.idxmin()
        rows.append(
            {
                "s02_candidate_id": first["s02_candidate_id"],
                "s02_candidate_rank": int(first["s02_candidate_rank"]),
                "panel": first["panel"],
                "candidate_reason": first["candidate_reason"],
                "policy_ids_json": first["policy_ids_json"],
                "display_names_json": first["display_names_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "arrangement_count": int(group["arrangement"].nunique()),
                "sortedness_range_across_arrangements": float(sortedness.max() - sortedness.min()),
                "aggregation_delta_range_across_arrangements": float(aggregation.max() - aggregation.min()),
                "initial_interface_range": float(initial_interfaces.max() - initial_interfaces.min()),
                "best_sortedness_arrangement": str(group.loc[best_idx, "arrangement"]),
                "best_sortedness": float(group.loc[best_idx, "mean_final_inversion_sortedness"]),
                "worst_sortedness_arrangement": str(group.loc[worst_idx, "arrangement"]),
                "worst_sortedness": float(group.loc[worst_idx, "mean_final_inversion_sortedness"]),
                "arrangement_sensitive_candidate": bool((sortedness.max() - sortedness.min()) >= 0.05 or (aggregation.max() - aggregation.min()) >= 10.0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["sortedness_range_across_arrangements", "aggregation_delta_range_across_arrangements"],
        ascending=[False, False],
        kind="mergesort",
    )


def validate_s03_outputs(
    run_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    config: S03Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    seed_set = set(int(seed) for seed in config.seeds)
    arrangement_set = set(config.arrangements)
    required_run_columns = {
        "condition_id",
        "s02_candidate_id",
        "ratio_targets_json",
        "arrangement",
        "seed",
        "realized_ratio_max_abs_error",
        "count_preservation_success",
        "invalid_action_count",
        "initial_interface_count",
        "initial_contiguous_run_count",
        "initial_largest_block_fraction",
        "initial_aggregation_delta_percent",
    }
    has_run_schema = required_run_columns.issubset(run_df.columns)
    seed_sets = []
    arrangement_sets = []
    layout_metric_distinct = False
    metric_complete = False
    ratio_success = False
    invalid_success = False
    if has_run_schema and not run_df.empty:
        seed_sets = run_df.groupby(["s02_candidate_id", "ratio_targets_json", "arrangement"])["seed"].agg(
            lambda values: set(int(value) for value in values)
        ).tolist()
        arrangement_sets = run_df.groupby(["s02_candidate_id", "ratio_targets_json", "seed"])["arrangement"].agg(
            lambda values: set(map(str, values))
        ).tolist()
        metric_columns = [
            "initial_interface_count",
            "initial_contiguous_run_count",
            "initial_largest_block_fraction",
            "initial_aggregation_delta_percent",
        ]
        metric_complete = bool(run_df[metric_columns].notna().all().all() and np.isfinite(run_df[metric_columns].to_numpy(dtype=float)).all())
        distinct_counts = run_df.groupby(["s02_candidate_id", "ratio_targets_json", "seed"])["initial_contiguous_run_count"].nunique()
        layout_metric_distinct = bool((distinct_counts >= 2).all())
        ratio_success = bool((run_df["realized_ratio_max_abs_error"] <= 1e-12).all() and run_df["count_preservation_success"].all())
        invalid_success = int(run_df["invalid_action_count"].sum()) == 0
    checks = [
        {
            "validation_case": "initial_arrangement_metrics_computed",
            "success": metric_complete,
            "observed": f"rows={len(run_df)} metric_columns_present={has_run_schema}",
            "expected": "initial arrangement metrics are present and finite for every run",
        },
        {
            "validation_case": "arrangement_metrics_distinguish_layouts",
            "success": layout_metric_distinct,
            "observed": "minimum distinct initial run counts per candidate-ratio-seed >= 2" if layout_metric_distinct else "some candidate-ratio-seed groups had fewer than two distinct layout metrics",
            "expected": "arrangement generators produce measurably different initial layouts",
        },
        {
            "validation_case": "matched_seed_sets_by_condition",
            "success": bool(seed_sets and all(item == seed_set for item in seed_sets)),
            "observed": f"{len(seed_sets)} condition groups; expected_seeds={sorted(seed_set)}",
            "expected": "each candidate-ratio-arrangement condition uses the same configured seed set",
        },
        {
            "validation_case": "matched_arrangements_by_seed",
            "success": bool(arrangement_sets and all(item == arrangement_set for item in arrangement_sets)),
            "observed": f"{len(arrangement_sets)} candidate-ratio-seed groups; expected_arrangements={sorted(arrangement_set)}",
            "expected": "each candidate-ratio-seed group includes every configured arrangement",
        },
        {
            "validation_case": "realized_ratios_and_counts_match",
            "success": ratio_success,
            "observed": f"max_error={run_df['realized_ratio_max_abs_error'].max():.3g}; count_failures={int((~run_df['count_preservation_success']).sum())}" if has_run_schema and not run_df.empty else "missing run rows",
            "expected": "zero realized-ratio error against integer counts and all counts preserved",
        },
        {
            "validation_case": "bounded_candidate_set_from_s02",
            "success": bool(not selected_candidates.empty and len(selected_candidates) <= config.max_candidate_pairs),
            "observed": f"selected_candidates={len(selected_candidates)} max={config.max_candidate_pairs}",
            "expected": "bounded nonempty S02-derived candidate set",
        },
        {
            "validation_case": "result_rows_complete",
            "success": bool(len(run_df) == len(condition_df) * len(seed_set) and not condition_df.empty),
            "observed": f"runs={len(run_df)} conditions={len(condition_df)} seeds={len(seed_set)}",
            "expected": "run rows equal condition count times seed count",
        },
        {
            "validation_case": "no_invalid_actions",
            "success": invalid_success,
            "observed": str(int(run_df["invalid_action_count"].sum())) if has_run_schema and not run_df.empty else "missing run rows",
            "expected": "zero out-of-bounds or invalid actions",
        },
        {
            "validation_case": "figure_written",
            "success": bool(figure_written),
            "observed": str(bool(figure_written)),
            "expected": "spatial arrangement outcome figure exists",
        },
        {
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests_success),
            "observed": str(bool(unit_tests_success)),
            "expected": "focused E06 unit tests pass",
        },
    ]
    return pd.DataFrame(checks)
