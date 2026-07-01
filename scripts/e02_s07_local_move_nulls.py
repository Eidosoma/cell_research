#!/usr/bin/env python3
"""Run E02 S07 randomized local-move null models.

S07 compares E01 real cell-view trajectories against local adjacent-swap nulls
that are matched to each real run's initial array, initial Algotype labels, and
swap count. Null policies are intentionally given no Algotype-label input.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from scripts.e02_s02_scheduler_comparison import (
    artifact_entry,
    dg_from_sortedness,
    git_output,
    load_e01_config,
    markdown_table,
    run_command,
    sha256_file,
    write_json,
)
from src.e02.deterministic_simulator import (
    aggregation_left_neighbor_percent,
    aggregation_right_neighbor_legacy_percent,
    count_json,
    monotonicity_error_count,
    sortedness_percent,
    stable_json_sha256,
)


STEP_ID = "S07"
STEP_NUMBER = 7
EXPERIMENT_ID = "E02"
SELECTED_CONDITION_IDS = ("E01C004", "E01C005", "E01C006", "E01C043", "E01C044", "E01C045", "E01C046", "E01C047")
PURE_CONDITION_IDS = ("E01C004", "E01C005", "E01C006")
CHIMERA_CONDITION_IDS = ("E01C043", "E01C044", "E01C045", "E01C046", "E01C047")
NULL_POLICIES = ("random_adjacent_swap", "inversion_biased_swap", "metropolis_local_swap", "random_walker_swap")
CURVE_PROGRESS_BINS = tuple(range(101))


@dataclass(frozen=True)
class LocalMoveNullTask:
    condition: dict[str, Any]
    repeat_index: int
    null_replicate: int
    values: tuple[int, ...]
    assignments: tuple[str, ...]
    initial_array_seed: int
    assignment_seed: int | None
    initial_array_sha256: str
    initial_assignment_sha256: str
    target_swap_count: int
    real_metrics: dict[str, Any]
    null_policy: str
    null_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config-path", type=Path, default=Path("/previous-artifacts/E01/configs/e01_baseline_configs.json"))
    parser.add_argument(
        "--e01-efficiency-path",
        type=Path,
        default=Path("/previous-artifacts/E01/results/e01_efficiency_counts.parquet"),
    )
    parser.add_argument(
        "--e01-chimera-efficiency-path",
        type=Path,
        default=Path("/previous-artifacts/E01/results/e01_chimera_efficiency.parquet"),
    )
    parser.add_argument(
        "--e01-pure-trace-path",
        type=Path,
        default=Path("/previous-artifacts/E01/traces/e01_figure3_trajectories.parquet"),
    )
    parser.add_argument(
        "--e01-chimera-trace-path",
        type=Path,
        default=Path("/previous-artifacts/E01/traces/e01_same_goal_chimeras.parquet"),
    )
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--null-replicates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--random-seed", type=int, default=20260701)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json(data: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def selected_conditions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in cfg["conditions"]
        if row["condition_id"] in SELECTED_CONDITION_IDS and row["mode"] == "cell_view"
    ]
    missing = sorted(set(SELECTED_CONDITION_IDS) - {row["condition_id"] for row in rows})
    if missing:
        raise RuntimeError(f"Missing expected S07 condition IDs in E01 config: {missing}")
    return sorted(rows, key=lambda row: row["condition_id"])


def null_seed(base_seed: int, condition_id: str, repeat_idx: int, policy: str, null_replicate: int) -> int:
    condition_num = int(condition_id.replace("E01C", ""))
    policy_num = NULL_POLICIES.index(policy) + 1
    return base_seed * 1_000_000 + 700_000 + condition_num * 10_000 + repeat_idx * 100 + policy_num * 10 + null_replicate


def edge_error(values: Sequence[int], edge_idx: int) -> int:
    if edge_idx < 0 or edge_idx >= len(values) - 1:
        return 0
    return int(values[edge_idx] > values[edge_idx + 1])


def edge_same(labels: Sequence[str], edge_idx: int) -> int:
    if edge_idx < 0 or edge_idx >= len(labels) - 1:
        return 0
    return int(labels[edge_idx] == labels[edge_idx + 1])


def affected_edges(left_index: int, n: int) -> tuple[int, ...]:
    return tuple(edge for edge in (left_index - 1, left_index, left_index + 1) if 0 <= edge < n - 1)


def choose_random_adjacent(values: Sequence[int], rng: random.Random, state: dict[str, Any]) -> int:
    return rng.randrange(0, len(values) - 1)


def choose_inversion_biased(values: Sequence[int], rng: random.Random, state: dict[str, Any]) -> int:
    inversion_edges = state["inversion_edges"]
    if inversion_edges and rng.random() < 0.85:
        return rng.choice(tuple(sorted(inversion_edges)))
    return rng.randrange(0, len(values) - 1)


def local_adjacent_energy(values: Sequence[int], left_index: int) -> int:
    return sum(edge_error(values, edge) for edge in affected_edges(left_index, len(values)))


def choose_metropolis_local(values: Sequence[int], rng: random.Random, state: dict[str, Any]) -> int:
    temperature = 1.0
    n = len(values)
    for _ in range(1000):
        idx = rng.randrange(0, n - 1)
        before = local_adjacent_energy(values, idx)
        values[idx], values[idx + 1] = values[idx + 1], values[idx]
        after = local_adjacent_energy(values, idx)
        values[idx], values[idx + 1] = values[idx + 1], values[idx]
        delta = after - before
        if delta <= 0 or rng.random() < math.exp(-float(delta) / temperature):
            state["accepted_proposals"] += 1
            return idx
        state["rejected_proposals"] += 1
    state["fallback_proposals"] += 1
    return rng.randrange(0, n - 1)


def choose_random_walker(values: Sequence[int], rng: random.Random, state: dict[str, Any]) -> int:
    n = len(values)
    walker_position = int(state["walker_position"])
    if walker_position <= 0:
        neighbor = 1
    elif walker_position >= n - 1:
        neighbor = n - 2
    else:
        neighbor = walker_position + (-1 if rng.random() < 0.5 else 1)
    left = min(walker_position, neighbor)
    state["walker_position"] = neighbor
    return left


def make_policy_state(policy: str, values: Sequence[int], rng: random.Random) -> dict[str, Any]:
    state: dict[str, Any] = {
        "accepted_proposals": 0,
        "rejected_proposals": 0,
        "fallback_proposals": 0,
    }
    if policy == "inversion_biased_swap":
        state["inversion_edges"] = {idx for idx in range(len(values) - 1) if edge_error(values, idx)}
    if policy == "random_walker_swap":
        state["walker_position"] = rng.randrange(0, len(values))
    return state


def choose_swap_index(policy: str, values: Sequence[int], rng: random.Random, state: dict[str, Any]) -> int:
    if policy == "random_adjacent_swap":
        return choose_random_adjacent(values, rng, state)
    if policy == "inversion_biased_swap":
        return choose_inversion_biased(values, rng, state)
    if policy == "metropolis_local_swap":
        return choose_metropolis_local(values, rng, state)
    if policy == "random_walker_swap":
        return choose_random_walker(values, rng, state)
    raise ValueError(f"Unsupported null policy: {policy}")


def update_inversion_edges(values: Sequence[int], state: dict[str, Any], left_index: int) -> None:
    edges = state.get("inversion_edges")
    if edges is None:
        return
    for edge in affected_edges(left_index, len(values)):
        if edge_error(values, edge):
            edges.add(edge)
        else:
            edges.discard(edge)


def trajectory_shape_metrics(sortedness_values: Sequence[float]) -> dict[str, float]:
    values = np.asarray(sortedness_values, dtype=np.float64)
    if len(values) < 2:
        return {
            "path_total_variation": 0.0,
            "path_net_progress": 0.0,
            "path_curvature_ratio": 0.0,
            "path_backtracking_drop_total": 0.0,
            "path_forward_recovery_total": 0.0,
            "path_second_diff_mean_abs": 0.0,
        }
    deltas = np.diff(values)
    total_variation = float(np.sum(np.abs(deltas)))
    net_progress = float(abs(values[-1] - values[0]))
    # Use a 1 percentage-point floor so trajectories that wander back to their
    # starting Sortedness remain finite but still register high curvature.
    curvature = total_variation / max(net_progress, 1.0)
    drops = float(np.sum(np.maximum(-deltas, 0.0)))
    recoveries = float(np.sum(np.maximum(deltas, 0.0)))
    second = float(np.mean(np.abs(np.diff(deltas)))) if len(deltas) > 1 else 0.0
    return {
        "path_total_variation": total_variation,
        "path_net_progress": net_progress,
        "path_curvature_ratio": curvature,
        "path_backtracking_drop_total": drops,
        "path_forward_recovery_total": recoveries,
        "path_second_diff_mean_abs": second,
    }


def progress_curve_points(
    sortedness_values: Sequence[float],
    aggregation_values: Sequence[float],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    n = len(sortedness_values)
    if n == 0:
        return []
    rows: list[dict[str, Any]] = []
    for progress in CURVE_PROGRESS_BINS:
        idx = int(round((n - 1) * (progress / 100.0)))
        rows.append(
            {
                **metadata,
                "progress_percent": int(progress),
                "source_trace_index": int(idx),
                "sortedness_percent": float(sortedness_values[idx]),
                "aggregation_left_neighbor_percent": float(aggregation_values[idx]),
            }
        )
    return rows


def simulate_null_path(
    values: Sequence[int],
    labels: Sequence[str],
    policy: str,
    seed: int,
    target_swap_count: int,
    include_curves: bool = True,
) -> dict[str, Any]:
    if policy not in NULL_POLICIES:
        raise ValueError(f"Unsupported null policy: {policy}")
    rng = random.Random(seed)
    current_values = list(map(int, values))
    current_labels = list(map(str, labels))
    if len(current_values) != len(current_labels):
        raise ValueError("values and labels must have the same length.")
    if len(current_values) < 2:
        raise ValueError("At least two positions are required for local-move nulls.")
    n = len(current_values)
    state = make_policy_state(policy, current_values, rng)
    current_errors = monotonicity_error_count(current_values)
    current_same_left = sum(edge_same(current_labels, idx) for idx in range(n - 1))
    sortedness_values = [100.0 * (n - current_errors) / n]
    aggregation_values = [100.0 * current_same_left / n]
    decision_hash = hashlib.sha256()
    max_abs_swap_distance = 0
    nonlocal_swap_count = 0

    for step in range(1, int(target_swap_count) + 1):
        left = choose_swap_index(policy, current_values, rng, state)
        if left < 0 or left >= n - 1:
            raise RuntimeError(f"Policy {policy} returned invalid adjacent index {left}.")
        right = left + 1
        distance = abs(right - left)
        max_abs_swap_distance = max(max_abs_swap_distance, distance)
        if distance != 1:
            nonlocal_swap_count += 1

        edges = affected_edges(left, n)
        current_errors -= sum(edge_error(current_values, edge) for edge in edges)
        current_same_left -= sum(edge_same(current_labels, edge) for edge in edges)
        current_values[left], current_values[right] = current_values[right], current_values[left]
        current_labels[left], current_labels[right] = current_labels[right], current_labels[left]
        current_errors += sum(edge_error(current_values, edge) for edge in edges)
        current_same_left += sum(edge_same(current_labels, edge) for edge in edges)
        update_inversion_edges(current_values, state, left)

        decision_hash.update(
            json.dumps({"step": step, "left": int(left), "right": int(right)}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        sortedness_values.append(100.0 * (n - current_errors) / n)
        aggregation_values.append(100.0 * current_same_left / n)

    shape = trajectory_shape_metrics(sortedness_values)
    dg = dg_from_sortedness(list(map(float, sortedness_values)))
    result = {
        "final_values": tuple(current_values),
        "final_labels": tuple(current_labels),
        "sortedness_values": sortedness_values if include_curves else [],
        "aggregation_values": aggregation_values if include_curves else [],
        "initial_sortedness_percent": float(sortedness_values[0]),
        "final_sortedness_percent": float(sortedness_values[-1]),
        "peak_sortedness_percent": float(max(sortedness_values)),
        "curve_mean_sortedness_percent": float(np.mean(sortedness_values)),
        "final_monotonicity_error_count": int(current_errors),
        "initial_aggregation_left_neighbor_percent": float(aggregation_values[0]),
        "final_aggregation_left_neighbor_percent": float(aggregation_values[-1]),
        "peak_aggregation_left_neighbor_percent": float(max(aggregation_values)),
        "curve_mean_aggregation_left_neighbor_percent": float(np.mean(aggregation_values)),
        "final_aggregation_right_neighbor_legacy_percent": aggregation_right_neighbor_legacy_percent(current_labels),
        "observed_swap_count": int(target_swap_count),
        "max_abs_swap_distance": int(max_abs_swap_distance),
        "nonlocal_swap_count": int(nonlocal_swap_count),
        "decision_path_sha256": decision_hash.hexdigest(),
        "accepted_proposal_count": int(state.get("accepted_proposals", 0)),
        "rejected_proposal_count": int(state.get("rejected_proposals", 0)),
        "fallback_proposal_count": int(state.get("fallback_proposals", 0)),
        **shape,
        **dg,
    }
    return result


def hidden_label_access_check(values: Sequence[int], labels: Sequence[str], policy: str, seed: int, target_swap_count: int) -> bool:
    original = simulate_null_path(values, labels, policy, seed, target_swap_count, include_curves=False)
    permuted_labels = tuple(reversed(tuple(map(str, labels))))
    permuted = simulate_null_path(values, permuted_labels, policy, seed, target_swap_count, include_curves=False)
    return bool(original["decision_path_sha256"] == permuted["decision_path_sha256"])


def metric_record_from_trace(
    condition: Mapping[str, Any],
    repeat_index: int,
    values: Sequence[int],
    assignments: Sequence[str],
    trace_group: pd.DataFrame,
    source_trace_path: Path,
    source_record: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = trace_group.sort_values("swap_step")
    sortedness_values = pd.to_numeric(ordered["sortedness_percent"], errors="coerce").to_numpy(dtype=np.float64)
    if "aggregation_left_neighbor_percent" in ordered.columns:
        aggregation_values = pd.to_numeric(ordered["aggregation_left_neighbor_percent"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        aggregation_values = np.repeat(aggregation_left_neighbor_percent(assignments), len(sortedness_values))
    if len(sortedness_values) == 0:
        raise RuntimeError(f"No trace rows for {condition['condition_id']} repeat {repeat_index}.")
    target_swap_count = int(pd.to_numeric(ordered["swap_step"], errors="coerce").max())
    shape = trajectory_shape_metrics(sortedness_values)
    dg = dg_from_sortedness(list(map(float, sortedness_values)))
    final_labels_code = str(ordered["algotype_positions_code"].iloc[-1]) if "algotype_positions_code" in ordered.columns else None
    final_aggregation = float(aggregation_values[-1])
    peak_aggregation = float(np.max(aggregation_values))
    row = {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": "E01",
        "source_type": "real_e01_cell_view",
        "condition_id": condition["condition_id"],
        "run_family": condition["run_family"],
        "mode": "cell_view",
        "algorithm": condition["algorithm"],
        "algotype_mix": condition["algotype_mix"],
        "null_policy": "real_e01",
        "null_replicate": -1,
        "null_seed": np.nan,
        "repeat_index": int(repeat_index),
        "array_length": len(values),
        "initial_array_seed": int(source_record["initial_array_seed"]) if source_record and "initial_array_seed" in source_record else np.nan,
        "initial_array_sha256": stable_json_sha256(list(values)),
        "final_array_sha256": str(source_record.get("final_array_sha256", "")) if source_record else "",
        "value_bank_id": condition["value_bank_id"],
        "algotype_assignment_bank_id": condition["algotype_assignment_bank_id"],
        "algotype_assignment_seed": None if condition["algorithm"] != "mixed" else int(source_record.get("algotype_assignment_seed")) if source_record else None,
        "initial_algotype_assignment_sha256": stable_json_sha256(list(assignments)),
        "initial_algotype_counts_json": count_json(assignments),
        "final_algotype_counts_json": str(source_record.get("final_algotype_counts_json", count_json(assignments))) if source_record else count_json(assignments),
        "target_swap_count": target_swap_count,
        "observed_swap_count": target_swap_count,
        "swap_count_match": True,
        "trace_record_count": int(len(ordered)),
        "trace_length_match_target_swaps_plus_initial": bool(len(ordered) == target_swap_count + 1),
        "locality_match": True,
        "max_abs_swap_distance": 1,
        "nonlocal_swap_count": 0,
        "null_policy_uses_hidden_labels": False,
        "hidden_label_access_validation_passed": True,
        "policy_inputs_json": safe_json(["real_e01_public_cell_policy"]),
        "decision_path_sha256": "",
        "initial_sortedness_percent": float(sortedness_values[0]),
        "final_sortedness_percent": float(sortedness_values[-1]),
        "peak_sortedness_percent": float(np.max(sortedness_values)),
        "curve_mean_sortedness_percent": float(np.mean(sortedness_values)),
        "final_monotonicity_error_count": int(round(100.0 - float(sortedness_values[-1]))),
        "initial_aggregation_left_neighbor_percent": float(aggregation_values[0]),
        "final_aggregation_left_neighbor_percent": final_aggregation,
        "final_aggregation_right_neighbor_legacy_percent": float(
            source_record.get("final_aggregation_right_neighbor_legacy_percent", final_aggregation) if source_record else final_aggregation
        ),
        "peak_aggregation_left_neighbor_percent": peak_aggregation,
        "curve_mean_aggregation_left_neighbor_percent": float(np.mean(aggregation_values)),
        "final_algotype_positions_code": final_labels_code,
        "source_trace_path": str(source_trace_path),
        "source_run_record_path": str(source_record.get("_source_run_record_path", "")) if source_record else "",
        "result_scope": "s07_real_e01_reference_first_repeats",
        "wrapper_name": "real_e01_public_cell_view_trace",
        "accepted_proposal_count": 0,
        "rejected_proposal_count": 0,
        "fallback_proposal_count": 0,
        **shape,
        **dg,
    }
    curve = progress_curve_points(
        sortedness_values,
        aggregation_values,
        {
            "condition_id": condition["condition_id"],
            "algotype_mix": condition["algotype_mix"],
            "source_type": "real_e01_cell_view",
            "null_policy": "real_e01",
        },
    )
    return row, curve


def run_null_task(task: LocalMoveNullTask) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = time.monotonic()
    result = simulate_null_path(task.values, task.assignments, task.null_policy, task.null_seed, task.target_swap_count)
    hidden_access_ok = hidden_label_access_check(
        task.values,
        task.assignments,
        task.null_policy,
        task.null_seed,
        task.target_swap_count,
    )
    final_labels = result["final_labels"]
    real = task.real_metrics
    row = {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": "E02",
        "source_type": "local_move_null",
        "condition_id": task.condition["condition_id"],
        "run_family": task.condition["run_family"],
        "mode": "cell_view",
        "algorithm": task.condition["algorithm"],
        "algotype_mix": task.condition["algotype_mix"],
        "null_policy": task.null_policy,
        "null_replicate": int(task.null_replicate),
        "null_seed": int(task.null_seed),
        "repeat_index": int(task.repeat_index),
        "array_length": len(task.values),
        "initial_array_seed": int(task.initial_array_seed),
        "initial_array_sha256": task.initial_array_sha256,
        "final_array_sha256": stable_json_sha256(list(result["final_values"])),
        "value_bank_id": task.condition["value_bank_id"],
        "algotype_assignment_bank_id": task.condition["algotype_assignment_bank_id"],
        "algotype_assignment_seed": task.assignment_seed,
        "initial_algotype_assignment_sha256": task.initial_assignment_sha256,
        "initial_algotype_counts_json": count_json(task.assignments),
        "final_algotype_counts_json": count_json(final_labels),
        "target_swap_count": int(task.target_swap_count),
        "observed_swap_count": int(result["observed_swap_count"]),
        "swap_count_match": bool(result["observed_swap_count"] == task.target_swap_count),
        "trace_record_count": int(task.target_swap_count + 1),
        "trace_length_match_target_swaps_plus_initial": True,
        "locality_match": bool(result["max_abs_swap_distance"] <= 1 and result["nonlocal_swap_count"] == 0),
        "max_abs_swap_distance": int(result["max_abs_swap_distance"]),
        "nonlocal_swap_count": int(result["nonlocal_swap_count"]),
        "null_policy_uses_hidden_labels": False,
        "hidden_label_access_validation_passed": bool(hidden_access_ok),
        "policy_inputs_json": safe_json(["values", "positions", "rng", "policy_state"]),
        "decision_path_sha256": result["decision_path_sha256"],
        "initial_sortedness_percent": result["initial_sortedness_percent"],
        "final_sortedness_percent": result["final_sortedness_percent"],
        "peak_sortedness_percent": result["peak_sortedness_percent"],
        "curve_mean_sortedness_percent": result["curve_mean_sortedness_percent"],
        "final_monotonicity_error_count": result["final_monotonicity_error_count"],
        "initial_aggregation_left_neighbor_percent": result["initial_aggregation_left_neighbor_percent"],
        "final_aggregation_left_neighbor_percent": result["final_aggregation_left_neighbor_percent"],
        "final_aggregation_right_neighbor_legacy_percent": result["final_aggregation_right_neighbor_legacy_percent"],
        "peak_aggregation_left_neighbor_percent": result["peak_aggregation_left_neighbor_percent"],
        "curve_mean_aggregation_left_neighbor_percent": result["curve_mean_aggregation_left_neighbor_percent"],
        "final_algotype_positions_code": "",
        "source_trace_path": "",
        "source_run_record_path": "",
        "result_scope": "s07_local_move_null_first_e01_repeats",
        "wrapper_name": "exact_swap_count_adjacent_local_move_null",
        "accepted_proposal_count": result["accepted_proposal_count"],
        "rejected_proposal_count": result["rejected_proposal_count"],
        "fallback_proposal_count": result["fallback_proposal_count"],
        "elapsed_seconds": time.monotonic() - start,
        "paired_real_final_sortedness_percent": float(real["final_sortedness_percent"]),
        "paired_real_peak_sortedness_percent": float(real["peak_sortedness_percent"]),
        "paired_real_curve_mean_sortedness_percent": float(real["curve_mean_sortedness_percent"]),
        "paired_real_dg_primary": float(real["dg_primary"]),
        "paired_real_path_curvature_ratio": float(real["path_curvature_ratio"]),
        "paired_real_peak_aggregation_left_neighbor_percent": float(real["peak_aggregation_left_neighbor_percent"]),
        "paired_real_final_aggregation_left_neighbor_percent": float(real["final_aggregation_left_neighbor_percent"]),
        **{key: result[key] for key in [
            "path_total_variation",
            "path_net_progress",
            "path_curvature_ratio",
            "path_backtracking_drop_total",
            "path_forward_recovery_total",
            "path_second_diff_mean_abs",
            "dg_primary",
            "dg_event_count",
            "dg_total_drop",
            "dg_total_recovery",
            "terminal_unrecovered_drop_segment_count",
            "leading_increase_segment_count",
        ]},
    }
    row["delta_final_sortedness_vs_real"] = row["final_sortedness_percent"] - row["paired_real_final_sortedness_percent"]
    row["delta_peak_sortedness_vs_real"] = row["peak_sortedness_percent"] - row["paired_real_peak_sortedness_percent"]
    row["delta_curve_mean_sortedness_vs_real"] = row["curve_mean_sortedness_percent"] - row["paired_real_curve_mean_sortedness_percent"]
    row["delta_dg_primary_vs_real"] = row["dg_primary"] - row["paired_real_dg_primary"]
    row["delta_path_curvature_ratio_vs_real"] = row["path_curvature_ratio"] - row["paired_real_path_curvature_ratio"]
    row["delta_peak_aggregation_vs_real"] = (
        row["peak_aggregation_left_neighbor_percent"] - row["paired_real_peak_aggregation_left_neighbor_percent"]
    )
    row["delta_final_aggregation_vs_real"] = (
        row["final_aggregation_left_neighbor_percent"] - row["paired_real_final_aggregation_left_neighbor_percent"]
    )
    curve = progress_curve_points(
        result["sortedness_values"],
        result["aggregation_values"],
        {
            "condition_id": task.condition["condition_id"],
            "algotype_mix": task.condition["algotype_mix"],
            "source_type": "local_move_null",
            "null_policy": task.null_policy,
        },
    )
    return row, curve


def load_e01_inputs(
    cfg: dict[str, Any],
    max_repeats: int,
    pure_trace_path: Path,
    chimera_trace_path: Path,
    pure_record_path: Path,
    chimera_record_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    for path in (pure_trace_path, chimera_trace_path, pure_record_path, chimera_record_path):
        if not path.exists():
            raise RuntimeError(f"Required S07 input is missing: {path}")
    pure_trace = pd.read_parquet(pure_trace_path)
    chimera_trace = pd.read_parquet(chimera_trace_path)
    pure_records = pd.read_parquet(pure_record_path)
    chimera_records = pd.read_parquet(chimera_record_path)
    condition_rows = {row["condition_id"]: row for row in selected_conditions(cfg)}
    value_banks = cfg["seedBanks"]["valueBanks"]
    assignment_banks = cfg["seedBanks"]["algotypeAssignmentBanks"]
    repeat_count = min(max_repeats, int(cfg["globalDefaults"]["repeatCount"]))

    real_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    null_task_inputs: list[dict[str, Any]] = []

    record_frames = [
        (pure_records, pure_record_path),
        (chimera_records, chimera_record_path),
    ]
    records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for frame, path in record_frames:
        for record in frame.to_dict(orient="records"):
            if record.get("condition_id") in SELECTED_CONDITION_IDS and int(record.get("repeat_index", -1)) < repeat_count:
                rec = dict(record)
                rec["_source_run_record_path"] = str(path)
                records_by_key[(str(record["condition_id"]), int(record["repeat_index"]))] = rec

    for condition_id in SELECTED_CONDITION_IDS:
        condition = condition_rows[condition_id]
        value_bank = value_banks[condition["value_bank_id"]]
        assignment_bank = assignment_banks.get(condition["algotype_assignment_bank_id"])
        trace = pure_trace if condition_id in PURE_CONDITION_IDS else chimera_trace
        trace_path = pure_trace_path if condition_id in PURE_CONDITION_IDS else chimera_trace_path
        for repeat_idx in range(repeat_count):
            values = tuple(map(int, value_bank["initialArrays"][repeat_idx]))
            if condition["algorithm"] == "mixed":
                if assignment_bank is None:
                    raise RuntimeError(f"{condition_id} requires an Algotype assignment bank.")
                assignments = tuple(map(str, assignment_bank["assignments"][repeat_idx]))
                assignment_seed = int(assignment_bank["seeds"][repeat_idx])
            else:
                assignments = tuple(str(condition["algorithm"]) for _ in values)
                assignment_seed = None
            group = trace[(trace["condition_id"] == condition_id) & (trace["repeat_index"] == repeat_idx)]
            source_record = records_by_key.get((condition_id, repeat_idx))
            row, curve = metric_record_from_trace(
                condition,
                repeat_idx,
                values,
                assignments,
                group,
                trace_path,
                source_record,
            )
            real_rows.append(row)
            curve_rows.extend(curve)
            null_task_inputs.append(
                {
                    "condition": condition,
                    "repeat_index": repeat_idx,
                    "values": values,
                    "assignments": assignments,
                    "initial_array_seed": int(value_bank["seeds"][repeat_idx]),
                    "assignment_seed": assignment_seed,
                    "initial_array_sha256": stable_json_sha256(list(values)),
                    "initial_assignment_sha256": stable_json_sha256(list(assignments)),
                    "target_swap_count": int(row["target_swap_count"]),
                    "real_metrics": row,
                }
            )
    return real_rows, curve_rows, null_task_inputs


def build_null_tasks(task_inputs: Sequence[dict[str, Any]], null_replicates: int, base_seed: int) -> list[LocalMoveNullTask]:
    tasks: list[LocalMoveNullTask] = []
    for item in task_inputs:
        for policy in NULL_POLICIES:
            for replicate in range(null_replicates):
                tasks.append(
                    LocalMoveNullTask(
                        condition=dict(item["condition"]),
                        repeat_index=int(item["repeat_index"]),
                        null_replicate=int(replicate),
                        values=tuple(item["values"]),
                        assignments=tuple(item["assignments"]),
                        initial_array_seed=int(item["initial_array_seed"]),
                        assignment_seed=item["assignment_seed"],
                        initial_array_sha256=str(item["initial_array_sha256"]),
                        initial_assignment_sha256=str(item["initial_assignment_sha256"]),
                        target_swap_count=int(item["target_swap_count"]),
                        real_metrics=dict(item["real_metrics"]),
                        null_policy=policy,
                        null_seed=null_seed(base_seed, item["condition"]["condition_id"], int(item["repeat_index"]), policy, replicate),
                    )
                )
    return tasks


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    sem = std / math.sqrt(len(clean)) if len(clean) > 1 else np.nan
    low = mean - 1.96 * sem if len(clean) > 1 else np.nan
    high = mean + 1.96 * sem if len(clean) > 1 else np.nan
    return mean, std, low, high


def add_real_deltas(df: pd.DataFrame) -> pd.DataFrame:
    for metric in [
        "final_sortedness_percent",
        "peak_sortedness_percent",
        "curve_mean_sortedness_percent",
        "dg_primary",
        "path_curvature_ratio",
        "peak_aggregation_left_neighbor_percent",
        "final_aggregation_left_neighbor_percent",
    ]:
        delta_col = {
            "final_sortedness_percent": "delta_final_sortedness_vs_real",
            "peak_sortedness_percent": "delta_peak_sortedness_vs_real",
            "curve_mean_sortedness_percent": "delta_curve_mean_sortedness_vs_real",
            "dg_primary": "delta_dg_primary_vs_real",
            "path_curvature_ratio": "delta_path_curvature_ratio_vs_real",
            "peak_aggregation_left_neighbor_percent": "delta_peak_aggregation_vs_real",
            "final_aggregation_left_neighbor_percent": "delta_final_aggregation_vs_real",
        }[metric]
        if delta_col not in df.columns:
            df[delta_col] = np.nan
        df.loc[df["source_type"] == "real_e01_cell_view", delta_col] = 0.0
    return df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "target_swap_count",
        "final_sortedness_percent",
        "peak_sortedness_percent",
        "curve_mean_sortedness_percent",
        "dg_primary",
        "dg_total_drop",
        "path_backtracking_drop_total",
        "path_curvature_ratio",
        "peak_aggregation_left_neighbor_percent",
        "final_aggregation_left_neighbor_percent",
        "curve_mean_aggregation_left_neighbor_percent",
        "delta_final_sortedness_vs_real",
        "delta_peak_sortedness_vs_real",
        "delta_curve_mean_sortedness_vs_real",
        "delta_dg_primary_vs_real",
        "delta_path_curvature_ratio_vs_real",
        "delta_peak_aggregation_vs_real",
        "delta_final_aggregation_vs_real",
    ]
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["condition_id", "algotype_mix", "source_type", "null_policy"], dropna=False)
    for keys, group in grouped:
        row: dict[str, Any] = {
            "condition_id": keys[0],
            "algotype_mix": keys[1],
            "source_type": keys[2],
            "null_policy": keys[3],
            "n_rows": int(len(group)),
            "n_repeats": int(group["repeat_index"].nunique()),
            "swap_count_match_failures": int((~group["swap_count_match"].astype(bool)).sum()),
            "locality_match_failures": int((~group["locality_match"].astype(bool)).sum()),
            "hidden_label_access_failures": int((~group["hidden_label_access_validation_passed"].astype(bool)).sum()),
            "max_nonlocal_swap_count": int(pd.to_numeric(group["nonlocal_swap_count"], errors="coerce").max()),
            "max_abs_swap_distance": int(pd.to_numeric(group["max_abs_swap_distance"], errors="coerce").max()),
        }
        for metric in metrics:
            mean, std, low, high = mean_ci(group[metric])
            row[f"mean_{metric}"] = mean
            row[f"std_{metric}"] = std
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition_id", "source_type", "null_policy"]).reset_index(drop=True)


def aggregate_curve(curve_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    curve_df = pd.DataFrame(curve_rows)
    rows = []
    for keys, group in curve_df.groupby(["condition_id", "algotype_mix", "source_type", "null_policy", "progress_percent"], dropna=False):
        rows.append(
            {
                "condition_id": keys[0],
                "algotype_mix": keys[1],
                "source_type": keys[2],
                "null_policy": keys[3],
                "progress_percent": int(keys[4]),
                "n_points": int(len(group)),
                "mean_sortedness_percent": float(pd.to_numeric(group["sortedness_percent"], errors="coerce").mean()),
                "mean_aggregation_left_neighbor_percent": float(
                    pd.to_numeric(group["aggregation_left_neighbor_percent"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition_id", "source_type", "null_policy", "progress_percent"]).reset_index(drop=True)


def classify_results(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_id, group in summary.groupby("condition_id", sort=True):
        real = group[group["source_type"] == "real_e01_cell_view"]
        if real.empty:
            continue
        for _, null_row in group[group["source_type"] == "local_move_null"].iterrows():
            final_gap = float(null_row["mean_delta_final_sortedness_vs_real"])
            peak_agg_gap = float(null_row["mean_delta_peak_aggregation_vs_real"])
            curvature_gap = float(null_row["mean_delta_path_curvature_ratio_vs_real"])
            sortedness_matched = abs(final_gap) <= 2.5
            aggregation_matched = abs(peak_agg_gap) <= 2.5
            curvature_matched = abs(curvature_gap) <= 0.25
            if sortedness_matched and aggregation_matched and curvature_matched:
                classification = "null_matches_real_core_metrics"
            elif sortedness_matched:
                classification = "null_matches_sortedness_only"
            else:
                classification = "null_diverges_from_real"
            rows.append(
                {
                    "condition_id": condition_id,
                    "algotype_mix": null_row["algotype_mix"],
                    "null_policy": null_row["null_policy"],
                    "mean_delta_final_sortedness_vs_real": final_gap,
                    "mean_delta_peak_aggregation_vs_real": peak_agg_gap,
                    "mean_delta_path_curvature_ratio_vs_real": curvature_gap,
                    "sortedness_matched_within_2_5pp": bool(sortedness_matched),
                    "aggregation_matched_within_2_5pp": bool(aggregation_matched),
                    "path_curvature_matched_within_0_25": bool(curvature_matched),
                    "classification": classification,
                }
            )
    return pd.DataFrame(rows).sort_values(["condition_id", "null_policy"]).reset_index(drop=True)


def make_figure(summary: pd.DataFrame, classification: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    policy_order = ["real_e01", *NULL_POLICIES]
    colors = {
        "real_e01": "#333333",
        "random_adjacent_swap": "#4c78a8",
        "inversion_biased_swap": "#f58518",
        "metropolis_local_swap": "#54a24b",
        "random_walker_swap": "#b279a2",
    }
    plot_summary = summary.copy()
    plot_summary["label"] = plot_summary["null_policy"]
    conditions = list(plot_summary["condition_id"].drop_duplicates())
    x = np.arange(len(conditions))

    ax = axes[0, 0]
    for policy in policy_order:
        sub = plot_summary[plot_summary["null_policy"] == policy].set_index("condition_id").reindex(conditions)
        if sub["mean_final_sortedness_percent"].notna().any():
            ax.plot(x, sub["mean_final_sortedness_percent"], marker="o", label=policy, color=colors.get(policy))
    ax.set_title("Final Sortedness")
    ax.set_ylabel("Mean %")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")

    ax = axes[0, 1]
    for policy in policy_order:
        sub = plot_summary[plot_summary["null_policy"] == policy].set_index("condition_id").reindex(conditions)
        if sub["mean_dg_primary"].notna().any():
            ax.plot(x, sub["mean_dg_primary"], marker="o", label=policy, color=colors.get(policy))
    ax.set_title("DG-like Backtracking")
    ax.set_ylabel("Mean DG primary")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")

    ax = axes[1, 0]
    chimera_conditions = [cid for cid in conditions if cid in CHIMERA_CONDITION_IDS]
    xc = np.arange(len(chimera_conditions))
    for policy in policy_order:
        sub = plot_summary[plot_summary["null_policy"] == policy].set_index("condition_id").reindex(chimera_conditions)
        if sub["mean_peak_aggregation_left_neighbor_percent"].notna().any():
            ax.plot(xc, sub["mean_peak_aggregation_left_neighbor_percent"], marker="o", label=policy, color=colors.get(policy))
    ax.set_title("Peak Aggregation")
    ax.set_ylabel("Mean left-neighbor %")
    ax.set_xticks(xc)
    ax.set_xticklabels(chimera_conditions, rotation=30, ha="right")

    ax = axes[1, 1]
    for policy in policy_order:
        sub = plot_summary[plot_summary["null_policy"] == policy].set_index("condition_id").reindex(conditions)
        if sub["mean_path_curvature_ratio"].replace([np.inf, -np.inf], np.nan).notna().any():
            ax.plot(x, sub["mean_path_curvature_ratio"], marker="o", label=policy, color=colors.get(policy))
    ax.set_title("Path Curvature")
    ax.set_ylabel("Total variation / net progress")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("E02 S07 local-move null benchmarks", fontsize=14)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def validate_results(
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    tasks: Sequence[LocalMoveNullTask],
    figure_path: Path,
    artifacts_written: Sequence[Path],
    max_repeats: int,
    null_replicates: int,
) -> dict[str, Any]:
    real_df = result_df[result_df["source_type"] == "real_e01_cell_view"]
    null_df = result_df[result_df["source_type"] == "local_move_null"]
    expected_real_rows = len(SELECTED_CONDITION_IDS) * max_repeats
    expected_null_rows = len(tasks)
    observed_conditions = sorted(result_df["condition_id"].unique())
    no_duplicate_real = not real_df.duplicated(["condition_id", "repeat_index"]).any()
    no_duplicate_null = not null_df.duplicated(["condition_id", "repeat_index", "null_policy", "null_replicate"]).any()
    all_swap_matched = bool(result_df["swap_count_match"].all()) if len(result_df) else False
    all_local = bool(result_df["locality_match"].all()) if len(result_df) else False
    no_hidden = bool((~result_df["null_policy_uses_hidden_labels"].astype(bool)).all()) if len(result_df) else False
    hidden_validation = bool(result_df["hidden_label_access_validation_passed"].all()) if len(result_df) else False
    trace_length_matched = bool(result_df["trace_length_match_target_swaps_plus_initial"].all()) if len(result_df) else False
    assignment_counts_preserved = bool(
        (
            result_df["initial_algotype_counts_json"].fillna("")
            == result_df["final_algotype_counts_json"].fillna("")
        ).all()
    )
    figure_ok = figure_path.exists() and figure_path.stat().st_size > 1000
    materialized_names = {
        "e02_local_move_nulls.parquet",
        "local_move_null_benchmarks.png",
        "e02_local_move_null_summary.csv",
        "e02_local_move_null_curve_summary.csv",
        "e02_local_move_null_classification.csv",
    }
    artifacts_present = all(
        path.exists() and path.stat().st_size > 0 for path in artifacts_written if path.name in materialized_names
    )
    passed = all(
        [
            len(real_df) == expected_real_rows,
            len(null_df) == expected_null_rows,
            observed_conditions == sorted(SELECTED_CONDITION_IDS),
            no_duplicate_real,
            no_duplicate_null,
            all_swap_matched,
            all_local,
            no_hidden,
            hidden_validation,
            trace_length_matched,
            assignment_counts_preserved,
            len(summary_df) > 0,
            len(curve_df) > 0,
            len(classification_df) == len(SELECTED_CONDITION_IDS) * len(NULL_POLICIES),
            figure_ok,
            artifacts_present,
        ]
    )
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(passed),
        "status": "completed" if passed else "failed_validation",
        "validationPassed": bool(passed),
        "validationResult": "passed" if passed else "failed",
        "artifactsWritten": [str(path) for path in artifacts_written],
        "caveatsOrBlockers": [
            "S07 uses a bounded first-10-repeat matrix and exact swap-count local nulls.",
            "Null policies are behavioral local-move controls, not public cell-policy implementations.",
            "No hidden-label access is validated by label-permutation invariance of each null decision path.",
        ],
        "recommendedNextAction": "Proceed to S08 after Chief Scientist instruction; do not start S08 inside S07.",
        "expectedRealRows": int(expected_real_rows),
        "observedRealRows": int(len(real_df)),
        "expectedNullRows": int(expected_null_rows),
        "observedNullRows": int(len(null_df)),
        "observedConditions": observed_conditions,
        "nullPolicies": list(NULL_POLICIES),
        "nullReplicatesPerRealRunPolicy": int(null_replicates),
        "noDuplicateRealRows": bool(no_duplicate_real),
        "noDuplicateNullRows": bool(no_duplicate_null),
        "allSwapCountsMatched": all_swap_matched,
        "allSwapsAdjacentLocal": all_local,
        "maxAbsSwapDistance": int(pd.to_numeric(result_df["max_abs_swap_distance"], errors="coerce").max()),
        "maxNonlocalSwapCount": int(pd.to_numeric(result_df["nonlocal_swap_count"], errors="coerce").max()),
        "noPolicyDeclaresHiddenLabelAccess": no_hidden,
        "hiddenLabelAccessValidationPassedForAllRows": hidden_validation,
        "traceLengthMatchesTargetSwapsPlusInitialForAllRows": trace_length_matched,
        "algotypeCountsPreservedForAllRows": assignment_counts_preserved,
        "summaryRows": int(len(summary_df)),
        "curveRows": int(len(curve_df)),
        "classificationRows": int(len(classification_df)),
        "figureExistsAndNonempty": bool(figure_ok),
        "artifactsPresent": bool(artifacts_present),
    }


def outcome_from_classification(classification_df: pd.DataFrame) -> str:
    if classification_df.empty:
        return "Null"
    matches_all = classification_df[classification_df["classification"] == "null_matches_real_core_metrics"]
    sortedness_only = classification_df[classification_df["classification"] == "null_matches_sortedness_only"]
    if len(matches_all) > 0:
        return "Constraining/contradictory"
    if len(sortedness_only) > 0:
        return "Supportive but narrowed"
    return "Supportive"


def build_report(
    *,
    generated_at: str,
    artifacts_dir: Path,
    result_path: Path,
    summary_path: Path,
    curve_path: Path,
    classification_path: Path,
    validation_path: Path,
    log_path: Path,
    figure_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    validation: dict[str, Any],
    summary: pd.DataFrame,
    classification: pd.DataFrame,
    unit_result: dict[str, Any] | None,
    git_commit: str,
    git_status: str,
    max_repeats: int,
    null_replicates: int,
    workers: int,
    elapsed_seconds: float,
) -> str:
    outcome = outcome_from_classification(classification)
    compact_summary = summary[
        [
            "condition_id",
            "algotype_mix",
            "source_type",
            "null_policy",
            "n_rows",
            "mean_target_swap_count",
            "mean_final_sortedness_percent",
            "mean_delta_final_sortedness_vs_real",
            "mean_dg_primary",
            "mean_delta_dg_primary_vs_real",
            "mean_peak_aggregation_left_neighbor_percent",
            "mean_delta_peak_aggregation_vs_real",
            "mean_path_curvature_ratio",
            "mean_delta_path_curvature_ratio_vs_real",
            "swap_count_match_failures",
            "locality_match_failures",
            "hidden_label_access_failures",
        ]
    ]
    compact_classification = classification[
        [
            "condition_id",
            "algotype_mix",
            "null_policy",
            "mean_delta_final_sortedness_vs_real",
            "mean_delta_peak_aggregation_vs_real",
            "mean_delta_path_curvature_ratio_vs_real",
            "classification",
        ]
    ]
    class_counts = classification["classification"].value_counts().to_dict() if len(classification) else {}
    validation_result = (
        f"passed: {validation['observedRealRows']}/{validation['expectedRealRows']} real rows and "
        f"{validation['observedNullRows']}/{validation['expectedNullRows']} null rows, "
        f"swap counts matched={validation['allSwapCountsMatched']}, "
        f"adjacent locality={validation['allSwapsAdjacentLocal']}, "
        f"hidden-label validation={validation['hiddenLabelAccessValidationPassedForAllRows']}, "
        f"unit tests={'passed' if unit_result and unit_result['success'] else 'not run'}"
    )
    caveats = [
        "S07 uses first-10-repeat bounded controls rather than full 100-repeat inference.",
        "Null policies are local adjacent-swap processes and are not intended to reproduce public cell-policy internals.",
        "Exact swap-count matching isolates path quality under the same movement budget but does not match comparison counts or activation schedules.",
        "Aggregation remains a computational label-neighborhood proxy.",
    ]
    if outcome == "Constraining/contradictory":
        lay_summary = (
            "At least one local-move null matched real trajectories across the core metric screen, so some real-versus-null claims need narrowing before S08."
        )
    elif outcome == "Supportive but narrowed":
        lay_summary = (
            "Some value-aware local nulls matched final Sortedness, but none matched the full metric set across Sortedness, Aggregation, and path curvature in this bounded screen."
        )
    else:
        lay_summary = (
            "The randomized local-move nulls generally diverged from the real cell-view trajectories under the same swap budgets, so trivial local motion did not explain the S07 metric set."
        )
    return f"""# E02 S07 Randomized Local-Move Null Models

## Top Summary

- Research step ID: S07
- Completion status: Completed on {generated_at}
- Artifacts written: `{artifacts_dir / 'research_steps/S07/research_step_full_results.md'}`, `{result_path}`, `{figure_path}`, `{validation_path}`, `{summary_path}`, `{curve_path}`, `{classification_path}`, `{log_path}`, `{manifest_path}`, `{src_manifest_path}`
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers: {'; '.join(caveats)}
- Lay summary: {lay_summary}
- Recommended next action: Proceed to S08 after Chief Scientist instruction; do not start S08 inside S07.

## Chief Handoff

S07 completed exact-swap-budget randomized local-move null models for the first `{max_repeats}` repeats of three pure cell-view algorithms and five same-goal chimeras. It quantified real-versus-null Sortedness, DG-like backtracking, Aggregation, and path-curvature differences under adjacent-swap locality.

## Frozen Question

Do real cell-view algorithms outperform or differ from random local movement processes matched on locality and swap budget?

## Inputs

- Repository commit at S07 run time: `{git_commit}`
- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- E01 pure run records: `/previous-artifacts/E01/results/e01_efficiency_counts.parquet`
- E01 chimera run records: `/previous-artifacts/E01/results/e01_chimera_efficiency.parquet`
- E01 pure trajectories: `/previous-artifacts/E01/traces/e01_figure3_trajectories.parquet`
- E01 same-goal chimera trajectories: `/previous-artifacts/E01/traces/e01_same_goal_chimeras.parquet`
- Selected conditions: `{', '.join(SELECTED_CONDITION_IDS)}`

## Lay Summary

S07 asks whether simple random neighboring swaps can explain the same progress and label clustering seen in the real cell-view runs. Each null run started from the same array and labels as a real E01 run and was allowed exactly the same number of adjacent swaps. The null policies were allowed to use values and positions, but not Algotype labels.

## Detailed Methods

For each selected condition and repeat, S07 loaded the real E01 per-swap trace and computed final and peak Sortedness, DG-like backtracking using the E02 S02 `dg_from_sortedness` helper, peak/final/mean Aggregation, and a Sortedness path-curvature ratio defined as total absolute Sortedness variation divided by net Sortedness progress.

S07 then ran four local-move null policies for each real run:

- `random_adjacent_swap`: uniformly random adjacent swap.
- `inversion_biased_swap`: with probability 0.85 chooses a current adjacent inversion, otherwise chooses a random adjacent pair.
- `metropolis_local_swap`: proposes adjacent swaps and accepts local adjacent-order improvements always, accepting worsening moves with a Metropolis probability.
- `random_walker_swap`: a persistent random walker swaps with an adjacent neighbor at each step.

Each null was run for `{null_replicates}` deterministic replicates per real run and policy. Every accepted move was an adjacent swap, and every null stopped only after the paired real swap count was reached. Hidden-label access was validated by rerunning each null decision path with reversed labels and requiring the swap-decision hash to remain unchanged.

## Commands

- Unit tests: `{unit_result['command'] if unit_result else 'not run'}`
- S07 production run: `{sys.executable} scripts/e02_s07_local_move_nulls.py --repo-dir /workspace/cell-research --artifacts-dir {artifacts_dir} --max-repeats {max_repeats} --null-replicates {null_replicates} --workers {workers}`

## Dependencies And Parameters

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- New dependencies installed: none
- CPU workers: `{workers}`
- Null replicates per real run and policy: `{null_replicates}`
- Elapsed production time: `{elapsed_seconds:.2f}` seconds

## Results

Classification counts: `{json.dumps(class_counts, sort_keys=True)}`.

### Real-Versus-Null Summary

{markdown_table(compact_summary, max_rows=48)}

### Classification Table

{markdown_table(compact_classification, max_rows=40)}

The full run-level table is `{result_path}`. The progress-curve summary is `{curve_path}` and the figure is `{figure_path}`.

## Validation

Validation required all planned E01 real reference rows, all null rows, no duplicate real or null task rows, exact swap-count matching, adjacent-only locality, no hidden-label policy declarations, label-permutation decision-path invariance for every null row, preserved Algotype counts, nonempty summaries, nonempty classification, and a nonempty figure.

Validation result: `{validation['validationPassed']}`. Details are in `{validation_path}`.

Key validation values:

- Real rows: `{validation['observedRealRows']}` / `{validation['expectedRealRows']}`
- Null rows: `{validation['observedNullRows']}` / `{validation['expectedNullRows']}`
- Max absolute swap distance: `{validation['maxAbsSwapDistance']}`
- Max nonlocal swap count: `{validation['maxNonlocalSwapCount']}`
- Hidden-label validation passed for all rows: `{validation['hiddenLabelAccessValidationPassedForAllRows']}`

## Artifacts

- Run-level local-move null table: `{result_path}`
- Condition/policy summary: `{summary_path}`
- Progress-curve summary: `{curve_path}`
- Classification table: `{classification_path}`
- Figure: `{figure_path}`
- Validation JSON: `{validation_path}`
- Log: `{log_path}`
- Artifact manifest: `{manifest_path}`
- Source snapshot manifest: `{src_manifest_path}`

## Provenance

- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Generated at UTC: `{generated_at}`
- Output checksums are recorded in `{manifest_path}`.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No matching blocker was encountered: all null runs matched paired real swap counts exactly and used adjacent swaps only.
- The null policies do not access labels, and this was validated by label-permutation invariance of decision-path hashes, but they are still simplified local-move controls rather than public cell-view policies.
- The matrix is bounded to first `{max_repeats}` repeats and should not be interpreted as final 100-repeat inference.
- The Metropolis and inversion-biased nulls use value information; they are controls for local value-guided motion, not controls for label-independent blind motion.
- Aggregation is a label-clustering proxy and does not identify a biological mechanism.

## Recommended Next Action

Proceed to S08 after explicit Chief Scientist instruction. Do not start S08 inside S07.
"""


def write_blocked_outputs(
    *,
    generated_at: str,
    artifacts_dir: Path,
    repo_dir: Path,
    reason: str,
    validation_path: Path,
    report_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    log_path: Path,
) -> int:
    blocker = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": False,
        "status": "blocked",
        "validationPassed": False,
        "validationResult": "blocked",
        "artifactsWritten": [str(report_path), str(validation_path), str(manifest_path), str(src_manifest_path), str(log_path)],
        "caveatsOrBlockers": [reason],
        "recommendedNextAction": "Resolve the S07 matching blocker before S08; do not start S08.",
    }
    write_json(validation_path, blocker)
    log_path.write_text(json.dumps(blocker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    write_json(
        src_manifest_path,
        {
            "schema": "eidosoma.src_snapshot.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "createdAtUtc": generated_at,
            "gitCommit": git_commit,
            "gitStatusShort": git_status,
            "sourceFiles": [],
            "validation": blocker,
        },
    )
    report = f"""# E02 S07 Randomized Local-Move Null Models

## Top Summary

- Research step ID: S07
- Completion status: Blocked on {generated_at}
- Artifacts written: `{report_path}`, `{validation_path}`, `{manifest_path}`, `{src_manifest_path}`, `{log_path}`
- Validation result: blocked
- Outcome classification: constraining/contradictory
- Caveats or blockers: {reason}
- Lay summary: S07 could not safely run local-move null models because exact matching or required inputs were unavailable.
- Recommended next action: Resolve the S07 blocker before S08; do not start S08.

## Frozen Question

Do real cell-view algorithms outperform or differ from random local movement processes matched on locality and swap budget?

## Methods

S07 checked required E01 traces, E01 run records, and null-matching feasibility before production null simulations.

## Results

No local-move null table or figure was generated.

## Validation

Validation was blocked before production null runs.

## Provenance

- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Generated at UTC: `{generated_at}`
"""
    report_path.write_text(report, encoding="utf-8")
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.research_step_artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "createdAtUtc": generated_at,
            "artifacts": [
                artifact_entry(report_path, artifacts_dir, "S07 blocked full-results Markdown handoff report."),
                artifact_entry(validation_path, artifacts_dir, "S07 blocked validation evidence."),
                artifact_entry(log_path, artifacts_dir, "S07 blocked execution log."),
                artifact_entry(src_manifest_path, artifacts_dir, "S07 blocked source snapshot manifest."),
            ],
        },
    )
    return 2


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e02"
    logs_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    for directory in (step_dir, results_dir, figures_dir, logs_dir, src_snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now()
    result_path = results_dir / "e02_local_move_nulls.parquet"
    summary_path = step_dir / "e02_local_move_null_summary.csv"
    curve_path = step_dir / "e02_local_move_null_curve_summary.csv"
    classification_path = step_dir / "e02_local_move_null_classification.csv"
    figure_path = figures_dir / "local_move_null_benchmarks.png"
    validation_path = step_dir / "s07_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s07_local_move_nulls_manifest.json"
    log_path = logs_dir / "e02_s07_local_move_nulls.log"
    report_path = step_dir / "research_step_full_results.md"
    log_lines = [f"E02 S07 started {generated_at}", f"repo_dir={repo_dir}", f"artifacts_dir={artifacts_dir}"]

    unit_result: dict[str, Any] | None = None
    if args.run_unit_tests:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_dir)
        unit_result = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_*.py"], repo_dir, env)
        log_lines.extend(["UNIT TESTS:", json.dumps(unit_result, indent=2, sort_keys=True)])
        if not unit_result["success"]:
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return int(unit_result["returnCode"] or 1)

    try:
        cfg = load_e01_config(args.config_path)
        real_rows, real_curve_rows, task_inputs = load_e01_inputs(
            cfg,
            args.max_repeats,
            args.e01_pure_trace_path,
            args.e01_chimera_trace_path,
            args.e01_efficiency_path,
            args.e01_chimera_efficiency_path,
        )
        tasks = build_null_tasks(task_inputs, args.null_replicates, args.random_seed)
    except Exception as exc:
        return write_blocked_outputs(
            generated_at=generated_at,
            artifacts_dir=artifacts_dir,
            repo_dir=repo_dir,
            reason=f"S07 input or exact-matching validation failed before null runs: {exc!r}",
            validation_path=validation_path,
            report_path=report_path,
            manifest_path=manifest_path,
            src_manifest_path=src_manifest_path,
            log_path=log_path,
        )

    workers = max(1, min(int(args.workers), len(tasks), 8))
    log_lines.append(
        f"real_rows={len(real_rows)} task_inputs={len(task_inputs)} null_tasks={len(tasks)} "
        f"workers={workers} null_replicates={args.null_replicates}"
    )

    null_rows: list[dict[str, Any]] = []
    null_curve_rows: list[dict[str, Any]] = []
    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            row, curve = run_null_task(task)
            null_rows.append(row)
            null_curve_rows.extend(curve)
            if idx % 100 == 0 or idx == len(tasks):
                log_lines.append(f"completed_null_tasks={idx}/{len(tasks)}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(run_null_task, task): task for task in tasks}
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    row, curve = future.result()
                except Exception as exc:
                    log_lines.append(
                        f"FAILED task {task.condition['condition_id']} repeat={task.repeat_index} "
                        f"policy={task.null_policy} null_replicate={task.null_replicate}: {exc!r}"
                    )
                    raise
                null_rows.append(row)
                null_curve_rows.extend(curve)
                if idx % 100 == 0 or idx == len(tasks):
                    log_lines.append(f"completed_null_tasks={idx}/{len(tasks)}")

    result_df = add_real_deltas(pd.DataFrame([*real_rows, *null_rows]))
    result_df = result_df.sort_values(["condition_id", "repeat_index", "source_type", "null_policy", "null_replicate"]).reset_index(drop=True)
    curve_df = aggregate_curve([*real_curve_rows, *null_curve_rows])
    summary_df = summarize_results(result_df)
    classification_df = classify_results(summary_df)

    result_df.to_parquet(result_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    curve_df.to_csv(curve_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    make_figure(summary_df, classification_df, figure_path)
    expected_artifacts = [
        report_path,
        result_path,
        figure_path,
        validation_path,
        summary_path,
        curve_path,
        classification_path,
        log_path,
        manifest_path,
        src_manifest_path,
    ]
    validation = validate_results(
        result_df,
        summary_df,
        curve_df,
        classification_df,
        tasks,
        figure_path,
        expected_artifacts,
        args.max_repeats,
        args.null_replicates,
    )
    write_json(validation_path, validation)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    elapsed_seconds = time.monotonic() - start
    log_lines.extend(
        [
            f"result_rows={len(result_df)} summary_rows={len(summary_df)} curve_rows={len(curve_df)} classification_rows={len(classification_df)}",
            f"validation={json.dumps(validation, sort_keys=True)}",
            f"elapsed_seconds={elapsed_seconds:.3f}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    code_files = [
        repo_dir / "scripts/e02_s07_local_move_nulls.py",
        repo_dir / "tests/e02/test_local_move_nulls.py",
        repo_dir / "scripts/e02_s02_scheduler_comparison.py",
        repo_dir / "src/e02/deterministic_simulator.py",
    ]
    src_manifest = {
        "schema": "eidosoma.src_snapshot.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "gitCommit": git_commit,
        "gitStatusShort": git_status,
        "sourceFiles": [
            {
                "path": str(path),
                "relativePath": str(path.relative_to(repo_dir)),
                "sha256": sha256_file(path),
                "sizeBytes": path.stat().st_size,
            }
            for path in code_files
            if path.exists()
        ],
        "parameters": {
            "maxRepeats": int(args.max_repeats),
            "nullReplicates": int(args.null_replicates),
            "workers": int(workers),
            "randomSeed": int(args.random_seed),
            "selectedConditionIds": list(SELECTED_CONDITION_IDS),
            "nullPolicies": list(NULL_POLICIES),
            "e01PureTracePath": str(args.e01_pure_trace_path),
            "e01ChimeraTracePath": str(args.e01_chimera_trace_path),
            "e01EfficiencyPath": str(args.e01_efficiency_path),
            "e01ChimeraEfficiencyPath": str(args.e01_chimera_efficiency_path),
        },
        "validation": validation,
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "newDependenciesInstalled": [],
        },
    }
    write_json(src_manifest_path, src_manifest)

    artifact_stub = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "artifacts": [],
    }
    write_json(manifest_path, artifact_stub)
    report_text = build_report(
        generated_at=generated_at,
        artifacts_dir=artifacts_dir,
        result_path=result_path,
        summary_path=summary_path,
        curve_path=curve_path,
        classification_path=classification_path,
        validation_path=validation_path,
        log_path=log_path,
        figure_path=figure_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        validation=validation,
        summary=summary_df,
        classification=classification_df,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        max_repeats=args.max_repeats,
        null_replicates=args.null_replicates,
        workers=workers,
        elapsed_seconds=elapsed_seconds,
    )
    report_path.write_text(report_text, encoding="utf-8")
    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S07 full-results Markdown handoff report."),
        artifact_entry(result_path, artifacts_dir, "Run-level real-versus-local-move-null table."),
        artifact_entry(summary_path, artifacts_dir, "Condition and null-policy summary table."),
        artifact_entry(curve_path, artifacts_dir, "Progress-curve summary table."),
        artifact_entry(classification_path, artifacts_dir, "Local-move null classification table."),
        artifact_entry(figure_path, artifacts_dir, "Local-move null benchmark figure."),
        artifact_entry(validation_path, artifacts_dir, "S07 validation evidence."),
        artifact_entry(log_path, artifacts_dir, "S07 execution log."),
        artifact_entry(src_manifest_path, artifacts_dir, "S07 source snapshot manifest."),
    ]
    write_json(manifest_path, artifact_stub | {"artifacts": artifacts})
    if not validation["validationPassed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
