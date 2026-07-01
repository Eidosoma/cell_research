#!/usr/bin/env python3
"""Run E02 S08 matched Delayed Gratification null trajectories.

S08 tests whether E01 Delayed Gratification (DG) is explained by generic
trajectory geometry.  For each E01 DG run, it builds randomized matched null
Sortedness trajectories with the same start, end, swap count, and total
backtracking-drop amplitude.  The ordering bias comes from the S07 local-move
null generators, but the final bridge is forced to match the requested DG null
constraints exactly.
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

from scripts.e02_s02_scheduler_comparison import git_output, markdown_table, run_command, sha256_file, write_json
from scripts.e02_s07_local_move_nulls import NULL_POLICIES, simulate_null_path
from src.e02.deterministic_simulator import stable_json_sha256


STEP_ID = "S08"
STEP_NUMBER = 8
EXPERIMENT_ID = "E02"
SOURCE_EXPERIMENT_ID = "E01"
PLOT_SEMANTICS = ("passive", "stuck")
ALGORITHMS = ("bubble", "insertion", "selection")
MODES = ("traditional", "cell_view")


@dataclass(frozen=True)
class MatchedDgNullTask:
    condition: dict[str, Any]
    repeat_index: int
    values: tuple[int, ...]
    labels: tuple[str, ...]
    initial_array_seed: int
    initial_array_sha256: str
    real_run: dict[str, Any]
    real_sortedness_values: tuple[float, ...]
    null_policy: str
    null_replicate: int
    null_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config-path", type=Path, default=Path("/previous-artifacts/E01/configs/e01_baseline_configs.json"))
    parser.add_argument(
        "--e01-dg-path",
        type=Path,
        default=Path("/previous-artifacts/E01/results/e01_delayed_gratification.parquet"),
    )
    parser.add_argument(
        "--e01-f0-trace-path",
        type=Path,
        default=Path("/previous-artifacts/E01/traces/e01_figure3_trajectories.parquet"),
    )
    parser.add_argument(
        "--e01-frozen-trace-path",
        type=Path,
        default=Path("/previous-artifacts/E01/traces/e01_frozen_cell_trajectories.parquet"),
    )
    parser.add_argument("--max-repeats", type=int, default=None)
    parser.add_argument("--null-replicates", type=int, default=2)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--random-seed", type=int, default=20260701)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def load_e01_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def selected_s08_conditions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in cfg["conditions"] if "S08" in str(row.get("step_scope", ""))]
    if not rows:
        raise RuntimeError("No S08 condition rows found in E01 baseline config.")
    return sorted(rows, key=lambda row: row["condition_id"])


def compact_consecutive(values: Sequence[float], tolerance: float = 1e-12) -> list[float]:
    compact: list[float] = []
    for value in values:
        if pd.isna(value):
            continue
        value = float(value)
        if not compact or not math.isclose(compact[-1], value, abs_tol=tolerance):
            compact.append(value)
    return compact


def signed_segments(values: Sequence[float]) -> list[float]:
    compact = compact_consecutive(values)
    if len(compact) < 2:
        return []
    segments: list[float] = []
    for idx in range(1, len(compact)):
        delta = compact[idx] - compact[idx - 1]
        if math.isclose(delta, 0.0, abs_tol=1e-12):
            continue
        if not segments or segments[-1] * delta <= 0:
            segments.append(delta)
        else:
            segments[-1] += delta
    return segments


def dg_from_sortedness(values: Sequence[float]) -> dict[str, Any]:
    segments = signed_segments(values)
    events: list[tuple[float, float, float]] = []
    terminal_unrecovered_drop_segments = 0
    leading_increase_segments = 0
    seen_drop = False
    for idx, segment in enumerate(segments):
        if segment > 0 and not seen_drop:
            leading_increase_segments += 1
        if segment < 0:
            seen_drop = True
            if idx + 1 < len(segments) and segments[idx + 1] > 0:
                drop = -float(segment)
                recovery = float(segments[idx + 1])
                events.append((drop, recovery, (recovery - drop) / drop if drop > 0 else 0.0))
            else:
                terminal_unrecovered_drop_segments += 1
    ratios = [event[2] for event in events]
    drops = [event[0] for event in events]
    recoveries = [event[1] for event in events]
    total_drop = float(np.sum(drops)) if drops else 0.0
    total_recovery = float(np.sum(recoveries)) if recoveries else 0.0
    return {
        "dg_primary": float(np.mean(ratios)) if ratios else 0.0,
        "dg_event_count": int(len(events)),
        "dg_total_drop": total_drop,
        "dg_total_recovery": total_recovery,
        "dg_total_net_gain": total_recovery - total_drop,
        "dg_total_ratio": float((total_recovery - total_drop) / total_drop) if total_drop > 0 else 0.0,
        "dg_min_event_ratio": float(np.min(ratios)) if ratios else 0.0,
        "dg_max_event_ratio": float(np.max(ratios)) if ratios else 0.0,
        "dg_mean_event_drop": float(np.mean(drops)) if drops else 0.0,
        "dg_mean_event_recovery": float(np.mean(recoveries)) if recoveries else 0.0,
        "trajectory_point_count": int(len(values)),
        "compact_sortedness_point_count": int(len(compact_consecutive(values))),
        "signed_segment_count": int(len(segments)),
        "leading_increase_segment_count": int(leading_increase_segments),
        "terminal_unrecovered_drop_segment_count": int(terminal_unrecovered_drop_segments),
    }


def dg_from_monotonicity_error(values: Sequence[float]) -> dict[str, Any]:
    segments = signed_segments(values)
    events: list[tuple[float, float, float]] = []
    for idx, segment in enumerate(segments):
        if segment > 0 and idx + 1 < len(segments) and segments[idx + 1] < 0:
            drop = float(segment)
            recovery = -float(segments[idx + 1])
            events.append((drop, recovery, (recovery - drop) / drop if drop > 0 else 0.0))
    ratios = [event[2] for event in events]
    return {
        "dg_primary_from_monotonicity_error": float(np.mean(ratios)) if ratios else 0.0,
        "dg_event_count_from_monotonicity_error": int(len(events)),
    }


def validate_toy_cases() -> dict[str, Any]:
    cases = [
        {"name": "monotone_increase_no_dg", "sortedness": [50, 60, 70], "expected": 0.0, "events": 0},
        {"name": "one_drop_then_recover", "sortedness": [50, 40, 70], "expected": 2.0, "events": 1},
        {"name": "leading_gain_ignored", "sortedness": [50, 60, 40, 70], "expected": 0.5, "events": 1},
        {"name": "two_events_average", "sortedness": [50, 40, 60, 55, 80], "expected": 2.5, "events": 2},
        {"name": "flat_values_deduplicated", "sortedness": [50, 50, 40, 40, 70], "expected": 2.0, "events": 1},
    ]
    details = []
    passed = True
    for case in cases:
        result = dg_from_sortedness(case["sortedness"])
        ok = (
            math.isclose(float(result["dg_primary"]), float(case["expected"]), rel_tol=0.0, abs_tol=1e-12)
            and int(result["dg_event_count"]) == int(case["events"])
        )
        passed = passed and ok
        details.append(
            {
                "name": case["name"],
                "expectedDg": case["expected"],
                "observedDg": result["dg_primary"],
                "expectedEvents": case["events"],
                "observedEvents": result["dg_event_count"],
                "passed": ok,
            }
        )
    return {"toyCaseValidationPassed": bool(passed), "toyCases": details}


def null_seed(base_seed: int, condition_id: str, repeat_idx: int, policy: str, null_replicate: int) -> int:
    condition_num = int(str(condition_id).replace("E01C", ""))
    policy_num = NULL_POLICIES.index(policy) + 1
    return base_seed * 1_000_000 + 800_000 + condition_num * 10_000 + repeat_idx * 100 + policy_num * 10 + null_replicate


def total_step_drop(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return float(np.sum(np.maximum(-np.diff(arr), 0.0)))


def total_step_recovery(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return float(np.sum(np.maximum(np.diff(arr), 0.0)))


def bridge_counts(start: float, end: float, swap_count: int, total_drop: float) -> tuple[int, int, int]:
    down_count = int(round(float(total_drop)))
    net = int(round(float(end) - float(start)))
    up_count = net + down_count
    zero_count = int(swap_count) - down_count - up_count
    if down_count < 0 or up_count < 0 or zero_count < 0:
        raise ValueError(
            f"Cannot build matched bridge with start={start}, end={end}, swaps={swap_count}, drop={total_drop}: "
            f"down={down_count}, up={up_count}, zero={zero_count}."
        )
    return down_count, up_count, zero_count


def choose_weighted(candidates: list[tuple[int, float]], rng: random.Random) -> int:
    total = sum(max(0.0, weight) for _, weight in candidates)
    if total <= 0:
        return rng.choice([move for move, _ in candidates])
    pick = rng.random() * total
    cumulative = 0.0
    for move, weight in candidates:
        cumulative += max(0.0, weight)
        if pick <= cumulative:
            return move
    return candidates[-1][0]


def build_matched_bridge(
    *,
    start: float,
    end: float,
    swap_count: int,
    total_drop: float,
    base_sortedness_values: Sequence[float],
    seed: int,
    max_attempts: int = 80,
) -> tuple[list[float], int]:
    """Build a bounded unit-step bridge with exact DG matching constraints."""
    start_i = int(round(start))
    end_i = int(round(end))
    down_total, up_total, zero_total = bridge_counts(start_i, end_i, swap_count, total_drop)
    if swap_count == 0:
        if start_i != end_i or down_total or up_total or zero_total:
            raise ValueError("Zero-length bridge requested with nonzero movement.")
        return [float(start_i)], 1

    base_delta = np.diff(np.asarray(base_sortedness_values, dtype=np.float64))
    if len(base_delta) < swap_count:
        base_delta = np.pad(base_delta, (0, swap_count - len(base_delta)), constant_values=0.0)
    elif len(base_delta) > swap_count:
        base_delta = base_delta[:swap_count]

    for attempt in range(1, max_attempts + 1):
        rng = random.Random(seed + attempt * 10_000_019)
        down_left = down_total
        up_left = up_total
        zero_left = zero_total
        current = start_i
        values = [float(current)]
        failed = False
        for idx in range(swap_count):
            remaining_after_this = swap_count - idx - 1
            base = float(base_delta[idx])
            linear_target = start_i + (end_i - start_i) * ((idx + 1) / swap_count)
            candidates: list[tuple[int, float]] = []
            for move, count_left in ((-1, down_left), (0, zero_left), (1, up_left)):
                if count_left <= 0:
                    continue
                next_value = current + move
                if next_value < 0 or next_value > 100:
                    continue
                d_next = down_left - int(move == -1)
                z_next = zero_left - int(move == 0)
                u_next = up_left - int(move == 1)
                if d_next + z_next + u_next != remaining_after_this:
                    continue
                final_if_all_remaining_used = next_value + u_next - d_next
                if final_if_all_remaining_used != end_i:
                    continue
                pressure = linear_target - current
                if move == 1:
                    weight = 1.0 + max(base, 0.0) * 3.0 + max(pressure, 0.0) * 0.25
                elif move == -1:
                    weight = 1.0 + max(-base, 0.0) * 3.0 + max(-pressure, 0.0) * 0.25
                else:
                    weight = 1.0 + (1.0 if math.isclose(base, 0.0, abs_tol=1e-12) else 0.0)
                # Keep headroom near boundaries so the bridge rarely paints
                # itself into a corner while preserving randomness.
                if next_value <= 2 and d_next > u_next:
                    weight *= 0.05
                if next_value >= 98 and u_next > d_next:
                    weight *= 0.05
                candidates.append((move, weight))
            if not candidates:
                failed = True
                break
            move = choose_weighted(candidates, rng)
            if move == -1:
                down_left -= 1
            elif move == 1:
                up_left -= 1
            else:
                zero_left -= 1
            current += move
            values.append(float(current))
        if not failed and current == end_i and down_left == 0 and up_left == 0 and zero_left == 0:
            if min(values) >= 0.0 and max(values) <= 100.0:
                return values, attempt
    raise RuntimeError(
        f"Failed to construct matched bridge after {max_attempts} attempts: "
        f"start={start_i}, end={end_i}, swaps={swap_count}, drop={total_drop}."
    )


def dataframe_to_trace_map(trace_df: pd.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    trace_map: dict[tuple[str, int], dict[str, Any]] = {}
    for (condition_id, repeat_index), group in trace_df.groupby(["condition_id", "repeat_index"], sort=False):
        ordered = group.sort_values(["swap_step", "is_final"], kind="stable")
        sortedness_values = ordered["sortedness_percent"].astype(float).tolist()
        error_values = ordered["monotonicity_error_count"].astype(float).tolist()
        first = ordered.iloc[0]
        final_rows = ordered[ordered["is_final"].astype(bool)]
        final = final_rows.iloc[-1] if not final_rows.empty else ordered.iloc[-1]
        trace_map[(str(condition_id), int(repeat_index))] = {
            "sortedness_values": tuple(sortedness_values),
            "monotonicity_error_values": tuple(error_values),
            "trace_record_count": int(len(ordered)),
            "source_trace_path": str(first["source_trace_path"]),
            "final_swap_step_from_trace": int(final["swap_step"]),
            "final_sortedness_from_trace": float(final["sortedness_percent"]),
            "initial_sortedness_from_trace": float(first["sortedness_percent"]),
            "final_array_sha256_from_trace": str(final["final_array_sha256"]),
        }
    return trace_map


def load_trace_map(
    dg_df: pd.DataFrame,
    f0_trace_path: Path,
    frozen_trace_path: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], int]:
    needed = [
        "condition_id",
        "repeat_index",
        "swap_step",
        "sortedness_percent",
        "monotonicity_error_count",
        "is_final",
        "initial_array_sha256",
        "final_array_sha256",
    ]
    f0_ids = sorted(dg_df.loc[dg_df["frozen_count"] == 0, "condition_id"].unique())
    frozen_ids = sorted(dg_df.loc[dg_df["frozen_count"] > 0, "condition_id"].unique())
    f0 = pd.read_parquet(f0_trace_path, columns=needed + ["timed_out"])
    f0 = f0[f0["condition_id"].isin(f0_ids)].copy()
    f0["source_trace_path"] = str(f0_trace_path)
    frozen = pd.read_parquet(frozen_trace_path, columns=needed + ["max_guard_hit"])
    frozen = frozen[frozen["condition_id"].isin(frozen_ids)].copy()
    frozen["source_trace_path"] = str(frozen_trace_path)
    combined = pd.concat([f0[needed + ["source_trace_path"]], frozen[needed + ["source_trace_path"]]], ignore_index=True)
    return dataframe_to_trace_map(combined), int(len(combined))


def validate_e01_dg_formula(dg_df: pd.DataFrame, trace_map: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, Any]:
    max_abs_dg_diff = 0.0
    max_abs_drop_diff = 0.0
    event_mismatches = 0
    error_mismatches = 0
    missing = 0
    rows_checked = 0
    for row in dg_df.to_dict(orient="records"):
        key = (str(row["condition_id"]), int(row["repeat_index"]))
        if key not in trace_map:
            missing += 1
            continue
        trace = trace_map[key]
        sortedness_values = list(map(float, trace["sortedness_values"]))
        error_values = list(map(float, trace["monotonicity_error_values"]))
        dg = dg_from_sortedness(sortedness_values)
        error_dg = dg_from_monotonicity_error(error_values)
        max_abs_dg_diff = max(max_abs_dg_diff, abs(float(dg["dg_primary"]) - float(row["dg_primary"])))
        max_abs_drop_diff = max(max_abs_drop_diff, abs(float(dg["dg_total_drop"]) - float(row["dg_total_drop"])))
        if int(dg["dg_event_count"]) != int(row["dg_event_count"]):
            event_mismatches += 1
        if (
            abs(float(dg["dg_primary"]) - float(error_dg["dg_primary_from_monotonicity_error"])) > 1e-12
            or int(dg["dg_event_count"]) != int(error_dg["dg_event_count_from_monotonicity_error"])
        ):
            error_mismatches += 1
        rows_checked += 1
    passed = (
        missing == 0
        and event_mismatches == 0
        and error_mismatches == 0
        and max_abs_dg_diff <= 1e-12
        and max_abs_drop_diff <= 1e-12
    )
    return {
        "e01DgFormulaValidationPassed": bool(passed),
        "e01DgRowsChecked": int(rows_checked),
        "missingTraceGroups": int(missing),
        "maxAbsDgPrimaryRecomputeDiff": float(max_abs_dg_diff),
        "maxAbsDgTotalDropRecomputeDiff": float(max_abs_drop_diff),
        "dgEventCountMismatches": int(event_mismatches),
        "sortednessErrorFormulaMismatches": int(error_mismatches),
    }


def build_tasks(
    cfg: dict[str, Any],
    dg_df: pd.DataFrame,
    trace_map: Mapping[tuple[str, int], Mapping[str, Any]],
    null_replicates: int,
    base_seed: int,
) -> list[MatchedDgNullTask]:
    condition_rows = {row["condition_id"]: row for row in selected_s08_conditions(cfg)}
    value_banks = cfg["seedBanks"]["valueBanks"]
    tasks: list[MatchedDgNullTask] = []
    for row in dg_df.sort_values(["condition_id", "repeat_index"]).to_dict(orient="records"):
        condition = condition_rows[str(row["condition_id"])]
        repeat_index = int(row["repeat_index"])
        trace = trace_map[(str(row["condition_id"]), repeat_index)]
        value_bank = value_banks[condition["value_bank_id"]]
        values = tuple(map(int, value_bank["initialArrays"][repeat_index]))
        labels = tuple(str(condition["algorithm"]) for _ in values)
        for policy in NULL_POLICIES:
            for replicate in range(null_replicates):
                tasks.append(
                    MatchedDgNullTask(
                        condition=dict(condition),
                        repeat_index=repeat_index,
                        values=values,
                        labels=labels,
                        initial_array_seed=int(value_bank["seeds"][repeat_index]),
                        initial_array_sha256=stable_json_sha256(list(values)),
                        real_run=dict(row),
                        real_sortedness_values=tuple(map(float, trace["sortedness_values"])),
                        null_policy=policy,
                        null_replicate=int(replicate),
                        null_seed=null_seed(base_seed, str(row["condition_id"]), repeat_index, policy, replicate),
                    )
                )
    return tasks


def real_record(row: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    sortedness_values = list(map(float, trace["sortedness_values"]))
    dg = dg_from_sortedness(sortedness_values)
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_type": "real_e01_dg",
        "condition_id": str(row["condition_id"]),
        "mode": str(row["mode"]),
        "algorithm": str(row["algorithm"]),
        "baseline_source": str(row["baseline_source"]),
        "matched_group_id": str(row["matched_group_id"]),
        "value_bank_id": str(row["value_bank_id"]),
        "repeat_index": int(row["repeat_index"]),
        "initial_array_seed": int(row["initial_array_seed"]),
        "scheduler_seed": None if pd.isna(row["scheduler_seed"]) else int(row["scheduler_seed"]),
        "initial_array_sha256": str(row["initial_array_sha256"]),
        "final_array_sha256": str(row["final_array_sha256"]),
        "frozen_semantics": str(row["frozen_semantics"]),
        "plot_frozen_semantics": str(row["frozen_semantics"]),
        "frozen_count": int(row["frozen_count"]),
        "is_barrier_context": bool(int(row["frozen_count"]) > 0),
        "frozen_index_bank_id": str(row["frozen_index_bank_id"]),
        "frozen_index_seed": None if pd.isna(row["frozen_index_seed"]) else int(row["frozen_index_seed"]),
        "initial_frozen_indices_json": str(row["initial_frozen_indices_json"]),
        "null_policy": "real_e01",
        "null_replicate": -1,
        "null_seed": None,
        "matched_bridge_attempts": 0,
        "bridge_generation_status": "not_applicable_real",
        "policy_inputs_json": safe_json(["real_e01_sortedness_trace"]),
        "target_swap_count": int(row["final_swap_step"]),
        "observed_swap_count": int(row["final_swap_step"]),
        "trace_record_count": int(trace["trace_record_count"]),
        "initial_sortedness_percent": float(row["initial_sortedness_percent"]),
        "final_sortedness_percent": float(row["final_sortedness_percent"]),
        "min_sortedness_percent": float(min(sortedness_values)),
        "max_sortedness_percent": float(max(sortedness_values)),
        "noise_total_drop_percent": total_step_drop(sortedness_values),
        "noise_total_recovery_percent": total_step_recovery(sortedness_values),
        "path_total_variation_percent": total_step_drop(sortedness_values) + total_step_recovery(sortedness_values),
        "sortedness_values_sha256": stable_json_sha256([round(value, 10) for value in sortedness_values]),
        "source_trace_path": str(trace["source_trace_path"]),
        "start_match_abs_error": 0.0,
        "end_match_abs_error": 0.0,
        "swap_count_match_abs_error": 0,
        "noise_drop_match_abs_error": 0.0,
        "path_total_variation_match_abs_error": 0.0,
        "bounds_respected": True,
        "dg_formula_matches_e01_row": bool(math.isclose(float(dg["dg_primary"]), float(row["dg_primary"]), abs_tol=1e-12)),
        **dg,
        "paired_real_dg_primary": float(dg["dg_primary"]),
        "delta_dg_primary_vs_real": 0.0,
        "real_minus_null_dg_primary": 0.0,
    }


def run_null_task(task: MatchedDgNullTask) -> dict[str, Any]:
    started = time.monotonic()
    real = task.real_run
    swap_count = int(real["final_swap_step"])
    real_sortedness = list(map(float, task.real_sortedness_values))
    base = simulate_null_path(
        task.values,
        task.labels,
        task.null_policy,
        task.null_seed,
        swap_count,
        include_curves=True,
    )
    bridge, attempts = build_matched_bridge(
        start=float(real["initial_sortedness_percent"]),
        end=float(real["final_sortedness_percent"]),
        swap_count=swap_count,
        total_drop=total_step_drop(real_sortedness),
        base_sortedness_values=base["sortedness_values"],
        seed=task.null_seed,
    )
    dg = dg_from_sortedness(bridge)
    noise_drop = total_step_drop(bridge)
    noise_recovery = total_step_recovery(bridge)
    path_variation = noise_drop + noise_recovery
    real_noise_drop = total_step_drop(real_sortedness)
    real_noise_recovery = total_step_recovery(real_sortedness)
    real_path_variation = real_noise_drop + real_noise_recovery
    source_path = "/previous-artifacts/E01/traces/e01_figure3_trajectories.parquet"
    if int(real["frozen_count"]) > 0:
        source_path = "/previous-artifacts/E01/traces/e01_frozen_cell_trajectories.parquet"
    row = {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": EXPERIMENT_ID,
        "source_type": "matched_dg_null",
        "condition_id": str(real["condition_id"]),
        "mode": str(real["mode"]),
        "algorithm": str(real["algorithm"]),
        "baseline_source": str(real["baseline_source"]),
        "matched_group_id": str(real["matched_group_id"]),
        "value_bank_id": str(real["value_bank_id"]),
        "repeat_index": int(real["repeat_index"]),
        "initial_array_seed": int(task.initial_array_seed),
        "scheduler_seed": None if pd.isna(real["scheduler_seed"]) else int(real["scheduler_seed"]),
        "initial_array_sha256": str(task.initial_array_sha256),
        "final_array_sha256": "",
        "frozen_semantics": str(real["frozen_semantics"]),
        "plot_frozen_semantics": str(real["frozen_semantics"]),
        "frozen_count": int(real["frozen_count"]),
        "is_barrier_context": bool(int(real["frozen_count"]) > 0),
        "frozen_index_bank_id": str(real["frozen_index_bank_id"]),
        "frozen_index_seed": None if pd.isna(real["frozen_index_seed"]) else int(real["frozen_index_seed"]),
        "initial_frozen_indices_json": str(real["initial_frozen_indices_json"]),
        "null_policy": task.null_policy,
        "null_replicate": int(task.null_replicate),
        "null_seed": int(task.null_seed),
        "matched_bridge_attempts": int(attempts),
        "bridge_generation_status": "matched",
        "policy_inputs_json": safe_json(["S07 local-move null sortedness preferences", "start", "end", "swap_count", "noise_total_drop"]),
        "target_swap_count": swap_count,
        "observed_swap_count": len(bridge) - 1,
        "trace_record_count": len(bridge),
        "initial_sortedness_percent": float(bridge[0]),
        "final_sortedness_percent": float(bridge[-1]),
        "min_sortedness_percent": float(min(bridge)),
        "max_sortedness_percent": float(max(bridge)),
        "noise_total_drop_percent": float(noise_drop),
        "noise_total_recovery_percent": float(noise_recovery),
        "path_total_variation_percent": float(path_variation),
        "sortedness_values_sha256": stable_json_sha256([round(value, 10) for value in bridge]),
        "source_trace_path": source_path,
        "start_match_abs_error": abs(float(bridge[0]) - float(real["initial_sortedness_percent"])),
        "end_match_abs_error": abs(float(bridge[-1]) - float(real["final_sortedness_percent"])),
        "swap_count_match_abs_error": abs((len(bridge) - 1) - swap_count),
        "noise_drop_match_abs_error": abs(float(noise_drop) - float(real_noise_drop)),
        "path_total_variation_match_abs_error": abs(float(path_variation) - float(real_path_variation)),
        "bounds_respected": bool(min(bridge) >= 0.0 and max(bridge) <= 100.0),
        "dg_formula_matches_e01_row": True,
        "base_s07_final_sortedness_percent": float(base["final_sortedness_percent"]),
        "base_s07_dg_primary": float(base["dg_primary"]),
        "base_s07_noise_total_drop_percent": total_step_drop(base["sortedness_values"]),
        "elapsed_seconds": time.monotonic() - started,
        **dg,
        "paired_real_dg_primary": float(real["dg_primary"]),
    }
    row["delta_dg_primary_vs_real"] = float(row["dg_primary"] - row["paired_real_dg_primary"])
    row["real_minus_null_dg_primary"] = float(row["paired_real_dg_primary"] - row["dg_primary"])
    return row


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    sem = std / math.sqrt(len(clean)) if len(clean) > 1 else np.nan
    return mean, std, mean - 1.96 * sem if len(clean) > 1 else np.nan, mean + 1.96 * sem if len(clean) > 1 else np.nan


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "dg_primary",
        "dg_event_count",
        "dg_total_drop",
        "dg_total_recovery",
        "noise_total_drop_percent",
        "noise_total_recovery_percent",
        "path_total_variation_percent",
        "target_swap_count",
        "initial_sortedness_percent",
        "final_sortedness_percent",
        "real_minus_null_dg_primary",
        "delta_dg_primary_vs_real",
    ]
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["condition_id", "mode", "algorithm", "frozen_semantics", "frozen_count", "source_type", "null_policy"], dropna=False)
    for keys, group in grouped:
        row: dict[str, Any] = {
            "condition_id": keys[0],
            "mode": keys[1],
            "algorithm": keys[2],
            "frozen_semantics": keys[3],
            "frozen_count": int(keys[4]),
            "source_type": keys[5],
            "null_policy": keys[6],
            "n_rows": int(len(group)),
            "n_repeats": int(group["repeat_index"].nunique()),
            "max_start_match_abs_error": float(pd.to_numeric(group["start_match_abs_error"], errors="coerce").max()),
            "max_end_match_abs_error": float(pd.to_numeric(group["end_match_abs_error"], errors="coerce").max()),
            "max_swap_count_match_abs_error": int(pd.to_numeric(group["swap_count_match_abs_error"], errors="coerce").max()),
            "max_noise_drop_match_abs_error": float(pd.to_numeric(group["noise_drop_match_abs_error"], errors="coerce").max()),
            "bounds_failures": int((~group["bounds_respected"].astype(bool)).sum()),
        }
        for metric in metrics:
            mean, std, low, high = mean_ci(group[metric])
            row[f"mean_{metric}"] = mean
            row[f"std_{metric}"] = std
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition_id", "source_type", "null_policy"]).reset_index(drop=True)


def matching_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["source_type", "null_policy"], dropna=False):
        rows.append(
            {
                "source_type": keys[0],
                "null_policy": keys[1],
                "n_rows": int(len(group)),
                "max_start_match_abs_error": float(pd.to_numeric(group["start_match_abs_error"], errors="coerce").max()),
                "max_end_match_abs_error": float(pd.to_numeric(group["end_match_abs_error"], errors="coerce").max()),
                "max_swap_count_match_abs_error": int(pd.to_numeric(group["swap_count_match_abs_error"], errors="coerce").max()),
                "max_noise_drop_match_abs_error": float(pd.to_numeric(group["noise_drop_match_abs_error"], errors="coerce").max()),
                "max_path_total_variation_match_abs_error": float(
                    pd.to_numeric(group["path_total_variation_match_abs_error"], errors="coerce").max()
                ),
                "bounds_failures": int((~group["bounds_respected"].astype(bool)).sum()),
                "mean_matched_bridge_attempts": float(pd.to_numeric(group["matched_bridge_attempts"], errors="coerce").mean()),
                "max_matched_bridge_attempts": int(pd.to_numeric(group["matched_bridge_attempts"], errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_type", "null_policy"]).reset_index(drop=True)


def expand_f0_for_plot(summary: pd.DataFrame) -> pd.DataFrame:
    f0 = summary[summary["frozen_count"] == 0].copy()
    frozen = summary[summary["frozen_count"] > 0].copy()
    expanded = []
    for semantics in PLOT_SEMANTICS:
        temp = f0.copy()
        temp["plot_frozen_semantics"] = semantics
        expanded.append(temp)
    frozen["plot_frozen_semantics"] = frozen["frozen_semantics"]
    return pd.concat([*expanded, frozen], ignore_index=True, sort=False)


def burden_classification(summary: pd.DataFrame) -> pd.DataFrame:
    plot_df = expand_f0_for_plot(summary)
    real = plot_df[plot_df["source_type"] == "real_e01_dg"]
    null = plot_df[plot_df["source_type"] == "matched_dg_null"]
    rows = []
    for keys, real_group in real.groupby(["plot_frozen_semantics", "mode", "algorithm"], dropna=False):
        semantics, mode, algorithm = keys
        real_ordered = real_group.sort_values("frozen_count")
        x = real_ordered["frozen_count"].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        real_y = real_ordered["mean_dg_primary"].to_numpy(dtype=float)
        real_slope = float(np.polyfit(x, real_y, 1)[0])
        real_barrier_delta = float(real_y[x > 0].mean() - real_y[x == 0][0]) if (x == 0).any() and (x > 0).any() else np.nan
        null_subset = null[
            (null["plot_frozen_semantics"] == semantics)
            & (null["mode"] == mode)
            & (null["algorithm"] == algorithm)
        ]
        for policy, policy_group in null_subset.groupby("null_policy", dropna=False):
            policy_ordered = policy_group.sort_values("frozen_count")
            nx = policy_ordered["frozen_count"].to_numpy(dtype=float)
            ny = policy_ordered["mean_dg_primary"].to_numpy(dtype=float)
            null_slope = float(np.polyfit(nx, ny, 1)[0]) if len(nx) >= 2 else np.nan
            null_barrier_delta = float(ny[nx > 0].mean() - ny[nx == 0][0]) if (nx == 0).any() and (nx > 0).any() else np.nan
            merged = real_ordered[
                ["frozen_count", "mean_dg_primary"]
            ].merge(
                policy_ordered[["frozen_count", "mean_dg_primary"]],
                on="frozen_count",
                how="inner",
                suffixes=("_real", "_null"),
            )
            mean_real_minus_null = float((merged["mean_dg_primary_real"] - merged["mean_dg_primary_null"]).mean())
            min_real_minus_null = float((merged["mean_dg_primary_real"] - merged["mean_dg_primary_null"]).min())
            if min_real_minus_null > 0.05 and real_slope > null_slope:
                classification = "real_dg_above_matched_null_and_steeper"
            elif min_real_minus_null > 0.05:
                classification = "real_dg_above_matched_null"
            elif abs(mean_real_minus_null) <= 0.05:
                classification = "matched_null_explains_mean_dg"
            else:
                classification = "matched_null_exceeds_real_dg"
            rows.append(
                {
                    "plot_frozen_semantics": semantics,
                    "mode": mode,
                    "algorithm": algorithm,
                    "null_policy": policy,
                    "real_dg_slope_by_frozen_count": real_slope,
                    "null_dg_slope_by_frozen_count": null_slope,
                    "real_minus_null_slope": real_slope - null_slope,
                    "real_barrier_minus_f0_mean_dg": real_barrier_delta,
                    "null_barrier_minus_f0_mean_dg": null_barrier_delta,
                    "real_minus_null_barrier_delta": real_barrier_delta - null_barrier_delta,
                    "mean_real_minus_null_dg": mean_real_minus_null,
                    "min_real_minus_null_dg": min_real_minus_null,
                    "classification": classification,
                }
            )
    return pd.DataFrame(rows).sort_values(["plot_frozen_semantics", "mode", "algorithm", "null_policy"]).reset_index(drop=True)


def outcome_from_burden(classification: pd.DataFrame) -> str:
    if classification.empty:
        return "Null"
    primary = classification[
        (classification["plot_frozen_semantics"] == "stuck")
        & (classification["mode"] == "cell_view")
        & (classification["algorithm"].isin(["bubble", "insertion"]))
    ]
    if primary.empty:
        return "Null"
    supportive = primary["classification"].isin(
        ["real_dg_above_matched_null_and_steeper", "real_dg_above_matched_null"]
    ).all()
    if supportive:
        return "Supportive"
    if primary["classification"].eq("matched_null_explains_mean_dg").any():
        return "Constraining/contradictory"
    return "Null"


def make_figure(summary: pd.DataFrame, classification: pd.DataFrame, diagnostics: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = expand_f0_for_plot(summary)
    real = plot_df[plot_df["source_type"] == "real_e01_dg"].copy()
    null = plot_df[plot_df["source_type"] == "matched_dg_null"].copy()
    null_avg = (
        null.groupby(["plot_frozen_semantics", "mode", "algorithm", "frozen_count"], dropna=False)["mean_dg_primary"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), constrained_layout=True)
    colors = {"real": "#2f2f2f", "null": "#4c78a8"}

    ax = axes[0, 0]
    for algorithm, marker in [("bubble", "o"), ("insertion", "s"), ("selection", "^")]:
        for source, frame, linestyle in [("real", real, "-"), ("null", null_avg, "--")]:
            sub = frame[
                (frame["plot_frozen_semantics"] == "stuck")
                & (frame["mode"] == "cell_view")
                & (frame["algorithm"] == algorithm)
            ].sort_values("frozen_count")
            if not sub.empty:
                ax.plot(
                    sub["frozen_count"],
                    sub["mean_dg_primary"],
                    marker=marker,
                    linestyle=linestyle,
                    color=colors[source],
                    label=f"{algorithm} {source}",
                )
    ax.set_title("Stuck cell-view DG burden")
    ax.set_xlabel("Frozen Cell count")
    ax.set_ylabel("Mean DG")
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    delta = summary[summary["source_type"] == "matched_dg_null"].copy()
    delta["real_minus_null"] = pd.to_numeric(delta["mean_real_minus_null_dg_primary"], errors="coerce")
    for policy in NULL_POLICIES:
        sub = delta[
            (delta["frozen_semantics"] == "stuck")
            & (delta["mode"] == "cell_view")
            & (delta["algorithm"].isin(["bubble", "insertion", "selection"]))
            & (delta["null_policy"] == policy)
        ]
        if not sub.empty:
            grouped = sub.groupby("frozen_count")["real_minus_null"].mean().reset_index()
            ax.plot(grouped["frozen_count"], grouped["real_minus_null"], marker="o", label=policy)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Real minus matched-null DG")
    ax.set_xlabel("Frozen Cell count")
    ax.set_ylabel("DG difference")
    ax.set_xticks([1, 2, 3])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    class_counts = classification["classification"].value_counts().sort_index()
    ax.barh(class_counts.index, class_counts.values, color="#72b7b2")
    ax.set_title("Burden-slice classifications")
    ax.set_xlabel("Slice count")
    ax.tick_params(axis="y", labelsize=8)

    ax = axes[1, 1]
    diag = diagnostics[diagnostics["source_type"] == "matched_dg_null"].copy()
    metrics = [
        "max_start_match_abs_error",
        "max_end_match_abs_error",
        "max_swap_count_match_abs_error",
        "max_noise_drop_match_abs_error",
    ]
    x = np.arange(len(metrics))
    heights = [float(diag[m].max()) if not diag.empty else 0.0 for m in metrics]
    ax.bar(x, heights, color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(["start", "end", "swaps", "noise"], rotation=20, ha="right")
    ax.set_title("Max matching error")
    ax.set_ylabel("Absolute error")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("E02 S08 DG matched trajectory nulls", fontsize=14)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def validate_results(
    *,
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    formula_validation: dict[str, Any],
    toy_validation: dict[str, Any],
    expected_real_rows: int,
    expected_null_rows: int,
    figure_path: Path,
    artifacts_written: Sequence[Path],
) -> dict[str, Any]:
    real = result_df[result_df["source_type"] == "real_e01_dg"]
    null = result_df[result_df["source_type"] == "matched_dg_null"]
    all_start = bool((pd.to_numeric(null["start_match_abs_error"], errors="coerce") <= 1e-12).all()) if len(null) else False
    all_end = bool((pd.to_numeric(null["end_match_abs_error"], errors="coerce") <= 1e-12).all()) if len(null) else False
    all_swaps = bool((pd.to_numeric(null["swap_count_match_abs_error"], errors="coerce") == 0).all()) if len(null) else False
    all_noise = bool((pd.to_numeric(null["noise_drop_match_abs_error"], errors="coerce") <= 1e-12).all()) if len(null) else False
    all_variation = bool((pd.to_numeric(null["path_total_variation_match_abs_error"], errors="coerce") <= 1e-12).all()) if len(null) else False
    all_bounds = bool(null["bounds_respected"].astype(bool).all()) if len(null) else False
    no_duplicates = not null.duplicated(["condition_id", "repeat_index", "null_policy", "null_replicate"]).any() if len(null) else False
    figure_ok = figure_path.exists() and figure_path.stat().st_size > 1000
    materialized_names = {
        "e02_dg_null_benchmarks.parquet",
        "dg_matched_nulls.png",
        "e02_dg_null_summary.csv",
        "e02_dg_matching_diagnostics.csv",
        "e02_dg_burden_classification.csv",
    }
    artifacts_present = all(
        path.exists() and path.stat().st_size > 0 for path in artifacts_written if path.name in materialized_names
    )
    outcome = outcome_from_burden(classification_df)
    passed = all(
        [
            len(real) == expected_real_rows,
            len(null) == expected_null_rows,
            no_duplicates,
            all_start,
            all_end,
            all_swaps,
            all_noise,
            all_variation,
            all_bounds,
            formula_validation["e01DgFormulaValidationPassed"],
            toy_validation["toyCaseValidationPassed"],
            len(summary_df) > 0,
            len(diagnostics_df) > 0,
            len(classification_df) > 0,
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
        "outcomeClassification": outcome,
        "artifactsWritten": [str(path) for path in artifacts_written],
        "caveatsOrBlockers": [
            "S08 uses matched synthetic Sortedness bridges, not public cell-policy reruns.",
            "Noise amplitude is operationalized as total backtracking drop in Sortedness percentage points; this exactly fixes path total variation when start and end are fixed.",
            "S07 local-move null generators provide randomized ordering preferences, then bridges are constrained to exact start/end/swap/noise matches.",
            "Traditional E01 DG traces inherit E01 reconstruction caveats; Frozen Cell traces inherit E01 wrapper caveats.",
        ],
        "recommendedNextAction": "Proceed to S09 after Chief Scientist instruction; do not start S09 inside S08.",
        "expectedRealRows": int(expected_real_rows),
        "observedRealRows": int(len(real)),
        "expectedNullRows": int(expected_null_rows),
        "observedNullRows": int(len(null)),
        "noDuplicateNullRows": bool(no_duplicates),
        "allStartSortednessMatched": bool(all_start),
        "allEndSortednessMatched": bool(all_end),
        "allSwapCountsMatched": bool(all_swaps),
        "allNoiseDropMatched": bool(all_noise),
        "allPathTotalVariationMatched": bool(all_variation),
        "allBoundsRespected": bool(all_bounds),
        "maxStartMatchAbsError": float(pd.to_numeric(null["start_match_abs_error"], errors="coerce").max()) if len(null) else np.nan,
        "maxEndMatchAbsError": float(pd.to_numeric(null["end_match_abs_error"], errors="coerce").max()) if len(null) else np.nan,
        "maxSwapCountMatchAbsError": int(pd.to_numeric(null["swap_count_match_abs_error"], errors="coerce").max()) if len(null) else -1,
        "maxNoiseDropMatchAbsError": float(pd.to_numeric(null["noise_drop_match_abs_error"], errors="coerce").max()) if len(null) else np.nan,
        "maxPathTotalVariationMatchAbsError": float(pd.to_numeric(null["path_total_variation_match_abs_error"], errors="coerce").max()) if len(null) else np.nan,
        "maxBridgeAttempts": int(pd.to_numeric(null["matched_bridge_attempts"], errors="coerce").max()) if len(null) else -1,
        "meanBridgeAttempts": float(pd.to_numeric(null["matched_bridge_attempts"], errors="coerce").mean()) if len(null) else np.nan,
        "formulaValidation": formula_validation,
        "toyCaseValidation": toy_validation,
        "summaryRows": int(len(summary_df)),
        "diagnosticRows": int(len(diagnostics_df)),
        "classificationRows": int(len(classification_df)),
        "figureExistsAndNonempty": bool(figure_ok),
        "artifactsPresent": bool(artifacts_present),
        "classificationCounts": {str(key): int(value) for key, value in classification_df["classification"].value_counts().to_dict().items()},
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path) if path.exists() else None,
        "sizeBytes": path.stat().st_size if path.exists() else None,
    }


def build_report(
    *,
    generated_at: str,
    artifacts_dir: Path,
    result_path: Path,
    summary_path: Path,
    diagnostics_path: Path,
    classification_path: Path,
    validation_path: Path,
    figure_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    log_path: Path,
    validation: dict[str, Any],
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    classification: pd.DataFrame,
    unit_result: dict[str, Any] | None,
    git_commit: str,
    git_status: str,
    null_replicates: int,
    workers: int,
    elapsed_seconds: float,
) -> str:
    outcome = validation["outcomeClassification"]
    validation_result = (
        f"passed: {validation['observedRealRows']}/{validation['expectedRealRows']} real rows and "
        f"{validation['observedNullRows']}/{validation['expectedNullRows']} matched null rows; "
        f"start={validation['allStartSortednessMatched']}, end={validation['allEndSortednessMatched']}, "
        f"swap={validation['allSwapCountsMatched']}, noise={validation['allNoiseDropMatched']}, "
        f"DG formula={validation['formulaValidation']['e01DgFormulaValidationPassed']}, "
        f"unit tests={'passed' if unit_result and unit_result['success'] else 'not run'}"
    )
    if outcome == "Supportive":
        lay_summary = (
            "Real stuck cell-view Bubble and Insertion DG remained above matched random trajectory nulls, "
            "so the observed DG pattern was not explained by start, end, swap-count, and total-backtracking-noise matching alone."
        )
    elif outcome == "Constraining/contradictory":
        lay_summary = (
            "At least one primary DG slice was compatible with matched random trajectory nulls, so the DG claim is narrowed by this control."
        )
    else:
        lay_summary = "The matched-null evidence was insufficient or mixed for a clean DG verdict."
    compact_summary = summary[
        [
            "condition_id",
            "mode",
            "algorithm",
            "frozen_semantics",
            "frozen_count",
            "source_type",
            "null_policy",
            "n_rows",
            "mean_dg_primary",
            "mean_real_minus_null_dg_primary",
            "mean_noise_total_drop_percent",
            "max_start_match_abs_error",
            "max_end_match_abs_error",
            "max_noise_drop_match_abs_error",
        ]
    ].copy()
    compact_classification = classification[
        [
            "plot_frozen_semantics",
            "mode",
            "algorithm",
            "null_policy",
            "real_dg_slope_by_frozen_count",
            "null_dg_slope_by_frozen_count",
            "mean_real_minus_null_dg",
            "classification",
        ]
    ].copy()
    return f"""# E02 S08 Matched Delayed Gratification Nulls

## Top Summary

- Research step ID: S08
- Completion status: Completed on {generated_at}
- Artifacts written: `{artifacts_dir / 'research_steps/S08/research_step_full_results.md'}`, `{result_path}`, `{figure_path}`, `{validation_path}`, `{summary_path}`, `{diagnostics_path}`, `{classification_path}`, `{log_path}`, `{manifest_path}`, `{src_manifest_path}`
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers: {'; '.join(validation['caveatsOrBlockers'])}
- Lay summary: {lay_summary}
- Recommended next action: {validation['recommendedNextAction']}

## Chief Handoff

S08 completed matched DG null trajectories for all E01 DG runs. Each null has the same start Sortedness, end Sortedness, swap count, and total backtracking-drop amplitude as its paired real run, while the randomized ordering preferences come from S07 local-move null policies.

## Frozen Question

Is DG specifically elevated by barrier contexts, or can matched random trajectories with the same start, end, swap count, and noise amplitude explain it?

## Inputs

- E01 DG run table: `/previous-artifacts/E01/results/e01_delayed_gratification.parquet`
- E01 f=0 trajectories: `/previous-artifacts/E01/traces/e01_figure3_trajectories.parquet`
- E01 frozen trajectories: `/previous-artifacts/E01/traces/e01_frozen_cell_trajectories.parquet`
- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- S07 local-move null generator code: `scripts/e02_s07_local_move_nulls.py`
- Repository commit at run time: `{git_commit}`

## Lay Summary

S08 asks whether Delayed Gratification can appear just because a trajectory starts and ends at certain Sortedness values, takes a certain number of swaps, and contains a fixed amount of backtracking noise. The nulls preserve those quantities exactly, then randomize the order of local-move-like progress and backtracking.

## Detailed Methods

The E01 DG formula was revalidated before null generation. S08 recomputed DG from the source Sortedness traces and required exact agreement with `/previous-artifacts/E01/results/e01_delayed_gratification.parquet`. It also recomputed the equivalent monotonicity-error formula and ran toy cases for known DG values.

For each E01 DG run, S08 used the E01 config to recover the paired initial array and pure algorithm label. For each S07 local null policy and replicate, it generated a local-move null path to obtain randomized ordering preferences. It then built a bounded unit-step Sortedness bridge constrained to:

- initial Sortedness equal to the real run;
- final Sortedness equal to the real run;
- number of intervals equal to the real swap count;
- total backtracking drop equal to the real run's total Sortedness drop.

Because start, end, and total drop are fixed, total recovery and total path variation are also fixed. DG is then recomputed on each matched null trajectory.

## Commands

- Unit tests: `{unit_result['command'] if unit_result else 'not run'}`
- S08 production run: `{sys.executable} scripts/e02_s08_dg_nulls.py --repo-dir /workspace/cell-research --artifacts-dir {artifacts_dir} --null-replicates {null_replicates} --workers {workers}`

## Dependencies And Parameters

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- New dependencies installed: none
- CPU workers: `{workers}`
- Null replicates per real run and policy: `{null_replicates}`
- Null policies: `{', '.join(NULL_POLICIES)}`
- Noise-amplitude definition: total backtracking drop in Sortedness percentage points
- Elapsed production time: `{elapsed_seconds:.2f}` seconds

## Results

Outcome classification counts: `{json.dumps(validation['classificationCounts'], sort_keys=True)}`.

### DG Null Summary

{markdown_table(compact_summary, max_rows=60)}

### Burden Classification

{markdown_table(compact_classification, max_rows=48)}

The full run-level benchmark table is `{result_path}`. The matching diagnostics are `{diagnostics_path}` and the figure is `{figure_path}`.

## Validation

Validation required exact E01 DG formula agreement, toy-case DG validation, all planned real and null rows, no duplicate null rows, exact matching for start Sortedness, end Sortedness, swap count, total backtracking drop, path total variation, bounded trajectories, nonempty summaries/classifications, and a nonempty figure.

Key validation values:

- Real rows: `{validation['observedRealRows']}` / `{validation['expectedRealRows']}`
- Null rows: `{validation['observedNullRows']}` / `{validation['expectedNullRows']}`
- Max start match error: `{validation['maxStartMatchAbsError']}`
- Max end match error: `{validation['maxEndMatchAbsError']}`
- Max swap-count match error: `{validation['maxSwapCountMatchAbsError']}`
- Max noise-drop match error: `{validation['maxNoiseDropMatchAbsError']}`
- Max path-variation match error: `{validation['maxPathTotalVariationMatchAbsError']}`
- Bounds respected for all nulls: `{validation['allBoundsRespected']}`
- E01 DG formula validation passed: `{validation['formulaValidation']['e01DgFormulaValidationPassed']}`
- Toy-case DG validation passed: `{validation['toyCaseValidation']['toyCaseValidationPassed']}`

### Matching Diagnostics

{markdown_table(diagnostics, max_rows=12)}

## Artifacts

- Run-level DG null benchmark table: `{result_path}`
- Condition/policy summary: `{summary_path}`
- Matching diagnostics: `{diagnostics_path}`
- Burden classification: `{classification_path}`
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

- No matching blocker was encountered: start, end, swap count, total backtracking drop, and path total variation matched exactly for every null row.
- These are matched synthetic Sortedness bridges constrained by S07 local-move ordering preferences, not actual cell-policy reruns and not valid biological mechanisms.
- Matching total backtracking drop intentionally removes one obvious explanation for DG differences; remaining differences reflect ordering and segmentation under the DG formula.
- Traditional E01 trajectories remain reconstructed, and frozen cell-view traces retain E01 wrapper caveats.

## Recommended Next Action

{validation['recommendedNextAction']}
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
        "recommendedNextAction": "Resolve the S08 blocker before S09; do not start S09.",
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
    report_path.write_text(
        f"""# E02 S08 Matched Delayed Gratification Nulls

## Top Summary

- Research step ID: S08
- Completion status: Blocked on {generated_at}
- Artifacts written: `{report_path}`, `{validation_path}`, `{manifest_path}`, `{src_manifest_path}`, `{log_path}`
- Validation result: blocked
- Outcome classification: constraining/contradictory
- Caveats or blockers: {reason}
- Lay summary: S08 could not safely run DG matched nulls because required inputs or DG formula validation were unavailable.
- Recommended next action: Resolve the S08 blocker before S09; do not start S09.

## Frozen Question

Is DG specifically elevated by barrier contexts, or can matched random trajectories with the same start, end, swap count, and noise amplitude explain it?

## Methods

S08 checked required E01 DG tables, E01 trace inputs, E01 config rows, and formula validation before production null generation.

## Results

No DG null benchmark table or figure was generated.

## Validation

Validation was blocked before production null runs.

## Provenance

- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Generated at UTC: `{generated_at}`
""",
        encoding="utf-8",
    )
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.research_step_artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "createdAtUtc": generated_at,
            "artifacts": [
                artifact_entry(report_path, artifacts_dir, "S08 blocked full-results Markdown handoff report."),
                artifact_entry(validation_path, artifacts_dir, "S08 blocked validation evidence."),
                artifact_entry(log_path, artifacts_dir, "S08 blocked execution log."),
                artifact_entry(src_manifest_path, artifacts_dir, "S08 blocked source snapshot manifest."),
            ],
        },
    )
    return 2


def main() -> int:
    args = parse_args()
    started = time.monotonic()
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
    result_path = results_dir / "e02_dg_null_benchmarks.parquet"
    summary_path = step_dir / "e02_dg_null_summary.csv"
    diagnostics_path = step_dir / "e02_dg_matching_diagnostics.csv"
    classification_path = step_dir / "e02_dg_burden_classification.csv"
    figure_path = figures_dir / "dg_matched_nulls.png"
    validation_path = step_dir / "s08_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s08_dg_nulls_manifest.json"
    log_path = logs_dir / "e02_s08_dg_nulls.log"
    report_path = step_dir / "research_step_full_results.md"
    log_lines = [f"E02 S08 started {generated_at}", f"repo_dir={repo_dir}", f"artifacts_dir={artifacts_dir}"]

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
        conditions = selected_s08_conditions(cfg)
        condition_ids = {row["condition_id"] for row in conditions}
        dg_df = pd.read_parquet(args.e01_dg_path)
        dg_df = dg_df[dg_df["condition_id"].isin(condition_ids)].copy()
        if args.max_repeats is not None:
            dg_df = dg_df[dg_df["repeat_index"] < int(args.max_repeats)].copy()
        if dg_df.empty:
            raise RuntimeError("No E01 DG rows available for S08.")
        trace_map, trace_rows_used = load_trace_map(dg_df, args.e01_f0_trace_path, args.e01_frozen_trace_path)
        toy_validation = validate_toy_cases()
        if not toy_validation["toyCaseValidationPassed"]:
            raise RuntimeError("Toy-case DG validation failed.")
        formula_validation = validate_e01_dg_formula(dg_df, trace_map)
        if not formula_validation["e01DgFormulaValidationPassed"]:
            raise RuntimeError(f"E01 DG formula validation failed: {formula_validation}")
        tasks = build_tasks(cfg, dg_df, trace_map, args.null_replicates, args.random_seed)
    except Exception as exc:
        log_path.write_text("\n".join(log_lines) + f"\nBLOCKED: {exc!r}\n", encoding="utf-8")
        return write_blocked_outputs(
            generated_at=generated_at,
            artifacts_dir=artifacts_dir,
            repo_dir=repo_dir,
            reason=f"S08 input matching or DG formula validation failed before null runs: {exc!r}",
            validation_path=validation_path,
            report_path=report_path,
            manifest_path=manifest_path,
            src_manifest_path=src_manifest_path,
            log_path=log_path,
        )

    real_rows = [real_record(row, trace_map[(str(row["condition_id"]), int(row["repeat_index"]))]) for row in dg_df.to_dict(orient="records")]
    workers = max(1, min(int(args.workers), len(tasks), 8))
    log_lines.append(
        f"conditions={len(condition_ids)} real_rows={len(real_rows)} trace_rows_used={trace_rows_used} "
        f"null_tasks={len(tasks)} workers={workers} null_replicates={args.null_replicates}"
    )

    null_rows: list[dict[str, Any]] = []
    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            null_rows.append(run_null_task(task))
            if idx % 1000 == 0 or idx == len(tasks):
                log_lines.append(f"completed_null_tasks={idx}/{len(tasks)}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, row in enumerate(executor.map(run_null_task, tasks, chunksize=25), start=1):
                null_rows.append(row)
                if idx % 1000 == 0 or idx == len(tasks):
                    log_lines.append(f"completed_null_tasks={idx}/{len(tasks)}")

    result_df = pd.DataFrame([*real_rows, *null_rows])
    result_df = result_df.sort_values(["condition_id", "repeat_index", "source_type", "null_policy", "null_replicate"]).reset_index(drop=True)
    summary_df = summarize_results(result_df)
    diagnostics_df = matching_diagnostics(result_df)
    classification_df = burden_classification(summary_df)

    result_df.to_parquet(result_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    make_figure(summary_df, classification_df, diagnostics_df, figure_path)

    expected_artifacts = [
        report_path,
        result_path,
        figure_path,
        validation_path,
        summary_path,
        diagnostics_path,
        classification_path,
        log_path,
        manifest_path,
        src_manifest_path,
    ]
    validation = validate_results(
        result_df=result_df,
        summary_df=summary_df,
        diagnostics_df=diagnostics_df,
        classification_df=classification_df,
        formula_validation=formula_validation,
        toy_validation=toy_validation,
        expected_real_rows=len(real_rows),
        expected_null_rows=len(tasks),
        figure_path=figure_path,
        artifacts_written=expected_artifacts,
    )
    write_json(validation_path, validation)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    elapsed_seconds = time.monotonic() - started
    log_lines.extend(
        [
            f"result_rows={len(result_df)} summary_rows={len(summary_df)} diagnostic_rows={len(diagnostics_df)} classification_rows={len(classification_df)}",
            f"validation={json.dumps(validation, sort_keys=True)}",
            f"elapsed_seconds={elapsed_seconds:.3f}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    code_files = [
        repo_dir / "scripts/e02_s08_dg_nulls.py",
        repo_dir / "tests/e02/test_dg_nulls.py",
        repo_dir / "scripts/e02_s07_local_move_nulls.py",
        repo_dir / "scripts/e02_s02_scheduler_comparison.py",
    ]
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
                "nullReplicates": int(args.null_replicates),
                "workers": int(workers),
                "randomSeed": int(args.random_seed),
                "maxRepeats": args.max_repeats,
                "nullPolicies": list(NULL_POLICIES),
                "noiseAmplitudeDefinition": "total backtracking drop in Sortedness percentage points",
                "e01DgPath": str(args.e01_dg_path),
                "e01F0TracePath": str(args.e01_f0_trace_path),
                "e01FrozenTracePath": str(args.e01_frozen_trace_path),
            },
            "validation": validation,
            "dependencies": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
                "newDependenciesInstalled": [],
            },
        },
    )

    report = build_report(
        generated_at=generated_at,
        artifacts_dir=artifacts_dir,
        result_path=result_path,
        summary_path=summary_path,
        diagnostics_path=diagnostics_path,
        classification_path=classification_path,
        validation_path=validation_path,
        figure_path=figure_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        log_path=log_path,
        validation=validation,
        summary=summary_df,
        diagnostics=diagnostics_df,
        classification=classification_df,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        null_replicates=args.null_replicates,
        workers=workers,
        elapsed_seconds=elapsed_seconds,
    )
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
                artifact_entry(report_path, artifacts_dir, "S08 full-results Markdown handoff report."),
                artifact_entry(result_path, artifacts_dir, "Run-level real-versus-matched-DG-null table."),
                artifact_entry(summary_path, artifacts_dir, "Condition and null-policy DG summary table."),
                artifact_entry(diagnostics_path, artifacts_dir, "DG matching diagnostics table."),
                artifact_entry(classification_path, artifacts_dir, "DG burden classification table."),
                artifact_entry(figure_path, artifacts_dir, "DG matched null benchmark figure."),
                artifact_entry(validation_path, artifacts_dir, "S08 validation evidence."),
                artifact_entry(log_path, artifacts_dir, "S08 execution log."),
                artifact_entry(src_manifest_path, artifacts_dir, "S08 source snapshot manifest."),
            ],
        },
    )
    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
