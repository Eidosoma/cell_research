#!/usr/bin/env python3
"""Run E01 S04 Figure 3-style trajectory reproduction.

The script consumes the S03 frozen config and condition matrix. Traditional
baselines are reconstructed exactly as labelled in S03; cell-view baselines
wrap the public repository thread classes and StatusProbe snapshots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


STEP_ID = "S04"
STEP_NUMBER = 4
EXPERIMENT_ID = "E01"
ALGORITHMS = ("bubble", "insertion", "selection")
MODES = ("traditional", "cell_view")
TRACE_COLUMNS = [
    "research_step_id",
    "experiment_id",
    "condition_id",
    "mode",
    "algorithm",
    "baseline_source",
    "matched_group_id",
    "value_bank_id",
    "repeat_index",
    "initial_array_seed",
    "scheduler_seed",
    "swap_step",
    "comparison_count_at_step",
    "sortedness_percent",
    "monotonicity_error_count",
    "is_initial",
    "is_final",
    "stop_reason",
    "timed_out",
    "initial_array_sha256",
    "final_array_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config-path", type=Path, default=Path("/artifacts/configs/e01_baseline_configs.json"))
    parser.add_argument("--condition-matrix-path", type=Path, default=Path("/artifacts/tables/e01_condition_matrix.csv"))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--max-repeats", type=int, default=None, help="Optional smoke-test cap per condition.")
    parser.add_argument("--cell-view-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--cell-view-max-swaps", type=int, default=50000)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.001)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_status(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nondecreasing(values: list[int]) -> bool:
    return all(values[i - 1] <= values[i] for i in range(1, len(values)))


def sortedness_percent(values: list[int], direction: str = "increasing") -> float:
    if not values:
        return 100.0
    if direction in {"increasing", "nondecreasing", "all_increasing"}:
        ordered_pairs = sum(1 for idx in range(1, len(values)) if values[idx - 1] <= values[idx])
    elif direction in {"decreasing", "nonincreasing", "all_decreasing"}:
        ordered_pairs = sum(1 for idx in range(1, len(values)) if values[idx - 1] >= values[idx])
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return 100.0 * (1 + ordered_pairs) / len(values)


def monotonicity_error_count(values: list[int], direction: str = "increasing") -> int:
    if direction in {"increasing", "nondecreasing", "all_increasing"}:
        return sum(1 for idx in range(1, len(values)) if values[idx] < values[idx - 1])
    if direction in {"decreasing", "nonincreasing", "all_decreasing"}:
        return sum(1 for idx in range(1, len(values)) if values[idx] > values[idx - 1])
    raise ValueError(f"Unsupported direction: {direction}")


def metric_record(values: list[int], swap_step: int, comparison_count: int | None) -> tuple[int, int | None, float, int]:
    return (
        swap_step,
        comparison_count,
        sortedness_percent(values, "increasing"),
        monotonicity_error_count(values, "increasing"),
    )


def traditional_bubble(values: list[int]) -> dict[str, Any]:
    arr = list(values)
    comparisons = 0
    swaps = 0
    records = [metric_record(arr, swaps, comparisons)]
    while True:
        inversion_idx = None
        for idx in range(len(arr) - 1):
            comparisons += 1
            if arr[idx] > arr[idx + 1]:
                inversion_idx = idx
                break
        if inversion_idx is None:
            break
        idx = inversion_idx
        while idx < len(arr) - 1:
            comparisons += 1
            if arr[idx] <= arr[idx + 1]:
                break
            arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]
            swaps += 1
            records.append(metric_record(arr, swaps, comparisons))
            idx += 1
    return {
        "records": records,
        "final_values": arr,
        "swap_count": swaps,
        "comparison_count": comparisons,
        "compare_plus_swap_count": comparisons + swaps,
        "stop_reason": "sorted",
        "timed_out": False,
        "elapsed_seconds": None,
        "alive_threads_after_cleanup": [],
    }


def traditional_insertion(values: list[int]) -> dict[str, Any]:
    arr = list(values)
    comparisons = 0
    swaps = 0
    records = [metric_record(arr, swaps, comparisons)]
    for unsorted_idx in range(1, len(arr)):
        idx = unsorted_idx
        while idx > 0:
            comparisons += 1
            if arr[idx - 1] <= arr[idx]:
                break
            arr[idx - 1], arr[idx] = arr[idx], arr[idx - 1]
            swaps += 1
            records.append(metric_record(arr, swaps, comparisons))
            idx -= 1
    return {
        "records": records,
        "final_values": arr,
        "swap_count": swaps,
        "comparison_count": comparisons,
        "compare_plus_swap_count": comparisons + swaps,
        "stop_reason": "sorted",
        "timed_out": False,
        "elapsed_seconds": None,
        "alive_threads_after_cleanup": [],
    }


def traditional_selection(values: list[int]) -> dict[str, Any]:
    arr = list(values)
    comparisons = 0
    swaps = 0
    records = [metric_record(arr, swaps, comparisons)]
    for boundary in range(len(arr)):
        min_idx = boundary
        for idx in range(boundary + 1, len(arr)):
            comparisons += 1
            if arr[idx] < arr[min_idx]:
                min_idx = idx
        if min_idx != boundary:
            arr[boundary], arr[min_idx] = arr[min_idx], arr[boundary]
            swaps += 1
            records.append(metric_record(arr, swaps, comparisons))
    return {
        "records": records,
        "final_values": arr,
        "swap_count": swaps,
        "comparison_count": comparisons,
        "compare_plus_swap_count": comparisons + swaps,
        "stop_reason": "sorted",
        "timed_out": False,
        "elapsed_seconds": None,
        "alive_threads_after_cleanup": [],
    }


TRADITIONAL_RUNNERS = {
    "bubble": traditional_bubble,
    "insertion": traditional_insertion,
    "selection": traditional_selection,
}


def import_cell_modules(repo_dir: Path) -> dict[str, Any]:
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from modules.multithread.BubbleSortCell import BubbleSortCell
    from modules.multithread.CellGroup import CellGroup, GroupStatus
    from modules.multithread.InsertionSortCell import InsertionSortCell
    from modules.multithread.MultiThreadCell import CellStatus
    from modules.multithread.SelectionSortCell import SelectionSortCell
    from modules.multithread.StatusProbe import StatusProbe

    return {
        "BubbleSortCell": BubbleSortCell,
        "CellGroup": CellGroup,
        "CellStatus": CellStatus,
        "GroupStatus": GroupStatus,
        "InsertionSortCell": InsertionSortCell,
        "SelectionSortCell": SelectionSortCell,
        "StatusProbe": StatusProbe,
    }


def stop_cell_threads(
    cells: list[Any],
    group: Any,
    lock: threading.Lock,
    cell_status: Any,
    group_status: Any,
) -> tuple[list[Any], bool]:
    acquired = lock.acquire(timeout=2)
    try:
        for cell in cells:
            cell.status = cell_status.INACTIVE
        group.status = group_status.MERGED
    finally:
        if acquired:
            lock.release()

    for cell in cells:
        cell.join(timeout=1)
    group.join(timeout=1)
    alive: list[Any] = [cell.threadID for cell in cells if cell.is_alive()]
    if group.is_alive():
        alive.append("group")
    return alive, acquired


def run_cell_view(
    repo_dir: Path,
    algorithm: str,
    values: list[int],
    scheduler_seed: int,
    timeout_seconds: float,
    max_swaps: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    modules = import_cell_modules(repo_dir)
    cls_by_algorithm = {
        "bubble": modules["BubbleSortCell"],
        "insertion": modules["InsertionSortCell"],
        "selection": modules["SelectionSortCell"],
    }
    cls = cls_by_algorithm[algorithm]
    random.seed(scheduler_seed)

    status_probe = modules["StatusProbe"]()
    lock = threading.Lock()
    left_boundary = (0, 1)
    right_boundary = (len(values) - 1, 1)
    cells: list[Any] = []
    for idx, value in enumerate(values):
        cell = cls(
            idx + 1,
            value,
            lock,
            (idx, 1),
            cells,
            left_boundary,
            right_boundary,
            status_probe,
            disable_visualization=True,
        )
        cell.daemon = True
        cells.append(cell)
    group = modules["CellGroup"](
        cells,
        cells,
        0,
        left_boundary,
        right_boundary,
        modules["GroupStatus"].ACTIVE,
        lock,
        100_000_000,
        100_000_000,
    )
    group.daemon = True
    for cell in cells:
        cell.group = group

    started = time.monotonic()
    lock.acquire()
    try:
        for cell in cells:
            cell.start()
        group.start()
    finally:
        lock.release()

    stop_reason = "sorted"
    timed_out = False
    while not nondecreasing([cell.value for cell in cells]):
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            stop_reason = "timeout"
            timed_out = True
            break
        if status_probe.swap_count > max_swaps:
            stop_reason = "max_swaps_exceeded"
            break
        time.sleep(poll_interval_seconds)

    final_values_before_cleanup = [cell.value for cell in cells]
    alive_threads, cleanup_lock_acquired = stop_cell_threads(
        cells,
        group,
        lock,
        modules["CellStatus"],
        modules["GroupStatus"],
    )
    elapsed_seconds = time.monotonic() - started
    final_values = [cell.value for cell in cells]
    if final_values != final_values_before_cleanup and nondecreasing(final_values) and stop_reason != "timeout":
        stop_reason = "sorted_after_cleanup"

    records = [metric_record(values, 0, 0)]
    for step_idx, snapshot in enumerate(status_probe.sorting_steps, start=1):
        records.append(metric_record(snapshot, step_idx, None))
    if not records or records[-1][2] != sortedness_percent(final_values, "increasing"):
        records.append(metric_record(final_values, status_probe.swap_count, None))

    return {
        "records": records,
        "final_values": final_values,
        "swap_count": status_probe.swap_count,
        "comparison_count": status_probe.compare_and_swap_count,
        "compare_plus_swap_count": status_probe.swap_count + status_probe.compare_and_swap_count,
        "stop_reason": stop_reason,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed_seconds,
        "alive_threads_after_cleanup": alive_threads,
        "probe_sorting_step_count": len(status_probe.sorting_steps),
        "probe_cell_type_step_count": len(status_probe.cell_types),
        "cleanup_lock_acquired": cleanup_lock_acquired,
    }


def scheduler_seed(base_seed: int, condition_id: str, repeat_idx: int) -> int:
    condition_num = int(condition_id.replace("E01C", ""))
    return base_seed * 1_000_000 + 40 * 10_000 + condition_num * 1_000 + repeat_idx


def load_s04_conditions(condition_matrix_path: Path) -> list[dict[str, str]]:
    with condition_matrix_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    s04_rows = [row for row in rows if "S04" in row["step_scope"].split(",")]
    return sorted(s04_rows, key=lambda row: (row["algorithm"], row["mode"]))


def validate_condition_inputs(cfg: dict[str, Any], rows: list[dict[str, str]], max_repeats: int | None) -> dict[str, Any]:
    defaults = cfg["globalDefaults"]
    repeat_count = int(defaults["repeatCount"])
    repeats_to_run = min(max_repeats or repeat_count, repeat_count)
    errors: list[str] = []
    if len(rows) != 6:
        errors.append(f"Expected 6 S04 condition rows; found {len(rows)}.")
    expected_pairs = {(mode, algorithm) for mode in MODES for algorithm in ALGORITHMS}
    observed_pairs = {(row["mode"], row["algorithm"]) for row in rows}
    if observed_pairs != expected_pairs:
        errors.append(f"S04 mode/algorithm set mismatch: observed {sorted(observed_pairs)}.")
    for row in rows:
        if int(row["repeat_count"]) != repeat_count:
            errors.append(f"{row['condition_id']} repeat_count={row['repeat_count']} differs from config repeatCount={repeat_count}.")
        value_bank = cfg["seedBanks"]["valueBanks"][row["value_bank_id"]]
        if len(value_bank["initialArrays"]) < repeats_to_run:
            errors.append(f"{row['condition_id']} value bank {row['value_bank_id']} has fewer than {repeats_to_run} arrays.")
        if len(value_bank["seeds"]) < repeats_to_run:
            errors.append(f"{row['condition_id']} value bank {row['value_bank_id']} has fewer than {repeats_to_run} seeds.")
    return {
        "expectedConditionRows": 6,
        "observedConditionRows": len(rows),
        "expectedRepeatsPerCondition": repeat_count,
        "repeatsToRun": repeats_to_run,
        "inputValidationErrors": errors,
        "inputValidationPassed": not errors,
    }


def append_trace_rows(
    trace_rows: list[tuple[Any, ...]],
    row: dict[str, str],
    repeat_idx: int,
    initial_seed: int,
    scheduler_seed_value: int | None,
    initial_values: list[int],
    run_result: dict[str, Any],
) -> None:
    initial_hash = sha256_json(initial_values)
    final_hash = sha256_json(run_result["final_values"])
    records = list(run_result["records"])
    last_idx = len(records) - 1
    for idx, (swap_step, comparison_count, sortedness, error_count) in enumerate(records):
        trace_rows.append(
            (
                STEP_ID,
                EXPERIMENT_ID,
                row["condition_id"],
                row["mode"],
                row["algorithm"],
                row["baseline_source"],
                row["matched_group_id"],
                row["value_bank_id"],
                repeat_idx,
                initial_seed,
                scheduler_seed_value,
                swap_step,
                comparison_count,
                sortedness,
                error_count,
                idx == 0,
                idx == last_idx,
                run_result["stop_reason"],
                bool(run_result["timed_out"]),
                initial_hash,
                final_hash,
            )
        )


def run_condition(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    row: dict[str, str],
    repeats_to_run: int,
    trace_rows: list[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    value_bank = cfg["seedBanks"]["valueBanks"][row["value_bank_id"]]
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    run_records: list[dict[str, Any]] = []
    print(f"[{utc_now()}] running {row['condition_id']} {row['mode']} {row['algorithm']} repeats={repeats_to_run}", flush=True)
    for repeat_idx in range(repeats_to_run):
        initial_values = list(value_bank["initialArrays"][repeat_idx])
        initial_seed = int(value_bank["seeds"][repeat_idx])
        started = time.monotonic()
        if row["mode"] == "traditional":
            run_result = TRADITIONAL_RUNNERS[row["algorithm"]](initial_values)
            run_result["elapsed_seconds"] = time.monotonic() - started
            scheduler_seed_value = None
        else:
            scheduler_seed_value = scheduler_seed(base_seed, row["condition_id"], repeat_idx)
            run_result = run_cell_view(
                args.repo_dir,
                row["algorithm"],
                initial_values,
                scheduler_seed_value,
                args.cell_view_timeout_seconds,
                args.cell_view_max_swaps,
                args.poll_interval_seconds,
            )
        final_sortedness = sortedness_percent(run_result["final_values"], "increasing")
        final_error = monotonicity_error_count(run_result["final_values"], "increasing")
        append_trace_rows(
            trace_rows,
            row,
            repeat_idx,
            initial_seed,
            scheduler_seed_value,
            initial_values,
            run_result,
        )
        run_records.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": row["condition_id"],
                "mode": row["mode"],
                "algorithm": row["algorithm"],
                "baseline_source": row["baseline_source"],
                "matched_group_id": row["matched_group_id"],
                "value_bank_id": row["value_bank_id"],
                "repeat_index": repeat_idx,
                "initial_array_seed": initial_seed,
                "scheduler_seed": scheduler_seed_value,
                "initial_array_sha256": sha256_json(initial_values),
                "final_array_sha256": sha256_json(run_result["final_values"]),
                "swap_count": int(run_result["swap_count"]),
                "comparison_count": int(run_result["comparison_count"]),
                "compare_plus_swap_count": int(run_result["compare_plus_swap_count"]),
                "final_sortedness_percent": final_sortedness,
                "final_monotonicity_error_count": final_error,
                "stop_reason": run_result["stop_reason"],
                "timed_out": bool(run_result["timed_out"]),
                "alive_threads_after_cleanup": json.dumps(run_result["alive_threads_after_cleanup"]),
                "elapsed_seconds": run_result["elapsed_seconds"],
                "probe_sorting_step_count": run_result.get("probe_sorting_step_count"),
                "probe_cell_type_step_count": run_result.get("probe_cell_type_step_count"),
                "cleanup_lock_acquired": run_result.get("cleanup_lock_acquired"),
            }
        )
        if (repeat_idx + 1) % 10 == 0 or repeat_idx + 1 == repeats_to_run:
            print(
                f"[{utc_now()}] finished {row['condition_id']} repeat {repeat_idx + 1}/{repeats_to_run}",
                flush=True,
            )
    return run_records


def validate_outputs(
    cfg: dict[str, Any],
    s04_rows: list[dict[str, str]],
    run_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    repeats_to_run: int,
) -> dict[str, Any]:
    repeat_counts = run_df.groupby(["mode", "algorithm"])["repeat_index"].nunique().to_dict()
    expected_repeat_counts = {(mode, algorithm): repeats_to_run for mode in MODES for algorithm in ALGORITHMS}
    repeat_count_ok = repeat_counts == expected_repeat_counts

    all_final_sorted = bool((run_df["final_sortedness_percent"] == 100.0).all())
    no_timeouts = bool((~run_df["timed_out"]).all())
    no_alive_threads = bool(run_df["alive_threads_after_cleanup"].eq("[]").all())
    final_rows = trace_df[trace_df["is_final"]]
    exactly_one_final_trace_row = bool(
        final_rows.groupby(["condition_id", "repeat_index"]).size().eq(1).all()
        and len(final_rows) == len(run_df)
    )

    matched_details: list[dict[str, Any]] = []
    matched_initial_arrays_valid = True
    for algorithm in ALGORITHMS:
        trad = run_df[(run_df["algorithm"] == algorithm) & (run_df["mode"] == "traditional")].sort_values("repeat_index")
        cell = run_df[(run_df["algorithm"] == algorithm) & (run_df["mode"] == "cell_view")].sort_values("repeat_index")
        same_hashes = trad["initial_array_sha256"].tolist() == cell["initial_array_sha256"].tolist()
        same_seeds = trad["initial_array_seed"].tolist() == cell["initial_array_seed"].tolist()
        matched_initial_arrays_valid = matched_initial_arrays_valid and same_hashes and same_seeds
        matched_details.append(
            {
                "algorithm": algorithm,
                "traditionalConditionId": trad["condition_id"].iloc[0] if not trad.empty else None,
                "cellViewConditionId": cell["condition_id"].iloc[0] if not cell.empty else None,
                "sameInitialArrayHashesByRepeat": same_hashes,
                "sameInitialArraySeedsByRepeat": same_seeds,
                "repeatCountCompared": min(len(trad), len(cell)),
            }
        )

    success = (
        repeat_count_ok
        and all_final_sorted
        and matched_initial_arrays_valid
        and no_timeouts
        and no_alive_threads
        and exactly_one_final_trace_row
        and repeats_to_run == int(cfg["globalDefaults"]["repeatCount"])
    )
    validation_result = "passed" if success else "failed"
    if repeats_to_run != int(cfg["globalDefaults"]["repeatCount"]):
        validation_result = "smoke_only_failed_expected_repeat_count"
    elif success and any("traditional" == row["mode"] for row in s04_rows):
        validation_result = "passed_with_reconstructed_traditional_caveat"

    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed_with_caveats" if success else "completed_validation_failed",
        "validationResult": validation_result,
        "expectedRepeatsPerModeAlgorithm": int(cfg["globalDefaults"]["repeatCount"]),
        "observedRepeatsPerModeAlgorithm": {f"{mode}:{algorithm}": count for (mode, algorithm), count in repeat_counts.items()},
        "repeatCountOk": repeat_count_ok,
        "allFinalSortedness100": all_final_sorted,
        "matchedInitialArraysValid": matched_initial_arrays_valid,
        "matchedInitialArrayDetails": matched_details,
        "noTimeouts": no_timeouts,
        "noAliveThreadsAfterCleanup": no_alive_threads,
        "exactlyOneFinalTraceRowPerRun": exactly_one_final_trace_row,
        "traceRows": int(len(trace_df)),
        "runRows": int(len(run_df)),
        "caveatsOrBlockers": [
            "Traditional Bubble, Insertion, and Selection are reconstructed baselines because S02 found no public traditional-generation runner.",
            "Cell-view classes are public repository thread classes; Python thread scheduling remains platform-dependent even with recorded scheduler seeds.",
        ],
        "recommendedNextAction": "Proceed to S05 efficiency comparisons using the S04 wrappers and trace/run records; do not change S03 assumptions unless S05 comparison-count sensitivity requires a documented branch.",
    }


def build_condition_summary(run_df: pd.DataFrame, validation: dict[str, Any]) -> pd.DataFrame:
    grouped = []
    stop_counts = run_df.groupby("condition_id")["stop_reason"].apply(lambda values: json.dumps(dict(Counter(values)))).to_dict()
    matched_valid_by_algorithm = {
        detail["algorithm"]: detail["sameInitialArrayHashesByRepeat"] and detail["sameInitialArraySeedsByRepeat"]
        for detail in validation["matchedInitialArrayDetails"]
    }
    for condition_id, condition_df in run_df.groupby("condition_id", sort=True):
        first = condition_df.iloc[0]
        grouped.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "mode": first["mode"],
                "algorithm": first["algorithm"],
                "baseline_source": first["baseline_source"],
                "matched_group_id": first["matched_group_id"],
                "value_bank_id": first["value_bank_id"],
                "repetitions_expected": validation["expectedRepeatsPerModeAlgorithm"],
                "repetitions_observed": condition_df["repeat_index"].nunique(),
                "matched_initial_arrays_valid": matched_valid_by_algorithm.get(first["algorithm"], False),
                "all_final_sorted": bool((condition_df["final_sortedness_percent"] == 100.0).all()),
                "min_final_sortedness_percent": condition_df["final_sortedness_percent"].min(),
                "mean_final_sortedness_percent": condition_df["final_sortedness_percent"].mean(),
                "max_final_sortedness_percent": condition_df["final_sortedness_percent"].max(),
                "mean_swap_steps": condition_df["swap_count"].mean(),
                "std_swap_steps": condition_df["swap_count"].std(ddof=1),
                "min_swap_steps": int(condition_df["swap_count"].min()),
                "max_swap_steps": int(condition_df["swap_count"].max()),
                "mean_compare_plus_swap_count": condition_df["compare_plus_swap_count"].mean(),
                "timed_out_runs": int(condition_df["timed_out"].sum()),
                "stop_reason_counts": stop_counts[condition_id],
                "mean_elapsed_seconds": condition_df["elapsed_seconds"].mean(),
                "caveats": "traditional baseline reconstructed" if first["mode"] == "traditional" else "thread scheduling platform-dependent",
            }
        )
    return pd.DataFrame(grouped)


def plot_figure(trace_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharey=True)
    colors = {
        "traditional": "#4f6fad",
        "cell_view": "#d95f02",
    }
    mode_titles = {"traditional": "Traditional", "cell_view": "Cell-view"}
    algorithm_titles = {"bubble": "Bubble", "insertion": "Insertion", "selection": "Selection"}
    for row_idx, algorithm in enumerate(ALGORITHMS):
        for col_idx, mode in enumerate(MODES):
            ax = axes[row_idx, col_idx]
            subset = trace_df[(trace_df["algorithm"] == algorithm) & (trace_df["mode"] == mode)]
            for _, run_df in subset.groupby("repeat_index", sort=False):
                ax.plot(
                    run_df["swap_step"],
                    run_df["sortedness_percent"],
                    color=colors[mode],
                    alpha=0.12,
                    linewidth=0.8,
                )
            mean_df = subset.groupby("swap_step", as_index=False)["sortedness_percent"].mean()
            ax.plot(
                mean_df["swap_step"],
                mean_df["sortedness_percent"],
                color="#111111",
                linewidth=1.7,
            )
            ax.set_title(f"{algorithm_titles[algorithm]} {mode_titles[mode]}", fontsize=12)
            ax.set_ylim(35, 101)
            ax.grid(True, color="#d0d0d0", alpha=0.45, linewidth=0.7)
            if row_idx == 2:
                ax.set_xlabel("Swap step")
            if col_idx == 0:
                ax.set_ylabel("Sortedness (%)")
    fig.suptitle("E01 S04 Figure 3-style sorting trajectories", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    stringified = df.fillna("").astype(str)
    headers = list(stringified.columns)
    rows = stringified.values.tolist()
    widths = [
        max(len(str(header)), *(len(row[col_idx]) for row in rows))
        for col_idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    divider_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider_line, *body_lines])


def update_checksum_file(checksum_path: Path, paths: list[Path]) -> dict[str, str]:
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    order: list[str] = []
    if checksum_path.exists():
        for line in checksum_path.read_text().splitlines():
            if not line.strip():
                continue
            checksum, path = line.split(maxsplit=1)
            existing[path] = checksum
            order.append(path)
    for path in paths:
        resolved = str(path)
        existing[resolved] = sha256_file(path)
        if resolved not in order:
            order.append(resolved)
    checksum_path.write_text("".join(f"{existing[path]}  {path}\n" for path in order), encoding="utf-8")
    return {str(path): existing[str(path)] for path in paths}


def update_run_manifest(
    artifacts_dir: Path,
    paths: dict[str, Path],
    validation: dict[str, Any],
    summary: dict[str, Any],
    repo_dir: Path,
    command: list[str],
) -> None:
    manifest_path = artifacts_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest["researchStepId"] = STEP_ID
    manifest["updatedAtUtc"] = utc_now()
    manifest.setdefault("artifacts", {})
    manifest["artifacts"].update({key: str(path) for key, path in paths.items()})
    manifest.setdefault("checksums", {})
    for path in paths.values():
        if path.exists():
            manifest["checksums"][str(path)] = sha256_file(path)
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": "Reproduce Figure 3-style sorting trajectories",
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": summary["outcome_classification"],
        "artifactsWritten": [str(path) for path in paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommit": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s04_run_figure3_trajectories.py"),
        "command": " ".join(command),
        "summary": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_artifact_manifest(
    manifest_path: Path,
    artifact_paths: dict[str, Path],
    validation: dict[str, Any],
    repo_dir: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "repositoryCommit": git_commit(repo_dir),
        "validationResult": validation["validationResult"],
        "artifacts": [
            {
                "label": label,
                "path": str(path),
                "sha256": sha256_file(path) if path.exists() else None,
                "sizeBytes": path.stat().st_size if path.exists() else None,
            }
            for label, path in artifact_paths.items()
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    report_path: Path,
    cfg: dict[str, Any],
    condition_summary: pd.DataFrame,
    validation: dict[str, Any],
    artifact_paths: dict[str, Path],
    command: list[str],
    repo_dir: Path,
    wall_seconds: float,
    config_sha256: str,
    condition_matrix_sha256: str,
    run_df: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    input_artifact_keys = {"baselineConfig", "conditionMatrix"}
    artifacts_written = [
        str(path)
        for key, path in artifact_paths.items()
        if key not in input_artifact_keys
    ]
    outcome = "supportive" if validation["success"] else "constraining/contradictory"
    caveats = "; ".join(validation["caveatsOrBlockers"])
    lay_summary = (
        "Using the S03 frozen 100 unique-value arrays, all six unperturbed traditional and cell-view "
        "Bubble, Insertion, and Selection conditions were run for 100 matched repeats. Every run reached "
        "100% Sortedness; traditional modes remain reconstructed because the public repository lacks their "
        "original runner."
        if validation["success"]
        else "S04 produced trajectory artifacts, but one or more validation checks failed; see validation details."
    )
    summary_markdown = dataframe_to_markdown(condition_summary)
    stop_table = (
        run_df.groupby(["mode", "algorithm", "stop_reason"], as_index=False)
        .size()
        .rename(columns={"size": "runs"})
    )
    stop_table_markdown = dataframe_to_markdown(stop_table)
    content = f"""# E01 S04 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: {validation["status"]}
- Artifacts written: {", ".join(artifacts_written)}
- Validation result: {validation["validationResult"]}
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Lay summary: {lay_summary}
- Recommended next action: {validation["recommendedNextAction"]}

## Frozen Question

Do traditional and cell-view Bubble, Insertion, and Selection sorts all navigate from random states to fully sorted states with trajectory shapes qualitatively and quantitatively consistent with Figure 3?

## Inputs

- Frozen S03 config: `{artifact_paths["baselineConfig"]}`
- S03 condition matrix: `{artifact_paths["conditionMatrix"]}`
- Config SHA256: `{config_sha256}`
- Condition matrix SHA256: `{condition_matrix_sha256}`
- S04 condition rows: `{", ".join(condition_summary["condition_id"].tolist())}`
- Array bank: `unique_1_to_100`, 100 arrays of length 100.
- Attached paper reference: Figure 3 caption states that each plot contains 100 random no-repeat input trajectories and shows Sortedness moving to the fully sorted state.

## Methods

Traditional Bubble, Insertion, and Selection were run with the S03 reconstructed definitions because S02 found no public traditional-generation runner or saved original arrays. Cell-view Bubble, Insertion, and Selection were run by wrapping the public `modules/multithread/*SortCell.py`, `CellGroup`, and `StatusProbe` classes. For each condition and repeat, the script used the frozen S03 initial array and seed. For cell-view runs, it additionally recorded a deterministic scheduler seed derived from S03 `baseSeed`, condition ID, and repeat index; Python thread scheduling remains non-deterministic at the platform level.

Sortedness used the S03 primary metric: `100 * (1 + ordered adjacent pairs) / n`, with nondecreasing order for this unperturbed Figure 3 baseline. The trajectory table records Sortedness and monotonicity error at the initial state and at every recorded swap step. The Figure 3-style PNG plots 100 replicate trajectories per algorithm/mode panel plus a mean curve.

## Commands

```bash
{" ".join(command)}
```

Validation/smoke commands also run before final execution:

```bash
python -m py_compile scripts/e01_s04_run_figure3_trajectories.py
python scripts/e01_s04_run_figure3_trajectories.py --repo-dir /workspace/cell-research --artifacts-dir /cache/e01_s04_smoke_artifacts --config-path /artifacts/configs/e01_baseline_configs.json --condition-matrix-path /artifacts/tables/e01_condition_matrix.csv --max-repeats 1 --cell-view-timeout-seconds 60
```

## Dependencies And Parameters

- Python executable: `{sys.executable}`
- Python version: `{sys.version.splitlines()[0]}`
- Platform: `{platform.platform()}`
- Pandas version: `{pd.__version__}`
- Matplotlib version: `{matplotlib.__version__}`
- Repository commit at run time: `{git_commit(repo_dir)}`
- Repository status before report write: `{git_status(repo_dir) or "clean"}`
- S03 base seed: `{cfg["globalDefaults"]["baseSeed"]}`
- Repeats per condition: `{validation["expectedRepeatsPerModeAlgorithm"]}`
- Cell-view timeout per run: command-line `--cell-view-timeout-seconds`
- Cell-view max swap guard: command-line `--cell-view-max-swaps`
- Wall time: `{wall_seconds:.3f}` seconds

No new Python, system, R, Rust, or Node dependencies were installed for S04.

## Results

Condition-level numeric summary:

{summary_markdown}

Stop reason counts:

{stop_table_markdown}

The required trace artifact contains `{validation["traceRows"]}` rows. The run-level table used to build the condition summary contains `{validation["runRows"]}` runs.

## Validation

- 100 repetitions per algorithm/mode: `{validation["repeatCountOk"]}`.
- Matched initial arrays across traditional and cell-view modes for each algorithm: `{validation["matchedInitialArraysValid"]}`.
- Final Sortedness reached 100% for every unperturbed run: `{validation["allFinalSortedness100"]}`.
- No cell-view timeout: `{validation["noTimeouts"]}`.
- No live cell or group threads after cleanup: `{validation["noAliveThreadsAfterCleanup"]}`.
- Exactly one final trace row per run: `{validation["exactlyOneFinalTraceRowPerRun"]}`.

Matched-array details:

```json
{json.dumps(validation["matchedInitialArrayDetails"], indent=2)}
```

## Artifacts And Provenance

Reusable outputs:

{chr(10).join(f"- `{label}`: `{path}`" for label, path in artifact_paths.items())}

The global run manifest and checksum file were updated after artifact creation. The S04 artifact manifest records paths, sizes, and SHA256 hashes.

## Caveats, Blockers, Failed Assumptions, And Limitations

- The traditional baselines are reconstructed from the paper's Figure 2 descriptions and S03 frozen assumptions; this is not an exact original traditional runner reproduction.
- Cell-view runs use the public thread classes, but OS/Python scheduling can alter exact trajectories even when initial arrays and scheduler seeds are recorded.
- S04 validates Figure 3-style completion and trajectory availability. It does not settle comparison-count conventions, which are reserved for S05.
- The attached Figure 3 image was not used for pixel-level numeric matching; the extracted caption and Methods text guided this step.

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    s04_dir = artifacts_dir / "research_steps" / STEP_ID
    traces_dir = artifacts_dir / "traces"
    figures_dir = artifacts_dir / "figures" / "e01"
    results_dir = artifacts_dir / "results"
    for path in (s04_dir, traces_dir, figures_dir, results_dir, artifacts_dir / "checksums"):
        path.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.config_path.read_text())
    s04_rows = load_s04_conditions(args.condition_matrix_path)
    input_validation = validate_condition_inputs(cfg, s04_rows, args.max_repeats)
    if not input_validation["inputValidationPassed"]:
        raise SystemExit("Input validation failed: " + "; ".join(input_validation["inputValidationErrors"]))

    repeats_to_run = int(input_validation["repeatsToRun"])
    trace_rows: list[tuple[Any, ...]] = []
    run_records: list[dict[str, Any]] = []
    for row in s04_rows:
        run_records.extend(run_condition(args, cfg, row, repeats_to_run, trace_rows))

    trace_df = pd.DataFrame.from_records(trace_rows, columns=TRACE_COLUMNS)
    run_df = pd.DataFrame(run_records)
    trace_path = traces_dir / "e01_figure3_trajectories.parquet"
    figure_path = figures_dir / "figure3_reproduction.png"
    summary_path = results_dir / "e01_figure3_summary.csv"
    run_records_path = s04_dir / "e01_figure3_run_records.csv"
    validation_path = s04_dir / "s04_validation.json"
    report_path = s04_dir / "research_step_full_results.md"
    artifact_manifest_path = s04_dir / "artifact_manifest.json"

    trace_df.to_parquet(trace_path, index=False)
    plot_figure(trace_df, figure_path)

    validation = validate_outputs(cfg, s04_rows, run_df, trace_df, repeats_to_run)
    condition_summary = build_condition_summary(run_df, validation)
    condition_summary.to_csv(summary_path, index=False)
    run_df.to_csv(run_records_path, index=False)

    artifact_paths = {
        "fullResultsReport": report_path,
        "trajectoryParquet": trace_path,
        "figurePng": figure_path,
        "summaryCsv": summary_path,
        "runRecordsCsv": run_records_path,
        "validationJson": validation_path,
        "artifactManifest": artifact_manifest_path,
        "baselineConfig": args.config_path,
        "conditionMatrix": args.condition_matrix_path,
    }
    summary_payload = {
        "outcome_classification": "supportive" if validation["success"] else "constraining/contradictory",
        "conditionSummaryRows": int(len(condition_summary)),
        "traceRows": int(len(trace_df)),
        "runRows": int(len(run_df)),
        "meanSwapStepsByModeAlgorithm": {
            f"{mode}:{algorithm}": float(value)
            for (mode, algorithm), value in run_df.groupby(["mode", "algorithm"])["swap_count"].mean().items()
        },
    }
    validation.update(
        {
            "artifactsWritten": [str(path) for path in artifact_paths.values() if path not in {args.config_path, args.condition_matrix_path}],
            "commands": [" ".join(sys.argv)],
            "inputValidation": input_validation,
            "outcomeClassification": summary_payload["outcome_classification"],
            "createdAtUtc": utc_now(),
            "repositoryCommit": git_commit(args.repo_dir),
            "script": str(args.repo_dir / "scripts/e01_s04_run_figure3_trajectories.py"),
        }
    )
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    wall_seconds = time.monotonic() - started
    write_report(
        report_path,
        cfg,
        condition_summary,
        validation,
        artifact_paths,
        sys.argv,
        args.repo_dir,
        wall_seconds,
        sha256_file(args.config_path),
        sha256_file(args.condition_matrix_path),
        run_df,
    )
    write_artifact_manifest(artifact_manifest_path, artifact_paths, validation, args.repo_dir)

    checksum_paths = list(artifact_paths.values()) + [args.research_plan_path, args.repo_dir / "scripts/e01_s04_run_figure3_trajectories.py"]
    checksum_paths = [path for path in checksum_paths if path.exists()]
    checksum_path = artifacts_dir / "checksums" / "sha256sums.txt"
    update_checksum_file(checksum_path, checksum_paths)

    run_manifest_paths = {
        "s04FullResultsReport": report_path,
        "s04TrajectoryParquet": trace_path,
        "s04Figure": figure_path,
        "s04SummaryCsv": summary_path,
        "s04RunRecords": run_records_path,
        "s04Validation": validation_path,
        "s04ArtifactManifest": artifact_manifest_path,
        "s04Script": args.repo_dir / "scripts/e01_s04_run_figure3_trajectories.py",
    }
    update_run_manifest(
        artifacts_dir,
        run_manifest_paths,
        validation,
        summary_payload,
        args.repo_dir,
        sys.argv,
    )
    update_checksum_file(checksum_path, [artifacts_dir / "run_manifest.json", checksum_path])

    print(json.dumps({"validation": validation, "summary": summary_payload}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
