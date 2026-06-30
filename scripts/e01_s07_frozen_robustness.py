#!/usr/bin/env python3
"""Run E01 S07 Frozen Cell robustness replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STEP_ID = "S07"
STEP_NUMBER = 7
EXPERIMENT_ID = "E01"
ALGORITHMS = ("bubble", "insertion", "selection")
MODES = ("traditional", "cell_view")
FROZEN_SEMANTICS = ("passive", "stuck")
FROZEN_COUNTS = (1, 2, 3)
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
    "frozen_semantics",
    "frozen_count",
    "frozen_index_seed",
    "initial_frozen_indices_json",
    "swap_step",
    "comparison_count_at_step",
    "frozen_attempt_count_at_step",
    "sortedness_percent",
    "monotonicity_error_count",
    "is_initial",
    "is_final",
    "stop_reason",
    "max_guard_hit",
    "final_values_json",
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
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/e01_s07"))
    parser.add_argument("--max-repeats", type=int, default=None, help="Optional smoke-test cap per condition.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-successful-swaps", type=int, default=15000)
    parser.add_argument("--max-sweeps", type=int, default=20000)
    parser.add_argument("--stall-sweeps", type=int, default=2)
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


def nondecreasing(values: list[int]) -> bool:
    return monotonicity_error_count(values) == 0


def metric_record(
    values: list[int],
    swap_step: int,
    comparison_count: int | None,
    frozen_attempt_count: int | None,
) -> dict[str, Any]:
    return {
        "values": list(values),
        "swap_step": int(swap_step),
        "comparison_count": comparison_count,
        "frozen_attempt_count": frozen_attempt_count,
        "sortedness_percent": sortedness_percent(values),
        "monotonicity_error_count": monotonicity_error_count(values),
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


def scheduler_seed(base_seed: int, condition_id: str, repeat_idx: int) -> int:
    condition_num = int(condition_id.replace("E01C", ""))
    return base_seed * 1_000_000 + 70 * 10_000 + condition_num * 1_000 + repeat_idx


def load_s07_conditions(condition_matrix_path: Path) -> list[dict[str, str]]:
    with condition_matrix_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    s07_rows = [row for row in rows if "S07" in row["step_scope"].split(",")]
    return sorted(
        s07_rows,
        key=lambda row: (
            row["frozen_semantics"],
            int(row["frozen_count"]),
            row["algorithm"],
            row["mode"],
        ),
    )


def validate_condition_inputs(cfg: dict[str, Any], rows: list[dict[str, str]], max_repeats: int | None) -> dict[str, Any]:
    defaults = cfg["globalDefaults"]
    repeat_count = int(defaults["repeatCount"])
    repeats_to_run = min(max_repeats or repeat_count, repeat_count)
    errors: list[str] = []
    expected_combinations = {
        (mode, algorithm, semantics, frozen_count)
        for mode in MODES
        for algorithm in ALGORITHMS
        for semantics in FROZEN_SEMANTICS
        for frozen_count in FROZEN_COUNTS
    }
    observed_combinations = {
        (row["mode"], row["algorithm"], row["frozen_semantics"], int(row["frozen_count"]))
        for row in rows
    }
    if len(rows) != len(expected_combinations):
        errors.append(f"Expected {len(expected_combinations)} S07 condition rows; found {len(rows)}.")
    if observed_combinations != expected_combinations:
        errors.append(
            "S07 condition combination set mismatch: "
            f"missing={sorted(expected_combinations - observed_combinations)}, "
            f"extra={sorted(observed_combinations - expected_combinations)}."
        )
    value_banks = cfg["seedBanks"]["valueBanks"]
    frozen_banks = cfg["seedBanks"]["frozenIndexBanks"]
    for row in rows:
        if int(row["repeat_count"]) != repeat_count:
            errors.append(f"{row['condition_id']} repeat_count={row['repeat_count']} differs from config repeatCount={repeat_count}.")
        value_bank = value_banks.get(row["value_bank_id"])
        if not value_bank:
            errors.append(f"{row['condition_id']} missing value bank {row['value_bank_id']}.")
            continue
        if len(value_bank["initialArrays"]) < repeats_to_run or len(value_bank["seeds"]) < repeats_to_run:
            errors.append(f"{row['condition_id']} value bank {row['value_bank_id']} has fewer than {repeats_to_run} repeats.")
        frozen_bank = frozen_banks.get(row["frozen_index_bank_id"])
        if not frozen_bank:
            errors.append(f"{row['condition_id']} missing frozen index bank {row['frozen_index_bank_id']}.")
            continue
        frozen_count = int(row["frozen_count"])
        if int(frozen_bank["frozenCount"]) != frozen_count:
            errors.append(f"{row['condition_id']} frozen bank count mismatch.")
        if len(frozen_bank["indices"]) < repeats_to_run or len(frozen_bank["seeds"]) < repeats_to_run:
            errors.append(f"{row['condition_id']} frozen bank {row['frozen_index_bank_id']} has fewer than {repeats_to_run} repeats.")
        for repeat_idx, indices in enumerate(frozen_bank["indices"][:repeats_to_run]):
            if len(indices) != frozen_count or len(set(indices)) != frozen_count:
                errors.append(f"{row['condition_id']} repeat {repeat_idx} frozen index count mismatch.")
            if any(idx < 0 or idx >= int(defaults["arrayLength"]) for idx in indices):
                errors.append(f"{row['condition_id']} repeat {repeat_idx} frozen index out of range.")
    return {
        "expectedConditionRows": len(expected_combinations),
        "observedConditionRows": len(rows),
        "expectedRepeatsPerCondition": repeat_count,
        "repeatsToRun": repeats_to_run,
        "inputValidationErrors": errors,
        "inputValidationPassed": not errors,
    }


def make_traditional_cells(values: list[int], frozen_indices: list[int]) -> list[dict[str, Any]]:
    frozen_set = set(frozen_indices)
    return [
        {
            "id": idx,
            "value": int(value),
            "frozen": idx in frozen_set,
        }
        for idx, value in enumerate(values)
    ]


def traditional_values(cells: list[dict[str, Any]]) -> list[int]:
    return [int(cell["value"]) for cell in cells]


def traditional_frozen_positions(cells: list[dict[str, Any]]) -> list[int]:
    return [idx for idx, cell in enumerate(cells) if cell["frozen"]]


def traditional_frozen_values(cells: list[dict[str, Any]]) -> list[int]:
    return [int(cell["value"]) for cell in cells if cell["frozen"]]


def traditional_swap_allowed(
    cells: list[dict[str, Any]],
    left_idx: int,
    right_idx: int,
    active_idx: int,
    semantics: str,
) -> tuple[bool, bool]:
    active_frozen = bool(cells[active_idx]["frozen"])
    target_frozen = bool(cells[right_idx if active_idx == left_idx else left_idx]["frozen"])
    if active_frozen:
        return False, True
    if semantics == "stuck" and (cells[left_idx]["frozen"] or cells[right_idx]["frozen"]):
        return False, True
    return True, target_frozen


def traditional_bubble_frozen(values: list[int], frozen_indices: list[int], semantics: str) -> dict[str, Any]:
    cells = make_traditional_cells(values, frozen_indices)
    comparisons = 0
    swaps = 0
    frozen_attempts = 0
    records = [metric_record(traditional_values(cells), swaps, comparisons, frozen_attempts)]
    while True:
        moved = False
        idx = 0
        while idx < len(cells) - 1:
            comparisons += 1
            if cells[idx]["value"] <= cells[idx + 1]["value"]:
                idx += 1
                continue
            allowed, blocked_by_frozen = traditional_swap_allowed(cells, idx, idx + 1, idx, semantics)
            if not allowed:
                frozen_attempts += int(blocked_by_frozen)
                idx += 1
                continue
            cells[idx], cells[idx + 1] = cells[idx + 1], cells[idx]
            swaps += 1
            moved = True
            records.append(metric_record(traditional_values(cells), swaps, comparisons, frozen_attempts))
            bubble_idx = idx + 1
            while bubble_idx < len(cells) - 1:
                comparisons += 1
                if cells[bubble_idx]["value"] <= cells[bubble_idx + 1]["value"]:
                    break
                allowed, blocked_by_frozen = traditional_swap_allowed(cells, bubble_idx, bubble_idx + 1, bubble_idx, semantics)
                if not allowed:
                    frozen_attempts += int(blocked_by_frozen)
                    break
                cells[bubble_idx], cells[bubble_idx + 1] = cells[bubble_idx + 1], cells[bubble_idx]
                swaps += 1
                records.append(metric_record(traditional_values(cells), swaps, comparisons, frozen_attempts))
                bubble_idx += 1
            idx += 1
        if not moved:
            break
    return finish_traditional_result(cells, records, swaps, comparisons, frozen_attempts, "no_legal_move_twice")


def traditional_insertion_frozen(values: list[int], frozen_indices: list[int], semantics: str) -> dict[str, Any]:
    cells = make_traditional_cells(values, frozen_indices)
    comparisons = 0
    swaps = 0
    frozen_attempts = 0
    records = [metric_record(traditional_values(cells), swaps, comparisons, frozen_attempts)]
    while True:
        moved = False
        for unsorted_idx in range(1, len(cells)):
            idx = unsorted_idx
            while idx > 0:
                comparisons += 1
                if cells[idx - 1]["value"] <= cells[idx]["value"]:
                    break
                allowed, blocked_by_frozen = traditional_swap_allowed(cells, idx - 1, idx, idx, semantics)
                if not allowed:
                    frozen_attempts += int(blocked_by_frozen)
                    break
                cells[idx - 1], cells[idx] = cells[idx], cells[idx - 1]
                swaps += 1
                moved = True
                records.append(metric_record(traditional_values(cells), swaps, comparisons, frozen_attempts))
                idx -= 1
        if not moved:
            break
    stop_reason = "sorted" if nondecreasing(traditional_values(cells)) else "no_legal_move_twice"
    return finish_traditional_result(cells, records, swaps, comparisons, frozen_attempts, stop_reason)


def traditional_selection_frozen(values: list[int], frozen_indices: list[int], semantics: str) -> dict[str, Any]:
    cells = make_traditional_cells(values, frozen_indices)
    comparisons = 0
    swaps = 0
    frozen_attempts = 0
    records = [metric_record(traditional_values(cells), swaps, comparisons, frozen_attempts)]
    for boundary in range(len(cells)):
        candidates = [idx for idx in range(boundary, len(cells)) if not cells[idx]["frozen"]]
        if not candidates:
            continue
        min_idx = candidates[0]
        for idx in candidates[1:]:
            comparisons += 1
            if cells[idx]["value"] < cells[min_idx]["value"]:
                min_idx = idx
        if min_idx == boundary:
            continue
        allowed, blocked_by_frozen = traditional_swap_allowed(cells, boundary, min_idx, min_idx, semantics)
        if not allowed:
            frozen_attempts += int(blocked_by_frozen)
            continue
        cells[boundary], cells[min_idx] = cells[min_idx], cells[boundary]
        swaps += 1
        records.append(metric_record(traditional_values(cells), swaps, comparisons, frozen_attempts))
    stop_reason = "sorted" if nondecreasing(traditional_values(cells)) else "no_legal_move_twice"
    return finish_traditional_result(cells, records, swaps, comparisons, frozen_attempts, stop_reason)


def finish_traditional_result(
    cells: list[dict[str, Any]],
    records: list[dict[str, Any]],
    swaps: int,
    comparisons: int,
    frozen_attempts: int,
    stop_reason: str,
) -> dict[str, Any]:
    final_values = traditional_values(cells)
    if records[-1]["values"] != final_values:
        records.append(metric_record(final_values, swaps, comparisons, frozen_attempts))
    if nondecreasing(final_values):
        stop_reason = "sorted"
    return {
        "records": records,
        "final_values": final_values,
        "swap_count": int(swaps),
        "comparison_count": int(comparisons),
        "compare_plus_swap_count": int(swaps + comparisons),
        "frozen_attempt_count": int(frozen_attempts),
        "stop_reason": stop_reason,
        "max_guard_hit": False,
        "sweep_count": None,
        "final_frozen_positions": traditional_frozen_positions(cells),
        "final_frozen_values": traditional_frozen_values(cells),
        "wrapper_name": "reconstructed_traditional_frozen_controller",
    }


TRADITIONAL_FROZEN_RUNNERS = {
    "bubble": traditional_bubble_frozen,
    "insertion": traditional_insertion_frozen,
    "selection": traditional_selection_frozen,
}


def patch_stuck_swap(cells: list[Any], frozen_thread_ids: set[int]) -> None:
    def make_swap(cell: Any):
        original_swap = cell.swap

        def stuck_swap(target_position: tuple[int, int], skip_stats: bool = False) -> Any:
            if cell.status == cell_status_enum.FREEZE:
                if not cell.tried_to_swap_with_frozen:
                    cell.status_probe.count_frozen_cell_attempt()
                    cell.tried_to_swap_with_frozen = True
                return None
            target = cell.cells[int(target_position[0])]
            if target.threadID in frozen_thread_ids or target.status == cell_status_enum.FREEZE:
                if not cell.tried_to_swap_with_frozen:
                    cell.status_probe.count_frozen_cell_attempt()
                    cell.tried_to_swap_with_frozen = True
                return None
            cell.tried_to_swap_with_frozen = False
            target.tried_to_swap_with_frozen = False
            return original_swap(target_position, skip_stats)

        return stuck_swap

    cell_status_enum = cells[0].status.__class__
    for cell in cells:
        cell.swap = make_swap(cell)


def cell_view_values(cells: list[Any]) -> list[int]:
    return [int(cell.value) for cell in cells]


def cell_view_frozen_positions(cells: list[Any], cell_status: Any) -> list[int]:
    return [int(cell.current_position[0]) for cell in cells if cell.status == cell_status.FREEZE]


def cell_view_frozen_values(cells: list[Any], cell_status: Any) -> list[int]:
    return [int(cell.value) for cell in cells if cell.status == cell_status.FREEZE]


def cell_state_signature(cells: list[Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            int(cell.threadID),
            int(cell.value),
            str(cell.status),
            int(cell.current_position[0]),
            int(cell.ideal_position[0]) if cell.ideal_position is not None else None,
        )
        for cell in cells
    )


def has_public_legal_swap(cells: list[Any], algorithm: str, semantics: str, cell_status: Any) -> bool:
    if algorithm == "selection":
        for cell in cells:
            if cell.status != cell_status.ACTIVE or cell.current_position == cell.ideal_position:
                continue
            target_idx = int(cell.ideal_position[0])
            if target_idx < 0 or target_idx >= len(cells):
                continue
            target = cells[target_idx]
            if target.status == cell_status.ACTIVE and cell.value < target.value:
                return True
            if semantics == "passive" and target.status == cell_status.FREEZE and cell.value < target.value:
                return True
        return False
    if algorithm == "bubble":
        for cell in cells:
            if cell.status != cell_status.ACTIVE:
                continue
            current_idx = int(cell.current_position[0])
            for target_idx, check_right in ((current_idx - 1, False), (current_idx + 1, True)):
                if target_idx < 0 or target_idx >= len(cells):
                    continue
                target = cells[target_idx]
                if target.status != cell_status.ACTIVE and not (semantics == "passive" and target.status == cell_status.FREEZE):
                    continue
                if check_right and cell.value > target.value:
                    return True
                if not check_right and cell.value < target.value:
                    return True
        return False
    if algorithm == "insertion":
        for cell in cells:
            if cell.status != cell_status.ACTIVE:
                continue
            current_idx = int(cell.current_position[0])
            target_idx = current_idx - 1
            if target_idx < 0:
                continue
            target = cells[target_idx]
            if target.status != cell_status.ACTIVE and not (semantics == "passive" and target.status == cell_status.FREEZE):
                continue
            if cell.is_enable_to_move() and cell.value < target.value:
                return True
        return False
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def run_cell_view_frozen_sync(
    repo_dir: Path,
    algorithm: str,
    values: list[int],
    frozen_indices: list[int],
    semantics: str,
    seed: int,
    max_successful_swaps: int,
    max_sweeps: int,
    stall_sweeps: int,
) -> dict[str, Any]:
    modules = import_cell_modules(repo_dir)
    cls_by_algorithm = {
        "bubble": modules["BubbleSortCell"],
        "insertion": modules["InsertionSortCell"],
        "selection": modules["SelectionSortCell"],
    }
    cell_status = modules["CellStatus"]
    random.seed(seed)
    lock = threading.RLock()
    status_probe = modules["StatusProbe"]()
    left_boundary = (0, 1)
    right_boundary = (len(values) - 1, 1)
    cells: list[Any] = []
    cls = cls_by_algorithm[algorithm]
    for idx, value in enumerate(values):
        cell = cls(
            idx + 1,
            int(value),
            lock,
            (idx, 1),
            cells,
            left_boundary,
            right_boundary,
            status_probe,
            disable_visualization=True,
        )
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
    for cell in cells:
        cell.group = group
    for idx in frozen_indices:
        cells[idx].set_cell_to_freeze()
    frozen_thread_ids = {cells[idx].threadID for idx in frozen_indices}
    if semantics == "stuck":
        patch_stuck_swap(cells, frozen_thread_ids)

    sweeps = 0
    no_progress_sweeps = 0
    stop_reason = "max_sweeps_exceeded"
    max_guard_hit = False
    while sweeps < max_sweeps:
        current_values = cell_view_values(cells)
        if nondecreasing(current_values):
            stop_reason = "sorted"
            break
        if status_probe.swap_count > max_successful_swaps:
            stop_reason = "max_successful_swaps_exceeded"
            max_guard_hit = True
            break
        legal_swap_exists = has_public_legal_swap(cells, algorithm, semantics, cell_status)
        before_swap_count = status_probe.swap_count
        before_signature = cell_state_signature(cells)
        order = list(range(len(cells)))
        random.shuffle(order)
        for idx in order:
            cell = cells[idx]
            if cell.status != cell_status.FREEZE:
                cell.move()
        sweeps += 1
        after_signature = cell_state_signature(cells)
        made_progress = status_probe.swap_count != before_swap_count or after_signature != before_signature
        if not made_progress and (algorithm == "selection" or not legal_swap_exists):
            no_progress_sweeps += 1
            if no_progress_sweeps >= stall_sweeps:
                stop_reason = "no_legal_move_twice"
                break
        else:
            no_progress_sweeps = 0
    else:
        max_guard_hit = True

    final_values = cell_view_values(cells)
    if nondecreasing(final_values):
        stop_reason = "sorted"
        max_guard_hit = False
    records = [metric_record(values, 0, 0, 0)]
    for step_idx, snapshot in enumerate(status_probe.sorting_steps, start=1):
        records.append(metric_record(snapshot, step_idx, None, None))
    if records[-1]["values"] != final_values:
        records.append(metric_record(final_values, status_probe.swap_count, status_probe.compare_and_swap_count, status_probe.frozen_swap_attempts))
    else:
        records[-1]["comparison_count"] = status_probe.compare_and_swap_count
        records[-1]["frozen_attempt_count"] = status_probe.frozen_swap_attempts
    return {
        "records": records,
        "final_values": final_values,
        "swap_count": int(status_probe.swap_count),
        "comparison_count": int(status_probe.compare_and_swap_count),
        "compare_plus_swap_count": int(status_probe.swap_count + status_probe.compare_and_swap_count),
        "frozen_attempt_count": int(status_probe.frozen_swap_attempts),
        "stop_reason": stop_reason,
        "max_guard_hit": bool(max_guard_hit),
        "sweep_count": int(sweeps),
        "final_frozen_positions": sorted(cell_view_frozen_positions(cells, cell_status)),
        "final_frozen_values": sorted(cell_view_frozen_values(cells, cell_status)),
        "wrapper_name": "deterministic_single_thread_public_cell_methods",
    }


def append_trace_rows(
    trace_rows: list[tuple[Any, ...]],
    row: dict[str, str],
    repeat_idx: int,
    initial_seed: int,
    scheduler_seed_value: int | None,
    frozen_indices: list[int],
    frozen_seed: int,
    initial_values: list[int],
    run_result: dict[str, Any],
) -> None:
    initial_hash = sha256_json(initial_values)
    final_hash = sha256_json(run_result["final_values"])
    records = list(run_result["records"])
    last_idx = len(records) - 1
    frozen_indices_json = json.dumps(list(map(int, frozen_indices)), separators=(",", ":"))
    for idx, record in enumerate(records):
        is_final = idx == last_idx
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
                row["frozen_semantics"],
                int(row["frozen_count"]),
                frozen_seed,
                frozen_indices_json,
                int(record["swap_step"]),
                record["comparison_count"],
                record["frozen_attempt_count"],
                float(record["sortedness_percent"]),
                int(record["monotonicity_error_count"]),
                idx == 0,
                is_final,
                run_result["stop_reason"],
                bool(run_result["max_guard_hit"]),
                json.dumps(run_result["final_values"], separators=(",", ":")) if is_final else None,
                initial_hash,
                final_hash,
            )
        )


def build_run_record(
    row: dict[str, str],
    repeat_idx: int,
    initial_seed: int,
    scheduler_seed_value: int | None,
    frozen_indices: list[int],
    frozen_seed: int,
    initial_values: list[int],
    run_result: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    final_values = list(map(int, run_result["final_values"]))
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "condition_id": row["condition_id"],
        "mode": row["mode"],
        "algorithm": row["algorithm"],
        "baseline_source": row["baseline_source"],
        "matched_group_id": row["matched_group_id"],
        "value_bank_id": row["value_bank_id"],
        "repeat_index": int(repeat_idx),
        "initial_array_seed": int(initial_seed),
        "scheduler_seed": scheduler_seed_value,
        "initial_array_sha256": sha256_json(initial_values),
        "final_array_sha256": sha256_json(final_values),
        "frozen_semantics": row["frozen_semantics"],
        "frozen_count": int(row["frozen_count"]),
        "frozen_index_bank_id": row["frozen_index_bank_id"],
        "frozen_index_seed": int(frozen_seed),
        "initial_frozen_indices_json": json.dumps(list(map(int, frozen_indices)), separators=(",", ":")),
        "final_frozen_positions_json": json.dumps(list(map(int, run_result["final_frozen_positions"])), separators=(",", ":")),
        "final_frozen_values_json": json.dumps(list(map(int, run_result["final_frozen_values"])), separators=(",", ":")),
        "final_values_json": json.dumps(final_values, separators=(",", ":")),
        "swap_count": int(run_result["swap_count"]),
        "comparison_count": int(run_result["comparison_count"]),
        "compare_plus_swap_count": int(run_result["compare_plus_swap_count"]),
        "frozen_attempt_count": int(run_result["frozen_attempt_count"]),
        "sweep_count": run_result["sweep_count"],
        "final_sortedness_percent": sortedness_percent(final_values),
        "final_monotonicity_error_count": monotonicity_error_count(final_values),
        "stop_reason": run_result["stop_reason"],
        "max_guard_hit": bool(run_result["max_guard_hit"]),
        "elapsed_seconds": float(elapsed_seconds),
        "wrapper_name": run_result["wrapper_name"],
    }


def run_condition_worker(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload["cfg"]
    row = payload["row"]
    repo_dir = Path(payload["repo_dir"])
    condition_cache_dir = Path(payload["condition_cache_dir"])
    repeats_to_run = int(payload["repeats_to_run"])
    max_successful_swaps = int(payload["max_successful_swaps"])
    max_sweeps = int(payload["max_sweeps"])
    stall_sweeps = int(payload["stall_sweeps"])
    condition_cache_dir.mkdir(parents=True, exist_ok=True)
    value_bank = cfg["seedBanks"]["valueBanks"][row["value_bank_id"]]
    frozen_bank = cfg["seedBanks"]["frozenIndexBanks"][row["frozen_index_bank_id"]]
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    trace_rows: list[tuple[Any, ...]] = []
    run_records: list[dict[str, Any]] = []
    for repeat_idx in range(repeats_to_run):
        initial_values = list(map(int, value_bank["initialArrays"][repeat_idx]))
        initial_seed = int(value_bank["seeds"][repeat_idx])
        frozen_indices = list(map(int, frozen_bank["indices"][repeat_idx]))
        frozen_seed = int(frozen_bank["seeds"][repeat_idx])
        started = time.monotonic()
        if row["mode"] == "traditional":
            scheduler_seed_value = None
            run_result = TRADITIONAL_FROZEN_RUNNERS[row["algorithm"]](initial_values, frozen_indices, row["frozen_semantics"])
        else:
            scheduler_seed_value = scheduler_seed(base_seed, row["condition_id"], repeat_idx)
            run_result = run_cell_view_frozen_sync(
                repo_dir,
                row["algorithm"],
                initial_values,
                frozen_indices,
                row["frozen_semantics"],
                scheduler_seed_value,
                max_successful_swaps,
                max_sweeps,
                stall_sweeps,
            )
        elapsed_seconds = time.monotonic() - started
        append_trace_rows(
            trace_rows,
            row,
            repeat_idx,
            initial_seed,
            scheduler_seed_value,
            frozen_indices,
            frozen_seed,
            initial_values,
            run_result,
        )
        run_records.append(
            build_run_record(
                row,
                repeat_idx,
                initial_seed,
                scheduler_seed_value,
                frozen_indices,
                frozen_seed,
                initial_values,
                run_result,
                elapsed_seconds,
            )
        )
    trace_path = condition_cache_dir / f"{row['condition_id']}_trace.parquet"
    run_path = condition_cache_dir / f"{row['condition_id']}_runs.parquet"
    pd.DataFrame.from_records(trace_rows, columns=TRACE_COLUMNS).to_parquet(trace_path, index=False)
    pd.DataFrame.from_records(run_records).to_parquet(run_path, index=False)
    return {
        "condition_id": row["condition_id"],
        "mode": row["mode"],
        "algorithm": row["algorithm"],
        "frozen_semantics": row["frozen_semantics"],
        "frozen_count": int(row["frozen_count"]),
        "trace_path": str(trace_path),
        "run_path": str(run_path),
        "run_rows": len(run_records),
        "trace_rows": len(trace_rows),
    }


def read_condition_outputs(worker_results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace_frames = [pd.read_parquet(result["trace_path"]) for result in worker_results]
    run_frames = [pd.read_parquet(result["run_path"]) for result in worker_results]
    trace_df = pd.concat(trace_frames, ignore_index=True)
    run_df = pd.concat(run_frames, ignore_index=True)
    trace_df.sort_values(["condition_id", "repeat_index", "swap_step", "is_final"], inplace=True)
    run_df.sort_values(["condition_id", "repeat_index"], inplace=True)
    return trace_df, run_df


def build_condition_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stop_counts = run_df.groupby("condition_id")["stop_reason"].apply(lambda values: json.dumps(dict(Counter(values)), sort_keys=True)).to_dict()
    for condition_id, condition_df in run_df.groupby("condition_id", sort=True):
        first = condition_df.iloc[0]
        rows.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "mode": first["mode"],
                "algorithm": first["algorithm"],
                "frozen_semantics": first["frozen_semantics"],
                "frozen_count": int(first["frozen_count"]),
                "baseline_source": first["baseline_source"],
                "matched_group_id": first["matched_group_id"],
                "repetitions_observed": int(condition_df["repeat_index"].nunique()),
                "mean_final_monotonicity_error": float(condition_df["final_monotonicity_error_count"].mean()),
                "std_final_monotonicity_error": float(condition_df["final_monotonicity_error_count"].std(ddof=1)),
                "min_final_monotonicity_error": int(condition_df["final_monotonicity_error_count"].min()),
                "max_final_monotonicity_error": int(condition_df["final_monotonicity_error_count"].max()),
                "mean_final_sortedness_percent": float(condition_df["final_sortedness_percent"].mean()),
                "mean_swap_count": float(condition_df["swap_count"].mean()),
                "mean_compare_plus_swap_count": float(condition_df["compare_plus_swap_count"].mean()),
                "mean_frozen_attempt_count": float(condition_df["frozen_attempt_count"].mean()),
                "max_guard_runs": int(condition_df["max_guard_hit"].sum()),
                "stop_reason_counts": stop_counts[condition_id],
                "wrapper_name": first["wrapper_name"],
            }
        )
    return pd.DataFrame(rows)


def build_comparison_summary(condition_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = condition_summary.pivot_table(
        index=["algorithm", "frozen_semantics", "frozen_count"],
        columns="mode",
        values="mean_final_monotonicity_error",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot["cell_minus_traditional_mean_error"] = pivot["cell_view"] - pivot["traditional"]
    pivot["cell_lower_error_than_traditional"] = pivot["cell_view"] < pivot["traditional"]
    return pivot.sort_values(["frozen_semantics", "frozen_count", "algorithm"]).reset_index(drop=True)


def validate_outputs(
    cfg: dict[str, Any],
    rows: list[dict[str, str]],
    run_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    repeats_to_run: int,
    condition_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
) -> dict[str, Any]:
    expected_run_rows = len(rows) * repeats_to_run
    observed_combo_counts = run_df.groupby(["mode", "algorithm", "frozen_semantics", "frozen_count"])["repeat_index"].nunique().to_dict()
    expected_combo_counts = {
        (mode, algorithm, semantics, frozen_count): repeats_to_run
        for mode in MODES
        for algorithm in ALGORITHMS
        for semantics in FROZEN_SEMANTICS
        for frozen_count in FROZEN_COUNTS
    }
    repeat_count_ok = observed_combo_counts == expected_combo_counts
    final_rows = trace_df[trace_df["is_final"]].copy()
    exactly_one_final_trace_row = bool(
        final_rows.groupby(["condition_id", "repeat_index"]).size().eq(1).all()
        and len(final_rows) == len(run_df)
    )
    final_rows["recomputed_final_monotonicity_error_count"] = final_rows["final_values_json"].map(lambda payload: monotonicity_error_count(json.loads(payload)))
    final_error_join = run_df.merge(
        final_rows[
            [
                "condition_id",
                "repeat_index",
                "recomputed_final_monotonicity_error_count",
                "final_array_sha256",
            ]
        ],
        on=["condition_id", "repeat_index", "final_array_sha256"],
        how="left",
        validate="one_to_one",
    )
    final_error_recomputed_ok = bool(
        final_error_join["recomputed_final_monotonicity_error_count"].notna().all()
        and (
            final_error_join["recomputed_final_monotonicity_error_count"].astype(int)
            == final_error_join["final_monotonicity_error_count"].astype(int)
        ).all()
    )
    frozen_counts_ok = True
    stuck_positions_fixed = True
    frozen_details: list[dict[str, Any]] = []
    for _, record in run_df.iterrows():
        initial_frozen = json.loads(record["initial_frozen_indices_json"])
        final_frozen = json.loads(record["final_frozen_positions_json"])
        frozen_count = int(record["frozen_count"])
        count_ok = len(initial_frozen) == frozen_count and len(final_frozen) == frozen_count
        frozen_counts_ok = frozen_counts_ok and count_ok
        if record["frozen_semantics"] == "stuck":
            fixed = sorted(initial_frozen) == sorted(final_frozen)
            stuck_positions_fixed = stuck_positions_fixed and fixed
        else:
            fixed = None
        if len(frozen_details) < 12 and (not count_ok or fixed is False):
            frozen_details.append(
                {
                    "conditionId": record["condition_id"],
                    "repeatIndex": int(record["repeat_index"]),
                    "semantics": record["frozen_semantics"],
                    "frozenCount": frozen_count,
                    "initialFrozen": initial_frozen,
                    "finalFrozen": final_frozen,
                    "countOk": count_ok,
                    "stuckFixed": fixed,
                }
            )
    matched_initial_details: list[dict[str, Any]] = []
    matched_initial_valid = True
    for algorithm in ALGORITHMS:
        for semantics in FROZEN_SEMANTICS:
            for frozen_count in FROZEN_COUNTS:
                subset = run_df[
                    (run_df["algorithm"] == algorithm)
                    & (run_df["frozen_semantics"] == semantics)
                    & (run_df["frozen_count"] == frozen_count)
                ]
                trad = subset[subset["mode"] == "traditional"].sort_values("repeat_index")
                cell = subset[subset["mode"] == "cell_view"].sort_values("repeat_index")
                same_hash = trad["initial_array_sha256"].tolist() == cell["initial_array_sha256"].tolist()
                same_seed = trad["initial_array_seed"].tolist() == cell["initial_array_seed"].tolist()
                same_frozen = trad["initial_frozen_indices_json"].tolist() == cell["initial_frozen_indices_json"].tolist()
                valid = same_hash and same_seed and same_frozen and len(trad) == repeats_to_run and len(cell) == repeats_to_run
                matched_initial_valid = matched_initial_valid and valid
                matched_initial_details.append(
                    {
                        "algorithm": algorithm,
                        "frozenSemantics": semantics,
                        "frozenCount": frozen_count,
                        "sameInitialArrayHashesByRepeat": same_hash,
                        "sameInitialArraySeedsByRepeat": same_seed,
                        "sameFrozenIndicesByRepeat": same_frozen,
                        "repeatCountCompared": min(len(trad), len(cell)),
                        "valid": valid,
                    }
                )
    no_max_guard = bool((~run_df["max_guard_hit"]).all())
    observed_semantics = sorted(run_df["frozen_semantics"].unique().tolist())
    observed_frozen_counts = sorted(int(value) for value in run_df["frozen_count"].unique().tolist())
    observed_algorithms = sorted(run_df["algorithm"].unique().tolist())
    observed_modes = sorted(run_df["mode"].unique().tolist())
    figure5_cell_lower = bool(comparison_summary["cell_lower_error_than_traditional"].all())
    cell_summary = condition_summary[condition_summary["mode"] == "cell_view"].copy()
    passive_best_rows = []
    stuck_best_rows = []
    for frozen_count in FROZEN_COUNTS:
        passive = cell_summary[(cell_summary["frozen_semantics"] == "passive") & (cell_summary["frozen_count"] == frozen_count)].sort_values("mean_final_monotonicity_error")
        stuck = cell_summary[(cell_summary["frozen_semantics"] == "stuck") & (cell_summary["frozen_count"] == frozen_count)].sort_values("mean_final_monotonicity_error")
        passive_best_rows.append(passive.iloc[0]["algorithm"] == "bubble")
        stuck_best_rows.append(stuck.iloc[0]["algorithm"] == "selection")
    passive_bubble_best = bool(all(passive_best_rows))
    stuck_selection_best = bool(all(stuck_best_rows))
    success = bool(
        len(run_df) == expected_run_rows
        and repeat_count_ok
        and exactly_one_final_trace_row
        and final_error_recomputed_ok
        and frozen_counts_ok
        and stuck_positions_fixed
        and matched_initial_valid
        and no_max_guard
        and observed_semantics == list(FROZEN_SEMANTICS)
        and observed_frozen_counts == list(FROZEN_COUNTS)
        and observed_algorithms == sorted(ALGORITHMS)
        and observed_modes == sorted(MODES)
        and repeats_to_run == int(cfg["globalDefaults"]["repeatCount"])
    )
    outcome_classification = "supportive" if (success and figure5_cell_lower and passive_bubble_best and stuck_selection_best) else "constraining/contradictory"
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed_with_caveats" if success else "completed_validation_failed",
        "validationResult": "passed_with_wrapper_and_reconstructed_traditional_caveats" if success else "failed",
        "outcomeClassification": outcome_classification,
        "expectedConditionRows": len(rows),
        "expectedRuns": expected_run_rows,
        "runRows": int(len(run_df)),
        "traceRows": int(len(trace_df)),
        "expectedRepeatsPerCondition": int(cfg["globalDefaults"]["repeatCount"]),
        "repeatsToRun": int(repeats_to_run),
        "observedRepeatsByCombination": {
            f"{mode}:{algorithm}:{semantics}:f{frozen_count}": int(count)
            for (mode, algorithm, semantics, frozen_count), count in observed_combo_counts.items()
        },
        "repeatCountOk": repeat_count_ok,
        "observedFrozenSemantics": observed_semantics,
        "observedFrozenCounts": observed_frozen_counts,
        "observedAlgorithms": observed_algorithms,
        "observedModes": observed_modes,
        "exactlyOneFinalTraceRowPerRun": exactly_one_final_trace_row,
        "finalMonotonicityErrorRecomputedFromTraceFinalArrays": final_error_recomputed_ok,
        "frozenCountsPreserved": frozen_counts_ok,
        "stuckFrozenPositionsFixed": stuck_positions_fixed,
        "frozenValidationDetailSample": frozen_details,
        "matchedInitialArraysAndFrozenIndicesValid": matched_initial_valid,
        "matchedInitialDetails": matched_initial_details,
        "noMaxGuardRuns": no_max_guard,
        "figure5CellViewLowerErrorAllComparisons": figure5_cell_lower,
        "cellViewPassiveBubbleBestAllFrozenCounts": passive_bubble_best,
        "cellViewStuckSelectionBestAllFrozenCounts": stuck_selection_best,
        "caveatsOrBlockers": [
            "Traditional Frozen Cell baselines are reconstructed because no public traditional frozen runner or saved original arrays were found.",
            "Cell-view Frozen Cell runs use a deterministic single-threaded scheduler over public cell move methods to make S03 passive/stuck semantics auditable; this removes OS-thread scheduling noise but is not an exact public threaded run.",
            "Passive semantics allow frozen cell identities to be displaced by non-frozen initiators; stuck semantics block any swap involving frozen identities.",
            "S05/S06 comparison-count caveats remain unchanged and are not reinterpreted in S07.",
        ],
        "recommendedNextAction": "Proceed to S08 Delayed Gratification analysis using S07 frozen traces; do not start S08 inside S07.",
    }


def plot_figure(condition_summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    colors = {"bubble": "#4f6fad", "insertion": "#2b8a5e", "selection": "#c25b3d"}
    linestyles = {"traditional": "--", "cell_view": "-"}
    markers = {"traditional": "s", "cell_view": "o"}
    titles = {
        "passive": "Passive Frozen Cells",
        "stuck": "Stuck Frozen Cells",
    }
    for ax, semantics in zip(axes, FROZEN_SEMANTICS):
        subset = condition_summary[condition_summary["frozen_semantics"] == semantics]
        for algorithm in ALGORITHMS:
            for mode in MODES:
                data = subset[(subset["algorithm"] == algorithm) & (subset["mode"] == mode)].sort_values("frozen_count")
                label = f"{algorithm.title()} {'cell-view' if mode == 'cell_view' else 'traditional'}"
                ax.errorbar(
                    data["frozen_count"],
                    data["mean_final_monotonicity_error"],
                    yerr=data["std_final_monotonicity_error"].fillna(0.0),
                    color=colors[algorithm],
                    linestyle=linestyles[mode],
                    marker=markers[mode],
                    linewidth=1.8,
                    capsize=3,
                    label=label,
                )
        ax.set_title(titles[semantics])
        ax.set_xlabel("Frozen Cell count")
        ax.set_xticks(list(FROZEN_COUNTS))
        ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.45)
    axes[0].set_ylabel("Final monotonicity error")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("E01 S07 Figure 5-style Frozen Cell robustness", fontsize=14)
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    stringified = df.copy()
    for col in stringified.columns:
        if pd.api.types.is_float_dtype(stringified[col]):
            stringified[col] = stringified[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    stringified = stringified.astype("string").fillna("").astype(str)
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


def update_checksum_file(checksum_path: Path, paths: list[Path]) -> None:
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    order: list[str] = []
    if checksum_path.exists():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if "  " not in line:
                continue
            digest, path_str = line.split("  ", 1)
            existing[path_str] = digest
            order.append(path_str)
    for path in paths:
        if not path.exists() or path == checksum_path:
            continue
        resolved = str(path)
        existing[resolved] = sha256_file(path)
        if resolved not in order:
            order.append(resolved)
    checksum_path.write_text("".join(f"{existing[path]}  {path}\n" for path in order), encoding="utf-8")


def write_artifact_manifest(manifest_path: Path, artifact_paths: dict[str, Path], validation: dict[str, Any], repo_dir: Path) -> None:
    payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "repositoryCommitAtRunTime": git_commit(repo_dir),
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


def update_run_manifest(
    artifacts_dir: Path,
    paths: dict[str, Path],
    validation: dict[str, Any],
    repo_dir: Path,
    command: list[str],
) -> None:
    manifest_path = artifacts_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest["researchStepId"] = STEP_ID
    manifest["updatedAtUtc"] = utc_now()
    manifest.setdefault("artifacts", {}).update({key: str(path) for key, path in paths.items()})
    manifest.setdefault("checksums", {})
    for path in paths.values():
        if path.exists():
            manifest["checksums"][str(path)] = sha256_file(path)
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": "Reproduce Frozen Cell robustness",
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
        "artifactsWritten": [str(path) for path in paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s07_frozen_robustness.py"),
        "command": " ".join(command),
        "summary": validation["summary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    report_path: Path,
    artifact_paths: dict[str, Path],
    validation: dict[str, Any],
    condition_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    command: list[str],
    repo_dir: Path,
    args: argparse.Namespace,
    wall_seconds: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    caveats = "; ".join(validation["caveatsOrBlockers"])
    if validation["outcomeClassification"] == "supportive":
        lay_summary = (
            "S07 reproduced the paper's Frozen Cell robustness direction under the frozen S03 assumptions: "
            "cell-view final monotonicity errors were lower than reconstructed traditional errors, Bubble was "
            "best among passive cell-view runs, and Selection was best among stuck cell-view runs."
        )
    else:
        lay_summary = (
            "S07 completed the frozen robustness runs and validations, but the reproduced pattern is not a full "
            "supportive match to the paper under the current reconstructed wrappers. The result is still useful "
            "because the frozen counts, stuck/passive semantics, and final-error recomputation are auditable."
        )
    compact_condition = condition_summary[
        [
            "condition_id",
            "mode",
            "algorithm",
            "frozen_semantics",
            "frozen_count",
            "mean_final_monotonicity_error",
            "std_final_monotonicity_error",
            "mean_final_sortedness_percent",
            "max_guard_runs",
            "stop_reason_counts",
        ]
    ].copy()
    report = f"""# E01 S07 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: {validation["status"]}
- Artifacts written: {", ".join(str(path) for path in artifact_paths.values())}
- Validation result: {validation["validationResult"]}
- Outcome classification: {validation["outcomeClassification"]}
- Caveats or blockers: {caveats}
- Lay summary: {lay_summary}
- Recommended next action: {validation["recommendedNextAction"]}

## Frozen Question

Do cell-view algorithms exhibit lower final monotonicity error than traditional algorithms under passive and stuck Frozen Cells, with Bubble strongest for passive and Selection strongest for stuck cell-view conditions?

## Inputs

- S03 config: `{args.config_path}` (`{sha256_file(args.config_path)}`)
- S03 condition matrix: `{args.condition_matrix_path}` (`{sha256_file(args.condition_matrix_path)}`)
- S07 condition rows: `{validation["expectedConditionRows"]}`
- Repeats per condition: `{validation["expectedRepeatsPerCondition"]}`
- Frozen counts: `{validation["observedFrozenCounts"]}`
- Frozen semantics: `{validation["observedFrozenSemantics"]}`
- Paper reference: Figure 5 and Section 4.2 describe passive and stuck Frozen Cells, 100 runs per condition, n=100, and final monotonicity error.

## Detailed Methods

S07 used the S03 frozen condition matrix without changing frozen-count or frozen-semantics assumptions. Traditional Bubble, Insertion, and Selection remain reconstructed controllers because S02 found no public traditional frozen runner. Passive frozen traditional elements cannot initiate actions but can be displaced by non-frozen active moves; stuck frozen elements block any swap that would move them.

For cell-view runs, S07 instantiated the public `BubbleSortCell`, `InsertionSortCell`, `SelectionSortCell`, `StatusProbe`, and `CellGroup` objects, but called each cell's public `move()` method from a deterministic single-threaded scheduler rather than using OS threads. This preserves the public local move code while making passive and stuck wrappers auditable. Passive uses the public base behavior where frozen initiators do not move but frozen targets can be displaced. Stuck monkey-patches each cell instance's `swap()` method to block any target frozen identity; if that wrapper had required changing S03 semantics, S07 would have stopped as blocked.

Each run stopped when sorted, when two no-legal/no-progress sweeps were observed, or when a max guard was hit. The max guard was `{args.max_successful_swaps}` successful swaps or `{args.max_sweeps}` scheduler sweeps. The final monotonicity error is the count of adjacent inversions in the final array. Trace final rows include `final_values_json`, and S07 independently recomputed final monotonicity error from those trace final arrays during validation.

## Commands

```bash
{" ".join(command)}
```

Validation/smoke command:

```bash
python -m py_compile scripts/e01_s07_frozen_robustness.py
python scripts/e01_s07_frozen_robustness.py --repo-dir /workspace/cell-research --artifacts-dir /cache/e01_s07_smoke_artifacts --cache-dir /cache/e01_s07_smoke --max-repeats 1 --workers 2
```

## Dependencies And Parameters

- Python executable: `{sys.executable}`
- Python version: `{sys.version.splitlines()[0]}`
- Platform: `{platform.platform()}`
- Pandas version: `{pd.__version__}`
- NumPy version: `{np.__version__}`
- Matplotlib version: `{matplotlib.__version__}`
- Repository commit at run time: `{git_commit(repo_dir)}`
- Repository status during report generation: `{git_status(repo_dir) or "clean"}`
- Workers: `{args.workers}`
- Max successful swaps: `{args.max_successful_swaps}`
- Max sweeps: `{args.max_sweeps}`
- Stall sweeps: `{args.stall_sweeps}`
- Wall time: `{wall_seconds:.3f}` seconds

No new Python, system, R, Rust, or Node dependencies were installed for S07.

## Results

Condition-level Figure 5-style summary:

{dataframe_to_markdown(compact_condition)}

Traditional versus cell-view mean final monotonicity error:

{dataframe_to_markdown(comparison_summary)}

Primary claim checks:

- Cell-view lower than traditional for every algorithm/semantics/frozen-count comparison: `{validation["figure5CellViewLowerErrorAllComparisons"]}`.
- Cell-view Bubble lowest error for passive Frozen Cells at every frozen count: `{validation["cellViewPassiveBubbleBestAllFrozenCounts"]}`.
- Cell-view Selection lowest error for stuck Frozen Cells at every frozen count: `{validation["cellViewStuckSelectionBestAllFrozenCounts"]}`.

## Validation

- Run rows: `{validation["runRows"]}` of expected `{validation["expectedRuns"]}`.
- Trace rows: `{validation["traceRows"]}`.
- 100 repeats per condition and all mode/algorithm/variant/frozen-count combinations present: `{validation["repeatCountOk"]}`.
- Frozen counts preserved in initial and final states: `{validation["frozenCountsPreserved"]}`.
- Stuck Frozen Cell positions unchanged from S03 indices: `{validation["stuckFrozenPositionsFixed"]}`.
- Matched initial arrays and frozen indices across traditional/cell-view modes: `{validation["matchedInitialArraysAndFrozenIndicesValid"]}`.
- Exactly one final trace row per run: `{validation["exactlyOneFinalTraceRowPerRun"]}`.
- Final monotonicity error recomputed from trace final arrays: `{validation["finalMonotonicityErrorRecomputedFromTraceFinalArrays"]}`.
- No max-guard runs: `{validation["noMaxGuardRuns"]}`.

Matched initial/frozen-index details:

```json
{json.dumps(validation["matchedInitialDetails"], indent=2)}
```

## Artifacts And Provenance

Reusable outputs:

{chr(10).join(f"- `{label}`: `{path}`" for label, path in artifact_paths.items())}

The global run manifest and checksum file were updated after artifact creation. The S07 artifact manifest records paths, sizes, and SHA256 hashes. Temporary per-condition parquet shards were written under `{args.cache_dir}` and are disposable.

## Caveats, Blockers, Failed Assumptions, And Limitations

- Traditional Frozen Cell baselines are reconstructed, not recovered from public original outputs.
- The cell-view scheduler is deterministic single-threaded over public `move()` methods, not the public threaded runner. This removes thread nondeterminism but may change exact trajectory timing and final no-progress states.
- S07 preserves S03 passive/stuck semantics. No wrapper-level change to the S03 frozen definitions was required.
- Figure 5 numeric reproduction is evaluated from generated final monotonicity errors, not pixel-level extraction from the paper image.
- S05/S06 comparison-count ambiguity is unchanged and is not used to reinterpret S07 robustness.

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    s07_dir = artifacts_dir / "research_steps" / STEP_ID
    traces_dir = artifacts_dir / "traces"
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e01"
    tables_dir = artifacts_dir / "tables"
    for path in (s07_dir, traces_dir, results_dir, figures_dir, tables_dir, artifacts_dir / "checksums"):
        path.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir.resolve()
    work_dir = cache_dir / "condition_shards"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.config_path.read_text(encoding="utf-8"))
    s07_rows = load_s07_conditions(args.condition_matrix_path)
    input_validation = validate_condition_inputs(cfg, s07_rows, args.max_repeats)
    if not input_validation["inputValidationPassed"]:
        raise SystemExit("Input validation failed: " + "; ".join(input_validation["inputValidationErrors"]))
    repeats_to_run = int(input_validation["repeatsToRun"])
    workers = max(1, min(int(args.workers), 8, len(s07_rows)))
    print(f"[{utc_now()}] S07 running {len(s07_rows)} conditions x {repeats_to_run} repeats with workers={workers}", flush=True)
    payloads = [
        {
            "cfg": cfg,
            "row": row,
            "repo_dir": str(args.repo_dir),
            "condition_cache_dir": str(work_dir),
            "repeats_to_run": repeats_to_run,
            "max_successful_swaps": args.max_successful_swaps,
            "max_sweeps": args.max_sweeps,
            "stall_sweeps": args.stall_sweeps,
        }
        for row in s07_rows
    ]
    worker_results: list[dict[str, Any]] = []
    if workers == 1:
        for payload in payloads:
            result = run_condition_worker(payload)
            worker_results.append(result)
            print(f"[{utc_now()}] finished {result['condition_id']} trace_rows={result['trace_rows']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_condition_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                result = future.result()
                worker_results.append(result)
                print(f"[{utc_now()}] finished {result['condition_id']} trace_rows={result['trace_rows']}", flush=True)
    worker_results.sort(key=lambda result: result["condition_id"])
    trace_df, run_df = read_condition_outputs(worker_results)

    trace_path = traces_dir / "e01_frozen_cell_trajectories.parquet"
    robustness_path = results_dir / "e01_frozen_cell_robustness.parquet"
    figure_path = figures_dir / "figure5_frozen_robustness.png"
    summary_csv_path = tables_dir / "e01_frozen_cell_robustness_numeric_table.csv"
    comparison_csv_path = tables_dir / "e01_frozen_cell_robustness_comparison_table.csv"
    validation_path = s07_dir / "s07_validation.json"
    report_path = s07_dir / "research_step_full_results.md"
    artifact_manifest_path = s07_dir / "artifact_manifest.json"

    trace_df.to_parquet(trace_path, index=False)
    run_df.to_parquet(robustness_path, index=False)
    condition_summary = build_condition_summary(run_df)
    comparison_summary = build_comparison_summary(condition_summary)
    condition_summary.to_csv(summary_csv_path, index=False)
    comparison_summary.to_csv(comparison_csv_path, index=False)
    plot_figure(condition_summary, figure_path)

    validation = validate_outputs(cfg, s07_rows, run_df, trace_df, repeats_to_run, condition_summary, comparison_summary)
    validation.update(
        {
            "artifactsWritten": [
                str(report_path),
                str(trace_path),
                str(robustness_path),
                str(figure_path),
                str(summary_csv_path),
                str(comparison_csv_path),
                str(validation_path),
                str(artifact_manifest_path),
            ],
            "createdAtUtc": utc_now(),
            "repositoryCommitAtRunTime": git_commit(args.repo_dir),
            "script": str(args.repo_dir / "scripts/e01_s07_frozen_robustness.py"),
            "inputValidation": input_validation,
            "workerCount": workers,
            "threadEnvironment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
            "summary": {
                "conditionSummaryRows": int(len(condition_summary)),
                "comparisonRows": int(len(comparison_summary)),
                "meanFinalErrorOverall": float(run_df["final_monotonicity_error_count"].mean()),
                "maxFinalErrorObserved": int(run_df["final_monotonicity_error_count"].max()),
                "stopReasonCounts": {str(key): int(value) for key, value in Counter(run_df["stop_reason"]).items()},
                "figure5CellViewLowerErrorAllComparisons": validation["figure5CellViewLowerErrorAllComparisons"],
                "cellViewPassiveBubbleBestAllFrozenCounts": validation["cellViewPassiveBubbleBestAllFrozenCounts"],
                "cellViewStuckSelectionBestAllFrozenCounts": validation["cellViewStuckSelectionBestAllFrozenCounts"],
            },
        }
    )
    validation["allArtifactsExist"] = False

    artifact_paths = {
        "fullResultsReport": report_path,
        "trajectoryParquet": trace_path,
        "robustnessParquet": robustness_path,
        "figurePng": figure_path,
        "numericSummaryCsv": summary_csv_path,
        "comparisonCsv": comparison_csv_path,
        "validationJson": validation_path,
        "artifactManifest": artifact_manifest_path,
    }
    wall_seconds = time.monotonic() - started
    validation["wallSeconds"] = wall_seconds
    write_report(
        report_path,
        artifact_paths,
        validation,
        condition_summary,
        comparison_summary,
        sys.argv,
        args.repo_dir,
        args,
        wall_seconds,
    )
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(artifact_manifest_path, artifact_paths, validation, args.repo_dir)
    validation["allArtifactsExist"] = all(path.exists() for path in artifact_paths.values())
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(
        report_path,
        artifact_paths,
        validation,
        condition_summary,
        comparison_summary,
        sys.argv,
        args.repo_dir,
        args,
        wall_seconds,
    )
    write_artifact_manifest(artifact_manifest_path, artifact_paths, validation, args.repo_dir)

    update_run_manifest(artifacts_dir, {f"s07_{key}": value for key, value in artifact_paths.items()}, validation, args.repo_dir, sys.argv)
    checksum_path = artifacts_dir / "checksums" / "sha256sums.txt"
    update_checksum_file(
        checksum_path,
        list(artifact_paths.values())
        + [
            args.config_path,
            args.condition_matrix_path,
            args.research_plan_path,
            args.repo_dir / "scripts/e01_s07_frozen_robustness.py",
            artifacts_dir / "run_manifest.json",
            checksum_path,
        ],
    )
    print(json.dumps({"validation": validation, "summary": validation["summary"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
