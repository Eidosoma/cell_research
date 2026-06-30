#!/usr/bin/env python3
"""Run E01 S09 same-goal chimeric Algotype arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


STEP_ID = "S09"
STEP_NUMBER = 9
EXPERIMENT_ID = "E01"
ALGORITHMS = ("bubble", "insertion", "selection")
CORE_CHIMERA_MIXES = {
    "same_goal_bubble_insertion",
    "same_goal_bubble_selection",
    "same_goal_insertion_selection",
    "same_goal_bubble_insertion_selection",
}
LABEL_TO_BEHAVIOR = {
    "bubble": "bubble",
    "insertion": "insertion",
    "selection": "selection",
    "bubble_label_a": "bubble",
    "bubble_label_b": "bubble",
}
LABEL_TO_CODE = {
    "bubble": "B",
    "insertion": "I",
    "selection": "S",
    "bubble_label_a": "A",
    "bubble_label_b": "C",
}
TRACE_COLUMNS = [
    "research_step_id",
    "experiment_id",
    "condition_id",
    "mode",
    "algorithm",
    "baseline_source",
    "matched_group_id",
    "value_bank_id",
    "algotype_mix",
    "algotype_assignment_bank_id",
    "repeat_index",
    "initial_array_seed",
    "scheduler_seed",
    "algotype_assignment_seed",
    "configured_algotype_counts_json",
    "initial_algotype_counts_json",
    "final_algotype_counts_json",
    "swap_step",
    "comparison_count_at_step",
    "sortedness_percent",
    "monotonicity_error_count",
    "aggregation_left_neighbor_percent",
    "aggregation_right_neighbor_legacy_percent",
    "algotype_positions_code",
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
    parser.add_argument("--s05-efficiency-path", type=Path, default=Path("/artifacts/results/e01_efficiency_counts.parquet"))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/e01_s09"))
    parser.add_argument("--max-repeats", type=int, default=None, help="Optional smoke-test cap per condition.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-successful-swaps", type=int, default=50000)
    parser.add_argument("--max-sweeps", type=int, default=20000)
    parser.add_argument("--stall-sweeps", type=int, default=2)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_status(repo_dir: Path) -> str:
    try:
        result = subprocess.run(["git", "status", "--short"], cwd=repo_dir, check=True, capture_output=True, text=True)
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


def sortedness_percent(values: list[int]) -> float:
    if not values:
        return 100.0
    ordered_pairs = sum(1 for idx in range(1, len(values)) if values[idx - 1] <= values[idx])
    return 100.0 * (1 + ordered_pairs) / len(values)


def monotonicity_error_count(values: list[int]) -> int:
    return sum(1 for idx in range(1, len(values)) if values[idx] < values[idx - 1])


def nondecreasing(values: list[int]) -> bool:
    return monotonicity_error_count(values) == 0


def aggregation_left_neighbor_percent(labels: list[str]) -> float:
    if not labels:
        return 0.0
    same_left = sum(1 for idx in range(1, len(labels)) if labels[idx] == labels[idx - 1])
    return 100.0 * same_left / len(labels)


def aggregation_right_neighbor_legacy_percent(labels: list[str]) -> float:
    if not labels:
        return 0.0
    same_right = sum(1 for idx in range(len(labels) - 1) if labels[idx] == labels[idx + 1])
    return 100.0 * same_right / len(labels)


def count_json(labels: list[str]) -> str:
    return json.dumps(dict(sorted(Counter(labels).items())), separators=(",", ":"))


def label_code(labels: list[str]) -> str:
    return "".join(LABEL_TO_CODE[label] for label in labels)


class TracingStatusProbe:
    """StatusProbe-compatible recorder with comparison counts at swap snapshots."""

    def __init__(self) -> None:
        self.sorting_steps: list[list[int]] = []
        self.swap_count = 0
        self.cell_types: list[list[list[Any]]] = []
        self.frozen_swap_attempts = 0
        self.compare_and_swap_count = 0
        self.comparison_counts_at_step: list[int] = []
        self.swap_counts_at_step: list[int] = []

    def record_swap(self) -> None:
        self.swap_count += 1

    def record_compare_and_swap(self) -> None:
        self.compare_and_swap_count += 1

    def record_sorting_step(self, snapshot: list[int]) -> None:
        self.sorting_steps.append(list(snapshot))
        self.comparison_counts_at_step.append(int(self.compare_and_swap_count))
        self.swap_counts_at_step.append(int(self.swap_count))

    def record_cell_type(self, snapshot: list[list[Any]]) -> None:
        self.cell_types.append(snapshot)

    def count_frozen_cell_attempt(self) -> None:
        self.frozen_swap_attempts += 1


def import_cell_modules(repo_dir: Path) -> dict[str, Any]:
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from modules.multithread.BubbleSortCell import BubbleSortCell
    from modules.multithread.CellGroup import CellGroup, GroupStatus
    from modules.multithread.InsertionSortCell import InsertionSortCell
    from modules.multithread.MultiThreadCell import CellStatus
    from modules.multithread.SelectionSortCell import SelectionSortCell

    return {
        "BubbleSortCell": BubbleSortCell,
        "CellGroup": CellGroup,
        "CellStatus": CellStatus,
        "GroupStatus": GroupStatus,
        "InsertionSortCell": InsertionSortCell,
        "SelectionSortCell": SelectionSortCell,
    }


def scheduler_seed(base_seed: int, condition_id: str, repeat_idx: int) -> int:
    condition_num = int(condition_id.replace("E01C", ""))
    return base_seed * 1_000_000 + 90 * 10_000 + condition_num * 1_000 + repeat_idx


def load_s09_conditions(condition_matrix_path: Path) -> list[dict[str, str]]:
    with condition_matrix_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    s09_rows = [row for row in rows if "S09" in row["step_scope"].split(",")]
    return sorted(s09_rows, key=lambda row: row["condition_id"])


def expected_counts_for_assignment_bank(bank: dict[str, Any], repeat_idx: int, assignment: list[str]) -> dict[str, int]:
    if bank.get("countsPerRepeat"):
        return {str(key): int(value) for key, value in bank["countsPerRepeat"][repeat_idx].items()}
    labels = list(map(str, bank["labels"]))
    per_label = len(assignment) // len(labels)
    remainder = len(assignment) % len(labels)
    expected = {label: per_label for label in labels}
    if remainder:
        for label in labels[:remainder]:
            expected[label] += 1
    return expected


def validate_condition_inputs(
    cfg: dict[str, Any],
    rows: list[dict[str, str]],
    s05_efficiency: pd.DataFrame,
    max_repeats: int | None,
) -> dict[str, Any]:
    defaults = cfg["globalDefaults"]
    repeat_count = int(defaults["repeatCount"])
    array_length = int(defaults["arrayLength"])
    repeats_to_run = min(max_repeats or repeat_count, repeat_count)
    errors: list[str] = []
    warnings: list[str] = []
    expected_mixes = {
        "same_goal_bubble_insertion",
        "same_goal_bubble_selection",
        "same_goal_insertion_selection",
        "same_goal_bubble_insertion_selection",
        "same_algorithm_bubble_label_control",
    }
    observed_mixes = {row["algotype_mix"] for row in rows}
    if len(rows) != 5:
        errors.append(f"Expected 5 S09 condition rows; found {len(rows)}.")
    if observed_mixes != expected_mixes:
        errors.append(f"S09 algotype mix mismatch: missing={sorted(expected_mixes - observed_mixes)}, extra={sorted(observed_mixes - expected_mixes)}.")
    value_banks = cfg["seedBanks"]["valueBanks"]
    assignment_banks = cfg["seedBanks"]["algotypeAssignmentBanks"]
    for row in rows:
        if row["mode"] != "cell_view" or row["algorithm"] != "mixed":
            errors.append(f"{row['condition_id']} is not a cell_view mixed condition.")
        if int(row["repeat_count"]) != repeat_count:
            errors.append(f"{row['condition_id']} repeat_count={row['repeat_count']} differs from config repeatCount={repeat_count}.")
        if int(row["frozen_count"]) != 0 or row["frozen_semantics"] != "none":
            errors.append(f"{row['condition_id']} should be unperturbed for S09.")
        value_bank = value_banks.get(row["value_bank_id"])
        if not value_bank:
            errors.append(f"{row['condition_id']} missing value bank {row['value_bank_id']}.")
            continue
        assignment_bank = assignment_banks.get(row["algotype_assignment_bank_id"])
        if not assignment_bank:
            errors.append(f"{row['condition_id']} missing assignment bank {row['algotype_assignment_bank_id']}.")
            continue
        if len(value_bank["initialArrays"]) < repeats_to_run or len(value_bank["seeds"]) < repeats_to_run:
            errors.append(f"{row['condition_id']} value bank {row['value_bank_id']} has fewer than {repeats_to_run} repeats.")
        if len(assignment_bank["assignments"]) < repeats_to_run or len(assignment_bank["seeds"]) < repeats_to_run:
            errors.append(f"{row['condition_id']} assignment bank {row['algotype_assignment_bank_id']} has fewer than {repeats_to_run} repeats.")
        for repeat_idx in range(repeats_to_run):
            values = list(map(int, value_bank["initialArrays"][repeat_idx]))
            assignment = list(map(str, assignment_bank["assignments"][repeat_idx]))
            if len(values) != array_length or sorted(values) != list(range(1, array_length + 1)):
                errors.append(f"{row['condition_id']} repeat {repeat_idx} does not contain unique 1..{array_length} values.")
            if len(assignment) != array_length:
                errors.append(f"{row['condition_id']} repeat {repeat_idx} assignment length mismatch.")
            if any(label not in LABEL_TO_BEHAVIOR for label in assignment):
                errors.append(f"{row['condition_id']} repeat {repeat_idx} has unsupported labels: {sorted(set(assignment) - set(LABEL_TO_BEHAVIOR))}.")
            expected_counts = expected_counts_for_assignment_bank(assignment_bank, repeat_idx, assignment)
            observed_counts = dict(Counter(assignment))
            if observed_counts != expected_counts:
                errors.append(f"{row['condition_id']} repeat {repeat_idx} assignment counts {observed_counts} != expected {expected_counts}.")
    required_s05_cols = {
        "mode",
        "algorithm",
        "repeat_index",
        "initial_array_sha256",
        "swap_only_steps",
        "comparison_steps_observed",
        "compare_plus_swap_steps",
        "final_sortedness_percent",
    }
    missing_s05_cols = sorted(required_s05_cols - set(s05_efficiency.columns))
    if missing_s05_cols:
        errors.append(f"S05 efficiency table missing columns: {missing_s05_cols}.")
        pure_cell = pd.DataFrame()
    else:
        pure_cell = s05_efficiency[
            (s05_efficiency["mode"] == "cell_view")
            & (s05_efficiency["algorithm"].isin(ALGORITHMS))
        ].copy()
        observed_repeats = pure_cell.groupby("algorithm")["repeat_index"].nunique().to_dict()
        if observed_repeats != {algorithm: repeat_count for algorithm in ALGORITHMS}:
            errors.append(f"S05 pure cell-view repeat counts mismatch: {observed_repeats}.")
        if not (pure_cell["final_sortedness_percent"] == 100.0).all():
            errors.append("S05 pure cell-view baselines include final Sortedness below 100%.")
        count_fields_ok = (
            pure_cell[["swap_only_steps", "comparison_steps_observed", "compare_plus_swap_steps"]].notna().all(axis=1)
            & (pure_cell["swap_only_steps"] + pure_cell["comparison_steps_observed"] == pure_cell["compare_plus_swap_steps"])
        )
        if not bool(count_fields_ok.all()):
            errors.append("S05 pure cell-view baselines have incomplete or non-additive count fields.")
        value_bank = value_banks.get("unique_1_to_100")
        if value_bank:
            expected_hashes = [sha256_json(list(map(int, value_bank["initialArrays"][idx]))) for idx in range(repeat_count)]
            hash_ok = True
            for algorithm in ALGORITHMS:
                subset = pure_cell[pure_cell["algorithm"] == algorithm].sort_values("repeat_index")
                if subset["initial_array_sha256"].tolist() != expected_hashes:
                    hash_ok = False
            if not hash_ok:
                errors.append("S05 pure cell-view baseline initial array hashes do not match the S03 unique_1_to_100 bank.")
        else:
            warnings.append("S03 unique_1_to_100 bank unavailable while validating S05 pure baselines.")
    return {
        "expectedConditionRows": 5,
        "observedConditionRows": len(rows),
        "expectedRepeatsPerCondition": repeat_count,
        "repeatsToRun": repeats_to_run,
        "inputValidationErrors": errors,
        "inputValidationWarnings": warnings,
        "inputValidationPassed": not errors,
    }


def cell_values(cells: list[Any]) -> list[int]:
    return [int(cell.value) for cell in cells]


def cell_labels(cells: list[Any]) -> list[str]:
    return [str(cell.label) for cell in cells]


def labels_from_snapshot(snapshot: list[list[Any]]) -> list[str]:
    return [str(item[1]) for item in snapshot]


def cell_state_signature(cells: list[Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            int(cell.threadID),
            int(cell.value),
            str(cell.label),
            str(cell.status),
            int(cell.current_position[0]),
            int(cell.ideal_position[0]) if cell.ideal_position is not None else None,
        )
        for cell in cells
    )


def has_public_legal_action(cells: list[Any], cell_status: Any) -> bool:
    for cell in cells:
        if cell.status != cell_status.ACTIVE:
            continue
        behavior = LABEL_TO_BEHAVIOR[str(cell.label)]
        current_idx = int(cell.current_position[0])
        if behavior == "bubble":
            for target_idx, check_right in ((current_idx - 1, False), (current_idx + 1, True)):
                if target_idx < 0 or target_idx >= len(cells):
                    continue
                target = cells[target_idx]
                if target.status != cell_status.ACTIVE:
                    continue
                if check_right and cell.value > target.value:
                    return True
                if not check_right and cell.value < target.value:
                    return True
        elif behavior == "insertion":
            target_idx = current_idx - 1
            if target_idx < 0:
                continue
            target = cells[target_idx]
            if target.status == cell_status.ACTIVE and cell.is_enable_to_move() and cell.value < target.value:
                return True
        elif behavior == "selection":
            if cell.current_position == cell.ideal_position:
                continue
            target_idx = int(cell.ideal_position[0])
            if 0 <= target_idx < len(cells) and cells[target_idx].status == cell_status.ACTIVE:
                return True
        else:
            raise ValueError(f"Unsupported behavior: {behavior}")
    return False


def make_metric_record(
    values: list[int],
    labels: list[str],
    swap_step: int,
    comparison_count: int,
) -> dict[str, Any]:
    return {
        "values": list(map(int, values)),
        "labels": list(map(str, labels)),
        "swap_step": int(swap_step),
        "comparison_count": int(comparison_count),
        "sortedness_percent": sortedness_percent(values),
        "monotonicity_error_count": monotonicity_error_count(values),
        "aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(labels),
        "aggregation_right_neighbor_legacy_percent": aggregation_right_neighbor_legacy_percent(labels),
        "algotype_positions_code": label_code(labels),
    }


def run_same_goal_chimera_sync(
    repo_dir: Path,
    values: list[int],
    assignments: list[str],
    seed: int,
    max_successful_swaps: int,
    max_sweeps: int,
    stall_sweeps: int,
) -> dict[str, Any]:
    modules = import_cell_modules(repo_dir)
    cls_by_behavior = {
        "bubble": modules["BubbleSortCell"],
        "insertion": modules["InsertionSortCell"],
        "selection": modules["SelectionSortCell"],
    }
    cell_status = modules["CellStatus"]
    random.seed(seed)
    lock = threading.RLock()
    status_probe = TracingStatusProbe()
    left_boundary = (0, 1)
    right_boundary = (len(values) - 1, 1)
    cells: list[Any] = []
    for idx, (value, label) in enumerate(zip(values, assignments, strict=True)):
        behavior = LABEL_TO_BEHAVIOR[label]
        cls = cls_by_behavior[behavior]
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
            label=label,
            reverse_direction=False,
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

    initial_values = cell_values(cells)
    initial_labels = cell_labels(cells)
    sweeps = 0
    no_progress_sweeps = 0
    stop_reason = "max_sweeps_exceeded"
    max_guard_hit = False
    while sweeps < max_sweeps:
        current_values = cell_values(cells)
        if nondecreasing(current_values):
            stop_reason = "sorted"
            break
        if status_probe.swap_count >= max_successful_swaps:
            stop_reason = "max_successful_swaps_exceeded"
            max_guard_hit = True
            break
        legal_action_exists = has_public_legal_action(cells, cell_status)
        before_swap_count = status_probe.swap_count
        before_signature = cell_state_signature(cells)
        order = list(range(len(cells)))
        random.shuffle(order)
        for idx in order:
            cell = cells[idx]
            if cell.status == cell_status.ACTIVE:
                cell.move()
        sweeps += 1
        after_signature = cell_state_signature(cells)
        made_progress = status_probe.swap_count != before_swap_count or after_signature != before_signature
        if not made_progress and not legal_action_exists:
            no_progress_sweeps += 1
            if no_progress_sweeps >= stall_sweeps:
                stop_reason = "no_legal_move_twice"
                break
        else:
            no_progress_sweeps = 0
    else:
        max_guard_hit = True

    final_values = cell_values(cells)
    final_labels = cell_labels(cells)
    if nondecreasing(final_values):
        stop_reason = "sorted"
        max_guard_hit = False

    records = [make_metric_record(initial_values, initial_labels, 0, 0)]
    if not (
        len(status_probe.sorting_steps)
        == len(status_probe.cell_types)
        == len(status_probe.comparison_counts_at_step)
        == len(status_probe.swap_counts_at_step)
    ):
        raise RuntimeError("StatusProbe sorting-step, cell-type, and count vectors diverged.")
    for values_snapshot, type_snapshot, comparison_count, swap_count in zip(
        status_probe.sorting_steps,
        status_probe.cell_types,
        status_probe.comparison_counts_at_step,
        status_probe.swap_counts_at_step,
        strict=True,
    ):
        records.append(
            make_metric_record(
                list(map(int, values_snapshot)),
                labels_from_snapshot(type_snapshot),
                int(swap_count),
                int(comparison_count),
            )
        )
    if records[-1]["values"] != final_values or records[-1]["labels"] != final_labels:
        records.append(make_metric_record(final_values, final_labels, status_probe.swap_count, status_probe.compare_and_swap_count))
    else:
        records[-1]["comparison_count"] = int(status_probe.compare_and_swap_count)
    return {
        "records": records,
        "final_values": final_values,
        "initial_labels": initial_labels,
        "final_labels": final_labels,
        "swap_count": int(status_probe.swap_count),
        "comparison_count": int(status_probe.compare_and_swap_count),
        "compare_plus_swap_count": int(status_probe.swap_count + status_probe.compare_and_swap_count),
        "stop_reason": stop_reason,
        "max_guard_hit": bool(max_guard_hit),
        "sweep_count": int(sweeps),
        "wrapper_name": "deterministic_single_thread_public_cell_methods_s03_assignments",
    }


def component_algorithms(assignments: list[str]) -> list[str]:
    return sorted(set(LABEL_TO_BEHAVIOR[label] for label in assignments))


def append_trace_rows(
    trace_rows: list[tuple[Any, ...]],
    row: dict[str, str],
    repeat_idx: int,
    initial_seed: int,
    scheduler_seed_value: int,
    assignment_seed: int,
    configured_counts: str,
    initial_values: list[int],
    run_result: dict[str, Any],
) -> None:
    initial_hash = sha256_json(initial_values)
    final_hash = sha256_json(run_result["final_values"])
    final_counts = count_json(run_result["final_labels"])
    initial_counts = count_json(run_result["initial_labels"])
    records = list(run_result["records"])
    last_idx = len(records) - 1
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
                row["algotype_mix"],
                row["algotype_assignment_bank_id"],
                int(repeat_idx),
                int(initial_seed),
                int(scheduler_seed_value),
                int(assignment_seed),
                configured_counts,
                initial_counts,
                final_counts,
                int(record["swap_step"]),
                int(record["comparison_count"]),
                float(record["sortedness_percent"]),
                int(record["monotonicity_error_count"]),
                float(record["aggregation_left_neighbor_percent"]),
                float(record["aggregation_right_neighbor_legacy_percent"]),
                record["algotype_positions_code"],
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
    scheduler_seed_value: int,
    assignment_seed: int,
    assignments: list[str],
    configured_counts: dict[str, int],
    initial_values: list[int],
    run_result: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    final_values = list(map(int, run_result["final_values"]))
    initial_counts = dict(sorted(Counter(run_result["initial_labels"]).items()))
    final_counts = dict(sorted(Counter(run_result["final_labels"]).items()))
    components = component_algorithms(assignments)
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "condition_id": row["condition_id"],
        "mode": row["mode"],
        "algorithm": row["algorithm"],
        "algotype_mix": row["algotype_mix"],
        "baseline_source": row["baseline_source"],
        "matched_group_id": row["matched_group_id"],
        "value_bank_id": row["value_bank_id"],
        "algotype_assignment_bank_id": row["algotype_assignment_bank_id"],
        "repeat_index": int(repeat_idx),
        "initial_array_seed": int(initial_seed),
        "scheduler_seed": int(scheduler_seed_value),
        "algotype_assignment_seed": int(assignment_seed),
        "component_algorithms_json": json.dumps(components, separators=(",", ":")),
        "configured_algotype_counts_json": json.dumps(dict(sorted(configured_counts.items())), separators=(",", ":")),
        "initial_algotype_counts_json": json.dumps(initial_counts, separators=(",", ":")),
        "final_algotype_counts_json": json.dumps(final_counts, separators=(",", ":")),
        "configured_algotype_counts_match_initial": initial_counts == configured_counts,
        "final_algotype_counts_preserved": final_counts == initial_counts,
        "initial_array_sha256": sha256_json(initial_values),
        "final_array_sha256": sha256_json(final_values),
        "final_values_json": json.dumps(final_values, separators=(",", ":")),
        "swap_only_steps": int(run_result["swap_count"]),
        "comparison_steps_observed": int(run_result["comparison_count"]),
        "compare_plus_swap_steps": int(run_result["compare_plus_swap_count"]),
        "sweep_count": int(run_result["sweep_count"]),
        "final_sortedness_percent": sortedness_percent(final_values),
        "final_monotonicity_error_count": monotonicity_error_count(final_values),
        "final_aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(run_result["final_labels"]),
        "final_aggregation_right_neighbor_legacy_percent": aggregation_right_neighbor_legacy_percent(run_result["final_labels"]),
        "stop_reason": run_result["stop_reason"],
        "max_guard_hit": bool(run_result["max_guard_hit"]),
        "elapsed_seconds": float(elapsed_seconds),
        "wrapper_name": run_result["wrapper_name"],
        "comparison_count_source": "public_status_probe_compare_and_swap_count",
        "comparison_count_semantics": "Actionable-comparison proxy: public cell classes increment StatusProbe.compare_and_swap_count when should_move() is true, not for every value read.",
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
    assignment_bank = cfg["seedBanks"]["algotypeAssignmentBanks"][row["algotype_assignment_bank_id"]]
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    trace_rows: list[tuple[Any, ...]] = []
    run_records: list[dict[str, Any]] = []
    for repeat_idx in range(repeats_to_run):
        initial_values = list(map(int, value_bank["initialArrays"][repeat_idx]))
        initial_seed = int(value_bank["seeds"][repeat_idx])
        assignments = list(map(str, assignment_bank["assignments"][repeat_idx]))
        assignment_seed = int(assignment_bank["seeds"][repeat_idx])
        configured_counts = expected_counts_for_assignment_bank(assignment_bank, repeat_idx, assignments)
        configured_counts_json = json.dumps(dict(sorted(configured_counts.items())), separators=(",", ":"))
        scheduler_seed_value = scheduler_seed(base_seed, row["condition_id"], repeat_idx)
        started = time.monotonic()
        run_result = run_same_goal_chimera_sync(
            repo_dir,
            initial_values,
            assignments,
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
            assignment_seed,
            configured_counts_json,
            initial_values,
            run_result,
        )
        run_records.append(
            build_run_record(
                row,
                repeat_idx,
                initial_seed,
                scheduler_seed_value,
                assignment_seed,
                assignments,
                configured_counts,
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
        "algotype_mix": row["algotype_mix"],
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


def pure_baseline_summary(s05_efficiency: pd.DataFrame) -> pd.DataFrame:
    pure = s05_efficiency[(s05_efficiency["mode"] == "cell_view") & (s05_efficiency["algorithm"].isin(ALGORITHMS))].copy()
    rows: list[dict[str, Any]] = []
    for algorithm, subset in pure.groupby("algorithm", sort=True):
        rows.append(
            {
                "algorithm": algorithm,
                "pure_repetitions": int(subset["repeat_index"].nunique()),
                "pure_mean_swap_only_steps": float(subset["swap_only_steps"].mean()),
                "pure_std_swap_only_steps": float(subset["swap_only_steps"].std(ddof=1)),
                "pure_mean_comparison_steps_observed": float(subset["comparison_steps_observed"].mean()),
                "pure_mean_compare_plus_swap_steps": float(subset["compare_plus_swap_steps"].mean()),
                "pure_final_sortedness_min": float(subset["final_sortedness_percent"].min()),
                "pure_final_sortedness_max": float(subset["final_sortedness_percent"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_condition_summary(run_df: pd.DataFrame, pure_summary: pd.DataFrame) -> pd.DataFrame:
    pure_lookup = pure_summary.set_index("algorithm").to_dict("index")
    rows: list[dict[str, Any]] = []
    for condition_id, subset in run_df.groupby("condition_id", sort=True):
        first = subset.iloc[0]
        components = json.loads(first["component_algorithms_json"])
        component_pure_swap_means = [pure_lookup[algorithm]["pure_mean_swap_only_steps"] for algorithm in components]
        component_pure_compare_plus_means = [pure_lookup[algorithm]["pure_mean_compare_plus_swap_steps"] for algorithm in components]
        mean_swap = float(subset["swap_only_steps"].mean())
        mean_compare_plus = float(subset["compare_plus_swap_steps"].mean())
        is_core = first["algotype_mix"] in CORE_CHIMERA_MIXES
        swap_low = min(component_pure_swap_means)
        swap_high = max(component_pure_swap_means)
        compare_plus_low = min(component_pure_compare_plus_means)
        compare_plus_high = max(component_pure_compare_plus_means)
        rows.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "algotype_mix": first["algotype_mix"],
                "matched_group_id": first["matched_group_id"],
                "component_algorithms_json": first["component_algorithms_json"],
                "is_core_mixed_chimera": bool(is_core),
                "repetitions_observed": int(subset["repeat_index"].nunique()),
                "mean_final_sortedness_percent": float(subset["final_sortedness_percent"].mean()),
                "min_final_sortedness_percent": float(subset["final_sortedness_percent"].min()),
                "max_final_monotonicity_error_count": int(subset["final_monotonicity_error_count"].max()),
                "all_runs_sorted": bool((subset["final_sortedness_percent"] == 100.0).all()),
                "stop_reason_counts": json.dumps(dict(sorted(Counter(subset["stop_reason"]).items())), separators=(",", ":")),
                "max_guard_runs": int(subset["max_guard_hit"].sum()),
                "mean_swap_only_steps": mean_swap,
                "std_swap_only_steps": float(subset["swap_only_steps"].std(ddof=1)),
                "mean_comparison_steps_observed": float(subset["comparison_steps_observed"].mean()),
                "mean_compare_plus_swap_steps": mean_compare_plus,
                "mean_sweep_count": float(subset["sweep_count"].mean()),
                "mean_final_aggregation_left_neighbor_percent": float(subset["final_aggregation_left_neighbor_percent"].mean()),
                "component_pure_mean_swap_only_min": float(swap_low),
                "component_pure_mean_swap_only_max": float(swap_high),
                "component_pure_mean_compare_plus_min": float(compare_plus_low),
                "component_pure_mean_compare_plus_max": float(compare_plus_high),
                "swap_mean_between_component_pure_means": bool(swap_low <= mean_swap <= swap_high),
                "compare_plus_mean_between_component_pure_means": bool(compare_plus_low <= mean_compare_plus <= compare_plus_high),
                "control_delta_from_pure_bubble_swap_mean": float(mean_swap - pure_lookup["bubble"]["pure_mean_swap_only_steps"]) if first["algotype_mix"] == "same_algorithm_bubble_label_control" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(
    cfg: dict[str, Any],
    rows: list[dict[str, str]],
    run_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    repeats_to_run: int,
    input_validation: dict[str, Any],
    condition_summary: pd.DataFrame,
    pure_summary: pd.DataFrame,
) -> dict[str, Any]:
    expected_run_rows = len(rows) * repeats_to_run
    final_trace = trace_df[trace_df["is_final"]].copy()
    final_rows_one_per_run = bool(
        final_trace.groupby(["condition_id", "repeat_index"]).size().eq(1).all()
        and len(final_trace) == len(run_df)
    )
    repeat_counts = run_df.groupby("condition_id")["repeat_index"].nunique().to_dict()
    repeat_count_ok = repeat_counts == {row["condition_id"]: repeats_to_run for row in rows}
    algotype_counts_ok = bool(
        run_df["configured_algotype_counts_match_initial"].all()
        and run_df["final_algotype_counts_preserved"].all()
    )
    final_sortedness_present = bool(run_df["final_sortedness_percent"].notna().all())
    stopping_reasons_present = bool(run_df["stop_reason"].notna().all() and (run_df["stop_reason"].astype(str).str.len() > 0).all())
    all_runs_sorted = bool((run_df["final_sortedness_percent"] == 100.0).all())
    no_max_guard = bool((~run_df["max_guard_hit"]).all())
    count_fields_complete = bool(
        run_df[["swap_only_steps", "comparison_steps_observed", "compare_plus_swap_steps"]].notna().all(axis=1).all()
        and (run_df["swap_only_steps"] + run_df["comparison_steps_observed"] == run_df["compare_plus_swap_steps"]).all()
    )
    trace_codes_complete = bool(trace_df["algotype_positions_code"].map(len).eq(int(cfg["globalDefaults"]["arrayLength"])).all())
    final_join = run_df.merge(
        final_trace[["condition_id", "repeat_index", "final_array_sha256", "final_values_json", "sortedness_percent", "monotonicity_error_count"]],
        on=["condition_id", "repeat_index", "final_array_sha256"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_trace"),
    )
    trace_final_matches_run = bool(
        final_join["final_values_json_trace"].notna().all()
        and (final_join["sortedness_percent"] == final_join["final_sortedness_percent"]).all()
        and (final_join["monotonicity_error_count"] == final_join["final_monotonicity_error_count"]).all()
    )
    pure_baselines_valid = bool(
        len(pure_summary) == 3
        and set(pure_summary["algorithm"]) == set(ALGORITHMS)
        and (pure_summary["pure_repetitions"] == int(cfg["globalDefaults"]["repeatCount"])).all()
        and (pure_summary["pure_final_sortedness_min"] == 100.0).all()
    )
    core_summary = condition_summary[condition_summary["is_core_mixed_chimera"]].copy()
    all_core_swap_between = bool(core_summary["swap_mean_between_component_pure_means"].all())
    all_core_compare_plus_between = bool(core_summary["compare_plus_mean_between_component_pure_means"].all())
    success = bool(
        input_validation["inputValidationPassed"]
        and len(run_df) == expected_run_rows
        and repeat_count_ok
        and final_rows_one_per_run
        and algotype_counts_ok
        and final_sortedness_present
        and stopping_reasons_present
        and all_runs_sorted
        and no_max_guard
        and count_fields_complete
        and trace_codes_complete
        and trace_final_matches_run
        and pure_baselines_valid
        and repeats_to_run == int(cfg["globalDefaults"]["repeatCount"])
    )
    outcome = "supportive" if success and all_core_swap_between else "constraining/contradictory"
    caveats = [
        "Traditional baselines remain reconstructed from S03/S04/S05 because no public traditional runner was found; S09 itself is cell-view only.",
        "S05 pure-efficiency comparison counts are preserved as the public StatusProbe actionable-comparison proxy, not a complete read census.",
        "The public mixed-Algotype runner was not used as-is because it generates its own assignment list and sets Selection cells to reverse direction; S09 directly instantiates public cell classes with S03 same-goal increasing assignments.",
        "Cell-view chimeras use a deterministic single-threaded scheduler over public move methods to make S03 assignment banks and stopping checks auditable, so exact OS-thread timing is not reproduced.",
        "The all-three S03 mix uses rotating 34/33/33 assignments because 100 cells cannot be split equally across three Algotypes.",
    ]
    if not all_core_swap_between:
        caveats.append("At least one core chimera mean swap count did not lie between S05 component pure means under the frozen S05 pure baselines.")
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed_with_caveats" if success else "completed_validation_failed",
        "validationResult": "passed_with_chimera_wrapper_and_counting_caveats" if success else "failed",
        "outcomeClassification": outcome,
        "expectedRuns": expected_run_rows,
        "runRows": int(len(run_df)),
        "traceRows": int(len(trace_df)),
        "expectedConditionRows": len(rows),
        "observedConditionRows": int(run_df["condition_id"].nunique()),
        "expectedRepeatsPerCondition": int(cfg["globalDefaults"]["repeatCount"]),
        "repeatsToRun": int(repeats_to_run),
        "repeatCountsByCondition": {str(key): int(value) for key, value in repeat_counts.items()},
        "repeatCountOk": repeat_count_ok,
        "configuredAlgotypeProportionsVerified": algotype_counts_ok,
        "finalAlgotypeCountsPreserved": bool(run_df["final_algotype_counts_preserved"].all()),
        "finalSortednessPresent": final_sortedness_present,
        "stoppingReasonsPresent": stopping_reasons_present,
        "allFinalSortedness100": all_runs_sorted,
        "noMaxGuardRuns": no_max_guard,
        "countFieldsCompleteAndAdditive": count_fields_complete,
        "exactlyOneFinalTraceRowPerRun": final_rows_one_per_run,
        "traceFinalMatchesRunRecords": trace_final_matches_run,
        "traceAlgotypePositionCodesComplete": trace_codes_complete,
        "s05PureBaselinesValid": pure_baselines_valid,
        "coreMixedSwapMeansBetweenComponentPureMeans": all_core_swap_between,
        "coreMixedComparePlusMeansBetweenComponentPureMeans": all_core_compare_plus_between,
        "stopReasonCounts": {str(key): int(value) for key, value in Counter(run_df["stop_reason"]).items()},
        "inputValidation": input_validation,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Proceed to S10 Aggregation curves using the S09 per-step Algotype position traces; do not start S10 inside S09.",
    }


def interpolate_sortedness(trace_df: pd.DataFrame) -> pd.DataFrame:
    grid = np.linspace(0.0, 100.0, 101)
    rows: list[dict[str, Any]] = []
    for (condition_id, repeat_index), subset in trace_df.groupby(["condition_id", "repeat_index"], sort=True):
        subset = subset.sort_values("swap_step")
        final_swap = float(subset["swap_step"].max())
        if final_swap <= 0:
            progress = np.zeros(len(subset))
        else:
            progress = 100.0 * subset["swap_step"].to_numpy(dtype=float) / final_swap
        sortedness = subset["sortedness_percent"].to_numpy(dtype=float)
        interp = np.interp(grid, progress, sortedness)
        first = subset.iloc[0]
        for pct, value in zip(grid, interp, strict=True):
            rows.append(
                {
                    "condition_id": condition_id,
                    "algotype_mix": first["algotype_mix"],
                    "repeat_index": int(repeat_index),
                    "progress_percent": float(pct),
                    "sortedness_percent": float(value),
                }
            )
    return pd.DataFrame(rows)


def plot_figure(trace_df: pd.DataFrame, condition_summary: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interpolated = interpolate_sortedness(trace_df)
    mean_trace = (
        interpolated.groupby(["condition_id", "algotype_mix", "progress_percent"], as_index=False)["sortedness_percent"]
        .mean()
        .sort_values(["condition_id", "progress_percent"])
    )
    colors = {
        "same_goal_bubble_insertion": "#2b8a5e",
        "same_goal_bubble_selection": "#4169a8",
        "same_goal_insertion_selection": "#b34d4d",
        "same_goal_bubble_insertion_selection": "#6b5aa6",
        "same_algorithm_bubble_label_control": "#c77aa3",
    }
    labels = {
        "same_goal_bubble_insertion": "Bubble-Insertion",
        "same_goal_bubble_selection": "Bubble-Selection",
        "same_goal_insertion_selection": "Insertion-Selection",
        "same_goal_bubble_insertion_selection": "Bubble-Insertion-Selection",
        "same_algorithm_bubble_label_control": "Bubble label control",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), gridspec_kw={"width_ratios": [1.8, 1.0]})
    for algotype_mix, subset in mean_trace.groupby("algotype_mix", sort=False):
        axes[0].plot(
            subset["progress_percent"],
            subset["sortedness_percent"],
            color=colors.get(algotype_mix, "#555555"),
            linewidth=2.0,
            label=labels.get(algotype_mix, algotype_mix),
        )
    axes[0].set_title("Same-goal chimera Sortedness trajectories")
    axes[0].set_xlabel("Run progress (% of final swap count)")
    axes[0].set_ylabel("Sortedness (%)")
    axes[0].set_ylim(45, 101)
    axes[0].grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.45)
    axes[0].legend(frameon=False, fontsize=8)

    bar_df = condition_summary.sort_values("mean_swap_only_steps")
    bar_labels = [labels.get(value, value) for value in bar_df["algotype_mix"]]
    axes[1].barh(bar_labels, bar_df["mean_swap_only_steps"], color=[colors.get(value, "#777777") for value in bar_df["algotype_mix"]])
    axes[1].set_title("Mean swap steps")
    axes[1].set_xlabel("Swap-only steps")
    axes[1].grid(True, axis="x", color="#d0d0d0", linewidth=0.7, alpha=0.45)
    fig.suptitle("E01 S09 Figure 8-style same-goal chimeras", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return mean_trace


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
    widths = [max(len(str(header)), *(len(row[col_idx]) for row in rows)) for col_idx, header in enumerate(headers)]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    divider_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body_lines = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
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
        "title": "Reproduce chimeric same-goal arrays",
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
        "artifactsWritten": [str(path) for path in paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s09_same_goal_chimeras.py"),
        "command": " ".join(command),
        "summary": validation["summary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    report_path: Path,
    artifact_paths: dict[str, Path],
    validation: dict[str, Any],
    condition_summary: pd.DataFrame,
    pure_summary: pd.DataFrame,
    command: list[str],
    repo_dir: Path,
    args: argparse.Namespace,
    wall_seconds: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    caveats = "; ".join(validation["caveatsOrBlockers"])
    if validation["outcomeClassification"] == "supportive":
        lay_summary = (
            "S09 reproduced the same-goal chimera result under the frozen S03/S05 setup: all mixed arrays sorted "
            "to 100% Sortedness and core mixed swap means fell between their component pure cell-view baselines."
        )
    else:
        lay_summary = (
            "S09 completed the same-goal chimera runs and all arrays reached 100% Sortedness, but the efficiency "
            "between-component check is constraining under the frozen S05 pure baselines or validation caveats."
        )
    compact_summary = condition_summary[
        [
            "condition_id",
            "algotype_mix",
            "repetitions_observed",
            "all_runs_sorted",
            "mean_swap_only_steps",
            "component_pure_mean_swap_only_min",
            "component_pure_mean_swap_only_max",
            "swap_mean_between_component_pure_means",
            "mean_compare_plus_swap_steps",
            "compare_plus_mean_between_component_pure_means",
            "stop_reason_counts",
        ]
    ].copy()
    report = f"""# E01 S09 Research Step Full Results

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

Can mixed Algotype arrays with shared increasing-order goals fully sort, and do their efficiencies fall between the efficiencies of their component pure Algotypes?

## Inputs

- S03 config: `{args.config_path}` (`{sha256_file(args.config_path)}`)
- S03 condition matrix: `{args.condition_matrix_path}` (`{sha256_file(args.condition_matrix_path)}`)
- S05 pure efficiency baselines: `{args.s05_efficiency_path}` (`{sha256_file(args.s05_efficiency_path)}`)
- S09 condition rows: `{validation["expectedConditionRows"]}`
- Repeats per condition: `{validation["expectedRepeatsPerCondition"]}`
- Paper reference: Section 4.4 and Figure 8 describe mixed same-goal Algotype arrays, 100 repeats, complete sorting, and swap-count comparison with pure cell-view Algotypes.

## Detailed Methods

S09 used only the S03 frozen same-goal chimera condition rows. Each run used the S03 `unique_1_to_100` value bank, the S03 Algotype assignment bank named by the condition row, and a deterministic scheduler seed derived from the S03 base seed, condition ID, and repeat index. The 50/50 two-way banks were used unchanged. The all-three bank preserved the S03 rotating 34/33/33 remainder rule because 100 cells cannot be split equally three ways. The Bubble label-control bank used two labels, `bubble_label_a` and `bubble_label_b`, while both labels executed Bubble behavior.

The public mixed runner was inspected but not executed as-is because it generates its own assignment list and sets `SelectionSortCell(..., reverse_direction=True)`. That would change the frozen S03 same-goal increasing semantics. Instead, S09 directly instantiated the public `BubbleSortCell`, `InsertionSortCell`, `SelectionSortCell`, and `CellGroup` classes with `reverse_direction=False` for all behaviors and with each cell's S03 Algotype label attached to the moving cell object. The scheduler calls public `move()` methods in shuffled order until the array is sorted, no legal action/no progress is observed twice, or a guard is reached.

The S09 trace records initial, per-swap, and final rows. Each row includes Sortedness, monotonicity error, primary left-neighbor Aggregation, legacy right-neighbor Aggregation, and a compact `algotype_positions_code` string. Code mapping is `B=bubble`, `I=insertion`, `S=selection`, `A=bubble_label_a`, and `C=bubble_label_b`. This keeps per-step Algotype positions available for S10 without writing large JSON label arrays on every row.

Efficiency comparisons use S05 pure cell-view baselines unchanged. Swap-only counts are primary for Figure 8(b)-style efficiency. `compare_plus_swap_steps` are included as a caveated secondary metric because S05 established that cell-view comparison counts are public `StatusProbe.compare_and_swap_count` actionable-comparison events rather than a complete read census.

## Commands

```bash
{" ".join(command)}
```

Validation/smoke commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/e01_s09_same_goal_chimeras.py
PYTHONDONTWRITEBYTECODE=1 python scripts/e01_s09_same_goal_chimeras.py --repo-dir /workspace/cell-research --artifacts-dir /cache/e01_s09_smoke_artifacts --cache-dir /cache/e01_s09_smoke --max-repeats 1 --workers 2
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

No new Python, system, R, Rust, or Node dependencies were installed for S09.

## Results

Condition-level S09 chimera efficiency summary:

{dataframe_to_markdown(compact_summary)}

S05 pure cell-view baseline summary:

{dataframe_to_markdown(pure_summary)}

Primary checks:

- All S09 runs reached final Sortedness 100%: `{validation["allFinalSortedness100"]}`.
- Configured Algotype proportions matched initial labels and were preserved to final states: `{validation["configuredAlgotypeProportionsVerified"]}`.
- Core mixed chimera swap means fell between component pure means: `{validation["coreMixedSwapMeansBetweenComponentPureMeans"]}`.
- Core mixed chimera swap-plus-comparison means fell between component pure means: `{validation["coreMixedComparePlusMeansBetweenComponentPureMeans"]}`. This is secondary and inherits the S05 comparison-count caveat.

## Validation

- Run rows: `{validation["runRows"]}` of expected `{validation["expectedRuns"]}`.
- Trace rows: `{validation["traceRows"]}`.
- Repeat counts by condition: `{validation["repeatCountsByCondition"]}`.
- Exactly one final trace row per run: `{validation["exactlyOneFinalTraceRowPerRun"]}`.
- Trace final metrics match run records: `{validation["traceFinalMatchesRunRecords"]}`.
- Per-step Algotype position codes have length 100: `{validation["traceAlgotypePositionCodesComplete"]}`.
- Final Sortedness present for every run: `{validation["finalSortednessPresent"]}`.
- Stopping reason present for every run: `{validation["stoppingReasonsPresent"]}`.
- Stop reason counts: `{validation["stopReasonCounts"]}`.
- No max-guard runs: `{validation["noMaxGuardRuns"]}`.
- Count fields complete and additive: `{validation["countFieldsCompleteAndAdditive"]}`.
- S05 pure baselines valid: `{validation["s05PureBaselinesValid"]}`.

Input validation:

```json
{json.dumps(validation["inputValidation"], indent=2)}
```

## Artifacts And Provenance

Reusable outputs:

{chr(10).join(f"- `{label}`: `{path}`" for label, path in artifact_paths.items())}

The global run manifest and checksum file were updated after artifact creation. The S09 artifact manifest records paths, sizes, and SHA256 hashes. Temporary per-condition parquet shards were written under `{args.cache_dir}` and are disposable.

## Caveats, Blockers, Failed Assumptions, And Limitations

- Traditional baselines remain reconstructed and are not rerun in S09; pure-comparison inputs are the S05 cell-view baseline table.
- S05 comparison-count ambiguity is preserved unchanged. S09 reports swap-only and swap-plus-comparison counts separately.
- The public mixed runner could not preserve the frozen S03 same-goal assumptions without wrapper-level reconstruction, because it uses self-generated assignment counts and reverse-direction Selection cells.
- Deterministic single-thread scheduling preserves public local `move()` code and S03 assignments, but not the exact public OS-thread interleaving.
- The S09 figure is Figure 8-style and uses normalized run progress for mean Sortedness trajectories; exact paper pixel-level reproduction was not attempted.

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    s09_dir = artifacts_dir / "research_steps" / STEP_ID
    traces_dir = artifacts_dir / "traces"
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e01"
    tables_dir = artifacts_dir / "tables"
    for path in (s09_dir, traces_dir, results_dir, figures_dir, tables_dir, artifacts_dir / "checksums"):
        path.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir.resolve()
    work_dir = cache_dir / "condition_shards"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.config_path.read_text(encoding="utf-8"))
    s09_rows = load_s09_conditions(args.condition_matrix_path)
    s05_efficiency = pd.read_parquet(args.s05_efficiency_path)
    input_validation = validate_condition_inputs(cfg, s09_rows, s05_efficiency, args.max_repeats)
    if not input_validation["inputValidationPassed"]:
        raise SystemExit("Input validation failed: " + "; ".join(input_validation["inputValidationErrors"]))
    repeats_to_run = int(input_validation["repeatsToRun"])
    workers = max(1, min(int(args.workers), 8, len(s09_rows)))
    print(f"[{utc_now()}] S09 running {len(s09_rows)} conditions x {repeats_to_run} repeats with workers={workers}", flush=True)

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
        for row in s09_rows
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

    trace_path = traces_dir / "e01_same_goal_chimeras.parquet"
    efficiency_path = results_dir / "e01_chimera_efficiency.parquet"
    figure_path = figures_dir / "figure8_chimera_sortedness.png"
    summary_csv_path = tables_dir / "e01_chimera_efficiency_numeric_table.csv"
    pure_csv_path = tables_dir / "e01_chimera_pure_baseline_summary.csv"
    mean_trace_csv_path = tables_dir / "e01_chimera_sortedness_mean_trace.csv"
    validation_path = s09_dir / "s09_validation.json"
    status_path = s09_dir / "status.json"
    report_path = s09_dir / "research_step_full_results.md"
    artifact_manifest_path = s09_dir / "artifact_manifest.json"

    trace_df.to_parquet(trace_path, index=False)
    run_df.to_parquet(efficiency_path, index=False)
    pure_summary = pure_baseline_summary(s05_efficiency)
    condition_summary = build_condition_summary(run_df, pure_summary)
    condition_summary.to_csv(summary_csv_path, index=False)
    pure_summary.to_csv(pure_csv_path, index=False)
    mean_trace = plot_figure(trace_df, condition_summary, figure_path)
    mean_trace.to_csv(mean_trace_csv_path, index=False)

    validation = validate_outputs(
        cfg,
        s09_rows,
        run_df,
        trace_df,
        repeats_to_run,
        input_validation,
        condition_summary,
        pure_summary,
    )
    validation.update(
        {
            "createdAtUtc": utc_now(),
            "repositoryCommitAtRunTime": git_commit(args.repo_dir),
            "script": str(args.repo_dir / "scripts/e01_s09_same_goal_chimeras.py"),
            "workerCount": workers,
            "threadEnvironment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
            "summary": {
                "conditionSummaryRows": int(len(condition_summary)),
                "meanSwapOnlyByMix": {
                    str(row["algotype_mix"]): float(row["mean_swap_only_steps"])
                    for _, row in condition_summary.iterrows()
                },
                "allFinalSortedness100": validation["allFinalSortedness100"],
                "coreMixedSwapMeansBetweenComponentPureMeans": validation["coreMixedSwapMeansBetweenComponentPureMeans"],
                "stopReasonCounts": validation["stopReasonCounts"],
            },
            "artifactsWritten": [
                str(report_path),
                str(trace_path),
                str(efficiency_path),
                str(figure_path),
                str(summary_csv_path),
                str(pure_csv_path),
                str(mean_trace_csv_path),
                str(validation_path),
                str(status_path),
                str(artifact_manifest_path),
            ],
        }
    )
    artifact_paths = {
        "fullResultsReport": report_path,
        "sameGoalChimeraTraceParquet": trace_path,
        "chimeraEfficiencyParquet": efficiency_path,
        "figurePng": figure_path,
        "numericSummaryCsv": summary_csv_path,
        "pureBaselineSummaryCsv": pure_csv_path,
        "meanTraceCsv": mean_trace_csv_path,
        "validationJson": validation_path,
        "statusJson": status_path,
        "artifactManifest": artifact_manifest_path,
    }
    validation["allArtifactsExist"] = False
    wall_seconds = time.monotonic() - started
    validation["wallSeconds"] = wall_seconds
    write_report(report_path, artifact_paths, validation, condition_summary, pure_summary, sys.argv, args.repo_dir, args, wall_seconds)
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation["success"],
        "status": validation["status"],
        "artifactsWritten": validation["artifactsWritten"],
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
    }
    status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(artifact_manifest_path, artifact_paths, validation, args.repo_dir)
    validation["allArtifactsExist"] = all(path.exists() for path in artifact_paths.values())
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, artifact_paths, validation, condition_summary, pure_summary, sys.argv, args.repo_dir, args, wall_seconds)
    write_artifact_manifest(artifact_manifest_path, artifact_paths, validation, args.repo_dir)

    update_run_manifest(artifacts_dir, {f"s09_{key}": value for key, value in artifact_paths.items()}, validation, args.repo_dir, sys.argv)
    update_checksum_file(
        artifacts_dir / "checksums" / "sha256sums.txt",
        list(artifact_paths.values())
        + [
            args.config_path,
            args.condition_matrix_path,
            args.s05_efficiency_path,
            args.research_plan_path,
            args.repo_dir / "scripts/e01_s09_same_goal_chimeras.py",
            artifacts_dir / "run_manifest.json",
        ],
    )
    print(json.dumps({"validation": validation, "summary": validation["summary"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
