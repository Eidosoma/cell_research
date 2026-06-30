#!/usr/bin/env python3
"""Run E01 S12 opposite-direction chimeric Algotype arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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


STEP_ID = "S12"
STEP_NUMBER = 12
EXPERIMENT_ID = "E01"
SOURCE_WRAPPER_STEP_ID = "S09"
SOURCE_AGGREGATION_STEP_ID = "S10"
SOURCE_DUPLICATE_STEP_ID = "S11"
EXPECTED_UNIQUE_MIXES = {
    "opposite_unique_bubble_down_selection_up",
    "opposite_unique_bubble_up_insertion_down",
    "opposite_unique_insertion_up_selection_down",
}
EXPECTED_DUPLICATE_MIXES = {
    "opposite_duplicate_bubble_down_selection_up",
    "opposite_duplicate_bubble_up_insertion_down",
    "opposite_duplicate_insertion_up_selection_down",
}
EXPECTED_MIXES = EXPECTED_UNIQUE_MIXES | EXPECTED_DUPLICATE_MIXES
EXPECTED_VALUE_COUNTS_DUPLICATE = {value: 10 for value in range(1, 11)}
MIX_LABELS = {
    "opposite_unique_bubble_down_selection_up": "Bubble down / Selection up",
    "opposite_unique_bubble_up_insertion_down": "Bubble up / Insertion down",
    "opposite_unique_insertion_up_selection_down": "Insertion up / Selection down",
    "opposite_duplicate_bubble_down_selection_up": "Bubble down / Selection up",
    "opposite_duplicate_bubble_up_insertion_down": "Bubble up / Insertion down",
    "opposite_duplicate_insertion_up_selection_down": "Insertion up / Selection down",
}
MIX_COLORS = {
    "opposite_unique_bubble_down_selection_up": "#4169a8",
    "opposite_unique_bubble_up_insertion_down": "#2b8a5e",
    "opposite_unique_insertion_up_selection_down": "#b34d4d",
    "opposite_duplicate_bubble_down_selection_up": "#4169a8",
    "opposite_duplicate_bubble_up_insertion_down": "#2b8a5e",
    "opposite_duplicate_insertion_up_selection_down": "#b34d4d",
}
PAPER_UNIQUE_FINAL_SORTEDNESS = {
    "opposite_unique_bubble_down_selection_up": 42.5,
    "opposite_unique_bubble_up_insertion_down": 73.73,
    "opposite_unique_insertion_up_selection_down": 38.31,
}
PAPER_EXPECTED_DOMINANT_LABEL = {
    "opposite_unique_bubble_down_selection_up": "bubble",
    "opposite_unique_bubble_up_insertion_down": "bubble",
    "opposite_unique_insertion_up_selection_down": "selection",
}
LABEL_TO_BEHAVIOR = {
    "bubble": "bubble",
    "insertion": "insertion",
    "selection": "selection",
}
LABEL_TO_CODE = {
    "bubble": "B",
    "insertion": "I",
    "selection": "S",
}
CODE_TO_LABEL = {value: key for key, value in LABEL_TO_CODE.items()}
TRACE_COLUMNS = [
    "research_step_id",
    "experiment_id",
    "condition_id",
    "mode",
    "algorithm",
    "baseline_source",
    "matched_group_id",
    "value_bank_id",
    "value_distribution",
    "algotype_mix",
    "algotype_roles",
    "direction_profile",
    "algotype_assignment_bank_id",
    "repeat_index",
    "initial_array_seed",
    "scheduler_seed",
    "algotype_assignment_seed",
    "configured_algotype_counts_json",
    "initial_algotype_counts_json",
    "final_algotype_counts_json",
    "configured_direction_roles_json",
    "initial_direction_counts_json",
    "final_direction_counts_json",
    "swap_step",
    "comparison_count_at_step",
    "sortedness_percent",
    "sortedness_decreasing_percent",
    "monotonicity_error_count",
    "monotonicity_error_decreasing_count",
    "aggregation_left_neighbor_percent",
    "aggregation_right_neighbor_legacy_percent",
    "algotype_positions_code",
    "direction_positions_code",
    "is_initial",
    "is_final",
    "stop_reason",
    "stable_equilibrium_guard_hit",
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
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/e01_s12"))
    parser.add_argument("--max-repeats", type=int, default=None, help="Optional smoke-test cap per condition.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-successful-swaps", type=int, default=15000)
    parser.add_argument("--max-sweeps", type=int, default=20000)
    parser.add_argument("--stall-sweeps", type=int, default=2)
    parser.add_argument("--stable-sweeps", type=int, default=75)
    parser.add_argument("--stable-min-sweeps", type=int, default=100)
    parser.add_argument("--progress-points", type=int, default=101)
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


def load_s09_module(repo_dir: Path) -> Any:
    module_path = repo_dir / "scripts" / "e01_s09_same_goal_chimeras.py"
    spec = importlib.util.spec_from_file_location("e01_s09_same_goal_chimeras", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load S09 wrapper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sortedness_percent(values: list[int]) -> float:
    if not values:
        return 100.0
    ordered_pairs = sum(1 for idx in range(1, len(values)) if values[idx - 1] <= values[idx])
    return 100.0 * (1 + ordered_pairs) / len(values)


def sortedness_decreasing_percent(values: list[int]) -> float:
    if not values:
        return 100.0
    ordered_pairs = sum(1 for idx in range(1, len(values)) if values[idx - 1] >= values[idx])
    return 100.0 * (1 + ordered_pairs) / len(values)


def monotonicity_error_count(values: list[int]) -> int:
    return sum(1 for idx in range(1, len(values)) if values[idx] < values[idx - 1])


def monotonicity_error_decreasing_count(values: list[int]) -> int:
    return sum(1 for idx in range(1, len(values)) if values[idx] > values[idx - 1])


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


def aggregation_left_from_code(code: str) -> float:
    same_left = sum(1 for idx in range(1, len(code)) if code[idx] == code[idx - 1])
    return 100.0 * same_left / len(code)


def aggregation_right_from_code(code: str) -> float:
    same_right = sum(1 for idx in range(len(code) - 1) if code[idx] == code[idx + 1])
    return 100.0 * same_right / len(code)


def expected_left_chance_from_counts(counts_payload: str) -> float:
    counts = json.loads(counts_payload)
    total = sum(int(value) for value in counts.values())
    numerator = sum(int(value) * (int(value) - 1) for value in counts.values())
    return 100.0 * numerator / (total * total)


def label_code(labels: list[str]) -> str:
    return "".join(LABEL_TO_CODE[label] for label in labels)


def labels_from_code(code: str) -> list[str]:
    return [CODE_TO_LABEL[char] for char in code]


def direction_code(directions: list[str]) -> str:
    return "".join("U" if direction == "increasing" else "D" for direction in directions)


def count_json(items: list[Any]) -> str:
    return json.dumps(dict(sorted(Counter(items).items())), separators=(",", ":"))


def value_counts(values: list[int]) -> dict[int, int]:
    return {int(key): int(value) for key, value in sorted(Counter(map(int, values)).items())}


def duplicate_multiplicities_valid(values: list[int]) -> bool:
    return value_counts(values) == EXPECTED_VALUE_COUNTS_DUPLICATE


def unique_multiplicities_valid(values: list[int], array_length: int) -> bool:
    return sorted(map(int, values)) == list(range(1, array_length + 1))


def adjacent_equal_value_metrics(values: list[int], labels: list[str]) -> dict[str, Any]:
    equal_pairs = 0
    same_label_equal_pairs = 0
    for idx in range(1, len(values)):
        if values[idx] == values[idx - 1]:
            equal_pairs += 1
            if labels[idx] == labels[idx - 1]:
                same_label_equal_pairs += 1
    return {
        "adjacent_equal_value_pair_count": int(equal_pairs),
        "adjacent_equal_value_same_label_pair_count": int(same_label_equal_pairs),
        "adjacent_equal_value_same_label_pair_percent": float(100.0 * same_label_equal_pairs / equal_pairs) if equal_pairs else np.nan,
        "adjacent_equal_value_same_label_pair_n100_percent": float(100.0 * same_label_equal_pairs / len(values)) if values else np.nan,
    }


def ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    if n <= 1 or np.isnan(std):
        return (mean, mean)
    half_width = 1.96 * std / np.sqrt(n)
    return (mean - half_width, mean + half_width)


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


def load_s12_conditions(condition_matrix_path: Path) -> list[dict[str, str]]:
    with condition_matrix_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    s12_rows = [row for row in rows if "S12" in row["step_scope"].split(",")]
    return sorted(s12_rows, key=lambda row: row["condition_id"])


def parse_direction_roles(payload: str) -> dict[str, str]:
    roles: dict[str, str] = {}
    for part in payload.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid algotype_roles part: {part!r}")
        label, direction = [piece.strip() for piece in part.split(":", 1)]
        if label not in LABEL_TO_BEHAVIOR:
            raise ValueError(f"Unsupported Algotype label in roles: {label}")
        if direction not in {"increasing", "decreasing"}:
            raise ValueError(f"Unsupported direction in roles: {direction}")
        roles[label] = direction
    return roles


def expected_counts_for_assignment_bank(bank: dict[str, Any], assignment: list[str]) -> dict[str, int]:
    labels = list(map(str, bank["labels"]))
    per_label = len(assignment) // len(labels)
    remainder = len(assignment) % len(labels)
    expected = {label: per_label for label in labels}
    for label in labels[:remainder]:
        expected[label] += 1
    return expected


def scheduler_seed(base_seed: int, condition_id: str, repeat_idx: int) -> int:
    condition_num = int(condition_id.replace("E01C", ""))
    return base_seed * 1_000_000 + 120 * 10_000 + condition_num * 1_000 + repeat_idx


def validate_condition_inputs(
    cfg: dict[str, Any],
    rows: list[dict[str, str]],
    max_repeats: int | None,
) -> dict[str, Any]:
    defaults = cfg["globalDefaults"]
    repeat_count = int(defaults["repeatCount"])
    array_length = int(defaults["arrayLength"])
    repeats_to_run = min(max_repeats or repeat_count, repeat_count)
    errors: list[str] = []
    warnings: list[str] = []
    observed_mixes = {row["algotype_mix"] for row in rows}
    if len(rows) != 6:
        errors.append(f"Expected 6 S12 condition rows; found {len(rows)}.")
    if observed_mixes != EXPECTED_MIXES:
        errors.append(f"S12 mix mismatch: missing={sorted(EXPECTED_MIXES - observed_mixes)}, extra={sorted(observed_mixes - EXPECTED_MIXES)}.")
    value_banks = cfg["seedBanks"]["valueBanks"]
    assignment_banks = cfg["seedBanks"]["algotypeAssignmentBanks"]
    for row in rows:
        roles: dict[str, str]
        try:
            roles = parse_direction_roles(row["algotype_roles"])
        except ValueError as exc:
            errors.append(f"{row['condition_id']} has invalid direction roles: {exc}")
            roles = {}
        if row["mode"] != "cell_view" or row["algorithm"] != "mixed":
            errors.append(f"{row['condition_id']} is not a cell_view mixed condition.")
        if not row["run_family"].startswith("opposite_direction_chimera"):
            errors.append(f"{row['condition_id']} has unexpected run_family={row['run_family']}.")
        if row["direction_profile"] != "mixed_increasing_decreasing":
            errors.append(f"{row['condition_id']} direction_profile={row['direction_profile']} is not mixed_increasing_decreasing.")
        if set(roles.values()) != {"increasing", "decreasing"}:
            errors.append(f"{row['condition_id']} must have one increasing and one decreasing role; found {roles}.")
        if int(row["repeat_count"]) != repeat_count:
            errors.append(f"{row['condition_id']} repeat_count={row['repeat_count']} differs from config repeatCount={repeat_count}.")
        if int(row["frozen_count"]) != 0 or row["frozen_semantics"] != "none":
            errors.append(f"{row['condition_id']} should be unperturbed for S12.")
        value_bank = value_banks.get(row["value_bank_id"])
        assignment_bank = assignment_banks.get(row["algotype_assignment_bank_id"])
        if not value_bank:
            errors.append(f"{row['condition_id']} missing value bank {row['value_bank_id']}.")
            continue
        if not assignment_bank:
            errors.append(f"{row['condition_id']} missing assignment bank {row['algotype_assignment_bank_id']}.")
            continue
        if len(value_bank["initialArrays"]) < repeats_to_run or len(value_bank["seeds"]) < repeats_to_run:
            errors.append(f"{row['condition_id']} value bank has fewer than {repeats_to_run} repeats.")
        if len(assignment_bank["assignments"]) < repeats_to_run or len(assignment_bank["seeds"]) < repeats_to_run:
            errors.append(f"{row['condition_id']} assignment bank has fewer than {repeats_to_run} repeats.")
        if set(map(str, assignment_bank["labels"])) != set(roles):
            errors.append(f"{row['condition_id']} assignment labels {assignment_bank['labels']} do not match role labels {sorted(roles)}.")
        for repeat_idx in range(repeats_to_run):
            values = list(map(int, value_bank["initialArrays"][repeat_idx]))
            assignment = list(map(str, assignment_bank["assignments"][repeat_idx]))
            if len(values) != array_length:
                errors.append(f"{row['condition_id']} repeat {repeat_idx} value length {len(values)} != {array_length}.")
            if row["value_bank_id"] == "unique_1_to_100" and not unique_multiplicities_valid(values, array_length):
                errors.append(f"{row['condition_id']} repeat {repeat_idx} is not unique 1..{array_length}.")
            if row["value_bank_id"] == "duplicate_1_to_10_x10" and not duplicate_multiplicities_valid(values):
                errors.append(f"{row['condition_id']} repeat {repeat_idx} value counts {value_counts(values)} != {EXPECTED_VALUE_COUNTS_DUPLICATE}.")
            if len(assignment) != array_length:
                errors.append(f"{row['condition_id']} repeat {repeat_idx} assignment length mismatch.")
            if any(label not in roles for label in assignment):
                errors.append(f"{row['condition_id']} repeat {repeat_idx} assignment labels are outside roles: {sorted(set(assignment) - set(roles))}.")
            expected_counts = expected_counts_for_assignment_bank(assignment_bank, assignment)
            observed_counts = dict(Counter(assignment))
            if observed_counts != expected_counts:
                errors.append(f"{row['condition_id']} repeat {repeat_idx} assignment counts {observed_counts} != expected {expected_counts}.")
            direction_counts = Counter(roles[label] for label in assignment)
            if dict(direction_counts) != {"increasing": 50, "decreasing": 50}:
                errors.append(f"{row['condition_id']} repeat {repeat_idx} direction counts {dict(direction_counts)} != 50/50.")
    return {
        "expectedConditionRows": 6,
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


def cell_directions(cells: list[Any]) -> list[str]:
    return ["decreasing" if bool(cell.reverse_direction) else "increasing" for cell in cells]


def labels_from_snapshot(snapshot: list[list[Any]]) -> list[str]:
    return [str(item[1]) for item in snapshot]


def cell_state_signature(cells: list[Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            int(cell.threadID),
            int(cell.value),
            str(cell.label),
            bool(cell.reverse_direction),
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
        reverse = bool(cell.reverse_direction)
        if behavior == "bubble":
            left_idx = current_idx - 1
            right_idx = current_idx + 1
            if left_idx >= 0 and cells[left_idx].status == cell_status.ACTIVE:
                if (reverse and cell.value > cells[left_idx].value) or ((not reverse) and cell.value < cells[left_idx].value):
                    return True
            if right_idx < len(cells) and cells[right_idx].status == cell_status.ACTIVE:
                if (reverse and cell.value < cells[right_idx].value) or ((not reverse) and cell.value > cells[right_idx].value):
                    return True
        elif behavior == "insertion":
            left_idx = current_idx - 1
            if left_idx < 0:
                continue
            if cells[left_idx].status != cell_status.ACTIVE or not cell.is_enable_to_move():
                continue
            if (reverse and cell.value > cells[left_idx].value) or ((not reverse) and cell.value < cells[left_idx].value):
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
    directions: list[str],
    swap_step: int,
    comparison_count: int,
) -> dict[str, Any]:
    return {
        "values": list(map(int, values)),
        "labels": list(map(str, labels)),
        "directions": list(map(str, directions)),
        "swap_step": int(swap_step),
        "comparison_count": int(comparison_count),
        "sortedness_percent": sortedness_percent(values),
        "sortedness_decreasing_percent": sortedness_decreasing_percent(values),
        "monotonicity_error_count": monotonicity_error_count(values),
        "monotonicity_error_decreasing_count": monotonicity_error_decreasing_count(values),
        "aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(labels),
        "aggregation_right_neighbor_legacy_percent": aggregation_right_neighbor_legacy_percent(labels),
        "algotype_positions_code": label_code(labels),
        "direction_positions_code": direction_code(directions),
    }


def stable_metric_signature(values: list[int], labels: list[str]) -> tuple[float, float, int, int, float, float]:
    return (
        round(sortedness_percent(values), 9),
        round(sortedness_decreasing_percent(values), 9),
        int(monotonicity_error_count(values)),
        int(monotonicity_error_decreasing_count(values)),
        round(aggregation_left_neighbor_percent(labels), 9),
        round(aggregation_right_neighbor_legacy_percent(labels), 9),
    )


def run_opposite_direction_chimera_sync(
    repo_dir: Path,
    values: list[int],
    assignments: list[str],
    roles: dict[str, str],
    seed: int,
    max_successful_swaps: int,
    max_sweeps: int,
    stall_sweeps: int,
    stable_sweeps: int,
    stable_min_sweeps: int,
) -> dict[str, Any]:
    s09 = load_s09_module(repo_dir)
    modules = s09.import_cell_modules(repo_dir)
    cls_by_behavior = {
        "bubble": modules["BubbleSortCell"],
        "insertion": modules["InsertionSortCell"],
        "selection": modules["SelectionSortCell"],
    }
    cell_status = modules["CellStatus"]
    random.seed(seed)
    lock = threading.RLock()
    status_probe = s09.TracingStatusProbe()
    left_boundary = (0, 1)
    right_boundary = (len(values) - 1, 1)
    cells: list[Any] = []
    for idx, (value, label) in enumerate(zip(values, assignments, strict=True)):
        behavior = LABEL_TO_BEHAVIOR[label]
        cls = cls_by_behavior[behavior]
        reverse_direction = roles[label] == "decreasing"
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
            reverse_direction=reverse_direction,
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
    initial_directions = cell_directions(cells)
    sweeps = 0
    no_progress_sweeps = 0
    stable_counter = 0
    stable_key = stable_metric_signature(initial_values, initial_labels)
    seen_state_signatures = {cell_state_signature(cells): 0}
    stop_reason = "max_sweeps_exceeded"
    stable_equilibrium_guard_hit = False
    max_guard_hit = False
    cycle_first_seen_sweep: int | None = None
    while sweeps < max_sweeps:
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

        if sweeps >= stable_min_sweeps:
            prior_sweep = seen_state_signatures.get(after_signature)
            if prior_sweep is not None:
                cycle_first_seen_sweep = int(prior_sweep)
                stop_reason = "state_cycle_detected"
                stable_equilibrium_guard_hit = True
                break
            seen_state_signatures[after_signature] = sweeps
            current_values = cell_values(cells)
            current_labels = cell_labels(cells)
            current_key = stable_metric_signature(current_values, current_labels)
            if current_key == stable_key:
                stable_counter += 1
            else:
                stable_key = current_key
                stable_counter = 1
            if stable_counter >= stable_sweeps:
                stop_reason = "stable_metric_window"
                stable_equilibrium_guard_hit = True
                break
    else:
        max_guard_hit = True

    final_values = cell_values(cells)
    final_labels = cell_labels(cells)
    final_directions = cell_directions(cells)
    records = [make_metric_record(initial_values, initial_labels, initial_directions, 0, 0)]
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
        labels = labels_from_snapshot(type_snapshot)
        directions = [roles[label] for label in labels]
        records.append(
            make_metric_record(
                list(map(int, values_snapshot)),
                labels,
                directions,
                int(swap_count),
                int(comparison_count),
            )
        )
    if records[-1]["values"] != final_values or records[-1]["labels"] != final_labels:
        records.append(
            make_metric_record(final_values, final_labels, final_directions, status_probe.swap_count, status_probe.compare_and_swap_count)
        )
    else:
        records[-1]["comparison_count"] = int(status_probe.compare_and_swap_count)
    return {
        "records": records,
        "final_values": final_values,
        "initial_labels": initial_labels,
        "final_labels": final_labels,
        "initial_directions": initial_directions,
        "final_directions": final_directions,
        "swap_count": int(status_probe.swap_count),
        "comparison_count": int(status_probe.compare_and_swap_count),
        "compare_plus_swap_count": int(status_probe.swap_count + status_probe.compare_and_swap_count),
        "stop_reason": stop_reason,
        "stable_equilibrium_guard_hit": bool(stable_equilibrium_guard_hit),
        "max_guard_hit": bool(max_guard_hit),
        "sweep_count": int(sweeps),
        "stable_metric_counter_at_stop": int(stable_counter),
        "cycle_first_seen_sweep": cycle_first_seen_sweep,
        "wrapper_name": "deterministic_single_thread_public_cell_methods_s03_opposite_direction_assignments",
    }


def append_trace_rows(
    trace_rows: list[tuple[Any, ...]],
    row: dict[str, str],
    repeat_idx: int,
    initial_seed: int,
    scheduler_seed_value: int,
    assignment_seed: int,
    configured_counts: str,
    roles_json: str,
    initial_values: list[int],
    run_result: dict[str, Any],
) -> None:
    initial_hash = sha256_json(initial_values)
    final_hash = sha256_json(run_result["final_values"])
    final_counts = count_json(run_result["final_labels"])
    initial_counts = count_json(run_result["initial_labels"])
    final_direction_counts = count_json(run_result["final_directions"])
    initial_direction_counts = count_json(run_result["initial_directions"])
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
                row["value_distribution"],
                row["algotype_mix"],
                row["algotype_roles"],
                row["direction_profile"],
                row["algotype_assignment_bank_id"],
                int(repeat_idx),
                int(initial_seed),
                int(scheduler_seed_value),
                int(assignment_seed),
                configured_counts,
                initial_counts,
                final_counts,
                roles_json,
                initial_direction_counts,
                final_direction_counts,
                int(record["swap_step"]),
                int(record["comparison_count"]),
                float(record["sortedness_percent"]),
                float(record["sortedness_decreasing_percent"]),
                int(record["monotonicity_error_count"]),
                int(record["monotonicity_error_decreasing_count"]),
                float(record["aggregation_left_neighbor_percent"]),
                float(record["aggregation_right_neighbor_legacy_percent"]),
                record["algotype_positions_code"],
                record["direction_positions_code"],
                idx == 0,
                is_final,
                run_result["stop_reason"],
                bool(run_result["stable_equilibrium_guard_hit"]),
                bool(run_result["max_guard_hit"]),
                json.dumps(run_result["final_values"], separators=(",", ":")) if is_final else None,
                initial_hash,
                final_hash,
            )
        )


def direction_counts_for_labels(labels: list[str], roles: dict[str, str]) -> dict[str, int]:
    return dict(sorted(Counter(roles[label] for label in labels).items()))


def dominant_label_from_goal_alignment(roles: dict[str, str], final_increasing: float, final_decreasing: float) -> str:
    increasing_labels = sorted(label for label, direction in roles.items() if direction == "increasing")
    decreasing_labels = sorted(label for label, direction in roles.items() if direction == "decreasing")
    if final_increasing > final_decreasing:
        return increasing_labels[0]
    if final_decreasing > final_increasing:
        return decreasing_labels[0]
    return "tie"


def build_run_record(
    row: dict[str, str],
    repeat_idx: int,
    initial_seed: int,
    scheduler_seed_value: int,
    assignment_seed: int,
    assignments: list[str],
    configured_counts: dict[str, int],
    roles: dict[str, str],
    initial_values: list[int],
    run_result: dict[str, Any],
    elapsed_seconds: float,
    stable_sweeps: int,
    stable_min_sweeps: int,
) -> dict[str, Any]:
    final_values = list(map(int, run_result["final_values"]))
    initial_labels = list(map(str, run_result["initial_labels"]))
    final_labels = list(map(str, run_result["final_labels"]))
    initial_counts = dict(sorted(Counter(initial_labels).items()))
    final_counts = dict(sorted(Counter(final_labels).items()))
    records = list(run_result["records"])
    peak_record = max(records, key=lambda record: float(record["aggregation_left_neighbor_percent"]))
    trough_sortedness_record = min(records, key=lambda record: float(record["sortedness_percent"]))
    peak_sortedness_record = max(records, key=lambda record: float(record["sortedness_percent"]))
    final_sortedness_inc = sortedness_percent(final_values)
    final_sortedness_dec = sortedness_decreasing_percent(final_values)
    adjacent_metrics = adjacent_equal_value_metrics(final_values, final_labels)
    array_length = len(initial_values)
    is_duplicate = row["value_bank_id"] == "duplicate_1_to_10_x10"
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "condition_id": row["condition_id"],
        "mode": row["mode"],
        "algorithm": row["algorithm"],
        "algotype_mix": row["algotype_mix"],
        "algotype_roles": row["algotype_roles"],
        "direction_profile": row["direction_profile"],
        "baseline_source": row["baseline_source"],
        "matched_group_id": row["matched_group_id"],
        "value_bank_id": row["value_bank_id"],
        "value_distribution": row["value_distribution"],
        "algotype_assignment_bank_id": row["algotype_assignment_bank_id"],
        "repeat_index": int(repeat_idx),
        "initial_array_seed": int(initial_seed),
        "scheduler_seed": int(scheduler_seed_value),
        "algotype_assignment_seed": int(assignment_seed),
        "component_algorithms_json": json.dumps(sorted(set(assignments)), separators=(",", ":")),
        "configured_direction_roles_json": json.dumps(dict(sorted(roles.items())), separators=(",", ":")),
        "configured_algotype_counts_json": json.dumps(dict(sorted(configured_counts.items())), separators=(",", ":")),
        "initial_algotype_counts_json": json.dumps(initial_counts, separators=(",", ":")),
        "final_algotype_counts_json": json.dumps(final_counts, separators=(",", ":")),
        "initial_direction_counts_json": json.dumps(direction_counts_for_labels(initial_labels, roles), separators=(",", ":")),
        "final_direction_counts_json": json.dumps(direction_counts_for_labels(final_labels, roles), separators=(",", ":")),
        "configured_algotype_counts_match_initial": initial_counts == configured_counts,
        "final_algotype_counts_preserved": final_counts == initial_counts,
        "direction_assignments_verified": direction_counts_for_labels(initial_labels, roles) == {"decreasing": 50, "increasing": 50},
        "final_direction_counts_preserved": direction_counts_for_labels(final_labels, roles) == direction_counts_for_labels(initial_labels, roles),
        "initial_value_counts_json": json.dumps({str(k): v for k, v in value_counts(initial_values).items()}, separators=(",", ":")),
        "final_value_counts_json": json.dumps({str(k): v for k, v in value_counts(final_values).items()}, separators=(",", ":")),
        "initial_value_multiplicities_valid": duplicate_multiplicities_valid(initial_values) if is_duplicate else unique_multiplicities_valid(initial_values, array_length),
        "final_value_multiplicities_valid": duplicate_multiplicities_valid(final_values) if is_duplicate else unique_multiplicities_valid(final_values, array_length),
        "duplicate_value_run": bool(is_duplicate),
        "initial_array_sha256": sha256_json(initial_values),
        "final_array_sha256": sha256_json(final_values),
        "final_values_json": json.dumps(final_values, separators=(",", ":")),
        "swap_only_steps": int(run_result["swap_count"]),
        "comparison_steps_observed": int(run_result["comparison_count"]),
        "compare_plus_swap_steps": int(run_result["compare_plus_swap_count"]),
        "sweep_count": int(run_result["sweep_count"]),
        "final_sortedness_percent": final_sortedness_inc,
        "final_sortedness_decreasing_percent": final_sortedness_dec,
        "final_monotonicity_error_count": monotonicity_error_count(final_values),
        "final_monotonicity_error_decreasing_count": monotonicity_error_decreasing_count(final_values),
        "dominant_label_by_final_goal_alignment": dominant_label_from_goal_alignment(roles, final_sortedness_inc, final_sortedness_dec),
        "initial_sortedness_percent": sortedness_percent(initial_values),
        "initial_sortedness_decreasing_percent": sortedness_decreasing_percent(initial_values),
        "initial_aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(initial_labels),
        "initial_aggregation_right_neighbor_legacy_percent": aggregation_right_neighbor_legacy_percent(initial_labels),
        "final_aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(final_labels),
        "final_aggregation_right_neighbor_legacy_percent": aggregation_right_neighbor_legacy_percent(final_labels),
        "peak_raw_aggregation_left_neighbor_percent": float(peak_record["aggregation_left_neighbor_percent"]),
        "peak_raw_aggregation_left_neighbor_swap_step": int(peak_record["swap_step"]),
        "final_minus_initial_aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(final_labels) - aggregation_left_neighbor_percent(initial_labels),
        "peak_sortedness_percent": float(peak_sortedness_record["sortedness_percent"]),
        "peak_sortedness_swap_step": int(peak_sortedness_record["swap_step"]),
        "trough_sortedness_percent": float(trough_sortedness_record["sortedness_percent"]),
        "trough_sortedness_swap_step": int(trough_sortedness_record["swap_step"]),
        **adjacent_metrics,
        "stop_reason": run_result["stop_reason"],
        "stable_equilibrium_guard_hit": bool(run_result["stable_equilibrium_guard_hit"]),
        "max_guard_hit": bool(run_result["max_guard_hit"]),
        "stable_metric_counter_at_stop": int(run_result["stable_metric_counter_at_stop"]),
        "cycle_first_seen_sweep": run_result["cycle_first_seen_sweep"],
        "stable_sweeps_required": int(stable_sweeps),
        "stable_min_sweeps": int(stable_min_sweeps),
        "elapsed_seconds": float(elapsed_seconds),
        "wrapper_name": run_result["wrapper_name"],
        "comparison_count_source": "public_status_probe_compare_and_swap_count",
        "comparison_count_semantics": "Actionable-comparison proxy inherited from S05/S09; public cell classes increment StatusProbe.compare_and_swap_count when should_move() is true, not for every value read.",
        "opposite_direction_sortedness_semantics": "Primary Sortedness is global nondecreasing order for paper Figure 9/10 comparability; decreasing Sortedness is recorded as goal-alignment sensitivity.",
        "stable_equilibrium_semantics": "Stop when no legal action/no progress recurs, exact sweep-boundary state cycles, or aggregate metrics remain unchanged for the configured stability window; individual swaps may still be possible under the aggregate-metric window.",
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
    stable_sweeps = int(payload["stable_sweeps"])
    stable_min_sweeps = int(payload["stable_min_sweeps"])
    condition_cache_dir.mkdir(parents=True, exist_ok=True)
    value_bank = cfg["seedBanks"]["valueBanks"][row["value_bank_id"]]
    assignment_bank = cfg["seedBanks"]["algotypeAssignmentBanks"][row["algotype_assignment_bank_id"]]
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    roles = parse_direction_roles(row["algotype_roles"])
    roles_json = json.dumps(dict(sorted(roles.items())), separators=(",", ":"))
    trace_rows: list[tuple[Any, ...]] = []
    run_records: list[dict[str, Any]] = []
    for repeat_idx in range(repeats_to_run):
        initial_values = list(map(int, value_bank["initialArrays"][repeat_idx]))
        initial_seed = int(value_bank["seeds"][repeat_idx])
        assignments = list(map(str, assignment_bank["assignments"][repeat_idx]))
        assignment_seed = int(assignment_bank["seeds"][repeat_idx])
        configured_counts = expected_counts_for_assignment_bank(assignment_bank, assignments)
        configured_counts_json = json.dumps(dict(sorted(configured_counts.items())), separators=(",", ":"))
        scheduler_seed_value = scheduler_seed(base_seed, row["condition_id"], repeat_idx)
        started = time.monotonic()
        run_result = run_opposite_direction_chimera_sync(
            repo_dir,
            initial_values,
            assignments,
            roles,
            scheduler_seed_value,
            max_successful_swaps,
            max_sweeps,
            stall_sweeps,
            stable_sweeps,
            stable_min_sweeps,
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
            roles_json,
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
                roles,
                initial_values,
                run_result,
                elapsed_seconds,
                stable_sweeps,
                stable_min_sweeps,
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


def build_run_grid(trace_df: pd.DataFrame, progress_points: int) -> pd.DataFrame:
    grid = np.linspace(0.0, 100.0, progress_points)
    rows: list[dict[str, Any]] = []
    for (condition_id, repeat_index), subset in trace_df.groupby(["condition_id", "repeat_index"], sort=True):
        subset = subset.sort_values(["swap_step", "is_final"]).groupby("swap_step", as_index=False).tail(1)
        final_swap = float(subset["swap_step"].max())
        progress = np.zeros(len(subset)) if final_swap <= 0 else 100.0 * subset["swap_step"].to_numpy(dtype=float) / final_swap
        first = subset.iloc[0]
        chance = expected_left_chance_from_counts(str(first["initial_algotype_counts_json"]))
        interp_left = np.interp(grid, progress, subset["aggregation_left_neighbor_percent"].to_numpy(dtype=float))
        interp_right = np.interp(grid, progress, subset["aggregation_right_neighbor_legacy_percent"].to_numpy(dtype=float))
        interp_sortedness = np.interp(grid, progress, subset["sortedness_percent"].to_numpy(dtype=float))
        interp_sortedness_dec = np.interp(grid, progress, subset["sortedness_decreasing_percent"].to_numpy(dtype=float))
        for pct, left_value, right_value, sortedness_value, sortedness_dec_value in zip(
            grid,
            interp_left,
            interp_right,
            interp_sortedness,
            interp_sortedness_dec,
            strict=True,
        ):
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "source_wrapper_step_id": SOURCE_WRAPPER_STEP_ID,
                    "source_aggregation_step_id": SOURCE_AGGREGATION_STEP_ID,
                    "source_duplicate_step_id": SOURCE_DUPLICATE_STEP_ID,
                    "experiment_id": EXPERIMENT_ID,
                    "condition_id": condition_id,
                    "algotype_mix": first["algotype_mix"],
                    "value_distribution": first["value_distribution"],
                    "algotype_roles": first["algotype_roles"],
                    "repeat_index": int(repeat_index),
                    "progress_percent": float(pct),
                    "aggregation_left_neighbor_percent": float(left_value),
                    "aggregation_right_neighbor_legacy_percent": float(right_value),
                    "sortedness_percent": float(sortedness_value),
                    "sortedness_decreasing_percent": float(sortedness_dec_value),
                    "expected_random_left_neighbor_percent": float(chance),
                    "final_swap_step": int(final_swap),
                }
            )
    return pd.DataFrame(rows)


def summarize_curves(run_grid: pd.DataFrame) -> pd.DataFrame:
    grouped = run_grid.groupby(["condition_id", "algotype_mix", "value_distribution", "algotype_roles", "progress_percent"], as_index=False)
    curves = grouped.agg(
        n_runs=("repeat_index", "nunique"),
        mean_aggregation_left_neighbor_percent=("aggregation_left_neighbor_percent", "mean"),
        std_aggregation_left_neighbor_percent=("aggregation_left_neighbor_percent", "std"),
        mean_aggregation_right_neighbor_legacy_percent=("aggregation_right_neighbor_legacy_percent", "mean"),
        std_aggregation_right_neighbor_legacy_percent=("aggregation_right_neighbor_legacy_percent", "std"),
        mean_sortedness_percent=("sortedness_percent", "mean"),
        std_sortedness_percent=("sortedness_percent", "std"),
        mean_sortedness_decreasing_percent=("sortedness_decreasing_percent", "mean"),
        expected_random_left_neighbor_percent=("expected_random_left_neighbor_percent", "mean"),
        mean_final_swap_step=("final_swap_step", "mean"),
    )
    for prefix in ("left_neighbor", "right_neighbor_legacy"):
        mean_col = f"mean_aggregation_{prefix}_percent"
        std_col = f"std_aggregation_{prefix}_percent"
        sem_col = f"sem_aggregation_{prefix}_percent"
        low_col = f"ci95_low_aggregation_{prefix}_percent"
        high_col = f"ci95_high_aggregation_{prefix}_percent"
        curves[sem_col] = curves[std_col] / np.sqrt(curves["n_runs"])
        cis = [ci95(float(row[mean_col]), float(row[std_col]), int(row["n_runs"])) for _, row in curves.iterrows()]
        curves[low_col] = [value[0] for value in cis]
        curves[high_col] = [value[1] for value in cis]
    curves["sem_sortedness_percent"] = curves["std_sortedness_percent"] / np.sqrt(curves["n_runs"])
    sortedness_cis = [
        ci95(float(row["mean_sortedness_percent"]), float(row["std_sortedness_percent"]), int(row["n_runs"]))
        for _, row in curves.iterrows()
    ]
    curves["ci95_low_sortedness_percent"] = [value[0] for value in sortedness_cis]
    curves["ci95_high_sortedness_percent"] = [value[1] for value in sortedness_cis]
    return curves.sort_values(["condition_id", "progress_percent"]).reset_index(drop=True)


def build_condition_summary(run_df: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition_id, subset in run_df.groupby("condition_id", sort=True):
        first = subset.iloc[0]
        mix = first["algotype_mix"]
        curve_subset = curves[curves["condition_id"] == condition_id].sort_values("progress_percent")
        left_peak = curve_subset.loc[curve_subset["mean_aggregation_left_neighbor_percent"].idxmax()]
        right_peak = curve_subset.loc[curve_subset["mean_aggregation_right_neighbor_legacy_percent"].idxmax()]
        sortedness_peak = curve_subset.loc[curve_subset["mean_sortedness_percent"].idxmax()]
        sortedness_trough = curve_subset.loc[curve_subset["mean_sortedness_percent"].idxmin()]
        dominant_counts = dict(sorted(Counter(subset["dominant_label_by_final_goal_alignment"]).items()))
        dominant_label = max(dominant_counts.items(), key=lambda item: int(item[1]))[0]
        paper_final = PAPER_UNIQUE_FINAL_SORTEDNESS.get(mix, np.nan)
        paper_dom = PAPER_EXPECTED_DOMINANT_LABEL.get(mix)
        rows.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "algotype_mix": mix,
                "display_label": MIX_LABELS.get(mix, mix),
                "value_distribution": first["value_distribution"],
                "matched_group_id": first["matched_group_id"],
                "component_algorithms_json": first["component_algorithms_json"],
                "configured_direction_roles_json": first["configured_direction_roles_json"],
                "repetitions_observed": int(subset["repeat_index"].nunique()),
                "mean_initial_sortedness_percent": float(subset["initial_sortedness_percent"].mean()),
                "mean_final_sortedness_percent": float(subset["final_sortedness_percent"].mean()),
                "std_final_sortedness_percent": float(subset["final_sortedness_percent"].std(ddof=1)),
                "mean_final_sortedness_decreasing_percent": float(subset["final_sortedness_decreasing_percent"].mean()),
                "mean_final_monotonicity_error_count": float(subset["final_monotonicity_error_count"].mean()),
                "mean_final_monotonicity_error_decreasing_count": float(subset["final_monotonicity_error_decreasing_count"].mean()),
                "mean_curve_peak_sortedness_percent": float(sortedness_peak["mean_sortedness_percent"]),
                "mean_curve_peak_sortedness_progress_percent": float(sortedness_peak["progress_percent"]),
                "mean_curve_trough_sortedness_percent": float(sortedness_trough["mean_sortedness_percent"]),
                "mean_curve_trough_sortedness_progress_percent": float(sortedness_trough["progress_percent"]),
                "paper_reported_unique_final_sortedness_percent": float(paper_final) if not np.isnan(paper_final) else np.nan,
                "delta_from_paper_unique_final_sortedness_percent": float(subset["final_sortedness_percent"].mean() - paper_final) if not np.isnan(paper_final) else np.nan,
                "dominant_label_by_goal_alignment_mode": dominant_label,
                "dominant_label_counts_json": json.dumps(dominant_counts, separators=(",", ":")),
                "paper_expected_dominant_label": paper_dom,
                "paper_dominance_order_supported": bool(paper_dom is not None and dominant_label == paper_dom) if paper_dom is not None else np.nan,
                "stop_reason_counts": json.dumps(dict(sorted(Counter(subset["stop_reason"]).items())), separators=(",", ":")),
                "stable_equilibrium_guard_runs": int(subset["stable_equilibrium_guard_hit"].sum()),
                "max_guard_runs": int(subset["max_guard_hit"].sum()),
                "mean_swap_only_steps": float(subset["swap_only_steps"].mean()),
                "std_swap_only_steps": float(subset["swap_only_steps"].std(ddof=1)),
                "mean_comparison_steps_observed": float(subset["comparison_steps_observed"].mean()),
                "mean_compare_plus_swap_steps": float(subset["compare_plus_swap_steps"].mean()),
                "mean_sweep_count": float(subset["sweep_count"].mean()),
                "initial_mean_aggregation_left_neighbor_percent": float(subset["initial_aggregation_left_neighbor_percent"].mean()),
                "final_mean_aggregation_left_neighbor_percent": float(subset["final_aggregation_left_neighbor_percent"].mean()),
                "final_std_aggregation_left_neighbor_percent": float(subset["final_aggregation_left_neighbor_percent"].std(ddof=1)),
                "final_minus_initial_mean_aggregation_left_neighbor_percent": float(subset["final_minus_initial_aggregation_left_neighbor_percent"].mean()),
                "mean_curve_peak_aggregation_left_neighbor_percent": float(left_peak["mean_aggregation_left_neighbor_percent"]),
                "mean_curve_peak_left_neighbor_progress_percent": float(left_peak["progress_percent"]),
                "mean_curve_peak_aggregation_right_neighbor_legacy_percent": float(right_peak["mean_aggregation_right_neighbor_legacy_percent"]),
                "mean_curve_peak_right_neighbor_legacy_progress_percent": float(right_peak["progress_percent"]),
                "expected_random_left_neighbor_percent": float(left_peak["expected_random_left_neighbor_percent"]),
                "mean_adjacent_equal_value_pair_count": float(subset["adjacent_equal_value_pair_count"].mean()),
                "mean_adjacent_equal_value_same_label_pair_percent": float(subset["adjacent_equal_value_same_label_pair_percent"].mean()),
                "all_initial_value_multiplicities_valid": bool(subset["initial_value_multiplicities_valid"].all()),
                "all_final_value_multiplicities_valid": bool(subset["final_value_multiplicities_valid"].all()),
                "all_direction_assignments_verified": bool(subset["direction_assignments_verified"].all()),
                "all_final_direction_counts_preserved": bool(subset["final_direction_counts_preserved"].all()),
            }
        )
    return pd.DataFrame(rows).sort_values("condition_id").reset_index(drop=True)


def validate_trace_and_runs(
    cfg: dict[str, Any],
    rows: list[dict[str, str]],
    run_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    curves: pd.DataFrame,
    condition_summary: pd.DataFrame,
    repeats_to_run: int,
    input_validation: dict[str, Any],
    max_repeats: int | None,
    progress_points: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_run_rows = len(rows) * repeats_to_run
    final_trace = trace_df[trace_df["is_final"]].copy()
    repeat_counts = run_df.groupby("condition_id")["repeat_index"].nunique().to_dict()
    repeat_count_ok = repeat_counts == {row["condition_id"]: repeats_to_run for row in rows}
    final_rows_one_per_run = bool(
        final_trace.groupby(["condition_id", "repeat_index"]).size().eq(1).all()
        and len(final_trace) == len(run_df)
    )
    trace_check = trace_df.copy()
    trace_check["recomputed_left"] = trace_check["algotype_positions_code"].map(aggregation_left_from_code)
    trace_check["recomputed_right"] = trace_check["algotype_positions_code"].map(aggregation_right_from_code)
    left_diff = (trace_check["recomputed_left"] - trace_check["aggregation_left_neighbor_percent"]).abs()
    right_diff = (trace_check["recomputed_right"] - trace_check["aggregation_right_neighbor_legacy_percent"]).abs()
    left_matches = bool((left_diff <= 1e-9).all())
    right_matches = bool((right_diff <= 1e-9).all())
    if not left_matches:
        errors.append(f"Left-neighbor aggregation reconstruction mismatch; max abs diff={float(left_diff.max())}.")
    if not right_matches:
        errors.append(f"Right-neighbor aggregation reconstruction mismatch; max abs diff={float(right_diff.max())}.")
    trace_codes_complete = bool(trace_df["algotype_positions_code"].map(len).eq(int(cfg["globalDefaults"]["arrayLength"])).all())
    direction_codes_complete = bool(trace_df["direction_positions_code"].map(len).eq(int(cfg["globalDefaults"]["arrayLength"])).all())
    invalid_codes = sorted(set("".join(trace_df["algotype_positions_code"].astype(str).unique())) - set(CODE_TO_LABEL))
    invalid_direction_codes = sorted(set("".join(trace_df["direction_positions_code"].astype(str).unique())) - {"U", "D"})
    final_join = run_df.merge(
        final_trace[
            [
                "condition_id",
                "repeat_index",
                "final_array_sha256",
                "final_values_json",
                "sortedness_percent",
                "sortedness_decreasing_percent",
                "monotonicity_error_count",
                "monotonicity_error_decreasing_count",
                "aggregation_left_neighbor_percent",
                "algotype_positions_code",
            ]
        ],
        on=["condition_id", "repeat_index", "final_array_sha256"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_trace"),
    )
    trace_final_matches_run = bool(
        final_join["final_values_json_trace"].notna().all()
        and np.allclose(final_join["sortedness_percent"], final_join["final_sortedness_percent"])
        and np.allclose(final_join["sortedness_decreasing_percent"], final_join["final_sortedness_decreasing_percent"])
        and (final_join["monotonicity_error_count"] == final_join["final_monotonicity_error_count"]).all()
        and (final_join["monotonicity_error_decreasing_count"] == final_join["final_monotonicity_error_decreasing_count"]).all()
        and np.allclose(final_join["aggregation_left_neighbor_percent"], final_join["final_aggregation_left_neighbor_percent"])
    )
    if not trace_final_matches_run:
        errors.append("Final trace rows do not match run records.")
    final_sortedness_recomputed = []
    final_sortedness_dec_recomputed = []
    for _, row in final_trace.iterrows():
        values = list(map(int, json.loads(row["final_values_json"])))
        final_sortedness_recomputed.append(sortedness_percent(values))
        final_sortedness_dec_recomputed.append(sortedness_decreasing_percent(values))
    final_sortedness_recompute_matches = bool(
        np.allclose(final_trace["sortedness_percent"].to_numpy(dtype=float), np.asarray(final_sortedness_recomputed))
        and np.allclose(final_trace["sortedness_decreasing_percent"].to_numpy(dtype=float), np.asarray(final_sortedness_dec_recomputed))
    )
    if not final_sortedness_recompute_matches:
        errors.append("Final increasing/decreasing Sortedness did not recompute from final traces.")
    curve_rows_ok = len(curves) == len(rows) * int(progress_points)
    summary_rows_ok = len(condition_summary) == len(rows)
    all_value_multiplicities_valid = bool(
        run_df["initial_value_multiplicities_valid"].all()
        and run_df["final_value_multiplicities_valid"].all()
    )
    count_fields_complete = bool(
        run_df[["swap_only_steps", "comparison_steps_observed", "compare_plus_swap_steps"]].notna().all(axis=1).all()
        and (run_df["swap_only_steps"] + run_df["comparison_steps_observed"] == run_df["compare_plus_swap_steps"]).all()
    )
    algotype_counts_ok = bool(
        run_df["configured_algotype_counts_match_initial"].all()
        and run_df["final_algotype_counts_preserved"].all()
    )
    direction_assignments_ok = bool(
        run_df["direction_assignments_verified"].all()
        and run_df["final_direction_counts_preserved"].all()
    )
    stopping_reasons_present = bool(run_df["stop_reason"].astype(str).str.len().gt(0).all())
    final_sortedness_present = bool(
        run_df[["final_sortedness_percent", "final_sortedness_decreasing_percent"]].notna().all(axis=1).all()
    )
    full_production = bool(max_repeats is None and repeats_to_run == int(cfg["globalDefaults"]["repeatCount"]))
    if not full_production:
        warnings.append(f"Run used smoke repeat count {repeats_to_run}, not full production repeat count {cfg['globalDefaults']['repeatCount']}.")
    no_max_guard = bool((~run_df["max_guard_hit"]).all())
    if not no_max_guard:
        warnings.append("At least one run stopped on a max guard instead of the no-legal/stable-equilibrium criteria.")
    all_final_aggregation_above_initial = bool((condition_summary["final_minus_initial_mean_aggregation_left_neighbor_percent"] > 0).all())
    unique_summary = condition_summary[condition_summary["value_distribution"] == "unique_1_to_100"].copy()
    paper_dominance_supported = (
        bool(unique_summary["paper_dominance_order_supported"].map(lambda value: bool(value) if pd.notna(value) else False).all())
        if not unique_summary.empty
        else False
    )
    all_unique_initial_near_50 = bool((unique_summary["mean_initial_sortedness_percent"].sub(50.0).abs() <= 5.0).all()) if not unique_summary.empty else False
    success = bool(
        input_validation["inputValidationPassed"]
        and len(run_df) == expected_run_rows
        and repeat_count_ok
        and final_rows_one_per_run
        and left_matches
        and right_matches
        and trace_codes_complete
        and direction_codes_complete
        and not invalid_codes
        and not invalid_direction_codes
        and trace_final_matches_run
        and final_sortedness_recompute_matches
        and curve_rows_ok
        and summary_rows_ok
        and all_value_multiplicities_valid
        and count_fields_complete
        and algotype_counts_ok
        and direction_assignments_ok
        and stopping_reasons_present
        and final_sortedness_present
        and full_production
    )
    outcome = "supportive" if success and all_final_aggregation_above_initial and paper_dominance_supported and no_max_guard else "constraining/contradictory"
    caveats = [
        "S12 uses a deterministic single-thread wrapper around public cell classes to preserve S03 value banks, Algotype assignments, and per-Algotype direction roles.",
        "Primary Sortedness is global nondecreasing Sortedness for Figure 9/10 comparability; decreasing Sortedness is retained as a goal-alignment sensitivity.",
        "The stable-equilibrium guard is operationalized as no-legal/no-progress twice, exact sweep-boundary state-cycle detection, or unchanged aggregate metrics for the configured stability window; this is a reconstructed stopping rule because the paper text does not specify executable thresholds.",
        "Aggregation uses the S10/S03 left-neighbor denominator n=100, with right-neighbor sensitivity retained.",
        "Duplicate-value conflict runs preserve S11 value-multiplicity checks, but final opposite-direction equilibria are not expected to form contiguous nondecreasing equal-value blocks.",
        "Comparison counts inherit the S05/S09 actionable-comparison proxy caveat.",
    ]
    if not all_final_aggregation_above_initial:
        caveats.append("At least one opposite-direction condition did not finish with mean Aggregation above its mean initial Aggregation.")
    if not paper_dominance_supported:
        caveats.append("The unique-value dominant-label ordering inferred from final increasing versus decreasing Sortedness did not fully match Bubble > Selection > Insertion.")
    if not no_max_guard:
        caveats.append("Some runs hit a max guard; their final states are reported as guarded endpoints rather than stable equilibria.")
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed_with_caveats" if success else "completed_validation_failed",
        "validationResult": "passed_with_reconstructed_stable_equilibrium_caveat" if success else "failed",
        "outcomeClassification": outcome,
        "expectedRuns": expected_run_rows,
        "runRows": int(len(run_df)),
        "traceRows": int(len(trace_df)),
        "curveRows": int(len(curves)),
        "summaryRows": int(len(condition_summary)),
        "expectedConditionRows": len(rows),
        "observedConditionRows": int(run_df["condition_id"].nunique()),
        "expectedRepeatsPerCondition": int(cfg["globalDefaults"]["repeatCount"]),
        "repeatsToRun": int(repeats_to_run),
        "fullProductionRepeatCount": full_production,
        "repeatCountsByCondition": {str(key): int(value) for key, value in repeat_counts.items()},
        "repeatCountOk": repeat_count_ok,
        "configuredAlgotypeProportionsVerified": algotype_counts_ok,
        "directionAssignmentsVerified": direction_assignments_ok,
        "finalAlgotypeCountsPreserved": bool(run_df["final_algotype_counts_preserved"].all()),
        "finalDirectionCountsPreserved": bool(run_df["final_direction_counts_preserved"].all()),
        "allValueMultiplicitiesValid": all_value_multiplicities_valid,
        "leftNeighborAggregationRecomputedMatchesTrace": left_matches,
        "rightNeighborAggregationRecomputedMatchesTrace": right_matches,
        "maxLeftAggregationRecomputeAbsDiff": float(left_diff.max()) if len(left_diff) else None,
        "maxRightAggregationRecomputeAbsDiff": float(right_diff.max()) if len(right_diff) else None,
        "invalidPositionCodeCharacters": invalid_codes,
        "invalidDirectionCodeCharacters": invalid_direction_codes,
        "traceAlgotypePositionCodesComplete": trace_codes_complete,
        "traceDirectionPositionCodesComplete": direction_codes_complete,
        "exactlyOneFinalTraceRowPerRun": final_rows_one_per_run,
        "traceFinalMatchesRunRecords": trace_final_matches_run,
        "finalSortednessRecomputedFromTrace": final_sortedness_recompute_matches,
        "stoppingReasonsPresent": stopping_reasons_present,
        "finalSortednessPresent": final_sortedness_present,
        "noMaxGuardRuns": no_max_guard,
        "countFieldsCompleteAndAdditive": count_fields_complete,
        "stopReasonCounts": {str(key): int(value) for key, value in Counter(run_df["stop_reason"]).items()},
        "stableEquilibriumGuardRuns": int(run_df["stable_equilibrium_guard_hit"].sum()),
        "allFinalAggregationAboveInitial": all_final_aggregation_above_initial,
        "uniqueInitialSortednessNear50": all_unique_initial_near_50,
        "uniqueDominanceOrderSupported": paper_dominance_supported,
        "inputValidation": input_validation,
        "validationErrors": errors,
        "validationWarnings": warnings,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Proceed to S13 final replication synthesis using S04-S12 artifacts; do not start S13 inside S12.",
    }


def plot_conflict_figure(curves: pd.DataFrame, condition_summary: pd.DataFrame, value_distribution: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_subset = condition_summary[condition_summary["value_distribution"] == value_distribution].sort_values("condition_id")
    fig, axes = plt.subplots(len(summary_subset), 1, figsize=(9.2, 9.0), sharex=True)
    if len(summary_subset) == 1:
        axes = [axes]
    for ax, (_, summary_row) in zip(axes, summary_subset.iterrows(), strict=True):
        mix = summary_row["algotype_mix"]
        subset = curves[curves["condition_id"] == summary_row["condition_id"]].sort_values("progress_percent")
        color = MIX_COLORS.get(mix, "#555555")
        ax.plot(
            subset["progress_percent"],
            subset["mean_sortedness_percent"],
            color=color,
            linewidth=2.2,
            label="Sortedness",
        )
        ax.fill_between(
            subset["progress_percent"].to_numpy(dtype=float),
            subset["ci95_low_sortedness_percent"].to_numpy(dtype=float),
            subset["ci95_high_sortedness_percent"].to_numpy(dtype=float),
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(
            subset["progress_percent"],
            subset["mean_aggregation_left_neighbor_percent"],
            color="#c23b3b",
            linewidth=1.9,
            label="Aggregation",
        )
        ax.axhline(
            float(summary_row["mean_final_sortedness_percent"]),
            color="#6f6f6f",
            linestyle=":",
            linewidth=1.2,
            label="Final Sortedness mean",
        )
        ax.axhline(50.0, color="#bdbdbd", linestyle="--", linewidth=0.9)
        ax.set_ylim(20, 90)
        ax.set_ylabel("Percent")
        ax.set_title(MIX_LABELS.get(mix, mix), fontsize=10)
        ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.45)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Run progress (% of final swap count)")
    title = "unique values 1..100" if value_distribution == "unique_1_to_100" else "duplicate values 1..10 x10"
    fig.suptitle(f"E01 S12 opposite-direction chimeras: {title}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
        "title": "Reproduce opposite-direction chimeras",
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
        "artifactsWritten": [str(path) for path in paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s12_opposite_direction_chimeras.py"),
        "command": " ".join(command),
        "summary": validation["summary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    report_path: Path,
    artifact_paths: dict[str, Path],
    validation: dict[str, Any],
    condition_summary: pd.DataFrame,
    command: list[str],
    repo_dir: Path,
    args: argparse.Namespace,
    wall_seconds: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    caveats = "; ".join(validation["caveatsOrBlockers"])
    if validation["outcomeClassification"] == "supportive":
        lay_summary = (
            "S12 reproduced the qualitative opposite-direction chimera pattern under the frozen setup: "
            "all six configured conflict conditions ran for 100 matched repeats, direction assignments were "
            "preserved, final Aggregation rose above the initial random baseline, and the unique-value dominance "
            "ordering matched the paper's Bubble > Selection > Insertion description."
        )
    else:
        lay_summary = (
            "S12 completed the opposite-direction chimera runs and validated direction assignments, value "
            "multiplicities, final Sortedness, stopping reasons, and Aggregation reconstruction. Under the frozen "
            "wrapper and reconstructed stable-equilibrium rule, at least one paper-level qualitative check is "
            "constraining rather than fully supportive."
        )
    compact_summary = condition_summary[
        [
            "condition_id",
            "display_label",
            "value_distribution",
            "repetitions_observed",
            "mean_initial_sortedness_percent",
            "mean_final_sortedness_percent",
            "mean_final_sortedness_decreasing_percent",
            "paper_reported_unique_final_sortedness_percent",
            "delta_from_paper_unique_final_sortedness_percent",
            "dominant_label_by_goal_alignment_mode",
            "paper_expected_dominant_label",
            "paper_dominance_order_supported",
            "initial_mean_aggregation_left_neighbor_percent",
            "final_mean_aggregation_left_neighbor_percent",
            "mean_curve_peak_aggregation_left_neighbor_percent",
            "stop_reason_counts",
            "max_guard_runs",
        ]
    ].copy()
    report = f"""# E01 S12 Research Step Full Results

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

Do opposite-goal chimeras reach stable conflict equilibria with dominance-like ordering among algorithms, for both unique-value and repeated-value arrays?

## Inputs

- S03 baseline config: `{args.config_path}` (`{sha256_file(args.config_path)}`)
- S03 condition matrix: `{args.condition_matrix_path}` (`{sha256_file(args.condition_matrix_path)}`)
- S12 condition IDs: E01C051 through E01C056.
- Unique bank: `unique_1_to_100`, 100 arrays, no repeated values.
- Duplicate bank: `duplicate_1_to_10_x10`, 100 arrays, ten copies each of values 1 through 10.
- Assignment banks: `bubble_selection_50_50`, `bubble_insertion_50_50`, and `insertion_selection_50_50`.
- Paper reference: Figure 9 and Figure 10 describe 100-repeat mixed Algotypes with one increasing and one decreasing role; the text reports unique-value final Sortedness values 42.5, 73.73, and 38.31 for the three pairings.

## Detailed Methods

S12 used only the six opposite-direction condition rows frozen in S03. The values and Algotype assignments came directly from S03 seed banks. The direction role was parsed from each condition's `algotype_roles` field. For example, `bubble:decreasing; selection:increasing` made every Bubble cell use `reverse_direction=True` and every Selection cell use `reverse_direction=False`.

The public mixed disorder runner was not executed as-is because S03 explicitly records that it cannot preserve the frozen direction assignment semantics. Instead, S12 used the same deterministic single-thread wrapper pattern introduced in S09 and reused in S11, while still calling the public `move()` methods on `BubbleSortCell`, `InsertionSortCell`, and `SelectionSortCell`. This preserves public cell logic, frozen S03 assignments, and repeatable scheduler seeds, but it does not reproduce OS-thread interleaving.

Stopping criteria followed the S03 `stop_when_no_legal_move_twice_or_stable_equilibrium_guard` rule as a reconstructed executable policy:

- `no_legal_move_twice`: no public legal action and no state progress for `{args.stall_sweeps}` consecutive sweeps.
- `state_cycle_detected`: an exact sweep-boundary state signature repeats after at least `{args.stable_min_sweeps}` sweeps.
- `stable_metric_window`: increasing Sortedness, decreasing Sortedness, increasing and decreasing monotonicity error, and left/right Aggregation are unchanged for `{args.stable_sweeps}` consecutive sweeps after at least `{args.stable_min_sweeps}` sweeps.
- Max guards: `{args.max_successful_swaps}` successful swaps or `{args.max_sweeps}` sweeps. Guarded endpoints are reported explicitly.

Primary Sortedness is global nondecreasing Sortedness, matching the Figure 9/10 paper text's reported values. Decreasing Sortedness is recorded as a sensitivity field for goal alignment. The dominant-label proxy compares final nondecreasing and decreasing Sortedness: if the final state is closer to increasing order, the increasing-role Algotype is counted as dominant; if closer to decreasing order, the decreasing-role Algotype is counted as dominant.

Aggregation uses the S10 definitions unchanged. The primary left-neighbor metric is `100 * count(label[i] == label[i - 1] for i=1..99) / 100`, and the right-neighbor legacy sensitivity uses the analogous right-neighbor count. Duplicate-value handling preserves S11's value-multiplicity validation. Because opposite-direction equilibria are not expected to be fully nondecreasing, S12 reports adjacent equal-value same-label pairs rather than requiring final contiguous equal-value blocks.

## Commands

```bash
{" ".join(command)}
```

Validation/smoke commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/e01_s12_opposite_direction_chimeras.py
PYTHONDONTWRITEBYTECODE=1 python scripts/e01_s12_opposite_direction_chimeras.py --repo-dir /workspace/cell-research --artifacts-dir /cache/e01_s12_smoke_artifacts --cache-dir /cache/e01_s12_smoke --max-repeats 1 --workers 2 --stable-sweeps 10 --stable-min-sweeps 20
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
- Worker count requested: `{args.workers}`
- Max successful swaps guard: `{args.max_successful_swaps}`
- Max sweeps guard: `{args.max_sweeps}`
- Stall sweeps: `{args.stall_sweeps}`
- Stable sweeps: `{args.stable_sweeps}`
- Stable minimum sweeps: `{args.stable_min_sweeps}`
- Progress grid points: `{args.progress_points}`
- Wall time: `{wall_seconds:.3f}` seconds

No new Python, system, R, Rust, or Node dependencies were installed for S12.

## Results

Opposite-direction condition summary:

{dataframe_to_markdown(compact_summary)}

Primary interpretation:

- Production runs completed: `{validation["fullProductionRepeatCount"]}`.
- Direction assignments and 50/50 increasing/decreasing counts verified: `{validation["directionAssignmentsVerified"]}`.
- Value multiplicities preserved for unique and duplicate banks: `{validation["allValueMultiplicitiesValid"]}`.
- Final Sortedness and decreasing Sortedness recomputed from final traces: `{validation["finalSortednessRecomputedFromTrace"]}`.
- Stopping reasons present for all runs: `{validation["stoppingReasonsPresent"]}`.
- No max-guard runs: `{validation["noMaxGuardRuns"]}`.
- All condition-level final mean Aggregation values were above their initial mean Aggregation values: `{validation["allFinalAggregationAboveInitial"]}`.
- Unique-value dominance order matched Bubble > Selection > Insertion under the final goal-alignment proxy: `{validation["uniqueDominanceOrderSupported"]}`.

## Validation

- Run rows: `{validation["runRows"]}` of expected `{validation["expectedRuns"]}`.
- Trace rows: `{validation["traceRows"]}`.
- Curve rows: `{validation["curveRows"]}`.
- Repeat counts by condition: `{validation["repeatCountsByCondition"]}`.
- Exactly one final trace row per run: `{validation["exactlyOneFinalTraceRowPerRun"]}`.
- Trace final metrics match run records: `{validation["traceFinalMatchesRunRecords"]}`.
- Left-neighbor Aggregation recomputed from trace codes: `{validation["leftNeighborAggregationRecomputedMatchesTrace"]}`.
- Right-neighbor sensitivity recomputed from trace codes: `{validation["rightNeighborAggregationRecomputedMatchesTrace"]}`.
- Algotype position codes complete: `{validation["traceAlgotypePositionCodesComplete"]}`.
- Direction position codes complete: `{validation["traceDirectionPositionCodesComplete"]}`.
- Count fields complete and additive: `{validation["countFieldsCompleteAndAdditive"]}`.
- Stop reason counts: `{validation["stopReasonCounts"]}`.
- Stable-equilibrium guard runs: `{validation["stableEquilibriumGuardRuns"]}`.
- Validation warnings: `{validation["validationWarnings"]}`.
- Validation errors: `{validation["validationErrors"]}`.

Detailed validation JSON:

```json
{json.dumps(validation, indent=2)}
```

## Artifacts And Provenance

Reusable outputs:

{chr(10).join(f"- `{label}`: `{path}`" for label, path in artifact_paths.items())}

The global run manifest and checksum file were updated after artifact creation. The S12 artifact manifest records paths, sizes, and SHA256 hashes. Temporary per-condition parquet shards were written under `{args.cache_dir}` and are disposable.

## Caveats, Blockers, Failed Assumptions, And Limitations

- The original repository contains public cell classes with `reverse_direction`, but the public disorder runner self-generates some assignments. S12 therefore uses a deterministic wrapper to preserve S03 assignments and roles.
- The stable-equilibrium criterion is reconstructed. The paper says the conflict trajectories flatten out and reach a stable point, but it does not define a numeric threshold or exact stopping algorithm. S12's aggregate-metric window is auditable but not guaranteed to match the authors' original runtime stop.
- The dominant-label ordering is a proxy based on final global increasing versus decreasing Sortedness, not a direct causal measure of algorithm strength.
- Duplicate-value conflict Sortedness remains ambiguous because ties can satisfy both increasing and decreasing local order. S12 reports both directions and preserves value multiplicity validation without inventing a strict tie-breaker.
- Selection cells retain the public `SelectionSortCell` target-update behavior. With duplicates, equal values can alter which instance occupies a position without changing value-level metrics.
- S12 inherits the S05/S09 comparison-counting caveat: comparison counts are public actionable-comparison events, not a complete read census.

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    s12_dir = artifacts_dir / "research_steps" / STEP_ID
    traces_dir = artifacts_dir / "traces"
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e01"
    tables_dir = artifacts_dir / "tables"
    for path in (s12_dir, traces_dir, results_dir, figures_dir, tables_dir, artifacts_dir / "checksums"):
        path.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir.resolve()
    work_dir = cache_dir / "condition_shards"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.config_path.read_text(encoding="utf-8"))
    rows = load_s12_conditions(args.condition_matrix_path)
    input_validation = validate_condition_inputs(cfg, rows, args.max_repeats)
    if not input_validation["inputValidationPassed"]:
        raise SystemExit("S12 input validation failed: " + json.dumps(input_validation, indent=2))
    repeats_to_run = int(input_validation["repeatsToRun"])
    workers = max(1, min(int(args.workers), 8, len(rows)))
    print(f"[{utc_now()}] S12 running {len(rows)} conditions x {repeats_to_run} repeats with workers={workers}", flush=True)

    payloads = [
        {
            "cfg": cfg,
            "row": row,
            "repo_dir": str(args.repo_dir),
            "condition_cache_dir": str(work_dir / row["condition_id"]),
            "repeats_to_run": repeats_to_run,
            "max_successful_swaps": args.max_successful_swaps,
            "max_sweeps": args.max_sweeps,
            "stall_sweeps": args.stall_sweeps,
            "stable_sweeps": args.stable_sweeps,
            "stable_min_sweeps": args.stable_min_sweeps,
        }
        for row in rows
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
    run_grid = build_run_grid(trace_df, args.progress_points)
    curves = summarize_curves(run_grid)
    condition_summary = build_condition_summary(run_df, curves)

    report_path = s12_dir / "research_step_full_results.md"
    trace_path = traces_dir / "e01_opposite_direction_chimeras.parquet"
    result_path = results_dir / "e01_conflict_equilibria.parquet"
    curves_path = results_dir / "e01_conflict_curves.parquet"
    figure9_path = figures_dir / "figure9_unique_conflict.png"
    figure10_path = figures_dir / "figure10_repeated_conflict.png"
    run_records_path = s12_dir / "e01_conflict_run_records.parquet"
    summary_csv_path = tables_dir / "e01_conflict_equilibria_summary.csv"
    validation_path = s12_dir / "s12_validation.json"
    status_path = s12_dir / "status.json"
    manifest_path = s12_dir / "artifact_manifest.json"

    trace_df.to_parquet(trace_path, index=False)
    run_df.to_parquet(run_records_path, index=False)
    condition_summary.to_parquet(result_path, index=False)
    curves.to_parquet(curves_path, index=False)
    condition_summary.to_csv(summary_csv_path, index=False)
    plot_conflict_figure(curves, condition_summary, "unique_1_to_100", figure9_path)
    plot_conflict_figure(curves, condition_summary, "duplicate_1_to_10_x10", figure10_path)

    validation = validate_trace_and_runs(
        cfg,
        rows,
        run_df,
        trace_df,
        curves,
        condition_summary,
        repeats_to_run,
        input_validation,
        args.max_repeats,
        args.progress_points,
    )
    validation["summary"] = {
        "finalMeanSortednessPercentByMix": {
            str(row["algotype_mix"]): float(row["mean_final_sortedness_percent"])
            for _, row in condition_summary.iterrows()
        },
        "finalMeanAggregationLeftNeighborPercentByMix": {
            str(row["algotype_mix"]): float(row["final_mean_aggregation_left_neighbor_percent"])
            for _, row in condition_summary.iterrows()
        },
        "dominantLabelByMix": {
            str(row["algotype_mix"]): str(row["dominant_label_by_goal_alignment_mode"])
            for _, row in condition_summary.iterrows()
        },
        "stopReasonCounts": validation["stopReasonCounts"],
    }

    artifact_paths = {
        "full_results_report": report_path,
        "opposite_direction_trace_parquet": trace_path,
        "conflict_equilibria_result_parquet": result_path,
        "conflict_curves_parquet": curves_path,
        "figure9_unique_conflict": figure9_path,
        "figure10_repeated_conflict": figure10_path,
        "conflict_run_records": run_records_path,
        "conflict_equilibria_summary_csv": summary_csv_path,
        "validation_json": validation_path,
        "status_json": status_path,
        "artifact_manifest": manifest_path,
    }
    command = sys.argv
    wall_seconds = time.monotonic() - started
    write_report(report_path, artifact_paths, validation, condition_summary, command, args.repo_dir, args, wall_seconds)
    validation["createdAtUtc"] = utc_now()
    validation["wallSeconds"] = wall_seconds
    validation["repositoryCommitAtRunTime"] = git_commit(args.repo_dir)
    validation["script"] = str(args.repo_dir / "scripts/e01_s12_opposite_direction_chimeras.py")
    validation["artifactsWritten"] = [str(path) for path in artifact_paths.values()]
    validation["allArtifactsExist"] = all(path.exists() for path in artifact_paths.values() if path not in {validation_path, status_path, manifest_path})
    validation["workerCount"] = workers
    validation["threadEnvironment"] = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation["success"],
        "status": validation["status"],
        "artifactsWritten": [str(path) for path in artifact_paths.values()],
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
    }
    status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(manifest_path, artifact_paths, validation, args.repo_dir)
    update_run_manifest(artifacts_dir, artifact_paths, validation, args.repo_dir, command)
    update_checksum_file(
        artifacts_dir / "checksums" / "sha256sums.txt",
        [
            args.research_plan_path,
            *artifact_paths.values(),
            artifacts_dir / "run_manifest.json",
        ],
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
