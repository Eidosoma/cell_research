#!/usr/bin/env python3
"""Run E01 S11 duplicate-value chimeras and aggregation analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
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


STEP_ID = "S11"
STEP_NUMBER = 11
EXPERIMENT_ID = "E01"
SOURCE_WRAPPER_STEP_ID = "S09"
SOURCE_AGGREGATION_STEP_ID = "S10"
EXPECTED_VALUE_COUNTS = {value: 10 for value in range(1, 11)}
EXPECTED_MIXES = {
    "duplicate_bubble_insertion",
    "duplicate_bubble_selection",
    "duplicate_insertion_selection",
}
MIX_LABELS = {
    "duplicate_bubble_insertion": "Bubble-Insertion",
    "duplicate_bubble_selection": "Bubble-Selection",
    "duplicate_insertion_selection": "Insertion-Selection",
}
MIX_TO_S10_UNIQUE = {
    "duplicate_bubble_insertion": "same_goal_bubble_insertion",
    "duplicate_bubble_selection": "same_goal_bubble_selection",
    "duplicate_insertion_selection": "same_goal_insertion_selection",
}
MIX_COLORS = {
    "duplicate_bubble_insertion": "#2b8a5e",
    "duplicate_bubble_selection": "#4169a8",
    "duplicate_insertion_selection": "#b34d4d",
}
CODE_TO_LABEL = {
    "B": "bubble",
    "I": "insertion",
    "S": "selection",
    "A": "bubble_label_a",
    "C": "bubble_label_b",
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
    parser.add_argument("--s10-peak-table-path", type=Path, default=Path("/artifacts/tables/e01_aggregation_peak_table.csv"))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/e01_s11"))
    parser.add_argument("--max-repeats", type=int, default=None, help="Optional smoke-test cap per condition.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-successful-swaps", type=int, default=50000)
    parser.add_argument("--max-sweeps", type=int, default=20000)
    parser.add_argument("--stall-sweeps", type=int, default=2)
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


def monotonicity_error_count(values: list[int]) -> int:
    return sum(1 for idx in range(1, len(values)) if values[idx] < values[idx - 1])


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


def count_json(items: list[Any]) -> str:
    return json.dumps(dict(sorted(Counter(items).items())), separators=(",", ":"))


def labels_from_code(code: str) -> list[str]:
    return [CODE_TO_LABEL[char] for char in code]


def value_counts(values: list[int]) -> dict[int, int]:
    return {int(key): int(value) for key, value in sorted(Counter(map(int, values)).items())}


def duplicate_multiplicities_valid(values: list[int]) -> bool:
    return value_counts(values) == EXPECTED_VALUE_COUNTS


def equal_value_block_metrics(values: list[int], labels: list[str]) -> dict[str, Any]:
    if len(values) != len(labels):
        raise ValueError("values and labels must have equal length")
    same_pairs = 0
    equal_pairs = 0
    block_lengths: dict[int, list[int]] = {}
    block_label_counts: dict[int, list[dict[str, int]]] = {}
    start = 0
    for idx in range(1, len(values) + 1):
        if idx == len(values) or values[idx] != values[start]:
            value = int(values[start])
            block_labels = labels[start:idx]
            block_lengths.setdefault(value, []).append(idx - start)
            block_label_counts.setdefault(value, []).append(dict(sorted(Counter(block_labels).items())))
            start = idx
    for idx in range(1, len(values)):
        if values[idx] == values[idx - 1]:
            equal_pairs += 1
            if labels[idx] == labels[idx - 1]:
                same_pairs += 1
    expected_block_lengths = {value: [10] for value in range(1, 11)}
    blocks_valid = block_lengths == expected_block_lengths
    return {
        "within_equal_value_pair_count": int(equal_pairs),
        "within_equal_value_same_label_pair_count": int(same_pairs),
        "within_equal_value_same_label_pair_percent": float(100.0 * same_pairs / equal_pairs) if equal_pairs else np.nan,
        "within_equal_value_same_label_pair_n100_percent": float(100.0 * same_pairs / len(values)) if values else np.nan,
        "equal_value_blocks_valid": bool(blocks_valid),
        "equal_value_block_lengths_json": json.dumps({str(k): v for k, v in sorted(block_lengths.items())}, separators=(",", ":")),
        "equal_value_block_label_counts_json": json.dumps({str(k): v for k, v in sorted(block_label_counts.items())}, separators=(",", ":")),
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


def load_s11_conditions(condition_matrix_path: Path) -> list[dict[str, str]]:
    with condition_matrix_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    s11_rows = [row for row in rows if "S11" in row["step_scope"].split(",")]
    return sorted(s11_rows, key=lambda row: row["condition_id"])


def expected_counts_for_assignment_bank(bank: dict[str, Any], repeat_idx: int, assignment: list[str]) -> dict[str, int]:
    if bank.get("countsPerRepeat"):
        return {str(key): int(value) for key, value in bank["countsPerRepeat"][repeat_idx].items()}
    labels = list(map(str, bank["labels"]))
    per_label = len(assignment) // len(labels)
    remainder = len(assignment) % len(labels)
    expected = {label: per_label for label in labels}
    for label in labels[:remainder]:
        expected[label] += 1
    return expected


def scheduler_seed(base_seed: int, condition_id: str, repeat_idx: int) -> int:
    condition_num = int(condition_id.replace("E01C", ""))
    return base_seed * 1_000_000 + 110 * 10_000 + condition_num * 1_000 + repeat_idx


def component_algorithms(assignments: list[str]) -> list[str]:
    label_to_behavior = {
        "bubble": "bubble",
        "insertion": "insertion",
        "selection": "selection",
        "bubble_label_a": "bubble",
        "bubble_label_b": "bubble",
    }
    return sorted(set(label_to_behavior[label] for label in assignments))


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
    if len(rows) != 3:
        errors.append(f"Expected 3 S11 condition rows; found {len(rows)}.")
    if observed_mixes != EXPECTED_MIXES:
        errors.append(f"S11 mix mismatch: missing={sorted(EXPECTED_MIXES - observed_mixes)}, extra={sorted(observed_mixes - EXPECTED_MIXES)}.")
    value_banks = cfg["seedBanks"]["valueBanks"]
    assignment_banks = cfg["seedBanks"]["algotypeAssignmentBanks"]
    for row in rows:
        if row["mode"] != "cell_view" or row["algorithm"] != "mixed":
            errors.append(f"{row['condition_id']} is not a cell_view mixed condition.")
        if row["run_family"] != "duplicate_value_chimera":
            errors.append(f"{row['condition_id']} has unexpected run_family={row['run_family']}.")
        if row["value_bank_id"] != "duplicate_1_to_10_x10":
            errors.append(f"{row['condition_id']} value_bank_id={row['value_bank_id']} is not duplicate_1_to_10_x10.")
        if int(row["repeat_count"]) != repeat_count:
            errors.append(f"{row['condition_id']} repeat_count={row['repeat_count']} differs from config repeatCount={repeat_count}.")
        if int(row["frozen_count"]) != 0 or row["frozen_semantics"] != "none":
            errors.append(f"{row['condition_id']} should be unperturbed for S11.")
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
        for repeat_idx in range(repeats_to_run):
            values = list(map(int, value_bank["initialArrays"][repeat_idx]))
            assignment = list(map(str, assignment_bank["assignments"][repeat_idx]))
            if len(values) != array_length:
                errors.append(f"{row['condition_id']} repeat {repeat_idx} value length {len(values)} != {array_length}.")
            if not duplicate_multiplicities_valid(values):
                errors.append(f"{row['condition_id']} repeat {repeat_idx} value counts {value_counts(values)} != {EXPECTED_VALUE_COUNTS}.")
            if len(assignment) != array_length:
                errors.append(f"{row['condition_id']} repeat {repeat_idx} assignment length mismatch.")
            if any(label not in {"bubble", "insertion", "selection"} for label in assignment):
                errors.append(f"{row['condition_id']} repeat {repeat_idx} has unsupported S11 labels: {sorted(set(assignment) - {'bubble', 'insertion', 'selection'})}.")
            expected_counts = expected_counts_for_assignment_bank(assignment_bank, repeat_idx, assignment)
            observed_counts = dict(Counter(assignment))
            if observed_counts != expected_counts:
                errors.append(f"{row['condition_id']} repeat {repeat_idx} assignment counts {observed_counts} != expected {expected_counts}.")
    return {
        "expectedConditionRows": 3,
        "observedConditionRows": len(rows),
        "expectedRepeatsPerCondition": repeat_count,
        "repeatsToRun": repeats_to_run,
        "inputValidationErrors": errors,
        "inputValidationWarnings": warnings,
        "inputValidationPassed": not errors,
    }


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
    initial_labels = list(map(str, run_result["initial_labels"]))
    final_labels = list(map(str, run_result["final_labels"]))
    initial_counts = dict(sorted(Counter(initial_labels).items()))
    final_counts = dict(sorted(Counter(final_labels).items()))
    initial_block_metrics = equal_value_block_metrics(initial_values, initial_labels)
    final_block_metrics = equal_value_block_metrics(final_values, final_labels)
    records = list(run_result["records"])
    peak_record = max(records, key=lambda record: float(record["aggregation_left_neighbor_percent"]))
    final_sortedness = sortedness_percent(final_values)
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
        "value_distribution": row["value_distribution"],
        "algotype_assignment_bank_id": row["algotype_assignment_bank_id"],
        "repeat_index": int(repeat_idx),
        "initial_array_seed": int(initial_seed),
        "scheduler_seed": int(scheduler_seed_value),
        "algotype_assignment_seed": int(assignment_seed),
        "component_algorithms_json": json.dumps(component_algorithms(assignments), separators=(",", ":")),
        "configured_algotype_counts_json": json.dumps(dict(sorted(configured_counts.items())), separators=(",", ":")),
        "initial_algotype_counts_json": json.dumps(initial_counts, separators=(",", ":")),
        "final_algotype_counts_json": json.dumps(final_counts, separators=(",", ":")),
        "configured_algotype_counts_match_initial": initial_counts == configured_counts,
        "final_algotype_counts_preserved": final_counts == initial_counts,
        "initial_value_counts_json": json.dumps({str(k): v for k, v in value_counts(initial_values).items()}, separators=(",", ":")),
        "final_value_counts_json": json.dumps({str(k): v for k, v in value_counts(final_values).items()}, separators=(",", ":")),
        "initial_value_multiplicities_valid": duplicate_multiplicities_valid(initial_values),
        "final_value_multiplicities_valid": duplicate_multiplicities_valid(final_values),
        "initial_array_sha256": sha256_json(initial_values),
        "final_array_sha256": sha256_json(final_values),
        "final_values_json": json.dumps(final_values, separators=(",", ":")),
        "swap_only_steps": int(run_result["swap_count"]),
        "comparison_steps_observed": int(run_result["comparison_count"]),
        "compare_plus_swap_steps": int(run_result["compare_plus_swap_count"]),
        "sweep_count": int(run_result["sweep_count"]),
        "final_sortedness_percent": final_sortedness,
        "final_monotonicity_error_count": monotonicity_error_count(final_values),
        "final_nondecreasing_ties_allowed": bool(final_sortedness == 100.0),
        "strict_unique_sortedness_applicable": False,
        "initial_aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(initial_labels),
        "initial_aggregation_right_neighbor_legacy_percent": aggregation_right_neighbor_legacy_percent(initial_labels),
        "final_aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(final_labels),
        "final_aggregation_right_neighbor_legacy_percent": aggregation_right_neighbor_legacy_percent(final_labels),
        "peak_raw_aggregation_left_neighbor_percent": float(peak_record["aggregation_left_neighbor_percent"]),
        "peak_raw_aggregation_left_neighbor_swap_step": int(peak_record["swap_step"]),
        "final_minus_initial_aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(final_labels) - aggregation_left_neighbor_percent(initial_labels),
        "initial_equal_value_blocks_valid": bool(initial_block_metrics["equal_value_blocks_valid"]),
        "final_equal_value_blocks_valid": bool(final_block_metrics["equal_value_blocks_valid"]),
        "initial_within_equal_value_same_label_pair_percent": initial_block_metrics["within_equal_value_same_label_pair_percent"],
        "final_within_equal_value_same_label_pair_percent": final_block_metrics["within_equal_value_same_label_pair_percent"],
        "final_within_equal_value_same_label_pair_n100_percent": final_block_metrics["within_equal_value_same_label_pair_n100_percent"],
        "final_within_equal_value_pair_count": final_block_metrics["within_equal_value_pair_count"],
        "final_within_equal_value_same_label_pair_count": final_block_metrics["within_equal_value_same_label_pair_count"],
        "final_equal_value_block_lengths_json": final_block_metrics["equal_value_block_lengths_json"],
        "final_equal_value_block_label_counts_json": final_block_metrics["equal_value_block_label_counts_json"],
        "stop_reason": run_result["stop_reason"],
        "max_guard_hit": bool(run_result["max_guard_hit"]),
        "elapsed_seconds": float(elapsed_seconds),
        "wrapper_name": run_result["wrapper_name"],
        "comparison_count_source": "public_status_probe_compare_and_swap_count",
        "comparison_count_semantics": "Actionable-comparison proxy inherited from S09/S05; public cell classes increment StatusProbe.compare_and_swap_count when should_move() is true, not for every value read.",
        "duplicate_sortedness_semantics": "Nondecreasing order with ties allowed; strict unique-value ranking is not applicable for 10 copies each of values 1..10.",
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
    s09 = load_s09_module(repo_dir)
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
        run_result = s09.run_same_goal_chimera_sync(
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
        for pct, left_value, right_value, sortedness_value in zip(grid, interp_left, interp_right, interp_sortedness, strict=True):
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "source_wrapper_step_id": SOURCE_WRAPPER_STEP_ID,
                    "source_aggregation_step_id": SOURCE_AGGREGATION_STEP_ID,
                    "experiment_id": EXPERIMENT_ID,
                    "condition_id": condition_id,
                    "algotype_mix": first["algotype_mix"],
                    "repeat_index": int(repeat_index),
                    "progress_percent": float(pct),
                    "aggregation_left_neighbor_percent": float(left_value),
                    "aggregation_right_neighbor_legacy_percent": float(right_value),
                    "sortedness_percent": float(sortedness_value),
                    "expected_random_left_neighbor_percent": float(chance),
                    "final_swap_step": int(final_swap),
                }
            )
    return pd.DataFrame(rows)


def summarize_curves(run_grid: pd.DataFrame) -> pd.DataFrame:
    grouped = run_grid.groupby(["condition_id", "algotype_mix", "progress_percent"], as_index=False)
    curves = grouped.agg(
        n_runs=("repeat_index", "nunique"),
        mean_aggregation_left_neighbor_percent=("aggregation_left_neighbor_percent", "mean"),
        std_aggregation_left_neighbor_percent=("aggregation_left_neighbor_percent", "std"),
        mean_aggregation_right_neighbor_legacy_percent=("aggregation_right_neighbor_legacy_percent", "mean"),
        std_aggregation_right_neighbor_legacy_percent=("aggregation_right_neighbor_legacy_percent", "std"),
        mean_sortedness_percent=("sortedness_percent", "mean"),
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
    return curves.sort_values(["condition_id", "progress_percent"]).reset_index(drop=True)


def build_within_block_table(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in run_df.iterrows():
        block_counts = json.loads(row["final_equal_value_block_label_counts_json"])
        for value, blocks in block_counts.items():
            for block_index, counts in enumerate(blocks):
                rows.append(
                    {
                        "research_step_id": STEP_ID,
                        "condition_id": row["condition_id"],
                        "algotype_mix": row["algotype_mix"],
                        "repeat_index": int(row["repeat_index"]),
                        "value": int(value),
                        "block_index": int(block_index),
                        "block_label_counts_json": json.dumps(counts, separators=(",", ":")),
                        "block_size": int(sum(int(v) for v in counts.values())),
                        "dominant_label": max(counts.items(), key=lambda item: int(item[1]))[0],
                        "dominant_label_count": int(max(int(v) for v in counts.values())),
                    }
                )
    return pd.DataFrame(rows)


def build_condition_summary(
    run_df: pd.DataFrame,
    curves: pd.DataFrame,
    s10_peak_table: pd.DataFrame,
) -> pd.DataFrame:
    s10_lookup = s10_peak_table.set_index("algotype_mix").to_dict("index") if not s10_peak_table.empty else {}
    rows: list[dict[str, Any]] = []
    for condition_id, subset in run_df.groupby("condition_id", sort=True):
        first = subset.iloc[0]
        mix = first["algotype_mix"]
        curve_subset = curves[curves["condition_id"] == condition_id].sort_values("progress_percent")
        left_peak = curve_subset.loc[curve_subset["mean_aggregation_left_neighbor_percent"].idxmax()]
        right_peak = curve_subset.loc[curve_subset["mean_aggregation_right_neighbor_legacy_percent"].idxmax()]
        s10_mix = MIX_TO_S10_UNIQUE[mix]
        s10_ref = s10_lookup.get(s10_mix, {})
        unique_peak = float(s10_ref.get("mean_curve_peak_aggregation_left_neighbor_percent", np.nan))
        unique_final = float(s10_ref.get("final_mean_aggregation_left_neighbor_percent", np.nan))
        rows.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "algotype_mix": mix,
                "display_label": MIX_LABELS.get(mix, mix),
                "matched_group_id": first["matched_group_id"],
                "component_algorithms_json": first["component_algorithms_json"],
                "repetitions_observed": int(subset["repeat_index"].nunique()),
                "mean_final_sortedness_percent": float(subset["final_sortedness_percent"].mean()),
                "min_final_sortedness_percent": float(subset["final_sortedness_percent"].min()),
                "max_final_monotonicity_error_count": int(subset["final_monotonicity_error_count"].max()),
                "all_runs_nondecreasing_ties_allowed": bool(subset["final_nondecreasing_ties_allowed"].all()),
                "stop_reason_counts": json.dumps(dict(sorted(Counter(subset["stop_reason"]).items())), separators=(",", ":")),
                "max_guard_runs": int(subset["max_guard_hit"].sum()),
                "mean_swap_only_steps": float(subset["swap_only_steps"].mean()),
                "std_swap_only_steps": float(subset["swap_only_steps"].std(ddof=1)),
                "mean_comparison_steps_observed": float(subset["comparison_steps_observed"].mean()),
                "mean_compare_plus_swap_steps": float(subset["compare_plus_swap_steps"].mean()),
                "initial_mean_aggregation_left_neighbor_percent": float(subset["initial_aggregation_left_neighbor_percent"].mean()),
                "final_mean_aggregation_left_neighbor_percent": float(subset["final_aggregation_left_neighbor_percent"].mean()),
                "final_std_aggregation_left_neighbor_percent": float(subset["final_aggregation_left_neighbor_percent"].std(ddof=1)),
                "final_minus_initial_mean_aggregation_left_neighbor_percent": float(subset["final_minus_initial_aggregation_left_neighbor_percent"].mean()),
                "mean_curve_peak_aggregation_left_neighbor_percent": float(left_peak["mean_aggregation_left_neighbor_percent"]),
                "mean_curve_peak_left_neighbor_progress_percent": float(left_peak["progress_percent"]),
                "mean_curve_peak_aggregation_right_neighbor_legacy_percent": float(right_peak["mean_aggregation_right_neighbor_legacy_percent"]),
                "mean_curve_peak_right_neighbor_legacy_progress_percent": float(right_peak["progress_percent"]),
                "expected_random_left_neighbor_percent": float(left_peak["expected_random_left_neighbor_percent"]),
                "final_mean_within_equal_value_same_label_pair_percent": float(subset["final_within_equal_value_same_label_pair_percent"].mean()),
                "final_std_within_equal_value_same_label_pair_percent": float(subset["final_within_equal_value_same_label_pair_percent"].std(ddof=1)),
                "final_mean_within_equal_value_same_label_pair_n100_percent": float(subset["final_within_equal_value_same_label_pair_n100_percent"].mean()),
                "final_within_equal_value_pair_count_min": int(subset["final_within_equal_value_pair_count"].min()),
                "final_within_equal_value_pair_count_max": int(subset["final_within_equal_value_pair_count"].max()),
                "all_initial_value_multiplicities_valid": bool(subset["initial_value_multiplicities_valid"].all()),
                "all_final_value_multiplicities_valid": bool(subset["final_value_multiplicities_valid"].all()),
                "all_final_equal_value_blocks_valid": bool(subset["final_equal_value_blocks_valid"].all()),
                "s10_unique_algotype_mix": s10_mix,
                "s10_unique_mean_curve_peak_aggregation_left_neighbor_percent": unique_peak,
                "s10_unique_final_mean_aggregation_left_neighbor_percent": unique_final,
                "final_minus_s10_unique_final_left_neighbor_percent": float(subset["final_aggregation_left_neighbor_percent"].mean() - unique_final) if not np.isnan(unique_final) else np.nan,
                "final_minus_s10_unique_peak_left_neighbor_percent": float(subset["final_aggregation_left_neighbor_percent"].mean() - unique_peak) if not np.isnan(unique_peak) else np.nan,
                "final_above_s10_unique_final": bool(subset["final_aggregation_left_neighbor_percent"].mean() > unique_final) if not np.isnan(unique_final) else False,
                "final_above_s10_unique_peak": bool(subset["final_aggregation_left_neighbor_percent"].mean() > unique_peak) if not np.isnan(unique_peak) else False,
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
    trace_df = trace_df.copy()
    trace_df["recomputed_left"] = trace_df["algotype_positions_code"].map(aggregation_left_from_code)
    trace_df["recomputed_right"] = trace_df["algotype_positions_code"].map(aggregation_right_from_code)
    left_diff = (trace_df["recomputed_left"] - trace_df["aggregation_left_neighbor_percent"]).abs()
    right_diff = (trace_df["recomputed_right"] - trace_df["aggregation_right_neighbor_legacy_percent"]).abs()
    left_matches = bool((left_diff <= 1e-9).all())
    right_matches = bool((right_diff <= 1e-9).all())
    if not left_matches:
        errors.append(f"Left-neighbor aggregation reconstruction mismatch; max abs diff={float(left_diff.max())}.")
    if not right_matches:
        errors.append(f"Right-neighbor aggregation reconstruction mismatch; max abs diff={float(right_diff.max())}.")
    trace_codes_complete = bool(trace_df["algotype_positions_code"].map(len).eq(int(cfg["globalDefaults"]["arrayLength"])).all())
    invalid_codes = sorted(set("".join(trace_df["algotype_positions_code"].astype(str).unique())) - set(CODE_TO_LABEL))
    final_join = run_df.merge(
        final_trace[
            [
                "condition_id",
                "repeat_index",
                "final_array_sha256",
                "final_values_json",
                "sortedness_percent",
                "monotonicity_error_count",
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
        and (final_join["sortedness_percent"] == final_join["final_sortedness_percent"]).all()
        and (final_join["monotonicity_error_count"] == final_join["final_monotonicity_error_count"]).all()
        and (final_join["aggregation_left_neighbor_percent"] == final_join["final_aggregation_left_neighbor_percent"]).all()
    )
    recomputed_block_rows: list[dict[str, Any]] = []
    for _, row in final_trace.iterrows():
        values = list(map(int, json.loads(row["final_values_json"])))
        labels = labels_from_code(str(row["algotype_positions_code"]))
        metrics = equal_value_block_metrics(values, labels)
        recomputed_block_rows.append(
            {
                "condition_id": row["condition_id"],
                "repeat_index": int(row["repeat_index"]),
                "final_equal_value_blocks_valid_trace": bool(metrics["equal_value_blocks_valid"]),
                "final_within_equal_value_same_label_pair_percent_trace": float(metrics["within_equal_value_same_label_pair_percent"]),
                "final_within_equal_value_pair_count_trace": int(metrics["within_equal_value_pair_count"]),
            }
        )
    block_check = pd.DataFrame(recomputed_block_rows)
    block_join = run_df.merge(block_check, on=["condition_id", "repeat_index"], how="left", validate="one_to_one")
    block_metrics_recomputed_match = bool(
        block_join["final_equal_value_blocks_valid_trace"].notna().all()
        and (block_join["final_equal_value_blocks_valid"] == block_join["final_equal_value_blocks_valid_trace"]).all()
        and np.allclose(
            block_join["final_within_equal_value_same_label_pair_percent"],
            block_join["final_within_equal_value_same_label_pair_percent_trace"],
            equal_nan=True,
        )
        and (block_join["final_within_equal_value_pair_count"] == block_join["final_within_equal_value_pair_count_trace"]).all()
    )
    if not block_metrics_recomputed_match:
        errors.append("Final within-equal-value block metrics do not recompute from final trace rows.")
    curve_rows_ok = len(curves) == len(rows) * int(progress_points)
    summary_rows_ok = len(condition_summary) == len(rows)
    all_value_multiplicities_valid = bool(
        run_df["initial_value_multiplicities_valid"].all()
        and run_df["final_value_multiplicities_valid"].all()
    )
    all_final_blocks_valid = bool(run_df["final_equal_value_blocks_valid"].all())
    all_runs_sorted = bool(run_df["final_nondecreasing_ties_allowed"].all())
    no_max_guard = bool((~run_df["max_guard_hit"]).all())
    count_fields_complete = bool(
        run_df[["swap_only_steps", "comparison_steps_observed", "compare_plus_swap_steps"]].notna().all(axis=1).all()
        and (run_df["swap_only_steps"] + run_df["comparison_steps_observed"] == run_df["compare_plus_swap_steps"]).all()
    )
    algotype_counts_ok = bool(
        run_df["configured_algotype_counts_match_initial"].all()
        and run_df["final_algotype_counts_preserved"].all()
    )
    full_production = bool(max_repeats is None and repeats_to_run == int(cfg["globalDefaults"]["repeatCount"]))
    if not full_production:
        warnings.append(f"Run used smoke repeat count {repeats_to_run}, not full production repeat count {cfg['globalDefaults']['repeatCount']}.")
    primary_mixes = {"duplicate_bubble_selection", "duplicate_insertion_selection"}
    primary_summary = condition_summary[condition_summary["algotype_mix"].isin(primary_mixes)]
    primary_final_above_initial = bool((primary_summary["final_minus_initial_mean_aggregation_left_neighbor_percent"] > 0).all())
    primary_final_above_unique_final = bool(primary_summary["final_above_s10_unique_final"].all())
    primary_final_above_unique_peak = bool(primary_summary["final_above_s10_unique_peak"].all())
    success = bool(
        input_validation["inputValidationPassed"]
        and len(run_df) == expected_run_rows
        and repeat_count_ok
        and final_rows_one_per_run
        and left_matches
        and right_matches
        and trace_codes_complete
        and not invalid_codes
        and trace_final_matches_run
        and block_metrics_recomputed_match
        and curve_rows_ok
        and summary_rows_ok
        and all_value_multiplicities_valid
        and all_final_blocks_valid
        and all_runs_sorted
        and no_max_guard
        and count_fields_complete
        and algotype_counts_ok
        and full_production
    )
    outcome = "supportive" if success and primary_final_above_initial and primary_final_above_unique_final else "constraining/contradictory"
    if not primary_final_above_unique_peak:
        warnings.append("At least one S11 primary duplicate mix final Aggregation did not exceed its S10 unique-value mean-curve peak.")
    caveats = [
        "S11 uses the S09 deterministic single-thread public-cell wrapper because the public mixed runner would not preserve S03 assignment semantics.",
        "Duplicate Sortedness is evaluated as nondecreasing order with ties allowed; strict unique-value ranking is not applicable for ten copies each of values 1..10.",
        "Selection cells retain the public SelectionSortCell target-update behavior; duplicate ties can change which equal-valued cell occupies a given final position without changing nondecreasing Sortedness.",
        "Aggregation uses the S10/S03 left-neighbor denominator n=100, with right-neighbor sensitivity retained.",
        "Comparison counts inherit the S05/S09 actionable-comparison proxy caveat.",
    ]
    if not primary_final_above_unique_peak:
        caveats.append("Primary duplicate final Aggregation is compared against S10 unique final values as the main persistence check; the stricter unique-peak comparison is recorded as a caveat/sensitivity.")
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed_with_caveats" if success else "completed_validation_failed",
        "validationResult": "passed_with_duplicate_value_and_block_aggregation_caveats" if success else "failed",
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
        "finalAlgotypeCountsPreserved": bool(run_df["final_algotype_counts_preserved"].all()),
        "initialValueMultiplicitiesValid": bool(run_df["initial_value_multiplicities_valid"].all()),
        "finalValueMultiplicitiesValid": bool(run_df["final_value_multiplicities_valid"].all()),
        "allValueMultiplicitiesValid": all_value_multiplicities_valid,
        "finalEqualValueBlocksValid": all_final_blocks_valid,
        "finalWithinEqualValuePairCountMin": int(run_df["final_within_equal_value_pair_count"].min()),
        "finalWithinEqualValuePairCountMax": int(run_df["final_within_equal_value_pair_count"].max()),
        "withinEqualValueBlockMetricsRecomputedFromTrace": block_metrics_recomputed_match,
        "leftNeighborAggregationRecomputedMatchesTrace": left_matches,
        "rightNeighborAggregationRecomputedMatchesTrace": right_matches,
        "maxLeftAggregationRecomputeAbsDiff": float(left_diff.max()) if len(left_diff) else None,
        "maxRightAggregationRecomputeAbsDiff": float(right_diff.max()) if len(right_diff) else None,
        "invalidPositionCodeCharacters": invalid_codes,
        "traceAlgotypePositionCodesComplete": trace_codes_complete,
        "exactlyOneFinalTraceRowPerRun": final_rows_one_per_run,
        "traceFinalMatchesRunRecords": trace_final_matches_run,
        "allRunsNondecreasingTiesAllowed": all_runs_sorted,
        "noMaxGuardRuns": no_max_guard,
        "countFieldsCompleteAndAdditive": count_fields_complete,
        "stopReasonCounts": {str(key): int(value) for key, value in Counter(run_df["stop_reason"]).items()},
        "primaryDuplicateMixesFinalAboveInitialAggregation": primary_final_above_initial,
        "primaryDuplicateMixesFinalAboveS10UniqueFinalAggregation": primary_final_above_unique_final,
        "primaryDuplicateMixesFinalAboveS10UniquePeakAggregation": primary_final_above_unique_peak,
        "inputValidation": input_validation,
        "validationErrors": errors,
        "validationWarnings": warnings,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Proceed to S12 opposite-direction chimeras using S03 opposite-direction configs, S10 aggregation definitions, and the S11 duplicate-value handling; do not start S12 inside S11.",
    }


def plot_figure(curves: pd.DataFrame, condition_summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.55, 1.0]})
    for mix in ["duplicate_bubble_selection", "duplicate_bubble_insertion", "duplicate_insertion_selection"]:
        subset = curves[curves["algotype_mix"] == mix].sort_values("progress_percent")
        if subset.empty:
            continue
        color = MIX_COLORS.get(mix, "#555555")
        axes[0].plot(
            subset["progress_percent"],
            subset["mean_aggregation_left_neighbor_percent"],
            color=color,
            linewidth=2.2,
            label=MIX_LABELS.get(mix, mix),
        )
        axes[0].fill_between(
            subset["progress_percent"].to_numpy(dtype=float),
            subset["ci95_low_aggregation_left_neighbor_percent"].to_numpy(dtype=float),
            subset["ci95_high_aggregation_left_neighbor_percent"].to_numpy(dtype=float),
            color=color,
            alpha=0.13,
            linewidth=0,
        )
    axes[0].set_title("Duplicate-value Aggregation trajectories")
    axes[0].set_xlabel("Run progress (% of final swap count)")
    axes[0].set_ylabel("Aggregation (%)")
    axes[0].set_ylim(40, max(75.0, float(curves["mean_aggregation_left_neighbor_percent"].max()) + 5.0))
    axes[0].grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.45)
    axes[0].legend(frameon=False, fontsize=8)

    bar_df = condition_summary.sort_values("final_mean_aggregation_left_neighbor_percent")
    y = np.arange(len(bar_df))
    axes[1].barh(
        y - 0.18,
        bar_df["final_mean_aggregation_left_neighbor_percent"],
        height=0.34,
        color=[MIX_COLORS.get(mix, "#777777") for mix in bar_df["algotype_mix"]],
        label="Final global",
    )
    axes[1].barh(
        y + 0.18,
        bar_df["final_mean_within_equal_value_same_label_pair_percent"],
        height=0.34,
        color="#7a7f85",
        label="Final within-value",
    )
    axes[1].set_yticks(y, [MIX_LABELS.get(mix, mix) for mix in bar_df["algotype_mix"]])
    axes[1].set_title("Final Aggregation")
    axes[1].set_xlabel("Same-Algotype adjacent pairs (%)")
    axes[1].grid(True, axis="x", color="#d0d0d0", linewidth=0.7, alpha=0.45)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("E01 S11 duplicate-value chimeras", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
        "title": "Reproduce duplicate-value chimeras",
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
        "artifactsWritten": [str(path) for path in paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s11_duplicate_chimeras.py"),
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
            "S11 reproduced the qualitative duplicate-value aggregation effect for the two Selection-containing "
            "same-goal chimeras: final Aggregation stayed above the unique-value final baseline while all runs "
            "ended nondecreasing with ten copies of each value preserved."
        )
    else:
        lay_summary = (
            "S11 completed the duplicate-value chimera runs and preserved value multiplicities, but the primary "
            "duplicate-value aggregation persistence checks were not all supportive under the frozen definitions."
        )
    compact_summary = condition_summary[
        [
            "condition_id",
            "display_label",
            "repetitions_observed",
            "mean_final_sortedness_percent",
            "mean_curve_peak_aggregation_left_neighbor_percent",
            "mean_curve_peak_left_neighbor_progress_percent",
            "final_mean_aggregation_left_neighbor_percent",
            "final_mean_within_equal_value_same_label_pair_percent",
            "s10_unique_mean_curve_peak_aggregation_left_neighbor_percent",
            "s10_unique_final_mean_aggregation_left_neighbor_percent",
            "final_minus_s10_unique_final_left_neighbor_percent",
            "final_minus_s10_unique_peak_left_neighbor_percent",
        ]
    ].copy()
    report = f"""# E01 S11 Research Step Full Results

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

When duplicate values relax final-order constraints, does Algotype aggregation persist or increase at the end of Bubble-Selection and Insertion-Selection mixed runs?

## Inputs

- S03 baseline config: `{args.config_path}` (`{sha256_file(args.config_path)}`)
- S03 condition matrix: `{args.condition_matrix_path}` (`{sha256_file(args.condition_matrix_path)}`)
- S10 peak table for unique-value comparison: `{args.s10_peak_table_path}` (`{sha256_file(args.s10_peak_table_path)}`)
- Duplicate value bank: `duplicate_1_to_10_x10`, 100 arrays, each intended to contain ten copies each of values 1..10.
- S11 condition IDs: E01C048, E01C049, and E01C050.

## Detailed Methods

S11 ran only the duplicate-value same-goal chimeras frozen in S03. It reused the S09 deterministic single-thread wrapper around the public `BubbleSortCell`, `InsertionSortCell`, and `SelectionSortCell` classes so that S03 Algotype assignments and scheduler seeds were auditable. The public mixed-thread runner was not used because S09 already found it could not preserve the frozen same-goal assignment semantics.

For each condition and repeat, the initial values came from the S03 `duplicate_1_to_10_x10` bank. The validation required exactly ten copies of each value from 1 through 10 before and after the run. The stop condition used nondecreasing order with ties allowed. This is the duplicate-value Sortedness assumption: strict unique-value ranking is not meaningful because there are ten cells with each value.

Aggregation used the S10 definitions unchanged. The primary metric is left-neighbor Aggregation with denominator n=100: `100 * count(label[i] == label[i - 1] for i=1..99) / 100`. The right-neighbor legacy sensitivity was retained. S11 additionally computed final within-equal-value-block Aggregation as the percentage of adjacent equal-value pairs whose Algotype labels match. For a valid final sorted duplicate array, there are 90 within-value adjacent pairs: 9 per value block times 10 values.

Figure-style curves were aligned by percent of final swap count on a `{args.progress_points}`-point grid. Condition summaries compare duplicate final Aggregation to the matching S10 unique-value final Aggregation and mean-curve peak.

## Commands

```bash
{" ".join(command)}
```

Validation/smoke commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/e01_s11_duplicate_chimeras.py
PYTHONDONTWRITEBYTECODE=1 python scripts/e01_s11_duplicate_chimeras.py --repo-dir /workspace/cell-research --artifacts-dir /cache/e01_s11_smoke_artifacts --max-repeats 2 --workers 2
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
- Wall time: `{wall_seconds:.3f}` seconds

No new Python, system, R, Rust, or Node dependencies were installed for S11.

## Results

Duplicate-value condition summary:

{dataframe_to_markdown(compact_summary)}

Primary interpretation:

- Production runs completed: `{validation["fullProductionRepeatCount"]}`.
- Initial and final value multiplicities valid: `{validation["allValueMultiplicitiesValid"]}`.
- Final equal-value blocks valid: `{validation["finalEqualValueBlocksValid"]}`.
- Final within-equal-value block metrics recomputed from trace: `{validation["withinEqualValueBlockMetricsRecomputedFromTrace"]}`.
- Bubble-Selection and Insertion-Selection final Aggregation above their own initial Aggregation: `{validation["primaryDuplicateMixesFinalAboveInitialAggregation"]}`.
- Bubble-Selection and Insertion-Selection final Aggregation above matching S10 unique-value final Aggregation: `{validation["primaryDuplicateMixesFinalAboveS10UniqueFinalAggregation"]}`.
- Bubble-Selection and Insertion-Selection final Aggregation above matching S10 unique-value peak Aggregation: `{validation["primaryDuplicateMixesFinalAboveS10UniquePeakAggregation"]}`.

## Validation

- Run rows: `{validation["runRows"]}` of expected `{validation["expectedRuns"]}`.
- Trace rows: `{validation["traceRows"]}`.
- Curve rows: `{validation["curveRows"]}`.
- Repeat counts by condition: `{validation["repeatCountsByCondition"]}`.
- Stop reason counts: `{validation["stopReasonCounts"]}`.
- Final Sortedness uses nondecreasing duplicate ties: `{validation["allRunsNondecreasingTiesAllowed"]}`.
- No max-guard runs: `{validation["noMaxGuardRuns"]}`.
- Count fields complete and additive: `{validation["countFieldsCompleteAndAdditive"]}`.
- Left-neighbor Aggregation recomputed from trace codes: `{validation["leftNeighborAggregationRecomputedMatchesTrace"]}`.
- Right-neighbor sensitivity recomputed from trace codes: `{validation["rightNeighborAggregationRecomputedMatchesTrace"]}`.
- Final trace rows match run records: `{validation["traceFinalMatchesRunRecords"]}`.
- Within-equal-value final adjacent-pair count min/max: `{validation["finalWithinEqualValuePairCountMin"]}` / `{validation["finalWithinEqualValuePairCountMax"]}`.
- Validation warnings: `{validation["validationWarnings"]}`.
- Validation errors: `{validation["validationErrors"]}`.

Detailed validation JSON:

```json
{json.dumps(validation, indent=2)}
```

## Artifacts And Provenance

Reusable outputs:

{chr(10).join(f"- `{label}`: `{path}`" for label, path in artifact_paths.items())}

The global run manifest and checksum file were updated after artifact creation. The S11 artifact manifest records paths, sizes, and SHA256 hashes.

## Caveats, Blockers, Failed Assumptions, And Limitations

- Duplicate-value Sortedness is nondecreasing Sortedness with ties allowed. Strict rank-order Sortedness is undefined for ten copies of each value unless an extra tie-breaker is invented; S11 did not invent one.
- Public `SelectionSortCell` behavior with duplicates can choose among equal-valued cells arbitrarily while still satisfying the value-level goal. That is part of the frozen duplicate-value semantic assumption.
- S11 preserves S10's left-neighbor primary Aggregation denominator n=100. The within-equal-value-block metric uses eligible equal-value adjacent pairs as its denominator and is reported separately.
- S11 inherits the S09 deterministic wrapper caveat and the S05/S09 comparison-counting caveat.
- The stricter comparison of duplicate final Aggregation against S10 unique-value peak Aggregation is recorded as a sensitivity check; the main persistence check is final duplicate versus final unique and initial duplicate Aggregation.

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    s11_dir = artifacts_dir / "research_steps" / STEP_ID
    traces_dir = artifacts_dir / "traces"
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e01"
    tables_dir = artifacts_dir / "tables"
    for path in (s11_dir, traces_dir, results_dir, figures_dir, tables_dir, artifacts_dir / "checksums"):
        path.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.config_path.read_text(encoding="utf-8"))
    rows = load_s11_conditions(args.condition_matrix_path)
    input_validation = validate_condition_inputs(cfg, rows, args.max_repeats)
    if not input_validation["inputValidationPassed"]:
        raise SystemExit("S11 input validation failed: " + json.dumps(input_validation, indent=2))

    repeats_to_run = int(input_validation["repeatsToRun"])
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    worker_payloads = [
        {
            "cfg": cfg,
            "row": row,
            "repo_dir": str(args.repo_dir),
            "condition_cache_dir": str(args.cache_dir / row["condition_id"]),
            "repeats_to_run": repeats_to_run,
            "max_successful_swaps": int(args.max_successful_swaps),
            "max_sweeps": int(args.max_sweeps),
            "stall_sweeps": int(args.stall_sweeps),
        }
        for row in rows
    ]
    worker_results: list[dict[str, Any]] = []
    workers = max(1, min(int(args.workers), len(worker_payloads)))
    if workers == 1:
        for payload in worker_payloads:
            worker_results.append(run_condition_worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_condition_worker, payload) for payload in worker_payloads]
            for future in as_completed(futures):
                worker_results.append(future.result())
    worker_results.sort(key=lambda result: result["condition_id"])
    trace_df, run_df = read_condition_outputs(worker_results)
    run_grid = build_run_grid(trace_df, args.progress_points)
    curves = summarize_curves(run_grid)
    s10_peak_table = pd.read_csv(args.s10_peak_table_path)
    condition_summary = build_condition_summary(run_df, curves, s10_peak_table)
    within_block_table = build_within_block_table(run_df)
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
        "finalMeanAggregationLeftNeighborPercentByMix": {
            str(row["algotype_mix"]): float(row["final_mean_aggregation_left_neighbor_percent"])
            for _, row in condition_summary.iterrows()
        },
        "peakMeanAggregationLeftNeighborPercentByMix": {
            str(row["algotype_mix"]): float(row["mean_curve_peak_aggregation_left_neighbor_percent"])
            for _, row in condition_summary.iterrows()
        },
        "finalWithinEqualValueSameLabelPairPercentByMix": {
            str(row["algotype_mix"]): float(row["final_mean_within_equal_value_same_label_pair_percent"])
            for _, row in condition_summary.iterrows()
        },
    }

    report_path = s11_dir / "research_step_full_results.md"
    trace_path = traces_dir / "e01_duplicate_value_chimeras.parquet"
    result_path = results_dir / "e01_duplicate_aggregation.parquet"
    curves_path = results_dir / "e01_duplicate_aggregation_curves.parquet"
    figure_path = figures_dir / "figure8_duplicate_chimeras.png"
    run_records_path = s11_dir / "e01_duplicate_chimera_run_records.parquet"
    summary_csv_path = tables_dir / "e01_duplicate_aggregation_summary.csv"
    within_block_csv_path = tables_dir / "e01_duplicate_within_block_summary.csv"
    validation_path = s11_dir / "s11_validation.json"
    status_path = s11_dir / "status.json"
    manifest_path = s11_dir / "artifact_manifest.json"

    trace_df.to_parquet(trace_path, index=False)
    run_df.to_parquet(run_records_path, index=False)
    condition_summary.to_parquet(result_path, index=False)
    curves.to_parquet(curves_path, index=False)
    condition_summary.to_csv(summary_csv_path, index=False)
    within_block_table.to_csv(within_block_csv_path, index=False)
    plot_figure(curves, condition_summary, figure_path)

    artifact_paths = {
        "full_results_report": report_path,
        "duplicate_trace_parquet": trace_path,
        "duplicate_aggregation_result_parquet": result_path,
        "duplicate_aggregation_curves_parquet": curves_path,
        "duplicate_aggregation_figure": figure_path,
        "duplicate_run_records": run_records_path,
        "duplicate_aggregation_summary_csv": summary_csv_path,
        "duplicate_within_block_summary_csv": within_block_csv_path,
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
    validation["script"] = str(args.repo_dir / "scripts/e01_s11_duplicate_chimeras.py")
    validation["artifactsWritten"] = [str(path) for path in artifact_paths.values()]
    validation["allArtifactsExist"] = all(path.exists() for path in artifact_paths.values() if path not in {validation_path, status_path, manifest_path})
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
