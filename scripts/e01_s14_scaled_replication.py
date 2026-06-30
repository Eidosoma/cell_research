#!/usr/bin/env python3
"""Scale selected E01 replication conditions prioritized by S13."""

from __future__ import annotations

import argparse
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STEP_ID = "S14"
STEP_NUMBER = 14
EXPERIMENT_ID = "E01"
SCALED_REPEATS = 1000
BASE_SEED = 2026063014
ARRAY_LENGTH = 100
ALGORITHMS = ("bubble", "insertion", "selection")
FROZEN_SEMANTICS = ("passive", "stuck")
FROZEN_COUNT_SELECTED = 3
EFFICIENCY_METRICS = ("swap_only_steps", "compare_plus_swap_steps")
AGGREGATION_MIXES = (
    "same_goal_bubble_insertion",
    "same_goal_bubble_selection",
    "same_goal_insertion_selection",
    "same_goal_bubble_insertion_selection",
    "same_algorithm_bubble_label_control",
)
CORE_AGGREGATION_MIXES = tuple(mix for mix in AGGREGATION_MIXES if mix != "same_algorithm_bubble_label_control")
PAPER_RATIO_TARGETS = {
    ("bubble", "swap_only_steps"): 1.0,
    ("bubble", "compare_plus_swap_steps"): 1.0 / 1.5,
    ("insertion", "swap_only_steps"): 1.0,
    ("insertion", "compare_plus_swap_steps"): 1.0 / 2.03,
    ("selection", "swap_only_steps"): 11.0,
    ("selection", "compare_plus_swap_steps"): 1.17,
}
PAPER_AGGREGATION_PEAKS = {
    "same_goal_bubble_insertion": 65.0,
    "same_goal_bubble_selection": 72.0,
    "same_goal_insertion_selection": 69.0,
    "same_goal_bubble_insertion_selection": 62.0,
}
MIX_DISPLAY = {
    "same_goal_bubble_insertion": "Bubble-Insertion",
    "same_goal_bubble_selection": "Bubble-Selection",
    "same_goal_insertion_selection": "Insertion-Selection",
    "same_goal_bubble_insertion_selection": "Bubble-Insertion-Selection",
    "same_algorithm_bubble_label_control": "Bubble label control",
}
MIX_COLORS = {
    "same_goal_bubble_insertion": "#2b8a5e",
    "same_goal_bubble_selection": "#4169a8",
    "same_goal_insertion_selection": "#b34d4d",
    "same_goal_bubble_insertion_selection": "#6b5aa6",
    "same_algorithm_bubble_label_control": "#c77aa3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--config-path", type=Path, default=Path("/artifacts/configs/e01_baseline_configs.json"))
    parser.add_argument("--s13-status-path", type=Path, default=Path("/artifacts/tables/e01_replication_status.csv"))
    parser.add_argument("--s05-efficiency-table-path", type=Path, default=Path("/artifacts/tables/e01_efficiency_numeric_table.csv"))
    parser.add_argument("--s07-frozen-table-path", type=Path, default=Path("/artifacts/tables/e01_frozen_cell_robustness_comparison_table.csv"))
    parser.add_argument("--s10-aggregation-table-path", type=Path, default=Path("/artifacts/tables/e01_aggregation_peak_table.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/e01_s14"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-repeats", type=int, default=None, help="Optional smoke-test cap; production is 1000.")
    parser.add_argument("--force", action="store_true", help="Recompute checkpoints even when complete.")
    parser.add_argument("--max-successful-swaps", type=int, default=50000)
    parser.add_argument("--max-sweeps", type=int, default=20000)
    parser.add_argument("--stall-sweeps", type=int, default=2)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
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


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    tmp = df.copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    tmp = tmp.astype("string").fillna("").astype(str)
    headers = list(tmp.columns)
    rows = tmp.values.tolist()
    widths = [max(len(str(header)), *(len(row[col_idx]) for row in rows)) for col_idx, header in enumerate(headers)]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    divider_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body_lines = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def sortedness_percent(values: list[int]) -> float:
    if not values:
        return 100.0
    ordered_pairs = sum(1 for idx in range(1, len(values)) if values[idx - 1] <= values[idx])
    return 100.0 * (1 + ordered_pairs) / len(values)


def monotonicity_error_count(values: list[int]) -> int:
    return sum(1 for idx in range(1, len(values)) if values[idx] < values[idx - 1])


def aggregation_left_neighbor_percent(labels: list[str]) -> float:
    return 100.0 * sum(1 for idx in range(1, len(labels)) if labels[idx] == labels[idx - 1]) / len(labels)


def aggregation_right_neighbor_percent(labels: list[str]) -> float:
    return 100.0 * sum(1 for idx in range(len(labels) - 1) if labels[idx] == labels[idx + 1]) / len(labels)


def initial_values(seed_family: str, repeat_idx: int) -> tuple[list[int], int]:
    offsets = {"efficiency": 10_000, "frozen": 20_000, "aggregation": 30_000}
    seed = BASE_SEED + offsets[seed_family] + repeat_idx
    rng = random.Random(seed)
    values = list(range(1, ARRAY_LENGTH + 1))
    rng.shuffle(values)
    return values, seed


def frozen_indices(semantics: str, frozen_count: int, repeat_idx: int) -> tuple[list[int], int]:
    sem_offset = 0 if semantics == "passive" else 100_000
    seed = BASE_SEED + 40_000 + sem_offset + frozen_count * 1_000 + repeat_idx
    rng = random.Random(seed)
    return sorted(rng.sample(range(ARRAY_LENGTH), frozen_count)), seed


def assignments_for_mix(mix: str, repeat_idx: int) -> tuple[list[str], int, dict[str, int]]:
    seed = BASE_SEED + 50_000 + AGGREGATION_MIXES.index(mix) * 10_000 + repeat_idx
    rng = random.Random(seed)
    if mix == "same_goal_bubble_insertion":
        labels = ["bubble"] * 50 + ["insertion"] * 50
    elif mix == "same_goal_bubble_selection":
        labels = ["bubble"] * 50 + ["selection"] * 50
    elif mix == "same_goal_insertion_selection":
        labels = ["insertion"] * 50 + ["selection"] * 50
    elif mix == "same_algorithm_bubble_label_control":
        labels = ["bubble_label_a"] * 50 + ["bubble_label_b"] * 50
    elif mix == "same_goal_bubble_insertion_selection":
        order = [
            ("bubble", 34),
            ("insertion", 33),
            ("selection", 33),
        ]
        # Rotate the 34-cell remainder across repeats, matching S03's semantic assumption.
        shift = repeat_idx % 3
        rotated = order[shift:] + order[:shift]
        labels = []
        for label, count in rotated:
            labels.extend([label] * count)
    else:
        raise ValueError(f"Unknown mix: {mix}")
    rng.shuffle(labels)
    return labels, seed, dict(sorted(Counter(labels).items()))


def expected_random_left_neighbor_percent(counts: dict[str, int]) -> float:
    total = sum(int(value) for value in counts.values())
    numerator = sum(int(value) * (int(value) - 1) for value in counts.values())
    return 100.0 * numerator / (total * total)


def scheduler_seed(task_family: str, repeat_idx: int, *parts: str | int) -> int:
    payload = "|".join([task_family, str(repeat_idx), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return BASE_SEED * 1_000_000 + int(digest, 16) % 1_000_000


def stable_seed(*parts: str | int) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return BASE_SEED + int(digest, 16) % 100_000


def ensure_imports(repo_dir: Path) -> None:
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def condition_id_for_task(task: dict[str, Any]) -> str:
    if task["task_family"] == "efficiency":
        return f"S14F4_{task['algorithm']}_{task['mode']}"
    if task["task_family"] == "frozen":
        return f"S14F5_{task['frozen_semantics']}_f{task['frozen_count']}_{task['algorithm']}_{task['mode']}"
    if task["task_family"] == "aggregation":
        return f"S14F8_{task['algotype_mix']}"
    raise ValueError(f"Unknown task family: {task['task_family']}")


def run_efficiency_repeat(repo_dir: Path, algorithm: str, mode: str, repeat_idx: int, limits: dict[str, int]) -> dict[str, Any]:
    ensure_imports(repo_dir)
    from scripts.e01_s04_run_figure3_trajectories import TRADITIONAL_RUNNERS
    from scripts.e01_s09_same_goal_chimeras import run_same_goal_chimera_sync

    values, value_seed = initial_values("efficiency", repeat_idx)
    started = time.monotonic()
    if mode == "traditional":
        result = TRADITIONAL_RUNNERS[algorithm](values)
        wrapper_name = "reconstructed_traditional_controller"
        comparison_source = "reconstructed_controller_comparison_count"
    else:
        assignments = [algorithm] * len(values)
        result = run_same_goal_chimera_sync(
            repo_dir,
            values,
            assignments,
            scheduler_seed("efficiency", repeat_idx, algorithm, mode),
            limits["max_successful_swaps"],
            limits["max_sweeps"],
            limits["stall_sweeps"],
        )
        wrapper_name = "deterministic_single_thread_public_cell_methods_all_same_algotype"
        comparison_source = "public_status_probe_compare_and_swap_count"
    elapsed = time.monotonic() - started
    final_values = list(map(int, result["final_values"]))
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_family": "efficiency",
        "source_claim_id": "figure4_exact_fold_change_magnitudes",
        "condition_id": f"S14F4_{algorithm}_{mode}",
        "mode": mode,
        "algorithm": algorithm,
        "frozen_semantics": "none",
        "frozen_count": 0,
        "algotype_mix": "none",
        "repeat_index": repeat_idx,
        "sample_size_target": SCALED_REPEATS,
        "array_length": ARRAY_LENGTH,
        "initial_array_seed": value_seed,
        "scheduler_seed": scheduler_seed("efficiency", repeat_idx, algorithm, mode) if mode != "traditional" else np.nan,
        "frozen_index_seed": np.nan,
        "algotype_assignment_seed": np.nan,
        "initial_array_sha256": sha256_json(values),
        "initial_frozen_indices_json": "[]",
        "initial_algotype_counts_json": json.dumps({algorithm: ARRAY_LENGTH}, separators=(",", ":")) if mode == "cell_view" else "{}",
        "swap_only_steps": int(result["swap_count"]),
        "comparison_steps_observed": int(result["comparison_count"]),
        "compare_plus_swap_steps": int(result["compare_plus_swap_count"]),
        "final_sortedness_percent": sortedness_percent(final_values),
        "final_monotonicity_error_count": monotonicity_error_count(final_values),
        "final_aggregation_left_neighbor_percent": np.nan,
        "final_aggregation_right_neighbor_percent": np.nan,
        "run_peak_aggregation_left_neighbor_percent": np.nan,
        "run_peak_left_neighbor_progress_percent": np.nan,
        "sweep_count": int(result["sweep_count"]) if result.get("sweep_count") is not None else np.nan,
        "stop_reason": result["stop_reason"],
        "max_guard_hit": bool(result.get("max_guard_hit", result.get("timed_out", False))),
        "timed_out": bool(result.get("timed_out", False)),
        "wrapper_name": wrapper_name,
        "comparison_count_source": comparison_source,
        "elapsed_seconds": float(elapsed),
    }


def run_frozen_repeat(
    repo_dir: Path,
    algorithm: str,
    mode: str,
    semantics: str,
    frozen_count: int,
    repeat_idx: int,
    limits: dict[str, int],
) -> dict[str, Any]:
    ensure_imports(repo_dir)
    from scripts.e01_s07_frozen_robustness import TRADITIONAL_FROZEN_RUNNERS, run_cell_view_frozen_sync

    values, value_seed = initial_values("frozen", repeat_idx)
    indices, index_seed = frozen_indices(semantics, frozen_count, repeat_idx)
    started = time.monotonic()
    if mode == "traditional":
        result = TRADITIONAL_FROZEN_RUNNERS[algorithm](values, indices, semantics)
        wrapper_name = "reconstructed_traditional_frozen_controller"
        comparison_source = "reconstructed_controller_comparison_count"
    else:
        result = run_cell_view_frozen_sync(
            repo_dir,
            algorithm,
            values,
            indices,
            semantics,
            scheduler_seed("frozen", repeat_idx, algorithm, mode, semantics, frozen_count),
            limits["max_successful_swaps"],
            limits["max_sweeps"],
            limits["stall_sweeps"],
        )
        wrapper_name = "deterministic_single_thread_public_cell_methods"
        comparison_source = "public_status_probe_compare_and_swap_count"
    elapsed = time.monotonic() - started
    final_values = list(map(int, result["final_values"]))
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_family": "frozen_cell",
        "source_claim_id": "figure5_all_cell_view_lower_error_than_traditional",
        "condition_id": f"S14F5_{semantics}_f{frozen_count}_{algorithm}_{mode}",
        "mode": mode,
        "algorithm": algorithm,
        "frozen_semantics": semantics,
        "frozen_count": frozen_count,
        "algotype_mix": "none",
        "repeat_index": repeat_idx,
        "sample_size_target": SCALED_REPEATS,
        "array_length": ARRAY_LENGTH,
        "initial_array_seed": value_seed,
        "scheduler_seed": scheduler_seed("frozen", repeat_idx, algorithm, mode, semantics, frozen_count) if mode != "traditional" else np.nan,
        "frozen_index_seed": index_seed,
        "algotype_assignment_seed": np.nan,
        "initial_array_sha256": sha256_json(values),
        "initial_frozen_indices_json": json.dumps(indices, separators=(",", ":")),
        "initial_algotype_counts_json": "{}",
        "swap_only_steps": int(result["swap_count"]),
        "comparison_steps_observed": int(result["comparison_count"]),
        "compare_plus_swap_steps": int(result["compare_plus_swap_count"]),
        "final_sortedness_percent": sortedness_percent(final_values),
        "final_monotonicity_error_count": monotonicity_error_count(final_values),
        "final_aggregation_left_neighbor_percent": np.nan,
        "final_aggregation_right_neighbor_percent": np.nan,
        "run_peak_aggregation_left_neighbor_percent": np.nan,
        "run_peak_left_neighbor_progress_percent": np.nan,
        "sweep_count": int(result["sweep_count"]) if result.get("sweep_count") is not None else np.nan,
        "stop_reason": result["stop_reason"],
        "max_guard_hit": bool(result["max_guard_hit"]),
        "timed_out": False,
        "wrapper_name": wrapper_name,
        "comparison_count_source": comparison_source,
        "elapsed_seconds": float(elapsed),
    }


def build_aggregation_grid(records: list[dict[str, Any]], mix: str, repeat_idx: int, counts: dict[str, int]) -> tuple[list[dict[str, Any]], float, float]:
    grid = np.linspace(0.0, 100.0, 101)
    steps = np.array([record["swap_step"] for record in records], dtype=float)
    final_step = float(steps.max())
    progress = np.zeros(len(steps)) if final_step <= 0 else 100.0 * steps / final_step
    left = np.array([record["aggregation_left_neighbor_percent"] for record in records], dtype=float)
    right = np.array([record["aggregation_right_neighbor_legacy_percent"] for record in records], dtype=float)
    sortedness = np.array([record["sortedness_percent"] for record in records], dtype=float)
    interp_left = np.interp(grid, progress, left)
    interp_right = np.interp(grid, progress, right)
    interp_sortedness = np.interp(grid, progress, sortedness)
    expected_random = expected_random_left_neighbor_percent(counts)
    grid_rows = []
    for pct, left_value, right_value, sortedness_value in zip(grid, interp_left, interp_right, interp_sortedness, strict=True):
        grid_rows.append(
            {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "condition_id": f"S14F8_{mix}",
                "algotype_mix": mix,
                "is_negative_control": mix == "same_algorithm_bubble_label_control",
                "repeat_index": repeat_idx,
                "progress_percent": float(pct),
                "aggregation_left_neighbor_percent": float(left_value),
                "aggregation_right_neighbor_percent": float(right_value),
                "sortedness_percent": float(sortedness_value),
                "expected_random_left_neighbor_percent": float(expected_random),
                "final_swap_step": int(final_step),
            }
        )
    peak_idx = int(np.argmax(interp_left))
    return grid_rows, float(interp_left[peak_idx]), float(grid[peak_idx])


def run_aggregation_repeat(repo_dir: Path, mix: str, repeat_idx: int, limits: dict[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ensure_imports(repo_dir)
    from scripts.e01_s09_same_goal_chimeras import run_same_goal_chimera_sync

    values, value_seed = initial_values("aggregation", repeat_idx)
    assignments, assignment_seed, counts = assignments_for_mix(mix, repeat_idx)
    seed = scheduler_seed("aggregation", repeat_idx, mix)
    started = time.monotonic()
    result = run_same_goal_chimera_sync(
        repo_dir,
        values,
        assignments,
        seed,
        limits["max_successful_swaps"],
        limits["max_sweeps"],
        limits["stall_sweeps"],
    )
    elapsed = time.monotonic() - started
    grid_rows, run_peak, run_peak_progress = build_aggregation_grid(result["records"], mix, repeat_idx, counts)
    final_values = list(map(int, result["final_values"]))
    final_labels = list(map(str, result["final_labels"]))
    run_row = {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "task_family": "aggregation",
        "source_claim_id": "figure8_unique_aggregation_exact_peak_magnitudes",
        "condition_id": f"S14F8_{mix}",
        "mode": "cell_view",
        "algorithm": "mixed",
        "frozen_semantics": "none",
        "frozen_count": 0,
        "algotype_mix": mix,
        "repeat_index": repeat_idx,
        "sample_size_target": SCALED_REPEATS,
        "array_length": ARRAY_LENGTH,
        "initial_array_seed": value_seed,
        "scheduler_seed": seed,
        "frozen_index_seed": np.nan,
        "algotype_assignment_seed": assignment_seed,
        "initial_array_sha256": sha256_json(values),
        "initial_frozen_indices_json": "[]",
        "initial_algotype_counts_json": json.dumps(counts, separators=(",", ":")),
        "swap_only_steps": int(result["swap_count"]),
        "comparison_steps_observed": int(result["comparison_count"]),
        "compare_plus_swap_steps": int(result["compare_plus_swap_count"]),
        "final_sortedness_percent": sortedness_percent(final_values),
        "final_monotonicity_error_count": monotonicity_error_count(final_values),
        "final_aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(final_labels),
        "final_aggregation_right_neighbor_percent": aggregation_right_neighbor_percent(final_labels),
        "run_peak_aggregation_left_neighbor_percent": run_peak,
        "run_peak_left_neighbor_progress_percent": run_peak_progress,
        "sweep_count": int(result["sweep_count"]),
        "stop_reason": result["stop_reason"],
        "max_guard_hit": bool(result["max_guard_hit"]),
        "timed_out": False,
        "wrapper_name": "deterministic_single_thread_public_cell_methods_s03_assignments",
        "comparison_count_source": "public_status_probe_compare_and_swap_count",
        "elapsed_seconds": float(elapsed),
    }
    return run_row, grid_rows


def run_task_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    repo_dir = Path(payload["repo_dir"])
    checkpoint_path = Path(payload["checkpoint_path"])
    curve_checkpoint_path = Path(payload["curve_checkpoint_path"]) if payload.get("curve_checkpoint_path") else None
    expected_rows = int(payload["expected_rows"])
    force = bool(payload["force"])
    if not force and checkpoint_path.exists():
        try:
            existing = pd.read_parquet(checkpoint_path)
            if len(existing) == expected_rows:
                curve_ok = True
                if curve_checkpoint_path is not None:
                    curve_ok = curve_checkpoint_path.exists()
                if curve_ok:
                    return {
                        "checkpoint_path": str(checkpoint_path),
                        "curve_checkpoint_path": str(curve_checkpoint_path) if curve_checkpoint_path else None,
                        "run_rows": int(len(existing)),
                        "curve_rows": int(pd.read_parquet(curve_checkpoint_path).shape[0]) if curve_checkpoint_path else 0,
                        "skipped_existing": True,
                        "elapsed_seconds": 0.0,
                    }
        except Exception:
            pass
    limits = {
        "max_successful_swaps": int(payload["max_successful_swaps"]),
        "max_sweeps": int(payload["max_sweeps"]),
        "stall_sweeps": int(payload["stall_sweeps"]),
    }
    started = time.monotonic()
    run_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for repeat_idx in range(int(payload["repeat_start"]), int(payload["repeat_stop"])):
        if payload["task_family"] == "efficiency":
            run_rows.append(run_efficiency_repeat(repo_dir, payload["algorithm"], payload["mode"], repeat_idx, limits))
        elif payload["task_family"] == "frozen":
            run_rows.append(
                run_frozen_repeat(
                    repo_dir,
                    payload["algorithm"],
                    payload["mode"],
                    payload["frozen_semantics"],
                    int(payload["frozen_count"]),
                    repeat_idx,
                    limits,
                )
            )
        elif payload["task_family"] == "aggregation":
            row, grid = run_aggregation_repeat(repo_dir, payload["algotype_mix"], repeat_idx, limits)
            run_rows.append(row)
            curve_rows.extend(grid)
        else:
            raise ValueError(f"Unknown task family: {payload['task_family']}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_df = pd.DataFrame.from_records(run_rows)
    run_df.to_parquet(checkpoint_path, index=False)
    if curve_checkpoint_path is not None:
        pd.DataFrame.from_records(curve_rows).to_parquet(curve_checkpoint_path, index=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "curve_checkpoint_path": str(curve_checkpoint_path) if curve_checkpoint_path else None,
        "run_rows": int(len(run_rows)),
        "curve_rows": int(len(curve_rows)),
        "skipped_existing": False,
        "elapsed_seconds": time.monotonic() - started,
    }


def build_tasks(args: argparse.Namespace, repeats_to_run: int) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    tasks: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    checkpoint_dir = args.cache_dir / "checkpoints"
    limits = {
        "max_successful_swaps": args.max_successful_swaps,
        "max_sweeps": args.max_sweeps,
        "stall_sweeps": args.stall_sweeps,
    }
    condition_specs: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for mode in ("traditional", "cell_view"):
            condition_specs.append({"task_family": "efficiency", "algorithm": algorithm, "mode": mode})
    for semantics in FROZEN_SEMANTICS:
        for algorithm in ALGORITHMS:
            for mode in ("traditional", "cell_view"):
                condition_specs.append(
                    {
                        "task_family": "frozen",
                        "algorithm": algorithm,
                        "mode": mode,
                        "frozen_semantics": semantics,
                        "frozen_count": FROZEN_COUNT_SELECTED,
                    }
                )
    for mix in AGGREGATION_MIXES:
        condition_specs.append({"task_family": "aggregation", "algotype_mix": mix})

    for spec in condition_specs:
        condition_id = condition_id_for_task(spec)
        condition_rows.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "task_family": spec["task_family"],
                "algorithm": spec.get("algorithm", "mixed"),
                "mode": spec.get("mode", "cell_view"),
                "frozen_semantics": spec.get("frozen_semantics", "none"),
                "frozen_count": int(spec.get("frozen_count", 0)),
                "algotype_mix": spec.get("algotype_mix", "none"),
                "target_repeats": repeats_to_run,
                "source_claim_id": {
                    "efficiency": "figure4_exact_fold_change_magnitudes",
                    "frozen": "figure5_all_cell_view_lower_error_than_traditional",
                    "aggregation": "figure8_unique_aggregation_exact_peak_magnitudes",
                }[spec["task_family"]],
            }
        )
        for chunk_idx, repeat_start in enumerate(range(0, repeats_to_run, args.chunk_size)):
            repeat_stop = min(repeat_start + args.chunk_size, repeats_to_run)
            chunk_payload = {
                **spec,
                **limits,
                "repo_dir": str(args.repo_dir.resolve()),
                "repeat_start": repeat_start,
                "repeat_stop": repeat_stop,
                "expected_rows": repeat_stop - repeat_start,
                "checkpoint_path": str(checkpoint_dir / spec["task_family"] / condition_id / f"chunk_{chunk_idx:04d}_runs.parquet"),
                "curve_checkpoint_path": None,
                "force": args.force,
            }
            if spec["task_family"] == "aggregation":
                chunk_payload["curve_checkpoint_path"] = str(checkpoint_dir / spec["task_family"] / condition_id / f"chunk_{chunk_idx:04d}_curves.parquet")
            tasks.append(chunk_payload)
    return tasks, pd.DataFrame.from_records(condition_rows)


def execute_tasks(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_task_chunk, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if idx % 25 == 0 or idx == len(futures):
                print(f"S14 completed {idx}/{len(futures)} checkpoint chunks", flush=True)
    return results


def load_checkpoint_outputs(results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_frames = [pd.read_parquet(result["checkpoint_path"]) for result in results]
    run_df = pd.concat(run_frames, ignore_index=True)
    curve_paths = [result["curve_checkpoint_path"] for result in results if result.get("curve_checkpoint_path")]
    if curve_paths:
        curve_df = pd.concat([pd.read_parquet(path) for path in curve_paths], ignore_index=True)
    else:
        curve_df = pd.DataFrame()
    run_df.sort_values(["task_family", "condition_id", "repeat_index"], inplace=True)
    if not curve_df.empty:
        curve_df.sort_values(["condition_id", "repeat_index", "progress_percent"], inplace=True)
    return run_df.reset_index(drop=True), curve_df.reset_index(drop=True)


def bootstrap_paired(values_a: np.ndarray, values_b: np.ndarray, func: str, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values_a)
    idx = rng.integers(0, n, size=(reps, n))
    a = values_a[idx]
    b = values_b[idx]
    if func == "diff":
        draws = (b - a).mean(axis=1)
    elif func == "ratio":
        draws = b.mean(axis=1) / a.mean(axis=1)
    else:
        raise ValueError(func)
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def bootstrap_mean(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    draws = values[idx].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def summarize_efficiency(run_df: pd.DataFrame, s05: pd.DataFrame, bootstrap_reps: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = run_df[run_df["task_family"] == "efficiency"].copy()
    s05_lookup = {
        (str(row.algorithm), str(row.count_metric)): row
        for row in s05[s05["included_in_figure4"] == True].itertuples()  # noqa: E712
    }
    for algorithm in ALGORITHMS:
        trad = subset[(subset["algorithm"] == algorithm) & (subset["mode"] == "traditional")].sort_values("repeat_index")
        cell = subset[(subset["algorithm"] == algorithm) & (subset["mode"] == "cell_view")].sort_values("repeat_index")
        for metric in EFFICIENCY_METRICS:
            trad_values = trad[metric].to_numpy(dtype=float)
            cell_values = cell[metric].to_numpy(dtype=float)
            ratio = float(cell_values.mean() / trad_values.mean())
            low, high = bootstrap_paired(trad_values, cell_values, "ratio", bootstrap_reps, stable_seed("efficiency", algorithm, metric))
            s05_row = s05_lookup[(algorithm, metric)]
            paper_ratio = float(PAPER_RATIO_TARGETS[(algorithm, metric)])
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "source_claim_id": "figure4_exact_fold_change_magnitudes",
                    "claim_family": "figure4_efficiency",
                    "condition_key": f"{algorithm}_{metric}",
                    "algorithm": algorithm,
                    "mode_or_mix": "cell_view_over_traditional",
                    "metric": metric,
                    "scaled_n": int(len(trad_values)),
                    "scaled_mean_traditional": float(trad_values.mean()),
                    "scaled_mean_cell_view": float(cell_values.mean()),
                    "scaled_effect": ratio,
                    "scaled_ci95_low": float(low),
                    "scaled_ci95_high": float(high),
                    "s04_s13_reference_effect": float(s05_row.ratio_of_means_cell_over_traditional),
                    "paper_reference_effect": paper_ratio,
                    "scaled_minus_paper": float(ratio - paper_ratio),
                    "reference_minus_paper": float(s05_row.ratio_of_means_cell_over_traditional - paper_ratio),
                    "paper_inside_scaled_ci95": bool(low <= paper_ratio <= high),
                    "s04_reference_inside_scaled_ci95": bool(low <= float(s05_row.ratio_of_means_cell_over_traditional) <= high),
                    "stability_result": "paper_magnitude_not_supported" if not (low <= paper_ratio <= high) else "paper_magnitude_within_ci",
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_frozen(run_df: pd.DataFrame, s07: pd.DataFrame, bootstrap_reps: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = run_df[run_df["task_family"] == "frozen_cell"].copy()
    s07_lookup = {
        (str(row.algorithm), str(row.frozen_semantics), int(row.frozen_count)): row
        for row in s07.itertuples()
    }
    for semantics in FROZEN_SEMANTICS:
        for algorithm in ALGORITHMS:
            trad = subset[
                (subset["algorithm"] == algorithm)
                & (subset["mode"] == "traditional")
                & (subset["frozen_semantics"] == semantics)
            ].sort_values("repeat_index")
            cell = subset[
                (subset["algorithm"] == algorithm)
                & (subset["mode"] == "cell_view")
                & (subset["frozen_semantics"] == semantics)
            ].sort_values("repeat_index")
            trad_values = trad["final_monotonicity_error_count"].to_numpy(dtype=float)
            cell_values = cell["final_monotonicity_error_count"].to_numpy(dtype=float)
            diff = float(cell_values.mean() - trad_values.mean())
            low, high = bootstrap_paired(trad_values, cell_values, "diff", bootstrap_reps, stable_seed("frozen", algorithm, semantics))
            s07_row = s07_lookup[(algorithm, semantics, FROZEN_COUNT_SELECTED)]
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "source_claim_id": "figure5_all_cell_view_lower_error_than_traditional",
                    "claim_family": "figure5_frozen_cell",
                    "condition_key": f"{algorithm}_{semantics}_f{FROZEN_COUNT_SELECTED}",
                    "algorithm": algorithm,
                    "mode_or_mix": semantics,
                    "metric": "cell_minus_traditional_final_monotonicity_error",
                    "scaled_n": int(len(trad_values)),
                    "scaled_mean_traditional": float(trad_values.mean()),
                    "scaled_mean_cell_view": float(cell_values.mean()),
                    "scaled_effect": diff,
                    "scaled_ci95_low": float(low),
                    "scaled_ci95_high": float(high),
                    "s04_s13_reference_effect": float(s07_row.cell_minus_traditional_mean_error),
                    "paper_reference_effect": 0.0,
                    "scaled_minus_paper": diff,
                    "reference_minus_paper": float(s07_row.cell_minus_traditional_mean_error),
                    "paper_inside_scaled_ci95": bool(low <= 0.0 <= high),
                    "s04_reference_inside_scaled_ci95": bool(low <= float(s07_row.cell_minus_traditional_mean_error) <= high),
                    "stability_result": "cell_lower_supported" if high < 0.0 else ("cell_higher_supported" if low > 0.0 else "not_cell_lower_or_tied"),
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_curves(curve_df: pd.DataFrame) -> pd.DataFrame:
    grouped = curve_df.groupby(["condition_id", "algotype_mix", "is_negative_control", "progress_percent"], as_index=False)
    curves = grouped.agg(
        n_runs=("repeat_index", "nunique"),
        mean_aggregation_left_neighbor_percent=("aggregation_left_neighbor_percent", "mean"),
        std_aggregation_left_neighbor_percent=("aggregation_left_neighbor_percent", "std"),
        mean_aggregation_right_neighbor_percent=("aggregation_right_neighbor_percent", "mean"),
        std_aggregation_right_neighbor_percent=("aggregation_right_neighbor_percent", "std"),
        mean_sortedness_percent=("sortedness_percent", "mean"),
        expected_random_left_neighbor_percent=("expected_random_left_neighbor_percent", "mean"),
        mean_final_swap_step=("final_swap_step", "mean"),
    )
    for prefix in ("left_neighbor", "right_neighbor"):
        mean_col = f"mean_aggregation_{prefix}_percent"
        std_col = f"std_aggregation_{prefix}_percent"
        curves[f"sem_aggregation_{prefix}_percent"] = curves[std_col] / np.sqrt(curves["n_runs"])
        curves[f"ci95_low_aggregation_{prefix}_percent"] = curves[mean_col] - 1.96 * curves[f"sem_aggregation_{prefix}_percent"]
        curves[f"ci95_high_aggregation_{prefix}_percent"] = curves[mean_col] + 1.96 * curves[f"sem_aggregation_{prefix}_percent"]
    return curves.sort_values(["condition_id", "progress_percent"]).reset_index(drop=True)


def build_aggregation_peak_table(curves: pd.DataFrame, s10: pd.DataFrame) -> pd.DataFrame:
    control_curve = curves[curves["algotype_mix"] == "same_algorithm_bubble_label_control"].set_index("progress_percent")
    control_peak = float(control_curve["mean_aggregation_left_neighbor_percent"].max())
    s10_lookup = {str(row.algotype_mix): row for row in s10.itertuples()}
    rows: list[dict[str, Any]] = []
    for condition_id, subset in curves.groupby("condition_id", sort=True):
        subset = subset.sort_values("progress_percent")
        left_peak = subset.loc[subset["mean_aggregation_left_neighbor_percent"].idxmax()]
        first = subset.iloc[0]
        mix = str(first["algotype_mix"])
        progress = float(left_peak["progress_percent"])
        paper_peak = PAPER_AGGREGATION_PEAKS.get(mix, np.nan)
        s10_peak = float(s10_lookup[mix].mean_curve_peak_aggregation_left_neighbor_percent)
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_claim_id": "figure8_unique_aggregation_exact_peak_magnitudes",
                "claim_family": "figure8_aggregation",
                "condition_key": mix,
                "algorithm": "mixed",
                "mode_or_mix": mix,
                "metric": "mean_curve_peak_aggregation_left_neighbor_percent",
                "scaled_n": int(first["n_runs"]),
                "scaled_mean_traditional": np.nan,
                "scaled_mean_cell_view": np.nan,
                "scaled_effect": float(left_peak["mean_aggregation_left_neighbor_percent"]),
                "scaled_ci95_low": float(left_peak["ci95_low_aggregation_left_neighbor_percent"]),
                "scaled_ci95_high": float(left_peak["ci95_high_aggregation_left_neighbor_percent"]),
                "s04_s13_reference_effect": s10_peak,
                "paper_reference_effect": float(paper_peak) if not pd.isna(paper_peak) else np.nan,
                "scaled_minus_paper": float(left_peak["mean_aggregation_left_neighbor_percent"] - paper_peak) if not pd.isna(paper_peak) else np.nan,
                "reference_minus_paper": float(s10_peak - paper_peak) if not pd.isna(paper_peak) else np.nan,
                "paper_inside_scaled_ci95": bool(left_peak["ci95_low_aggregation_left_neighbor_percent"] <= paper_peak <= left_peak["ci95_high_aggregation_left_neighbor_percent"]) if not pd.isna(paper_peak) else False,
                "s04_reference_inside_scaled_ci95": bool(left_peak["ci95_low_aggregation_left_neighbor_percent"] <= s10_peak <= left_peak["ci95_high_aggregation_left_neighbor_percent"]),
                "stability_result": "above_control_and_chance" if (mix != "same_algorithm_bubble_label_control" and left_peak["mean_aggregation_left_neighbor_percent"] > control_peak and left_peak["mean_aggregation_left_neighbor_percent"] > first["expected_random_left_neighbor_percent"]) else "control_or_not_above_baseline",
                "is_negative_control": bool(first["is_negative_control"]),
                "display_label": MIX_DISPLAY.get(mix, mix),
                "expected_random_left_neighbor_percent": float(first["expected_random_left_neighbor_percent"]),
                "negative_control_peak_left_neighbor_percent": control_peak,
                "mean_curve_peak_left_neighbor_progress_percent": progress,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["is_negative_control", "condition_key"]).reset_index(drop=True)


def build_noise_estimates(run_df: pd.DataFrame, curve_df: pd.DataFrame, s05: pd.DataFrame, s07: pd.DataFrame, s10: pd.DataFrame, bootstrap_reps: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eff = summarize_efficiency(run_df, s05, bootstrap_reps)
    frozen = summarize_frozen(run_df, s07, bootstrap_reps)
    curves = summarize_curves(curve_df)
    agg = build_aggregation_peak_table(curves, s10)
    shared_cols = [
        "research_step_id",
        "source_claim_id",
        "claim_family",
        "condition_key",
        "algorithm",
        "mode_or_mix",
        "metric",
        "scaled_n",
        "scaled_mean_traditional",
        "scaled_mean_cell_view",
        "scaled_effect",
        "scaled_ci95_low",
        "scaled_ci95_high",
        "s04_s13_reference_effect",
        "paper_reference_effect",
        "scaled_minus_paper",
        "reference_minus_paper",
        "paper_inside_scaled_ci95",
        "s04_reference_inside_scaled_ci95",
        "stability_result",
    ]
    noise = pd.concat([eff[shared_cols], frozen[shared_cols], agg[shared_cols]], ignore_index=True)
    return noise, curves, agg


def validate_outputs(
    run_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    noise: pd.DataFrame,
    condition_df: pd.DataFrame,
    checkpoint_results: list[dict[str, Any]],
    repeats_to_run: int,
    workers: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_conditions = 23
    expected_run_rows = expected_conditions * repeats_to_run
    if len(condition_df) != expected_conditions:
        errors.append(f"Expected {expected_conditions} selected conditions; found {len(condition_df)}.")
    if len(run_df) != expected_run_rows:
        errors.append(f"Expected {expected_run_rows} run rows; found {len(run_df)}.")
    repeat_counts = run_df.groupby("condition_id")["repeat_index"].nunique().to_dict()
    repeat_counts_ok = all(int(value) == repeats_to_run for value in repeat_counts.values()) and len(repeat_counts) == expected_conditions
    if not repeat_counts_ok:
        errors.append(f"Repeat counts are incomplete: {repeat_counts}.")
    count_fields_complete = bool(
        run_df[["swap_only_steps", "comparison_steps_observed", "compare_plus_swap_steps"]].notna().all(axis=1).all()
        and (run_df["swap_only_steps"] + run_df["comparison_steps_observed"] == run_df["compare_plus_swap_steps"]).all()
    )
    if not count_fields_complete:
        errors.append("Count fields are incomplete or not additive.")
    final_metrics_complete = bool(
        run_df[["final_sortedness_percent", "final_monotonicity_error_count", "stop_reason", "max_guard_hit"]].notna().all(axis=1).all()
    )
    if not final_metrics_complete:
        errors.append("Final metric or stopping fields are incomplete.")
    no_max_guard = bool(~run_df["max_guard_hit"].astype(bool).any())
    if not no_max_guard:
        warnings.append("At least one run hit a max guard.")
    efficiency = run_df[run_df["task_family"] == "efficiency"].copy()
    matched_eff = True
    for algorithm in ALGORITHMS:
        rows = [efficiency[(efficiency["algorithm"] == algorithm) & (efficiency["mode"] == mode)].sort_values("repeat_index") for mode in ("traditional", "cell_view")]
        matched_eff = matched_eff and rows[0]["initial_array_sha256"].tolist() == rows[1]["initial_array_sha256"].tolist()
    if not matched_eff:
        errors.append("Efficiency traditional/cell-view initial arrays are not matched by repeat.")
    frozen = run_df[run_df["task_family"] == "frozen_cell"].copy()
    matched_frozen = True
    for semantics in FROZEN_SEMANTICS:
        for algorithm in ALGORITHMS:
            rows = [
                frozen[
                    (frozen["algorithm"] == algorithm)
                    & (frozen["mode"] == mode)
                    & (frozen["frozen_semantics"] == semantics)
                ].sort_values("repeat_index")
                for mode in ("traditional", "cell_view")
            ]
            matched_frozen = matched_frozen and rows[0]["initial_array_sha256"].tolist() == rows[1]["initial_array_sha256"].tolist()
            matched_frozen = matched_frozen and rows[0]["initial_frozen_indices_json"].tolist() == rows[1]["initial_frozen_indices_json"].tolist()
    if not matched_frozen:
        errors.append("Frozen traditional/cell-view initial arrays or frozen indices are not matched by repeat.")
    aggregation = run_df[run_df["task_family"] == "aggregation"].copy()
    all_aggregation_sorted = bool((aggregation["final_sortedness_percent"] == 100.0).all())
    if not all_aggregation_sorted:
        errors.append("At least one scaled aggregation run did not finish sorted.")
    assignment_counts_ok = True
    for row in aggregation.itertuples():
        counts = json.loads(row.initial_algotype_counts_json)
        total = sum(int(value) for value in counts.values())
        assignment_counts_ok = assignment_counts_ok and total == ARRAY_LENGTH
    if not assignment_counts_ok:
        errors.append("Aggregation assignment counts are invalid.")
    expected_curve_rows = len(AGGREGATION_MIXES) * repeats_to_run * 101
    if len(curve_df) != expected_curve_rows:
        errors.append(f"Expected {expected_curve_rows} aggregation run-grid rows; found {len(curve_df)}.")
    required_noise_claims = {
        "figure4_exact_fold_change_magnitudes",
        "figure5_all_cell_view_lower_error_than_traditional",
        "figure8_unique_aggregation_exact_peak_magnitudes",
    }
    observed_noise_claims = set(noise["source_claim_id"].unique())
    if observed_noise_claims != required_noise_claims:
        errors.append(f"Noise table source claims mismatch: {observed_noise_claims}.")
    checkpoint_paths = [Path(result["checkpoint_path"]) for result in checkpoint_results]
    checkpoint_paths.extend(Path(result["curve_checkpoint_path"]) for result in checkpoint_results if result.get("curve_checkpoint_path"))
    checkpoints_exist = all(path.exists() for path in checkpoint_paths)
    if not checkpoints_exist:
        errors.append("One or more checkpoint files are missing.")
    success = not errors
    persisted_nonrep = {
        "figure4_paper_magnitudes_supported": bool(noise[(noise["claim_family"] == "figure4_efficiency")]["paper_inside_scaled_ci95"].all()),
        "figure5_all_selected_cell_lower": bool((noise[(noise["claim_family"] == "figure5_frozen_cell")]["scaled_ci95_high"] < 0.0).all()),
        "figure8_paper_peaks_supported": bool(noise[(noise["claim_family"] == "figure8_aggregation") & (noise["condition_key"] != "same_algorithm_bubble_label_control")]["paper_inside_scaled_ci95"].all()),
    }
    caveats = [
        "S14 uses N=1000 independent seed banks generated from S14 base seed 2026063014; these are not S03's original 100 repeats.",
        "S14 preserves S03 array length, value conventions, Frozen Cell semantics, f=3 selected perturbation strength, and same-goal Algotype proportions.",
        "S04's public threaded cell-view runner was not scaled to N=1000 because S04 measured 1,767 seconds for only 600 runs and insertion alone averaged 14.5 seconds per run; S14 uses the deterministic public-method wrapper already used for S07/S09/S11/S12 as a scalable sensitivity.",
        "Comparison counts retain the S05 actionable-comparison proxy caveat.",
        "N=10000 was not run for wrapper-heavy selected conditions in this bounded step; checkpointing makes a later extension restartable.",
    ]
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed_with_caveats" if success else "completed_validation_failed",
        "validationResult": "passed_scaled_selected_conditions_with_caveats" if success else "failed",
        "outcomeClassification": "constraining/contradictory" if success else "null",
        "selectedConditionCount": int(len(condition_df)),
        "targetRepeatsPerCondition": int(repeats_to_run),
        "runRows": int(len(run_df)),
        "expectedRunRows": int(expected_run_rows),
        "curveRows": int(len(curve_df)),
        "expectedCurveRows": int(expected_curve_rows),
        "noiseRows": int(len(noise)),
        "workers": int(workers),
        "checkpointChunkCount": int(len(checkpoint_results)),
        "checkpointPathsExist": checkpoints_exist,
        "skippedExistingCheckpointChunks": int(sum(bool(result.get("skipped_existing")) for result in checkpoint_results)),
        "repeatCountsByCondition": {str(key): int(value) for key, value in repeat_counts.items()},
        "repeatCountsOk": repeat_counts_ok,
        "countFieldsCompleteAndAdditive": count_fields_complete,
        "finalMetricsComplete": final_metrics_complete,
        "noMaxGuardRuns": no_max_guard,
        "efficiencyMatchedInitialArrays": matched_eff,
        "frozenMatchedInitialArraysAndIndices": matched_frozen,
        "aggregationAssignmentCountsValid": assignment_counts_ok,
        "aggregationAllRunsSorted": all_aggregation_sorted,
        "requiredNoiseClaimsCovered": sorted(observed_noise_claims),
        "persistedNonReplicationSummary": persisted_nonrep,
        "validationErrors": errors,
        "validationWarnings": warnings,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Proceed to S15 package the baseline, carrying forward the S13/S14 non-replication and magnitude-divergence caveats; do not start S15 inside S14.",
    }


def plot_summary(noise: pd.DataFrame, agg_curves: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    fig.patch.set_facecolor("white")

    eff = noise[noise["claim_family"] == "figure4_efficiency"].copy()
    eff["label"] = eff["algorithm"].str.title() + "\n" + eff["metric"].map({"swap_only_steps": "swaps", "compare_plus_swap_steps": "swaps+comparisons"})
    x = np.arange(len(eff))
    axes[0, 0].errorbar(
        x,
        eff["scaled_effect"],
        yerr=[eff["scaled_effect"] - eff["scaled_ci95_low"], eff["scaled_ci95_high"] - eff["scaled_effect"]],
        fmt="o",
        color="#305f72",
        ecolor="#8bb4c4",
        capsize=3,
        label="N=1000 scaled",
    )
    axes[0, 0].scatter(x, eff["paper_reference_effect"], marker="x", color="#b34d4d", label="paper ratio")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(eff["label"], rotation=35, ha="right")
    axes[0, 0].set_ylabel("Cell-view / traditional ratio")
    axes[0, 0].set_title("Figure 4 count ratios")
    axes[0, 0].grid(True, axis="y", alpha=0.35)
    axes[0, 0].legend(frameon=False, fontsize=8)

    frozen = noise[noise["claim_family"] == "figure5_frozen_cell"].copy()
    frozen["label"] = frozen["algorithm"].str.title() + "\n" + frozen["mode_or_mix"]
    x = np.arange(len(frozen))
    axes[0, 1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[0, 1].errorbar(
        x,
        frozen["scaled_effect"],
        yerr=[frozen["scaled_effect"] - frozen["scaled_ci95_low"], frozen["scaled_ci95_high"] - frozen["scaled_effect"]],
        fmt="o",
        color="#7a5c36",
        ecolor="#c6a76b",
        capsize=3,
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(frozen["label"], rotation=35, ha="right")
    axes[0, 1].set_ylabel("Cell-view minus traditional final error")
    axes[0, 1].set_title("Figure 5 f=3 Frozen Cell differences")
    axes[0, 1].grid(True, axis="y", alpha=0.35)

    agg = noise[noise["claim_family"] == "figure8_aggregation"].copy()
    agg = agg[agg["condition_key"] != "same_algorithm_bubble_label_control"].copy()
    x = np.arange(len(agg))
    axes[1, 0].errorbar(
        x,
        agg["scaled_effect"],
        yerr=[agg["scaled_effect"] - agg["scaled_ci95_low"], agg["scaled_ci95_high"] - agg["scaled_effect"]],
        fmt="o",
        color="#4e6b4a",
        ecolor="#a7c49e",
        capsize=3,
        label="N=1000 peak",
    )
    axes[1, 0].scatter(x, agg["paper_reference_effect"], marker="x", color="#b34d4d", label="paper peak")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([MIX_DISPLAY.get(v, v) for v in agg["condition_key"]], rotation=25, ha="right")
    axes[1, 0].set_ylabel("Peak Aggregation (%)")
    axes[1, 0].set_title("Figure 8 Aggregation peaks")
    axes[1, 0].grid(True, axis="y", alpha=0.35)
    axes[1, 0].legend(frameon=False, fontsize=8)

    for mix in AGGREGATION_MIXES:
        subset = agg_curves[agg_curves["algotype_mix"] == mix].sort_values("progress_percent")
        if subset.empty:
            continue
        axes[1, 1].plot(
            subset["progress_percent"],
            subset["mean_aggregation_left_neighbor_percent"],
            color=MIX_COLORS.get(mix, "#555555"),
            linestyle="--" if mix == "same_algorithm_bubble_label_control" else "-",
            linewidth=2.0,
            label=MIX_DISPLAY.get(mix, mix),
        )
    axes[1, 1].set_title("N=1000 Aggregation curves")
    axes[1, 1].set_xlabel("Run progress (% final swaps)")
    axes[1, 1].set_ylabel("Aggregation (%)")
    axes[1, 1].grid(True, alpha=0.35)
    axes[1, 1].legend(frameon=False, fontsize=7)

    fig.suptitle("E01 S14 scaled uncertainty summary", fontsize=15)
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


def write_artifact_manifest(path: Path, artifact_paths: dict[str, Path], validation: dict[str, Any], repo_dir: Path) -> None:
    payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "validationResult": validation["validationResult"],
        "artifacts": [
            {
                "label": label,
                "path": str(artifact_path),
                "sha256": None if artifact_path == path else (sha256_file(artifact_path) if artifact_path.exists() else None),
                "sizeBytes": None if artifact_path == path else (artifact_path.stat().st_size if artifact_path.exists() else None),
            }
            for label, artifact_path in artifact_paths.items()
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_run_manifest(artifacts_dir: Path, artifact_paths: dict[str, Path], validation: dict[str, Any], repo_dir: Path, command: list[str]) -> None:
    manifest_path = artifacts_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest["researchStepId"] = STEP_ID
    manifest["updatedAtUtc"] = utc_now()
    manifest.setdefault("artifacts", {}).update({label: str(path) for label, path in artifact_paths.items()})
    manifest.setdefault("checksums", {})
    for path in artifact_paths.values():
        if path.exists():
            manifest["checksums"][str(path)] = sha256_file(path)
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": "Scale the replication",
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
        "artifactsWritten": [str(path) for path in artifact_paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s14_scaled_replication.py"),
        "command": " ".join(command),
        "summary": validation["persistedNonReplicationSummary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    validation: dict[str, Any],
    artifact_paths: dict[str, Path],
    input_paths: dict[str, Path],
    noise: pd.DataFrame,
    selected_conditions: pd.DataFrame,
    checkpoint_manifest: dict[str, Any],
    command: list[str],
    repo_dir: Path,
    wall_seconds: float,
) -> None:
    lay_summary = (
        "S14 ran independent N=1000 seed sweeps for the S13 high-priority divergences. "
        "The larger sample supported stable qualitative behavior but did not rescue the exact Figure 4 count magnitudes, "
        "the broad Figure 5 all-cell-view superiority claim, or the exact Figure 8 unique Aggregation peak magnitudes under the frozen public-code definitions."
    )
    compact = noise[
        [
            "source_claim_id",
            "condition_key",
            "metric",
            "scaled_effect",
            "scaled_ci95_low",
            "scaled_ci95_high",
            "paper_reference_effect",
            "paper_inside_scaled_ci95",
            "stability_result",
        ]
    ].copy()
    report = f"""# E01 S14 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: {validation["status"]}
- Artifacts written: {", ".join(str(path) for path in artifact_paths.values())}
- Validation result: {validation["validationResult"]}
- Outcome classification: {validation["outcomeClassification"]}
- Caveats or blockers: {"; ".join(validation["caveatsOrBlockers"])}
- Lay summary: {lay_summary}
- Recommended next action: {validation["recommendedNextAction"]}

## Frozen Question

At larger sample sizes, which original core effects are stable, and which appear noisy under the paper's N equals 100 design?

## Inputs

{chr(10).join(f"- `{label}`: `{path}` (`{sha256_file(path)}`)" for label, path in input_paths.items() if path.exists())}

## Detailed Methods

S14 selected high-value conditions directly from the S13 replication-status table, prioritizing the three not-replicated or magnitude-divergent claim families:

- Figure 4 exact count fold-change magnitudes.
- Figure 5 broad claim that all cell-view Frozen Cell runs have lower monotonicity error than traditional counterparts.
- Figure 8 exact unique-value Aggregation peak magnitudes.

The scaled production sample used `{validation["targetRepeatsPerCondition"]}` independent repeats per selected condition, generated from S14 base seed `{BASE_SEED}`. S14 kept the frozen E01 settings that matter for interpretation: array length `{ARRAY_LENGTH}`, values 1..100, f=3 selected Frozen Cell stress, passive/stuck semantics, same-goal Algotype proportions, deterministic public-method wrappers for wrapper-heavy conditions, and the S05 comparison-count convention.

Checkpointing used chunk-level Parquet files under `/cache/e01_s14/checkpoints/`. Each chunk covers one selected condition and a contiguous repeat interval. Existing complete checkpoints are reused unless `--force` is passed. The artifact checkpoint manifest records chunk counts and cache paths; bulk checkpoint files are intentionally not copied into `/artifacts`.

N=10000 was not run in this bounded step. The S04 public threaded runner required 1,767 seconds for the original 600 runs and insertion cell-view averaged about 14.5 seconds per run, so scaling that exact threaded path to N=1000 or N=10000 would be disproportionate. The deterministic public-method wrapper was used as a scalable sensitivity for Figure 4 and as the already frozen method for S07/S09-style conditions.

## Commands

```bash
{" ".join(command)}
```

Validation command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/e01_s14_scaled_replication.py
```

## Dependencies And Parameters

- Python executable: `{sys.executable}`
- Python version: `{sys.version.splitlines()[0]}`
- Platform: `{platform.platform()}`
- Pandas version: `{pd.__version__}`
- NumPy version: `{np.__version__}`
- Workers requested/used: `{validation["workers"]}`.
- Chunk count: `{validation["checkpointChunkCount"]}`.
- Repository commit at run time: `{git_commit(repo_dir)}`
- Repository status during report generation: `{git_status(repo_dir) or "clean"}`
- Wall time: `{wall_seconds:.3f}` seconds

No new Python, system, R, Rust, or Node dependencies were installed for S14.

## Selected Conditions

{dataframe_to_markdown(selected_conditions)}

## Results

Noise and stability estimates:

{dataframe_to_markdown(compact)}

Persistence summary:

```json
{json.dumps(validation["persistedNonReplicationSummary"], indent=2)}
```

Interpretation:

- Figure 4: scaled ratios remain precise under the scalable public-method sensitivity, but the paper's exact fold-change magnitudes are not generally inside the N=1000 confidence intervals.
- Figure 5: selected f=3 Frozen Cell comparisons do not support the broad claim that every cell-view condition has lower final monotonicity error than the reconstructed traditional counterpart.
- Figure 8: all core unique-value Aggregation curves remain above control/chance, but exact paper peak magnitudes remain outside the N=1000 intervals for multiple mixes.

## Validation

```json
{json.dumps(validation, indent=2)}
```

Validation highlights:

- Run rows: `{validation["runRows"]}` of expected `{validation["expectedRunRows"]}`.
- Aggregation curve rows: `{validation["curveRows"]}` of expected `{validation["expectedCurveRows"]}`.
- Repeat counts complete: `{validation["repeatCountsOk"]}`.
- Count fields complete and additive: `{validation["countFieldsCompleteAndAdditive"]}`.
- Efficiency matched initial arrays: `{validation["efficiencyMatchedInitialArrays"]}`.
- Frozen matched initial arrays and indices: `{validation["frozenMatchedInitialArraysAndIndices"]}`.
- Aggregation assignment counts valid: `{validation["aggregationAssignmentCountsValid"]}`.
- No max-guard runs: `{validation["noMaxGuardRuns"]}`.

## Artifacts And Provenance

{chr(10).join(f"- `{label}`: `{path}`" for label, path in artifact_paths.items())}

Checkpoint manifest summary:

```json
{json.dumps({key: checkpoint_manifest[key] for key in ["checkpointRoot", "chunkCount", "skippedExistingChunks", "workerCount"]}, indent=2)}
```

The global run manifest and checksum file were updated after artifact creation. The S14 artifact manifest records paths, sizes, and SHA256 hashes.

## Caveats, Blockers, Failed Assumptions, And Limitations

- S14 is a scaled sensitivity over selected high-value conditions, not an exhaustive N=1000 rerun of every S04-S12 condition.
- S14 does not change frozen count definitions, duplicate semantics, Aggregation denominator, or comparison-count definitions.
- S14 does not resolve the absence of public traditional runners or original publication seeds.
- N=10000 remains a follow-up extension for optimized or narrowed conditions rather than a completed S14 production run.
- Computational results remain sorting-array proxy evidence, not biological validation.

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    artifacts_dir = args.artifacts_dir.resolve()
    repo_dir = args.repo_dir.resolve()
    s14_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e01"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (s14_dir, results_dir, tables_dir, figures_dir, checksums_dir, args.cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "research_plan": args.research_plan_path.resolve(),
        "baseline_config": args.config_path.resolve(),
        "s13_status_table": args.s13_status_path.resolve(),
        "s05_efficiency_numeric_table": args.s05_efficiency_table_path.resolve(),
        "s07_frozen_comparison_table": args.s07_frozen_table_path.resolve(),
        "s10_aggregation_peak_table": args.s10_aggregation_table_path.resolve(),
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing S14 inputs: " + "; ".join(missing))
    repeats_to_run = min(args.max_repeats or SCALED_REPEATS, SCALED_REPEATS)
    workers = min(8, max(1, int(args.workers)))

    s13_status = pd.read_csv(args.s13_status_path)
    required_claims = {
        "figure4_exact_fold_change_magnitudes",
        "figure5_all_cell_view_lower_error_than_traditional",
        "figure8_unique_aggregation_exact_peak_magnitudes",
    }
    missing_claims = sorted(required_claims - set(s13_status["claim_id"]))
    if missing_claims:
        raise SystemExit(f"S13 status table lacks required selected claims: {missing_claims}")

    tasks, selected_conditions = build_tasks(args, repeats_to_run)
    task_results = execute_tasks(tasks, workers)
    run_df, curve_grid = load_checkpoint_outputs(task_results)
    s05 = pd.read_csv(args.s05_efficiency_table_path)
    s07 = pd.read_csv(args.s07_frozen_table_path)
    s10 = pd.read_csv(args.s10_aggregation_table_path)
    noise, aggregation_curves, aggregation_peaks = build_noise_estimates(run_df, curve_grid, s05, s07, s10, args.bootstrap_reps)

    scaled_result_path = results_dir / "e01_scaled_replication.parquet"
    curve_path = results_dir / "e01_scaled_aggregation_curves.parquet"
    aggregation_peak_path = tables_dir / "e01_scaled_aggregation_peak_table.csv"
    noise_path = tables_dir / "e01_noise_estimates.csv"
    selected_conditions_path = s14_dir / "e01_s14_selected_conditions.csv"
    figure_path = figures_dir / "scaled_uncertainty_summary.png"
    validation_path = s14_dir / "s14_validation.json"
    checkpoint_manifest_path = s14_dir / "checkpoint_manifest.json"
    manifest_path = s14_dir / "artifact_manifest.json"
    report_path = s14_dir / "research_step_full_results.md"

    run_df.to_parquet(scaled_result_path, index=False)
    aggregation_curves.to_parquet(curve_path, index=False)
    aggregation_peaks.to_csv(aggregation_peak_path, index=False)
    noise.to_csv(noise_path, index=False)
    selected_conditions.to_csv(selected_conditions_path, index=False)
    plot_summary(noise, aggregation_curves, figure_path)

    checkpoint_manifest = {
        "researchStepId": STEP_ID,
        "createdAtUtc": utc_now(),
        "checkpointRoot": str((args.cache_dir / "checkpoints").resolve()),
        "chunkCount": len(task_results),
        "skippedExistingChunks": int(sum(bool(result.get("skipped_existing")) for result in task_results)),
        "workerCount": workers,
        "chunkSize": int(args.chunk_size),
        "targetRepeatsPerCondition": int(repeats_to_run),
        "checkpointResults": task_results,
    }
    checkpoint_manifest_path.write_text(json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = validate_outputs(run_df, curve_grid, noise, selected_conditions, task_results, repeats_to_run, workers)
    validation.update(
        {
            "createdAtUtc": utc_now(),
            "repositoryCommitAtRunTime": git_commit(repo_dir),
            "script": str(repo_dir / "scripts/e01_s14_scaled_replication.py"),
            "inputArtifactHashes": {label: sha256_file(path) for label, path in input_paths.items()},
            "threadEnvironment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
        }
    )

    artifact_paths = {
        "full_results_report": report_path,
        "scaled_replication_parquet": scaled_result_path,
        "scaled_aggregation_curves_parquet": curve_path,
        "scaled_uncertainty_figure": figure_path,
        "noise_estimates_csv": noise_path,
        "scaled_aggregation_peak_table_csv": aggregation_peak_path,
        "selected_conditions_csv": selected_conditions_path,
        "validation_json": validation_path,
        "checkpoint_manifest_json": checkpoint_manifest_path,
        "artifact_manifest": manifest_path,
    }
    validation["artifactsWritten"] = [str(path) for path in artifact_paths.values()]
    validation["wallSeconds"] = time.monotonic() - started
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, validation, artifact_paths, input_paths, noise, selected_conditions, checkpoint_manifest, sys.argv, repo_dir, validation["wallSeconds"])
    write_artifact_manifest(manifest_path, artifact_paths, validation, repo_dir)
    update_run_manifest(artifacts_dir, artifact_paths, validation, repo_dir, sys.argv)
    update_checksum_file(
        checksums_dir / "sha256sums.txt",
        [
            args.research_plan_path.resolve(),
            *artifact_paths.values(),
            artifacts_dir / "run_manifest.json",
        ],
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
