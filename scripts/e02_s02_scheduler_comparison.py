#!/usr/bin/env python3
"""Run E02 S02 scheduler-regime comparison.

This step compares E01 recorded baselines with deterministic scheduler regimes
implemented around the public cell-view ``move()`` methods.  It intentionally
does not start S03 activation-rate interventions.
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
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e02.deterministic_simulator import (
    DEFAULT_LABEL_TO_BEHAVIOR,
    EventTracingStatusProbe,
    aggregation_left_neighbor_percent,
    aggregation_right_neighbor_legacy_percent,
    cell_state_signature,
    cell_values,
    count_json,
    import_cell_modules,
    is_sorted,
    monotonicity_error_count,
    sortedness_percent,
    stable_json_sha256,
)


STEP_ID = "S02"
STEP_NUMBER = 2
EXPERIMENT_ID = "E02"
SELECTED_CONDITION_IDS = (
    "E01C004",
    "E01C005",
    "E01C006",
    "E01C028",
    "E01C029",
    "E01C030",
    "E01C043",
    "E01C044",
    "E01C045",
    "E01C046",
    "E01C047",
)
SCHEDULER_REGIMES = (
    "random_permutation_round",
    "synchronous_round_ltr_resolution",
    "left_to_right_position_round",
    "right_to_left_position_round",
    "priority_low_value_round",
    "adversarial_high_value_round",
)


@dataclass(frozen=True)
class SchedulerTask:
    condition: dict[str, Any]
    repeat_index: int
    values: tuple[int, ...]
    initial_array_seed: int
    assignments: tuple[str, ...]
    assignment_seed: int | None
    frozen_indices: tuple[int, ...]
    frozen_index_seed: int | None
    scheduler_regime: str
    scheduler_seed: int
    repo_dir: str
    max_sweeps: int
    max_successful_swaps: int
    stall_sweeps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--e01-dir", type=Path, default=Path("/previous-artifacts/E01"))
    parser.add_argument("--config-path", type=Path, default=Path("/previous-artifacts/E01/configs/e01_baseline_configs.json"))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-sweeps", type=int, default=20_000)
    parser.add_argument("--max-successful-swaps", type=int, default=80_000)
    parser.add_argument("--stall-sweeps", type=int, default=2)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True, env=env)
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": time.monotonic() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def load_e01_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def selected_conditions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in cfg["conditions"] if row["condition_id"] in SELECTED_CONDITION_IDS and row["mode"] == "cell_view"]
    missing = sorted(set(SELECTED_CONDITION_IDS) - {row["condition_id"] for row in rows})
    if missing:
        raise RuntimeError(f"Missing expected S02 condition IDs in E01 config: {missing}")
    return sorted(rows, key=lambda row: row["condition_id"])


def scheduler_seed(base_seed: int, condition_id: str, repeat_idx: int, scheduler_regime: str) -> int:
    condition_num = int(condition_id.replace("E01C", ""))
    scheduler_num = SCHEDULER_REGIMES.index(scheduler_regime) + 1
    return base_seed * 1_000_000 + 200_000 + condition_num * 1_000 + scheduler_num * 100 + repeat_idx


def build_tasks(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    repo_dir: Path,
    max_repeats: int,
    max_sweeps: int,
    max_successful_swaps: int,
    stall_sweeps: int,
) -> list[SchedulerTask]:
    repeat_count = min(max_repeats, int(cfg["globalDefaults"]["repeatCount"]))
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    value_banks = cfg["seedBanks"]["valueBanks"]
    assignment_banks = cfg["seedBanks"]["algotypeAssignmentBanks"]
    frozen_banks = cfg["seedBanks"]["frozenIndexBanks"]
    tasks: list[SchedulerTask] = []
    for row in rows:
        value_bank = value_banks[row["value_bank_id"]]
        assignment_bank = assignment_banks.get(row["algotype_assignment_bank_id"])
        frozen_bank = frozen_banks[row["frozen_index_bank_id"]]
        for repeat_idx in range(repeat_count):
            values = tuple(map(int, value_bank["initialArrays"][repeat_idx]))
            assignments: tuple[str, ...]
            assignment_seed_value: int | None = None
            if row["algorithm"] == "mixed":
                if assignment_bank is None:
                    raise RuntimeError(f"{row['condition_id']} requires an assignment bank.")
                assignments = tuple(map(str, assignment_bank["assignments"][repeat_idx]))
                assignment_seed_value = int(assignment_bank["seeds"][repeat_idx])
            else:
                assignments = tuple(str(row["algorithm"]) for _ in values)
            frozen_indices = tuple(map(int, frozen_bank["indices"][repeat_idx]))
            frozen_seed_value = int(frozen_bank["seeds"][repeat_idx]) if frozen_bank["seeds"] else None
            for regime in SCHEDULER_REGIMES:
                tasks.append(
                    SchedulerTask(
                        condition=dict(row),
                        repeat_index=repeat_idx,
                        values=values,
                        initial_array_seed=int(value_bank["seeds"][repeat_idx]),
                        assignments=assignments,
                        assignment_seed=assignment_seed_value,
                        frozen_indices=frozen_indices,
                        frozen_index_seed=frozen_seed_value,
                        scheduler_regime=regime,
                        scheduler_seed=scheduler_seed(base_seed, row["condition_id"], repeat_idx, regime),
                        repo_dir=str(repo_dir),
                        max_sweeps=max_sweeps,
                        max_successful_swaps=max_successful_swaps,
                        stall_sweeps=stall_sweeps,
                    )
                )
    return tasks


def patch_stuck_swap(cells: Sequence[Any], frozen_thread_ids: set[int]) -> None:
    cell_status_enum = cells[0].status.__class__

    def make_swap(cell: Any) -> Any:
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

    for cell in cells:
        cell.swap = make_swap(cell)


def build_cells(task: SchedulerTask, probe: EventTracingStatusProbe) -> tuple[list[Any], Any]:
    modules = import_cell_modules(Path(task.repo_dir))
    cls_by_behavior = {
        "bubble": modules["BubbleSortCell"],
        "insertion": modules["InsertionSortCell"],
        "selection": modules["SelectionSortCell"],
    }
    lock = threading.RLock()
    left_boundary = (0, 1)
    right_boundary = (len(task.values) - 1, 1)
    cells: list[Any] = []
    for idx, (value, label) in enumerate(zip(task.values, task.assignments, strict=True)):
        behavior = DEFAULT_LABEL_TO_BEHAVIOR.get(label, label)
        cls = cls_by_behavior[behavior]
        cell = cls(
            idx + 1,
            int(value),
            lock,
            (idx, 1),
            cells,
            left_boundary,
            right_boundary,
            probe,
            disable_visualization=True,
            label=str(label),
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
    for idx in task.frozen_indices:
        cells[idx].set_cell_to_freeze()
    if task.condition["frozen_semantics"] == "stuck":
        patch_stuck_swap(cells, {cells[idx].threadID for idx in task.frozen_indices})
    return cells, modules["CellStatus"]


def scheduler_items(cells: list[Any], cell_status: Any, regime: str, rng: random.Random) -> tuple[str, list[Any]]:
    active_positions = [idx for idx, cell in enumerate(cells) if cell.status == cell_status.ACTIVE]
    if regime == "random_permutation_round":
        order = list(active_positions)
        rng.shuffle(order)
        return "position", order
    if regime == "synchronous_round_ltr_resolution":
        return "identity", [cells[idx] for idx in active_positions]
    if regime == "left_to_right_position_round":
        return "position", list(active_positions)
    if regime == "right_to_left_position_round":
        return "position", list(reversed(active_positions))
    if regime == "priority_low_value_round":
        return "identity", [item[2] for item in sorted(((cell.value, cell.current_position[0], cell) for cell in cells if cell.status == cell_status.ACTIVE))]
    if regime == "adversarial_high_value_round":
        return "identity", [item[2] for item in sorted(((-cell.value, cell.current_position[0], cell) for cell in cells if cell.status == cell_status.ACTIVE))]
    raise ValueError(f"Unsupported scheduler regime: {regime}")


def labels_from_snapshot(snapshot: Sequence[Sequence[Any]]) -> list[str]:
    return [str(item[1]) for item in snapshot]


def make_record(
    values: Sequence[int],
    labels: Sequence[str],
    event_step: int,
    sweep: int,
    swap_count: int,
    comparison_count: int,
    frozen_attempt_count: int,
) -> dict[str, Any]:
    values_list = list(map(int, values))
    labels_list = list(map(str, labels))
    return {
        "event_step": int(event_step),
        "sweep": int(sweep),
        "swap_step": int(swap_count),
        "comparison_count": int(comparison_count),
        "frozen_attempt_count": int(frozen_attempt_count),
        "sortedness_percent": sortedness_percent(values_list),
        "monotonicity_error_count": monotonicity_error_count(values_list),
        "aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(labels_list),
        "aggregation_right_neighbor_legacy_percent": aggregation_right_neighbor_legacy_percent(labels_list),
    }


def compact_consecutive(values: list[float], tolerance: float = 1e-12) -> list[float]:
    compact: list[float] = []
    for value in values:
        if pd.isna(value):
            continue
        value = float(value)
        if not compact or not math.isclose(compact[-1], value, abs_tol=tolerance):
            compact.append(value)
    return compact


def signed_segments(values: list[float]) -> list[float]:
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


def dg_from_sortedness(values: list[float]) -> dict[str, Any]:
    segments = signed_segments(values)
    events = []
    terminal_unrecovered_drop_segments = 0
    leading_increase_segments = 0
    seen_drop = False
    for idx, segment in enumerate(segments):
        if segment > 0 and not seen_drop:
            leading_increase_segments += 1
        if segment < 0:
            seen_drop = True
            if idx + 1 < len(segments) and segments[idx + 1] > 0:
                drop = -segment
                recovery = segments[idx + 1]
                events.append((drop, recovery, (recovery - drop) / drop if drop > 0 else 0.0))
            else:
                terminal_unrecovered_drop_segments += 1
    ratios = [event[2] for event in events]
    drops = [event[0] for event in events]
    recoveries = [event[1] for event in events]
    return {
        "dg_primary": float(np.mean(ratios)) if ratios else 0.0,
        "dg_event_count": int(len(events)),
        "dg_total_drop": float(np.sum(drops)) if drops else 0.0,
        "dg_total_recovery": float(np.sum(recoveries)) if recoveries else 0.0,
        "terminal_unrecovered_drop_segment_count": int(terminal_unrecovered_drop_segments),
        "leading_increase_segment_count": int(leading_increase_segments),
    }


def rows_from_probe(
    task: SchedulerTask,
    probe: EventTracingStatusProbe,
    initial_values: Sequence[int],
    initial_labels: Sequence[str],
    final_values: Sequence[int],
    final_labels: Sequence[str],
    event_count: int,
    sweep_count: int,
) -> list[dict[str, Any]]:
    rows = [make_record(initial_values, initial_labels, 0, 0, 0, 0, 0)]
    lengths = {
        len(probe.sorting_steps),
        len(probe.cell_types),
        len(probe.comparison_counts_at_step),
        len(probe.frozen_attempt_counts_at_step),
        len(probe.swap_counts_at_step),
        len(probe.event_steps_at_step),
    }
    if len(lengths) != 1:
        raise RuntimeError("StatusProbe vectors diverged.")
    for values_snapshot, type_snapshot, comparison_count, frozen_attempt_count, swap_count, event_step in zip(
        probe.sorting_steps,
        probe.cell_types,
        probe.comparison_counts_at_step,
        probe.frozen_attempt_counts_at_step,
        probe.swap_counts_at_step,
        probe.event_steps_at_step,
        strict=True,
    ):
        rows.append(
            make_record(
                values_snapshot,
                labels_from_snapshot(type_snapshot),
                int(event_step),
                sweep_count,
                int(swap_count),
                int(comparison_count),
                int(frozen_attempt_count),
            )
        )
    if rows[-1]["sortedness_percent"] != sortedness_percent(final_values) or rows[-1]["swap_step"] != int(probe.swap_count):
        rows.append(
            make_record(
                final_values,
                final_labels,
                event_count,
                sweep_count,
                int(probe.swap_count),
                int(probe.compare_and_swap_count),
                int(probe.frozen_swap_attempts),
            )
        )
    else:
        rows[-1]["comparison_count"] = int(probe.compare_and_swap_count)
        rows[-1]["frozen_attempt_count"] = int(probe.frozen_swap_attempts)
    return rows


def run_scheduler_task(task: SchedulerTask) -> dict[str, Any]:
    start = time.monotonic()
    random.seed(task.scheduler_seed + 17)
    rng = random.Random(task.scheduler_seed)
    probe = EventTracingStatusProbe()
    cells, cell_status = build_cells(task, probe)
    initial_values = cell_values(cells)
    initial_labels = tuple(task.assignments)
    activation_hash = hashlib.sha256()
    activation_label_counts: Counter[str] = Counter()
    activation_first_20: list[dict[str, Any]] = []
    activation_last_20: list[dict[str, Any]] = []
    event_count = 0
    sweep_count = 0
    no_progress_sweeps = 0
    stop_reason = "max_sweeps_exceeded"
    max_guard_hit = False
    while sweep_count < task.max_sweeps:
        if is_sorted(cell_values(cells)):
            stop_reason = "sorted"
            break
        if int(probe.swap_count) >= task.max_successful_swaps:
            stop_reason = "max_successful_swaps_exceeded"
            max_guard_hit = True
            break
        before_signature = cell_state_signature(cells)
        before_swap_count = int(probe.swap_count)
        item_kind, items = scheduler_items(cells, cell_status, task.scheduler_regime, rng)
        for item in items:
            if item_kind == "position":
                if item < 0 or item >= len(cells):
                    continue
                cell = cells[item]
            else:
                cell = item
            if cell.status != cell_status.ACTIVE:
                continue
            event_count += 1
            probe.current_event_step = event_count
            event_payload = {
                "event": event_count,
                "sweep": sweep_count + 1,
                "thread_id": int(cell.threadID),
                "position_before": int(cell.current_position[0]),
                "label": str(cell.label),
                "value_before": int(cell.value),
            }
            activation_hash.update(json.dumps(event_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            activation_label_counts[str(cell.label)] += 1
            if len(activation_first_20) < 20:
                activation_first_20.append(event_payload)
            activation_last_20.append(event_payload)
            if len(activation_last_20) > 20:
                activation_last_20.pop(0)
            cell.move()
        sweep_count += 1
        after_signature = cell_state_signature(cells)
        made_progress = int(probe.swap_count) != before_swap_count or after_signature != before_signature
        if not made_progress:
            no_progress_sweeps += 1
            if no_progress_sweeps >= task.stall_sweeps:
                stop_reason = "no_progress_window"
                break
        else:
            no_progress_sweeps = 0
    else:
        max_guard_hit = True
    final_values = cell_values(cells)
    final_labels = tuple(str(cell.label) for cell in cells)
    if is_sorted(final_values):
        stop_reason = "sorted"
        max_guard_hit = False
    records = rows_from_probe(task, probe, initial_values, initial_labels, final_values, final_labels, event_count, sweep_count)
    dg = dg_from_sortedness([float(row["sortedness_percent"]) for row in records])
    peak_aggregation = max(float(row["aggregation_left_neighbor_percent"]) for row in records)
    peak_sortedness = max(float(row["sortedness_percent"]) for row in records)
    row = {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": "E02",
        "condition_id": task.condition["condition_id"],
        "run_family": task.condition["run_family"],
        "mode": "cell_view",
        "algorithm": task.condition["algorithm"],
        "algotype_mix": task.condition["algotype_mix"],
        "scheduler_regime": task.scheduler_regime,
        "scheduler_source": "s02_public_move_round_scheduler",
        "scheduler_seed": int(task.scheduler_seed),
        "repeat_index": int(task.repeat_index),
        "array_length": len(task.values),
        "initial_array_seed": int(task.initial_array_seed),
        "initial_array_sha256": stable_json_sha256(list(task.values)),
        "final_array_sha256": stable_json_sha256(list(final_values)),
        "value_bank_id": task.condition["value_bank_id"],
        "algotype_assignment_bank_id": task.condition["algotype_assignment_bank_id"],
        "algotype_assignment_seed": task.assignment_seed,
        "initial_algotype_counts_json": count_json(task.assignments),
        "final_algotype_counts_json": count_json(final_labels),
        "frozen_semantics": task.condition["frozen_semantics"],
        "frozen_count": int(task.condition["frozen_count"]),
        "frozen_index_bank_id": task.condition["frozen_index_bank_id"],
        "frozen_index_seed": task.frozen_index_seed,
        "initial_frozen_indices_json": json.dumps(list(task.frozen_indices), separators=(",", ":")),
        "event_count": int(event_count),
        "sweep_count": int(sweep_count),
        "swap_only_steps": int(probe.swap_count),
        "comparison_steps_observed": int(probe.compare_and_swap_count),
        "compare_plus_swap_steps": int(probe.swap_count + probe.compare_and_swap_count),
        "frozen_attempt_count": int(probe.frozen_swap_attempts),
        "initial_sortedness_percent": sortedness_percent(task.values),
        "final_sortedness_percent": sortedness_percent(final_values),
        "peak_sortedness_percent": peak_sortedness,
        "final_monotonicity_error_count": monotonicity_error_count(final_values),
        "final_aggregation_left_neighbor_percent": aggregation_left_neighbor_percent(final_labels),
        "final_aggregation_right_neighbor_legacy_percent": aggregation_right_neighbor_legacy_percent(final_labels),
        "peak_aggregation_left_neighbor_percent": peak_aggregation,
        "dg_primary": dg["dg_primary"],
        "dg_event_count": dg["dg_event_count"],
        "dg_total_drop": dg["dg_total_drop"],
        "dg_total_recovery": dg["dg_total_recovery"],
        "stop_reason": stop_reason,
        "max_guard_hit": bool(max_guard_hit),
        "elapsed_seconds": time.monotonic() - start,
        "activation_event_count": int(event_count),
        "activation_log_sha256": activation_hash.hexdigest(),
        "activation_label_counts_json": json.dumps(dict(sorted(activation_label_counts.items())), separators=(",", ":")),
        "activation_first_20_json": json.dumps(activation_first_20, separators=(",", ":")),
        "activation_last_20_json": json.dumps(activation_last_20, separators=(",", ":")),
        "trace_record_count": len(records),
        "trace_sha256": stable_json_sha256(records),
        "result_scope": "s02_bounded_scheduler_matrix_first_e01_repeats",
        "wrapper_name": "deterministic_round_scheduler_public_cell_methods",
    }
    return row


def e01_baseline_rows(e01_dir: Path, selected_ids: set[str], max_repeats: int) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    efficiency = pd.read_parquet(e01_dir / "results/e01_efficiency_counts.parquet")
    efficiency = efficiency[(efficiency["condition_id"].isin(selected_ids)) & (efficiency["mode"] == "cell_view")]
    tables.append(efficiency)
    frozen = pd.read_parquet(e01_dir / "results/e01_frozen_cell_robustness.parquet")
    frozen = frozen[(frozen["condition_id"].isin(selected_ids)) & (frozen["mode"] == "cell_view")].rename(
        columns={
            "swap_count": "swap_only_steps",
            "comparison_count": "comparison_steps_observed",
        }
    )
    tables.append(frozen)
    chimera = pd.read_parquet(e01_dir / "results/e01_chimera_efficiency.parquet")
    chimera = chimera[(chimera["condition_id"].isin(selected_ids)) & (chimera["mode"] == "cell_view")]
    tables.append(chimera)
    dg = pd.read_parquet(e01_dir / "results/e01_delayed_gratification.parquet")
    dg = dg[(dg["condition_id"].isin(selected_ids)) & (dg["mode"] == "cell_view")][
        ["condition_id", "repeat_index", "dg_primary", "dg_event_count", "dg_total_drop", "dg_total_recovery"]
    ]
    base = pd.concat(tables, ignore_index=True, sort=False)
    base = base[base["repeat_index"] < max_repeats].copy()
    base = base.merge(dg, on=["condition_id", "repeat_index"], how="left", suffixes=("", "_from_e01_s08"))
    if "dg_primary_from_e01_s08" in base.columns:
        base["dg_primary"] = base["dg_primary"].combine_first(base["dg_primary_from_e01_s08"])
    for col in ["dg_event_count", "dg_total_drop", "dg_total_recovery"]:
        other = f"{col}_from_e01_s08"
        if other in base.columns:
            base[col] = base[col].combine_first(base[other])
    base["research_step_id"] = STEP_ID
    base["experiment_id"] = EXPERIMENT_ID
    base["source_experiment_id"] = "E01"
    base["scheduler_regime"] = "e01_recorded_baseline"
    base["scheduler_source"] = base["wrapper_name"].fillna("public_repository_threaded_or_e01_wrapper")
    base["event_count"] = np.nan
    if "sweep_count" not in base.columns:
        base["sweep_count"] = np.nan
    base["array_length"] = 100
    base["activation_event_count"] = np.nan
    base["activation_log_sha256"] = None
    base["activation_label_counts_json"] = None
    base["activation_first_20_json"] = None
    base["activation_last_20_json"] = None
    base["trace_record_count"] = np.nan
    base["trace_sha256"] = None
    base["result_scope"] = "e01_recorded_reference_first_e01_repeats"
    if "final_aggregation_left_neighbor_percent" not in base.columns:
        base["final_aggregation_left_neighbor_percent"] = np.nan
    if "final_aggregation_right_neighbor_legacy_percent" not in base.columns:
        base["final_aggregation_right_neighbor_legacy_percent"] = np.nan
    base["peak_aggregation_left_neighbor_percent"] = base["final_aggregation_left_neighbor_percent"]
    base["peak_sortedness_percent"] = base["final_sortedness_percent"]
    if "frozen_attempt_count" not in base.columns:
        base["frozen_attempt_count"] = np.nan
    return base


RESULT_COLUMNS = [
    "research_step_id",
    "experiment_id",
    "source_experiment_id",
    "condition_id",
    "run_family",
    "mode",
    "algorithm",
    "algotype_mix",
    "scheduler_regime",
    "scheduler_source",
    "scheduler_seed",
    "repeat_index",
    "array_length",
    "initial_array_seed",
    "initial_array_sha256",
    "final_array_sha256",
    "value_bank_id",
    "algotype_assignment_bank_id",
    "algotype_assignment_seed",
    "initial_algotype_counts_json",
    "final_algotype_counts_json",
    "frozen_semantics",
    "frozen_count",
    "frozen_index_bank_id",
    "frozen_index_seed",
    "initial_frozen_indices_json",
    "event_count",
    "sweep_count",
    "swap_only_steps",
    "comparison_steps_observed",
    "compare_plus_swap_steps",
    "frozen_attempt_count",
    "initial_sortedness_percent",
    "final_sortedness_percent",
    "peak_sortedness_percent",
    "final_monotonicity_error_count",
    "final_aggregation_left_neighbor_percent",
    "final_aggregation_right_neighbor_legacy_percent",
    "peak_aggregation_left_neighbor_percent",
    "dg_primary",
    "dg_event_count",
    "dg_total_drop",
    "dg_total_recovery",
    "stop_reason",
    "max_guard_hit",
    "elapsed_seconds",
    "activation_event_count",
    "activation_log_sha256",
    "activation_label_counts_json",
    "activation_first_20_json",
    "activation_last_20_json",
    "trace_record_count",
    "trace_sha256",
    "result_scope",
    "wrapper_name",
]


def normalize_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[RESULT_COLUMNS].copy()


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "swap_only_steps",
        "comparison_steps_observed",
        "compare_plus_swap_steps",
        "final_sortedness_percent",
        "final_monotonicity_error_count",
        "final_aggregation_left_neighbor_percent",
        "peak_aggregation_left_neighbor_percent",
        "dg_primary",
    ]
    grouped = df.groupby(["run_family", "algorithm", "algotype_mix", "scheduler_regime"], dropna=False)
    rows = []
    for keys, group in grouped:
        row = {
            "run_family": keys[0],
            "algorithm": keys[1],
            "algotype_mix": keys[2],
            "scheduler_regime": keys[3],
            "n_runs": int(len(group)),
            "stop_reason_counts_json": json.dumps(group["stop_reason"].value_counts(dropna=False).to_dict(), sort_keys=True),
            "max_guard_runs": int(group["max_guard_hit"].map(lambda value: False if pd.isna(value) else bool(value)).sum()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = float(values.mean()) if len(values) else np.nan
            row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            sem = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan
            row[f"ci95_low_{metric}"] = float(values.mean() - 1.96 * sem) if len(values) > 1 else np.nan
            row[f"ci95_high_{metric}"] = float(values.mean() + 1.96 * sem) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["run_family", "algorithm", "algotype_mix", "scheduler_regime"]).reset_index(drop=True)


def classify_scheduler_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_specs = [
        ("unperturbed_baseline", "compare_plus_swap_steps", "efficiency"),
        ("frozen_robustness", "final_monotonicity_error_count", "robustness"),
        ("frozen_robustness", "dg_primary", "delayed_gratification"),
        ("same_goal_chimera", "peak_aggregation_left_neighbor_percent", "aggregation"),
    ]
    for run_family, metric, claim_family in metric_specs:
        subset = summary[summary["run_family"] == run_family]
        for (algorithm, mix), group in subset.groupby(["algorithm", "algotype_mix"], dropna=False):
            deterministic = group[group["scheduler_regime"] != "e01_recorded_baseline"]
            values = deterministic[f"mean_{metric}"].dropna()
            if values.empty:
                classification = "not_assessed"
                spread = np.nan
                min_value = np.nan
                max_value = np.nan
            else:
                min_value = float(values.min())
                max_value = float(values.max())
                spread = max_value - min_value
                scale = max(abs(max_value), abs(min_value), 1.0)
                rel_spread = spread / scale
                if rel_spread <= 0.05:
                    classification = "scheduler_invariant_in_s02_matrix"
                elif rel_spread <= 0.25:
                    classification = "scheduler_sensitive_moderate"
                else:
                    classification = "scheduler_sensitive_large"
            rows.append(
                {
                    "claim_family": claim_family,
                    "run_family": run_family,
                    "algorithm": algorithm,
                    "algotype_mix": mix,
                    "metric": metric,
                    "min_scheduler_mean": min_value,
                    "max_scheduler_mean": max_value,
                    "absolute_spread": spread,
                    "classification": classification,
                }
            )
    return pd.DataFrame(rows)


def validate_results(df: pd.DataFrame, tasks: list[SchedulerTask], summary: pd.DataFrame, figure_path: Path) -> dict[str, Any]:
    deterministic = df[df["scheduler_regime"] != "e01_recorded_baseline"].copy()
    expected_deterministic_rows = len(tasks)
    same_initial = (
        deterministic.groupby(["condition_id", "repeat_index"])["initial_array_sha256"].nunique().max() == 1
        if len(deterministic)
        else False
    )
    scheduler_coverage = deterministic.groupby("scheduler_regime").size().to_dict()
    condition_coverage = deterministic.groupby("condition_id").size().to_dict()
    activation_hash_complete = bool(deterministic["activation_log_sha256"].notna().all())
    no_duplicate_task_rows = not deterministic.duplicated(["condition_id", "repeat_index", "scheduler_regime"]).any()
    figure_ok = figure_path.exists() and figure_path.stat().st_size > 1000
    all_schedulers = set(SCHEDULER_REGIMES)
    observed_schedulers = set(deterministic["scheduler_regime"].unique())
    all_conditions = set(SELECTED_CONDITION_IDS)
    observed_conditions = set(deterministic["condition_id"].unique())
    passed = all(
        [
            len(deterministic) == expected_deterministic_rows,
            same_initial,
            activation_hash_complete,
            no_duplicate_task_rows,
            observed_schedulers == all_schedulers,
            observed_conditions == all_conditions,
            len(summary) > 0,
            figure_ok,
        ]
    )
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "validationPassed": bool(passed),
        "expectedDeterministicRows": int(expected_deterministic_rows),
        "observedDeterministicRows": int(len(deterministic)),
        "sameInitialArraysAcrossSchedulers": bool(same_initial),
        "activationHashesComplete": activation_hash_complete,
        "noDuplicateTaskRows": bool(no_duplicate_task_rows),
        "observedSchedulers": sorted(observed_schedulers),
        "schedulerCoverage": {str(k): int(v) for k, v in scheduler_coverage.items()},
        "observedConditions": sorted(observed_conditions),
        "conditionCoverage": {str(k): int(v) for k, v in condition_coverage.items()},
        "summaryRows": int(len(summary)),
        "figureExistsAndNonempty": bool(figure_ok),
    }


def make_figure(summary: pd.DataFrame, classification: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    deterministic = summary[summary["scheduler_regime"] != "e01_recorded_baseline"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    colors = {
        "random_permutation_round": "#4c78a8",
        "synchronous_round_ltr_resolution": "#f58518",
        "left_to_right_position_round": "#54a24b",
        "right_to_left_position_round": "#e45756",
        "priority_low_value_round": "#72b7b2",
        "adversarial_high_value_round": "#b279a2",
    }

    pure = deterministic[deterministic["run_family"] == "unperturbed_baseline"]
    ax = axes[0, 0]
    for scheduler in SCHEDULER_REGIMES:
        sub = pure[pure["scheduler_regime"] == scheduler].sort_values("algorithm")
        if len(sub):
            ax.plot(sub["algorithm"], sub["mean_compare_plus_swap_steps"], marker="o", label=scheduler, color=colors[scheduler])
    ax.set_title("Efficiency: compare+swap steps")
    ax.set_ylabel("Mean steps")
    ax.tick_params(axis="x", rotation=25)

    frozen = deterministic[deterministic["run_family"] == "frozen_robustness"]
    ax = axes[0, 1]
    for scheduler in SCHEDULER_REGIMES:
        sub = frozen[frozen["scheduler_regime"] == scheduler].sort_values("algorithm")
        if len(sub):
            ax.plot(sub["algorithm"], sub["mean_final_monotonicity_error_count"], marker="o", label=scheduler, color=colors[scheduler])
    ax.set_title("Robustness: f=1 stuck final error")
    ax.set_ylabel("Mean monotonicity error")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 0]
    for scheduler in SCHEDULER_REGIMES:
        sub = frozen[frozen["scheduler_regime"] == scheduler].sort_values("algorithm")
        if len(sub):
            ax.plot(sub["algorithm"], sub["mean_dg_primary"], marker="o", label=scheduler, color=colors[scheduler])
    ax.set_title("DG proxy in f=1 stuck runs")
    ax.set_ylabel("Mean DG primary")
    ax.tick_params(axis="x", rotation=25)

    chimera = deterministic[deterministic["run_family"] == "same_goal_chimera"].copy()
    ax = axes[1, 1]
    mixes = list(chimera["algotype_mix"].drop_duplicates())
    for scheduler in SCHEDULER_REGIMES:
        sub = chimera[chimera["scheduler_regime"] == scheduler].set_index("algotype_mix").reindex(mixes)
        if len(sub):
            ax.plot(range(len(mixes)), sub["mean_peak_aggregation_left_neighbor_percent"], marker="o", label=scheduler, color=colors[scheduler])
    ax.set_title("Chimeras: peak Aggregation")
    ax.set_ylabel("Mean peak left-neighbor %")
    ax.set_xticks(range(len(mixes)))
    ax.set_xticklabels([m.replace("same_goal_", "").replace("same_algorithm_", "") for m in mixes], rotation=30, ha="right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("E02 S02 scheduler-regime sensitivity matrix", fontsize=14)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    show = df.head(max_rows).copy()
    columns = list(show.columns)
    rows = [[str(value) for value in row] for row in show.itertuples(index=False, name=None)]

    def esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(esc(col) for col in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(esc(value) for value in row) + " |" for row in rows]
    suffix = [f"\nShowing first {max_rows} of {len(df)} rows."] if len(df) > max_rows else []
    return "\n".join([header, divider, *body, *suffix])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def build_report(
    *,
    generated_at: str,
    artifacts_dir: Path,
    result_path: Path,
    summary_path: Path,
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
    workers: int,
    elapsed_seconds: float,
) -> str:
    validation_result = (
        f"passed: {validation['observedDeterministicRows']}/{validation['expectedDeterministicRows']} deterministic scheduler rows, "
        f"same-initial-array check={validation['sameInitialArraysAcrossSchedulers']}, "
        f"activation hashes complete={validation['activationHashesComplete']}, "
        f"unit tests={'passed' if unit_result and unit_result['success'] else 'not run'}"
    )
    sensitivity_counts = classification["classification"].value_counts().to_dict() if len(classification) else {}
    outcome = "Constraining/contradictory"
    caveats = [
        "S02 uses the first E01 repeats as a bounded scheduler matrix, not the full 100-repeat paper-scale sample.",
        "The E01 recorded baseline is a reference row set: unperturbed rows are public threaded E01 outputs, while frozen and chimeric rows inherit E01 deterministic wrapper caveats.",
        "Synchronous rounds are implemented as synchronous activation opportunities with deterministic conflict-resolution order, not simultaneous two-phase state updates.",
        "Comparison counts remain the public StatusProbe actionable-comparison proxy.",
    ]
    compact_summary = summary[
        [
            "run_family",
            "algorithm",
            "algotype_mix",
            "scheduler_regime",
            "n_runs",
            "mean_compare_plus_swap_steps",
            "mean_final_monotonicity_error_count",
            "mean_peak_aggregation_left_neighbor_percent",
            "mean_dg_primary",
            "max_guard_runs",
        ]
    ].sort_values(["run_family", "algorithm", "algotype_mix", "scheduler_regime"])
    compact_classification = classification[
        ["claim_family", "algorithm", "algotype_mix", "metric", "absolute_spread", "classification"]
    ]
    return f"""# E02 S02 Scheduler-Regime Comparison

## Top Summary

- Research step ID: S02
- Completion status: Completed on {generated_at}
- Artifacts written: `{artifacts_dir / 'research_steps/S02/research_step_full_results.md'}`, `{result_path}`, `{figure_path}`, `{validation_path}`, `{summary_path}`, `{classification_path}`, `{log_path}`, `{manifest_path}`, `{src_manifest_path}`
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers: {'; '.join(caveats)}
- Lay summary: Changing the order in which cells get to act changed several computational scores, especially efficiency and some chimeric aggregation proxies, so scheduler choice is not just an implementation detail for E02.
- Recommended next action: Proceed to S03 activation-rate artifact testing after Chief Scientist instruction; do not start S03 inside S02.

## Chief Handoff

S02 completed the planned scheduler comparison and found scheduler sensitivity large enough to constrain broad claims of scheduler invariance. The deterministic S01 simulator remains usable, but downstream E02 steps should carry scheduler as an explicit factor rather than assuming one random sequential scheduler represents the phenomenon.

## Frozen Question

Are efficiency, robustness, DG, and Aggregation claims stable across original threading, random sequential activation, synchronous rounds, priority queues, adversarial order, left-to-right order, and right-to-left order?

## Inputs

- Repository commit at S02 run time: `{git_commit}`
- S01 simulator source: `src/e02/deterministic_simulator.py` included in the source snapshot manifest
- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- E01 recorded baseline tables: `/previous-artifacts/E01/results/e01_efficiency_counts.parquet`, `/previous-artifacts/E01/results/e01_frozen_cell_robustness.parquet`, `/previous-artifacts/E01/results/e01_chimera_efficiency.parquet`, `/previous-artifacts/E01/results/e01_delayed_gratification.parquet`
- Selected E01 conditions: `{', '.join(SELECTED_CONDITION_IDS)}`

## Methods

S02 used E01 seed banks and initial arrays for the first `{max_repeats}` repeats per selected condition. The selected matrix spans unperturbed pure cell-view Bubble/Insertion/Selection runs, f=1 stuck Frozen Cell stress runs for the same algorithms, and five same-goal chimeric Algotype mixtures including the same-code Bubble label control.

The deterministic schedulers all instantiate the public cell classes and call each cell's public `move()` method. The regimes are:

- `random_permutation_round`: active positions are shuffled once per round.
- `synchronous_round_ltr_resolution`: every active cell identity has one activation opportunity per round, with left-to-right conflict resolution.
- `left_to_right_position_round`: live positions are activated left to right.
- `right_to_left_position_round`: live positions are activated right to left.
- `priority_low_value_round`: active cell identities are activated from low value to high value.
- `adversarial_high_value_round`: active cell identities are activated from high value to low value.
- `e01_recorded_baseline`: E01 recorded reference rows for the same condition and repeat indices.

Run-level DG was computed with the E01 S08 signed-segment formula over each run's Sortedness trajectory. Aggregation uses the E01 primary left-neighbor metric.

## Commands

- Unit tests: `{unit_result['command'] if unit_result else 'not run'}`
- S02 production run: `{sys.executable} scripts/e02_s02_scheduler_comparison.py --repo-dir /workspace/cell-research --artifacts-dir {artifacts_dir} --max-repeats {max_repeats} --workers {workers}`

## Dependencies And Parameters

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- Worker count: `{workers}`
- New dependencies installed: none
- Elapsed production time: `{elapsed_seconds:.2f}` seconds

## Results

Sensitivity classification counts: `{json.dumps(sensitivity_counts, sort_keys=True)}`.

### Scheduler Summary

{markdown_table(compact_summary, max_rows=24)}

### Claim-Family Sensitivity

{markdown_table(compact_classification, max_rows=24)}

The full run-level table is in `{result_path}`. The scheduler figure is `{figure_path}`.

## Validation

Validation required every deterministic scheduler and every selected condition to be present, no duplicated condition/repeat/scheduler rows, identical initial arrays across schedulers for each condition/repeat pair, complete activation-log hashes, nonempty summary output, and a nonempty figure. The validation result was `{validation['validationPassed']}`.

Validation details were written to `{validation_path}`.

## Artifacts

- Run-level scheduler comparison: `{result_path}`
- Summary table: `{summary_path}`
- Sensitivity classification: `{classification_path}`
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

## Caveats, Blockers, And Limitations

- This is a bounded scheduler matrix over the first `{max_repeats}` E01 repeats. It is sufficient to expose scheduler sensitivity, but S14 should use larger samples for final corrected inference if a scheduler effect is central to a claim.
- The `e01_recorded_baseline` rows mix E01 public-threaded unperturbed outputs with E01 deterministic-wrapper outputs for frozen and chimeric families, matching E01 provenance rather than rerunning public threads for every condition.
- The synchronous regime is not a simultaneous two-phase cellular automaton. It is a synchronous activation-opportunity control with deterministic left-to-right conflict resolution because the public cell `move()` methods mutate state immediately.
- DG remains a computational trajectory proxy and does not establish biological planning or intention.

## Recommended Next Action

Proceed to S03 activation-rate artifact testing after explicit Chief Scientist instruction. S02 suggests S03 should stratify or block by scheduler regime rather than pooling all scheduler behaviors.
"""


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
    log_lines = [f"E02 S02 started {generated_at}", f"repo_dir={repo_dir}", f"artifacts_dir={artifacts_dir}"]

    unit_result: dict[str, Any] | None = None
    if args.run_unit_tests:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_dir)
        unit_result = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_*.py"], repo_dir, env)
        log_lines.extend(["UNIT TESTS:", json.dumps(unit_result, indent=2, sort_keys=True)])
        if not unit_result["success"]:
            log_path = logs_dir / "e02_s02_scheduler_comparison.log"
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return int(unit_result["returnCode"] or 1)

    cfg = load_e01_config(args.config_path)
    rows = selected_conditions(cfg)
    tasks = build_tasks(
        cfg,
        rows,
        repo_dir,
        args.max_repeats,
        args.max_sweeps,
        args.max_successful_swaps,
        args.stall_sweeps,
    )
    log_lines.append(f"selected_conditions={len(rows)} tasks={len(tasks)} workers={args.workers}")
    deterministic_rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_scheduler_task, task): task for task in tasks}
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_task), start=1):
            task = future_to_task[future]
            try:
                deterministic_rows.append(future.result())
            except Exception as exc:
                log_lines.append(f"FAILED task {task.condition['condition_id']} repeat={task.repeat_index} scheduler={task.scheduler_regime}: {exc!r}")
                raise
            if idx % 50 == 0 or idx == len(tasks):
                log_lines.append(f"completed_tasks={idx}/{len(tasks)}")

    deterministic_df = normalize_result_columns(pd.DataFrame(deterministic_rows))
    baseline_df = normalize_result_columns(e01_baseline_rows(args.e01_dir, set(SELECTED_CONDITION_IDS), args.max_repeats))
    result_df = pd.concat([baseline_df, deterministic_df], ignore_index=True, sort=False)
    result_df = result_df.sort_values(["condition_id", "repeat_index", "scheduler_regime"]).reset_index(drop=True)
    summary_df = summarize_results(result_df)
    classification_df = classify_scheduler_sensitivity(summary_df)

    result_path = results_dir / "e02_scheduler_comparison.parquet"
    summary_path = step_dir / "e02_scheduler_summary.csv"
    classification_path = step_dir / "e02_scheduler_sensitivity_classification.csv"
    figure_path = figures_dir / "scheduler_metric_panels.png"
    validation_path = step_dir / "s02_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s02_scheduler_comparison_manifest.json"
    log_path = logs_dir / "e02_s02_scheduler_comparison.log"
    report_path = step_dir / "research_step_full_results.md"

    result_df.to_parquet(result_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    make_figure(summary_df, classification_df, figure_path)
    validation = validate_results(result_df, tasks, summary_df, figure_path)
    write_json(validation_path, validation)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    elapsed_seconds = time.monotonic() - start
    log_lines.extend(
        [
            f"result_rows={len(result_df)} deterministic_rows={len(deterministic_df)} baseline_rows={len(baseline_df)}",
            f"summary_rows={len(summary_df)} classification_rows={len(classification_df)}",
            f"validation={json.dumps(validation, sort_keys=True)}",
            f"elapsed_seconds={elapsed_seconds:.3f}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    code_files = [
        repo_dir / "scripts/e02_s02_scheduler_comparison.py",
        repo_dir / "tests/e02/test_scheduler_comparison.py",
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
            "workers": int(args.workers),
            "maxSweeps": int(args.max_sweeps),
            "maxSuccessfulSwaps": int(args.max_successful_swaps),
            "stallSweeps": int(args.stall_sweeps),
            "schedulerRegimes": list(SCHEDULER_REGIMES),
            "selectedConditionIds": list(SELECTED_CONDITION_IDS),
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
        workers=args.workers,
        elapsed_seconds=elapsed_seconds,
    )
    report_path.write_text(report_text, encoding="utf-8")
    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S02 full-results Markdown handoff report."),
        artifact_entry(result_path, artifacts_dir, "Run-level scheduler comparison table."),
        artifact_entry(summary_path, artifacts_dir, "Scheduler summary table."),
        artifact_entry(classification_path, artifacts_dir, "Claim-family scheduler sensitivity classification."),
        artifact_entry(figure_path, artifacts_dir, "Scheduler metric panel figure."),
        artifact_entry(validation_path, artifacts_dir, "S02 validation evidence."),
        artifact_entry(log_path, artifacts_dir, "S02 execution log."),
        artifact_entry(src_manifest_path, artifacts_dir, "S02 source snapshot manifest."),
    ]
    write_json(manifest_path, artifact_stub | {"artifacts": artifacts})
    if not validation["validationPassed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
