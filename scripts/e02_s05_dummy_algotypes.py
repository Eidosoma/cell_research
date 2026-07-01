#!/usr/bin/env python3
"""Run E02 S05 behavior-preserving dummy-Algotype controls.

This step reruns the E01 same-code Bubble label control as an actual
deterministic simulation.  The two visible labels are distinct, but both labels
must instantiate the same public BubbleSortCell class and call the same move()
implementation.  Each dummy run is paired with an all-Bubble reference run
using the same values and activation order; S05 treats exact value-trajectory
identity as runtime evidence that labels did not alter behavior.
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
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.e02_s02_scheduler_comparison import (
    artifact_entry,
    build_cells,
    dg_from_sortedness,
    git_output,
    load_e01_config,
    markdown_table,
    run_command,
    scheduler_items,
    sha256_file,
    write_json,
)
from scripts.e02_s04_label_shuffle_nulls import (
    empirical_quantile,
    empirical_upper_tail,
    max_distribution_stats,
    normal_quantiles,
    normal_upper_tail,
    simulate_row_null_distribution,
)
from src.e02.deterministic_simulator import (
    DEFAULT_LABEL_TO_BEHAVIOR,
    aggregation_left_neighbor_percent,
    aggregation_right_neighbor_legacy_percent,
    cell_state_signature,
    cell_values,
    count_json,
    has_public_legal_action,
    is_sorted,
    label_code,
    monotonicity_error_count,
    sortedness_percent,
    stable_json_sha256,
)
from src.e02.deterministic_simulator import EventTracingStatusProbe


STEP_ID = "S05"
STEP_NUMBER = 5
EXPERIMENT_ID = "E02"
CONTROL_CONDITION_ID = "E01C047"
DUMMY_LABELS = ("bubble_label_a", "bubble_label_b")
DUMMY_BEHAVIOR = "bubble"
SCHEDULER_REGIME = "identity_per_cell_round_ltr"
SCHEDULER_SOURCE = "s05_identity_round_public_move_scheduler"
REFERENCE_SCOPE = "paired_all_bubble_reference_same_values_same_activation_order"


@dataclass(frozen=True)
class DummyTask:
    condition: dict[str, Any]
    repeat_index: int
    values: tuple[int, ...]
    initial_array_seed: int
    assignments: tuple[str, ...]
    assignment_seed: int | None
    frozen_indices: tuple[int, ...]
    frozen_index_seed: int | None
    scheduler_seed: int
    repo_dir: str
    max_sweeps: int
    max_successful_swaps: int
    stall_sweeps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config-path", type=Path, default=Path("/previous-artifacts/E01/configs/e01_baseline_configs.json"))
    parser.add_argument("--s04-null-path", type=Path, default=Path("/artifacts/results/e02_label_shuffle_aggregation_nulls.parquet"))
    parser.add_argument("--max-repeats", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-sweeps", type=int, default=20_000)
    parser.add_argument("--max-successful-swaps", type=int, default=80_000)
    parser.add_argument("--stall-sweeps", type=int, default=2)
    parser.add_argument("--null-samples", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=2026070105)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def control_condition(cfg: dict[str, Any]) -> dict[str, Any]:
    matches = [row for row in cfg["conditions"] if row["condition_id"] == CONTROL_CONDITION_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {CONTROL_CONDITION_ID} condition, found {len(matches)}.")
    row = dict(matches[0])
    if row.get("algotype_assignment_bank_id") != "bubble_bubble_label_control_50_50":
        raise RuntimeError(f"{CONTROL_CONDITION_ID} is not the expected Bubble dummy-label control.")
    return row


def scheduler_seed(base_seed: int, repeat_idx: int) -> int:
    condition_num = int(CONTROL_CONDITION_ID.replace("E01C", ""))
    return base_seed * 1_000_000 + 500_000 + condition_num * 1_000 + repeat_idx


def build_tasks(
    cfg: dict[str, Any],
    row: dict[str, Any],
    repo_dir: Path,
    max_repeats: int,
    max_sweeps: int,
    max_successful_swaps: int,
    stall_sweeps: int,
) -> list[DummyTask]:
    repeat_count = min(max_repeats, int(cfg["globalDefaults"]["repeatCount"]))
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    value_bank = cfg["seedBanks"]["valueBanks"][row["value_bank_id"]]
    assignment_bank = cfg["seedBanks"]["algotypeAssignmentBanks"][row["algotype_assignment_bank_id"]]
    frozen_bank = cfg["seedBanks"]["frozenIndexBanks"][row["frozen_index_bank_id"]]
    tasks: list[DummyTask] = []
    for repeat_idx in range(repeat_count):
        assignments = tuple(map(str, assignment_bank["assignments"][repeat_idx]))
        counts = Counter(assignments)
        if set(counts) != set(DUMMY_LABELS) or any(counts[label] != 50 for label in DUMMY_LABELS):
            raise RuntimeError(f"Repeat {repeat_idx} does not have balanced 50/50 dummy labels: {dict(counts)}")
        frozen_indices = tuple(map(int, frozen_bank["indices"][repeat_idx]))
        tasks.append(
            DummyTask(
                condition=row,
                repeat_index=repeat_idx,
                values=tuple(map(int, value_bank["initialArrays"][repeat_idx])),
                initial_array_seed=int(value_bank["seeds"][repeat_idx]),
                assignments=assignments,
                assignment_seed=int(assignment_bank["seeds"][repeat_idx]),
                frozen_indices=frozen_indices,
                frozen_index_seed=int(frozen_bank["seeds"][repeat_idx]) if frozen_bank["seeds"] else None,
                scheduler_seed=scheduler_seed(base_seed, repeat_idx),
                repo_dir=str(repo_dir),
                max_sweeps=max_sweeps,
                max_successful_swaps=max_successful_swaps,
                stall_sweeps=stall_sweeps,
            )
        )
    return tasks


def labels_from_snapshot(snapshot: Sequence[Sequence[Any]]) -> list[str]:
    return [str(item[1]) for item in snapshot]


def make_metric_record(
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
        "values": values_list,
        "labels": labels_list,
        "sortedness_percent": float(sortedness_percent(values_list)),
        "monotonicity_error_count": int(monotonicity_error_count(values_list)),
        "aggregation_left_neighbor_percent": float(aggregation_left_neighbor_percent(labels_list)),
        "aggregation_right_neighbor_legacy_percent": float(aggregation_right_neighbor_legacy_percent(labels_list)),
        "algotype_positions_code": label_code(labels_list),
        "values_sha256": stable_json_sha256(values_list),
        "labels_sha256": stable_json_sha256(labels_list),
    }


def records_from_probe(
    probe: EventTracingStatusProbe,
    initial_values: Sequence[int],
    initial_labels: Sequence[str],
    final_values: Sequence[int],
    final_labels: Sequence[str],
    event_count: int,
    sweep_count: int,
) -> list[dict[str, Any]]:
    rows = [make_metric_record(initial_values, initial_labels, 0, 0, 0, 0, 0)]
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
            make_metric_record(
                values_snapshot,
                labels_from_snapshot(type_snapshot),
                int(event_step),
                sweep_count,
                int(swap_count),
                int(comparison_count),
                int(frozen_attempt_count),
            )
        )
    if rows[-1]["values"] != list(final_values) or rows[-1]["labels"] != list(final_labels):
        rows.append(
            make_metric_record(
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


def dispatch_proof(cells: Sequence[Any]) -> dict[str, Any]:
    by_label: dict[str, dict[str, str]] = {}
    for cell in cells:
        label = str(cell.label)
        if label not in DUMMY_LABELS:
            continue
        cls = type(cell)
        by_label.setdefault(
            label,
            {
                "behavior": DEFAULT_LABEL_TO_BEHAVIOR.get(label, label),
                "class": f"{cls.__module__}.{cls.__qualname__}",
                "moveImplementation": f"{cls.move.__module__}.{cls.move.__qualname__}",
            },
        )
    behavior_values = {entry["behavior"] for entry in by_label.values()}
    class_values = {entry["class"] for entry in by_label.values()}
    move_values = {entry["moveImplementation"] for entry in by_label.values()}
    labels_present = set(by_label) == set(DUMMY_LABELS)
    return {
        "dispatchByLabel": by_label,
        "labelsPresent": bool(labels_present),
        "sameBehavior": bool(labels_present and behavior_values == {DUMMY_BEHAVIOR}),
        "sameClass": bool(labels_present and len(class_values) == 1),
        "sameMoveImplementation": bool(labels_present and len(move_values) == 1),
        "dispatchProven": bool(labels_present and behavior_values == {DUMMY_BEHAVIOR} and len(class_values) == 1 and len(move_values) == 1),
    }


def value_trace_sha256(records: Sequence[dict[str, Any]]) -> str:
    return stable_json_sha256([record["values"] for record in records])


def sortedness_trace_sha256(records: Sequence[dict[str, Any]]) -> str:
    return stable_json_sha256([record["sortedness_percent"] for record in records])


def label_count_mismatch_count(records: Sequence[dict[str, Any]], expected_counts: Counter[str]) -> int:
    mismatches = 0
    for record in records:
        if Counter(record["labels"]) != expected_counts:
            mismatches += 1
    return mismatches


def run_assignment(task: DummyTask, assignments: tuple[str, ...], scope: str) -> dict[str, Any]:
    rng = random.Random(task.scheduler_seed)
    random.seed(task.scheduler_seed + 17)
    probe = EventTracingStatusProbe()
    run_task = replace(task, assignments=assignments)
    cells, cell_status = build_cells(run_task, probe)
    initial_values = cell_values(cells)
    initial_labels = tuple(assignments)
    dispatch = dispatch_proof(cells)
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
        legal_action_exists = has_public_legal_action(
            cells,
            cell_status,
            DEFAULT_LABEL_TO_BEHAVIOR,
            task.condition["frozen_semantics"],
        )
        before_signature = cell_state_signature(cells)
        before_swap_count = int(probe.swap_count)
        item_kind, items = scheduler_items(cells, cell_status, "synchronous_round_ltr_resolution", rng)
        for item in items:
            cell = item if item_kind == "identity" else cells[item]
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
                "behavior": DEFAULT_LABEL_TO_BEHAVIOR.get(str(cell.label), str(cell.label)),
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
        if not made_progress and not legal_action_exists:
            no_progress_sweeps += 1
            if no_progress_sweeps >= task.stall_sweeps:
                stop_reason = "no_legal_action_window"
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
    records = records_from_probe(probe, initial_values, initial_labels, final_values, final_labels, event_count, sweep_count)
    return {
        "scope": scope,
        "records": records,
        "dispatch": dispatch,
        "final_values": tuple(final_values),
        "final_labels": tuple(final_labels),
        "event_count": int(event_count),
        "sweep_count": int(sweep_count),
        "swap_count": int(probe.swap_count),
        "comparison_count": int(probe.compare_and_swap_count),
        "frozen_attempt_count": int(probe.frozen_swap_attempts),
        "stop_reason": stop_reason,
        "max_guard_hit": bool(max_guard_hit),
        "activation_log_sha256": activation_hash.hexdigest(),
        "activation_label_counts": dict(sorted(activation_label_counts.items())),
        "activation_first_20": activation_first_20,
        "activation_last_20": activation_last_20,
        "value_trace_sha256": value_trace_sha256(records),
        "sortedness_trace_sha256": sortedness_trace_sha256(records),
        "trace_sha256": stable_json_sha256(records),
    }


def curve_points(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    final_swap_step = max(int(record["swap_step"]) for record in records) or 1
    rows = []
    for record in records:
        progress = int(np.clip(round(100.0 * int(record["swap_step"]) / final_swap_step), 0, 100))
        rows.append(
            {
                "progress_percent": progress,
                "aggregation_left_neighbor_percent": float(record["aggregation_left_neighbor_percent"]),
            }
        )
    return rows


def run_dummy_task(task: DummyTask) -> dict[str, Any]:
    start = time.monotonic()
    dummy = run_assignment(task, task.assignments, "dummy_bubble_label_control")
    pure_assignments = tuple(DUMMY_BEHAVIOR for _ in task.assignments)
    pure = run_assignment(task, pure_assignments, REFERENCE_SCOPE)
    records = dummy["records"]
    final_record = records[-1]
    sortedness_values = [float(record["sortedness_percent"]) for record in records]
    dg = dg_from_sortedness(sortedness_values)
    initial_counts = Counter(task.assignments)
    label_count_mismatches = label_count_mismatch_count(records, initial_counts)
    activation_counts = Counter(dummy["activation_label_counts"])
    activation_balance_abs_difference = abs(int(activation_counts[DUMMY_LABELS[0]]) - int(activation_counts[DUMMY_LABELS[1]]))
    value_trace_matches = dummy["value_trace_sha256"] == pure["value_trace_sha256"]
    sortedness_trace_matches = dummy["sortedness_trace_sha256"] == pure["sortedness_trace_sha256"]
    row = {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": "E02",
        "source_condition_id": task.condition["condition_id"],
        "condition_id": "E02S05C001",
        "run_family": "dummy_algotype_negative_control",
        "mode": "cell_view",
        "algorithm": "mixed",
        "algotype_mix": "dummy_same_behavior_bubble_label_a_b",
        "dummy_behavior": DUMMY_BEHAVIOR,
        "scheduler_regime": SCHEDULER_REGIME,
        "scheduler_source": SCHEDULER_SOURCE,
        "scheduler_seed": int(task.scheduler_seed),
        "repeat_index": int(task.repeat_index),
        "array_length": len(task.values),
        "initial_array_seed": int(task.initial_array_seed),
        "initial_array_sha256": stable_json_sha256(list(task.values)),
        "final_array_sha256": stable_json_sha256(list(dummy["final_values"])),
        "value_bank_id": task.condition["value_bank_id"],
        "algotype_assignment_bank_id": task.condition["algotype_assignment_bank_id"],
        "algotype_assignment_seed": task.assignment_seed,
        "initial_algotype_assignment_sha256": stable_json_sha256(list(task.assignments)),
        "initial_algotype_counts_json": count_json(task.assignments),
        "final_algotype_counts_json": count_json(dummy["final_labels"]),
        "label_count_mismatch_trace_rows": int(label_count_mismatches),
        "dummy_labels_balanced_initial": bool(len({initial_counts[label] for label in DUMMY_LABELS}) == 1),
        "dummy_labels_balanced_final": bool(Counter(dummy["final_labels"]) == initial_counts),
        "dispatch_by_label_json": json.dumps(dummy["dispatch"]["dispatchByLabel"], sort_keys=True, separators=(",", ":")),
        "dispatch_same_behavior": bool(dummy["dispatch"]["sameBehavior"]),
        "dispatch_same_class": bool(dummy["dispatch"]["sameClass"]),
        "dispatch_same_move_implementation": bool(dummy["dispatch"]["sameMoveImplementation"]),
        "dispatch_proven": bool(dummy["dispatch"]["dispatchProven"]),
        "frozen_semantics": task.condition["frozen_semantics"],
        "frozen_count": int(task.condition["frozen_count"]),
        "frozen_index_bank_id": task.condition["frozen_index_bank_id"],
        "frozen_index_seed": task.frozen_index_seed,
        "initial_frozen_indices_json": json.dumps(list(task.frozen_indices), separators=(",", ":")),
        "event_count": int(dummy["event_count"]),
        "sweep_count": int(dummy["sweep_count"]),
        "swap_only_steps": int(dummy["swap_count"]),
        "comparison_steps_observed": int(dummy["comparison_count"]),
        "compare_plus_swap_steps": int(dummy["swap_count"] + dummy["comparison_count"]),
        "frozen_attempt_count": int(dummy["frozen_attempt_count"]),
        "initial_sortedness_percent": sortedness_percent(task.values),
        "final_sortedness_percent": sortedness_percent(dummy["final_values"]),
        "peak_sortedness_percent": max(sortedness_values),
        "final_monotonicity_error_count": monotonicity_error_count(dummy["final_values"]),
        "initial_aggregation_left_neighbor_percent": float(records[0]["aggregation_left_neighbor_percent"]),
        "final_aggregation_left_neighbor_percent": float(final_record["aggregation_left_neighbor_percent"]),
        "final_aggregation_right_neighbor_legacy_percent": float(final_record["aggregation_right_neighbor_legacy_percent"]),
        "peak_aggregation_left_neighbor_percent": max(float(record["aggregation_left_neighbor_percent"]) for record in records),
        "curve_mean_aggregation_left_neighbor_percent": float(np.mean([record["aggregation_left_neighbor_percent"] for record in records])),
        "dg_primary": dg["dg_primary"],
        "dg_event_count": dg["dg_event_count"],
        "dg_total_drop": dg["dg_total_drop"],
        "dg_total_recovery": dg["dg_total_recovery"],
        "stop_reason": dummy["stop_reason"],
        "max_guard_hit": bool(dummy["max_guard_hit"]),
        "elapsed_seconds": time.monotonic() - start,
        "activation_event_count": int(dummy["event_count"]),
        "activation_log_sha256": dummy["activation_log_sha256"],
        "activation_label_counts_json": json.dumps(dummy["activation_label_counts"], sort_keys=True, separators=(",", ":")),
        "activation_label_balance_abs_difference": int(activation_balance_abs_difference),
        "activation_first_20_json": json.dumps(dummy["activation_first_20"], separators=(",", ":")),
        "activation_last_20_json": json.dumps(dummy["activation_last_20"], separators=(",", ":")),
        "trace_record_count": len(records),
        "trace_sha256": dummy["trace_sha256"],
        "value_trace_sha256": dummy["value_trace_sha256"],
        "sortedness_trace_sha256": dummy["sortedness_trace_sha256"],
        "paired_reference_scope": REFERENCE_SCOPE,
        "paired_reference_value_trace_sha256": pure["value_trace_sha256"],
        "paired_reference_sortedness_trace_sha256": pure["sortedness_trace_sha256"],
        "paired_reference_final_array_sha256": stable_json_sha256(list(pure["final_values"])),
        "paired_reference_event_count": int(pure["event_count"]),
        "paired_reference_sweep_count": int(pure["sweep_count"]),
        "paired_reference_swap_only_steps": int(pure["swap_count"]),
        "paired_reference_comparison_steps_observed": int(pure["comparison_count"]),
        "paired_reference_stop_reason": pure["stop_reason"],
        "value_trace_matches_pure_reference": bool(value_trace_matches),
        "sortedness_trace_matches_pure_reference": bool(sortedness_trace_matches),
        "result_scope": "s05_full_e01_repeat_dummy_bubble_label_control",
        "wrapper_name": "identity_round_public_bubble_dummy_label_control",
    }
    return {"row": row, "curvePoints": curve_points(records)}


def add_null_metrics(df: pd.DataFrame, row_null: np.ndarray) -> pd.DataFrame:
    rows = []
    row_mean = float(np.mean(row_null))
    row_sd = float(np.std(row_null, ddof=1))
    for record in df.to_dict(orient="records"):
        trace_count = int(record["trace_record_count"])
        observed_peak = float(record["peak_aggregation_left_neighbor_percent"])
        observed_final = float(record["final_aggregation_left_neighbor_percent"])
        observed_curve_mean = float(record["curve_mean_aggregation_left_neighbor_percent"])
        peak_stats = max_distribution_stats(row_null, trace_count, observed_peak)
        curve_sd = row_sd / math.sqrt(max(trace_count, 1))
        curve_q025, curve_q975 = normal_quantiles(row_mean, curve_sd)
        record.update(
            {
                "null_model": "label_shuffle_preserve_50_50_counts",
                "null_samples": int(len(row_null)),
                "null_row_mean_aggregation_left_neighbor_percent": row_mean,
                "null_row_sd_aggregation_left_neighbor_percent": row_sd,
                "null_final_q025": empirical_quantile(row_null, 0.025),
                "null_final_q50": empirical_quantile(row_null, 0.50),
                "null_final_q975": empirical_quantile(row_null, 0.975),
                "final_empirical_p_upper": empirical_upper_tail(row_null, observed_final),
                "curve_mean_normal_q025": curve_q025,
                "curve_mean_normal_q975": curve_q975,
                "curve_mean_normal_p_upper": normal_upper_tail(observed_curve_mean, row_mean, curve_sd),
                **peak_stats,
                "peak_minus_null_peak_mean": observed_peak - peak_stats["null_peak_mean"],
                "final_minus_null_row_mean": observed_final - row_mean,
                "curve_mean_minus_null_row_mean": observed_curve_mean - row_mean,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition_id, mix), group in df.groupby(["condition_id", "algotype_mix"], sort=True):
        rows.append(
            {
                "condition_id": condition_id,
                "source_condition_id": group["source_condition_id"].iloc[0],
                "algotype_mix": mix,
                "n_runs": int(len(group)),
                "mean_observed_peak": float(group["peak_aggregation_left_neighbor_percent"].mean()),
                "mean_null_peak_mean": float(group["null_peak_mean"].mean()),
                "mean_peak_minus_null": float(group["peak_minus_null_peak_mean"].mean()),
                "median_peak_p_upper": float(group["peak_empirical_p_upper"].median()),
                "runs_peak_p_le_0_05": int((group["peak_empirical_p_upper"] <= 0.05).sum()),
                "mean_observed_final": float(group["final_aggregation_left_neighbor_percent"].mean()),
                "mean_null_final": float(group["null_row_mean_aggregation_left_neighbor_percent"].mean()),
                "mean_final_minus_null": float(group["final_minus_null_row_mean"].mean()),
                "median_final_p_upper": float(group["final_empirical_p_upper"].median()),
                "dispatch_failures": int((~group["dispatch_proven"].astype(bool)).sum()),
                "value_trace_match_failures": int((~group["value_trace_matches_pure_reference"].astype(bool)).sum()),
                "label_count_balance_failures": int(
                    (
                        (~group["dummy_labels_balanced_initial"].astype(bool))
                        | (~group["dummy_labels_balanced_final"].astype(bool))
                        | (group["label_count_mismatch_trace_rows"].astype(int) > 0)
                    ).sum()
                ),
                "activation_balance_max_abs_difference": int(group["activation_label_balance_abs_difference"].max()),
                "max_guard_runs": int(group["max_guard_hit"].astype(bool).sum()),
                "stop_reason_counts_json": json.dumps(group["stop_reason"].value_counts(dropna=False).to_dict(), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def aggregate_curve(all_curve_points: Sequence[dict[str, Any]], row_null: np.ndarray) -> pd.DataFrame:
    row_mean = float(np.mean(row_null))
    row_sd = float(np.std(row_null, ddof=1))
    acc: dict[int, list[float]] = defaultdict(list)
    for item in all_curve_points:
        acc[int(item["progress_percent"])].append(float(item["aggregation_left_neighbor_percent"]))
    rows = []
    for progress in range(101):
        values = acc.get(progress, [])
        if not values:
            continue
        n = len(values)
        mean = float(np.mean(values))
        mean_sd = row_sd / math.sqrt(max(n, 1))
        q025, q975 = normal_quantiles(row_mean, mean_sd)
        rows.append(
            {
                "condition_id": "E02S05C001",
                "algotype_mix": "dummy_same_behavior_bubble_label_a_b",
                "progress_percent": int(progress),
                "trace_rows_in_bin": int(n),
                "observed_mean_aggregation_left_neighbor_percent": mean,
                "null_mean_aggregation_left_neighbor_percent": row_mean,
                "null_mean_q025": q025,
                "null_mean_q975": q975,
                "normal_p_upper": normal_upper_tail(mean, row_mean, mean_sd),
            }
        )
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, curve_df: pd.DataFrame, summary: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    ax = axes[0, 0]
    means = [
        float(df["peak_aggregation_left_neighbor_percent"].mean()),
        float(df["null_peak_mean"].mean()),
        float(df["final_aggregation_left_neighbor_percent"].mean()),
        float(df["null_row_mean_aggregation_left_neighbor_percent"].mean()),
    ]
    labels = ["Observed peak", "Null peak", "Observed final", "Null final"]
    ax.bar(labels, means, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_title("Dummy-label Aggregation versus null")
    ax.set_ylabel("Left-neighbor %")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    ax.hist(df["peak_empirical_p_upper"], bins=np.linspace(0, 1, 21), color="#72b7b2", edgecolor="white")
    ax.axvline(0.05, color="#555555", linestyle="--", linewidth=1)
    ax.set_title("Peak-null upper-tail p-values")
    ax.set_xlabel("p")
    ax.set_ylabel("Runs")

    ax = axes[1, 0]
    curve = curve_df.sort_values("progress_percent")
    ax.plot(curve["progress_percent"], curve["observed_mean_aggregation_left_neighbor_percent"], color="#4c78a8", label="Observed")
    ax.plot(curve["progress_percent"], curve["null_mean_aggregation_left_neighbor_percent"], color="#f58518", label="Null mean")
    ax.fill_between(curve["progress_percent"], curve["null_mean_q025"], curve["null_mean_q975"], color="#9ecae9", alpha=0.25, label="Null 95% band")
    ax.set_title("Aggregation curve")
    ax.set_xlabel("Run progress %")
    ax.set_ylabel("Mean left-neighbor %")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    activation_counts = Counter()
    for payload in df["activation_label_counts_json"]:
        activation_counts.update(json.loads(payload))
    ax.bar(list(activation_counts.keys()), list(activation_counts.values()), color=["#b279a2", "#ff9da6"])
    ax.set_title("Activation counts by dummy label")
    ax.set_ylabel("Activation events")
    max_diff = int(summary["activation_balance_max_abs_difference"].max()) if len(summary) else 0
    ax.text(0.5, 0.95, f"Max per-run difference: {max_diff}", ha="center", va="top", transform=ax.transAxes)

    fig.suptitle("E02 S05 behavior-preserving dummy Algotype controls", fontsize=14)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def validate_results(
    df: pd.DataFrame,
    tasks: Sequence[DummyTask],
    summary: pd.DataFrame,
    curve_df: pd.DataFrame,
    figure_path: Path,
    artifacts_written: Sequence[Path],
    null_samples: int,
    s04_null_path: Path,
) -> dict[str, Any]:
    expected_rows = len(tasks)
    no_duplicates = not df.duplicated(["condition_id", "repeat_index"]).any()
    dispatch_all = bool(df["dispatch_proven"].all()) if len(df) else False
    same_behavior = bool(df["dispatch_same_behavior"].all()) if len(df) else False
    same_class = bool(df["dispatch_same_class"].all()) if len(df) else False
    same_move = bool(df["dispatch_same_move_implementation"].all()) if len(df) else False
    labels_balanced = bool((df["dummy_labels_balanced_initial"] & df["dummy_labels_balanced_final"]).all()) if len(df) else False
    label_counts_preserved = bool((df["label_count_mismatch_trace_rows"].astype(int) == 0).all()) if len(df) else False
    value_trace_matches = bool(df["value_trace_matches_pure_reference"].all()) if len(df) else False
    sortedness_trace_matches = bool(df["sortedness_trace_matches_pure_reference"].all()) if len(df) else False
    activation_balanced = bool((df["activation_label_balance_abs_difference"].astype(int) == 0).all()) if len(df) else False
    all_sorted = bool((df["stop_reason"] == "sorted").all()) if len(df) else False
    no_max_guard = bool((~df["max_guard_hit"].astype(bool)).all()) if len(df) else False
    figure_ok = figure_path.exists() and figure_path.stat().st_size > 1000
    s04_null_input_exists = s04_null_path.exists() and s04_null_path.stat().st_size > 0
    materialized_names = {
        "e02_dummy_algotype_controls.parquet",
        "dummy_algotype_aggregation.png",
        "e02_dummy_algotype_curve_summary.csv",
        "e02_dummy_algotype_condition_summary.csv",
    }
    artifacts_present = all(
        path.exists() and path.stat().st_size > 0 for path in artifacts_written if path.name in materialized_names
    )
    median_peak_p = float(summary["median_peak_p_upper"].iloc[0]) if len(summary) else np.nan
    dummy_at_chance = bool(np.isfinite(median_peak_p) and median_peak_p > 0.05)
    passed = all(
        [
            len(df) == expected_rows,
            expected_rows > 0,
            no_duplicates,
            dispatch_all,
            same_behavior,
            same_class,
            same_move,
            labels_balanced,
            label_counts_preserved,
            value_trace_matches,
            sortedness_trace_matches,
            activation_balanced,
            all_sorted,
            no_max_guard,
            len(summary) == 1,
            len(curve_df) > 0,
            null_samples > 0,
            s04_null_input_exists,
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
            "S05 reruns the Bubble same-code label control only; it does not test Insertion or Selection dummy labels.",
            "The deterministic identity-round scheduler is a controlled actual rerun, not an OS-thread replay.",
        ],
        "recommendedNextAction": "Proceed to S06 speed-matched Algotypes after Chief Scientist instruction; do not start S06 inside S05.",
        "expectedRows": int(expected_rows),
        "observedRows": int(len(df)),
        "observedConditions": sorted(df["condition_id"].unique()) if len(df) else [],
        "sourceConditionId": CONTROL_CONDITION_ID,
        "dispatchProvenForAllRows": dispatch_all,
        "dispatchSameBehaviorForAllRows": same_behavior,
        "dispatchSameClassForAllRows": same_class,
        "dispatchSameMoveImplementationForAllRows": same_move,
        "balancedLabelCountsForAllRows": labels_balanced,
        "labelCountsPreservedAcrossTraceRows": label_counts_preserved,
        "valueTraceMatchesPureReferenceForAllRows": value_trace_matches,
        "sortednessTraceMatchesPureReferenceForAllRows": sortedness_trace_matches,
        "activationLabelCountsBalancedForAllRows": activation_balanced,
        "allRunsSorted": all_sorted,
        "noMaxGuardRuns": no_max_guard,
        "medianPeakPUpper": median_peak_p,
        "dummyAtChanceByMedianPeakP": dummy_at_chance,
        "nullSamples": int(null_samples),
        "s04NullInputExists": bool(s04_null_input_exists),
        "s04NullInputPath": str(s04_null_path),
        "summaryRows": int(len(summary)),
        "curveRows": int(len(curve_df)),
        "figureExistsAndNonempty": bool(figure_ok),
        "artifactsPresent": bool(artifacts_present),
        "noDuplicateTaskRows": bool(no_duplicates),
    }


def build_report(
    *,
    generated_at: str,
    artifacts_dir: Path,
    result_path: Path,
    curve_path: Path,
    summary_path: Path,
    validation_path: Path,
    log_path: Path,
    figure_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    validation: dict[str, Any],
    summary: pd.DataFrame,
    unit_result: dict[str, Any] | None,
    git_commit: str,
    git_status: str,
    max_repeats: int,
    workers: int,
    null_samples: int,
    elapsed_seconds: float,
) -> str:
    validation_result = (
        f"passed: {validation['observedRows']}/{validation['expectedRows']} dummy rerun rows, "
        f"identical dispatch={validation['dispatchProvenForAllRows']}, "
        f"value traces match pure Bubble references={validation['valueTraceMatchesPureReferenceForAllRows']}, "
        f"balanced labels={validation['balancedLabelCountsForAllRows']}, "
        f"unit tests={'passed' if unit_result and unit_result['success'] else 'not run'}"
    )
    median_peak_p = float(summary["median_peak_p_upper"].iloc[0]) if len(summary) else np.nan
    outcome = "Supportive" if validation.get("dummyAtChanceByMedianPeakP") else "Constraining/contradictory"
    lay_summary = (
        "Distinct dummy labels that executed identical Bubble code stayed compatible with chance-level label clustering in this actual rerun, "
        "so S05 did not find evidence that the Aggregation metric alone creates above-null clustering for identical behavior."
        if outcome == "Supportive"
        else "The dummy-label rerun did not remain clearly at chance or failed a validation requirement, so the same-code negative-control assumption needs review before S06."
    )
    compact_summary = summary[
        [
            "condition_id",
            "source_condition_id",
            "algotype_mix",
            "n_runs",
            "mean_observed_peak",
            "mean_null_peak_mean",
            "mean_peak_minus_null",
            "median_peak_p_upper",
            "runs_peak_p_le_0_05",
            "mean_observed_final",
            "mean_null_final",
            "mean_final_minus_null",
            "dispatch_failures",
            "value_trace_match_failures",
            "label_count_balance_failures",
        ]
    ]
    caveats = [
        "S05 tests the E01 Bubble same-code label control, not dummy labels for every algorithm family.",
        "The controlled identity-round scheduler is deterministic and balanced, but it is not a public OS-thread replay.",
        "Null p-values are computational Aggregation proxies and do not establish biological adhesion or recognition mechanisms.",
    ]
    return f"""# E02 S05 Behavior-Preserving Dummy Algotypes

## Top Summary

- Research step ID: S05
- Completion status: Completed on {generated_at}
- Artifacts written: `{artifacts_dir / 'research_steps/S05/research_step_full_results.md'}`, `{result_path}`, `{figure_path}`, `{validation_path}`, `{curve_path}`, `{summary_path}`, `{log_path}`, `{manifest_path}`, `{src_manifest_path}`
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers: {'; '.join(caveats)}
- Lay summary: {lay_summary}
- Recommended next action: Proceed to S06 speed-matched Algotypes after Chief Scientist instruction; do not start S06 inside S05.

## Chief Handoff

S05 completed an actual rerun negative control for behavior-preserving dummy Algotypes. It used the E01 same-code Bubble label-control assignment bank, instantiated two distinct labels, proved both labels dispatched to the same public BubbleSortCell `move()` implementation, and paired every dummy run with an all-Bubble reference using the same values and activation order.

## Frozen Question

If two labels execute identical code, does Aggregation remain at chance, as required for the metric and implementation to be trusted?

## Inputs

- Repository commit at S05 run time: `{git_commit}`
- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- Source E01 condition: `{CONTROL_CONDITION_ID}` same-algorithm Bubble label control
- S04 label-shuffle result table: `/artifacts/results/e02_label_shuffle_aggregation_nulls.parquet`
- Repeats requested: `{max_repeats}`
- Null samples: `{null_samples}`
- Worker count: `{workers}`

## Methods

S05 selected E01 condition `{CONTROL_CONDITION_ID}`, which assigns 50 `bubble_label_a` cells and 50 `bubble_label_b` cells while mapping both labels to Bubble behavior. Each run used the E01 initial value array and E01 dummy-label assignment for that repeat.

The actual rerun scheduler was a deterministic identity-per-cell round: at the start of each sweep, every active cell object was scheduled once in left-to-right order. This makes activation counts exactly balanced between the two 50/50 dummy labels and avoids any activation rule that keys on the label name.

For every dummy run, S05 also ran an all-`bubble` reference with the same initial values, same public BubbleSortCell implementation, same scheduler, and same activation order. Exact value-trajectory and Sortedness-trajectory hashes between dummy and pure reference runs were required as runtime evidence that labels did not alter behavior.

Aggregation was compared with a 100,000-sample 50/50 label-shuffle null, using the same peak-over-trajectory and final-row null calculations as S04.

## Commands

- Unit tests: `{unit_result['command'] if unit_result else 'not run'}`
- S05 production run: `{sys.executable} scripts/e02_s05_dummy_algotypes.py --repo-dir /workspace/cell-research --artifacts-dir {artifacts_dir} --max-repeats {max_repeats} --workers {workers} --null-samples {null_samples}`

## Dependencies And Parameters

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- New dependencies installed: none
- CPU workers: `{workers}`
- Elapsed production time: `{elapsed_seconds:.2f}` seconds

## Results

### Dummy-Control Summary

{markdown_table(compact_summary, max_rows=10)}

The median peak upper-tail p-value was `{median_peak_p:.6g}`. The full run-level table is `{result_path}`. The curve summary is `{curve_path}` and the figure is `{figure_path}`.

## Validation

Validation required all planned rerun rows, no duplicate repeat rows, balanced 50/50 dummy labels at initialization and final state, preserved label counts across trace records, identical dispatch to Bubble behavior, identical public class and `move()` implementation for both dummy labels, exact value-trajectory and Sortedness-trajectory matches to paired all-Bubble references, exactly balanced activation label counts, sorted completion without max guards, nonempty summaries, nonempty figure, and present artifacts. The validation result was `{validation['validationPassed']}`.

Validation details were written to `{validation_path}`.

## Artifacts

- Run-level dummy-control table: `{result_path}`
- Curve-level summary: `{curve_path}`
- Condition summary: `{summary_path}`
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

- S05 proves identical dispatch for the Bubble dummy labels already present in the E01 baseline config. It does not yet prove identical dummy-label controls for Insertion or Selection.
- The deterministic identity-round scheduler is an actual rerun of public cell methods, but not an OS-thread replay.
- Aggregation p-values remain computational metric-null evidence. They do not imply a biological mechanism.

## Recommended Next Action

Proceed to S06 speed-matched Algotypes after explicit Chief Scientist instruction. S06 should test whether aggregation in mixed-policy chimeras is speed-explained, speed-amplified, or speed-independent.
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
        "recommendedNextAction": "Resolve S05 dummy-label dispatch or simulator-support blocker before S06; do not start S06.",
    }
    write_json(validation_path, blocker)
    log_path.write_text(json.dumps(blocker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    src_manifest = {
        "schema": "eidosoma.src_snapshot.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "gitCommit": git_commit,
        "gitStatusShort": git_status,
        "sourceFiles": [],
        "validation": blocker,
    }
    write_json(src_manifest_path, src_manifest)
    report = f"""# E02 S05 Behavior-Preserving Dummy Algotypes

## Top Summary

- Research step ID: S05
- Completion status: Blocked on {generated_at}
- Artifacts written: `{report_path}`, `{validation_path}`, `{manifest_path}`, `{src_manifest_path}`, `{log_path}`
- Validation result: blocked
- Outcome classification: constraining/contradictory
- Caveats or blockers: {reason}
- Lay summary: S05 could not safely run the dummy-label negative control because identical dispatch or simulator support could not be established.
- Recommended next action: Resolve the S05 blocker before S06; do not start S06.

## Frozen Question

If two labels execute identical code, does Aggregation remain at chance, as required for the metric and implementation to be trusted?

## Methods

S05 inspected the E01 control configuration and simulator support before running production simulations.

## Results

No dummy-control result table or figure was generated.

## Validation

Validation was blocked before production reruns.

## Provenance

- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Generated at UTC: `{generated_at}`
"""
    report_path.write_text(report, encoding="utf-8")
    artifact_stub = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "artifacts": [
            artifact_entry(report_path, artifacts_dir, "S05 blocked full-results Markdown handoff report."),
            artifact_entry(validation_path, artifacts_dir, "S05 blocked validation evidence."),
            artifact_entry(log_path, artifacts_dir, "S05 blocked execution log."),
            artifact_entry(src_manifest_path, artifacts_dir, "S05 blocked source snapshot manifest."),
        ],
    }
    write_json(manifest_path, artifact_stub)
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
    result_path = results_dir / "e02_dummy_algotype_controls.parquet"
    curve_path = step_dir / "e02_dummy_algotype_curve_summary.csv"
    summary_path = step_dir / "e02_dummy_algotype_condition_summary.csv"
    figure_path = figures_dir / "dummy_algotype_aggregation.png"
    validation_path = step_dir / "s05_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s05_dummy_algotypes_manifest.json"
    log_path = logs_dir / "e02_s05_dummy_algotypes.log"
    report_path = step_dir / "research_step_full_results.md"
    log_lines = [f"E02 S05 started {generated_at}", f"repo_dir={repo_dir}", f"artifacts_dir={artifacts_dir}"]

    if any(DEFAULT_LABEL_TO_BEHAVIOR.get(label) != DUMMY_BEHAVIOR for label in DUMMY_LABELS):
        return write_blocked_outputs(
            generated_at=generated_at,
            artifacts_dir=artifacts_dir,
            repo_dir=repo_dir,
            reason=f"DEFAULT_LABEL_TO_BEHAVIOR does not map {DUMMY_LABELS} to {DUMMY_BEHAVIOR}.",
            validation_path=validation_path,
            report_path=report_path,
            manifest_path=manifest_path,
            src_manifest_path=src_manifest_path,
            log_path=log_path,
        )

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
        condition = control_condition(cfg)
        tasks = build_tasks(
            cfg,
            condition,
            repo_dir,
            args.max_repeats,
            args.max_sweeps,
            args.max_successful_swaps,
            args.stall_sweeps,
        )
    except Exception as exc:
        return write_blocked_outputs(
            generated_at=generated_at,
            artifacts_dir=artifacts_dir,
            repo_dir=repo_dir,
            reason=f"S05 input or simulator-support validation failed before rerun: {exc!r}",
            validation_path=validation_path,
            report_path=report_path,
            manifest_path=manifest_path,
            src_manifest_path=src_manifest_path,
            log_path=log_path,
        )

    workers = max(1, min(int(args.workers), len(tasks), 8))
    if workers == 1:
        outputs = [run_dummy_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(run_dummy_task, tasks))
    run_rows = [output["row"] for output in outputs]
    curve_points_all = [item for output in outputs for item in output["curvePoints"]]
    row_null = simulate_row_null_distribution((50, 50), args.null_samples, args.random_seed)
    result_df = add_null_metrics(pd.DataFrame(run_rows).sort_values("repeat_index").reset_index(drop=True), row_null)
    curve_df = aggregate_curve(curve_points_all, row_null)
    summary_df = summarize_results(result_df)

    result_df.to_parquet(result_path, index=False)
    curve_df.to_csv(curve_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    make_figure(result_df, curve_df, summary_df, figure_path)
    expected_artifacts = [
        report_path,
        result_path,
        figure_path,
        validation_path,
        curve_path,
        summary_path,
        log_path,
        manifest_path,
        src_manifest_path,
    ]
    validation = validate_results(
        result_df,
        tasks,
        summary_df,
        curve_df,
        figure_path,
        expected_artifacts,
        args.null_samples,
        args.s04_null_path,
    )
    write_json(validation_path, validation)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    elapsed_seconds = time.monotonic() - start
    log_lines.extend(
        [
            f"tasks={len(tasks)} workers={workers} rows={len(result_df)} null_samples={args.null_samples}",
            f"validation={json.dumps(validation, sort_keys=True)}",
            f"elapsed_seconds={elapsed_seconds:.3f}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    code_files = [
        repo_dir / "scripts/e02_s05_dummy_algotypes.py",
        repo_dir / "tests/e02/test_dummy_algotypes.py",
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
            "configPath": str(args.config_path),
            "s04NullPath": str(args.s04_null_path),
            "maxRepeats": int(args.max_repeats),
            "workers": int(workers),
            "nullSamples": int(args.null_samples),
            "randomSeed": int(args.random_seed),
            "schedulerRegime": SCHEDULER_REGIME,
            "controlConditionId": CONTROL_CONDITION_ID,
            "dummyLabels": list(DUMMY_LABELS),
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
        curve_path=curve_path,
        summary_path=summary_path,
        validation_path=validation_path,
        log_path=log_path,
        figure_path=figure_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        validation=validation,
        summary=summary_df,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        max_repeats=args.max_repeats,
        workers=workers,
        null_samples=args.null_samples,
        elapsed_seconds=elapsed_seconds,
    )
    report_path.write_text(report_text, encoding="utf-8")
    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S05 full-results Markdown handoff report."),
        artifact_entry(result_path, artifacts_dir, "Run-level dummy-Algotype negative-control table."),
        artifact_entry(curve_path, artifacts_dir, "Curve-level dummy-Algotype summary."),
        artifact_entry(summary_path, artifacts_dir, "Condition-level dummy-Algotype summary."),
        artifact_entry(figure_path, artifacts_dir, "Dummy-Algotype Aggregation figure."),
        artifact_entry(validation_path, artifacts_dir, "S05 validation evidence."),
        artifact_entry(log_path, artifacts_dir, "S05 execution log."),
        artifact_entry(src_manifest_path, artifacts_dir, "S05 source snapshot manifest."),
    ]
    write_json(manifest_path, artifact_stub | {"artifacts": artifacts})
    if not validation["validationPassed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
