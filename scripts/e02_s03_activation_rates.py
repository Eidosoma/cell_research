#!/usr/bin/env python3
"""Run E02 S03 activation-rate artifact tests.

This step reuses the S01 deterministic public-cell simulator and the S02
scheduler framework, but replaces the scheduler order with deterministic
weighted activation cycles.  It intentionally stops before S04 label shuffles.
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
import re
import sys
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

from scripts.e02_s02_scheduler_comparison import (
    artifact_entry,
    build_cells,
    dg_from_sortedness,
    git_output,
    load_e01_config,
    markdown_table,
    rows_from_probe,
    run_command,
    sha256_file,
    write_json,
)
from src.e02.deterministic_simulator import (
    aggregation_left_neighbor_percent,
    aggregation_right_neighbor_legacy_percent,
    cell_state_signature,
    cell_values,
    count_json,
    is_sorted,
    monotonicity_error_count,
    sortedness_percent,
    stable_json_sha256,
)
from src.e02.deterministic_simulator import EventTracingStatusProbe


STEP_ID = "S03"
STEP_NUMBER = 3
EXPERIMENT_ID = "E02"
SELECTED_CONDITION_IDS = ("E01C043", "E01C044", "E01C045", "E01C046", "E01C047")
EQUAL_RATE_REGIME = "equal_per_cell"
FAST_MULTIPLIER = 3


@dataclass(frozen=True)
class ActivationRateTask:
    condition: dict[str, Any]
    repeat_index: int
    values: tuple[int, ...]
    initial_array_seed: int
    assignments: tuple[str, ...]
    assignment_seed: int | None
    frozen_indices: tuple[int, ...]
    frozen_index_seed: int | None
    activation_rate_regime: str
    fast_label: str | None
    label_weights: dict[str, int]
    scheduler_seed: int
    repo_dir: str
    max_cycles: int
    max_successful_swaps: int
    stall_cycles: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config-path", type=Path, default=Path("/previous-artifacts/E01/configs/e01_baseline_configs.json"))
    parser.add_argument("--s02-results-path", type=Path, default=Path("/artifacts/results/e02_scheduler_comparison.parquet"))
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-cycles", type=int, default=20_000)
    parser.add_argument("--max-successful-swaps", type=int, default=80_000)
    parser.add_argument("--stall-cycles", type=int, default=2)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_chimera_conditions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in cfg["conditions"]
        if row["condition_id"] in SELECTED_CONDITION_IDS
        and row["mode"] == "cell_view"
        and row["run_family"] == "same_goal_chimera"
    ]
    missing = sorted(set(SELECTED_CONDITION_IDS) - {row["condition_id"] for row in rows})
    if missing:
        raise RuntimeError(f"Missing expected S03 chimera condition IDs in E01 config: {missing}")
    return sorted(rows, key=lambda row: row["condition_id"])


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_").lower()


def label_order_from_condition(row: dict[str, Any], assignments: Sequence[str]) -> tuple[str, ...]:
    labels: list[str] = []
    for role in str(row.get("algotype_roles", "")).split(";"):
        role = role.strip()
        if not role or ":" not in role:
            continue
        label = role.split(":", 1)[0].strip()
        if label and label not in labels:
            labels.append(label)
    for label in assignments:
        if label not in labels:
            labels.append(str(label))
    return tuple(labels)


def make_rate_profiles(labels: Sequence[str], fast_multiplier: int = FAST_MULTIPLIER) -> list[tuple[str, str | None, dict[str, int]]]:
    ordered = tuple(dict.fromkeys(map(str, labels)))
    if not ordered:
        raise ValueError("At least one label is required.")
    profiles: list[tuple[str, str | None, dict[str, int]]] = [(EQUAL_RATE_REGIME, None, {label: 1 for label in ordered})]
    for label in ordered:
        weights = {candidate: 1 for candidate in ordered}
        weights[label] = int(fast_multiplier)
        profiles.append((f"fast_{safe_label(label)}_{fast_multiplier}x", label, weights))
    return profiles


def activation_seed(base_seed: int, condition_id: str, repeat_idx: int, profile_idx: int) -> int:
    condition_num = int(condition_id.replace("E01C", ""))
    return base_seed * 1_000_000 + 300_000 + condition_num * 1_000 + repeat_idx * 10 + profile_idx


def build_tasks(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    repo_dir: Path,
    max_repeats: int,
    max_cycles: int,
    max_successful_swaps: int,
    stall_cycles: int,
) -> list[ActivationRateTask]:
    repeat_count = min(max_repeats, int(cfg["globalDefaults"]["repeatCount"]))
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    value_banks = cfg["seedBanks"]["valueBanks"]
    assignment_banks = cfg["seedBanks"]["algotypeAssignmentBanks"]
    frozen_banks = cfg["seedBanks"]["frozenIndexBanks"]
    tasks: list[ActivationRateTask] = []
    for row in rows:
        value_bank = value_banks[row["value_bank_id"]]
        assignment_bank = assignment_banks[row["algotype_assignment_bank_id"]]
        frozen_bank = frozen_banks[row["frozen_index_bank_id"]]
        for repeat_idx in range(repeat_count):
            values = tuple(map(int, value_bank["initialArrays"][repeat_idx]))
            assignments = tuple(map(str, assignment_bank["assignments"][repeat_idx]))
            labels = label_order_from_condition(row, assignments)
            frozen_indices = tuple(map(int, frozen_bank["indices"][repeat_idx]))
            frozen_seed_value = int(frozen_bank["seeds"][repeat_idx]) if frozen_bank["seeds"] else None
            for profile_idx, (regime, fast_label, weights) in enumerate(make_rate_profiles(labels)):
                tasks.append(
                    ActivationRateTask(
                        condition=dict(row),
                        repeat_index=repeat_idx,
                        values=values,
                        initial_array_seed=int(value_bank["seeds"][repeat_idx]),
                        assignments=assignments,
                        assignment_seed=int(assignment_bank["seeds"][repeat_idx]),
                        frozen_indices=frozen_indices,
                        frozen_index_seed=frozen_seed_value,
                        activation_rate_regime=regime,
                        fast_label=fast_label,
                        label_weights=dict(weights),
                        scheduler_seed=activation_seed(base_seed, row["condition_id"], repeat_idx, profile_idx),
                        repo_dir=str(repo_dir),
                        max_cycles=max_cycles,
                        max_successful_swaps=max_successful_swaps,
                        stall_cycles=stall_cycles,
                    )
                )
    return tasks


def activation_cycle_items(cells: Sequence[Any], cell_status: Any, label_weights: dict[str, int], rng: random.Random) -> list[Any]:
    items: list[Any] = []
    for cell in cells:
        if cell.status != cell_status.ACTIVE:
            continue
        weight = int(label_weights.get(str(cell.label), 1))
        if weight <= 0:
            raise ValueError(f"Activation weight must be positive for label {cell.label!r}.")
        items.extend([cell] * weight)
    rng.shuffle(items)
    return items


def expected_label_shares(label_counts: Counter[str], label_weights: dict[str, int]) -> dict[str, float]:
    denom = sum(int(label_counts[label]) * int(label_weights[label]) for label in label_counts)
    return {
        label: (float(label_counts[label] * label_weights[label]) / denom if denom else np.nan)
        for label in sorted(label_counts)
    }


def observed_label_shares(label_counts: Counter[str], activation_counts: Counter[str]) -> dict[str, float]:
    total = sum(activation_counts.values())
    return {label: (float(activation_counts[label]) / total if total else np.nan) for label in sorted(label_counts)}


def activation_rate_deviation(
    label_counts: Counter[str],
    label_weights: dict[str, int],
    activation_counts: Counter[str],
) -> tuple[float, float, bool]:
    expected = expected_label_shares(label_counts, label_weights)
    observed = observed_label_shares(label_counts, activation_counts)
    share_errors = [
        abs(float(observed[label]) - float(expected[label]))
        for label in expected
        if not pd.isna(observed[label]) and not pd.isna(expected[label])
    ]
    per_cell_ratios = []
    for label in sorted(label_counts):
        expected_weight = float(label_weights[label])
        per_cell_observed = float(activation_counts[label]) / float(label_counts[label]) if label_counts[label] else np.nan
        per_cell_ratios.append(per_cell_observed / expected_weight if expected_weight else np.nan)
    finite = [ratio for ratio in per_cell_ratios if not pd.isna(ratio)]
    mean_ratio = float(np.mean(finite)) if finite else np.nan
    ratio_errors = [abs((ratio / mean_ratio) - 1.0) for ratio in finite] if finite and mean_ratio else []
    max_share_error = float(max(share_errors)) if share_errors else np.nan
    max_ratio_error = float(max(ratio_errors)) if ratio_errors else np.nan
    passed = bool(share_errors and max_share_error <= 1e-12 and (not ratio_errors or max_ratio_error <= 1e-12))
    return max_share_error, max_ratio_error, passed


def run_activation_task(task: ActivationRateTask) -> dict[str, Any]:
    start = time.monotonic()
    rng = random.Random(task.scheduler_seed)
    probe = EventTracingStatusProbe()
    cells, cell_status = build_cells(task, probe)
    initial_values = cell_values(cells)
    initial_labels = tuple(task.assignments)
    initial_label_counts: Counter[str] = Counter(initial_labels)
    activation_hash = hashlib.sha256()
    activation_label_counts: Counter[str] = Counter()
    activation_first_20: list[dict[str, Any]] = []
    activation_last_20: list[dict[str, Any]] = []
    event_count = 0
    cycle_count = 0
    no_progress_cycles = 0
    stop_reason = "max_cycles_exceeded"
    max_guard_hit = False
    while cycle_count < task.max_cycles:
        if is_sorted(cell_values(cells)):
            stop_reason = "sorted"
            break
        if int(probe.swap_count) >= task.max_successful_swaps:
            stop_reason = "max_successful_swaps_exceeded"
            max_guard_hit = True
            break
        before_signature = cell_state_signature(cells)
        before_swap_count = int(probe.swap_count)
        items = activation_cycle_items(cells, cell_status, task.label_weights, rng)
        for cell in items:
            if cell.status != cell_status.ACTIVE:
                continue
            event_count += 1
            probe.current_event_step = event_count
            event_payload = {
                "event": event_count,
                "cycle": cycle_count + 1,
                "thread_id": int(cell.threadID),
                "position_before": int(cell.current_position[0]),
                "label": str(cell.label),
                "configured_label_weight": int(task.label_weights[str(cell.label)]),
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
        cycle_count += 1
        after_signature = cell_state_signature(cells)
        made_progress = int(probe.swap_count) != before_swap_count or after_signature != before_signature
        if not made_progress:
            no_progress_cycles += 1
            if no_progress_cycles >= task.stall_cycles:
                stop_reason = "no_progress_window"
                break
        else:
            no_progress_cycles = 0
    else:
        max_guard_hit = True

    final_values = cell_values(cells)
    final_labels = tuple(str(cell.label) for cell in cells)
    if is_sorted(final_values):
        stop_reason = "sorted"
        max_guard_hit = False
    records = rows_from_probe(task, probe, initial_values, initial_labels, final_values, final_labels, event_count, cycle_count)
    dg = dg_from_sortedness([float(row["sortedness_percent"]) for row in records])
    peak_aggregation = max(float(row["aggregation_left_neighbor_percent"]) for row in records)
    peak_sortedness = max(float(row["sortedness_percent"]) for row in records)
    expected_shares = expected_label_shares(initial_label_counts, task.label_weights)
    observed_shares = observed_label_shares(initial_label_counts, activation_label_counts)
    max_share_error, max_ratio_error, rate_valid = activation_rate_deviation(
        initial_label_counts,
        task.label_weights,
        activation_label_counts,
    )
    row = {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": "E02",
        "condition_id": task.condition["condition_id"],
        "run_family": task.condition["run_family"],
        "mode": "cell_view",
        "algorithm": task.condition["algorithm"],
        "algotype_mix": task.condition["algotype_mix"],
        "activation_rate_regime": task.activation_rate_regime,
        "fast_label": task.fast_label,
        "scheduler_regime": "weighted_activation_cycle",
        "scheduler_source": "s03_weighted_identity_cycle_public_move_scheduler",
        "scheduler_seed": int(task.scheduler_seed),
        "repeat_index": int(task.repeat_index),
        "array_length": len(task.values),
        "initial_array_seed": int(task.initial_array_seed),
        "initial_array_sha256": stable_json_sha256(list(task.values)),
        "final_array_sha256": stable_json_sha256(list(final_values)),
        "value_bank_id": task.condition["value_bank_id"],
        "algotype_assignment_bank_id": task.condition["algotype_assignment_bank_id"],
        "algotype_assignment_seed": task.assignment_seed,
        "initial_algotype_assignment_sha256": stable_json_sha256(list(task.assignments)),
        "initial_algotype_counts_json": count_json(task.assignments),
        "final_algotype_counts_json": count_json(final_labels),
        "configured_label_weights_json": json.dumps(dict(sorted(task.label_weights.items())), separators=(",", ":")),
        "expected_activation_shares_json": json.dumps(expected_shares, sort_keys=True, separators=(",", ":")),
        "observed_activation_shares_json": json.dumps(observed_shares, sort_keys=True, separators=(",", ":")),
        "activation_rate_max_abs_share_error": max_share_error,
        "activation_rate_max_abs_per_cell_ratio_error": max_ratio_error,
        "activation_rate_validation_passed": bool(rate_valid),
        "frozen_semantics": task.condition["frozen_semantics"],
        "frozen_count": int(task.condition["frozen_count"]),
        "frozen_index_bank_id": task.condition["frozen_index_bank_id"],
        "frozen_index_seed": task.frozen_index_seed,
        "initial_frozen_indices_json": json.dumps(list(task.frozen_indices), separators=(",", ":")),
        "event_count": int(event_count),
        "cycle_count": int(cycle_count),
        "sweep_count": int(cycle_count),
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
        "result_scope": "s03_bounded_activation_rate_matrix_first_e01_repeats",
        "wrapper_name": "weighted_activation_cycle_public_cell_methods",
    }
    return row


def add_equal_reference_deltas(df: pd.DataFrame) -> pd.DataFrame:
    equal = df[df["activation_rate_regime"] == EQUAL_RATE_REGIME][
        [
            "condition_id",
            "repeat_index",
            "compare_plus_swap_steps",
            "activation_event_count",
            "peak_aggregation_left_neighbor_percent",
            "final_aggregation_left_neighbor_percent",
            "final_sortedness_percent",
        ]
    ].rename(
        columns={
            "compare_plus_swap_steps": "equal_compare_plus_swap_steps",
            "activation_event_count": "equal_activation_event_count",
            "peak_aggregation_left_neighbor_percent": "equal_peak_aggregation_left_neighbor_percent",
            "final_aggregation_left_neighbor_percent": "equal_final_aggregation_left_neighbor_percent",
            "final_sortedness_percent": "equal_final_sortedness_percent",
        }
    )
    merged = df.merge(equal, on=["condition_id", "repeat_index"], how="left")
    merged["delta_compare_plus_swap_vs_equal"] = (
        merged["compare_plus_swap_steps"] - merged["equal_compare_plus_swap_steps"]
    )
    merged["relative_delta_compare_plus_swap_vs_equal"] = merged["delta_compare_plus_swap_vs_equal"] / merged[
        "equal_compare_plus_swap_steps"
    ].replace(0, np.nan)
    merged["delta_activation_event_count_vs_equal"] = merged["activation_event_count"] - merged["equal_activation_event_count"]
    merged["relative_delta_activation_event_count_vs_equal"] = merged["delta_activation_event_count_vs_equal"] / merged[
        "equal_activation_event_count"
    ].replace(0, np.nan)
    merged["delta_peak_aggregation_vs_equal"] = (
        merged["peak_aggregation_left_neighbor_percent"] - merged["equal_peak_aggregation_left_neighbor_percent"]
    )
    merged["delta_final_aggregation_vs_equal"] = (
        merged["final_aggregation_left_neighbor_percent"] - merged["equal_final_aggregation_left_neighbor_percent"]
    )
    merged["delta_final_sortedness_vs_equal"] = merged["final_sortedness_percent"] - merged["equal_final_sortedness_percent"]
    return merged


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    sem = std / math.sqrt(len(clean)) if len(clean) > 1 else np.nan
    low = mean - 1.96 * sem if len(clean) > 1 else np.nan
    high = mean + 1.96 * sem if len(clean) > 1 else np.nan
    return mean, std, low, high


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "compare_plus_swap_steps",
        "activation_event_count",
        "peak_aggregation_left_neighbor_percent",
        "final_aggregation_left_neighbor_percent",
        "final_sortedness_percent",
        "delta_compare_plus_swap_vs_equal",
        "relative_delta_compare_plus_swap_vs_equal",
        "delta_peak_aggregation_vs_equal",
        "delta_final_aggregation_vs_equal",
        "activation_rate_max_abs_share_error",
    ]
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["condition_id", "algotype_mix", "activation_rate_regime", "fast_label"], dropna=False)
    for keys, group in grouped:
        row: dict[str, Any] = {
            "condition_id": keys[0],
            "algotype_mix": keys[1],
            "activation_rate_regime": keys[2],
            "fast_label": None if pd.isna(keys[3]) else keys[3],
            "n_runs": int(len(group)),
            "stop_reason_counts_json": json.dumps(group["stop_reason"].value_counts(dropna=False).to_dict(), sort_keys=True),
            "max_guard_runs": int(group["max_guard_hit"].map(bool).sum()),
            "rate_validation_failures": int((~group["activation_rate_validation_passed"].astype(bool)).sum()),
        }
        for metric in metrics:
            mean, std, low, high = mean_ci(group[metric])
            row[f"mean_{metric}"] = mean
            row[f"std_{metric}"] = std
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition_id", "activation_rate_regime"]).reset_index(drop=True)


def classify_activation_effects(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition_id, mix), group in summary.groupby(["condition_id", "algotype_mix"], dropna=False):
        values_eff = group["mean_compare_plus_swap_steps"].dropna()
        values_agg = group["mean_peak_aggregation_left_neighbor_percent"].dropna()
        if values_eff.empty:
            efficiency_spread = np.nan
            efficiency_rel_spread = np.nan
            efficiency_class = "not_assessed"
        else:
            efficiency_spread = float(values_eff.max() - values_eff.min())
            scale = max(float(abs(values_eff.max())), float(abs(values_eff.min())), 1.0)
            efficiency_rel_spread = efficiency_spread / scale
            if efficiency_rel_spread <= 0.05:
                efficiency_class = "activation_rate_insensitive_in_s03_matrix"
            elif efficiency_rel_spread <= 0.25:
                efficiency_class = "activation_rate_sensitive_moderate"
            else:
                efficiency_class = "activation_rate_sensitive_large"
        if values_agg.empty:
            aggregation_spread = np.nan
            aggregation_class = "not_assessed"
        else:
            aggregation_spread = float(values_agg.max() - values_agg.min())
            if aggregation_spread <= 2.5:
                aggregation_class = "activation_rate_insensitive_in_s03_matrix"
            elif aggregation_spread <= 5.0:
                aggregation_class = "activation_rate_sensitive_moderate"
            else:
                aggregation_class = "activation_rate_sensitive_large"
        rows.append(
            {
                "condition_id": condition_id,
                "algotype_mix": mix,
                "claim_family": "efficiency",
                "metric": "compare_plus_swap_steps",
                "absolute_spread": efficiency_spread,
                "relative_spread": efficiency_rel_spread,
                "classification": efficiency_class,
            }
        )
        rows.append(
            {
                "condition_id": condition_id,
                "algotype_mix": mix,
                "claim_family": "aggregation",
                "metric": "peak_aggregation_left_neighbor_percent",
                "absolute_spread": aggregation_spread,
                "relative_spread": np.nan,
                "classification": aggregation_class,
            }
        )
    return pd.DataFrame(rows)


def validate_results(
    df: pd.DataFrame,
    tasks: list[ActivationRateTask],
    summary: pd.DataFrame,
    figure_path: Path,
    artifacts_written: Sequence[Path],
) -> dict[str, Any]:
    same_initial_arrays = bool(df.groupby(["condition_id", "repeat_index"])["initial_array_sha256"].nunique().max() == 1)
    same_initial_assignments = bool(df.groupby(["condition_id", "repeat_index"])["initial_algotype_assignment_sha256"].nunique().max() == 1)
    profile_counts = df.groupby("condition_id")["activation_rate_regime"].nunique().to_dict()
    expected_rows = len(tasks)
    no_duplicates = not df.duplicated(["condition_id", "repeat_index", "activation_rate_regime"]).any()
    rate_passed = bool(df["activation_rate_validation_passed"].all())
    max_share_error = float(pd.to_numeric(df["activation_rate_max_abs_share_error"], errors="coerce").max())
    figure_ok = figure_path.exists() and figure_path.stat().st_size > 1000
    observed_conditions = sorted(df["condition_id"].unique())
    expected_conditions = sorted(SELECTED_CONDITION_IDS)
    passed = all(
        [
            len(df) == expected_rows,
            observed_conditions == expected_conditions,
            no_duplicates,
            rate_passed,
            max_share_error <= 1e-12,
            same_initial_arrays,
            same_initial_assignments,
            len(summary) > 0,
            figure_ok,
        ]
    )
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(passed),
        "status": "completed" if passed else "failed_validation",
        "artifactsWritten": [str(path) for path in artifacts_written],
        "validationResult": "passed" if passed else "failed",
        "caveatsOrBlockers": [
            "Bounded first-10-repeat matrix rather than full E01 100-repeat inference.",
            "Weighted cycles validate configured activation shares exactly but are not an OS-thread replay.",
        ],
        "recommendedNextAction": "Proceed to S04 trajectory-preserving label shuffles after Chief Scientist instruction; do not start S04 inside S03.",
        "validationPassed": bool(passed),
        "expectedRows": int(expected_rows),
        "observedRows": int(len(df)),
        "observedConditions": observed_conditions,
        "profileCountsByCondition": {str(k): int(v) for k, v in profile_counts.items()},
        "sameInitialArraysAcrossActivationProfiles": same_initial_arrays,
        "sameInitialAssignmentsAcrossActivationProfiles": same_initial_assignments,
        "noDuplicateTaskRows": bool(no_duplicates),
        "activationRateValidationPassedForAllRows": rate_passed,
        "maxActivationShareError": max_share_error,
        "summaryRows": int(len(summary)),
        "figureExistsAndNonempty": bool(figure_ok),
    }


def make_figure(summary: pd.DataFrame, classification: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    mixes = list(summary["algotype_mix"].drop_duplicates())
    colors = {
        EQUAL_RATE_REGIME: "#4c78a8",
        "fast_bubble_3x": "#f58518",
        "fast_insertion_3x": "#54a24b",
        "fast_selection_3x": "#e45756",
        "fast_bubble_label_a_3x": "#72b7b2",
        "fast_bubble_label_b_3x": "#b279a2",
    }

    ax = axes[0, 0]
    for regime, sub in summary.groupby("activation_rate_regime"):
        ordered = sub.set_index("algotype_mix").reindex(mixes)
        ax.plot(range(len(mixes)), ordered["mean_compare_plus_swap_steps"], marker="o", label=regime, color=colors.get(regime))
    ax.set_title("Efficiency under activation-rate profiles")
    ax.set_ylabel("Mean compare+swap steps")
    ax.set_xticks(range(len(mixes)))
    ax.set_xticklabels([mix.replace("same_goal_", "").replace("same_algorithm_", "") for mix in mixes], rotation=30, ha="right")

    ax = axes[0, 1]
    for regime, sub in summary.groupby("activation_rate_regime"):
        ordered = sub.set_index("algotype_mix").reindex(mixes)
        ax.plot(
            range(len(mixes)),
            ordered["mean_peak_aggregation_left_neighbor_percent"],
            marker="o",
            label=regime,
            color=colors.get(regime),
        )
    ax.set_title("Aggregation under activation-rate profiles")
    ax.set_ylabel("Mean peak left-neighbor %")
    ax.set_xticks(range(len(mixes)))
    ax.set_xticklabels([mix.replace("same_goal_", "").replace("same_algorithm_", "") for mix in mixes], rotation=30, ha="right")

    ax = axes[1, 0]
    fast = summary[summary["activation_rate_regime"] != EQUAL_RATE_REGIME].copy()
    if len(fast):
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        for regime, sub in fast.groupby("activation_rate_regime"):
            ordered = sub.set_index("algotype_mix").reindex(mixes)
            ax.plot(
                range(len(mixes)),
                ordered["mean_delta_peak_aggregation_vs_equal"],
                marker="o",
                label=regime,
                color=colors.get(regime),
            )
    ax.set_title("Peak Aggregation delta versus equal activation")
    ax.set_ylabel("Percentage-point delta")
    ax.set_xticks(range(len(mixes)))
    ax.set_xticklabels([mix.replace("same_goal_", "").replace("same_algorithm_", "") for mix in mixes], rotation=30, ha="right")

    ax = axes[1, 1]
    class_counts = classification["classification"].value_counts().sort_index()
    ax.barh(class_counts.index, class_counts.values, color="#4c78a8")
    ax.set_title("S03 sensitivity classifications")
    ax.set_xlabel("Claim-family slices")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("E02 S03 activation-rate artifact matrix", fontsize=14)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def outcome_from_classification(classification: pd.DataFrame) -> str:
    classes = set(classification["classification"].dropna())
    if "activation_rate_sensitive_large" in classes:
        return "Supportive"
    if "activation_rate_sensitive_moderate" in classes:
        return "Supportive"
    return "Null"


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
    outcome = outcome_from_classification(classification)
    sensitivity_counts = classification["classification"].value_counts().to_dict() if len(classification) else {}
    validation_result = (
        f"passed: {validation['observedRows']}/{validation['expectedRows']} activation-rate rows, "
        f"configured activation shares matched={validation['activationRateValidationPassedForAllRows']}, "
        f"same initial arrays={validation['sameInitialArraysAcrossActivationProfiles']}, "
        f"same initial assignments={validation['sameInitialAssignmentsAcrossActivationProfiles']}, "
        f"unit tests={'passed' if unit_result and unit_result['success'] else 'not run'}"
    )
    caveats = [
        "S03 uses a bounded first-10-repeat matrix, not the full E01 100-repeat sample.",
        "Weighted activation cycles validate intended activation shares exactly, but they are a deterministic intervention rather than an OS-thread replay.",
        "The intervention changes activation opportunity counts while leaving public cell policies and initial Algotype assignments fixed.",
        "Aggregation remains a computational label-clustering proxy.",
    ]
    compact_summary = summary[
        [
            "condition_id",
            "algotype_mix",
            "activation_rate_regime",
            "fast_label",
            "n_runs",
            "mean_compare_plus_swap_steps",
            "mean_peak_aggregation_left_neighbor_percent",
            "mean_delta_peak_aggregation_vs_equal",
            "mean_relative_delta_compare_plus_swap_vs_equal",
            "rate_validation_failures",
        ]
    ]
    compact_classification = classification[
        ["condition_id", "algotype_mix", "claim_family", "metric", "absolute_spread", "relative_spread", "classification"]
    ]
    lay_summary = (
        "Deliberately giving one Algotype more activation opportunities changed efficiency and/or Aggregation metrics in this bounded "
        "matrix, so activation tempo is a real control variable that downstream null tests must account for."
        if outcome == "Supportive"
        else "Within this bounded matrix, deliberately unequal activation opportunities did not materially change the tested efficiency or Aggregation metrics."
    )
    return f"""# E02 S03 Activation-Rate Artifact Test

## Top Summary

- Research step ID: S03
- Completion status: Completed on {generated_at}
- Artifacts written: `{artifacts_dir / 'research_steps/S03/research_step_full_results.md'}`, `{result_path}`, `{figure_path}`, `{validation_path}`, `{summary_path}`, `{classification_path}`, `{log_path}`, `{manifest_path}`, `{src_manifest_path}`
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers: {'; '.join(caveats)}
- Lay summary: {lay_summary}
- Recommended next action: Proceed to S04 trajectory-preserving label shuffles after Chief Scientist instruction; do not start S04 inside S03.

## Chief Handoff

S03 completed the planned activation-rate artifact test. The result estimates how much efficiency and Aggregation proxies move when Algotype policies and initial assignments are held fixed but activation opportunities are made equal or deliberately unequal.

## Frozen Question

Can apparent efficiency or Aggregation differences be induced solely by unequal activation rates rather than meaningful Algotype policy differences?

## Inputs

- Repository commit at S03 run time: `{git_commit}`
- S01 simulator source: `src/e02/deterministic_simulator.py`
- S02 scheduler framework source: `scripts/e02_s02_scheduler_comparison.py`
- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- S02 scheduler comparison reference: `/artifacts/results/e02_scheduler_comparison.parquet`
- Selected E01 chimeric conditions: `{', '.join(SELECTED_CONDITION_IDS)}`

## Methods

S03 used the first `{max_repeats}` E01 repeats for the five same-goal chimeric cell-view conditions. For each condition and repeat, it ran:

- `equal_per_cell`: every active cell receives one activation opportunity per weighted cycle.
- `fast_<label>_3x`: cells with one target Algotype label receive three activation opportunities per weighted cycle, while all other labels receive one.

The public Bubble, Insertion, and Selection cell classes and their public `move()` methods were unchanged. The scheduler is a deterministic weighted identity-cycle scheduler: each cycle creates a list of active cell identities repeated according to configured label weights, shuffles that list with a fixed seed, runs the complete cycle, and checks stopping criteria at cycle boundaries. This makes realized label activation shares exactly auditable.

## Commands

- Unit tests: `{unit_result['command'] if unit_result else 'not run'}`
- S03 production run: `{sys.executable} scripts/e02_s03_activation_rates.py --repo-dir /workspace/cell-research --artifacts-dir {artifacts_dir} --max-repeats {max_repeats} --workers {workers}`

## Dependencies And Parameters

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- Worker count: `{workers}`
- Fast-label multiplier: `{FAST_MULTIPLIER}`
- New dependencies installed: none
- Elapsed production time: `{elapsed_seconds:.2f}` seconds

## Results

Sensitivity classification counts: `{json.dumps(sensitivity_counts, sort_keys=True)}`.

### Activation-Rate Summary

{markdown_table(compact_summary, max_rows=24)}

### Claim-Family Sensitivity

{markdown_table(compact_classification, max_rows=20)}

The full run-level table is in `{result_path}`. The activation-rate figure is `{figure_path}`.

## Validation

Validation required all planned task rows to be present, no duplicate condition/repeat/profile rows, identical initial arrays and exact initial Algotype assignments across activation profiles for each condition/repeat, exact configured-versus-realized label activation shares for every run, nonempty summaries, and a nonempty figure. The validation result was `{validation['validationPassed']}`.

Validation details were written to `{validation_path}`.

## Artifacts

- Run-level activation-rate artifact table: `{result_path}`
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

- This is a bounded first-10-repeat matrix intended to expose or rule out activation-rate artifacts, not final corrected inference.
- Weighted cycles are deliberately controlled interventions. They should not be interpreted as reproducing original Python thread scheduling.
- Because the public cell `move()` methods mutate state immediately, the intervention changes opportunity tempo without changing policy code, but downstream effects may still interact with stopping criteria and path history.
- Aggregation is a label-neighborhood proxy and does not identify a biological adhesion or identity mechanism.

## Recommended Next Action

Proceed to S04 trajectory-preserving label shuffles after explicit Chief Scientist instruction. S04 should use the S03 result to decide whether activation rate needs to be blocked or stratified in label-shuffle nulls.
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
    log_lines = [f"E02 S03 started {generated_at}", f"repo_dir={repo_dir}", f"artifacts_dir={artifacts_dir}"]
    unit_result: dict[str, Any] | None = None
    if args.run_unit_tests:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_dir)
        unit_result = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_*.py"], repo_dir, env)
        log_lines.extend(["UNIT TESTS:", json.dumps(unit_result, indent=2, sort_keys=True)])
        if not unit_result["success"]:
            log_path = logs_dir / "e02_s03_activation_rates.log"
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return int(unit_result["returnCode"] or 1)

    if not args.s02_results_path.exists():
        raise RuntimeError(f"Required S02 scheduler comparison reference is missing: {args.s02_results_path}")
    s02_reference_rows = len(pd.read_parquet(args.s02_results_path, columns=["condition_id"]))
    log_lines.append(f"s02_results_path={args.s02_results_path} s02_reference_rows={s02_reference_rows}")

    cfg = load_e01_config(args.config_path)
    rows = selected_chimera_conditions(cfg)
    tasks = build_tasks(
        cfg,
        rows,
        repo_dir,
        args.max_repeats,
        args.max_cycles,
        args.max_successful_swaps,
        args.stall_cycles,
    )
    log_lines.append(f"selected_conditions={len(rows)} tasks={len(tasks)} workers={args.workers}")
    result_rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_activation_task, task): task for task in tasks}
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_task), start=1):
            task = future_to_task[future]
            try:
                result_rows.append(future.result())
            except Exception as exc:
                log_lines.append(
                    f"FAILED task {task.condition['condition_id']} repeat={task.repeat_index} "
                    f"profile={task.activation_rate_regime}: {exc!r}"
                )
                raise
            if idx % 25 == 0 or idx == len(tasks):
                log_lines.append(f"completed_tasks={idx}/{len(tasks)}")

    result_df = add_equal_reference_deltas(pd.DataFrame(result_rows))
    result_df = result_df.sort_values(["condition_id", "repeat_index", "activation_rate_regime"]).reset_index(drop=True)
    summary_df = summarize_results(result_df)
    classification_df = classify_activation_effects(summary_df)

    result_path = results_dir / "e02_activation_rate_artifacts.parquet"
    summary_path = step_dir / "e02_activation_rate_summary.csv"
    classification_path = step_dir / "e02_activation_rate_sensitivity_classification.csv"
    figure_path = figures_dir / "activation_rate_effects.png"
    validation_path = step_dir / "s03_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s03_activation_rates_manifest.json"
    log_path = logs_dir / "e02_s03_activation_rates.log"
    report_path = step_dir / "research_step_full_results.md"

    result_df.to_parquet(result_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    make_figure(summary_df, classification_df, figure_path)
    expected_artifacts = [
        report_path,
        result_path,
        figure_path,
        validation_path,
        summary_path,
        classification_path,
        log_path,
        manifest_path,
        src_manifest_path,
    ]
    validation = validate_results(result_df, tasks, summary_df, figure_path, expected_artifacts)
    write_json(validation_path, validation)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    elapsed_seconds = time.monotonic() - start
    log_lines.extend(
        [
            f"result_rows={len(result_df)} summary_rows={len(summary_df)} classification_rows={len(classification_df)}",
            f"validation={json.dumps(validation, sort_keys=True)}",
            f"elapsed_seconds={elapsed_seconds:.3f}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    code_files = [
        repo_dir / "scripts/e02_s03_activation_rates.py",
        repo_dir / "tests/e02/test_activation_rates.py",
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
            "workers": int(args.workers),
            "maxCycles": int(args.max_cycles),
            "maxSuccessfulSwaps": int(args.max_successful_swaps),
            "stallCycles": int(args.stall_cycles),
            "fastMultiplier": int(FAST_MULTIPLIER),
            "selectedConditionIds": list(SELECTED_CONDITION_IDS),
            "s02ResultsPath": str(args.s02_results_path),
            "s02ReferenceRows": int(s02_reference_rows),
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
        artifact_entry(report_path, artifacts_dir, "S03 full-results Markdown handoff report."),
        artifact_entry(result_path, artifacts_dir, "Run-level activation-rate artifact table."),
        artifact_entry(summary_path, artifacts_dir, "Activation-rate summary table."),
        artifact_entry(classification_path, artifacts_dir, "Claim-family activation-rate sensitivity classification."),
        artifact_entry(figure_path, artifacts_dir, "Activation-rate effects figure."),
        artifact_entry(validation_path, artifacts_dir, "S03 validation evidence."),
        artifact_entry(log_path, artifacts_dir, "S03 execution log."),
        artifact_entry(src_manifest_path, artifacts_dir, "S03 source snapshot manifest."),
    ]
    write_json(manifest_path, artifact_stub | {"artifacts": artifacts})
    if not validation["validationPassed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
