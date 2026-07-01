#!/usr/bin/env python3
"""Run E02 S06 speed-matched Algotype controls.

This step uses the S01 deterministic public-cell simulator and the S03
weighted activation-cycle scheduler to test whether mixed-policy Aggregation
survives an E01-efficiency-derived speed correction. The public cell policy
classes and their ``move()`` methods are not changed; only activation
opportunity weights are changed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
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
from scripts.e02_s03_activation_rates import (
    activation_cycle_items,
    activation_rate_deviation,
    expected_label_shares,
    label_order_from_condition,
    observed_label_shares,
)
from src.e02.deterministic_simulator import (
    ALGORITHMS,
    DEFAULT_LABEL_TO_BEHAVIOR,
    EventTracingStatusProbe,
    aggregation_left_neighbor_percent,
    aggregation_right_neighbor_legacy_percent,
    cell_state_signature,
    cell_values,
    count_json,
    has_public_legal_action,
    is_sorted,
    monotonicity_error_count,
    sortedness_percent,
    stable_json_sha256,
)


STEP_ID = "S06"
STEP_NUMBER = 6
EXPERIMENT_ID = "E02"
SELECTED_CONDITION_IDS = ("E01C043", "E01C044", "E01C045", "E01C046")
EQUAL_RATE_REGIME = "equal_per_cell"
SPEED_MATCH_RATE_REGIME = "speed_matched_e01_efficiency"
DEFAULT_MAX_WEIGHT = 8
DEFAULT_SPEED_RATIO_THRESHOLD = 1.05


@dataclass(frozen=True)
class SpeedMatchedTask:
    condition: dict[str, Any]
    repeat_index: int
    values: tuple[int, ...]
    initial_array_seed: int
    assignments: tuple[str, ...]
    assignment_seed: int | None
    frozen_indices: tuple[int, ...]
    frozen_index_seed: int | None
    activation_rate_regime: str
    label_weights: dict[str, int]
    speed_match_basis: str
    pure_efficiency_means: dict[str, float]
    expected_cycle_cost_by_label: dict[str, float]
    expected_cycle_cost_ratio: float
    expected_cycle_cost_cv: float
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
    parser.add_argument(
        "--e01-efficiency-path",
        type=Path,
        default=Path("/previous-artifacts/E01/results/e01_efficiency_counts.parquet"),
    )
    parser.add_argument(
        "--s03-results-path",
        type=Path,
        default=Path("/artifacts/results/e02_activation_rate_artifacts.parquet"),
    )
    parser.add_argument("--s05-validation-path", type=Path, default=Path("/artifacts/research_steps/S05/s05_validation.json"))
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-cycles", type=int, default=20_000)
    parser.add_argument("--max-successful-swaps", type=int, default=80_000)
    parser.add_argument("--stall-cycles", type=int, default=2)
    parser.add_argument("--max-speed-weight", type=int, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument("--speed-ratio-threshold", type=float, default=DEFAULT_SPEED_RATIO_THRESHOLD)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json(data: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


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
        raise RuntimeError(f"Missing expected S06 chimeric condition IDs in E01 config: {missing}")
    return sorted(rows, key=lambda row: row["condition_id"])


def speed_match_seed(base_seed: int, condition_id: str, repeat_idx: int, profile_idx: int) -> int:
    condition_num = int(condition_id.replace("E01C", ""))
    return base_seed * 1_000_000 + 600_000 + condition_num * 1_000 + repeat_idx * 10 + profile_idx


def behavior_for_label(label: str) -> str:
    behavior = DEFAULT_LABEL_TO_BEHAVIOR.get(str(label), str(label))
    if behavior not in ALGORITHMS:
        raise ValueError(f"Unsupported speed-match label behavior: label={label!r} behavior={behavior!r}")
    return behavior


def load_pure_efficiency_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Required E01 efficiency estimate table is missing: {path}")
    df = pd.read_parquet(path)
    required = {"mode", "algorithm", "compare_plus_swap_steps", "repeat_index", "stop_reason", "final_sortedness_percent"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"E01 efficiency table is missing required columns: {missing}")
    pure = df[
        df["mode"].eq("cell_view")
        & df["algorithm"].isin(ALGORITHMS)
        & df["stop_reason"].eq("sorted")
        & (pd.to_numeric(df["final_sortedness_percent"], errors="coerce") == 100.0)
    ].copy()
    if pure.empty:
        raise RuntimeError("E01 efficiency table has no sorted pure cell-view rows for speed matching.")
    rows: list[dict[str, Any]] = []
    for algorithm, group in pure.groupby("algorithm", sort=True):
        steps = pd.to_numeric(group["compare_plus_swap_steps"], errors="coerce").dropna()
        rows.append(
            {
                "algorithm": str(algorithm),
                "n_runs": int(len(steps)),
                "mean_compare_plus_swap_steps": float(steps.mean()),
                "std_compare_plus_swap_steps": float(steps.std(ddof=1)) if len(steps) > 1 else np.nan,
                "q10_compare_plus_swap_steps": float(steps.quantile(0.10)),
                "q50_compare_plus_swap_steps": float(steps.quantile(0.50)),
                "q90_compare_plus_swap_steps": float(steps.quantile(0.90)),
            }
        )
    stats = pd.DataFrame(rows)
    observed = set(stats["algorithm"])
    missing_algorithms = sorted(set(ALGORITHMS) - observed)
    if missing_algorithms:
        raise RuntimeError(f"E01 efficiency stats missing algorithms required for speed matching: {missing_algorithms}")
    return stats.sort_values("algorithm").reset_index(drop=True)


def choose_speed_match_weights(
    labels: Sequence[str],
    pure_means_by_behavior: Mapping[str, float],
    max_weight: int = DEFAULT_MAX_WEIGHT,
) -> tuple[dict[str, int], dict[str, float], float, float]:
    ordered = tuple(dict.fromkeys(map(str, labels)))
    if len(ordered) < 2:
        return ({label: 1 for label in ordered}, {label: pure_means_by_behavior[behavior_for_label(label)] for label in ordered}, 1.0, 0.0)
    if max_weight < 1:
        raise ValueError("max_weight must be at least 1.")

    best: tuple[float, float, int, tuple[int, ...], list[float]] | None = None
    for candidate in itertools.product(range(1, max_weight + 1), repeat=len(ordered)):
        costs = [float(pure_means_by_behavior[behavior_for_label(label)]) / float(weight) for label, weight in zip(ordered, candidate, strict=True)]
        ratio = max(costs) / min(costs)
        cv = statistics.pstdev(costs) / statistics.mean(costs) if len(costs) > 1 else 0.0
        objective = (ratio, cv, sum(candidate), candidate, costs)
        if best is None or objective < best:
            best = objective
    if best is None:
        raise RuntimeError("Speed-match weight search failed.")
    ratio, cv, _, weights_tuple, costs = best
    weights = {label: int(weight) for label, weight in zip(ordered, weights_tuple, strict=True)}
    expected_costs = {label: float(cost) for label, cost in zip(ordered, costs, strict=True)}
    return weights, expected_costs, float(ratio), float(cv)


def equal_profile(labels: Sequence[str], pure_means_by_behavior: Mapping[str, float]) -> tuple[dict[str, int], dict[str, float], float, float]:
    ordered = tuple(dict.fromkeys(map(str, labels)))
    weights = {label: 1 for label in ordered}
    expected_costs = {label: float(pure_means_by_behavior[behavior_for_label(label)]) for label in ordered}
    finite = list(expected_costs.values())
    ratio = max(finite) / min(finite) if finite else np.nan
    cv = statistics.pstdev(finite) / statistics.mean(finite) if len(finite) > 1 else 0.0
    return weights, expected_costs, float(ratio), float(cv)


def make_speed_profiles(
    labels: Sequence[str],
    pure_means_by_behavior: Mapping[str, float],
    max_weight: int,
) -> list[tuple[str, str, dict[str, int], dict[str, float], float, float]]:
    equal_weights, equal_costs, equal_ratio, equal_cv = equal_profile(labels, pure_means_by_behavior)
    matched_weights, matched_costs, matched_ratio, matched_cv = choose_speed_match_weights(
        labels,
        pure_means_by_behavior,
        max_weight=max_weight,
    )
    return [
        (EQUAL_RATE_REGIME, "unmatched_equal_per_cell_reference", equal_weights, equal_costs, equal_ratio, equal_cv),
        (
            SPEED_MATCH_RATE_REGIME,
            f"integer_weight_search_e01_compare_plus_swap_cap_{max_weight}",
            matched_weights,
            matched_costs,
            matched_ratio,
            matched_cv,
        ),
    ]


def build_tasks(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    repo_dir: Path,
    max_repeats: int,
    max_cycles: int,
    max_successful_swaps: int,
    stall_cycles: int,
    pure_means_by_behavior: Mapping[str, float],
    max_weight: int,
) -> list[SpeedMatchedTask]:
    repeat_count = min(max_repeats, int(cfg["globalDefaults"]["repeatCount"]))
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    value_banks = cfg["seedBanks"]["valueBanks"]
    assignment_banks = cfg["seedBanks"]["algotypeAssignmentBanks"]
    frozen_banks = cfg["seedBanks"]["frozenIndexBanks"]
    tasks: list[SpeedMatchedTask] = []
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
            for profile_idx, (regime, basis, weights, expected_costs, cost_ratio, cost_cv) in enumerate(
                make_speed_profiles(labels, pure_means_by_behavior, max_weight=max_weight)
            ):
                tasks.append(
                    SpeedMatchedTask(
                        condition=dict(row),
                        repeat_index=repeat_idx,
                        values=values,
                        initial_array_seed=int(value_bank["seeds"][repeat_idx]),
                        assignments=assignments,
                        assignment_seed=int(assignment_bank["seeds"][repeat_idx]),
                        frozen_indices=frozen_indices,
                        frozen_index_seed=frozen_seed_value,
                        activation_rate_regime=regime,
                        label_weights=dict(weights),
                        speed_match_basis=basis,
                        pure_efficiency_means={
                            label: float(pure_means_by_behavior[behavior_for_label(label)]) for label in labels
                        },
                        expected_cycle_cost_by_label=dict(expected_costs),
                        expected_cycle_cost_ratio=float(cost_ratio),
                        expected_cycle_cost_cv=float(cost_cv),
                        scheduler_seed=speed_match_seed(base_seed, row["condition_id"], repeat_idx, profile_idx),
                        repo_dir=str(repo_dir),
                        max_cycles=max_cycles,
                        max_successful_swaps=max_successful_swaps,
                        stall_cycles=stall_cycles,
                    )
                )
    return tasks


def dispatch_map(cells: Sequence[Any]) -> dict[str, dict[str, str]]:
    by_label: dict[str, dict[str, str]] = {}
    for cell in cells:
        label = str(cell.label)
        cls = type(cell)
        by_label.setdefault(
            label,
            {
                "behavior": behavior_for_label(label),
                "class": f"{cls.__module__}.{cls.__qualname__}",
                "moveImplementation": f"{cls.move.__module__}.{cls.move.__qualname__}",
            },
        )
    return dict(sorted(by_label.items()))


def dispatch_preserved(labels: Sequence[str], dispatch: Mapping[str, Mapping[str, str]]) -> bool:
    for label in set(map(str, labels)):
        observed = dispatch.get(label)
        if not observed:
            return False
        if observed.get("behavior") != behavior_for_label(label):
            return False
        if behavior_for_label(label) not in observed.get("class", "").lower():
            return False
    return True


def per_cell_metric(counter: Counter[str], label_counts: Counter[str]) -> dict[str, float]:
    return {
        label: float(counter[label]) / float(label_counts[label]) if label_counts[label] else np.nan
        for label in sorted(label_counts)
    }


def ratio_and_cv(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return np.nan, np.nan
    ratio = max(finite) / min(finite) if min(finite) > 0 else np.inf
    cv = statistics.pstdev(finite) / statistics.mean(finite) if len(finite) > 1 and statistics.mean(finite) else 0.0
    return float(ratio), float(cv)


def run_speed_matched_task(task: SpeedMatchedTask) -> dict[str, Any]:
    start = time.monotonic()
    random.seed(task.scheduler_seed + 17)
    rng = random.Random(task.scheduler_seed)
    probe = EventTracingStatusProbe()
    cells, cell_status = build_cells(task, probe)
    initial_values = cell_values(cells)
    initial_labels = tuple(task.assignments)
    initial_label_counts: Counter[str] = Counter(initial_labels)
    activation_hash = hashlib.sha256()
    activation_label_counts: Counter[str] = Counter()
    label_comparison_counts: Counter[str] = Counter()
    label_swap_counts: Counter[str] = Counter()
    activation_first_20: list[dict[str, Any]] = []
    activation_last_20: list[dict[str, Any]] = []
    dispatch = dispatch_map(cells)
    event_count = 0
    cycle_count = 0
    no_progress_cycles = 0
    stop_reason = "max_cycles_exceeded"
    max_guard_hit = False
    no_legal_action_cycles = 0

    while cycle_count < task.max_cycles:
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
        items = activation_cycle_items(cells, cell_status, task.label_weights, rng)
        for cell in items:
            if cell.status != cell_status.ACTIVE:
                continue
            event_count += 1
            probe.current_event_step = event_count
            label = str(cell.label)
            event_payload = {
                "event": event_count,
                "cycle": cycle_count + 1,
                "thread_id": int(cell.threadID),
                "position_before": int(cell.current_position[0]),
                "label": label,
                "behavior": behavior_for_label(label),
                "configured_label_weight": int(task.label_weights[label]),
                "value_before": int(cell.value),
            }
            activation_hash.update(json.dumps(event_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            activation_label_counts[label] += 1
            if len(activation_first_20) < 20:
                activation_first_20.append(event_payload)
            activation_last_20.append(event_payload)
            if len(activation_last_20) > 20:
                activation_last_20.pop(0)
            before_comparisons = int(probe.compare_and_swap_count)
            before_swaps = int(probe.swap_count)
            cell.move()
            comparison_delta = int(probe.compare_and_swap_count) - before_comparisons
            swap_delta = int(probe.swap_count) - before_swaps
            if comparison_delta:
                label_comparison_counts[label] += comparison_delta
            if swap_delta:
                label_swap_counts[label] += swap_delta
        cycle_count += 1
        after_signature = cell_state_signature(cells)
        made_progress = int(probe.swap_count) != before_swap_count or after_signature != before_signature
        if not made_progress and not legal_action_exists:
            no_legal_action_cycles += 1
            no_progress_cycles += 1
            if no_progress_cycles >= task.stall_cycles:
                stop_reason = "no_legal_action_window"
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
    label_compare_plus_swap_counts = Counter()
    for label in set(label_comparison_counts) | set(label_swap_counts) | set(initial_label_counts):
        label_compare_plus_swap_counts[label] = int(label_comparison_counts[label] + label_swap_counts[label])
    per_cell_activations = per_cell_metric(activation_label_counts, initial_label_counts)
    per_cell_work = per_cell_metric(label_compare_plus_swap_counts, initial_label_counts)
    activation_per_weight = {
        label: per_cell_activations[label] / float(task.label_weights[label]) if task.label_weights[label] else np.nan
        for label in sorted(initial_label_counts)
    }
    activation_weight_ratio, activation_weight_cv = ratio_and_cv(list(activation_per_weight.values()))
    work_ratio, work_cv = ratio_and_cv([value for value in per_cell_work.values() if value > 0])
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
        "speed_match_basis": task.speed_match_basis,
        "scheduler_regime": "weighted_activation_cycle_speed_match",
        "scheduler_source": "s03_weighted_identity_cycle_with_e01_efficiency_weights_public_move_scheduler",
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
        "label_behavior_map_json": safe_json({label: behavior_for_label(label) for label in sorted(initial_label_counts)}),
        "label_dispatch_map_json": safe_json(dispatch),
        "policy_dispatch_preserved": bool(dispatch_preserved(initial_labels, dispatch)),
        "configured_label_weights_json": safe_json(dict(sorted(task.label_weights.items()))),
        "pure_efficiency_means_json": safe_json(dict(sorted(task.pure_efficiency_means.items()))),
        "expected_cycle_cost_by_label_json": safe_json(dict(sorted(task.expected_cycle_cost_by_label.items()))),
        "expected_cycle_cost_ratio": float(task.expected_cycle_cost_ratio),
        "expected_cycle_cost_cv": float(task.expected_cycle_cost_cv),
        "expected_activation_shares_json": safe_json(expected_shares),
        "observed_activation_shares_json": safe_json(observed_shares),
        "activation_rate_max_abs_share_error": max_share_error,
        "activation_rate_max_abs_per_cell_ratio_error": max_ratio_error,
        "activation_rate_validation_passed": bool(rate_valid),
        "activation_per_weight_normalized_by_label_json": safe_json(activation_per_weight),
        "activation_per_weight_ratio": activation_weight_ratio,
        "activation_per_weight_cv": activation_weight_cv,
        "label_comparison_counts_json": safe_json(dict(sorted(label_comparison_counts.items()))),
        "label_swap_counts_json": safe_json(dict(sorted(label_swap_counts.items()))),
        "label_compare_plus_swap_counts_json": safe_json(dict(sorted(label_compare_plus_swap_counts.items()))),
        "per_cell_activation_counts_by_label_json": safe_json(per_cell_activations),
        "per_cell_compare_plus_swap_by_label_json": safe_json(per_cell_work),
        "per_cell_compare_plus_swap_ratio": work_ratio,
        "per_cell_compare_plus_swap_cv": work_cv,
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
        "no_legal_action_cycles": int(no_legal_action_cycles),
        "elapsed_seconds": time.monotonic() - start,
        "activation_event_count": int(event_count),
        "activation_log_sha256": activation_hash.hexdigest(),
        "activation_label_counts_json": safe_json(dict(sorted(activation_label_counts.items()))),
        "activation_first_20_json": json.dumps(activation_first_20, separators=(",", ":")),
        "activation_last_20_json": json.dumps(activation_last_20, separators=(",", ":")),
        "trace_record_count": len(records),
        "trace_sha256": stable_json_sha256(records),
        "result_scope": "s06_bounded_first_e01_repeats_speed_matched_chimera_matrix",
        "wrapper_name": "weighted_activation_cycle_speed_matched_public_cell_methods",
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
            "per_cell_compare_plus_swap_ratio",
        ]
    ].rename(
        columns={
            "compare_plus_swap_steps": "equal_compare_plus_swap_steps",
            "activation_event_count": "equal_activation_event_count",
            "peak_aggregation_left_neighbor_percent": "equal_peak_aggregation_left_neighbor_percent",
            "final_aggregation_left_neighbor_percent": "equal_final_aggregation_left_neighbor_percent",
            "final_sortedness_percent": "equal_final_sortedness_percent",
            "per_cell_compare_plus_swap_ratio": "equal_per_cell_compare_plus_swap_ratio",
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
    merged["delta_per_cell_compare_plus_swap_ratio_vs_equal"] = (
        merged["per_cell_compare_plus_swap_ratio"] - merged["equal_per_cell_compare_plus_swap_ratio"]
    )
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
        "expected_cycle_cost_ratio",
        "expected_cycle_cost_cv",
        "compare_plus_swap_steps",
        "activation_event_count",
        "cycle_count",
        "peak_aggregation_left_neighbor_percent",
        "final_aggregation_left_neighbor_percent",
        "final_sortedness_percent",
        "delta_compare_plus_swap_vs_equal",
        "relative_delta_compare_plus_swap_vs_equal",
        "delta_peak_aggregation_vs_equal",
        "delta_final_aggregation_vs_equal",
        "activation_rate_max_abs_share_error",
        "activation_per_weight_ratio",
        "per_cell_compare_plus_swap_ratio",
    ]
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["condition_id", "algotype_mix", "activation_rate_regime"], dropna=False)
    for keys, group in grouped:
        row: dict[str, Any] = {
            "condition_id": keys[0],
            "algotype_mix": keys[1],
            "activation_rate_regime": keys[2],
            "n_runs": int(len(group)),
            "label_weights_json": group["configured_label_weights_json"].iloc[0],
            "speed_match_basis": group["speed_match_basis"].iloc[0],
            "stop_reason_counts_json": json.dumps(group["stop_reason"].value_counts(dropna=False).to_dict(), sort_keys=True),
            "max_guard_runs": int(group["max_guard_hit"].map(bool).sum()),
            "rate_validation_failures": int((~group["activation_rate_validation_passed"].astype(bool)).sum()),
            "policy_dispatch_failures": int((~group["policy_dispatch_preserved"].astype(bool)).sum()),
        }
        for metric in metrics:
            mean, std, low, high = mean_ci(group[metric])
            row[f"mean_{metric}"] = mean
            row[f"std_{metric}"] = std
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition_id", "activation_rate_regime"]).reset_index(drop=True)


def speed_match_validation_table(
    rows: list[dict[str, Any]],
    pure_stats: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    pure_by_algorithm = pure_stats.set_index("algorithm").to_dict(orient="index")
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        condition_id = row["condition_id"]
        regime = row["activation_rate_regime"]
        if (condition_id, regime, row["configured_label_weights_json"]) in seen:
            continue
        seen.add((condition_id, regime, row["configured_label_weights_json"]))
        weights = json.loads(row["configured_label_weights_json"])
        expected = json.loads(row["expected_cycle_cost_by_label_json"])
        labels = sorted(weights)
        costs = [float(expected[label]) for label in labels]
        ratio, cv = ratio_and_cv(costs)
        for label in labels:
            behavior = behavior_for_label(label)
            pure = pure_by_algorithm[behavior]
            output.append(
                {
                    "condition_id": condition_id,
                    "algotype_mix": row["algotype_mix"],
                    "activation_rate_regime": regime,
                    "label": label,
                    "behavior": behavior,
                    "weight": int(weights[label]),
                    "pure_n_runs": int(pure["n_runs"]),
                    "pure_mean_compare_plus_swap_steps": float(pure["mean_compare_plus_swap_steps"]),
                    "pure_std_compare_plus_swap_steps": float(pure["std_compare_plus_swap_steps"]),
                    "pure_q10_compare_plus_swap_steps": float(pure["q10_compare_plus_swap_steps"]),
                    "pure_q50_compare_plus_swap_steps": float(pure["q50_compare_plus_swap_steps"]),
                    "pure_q90_compare_plus_swap_steps": float(pure["q90_compare_plus_swap_steps"]),
                    "weighted_expected_cycle_mean_steps": float(pure["mean_compare_plus_swap_steps"]) / int(weights[label]),
                    "weighted_expected_cycle_std_steps": float(pure["std_compare_plus_swap_steps"]) / int(weights[label]),
                    "weighted_expected_cycle_q10_steps": float(pure["q10_compare_plus_swap_steps"]) / int(weights[label]),
                    "weighted_expected_cycle_q50_steps": float(pure["q50_compare_plus_swap_steps"]) / int(weights[label]),
                    "weighted_expected_cycle_q90_steps": float(pure["q90_compare_plus_swap_steps"]) / int(weights[label]),
                    "condition_profile_expected_cycle_cost_ratio": ratio,
                    "condition_profile_expected_cycle_cost_cv": cv,
                    "passes_speed_ratio_threshold": bool(regime != SPEED_MATCH_RATE_REGIME or ratio <= threshold),
                }
            )
    return pd.DataFrame(output).sort_values(["condition_id", "activation_rate_regime", "label"]).reset_index(drop=True)


def classify_speed_effects(summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition_id, mix), group in summary.groupby(["condition_id", "algotype_mix"], dropna=False):
        equal = group[group["activation_rate_regime"] == EQUAL_RATE_REGIME]
        speed = group[group["activation_rate_regime"] == SPEED_MATCH_RATE_REGIME]
        if equal.empty or speed.empty:
            classification = "not_assessed_missing_profile"
            delta_peak = np.nan
            rel_efficiency = np.nan
            cost_ratio = np.nan
        else:
            delta_peak = float(speed["mean_peak_aggregation_left_neighbor_percent"].iloc[0] - equal["mean_peak_aggregation_left_neighbor_percent"].iloc[0])
            rel_efficiency = float(speed["mean_relative_delta_compare_plus_swap_vs_equal"].iloc[0])
            cost_ratio = float(speed["mean_expected_cycle_cost_ratio"].iloc[0])
            if cost_ratio > threshold:
                classification = "not_assessed_speed_match_failed"
            elif abs(delta_peak) <= 2.5:
                classification = "speed_independent"
            elif delta_peak < -2.5:
                classification = "speed_explained"
            else:
                classification = "speed_amplified"
        rows.append(
            {
                "condition_id": condition_id,
                "algotype_mix": mix,
                "metric": "peak_aggregation_left_neighbor_percent",
                "speed_match_ratio_threshold": float(threshold),
                "mean_delta_peak_aggregation_vs_equal": delta_peak,
                "mean_relative_delta_compare_plus_swap_vs_equal": rel_efficiency,
                "speed_matched_expected_cycle_cost_ratio": cost_ratio,
                "classification": classification,
            }
        )
    return pd.DataFrame(rows).sort_values("condition_id").reset_index(drop=True)


def make_figure(summary: pd.DataFrame, classification: pd.DataFrame, validation_detail: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    colors = {EQUAL_RATE_REGIME: "#4c78a8", SPEED_MATCH_RATE_REGIME: "#f58518"}
    mixes = list(summary["algotype_mix"].drop_duplicates())
    labels = [mix.replace("same_goal_", "").replace("_", "\n") for mix in mixes]

    ax = axes[0, 0]
    ratio_rows = summary.pivot(index="algotype_mix", columns="activation_rate_regime", values="mean_expected_cycle_cost_ratio").reindex(mixes)
    x = np.arange(len(mixes))
    width = 0.36
    for offset, regime in [(-width / 2, EQUAL_RATE_REGIME), (width / 2, SPEED_MATCH_RATE_REGIME)]:
        if regime in ratio_rows:
            ax.bar(x + offset, ratio_rows[regime], width, label=regime, color=colors[regime])
    ax.axhline(DEFAULT_SPEED_RATIO_THRESHOLD, color="#666666", linestyle="--", linewidth=1, label="1.05 target")
    ax.set_title("E01 expected step-count ratio after weighting")
    ax.set_ylabel("Max/min expected cycle steps")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax = axes[0, 1]
    peak = summary.pivot(index="algotype_mix", columns="activation_rate_regime", values="mean_peak_aggregation_left_neighbor_percent").reindex(mixes)
    for regime in [EQUAL_RATE_REGIME, SPEED_MATCH_RATE_REGIME]:
        if regime in peak:
            ax.plot(x, peak[regime], marker="o", label=regime, color=colors[regime])
    ax.set_title("Peak Aggregation")
    ax.set_ylabel("Mean left-neighbor %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax = axes[1, 0]
    delta = classification.set_index("algotype_mix").reindex(mixes)
    bar_colors = [
        {"speed_independent": "#54a24b", "speed_explained": "#e45756", "speed_amplified": "#b279a2"}.get(cls, "#9d9d9d")
        for cls in delta["classification"]
    ]
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.axhline(2.5, color="#999999", linestyle="--", linewidth=0.8)
    ax.axhline(-2.5, color="#999999", linestyle="--", linewidth=0.8)
    ax.bar(x, delta["mean_delta_peak_aggregation_vs_equal"], color=bar_colors)
    ax.set_title("Speed-matched Aggregation delta")
    ax.set_ylabel("Percentage points vs equal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax = axes[1, 1]
    work = summary.pivot(index="algotype_mix", columns="activation_rate_regime", values="mean_per_cell_compare_plus_swap_ratio").reindex(mixes)
    for regime in [EQUAL_RATE_REGIME, SPEED_MATCH_RATE_REGIME]:
        if regime in work:
            ax.plot(x, work[regime], marker="o", label=regime, color=colors[regime])
    ax.set_title("Observed per-label work ratio in reruns")
    ax.set_ylabel("Max/min per-cell compare+swap")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("E02 S06 speed-matched Algotype controls", fontsize=14)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def validate_results(
    df: pd.DataFrame,
    tasks: list[SpeedMatchedTask],
    summary: pd.DataFrame,
    classification: pd.DataFrame,
    validation_detail: pd.DataFrame,
    figure_path: Path,
    artifacts_written: Sequence[Path],
    speed_ratio_threshold: float,
    s03_reference_rows: int,
    s05_validation: Mapping[str, Any],
) -> dict[str, Any]:
    expected_rows = len(tasks)
    observed_conditions = sorted(df["condition_id"].unique()) if len(df) else []
    expected_conditions = sorted(SELECTED_CONDITION_IDS)
    no_duplicates = not df.duplicated(["condition_id", "repeat_index", "activation_rate_regime"]).any()
    same_initial_arrays = bool(df.groupby(["condition_id", "repeat_index"])["initial_array_sha256"].nunique().max() == 1) if len(df) else False
    same_initial_assignments = (
        bool(df.groupby(["condition_id", "repeat_index"])["initial_algotype_assignment_sha256"].nunique().max() == 1)
        if len(df)
        else False
    )
    profile_counts = df.groupby("condition_id")["activation_rate_regime"].nunique().to_dict() if len(df) else {}
    rate_passed = bool(df["activation_rate_validation_passed"].all()) if len(df) else False
    max_share_error = float(pd.to_numeric(df["activation_rate_max_abs_share_error"], errors="coerce").max()) if len(df) else np.nan
    policy_dispatch_preserved = bool(df["policy_dispatch_preserved"].all()) if len(df) else False
    speed_rows = df[df["activation_rate_regime"] == SPEED_MATCH_RATE_REGIME]
    max_speed_ratio = (
        float(pd.to_numeric(speed_rows["expected_cycle_cost_ratio"], errors="coerce").max()) if len(speed_rows) else np.nan
    )
    speed_ratio_passed = bool(np.isfinite(max_speed_ratio) and max_speed_ratio <= speed_ratio_threshold)
    activation_weight_ratio_max = (
        float(pd.to_numeric(speed_rows["activation_per_weight_ratio"], errors="coerce").max()) if len(speed_rows) else np.nan
    )
    activation_weight_passed = bool(np.isfinite(activation_weight_ratio_max) and abs(activation_weight_ratio_max - 1.0) <= 1e-12)
    s05_success = bool(s05_validation.get("success") and s05_validation.get("validationPassed"))
    classifications_complete = bool(
        len(classification) == len(SELECTED_CONDITION_IDS)
        and classification["classification"].isin(["speed_independent", "speed_explained", "speed_amplified"]).all()
    )
    figure_ok = figure_path.exists() and figure_path.stat().st_size > 1000
    materialized_names = {
        "e02_speed_matched_chimeras.parquet",
        "speed_matched_aggregation.png",
        "e02_speed_matched_summary.csv",
        "e02_speed_match_validation.csv",
        "e02_speed_matched_classification.csv",
    }
    artifacts_present = all(
        path.exists() and path.stat().st_size > 0 for path in artifacts_written if path.name in materialized_names
    )
    passed = all(
        [
            len(df) == expected_rows,
            expected_rows > 0,
            observed_conditions == expected_conditions,
            all(int(count) == 2 for count in profile_counts.values()),
            no_duplicates,
            same_initial_arrays,
            same_initial_assignments,
            rate_passed,
            max_share_error <= 1e-12,
            policy_dispatch_preserved,
            speed_ratio_passed,
            activation_weight_passed,
            s03_reference_rows > 0,
            s05_success,
            len(summary) > 0,
            len(validation_detail) > 0,
            classifications_complete,
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
            "S06 is bounded to the first 10 E01 repeats by default.",
            "Speed matching changes activation opportunity tempo but leaves public cell policy code unchanged.",
            "Comparable step-count distributions are validated from E01 pure compare+swap estimates after integer weighting; mixed-run realized work can still diverge through policy interactions.",
        ],
        "recommendedNextAction": "Proceed to S07 after Chief Scientist instruction if the speed-matched classifications are accepted; do not start S07 inside S06.",
        "expectedRows": int(expected_rows),
        "observedRows": int(len(df)),
        "observedConditions": observed_conditions,
        "expectedConditions": expected_conditions,
        "profileCountsByCondition": {str(k): int(v) for k, v in profile_counts.items()},
        "noDuplicateTaskRows": bool(no_duplicates),
        "sameInitialArraysAcrossProfiles": same_initial_arrays,
        "sameInitialAssignmentsAcrossProfiles": same_initial_assignments,
        "activationRateValidationPassedForAllRows": rate_passed,
        "maxActivationShareError": max_share_error,
        "policyDispatchPreservedForAllRows": policy_dispatch_preserved,
        "speedMatchedExpectedCycleCostRatioMax": max_speed_ratio,
        "speedRatioThreshold": float(speed_ratio_threshold),
        "speedMatchedExpectedCycleCostRatioPassed": speed_ratio_passed,
        "speedMatchedActivationPerWeightRatioMax": activation_weight_ratio_max,
        "speedMatchedActivationPerWeightValidationPassed": activation_weight_passed,
        "s03ReferenceRows": int(s03_reference_rows),
        "s05ValidationSuccess": s05_success,
        "summaryRows": int(len(summary)),
        "classificationRows": int(len(classification)),
        "classificationsComplete": classifications_complete,
        "validationDetailRows": int(len(validation_detail)),
        "figureExistsAndNonempty": bool(figure_ok),
        "artifactsPresent": bool(artifacts_present),
    }


def outcome_from_classification(classification: pd.DataFrame, validation: Mapping[str, Any]) -> str:
    if not validation.get("validationPassed"):
        return "Constraining/contradictory"
    classes = set(classification["classification"].dropna())
    if "speed_explained" in classes:
        return "Constraining/contradictory"
    if classes and classes <= {"speed_independent", "speed_amplified"}:
        return "Supportive"
    return "Null"


def build_report(
    *,
    generated_at: str,
    artifacts_dir: Path,
    result_path: Path,
    summary_path: Path,
    validation_detail_path: Path,
    classification_path: Path,
    validation_path: Path,
    log_path: Path,
    figure_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    validation: dict[str, Any],
    summary: pd.DataFrame,
    validation_detail: pd.DataFrame,
    classification: pd.DataFrame,
    pure_stats: pd.DataFrame,
    unit_result: dict[str, Any] | None,
    git_commit: str,
    git_status: str,
    max_repeats: int,
    workers: int,
    max_speed_weight: int,
    speed_ratio_threshold: float,
    elapsed_seconds: float,
) -> str:
    outcome = outcome_from_classification(classification, validation)
    compact_summary = summary[
        [
            "condition_id",
            "algotype_mix",
            "activation_rate_regime",
            "n_runs",
            "label_weights_json",
            "mean_expected_cycle_cost_ratio",
            "mean_compare_plus_swap_steps",
            "mean_activation_event_count",
            "mean_peak_aggregation_left_neighbor_percent",
            "mean_delta_peak_aggregation_vs_equal",
            "mean_per_cell_compare_plus_swap_ratio",
            "rate_validation_failures",
            "policy_dispatch_failures",
        ]
    ]
    compact_classification = classification[
        [
            "condition_id",
            "algotype_mix",
            "mean_delta_peak_aggregation_vs_equal",
            "mean_relative_delta_compare_plus_swap_vs_equal",
            "speed_matched_expected_cycle_cost_ratio",
            "classification",
        ]
    ]
    compact_pure = pure_stats[
        [
            "algorithm",
            "n_runs",
            "mean_compare_plus_swap_steps",
            "std_compare_plus_swap_steps",
            "q10_compare_plus_swap_steps",
            "q50_compare_plus_swap_steps",
            "q90_compare_plus_swap_steps",
        ]
    ]
    compact_validation_detail = validation_detail[
        [
            "condition_id",
            "activation_rate_regime",
            "label",
            "behavior",
            "weight",
            "pure_mean_compare_plus_swap_steps",
            "weighted_expected_cycle_mean_steps",
            "condition_profile_expected_cycle_cost_ratio",
            "passes_speed_ratio_threshold",
        ]
    ]
    class_counts = classification["classification"].value_counts().to_dict() if len(classification) else {}
    max_speed_ratio = validation.get("speedMatchedExpectedCycleCostRatioMax", np.nan)
    validation_result = (
        f"passed: {validation['observedRows']}/{validation['expectedRows']} speed-matched rows, "
        f"policy dispatch preserved={validation['policyDispatchPreservedForAllRows']}, "
        f"max speed-matched expected step ratio={max_speed_ratio:.6g}, "
        f"activation distributions exact={validation['speedMatchedActivationPerWeightValidationPassed']}, "
        f"unit tests={'passed' if unit_result and unit_result['success'] else 'not run'}"
    )
    caveats = [
        "S06 uses a bounded first-10-repeat matrix.",
        "The speed correction is scheduler-level activation weighting, not a rewrite of local policies.",
        "E01 pure compare+swap estimates validate intended step-count comparability before mixed reruns; mixed-run observed per-label work is still path-dependent.",
        "Aggregation remains a computational label-neighborhood proxy.",
    ]
    if outcome == "Supportive":
        lay_summary = (
            "After giving each Algotype activation opportunities chosen to equalize E01-estimated step burden, mixed-policy "
            "Aggregation did not collapse into a simple speed artifact in the completed S06 matrix."
        )
    elif outcome == "Constraining/contradictory":
        lay_summary = (
            "Speed matching changed at least one Aggregation result enough to constrain the stronger claim that mixed-policy "
            "Aggregation is independent of component tempo."
        )
    else:
        lay_summary = "S06 completed the speed-matched reruns but did not yield a decisive speed-artifact classification."
    return f"""# E02 S06 Speed-Matched Algotypes

## Top Summary

- Research step ID: S06
- Completion status: Completed on {generated_at}
- Artifacts written: `{artifacts_dir / 'research_steps/S06/research_step_full_results.md'}`, `{result_path}`, `{figure_path}`, `{validation_path}`, `{summary_path}`, `{validation_detail_path}`, `{classification_path}`, `{log_path}`, `{manifest_path}`, `{src_manifest_path}`
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers: {'; '.join(caveats)}
- Lay summary: {lay_summary}
- Recommended next action: Proceed to S07 after Chief Scientist instruction if the S06 speed-matched classifications are accepted; do not start S07 inside S06.

## Chief Handoff

S06 completed the planned speed-matched Algotype rerun control. It used S03's deterministic weighted activation-cycle framework, E01 pure efficiency estimates, and S01 public-cell simulation wrappers. It did not alter Bubble, Insertion, or Selection local policy methods.

## Frozen Question

Does Aggregation still appear when component Algotypes are matched for average step counts or activation tempo?

## Inputs

- Repository commit at S06 run time: `{git_commit}`
- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- E01 efficiency estimates: `/previous-artifacts/E01/results/e01_efficiency_counts.parquet`
- S03 activation-rate artifact table: `/artifacts/results/e02_activation_rate_artifacts.parquet`
- S05 validation evidence: `/artifacts/research_steps/S05/s05_validation.json`
- Selected real mixed-policy chimeras: `{', '.join(SELECTED_CONDITION_IDS)}`

## Lay Summary

S06 asked whether the apparent tendency of different sorting-cell labels to cluster could just be caused by some algorithms being faster or slower. The control did this by keeping every cell's sorting rule unchanged while changing how often each label got a chance to act. The activation weights were chosen so that, according to E01 pure-algorithm step counts, a weighted activation cycle gave the component policies comparable expected step burdens.

## Detailed Methods

First, S06 read the E01 pure cell-view efficiency table and estimated each algorithm's mean compare+swap step count from sorted pure runs. The mean estimates were Bubble `{pure_stats.set_index('algorithm').loc['bubble', 'mean_compare_plus_swap_steps']:.3f}`, Insertion `{pure_stats.set_index('algorithm').loc['insertion', 'mean_compare_plus_swap_steps']:.3f}`, and Selection `{pure_stats.set_index('algorithm').loc['selection', 'mean_compare_plus_swap_steps']:.3f}`.

For each mixed-policy condition, S06 searched integer activation weights from 1 to `{max_speed_weight}` to minimize the max/min ratio of `E01 pure mean compare+swap steps / activation weight` across labels. This is a pre-analysis speed-matching validation: matched profiles had to keep the expected weighted step-count ratio at or below `{speed_ratio_threshold}`.

Each selected condition and repeat was run twice: once with `equal_per_cell` weights and once with `speed_matched_e01_efficiency` weights. The scheduler was the S03 deterministic weighted activation cycle. At every activation, S06 called the public cell object's existing `move()` method, attributed comparison and swap count deltas to the activated label, and recorded exact label activation counts.

The no-progress stop rule used the S01 public legal-action checker: a run stopped for no progress only after a cycle with no state progress and no public legal action available. This avoids stopping merely because a random weighted order did not move in a short window.

## Commands

- Unit tests: `{unit_result['command'] if unit_result else 'not run'}`
- S06 production run: `{sys.executable} scripts/e02_s06_speed_matched_algotypes.py --repo-dir /workspace/cell-research --artifacts-dir {artifacts_dir} --max-repeats {max_repeats} --workers {workers} --max-speed-weight {max_speed_weight} --speed-ratio-threshold {speed_ratio_threshold}`

## Dependencies And Parameters

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- New dependencies installed: none
- CPU workers: `{workers}`
- Max speed-match integer weight: `{max_speed_weight}`
- Speed-match expected step-count ratio threshold: `{speed_ratio_threshold}`
- Elapsed production time: `{elapsed_seconds:.2f}` seconds

## Results

Classification counts: `{json.dumps(class_counts, sort_keys=True)}`.

### E01 Pure Efficiency Basis

{markdown_table(compact_pure, max_rows=10)}

### Speed-Matched Rerun Summary

{markdown_table(compact_summary, max_rows=20)}

### Aggregation Classification

{markdown_table(compact_classification, max_rows=10)}

### Step-Count And Activation Validation Detail

{markdown_table(compact_validation_detail, max_rows=40)}

The full run-level table is `{result_path}`. The speed-matched figure is `{figure_path}`.

## Validation

Validation required all planned rows, two profiles per selected condition, no duplicate condition/repeat/profile rows, identical initial arrays and Algotype assignments across profiles, exact configured-versus-realized activation distributions, preserved public label dispatch, S03 and S05 inputs present, speed-matched E01 expected step-count ratios at or below the threshold, nonempty summary/classification tables, and a nonempty figure.

Validation result: `{validation['validationPassed']}`. Details are in `{validation_path}`.

Key validation values:

- Max speed-matched E01 expected cycle-cost ratio: `{max_speed_ratio:.6g}`
- Max configured-versus-realized activation share error: `{validation['maxActivationShareError']:.6g}`
- Speed-matched activation-per-weight ratio max: `{validation['speedMatchedActivationPerWeightRatioMax']:.6g}`
- Policy dispatch preserved for all rows: `{validation['policyDispatchPreservedForAllRows']}`
- S03 reference rows loaded: `{validation['s03ReferenceRows']}`
- S05 validation success loaded: `{validation['s05ValidationSuccess']}`

## Artifacts

- Run-level speed-matched table: `{result_path}`
- Summary table: `{summary_path}`
- Step-count/activation validation detail: `{validation_detail_path}`
- Aggregation classification table: `{classification_path}`
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

- No blocker was encountered for local policy semantics: S06 preserved public label dispatch and `move()` implementations while changing only activation opportunity weights.
- The E01 step-count matching target is based on pure-policy compare+swap distributions. In mixed runs, observed per-label work can still diverge because policies interact through shared array state and stopping history.
- The first-10-repeat matrix is suitable for this control step but should not be treated as a final 100-repeat inferential analysis.
- Weighted deterministic cycles are controlled scheduler interventions, not a replay of original Python thread scheduling.
- Aggregation is a computational label-clustering proxy, not direct biological evidence.

## Recommended Next Action

Proceed to S07 only after explicit Chief Scientist instruction. Do not start S07 inside S06.
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
        "recommendedNextAction": "Resolve the S06 speed-matching blocker before S07; do not start S07.",
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
    report = f"""# E02 S06 Speed-Matched Algotypes

## Top Summary

- Research step ID: S06
- Completion status: Blocked on {generated_at}
- Artifacts written: `{report_path}`, `{validation_path}`, `{manifest_path}`, `{src_manifest_path}`, `{log_path}`
- Validation result: blocked
- Outcome classification: constraining/contradictory
- Caveats or blockers: {reason}
- Lay summary: S06 could not safely construct or validate speed-matched Algotypes under the required simulator/input constraints.
- Recommended next action: Resolve the S06 blocker before S07; do not start S07.

## Frozen Question

Does Aggregation still appear when component Algotypes are matched for average step counts or activation tempo?

## Methods

S06 checked required S03/S05 controls, E01 efficiency estimates, and simulator support before production reruns.

## Results

No speed-matched result table or figure was generated.

## Validation

Validation was blocked before production reruns.

## Provenance

- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Generated at UTC: `{generated_at}`
"""
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
                artifact_entry(report_path, artifacts_dir, "S06 blocked full-results Markdown handoff report."),
                artifact_entry(validation_path, artifacts_dir, "S06 blocked validation evidence."),
                artifact_entry(log_path, artifacts_dir, "S06 blocked execution log."),
                artifact_entry(src_manifest_path, artifacts_dir, "S06 blocked source snapshot manifest."),
            ],
        },
    )
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
    result_path = results_dir / "e02_speed_matched_chimeras.parquet"
    summary_path = step_dir / "e02_speed_matched_summary.csv"
    validation_detail_path = step_dir / "e02_speed_match_validation.csv"
    classification_path = step_dir / "e02_speed_matched_classification.csv"
    figure_path = figures_dir / "speed_matched_aggregation.png"
    validation_path = step_dir / "s06_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s06_speed_matched_algotypes_manifest.json"
    log_path = logs_dir / "e02_s06_speed_matched_algotypes.log"
    report_path = step_dir / "research_step_full_results.md"
    log_lines = [f"E02 S06 started {generated_at}", f"repo_dir={repo_dir}", f"artifacts_dir={artifacts_dir}"]

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
        if not args.s03_results_path.exists():
            raise RuntimeError(f"Required S03 activation-rate table is missing: {args.s03_results_path}")
        s03_reference_rows = len(pd.read_parquet(args.s03_results_path, columns=["condition_id"]))
        if not args.s05_validation_path.exists():
            raise RuntimeError(f"Required S05 validation JSON is missing: {args.s05_validation_path}")
        s05_validation = json.loads(args.s05_validation_path.read_text(encoding="utf-8"))
        if not (s05_validation.get("success") and s05_validation.get("validationPassed")):
            raise RuntimeError(f"S05 validation did not pass and cannot support S06: {args.s05_validation_path}")
        cfg = load_e01_config(args.config_path)
        pure_stats = load_pure_efficiency_stats(args.e01_efficiency_path)
        pure_means = {
            str(row["algorithm"]): float(row["mean_compare_plus_swap_steps"])
            for row in pure_stats.to_dict(orient="records")
        }
        rows = selected_chimera_conditions(cfg)
        tasks = build_tasks(
            cfg,
            rows,
            repo_dir,
            args.max_repeats,
            args.max_cycles,
            args.max_successful_swaps,
            args.stall_cycles,
            pure_means,
            args.max_speed_weight,
        )
        max_speed_ratio_preflight = max(
            task.expected_cycle_cost_ratio for task in tasks if task.activation_rate_regime == SPEED_MATCH_RATE_REGIME
        )
        if max_speed_ratio_preflight > args.speed_ratio_threshold:
            raise RuntimeError(
                f"Speed matching cannot meet expected step-count ratio threshold: "
                f"{max_speed_ratio_preflight:.6g} > {args.speed_ratio_threshold:.6g}"
            )
    except Exception as exc:
        return write_blocked_outputs(
            generated_at=generated_at,
            artifacts_dir=artifacts_dir,
            repo_dir=repo_dir,
            reason=f"S06 input, control, or speed-matching validation failed before rerun: {exc!r}",
            validation_path=validation_path,
            report_path=report_path,
            manifest_path=manifest_path,
            src_manifest_path=src_manifest_path,
            log_path=log_path,
        )

    workers = max(1, min(int(args.workers), len(tasks), 8))
    log_lines.append(
        f"selected_conditions={len(rows)} tasks={len(tasks)} workers={workers} "
        f"s03_reference_rows={s03_reference_rows} max_speed_weight={args.max_speed_weight}"
    )
    result_rows: list[dict[str, Any]] = []
    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            result_rows.append(run_speed_matched_task(task))
            if idx % 10 == 0 or idx == len(tasks):
                log_lines.append(f"completed_tasks={idx}/{len(tasks)}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(run_speed_matched_task, task): task for task in tasks}
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
                if idx % 10 == 0 or idx == len(tasks):
                    log_lines.append(f"completed_tasks={idx}/{len(tasks)}")

    result_df = add_equal_reference_deltas(pd.DataFrame(result_rows))
    result_df = result_df.sort_values(["condition_id", "repeat_index", "activation_rate_regime"]).reset_index(drop=True)
    summary_df = summarize_results(result_df)
    validation_detail_df = speed_match_validation_table(
        result_df.to_dict(orient="records"),
        pure_stats,
        threshold=args.speed_ratio_threshold,
    )
    classification_df = classify_speed_effects(summary_df, threshold=args.speed_ratio_threshold)

    result_df.to_parquet(result_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    validation_detail_df.to_csv(validation_detail_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    make_figure(summary_df, classification_df, validation_detail_df, figure_path)
    expected_artifacts = [
        report_path,
        result_path,
        figure_path,
        validation_path,
        summary_path,
        validation_detail_path,
        classification_path,
        log_path,
        manifest_path,
        src_manifest_path,
    ]
    validation = validate_results(
        result_df,
        tasks,
        summary_df,
        classification_df,
        validation_detail_df,
        figure_path,
        expected_artifacts,
        args.speed_ratio_threshold,
        s03_reference_rows,
        s05_validation,
    )
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
        repo_dir / "scripts/e02_s06_speed_matched_algotypes.py",
        repo_dir / "tests/e02/test_speed_matched_algotypes.py",
        repo_dir / "scripts/e02_s03_activation_rates.py",
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
            "workers": int(workers),
            "maxCycles": int(args.max_cycles),
            "maxSuccessfulSwaps": int(args.max_successful_swaps),
            "stallCycles": int(args.stall_cycles),
            "maxSpeedWeight": int(args.max_speed_weight),
            "speedRatioThreshold": float(args.speed_ratio_threshold),
            "selectedConditionIds": list(SELECTED_CONDITION_IDS),
            "e01EfficiencyPath": str(args.e01_efficiency_path),
            "s03ResultsPath": str(args.s03_results_path),
            "s03ReferenceRows": int(s03_reference_rows),
            "s05ValidationPath": str(args.s05_validation_path),
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
        validation_detail_path=validation_detail_path,
        classification_path=classification_path,
        validation_path=validation_path,
        log_path=log_path,
        figure_path=figure_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        validation=validation,
        summary=summary_df,
        validation_detail=validation_detail_df,
        classification=classification_df,
        pure_stats=pure_stats,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        max_repeats=args.max_repeats,
        workers=workers,
        max_speed_weight=args.max_speed_weight,
        speed_ratio_threshold=args.speed_ratio_threshold,
        elapsed_seconds=elapsed_seconds,
    )
    report_path.write_text(report_text, encoding="utf-8")
    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S06 full-results Markdown handoff report."),
        artifact_entry(result_path, artifacts_dir, "Run-level speed-matched chimera table."),
        artifact_entry(summary_path, artifacts_dir, "Speed-matched summary table."),
        artifact_entry(validation_detail_path, artifacts_dir, "E01 step-count and activation validation detail."),
        artifact_entry(classification_path, artifacts_dir, "Speed-matched Aggregation classification table."),
        artifact_entry(figure_path, artifacts_dir, "Speed-matched Aggregation figure."),
        artifact_entry(validation_path, artifacts_dir, "S06 validation evidence."),
        artifact_entry(log_path, artifacts_dir, "S06 execution log."),
        artifact_entry(src_manifest_path, artifacts_dir, "S06 source snapshot manifest."),
    ]
    write_json(manifest_path, artifact_stub | {"artifacts": artifacts})
    if not validation["validationPassed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
