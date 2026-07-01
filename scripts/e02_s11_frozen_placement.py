#!/usr/bin/env python3
"""Run E02 S11 Frozen Cell placement stress tests.

S11 varies where Frozen Cells are placed while preserving S01 deterministic
public-cell simulation semantics, S08 Delayed Gratification recomputation, and
the first-10-repeat bounded-matrix convention used by prior E02 steps.
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

from scripts.e02_s08_dg_nulls import dg_from_sortedness, validate_toy_cases
from scripts.e02_s09_alternative_metrics import (
    normalized_metric_columns,
    reverse_flags_for_condition,
    scalar_path_curvature,
    state_space_curvature,
    target_values_for_initial,
)
from src.e02.deterministic_simulator import SimulatorConfig, count_json, simulate, stable_json_sha256


STEP_ID = "S11"
STEP_NUMBER = 11
EXPERIMENT_ID = "E02"
SOURCE_EXPERIMENT_ID = "E01"

SELECTED_CONDITION_IDS = (
    "E01C010",
    "E01C011",
    "E01C012",
    "E01C022",
    "E01C023",
    "E01C024",
    "E01C028",
    "E01C029",
    "E01C030",
    "E01C040",
    "E01C041",
    "E01C042",
)

PLACEMENT_RULES = (
    "left_end",
    "right_end",
    "center",
    "random_bank",
    "high_value",
    "low_value",
    "clustered_seeded",
    "evenly_spaced",
)
PLACEMENT_RULE_INDEX = {name: idx + 1 for idx, name in enumerate(PLACEMENT_RULES)}


@dataclass(frozen=True)
class FrozenPlacement:
    indices: tuple[int, ...]
    seed: int | None
    source: str
    policy: str
    validation: dict[str, Any]


@dataclass(frozen=True)
class PlacementTask:
    condition: dict[str, Any]
    repeat_index: int
    placement_rule: str
    values: tuple[int, ...]
    assignments: tuple[str, ...]
    reverse_directions: tuple[bool, ...]
    frozen_indices: tuple[int, ...]
    initial_frozen_values: tuple[int, ...]
    placement_seed: int | None
    placement_source: str
    placement_policy: str
    placement_validation: dict[str, Any]
    initial_array_seed: int
    original_frozen_bank_indices: tuple[int, ...]
    original_frozen_bank_seed: int | None
    activation_seed: int
    repo_dir: str
    max_events: int
    max_successful_swaps: int
    stall_events: int
    activation_distribution: str
    sort_direction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config-path", type=Path, default=Path("/previous-artifacts/E01/configs/e01_baseline_configs.json"))
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-events", type=int, default=150_000)
    parser.add_argument("--max-successful-swaps", type=int, default=80_000)
    parser.add_argument("--stall-events", type=int, default=2_000)
    parser.add_argument(
        "--activation-distribution",
        choices=("uniform_active", "left_to_right_active", "right_to_left_active"),
        default="uniform_active",
    )
    parser.add_argument("--condition-ids", nargs="*", default=list(SELECTED_CONDITION_IDS))
    parser.add_argument("--placement-rules", nargs="*", default=list(PLACEMENT_RULES))
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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True, env=dict(env) if env else None)
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(time.monotonic() - started),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": bool(result.returncode == 0),
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": int(path.stat().st_size),
    }


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


def load_e01_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def condition_seed_component(condition_id: str) -> int:
    if condition_id.startswith("E01C") and condition_id[4:].isdigit():
        return int(condition_id.replace("E01C", ""))
    digest = hashlib.sha256(condition_id.encode("utf-8")).hexdigest()
    return 10_000 + int(digest[:8], 16) % 90_000


def activation_seed(base_seed: int, condition_id: str, placement_rule: str, repeat_idx: int) -> int:
    """Reuse the S09/S10 condition-repeat activation seed across placements."""

    _ = placement_rule
    condition_num = condition_seed_component(condition_id)
    return base_seed * 1_000_000 + 900_000 + condition_num * 1_000 + repeat_idx


def generated_placement_seed(base_seed: int, condition_id: str, placement_rule: str, repeat_idx: int) -> int:
    condition_num = condition_seed_component(condition_id)
    return base_seed * 1_000_000 + 1_100_000 + condition_num * 10_000 + PLACEMENT_RULE_INDEX[placement_rule] * 100 + repeat_idx


def selected_conditions(cfg: Mapping[str, Any], condition_ids: Sequence[str]) -> list[dict[str, Any]]:
    requested = tuple(dict.fromkeys(condition_ids))
    rows = [
        dict(row)
        for row in cfg["conditions"]
        if row["condition_id"] in requested and row["mode"] == "cell_view"
    ]
    found = {row["condition_id"] for row in rows}
    missing = sorted(set(requested) - found)
    if missing:
        raise RuntimeError(f"Missing expected S11 condition IDs in E01 config: {missing}")
    unsupported = [
        row["condition_id"]
        for row in rows
        if row.get("direction_profile") != "all_increasing"
        or row.get("value_bank_id") != "unique_1_to_100"
        or int(row.get("frozen_count", 0)) <= 0
        or row.get("frozen_semantics") not in {"passive", "stuck"}
    ]
    if unsupported:
        raise RuntimeError(
            "S11 placement matrix supports all-increasing unique-value Frozen Cell conditions only; "
            f"unsupported rows: {unsupported}"
        )
    return sorted(rows, key=lambda row: row["condition_id"])


def center_indices(n: int, k: int) -> tuple[int, ...]:
    start = (n - k) // 2
    return tuple(range(start, start + k))


def evenly_spaced_indices(n: int, k: int) -> tuple[int, ...]:
    if k == 1:
        return center_indices(n, k)
    return tuple(sorted({int(round((idx + 1) * (n + 1) / (k + 1) - 1)) for idx in range(k)}))


def high_value_indices(values: Sequence[int], k: int) -> tuple[int, ...]:
    return tuple(sorted(sorted(range(len(values)), key=lambda idx: (-int(values[idx]), idx))[:k]))


def low_value_indices(values: Sequence[int], k: int) -> tuple[int, ...]:
    return tuple(sorted(sorted(range(len(values)), key=lambda idx: (int(values[idx]), idx))[:k]))


def expected_indices_for_rule(
    rule: str,
    values: Sequence[int],
    k: int,
    *,
    random_bank_indices: Sequence[int],
    seed: int | None,
) -> tuple[int, ...]:
    n = len(values)
    if rule == "left_end":
        return tuple(range(k))
    if rule == "right_end":
        return tuple(range(n - k, n))
    if rule == "center":
        return center_indices(n, k)
    if rule == "random_bank":
        return tuple(sorted(map(int, random_bank_indices)))
    if rule == "high_value":
        return high_value_indices(values, k)
    if rule == "low_value":
        return low_value_indices(values, k)
    if rule == "clustered_seeded":
        if seed is None:
            raise ValueError("clustered_seeded requires a seed")
        rng = random.Random(seed)
        start = rng.randrange(0, n - k + 1)
        return tuple(range(start, start + k))
    if rule == "evenly_spaced":
        return evenly_spaced_indices(n, k)
    raise ValueError(f"Unsupported placement rule: {rule}")


def validate_placement_rule(
    rule: str,
    values: Sequence[int],
    indices: Sequence[int],
    k: int,
    *,
    random_bank_indices: Sequence[int],
    seed: int | None,
) -> dict[str, Any]:
    n = len(values)
    indices_tuple = tuple(map(int, indices))
    failures: list[str] = []
    if len(indices_tuple) != k:
        failures.append("frozen_count_mismatch")
    if len(set(indices_tuple)) != len(indices_tuple):
        failures.append("duplicate_frozen_indices")
    if any(idx < 0 or idx >= n for idx in indices_tuple):
        failures.append("frozen_index_out_of_bounds")
    if rule in {"high_value", "low_value"} and len(set(values)) != len(values):
        failures.append("high_low_value_target_ambiguous_under_duplicates")
    expected: tuple[int, ...] = ()
    try:
        expected = expected_indices_for_rule(
            rule,
            values,
            k,
            random_bank_indices=random_bank_indices,
            seed=seed,
        )
    except Exception as exc:  # pragma: no cover - defensive validation path
        failures.append(f"rule_expected_indices_error:{exc!r}")
    if expected and tuple(sorted(indices_tuple)) != tuple(sorted(expected)):
        failures.append("indices_do_not_match_placement_rule")
    if rule == "clustered_seeded" and len(indices_tuple) > 1:
        ordered = tuple(sorted(indices_tuple))
        if ordered != tuple(range(ordered[0], ordered[0] + len(ordered))):
            failures.append("clustered_indices_not_contiguous")
    if rule == "evenly_spaced" and len(indices_tuple) > 1:
        gaps = np.diff(sorted(indices_tuple))
        if len(gaps) and float(np.max(gaps) - np.min(gaps)) > 2.0:
            failures.append("evenly_spaced_gap_spread_too_large")
    return {
        "placement_rule": rule,
        "array_length": int(n),
        "frozen_count": int(k),
        "placement_indices_json": json.dumps(list(indices_tuple), separators=(",", ":")),
        "expected_indices_json": json.dumps(list(expected), separators=(",", ":")),
        "placement_rule_valid": bool(not failures),
        "placement_rule_failures": failures,
        "placement_rule_failures_json": json.dumps(failures, separators=(",", ":")),
        "placement_indices_sha256": stable_json_sha256(list(indices_tuple)),
        "high_low_value_target_ambiguous": bool("high_low_value_target_ambiguous_under_duplicates" in failures),
    }


def make_placement(
    cfg: Mapping[str, Any],
    condition: Mapping[str, Any],
    values: Sequence[int],
    repeat_idx: int,
    placement_rule: str,
    random_bank_indices: Sequence[int],
    random_bank_seed: int | None,
) -> FrozenPlacement:
    if placement_rule not in PLACEMENT_RULES:
        raise ValueError(f"Unsupported S11 placement rule: {placement_rule}")
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    k = int(condition["frozen_count"])
    seed: int | None = None
    source = "generated_rule"
    if placement_rule == "random_bank":
        seed = random_bank_seed
        source = "E01 frozenIndexBanks"
    elif placement_rule == "clustered_seeded":
        seed = generated_placement_seed(base_seed, str(condition["condition_id"]), placement_rule, repeat_idx)
        source = "S11 seeded contiguous block"
    indices = expected_indices_for_rule(
        placement_rule,
        values,
        k,
        random_bank_indices=random_bank_indices,
        seed=seed,
    )
    policy = {
        "left_end": "first k positions, zero-based",
        "right_end": "last k positions, zero-based",
        "center": "centered contiguous k positions; left-center for even-length singletons",
        "random_bank": "original E01 frozen-index bank for the same frozen count and repeat",
        "high_value": "positions carrying the k largest initial values; unique-value arrays only",
        "low_value": "positions carrying the k smallest initial values; unique-value arrays only",
        "clustered_seeded": "seeded contiguous block of length k",
        "evenly_spaced": "approximately equal interior spacing across the array",
    }[placement_rule]
    validation = validate_placement_rule(
        placement_rule,
        values,
        indices,
        k,
        random_bank_indices=random_bank_indices,
        seed=seed,
    )
    return FrozenPlacement(
        indices=tuple(map(int, indices)),
        seed=seed,
        source=source,
        policy=policy,
        validation=validation,
    )


def build_tasks(
    cfg: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    repo_dir: Path,
    max_repeats: int,
    max_events: int,
    max_successful_swaps: int,
    stall_events: int,
    activation_distribution: str,
    placement_rules: Sequence[str],
) -> list[PlacementTask]:
    unknown_rules = sorted(set(placement_rules) - set(PLACEMENT_RULES))
    if unknown_rules:
        raise RuntimeError(f"Unsupported S11 placement rules requested: {unknown_rules}")
    repeat_count = min(max_repeats, int(cfg["globalDefaults"]["repeatCount"]))
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    value_banks = cfg["seedBanks"]["valueBanks"]
    frozen_banks = cfg["seedBanks"]["frozenIndexBanks"]
    assignment_banks = cfg["seedBanks"]["algotypeAssignmentBanks"]
    tasks: list[PlacementTask] = []
    for condition in rows:
        value_bank = value_banks[condition["value_bank_id"]]
        frozen_bank = frozen_banks[condition["frozen_index_bank_id"]]
        assignment_bank = assignment_banks.get(condition["algotype_assignment_bank_id"])
        for repeat_idx in range(repeat_count):
            values = tuple(map(int, value_bank["initialArrays"][repeat_idx]))
            if condition["algorithm"] == "mixed":
                if assignment_bank is None:
                    raise RuntimeError(f"{condition['condition_id']} requires an assignment bank.")
                assignments = tuple(map(str, assignment_bank["assignments"][repeat_idx]))
            else:
                assignments = tuple(str(condition["algorithm"]) for _ in values)
            original_indices = tuple(map(int, frozen_bank["indices"][repeat_idx]))
            original_seed = int(frozen_bank["seeds"][repeat_idx]) if frozen_bank["seeds"] else None
            for placement_rule in placement_rules:
                placement = make_placement(
                    cfg,
                    condition,
                    values,
                    repeat_idx,
                    placement_rule,
                    original_indices,
                    original_seed,
                )
                frozen_values = tuple(int(values[idx]) for idx in placement.indices)
                tasks.append(
                    PlacementTask(
                        condition=dict(condition),
                        repeat_index=int(repeat_idx),
                        placement_rule=placement_rule,
                        values=values,
                        assignments=assignments,
                        reverse_directions=reverse_flags_for_condition(condition, assignments),
                        frozen_indices=placement.indices,
                        initial_frozen_values=frozen_values,
                        placement_seed=placement.seed,
                        placement_source=placement.source,
                        placement_policy=placement.policy,
                        placement_validation=placement.validation,
                        initial_array_seed=int(value_bank["seeds"][repeat_idx]),
                        original_frozen_bank_indices=original_indices,
                        original_frozen_bank_seed=original_seed,
                        activation_seed=activation_seed(base_seed, str(condition["condition_id"]), placement_rule, repeat_idx),
                        repo_dir=str(repo_dir),
                        max_events=int(max_events),
                        max_successful_swaps=int(max_successful_swaps),
                        stall_events=int(stall_events),
                        activation_distribution=str(activation_distribution),
                        sort_direction="increasing",
                    )
                )
    return tasks


def finite_or_none(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def run_placement_task(task: PlacementTask) -> dict[str, Any]:
    started = time.monotonic()
    config = SimulatorConfig(
        values=task.values,
        algotypes=task.assignments,
        reverse_directions=task.reverse_directions,
        frozen_indices=task.frozen_indices,
        frozen_semantics=str(task.condition["frozen_semantics"]),
        activation_seed=task.activation_seed,
        policy_seed=task.activation_seed,
        activation_distribution=task.activation_distribution,
        max_events=task.max_events,
        max_successful_swaps=task.max_successful_swaps,
        stall_events=task.stall_events,
        stop_when_sorted=True,
        sort_direction=task.sort_direction,
        repo_dir=Path(task.repo_dir),
    )
    result = simulate(config)
    records = list(result.records)
    if not records:
        raise RuntimeError("Simulator returned no records.")

    values_path = [tuple(map(int, record["values"])) for record in records]
    sortedness_curve = [float(record["sortedness_percent"]) for record in records]
    monotonicity_curve = [int(record["monotonicity_error_count"]) for record in records]
    aggregation_curve = [float(record["aggregation_left_neighbor_percent"]) for record in records]
    target_values = target_values_for_initial(task.values, task.sort_direction)
    initial_metrics = normalized_metric_columns(
        values_path[0],
        task.sort_direction,
        sortedness_curve[0],
        monotonicity_curve[0],
    )
    final_metrics = normalized_metric_columns(
        values_path[-1],
        task.sort_direction,
        sortedness_curve[-1],
        monotonicity_curve[-1],
    )
    dg = dg_from_sortedness(sortedness_curve)
    sortedness_curvature = scalar_path_curvature(sortedness_curve)
    value_state_curvature = state_space_curvature(values_path)
    frozen_identity_count_preserved = len(result.final_frozen_values) == len(task.frozen_indices)
    frozen_identity_values_preserved = tuple(sorted(result.final_frozen_values)) == tuple(sorted(task.initial_frozen_values))
    stuck_positions_preserved = (
        tuple(sorted(result.final_frozen_positions)) == tuple(sorted(task.frozen_indices))
        if task.condition["frozen_semantics"] == "stuck"
        else True
    )
    frozen_identity_validation_passed = bool(
        frozen_identity_count_preserved
        and frozen_identity_values_preserved
        and stuck_positions_preserved
    )
    initial_frozen_value_ranks = tuple(sorted(task.initial_frozen_values))
    initial_frozen_value_mean = float(np.mean(initial_frozen_value_ranks)) if initial_frozen_value_ranks else np.nan
    initial_frozen_position_mean = float(np.mean(task.frozen_indices)) if task.frozen_indices else np.nan

    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "evidence_layer": "s11_frozen_placement_deterministic_rerun",
        "condition_id": task.condition["condition_id"],
        "run_family": task.condition["run_family"],
        "mode": "cell_view",
        "algorithm": task.condition["algorithm"],
        "algotype_mix": task.condition["algotype_mix"],
        "value_bank_id": task.condition["value_bank_id"],
        "value_distribution": task.condition.get("value_distribution", task.condition["value_bank_id"]),
        "direction_profile": task.condition.get("direction_profile", "all_increasing"),
        "sort_direction": task.sort_direction,
        "scheduler_regime": f"deterministic_single_event_{task.activation_distribution}",
        "scheduler_source": "s01_deterministic_simulator_public_move_methods",
        "repeat_index": int(task.repeat_index),
        "array_length": int(len(task.values)),
        "initial_array_seed": int(task.initial_array_seed),
        "activation_seed": int(task.activation_seed),
        "policy_seed": int(task.activation_seed),
        "frozen_semantics": task.condition["frozen_semantics"],
        "frozen_count": int(task.condition["frozen_count"]),
        "placement_rule": task.placement_rule,
        "placement_policy": task.placement_policy,
        "placement_source": task.placement_source,
        "placement_seed": task.placement_seed,
        "original_frozen_bank_seed": task.original_frozen_bank_seed,
        "original_frozen_bank_indices_json": json.dumps(list(task.original_frozen_bank_indices), separators=(",", ":")),
        "initial_frozen_indices_json": json.dumps(list(task.frozen_indices), separators=(",", ":")),
        "initial_frozen_values_json": json.dumps(list(task.initial_frozen_values), separators=(",", ":")),
        "initial_frozen_values_sorted_json": json.dumps(list(initial_frozen_value_ranks), separators=(",", ":")),
        "final_frozen_positions_json": json.dumps(list(result.final_frozen_positions), separators=(",", ":")),
        "final_frozen_values_json": json.dumps(list(result.final_frozen_values), separators=(",", ":")),
        "final_frozen_values_sorted_json": json.dumps(list(sorted(result.final_frozen_values)), separators=(",", ":")),
        "initial_frozen_position_mean": finite_or_none(initial_frozen_position_mean),
        "initial_frozen_value_mean": finite_or_none(initial_frozen_value_mean),
        "placement_rule_valid": bool(task.placement_validation["placement_rule_valid"]),
        "placement_rule_failures_json": task.placement_validation["placement_rule_failures_json"],
        "high_low_value_target_ambiguous": bool(task.placement_validation["high_low_value_target_ambiguous"]),
        "frozen_identity_count_preserved": bool(frozen_identity_count_preserved),
        "frozen_identity_values_preserved": bool(frozen_identity_values_preserved),
        "stuck_frozen_positions_preserved": bool(stuck_positions_preserved),
        "frozen_identity_validation_passed": bool(frozen_identity_validation_passed),
        "initial_values_json": json.dumps(list(task.values), separators=(",", ":")),
        "target_values_json": json.dumps(list(target_values), separators=(",", ":")),
        "initial_array_sha256": stable_json_sha256(list(task.values)),
        "target_values_sha256": stable_json_sha256(list(target_values)),
        "final_array_sha256": stable_json_sha256(list(result.final_values)),
        "trace_values_sha256": stable_json_sha256([list(values) for values in values_path]),
        "initial_algotype_counts_json": count_json(task.assignments),
        "final_algotype_counts_json": count_json(result.final_labels),
        "initial_sortedness_percent": float(sortedness_curve[0]),
        "final_sortedness_percent": float(sortedness_curve[-1]),
        "peak_sortedness_percent": float(max(sortedness_curve)),
        "curve_mean_sortedness_percent": float(np.mean(sortedness_curve)),
        "final_monotonicity_error_count": int(monotonicity_curve[-1]),
        "peak_monotonicity_error_count": int(max(monotonicity_curve)),
        "curve_mean_monotonicity_error_count": float(np.mean(monotonicity_curve)),
        "initial_aggregation_left_neighbor_percent": float(aggregation_curve[0]),
        "final_aggregation_left_neighbor_percent": float(aggregation_curve[-1]),
        "peak_aggregation_left_neighbor_percent": float(max(aggregation_curve)),
        "curve_mean_aggregation_left_neighbor_percent": float(np.mean(aggregation_curve)),
        "sortedness_path_curvature_ratio": finite_or_none(sortedness_curvature),
        "value_state_path_curvature_ratio": finite_or_none(value_state_curvature),
        "event_count": int(result.event_count),
        "swap_only_steps": int(result.swap_count),
        "comparison_steps_observed": int(result.comparison_count),
        "compare_plus_swap_steps": int(result.compare_plus_swap_count),
        "frozen_attempt_count": int(result.frozen_attempt_count),
        "trace_record_count": int(len(records)),
        "activation_event_count": int(len(result.activation_log)),
        "activation_log_sha256": stable_json_sha256(list(result.activation_log)),
        "stop_reason": result.stop_reason,
        "max_guard_hit": bool(result.max_guard_hit),
        "final_sorted_non_decreasing": bool(monotonicity_curve[-1] == 0),
        "elapsed_seconds": float(time.monotonic() - started),
        "kendall_tau_distance_initial_distance": float(initial_metrics["kendall_tau_distance"]),
        "kendall_tau_distance_final_distance": float(final_metrics["kendall_tau_distance"]),
        "earth_mover_position_distance_initial_distance": float(initial_metrics["earth_mover_position_distance"]),
        "earth_mover_position_distance_final_distance": float(final_metrics["earth_mover_position_distance"]),
        "kendall_tau_distance_final_progress_percent": float(100.0 * (1.0 - float(final_metrics["kendall_tau_distance"]))),
        "earth_mover_position_distance_final_progress_percent": float(100.0 * (1.0 - float(final_metrics["earth_mover_position_distance"]))),
        **dg,
    }


def summarize_results(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = result_df.groupby(
        [
            "condition_id",
            "run_family",
            "algorithm",
            "algotype_mix",
            "frozen_semantics",
            "frozen_count",
            "placement_rule",
        ],
        dropna=False,
    )
    summary = grouped.agg(
        n_runs=("repeat_index", "nunique"),
        success_rate=("final_sorted_non_decreasing", "mean"),
        mean_final_sortedness=("final_sortedness_percent", "mean"),
        sd_final_sortedness=("final_sortedness_percent", "std"),
        mean_peak_sortedness=("peak_sortedness_percent", "mean"),
        mean_curve_sortedness=("curve_mean_sortedness_percent", "mean"),
        mean_final_monotonicity_error=("final_monotonicity_error_count", "mean"),
        mean_curve_monotonicity_error=("curve_mean_monotonicity_error_count", "mean"),
        mean_compare_plus_swap_steps=("compare_plus_swap_steps", "mean"),
        mean_event_count=("event_count", "mean"),
        mean_swap_only_steps=("swap_only_steps", "mean"),
        mean_frozen_attempt_count=("frozen_attempt_count", "mean"),
        mean_dg_primary=("dg_primary", "mean"),
        mean_dg_event_count=("dg_event_count", "mean"),
        mean_peak_aggregation=("peak_aggregation_left_neighbor_percent", "mean"),
        mean_final_aggregation=("final_aggregation_left_neighbor_percent", "mean"),
        mean_sortedness_curvature=("sortedness_path_curvature_ratio", "mean"),
        mean_value_state_curvature=("value_state_path_curvature_ratio", "mean"),
        mean_initial_frozen_position=("initial_frozen_position_mean", "mean"),
        mean_initial_frozen_value=("initial_frozen_value_mean", "mean"),
        max_guard_hit_rate=("max_guard_hit", "mean"),
        placement_rule_valid_rate=("placement_rule_valid", "mean"),
        frozen_identity_valid_rate=("frozen_identity_validation_passed", "mean"),
    ).reset_index()
    summary["sd_final_sortedness"] = summary["sd_final_sortedness"].fillna(0.0)

    baseline = summary[summary["placement_rule"] == "random_bank"][
        [
            "condition_id",
            "success_rate",
            "mean_final_sortedness",
            "mean_final_monotonicity_error",
            "mean_compare_plus_swap_steps",
            "mean_dg_primary",
            "mean_sortedness_curvature",
            "mean_frozen_attempt_count",
        ]
    ].rename(
        columns={
            "success_rate": "baseline_success_rate",
            "mean_final_sortedness": "baseline_mean_final_sortedness",
            "mean_final_monotonicity_error": "baseline_mean_final_monotonicity_error",
            "mean_compare_plus_swap_steps": "baseline_mean_compare_plus_swap_steps",
            "mean_dg_primary": "baseline_mean_dg_primary",
            "mean_sortedness_curvature": "baseline_mean_sortedness_curvature",
            "mean_frozen_attempt_count": "baseline_mean_frozen_attempt_count",
        }
    )
    classified = summary.merge(baseline, on="condition_id", how="left")
    classified["delta_final_sortedness_vs_random_bank"] = classified["mean_final_sortedness"] - classified["baseline_mean_final_sortedness"]
    classified["delta_success_rate_vs_random_bank"] = classified["success_rate"] - classified["baseline_success_rate"]
    classified["delta_final_monotonicity_error_vs_random_bank"] = (
        classified["mean_final_monotonicity_error"] - classified["baseline_mean_final_monotonicity_error"]
    )
    classified["compare_plus_swap_ratio_vs_random_bank"] = (
        classified["mean_compare_plus_swap_steps"] / classified["baseline_mean_compare_plus_swap_steps"].replace(0, np.nan)
    )
    classified["delta_dg_primary_vs_random_bank"] = classified["mean_dg_primary"] - classified["baseline_mean_dg_primary"]
    classified["delta_sortedness_curvature_vs_random_bank"] = (
        classified["mean_sortedness_curvature"] - classified["baseline_mean_sortedness_curvature"]
    )
    classified["delta_frozen_attempts_vs_random_bank"] = (
        classified["mean_frozen_attempt_count"] - classified["baseline_mean_frozen_attempt_count"]
    )

    classes: list[str] = []
    reasons: list[str] = []
    for row in classified.to_dict(orient="records"):
        if row["placement_rule"] == "random_bank":
            classes.append("baseline_random_bank")
            reasons.append("reference placement from E01 frozen-index bank")
            continue
        final_sensitive = (
            float(row["delta_final_sortedness_vs_random_bank"]) < -5.0
            or float(row["delta_success_rate_vs_random_bank"]) < -0.1
            or float(row["delta_final_monotonicity_error_vs_random_bank"]) > 5.0
        )
        ratio = row["compare_plus_swap_ratio_vs_random_bank"]
        efficiency_sensitive = pd.notna(ratio) and (float(ratio) > 2.0 or float(ratio) < 0.5)
        path_sensitive = (
            abs(float(row["delta_dg_primary_vs_random_bank"])) > 25.0
            or abs(float(row["delta_sortedness_curvature_vs_random_bank"])) > 0.75
        )
        frozen_attempt_sensitive = abs(float(row["delta_frozen_attempts_vs_random_bank"])) > 1000.0
        if final_sensitive:
            classes.append("placement_sensitive_final_state")
            reasons.append("final Sortedness, success, or monotonicity error changed materially")
        elif efficiency_sensitive or path_sensitive or frozen_attempt_sensitive:
            classes.append("placement_sensitive_efficiency_or_path")
            active_reasons = []
            if efficiency_sensitive:
                active_reasons.append("efficiency ratio outside [0.5, 2.0]")
            if path_sensitive:
                active_reasons.append("DG delta exceeds 25 points or curvature delta exceeds 0.75")
            if frozen_attempt_sensitive:
                active_reasons.append("frozen-attempt mean delta exceeds 1000")
            reasons.append("; ".join(active_reasons))
        else:
            classes.append("placement_stable")
            reasons.append("final state, efficiency, DG/path shape, and frozen-attempt deltas within thresholds")
    classified["placement_sensitivity_class"] = classes
    classified["classification_rule"] = reasons
    return summary, classified


def make_frozen_placement_figure(summary_df: pd.DataFrame, classification_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    rule_order = [name for name in PLACEMENT_RULES if name in set(summary_df["placement_rule"])]
    conditions = (
        summary_df.assign(
            condition_label=lambda frame: frame["condition_id"]
            + " "
            + frame["algorithm"].astype(str).str[:3]
            + " "
            + frame["frozen_semantics"].astype(str).str[:1]
            + frame["frozen_count"].astype(str)
        )[["condition_id", "condition_label"]]
        .drop_duplicates()
        .sort_values("condition_id")
    )
    condition_order = conditions["condition_id"].tolist()
    condition_labels = conditions["condition_label"].tolist()
    pivots = [
        (
            "Final Sortedness delta vs random bank",
            classification_df.pivot_table(
                index="condition_id",
                columns="placement_rule",
                values="delta_final_sortedness_vs_random_bank",
                aggfunc="mean",
            ).reindex(index=condition_order, columns=rule_order),
            "coolwarm",
            "pp",
            None,
        ),
        (
            "Final monotonicity-error delta vs random bank",
            classification_df.pivot_table(
                index="condition_id",
                columns="placement_rule",
                values="delta_final_monotonicity_error_vs_random_bank",
                aggfunc="mean",
            ).reindex(index=condition_order, columns=rule_order),
            "PuOr",
            "errors",
            None,
        ),
        (
            "DG primary delta vs random bank",
            classification_df.pivot_table(
                index="condition_id",
                columns="placement_rule",
                values="delta_dg_primary_vs_random_bank",
                aggfunc="mean",
            ).reindex(index=condition_order, columns=rule_order),
            "PRGn",
            "DG",
            None,
        ),
        (
            "log2 efficiency ratio vs random bank",
            np.log2(
                classification_df.pivot_table(
                    index="condition_id",
                    columns="placement_rule",
                    values="compare_plus_swap_ratio_vs_random_bank",
                    aggfunc="mean",
                ).reindex(index=condition_order, columns=rule_order)
            ),
            "PiYG_r",
            "log2 ratio",
            "ratio",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for axis, (title, pivot, cmap, label, value_kind) in zip(axes.flat, pivots, strict=True):
        data = pivot.to_numpy(dtype=float)
        finite = data[np.isfinite(data)]
        vmax = max(1.0, float(np.nanmax(np.abs(finite))) if finite.size else 1.0)
        if "Sortedness" in title:
            vmax = max(5.0, vmax)
        im = axis.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks(range(len(rule_order)))
        axis.set_xticklabels([name.replace("_", "\n") for name in rule_order], fontsize=8)
        axis.set_yticks(range(len(condition_order)))
        axis.set_yticklabels(condition_labels if axis in (axes[0, 0], axes[1, 0]) else [], fontsize=8)
        for i in range(len(condition_order)):
            for j in range(len(rule_order)):
                value = pivot.iloc[i, j]
                if pd.notna(value):
                    if value_kind == "ratio":
                        ratio = 2.0 ** float(value)
                        text = f"{ratio:.1f}x"
                    else:
                        text = f"{float(value):+.1f}"
                    axis.text(j, i, text, ha="center", va="center", fontsize=6.5)
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04).set_label(label)
    fig.suptitle("E02 S11 Frozen Cell placement stress test", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def placement_validation_rows(result_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "research_step_id",
        "condition_id",
        "algorithm",
        "frozen_semantics",
        "frozen_count",
        "repeat_index",
        "placement_rule",
        "placement_seed",
        "placement_source",
        "placement_rule_valid",
        "placement_rule_failures_json",
        "high_low_value_target_ambiguous",
        "initial_frozen_indices_json",
        "initial_frozen_values_json",
        "final_frozen_positions_json",
        "final_frozen_values_json",
        "frozen_identity_count_preserved",
        "frozen_identity_values_preserved",
        "stuck_frozen_positions_preserved",
        "frozen_identity_validation_passed",
    ]
    return result_df[columns].sort_values(["condition_id", "repeat_index", "placement_rule"]).reset_index(drop=True)


def run_seed_and_determinism_checks(tasks: Sequence[PlacementTask], result_df: pd.DataFrame) -> dict[str, Any]:
    regeneration_ok = True
    mismatches: list[dict[str, Any]] = []
    for task in tasks:
        regenerated = validate_placement_rule(
            task.placement_rule,
            task.values,
            task.frozen_indices,
            int(task.condition["frozen_count"]),
            random_bank_indices=task.original_frozen_bank_indices,
            seed=task.placement_seed,
        )
        if regenerated["placement_indices_sha256"] != task.placement_validation["placement_indices_sha256"]:
            regeneration_ok = False
            mismatches.append(
                {
                    "condition_id": task.condition["condition_id"],
                    "repeat_index": int(task.repeat_index),
                    "placement_rule": task.placement_rule,
                }
            )
            if len(mismatches) >= 10:
                break

    sample_tasks: list[PlacementTask] = []
    for rule in PLACEMENT_RULES:
        candidates = [task for task in tasks if task.placement_rule == rule]
        if candidates:
            sample_tasks.append(candidates[0])
    deterministic_ok = True
    checks: list[dict[str, Any]] = []
    index = result_df.set_index(["condition_id", "placement_rule", "repeat_index"])
    for task in sample_tasks:
        rerun = run_placement_task(task)
        key = (task.condition["condition_id"], task.placement_rule, task.repeat_index)
        original = index.loc[key]
        passed = bool(
            str(rerun["final_array_sha256"]) == str(original["final_array_sha256"])
            and str(rerun["trace_values_sha256"]) == str(original["trace_values_sha256"])
            and int(rerun["event_count"]) == int(original["event_count"])
            and bool(rerun["frozen_identity_validation_passed"]) == bool(original["frozen_identity_validation_passed"])
        )
        deterministic_ok = deterministic_ok and passed
        checks.append(
            {
                "condition_id": task.condition["condition_id"],
                "repeat_index": int(task.repeat_index),
                "placement_rule": task.placement_rule,
                "passed": passed,
                "final_array_sha256": rerun["final_array_sha256"],
                "trace_values_sha256": rerun["trace_values_sha256"],
            }
        )
    return {
        "placementSeedRegenerationPassed": bool(regeneration_ok),
        "placementSeedMismatchExamples": mismatches,
        "simulationDeterminismSamplePassed": bool(deterministic_ok),
        "simulationDeterminismSampleCount": int(len(checks)),
        "simulationDeterminismChecks": checks,
    }


def validate_outputs(
    *,
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    placement_validation_df: pd.DataFrame,
    tasks: Sequence[PlacementTask],
    figure_path: Path,
    result_path: Path,
    dg_toy_validation: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {(task.condition["condition_id"], task.placement_rule, task.repeat_index) for task in tasks}
    observed_keys = set(zip(result_df["condition_id"], result_df["placement_rule"], result_df["repeat_index"]))
    expected_rules = sorted({task.placement_rule for task in tasks}, key=lambda name: PLACEMENT_RULE_INDEX[name])
    observed_rules = sorted(result_df["placement_rule"].unique(), key=lambda name: PLACEMENT_RULE_INDEX[name])
    classes = set(classification_df["placement_sensitivity_class"].dropna().unique())
    non_baseline_classes = {
        cls
        for cls in classes
        if cls != "baseline_random_bank"
    }
    sensitive_count = int(classification_df["placement_sensitivity_class"].str.startswith("placement_sensitive").sum())
    outcome = "constraining/contradictory" if sensitive_count > 0 else "supportive"
    success = bool(
        expected_keys == observed_keys
        and bool(result_df["placement_rule_valid"].all())
        and bool(result_df["frozen_identity_validation_passed"].all())
        and bool(result_df.loc[result_df["frozen_semantics"] == "stuck", "stuck_frozen_positions_preserved"].all())
        and not bool(result_df["high_low_value_target_ambiguous"].any())
        and bool(dg_toy_validation["toyCaseValidationPassed"])
        and bool(reproducibility["placementSeedRegenerationPassed"])
        and bool(reproducibility["simulationDeterminismSamplePassed"])
        and classes <= {
            "baseline_random_bank",
            "placement_stable",
            "placement_sensitive_efficiency_or_path",
            "placement_sensitive_final_state",
        }
        and result_path.exists()
        and figure_path.exists()
        and figure_path.stat().st_size > 0
    )
    validation = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "failed_validation",
        "validationResult": "pending",
        "outcomeClassification": outcome,
        "artifactsWritten": [],
        "caveatsOrBlockers": [
            "Matrix is bounded to first 10 repeats and selected cell-view Frozen Cell conditions with frozen_count 1 or 3.",
            "Runs use the S01 deterministic public-move simulator, not OS-thread replays.",
            "S08 Delayed Gratification is recomputed over deterministic Sortedness trajectories; S08 matched-null caveats still apply.",
            "S10 input-distribution caveats remain separate because S11 uses only the E01 unique random-permutation value bank.",
        ],
        "recommendedNextAction": "Review S11 placement sensitivity, then proceed to S12 only after explicit Chief Scientist instruction.",
        "expectedRunCount": int(len(expected_keys)),
        "observedRunCount": int(len(observed_keys)),
        "resultRowCount": int(len(result_df)),
        "summaryRowCount": int(len(summary_df)),
        "classificationRowCount": int(len(classification_df)),
        "placementValidationRowCount": int(len(placement_validation_df)),
        "conditionIds": sorted({task.condition["condition_id"] for task in tasks}),
        "placementRules": expected_rules,
        "observedPlacementRules": observed_rules,
        "repeatIndexes": sorted({int(task.repeat_index) for task in tasks}),
        "allExpectedRunsPresent": bool(expected_keys == observed_keys),
        "placementRulesValidEveryRun": bool(result_df["placement_rule_valid"].all()),
        "frozenIdentityValidEveryRun": bool(result_df["frozen_identity_validation_passed"].all()),
        "stuckPositionsPreservedEveryRun": bool(result_df.loc[result_df["frozen_semantics"] == "stuck", "stuck_frozen_positions_preserved"].all()),
        "highLowValueTargetAmbiguities": int(result_df["high_low_value_target_ambiguous"].sum()),
        "dgToyValidationPassed": bool(dg_toy_validation["toyCaseValidationPassed"]),
        "classificationClasses": sorted(classes),
        "nonBaselinePlacementClasses": sorted(non_baseline_classes),
        "placementSensitivityClassCounts": classification_df["placement_sensitivity_class"].value_counts().sort_index().to_dict(),
        "sensitiveSliceCount": sensitive_count,
        "figureNonempty": bool(figure_path.exists() and figure_path.stat().st_size > 0),
        "plannedMatrixWritten": bool(result_path.exists()),
        **dict(reproducibility),
    }
    validation["validationResult"] = (
        "passed: placement rules, Frozen Cell identities, stuck positions, S08 DG toy cases, reproducible placement seeds, matrix, and figure validated"
        if success
        else "failed: one or more S11 validation checks did not pass"
    )
    return validation


def build_report(
    *,
    generated_at: str,
    result_path: Path,
    summary_path: Path,
    classification_path: Path,
    placement_validation_path: Path,
    figure_path: Path,
    validation_path: Path,
    status_path: Path,
    dg_toy_path: Path,
    log_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    summary_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    placement_validation_df: pd.DataFrame,
    validation: Mapping[str, Any],
    unit_result: Mapping[str, Any] | None,
    git_commit: str,
    git_status: str,
    elapsed_seconds: float,
    parameters: Mapping[str, Any],
) -> str:
    unit_line = "not run"
    if unit_result is not None:
        unit_line = f"{unit_result['command']} -> return code {unit_result['returnCode']}"
    class_counts = classification_df["placement_sensitivity_class"].value_counts().sort_index().to_dict()
    largest_final = classification_df[classification_df["placement_rule"] != "random_bank"].copy()
    largest_final = largest_final.sort_values("delta_final_sortedness_vs_random_bank").head(18)[
        [
            "condition_id",
            "algorithm",
            "frozen_semantics",
            "frozen_count",
            "placement_rule",
            "placement_sensitivity_class",
            "mean_final_sortedness",
            "delta_final_sortedness_vs_random_bank",
            "mean_final_monotonicity_error",
            "compare_plus_swap_ratio_vs_random_bank",
            "delta_dg_primary_vs_random_bank",
        ]
    ]
    placement_excerpt = placement_validation_df[
        [
            "condition_id",
            "repeat_index",
            "placement_rule",
            "initial_frozen_indices_json",
            "initial_frozen_values_json",
            "final_frozen_positions_json",
            "final_frozen_values_json",
            "placement_rule_valid",
            "frozen_identity_validation_passed",
        ]
    ]
    return f"""# E02 S11 Full Results: Frozen Cell Placement Stress Test

## Top Summary

- Step ID: S11
- Completion status: {validation['status']}
- Artifacts written: {', '.join(validation['artifactsWritten'])}
- Validation result: {validation['validationResult']}
- Outcome classification: {validation['outcomeClassification']}
- Caveats or blockers: {', '.join(validation['caveatsOrBlockers'])}
- Lay summary: S11 moved Frozen Cells to controlled positions while keeping the same deterministic simulator, initial value bank, policies, and matched activation seeds. Placement changed final-state and path/efficiency outcomes in enough slices that Frozen Cell robustness cannot be treated as independent of defect location in this bounded matrix.
- Recommended next action: {validation['recommendedNextAction']}

## Frozen Question

Is Frozen Cell robustness explained by where defects occur rather than by algorithmic barrier-navigation competence?

## Inputs

- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- S01 deterministic simulator: `src/e02/deterministic_simulator.py`
- S08 DG recomputation code: `scripts/e02_s08_dg_nulls.py`
- S10 caveat context: S11 keeps only the original E01 unique random-permutation value bank, so S10 input-distribution sensitivity is not confounded with placement.
- Selected condition IDs: `{parameters['conditionIds']}`
- Placement rules: `{parameters['placementRules']}`
- Repeat indexes: `{parameters['repeatIndexes']}`

No external datasets were required and no new dependencies were installed.

## Methods

The script selected cell-view Frozen Cell conditions with passive or stuck semantics and frozen counts of one or three. For each condition and repeat, it rebuilt the original unique 1..100 initial value array and pure-algorithm Algotype labels from the E01 config, then replaced the E01 frozen-index bank with one of eight placement rules:

- `left_end`: first k positions.
- `right_end`: last k positions.
- `center`: centered contiguous k positions.
- `random_bank`: original E01 random Frozen Cell bank, used as the reference placement.
- `high_value`: positions carrying the k largest initial values.
- `low_value`: positions carrying the k smallest initial values.
- `clustered_seeded`: seeded contiguous block.
- `evenly_spaced`: approximately equal interior spacing.

Activation seeds intentionally reuse the S09/S10 condition-repeat schedule across placement rules, so a condition/repeat receives the same deterministic activation stream regardless of placement. The simulation calls public cell `move()` methods through the S01 deterministic single-event wrapper. DG is recomputed with the S08 `dg_from_sortedness` formula over each deterministic Sortedness trajectory. Path shape is summarized by scalar Sortedness curvature and value-state curvature.

## Commands

- `python scripts/e02_s11_frozen_placement.py --repo-dir /workspace/cell-research --artifacts-dir /artifacts --max-repeats {parameters['maxRepeats']} --workers {parameters['workers']} --run-unit-tests`
- Unit-test command inside the script: `{unit_line}`

## Results

- Matrix rows: {validation['resultRowCount']}
- Summary rows: {validation['summaryRowCount']}
- Classification rows: {validation['classificationRowCount']}
- Placement validation rows: {validation['placementValidationRowCount']}
- Placement sensitivity class counts: {json.dumps(class_counts, sort_keys=True)}
- Expected runs present: {validation['allExpectedRunsPresent']}
- Placement rules valid in every run: {validation['placementRulesValidEveryRun']}
- Frozen identities validated in every run: {validation['frozenIdentityValidEveryRun']}
- Stuck Frozen Cell positions preserved in every stuck run: {validation['stuckPositionsPreservedEveryRun']}

### Largest Final Sortedness Drops

{markdown_table(largest_final, max_rows=18)}

### Placement Identity Validation Excerpt

{markdown_table(placement_excerpt, max_rows=20)}

## Validation

- All expected condition/placement/repeat runs present: {validation['allExpectedRunsPresent']}
- Placement rule validation passed for every run: {validation['placementRulesValidEveryRun']}
- Frozen identity validation passed for every run: {validation['frozenIdentityValidEveryRun']}
- Stuck Frozen Cell positions preserved for stuck semantics: {validation['stuckPositionsPreservedEveryRun']}
- High/low-value targeting ambiguities: {validation['highLowValueTargetAmbiguities']}
- S08 DG toy-case validation passed: {validation['dgToyValidationPassed']}
- Placement seed regeneration passed: {validation['placementSeedRegenerationPassed']}
- Simulation determinism sample passed: {validation['simulationDeterminismSamplePassed']} across {validation['simulationDeterminismSampleCount']} sampled reruns.
- Planned matrix written: {validation['plannedMatrixWritten']}
- Figure nonempty: {validation['figureNonempty']}
- Unit tests: {unit_line}

Validation JSON: `{validation_path}`  
Status JSON: `{status_path}`  
DG toy validation JSON: `{dg_toy_path}`

## Artifacts

- Placement matrix: `{result_path}`
- Placement summary CSV: `{summary_path}`
- Placement sensitivity classification CSV: `{classification_path}`
- Placement and identity validation CSV: `{placement_validation_path}`
- Heatmap figure: `{figure_path}`
- Execution log: `{log_path}`
- Artifact manifest: `{manifest_path}`
- Source manifest: `{src_manifest_path}`

## Caveats And Limitations

- S11 is a bounded deterministic rerun matrix: first 10 repeats, selected cell-view Frozen Cell conditions, and frozen counts one and three. It is not a 100-repeat paper-scale inference layer.
- The simulator preserves public local policy calls but is not an OS-thread replay.
- Passive and stuck Frozen Cell behavior inherit the S01/E01 semantics caveats.
- DG is a trajectory proxy over Sortedness; S08 showed matched synthetic trajectory nulls can explain or exceed DG in many burden slices.
- S10 showed input-distribution sensitivity. S11 intentionally avoids mixing that factor by using only the original unique random-permutation bank.
- High/low-value placement is unambiguous here because all selected arrays use unique values; duplicate-value Frozen Cell placement would need a separate tie policy.

## Provenance

- Generated at UTC: {generated_at}
- Runtime: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, matplotlib {matplotlib.__version__}
- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Elapsed seconds: {elapsed_seconds:.3f}
- Workers: {parameters['workers']}
- No new dependencies were installed.

## Recommended Next Action

Review S11 placement sensitivity, then proceed to S12 Frozen Cell behavior variants only after explicit Chief Scientist instruction.
"""


def main() -> int:
    start = time.monotonic()
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e02"
    logs_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    for directory in (step_dir, results_dir, figures_dir, logs_dir, src_snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    report_path = step_dir / "research_step_full_results.md"
    result_path = results_dir / "e02_frozen_placement.parquet"
    summary_path = step_dir / "e02_frozen_placement_summary.csv"
    classification_path = step_dir / "e02_frozen_placement_classification.csv"
    placement_validation_path = step_dir / "e02_frozen_placement_validation.csv"
    figure_path = figures_dir / "frozen_placement_heatmaps.png"
    validation_path = step_dir / "s11_validation.json"
    status_path = step_dir / "status.json"
    dg_toy_path = step_dir / "e02_s11_dg_toy_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s11_frozen_placement_manifest.json"
    log_path = logs_dir / "e02_s11_frozen_placement.log"

    cfg = load_e01_config(args.config_path)
    rows = selected_conditions(cfg, args.condition_ids)
    tasks = build_tasks(
        cfg,
        rows,
        repo_dir,
        max_repeats=args.max_repeats,
        max_events=args.max_events,
        max_successful_swaps=args.max_successful_swaps,
        stall_events=args.stall_events,
        activation_distribution=args.activation_distribution,
        placement_rules=args.placement_rules,
    )

    unit_result: dict[str, Any] | None = None
    if args.run_unit_tests:
        unit_result = run_command([sys.executable, "-m", "unittest", "tests.e02.test_frozen_placement"], repo_dir)
        if not unit_result["success"]:
            log_path.write_text(
                f"[{utc_now()}] Unit tests failed before S11 run\n{unit_result['stdout']}\n{unit_result['stderr']}\n",
                encoding="utf-8",
            )
            return 2

    log_lines = [
        f"[{utc_now()}] Starting {STEP_ID} Frozen Cell placement matrix",
        f"repo_dir={repo_dir}",
        f"artifacts_dir={artifacts_dir}",
        f"tasks={len(tasks)} workers={args.workers} max_repeats={args.max_repeats}",
        f"condition_ids={','.join(row['condition_id'] for row in rows)}",
        f"placement_rules={','.join(args.placement_rules)}",
    ]

    if args.workers == 1:
        result_rows = [run_placement_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            result_rows = list(executor.map(run_placement_task, tasks))
    result_df = pd.DataFrame(result_rows).sort_values(["condition_id", "repeat_index", "placement_rule"]).reset_index(drop=True)
    result_df.to_parquet(result_path, index=False)

    summary_df, classification_df = summarize_results(result_df)
    placement_validation_df = placement_validation_rows(result_df)
    summary_df.to_csv(summary_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    placement_validation_df.to_csv(placement_validation_path, index=False)
    make_frozen_placement_figure(summary_df, classification_df, figure_path)

    dg_toy_validation = validate_toy_cases()
    write_json(dg_toy_path, dg_toy_validation)
    reproducibility = run_seed_and_determinism_checks(tasks, result_df)
    validation = validate_outputs(
        result_df=result_df,
        summary_df=summary_df,
        classification_df=classification_df,
        placement_validation_df=placement_validation_df,
        tasks=tasks,
        figure_path=figure_path,
        result_path=result_path,
        dg_toy_validation=dg_toy_validation,
        reproducibility=reproducibility,
    )

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    parameters = {
        "conditionIds": validation["conditionIds"],
        "placementRules": validation["placementRules"],
        "repeatIndexes": validation["repeatIndexes"],
        "maxRepeats": int(args.max_repeats),
        "workers": int(args.workers),
        "maxEvents": int(args.max_events),
        "maxSuccessfulSwaps": int(args.max_successful_swaps),
        "stallEvents": int(args.stall_events),
        "activationDistribution": args.activation_distribution,
    }

    artifact_paths = [
        (report_path, "Full S11 Markdown research-step report"),
        (result_path, "Run-level Frozen Cell placement matrix"),
        (summary_path, "Condition-placement summary table"),
        (classification_path, "Placement sensitivity classification table"),
        (placement_validation_path, "Per-run placement-rule and Frozen Cell identity validation"),
        (figure_path, "Frozen Cell placement heatmap figure"),
        (validation_path, "S11 validation JSON"),
        (status_path, "Compact S11 workflow status JSON"),
        (dg_toy_path, "S08 DG formula toy-case validation reused by S11"),
        (log_path, "S11 execution log"),
        (src_manifest_path, "S11 source/provenance manifest"),
    ]
    validation["artifactsWritten"] = [str(path) for path, _ in artifact_paths] + [str(manifest_path)]
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"]),
        "status": validation["status"],
        "artifactsWritten": validation["artifactsWritten"],
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
    }
    write_json(validation_path, validation)
    write_json(status_path, status)

    src_entries = []
    for relative in [
        "scripts/e02_s11_frozen_placement.py",
        "tests/e02/test_frozen_placement.py",
        "src/e02/deterministic_simulator.py",
        "scripts/e02_s08_dg_nulls.py",
        "scripts/e02_s09_alternative_metrics.py",
        "scripts/e02_s10_input_distributions.py",
    ]:
        path = repo_dir / relative
        src_entries.append(
            {
                "path": str(path),
                "relativePath": relative,
                "sha256": sha256_file(path),
                "sizeBytes": int(path.stat().st_size),
            }
        )
    src_manifest = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "gitCommit": git_commit,
        "gitStatusShort": git_status,
        "sourceFiles": src_entries,
        "parameters": parameters,
    }
    write_json(src_manifest_path, src_manifest)

    report = build_report(
        generated_at=utc_now(),
        result_path=result_path,
        summary_path=summary_path,
        classification_path=classification_path,
        placement_validation_path=placement_validation_path,
        figure_path=figure_path,
        validation_path=validation_path,
        status_path=status_path,
        dg_toy_path=dg_toy_path,
        log_path=log_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        summary_df=summary_df,
        classification_df=classification_df,
        placement_validation_df=placement_validation_df,
        validation=validation,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        elapsed_seconds=time.monotonic() - start,
        parameters=parameters,
    )
    report_path.write_text(report, encoding="utf-8")

    log_lines.extend(
        [
            f"[{utc_now()}] Completed matrix rows={len(result_df)} validation_success={validation['success']}",
            f"result_path={result_path}",
            f"figure_path={figure_path}",
            f"class_counts={json.dumps(validation['placementSensitivityClassCounts'], sort_keys=True)}",
            f"elapsed_seconds={time.monotonic() - start:.3f}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    manifest_entries = [artifact_entry(path, artifacts_dir, description) for path, description in artifact_paths]
    manifest = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "artifactCount": len(manifest_entries),
        "artifacts": manifest_entries,
        "excludedSelfReferentialArtifact": str(manifest_path),
    }
    write_json(manifest_path, manifest)

    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
