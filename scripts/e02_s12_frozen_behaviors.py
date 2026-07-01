#!/usr/bin/env python3
"""Run E02 S12 Frozen Cell behavior-variant stress tests."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
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


STEP_ID = "S12"
STEP_NUMBER = 12
EXPERIMENT_ID = "E02"
SOURCE_EXPERIMENT_ID = "E01"

TEMPLATE_PASSIVE_CONDITION_IDS = (
    "E01C010",
    "E01C011",
    "E01C012",
    "E01C022",
    "E01C023",
    "E01C024",
)

BEHAVIOR_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "name": "original_passive",
        "family": "original_limit",
        "frozen_semantics": "passive",
        "behavior": None,
        "description": "original passive Frozen Cell semantics from the S01 simulator",
    },
    {
        "name": "original_stuck",
        "family": "original_limit",
        "frozen_semantics": "stuck",
        "behavior": None,
        "description": "original stuck Frozen Cell semantics from the S01 simulator",
    },
    {
        "name": "dynamic_passive_limit",
        "family": "dynamic_limit",
        "frozen_semantics": "dynamic",
        "behavior": {"type": "passive_limit"},
        "description": "dynamic controller configured to exactly recover passive target mobility",
    },
    {
        "name": "dynamic_stuck_limit",
        "family": "dynamic_limit",
        "frozen_semantics": "dynamic",
        "behavior": {"type": "stuck_limit"},
        "description": "dynamic controller configured to exactly recover stuck target blocking",
    },
    {
        "name": "probabilistic_stuck_p25",
        "family": "probabilistic",
        "frozen_semantics": "dynamic",
        "behavior": {"type": "probabilistic_stuck", "block_probability": 0.25},
        "description": "Frozen targets block 25 percent of contact attempts using a seeded policy RNG",
    },
    {
        "name": "probabilistic_stuck_p75",
        "family": "probabilistic",
        "frozen_semantics": "dynamic",
        "behavior": {"type": "probabilistic_stuck", "block_probability": 0.75},
        "description": "Frozen targets block 75 percent of contact attempts using a seeded policy RNG",
    },
    {
        "name": "time_varying_periodic",
        "family": "time_varying",
        "frozen_semantics": "dynamic",
        "behavior": {"type": "time_varying", "window_events": 250},
        "description": "Frozen targets alternate between stuck and passive phases every 250 events",
    },
    {
        "name": "fatigue_recovery",
        "family": "fatigue_recovery",
        "frozen_semantics": "dynamic",
        "behavior": {"type": "fatigue_recovery", "fatigue_threshold": 4, "recovery_events": 250},
        "description": "Frozen targets become temporarily stuck after repeated nudges, then recover",
    },
    {
        "name": "directional_sticky_left",
        "family": "directional_sticky",
        "frozen_semantics": "dynamic",
        "behavior": {"type": "directional_sticky", "blocked_from": "left"},
        "description": "Frozen targets resist swaps initiated from their left side only",
    },
    {
        "name": "directional_sticky_right",
        "family": "directional_sticky",
        "frozen_semantics": "dynamic",
        "behavior": {"type": "directional_sticky", "blocked_from": "right"},
        "description": "Frozen targets resist swaps initiated from their right side only",
    },
)
BEHAVIOR_VARIANT_INDEX = {row["name"]: idx + 1 for idx, row in enumerate(BEHAVIOR_VARIANTS)}


@dataclass(frozen=True)
class ConditionPair:
    pair_id: str
    passive_condition: dict[str, Any]
    stuck_condition: dict[str, Any]


@dataclass(frozen=True)
class BehaviorTask:
    pair: ConditionPair
    repeat_index: int
    behavior_variant: str
    behavior_family: str
    behavior_description: str
    behavior_spec: dict[str, Any] | None
    frozen_semantics: str
    values: tuple[int, ...]
    assignments: tuple[str, ...]
    reverse_directions: tuple[bool, ...]
    frozen_indices: tuple[int, ...]
    initial_frozen_values: tuple[int, ...]
    initial_array_seed: int
    frozen_bank_seed: int | None
    activation_seed: int
    policy_seed: int
    behavior_seed: int | None
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
    parser.add_argument("--template-condition-ids", nargs="*", default=list(TEMPLATE_PASSIVE_CONDITION_IDS))
    parser.add_argument("--behavior-variants", nargs="*", default=[row["name"] for row in BEHAVIOR_VARIANTS])
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


def base_activation_seed(base_seed: int, pair_id: str, repeat_idx: int) -> int:
    digest = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()
    pair_component = int(digest[:8], 16) % 90_000
    return base_seed * 1_000_000 + 1_200_000 + pair_component * 1_000 + repeat_idx


def behavior_seed(base_seed: int, pair_id: str, variant: str, repeat_idx: int) -> int:
    digest = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()
    pair_component = int(digest[:8], 16) % 90_000
    return base_seed * 1_000_000 + 1_300_000 + pair_component * 10_000 + BEHAVIOR_VARIANT_INDEX[variant] * 100 + repeat_idx


def selected_condition_pairs(cfg: Mapping[str, Any], template_condition_ids: Sequence[str]) -> list[ConditionPair]:
    requested = tuple(dict.fromkeys(template_condition_ids))
    by_id = {row["condition_id"]: dict(row) for row in cfg["conditions"]}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise RuntimeError(f"Missing expected S12 template condition IDs in E01 config: {missing}")
    candidate_rows = [
        dict(row)
        for row in cfg["conditions"]
        if row["mode"] == "cell_view"
        and row.get("direction_profile") == "all_increasing"
        and row.get("value_bank_id") == "unique_1_to_100"
        and int(row.get("frozen_count", 0)) in {1, 3}
        and row.get("frozen_semantics") in {"passive", "stuck"}
    ]
    lookup = {
        (row["algorithm"], int(row["frozen_count"]), row["frozen_semantics"]): row
        for row in candidate_rows
    }
    pairs: list[ConditionPair] = []
    for condition_id in requested:
        passive = dict(by_id[condition_id])
        if passive.get("frozen_semantics") != "passive":
            raise RuntimeError(f"S12 template {condition_id} is not a passive Frozen Cell condition.")
        key = (passive["algorithm"], int(passive["frozen_count"]), "stuck")
        stuck = lookup.get(key)
        if stuck is None:
            raise RuntimeError(f"Missing matched stuck condition for {condition_id}: {key}")
        if stuck["frozen_index_bank_id"] != passive["frozen_index_bank_id"]:
            raise RuntimeError(f"Matched passive/stuck conditions do not share frozen-index bank for {condition_id}.")
        pair_id = f"{passive['algorithm']}_f{int(passive['frozen_count'])}"
        pairs.append(ConditionPair(pair_id=pair_id, passive_condition=passive, stuck_condition=dict(stuck)))
    return sorted(pairs, key=lambda pair: (int(pair.passive_condition["frozen_count"]), pair.passive_condition["algorithm"]))


def selected_behavior_variants(variant_names: Sequence[str]) -> list[dict[str, Any]]:
    available = {row["name"]: row for row in BEHAVIOR_VARIANTS}
    unknown = sorted(set(variant_names) - set(available))
    if unknown:
        raise RuntimeError(f"Unsupported S12 behavior variants requested: {unknown}")
    return [dict(available[name]) for name in dict.fromkeys(variant_names)]


def behavior_spec_for_variant(variant: Mapping[str, Any], seed: int | None) -> dict[str, Any] | None:
    spec = variant["behavior"]
    if spec is None:
        return None
    resolved = dict(spec)
    resolved["label"] = variant["name"]
    if seed is not None:
        resolved["seed"] = int(seed)
    return resolved


def build_tasks(
    cfg: Mapping[str, Any],
    pairs: Sequence[ConditionPair],
    variants: Sequence[Mapping[str, Any]],
    repo_dir: Path,
    max_repeats: int,
    max_events: int,
    max_successful_swaps: int,
    stall_events: int,
    activation_distribution: str,
) -> list[BehaviorTask]:
    repeat_count = min(max_repeats, int(cfg["globalDefaults"]["repeatCount"]))
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    value_banks = cfg["seedBanks"]["valueBanks"]
    frozen_banks = cfg["seedBanks"]["frozenIndexBanks"]
    tasks: list[BehaviorTask] = []
    for pair in pairs:
        condition = pair.passive_condition
        value_bank = value_banks[condition["value_bank_id"]]
        frozen_bank = frozen_banks[condition["frozen_index_bank_id"]]
        for repeat_idx in range(repeat_count):
            values = tuple(map(int, value_bank["initialArrays"][repeat_idx]))
            assignments = tuple(str(condition["algorithm"]) for _ in values)
            frozen_indices = tuple(map(int, frozen_bank["indices"][repeat_idx]))
            frozen_seed = int(frozen_bank["seeds"][repeat_idx]) if frozen_bank["seeds"] else None
            initial_frozen_values = tuple(int(values[idx]) for idx in frozen_indices)
            act_seed = base_activation_seed(base_seed, pair.pair_id, repeat_idx)
            for variant in variants:
                variant_name = str(variant["name"])
                dyn_seed = (
                    behavior_seed(base_seed, pair.pair_id, variant_name, repeat_idx)
                    if variant["frozen_semantics"] == "dynamic"
                    else None
                )
                tasks.append(
                    BehaviorTask(
                        pair=pair,
                        repeat_index=int(repeat_idx),
                        behavior_variant=variant_name,
                        behavior_family=str(variant["family"]),
                        behavior_description=str(variant["description"]),
                        behavior_spec=behavior_spec_for_variant(variant, dyn_seed),
                        frozen_semantics=str(variant["frozen_semantics"]),
                        values=values,
                        assignments=assignments,
                        reverse_directions=reverse_flags_for_condition(condition, assignments),
                        frozen_indices=frozen_indices,
                        initial_frozen_values=initial_frozen_values,
                        initial_array_seed=int(value_bank["seeds"][repeat_idx]),
                        frozen_bank_seed=frozen_seed,
                        activation_seed=act_seed,
                        policy_seed=act_seed,
                        behavior_seed=dyn_seed,
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


def behavior_attempt_keys(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary.get("attempt_counts", {}), sort_keys=True, separators=(",", ":"))


def run_behavior_task(task: BehaviorTask) -> dict[str, Any]:
    started = time.monotonic()
    config = SimulatorConfig(
        values=task.values,
        algotypes=task.assignments,
        reverse_directions=task.reverse_directions,
        frozen_indices=task.frozen_indices,
        frozen_semantics=task.frozen_semantics,
        frozen_behavior=task.behavior_spec,
        activation_seed=task.activation_seed,
        policy_seed=task.policy_seed,
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
    stuck_limit_positions_preserved = (
        tuple(sorted(result.final_frozen_positions)) == tuple(sorted(task.frozen_indices))
        if task.behavior_variant in {"original_stuck", "dynamic_stuck_limit"}
        else True
    )
    frozen_identity_validation_passed = bool(
        frozen_identity_count_preserved
        and frozen_identity_values_preserved
        and stuck_limit_positions_preserved
    )
    behavior_summary = dict(result.frozen_behavior_summary)
    transition_log = list(result.frozen_behavior_transition_log)
    attempt_log = list(result.frozen_behavior_attempt_log)
    initial_frozen_value_mean = float(np.mean(task.initial_frozen_values)) if task.initial_frozen_values else np.nan
    initial_frozen_position_mean = float(np.mean(task.frozen_indices)) if task.frozen_indices else np.nan

    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "evidence_layer": "s12_frozen_behavior_deterministic_rerun",
        "condition_pair_id": task.pair.pair_id,
        "passive_condition_id": task.pair.passive_condition["condition_id"],
        "stuck_condition_id": task.pair.stuck_condition["condition_id"],
        "run_family": task.pair.passive_condition["run_family"],
        "mode": "cell_view",
        "algorithm": task.pair.passive_condition["algorithm"],
        "algotype_mix": task.pair.passive_condition["algotype_mix"],
        "value_bank_id": task.pair.passive_condition["value_bank_id"],
        "value_distribution": task.pair.passive_condition.get("value_distribution", task.pair.passive_condition["value_bank_id"]),
        "direction_profile": task.pair.passive_condition.get("direction_profile", "all_increasing"),
        "sort_direction": task.sort_direction,
        "scheduler_regime": f"deterministic_single_event_{task.activation_distribution}",
        "scheduler_source": "s01_deterministic_simulator_public_move_methods",
        "repeat_index": int(task.repeat_index),
        "array_length": int(len(task.values)),
        "initial_array_seed": int(task.initial_array_seed),
        "frozen_bank_seed": task.frozen_bank_seed,
        "activation_seed": int(task.activation_seed),
        "policy_seed": int(task.policy_seed),
        "behavior_seed": task.behavior_seed,
        "behavior_variant": task.behavior_variant,
        "behavior_family": task.behavior_family,
        "behavior_description": task.behavior_description,
        "behavior_spec_json": json.dumps(task.behavior_spec, sort_keys=True, separators=(",", ":")) if task.behavior_spec is not None else None,
        "frozen_semantics": task.frozen_semantics,
        "frozen_count": int(task.pair.passive_condition["frozen_count"]),
        "placement_rule": "random_bank",
        "placement_source": "E01 frozenIndexBanks reused from S11 random_bank placement",
        "initial_frozen_indices_json": json.dumps(list(task.frozen_indices), separators=(",", ":")),
        "initial_frozen_values_json": json.dumps(list(task.initial_frozen_values), separators=(",", ":")),
        "initial_frozen_values_sorted_json": json.dumps(list(sorted(task.initial_frozen_values)), separators=(",", ":")),
        "final_frozen_positions_json": json.dumps(list(result.final_frozen_positions), separators=(",", ":")),
        "final_frozen_values_json": json.dumps(list(result.final_frozen_values), separators=(",", ":")),
        "final_frozen_values_sorted_json": json.dumps(list(sorted(result.final_frozen_values)), separators=(",", ":")),
        "initial_frozen_position_mean": finite_or_none(initial_frozen_position_mean),
        "initial_frozen_value_mean": finite_or_none(initial_frozen_value_mean),
        "frozen_identity_count_preserved": bool(frozen_identity_count_preserved),
        "frozen_identity_values_preserved": bool(frozen_identity_values_preserved),
        "stuck_limit_positions_preserved": bool(stuck_limit_positions_preserved),
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
        "behavior_transition_count": int(behavior_summary.get("transition_count", 0)),
        "behavior_transition_log_truncated": bool(behavior_summary.get("transition_log_truncated", False)),
        "behavior_transition_log_json": json.dumps(transition_log, sort_keys=True, separators=(",", ":")),
        "behavior_transition_log_sha256": stable_json_sha256(transition_log),
        "behavior_attempt_count": int(behavior_summary.get("attempt_count", 0)),
        "behavior_allowed_attempt_count": int(behavior_summary.get("allowed_attempt_count", 0)),
        "behavior_blocked_attempt_count": int(behavior_summary.get("blocked_attempt_count", 0)),
        "behavior_attempt_log_truncated": bool(behavior_summary.get("attempt_log_truncated", False)),
        "behavior_attempt_counts_json": behavior_attempt_keys(behavior_summary),
        "behavior_attempt_log_json": json.dumps(attempt_log, sort_keys=True, separators=(",", ":")),
        "behavior_attempt_log_sha256": stable_json_sha256(attempt_log),
        "behavior_final_state_by_thread_json": json.dumps(behavior_summary.get("final_state_by_thread", {}), sort_keys=True, separators=(",", ":")),
        "behavior_summary_json": json.dumps(behavior_summary, sort_keys=True, separators=(",", ":"), default=str),
        **dg,
    }


def summarize_results(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = result_df.groupby(
        [
            "condition_pair_id",
            "passive_condition_id",
            "stuck_condition_id",
            "run_family",
            "algorithm",
            "algotype_mix",
            "frozen_count",
            "behavior_variant",
            "behavior_family",
            "frozen_semantics",
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
        mean_behavior_transitions=("behavior_transition_count", "mean"),
        mean_behavior_attempts=("behavior_attempt_count", "mean"),
        mean_behavior_allowed_attempts=("behavior_allowed_attempt_count", "mean"),
        mean_behavior_blocked_attempts=("behavior_blocked_attempt_count", "mean"),
        max_guard_hit_rate=("max_guard_hit", "mean"),
        frozen_identity_valid_rate=("frozen_identity_validation_passed", "mean"),
    ).reset_index()
    summary["sd_final_sortedness"] = summary["sd_final_sortedness"].fillna(0.0)

    baseline_cols = [
        "condition_pair_id",
        "behavior_variant",
        "success_rate",
        "mean_final_sortedness",
        "mean_final_monotonicity_error",
        "mean_compare_plus_swap_steps",
        "mean_dg_primary",
        "mean_sortedness_curvature",
        "mean_frozen_attempt_count",
    ]
    baseline = summary[summary["behavior_variant"].isin(["original_passive", "original_stuck"])][baseline_cols].copy()
    passive = baseline[baseline["behavior_variant"] == "original_passive"].drop(columns=["behavior_variant"]).rename(
        columns={
            "success_rate": "passive_success_rate",
            "mean_final_sortedness": "passive_mean_final_sortedness",
            "mean_final_monotonicity_error": "passive_mean_final_monotonicity_error",
            "mean_compare_plus_swap_steps": "passive_mean_compare_plus_swap_steps",
            "mean_dg_primary": "passive_mean_dg_primary",
            "mean_sortedness_curvature": "passive_mean_sortedness_curvature",
            "mean_frozen_attempt_count": "passive_mean_frozen_attempt_count",
        }
    )
    stuck = baseline[baseline["behavior_variant"] == "original_stuck"].drop(columns=["behavior_variant"]).rename(
        columns={
            "success_rate": "stuck_success_rate",
            "mean_final_sortedness": "stuck_mean_final_sortedness",
            "mean_final_monotonicity_error": "stuck_mean_final_monotonicity_error",
            "mean_compare_plus_swap_steps": "stuck_mean_compare_plus_swap_steps",
            "mean_dg_primary": "stuck_mean_dg_primary",
            "mean_sortedness_curvature": "stuck_mean_sortedness_curvature",
            "mean_frozen_attempt_count": "stuck_mean_frozen_attempt_count",
        }
    )
    classified = summary.merge(passive, on="condition_pair_id", how="left").merge(stuck, on="condition_pair_id", how="left")
    classified["delta_final_sortedness_vs_passive"] = classified["mean_final_sortedness"] - classified["passive_mean_final_sortedness"]
    classified["delta_final_sortedness_vs_stuck"] = classified["mean_final_sortedness"] - classified["stuck_mean_final_sortedness"]
    classified["delta_success_rate_vs_passive"] = classified["success_rate"] - classified["passive_success_rate"]
    classified["delta_success_rate_vs_stuck"] = classified["success_rate"] - classified["stuck_success_rate"]
    classified["delta_final_monotonicity_error_vs_passive"] = (
        classified["mean_final_monotonicity_error"] - classified["passive_mean_final_monotonicity_error"]
    )
    classified["delta_final_monotonicity_error_vs_stuck"] = (
        classified["mean_final_monotonicity_error"] - classified["stuck_mean_final_monotonicity_error"]
    )
    classified["compare_plus_swap_ratio_vs_passive"] = (
        classified["mean_compare_plus_swap_steps"] / classified["passive_mean_compare_plus_swap_steps"].replace(0, np.nan)
    )
    classified["compare_plus_swap_ratio_vs_stuck"] = (
        classified["mean_compare_plus_swap_steps"] / classified["stuck_mean_compare_plus_swap_steps"].replace(0, np.nan)
    )
    classified["delta_dg_primary_vs_passive"] = classified["mean_dg_primary"] - classified["passive_mean_dg_primary"]
    classified["delta_dg_primary_vs_stuck"] = classified["mean_dg_primary"] - classified["stuck_mean_dg_primary"]
    classified["delta_sortedness_curvature_vs_passive"] = (
        classified["mean_sortedness_curvature"] - classified["passive_mean_sortedness_curvature"]
    )
    classified["delta_sortedness_curvature_vs_stuck"] = (
        classified["mean_sortedness_curvature"] - classified["stuck_mean_sortedness_curvature"]
    )
    classified["delta_frozen_attempts_vs_passive"] = classified["mean_frozen_attempt_count"] - classified["passive_mean_frozen_attempt_count"]
    classified["delta_frozen_attempts_vs_stuck"] = classified["mean_frozen_attempt_count"] - classified["stuck_mean_frozen_attempt_count"]

    classes: list[str] = []
    reasons: list[str] = []
    for row in classified.to_dict(orient="records"):
        variant = row["behavior_variant"]
        if variant in {"original_passive", "original_stuck"}:
            classes.append("original_baseline")
            reasons.append("reference original S01 passive/stuck semantics")
            continue
        if variant in {"dynamic_passive_limit", "dynamic_stuck_limit"}:
            classes.append("dynamic_limit_case")
            reasons.append("dynamic controller limiting case; exact recovery checked in validation")
            continue
        min_final = min(float(row["passive_mean_final_sortedness"]), float(row["stuck_mean_final_sortedness"]))
        max_error = max(float(row["passive_mean_final_monotonicity_error"]), float(row["stuck_mean_final_monotonicity_error"]))
        min_success = min(float(row["passive_success_rate"]), float(row["stuck_success_rate"]))
        final_sensitive = (
            float(row["mean_final_sortedness"]) < min_final - 5.0
            or float(row["mean_final_monotonicity_error"]) > max_error + 5.0
            or float(row["success_rate"]) < min_success - 0.1
        )
        passive_ratio = row["compare_plus_swap_ratio_vs_passive"]
        stuck_ratio = row["compare_plus_swap_ratio_vs_stuck"]
        ratio_values = [float(value) for value in (passive_ratio, stuck_ratio) if pd.notna(value) and math.isfinite(float(value))]
        efficiency_sensitive = bool(ratio_values and all(value > 2.0 or value < 0.5 for value in ratio_values))
        nearest_dg_delta = min(abs(float(row["delta_dg_primary_vs_passive"])), abs(float(row["delta_dg_primary_vs_stuck"])))
        nearest_curv_delta = min(
            abs(float(row["delta_sortedness_curvature_vs_passive"])),
            abs(float(row["delta_sortedness_curvature_vs_stuck"])),
        )
        nearest_attempt_delta = min(
            abs(float(row["delta_frozen_attempts_vs_passive"])),
            abs(float(row["delta_frozen_attempts_vs_stuck"])),
        )
        path_sensitive = nearest_dg_delta > 25.0 or nearest_curv_delta > 0.75
        frozen_attempt_sensitive = nearest_attempt_delta > 1000.0
        if final_sensitive:
            classes.append("behavior_sensitive_final_state")
            reasons.append("final Sortedness, success, or monotonicity error moved outside passive/stuck envelope")
        elif efficiency_sensitive or path_sensitive or frozen_attempt_sensitive:
            classes.append("behavior_sensitive_efficiency_or_path")
            active_reasons = []
            if efficiency_sensitive:
                active_reasons.append("efficiency ratio outside [0.5, 2.0] against both original limits")
            if path_sensitive:
                active_reasons.append("DG delta exceeds 25 points or curvature delta exceeds 0.75 against nearest original limit")
            if frozen_attempt_sensitive:
                active_reasons.append("frozen-attempt mean delta exceeds 1000 against nearest original limit")
            reasons.append("; ".join(active_reasons))
        else:
            classes.append("behavior_within_original_envelope")
            reasons.append("final state, efficiency, DG/path shape, and frozen attempts within passive/stuck envelope thresholds")
    classified["behavior_sensitivity_class"] = classes
    classified["classification_rule"] = reasons
    return summary, classified


def make_behavior_figure(summary_df: pd.DataFrame, classification_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    variant_order = [row["name"] for row in BEHAVIOR_VARIANTS if row["name"] in set(summary_df["behavior_variant"])]
    conditions = (
        summary_df.assign(
            condition_label=lambda frame: frame["condition_pair_id"].str.replace("_", " ", regex=False)
            + " "
            + frame["passive_condition_id"].astype(str)
            + "/"
            + frame["stuck_condition_id"].astype(str)
        )[["condition_pair_id", "condition_label"]]
        .drop_duplicates()
        .sort_values("condition_pair_id")
    )
    condition_order = conditions["condition_pair_id"].tolist()
    condition_labels = conditions["condition_label"].tolist()
    pivots = [
        (
            "Mean final Sortedness",
            summary_df.pivot_table(index="condition_pair_id", columns="behavior_variant", values="mean_final_sortedness", aggfunc="mean").reindex(
                index=condition_order, columns=variant_order
            ),
            "viridis",
            "%",
            None,
        ),
        (
            "DG primary delta vs original passive",
            classification_df.pivot_table(index="condition_pair_id", columns="behavior_variant", values="delta_dg_primary_vs_passive", aggfunc="mean").reindex(
                index=condition_order, columns=variant_order
            ),
            "PRGn",
            "DG",
            "signed",
        ),
        (
            "log2 efficiency ratio vs original passive",
            np.log2(
                classification_df.pivot_table(
                    index="condition_pair_id",
                    columns="behavior_variant",
                    values="compare_plus_swap_ratio_vs_passive",
                    aggfunc="mean",
                ).reindex(index=condition_order, columns=variant_order)
            ),
            "coolwarm",
            "log2 ratio",
            "ratio",
        ),
        (
            "Mean blocked behavior attempts",
            summary_df.pivot_table(index="condition_pair_id", columns="behavior_variant", values="mean_behavior_blocked_attempts", aggfunc="mean").reindex(
                index=condition_order, columns=variant_order
            ),
            "magma",
            "blocked",
            "count",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(19, 10.5))
    for axis, (title, pivot, cmap, label, value_kind) in zip(axes.flat, pivots, strict=True):
        data = pivot.to_numpy(dtype=float)
        finite = data[np.isfinite(data)]
        if value_kind in {"signed", "ratio"}:
            vmax = max(1.0, float(np.nanmax(np.abs(finite))) if finite.size else 1.0)
            vmin = -vmax
        elif value_kind == "count":
            vmin = 0.0
            vmax = max(1.0, float(np.nanmax(finite)) if finite.size else 1.0)
        else:
            vmin = 0.0
            vmax = 100.0
        im = axis.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks(range(len(variant_order)))
        axis.set_xticklabels([name.replace("_", "\n") for name in variant_order], fontsize=7)
        axis.set_yticks(range(len(condition_order)))
        axis.set_yticklabels(condition_labels if axis in (axes[0, 0], axes[1, 0]) else [], fontsize=8)
        for i in range(len(condition_order)):
            for j in range(len(variant_order)):
                value = pivot.iloc[i, j]
                if pd.notna(value):
                    if value_kind == "ratio":
                        text = f"{2.0 ** float(value):.1f}x"
                    elif value_kind == "count":
                        text = f"{float(value):.0f}"
                    else:
                        text = f"{float(value):.1f}"
                    axis.text(j, i, text, ha="center", va="center", fontsize=6.2, color="white" if value_kind == "count" else "black")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04).set_label(label)
    fig.suptitle("E02 S12 Frozen Cell behavior variants", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def behavior_validation_rows(result_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "research_step_id",
        "condition_pair_id",
        "passive_condition_id",
        "stuck_condition_id",
        "algorithm",
        "frozen_count",
        "repeat_index",
        "behavior_variant",
        "behavior_family",
        "frozen_semantics",
        "behavior_seed",
        "initial_frozen_indices_json",
        "initial_frozen_values_json",
        "final_frozen_positions_json",
        "final_frozen_values_json",
        "frozen_identity_count_preserved",
        "frozen_identity_values_preserved",
        "stuck_limit_positions_preserved",
        "frozen_identity_validation_passed",
        "behavior_transition_count",
        "behavior_attempt_count",
        "behavior_allowed_attempt_count",
        "behavior_blocked_attempt_count",
        "behavior_attempt_counts_json",
        "behavior_transition_log_sha256",
        "behavior_attempt_log_sha256",
    ]
    return result_df[columns].sort_values(["condition_pair_id", "repeat_index", "behavior_variant"]).reset_index(drop=True)


def micro_behavior_validation(repo_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    fixtures = [
        (
            "probabilistic_stuck_decisions",
            {"type": "probabilistic_stuck", "label": "micro_probabilistic", "block_probability": 0.5, "seed": 101},
            lambda summary, transitions: summary["attempt_count"] > 0,
        ),
        (
            "time_varying_transitions",
            {"type": "time_varying", "label": "micro_time", "window_events": 1, "seed": 102},
            lambda summary, transitions: summary["transition_count"] > 1,
        ),
        (
            "fatigue_recovery_transition",
            {"type": "fatigue_recovery", "label": "micro_fatigue", "fatigue_threshold": 1, "recovery_events": 1, "seed": 103},
            lambda summary, transitions: any(row.get("to_state") == "recovering_stuck" for row in transitions)
            and any(row.get("reason") == "recovery_window_elapsed" for row in transitions),
        ),
        (
            "directional_sticky_left_blocking",
            {"type": "directional_sticky", "label": "micro_sticky_left", "blocked_from": "left", "seed": 104},
            lambda summary, transitions: summary["blocked_attempt_count"] > 0,
        ),
        (
            "directional_sticky_right_state",
            {"type": "directional_sticky", "label": "micro_sticky_right", "blocked_from": "right", "seed": 105},
            lambda summary, transitions: summary["transition_count"] > 0,
        ),
    ]
    for name, behavior, predicate in fixtures:
        result = simulate(
            SimulatorConfig(
                values=(2, 1, 3),
                algorithm="bubble",
                frozen_indices=(1,),
                frozen_semantics="dynamic",
                frozen_behavior=behavior,
                activation_seed=777,
                policy_seed=777,
                activation_distribution="left_to_right_active",
                max_events=300,
                max_successful_swaps=100,
                stall_events=80,
                stop_when_sorted=False,
                repo_dir=repo_dir,
            )
        )
        summary = dict(result.frozen_behavior_summary)
        transitions = list(result.frozen_behavior_transition_log)
        passed = bool(predicate(summary, transitions))
        checks.append(
            {
                "name": name,
                "passed": passed,
                "transitionCount": int(summary.get("transition_count", 0)),
                "attemptCount": int(summary.get("attempt_count", 0)),
                "allowedAttemptCount": int(summary.get("allowed_attempt_count", 0)),
                "blockedAttemptCount": int(summary.get("blocked_attempt_count", 0)),
                "finalStateByThread": summary.get("final_state_by_thread", {}),
                "attemptCounts": summary.get("attempt_counts", {}),
                "transitionStates": sorted({str(row.get("to_state")) for row in transitions}),
                "traceSha256": stable_json_sha256(result.records),
            }
        )
    return {
        "microBehaviorValidationPassed": bool(all(row["passed"] for row in checks)),
        "microBehaviorValidationChecks": checks,
    }


def run_seed_and_determinism_checks(tasks: Sequence[BehaviorTask], result_df: pd.DataFrame) -> dict[str, Any]:
    sample_tasks: list[BehaviorTask] = []
    for variant in [row["name"] for row in BEHAVIOR_VARIANTS]:
        candidates = [task for task in tasks if task.behavior_variant == variant]
        if candidates:
            sample_tasks.append(candidates[0])
    deterministic_ok = True
    checks: list[dict[str, Any]] = []
    index = result_df.set_index(["condition_pair_id", "behavior_variant", "repeat_index"])
    for task in sample_tasks:
        rerun = run_behavior_task(task)
        key = (task.pair.pair_id, task.behavior_variant, task.repeat_index)
        original = index.loc[key]
        passed = bool(
            str(rerun["final_array_sha256"]) == str(original["final_array_sha256"])
            and str(rerun["trace_values_sha256"]) == str(original["trace_values_sha256"])
            and str(rerun["behavior_transition_log_sha256"]) == str(original["behavior_transition_log_sha256"])
            and str(rerun["behavior_attempt_log_sha256"]) == str(original["behavior_attempt_log_sha256"])
            and int(rerun["event_count"]) == int(original["event_count"])
        )
        deterministic_ok = deterministic_ok and passed
        checks.append(
            {
                "condition_pair_id": task.pair.pair_id,
                "repeat_index": int(task.repeat_index),
                "behavior_variant": task.behavior_variant,
                "passed": passed,
                "final_array_sha256": rerun["final_array_sha256"],
                "trace_values_sha256": rerun["trace_values_sha256"],
                "behavior_transition_log_sha256": rerun["behavior_transition_log_sha256"],
                "behavior_attempt_log_sha256": rerun["behavior_attempt_log_sha256"],
            }
        )
    return {
        "simulationDeterminismSamplePassed": bool(deterministic_ok),
        "simulationDeterminismSampleCount": int(len(checks)),
        "simulationDeterminismChecks": checks,
    }


def validate_limiting_case_recovery(result_df: pd.DataFrame) -> dict[str, Any]:
    index = result_df.set_index(["condition_pair_id", "repeat_index", "behavior_variant"])
    passive_mismatches: list[dict[str, Any]] = []
    stuck_mismatches: list[dict[str, Any]] = []
    compare_fields = [
        "final_array_sha256",
        "trace_values_sha256",
        "activation_log_sha256",
        "event_count",
        "swap_only_steps",
        "comparison_steps_observed",
        "frozen_attempt_count",
        "stop_reason",
    ]
    keys = sorted({(pair, repeat) for pair, repeat, _variant in index.index})
    for pair, repeat in keys:
        original_passive = index.loc[(pair, repeat, "original_passive")]
        dynamic_passive = index.loc[(pair, repeat, "dynamic_passive_limit")]
        original_stuck = index.loc[(pair, repeat, "original_stuck")]
        dynamic_stuck = index.loc[(pair, repeat, "dynamic_stuck_limit")]
        passive_bad = [field for field in compare_fields if original_passive[field] != dynamic_passive[field]]
        stuck_bad = [field for field in compare_fields if original_stuck[field] != dynamic_stuck[field]]
        if passive_bad and len(passive_mismatches) < 12:
            passive_mismatches.append({"condition_pair_id": pair, "repeat_index": int(repeat), "fields": passive_bad})
        if stuck_bad and len(stuck_mismatches) < 12:
            stuck_mismatches.append({"condition_pair_id": pair, "repeat_index": int(repeat), "fields": stuck_bad})
    return {
        "passiveLimitRecoveryPassed": bool(not passive_mismatches),
        "stuckLimitRecoveryPassed": bool(not stuck_mismatches),
        "passiveLimitMismatchExamples": passive_mismatches,
        "stuckLimitMismatchExamples": stuck_mismatches,
        "limitingCaseCompareFields": compare_fields,
    }


def validate_outputs(
    *,
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    behavior_validation_df: pd.DataFrame,
    tasks: Sequence[BehaviorTask],
    figure_path: Path,
    result_path: Path,
    dg_toy_validation: Mapping[str, Any],
    micro_validation: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
    limit_recovery: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {(task.pair.pair_id, task.behavior_variant, task.repeat_index) for task in tasks}
    observed_keys = set(zip(result_df["condition_pair_id"], result_df["behavior_variant"], result_df["repeat_index"]))
    expected_variants = [row["name"] for row in BEHAVIOR_VARIANTS if any(task.behavior_variant == row["name"] for task in tasks)]
    observed_variants = [name for name in expected_variants if name in set(result_df["behavior_variant"])]
    dynamic = result_df[result_df["frozen_semantics"] == "dynamic"]
    perturbation = result_df[result_df["behavior_family"].isin(["probabilistic", "time_varying", "fatigue_recovery", "directional_sticky"])]
    classes = set(classification_df["behavior_sensitivity_class"].dropna().unique())
    sensitive_count = int(classification_df["behavior_sensitivity_class"].str.startswith("behavior_sensitive").sum())
    outcome = "constraining/contradictory" if sensitive_count > 0 else "supportive"
    behavior_attempts_by_family = (
        perturbation.groupby("behavior_family", dropna=False)["behavior_attempt_count"].sum().astype(int).to_dict()
        if len(perturbation)
        else {}
    )
    success = bool(
        expected_keys == observed_keys
        and result_path.exists()
        and figure_path.exists()
        and figure_path.stat().st_size > 0
        and bool(result_df["frozen_identity_validation_passed"].all())
        and bool(result_df.loc[result_df["behavior_variant"].isin(["original_stuck", "dynamic_stuck_limit"]), "stuck_limit_positions_preserved"].all())
        and bool(dynamic["behavior_transition_count"].gt(0).all())
        and bool(all(value > 0 for value in behavior_attempts_by_family.values()))
        and bool(limit_recovery["passiveLimitRecoveryPassed"])
        and bool(limit_recovery["stuckLimitRecoveryPassed"])
        and bool(dg_toy_validation["toyCaseValidationPassed"])
        and bool(micro_validation["microBehaviorValidationPassed"])
        and bool(reproducibility["simulationDeterminismSamplePassed"])
        and classes <= {
            "original_baseline",
            "dynamic_limit_case",
            "behavior_within_original_envelope",
            "behavior_sensitive_efficiency_or_path",
            "behavior_sensitive_final_state",
        }
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
            "Matrix is bounded to first 10 repeats and six pure-algorithm Frozen Cell condition pairs with frozen_count 1 or 3.",
            "Runs use the S01 deterministic public-move simulator, not OS-thread replays.",
            "S12 fixes Frozen Cell placement to the E01 random-bank placement used as S11's random_bank reference; other placements remain S11 caveats.",
            "Dynamic behavior variants are model extensions, not paper-reported perturbation semantics.",
            "S08 Delayed Gratification is recomputed over deterministic Sortedness trajectories; S08 matched-null caveats still apply.",
        ],
        "recommendedNextAction": "Review S12 behavior sensitivity, then proceed to S13 only after explicit Chief Scientist instruction.",
        "expectedRunCount": int(len(expected_keys)),
        "observedRunCount": int(len(observed_keys)),
        "resultRowCount": int(len(result_df)),
        "summaryRowCount": int(len(summary_df)),
        "classificationRowCount": int(len(classification_df)),
        "behaviorValidationRowCount": int(len(behavior_validation_df)),
        "conditionPairIds": sorted({task.pair.pair_id for task in tasks}),
        "passiveConditionIds": sorted({task.pair.passive_condition["condition_id"] for task in tasks}),
        "stuckConditionIds": sorted({task.pair.stuck_condition["condition_id"] for task in tasks}),
        "behaviorVariants": expected_variants,
        "observedBehaviorVariants": observed_variants,
        "repeatIndexes": sorted({int(task.repeat_index) for task in tasks}),
        "allExpectedRunsPresent": bool(expected_keys == observed_keys),
        "frozenIdentityValidEveryRun": bool(result_df["frozen_identity_validation_passed"].all()),
        "stuckLimitPositionsPreservedEveryRun": bool(
            result_df.loc[result_df["behavior_variant"].isin(["original_stuck", "dynamic_stuck_limit"]), "stuck_limit_positions_preserved"].all()
        ),
        "dynamicTransitionsLoggedEveryRun": bool(dynamic["behavior_transition_count"].gt(0).all()),
        "perturbationAttemptCountsByFamily": behavior_attempts_by_family,
        "perturbationAttemptsLoggedEveryFamily": bool(all(value > 0 for value in behavior_attempts_by_family.values())),
        "dgToyValidationPassed": bool(dg_toy_validation["toyCaseValidationPassed"]),
        "classificationClasses": sorted(classes),
        "behaviorSensitivityClassCounts": classification_df["behavior_sensitivity_class"].value_counts().sort_index().to_dict(),
        "sensitiveSliceCount": sensitive_count,
        "figureNonempty": bool(figure_path.exists() and figure_path.stat().st_size > 0),
        "plannedMatrixWritten": bool(result_path.exists()),
        **dict(limit_recovery),
        **dict(micro_validation),
        **dict(reproducibility),
    }
    validation["validationResult"] = (
        "passed: expected behavior matrix, Frozen Cell identities, behavior transition logging, perturbation attempt logging, passive/stuck limiting-case recovery, DG toy cases, deterministic rerun samples, and figure validated"
        if success
        else "failed: one or more S12 validation checks did not pass"
    )
    return validation


def build_report(
    *,
    generated_at: str,
    result_path: Path,
    summary_path: Path,
    classification_path: Path,
    behavior_validation_path: Path,
    figure_path: Path,
    validation_path: Path,
    status_path: Path,
    dg_toy_path: Path,
    micro_validation_path: Path,
    log_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    summary_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    behavior_validation_df: pd.DataFrame,
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
    class_counts = classification_df["behavior_sensitivity_class"].value_counts().sort_index().to_dict()
    largest_final = classification_df[~classification_df["behavior_variant"].isin(["original_passive", "original_stuck"])].copy()
    largest_final = largest_final.sort_values("delta_final_sortedness_vs_passive").head(18)[
        [
            "condition_pair_id",
            "algorithm",
            "frozen_count",
            "behavior_variant",
            "behavior_sensitivity_class",
            "mean_final_sortedness",
            "delta_final_sortedness_vs_passive",
            "mean_final_monotonicity_error",
            "compare_plus_swap_ratio_vs_passive",
            "delta_dg_primary_vs_passive",
            "mean_behavior_blocked_attempts",
        ]
    ]
    transition_excerpt = behavior_validation_df[
        [
            "condition_pair_id",
            "repeat_index",
            "behavior_variant",
            "behavior_transition_count",
            "behavior_attempt_count",
            "behavior_allowed_attempt_count",
            "behavior_blocked_attempt_count",
            "behavior_attempt_counts_json",
        ]
    ].head(24)
    return f"""# E02 S12 Full Results: Frozen Cell Behavior Variants

## Top Summary

- Step ID: S12
- Completion status: {validation['status']}
- Artifacts written: {', '.join(validation['artifactsWritten'])}
- Validation result: {validation['validationResult']}
- Outcome classification: {validation['outcomeClassification']}
- Caveats or blockers: {', '.join(validation['caveatsOrBlockers'])}
- Lay summary: S12 changed what Frozen Cells do rather than where they sit. In this bounded deterministic rerun, passive/stuck limiting cases were recovered exactly, while probabilistic, time-varying, fatigue/recovery, and directionally sticky extensions changed path or efficiency metrics in enough slices that Frozen Cell robustness remains behavior-assumption sensitive.
- Recommended next action: {validation['recommendedNextAction']}

## Frozen Question

Do robustness and DG persist when Frozen Cells are probabilistic, time-varying, fatigued, recovering, or directionally sticky rather than only passive or stuck?

## Inputs

- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- S01 deterministic simulator with S12 dynamic Frozen Cell controller: `src/e02/deterministic_simulator.py`
- S11 placement framework: S12 fixes placement to the original E01 `random_bank` Frozen Cell index bank, matching S11's reference placement.
- S08 DG recomputation code: `scripts/e02_s08_dg_nulls.py`
- Selected condition pairs: `{parameters['conditionPairIds']}`
- Behavior variants: `{parameters['behaviorVariants']}`
- Repeat indexes: `{parameters['repeatIndexes']}`

No external datasets were required and no new dependencies were installed.

## Methods

S12 selected six pure cell-view Frozen Cell condition pairs: Bubble, Insertion, and Selection with one or three Frozen Cells. Each pair uses the E01 passive condition as the value/assignment template and the matched E01 stuck condition as the original stuck reference. For every condition pair and repeat, Frozen Cell placement was held at the original E01 random-bank indices used as the S11 reference placement.

The simulator was extended with an optional dynamic Frozen Cell behavior controller. The controller leaves public cell `move()` methods intact but intercepts swaps involving Frozen Cell target identities. It logs per-run behavior-state transitions, sampled contact attempts, allowed/blocked counts, and final per-Frozen-identity states. The original `passive` and `stuck` modes remain unchanged; `dynamic_passive_limit` and `dynamic_stuck_limit` were run and compared field-by-field to prove limiting-case recovery.

Dynamic variants were:

- `probabilistic_stuck_p25` and `probabilistic_stuck_p75`: target contacts are blocked by seeded Bernoulli decisions.
- `time_varying_periodic`: target behavior alternates between stuck and passive every 250 events.
- `fatigue_recovery`: repeated allowed nudges fatigue a Frozen target into a temporary stuck state, then recovery restores passive target mobility.
- `directional_sticky_left` and `directional_sticky_right`: target contacts are blocked only when initiated from one side.

Activation and public policy seeds were matched across behavior variants within each condition/repeat. Dynamic behavior RNG seeds were variant-specific and recorded. DG was recomputed with the S08 formula over each deterministic Sortedness trajectory. Path shape is summarized by scalar Sortedness curvature and value-state curvature.

## Commands

- `python scripts/e02_s12_frozen_behaviors.py --repo-dir /workspace/cell-research --artifacts-dir /artifacts --max-repeats {parameters['maxRepeats']} --workers {parameters['workers']} --run-unit-tests`
- Unit-test command inside the script: `{unit_line}`

## Results

- Matrix rows: {validation['resultRowCount']}
- Summary rows: {validation['summaryRowCount']}
- Classification rows: {validation['classificationRowCount']}
- Behavior validation rows: {validation['behaviorValidationRowCount']}
- Behavior sensitivity class counts: {json.dumps(class_counts, sort_keys=True)}
- Expected runs present: {validation['allExpectedRunsPresent']}
- Passive limiting-case recovery: {validation['passiveLimitRecoveryPassed']}
- Stuck limiting-case recovery: {validation['stuckLimitRecoveryPassed']}
- Dynamic transitions logged every dynamic run: {validation['dynamicTransitionsLoggedEveryRun']}
- Perturbation attempt counts by family: {json.dumps(validation['perturbationAttemptCountsByFamily'], sort_keys=True)}

### Largest Final Sortedness Drops

{markdown_table(largest_final, max_rows=18)}

### Behavior Logging Validation Excerpt

{markdown_table(transition_excerpt, max_rows=24)}

## Validation

- All expected condition/variant/repeat runs present: {validation['allExpectedRunsPresent']}
- Frozen identities validated in every run: {validation['frozenIdentityValidEveryRun']}
- Stuck limiting-case positions preserved: {validation['stuckLimitPositionsPreservedEveryRun']}
- Dynamic behavior transitions logged in every dynamic run: {validation['dynamicTransitionsLoggedEveryRun']}
- Perturbation behavior attempts logged for every behavior family: {validation['perturbationAttemptsLoggedEveryFamily']}
- Passive limiting case exactly recovered against original passive: {validation['passiveLimitRecoveryPassed']}
- Stuck limiting case exactly recovered against original stuck: {validation['stuckLimitRecoveryPassed']}
- S08 DG toy-case validation passed: {validation['dgToyValidationPassed']}
- Micro behavior-state validation passed: {validation['microBehaviorValidationPassed']}
- Simulation determinism sample passed: {validation['simulationDeterminismSamplePassed']} across {validation['simulationDeterminismSampleCount']} sampled reruns.
- Planned matrix written: {validation['plannedMatrixWritten']}
- Figure nonempty: {validation['figureNonempty']}
- Unit tests: {unit_line}

Validation JSON: `{validation_path}`  
Status JSON: `{status_path}`  
DG toy validation JSON: `{dg_toy_path}`  
Micro behavior validation JSON: `{micro_validation_path}`

## Artifacts

- Result matrix: `{result_path}`
- Summary table: `{summary_path}`
- Classification table: `{classification_path}`
- Behavior validation table: `{behavior_validation_path}`
- Figure: `{figure_path}`
- Validation JSON: `{validation_path}`
- Status JSON: `{status_path}`
- Log: `{log_path}`
- Artifact manifest: `{manifest_path}`
- Source snapshot manifest: `{src_manifest_path}`

## Provenance

- Generated at: {generated_at}
- Elapsed seconds: {elapsed_seconds:.2f}
- Git commit before final commit: `{git_commit}`
- Git status before final commit: `{git_status}`
- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Worker count: {parameters['workers']}
- Thread environment: `OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}`, `MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}`, `OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}`

## Caveats, Blockers, And Limitations

- This is a bounded first-10-repeat deterministic rerun, not full 100-repeat paper-scale inference.
- Dynamic Frozen Cell behaviors are extensions introduced for E02 stress testing, not reported paper semantics.
- Frozen Cell placement is fixed to the E01 random bank; S11 showed placement itself can change outcomes.
- The simulator preserves public local cell `move()` methods but replaces OS/Python thread scheduling with explicit deterministic activation.
- Comparison counts remain the public `StatusProbe` actionable-comparison proxy inherited from E01/S01.
- DG remains a Sortedness-trajectory proxy; S08 showed matched nulls can explain or exceed many real DG values.

## Recommended Next Action

Proceed to S13 stop-condition sensitivity only after explicit Chief Scientist instruction. S13 should treat S12 dynamic variants as behavior-assumption stress evidence, not as new paper-replication baselines.
"""


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    generated_at = utc_now()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_dir = artifacts_dir / "results"
    figure_dir = artifacts_dir / "figures" / "e02"
    log_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    for path in (step_dir, result_dir, figure_dir, log_dir, src_snapshot_dir):
        path.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "e02_s12_frozen_behaviors.log"
    result_path = result_dir / "e02_frozen_behavior_variants.parquet"
    figure_path = figure_dir / "frozen_behavior_effects.png"
    summary_path = step_dir / "e02_frozen_behavior_summary.csv"
    classification_path = step_dir / "e02_frozen_behavior_classification.csv"
    behavior_validation_path = step_dir / "e02_frozen_behavior_validation.csv"
    validation_path = step_dir / "s12_validation.json"
    status_path = step_dir / "status.json"
    dg_toy_path = step_dir / "e02_s12_dg_toy_validation.json"
    micro_validation_path = step_dir / "e02_s12_behavior_micro_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s12_frozen_behaviors_manifest.json"
    report_path = step_dir / "research_step_full_results.md"

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"{generated_at} Starting E02 S12 Frozen Cell behavior variants\n")
        log.write(f"repo_dir={repo_dir}\nartifacts_dir={artifacts_dir}\n")

    cfg = load_e01_config(args.config_path)
    pairs = selected_condition_pairs(cfg, args.template_condition_ids)
    variants = selected_behavior_variants(args.behavior_variants)
    tasks = build_tasks(
        cfg,
        pairs,
        variants,
        repo_dir,
        args.max_repeats,
        args.max_events,
        args.max_successful_swaps,
        args.stall_events,
        args.activation_distribution,
    )
    parameters = {
        "maxRepeats": int(args.max_repeats),
        "workers": int(args.workers),
        "maxEvents": int(args.max_events),
        "maxSuccessfulSwaps": int(args.max_successful_swaps),
        "stallEvents": int(args.stall_events),
        "activationDistribution": args.activation_distribution,
        "conditionPairIds": [pair.pair_id for pair in pairs],
        "passiveConditionIds": [pair.passive_condition["condition_id"] for pair in pairs],
        "stuckConditionIds": [pair.stuck_condition["condition_id"] for pair in pairs],
        "behaviorVariants": [row["name"] for row in variants],
        "repeatIndexes": sorted({task.repeat_index for task in tasks}),
        "taskCount": len(tasks),
    }

    unit_result: Mapping[str, Any] | None = None
    if args.run_unit_tests:
        unit_result = run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.e02.test_deterministic_simulator",
                "tests.e02.test_frozen_behaviors",
            ],
            repo_dir,
        )
        if not unit_result["success"]:
            write_json(step_dir / "s12_unit_test_failure.json", unit_result)
            raise RuntimeError(f"S12 unit tests failed; see {step_dir / 's12_unit_test_failure.json'}")

    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for task in tasks:
            rows.append(run_behavior_task(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            future_to_task = {executor.submit(run_behavior_task, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    raise RuntimeError(
                        f"S12 task failed for {task.pair.pair_id} repeat={task.repeat_index} variant={task.behavior_variant}"
                    ) from exc

    result_df = pd.DataFrame(rows).sort_values(["condition_pair_id", "repeat_index", "behavior_variant"]).reset_index(drop=True)
    result_df.to_parquet(result_path, index=False)
    summary_df, classification_df = summarize_results(result_df)
    behavior_validation_df = behavior_validation_rows(result_df)
    summary_df.to_csv(summary_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    behavior_validation_df.to_csv(behavior_validation_path, index=False)
    make_behavior_figure(summary_df, classification_df, figure_path)

    dg_toy_validation = validate_toy_cases()
    write_json(dg_toy_path, dg_toy_validation)
    micro_validation = micro_behavior_validation(repo_dir)
    write_json(micro_validation_path, micro_validation)
    reproducibility = run_seed_and_determinism_checks(tasks, result_df)
    limit_recovery = validate_limiting_case_recovery(result_df)
    validation = validate_outputs(
        result_df=result_df,
        summary_df=summary_df,
        classification_df=classification_df,
        behavior_validation_df=behavior_validation_df,
        tasks=tasks,
        figure_path=figure_path,
        result_path=result_path,
        dg_toy_validation=dg_toy_validation,
        micro_validation=micro_validation,
        reproducibility=reproducibility,
        limit_recovery=limit_recovery,
    )
    artifact_paths = [
        (result_path, "planned S12 behavior-variant result matrix"),
        (figure_path, "planned S12 Frozen Cell behavior effects figure"),
        (summary_path, "condition-by-behavior summary metrics"),
        (classification_path, "behavior sensitivity classification table"),
        (behavior_validation_path, "behavior-state and identity validation table"),
        (validation_path, "S12 validation JSON"),
        (status_path, "S12 compact status JSON"),
        (dg_toy_path, "S08 DG formula toy validation reused by S12"),
        (micro_validation_path, "S12 dynamic behavior micro-validation JSON"),
        (log_path, "S12 execution log"),
        (manifest_path, "S12 artifact manifest"),
        (src_manifest_path, "S12 source snapshot manifest"),
        (report_path, "S12 full-results report"),
    ]
    validation["artifactsWritten"] = [str(path) for path, _description in artifact_paths]
    validation["validationResult"] = (
        "passed: expected behavior matrix, Frozen Cell identities, behavior transition logging, perturbation attempt logging, passive/stuck limiting-case recovery, DG toy cases, deterministic rerun samples, and figure validated"
        if validation["success"]
        else "failed: one or more S12 validation checks did not pass"
    )
    write_json(validation_path, validation)
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
    write_json(status_path, status)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    src_manifest = {
        "researchStepId": STEP_ID,
        "generatedAt": generated_at,
        "gitCommitBeforeFinalCommit": git_commit,
        "gitStatusBeforeFinalCommit": git_status,
        "sourceFiles": [
            artifact_entry(repo_dir / "src/e02/deterministic_simulator.py", repo_dir, "S01 simulator extended with S12 dynamic Frozen Cell behavior controller"),
            artifact_entry(repo_dir / "scripts/e02_s12_frozen_behaviors.py", repo_dir, "S12 behavior-variant runner"),
            artifact_entry(repo_dir / "tests/e02/test_deterministic_simulator.py", repo_dir, "simulator tests including S12 limiting-case checks"),
            artifact_entry(repo_dir / "tests/e02/test_frozen_behaviors.py", repo_dir, "focused S12 behavior tests"),
        ],
    }
    write_json(src_manifest_path, src_manifest)

    manifest_entries = []
    for path, description in artifact_paths:
        if path == manifest_path:
            continue
        if path.exists():
            manifest_entries.append(artifact_entry(path, artifacts_dir, description))
    manifest = {
        "researchStepId": STEP_ID,
        "generatedAt": generated_at,
        "artifacts": manifest_entries,
        "parameters": parameters,
        "validation": {
            "success": bool(validation["success"]),
            "validationResult": validation["validationResult"],
            "outcomeClassification": validation["outcomeClassification"],
        },
    }
    write_json(manifest_path, manifest)
    validation["artifactsWritten"] = [str(path) for path, _description in artifact_paths]
    write_json(validation_path, validation)
    status["artifactsWritten"] = validation["artifactsWritten"]
    write_json(status_path, status)

    elapsed = time.monotonic() - started
    report = build_report(
        generated_at=generated_at,
        result_path=result_path,
        summary_path=summary_path,
        classification_path=classification_path,
        behavior_validation_path=behavior_validation_path,
        figure_path=figure_path,
        validation_path=validation_path,
        status_path=status_path,
        dg_toy_path=dg_toy_path,
        micro_validation_path=micro_validation_path,
        log_path=log_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        summary_df=summary_df,
        classification_df=classification_df,
        behavior_validation_df=behavior_validation_df,
        validation=validation,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        elapsed_seconds=elapsed,
        parameters=parameters,
    )
    report_path.write_text(report, encoding="utf-8")
    manifest["artifacts"] = [
        artifact_entry(path, artifacts_dir, description)
        for path, description in artifact_paths
        if path.exists() and path != manifest_path
    ]
    write_json(manifest_path, manifest)
    validation["artifactsWritten"] = [str(path) for path, _description in artifact_paths if path.exists()]
    write_json(validation_path, validation)
    status["artifactsWritten"] = validation["artifactsWritten"]
    write_json(status_path, status)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{utc_now()} Completed S12 status={validation['status']} success={validation['success']} elapsed={elapsed:.2f}s\n")
        log.write(json.dumps({"validation": validation["validationResult"], "artifacts": validation["artifactsWritten"]}, indent=2) + "\n")
    if not validation["success"]:
        raise RuntimeError(f"S12 validation failed; see {validation_path}")


if __name__ == "__main__":
    main()
