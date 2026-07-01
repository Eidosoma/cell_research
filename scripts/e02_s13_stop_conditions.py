#!/usr/bin/env python3
"""Run E02 S13 stop-condition sensitivity tests."""

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


STEP_ID = "S13"
STEP_NUMBER = 13
EXPERIMENT_ID = "E02"
SOURCE_EXPERIMENT_ID = "E01"

SELECTED_CONDITION_IDS = (
    "E01C004",
    "E01C005",
    "E01C006",
    "E01C022",
    "E01C023",
    "E01C024",
    "E01C040",
    "E01C041",
    "E01C042",
)

BASE_MAX_EVENTS = 150_000
BASE_MAX_SUCCESSFUL_SWAPS = 80_000
BASE_STALL_EVENTS = 2_000

STOP_REGIMES: tuple[dict[str, Any], ...] = (
    {
        "name": "reference_no_legal_2000",
        "family": "reference",
        "max_events": BASE_MAX_EVENTS,
        "max_successful_swaps": BASE_MAX_SUCCESSFUL_SWAPS,
        "stall_events": BASE_STALL_EVENTS,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": None,
        "convergence_criterion": "no_legal_action",
        "description": "S11/S12 reference: exact sorted or no-legal-action window with baseline caps.",
    },
    {
        "name": "short_no_legal_250",
        "family": "no_movement_window",
        "max_events": BASE_MAX_EVENTS,
        "max_successful_swaps": BASE_MAX_SUCCESSFUL_SWAPS,
        "stall_events": 250,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": None,
        "convergence_criterion": "no_legal_action",
        "description": "Shorter no-legal-action window.",
    },
    {
        "name": "long_no_legal_8000",
        "family": "no_movement_window",
        "max_events": BASE_MAX_EVENTS,
        "max_successful_swaps": BASE_MAX_SUCCESSFUL_SWAPS,
        "stall_events": 8_000,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": None,
        "convergence_criterion": "no_legal_action",
        "description": "Longer no-legal-action window.",
    },
    {
        "name": "low_event_cap_25000",
        "family": "max_step_cap",
        "max_events": 25_000,
        "max_successful_swaps": BASE_MAX_SUCCESSFUL_SWAPS,
        "stall_events": BASE_STALL_EVENTS,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": None,
        "convergence_criterion": "no_legal_action",
        "description": "Lower maximum event cap.",
    },
    {
        "name": "low_swap_cap_5000",
        "family": "max_step_cap",
        "max_events": BASE_MAX_EVENTS,
        "max_successful_swaps": 5_000,
        "stall_events": BASE_STALL_EVENTS,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": None,
        "convergence_criterion": "no_legal_action",
        "description": "Lower successful-swap cap.",
    },
    {
        "name": "sortedness_threshold_95",
        "family": "sortedness_threshold",
        "max_events": BASE_MAX_EVENTS,
        "max_successful_swaps": BASE_MAX_SUCCESSFUL_SWAPS,
        "stall_events": BASE_STALL_EVENTS,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": 95.0,
        "convergence_criterion": "no_legal_action",
        "description": "Allow early stop at at least 95 percent adjacent Sortedness.",
    },
    {
        "name": "sortedness_threshold_99",
        "family": "sortedness_threshold",
        "max_events": BASE_MAX_EVENTS,
        "max_successful_swaps": BASE_MAX_SUCCESSFUL_SWAPS,
        "stall_events": BASE_STALL_EVENTS,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": 99.0,
        "convergence_criterion": "no_legal_action",
        "description": "Allow early stop at at least 99 percent adjacent Sortedness.",
    },
    {
        "name": "no_state_change_2000",
        "family": "convergence_criterion",
        "max_events": BASE_MAX_EVENTS,
        "max_successful_swaps": BASE_MAX_SUCCESSFUL_SWAPS,
        "stall_events": BASE_STALL_EVENTS,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": None,
        "convergence_criterion": "no_state_change",
        "description": "Stop after a window with no state-signature changes.",
    },
    {
        "name": "no_swap_2000",
        "family": "convergence_criterion",
        "max_events": BASE_MAX_EVENTS,
        "max_successful_swaps": BASE_MAX_SUCCESSFUL_SWAPS,
        "stall_events": BASE_STALL_EVENTS,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": None,
        "convergence_criterion": "no_swap",
        "description": "Stop after a window with no successful swaps.",
    },
    {
        "name": "sortedness_plateau_2000",
        "family": "convergence_criterion",
        "max_events": BASE_MAX_EVENTS,
        "max_successful_swaps": BASE_MAX_SUCCESSFUL_SWAPS,
        "stall_events": BASE_STALL_EVENTS,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": None,
        "convergence_criterion": "sortedness_plateau",
        "description": "Stop after a window with unchanged adjacent Sortedness.",
    },
    {
        "name": "no_convergence_event_cap_25000",
        "family": "convergence_criterion",
        "max_events": 25_000,
        "max_successful_swaps": BASE_MAX_SUCCESSFUL_SWAPS,
        "stall_events": BASE_STALL_EVENTS,
        "stop_when_sorted": True,
        "stop_sortedness_threshold": None,
        "convergence_criterion": "none",
        "description": "Disable convergence windows and rely on exact sortedness or event cap.",
    },
)
STOP_REGIME_INDEX = {row["name"]: idx + 1 for idx, row in enumerate(STOP_REGIMES)}


@dataclass(frozen=True)
class StopTask:
    condition: dict[str, Any]
    repeat_index: int
    stop_regime: dict[str, Any]
    values: tuple[int, ...]
    assignments: tuple[str, ...]
    reverse_directions: tuple[bool, ...]
    frozen_indices: tuple[int, ...]
    initial_frozen_values: tuple[int, ...]
    initial_array_seed: int
    frozen_bank_seed: int | None
    activation_seed: int
    repo_dir: str
    activation_distribution: str
    sort_direction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config-path", type=Path, default=Path("/previous-artifacts/E01/configs/e01_baseline_configs.json"))
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--activation-distribution",
        choices=("uniform_active", "left_to_right_active", "right_to_left_active"),
        default="uniform_active",
    )
    parser.add_argument("--condition-ids", nargs="*", default=list(SELECTED_CONDITION_IDS))
    parser.add_argument("--stop-regimes", nargs="*", default=[row["name"] for row in STOP_REGIMES])
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


def activation_seed(base_seed: int, condition_id: str, repeat_idx: int) -> int:
    condition_num = condition_seed_component(condition_id)
    return base_seed * 1_000_000 + 1_500_000 + condition_num * 1_000 + repeat_idx


def selected_stop_regimes(names: Sequence[str]) -> list[dict[str, Any]]:
    available = {row["name"]: row for row in STOP_REGIMES}
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise RuntimeError(f"Unsupported S13 stop regimes requested: {unknown}")
    return [dict(available[name]) for name in dict.fromkeys(names)]


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
        raise RuntimeError(f"Missing expected S13 condition IDs in E01 config: {missing}")
    unsupported = [
        row["condition_id"]
        for row in rows
        if row.get("direction_profile") != "all_increasing"
        or row.get("value_bank_id") != "unique_1_to_100"
        or row.get("algorithm") not in {"bubble", "insertion", "selection"}
        or row.get("frozen_semantics") not in {"none", "passive", "stuck"}
    ]
    if unsupported:
        raise RuntimeError(
            "S13 stop-condition matrix supports pure all-increasing unique-value cell-view rows only; "
            f"unsupported rows: {unsupported}"
        )
    return sorted(rows, key=lambda row: row["condition_id"])


def build_tasks(
    cfg: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    stop_regimes: Sequence[Mapping[str, Any]],
    repo_dir: Path,
    max_repeats: int,
    activation_distribution: str,
) -> list[StopTask]:
    repeat_count = min(max_repeats, int(cfg["globalDefaults"]["repeatCount"]))
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    value_banks = cfg["seedBanks"]["valueBanks"]
    frozen_banks = cfg["seedBanks"]["frozenIndexBanks"]
    tasks: list[StopTask] = []
    for condition in rows:
        value_bank = value_banks[condition["value_bank_id"]]
        frozen_bank = frozen_banks.get(condition["frozen_index_bank_id"])
        for repeat_idx in range(repeat_count):
            values = tuple(map(int, value_bank["initialArrays"][repeat_idx]))
            assignments = tuple(str(condition["algorithm"]) for _ in values)
            frozen_indices: tuple[int, ...] = ()
            frozen_seed: int | None = None
            if int(condition.get("frozen_count", 0)) > 0:
                if frozen_bank is None:
                    raise RuntimeError(f"{condition['condition_id']} requires frozen index bank.")
                frozen_indices = tuple(map(int, frozen_bank["indices"][repeat_idx]))
                frozen_seed = int(frozen_bank["seeds"][repeat_idx]) if frozen_bank["seeds"] else None
            initial_frozen_values = tuple(int(values[idx]) for idx in frozen_indices)
            act_seed = activation_seed(base_seed, str(condition["condition_id"]), repeat_idx)
            for regime in stop_regimes:
                tasks.append(
                    StopTask(
                        condition=dict(condition),
                        repeat_index=int(repeat_idx),
                        stop_regime=dict(regime),
                        values=values,
                        assignments=assignments,
                        reverse_directions=reverse_flags_for_condition(condition, assignments),
                        frozen_indices=frozen_indices,
                        initial_frozen_values=initial_frozen_values,
                        initial_array_seed=int(value_bank["seeds"][repeat_idx]),
                        frozen_bank_seed=frozen_seed,
                        activation_seed=act_seed,
                        repo_dir=str(repo_dir),
                        activation_distribution=str(activation_distribution),
                        sort_direction="increasing",
                    )
                )
    return tasks


def finite_or_none(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def stop_config_for_regime(regime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "max_events": int(regime["max_events"]),
        "max_successful_swaps": None
        if regime["max_successful_swaps"] is None
        else int(regime["max_successful_swaps"]),
        "stall_events": int(regime["stall_events"]),
        "stop_when_sorted": bool(regime["stop_when_sorted"]),
        "stop_sortedness_threshold": None
        if regime["stop_sortedness_threshold"] is None
        else float(regime["stop_sortedness_threshold"]),
        "convergence_criterion": str(regime["convergence_criterion"]),
    }


def run_stop_task(task: StopTask) -> dict[str, Any]:
    started = time.monotonic()
    stop_config = stop_config_for_regime(task.stop_regime)
    config = SimulatorConfig(
        values=task.values,
        algotypes=task.assignments,
        reverse_directions=task.reverse_directions,
        frozen_indices=task.frozen_indices,
        frozen_semantics=str(task.condition["frozen_semantics"]),
        activation_seed=task.activation_seed,
        policy_seed=task.activation_seed,
        activation_distribution=task.activation_distribution,
        max_events=int(stop_config["max_events"]),
        max_successful_swaps=stop_config["max_successful_swaps"],
        stall_events=int(stop_config["stall_events"]),
        stop_when_sorted=bool(stop_config["stop_when_sorted"]),
        stop_sortedness_threshold=stop_config["stop_sortedness_threshold"],
        convergence_criterion=str(stop_config["convergence_criterion"]),
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
    initial_metrics = normalized_metric_columns(values_path[0], task.sort_direction, sortedness_curve[0], monotonicity_curve[0])
    final_metrics = normalized_metric_columns(values_path[-1], task.sort_direction, sortedness_curve[-1], monotonicity_curve[-1])
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
    stop_config_json = json.dumps(stop_config, sort_keys=True, separators=(",", ":"))
    initial_frozen_position_mean = float(np.mean(task.frozen_indices)) if task.frozen_indices else np.nan
    initial_frozen_value_mean = float(np.mean(task.initial_frozen_values)) if task.initial_frozen_values else np.nan

    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "evidence_layer": "s13_stop_condition_deterministic_rerun",
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
        "frozen_bank_seed": task.frozen_bank_seed,
        "activation_seed": int(task.activation_seed),
        "policy_seed": int(task.activation_seed),
        "frozen_semantics": task.condition["frozen_semantics"],
        "frozen_count": int(task.condition["frozen_count"]),
        "initial_frozen_indices_json": json.dumps(list(task.frozen_indices), separators=(",", ":")),
        "initial_frozen_values_json": json.dumps(list(task.initial_frozen_values), separators=(",", ":")),
        "final_frozen_positions_json": json.dumps(list(result.final_frozen_positions), separators=(",", ":")),
        "final_frozen_values_json": json.dumps(list(result.final_frozen_values), separators=(",", ":")),
        "frozen_identity_count_preserved": bool(frozen_identity_count_preserved),
        "frozen_identity_values_preserved": bool(frozen_identity_values_preserved),
        "stuck_frozen_positions_preserved": bool(stuck_positions_preserved),
        "frozen_identity_validation_passed": bool(frozen_identity_validation_passed),
        "stop_regime": task.stop_regime["name"],
        "stop_regime_family": task.stop_regime["family"],
        "stop_regime_description": task.stop_regime["description"],
        "stop_config_json": stop_config_json,
        "stop_config_sha256": stable_json_sha256(stop_config),
        "max_events_config": int(stop_config["max_events"]),
        "max_successful_swaps_config": stop_config["max_successful_swaps"],
        "stall_events_config": int(stop_config["stall_events"]),
        "stop_when_sorted_config": bool(stop_config["stop_when_sorted"]),
        "stop_sortedness_threshold_config": stop_config["stop_sortedness_threshold"],
        "convergence_criterion_config": stop_config["convergence_criterion"],
        "stop_reason": result.stop_reason,
        "stop_reason_logged": bool(result.stop_reason),
        "max_guard_hit": bool(result.max_guard_hit),
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
        "final_sorted_non_decreasing": bool(monotonicity_curve[-1] == 0),
        "elapsed_seconds": float(time.monotonic() - started),
        "initial_frozen_position_mean": finite_or_none(initial_frozen_position_mean),
        "initial_frozen_value_mean": finite_or_none(initial_frozen_value_mean),
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
            "stop_regime",
            "stop_regime_family",
        ],
        dropna=False,
    )
    summary = grouped.agg(
        n_runs=("repeat_index", "nunique"),
        success_rate=("final_sorted_non_decreasing", "mean"),
        mean_final_sortedness=("final_sortedness_percent", "mean"),
        sd_final_sortedness=("final_sortedness_percent", "std"),
        mean_final_monotonicity_error=("final_monotonicity_error_count", "mean"),
        mean_compare_plus_swap_steps=("compare_plus_swap_steps", "mean"),
        mean_event_count=("event_count", "mean"),
        mean_swap_only_steps=("swap_only_steps", "mean"),
        mean_frozen_attempt_count=("frozen_attempt_count", "mean"),
        mean_dg_primary=("dg_primary", "mean"),
        mean_sortedness_curvature=("sortedness_path_curvature_ratio", "mean"),
        mean_value_state_curvature=("value_state_path_curvature_ratio", "mean"),
        max_guard_hit_rate=("max_guard_hit", "mean"),
        stop_reason_count=("stop_reason", "nunique"),
        frozen_identity_valid_rate=("frozen_identity_validation_passed", "mean"),
    ).reset_index()
    summary["sd_final_sortedness"] = summary["sd_final_sortedness"].fillna(0.0)
    reason_modes = (
        result_df.groupby(
            [
                "condition_id",
                "run_family",
                "algorithm",
                "algotype_mix",
                "frozen_semantics",
                "frozen_count",
                "stop_regime",
                "stop_regime_family",
            ],
            dropna=False,
        )["stop_reason"]
        .agg(lambda values: json.dumps(dict(sorted(pd.Series(values).value_counts().to_dict().items())), separators=(",", ":")))
        .reset_index(name="stop_reason_counts_json")
    )
    summary = summary.merge(
        reason_modes,
        on=[
            "condition_id",
            "run_family",
            "algorithm",
            "algotype_mix",
            "frozen_semantics",
            "frozen_count",
            "stop_regime",
            "stop_regime_family",
        ],
        how="left",
    )
    baseline = summary[summary["stop_regime"] == "reference_no_legal_2000"][
        [
            "condition_id",
            "success_rate",
            "mean_final_sortedness",
            "mean_final_monotonicity_error",
            "mean_compare_plus_swap_steps",
            "mean_event_count",
            "mean_dg_primary",
            "mean_sortedness_curvature",
            "mean_frozen_attempt_count",
            "max_guard_hit_rate",
            "stop_reason_counts_json",
        ]
    ].rename(
        columns={
            "success_rate": "reference_success_rate",
            "mean_final_sortedness": "reference_mean_final_sortedness",
            "mean_final_monotonicity_error": "reference_mean_final_monotonicity_error",
            "mean_compare_plus_swap_steps": "reference_mean_compare_plus_swap_steps",
            "mean_event_count": "reference_mean_event_count",
            "mean_dg_primary": "reference_mean_dg_primary",
            "mean_sortedness_curvature": "reference_mean_sortedness_curvature",
            "mean_frozen_attempt_count": "reference_mean_frozen_attempt_count",
            "max_guard_hit_rate": "reference_max_guard_hit_rate",
            "stop_reason_counts_json": "reference_stop_reason_counts_json",
        }
    )
    classified = summary.merge(baseline, on="condition_id", how="left")
    classified["delta_final_sortedness_vs_reference"] = classified["mean_final_sortedness"] - classified["reference_mean_final_sortedness"]
    classified["delta_success_rate_vs_reference"] = classified["success_rate"] - classified["reference_success_rate"]
    classified["delta_final_monotonicity_error_vs_reference"] = (
        classified["mean_final_monotonicity_error"] - classified["reference_mean_final_monotonicity_error"]
    )
    classified["compare_plus_swap_ratio_vs_reference"] = (
        classified["mean_compare_plus_swap_steps"] / classified["reference_mean_compare_plus_swap_steps"].replace(0, np.nan)
    )
    classified["event_count_ratio_vs_reference"] = (
        classified["mean_event_count"] / classified["reference_mean_event_count"].replace(0, np.nan)
    )
    classified["delta_dg_primary_vs_reference"] = classified["mean_dg_primary"] - classified["reference_mean_dg_primary"]
    classified["delta_sortedness_curvature_vs_reference"] = (
        classified["mean_sortedness_curvature"] - classified["reference_mean_sortedness_curvature"]
    )
    classified["delta_frozen_attempts_vs_reference"] = (
        classified["mean_frozen_attempt_count"] - classified["reference_mean_frozen_attempt_count"]
    )
    classified["delta_max_guard_hit_rate_vs_reference"] = (
        classified["max_guard_hit_rate"] - classified["reference_max_guard_hit_rate"]
    )
    classified["stop_reason_distribution_changed"] = (
        classified["stop_reason_counts_json"].astype(str) != classified["reference_stop_reason_counts_json"].astype(str)
    )

    classes: list[str] = []
    reasons: list[str] = []
    for row in classified.to_dict(orient="records"):
        if row["stop_regime"] == "reference_no_legal_2000":
            classes.append("reference_stop_rule")
            reasons.append("reference S11/S12 stop settings")
            continue
        final_sensitive = (
            float(row["delta_final_sortedness_vs_reference"]) < -5.0
            or float(row["delta_success_rate_vs_reference"]) < -0.1
            or float(row["delta_final_monotonicity_error_vs_reference"]) > 5.0
        )
        ratio = row["compare_plus_swap_ratio_vs_reference"]
        event_ratio = row["event_count_ratio_vs_reference"]
        efficiency_sensitive = pd.notna(ratio) and (float(ratio) > 2.0 or float(ratio) < 0.5)
        event_sensitive = pd.notna(event_ratio) and (float(event_ratio) > 2.0 or float(event_ratio) < 0.5)
        path_sensitive = (
            abs(float(row["delta_dg_primary_vs_reference"])) > 25.0
            or abs(float(row["delta_sortedness_curvature_vs_reference"])) > 0.75
        )
        reason_sensitive = bool(row["stop_reason_distribution_changed"]) or abs(float(row["delta_max_guard_hit_rate_vs_reference"])) > 0.1
        if final_sensitive:
            classes.append("stop_sensitive_final_state")
            reasons.append("final Sortedness, success, or monotonicity error changed materially")
        elif efficiency_sensitive or event_sensitive or path_sensitive:
            classes.append("stop_sensitive_efficiency_or_path")
            active_reasons = []
            if efficiency_sensitive:
                active_reasons.append("compare-plus-swap ratio outside [0.5, 2.0]")
            if event_sensitive:
                active_reasons.append("event-count ratio outside [0.5, 2.0]")
            if path_sensitive:
                active_reasons.append("DG delta exceeds 25 points or curvature delta exceeds 0.75")
            reasons.append("; ".join(active_reasons))
        elif reason_sensitive:
            classes.append("stop_sensitive_reason_or_cap")
            reasons.append("stop-reason distribution or max-guard rate changed")
        else:
            classes.append("stop_stable")
            reasons.append("final state, path, efficiency, and stop reasons stayed within thresholds")
    classified["stop_sensitivity_class"] = classes
    classified["classification_rule"] = reasons
    return summary, classified


def make_stop_condition_figure(summary_df: pd.DataFrame, classification_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    regime_order = [row["name"] for row in STOP_REGIMES if row["name"] in set(summary_df["stop_regime"])]
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
            "Final Sortedness delta vs reference",
            classification_df.pivot_table(index="condition_id", columns="stop_regime", values="delta_final_sortedness_vs_reference", aggfunc="mean").reindex(
                index=condition_order, columns=regime_order
            ),
            "coolwarm",
            "pp",
            "signed",
        ),
        (
            "log2 event-count ratio vs reference",
            np.log2(
                classification_df.pivot_table(index="condition_id", columns="stop_regime", values="event_count_ratio_vs_reference", aggfunc="mean").reindex(
                    index=condition_order, columns=regime_order
                )
            ),
            "PiYG_r",
            "log2 ratio",
            "ratio",
        ),
        (
            "DG primary delta vs reference",
            classification_df.pivot_table(index="condition_id", columns="stop_regime", values="delta_dg_primary_vs_reference", aggfunc="mean").reindex(
                index=condition_order, columns=regime_order
            ),
            "PRGn",
            "DG",
            "signed",
        ),
        (
            "Max-guard hit rate",
            summary_df.pivot_table(index="condition_id", columns="stop_regime", values="max_guard_hit_rate", aggfunc="mean").reindex(
                index=condition_order, columns=regime_order
            ),
            "magma",
            "rate",
            "rate",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    for axis, (title, pivot, cmap, label, value_kind) in zip(axes.flat, pivots, strict=True):
        data = pivot.to_numpy(dtype=float)
        finite = data[np.isfinite(data)]
        if value_kind in {"signed", "ratio"}:
            vmax = max(1.0, float(np.nanmax(np.abs(finite))) if finite.size else 1.0)
            vmin = -vmax
        else:
            vmin = 0.0
            vmax = max(1.0, float(np.nanmax(finite)) if finite.size else 1.0)
        im = axis.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks(range(len(regime_order)))
        axis.set_xticklabels([name.replace("_", "\n") for name in regime_order], fontsize=6.5)
        axis.set_yticks(range(len(condition_order)))
        axis.set_yticklabels(condition_labels if axis in (axes[0, 0], axes[1, 0]) else [], fontsize=8)
        for i in range(len(condition_order)):
            for j in range(len(regime_order)):
                value = pivot.iloc[i, j]
                if pd.notna(value):
                    if value_kind == "ratio":
                        text = f"{2.0 ** float(value):.1f}x"
                    elif value_kind == "rate":
                        text = f"{float(value):.1f}"
                    else:
                        text = f"{float(value):+.1f}"
                    axis.text(j, i, text, ha="center", va="center", fontsize=5.8, color="white" if value_kind == "rate" else "black")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04).set_label(label)
    fig.suptitle("E02 S13 stop-condition sensitivity", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def stop_validation_rows(result_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "research_step_id",
        "condition_id",
        "algorithm",
        "frozen_semantics",
        "frozen_count",
        "repeat_index",
        "stop_regime",
        "stop_regime_family",
        "stop_reason",
        "stop_reason_logged",
        "max_events_config",
        "max_successful_swaps_config",
        "stall_events_config",
        "stop_when_sorted_config",
        "stop_sortedness_threshold_config",
        "convergence_criterion_config",
        "stop_config_json",
        "stop_config_sha256",
        "event_count",
        "swap_only_steps",
        "max_guard_hit",
        "final_sortedness_percent",
        "frozen_identity_validation_passed",
    ]
    return result_df[columns].sort_values(["condition_id", "repeat_index", "stop_regime"]).reset_index(drop=True)


def run_seed_and_determinism_checks(tasks: Sequence[StopTask], result_df: pd.DataFrame) -> dict[str, Any]:
    sample_tasks: list[StopTask] = []
    for regime in [row["name"] for row in STOP_REGIMES]:
        candidates = [task for task in tasks if task.stop_regime["name"] == regime]
        if candidates:
            sample_tasks.append(candidates[0])
    deterministic_ok = True
    checks: list[dict[str, Any]] = []
    index = result_df.set_index(["condition_id", "stop_regime", "repeat_index"])
    for task in sample_tasks:
        rerun = run_stop_task(task)
        key = (task.condition["condition_id"], task.stop_regime["name"], task.repeat_index)
        original = index.loc[key]
        passed = bool(
            str(rerun["final_array_sha256"]) == str(original["final_array_sha256"])
            and str(rerun["trace_values_sha256"]) == str(original["trace_values_sha256"])
            and str(rerun["activation_log_sha256"]) == str(original["activation_log_sha256"])
            and str(rerun["stop_reason"]) == str(original["stop_reason"])
            and int(rerun["event_count"]) == int(original["event_count"])
        )
        deterministic_ok = deterministic_ok and passed
        checks.append(
            {
                "condition_id": task.condition["condition_id"],
                "repeat_index": int(task.repeat_index),
                "stop_regime": task.stop_regime["name"],
                "passed": passed,
                "stop_reason": rerun["stop_reason"],
                "final_array_sha256": rerun["final_array_sha256"],
                "trace_values_sha256": rerun["trace_values_sha256"],
            }
        )
    return {
        "simulationDeterminismSamplePassed": bool(deterministic_ok),
        "simulationDeterminismSampleCount": int(len(checks)),
        "simulationDeterminismChecks": checks,
    }


def validate_outputs(
    *,
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    stop_validation_df: pd.DataFrame,
    tasks: Sequence[StopTask],
    figure_path: Path,
    result_path: Path,
    dg_toy_validation: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {(task.condition["condition_id"], task.stop_regime["name"], task.repeat_index) for task in tasks}
    observed_keys = set(zip(result_df["condition_id"], result_df["stop_regime"], result_df["repeat_index"]))
    expected_regimes = [row["name"] for row in STOP_REGIMES if any(task.stop_regime["name"] == row["name"] for task in tasks)]
    observed_regimes = [name for name in expected_regimes if name in set(result_df["stop_regime"])]
    required_families = {"reference", "no_movement_window", "max_step_cap", "sortedness_threshold", "convergence_criterion"}
    observed_families = set(result_df["stop_regime_family"].unique())
    classes = set(classification_df["stop_sensitivity_class"].dropna().unique())
    sensitive_count = int(classification_df["stop_sensitivity_class"].str.startswith("stop_sensitive").sum())
    outcome = "constraining/contradictory" if sensitive_count > 0 else "supportive"
    stop_reasons = sorted(result_df["stop_reason"].dropna().unique().tolist())
    success = bool(
        expected_keys == observed_keys
        and required_families <= observed_families
        and bool(result_df["stop_reason_logged"].all())
        and not bool(result_df["stop_config_json"].isna().any())
        and not bool(result_df["stop_config_sha256"].isna().any())
        and bool(result_df["max_events_config"].notna().all())
        and bool(result_df["stall_events_config"].notna().all())
        and bool(result_df["convergence_criterion_config"].notna().all())
        and bool(result_df["frozen_identity_validation_passed"].all())
        and bool(dg_toy_validation["toyCaseValidationPassed"])
        and bool(reproducibility["simulationDeterminismSamplePassed"])
        and result_path.exists()
        and figure_path.exists()
        and figure_path.stat().st_size > 0
        and classes <= {
            "reference_stop_rule",
            "stop_stable",
            "stop_sensitive_reason_or_cap",
            "stop_sensitive_efficiency_or_path",
            "stop_sensitive_final_state",
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
            "Matrix is bounded to first 10 repeats and selected pure cell-view unperturbed plus frozen_count 3 conditions.",
            "Runs use the S01 deterministic public-move simulator, not OS-thread replays.",
            "S13 uses S11/S12 caveats: unique random-permutation inputs, fixed E01 random-bank Frozen Cell placement, and no dynamic Frozen Cell behavior variants in the primary stop-condition matrix.",
            "Comparison counts remain the public StatusProbe actionable-comparison proxy.",
        ],
        "recommendedNextAction": "Review S13 stop-condition sensitivity, then proceed to S14 only after explicit Chief Scientist instruction.",
        "expectedRunCount": int(len(expected_keys)),
        "observedRunCount": int(len(observed_keys)),
        "resultRowCount": int(len(result_df)),
        "summaryRowCount": int(len(summary_df)),
        "classificationRowCount": int(len(classification_df)),
        "stopValidationRowCount": int(len(stop_validation_df)),
        "conditionIds": sorted({task.condition["condition_id"] for task in tasks}),
        "stopRegimes": expected_regimes,
        "observedStopRegimes": observed_regimes,
        "stopRegimeFamilies": sorted(observed_families),
        "repeatIndexes": sorted({int(task.repeat_index) for task in tasks}),
        "allExpectedRunsPresent": bool(expected_keys == observed_keys),
        "stopReasonLoggedEveryRun": bool(result_df["stop_reason_logged"].all()),
        "capsAndWindowsEncodedEveryRun": bool(
            result_df["max_events_config"].notna().all()
            and result_df["stall_events_config"].notna().all()
            and result_df["convergence_criterion_config"].notna().all()
            and result_df["stop_config_sha256"].notna().all()
        ),
        "frozenIdentityValidEveryRun": bool(result_df["frozen_identity_validation_passed"].all()),
        "dgToyValidationPassed": bool(dg_toy_validation["toyCaseValidationPassed"]),
        "classificationClasses": sorted(classes),
        "stopSensitivityClassCounts": classification_df["stop_sensitivity_class"].value_counts().sort_index().to_dict(),
        "sensitiveSliceCount": sensitive_count,
        "stopReasonsObserved": stop_reasons,
        "stopReasonCounts": result_df["stop_reason"].value_counts().sort_index().to_dict(),
        "figureNonempty": bool(figure_path.exists() and figure_path.stat().st_size > 0),
        "plannedMatrixWritten": bool(result_path.exists()),
        **dict(reproducibility),
    }
    validation["validationResult"] = (
        "passed: expected stop-condition matrix, stop reasons, caps/windows config encoding, Frozen Cell identities, DG toy cases, deterministic rerun samples, and figure validated"
        if success
        else "failed: one or more S13 validation checks did not pass"
    )
    return validation


def build_report(
    *,
    generated_at: str,
    result_path: Path,
    summary_path: Path,
    classification_path: Path,
    stop_validation_path: Path,
    figure_path: Path,
    validation_path: Path,
    status_path: Path,
    dg_toy_path: Path,
    log_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    summary_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    stop_validation_df: pd.DataFrame,
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
    class_counts = classification_df["stop_sensitivity_class"].value_counts().sort_index().to_dict()
    largest_final = classification_df[classification_df["stop_regime"] != "reference_no_legal_2000"].copy()
    largest_final = largest_final.sort_values("delta_final_sortedness_vs_reference").head(18)[
        [
            "condition_id",
            "algorithm",
            "frozen_semantics",
            "frozen_count",
            "stop_regime",
            "stop_sensitivity_class",
            "mean_final_sortedness",
            "delta_final_sortedness_vs_reference",
            "mean_final_monotonicity_error",
            "event_count_ratio_vs_reference",
            "compare_plus_swap_ratio_vs_reference",
            "delta_dg_primary_vs_reference",
            "stop_reason_counts_json",
        ]
    ]
    stop_excerpt = stop_validation_df[
        [
            "condition_id",
            "repeat_index",
            "stop_regime",
            "stop_reason",
            "max_events_config",
            "max_successful_swaps_config",
            "stall_events_config",
            "stop_sortedness_threshold_config",
            "convergence_criterion_config",
            "event_count",
            "final_sortedness_percent",
        ]
    ].head(24)
    return f"""# E02 S13 Full Results: Stop-Condition Sensitivity

## Top Summary

- Step ID: S13
- Completion status: {validation['status']}
- Artifacts written: {', '.join(validation['artifactsWritten'])}
- Validation result: {validation['validationResult']}
- Outcome classification: {validation['outcomeClassification']}
- Caveats or blockers: {', '.join(validation['caveatsOrBlockers'])}
- Lay summary: S13 changed when the deterministic simulator stops rather than changing the local cell policies. Stop settings materially changed stop reasons, efficiency/path metrics, and in some capped or thresholded cases final-state metrics, so robustness and DG summaries remain conditional on the chosen stopping rule.
- Recommended next action: {validation['recommendedNextAction']}

## Frozen Question

Are failures, equilibria, or robustness claims sensitive to no-movement windows, maximum-step caps, Sortedness thresholds, and convergence criteria?

## Inputs

- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- S01 deterministic simulator with S13 stop-condition controls: `src/e02/deterministic_simulator.py`
- S11/S12 caveats: fixed E01 random-bank Frozen Cell placement, unique random-permutation inputs, deterministic public-move simulation, and bounded first-10-repeat matrix.
- S08 DG recomputation code: `scripts/e02_s08_dg_nulls.py`
- Selected condition IDs: `{parameters['conditionIds']}`
- Stop regimes: `{parameters['stopRegimes']}`
- Repeat indexes: `{parameters['repeatIndexes']}`

No external datasets were required and no new dependencies were installed.

## Methods

S13 selected pure cell-view unperturbed controls plus passive/stuck Frozen Cell conditions with three Frozen Cells. For each condition/repeat, it held the initial value array, Frozen Cell bank, activation seed, and public cell policy fixed while varying only the stop-condition configuration:

- no-movement windows: 250, 2,000, and 8,000 event windows under no-legal-action convergence.
- max-step caps: 25,000 event cap and 5,000 successful-swap cap.
- Sortedness thresholds: early stop at 95 percent or 99 percent adjacent Sortedness.
- convergence criteria: no legal action, no state change, no swap, Sortedness plateau, and no convergence window.

The simulator logs a `stop_reason` for every run and the result table records `max_events`, `max_successful_swaps`, `stall_events`, `stop_when_sorted`, `stop_sortedness_threshold`, and `convergence_criterion` both as columns and as a hashed JSON config.

## Commands

- `python scripts/e02_s13_stop_conditions.py --repo-dir /workspace/cell-research --artifacts-dir /artifacts --max-repeats {parameters['maxRepeats']} --workers {parameters['workers']} --run-unit-tests`
- Unit-test command inside the script: `{unit_line}`

## Results

- Matrix rows: {validation['resultRowCount']}
- Summary rows: {validation['summaryRowCount']}
- Classification rows: {validation['classificationRowCount']}
- Stop-validation rows: {validation['stopValidationRowCount']}
- Stop sensitivity class counts: {json.dumps(class_counts, sort_keys=True)}
- Stop reasons observed: {json.dumps(validation['stopReasonsObserved'])}
- Stop reason counts: {json.dumps(validation['stopReasonCounts'], sort_keys=True)}
- Expected runs present: {validation['allExpectedRunsPresent']}
- Stop reason logged every run: {validation['stopReasonLoggedEveryRun']}
- Caps/windows encoded every run: {validation['capsAndWindowsEncodedEveryRun']}

### Largest Final Sortedness Drops

{markdown_table(largest_final, max_rows=18)}

### Stop Config Validation Excerpt

{markdown_table(stop_excerpt, max_rows=24)}

## Validation

- All expected condition/stop-regime/repeat runs present: {validation['allExpectedRunsPresent']}
- Stop reason logged for every run: {validation['stopReasonLoggedEveryRun']}
- Caps/windows encoded for every run: {validation['capsAndWindowsEncodedEveryRun']}
- Frozen identities validated in every Frozen Cell run: {validation['frozenIdentityValidEveryRun']}
- S08 DG toy-case validation passed: {validation['dgToyValidationPassed']}
- Simulation determinism sample passed: {validation['simulationDeterminismSamplePassed']} across {validation['simulationDeterminismSampleCount']} sampled reruns.
- Planned matrix written: {validation['plannedMatrixWritten']}
- Figure nonempty: {validation['figureNonempty']}
- Unit tests: {unit_line}

Validation JSON: `{validation_path}`  
Status JSON: `{status_path}`  
DG toy validation JSON: `{dg_toy_path}`

## Artifacts

- Result matrix: `{result_path}`
- Summary table: `{summary_path}`
- Classification table: `{classification_path}`
- Stop validation table: `{stop_validation_path}`
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
- S13 uses selected pure cell-view conditions, not chimeric or opposite-direction cases.
- The matrix fixes Frozen Cell placement and behavior; S11 and S12 show those perturbation assumptions matter separately.
- The simulator preserves public local cell `move()` methods but replaces OS/Python thread scheduling with explicit deterministic activation.
- Comparison counts remain the public `StatusProbe` actionable-comparison proxy inherited from E01/S01.
- Early Sortedness thresholds are intentionally looser than exact sorting and should be interpreted as sensitivity probes, not successful exact-sort completion.

## Recommended Next Action

Proceed to S14 stronger statistics only after explicit Chief Scientist instruction. S14 should carry S13's stop-condition dependence into claim-level models and corrected comparisons.
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

    log_path = log_dir / "e02_s13_stop_conditions.log"
    result_path = result_dir / "e02_stop_condition_sensitivity.parquet"
    figure_path = figure_dir / "stop_condition_sensitivity.png"
    summary_path = step_dir / "e02_stop_condition_summary.csv"
    classification_path = step_dir / "e02_stop_condition_classification.csv"
    stop_validation_path = step_dir / "e02_stop_condition_validation.csv"
    validation_path = step_dir / "s13_validation.json"
    status_path = step_dir / "status.json"
    dg_toy_path = step_dir / "e02_s13_dg_toy_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s13_stop_conditions_manifest.json"
    report_path = step_dir / "research_step_full_results.md"

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"{generated_at} Starting E02 S13 stop-condition sensitivity\n")
        log.write(f"repo_dir={repo_dir}\nartifacts_dir={artifacts_dir}\n")

    cfg = load_e01_config(args.config_path)
    rows = selected_conditions(cfg, args.condition_ids)
    stop_regimes = selected_stop_regimes(args.stop_regimes)
    tasks = build_tasks(cfg, rows, stop_regimes, repo_dir, args.max_repeats, args.activation_distribution)
    parameters = {
        "maxRepeats": int(args.max_repeats),
        "workers": int(args.workers),
        "activationDistribution": args.activation_distribution,
        "conditionIds": [row["condition_id"] for row in rows],
        "stopRegimes": [row["name"] for row in stop_regimes],
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
                "tests.e02.test_stop_conditions",
            ],
            repo_dir,
        )
        if not unit_result["success"]:
            write_json(step_dir / "s13_unit_test_failure.json", unit_result)
            raise RuntimeError(f"S13 unit tests failed; see {step_dir / 's13_unit_test_failure.json'}")

    rows_out: list[dict[str, Any]] = []
    if args.workers <= 1:
        for task in tasks:
            rows_out.append(run_stop_task(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            future_to_task = {executor.submit(run_stop_task, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    rows_out.append(future.result())
                except Exception as exc:
                    raise RuntimeError(
                        f"S13 task failed for {task.condition['condition_id']} repeat={task.repeat_index} stop_regime={task.stop_regime['name']}"
                    ) from exc

    result_df = pd.DataFrame(rows_out).sort_values(["condition_id", "repeat_index", "stop_regime"]).reset_index(drop=True)
    result_df.to_parquet(result_path, index=False)
    summary_df, classification_df = summarize_results(result_df)
    stop_validation_df = stop_validation_rows(result_df)
    summary_df.to_csv(summary_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    stop_validation_df.to_csv(stop_validation_path, index=False)
    make_stop_condition_figure(summary_df, classification_df, figure_path)

    dg_toy_validation = validate_toy_cases()
    write_json(dg_toy_path, dg_toy_validation)
    reproducibility = run_seed_and_determinism_checks(tasks, result_df)
    validation = validate_outputs(
        result_df=result_df,
        summary_df=summary_df,
        classification_df=classification_df,
        stop_validation_df=stop_validation_df,
        tasks=tasks,
        figure_path=figure_path,
        result_path=result_path,
        dg_toy_validation=dg_toy_validation,
        reproducibility=reproducibility,
    )
    artifact_paths = [
        (result_path, "planned S13 stop-condition sensitivity matrix"),
        (figure_path, "planned S13 stop-condition sensitivity figure"),
        (summary_path, "condition-by-stop-regime summary metrics"),
        (classification_path, "stop sensitivity classification table"),
        (stop_validation_path, "stop-reason and stop-config validation table"),
        (validation_path, "S13 validation JSON"),
        (status_path, "S13 compact status JSON"),
        (dg_toy_path, "S08 DG formula toy validation reused by S13"),
        (log_path, "S13 execution log"),
        (manifest_path, "S13 artifact manifest"),
        (src_manifest_path, "S13 source snapshot manifest"),
        (report_path, "S13 full-results report"),
    ]
    validation["artifactsWritten"] = [str(path) for path, _description in artifact_paths]
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
            artifact_entry(repo_dir / "src/e02/deterministic_simulator.py", repo_dir, "S01 simulator extended with S13 stop-condition controls"),
            artifact_entry(repo_dir / "scripts/e02_s13_stop_conditions.py", repo_dir, "S13 stop-condition sensitivity runner"),
            artifact_entry(repo_dir / "tests/e02/test_deterministic_simulator.py", repo_dir, "simulator tests including S13 stop-mode checks"),
            artifact_entry(repo_dir / "tests/e02/test_stop_conditions.py", repo_dir, "focused S13 stop-condition tests"),
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

    elapsed = time.monotonic() - started
    report = build_report(
        generated_at=generated_at,
        result_path=result_path,
        summary_path=summary_path,
        classification_path=classification_path,
        stop_validation_path=stop_validation_path,
        figure_path=figure_path,
        validation_path=validation_path,
        status_path=status_path,
        dg_toy_path=dg_toy_path,
        log_path=log_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        summary_df=summary_df,
        classification_df=classification_df,
        stop_validation_df=stop_validation_df,
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
        log.write(f"{utc_now()} Completed S13 status={validation['status']} success={validation['success']} elapsed={elapsed:.2f}s\n")
        log.write(json.dumps({"validation": validation["validationResult"], "artifacts": validation["artifactsWritten"]}, indent=2) + "\n")
    if not validation["success"]:
        raise RuntimeError(f"S13 validation failed; see {validation_path}")


if __name__ == "__main__":
    main()
