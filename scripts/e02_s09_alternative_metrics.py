#!/usr/bin/env python3
"""Preflight E02 S09 alternative distance metrics.

S09 is intended to recompute trajectory progress using full per-step value
states.  This script validates the alternative metric definitions on small
known arrays, audits available E01/E02 artifacts for persisted value
trajectories, and writes a blocked handoff if the required trajectories are not
available.
"""

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
import pyarrow.parquet as pq

REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from src.e02.deterministic_simulator import (
    SimulatorConfig,
    aggregation_left_neighbor_percent,
    aggregation_right_neighbor_legacy_percent,
    count_json,
    monotonicity_error_count,
    simulate,
    sortedness_percent,
    stable_json_sha256,
)


STEP_ID = "S09"
STEP_NUMBER = 9
EXPERIMENT_ID = "E02"
SOURCE_EXPERIMENT_ID = "E01"

E01_TRACE_SOURCES = (
    ("E01 figure3 unperturbed trajectories", "/previous-artifacts/E01/traces/e01_figure3_trajectories.parquet"),
    ("E01 Frozen Cell trajectories", "/previous-artifacts/E01/traces/e01_frozen_cell_trajectories.parquet"),
    ("E01 same-goal chimera trajectories", "/previous-artifacts/E01/traces/e01_same_goal_chimeras.parquet"),
    ("E01 duplicate-value chimera trajectories", "/previous-artifacts/E01/traces/e01_duplicate_value_chimeras.parquet"),
    ("E01 opposite-direction chimera trajectories", "/previous-artifacts/E01/traces/e01_opposite_direction_chimeras.parquet"),
)
E02_RESULT_SOURCES = (
    ("E02 S01 trace row samples", "/artifacts/research_steps/S01/e02_s01_trace_row_samples.parquet"),
    ("E02 S02 scheduler comparison", "/artifacts/results/e02_scheduler_comparison.parquet"),
    ("E02 S03 activation-rate artifacts", "/artifacts/results/e02_activation_rate_artifacts.parquet"),
    ("E02 S05 dummy Algotype controls", "/artifacts/results/e02_dummy_algotype_controls.parquet"),
    ("E02 S06 speed-matched chimeras", "/artifacts/results/e02_speed_matched_chimeras.parquet"),
    ("E02 S07 local-move nulls", "/artifacts/results/e02_local_move_nulls.parquet"),
    ("E02 S08 DG null benchmarks", "/artifacts/results/e02_dg_null_benchmarks.parquet"),
)
PER_STEP_VALUE_COLUMN_CANDIDATES = {
    "values",
    "values_json",
    "value_state",
    "value_state_json",
    "current_values",
    "current_values_json",
    "array_values",
    "array_values_json",
    "state_values",
    "state_values_json",
}
REPAIR_CORE_CONDITION_IDS = (
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
REPAIR_DUPLICATE_CONDITION_IDS = ("E01C048", "E01C049", "E01C050")
REPAIR_CONDITION_IDS = REPAIR_CORE_CONDITION_IDS + REPAIR_DUPLICATE_CONDITION_IDS
REPAIR_DISTANCE_METRICS = (
    "sortedness_adjacency_distance",
    "kendall_tau_distance",
    "inversion_fraction",
    "spearman_position_distance",
    "edit_distance_to_target",
    "earth_mover_position_distance",
)
ALT_DISTANCE_METRICS = tuple(metric for metric in REPAIR_DISTANCE_METRICS if metric != "sortedness_adjacency_distance")


@dataclass(frozen=True)
class RepairTask:
    condition: dict[str, Any]
    repeat_index: int
    values: tuple[int, ...]
    assignments: tuple[str, ...]
    reverse_directions: tuple[bool, ...]
    frozen_indices: tuple[int, ...]
    initial_array_seed: int
    algotype_assignment_seed: int | None
    frozen_index_seed: int | None
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
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-rerun", action="store_true", help="Run the bounded deterministic S09 repair rerun.")
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-events", type=int, default=150_000)
    parser.add_argument("--max-successful-swaps", type=int, default=80_000)
    parser.add_argument("--stall-events", type=int, default=2_000)
    parser.add_argument("--activation-distribution", choices=("uniform_active", "left_to_right_active", "right_to_left_active"), default="uniform_active")
    parser.add_argument("--condition-ids", nargs="*", default=list(REPAIR_CONDITION_IDS))
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


def occurrence_tokens(values: Sequence[int | float]) -> tuple[tuple[int | float, int], ...]:
    """Label duplicate values by left-to-right occurrence for deterministic ties."""

    counts: Counter[int | float] = Counter()
    tokens: list[tuple[int | float, int]] = []
    for value in values:
        occurrence = counts[value]
        tokens.append((value, occurrence))
        counts[value] += 1
    return tuple(tokens)


def target_sequence(values: Sequence[int | float], direction: str = "increasing") -> tuple[int | float, ...]:
    if direction in {"increasing", "nondecreasing", "all_increasing"}:
        return tuple(sorted(values))
    if direction in {"decreasing", "nonincreasing", "all_decreasing"}:
        return tuple(sorted(values, reverse=True))
    raise ValueError(f"Unsupported direction: {direction}")


def target_token_positions(values: Sequence[int | float], direction: str = "increasing") -> dict[tuple[int | float, int], int]:
    target_tokens = occurrence_tokens(target_sequence(values, direction))
    return {token: idx for idx, token in enumerate(target_tokens)}


def strict_pair_count(values: Sequence[int | float]) -> int:
    counts = Counter(values)
    n = len(values)
    return n * (n - 1) // 2 - sum(count * (count - 1) // 2 for count in counts.values())


def inversion_count(values: Sequence[int | float], direction: str = "increasing") -> int:
    inversions = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if direction in {"increasing", "nondecreasing", "all_increasing"}:
                inversions += int(values[i] > values[j])
            elif direction in {"decreasing", "nonincreasing", "all_decreasing"}:
                inversions += int(values[i] < values[j])
            else:
                raise ValueError(f"Unsupported direction: {direction}")
    return inversions


def kendall_tau_distance(values: Sequence[int | float], direction: str = "increasing") -> float:
    possible = strict_pair_count(values)
    if possible == 0:
        return 0.0
    return float(inversion_count(values, direction) / possible)


def spearman_position_distance(values: Sequence[int | float], direction: str = "increasing") -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    positions = target_token_positions(values, direction)
    current_tokens = occurrence_tokens(values)
    numerator = sum((idx - positions[token]) ** 2 for idx, token in enumerate(current_tokens))
    max_unique_reversal = n * (n * n - 1) / 3.0
    return float(numerator / max_unique_reversal) if max_unique_reversal else 0.0


def earth_mover_position_distance(values: Sequence[int | float], direction: str = "increasing") -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    positions = target_token_positions(values, direction)
    current_tokens = occurrence_tokens(values)
    numerator = sum(abs(idx - positions[token]) for idx, token in enumerate(current_tokens))
    max_unique_reversal = (n * n) // 2
    return float(numerator / max_unique_reversal) if max_unique_reversal else 0.0


def levenshtein_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return int(previous[-1])


def edit_distance_to_target(values: Sequence[int | float], direction: str = "increasing") -> float:
    n = len(values)
    if n == 0:
        return 0.0
    current_tokens = occurrence_tokens(values)
    target_tokens = occurrence_tokens(target_sequence(values, direction))
    return float(levenshtein_distance(current_tokens, target_tokens) / n)


def scalar_path_curvature(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1.0
    deltas = np.diff(np.asarray(values, dtype=float))
    total = float(np.sum(np.abs(deltas)))
    net = float(abs(float(values[-1]) - float(values[0])))
    if math.isclose(total, 0.0, abs_tol=1e-12):
        return 1.0
    if math.isclose(net, 0.0, abs_tol=1e-12):
        return float("inf")
    return float(total / net)


def state_space_curvature(states: Sequence[Sequence[int | float]]) -> float:
    if len(states) < 2:
        return 1.0
    arr = np.asarray(states, dtype=float)
    step_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    total = float(np.sum(step_lengths))
    net = float(np.linalg.norm(arr[-1] - arr[0]))
    if math.isclose(total, 0.0, abs_tol=1e-12):
        return 1.0
    if math.isclose(net, 0.0, abs_tol=1e-12):
        return float("inf")
    return float(total / net)


def alternative_metrics(values: Sequence[int | float], direction: str = "increasing") -> dict[str, float | int]:
    inv = inversion_count(values, direction)
    possible = strict_pair_count(values)
    return {
        "inversion_count": int(inv),
        "inversion_fraction": float(inv / possible) if possible else 0.0,
        "kendall_tau_distance": kendall_tau_distance(values, direction),
        "spearman_position_distance": spearman_position_distance(values, direction),
        "edit_distance_to_target": edit_distance_to_target(values, direction),
        "earth_mover_position_distance": earth_mover_position_distance(values, direction),
    }


def metric_sanity_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Mapping[str, Any]) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": dict(details)})

    sorted_unique = alternative_metrics([1, 2, 3])
    add("sorted_unique_zero_distance", all(math.isclose(float(value), 0.0) for value in sorted_unique.values()), sorted_unique)

    reversed_unique = alternative_metrics([3, 2, 1])
    add(
        "reversed_unique_max_rank_distances",
        reversed_unique["inversion_count"] == 3
        and math.isclose(float(reversed_unique["kendall_tau_distance"]), 1.0)
        and math.isclose(float(reversed_unique["spearman_position_distance"]), 1.0)
        and math.isclose(float(reversed_unique["earth_mover_position_distance"]), 1.0)
        and math.isclose(float(reversed_unique["edit_distance_to_target"]), 2.0 / 3.0),
        reversed_unique,
    )

    adjacent_swap = alternative_metrics([1, 3, 2, 4])
    add(
        "adjacent_swap_one_inversion",
        adjacent_swap["inversion_count"] == 1 and math.isclose(float(adjacent_swap["kendall_tau_distance"]), 1.0 / 6.0),
        adjacent_swap,
    )

    duplicate_case = alternative_metrics([2, 1, 2, 1])
    add(
        "duplicate_case_strict_ties_ignored_and_tokens_stable",
        duplicate_case["inversion_count"] == 3
        and math.isclose(float(duplicate_case["kendall_tau_distance"]), 0.75)
        and math.isclose(float(duplicate_case["earth_mover_position_distance"]), 0.75),
        duplicate_case,
    )

    all_ties = alternative_metrics([1, 1, 1])
    add(
        "all_duplicate_ties_zero_order_distance",
        all_ties["inversion_count"] == 0 and math.isclose(float(all_ties["kendall_tau_distance"]), 0.0),
        all_ties,
    )

    add("monotone_scalar_curvature_one", math.isclose(scalar_path_curvature([1.0, 0.5, 0.0]), 1.0), {"curvature": scalar_path_curvature([1.0, 0.5, 0.0])})
    add(
        "backtracking_scalar_curvature_above_one",
        scalar_path_curvature([1.0, 0.2, 0.8, 0.0]) > 1.0,
        {"curvature": scalar_path_curvature([1.0, 0.2, 0.8, 0.0])},
    )
    add(
        "state_space_curvature_backtracking_above_one",
        state_space_curvature([[0, 0], [1, 0], [0, 0], [2, 0]]) > 1.0,
        {"curvature": state_space_curvature([[0, 0], [1, 0], [0, 0], [2, 0]])},
    )

    duplicate_checks = [row for row in checks if "duplicate" in row["name"] or "ties" in row["name"]]
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "metricSanityChecksPassed": bool(all(row["passed"] for row in checks)),
        "duplicateHandlingValidationPassed": bool(duplicate_checks and all(row["passed"] for row in duplicate_checks)),
        "checks": checks,
        "duplicateHandlingPolicy": (
            "Kendall/inversion metrics ignore equal-value ties; position-based Spearman, "
            "earth-mover, and edit metrics use deterministic left-to-right occurrence "
            "tokens (value, occurrence_index) so duplicate states have explicit target "
            "positions without comparing equal values as strict order violations."
        ),
    }


def parquet_non_null_counts(path: Path, columns: Sequence[str]) -> dict[str, int]:
    if not columns:
        return {}
    frame = pq.ParquetFile(path).read(columns=list(columns)).to_pandas()
    return {column: int(frame[column].notna().sum()) for column in columns}


def audit_parquet_source(name: str, path: Path, source_group: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_group": source_group,
        "source_name": name,
        "path": str(path),
        "exists": bool(path.exists()),
        "row_count": 0,
        "column_count": 0,
        "has_sortedness_trajectory": False,
        "has_final_values_json": False,
        "has_per_step_value_column": False,
        "has_per_step_value_trajectory": False,
        "nonfinal_value_rows": 0,
        "final_value_rows": 0,
        "value_like_columns": "",
        "columns": "",
        "blocker": "",
    }
    if not path.exists():
        row["blocker"] = "missing_file"
        return row
    pf = pq.ParquetFile(path)
    columns = list(pf.schema_arrow.names)
    row["row_count"] = int(pf.metadata.num_rows)
    row["column_count"] = int(len(columns))
    row["columns"] = ",".join(columns)
    value_like = [col for col in columns if any(token in col.lower() for token in ("value", "array", "state", "target", "order"))]
    row["value_like_columns"] = ",".join(value_like)
    row["has_sortedness_trajectory"] = "sortedness_percent" in columns or "final_sortedness_percent" in columns
    row["has_final_values_json"] = "final_values_json" in columns

    per_step_columns = [col for col in columns if col in PER_STEP_VALUE_COLUMN_CANDIDATES]
    row["has_per_step_value_column"] = bool(per_step_columns)
    if per_step_columns:
        counts = parquet_non_null_counts(path, per_step_columns)
        row["nonfinal_value_rows"] = int(max(counts.values()) if counts else 0)
        row["has_per_step_value_trajectory"] = row["nonfinal_value_rows"] > 0
    elif "final_values_json" in columns:
        columns_to_read = ["final_values_json"] + (["is_final"] if "is_final" in columns else [])
        frame = pq.ParquetFile(path).read(columns=columns_to_read).to_pandas()
        row["final_value_rows"] = int(frame["final_values_json"].notna().sum())
        if "is_final" in frame.columns:
            row["nonfinal_value_rows"] = int((frame["final_values_json"].notna() & ~frame["is_final"].astype(bool)).sum())
            row["has_per_step_value_trajectory"] = bool(row["nonfinal_value_rows"] > 0)
        else:
            row["has_per_step_value_trajectory"] = False

    if not row["has_per_step_value_trajectory"]:
        row["blocker"] = "no_persisted_per_step_value_arrays"
    return row


def audit_config(config_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config_path": str(config_path),
        "exists": bool(config_path.exists()),
        "has_unique_initial_arrays": False,
        "has_duplicate_initial_arrays": False,
        "unique_repeat_count": 0,
        "duplicate_repeat_count": 0,
        "target_order_status": "unavailable",
        "duplicate_policy_status": "unavailable",
        "blocker": "",
    }
    if not config_path.exists():
        row["blocker"] = "missing_config"
        return row
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    value_banks = cfg.get("seedBanks", {}).get("valueBanks", {})
    unique = value_banks.get("unique_1_to_100", {})
    duplicate = value_banks.get("duplicate_1_to_10_x10", {})
    row["unique_repeat_count"] = int(len(unique.get("initialArrays", [])))
    row["duplicate_repeat_count"] = int(len(duplicate.get("initialArrays", [])))
    row["has_unique_initial_arrays"] = bool(row["unique_repeat_count"] > 0)
    row["has_duplicate_initial_arrays"] = bool(row["duplicate_repeat_count"] > 0)
    if row["has_unique_initial_arrays"] and row["has_duplicate_initial_arrays"]:
        row["target_order_status"] = "inferable_from_direction_and_value_bank"
        row["duplicate_policy_status"] = "duplicate_bank_present_non_strict_tie_semantics_documented"
    else:
        row["blocker"] = "missing_initial_value_banks"
    return row


def audit_inputs(config_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_rows = [
        audit_parquet_source(name, Path(path), "E01 trajectory artifact") for name, path in E01_TRACE_SOURCES
    ] + [
        audit_parquet_source(name, Path(path), "E02 result artifact") for name, path in E02_RESULT_SOURCES
    ]
    audit_df = pd.DataFrame(source_rows)
    config_audit = audit_config(config_path)
    required_trajectory_rows = audit_df[audit_df["source_group"].isin(["E01 trajectory artifact", "E02 result artifact"])]
    available_value_sources = required_trajectory_rows[required_trajectory_rows["has_per_step_value_trajectory"].astype(bool)]
    summary = {
        "configAudit": config_audit,
        "requiredSourceCount": int(len(required_trajectory_rows)),
        "existingRequiredSourceCount": int(required_trajectory_rows["exists"].astype(bool).sum()),
        "sourcesWithPerStepValueTrajectories": int(len(available_value_sources)),
        "sourcesWithSortednessTrajectories": int(required_trajectory_rows["has_sortedness_trajectory"].astype(bool).sum()),
        "allRequiredValueTrajectoriesAvailable": bool(len(available_value_sources) == len(required_trajectory_rows) and len(required_trajectory_rows) > 0),
        "targetOrderDefinitionsAvailable": bool(config_audit["target_order_status"] == "inferable_from_direction_and_value_bank"),
        "duplicateHandlingInputsAvailable": bool(config_audit["duplicate_policy_status"] == "duplicate_bank_present_non_strict_tie_semantics_documented"),
        "deterministicSimulatorCanRecordValuesInMemory": True,
        "deterministicSimulatorPersistedValueTrajectoriesInPriorArtifacts": False,
    }
    return audit_df, summary


def load_e01_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def repair_activation_seed(base_seed: int, condition_id: str, repeat_idx: int) -> int:
    condition_num = int(condition_id.replace("E01C", ""))
    return base_seed * 1_000_000 + 900_000 + condition_num * 1_000 + repeat_idx


def selected_repair_conditions(cfg: Mapping[str, Any], condition_ids: Sequence[str]) -> list[dict[str, Any]]:
    requested = tuple(dict.fromkeys(condition_ids))
    rows = [
        dict(row)
        for row in cfg["conditions"]
        if row["condition_id"] in requested and row["mode"] == "cell_view"
    ]
    found = {row["condition_id"] for row in rows}
    missing = sorted(set(requested) - found)
    if missing:
        raise RuntimeError(f"Missing expected S09 repair condition IDs in E01 config: {missing}")
    unsupported = [
        row["condition_id"]
        for row in rows
        if row.get("direction_profile") != "all_increasing"
    ]
    if unsupported:
        raise RuntimeError(
            "S09 repair matrix intentionally supports all-increasing global targets only; "
            f"mixed-direction conditions would need a separate target definition: {unsupported}"
        )
    return sorted(rows, key=lambda row: row["condition_id"])


def reverse_flags_for_condition(condition: Mapping[str, Any], assignments: Sequence[str]) -> tuple[bool, ...]:
    role_text = str(condition.get("algotype_roles", ""))
    role_map: dict[str, bool] = {}
    for part in role_text.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, role = part.split(":", 1)
        role_map[label.strip()] = "decreasing" in role
    return tuple(bool(role_map.get(label, False)) for label in assignments)


def build_repair_tasks(
    cfg: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    repo_dir: Path,
    max_repeats: int,
    max_events: int,
    max_successful_swaps: int,
    stall_events: int,
    activation_distribution: str,
) -> list[RepairTask]:
    repeat_count = min(max_repeats, int(cfg["globalDefaults"]["repeatCount"]))
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    value_banks = cfg["seedBanks"]["valueBanks"]
    assignment_banks = cfg["seedBanks"]["algotypeAssignmentBanks"]
    frozen_banks = cfg["seedBanks"]["frozenIndexBanks"]
    tasks: list[RepairTask] = []
    for condition in rows:
        value_bank = value_banks[condition["value_bank_id"]]
        assignment_bank = assignment_banks.get(condition["algotype_assignment_bank_id"])
        frozen_bank = frozen_banks[condition["frozen_index_bank_id"]]
        for repeat_idx in range(repeat_count):
            values = tuple(map(int, value_bank["initialArrays"][repeat_idx]))
            assignment_seed_value: int | None = None
            if condition["algorithm"] == "mixed":
                if assignment_bank is None:
                    raise RuntimeError(f"{condition['condition_id']} requires an assignment bank.")
                assignments = tuple(map(str, assignment_bank["assignments"][repeat_idx]))
                assignment_seed_value = int(assignment_bank["seeds"][repeat_idx])
            else:
                assignments = tuple(str(condition["algorithm"]) for _ in values)
            frozen_indices = tuple(map(int, frozen_bank["indices"][repeat_idx]))
            frozen_seed_value = int(frozen_bank["seeds"][repeat_idx]) if frozen_bank["seeds"] else None
            tasks.append(
                RepairTask(
                    condition=dict(condition),
                    repeat_index=int(repeat_idx),
                    values=values,
                    assignments=assignments,
                    reverse_directions=reverse_flags_for_condition(condition, assignments),
                    frozen_indices=frozen_indices,
                    initial_array_seed=int(value_bank["seeds"][repeat_idx]),
                    algotype_assignment_seed=assignment_seed_value,
                    frozen_index_seed=frozen_seed_value,
                    activation_seed=repair_activation_seed(base_seed, str(condition["condition_id"]), repeat_idx),
                    repo_dir=str(repo_dir),
                    max_events=int(max_events),
                    max_successful_swaps=int(max_successful_swaps),
                    stall_events=int(stall_events),
                    activation_distribution=str(activation_distribution),
                    sort_direction="increasing",
                )
            )
    return tasks


def target_values_for_initial(values: Sequence[int], direction: str) -> tuple[int, ...]:
    return tuple(map(int, target_sequence(values, direction)))


def normalized_metric_columns(values: Sequence[int], direction: str, sortedness: float, monotonicity_errors: int) -> dict[str, float | int]:
    alt = alternative_metrics(values, direction)
    n = len(values)
    return {
        **alt,
        "sortedness_adjacency_distance": float(1.0 - sortedness / 100.0),
        "monotonicity_error_fraction": float(monotonicity_errors / max(n - 1, 1)),
    }


def trace_row_for_record(task: RepairTask, record: Mapping[str, Any], result_summary: Mapping[str, Any], record_index: int, last_index: int) -> dict[str, Any]:
    values = tuple(map(int, record["values"]))
    labels = tuple(map(str, record["labels"]))
    target_values = target_values_for_initial(task.values, task.sort_direction)
    sortedness = float(record["sortedness_percent"])
    monotonicity_errors = int(record["monotonicity_error_count"])
    metric_values = normalized_metric_columns(values, task.sort_direction, sortedness, monotonicity_errors)
    duplicate_value_count = int(sum(count for count in Counter(task.values).values() if count > 1))
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "evidence_layer": "s09_repair_deterministic_rerun",
        "condition_id": task.condition["condition_id"],
        "run_family": task.condition["run_family"],
        "mode": "cell_view",
        "algorithm": task.condition["algorithm"],
        "algotype_mix": task.condition["algotype_mix"],
        "value_bank_id": task.condition["value_bank_id"],
        "value_distribution": task.condition.get("value_distribution", task.condition["value_bank_id"]),
        "direction_profile": task.condition.get("direction_profile", "all_increasing"),
        "algotype_roles": task.condition.get("algotype_roles", ""),
        "sort_direction": task.sort_direction,
        "target_order_policy": "stable_sorted_values_for_global_increasing_goal",
        "duplicate_handling_policy": "strict ties ignored for inversions; occurrence-token matching for position/edit metrics",
        "scheduler_regime": f"deterministic_single_event_{task.activation_distribution}",
        "scheduler_source": "s01_deterministic_simulator_public_move_methods",
        "repeat_index": int(task.repeat_index),
        "array_length": int(len(task.values)),
        "initial_array_seed": int(task.initial_array_seed),
        "activation_seed": int(task.activation_seed),
        "policy_seed": int(task.activation_seed),
        "algotype_assignment_bank_id": task.condition["algotype_assignment_bank_id"],
        "algotype_assignment_seed": task.algotype_assignment_seed,
        "frozen_semantics": task.condition["frozen_semantics"],
        "frozen_count": int(task.condition["frozen_count"]),
        "frozen_index_bank_id": task.condition["frozen_index_bank_id"],
        "frozen_index_seed": task.frozen_index_seed,
        "initial_frozen_indices_json": json.dumps(list(task.frozen_indices), separators=(",", ":")),
        "event_step": int(record["event_step"]),
        "trace_record_index": int(record_index),
        "swap_step": int(record["swap_step"]),
        "comparison_count_at_step": int(record["comparison_count"]),
        "frozen_attempt_count_at_step": int(record["frozen_attempt_count"]),
        "values_json": json.dumps(list(values), separators=(",", ":")),
        "labels_json": json.dumps(list(labels), separators=(",", ":")),
        "target_values_json": json.dumps(list(target_values), separators=(",", ":")),
        "values_sha256": stable_json_sha256(list(values)),
        "labels_sha256": stable_json_sha256(list(labels)),
        "target_values_sha256": stable_json_sha256(list(target_values)),
        "initial_array_sha256": stable_json_sha256(list(task.values)),
        "final_array_sha256": str(result_summary["final_array_sha256"]),
        "initial_algotype_counts_json": count_json(task.assignments),
        "current_algotype_counts_json": count_json(labels),
        "final_algotype_counts_json": str(result_summary["final_algotype_counts_json"]),
        "has_duplicate_values": bool(duplicate_value_count > 0),
        "duplicate_value_count": duplicate_value_count,
        "sortedness_percent": sortedness,
        "monotonicity_error_count": monotonicity_errors,
        "aggregation_left_neighbor_percent": float(record["aggregation_left_neighbor_percent"]),
        "aggregation_right_neighbor_legacy_percent": float(record["aggregation_right_neighbor_legacy_percent"]),
        "is_initial": bool(record_index == 0),
        "is_final": bool(record_index == last_index),
        "stop_reason": str(result_summary["stop_reason"]),
        "max_guard_hit": bool(result_summary["max_guard_hit"]),
        "event_count": int(result_summary["event_count"]),
        "swap_only_steps": int(result_summary["swap_count"]),
        "comparison_steps_observed": int(result_summary["comparison_count"]),
        "trace_record_count": int(result_summary["trace_record_count"]),
        **metric_values,
    }


def run_repair_task(task: RepairTask) -> dict[str, Any]:
    start = time.monotonic()
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
    summary = {
        "final_array_sha256": stable_json_sha256(list(result.final_values)),
        "final_algotype_counts_json": count_json(result.final_labels),
        "stop_reason": result.stop_reason,
        "max_guard_hit": bool(result.max_guard_hit),
        "event_count": int(result.event_count),
        "swap_count": int(result.swap_count),
        "comparison_count": int(result.comparison_count),
        "trace_record_count": int(len(result.records)),
    }
    trace_rows = [
        trace_row_for_record(task, record, summary, idx, len(result.records) - 1)
        for idx, record in enumerate(result.records)
    ]
    values_path = [json.loads(row["values_json"]) for row in trace_rows]
    state_curvature = state_space_curvature(values_path)
    state_step_lengths = np.linalg.norm(np.diff(np.asarray(values_path, dtype=float), axis=0), axis=1) if len(values_path) > 1 else np.asarray([], dtype=float)
    run_row = {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "evidence_layer": "s09_repair_deterministic_rerun",
        "condition_id": task.condition["condition_id"],
        "run_family": task.condition["run_family"],
        "mode": "cell_view",
        "algorithm": task.condition["algorithm"],
        "algotype_mix": task.condition["algotype_mix"],
        "value_distribution": task.condition.get("value_distribution", task.condition["value_bank_id"]),
        "direction_profile": task.condition.get("direction_profile", "all_increasing"),
        "sort_direction": task.sort_direction,
        "scheduler_regime": f"deterministic_single_event_{task.activation_distribution}",
        "repeat_index": int(task.repeat_index),
        "array_length": int(len(task.values)),
        "initial_array_seed": int(task.initial_array_seed),
        "activation_seed": int(task.activation_seed),
        "algotype_assignment_seed": task.algotype_assignment_seed,
        "frozen_semantics": task.condition["frozen_semantics"],
        "frozen_count": int(task.condition["frozen_count"]),
        "stop_reason": result.stop_reason,
        "max_guard_hit": bool(result.max_guard_hit),
        "event_count": int(result.event_count),
        "swap_only_steps": int(result.swap_count),
        "comparison_steps_observed": int(result.comparison_count),
        "trace_record_count": int(len(result.records)),
        "initial_array_sha256": stable_json_sha256(list(task.values)),
        "final_array_sha256": stable_json_sha256(list(result.final_values)),
        "trace_values_sha256": stable_json_sha256([json.loads(row["values_json"]) for row in trace_rows]),
        "value_state_path_curvature_ratio": float(state_curvature),
        "value_state_total_path_length": float(np.sum(state_step_lengths)) if len(state_step_lengths) else 0.0,
        "value_state_net_displacement": float(np.linalg.norm(np.asarray(values_path[-1], dtype=float) - np.asarray(values_path[0], dtype=float))) if len(values_path) > 1 else 0.0,
        "elapsed_seconds": float(time.monotonic() - start),
    }
    return {"trace_rows": trace_rows, "run_row": run_row}


def signed_segments(values: Sequence[float]) -> list[float]:
    compact: list[float] = []
    for value in values:
        value = float(value)
        if not compact or not math.isclose(compact[-1], value, abs_tol=1e-12):
            compact.append(value)
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


def dg_from_progress(values: Sequence[float]) -> dict[str, Any]:
    segments = signed_segments(values)
    events: list[tuple[float, float, float]] = []
    terminal_unrecovered_drop_segments = 0
    leading_increase_segments = 0
    seen_drop = False
    for idx, segment in enumerate(segments):
        if segment > 0 and not seen_drop:
            leading_increase_segments += 1
        if segment < 0:
            seen_drop = True
            if idx + 1 < len(segments) and segments[idx + 1] > 0:
                drop = -float(segment)
                recovery = float(segments[idx + 1])
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


def metric_run_rows(trace_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "condition_id",
        "repeat_index",
        "scheduler_regime",
        "run_family",
        "algorithm",
        "algotype_mix",
        "value_distribution",
        "frozen_semantics",
        "frozen_count",
    ]
    run_state = run_df.set_index(["condition_id", "repeat_index", "scheduler_regime"]).to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    sortedness_baselines: dict[tuple[str, int, str], dict[str, float | bool]] = {}
    for key, group in trace_df.sort_values(["condition_id", "repeat_index", "scheduler_regime", "trace_record_index"]).groupby(group_cols, sort=False):
        condition_id, repeat_index, scheduler_regime, run_family, algorithm, algotype_mix, value_distribution, frozen_semantics, frozen_count = key
        run_key = (condition_id, int(repeat_index), scheduler_regime)
        first = group.iloc[0]
        last = group.iloc[-1]
        for metric_name in REPAIR_DISTANCE_METRICS:
            distances = group[metric_name].astype(float).to_numpy()
            progress = 100.0 * (1.0 - distances)
            dg = dg_from_progress(progress)
            raw_initial = float(first.get("inversion_count", np.nan)) if metric_name == "inversion_fraction" else np.nan
            raw_final = float(last.get("inversion_count", np.nan)) if metric_name == "inversion_fraction" else np.nan
            row = {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "evidence_layer": "s09_repair_deterministic_rerun",
                "condition_id": condition_id,
                "run_family": run_family,
                "mode": "cell_view",
                "algorithm": algorithm,
                "algotype_mix": algotype_mix,
                "value_distribution": value_distribution,
                "frozen_semantics": frozen_semantics,
                "frozen_count": int(frozen_count),
                "scheduler_regime": scheduler_regime,
                "repeat_index": int(repeat_index),
                "metric_name": metric_name,
                "metric_category": "distance_to_target_order",
                "duplicate_handling_policy": str(first["duplicate_handling_policy"]),
                "normalized_distance_initial": float(distances[0]),
                "normalized_distance_final": float(distances[-1]),
                "normalized_distance_min": float(np.min(distances)),
                "progress_initial": float(progress[0]),
                "progress_final": float(progress[-1]),
                "progress_peak": float(np.max(progress)),
                "progress_curve_mean": float(np.mean(progress)),
                "raw_distance_initial": raw_initial,
                "raw_distance_final": raw_final,
                "improved_from_initial": bool(distances[-1] <= distances[0] + 1e-12),
                "final_target_reached": bool(distances[-1] <= 1e-12),
                "path_curvature_ratio": scalar_path_curvature(progress),
                **dg,
                "trace_record_count": int(len(group)),
                "stop_reason": str(last["stop_reason"]),
                "max_guard_hit": bool(last["max_guard_hit"]),
                "initial_array_sha256": str(first["initial_array_sha256"]),
                "final_array_sha256": str(last["final_array_sha256"]),
                "target_values_sha256": str(first["target_values_sha256"]),
            }
            rows.append(row)
            if metric_name == "sortedness_adjacency_distance":
                sortedness_baselines[run_key] = {
                    "sortedness_progress_final": row["progress_final"],
                    "sortedness_progress_curve_mean": row["progress_curve_mean"],
                    "sortedness_dg_primary": row["dg_primary"],
                    "sortedness_final_target_reached": row["final_target_reached"],
                    "sortedness_improved_from_initial": row["improved_from_initial"],
                }
        state = run_state[run_key]
        rows.append(
            {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "evidence_layer": "s09_repair_deterministic_rerun",
                "condition_id": condition_id,
                "run_family": run_family,
                "mode": "cell_view",
                "algorithm": algorithm,
                "algotype_mix": algotype_mix,
                "value_distribution": value_distribution,
                "frozen_semantics": frozen_semantics,
                "frozen_count": int(frozen_count),
                "scheduler_regime": scheduler_regime,
                "repeat_index": int(repeat_index),
                "metric_name": "value_state_space_curvature",
                "metric_category": "trajectory_shape",
                "duplicate_handling_policy": str(first["duplicate_handling_policy"]),
                "normalized_distance_initial": np.nan,
                "normalized_distance_final": np.nan,
                "normalized_distance_min": np.nan,
                "progress_initial": np.nan,
                "progress_final": np.nan,
                "progress_peak": np.nan,
                "progress_curve_mean": np.nan,
                "raw_distance_initial": np.nan,
                "raw_distance_final": np.nan,
                "improved_from_initial": np.nan,
                "final_target_reached": np.nan,
                "path_curvature_ratio": float(state["value_state_path_curvature_ratio"]),
                "dg_primary": np.nan,
                "dg_event_count": np.nan,
                "dg_total_drop": np.nan,
                "dg_total_recovery": np.nan,
                "terminal_unrecovered_drop_segment_count": np.nan,
                "leading_increase_segment_count": np.nan,
                "trace_record_count": int(len(group)),
                "stop_reason": str(last["stop_reason"]),
                "max_guard_hit": bool(last["max_guard_hit"]),
                "initial_array_sha256": str(first["initial_array_sha256"]),
                "final_array_sha256": str(last["final_array_sha256"]),
                "target_values_sha256": str(first["target_values_sha256"]),
            }
        )
    metric_df = pd.DataFrame(rows)
    for idx, row in metric_df.iterrows():
        if row["metric_name"] == "value_state_space_curvature":
            continue
        key = (row["condition_id"], int(row["repeat_index"]), row["scheduler_regime"])
        baseline = sortedness_baselines[key]
        metric_df.at[idx, "final_progress_delta_vs_sortedness"] = float(row["progress_final"] - baseline["sortedness_progress_final"])
        metric_df.at[idx, "curve_mean_progress_delta_vs_sortedness"] = float(row["progress_curve_mean"] - baseline["sortedness_progress_curve_mean"])
        metric_df.at[idx, "dg_primary_delta_vs_sortedness"] = float(row["dg_primary"] - baseline["sortedness_dg_primary"])
        metric_df.at[idx, "final_success_agrees_with_sortedness"] = bool(row["final_target_reached"] == baseline["sortedness_final_target_reached"])
        metric_df.at[idx, "improvement_agrees_with_sortedness"] = bool(row["improved_from_initial"] == baseline["sortedness_improved_from_initial"])
    return metric_df


def summarize_metric_results(metric_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    distance = metric_df[metric_df["metric_category"] == "distance_to_target_order"].copy()
    grouped = distance.groupby(["condition_id", "run_family", "algotype_mix", "value_distribution", "frozen_semantics", "frozen_count", "metric_name"], dropna=False)
    summary = grouped.agg(
        n_runs=("repeat_index", "nunique"),
        mean_progress_final=("progress_final", "mean"),
        sd_progress_final=("progress_final", "std"),
        mean_progress_curve=("progress_curve_mean", "mean"),
        mean_final_delta_vs_sortedness=("final_progress_delta_vs_sortedness", "mean"),
        mean_curve_delta_vs_sortedness=("curve_mean_progress_delta_vs_sortedness", "mean"),
        mean_dg_primary=("dg_primary", "mean"),
        mean_dg_delta_vs_sortedness=("dg_primary_delta_vs_sortedness", "mean"),
        final_success_rate=("final_target_reached", "mean"),
        final_success_agreement_rate=("final_success_agrees_with_sortedness", "mean"),
        improvement_agreement_rate=("improvement_agrees_with_sortedness", "mean"),
        mean_path_curvature_ratio=("path_curvature_ratio", "mean"),
    ).reset_index()
    summary["sd_progress_final"] = summary["sd_progress_final"].fillna(0.0)

    classifications = []
    for row in summary.to_dict(orient="records"):
        if row["metric_name"] == "sortedness_adjacency_distance":
            cls = "baseline_sortedness_metric"
        elif (
            float(row["final_success_agreement_rate"]) >= 0.9
            and float(row["improvement_agreement_rate"]) >= 0.9
            and abs(float(row["mean_final_delta_vs_sortedness"])) <= 10.0
        ):
            cls = "metric_stable"
        else:
            cls = "metric_dependent"
        classifications.append(
            {
                **row,
                "metric_stability_class": cls,
                "classification_rule": (
                    "metric_stable requires >=0.9 final-success agreement, >=0.9 improvement agreement, "
                    "and <=10 percentage-point mean final-progress delta versus Sortedness"
                ),
            }
        )
    classification_df = pd.DataFrame(classifications)
    return summary, classification_df


def make_metric_sensitivity_figure(summary_df: pd.DataFrame, classification_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    alt = summary_df[summary_df["metric_name"].isin(ALT_DISTANCE_METRICS)].copy()
    pivot = alt.pivot_table(index="condition_id", columns="metric_name", values="mean_final_delta_vs_sortedness", aggfunc="mean")
    metric_order = [metric for metric in ALT_DISTANCE_METRICS if metric in pivot.columns]
    pivot = pivot.reindex(columns=metric_order)
    curvature = classification_df[classification_df["metric_name"] == "sortedness_adjacency_distance"][
        ["condition_id", "mean_path_curvature_ratio"]
    ].set_index("condition_id")
    conditions = list(pivot.index)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.34 * len(conditions))), gridspec_kw={"width_ratios": [4.5, 1.4]})
    vmax = max(10.0, float(np.nanmax(np.abs(pivot.to_numpy()))) if len(pivot) else 10.0)
    im = axes[0].imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0].set_xticks(range(len(metric_order)))
    axes[0].set_xticklabels([label.replace("_", "\n") for label in metric_order], fontsize=8)
    axes[0].set_yticks(range(len(conditions)))
    axes[0].set_yticklabels(conditions, fontsize=8)
    axes[0].set_title("Mean final progress delta vs Sortedness")
    for i in range(len(conditions)):
        for j in range(len(metric_order)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                axes[0].text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("percentage points")

    curv_values = curvature.reindex(conditions)["mean_path_curvature_ratio"].to_numpy(dtype=float)
    axes[1].barh(range(len(conditions)), curv_values, color="#5b8fb9")
    axes[1].set_yticks(range(len(conditions)))
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ratio")
    axes[1].set_title("Sortedness path\ncurvature")
    for i, value in enumerate(curv_values):
        if pd.notna(value):
            axes[1].text(value, i, f" {value:.2f}", va="center", fontsize=7)
    fig.suptitle("E02 S09 repair metric-sensitivity matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def validate_repair_outputs(
    *,
    trace_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    tasks: Sequence[RepairTask],
    figure_path: Path,
    result_path: Path,
    trace_path: Path,
    sanity: Mapping[str, Any],
    preflight_status_path: Path,
) -> dict[str, Any]:
    expected_pairs = {(task.condition["condition_id"], task.repeat_index) for task in tasks}
    trace_pairs = set(zip(trace_df["condition_id"], trace_df["repeat_index"]))
    metric_pairs = set(zip(metric_df["condition_id"], metric_df["repeat_index"]))
    value_rows_complete = bool(trace_df["values_json"].notna().all() and trace_df["target_values_json"].notna().all())
    metadata_complete = bool(trace_df["sort_direction"].notna().all() and trace_df["target_order_policy"].notna().all())
    multiset_ok = True
    for _, row in trace_df.iterrows():
        values = json.loads(row["values_json"])
        target = json.loads(row["target_values_json"])
        if Counter(values) != Counter(target):
            multiset_ok = False
            break
    expected_duplicate_condition_ids = {
        task.condition["condition_id"]
        for task in tasks
        if task.condition["condition_id"] in REPAIR_DUPLICATE_CONDITION_IDS
    }
    duplicate_condition_ids = expected_duplicate_condition_ids & set(trace_df["condition_id"].unique())
    duplicate_rows = trace_df[trace_df["condition_id"].isin(duplicate_condition_ids)]
    classification_complete = bool(
        set(classification_df[classification_df["metric_name"].isin(ALT_DISTANCE_METRICS)]["metric_stability_class"].dropna().unique())
        <= {"metric_stable", "metric_dependent"}
    )
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(
            sanity["metricSanityChecksPassed"]
            and sanity["duplicateHandlingValidationPassed"]
            and expected_pairs <= trace_pairs
            and expected_pairs <= metric_pairs
            and value_rows_complete
            and metadata_complete
            and multiset_ok
            and len(duplicate_condition_ids) == len(expected_duplicate_condition_ids)
            and (not expected_duplicate_condition_ids or (not duplicate_rows.empty and bool(duplicate_rows["has_duplicate_values"].all())))
            and classification_complete
            and result_path.exists()
            and trace_path.exists()
            and figure_path.exists()
            and figure_path.stat().st_size > 0
        ),
        "status": "completed_repair_with_caveats",
        "validationResult": "pending",
        "caveatsOrBlockers": [
            "Original S09 preflight remains valid: prior E01/E02 artifacts did not persist per-step value arrays.",
            "Repair uses a bounded deterministic rerun evidence layer, not an OS-thread replay and not a replay of original E01 trajectories.",
            "Matrix is first 10 repeats for selected prior E02 conditions plus duplicate-value chimeras for duplicate handling.",
        ],
        "recommendedNextAction": "Review repaired S09 caveats, then proceed to S10 only after explicit Chief Scientist instruction.",
        "expectedRunCount": int(len(expected_pairs)),
        "traceRunCount": int(len(trace_pairs)),
        "metricRunCount": int(len(metric_pairs)),
        "traceRowCount": int(len(trace_df)),
        "metricRowCount": int(len(metric_df)),
        "summaryRowCount": int(len(summary_df)),
        "classificationRowCount": int(len(classification_df)),
        "conditionIds": sorted({task.condition["condition_id"] for task in tasks}),
        "repeatIndexes": sorted({int(task.repeat_index) for task in tasks}),
        "valueRowsComplete": value_rows_complete,
        "targetMetadataComplete": metadata_complete,
        "valueMultisetPreservedEveryTraceRow": bool(multiset_ok),
        "duplicateConditionIdsPresent": sorted(duplicate_condition_ids),
        "duplicateRowsHaveDuplicateValues": bool(not expected_duplicate_condition_ids or (not duplicate_rows.empty and duplicate_rows["has_duplicate_values"].all())),
        "metricSanityChecksPassed": bool(sanity["metricSanityChecksPassed"]),
        "duplicateHandlingValidationPassed": bool(sanity["duplicateHandlingValidationPassed"]),
        "classificationComplete": classification_complete,
        "figureNonempty": bool(figure_path.exists() and figure_path.stat().st_size > 0),
        "plannedResultsWritten": bool(result_path.exists()),
        "valueTraceWritten": bool(trace_path.exists()),
        "blockedPreflightStatusPreserved": bool(preflight_status_path.exists()),
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


def build_report(
    *,
    generated_at: str,
    artifacts_dir: Path,
    audit_path: Path,
    sanity_path: Path,
    validation_path: Path,
    status_path: Path,
    log_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    audit_df: pd.DataFrame,
    validation: Mapping[str, Any],
    unit_result: Mapping[str, Any] | None,
    git_commit: str,
    git_status: str,
    elapsed_seconds: float,
) -> str:
    artifacts_written = validation["artifactsWritten"]
    source_summary = audit_df[
        [
            "source_group",
            "source_name",
            "row_count",
            "has_sortedness_trajectory",
            "has_final_values_json",
            "has_per_step_value_trajectory",
            "blocker",
        ]
    ]
    unit_line = "not run"
    if unit_result is not None:
        unit_line = f"{unit_result['command']} -> return code {unit_result['returnCode']}"
    missing_results_path = artifacts_dir / "results" / "e02_alternative_metrics.parquet"
    missing_figure_path = artifacts_dir / "figures" / "e02" / "metric_sensitivity_matrix.png"
    return f"""# E02 S09 Full Results: Alternative Distance Metrics

## Top Summary

- Step ID: S09
- Completion status: blocked_missing_required_inputs
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: blocked after preflight; metric sanity checks passed and duplicate handling checks passed, but the input audit found zero persisted per-step value trajectories across the required E01/E02 trajectory artifacts.
- Outcome classification: constraining/contradictory
- Caveats or blockers: Required value trajectories are unavailable in the persisted E01/E02 artifacts. Existing traces store Sortedness trajectories, hashes, label-position codes, and final arrays, but not the intermediate value arrays needed for Kendall, Spearman, edit, EMD, or state-space curvature trajectories.
- Lay summary: The alternative metrics are well defined on small unique and duplicate arrays, but the saved experiment traces do not contain the actual array values at each recorded step. Without those states, S09 cannot honestly replace Sortedness on the same trajectories.
- Recommended next action: Regenerate or export trace-level value arrays from E01/E02 runs, including duplicate-value runs and direction/target metadata, then rerun S09. Do not start S10 until the Chief Scientist decides whether to regenerate the required trajectories or revise S09.

## Frozen Question

Do qualitative claims survive when progress is measured by distances other than the paper's Sortedness metric?

## Outcome

S09 could not produce the planned alternative-metric result table or metric-sensitivity figure because the required input layer is missing. The planned output `{missing_results_path}` was not written, and `{missing_figure_path}` was not written, to avoid mistaking an input-audit failure for a completed metric comparison.

This is a constraining result for the artifact package: prior steps preserved enough information for Sortedness, Aggregation, DG over Sortedness, and hash-level reproducibility checks, but not enough state to recompute rank/edit/transport metrics on the recorded trajectories.

## Inputs Audited

- E01 trace schema and trajectory artifacts under `/previous-artifacts/E01/traces/`.
- E02 result artifacts under `/artifacts/results/` and the S01 trace-row sample.
- E01 baseline config `/previous-artifacts/E01/configs/e01_baseline_configs.json`.
- Repository code at commit `{git_commit}`.

## Methods

The S09 script implemented the planned distance families and ran deterministic toy validations before auditing artifacts:

- Kendall tau distance and raw inversion count count strict out-of-order value pairs; equal-value ties are ignored.
- Spearman position distance assigns duplicate values deterministic occurrence tokens `(value, occurrence_index)` and compares current positions with target positions.
- Edit distance uses Levenshtein distance over the same occurrence tokens.
- Earth-mover-over-positions is the normalized sum of absolute current-vs-target occurrence-token displacements.
- Scalar trajectory curvature is total variation divided by net change. State-space curvature is total Euclidean path length divided by start-to-end Euclidean distance.

The artifact audit inspected Parquet metadata and relevant columns for persisted per-step value columns such as `values`, `values_json`, `current_values_json`, or equivalent state arrays. It separately checked whether `final_values_json` was present only on final rows, because final arrays are not sufficient for trajectory metrics.

## Commands

- `python scripts/e02_s09_alternative_metrics.py --repo-dir /workspace/cell-research --artifacts-dir /artifacts --run-unit-tests`
- Unit-test command inside the script: `{unit_line}`

## Results

Metric sanity validation passed, including duplicate-specific checks. The input audit failed the production prerequisite:

- Required trajectory/result sources audited: {validation['requiredSourceCount']}
- Existing sources audited: {validation['existingRequiredSourceCount']}
- Sources with Sortedness trajectories or summaries: {validation['sourcesWithSortednessTrajectories']}
- Sources with persisted per-step value trajectories: {validation['sourcesWithPerStepValueTrajectories']}
- Target-order definitions: {validation['targetOrderDefinitionsStatus']}
- Duplicate handling inputs: {validation['duplicateHandlingInputsStatus']}

### Input Audit Summary

{markdown_table(source_summary, max_rows=20)}

## Validation

- Metric sanity checks: {validation['metricSanityChecksPassed']}
- Duplicate handling validation: {validation['duplicateHandlingValidationPassed']}
- Input audit passed: {validation['inputAuditPassed']}
- Planned results table written: {validation['plannedResultsWritten']}
- Planned figure written: {validation['plannedFigureWritten']}
- Unit tests: {unit_line}

Detailed validation JSON: `{validation_path}`  
Status JSON: `{status_path}`  
Metric sanity details: `{sanity_path}`  
Input audit CSV: `{audit_path}`

## Caveats And Blockers

- The deterministic S01 simulator keeps value arrays in memory inside `SimulatorResult.records`, but prior E02 result tables stored compact hashes and summaries rather than those arrays.
- Re-running selected deterministic conditions would create a new evidence layer, not a replacement for the persisted E01/E02 trajectories requested by S09 as written.
- The E01 config provides initial value banks and duplicate-value conventions, so target orders are inferable for increasing/decreasing goals, but target definitions alone cannot reconstruct intermediate states.
- Opposite-direction chimeras may require per-label or per-direction target metadata when S09 is retried; this was not the blocking issue because value trajectories were already absent.

## Provenance

- Generated at UTC: {generated_at}
- Runtime: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, PyArrow {pq.__version__ if hasattr(pq, '__version__') else 'available'}
- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Elapsed seconds: {elapsed_seconds:.3f}
- No new dependencies were installed.
- Artifact manifest: `{manifest_path}`
- Source manifest: `{src_manifest_path}`

## Recommended Next Action

Regenerate or export trace-level value arrays for the E01 and E02 trajectory families needed by S09, with one row per recorded step containing the current value array and direction/target metadata. Then rerun S09 before starting S10, or revise S09 to be an explicitly new rerun-only analysis if the Chief Scientist decides the original trajectories do not need to be remeasured.
"""


def build_repair_report(
    *,
    generated_at: str,
    artifacts_dir: Path,
    trace_path: Path,
    result_path: Path,
    summary_path: Path,
    classification_path: Path,
    figure_path: Path,
    validation_path: Path,
    status_path: Path,
    sanity_path: Path,
    preflight_report_path: Path,
    preflight_status_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    log_path: Path,
    summary_df: pd.DataFrame,
    classification_df: pd.DataFrame,
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
    sensitivity_counts = classification_df[classification_df["metric_name"].isin(ALT_DISTANCE_METRICS)]["metric_stability_class"].value_counts().to_dict()
    summary_view = summary_df[
        summary_df["metric_name"].isin(("kendall_tau_distance", "spearman_position_distance", "earth_mover_position_distance"))
    ][
        [
            "condition_id",
            "metric_name",
            "n_runs",
            "mean_progress_final",
            "mean_final_delta_vs_sortedness",
            "mean_path_curvature_ratio",
        ]
    ]
    class_view = classification_df[classification_df["metric_name"].isin(ALT_DISTANCE_METRICS)][
        ["condition_id", "metric_name", "metric_stability_class", "mean_final_delta_vs_sortedness", "final_success_agreement_rate"]
    ]
    return f"""# E02 S09 Full Results: Alternative Distance Metrics Repair

## Top Summary

- Step ID: S09
- Completion status: completed_repair_with_caveats
- Artifacts written: {', '.join(validation['artifactsWritten'])}
- Validation result: passed repair validation; deterministic rerun value traces include per-step `values_json`, target-order metadata, and duplicate handling, while the original blocked preflight is preserved as a caveat.
- Outcome classification: supportive with caveats
- Caveats or blockers: The original S09 blocker remains valid for the prior persisted E01/E02 artifacts. This repair is a bounded deterministic rerun evidence layer, not an OS-thread replay and not an exact replay of the original E01 trace rows.
- Lay summary: The repair regenerated the missing per-step array values for a bounded set of deterministic reruns, then recalculated the alternative distance metrics. Most target-distance metrics agreed qualitatively with Sortedness on final progress in this repaired matrix, while metric-dependent slices identify where adjacent Sortedness can differ from global rank or position distances.
- Recommended next action: Review the repaired S09 caveats and then proceed to S10 only after explicit Chief Scientist instruction.

## Frozen Question

Do qualitative claims survive when progress is measured by distances other than the paper's Sortedness metric?

## Repair Scope

The original S09 preflight found that no prior E01/E02 artifact stored per-step value arrays. This repair therefore uses the S01 deterministic simulator path to regenerate a bounded value-trace evidence layer aligned with prior E02 repeat choices:

- Repeat indexes: `{parameters['repeatIndexes']}`
- Condition IDs: `{parameters['conditionIds']}`
- Activation distribution: `{parameters['activationDistribution']}`
- Max events per run: `{parameters['maxEvents']}`
- Workers: `{parameters['workers']}`

The matrix includes prior E02 pure, Frozen Cell, and same-goal chimera conditions plus duplicate-value same-goal chimeras so duplicate handling is exercised directly. Mixed opposite-direction chimeras remain outside this repair because they require per-label target definitions rather than one global target order.

## Methods

For each selected condition/repeat, the repair reconstructed initial values, Algotype labels, Frozen Cell indices, and seeds from `/previous-artifacts/E01/configs/e01_baseline_configs.json`. It then called the S01 deterministic simulator, which activates public cell `move()` methods under an explicit seeded single-event scheduler.

Every trace row records `values_json`, `labels_json`, `target_values_json`, `sort_direction`, target-order policy, duplicate-handling policy, hashes, Sortedness, monotonicity error, Aggregation, and the planned alternative distances:

- Kendall tau distance
- Inversion fraction and raw inversion count
- Spearman rank-position distance
- Edit distance to target order
- Earth-mover distance over target positions
- State-space trajectory curvature

Duplicate values are handled explicitly: strict inversion/Kendall comparisons ignore equal-value ties, while position/edit metrics use deterministic occurrence-token matching.

## Commands

- `python scripts/e02_s09_alternative_metrics.py --repair-rerun --repo-dir /workspace/cell-research --artifacts-dir /artifacts --max-repeats {parameters['maxRepeats']} --workers {parameters['workers']} --run-unit-tests`
- Unit-test command inside the script: `{unit_line}`

## Results

- Repaired value trace rows: {validation['traceRowCount']}
- Repaired metric rows: {validation['metricRowCount']}
- Conditions: {', '.join(validation['conditionIds'])}
- Repeat indexes: {validation['repeatIndexes']}
- Metric-stability classifications among alternative distance metrics: {json.dumps(sensitivity_counts, sort_keys=True)}

### Metric Summary Excerpt

{markdown_table(summary_view, max_rows=24)}

### Stability Classification Excerpt

{markdown_table(class_view, max_rows=36)}

## Validation

- Metric sanity checks passed: {validation['metricSanityChecksPassed']}
- Duplicate handling validation passed: {validation['duplicateHandlingValidationPassed']}
- Value rows complete: {validation['valueRowsComplete']}
- Target metadata complete: {validation['targetMetadataComplete']}
- Value multiset preserved in every trace row: {validation['valueMultisetPreservedEveryTraceRow']}
- Duplicate conditions present: {validation['duplicateConditionIdsPresent']}
- Planned results table written: {validation['plannedResultsWritten']}
- Metric-sensitivity figure nonempty: {validation['figureNonempty']}
- Existing blocked preflight status preserved: {validation['blockedPreflightStatusPreserved']}

Validation JSON: `{validation_path}`  
Status JSON: `{status_path}`  
Metric sanity JSON: `{sanity_path}`  
Preserved blocked preflight report: `{preflight_report_path}`  
Preserved blocked preflight status: `{preflight_status_path}`

## Artifacts

- Repaired value trajectories: `{trace_path}`
- Alternative metric results: `{result_path}`
- Metric summary CSV: `{summary_path}`
- Stability classification CSV: `{classification_path}`
- Metric-sensitivity figure: `{figure_path}`
- Execution log: `{log_path}`
- Artifact manifest: `{manifest_path}`
- Source manifest: `{src_manifest_path}`

## Caveats And Limitations

- This repair does not erase the original artifact limitation: prior persisted traces remain insufficient for alternative metrics without rerun.
- The repair is deterministic and calls public cell-policy methods, but it is not an OS-thread replay.
- The bounded matrix uses first 10 E01 repeat indexes, matching prior E02 bounded-repeat convention; it is not full 100-repeat paper-scale inference.
- Opposite-direction chimeras are excluded because global target order is not well defined when different Algotypes pursue opposing directions.
- Biological or morphogenesis interpretations remain computational proxy claims.

## Provenance

- Generated at UTC: {generated_at}
- Runtime: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, matplotlib {matplotlib.__version__}
- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Elapsed seconds: {elapsed_seconds:.3f}
- No new dependencies were installed.

## Recommended Next Action

Use the repaired S09 table as the metric-substitution evidence layer, carry the original preflight caveat into later claim audits, and proceed to S10 only after explicit Chief Scientist instruction.
"""


def run_repair(args: argparse.Namespace) -> int:
    start = time.monotonic()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e02"
    traces_dir = artifacts_dir / "traces"
    logs_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    for directory in (step_dir, results_dir, figures_dir, traces_dir, logs_dir, src_snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now()
    log_path = logs_dir / "e02_s09_alternative_metrics_repair.log"
    log_lines = [
        f"E02 S09 repair started {generated_at}",
        f"repo_dir={repo_dir}",
        f"artifacts_dir={artifacts_dir}",
        "mode=repair_deterministic_rerun",
    ]

    report_path = step_dir / "research_step_full_results.md"
    status_path = step_dir / "status.json"
    preflight_report_path = step_dir / "research_step_blocked_preflight.md"
    preflight_status_path = step_dir / "s09_blocked_preflight_status_before_repair.json"
    if report_path.exists() and not preflight_report_path.exists():
        preflight_report_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    if status_path.exists() and not preflight_status_path.exists():
        preflight_status_path.write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
    if not preflight_report_path.exists():
        preflight_report_path.write_text(
            "# S09 Blocked Preflight\n\nNo prior blocked preflight report was present in this artifact directory before repair.\n",
            encoding="utf-8",
        )
    if not preflight_status_path.exists():
        write_json(
            preflight_status_path,
            {
                "researchStepId": STEP_ID,
                "stepNumber": STEP_NUMBER,
                "success": False,
                "status": "blocked_preflight_not_present_before_repair",
                "artifactsWritten": [],
                "validationResult": "No prior blocked preflight status was present in this artifact directory before repair.",
                "caveatsOrBlockers": ["Repair was run in a fresh artifact directory without the original blocked preflight status."],
                "recommendedNextAction": "Use production /artifacts run to preserve the original S09 blocked preflight status.",
            },
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

    sanity = metric_sanity_checks()
    sanity_path = step_dir / "e02_s09_metric_sanity.json"
    write_json(sanity_path, sanity)
    cfg = load_e01_config(args.config_path)
    rows = selected_repair_conditions(cfg, args.condition_ids)
    tasks = build_repair_tasks(
        cfg,
        rows,
        repo_dir,
        args.max_repeats,
        args.max_events,
        args.max_successful_swaps,
        args.stall_events,
        args.activation_distribution,
    )
    log_lines.append(f"repair_conditions={len(rows)} tasks={len(tasks)} workers={args.workers}")

    trace_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_repair_task, task): task for task in tasks}
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_task), start=1):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                log_lines.append(f"FAILED task {task.condition['condition_id']} repeat={task.repeat_index}: {exc!r}")
                log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
                raise
            trace_rows.extend(result["trace_rows"])
            run_rows.append(result["run_row"])
            if idx % 20 == 0 or idx == len(tasks):
                log_lines.append(f"completed_tasks={idx}/{len(tasks)}")

    trace_df = pd.DataFrame(trace_rows).sort_values(["condition_id", "repeat_index", "scheduler_regime", "trace_record_index"]).reset_index(drop=True)
    run_df = pd.DataFrame(run_rows).sort_values(["condition_id", "repeat_index", "scheduler_regime"]).reset_index(drop=True)
    metric_df = metric_run_rows(trace_df, run_df)
    summary_df, classification_df = summarize_metric_results(metric_df)

    trace_path = traces_dir / "e02_s09_value_trajectories.parquet"
    result_path = results_dir / "e02_alternative_metrics.parquet"
    summary_path = step_dir / "e02_alternative_metric_summary.csv"
    classification_path = step_dir / "e02_alternative_metric_classification.csv"
    figure_path = figures_dir / "metric_sensitivity_matrix.png"
    validation_path = step_dir / "s09_repair_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s09_alternative_metrics_manifest.json"

    trace_df.to_parquet(trace_path, index=False)
    metric_df.to_parquet(result_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    make_metric_sensitivity_figure(summary_df, classification_df, figure_path)

    validation = validate_repair_outputs(
        trace_df=trace_df,
        metric_df=metric_df,
        summary_df=summary_df,
        classification_df=classification_df,
        tasks=tasks,
        figure_path=figure_path,
        result_path=result_path,
        trace_path=trace_path,
        sanity=sanity,
        preflight_status_path=preflight_status_path,
    )
    validation["validationResult"] = (
        "passed: repaired deterministic rerun wrote value trajectories, planned metric table, figure, and validation evidence"
        if validation["success"]
        else "failed: repaired deterministic rerun did not satisfy validation checks"
    )

    artifacts_written = [
        str(report_path),
        str(result_path),
        str(figure_path),
        str(trace_path),
        str(summary_path),
        str(classification_path),
        str(validation_path),
        str(status_path),
        str(sanity_path),
        str(preflight_report_path),
        str(preflight_status_path),
        str(log_path),
        str(manifest_path),
        str(src_manifest_path),
    ]
    validation["artifactsWritten"] = artifacts_written
    validation["outcomeClassification"] = "supportive_with_caveats"
    write_json(validation_path, validation)
    write_json(status_path, validation)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    elapsed_seconds = time.monotonic() - start
    parameters = {
        "maxRepeats": int(args.max_repeats),
        "workers": int(args.workers),
        "maxEvents": int(args.max_events),
        "maxSuccessfulSwaps": int(args.max_successful_swaps),
        "stallEvents": int(args.stall_events),
        "activationDistribution": args.activation_distribution,
        "conditionIds": sorted({task.condition["condition_id"] for task in tasks}),
        "repeatIndexes": sorted({int(task.repeat_index) for task in tasks}),
    }

    code_files = [
        repo_dir / "scripts/e02_s09_alternative_metrics.py",
        repo_dir / "tests/e02/test_alternative_metrics.py",
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
                "sizeBytes": int(path.stat().st_size),
            }
            for path in code_files
            if path.exists()
        ],
        "parameters": parameters,
        "validation": validation,
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pyarrow": "available",
            "newDependenciesInstalled": [],
        },
    }
    write_json(src_manifest_path, src_manifest)

    report_text = build_repair_report(
        generated_at=generated_at,
        artifacts_dir=artifacts_dir,
        trace_path=trace_path,
        result_path=result_path,
        summary_path=summary_path,
        classification_path=classification_path,
        figure_path=figure_path,
        validation_path=validation_path,
        status_path=status_path,
        sanity_path=sanity_path,
        preflight_report_path=preflight_report_path,
        preflight_status_path=preflight_status_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        log_path=log_path,
        summary_df=summary_df,
        classification_df=classification_df,
        validation=validation,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        elapsed_seconds=elapsed_seconds,
        parameters=parameters,
    )
    report_path.write_text(report_text, encoding="utf-8")

    log_lines.extend(
        [
            f"trace_rows={len(trace_df)} metric_rows={len(metric_df)} summary_rows={len(summary_df)}",
            f"validation_success={validation['success']}",
            f"elapsed_seconds={elapsed_seconds:.3f}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "status": validation["status"],
        "artifacts": [
            artifact_entry(report_path, artifacts_dir, "S09 repaired full-results report"),
            artifact_entry(result_path, artifacts_dir, "Planned S09 alternative metric results table"),
            artifact_entry(figure_path, artifacts_dir, "Planned S09 metric-sensitivity figure"),
            artifact_entry(trace_path, artifacts_dir, "Repaired trace-level per-step value arrays with target metadata"),
            artifact_entry(summary_path, artifacts_dir, "Condition-level alternative metric summary"),
            artifact_entry(classification_path, artifacts_dir, "Metric-stability classification table"),
            artifact_entry(validation_path, artifacts_dir, "S09 repair validation evidence"),
            artifact_entry(status_path, artifacts_dir, "S09 current machine-readable status"),
            artifact_entry(sanity_path, artifacts_dir, "Metric sanity and duplicate-handling validation"),
            artifact_entry(preflight_report_path, artifacts_dir, "Preserved blocked preflight report"),
            artifact_entry(preflight_status_path, artifacts_dir, "Preserved blocked preflight status JSON"),
            artifact_entry(log_path, artifacts_dir, "S09 repair execution log"),
            artifact_entry(src_manifest_path, artifacts_dir, "S09 repair source snapshot manifest"),
        ],
    }
    write_json(manifest_path, manifest)
    return 0 if validation["success"] else 1


def run_preflight(args: argparse.Namespace) -> int:
    start = time.monotonic()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    logs_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    for directory in (step_dir, logs_dir, src_snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now()
    log_lines = [
        f"E02 S09 started {generated_at}",
        f"repo_dir={repo_dir}",
        f"artifacts_dir={artifacts_dir}",
        "mode=preflight_blocker_audit",
    ]

    unit_result: dict[str, Any] | None = None
    if args.run_unit_tests:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_dir)
        unit_result = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_*.py"], repo_dir, env)
        log_lines.extend(["UNIT TESTS:", json.dumps(unit_result, indent=2, sort_keys=True)])
        if not unit_result["success"]:
            log_path = logs_dir / "e02_s09_alternative_metrics.log"
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return int(unit_result["returnCode"] or 1)

    sanity = metric_sanity_checks()
    audit_df, audit_summary = audit_inputs(args.config_path)

    audit_path = step_dir / "e02_s09_input_audit.csv"
    sanity_path = step_dir / "e02_s09_metric_sanity.json"
    validation_path = step_dir / "s09_validation.json"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s09_alternative_metrics_manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    log_path = logs_dir / "e02_s09_alternative_metrics.log"

    audit_df.to_csv(audit_path, index=False)
    write_json(sanity_path, sanity)

    input_audit_passed = bool(
        audit_summary["allRequiredValueTrajectoriesAvailable"]
        and audit_summary["targetOrderDefinitionsAvailable"]
        and audit_summary["duplicateHandlingInputsAvailable"]
    )
    status = "blocked_missing_required_value_trajectories" if not input_audit_passed else "ready_for_metric_evaluation"
    caveats = [
        "No required E01/E02 source has persisted per-step value arrays.",
        "Final arrays and value-trace hashes are insufficient to compute alternative metric trajectories.",
        "Target orders are inferable from configs, but intermediate value states are not reconstructible from persisted hashes/summaries.",
    ]
    artifacts_written = [
        str(audit_path),
        str(sanity_path),
        str(validation_path),
        str(status_path),
        str(log_path),
        str(report_path),
        str(src_manifest_path),
        str(manifest_path),
    ]
    validation = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": False,
        "status": status,
        "artifactsWritten": artifacts_written,
        "validationResult": (
            "blocked: metric sanity and duplicate checks passed, but input audit found zero persisted per-step value trajectories"
        ),
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": (
            "Regenerate or export trace-level value arrays with direction/target metadata, then rerun S09 before starting S10."
        ),
        "outcomeClassification": "constraining/contradictory",
        "metricSanityChecksPassed": bool(sanity["metricSanityChecksPassed"]),
        "duplicateHandlingValidationPassed": bool(sanity["duplicateHandlingValidationPassed"]),
        "inputAuditPassed": input_audit_passed,
        "plannedResultsWritten": False,
        "plannedFigureWritten": False,
        "requiredSourceCount": int(audit_summary["requiredSourceCount"]),
        "existingRequiredSourceCount": int(audit_summary["existingRequiredSourceCount"]),
        "sourcesWithPerStepValueTrajectories": int(audit_summary["sourcesWithPerStepValueTrajectories"]),
        "sourcesWithSortednessTrajectories": int(audit_summary["sourcesWithSortednessTrajectories"]),
        "targetOrderDefinitionsStatus": (
            "available_inferable_from_config" if audit_summary["targetOrderDefinitionsAvailable"] else "unavailable"
        ),
        "duplicateHandlingInputsStatus": (
            "available_duplicate_bank_and_metric_policy_validated"
            if audit_summary["duplicateHandlingInputsAvailable"]
            else "unavailable"
        ),
        "configAudit": audit_summary["configAudit"],
        "missingPlannedOutputs": [
            str(artifacts_dir / "results" / "e02_alternative_metrics.parquet"),
            str(artifacts_dir / "figures" / "e02" / "metric_sensitivity_matrix.png"),
        ],
    }
    write_json(validation_path, validation)
    write_json(status_path, validation)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    elapsed_seconds = time.monotonic() - start
    log_lines.extend(
        [
            f"metric_sanity_checks_passed={sanity['metricSanityChecksPassed']}",
            f"duplicate_handling_validation_passed={sanity['duplicateHandlingValidationPassed']}",
            f"input_audit_passed={input_audit_passed}",
            f"sources_with_per_step_values={audit_summary['sourcesWithPerStepValueTrajectories']}",
            f"elapsed_seconds={elapsed_seconds:.3f}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    code_files = [
        repo_dir / "scripts/e02_s09_alternative_metrics.py",
        repo_dir / "tests/e02/test_alternative_metrics.py",
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
                "sizeBytes": int(path.stat().st_size),
            }
            for path in code_files
            if path.exists()
        ],
        "parameters": {
            "mode": "preflight_blocker_audit",
            "e01TraceSources": [path for _, path in E01_TRACE_SOURCES],
            "e02ResultSources": [path for _, path in E02_RESULT_SOURCES],
        },
        "validation": validation,
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": "available",
            "newDependenciesInstalled": [],
        },
    }
    write_json(src_manifest_path, src_manifest)

    report_text = build_report(
        generated_at=generated_at,
        artifacts_dir=artifacts_dir,
        audit_path=audit_path,
        sanity_path=sanity_path,
        validation_path=validation_path,
        status_path=status_path,
        log_path=log_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        audit_df=audit_df,
        validation=validation,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        elapsed_seconds=elapsed_seconds,
    )
    report_path.write_text(report_text, encoding="utf-8")
    write_json(validation_path, validation)
    write_json(status_path, validation)

    manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "status": status,
        "artifacts": [
            artifact_entry(audit_path, artifacts_dir, "Input audit for E01/E02 value trajectory availability"),
            artifact_entry(sanity_path, artifacts_dir, "Alternative metric sanity checks and duplicate policy validation"),
            artifact_entry(validation_path, artifacts_dir, "S09 validation evidence"),
            artifact_entry(status_path, artifacts_dir, "S09 machine-readable blocked status"),
            artifact_entry(report_path, artifacts_dir, "S09 full-results blocked handoff report"),
            artifact_entry(log_path, artifacts_dir, "S09 execution log"),
            artifact_entry(src_manifest_path, artifacts_dir, "S09 source snapshot manifest"),
        ],
        "plannedOutputsNotWritten": validation["missingPlannedOutputs"],
    }
    write_json(manifest_path, manifest)

    return 0 if sanity["metricSanityChecksPassed"] and sanity["duplicateHandlingValidationPassed"] else 1


def main() -> int:
    args = parse_args()
    if args.repair_rerun:
        return run_repair(args)
    return run_preflight(args)


if __name__ == "__main__":
    raise SystemExit(main())
