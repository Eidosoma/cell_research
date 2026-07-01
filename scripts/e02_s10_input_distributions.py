#!/usr/bin/env python3
"""Run E02 S10 input-distribution stress tests.

This step reuses the S01 deterministic simulator and the S09 repaired
duplicate-aware metric definitions.  It varies only the initial value
distribution while preserving public cell ``move()`` dispatch, selected E01
condition metadata, Algotype assignments, Frozen Cell semantics, and the first
10-repeat convention used by prior E02 bounded matrices.
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

from scripts.e02_s09_alternative_metrics import (
    dg_from_progress,
    metric_sanity_checks,
    normalized_metric_columns,
    reverse_flags_for_condition,
    scalar_path_curvature,
    state_space_curvature,
    target_values_for_initial,
)
from src.e02.deterministic_simulator import (
    SimulatorConfig,
    count_json,
    simulate,
    stable_json_sha256,
)


STEP_ID = "S10"
STEP_NUMBER = 10
EXPERIMENT_ID = "E02"
SOURCE_EXPERIMENT_ID = "E01"

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

INPUT_DISTRIBUTIONS = (
    "random_permutation",
    "nearly_sorted",
    "reverse_sorted",
    "block_shuffled",
    "duplicate_heavy",
    "heavy_tailed",
    "local_repeated_motifs",
)
INPUT_DISTRIBUTION_INDEX = {name: idx + 1 for idx, name in enumerate(INPUT_DISTRIBUTIONS)}
ALT_PROGRESS_METRICS = (
    "kendall_tau_distance",
    "inversion_fraction",
    "spearman_position_distance",
    "edit_distance_to_target",
    "earth_mover_position_distance",
)


@dataclass(frozen=True)
class DistributionValues:
    values: tuple[int, ...]
    input_distribution_seed: int
    source_value_bank_id: str
    generation_policy: str
    validation: dict[str, Any]


@dataclass(frozen=True)
class InputTask:
    condition: dict[str, Any]
    repeat_index: int
    input_distribution: str
    values: tuple[int, ...]
    assignments: tuple[str, ...]
    reverse_directions: tuple[bool, ...]
    frozen_indices: tuple[int, ...]
    input_distribution_seed: int
    source_value_bank_id: str
    generation_policy: str
    generation_validation: dict[str, Any]
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
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-events", type=int, default=150_000)
    parser.add_argument("--max-successful-swaps", type=int, default=80_000)
    parser.add_argument("--stall-events", type=int, default=2_000)
    parser.add_argument("--activation-distribution", choices=("uniform_active", "left_to_right_active", "right_to_left_active"), default="uniform_active")
    parser.add_argument("--condition-ids", nargs="*", default=list(SELECTED_CONDITION_IDS))
    parser.add_argument("--input-distributions", nargs="*", default=list(INPUT_DISTRIBUTIONS))
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


def load_e01_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


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
        raise RuntimeError(f"Missing expected S10 condition IDs in E01 config: {missing}")
    unsupported = [
        row["condition_id"]
        for row in rows
        if row.get("direction_profile") != "all_increasing"
    ]
    if unsupported:
        raise RuntimeError(
            "S10 input-distribution matrix intentionally keeps one global increasing target; "
            f"mixed-direction conditions require a separate target definition: {unsupported}"
        )
    return sorted(rows, key=lambda row: row["condition_id"])


def distribution_seed(base_seed: int, distribution_name: str, repeat_idx: int) -> int:
    return base_seed * 1_000_000 + 1_000_000 + INPUT_DISTRIBUTION_INDEX[distribution_name] * 10_000 + repeat_idx


def condition_seed_component(condition_id: str) -> int:
    if condition_id.startswith("E01C") and condition_id[4:].isdigit():
        return int(condition_id.replace("E01C", ""))
    digest = hashlib.sha256(condition_id.encode("utf-8")).hexdigest()
    return 10_000 + int(digest[:8], 16) % 90_000


def activation_seed(base_seed: int, condition_id: str, distribution_name: str, repeat_idx: int) -> int:
    condition_num = condition_seed_component(condition_id)
    _ = distribution_name
    return base_seed * 1_000_000 + 900_000 + condition_num * 1_000 + repeat_idx


def gini(values: Sequence[int]) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return 0.0
    total = float(arr.sum())
    if math.isclose(total, 0.0):
        return 0.0
    weighted = float(sum((idx + 1) * value for idx, value in enumerate(arr)))
    return float((2.0 * weighted) / (arr.size * total) - (arr.size + 1.0) / arr.size)


def strict_inversion_count(values: Sequence[int], direction: str = "increasing") -> int:
    inversions = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if direction == "increasing":
                inversions += int(values[i] > values[j])
            else:
                inversions += int(values[i] < values[j])
    return inversions


def strict_pair_count(values: Sequence[int]) -> int:
    counts = Counter(values)
    n = len(values)
    return n * (n - 1) // 2 - sum(count * (count - 1) // 2 for count in counts.values())


def inversion_fraction(values: Sequence[int]) -> float:
    possible = strict_pair_count(values)
    return float(strict_inversion_count(values) / possible) if possible else 0.0


def local_motif_template(seed: int) -> tuple[int, ...]:
    templates = (
        (0, 2, 1, 4, 3, 6, 5, 8, 7, 9),
        (1, 0, 2, 4, 3, 5, 7, 6, 8, 9),
        (0, 1, 3, 2, 5, 4, 6, 7, 9, 8),
    )
    return templates[seed % len(templates)]


def validate_distribution_values(name: str, values: Sequence[int], n: int, seed: int) -> dict[str, Any]:
    values_tuple = tuple(map(int, values))
    sorted_unique = tuple(range(1, n + 1))
    result: dict[str, Any] = {
        "input_distribution": name,
        "input_distribution_seed": int(seed),
        "array_length": int(len(values_tuple)),
        "values_sha256": stable_json_sha256(list(values_tuple)),
        "has_duplicate_values": bool(len(set(values_tuple)) < len(values_tuple)),
        "duplicate_value_count": int(sum(count for count in Counter(values_tuple).values() if count > 1)),
        "inversion_fraction": float(inversion_fraction(values_tuple)),
        "gini": float(gini(values_tuple)),
        "constraints_passed": False,
        "constraint_failures": [],
    }
    failures: list[str] = []
    if len(values_tuple) != n:
        failures.append("length_mismatch")
    if not all(isinstance(value, int) for value in values_tuple):
        failures.append("non_integer_value")

    if name == "random_permutation":
        if tuple(sorted(values_tuple)) != sorted_unique:
            failures.append("not_unique_1_to_n_permutation")
    elif name == "nearly_sorted":
        if tuple(sorted(values_tuple)) != sorted_unique:
            failures.append("not_unique_1_to_n_permutation")
        if not (0.0 < result["inversion_fraction"] <= 0.01):
            failures.append("nearly_sorted_inversion_fraction_out_of_bounds")
    elif name == "reverse_sorted":
        if values_tuple != tuple(range(n, 0, -1)):
            failures.append("not_reverse_sorted")
    elif name == "block_shuffled":
        block_size = 10
        chunks = [values_tuple[idx : idx + block_size] for idx in range(0, n, block_size)]
        block_ids = [int((chunk[0] - 1) // block_size) for chunk in chunks if chunk]
        if tuple(sorted(values_tuple)) != sorted_unique:
            failures.append("not_unique_1_to_n_permutation")
        if any(tuple(chunk) != tuple(sorted(chunk)) for chunk in chunks):
            failures.append("block_internal_order_not_sorted")
        if sorted(block_ids) != list(range(n // block_size)):
            failures.append("block_ids_not_permutation")
        if block_ids == list(range(n // block_size)):
            failures.append("block_shuffle_identity_order")
    elif name == "duplicate_heavy":
        counts = Counter(values_tuple)
        if sorted(counts) != list(range(1, 11)):
            failures.append("duplicate_values_not_1_to_10")
        if any(count != 10 for count in counts.values()):
            failures.append("duplicate_counts_not_balanced_10_each")
    elif name == "heavy_tailed":
        expected = tuple(i**4 for i in range(1, n + 1))
        if tuple(sorted(values_tuple)) != expected:
            failures.append("not_shuffled_fourth_power_values")
        if result["gini"] < 0.6:
            failures.append("gini_below_heavy_tail_threshold")
        if max(values_tuple) / max(min(values_tuple), 1) < 1_000_000:
            failures.append("tail_ratio_below_threshold")
    elif name == "local_repeated_motifs":
        block_size = 10
        motif = local_motif_template(seed)
        chunks = [values_tuple[idx : idx + block_size] for idx in range(0, n, block_size)]
        if tuple(sorted(values_tuple)) != sorted_unique:
            failures.append("not_unique_1_to_n_permutation")
        for block_idx, chunk in enumerate(chunks):
            expected_chunk = tuple(block_idx * block_size + motif_idx + 1 for motif_idx in motif)
            if tuple(chunk) != expected_chunk:
                failures.append("local_motif_not_repeated")
                break
    else:
        failures.append("unknown_distribution")

    result["constraints_passed"] = bool(not failures)
    result["constraint_failures"] = failures
    return result


def generate_distribution_values(
    cfg: Mapping[str, Any],
    distribution_name: str,
    repeat_idx: int,
    n: int,
) -> DistributionValues:
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    value_banks = cfg["seedBanks"]["valueBanks"]
    seed = distribution_seed(base_seed, distribution_name, repeat_idx)
    rng = random.Random(seed)

    if distribution_name == "random_permutation":
        bank = value_banks["unique_1_to_100"]
        values = tuple(map(int, bank["initialArrays"][repeat_idx]))
        seed = int(bank["seeds"][repeat_idx])
        source_bank_id = "unique_1_to_100"
        policy = "E01 unique permutation value bank"
    elif distribution_name == "duplicate_heavy":
        bank = value_banks["duplicate_1_to_10_x10"]
        values = tuple(map(int, bank["initialArrays"][repeat_idx]))
        seed = int(bank["seeds"][repeat_idx])
        source_bank_id = "duplicate_1_to_10_x10"
        policy = "E01 duplicate 1-to-10 x10 value bank"
    elif distribution_name == "nearly_sorted":
        values_list = list(range(1, n + 1))
        target_swaps = max(1, n // 10)
        candidates = list(range(n - 1))
        rng.shuffle(candidates)
        used: set[int] = set()
        positions: list[int] = []
        for pos in candidates:
            if pos in used or pos + 1 in used:
                continue
            positions.append(pos)
            used.add(pos)
            used.add(pos + 1)
            if len(positions) == target_swaps:
                break
        for pos in sorted(positions):
            values_list[pos], values_list[pos + 1] = values_list[pos + 1], values_list[pos]
        values = tuple(values_list)
        source_bank_id = "generated_nearly_sorted"
        policy = f"sorted 1..{n} with {len(positions)} non-overlapping seeded adjacent swaps"
    elif distribution_name == "reverse_sorted":
        values = tuple(range(n, 0, -1))
        source_bank_id = "generated_reverse_sorted"
        policy = f"exact reverse order {n}..1"
    elif distribution_name == "block_shuffled":
        block_size = 10
        blocks = [list(range(block_start + 1, block_start + block_size + 1)) for block_start in range(0, n, block_size)]
        original_blocks = [tuple(block) for block in blocks]
        rng.shuffle(blocks)
        if [tuple(block) for block in blocks] == original_blocks:
            blocks = blocks[1:] + blocks[:1]
        values = tuple(value for block in blocks for value in block)
        source_bank_id = "generated_block_shuffled"
        policy = "seeded shuffle of ten sorted contiguous value blocks"
    elif distribution_name == "heavy_tailed":
        values_list = [i**4 for i in range(1, n + 1)]
        rng.shuffle(values_list)
        values = tuple(values_list)
        source_bank_id = "generated_heavy_tailed_fourth_power"
        policy = "seeded shuffle of unique fourth-power values 1^4..100^4"
    elif distribution_name == "local_repeated_motifs":
        block_size = 10
        motif = local_motif_template(seed)
        values = tuple(
            block_idx * block_size + motif_idx + 1
            for block_idx in range(n // block_size)
            for motif_idx in motif
        )
        source_bank_id = "generated_local_repeated_motifs"
        policy = f"same seeded within-block rank motif repeated across {n // block_size} blocks"
    else:
        raise ValueError(f"Unsupported input distribution: {distribution_name}")

    validation = validate_distribution_values(distribution_name, values, n, seed)
    return DistributionValues(
        values=tuple(map(int, values)),
        input_distribution_seed=int(seed),
        source_value_bank_id=source_bank_id,
        generation_policy=policy,
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
    input_distributions: Sequence[str],
) -> list[InputTask]:
    unknown_distributions = sorted(set(input_distributions) - set(INPUT_DISTRIBUTIONS))
    if unknown_distributions:
        raise RuntimeError(f"Unsupported S10 input distributions requested: {unknown_distributions}")
    repeat_count = min(max_repeats, int(cfg["globalDefaults"]["repeatCount"]))
    n = int(cfg["globalDefaults"]["arrayLength"])
    base_seed = int(cfg["globalDefaults"]["baseSeed"])
    assignment_banks = cfg["seedBanks"]["algotypeAssignmentBanks"]
    frozen_banks = cfg["seedBanks"]["frozenIndexBanks"]
    tasks: list[InputTask] = []
    generated_by_distribution_repeat = {
        (distribution_name, repeat_idx): generate_distribution_values(cfg, distribution_name, repeat_idx, n)
        for distribution_name in input_distributions
        for repeat_idx in range(repeat_count)
    }
    for condition in rows:
        assignment_bank = assignment_banks.get(condition["algotype_assignment_bank_id"])
        frozen_bank = frozen_banks[condition["frozen_index_bank_id"]]
        for distribution_name in input_distributions:
            for repeat_idx in range(repeat_count):
                generated = generated_by_distribution_repeat[(distribution_name, repeat_idx)]
                values = generated.values
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
                    InputTask(
                        condition=dict(condition),
                        repeat_index=int(repeat_idx),
                        input_distribution=distribution_name,
                        values=values,
                        assignments=assignments,
                        reverse_directions=reverse_flags_for_condition(condition, assignments),
                        frozen_indices=frozen_indices,
                        input_distribution_seed=generated.input_distribution_seed,
                        source_value_bank_id=generated.source_value_bank_id,
                        generation_policy=generated.generation_policy,
                        generation_validation=generated.validation,
                        algotype_assignment_seed=assignment_seed_value,
                        frozen_index_seed=frozen_seed_value,
                        activation_seed=activation_seed(base_seed, str(condition["condition_id"]), distribution_name, repeat_idx),
                        repo_dir=str(repo_dir),
                        max_events=int(max_events),
                        max_successful_swaps=int(max_successful_swaps),
                        stall_events=int(stall_events),
                        activation_distribution=str(activation_distribution),
                        sort_direction="increasing",
                    )
                )
    return tasks


def distribution_validation_rows(tasks: Sequence[InputTask]) -> pd.DataFrame:
    seen: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.input_distribution, task.repeat_index)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "input_distribution": task.input_distribution,
                "repeat_index": int(task.repeat_index),
                "input_distribution_seed": int(task.input_distribution_seed),
                "source_value_bank_id": task.source_value_bank_id,
                "generation_policy": task.generation_policy,
                **task.generation_validation,
            }
        )
    return pd.DataFrame(rows).sort_values(["input_distribution", "repeat_index"]).reset_index(drop=True)


def finite_or_none(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def run_input_task(task: InputTask) -> dict[str, Any]:
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
    target_values = target_values_for_initial(task.values, task.sort_direction)
    values_path = [tuple(map(int, record["values"])) for record in records]
    sortedness_curve = [float(record["sortedness_percent"]) for record in records]
    aggregation_curve = [float(record["aggregation_left_neighbor_percent"]) for record in records]
    monotonicity_curve = [int(record["monotonicity_error_count"]) for record in records]
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
    dg = dg_from_progress(sortedness_curve)
    value_state_curvature = state_space_curvature(values_path)
    sortedness_curvature = scalar_path_curvature(sortedness_curve)
    final_progress = {
        f"{metric}_final_progress_percent": float(100.0 * (1.0 - float(final_metrics[metric])))
        for metric in ALT_PROGRESS_METRICS
    }
    initial_progress = {
        f"{metric}_initial_progress_percent": float(100.0 * (1.0 - float(initial_metrics[metric])))
        for metric in ALT_PROGRESS_METRICS
    }
    row = {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "evidence_layer": "s10_input_distribution_deterministic_rerun",
        "condition_id": task.condition["condition_id"],
        "run_family": task.condition["run_family"],
        "mode": "cell_view",
        "algorithm": task.condition["algorithm"],
        "algotype_mix": task.condition["algotype_mix"],
        "input_distribution": task.input_distribution,
        "source_value_bank_id": task.source_value_bank_id,
        "generation_policy": task.generation_policy,
        "sort_direction": task.sort_direction,
        "target_order_policy": "stable_sorted_values_for_global_increasing_goal",
        "duplicate_handling_policy": "S09 repaired policy: strict ties ignored for inversions; occurrence-token matching for position/edit metrics",
        "scheduler_regime": f"deterministic_single_event_{task.activation_distribution}",
        "scheduler_source": "s01_deterministic_simulator_public_move_methods",
        "repeat_index": int(task.repeat_index),
        "array_length": int(len(task.values)),
        "input_distribution_seed": int(task.input_distribution_seed),
        "activation_seed": int(task.activation_seed),
        "policy_seed": int(task.activation_seed),
        "algotype_assignment_bank_id": task.condition["algotype_assignment_bank_id"],
        "algotype_assignment_seed": task.algotype_assignment_seed,
        "frozen_semantics": task.condition["frozen_semantics"],
        "frozen_count": int(task.condition["frozen_count"]),
        "frozen_index_bank_id": task.condition["frozen_index_bank_id"],
        "frozen_index_seed": task.frozen_index_seed,
        "initial_frozen_indices_json": json.dumps(list(task.frozen_indices), separators=(",", ":")),
        "initial_values_json": json.dumps(list(task.values), separators=(",", ":")),
        "target_values_json": json.dumps(list(target_values), separators=(",", ":")),
        "initial_array_sha256": stable_json_sha256(list(task.values)),
        "target_values_sha256": stable_json_sha256(list(target_values)),
        "final_array_sha256": stable_json_sha256(list(result.final_values)),
        "trace_values_sha256": stable_json_sha256([list(values) for values in values_path]),
        "initial_algotype_counts_json": count_json(task.assignments),
        "final_algotype_counts_json": count_json(result.final_labels),
        "input_constraints_passed": bool(task.generation_validation["constraints_passed"]),
        "input_constraint_failures_json": json.dumps(task.generation_validation["constraint_failures"], separators=(",", ":")),
        "has_duplicate_values": bool(task.generation_validation["has_duplicate_values"]),
        "duplicate_value_count": int(task.generation_validation["duplicate_value_count"]),
        "input_inversion_fraction": float(task.generation_validation["inversion_fraction"]),
        "input_gini": float(task.generation_validation["gini"]),
        "initial_sortedness_percent": float(sortedness_curve[0]),
        "final_sortedness_percent": float(sortedness_curve[-1]),
        "peak_sortedness_percent": float(max(sortedness_curve)),
        "curve_mean_sortedness_percent": float(np.mean(sortedness_curve)),
        "final_monotonicity_error_count": int(monotonicity_curve[-1]),
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
        **{f"{metric}_initial_distance": float(initial_metrics[metric]) for metric in ALT_PROGRESS_METRICS},
        **{f"{metric}_final_distance": float(final_metrics[metric]) for metric in ALT_PROGRESS_METRICS},
        **initial_progress,
        **final_progress,
        **dg,
    }
    return row


def summarize_results(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = result_df.groupby(
        [
            "condition_id",
            "run_family",
            "algorithm",
            "algotype_mix",
            "frozen_semantics",
            "frozen_count",
            "input_distribution",
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
        mean_compare_plus_swap_steps=("compare_plus_swap_steps", "mean"),
        mean_event_count=("event_count", "mean"),
        mean_swap_only_steps=("swap_only_steps", "mean"),
        mean_dg_primary=("dg_primary", "mean"),
        mean_dg_event_count=("dg_event_count", "mean"),
        mean_peak_aggregation=("peak_aggregation_left_neighbor_percent", "mean"),
        mean_final_aggregation=("final_aggregation_left_neighbor_percent", "mean"),
        mean_sortedness_curvature=("sortedness_path_curvature_ratio", "mean"),
        mean_value_state_curvature=("value_state_path_curvature_ratio", "mean"),
        mean_kendall_final_progress=("kendall_tau_distance_final_progress_percent", "mean"),
        mean_emd_final_progress=("earth_mover_position_distance_final_progress_percent", "mean"),
        max_guard_hit_rate=("max_guard_hit", "mean"),
    ).reset_index()
    summary["sd_final_sortedness"] = summary["sd_final_sortedness"].fillna(0.0)

    baseline = summary[summary["input_distribution"] == "random_permutation"][
        [
            "condition_id",
            "success_rate",
            "mean_final_sortedness",
            "mean_compare_plus_swap_steps",
            "mean_dg_primary",
            "mean_peak_aggregation",
            "mean_kendall_final_progress",
            "mean_emd_final_progress",
        ]
    ].rename(
        columns={
            "success_rate": "baseline_success_rate",
            "mean_final_sortedness": "baseline_mean_final_sortedness",
            "mean_compare_plus_swap_steps": "baseline_mean_compare_plus_swap_steps",
            "mean_dg_primary": "baseline_mean_dg_primary",
            "mean_peak_aggregation": "baseline_mean_peak_aggregation",
            "mean_kendall_final_progress": "baseline_mean_kendall_final_progress",
            "mean_emd_final_progress": "baseline_mean_emd_final_progress",
        }
    )
    classified = summary.merge(baseline, on="condition_id", how="left")
    classified["delta_final_sortedness_vs_random"] = classified["mean_final_sortedness"] - classified["baseline_mean_final_sortedness"]
    classified["delta_success_rate_vs_random"] = classified["success_rate"] - classified["baseline_success_rate"]
    classified["compare_plus_swap_ratio_vs_random"] = classified["mean_compare_plus_swap_steps"] / classified["baseline_mean_compare_plus_swap_steps"].replace(0, np.nan)
    classified["delta_dg_primary_vs_random"] = classified["mean_dg_primary"] - classified["baseline_mean_dg_primary"]
    classified["delta_peak_aggregation_vs_random"] = classified["mean_peak_aggregation"] - classified["baseline_mean_peak_aggregation"]
    classified["delta_kendall_final_progress_vs_random"] = classified["mean_kendall_final_progress"] - classified["baseline_mean_kendall_final_progress"]
    classified["delta_emd_final_progress_vs_random"] = classified["mean_emd_final_progress"] - classified["baseline_mean_emd_final_progress"]

    classes: list[str] = []
    reasons: list[str] = []
    for row in classified.to_dict(orient="records"):
        if row["input_distribution"] == "random_permutation":
            classes.append("baseline_random_permutation")
            reasons.append("reference distribution")
            continue
        final_sensitive = float(row["delta_final_sortedness_vs_random"]) < -5.0 or float(row["delta_success_rate_vs_random"]) < -0.1
        ratio = row["compare_plus_swap_ratio_vs_random"]
        efficiency_sensitive = pd.notna(ratio) and (float(ratio) > 2.0 or float(ratio) < 0.5)
        path_sensitive = abs(float(row["delta_dg_primary_vs_random"])) > 25.0
        aggregation_sensitive = (
            str(row["algorithm"]) == "mixed"
            and abs(float(row["delta_peak_aggregation_vs_random"])) > 10.0
        )
        metric_sensitive = (
            abs(float(row["delta_kendall_final_progress_vs_random"])) > 10.0
            or abs(float(row["delta_emd_final_progress_vs_random"])) > 10.0
        )
        if final_sensitive:
            classes.append("input_sensitive_final_state")
            reasons.append("final sortedness or success rate changed materially")
        elif efficiency_sensitive or path_sensitive or aggregation_sensitive or metric_sensitive:
            classes.append("input_sensitive_efficiency_or_path")
            active_reasons = []
            if efficiency_sensitive:
                active_reasons.append("efficiency ratio outside [0.5, 2.0]")
            if path_sensitive:
                active_reasons.append("DG delta exceeds 25 percentage points")
            if aggregation_sensitive:
                active_reasons.append("mixed-run peak Aggregation delta exceeds 10 percentage points")
            if metric_sensitive:
                active_reasons.append("alternative final-progress delta exceeds 10 percentage points")
            reasons.append("; ".join(active_reasons))
        else:
            classes.append("input_stable")
            reasons.append("final state, efficiency, path, aggregation, and alternative-progress deltas within thresholds")
    classified["input_generalization_class"] = classes
    classified["classification_rule"] = reasons
    return summary, classified


def make_input_distribution_figure(summary_df: pd.DataFrame, classification_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    non_baseline = classification_df[classification_df["input_distribution"] != "random_permutation"].copy()
    dist_order = [name for name in INPUT_DISTRIBUTIONS if name in set(summary_df["input_distribution"])]
    condition_order = sorted(summary_df["condition_id"].unique())

    final_pivot = classification_df.pivot_table(
        index="condition_id",
        columns="input_distribution",
        values="delta_final_sortedness_vs_random",
        aggfunc="mean",
    ).reindex(index=condition_order, columns=dist_order)
    ratio_pivot = classification_df.pivot_table(
        index="condition_id",
        columns="input_distribution",
        values="compare_plus_swap_ratio_vs_random",
        aggfunc="mean",
    ).reindex(index=condition_order, columns=dist_order)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={"height_ratios": [1.25, 1.0]})
    vmax = max(5.0, float(np.nanmax(np.abs(final_pivot.to_numpy()))) if len(final_pivot) else 5.0)
    im = axes[0, 0].imshow(final_pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title("Final Sortedness delta vs random permutation")
    axes[0, 0].set_xticks(range(len(dist_order)))
    axes[0, 0].set_xticklabels([name.replace("_", "\n") for name in dist_order], fontsize=8)
    axes[0, 0].set_yticks(range(len(condition_order)))
    axes[0, 0].set_yticklabels(condition_order, fontsize=8)
    for i in range(len(condition_order)):
        for j in range(len(dist_order)):
            value = final_pivot.iloc[i, j]
            if pd.notna(value):
                axes[0, 0].text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04).set_label("percentage points")

    ratio_data = np.log2(ratio_pivot.to_numpy(dtype=float))
    finite_ratio = ratio_data[np.isfinite(ratio_data)]
    rmax = max(1.0, float(np.nanmax(np.abs(finite_ratio))) if finite_ratio.size else 1.0)
    im2 = axes[0, 1].imshow(ratio_data, aspect="auto", cmap="PiYG_r", vmin=-rmax, vmax=rmax)
    axes[0, 1].set_title("log2 efficiency ratio vs random permutation")
    axes[0, 1].set_xticks(range(len(dist_order)))
    axes[0, 1].set_xticklabels([name.replace("_", "\n") for name in dist_order], fontsize=8)
    axes[0, 1].set_yticks(range(len(condition_order)))
    axes[0, 1].set_yticklabels([])
    for i in range(len(condition_order)):
        for j in range(len(dist_order)):
            value = ratio_pivot.iloc[i, j]
            if pd.notna(value):
                axes[0, 1].text(j, i, f"{value:.1f}x", ha="center", va="center", fontsize=7)
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04).set_label("log2 ratio")

    dist_summary = summary_df.groupby("input_distribution").agg(
        mean_dg=("mean_dg_primary", "mean"),
        mean_final_sortedness=("mean_final_sortedness", "mean"),
    ).reindex(dist_order)
    axes[1, 0].bar(range(len(dist_order)), dist_summary["mean_dg"], color="#6b8f71")
    axes[1, 0].set_xticks(range(len(dist_order)))
    axes[1, 0].set_xticklabels([name.replace("_", "\n") for name in dist_order], fontsize=8)
    axes[1, 0].set_ylabel("DG primary")
    axes[1, 0].set_title("Mean DG by input distribution")

    class_counts = non_baseline.groupby(["input_distribution", "input_generalization_class"]).size().unstack(fill_value=0).reindex(dist_order)
    class_order = ["input_stable", "input_sensitive_efficiency_or_path", "input_sensitive_final_state"]
    bottoms = np.zeros(len(dist_order))
    colors = {
        "input_stable": "#4c78a8",
        "input_sensitive_efficiency_or_path": "#f58518",
        "input_sensitive_final_state": "#e45756",
    }
    for cls in class_order:
        values = class_counts[cls].to_numpy() if cls in class_counts else np.zeros(len(dist_order))
        axes[1, 1].bar(range(len(dist_order)), values, bottom=bottoms, color=colors[cls], label=cls.replace("_", " "))
        bottoms += values
    axes[1, 1].set_xticks(range(len(dist_order)))
    axes[1, 1].set_xticklabels([name.replace("_", "\n") for name in dist_order], fontsize=8)
    axes[1, 1].set_ylabel("condition slices")
    axes[1, 1].set_title("Input generalization classes")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("E02 S10 input-distribution stress test", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def regenerate_task_values(cfg: Mapping[str, Any], task: InputTask) -> tuple[int, ...]:
    return generate_distribution_values(
        cfg,
        task.input_distribution,
        task.repeat_index,
        len(task.values),
    ).values


def run_seed_reproducibility_checks(cfg: Mapping[str, Any], tasks: Sequence[InputTask], result_df: pd.DataFrame) -> dict[str, Any]:
    regenerated_ok = True
    mismatches: list[dict[str, Any]] = []
    for task in tasks:
        regenerated = regenerate_task_values(cfg, task)
        if regenerated != task.values:
            regenerated_ok = False
            mismatches.append(
                {
                    "condition_id": task.condition["condition_id"],
                    "input_distribution": task.input_distribution,
                    "repeat_index": int(task.repeat_index),
                }
            )
            if len(mismatches) >= 10:
                break

    sample_rows = []
    preferred_condition_order = ("E01C006", "E01C005", "E01C004", *SELECTED_CONDITION_IDS)
    for distribution_name in INPUT_DISTRIBUTIONS:
        distribution_tasks = [task for task in tasks if task.input_distribution == distribution_name]
        for condition_id in preferred_condition_order:
            candidates = [task for task in distribution_tasks if task.condition["condition_id"] == condition_id]
            if candidates:
                sample_rows.append(candidates[0])
                break
        else:
            if distribution_tasks:
                sample_rows.append(distribution_tasks[0])
    deterministic_ok = True
    deterministic_checks: list[dict[str, Any]] = []
    index = result_df.set_index(["condition_id", "input_distribution", "repeat_index"])
    for task in sample_rows:
        rerun = run_input_task(task)
        key = (task.condition["condition_id"], task.input_distribution, task.repeat_index)
        original = index.loc[key]
        passed = bool(
            str(rerun["final_array_sha256"]) == str(original["final_array_sha256"])
            and str(rerun["trace_values_sha256"]) == str(original["trace_values_sha256"])
            and int(rerun["event_count"]) == int(original["event_count"])
        )
        deterministic_ok = deterministic_ok and passed
        deterministic_checks.append(
            {
                "condition_id": task.condition["condition_id"],
                "input_distribution": task.input_distribution,
                "repeat_index": int(task.repeat_index),
                "passed": passed,
                "final_array_sha256": rerun["final_array_sha256"],
                "trace_values_sha256": rerun["trace_values_sha256"],
            }
        )
    return {
        "inputSeedRegenerationPassed": bool(regenerated_ok),
        "inputSeedMismatchExamples": mismatches,
        "simulationDeterminismSamplePassed": bool(deterministic_ok),
        "simulationDeterminismSampleCount": int(len(deterministic_checks)),
        "simulationDeterminismChecks": deterministic_checks,
    }


def validate_outputs(
    *,
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    tasks: Sequence[InputTask],
    figure_path: Path,
    result_path: Path,
    sanity: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {(task.condition["condition_id"], task.input_distribution, task.repeat_index) for task in tasks}
    observed_keys = set(zip(result_df["condition_id"], result_df["input_distribution"], result_df["repeat_index"]))
    expected_distributions = sorted({task.input_distribution for task in tasks})
    observed_distributions = sorted(result_df["input_distribution"].unique())
    duplicate_required = "duplicate_heavy" in expected_distributions
    heavy_tail_required = "heavy_tailed" in expected_distributions
    duplicate_rows = result_df[result_df["input_distribution"] == "duplicate_heavy"]
    heavy_rows = result_df[result_df["input_distribution"] == "heavy_tailed"]
    alt_metric_columns = [f"{metric}_final_progress_percent" for metric in ALT_PROGRESS_METRICS]
    classification_classes = set(classification_df["input_generalization_class"].dropna().unique())
    success = bool(
        expected_keys == observed_keys
        and bool(distribution_df["constraints_passed"].all())
        and bool(result_df["input_constraints_passed"].all())
        and bool(sanity["metricSanityChecksPassed"])
        and bool(sanity["duplicateHandlingValidationPassed"])
        and bool(reproducibility["inputSeedRegenerationPassed"])
        and bool(reproducibility["simulationDeterminismSamplePassed"])
        and (not duplicate_required or not duplicate_rows.empty)
        and (not duplicate_required or bool(duplicate_rows["has_duplicate_values"].all()))
        and (not duplicate_required or bool(duplicate_rows[alt_metric_columns].notna().all().all()))
        and (not heavy_tail_required or not heavy_rows.empty)
        and (not heavy_tail_required or bool((heavy_rows["input_gini"] >= 0.6).all()))
        and classification_classes <= {
            "baseline_random_permutation",
            "input_stable",
            "input_sensitive_efficiency_or_path",
            "input_sensitive_final_state",
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
        "outcomeClassification": "constraining/contradictory",
        "artifactsWritten": [],
        "caveatsOrBlockers": [
            "Matrix is bounded to first 10 repeats and selected prior-E02 cell-view conditions.",
            "Duplicate-heavy metrics use the S09 repaired tie policy, not the paper's strict unique-value convention.",
            "Heavy-tailed values are deterministic positive integer rank transforms; value-state curvature is magnitude-sensitive under this distribution.",
            "Runs use the S01 deterministic public-move simulator, not OS-thread replays.",
        ],
        "recommendedNextAction": "Review S10 input-distribution sensitivity, then proceed to S11 only after explicit Chief Scientist instruction.",
        "expectedRunCount": int(len(expected_keys)),
        "observedRunCount": int(len(observed_keys)),
        "resultRowCount": int(len(result_df)),
        "summaryRowCount": int(len(summary_df)),
        "classificationRowCount": int(len(classification_df)),
        "distributionValidationRowCount": int(len(distribution_df)),
        "conditionIds": sorted({task.condition["condition_id"] for task in tasks}),
        "inputDistributions": expected_distributions,
        "observedInputDistributions": observed_distributions,
        "repeatIndexes": sorted({int(task.repeat_index) for task in tasks}),
        "allExpectedRunsPresent": bool(expected_keys == observed_keys),
        "distributionConstraintsPassed": bool(distribution_df["constraints_passed"].all()),
        "runInputConstraintsPassed": bool(result_df["input_constraints_passed"].all()),
        "metricSanityChecksPassed": bool(sanity["metricSanityChecksPassed"]),
        "duplicateHandlingValidationPassed": bool(sanity["duplicateHandlingValidationPassed"]),
        "duplicateDistributionRowsPresent": bool(not duplicate_required or not duplicate_rows.empty),
        "duplicateRowsHaveDuplicateValues": bool(not duplicate_required or (not duplicate_rows.empty and duplicate_rows["has_duplicate_values"].all())),
        "duplicateAltMetricColumnsComplete": bool(not duplicate_required or (not duplicate_rows.empty and duplicate_rows[alt_metric_columns].notna().all().all())),
        "heavyTailRowsPresent": bool(not heavy_tail_required or not heavy_rows.empty),
        "heavyTailGiniConstraintPassed": bool(not heavy_tail_required or (not heavy_rows.empty and (heavy_rows["input_gini"] >= 0.6).all())),
        "classificationClasses": sorted(classification_classes),
        "inputGeneralizationClassCounts": classification_df["input_generalization_class"].value_counts().sort_index().to_dict(),
        "figureNonempty": bool(figure_path.exists() and figure_path.stat().st_size > 0),
        "plannedMatrixWritten": bool(result_path.exists()),
        **dict(reproducibility),
    }
    validation["validationResult"] = (
        "passed: generated distribution constraints, duplicate-aware metrics, reproducible seeds, planned matrix, and figure validated"
        if success
        else "failed: one or more S10 validation checks did not pass"
    )
    return validation


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
    result_path: Path,
    summary_path: Path,
    classification_path: Path,
    distribution_validation_path: Path,
    figure_path: Path,
    validation_path: Path,
    status_path: Path,
    sanity_path: Path,
    log_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    summary_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
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
    class_counts = classification_df["input_generalization_class"].value_counts().sort_index().to_dict()
    largest_final = classification_df[classification_df["input_distribution"] != "random_permutation"].copy()
    largest_final = largest_final.sort_values("delta_final_sortedness_vs_random").head(18)[
        [
            "condition_id",
            "input_distribution",
            "input_generalization_class",
            "mean_final_sortedness",
            "delta_final_sortedness_vs_random",
            "compare_plus_swap_ratio_vs_random",
            "delta_dg_primary_vs_random",
        ]
    ]
    distribution_view = distribution_df[
        [
            "input_distribution",
            "repeat_index",
            "constraints_passed",
            "inversion_fraction",
            "gini",
            "has_duplicate_values",
            "duplicate_value_count",
        ]
    ]
    return f"""# E02 S10 Full Results: Input Distribution Stress Test

## Top Summary

- Step ID: S10
- Completion status: {validation['status']}
- Artifacts written: {', '.join(validation['artifactsWritten'])}
- Validation result: {validation['validationResult']}
- Outcome classification: {validation['outcomeClassification']}
- Caveats or blockers: {', '.join(validation['caveatsOrBlockers'])}
- Lay summary: S10 regenerated deterministic runs across seven starting-array distributions. Final ordering often remained achievable, but efficiency, DG/path shape, Aggregation, and alternative-distance progress were input-sensitive enough that random-permutation results should not be treated as distribution-general without qualification.
- Recommended next action: {validation['recommendedNextAction']}

## Frozen Question

Are robustness, DG, efficiency, and Aggregation specific to random permutations, or do they generalize across structured and duplicate-heavy initial conditions?

## Inputs

- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- S01 deterministic simulator: `src/e02/deterministic_simulator.py`
- S09 repaired metric code and caveats: `scripts/e02_s09_alternative_metrics.py`
- Selected condition IDs: `{parameters['conditionIds']}`
- Input distributions: `{parameters['inputDistributions']}`
- Repeat indexes: `{parameters['repeatIndexes']}`

No external datasets were required and no new dependencies were installed.

## Methods

The S10 matrix holds the public cell-policy semantics fixed and varies only initial values. For each selected E01 cell-view condition and repeat, the script reconstructed Algotype labels and Frozen Cell metadata from the E01 config, generated one of seven value distributions, and ran the S01 deterministic single-event simulator with public `move()` methods. Activation seeds reuse the S09 condition/repeat schedule and are matched across input distributions so the distribution contrast is not confounded with a new scheduler draw.

The seven input families were:

- `random_permutation`: original E01 unique 1..100 value-bank rows.
- `nearly_sorted`: sorted 1..100 with 10 non-overlapping seeded adjacent swaps.
- `reverse_sorted`: exact 100..1 order.
- `block_shuffled`: sorted contiguous blocks internally, with the blocks shuffled.
- `duplicate_heavy`: original E01 duplicate 1..10 x10 value-bank rows.
- `heavy_tailed`: seeded shuffle of unique fourth-power values 1^4..100^4.
- `local_repeated_motifs`: the same local within-block inversion motif repeated across ten blocks.

Metrics include final/peak/curve Sortedness, monotonicity error, compare-plus-swap efficiency, DG over Sortedness, Aggregation, S09 duplicate-aware alternative target distances, and scalar/state-space path curvature.

## Commands

- `python scripts/e02_s10_input_distributions.py --repo-dir /workspace/cell-research --artifacts-dir /artifacts --max-repeats {parameters['maxRepeats']} --workers {parameters['workers']} --run-unit-tests`
- Unit-test command inside the script: `{unit_line}`

## Results

- Matrix rows: {validation['resultRowCount']}
- Summary rows: {validation['summaryRowCount']}
- Classification rows: {validation['classificationRowCount']}
- Input generalization class counts: {json.dumps(class_counts, sort_keys=True)}
- Expected runs present: {validation['allExpectedRunsPresent']}
- Distribution constraints passed: {validation['distributionConstraintsPassed']}
- Duplicate-heavy metric validation passed: {validation['duplicateAltMetricColumnsComplete']}
- Heavy-tail validation passed: {validation['heavyTailGiniConstraintPassed']}

### Largest Final Sortedness Drops

{markdown_table(largest_final, max_rows=18)}

### Distribution Validation Excerpt

{markdown_table(distribution_view, max_rows=21)}

## Validation

- Metric sanity checks passed: {validation['metricSanityChecksPassed']}
- Duplicate handling validation passed: {validation['duplicateHandlingValidationPassed']}
- Input seed regeneration passed: {validation['inputSeedRegenerationPassed']}
- Simulation determinism sample passed: {validation['simulationDeterminismSamplePassed']} across {validation['simulationDeterminismSampleCount']} sampled reruns.
- Generated distribution constraints passed: {validation['distributionConstraintsPassed']}
- Planned matrix written: {validation['plannedMatrixWritten']}
- Figure nonempty: {validation['figureNonempty']}
- Unit tests: {unit_line}

Validation JSON: `{validation_path}`  
Status JSON: `{status_path}`  
Metric sanity JSON: `{sanity_path}`

## Artifacts

- Input distribution matrix: `{result_path}`
- Condition-distribution summary CSV: `{summary_path}`
- Generalization classification CSV: `{classification_path}`
- Distribution validation CSV: `{distribution_validation_path}`
- Figure: `{figure_path}`
- Execution log: `{log_path}`
- Artifact manifest: `{manifest_path}`
- Source manifest: `{src_manifest_path}`

## Caveats And Limitations

- S10 is a bounded deterministic rerun matrix: first 10 repeats and selected prior-E02 cell-view conditions, not full 100-repeat paper-scale inference.
- The duplicate-heavy distribution uses S09 repaired duplicate handling. This is appropriate for a stress test but not the same as the paper's strict unique-value metric convention.
- Heavy-tailed values preserve rank order but change value magnitudes; state-space curvature is therefore magnitude-sensitive and should be compared cautiously against 1..100 distributions.
- The simulator preserves public local policy calls but is not an OS-thread replay.
- Opposite-direction chimeras remain outside this step because they need per-label target definitions rather than one global target order.

## Provenance

- Generated at UTC: {generated_at}
- Runtime: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, matplotlib {matplotlib.__version__}
- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Elapsed seconds: {elapsed_seconds:.3f}
- Workers: {parameters['workers']}
- No new dependencies were installed.

## Recommended Next Action

Carry S10's input-sensitivity caveats into the final claim audit and proceed to S11 Frozen Cell placement only after explicit Chief Scientist instruction.
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
    result_path = results_dir / "e02_input_distribution_matrix.parquet"
    summary_path = step_dir / "e02_input_distribution_summary.csv"
    classification_path = step_dir / "e02_input_distribution_classification.csv"
    distribution_validation_path = step_dir / "e02_input_distribution_validation.csv"
    figure_path = figures_dir / "input_distribution_effects.png"
    validation_path = step_dir / "s10_validation.json"
    status_path = step_dir / "status.json"
    sanity_path = step_dir / "e02_s10_metric_sanity.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s10_input_distributions_manifest.json"
    log_path = logs_dir / "e02_s10_input_distributions.log"

    cfg = load_e01_config(args.config_path)
    rows = selected_conditions(cfg, args.condition_ids)
    tasks = build_tasks(
        cfg,
        rows,
        repo_dir,
        args.max_repeats,
        args.max_events,
        args.max_successful_swaps,
        args.stall_events,
        args.activation_distribution,
        args.input_distributions,
    )
    distribution_df = distribution_validation_rows(tasks)
    sanity = metric_sanity_checks()
    write_json(sanity_path, sanity)

    unit_result = None
    if args.run_unit_tests:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_dir)
        unit_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_*.py"],
            repo_dir,
            env=env,
        )
        if not unit_result["success"]:
            write_json(step_dir / "s10_unit_test_failure.json", unit_result)
            return 2

    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        result_rows = list(executor.map(run_input_task, tasks))
    result_df = pd.DataFrame(result_rows).sort_values(["condition_id", "input_distribution", "repeat_index"]).reset_index(drop=True)
    summary_df, classification_df = summarize_results(result_df)
    make_input_distribution_figure(summary_df, classification_df, figure_path)

    result_df.to_parquet(result_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    distribution_df.to_csv(distribution_validation_path, index=False)

    reproducibility = run_seed_reproducibility_checks(cfg, tasks, result_df)
    validation = validate_outputs(
        result_df=result_df,
        summary_df=summary_df,
        classification_df=classification_df,
        distribution_df=distribution_df,
        tasks=tasks,
        figure_path=figure_path,
        result_path=result_path,
        sanity=sanity,
        reproducibility=reproducibility,
    )

    planned_artifacts = [
        (report_path, "S10 full-results report"),
        (result_path, "Planned S10 input distribution matrix"),
        (figure_path, "Planned S10 input distribution effects figure"),
        (summary_path, "Condition-distribution summary table"),
        (classification_path, "Input generalization classification table"),
        (distribution_validation_path, "Generated distribution constraint validation table"),
        (validation_path, "S10 validation evidence"),
        (status_path, "S10 machine-readable status"),
        (sanity_path, "S10 metric sanity and duplicate handling validation"),
        (log_path, "S10 execution log"),
        (manifest_path, "S10 artifact manifest"),
        (src_manifest_path, "S10 source snapshot manifest"),
    ]
    validation["artifactsWritten"] = [str(path) for path, _ in planned_artifacts]
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"]),
        "status": validation["status"],
        "validationResult": validation["validationResult"],
        "artifactsWritten": validation["artifactsWritten"],
        "outcomeClassification": validation["outcomeClassification"],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
    }
    write_json(validation_path, validation)
    write_json(status_path, status)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    parameters = {
        "conditionIds": sorted({task.condition["condition_id"] for task in tasks}),
        "inputDistributions": sorted({task.input_distribution for task in tasks}, key=lambda name: INPUT_DISTRIBUTION_INDEX[name]),
        "repeatIndexes": sorted({int(task.repeat_index) for task in tasks}),
        "activationDistribution": args.activation_distribution,
        "maxRepeats": int(args.max_repeats),
        "maxEvents": int(args.max_events),
        "maxSuccessfulSwaps": int(args.max_successful_swaps),
        "stallEvents": int(args.stall_events),
        "workers": int(args.workers),
    }
    generated_at = utc_now()
    elapsed_seconds = float(time.monotonic() - start)
    report = build_report(
        generated_at=generated_at,
        artifacts_dir=artifacts_dir,
        result_path=result_path,
        summary_path=summary_path,
        classification_path=classification_path,
        distribution_validation_path=distribution_validation_path,
        figure_path=figure_path,
        validation_path=validation_path,
        status_path=status_path,
        sanity_path=sanity_path,
        log_path=log_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        summary_df=summary_df,
        classification_df=classification_df,
        distribution_df=distribution_df,
        validation=validation,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        elapsed_seconds=elapsed_seconds,
        parameters=parameters,
    )
    report_path.write_text(report, encoding="utf-8")

    log_payload = {
        "generatedAtUtc": generated_at,
        "researchStepId": STEP_ID,
        "status": validation["status"],
        "success": bool(validation["success"]),
        "elapsedSeconds": elapsed_seconds,
        "parameters": parameters,
        "unitTests": unit_result,
    }
    write_json(log_path, log_payload)

    src_manifest = {
        "schema": "eidosoma.src_snapshot.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": generated_at,
        "gitCommit": git_commit,
        "gitStatusShort": git_status,
        "parameters": parameters,
        "sourceFiles": [
            artifact_entry(repo_dir / "scripts" / "e02_s10_input_distributions.py", repo_dir, "S10 runner script"),
            artifact_entry(repo_dir / "tests" / "e02" / "test_input_distributions.py", repo_dir, "S10 focused tests"),
            artifact_entry(repo_dir / "src" / "e02" / "deterministic_simulator.py", repo_dir, "S01 deterministic simulator"),
            artifact_entry(repo_dir / "scripts" / "e02_s09_alternative_metrics.py", repo_dir, "S09 repaired metric helpers"),
        ],
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "newDependenciesInstalled": [],
        },
        "validation": validation,
    }
    write_json(src_manifest_path, src_manifest)

    manifest_entries = [artifact_entry(path, artifacts_dir, description) for path, description in planned_artifacts if path.exists()]
    manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": validation["status"],
        "createdAtUtc": generated_at,
        "artifacts": manifest_entries,
    }
    write_json(manifest_path, manifest)
    manifest["artifacts"] = [artifact_entry(path, artifacts_dir, description) for path, description in planned_artifacts if path.exists()]
    write_json(manifest_path, manifest)

    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
