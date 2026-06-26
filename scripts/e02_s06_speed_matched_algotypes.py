#!/usr/bin/env python3
"""Execute E02 S06 speed-matched Algotype controls.

S06 calibrates pure-policy movement speed on the same initial arrays used for
same-goal chimeras, then uses inverse pure swap-per-activation rates as
scheduler weights for mixed Algotype runs. Local policy code is unchanged; only
activation order is throttled to test whether faster policies moving earlier
explain Aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from e02_deterministic_simulator import (  # noqa: E402
    DeterministicEventSimulator,
    aggregation,
    initial_values_from_seed,
    monotonicity_error,
    sortedness_percent,
    state_hash,
)


EXPERIMENT_ID = "E02"
STEP_ID = "S06"
STEP_NUMBER = 6
DEFAULT_E01_ARTIFACTS = Path("/previous-artifacts/E01")
BASE_POLICIES = ["bubble", "insertion", "selection"]
CHIMERIC_MIXTURES = [
    "bubble_insertion",
    "bubble_selection",
    "insertion_selection",
    "bubble_insertion_selection",
]
PROFILE_EQUAL = "equal_activation_reference"
PROFILE_SPEED = "speed_matched_inverse_pure_swap_rate"
PROFILE_IDS = [PROFILE_EQUAL, PROFILE_SPEED]
ALGOTYPE_TO_CODE = {"bubble": 0, "insertion": 1, "selection": 2}
CODE_TO_ALGOTYPE = {value: key for key, value in ALGOTYPE_TO_CODE.items()}
POLICY_LOGIC_ID = "DeterministicEventSimulator:S01_policy_semantics"


@dataclass
class CommandResult:
    args: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    ok: bool


@dataclass
class PureCalibration:
    policy: str
    completed: bool
    stop_reason: str
    swap_count: int
    activation_count: int
    comparison_count: int
    final_sortedness_percent: float
    swap_rate_per_activation: float
    comparison_rate_per_activation: float
    wall_time_seconds: float


@dataclass
class SpeedRunResult:
    condition: dict[str, Any]
    seed_row: dict[str, Any]
    n: int
    mixture_id: str
    speed_profile_id: str
    speed_profile_family: str
    speed_weights_by_algotype: dict[str, float]
    pure_calibration_by_algotype: dict[str, PureCalibration]
    initial_values: list[int]
    final_values: list[int]
    initial_algotypes: list[str]
    final_algotypes: list[str]
    completed: bool
    stop_reason: str
    swap_count: int
    comparison_count: int
    archived_compare_and_swap_count: int
    activation_count: int
    event_count: int
    wall_time_seconds: float
    trace_rows: list[dict[str, Any]]
    history_hash: str
    observed_curve: np.ndarray
    activation_counts_by_algotype: dict[str, int]
    activation_shares_by_algotype: dict[str, float]
    target_activation_shares_by_algotype: dict[str, float]
    max_activation_share_abs_error: float
    activation_share_tolerance: float
    actor_swap_counts_by_algotype: dict[str, int]
    actor_swap_shares_by_algotype: dict[str, float]
    cell_move_counts_by_algotype: dict[str, int]
    cell_move_shares_by_algotype: dict[str, float]
    target_movement_shares_by_algotype: dict[str, float]
    max_actor_swap_share_abs_error: float
    max_cell_move_share_abs_error: float
    movement_share_tolerance: float
    dg_max_drop_percent: float
    dg_decrease_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return CommandResult(args, proc.returncode, proc.stdout, proc.stderr, proc.returncode == 0)
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return CommandResult(args, None, "", repr(exc), False)


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit.stdout.strip() if commit.ok else "unknown",
        "branch": branch.stdout.strip() if branch.ok else "unknown",
        "dirtyStatus": status.stdout.strip(),
        "remote": remote.stdout.strip(),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def sha256_json(value: Any) -> str:
    payload = json.dumps(json_ready(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), separators=(",", ":"))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            if abs(value) >= 1000 or (0 < abs(value) < 0.001):
                return f"{value:.3g}"
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def scaled_allocation(mixture_id: str, n: int) -> dict[str, int]:
    if mixture_id == "bubble_insertion":
        return {"bubble": n // 2, "insertion": n - n // 2}
    if mixture_id == "bubble_selection":
        return {"bubble": n // 2, "selection": n - n // 2}
    if mixture_id == "insertion_selection":
        return {"insertion": n // 2, "selection": n - n // 2}
    if mixture_id == "bubble_insertion_selection":
        base = n // 3
        return {
            "bubble": base + (1 if n % 3 > 0 else 0),
            "insertion": base + (1 if n % 3 > 1 else 0),
            "selection": base,
        }
    raise ValueError(f"unsupported S06 chimeric mixture: {mixture_id}")


def algotypes_from_seed(mixture_id: str, n: int, seed: Any) -> list[str]:
    allocation = scaled_allocation(mixture_id, n)
    algotypes = [algorithm for algorithm, count in allocation.items() for _ in range(count)]
    if len(algotypes) != n:
        raise ValueError(f"allocation for {mixture_id} gives {len(algotypes)} cells, expected {n}")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(algotypes)
    return algotypes


def same_goal_condition_id(mixture_id: str) -> str:
    return f"S09_cell_view_{mixture_id}_unique_same_goal"


def load_s06_inputs(e01_artifacts: Path, *, n: int, replicate_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_ids = [same_goal_condition_id(mixture) for mixture in CHIMERIC_MIXTURES]
    condition_matrix = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "condition_matrix.csv")
    seed_table = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "seed_table.csv")
    conditions = condition_matrix[condition_matrix["conditionId"].isin(condition_ids)].copy()
    conditions["s06N"] = int(n)
    conditions = conditions.sort_values("conditionId").reset_index(drop=True)
    seeds = seed_table[
        (seed_table["conditionId"].isin(condition_ids))
        & (seed_table["replicateIndex"] < int(replicate_count))
    ].copy()
    seeds = seeds.sort_values(["conditionId", "replicateIndex"]).reset_index(drop=True)
    if len(conditions) != len(condition_ids):
        missing = sorted(set(condition_ids) - set(conditions["conditionId"]))
        raise ValueError(f"Missing S06 same-goal chimeric conditions in E01 matrix: {missing}")
    expected_seed_rows = len(condition_ids) * int(replicate_count)
    if len(seeds) != expected_seed_rows:
        raise ValueError(f"Expected {expected_seed_rows} S06 seed rows, found {len(seeds)}")
    return conditions, seeds


def policy_signature() -> str:
    return sha256_path(REPO_ROOT / "e02_deterministic_simulator" / "simulator.py")


def current_cell_ids(sim: DeterministicEventSimulator) -> list[int]:
    return [int(cell.cell_id) for cell in sim.cells]


def normalized_auc(curve: np.ndarray) -> float:
    if len(curve) == 0:
        return float("nan")
    if len(curve) == 1:
        return float(curve[0])
    return float(np.trapezoid(curve.astype(float), dx=1.0) / (len(curve) - 1))


def empirical_p_high(observed: float, null_values: np.ndarray | list[float]) -> float:
    values = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(values >= float(observed))) / (len(values) + 1))


def z_effect(observed: float, null_values: np.ndarray | list[float]) -> float:
    values = np.asarray(null_values, dtype=float)
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if sd <= 0:
        return 0.0 if math.isclose(float(observed), float(np.mean(values))) else float("inf")
    return float((float(observed) - float(np.mean(values))) / sd)


def trajectory_history_hash(trace_rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "eventIndex": int(row["event_index"]),
            "eventKind": row["event_kind"],
            "activationIndex": int(row["activation_index"]),
            "actorCellId": None if pd.isna(row["actor_cell_id"]) else int(row["actor_cell_id"]),
            "targetPosition": None if pd.isna(row["target_position"]) else int(row["target_position"]),
            "swapCount": int(row["swap_count"]),
            "comparisonCount": int(row["comparison_count"]),
            "stateHash": row["state_hash"],
            "values": json.loads(row["values_json"]),
            "cellIds": json.loads(row["cell_ids_json"]),
            "algotypes": json.loads(row["algotypes_json"]),
        }
        for row in trace_rows
    ]
    return sha256_json(payload)


def trajectory_diagnostics(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sortedness_values = [float(row["sortedness_percent"]) for row in trace_rows]
    aggregations = [float(row["observedAggregation"]) for row in trace_rows]
    running_best = -math.inf
    max_drop = 0.0
    decrease_count = 0
    previous = None
    for value in sortedness_values:
        running_best = max(running_best, value)
        max_drop = max(max_drop, running_best - value)
        if previous is not None and value < previous:
            decrease_count += 1
        previous = value
    return {
        "dgMaxDropPercent": float(max_drop),
        "dgDecreaseCount": int(decrease_count),
        "initialAggregation": aggregations[0] if aggregations else None,
        "peakAggregation": max(aggregations) if aggregations else None,
    }


def codes_for_algotypes(algotypes: list[str]) -> np.ndarray:
    return np.asarray([ALGOTYPE_TO_CODE[item] for item in algotypes], dtype=np.int16)


def aggregation_curve_for_assignment(cell_ids_by_event: np.ndarray, labels_by_cell_id: np.ndarray) -> np.ndarray:
    labels_by_position = labels_by_cell_id[cell_ids_by_event]
    if labels_by_position.shape[1] < 2:
        return np.ones(labels_by_position.shape[0], dtype=float)
    return np.mean(labels_by_position[:, :-1] == labels_by_position[:, 1:], axis=1).astype(float)


def share_dict(counts: dict[str, int], keys: list[str]) -> dict[str, float]:
    total = sum(int(counts.get(key, 0)) for key in keys)
    if total <= 0:
        return {key: 0.0 for key in keys}
    return {key: int(counts.get(key, 0)) / total for key in keys}


def max_share_error(realized: dict[str, float], target: dict[str, float]) -> float:
    return max(abs(float(realized.get(key, 0.0)) - float(target.get(key, 0.0))) for key in target) if target else 0.0


def movement_share_tolerance(total_cell_moves: int) -> float:
    return max(0.15, 4.0 / math.sqrt(max(1, int(total_cell_moves))))


def activation_share_tolerance(activation_count: int) -> float:
    return max(0.08, 4.0 / math.sqrt(max(1, int(activation_count))))


def target_activation_shares(initial_algotypes: list[str], weights_by_algotype: dict[str, float]) -> dict[str, float]:
    counts = Counter(initial_algotypes)
    weighted = {algotype: counts[algotype] * float(weights_by_algotype[algotype]) for algotype in counts}
    denom = sum(weighted.values())
    if denom <= 0:
        raise ValueError("speed-match weights must have positive total")
    return {algotype: weighted[algotype] / denom for algotype in sorted(weighted)}


def target_movement_shares(initial_algotypes: list[str]) -> dict[str, float]:
    counts = Counter(initial_algotypes)
    n = len(initial_algotypes)
    return {algotype: counts[algotype] / n for algotype in sorted(counts)}


def pure_calibration_table(
    *,
    initial_values: list[int],
    scheduler_seed: int,
    tie_breaker_seed: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> dict[str, PureCalibration]:
    calibrations: dict[str, PureCalibration] = {}
    for policy in BASE_POLICIES:
        started_at = time.perf_counter()
        sim = DeterministicEventSimulator(
            initial_values,
            [policy] * len(initial_values),
            scheduler_seed=scheduler_seed,
            tie_breaker_seed=tie_breaker_seed,
            condition_id=f"S06_pure_calibration_{policy}",
            research_step_id=STEP_ID,
        )
        result = sim.run(
            max_activations=max_activations,
            max_swaps=max_swaps,
            max_comparisons=max_comparisons,
        )
        activation_count = max(1, int(result.activation_count))
        calibrations[policy] = PureCalibration(
            policy=policy,
            completed=bool(result.completed),
            stop_reason=result.stop_reason,
            swap_count=int(result.swap_count),
            activation_count=int(result.activation_count),
            comparison_count=int(result.comparison_count),
            final_sortedness_percent=float(result.final_sortedness_percent),
            swap_rate_per_activation=float(result.swap_count / activation_count),
            comparison_rate_per_activation=float(result.comparison_count / activation_count),
            wall_time_seconds=float(time.perf_counter() - started_at),
        )
    return calibrations


def speed_match_weights(
    calibrations: dict[str, PureCalibration],
    algotypes: list[str],
    *,
    rate_floor: float = 1.0e-6,
) -> dict[str, float]:
    raw = {}
    for algotype in sorted(set(algotypes)):
        rate = max(float(calibrations[algotype].swap_rate_per_activation), rate_floor)
        raw[algotype] = 1.0 / rate
    min_raw = min(raw.values())
    return {algotype: raw[algotype] / min_raw for algotype in sorted(raw)}


def choose_weighted_cell_id(
    sim: DeterministicEventSimulator,
    weights_by_algotype: dict[str, float],
    rng: np.random.Generator,
) -> int | None:
    eligible = sim.eligible_cell_ids()
    if not eligible:
        return None
    weights = np.asarray(
        [float(weights_by_algotype[sim.cells[sim.positions_by_id[cell_id]].algotype]) for cell_id in eligible],
        dtype=float,
    )
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("scheduler weights must be nonnegative and have positive total")
    if np.allclose(weights, weights[0]):
        return int(rng.choice(eligible))
    return int(rng.choice(eligible, p=weights / weights.sum()))


def snapshot_trace_row(
    sim: DeterministicEventSimulator,
    base_row: dict[str, Any],
    *,
    speed_profile_id: str,
    speed_profile_family: str,
    speed_weights_by_algotype: dict[str, float],
) -> dict[str, Any]:
    values = sim.current_values()
    algotypes = sim.current_algotypes()
    row = dict(base_row)
    row["research_step_id"] = STEP_ID
    row["policyLogicId"] = POLICY_LOGIC_ID
    row["policySignatureSha256"] = policy_signature()
    row["speedProfileId"] = speed_profile_id
    row["speedProfileFamily"] = speed_profile_family
    row["speedWeightsByAlgotype"] = compact_json(speed_weights_by_algotype)
    row["values_json"] = compact_json(values)
    row["cell_ids_json"] = compact_json(current_cell_ids(sim))
    row["observedAggregation"] = aggregation(algorithms := algotypes)
    row["observedAlgotypeCountsJson"] = compact_json(dict(sorted(Counter(algorithms).items())))
    row["values_hash_from_json"] = state_hash(values)
    return row


def validate_fixed_trajectory(
    trace_rows: list[dict[str, Any]],
    *,
    initial_values: list[int],
    initial_algotypes: list[str],
) -> list[str]:
    failures: list[str] = []
    expected_ids = list(range(len(initial_values)))
    expected_counts = Counter(initial_algotypes)
    for row in trace_rows:
        event_index = int(row["event_index"])
        values = json.loads(row["values_json"])
        cell_ids = json.loads(row["cell_ids_json"])
        algotypes = json.loads(row["algotypes_json"])
        if sorted(cell_ids) != expected_ids:
            failures.append(f"event {event_index}: cell IDs are not a position permutation")
        if values != [initial_values[int(cell_id)] for cell_id in cell_ids]:
            failures.append(f"event {event_index}: values do not match fixed cell identity history")
        if algotypes != [initial_algotypes[int(cell_id)] for cell_id in cell_ids]:
            failures.append(f"event {event_index}: Algotypes do not follow fixed cell identities")
        if Counter(algotypes) != expected_counts:
            failures.append(f"event {event_index}: Algotype counts changed")
        if row["state_hash"] != state_hash(values):
            failures.append(f"event {event_index}: state hash does not match values_json")
    return failures


def run_speed_profile(
    *,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    n: int,
    mixture_id: str,
    initial_values: list[int],
    initial_algotypes: list[str],
    calibrations: dict[str, PureCalibration],
    speed_profile_id: str,
    speed_profile_family: str,
    speed_weights_by_algotype: dict[str, float],
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int = 2,
) -> SpeedRunResult:
    started_at = time.perf_counter()
    sim = DeterministicEventSimulator(
        initial_values,
        initial_algotypes,
        scheduler_seed=int(seed_row["schedulerSeed"]),
        tie_breaker_seed=int(seed_row["tieBreakerSeed"]),
        condition_id=f"S06_{mixture_id}_{speed_profile_id}",
        research_step_id=STEP_ID,
    )
    rng = sim.scheduler_rng
    trace_rows = [
        snapshot_trace_row(
            sim,
            sim.trace_rows[-1],
            speed_profile_id=speed_profile_id,
            speed_profile_family=speed_profile_family,
            speed_weights_by_algotype=speed_weights_by_algotype,
        )
    ]
    activation_counts: Counter[str] = Counter()
    actor_swap_counts: Counter[str] = Counter()
    cell_move_counts: Counter[str] = Counter()
    stop_reason = "sorted"
    no_move_checks = 0
    interval = max(1, len(initial_values))
    while True:
        if sim.is_sorted():
            stop_reason = "sorted"
            break
        if sim.activation_count >= max_activations:
            stop_reason = "max_activation_cap"
            break
        if sim.swap_count >= max_swaps:
            stop_reason = "max_step_cap"
            break
        if sim.comparison_count >= max_comparisons:
            stop_reason = "max_comparison_cap"
            break
        if sim.activation_count % interval == 0:
            if not sim.legal_action_exists():
                no_move_checks += 1
                if no_move_checks >= no_move_checks_required:
                    stop_reason = "no_cell_can_move_after_two_checks"
                    break
            else:
                no_move_checks = 0

        cell_id = choose_weighted_cell_id(sim, speed_weights_by_algotype, rng)
        if cell_id is None:
            stop_reason = "no_eligible_cells"
            break
        before_cell_ids = current_cell_ids(sim)
        before_algotype_by_cell_id = {int(cell.cell_id): cell.algotype for cell in sim.cells}
        actor_algotype = before_algotype_by_cell_id[int(cell_id)]
        outcome = sim.step(forced_cell_id=cell_id)
        if outcome.activated:
            activation_counts[actor_algotype] += 1
        if outcome.swapped:
            actor_swap_counts[actor_algotype] += 1
            after_cell_ids = current_cell_ids(sim)
            for before_cell_id, after_cell_id in zip(before_cell_ids, after_cell_ids):
                if int(before_cell_id) != int(after_cell_id):
                    cell_move_counts[before_algotype_by_cell_id[int(before_cell_id)]] += 1
            trace_rows.append(
                snapshot_trace_row(
                    sim,
                    sim.trace_rows[-1],
                    speed_profile_id=speed_profile_id,
                    speed_profile_family=speed_profile_family,
                    speed_weights_by_algotype=speed_weights_by_algotype,
                )
            )

    validation_failures = validate_fixed_trajectory(
        trace_rows,
        initial_values=initial_values,
        initial_algotypes=initial_algotypes,
    )
    if validation_failures:
        raise ValueError("; ".join(validation_failures[:5]))

    keys = sorted(set(initial_algotypes))
    target_activation = target_activation_shares(initial_algotypes, speed_weights_by_algotype)
    activation_shares = share_dict(dict(activation_counts), keys)
    target_movement = target_movement_shares(initial_algotypes)
    actor_swap_shares = share_dict(dict(actor_swap_counts), keys)
    cell_move_shares = share_dict(dict(cell_move_counts), keys)
    activation_error = max_share_error(activation_shares, target_activation)
    actor_swap_error = max_share_error(actor_swap_shares, target_movement)
    cell_move_error = max_share_error(cell_move_shares, target_movement)
    move_tolerance = movement_share_tolerance(sum(cell_move_counts.values()))
    observed_curve = np.asarray([float(row["observedAggregation"]) for row in trace_rows], dtype=float)
    diagnostics = trajectory_diagnostics(trace_rows)
    return SpeedRunResult(
        condition=condition,
        seed_row=seed_row,
        n=int(n),
        mixture_id=mixture_id,
        speed_profile_id=speed_profile_id,
        speed_profile_family=speed_profile_family,
        speed_weights_by_algotype=dict(sorted(speed_weights_by_algotype.items())),
        pure_calibration_by_algotype=calibrations,
        initial_values=list(initial_values),
        final_values=sim.current_values(),
        initial_algotypes=list(initial_algotypes),
        final_algotypes=sim.current_algotypes(),
        completed=sim.is_sorted(),
        stop_reason=stop_reason,
        swap_count=int(sim.swap_count),
        comparison_count=int(sim.comparison_count),
        archived_compare_and_swap_count=int(sim.archived_compare_and_swap_count),
        activation_count=int(sim.activation_count),
        event_count=len(trace_rows),
        wall_time_seconds=float(time.perf_counter() - started_at),
        trace_rows=trace_rows,
        history_hash=trajectory_history_hash(trace_rows),
        observed_curve=observed_curve,
        activation_counts_by_algotype=dict(sorted(activation_counts.items())),
        activation_shares_by_algotype=activation_shares,
        target_activation_shares_by_algotype=target_activation,
        max_activation_share_abs_error=float(activation_error),
        activation_share_tolerance=activation_share_tolerance(sim.activation_count),
        actor_swap_counts_by_algotype=dict(sorted(actor_swap_counts.items())),
        actor_swap_shares_by_algotype=actor_swap_shares,
        cell_move_counts_by_algotype=dict(sorted(cell_move_counts.items())),
        cell_move_shares_by_algotype=cell_move_shares,
        target_movement_shares_by_algotype=target_movement,
        max_actor_swap_share_abs_error=float(actor_swap_error),
        max_cell_move_share_abs_error=float(cell_move_error),
        movement_share_tolerance=float(move_tolerance),
        dg_max_drop_percent=float(diagnostics["dgMaxDropPercent"]),
        dg_decrease_count=int(diagnostics["dgDecreaseCount"]),
    )


def run_label_shuffle_nulls(
    result: SpeedRunResult,
    *,
    null_replicates: int,
    null_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(null_seed))
    cell_ids_by_event = np.asarray([json.loads(row["cell_ids_json"]) for row in result.trace_rows], dtype=np.int16)
    initial_codes = codes_for_algotypes(result.initial_algotypes)
    observed = result.observed_curve
    null_curves = np.empty((int(null_replicates), len(observed)), dtype=np.float32)
    null_rows: list[dict[str, Any]] = []
    label_count_failures = 0
    expected_codes = Counter(initial_codes.tolist())
    expected_labels = Counter(result.initial_algotypes)
    for null_index in range(int(null_replicates)):
        labels_by_cell_id = np.array(initial_codes, copy=True)
        rng.shuffle(labels_by_cell_id)
        if Counter(labels_by_cell_id.tolist()) != expected_codes:
            label_count_failures += 1
        label_counts = Counter(CODE_TO_ALGOTYPE[int(code)] for code in labels_by_cell_id.tolist())
        if label_counts != expected_labels:
            label_count_failures += 1
        curve = aggregation_curve_for_assignment(cell_ids_by_event, labels_by_cell_id)
        null_curves[null_index, :] = curve.astype(np.float32)
        null_rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "conditionId": result.condition["conditionId"],
                "mixtureId": result.mixture_id,
                "speedProfileId": result.speed_profile_id,
                "replicateIndex": int(result.seed_row["replicateIndex"]),
                "replicateNumber": int(result.seed_row["replicateNumber"]),
                "nullReplicateIndex": int(null_index),
                "nullSeed": int(null_seed),
                "trajectoryHistoryHash": result.history_hash,
                "labelAssignmentHash": sha256_json(labels_by_cell_id.tolist()),
                "labelCounts": compact_json(dict(sorted(label_counts.items()))),
                "nullPeakAggregation": float(np.max(curve)),
                "nullPeakEventIndex": int(np.argmax(curve)),
                "nullFinalAggregation": float(curve[-1]),
                "nullAucAggregation": normalized_auc(curve),
            }
        )

    timepoint_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(result.trace_rows):
        null_values = null_curves[:, idx].astype(float)
        observed_value = float(observed[idx])
        timepoint_rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "conditionId": result.condition["conditionId"],
                "mixtureId": result.mixture_id,
                "speedProfileId": result.speed_profile_id,
                "replicateIndex": int(result.seed_row["replicateIndex"]),
                "replicateNumber": int(result.seed_row["replicateNumber"]),
                "eventIndex": int(row["event_index"]),
                "activationIndex": int(row["activation_index"]),
                "swapCount": int(row["swap_count"]),
                "stateHash": row["state_hash"],
                "trajectoryHistoryHash": result.history_hash,
                "observedAggregation": observed_value,
                "nullMeanAggregation": float(np.mean(null_values)),
                "nullSdAggregation": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0,
                "nullQ025Aggregation": float(np.quantile(null_values, 0.025)),
                "nullQ50Aggregation": float(np.quantile(null_values, 0.5)),
                "nullQ975Aggregation": float(np.quantile(null_values, 0.975)),
                "empiricalPHighAtTimepoint": empirical_p_high(observed_value, null_values),
            }
        )
    validation = {
        "labelCountFailures": int(label_count_failures),
        "cellIdPermutationRows": int(np.sum([sorted(row.tolist()) == list(range(result.n)) for row in cell_ids_by_event])),
        "cellIdRows": int(len(cell_ids_by_event)),
        "nullReplicates": int(null_replicates),
    }
    return pd.DataFrame(null_rows), pd.DataFrame(timepoint_rows), validation


def calibration_json(calibrations: dict[str, PureCalibration], algotypes: list[str]) -> str:
    payload = {
        policy: {
            "completed": calibrations[policy].completed,
            "stopReason": calibrations[policy].stop_reason,
            "swapCount": calibrations[policy].swap_count,
            "activationCount": calibrations[policy].activation_count,
            "comparisonCount": calibrations[policy].comparison_count,
            "swapRatePerActivation": calibrations[policy].swap_rate_per_activation,
            "comparisonRatePerActivation": calibrations[policy].comparison_rate_per_activation,
            "finalSortednessPercent": calibrations[policy].final_sortedness_percent,
        }
        for policy in sorted(set(algotypes))
    }
    return compact_json(payload)


def make_observed_record(result: SpeedRunResult, null_df: pd.DataFrame, validation: dict[str, Any]) -> dict[str, Any]:
    observed_curve = result.observed_curve
    null_peak = null_df["nullPeakAggregation"].to_numpy(dtype=float)
    null_auc = null_df["nullAucAggregation"].to_numpy(dtype=float)
    null_final = null_df["nullFinalAggregation"].to_numpy(dtype=float)
    observed_peak = float(np.max(observed_curve))
    observed_auc = normalized_auc(observed_curve)
    observed_final = float(observed_curve[-1])
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": result.condition["conditionId"],
        "s06ConditionId": f"S06_{result.mixture_id}_{result.speed_profile_id}",
        "mixtureId": result.mixture_id,
        "speedProfileId": result.speed_profile_id,
        "speedProfileFamily": result.speed_profile_family,
        "policyLogicId": POLICY_LOGIC_ID,
        "policySignatureSha256": policy_signature(),
        "n": int(result.n),
        "replicateIndex": int(result.seed_row["replicateIndex"]),
        "replicateNumber": int(result.seed_row["replicateNumber"]),
        "inputPermutationSeed": int(result.seed_row["inputPermutationSeed"]),
        "algotypeAssignmentSeed": int(result.seed_row["algotypeAssignmentSeed"]),
        "schedulerSeed": int(result.seed_row["schedulerSeed"]),
        "tieBreakerSeed": int(result.seed_row["tieBreakerSeed"]),
        "speedWeightsByAlgotype": compact_json(result.speed_weights_by_algotype),
        "pureCalibrationByAlgotype": calibration_json(result.pure_calibration_by_algotype, result.initial_algotypes),
        "targetActivationShares": compact_json(result.target_activation_shares_by_algotype),
        "realizedActivationCounts": compact_json(result.activation_counts_by_algotype),
        "realizedActivationShares": compact_json(result.activation_shares_by_algotype),
        "maxActivationShareAbsError": float(result.max_activation_share_abs_error),
        "activationShareTolerance": float(result.activation_share_tolerance),
        "activationShareWithinTolerance": bool(result.max_activation_share_abs_error <= result.activation_share_tolerance),
        "targetMovementShares": compact_json(result.target_movement_shares_by_algotype),
        "actorSwapCountsByAlgotype": compact_json(result.actor_swap_counts_by_algotype),
        "actorSwapSharesByAlgotype": compact_json(result.actor_swap_shares_by_algotype),
        "cellMoveCountsByAlgotype": compact_json(result.cell_move_counts_by_algotype),
        "cellMoveSharesByAlgotype": compact_json(result.cell_move_shares_by_algotype),
        "maxActorSwapShareAbsError": float(result.max_actor_swap_share_abs_error),
        "maxCellMoveShareAbsError": float(result.max_cell_move_share_abs_error),
        "movementShareTolerance": float(result.movement_share_tolerance),
        "cellMoveShareWithinTolerance": bool(result.max_cell_move_share_abs_error <= result.movement_share_tolerance),
        "trajectoryHistoryHash": result.history_hash,
        "initialValuesHash": state_hash(result.initial_values),
        "finalValuesHash": state_hash(result.final_values),
        "initialValues": compact_json(result.initial_values),
        "finalValues": compact_json(result.final_values),
        "initialAlgotypes": compact_json(result.initial_algotypes),
        "finalAlgotypes": compact_json(result.final_algotypes),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "archivedCompareAndSwapCount": int(result.archived_compare_and_swap_count),
        "activationCount": int(result.activation_count),
        "eventCount": int(result.event_count),
        "finalSortednessPercent": sortedness_percent(result.final_values),
        "finalMonotonicityError": monotonicity_error(result.final_values),
        "observedInitialAggregation": float(observed_curve[0]),
        "observedPeakAggregation": observed_peak,
        "observedPeakEventIndex": int(np.argmax(observed_curve)),
        "observedFinalAggregation": observed_final,
        "observedAucAggregation": observed_auc,
        "nullReplicateCount": int(len(null_df)),
        "nullPeakMean": float(np.mean(null_peak)),
        "nullPeakSd": float(np.std(null_peak, ddof=1)),
        "nullPeakQ025": float(np.quantile(null_peak, 0.025)),
        "nullPeakQ50": float(np.quantile(null_peak, 0.5)),
        "nullPeakQ975": float(np.quantile(null_peak, 0.975)),
        "nullAucMean": float(np.mean(null_auc)),
        "nullAucSd": float(np.std(null_auc, ddof=1)),
        "nullAucQ025": float(np.quantile(null_auc, 0.025)),
        "nullAucQ50": float(np.quantile(null_auc, 0.5)),
        "nullAucQ975": float(np.quantile(null_auc, 0.975)),
        "nullFinalMean": float(np.mean(null_final)),
        "nullFinalQ025": float(np.quantile(null_final, 0.025)),
        "nullFinalQ975": float(np.quantile(null_final, 0.975)),
        "empiricalPHighPeak": empirical_p_high(observed_peak, null_peak),
        "empiricalPHighAuc": empirical_p_high(observed_auc, null_auc),
        "empiricalPHighFinal": empirical_p_high(observed_final, null_final),
        "zPeakVsNull": z_effect(observed_peak, null_peak),
        "zAucVsNull": z_effect(observed_auc, null_auc),
        "zFinalVsNull": z_effect(observed_final, null_final),
        "dgMaxDropPercent": float(result.dg_max_drop_percent),
        "dgDecreaseCount": int(result.dg_decrease_count),
        "wallTimeSeconds": float(result.wall_time_seconds),
        "nullLabelCountFailures": int(validation["labelCountFailures"]),
        "traceCellIdPermutationRows": int(validation["cellIdPermutationRows"]),
        "traceCellIdRows": int(validation["cellIdRows"]),
    }


def make_diagnostic_record(result: SpeedRunResult, validation: dict[str, Any]) -> dict[str, Any]:
    calibration_rates = {
        policy: result.pure_calibration_by_algotype[policy].swap_rate_per_activation
        for policy in sorted(set(result.initial_algotypes))
    }
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": result.condition["conditionId"],
        "mixtureId": result.mixture_id,
        "speedProfileId": result.speed_profile_id,
        "replicateIndex": int(result.seed_row["replicateIndex"]),
        "replicateNumber": int(result.seed_row["replicateNumber"]),
        "pureCalibrationRates": compact_json(calibration_rates),
        "pureCalibrationAllSorted": bool(
            all(result.pure_calibration_by_algotype[policy].completed for policy in sorted(set(result.initial_algotypes)))
        ),
        "speedWeightsByAlgotype": compact_json(result.speed_weights_by_algotype),
        "speedWeightsNonUniform": bool(len({round(value, 12) for value in result.speed_weights_by_algotype.values()}) > 1)
        if len(result.speed_weights_by_algotype) > 1
        else True,
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "activationShareWithinTolerance": bool(result.max_activation_share_abs_error <= result.activation_share_tolerance),
        "cellMoveShareWithinTolerance": bool(result.max_cell_move_share_abs_error <= result.movement_share_tolerance),
        "maxActivationShareAbsError": float(result.max_activation_share_abs_error),
        "activationShareTolerance": float(result.activation_share_tolerance),
        "maxActorSwapShareAbsError": float(result.max_actor_swap_share_abs_error),
        "maxCellMoveShareAbsError": float(result.max_cell_move_share_abs_error),
        "movementShareTolerance": float(result.movement_share_tolerance),
        "nullLabelCountFailures": int(validation["labelCountFailures"]),
        "traceCellIdPermutationRows": int(validation["cellIdPermutationRows"]),
        "traceCellIdRows": int(validation["cellIdRows"]),
        "trajectoryHistoryHash": result.history_hash,
    }


def annotate_trace_rows(result: SpeedRunResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in result.trace_rows:
        item = dict(row)
        item["experimentId"] = EXPERIMENT_ID
        item["researchStepId"] = STEP_ID
        item["conditionId"] = result.condition["conditionId"]
        item["mixtureId"] = result.mixture_id
        item["speedProfileId"] = result.speed_profile_id
        item["speedProfileFamily"] = result.speed_profile_family
        item["replicateIndex"] = int(result.seed_row["replicateIndex"])
        item["replicateNumber"] = int(result.seed_row["replicateNumber"])
        item["inputPermutationSeed"] = int(result.seed_row["inputPermutationSeed"])
        item["algotypeAssignmentSeed"] = int(result.seed_row["algotypeAssignmentSeed"])
        item["trajectoryHistoryHash"] = result.history_hash
        rows.append(item)
    return rows


def run_s06_matrix(
    *,
    e01_artifacts: Path,
    n: int,
    replicate_count: int,
    null_replicates: int,
    null_seed_base: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    conditions_df, seeds_df = load_s06_inputs(e01_artifacts, n=n, replicate_count=replicate_count)
    observed_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    null_frames: list[pd.DataFrame] = []
    timepoint_frames: list[pd.DataFrame] = []
    trace_rows: list[dict[str, Any]] = []
    validation_runs: list[dict[str, Any]] = []
    run_index = 0
    for condition_tuple in conditions_df.itertuples(index=False):
        condition = condition_tuple._asdict()
        mixture_id = str(condition["mixtureId"])
        condition_seeds = seeds_df[seeds_df["conditionId"] == condition["conditionId"]].copy()
        for seed_tuple in condition_seeds.itertuples(index=False):
            seed_row = seed_tuple._asdict()
            input_seed = int(seed_row["inputPermutationSeed"])
            algotype_seed = int(seed_row["algotypeAssignmentSeed"])
            initial_values = initial_values_from_seed(input_seed, n=n, profile=str(condition["inputProfile"]))
            initial_algotypes = algotypes_from_seed(mixture_id, n, algotype_seed)
            calibrations = pure_calibration_table(
                initial_values=initial_values,
                scheduler_seed=int(seed_row["schedulerSeed"]),
                tie_breaker_seed=int(seed_row["tieBreakerSeed"]),
                max_activations=max_activations,
                max_swaps=max_swaps,
                max_comparisons=max_comparisons,
            )
            profiles = [
                (PROFILE_EQUAL, "equal_activation_reference", {algotype: 1.0 for algotype in sorted(set(initial_algotypes))}),
                (PROFILE_SPEED, "speed_matched", speed_match_weights(calibrations, initial_algotypes)),
            ]
            for speed_profile_id, speed_profile_family, weights in profiles:
                result = run_speed_profile(
                    condition=condition,
                    seed_row=seed_row,
                    n=n,
                    mixture_id=mixture_id,
                    initial_values=initial_values,
                    initial_algotypes=initial_algotypes,
                    calibrations=calibrations,
                    speed_profile_id=speed_profile_id,
                    speed_profile_family=speed_profile_family,
                    speed_weights_by_algotype=weights,
                    max_activations=max_activations,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                )
                null_seed = int(null_seed_base + run_index * 1_000_003)
                null_df, timepoint_df, run_validation = run_label_shuffle_nulls(
                    result,
                    null_replicates=null_replicates,
                    null_seed=null_seed,
                )
                observed_rows.append(make_observed_record(result, null_df, run_validation))
                diagnostics_rows.append(make_diagnostic_record(result, run_validation))
                null_frames.append(null_df)
                timepoint_frames.append(timepoint_df)
                trace_rows.extend(annotate_trace_rows(result))
                validation_runs.append(
                    {
                        "conditionId": condition["conditionId"],
                        "mixtureId": mixture_id,
                        "speedProfileId": speed_profile_id,
                        "replicateIndex": int(seed_row["replicateIndex"]),
                        "eventCount": int(result.event_count),
                        "trajectoryHistoryHash": result.history_hash,
                        **run_validation,
                    }
                )
                run_index += 1
    observed_df = pd.DataFrame(observed_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    null_df = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    timepoint_df = pd.concat(timepoint_frames, ignore_index=True) if timepoint_frames else pd.DataFrame()
    trace_df = pd.DataFrame(trace_rows)
    validation = {
        "conditionCount": int(len(conditions_df)),
        "replicateCountPerCondition": int(replicate_count),
        "profileCountPerRun": int(len(PROFILE_IDS)),
        "expectedObservedRows": int(len(conditions_df) * replicate_count * len(PROFILE_IDS)),
        "expectedNullRows": int(len(conditions_df) * replicate_count * len(PROFILE_IDS) * null_replicates),
        "nullReplicatesPerRun": int(null_replicates),
        "runs": validation_runs,
    }
    return observed_df, diagnostics_df, null_df, timepoint_df, trace_df, validation


def summarize_speed_matched(observed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        observed_df.groupby(["mixtureId", "speedProfileId", "speedProfileFamily"], dropna=False)
        .agg(
            runCount=("conditionId", "count"),
            completedCount=("completed", "sum"),
            completionRate=("completed", "mean"),
            meanSwapCount=("swapCount", "mean"),
            meanActivationCount=("activationCount", "mean"),
            meanPeakAggregation=("observedPeakAggregation", "mean"),
            meanNullPeakMean=("nullPeakMean", "mean"),
            medianPHighPeak=("empiricalPHighPeak", "median"),
            minPHighPeak=("empiricalPHighPeak", "min"),
            meanAucAggregation=("observedAucAggregation", "mean"),
            meanNullAucMean=("nullAucMean", "mean"),
            medianPHighAuc=("empiricalPHighAuc", "median"),
            minPHighAuc=("empiricalPHighAuc", "min"),
            meanFinalAggregation=("observedFinalAggregation", "mean"),
            meanMaxActivationShareAbsError=("maxActivationShareAbsError", "mean"),
            maxActivationShareAbsError=("maxActivationShareAbsError", "max"),
            meanMaxActorSwapShareAbsError=("maxActorSwapShareAbsError", "mean"),
            maxActorSwapShareAbsError=("maxActorSwapShareAbsError", "max"),
            meanMaxCellMoveShareAbsError=("maxCellMoveShareAbsError", "mean"),
            maxCellMoveShareAbsError=("maxCellMoveShareAbsError", "max"),
            significantPeakRuns=("empiricalPHighPeak", lambda x: int(np.sum(np.asarray(x, dtype=float) <= 0.05))),
            significantAucRuns=("empiricalPHighAuc", lambda x: int(np.sum(np.asarray(x, dtype=float) <= 0.05))),
            meanDgMaxDropPercent=("dgMaxDropPercent", "mean"),
        )
        .reset_index()
    )
    grouped["meanPeakDeltaVsNull"] = grouped["meanPeakAggregation"] - grouped["meanNullPeakMean"]
    grouped["meanAucDeltaVsNull"] = grouped["meanAucAggregation"] - grouped["meanNullAucMean"]
    reference = grouped[grouped["speedProfileId"] == PROFILE_EQUAL][
        ["mixtureId", "meanPeakAggregation", "meanAucAggregation", "meanActivationCount", "meanSwapCount", "meanDgMaxDropPercent"]
    ].rename(
        columns={
            "meanPeakAggregation": "equalReferenceMeanPeakAggregation",
            "meanAucAggregation": "equalReferenceMeanAucAggregation",
            "meanActivationCount": "equalReferenceMeanActivationCount",
            "meanSwapCount": "equalReferenceMeanSwapCount",
            "meanDgMaxDropPercent": "equalReferenceMeanDgMaxDropPercent",
        }
    )
    grouped = grouped.merge(reference, on="mixtureId", how="left")
    grouped["deltaPeakAggregationVsEqualReference"] = grouped["meanPeakAggregation"] - grouped["equalReferenceMeanPeakAggregation"]
    grouped["deltaAucAggregationVsEqualReference"] = grouped["meanAucAggregation"] - grouped["equalReferenceMeanAucAggregation"]
    grouped["deltaActivationCountVsEqualReference"] = grouped["meanActivationCount"] - grouped["equalReferenceMeanActivationCount"]
    grouped["deltaSwapCountVsEqualReference"] = grouped["meanSwapCount"] - grouped["equalReferenceMeanSwapCount"]
    grouped["deltaDgMaxDropVsEqualReference"] = grouped["meanDgMaxDropPercent"] - grouped["equalReferenceMeanDgMaxDropPercent"]
    return grouped.sort_values(["mixtureId", "speedProfileId"]).reset_index(drop=True)


def context_comparison(summary_df: pd.DataFrame, artifacts_dir: Path) -> pd.DataFrame:
    s03_path = artifacts_dir / "results" / "e02_activation_rate_summary.parquet"
    s04_path = artifacts_dir / "results" / "e02_label_shuffle_mixture_summary.parquet"
    s03 = pd.read_parquet(s03_path) if s03_path.exists() else pd.DataFrame()
    s04 = pd.read_parquet(s04_path) if s04_path.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in summary_df.itertuples(index=False):
        s03_match = s03[(s03.get("mixtureId", pd.Series(dtype=str)) == row.mixtureId) & (s03.get("rateProfileId", pd.Series(dtype=str)) == "equalized_all_1x")]
        s04_match = s04[s04.get("mixtureId", pd.Series(dtype=str)) == row.mixtureId]
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "mixtureId": row.mixtureId,
                "speedProfileId": row.speedProfileId,
                "s06MeanPeakAggregation": float(row.meanPeakAggregation),
                "s06MedianPHighPeak": float(row.medianPHighPeak),
                "s06MeanAucAggregation": float(row.meanAucAggregation),
                "s06MedianPHighAuc": float(row.medianPHighAuc),
                "s06MeanMaxCellMoveShareAbsError": float(row.meanMaxCellMoveShareAbsError),
                "s03EqualizedMeanPeakAggregation": float(s03_match["meanPeakAggregation"].iloc[0]) if len(s03_match) else None,
                "s03EqualizedMeanActivationCount": float(s03_match["meanActivationCount"].iloc[0]) if len(s03_match) else None,
                "s04OriginalMeanPeakAggregation": float(s04_match["meanObservedPeakAggregation"].iloc[0]) if len(s04_match) else None,
                "s04OriginalMedianPHighPeak": float(s04_match["medianPHighPeak"].iloc[0]) if len(s04_match) else None,
                "interpretation": "S06 speed profile uses matched local policies and inverse pure-speed scheduler weights",
            }
        )
    return pd.DataFrame(rows)


def write_tables(
    observed_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    null_df: pd.DataFrame,
    timepoint_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    artifacts_dir: Path,
) -> dict[str, Path]:
    results_dir = artifacts_dir / "results"
    traces_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    summary_df = summarize_speed_matched(observed_df)
    context_df = context_comparison(summary_df, artifacts_dir)
    paths = {
        "observed_parquet": results_dir / "e02_speed_matched_algotypes.parquet",
        "observed_csv": results_dir / "e02_speed_matched_algotypes.csv",
        "summary_parquet": results_dir / "e02_speed_matched_summary.parquet",
        "summary_csv": results_dir / "e02_speed_matched_summary.csv",
        "diagnostics_parquet": results_dir / "e02_speed_matched_diagnostics.parquet",
        "diagnostics_csv": results_dir / "e02_speed_matched_diagnostics.csv",
        "nulls_parquet": results_dir / "e02_speed_matched_label_shuffle_nulls.parquet",
        "nulls_csv": results_dir / "e02_speed_matched_label_shuffle_nulls.csv",
        "timepoint_parquet": results_dir / "e02_speed_matched_timepoint_summary.parquet",
        "timepoint_csv": results_dir / "e02_speed_matched_timepoint_summary.csv",
        "context_parquet": results_dir / "e02_speed_matched_context_comparison.parquet",
        "context_csv": results_dir / "e02_speed_matched_context_comparison.csv",
        "trace_parquet": traces_dir / "e02_speed_matched_trace_events.parquet",
        "trace_csv_gz": traces_dir / "e02_speed_matched_trace_events.csv.gz",
    }
    observed_df.to_parquet(paths["observed_parquet"], index=False)
    observed_df.to_csv(paths["observed_csv"], index=False)
    summary_df.to_parquet(paths["summary_parquet"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    diagnostics_df.to_parquet(paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(paths["diagnostics_csv"], index=False)
    null_df.to_parquet(paths["nulls_parquet"], index=False)
    null_df.to_csv(paths["nulls_csv"], index=False)
    timepoint_df.to_parquet(paths["timepoint_parquet"], index=False)
    timepoint_df.to_csv(paths["timepoint_csv"], index=False)
    context_df.to_parquet(paths["context_parquet"], index=False)
    context_df.to_csv(paths["context_csv"], index=False)
    trace_df.to_parquet(paths["trace_parquet"], index=False)
    trace_df.to_csv(paths["trace_csv_gz"], index=False, compression="gzip")
    return paths


def plot_speed_matched_summary(summary_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    mixtures = CHIMERIC_MIXTURES
    x = np.arange(len(mixtures), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = {PROFILE_EQUAL: "#577590", PROFILE_SPEED: "#d97338"}
    labels = {PROFILE_EQUAL: "equal activation", PROFILE_SPEED: "speed matched"}
    for offset, profile in [(-0.18, PROFILE_EQUAL), (0.18, PROFILE_SPEED)]:
        subset = summary_df[summary_df["speedProfileId"] == profile].set_index("mixtureId").reindex(mixtures)
        axes[0].bar(x + offset, subset["meanPeakAggregation"].to_numpy(dtype=float), width=0.34, color=colors[profile], label=labels[profile])
        axes[1].bar(x + offset, subset["medianPHighPeak"].to_numpy(dtype=float), width=0.34, color=colors[profile], label=labels[profile])
        axes[2].bar(x + offset, subset["meanMaxCellMoveShareAbsError"].to_numpy(dtype=float), width=0.34, color=colors[profile], label=labels[profile])
    axes[0].set_ylabel("Mean peak Aggregation")
    axes[1].set_ylabel("Median high-tail p")
    axes[2].set_ylabel("Mean cell-move share error")
    axes[0].set_title("Peak Aggregation")
    axes[1].set_title("Label-shuffle null p")
    axes[2].set_title("Movement matching")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([item.replace("_", "\n") for item in mixtures], fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    png = figure_dir / "e02_speed_matched_algotypes_summary.png"
    pdf = figure_dir / "e02_speed_matched_algotypes_summary.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def validate_outputs(
    observed_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    null_df: pd.DataFrame,
    timepoint_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    validation: dict[str, Any],
) -> tuple[bool, list[str], list[str]]:
    checks: list[str] = []
    failures: list[str] = []
    if len(observed_df) == int(validation["expectedObservedRows"]):
        checks.append(f"Speed-matched observed row count matched expected {validation['expectedObservedRows']}.")
    else:
        failures.append(f"Observed rows {len(observed_df)} did not match expected {validation['expectedObservedRows']}.")
    if len(null_df) == int(validation["expectedNullRows"]):
        checks.append(f"Speed-matched label-shuffle null row count matched expected {validation['expectedNullRows']}.")
    else:
        failures.append(f"Null rows {len(null_df)} did not match expected {validation['expectedNullRows']}.")
    if len(timepoint_df) == len(trace_df):
        checks.append("Timepoint summaries align one-to-one with speed-matched trace rows.")
    else:
        failures.append("Timepoint summary rows do not align with trace rows.")
    seed_groups = observed_df.groupby(["mixtureId", "replicateIndex"])
    if all(group["initialValuesHash"].nunique() == 1 for _, group in seed_groups):
        checks.append("Initial value hashes are matched across speed profiles for every mixture/replicate.")
    else:
        failures.append("At least one mixture/replicate has mismatched initial values across speed profiles.")
    if all(group["initialAlgotypes"].nunique() == 1 for _, group in seed_groups):
        checks.append("Initial Algotype assignments are matched across speed profiles for every mixture/replicate.")
    else:
        failures.append("At least one mixture/replicate has mismatched initial Algotype assignments across speed profiles.")
    if bool(observed_df["completed"].all()) and set(observed_df["stopReason"]) == {"sorted"}:
        checks.append("All S06 speed-profile runs completed with sorted final states.")
    else:
        failures.append("At least one S06 speed-profile run did not complete sorted.")
    value_conserved = observed_df["finalValues"].apply(lambda text: Counter(json.loads(text))) == observed_df["initialValues"].apply(lambda text: Counter(json.loads(text)))
    if bool(value_conserved.all()):
        checks.append("All S06 runs conserved value multisets.")
    else:
        failures.append("At least one S06 run failed value-multiset conservation.")
    algotype_conserved = observed_df["finalAlgotypes"].apply(lambda text: Counter(json.loads(text))) == observed_df["initialAlgotypes"].apply(lambda text: Counter(json.loads(text)))
    if bool(algotype_conserved.all()):
        checks.append("All S06 runs conserved Algotype multisets.")
    else:
        failures.append("At least one S06 run failed Algotype-multiset conservation.")
    if observed_df["policySignatureSha256"].nunique() == 1:
        checks.append("Policy signature is unchanged across speed profiles.")
    else:
        failures.append("Policy signature differs across speed profiles.")
    if bool(observed_df["activationShareWithinTolerance"].all()) and bool(diagnostics_df["activationShareWithinTolerance"].all()):
        checks.append("Realized activation shares matched each profile's target scheduler distribution within tolerance.")
    else:
        failures.append("At least one speed-profile run missed activation-share tolerance.")
    speed_rows = observed_df[observed_df["speedProfileId"] == PROFILE_SPEED]
    speed_diag = diagnostics_df[diagnostics_df["speedProfileId"] == PROFILE_SPEED]
    if bool(speed_rows["cellMoveShareWithinTolerance"].all()) and bool(speed_diag["cellMoveShareWithinTolerance"].all()):
        checks.append("Speed-matched runs kept realized cell-move shares within the predeclared tolerance.")
    else:
        failures.append("At least one speed-matched run missed realized cell-move share tolerance.")
    if bool(speed_diag["pureCalibrationAllSorted"].all()):
        checks.append("Pure-policy calibration pretests sorted successfully for every speed-matched run.")
    else:
        failures.append("At least one pure-policy calibration pretest did not sort.")
    if bool(speed_diag["speedWeightsNonUniform"].all()):
        checks.append("Speed-matched profiles used non-uniform scheduler weights when multiple Algotypes were present.")
    else:
        failures.append("At least one speed-matched profile failed to apply non-uniform weights.")
    if all(int(run["labelCountFailures"]) == 0 for run in validation.get("runs", [])):
        checks.append("All speed-profile label-shuffle null assignments preserved Algotype label counts.")
    else:
        failures.append("At least one speed-profile null assignment changed label counts.")
    if all(int(run["cellIdPermutationRows"]) == int(run["cellIdRows"]) for run in validation.get("runs", [])):
        checks.append("All S06 trace rows preserved fixed cell-position permutations.")
    else:
        failures.append("At least one S06 trace row lost the fixed cell-position permutation.")
    if bool((trace_df["state_hash"] == trace_df["values_hash_from_json"]).all()):
        checks.append("Values JSON reproduces recorded state hashes for all S06 trace rows.")
    else:
        failures.append("At least one S06 trace row has values_json inconsistent with state_hash.")
    if set(observed_df["mixtureId"].unique()) == set(CHIMERIC_MIXTURES) and set(observed_df["speedProfileId"].unique()) == set(PROFILE_IDS):
        checks.append("All planned mixtures and speed profiles are represented.")
    else:
        failures.append("The S06 output is missing a planned mixture or speed profile.")
    return not failures, checks, failures


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    script_dst = code_dir / "scripts" / "e02_s06_speed_matched_algotypes.py"
    test_dst = code_dir / "tests" / "test_e02_speed_matched_algotypes.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "e02_s06_speed_matched_algotypes.py", script_dst)
    if (REPO_ROOT / "tests" / "test_e02_speed_matched_algotypes.py").exists():
        shutil.copy2(REPO_ROOT / "tests" / "test_e02_speed_matched_algotypes.py", test_dst)
    return [path for path in [script_dst, test_dst] if path.exists()]


def write_validation_report(step_dir: Path, checks: list[str], failures: list[str]) -> Path:
    path = step_dir / "validation_report.md"
    checks_text = "\n".join(f"- {item}" for item in checks) or "- None"
    failures_text = "\n".join(f"- {item}" for item in failures) or "- None"
    path.write_text(
        f"""# S06 Validation Report

- Research step ID: {STEP_ID}
- Completion status: {'completed' if not failures else 'completed with validation failures'}
- Artifacts written: speed-matched result tables, diagnostics, label-shuffle null tables, trace events, figures, copied code, manifests, and status files under `$ARTIFACTS_DIR`.
- Validation result: {'passed' if not failures else 'failed'}
- Caveats or blockers: speed matching is scheduler-level throttling based on pure-policy swap-per-activation calibration; it matches realized cell movement shares within tolerance but does not force exact actor-initiated swap parity.
- Recommended next action: proceed to S07 randomized local-move null models only after Chief Scientist review.

## Checks

{checks_text}

## Failures

{failures_text}
""",
        encoding="utf-8",
    )
    return path


def run_repo_tests(step_dir: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    result = run_command(command, cwd=REPO_ROOT, env=env)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + result.stdout + "\n\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    return {"command": command, "returnCode": result.returncode, "success": result.ok, "logPath": str(log_path)}


def outcome_classification(summary_df: pd.DataFrame, success: bool) -> str:
    if not success:
        return "constraining/contradictory"
    speed_three_way = summary_df[
        (summary_df["mixtureId"] == "bubble_insertion_selection")
        & (summary_df["speedProfileId"] == PROFILE_SPEED)
    ]
    if len(speed_three_way) and float(speed_three_way["medianPHighPeak"].iloc[0]) <= 0.05:
        return "supportive"
    return "constraining/contradictory"


def write_reports_and_manifests(
    *,
    artifacts_dir: Path,
    step_dir: Path,
    observed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    null_df: pd.DataFrame,
    checks: list[str],
    failures: list[str],
    validation: dict[str, Any],
    repo_test_payload: dict[str, Any],
    artifact_paths: list[Path],
    started_at: str,
    n: int,
    replicate_count: int,
    null_replicates: int,
    max_activations: int,
) -> tuple[Path, Path, Path]:
    success = not failures and bool(repo_test_payload["success"])
    classification = outcome_classification(summary_df, success)
    speed_summary = summary_df[summary_df["speedProfileId"] == PROFILE_SPEED]
    three_way = speed_summary[speed_summary["mixtureId"] == "bubble_insertion_selection"]
    three_way_peak_p = float(three_way["medianPHighPeak"].iloc[0]) if len(three_way) else None
    three_way_peak_delta = float(three_way["meanPeakDeltaVsNull"].iloc[0]) if len(three_way) else None
    max_cell_move_error = float(observed_df[observed_df["speedProfileId"] == PROFILE_SPEED]["maxCellMoveShareAbsError"].max())
    max_actor_swap_error = float(observed_df[observed_df["speedProfileId"] == PROFILE_SPEED]["maxActorSwapShareAbsError"].max())
    max_activation_error = float(observed_df["maxActivationShareAbsError"].max())
    validation_result = (
        "passed: pure-policy speed calibration sorted, mixed speed profiles sorted, movement shares matched tolerance, label nulls preserved counts, and repository tests passed"
        if success
        else "failed: see validation_report.md and status.json"
    )
    caveats = [
        "Speed matching is implemented as scheduler-level inverse pure swap-per-activation throttling; local policy code is unchanged.",
        "The matching target is realized cell-move share by Algotype, not exact actor-initiated swap parity.",
        "Actor-initiated swap shares remain diagnostic because local policies create different legal-move opportunities.",
        "Label-shuffle p-values are high-tail and unadjusted across the E02 family.",
        f"S06 uses deterministic n={n} bounded controls to match S02-S05 audit scale rather than full E01 n=100 sweeps.",
    ]
    recommended_next_action = "Proceed to S07 randomized local-move null models only after Chief Scientist review; carry S06 speed-matching caveats into Aggregation verdicts."
    if classification == "supportive":
        lay_tail = "the three-way speed-matched mixture still exceeded its trajectory-preserving label null for peak Aggregation."
    else:
        lay_tail = "speed matching narrowed the Aggregation evidence because the three-way speed-matched peak did not remain clearly above its label null."
    lay_summary = (
        "S06 calibrated pure Bubble, Insertion, and Selection movement speeds on matched input arrays, then ran equal-activation and inverse-speed-matched mixed controls. "
        "All mixed controls sorted and realized cell movement shares stayed within tolerance. "
        f"In this bounded matrix, {lay_tail}"
    )
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "provenance" / "run_manifest.json"
    reported_artifact_paths = [*artifact_paths, summary_path, status_path, manifest_path, run_manifest_path]
    reported_artifact_text_paths = sorted({str(path) for path in reported_artifact_paths})
    artifact_records = collect_artifacts(reported_artifact_paths)
    artifact_text = "\n".join(f"- `{path}`" for path in reported_artifact_text_paths)
    matrix_table = markdown_table(
        ["Check type", "Count"],
        [
            ["observed speed-profile rows", len(observed_df)],
            ["label-shuffle null rows", len(null_df)],
            ["mixtures", observed_df["mixtureId"].nunique()],
            ["speed profiles", observed_df["speedProfileId"].nunique()],
            ["replicates per condition", replicate_count],
            ["null replicates per run", null_replicates],
        ],
    )
    preview_table = markdown_table(
        ["Mixture", "Profile", "Runs", "Peak delta", "Median p peak", "AUC delta", "Move err.", "Sorted"],
        [
            [
                row.mixtureId,
                row.speedProfileId,
                int(row.runCount),
                row.meanPeakDeltaVsNull,
                row.medianPHighPeak,
                row.meanAucDeltaVsNull,
                row.meanMaxCellMoveShareAbsError,
                bool(row.completedCount == row.runCount),
            ]
            for row in summary_df.itertuples(index=False)
        ],
    )
    summary_path.write_text(
        f"""# E02 S06 Status Summary

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {'completed' if success else 'completed with validation failures'}
- Outcome classification: {classification}
- Artifacts written:
{artifact_text}
- Validation result: {validation_result}
- Caveats or blockers: {' '.join(caveats)}
- Lay summary: {lay_summary}
- Recommended next action: {recommended_next_action}

## Run Matrix

{matrix_table}

## Speed-Match Preview

{preview_table}

## Anchor Notes

- Maximum speed-matched cell-move share absolute error: {max_cell_move_error}
- Maximum speed-matched actor-swap share absolute error: {max_actor_swap_error}
- Maximum activation-share absolute error across profiles: {max_activation_error}
- Three-way speed-matched median high-tail p-value for peak Aggregation: {three_way_peak_p}
- Three-way speed-matched mean peak Aggregation delta versus label null: {three_way_peak_delta}
- n: {n}
- Replicates per condition: {replicate_count}
- Null replicates per observed run: {null_replicates}
- Max activation cap: {max_activations}
""",
        encoding="utf-8",
    )
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": reported_artifact_text_paths,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats if not failures else caveats + failures,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": classification,
        "laySummary": lay_summary,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "n": int(n),
        "replicateCountPerCondition": int(replicate_count),
        "nullReplicatesPerRun": int(null_replicates),
        "maxActivations": int(max_activations),
        "chimericMixtures": CHIMERIC_MIXTURES,
        "speedProfiles": PROFILE_IDS,
        "repoUnitTestCommand": repo_test_payload,
        "validationChecks": checks,
        "validationFailures": failures,
        "validationMatrix": validation,
        "maxSpeedMatchedCellMoveShareAbsError": max_cell_move_error,
        "maxSpeedMatchedActorSwapShareAbsError": max_actor_swap_error,
        "maxActivationShareAbsError": max_activation_error,
        "threeWaySpeedMatchedMedianPHighPeak": three_way_peak_p,
        "threeWaySpeedMatchedMeanPeakDeltaVsNull": three_way_peak_delta,
        "allSpeedMatchedRunsSorted": bool(observed_df[observed_df["speedProfileId"] == PROFILE_SPEED]["completed"].all()),
        "allSpeedMatchedCellMoveSharesWithinTolerance": bool(
            observed_df[observed_df["speedProfileId"] == PROFILE_SPEED]["cellMoveShareWithinTolerance"].all()
        ),
    }
    write_json(status_path, status_payload)
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.e02.s06.artifact_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": success,
            "status": status_payload["status"],
            "generatedAt": utc_now(),
            "git": get_git_metadata(),
            "artifacts": artifact_records,
            "validationResult": validation_result,
            "caveatsOrBlockers": caveats,
        },
    )
    write_json(
        run_manifest_path,
        {
            "schema": "eidosoma.e02.run_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "latestResearchStepId": STEP_ID,
            "generatedAt": utc_now(),
            "startedAt": started_at,
            "statusPath": str(status_path),
            "git": get_git_metadata(),
            "hardware": {
                "platform": platform.platform(),
                "python": sys.version,
                "cpuCount": os.cpu_count(),
                "workerCount": 1,
                "gpuUsed": False,
            },
            "packageVersions": {"numpy": np.__version__, "pandas": pd.__version__},
            "artifacts": artifact_records,
        },
    )
    return summary_path, status_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--e01-artifacts-dir", type=Path, default=DEFAULT_E01_ARTIFACTS)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--replicate-count", type=int, default=3)
    parser.add_argument("--null-replicates", type=int, default=500)
    parser.add_argument("--null-seed-base", type=int, default=606_000)
    parser.add_argument("--max-activations", type=int, default=250_000)
    parser.add_argument("--max-swaps", type=int, default=50_000)
    parser.add_argument("--max-comparisons", type=int, default=500_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    artifacts_dir: Path = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    figure_dir = artifacts_dir / "figures" / "e02"
    step_dir.mkdir(parents=True, exist_ok=True)
    observed_df, diagnostics_df, null_df, timepoint_df, trace_df, validation = run_s06_matrix(
        e01_artifacts=args.e01_artifacts_dir,
        n=int(args.n),
        replicate_count=int(args.replicate_count),
        null_replicates=int(args.null_replicates),
        null_seed_base=int(args.null_seed_base),
        max_activations=int(args.max_activations),
        max_swaps=int(args.max_swaps),
        max_comparisons=int(args.max_comparisons),
    )
    table_paths = write_tables(observed_df, diagnostics_df, null_df, timepoint_df, trace_df, artifacts_dir)
    summary_df = pd.read_parquet(table_paths["summary_parquet"])
    figure_png, figure_pdf = plot_speed_matched_summary(summary_df, figure_dir)
    validation_success, checks, failures = validate_outputs(
        observed_df,
        diagnostics_df,
        null_df,
        timepoint_df,
        trace_df,
        validation,
    )
    repo_test_payload = run_repo_tests(step_dir)
    validation_report = write_validation_report(step_dir, checks, failures)
    code_paths = copy_code_artifacts(step_dir)
    artifact_paths = [
        *table_paths.values(),
        figure_png,
        figure_pdf,
        validation_report,
        step_dir / "repo_unit_test_log.txt",
        *code_paths,
    ]
    summary_path, status_path, manifest_path = write_reports_and_manifests(
        artifacts_dir=artifacts_dir,
        step_dir=step_dir,
        observed_df=observed_df,
        summary_df=summary_df,
        diagnostics_df=diagnostics_df,
        null_df=null_df,
        checks=checks,
        failures=failures,
        validation=validation,
        repo_test_payload=repo_test_payload,
        artifact_paths=artifact_paths,
        started_at=started_at,
        n=int(args.n),
        replicate_count=int(args.replicate_count),
        null_replicates=int(args.null_replicates),
        max_activations=int(args.max_activations),
    )
    final_paths = [*artifact_paths, summary_path, status_path, manifest_path, artifacts_dir / "provenance" / "run_manifest.json"]
    manifest_payload = read_json(manifest_path)
    manifest_payload["artifacts"] = collect_artifacts(final_paths)
    write_json(manifest_path, manifest_payload)
    run_payload = read_json(artifacts_dir / "provenance" / "run_manifest.json")
    run_payload["artifacts"] = collect_artifacts(final_paths)
    write_json(artifacts_dir / "provenance" / "run_manifest.json", run_payload)
    return 0 if validation_success and bool(repo_test_payload["success"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
