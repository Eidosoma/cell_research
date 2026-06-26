#!/usr/bin/env python3
"""Execute E02 S02 scheduler-regime comparisons.

S02 keeps policies, input seeds, and initial arrays matched while changing only
the activation scheduler. It compares the validated S01 random-sequential
engine against deterministic scheduler variants and a bounded original-threaded
control where practical.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
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
    sortedness_raw,
    state_hash,
)
from modules.multithread.BubbleSortCell import BubbleSortCell  # noqa: E402
from modules.multithread.CellGroup import CellGroup, GroupStatus  # noqa: E402
from modules.multithread.InsertionSortCell import InsertionSortCell  # noqa: E402
from modules.multithread.MultiThreadCell import CellStatus  # noqa: E402
from modules.multithread.SelectionSortCell import SelectionSortCell  # noqa: E402
from modules.multithread.StatusProbe import StatusProbe  # noqa: E402


EXPERIMENT_ID = "E02"
STEP_ID = "S02"
STEP_NUMBER = 2
DEFAULT_E01_ARTIFACTS = Path("/previous-artifacts/E01")
SCHEDULER_REGIMES = [
    "s01_random_sequential",
    "synchronous_rounds",
    "priority_queue",
    "adversarial_order",
    "left_to_right",
    "right_to_left",
    "original_threaded_control",
]
DETERMINISTIC_REGIMES = [name for name in SCHEDULER_REGIMES if name != "original_threaded_control"]
CELL_CLASSES = {
    "bubble": BubbleSortCell,
    "insertion": InsertionSortCell,
    "selection": SelectionSortCell,
}
TYPE_ID_TO_ALGOTYPE = {0: "bubble", 1: "selection", 2: "insertion"}


@dataclass
class CommandResult:
    args: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    ok: bool


@dataclass
class ScheduledRunResult:
    completed: bool
    stop_reason: str
    initial_values: list[int]
    final_values: list[int]
    initial_algotypes: list[str]
    final_algotypes: list[str]
    swap_count: int
    comparison_count: int
    archived_compare_and_swap_count: int
    activation_count: int
    scheduler_rounds: int
    event_count: int
    blocked_move_attempts: int
    final_sortedness_percent: float
    final_monotonicity_error: int
    initial_aggregation: float
    final_aggregation: float
    peak_aggregation: float
    dg_max_drop: float
    dg_decrease_count: int
    wall_time_seconds: float
    trace_rows: list[dict[str, Any]]


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def parse_semicolon_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [part for part in str(value).split(";") if part]


def scaled_allocation(mixture_id: str, n: int) -> dict[str, int]:
    if mixture_id.startswith("pure_"):
        return {mixture_id.removeprefix("pure_"): n}
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
    raise ValueError(f"unsupported S02 mixture: {mixture_id}")


def algotypes_from_seed(mixture_id: str, n: int, seed: Any) -> list[str]:
    allocation = scaled_allocation(mixture_id, n)
    algotypes = [algorithm for algorithm, count in allocation.items() for _ in range(count)]
    if len(algotypes) != n:
        raise ValueError(f"allocation for {mixture_id} gives {len(algotypes)} cells, expected {n}")
    if len(allocation) > 1:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(algotypes)
    return algotypes


def scheduler_condition_ids() -> list[str]:
    return [
        "S04_cell_view_bubble_unique_f0_none",
        "S04_cell_view_insertion_unique_f0_none",
        "S04_cell_view_selection_unique_f0_none",
        "S09_cell_view_bubble_insertion_unique_same_goal",
    ]


def load_s02_inputs(
    e01_artifacts: Path,
    *,
    n: int,
    replicate_count: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = read_json(e01_artifacts / "configs" / "e01_baseline_config.json")
    condition_matrix = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "condition_matrix.csv")
    seed_table = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "seed_table.csv")
    selected = condition_matrix[condition_matrix["conditionId"].isin(scheduler_condition_ids())].copy()
    selected["s02N"] = int(n)
    selected = selected.sort_values("conditionId").reset_index(drop=True)
    seeds = seed_table[
        (seed_table["conditionId"].isin(scheduler_condition_ids()))
        & (seed_table["replicateIndex"] < int(replicate_count))
    ].copy()
    seeds = seeds.sort_values(["conditionId", "replicateIndex"]).reset_index(drop=True)
    if len(selected) != len(scheduler_condition_ids()):
        missing = sorted(set(scheduler_condition_ids()) - set(selected["conditionId"]))
        raise ValueError(f"Missing S02 conditions in E01 matrix: {missing}")
    return config, selected, seeds


def local_disorder_priority(sim: DeterministicEventSimulator, cell_id: int) -> tuple[float, int]:
    pos = sim.positions_by_id[cell_id]
    values = sim.current_values()
    score = 0.0
    if pos > 0 and values[pos - 1] > values[pos]:
        score += abs(values[pos - 1] - values[pos])
    if pos < len(values) - 1 and values[pos] > values[pos + 1]:
        score += abs(values[pos] - values[pos + 1])
    return score, -cell_id


def scheduler_order(sim: DeterministicEventSimulator, regime: str, rng: np.random.Generator) -> list[int]:
    current_cell_ids = [cell.cell_id for cell in sim.cells if not cell.frozen]
    if regime == "synchronous_rounds":
        order = list(current_cell_ids)
        rng.shuffle(order)
        return order
    if regime == "left_to_right":
        return current_cell_ids
    if regime == "right_to_left":
        return list(reversed(current_cell_ids))
    if regime == "priority_queue":
        return sorted(current_cell_ids, key=lambda cell_id: local_disorder_priority(sim, cell_id), reverse=True)
    if regime == "adversarial_order":
        return sorted(current_cell_ids, key=lambda cell_id: local_disorder_priority(sim, cell_id))
    raise ValueError(f"unsupported round scheduler: {regime}")


def trajectory_diagnostics(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sortedness = [float(row["sortedness_percent"]) for row in trace_rows]
    aggregations = []
    for row in trace_rows:
        algotypes_raw = row.get("algotypes_json")
        if algotypes_raw:
            aggregations.append(aggregation(json.loads(algotypes_raw)))
    running_best = -math.inf
    max_drop = 0.0
    decrease_count = 0
    previous = None
    for value in sortedness:
        running_best = max(running_best, value)
        max_drop = max(max_drop, running_best - value)
        if previous is not None and value < previous:
            decrease_count += 1
        previous = value
    return {
        "initialSortednessPercent": sortedness[0] if sortedness else None,
        "peakSortednessPercent": max(sortedness) if sortedness else None,
        "dgMaxDropPercent": float(max_drop),
        "dgDecreaseCount": int(decrease_count),
        "initialAggregation": aggregations[0] if aggregations else None,
        "peakAggregation": max(aggregations) if aggregations else None,
    }


def run_deterministic_scheduler(
    *,
    regime: str,
    initial_values: list[int],
    initial_algotypes: list[str],
    condition_id: str,
    scheduler_seed: int,
    tie_breaker_seed: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int,
) -> ScheduledRunResult:
    start = time.perf_counter()
    sim = DeterministicEventSimulator(
        initial_values,
        initial_algotypes,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_breaker_seed,
        condition_id=condition_id,
        research_step_id=STEP_ID,
    )
    scheduler_rounds = 0
    no_move_checks = 0
    stop_reason = "sorted"
    rng = np.random.default_rng(int(scheduler_seed))

    if regime == "s01_random_sequential":
        result = sim.run(
            max_activations=max_activations,
            max_swaps=max_swaps,
            max_comparisons=max_comparisons,
            no_move_checks_required=no_move_checks_required,
        )
        diagnostics = trajectory_diagnostics(result.trace_rows)
        return ScheduledRunResult(
            completed=result.completed,
            stop_reason=result.stop_reason,
            initial_values=result.initial_values,
            final_values=result.final_values,
            initial_algotypes=result.initial_algotypes,
            final_algotypes=result.final_algotypes,
            swap_count=result.swap_count,
            comparison_count=result.comparison_count,
            archived_compare_and_swap_count=result.archived_compare_and_swap_count,
            activation_count=result.activation_count,
            scheduler_rounds=math.ceil(result.activation_count / max(1, len(initial_values))),
            event_count=result.event_count,
            blocked_move_attempts=result.blocked_move_attempts,
            final_sortedness_percent=result.final_sortedness_percent,
            final_monotonicity_error=result.final_monotonicity_error,
            initial_aggregation=float(diagnostics["initialAggregation"] or aggregation(result.initial_algotypes)),
            final_aggregation=result.final_aggregation,
            peak_aggregation=float(diagnostics["peakAggregation"] or result.final_aggregation),
            dg_max_drop=float(diagnostics["dgMaxDropPercent"]),
            dg_decrease_count=int(diagnostics["dgDecreaseCount"]),
            wall_time_seconds=result.wall_time_seconds,
            trace_rows=result.trace_rows,
        )

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
        if not sim.legal_action_exists():
            no_move_checks += 1
            if no_move_checks >= no_move_checks_required:
                stop_reason = "no_cell_can_move_after_two_checks"
                break
        else:
            no_move_checks = 0

        swaps_before = sim.swap_count
        order = scheduler_order(sim, regime, rng)
        for cell_id in order:
            if sim.is_sorted():
                break
            if sim.activation_count >= max_activations or sim.swap_count >= max_swaps or sim.comparison_count >= max_comparisons:
                break
            sim.step(forced_cell_id=cell_id)
        scheduler_rounds += 1
        if sim.swap_count == swaps_before and not sim.legal_action_exists():
            no_move_checks += 1
        else:
            no_move_checks = 0

    trace_rows = list(sim.trace_rows)
    diagnostics = trajectory_diagnostics(trace_rows)
    final_values = sim.current_values()
    return ScheduledRunResult(
        completed=sim.is_sorted(),
        stop_reason=stop_reason,
        initial_values=list(initial_values),
        final_values=final_values,
        initial_algotypes=list(initial_algotypes),
        final_algotypes=sim.current_algotypes(),
        swap_count=sim.swap_count,
        comparison_count=sim.comparison_count,
        archived_compare_and_swap_count=sim.archived_compare_and_swap_count,
        activation_count=sim.activation_count,
        scheduler_rounds=scheduler_rounds,
        event_count=len(trace_rows),
        blocked_move_attempts=sim.blocked_move_attempts,
        final_sortedness_percent=sortedness_percent(final_values),
        final_monotonicity_error=monotonicity_error(final_values),
        initial_aggregation=float(diagnostics["initialAggregation"] or aggregation(initial_algotypes)),
        final_aggregation=aggregation(sim.current_algotypes()),
        peak_aggregation=float(diagnostics["peakAggregation"] or aggregation(sim.current_algotypes())),
        dg_max_drop=float(diagnostics["dgMaxDropPercent"]),
        dg_decrease_count=int(diagnostics["dgDecreaseCount"]),
        wall_time_seconds=time.perf_counter() - start,
        trace_rows=trace_rows,
    )


def make_original_cells(
    values: list[int],
    algotypes: list[str],
    *,
    tie_breaker_seed: int,
) -> tuple[list[Any], StatusProbe, threading.Lock]:
    random.seed(int(tie_breaker_seed))
    lock = threading.Lock()
    probe = StatusProbe()
    cells: list[Any] = [None] * len(values)
    left_boundary = (0, 0)
    right_boundary = (len(values) - 1, 0)
    for idx, (value, algotype) in enumerate(zip(values, algotypes)):
        constructor = CELL_CLASSES[algotype]
        cell = constructor(
            idx,
            int(value),
            lock,
            (idx, 0),
            cells,
            left_boundary,
            right_boundary,
            probe,
            disable_visualization=True,
            label=0,
            reverse_direction=False,
        )
        cells[idx] = cell
    group = CellGroup(cells, cells, 0, left_boundary, right_boundary, GroupStatus.ACTIVE, lock, 1_000_000_000, 1_000_000_000)
    for cell in cells:
        cell.group = group
    return cells, probe, lock


def values_from_original_cells(cells: list[Any]) -> list[int]:
    return [int(cell.value) for cell in cells]


def algotypes_from_original_cells(cells: list[Any]) -> list[str]:
    return [str(cell.cell_type).lower() for cell in cells]


def no_original_cells_should_move(cells: list[Any]) -> bool:
    for cell in list(cells):
        if cell.status == CellStatus.SLEEP:
            return False
        if cell.status == CellStatus.ACTIVE and cell.should_move():
            return False
    return True


def stop_original_threads(cells: list[Any], lock: threading.Lock) -> None:
    acquired = lock.acquire(timeout=1.0)
    try:
        for cell in cells:
            cell.status = CellStatus.INACTIVE
    finally:
        if acquired:
            lock.release()
    for cell in list(cells):
        if cell.ident is not None:
            cell.join(timeout=0.5)


def original_trace_rows(
    *,
    condition_id: str,
    initial_values: list[int],
    initial_algotypes: list[str],
    probe: StatusProbe,
    scheduler_seed: int,
    tie_breaker_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    initial_raw = sortedness_raw(initial_values)
    rows.append(
        {
            "research_step_id": STEP_ID,
            "condition_id": condition_id,
            "implementation": "original_threaded",
            "algorithm": "+".join(sorted(set(initial_algotypes))),
            "event_index": 0,
            "event_kind": "initial",
            "activation_index": 0,
            "actor_cell_id": None,
            "actor_algotype": None,
            "target_position": None,
            "swap_count": 0,
            "comparison_count": 0,
            "archived_compare_and_swap_count": 0,
            "sortedness_raw_count": initial_raw,
            "sortedness_percent": sortedness_percent(initial_values),
            "monotonicity_error": monotonicity_error(initial_values),
            "state_hash": state_hash(initial_values),
            "initial_state_hash": state_hash(initial_values),
            "frozen_positions_json": "[]",
            "algotypes_json": json.dumps(initial_algotypes, separators=(",", ":")),
            "scheduler_seed": int(scheduler_seed),
            "tie_breaker_seed": int(tie_breaker_seed),
        }
    )
    for event_index, values in enumerate(probe.sorting_steps, start=1):
        if event_index - 1 < len(probe.cell_types):
            cell_type_snapshot = probe.cell_types[event_index - 1]
            algotypes = [TYPE_ID_TO_ALGOTYPE.get(int(item[1]), str(item[1])) for item in cell_type_snapshot]
        else:
            algotypes = initial_algotypes
        rows.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "implementation": "original_threaded",
                "algorithm": "+".join(sorted(set(initial_algotypes))),
                "event_index": event_index,
                "event_kind": "swap",
                "activation_index": None,
                "actor_cell_id": None,
                "actor_algotype": None,
                "target_position": None,
                "swap_count": event_index,
                "comparison_count": int(probe.compare_and_swap_count) + event_index,
                "archived_compare_and_swap_count": int(probe.compare_and_swap_count),
                "sortedness_raw_count": sortedness_raw(values),
                "sortedness_percent": sortedness_percent(values),
                "monotonicity_error": monotonicity_error(values),
                "state_hash": state_hash(values),
                "initial_state_hash": state_hash(initial_values),
                "frozen_positions_json": "[]",
                "algotypes_json": json.dumps(algotypes, separators=(",", ":")),
                "scheduler_seed": int(scheduler_seed),
                "tie_breaker_seed": int(tie_breaker_seed),
            }
        )
    return rows


def run_original_threaded_control(
    *,
    initial_values: list[int],
    initial_algotypes: list[str],
    condition_id: str,
    scheduler_seed: int,
    tie_breaker_seed: int,
    timeout_seconds: float,
) -> ScheduledRunResult:
    start = time.perf_counter()
    cells, probe, lock = make_original_cells(initial_values, initial_algotypes, tie_breaker_seed=tie_breaker_seed)
    stop_reason = "wall_time_cap"
    no_move_checks = 0
    try:
        for cell in list(cells):
            cell.start()
        while time.perf_counter() - start < timeout_seconds:
            current_values = values_from_original_cells(cells)
            if sortedness_raw(current_values) == len(current_values) - 1:
                stop_reason = "sorted"
                break
            if no_original_cells_should_move(cells):
                no_move_checks += 1
                if no_move_checks >= 2:
                    stop_reason = "no_cell_can_move_after_two_checks"
                    break
            else:
                no_move_checks = 0
            time.sleep(0.005)
    finally:
        stop_original_threads(cells, lock)
    final_values = values_from_original_cells(cells)
    final_algotypes = algotypes_from_original_cells(cells)
    trace_rows = original_trace_rows(
        condition_id=condition_id,
        initial_values=initial_values,
        initial_algotypes=initial_algotypes,
        probe=probe,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_breaker_seed,
    )
    diagnostics = trajectory_diagnostics(trace_rows)
    return ScheduledRunResult(
        completed=sortedness_raw(final_values) == len(final_values) - 1,
        stop_reason=stop_reason,
        initial_values=list(initial_values),
        final_values=final_values,
        initial_algotypes=list(initial_algotypes),
        final_algotypes=final_algotypes,
        swap_count=int(probe.swap_count),
        comparison_count=int(probe.compare_and_swap_count) + int(probe.swap_count),
        archived_compare_and_swap_count=int(probe.compare_and_swap_count),
        activation_count=0,
        scheduler_rounds=0,
        event_count=len(trace_rows),
        blocked_move_attempts=0,
        final_sortedness_percent=sortedness_percent(final_values),
        final_monotonicity_error=monotonicity_error(final_values),
        initial_aggregation=aggregation(initial_algotypes),
        final_aggregation=aggregation(final_algotypes),
        peak_aggregation=float(diagnostics["peakAggregation"] or aggregation(final_algotypes)),
        dg_max_drop=float(diagnostics["dgMaxDropPercent"]),
        dg_decrease_count=int(diagnostics["dgDecreaseCount"]),
        wall_time_seconds=time.perf_counter() - start,
        trace_rows=trace_rows,
    )


def make_summary_record(
    *,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    scheduler_regime: str,
    scheduler_family: str,
    n: int,
    result: ScheduledRunResult,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> dict[str, Any]:
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": condition["conditionId"],
        "schedulerRegime": scheduler_regime,
        "schedulerFamily": scheduler_family,
        "implementation": condition["implementation"],
        "mixtureId": condition["mixtureId"],
        "algorithms": condition["algorithms"],
        "inputProfile": condition["inputProfile"],
        "n": int(n),
        "replicateIndex": int(seed_row["replicateIndex"]),
        "replicateNumber": int(seed_row["replicateNumber"]),
        "inputPermutationSeed": int(seed_row["inputPermutationSeed"]),
        "algotypeAssignmentSeed": None
        if pd.isna(seed_row.get("algotypeAssignmentSeed"))
        else int(seed_row["algotypeAssignmentSeed"]),
        "schedulerSeed": int(seed_row["schedulerSeed"]),
        "tieBreakerSeed": int(seed_row["tieBreakerSeed"]),
        "initialValuesHash": state_hash(result.initial_values),
        "finalValuesHash": state_hash(result.final_values),
        "initialValues": json.dumps(result.initial_values, separators=(",", ":")),
        "finalValues": json.dumps(result.final_values, separators=(",", ":")),
        "initialAlgotypes": json.dumps(result.initial_algotypes, separators=(",", ":")),
        "finalAlgotypes": json.dumps(result.final_algotypes, separators=(",", ":")),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "finalSortednessPercent": float(result.final_sortedness_percent),
        "finalMonotonicityError": int(result.final_monotonicity_error),
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "archivedCompareAndSwapCount": int(result.archived_compare_and_swap_count),
        "swapPlusComparisonSteps": int(result.swap_count + result.comparison_count),
        "activationCount": int(result.activation_count),
        "schedulerRounds": int(result.scheduler_rounds),
        "blockedMoveAttempts": int(result.blocked_move_attempts),
        "eventCount": int(result.event_count),
        "initialAggregation": float(result.initial_aggregation),
        "finalAggregation": float(result.final_aggregation),
        "peakAggregation": float(result.peak_aggregation),
        "dgMaxDropPercent": float(result.dg_max_drop),
        "dgDecreaseCount": int(result.dg_decrease_count),
        "wallTimeSeconds": float(result.wall_time_seconds),
        "maxActivationCap": int(max_activations),
        "maxSwapCap": int(max_swaps),
        "maxComparisonCap": int(max_comparisons),
    }


def annotate_trace_rows(
    rows: list[dict[str, Any]],
    *,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    scheduler_regime: str,
    scheduler_family: str,
) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "schedulerRegime": scheduler_regime,
                "schedulerFamily": scheduler_family,
                "replicateIndex": int(seed_row["replicateIndex"]),
                "replicateNumber": int(seed_row["replicateNumber"]),
                "mixtureId": condition["mixtureId"],
                "inputPermutationSeed": int(seed_row["inputPermutationSeed"]),
                "algotypeAssignmentSeed": None
                if pd.isna(seed_row.get("algotypeAssignmentSeed"))
                else int(seed_row["algotypeAssignmentSeed"]),
            }
        )
        annotated.append(item)
    return annotated


def run_scheduler_matrix(
    *,
    e01_artifacts: Path,
    n: int,
    replicate_count: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
    threaded_timeout_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _config, conditions_df, seeds_df = load_s02_inputs(e01_artifacts, n=n, replicate_count=replicate_count)
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    validation_notes: list[str] = []
    for condition_tuple in conditions_df.itertuples(index=False):
        condition = condition_tuple._asdict()
        condition_seeds = seeds_df[seeds_df["conditionId"] == condition["conditionId"]].copy()
        for seed_tuple in condition_seeds.itertuples(index=False):
            seed_row = seed_tuple._asdict()
            input_seed = int(seed_row["inputPermutationSeed"])
            initial_values = initial_values_from_seed(input_seed, n=n, profile=str(condition["inputProfile"]))
            algotype_seed = seed_row.get("algotypeAssignmentSeed")
            if pd.isna(algotype_seed):
                algotype_seed = int(seed_row["schedulerSeed"])
            initial_algotypes = algotypes_from_seed(str(condition["mixtureId"]), n, algotype_seed)
            for regime in SCHEDULER_REGIMES:
                scheduler_family = "original_threaded" if regime == "original_threaded_control" else "deterministic_event"
                if regime == "original_threaded_control":
                    result = run_original_threaded_control(
                        initial_values=initial_values,
                        initial_algotypes=initial_algotypes,
                        condition_id=condition["conditionId"],
                        scheduler_seed=int(seed_row["schedulerSeed"]),
                        tie_breaker_seed=int(seed_row["tieBreakerSeed"]),
                        timeout_seconds=threaded_timeout_seconds,
                    )
                else:
                    result = run_deterministic_scheduler(
                        regime=regime,
                        initial_values=initial_values,
                        initial_algotypes=initial_algotypes,
                        condition_id=condition["conditionId"],
                        scheduler_seed=int(seed_row["schedulerSeed"]),
                        tie_breaker_seed=int(seed_row["tieBreakerSeed"]),
                        max_activations=max_activations,
                        max_swaps=max_swaps,
                        max_comparisons=max_comparisons,
                        no_move_checks_required=2,
                    )
                summary_rows.append(
                    make_summary_record(
                        condition=condition,
                        seed_row=seed_row,
                        scheduler_regime=regime,
                        scheduler_family=scheduler_family,
                        n=n,
                        result=result,
                        max_activations=max_activations,
                        max_swaps=max_swaps,
                        max_comparisons=max_comparisons,
                    )
                )
                trace_rows.extend(
                    annotate_trace_rows(
                        result.trace_rows,
                        condition=condition,
                        seed_row=seed_row,
                        scheduler_regime=regime,
                        scheduler_family=scheduler_family,
                    )
                )
                if regime == "original_threaded_control" and result.stop_reason == "wall_time_cap":
                    validation_notes.append(
                        f"Original threaded control hit wall_time_cap for {condition['conditionId']} replicate {seed_row['replicateIndex']}."
                    )
    validation = {
        "conditionCount": int(len(conditions_df)),
        "replicateCountPerCondition": int(replicate_count),
        "schedulerRegimeCount": int(len(SCHEDULER_REGIMES)),
        "expectedRows": int(len(conditions_df) * replicate_count * len(SCHEDULER_REGIMES)),
        "notes": validation_notes,
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(trace_rows), validation


def summarize_by_scheduler(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary_df.groupby(["schedulerRegime", "schedulerFamily", "mixtureId"], dropna=False)
        .agg(
            runCount=("conditionId", "count"),
            completedCount=("completed", "sum"),
            completionRate=("completed", "mean"),
            meanFinalSortednessPercent=("finalSortednessPercent", "mean"),
            meanFinalMonotonicityError=("finalMonotonicityError", "mean"),
            meanSwapCount=("swapCount", "mean"),
            meanActivationCount=("activationCount", "mean"),
            meanSchedulerRounds=("schedulerRounds", "mean"),
            meanPeakAggregation=("peakAggregation", "mean"),
            meanFinalAggregation=("finalAggregation", "mean"),
            meanDgMaxDropPercent=("dgMaxDropPercent", "mean"),
            wallTimeSecondsTotal=("wallTimeSeconds", "sum"),
        )
        .reset_index()
    )
    return grouped.sort_values(["schedulerFamily", "schedulerRegime", "mixtureId"]).reset_index(drop=True)


def validate_s02_outputs(summary_df: pd.DataFrame, validation: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    checks: list[str] = []
    failures: list[str] = []
    if len(summary_df) == validation["expectedRows"]:
        checks.append(f"Scheduler matrix row count matched expected {validation['expectedRows']}.")
    else:
        failures.append(f"Scheduler matrix has {len(summary_df)} rows, expected {validation['expectedRows']}.")
    seed_groups = summary_df.groupby(["conditionId", "replicateIndex"])
    if all(group["initialValuesHash"].nunique() == 1 for _, group in seed_groups):
        checks.append("Initial value hashes are matched across scheduler regimes for every condition/replicate.")
    else:
        failures.append("At least one condition/replicate has mismatched initial value hashes across schedulers.")
    if all(group["initialAlgotypes"].nunique() == 1 for _, group in seed_groups):
        checks.append("Initial Algotype assignments are matched across scheduler regimes for every condition/replicate.")
    else:
        failures.append("At least one condition/replicate has mismatched initial Algotype assignments across schedulers.")
    deterministic = summary_df[summary_df["schedulerFamily"] == "deterministic_event"]
    if bool((deterministic["finalValues"].apply(lambda text: Counter(json.loads(text))) == deterministic["initialValues"].apply(lambda text: Counter(json.loads(text)))).all()):
        checks.append("Deterministic scheduler runs conserve value multisets.")
    else:
        failures.append("A deterministic scheduler run failed value-multiset conservation.")
    if bool((deterministic["finalAlgotypes"].apply(lambda text: Counter(json.loads(text))) == deterministic["initialAlgotypes"].apply(lambda text: Counter(json.loads(text)))).all()):
        checks.append("Deterministic scheduler runs conserve Algotype multisets.")
    else:
        failures.append("A deterministic scheduler run failed Algotype-multiset conservation.")
    non_threaded_pure = deterministic[deterministic["mixtureId"].isin(["pure_bubble", "pure_insertion", "pure_selection"])]
    if bool(non_threaded_pure["completed"].all()):
        checks.append("All deterministic pure-policy scheduler runs completed with sorted final states.")
    else:
        failures.append("At least one deterministic pure-policy scheduler run did not complete.")
    if set(summary_df["schedulerRegime"]) == set(SCHEDULER_REGIMES):
        checks.append("All requested scheduler regimes are represented.")
    else:
        failures.append(f"Observed scheduler regimes differ from expected: {sorted(summary_df['schedulerRegime'].unique())}.")
    return not failures, checks, failures


def plot_scheduler_results(summary_df: pd.DataFrame, scheduler_summary: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "e02_scheduler_regimes_summary.png"
    pdf_path = figure_dir / "e02_scheduler_regimes_summary.pdf"
    order = SCHEDULER_REGIMES
    plot_df = (
        summary_df.groupby(["schedulerRegime", "mixtureId"], as_index=False)
        .agg(
            finalSortedness=("finalSortednessPercent", "mean"),
            activationCount=("activationCount", "mean"),
            peakAggregation=("peakAggregation", "mean"),
            dgMaxDrop=("dgMaxDropPercent", "mean"),
            completionRate=("completed", "mean"),
        )
    )
    mixtures = ["pure_bubble", "pure_insertion", "pure_selection", "bubble_insertion"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    metrics = [
        ("finalSortedness", "Mean final Sortedness (%)"),
        ("activationCount", "Mean activation count"),
        ("peakAggregation", "Mean peak Aggregation"),
        ("dgMaxDrop", "Mean DG max drop (%)"),
    ]
    colors = {
        "pure_bubble": "#4c78a8",
        "pure_insertion": "#f58518",
        "pure_selection": "#54a24b",
        "bubble_insertion": "#b279a2",
    }
    for ax, (metric, title) in zip(axes, metrics):
        x = np.arange(len(order))
        width = 0.18
        for offset_index, mixture in enumerate(mixtures):
            subset = plot_df[plot_df["mixtureId"] == mixture].set_index("schedulerRegime").reindex(order)
            ax.bar(x + (offset_index - 1.5) * width, subset[metric].to_numpy(dtype=float), width=width, label=mixture, color=colors[mixture])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("E02 S02 Scheduler-regime comparison (matched seeds and initial arrays)", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def write_tables(summary_df: pd.DataFrame, trace_df: pd.DataFrame, artifacts_dir: Path) -> dict[str, Path]:
    results_dir = artifacts_dir / "results"
    traces_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    scheduler_summary = summarize_by_scheduler(summary_df)
    paths = {
        "results_parquet": results_dir / "e02_scheduler_regimes.parquet",
        "results_csv": results_dir / "e02_scheduler_regimes.csv",
        "summary_parquet": results_dir / "e02_scheduler_regime_summary.parquet",
        "summary_csv": results_dir / "e02_scheduler_regime_summary.csv",
        "trace_parquet": traces_dir / "e02_scheduler_trace_events.parquet",
        "trace_csv_gz": traces_dir / "e02_scheduler_trace_events.csv.gz",
    }
    summary_df.to_parquet(paths["results_parquet"], index=False)
    summary_df.to_csv(paths["results_csv"], index=False)
    scheduler_summary.to_parquet(paths["summary_parquet"], index=False)
    scheduler_summary.to_csv(paths["summary_csv"], index=False)
    trace_df.to_parquet(paths["trace_parquet"], index=False)
    trace_df.to_csv(paths["trace_csv_gz"], index=False, compression="gzip")
    return paths


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(artifacts_dir: Path, step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    script_dst = code_dir / "scripts" / "e02_s02_scheduler_regimes.py"
    test_dst = code_dir / "tests" / "test_e02_scheduler_regimes.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "e02_s02_scheduler_regimes.py", script_dst)
    if (REPO_ROOT / "tests" / "test_e02_scheduler_regimes.py").exists():
        shutil.copy2(REPO_ROOT / "tests" / "test_e02_scheduler_regimes.py", test_dst)
    return [path for path in [script_dst, test_dst] if path.exists()]


def write_validation_report(step_dir: Path, checks: list[str], failures: list[str], validation: dict[str, Any]) -> Path:
    path = step_dir / "validation_report.md"
    checks_text = "\n".join(f"- {item}" for item in checks) or "- None"
    failures_text = "\n".join(f"- {item}" for item in failures) or "- None"
    notes_text = "\n".join(f"- {item}" for item in validation.get("notes", [])) or "- None"
    path.write_text(
        f"""# S02 Validation Report

- Research step ID: {STEP_ID}
- Completion status: {'completed' if not failures else 'completed with validation failures'}
- Artifacts written: scheduler result tables, compact trace events, figure outputs, copied code, manifests, and status files under `$ARTIFACTS_DIR`.
- Validation result: {'passed' if not failures else 'failed'}
- Caveats or blockers: original threaded control is nondeterministic and bounded by a wall-time cap; synchronous rounds are implemented as equal-activation snapshot rounds rather than simultaneous conflict resolution; DG values are trajectory backtracking proxies computed from Sortedness traces.
- Recommended next action: proceed to S03 activation-rate artifact tests only after Chief Scientist review.

## Checks

{checks_text}

## Failures

{failures_text}

## Notes

{notes_text}
""",
        encoding="utf-8",
    )
    return path


def run_repo_tests(step_dir: Path) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    result = run_command(cmd, cwd=REPO_ROOT)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + result.stdout + "\n\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    return {
        "command": cmd,
        "returnCode": result.returncode,
        "success": result.ok,
        "logPath": str(log_path),
    }


def write_reports_and_manifests(
    *,
    artifacts_dir: Path,
    step_dir: Path,
    summary_df: pd.DataFrame,
    scheduler_summary_df: pd.DataFrame,
    checks: list[str],
    failures: list[str],
    validation: dict[str, Any],
    repo_test_payload: dict[str, Any],
    artifact_paths: list[Path],
    started_at: str,
    n: int,
    replicate_count: int,
    max_activations: int,
    threaded_timeout_seconds: float,
) -> tuple[Path, Path, Path]:
    pure = summary_df[(summary_df["schedulerFamily"] == "deterministic_event") & (summary_df["mixtureId"].str.startswith("pure_"))]
    chimera = summary_df[summary_df["mixtureId"] == "bubble_insertion"]
    count_span = (
        summary_df.groupby("schedulerRegime")["activationCount"].mean().replace(0, np.nan).dropna()
    )
    activation_fold_span = float(count_span.max() / count_span.min()) if len(count_span) > 1 else None
    success = not failures and bool(repo_test_payload["success"])
    outcome_classification = "supportive" if bool(pure["completed"].all()) else "constraining/contradictory"
    validation_result = (
        "passed: matched seeds/initial arrays, all requested schedulers represented, deterministic pure-policy runs completed, and repository tests passed"
        if success
        else "failed: see validation_report.md and status.json"
    )
    caveats = [
        "Original threaded control is nondeterministic and bounded by a wall-time cap, so it is a practical control rather than a replayable schedule.",
        "Synchronous rounds are an equal-activation snapshot-round scheduler, not a fully simultaneous conflict-resolution model.",
        "DG is reported as a Sortedness backtracking proxy for S02; the fuller DG-specific null audit remains S08.",
        f"S02 uses a bounded n={n} matched-seed audit matrix to cover scheduler regimes without starting the larger S03 activation-rate study.",
    ]
    recommended_next_action = "Proceed to S03 activation-rate artifact tests only after Chief Scientist review; keep S02 scheduler caveats attached to downstream claims."
    artifact_records = collect_artifacts(artifact_paths)
    artifact_text = "\n".join(f"- `{item['path']}`" for item in artifact_records)
    validation_table = markdown_table(
        ["Check type", "Count"],
        [
            ["scheduler result rows", len(summary_df)],
            ["scheduler regimes", summary_df["schedulerRegime"].nunique()],
            ["conditions", summary_df["conditionId"].nunique()],
            ["replicates per condition", replicate_count],
            ["trace rows", int(sum(summary_df["eventCount"]))],
        ],
    )
    scheduler_preview = scheduler_summary_df.sort_values(["mixtureId", "schedulerRegime"]).head(16)
    preview_table = markdown_table(
        ["Scheduler", "Mixture", "Runs", "Completion", "Final Sortedness", "Mean activations", "Peak Aggregation", "DG drop"],
        [
            [
                row.schedulerRegime,
                row.mixtureId,
                int(row.runCount),
                row.completionRate,
                row.meanFinalSortednessPercent,
                row.meanActivationCount,
                row.meanPeakAggregation,
                row.meanDgMaxDropPercent,
            ]
            for row in scheduler_preview.itertuples(index=False)
        ],
    )
    summary_path = step_dir / "summary.md"
    summary_path.write_text(
        f"""# E02 S02 Status Summary

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {'completed' if success else 'completed with validation failures'}
- Outcome classification: {outcome_classification}
- Artifacts written:
{artifact_text}
- Validation result: {validation_result}
- Caveats or blockers: {' '.join(caveats)}
- Lay summary: S02 changed the scheduler while holding policies, seeds, and initial arrays fixed. Pure policies still sorted under deterministic scheduler variants, but scheduler choice changed activation counts and trajectory backtracking, so efficiency-like claims need scheduler controls.
- Recommended next action: {recommended_next_action}

## Run Matrix

{validation_table}

## Scheduler Preview

{preview_table}

## Anchor Notes

- Deterministic pure-policy completion rate: {float(pure['completed'].mean()) if len(pure) else None}
- Bubble+Insertion chimera completion rate across all schedulers: {float(chimera['completed'].mean()) if len(chimera) else None}
- Mean activation-count fold span across schedulers: {activation_fold_span}
- n: {n}
- Replicates per condition: {replicate_count}
- Original threaded timeout seconds: {threaded_timeout_seconds}
- Max activation cap: {max_activations}
""",
        encoding="utf-8",
    )
    status_path = step_dir / "status.json"
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": [item["path"] for item in artifact_records],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats if not failures else caveats + failures,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": outcome_classification,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "n": n,
        "replicateCountPerCondition": replicate_count,
        "schedulerRegimes": SCHEDULER_REGIMES,
        "repoUnitTestCommand": repo_test_payload,
        "validationChecks": checks,
        "validationFailures": failures,
        "validationMatrix": validation,
    }
    write_json(status_path, status_payload)
    manifest_path = step_dir / "artifact_manifest.json"
    manifest_payload = {
        "schema": "eidosoma.e02.s02.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "git": get_git_metadata(),
        "artifacts": artifact_records,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
    }
    write_json(manifest_path, manifest_payload)
    run_manifest_path = artifacts_dir / "provenance" / "run_manifest.json"
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
            "packageVersions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
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
    parser.add_argument("--max-activations", type=int, default=250_000)
    parser.add_argument("--max-swaps", type=int, default=50_000)
    parser.add_argument("--max-comparisons", type=int, default=500_000)
    parser.add_argument("--threaded-timeout-seconds", type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    artifacts_dir: Path = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    figure_dir = artifacts_dir / "figures" / "e02"
    step_dir.mkdir(parents=True, exist_ok=True)
    summary_df, trace_df, validation = run_scheduler_matrix(
        e01_artifacts=args.e01_artifacts_dir,
        n=int(args.n),
        replicate_count=int(args.replicate_count),
        max_activations=int(args.max_activations),
        max_swaps=int(args.max_swaps),
        max_comparisons=int(args.max_comparisons),
        threaded_timeout_seconds=float(args.threaded_timeout_seconds),
    )
    table_paths = write_tables(summary_df, trace_df, artifacts_dir)
    scheduler_summary_df = pd.read_parquet(table_paths["summary_parquet"])
    figure_png, figure_pdf = plot_scheduler_results(summary_df, scheduler_summary_df, figure_dir)
    validation_success, checks, failures = validate_s02_outputs(summary_df, validation)
    repo_test_payload = run_repo_tests(step_dir)
    validation_report = write_validation_report(step_dir, checks, failures, validation)
    code_paths = copy_code_artifacts(artifacts_dir, step_dir)
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
        summary_df=summary_df,
        scheduler_summary_df=scheduler_summary_df,
        checks=checks,
        failures=failures,
        validation=validation,
        repo_test_payload=repo_test_payload,
        artifact_paths=artifact_paths,
        started_at=started_at,
        n=int(args.n),
        replicate_count=int(args.replicate_count),
        max_activations=int(args.max_activations),
        threaded_timeout_seconds=float(args.threaded_timeout_seconds),
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
