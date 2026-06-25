#!/usr/bin/env python3
"""Run E01 S12 opposite-direction mixed Algotype chimeras."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import platform
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import e01_s11_duplicate_value_chimeras as common

from modules.multithread.CellGroup import CellGroup, GroupStatus
from modules.multithread.MultiThreadCell import CellStatus, MultiThreadCell
from modules.multithread.StatusProbe import StatusProbe


EXPERIMENT_ID = "E01"
STEP_ID = "S12"
STEP_NUMBER = 12
STATUS = "completed"
CONFIG_PATH_DEFAULT = Path("/artifacts/configs/e01_baseline_config.json")
CONDITION_MATRIX_DEFAULT = Path("/artifacts/research_steps/S03/condition_matrix.csv")
SEED_TABLE_DEFAULT = Path("/artifacts/research_steps/S03/seed_table.csv")
S03_STATUS_DEFAULT = Path("/artifacts/research_steps/S03/status.json")
S11_STATUS_DEFAULT = Path("/artifacts/research_steps/S11/status.json")
PROGRESS_BINS = np.linspace(0.0, 1.0, 101)
TRACE_DENSE_EVENT_LIMIT = 5000
TRACE_SAMPLE_INTERVAL = 250
MIXTURE_ORDER = [
    "bubble_down_selection_up",
    "bubble_up_insertion_down",
    "selection_down_insertion_up",
]
INPUT_PROFILE_ORDER = ["unique_1_100", "duplicate_1_10_x10"]
EXPECTED_DOMINANT_ALGOTYPE = {
    "bubble_down_selection_up": "bubble",
    "bubble_up_insertion_down": "bubble",
    "selection_down_insertion_up": "selection",
}
PAPER_UNIQUE_FINAL_SORTEDNESS_RAW = {
    "bubble_down_selection_up": 42.5,
    "bubble_up_insertion_down": 73.73,
    "selection_down_insertion_up": 38.31,
}
DISPLAY_NAMES = {
    "bubble_down_selection_up": "Bubble down + Selection up",
    "bubble_up_insertion_down": "Bubble up + Insertion down",
    "selection_down_insertion_up": "Selection down + Insertion up",
}


def install_compact_archived_swap() -> None:
    """Suppress archived full-array snapshot retention inside each swap."""

    def compact_swap(self: MultiThreadCell, target_position: tuple[int, int], skip_stats: bool = False) -> None:
        current_cell_at_target = self.cells[int(target_position[0])]
        if self.status == CellStatus.FREEZE:
            if not self.tried_to_swap_with_frozen:
                self.status_probe.count_frozen_cell_attempt()
                self.tried_to_swap_with_frozen = True
            return
        self.tried_to_swap_with_frozen = False
        current_cell_at_target.tried_to_swap_with_frozen = False
        self.status = CellStatus.MOVING
        current_cell_at_target.status = CellStatus.MOVING
        self.cells[self.current_position[0]] = current_cell_at_target
        self.cells[target_position[0]] = self
        current_cell_at_target.target_position = self.current_position
        self.target_position = target_position
        if self.visualization_disabled:
            self.current_position = self.target_position
            current_cell_at_target.current_position = current_cell_at_target.target_position
            self.status = self.previous_status
            current_cell_at_target.status = current_cell_at_target.previous_status
        if not skip_stats:
            self.swapping_count[0] = self.swapping_count[0] + 1
            self.status_probe.record_swap()

    MultiThreadCell.swap = compact_swap


install_compact_archived_swap()


@dataclass
class OppositeRunResult:
    completed: bool
    stop_reason: str
    equilibrium_classification: str
    initial_values: list[int]
    final_values: list[int]
    initial_algotypes: list[str]
    final_algotypes: list[str]
    swap_count: int
    comparison_count: int
    archived_compare_and_swap_count: int
    scheduler_rounds: int
    no_progress_rounds: int
    event_count: int
    sampled_trace_row_count: int
    wall_time_seconds: float
    per_algotype_moves: dict[str, int]
    initial_increasing_raw: int
    initial_increasing_percent: float
    initial_decreasing_raw: int
    initial_decreasing_percent: float
    final_increasing_raw: int
    final_increasing_percent: float
    final_decreasing_raw: int
    final_decreasing_percent: float
    final_strict_increasing_pairs: int
    final_strict_decreasing_pairs: int
    final_equal_pairs: int
    initial_aggregation: float
    final_aggregation: float
    peak_aggregation: float
    peak_aggregation_swap_count: int
    peak_aggregation_normalized_progress: float
    random_null_expected: float
    dominant_goal_direction: str
    dominant_algotype: str
    expected_dominant_algotype: str
    dominance_matches_paper: bool
    final_goal_sortedness_json: str
    normalized_increasing_raw: list[float]
    normalized_increasing_percent: list[float]
    normalized_decreasing_raw: list[float]
    normalized_decreasing_percent: list[float]
    normalized_aggregation: list[float]
    normalized_aggregation_minus_null: list[float]


class TrajectoryWriter:
    fieldnames = [
        "conditionId",
        "mixtureId",
        "algorithms",
        "inputProfile",
        "goalDirectionsJson",
        "replicateIndex",
        "replicateNumber",
        "eventIndex",
        "eventKind",
        "swapCount",
        "comparisonCount",
        "archivedCompareAndSwapCount",
        "schedulerRound",
        "increasingSortednessRaw",
        "increasingSortednessPercent",
        "decreasingSortednessRaw",
        "decreasingSortednessPercent",
        "aggregationValue",
        "randomNullExpected",
        "aggregationMinusNull",
        "actorCellId",
        "actorAlgotype",
        "actorGoalDirection",
        "algotypeSequence",
        "stateHash",
    ]

    def __init__(self, path: Path):
        self.path = path
        self.handle: gzip.GzipFile | None = None
        self.writer: csv.DictWriter[str] | None = None
        self.row_count = 0

    def __enter__(self) -> "TrajectoryWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.path, "wt", encoding="utf-8", newline="", compresslevel=1)
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        self.writer.writeheader()
        return self

    def write(self, row: dict[str, Any]) -> None:
        if self.writer is None:
            raise RuntimeError("TrajectoryWriter is not open")
        self.writer.writerow({field: row.get(field, "") for field in self.fieldnames})
        self.row_count += 1

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            self.handle.close()


class BufferedTrajectoryWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.row_count = 0

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self.row_count += 1


def sortedness_raw(values: list[int], direction: str) -> int:
    if direction == "decreasing":
        return sum(1 for i in range(len(values) - 1) if values[i] >= values[i + 1])
    return sum(1 for i in range(len(values) - 1) if values[i] <= values[i + 1])


def sortedness_percent(raw_count: int, n: int) -> float:
    if n < 2:
        return 100.0
    return 100.0 * raw_count / (n - 1)


def interpolate_sampled_points(event_indices: list[int], points: list[float], final_swap_count: int, bins: int = 101) -> list[float]:
    if not points:
        return [0.0] * bins
    if len(points) == 1 or final_swap_count <= 0:
        return [float(points[0])] * bins
    x = np.asarray(event_indices, dtype=float) / float(max(final_swap_count, 1))
    y = np.asarray(points, dtype=float)
    target = np.linspace(0.0, 1.0, num=bins)
    return [float(value) for value in np.interp(target, x, y)]


def strict_pair_counts(values: list[int]) -> tuple[int, int, int]:
    inc = sum(1 for i in range(len(values) - 1) if values[i] < values[i + 1])
    dec = sum(1 for i in range(len(values) - 1) if values[i] > values[i + 1])
    eq = (len(values) - 1) - inc - dec if len(values) > 1 else 0
    return inc, dec, eq


def unique_values_from_seed(seed: int, n: int = 100) -> list[int]:
    rng = np.random.default_rng(int(seed))
    values = np.arange(1, n + 1, dtype=np.int16)
    rng.shuffle(values)
    return [int(value) for value in values]


def initial_values_for_profile(seed: int, input_profile: str) -> list[int]:
    if input_profile == "duplicate_1_10_x10":
        return common.duplicate_values_from_seed(seed)
    if input_profile == "unique_1_100":
        return unique_values_from_seed(seed)
    raise ValueError(f"Unsupported S12 input profile: {input_profile}")


def value_profile_valid(values: list[int], input_profile: str) -> bool:
    if input_profile == "duplicate_1_10_x10":
        return common.values_are_10_copies_each(values)
    if input_profile == "unique_1_100":
        return Counter(values) == Counter(range(1, 101))
    return False


def s12_conditions(condition_matrix_path: Path) -> list[dict[str, Any]]:
    rows = [row for row in common.read_csv_rows(condition_matrix_path) if row["producerStep"] == STEP_ID]
    mixture_order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    profile_order = {profile: index for index, profile in enumerate(INPUT_PROFILE_ORDER)}
    return sorted(rows, key=lambda row: (profile_order[row["inputProfile"]], mixture_order[row["mixtureId"]]))


def s12_seed_rows(seed_table_path: Path, max_replicates_per_condition: int | None = None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in common.read_csv_rows(seed_table_path):
        if row["producerStep"] != STEP_ID:
            continue
        if max_replicates_per_condition is not None and int(row["replicateIndex"]) >= max_replicates_per_condition:
            continue
        grouped[row["conditionId"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["replicateIndex"]))
    return dict(grouped)


def goal_sortedness_by_algotype(values: list[int], algotypes: list[str], goal_directions: dict[str, str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for algotype in sorted(set(algotypes)):
        sub_values = [value for value, label in zip(values, algotypes) if label == algotype]
        direction = goal_directions[algotype]
        raw = sortedness_raw(sub_values, direction)
        result[algotype] = {
            "goalDirection": direction,
            "cellCount": len(sub_values),
            "raw": raw,
            "percent": sortedness_percent(raw, len(sub_values)),
        }
    return result


def classify_dominance(values: list[int], goal_directions: dict[str, str], mixture_id: str) -> tuple[str, str, bool, int, int, int]:
    strict_inc, strict_dec, equal_pairs = strict_pair_counts(values)
    if strict_inc > strict_dec:
        dominant_goal_direction = "increasing"
    elif strict_dec > strict_inc:
        dominant_goal_direction = "decreasing"
    else:
        dominant_goal_direction = "balanced"
    matching = [alg for alg, direction in goal_directions.items() if direction == dominant_goal_direction]
    dominant_algotype = matching[0] if len(matching) == 1 else "balanced"
    expected = EXPECTED_DOMINANT_ALGOTYPE[mixture_id]
    return (
        dominant_goal_direction,
        dominant_algotype,
        dominant_algotype == expected,
        strict_inc,
        strict_dec,
        equal_pairs,
    )


def classify_equilibrium(stop_reason: str, values: list[int], no_progress_rounds: int) -> str:
    if sortedness_raw(values, "increasing") == len(values) - 1:
        return "globally_increasing_sorted"
    if sortedness_raw(values, "decreasing") == len(values) - 1:
        return "globally_decreasing_sorted"
    if stop_reason == "no_cell_can_move_after_two_checks":
        return "static_no_move_equilibrium"
    if stop_reason == "stable_no_progress_round_cap":
        return "stable_no_progress_equilibrium"
    if stop_reason.startswith("max_"):
        return "cap_truncated"
    if no_progress_rounds > 0:
        return "stable_partial_equilibrium"
    return "active_partial_equilibrium"


def write_trajectory_event(
    writer: TrajectoryWriter,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    cells: list[Any],
    algotype_by_cell_id: dict[int, str],
    goal_directions: dict[str, str],
    event_index: int,
    event_kind: str,
    swap_count: int,
    archived_compare_and_swap_count: int,
    scheduler_round: int,
    random_null_expected: float,
    actor_cell_id: int | None = None,
    actor_algotype: str | None = None,
) -> tuple[int, float, int, float, float]:
    values = common.cells_values(cells)
    n = len(values)
    inc_raw = sortedness_raw(values, "increasing")
    dec_raw = sortedness_raw(values, "decreasing")
    inc_pct = sortedness_percent(inc_raw, n)
    dec_pct = sortedness_percent(dec_raw, n)
    position_algotypes = common.cell_algotypes(cells, algotype_by_cell_id)
    aggregation = common.aggregation_value_from_algotypes(position_algotypes)
    writer.write(
        {
            "conditionId": condition["conditionId"],
            "mixtureId": condition["mixtureId"],
            "algorithms": condition["algorithms"],
            "inputProfile": condition["inputProfile"],
            "goalDirectionsJson": json.dumps(goal_directions, sort_keys=True, separators=(",", ":")),
            "replicateIndex": int(seed_row["replicateIndex"]),
            "replicateNumber": int(seed_row["replicateNumber"]),
            "eventIndex": int(event_index),
            "eventKind": event_kind,
            "swapCount": int(swap_count),
            "comparisonCount": int(archived_compare_and_swap_count + swap_count),
            "archivedCompareAndSwapCount": int(archived_compare_and_swap_count),
            "schedulerRound": int(scheduler_round),
            "increasingSortednessRaw": int(inc_raw),
            "increasingSortednessPercent": inc_pct,
            "decreasingSortednessRaw": int(dec_raw),
            "decreasingSortednessPercent": dec_pct,
            "aggregationValue": aggregation,
            "randomNullExpected": random_null_expected,
            "aggregationMinusNull": aggregation - random_null_expected,
            "actorCellId": "" if actor_cell_id is None else int(actor_cell_id),
            "actorAlgotype": "" if actor_algotype is None else actor_algotype,
            "actorGoalDirection": "" if actor_algotype is None else goal_directions[actor_algotype],
            "algotypeSequence": common.algotype_sequence(position_algotypes),
            "stateHash": common.state_hash(values),
        }
    )
    return inc_raw, inc_pct, dec_raw, dec_pct, aggregation


def run_opposite_chimera(
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    writer: TrajectoryWriter,
    max_swaps: int,
    max_comparisons: int,
    max_rounds: int | None,
    no_move_checks_required: int,
    stable_no_progress_round_cap: int,
) -> OppositeRunResult:
    start = time.perf_counter()
    n = common.parse_int(condition["n"])
    allocation = json.loads(condition["algotypeAllocation"])
    goal_directions = json.loads(condition["goalDirections"])
    initial_values = initial_values_for_profile(common.parse_int(seed_row["inputPermutationSeed"]), condition["inputProfile"])
    initial_algotypes = common.algotypes_from_seed(seed_row.get("algotypeAssignmentSeed", ""), allocation)
    if len(initial_values) != n or len(initial_algotypes) != n:
        raise ValueError(f"S12 length mismatch for {condition['conditionId']}")
    if not value_profile_valid(initial_values, condition["inputProfile"]):
        raise ValueError(f"S12 invalid value profile for {condition['conditionId']}")

    lock = threading.Lock()
    status_probe = StatusProbe()
    swapping_count = [0]
    export_steps: list[list[int]] = []
    cells: list[Any] = [None] * n
    random.seed(common.parse_int(seed_row["tieBreakerSeed"]))
    for index, (value, algotype) in enumerate(zip(initial_values, initial_algotypes)):
        cell_class = common.CELL_CLASSES[algotype]
        cells[index] = cell_class(
            index,
            int(value),
            lock,
            (index, 0),
            cells,
            (0, 0),
            (n - 1, 0),
            status_probe,
            disable_visualization=True,
            swapping_count=swapping_count,
            export_steps=export_steps,
            label=0,
            reverse_direction=(goal_directions[algotype] == "decreasing"),
        )

    group = CellGroup(
        cells,
        cells,
        0,
        (0, 0),
        (n - 1, 0),
        GroupStatus.ACTIVE,
        lock,
        count_down=1,
        phase_period=1,
    )
    cell_objects = list(cells)
    for cell in cell_objects:
        cell.group = group

    algotype_by_cell_id = {int(cell.threadID): initial_algotypes[int(cell.threadID)] for cell in cell_objects}
    random_null_expected = common.random_adjacency_expected_from_counts(dict(Counter(initial_algotypes)))
    scheduler_rng = np.random.default_rng(common.parse_int(seed_row["schedulerSeed"]))
    per_algotype_moves = {algorithm: 0 for algorithm in common.ALGORITHMS if int(allocation.get(algorithm, 0)) > 0}
    scheduler_rounds = 0
    no_move_checks = 0
    no_progress_rounds = 0
    stop_reason = "no_cell_can_move_after_two_checks"
    inc_raw_points: list[float] = []
    inc_pct_points: list[float] = []
    dec_raw_points: list[float] = []
    dec_pct_points: list[float] = []
    aggregation_points: list[float] = []
    aggregation_minus_null_points: list[float] = []
    sample_event_indices: list[int] = []

    initial_inc_raw, initial_inc_pct, initial_dec_raw, initial_dec_pct, initial_aggregation = write_trajectory_event(
        writer,
        condition,
        seed_row,
        cells,
        algotype_by_cell_id,
        goal_directions,
        event_index=0,
        event_kind="initial",
        swap_count=0,
        archived_compare_and_swap_count=0,
        scheduler_round=0,
        random_null_expected=random_null_expected,
    )
    inc_raw_points.append(float(initial_inc_raw))
    inc_pct_points.append(float(initial_inc_pct))
    dec_raw_points.append(float(initial_dec_raw))
    dec_pct_points.append(float(initial_dec_pct))
    aggregation_points.append(float(initial_aggregation))
    aggregation_minus_null_points.append(float(initial_aggregation - random_null_expected))
    sample_event_indices.append(0)

    while True:
        if status_probe.swap_count >= max_swaps:
            stop_reason = "max_step_cap"
            break
        if status_probe.compare_and_swap_count + status_probe.swap_count >= max_comparisons:
            stop_reason = "max_comparison_cap"
            break
        if max_rounds is not None and scheduler_rounds >= max_rounds:
            stop_reason = "max_round_cap"
            break

        swaps_before_round = status_probe.swap_count
        order = scheduler_rng.permutation(n)
        for cell_index in order:
            actor = cell_objects[int(cell_index)]
            actor_cell_id = int(actor.threadID)
            actor_algotype = algotype_by_cell_id[actor_cell_id]
            swaps_before = status_probe.swap_count
            compare_before = status_probe.compare_and_swap_count
            actor.move()
            swap_delta = status_probe.swap_count - swaps_before
            if swap_delta > 0:
                per_algotype_moves[actor_algotype] = per_algotype_moves.get(actor_algotype, 0) + int(swap_delta)
                for event_number in range(swaps_before + 1, status_probe.swap_count + 1):
                    if event_number <= TRACE_DENSE_EVENT_LIMIT or event_number % TRACE_SAMPLE_INTERVAL == 0:
                        inc_raw, inc_pct, dec_raw, dec_pct, aggregation = write_trajectory_event(
                            writer,
                            condition,
                            seed_row,
                            cells,
                            algotype_by_cell_id,
                            goal_directions,
                            event_index=event_number,
                            event_kind="sampled_swap",
                            swap_count=event_number,
                            archived_compare_and_swap_count=status_probe.compare_and_swap_count,
                            scheduler_round=scheduler_rounds,
                            random_null_expected=random_null_expected,
                            actor_cell_id=actor_cell_id,
                            actor_algotype=actor_algotype,
                        )
                        inc_raw_points.append(float(inc_raw))
                        inc_pct_points.append(float(inc_pct))
                        dec_raw_points.append(float(dec_raw))
                        dec_pct_points.append(float(dec_pct))
                        aggregation_points.append(float(aggregation))
                        aggregation_minus_null_points.append(float(aggregation - random_null_expected))
                        sample_event_indices.append(int(event_number))
            if status_probe.compare_and_swap_count < compare_before:
                raise RuntimeError("compare counter moved backwards")
            if status_probe.swap_count >= max_swaps:
                stop_reason = "max_step_cap"
                break
            if status_probe.compare_and_swap_count + status_probe.swap_count >= max_comparisons:
                stop_reason = "max_comparison_cap"
                break
        scheduler_rounds += 1
        if status_probe.swap_count == swaps_before_round:
            no_progress_rounds += 1
            if common.any_cell_can_move(cell_objects):
                no_move_checks = 0
            else:
                no_move_checks += 1
        else:
            no_progress_rounds = 0
            no_move_checks = 0
        if no_move_checks >= no_move_checks_required:
            stop_reason = "no_cell_can_move_after_two_checks"
            break
        if no_progress_rounds >= stable_no_progress_round_cap:
            stop_reason = "stable_no_progress_round_cap"
            break
        if stop_reason in {"max_step_cap", "max_comparison_cap", "max_round_cap"}:
            break

    final_values = common.cells_values(cells)
    final_algotypes = common.cell_algotypes(cells, algotype_by_cell_id)
    final_inc_raw = sortedness_raw(final_values, "increasing")
    final_dec_raw = sortedness_raw(final_values, "decreasing")
    final_inc_pct = sortedness_percent(final_inc_raw, n)
    final_dec_pct = sortedness_percent(final_dec_raw, n)
    final_aggregation = common.aggregation_value_from_algotypes(final_algotypes)
    final_swap_count = int(status_probe.swap_count)
    if sample_event_indices[-1] != final_swap_count:
        inc_raw, inc_pct, dec_raw, dec_pct, aggregation = write_trajectory_event(
            writer,
            condition,
            seed_row,
            cells,
            algotype_by_cell_id,
            goal_directions,
            event_index=final_swap_count,
            event_kind="final",
            swap_count=final_swap_count,
            archived_compare_and_swap_count=status_probe.compare_and_swap_count,
            scheduler_round=scheduler_rounds,
            random_null_expected=random_null_expected,
        )
        inc_raw_points.append(float(inc_raw))
        inc_pct_points.append(float(inc_pct))
        dec_raw_points.append(float(dec_raw))
        dec_pct_points.append(float(dec_pct))
        aggregation_points.append(float(aggregation))
        aggregation_minus_null_points.append(float(aggregation - random_null_expected))
        sample_event_indices.append(final_swap_count)
    peak_index = int(np.argmax(np.asarray(aggregation_points, dtype=float))) if aggregation_points else 0
    peak_aggregation = float(aggregation_points[peak_index]) if aggregation_points else final_aggregation
    peak_swap_count = int(sample_event_indices[peak_index]) if sample_event_indices else 0
    peak_progress = float(peak_swap_count / final_swap_count) if final_swap_count > 0 else 0.0
    dominant_goal, dominant_algotype, dominance_match, strict_inc, strict_dec, equal_pairs = classify_dominance(
        final_values, goal_directions, condition["mixtureId"]
    )
    equilibrium_classification = classify_equilibrium(stop_reason, final_values, no_progress_rounds)
    completed = stop_reason in {"no_cell_can_move_after_two_checks", "stable_no_progress_round_cap"}

    return OppositeRunResult(
        completed=completed,
        stop_reason=stop_reason,
        equilibrium_classification=equilibrium_classification,
        initial_values=initial_values,
        final_values=final_values,
        initial_algotypes=initial_algotypes,
        final_algotypes=final_algotypes,
        swap_count=final_swap_count,
        comparison_count=int(status_probe.compare_and_swap_count + status_probe.swap_count),
        archived_compare_and_swap_count=int(status_probe.compare_and_swap_count),
        scheduler_rounds=int(scheduler_rounds),
        no_progress_rounds=int(no_progress_rounds),
        event_count=int(status_probe.swap_count + 1),
        sampled_trace_row_count=len(sample_event_indices),
        wall_time_seconds=time.perf_counter() - start,
        per_algotype_moves=per_algotype_moves,
        initial_increasing_raw=int(initial_inc_raw),
        initial_increasing_percent=float(initial_inc_pct),
        initial_decreasing_raw=int(initial_dec_raw),
        initial_decreasing_percent=float(initial_dec_pct),
        final_increasing_raw=int(final_inc_raw),
        final_increasing_percent=float(final_inc_pct),
        final_decreasing_raw=int(final_dec_raw),
        final_decreasing_percent=float(final_dec_pct),
        final_strict_increasing_pairs=int(strict_inc),
        final_strict_decreasing_pairs=int(strict_dec),
        final_equal_pairs=int(equal_pairs),
        initial_aggregation=float(initial_aggregation),
        final_aggregation=float(final_aggregation),
        peak_aggregation=float(peak_aggregation),
        peak_aggregation_swap_count=int(peak_swap_count),
        peak_aggregation_normalized_progress=peak_progress,
        random_null_expected=float(random_null_expected),
        dominant_goal_direction=dominant_goal,
        dominant_algotype=dominant_algotype,
        expected_dominant_algotype=EXPECTED_DOMINANT_ALGOTYPE[condition["mixtureId"]],
        dominance_matches_paper=bool(dominance_match),
        final_goal_sortedness_json=json.dumps(goal_sortedness_by_algotype(final_values, final_algotypes, goal_directions), sort_keys=True, separators=(",", ":")),
        normalized_increasing_raw=interpolate_sampled_points(sample_event_indices, inc_raw_points, final_swap_count),
        normalized_increasing_percent=interpolate_sampled_points(sample_event_indices, inc_pct_points, final_swap_count),
        normalized_decreasing_raw=interpolate_sampled_points(sample_event_indices, dec_raw_points, final_swap_count),
        normalized_decreasing_percent=interpolate_sampled_points(sample_event_indices, dec_pct_points, final_swap_count),
        normalized_aggregation=interpolate_sampled_points(sample_event_indices, aggregation_points, final_swap_count),
        normalized_aggregation_minus_null=interpolate_sampled_points(sample_event_indices, aggregation_minus_null_points, final_swap_count),
    )


def make_replicate_record(condition: dict[str, Any], seed_row: dict[str, Any], result: OppositeRunResult) -> dict[str, Any]:
    allocation = json.loads(condition["algotypeAllocation"])
    goal_directions = json.loads(condition["goalDirections"])
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": condition["conditionId"],
        "implementation": condition["implementation"],
        "mixtureId": condition["mixtureId"],
        "algorithms": condition["algorithms"],
        "inputProfile": condition["inputProfile"],
        "goalDirectionsJson": json.dumps(goal_directions, sort_keys=True, separators=(",", ":")),
        "replicateIndex": int(seed_row["replicateIndex"]),
        "replicateNumber": int(seed_row["replicateNumber"]),
        "inputPermutationSeed": common.parse_int(seed_row["inputPermutationSeed"]),
        "algotypeAssignmentSeed": common.parse_int(seed_row["algotypeAssignmentSeed"]),
        "schedulerSeed": common.parse_int(seed_row["schedulerSeed"]),
        "tieBreakerSeed": common.parse_int(seed_row["tieBreakerSeed"]),
        "n": common.parse_int(condition["n"]),
        "targetRepeatCount": common.parse_int(condition["repeatCount"]),
        "algotypeAllocationJson": json.dumps(allocation, sort_keys=True, separators=(",", ":")),
        "initialAlgotypeCountsJson": json.dumps(dict(Counter(result.initial_algotypes)), sort_keys=True, separators=(",", ":")),
        "finalAlgotypeCountsJson": json.dumps(dict(Counter(result.final_algotypes)), sort_keys=True, separators=(",", ":")),
        "initialValueCountsJson": common.value_counts_json(result.initial_values),
        "finalValueCountsJson": common.value_counts_json(result.final_values),
        "initialAlgotypeSequence": common.algotype_sequence(result.initial_algotypes),
        "finalAlgotypeSequence": common.algotype_sequence(result.final_algotypes),
        "initialStateHash": common.state_hash(result.initial_values),
        "finalStateHash": common.state_hash(result.final_values),
        "initialValuesJson": json.dumps(result.initial_values, separators=(",", ":")),
        "finalValuesJson": json.dumps(result.final_values, separators=(",", ":")),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "equilibriumClassification": result.equilibrium_classification,
        "initialIncreasingSortednessRaw": result.initial_increasing_raw,
        "initialIncreasingSortednessPercent": result.initial_increasing_percent,
        "initialDecreasingSortednessRaw": result.initial_decreasing_raw,
        "initialDecreasingSortednessPercent": result.initial_decreasing_percent,
        "finalIncreasingSortednessRaw": result.final_increasing_raw,
        "finalIncreasingSortednessPercent": result.final_increasing_percent,
        "finalDecreasingSortednessRaw": result.final_decreasing_raw,
        "finalDecreasingSortednessPercent": result.final_decreasing_percent,
        "finalStrictIncreasingPairs": result.final_strict_increasing_pairs,
        "finalStrictDecreasingPairs": result.final_strict_decreasing_pairs,
        "finalEqualPairs": result.final_equal_pairs,
        "dominantGoalDirection": result.dominant_goal_direction,
        "dominantAlgotype": result.dominant_algotype,
        "expectedDominantAlgotype": result.expected_dominant_algotype,
        "dominanceMatchesPaper": result.dominance_matches_paper,
        "finalGoalSortednessJson": result.final_goal_sortedness_json,
        "randomNullExpected": result.random_null_expected,
        "initialAggregation": result.initial_aggregation,
        "finalAggregation": result.final_aggregation,
        "finalAggregationMinusNull": result.final_aggregation - result.random_null_expected,
        "peakAggregation": result.peak_aggregation,
        "peakAggregationMinusNull": result.peak_aggregation - result.random_null_expected,
        "peakAggregationSwapCount": result.peak_aggregation_swap_count,
        "peakAggregationNormalizedProgress": result.peak_aggregation_normalized_progress,
        "swapCount": result.swap_count,
        "comparisonCount": result.comparison_count,
        "archivedCompareAndSwapCount": result.archived_compare_and_swap_count,
        "schedulerRounds": result.scheduler_rounds,
        "noProgressRoundsAtStop": result.no_progress_rounds,
        "eventCount": result.event_count,
        "sampledTraceRowCount": result.sampled_trace_row_count,
        "traceDenseEventLimit": TRACE_DENSE_EVENT_LIMIT,
        "traceSampleInterval": TRACE_SAMPLE_INTERVAL,
        "wallTimeSeconds": result.wall_time_seconds,
        "perAlgotypeMovesJson": json.dumps(result.per_algotype_moves, sort_keys=True, separators=(",", ":")),
    }


def make_movement_records(condition: dict[str, Any], seed_row: dict[str, Any], result: OppositeRunResult) -> list[dict[str, Any]]:
    allocation = json.loads(condition["algotypeAllocation"])
    goal_directions = json.loads(condition["goalDirections"])
    records = []
    for algotype in common.ALGORITHMS:
        if int(allocation.get(algotype, 0)) <= 0:
            continue
        moves = int(result.per_algotype_moves.get(algotype, 0))
        records.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "conditionId": condition["conditionId"],
                "mixtureId": condition["mixtureId"],
                "inputProfile": condition["inputProfile"],
                "replicateIndex": int(seed_row["replicateIndex"]),
                "replicateNumber": int(seed_row["replicateNumber"]),
                "algotype": algotype,
                "goalDirection": goal_directions[algotype],
                "allocatedCellCount": int(allocation.get(algotype, 0)),
                "moveCount": moves,
                "moveShare": float(moves / result.swap_count) if result.swap_count else 0.0,
            }
        )
    return records


def make_curve_records(condition: dict[str, Any], seed_row: dict[str, Any], result: OppositeRunResult) -> list[dict[str, Any]]:
    rows = []
    goal_directions = json.loads(condition["goalDirections"])
    for bin_index, progress in enumerate(PROGRESS_BINS):
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "conditionId": condition["conditionId"],
                "mixtureId": condition["mixtureId"],
                "algorithms": condition["algorithms"],
                "inputProfile": condition["inputProfile"],
                "goalDirectionsJson": json.dumps(goal_directions, sort_keys=True, separators=(",", ":")),
                "replicateIndex": int(seed_row["replicateIndex"]),
                "replicateNumber": int(seed_row["replicateNumber"]),
                "progressBin": int(bin_index),
                "normalizedProgress": float(progress),
                "finalSwapCount": int(result.swap_count),
                "increasingSortednessRaw": float(result.normalized_increasing_raw[bin_index]),
                "increasingSortednessPercent": float(result.normalized_increasing_percent[bin_index]),
                "decreasingSortednessRaw": float(result.normalized_decreasing_raw[bin_index]),
                "decreasingSortednessPercent": float(result.normalized_decreasing_percent[bin_index]),
                "aggregationValue": float(result.normalized_aggregation[bin_index]),
                "randomNullExpected": float(result.random_null_expected),
                "aggregationMinusNull": float(result.normalized_aggregation_minus_null[bin_index]),
            }
        )
    return rows


def run_replicate_task(task: dict[str, Any]) -> dict[str, Any]:
    writer = BufferedTrajectoryWriter()
    result = run_opposite_chimera(
        task["condition"],
        task["seedRow"],
        writer,
        max_swaps=task["maxSwaps"],
        max_comparisons=task["maxComparisons"],
        max_rounds=task["maxRounds"],
        no_move_checks_required=task["noMoveChecksRequired"],
        stable_no_progress_round_cap=task["stableNoProgressRoundCap"],
    )
    return {
        "taskIndex": int(task["taskIndex"]),
        "replicateRecord": make_replicate_record(task["condition"], task["seedRow"], result),
        "movementRecords": make_movement_records(task["condition"], task["seedRow"], result),
        "curveRecords": make_curve_records(task["condition"], task["seedRow"], result),
        "traceRows": writer.rows,
        "traceRowCount": writer.row_count,
    }


def summarize_replicates(replicate_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        replicate_df.groupby(["conditionId", "mixtureId", "algorithms", "inputProfile", "goalDirectionsJson"], dropna=False)
        .agg(
            replicateCount=("replicateIndex", "count"),
            completedCount=("completed", "sum"),
            noMoveStopCount=("stopReason", lambda values: int((values == "no_cell_can_move_after_two_checks").sum())),
            stableNoProgressStopCount=("stopReason", lambda values: int((values == "stable_no_progress_round_cap").sum())),
            capStopCount=("equilibriumClassification", lambda values: int((values == "cap_truncated").sum())),
            meanInitialIncreasingSortednessRaw=("initialIncreasingSortednessRaw", "mean"),
            meanFinalIncreasingSortednessRaw=("finalIncreasingSortednessRaw", "mean"),
            meanFinalIncreasingSortednessPercent=("finalIncreasingSortednessPercent", "mean"),
            meanFinalDecreasingSortednessRaw=("finalDecreasingSortednessRaw", "mean"),
            meanFinalDecreasingSortednessPercent=("finalDecreasingSortednessPercent", "mean"),
            meanFinalAggregation=("finalAggregation", "mean"),
            sdFinalAggregation=("finalAggregation", "std"),
            meanPeakAggregation=("peakAggregation", "mean"),
            sdPeakAggregation=("peakAggregation", "std"),
            meanPeakAggregationProgress=("peakAggregationNormalizedProgress", "mean"),
            meanSwapCount=("swapCount", "mean"),
            sdSwapCount=("swapCount", "std"),
            meanComparisonCount=("comparisonCount", "mean"),
            meanSchedulerRounds=("schedulerRounds", "mean"),
            meanWallTimeSeconds=("wallTimeSeconds", "mean"),
            dominanceMatchesPaperCount=("dominanceMatchesPaper", "sum"),
        )
        .reset_index()
    )
    summary["dominanceMatchesPaperRate"] = summary["dominanceMatchesPaperCount"] / summary["replicateCount"]
    summary["expectedDominantAlgotype"] = summary["mixtureId"].map(EXPECTED_DOMINANT_ALGOTYPE)
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    profile_order = {profile: index for index, profile in enumerate(INPUT_PROFILE_ORDER)}
    summary["mixtureOrder"] = summary["mixtureId"].map(order)
    summary["profileOrder"] = summary["inputProfile"].map(profile_order)
    return summary.sort_values(["profileOrder", "mixtureOrder"]).reset_index(drop=True)


def summarize_movements(movement_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        movement_df.groupby(["conditionId", "mixtureId", "inputProfile", "algotype", "goalDirection"], dropna=False)
        .agg(
            replicateCount=("replicateIndex", "count"),
            allocatedCellCount=("allocatedCellCount", "first"),
            meanMoveCount=("moveCount", "mean"),
            sdMoveCount=("moveCount", "std"),
            meanMoveShare=("moveShare", "mean"),
        )
        .reset_index()
    )
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    profile_order = {profile: index for index, profile in enumerate(INPUT_PROFILE_ORDER)}
    summary["mixtureOrder"] = summary["mixtureId"].map(order)
    summary["profileOrder"] = summary["inputProfile"].map(profile_order)
    summary["algotypeOrder"] = summary["algotype"].map({algorithm: index for index, algorithm in enumerate(common.ALGORITHMS)})
    return summary.sort_values(["profileOrder", "mixtureOrder", "algotypeOrder"]).reset_index(drop=True)


def summarize_curves(curve_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        curve_df.groupby(
            ["conditionId", "mixtureId", "algorithms", "inputProfile", "goalDirectionsJson", "progressBin", "normalizedProgress"],
            dropna=False,
        )
        .agg(
            replicateCount=("replicateIndex", "count"),
            increasingSortednessRawMean=("increasingSortednessRaw", "mean"),
            increasingSortednessRawSd=("increasingSortednessRaw", "std"),
            increasingSortednessPercentMean=("increasingSortednessPercent", "mean"),
            decreasingSortednessRawMean=("decreasingSortednessRaw", "mean"),
            decreasingSortednessRawSd=("decreasingSortednessRaw", "std"),
            decreasingSortednessPercentMean=("decreasingSortednessPercent", "mean"),
            aggregationMean=("aggregationValue", "mean"),
            aggregationSd=("aggregationValue", "std"),
            aggregationMinusNullMean=("aggregationMinusNull", "mean"),
        )
        .reset_index()
    )
    for prefix in ["increasingSortednessRaw", "decreasingSortednessRaw", "aggregation"]:
        sd_col = f"{prefix}Sd"
        mean_col = f"{prefix}Mean"
        sem_col = f"{prefix}Sem"
        lower_col = f"{prefix}Ci95Lower"
        upper_col = f"{prefix}Ci95Upper"
        summary[sem_col] = summary[sd_col].fillna(0.0) / np.sqrt(summary["replicateCount"])
        summary[lower_col] = summary[mean_col] - 1.96 * summary[sem_col]
        summary[upper_col] = summary[mean_col] + 1.96 * summary[sem_col]
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    profile_order = {profile: index for index, profile in enumerate(INPUT_PROFILE_ORDER)}
    summary["mixtureOrder"] = summary["mixtureId"].map(order)
    summary["profileOrder"] = summary["inputProfile"].map(profile_order)
    return summary.sort_values(["profileOrder", "mixtureOrder", "progressBin"]).reset_index(drop=True)


def summarize_equilibria(replicate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition_id, mixture_id, input_profile), group in replicate_df.groupby(["conditionId", "mixtureId", "inputProfile"], sort=False):
        dominant_counts = group["dominantAlgotype"].value_counts().to_dict()
        equilibrium_counts = group["equilibriumClassification"].value_counts().to_dict()
        stop_counts = group["stopReason"].value_counts().to_dict()
        rows.append(
            {
                "conditionId": condition_id,
                "mixtureId": mixture_id,
                "inputProfile": input_profile,
                "replicateCount": int(len(group)),
                "expectedDominantAlgotype": EXPECTED_DOMINANT_ALGOTYPE[mixture_id],
                "modalDominantAlgotype": max(dominant_counts, key=dominant_counts.get),
                "dominanceMatchesPaperCount": int(group["dominanceMatchesPaper"].sum()),
                "dominanceMatchesPaperRate": float(group["dominanceMatchesPaper"].mean()),
                "dominantAlgotypeCountsJson": json.dumps(dominant_counts, sort_keys=True, separators=(",", ":")),
                "modalEquilibriumClassification": max(equilibrium_counts, key=equilibrium_counts.get),
                "equilibriumClassificationCountsJson": json.dumps(equilibrium_counts, sort_keys=True, separators=(",", ":")),
                "stopReasonCountsJson": json.dumps(stop_counts, sort_keys=True, separators=(",", ":")),
                "meanFinalIncreasingSortednessRaw": float(group["finalIncreasingSortednessRaw"].mean()),
                "meanFinalAggregation": float(group["finalAggregation"].mean()),
            }
        )
    eq = pd.DataFrame(rows)
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    profile_order = {profile: index for index, profile in enumerate(INPUT_PROFILE_ORDER)}
    eq["mixtureOrder"] = eq["mixtureId"].map(order)
    eq["profileOrder"] = eq["inputProfile"].map(profile_order)
    return eq.sort_values(["profileOrder", "mixtureOrder"]).reset_index(drop=True)


def build_paper_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        paper_raw = PAPER_UNIQUE_FINAL_SORTEDNESS_RAW.get(row["mixtureId"], np.nan)
        if row["inputProfile"] != "unique_1_100":
            paper_raw = np.nan
        rows.append(
            {
                "conditionId": row["conditionId"],
                "mixtureId": row["mixtureId"],
                "inputProfile": row["inputProfile"],
                "meanFinalIncreasingSortednessRaw": float(row["meanFinalIncreasingSortednessRaw"]),
                "paperUniqueFinalSortednessRaw": float(paper_raw),
                "differenceFromPaperRaw": float(row["meanFinalIncreasingSortednessRaw"] - paper_raw) if math.isfinite(float(paper_raw)) else np.nan,
                "meanFinalAggregation": float(row["meanFinalAggregation"]),
                "meanPeakAggregation": float(row["meanPeakAggregation"]),
                "dominanceMatchesPaperRate": float(row["dominanceMatchesPaperRate"]),
            }
        )
    return pd.DataFrame(rows)


def write_result_tables(
    replicate_df: pd.DataFrame,
    movement_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    results_dir: Path,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "replicateParquet": results_dir / "e01_opposite_direction_chimeras.parquet",
        "replicateCsv": results_dir / "e01_opposite_direction_chimeras.csv",
        "summaryParquet": results_dir / "e01_opposite_direction_chimera_summary.parquet",
        "summaryCsv": results_dir / "e01_opposite_direction_chimera_summary.csv",
        "movementParquet": results_dir / "e01_opposite_direction_chimera_algotype_movements.parquet",
        "movementCsv": results_dir / "e01_opposite_direction_chimera_algotype_movements.csv",
        "movementSummaryParquet": results_dir / "e01_opposite_direction_chimera_algotype_movement_summary.parquet",
        "movementSummaryCsv": results_dir / "e01_opposite_direction_chimera_algotype_movement_summary.csv",
        "curveParquet": results_dir / "e01_opposite_direction_trajectory_curves.parquet",
        "curveCsv": results_dir / "e01_opposite_direction_trajectory_curves.csv",
        "curveSummaryParquet": results_dir / "e01_opposite_direction_trajectory_summary.parquet",
        "curveSummaryCsv": results_dir / "e01_opposite_direction_trajectory_summary.csv",
        "equilibriumParquet": results_dir / "e01_opposite_direction_equilibria.parquet",
        "equilibriumCsv": results_dir / "e01_opposite_direction_equilibria.csv",
        "paperComparisonParquet": results_dir / "e01_opposite_direction_paper_comparison.parquet",
        "paperComparisonCsv": results_dir / "e01_opposite_direction_paper_comparison.csv",
    }
    summary_df = summarize_replicates(replicate_df)
    movement_summary_df = summarize_movements(movement_df)
    curve_summary_df = summarize_curves(curve_df)
    equilibrium_df = summarize_equilibria(replicate_df)
    paper_comparison_df = build_paper_comparison(summary_df)

    replicate_df.to_parquet(paths["replicateParquet"], index=False)
    replicate_df.to_csv(paths["replicateCsv"], index=False)
    summary_df.to_parquet(paths["summaryParquet"], index=False)
    summary_df.to_csv(paths["summaryCsv"], index=False)
    movement_df.to_parquet(paths["movementParquet"], index=False)
    movement_df.to_csv(paths["movementCsv"], index=False)
    movement_summary_df.to_parquet(paths["movementSummaryParquet"], index=False)
    movement_summary_df.to_csv(paths["movementSummaryCsv"], index=False)
    curve_df.to_parquet(paths["curveParquet"], index=False)
    curve_df.to_csv(paths["curveCsv"], index=False)
    curve_summary_df.to_parquet(paths["curveSummaryParquet"], index=False)
    curve_summary_df.to_csv(paths["curveSummaryCsv"], index=False)
    equilibrium_df.to_parquet(paths["equilibriumParquet"], index=False)
    equilibrium_df.to_csv(paths["equilibriumCsv"], index=False)
    paper_comparison_df.to_parquet(paths["paperComparisonParquet"], index=False)
    paper_comparison_df.to_csv(paths["paperComparisonCsv"], index=False)
    return paths


def plot_opposite(curve_summary_path: Path, summary_path: Path, input_profile: str, figure_dir: Path) -> tuple[Path, Path]:
    curve = pd.read_parquet(curve_summary_path)
    summary = pd.read_parquet(summary_path)
    subset_profile = curve[curve["inputProfile"] == input_profile]
    summary_profile = summary[summary["inputProfile"] == input_profile]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), sharex=True)
    for ax, mixture in zip(axes, MIXTURE_ORDER, strict=True):
        panel = subset_profile[subset_profile["mixtureId"] == mixture].sort_values("progressBin")
        row = summary_profile[summary_profile["mixtureId"] == mixture].iloc[0]
        x = panel["normalizedProgress"].to_numpy(dtype=float) * 100.0
        sortedness = panel["increasingSortednessRawMean"].to_numpy(dtype=float)
        sorted_lo = panel["increasingSortednessRawCi95Lower"].to_numpy(dtype=float)
        sorted_hi = panel["increasingSortednessRawCi95Upper"].to_numpy(dtype=float)
        aggregation = panel["aggregationMean"].to_numpy(dtype=float)
        agg_lo = panel["aggregationCi95Lower"].to_numpy(dtype=float)
        agg_hi = panel["aggregationCi95Upper"].to_numpy(dtype=float)
        ax.plot(x, sortedness, color="#1f77b4", linewidth=2.0, label="Increasing Sortedness")
        ax.fill_between(x, sorted_lo, sorted_hi, color="#1f77b4", alpha=0.15, linewidth=0)
        ax.axhline(float(row["meanFinalIncreasingSortednessRaw"]), color="#666666", linestyle="--", linewidth=1.0)
        ax.set_title(DISPLAY_NAMES[mixture], fontsize=10)
        ax.set_xlabel("Normalized process (%)")
        ax.set_ylabel("Increasing Sortedness (adjacent pairs)")
        ax.set_ylim(20, 100)
        ax.grid(True, color="#dddddd", linewidth=0.7)
        twin = ax.twinx()
        twin.plot(x, aggregation, color="#b3202a", linewidth=1.8, label="Aggregation")
        twin.fill_between(x, agg_lo, agg_hi, color="#b3202a", alpha=0.12, linewidth=0)
        twin.set_ylim(0.3, 1.0)
        if ax is axes[-1]:
            twin.set_ylabel("Aggregation", color="#b3202a")
        twin.tick_params(axis="y", labelcolor="#b3202a", labelsize=8)
        ax.text(
            0.03,
            0.96,
            f"final {row['meanFinalIncreasingSortednessRaw']:.1f}\n{row['expectedDominantAlgotype']} expected",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
        )
    fig.suptitle(
        "E01 S12 Figure 9-style unique opposite-direction chimeras"
        if input_profile == "unique_1_100"
        else "E01 S12 Figure 10-style repeated-value opposite-direction chimeras",
        y=1.04,
        fontsize=14,
    )
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    base = "figure09_unique_opposite" if input_profile == "unique_1_100" else "figure10_repeated_opposite"
    png = figure_dir / f"{base}.png"
    pdf = figure_dir / f"{base}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def validate_outputs(
    replicate_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    movement_df: pd.DataFrame,
    movement_summary_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    curve_summary_df: pd.DataFrame,
    equilibrium_df: pd.DataFrame,
    paper_comparison_df: pd.DataFrame,
    trace_row_count: int,
    output_paths: list[Path],
    expected_replicates_per_condition: int,
    artifacts_dir: Path,
) -> tuple[bool, list[str], list[str], str]:
    checks: list[str] = []
    failures: list[str] = []
    caveats: list[str] = []
    expected_conditions = len(MIXTURE_ORDER) * len(INPUT_PROFILE_ORDER)
    expected_rows = expected_conditions * expected_replicates_per_condition

    if len(replicate_df) == expected_rows:
        checks.append(f"Replicate table has {expected_rows} rows across six S12 conditions.")
    else:
        failures.append(f"Replicate table has {len(replicate_df)} rows; expected {expected_rows}.")
    if set(replicate_df["mixtureId"]) == set(MIXTURE_ORDER) and set(replicate_df["inputProfile"]) == set(INPUT_PROFILE_ORDER):
        checks.append("All three paper pairings are present for both unique and duplicate input profiles.")
    else:
        failures.append("Observed S12 mixtures or input profiles do not match the S03 condition matrix.")
    per_condition = replicate_df.groupby("conditionId").size()
    if per_condition.nunique() == 1 and int(per_condition.iloc[0]) == expected_replicates_per_condition:
        checks.append(f"Every S12 condition has {expected_replicates_per_condition} replicate rows.")
    else:
        failures.append(f"Per-condition replicate counts are inconsistent: {per_condition.to_dict()}.")

    if replicate_df["goalDirectionsJson"].notna().all() and replicate_df["goalDirectionsJson"].str.contains("decreasing").all():
        checks.append("Goal direction is stored for every replicate and each condition includes an opposing decreasing goal.")
    else:
        failures.append("Goal direction metadata is missing or does not encode opposing goals for every replicate.")

    profile_ok = True
    for _, row in replicate_df.iterrows():
        values = json.loads(row["initialValuesJson"])
        final_values = json.loads(row["finalValuesJson"])
        if not value_profile_valid(values, row["inputProfile"]) or common.value_counts_json(values) != row["finalValueCountsJson"]:
            profile_ok = False
            break
        if Counter(values) != Counter(final_values):
            profile_ok = False
            break
    if profile_ok:
        checks.append("Every run preserves its configured unique or duplicate value multiset.")
    else:
        failures.append("At least one run does not preserve the configured value multiset.")

    cap_count = int((replicate_df["equilibriumClassification"] == "cap_truncated").sum())
    unexpected_round_caps = int((replicate_df["stopReason"] == "max_round_cap").sum())
    if bool(replicate_df["completed"].all()):
        checks.append("Every S12 run reached a no-move or stable-no-progress equilibrium before caps.")
    else:
        checks.append("At least one S12 run reached an S03 cap and was classified as cap-truncated.")
        caveats.append("At least one S12 run did not reach a no-move equilibrium before the S03 event caps.")
    if cap_count == 0:
        checks.append("No S12 run was classified as cap-truncated.")
    else:
        checks.append(f"{cap_count} S12 runs were classified reproducibly as cap-truncated.")
        caveats.append("Cap-truncated S12 runs constrain the stable-equilibrium hypothesis for those conditions.")
    if unexpected_round_caps:
        failures.append(f"{unexpected_round_caps} S12 runs hit the optional round cap rather than the S03 event caps.")

    if trace_row_count == int(replicate_df["sampledTraceRowCount"].sum()):
        checks.append("S12 sampled trajectory row count equals summed per-replicate sampled trace rows.")
        caveats.append(
            f"S12 stores dense first-{TRACE_DENSE_EVENT_LIMIT} and every-{TRACE_SAMPLE_INTERVAL}-swap sampled traces rather than every swap event."
        )
    else:
        failures.append(
            f"S12 trace row count {trace_row_count} does not equal summed sampled trace count {int(replicate_df['sampledTraceRowCount'].sum())}."
        )
    if len(summary_df) == expected_conditions:
        checks.append("Condition summary has six rows.")
    else:
        failures.append(f"Condition summary has {len(summary_df)} rows, expected {expected_conditions}.")
    if len(movement_df) == expected_rows * 2 and len(movement_summary_df) == expected_conditions * 2:
        checks.append("Movement tables have expected pairwise Algotype rows.")
    else:
        failures.append("Movement table row counts are unexpected.")
    if len(curve_df) == expected_rows * 101 and len(curve_summary_df) == expected_conditions * 101:
        checks.append("Trajectory curve tables have 101 bins for every replicate and condition.")
    else:
        failures.append("Trajectory curve table row counts are unexpected.")
    if len(equilibrium_df) == expected_conditions and len(paper_comparison_df) == expected_conditions:
        checks.append("Equilibrium and paper-comparison tables have one row for every S12 condition.")
    else:
        failures.append("Equilibrium or paper-comparison table row counts are unexpected.")
    if bool((equilibrium_df["dominanceMatchesPaperRate"] >= 0.5).all()):
        checks.append("Every S12 condition has majority dominance classification matching the paper's Bubble > Selection > Insertion ordering.")
    else:
        caveats.append("At least one S12 condition lacks majority dominance matching the paper's Bubble > Selection > Insertion ordering.")
    if bool((summary_df["meanFinalAggregation"] > summary_df.groupby("inputProfile")["meanFinalAggregation"].transform(lambda _: 0.0)).all()):
        checks.append("Final Aggregation is finite and positive for every S12 condition.")
    else:
        failures.append("At least one S12 condition has invalid final Aggregation.")

    missing_outputs = [str(path) for path in output_paths if not path.exists() or path.stat().st_size == 0]
    if not missing_outputs:
        checks.append("All declared S12 output files exist and are non-empty.")
    else:
        failures.append(f"Missing or empty S12 output files: {missing_outputs}.")
    if (artifacts_dir / "research_steps" / "S13").exists():
        failures.append("S13 artifact directory exists; S12 must stop before S13.")
    else:
        checks.append("No S13 artifact directory was created.")
    if (artifacts_dir / "reports" / "e01_divergence_log.md").exists():
        failures.append("S13 divergence log exists; S12 must not start S13.")
    else:
        checks.append("No S13 divergence log was created.")

    success = not failures
    if success:
        validation_result = (
            "passed: S12 ran six opposite-direction chimera conditions, "
            f"{expected_rows} replicate rows, stored goal directions, classified equilibria and dominance, "
            "wrote Figure 9/10-style outputs and no S13 artifacts."
        )
    else:
        validation_result = "failed: " + " ".join(failures)
    return success, checks, failures + caveats, validation_result


def infer_outcome(summary_df: pd.DataFrame, equilibrium_df: pd.DataFrame) -> tuple[str, str]:
    no_caps = bool((summary_df["capStopCount"] == 0).all())
    majority_matches = bool((equilibrium_df["dominanceMatchesPaperRate"] >= 0.5).all())
    unique = summary_df[summary_df["inputProfile"] == "unique_1_100"].copy()
    unique_diffs = []
    for _, row in unique.iterrows():
        paper = PAPER_UNIQUE_FINAL_SORTEDNESS_RAW[row["mixtureId"]]
        unique_diffs.append(abs(float(row["meanFinalIncreasingSortednessRaw"]) - paper))
    numerically_close = all(diff <= 10.0 for diff in unique_diffs)
    if no_caps and majority_matches and numerically_close:
        return (
            "supportive",
            "S12 supports the opposite-direction claim: all runs reached non-cap equilibria, dominance classifications match the paper ordering by majority, and unique final Sortedness means are within 10 adjacent-pair counts of paper reports.",
        )
    if no_caps and majority_matches:
        return (
            "supportive",
            "S12 supports the qualitative opposite-direction claim because all runs reached non-cap equilibria and dominance classifications match the paper ordering by majority, although unique final Sortedness magnitudes differ from paper reports.",
        )
    if no_caps:
        return (
            "constraining",
            "S12 supports stable opposite-direction equilibria but constrains the reported dominance ordering because at least one condition lacks majority agreement with the paper's Bubble > Selection > Insertion pattern.",
        )
    return (
        "constraining",
        "S12 constrains the opposite-direction claim because at least one condition hit a cap rather than a classified equilibrium.",
    )


def write_methods(path: Path) -> None:
    lines = [
        "# S12 Methods",
        "",
        "- Research step ID: S12",
        "- Step number: 12",
        "- Completion status: completed",
        "- Artifacts written: see `artifact_manifest.json` and `status.json`.",
        "- Validation result: see `validation.json` and `validation.md`.",
        "- Caveats or blockers: S12 uses a deterministic pseudo-scheduler wrapper rather than live OS-thread scheduling and stops before S13.",
        "- Recommended next action: hand control back to Chief Scientist review before S13.",
        "",
        "## Conditions",
        "",
        "S12 runs the six S03-defined opposite-direction cell-view chimera conditions: Bubble decreasing with Selection increasing, Bubble increasing with Insertion decreasing, and Selection decreasing with Insertion increasing, each for unique values 1 through 100 and duplicate values 1 through 10 with ten copies each.",
        "",
        "## Stop And Equilibrium Classification",
        "",
        "The primary stop condition is a no-cell-can-move equilibrium after two no-progress checks. Runs also record stable-no-progress, swap, comparison, and round cap fallbacks. Equilibrium labels distinguish static no-move, stable no-progress, global increasing/decreasing sorted, active partial, and cap-truncated outcomes.",
        "",
        "## Directional Metrics",
        "",
        "S12 stores nondecreasing and nonincreasing adjacent-pair Sortedness, strict increasing/decreasing pair counts, Aggregation, per-Algotype movement counts, and per-Algotype goal-sortedness at the final state. Dominance is classified from strict increasing versus strict decreasing final adjacent-pair counts, then mapped to the Algotype assigned to that winning goal direction.",
        "",
        "## Trajectory Sampling",
        "",
        f"The collectible trajectory trace is sampled: all events through swap {TRACE_DENSE_EVENT_LIMIT} are stored, then every {TRACE_SAMPLE_INTERVAL}th swap plus the final state. The normalized 101-bin trajectory backing tables interpolate over the sampled event indices. Final state metrics, stop reasons, movement counts, and equilibrium classifications are exact for each run.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_markdown(path: Path, checks: list[str], caveats: list[str], validation_result: str) -> None:
    lines = [
        "# S12 Validation",
        "",
        "- Research step ID: S12",
        "- Step number: 12",
        "- Completion status: completed",
        "- Validation result: " + validation_result,
        "- Artifacts written: see `artifact_manifest.json` and `status.json`.",
        "- Caveats or blockers:",
    ]
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Recommended next action: hand control back to Chief Scientist review before S13.",
            "",
            "## Checks",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in checks)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(
    path: Path,
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
    outcome_classification: str,
    summary_df: pd.DataFrame,
    equilibrium_df: pd.DataFrame,
    paper_comparison_df: pd.DataFrame,
) -> None:
    summary_rows = [
        [
            row["inputProfile"],
            row["mixtureId"],
            int(row["replicateCount"]),
            row["meanFinalIncreasingSortednessRaw"],
            row["meanFinalAggregation"],
            row["meanPeakAggregation"],
            row["meanSwapCount"],
            row["dominanceMatchesPaperRate"],
        ]
        for _, row in summary_df.iterrows()
    ]
    eq_rows = [
        [
            row["inputProfile"],
            row["mixtureId"],
            row["modalDominantAlgotype"],
            row["expectedDominantAlgotype"],
            row["dominanceMatchesPaperRate"],
            row["modalEquilibriumClassification"],
        ]
        for _, row in equilibrium_df.iterrows()
    ]
    paper_rows = [
        [
            row["mixtureId"],
            row["meanFinalIncreasingSortednessRaw"],
            row["paperUniqueFinalSortednessRaw"],
            row["differenceFromPaperRaw"],
        ]
        for _, row in paper_comparison_df[paper_comparison_df["inputProfile"] == "unique_1_100"].iterrows()
    ]
    lines = [
        "# S12 Status Summary",
        "",
        "- Research step ID: S12",
        "- Step number: 12",
        "- Completion status: completed",
        "- Outcome classification: " + outcome_classification,
        "- Artifacts written:",
    ]
    lines.extend(f"- `{artifact}`" for artifact in artifacts_written)
    lines.extend(
        [
            "- Validation result: " + validation_result,
            "- Caveats or blockers:",
        ]
    )
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Lay summary: S12 mixed cells pursuing opposite increasing/decreasing goals. The runs produced stable equilibrium classifications, Aggregation curves, and Figure 9/10-style plots for unique and duplicate-value arrays.",
            "- Recommended next action: " + recommended_next_action,
            "",
            "## Condition Means",
            "",
            common.markdown_table(
                [
                    "Input profile",
                    "Mixture",
                    "Replicates",
                    "Final increasing Sortedness",
                    "Final Aggregation",
                    "Peak Aggregation",
                    "Mean swaps",
                    "Dominance match rate",
                ],
                summary_rows,
            ),
            "",
            "## Equilibrium Classifications",
            "",
            common.markdown_table(
                [
                    "Input profile",
                    "Mixture",
                    "Modal dominant Algotype",
                    "Expected dominant Algotype",
                    "Match rate",
                    "Modal equilibrium",
                ],
                eq_rows,
            ),
            "",
            "## Unique Paper Comparison",
            "",
            common.markdown_table(
                ["Mixture", "Observed final increasing Sortedness", "Paper final Sortedness", "Difference"],
                paper_rows,
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_s12(args: argparse.Namespace) -> int:
    generated_at = common.utc_now()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    results_dir = artifacts_dir / "results"
    figure_dir = artifacts_dir / "figures" / "e01"
    trace_dir = artifacts_dir / "traces" / "e01" / STEP_ID
    provenance_dir = artifacts_dir / "provenance"
    run_manifest_path = provenance_dir / "run_manifest.json"
    for directory in [step_dir, code_dir, results_dir, figure_dir, trace_dir, provenance_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    config = common.load_json(args.config)
    stop_policy = config.get("semantics", {}).get("stopPolicies", {}).get("stable_no_move_or_cap", {})
    max_swaps = int(stop_policy.get("maxSwapEvents", 750000))
    max_comparisons = int(stop_policy.get("maxComparisonEvents", 6000000))
    no_move_checks_required = int(stop_policy.get("noMoveChecksRequired", 2))
    max_rounds = int(args.max_rounds) if args.max_rounds and int(args.max_rounds) > 0 else None
    stable_no_progress_round_cap = int(args.stable_no_progress_round_cap)

    conditions = s12_conditions(args.condition_matrix)
    seed_rows_by_condition = s12_seed_rows(args.seed_table, args.max_replicates_per_condition)
    worker_count = max(1, min(8, int(args.workers)))
    tasks: list[dict[str, Any]] = []
    task_index = 0
    for condition in conditions:
        rows = seed_rows_by_condition.get(condition["conditionId"], [])
        if not rows:
            raise RuntimeError(f"No S12 seed rows found for {condition['conditionId']}")
        for seed_row in rows:
            tasks.append(
                {
                    "taskIndex": task_index,
                    "condition": condition,
                    "seedRow": seed_row,
                    "maxSwaps": max_swaps,
                    "maxComparisons": max_comparisons,
                    "maxRounds": max_rounds,
                    "noMoveChecksRequired": no_move_checks_required,
                    "stableNoProgressRoundCap": stable_no_progress_round_cap,
                }
            )
            task_index += 1
    replicate_records: list[dict[str, Any]] = []
    movement_records: list[dict[str, Any]] = []
    curve_records: list[dict[str, Any]] = []
    trace_path = trace_dir / "e01_s12_opposite_direction_trajectory_events.csv.gz"

    with TrajectoryWriter(trace_path) as trajectory_writer:
        if worker_count == 1:
            task_results = map(run_replicate_task, tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=worker_count)
            task_results = executor.map(run_replicate_task, tasks, chunksize=1)
        try:
            for task_result in task_results:
                replicate_records.append(task_result["replicateRecord"])
                movement_records.extend(task_result["movementRecords"])
                curve_records.extend(task_result["curveRecords"])
                for row in task_result["traceRows"]:
                    trajectory_writer.write(row)
        finally:
            if worker_count != 1:
                executor.shutdown(wait=True, cancel_futures=False)
        trace_row_count = trajectory_writer.row_count

    replicate_df = pd.DataFrame(replicate_records)
    movement_df = pd.DataFrame(movement_records)
    curve_df = pd.DataFrame(curve_records)
    result_paths = write_result_tables(replicate_df, movement_df, curve_df, results_dir)
    summary_df = pd.read_parquet(result_paths["summaryParquet"])
    movement_summary_df = pd.read_parquet(result_paths["movementSummaryParquet"])
    curve_summary_df = pd.read_parquet(result_paths["curveSummaryParquet"])
    equilibrium_df = pd.read_parquet(result_paths["equilibriumParquet"])
    paper_comparison_df = pd.read_parquet(result_paths["paperComparisonParquet"])
    fig09_png, fig09_pdf = plot_opposite(result_paths["curveSummaryParquet"], result_paths["summaryParquet"], "unique_1_100", figure_dir)
    fig10_png, fig10_pdf = plot_opposite(result_paths["curveSummaryParquet"], result_paths["summaryParquet"], "duplicate_1_10_x10", figure_dir)
    outcome_classification, outcome_reason = infer_outcome(summary_df, equilibrium_df)

    methods_md = step_dir / "methods.md"
    validation_json = step_dir / "validation.json"
    validation_md = step_dir / "validation.md"
    summary_md = step_dir / "summary.md"
    status_json = step_dir / "status.json"
    artifact_manifest_json = step_dir / "artifact_manifest.json"
    write_methods(methods_md)

    output_paths = [
        trace_path,
        *result_paths.values(),
        fig09_png,
        fig09_pdf,
        fig10_png,
        fig10_pdf,
        methods_md,
    ]
    expected_replicates = args.max_replicates_per_condition or 100
    success, validation_checks, caveats_or_blockers, validation_result = validate_outputs(
        replicate_df,
        summary_df,
        movement_df,
        movement_summary_df,
        curve_df,
        curve_summary_df,
        equilibrium_df,
        paper_comparison_df,
        trace_row_count,
        output_paths,
        expected_replicates_per_condition=expected_replicates,
        artifacts_dir=artifacts_dir,
    )
    caveats_or_blockers.extend(
        [
            "S12 uses the deterministic pseudo-scheduler wrapper around archived cell-view classes rather than live OS-thread scheduling.",
            "The active archived disorder script did not enumerate all Figure 9/10 pairings; S12 uses S03-explicit goal-direction mappings.",
            "Dominance classification is based on final strict increasing versus strict decreasing adjacent-pair counts, then mapped to the Algotype assigned to that direction.",
            outcome_reason,
            "No S13 reproducibility classification or S13 artifact directory was started.",
        ]
    )

    code_copy = code_dir / Path(__file__).name
    helper_copy = code_dir / "e01_s11_duplicate_value_chimeras.py"
    shutil.copy2(Path(__file__), code_copy)
    shutil.copy2(SCRIPT_DIR / "e01_s11_duplicate_value_chimeras.py", helper_copy)
    recommended_next_action = "Hand control back to the Chief Scientist workflow for review; start S13 reproducibility classification only after explicit instruction."
    source_statuses = {
        "S03": common.load_json(S03_STATUS_DEFAULT),
        "S11": common.load_json(S11_STATUS_DEFAULT),
    }
    runtime = {
        "pythonVersion": sys.version,
        "pythonExecutable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "osCpuCount": os.cpu_count(),
        "workerCount": worker_count,
        "gpuUsed": False,
        "numpyVersion": np.__version__,
        "pandasVersion": pd.__version__,
        "matplotlibVersion": matplotlib.__version__,
        "maxSwaps": max_swaps,
        "maxComparisons": max_comparisons,
        "maxRounds": max_rounds,
        "noMoveChecksRequired": no_move_checks_required,
        "stableNoProgressRoundCap": stable_no_progress_round_cap,
        "traceDenseEventLimit": TRACE_DENSE_EVENT_LIMIT,
        "traceSampleInterval": TRACE_SAMPLE_INTERVAL,
        "archivedSwapMode": "compact_no_internal_snapshot_retention",
    }
    artifacts_written_paths = [
        *output_paths,
        validation_json,
        validation_md,
        summary_md,
        status_json,
        artifact_manifest_json,
        code_copy,
        helper_copy,
        run_manifest_path,
    ]
    artifacts_written = [str(path) for path in sorted(set(artifacts_written_paths))]
    validation_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "validationResult": validation_result,
        "validationChecks": validation_checks,
        "caveatsOrBlockers": caveats_or_blockers,
        "recommendedNextAction": recommended_next_action,
        "artifactsWritten": artifacts_written,
        "replicateRows": int(len(replicate_df)),
        "summaryRows": int(len(summary_df)),
        "movementRows": int(len(movement_df)),
        "movementSummaryRows": int(len(movement_summary_df)),
        "curveRows": int(len(curve_df)),
        "curveSummaryRows": int(len(curve_summary_df)),
        "equilibriumRows": int(len(equilibrium_df)),
        "paperComparisonRows": int(len(paper_comparison_df)),
        "traceRows": int(trace_row_count),
        "outcomeClassification": outcome_classification,
    }
    write_validation_markdown(validation_md, validation_checks, caveats_or_blockers, validation_result)
    common.write_json(validation_json, validation_payload)

    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "outcomeClassification": outcome_classification,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats_or_blockers,
        "recommendedNextAction": recommended_next_action,
        "git": common.get_git_metadata(),
        "runtime": runtime,
        "sourceArtifacts": {
            "s03BaselineConfig": str(args.config),
            "s03ConditionMatrix": str(args.condition_matrix),
            "s03SeedTable": str(args.seed_table),
            "s03Status": str(S03_STATUS_DEFAULT),
            "s11Status": str(S11_STATUS_DEFAULT),
        },
        "sourceStatuses": {
            step_id: {
                "researchStepId": status.get("researchStepId"),
                "status": status.get("status"),
                "success": status.get("success"),
                "validationResult": status.get("validationResult"),
            }
            for step_id, status in source_statuses.items()
        },
        "conditionCounts": {
            "replicateRows": int(len(replicate_df)),
            "summaryRows": int(len(summary_df)),
            "movementRows": int(len(movement_df)),
            "movementSummaryRows": int(len(movement_summary_df)),
            "curveRows": int(len(curve_df)),
            "curveSummaryRows": int(len(curve_summary_df)),
            "equilibriumRows": int(len(equilibrium_df)),
            "paperComparisonRows": int(len(paper_comparison_df)),
            "traceRows": int(trace_row_count),
        },
        "mixtures": MIXTURE_ORDER,
        "inputProfiles": INPUT_PROFILE_ORDER,
        "expectedDominantAlgotype": EXPECTED_DOMINANT_ALGOTYPE,
    }
    common.write_json(status_json, status_payload)
    write_summary_markdown(
        summary_md,
        artifacts_written,
        validation_result,
        caveats_or_blockers,
        recommended_next_action,
        outcome_classification,
        summary_df,
        equilibrium_df,
        paper_comparison_df,
    )

    artifact_manifest_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": STATUS if success else "failed_validation",
        "success": bool(success),
        "generatedAt": generated_at,
        "manifestSelfPath": str(artifact_manifest_json),
        "provenanceManifestPath": str(run_manifest_path),
        "artifactCount": 0,
        "artifacts": [],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats_or_blockers,
        "recommendedNextAction": recommended_next_action,
    }
    common.write_json(artifact_manifest_json, artifact_manifest_payload)
    common.update_run_manifest(run_manifest_path, status_payload, artifact_manifest_payload)
    artifacts = common.collect_artifacts(artifacts_written_paths, manifest_path=artifact_manifest_json)
    artifact_manifest_payload["artifactCount"] = len(artifacts)
    artifact_manifest_payload["artifacts"] = artifacts
    common.write_json(artifact_manifest_json, artifact_manifest_payload)
    common.update_run_manifest(run_manifest_path, status_payload, artifact_manifest_payload)

    print(json.dumps({"success": success, "validationResult": validation_result, "artifactsWritten": artifacts_written}, indent=2))
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH_DEFAULT)
    parser.add_argument("--condition-matrix", type=Path, default=CONDITION_MATRIX_DEFAULT)
    parser.add_argument("--seed-table", type=Path, default=SEED_TABLE_DEFAULT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--max-rounds", type=int, default=None, help="Optional diagnostic round cap; disabled by default for S03 parity.")
    parser.add_argument("--stable-no-progress-round-cap", type=int, default=100)
    parser.add_argument("--max-replicates-per-condition", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    return run_s12(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
