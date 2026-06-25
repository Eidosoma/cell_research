#!/usr/bin/env python3
"""Run E01 S11 duplicate-value same-goal mixed Algotype chimeras."""

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
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.multithread.BubbleSortCell import BubbleSortCell
from modules.multithread.CellGroup import CellGroup, GroupStatus
from modules.multithread.InsertionSortCell import InsertionSortCell
from modules.multithread.SelectionSortCell import SelectionSortCell
from modules.multithread.StatusProbe import StatusProbe


EXPERIMENT_ID = "E01"
STEP_ID = "S11"
STEP_NUMBER = 11
STATUS = "completed"
CONFIG_PATH_DEFAULT = Path("/artifacts/configs/e01_baseline_config.json")
CONDITION_MATRIX_DEFAULT = Path("/artifacts/research_steps/S03/condition_matrix.csv")
SEED_TABLE_DEFAULT = Path("/artifacts/research_steps/S03/seed_table.csv")
S03_STATUS_DEFAULT = Path("/artifacts/research_steps/S03/status.json")
S09_STATUS_DEFAULT = Path("/artifacts/research_steps/S09/status.json")
S10_STATUS_DEFAULT = Path("/artifacts/research_steps/S10/status.json")
S10_PEAK_SUMMARY_DEFAULT = Path("/artifacts/results/e01_aggregation_peak_summary.parquet")
S10_CURVE_SUMMARY_DEFAULT = Path("/artifacts/results/e01_aggregation_curve_summary.parquet")
ALGORITHMS = ["bubble", "insertion", "selection"]
MIXTURE_ORDER = ["bubble_insertion", "bubble_selection", "insertion_selection"]
CELL_CLASSES = {
    "bubble": BubbleSortCell,
    "insertion": InsertionSortCell,
    "selection": SelectionSortCell,
}
ALGOTYPE_CODES = {
    "bubble": "B",
    "insertion": "I",
    "selection": "S",
}
DISPLAY_NAMES = {
    "bubble_insertion": "Bubble + Insertion",
    "bubble_selection": "Bubble + Selection",
    "insertion_selection": "Insertion + Selection",
}
PAPER_DUPLICATE_REPORTS = {
    "bubble_insertion": {
        "paperDuplicatePeakAggregation": 0.63,
        "paperDuplicatePeakProgress": 0.13,
        "paperDuplicateFinalAggregation": np.nan,
    },
    "bubble_selection": {
        "paperDuplicatePeakAggregation": 0.69,
        "paperDuplicatePeakProgress": 1.00,
        "paperDuplicateFinalAggregation": 0.65,
    },
    "insertion_selection": {
        "paperDuplicatePeakAggregation": 0.71,
        "paperDuplicatePeakProgress": 1.00,
        "paperDuplicateFinalAggregation": 0.70,
    },
}
PROGRESS_BINS = np.linspace(0.0, 1.0, 101)


@dataclass
class DuplicateRunResult:
    completed: bool
    stop_reason: str
    initial_values: list[int]
    final_values: list[int]
    initial_algotypes: list[str]
    final_algotypes: list[str]
    swap_count: int
    comparison_count: int
    archived_compare_and_swap_count: int
    scheduler_rounds: int
    event_count: int
    wall_time_seconds: float
    per_algotype_moves: dict[str, int]
    initial_sortedness_raw_count: int
    initial_sortedness_percent: float
    final_sortedness_raw_count: int
    final_sortedness_percent: float
    final_monotonicity_error: int
    initial_aggregation: float
    final_aggregation: float
    peak_aggregation: float
    peak_aggregation_swap_count: int
    peak_aggregation_normalized_progress: float
    random_null_expected: float
    normalized_sortedness: list[float]
    normalized_aggregation: list[float]
    normalized_aggregation_minus_null: list[float]


class TrajectoryWriter:
    fieldnames = [
        "conditionId",
        "mixtureId",
        "algorithms",
        "inputProfile",
        "replicateIndex",
        "replicateNumber",
        "eventIndex",
        "eventKind",
        "swapCount",
        "comparisonCount",
        "archivedCompareAndSwapCount",
        "schedulerRound",
        "sortednessRawCount",
        "sortednessPercent",
        "monotonicityError",
        "aggregationValue",
        "randomNullExpected",
        "aggregationMinusNull",
        "actorCellId",
        "actorAlgotype",
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover
        return {
            "args": args,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"] if commit["ok"] else "unknown",
        "branch": branch["stdout"] if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"],
        "remote": remote["stdout"],
    }


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_artifacts(paths: list[Path], manifest_path: Path | None = None) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if manifest_path is not None and path.resolve() == manifest_path.resolve():
            continue
        if path.name == "run_manifest.json" and "provenance" in path.parts:
            continue
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append(
                {
                    "path": str(path),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    return sorted(artifacts, key=lambda item: item["path"])


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_int(value: Any, default: int | None = None) -> int:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        if default is None:
            raise ValueError("missing integer value")
        return default
    return int(float(value))


def sortedness_raw(values: list[int]) -> int:
    return sum(1 for i in range(len(values) - 1) if values[i] <= values[i + 1])


def sortedness_percent_from_raw(raw_count: int, n: int) -> float:
    if n < 2:
        return 100.0
    return 100.0 * raw_count / (n - 1)


def monotonicity_error_from_raw(raw_count: int, n: int) -> int:
    return (n - 1) - raw_count


def is_sorted_values(values: list[int]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def duplicate_values_from_seed(seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    values = np.repeat(np.arange(1, 11, dtype=np.int16), 10)
    rng.shuffle(values)
    return [int(value) for value in values]


def value_counts_json(values: list[int]) -> str:
    counts = {str(value): int(count) for value, count in sorted(Counter(values).items())}
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


def values_are_10_copies_each(values: list[int]) -> bool:
    return Counter(values) == Counter({value: 10 for value in range(1, 11)})


def algotypes_from_seed(seed: Any, allocation: dict[str, int]) -> list[str]:
    algotypes: list[str] = []
    for algorithm in ALGORITHMS:
        algotypes.extend([algorithm] * int(allocation.get(algorithm, 0)))
    rng = np.random.default_rng(parse_int(seed))
    indices = rng.permutation(len(algotypes))
    return [algotypes[int(index)] for index in indices]


def algotype_sequence(algotypes: list[str]) -> str:
    return "".join(ALGOTYPE_CODES[item] for item in algotypes)


def cells_values(cells: list[Any]) -> list[int]:
    return [int(cell.value) for cell in cells]


def cell_algotypes(cells: list[Any], algotype_by_cell_id: dict[int, str]) -> list[str]:
    return [algotype_by_cell_id[int(cell.threadID)] for cell in cells]


def state_hash(values: list[int]) -> str:
    arr = np.asarray(values, dtype=np.int16)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def aggregation_value_from_algotypes(algotypes: list[str]) -> float:
    if len(algotypes) < 2:
        return 0.0
    matches = sum(algotypes[idx] == algotypes[idx - 1] for idx in range(1, len(algotypes)))
    return matches / (len(algotypes) - 1)


def random_adjacency_expected_from_counts(counts: dict[str, int]) -> float:
    n = sum(int(value) for value in counts.values())
    if n < 2:
        return 0.0
    numerator = sum(int(count) * (int(count) - 1) for count in counts.values())
    return numerator / (n * (n - 1))


def any_cell_can_move(cell_objects: list[Any]) -> bool:
    return any(bool(cell.should_move()) for cell in cell_objects)


def interpolate_points(points: list[float], final_swap_count: int, bins: int = 101) -> list[float]:
    if not points:
        return [0.0] * bins
    if len(points) == 1 or final_swap_count <= 0:
        return [float(points[0])] * bins
    x = np.linspace(0.0, 1.0, num=len(points))
    target = np.linspace(0.0, 1.0, num=bins)
    return [float(value) for value in np.interp(target, x, np.asarray(points, dtype=float))]


def write_trajectory_event(
    writer: TrajectoryWriter,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    cells: list[Any],
    algotype_by_cell_id: dict[int, str],
    event_index: int,
    event_kind: str,
    swap_count: int,
    archived_compare_and_swap_count: int,
    scheduler_round: int,
    random_null_expected: float,
    actor_cell_id: int | None = None,
    actor_algotype: str | None = None,
) -> float:
    values = cells_values(cells)
    raw = sortedness_raw(values)
    n = len(values)
    position_algotypes = cell_algotypes(cells, algotype_by_cell_id)
    aggregation = aggregation_value_from_algotypes(position_algotypes)
    writer.write(
        {
            "conditionId": condition["conditionId"],
            "mixtureId": condition["mixtureId"],
            "algorithms": condition["algorithms"],
            "inputProfile": condition["inputProfile"],
            "replicateIndex": int(seed_row["replicateIndex"]),
            "replicateNumber": int(seed_row["replicateNumber"]),
            "eventIndex": int(event_index),
            "eventKind": event_kind,
            "swapCount": int(swap_count),
            "comparisonCount": int(archived_compare_and_swap_count + swap_count),
            "archivedCompareAndSwapCount": int(archived_compare_and_swap_count),
            "schedulerRound": int(scheduler_round),
            "sortednessRawCount": int(raw),
            "sortednessPercent": sortedness_percent_from_raw(raw, n),
            "monotonicityError": monotonicity_error_from_raw(raw, n),
            "aggregationValue": aggregation,
            "randomNullExpected": random_null_expected,
            "aggregationMinusNull": aggregation - random_null_expected,
            "actorCellId": "" if actor_cell_id is None else int(actor_cell_id),
            "actorAlgotype": "" if actor_algotype is None else actor_algotype,
            "algotypeSequence": algotype_sequence(position_algotypes),
            "stateHash": state_hash(values),
        }
    )
    return aggregation


def run_duplicate_chimera(
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    writer: TrajectoryWriter,
    max_swaps: int,
    max_comparisons: int,
    max_rounds: int,
    no_move_checks_required: int,
    stable_no_progress_round_cap: int,
) -> DuplicateRunResult:
    start = time.perf_counter()
    n = parse_int(condition["n"])
    if n != 100:
        raise ValueError(f"S11 expects n=100, observed {n}")
    allocation = json.loads(condition["algotypeAllocation"])
    initial_values = duplicate_values_from_seed(parse_int(seed_row["inputPermutationSeed"]))
    initial_algotypes = algotypes_from_seed(seed_row.get("algotypeAssignmentSeed", ""), allocation)
    if len(initial_values) != len(initial_algotypes):
        raise ValueError(f"initial value/Algotype length mismatch for {condition['conditionId']}")
    if not values_are_10_copies_each(initial_values):
        raise ValueError("S11 duplicate initial values do not contain exactly ten copies of 1..10")

    lock = threading.Lock()
    status_probe = StatusProbe()
    swapping_count = [0]
    export_steps: list[list[int]] = []
    cells: list[Any] = [None] * n
    random.seed(parse_int(seed_row["tieBreakerSeed"]))
    for index, (value, algotype) in enumerate(zip(initial_values, initial_algotypes)):
        cell_class = CELL_CLASSES[algotype]
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
            reverse_direction=False,
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
    random_null_expected = random_adjacency_expected_from_counts(dict(Counter(initial_algotypes)))
    scheduler_rng = np.random.default_rng(parse_int(seed_row["schedulerSeed"]))
    per_algotype_moves = {algorithm: 0 for algorithm in ALGORITHMS if int(allocation.get(algorithm, 0)) > 0}
    scheduler_rounds = 0
    no_move_checks = 0
    no_progress_rounds = 0
    stop_reason = "sorted"
    sortedness_points: list[float] = []
    aggregation_points: list[float] = []
    aggregation_minus_null_points: list[float] = []

    initial_raw = sortedness_raw(initial_values)
    initial_percent = sortedness_percent_from_raw(initial_raw, n)
    sortedness_points.append(initial_percent)
    initial_aggregation = write_trajectory_event(
        writer,
        condition,
        seed_row,
        cells,
        algotype_by_cell_id,
        event_index=0,
        event_kind="initial",
        swap_count=0,
        archived_compare_and_swap_count=0,
        scheduler_round=0,
        random_null_expected=random_null_expected,
    )
    aggregation_points.append(initial_aggregation)
    aggregation_minus_null_points.append(initial_aggregation - random_null_expected)

    while not is_sorted_values(cells_values(cells)):
        if status_probe.swap_count >= max_swaps:
            stop_reason = "max_step_cap"
            break
        if status_probe.compare_and_swap_count + status_probe.swap_count >= max_comparisons:
            stop_reason = "max_comparison_cap"
            break
        if scheduler_rounds >= max_rounds:
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
                    values = cells_values(cells)
                    raw = sortedness_raw(values)
                    sortedness_points.append(sortedness_percent_from_raw(raw, n))
                    aggregation = write_trajectory_event(
                        writer,
                        condition,
                        seed_row,
                        cells,
                        algotype_by_cell_id,
                        event_index=event_number,
                        event_kind="swap",
                        swap_count=event_number,
                        archived_compare_and_swap_count=status_probe.compare_and_swap_count,
                        scheduler_round=scheduler_rounds,
                        random_null_expected=random_null_expected,
                        actor_cell_id=actor_cell_id,
                        actor_algotype=actor_algotype,
                    )
                    aggregation_points.append(aggregation)
                    aggregation_minus_null_points.append(aggregation - random_null_expected)
            if status_probe.compare_and_swap_count < compare_before:
                raise RuntimeError("compare counter moved backwards")
            if is_sorted_values(cells_values(cells)):
                break
            if status_probe.swap_count >= max_swaps:
                stop_reason = "max_step_cap"
                break
            if status_probe.compare_and_swap_count + status_probe.swap_count >= max_comparisons:
                stop_reason = "max_comparison_cap"
                break
        scheduler_rounds += 1
        if status_probe.swap_count == swaps_before_round:
            no_progress_rounds += 1
            if any_cell_can_move(cell_objects):
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
            stop_reason = "no_cell_can_move_after_two_checks"
            break
        if stop_reason in {"max_step_cap", "max_comparison_cap", "max_round_cap"}:
            break

    final_values = cells_values(cells)
    final_algotypes = cell_algotypes(cells, algotype_by_cell_id)
    final_raw = sortedness_raw(final_values)
    completed = is_sorted_values(final_values)
    if completed:
        stop_reason = "sorted"

    final_aggregation = aggregation_value_from_algotypes(final_algotypes)
    if aggregation_points:
        peak_index = int(np.argmax(np.asarray(aggregation_points, dtype=float)))
        peak_aggregation = float(aggregation_points[peak_index])
    else:
        peak_index = 0
        peak_aggregation = final_aggregation
    final_swap_count = int(status_probe.swap_count)
    peak_swap_count = min(peak_index, final_swap_count)
    peak_progress = float(peak_swap_count / final_swap_count) if final_swap_count > 0 else 0.0

    return DuplicateRunResult(
        completed=completed,
        stop_reason=stop_reason,
        initial_values=initial_values,
        final_values=final_values,
        initial_algotypes=initial_algotypes,
        final_algotypes=final_algotypes,
        swap_count=final_swap_count,
        comparison_count=int(status_probe.compare_and_swap_count + status_probe.swap_count),
        archived_compare_and_swap_count=int(status_probe.compare_and_swap_count),
        scheduler_rounds=int(scheduler_rounds),
        event_count=int(status_probe.swap_count + 1),
        wall_time_seconds=time.perf_counter() - start,
        per_algotype_moves=per_algotype_moves,
        initial_sortedness_raw_count=int(initial_raw),
        initial_sortedness_percent=float(initial_percent),
        final_sortedness_raw_count=int(final_raw),
        final_sortedness_percent=sortedness_percent_from_raw(final_raw, n),
        final_monotonicity_error=monotonicity_error_from_raw(final_raw, n),
        initial_aggregation=float(initial_aggregation),
        final_aggregation=float(final_aggregation),
        peak_aggregation=peak_aggregation,
        peak_aggregation_swap_count=int(peak_swap_count),
        peak_aggregation_normalized_progress=peak_progress,
        random_null_expected=float(random_null_expected),
        normalized_sortedness=interpolate_points(sortedness_points, final_swap_count),
        normalized_aggregation=interpolate_points(aggregation_points, final_swap_count),
        normalized_aggregation_minus_null=interpolate_points(aggregation_minus_null_points, final_swap_count),
    )


def s11_conditions(condition_matrix_path: Path) -> list[dict[str, Any]]:
    conditions = [row for row in read_csv_rows(condition_matrix_path) if row["producerStep"] == STEP_ID]
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    return sorted(conditions, key=lambda row: order[row["mixtureId"]])


def s11_seed_rows(seed_table_path: Path, max_replicates_per_condition: int | None = None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv_rows(seed_table_path):
        if row["producerStep"] != STEP_ID:
            continue
        if max_replicates_per_condition is not None and int(row["replicateIndex"]) >= max_replicates_per_condition:
            continue
        grouped[row["conditionId"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["replicateIndex"]))
    return dict(grouped)


def make_replicate_record(condition: dict[str, Any], seed_row: dict[str, Any], result: DuplicateRunResult) -> dict[str, Any]:
    allocation = json.loads(condition["algotypeAllocation"])
    observed_initial_counts = dict(Counter(result.initial_algotypes))
    observed_final_counts = dict(Counter(result.final_algotypes))
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": condition["conditionId"],
        "implementation": condition["implementation"],
        "mixtureId": condition["mixtureId"],
        "algorithms": condition["algorithms"],
        "inputProfile": condition["inputProfile"],
        "replicateIndex": int(seed_row["replicateIndex"]),
        "replicateNumber": int(seed_row["replicateNumber"]),
        "inputPermutationSeed": parse_int(seed_row["inputPermutationSeed"]),
        "algotypeAssignmentSeed": parse_int(seed_row["algotypeAssignmentSeed"]),
        "schedulerSeed": parse_int(seed_row["schedulerSeed"]),
        "tieBreakerSeed": parse_int(seed_row["tieBreakerSeed"]),
        "n": parse_int(condition["n"]),
        "targetRepeatCount": parse_int(condition["repeatCount"]),
        "algotypeAllocationJson": json.dumps(allocation, sort_keys=True, separators=(",", ":")),
        "initialAlgotypeCountsJson": json.dumps(observed_initial_counts, sort_keys=True, separators=(",", ":")),
        "finalAlgotypeCountsJson": json.dumps(observed_final_counts, sort_keys=True, separators=(",", ":")),
        "initialValueCountsJson": value_counts_json(result.initial_values),
        "finalValueCountsJson": value_counts_json(result.final_values),
        "initialAlgotypeSequence": algotype_sequence(result.initial_algotypes),
        "finalAlgotypeSequence": algotype_sequence(result.final_algotypes),
        "initialStateHash": state_hash(result.initial_values),
        "finalStateHash": state_hash(result.final_values),
        "initialValuesJson": json.dumps(result.initial_values, separators=(",", ":")),
        "finalValuesJson": json.dumps(result.final_values, separators=(",", ":")),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "initialSortednessRawCount": result.initial_sortedness_raw_count,
        "initialSortednessPercent": result.initial_sortedness_percent,
        "finalSortednessRawCount": result.final_sortedness_raw_count,
        "finalSortednessPercent": result.final_sortedness_percent,
        "finalMonotonicityError": result.final_monotonicity_error,
        "randomNullExpected": result.random_null_expected,
        "initialAggregation": result.initial_aggregation,
        "initialAggregationMinusNull": result.initial_aggregation - result.random_null_expected,
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
        "eventCount": result.event_count,
        "wallTimeSeconds": result.wall_time_seconds,
        "perAlgotypeMovesJson": json.dumps(result.per_algotype_moves, sort_keys=True, separators=(",", ":")),
    }


def make_movement_records(condition: dict[str, Any], seed_row: dict[str, Any], result: DuplicateRunResult) -> list[dict[str, Any]]:
    allocation = json.loads(condition["algotypeAllocation"])
    records = []
    for algotype in ALGORITHMS:
        if int(allocation.get(algotype, 0)) <= 0:
            continue
        moves = int(result.per_algotype_moves.get(algotype, 0))
        records.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "conditionId": condition["conditionId"],
                "mixtureId": condition["mixtureId"],
                "replicateIndex": int(seed_row["replicateIndex"]),
                "replicateNumber": int(seed_row["replicateNumber"]),
                "algotype": algotype,
                "allocatedCellCount": int(allocation.get(algotype, 0)),
                "moveCount": moves,
                "moveShare": float(moves / result.swap_count) if result.swap_count else 0.0,
            }
        )
    return records


def make_curve_records(condition: dict[str, Any], seed_row: dict[str, Any], result: DuplicateRunResult) -> list[dict[str, Any]]:
    rows = []
    for bin_index, progress in enumerate(PROGRESS_BINS):
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "conditionId": condition["conditionId"],
                "mixtureId": condition["mixtureId"],
                "algorithms": condition["algorithms"],
                "inputProfile": condition["inputProfile"],
                "replicateIndex": int(seed_row["replicateIndex"]),
                "replicateNumber": int(seed_row["replicateNumber"]),
                "progressBin": int(bin_index),
                "normalizedProgress": float(progress),
                "finalSwapCount": int(result.swap_count),
                "sortednessPercent": float(result.normalized_sortedness[bin_index]),
                "aggregationValue": float(result.normalized_aggregation[bin_index]),
                "randomNullExpected": float(result.random_null_expected),
                "aggregationMinusNull": float(result.normalized_aggregation_minus_null[bin_index]),
            }
        )
    return rows


def summarize_conditions(replicate_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        replicate_df.groupby(["conditionId", "mixtureId", "algorithms", "algotypeAllocationJson"], dropna=False)
        .agg(
            replicateCount=("replicateIndex", "count"),
            completedCount=("completed", "sum"),
            sortedStopCount=("stopReason", lambda values: int((values == "sorted").sum())),
            meanInitialSortednessPercent=("initialSortednessPercent", "mean"),
            meanFinalSortednessPercent=("finalSortednessPercent", "mean"),
            meanFinalMonotonicityError=("finalMonotonicityError", "mean"),
            meanInitialAggregation=("initialAggregation", "mean"),
            sdInitialAggregation=("initialAggregation", "std"),
            meanFinalAggregation=("finalAggregation", "mean"),
            sdFinalAggregation=("finalAggregation", "std"),
            meanPeakAggregation=("peakAggregation", "mean"),
            sdPeakAggregation=("peakAggregation", "std"),
            meanPeakAggregationProgress=("peakAggregationNormalizedProgress", "mean"),
            meanRandomNullExpected=("randomNullExpected", "mean"),
            meanSwapCount=("swapCount", "mean"),
            sdSwapCount=("swapCount", "std"),
            medianSwapCount=("swapCount", "median"),
            minSwapCount=("swapCount", "min"),
            maxSwapCount=("swapCount", "max"),
            meanComparisonCount=("comparisonCount", "mean"),
            meanArchivedCompareAndSwapCount=("archivedCompareAndSwapCount", "mean"),
            meanSchedulerRounds=("schedulerRounds", "mean"),
            meanWallTimeSeconds=("wallTimeSeconds", "mean"),
            maxWallTimeSeconds=("wallTimeSeconds", "max"),
        )
        .reset_index()
    )
    summary["semFinalAggregation"] = summary["sdFinalAggregation"] / np.sqrt(summary["replicateCount"])
    summary["semPeakAggregation"] = summary["sdPeakAggregation"] / np.sqrt(summary["replicateCount"])
    summary["semSwapCount"] = summary["sdSwapCount"] / np.sqrt(summary["replicateCount"])
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    summary["mixtureOrder"] = summary["mixtureId"].map(order)
    return summary.sort_values("mixtureOrder").reset_index(drop=True)


def summarize_movements(movement_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        movement_df.groupby(["conditionId", "mixtureId", "algotype"], dropna=False)
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
    summary["mixtureOrder"] = summary["mixtureId"].map(order)
    summary["algotypeOrder"] = summary["algotype"].map({algorithm: index for index, algorithm in enumerate(ALGORITHMS)})
    return summary.sort_values(["mixtureOrder", "algotypeOrder"]).reset_index(drop=True)


def summarize_curves(curves_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        curves_df.groupby(
            ["conditionId", "mixtureId", "algorithms", "inputProfile", "progressBin", "normalizedProgress"],
            dropna=False,
        )
        .agg(
            replicateCount=("replicateIndex", "count"),
            sortednessMean=("sortednessPercent", "mean"),
            sortednessSd=("sortednessPercent", "std"),
            aggregationMean=("aggregationValue", "mean"),
            aggregationSd=("aggregationValue", "std"),
            randomNullMean=("randomNullExpected", "mean"),
            aggregationMinusNullMean=("aggregationMinusNull", "mean"),
            aggregationMinusNullSd=("aggregationMinusNull", "std"),
        )
        .reset_index()
    )
    for prefix in ["sortedness", "aggregation", "aggregationMinusNull"]:
        sd_col = f"{prefix}Sd"
        mean_col = f"{prefix}Mean"
        sem_col = f"{prefix}Sem"
        lower_col = f"{prefix}Ci95Lower"
        upper_col = f"{prefix}Ci95Upper"
        summary[sem_col] = summary[sd_col].fillna(0.0) / np.sqrt(summary["replicateCount"])
        summary[lower_col] = summary[mean_col] - 1.96 * summary[sem_col]
        summary[upper_col] = summary[mean_col] + 1.96 * summary[sem_col]
    summary["aggregationCi95Lower"] = summary["aggregationCi95Lower"].clip(0.0, 1.0)
    summary["aggregationCi95Upper"] = summary["aggregationCi95Upper"].clip(0.0, 1.0)
    summary["sortednessCi95Lower"] = summary["sortednessCi95Lower"].clip(0.0, 100.0)
    summary["sortednessCi95Upper"] = summary["sortednessCi95Upper"].clip(0.0, 100.0)
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    summary["mixtureOrder"] = summary["mixtureId"].map(order)
    return summary.sort_values(["mixtureOrder", "progressBin"]).reset_index(drop=True)


def summarize_curve_peaks(curve_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition_id, mixture_id, algorithms), group in curve_summary_df.groupby(
        ["conditionId", "mixtureId", "algorithms"], sort=False, dropna=False
    ):
        group = group.sort_values("progressBin")
        peak = group.loc[group["aggregationMean"].idxmax()]
        final = group.iloc[-1]
        initial = group.iloc[0]
        rows.append(
            {
                "conditionId": condition_id,
                "mixtureId": mixture_id,
                "algorithms": algorithms,
                "meanCurveInitialAggregation": float(initial["aggregationMean"]),
                "meanCurveFinalAggregation": float(final["aggregationMean"]),
                "meanCurvePeakAggregation": float(peak["aggregationMean"]),
                "meanCurvePeakProgress": float(peak["normalizedProgress"]),
                "meanCurvePeakProgressBin": int(peak["progressBin"]),
                "randomNullExpected": float(peak["randomNullMean"]),
            }
        )
    result = pd.DataFrame(rows)
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    result["mixtureOrder"] = result["mixtureId"].map(order)
    return result.sort_values("mixtureOrder").reset_index(drop=True)


def bootstrap_mean_ci(values: np.ndarray, seed: int, resamples: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"S11:{label}:2646294683".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def build_unique_comparison(
    duplicate_summary: pd.DataFrame,
    duplicate_curve_peaks: pd.DataFrame,
    unique_peak_summary_path: Path,
) -> pd.DataFrame:
    unique = pd.read_parquet(unique_peak_summary_path)
    unique = unique[unique["mixtureId"].isin(MIXTURE_ORDER)].copy()
    rows = []
    for _, dup in duplicate_summary.iterrows():
        mixture = dup["mixtureId"]
        unique_row = unique[unique["mixtureId"] == mixture].iloc[0]
        curve_peak = duplicate_curve_peaks[duplicate_curve_peaks["mixtureId"] == mixture].iloc[0]
        report = PAPER_DUPLICATE_REPORTS[mixture]
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "mixtureId": mixture,
                "algorithms": dup["algorithms"],
                "replicateCount": int(dup["replicateCount"]),
                "duplicateFinalAggregationMean": float(dup["meanFinalAggregation"]),
                "duplicateFinalAggregationSd": float(dup["sdFinalAggregation"]),
                "duplicatePeakAggregationMean": float(dup["meanPeakAggregation"]),
                "duplicatePeakAggregationSd": float(dup["sdPeakAggregation"]),
                "duplicateMeanCurvePeakAggregation": float(curve_peak["meanCurvePeakAggregation"]),
                "duplicateMeanCurvePeakProgress": float(curve_peak["meanCurvePeakProgress"]),
                "duplicateMeanFinalSortednessPercent": float(dup["meanFinalSortednessPercent"]),
                "uniqueFinalAggregationMean": float(unique_row["finalAggregationMean"]),
                "uniquePeakAggregationMean": float(unique_row["peakAggregationMean"]),
                "uniqueMeanCurvePeakAggregation": float(unique_row["meanCurvePeakAggregation"]),
                "uniqueMeanCurvePeakProgress": float(unique_row["meanCurvePeakProgress"]),
                "duplicateMinusUniqueFinalAggregation": float(dup["meanFinalAggregation"] - unique_row["finalAggregationMean"]),
                "duplicateMinusUniquePeakAggregation": float(dup["meanPeakAggregation"] - unique_row["peakAggregationMean"]),
                "duplicateMeanCurvePeakMinusUnique": float(curve_peak["meanCurvePeakAggregation"] - unique_row["meanCurvePeakAggregation"]),
                "paperDuplicatePeakAggregation": float(report["paperDuplicatePeakAggregation"]),
                "paperDuplicatePeakProgress": float(report["paperDuplicatePeakProgress"]),
                "paperDuplicateFinalAggregation": float(report["paperDuplicateFinalAggregation"]),
            }
        )
    comparison = pd.DataFrame(rows)
    return comparison


def add_bootstrap_to_comparison(comparison: pd.DataFrame, replicate_df: pd.DataFrame) -> pd.DataFrame:
    comparison = comparison.copy()
    for mixture in comparison["mixtureId"]:
        group = replicate_df[replicate_df["mixtureId"] == mixture]
        final_ci = bootstrap_mean_ci(group["finalAggregation"].to_numpy(dtype=float), stable_seed(f"{mixture}:final"))
        peak_ci = bootstrap_mean_ci(group["peakAggregation"].to_numpy(dtype=float), stable_seed(f"{mixture}:peak"))
        comparison.loc[comparison["mixtureId"] == mixture, "duplicateFinalAggregationBootstrapCi95Lower"] = final_ci[0]
        comparison.loc[comparison["mixtureId"] == mixture, "duplicateFinalAggregationBootstrapCi95Upper"] = final_ci[1]
        comparison.loc[comparison["mixtureId"] == mixture, "duplicatePeakAggregationBootstrapCi95Lower"] = peak_ci[0]
        comparison.loc[comparison["mixtureId"] == mixture, "duplicatePeakAggregationBootstrapCi95Upper"] = peak_ci[1]
    return comparison


def write_result_tables(
    replicate_df: pd.DataFrame,
    movement_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    results_dir: Path,
    unique_peak_summary_path: Path,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "replicateParquet": results_dir / "e01_duplicate_value_chimeras.parquet",
        "replicateCsv": results_dir / "e01_duplicate_value_chimeras.csv",
        "summaryParquet": results_dir / "e01_duplicate_value_chimera_summary.parquet",
        "summaryCsv": results_dir / "e01_duplicate_value_chimera_summary.csv",
        "movementParquet": results_dir / "e01_duplicate_value_chimera_algotype_movements.parquet",
        "movementCsv": results_dir / "e01_duplicate_value_chimera_algotype_movements.csv",
        "movementSummaryParquet": results_dir / "e01_duplicate_value_chimera_algotype_movement_summary.parquet",
        "movementSummaryCsv": results_dir / "e01_duplicate_value_chimera_algotype_movement_summary.csv",
        "curveParquet": results_dir / "e01_duplicate_value_aggregation_curves.parquet",
        "curveCsv": results_dir / "e01_duplicate_value_aggregation_curves.csv",
        "curveSummaryParquet": results_dir / "e01_duplicate_value_aggregation_curve_summary.parquet",
        "curveSummaryCsv": results_dir / "e01_duplicate_value_aggregation_curve_summary.csv",
        "curvePeakParquet": results_dir / "e01_duplicate_value_aggregation_curve_peaks.parquet",
        "curvePeakCsv": results_dir / "e01_duplicate_value_aggregation_curve_peaks.csv",
        "comparisonParquet": results_dir / "e01_duplicate_value_unique_comparison.parquet",
        "comparisonCsv": results_dir / "e01_duplicate_value_unique_comparison.csv",
    }
    summary_df = summarize_conditions(replicate_df)
    movement_summary_df = summarize_movements(movement_df)
    curve_summary_df = summarize_curves(curve_df)
    curve_peak_df = summarize_curve_peaks(curve_summary_df)
    comparison_df = build_unique_comparison(summary_df, curve_peak_df, unique_peak_summary_path)
    comparison_df = add_bootstrap_to_comparison(comparison_df, replicate_df)

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
    curve_peak_df.to_parquet(paths["curvePeakParquet"], index=False)
    curve_peak_df.to_csv(paths["curvePeakCsv"], index=False)
    comparison_df.to_parquet(paths["comparisonParquet"], index=False)
    comparison_df.to_csv(paths["comparisonCsv"], index=False)
    return paths


def plot_duplicate_chimeras(curve_summary_path: Path, comparison_path: Path, figure_dir: Path) -> tuple[Path, Path]:
    curve_summary = pd.read_parquet(curve_summary_path)
    comparison = pd.read_parquet(comparison_path)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), sharex=True, sharey=True)
    for ax, mixture in zip(axes, MIXTURE_ORDER, strict=True):
        subset = curve_summary[curve_summary["mixtureId"] == mixture].sort_values("progressBin")
        comp = comparison[comparison["mixtureId"] == mixture].iloc[0]
        x = subset["normalizedProgress"].to_numpy(dtype=float) * 100.0
        agg = subset["aggregationMean"].to_numpy(dtype=float)
        agg_lo = subset["aggregationCi95Lower"].to_numpy(dtype=float)
        agg_hi = subset["aggregationCi95Upper"].to_numpy(dtype=float)
        sortedness = subset["sortednessMean"].to_numpy(dtype=float)
        ax.plot(x, agg, color="#b3202a", linewidth=2.2, label="Duplicate Aggregation")
        ax.fill_between(x, agg_lo, agg_hi, color="#b3202a", alpha=0.15, linewidth=0)
        ax.axhline(float(comp["uniqueMeanCurvePeakAggregation"]), color="#666666", linestyle="--", linewidth=1.2, label="Unique peak")
        ax.axhline(float(comp["uniqueFinalAggregationMean"]), color="#999999", linestyle=":", linewidth=1.2, label="Unique final")
        ax.scatter(
            [float(comp["duplicateMeanCurvePeakProgress"]) * 100.0],
            [float(comp["duplicateMeanCurvePeakAggregation"])],
            color="#b3202a",
            s=36,
            zorder=5,
        )
        ax.set_title(DISPLAY_NAMES[mixture], fontsize=11)
        ax.set_xlabel("Normalized sorting progress (%)")
        ax.grid(True, color="#dddddd", linewidth=0.7)
        ax.set_xlim(0, 100)
        ax.set_ylim(0.42, 0.9)
        twin = ax.twinx()
        twin.plot(x, sortedness, color="#1f77b4", alpha=0.38, linewidth=1.2, label="Sortedness")
        twin.set_ylim(40, 100)
        twin.tick_params(axis="y", labelcolor="#1f77b4", labelsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Aggregation")
        if ax is axes[-1]:
            twin.set_ylabel("Sortedness (%)", color="#1f77b4")
        ax.text(
            0.04,
            0.95,
            f"final {comp['duplicateFinalAggregationMean']:.2f}\npeak {comp['duplicateMeanCurvePeakAggregation']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("E01 S11 duplicate-value same-goal chimeras", y=1.08, fontsize=14)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "figure08_duplicate_chimeras.png"
    pdf = figure_dir / "figure08_duplicate_chimeras.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_duplicate_vs_unique(comparison_path: Path, figure_dir: Path) -> tuple[Path, Path]:
    comparison = pd.read_parquet(comparison_path)
    labels = [DISPLAY_NAMES[item] for item in comparison["mixtureId"]]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - width, comparison["uniqueFinalAggregationMean"], width=width, label="Unique final", color="#8c8c8c")
    ax.bar(x, comparison["duplicateFinalAggregationMean"], width=width, label="Duplicate final", color="#b3202a")
    ax.bar(x + width, comparison["duplicatePeakAggregationMean"], width=width, label="Duplicate replicate peak", color="#d96c75")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Aggregation")
    ax.set_ylim(0.45, 0.9)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title("Duplicate-value Aggregation compared with unique-value S10")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "figure08_duplicate_vs_unique_aggregation.png"
    pdf = figure_dir / "figure08_duplicate_vs_unique_aggregation.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def validate_outputs(
    replicate_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    movement_df: pd.DataFrame,
    movement_summary_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    curve_summary_df: pd.DataFrame,
    curve_peak_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    trace_row_count: int,
    output_paths: list[Path],
    expected_replicates_per_condition: int,
    artifacts_dir: Path,
) -> tuple[bool, list[str], list[str], str]:
    checks: list[str] = []
    failures: list[str] = []
    caveats: list[str] = []

    observed_mixtures = set(replicate_df["mixtureId"])
    if observed_mixtures == set(MIXTURE_ORDER):
        checks.append("All three S11 pairwise duplicate-value mixtures are present.")
    else:
        failures.append(f"Observed mixtures {sorted(observed_mixtures)} do not match expected {MIXTURE_ORDER}.")

    expected_rows = expected_replicates_per_condition * len(MIXTURE_ORDER)
    if len(replicate_df) == expected_rows:
        checks.append(f"Replicate table has {expected_rows} rows.")
    else:
        failures.append(f"Replicate table has {len(replicate_df)} rows; expected {expected_rows}.")

    per_condition = replicate_df.groupby("conditionId").size()
    if per_condition.nunique() == 1 and int(per_condition.iloc[0]) == expected_replicates_per_condition:
        checks.append(f"Every S11 condition has {expected_replicates_per_condition} replicate rows.")
    else:
        failures.append(f"Per-condition replicate counts are inconsistent: {per_condition.to_dict()}.")

    expected_count_json = json.dumps({str(value): 10 for value in range(1, 11)}, sort_keys=True, separators=(",", ":"))
    if (replicate_df["initialValueCountsJson"] == expected_count_json).all() and (replicate_df["finalValueCountsJson"] == expected_count_json).all():
        checks.append("Every initial and final array has exactly ten copies each of values 1 through 10.")
    else:
        failures.append("At least one S11 run did not preserve exactly ten copies each of values 1 through 10.")

    if bool(replicate_df["completed"].all()) and (replicate_df["stopReason"] == "sorted").all():
        checks.append("Every duplicate-value chimera run reached a sorted stop state.")
    else:
        failures.append("At least one S11 run did not complete with stopReason=sorted.")

    if (replicate_df["finalSortednessPercent"] == 100.0).all() and (replicate_df["finalMonotonicityError"] == 0).all():
        checks.append("Every S11 run reached final 100% Sortedness and zero monotonicity error.")
    else:
        failures.append("At least one S11 run did not end at 100% Sortedness and zero monotonicity error.")

    input_hash_counts = replicate_df.groupby("replicateIndex")["initialStateHash"].nunique()
    if int(input_hash_counts.max()) == 1:
        checks.append("Duplicate-value input permutations are matched across the three S11 mixtures by replicate index.")
    else:
        failures.append("Duplicate-value input permutations are not matched across all S11 mixtures.")

    allocation_failures = []
    for _, row in replicate_df.iterrows():
        expected = json.loads(row["algotypeAllocationJson"])
        observed = json.loads(row["initialAlgotypeCountsJson"])
        if {key: int(value) for key, value in expected.items()} != {key: int(value) for key, value in observed.items()}:
            allocation_failures.append((row["conditionId"], int(row["replicateIndex"]), expected, observed))
            break
    if not allocation_failures:
        checks.append("Observed initial Algotype counts match S03 S11 allocations for every run.")
    else:
        failures.append(f"Algotype allocation mismatch example: {allocation_failures[0]}.")

    move_sums = []
    for _, row in replicate_df.iterrows():
        moves = json.loads(row["perAlgotypeMovesJson"])
        move_sums.append(sum(int(value) for value in moves.values()) == int(row["swapCount"]))
    if all(move_sums):
        checks.append("Per-Algotype movement counts sum to each run's total swap count.")
    else:
        failures.append("At least one run has per-Algotype movement counts that do not sum to total swaps.")

    if trace_row_count == int(replicate_df["eventCount"].sum()):
        checks.append("S11 trajectory row count equals the summed replicate event counts.")
    else:
        failures.append(f"S11 trajectory row count {trace_row_count} does not equal summed event count {int(replicate_df['eventCount'].sum())}.")

    if len(summary_df) == len(MIXTURE_ORDER):
        checks.append("Condition summary has three rows.")
    else:
        failures.append(f"Condition summary has {len(summary_df)} rows, expected three.")

    if len(movement_df) == expected_rows * 2:
        checks.append("Per-replicate Algotype movement table has the expected pairwise long-format rows.")
    else:
        failures.append(f"Movement table has {len(movement_df)} rows, expected {expected_rows * 2}.")

    if len(movement_summary_df) == len(MIXTURE_ORDER) * 2:
        checks.append("Algotype movement summary has expected condition-by-Algotype rows.")
    else:
        failures.append("Algotype movement summary row count is unexpected.")

    if len(curve_df) == expected_rows * 101:
        checks.append("Duplicate Aggregation curve table has 101 bins for every replicate.")
    else:
        failures.append(f"Curve table has {len(curve_df)} rows, expected {expected_rows * 101}.")

    if len(curve_summary_df) == len(MIXTURE_ORDER) * 101:
        checks.append("Duplicate Aggregation curve summary has 101 bins for each mixture.")
    else:
        failures.append(f"Curve summary has {len(curve_summary_df)} rows, expected {len(MIXTURE_ORDER) * 101}.")

    if len(curve_peak_df) == len(MIXTURE_ORDER):
        checks.append("Mean-curve peak table has one row for each S11 mixture.")
    else:
        failures.append("Mean-curve peak table row count is unexpected.")

    if len(comparison_df) == len(MIXTURE_ORDER):
        checks.append("Unique-vs-duplicate comparison table has one row for each S11 mixture.")
    else:
        failures.append("Unique-vs-duplicate comparison table row count is unexpected.")

    selection_pair_comparison = comparison_df[comparison_df["mixtureId"].isin(["bubble_selection", "insertion_selection"])]
    if bool((comparison_df["duplicateMinusUniqueFinalAggregation"] > 0).all()):
        checks.append("Every duplicate-value mixture has higher final Aggregation than its unique-value S10 counterpart.")
    else:
        caveats.append("At least one duplicate-value mixture did not have higher final Aggregation than its unique-value S10 counterpart.")

    if bool((selection_pair_comparison["duplicateFinalAggregationMean"] > selection_pair_comparison["uniqueFinalAggregationMean"]).all()):
        checks.append("Selection-containing duplicate-value mixtures have higher final Aggregation than their unique-value S10 counterparts.")
    else:
        failures.append("A Selection-containing duplicate-value mixture did not exceed its unique-value final Aggregation counterpart.")

    bubble_insertion = comparison_df[comparison_df["mixtureId"] == "bubble_insertion"]
    if not bubble_insertion.empty and bool(
        (bubble_insertion["duplicateMeanCurvePeakAggregation"] >= bubble_insertion["uniqueMeanCurvePeakAggregation"]).all()
    ):
        checks.append("Bubble-Insertion duplicate-value mean-curve peak Aggregation is at least as high as the unique-value S10 peak.")
    else:
        caveats.append("Bubble-Insertion duplicate-value mean-curve peak Aggregation did not exceed the unique-value S10 peak.")

    if bool((comparison_df["duplicateMeanFinalSortednessPercent"] == 100.0).all()):
        checks.append("Comparison table confirms final 100% Sortedness for every duplicate-value mixture.")
    else:
        failures.append("Comparison table does not confirm final 100% Sortedness for every duplicate-value mixture.")

    missing_outputs = [str(path) for path in output_paths if not path.exists() or path.stat().st_size == 0]
    if not missing_outputs:
        checks.append("All declared S11 output files exist and are non-empty.")
    else:
        failures.append(f"Missing or empty S11 output files: {missing_outputs}.")

    if (artifacts_dir / "research_steps" / "S12").exists():
        failures.append("S12 artifact directory exists; S11 must stop before S12.")
    else:
        checks.append("No S12 artifact directory was created.")

    if (artifacts_dir / "results" / "e01_opposite_direction_chimeras.parquet").exists():
        failures.append("S12 opposite-direction result exists; S11 must not compute S12 outputs.")
    else:
        checks.append("No S12 opposite-direction chimera result was created.")

    success = not failures
    if success:
        validation_result = (
            "passed: S11 ran three duplicate-value same-goal pairwise chimera conditions, "
            f"{expected_rows} replicate rows, validated 10 copies each of values 1..10, matched duplicate inputs, "
            "final 100% Sortedness, Aggregation peak/final tables, unique-value comparisons, declared artifacts, and no-S12-start guard."
        )
    else:
        validation_result = "failed: " + " ".join(failures)
    return success, checks, failures + caveats, validation_result


def infer_outcome(replicate_df: pd.DataFrame, comparison_df: pd.DataFrame) -> tuple[str, str]:
    all_sorted = bool(replicate_df["completed"].all()) and (replicate_df["finalSortednessPercent"] == 100.0).all()
    selection_comparison = comparison_df[comparison_df["mixtureId"].isin(["bubble_selection", "insertion_selection"])]
    selection_final_above_unique = bool(
        (selection_comparison["duplicateFinalAggregationMean"] > selection_comparison["uniqueFinalAggregationMean"]).all()
    )
    persistent_selection_pairs = bool(
        (
            selection_comparison["duplicateFinalAggregationMean"]
            >= selection_comparison["uniquePeakAggregationMean"]
        ).all()
    )
    bubble_insertion = comparison_df[comparison_df["mixtureId"] == "bubble_insertion"]
    bubble_insertion_peak_at_least_unique = bool(
        not bubble_insertion.empty
        and (bubble_insertion["duplicateMeanCurvePeakAggregation"] >= bubble_insertion["uniqueMeanCurvePeakAggregation"]).all()
    )
    if all_sorted and selection_final_above_unique and persistent_selection_pairs and bubble_insertion_peak_at_least_unique:
        return (
            "supportive",
            "S11 supports the duplicate-value Aggregation claim: all duplicate-value mixed arrays sorted, the Selection-containing pairs retained final Aggregation at or above their unique-value peak levels, and Bubble-Insertion reached a duplicate-value peak at least as high as its unique-value peak.",
        )
    if all_sorted and selection_final_above_unique and bubble_insertion_peak_at_least_unique:
        return (
            "supportive",
            "S11 supports the core duplicate-value relaxation claim because all duplicate-value mixed arrays sorted, Selection-containing pairs ended with higher Aggregation than their unique-value counterparts, and Bubble-Insertion reached a duplicate-value peak at least as high as its unique-value peak; persistence relative to unique-value peak levels is weaker for at least one Selection-containing pair.",
        )
    if all_sorted:
        return (
            "constraining",
            "S11 supports duplicate-value sorting completion but constrains the Aggregation-persistence claim because the expected duplicate-value Aggregation pattern did not hold across the pairwise mixtures.",
        )
    return (
        "constraining",
        "S11 constrains duplicate-value same-goal chimera completion because at least one run did not reach final 100% Sortedness.",
    )


def write_methods(path: Path) -> None:
    lines = [
        "# S11 Methods",
        "",
        "- Research step ID: S11",
        "- Step number: 11",
        "- Completion status: completed",
        "- Artifacts written: see `artifact_manifest.json` and `status.json`.",
        "- Validation result: see `validation.json` and `validation.md`.",
        "- Caveats or blockers: S11 inherits S09 deterministic pseudo-scheduler semantics and does not start S12.",
        "- Recommended next action: hand control back to Chief Scientist review before S12.",
        "",
        "## Simulation Scope",
        "",
        "S11 runs the three S03-defined duplicate-value same-goal mixed cell-view conditions: Bubble-Insertion, Bubble-Selection, and Insertion-Selection. Each replicate has 100 cells with values 1 through 10 repeated exactly ten times each, shuffled by the S03 input permutation seed. Algotype allocations are 50/50 for each pair and are shuffled by the S03 Algotype assignment seed.",
        "",
        "## Tie Handling",
        "",
        "The archived cell classes use strict comparisons for equal values. Bubble swaps only when a neighboring value is strictly out of order. Insertion moves left only when its value is strictly smaller than the left neighbor. Selection treats an equal value at the current ideal position as not smaller, advances the ideal position, and does not swap with the equal value. Therefore equal-valued cells can remain in any Algotype order while still contributing to final nondecreasing Sortedness.",
        "",
        "## Metrics",
        "",
        "Sortedness is the percent of adjacent value pairs satisfying nondecreasing order. Aggregation is the S10 deterministic left-neighbor same-Algotype adjacent-pair fraction. S11 records initial, final, and peak Aggregation per replicate, interpolates Aggregation and Sortedness onto 101 normalized swap-progress bins, and compares duplicate-value final/peak Aggregation with S10 unique-value pairwise outputs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_markdown(path: Path, checks: list[str], caveats: list[str], validation_result: str) -> None:
    lines = [
        "# S11 Validation",
        "",
        "- Research step ID: S11",
        "- Step number: 11",
        "- Completion status: completed",
        "- Validation result: " + validation_result,
        "- Artifacts written: see `artifact_manifest.json` and `status.json`.",
        "- Caveats or blockers:",
    ]
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Recommended next action: hand control back to Chief Scientist review before S12.",
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
    comparison_df: pd.DataFrame,
    movement_summary_df: pd.DataFrame,
) -> None:
    summary_rows = [
        [
            row["mixtureId"],
            int(row["replicateCount"]),
            int(row["completedCount"]),
            row["meanFinalSortednessPercent"],
            row["meanFinalAggregation"],
            row["meanPeakAggregation"],
            row["meanPeakAggregationProgress"],
            row["meanSwapCount"],
        ]
        for _, row in summary_df.iterrows()
    ]
    comparison_rows = [
        [
            row["mixtureId"],
            row["duplicateFinalAggregationMean"],
            row["uniqueFinalAggregationMean"],
            row["duplicateMinusUniqueFinalAggregation"],
            row["duplicateMeanCurvePeakAggregation"],
            row["uniqueMeanCurvePeakAggregation"],
            row["paperDuplicatePeakAggregation"],
        ]
        for _, row in comparison_df.iterrows()
    ]
    movement_rows = [
        [
            row["mixtureId"],
            row["algotype"],
            row["allocatedCellCount"],
            row["meanMoveCount"],
            row["meanMoveShare"],
        ]
        for _, row in movement_summary_df.iterrows()
    ]
    lines = [
        "# S11 Status Summary",
        "",
        "- Research step ID: S11",
        "- Step number: 11",
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
            "- Lay summary: S11 repeated the same-goal mixed Algotype runs with duplicate values, allowing equal-valued cells to remain in any order. All three pairwise mixtures sorted successfully, and duplicate-value final Aggregation was compared directly with the S10 unique-value Aggregation results.",
            "- Recommended next action: " + recommended_next_action,
            "",
            "## Duplicate-Value Condition Means",
            "",
            markdown_table(
                [
                    "Mixture",
                    "Replicates",
                    "Completed",
                    "Mean final Sortedness",
                    "Mean final Aggregation",
                    "Mean peak Aggregation",
                    "Mean peak progress",
                    "Mean swaps",
                ],
                summary_rows,
            ),
            "",
            "## Duplicate Versus Unique Aggregation",
            "",
            markdown_table(
                [
                    "Mixture",
                    "Duplicate final",
                    "Unique final",
                    "Final difference",
                    "Duplicate mean-curve peak",
                    "Unique mean-curve peak",
                    "Paper duplicate peak",
                ],
                comparison_rows,
            ),
            "",
            "## Per-Algotype Movement Means",
            "",
            markdown_table(
                ["Mixture", "Algotype", "Allocated cells", "Mean moves", "Mean move share"],
                movement_rows,
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def update_run_manifest(path: Path, status_payload: dict[str, Any], artifact_manifest_payload: dict[str, Any]) -> None:
    manifest = load_json(path)
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest.setdefault("researchSteps", {})
    manifest["updatedAt"] = status_payload["generatedAt"]
    manifest["latestResearchStepId"] = STEP_ID
    manifest["researchSteps"][STEP_ID] = {
        "status": status_payload["status"],
        "success": status_payload["success"],
        "outcomeClassification": status_payload["outcomeClassification"],
        "artifactCount": artifact_manifest_payload["artifactCount"],
        "artifacts": artifact_manifest_payload["artifacts"],
        "validationResult": status_payload["validationResult"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "generatedAt": status_payload["generatedAt"],
        "git": status_payload["git"],
    }
    write_json(path, manifest)


def run_s11(args: argparse.Namespace) -> int:
    generated_at = utc_now()
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

    config = load_json(args.config)
    stop_policy = config.get("semantics", {}).get("stopPolicies", {}).get("sorted_no_move_or_cap", {})
    max_swaps = int(stop_policy.get("maxSwapEvents", 500000))
    max_comparisons = int(stop_policy.get("maxComparisonEvents", 4000000))
    no_move_checks_required = int(stop_policy.get("noMoveChecksRequired", 2))
    max_rounds = int(args.max_rounds)
    stable_no_progress_round_cap = int(args.stable_no_progress_round_cap)

    conditions = s11_conditions(args.condition_matrix)
    seed_rows_by_condition = s11_seed_rows(args.seed_table, args.max_replicates_per_condition)
    replicate_records: list[dict[str, Any]] = []
    movement_records: list[dict[str, Any]] = []
    curve_records: list[dict[str, Any]] = []
    trace_path = trace_dir / "e01_s11_duplicate_value_chimera_trajectory_events.csv.gz"

    with TrajectoryWriter(trace_path) as trajectory_writer:
        for condition in conditions:
            rows = seed_rows_by_condition.get(condition["conditionId"], [])
            if not rows:
                raise RuntimeError(f"No S11 seed rows found for {condition['conditionId']}")
            for seed_row in rows:
                result = run_duplicate_chimera(
                    condition,
                    seed_row,
                    trajectory_writer,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                    max_rounds=max_rounds,
                    no_move_checks_required=no_move_checks_required,
                    stable_no_progress_round_cap=stable_no_progress_round_cap,
                )
                replicate_records.append(make_replicate_record(condition, seed_row, result))
                movement_records.extend(make_movement_records(condition, seed_row, result))
                curve_records.extend(make_curve_records(condition, seed_row, result))
        trace_row_count = trajectory_writer.row_count

    replicate_df = pd.DataFrame(replicate_records)
    movement_df = pd.DataFrame(movement_records)
    curve_df = pd.DataFrame(curve_records)
    result_paths = write_result_tables(replicate_df, movement_df, curve_df, results_dir, args.unique_peak_summary)
    summary_df = pd.read_parquet(result_paths["summaryParquet"])
    movement_summary_df = pd.read_parquet(result_paths["movementSummaryParquet"])
    curve_summary_df = pd.read_parquet(result_paths["curveSummaryParquet"])
    curve_peak_df = pd.read_parquet(result_paths["curvePeakParquet"])
    comparison_df = pd.read_parquet(result_paths["comparisonParquet"])
    duplicate_png, duplicate_pdf = plot_duplicate_chimeras(result_paths["curveSummaryParquet"], result_paths["comparisonParquet"], figure_dir)
    comparison_png, comparison_pdf = plot_duplicate_vs_unique(result_paths["comparisonParquet"], figure_dir)
    outcome_classification, outcome_reason = infer_outcome(replicate_df, comparison_df)

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
        duplicate_png,
        duplicate_pdf,
        comparison_png,
        comparison_pdf,
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
        curve_peak_df,
        comparison_df,
        trace_row_count,
        output_paths,
        expected_replicates_per_condition=expected_replicates,
        artifacts_dir=artifacts_dir,
    )
    caveats_or_blockers.extend(
        [
            "S11 uses the deterministic pseudo-scheduler wrapper around archived cell-view classes rather than live OS-thread scheduling.",
            "Tie handling follows archived strict-comparison behavior: Bubble and Insertion do not swap equal values; Selection advances past equal values rather than swapping them.",
            "S11 compares duplicate-value results to the available S10 unique-value Aggregation outputs, which used fixed-count random-label nulls rather than the paper's two-label same-code controls.",
            outcome_reason,
            "No S12 opposite-direction analysis or S12 artifact directory was started.",
        ]
    )

    code_copy = code_dir / Path(__file__).name
    shutil.copy2(Path(__file__), code_copy)
    recommended_next_action = "Hand control back to the Chief Scientist workflow for review; start S12 opposite-direction chimeras only after explicit instruction."
    source_statuses = {
        "S03": load_json(S03_STATUS_DEFAULT),
        "S09": load_json(S09_STATUS_DEFAULT),
        "S10": load_json(S10_STATUS_DEFAULT),
    }
    runtime = {
        "pythonVersion": sys.version,
        "pythonExecutable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "osCpuCount": os.cpu_count(),
        "workerCount": 1,
        "gpuUsed": False,
        "numpyVersion": np.__version__,
        "pandasVersion": pd.__version__,
        "matplotlibVersion": matplotlib.__version__,
        "maxSwaps": max_swaps,
        "maxComparisons": max_comparisons,
        "maxRounds": max_rounds,
        "noMoveChecksRequired": no_move_checks_required,
        "stableNoProgressRoundCap": stable_no_progress_round_cap,
    }
    artifacts_written_paths = [
        *output_paths,
        validation_json,
        validation_md,
        summary_md,
        status_json,
        artifact_manifest_json,
        code_copy,
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
        "curvePeakRows": int(len(curve_peak_df)),
        "comparisonRows": int(len(comparison_df)),
        "traceRows": int(trace_row_count),
        "outcomeClassification": outcome_classification,
    }
    write_validation_markdown(validation_md, validation_checks, caveats_or_blockers, validation_result)
    write_json(validation_json, validation_payload)

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
        "git": get_git_metadata(),
        "runtime": runtime,
        "sourceArtifacts": {
            "s03BaselineConfig": str(args.config),
            "s03ConditionMatrix": str(args.condition_matrix),
            "s03SeedTable": str(args.seed_table),
            "s09Status": str(S09_STATUS_DEFAULT),
            "s10Status": str(S10_STATUS_DEFAULT),
            "s10UniquePeakSummary": str(args.unique_peak_summary),
            "s10UniqueCurveSummary": str(args.unique_curve_summary),
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
            "curvePeakRows": int(len(curve_peak_df)),
            "comparisonRows": int(len(comparison_df)),
            "traceRows": int(trace_row_count),
        },
        "mixtures": MIXTURE_ORDER,
        "tieHandling": {
            "bubble": "strict adjacent inequality; equal values do not swap",
            "insertion": "strict left-neighbor inequality; equal values do not swap",
            "selection": "equal values at ideal position are skipped by advancing ideal position; equal values do not swap",
        },
    }
    write_json(status_json, status_payload)
    write_summary_markdown(
        summary_md,
        artifacts_written,
        validation_result,
        caveats_or_blockers,
        recommended_next_action,
        outcome_classification,
        summary_df,
        comparison_df,
        movement_summary_df,
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
    write_json(artifact_manifest_json, artifact_manifest_payload)
    update_run_manifest(run_manifest_path, status_payload, artifact_manifest_payload)
    artifacts = collect_artifacts(artifacts_written_paths, manifest_path=artifact_manifest_json)
    artifact_manifest_payload["artifactCount"] = len(artifacts)
    artifact_manifest_payload["artifacts"] = artifacts
    write_json(artifact_manifest_json, artifact_manifest_payload)
    update_run_manifest(run_manifest_path, status_payload, artifact_manifest_payload)

    print(json.dumps({"success": success, "validationResult": validation_result, "artifactsWritten": artifacts_written}, indent=2))
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH_DEFAULT)
    parser.add_argument("--condition-matrix", type=Path, default=CONDITION_MATRIX_DEFAULT)
    parser.add_argument("--seed-table", type=Path, default=SEED_TABLE_DEFAULT)
    parser.add_argument("--unique-peak-summary", type=Path, default=S10_PEAK_SUMMARY_DEFAULT)
    parser.add_argument("--unique-curve-summary", type=Path, default=S10_CURVE_SUMMARY_DEFAULT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--max-rounds", type=int, default=100000)
    parser.add_argument("--stable-no-progress-round-cap", type=int, default=100)
    parser.add_argument("--max-replicates-per-condition", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    return run_s11(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
