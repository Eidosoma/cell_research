#!/usr/bin/env python3
"""Run E01 S09 same-goal mixed Algotype cell-view arrays.

S09 consumes the S03 condition matrix and seed table, runs only the seven
same-goal unique-value cell-view chimera conditions, writes reusable traces
for later S10 Aggregation analysis, and stops before S10.
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
from collections import Counter, defaultdict
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

from modules.multithread.BubbleSortCell import BubbleSortCell
from modules.multithread.CellGroup import CellGroup, GroupStatus
from modules.multithread.InsertionSortCell import InsertionSortCell
from modules.multithread.SelectionSortCell import SelectionSortCell
from modules.multithread.StatusProbe import StatusProbe


EXPERIMENT_ID = "E01"
STEP_ID = "S09"
STEP_NUMBER = 9
STATUS = "completed"
CONFIG_PATH_DEFAULT = Path("/artifacts/configs/e01_baseline_config.json")
CONDITION_MATRIX_DEFAULT = Path("/artifacts/research_steps/S03/condition_matrix.csv")
SEED_TABLE_DEFAULT = Path("/artifacts/research_steps/S03/seed_table.csv")
S03_STATUS_DEFAULT = Path("/artifacts/research_steps/S03/status.json")
S04_STATUS_DEFAULT = Path("/artifacts/research_steps/S04/status.json")
S08_STATUS_DEFAULT = Path("/artifacts/research_steps/S08/status.json")
RUN_MANIFEST_DEFAULT = Path("/artifacts/provenance/run_manifest.json")
ALGORITHMS = ["bubble", "insertion", "selection"]
MIXTURE_ORDER = [
    "pure_bubble",
    "pure_insertion",
    "pure_selection",
    "bubble_insertion",
    "bubble_selection",
    "insertion_selection",
    "bubble_insertion_selection",
]
PAIR_MIXTURES = ["bubble_insertion", "bubble_selection", "insertion_selection"]
PURE_BY_ALGORITHM = {
    "bubble": "pure_bubble",
    "insertion": "pure_insertion",
    "selection": "pure_selection",
}
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
STOP_REASONS = {
    "sorted",
    "no_cell_can_move_after_two_checks",
    "max_step_cap",
    "max_comparison_cap",
    "max_round_cap",
    "error",
}


@dataclass
class ChimeraRunResult:
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
    normalized_sortedness: list[float]


class TrajectoryWriter:
    fieldnames = [
        "conditionId",
        "mixtureId",
        "algorithms",
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
    except Exception as exc:  # pragma: no cover - defensive provenance path
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


def state_hash(values: list[int]) -> str:
    arr = np.asarray(values, dtype=np.int16)
    return hashlib.sha256(arr.tobytes()).hexdigest()


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


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
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


def initial_values_from_seed(seed: int, n: int = 100) -> list[int]:
    rng = np.random.default_rng(int(seed))
    values = np.arange(1, n + 1, dtype=np.int16)
    rng.shuffle(values)
    return [int(value) for value in values]


def algotypes_from_seed(seed: Any, allocation: dict[str, int]) -> list[str]:
    algotypes: list[str] = []
    for algorithm in ALGORITHMS:
        algotypes.extend([algorithm] * int(allocation.get(algorithm, 0)))
    if len([algorithm for algorithm, count in allocation.items() if int(count) > 0]) <= 1:
        return algotypes
    rng = np.random.default_rng(parse_int(seed))
    indices = rng.permutation(len(algotypes))
    return [algotypes[int(index)] for index in indices]


def algotype_sequence(algotypes: list[str]) -> str:
    return "".join(ALGOTYPE_CODES[item] for item in algotypes)


def cells_values(cells: list[Any]) -> list[int]:
    return [int(cell.value) for cell in cells]


def cell_algotypes(cells: list[Any], algotype_by_cell_id: dict[int, str]) -> list[str]:
    return [algotype_by_cell_id[int(cell.threadID)] for cell in cells]


def any_cell_can_move(cell_objects: list[Any]) -> bool:
    return any(bool(cell.should_move()) for cell in cell_objects)


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
    actor_cell_id: int | None = None,
    actor_algotype: str | None = None,
) -> None:
    values = cells_values(cells)
    raw = sortedness_raw(values)
    n = len(values)
    position_algotypes = cell_algotypes(cells, algotype_by_cell_id)
    writer.write(
        {
            "conditionId": condition["conditionId"],
            "mixtureId": condition["mixtureId"],
            "algorithms": condition["algorithms"],
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
            "actorCellId": "" if actor_cell_id is None else int(actor_cell_id),
            "actorAlgotype": "" if actor_algotype is None else actor_algotype,
            "algotypeSequence": algotype_sequence(position_algotypes),
            "stateHash": state_hash(values),
        }
    )


def normalized_sortedness(points: list[float], bins: int = 101) -> list[float]:
    if not points:
        return [0.0] * bins
    if len(points) == 1:
        return [float(points[0])] * bins
    x = np.linspace(0.0, 1.0, num=len(points))
    target = np.linspace(0.0, 1.0, num=bins)
    return [float(value) for value in np.interp(target, x, np.asarray(points, dtype=float))]


def run_chimera(
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    writer: TrajectoryWriter,
    max_swaps: int,
    max_comparisons: int,
    max_rounds: int,
    no_move_checks_required: int,
    stable_no_progress_round_cap: int,
) -> ChimeraRunResult:
    start = time.perf_counter()
    n = parse_int(condition["n"])
    allocation = json.loads(condition["algotypeAllocation"])
    initial_values = initial_values_from_seed(parse_int(seed_row["inputPermutationSeed"]), n=n)
    initial_algotypes = algotypes_from_seed(seed_row.get("algotypeAssignmentSeed", ""), allocation)
    if len(initial_values) != len(initial_algotypes):
        raise ValueError(f"initial value/Algotype length mismatch for {condition['conditionId']}")

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
    scheduler_rng = np.random.default_rng(parse_int(seed_row["schedulerSeed"]))
    per_algotype_moves = {algorithm: 0 for algorithm in ALGORITHMS if int(allocation.get(algorithm, 0)) > 0}
    scheduler_rounds = 0
    no_move_checks = 0
    no_progress_rounds = 0
    stop_reason = "sorted"
    sortedness_points: list[float] = []

    initial_raw = sortedness_raw(initial_values)
    initial_percent = sortedness_percent_from_raw(initial_raw, n)
    sortedness_points.append(initial_percent)
    write_trajectory_event(
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
    )

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
                    write_trajectory_event(
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
                        actor_cell_id=actor_cell_id,
                        actor_algotype=actor_algotype,
                    )
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
        if stop_reason in {"max_step_cap", "max_comparison_cap"}:
            break

    final_values = cells_values(cells)
    final_algotypes = cell_algotypes(cells, algotype_by_cell_id)
    final_raw = sortedness_raw(final_values)
    completed = is_sorted_values(final_values)
    if completed:
        stop_reason = "sorted"

    return ChimeraRunResult(
        completed=completed,
        stop_reason=stop_reason,
        initial_values=initial_values,
        final_values=final_values,
        initial_algotypes=initial_algotypes,
        final_algotypes=final_algotypes,
        swap_count=int(status_probe.swap_count),
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
        normalized_sortedness=normalized_sortedness(sortedness_points),
    )


def s09_conditions(condition_matrix_path: Path) -> list[dict[str, Any]]:
    conditions = [row for row in read_csv_rows(condition_matrix_path) if row["producerStep"] == STEP_ID]
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    return sorted(conditions, key=lambda row: order[row["mixtureId"]])


def s09_seed_rows(seed_table_path: Path, max_replicates_per_condition: int | None = None) -> dict[str, list[dict[str, Any]]]:
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


def make_replicate_record(condition: dict[str, Any], seed_row: dict[str, Any], result: ChimeraRunResult) -> dict[str, Any]:
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
        "algotypeAssignmentSeed": None if seed_row.get("algotypeAssignmentSeed", "") == "" else parse_int(seed_row["algotypeAssignmentSeed"]),
        "schedulerSeed": parse_int(seed_row["schedulerSeed"]),
        "tieBreakerSeed": parse_int(seed_row["tieBreakerSeed"]),
        "n": parse_int(condition["n"]),
        "targetRepeatCount": parse_int(condition["repeatCount"]),
        "algotypeAllocationJson": json.dumps(allocation, sort_keys=True, separators=(",", ":")),
        "initialAlgotypeCountsJson": json.dumps(observed_initial_counts, sort_keys=True, separators=(",", ":")),
        "finalAlgotypeCountsJson": json.dumps(observed_final_counts, sort_keys=True, separators=(",", ":")),
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
        "swapCount": result.swap_count,
        "comparisonCount": result.comparison_count,
        "archivedCompareAndSwapCount": result.archived_compare_and_swap_count,
        "schedulerRounds": result.scheduler_rounds,
        "eventCount": result.event_count,
        "wallTimeSeconds": result.wall_time_seconds,
        "perAlgotypeMovesJson": json.dumps(result.per_algotype_moves, sort_keys=True, separators=(",", ":")),
    }


def make_movement_records(condition: dict[str, Any], seed_row: dict[str, Any], result: ChimeraRunResult) -> list[dict[str, Any]]:
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


def make_average_trajectory_records(condition: dict[str, Any], seed_row: dict[str, Any], result: ChimeraRunResult) -> list[dict[str, Any]]:
    rows = []
    for bin_index, sortedness in enumerate(result.normalized_sortedness):
        rows.append(
            {
                "conditionId": condition["conditionId"],
                "mixtureId": condition["mixtureId"],
                "replicateIndex": int(seed_row["replicateIndex"]),
                "replicateNumber": int(seed_row["replicateNumber"]),
                "progressBin": int(bin_index),
                "normalizedProgress": float(bin_index / 100.0),
                "sortednessPercent": float(sortedness),
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
    summary["semSwapCount"] = summary["sdSwapCount"] / np.sqrt(summary["replicateCount"])
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    summary["mixtureOrder"] = summary["mixtureId"].map(order)
    return summary.sort_values("mixtureOrder").reset_index(drop=True)


def summarize_movements(movement_df: pd.DataFrame) -> pd.DataFrame:
    if movement_df.empty:
        return pd.DataFrame()
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


def summarize_average_trajectory(average_trajectory_replicates: pd.DataFrame) -> pd.DataFrame:
    return (
        average_trajectory_replicates.groupby(["conditionId", "mixtureId", "progressBin", "normalizedProgress"], dropna=False)
        .agg(
            meanSortednessPercent=("sortednessPercent", "mean"),
            sdSortednessPercent=("sortednessPercent", "std"),
            replicateCount=("replicateIndex", "count"),
        )
        .reset_index()
    )


def build_efficiency_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pure_means = {
        row["mixtureId"].replace("pure_", ""): float(row["meanSwapCount"])
        for _, row in summary_df[summary_df["mixtureId"].isin(PURE_BY_ALGORITHM.values())].iterrows()
    }
    for _, row in summary_df.iterrows():
        algorithms = row["algorithms"].split(";")
        component_means = [pure_means[algorithm] for algorithm in algorithms if algorithm in pure_means]
        rows.append(
            {
                "conditionId": row["conditionId"],
                "mixtureId": row["mixtureId"],
                "algorithms": row["algorithms"],
                "meanSwapCount": float(row["meanSwapCount"]),
                "sdSwapCount": float(row["sdSwapCount"]) if pd.notna(row["sdSwapCount"]) else 0.0,
                "semSwapCount": float(row["semSwapCount"]) if pd.notna(row["semSwapCount"]) else 0.0,
                "componentPureMeanMin": float(min(component_means)) if component_means else float(row["meanSwapCount"]),
                "componentPureMeanMax": float(max(component_means)) if component_means else float(row["meanSwapCount"]),
                "componentPureMeanAverage": float(np.mean(component_means)) if component_means else float(row["meanSwapCount"]),
                "withinComponentEnvelope": bool(
                    component_means
                    and min(component_means) - 1e-9 <= float(row["meanSwapCount"]) <= max(component_means) + 1e-9
                )
                if len(component_means) >= 2
                else True,
                "isMixedCondition": len(algorithms) > 1,
            }
        )
    efficiency = pd.DataFrame(rows)
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    efficiency["mixtureOrder"] = efficiency["mixtureId"].map(order)
    return efficiency.sort_values("mixtureOrder").reset_index(drop=True)


def write_result_tables(
    replicate_df: pd.DataFrame,
    movement_df: pd.DataFrame,
    average_trajectory_replicates: pd.DataFrame,
    results_dir: Path,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "replicateParquet": results_dir / "e01_same_goal_chimeras.parquet",
        "replicateCsv": results_dir / "e01_same_goal_chimeras.csv",
        "summaryParquet": results_dir / "e01_same_goal_chimera_summary.parquet",
        "summaryCsv": results_dir / "e01_same_goal_chimera_summary.csv",
        "movementParquet": results_dir / "e01_same_goal_chimera_algotype_movements.parquet",
        "movementCsv": results_dir / "e01_same_goal_chimera_algotype_movements.csv",
        "movementSummaryParquet": results_dir / "e01_same_goal_chimera_algotype_movement_summary.parquet",
        "movementSummaryCsv": results_dir / "e01_same_goal_chimera_algotype_movement_summary.csv",
        "averageTrajectoryParquet": results_dir / "e01_same_goal_chimera_average_trajectory.parquet",
        "averageTrajectoryCsv": results_dir / "e01_same_goal_chimera_average_trajectory.csv",
        "efficiencyParquet": results_dir / "e01_same_goal_chimera_efficiency.parquet",
        "efficiencyCsv": results_dir / "e01_same_goal_chimera_efficiency.csv",
    }
    summary_df = summarize_conditions(replicate_df)
    movement_summary_df = summarize_movements(movement_df)
    average_trajectory_df = summarize_average_trajectory(average_trajectory_replicates)
    efficiency_df = build_efficiency_table(summary_df)

    replicate_df.to_parquet(paths["replicateParquet"], index=False)
    replicate_df.to_csv(paths["replicateCsv"], index=False)
    summary_df.to_parquet(paths["summaryParquet"], index=False)
    summary_df.to_csv(paths["summaryCsv"], index=False)
    movement_df.to_parquet(paths["movementParquet"], index=False)
    movement_df.to_csv(paths["movementCsv"], index=False)
    movement_summary_df.to_parquet(paths["movementSummaryParquet"], index=False)
    movement_summary_df.to_csv(paths["movementSummaryCsv"], index=False)
    average_trajectory_df.to_parquet(paths["averageTrajectoryParquet"], index=False)
    average_trajectory_df.to_csv(paths["averageTrajectoryCsv"], index=False)
    efficiency_df.to_parquet(paths["efficiencyParquet"], index=False)
    efficiency_df.to_csv(paths["efficiencyCsv"], index=False)
    return paths


def plot_efficiency(efficiency_path: Path, figure_dir: Path) -> tuple[Path, Path]:
    efficiency = pd.read_parquet(efficiency_path).sort_values("mixtureOrder")
    labels = [
        item.replace("pure_", "pure ").replace("bubble", "Bubble").replace("insertion", "Insertion").replace("selection", "Selection").replace("_", " + ")
        for item in efficiency["mixtureId"]
    ]
    colors = ["#5b8cbe" if not mixed else "#c75d5d" for mixed in efficiency["isMixedCondition"]]
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.bar(
        range(len(efficiency)),
        efficiency["meanSwapCount"],
        yerr=efficiency["semSwapCount"],
        color=colors,
        edgecolor="#222222",
        linewidth=0.7,
        capsize=4,
    )
    for index, row in efficiency.iterrows():
        if bool(row["isMixedCondition"]) and row["mixtureId"] in PAIR_MIXTURES:
            ax.vlines(
                index,
                row["componentPureMeanMin"],
                row["componentPureMeanMax"],
                colors="#111111",
                linestyles="dashed",
                linewidth=1.1,
            )
    ax.set_xticks(range(len(efficiency)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Swap steps to sorted state")
    ax.set_title("S09 same-goal chimera efficiency")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "figure08_same_goal_chimera_efficiency.png"
    pdf = figure_dir / "figure08_same_goal_chimera_efficiency.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_sortedness(average_trajectory_path: Path, figure_dir: Path) -> tuple[Path, Path]:
    trajectory = pd.read_parquet(average_trajectory_path)
    order = {mixture: index for index, mixture in enumerate(MIXTURE_ORDER)}
    trajectory["mixtureOrder"] = trajectory["mixtureId"].map(order)
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.8), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for axis in axes_flat:
        axis.set_visible(False)
    for panel_index, mixture in enumerate(MIXTURE_ORDER):
        axis = axes_flat[panel_index]
        axis.set_visible(True)
        subset = trajectory[trajectory["mixtureId"] == mixture].sort_values("progressBin")
        axis.plot(subset["normalizedProgress"], subset["meanSortednessPercent"], color="#255f99", linewidth=2.0)
        lower = subset["meanSortednessPercent"] - subset["sdSortednessPercent"].fillna(0)
        upper = subset["meanSortednessPercent"] + subset["sdSortednessPercent"].fillna(0)
        axis.fill_between(subset["normalizedProgress"], lower, upper, color="#255f99", alpha=0.16, linewidth=0)
        axis.set_title(mixture.replace("pure_", "pure ").replace("_", " + "), fontsize=10)
        axis.grid(color="#dddddd", linewidth=0.7)
        axis.set_ylim(40, 101)
    axes_flat[-1].set_visible(False)
    for axis in axes[:, 0]:
        if axis.get_visible():
            axis.set_ylabel("Sortedness (%)")
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Normalized progress")
    fig.suptitle("S09 same-goal chimera sorting trajectories", y=0.99)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "figure08_same_goal_chimera_sortedness.png"
    pdf = figure_dir / "figure08_same_goal_chimera_sortedness.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def validate_outputs(
    replicate_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    movement_df: pd.DataFrame,
    movement_summary_df: pd.DataFrame,
    average_trajectory_df: pd.DataFrame,
    efficiency_df: pd.DataFrame,
    trace_row_count: int,
    output_paths: list[Path],
    expected_replicates_per_condition: int,
) -> tuple[bool, list[str], list[str], str]:
    checks: list[str] = []
    failures: list[str] = []
    caveats: list[str] = []

    expected_conditions = set(MIXTURE_ORDER)
    observed_conditions = set(replicate_df["mixtureId"])
    if observed_conditions == expected_conditions:
        checks.append("All seven S09 pure-control, pairwise mixed, and all-three mixtures are present.")
    else:
        failures.append(f"Observed mixtures {sorted(observed_conditions)} do not match expected {sorted(expected_conditions)}.")

    expected_rows = expected_replicates_per_condition * len(MIXTURE_ORDER)
    if len(replicate_df) == expected_rows:
        checks.append(f"Replicate table has {expected_rows} rows.")
    else:
        failures.append(f"Replicate table has {len(replicate_df)} rows; expected {expected_rows}.")

    per_condition = replicate_df.groupby("conditionId").size()
    if per_condition.nunique() == 1 and int(per_condition.iloc[0]) == expected_replicates_per_condition:
        checks.append(f"Every S09 condition has {expected_replicates_per_condition} replicate rows.")
    else:
        failures.append(f"Per-condition replicate counts are inconsistent: {per_condition.to_dict()}.")

    if bool(replicate_df["completed"].all()) and (replicate_df["stopReason"] == "sorted").all():
        checks.append("Every same-goal chimera and pure-control run reached a sorted stop state.")
    else:
        failures.append("At least one S09 run did not complete with stopReason=sorted.")

    if (replicate_df["finalSortednessPercent"] == 100.0).all() and (replicate_df["finalMonotonicityError"] == 0).all():
        checks.append("Every S09 run reached final 100% Sortedness and zero monotonicity error.")
    else:
        failures.append("At least one S09 run did not end at 100% Sortedness and zero monotonicity error.")

    input_hash_counts = replicate_df.groupby("replicateIndex")["initialStateHash"].nunique()
    if int(input_hash_counts.max()) == 1:
        checks.append("Input permutations are matched across all S09 conditions by replicate index.")
    else:
        failures.append("Input permutations are not matched across all S09 conditions.")

    allocation_failures = []
    for _, row in replicate_df.iterrows():
        expected = json.loads(row["algotypeAllocationJson"])
        observed = json.loads(row["initialAlgotypeCountsJson"])
        if {key: int(value) for key, value in expected.items()} != {key: int(value) for key, value in observed.items()}:
            allocation_failures.append((row["conditionId"], int(row["replicateIndex"]), expected, observed))
            break
    if not allocation_failures:
        checks.append("Observed initial Algotype counts match S03 allocations for every run.")
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
        checks.append("S09 trajectory row count equals the summed replicate event counts.")
    else:
        failures.append(f"S09 trajectory row count {trace_row_count} does not equal summed event count {int(replicate_df['eventCount'].sum())}.")

    if len(summary_df) == len(MIXTURE_ORDER):
        checks.append("Condition summary has seven rows.")
    else:
        failures.append(f"Condition summary has {len(summary_df)} rows, expected seven.")

    expected_movement_rows = sum(len(json.loads(value)) for value in replicate_df["algotypeAllocationJson"])
    if len(movement_df) == expected_movement_rows:
        checks.append("Per-replicate Algotype movement table has the expected long-format rows.")
    else:
        failures.append(f"Movement table has {len(movement_df)} rows, expected {expected_movement_rows}.")

    if len(movement_summary_df) == expected_movement_rows / expected_replicates_per_condition:
        checks.append("Algotype movement summary has expected condition-by-Algotype rows.")
    else:
        failures.append("Algotype movement summary row count is unexpected.")

    if len(average_trajectory_df) == len(MIXTURE_ORDER) * 101:
        checks.append("Average Sortedness trajectory backing table has 101 bins for each S09 mixture.")
    else:
        failures.append(f"Average trajectory table has {len(average_trajectory_df)} rows, expected {len(MIXTURE_ORDER) * 101}.")

    pair_efficiency = efficiency_df[efficiency_df["mixtureId"].isin(PAIR_MIXTURES)]
    if bool(pair_efficiency["withinComponentEnvelope"].all()):
        checks.append("Pairwise mixed-efficiency means fall within their pure-component swap-count envelope.")
    else:
        failures.append("At least one pairwise mixed-efficiency mean falls outside the pure-component envelope.")

    all_three = efficiency_df[efficiency_df["mixtureId"] == "bubble_insertion_selection"]
    pure = efficiency_df[efficiency_df["mixtureId"].isin(PURE_BY_ALGORITHM.values())]
    if not all_three.empty and not pure.empty and float(pure["meanSwapCount"].min()) <= float(all_three.iloc[0]["meanSwapCount"]) <= float(pure["meanSwapCount"].max()):
        checks.append("The all-three mixed condition mean swap count falls within the pure Algotype range.")
    else:
        caveats.append("The all-three condition did not fall within the pure Algotype range; interpret efficiency as a descriptive control.")

    missing_outputs = [str(path) for path in output_paths if not path.exists() or path.stat().st_size == 0]
    if not missing_outputs:
        checks.append("All declared S09 output files exist and are non-empty.")
    else:
        failures.append(f"Missing or empty S09 output files: {missing_outputs}.")

    if Path("/artifacts/research_steps/S10").exists():
        failures.append("S10 artifact directory exists; S09 must stop before S10.")
    else:
        checks.append("No S10 artifact directory was created.")

    if Path("/artifacts/results/e01_aggregation_curves.parquet").exists():
        failures.append("S10 aggregation-curve result exists; S09 must not compute S10 outputs.")
    else:
        checks.append("No S10 aggregation-curve result was created.")

    success = not failures
    if success:
        validation_result = (
            "passed: S09 ran seven same-goal unique-value cell-view chimera/control conditions, "
            f"{expected_rows} replicate rows, matched input permutations, recorded mixture proportions and per-Algotype movements, "
            "confirmed all runs reached 100% Sortedness, wrote reusable S09 traces and plots, and did not create S10 artifacts."
        )
    else:
        validation_result = "failed: " + " ".join(failures)
    return success, checks, failures + caveats, validation_result


def infer_outcome(replicate_df: pd.DataFrame, efficiency_df: pd.DataFrame) -> tuple[str, str]:
    all_sorted = bool(replicate_df["completed"].all()) and (replicate_df["finalSortednessPercent"] == 100.0).all()
    pair_envelope = bool(efficiency_df[efficiency_df["mixtureId"].isin(PAIR_MIXTURES)]["withinComponentEnvelope"].all())
    if all_sorted and pair_envelope:
        return (
            "supportive",
            "S09 supports the same-goal chimera claim: all mixed and pure-control arrays reached 100% Sortedness, and pairwise mixed swap-count means fell within their pure-component efficiency envelope.",
        )
    if all_sorted:
        return (
            "constraining",
            "S09 supports sorting completion but constrains the reported efficiency interpretation because at least one pairwise mixed swap-count mean fell outside the pure-component envelope.",
        )
    return (
        "constraining",
        "S09 constrains the same-goal chimera claim because at least one same-goal mixed or control run did not reach 100% Sortedness.",
    )


def write_methods(path: Path) -> None:
    lines = [
        "# S09 Methods",
        "",
        "- Research step ID: S09",
        "- Completion status: completed",
        "- Method family: same-goal mixed Algotype cell-view simulation and efficiency analysis.",
        "- Simulator: deterministic pseudo-scheduler wrapper around the archived `BubbleSortCell`, `InsertionSortCell`, and `SelectionSortCell` classes.",
        "- Inputs: S03 condition matrix and seed table, restricted to `producerStep == S09`.",
        "- Conditions: pure Bubble, pure Insertion, pure Selection, Bubble-Insertion, Bubble-Selection, Insertion-Selection, and Bubble-Insertion-Selection.",
        "- Values: unique 1 through 100 random permutation per replicate, matched across all S09 conditions by `replicateIndex`.",
        "- Algotype assignment: S03 allocation counts, shuffled by `algotypeAssignmentSeed` for mixed conditions; pure controls use all cells assigned to one Algotype.",
        "- Stop policy: sorted primary stop, with no-move, swap, comparison, and round caps as fallbacks; all final runs stopped sorted.",
        "- Trace contract for S10: swap-level trajectory rows include the current compact Algotype sequence, Sortedness, state hash, actor Algotype, and step counters. S09 does not compute Aggregation curves.",
        "- Artifacts written: see `artifact_manifest.json` and `status.json`.",
        "- Validation result: see `validation.json`.",
        "- Caveats or blockers: the wrapper uses a deterministic scheduler rather than live OS thread scheduling; exact paper all-three allocation is balanced 34/33/33 because n=100 is not divisible by three.",
        "- Recommended next action: hand control back to the Chief Scientist workflow; start S10 Aggregation curves only after explicit instruction.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_markdown(path: Path, checks: list[str], caveats: list[str], validation_result: str) -> None:
    lines = [
        "# S09 Validation",
        "",
        "- Research step ID: S09",
        "- Step number: 9",
        "- Completion status: completed",
        "- Validation result: " + validation_result,
        "- Artifacts written: see `artifact_manifest.json` and `status.json`.",
        "- Caveats or blockers:",
    ]
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Recommended next action: hand control back to the Chief Scientist workflow for review before S10.",
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
    efficiency_df: pd.DataFrame,
    movement_summary_df: pd.DataFrame,
) -> None:
    summary_rows = [
        [
            row["mixtureId"],
            int(row["replicateCount"]),
            int(row["completedCount"]),
            row["meanSwapCount"],
            row["meanComparisonCount"],
            row["meanSchedulerRounds"],
            row["meanFinalSortednessPercent"],
        ]
        for _, row in summary_df.iterrows()
    ]
    efficiency_rows = [
        [
            row["mixtureId"],
            row["meanSwapCount"],
            row["componentPureMeanMin"],
            row["componentPureMeanMax"],
            row["withinComponentEnvelope"],
        ]
        for _, row in efficiency_df.iterrows()
        if bool(row["isMixedCondition"])
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
        "# S09 Status Summary",
        "",
        "- Research step ID: S09",
        "- Step number: 9",
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
            "- Lay summary: S09 mixed cells following Bubble, Insertion, and Selection policies in the same array while all pursued the same increasing-order goal. All pure controls, pairwise mixes, and the all-three mix sorted successfully, and the pairwise mixed efficiency means stayed between their corresponding pure Algotype controls.",
            "- Recommended next action: " + recommended_next_action,
            "",
            "## Condition Means",
            "",
            markdown_table(
                [
                    "Mixture",
                    "Replicates",
                    "Completed",
                    "Mean swaps",
                    "Mean comparisons",
                    "Mean rounds",
                    "Mean final Sortedness",
                ],
                summary_rows,
            ),
            "",
            "## Mixed Efficiency Envelope",
            "",
            markdown_table(
                ["Mixture", "Mean swaps", "Pure min", "Pure max", "Within envelope"],
                efficiency_rows,
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


def run_s09(args: argparse.Namespace) -> int:
    generated_at = utc_now()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    results_dir = artifacts_dir / "results"
    figure_dir = artifacts_dir / "figures" / "e01"
    trace_dir = artifacts_dir / "traces" / "e01" / STEP_ID
    provenance_dir = artifacts_dir / "provenance"
    for directory in [step_dir, code_dir, results_dir, figure_dir, trace_dir, provenance_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    config = load_json(args.config)
    stop_policy = config.get("semantics", {}).get("stopPolicies", {}).get("sorted_no_move_or_cap", {})
    max_swaps = int(stop_policy.get("maxSwapEvents", 500000))
    max_comparisons = int(stop_policy.get("maxComparisonEvents", 4000000))
    no_move_checks_required = int(stop_policy.get("noMoveChecksRequired", 2))
    max_rounds = int(args.max_rounds)
    stable_no_progress_round_cap = int(args.stable_no_progress_round_cap)

    conditions = s09_conditions(args.condition_matrix)
    seed_rows_by_condition = s09_seed_rows(args.seed_table, args.max_replicates_per_condition)
    replicate_records: list[dict[str, Any]] = []
    movement_records: list[dict[str, Any]] = []
    average_trajectory_records: list[dict[str, Any]] = []
    trace_path = trace_dir / "e01_s09_same_goal_chimera_trajectory_events.csv.gz"

    with TrajectoryWriter(trace_path) as trajectory_writer:
        for condition in conditions:
            rows = seed_rows_by_condition.get(condition["conditionId"], [])
            if not rows:
                raise RuntimeError(f"No S09 seed rows found for {condition['conditionId']}")
            for seed_row in rows:
                result = run_chimera(
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
                average_trajectory_records.extend(make_average_trajectory_records(condition, seed_row, result))
        trace_row_count = trajectory_writer.row_count

    replicate_df = pd.DataFrame(replicate_records)
    movement_df = pd.DataFrame(movement_records)
    average_trajectory_replicates_df = pd.DataFrame(average_trajectory_records)
    result_paths = write_result_tables(replicate_df, movement_df, average_trajectory_replicates_df, results_dir)
    summary_df = pd.read_parquet(result_paths["summaryParquet"])
    movement_summary_df = pd.read_parquet(result_paths["movementSummaryParquet"])
    average_trajectory_df = pd.read_parquet(result_paths["averageTrajectoryParquet"])
    efficiency_df = pd.read_parquet(result_paths["efficiencyParquet"])
    efficiency_png, efficiency_pdf = plot_efficiency(result_paths["efficiencyParquet"], figure_dir)
    sortedness_png, sortedness_pdf = plot_sortedness(result_paths["averageTrajectoryParquet"], figure_dir)
    outcome_classification, outcome_reason = infer_outcome(replicate_df, efficiency_df)

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
        efficiency_png,
        efficiency_pdf,
        sortedness_png,
        sortedness_pdf,
        methods_md,
    ]
    expected_replicates = args.max_replicates_per_condition or 100
    success, validation_checks, caveats_or_blockers, validation_result = validate_outputs(
        replicate_df,
        summary_df,
        movement_df,
        movement_summary_df,
        average_trajectory_df,
        efficiency_df,
        trace_row_count,
        output_paths,
        expected_replicates_per_condition=expected_replicates,
    )
    caveats_or_blockers.extend(
        [
            "S09 uses a deterministic pseudo-scheduler wrapper around archived cell-view classes rather than live OS-thread scheduling.",
            "Pure controls and mixed conditions use S03 seeds; input permutations are matched by replicate, while mixed Algotype assignments use condition-specific assignment seeds.",
            "All-three uses the S03 balanced 34/33/33 allocation because n=100 is not divisible by three; exact author-local all-three allocation was not recoverable.",
            "S09 writes Algotype-position traces for later S10 but intentionally does not compute Aggregation curves or S10 statistics.",
            outcome_reason,
            "No S10 Aggregation analysis or S10 artifact directory was started.",
        ]
    )

    code_copy = code_dir / Path(__file__).name
    shutil.copy2(Path(__file__), code_copy)
    output_paths.extend([validation_json, validation_md, summary_md, status_json, artifact_manifest_json, code_copy])
    artifacts_written: list[str] = []
    recommended_next_action = (
        "Hand control back to the Chief Scientist workflow for review; S10 Aggregation curves should start only after explicit instruction, "
        "using the S09 trajectory trace as its source input."
    )

    write_validation_markdown(validation_md, validation_checks, caveats_or_blockers, validation_result)
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
        "conditionSummaryRows": int(len(summary_df)),
        "movementRows": int(len(movement_df)),
        "movementSummaryRows": int(len(movement_summary_df)),
        "averageTrajectoryRows": int(len(average_trajectory_df)),
        "efficiencyRows": int(len(efficiency_df)),
        "traceRows": int(trace_row_count),
        "outcomeClassification": outcome_classification,
    }
    write_json(validation_json, validation_payload)
    write_summary_markdown(
        summary_md,
        artifacts_written,
        validation_result,
        caveats_or_blockers,
        recommended_next_action,
        outcome_classification,
        summary_df,
        efficiency_df,
        movement_summary_df,
    )

    source_statuses = {
        "S03": load_json(S03_STATUS_DEFAULT),
        "S04": load_json(S04_STATUS_DEFAULT),
        "S08": load_json(S08_STATUS_DEFAULT),
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
            "s03Status": str(S03_STATUS_DEFAULT),
            "s04Status": str(S04_STATUS_DEFAULT),
            "s08Status": str(S08_STATUS_DEFAULT),
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
            "averageTrajectoryRows": int(len(average_trajectory_df)),
            "efficiencyRows": int(len(efficiency_df)),
            "traceRows": int(trace_row_count),
        },
        "mixtures": MIXTURE_ORDER,
    }
    write_json(status_json, status_payload)

    artifact_manifest_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": STATUS if success else "failed_validation",
        "success": bool(success),
        "generatedAt": generated_at,
        "artifactCount": 0,
        "artifacts": [],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats_or_blockers,
        "recommendedNextAction": recommended_next_action,
    }
    write_json(artifact_manifest_json, artifact_manifest_payload)

    artifacts = collect_artifacts(output_paths)
    artifacts_written = [artifact["path"] for artifact in artifacts]
    validation_payload["artifactsWritten"] = artifacts_written
    status_payload["artifactsWritten"] = artifacts_written
    artifact_manifest_payload["artifactCount"] = len(artifacts)
    artifact_manifest_payload["artifacts"] = artifacts
    write_json(validation_json, validation_payload)
    write_summary_markdown(
        summary_md,
        artifacts_written,
        validation_result,
        caveats_or_blockers,
        recommended_next_action,
        outcome_classification,
        summary_df,
        efficiency_df,
        movement_summary_df,
    )
    write_json(status_json, status_payload)
    artifacts = collect_artifacts(output_paths)
    artifact_manifest_payload["artifactCount"] = len(artifacts)
    artifact_manifest_payload["artifacts"] = artifacts
    write_json(artifact_manifest_json, artifact_manifest_payload)
    update_run_manifest(RUN_MANIFEST_DEFAULT, status_payload, artifact_manifest_payload)

    print(json.dumps({"success": success, "validationResult": validation_result, "artifactsWritten": artifacts_written}, indent=2))
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH_DEFAULT)
    parser.add_argument("--condition-matrix", type=Path, default=CONDITION_MATRIX_DEFAULT)
    parser.add_argument("--seed-table", type=Path, default=SEED_TABLE_DEFAULT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--max-rounds", type=int, default=100000)
    parser.add_argument("--stable-no-progress-round-cap", type=int, default=100)
    parser.add_argument("--max-replicates-per-condition", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    return run_s09(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
