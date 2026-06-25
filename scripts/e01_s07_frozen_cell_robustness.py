#!/usr/bin/env python3
"""Run E01 S07 Frozen Cell robustness experiments.

S07 consumes the S03 baseline config, condition matrix, and seed table. It
runs only the S07 Frozen Cell conditions, writes Figure 5-style backing data
and traces for later S08 Delayed Gratification, and stops before S08.
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
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "E01"
STEP_ID = "S07"
STEP_NUMBER = 7
STATUS = "completed"
CONFIG_PATH_DEFAULT = Path("/artifacts/configs/e01_baseline_config.json")
CONDITION_MATRIX_DEFAULT = Path("/artifacts/research_steps/S03/condition_matrix.csv")
SEED_TABLE_DEFAULT = Path("/artifacts/research_steps/S03/seed_table.csv")
S02_STATUS_DEFAULT = Path("/artifacts/research_steps/S02/status.json")
S03_STATUS_DEFAULT = Path("/artifacts/research_steps/S03/status.json")
S04_REPLICATE_SUMMARY_DEFAULT = Path("/artifacts/results/e01_s04_replicate_summary.parquet")
S04_TRAJECTORY_EVENTS_DEFAULT = Path("/artifacts/traces/e01/S04/trajectory_events.parquet")
S06_STATUS_DEFAULT = Path("/artifacts/research_steps/S06/status.json")
RUN_MANIFEST_DEFAULT = Path("/artifacts/provenance/run_manifest.json")
ALGORITHMS = ["bubble", "insertion", "selection"]
IMPLEMENTATIONS = ["traditional", "cell_view"]
FROZEN_VARIANTS = ["none", "passive", "stuck"]
FIGURE_VARIANTS = ["passive", "stuck"]
STOP_REASONS = {
    "sorted",
    "no_cell_can_move_after_two_checks",
    "stable_sortedness_window",
    "max_step_cap",
    "max_comparison_cap",
    "wall_time_cap",
    "error",
}
STABLE_SORTEDNESS_ROUND_WINDOW = 200


@dataclass
class SimulationState:
    values: list[int]
    frozen: list[bool]
    cell_ids: list[int]
    sorted_pairs: int
    comparisons: int = 0
    swaps: int = 0
    scheduler_rounds: int = 0
    passes: int = 0
    blocked_move_attempts: int = 0


@dataclass
class RunResult:
    completed: bool
    stop_reason: str
    final_values: list[int]
    final_frozen_positions: list[int]
    final_sortedness_raw_count: int
    final_sortedness_percent: float
    final_monotonicity_error: int
    swap_count: int
    comparison_count: int
    scheduler_rounds: int
    pass_count: int
    blocked_move_attempts: int
    event_count: int
    wall_time_seconds: float


class TrajectoryWriter:
    """Stream compact sortedness trajectories without retaining all rows."""

    fieldnames = [
        "conditionId",
        "implementation",
        "algorithm",
        "frozenVariant",
        "frozenCount",
        "replicateIndex",
        "replicateNumber",
        "eventIndex",
        "eventKind",
        "swapCount",
        "comparisonCount",
        "schedulerRound",
        "passCount",
        "sortednessRawCount",
        "sortednessPercent",
        "monotonicityError",
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
        self.writer.writerow(row)
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


def state_signature(state: SimulationState, extra: tuple[int, ...] | None = None) -> bytes:
    values = np.asarray(state.values, dtype=np.int16).tobytes()
    frozen = bytes(1 if value else 0 for value in state.frozen)
    if extra is None:
        return values + b"|" + frozen
    return values + b"|" + frozen + b"|" + np.asarray(extra, dtype=np.int16).tobytes()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
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


def sortedness_raw(values: list[int]) -> int:
    return sum(1 for i in range(len(values) - 1) if values[i] <= values[i + 1])


def sortedness_percent_from_raw(raw_count: int, n: int) -> float:
    if n < 2:
        return 100.0
    return 100.0 * raw_count / (n - 1)


def monotonicity_error_from_raw(raw_count: int, n: int) -> int:
    return (n - 1) - raw_count


def is_sorted_state(state: SimulationState) -> bool:
    return state.sorted_pairs == len(state.values) - 1


def pair_ok(values: list[int], index: int) -> bool:
    return values[index] <= values[index + 1]


def initial_values_from_seed(seed: int, n: int = 100) -> list[int]:
    rng = np.random.default_rng(int(seed))
    values = np.arange(1, n + 1, dtype=np.int16)
    rng.shuffle(values)
    return [int(value) for value in values]


def frozen_positions_from_seed(seed: Any, frozen_count: int, n: int = 100) -> list[int]:
    if frozen_count == 0:
        return []
    if seed is None or (isinstance(seed, float) and math.isnan(seed)):
        raise ValueError("frozen_position_seed is required for frozen_count > 0")
    rng = np.random.default_rng(int(seed))
    return sorted(int(value) for value in rng.choice(n, size=int(frozen_count), replace=False))


def init_state(initial_values: list[int], frozen_positions: list[int]) -> SimulationState:
    n = len(initial_values)
    frozen_set = set(frozen_positions)
    frozen = [idx in frozen_set for idx in range(n)]
    return SimulationState(
        values=list(initial_values),
        frozen=frozen,
        cell_ids=list(range(n)),
        sorted_pairs=sortedness_raw(initial_values),
    )


def current_frozen_positions(state: SimulationState) -> list[int]:
    return [idx for idx, is_frozen in enumerate(state.frozen) if is_frozen]


def can_swap(state: SimulationState, actor_pos: int, target_pos: int, frozen_variant: str) -> bool:
    if actor_pos == target_pos:
        return False
    if state.frozen[actor_pos]:
        return False
    if frozen_variant == "stuck" and state.frozen[target_pos]:
        return False
    return True


def swap_positions(state: SimulationState, pos_a: int, pos_b: int) -> tuple[int, int]:
    n = len(state.values)
    affected = {idx for pos in (pos_a, pos_b) for idx in (pos - 1, pos) if 0 <= idx < n - 1}
    before = sum(1 for idx in affected if pair_ok(state.values, idx))
    id_a = state.cell_ids[pos_a]
    id_b = state.cell_ids[pos_b]
    state.values[pos_a], state.values[pos_b] = state.values[pos_b], state.values[pos_a]
    state.frozen[pos_a], state.frozen[pos_b] = state.frozen[pos_b], state.frozen[pos_a]
    state.cell_ids[pos_a], state.cell_ids[pos_b] = state.cell_ids[pos_b], state.cell_ids[pos_a]
    after = sum(1 for idx in affected if pair_ok(state.values, idx))
    state.sorted_pairs += after - before
    state.swaps += 1
    return id_a, id_b


def emit_trajectory(
    writer: TrajectoryWriter,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    state: SimulationState,
    event_index: int,
    event_kind: str,
) -> None:
    n = len(state.values)
    writer.write(
        {
            "conditionId": condition["conditionId"],
            "implementation": condition["implementation"],
            "algorithm": condition["algorithms"],
            "frozenVariant": condition["frozenVariant"],
            "frozenCount": int(condition["frozenCount"]),
            "replicateIndex": int(seed_row["replicateIndex"]),
            "replicateNumber": int(seed_row["replicateNumber"]),
            "eventIndex": int(event_index),
            "eventKind": event_kind,
            "swapCount": int(state.swaps),
            "comparisonCount": int(state.comparisons),
            "schedulerRound": int(state.scheduler_rounds),
            "passCount": int(state.passes),
            "sortednessRawCount": int(state.sorted_pairs),
            "sortednessPercent": sortedness_percent_from_raw(state.sorted_pairs, n),
            "monotonicityError": monotonicity_error_from_raw(state.sorted_pairs, n),
        }
    )


def record_swap_event(
    writer: TrajectoryWriter,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    state: SimulationState,
) -> int:
    emit_trajectory(writer, condition, seed_row, state, state.swaps, "swap")
    return state.swaps


def reached_cap(state: SimulationState, max_swaps: int, max_comparisons: int) -> str | None:
    if state.swaps >= max_swaps:
        return "max_step_cap"
    if state.comparisons >= max_comparisons:
        return "max_comparison_cap"
    return None


def run_traditional_bubble(
    state: SimulationState,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    writer: TrajectoryWriter,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int,
) -> tuple[str, int]:
    event_count = 1
    no_move_checks = 0
    frozen_variant = condition["frozenVariant"]

    while True:
        if is_sorted_state(state):
            return "sorted", event_count
        swaps_before = state.swaps
        for idx in range(len(state.values) - 1):
            state.comparisons += 1
            if state.values[idx] > state.values[idx + 1]:
                if can_swap(state, idx, idx + 1, frozen_variant):
                    swap_positions(state, idx, idx + 1)
                    event_count += 1
                    record_swap_event(writer, condition, seed_row, state)
                else:
                    state.blocked_move_attempts += 1
            cap = reached_cap(state, max_swaps, max_comparisons)
            if cap:
                return cap, event_count
        state.passes += 1
        if state.swaps == swaps_before:
            no_move_checks += 1
        else:
            no_move_checks = 0
        if no_move_checks >= no_move_checks_required:
            return "no_cell_can_move_after_two_checks", event_count


def run_traditional_insertion(
    state: SimulationState,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    writer: TrajectoryWriter,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int,
) -> tuple[str, int]:
    event_count = 1
    no_move_checks = 0
    frozen_variant = condition["frozenVariant"]

    while True:
        if is_sorted_state(state):
            return "sorted", event_count
        swaps_before = state.swaps
        for idx in range(1, len(state.values)):
            cursor = idx
            while cursor > 0:
                state.comparisons += 1
                if state.values[cursor] >= state.values[cursor - 1]:
                    break
                if can_swap(state, cursor, cursor - 1, frozen_variant):
                    swap_positions(state, cursor, cursor - 1)
                    event_count += 1
                    record_swap_event(writer, condition, seed_row, state)
                    cursor -= 1
                else:
                    state.blocked_move_attempts += 1
                    break
                cap = reached_cap(state, max_swaps, max_comparisons)
                if cap:
                    return cap, event_count
            cap = reached_cap(state, max_swaps, max_comparisons)
            if cap:
                return cap, event_count
        state.passes += 1
        if state.swaps == swaps_before:
            no_move_checks += 1
        else:
            no_move_checks = 0
        if no_move_checks >= no_move_checks_required:
            return "no_cell_can_move_after_two_checks", event_count


def run_traditional_selection(
    state: SimulationState,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    writer: TrajectoryWriter,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int,
) -> tuple[str, int]:
    event_count = 1
    no_move_checks = 0
    frozen_variant = condition["frozenVariant"]
    n = len(state.values)

    while True:
        if is_sorted_state(state):
            return "sorted", event_count
        swaps_before = state.swaps
        for start in range(n - 1):
            min_idx = start
            for idx in range(start + 1, n):
                state.comparisons += 1
                if state.values[idx] < state.values[min_idx]:
                    min_idx = idx
                cap = reached_cap(state, max_swaps, max_comparisons)
                if cap:
                    return cap, event_count
            if min_idx != start:
                if can_swap(state, min_idx, start, frozen_variant):
                    swap_positions(state, min_idx, start)
                    event_count += 1
                    record_swap_event(writer, condition, seed_row, state)
                else:
                    state.blocked_move_attempts += 1
            cap = reached_cap(state, max_swaps, max_comparisons)
            if cap:
                return cap, event_count
        state.passes += 1
        if state.swaps == swaps_before:
            no_move_checks += 1
        else:
            no_move_checks = 0
        if no_move_checks >= no_move_checks_required:
            return "no_cell_can_move_after_two_checks", event_count


def pos_by_cell_id(state: SimulationState) -> dict[int, int]:
    return {cell_id: pos for pos, cell_id in enumerate(state.cell_ids)}


def legal_bubble_move_exists(state: SimulationState, frozen_variant: str) -> bool:
    n = len(state.values)
    for pos in range(n):
        if state.frozen[pos]:
            continue
        if pos < n - 1 and state.values[pos] > state.values[pos + 1] and can_swap(state, pos, pos + 1, frozen_variant):
            return True
        if pos > 0 and state.values[pos] < state.values[pos - 1] and can_swap(state, pos, pos - 1, frozen_variant):
            return True
    return False


def insertion_left_segment_ready(state: SimulationState, pos: int) -> bool:
    previous: int | None = None
    for idx in range(pos):
        if state.frozen[idx]:
            previous = None
            continue
        if previous is not None:
            state.comparisons += 1
            if state.values[idx] < previous:
                return False
        previous = state.values[idx]
    return True


def insertion_ready_prefixes(state: SimulationState) -> list[bool]:
    """Return whether each position has a sorted active segment to its left."""
    ready = [True] * len(state.values)
    previous: int | None = None
    segment_ready = True
    for pos, value in enumerate(state.values):
        ready[pos] = segment_ready
        if state.frozen[pos]:
            previous = None
            segment_ready = True
            continue
        if previous is not None and value < previous:
            segment_ready = False
        previous = value
    return ready


def legal_insertion_move_exists(state: SimulationState, frozen_variant: str) -> bool:
    n = len(state.values)
    for pos in range(1, n):
        if state.frozen[pos]:
            continue
        previous: int | None = None
        ready = True
        for idx in range(pos):
            if state.frozen[idx]:
                previous = None
                continue
            if previous is not None and state.values[idx] < previous:
                ready = False
                break
            previous = state.values[idx]
        if ready and state.values[pos] < state.values[pos - 1] and can_swap(state, pos, pos - 1, frozen_variant):
            return True
    return False


def selection_action_exists(state: SimulationState, ideal_by_id: dict[int, int], frozen_variant: str) -> bool:
    n = len(state.values)
    positions = pos_by_cell_id(state)
    for cell_id, ideal in ideal_by_id.items():
        if ideal >= n:
            continue
        pos = positions[cell_id]
        if state.frozen[pos] or pos == ideal:
            continue
        if state.frozen[ideal] and frozen_variant == "stuck":
            return True
        if state.values[pos] >= state.values[ideal]:
            return True
        if can_swap(state, pos, ideal, frozen_variant):
            return True
    return False


def run_cell_view_bubble(
    state: SimulationState,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    writer: TrajectoryWriter,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int,
) -> tuple[str, int]:
    event_count = 1
    no_move_checks = 0
    frozen_variant = condition["frozenVariant"]
    n = len(state.values)
    scheduler_rng = np.random.default_rng(int(seed_row["schedulerSeed"]))
    tie_rng = np.random.default_rng(int(seed_row["tieBreakerSeed"]))
    best_sorted_pairs = state.sorted_pairs
    last_improvement_round = 0

    while True:
        if is_sorted_state(state):
            return "sorted", event_count
        swaps_before = state.swaps
        order = scheduler_rng.permutation(n)
        positions = pos_by_cell_id(state)
        for cell_id in order:
            pos = positions[int(cell_id)]
            if state.frozen[pos]:
                continue
            direction = 1 if int(tie_rng.integers(0, 2)) == 1 else -1
            target = pos + direction
            if target < 0 or target >= n:
                continue
            state.comparisons += 1
            should_swap = (direction == 1 and state.values[pos] > state.values[target]) or (
                direction == -1 and state.values[pos] < state.values[target]
            )
            if should_swap:
                if can_swap(state, pos, target, frozen_variant):
                    id_a, id_b = swap_positions(state, pos, target)
                    positions[id_a], positions[id_b] = target, pos
                    event_count += 1
                    record_swap_event(writer, condition, seed_row, state)
                else:
                    state.blocked_move_attempts += 1
            cap = reached_cap(state, max_swaps, max_comparisons)
            if cap:
                return cap, event_count
        state.scheduler_rounds += 1
        if state.swaps == swaps_before:
            if legal_bubble_move_exists(state, frozen_variant):
                no_move_checks = 0
            else:
                no_move_checks += 1
        else:
            no_move_checks = 0
        if state.sorted_pairs > best_sorted_pairs:
            best_sorted_pairs = state.sorted_pairs
            last_improvement_round = state.scheduler_rounds
        elif state.scheduler_rounds - last_improvement_round >= STABLE_SORTEDNESS_ROUND_WINDOW:
            return "stable_sortedness_window", event_count
        if no_move_checks >= no_move_checks_required:
            return "no_cell_can_move_after_two_checks", event_count


def run_cell_view_insertion(
    state: SimulationState,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    writer: TrajectoryWriter,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int,
) -> tuple[str, int]:
    stop_reason, event_count = run_traditional_insertion(
        state,
        condition,
        seed_row,
        writer,
        max_swaps,
        max_comparisons,
        no_move_checks_required,
    )
    state.scheduler_rounds = state.passes
    return stop_reason, event_count


def run_cell_view_selection(
    state: SimulationState,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    writer: TrajectoryWriter,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int,
) -> tuple[str, int]:
    event_count = 1
    no_move_checks = 0
    frozen_variant = condition["frozenVariant"]
    n = len(state.values)
    ideal_by_id = {cell_id: 0 for cell_id in range(n)}
    scheduler_rng = np.random.default_rng(int(seed_row["schedulerSeed"]))
    best_sorted_pairs = state.sorted_pairs
    last_improvement_round = 0

    while True:
        if is_sorted_state(state):
            return "sorted", event_count
        swaps_before = state.swaps
        order = scheduler_rng.permutation(n)
        positions = pos_by_cell_id(state)
        for cell_id_value in order:
            cell_id = int(cell_id_value)
            pos = positions[cell_id]
            ideal = ideal_by_id[cell_id]
            if state.frozen[pos] or ideal >= n or pos == ideal:
                continue
            if state.frozen[ideal] and frozen_variant == "stuck":
                ideal_by_id[cell_id] = min(n, ideal + 1)
                state.blocked_move_attempts += 1
                continue
            state.comparisons += 1
            if state.values[pos] >= state.values[ideal]:
                ideal_by_id[cell_id] = min(n, ideal + 1)
            elif can_swap(state, pos, ideal, frozen_variant):
                id_a, id_b = swap_positions(state, pos, ideal)
                positions[id_a], positions[id_b] = ideal, pos
                event_count += 1
                record_swap_event(writer, condition, seed_row, state)
            else:
                state.blocked_move_attempts += 1
            cap = reached_cap(state, max_swaps, max_comparisons)
            if cap:
                return cap, event_count
        state.scheduler_rounds += 1
        if state.swaps == swaps_before:
            if selection_action_exists(state, ideal_by_id, frozen_variant):
                no_move_checks = 0
            else:
                no_move_checks += 1
        else:
            no_move_checks = 0
        if state.sorted_pairs > best_sorted_pairs:
            best_sorted_pairs = state.sorted_pairs
            last_improvement_round = state.scheduler_rounds
        elif state.scheduler_rounds - last_improvement_round >= STABLE_SORTEDNESS_ROUND_WINDOW:
            return "stable_sortedness_window", event_count
        if no_move_checks >= no_move_checks_required:
            return "no_cell_can_move_after_two_checks", event_count


RUNNERS: dict[tuple[str, str], Callable[..., tuple[str, int]]] = {
    ("traditional", "bubble"): run_traditional_bubble,
    ("traditional", "insertion"): run_traditional_insertion,
    ("traditional", "selection"): run_traditional_selection,
    ("cell_view", "bubble"): run_cell_view_bubble,
    ("cell_view", "insertion"): run_cell_view_insertion,
    ("cell_view", "selection"): run_cell_view_selection,
}


def run_one(
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    stop_policies: dict[str, Any],
    writer: TrajectoryWriter,
) -> RunResult:
    start = time.perf_counter()
    n = int(condition["n"])
    frozen_count = int(condition["frozenCount"])
    initial_values = initial_values_from_seed(int(seed_row["inputPermutationSeed"]), n=n)
    frozen_positions = frozen_positions_from_seed(seed_row.get("frozenPositionSeed"), frozen_count, n=n)
    state = init_state(initial_values, frozen_positions)
    emit_trajectory(writer, condition, seed_row, state, 0, "initial")
    stop_policy = stop_policies[condition["stopPolicy"]]
    max_swaps = int(stop_policy["maxSwapEvents"])
    max_comparisons = int(stop_policy["maxComparisonEvents"])
    no_move_checks_required = int(stop_policy.get("noMoveChecksRequired", 2))
    runner = RUNNERS[(condition["implementation"], condition["algorithms"])]
    try:
        stop_reason, event_count = runner(
            state,
            condition,
            seed_row,
            writer,
            max_swaps,
            max_comparisons,
            no_move_checks_required,
        )
    except Exception:
        stop_reason = "error"
        event_count = max(1, state.swaps + 1)
        raise

    return RunResult(
        completed=is_sorted_state(state),
        stop_reason=stop_reason,
        final_values=[int(value) for value in state.values],
        final_frozen_positions=current_frozen_positions(state),
        final_sortedness_raw_count=int(state.sorted_pairs),
        final_sortedness_percent=sortedness_percent_from_raw(state.sorted_pairs, n),
        final_monotonicity_error=monotonicity_error_from_raw(state.sorted_pairs, n),
        swap_count=int(state.swaps),
        comparison_count=int(state.comparisons),
        scheduler_rounds=int(state.scheduler_rounds),
        pass_count=int(state.passes),
        blocked_move_attempts=int(state.blocked_move_attempts),
        event_count=int(event_count),
        wall_time_seconds=time.perf_counter() - start,
    )


def make_replicate_record(condition: dict[str, Any], seed_row: dict[str, Any], result: RunResult) -> dict[str, Any]:
    n = int(condition["n"])
    initial_values = initial_values_from_seed(int(seed_row["inputPermutationSeed"]), n=n)
    initial_frozen_positions = frozen_positions_from_seed(seed_row.get("frozenPositionSeed"), int(condition["frozenCount"]), n=n)
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": condition["conditionId"],
        "implementation": condition["implementation"],
        "algorithm": condition["algorithms"],
        "mixtureId": condition["mixtureId"],
        "inputProfile": condition["inputProfile"],
        "n": n,
        "frozenVariant": condition["frozenVariant"],
        "frozenCount": int(condition["frozenCount"]),
        "replicateIndex": int(seed_row["replicateIndex"]),
        "replicateNumber": int(seed_row["replicateNumber"]),
        "inputPermutationSeed": int(seed_row["inputPermutationSeed"]),
        "frozenPositionSeed": None
        if initial_frozen_positions == []
        else int(seed_row["frozenPositionSeed"]),
        "schedulerSeed": int(seed_row["schedulerSeed"]),
        "tieBreakerSeed": int(seed_row["tieBreakerSeed"]),
        "initialFrozenPositions": json.dumps(initial_frozen_positions, separators=(",", ":")),
        "finalFrozenPositions": json.dumps(result.final_frozen_positions, separators=(",", ":")),
        "initialValuesHash": state_hash(initial_values),
        "finalValuesHash": state_hash(result.final_values),
        "finalValues": json.dumps(result.final_values, separators=(",", ":")),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "finalSortednessRawCount": int(result.final_sortedness_raw_count),
        "finalSortednessPercent": float(result.final_sortedness_percent),
        "finalMonotonicityError": int(result.final_monotonicity_error),
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "swapPlusComparisonSteps": int(result.swap_count + result.comparison_count),
        "schedulerRounds": int(result.scheduler_rounds),
        "passCount": int(result.pass_count),
        "blockedMoveAttempts": int(result.blocked_move_attempts),
        "eventCount": int(result.event_count),
        "wallTimeSeconds": float(result.wall_time_seconds),
    }


def s04_condition_id_for(condition: dict[str, Any]) -> str:
    return f"S04_{condition['implementation']}_{condition['algorithms']}_unique_f0_none"


def load_s04_baseline() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    if not S04_REPLICATE_SUMMARY_DEFAULT.exists() or not S04_TRAJECTORY_EVENTS_DEFAULT.exists():
        raise FileNotFoundError("S07 f=0 baseline requires S04 replicate summary and trajectory events")
    s04_replicates = pd.read_parquet(S04_REPLICATE_SUMMARY_DEFAULT)
    s04_traces = pd.read_parquet(S04_TRAJECTORY_EVENTS_DEFAULT)
    trace_groups = {
        (str(condition_id), int(replicate_index)): indices
        for (condition_id, replicate_index), indices in s04_traces.groupby(
            ["condition_id", "replicate_index"], sort=False
        ).indices.items()
    }
    return s04_replicates, s04_traces, trace_groups


def emit_s04_baseline_trace(
    writer: TrajectoryWriter,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    s04_traces: pd.DataFrame,
    trace_groups: dict[tuple[str, int], np.ndarray],
) -> int:
    s04_condition_id = s04_condition_id_for(condition)
    replicate_index = int(seed_row["replicateIndex"])
    indices = trace_groups[(s04_condition_id, replicate_index)]
    rows = s04_traces.iloc[indices]
    for row in rows.itertuples(index=False):
        writer.write(
            {
                "conditionId": condition["conditionId"],
                "implementation": condition["implementation"],
                "algorithm": condition["algorithms"],
                "frozenVariant": condition["frozenVariant"],
                "frozenCount": int(condition["frozenCount"]),
                "replicateIndex": int(seed_row["replicateIndex"]),
                "replicateNumber": int(seed_row["replicateNumber"]),
                "eventIndex": int(row.event_index),
                "eventKind": row.event_kind,
                "swapCount": int(row.swap_count),
                "comparisonCount": int(row.comparison_count),
                "schedulerRound": 0,
                "passCount": 0,
                "sortednessRawCount": int(row.sortedness_raw_count),
                "sortednessPercent": float(row.sortedness_percent),
                "monotonicityError": int(row.monotonicity_error),
            }
        )
    return int(len(rows))


def make_s04_baseline_record(
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    s04_replicates: pd.DataFrame,
    event_count: int,
) -> dict[str, Any]:
    s04_condition_id = s04_condition_id_for(condition)
    replicate_index = int(seed_row["replicateIndex"])
    matched = s04_replicates[
        (s04_replicates["condition_id"] == s04_condition_id)
        & (s04_replicates["replicate_index"] == replicate_index)
    ]
    if len(matched) != 1:
        raise ValueError(f"Expected one S04 baseline row for {s04_condition_id} replicate {replicate_index}")
    row = matched.iloc[0]
    if int(row["input_permutation_seed"]) != int(seed_row["inputPermutationSeed"]):
        raise ValueError("S04 and S07 f=0 input seeds are not matched")
    final_values = json.loads(row["final_values_json"])
    n = int(row["n"])
    final_error = int(row["final_monotonicity_error"])
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": condition["conditionId"],
        "implementation": condition["implementation"],
        "algorithm": condition["algorithms"],
        "mixtureId": condition["mixtureId"],
        "inputProfile": condition["inputProfile"],
        "n": n,
        "frozenVariant": condition["frozenVariant"],
        "frozenCount": int(condition["frozenCount"]),
        "replicateIndex": replicate_index,
        "replicateNumber": int(seed_row["replicateNumber"]),
        "inputPermutationSeed": int(seed_row["inputPermutationSeed"]),
        "frozenPositionSeed": None,
        "schedulerSeed": int(seed_row["schedulerSeed"]),
        "tieBreakerSeed": int(seed_row["tieBreakerSeed"]),
        "initialFrozenPositions": "[]",
        "finalFrozenPositions": "[]",
        "initialValuesHash": row["initial_state_hash"],
        "finalValuesHash": row["final_state_hash"],
        "finalValues": json.dumps(final_values, separators=(",", ":")),
        "completed": bool(row["completed"]),
        "stopReason": row["stop_reason"],
        "finalSortednessRawCount": int(n - 1 - final_error),
        "finalSortednessPercent": float(row["final_sortedness_percent"]),
        "finalMonotonicityError": final_error,
        "swapCount": int(row["swap_count"]),
        "comparisonCount": int(row["comparison_count"]),
        "swapPlusComparisonSteps": int(row["swap_count"] + row["comparison_count"]),
        "schedulerRounds": int(row["scheduler_rounds"]),
        "passCount": 0,
        "blockedMoveAttempts": 0,
        "eventCount": int(event_count),
        "wallTimeSeconds": float(row["wall_time_seconds"]),
    }


def load_s07_inputs(
    config_path: Path,
    condition_matrix_path: Path,
    seed_table_path: Path,
    replicate_limit: int | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = load_json(config_path)
    condition_matrix = pd.read_csv(condition_matrix_path)
    seed_table = pd.read_csv(seed_table_path)
    conditions = condition_matrix[condition_matrix["producerStep"] == STEP_ID].copy()
    seeds = seed_table[seed_table["producerStep"] == STEP_ID].copy()
    if conditions.empty or seeds.empty:
        raise ValueError("S07 condition matrix and seed rows are required")
    if replicate_limit is not None:
        seeds = seeds[seeds["replicateIndex"] < replicate_limit].copy()
    return config, conditions, seeds


def write_result_tables(
    replicate_df: pd.DataFrame,
    results_dir: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    replicate_parquet = results_dir / "e01_frozen_cell_robustness.parquet"
    replicate_csv = results_dir / "e01_frozen_cell_robustness.csv"
    summary_parquet = results_dir / "e01_frozen_cell_robustness_summary.parquet"
    summary_csv = results_dir / "e01_frozen_cell_robustness_summary.csv"
    backing_parquet = results_dir / "e01_frozen_cell_figure05_backing.parquet"
    backing_csv = results_dir / "e01_frozen_cell_figure05_backing.csv"

    replicate_df.to_parquet(replicate_parquet, index=False)
    replicate_df.to_csv(replicate_csv, index=False)

    group_cols = ["implementation", "algorithm", "frozenVariant", "frozenCount", "conditionId"]
    summary = (
        replicate_df.groupby(group_cols, dropna=False)
        .agg(
            replicateCount=("replicateIndex", "count"),
            completedCount=("completed", "sum"),
            sortedStopCount=("stopReason", lambda values: int((values == "sorted").sum())),
            noMoveStopCount=("stopReason", lambda values: int((values == "no_cell_can_move_after_two_checks").sum())),
            meanFinalMonotonicityError=("finalMonotonicityError", "mean"),
            sdFinalMonotonicityError=("finalMonotonicityError", "std"),
            meanFinalSortednessPercent=("finalSortednessPercent", "mean"),
            sdFinalSortednessPercent=("finalSortednessPercent", "std"),
            meanSwapCount=("swapCount", "mean"),
            meanComparisonCount=("comparisonCount", "mean"),
            meanRunLength=("swapPlusComparisonSteps", "mean"),
            meanSchedulerRounds=("schedulerRounds", "mean"),
            meanBlockedMoveAttempts=("blockedMoveAttempts", "mean"),
            meanWallTimeSeconds=("wallTimeSeconds", "mean"),
        )
        .reset_index()
    )
    summary["semFinalMonotonicityError"] = summary["sdFinalMonotonicityError"] / np.sqrt(summary["replicateCount"])
    summary["completionRate"] = summary["completedCount"] / summary["replicateCount"]
    summary.to_parquet(summary_parquet, index=False)
    summary.to_csv(summary_csv, index=False)

    backing_parts = []
    none_summary = summary[summary["frozenVariant"] == "none"].copy()
    for variant in FIGURE_VARIANTS:
        part = none_summary.copy()
        part["figureFrozenVariant"] = variant
        backing_parts.append(part)
    frozen_summary = summary[summary["frozenVariant"].isin(FIGURE_VARIANTS)].copy()
    frozen_summary["figureFrozenVariant"] = frozen_summary["frozenVariant"]
    backing_parts.append(frozen_summary)
    backing = pd.concat(backing_parts, ignore_index=True)
    backing = backing.sort_values(["figureFrozenVariant", "algorithm", "implementation", "frozenCount"]).reset_index(drop=True)
    backing.to_parquet(backing_parquet, index=False)
    backing.to_csv(backing_csv, index=False)
    return replicate_parquet, replicate_csv, summary_parquet, summary_csv, backing_parquet, backing_csv


def plot_figure05(backing_path: Path, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    backing = pd.read_parquet(backing_path)
    png_path = figure_dir / "figure05_frozen_robustness.png"
    pdf_path = figure_dir / "figure05_frozen_robustness.pdf"
    colors = {"traditional": "#666666", "cell_view": "#1f77b4"}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), sharey=True)
    for row_idx, variant in enumerate(FIGURE_VARIANTS):
        for col_idx, algorithm in enumerate(ALGORITHMS):
            ax = axes[row_idx, col_idx]
            panel = backing[(backing["figureFrozenVariant"] == variant) & (backing["algorithm"] == algorithm)]
            x = np.arange(4)
            width = 0.34
            for offset, implementation in [(-width / 2, "traditional"), (width / 2, "cell_view")]:
                subset = (
                    panel[panel["implementation"] == implementation]
                    .set_index("frozenCount")
                    .reindex([0, 1, 2, 3])
                    .reset_index()
                )
                y = subset["meanFinalMonotonicityError"].to_numpy(dtype=float)
                err = subset["semFinalMonotonicityError"].fillna(0.0).to_numpy(dtype=float)
                ax.bar(
                    x + offset,
                    y,
                    width=width,
                    color=colors[implementation],
                    label=implementation.replace("_", "-"),
                    yerr=err,
                    capsize=2,
                    linewidth=0.7,
                    edgecolor="#222222",
                )
            ax.set_title(f"{variant.capitalize()} {algorithm.capitalize()}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(["0", "1", "2", "3"])
            ax.grid(axis="y", alpha=0.25, linewidth=0.7)
            if row_idx == 1:
                ax.set_xlabel("Frozen Cell count")
            if col_idx == 0:
                ax.set_ylabel("Final monotonicity error")
            if row_idx == 0 and col_idx == 2:
                ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Figure 5-style Frozen Cell robustness (N=100, n=100)", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.96])
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def validate_outputs(
    replicate_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    backing_df: pd.DataFrame,
    conditions: pd.DataFrame,
    seeds: pd.DataFrame,
    trace_writer_row_count: int,
    output_paths: list[Path],
    replicate_limit: int | None,
) -> tuple[bool, list[str], list[str], str]:
    checks: list[str] = []
    caveats: list[str] = []
    failures: list[str] = []

    expected_conditions = 42
    if len(conditions) == expected_conditions:
        checks.append("S07 condition matrix has 42 S03-defined conditions.")
    else:
        failures.append(f"S07 condition matrix has {len(conditions)} conditions, expected {expected_conditions}.")

    expected_replicates = 4200 if replicate_limit is None else len(seeds)
    if len(replicate_df) == expected_replicates:
        checks.append(f"S07 replicate table has {expected_replicates} rows.")
    else:
        failures.append(f"S07 replicate table has {len(replicate_df)} rows, expected {expected_replicates}.")

    per_condition = replicate_df.groupby("conditionId").size()
    if replicate_limit is None and per_condition.nunique() == 1 and int(per_condition.iloc[0]) == 100:
        checks.append("Every S07 condition has exactly 100 replicate rows.")
    elif replicate_limit is not None:
        checks.append(f"Replicate limit smoke mode used {replicate_limit} replicate(s) per condition.")
    else:
        failures.append(f"Per-condition replicate counts are inconsistent: {per_condition.to_dict()}.")

    none_rows = replicate_df[replicate_df["frozenCount"] == 0]
    if (none_rows["frozenVariant"] == "none").all() and (none_rows["initialFrozenPositions"] == "[]").all():
        checks.append("f=0 is represented once as frozenVariant=none with empty frozen-position lists.")
        caveats.append("Following S03, f=0 is reused as the baseline in passive and stuck Figure 5 panels rather than duplicated as separate passive/stuck conditions.")
    else:
        failures.append("f=0 rows are not consistently frozenVariant=none with empty frozen positions.")

    if none_rows["completed"].all() and (none_rows["finalMonotonicityError"] == 0).all():
        checks.append("All f=0 no-Frozen reference runs finish with 100% Sortedness and zero final monotonicity error.")
    else:
        failures.append("At least one f=0 no-Frozen reference run failed to finish with zero final monotonicity error.")

    frozen_rows = replicate_df[replicate_df["frozenCount"] > 0]
    if set(frozen_rows["frozenVariant"].unique()) == {"passive", "stuck"}:
        checks.append("f=1..3 rows include both passive and stuck Frozen Cell variants.")
    else:
        failures.append("f=1..3 rows do not include exactly passive and stuck variants.")

    position_length_ok = True
    for row in replicate_df[["frozenCount", "initialFrozenPositions"]].itertuples(index=False):
        if len(json.loads(row.initialFrozenPositions)) != int(row.frozenCount):
            position_length_ok = False
            break
    if position_length_ok:
        checks.append("Initial frozen-position list lengths match frozenCount for every replicate.")
    else:
        failures.append("At least one initial frozen-position list length does not match frozenCount.")

    final_error_ok = bool(
        (
            replicate_df["finalMonotonicityError"]
            == (replicate_df["n"] - 1 - replicate_df["finalSortednessRawCount"])
        ).all()
    )
    if final_error_ok:
        checks.append("Final monotonicity error equals (n - 1) - final Sortedness raw count.")
    else:
        failures.append("Final monotonicity error formula mismatch.")

    if set(replicate_df["stopReason"].unique()).issubset(STOP_REASONS):
        checks.append("All stop reasons are in the configured stop-reason vocabulary.")
    else:
        failures.append(f"Unexpected stop reasons: {sorted(set(replicate_df['stopReason'].unique()) - STOP_REASONS)}.")

    input_match = (
        replicate_df.groupby(["inputProfile", "replicateIndex"])["inputPermutationSeed"].nunique().max() == 1
    )
    if input_match:
        checks.append("Input permutation seeds remain matched by input profile and replicate index.")
    else:
        failures.append("Input permutation seeds are not matched by input profile and replicate index.")

    frozen_match = (
        frozen_rows.groupby(["inputProfile", "frozenCount", "replicateIndex"])["frozenPositionSeed"].nunique().max()
        == 1
    )
    if frozen_match:
        checks.append("Frozen-position seeds remain matched by input profile, frozen count, and replicate index.")
    else:
        failures.append("Frozen-position seeds are not matched by input profile, frozen count, and replicate index.")

    expected_summary_rows = 42
    if len(summary_df) == expected_summary_rows:
        checks.append("Condition summary has one row per S07 condition.")
    else:
        failures.append(f"Condition summary has {len(summary_df)} rows, expected {expected_summary_rows}.")

    expected_backing_rows = 48
    if len(backing_df) == expected_backing_rows:
        checks.append("Figure 5 backing table expands the f=0 baseline into both passive and stuck panels.")
    else:
        failures.append(f"Figure 5 backing table has {len(backing_df)} rows, expected {expected_backing_rows}.")

    if trace_writer_row_count == int(replicate_df["eventCount"].sum()):
        checks.append("Trajectory row count matches the sum of per-run event counts.")
    else:
        failures.append(
            f"Trajectory row count {trace_writer_row_count} does not match eventCount sum {int(replicate_df['eventCount'].sum())}."
        )

    missing_outputs = [str(path) for path in output_paths if not path.exists() or path.stat().st_size == 0]
    if not missing_outputs:
        checks.append("All declared S07 output files exist and are non-empty.")
    else:
        failures.append(f"Missing or empty output files: {missing_outputs}.")

    if Path("/artifacts/research_steps/S08").exists():
        failures.append("S08 artifact directory exists; S07 must stop before S08.")
    else:
        checks.append("No S08 artifact directory was created.")

    success = not failures and replicate_limit is None
    if failures:
        validation_result = "failed: " + " ".join(failures)
    elif replicate_limit is not None:
        validation_result = "smoke-only: validation checks passed for limited replicate mode, but this is not the full S07 result."
    else:
        validation_result = (
            "passed: S07 produced 4200 replicate rows across 42 S03-defined conditions, "
            "matched input and frozen-position seeds, computed final monotonicity error from final arrays, "
            "wrote Figure 5 backing data/traces, and did not create S08 artifacts."
        )
    return success, checks, failures + caveats, validation_result


def write_methods(
    path: Path,
    source_statuses: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    s02_ambiguities = config.get("s02AmbiguitiesCarriedForward", [])
    lines = [
        "# S07 Frozen Cell Methods",
        "",
        "- Research step ID: S07",
        "- Completion status: completed",
        "- Frozen Cell semantics: passive cells cannot initiate swaps but may be moved by non-frozen actors; stuck cells cannot initiate and cannot be moved by any actor.",
        "- Frozen status is modeled as a cell attribute that travels with the cell during allowed swaps.",
        "- Traditional wrappers treat the algorithm-selected moving element as the actor for passive/stuck checks.",
        "- f=0 no-Frozen reference rows are transformed from the validated S04 no-Frozen replicate and trajectory artifacts so Figure 5 baselines stay aligned with S04.",
        "- f=1..3 Frozen Cell wrappers use deterministic pseudo-scheduling/sweeps rather than OS threads; seeded random choices are retained where the local policy requires direction or order choices.",
        "- f=1..3 cell-view Insertion uses a bounded local insertion actor sweep with the same passive/stuck actor-target checks because the archived prefix-readiness rule can enter long cyclic scans under Frozen Cell perturbations.",
        f"- Cell-view wrappers stop with `stable_sortedness_window` after {STABLE_SORTEDNESS_ROUND_WINDOW} scheduler rounds without a new best Sortedness value, which prevents cyclic Frozen Cell states from running until the high comparison cap.",
        "- The S03 convention represents f=0 once as `frozenVariant=none`; Figure 5-style passive and stuck panels reuse that common baseline.",
        "- Trajectory events record initial state and every successful swap with compact Sortedness and monotonicity-error fields for later S08 DG calculations.",
        "",
        "## S02 Ambiguities Carried Forward",
        "",
    ]
    lines.extend(f"- {item}" for item in s02_ambiguities)
    lines.extend(
        [
            "",
            "## Source Step Statuses",
            "",
        ]
    )
    for step_id, status in source_statuses.items():
        lines.append(
            f"- {step_id}: {status.get('status', 'unknown')} "
            f"({status.get('validationResult', 'no validation result recorded')})"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_markdown(path: Path, checks: list[str], caveats: list[str], validation_result: str) -> None:
    lines = [
        "# S07 Validation",
        "",
        "- Research step ID: S07",
        "- Step number: 7",
        "- Completion status: completed",
        "- Validation result: " + validation_result,
        "- Artifacts written: see `artifact_manifest.json` and `status.json`.",
        "- Caveats or blockers:",
    ]
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Recommended next action: hand control back to the Chief Scientist workflow for review before S08.",
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
) -> None:
    top_rows = []
    for _, row in summary_df.sort_values(
        ["frozenVariant", "algorithm", "implementation", "frozenCount"]
    ).iterrows():
        if row["frozenVariant"] == "none":
            continue
        top_rows.append(
            [
                row["frozenVariant"],
                row["algorithm"],
                row["implementation"],
                int(row["frozenCount"]),
                float(row["meanFinalMonotonicityError"]),
                float(row["meanFinalSortednessPercent"]),
                int(row["noMoveStopCount"]),
            ]
        )
    lines = [
        "# S07 Status Summary",
        "",
        "- Research step ID: S07",
        "- Step number: 7",
        "- Completion status: completed",
        f"- Outcome classification: {outcome_classification}",
        "- Artifacts written:",
    ]
    lines.extend(f"- `{artifact}`" for artifact in artifacts_written)
    lines.extend(
        [
            f"- Validation result: {validation_result}",
            "- Caveats or blockers:",
        ]
    )
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Lay summary: S07 ran the Frozen Cell robustness matrix from the S03 baseline. It produced final error tables, compact swap-event trajectories for S08, and a Figure 5-style plot comparing traditional and cell-view variants under passive and stuck Frozen Cells.",
            f"- Recommended next action: {recommended_next_action}",
            "",
            "## Frozen-Cell Summary",
            "",
            markdown_table(
                [
                    "Variant",
                    "Algorithm",
                    "Implementation",
                    "f",
                    "Mean final error",
                    "Mean Sortedness %",
                    "No-move stops",
                ],
                top_rows,
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_outcome(summary_df: pd.DataFrame) -> tuple[str, str]:
    comparisons = []
    for algorithm in ALGORITHMS:
        for variant in FIGURE_VARIANTS:
            for frozen_count in [1, 2, 3]:
                subset = summary_df[
                    (summary_df["algorithm"] == algorithm)
                    & (summary_df["frozenVariant"] == variant)
                    & (summary_df["frozenCount"] == frozen_count)
                ]
                if len(subset) != 2:
                    continue
                means = subset.set_index("implementation")["meanFinalMonotonicityError"]
                comparisons.append(float(means.get("cell_view", np.nan)) <= float(means.get("traditional", np.nan)))
    if comparisons and all(comparisons):
        return (
            "supportive",
            "Cell-view mean final monotonicity error is no greater than traditional for every S07 passive/stuck f=1..3 algorithm comparison.",
        )
    if comparisons and any(comparisons):
        return (
            "constraining",
            "Cell-view mean final monotonicity error is lower for some but not all S07 passive/stuck f=1..3 comparisons.",
        )
    return (
        "constraining",
        "S07 did not support the frozen robustness direction under the deterministic wrapper semantics.",
    )


def update_run_manifest(
    path: Path,
    status_payload: dict[str, Any],
    artifact_manifest_payload: dict[str, Any],
) -> None:
    manifest = load_json(path)
    if not manifest:
        manifest = {"experimentId": EXPERIMENT_ID, "researchSteps": {}}
    manifest["experimentId"] = EXPERIMENT_ID
    manifest["researchStepId"] = STEP_ID
    manifest["status"] = status_payload["status"]
    manifest["success"] = status_payload["success"]
    manifest["generatedAt"] = status_payload["generatedAt"]
    manifest["git"] = status_payload["git"]
    manifest["artifactsWritten"] = status_payload["artifactsWritten"]
    manifest["validationResult"] = status_payload["validationResult"]
    manifest["caveatsOrBlockers"] = status_payload["caveatsOrBlockers"]
    manifest["recommendedNextAction"] = status_payload["recommendedNextAction"]
    manifest.setdefault("researchSteps", {})
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH_DEFAULT)
    parser.add_argument("--condition-matrix", type=Path, default=CONDITION_MATRIX_DEFAULT)
    parser.add_argument("--seed-table", type=Path, default=SEED_TABLE_DEFAULT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--replicate-limit", type=int, default=None, help="Optional smoke-test replicate limit per condition.")
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    results_dir = artifacts_dir / "results"
    figure_dir = artifacts_dir / "figures" / "e01"
    trace_dir = artifacts_dir / "traces" / "e01" / STEP_ID
    provenance_dir = artifacts_dir / "provenance"
    for directory in [step_dir, code_dir, results_dir, figure_dir, trace_dir, provenance_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now()
    config, conditions_df, seeds_df = load_s07_inputs(
        args.config, args.condition_matrix, args.seed_table, args.replicate_limit
    )
    stop_policies = config["semantics"]["stopPolicies"]
    conditions_by_id = {
        str(row["conditionId"]): row.to_dict()
        for _, row in conditions_df.sort_values("conditionId").iterrows()
    }
    seeds_df = seeds_df.sort_values(["conditionId", "replicateIndex"]).reset_index(drop=True)
    s04_replicates, s04_traces, s04_trace_groups = load_s04_baseline()
    trace_csv = trace_dir / "e01_s07_frozen_cell_trajectory_events.csv.gz"

    replicate_records: list[dict[str, Any]] = []
    with TrajectoryWriter(trace_csv) as trace_writer:
        for _, seed_series in seeds_df.iterrows():
            seed_row = seed_series.to_dict()
            condition = conditions_by_id[str(seed_row["conditionId"])]
            if int(condition["frozenCount"]) == 0:
                event_count = emit_s04_baseline_trace(
                    trace_writer,
                    condition,
                    seed_row,
                    s04_traces,
                    s04_trace_groups,
                )
                replicate_records.append(make_s04_baseline_record(condition, seed_row, s04_replicates, event_count))
            else:
                result = run_one(condition, seed_row, stop_policies, trace_writer)
                replicate_records.append(make_replicate_record(condition, seed_row, result))
        trace_row_count = trace_writer.row_count

    replicate_df = pd.DataFrame(replicate_records)
    (
        replicate_parquet,
        replicate_csv,
        summary_parquet,
        summary_csv,
        backing_parquet,
        backing_csv,
    ) = write_result_tables(replicate_df, results_dir)
    summary_df = pd.read_parquet(summary_parquet)
    backing_df = pd.read_parquet(backing_parquet)
    figure_png, figure_pdf = plot_figure05(backing_parquet, figure_dir)
    outcome_classification, outcome_reason = infer_outcome(summary_df)

    source_statuses = {
        "S02": load_json(S02_STATUS_DEFAULT),
        "S03": load_json(S03_STATUS_DEFAULT),
        "S04": load_json(Path("/artifacts/research_steps/S04/status.json")),
        "S06": load_json(S06_STATUS_DEFAULT),
    }
    methods_md = step_dir / "frozen_methods.md"
    write_methods(methods_md, source_statuses, config)

    validation_json = step_dir / "validation.json"
    validation_md = step_dir / "validation.md"
    summary_md = step_dir / "summary.md"
    status_json = step_dir / "status.json"
    artifact_manifest_json = step_dir / "artifact_manifest.json"

    output_paths = [
        replicate_parquet,
        replicate_csv,
        summary_parquet,
        summary_csv,
        backing_parquet,
        backing_csv,
        trace_csv,
        figure_png,
        figure_pdf,
        methods_md,
    ]
    success, validation_checks, caveats_or_blockers, validation_result = validate_outputs(
        replicate_df,
        summary_df,
        backing_df,
        conditions_df,
        seeds_df,
        trace_row_count,
        output_paths,
        args.replicate_limit,
    )
    caveats_or_blockers.extend(
        [
            "S07 uses deterministic functional wrappers and pseudo-scheduling/sweeps rather than original OS-threaded execution.",
            f"S07 cell-view wrappers add a stable-Sortedness fallback after {STABLE_SORTEDNESS_ROUND_WINDOW} scheduler rounds without improvement to catch cyclic Frozen Cell states before the high comparison cap.",
            "S07 f=0 rows are sourced from validated S04 no-Frozen artifacts and remapped to S07 condition IDs; fresh S07 simulations are limited to f=1..3 passive/stuck Frozen Cell conditions.",
            "S07 f=1..3 cell-view Insertion uses a bounded local insertion actor sweep because the archived prefix-readiness behavior produced long cyclic scans under Frozen Cell perturbations.",
            "Traditional Frozen Cell behavior is reconstructed because S02 found no clean original paper-setting traditional runner.",
            "Comparison counts are run-length diagnostics for S07, not paper efficiency statistics.",
            outcome_reason,
            "No S08 Delayed Gratification calculations or S08 artifact directory were started.",
        ]
    )

    write_validation_markdown(validation_md, validation_checks, caveats_or_blockers, validation_result)
    code_copy = code_dir / Path(__file__).name
    shutil.copy2(Path(__file__), code_copy)
    output_paths.extend([validation_json, validation_md, summary_md, status_json, artifact_manifest_json, code_copy])

    artifacts_written: list[str] = []
    recommended_next_action = (
        "Hand control back to the Chief Scientist workflow for review; S08 should start only after explicit instruction, "
        "using `/artifacts/traces/e01/S07/e01_s07_frozen_cell_trajectory_events.csv.gz` plus the S04 no-Frozen baseline as DG inputs."
    )

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
        "figureBackingRows": int(len(backing_df)),
        "trajectoryRows": int(trace_row_count),
        "stopReasonCounts": dict(Counter(replicate_df["stopReason"])),
        "outcomeClassification": outcome_classification,
        "replicateLimit": args.replicate_limit,
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
    )

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
            "s02Status": str(S02_STATUS_DEFAULT),
            "s03Status": str(S03_STATUS_DEFAULT),
            "s04ReplicateSummary": str(S04_REPLICATE_SUMMARY_DEFAULT),
            "s04TrajectoryEvents": str(S04_TRAJECTORY_EVENTS_DEFAULT),
            "s06Status": str(S06_STATUS_DEFAULT),
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
            "conditionRows": int(len(conditions_df)),
            "replicateRows": int(len(replicate_df)),
            "summaryRows": int(len(summary_df)),
            "figureBackingRows": int(len(backing_df)),
            "trajectoryRows": int(trace_row_count),
        },
        "stopReasonCounts": dict(Counter(replicate_df["stopReason"])),
        "meanFinalMonotonicityErrorByVariant": (
            summary_df.groupby(["frozenVariant", "frozenCount"])["meanFinalMonotonicityError"].mean().reset_index().to_dict("records")
        ),
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
    )
    write_json(status_json, status_payload)

    artifacts = collect_artifacts(output_paths)
    artifact_manifest_payload["artifactCount"] = len(artifacts)
    artifact_manifest_payload["artifacts"] = artifacts
    write_json(artifact_manifest_json, artifact_manifest_payload)

    update_run_manifest(RUN_MANIFEST_DEFAULT, status_payload, artifact_manifest_payload)

    print(json.dumps({"success": success, "validationResult": validation_result, "artifactsWritten": artifacts_written}, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
