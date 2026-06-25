#!/usr/bin/env python3
"""Run E01 S04 Figure 3-style no-Frozen sorting trajectories.

The S04 runner consumes the S03 baseline config, condition matrix, and seed
table. It runs only the six S04 no-Frozen conditions and does not start S05.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
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
from typing import Any, Callable

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
STEP_ID = "S04"
STEP_NUMBER = 4
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
CONFIG_PATH_DEFAULT = Path("/artifacts/configs/e01_baseline_config.json")
CONDITION_MATRIX_DEFAULT = Path("/artifacts/research_steps/S03/condition_matrix.csv")
SEED_TABLE_DEFAULT = Path("/artifacts/research_steps/S03/seed_table.csv")
ALGORITHMS = ["bubble", "insertion", "selection"]
IMPLEMENTATIONS = ["traditional", "cell_view"]
CELL_CLASSES = {
    "bubble": BubbleSortCell,
    "insertion": InsertionSortCell,
    "selection": SelectionSortCell,
}


@dataclass
class RunTrace:
    states: list[list[int]]
    swap_counts: list[int]
    comparison_counts: list[int]
    archived_compare_and_swap_counts: list[int]
    scheduler_rounds: int
    stop_reason: str
    completed: bool
    wall_time_seconds: float
    final_swap_count: int
    final_comparison_count: int
    final_archived_compare_and_swap_count: int


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


def state_hash(state: list[int]) -> str:
    arr = np.asarray(state, dtype=np.int16)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def sortedness_raw(values: list[int]) -> int:
    return sum(1 for i in range(len(values) - 1) if values[i] <= values[i + 1])


def sortedness_percent(values: list[int]) -> float:
    if len(values) < 2:
        return 100.0
    return 100.0 * sortedness_raw(values) / (len(values) - 1)


def monotonicity_error(values: list[int]) -> int:
    return (len(values) - 1) - sortedness_raw(values)


def is_sorted(values: list[int]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def initial_values_from_seed(seed: int, n: int = 100) -> list[int]:
    rng = np.random.default_rng(int(seed))
    values = np.arange(1, n + 1, dtype=np.int16)
    rng.shuffle(values)
    return [int(value) for value in values]


def append_state(
    states: list[list[int]],
    swap_counts: list[int],
    comparison_counts: list[int],
    archived_counts: list[int],
    values: list[int],
    swap_count: int,
    comparison_count: int,
    archived_compare_count: int,
) -> None:
    states.append([int(value) for value in values])
    swap_counts.append(int(swap_count))
    comparison_counts.append(int(comparison_count))
    archived_counts.append(int(archived_compare_count))


def run_traditional_bubble(initial_values: list[int], max_swaps: int) -> RunTrace:
    start = time.perf_counter()
    values = list(initial_values)
    states = [list(values)]
    swap_counts = [0]
    comparison_counts = [0]
    archived_counts = [0]
    comparisons = 0
    swaps = 0
    stop_reason = "sorted"

    while not is_sorted(values):
        found = None
        for i in range(len(values) - 1):
            comparisons += 1
            if values[i] > values[i + 1]:
                found = i
                break
        if found is None:
            break
        i = found
        first_pair_already_compared = True
        while i < len(values) - 1:
            if not first_pair_already_compared:
                comparisons += 1
                if values[i] <= values[i + 1]:
                    break
            first_pair_already_compared = False
            if values[i] <= values[i + 1]:
                break
            values[i], values[i + 1] = values[i + 1], values[i]
            swaps += 1
            append_state(states, swap_counts, comparison_counts, archived_counts, values, swaps, comparisons, 0)
            if swaps >= max_swaps:
                stop_reason = "max_step_cap"
                break
            i += 1
        if stop_reason == "max_step_cap":
            break

    return RunTrace(
        states=states,
        swap_counts=swap_counts,
        comparison_counts=comparison_counts,
        archived_compare_and_swap_counts=archived_counts,
        scheduler_rounds=0,
        stop_reason=stop_reason if is_sorted(values) else stop_reason,
        completed=is_sorted(values),
        wall_time_seconds=time.perf_counter() - start,
        final_swap_count=swaps,
        final_comparison_count=comparisons,
        final_archived_compare_and_swap_count=0,
    )


def run_traditional_insertion(initial_values: list[int], max_swaps: int) -> RunTrace:
    start = time.perf_counter()
    values = list(initial_values)
    states = [list(values)]
    swap_counts = [0]
    comparison_counts = [0]
    archived_counts = [0]
    comparisons = 0
    swaps = 0
    stop_reason = "sorted"

    for i in range(1, len(values)):
        j = i
        while j > 0:
            comparisons += 1
            if values[j] >= values[j - 1]:
                break
            values[j], values[j - 1] = values[j - 1], values[j]
            swaps += 1
            append_state(states, swap_counts, comparison_counts, archived_counts, values, swaps, comparisons, 0)
            if swaps >= max_swaps:
                stop_reason = "max_step_cap"
                break
            j -= 1
        if stop_reason == "max_step_cap":
            break

    return RunTrace(
        states=states,
        swap_counts=swap_counts,
        comparison_counts=comparison_counts,
        archived_compare_and_swap_counts=archived_counts,
        scheduler_rounds=0,
        stop_reason=stop_reason if is_sorted(values) else stop_reason,
        completed=is_sorted(values),
        wall_time_seconds=time.perf_counter() - start,
        final_swap_count=swaps,
        final_comparison_count=comparisons,
        final_archived_compare_and_swap_count=0,
    )


def run_traditional_selection(initial_values: list[int], max_swaps: int) -> RunTrace:
    start = time.perf_counter()
    values = list(initial_values)
    states = [list(values)]
    swap_counts = [0]
    comparison_counts = [0]
    archived_counts = [0]
    comparisons = 0
    swaps = 0
    stop_reason = "sorted"

    for i in range(len(values) - 1):
        min_index = i
        for j in range(i + 1, len(values)):
            comparisons += 1
            if values[j] < values[min_index]:
                min_index = j
        if min_index != i:
            values[i], values[min_index] = values[min_index], values[i]
            swaps += 1
            append_state(states, swap_counts, comparison_counts, archived_counts, values, swaps, comparisons, 0)
            if swaps >= max_swaps:
                stop_reason = "max_step_cap"
                break

    return RunTrace(
        states=states,
        swap_counts=swap_counts,
        comparison_counts=comparison_counts,
        archived_compare_and_swap_counts=archived_counts,
        scheduler_rounds=0,
        stop_reason=stop_reason if is_sorted(values) else stop_reason,
        completed=is_sorted(values),
        wall_time_seconds=time.perf_counter() - start,
        final_swap_count=swaps,
        final_comparison_count=comparisons,
        final_archived_compare_and_swap_count=0,
    )


TRADITIONAL_RUNNERS: dict[str, Callable[[list[int], int], RunTrace]] = {
    "bubble": run_traditional_bubble,
    "insertion": run_traditional_insertion,
    "selection": run_traditional_selection,
}


def values_from_cells(cells: list[Any]) -> list[int]:
    return [int(cell.value) for cell in cells]


def run_cell_view(
    algorithm: str,
    initial_values: list[int],
    scheduler_seed: int,
    tie_breaker_seed: int,
    max_swaps: int,
    max_rounds: int = 100000,
) -> RunTrace:
    start = time.perf_counter()
    lock = threading.Lock()
    status_probe = StatusProbe()
    swapping_count = [0]
    export_steps: list[list[int]] = []
    cells: list[Any] = [None] * len(initial_values)
    cell_class = CELL_CLASSES[algorithm]
    random.seed(int(tie_breaker_seed))
    for i, value in enumerate(initial_values):
        cell = cell_class(
            i,
            int(value),
            lock,
            (i, 0),
            cells,
            (0, 0),
            (len(initial_values) - 1, 0),
            status_probe,
            disable_visualization=True,
            swapping_count=swapping_count,
            export_steps=export_steps,
            label=0,
            reverse_direction=False,
        )
        cells[i] = cell

    group = CellGroup(
        cells,
        cells,
        0,
        (0, 0),
        (len(initial_values) - 1, 0),
        GroupStatus.ACTIVE,
        lock,
        count_down=1,
        phase_period=1,
    )
    cell_objects = list(cells)
    for cell in cell_objects:
        cell.group = group

    states = [list(initial_values)]
    swap_counts = [0]
    comparison_counts = [0]
    archived_counts = [0]
    scheduler_rng = np.random.default_rng(int(scheduler_seed))
    scheduler_rounds = 0
    no_progress_rounds = 0
    stop_reason = "sorted"

    while not is_sorted(values_from_cells(cells)):
        if status_probe.swap_count >= max_swaps:
            stop_reason = "max_step_cap"
            break
        if scheduler_rounds >= max_rounds:
            stop_reason = "max_round_cap"
            break
        before_swaps = status_probe.swap_count
        order = scheduler_rng.permutation(len(cell_objects))
        for cell_index in order:
            before_len = len(status_probe.sorting_steps)
            cell_objects[int(cell_index)].move()
            if len(status_probe.sorting_steps) > before_len:
                for snapshot in status_probe.sorting_steps[before_len:]:
                    append_state(
                        states,
                        swap_counts,
                        comparison_counts,
                        archived_counts,
                        [int(value) for value in snapshot],
                        status_probe.swap_count,
                        status_probe.compare_and_swap_count + status_probe.swap_count,
                        status_probe.compare_and_swap_count,
                    )
                    if status_probe.swap_count >= max_swaps:
                        stop_reason = "max_step_cap"
                        break
            if is_sorted(values_from_cells(cells)) or stop_reason == "max_step_cap":
                break
        scheduler_rounds += 1
        if status_probe.swap_count == before_swaps:
            no_progress_rounds += 1
        else:
            no_progress_rounds = 0
        if no_progress_rounds >= 100 and not is_sorted(values_from_cells(cells)):
            stop_reason = "no_progress_round_cap"
            break

    final_values = values_from_cells(cells)
    completed = is_sorted(final_values)
    if completed:
        stop_reason = "sorted"
    return RunTrace(
        states=states,
        swap_counts=swap_counts,
        comparison_counts=comparison_counts,
        archived_compare_and_swap_counts=archived_counts,
        scheduler_rounds=scheduler_rounds,
        stop_reason=stop_reason,
        completed=completed,
        wall_time_seconds=time.perf_counter() - start,
        final_swap_count=status_probe.swap_count,
        final_comparison_count=status_probe.compare_and_swap_count + status_probe.swap_count,
        final_archived_compare_and_swap_count=status_probe.compare_and_swap_count,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def s04_conditions(condition_matrix_path: Path) -> list[dict[str, str]]:
    rows = [row for row in read_csv_rows(condition_matrix_path) if row["producerStep"] == STEP_ID]
    rows.sort(key=lambda row: (IMPLEMENTATIONS.index(row["implementation"]), ALGORITHMS.index(row["algorithms"])))
    return rows


def s04_seed_rows(seed_table_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows = [row for row in read_csv_rows(seed_table_path) if row["producerStep"] == STEP_ID]
    return {(row["conditionId"], int(row["replicateIndex"])): row for row in rows}


def write_csv_gz(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_event_rows(
    condition: dict[str, str],
    seed_row: dict[str, str],
    replicate_index: int,
    trace: RunTrace,
    initial_hash: str,
) -> list[dict[str, Any]]:
    rows = []
    for event_index, state in enumerate(trace.states):
        rows.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition["conditionId"],
                "implementation": condition["implementation"],
                "algorithm": condition["algorithms"],
                "replicate_index": replicate_index,
                "replicate_number": replicate_index + 1,
                "event_index": event_index,
                "event_kind": "initial" if event_index == 0 else "swap",
                "swap_count": trace.swap_counts[event_index],
                "comparison_count": trace.comparison_counts[event_index],
                "archived_compare_and_swap_count": trace.archived_compare_and_swap_counts[event_index],
                "sortedness_raw_count": sortedness_raw(state),
                "sortedness_percent": sortedness_percent(state),
                "monotonicity_error": monotonicity_error(state),
                "state_hash": state_hash(state),
                "initial_state_hash": initial_hash,
                "input_permutation_seed": int(seed_row["inputPermutationSeed"]),
                "scheduler_seed": int(seed_row["schedulerSeed"]),
                "tie_breaker_seed": int(seed_row["tieBreakerSeed"]),
            }
        )
    return rows


def build_summary_row(
    condition: dict[str, str],
    seed_row: dict[str, str],
    replicate_index: int,
    initial_values: list[int],
    trace: RunTrace,
) -> dict[str, Any]:
    final_state = trace.states[-1]
    return {
        "research_step_id": STEP_ID,
        "condition_id": condition["conditionId"],
        "implementation": condition["implementation"],
        "algorithm": condition["algorithms"],
        "replicate_index": replicate_index,
        "replicate_number": replicate_index + 1,
        "input_profile": condition["inputProfile"],
        "n": len(initial_values),
        "input_permutation_seed": int(seed_row["inputPermutationSeed"]),
        "scheduler_seed": int(seed_row["schedulerSeed"]),
        "tie_breaker_seed": int(seed_row["tieBreakerSeed"]),
        "initial_state_hash": state_hash(initial_values),
        "final_state_hash": state_hash(final_state),
        "initial_sortedness_percent": sortedness_percent(initial_values),
        "final_sortedness_percent": sortedness_percent(final_state),
        "final_monotonicity_error": monotonicity_error(final_state),
        "completed": trace.completed,
        "stop_reason": trace.stop_reason,
        "swap_count": trace.final_swap_count,
        "comparison_count": trace.final_comparison_count,
        "archived_compare_and_swap_count": trace.final_archived_compare_and_swap_count,
        "event_count": len(trace.states),
        "scheduler_rounds": trace.scheduler_rounds,
        "wall_time_seconds": trace.wall_time_seconds,
        "initial_values_json": json.dumps(initial_values, separators=(",", ":")),
        "final_values_json": json.dumps(final_state, separators=(",", ":")),
    }


def condition_summary_rows(replicate_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    grouped = replicate_df.groupby(["condition_id", "implementation", "algorithm"], sort=True)
    for (condition_id, implementation, algorithm), group in grouped:
        rows.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "implementation": implementation,
                "algorithm": algorithm,
                "replicate_count": int(len(group)),
                "completed_count": int(group["completed"].sum()),
                "final_100pct_sortedness_count": int((group["final_sortedness_percent"] == 100.0).sum()),
                "mean_initial_sortedness_percent": float(group["initial_sortedness_percent"].mean()),
                "mean_final_sortedness_percent": float(group["final_sortedness_percent"].mean()),
                "mean_swap_count": float(group["swap_count"].mean()),
                "sd_swap_count": float(group["swap_count"].std(ddof=1)),
                "mean_comparison_count": float(group["comparison_count"].mean()),
                "mean_event_count": float(group["event_count"].mean()),
                "mean_wall_time_seconds": float(group["wall_time_seconds"].mean()),
                "max_wall_time_seconds": float(group["wall_time_seconds"].max()),
            }
        )
    return rows


def normalized_average_trajectory(event_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grid = np.linspace(0.0, 1.0, 101)
    for (condition_id, implementation, algorithm, replicate_index), group in event_df.groupby(
        ["condition_id", "implementation", "algorithm", "replicate_index"],
        sort=True,
    ):
        group = group.sort_values("event_index")
        x = group["event_index"].to_numpy(dtype=float)
        y = group["sortedness_percent"].to_numpy(dtype=float)
        if len(x) == 1 or x[-1] == 0:
            interp = np.repeat(y[-1], len(grid))
        else:
            interp = np.interp(grid, x / x[-1], y)
        for progress, sortedness in zip(grid, interp):
            rows.append(
                {
                    "condition_id": condition_id,
                    "implementation": implementation,
                    "algorithm": algorithm,
                    "replicate_index": int(replicate_index),
                    "progress_fraction": float(progress),
                    "sortedness_percent": float(sortedness),
                }
            )
    interp_df = pd.DataFrame(rows)
    avg = (
        interp_df.groupby(["condition_id", "implementation", "algorithm", "progress_fraction"], as_index=False)
        .agg(
            mean_sortedness_percent=("sortedness_percent", "mean"),
            sd_sortedness_percent=("sortedness_percent", "std"),
            replicate_count=("sortedness_percent", "size"),
        )
        .sort_values(["implementation", "algorithm", "progress_fraction"])
    )
    return avg


def plot_figure03(event_df: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharey=True)
    colors = {"traditional": "#2f6f9f", "cell_view": "#b54b35"}
    titles = {"traditional": "Traditional", "cell_view": "Cell-view"}
    for row_index, algorithm in enumerate(ALGORITHMS):
        for col_index, implementation in enumerate(IMPLEMENTATIONS):
            ax = axes[row_index, col_index]
            subset = event_df[
                (event_df["algorithm"] == algorithm)
                & (event_df["implementation"] == implementation)
            ]
            for _, group in subset.groupby("replicate_index", sort=False):
                group = group.sort_values("event_index")
                ax.plot(
                    group["event_index"],
                    group["sortedness_percent"],
                    color="#9a9a9a",
                    alpha=0.18,
                    linewidth=0.55,
                )
            mean_by_event = (
                subset.groupby("event_index", as_index=False)["sortedness_percent"]
                .mean()
                .sort_values("event_index")
            )
            ax.plot(
                mean_by_event["event_index"],
                mean_by_event["sortedness_percent"],
                color=colors[implementation],
                linewidth=2.0,
            )
            ax.set_title(f"{titles[implementation]} {algorithm.title()}")
            ax.set_ylim(35, 101)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            if col_index == 0:
                ax.set_ylabel("Sortedness (%)")
            if row_index == len(ALGORITHMS) - 1:
                ax.set_xlabel("Swap event index")
    fig.suptitle("E01 S04 Figure 3-style Sortedness trajectories, n=100, N=100", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_pdf)
    plt.close(fig)


def write_npz_states(condition_states: dict[str, dict[int, list[list[int]]]], trace_dir: Path) -> list[str]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for condition_id, replicate_states in sorted(condition_states.items()):
        arrays = {
            f"replicate_{replicate_index:03d}": np.asarray(states, dtype=np.int16)
            for replicate_index, states in sorted(replicate_states.items())
        }
        path = trace_dir / f"{condition_id}_states.npz"
        np.savez_compressed(path, **arrays)
        paths.append(str(path))
    return paths


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def write_notes(path: Path, condition_summary: list[dict[str, Any]], validation: dict[str, Any], artifacts: list[str]) -> None:
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifacts)
    condition_rows = [
        [
            row["implementation"],
            row["algorithm"],
            row["replicate_count"],
            f"{row['mean_initial_sortedness_percent']:.2f}",
            f"{row['mean_final_sortedness_percent']:.2f}",
            f"{row['mean_swap_count']:.2f}",
        ]
        for row in condition_summary
    ]
    path.write_text(
        f"""# S04 Figure 3 Trajectory Notes

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Outcome classification: {OUTCOME_CLASSIFICATION}
- Artifacts written:
{artifact_lines}
- Validation result: {validation["validationResult"]}
- Caveats or blockers: Traditional algorithms use S03-required wrapper implementations; cell-view runs use the S02-mapped cell classes under a seeded pseudo-scheduler to avoid OS-thread nondeterminism; visual styling is a Figure 3-style reproduction, not an exact layout clone.
- Recommended next action: Hand control back; proceed to S05 only after Chief Scientist instruction, using S04 trajectory and summary tables as inputs.

## Run Summary

{markdown_table(["Implementation", "Algorithm", "Replicates", "Mean initial Sortedness", "Mean final Sortedness", "Mean swaps"], condition_rows)}

## Validation

- Final 100% Sortedness rows: {validation["final100Count"]} / {validation["replicateCount"]}
- Completed replicate rows: {validation["completedCount"]} / {validation["replicateCount"]}
- Matched input seed groups: {validation["matchedInputSeedGroups"]}
- Matched initial state groups: {validation["matchedInitialStateGroups"]}
- S05 started: false
""",
        encoding="utf-8",
    )


def write_summary(
    path: Path,
    artifacts: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
) -> None:
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifacts)
    caveat_lines = "\n".join(f"- {caveat}" for caveat in caveats)
    path.write_text(
        f"""# S04 Status Summary

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Outcome classification: {OUTCOME_CLASSIFICATION}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Caveats or blockers:
{caveat_lines}
- Lay summary: The six no-Frozen baseline trajectory conditions now run from matched n=100 inputs for 100 repeats each. Every run reached 100% Sortedness, and the resulting trajectory tables and Figure 3-style plot are ready for S05 efficiency analysis.
- Recommended next action: {recommended_next_action}
""",
        encoding="utf-8",
    )


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


def update_run_manifest(
    artifacts_dir: Path,
    status_payload: dict[str, Any],
    artifact_paths: list[Path],
    skip: bool,
) -> None:
    if skip:
        return
    provenance_dir = artifacts_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = provenance_dir / "run_manifest.json"
    if run_manifest_path.exists():
        try:
            manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest["generatedAt"] = utc_now()
    manifest["researchStepId"] = STEP_ID
    manifest["recommendedNextAction"] = status_payload["recommendedNextAction"]
    manifest["caveatsOrBlockers"] = status_payload["caveatsOrBlockers"]
    manifest["artifactsWritten"] = status_payload["artifactsWritten"]
    manifest["git"] = status_payload["git"]
    manifest["runtime"] = status_payload["runtime"]
    manifest.setdefault("researchSteps", {})
    artifacts = collect_artifacts(artifact_paths)
    manifest["researchSteps"][STEP_ID] = {
        "status": status_payload["status"],
        "success": status_payload["success"],
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "validationResult": status_payload["validationResult"],
        "conditionCount": status_payload["conditionCount"],
        "replicateCount": status_payload["replicateCount"],
        "eventRowCount": status_payload["eventRowCount"],
        "generatedAt": status_payload["generatedAt"],
    }
    run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_s04(args: argparse.Namespace) -> int:
    artifacts_dir = Path(args.artifacts_dir)
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    trace_dir = artifacts_dir / "traces" / "e01" / STEP_ID
    states_dir = trace_dir / "states_npz"
    results_dir = artifacts_dir / "results"
    figure_dir = artifacts_dir / "figures" / "e01"
    for directory in [step_dir, code_dir, trace_dir, states_dir, results_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    conditions = s04_conditions(Path(args.condition_matrix))
    seed_lookup = s04_seed_rows(Path(args.seed_table))
    if len(conditions) != 6:
        raise RuntimeError(f"expected 6 S04 conditions, found {len(conditions)}")

    repeat_count = int(config["paperBaseline"]["repeatCount"])
    if args.replicate_limit is not None:
        repeat_count = min(repeat_count, int(args.replicate_limit))
    max_swaps = int(config["semantics"]["stopPolicies"]["sorted_or_cap"]["maxSwapEvents"])

    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    condition_states: dict[str, dict[int, list[list[int]]]] = defaultdict(dict)

    for condition in conditions:
        algorithm = condition["algorithms"]
        implementation = condition["implementation"]
        condition_id = condition["conditionId"]
        print(f"Running {condition_id} ({repeat_count} replicates)", flush=True)
        for replicate_index in range(repeat_count):
            seed_row = seed_lookup[(condition_id, replicate_index)]
            initial_values = initial_values_from_seed(int(seed_row["inputPermutationSeed"]), int(condition["n"]))
            if implementation == "traditional":
                trace = TRADITIONAL_RUNNERS[algorithm](initial_values, max_swaps)
            elif implementation == "cell_view":
                trace = run_cell_view(
                    algorithm,
                    initial_values,
                    int(seed_row["schedulerSeed"]),
                    int(seed_row["tieBreakerSeed"]),
                    max_swaps,
                )
            else:
                raise RuntimeError(f"unsupported implementation: {implementation}")
            initial_hash = state_hash(initial_values)
            event_rows.extend(build_event_rows(condition, seed_row, replicate_index, trace, initial_hash))
            summary_rows.append(build_summary_row(condition, seed_row, replicate_index, initial_values, trace))
            condition_states[condition_id][replicate_index] = trace.states

    event_df = pd.DataFrame(event_rows)
    summary_df = pd.DataFrame(summary_rows)
    condition_summary = condition_summary_rows(summary_df)
    condition_summary_df = pd.DataFrame(condition_summary)
    average_df = normalized_average_trajectory(event_df)

    trajectory_parquet = trace_dir / "trajectory_events.parquet"
    trajectory_csv_gz = trace_dir / "trajectory_events.csv.gz"
    replicate_summary_parquet = results_dir / "e01_s04_replicate_summary.parquet"
    replicate_summary_csv = results_dir / "e01_s04_replicate_summary.csv"
    condition_summary_csv = results_dir / "e01_s04_condition_summary.csv"
    average_parquet = results_dir / "e01_s04_average_trajectory.parquet"
    average_csv = results_dir / "e01_s04_average_trajectory.csv"
    figure_png = figure_dir / "figure03_sortedness_trajectories.png"
    figure_pdf = figure_dir / "figure03_sortedness_trajectories.pdf"

    event_df.to_parquet(trajectory_parquet, index=False)
    event_df.to_csv(trajectory_csv_gz, index=False, compression="gzip")
    summary_df.to_parquet(replicate_summary_parquet, index=False)
    summary_df.to_csv(replicate_summary_csv, index=False)
    condition_summary_df.to_csv(condition_summary_csv, index=False)
    average_df.to_parquet(average_parquet, index=False)
    average_df.to_csv(average_csv, index=False)
    state_npz_paths = [Path(path) for path in write_npz_states(condition_states, states_dir)]
    plot_figure03(event_df, figure_png, figure_pdf)

    matched_seed_rows = []
    matched_ok = True
    for replicate_index, group in summary_df.groupby("replicate_index"):
        input_seeds = sorted(set(int(value) for value in group["input_permutation_seed"]))
        initial_hashes = sorted(set(str(value) for value in group["initial_state_hash"]))
        row = {
            "replicate_index": int(replicate_index),
            "replicate_number": int(replicate_index) + 1,
            "unique_input_seed_count": len(input_seeds),
            "unique_initial_state_hash_count": len(initial_hashes),
            "input_permutation_seed": input_seeds[0] if input_seeds else "",
            "initial_state_hash": initial_hashes[0] if initial_hashes else "",
            "matched": len(input_seeds) == 1 and len(initial_hashes) == 1,
        }
        matched_seed_rows.append(row)
        matched_ok = matched_ok and bool(row["matched"])

    matched_seed_csv = step_dir / "matched_seed_validation.csv"
    matched_seed_json = step_dir / "matched_seed_validation.json"
    write_csv(
        matched_seed_csv,
        matched_seed_rows,
        [
            "replicate_index",
            "replicate_number",
            "unique_input_seed_count",
            "unique_initial_state_hash_count",
            "input_permutation_seed",
            "initial_state_hash",
            "matched",
        ],
    )
    matched_seed_json.write_text(json.dumps(matched_seed_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    final100 = int((summary_df["final_sortedness_percent"] == 100.0).sum())
    completed = int(summary_df["completed"].sum())
    replicate_count = int(len(summary_df))
    validation_errors = []
    if final100 != replicate_count:
        validation_errors.append(f"final 100% Sortedness failed for {replicate_count - final100} replicates")
    if completed != replicate_count:
        validation_errors.append(f"completion failed for {replicate_count - completed} replicates")
    if not matched_ok:
        validation_errors.append("matched seed or initial-state validation failed")
    if args.replicate_limit is not None:
        validation_errors.append("replicate_limit was used; this is not the full S04 run")
    success = not validation_errors
    validation = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "passed" if success else "failed",
        "validationResult": (
            f"passed: all {replicate_count} no-Frozen runs completed with final 100% Sortedness, and matched input seeds/initial states validated."
            if success
            else "failed: " + "; ".join(validation_errors)
        ),
        "errors": validation_errors,
        "replicateCount": replicate_count,
        "final100Count": final100,
        "completedCount": completed,
        "matchedInputSeedGroups": int(summary_df.groupby(["replicate_index"])["input_permutation_seed"].nunique().eq(1).sum()),
        "matchedInitialStateGroups": int(summary_df.groupby(["replicate_index"])["initial_state_hash"].nunique().eq(1).sum()),
        "conditionCount": int(summary_df["condition_id"].nunique()),
        "eventRowCount": int(len(event_df)),
        "s05Started": False,
    }
    validation_json = step_dir / "validation.json"
    validation_md = step_dir / "validation.md"
    validation_json.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_md.write_text(
        f"""# S04 Validation

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS if success else "failed"}
- Artifacts written: see `status.json` and `artifact_manifest.json`
- Validation result: {validation["validationResult"]}
- Caveats or blockers: validation covers S04 no-Frozen trajectory runs only; S05 efficiency comparisons were not started.
- Recommended next action: hand control back; proceed to S05 only after Chief Scientist instruction.

## Counts

{markdown_table(["Metric", "Value"], [[key, value] for key, value in validation.items() if key not in {"errors"}])}

## Errors

{markdown_table(["Error"], [[error] for error in validation_errors] or [["None"]])}
""",
        encoding="utf-8",
    )

    artifact_paths = [
        trajectory_parquet,
        trajectory_csv_gz,
        replicate_summary_parquet,
        replicate_summary_csv,
        condition_summary_csv,
        average_parquet,
        average_csv,
        figure_png,
        figure_pdf,
        matched_seed_csv,
        matched_seed_json,
        validation_json,
        validation_md,
        step_dir / "figure03_trajectory_notes.md",
        step_dir / "summary.md",
        step_dir / "status.json",
        step_dir / "artifact_manifest.json",
        code_dir / Path(__file__).name,
        *state_npz_paths,
    ]
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    artifacts_written = [str(path) for path in artifact_paths if path.exists() or path.name == "artifact_manifest.json"]

    caveats = [
        "Traditional algorithms are S03 wrapper implementations because S02 found no clean original traditional runners.",
        "Cell-view runs use S02-mapped cell classes with a seeded pseudo-scheduler rather than OS threads to make S03 scheduler seeds meaningful.",
        "Comparison counts are recorded as wrapper/proxy fields for later S05 analysis; S04 validation is based on final Sortedness and seed matching.",
        "Figure styling is Figure 3-style and not an exact reproduction of the paper layout.",
        "No Frozen Cell, Delayed Gratification, Aggregation, statistics, or S05 efficiency sweeps were run in S04.",
    ]
    recommended_next_action = (
        "Hand control back; proceed to S05 only after Chief Scientist instruction, using S04 trajectory tables for efficiency comparisons."
    )
    write_notes(step_dir / "figure03_trajectory_notes.md", condition_summary, validation, artifacts_written)
    write_summary(step_dir / "summary.md", artifacts_written, validation["validationResult"], caveats, recommended_next_action)

    git_meta = get_git_metadata()
    runtime = {
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "osCpuCount": os.cpu_count(),
        "workerCount": 1,
        "gpuUsed": False,
        "matplotlibBackend": matplotlib.get_backend(),
        "numpyVersion": np.__version__,
        "pandasVersion": pd.__version__,
    }
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": STATUS if success else "failed",
        "artifactsWritten": artifacts_written,
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": caveats + validation_errors,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION if success else "constraining",
        "generatedAt": utc_now(),
        "experimentId": EXPERIMENT_ID,
        "conditionCount": int(summary_df["condition_id"].nunique()),
        "replicateCount": replicate_count,
        "eventRowCount": int(len(event_df)),
        "final100Count": final100,
        "completedCount": completed,
        "matchedSeedValidationPassed": matched_ok,
        "s05Started": False,
        "configPath": str(args.config),
        "conditionMatrixPath": str(args.condition_matrix),
        "seedTablePath": str(args.seed_table),
        "git": git_meta,
        "runtime": runtime,
    }
    (step_dir / "status.json").write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    final_artifact_paths = [path for path in artifact_paths if path.exists()]
    artifact_manifest = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "status": STATUS if success else "failed",
        "success": success,
        "validationResult": validation["validationResult"],
        "artifacts": collect_artifacts(final_artifact_paths),
    }
    (step_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_artifact_paths.append(step_dir / "artifact_manifest.json")
    update_run_manifest(artifacts_dir, status_payload, final_artifact_paths, args.skip_manifest_update)

    print(validation["validationResult"])
    print(f"Wrote {len(final_artifact_paths)} S04 artifact files under {artifacts_dir}")
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--config", default=str(CONFIG_PATH_DEFAULT))
    parser.add_argument("--condition-matrix", default=str(CONDITION_MATRIX_DEFAULT))
    parser.add_argument("--seed-table", default=str(SEED_TABLE_DEFAULT))
    parser.add_argument("--replicate-limit", type=int, default=None, help="Optional smoke-test limit; full S04 omits this.")
    parser.add_argument("--skip-manifest-update", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run_s04(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
