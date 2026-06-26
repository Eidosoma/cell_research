#!/usr/bin/env python3
"""Execute E02 S07 randomized local-move null models.

S07 compares real deterministic cell-view trajectories against local stochastic
movement nulls that are matched on accepted swap count. The null policies do
not inspect Algotype policy code; labels are carried by fixed cell identity and
used only for Aggregation diagnostics.
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
    sortedness_raw,
    state_hash,
)


EXPERIMENT_ID = "E02"
STEP_ID = "S07"
STEP_NUMBER = 7
DEFAULT_E01_ARTIFACTS = Path("/previous-artifacts/E01")
DEFAULT_S04_OBSERVED = Path("/artifacts/results/e02_label_shuffle_observed.parquet")
DEFAULT_S04_TRACE = Path("/artifacts/traces/e02/S04/e02_label_shuffle_observed_trajectories.parquet")
POLICY_LOGIC_ID = "DeterministicEventSimulator:S01_policy_semantics"
NULL_MODELS = ["random_adjacent", "inversion_biased", "metropolis_local", "random_walker"]
BASE_POLICIES = ["bubble", "insertion", "selection"]
FROZEN_SOURCE_VARIANTS = [("none", 0), ("passive", 2), ("stuck", 2)]
DIAGNOSTIC_LABEL_MIXTURE = "bubble_insertion_selection"
ALGOTYPE_TO_CODE = {"bubble": 0, "insertion": 1, "selection": 2}


@dataclass
class CommandResult:
    args: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    ok: bool


@dataclass
class SourceRun:
    source_run_id: str
    source_context: str
    source_research_step_id: str
    source_condition_id: str
    mixture_id: str
    algorithm: str
    label_source: str
    n: int
    replicate_index: int
    replicate_number: int
    input_permutation_seed: int
    algotype_assignment_seed: int | None
    scheduler_seed: int
    tie_breaker_seed: int
    frozen_position_seed: int | None
    frozen_variant: str
    frozen_count: int
    initial_frozen_positions: list[int]
    final_frozen_positions: list[int]
    initial_values: list[int]
    initial_policy_algotypes: list[str]
    initial_labels_by_cell_id: list[str]
    final_values: list[int]
    final_labels_by_position: list[str]
    completed: bool
    stop_reason: str
    swap_count: int
    comparison_count: int
    archived_compare_and_swap_count: int
    activation_count: int
    event_count: int
    real_initial_aggregation: float
    real_peak_aggregation: float
    real_final_aggregation: float
    real_auc_aggregation: float
    real_final_sortedness_percent: float
    real_final_monotonicity_error: int
    real_dg_max_drop_percent: float
    real_dg_decrease_count: int
    real_path_curvature_sign_changes: int
    real_sortedness_total_variation: float
    source_trajectory_hash: str
    wall_time_seconds: float


@dataclass
class NullRun:
    source: SourceRun
    null_model: str
    null_replicate_index: int
    null_seed: int
    target_swap_count: int
    max_attempts: int
    completed: bool
    stop_reason: str
    attempt_count: int
    accepted_swap_count: int
    local_move_violations: int
    invalid_frozen_move_count: int
    label_count_preserved: bool
    value_count_preserved: bool
    matched_initial_values: bool
    matched_initial_labels: bool
    initial_values_hash: str
    initial_labels_hash: str
    final_values: list[int]
    final_labels_by_position: list[str]
    final_cell_ids: list[int]
    final_frozen_positions: list[int]
    final_sortedness_percent: float
    final_monotonicity_error: int
    initial_aggregation: float
    peak_aggregation: float
    final_aggregation: float
    auc_aggregation: float
    dg_max_drop_percent: float
    dg_decrease_count: int
    path_curvature_sign_changes: int
    sortedness_total_variation: float
    sortedness_net_gain: float
    cell_move_count_total: int
    cell_move_counts_by_label: dict[str, int]
    trajectory_hash: str
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


def sha256_json(value: Any) -> str:
    payload = json.dumps(json_ready(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(json.dumps(json_ready(parts), separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") % (2**32)


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


def policy_signature() -> str:
    return sha256_path(REPO_ROOT / "e02_deterministic_simulator" / "simulator.py")


def frozen_positions_from_seed(seed: Any, frozen_count: int, n: int) -> list[int]:
    if int(frozen_count) == 0:
        return []
    if seed is None or (isinstance(seed, float) and math.isnan(seed)):
        raise ValueError("frozen_position_seed required when frozen_count > 0")
    rng = np.random.default_rng(int(seed))
    return sorted(int(value) for value in rng.choice(int(n), size=int(frozen_count), replace=False))


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
    if mixture_id in {"pure_bubble", "bubble"}:
        return {"bubble": n}
    if mixture_id in {"pure_insertion", "insertion"}:
        return {"insertion": n}
    if mixture_id in {"pure_selection", "selection"}:
        return {"selection": n}
    raise ValueError(f"unsupported mixture: {mixture_id}")


def labels_from_seed(mixture_id: str, n: int, seed: int) -> list[str]:
    allocation = scaled_allocation(mixture_id, n)
    labels = [label for label, count in allocation.items() for _ in range(count)]
    if len(labels) != int(n):
        raise ValueError(f"allocation for {mixture_id} gives {len(labels)} labels, expected {n}")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(labels)
    return [str(label) for label in labels]


def labels_by_position(cell_ids: list[int], labels_by_cell_id: list[str]) -> list[str]:
    return [labels_by_cell_id[int(cell_id)] for cell_id in cell_ids]


def aggregation_for_cell_ids(cell_ids: list[int], labels_by_cell_id: list[str]) -> float:
    return aggregation(labels_by_position(cell_ids, labels_by_cell_id))


def normalized_auc(curve: np.ndarray | list[float]) -> float:
    arr = np.asarray(curve, dtype=float)
    if len(arr) == 0:
        return float("nan")
    if len(arr) == 1:
        return float(arr[0])
    return float(np.trapezoid(arr, dx=1.0) / (len(arr) - 1))


def trajectory_metrics(sortedness_curve: list[float] | np.ndarray, aggregation_curve: list[float] | np.ndarray) -> dict[str, Any]:
    sortedness = np.asarray(sortedness_curve, dtype=float)
    aggs = np.asarray(aggregation_curve, dtype=float)
    running_best = -math.inf
    max_drop = 0.0
    decrease_count = 0
    previous = None
    for value in sortedness:
        running_best = max(running_best, float(value))
        max_drop = max(max_drop, running_best - float(value))
        if previous is not None and float(value) < previous:
            decrease_count += 1
        previous = float(value)
    deltas = np.diff(sortedness)
    signs = [int(np.sign(delta)) for delta in deltas if not math.isclose(float(delta), 0.0)]
    sign_changes = sum(1 for left, right in zip(signs, signs[1:]) if left != right)
    total_variation = float(np.sum(np.abs(deltas))) if len(deltas) else 0.0
    net_gain = float(sortedness[-1] - sortedness[0]) if len(sortedness) else 0.0
    return {
        "initialAggregation": float(aggs[0]) if len(aggs) else float("nan"),
        "peakAggregation": float(np.max(aggs)) if len(aggs) else float("nan"),
        "finalAggregation": float(aggs[-1]) if len(aggs) else float("nan"),
        "aucAggregation": normalized_auc(aggs),
        "dgMaxDropPercent": float(max_drop),
        "dgDecreaseCount": int(decrease_count),
        "pathCurvatureSignChanges": int(sign_changes),
        "sortednessTotalVariation": float(total_variation),
        "sortednessNetGain": float(net_gain),
    }


def current_cell_ids(sim: DeterministicEventSimulator) -> list[int]:
    return [int(cell.cell_id) for cell in sim.cells]


def current_frozen_positions_from_ids(cell_ids: list[int], frozen_by_cell_id: list[bool]) -> list[int]:
    return [pos for pos, cell_id in enumerate(cell_ids) if frozen_by_cell_id[int(cell_id)]]


def trace_history_hash(trace_rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "eventIndex": int(row["eventIndex"]),
            "eventKind": row["eventKind"],
            "swapCount": int(row["swapCount"]),
            "attemptIndex": None if row.get("attemptIndex") is None else int(row["attemptIndex"]),
            "stateHash": row["stateHash"],
            "cellIds": json.loads(row["cellIds"]),
            "values": json.loads(row["values"]),
            "labels": json.loads(row["labels"]),
            "frozenPositions": json.loads(row["frozenPositions"]),
        }
        for row in trace_rows
    ]
    return sha256_json(payload)


def make_trace_row(
    *,
    source: SourceRun | None,
    source_run_id: str,
    source_context: str,
    source_condition_id: str,
    mixture_id: str,
    frozen_variant: str,
    frozen_count: int,
    null_model: str | None,
    null_replicate_index: int | None,
    event_index: int,
    event_kind: str,
    swap_count: int,
    attempt_index: int | None,
    actor_cell_id: int | None,
    actor_position_before: int | None,
    target_position: int | None,
    cell_ids: list[int],
    values: list[int],
    labels_by_cell_id: list[str],
    frozen_by_cell_id: list[bool],
    local_move_distance: int | None = None,
) -> dict[str, Any]:
    labels = labels_by_position(cell_ids, labels_by_cell_id)
    frozen_positions = current_frozen_positions_from_ids(cell_ids, frozen_by_cell_id)
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "sourceRunId": source_run_id,
        "sourceContext": source_context,
        "sourceConditionId": source_condition_id,
        "mixtureId": mixture_id,
        "frozenVariant": frozen_variant,
        "frozenCount": int(frozen_count),
        "nullModel": "real_policy_source" if null_model is None else null_model,
        "nullReplicateIndex": -1 if null_replicate_index is None else int(null_replicate_index),
        "eventIndex": int(event_index),
        "eventKind": event_kind,
        "swapCount": int(swap_count),
        "attemptIndex": None if attempt_index is None else int(attempt_index),
        "actorCellId": None if actor_cell_id is None else int(actor_cell_id),
        "actorPositionBefore": None if actor_position_before is None else int(actor_position_before),
        "targetPosition": None if target_position is None else int(target_position),
        "localMoveDistance": None if local_move_distance is None else int(local_move_distance),
        "sortednessRawCount": int(sortedness_raw(values)),
        "sortednessPercent": float(sortedness_percent(values)),
        "monotonicityError": int(monotonicity_error(values)),
        "aggregation": float(aggregation(labels)),
        "stateHash": state_hash(values),
        "values": compact_json(values),
        "cellIds": compact_json(cell_ids),
        "labels": compact_json(labels),
        "frozenPositions": compact_json(frozen_positions),
        "sourceTrajectoryHash": "" if source is None else source.source_trajectory_hash,
    }


def run_real_policy_source(
    *,
    source_run_id: str,
    source_context: str,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    n: int,
    initial_values: list[int],
    initial_policy_algotypes: list[str],
    initial_labels_by_cell_id: list[str],
    frozen_positions: list[int],
    frozen_variant: str,
    frozen_count: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[SourceRun, list[dict[str, Any]]]:
    started = time.perf_counter()
    sim = DeterministicEventSimulator(
        initial_values,
        initial_policy_algotypes,
        frozen_positions=frozen_positions,
        frozen_variant=frozen_variant,
        scheduler_seed=int(seed_row["schedulerSeed"]),
        tie_breaker_seed=int(seed_row["tieBreakerSeed"]),
        condition_id=str(condition["conditionId"]),
        research_step_id=STEP_ID,
    )
    frozen_by_cell_id = [False] * n
    for cell_id in sim.frozen_cell_ids():
        frozen_by_cell_id[int(cell_id)] = True
    trace_rows = [
        make_trace_row(
            source=None,
            source_run_id=source_run_id,
            source_context=source_context,
            source_condition_id=str(condition["conditionId"]),
            mixture_id=str(condition["mixtureId"]),
            frozen_variant=frozen_variant,
            frozen_count=frozen_count,
            null_model=None,
            null_replicate_index=None,
            event_index=0,
            event_kind="initial",
            swap_count=0,
            attempt_index=None,
            actor_cell_id=None,
            actor_position_before=None,
            target_position=None,
            cell_ids=current_cell_ids(sim),
            values=sim.current_values(),
            labels_by_cell_id=initial_labels_by_cell_id,
            frozen_by_cell_id=frozen_by_cell_id,
        )
    ]
    stop_reason = "sorted"
    no_move_checks = 0
    interval = max(1, n)
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
                if no_move_checks >= 2:
                    stop_reason = "no_cell_can_move_after_two_checks"
                    break
            else:
                no_move_checks = 0
        outcome = sim.step()
        if outcome.swapped:
            distance = None
            if outcome.actor_position_before is not None and outcome.target_position is not None:
                distance = abs(int(outcome.actor_position_before) - int(outcome.target_position))
            trace_rows.append(
                make_trace_row(
                    source=None,
                    source_run_id=source_run_id,
                    source_context=source_context,
                    source_condition_id=str(condition["conditionId"]),
                    mixture_id=str(condition["mixtureId"]),
                    frozen_variant=frozen_variant,
                    frozen_count=frozen_count,
                    null_model=None,
                    null_replicate_index=None,
                    event_index=len(trace_rows),
                    event_kind="swap",
                    swap_count=sim.swap_count,
                    attempt_index=sim.activation_count,
                    actor_cell_id=outcome.actor_cell_id,
                    actor_position_before=outcome.actor_position_before,
                    target_position=outcome.target_position,
                    cell_ids=current_cell_ids(sim),
                    values=sim.current_values(),
                    labels_by_cell_id=initial_labels_by_cell_id,
                    frozen_by_cell_id=frozen_by_cell_id,
                    local_move_distance=distance,
                )
            )

    sortedness_curve = [float(row["sortednessPercent"]) for row in trace_rows]
    aggregation_curve = [float(row["aggregation"]) for row in trace_rows]
    metrics = trajectory_metrics(sortedness_curve, aggregation_curve)
    history_hash = trace_history_hash(trace_rows)
    final_cell_ids = current_cell_ids(sim)
    source = SourceRun(
        source_run_id=source_run_id,
        source_context=source_context,
        source_research_step_id="S01_simulator_generated_for_S07",
        source_condition_id=str(condition["conditionId"]),
        mixture_id=str(condition["mixtureId"]),
        algorithm=str(condition["algorithms"]),
        label_source="diagnostic_chimeric_like_labels",
        n=n,
        replicate_index=int(seed_row["replicateIndex"]),
        replicate_number=int(seed_row["replicateNumber"]),
        input_permutation_seed=int(seed_row["inputPermutationSeed"]),
        algotype_assignment_seed=None,
        scheduler_seed=int(seed_row["schedulerSeed"]),
        tie_breaker_seed=int(seed_row["tieBreakerSeed"]),
        frozen_position_seed=None if frozen_count == 0 else int(seed_row["frozenPositionSeed"]),
        frozen_variant=frozen_variant,
        frozen_count=int(frozen_count),
        initial_frozen_positions=list(frozen_positions),
        final_frozen_positions=current_frozen_positions_from_ids(final_cell_ids, frozen_by_cell_id),
        initial_values=list(initial_values),
        initial_policy_algotypes=list(initial_policy_algotypes),
        initial_labels_by_cell_id=list(initial_labels_by_cell_id),
        final_values=sim.current_values(),
        final_labels_by_position=labels_by_position(final_cell_ids, initial_labels_by_cell_id),
        completed=sim.is_sorted(),
        stop_reason=stop_reason,
        swap_count=int(sim.swap_count),
        comparison_count=int(sim.comparison_count),
        archived_compare_and_swap_count=int(sim.archived_compare_and_swap_count),
        activation_count=int(sim.activation_count),
        event_count=len(trace_rows),
        real_initial_aggregation=float(metrics["initialAggregation"]),
        real_peak_aggregation=float(metrics["peakAggregation"]),
        real_final_aggregation=float(metrics["finalAggregation"]),
        real_auc_aggregation=float(metrics["aucAggregation"]),
        real_final_sortedness_percent=float(sortedness_percent(sim.current_values())),
        real_final_monotonicity_error=int(monotonicity_error(sim.current_values())),
        real_dg_max_drop_percent=float(metrics["dgMaxDropPercent"]),
        real_dg_decrease_count=int(metrics["dgDecreaseCount"]),
        real_path_curvature_sign_changes=int(metrics["pathCurvatureSignChanges"]),
        real_sortedness_total_variation=float(metrics["sortednessTotalVariation"]),
        source_trajectory_hash=history_hash,
        wall_time_seconds=float(time.perf_counter() - started),
    )
    for row in trace_rows:
        row["sourceTrajectoryHash"] = history_hash
    return source, trace_rows


def source_from_s04_row(row: pd.Series, trace_group: pd.DataFrame) -> SourceRun:
    initial_values = [int(value) for value in json.loads(row["initialValues"])]
    final_values = [int(value) for value in json.loads(row["finalValues"])]
    labels = [str(value) for value in json.loads(row["initialAlgotypes"])]
    final_labels = [str(value) for value in json.loads(row["finalAlgotypes"])]
    sorted_trace = trace_group.sort_values("event_index")
    sortedness_curve = sorted_trace["sortedness_percent"].astype(float).to_numpy()
    aggregation_col = "observed_aggregation" if "observed_aggregation" in sorted_trace.columns else "observedAggregation"
    aggregation_curve = sorted_trace[aggregation_col].astype(float).to_numpy()
    metrics = trajectory_metrics(sortedness_curve, aggregation_curve)
    source_run_id = (
        f"S04_{row['mixtureId']}_rep{int(row['replicateIndex']):03d}"
    )
    return SourceRun(
        source_run_id=source_run_id,
        source_context="same_goal_chimera",
        source_research_step_id="S04",
        source_condition_id=str(row["conditionId"]),
        mixture_id=str(row["mixtureId"]),
        algorithm=str(row["mixtureId"]).replace("_", "+"),
        label_source="policy_algotypes",
        n=int(row["n"]),
        replicate_index=int(row["replicateIndex"]),
        replicate_number=int(row["replicateNumber"]),
        input_permutation_seed=int(row["inputPermutationSeed"]),
        algotype_assignment_seed=int(row["algotypeAssignmentSeed"]),
        scheduler_seed=int(row["schedulerSeed"]),
        tie_breaker_seed=int(row["tieBreakerSeed"]),
        frozen_position_seed=None,
        frozen_variant="none",
        frozen_count=0,
        initial_frozen_positions=[],
        final_frozen_positions=[],
        initial_values=initial_values,
        initial_policy_algotypes=labels,
        initial_labels_by_cell_id=labels,
        final_values=final_values,
        final_labels_by_position=final_labels,
        completed=bool(row["completed"]),
        stop_reason=str(row["stopReason"]),
        swap_count=int(row["swapCount"]),
        comparison_count=int(row["comparisonCount"]),
        archived_compare_and_swap_count=int(row["archivedCompareAndSwapCount"]),
        activation_count=int(row["activationCount"]),
        event_count=int(row["eventCount"]),
        real_initial_aggregation=float(metrics["initialAggregation"]),
        real_peak_aggregation=float(metrics["peakAggregation"]),
        real_final_aggregation=float(metrics["finalAggregation"]),
        real_auc_aggregation=float(metrics["aucAggregation"]),
        real_final_sortedness_percent=float(row["finalSortednessPercent"]),
        real_final_monotonicity_error=int(row["finalMonotonicityError"]),
        real_dg_max_drop_percent=float(
            row["dgMaxDropPercent"] if "dgMaxDropPercent" in row and not pd.isna(row["dgMaxDropPercent"]) else metrics["dgMaxDropPercent"]
        ),
        real_dg_decrease_count=int(
            row["dgDecreaseCount"] if "dgDecreaseCount" in row and not pd.isna(row["dgDecreaseCount"]) else metrics["dgDecreaseCount"]
        ),
        real_path_curvature_sign_changes=int(metrics["pathCurvatureSignChanges"]),
        real_sortedness_total_variation=float(metrics["sortednessTotalVariation"]),
        source_trajectory_hash=str(row["trajectoryHistoryHash"]),
        wall_time_seconds=float(row["wallTimeSeconds"]),
    )


def load_chimeric_sources(s04_observed_path: Path, s04_trace_path: Path) -> tuple[list[SourceRun], list[dict[str, Any]]]:
    observed = pd.read_parquet(s04_observed_path).sort_values(["mixtureId", "replicateIndex"]).reset_index(drop=True)
    traces = pd.read_parquet(s04_trace_path)
    sources: list[SourceRun] = []
    source_trace_rows: list[dict[str, Any]] = []
    grouped = traces.groupby(["condition_id", "mixtureId", "replicateIndex"], sort=False)
    for _, row in observed.iterrows():
        key = (row["conditionId"], row["mixtureId"], int(row["replicateIndex"]))
        trace_group = grouped.get_group(key)
        source = source_from_s04_row(row, trace_group)
        sources.append(source)
        for _, trace_row in trace_group.sort_values("event_index").iterrows():
            values = [int(value) for value in json.loads(trace_row["values_json"])]
            cell_ids = [int(value) for value in json.loads(trace_row["cell_ids_json"])]
            labels = labels_by_position(cell_ids, source.initial_labels_by_cell_id)
            source_trace_rows.append(
                {
                    "experimentId": EXPERIMENT_ID,
                    "researchStepId": STEP_ID,
                    "sourceRunId": source.source_run_id,
                    "sourceContext": source.source_context,
                    "sourceConditionId": source.source_condition_id,
                    "mixtureId": source.mixture_id,
                    "frozenVariant": source.frozen_variant,
                    "frozenCount": source.frozen_count,
                    "nullModel": "real_policy_source",
                    "nullReplicateIndex": -1,
                    "eventIndex": int(trace_row["event_index"]),
                    "eventKind": str(trace_row["event_kind"]),
                    "swapCount": int(trace_row["swap_count"]),
                    "attemptIndex": int(trace_row["activation_index"]),
                    "actorCellId": None if pd.isna(trace_row["actor_cell_id"]) else int(trace_row["actor_cell_id"]),
                    "actorPositionBefore": None,
                    "targetPosition": None if pd.isna(trace_row["target_position"]) else int(trace_row["target_position"]),
                    "localMoveDistance": 1 if str(trace_row["event_kind"]) == "swap" else None,
                    "sortednessRawCount": int(trace_row["sortedness_raw_count"]),
                    "sortednessPercent": float(trace_row["sortedness_percent"]),
                    "monotonicityError": int(trace_row["monotonicity_error"]),
                    "aggregation": float(aggregation(labels)),
                    "stateHash": str(trace_row["state_hash"]),
                    "values": compact_json(values),
                    "cellIds": compact_json(cell_ids),
                    "labels": compact_json(labels),
                    "frozenPositions": "[]",
                    "sourceTrajectoryHash": source.source_trajectory_hash,
                }
            )
    return sources, source_trace_rows


def load_frozen_sources(
    *,
    e01_artifacts: Path,
    n: int,
    replicate_count: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[list[SourceRun], list[dict[str, Any]]]:
    condition_matrix = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "condition_matrix.csv")
    seed_table = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "seed_table.csv")
    condition_ids: list[str] = []
    for policy in BASE_POLICIES:
        for variant, count in FROZEN_SOURCE_VARIANTS:
            condition_ids.append(f"S07_cell_view_{policy}_unique_f{count}_{variant}")
    conditions = (
        condition_matrix[
            (condition_matrix["conditionId"].isin(condition_ids))
            & (condition_matrix["implementation"] == "cell_view")
        ]
        .sort_values(["algorithms", "frozenVariant", "frozenCount"])
        .reset_index(drop=True)
    )
    missing = sorted(set(condition_ids) - set(conditions["conditionId"]))
    if missing:
        raise ValueError(f"Missing frozen source conditions: {missing}")
    seeds = seed_table[
        (seed_table["conditionId"].isin(condition_ids)) & (seed_table["replicateIndex"] < int(replicate_count))
    ].copy()
    seeds = seeds.sort_values(["conditionId", "replicateIndex"]).reset_index(drop=True)
    sources: list[SourceRun] = []
    traces: list[dict[str, Any]] = []
    conditions_by_id = {str(row["conditionId"]): dict(row) for _, row in conditions.iterrows()}
    for _, seed_row_series in seeds.iterrows():
        seed_row = {str(key): seed_row_series[key] for key in seed_row_series.index}
        condition = conditions_by_id[str(seed_row["conditionId"])]
        policy = str(condition["algorithms"])
        frozen_variant = str(condition["frozenVariant"])
        frozen_count = int(condition["frozenCount"])
        initial_values = initial_values_from_seed(int(seed_row["inputPermutationSeed"]), n=n, profile="unique_1_100")
        initial_policy_algotypes = [policy] * n
        label_seed = stable_seed("S07_frozen_diagnostic_labels", condition["conditionId"], int(seed_row["replicateIndex"]))
        labels = labels_from_seed(DIAGNOSTIC_LABEL_MIXTURE, n=n, seed=label_seed)
        frozen_seed = None if frozen_count == 0 else int(seed_row["frozenPositionSeed"])
        frozen_positions = frozen_positions_from_seed(frozen_seed, frozen_count, n=n)
        source_run_id = f"S07_frozen_{policy}_{frozen_variant}_f{frozen_count}_rep{int(seed_row['replicateIndex']):03d}"
        source, trace_rows = run_real_policy_source(
            source_run_id=source_run_id,
            source_context="frozen_cell",
            condition=condition,
            seed_row=seed_row,
            n=n,
            initial_values=initial_values,
            initial_policy_algotypes=initial_policy_algotypes,
            initial_labels_by_cell_id=labels,
            frozen_positions=frozen_positions,
            frozen_variant=frozen_variant,
            frozen_count=frozen_count,
            max_activations=max_activations,
            max_swaps=max_swaps,
            max_comparisons=max_comparisons,
        )
        sources.append(source)
        traces.extend(trace_rows)
    return sources, traces


def pair_is_valid(cell_ids: list[int], frozen_by_cell_id: list[bool], pos: int, frozen_variant: str) -> bool:
    if pos < 0 or pos >= len(cell_ids) - 1:
        return False
    if frozen_variant == "none":
        return True
    left_frozen = bool(frozen_by_cell_id[int(cell_ids[pos])])
    right_frozen = bool(frozen_by_cell_id[int(cell_ids[pos + 1])])
    if frozen_variant == "passive":
        return not (left_frozen and right_frozen)
    if frozen_variant == "stuck":
        return not left_frozen and not right_frozen
    raise ValueError(f"unknown frozen variant: {frozen_variant}")


def actor_for_pair(
    cell_ids: list[int],
    frozen_by_cell_id: list[bool],
    pos: int,
    frozen_variant: str,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    if not pair_is_valid(cell_ids, frozen_by_cell_id, pos, frozen_variant):
        return None
    left_frozen = bool(frozen_by_cell_id[int(cell_ids[pos])])
    right_frozen = bool(frozen_by_cell_id[int(cell_ids[pos + 1])])
    if frozen_variant == "passive":
        if left_frozen and not right_frozen:
            return pos + 1, pos
        if right_frozen and not left_frozen:
            return pos, pos + 1
    actor_pos = pos if rng.random() < 0.5 else pos + 1
    target_pos = pos + 1 if actor_pos == pos else pos
    return actor_pos, target_pos


def valid_adjacent_pairs(cell_ids: list[int], frozen_by_cell_id: list[bool], frozen_variant: str) -> list[int]:
    return [pos for pos in range(len(cell_ids) - 1) if pair_is_valid(cell_ids, frozen_by_cell_id, pos, frozen_variant)]


def propose_pair(
    *,
    null_model: str,
    cell_ids: list[int],
    values_by_cell_id: list[int],
    frozen_by_cell_id: list[bool],
    frozen_variant: str,
    rng: np.random.Generator,
) -> tuple[int, int, bool]:
    values = [values_by_cell_id[int(cell_id)] for cell_id in cell_ids]
    if null_model == "random_walker":
        movable_positions = [
            pos for pos, cell_id in enumerate(cell_ids) if not frozen_by_cell_id[int(cell_id)]
        ]
        if not movable_positions:
            return -1, -1, False
        actor_pos = int(rng.choice(movable_positions))
        target_pos = actor_pos + (1 if rng.random() < 0.5 else -1)
        if target_pos < 0 or target_pos >= len(cell_ids):
            return actor_pos, target_pos, False
        left = min(actor_pos, target_pos)
        if not pair_is_valid(cell_ids, frozen_by_cell_id, left, frozen_variant):
            return actor_pos, target_pos, False
        return left, actor_pos, True

    pairs = valid_adjacent_pairs(cell_ids, frozen_by_cell_id, frozen_variant)
    if not pairs:
        return -1, -1, False
    if null_model == "inversion_biased":
        inverted = [pos for pos in pairs if values[pos] > values[pos + 1]]
        pos = int(rng.choice(inverted if inverted else pairs))
    else:
        pos = int(rng.choice(pairs))
    actor_target = actor_for_pair(cell_ids, frozen_by_cell_id, pos, frozen_variant, rng)
    if actor_target is None:
        return pos, -1, False
    actor_pos, _target_pos = actor_target
    return pos, actor_pos, True


def accepted_by_metropolis(
    values_before: list[int],
    pos: int,
    rng: np.random.Generator,
    temperature: float,
    min_acceptance: float,
) -> bool:
    before = sortedness_raw(values_before)
    values_after = list(values_before)
    values_after[pos], values_after[pos + 1] = values_after[pos + 1], values_after[pos]
    after = sortedness_raw(values_after)
    delta = after - before
    if delta >= 0:
        return True
    probability = max(float(min_acceptance), math.exp(float(delta) / max(float(temperature), 1.0e-9)))
    return bool(rng.random() < probability)


def run_local_move_null(
    *,
    source: SourceRun,
    null_model: str,
    null_replicate_index: int,
    null_seed: int,
    attempt_multiplier: int,
    metropolis_temperature: float,
    metropolis_min_acceptance: float,
) -> NullRun:
    rng = np.random.default_rng(int(null_seed))
    n = source.n
    values_by_cell_id = list(source.initial_values)
    labels_by_cell_id = list(source.initial_labels_by_cell_id)
    cell_ids = list(range(n))
    frozen_by_cell_id = [False] * n
    for pos in source.initial_frozen_positions:
        frozen_by_cell_id[int(pos)] = True
    target_swaps = int(source.swap_count)
    max_attempts = max(1000, target_swaps * int(attempt_multiplier) + n)
    initial_values_hash = state_hash([values_by_cell_id[int(cell_id)] for cell_id in cell_ids])
    initial_labels_hash = sha256_json(labels_by_position(cell_ids, labels_by_cell_id))
    trace_rows = [
        make_trace_row(
            source=source,
            source_run_id=source.source_run_id,
            source_context=source.source_context,
            source_condition_id=source.source_condition_id,
            mixture_id=source.mixture_id,
            frozen_variant=source.frozen_variant,
            frozen_count=source.frozen_count,
            null_model=null_model,
            null_replicate_index=null_replicate_index,
            event_index=0,
            event_kind="initial",
            swap_count=0,
            attempt_index=0,
            actor_cell_id=None,
            actor_position_before=None,
            target_position=None,
            cell_ids=cell_ids,
            values=[values_by_cell_id[int(cell_id)] for cell_id in cell_ids],
            labels_by_cell_id=labels_by_cell_id,
            frozen_by_cell_id=frozen_by_cell_id,
        )
    ]
    attempts = 0
    accepted_swaps = 0
    local_move_violations = 0
    invalid_frozen_moves = 0
    cell_move_counts_by_label: Counter[str] = Counter()
    stop_reason = "target_swap_budget_reached"
    if target_swaps == 0:
        stop_reason = "zero_target_swaps"
    while accepted_swaps < target_swaps:
        attempts += 1
        if attempts > max_attempts:
            stop_reason = "attempt_cap"
            break
        pos, actor_pos, proposed = propose_pair(
            null_model=null_model,
            cell_ids=cell_ids,
            values_by_cell_id=values_by_cell_id,
            frozen_by_cell_id=frozen_by_cell_id,
            frozen_variant=source.frozen_variant,
            rng=rng,
        )
        if not proposed:
            if not valid_adjacent_pairs(cell_ids, frozen_by_cell_id, source.frozen_variant):
                stop_reason = "no_valid_local_move"
                break
            continue
        if pos < 0 or pos >= n - 1:
            local_move_violations += 1
            continue
        target_pos = pos + 1 if int(actor_pos) == pos else pos
        if abs(pos - (pos + 1)) != 1:
            local_move_violations += 1
            continue
        actor_cell_id = int(cell_ids[int(actor_pos)])
        if source.frozen_variant != "none" and frozen_by_cell_id[actor_cell_id]:
            invalid_frozen_moves += 1
            continue
        if source.frozen_variant == "stuck" and (
            frozen_by_cell_id[int(cell_ids[pos])] or frozen_by_cell_id[int(cell_ids[pos + 1])]
        ):
            invalid_frozen_moves += 1
            continue
        if null_model == "metropolis_local":
            current_values = [values_by_cell_id[int(cell_id)] for cell_id in cell_ids]
            if not accepted_by_metropolis(
                current_values,
                pos,
                rng,
                temperature=metropolis_temperature,
                min_acceptance=metropolis_min_acceptance,
            ):
                continue
        left_cell = int(cell_ids[pos])
        right_cell = int(cell_ids[pos + 1])
        cell_ids[pos], cell_ids[pos + 1] = cell_ids[pos + 1], cell_ids[pos]
        accepted_swaps += 1
        cell_move_counts_by_label[labels_by_cell_id[left_cell]] += 1
        cell_move_counts_by_label[labels_by_cell_id[right_cell]] += 1
        values = [values_by_cell_id[int(cell_id)] for cell_id in cell_ids]
        trace_rows.append(
            make_trace_row(
                source=source,
                source_run_id=source.source_run_id,
                source_context=source.source_context,
                source_condition_id=source.source_condition_id,
                mixture_id=source.mixture_id,
                frozen_variant=source.frozen_variant,
                frozen_count=source.frozen_count,
                null_model=null_model,
                null_replicate_index=null_replicate_index,
                event_index=len(trace_rows),
                event_kind="swap",
                swap_count=accepted_swaps,
                attempt_index=attempts,
                actor_cell_id=actor_cell_id,
                actor_position_before=int(actor_pos),
                target_position=int(target_pos),
                cell_ids=cell_ids,
                values=values,
                labels_by_cell_id=labels_by_cell_id,
                frozen_by_cell_id=frozen_by_cell_id,
                local_move_distance=1,
            )
        )
    final_values = [values_by_cell_id[int(cell_id)] for cell_id in cell_ids]
    final_labels = labels_by_position(cell_ids, labels_by_cell_id)
    sortedness_curve = [float(row["sortednessPercent"]) for row in trace_rows]
    aggregation_curve = [float(row["aggregation"]) for row in trace_rows]
    metrics = trajectory_metrics(sortedness_curve, aggregation_curve)
    return NullRun(
        source=source,
        null_model=null_model,
        null_replicate_index=int(null_replicate_index),
        null_seed=int(null_seed),
        target_swap_count=target_swaps,
        max_attempts=int(max_attempts),
        completed=accepted_swaps == target_swaps,
        stop_reason=stop_reason,
        attempt_count=int(attempts),
        accepted_swap_count=int(accepted_swaps),
        local_move_violations=int(local_move_violations),
        invalid_frozen_move_count=int(invalid_frozen_moves),
        label_count_preserved=Counter(final_labels) == Counter(labels_by_cell_id),
        value_count_preserved=Counter(final_values) == Counter(values_by_cell_id),
        matched_initial_values=initial_values_hash == state_hash(source.initial_values),
        matched_initial_labels=initial_labels_hash == sha256_json(source.initial_labels_by_cell_id),
        initial_values_hash=initial_values_hash,
        initial_labels_hash=initial_labels_hash,
        final_values=final_values,
        final_labels_by_position=final_labels,
        final_cell_ids=list(cell_ids),
        final_frozen_positions=current_frozen_positions_from_ids(cell_ids, frozen_by_cell_id),
        final_sortedness_percent=float(sortedness_percent(final_values)),
        final_monotonicity_error=int(monotonicity_error(final_values)),
        initial_aggregation=float(metrics["initialAggregation"]),
        peak_aggregation=float(metrics["peakAggregation"]),
        final_aggregation=float(metrics["finalAggregation"]),
        auc_aggregation=float(metrics["aucAggregation"]),
        dg_max_drop_percent=float(metrics["dgMaxDropPercent"]),
        dg_decrease_count=int(metrics["dgDecreaseCount"]),
        path_curvature_sign_changes=int(metrics["pathCurvatureSignChanges"]),
        sortedness_total_variation=float(metrics["sortednessTotalVariation"]),
        sortedness_net_gain=float(metrics["sortednessNetGain"]),
        cell_move_count_total=int(accepted_swaps * 2),
        cell_move_counts_by_label=dict(sorted(cell_move_counts_by_label.items())),
        trajectory_hash=trace_history_hash(trace_rows),
        trace_rows=trace_rows,
    )


def source_record(source: SourceRun) -> dict[str, Any]:
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "sourceRunId": source.source_run_id,
        "sourceContext": source.source_context,
        "sourceResearchStepId": source.source_research_step_id,
        "sourceConditionId": source.source_condition_id,
        "mixtureId": source.mixture_id,
        "algorithm": source.algorithm,
        "labelSource": source.label_source,
        "policyLogicId": POLICY_LOGIC_ID,
        "policySignatureSha256": policy_signature(),
        "n": int(source.n),
        "replicateIndex": int(source.replicate_index),
        "replicateNumber": int(source.replicate_number),
        "inputPermutationSeed": int(source.input_permutation_seed),
        "algotypeAssignmentSeed": source.algotype_assignment_seed,
        "schedulerSeed": int(source.scheduler_seed),
        "tieBreakerSeed": int(source.tie_breaker_seed),
        "frozenPositionSeed": source.frozen_position_seed,
        "frozenVariant": source.frozen_variant,
        "frozenCount": int(source.frozen_count),
        "initialFrozenPositions": compact_json(source.initial_frozen_positions),
        "finalFrozenPositions": compact_json(source.final_frozen_positions),
        "initialValuesHash": state_hash(source.initial_values),
        "initialLabelsHash": sha256_json(source.initial_labels_by_cell_id),
        "finalValuesHash": state_hash(source.final_values),
        "initialValues": compact_json(source.initial_values),
        "initialPolicyAlgotypes": compact_json(source.initial_policy_algotypes),
        "initialLabelsByCellId": compact_json(source.initial_labels_by_cell_id),
        "finalValues": compact_json(source.final_values),
        "finalLabelsByPosition": compact_json(source.final_labels_by_position),
        "completed": bool(source.completed),
        "stopReason": source.stop_reason,
        "swapCount": int(source.swap_count),
        "comparisonCount": int(source.comparison_count),
        "archivedCompareAndSwapCount": int(source.archived_compare_and_swap_count),
        "activationCount": int(source.activation_count),
        "eventCount": int(source.event_count),
        "realInitialAggregation": float(source.real_initial_aggregation),
        "realPeakAggregation": float(source.real_peak_aggregation),
        "realFinalAggregation": float(source.real_final_aggregation),
        "realAucAggregation": float(source.real_auc_aggregation),
        "realFinalSortednessPercent": float(source.real_final_sortedness_percent),
        "realFinalMonotonicityError": int(source.real_final_monotonicity_error),
        "realDgMaxDropPercent": float(source.real_dg_max_drop_percent),
        "realDgDecreaseCount": int(source.real_dg_decrease_count),
        "realPathCurvatureSignChanges": int(source.real_path_curvature_sign_changes),
        "realSortednessTotalVariation": float(source.real_sortedness_total_variation),
        "sourceTrajectoryHash": source.source_trajectory_hash,
        "wallTimeSeconds": float(source.wall_time_seconds),
    }


def null_record(run: NullRun) -> dict[str, Any]:
    source = run.source
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "sourceRunId": source.source_run_id,
        "sourceContext": source.source_context,
        "sourceResearchStepId": source.source_research_step_id,
        "sourceConditionId": source.source_condition_id,
        "mixtureId": source.mixture_id,
        "algorithm": source.algorithm,
        "labelSource": source.label_source,
        "nullModel": run.null_model,
        "nullReplicateIndex": int(run.null_replicate_index),
        "nullSeed": int(run.null_seed),
        "n": int(source.n),
        "replicateIndex": int(source.replicate_index),
        "replicateNumber": int(source.replicate_number),
        "inputPermutationSeed": int(source.input_permutation_seed),
        "algotypeAssignmentSeed": source.algotype_assignment_seed,
        "schedulerSeed": int(source.scheduler_seed),
        "tieBreakerSeed": int(source.tie_breaker_seed),
        "frozenPositionSeed": source.frozen_position_seed,
        "frozenVariant": source.frozen_variant,
        "frozenCount": int(source.frozen_count),
        "targetSwapCount": int(run.target_swap_count),
        "acceptedSwapCount": int(run.accepted_swap_count),
        "swapCountMatched": bool(run.accepted_swap_count == run.target_swap_count),
        "attemptCount": int(run.attempt_count),
        "maxAttempts": int(run.max_attempts),
        "attemptCapHit": bool(run.stop_reason == "attempt_cap"),
        "completed": bool(run.completed),
        "stopReason": run.stop_reason,
        "localMoveViolations": int(run.local_move_violations),
        "invalidFrozenMoveCount": int(run.invalid_frozen_move_count),
        "matchedInitialValues": bool(run.matched_initial_values),
        "matchedInitialLabels": bool(run.matched_initial_labels),
        "labelCountPreserved": bool(run.label_count_preserved),
        "valueCountPreserved": bool(run.value_count_preserved),
        "initialValuesHash": run.initial_values_hash,
        "initialLabelsHash": run.initial_labels_hash,
        "sourceInitialValuesHash": state_hash(source.initial_values),
        "sourceInitialLabelsHash": sha256_json(source.initial_labels_by_cell_id),
        "sourceTrajectoryHash": source.source_trajectory_hash,
        "nullTrajectoryHash": run.trajectory_hash,
        "initialFrozenPositions": compact_json(source.initial_frozen_positions),
        "finalFrozenPositions": compact_json(run.final_frozen_positions),
        "finalValuesHash": state_hash(run.final_values),
        "finalValues": compact_json(run.final_values),
        "finalCellIds": compact_json(run.final_cell_ids),
        "finalLabelsByPosition": compact_json(run.final_labels_by_position),
        "realCompleted": bool(source.completed),
        "realStopReason": source.stop_reason,
        "realFinalSortednessPercent": float(source.real_final_sortedness_percent),
        "realFinalMonotonicityError": int(source.real_final_monotonicity_error),
        "realInitialAggregation": float(source.real_initial_aggregation),
        "realPeakAggregation": float(source.real_peak_aggregation),
        "realFinalAggregation": float(source.real_final_aggregation),
        "realAucAggregation": float(source.real_auc_aggregation),
        "realDgMaxDropPercent": float(source.real_dg_max_drop_percent),
        "realDgDecreaseCount": int(source.real_dg_decrease_count),
        "realPathCurvatureSignChanges": int(source.real_path_curvature_sign_changes),
        "realSortednessTotalVariation": float(source.real_sortedness_total_variation),
        "nullFinalSortednessPercent": float(run.final_sortedness_percent),
        "nullFinalMonotonicityError": int(run.final_monotonicity_error),
        "nullInitialAggregation": float(run.initial_aggregation),
        "nullPeakAggregation": float(run.peak_aggregation),
        "nullFinalAggregation": float(run.final_aggregation),
        "nullAucAggregation": float(run.auc_aggregation),
        "nullDgMaxDropPercent": float(run.dg_max_drop_percent),
        "nullDgDecreaseCount": int(run.dg_decrease_count),
        "nullPathCurvatureSignChanges": int(run.path_curvature_sign_changes),
        "nullSortednessTotalVariation": float(run.sortedness_total_variation),
        "nullSortednessNetGain": float(run.sortedness_net_gain),
        "realMinusNullFinalSortednessPercent": float(source.real_final_sortedness_percent - run.final_sortedness_percent),
        "realMinusNullPeakAggregation": float(source.real_peak_aggregation - run.peak_aggregation),
        "realMinusNullAucAggregation": float(source.real_auc_aggregation - run.auc_aggregation),
        "realMinusNullDgMaxDropPercent": float(source.real_dg_max_drop_percent - run.dg_max_drop_percent),
        "cellMoveCountTotal": int(run.cell_move_count_total),
        "cellMoveCountsByLabel": compact_json(run.cell_move_counts_by_label),
    }


def run_s07_matrix(
    *,
    e01_artifacts: Path,
    s04_observed_path: Path,
    s04_trace_path: Path,
    artifacts_dir: Path,
    n: int,
    frozen_replicate_count: int,
    null_replicates: int,
    null_seed_base: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
    attempt_multiplier: int,
    metropolis_temperature: float,
    metropolis_min_acceptance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    chimeric_sources, chimeric_source_traces = load_chimeric_sources(s04_observed_path, s04_trace_path)
    frozen_sources, frozen_source_traces = load_frozen_sources(
        e01_artifacts=e01_artifacts,
        n=n,
        replicate_count=frozen_replicate_count,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )
    sources = chimeric_sources + frozen_sources
    source_trace_rows = chimeric_source_traces + frozen_source_traces
    null_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = list(source_trace_rows)
    started = time.perf_counter()
    for source_index, source in enumerate(sources):
        for null_model in NULL_MODELS:
            for null_replicate_index in range(int(null_replicates)):
                null_seed = stable_seed(null_seed_base, source.source_run_id, null_model, null_replicate_index)
                null_run = run_local_move_null(
                    source=source,
                    null_model=null_model,
                    null_replicate_index=null_replicate_index,
                    null_seed=null_seed,
                    attempt_multiplier=attempt_multiplier,
                    metropolis_temperature=metropolis_temperature,
                    metropolis_min_acceptance=metropolis_min_acceptance,
                )
                null_rows.append(null_record(null_run))
                trace_rows.extend(null_run.trace_rows)
        if source_index and source_index % 10 == 0:
            print(f"[S07] completed {source_index + 1}/{len(sources)} sources", flush=True)
    source_df = pd.DataFrame([source_record(source) for source in sources])
    null_df = pd.DataFrame(null_rows)
    trace_df = pd.DataFrame(trace_rows)
    diagnostic_df = diagnostics_table(source_df, null_df, trace_df)
    validation = validate_outputs(source_df, null_df, trace_df, diagnostic_df)
    validation["wallTimeSeconds"] = float(time.perf_counter() - started)
    validation["sourceRunCount"] = int(len(source_df))
    validation["nullRunCount"] = int(len(null_df))
    validation["traceRowCount"] = int(len(trace_df))
    validation["nullReplicatesPerSourceModel"] = int(null_replicates)
    validation["frozenN"] = int(n)
    validation["frozenReplicateCount"] = int(frozen_replicate_count)
    validation["artifactsDir"] = str(artifacts_dir)
    return source_df, null_df, trace_df, diagnostic_df, validation


def diagnostics_table(source_df: pd.DataFrame, null_df: pd.DataFrame, trace_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    checks = {
        "swap_count_matching": bool(null_df["swapCountMatched"].all()),
        "local_move_constraints": bool((null_df["localMoveViolations"] == 0).all()),
        "frozen_move_constraints": bool((null_df["invalidFrozenMoveCount"] == 0).all()),
        "matched_initial_arrays": bool(null_df["matchedInitialValues"].all()),
        "matched_initial_labels": bool(null_df["matchedInitialLabels"].all()),
        "label_count_preservation": bool(null_df["labelCountPreserved"].all()),
        "value_count_preservation": bool(null_df["valueCountPreserved"].all()),
        "stop_reason_logged": bool(null_df["stopReason"].notna().all() and (null_df["stopReason"].astype(str).str.len() > 0).all()),
        "caps_logged": bool(null_df["maxAttempts"].notna().all() and null_df["attemptCount"].notna().all()),
        "trace_local_moves": bool(
            trace_df[trace_df["eventKind"].eq("swap") & trace_df["nullModel"].ne("real_policy_source")]["localMoveDistance"].fillna(1).eq(1).all()
        ),
    }
    for name, passed in checks.items():
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "diagnostic": name,
                "passed": bool(passed),
                "detail": "",
            }
        )
    stop_counts = null_df.groupby(["nullModel", "stopReason"]).size().reset_index(name="count")
    for _, row in stop_counts.iterrows():
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "diagnostic": "null_stop_reason_count",
                "passed": True,
                "detail": f"{row['nullModel']}:{row['stopReason']}={int(row['count'])}",
            }
        )
    source_stop_counts = source_df.groupby(["sourceContext", "stopReason"]).size().reset_index(name="count")
    for _, row in source_stop_counts.iterrows():
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "diagnostic": "source_stop_reason_count",
                "passed": True,
                "detail": f"{row['sourceContext']}:{row['stopReason']}={int(row['count'])}",
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(
    source_df: pd.DataFrame,
    null_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
) -> dict[str, Any]:
    checks: list[str] = []
    failures: list[str] = []
    if not source_df.empty and set(source_df["sourceContext"]) == {"same_goal_chimera", "frozen_cell"}:
        checks.append("Source table includes same-goal chimeric and Frozen Cell contexts.")
    else:
        failures.append(f"Unexpected source contexts: {sorted(source_df['sourceContext'].unique().tolist()) if not source_df.empty else []}.")
    if set(null_df["nullModel"]) == set(NULL_MODELS):
        checks.append("All four S07 local-move null models are represented.")
    else:
        failures.append(f"Unexpected null models: {sorted(null_df['nullModel'].unique().tolist()) if not null_df.empty else []}.")
    if bool(null_df["swapCountMatched"].all()):
        checks.append("Every null run matched the real accepted-swap budget.")
    else:
        failures.append("At least one null run did not match its target swap count.")
    if int(null_df["localMoveViolations"].sum()) == 0:
        checks.append("All accepted null moves were adjacent local swaps.")
    else:
        failures.append("At least one accepted null move violated the adjacent local-move constraint.")
    if int(null_df["invalidFrozenMoveCount"].sum()) == 0:
        checks.append("Frozen passive/stuck constraints were respected by all accepted null moves.")
    else:
        failures.append("At least one null move violated Frozen Cell movement constraints.")
    if bool(null_df["matchedInitialValues"].all()) and bool(null_df["matchedInitialLabels"].all()):
        checks.append("Initial arrays and labels matched the paired real source for every null run.")
    else:
        failures.append("At least one null run did not match the source initial array or labels.")
    if bool(null_df["labelCountPreserved"].all()) and bool(null_df["valueCountPreserved"].all()):
        checks.append("Final null states preserved value counts and label counts.")
    else:
        failures.append("At least one null run failed value-count or label-count preservation.")
    if bool(null_df["stopReason"].notna().all()) and bool(null_df["maxAttempts"].notna().all()):
        checks.append("Stop reasons, attempt counts, and caps are logged for every null run.")
    else:
        failures.append("Missing stop reason or cap logging in null rows.")
    null_swaps = trace_df[trace_df["eventKind"].eq("swap") & trace_df["nullModel"].ne("real_policy_source")]
    if null_swaps.empty or bool(null_swaps["localMoveDistance"].eq(1).all()):
        checks.append("Trace-level local-move distance is one for all accepted null swaps.")
    else:
        failures.append("Trace table contains nonlocal null swaps.")
    if bool(diagnostic_df["passed"].all()):
        checks.append("Machine-readable diagnostic checks all passed.")
    else:
        failures.append("At least one machine-readable diagnostic check failed.")
    return {
        "success": not failures,
        "checks": checks,
        "failures": failures,
        "attemptCapRuns": int(null_df["attemptCapHit"].sum()) if not null_df.empty else 0,
        "noValidLocalMoveRuns": int((null_df["stopReason"] == "no_valid_local_move").sum()) if not null_df.empty else 0,
    }


def summarize_nulls(source_df: pd.DataFrame, null_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        null_df.groupby(["sourceContext", "nullModel", "frozenVariant"], dropna=False)
        .agg(
            sourceRunCount=("sourceRunId", "nunique"),
            nullRunCount=("nullTrajectoryHash", "count"),
            swapCountMatchedRate=("swapCountMatched", "mean"),
            attemptCapRate=("attemptCapHit", "mean"),
            realFinalSortednessMean=("realFinalSortednessPercent", "mean"),
            nullFinalSortednessMean=("nullFinalSortednessPercent", "mean"),
            realMinusNullFinalSortednessMean=("realMinusNullFinalSortednessPercent", "mean"),
            realPeakAggregationMean=("realPeakAggregation", "mean"),
            nullPeakAggregationMean=("nullPeakAggregation", "mean"),
            realMinusNullPeakAggregationMean=("realMinusNullPeakAggregation", "mean"),
            realAucAggregationMean=("realAucAggregation", "mean"),
            nullAucAggregationMean=("nullAucAggregation", "mean"),
            realMinusNullAucAggregationMean=("realMinusNullAucAggregation", "mean"),
            realDgMaxDropMean=("realDgMaxDropPercent", "mean"),
            nullDgMaxDropMean=("nullDgMaxDropPercent", "mean"),
            realMinusNullDgMaxDropMean=("realMinusNullDgMaxDropPercent", "mean"),
            nullPathCurvatureMean=("nullPathCurvatureSignChanges", "mean"),
            nullSortednessTotalVariationMean=("nullSortednessTotalVariation", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "researchStepId", STEP_ID)
    grouped.insert(0, "experimentId", EXPERIMENT_ID)
    return grouped


def context_comparison(source_df: pd.DataFrame, null_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (source_context, null_model), group in null_df.groupby(["sourceContext", "nullModel"]):
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "sourceContext": source_context,
                "nullModel": null_model,
                "sourceRunCount": int(group["sourceRunId"].nunique()),
                "nullRunCount": int(len(group)),
                "realBetterFinalSortednessFraction": float(np.mean(group["realMinusNullFinalSortednessPercent"] > 0)),
                "realBetterPeakAggregationFraction": float(np.mean(group["realMinusNullPeakAggregation"] > 0)),
                "realBetterAucAggregationFraction": float(np.mean(group["realMinusNullAucAggregation"] > 0)),
                "realHigherDgDropFraction": float(np.mean(group["realMinusNullDgMaxDropPercent"] > 0)),
                "medianRealMinusNullFinalSortedness": float(np.median(group["realMinusNullFinalSortednessPercent"])),
                "medianRealMinusNullPeakAggregation": float(np.median(group["realMinusNullPeakAggregation"])),
                "medianRealMinusNullAucAggregation": float(np.median(group["realMinusNullAucAggregation"])),
            }
        )
    return pd.DataFrame(rows)


def write_tables(
    source_df: pd.DataFrame,
    null_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    artifacts_dir: Path,
) -> dict[str, Path]:
    results_dir = artifacts_dir / "results"
    trace_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sources_parquet": results_dir / "e02_local_move_null_sources.parquet",
        "sources_csv": results_dir / "e02_local_move_null_sources.csv",
        "nulls_parquet": results_dir / "e02_local_move_nulls.parquet",
        "nulls_csv": results_dir / "e02_local_move_nulls.csv",
        "summary_parquet": results_dir / "e02_local_move_null_summary.parquet",
        "summary_csv": results_dir / "e02_local_move_null_summary.csv",
        "diagnostics_parquet": results_dir / "e02_local_move_null_diagnostics.parquet",
        "diagnostics_csv": results_dir / "e02_local_move_null_diagnostics.csv",
        "comparison_parquet": results_dir / "e02_local_move_null_context_comparison.parquet",
        "comparison_csv": results_dir / "e02_local_move_null_context_comparison.csv",
        "trace_parquet": trace_dir / "e02_local_move_null_trace_events.parquet",
        "trace_csv_gz": trace_dir / "e02_local_move_null_trace_events.csv.gz",
    }
    source_df.to_parquet(paths["sources_parquet"], index=False)
    source_df.to_csv(paths["sources_csv"], index=False)
    null_df.to_parquet(paths["nulls_parquet"], index=False)
    null_df.to_csv(paths["nulls_csv"], index=False)
    summary_df.to_parquet(paths["summary_parquet"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    diagnostics_df.to_parquet(paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(paths["diagnostics_csv"], index=False)
    comparison_df.to_parquet(paths["comparison_parquet"], index=False)
    comparison_df.to_csv(paths["comparison_csv"], index=False)
    trace_df.to_parquet(paths["trace_parquet"], index=False)
    trace_df.to_csv(paths["trace_csv_gz"], index=False, compression="gzip")
    return paths


def plot_summary(summary_df: pd.DataFrame, comparison_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "e02_local_move_nulls_summary.png"
    pdf_path = figure_dir / "e02_local_move_nulls_summary.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    plot_df = comparison_df.sort_values(["sourceContext", "nullModel"])
    x = np.arange(len(plot_df))
    colors = ["#3b6ea8" if context == "same_goal_chimera" else "#b85c38" for context in plot_df["sourceContext"]]
    axes[0].bar(x, plot_df["medianRealMinusNullFinalSortedness"], color=colors)
    axes[0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{row.sourceContext}\n{row.nullModel}" for row in plot_df.itertuples()], rotation=45, ha="right")
    axes[0].set_ylabel("Median real minus null final Sortedness")
    axes[0].set_title("Sorting outcome")
    axes[1].bar(x, plot_df["medianRealMinusNullAucAggregation"], color=colors)
    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{row.sourceContext}\n{row.nullModel}" for row in plot_df.itertuples()], rotation=45, ha="right")
    axes[1].set_ylabel("Median real minus null Aggregation AUC")
    axes[1].set_title("Label clustering trajectory")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    fig.suptitle("E02 S07 local-move nulls matched on real swap budgets")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append(
                {
                    "path": str(path),
                    "sizeBytes": int(path.stat().st_size),
                    "sha256": sha256_path(path),
                }
            )
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    script_dir = code_dir / "scripts"
    test_dir = code_dir / "tests"
    package_dir = code_dir / "e02_deterministic_simulator"
    script_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    shutil.copy2(REPO_ROOT / "scripts" / "e02_s07_local_move_nulls.py", script_dir / "e02_s07_local_move_nulls.py")
    shutil.copy2(REPO_ROOT / "tests" / "test_e02_local_move_nulls.py", test_dir / "test_e02_local_move_nulls.py")
    shutil.copytree(REPO_ROOT / "e02_deterministic_simulator", package_dir, ignore=shutil.ignore_patterns("__pycache__"))
    return [
        script_dir / "e02_s07_local_move_nulls.py",
        test_dir / "test_e02_local_move_nulls.py",
        package_dir / "__init__.py",
        package_dir / "simulator.py",
        package_dir / "metrics.py",
    ]


def write_validation_report(step_dir: Path, validation: dict[str, Any]) -> Path:
    path = step_dir / "validation_report.md"
    lines = [
        "# E02 S07 Validation Report",
        "",
        "- Research step ID: S07",
        f"- Completion status: {'completed' if validation['success'] else 'failed'}",
        "- Artifacts written: local-move null tables, source budget tables, diagnostics, traces, figures, copied code, manifests, and status files.",
        f"- Validation result: {'passed' if validation['success'] else 'failed'}",
        f"- Caveats or blockers: {validation.get('caveats', 'none')}",
        "- Recommended next action: stop before S08 for Chief Scientist review.",
        "",
        "## Checks",
    ]
    lines.extend(f"- {item}" for item in validation["checks"])
    if validation["failures"]:
        lines.append("")
        lines.append("## Failures")
        lines.extend(f"- {item}" for item in validation["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_repo_tests(step_dir: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    cmd = [sys.executable, "-m", "unittest", "tests.test_e02_local_move_nulls"]
    result = run_command(cmd, cwd=REPO_ROOT, env=env)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr + "\n",
        encoding="utf-8",
    )
    return {
        "command": cmd,
        "returncode": result.returncode,
        "ok": result.ok,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "logPath": str(log_path),
    }


def outcome_classification(null_df: pd.DataFrame, success: bool) -> str:
    if not success:
        return "constraining/contradictory"
    if null_df.empty:
        return "null"
    final_advantage = float(null_df["realMinusNullFinalSortednessPercent"].median())
    aggregation_advantage = float(null_df["realMinusNullAucAggregation"].median())
    if final_advantage > 0 or aggregation_advantage > 0:
        return "supportive"
    return "null"


def write_reports_and_manifests(
    *,
    artifacts_dir: Path,
    table_paths: dict[str, Path],
    figure_paths: tuple[Path, Path],
    code_paths: list[Path],
    source_df: pd.DataFrame,
    null_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    validation: dict[str, Any],
    repo_tests: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Path]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    step_dir.mkdir(parents=True, exist_ok=True)
    validation["success"] = bool(validation["success"] and repo_tests["ok"])
    if repo_tests["ok"]:
        validation["checks"].append("Repository unit tests passed for S07 local-move null helpers.")
    else:
        validation["failures"].append("Repository unit tests failed for S07 local-move null helpers.")
    validation["caveats"] = (
        "Frozen Cell nulls use a bounded n=30 representative cell-view matrix with diagnostic chimeric-like labels; "
        "the local nulls match swap budgets exactly but do not reproduce policy-specific comparison logic."
    )
    validation_report_path = write_validation_report(step_dir, validation)
    classification = outcome_classification(null_df, bool(validation["success"]))
    artifact_inputs = list(table_paths.values()) + list(figure_paths) + code_paths + [
        validation_report_path,
        Path(repo_tests["logPath"]),
    ]
    manifest_path = step_dir / "artifact_manifest.json"
    manifest = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": get_git_metadata(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "parameters": vars(args),
        "policyLogicId": POLICY_LOGIC_ID,
        "policySignatureSha256": policy_signature(),
        "artifacts": collect_artifacts(artifact_inputs),
        "validation": validation,
        "repoTests": repo_tests,
        "outcomeClassification": classification,
    }
    write_json(manifest_path, manifest)
    status_path = step_dir / "status.json"
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"]),
        "status": "completed" if validation["success"] else "failed",
        "artifactsWritten": [item["path"] for item in manifest["artifacts"]] + [str(manifest_path)],
        "validationResult": "passed" if validation["success"] else "failed",
        "caveatsOrBlockers": validation["caveats"] if validation["success"] else "; ".join(validation["failures"]),
        "recommendedNextAction": "Stop before S08 for Chief Scientist review; if accepted, proceed to S08 DG-specific null stress tests.",
    }
    write_json(status_path, status_payload)
    top_summary = comparison_df.sort_values(["sourceContext", "nullModel"])
    summary_lines = [
        "# E02 S07 Summary",
        "",
        "- Research step ID: S07",
        f"- Completion status: {'completed' if validation['success'] else 'failed'}",
        "- Artifacts written: `$ARTIFACTS_DIR/results/e02_local_move_nulls.parquet`, source/summary/diagnostic tables, trace events, figures, copied code, manifest, and status JSON.",
        f"- Validation result: {'passed' if validation['success'] else 'failed'}",
        f"- Caveats or blockers: {validation['caveats']}",
        "- Lay summary: S07 asked whether simple local random movement with the same swap budget can reproduce the real policy outcomes. The nulls exactly matched each real run's accepted-swap count and respected adjacent-move and Frozen Cell constraints. The result table now shows where real policies exceed, match, or underperform those movement-only controls.",
        "- Recommended next action: stop before S08 for Chief Scientist review; if accepted, run DG-specific null stress tests in S08.",
        f"- Outcome classification: {classification}",
        "",
        "## Run Counts",
        "",
        markdown_table(
            ["Item", "Count"],
            [
                ["Source runs", len(source_df)],
                ["Null runs", len(null_df)],
                ["Trace rows", validation["traceRowCount"]],
                ["Attempt-cap null runs", validation["attemptCapRuns"]],
                ["No-valid-local-move null runs", validation["noValidLocalMoveRuns"]],
            ],
        ),
        "",
        "## Context Comparison",
        "",
        markdown_table(
            [
                "Source context",
                "Null model",
                "Null runs",
                "Real > null final Sortedness frac.",
                "Median real-null final Sortedness",
                "Median real-null Aggregation AUC",
            ],
            [
                [
                    row.sourceContext,
                    row.nullModel,
                    int(row.nullRunCount),
                    float(row.realBetterFinalSortednessFraction),
                    float(row.medianRealMinusNullFinalSortedness),
                    float(row.medianRealMinusNullAucAggregation),
                ]
                for row in top_summary.itertuples(index=False)
            ],
        ),
        "",
        "## Validation",
        "",
    ]
    summary_lines.extend(f"- {item}" for item in validation["checks"])
    if validation["failures"]:
        summary_lines.append("")
        summary_lines.append("## Failures")
        summary_lines.extend(f"- {item}" for item in validation["failures"])
    summary_path = step_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    run_manifest_path = artifacts_dir / "provenance" / "run_manifest.json"
    run_manifest = read_json(run_manifest_path) if run_manifest_path.exists() else {"runs": []}
    run_manifest.setdefault("runs", [])
    run_manifest["runs"].append(
        {
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "status": status_payload["status"],
            "success": status_payload["success"],
            "primaryArtifacts": {
                "nullResults": str(table_paths["nulls_parquet"]),
                "summary": str(summary_path),
                "status": str(status_path),
                "manifest": str(manifest_path),
            },
            "git": get_git_metadata(),
        }
    )
    write_json(run_manifest_path, run_manifest)
    final_artifact_inputs = artifact_inputs + [summary_path, status_path, manifest_path, run_manifest_path]
    manifest["artifacts"] = collect_artifacts(final_artifact_inputs)
    write_json(manifest_path, manifest)
    status_payload["artifactsWritten"] = [item["path"] for item in manifest["artifacts"]]
    write_json(status_path, status_payload)
    return {
        "summary": summary_path,
        "status": status_path,
        "manifest": manifest_path,
        "validation_report": validation_report_path,
        "run_manifest": run_manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--e01-artifacts", type=Path, default=DEFAULT_E01_ARTIFACTS)
    parser.add_argument("--s04-observed", type=Path, default=DEFAULT_S04_OBSERVED)
    parser.add_argument("--s04-trace", type=Path, default=DEFAULT_S04_TRACE)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--frozen-replicate-count", type=int, default=3)
    parser.add_argument("--null-replicates", type=int, default=25)
    parser.add_argument("--null-seed-base", type=int, default=7707001)
    parser.add_argument("--max-activations", type=int, default=250000)
    parser.add_argument("--max-swaps", type=int, default=50000)
    parser.add_argument("--max-comparisons", type=int, default=500000)
    parser.add_argument("--attempt-multiplier", type=int, default=250)
    parser.add_argument("--metropolis-temperature", type=float, default=2.0)
    parser.add_argument("--metropolis-min-acceptance", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    step_dir.mkdir(parents=True, exist_ok=True)
    source_df, null_df, trace_df, diagnostics_df, validation = run_s07_matrix(
        e01_artifacts=args.e01_artifacts,
        s04_observed_path=args.s04_observed,
        s04_trace_path=args.s04_trace,
        artifacts_dir=args.artifacts_dir,
        n=args.n,
        frozen_replicate_count=args.frozen_replicate_count,
        null_replicates=args.null_replicates,
        null_seed_base=args.null_seed_base,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
        attempt_multiplier=args.attempt_multiplier,
        metropolis_temperature=args.metropolis_temperature,
        metropolis_min_acceptance=args.metropolis_min_acceptance,
    )
    summary_df = summarize_nulls(source_df, null_df)
    comparison_df = context_comparison(source_df, null_df)
    table_paths = write_tables(source_df, null_df, summary_df, diagnostics_df, comparison_df, trace_df, args.artifacts_dir)
    figure_paths = plot_summary(summary_df, comparison_df, args.artifacts_dir / "figures" / "e02")
    code_paths = copy_code_artifacts(step_dir)
    repo_tests = run_repo_tests(step_dir)
    report_paths = write_reports_and_manifests(
        artifacts_dir=args.artifacts_dir,
        table_paths=table_paths,
        figure_paths=figure_paths,
        code_paths=code_paths,
        source_df=source_df,
        null_df=null_df,
        summary_df=summary_df,
        diagnostics_df=diagnostics_df,
        comparison_df=comparison_df,
        validation=validation,
        repo_tests=repo_tests,
        args=args,
    )
    print(f"Wrote S07 null results to {table_paths['nulls_parquet']}")
    print(f"Wrote S07 summary to {report_paths['summary']}")
    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
