#!/usr/bin/env python3
"""Execute E02 S04 trajectory-preserving label-shuffle nulls.

S04 reruns the validated deterministic S01 event simulator for same-goal
chimeric conditions, records fixed cell-position/value trajectories, then
randomly reassigns Algotype labels to cell identities. The null preserves
positions, values, and movement history while breaking the observed link
between Algotype label and trajectory.
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
STEP_ID = "S04"
STEP_NUMBER = 4
DEFAULT_E01_ARTIFACTS = Path("/previous-artifacts/E01")
CHIMERIC_MIXTURES = [
    "bubble_insertion",
    "bubble_selection",
    "insertion_selection",
    "bubble_insertion_selection",
]
ALGOTYPE_TO_CODE = {"bubble": 0, "insertion": 1, "selection": 2}
CODE_TO_ALGOTYPE = {value: key for key, value in ALGOTYPE_TO_CODE.items()}
POLICY_LOGIC_ID = "DeterministicEventSimulator:S01_policy_semantics"
POLICY_SIGNATURE_SHA256 = hashlib.sha256(POLICY_LOGIC_ID.encode("utf-8")).hexdigest()


@dataclass
class CommandResult:
    args: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    ok: bool


@dataclass
class ObservedTrajectory:
    condition: dict[str, Any]
    seed_row: dict[str, Any]
    n: int
    initial_values: list[int]
    initial_algotypes: list[str]
    final_values: list[int]
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


def sha256_json(value: Any) -> str:
    payload = json.dumps(json_ready(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    raise ValueError(f"unsupported S04 chimeric mixture: {mixture_id}")


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


def load_s04_inputs(e01_artifacts: Path, *, n: int, replicate_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_ids = [same_goal_condition_id(mixture) for mixture in CHIMERIC_MIXTURES]
    condition_matrix = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "condition_matrix.csv")
    seed_table = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "seed_table.csv")
    conditions = condition_matrix[condition_matrix["conditionId"].isin(condition_ids)].copy()
    conditions["s04N"] = int(n)
    conditions = conditions.sort_values("conditionId").reset_index(drop=True)
    seeds = seed_table[
        (seed_table["conditionId"].isin(condition_ids))
        & (seed_table["replicateIndex"] < int(replicate_count))
    ].copy()
    seeds = seeds.sort_values(["conditionId", "replicateIndex"]).reset_index(drop=True)
    if len(conditions) != len(condition_ids):
        missing = sorted(set(condition_ids) - set(conditions["conditionId"]))
        raise ValueError(f"Missing S04 same-goal chimeric conditions in E01 matrix: {missing}")
    expected_seed_rows = len(condition_ids) * int(replicate_count)
    if len(seeds) != expected_seed_rows:
        raise ValueError(f"Expected {expected_seed_rows} S04 seed rows, found {len(seeds)}")
    return conditions, seeds


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), separators=(",", ":"))


def codes_for_algotypes(algotypes: list[str]) -> np.ndarray:
    return np.asarray([ALGOTYPE_TO_CODE[item] for item in algotypes], dtype=np.int16)


def aggregation_from_codes(codes: np.ndarray) -> float:
    if len(codes) < 2:
        return 1.0
    return float(np.mean(codes[:-1] == codes[1:]))


def aggregation_curve_for_assignment(cell_ids_by_event: np.ndarray, labels_by_cell_id: np.ndarray) -> np.ndarray:
    labels_by_position = labels_by_cell_id[cell_ids_by_event]
    if labels_by_position.shape[1] < 2:
        return np.ones(labels_by_position.shape[0], dtype=float)
    return np.mean(labels_by_position[:, :-1] == labels_by_position[:, 1:], axis=1).astype(float)


def normalized_auc(curve: np.ndarray) -> float:
    if len(curve) == 0:
        return float("nan")
    if len(curve) == 1:
        return float(curve[0])
    x = np.arange(len(curve), dtype=float)
    return float(np.trapezoid(curve.astype(float), x) / (len(curve) - 1))


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
        }
        for row in trace_rows
    ]
    return sha256_json(payload)


def snapshot_trace_row(sim: DeterministicEventSimulator, base_row: dict[str, Any]) -> dict[str, Any]:
    values = sim.current_values()
    cell_ids = [int(cell.cell_id) for cell in sim.cells]
    algotypes = sim.current_algotypes()
    row = dict(base_row)
    row["research_step_id"] = STEP_ID
    row["policyLogicId"] = POLICY_LOGIC_ID
    row["policySignatureSha256"] = POLICY_SIGNATURE_SHA256
    row["values_json"] = compact_json(values)
    row["cell_ids_json"] = compact_json(cell_ids)
    row["observed_aggregation"] = aggregation(algotypes)
    row["observed_algotype_counts_json"] = compact_json(dict(sorted(Counter(algotypes).items())))
    row["values_hash_from_json"] = state_hash(values)
    return row


def validate_fixed_trajectory(
    trace_rows: list[dict[str, Any]],
    *,
    initial_values: list[int],
    initial_algotypes: list[str],
) -> list[str]:
    failures: list[str] = []
    n = len(initial_values)
    expected_ids = list(range(n))
    expected_label_counts = Counter(initial_algotypes)
    for row in trace_rows:
        event_index = int(row["event_index"])
        values = json.loads(row["values_json"])
        cell_ids = json.loads(row["cell_ids_json"])
        algotypes = json.loads(row["algotypes_json"])
        if sorted(cell_ids) != expected_ids:
            failures.append(f"event {event_index}: cell IDs are not a position permutation")
        reconstructed_values = [initial_values[int(cell_id)] for cell_id in cell_ids]
        reconstructed_algotypes = [initial_algotypes[int(cell_id)] for cell_id in cell_ids]
        if values != reconstructed_values:
            failures.append(f"event {event_index}: values do not match fixed cell identity history")
        if row["state_hash"] != state_hash(values):
            failures.append(f"event {event_index}: state hash does not match values_json")
        if algotypes != reconstructed_algotypes:
            failures.append(f"event {event_index}: observed Algotypes do not match fixed identity labels")
        if Counter(algotypes) != expected_label_counts:
            failures.append(f"event {event_index}: observed Algotype counts changed")
    return failures


def run_observed_trajectory(
    *,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    n: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int = 2,
) -> ObservedTrajectory:
    input_seed = int(seed_row["inputPermutationSeed"])
    algotype_seed = seed_row.get("algotypeAssignmentSeed")
    if pd.isna(algotype_seed):
        algotype_seed = int(seed_row["schedulerSeed"])
    initial_values = initial_values_from_seed(input_seed, n=n, profile=str(condition["inputProfile"]))
    initial_algotypes = algotypes_from_seed(str(condition["mixtureId"]), n, algotype_seed)
    sim = DeterministicEventSimulator(
        initial_values,
        initial_algotypes,
        scheduler_seed=int(seed_row["schedulerSeed"]),
        tie_breaker_seed=int(seed_row["tieBreakerSeed"]),
        condition_id=str(condition["conditionId"]),
        research_step_id=STEP_ID,
    )
    started_at = time.perf_counter()
    trace_rows = [snapshot_trace_row(sim, sim.trace_rows[-1])]
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

        outcome = sim.step()
        if outcome.swapped:
            trace_rows.append(snapshot_trace_row(sim, sim.trace_rows[-1]))

    validation_failures = validate_fixed_trajectory(
        trace_rows,
        initial_values=initial_values,
        initial_algotypes=initial_algotypes,
    )
    if validation_failures:
        raise ValueError("; ".join(validation_failures[:5]))
    observed_curve = np.asarray([float(row["observed_aggregation"]) for row in trace_rows], dtype=float)
    history_hash = trajectory_history_hash(trace_rows)
    return ObservedTrajectory(
        condition=condition,
        seed_row=seed_row,
        n=n,
        initial_values=initial_values,
        initial_algotypes=initial_algotypes,
        final_values=sim.current_values(),
        final_algotypes=sim.current_algotypes(),
        completed=sim.is_sorted(),
        stop_reason=stop_reason,
        swap_count=sim.swap_count,
        comparison_count=sim.comparison_count,
        archived_compare_and_swap_count=sim.archived_compare_and_swap_count,
        activation_count=sim.activation_count,
        event_count=len(trace_rows),
        wall_time_seconds=time.perf_counter() - started_at,
        trace_rows=trace_rows,
        history_hash=history_hash,
        observed_curve=observed_curve,
    )


def run_label_shuffle_nulls(
    trajectory: ObservedTrajectory,
    *,
    null_replicates: int,
    null_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(null_seed))
    cell_ids_by_event = np.asarray(
        [json.loads(row["cell_ids_json"]) for row in trajectory.trace_rows],
        dtype=np.int16,
    )
    initial_codes = codes_for_algotypes(trajectory.initial_algotypes)
    observed = trajectory.observed_curve
    null_curves = np.empty((int(null_replicates), len(observed)), dtype=np.float32)
    null_metric_rows: list[dict[str, Any]] = []
    label_count_failures = 0
    expected_counts = Counter(trajectory.initial_algotypes)
    expected_code_counts = Counter(initial_codes.tolist())
    for null_index in range(int(null_replicates)):
        labels_by_cell_id = np.array(initial_codes, copy=True)
        rng.shuffle(labels_by_cell_id)
        if Counter(labels_by_cell_id.tolist()) != expected_code_counts:
            label_count_failures += 1
        curve = aggregation_curve_for_assignment(cell_ids_by_event, labels_by_cell_id)
        null_curves[null_index, :] = curve.astype(np.float32)
        max_value = float(np.max(curve))
        peak_event = int(np.argmax(curve))
        label_counts = Counter(CODE_TO_ALGOTYPE[int(code)] for code in labels_by_cell_id.tolist())
        if label_counts != expected_counts:
            label_count_failures += 1
        null_metric_rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "conditionId": trajectory.condition["conditionId"],
                "mixtureId": trajectory.condition["mixtureId"],
                "replicateIndex": int(trajectory.seed_row["replicateIndex"]),
                "replicateNumber": int(trajectory.seed_row["replicateNumber"]),
                "nullReplicateIndex": int(null_index),
                "nullSeed": int(null_seed),
                "trajectoryHistoryHash": trajectory.history_hash,
                "labelAssignmentHash": sha256_json(labels_by_cell_id.tolist()),
                "labelCounts": compact_json(dict(sorted(label_counts.items()))),
                "nullPeakAggregation": max_value,
                "nullPeakEventIndex": peak_event,
                "nullFinalAggregation": float(curve[-1]),
                "nullAucAggregation": normalized_auc(curve),
            }
        )

    observed_peak = float(np.max(observed))
    observed_auc = normalized_auc(observed)
    observed_final = float(observed[-1])
    timepoint_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(trajectory.trace_rows):
        null_values = null_curves[:, idx].astype(float)
        obs_value = float(observed[idx])
        timepoint_rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "conditionId": trajectory.condition["conditionId"],
                "mixtureId": trajectory.condition["mixtureId"],
                "replicateIndex": int(trajectory.seed_row["replicateIndex"]),
                "replicateNumber": int(trajectory.seed_row["replicateNumber"]),
                "eventIndex": int(row["event_index"]),
                "activationIndex": int(row["activation_index"]),
                "swapCount": int(row["swap_count"]),
                "stateHash": row["state_hash"],
                "trajectoryHistoryHash": trajectory.history_hash,
                "observedAggregation": obs_value,
                "nullMeanAggregation": float(np.mean(null_values)),
                "nullSdAggregation": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0,
                "nullQ025Aggregation": float(np.quantile(null_values, 0.025)),
                "nullQ50Aggregation": float(np.quantile(null_values, 0.5)),
                "nullQ975Aggregation": float(np.quantile(null_values, 0.975)),
                "empiricalPHighAtTimepoint": empirical_p_high(obs_value, null_values),
            }
        )
    validation = {
        "labelCountFailures": int(label_count_failures),
        "cellIdPermutationRows": int(np.sum([sorted(row.tolist()) == list(range(trajectory.n)) for row in cell_ids_by_event])),
        "cellIdRows": int(len(cell_ids_by_event)),
        "nullReplicates": int(null_replicates),
        "observedPeakAggregation": observed_peak,
        "observedAucAggregation": observed_auc,
        "observedFinalAggregation": observed_final,
        "nullPeakMean": float(np.mean(null_curves.max(axis=1))),
        "nullAucMean": float(np.mean([normalized_auc(curve.astype(float)) for curve in null_curves])),
    }
    return pd.DataFrame(null_metric_rows), pd.DataFrame(timepoint_rows), validation


def empirical_p_high(observed: float, null_values: np.ndarray | list[float]) -> float:
    values = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(values >= float(observed))) / (len(values) + 1))


def z_effect(observed: float, null_values: np.ndarray | list[float]) -> float:
    values = np.asarray(null_values, dtype=float)
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if sd <= 0:
        return 0.0 if math.isclose(float(observed), float(np.mean(values))) else float("inf")
    return float((float(observed) - float(np.mean(values))) / sd)


def observed_summary_record(trajectory: ObservedTrajectory, null_df: pd.DataFrame) -> dict[str, Any]:
    observed_curve = trajectory.observed_curve
    null_peak = null_df["nullPeakAggregation"].to_numpy(dtype=float)
    null_auc = null_df["nullAucAggregation"].to_numpy(dtype=float)
    null_final = null_df["nullFinalAggregation"].to_numpy(dtype=float)
    observed_peak = float(np.max(observed_curve))
    observed_peak_event = int(np.argmax(observed_curve))
    observed_auc = normalized_auc(observed_curve)
    observed_final = float(observed_curve[-1])
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": trajectory.condition["conditionId"],
        "mixtureId": trajectory.condition["mixtureId"],
        "policyLogicId": POLICY_LOGIC_ID,
        "policySignatureSha256": POLICY_SIGNATURE_SHA256,
        "n": int(trajectory.n),
        "replicateIndex": int(trajectory.seed_row["replicateIndex"]),
        "replicateNumber": int(trajectory.seed_row["replicateNumber"]),
        "inputPermutationSeed": int(trajectory.seed_row["inputPermutationSeed"]),
        "algotypeAssignmentSeed": int(trajectory.seed_row["algotypeAssignmentSeed"]),
        "schedulerSeed": int(trajectory.seed_row["schedulerSeed"]),
        "tieBreakerSeed": int(trajectory.seed_row["tieBreakerSeed"]),
        "trajectoryHistoryHash": trajectory.history_hash,
        "initialValuesHash": state_hash(trajectory.initial_values),
        "finalValuesHash": state_hash(trajectory.final_values),
        "initialValues": compact_json(trajectory.initial_values),
        "finalValues": compact_json(trajectory.final_values),
        "initialAlgotypes": compact_json(trajectory.initial_algotypes),
        "finalAlgotypes": compact_json(trajectory.final_algotypes),
        "completed": bool(trajectory.completed),
        "stopReason": trajectory.stop_reason,
        "swapCount": int(trajectory.swap_count),
        "comparisonCount": int(trajectory.comparison_count),
        "archivedCompareAndSwapCount": int(trajectory.archived_compare_and_swap_count),
        "activationCount": int(trajectory.activation_count),
        "eventCount": int(trajectory.event_count),
        "finalSortednessPercent": sortedness_percent(trajectory.final_values),
        "finalMonotonicityError": monotonicity_error(trajectory.final_values),
        "observedInitialAggregation": float(observed_curve[0]),
        "observedPeakAggregation": observed_peak,
        "observedPeakEventIndex": observed_peak_event,
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
        "wallTimeSeconds": float(trajectory.wall_time_seconds),
    }


def annotate_trace_rows(trajectory: ObservedTrajectory) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in trajectory.trace_rows:
        item = dict(row)
        item["experimentId"] = EXPERIMENT_ID
        item["researchStepId"] = STEP_ID
        item["mixtureId"] = trajectory.condition["mixtureId"]
        item["replicateIndex"] = int(trajectory.seed_row["replicateIndex"])
        item["replicateNumber"] = int(trajectory.seed_row["replicateNumber"])
        item["inputPermutationSeed"] = int(trajectory.seed_row["inputPermutationSeed"])
        item["algotypeAssignmentSeed"] = int(trajectory.seed_row["algotypeAssignmentSeed"])
        item["trajectoryHistoryHash"] = trajectory.history_hash
        rows.append(item)
    return rows


def run_s04_matrix(
    *,
    e01_artifacts: Path,
    n: int,
    replicate_count: int,
    null_replicates: int,
    null_seed_base: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    conditions_df, seeds_df = load_s04_inputs(e01_artifacts, n=n, replicate_count=replicate_count)
    observed_rows: list[dict[str, Any]] = []
    null_rows: list[pd.DataFrame] = []
    timepoint_rows: list[pd.DataFrame] = []
    trace_rows: list[dict[str, Any]] = []
    validation_runs: list[dict[str, Any]] = []
    run_index = 0
    for condition_tuple in conditions_df.itertuples(index=False):
        condition = condition_tuple._asdict()
        condition_seeds = seeds_df[seeds_df["conditionId"] == condition["conditionId"]].copy()
        for seed_tuple in condition_seeds.itertuples(index=False):
            seed_row = seed_tuple._asdict()
            trajectory = run_observed_trajectory(
                condition=condition,
                seed_row=seed_row,
                n=n,
                max_activations=max_activations,
                max_swaps=max_swaps,
                max_comparisons=max_comparisons,
            )
            null_seed = int(null_seed_base + run_index * 1_000_003)
            run_null_df, run_timepoint_df, run_validation = run_label_shuffle_nulls(
                trajectory,
                null_replicates=null_replicates,
                null_seed=null_seed,
            )
            observed_rows.append(observed_summary_record(trajectory, run_null_df))
            null_rows.append(run_null_df)
            timepoint_rows.append(run_timepoint_df)
            trace_rows.extend(annotate_trace_rows(trajectory))
            validation_runs.append(
                {
                    "conditionId": condition["conditionId"],
                    "mixtureId": condition["mixtureId"],
                    "replicateIndex": int(seed_row["replicateIndex"]),
                    "eventCount": int(trajectory.event_count),
                    "trajectoryHistoryHash": trajectory.history_hash,
                    **run_validation,
                }
            )
            run_index += 1

    observed_df = pd.DataFrame(observed_rows)
    null_df = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    timepoint_df = pd.concat(timepoint_rows, ignore_index=True) if timepoint_rows else pd.DataFrame()
    trace_df = pd.DataFrame(trace_rows)
    validation = {
        "conditionCount": int(len(conditions_df)),
        "replicateCountPerCondition": int(replicate_count),
        "expectedObservedRuns": int(len(conditions_df) * replicate_count),
        "expectedNullRows": int(len(conditions_df) * replicate_count * null_replicates),
        "nullReplicatesPerRun": int(null_replicates),
        "runs": validation_runs,
    }
    return observed_df, null_df, timepoint_df, trace_df, validation


def summarize_by_mixture(observed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        observed_df.groupby("mixtureId")
        .agg(
            runCount=("conditionId", "count"),
            completedCount=("completed", "sum"),
            meanObservedPeakAggregation=("observedPeakAggregation", "mean"),
            meanNullPeakMean=("nullPeakMean", "mean"),
            meanPeakDeltaVsNull=("observedPeakAggregation", lambda x: float("nan")),
            medianPHighPeak=("empiricalPHighPeak", "median"),
            minPHighPeak=("empiricalPHighPeak", "min"),
            meanObservedAucAggregation=("observedAucAggregation", "mean"),
            meanNullAucMean=("nullAucMean", "mean"),
            medianPHighAuc=("empiricalPHighAuc", "median"),
            minPHighAuc=("empiricalPHighAuc", "min"),
            meanObservedFinalAggregation=("observedFinalAggregation", "mean"),
            meanNullFinalMean=("nullFinalMean", "mean"),
            medianPHighFinal=("empiricalPHighFinal", "median"),
            significantPeakRuns=("empiricalPHighPeak", lambda x: int(np.sum(np.asarray(x, dtype=float) <= 0.05))),
            significantAucRuns=("empiricalPHighAuc", lambda x: int(np.sum(np.asarray(x, dtype=float) <= 0.05))),
        )
        .reset_index()
    )
    grouped["meanPeakDeltaVsNull"] = grouped["meanObservedPeakAggregation"] - grouped["meanNullPeakMean"]
    grouped["meanAucDeltaVsNull"] = grouped["meanObservedAucAggregation"] - grouped["meanNullAucMean"]
    grouped["meanFinalDeltaVsNull"] = grouped["meanObservedFinalAggregation"] - grouped["meanNullFinalMean"]
    grouped["completionRate"] = grouped["completedCount"] / grouped["runCount"]
    return grouped


def write_tables(
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    timepoint_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    artifacts_dir: Path,
) -> dict[str, Path]:
    results_dir = artifacts_dir / "results"
    traces_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    summary_df = summarize_by_mixture(observed_df)
    paths = {
        "observed_parquet": results_dir / "e02_label_shuffle_observed.parquet",
        "observed_csv": results_dir / "e02_label_shuffle_observed.csv",
        "null_parquet": results_dir / "e02_label_shuffle_nulls.parquet",
        "null_csv": results_dir / "e02_label_shuffle_nulls.csv",
        "timepoint_parquet": results_dir / "e02_label_shuffle_timepoint_summary.parquet",
        "timepoint_csv": results_dir / "e02_label_shuffle_timepoint_summary.csv",
        "mixture_summary_parquet": results_dir / "e02_label_shuffle_mixture_summary.parquet",
        "mixture_summary_csv": results_dir / "e02_label_shuffle_mixture_summary.csv",
        "trace_parquet": traces_dir / "e02_label_shuffle_observed_trajectories.parquet",
        "trace_csv_gz": traces_dir / "e02_label_shuffle_observed_trajectories.csv.gz",
    }
    observed_df.to_parquet(paths["observed_parquet"], index=False)
    observed_df.to_csv(paths["observed_csv"], index=False)
    null_df.to_parquet(paths["null_parquet"], index=False)
    null_df.to_csv(paths["null_csv"], index=False)
    timepoint_df.to_parquet(paths["timepoint_parquet"], index=False)
    timepoint_df.to_csv(paths["timepoint_csv"], index=False)
    summary_df.to_parquet(paths["mixture_summary_parquet"], index=False)
    summary_df.to_csv(paths["mixture_summary_csv"], index=False)
    trace_df.to_parquet(paths["trace_parquet"], index=False)
    trace_df.to_csv(paths["trace_csv_gz"], index=False, compression="gzip")
    return paths


def plot_label_shuffle_summary(observed_df: pd.DataFrame, summary_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    mixtures = summary_df["mixtureId"].tolist()
    x = np.arange(len(mixtures), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].bar(x - 0.18, summary_df["meanObservedPeakAggregation"], width=0.36, label="observed peak", color="#2f6f73")
    axes[0].bar(x + 0.18, summary_df["meanNullPeakMean"], width=0.36, label="null peak mean", color="#c47f2d")
    for idx, mixture in enumerate(mixtures):
        y = observed_df[observed_df["mixtureId"] == mixture]["observedPeakAggregation"].to_numpy(dtype=float)
        axes[0].scatter(np.full(len(y), idx - 0.18), y, color="#143f42", s=18, zorder=3)
    axes[0].set_ylabel("Aggregation")
    axes[0].set_title("Peak Aggregation")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)

    axes[1].bar(x - 0.18, summary_df["meanObservedAucAggregation"], width=0.36, label="observed AUC", color="#6a5acd")
    axes[1].bar(x + 0.18, summary_df["meanNullAucMean"], width=0.36, label="null AUC mean", color="#7b8f2a")
    for idx, mixture in enumerate(mixtures):
        y = observed_df[observed_df["mixtureId"] == mixture]["observedAucAggregation"].to_numpy(dtype=float)
        axes[1].scatter(np.full(len(y), idx - 0.18), y, color="#342a70", s=18, zorder=3)
    axes[1].set_ylabel("Normalized event-index AUC")
    axes[1].set_title("Aggregation Curve Area")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([item.replace("_", "\n") for item in mixtures], fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    png = figure_dir / "e02_label_shuffle_nulls_summary.png"
    pdf = figure_dir / "e02_label_shuffle_nulls_summary.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def validate_outputs(
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    timepoint_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    validation: dict[str, Any],
) -> tuple[bool, list[str], list[str]]:
    checks: list[str] = []
    failures: list[str] = []
    expected_runs = int(validation["expectedObservedRuns"])
    expected_null_rows = int(validation["expectedNullRows"])
    if len(observed_df) == expected_runs:
        checks.append(f"Observed chimeric trajectory row count matched expected {expected_runs}.")
    else:
        failures.append(f"Observed rows {len(observed_df)} did not match expected {expected_runs}.")
    if len(null_df) == expected_null_rows:
        checks.append(f"Label-shuffle null row count matched expected {expected_null_rows}.")
    else:
        failures.append(f"Null rows {len(null_df)} did not match expected {expected_null_rows}.")
    if len(timepoint_df) == len(trace_df):
        checks.append("Timepoint null summaries align one-to-one with observed trajectory rows.")
    else:
        failures.append("Timepoint summary rows do not align with observed trajectory rows.")
    if observed_df["completed"].all() and set(observed_df["stopReason"]) == {"sorted"}:
        checks.append("All observed S04 trajectories completed with sorted final states.")
    else:
        failures.append("At least one observed S04 trajectory did not complete sorted.")
    if observed_df["trajectoryHistoryHash"].nunique() == len(observed_df):
        checks.append("Each observed run has a distinct fixed trajectory-history hash.")
    else:
        failures.append("Trajectory-history hashes are not unique by observed run.")
    merged_hash_counts = null_df.groupby(["conditionId", "replicateIndex"])["trajectoryHistoryHash"].nunique()
    if len(merged_hash_counts) == expected_runs and int(merged_hash_counts.max()) == 1:
        checks.append("Every null replicate references exactly one unchanged observed trajectory-history hash.")
    else:
        failures.append("Null replicate history hashes are missing or inconsistent.")
    if trace_df["policySignatureSha256"].nunique() == 1:
        checks.append("Policy signature is unchanged for all observed S04 trajectories.")
    else:
        failures.append("Policy signature varied across S04 trajectories.")
    validation_runs = validation.get("runs", [])
    if all(int(run["labelCountFailures"]) == 0 for run in validation_runs):
        checks.append("All label-shuffle assignments preserved Algotype label counts.")
    else:
        failures.append("At least one label-shuffle assignment changed Algotype label counts.")
    if all(int(run["cellIdPermutationRows"]) == int(run["cellIdRows"]) for run in validation_runs):
        checks.append("All observed trajectory rows preserved cell-position permutations.")
    else:
        failures.append("At least one trajectory row lost the fixed cell-position permutation.")
    if all(trace_df["state_hash"] == trace_df["values_hash_from_json"]):
        checks.append("Values JSON reproduces the recorded state hash for every observed trajectory row.")
    else:
        failures.append("At least one observed row has values_json inconsistent with its state hash.")
    if set(observed_df["mixtureId"]) == set(CHIMERIC_MIXTURES):
        checks.append("All planned same-goal chimeric mixtures are represented.")
    else:
        failures.append("The S04 output is missing at least one planned chimeric mixture.")
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
    script_dst = code_dir / "scripts" / "e02_s04_label_shuffle_nulls.py"
    test_dst = code_dir / "tests" / "test_e02_label_shuffle_nulls.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "e02_s04_label_shuffle_nulls.py", script_dst)
    if (REPO_ROOT / "tests" / "test_e02_label_shuffle_nulls.py").exists():
        shutil.copy2(REPO_ROOT / "tests" / "test_e02_label_shuffle_nulls.py", test_dst)
    return [path for path in [script_dst, test_dst] if path.exists()]


def write_validation_report(step_dir: Path, checks: list[str], failures: list[str]) -> Path:
    path = step_dir / "validation_report.md"
    report = [
        "# S04 Validation Report",
        "",
        "- Research step ID: S04",
        "- Completion status: completed" if not failures else "- Completion status: completed with validation failures",
        "- Artifacts written: label-shuffle null tables, observed trajectory tables, figures, copied code, manifests, and status files under `$ARTIFACTS_DIR`.",
        f"- Validation result: {'passed' if not failures else 'failed'}",
        "- Caveats or blockers: label shuffles preserve observed trajectories and test label clustering only; they do not create alternative policy dynamics.",
        "- Recommended next action: proceed to S05 behavior-preserving dummy Algotypes only after Chief Scientist review.",
        "",
        "## Checks",
        "",
    ]
    report.extend(f"- {check}" for check in checks)
    report.extend(["", "## Failures", ""])
    report.extend(f"- {failure}" for failure in failures) if failures else report.append("- None")
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def run_repo_tests(step_dir: Path) -> dict[str, Any]:
    log_path = step_dir / "repo_unit_test_log.txt"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    result = run_command(command, cwd=REPO_ROOT, env=env)
    log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8")
    return {
        "command": command,
        "returnCode": result.returncode,
        "success": bool(result.ok),
        "logPath": str(log_path),
    }


def outcome_classification(observed_df: pd.DataFrame) -> str:
    significant_peak_runs = int(np.sum(observed_df["empiricalPHighPeak"].to_numpy(dtype=float) <= 0.05))
    significant_auc_runs = int(np.sum(observed_df["empiricalPHighAuc"].to_numpy(dtype=float) <= 0.05))
    if significant_peak_runs >= max(1, math.ceil(0.5 * len(observed_df))) or significant_auc_runs >= max(
        1, math.ceil(0.5 * len(observed_df))
    ):
        return "supportive"
    if (observed_df["observedPeakAggregation"] - observed_df["nullPeakMean"]).mean() <= 0:
        return "null"
    return "constraining/contradictory"


def write_reports_and_manifests(
    *,
    artifacts_dir: Path,
    step_dir: Path,
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    timepoint_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    checks: list[str],
    failures: list[str],
    validation: dict[str, Any],
    repo_test_payload: dict[str, Any],
    artifact_paths: list[Path],
    started_at: str,
    n: int,
    replicate_count: int,
    null_replicates: int,
) -> tuple[Path, Path, Path]:
    success = not failures and bool(repo_test_payload["success"])
    classification = outcome_classification(observed_df)
    validation_result = (
        "passed: label counts preserved, positions/values/history unchanged, policy signature unchanged, and repository tests passed"
        if success
        else "failed: see validation_report.md and status.json"
    )
    caveats = [
        "S04 is a trajectory-preserving label null: it tests label clustering on fixed trajectories, not alternative policy dynamics.",
        "Null labels are reassigned to cell identities after the observed run; positions, values, event order, and movement history are not changed.",
        f"The audit uses deterministic n={n} same-goal chimeric trajectories with bounded caps, matching S02/S03 scale rather than full E01 n=100 sweeps.",
        "Empirical p-values are one-sided high-tail tests for Aggregation exceeding shuffled labels and are not yet adjusted across the full E02 family.",
    ]
    recommended_next_action = "Proceed to S05 behavior-preserving dummy Algotype controls only after Chief Scientist review; carry S04 label-null results into later Aggregation verdicts."
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
            ["observed chimeric runs", len(observed_df)],
            ["null replicate rows", len(null_df)],
            ["timepoint summary rows", len(timepoint_df)],
            ["same-goal chimeric mixtures", observed_df["mixtureId"].nunique()],
            ["null replicates per run", null_replicates],
        ],
    )
    preview_table = markdown_table(
        ["Mixture", "Runs", "Obs. peak", "Null peak", "Delta peak", "Median p peak", "Obs. AUC", "Null AUC", "Median p AUC"],
        [
            [
                row.mixtureId,
                int(row.runCount),
                row.meanObservedPeakAggregation,
                row.meanNullPeakMean,
                row.meanPeakDeltaVsNull,
                row.medianPHighPeak,
                row.meanObservedAucAggregation,
                row.meanNullAucMean,
                row.medianPHighAuc,
            ]
            for row in summary_df.itertuples(index=False)
        ],
    )
    max_peak_delta = float((observed_df["observedPeakAggregation"] - observed_df["nullPeakMean"]).max())
    min_peak_p = float(observed_df["empiricalPHighPeak"].min())
    min_auc_p = float(observed_df["empiricalPHighAuc"].min())
    strong_mask = (summary_df["medianPHighPeak"] <= 0.05) | (summary_df["medianPHighAuc"] <= 0.05)
    strong_mixtures = summary_df.loc[strong_mask, "mixtureId"].tolist()
    if strong_mixtures:
        strong_text = "median run-level support was strongest for " + ", ".join(strong_mixtures)
    else:
        strong_text = "no mixture showed median run-level support at the 0.05 high-tail threshold"
    lay_summary = (
        "S04 held every observed chimeric trajectory fixed and only shuffled Algotype labels over cell identities. "
        "This directly tests whether observed Aggregation is larger than expected from the same positions, values, and movement history with labels randomized. "
        f"In this bounded matrix, {strong_text}; pairwise mixtures were therefore not uniformly above the trajectory-preserving label null."
    )
    summary_path.write_text(
        f"""# E02 S04 Status Summary

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

## Label-Shuffle Preview

{preview_table}

## Anchor Notes

- Max observed peak Aggregation delta versus each run's null peak mean: {max_peak_delta}
- Smallest empirical high-tail p-value for peak Aggregation: {min_peak_p}
- Smallest empirical high-tail p-value for Aggregation AUC: {min_auc_p}
- n: {n}
- Replicates per condition: {replicate_count}
- Null replicates per observed run: {null_replicates}
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
        "startedAt": started_at,
        "completedAt": utc_now(),
        "n": int(n),
        "replicateCountPerCondition": int(replicate_count),
        "nullReplicatesPerRun": int(null_replicates),
        "sameGoalChimericMixtures": CHIMERIC_MIXTURES,
        "repoUnitTestCommand": repo_test_payload,
        "validationChecks": checks,
        "validationFailures": failures,
        "validationMatrix": validation,
        "laySummary": lay_summary,
        "maxPeakDeltaVsRunNullMean": max_peak_delta,
        "minEmpiricalPHighPeak": min_peak_p,
        "minEmpiricalPHighAuc": min_auc_p,
    }
    write_json(status_path, status_payload)
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.e02.s04.artifact_manifest.v1",
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
    parser.add_argument("--null-seed-base", type=int, default=404_000)
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
    observed_df, null_df, timepoint_df, trace_df, validation = run_s04_matrix(
        e01_artifacts=args.e01_artifacts_dir,
        n=int(args.n),
        replicate_count=int(args.replicate_count),
        null_replicates=int(args.null_replicates),
        null_seed_base=int(args.null_seed_base),
        max_activations=int(args.max_activations),
        max_swaps=int(args.max_swaps),
        max_comparisons=int(args.max_comparisons),
    )
    table_paths = write_tables(observed_df, null_df, timepoint_df, trace_df, artifacts_dir)
    summary_df = pd.read_parquet(table_paths["mixture_summary_parquet"])
    figure_png, figure_pdf = plot_label_shuffle_summary(observed_df, summary_df, figure_dir)
    validation_success, checks, failures = validate_outputs(observed_df, null_df, timepoint_df, trace_df, validation)
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
        null_df=null_df,
        timepoint_df=timepoint_df,
        summary_df=summary_df,
        checks=checks,
        failures=failures,
        validation=validation,
        repo_test_payload=repo_test_payload,
        artifact_paths=artifact_paths,
        started_at=started_at,
        n=int(args.n),
        replicate_count=int(args.replicate_count),
        null_replicates=int(args.null_replicates),
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
