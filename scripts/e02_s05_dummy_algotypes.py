#!/usr/bin/env python3
"""Execute E02 S05 behavior-preserving dummy Algotype controls.

S05 attaches two-label and three-label dummy identities to cells while every
cell in a run executes the same validated S01 policy. Dummy labels are used only
for Aggregation and activation diagnostics; paired reference runs verify that
labels do not change code paths, activation histories, tie behavior, or values.
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
STEP_ID = "S05"
STEP_NUMBER = 5
DEFAULT_E01_ARTIFACTS = Path("/previous-artifacts/E01")
BASE_POLICIES = ["bubble", "insertion", "selection"]
LABEL_SCHEMES = ["two_label_balanced", "three_label_balanced"]
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
class DummyTrajectory:
    condition: dict[str, Any]
    seed_row: dict[str, Any]
    n: int
    base_policy: str
    label_scheme_id: str
    dummy_labels_by_cell_id: list[str]
    initial_values: list[int]
    final_values: list[int]
    completed: bool
    stop_reason: str
    swap_count: int
    comparison_count: int
    archived_compare_and_swap_count: int
    activation_count: int
    event_count: int
    wall_time_seconds: float
    trace_rows: list[dict[str, Any]]
    activation_events: list[dict[str, Any]]
    behavior_history_hash: str
    activation_history_hash: str
    dummy_curve: np.ndarray
    activation_counts_by_dummy_label: dict[str, int]
    activation_shares_by_dummy_label: dict[str, float]
    target_activation_shares_by_dummy_label: dict[str, float]
    max_activation_share_abs_error: float


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


def pure_condition_id(base_policy: str) -> str:
    return f"S09_cell_view_pure_{base_policy}_unique_same_goal"


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def load_s05_inputs(e01_artifacts: Path, *, replicate_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_ids = [pure_condition_id(policy) for policy in BASE_POLICIES]
    condition_matrix = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "condition_matrix.csv")
    seed_table = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "seed_table.csv")
    conditions = condition_matrix[condition_matrix["conditionId"].isin(condition_ids)].copy()
    conditions = conditions.sort_values("conditionId").reset_index(drop=True)
    seeds = seed_table[
        (seed_table["conditionId"].isin(condition_ids))
        & (seed_table["replicateIndex"] < int(replicate_count))
    ].copy()
    seeds = seeds.sort_values(["conditionId", "replicateIndex"]).reset_index(drop=True)
    if len(conditions) != len(condition_ids):
        missing = sorted(set(condition_ids) - set(conditions["conditionId"]))
        raise ValueError(f"Missing S05 pure policy conditions in E01 matrix: {missing}")
    expected_seed_rows = len(condition_ids) * int(replicate_count)
    if len(seeds) != expected_seed_rows:
        raise ValueError(f"Expected {expected_seed_rows} S05 seed rows, found {len(seeds)}")
    return conditions, seeds


def label_counts_for_scheme(label_scheme_id: str, n: int) -> dict[str, int]:
    if label_scheme_id == "reference_single_label":
        return {"reference": n}
    if label_scheme_id == "two_label_balanced":
        return {"dummy_A": n // 2, "dummy_B": n - n // 2}
    if label_scheme_id == "three_label_balanced":
        base = n // 3
        return {
            "dummy_A": base + (1 if n % 3 > 0 else 0),
            "dummy_B": base + (1 if n % 3 > 1 else 0),
            "dummy_C": base,
        }
    raise ValueError(f"unknown dummy label scheme: {label_scheme_id}")


def dummy_labels_from_seed(label_scheme_id: str, n: int, seed: int) -> list[str]:
    counts = label_counts_for_scheme(label_scheme_id, n)
    labels = [label for label, count in counts.items() for _ in range(count)]
    if len(labels) != n:
        raise ValueError(f"{label_scheme_id} produced {len(labels)} labels, expected {n}")
    if len(counts) > 1:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(labels)
    return labels


def label_codes(labels: list[str]) -> list[int]:
    mapping = {label: index for index, label in enumerate(sorted(set(labels)))}
    return [mapping[label] for label in labels]


def labels_by_position(sim: DeterministicEventSimulator, labels_by_cell_id: list[str]) -> list[str]:
    return [labels_by_cell_id[int(cell.cell_id)] for cell in sim.cells]


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


def aggregation_curve_for_labels(cell_ids_by_event: np.ndarray, labels_by_cell_id: np.ndarray) -> np.ndarray:
    labels_by_position = labels_by_cell_id[cell_ids_by_event]
    if labels_by_position.shape[1] < 2:
        return np.ones(labels_by_position.shape[0], dtype=float)
    return np.mean(labels_by_position[:, :-1] == labels_by_position[:, 1:], axis=1).astype(float)


def fixed_count_expected_aggregation(labels: list[str]) -> float:
    counts = Counter(labels)
    n = len(labels)
    if n < 2:
        return 1.0
    return float(sum(count * (count - 1) for count in counts.values()) / (n * (n - 1)))


def target_activation_shares(labels_by_cell_id: list[str]) -> dict[str, float]:
    counts = Counter(labels_by_cell_id)
    total = len(labels_by_cell_id)
    return {label: count / total for label, count in sorted(counts.items())}


def activation_share_error(
    counts: dict[str, int],
    target_shares: dict[str, float],
    activation_count: int,
) -> tuple[dict[str, float], float, float]:
    if activation_count <= 0:
        shares = {label: 0.0 for label in target_shares}
    else:
        shares = {label: int(counts.get(label, 0)) / activation_count for label in target_shares}
    max_error = max(abs(shares[label] - target_shares[label]) for label in target_shares) if target_shares else 0.0
    tolerance = max(0.08, 4.0 / math.sqrt(max(1, activation_count)))
    return shares, max_error, tolerance


def snapshot_trace_row(
    sim: DeterministicEventSimulator,
    base_row: dict[str, Any],
    *,
    base_policy: str,
    label_scheme_id: str,
    dummy_labels_by_cell_id: list[str],
) -> dict[str, Any]:
    values = sim.current_values()
    dummy_labels = labels_by_position(sim, dummy_labels_by_cell_id)
    policy_algotypes = sim.current_algotypes()
    row = dict(base_row)
    row["research_step_id"] = STEP_ID
    row["policyLogicId"] = POLICY_LOGIC_ID
    row["policySignatureSha256"] = POLICY_SIGNATURE_SHA256
    row["basePolicy"] = base_policy
    row["labelSchemeId"] = label_scheme_id
    row["values_json"] = compact_json(values)
    row["cell_ids_json"] = compact_json(current_cell_ids(sim))
    row["policy_algotypes_json"] = compact_json(policy_algotypes)
    row["dummy_labels_json"] = compact_json(dummy_labels)
    row["dummyAggregation"] = aggregation(dummy_labels)
    row["policyAggregation"] = aggregation(policy_algotypes)
    row["dummyLabelCountsJson"] = compact_json(dict(sorted(Counter(dummy_labels).items())))
    row["values_hash_from_json"] = state_hash(values)
    return row


def behavior_history_hash(trace_rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "eventIndex": int(row["event_index"]),
            "eventKind": row["event_kind"],
            "activationIndex": int(row["activation_index"]),
            "actorCellId": None if pd.isna(row["actor_cell_id"]) else int(row["actor_cell_id"]),
            "targetPosition": None if pd.isna(row["target_position"]) else int(row["target_position"]),
            "swapCount": int(row["swap_count"]),
            "comparisonCount": int(row["comparison_count"]),
            "archivedCompareAndSwapCount": int(row["archived_compare_and_swap_count"]),
            "stateHash": row["state_hash"],
            "values": json.loads(row["values_json"]),
            "cellIds": json.loads(row["cell_ids_json"]),
            "policyAlgotypes": json.loads(row["policy_algotypes_json"]),
        }
        for row in trace_rows
    ]
    return sha256_json(payload)


def activation_history_hash(activation_events: list[dict[str, Any]]) -> str:
    payload = [
        {
            "activationIndex": int(row["activationIndex"]),
            "actorCellId": int(row["actorCellId"]),
            "actorPositionBefore": int(row["actorPositionBefore"]),
            "actorPositionAfter": int(row["actorPositionAfter"]),
            "targetPosition": None if row["targetPosition"] is None else int(row["targetPosition"]),
            "swapped": bool(row["swapped"]),
            "comparisonDelta": int(row["comparisonDelta"]),
            "archivedCompareDelta": int(row["archivedCompareDelta"]),
            "reason": str(row["reason"]),
        }
        for row in activation_events
    ]
    return sha256_json(payload)


def validate_trace_identity(
    trace_rows: list[dict[str, Any]],
    *,
    initial_values: list[int],
    base_policy: str,
    dummy_labels_by_cell_id: list[str],
) -> list[str]:
    failures: list[str] = []
    n = len(initial_values)
    expected_ids = list(range(n))
    expected_label_counts = Counter(dummy_labels_by_cell_id)
    for row in trace_rows:
        event_index = int(row["event_index"])
        values = json.loads(row["values_json"])
        cell_ids = json.loads(row["cell_ids_json"])
        policy_algotypes = json.loads(row["policy_algotypes_json"])
        dummy_labels = json.loads(row["dummy_labels_json"])
        if sorted(cell_ids) != expected_ids:
            failures.append(f"event {event_index}: cell IDs are not a position permutation")
        if values != [initial_values[int(cell_id)] for cell_id in cell_ids]:
            failures.append(f"event {event_index}: values do not match fixed cell identity history")
        if policy_algotypes != [base_policy] * n:
            failures.append(f"event {event_index}: policy Algotypes are not all {base_policy}")
        if dummy_labels != [dummy_labels_by_cell_id[int(cell_id)] for cell_id in cell_ids]:
            failures.append(f"event {event_index}: dummy labels do not follow cell identities")
        if Counter(dummy_labels) != expected_label_counts:
            failures.append(f"event {event_index}: dummy label counts changed")
        if row["state_hash"] != state_hash(values):
            failures.append(f"event {event_index}: state hash does not match values_json")
    return failures


def run_policy_trajectory(
    *,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    n: int,
    base_policy: str,
    label_scheme_id: str,
    dummy_labels_by_cell_id: list[str],
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
    no_move_checks_required: int = 2,
) -> DummyTrajectory:
    input_seed = int(seed_row["inputPermutationSeed"])
    initial_values = initial_values_from_seed(input_seed, n=n, profile=str(condition["inputProfile"]))
    policy_algotypes = [base_policy] * n
    sim = DeterministicEventSimulator(
        initial_values,
        policy_algotypes,
        labels=label_codes(dummy_labels_by_cell_id),
        scheduler_seed=int(seed_row["schedulerSeed"]),
        tie_breaker_seed=int(seed_row["tieBreakerSeed"]),
        condition_id=f"S05_dummy_{base_policy}_{label_scheme_id}",
        research_step_id=STEP_ID,
    )
    started_at = time.perf_counter()
    trace_rows = [
        snapshot_trace_row(
            sim,
            sim.trace_rows[-1],
            base_policy=base_policy,
            label_scheme_id=label_scheme_id,
            dummy_labels_by_cell_id=dummy_labels_by_cell_id,
        )
    ]
    activation_events: list[dict[str, Any]] = []
    activation_counts: Counter[str] = Counter()
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
        if outcome.activated and outcome.actor_cell_id is not None:
            label = dummy_labels_by_cell_id[int(outcome.actor_cell_id)]
            activation_counts[label] += 1
            activation_events.append(
                {
                    "activationIndex": int(sim.activation_count),
                    "actorCellId": int(outcome.actor_cell_id),
                    "actorPositionBefore": int(outcome.actor_position_before),
                    "actorPositionAfter": int(outcome.actor_position_after),
                    "targetPosition": None if outcome.target_position is None else int(outcome.target_position),
                    "swapped": bool(outcome.swapped),
                    "comparisonDelta": int(outcome.comparison_delta),
                    "archivedCompareDelta": int(outcome.archived_compare_delta),
                    "reason": str(outcome.reason),
                    "dummyLabel": label,
                }
            )
        if outcome.swapped:
            trace_rows.append(
                snapshot_trace_row(
                    sim,
                    sim.trace_rows[-1],
                    base_policy=base_policy,
                    label_scheme_id=label_scheme_id,
                    dummy_labels_by_cell_id=dummy_labels_by_cell_id,
                )
            )

    identity_failures = validate_trace_identity(
        trace_rows,
        initial_values=initial_values,
        base_policy=base_policy,
        dummy_labels_by_cell_id=dummy_labels_by_cell_id,
    )
    if identity_failures:
        raise ValueError("; ".join(identity_failures[:5]))
    target_shares = target_activation_shares(dummy_labels_by_cell_id)
    shares, max_error, _ = activation_share_error(dict(activation_counts), target_shares, sim.activation_count)
    dummy_curve = np.asarray([float(row["dummyAggregation"]) for row in trace_rows], dtype=float)
    return DummyTrajectory(
        condition=condition,
        seed_row=seed_row,
        n=n,
        base_policy=base_policy,
        label_scheme_id=label_scheme_id,
        dummy_labels_by_cell_id=dummy_labels_by_cell_id,
        initial_values=initial_values,
        final_values=sim.current_values(),
        completed=sim.is_sorted(),
        stop_reason=stop_reason,
        swap_count=sim.swap_count,
        comparison_count=sim.comparison_count,
        archived_compare_and_swap_count=sim.archived_compare_and_swap_count,
        activation_count=sim.activation_count,
        event_count=len(trace_rows),
        wall_time_seconds=time.perf_counter() - started_at,
        trace_rows=trace_rows,
        activation_events=activation_events,
        behavior_history_hash=behavior_history_hash(trace_rows),
        activation_history_hash=activation_history_hash(activation_events),
        dummy_curve=dummy_curve,
        activation_counts_by_dummy_label=dict(sorted(activation_counts.items())),
        activation_shares_by_dummy_label=shares,
        target_activation_shares_by_dummy_label=target_shares,
        max_activation_share_abs_error=max_error,
    )


def run_dummy_label_nulls(
    trajectory: DummyTrajectory,
    *,
    null_replicates: int,
    null_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(null_seed))
    cell_ids_by_event = np.asarray(
        [json.loads(row["cell_ids_json"]) for row in trajectory.trace_rows],
        dtype=np.int16,
    )
    labels = np.asarray(label_codes(trajectory.dummy_labels_by_cell_id), dtype=np.int16)
    expected_counts = Counter(labels.tolist())
    observed = trajectory.dummy_curve
    null_curves = np.empty((int(null_replicates), len(observed)), dtype=np.float32)
    null_rows: list[dict[str, Any]] = []
    label_count_failures = 0
    for null_index in range(int(null_replicates)):
        labels_by_cell_id = np.array(labels, copy=True)
        rng.shuffle(labels_by_cell_id)
        if Counter(labels_by_cell_id.tolist()) != expected_counts:
            label_count_failures += 1
        curve = aggregation_curve_for_labels(cell_ids_by_event, labels_by_cell_id)
        null_curves[null_index, :] = curve.astype(np.float32)
        null_rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "sourcePureConditionId": trajectory.condition["conditionId"],
                "basePolicy": trajectory.base_policy,
                "labelSchemeId": trajectory.label_scheme_id,
                "replicateIndex": int(trajectory.seed_row["replicateIndex"]),
                "replicateNumber": int(trajectory.seed_row["replicateNumber"]),
                "nullReplicateIndex": int(null_index),
                "nullSeed": int(null_seed),
                "behaviorHistoryHash": trajectory.behavior_history_hash,
                "labelAssignmentHash": sha256_json(labels_by_cell_id.tolist()),
                "nullPeakDummyAggregation": float(np.max(curve)),
                "nullPeakEventIndex": int(np.argmax(curve)),
                "nullFinalDummyAggregation": float(curve[-1]),
                "nullAucDummyAggregation": normalized_auc(curve),
            }
        )
    timepoint_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(trajectory.trace_rows):
        null_values = null_curves[:, idx].astype(float)
        observed_value = float(observed[idx])
        timepoint_rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "sourcePureConditionId": trajectory.condition["conditionId"],
                "basePolicy": trajectory.base_policy,
                "labelSchemeId": trajectory.label_scheme_id,
                "replicateIndex": int(trajectory.seed_row["replicateIndex"]),
                "replicateNumber": int(trajectory.seed_row["replicateNumber"]),
                "eventIndex": int(row["event_index"]),
                "activationIndex": int(row["activation_index"]),
                "swapCount": int(row["swap_count"]),
                "stateHash": row["state_hash"],
                "behaviorHistoryHash": trajectory.behavior_history_hash,
                "observedDummyAggregation": observed_value,
                "nullMeanDummyAggregation": float(np.mean(null_values)),
                "nullSdDummyAggregation": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0,
                "nullQ025DummyAggregation": float(np.quantile(null_values, 0.025)),
                "nullQ50DummyAggregation": float(np.quantile(null_values, 0.5)),
                "nullQ975DummyAggregation": float(np.quantile(null_values, 0.975)),
                "empiricalPHighAtTimepoint": empirical_p_high(observed_value, null_values),
            }
        )
    validation = {
        "labelCountFailures": int(label_count_failures),
        "cellIdPermutationRows": int(np.sum([sorted(row.tolist()) == list(range(trajectory.n)) for row in cell_ids_by_event])),
        "cellIdRows": int(len(cell_ids_by_event)),
        "nullReplicates": int(null_replicates),
    }
    return pd.DataFrame(null_rows), pd.DataFrame(timepoint_rows), validation


def make_observed_record(
    trajectory: DummyTrajectory,
    null_df: pd.DataFrame,
    *,
    reference: DummyTrajectory,
) -> dict[str, Any]:
    curve = trajectory.dummy_curve
    null_peak = null_df["nullPeakDummyAggregation"].to_numpy(dtype=float)
    null_auc = null_df["nullAucDummyAggregation"].to_numpy(dtype=float)
    null_final = null_df["nullFinalDummyAggregation"].to_numpy(dtype=float)
    observed_peak = float(np.max(curve))
    observed_auc = normalized_auc(curve)
    observed_final = float(curve[-1])
    _, _, activation_tolerance = activation_share_error(
        trajectory.activation_counts_by_dummy_label,
        trajectory.target_activation_shares_by_dummy_label,
        trajectory.activation_count,
    )
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "sourcePureConditionId": trajectory.condition["conditionId"],
        "conditionId": f"S05_dummy_{trajectory.base_policy}_{trajectory.label_scheme_id}",
        "basePolicy": trajectory.base_policy,
        "labelSchemeId": trajectory.label_scheme_id,
        "dummyLabelCount": int(len(set(trajectory.dummy_labels_by_cell_id))),
        "policyLogicId": POLICY_LOGIC_ID,
        "policySignatureSha256": POLICY_SIGNATURE_SHA256,
        "n": int(trajectory.n),
        "replicateIndex": int(trajectory.seed_row["replicateIndex"]),
        "replicateNumber": int(trajectory.seed_row["replicateNumber"]),
        "inputPermutationSeed": int(trajectory.seed_row["inputPermutationSeed"]),
        "dummyLabelAssignmentSeed": stable_seed(
            trajectory.condition["conditionId"],
            trajectory.seed_row["replicateIndex"],
            trajectory.label_scheme_id,
            trajectory.n,
        ),
        "schedulerSeed": int(trajectory.seed_row["schedulerSeed"]),
        "tieBreakerSeed": int(trajectory.seed_row["tieBreakerSeed"]),
        "initialValuesHash": state_hash(trajectory.initial_values),
        "finalValuesHash": state_hash(trajectory.final_values),
        "referenceBehaviorHistoryHash": reference.behavior_history_hash,
        "dummyBehaviorHistoryHash": trajectory.behavior_history_hash,
        "referenceActivationHistoryHash": reference.activation_history_hash,
        "dummyActivationHistoryHash": trajectory.activation_history_hash,
        "behaviorHistoryMatchesReference": bool(trajectory.behavior_history_hash == reference.behavior_history_hash),
        "activationHistoryMatchesReference": bool(trajectory.activation_history_hash == reference.activation_history_hash),
        "finalValuesMatchReference": bool(trajectory.final_values == reference.final_values),
        "countsMatchReference": bool(
            trajectory.swap_count == reference.swap_count
            and trajectory.comparison_count == reference.comparison_count
            and trajectory.activation_count == reference.activation_count
        ),
        "dummyLabelsByCellId": compact_json(trajectory.dummy_labels_by_cell_id),
        "dummyLabelCounts": compact_json(dict(sorted(Counter(trajectory.dummy_labels_by_cell_id).items()))),
        "targetActivationSharesByDummyLabel": compact_json(trajectory.target_activation_shares_by_dummy_label),
        "realizedActivationCountsByDummyLabel": compact_json(trajectory.activation_counts_by_dummy_label),
        "realizedActivationSharesByDummyLabel": compact_json(trajectory.activation_shares_by_dummy_label),
        "maxActivationShareAbsError": float(trajectory.max_activation_share_abs_error),
        "activationShareTolerance": float(activation_tolerance),
        "activationShareWithinTolerance": bool(trajectory.max_activation_share_abs_error <= activation_tolerance),
        "completed": bool(trajectory.completed),
        "stopReason": trajectory.stop_reason,
        "swapCount": int(trajectory.swap_count),
        "comparisonCount": int(trajectory.comparison_count),
        "archivedCompareAndSwapCount": int(trajectory.archived_compare_and_swap_count),
        "activationCount": int(trajectory.activation_count),
        "eventCount": int(trajectory.event_count),
        "finalSortednessPercent": sortedness_percent(trajectory.final_values),
        "finalMonotonicityError": monotonicity_error(trajectory.final_values),
        "fixedCountExpectedAggregation": fixed_count_expected_aggregation(trajectory.dummy_labels_by_cell_id),
        "observedInitialDummyAggregation": float(curve[0]),
        "observedPeakDummyAggregation": observed_peak,
        "observedPeakEventIndex": int(np.argmax(curve)),
        "observedFinalDummyAggregation": observed_final,
        "observedAucDummyAggregation": observed_auc,
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


def make_diagnostic_record(trajectory: DummyTrajectory, reference: DummyTrajectory, validation: dict[str, Any]) -> dict[str, Any]:
    policy_paths = sorted({json.loads(row["policy_algotypes_json"])[0] for row in trajectory.trace_rows})
    actor_policy_paths = sorted({str(event.get("reason")) for event in trajectory.activation_events})
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "sourcePureConditionId": trajectory.condition["conditionId"],
        "basePolicy": trajectory.base_policy,
        "labelSchemeId": trajectory.label_scheme_id,
        "replicateIndex": int(trajectory.seed_row["replicateIndex"]),
        "replicateNumber": int(trajectory.seed_row["replicateNumber"]),
        "policyPathSet": compact_json(policy_paths),
        "activationReasonSet": compact_json(actor_policy_paths),
        "allPolicyAlgotypesIdentical": bool(policy_paths == [trajectory.base_policy]),
        "behaviorHistoryMatchesReference": bool(trajectory.behavior_history_hash == reference.behavior_history_hash),
        "activationHistoryMatchesReference": bool(trajectory.activation_history_hash == reference.activation_history_hash),
        "finalValuesMatchReference": bool(trajectory.final_values == reference.final_values),
        "countsMatchReference": bool(
            trajectory.swap_count == reference.swap_count
            and trajectory.comparison_count == reference.comparison_count
            and trajectory.activation_count == reference.activation_count
        ),
        "labelCountFailures": int(validation["labelCountFailures"]),
        "cellIdPermutationRows": int(validation["cellIdPermutationRows"]),
        "cellIdRows": int(validation["cellIdRows"]),
        "maxActivationShareAbsError": float(trajectory.max_activation_share_abs_error),
        "activationShareWithinTolerance": bool(
            trajectory.max_activation_share_abs_error
            <= activation_share_error(
                trajectory.activation_counts_by_dummy_label,
                trajectory.target_activation_shares_by_dummy_label,
                trajectory.activation_count,
            )[2]
        ),
        "behaviorHistoryHash": trajectory.behavior_history_hash,
        "activationHistoryHash": trajectory.activation_history_hash,
        "referenceBehaviorHistoryHash": reference.behavior_history_hash,
        "referenceActivationHistoryHash": reference.activation_history_hash,
    }


def annotate_trace_rows(trajectory: DummyTrajectory) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in trajectory.trace_rows:
        item = dict(row)
        item["experimentId"] = EXPERIMENT_ID
        item["researchStepId"] = STEP_ID
        item["sourcePureConditionId"] = trajectory.condition["conditionId"]
        item["basePolicy"] = trajectory.base_policy
        item["labelSchemeId"] = trajectory.label_scheme_id
        item["replicateIndex"] = int(trajectory.seed_row["replicateIndex"])
        item["replicateNumber"] = int(trajectory.seed_row["replicateNumber"])
        item["inputPermutationSeed"] = int(trajectory.seed_row["inputPermutationSeed"])
        item["behaviorHistoryHash"] = trajectory.behavior_history_hash
        rows.append(item)
    return rows


def run_s05_matrix(
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
    conditions_df, seeds_df = load_s05_inputs(e01_artifacts, replicate_count=replicate_count)
    observed_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    null_dfs: list[pd.DataFrame] = []
    timepoint_dfs: list[pd.DataFrame] = []
    trace_rows: list[dict[str, Any]] = []
    validation_runs: list[dict[str, Any]] = []
    run_index = 0
    for condition_tuple in conditions_df.itertuples(index=False):
        condition = condition_tuple._asdict()
        mixture = str(condition["mixtureId"])
        base_policy = mixture.removeprefix("pure_")
        if base_policy not in BASE_POLICIES:
            continue
        condition_seeds = seeds_df[seeds_df["conditionId"] == condition["conditionId"]].copy()
        for seed_tuple in condition_seeds.itertuples(index=False):
            seed_row = seed_tuple._asdict()
            reference_labels = dummy_labels_from_seed("reference_single_label", n, 0)
            reference = run_policy_trajectory(
                condition=condition,
                seed_row=seed_row,
                n=n,
                base_policy=base_policy,
                label_scheme_id="reference_single_label",
                dummy_labels_by_cell_id=reference_labels,
                max_activations=max_activations,
                max_swaps=max_swaps,
                max_comparisons=max_comparisons,
            )
            sibling_hashes: set[str] = {reference.behavior_history_hash}
            for label_scheme_id in LABEL_SCHEMES:
                assignment_seed = stable_seed(condition["conditionId"], seed_row["replicateIndex"], label_scheme_id, n)
                dummy_labels = dummy_labels_from_seed(label_scheme_id, n, assignment_seed)
                trajectory = run_policy_trajectory(
                    condition=condition,
                    seed_row=seed_row,
                    n=n,
                    base_policy=base_policy,
                    label_scheme_id=label_scheme_id,
                    dummy_labels_by_cell_id=dummy_labels,
                    max_activations=max_activations,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                )
                sibling_hashes.add(trajectory.behavior_history_hash)
                null_seed = int(null_seed_base + run_index * 1_000_003)
                null_df, timepoint_df, run_validation = run_dummy_label_nulls(
                    trajectory,
                    null_replicates=null_replicates,
                    null_seed=null_seed,
                )
                observed_rows.append(make_observed_record(trajectory, null_df, reference=reference))
                diagnostic_rows.append(make_diagnostic_record(trajectory, reference, run_validation))
                null_dfs.append(null_df)
                timepoint_dfs.append(timepoint_df)
                trace_rows.extend(annotate_trace_rows(trajectory))
                validation_runs.append(
                    {
                        "sourcePureConditionId": condition["conditionId"],
                        "basePolicy": base_policy,
                        "labelSchemeId": label_scheme_id,
                        "replicateIndex": int(seed_row["replicateIndex"]),
                        "siblingBehaviorHistoryHashCount": int(len(sibling_hashes)),
                        **run_validation,
                    }
                )
                run_index += 1
    observed_df = pd.DataFrame(observed_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    null_df = pd.concat(null_dfs, ignore_index=True) if null_dfs else pd.DataFrame()
    timepoint_df = pd.concat(timepoint_dfs, ignore_index=True) if timepoint_dfs else pd.DataFrame()
    trace_df = pd.DataFrame(trace_rows)
    validation = {
        "basePolicyCount": len(BASE_POLICIES),
        "labelSchemeCount": len(LABEL_SCHEMES),
        "replicateCountPerCondition": int(replicate_count),
        "nullReplicatesPerRun": int(null_replicates),
        "expectedObservedRows": int(len(BASE_POLICIES) * len(LABEL_SCHEMES) * replicate_count),
        "expectedNullRows": int(len(BASE_POLICIES) * len(LABEL_SCHEMES) * replicate_count * null_replicates),
        "runs": validation_runs,
    }
    return observed_df, diagnostics_df, null_df, timepoint_df, trace_df, validation


def summarize_dummy_runs(observed_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        observed_df.groupby(["basePolicy", "labelSchemeId", "dummyLabelCount"], dropna=False)
        .agg(
            runCount=("conditionId", "count"),
            completedCount=("completed", "sum"),
            completionRate=("completed", "mean"),
            meanObservedPeakDummyAggregation=("observedPeakDummyAggregation", "mean"),
            meanNullPeakMean=("nullPeakMean", "mean"),
            meanPeakDeltaVsNull=("observedPeakDummyAggregation", lambda x: float("nan")),
            medianPHighPeak=("empiricalPHighPeak", "median"),
            minPHighPeak=("empiricalPHighPeak", "min"),
            meanObservedAucDummyAggregation=("observedAucDummyAggregation", "mean"),
            meanNullAucMean=("nullAucMean", "mean"),
            meanAucDeltaVsNull=("observedAucDummyAggregation", lambda x: float("nan")),
            medianPHighAuc=("empiricalPHighAuc", "median"),
            minPHighAuc=("empiricalPHighAuc", "min"),
            meanObservedFinalDummyAggregation=("observedFinalDummyAggregation", "mean"),
            meanNullFinalMean=("nullFinalMean", "mean"),
            medianPHighFinal=("empiricalPHighFinal", "median"),
            meanFixedCountExpectedAggregation=("fixedCountExpectedAggregation", "mean"),
            significantPeakRuns=("empiricalPHighPeak", lambda x: int(np.sum(np.asarray(x, dtype=float) <= 0.05))),
            significantAucRuns=("empiricalPHighAuc", lambda x: int(np.sum(np.asarray(x, dtype=float) <= 0.05))),
            maxActivationShareAbsError=("maxActivationShareAbsError", "max"),
            allBehaviorHistoryMatched=("behaviorHistoryMatchesReference", "all"),
            allActivationHistoryMatched=("activationHistoryMatchesReference", "all"),
        )
        .reset_index()
    )
    summary["meanPeakDeltaVsNull"] = summary["meanObservedPeakDummyAggregation"] - summary["meanNullPeakMean"]
    summary["meanAucDeltaVsNull"] = summary["meanObservedAucDummyAggregation"] - summary["meanNullAucMean"]
    summary["meanFinalDeltaVsNull"] = summary["meanObservedFinalDummyAggregation"] - summary["meanNullFinalMean"]
    return summary


def context_comparison(summary_df: pd.DataFrame, e01_artifacts: Path, artifacts_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    e01_null = pd.read_parquet(e01_artifacts / "results" / "e01_aggregation_null_baselines.parquet")
    s04_summary_path = artifacts_dir / "results" / "e02_label_shuffle_mixture_summary.parquet"
    s04_summary = pd.read_parquet(s04_summary_path) if s04_summary_path.exists() else pd.DataFrame()
    for card, group in summary_df.groupby("dummyLabelCount"):
        if int(card) == 2:
            e01_match = e01_null[e01_null["mixtureId"].isin(["bubble_insertion", "bubble_selection", "insertion_selection"])]
            s04_match = s04_summary[s04_summary["mixtureId"].isin(["bubble_insertion", "bubble_selection", "insertion_selection"])]
            label_family = "two_label_pairwise"
        else:
            e01_match = e01_null[e01_null["mixtureId"] == "bubble_insertion_selection"]
            s04_match = s04_summary[s04_summary["mixtureId"] == "bubble_insertion_selection"]
            label_family = "three_label"
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "labelFamily": label_family,
                "dummyLabelCount": int(card),
                "dummyMeanPeakAggregation": float(group["meanObservedPeakDummyAggregation"].mean()),
                "dummyMeanAucAggregation": float(group["meanObservedAucDummyAggregation"].mean()),
                "dummyMedianPeakPHigh": float(group["medianPHighPeak"].median()),
                "dummyMedianAucPHigh": float(group["medianPHighAuc"].median()),
                "e01FixedCountRandomNullMean": float(e01_match["fixedCountRandomNullMean"].mean()) if len(e01_match) else None,
                "s04RealMixedMeanPeakAggregation": float(s04_match["meanObservedPeakAggregation"].mean()) if len(s04_match) else None,
                "s04RealMixedMedianPeakPHigh": float(s04_match["medianPHighPeak"].median()) if len(s04_match) else None,
                "interpretation": "dummy labels execute one policy; S04 labels execute mixed policies",
            }
        )
    return pd.DataFrame(rows)


def write_tables(
    observed_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    null_df: pd.DataFrame,
    timepoint_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    e01_artifacts: Path,
    artifacts_dir: Path,
) -> dict[str, Path]:
    results_dir = artifacts_dir / "results"
    traces_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    summary_df = summarize_dummy_runs(observed_df)
    context_df = context_comparison(summary_df, e01_artifacts, artifacts_dir)
    paths = {
        "observed_parquet": results_dir / "e02_dummy_algotypes.parquet",
        "observed_csv": results_dir / "e02_dummy_algotypes.csv",
        "summary_parquet": results_dir / "e02_dummy_algotypes_summary.parquet",
        "summary_csv": results_dir / "e02_dummy_algotypes_summary.csv",
        "diagnostics_parquet": results_dir / "e02_dummy_algotypes_diagnostics.parquet",
        "diagnostics_csv": results_dir / "e02_dummy_algotypes_diagnostics.csv",
        "nulls_parquet": results_dir / "e02_dummy_algotypes_nulls.parquet",
        "nulls_csv": results_dir / "e02_dummy_algotypes_nulls.csv",
        "timepoint_parquet": results_dir / "e02_dummy_algotypes_timepoint_summary.parquet",
        "timepoint_csv": results_dir / "e02_dummy_algotypes_timepoint_summary.csv",
        "context_parquet": results_dir / "e02_dummy_algotypes_context_comparison.parquet",
        "context_csv": results_dir / "e02_dummy_algotypes_context_comparison.csv",
        "trace_parquet": traces_dir / "e02_dummy_algotypes_trace_events.parquet",
        "trace_csv_gz": traces_dir / "e02_dummy_algotypes_trace_events.csv.gz",
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


def plot_dummy_summary(summary_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_df = summary_df.sort_values(["labelSchemeId", "basePolicy"])
    labels = [f"{row.basePolicy}\n{row.labelSchemeId.replace('_balanced', '').replace('_', ' ')}" for row in summary_df.itertuples(index=False)]
    x = np.arange(len(summary_df), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(x - 0.18, summary_df["meanObservedPeakDummyAggregation"], width=0.36, color="#40798c", label="observed dummy peak")
    axes[0].bar(x + 0.18, summary_df["meanNullPeakMean"], width=0.36, color="#d08c60", label="label-shuffle peak mean")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Aggregation")
    axes[0].set_title("Dummy Label Peak Aggregation")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.18, summary_df["meanObservedAucDummyAggregation"], width=0.36, color="#4b4e6d", label="observed dummy AUC")
    axes[1].bar(x + 0.18, summary_df["meanNullAucMean"], width=0.36, color="#8baaad", label="label-shuffle AUC mean")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Normalized event-index AUC")
    axes[1].set_title("Dummy Label Aggregation AUC")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    png = figure_dir / "e02_dummy_algotypes_summary.png"
    pdf = figure_dir / "e02_dummy_algotypes_summary.pdf"
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
    expected_observed = int(validation["expectedObservedRows"])
    expected_null = int(validation["expectedNullRows"])
    if len(observed_df) == expected_observed:
        checks.append(f"Observed dummy-label row count matched expected {expected_observed}.")
    else:
        failures.append(f"Observed rows {len(observed_df)} did not match expected {expected_observed}.")
    if len(null_df) == expected_null:
        checks.append(f"Dummy-label null row count matched expected {expected_null}.")
    else:
        failures.append(f"Null rows {len(null_df)} did not match expected {expected_null}.")
    if len(timepoint_df) == len(trace_df):
        checks.append("Timepoint null summaries align one-to-one with dummy trace rows.")
    else:
        failures.append("Timepoint summary rows do not align with dummy trace rows.")
    if observed_df["completed"].all() and set(observed_df["stopReason"]) == {"sorted"}:
        checks.append("All dummy-label runs completed with sorted final states.")
    else:
        failures.append("At least one dummy-label run did not complete sorted.")
    if observed_df["behaviorHistoryMatchesReference"].all() and diagnostics_df["behaviorHistoryMatchesReference"].all():
        checks.append("Dummy labels did not change behavior-history hashes versus paired uniform-label references.")
    else:
        failures.append("At least one dummy run changed the behavior-history hash.")
    if observed_df["activationHistoryMatchesReference"].all() and diagnostics_df["activationHistoryMatchesReference"].all():
        checks.append("Dummy labels did not change activation/tie history hashes versus paired references.")
    else:
        failures.append("At least one dummy run changed the activation/tie history hash.")
    if observed_df["countsMatchReference"].all() and diagnostics_df["countsMatchReference"].all():
        checks.append("Swap, comparison, and activation counts matched paired references for every dummy run.")
    else:
        failures.append("At least one dummy run had counts different from its paired reference.")
    if observed_df["activationShareWithinTolerance"].all() and diagnostics_df["activationShareWithinTolerance"].all():
        checks.append("Realized dummy-label activation shares matched label-count targets within tolerance.")
    else:
        failures.append("At least one dummy-label activation share missed the tolerance.")
    if diagnostics_df["allPolicyAlgotypesIdentical"].all():
        checks.append("Every S05 run used one policy Algotype code path; labels were diagnostic only.")
    else:
        failures.append("At least one S05 run used multiple policy Algotype paths.")
    if all(int(run["labelCountFailures"]) == 0 for run in validation.get("runs", [])):
        checks.append("All dummy-label null assignments preserved dummy label counts.")
    else:
        failures.append("At least one dummy-label null assignment changed label counts.")
    if all(int(run["cellIdPermutationRows"]) == int(run["cellIdRows"]) for run in validation.get("runs", [])):
        checks.append("All dummy trace rows preserved cell-position permutations.")
    else:
        failures.append("At least one dummy trace row lost the fixed cell-position permutation.")
    if all(trace_df["state_hash"] == trace_df["values_hash_from_json"]):
        checks.append("Values JSON reproduces the recorded state hash for every dummy trace row.")
    else:
        failures.append("At least one dummy trace row has values_json inconsistent with its state hash.")
    if set(observed_df["basePolicy"]) == set(BASE_POLICIES) and set(observed_df["labelSchemeId"]) == set(LABEL_SCHEMES):
        checks.append("All planned base policies and dummy label schemes are represented.")
    else:
        failures.append("The S05 output is missing a planned base policy or label scheme.")
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
    script_dst = code_dir / "scripts" / "e02_s05_dummy_algotypes.py"
    test_dst = code_dir / "tests" / "test_e02_dummy_algotypes.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "e02_s05_dummy_algotypes.py", script_dst)
    if (REPO_ROOT / "tests" / "test_e02_dummy_algotypes.py").exists():
        shutil.copy2(REPO_ROOT / "tests" / "test_e02_dummy_algotypes.py", test_dst)
    return [path for path in [script_dst, test_dst] if path.exists()]


def write_validation_report(step_dir: Path, checks: list[str], failures: list[str]) -> Path:
    path = step_dir / "validation_report.md"
    report = [
        "# S05 Validation Report",
        "",
        "- Research step ID: S05",
        "- Completion status: completed" if not failures else "- Completion status: completed with validation failures",
        "- Artifacts written: dummy-label result tables, diagnostics, null tables, trace events, figures, copied code, manifests, and status files under `$ARTIFACTS_DIR`.",
        f"- Validation result: {'passed' if not failures else 'failed'}",
        "- Caveats or blockers: dummy labels are diagnostic identities over one shared policy path; this does not model mixed-policy dynamics.",
        "- Recommended next action: proceed to S06 speed-matched Algotype tests only after Chief Scientist review.",
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


def outcome_classification(summary_df: pd.DataFrame, success: bool) -> str:
    if not success:
        return "constraining/contradictory"
    persistent_signal = bool(
        ((summary_df["medianPHighPeak"] <= 0.05) | (summary_df["medianPHighAuc"] <= 0.05)).any()
    )
    return "constraining/contradictory" if persistent_signal else "supportive"


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
) -> tuple[Path, Path, Path]:
    success = not failures and bool(repo_test_payload["success"])
    classification = outcome_classification(summary_df, success)
    validation_result = (
        "passed: identical code paths, activation/tie histories, activation shares, and label-count conservation verified; repository tests passed"
        if success
        else "failed: see validation_report.md and status.json"
    )
    caveats = [
        "Dummy labels are diagnostic identities attached to cells; they do not alter simulator policy code.",
        "S05 tests label and metric artifacts under single-policy dynamics, not mixed-policy chimeric governance.",
        "Activation-share checks are finite-run binomial-tolerant diagnostics, not exact equal-frequency guarantees.",
        f"S05 uses deterministic n={n} bounded controls to match S02/S03/S04 audit scale rather than full E01 n=100 sweeps.",
    ]
    recommended_next_action = "Proceed to S06 speed-matched Algotype tests only after Chief Scientist review; carry S05 dummy-label null baselines into Aggregation verdicts."
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
            ["dummy observed runs", len(observed_df)],
            ["dummy null rows", len(null_df)],
            ["base policies", observed_df["basePolicy"].nunique()],
            ["label schemes", observed_df["labelSchemeId"].nunique()],
            ["null replicates per run", null_replicates],
        ],
    )
    preview_table = markdown_table(
        ["Policy", "Scheme", "Runs", "Peak delta", "Median p peak", "AUC delta", "Median p AUC", "Histories matched"],
        [
            [
                row.basePolicy,
                row.labelSchemeId,
                int(row.runCount),
                row.meanPeakDeltaVsNull,
                row.medianPHighPeak,
                row.meanAucDeltaVsNull,
                row.medianPHighAuc,
                bool(row.allBehaviorHistoryMatched and row.allActivationHistoryMatched),
            ]
            for row in summary_df.itertuples(index=False)
        ],
    )
    persistent_rows = summary_df[(summary_df["medianPHighPeak"] <= 0.05) | (summary_df["medianPHighAuc"] <= 0.05)]
    if len(persistent_rows):
        signal_text = "persistent dummy-label Aggregation was detected in " + ", ".join(
            f"{row.basePolicy}/{row.labelSchemeId}" for row in persistent_rows.itertuples(index=False)
        )
    else:
        signal_text = "no base-policy/label-scheme group showed persistent high-tail Aggregation above its label-shuffle null"
    lay_summary = (
        "S05 ran two-label and three-label dummy identities on top of single-policy Bubble, Insertion, and Selection trajectories. "
        "Every dummy run matched a paired uniform-label reference in behavior history and activation/tie history, so labels were diagnostic only. "
        f"In this bounded matrix, {signal_text}."
    )
    summary_path.write_text(
        f"""# E02 S05 Status Summary

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

## Dummy-Control Preview

{preview_table}

## Anchor Notes

- Maximum activation-share absolute error: {float(observed_df['maxActivationShareAbsError'].max())}
- Minimum empirical high-tail p-value for peak dummy Aggregation: {float(observed_df['empiricalPHighPeak'].min())}
- Minimum empirical high-tail p-value for dummy Aggregation AUC: {float(observed_df['empiricalPHighAuc'].min())}
- Behavior histories matched paired references: {bool(diagnostics_df['behaviorHistoryMatchesReference'].all())}
- Activation/tie histories matched paired references: {bool(diagnostics_df['activationHistoryMatchesReference'].all())}
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
        "laySummary": lay_summary,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "n": int(n),
        "replicateCountPerCondition": int(replicate_count),
        "nullReplicatesPerRun": int(null_replicates),
        "basePolicies": BASE_POLICIES,
        "labelSchemes": LABEL_SCHEMES,
        "repoUnitTestCommand": repo_test_payload,
        "validationChecks": checks,
        "validationFailures": failures,
        "validationMatrix": validation,
        "maxActivationShareAbsError": float(observed_df["maxActivationShareAbsError"].max()),
        "minEmpiricalPHighPeak": float(observed_df["empiricalPHighPeak"].min()),
        "minEmpiricalPHighAuc": float(observed_df["empiricalPHighAuc"].min()),
        "allBehaviorHistoriesMatchedReference": bool(diagnostics_df["behaviorHistoryMatchesReference"].all()),
        "allActivationHistoriesMatchedReference": bool(diagnostics_df["activationHistoryMatchesReference"].all()),
    }
    write_json(status_path, status_payload)
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.e02.s05.artifact_manifest.v1",
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
    parser.add_argument("--null-seed-base", type=int, default=505_000)
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
    observed_df, diagnostics_df, null_df, timepoint_df, trace_df, validation = run_s05_matrix(
        e01_artifacts=args.e01_artifacts_dir,
        n=int(args.n),
        replicate_count=int(args.replicate_count),
        null_replicates=int(args.null_replicates),
        null_seed_base=int(args.null_seed_base),
        max_activations=int(args.max_activations),
        max_swaps=int(args.max_swaps),
        max_comparisons=int(args.max_comparisons),
    )
    table_paths = write_tables(
        observed_df,
        diagnostics_df,
        null_df,
        timepoint_df,
        trace_df,
        args.e01_artifacts_dir,
        artifacts_dir,
    )
    summary_df = pd.read_parquet(table_paths["summary_parquet"])
    figure_png, figure_pdf = plot_dummy_summary(summary_df, figure_dir)
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
