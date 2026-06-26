#!/usr/bin/env python3
"""Execute E02 S08 Delayed Gratification matched trajectory nulls.

S08 uses the E01 Delayed Gratification convention and tests whether observed
DG in S07 real-policy trajectories exceeds null trajectories matched on start
Sortedness, end Sortedness, swap/event count, and Sortedness noise amplitude.
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


EXPERIMENT_ID = "E02"
STEP_ID = "S08"
STEP_NUMBER = 8
DEFAULT_S07_SOURCES = Path("/artifacts/results/e02_local_move_null_sources.parquet")
DEFAULT_S07_TRACE = Path("/artifacts/traces/e02/S07/e02_local_move_null_trace_events.parquet")
NULL_MODEL = "matched_delta_shuffle"
DG_FUNCTION_ID = "E01_S08_delayed_gratification_from_sortedness"


@dataclass
class CommandResult:
    args: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    ok: bool


@dataclass
class SourceTrajectory:
    source_run_id: str
    source_context: str
    source_condition_id: str
    mixture_id: str
    algorithm: str
    label_source: str
    n: int
    replicate_index: int
    replicate_number: int
    frozen_variant: str
    frozen_count: int
    stop_reason: str
    completed: bool
    source_trajectory_hash: str
    raw_sortedness: np.ndarray
    swap_counts: np.ndarray


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
    payload = json.dumps(json_ready(parts), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


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


def dedup_consecutive(values: list[float] | np.ndarray, tolerance: float = 1e-12) -> list[float]:
    deduped: list[float] = []
    for raw_value in values:
        value = float(raw_value)
        if not deduped or not math.isclose(value, deduped[-1], rel_tol=0.0, abs_tol=tolerance):
            deduped.append(value)
    return deduped


def signed_segments(values: list[float] | np.ndarray, tolerance: float = 1e-12) -> list[float]:
    deduped = dedup_consecutive(values, tolerance=tolerance)
    segments: list[float] = []
    for previous, current in zip(deduped, deduped[1:]):
        delta = float(current - previous)
        if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=tolerance):
            continue
        if segments and segments[-1] * delta > 0:
            segments[-1] += delta
        else:
            segments.append(delta)
    return segments


def delayed_gratification_from_sortedness(values: list[float] | np.ndarray) -> dict[str, Any]:
    """Compute the E01 S08 paper/code DG convention from Sortedness."""

    segments = signed_segments(values)
    event_scores: list[float] = []
    drops: list[float] = []
    recoveries: list[float] = []
    event_segment_indices: list[int] = []

    index = 0
    while index < len(segments) and segments[index] >= 0:
        index += 1

    while index < len(segments) - 1:
        drop_segment = segments[index]
        recovery_segment = segments[index + 1]
        if drop_segment < 0 and recovery_segment > 0:
            drop = -float(drop_segment)
            recovery = float(recovery_segment)
            event_scores.append((recovery - drop) / drop if drop else 0.0)
            drops.append(drop)
            recoveries.append(recovery)
            event_segment_indices.append(index)
            index += 2
        else:
            index += 1

    return {
        "delayedGratification": float(np.mean(event_scores)) if event_scores else 0.0,
        "dgEventCount": int(len(event_scores)),
        "dgPositiveEventCount": int(sum(1 for score in event_scores if score > 0)),
        "dgNegativeEventCount": int(sum(1 for score in event_scores if score < 0)),
        "dgZeroEventCount": int(sum(1 for score in event_scores if math.isclose(score, 0.0, abs_tol=1e-12))),
        "dgTotalDrop": float(sum(drops)),
        "dgTotalRecovery": float(sum(recoveries)),
        "dgMeanDrop": float(np.mean(drops)) if drops else 0.0,
        "dgMeanRecovery": float(np.mean(recoveries)) if recoveries else 0.0,
        "dgMaxEventScore": float(max(event_scores)) if event_scores else 0.0,
        "dgMinEventScore": float(min(event_scores)) if event_scores else 0.0,
        "dgSignedEventScoresJson": json.dumps([round(score, 12) for score in event_scores], separators=(",", ":")),
        "dgSignedSegmentsJson": json.dumps([round(segment, 12) for segment in segments], separators=(",", ":")),
        "dgEventSegmentIndicesJson": json.dumps(event_segment_indices, separators=(",", ":")),
    }


def run_dg_unit_cases() -> tuple[bool, pd.DataFrame]:
    cases = [
        ("empty", [], 0.0, 0),
        ("single_point", [50.0], 0.0, 0),
        ("monotone_increase", [40.0, 50.0, 60.0], 0.0, 0),
        ("single_complete_gain", [50.0, 60.0, 55.0, 70.0], 2.0, 1),
        ("single_incomplete_recovery", [50.0, 60.0, 55.0, 58.0], -0.4, 1),
        ("two_equal_gain_events", [50.0, 60.0, 55.0, 70.0, 65.0, 80.0], 2.0, 2),
        ("deduplicated_plateaus", [50.0, 50.0, 60.0, 60.0, 55.0, 55.0, 70.0], 2.0, 1),
        ("starts_with_drop", [60.0, 55.0, 70.0], 2.0, 1),
        ("trailing_unrecovered_drop", [50.0, 60.0, 55.0], 0.0, 0),
    ]
    rows: list[dict[str, Any]] = []
    for case_id, trajectory, expected_dg, expected_events in cases:
        observed = delayed_gratification_from_sortedness(trajectory)
        observed_dg = float(observed["delayedGratification"])
        observed_events = int(observed["dgEventCount"])
        passed = math.isclose(observed_dg, expected_dg, rel_tol=1e-12, abs_tol=1e-12) and (
            observed_events == expected_events
        )
        rows.append(
            {
                "caseId": case_id,
                "expectedDg": float(expected_dg),
                "observedDg": observed_dg,
                "expectedEvents": int(expected_events),
                "observedEvents": observed_events,
                "passed": bool(passed),
            }
        )
    df = pd.DataFrame(rows)
    return bool(df["passed"].all()), df


def sortedness_percent_from_raw(raw: np.ndarray, n: int) -> np.ndarray:
    if int(n) <= 1:
        return np.full(len(raw), 100.0, dtype=float)
    return raw.astype(float) * 100.0 / (int(n) - 1)


def raw_path_from_deltas(start_raw: int, deltas: np.ndarray) -> np.ndarray:
    path = np.empty(len(deltas) + 1, dtype=np.int16)
    path[0] = int(start_raw)
    if len(deltas):
        path[1:] = int(start_raw) + np.cumsum(deltas, dtype=np.int64)
    return path


def raw_path_in_bounds(path: np.ndarray, n: int) -> bool:
    return bool(np.all(path >= 0) and np.all(path <= int(n) - 1))


def delta_total_variation(deltas: np.ndarray) -> int:
    return int(np.sum(np.abs(deltas.astype(np.int64))))


def trajectory_hash(raw_sortedness: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(raw_sortedness, dtype=np.int16).tobytes()).hexdigest()


def shuffled_delta_bridge(
    observed_raw: np.ndarray,
    *,
    n: int,
    rng: np.random.Generator,
    mix_multiplier: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    deltas = np.diff(observed_raw.astype(np.int16)).astype(np.int16)
    if len(deltas) <= 1:
        return observed_raw.copy(), {"mixAttemptCount": 0, "mixAcceptedCount": 0, "nullEqualsObserved": True}
    start_raw = int(observed_raw[0])
    max_attempts = max(50, int(mix_multiplier) * 25)
    for attempt in range(1, max_attempts + 1):
        shuffled = rng.permutation(deltas).astype(np.int16, copy=False)
        candidate = raw_path_from_deltas(start_raw, shuffled)
        if raw_path_in_bounds(candidate, n):
            return candidate, {
                "mixAttemptCount": int(attempt),
                "mixAcceptedCount": 1,
                "nullEqualsObserved": bool(np.array_equal(candidate, observed_raw)),
            }
    return observed_raw.copy(), {
        "mixAttemptCount": int(max_attempts),
        "mixAcceptedCount": 0,
        "nullEqualsObserved": True,
    }


def load_source_trajectories(s07_sources_path: Path, s07_trace_path: Path) -> list[SourceTrajectory]:
    source_df = pd.read_parquet(s07_sources_path)
    source_meta = {str(row.sourceRunId): row for row in source_df.itertuples(index=False)}
    trace = pd.read_parquet(
        s07_trace_path,
        columns=[
            "sourceRunId",
            "sourceContext",
            "sourceConditionId",
            "mixtureId",
            "frozenVariant",
            "frozenCount",
            "nullModel",
            "eventIndex",
            "swapCount",
            "sortednessRawCount",
            "sourceTrajectoryHash",
        ],
    )
    trace = trace[trace["nullModel"] == "real_policy_source"].copy()
    sources: list[SourceTrajectory] = []
    for source_run_id, group in trace.groupby("sourceRunId", sort=False):
        meta = source_meta[str(source_run_id)]
        ordered = group.sort_values("eventIndex")
        raw = ordered["sortednessRawCount"].to_numpy(dtype=np.int16)
        swaps = ordered["swapCount"].to_numpy(dtype=np.int32)
        sources.append(
            SourceTrajectory(
                source_run_id=str(source_run_id),
                source_context=str(meta.sourceContext),
                source_condition_id=str(meta.sourceConditionId),
                mixture_id=str(meta.mixtureId),
                algorithm=str(meta.algorithm),
                label_source=str(meta.labelSource),
                n=int(meta.n),
                replicate_index=int(meta.replicateIndex),
                replicate_number=int(meta.replicateNumber),
                frozen_variant=str(meta.frozenVariant),
                frozen_count=int(meta.frozenCount),
                stop_reason=str(meta.stopReason),
                completed=bool(meta.completed),
                source_trajectory_hash=str(meta.sourceTrajectoryHash),
                raw_sortedness=raw,
                swap_counts=swaps,
            )
        )
    return sources


def dg_metrics_for_raw(raw_sortedness: np.ndarray, n: int) -> dict[str, Any]:
    percent = sortedness_percent_from_raw(raw_sortedness, n)
    return delayed_gratification_from_sortedness(percent)


def source_observed_record(source: SourceTrajectory) -> dict[str, Any]:
    raw = source.raw_sortedness
    deltas = np.diff(raw.astype(np.int16)).astype(np.int16)
    metrics = dg_metrics_for_raw(raw, source.n)
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "sourceRunId": source.source_run_id,
        "sourceContext": source.source_context,
        "sourceConditionId": source.source_condition_id,
        "mixtureId": source.mixture_id,
        "algorithm": source.algorithm,
        "labelSource": source.label_source,
        "n": int(source.n),
        "replicateIndex": int(source.replicate_index),
        "replicateNumber": int(source.replicate_number),
        "frozenVariant": source.frozen_variant,
        "frozenCount": int(source.frozen_count),
        "completed": bool(source.completed),
        "stopReason": source.stop_reason,
        "sourceTrajectoryHash": source.source_trajectory_hash,
        "trajectoryHashFromRawSortedness": trajectory_hash(raw),
        "swapCount": int(source.swap_counts[-1]) if len(source.swap_counts) else 0,
        "trajectoryEventCount": int(len(raw)),
        "initialSortednessRaw": int(raw[0]) if len(raw) else 0,
        "finalSortednessRaw": int(raw[-1]) if len(raw) else 0,
        "initialSortednessPercent": float(sortedness_percent_from_raw(raw[:1], source.n)[0]) if len(raw) else 0.0,
        "finalSortednessPercent": float(sortedness_percent_from_raw(raw[-1:], source.n)[0]) if len(raw) else 0.0,
        "noiseTotalVariationRaw": delta_total_variation(deltas),
        "noiseTotalVariationPercent": float(delta_total_variation(deltas) * 100.0 / max(1, source.n - 1)),
        "dgFunctionId": DG_FUNCTION_ID,
        "delayedGratification": float(metrics["delayedGratification"]),
        "dgEventCount": int(metrics["dgEventCount"]),
        "dgPositiveEventCount": int(metrics["dgPositiveEventCount"]),
        "dgNegativeEventCount": int(metrics["dgNegativeEventCount"]),
        "dgZeroEventCount": int(metrics["dgZeroEventCount"]),
        "dgTotalDrop": float(metrics["dgTotalDrop"]),
        "dgTotalRecovery": float(metrics["dgTotalRecovery"]),
        "dgMeanDrop": float(metrics["dgMeanDrop"]),
        "dgMeanRecovery": float(metrics["dgMeanRecovery"]),
        "dgMaxEventScore": float(metrics["dgMaxEventScore"]),
        "dgMinEventScore": float(metrics["dgMinEventScore"]),
        "dgSignedEventScoresJson": metrics["dgSignedEventScoresJson"],
        "dgSignedSegmentsJson": metrics["dgSignedSegmentsJson"],
        "dgEventSegmentIndicesJson": metrics["dgEventSegmentIndicesJson"],
    }


def null_record_for_source(
    source: SourceTrajectory,
    *,
    null_replicate_index: int,
    null_seed: int,
    mix_multiplier: int,
    observed_record: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(int(null_seed))
    raw_null, mix_info = shuffled_delta_bridge(
        source.raw_sortedness,
        n=source.n,
        rng=rng,
        mix_multiplier=mix_multiplier,
    )
    observed_raw = source.raw_sortedness
    observed_deltas = np.diff(observed_raw.astype(np.int16)).astype(np.int16)
    null_deltas = np.diff(raw_null.astype(np.int16)).astype(np.int16)
    metrics = dg_metrics_for_raw(raw_null, source.n)
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "sourceRunId": source.source_run_id,
        "sourceContext": source.source_context,
        "sourceConditionId": source.source_condition_id,
        "mixtureId": source.mixture_id,
        "algorithm": source.algorithm,
        "labelSource": source.label_source,
        "n": int(source.n),
        "replicateIndex": int(source.replicate_index),
        "replicateNumber": int(source.replicate_number),
        "frozenVariant": source.frozen_variant,
        "frozenCount": int(source.frozen_count),
        "nullModel": NULL_MODEL,
        "nullReplicateIndex": int(null_replicate_index),
        "nullSeed": int(null_seed),
        "sourceTrajectoryHash": source.source_trajectory_hash,
        "nullTrajectoryHash": trajectory_hash(raw_null),
        "swapCount": int(source.swap_counts[-1]) if len(source.swap_counts) else 0,
        "trajectoryEventCount": int(len(raw_null)),
        "initialSortednessRaw": int(raw_null[0]) if len(raw_null) else 0,
        "finalSortednessRaw": int(raw_null[-1]) if len(raw_null) else 0,
        "initialSortednessPercent": float(sortedness_percent_from_raw(raw_null[:1], source.n)[0]) if len(raw_null) else 0.0,
        "finalSortednessPercent": float(sortedness_percent_from_raw(raw_null[-1:], source.n)[0]) if len(raw_null) else 0.0,
        "noiseTotalVariationRaw": delta_total_variation(null_deltas),
        "noiseTotalVariationPercent": float(delta_total_variation(null_deltas) * 100.0 / max(1, source.n - 1)),
        "matchedStartSortedness": bool(len(raw_null) == len(observed_raw) and int(raw_null[0]) == int(observed_raw[0])),
        "matchedEndSortedness": bool(len(raw_null) == len(observed_raw) and int(raw_null[-1]) == int(observed_raw[-1])),
        "matchedSwapCount": bool(len(raw_null) == len(observed_raw)),
        "matchedNoiseAmplitude": bool(delta_total_variation(null_deltas) == delta_total_variation(observed_deltas)),
        "boundedSortedness": raw_path_in_bounds(raw_null, source.n),
        "matchedSignedDeltaMultiset": bool(sorted(null_deltas.tolist()) == sorted(observed_deltas.tolist())),
        "mixAttemptCount": int(mix_info["mixAttemptCount"]),
        "mixAcceptedCount": int(mix_info["mixAcceptedCount"]),
        "nullEqualsObserved": bool(mix_info["nullEqualsObserved"]),
        "dgFunctionId": DG_FUNCTION_ID,
        "observedDelayedGratification": float(observed_record["delayedGratification"]),
        "observedDgEventCount": int(observed_record["dgEventCount"]),
        "nullDelayedGratification": float(metrics["delayedGratification"]),
        "nullDgEventCount": int(metrics["dgEventCount"]),
        "nullDgPositiveEventCount": int(metrics["dgPositiveEventCount"]),
        "nullDgNegativeEventCount": int(metrics["dgNegativeEventCount"]),
        "nullDgZeroEventCount": int(metrics["dgZeroEventCount"]),
        "nullDgTotalDrop": float(metrics["dgTotalDrop"]),
        "nullDgTotalRecovery": float(metrics["dgTotalRecovery"]),
        "nullDgMeanDrop": float(metrics["dgMeanDrop"]),
        "nullDgMeanRecovery": float(metrics["dgMeanRecovery"]),
        "nullDgMaxEventScore": float(metrics["dgMaxEventScore"]),
        "nullDgMinEventScore": float(metrics["dgMinEventScore"]),
        "observedMinusNullDg": float(observed_record["delayedGratification"] - metrics["delayedGratification"]),
        "observedMinusNullDgEventCount": int(observed_record["dgEventCount"] - metrics["dgEventCount"]),
        "rawSortednessPreview": compact_json([int(value) for value in raw_null[: min(20, len(raw_null))]]),
    }


def empirical_p_high(observed: float, null_values: np.ndarray | list[float]) -> float:
    values = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(values >= float(observed))) / (len(values) + 1))


def empirical_p_low(observed: float, null_values: np.ndarray | list[float]) -> float:
    values = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(values <= float(observed))) / (len(values) + 1))


def run_s08_matrix(
    *,
    s07_sources_path: Path,
    s07_trace_path: Path,
    null_replicates: int,
    null_seed_base: int,
    mix_multiplier: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    unit_passed, unit_df = run_dg_unit_cases()
    sources = load_source_trajectories(s07_sources_path, s07_trace_path)
    observed_records = [source_observed_record(source) for source in sources]
    observed_by_id = {record["sourceRunId"]: record for record in observed_records}
    null_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for source in sources:
        observed = observed_by_id[source.source_run_id]
        for null_index in range(int(null_replicates)):
            seed = stable_seed(null_seed_base, source.source_run_id, NULL_MODEL, null_index)
            null_rows.append(
                null_record_for_source(
                    source,
                    null_replicate_index=null_index,
                    null_seed=seed,
                    mix_multiplier=mix_multiplier,
                    observed_record=observed,
                )
            )
    observed_df = pd.DataFrame(observed_records)
    null_df = pd.DataFrame(null_rows)
    source_summary_df = summarize_by_source(null_df)
    context_summary_df = summarize_by_context(null_df)
    diagnostics_df = diagnostics_table(observed_df, null_df, unit_passed)
    validation = validate_outputs(observed_df, null_df, source_summary_df, context_summary_df, diagnostics_df, unit_passed)
    validation["wallTimeSeconds"] = float(time.perf_counter() - started)
    validation["sourceRunCount"] = int(len(observed_df))
    validation["nullRunCount"] = int(len(null_df))
    validation["nullReplicatesPerSource"] = int(null_replicates)
    validation["nullModel"] = NULL_MODEL
    validation["mixMultiplier"] = int(mix_multiplier)
    return observed_df, null_df, source_summary_df, context_summary_df, diagnostics_df, unit_df, validation


def summarize_by_source(null_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (source_run_id, null_model), group in null_df.groupby(["sourceRunId", "nullModel"], sort=False):
        null_values = group["nullDelayedGratification"].to_numpy(dtype=float)
        first = group.iloc[0]
        observed = float(first["observedDelayedGratification"])
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "sourceRunId": source_run_id,
                "sourceContext": first["sourceContext"],
                "sourceConditionId": first["sourceConditionId"],
                "mixtureId": first["mixtureId"],
                "algorithm": first["algorithm"],
                "frozenVariant": first["frozenVariant"],
                "frozenCount": int(first["frozenCount"]),
                "nullModel": null_model,
                "nullReplicateCount": int(len(group)),
                "observedDelayedGratification": observed,
                "nullDgMean": float(np.mean(null_values)),
                "nullDgSd": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0,
                "nullDgQ025": float(np.quantile(null_values, 0.025)),
                "nullDgQ50": float(np.quantile(null_values, 0.5)),
                "nullDgQ975": float(np.quantile(null_values, 0.975)),
                "observedMinusNullDgMean": float(observed - np.mean(null_values)),
                "observedMinusNullDgMedian": float(observed - np.median(null_values)),
                "empiricalPHighDg": empirical_p_high(observed, null_values),
                "empiricalPLowDg": empirical_p_low(observed, null_values),
                "matchedAllNulls": bool(
                    group[
                        [
                            "matchedStartSortedness",
                            "matchedEndSortedness",
                            "matchedSwapCount",
                            "matchedNoiseAmplitude",
                            "boundedSortedness",
                        ]
                    ].all(axis=None)
                ),
                "nullEqualsObservedRate": float(group["nullEqualsObserved"].mean()),
                "meanMixAcceptedCount": float(group["mixAcceptedCount"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_context(null_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        null_df.groupby(["sourceContext", "algorithm", "frozenVariant", "frozenCount", "nullModel"], dropna=False)
        .agg(
            sourceRunCount=("sourceRunId", "nunique"),
            nullRunCount=("nullTrajectoryHash", "count"),
            observedDgMean=("observedDelayedGratification", "mean"),
            observedDgMedian=("observedDelayedGratification", "median"),
            nullDgMean=("nullDelayedGratification", "mean"),
            nullDgMedian=("nullDelayedGratification", "median"),
            observedMinusNullDgMean=("observedMinusNullDg", "mean"),
            observedMinusNullDgMedian=("observedMinusNullDg", "median"),
            observedGreaterThanNullFraction=("observedMinusNullDg", lambda s: float(np.mean(np.asarray(s) > 0))),
            observedDgEventCountMean=("observedDgEventCount", "mean"),
            nullDgEventCountMean=("nullDgEventCount", "mean"),
            nullEqualsObservedRate=("nullEqualsObserved", "mean"),
            matchedStartRate=("matchedStartSortedness", "mean"),
            matchedEndRate=("matchedEndSortedness", "mean"),
            matchedSwapCountRate=("matchedSwapCount", "mean"),
            matchedNoiseAmplitudeRate=("matchedNoiseAmplitude", "mean"),
            boundedSortednessRate=("boundedSortedness", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "researchStepId", STEP_ID)
    grouped.insert(0, "experimentId", EXPERIMENT_ID)
    return grouped


def diagnostics_table(observed_df: pd.DataFrame, null_df: pd.DataFrame, unit_passed: bool) -> pd.DataFrame:
    checks = [
        ("e01_dg_unit_cases", bool(unit_passed), "E01 S08 hand-worked DG examples"),
        ("has_no_frozen_controls", bool((observed_df["frozenCount"] == 0).any()), "Observed sources include f=0 controls"),
        ("has_frozen_sources", bool((observed_df["frozenCount"] > 0).any()), "Observed sources include Frozen Cell trajectories"),
        (
            "has_passive_and_stuck",
            {"passive", "stuck"}.issubset(set(observed_df["frozenVariant"])),
            "Observed Frozen sources include passive and stuck variants",
        ),
        ("matched_start_sortedness", bool(null_df["matchedStartSortedness"].all()), "Null starts match observed starts"),
        ("matched_end_sortedness", bool(null_df["matchedEndSortedness"].all()), "Null ends match observed ends"),
        ("matched_swap_count", bool(null_df["matchedSwapCount"].all()), "Null event/swap counts match observed"),
        ("matched_noise_amplitude", bool(null_df["matchedNoiseAmplitude"].all()), "Null total variation matches observed"),
        ("bounded_sortedness", bool(null_df["boundedSortedness"].all()), "Null raw Sortedness remains in valid range"),
        ("finite_dg_values", bool(np.isfinite(null_df["nullDelayedGratification"]).all()), "All null DG values finite"),
    ]
    return pd.DataFrame(
        [
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "diagnostic": name,
                "passed": bool(passed),
                "detail": detail,
            }
            for name, passed, detail in checks
        ]
    )


def validate_outputs(
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    source_summary_df: pd.DataFrame,
    context_summary_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    unit_passed: bool,
) -> dict[str, Any]:
    checks: list[str] = []
    failures: list[str] = []
    if unit_passed:
        checks.append("E01 S08 DG hand-worked unit cases passed.")
    else:
        failures.append("E01 S08 DG hand-worked unit cases failed.")
    if len(observed_df) == 39:
        checks.append("Observed source table has the 39 S07 real-policy source trajectories.")
    else:
        failures.append(f"Observed source table has {len(observed_df)} rows; expected 39.")
    if (observed_df["frozenCount"] == 0).any() and (observed_df["frozenCount"] > 0).any():
        checks.append("Observed source table includes no-Frozen controls and Frozen Cell sources.")
    else:
        failures.append("Observed sources do not include both no-Frozen and Frozen Cell cases.")
    if {"passive", "stuck"}.issubset(set(observed_df["frozenVariant"])):
        checks.append("Frozen Cell sources include passive and stuck variants.")
    else:
        failures.append("Frozen Cell sources do not include both passive and stuck variants.")
    expected_null_rows = len(observed_df) * null_df["nullReplicateIndex"].nunique()
    if len(null_df) == expected_null_rows:
        checks.append(f"DG null table has expected row count: {len(null_df)}.")
    else:
        failures.append(f"DG null table has {len(null_df)} rows; expected {expected_null_rows}.")
    for col, label in [
        ("matchedStartSortedness", "start Sortedness"),
        ("matchedEndSortedness", "end Sortedness"),
        ("matchedSwapCount", "swap/event count"),
        ("matchedNoiseAmplitude", "noise amplitude"),
        ("boundedSortedness", "valid raw Sortedness bounds"),
    ]:
        if bool(null_df[col].all()):
            checks.append(f"All null trajectories matched {label}.")
        else:
            failures.append(f"At least one null trajectory failed {label} matching.")
    if bool(diagnostics_df["passed"].all()):
        checks.append("Machine-readable S08 diagnostics all passed.")
    else:
        failures.append("At least one machine-readable S08 diagnostic failed.")
    if len(source_summary_df) == len(observed_df):
        checks.append("Source summary has one row per observed source.")
    else:
        failures.append("Source summary does not have one row per observed source.")
    if not context_summary_df.empty:
        checks.append("Context summary table was written.")
    else:
        failures.append("Context summary table is empty.")
    return {
        "success": not failures,
        "checks": checks,
        "failures": failures,
        "nullEqualsObservedRows": int(null_df["nullEqualsObserved"].sum()) if not null_df.empty else 0,
        "matchedStartRate": float(null_df["matchedStartSortedness"].mean()) if not null_df.empty else 0.0,
        "matchedEndRate": float(null_df["matchedEndSortedness"].mean()) if not null_df.empty else 0.0,
        "matchedSwapCountRate": float(null_df["matchedSwapCount"].mean()) if not null_df.empty else 0.0,
        "matchedNoiseAmplitudeRate": float(null_df["matchedNoiseAmplitude"].mean()) if not null_df.empty else 0.0,
        "boundedSortednessRate": float(null_df["boundedSortedness"].mean()) if not null_df.empty else 0.0,
    }


def write_tables(
    *,
    artifacts_dir: Path,
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    source_summary_df: pd.DataFrame,
    context_summary_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    unit_df: pd.DataFrame,
) -> dict[str, Path]:
    results_dir = artifacts_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "observed_parquet": results_dir / "e02_dg_observed.parquet",
        "observed_csv": results_dir / "e02_dg_observed.csv",
        "nulls_parquet": results_dir / "e02_dg_nulls.parquet",
        "nulls_csv": results_dir / "e02_dg_nulls.csv",
        "source_summary_parquet": results_dir / "e02_dg_source_summary.parquet",
        "source_summary_csv": results_dir / "e02_dg_source_summary.csv",
        "context_summary_parquet": results_dir / "e02_dg_null_summary.parquet",
        "context_summary_csv": results_dir / "e02_dg_null_summary.csv",
        "diagnostics_parquet": results_dir / "e02_dg_null_diagnostics.parquet",
        "diagnostics_csv": results_dir / "e02_dg_null_diagnostics.csv",
        "unit_tests_json": artifacts_dir / "research_steps" / STEP_ID / "unit_tests.json",
    }
    observed_df.to_parquet(paths["observed_parquet"], index=False)
    observed_df.to_csv(paths["observed_csv"], index=False)
    null_df.to_parquet(paths["nulls_parquet"], index=False)
    null_df.to_csv(paths["nulls_csv"], index=False)
    source_summary_df.to_parquet(paths["source_summary_parquet"], index=False)
    source_summary_df.to_csv(paths["source_summary_csv"], index=False)
    context_summary_df.to_parquet(paths["context_summary_parquet"], index=False)
    context_summary_df.to_csv(paths["context_summary_csv"], index=False)
    diagnostics_df.to_parquet(paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(paths["diagnostics_csv"], index=False)
    write_json(paths["unit_tests_json"], {"researchStepId": STEP_ID, "unitTests": unit_df.to_dict(orient="records")})
    return paths


def plot_summary(context_summary_df: pd.DataFrame, source_summary_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "e02_dg_nulls_summary.png"
    pdf_path = figure_dir / "e02_dg_nulls_summary.pdf"
    plot_df = context_summary_df.copy()
    plot_df["label"] = (
        plot_df["sourceContext"].astype(str)
        + "\n"
        + plot_df["algorithm"].astype(str)
        + "\n"
        + plot_df["frozenVariant"].astype(str)
        + " f="
        + plot_df["frozenCount"].astype(str)
    )
    plot_df = plot_df.sort_values(["sourceContext", "algorithm", "frozenVariant", "frozenCount"]).reset_index(drop=True)
    x = np.arange(len(plot_df))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), constrained_layout=True)
    axes[0].bar(x - width / 2, plot_df["observedDgMean"], width=width, label="Observed", color="#3b6ea8")
    axes[0].bar(x + width / 2, plot_df["nullDgMean"], width=width, label="Matched null mean", color="#b85c38")
    axes[0].set_ylabel("Delayed Gratification")
    axes[0].set_title("Observed DG versus exact start/end/swap/noise-matched nulls")
    axes[0].legend(frameon=False)
    axes[1].bar(x, plot_df["observedMinusNullDgMean"], color="#496f5d")
    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_ylabel("Observed minus null DG")
    axes[1].set_title("DG excess over matched trajectory null")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["label"], rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
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
    script_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        script_dir / "e02_s08_dg_nulls.py",
        script_dir / "e01_s08_delayed_gratification.py",
        test_dir / "test_e02_dg_nulls.py",
    ]
    shutil.copy2(REPO_ROOT / "scripts" / "e02_s08_dg_nulls.py", paths[0])
    shutil.copy2(REPO_ROOT / "scripts" / "e01_s08_delayed_gratification.py", paths[1])
    shutil.copy2(REPO_ROOT / "tests" / "test_e02_dg_nulls.py", paths[2])
    return paths


def write_validation_report(step_dir: Path, validation: dict[str, Any]) -> Path:
    path = step_dir / "validation_report.md"
    lines = [
        "# E02 S08 Validation Report",
        "",
        "- Research step ID: S08",
        f"- Completion status: {'completed' if validation['success'] else 'failed'}",
        "- Artifacts written: DG observed/null tables, source/context summaries, diagnostics, unit-test JSON, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {'passed' if validation['success'] else 'failed'}",
        f"- Caveats or blockers: {validation.get('caveats', 'none')}",
        "- Recommended next action: stop before S09 for Chief Scientist review.",
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
    cmd = [sys.executable, "-m", "unittest", "tests.test_e02_dg_nulls"]
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


def outcome_classification(source_summary_df: pd.DataFrame, validation_success: bool) -> str:
    if not validation_success:
        return "constraining/contradictory"
    if source_summary_df.empty:
        return "null"
    median_delta = float(source_summary_df["observedMinusNullDgMedian"].median())
    high_tail_hits = float(np.mean(source_summary_df["empiricalPHighDg"] <= 0.05))
    if median_delta > 0 and high_tail_hits >= 0.1:
        return "supportive"
    if abs(median_delta) <= 0.05:
        return "null"
    return "constraining/contradictory"


def write_reports_and_manifests(
    *,
    artifacts_dir: Path,
    table_paths: dict[str, Path],
    figure_paths: tuple[Path, Path],
    code_paths: list[Path],
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    source_summary_df: pd.DataFrame,
    context_summary_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    validation: dict[str, Any],
    repo_tests: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Path]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    step_dir.mkdir(parents=True, exist_ok=True)
    validation["success"] = bool(validation["success"] and repo_tests["ok"])
    if repo_tests["ok"]:
        validation["checks"].append("Repository unit tests passed for S08 DG null helpers.")
    else:
        validation["failures"].append("Repository unit tests failed for S08 DG null helpers.")
    validation["caveats"] = (
        "S08 uses the bounded S07 n=30 real-policy source matrix, which includes no-Frozen controls and f=2 passive/stuck Frozen Cell sources; "
        "full f=1/f=3 E01 Frozen count sweeps remain upstream context rather than rerun here. The matched-delta null preserves Sortedness delta multiset and total variation, so it tests temporal ordering of backtracking rather than all possible trajectory-null families."
    )
    validation_report_path = write_validation_report(step_dir, validation)
    classification = outcome_classification(source_summary_df, bool(validation["success"]))
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "provenance" / "run_manifest.json"
    context_rows = [
        [
            row.sourceContext,
            row.algorithm,
            row.frozenVariant,
            int(row.frozenCount),
            int(row.sourceRunCount),
            float(row.observedDgMean),
            float(row.nullDgMean),
            float(row.observedMinusNullDgMean),
            float(row.observedGreaterThanNullFraction),
        ]
        for row in context_summary_df.sort_values(["sourceContext", "algorithm", "frozenVariant", "frozenCount"]).itertuples(index=False)
    ]
    summary_lines = [
        "# E02 S08 Summary",
        "",
        "- Research step ID: S08",
        f"- Completion status: {'completed' if validation['success'] else 'failed'}",
        "- Artifacts written: `$ARTIFACTS_DIR/results/e02_dg_nulls.parquet`, observed/source/context summary tables, diagnostics, unit-test JSON, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {'passed' if validation['success'] else 'failed'}",
        f"- Caveats or blockers: {validation['caveats']}",
        "- Lay summary: S08 recalculated Delayed Gratification with the E01 formula and compared each real trajectory with random reorderings of the same Sortedness changes. The nulls exactly match the observed start, end, event count, and total backtracking amplitude, so differences isolate whether the temporal order of drops and recoveries is unusually DG-like.",
        "- Recommended next action: stop before S09 for Chief Scientist review; if accepted, run S09 alternative metric sensitivity only after explicit instruction.",
        f"- Outcome classification: {classification}",
        "",
        "## Run Counts",
        "",
        markdown_table(
            ["Item", "Count"],
            [
                ["Observed source runs", len(observed_df)],
                ["Null rows", len(null_df)],
                ["Null model", NULL_MODEL],
                ["Null rows equal observed order", validation["nullEqualsObservedRows"]],
            ],
        ),
        "",
        "## Context Summary",
        "",
        markdown_table(
            [
                "Source context",
                "Algorithm",
                "Frozen variant",
                "f",
                "Sources",
                "Observed DG mean",
                "Null DG mean",
                "Observed-null DG",
                "Observed > null frac.",
            ],
            context_rows,
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
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    artifact_inputs = list(table_paths.values()) + list(figure_paths) + code_paths + [
        validation_report_path,
        Path(repo_tests["logPath"]),
        summary_path,
        status_path,
        manifest_path,
        run_manifest_path,
    ]
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
        "dgFunctionId": DG_FUNCTION_ID,
        "e01DgScriptSha256": sha256_path(REPO_ROOT / "scripts" / "e01_s08_delayed_gratification.py"),
        "validation": validation,
        "repoTests": repo_tests,
        "outcomeClassification": classification,
        "artifacts": [],
    }
    write_json(manifest_path, manifest)
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"]),
        "status": "completed" if validation["success"] else "failed",
        "artifactsWritten": [],
        "validationResult": "passed" if validation["success"] else "failed",
        "caveatsOrBlockers": validation["caveats"] if validation["success"] else "; ".join(validation["failures"]),
        "recommendedNextAction": "Stop before S09 for Chief Scientist review; if accepted, proceed to S09 alternative metric sensitivity.",
    }
    write_json(status_path, status_payload)
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
                "dgNulls": str(table_paths["nulls_parquet"]),
                "summary": str(summary_path),
                "status": str(status_path),
                "manifest": str(manifest_path),
            },
            "git": get_git_metadata(),
        }
    )
    write_json(run_manifest_path, run_manifest)
    manifest["artifacts"] = collect_artifacts(artifact_inputs)
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
    parser.add_argument("--s07-sources", type=Path, default=DEFAULT_S07_SOURCES)
    parser.add_argument("--s07-trace", type=Path, default=DEFAULT_S07_TRACE)
    parser.add_argument("--null-replicates", type=int, default=500)
    parser.add_argument("--null-seed-base", type=int, default=8808001)
    parser.add_argument("--mix-multiplier", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    step_dir.mkdir(parents=True, exist_ok=True)
    observed_df, null_df, source_summary_df, context_summary_df, diagnostics_df, unit_df, validation = run_s08_matrix(
        s07_sources_path=args.s07_sources,
        s07_trace_path=args.s07_trace,
        null_replicates=args.null_replicates,
        null_seed_base=args.null_seed_base,
        mix_multiplier=args.mix_multiplier,
    )
    table_paths = write_tables(
        artifacts_dir=args.artifacts_dir,
        observed_df=observed_df,
        null_df=null_df,
        source_summary_df=source_summary_df,
        context_summary_df=context_summary_df,
        diagnostics_df=diagnostics_df,
        unit_df=unit_df,
    )
    figure_paths = plot_summary(context_summary_df, source_summary_df, args.artifacts_dir / "figures" / "e02")
    code_paths = copy_code_artifacts(step_dir)
    repo_tests = run_repo_tests(step_dir)
    report_paths = write_reports_and_manifests(
        artifacts_dir=args.artifacts_dir,
        table_paths=table_paths,
        figure_paths=figure_paths,
        code_paths=code_paths,
        observed_df=observed_df,
        null_df=null_df,
        source_summary_df=source_summary_df,
        context_summary_df=context_summary_df,
        diagnostics_df=diagnostics_df,
        validation=validation,
        repo_tests=repo_tests,
        args=args,
    )
    print(f"Wrote S08 DG null results to {table_paths['nulls_parquet']}")
    print(f"Wrote S08 summary to {report_paths['summary']}")
    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
