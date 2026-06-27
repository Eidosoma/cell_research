"""Coarse morphospace sweep helpers for E03 S07."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from e02_deterministic_simulator.metrics import aggregation

from .batch_simulator import BatchSimulationResult
from .competence import canonical_json, parse_int_array, summarize_competence_vectors, vector_from_summary_record


COARSE_SWEEP_VERSION = "e03_s07_coarse_sweep.v1"


@dataclass(frozen=True)
class SweepTask:
    """A single policy-evaluation task in the S07 coarse sweep."""

    task_id: str
    task_panel: str
    task_family: str
    input_profile: str
    initial_values: tuple[int, ...]
    scheduler_seed_base: int
    replicate_index: int
    backend: str
    frozen_variant: str = "none"
    frozen_positions: tuple[int, ...] = ()
    policy_mode: str = "single_policy"
    max_activations: int = 128
    max_swaps: int = 128
    max_comparisons: int = 512

    @property
    def n(self) -> int:
        return len(self.initial_values)

    @property
    def frozen_count(self) -> int:
        return len(self.frozen_positions)

    def scheduler_seed(self, policy_index: int) -> int:
        return int(self.scheduler_seed_base + policy_index)


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def value_counts_conserved(initial_values: Sequence[int] | str, final_values: Sequence[int] | str) -> bool:
    """Return true when the final value multiset equals the input multiset."""

    initial = parse_int_array(initial_values)
    final = parse_int_array(final_values)
    return Counter(initial) == Counter(final)


def policy_metadata(policy_row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize S05 policy-corpus metadata used by S07 tables."""

    if policy_row is None:
        policy_row = {}
    return {
        "policyFamily": str(policy_row.get("family", "unknown")),
        "generationMethod": str(policy_row.get("generationMethod", "unknown")),
        "lineageId": str(policy_row.get("lineageId", "")),
        "lineageDepth": int(policy_row.get("lineageDepth", 0) or 0),
        "structureHash": str(policy_row.get("structureHash", "")),
        "complexityScore": int(policy_row.get("complexityScore", 0) or 0),
        "ruleCount": int(policy_row.get("ruleCount", 0) or 0),
        "predicateCount": int(policy_row.get("predicateCount", 0) or 0),
    }


def _task_fields(task: SweepTask, policy_id: str, policy_index: int) -> dict[str, Any]:
    source_run_id = f"S07::{task.task_id}::{policy_id}::seed{task.scheduler_seed(policy_index)}"
    return {
        "coarseSweepVersion": COARSE_SWEEP_VERSION,
        "researchStepId": "S07",
        "policyId": policy_id,
        "taskId": task.task_id,
        "taskFamily": task.task_family,
        "taskPanel": task.task_panel,
        "inputProfile": task.input_profile,
        "n": task.n,
        "initialValuesJson": json_compact(list(task.initial_values)),
        "schedulerSeed": task.scheduler_seed(policy_index),
        "replicateIndex": task.replicate_index,
        "backend": task.backend,
        "policyMode": task.policy_mode,
        "frozenVariant": task.frozen_variant,
        "frozenCount": task.frozen_count,
        "frozenPositionsJson": json_compact(list(task.frozen_positions)),
        "sourceRunId": source_run_id,
        "condition_id": source_run_id,
    }


def batch_result_to_run_records(
    result: BatchSimulationResult,
    task: SweepTask,
    metadata_by_policy: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert one JAX batch result into S07 run records."""

    records: list[dict[str, Any]] = []
    for policy_index, row in enumerate(result.to_rows()):
        policy_id = str(row["policyId"])
        final_values = parse_int_array(row["finalValuesJson"])
        record = {
            **_task_fields(task, policy_id, policy_index),
            **policy_metadata(metadata_by_policy.get(policy_id)),
            "algorithm": policy_id,
            "batchSimulatorVersion": row.get("batchSimulatorVersion"),
            "backendDevice": row.get("backendDevice"),
            "finalValuesJson": json_compact(final_values),
            "completed": bool(row["completed"]),
            "stopReason": str(row["stopReason"]),
            "swapCount": int(row["swapCount"]),
            "comparisonCount": int(row["comparisonCount"]),
            "activationCount": int(row["activationCount"]),
            "eventCount": int(row["eventCount"]),
            "blockedMoveAttempts": 0,
            "frozenSwapAttempts": 0,
            "finalSortednessRawCount": int(row["finalSortednessRawCount"]),
            "finalSortednessPercent": float(row["finalSortednessPercent"]),
            "finalMonotonicityError": int(row["finalMonotonicityError"]),
            "finalAggregation": None,
            "traceHashSequenceJson": row.get("traceHashSequenceJson", "[]"),
            "trajectoryStateCount": None,
            "runtimeSeconds": None,
        }
        record["valueCountsConserved"] = value_counts_conserved(record["initialValuesJson"], record["finalValuesJson"])
        records.append(record)
    return records


def simulation_result_to_run_record(
    result: Any,
    task: SweepTask,
    policy_id: str,
    policy_index: int,
    metadata_by_policy: Mapping[str, Mapping[str, Any]],
    *,
    backend_device: str = "cpu_reference",
) -> dict[str, Any]:
    """Convert a CPU reference SimulationResult into an S07 run record."""

    record = {
        **_task_fields(task, policy_id, policy_index),
        **policy_metadata(metadata_by_policy.get(policy_id)),
        "algorithm": result.algorithm,
        "batchSimulatorVersion": None,
        "backendDevice": backend_device,
        "finalValuesJson": json_compact(list(result.final_values)),
        "completed": bool(result.completed),
        "stopReason": str(result.stop_reason),
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "activationCount": int(result.activation_count),
        "eventCount": int(result.event_count),
        "blockedMoveAttempts": int(result.blocked_move_attempts),
        "frozenSwapAttempts": int(result.frozen_swap_attempts),
        "finalSortednessRawCount": int(result.final_sortedness_raw_count),
        "finalSortednessPercent": float(result.final_sortedness_percent),
        "finalMonotonicityError": int(result.final_monotonicity_error),
        "finalAggregation": float(result.final_aggregation),
        "traceHashSequenceJson": json_compact([row["state_hash"] for row in result.trace_rows]),
        "trajectoryStateCount": int(len(result.trace_rows)),
        "runtimeSeconds": float(result.wall_time_seconds),
    }
    record["valueCountsConserved"] = value_counts_conserved(record["initialValuesJson"], record["finalValuesJson"])
    return record


def run_records_to_competence_vectors(
    run_records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    source_artifact_path: str | None = None,
) -> pd.DataFrame:
    """Map S07 run records into the S04 competence-vector schema."""

    df = pd.DataFrame(run_records)
    vectors = []
    for _, row in df.iterrows():
        source = row.to_dict()
        vector = vector_from_summary_record(
            source,
            source_metric_source="s07_coarse_sweep",
            source_artifact_path=source_artifact_path,
        )
        vector["backend"] = source.get("backend")
        vector["coarseSweepVersion"] = source.get("coarseSweepVersion", COARSE_SWEEP_VERSION)
        vectors.append(vector)
    return pd.DataFrame(vectors)


def summarize_s07_competence(vectors: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    return summarize_competence_vectors(
        vectors,
        group_columns=("policyId", "policyFamily", "taskPanel", "backend"),
        metric_columns=(
            "completionSuccess",
            "finalSortednessScore",
            "finalMonotonicityScore",
            "swapEfficiencyScore",
            "comparisonEfficiencyScore",
            "activationEfficiencyScore",
            "energyScore",
            "robustnessScore",
            "aggregationPeakScore",
            "failureOscillationScore",
        ),
    )


def select_stratified_policy_ids(
    corpus_df: pd.DataFrame,
    candidate_policy_ids: Sequence[str],
    *,
    sample_size: int,
    strata_columns: Sequence[str] = ("family", "generationMethod"),
) -> list[str]:
    """Deterministically sample policies round-robin across available strata."""

    if sample_size <= 0:
        return []
    candidates = corpus_df[corpus_df["policyId"].isin(set(candidate_policy_ids))].copy()
    if candidates.empty:
        return []
    sort_columns = [column for column in strata_columns if column in candidates.columns] + [
        "complexityScore",
        "policyId",
    ]
    candidates = candidates.sort_values(sort_columns, kind="mergesort")
    group_columns = [column for column in strata_columns if column in candidates.columns]
    grouped = (
        [((), candidates)]
        if not group_columns
        else list(candidates.groupby(group_columns, dropna=False, sort=True))
    )
    selected: list[str] = []
    offsets = {index: 0 for index in range(len(grouped))}
    while len(selected) < min(sample_size, len(candidates)):
        progressed = False
        for group_index, (_, group) in enumerate(grouped):
            offset = offsets[group_index]
            if offset >= len(group):
                continue
            selected.append(str(group.iloc[offset]["policyId"]))
            offsets[group_index] = offset + 1
            progressed = True
            if len(selected) >= sample_size:
                break
        if not progressed:
            break
    return selected


def make_missing_record(
    *,
    policy_row: Mapping[str, Any],
    task_id: str,
    task_panel: str,
    task_family: str,
    backend: str,
    missing_reason: str,
    support_reasons_json: str = "[]",
    replacement_handling: str = "",
) -> dict[str, Any]:
    metadata = policy_metadata(policy_row)
    return {
        "coarseSweepVersion": COARSE_SWEEP_VERSION,
        "researchStepId": "S07",
        "policyId": str(policy_row.get("policyId", "unknown")),
        **metadata,
        "taskId": task_id,
        "taskPanel": task_panel,
        "taskFamily": task_family,
        "backend": backend,
        "recordStatus": "missing",
        "missingReason": missing_reason,
        "supportReasonsJson": support_reasons_json,
        "replacementHandling": replacement_handling,
    }


def task_coverage_table(
    run_records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    missing_records: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Summarize evaluated and explicitly missing task/seed coverage."""

    rows: list[dict[str, Any]] = []
    run_df = pd.DataFrame(run_records)
    if not run_df.empty:
        grouped = run_df.groupby(["taskId", "taskPanel", "taskFamily", "backend", "frozenVariant"], dropna=False)
        for (task_id, task_panel, task_family, backend, frozen_variant), group in grouped:
            rows.append(
                {
                    "taskId": task_id,
                    "taskPanel": task_panel,
                    "taskFamily": task_family,
                    "backend": backend,
                    "frozenVariant": frozen_variant,
                    "recordStatus": "evaluated",
                    "runRows": int(len(group)),
                    "policyCount": int(group["policyId"].nunique()),
                    "seedCount": int(group["schedulerSeed"].nunique()) if "schedulerSeed" in group else 0,
                    "inputProfilesJson": json_compact(sorted(map(str, group["inputProfile"].dropna().unique()))),
                    "missingReasonCount": 0,
                }
            )
    missing_df = pd.DataFrame(missing_records)
    if not missing_df.empty:
        grouped = missing_df.groupby(["taskId", "taskPanel", "taskFamily", "backend"], dropna=False)
        for (task_id, task_panel, task_family, backend), group in grouped:
            rows.append(
                {
                    "taskId": task_id,
                    "taskPanel": task_panel,
                    "taskFamily": task_family,
                    "backend": backend,
                    "frozenVariant": "various_or_not_evaluated",
                    "recordStatus": "missing",
                    "runRows": int(len(group)),
                    "policyCount": int(group["policyId"].nunique()),
                    "seedCount": 0,
                    "inputProfilesJson": "[]",
                    "missingReasonCount": int(group["missingReason"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(["recordStatus", "taskPanel", "taskId", "backend"]).reset_index(drop=True)


def support_reason_lookup(support_df: pd.DataFrame) -> dict[str, str]:
    return {
        str(row["policyId"]): str(row.get("supportReasonsJson", "[]"))
        for _, row in support_df.iterrows()
    }


def run_record_validation(run_records: Sequence[Mapping[str, Any]] | pd.DataFrame) -> dict[str, Any]:
    """Return compact integrity checks for S07 run records."""

    df = pd.DataFrame(run_records)
    if df.empty:
        return {"rowCount": 0, "valueCountsConserved": False, "metricRangesValid": False}
    final_sorted = pd.to_numeric(df["finalSortednessPercent"], errors="coerce")
    final_error = pd.to_numeric(df["finalMonotonicityError"], errors="coerce")
    return {
        "rowCount": int(len(df)),
        "policyCount": int(df["policyId"].nunique()),
        "taskCount": int(df["taskId"].nunique()),
        "valueCountsConserved": bool(df["valueCountsConserved"].all()),
        "metricRangesValid": bool(final_sorted.between(0.0, 100.0).all() and (final_error >= 0).all()),
        "completedRows": int(df["completed"].sum()),
        "stopReasonsJson": canonical_json(sorted(map(str, df["stopReason"].dropna().unique()))),
    }


def chimera_compatibility_score(record: Mapping[str, Any]) -> float | None:
    """Simple S07 chimera proxy: sorting success weighted by final aggregation."""

    if record.get("taskFamily") != "chimera":
        return None
    final_sortedness = float(record.get("finalSortednessPercent", 0.0)) / 100.0
    final_aggregation = record.get("finalAggregation")
    if final_aggregation is None:
        return None
    return float(final_sortedness * float(final_aggregation))


def aggregation_from_algotypes(algotypes: Sequence[str]) -> float:
    return float(aggregation(algotypes))
