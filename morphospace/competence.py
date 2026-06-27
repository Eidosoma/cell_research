"""Competence-vector schema and metrics for E03 morphospace policies.

S04 defines one wide record shape that later sweeps can use for classic,
generated, chimeric, perturbation, and transfer tasks. Metrics that are not
meaningful for a task stay missing and carry an explicit reason in
``missingMetricReasons``.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from e02_deterministic_simulator.metrics import aggregation, monotonicity_error, sortedness_percent, sortedness_raw
from e02_deterministic_simulator.simulator import SimulationResult


COMPETENCE_VECTOR_VERSION = "e03_s04_competence_vector.v1"
NORMALIZATION_VERSION = "e03_s04_normalization.v1"
MISSING_VALUE_POLICY = "Use null/NaN for not-applicable or unavailable metrics and record a reason in missingMetricReasons."


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def parse_json_array(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return list(json.loads(text))
    if isinstance(value, np.ndarray):
        return list(value.tolist())
    if isinstance(value, (list, tuple)):
        return list(value)
    if pd.isna(value):
        return []
    raise TypeError(f"cannot parse array from {type(value)}")


def parse_int_array(value: Any) -> list[int]:
    return [int(item) for item in parse_json_array(value)]


def parse_str_array(value: Any) -> list[str]:
    return [str(item) for item in parse_json_array(value)]


def bounded_score(value: float | int | None, denominator: float | int | None, *, invert: bool = True) -> float | None:
    """Map a nonnegative raw cost to [0, 1], optionally as higher-is-better."""

    if value is None or denominator is None:
        return None
    value = float(value)
    denominator = float(denominator)
    if denominator <= 0 or not math.isfinite(value) or not math.isfinite(denominator):
        return None
    ratio = max(0.0, min(1.0, value / denominator))
    return float(1.0 - ratio if invert else ratio)


def bounded_unit(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(max(0.0, min(1.0, value)))


def lower_is_better_unit(value: float | int | None) -> float | None:
    value = bounded_unit(value)
    return None if value is None else float(1.0 - value)


def dg_score(value: float | int | None) -> float | None:
    """Monotone bounded transform for signed Delayed Gratification."""

    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(0.5 + math.atan(value) / math.pi)


def direction_key(value: int, direction: str) -> int:
    if direction == "increasing":
        return int(value)
    if direction == "decreasing":
        return -int(value)
    raise ValueError(f"unknown direction: {direction}")


def target_position_assignment(values: Sequence[int], direction: str = "increasing") -> np.ndarray:
    values_list = [int(value) for value in values]
    sorted_values = sorted(values_list, reverse=(direction == "decreasing"))
    target_slots: dict[int, deque[int]] = defaultdict(deque)
    for index, value in enumerate(sorted_values):
        target_slots[int(value)].append(index)
    assigned: list[int] = []
    for value in values_list:
        assigned.append(target_slots[int(value)].popleft())
    return np.asarray(assigned, dtype=np.int64)


def tie_aware_inversion_count(values: Sequence[int], direction: str = "increasing") -> int:
    transformed = [direction_key(int(value), direction) for value in values]
    inversions = 0
    for left_index, left in enumerate(transformed):
        for right in transformed[left_index + 1 :]:
            if left > right:
                inversions += 1
    return int(inversions)


def comparable_pair_count(values: Sequence[int]) -> int:
    values_list = [int(value) for value in values]
    total = len(values_list) * (len(values_list) - 1) // 2
    counts: dict[int, int] = defaultdict(int)
    for value in values_list:
        counts[int(value)] += 1
    tie_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    return int(total - tie_pairs)


def longest_nondecreasing_subsequence_length(values: Sequence[int], direction: str = "increasing") -> int:
    transformed = [direction_key(int(value), direction) for value in values]
    tails: list[int] = []
    for value in transformed:
        position = bisect.bisect_right(tails, value)
        if position == len(tails):
            tails.append(value)
        else:
            tails[position] = value
    return len(tails)


def target_displacements(values: Sequence[int], direction: str = "increasing") -> np.ndarray:
    assigned = target_position_assignment(values, direction=direction)
    positions = np.arange(len(assigned), dtype=np.int64)
    return assigned - positions


def reverse_target_footrule(values: Sequence[int], direction: str = "increasing") -> int:
    values_list = [int(value) for value in values]
    reverse_values = sorted(values_list, reverse=(direction == "increasing"))
    return int(np.abs(target_displacements(reverse_values, direction=direction)).sum())


def state_distance_metrics(values: Sequence[int], direction: str = "increasing") -> dict[str, Any]:
    values_list = [int(value) for value in values]
    n = len(values_list)
    comparable = comparable_pair_count(values_list)
    inversions = tie_aware_inversion_count(values_list, direction=direction)
    displacement = target_displacements(values_list, direction=direction)
    footrule = int(np.abs(displacement).sum())
    squared = int(np.square(displacement).sum())
    reverse_footrule = reverse_target_footrule(values_list, direction=direction)
    reverse_values = sorted(values_list, reverse=(direction == "increasing"))
    reverse_squared = int(np.square(target_displacements(reverse_values, direction=direction)).sum())
    lnds = longest_nondecreasing_subsequence_length(values_list, direction=direction)
    edit_distance = max(0, n - lnds)
    sortedness = sortedness_percent(values_list, direction=direction) if n else 100.0
    return {
        "n": int(n),
        "hasDuplicates": len(set(values_list)) < n,
        "uniqueValueCount": int(len(set(values_list))),
        "sortednessRawCount": int(sortedness_raw(values_list, direction=direction)) if n else 0,
        "sortednessPercent": float(sortedness),
        "sortednessDistanceNormalized": float(1.0 - sortedness / 100.0),
        "monotonicityError": int(monotonicity_error(values_list, direction=direction)) if n else 0,
        "inversionCount": int(inversions),
        "comparablePairCount": int(comparable),
        "inversionDistanceNormalized": float(inversions / comparable) if comparable else 0.0,
        "kendallTauDistanceNormalized": float(inversions / comparable) if comparable else 0.0,
        "spearmanFootruleDistance": int(footrule),
        "spearmanFootruleDistanceNormalized": float(footrule / reverse_footrule) if reverse_footrule else 0.0,
        "spearmanSquaredDistance": int(squared),
        "spearmanSquaredDistanceNormalized": float(squared / reverse_squared) if reverse_squared else 0.0,
        "earthMoverPositionDistance": float(footrule / n) if n else 0.0,
        "earthMoverPositionDistanceNormalized": float(footrule / reverse_footrule) if reverse_footrule else 0.0,
        "editDistanceToTargetOrder": int(edit_distance),
        "editDistanceToTargetOrderNormalized": float(edit_distance / n) if n else 0.0,
    }


def prefixed_metrics(prefix: str, values: Sequence[int], direction: str = "increasing") -> dict[str, Any]:
    metrics = state_distance_metrics(values, direction=direction)
    return {f"{prefix}{key[0].upper()}{key[1:]}": value for key, value in metrics.items()}


def trajectory_curvature_metrics(states: Sequence[Sequence[int]]) -> dict[str, Any]:
    if len(states) < 2:
        return {
            "trajectoryStateCount": int(len(states)),
            "trajectoryTurnCount": 0,
            "trajectoryPathLengthL2": 0.0,
            "trajectoryDirectDistanceL2": 0.0,
            "trajectoryExcessPathRatio": 0.0,
            "stateSpaceCurvatureTotalTurnRadians": 0.0,
            "stateSpaceCurvatureMeanTurnRadians": 0.0,
            "stateSpaceCurvatureMeanOneMinusCosine": 0.0,
        }
    arr = np.asarray(states, dtype=float)
    deltas = np.diff(arr, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    path_length = float(norms.sum())
    direct = float(np.linalg.norm(arr[-1] - arr[0]))
    angles: list[float] = []
    one_minus_cosines: list[float] = []
    for left, right, left_norm, right_norm in zip(deltas[:-1], deltas[1:], norms[:-1], norms[1:]):
        if left_norm <= 0 or right_norm <= 0:
            continue
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
        cosine = max(-1.0, min(1.0, cosine))
        angles.append(float(math.acos(cosine)))
        one_minus_cosines.append(float(1.0 - cosine))
    return {
        "trajectoryStateCount": int(len(states)),
        "trajectoryTurnCount": int(len(angles)),
        "trajectoryPathLengthL2": path_length,
        "trajectoryDirectDistanceL2": direct,
        "trajectoryExcessPathRatio": float((path_length / direct) - 1.0) if direct > 0 else 0.0,
        "stateSpaceCurvatureTotalTurnRadians": float(np.sum(angles)) if angles else 0.0,
        "stateSpaceCurvatureMeanTurnRadians": float(np.mean(angles)) if angles else 0.0,
        "stateSpaceCurvatureMeanOneMinusCosine": float(np.mean(one_minus_cosines)) if one_minus_cosines else 0.0,
    }


def signed_segments(values: Sequence[float], *, tolerance: float = 1e-12) -> list[float]:
    values_list = [float(value) for value in values]
    if len(values_list) < 2:
        return []
    deduped = [values_list[0]]
    for value in values_list[1:]:
        if not math.isclose(value, deduped[-1], rel_tol=0.0, abs_tol=tolerance):
            deduped.append(value)
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


def delayed_gratification_from_sortedness(values: Sequence[float]) -> dict[str, Any]:
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


def sortedness_sign_change_count(values: Sequence[float]) -> int:
    segments = signed_segments(values)
    signs = [1 if value > 0 else -1 for value in segments if not math.isclose(value, 0.0, abs_tol=1e-12)]
    return int(sum(1 for left, right in zip(signs, signs[1:]) if left != right))


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    column: str
    family: str
    direction: str
    normalization: str
    normalized_column: str | None
    missing_policy: str
    uncertainty_columns: tuple[str, ...]
    scope: str
    description: str
    source: str = "S04"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metricId": self.metric_id,
            "column": self.column,
            "family": self.family,
            "direction": self.direction,
            "normalization": self.normalization,
            "normalizedColumn": self.normalized_column,
            "missingPolicy": self.missing_policy,
            "uncertaintyColumns": list(self.uncertainty_columns),
            "scope": self.scope,
            "description": self.description,
            "source": self.source,
        }


UNCERTAINTY_COLUMNS = (
    "uncertaintyN",
    "uncertaintyMean",
    "uncertaintySd",
    "uncertaintySem",
    "uncertaintyCi95Low",
    "uncertaintyCi95High",
    "uncertaintyMethod",
)


def competence_metric_specs() -> tuple[MetricSpec, ...]:
    common_missing = "required for all run-level vectors"
    optional = MISSING_VALUE_POLICY
    return (
        MetricSpec("completion_success", "completionSuccess", "sorting", "higher", "identity 0/1", "completionSuccess", common_missing, UNCERTAINTY_COLUMNS, "run", "Whether the run reached the task stop condition interpreted as success."),
        MetricSpec("final_sortedness_percent", "finalSortednessPercent", "sorting", "higher", "divide by 100", "finalSortednessScore", common_missing, UNCERTAINTY_COLUMNS, "run", "E01/E02 final percent Sortedness."),
        MetricSpec("final_monotonicity_error", "finalMonotonicityError", "sorting", "lower", "1 - error / max(n-1,1)", "finalMonotonicityScore", common_missing, UNCERTAINTY_COLUMNS, "run", "Final adjacent-pair monotonicity error."),
        MetricSpec("final_kendall_tau_distance", "finalKendallTauDistanceNormalized", "sorting", "lower", "1 - normalized distance", "finalKendallTauScore", optional, UNCERTAINTY_COLUMNS, "run", "Tie-aware final Kendall tau distance to target order."),
        MetricSpec("final_earth_mover_position_distance", "finalEarthMoverPositionDistanceNormalized", "sorting", "lower", "1 - normalized distance", "finalEarthMoverPositionScore", optional, UNCERTAINTY_COLUMNS, "run", "Position-footrule Earth-mover proxy normalized by reverse order."),
        MetricSpec("swap_count", "swapCount", "efficiency", "lower", "1 - min(swaps / max_pairs, 1)", "swapEfficiencyScore", common_missing, UNCERTAINTY_COLUMNS, "run", "Adjacent swap count as a low-is-better work proxy."),
        MetricSpec("comparison_count", "comparisonCount", "efficiency", "lower", "1 - min(comparisons / n^2, 1)", "comparisonEfficiencyScore", common_missing, UNCERTAINTY_COLUMNS, "run", "Policy comparison count as a low-is-better work proxy."),
        MetricSpec("activation_count", "activationCount", "efficiency", "lower", "1 - min(activations / n^2, 1)", "activationEfficiencyScore", common_missing, UNCERTAINTY_COLUMNS, "run", "Scheduler activation count as a low-is-better work proxy."),
        MetricSpec("energy_proxy", "energyProxy", "efficiency", "lower", "1 - min((swaps+comparisons+activations)/(3*n^2), 1)", "energyScore", common_missing, UNCERTAINTY_COLUMNS, "run", "Simple compute/work proxy; not physical energy."),
        MetricSpec("passive_or_stuck_robustness", "robustnessScore", "perturbation", "higher", "completionSuccess * finalSortednessScore on Frozen Cell tasks", "robustnessScore", optional, UNCERTAINTY_COLUMNS, "run", "Frozen Cell robustness proxy; missing on no-Frozen tasks."),
        MetricSpec("delayed_gratification", "delayedGratification", "trajectory", "higher", "0.5 + atan(DG)/pi", "delayedGratificationScore", optional, UNCERTAINTY_COLUMNS, "trajectory", "E01/E02 signed DG score from Sortedness trajectory."),
        MetricSpec("aggregation_peak", "aggregationPeak", "chimera", "higher", "identity 0..1", "aggregationPeakScore", optional, UNCERTAINTY_COLUMNS, "trajectory", "Peak adjacent same-label or same-Algotype aggregation."),
        MetricSpec("aggregation_auc", "aggregationAuc", "chimera", "higher", "identity 0..1", "aggregationAucScore", optional, UNCERTAINTY_COLUMNS, "trajectory", "Mean aggregation over trace events."),
        MetricSpec("dominance_score", "dominanceScore", "conflict", "contextual", "identity 0..1 when supplied", "dominanceScore", optional, UNCERTAINTY_COLUMNS, "task_panel", "Opposing-goal dominance proxy; missing outside conflict tasks."),
        MetricSpec("compatibility_score", "compatibilityScore", "chimera", "higher", "identity 0..1 when supplied", "compatibilityScore", optional, UNCERTAINTY_COLUMNS, "task_panel", "Chimeric compatibility proxy; missing until a chimera panel defines it."),
        MetricSpec("trajectory_excess_path_ratio", "trajectoryExcessPathRatio", "trajectory", "lower", "1/(1 + excess path ratio)", "pathDirectnessScore", optional, UNCERTAINTY_COLUMNS, "trajectory", "State-space path inefficiency relative to direct displacement."),
        MetricSpec("oscillation_proxy", "oscillationProxy", "stability", "lower", "1 - sign-change rate", "oscillationScore", optional, UNCERTAINTY_COLUMNS, "trajectory", "Sortedness sign-change rate as an oscillation proxy."),
        MetricSpec("failure_or_oscillation", "failureOscillationScore", "stability", "higher", "min(completionSuccess, oscillationScore)", "failureOscillationScore", common_missing, UNCERTAINTY_COLUMNS, "run", "Combined nonfailure and low-oscillation proxy."),
        MetricSpec("transfer_score", "transferScore", "transfer", "higher", "identity 0..1 when supplied", "transferScore", optional, UNCERTAINTY_COLUMNS, "task_panel", "Cross-size or cross-world transfer score; missing until transfer panel exists."),
        MetricSpec("repair_success", "repairSuccessScore", "repair", "higher", "identity 0..1 when supplied", "repairSuccessScore", optional, UNCERTAINTY_COLUMNS, "task_panel", "Repair/regeneration success proxy reserved for later experiments."),
    )


IDENTITY_FIELDS: tuple[dict[str, str], ...] = (
    {"name": "vectorId", "type": "string", "description": "Stable hash over policy/task/source identifiers."},
    {"name": "policyId", "type": "string", "description": "Stable policy identifier."},
    {"name": "policyFamily", "type": "string", "description": "Classic, DSL, generated, null, or randomized family."},
    {"name": "algorithm", "type": "string", "description": "Human-readable algorithm or policy group label."},
    {"name": "taskId", "type": "string", "description": "Task or condition identifier."},
    {"name": "taskFamily", "type": "string", "description": "sorting, frozen, chimera, conflict, transfer, repair, or other task family."},
    {"name": "taskPanel", "type": "string", "description": "Panel name used for S07 and later sweeps."},
    {"name": "n", "type": "integer", "description": "World size."},
    {"name": "inputProfile", "type": "string", "description": "Input distribution profile."},
    {"name": "frozenVariant", "type": "string", "description": "none, passive, stuck, or later perturbation label."},
    {"name": "frozenCount", "type": "integer", "description": "Number of Frozen Cells."},
    {"name": "replicateIndex", "type": "integer|null", "description": "Zero-based replicate index when available."},
    {"name": "sourceMetricSource", "type": "string", "description": "Simulation, artifact-ingest, or derived source."},
    {"name": "sourceRunId", "type": "string", "description": "Source run identifier for traceability."},
    {"name": "sourceArtifactPath", "type": "string|null", "description": "Input artifact path if ingested from a table."},
    {"name": "missingMetricReasons", "type": "object", "description": "Metric-id to missing reason mapping."},
)


def competence_schema() -> dict[str, Any]:
    return {
        "schema": "eidosoma.e03.s04.competence_vector_schema.v1",
        "version": COMPETENCE_VECTOR_VERSION,
        "normalizationVersion": NORMALIZATION_VERSION,
        "missingValuePolicy": MISSING_VALUE_POLICY,
        "identityFields": list(IDENTITY_FIELDS),
        "metricSpecs": [spec.to_dict() for spec in competence_metric_specs()],
        "uncertaintyFieldContract": {
            "method": "Group-level summaries use mean, sample SD, SEM, and normal-approximation 95% CI over available nonmissing values. Run-level vectors set uncertaintyN=1 where a metric is present.",
            "columns": list(UNCERTAINTY_COLUMNS),
            "missingHandling": "Missing values are omitted from metric-specific uncertainty summaries and counted in missingCount.",
        },
        "normalizationNotes": [
            "All normalized score columns are higher-is-better and bounded to [0, 1] unless direction is contextual.",
            "Efficiency denominators are simple bounded proxies for early morphospace screening, not optimality proofs.",
            "DG is a signed trajectory proxy transformed by atan for bounded ranking while preserving the raw DG value.",
            "Context-specific fields remain missing until their task panel defines the source measurement.",
        ],
    }


def vector_id_from_fields(fields: Mapping[str, Any]) -> str:
    keys = [
        "policyId",
        "taskId",
        "taskFamily",
        "replicateIndex",
        "sourceRunId",
        "sourceMetricSource",
        "frozenVariant",
        "frozenCount",
    ]
    payload = {key: fields.get(key) for key in keys}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
    return f"cv_{digest}"


def _trace_sortedness(result: SimulationResult) -> list[float]:
    return [float(row["sortedness_percent"]) for row in result.trace_rows]


def _trace_aggregation(result: SimulationResult) -> list[float]:
    values: list[float] = []
    for row in result.trace_rows:
        raw = row.get("algotypes_json")
        if raw is None:
            continue
        values.append(float(aggregation(parse_str_array(raw))))
    return values


def _missing_base() -> dict[str, str]:
    return {
        "dominanceScore": "not_a_conflict_task",
        "compatibilityScore": "not_a_chimera_panel",
        "transferScore": "not_a_transfer_panel",
        "repairSuccessScore": "not_a_repair_panel",
    }


def compute_competence_vector(
    result: SimulationResult,
    *,
    policy_id: str | None = None,
    policy_family: str | None = None,
    task_id: str | None = None,
    task_family: str = "sorting",
    task_panel: str = "toy",
    input_profile: str = "manual",
    frozen_variant: str | None = None,
    frozen_count: int | None = None,
    replicate_index: int | None = None,
    source_metric_source: str = "simulation_result",
    source_artifact_path: str | None = None,
    direction: str = "increasing",
    dominance_score: float | None = None,
    compatibility_score: float | None = None,
    transfer_score: float | None = None,
    repair_success_score: float | None = None,
    state_trajectory: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Compute one wide competence-vector row from a simulation result."""

    n = len(result.initial_values)
    max_pairs = max(1, n * (n - 1) // 2)
    n_squared = max(1, n * n)
    frozen_variant = frozen_variant if frozen_variant is not None else ("none" if not result.initial_frozen_positions else "unknown")
    frozen_count = frozen_count if frozen_count is not None else len(result.initial_frozen_positions)
    initial_metrics = prefixed_metrics("initial", result.initial_values, direction=direction)
    final_metrics = prefixed_metrics("final", result.final_values, direction=direction)
    sortedness_values = _trace_sortedness(result)
    aggregation_values = _trace_aggregation(result)
    dg = delayed_gratification_from_sortedness(sortedness_values)
    sign_change_rate = (
        sortedness_sign_change_count(sortedness_values) / max(1, len(sortedness_values) - 1)
        if len(sortedness_values) > 1
        else 0.0
    )
    curvature = trajectory_curvature_metrics(state_trajectory or [])
    missing = _missing_base()
    if not state_trajectory:
        missing["trajectoryExcessPathRatio"] = "state_trajectory_not_available"
        missing["pathDirectnessScore"] = "state_trajectory_not_available"
    if frozen_variant == "none" or int(frozen_count) == 0:
        missing["robustnessScore"] = "not_a_frozen_task"

    completion_success = 1.0 if result.completed else 0.0
    final_sortedness_score = bounded_unit(float(final_metrics["finalSortednessPercent"]) / 100.0)
    final_monotonicity_score = bounded_score(final_metrics["finalMonotonicityError"], max(1, n - 1))
    comparison_score = bounded_score(result.comparison_count, n_squared)
    activation_score = bounded_score(result.activation_count, n_squared)
    swap_score = bounded_score(result.swap_count, max_pairs)
    energy_proxy = float(result.swap_count + result.comparison_count + result.activation_count)
    oscillation_score = lower_is_better_unit(sign_change_rate)
    path_directness = None if not state_trajectory else float(1.0 / (1.0 + curvature["trajectoryExcessPathRatio"]))

    record: dict[str, Any] = {
        "competenceVectorVersion": COMPETENCE_VECTOR_VERSION,
        "normalizationVersion": NORMALIZATION_VERSION,
        "policyId": policy_id or result.algorithm,
        "policyFamily": policy_family or "classic",
        "algorithm": result.algorithm,
        "taskId": task_id or result.condition_id,
        "taskFamily": task_family,
        "taskPanel": task_panel,
        "n": int(n),
        "inputProfile": input_profile,
        "frozenVariant": frozen_variant,
        "frozenCount": int(frozen_count),
        "replicateIndex": replicate_index,
        "sourceMetricSource": source_metric_source,
        "sourceRunId": result.condition_id,
        "sourceArtifactPath": source_artifact_path,
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "completionSuccess": completion_success,
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "activationCount": int(result.activation_count),
        "blockedMoveAttempts": int(result.blocked_move_attempts),
        "frozenSwapAttempts": int(result.frozen_swap_attempts),
        "eventCount": int(result.event_count),
        "swapEfficiencyScore": swap_score,
        "comparisonEfficiencyScore": comparison_score,
        "activationEfficiencyScore": activation_score,
        "energyProxy": energy_proxy,
        "energyScore": bounded_score(energy_proxy, 3 * n_squared),
        "blockedMoveRate": float(result.blocked_move_attempts / max(1, result.activation_count)),
        "frozenSwapAttemptRate": float(result.frozen_swap_attempts / max(1, result.activation_count)),
        "robustnessScore": None
        if "robustnessScore" in missing
        else float(completion_success * (final_sortedness_score or 0.0)),
        "delayedGratification": float(dg["delayedGratification"]),
        "delayedGratificationScore": dg_score(dg["delayedGratification"]),
        "dgEventCount": int(dg["dgEventCount"]),
        "aggregationInitial": aggregation_values[0] if aggregation_values else None,
        "aggregationFinal": aggregation_values[-1] if aggregation_values else None,
        "aggregationPeak": max(aggregation_values) if aggregation_values else None,
        "aggregationAuc": float(np.mean(aggregation_values)) if aggregation_values else None,
        "aggregationPeakScore": bounded_unit(max(aggregation_values)) if aggregation_values else None,
        "aggregationAucScore": bounded_unit(float(np.mean(aggregation_values))) if aggregation_values else None,
        "dominanceScore": bounded_unit(dominance_score),
        "compatibilityScore": bounded_unit(compatibility_score),
        "transferScore": bounded_unit(transfer_score),
        "repairSuccessScore": bounded_unit(repair_success_score),
        "oscillationProxy": float(sign_change_rate),
        "oscillationScore": oscillation_score,
        "failureFlag": 0 if result.completed else 1,
        "failureOscillationScore": min(completion_success, oscillation_score if oscillation_score is not None else 1.0),
        "trajectoryExcessPathRatio": curvature["trajectoryExcessPathRatio"] if state_trajectory else None,
        "pathDirectnessScore": path_directness,
        "trajectoryStateCount": curvature["trajectoryStateCount"] if state_trajectory else None,
        "trajectoryTurnCount": curvature["trajectoryTurnCount"] if state_trajectory else None,
        "uncertaintyN": 1,
        "uncertaintyMethod": "single_run",
    }
    record.update(initial_metrics)
    record.update(final_metrics)
    record["finalSortednessScore"] = final_sortedness_score
    record["finalMonotonicityScore"] = final_monotonicity_score
    record["finalKendallTauScore"] = lower_is_better_unit(record.get("finalKendallTauDistanceNormalized"))
    record["finalEarthMoverPositionScore"] = lower_is_better_unit(record.get("finalEarthMoverPositionDistanceNormalized"))

    if dominance_score is not None:
        missing.pop("dominanceScore", None)
    if compatibility_score is not None:
        missing.pop("compatibilityScore", None)
    if transfer_score is not None:
        missing.pop("transferScore", None)
    if repair_success_score is not None:
        missing.pop("repairSuccessScore", None)
    record["missingMetricReasons"] = missing
    record["missingMetricReasonsJson"] = canonical_json(missing)
    record["vectorId"] = vector_id_from_fields(record)
    return record


def vector_from_summary_record(
    row: Mapping[str, Any],
    *,
    source_metric_source: str = "artifact_ingest",
    source_artifact_path: str | None = None,
) -> dict[str, Any]:
    """Create a competence row from an existing E01/E02 summary-like record."""

    def first(*keys: str, default: Any = None) -> Any:
        for key in keys:
            value = row.get(key)
            if value is not None:
                try:
                    if pd.isna(value):
                        continue
                except TypeError:
                    pass
                return value
        return default

    n = int(first("n", "finalN", "initialN", default=0) or 0)
    max_pairs = max(1, n * (n - 1) // 2)
    n_squared = max(1, n * n)
    final_sortedness = float(first("final_sortedness_percent", "finalSortednessPercent", default=math.nan))
    final_error = int(first("final_monotonicity_error", "finalMonotonicityError", default=0) or 0)
    swap_count = int(first("swap_count", "swapCount", default=0) or 0)
    comparison_count = int(first("comparison_count", "comparisonCount", default=0) or 0)
    activation_count = int(first("activation_count", "activationCount", default=first("eventCount", default=0)) or 0)
    completed = bool(first("completed", default=final_sortedness >= 100.0))
    frozen_variant = str(first("frozen_variant", "frozenVariant", default="none"))
    frozen_count = int(first("frozen_count", "frozenCount", default=0) or 0)
    dg_value = first("delayedGratification", default=None)
    missing = _missing_base()
    missing["trajectoryExcessPathRatio"] = "state_trajectory_not_available"
    missing["pathDirectnessScore"] = "state_trajectory_not_available"
    if frozen_variant == "none" or frozen_count == 0:
        missing["robustnessScore"] = "not_a_frozen_task"

    record: dict[str, Any] = {
        "competenceVectorVersion": COMPETENCE_VECTOR_VERSION,
        "normalizationVersion": NORMALIZATION_VERSION,
        "policyId": str(first("policyId", "algorithm", default="unknown")),
        "policyFamily": str(first("policyFamily", default="classic")),
        "algorithm": str(first("algorithm", default="unknown")),
        "taskId": str(first("taskId", "condition_id", "conditionId", "sourceConditionId", default="unknown")),
        "taskFamily": str(first("taskFamily", default="artifact_context")),
        "taskPanel": str(first("taskPanel", "sourceContext", "contextForResearchStepId", default="artifact_ingest")),
        "n": int(n),
        "inputProfile": str(first("input_profile", "inputProfile", default="unknown")),
        "frozenVariant": frozen_variant,
        "frozenCount": frozen_count,
        "replicateIndex": None if first("replicate_index", "replicateIndex", default=None) is None else int(first("replicate_index", "replicateIndex")),
        "sourceMetricSource": source_metric_source,
        "sourceRunId": str(first("sourceRunId", "condition_id", "conditionId", default="unknown")),
        "sourceArtifactPath": source_artifact_path,
        "completed": completed,
        "stopReason": str(first("stop_reason", "stopReason", default="unknown")),
        "completionSuccess": 1.0 if completed else 0.0,
        "swapCount": swap_count,
        "comparisonCount": comparison_count,
        "activationCount": activation_count,
        "eventCount": int(first("event_count", "eventCount", "trajectoryEventCount", default=0) or 0),
        "finalSortednessPercent": final_sortedness,
        "finalSortednessScore": bounded_unit(final_sortedness / 100.0),
        "finalMonotonicityError": final_error,
        "finalMonotonicityScore": bounded_score(final_error, max(1, n - 1)),
        "finalKendallTauDistanceNormalized": first("finalKendallTauDistanceNormalized", default=None),
        "finalKendallTauScore": lower_is_better_unit(first("finalKendallTauDistanceNormalized", default=None)),
        "finalEarthMoverPositionDistanceNormalized": first("finalEarthMoverPositionDistanceNormalized", default=None),
        "finalEarthMoverPositionScore": lower_is_better_unit(first("finalEarthMoverPositionDistanceNormalized", default=None)),
        "swapEfficiencyScore": bounded_score(swap_count, max_pairs),
        "comparisonEfficiencyScore": bounded_score(comparison_count, n_squared),
        "activationEfficiencyScore": bounded_score(activation_count, n_squared),
        "energyProxy": float(swap_count + comparison_count + activation_count),
        "energyScore": bounded_score(swap_count + comparison_count + activation_count, 3 * n_squared),
        "blockedMoveAttempts": int(first("blocked_move_attempts", "blockedMoveAttempts", default=0) or 0),
        "frozenSwapAttempts": int(first("frozen_swap_attempts", "frozenSwapAttempts", default=0) or 0),
        "blockedMoveRate": 0.0,
        "frozenSwapAttemptRate": 0.0,
        "robustnessScore": None if "robustnessScore" in missing else (1.0 if completed else 0.0) * bounded_unit(final_sortedness / 100.0),
        "delayedGratification": None if dg_value is None else float(dg_value),
        "delayedGratificationScore": dg_score(dg_value),
        "dgEventCount": first("dgEventCount", default=None),
        "aggregationInitial": first("initialAggregation", default=None),
        "aggregationFinal": first("final_aggregation", "finalAggregation", default=None),
        "aggregationPeak": first("peakAggregation", default=None),
        "aggregationAuc": first("aucAggregation", default=None),
        "aggregationPeakScore": bounded_unit(first("peakAggregation", default=None)),
        "aggregationAucScore": bounded_unit(first("aucAggregation", default=None)),
        "dominanceScore": first("dominanceScore", default=None),
        "compatibilityScore": first("compatibilityScore", default=None),
        "transferScore": first("transferScore", default=None),
        "repairSuccessScore": first("repairSuccessScore", default=None),
        "oscillationProxy": first("oscillationProxy", default=None),
        "oscillationScore": lower_is_better_unit(first("oscillationProxy", default=None)),
        "failureFlag": 0 if completed else 1,
        "failureOscillationScore": 1.0 if completed else 0.0,
        "trajectoryExcessPathRatio": first("trajectoryExcessPathRatio", default=None),
        "pathDirectnessScore": None,
        "trajectoryStateCount": first("trajectoryStateCount", default=None),
        "trajectoryTurnCount": first("trajectoryTurnCount", default=None),
        "uncertaintyN": 1,
        "uncertaintyMethod": "single_artifact_row",
    }
    if record["delayedGratification"] is not None:
        missing.pop("delayedGratification", None)
    else:
        missing["delayedGratification"] = "trajectory_or_dg_source_not_available"
        missing["delayedGratificationScore"] = "trajectory_or_dg_source_not_available"
    for key in ["dominanceScore", "compatibilityScore", "transferScore", "repairSuccessScore"]:
        if record.get(key) is not None:
            missing.pop(key, None)
    record["missingMetricReasons"] = missing
    record["missingMetricReasonsJson"] = canonical_json(missing)
    record["vectorId"] = vector_id_from_fields(record)
    return record


def summarize_competence_vectors(
    vectors: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("policyId", "taskFamily", "taskPanel"),
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(vectors)
    if df.empty:
        return pd.DataFrame()
    if metric_columns is None:
        metric_columns = [
            spec.normalized_column
            for spec in competence_metric_specs()
            if spec.normalized_column and spec.normalized_column in df.columns
        ]
    rows: list[dict[str, Any]] = []
    available_groups = [column for column in group_columns if column in df.columns]
    grouped = df.groupby(available_groups, dropna=False) if available_groups else [((), df)]
    for group_key, group in grouped:
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(available_groups, group_values))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = int(len(values))
            mean = float(values.mean()) if n else None
            sd = float(values.std(ddof=1)) if n > 1 else 0.0 if n == 1 else None
            sem = float(sd / math.sqrt(n)) if n and sd is not None else None
            rows.append(
                {
                    **base,
                    "metricColumn": metric,
                    "uncertaintyN": n,
                    "missingCount": int(len(group) - n),
                    "uncertaintyMean": mean,
                    "uncertaintySd": sd,
                    "uncertaintySem": sem,
                    "uncertaintyCi95Low": None if mean is None or sem is None else float(mean - 1.96 * sem),
                    "uncertaintyCi95High": None if mean is None or sem is None else float(mean + 1.96 * sem),
                    "uncertaintyMethod": "normal_approximation_over_replicates",
                }
            )
    return pd.DataFrame(rows)


def toy_competence_examples() -> pd.DataFrame:
    from .policies import NullPolicy, PolicyEventSimulator

    rows: list[dict[str, Any]] = []
    bubble = PolicyEventSimulator([3, 1, 2], "bubble", scheduler_seed=1, tie_breaker_seed=2).run()
    rows.append(
        compute_competence_vector(
            bubble,
            policy_id="classic_bubble",
            task_id="toy_bubble_three_cell",
            task_family="sorting",
            task_panel="toy_examples",
            input_profile="manual_unique",
            replicate_index=0,
            state_trajectory=[[3, 1, 2], [1, 3, 2], [1, 2, 3]],
        )
    )
    null = PolicyEventSimulator([3, 1, 2], NullPolicy(), scheduler_seed=1, tie_breaker_seed=2).run()
    rows.append(
        compute_competence_vector(
            null,
            policy_id="null_wait",
            policy_family="null",
            task_id="toy_null_three_cell",
            task_family="sorting",
            task_panel="toy_examples",
            input_profile="manual_unique",
            replicate_index=0,
        )
    )
    frozen = PolicyEventSimulator(
        [3, 1, 2],
        "bubble",
        frozen_positions=[1],
        frozen_variant="stuck",
        scheduler_seed=1,
        tie_breaker_seed=2,
    ).run(max_activations=200)
    rows.append(
        compute_competence_vector(
            frozen,
            policy_id="classic_bubble",
            task_id="toy_bubble_stuck_frozen",
            task_family="frozen",
            task_panel="toy_examples",
            input_profile="manual_unique",
            frozen_variant="stuck",
            frozen_count=1,
            replicate_index=0,
        )
    )
    return pd.DataFrame(rows)


def run_metric_unit_cases() -> tuple[bool, pd.DataFrame]:
    cases: list[dict[str, Any]] = []

    def record(case_id: str, observed: Any, expected: Any, passed: bool) -> None:
        cases.append({"caseId": case_id, "observed": observed, "expected": expected, "passed": bool(passed)})

    sorted_unique = state_distance_metrics([1, 2, 3, 4])
    record("sorted_unique_kendall", sorted_unique["kendallTauDistanceNormalized"], 0.0, math.isclose(sorted_unique["kendallTauDistanceNormalized"], 0.0))
    record("sorted_unique_edit", sorted_unique["editDistanceToTargetOrder"], 0, sorted_unique["editDistanceToTargetOrder"] == 0)
    reverse_unique = state_distance_metrics([4, 3, 2, 1])
    record("reverse_unique_inversions", reverse_unique["inversionCount"], 6, reverse_unique["inversionCount"] == 6)
    record("reverse_unique_kendall", reverse_unique["kendallTauDistanceNormalized"], 1.0, math.isclose(reverse_unique["kendallTauDistanceNormalized"], 1.0))
    duplicate = state_distance_metrics([2, 1, 2, 1])
    record("duplicate_comparable_pairs", duplicate["comparablePairCount"], 4, duplicate["comparablePairCount"] == 4)
    record("duplicate_kendall", duplicate["kendallTauDistanceNormalized"], 0.75, math.isclose(duplicate["kendallTauDistanceNormalized"], 0.75))
    curve = trajectory_curvature_metrics([[3, 1, 2], [1, 3, 2], [1, 2, 3]])
    record("trajectory_turn_count", curve["trajectoryTurnCount"], 1, curve["trajectoryTurnCount"] == 1)
    dg = delayed_gratification_from_sortedness([50.0, 60.0, 55.0, 70.0])
    record("dg_single_complete_gain", dg["delayedGratification"], 2.0, math.isclose(dg["delayedGratification"], 2.0))
    examples = toy_competence_examples()
    record("toy_examples_have_required_rows", len(examples), 3, len(examples) == 3)
    record("toy_bubble_sorts", examples.iloc[0]["completionSuccess"], 1.0, math.isclose(float(examples.iloc[0]["completionSuccess"]), 1.0))
    record("toy_null_records_failure", examples.iloc[1]["failureFlag"], 1, int(examples.iloc[1]["failureFlag"]) == 1)
    df = pd.DataFrame(cases)
    return bool(df["passed"].all()), df
