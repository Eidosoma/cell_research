"""Versioned, outcome-free native event-feature extraction for E07 S10P.

The extractor has three deliberately separate planes:

* analysis features: task-native event counts/rates only;
* confounds: task, native status, horizon/length, and trace availability;
* availability/provenance: one explicit record for every registered feature.

It accepts no outcome envelope and never manufactures an ordered sequence from
terminal summaries.  Cost-ledger fields are used only where they are the native
audited event-count projection; no cross-family total is computed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence


EVENT_FEATURE_SCHEMA_VERSION = "e07.s10p.native-event-features.v1"
REGISTRY_SCHEMA_VERSION = "e07.s10p.native-event-feature-registry.v1"
FROZEN_UNAVAILABLE_REASONS = frozenset(
    {
        "source_field_absent",
        "phase_not_reached",
        "source_terminal",
        "right_censored_before_phase",
        "structurally_not_applicable",
        "ordered_event_summary_unavailable",
    }
)

TASKS = (
    "e07_s02_sorting_1d",
    "e07_s02_faults_1d",
    "e07_s02_detour_1d",
    "e07_s02_chimera_1d",
    "e07_s02_regeneration_1d",
    "e07_s02_target_change_1d",
    "e07_s02_spatial2d_local",
    "e07_s02_spatial2d_memory",
)

PROHIBITED_INPUT_KEYS = frozenset(
    {
        "outcome",
        "nativeOutcome",
        "objective",
        "objectives",
        "validationOutcome",
        "confirmationOutcome",
        "protectedConfirmationOutcome",
    }
)
PROHIBITED_PATH_TOKENS = frozenset(
    {
        "outcome",
        "objective",
        "descriptor",
        "descriptors",
        "completion",
        "distance",
        "aggregation",
        "score",
    }
)


class FeatureExtractionError(ValueError):
    """Raised when an event projection is missing authority or is malformed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(value)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    task_id: str
    feature_id: str
    source_path: str
    transform: str = "direct"
    denominator_path: str | None = None
    source_authority: str = "native_event_or_audited_native_event_count_ledger"
    analysis_plane: str = "task_local_event_feature"
    cross_task_pooling: bool = False

    def __post_init__(self) -> None:
        if self.task_id not in TASKS:
            raise FeatureExtractionError(
                f"unknown task in feature registry: {self.task_id}"
            )
        if self.transform not in {"direct", "ratio", "ordered_summary"}:
            raise FeatureExtractionError(f"unknown feature transform: {self.transform}")
        tokens = {
            token
            for path in (self.source_path, self.denominator_path or "")
            for token in path.replace("[", ".").replace("]", "").split(".")
        }
        if tokens & PROHIBITED_PATH_TOKENS:
            raise FeatureExtractionError(
                f"prohibited efficacy/descriptor path in registry: {self.feature_id}"
            )
        if self.cross_task_pooling:
            raise FeatureExtractionError("cross-task feature pooling is forbidden")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _line_specs(task_id: str, family: str) -> list[FeatureSpec]:
    base = f"costs.{family}"
    denominator = f"{base}.activations"
    fields = (
        ("proposal_rate", "proposals"),
        ("accepted_swap_rate", "acceptedSwaps"),
        ("rejection_rate", "rejections"),
        ("noop_rate", "noOps"),
        ("memory_update_rate", "memoryUpdates"),
        ("observation_read_rate", "observationReads"),
        ("comparison_rate", "valueComparisons"),
        ("conflict_loss_rate", "conflictLosses"),
        ("displaced_cell_rate", "displacedCells"),
    )
    return [
        FeatureSpec(
            task_id,
            f"{task_id}.event.{feature}",
            f"{base}.{field}",
            "ratio",
            denominator,
        )
        for feature, field in fields
    ]


def build_feature_registry() -> tuple[FeatureSpec, ...]:
    """Return the frozen task-local registry in canonical order."""

    specs: list[FeatureSpec] = []
    specs.extend(_line_specs("e07_s02_sorting_1d", "e01ReferenceLedger"))
    specs.extend(_line_specs("e07_s02_faults_1d", "e01ReferenceLedger"))
    specs.append(
        FeatureSpec(
            "e07_s02_faults_1d",
            "e07_s02_faults_1d.event.fault_audit_per_activation",
            "event.faultAuditCount",
            "ratio",
            "costs.e01ReferenceLedger.activations",
        )
    )
    specs.extend(_line_specs("e07_s02_detour_1d", "e01ReferenceLedger"))
    specs.extend(_line_specs("e07_s02_chimera_1d", "e01ReferenceLedger"))

    for phase in (
        "development",
        "stabilization",
        "recovery",
        "robustnessFault",
        "transfer",
    ):
        family = f"e05_{phase}"
        denominator = f"costs.{family}.activations"
        for feature, field in (
            ("accepted_swap_rate", "acceptedSwaps"),
            ("rejection_rate", "rejections"),
            ("noop_rate", "noOps"),
            ("memory_update_rate", "memoryUpdates"),
            ("observation_read_rate", "observationReads"),
            ("comparison_rate", "valueComparisons"),
            ("displaced_cell_rate", "displacedCells"),
        ):
            specs.append(
                FeatureSpec(
                    "e07_s02_regeneration_1d",
                    f"e07_s02_regeneration_1d.{phase}.{feature}",
                    f"costs.{family}.{field}",
                    "ratio",
                    denominator,
                )
            )

    target_denominator = "costs.e01ReferenceLedgerDelta.activations"
    for feature, path in (
        ("accepted_swap_rate", "costs.e01ReferenceLedgerDelta.acceptedSwaps"),
        ("rejection_rate", "costs.e01ReferenceLedgerDelta.rejections"),
        ("noop_rate", "costs.e01ReferenceLedgerDelta.noOps"),
        ("memory_update_rate", "costs.e01ReferenceLedgerDelta.memoryUpdates"),
        ("observation_read_rate", "costs.e01ReferenceLedgerDelta.observationReads"),
        ("comparison_rate", "costs.e01ReferenceLedgerDelta.valueComparisons"),
        ("target_signal_query_rate", "costs.e05TargetSignalLedger.signalQueries"),
        ("target_signal_delivery_rate", "costs.e05TargetSignalLedger.signalDeliveries"),
        ("target_record_read_rate", "costs.e05TargetSignalLedger.targetRecordReads"),
        (
            "controller_computation_rate",
            "costs.e05TargetSignalLedger.controllerComputations",
        ),
        (
            "target_aware_candidate_rate",
            "costs.e05TargetSignalLedger.targetAwareCandidates",
        ),
    ):
        specs.append(
            FeatureSpec(
                "e07_s02_target_change_1d",
                f"e07_s02_target_change_1d.event.{feature}",
                path,
                "ratio",
                target_denominator,
            )
        )

    spatial_ledger_fields = (
        "adjacentSwaps",
        "vacancyMoves",
        "shortExchanges",
        "rotations",
        "validProposals",
        "conflictCandidates",
        "reservedSiteClaims",
        "totalGraphDisplacement",
    )
    ordered_fields = (
        "proposal_count_mean",
        "proposal_count_sd",
        "accepted_count_mean",
        "accepted_count_sd",
        "conflict_loss_mean",
        "invalid_proposal_mean",
        "state_hash_change_fraction",
        "state_hash_change_switch_fraction",
        "longest_state_hash_change_run_fraction",
        "accepted_zero_transition_fraction",
    )
    for task_id in ("e07_s02_spatial2d_local", "e07_s02_spatial2d_memory"):
        denominator = "costs.e06MovementLedger.submittedProposals"
        for field in spatial_ledger_fields:
            specs.append(
                FeatureSpec(
                    task_id,
                    f"{task_id}.movement.{field}",
                    f"costs.e06MovementLedger.{field}",
                    "ratio",
                    denominator,
                )
            )
        for field in ordered_fields:
            specs.append(
                FeatureSpec(
                    task_id,
                    f"{task_id}.ordered.{field}",
                    f"orderedEventSummaries.{field}",
                    "ordered_summary",
                )
            )

    ordered = tuple(sorted(specs, key=lambda item: (item.task_id, item.feature_id)))
    if len({(item.task_id, item.feature_id) for item in ordered}) != len(ordered):
        raise FeatureExtractionError("duplicate feature registry entry")
    return ordered


def registry_document() -> dict[str, Any]:
    specs = build_feature_registry()
    by_task = {
        task_id: sum(item.task_id == task_id for item in specs) for task_id in TASKS
    }
    return {
        "schemaVersion": REGISTRY_SCHEMA_VERSION,
        "featureSchemaVersion": EVENT_FEATURE_SCHEMA_VERSION,
        "crossTaskPooling": False,
        "universalScore": None,
        "outcomePathsPermitted": False,
        "objectivePathsPermitted": False,
        "s04DescriptorPathsPermitted": False,
        "featureCount": len(specs),
        "featureCountByTask": by_task,
        "features": [item.to_dict() for item in specs],
    }


def _path_get(payload: Mapping[str, Any], path: str) -> Any:
    cursor: Any = payload
    for token in path.split("."):
        if not isinstance(cursor, Mapping) or token not in cursor:
            raise KeyError(path)
        cursor = cursor[token]
    return cursor


def _numeric(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureExtractionError(f"{path} must be numeric and non-boolean")
    result = float(value)
    if not math.isfinite(result):
        raise FeatureExtractionError(f"{path} must be finite")
    if result < 0:
        raise FeatureExtractionError(f"{path} must be nonnegative")
    return result


def validate_ordered_transition_summaries(
    summaries: Sequence[Mapping[str, Any]], *, expected_count: int = 32
) -> dict[str, Any]:
    """Validate authentic ordered E06 summaries without inferring state content."""

    if len(summaries) != expected_count:
        raise FeatureExtractionError(
            f"ordered transition count {len(summaries)} != {expected_count}"
        )
    required = {
        "transitionIndex",
        "scheduledActorCount",
        "proposalCount",
        "acceptedCount",
        "conflictLosses",
        "invalidProposals",
        "preStateSha256",
        "postStateSha256",
        "transitionSha256",
    }
    previous_post: str | None = None
    for index, item in enumerate(summaries):
        if not required.issubset(item):
            raise FeatureExtractionError(
                f"ordered summary {index} missing {sorted(required - set(item))}"
            )
        if int(item["transitionIndex"]) != index:
            raise FeatureExtractionError(
                "ordered transition indices must be contiguous"
            )
        for field in (
            "scheduledActorCount",
            "proposalCount",
            "acceptedCount",
            "conflictLosses",
            "invalidProposals",
        ):
            _numeric(item[field], f"orderedEventSummaries[{index}].{field}")
        for field in ("preStateSha256", "postStateSha256", "transitionSha256"):
            value = str(item[field])
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise FeatureExtractionError(
                    f"orderedEventSummaries[{index}].{field} must be SHA-256"
                )
        if previous_post is not None and str(item["preStateSha256"]) != previous_post:
            raise FeatureExtractionError(
                "ordered transition state-hash chain is broken"
            )
        previous_post = str(item["postStateSha256"])
    return {
        "count": len(summaries),
        "stateHashChainPass": True,
        "contentSha256": canonical_sha256(
            "E07/S10P/ordered-transition-summaries/v1", list(summaries)
        ),
    }


def _longest_true_run(values: Sequence[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _ordered_features(summaries: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    validate_ordered_transition_summaries(summaries)
    proposals = [float(item["proposalCount"]) for item in summaries]
    accepted = [float(item["acceptedCount"]) for item in summaries]
    conflicts = [float(item["conflictLosses"]) for item in summaries]
    invalid = [float(item["invalidProposals"]) for item in summaries]
    changed = [
        str(item["preStateSha256"]) != str(item["postStateSha256"])
        for item in summaries
    ]
    return {
        "proposal_count_mean": fmean(proposals),
        "proposal_count_sd": pstdev(proposals),
        "accepted_count_mean": fmean(accepted),
        "accepted_count_sd": pstdev(accepted),
        "conflict_loss_mean": fmean(conflicts),
        "invalid_proposal_mean": fmean(invalid),
        "state_hash_change_fraction": fmean(map(float, changed)),
        "state_hash_change_switch_fraction": sum(
            left != right for left, right in zip(changed, changed[1:])
        )
        / max(1, len(changed) - 1),
        "longest_state_hash_change_run_fraction": _longest_true_run(changed)
        / len(changed),
        "accepted_zero_transition_fraction": sum(value == 0 for value in accepted)
        / len(accepted),
    }


def _event_length(task_id: str, payload: Mapping[str, Any]) -> int | None:
    summaries = payload.get("orderedEventSummaries")
    if isinstance(summaries, Sequence) and not isinstance(summaries, (str, bytes)):
        return len(summaries)
    event = payload.get("event", {})
    if not isinstance(event, Mapping):
        return None
    for path in ("activationCount", "distanceProfileCount", "transitionCount"):
        value = event.get(path)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    if task_id == "e07_s02_regeneration_1d":
        phase_rows = event.get("phaseRows")
        if isinstance(phase_rows, Sequence):
            return sum(
                int(item.get("opportunities", 0))
                for item in phase_rows
                if isinstance(item, Mapping)
            )
    return None


def extract_native_event_features(
    task_id: str,
    payload: Mapping[str, Any],
    *,
    availability_reasons: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Extract a canonical event-feature record without reading outcomes."""

    if task_id not in TASKS:
        raise FeatureExtractionError(f"unknown task: {task_id}")
    forbidden = sorted(set(payload) & PROHIBITED_INPUT_KEYS)
    if forbidden:
        raise FeatureExtractionError(
            f"outcome/protected planes are forbidden inputs: {forbidden}"
        )
    required = {"event", "costs", "status", "horizon", "structuralScenarioSize"}
    if not required.issubset(payload):
        raise FeatureExtractionError(
            f"feature payload missing {sorted(required - set(payload))}"
        )
    status = payload["status"]
    if not isinstance(status, Mapping):
        raise FeatureExtractionError("status plane must be a mapping")
    for key in ("stopReason", "failed", "censored"):
        if key not in status:
            raise FeatureExtractionError(f"status.{key} is required")
    if not isinstance(status["failed"], bool) or not isinstance(
        status["censored"], bool
    ):
        raise FeatureExtractionError("failed/censored status must be boolean")

    summaries = payload.get("orderedEventSummaries")
    ordered: dict[str, float] = {}
    ordered_available = summaries is not None
    if ordered_available:
        if task_id not in {
            "e07_s02_spatial2d_local",
            "e07_s02_spatial2d_memory",
        }:
            raise FeatureExtractionError(
                "ordered event summaries are unauthorized for this task contract"
            )
        if not isinstance(summaries, Sequence) or isinstance(summaries, (str, bytes)):
            raise FeatureExtractionError("orderedEventSummaries must be a sequence")
        ordered = _ordered_features(summaries)

    reasons = dict(availability_reasons or {})
    features: dict[str, float] = {}
    availability: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, dict[str, Any]] = {}
    for spec in build_feature_registry():
        if spec.task_id != task_id:
            continue
        reason = reasons.get(spec.feature_id)
        try:
            if spec.transform == "ordered_summary":
                if not ordered_available:
                    raise KeyError(spec.source_path)
                value = ordered[spec.source_path.rsplit(".", 1)[-1]]
            else:
                numerator = _numeric(
                    _path_get(payload, spec.source_path), spec.source_path
                )
                if spec.transform == "direct":
                    value = numerator
                else:
                    assert spec.denominator_path is not None
                    denominator = _numeric(
                        _path_get(payload, spec.denominator_path),
                        spec.denominator_path,
                    )
                    if denominator == 0:
                        raise ZeroDivisionError(spec.denominator_path)
                    value = numerator / denominator
            if not math.isfinite(value):
                raise FeatureExtractionError(
                    f"nonfinite derived feature {spec.feature_id}"
                )
            features[spec.feature_id] = float(value)
            availability[spec.feature_id] = {
                "state": "observed_or_exact_native_derived",
                "reason": None,
            }
        except ZeroDivisionError:
            frozen_reason = reason or "structurally_not_applicable"
            if frozen_reason not in FROZEN_UNAVAILABLE_REASONS:
                raise FeatureExtractionError(
                    f"unknown feature-unavailability reason: {frozen_reason}"
                )
            availability[spec.feature_id] = {
                "state": "unavailable",
                "reason": frozen_reason,
                "reasonCode": (
                    frozen_reason if reason is not None else "zero_native_denominator"
                ),
            }
        except KeyError:
            frozen_reason = reason or (
                "ordered_event_summary_unavailable"
                if spec.transform == "ordered_summary"
                else "source_field_absent"
            )
            if frozen_reason not in FROZEN_UNAVAILABLE_REASONS:
                raise FeatureExtractionError(
                    f"unknown feature-unavailability reason: {frozen_reason}"
                )
            availability[spec.feature_id] = {
                "state": "unavailable",
                "reason": frozen_reason,
                "reasonCode": frozen_reason,
            }
        source_paths[spec.feature_id] = {
            "sourcePath": spec.source_path,
            "denominatorPath": spec.denominator_path,
            "transform": spec.transform,
        }

    event_length = _event_length(task_id, payload)
    if event_length is None:
        event_length_state = "unavailable"
    else:
        if event_length < 0:
            raise FeatureExtractionError("observed event length must be nonnegative")
        event_length_state = "observed"
    confounds = {
        "taskId": task_id,
        "stopReason": str(status["stopReason"]),
        "failed": bool(status["failed"]),
        "censored": bool(status["censored"]),
        "nativeHorizon": payload["horizon"],
        "observedEventLength": event_length,
        "observedEventLengthState": event_length_state,
        "structuralScenarioSize": int(payload["structuralScenarioSize"]),
        "traceAvailability": (
            "authentic_ordered_event_summary"
            if ordered_available
            else "terminal_summary_only"
        ),
        "traceSelectionReason": str(
            payload.get("traceSelectionReason", "all_event_summaries")
        ),
    }
    body = {
        "schemaVersion": EVENT_FEATURE_SCHEMA_VERSION,
        "taskId": task_id,
        "analysisFeatures": dict(sorted(features.items())),
        "confounds": confounds,
        "availability": dict(sorted(availability.items())),
        "provenance": {
            "sourceEventSchemaVersion": str(
                payload.get("event", {}).get("schemaVersion", "unknown")
            ),
            "sourcePaths": dict(sorted(source_paths.items())),
            "outcomePlaneRead": False,
            "objectivePlaneRead": False,
            "s04DescriptorPlaneRead": False,
            "crossTaskPooling": False,
        },
    }
    body["featureRecordSha256"] = canonical_sha256(
        "E07/S10P/native-event-feature-record/v1", body
    )
    return body


def validate_feature_availability_record(
    record: Mapping[str, Any],
    *,
    expected_task_id: str | None = None,
) -> dict[str, Any]:
    """Validate complete availability without requiring per-row completeness."""

    task_id = str(record.get("taskId", ""))
    if expected_task_id is not None and task_id != expected_task_id:
        raise FeatureExtractionError(
            f"feature task mismatch: expected {expected_task_id}, got {task_id}"
        )
    registered = sorted(
        spec.feature_id for spec in build_feature_registry() if spec.task_id == task_id
    )
    if not registered:
        raise FeatureExtractionError(f"no registered features for task: {task_id}")
    analysis = record.get("analysisFeatures")
    availability = record.get("availability")
    if not isinstance(analysis, Mapping) or not isinstance(availability, Mapping):
        raise FeatureExtractionError(
            "analysisFeatures and availability must both be mappings"
        )
    if sorted(availability) != registered:
        missing = sorted(set(registered) - set(availability))
        extra = sorted(set(availability) - set(registered))
        raise FeatureExtractionError(
            f"availability plane does not equal registry; missing={missing}; extra={extra}"
        )
    if not set(analysis).issubset(registered):
        raise FeatureExtractionError("analysis plane contains an unregistered feature")

    reason_counts: dict[str, int] = {}
    observed = 0
    unavailable = 0
    for feature_id in registered:
        state_record = availability[feature_id]
        if not isinstance(state_record, Mapping):
            raise FeatureExtractionError(
                f"availability record is not a mapping: {feature_id}"
            )
        state = state_record.get("state")
        if state == "observed_or_exact_native_derived":
            if feature_id not in analysis:
                raise FeatureExtractionError(
                    f"observed availability lacks analysis value: {feature_id}"
                )
            value = analysis[feature_id]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise FeatureExtractionError(
                    f"observed analysis value is not finite numeric: {feature_id}"
                )
            if state_record.get("reason") is not None:
                raise FeatureExtractionError(
                    f"observed availability has an unavailable reason: {feature_id}"
                )
            observed += 1
        elif state == "unavailable":
            if feature_id in analysis:
                raise FeatureExtractionError(
                    f"unavailable feature has an analysis value: {feature_id}"
                )
            reason = state_record.get("reason")
            if reason not in FROZEN_UNAVAILABLE_REASONS:
                raise FeatureExtractionError(
                    f"unavailable feature has invalid frozen reason: {feature_id}"
                )
            reason_code = state_record.get("reasonCode", reason)
            if not isinstance(reason_code, str) or not reason_code:
                raise FeatureExtractionError(
                    f"unavailable feature lacks a reason code: {feature_id}"
                )
            reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
            unavailable += 1
        else:
            raise FeatureExtractionError(
                f"unknown availability state for {feature_id}: {state!r}"
            )
    return {
        "schemaVersion": "e07.s10a.feature-availability-summary.v1",
        "state": (
            "complete" if unavailable == 0 else "complete_with_explicit_unavailability"
        ),
        "registeredFeatureCount": len(registered),
        "observedFeatureCount": observed,
        "unavailableFeatureCount": unavailable,
        "reasonCodeCounts": dict(sorted(reason_counts.items())),
        "imputedFeatureCount": 0,
        "silentDropCount": 0,
    }


def exact_change_points(
    series: Sequence[Sequence[float]],
    *,
    minimum_segment_length: int = 4,
    maximum_change_points: int = 3,
) -> tuple[int, ...]:
    """Exact deterministic piecewise-constant SSE segmentation with BIC penalty."""

    rows = tuple(tuple(float(value) for value in row) for row in series)
    if not rows:
        raise FeatureExtractionError("change-point series must be nonempty")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise FeatureExtractionError("change-point series must be rectangular")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise FeatureExtractionError("change-point series must be finite")
    n = len(rows)
    if n < 2 * minimum_segment_length:
        return ()

    prefix = [[0.0] * (n + 1) for _ in range(width)]
    prefix_sq = [[0.0] * (n + 1) for _ in range(width)]
    for column in range(width):
        for index, row in enumerate(rows, 1):
            value = row[column]
            prefix[column][index] = prefix[column][index - 1] + value
            prefix_sq[column][index] = prefix_sq[column][index - 1] + value * value

    def segment_cost(start: int, end: int) -> float:
        length = end - start
        cost = 0.0
        for column in range(width):
            total = prefix[column][end] - prefix[column][start]
            total_sq = prefix_sq[column][end] - prefix_sq[column][start]
            cost += max(0.0, total_sq - total * total / length)
        return cost

    penalty = (width + 1) * math.log(n)
    best: tuple[float, tuple[int, ...]] | None = None

    def visit(start: int, remaining: int, points: tuple[int, ...]) -> None:
        nonlocal best
        if remaining == 0:
            bounds = (0,) + points + (n,)
            value = sum(
                segment_cost(left, right) for left, right in zip(bounds, bounds[1:])
            ) + penalty * len(points)
            candidate = (value, points)
            if best is None or candidate < best:
                best = candidate
            return
        minimum = start + minimum_segment_length
        maximum = n - remaining * minimum_segment_length
        for point in range(minimum, maximum + 1):
            visit(point, remaining - 1, points + (point,))

    for count in range(maximum_change_points + 1):
        if (count + 1) * minimum_segment_length > n:
            break
        if count == 0:
            candidate = (segment_cost(0, n), ())
            if best is None or candidate < best:
                best = candidate
        else:
            visit(0, count, ())
    assert best is not None
    return best[1]
