"""Prospective, outcome-independent S12R estimator-feasibility controls.

The frozen S12R scientific definitions remain external to this module.  The
module makes their declared domains executable: it validates native status and
pair support before arithmetic, emits explicit fixed non-evidentiary slots for
declared infeasibility, and rejects ambiguous evidentiary projections.

It deliberately contains no episode runner, cache reader, candidate selector,
or outcome-specific special case.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.phenotype_discovery.publication import (
    canonical_json_bytes,
    canonical_sha256,
)
from src.morph2d.elapsed_clock import (
    ElapsedClockValidationError,
    validate_elapsed_clock_summary,
    validate_first_completion_projection,
)

FIXED_FAMILIES = (
    "parent_compressed_calibrated_endpoint_family",
    "adaptation_calibrated_endpoint_family",
    "matched_random_calibrated_endpoint_family",
    "scheduler_calibrated_endpoint_family",
    "fault_repair_endpoint_family",
    "failure_harm_family",
    "native_cost_harm_family_by_component",
    "extension_cost_harm_family_by_component",
)

STATUS_CONTRACTS: Mapping[str, Mapping[str, bool]] = {
    "calibrated_endpoint_retained": {
        "endpointAvailable": True,
        "failed": False,
        "censored": False,
        "diagnostic": False,
    },
    "repair_observed": {
        "endpointAvailable": True,
        "failed": False,
        "censored": False,
        "diagnostic": False,
    },
    "right_censored_at_transition_32": {
        "endpointAvailable": True,
        "failed": False,
        "censored": True,
        "diagnostic": False,
    },
    "endpoint_unavailable_diagnostic_retained": {
        "endpointAvailable": False,
        "failed": False,
        "censored": False,
        "diagnostic": True,
    },
    "execution_or_invariant_failure_retained": {
        "endpointAvailable": False,
        "failed": True,
        "censored": False,
        "diagnostic": False,
    },
}

TASK_ENDPOINT_KINDS = frozenset(
    {
        "terminal_conjunctive_completion_binary",
        "minimum_mismatch_fraction_continuous",
        "repair_by_transition_32_binary",
        "departure_after_initial_completion_binary",
        "restricted_native_transition_time_at_32",
    }
)
ALL_RESERVED_ENDPOINT_KINDS = frozenset(
    {
        "reserved_failure_binary",
        "separate_cost_component_nonnegative_integer",
    }
)


class EstimatorFeasibilityError(RuntimeError):
    """A fixed estimator slot cannot be projected without ambiguity."""


_DENOMINATOR_NOT_SUPPLIED = object()


@dataclass(frozen=True)
class EndpointSpec:
    endpoint_kind: str
    output_endpoint: str
    source_column: str
    benefit_direction: int
    binary: bool


@dataclass(frozen=True)
class FixedSlot:
    family: str
    contrast_id: str
    task_id: str
    panel_id: str
    condition_id: str
    lineage_id: str
    endpoint: EndpointSpec
    left_label: str
    right_label: str

    def identity_payload(self) -> dict[str, str]:
        return {
            "family": self.family,
            "contrastId": self.contrast_id,
            "taskId": self.task_id,
            "panelId": self.panel_id,
            "conditionId": self.condition_id,
            "lineageId": self.lineage_id,
            "endpoint": self.endpoint.output_endpoint,
            "left": self.left_label,
            "right": self.right_label,
        }

    @property
    def test_id(self) -> str:
        return canonical_sha256(
            "E07/S12R/paired-test/v1",
            self.identity_payload(),
        )


ENDPOINT_SPECS: Mapping[str, EndpointSpec] = {
    "terminal_conjunctive_completion_binary": EndpointSpec(
        "terminal_conjunctive_completion_binary",
        "terminalConjunctiveCompletion",
        "terminalConjunctiveCompletion",
        1,
        True,
    ),
    "minimum_mismatch_fraction_continuous": EndpointSpec(
        "minimum_mismatch_fraction_continuous",
        "minimumMismatchFraction",
        "minimumMismatchFraction",
        -1,
        False,
    ),
    "repair_by_transition_32_binary": EndpointSpec(
        "repair_by_transition_32_binary",
        "repairByTransition32",
        "repairByTransition32",
        1,
        True,
    ),
    "departure_after_initial_completion_binary": EndpointSpec(
        "departure_after_initial_completion_binary",
        "departureAfterInitiallyComplete",
        "departureAfterInitiallyComplete",
        -1,
        True,
    ),
    "reserved_failure_binary": EndpointSpec(
        "reserved_failure_binary",
        "failed",
        "failed",
        -1,
        True,
    ),
    "separate_cost_component_nonnegative_integer": EndpointSpec(
        "separate_cost_component_nonnegative_integer",
        "separateCostComponent",
        "separateCostComponent",
        -1,
        False,
    ),
    "restricted_native_transition_time_at_32": EndpointSpec(
        "restricted_native_transition_time_at_32",
        "restrictedNativeTransitionTimeAt32",
        "firstCompletionTransition",
        -1,
        False,
    ),
}


def _strict_bool(value: Any, *, field: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    raise EstimatorFeasibilityError(f"{field}: exact boolean required")


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def validate_native_status(row: Mapping[str, Any]) -> str:
    status = row.get("status")
    if not isinstance(status, str) or status not in STATUS_CONTRACTS:
        raise EstimatorFeasibilityError("invalid or unknown native status")
    diagnostic = row.get(
        "diagnostic",
        status == "endpoint_unavailable_diagnostic_retained",
    )
    observed = {
        "endpointAvailable": _strict_bool(
            row.get("endpointAvailable"), field="endpointAvailable"
        ),
        "failed": _strict_bool(row.get("failed"), field="failed"),
        "censored": _strict_bool(row.get("censored"), field="censored"),
        "diagnostic": _strict_bool(diagnostic, field="diagnostic"),
    }
    if observed != STATUS_CONTRACTS[status]:
        raise EstimatorFeasibilityError(
            f"native status flags differ for {status}: {observed}"
        )
    return status


def endpoint_applies(endpoint_kind: str, native_status: str) -> bool:
    if endpoint_kind in ALL_RESERVED_ENDPOINT_KINDS:
        return native_status in STATUS_CONTRACTS
    if endpoint_kind in {
        "terminal_conjunctive_completion_binary",
        "minimum_mismatch_fraction_continuous",
        "restricted_native_transition_time_at_32",
    }:
        return native_status in {
            "calibrated_endpoint_retained",
            "repair_observed",
            "right_censored_at_transition_32",
        }
    if endpoint_kind == "repair_by_transition_32_binary":
        return native_status in {
            "repair_observed",
            "right_censored_at_transition_32",
        }
    if endpoint_kind == "departure_after_initial_completion_binary":
        return native_status == "calibrated_endpoint_retained"
    raise EstimatorFeasibilityError(f"unknown endpoint kind: {endpoint_kind}")


def classify_pre_arithmetic_state(
    *,
    endpoint_kind: str,
    native_status: str,
    denominator_state: str,
    pair_support_state: str,
) -> str:
    """Return the frozen S12V disposition before endpoint arithmetic."""

    if native_status not in STATUS_CONTRACTS:
        return "fail_closed_invalid_status_combination"
    if pair_support_state in {"duplicate_left", "duplicate_right"}:
        return "fail_closed_ambiguous_pair_identity"
    if pair_support_state == "endpoint_missing":
        return "fail_closed_missing_endpoint_before_arithmetic"
    if pair_support_state == "endpoint_nonfinite":
        return "fail_closed_nonfinite_endpoint_before_arithmetic"
    if denominator_state == "nonfinite":
        return "fail_closed_nonfinite_denominator"
    if not endpoint_applies(endpoint_kind, native_status):
        return "fixed_slot_non_evidentiary_endpoint_unavailable_or_inapplicable"
    if pair_support_state == "empty":
        return "fixed_slot_non_evidentiary_no_paired_rows"
    if pair_support_state in {"partial_left_only", "partial_right_only"}:
        return "fail_closed_incomplete_pair_support"
    if denominator_state in {"zero", "missing"}:
        return "fixed_slot_non_evidentiary_denominator_unavailable"
    if (
        pair_support_state == "complete_one_to_one"
        and denominator_state == "positive_finite"
    ):
        return "finite_evidentiary_record"
    raise EstimatorFeasibilityError("unclassified pre-arithmetic state")


def exact_one_to_one_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Pair without inner-intersection loss or ambiguous identity reuse."""

    key_list = list(keys)
    if not key_list:
        raise EstimatorFeasibilityError("at least one pair key is required")
    for side_name, frame in (("left", left), ("right", right)):
        missing_columns = sorted(set(key_list) - set(frame.columns))
        if missing_columns:
            raise EstimatorFeasibilityError(
                f"{side_name}: missing pair-key columns {missing_columns}"
            )
        if frame[key_list].isna().any(axis=None):
            raise EstimatorFeasibilityError(
                f"{side_name}: missing or ambiguous pair identity"
            )
        if frame.duplicated(key_list, keep=False).any():
            raise EstimatorFeasibilityError(
                f"{side_name}: duplicate or ambiguous pair identity"
            )
    if left.empty and right.empty:
        return left.merge(
            right,
            on=key_list,
            suffixes=("_left", "_right"),
            how="inner",
            validate="one_to_one",
        )
    left_keys = {
        tuple(values)
        for values in left[key_list].itertuples(index=False, name=None)
    }
    right_keys = {
        tuple(values)
        for values in right[key_list].itertuples(index=False, name=None)
    }
    if left_keys != right_keys:
        left_only = len(left_keys - right_keys)
        right_only = len(right_keys - left_keys)
        raise EstimatorFeasibilityError(
            "incomplete pair support; "
            f"left_only={left_only}, right_only={right_only}"
        )
    merged = left.merge(
        right,
        on=key_list,
        suffixes=("_left", "_right"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all() or len(merged) != len(left):
        raise EstimatorFeasibilityError("incomplete pair support after merge")
    return merged.drop(columns=["_merge"]).sort_values(key_list).reset_index(
        drop=True
    )


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or _is_missing(value):
        raise EstimatorFeasibilityError(
            f"{field}: missing or nonnumeric evidentiary value"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EstimatorFeasibilityError(
            f"{field}: nonnumeric evidentiary value"
        ) from exc
    if not math.isfinite(result):
        raise EstimatorFeasibilityError(f"{field}: non-finite evidentiary value")
    return result


def _strict_json_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, str) or not value:
        raise EstimatorFeasibilityError(f"{field}: authenticated JSON required")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EstimatorFeasibilityError(
            f"{field}: invalid authenticated JSON"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise EstimatorFeasibilityError(f"{field}: JSON object required")
    return parsed


def project_authenticated_restricted_time(
    row: Mapping[str, Any],
) -> float:
    """Project S12Z time only from authenticated elapsed-clock metadata."""

    status = validate_native_status(row)
    if not endpoint_applies("restricted_native_transition_time_at_32", status):
        raise EstimatorFeasibilityError(
            "restrictedNativeTransitionTimeAt32: status outside declared endpoint domain"
        )
    scenario_id = row.get("scenarioFamilyId")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise EstimatorFeasibilityError(
            "scenarioFamilyId: required for authenticated clock projection"
        )
    summary_raw = _strict_json_mapping(
        row.get("authenticatedElapsedClockSummaryJson"),
        field="authenticatedElapsedClockSummaryJson",
    )
    projection_raw = _strict_json_mapping(
        row.get("authenticatedFirstCompletionProjectionJson"),
        field="authenticatedFirstCompletionProjectionJson",
    )
    try:
        summary = validate_elapsed_clock_summary(
            summary_raw,
            expected_scenario_id=scenario_id,
            expected_horizon=32,
        )
        projection = validate_first_completion_projection(
            projection_raw,
            expected_scenario_id=scenario_id,
            expected_horizon=32,
            clock_summary=summary,
        )
    except ElapsedClockValidationError as exc:
        raise EstimatorFeasibilityError(
            f"authenticated elapsed-clock projection failed: {exc}"
        ) from exc
    if (
        projection["clockSummaryCommitmentSha256"]
        != summary["summaryCommitmentSha256"]
        or projection["clockFinalRecordCommitmentSha256"]
        != summary["finalRecordCommitmentSha256"]
    ):
        raise EstimatorFeasibilityError(
            "authenticated completion projection is detached from clock summary"
        )
    if row.get("rawObservationLabelsUsedAsScientificTime") is not False:
        raise EstimatorFeasibilityError(
            "raw observation labels cannot control scientific time"
        )
    first = projection["firstCompletionTransition"]
    if row.get("firstCompletionTransition") != first:
        raise EstimatorFeasibilityError(
            "persisted firstCompletionTransition disagrees with authenticated projection"
        )
    if status == "right_censored_at_transition_32":
        if first is not None:
            raise EstimatorFeasibilityError(
                "right-censored row cannot contain an authenticated completion event"
            )
        return 32.0
    return 32.0 if first is None else float(first)


def project_endpoint_value(
    row: Mapping[str, Any],
    spec: EndpointSpec,
) -> float:
    """Project one endpoint only after status/domain validation."""

    status = validate_native_status(row)
    if not endpoint_applies(spec.endpoint_kind, status):
        raise EstimatorFeasibilityError(
            f"{spec.output_endpoint}: status outside declared endpoint domain"
        )
    if spec.endpoint_kind == "restricted_native_transition_time_at_32":
        if (
            row.get("schemaVersion") == "e07.s12z.physical-result-projection.v2"
            or "authenticatedElapsedClockSummaryJson" in row
            or "authenticatedFirstCompletionProjectionJson" in row
        ):
            return project_authenticated_restricted_time(row)
        value = row.get(spec.source_column)
        if value is None:
            # The frozen registered estimand treats noncompletion as a
            # right-censor at transition 32, including calibrated no-fault
            # rows whose general episode status is not otherwise a censor.
            return 32.0
        if _is_missing(value):
            raise EstimatorFeasibilityError(
                "firstCompletionTransition: ambiguous missing scalar prohibited"
            )
        projected = _finite_number(value, field=spec.source_column)
        if not projected.is_integer() or not 0.0 <= projected <= 32.0:
            raise EstimatorFeasibilityError(
                "firstCompletionTransition must be an integer in [0,32]"
            )
        if (
            status == "right_censored_at_transition_32"
            and projected != 32.0
        ):
            raise EstimatorFeasibilityError(
                "right-censored transition time must project exactly to 32"
            )
        return min(projected, 32.0)

    value = row.get(spec.source_column)
    if spec.binary:
        return float(_strict_bool(value, field=spec.source_column))
    projected = _finite_number(value, field=spec.source_column)
    if spec.endpoint_kind == "minimum_mismatch_fraction_continuous":
        if not 0.0 <= projected <= 1.0:
            raise EstimatorFeasibilityError(
                "minimum mismatch fraction outside [0,1]"
            )
    elif (
        spec.endpoint_kind == "separate_cost_component_nonnegative_integer"
        and (projected < 0.0 or not projected.is_integer())
    ):
        raise EstimatorFeasibilityError(
            "separate cost component must be a nonnegative integer"
        )
    return projected


def _record_base(slot: FixedSlot) -> dict[str, Any]:
    if slot.family not in FIXED_FAMILIES:
        raise EstimatorFeasibilityError(
            f"slot uses unregistered multiplicity family: {slot.family}"
        )
    return {
        "testId": slot.test_id,
        "family": slot.family,
        "contrastId": slot.contrast_id,
        "taskId": slot.task_id,
        "panelId": slot.panel_id,
        "conditionId": slot.condition_id,
        "lineageId": slot.lineage_id,
        "endpoint": slot.endpoint.output_endpoint,
        "leftLabel": slot.left_label,
        "rightLabel": slot.right_label,
        "binary": slot.endpoint.binary,
        "benefitDirection": slot.endpoint.benefit_direction,
    }


def non_evidentiary_record(
    slot: FixedSlot,
    *,
    reason: str,
    pair_count: int = 0,
) -> dict[str, Any]:
    if not reason or not reason.isupper():
        raise EstimatorFeasibilityError(
            "non-evidentiary reason must be an explicit uppercase code"
        )
    if pair_count < 0:
        raise EstimatorFeasibilityError("pair count cannot be negative")
    record = {
        **_record_base(slot),
        "pairCount": pair_count,
        "rawEffectLeftMinusRight": None,
        "benefitEffect": None,
        "pairedBootstrap95CiLeftMinusRight": [None, None],
        "hodgesLehmannSensitivity": None,
        "rawPValue": 1.0,
        "evidentiary": False,
        "nonEvidentiaryReason": reason,
        "holmAdjustedPValue": None,
        "holmReject": False,
    }
    canonical_json_bytes(record)
    return record


def finite_evidentiary_record(
    slot: FixedSlot,
    *,
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 1 or right_array.ndim != 1:
        raise EstimatorFeasibilityError("paired endpoint arrays must be vectors")
    if left_array.shape != right_array.shape or len(left_array) == 0:
        raise EstimatorFeasibilityError(
            "evidentiary arrays require positive equal cardinality"
        )
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise EstimatorFeasibilityError(
            "non-finite evidentiary endpoint before arithmetic"
        )
    differences = left_array - right_array
    if not np.isfinite(differences).all():
        raise EstimatorFeasibilityError("non-finite paired difference")
    raw_effect = float(differences.mean())
    hl = float(np.median(differences))
    if not math.isfinite(raw_effect) or not math.isfinite(hl):
        raise EstimatorFeasibilityError("non-finite evidentiary summary")
    rng = np.random.Generator(np.random.PCG64(int(slot.test_id[:16], 16)))
    indices = rng.integers(
        0,
        len(differences),
        size=(2_000, len(differences)),
        endpoint=False,
    )
    boot = differences[indices].mean(axis=1)
    if not np.isfinite(boot).all():
        raise EstimatorFeasibilityError("non-finite paired bootstrap")
    ci = [
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]
    if slot.endpoint.binary:
        positive = int((differences > 0).sum())
        negative = int((differences < 0).sum())
        discordant = positive + negative
        raw_p = (
            1.0
            if discordant == 0
            else float(
                stats.binomtest(
                    positive,
                    discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
        )
    elif np.allclose(differences, 0.0):
        raw_p = 1.0
    else:
        raw_p = float(
            stats.wilcoxon(
                differences,
                zero_method="pratt",
                alternative="two-sided",
            ).pvalue
        )
    if not math.isfinite(raw_p) or not 0.0 <= raw_p <= 1.0:
        raise EstimatorFeasibilityError("invalid evidentiary p-value")
    record = {
        **_record_base(slot),
        "pairCount": len(differences),
        "rawEffectLeftMinusRight": raw_effect,
        "benefitEffect": raw_effect * slot.endpoint.benefit_direction,
        "pairedBootstrap95CiLeftMinusRight": ci,
        "hodgesLehmannSensitivity": hl,
        "rawPValue": raw_p,
        "evidentiary": True,
        "nonEvidentiaryReason": None,
        "holmAdjustedPValue": None,
        "holmReject": False,
    }
    canonical_json_bytes(record)
    return record


def _row_from_merged(
    row: Mapping[str, Any],
    side: str,
    keys: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    key_set = set(keys)
    suffix = f"_{side}"
    for column, value in row.items():
        if column in key_set:
            result[column] = value
        elif column.endswith(suffix):
            result[column[: -len(suffix)]] = value
    return result


def build_fixed_slot_record(
    slot: FixedSlot,
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    pair_keys: Sequence[str],
    denominator: Any = _DENOMINATOR_NOT_SUPPLIED,
) -> dict[str, Any]:
    """Build exactly one fixed slot, or fail before arithmetic."""

    paired = exact_one_to_one_pair(left, right, pair_keys)
    if paired.empty:
        return non_evidentiary_record(slot, reason="NO_PAIRED_ROWS")
    denominator_value: float
    if denominator is _DENOMINATOR_NOT_SUPPLIED:
        denominator_value = float(len(paired))
    elif denominator is None or denominator is pd.NA or denominator is pd.NaT:
        return non_evidentiary_record(
            slot,
            reason="DENOMINATOR_UNAVAILABLE",
            pair_count=len(paired),
        )
    else:
        denominator_value = _finite_number(denominator, field="denominator")
    if denominator_value == 0.0:
        return non_evidentiary_record(
            slot,
            reason="DENOMINATOR_UNAVAILABLE",
            pair_count=len(paired),
        )
    if denominator_value < 0.0:
        raise EstimatorFeasibilityError("denominator cannot be negative")

    merged_rows = paired.to_dict(orient="records")
    left_rows = [
        _row_from_merged(row, "left", pair_keys) for row in merged_rows
    ]
    right_rows = [
        _row_from_merged(row, "right", pair_keys) for row in merged_rows
    ]
    left_status = [validate_native_status(row) for row in left_rows]
    right_status = [validate_native_status(row) for row in right_rows]
    left_applies = [
        endpoint_applies(slot.endpoint.endpoint_kind, status)
        for status in left_status
    ]
    right_applies = [
        endpoint_applies(slot.endpoint.endpoint_kind, status)
        for status in right_status
    ]
    if not all(left_applies) or not all(right_applies):
        if all(left_applies) and not any(right_applies):
            reason = "RIGHT_CONDITION_OUTSIDE_ENDPOINT_DOMAIN"
        elif all(right_applies) and not any(left_applies):
            reason = "LEFT_CONDITION_OUTSIDE_ENDPOINT_DOMAIN"
        else:
            reason = "PAIR_CONTAINS_ENDPOINT_UNAVAILABLE_STATUS"
        return non_evidentiary_record(
            slot,
            reason=reason,
            pair_count=len(paired),
        )
    left_values = [
        project_endpoint_value(row, slot.endpoint) for row in left_rows
    ]
    right_values = [
        project_endpoint_value(row, slot.endpoint) for row in right_rows
    ]
    return finite_evidentiary_record(
        slot,
        left=left_values,
        right=right_values,
    )


def apply_fixed_holm(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_test_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Apply Holm without dropping any fixed slot or family."""

    mutable = [dict(record) for record in records]
    actual_ids = [str(record.get("testId")) for record in mutable]
    expected_ids = list(expected_test_ids)
    if len(set(actual_ids)) != len(actual_ids):
        raise EstimatorFeasibilityError("duplicate fixed test identity")
    if len(set(expected_ids)) != len(expected_ids):
        raise EstimatorFeasibilityError("duplicate expected fixed test identity")
    if set(actual_ids) != set(expected_ids):
        raise EstimatorFeasibilityError(
            "fixed Holm slot set differs from preregistration"
        )
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in mutable:
        if record.get("family") not in FIXED_FAMILIES:
            raise EstimatorFeasibilityError("record uses unknown fixed family")
        raw_p = _finite_number(record.get("rawPValue"), field="rawPValue")
        if not 0.0 <= raw_p <= 1.0:
            raise EstimatorFeasibilityError("raw p-value outside [0,1]")
        if (
            not record.get("evidentiary")
            and (raw_p != 1.0 or not record.get("nonEvidentiaryReason"))
        ):
            raise EstimatorFeasibilityError(
                "non-evidentiary fixed slot must be explicit raw p=1"
            )
        by_family[str(record["family"])].append(record)
    for family_rows in by_family.values():
        ordered = sorted(
            family_rows,
            key=lambda record: (
                float(record["rawPValue"]),
                str(record["testId"]),
            ),
        )
        running = 0.0
        total = len(ordered)
        for index, record in enumerate(ordered):
            adjusted = min(
                1.0,
                (total - index) * float(record["rawPValue"]),
            )
            running = max(running, adjusted)
            record["holmAdjustedPValue"] = running
            record["holmReject"] = bool(
                record["evidentiary"] and running <= 0.05
            )
    result = sorted(mutable, key=lambda record: str(record["testId"]))
    for record in result:
        canonical_json_bytes(record)
    return result


def validate_estimator_payload(
    payload: Mapping[str, Any],
    *,
    expected_test_ids: Sequence[str],
) -> None:
    """Reject non-finite, ambiguous, or family-shrunk scientific payloads."""

    if payload.get("schemaVersion") != "e07.s12w.paired-estimands.v1":
        raise EstimatorFeasibilityError("paired-estimator schema mismatch")
    if payload.get("taskLocalOnly") is not True:
        raise EstimatorFeasibilityError("task-local boundary missing")
    if payload.get("universalScore") is not None:
        raise EstimatorFeasibilityError("universal score is prohibited")
    if payload.get("fixedFamilies") != list(FIXED_FAMILIES):
        raise EstimatorFeasibilityError("fixed family registry changed")
    records = payload.get("records")
    if not isinstance(records, list):
        raise EstimatorFeasibilityError("records must be a list")
    actual_ids = [record.get("testId") for record in records]
    if set(actual_ids) != set(expected_test_ids) or len(actual_ids) != len(
        expected_test_ids
    ):
        raise EstimatorFeasibilityError("fixed test family was shrunk or changed")
    for record in records:
        if record.get("evidentiary"):
            for field in (
                "rawEffectLeftMinusRight",
                "benefitEffect",
                "hodgesLehmannSensitivity",
                "rawPValue",
                "holmAdjustedPValue",
            ):
                _finite_number(record.get(field), field=field)
        else:
            if (
                record.get("rawPValue") != 1.0
                or record.get("nonEvidentiaryReason") is None
            ):
                raise EstimatorFeasibilityError(
                    "invalid non-evidentiary fixed slot"
                )
    canonical_json_bytes(payload)
