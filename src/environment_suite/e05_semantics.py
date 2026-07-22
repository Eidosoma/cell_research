"""Outcome-independent E05 target-change semantic contracts."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import canonical_sha256


TARGET_CHANGE_SEMANTICS_VERSION = "e07.s08h.e05-target-deadline-probe.v1"
TARGET_CHANGE_AUDIT_PROJECTION_SCHEMA_VERSION = (
    "e07.s08j.e05-target-change-audit-projection.v1"
)
_TARGET_CHANGE_PROJECTION_DOMAIN = "E07/S08J/E05-target-audit-projection/v1"
_TARGET_CHANGE_SOURCE_FIELDS = {
    "targetChangeSemanticsVersion",
    "stopReason",
    "targetCompleted",
    "phaseActivationCount",
    "adaptationTime",
    "adaptationCensored",
    "overshootCensored",
    "postHitProbeOpportunities",
    "postHitProbeRetained",
    "postHitProbeApplicable",
    "targetChangeSemanticAudit",
}
_TARGET_CHANGE_PROJECTION_FIELDS = {
    "schemaVersion",
    "metadataOrigin",
    "targetChangeSemanticsVersion",
    "adaptationBudget",
    "probeBudget",
    "stopReason",
    "targetCompleted",
    "phaseActivationCount",
    "adaptationTime",
    "adaptationCensored",
    "overshootCensored",
    "postHitProbeOpportunities",
    "postHitProbeRetained",
    "postHitProbeApplicable",
    "targetChangeSemanticAudit",
    "sourceResultSha256",
    "projectionSha256",
}


def target_change_terminal_transition(
    *,
    elapsed: int,
    distance: int,
    first_hit: int | None,
    quiescent: bool,
    invariant_error: bool,
    adaptation_budget: int,
    probe_budget: int,
) -> tuple[int | None, str | None]:
    """Apply the frozen deadline/probe terminal state machine."""

    if invariant_error:
        return first_hit, "invariant_error"
    if elapsed < 0 or adaptation_budget < 1 or probe_budget < 1:
        raise ValueError("invalid target-change clock or budget")
    if first_hit is None and distance == 0:
        if elapsed > adaptation_budget:
            raise ValueError("target hit observed after the adaptation deadline")
        first_hit = elapsed
    if first_hit is not None:
        if not 0 <= first_hit <= adaptation_budget or elapsed < first_hit:
            raise ValueError("invalid target-change hit clock")
        if elapsed - first_hit >= probe_budget:
            if elapsed - first_hit != probe_budget:
                raise ValueError("post-hit probe exceeded its exact budget")
            return first_hit, "post_adaptation_probe_complete"
        return first_hit, None
    if quiescent:
        return None, "controller_quiescent"
    if elapsed >= adaptation_budget:
        if elapsed != adaptation_budget:
            raise ValueError("adaptation deadline was overrun without a target hit")
        return None, "phase_event_budget"
    return None, None


def validate_target_change_result_semantics(
    result: Mapping[str, Any], *, adaptation_budget: int, probe_budget: int
) -> dict[str, Any]:
    """Classify a target-change return without relabeling its native status."""

    stop = str(result.get("stopReason"))
    completed = bool(result.get("targetCompleted"))
    adaptation_censored = bool(result.get("adaptationCensored"))
    overshoot_censored = bool(result.get("overshootCensored"))
    retained = bool(result.get("postHitProbeRetained"))
    phase = int(result.get("phaseActivationCount", -1))
    probe_count = int(result.get("postHitProbeOpportunities", -1))
    hit_raw = result.get("adaptationTime")
    hit = None if hit_raw is None else int(hit_raw)
    errors: list[str] = []
    classification = "adapter_failure"
    retained_outcome = False
    policy_noncompletion = False

    if stop == "post_adaptation_probe_complete":
        classification = "valid_completed_probe"
        retained_outcome = True
        if not completed or hit is None:
            errors.append("completed_probe_without_target_hit")
        if adaptation_censored or overshoot_censored:
            errors.append("completed_probe_marked_censored")
        if hit is not None and not 0 <= hit <= adaptation_budget:
            errors.append("target_hit_outside_adaptation_deadline")
        if probe_count != probe_budget or not retained:
            errors.append("post_hit_probe_not_exactly_retained")
        if hit is not None and phase != hit + probe_budget:
            errors.append("phase_clock_not_hit_plus_probe")
    elif stop in {"controller_quiescent", "phase_event_budget"}:
        classification = (
            "valid_retained_quiescent_censor"
            if stop == "controller_quiescent"
            else "valid_retained_phase_budget_censor"
        )
        retained_outcome = True
        policy_noncompletion = True
        if completed or hit is not None:
            errors.append("nonadaptation_terminal_contains_target_hit")
        if not adaptation_censored or not overshoot_censored:
            errors.append("nonadaptation_terminal_missing_explicit_censors")
        if probe_count != 0 or not retained:
            errors.append("nonadaptation_probe_censor_not_retained")
        if stop == "phase_event_budget" and phase != adaptation_budget:
            errors.append("phase_event_budget_not_at_adaptation_deadline")
        if stop == "controller_quiescent" and not 0 <= phase <= adaptation_budget:
            errors.append("quiescent_terminal_outside_adaptation_deadline")
    elif stop == "invariant_error":
        errors.append("native_invariant_error")
    else:
        errors.append("unknown_target_change_terminal")

    if phase < 0 or phase > adaptation_budget + probe_budget:
        errors.append("phase_clock_outside_native_maximum")
    valid = not errors
    return {
        "version": TARGET_CHANGE_SEMANTICS_VERSION,
        "validNativeContract": valid,
        "classification": classification if valid else "adapter_failure",
        "retainedOutcome": retained_outcome and valid,
        "policyNoncompletion": policy_noncompletion and valid,
        "adapterFailure": not valid,
        "nativeStopReasonPreserved": stop,
        "adaptationCensoredPreserved": adaptation_censored,
        "overshootCensoredPreserved": overshoot_censored,
        "errors": errors,
    }


def _require_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _require_sha256(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def build_target_change_audit_projection(
    result: Mapping[str, Any],
    *,
    adaptation_budget: int,
    probe_budget: int,
    source_result_sha256: str | None,
) -> dict[str, Any]:
    """Project the complete target result into the persisted native-event plane.

    This operation does not relabel a terminal or repair an invalid result. It
    verifies that the complete source result carries exact S08H metadata, then
    commits a lossless audit projection. Invalid native/adapter branches remain
    invalid and are serialized with their authoritative failure audit.
    """

    missing = sorted(_TARGET_CHANGE_SOURCE_FIELDS - set(result))
    if missing:
        raise ValueError(f"target-change source metadata missing: {missing}")
    if result["targetChangeSemanticsVersion"] != TARGET_CHANGE_SEMANTICS_VERSION:
        raise ValueError("target-change source semantics version mismatch")
    adaptation_budget = _require_int(adaptation_budget, "adaptationBudget")
    probe_budget = _require_int(probe_budget, "probeBudget")
    if adaptation_budget < 1 or probe_budget < 1:
        raise ValueError("target-change budgets must be positive")
    completed = _require_bool(result["targetCompleted"], "targetCompleted")
    adaptation_censored = _require_bool(
        result["adaptationCensored"], "adaptationCensored"
    )
    overshoot_censored = _require_bool(result["overshootCensored"], "overshootCensored")
    retained = _require_bool(result["postHitProbeRetained"], "postHitProbeRetained")
    applicable = _require_bool(
        result["postHitProbeApplicable"], "postHitProbeApplicable"
    )
    phase = _require_int(result["phaseActivationCount"], "phaseActivationCount")
    probe_count = _require_int(
        result["postHitProbeOpportunities"], "postHitProbeOpportunities"
    )
    hit_raw = result["adaptationTime"]
    if hit_raw is not None:
        hit_raw = _require_int(hit_raw, "adaptationTime")
    if applicable != completed:
        raise ValueError("postHitProbeApplicable must equal targetCompleted")
    if completed == adaptation_censored or completed == overshoot_censored:
        raise ValueError("completion and target censors are inconsistent")
    source_result_sha256 = _require_sha256(
        source_result_sha256, "sourceResultSha256", nullable=True
    )
    authoritative = validate_target_change_result_semantics(
        result,
        adaptation_budget=adaptation_budget,
        probe_budget=probe_budget,
    )
    source_audit = result["targetChangeSemanticAudit"]
    if not isinstance(source_audit, Mapping) or dict(source_audit) != authoritative:
        raise ValueError("source target-change semantic audit is inconsistent")
    payload = {
        "schemaVersion": TARGET_CHANGE_AUDIT_PROJECTION_SCHEMA_VERSION,
        "metadataOrigin": "complete_e05_target_result_before_endpoint_projection",
        "targetChangeSemanticsVersion": TARGET_CHANGE_SEMANTICS_VERSION,
        "adaptationBudget": adaptation_budget,
        "probeBudget": probe_budget,
        "stopReason": str(result["stopReason"]),
        "targetCompleted": completed,
        "phaseActivationCount": phase,
        "adaptationTime": hit_raw,
        "adaptationCensored": adaptation_censored,
        "overshootCensored": overshoot_censored,
        "postHitProbeOpportunities": probe_count,
        "postHitProbeRetained": retained,
        "postHitProbeApplicable": applicable,
        "targetChangeSemanticAudit": authoritative,
        "sourceResultSha256": source_result_sha256,
    }
    return {
        **payload,
        "projectionSha256": canonical_sha256(_TARGET_CHANGE_PROJECTION_DOMAIN, payload),
    }


def validate_persisted_target_change_audit_projection(
    *,
    native_event: Mapping[str, Any],
    native_outcome: Mapping[str, Any],
    stop_reason: str,
    validation: Mapping[str, Any],
    adaptation_budget: int,
    probe_budget: int,
) -> dict[str, Any]:
    """Validate a persisted row against the unchanged S08H state machine."""

    integrity_errors: list[str] = []
    projection_raw = native_event.get("targetChangeAuditProjection")
    if not isinstance(projection_raw, Mapping):
        return {
            "schemaVersion": "e07.s08j.persisted-target-audit-validation.v1",
            "projectionAuthentic": False,
            "validNativeContract": False,
            "validPersistedContract": False,
            "classification": "adapter_failure",
            "adapterFailure": True,
            "integrityErrors": ["missing_target_change_audit_projection"],
            "semanticErrors": [],
        }
    projection = dict(projection_raw)
    missing = sorted(_TARGET_CHANGE_PROJECTION_FIELDS - set(projection))
    extra = sorted(set(projection) - _TARGET_CHANGE_PROJECTION_FIELDS)
    if missing:
        integrity_errors.append(f"missing_projection_fields:{','.join(missing)}")
    if extra:
        integrity_errors.append(f"extra_projection_fields:{','.join(extra)}")
    authoritative: dict[str, Any] | None = None
    if not missing:
        try:
            if projection["schemaVersion"] != (
                TARGET_CHANGE_AUDIT_PROJECTION_SCHEMA_VERSION
            ):
                integrity_errors.append("projection_schema_version_mismatch")
            if projection["metadataOrigin"] != (
                "complete_e05_target_result_before_endpoint_projection"
            ):
                integrity_errors.append("projection_metadata_origin_mismatch")
            if projection["adaptationBudget"] != adaptation_budget:
                integrity_errors.append("projection_adaptation_budget_mismatch")
            if projection["probeBudget"] != probe_budget:
                integrity_errors.append("projection_probe_budget_mismatch")
            stored_sha = _require_sha256(
                projection["projectionSha256"], "projectionSha256"
            )
            payload = {
                key: value
                for key, value in projection.items()
                if key != "projectionSha256"
            }
            expected_sha = canonical_sha256(_TARGET_CHANGE_PROJECTION_DOMAIN, payload)
            if stored_sha != expected_sha:
                integrity_errors.append("projection_commitment_mismatch")
            source_sha = _require_sha256(
                projection["sourceResultSha256"],
                "sourceResultSha256",
                nullable=True,
            )
            event_source_sha = native_event.get("resultSha256")
            if source_sha is not None and event_source_sha != source_sha:
                integrity_errors.append("source_result_commitment_mismatch")
            if source_sha is None and event_source_sha is not None:
                integrity_errors.append("missing_source_result_commitment")
            source = {key: projection[key] for key in _TARGET_CHANGE_SOURCE_FIELDS}
            rebuilt = build_target_change_audit_projection(
                source,
                adaptation_budget=adaptation_budget,
                probe_budget=probe_budget,
                source_result_sha256=source_sha,
            )
            if rebuilt != projection:
                integrity_errors.append("projection_round_trip_mismatch")
            authoritative = dict(rebuilt["targetChangeSemanticAudit"])
        except (TypeError, ValueError) as exc:
            integrity_errors.append(f"invalid_projection:{exc}")
    if str(stop_reason) != projection.get("stopReason"):
        integrity_errors.append("persisted_stop_reason_mismatch")
    endpoint_fields = (
        "targetCompleted",
        "phaseActivationCount",
        "adaptationTime",
        "adaptationCensored",
        "overshootCensored",
    )
    for field in endpoint_fields:
        if field not in native_outcome:
            integrity_errors.append(f"missing_persisted_endpoint:{field}")
        elif native_outcome[field] != projection.get(field) or type(
            native_outcome[field]
        ) is not type(projection.get(field)):
            integrity_errors.append(f"persisted_endpoint_mismatch:{field}")
    expected_validation = {
        "targetChangeSemanticContract": bool(
            authoritative and authoritative["validNativeContract"]
        ),
        "postHitProbeRetained": projection.get("postHitProbeRetained"),
    }
    for field, expected in expected_validation.items():
        if field not in validation:
            integrity_errors.append(f"missing_validation_flag:{field}")
        elif type(validation[field]) is not bool or validation[field] != expected:
            integrity_errors.append(f"persisted_validation_mismatch:{field}")
    semantic_errors = list(authoritative["errors"]) if authoritative is not None else []
    projection_authentic = not integrity_errors
    valid_native = bool(
        authoritative is not None and authoritative["validNativeContract"]
    )
    return {
        "schemaVersion": "e07.s08j.persisted-target-audit-validation.v1",
        "projectionAuthentic": projection_authentic,
        "validNativeContract": valid_native,
        "validPersistedContract": projection_authentic and valid_native,
        "classification": (
            authoritative["classification"]
            if authoritative is not None
            else "adapter_failure"
        ),
        "adapterFailure": not (projection_authentic and valid_native),
        "integrityErrors": integrity_errors,
        "semanticErrors": semantic_errors,
    }
