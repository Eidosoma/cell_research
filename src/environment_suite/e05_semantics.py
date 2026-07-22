"""Outcome-independent E05 target-change semantic contracts."""

from __future__ import annotations

from typing import Any, Mapping


TARGET_CHANGE_SEMANTICS_VERSION = "e07.s08h.e05-target-deadline-probe.v1"


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
