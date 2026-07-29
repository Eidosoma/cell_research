from __future__ import annotations

import json

from src.spatial_transfer.first_completion_forensics import (
    REPRESENTATION_CASES,
    build_synthetic_matrix,
    matrix_summary,
    native_clock_domains,
)


def _record(rows, status: str, representation_id: str):
    return next(
        row
        for row in rows
        if row["status"] == status
        and row["representationId"] == representation_id
    )


def test_complete_matrix_is_deterministic_and_order_independent() -> None:
    forward = build_synthetic_matrix()
    reverse = build_synthetic_matrix(reverse=True)
    assert forward == reverse
    assert len(forward) == 5 * len(REPRESENTATION_CASES)
    assert len(REPRESENTATION_CASES) == 21
    assert json.dumps(forward, sort_keys=True, allow_nan=False)


def test_pre_transition_sentinel_reproduces_failure_class_synthetically() -> None:
    rows = build_synthetic_matrix()
    record = _record(
        rows,
        "calibrated_endpoint_retained",
        "pre_transition_initial_sentinel",
    )
    assert record["dslAdapter"] is True
    assert record["nativeBaseline"] is False
    assert record["disposition"] == "fail_closed"
    assert (
        record["message"]
        == "firstCompletionTransition must be an integer in [0,32]"
    )


def test_elapsed_initial_and_censor_boundary_are_finite() -> None:
    rows = build_synthetic_matrix()
    initial = _record(rows, "calibrated_endpoint_retained", "zero_integer")
    censor = _record(
        rows,
        "right_censored_at_transition_32",
        "censor_boundary_32",
    )
    missing_censor = _record(
        rows,
        "right_censored_at_transition_32",
        "explicit_null",
    )
    assert initial["projectedValue"] == 0.0
    assert censor["projectedValue"] == 32.0
    assert missing_censor["projectedValue"] == 32.0


def test_diagnostic_and_failure_statuses_are_non_evidentiary_for_time() -> None:
    rows = build_synthetic_matrix()
    for status in (
        "endpoint_unavailable_diagnostic_retained",
        "execution_or_invariant_failure_retained",
    ):
        assert all(
            row["disposition"]
            == "non_evidentiary_status_outside_endpoint_domain"
            for row in rows
            if row["status"] == status
        )


def test_native_and_tracker_clock_domains_are_not_the_same_plane() -> None:
    domains = native_clock_domains()
    assert (
        domains["authoritativeMovementStateElapsedTransitionCount"]["initial"] == 0
    )
    assert domains["dslTrackerArgument"]["initial"] == -1
    assert domains["nativeBaselineTrackerArgument"]["initial"] == "not_observed"
    assert domains["persistedProjection"] == (
        "raw tracker value copied without normalization"
    )


def test_matrix_summary_counts_every_state_once() -> None:
    summary = matrix_summary(build_synthetic_matrix())
    assert summary["rowCount"] == 105
    assert summary["statusCount"] == 5
    assert summary["representationCount"] == 21
    assert sum(summary["dispositionCounts"].values()) == 105
    assert summary["exactRecordedErrorClassCount"] > 0
    assert summary["sourceReachableRecordedErrorClassCount"] > 0
