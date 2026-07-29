from __future__ import annotations

import copy
import math

import pandas as pd
import pytest

from src.phenotype_discovery.publication import canonical_json_bytes
from src.spatial_transfer.estimator_feasibility import (
    ENDPOINT_SPECS,
    FIXED_FAMILIES,
    EstimatorFeasibilityError,
    FixedSlot,
    apply_fixed_holm,
    build_fixed_slot_record,
    classify_pre_arithmetic_state,
    exact_one_to_one_pair,
    project_endpoint_value,
    validate_estimator_payload,
)
from src.spatial_transfer.paired_effect_forensics import (
    enumerate_contract_states,
)


def _row(
    pair_id: str,
    status: str = "calibrated_endpoint_retained",
    **fields,
) -> dict:
    status_flags = {
        "calibrated_endpoint_retained": (True, False, False, False),
        "repair_observed": (True, False, False, False),
        "right_censored_at_transition_32": (True, False, True, False),
        "endpoint_unavailable_diagnostic_retained": (
            False,
            False,
            False,
            True,
        ),
        "execution_or_invariant_failure_retained": (
            False,
            True,
            False,
            False,
        ),
    }
    available, failed, censored, diagnostic = status_flags[status]
    return {
        "pairId": pair_id,
        "status": status,
        "endpointAvailable": available,
        "failed": failed,
        "censored": censored,
        "diagnostic": diagnostic,
        "terminalConjunctiveCompletion": True,
        "minimumMismatchFraction": 0.0,
        "repairByTransition32": status == "repair_observed",
        "departureAfterInitiallyComplete": False,
        "firstCompletionTransition": 7,
        "separateCostComponent": 0,
        **fields,
    }


def _slot(
    endpoint_kind: str,
    *,
    family: str = "fault_repair_endpoint_family",
) -> FixedSlot:
    return FixedSlot(
        family=family,
        contrast_id="synthetic_fixed_contrast",
        task_id="synthetic_spatial_task",
        panel_id="synthetic_calibrated_panel",
        condition_id="synthetic_condition",
        lineage_id="synthetic_lineage",
        endpoint=ENDPOINT_SPECS[endpoint_kind],
        left_label="left",
        right_label="right",
    )


def test_repaired_disposition_engine_matches_every_frozen_s12v_state() -> None:
    rows = enumerate_contract_states()
    assert len(rows) == 3_808
    for row in rows:
        actual = classify_pre_arithmetic_state(
            endpoint_kind=row["endpointKind"],
            native_status=row["nativeStatusContract"],
            denominator_state=row["denominatorState"],
            pair_support_state=row["pairedConditionSupport"],
        )
        assert actual == row["expectedDisposition"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_exact_pair_rejects_duplicate_identity(side: str) -> None:
    left = pd.DataFrame([_row("a")])
    right = pd.DataFrame([_row("a")])
    target = left if side == "left" else right
    target.loc[1] = target.loc[0]
    with pytest.raises(
        EstimatorFeasibilityError,
        match="duplicate or ambiguous pair identity",
    ):
        exact_one_to_one_pair(left, right, ["pairId"])


@pytest.mark.parametrize(
    ("left_ids", "right_ids"),
    [
        (["a"], []),
        ([], ["a"]),
        (["a", "b"], ["a"]),
        (["a"], ["a", "b"]),
    ],
)
def test_exact_pair_rejects_incomplete_support(left_ids, right_ids) -> None:
    columns = list(_row("template"))
    left = pd.DataFrame([_row(value) for value in left_ids], columns=columns)
    right = pd.DataFrame([_row(value) for value in right_ids], columns=columns)
    with pytest.raises(EstimatorFeasibilityError, match="incomplete pair support"):
        exact_one_to_one_pair(left, right, ["pairId"])


def test_no_fault_repair_side_is_explicit_fixed_non_evidentiary_slot() -> None:
    left = pd.DataFrame(
        [_row("a", "repair_observed", repairByTransition32=True)]
    )
    right = pd.DataFrame(
        [
            _row(
                "a",
                "calibrated_endpoint_retained",
                repairByTransition32=None,
            )
        ]
    )
    record = build_fixed_slot_record(
        _slot("repair_by_transition_32_binary"),
        left=left,
        right=right,
        pair_keys=["pairId"],
    )
    assert record["evidentiary"] is False
    assert record["rawPValue"] == 1.0
    assert record["rawEffectLeftMinusRight"] is None
    assert (
        record["nonEvidentiaryReason"]
        == "RIGHT_CONDITION_OUTSIDE_ENDPOINT_DOMAIN"
    )
    canonical_json_bytes(record)


def test_empty_population_retains_fixed_slot() -> None:
    empty = pd.DataFrame(columns=list(_row("a")))
    record = build_fixed_slot_record(
        _slot("minimum_mismatch_fraction_continuous"),
        left=empty,
        right=empty,
        pair_keys=["pairId"],
    )
    assert record["nonEvidentiaryReason"] == "NO_PAIRED_ROWS"
    assert record["pairCount"] == 0
    assert record["rawPValue"] == 1.0


@pytest.mark.parametrize("missing", [None, pd.NA, pd.NaT])
def test_missing_denominator_retains_fixed_non_evidentiary_slot(missing) -> None:
    record = build_fixed_slot_record(
        _slot("minimum_mismatch_fraction_continuous"),
        left=pd.DataFrame([_row("a", minimumMismatchFraction=0.0)]),
        right=pd.DataFrame([_row("a", minimumMismatchFraction=1.0)]),
        pair_keys=["pairId"],
        denominator=missing,
    )
    assert record["nonEvidentiaryReason"] == "DENOMINATOR_UNAVAILABLE"
    assert record["rawPValue"] == 1.0
    assert record["pairCount"] == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_denominator_fails_closed(bad) -> None:
    with pytest.raises(EstimatorFeasibilityError, match="denominator"):
        build_fixed_slot_record(
            _slot("minimum_mismatch_fraction_continuous"),
            left=pd.DataFrame([_row("a", minimumMismatchFraction=0.0)]),
            right=pd.DataFrame([_row("a", minimumMismatchFraction=1.0)]),
            pair_keys=["pairId"],
            denominator=bad,
        )


@pytest.mark.parametrize(
    ("status", "first_transition", "expected"),
    [
        ("calibrated_endpoint_retained", 0, 0.0),
        ("calibrated_endpoint_retained", 31, 31.0),
        ("calibrated_endpoint_retained", 32, 32.0),
        ("calibrated_endpoint_retained", None, 32.0),
        ("repair_observed", 8, 8.0),
        ("right_censored_at_transition_32", None, 32.0),
        ("right_censored_at_transition_32", 32, 32.0),
    ],
)
def test_restricted_transition_time_exact_censor_at_32(
    status, first_transition, expected
) -> None:
    row = _row(
        "a",
        status,
        firstCompletionTransition=first_transition,
        repairByTransition32=(
            status == "repair_observed"
            if status != "right_censored_at_transition_32"
            else False
        ),
    )
    actual = project_endpoint_value(
        row,
        ENDPOINT_SPECS["restricted_native_transition_time_at_32"],
    )
    assert actual == expected


@pytest.mark.parametrize("bad", [-1, 33, 1.5, float("nan"), float("inf")])
def test_restricted_transition_time_rejects_values_outside_contract(bad) -> None:
    row = _row("a", firstCompletionTransition=bad)
    with pytest.raises(EstimatorFeasibilityError):
        project_endpoint_value(
            row,
            ENDPOINT_SPECS["restricted_native_transition_time_at_32"],
        )


def test_all_reserved_failure_and_cost_endpoints_ignore_task_availability() -> None:
    failure = _row(
        "failure",
        "execution_or_invariant_failure_retained",
        terminalConjunctiveCompletion=None,
        minimumMismatchFraction=None,
        firstCompletionTransition=None,
        separateCostComponent=11,
    )
    diagnostic = _row(
        "diagnostic",
        "endpoint_unavailable_diagnostic_retained",
        terminalConjunctiveCompletion=None,
        minimumMismatchFraction=None,
        firstCompletionTransition=None,
        separateCostComponent=13,
    )
    assert (
        project_endpoint_value(
            failure,
            ENDPOINT_SPECS["reserved_failure_binary"],
        )
        == 1.0
    )
    assert (
        project_endpoint_value(
            failure,
            ENDPOINT_SPECS["separate_cost_component_nonnegative_integer"],
        )
        == 11.0
    )
    assert (
        project_endpoint_value(
            diagnostic,
            ENDPOINT_SPECS["reserved_failure_binary"],
        )
        == 0.0
    )
    assert (
        project_endpoint_value(
            diagnostic,
            ENDPOINT_SPECS["separate_cost_component_nonnegative_integer"],
        )
        == 13.0
    )


@pytest.mark.parametrize("bad", [None, pd.NA, float("nan"), float("inf")])
def test_nonfinite_or_ambiguous_evidentiary_endpoint_fails_before_arithmetic(
    bad,
) -> None:
    left = pd.DataFrame([_row("a", minimumMismatchFraction=bad)])
    right = pd.DataFrame([_row("a", minimumMismatchFraction=0.0)])
    with pytest.raises(EstimatorFeasibilityError):
        build_fixed_slot_record(
            _slot("minimum_mismatch_fraction_continuous"),
            left=left,
            right=right,
            pair_keys=["pairId"],
        )


def test_fixed_holm_conserves_slots_and_non_evidentiary_p_equals_one() -> None:
    finite_slot = _slot(
        "minimum_mismatch_fraction_continuous",
        family="parent_compressed_calibrated_endpoint_family",
    )
    empty_slot = FixedSlot(
        **{
            **finite_slot.__dict__,
            "condition_id": "synthetic_empty_condition",
        }
    )
    finite = build_fixed_slot_record(
        finite_slot,
        left=pd.DataFrame([_row("a", minimumMismatchFraction=0.0)]),
        right=pd.DataFrame([_row("a", minimumMismatchFraction=1.0)]),
        pair_keys=["pairId"],
    )
    empty_frame = pd.DataFrame(columns=list(_row("a")))
    empty = build_fixed_slot_record(
        empty_slot,
        left=empty_frame,
        right=empty_frame,
        pair_keys=["pairId"],
    )
    records = apply_fixed_holm(
        [empty, finite],
        expected_test_ids=[finite_slot.test_id, empty_slot.test_id],
    )
    assert len(records) == 2
    assert {row["testId"] for row in records} == {
        finite_slot.test_id,
        empty_slot.test_id,
    }
    empty_result = next(row for row in records if not row["evidentiary"])
    assert empty_result["rawPValue"] == 1.0
    assert math.isfinite(empty_result["holmAdjustedPValue"])

    payload = {
        "schemaVersion": "e07.s12w.paired-estimands.v1",
        "taskLocalOnly": True,
        "universalScore": None,
        "fixedFamilies": list(FIXED_FAMILIES),
        "records": records,
    }
    validate_estimator_payload(
        payload,
        expected_test_ids=[finite_slot.test_id, empty_slot.test_id],
    )
    canonical_json_bytes(payload)


def test_fixed_holm_fails_closed_on_slot_loss_or_copy() -> None:
    slot = _slot(
        "minimum_mismatch_fraction_continuous",
        family="parent_compressed_calibrated_endpoint_family",
    )
    record = build_fixed_slot_record(
        slot,
        left=pd.DataFrame([_row("a", minimumMismatchFraction=0.0)]),
        right=pd.DataFrame([_row("a", minimumMismatchFraction=1.0)]),
        pair_keys=["pairId"],
    )
    with pytest.raises(EstimatorFeasibilityError, match="slot set differs"):
        apply_fixed_holm([record], expected_test_ids=[])
    copied = copy.deepcopy(record)
    with pytest.raises(EstimatorFeasibilityError, match="duplicate fixed"):
        apply_fixed_holm(
            [record, copied],
            expected_test_ids=[slot.test_id],
        )
