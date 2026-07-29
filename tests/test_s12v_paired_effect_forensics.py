from __future__ import annotations

import math

import pytest

from scripts.execute_replacement_spatial_transfer_s12u import _paired_record
from src.phenotype_discovery.publication import (
    SerializationContractError,
    canonical_json_bytes,
)
from src.spatial_transfer.paired_effect_forensics import (
    arithmetic_bound_evidence,
    disposition_counts,
    enumerate_contract_states,
)


def _record(left, right, *, binary: bool = False):
    return _paired_record(
        family="synthetic_S12V",
        contrast_id="outcome_independent_fixture",
        task_id="synthetic_task",
        panel_id="synthetic_panel",
        condition_id="synthetic_condition",
        lineage_id="synthetic_lineage",
        endpoint="synthetic_endpoint",
        left_label="left",
        right_label="right",
        left=left,
        right=right,
        benefit_direction=1,
        binary=binary,
    )


def test_cartesian_contract_state_registry_is_total_and_deterministic() -> None:
    first = enumerate_contract_states()
    second = enumerate_contract_states()
    assert first == second
    assert len(first) == 3_808
    assert all(row["expectedDisposition"] for row in first)
    counts = disposition_counts(first)
    assert sum(counts.values()) == 3_808
    assert counts["finite_evidentiary_record"] > 0
    assert (
        counts["fixed_slot_non_evidentiary_endpoint_unavailable_or_inapplicable"]
        > 0
    )
    assert counts["fail_closed_nonfinite_endpoint_before_arithmetic"] > 0


@pytest.mark.parametrize(
    ("left", "right", "binary"),
    [
        ([0.0], [0.0], True),
        ([1.0], [0.0], True),
        ([0.0, 0.5, 1.0], [1.0, 0.5, 0.0], False),
        ([32.0, 0.0], [0.0, 32.0], False),
        ([float((1 << 63) - 1)], [0.0], False),
    ],
)
def test_frozen_estimator_and_serializer_accept_finite_bounded_records(
    left, right, binary
) -> None:
    record = _record(left, right, binary=binary)
    assert math.isfinite(record["rawEffectLeftMinusRight"])
    canonical_json_bytes({"records": [record]})


def test_frozen_helper_represents_direct_empty_input_as_non_evidentiary() -> None:
    record = _record([], [])
    assert record["rawEffectLeftMinusRight"] is None
    assert record["evidentiary"] is False
    assert record["nonEvidentiaryReason"] == "NO_PAIRED_ROWS"
    canonical_json_bytes({"records": [record]})


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_frozen_estimator_admits_bad_endpoint_then_serializer_rejects(bad) -> None:
    record = _record([bad], [0.0])
    assert not math.isfinite(record["rawEffectLeftMinusRight"])
    with pytest.raises(
        SerializationContractError,
        match=r"rawEffectLeftMinusRight: non-finite floating value prohibited",
    ):
        canonical_json_bytes({"records": [record]})


def test_fault_repair_against_no_fault_undefined_reproduces_failure_class() -> None:
    # Frozen execution.py emits None for repairByTransition32 in no-fault
    # rows.  Frozen build_paired_estimands selects repairByTransition32 from
    # the fault-side condition and casts the no-fault side to NaN.
    record = _record([1.0, 0.0], [float("nan"), float("nan")], binary=True)
    assert record["evidentiary"] is True
    assert math.isnan(record["rawEffectLeftMinusRight"])
    with pytest.raises(SerializationContractError):
        canonical_json_bytes({"records": [record]})


def test_partial_pair_support_fails_before_record_projection() -> None:
    with pytest.raises(RuntimeError, match="paired endpoint arrays differ"):
        _record([1.0], [])


def test_contract_valid_bounded_arithmetic_is_finite_at_full_budget() -> None:
    evidence = arithmetic_bound_evidence()
    assert evidence["logicalBudget"] == 195_072
    assert evidence["allFinite"] is True
