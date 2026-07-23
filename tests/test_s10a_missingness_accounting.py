from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.phenotype_discovery.accounting import (
    FailAtomicPhaseError,
    build_physical_replay_plan,
    execute_fail_atomic_replays,
)
from src.phenotype_discovery.native_features import (
    extract_native_event_features,
    validate_feature_availability_record,
)
from src.phenotype_discovery.search import (
    feature_support_by_task_status,
    support_decision,
)
from tests.test_s10p_native_event_discovery import _spatial_payload


def _reservation(position: int, *, fail_replays: tuple[int, ...] = ()) -> dict:
    return {
        "logicalOrdinal": position,
        "logicalReservationId": f"{position:064x}",
        "split": "train",
        "taskId": "qualification_fixture",
        "statusStratum": "fixture|failed=false|censored=false",
        "failReplays": list(fail_replays),
    }


def _synthetic_physical_evaluator(physical: dict) -> dict:
    reservation = physical["reservation"]
    replay = int(physical["replayOrdinal"])
    if replay in reservation["failReplays"]:
        raise RuntimeError(
            f"injected failure at logical={physical['logicalPosition']} replay={replay}"
        )
    logical = int(physical["logicalPosition"])
    commitment = f"{logical + 1:064x}"
    return {
        "resultBody": {
            "logicalOrdinal": logical,
            "logicalReservationId": reservation["logicalReservationId"],
            "fixture": True,
        },
        "deterministicResultSha256": commitment,
        "nativeReplaySha256": commitment,
        "availabilitySummary": {
            "state": "complete_with_explicit_unavailability",
            "registeredFeatureCount": 18,
            "observedFeatureCount": 17,
            "unavailableFeatureCount": 1,
            "reasonCodeCounts": {"zero_native_denominator": 1},
            "imputedFeatureCount": 0,
            "silentDropCount": 0,
        },
    }


def test_zero_native_denominator_is_explicit_and_not_imputed() -> None:
    payload = _spatial_payload()
    movement = payload["costs"]["e06MovementLedger"]
    for key in (
        "submittedProposals",
        "adjacentSwaps",
        "vacancyMoves",
        "shortExchanges",
        "rotations",
        "validProposals",
        "conflictCandidates",
        "reservedSiteClaims",
        "totalGraphDisplacement",
    ):
        movement[key] = 0
    record = extract_native_event_features("e07_s02_spatial2d_local", payload)
    summary = validate_feature_availability_record(
        record, expected_task_id="e07_s02_spatial2d_local"
    )
    zero_denominator = [
        value
        for value in record["availability"].values()
        if value.get("reasonCode") == "zero_native_denominator"
    ]
    assert zero_denominator
    assert all(value["state"] == "unavailable" for value in zero_denominator)
    assert all(
        value["reason"] == "structurally_not_applicable" for value in zero_denominator
    )
    assert summary["imputedFeatureCount"] == 0
    assert summary["silentDropCount"] == 0
    assert (
        summary["observedFeatureCount"] + summary["unavailableFeatureCount"]
        == summary["registeredFeatureCount"]
    )


def test_frozen_support_boundary_is_inclusive_and_zero_safe() -> None:
    below = support_decision(89, 100)
    at = support_decision(90, 100)
    above = support_decision(91, 100)
    zero = support_decision(0, 0)
    assert below["support"] == 0.89 and not below["eligible"]
    assert at["support"] == 0.90 and at["eligible"]
    assert above["support"] == 0.91 and above["eligible"]
    assert zero["support"] is None
    assert zero["state"] == "undefined_zero_denominator"
    assert not zero["eligible"]


def test_support_is_task_status_stratum_local_with_mixed_availability() -> None:
    complete = extract_native_event_features(
        "e07_s02_spatial2d_local", _spatial_payload()
    )
    feature_ids = sorted(complete["availability"])
    target = feature_ids[0]
    rows = []
    for label, observed_count in (("below", 89), ("at", 90), ("above", 91)):
        for index in range(100):
            record = deepcopy(complete)
            if index >= observed_count:
                record["analysisFeatures"].pop(target)
                record["availability"][target] = {
                    "state": "unavailable",
                    "reason": "structurally_not_applicable",
                    "reasonCode": "qualification_fixture_unavailable",
                }
            rows.append(
                {
                    "taskId": "e07_s02_spatial2d_local",
                    "statusStratum": label,
                    "analysisFeatures": record["analysisFeatures"],
                    "availability": record["availability"],
                }
            )
    support = feature_support_by_task_status(rows)
    by_status = {row["statusStratum"]: row for row in support["strata"]}
    assert by_status["below"]["featureDecisions"][target]["support"] == 0.89
    assert not by_status["below"]["featureDecisions"][target]["eligible"]
    assert by_status["at"]["featureDecisions"][target]["eligible"]
    assert by_status["above"]["featureDecisions"][target]["eligible"]
    assert all(row["minimumFeatureCountPass"] for row in by_status.values())
    assert all(row["imputedFeatureCount"] == 0 for row in by_status.values())
    assert all(not row["completeCaseRowFilterApplied"] for row in by_status.values())


def test_physical_plan_restores_two_distinct_replays_per_logical() -> None:
    reservations = [_reservation(index) for index in range(7)]
    plan = build_physical_replay_plan(reservations)
    assert len(plan) == 14
    assert len({row["physicalExecutionId"] for row in plan}) == 14
    for index in range(7):
        pair = plan[2 * index : 2 * index + 2]
        assert [row["replayOrdinal"] for row in pair] == [0, 1]
        assert {row["logicalPosition"] for row in pair} == {index}
        assert len({row["reservationCommitmentSha256"] for row in pair}) == 1


def test_fail_atomic_success_is_order_independent(tmp_path: Path) -> None:
    reservations = [_reservation(index) for index in range(9)]
    natural_path = tmp_path / "natural.json"
    reverse_path = tmp_path / "reverse.json"
    natural, natural_accounting = execute_fail_atomic_replays(
        reservations,
        _synthetic_physical_evaluator,
        disposition_path=natural_path,
        phase="qualification",
        workers=4,
        executor_kind="thread",
    )
    reverse, reverse_accounting = execute_fail_atomic_replays(
        reservations,
        _synthetic_physical_evaluator,
        disposition_path=reverse_path,
        phase="qualification",
        workers=4,
        executor_kind="thread",
        submission_order=list(reversed(range(18))),
    )
    assert natural == reverse
    left = json.loads(natural_path.read_text())
    right = json.loads(reverse_path.read_text())
    assert left["dispositionSha256"] == right["dispositionSha256"]
    assert left["accountingConserved"] and right["accountingConserved"]
    assert left["resultRowsEligibleForCallerPublication"] == 9
    assert left["resultRowsPublishedByExecutor"] == 0
    assert natural_accounting["publishedRows"] == 9
    assert reverse_accounting["publishedRows"] == 9


def test_fail_atomic_process_executor_path(tmp_path: Path) -> None:
    reservations = [_reservation(index) for index in range(4)]
    results, accounting = execute_fail_atomic_replays(
        reservations,
        _synthetic_physical_evaluator,
        disposition_path=tmp_path / "process.json",
        phase="qualification_process",
        workers=2,
        executor_kind="process",
    )
    ledger = json.loads((tmp_path / "process.json").read_text())
    assert len(results) == 4
    assert accounting["publishedRows"] == 4
    assert ledger["terminalLogicalDispositionCount"] == 4
    assert ledger["terminalPhysicalDispositionCount"] == 8
    assert ledger["accountingConserved"]


@pytest.mark.parametrize("logical_position", [0, 4, 8])
def test_injected_failures_keep_every_disposition_and_publish_zero(
    tmp_path: Path, logical_position: int
) -> None:
    reservations = [
        _reservation(index, fail_replays=(0,) if index == logical_position else ())
        for index in range(9)
    ]
    path = tmp_path / f"failure-{logical_position}.json"
    with pytest.raises(FailAtomicPhaseError) as captured:
        execute_fail_atomic_replays(
            reservations,
            _synthetic_physical_evaluator,
            disposition_path=path,
            phase="qualification",
            workers=4,
            executor_kind="thread",
            submission_order=list(reversed(range(18))),
        )
    ledger = json.loads(path.read_text())
    assert captured.value.accounting["publishedRows"] == 0
    assert ledger["publicationState"] == "withheld"
    assert ledger["resultRowsEligibleForCallerPublication"] == 0
    assert ledger["terminalLogicalDispositionCount"] == 9
    assert ledger["terminalPhysicalDispositionCount"] == 18
    assert ledger["accountingConserved"]
    failed_logical = [
        row for row in ledger["logicalDispositions"] if row["state"] != "success"
    ]
    assert [row["logicalPosition"] for row in failed_logical] == [logical_position]
    failed_physical = [
        row for row in ledger["physicalDispositions"] if row["state"] != "success"
    ]
    assert len(failed_physical) == 1
    assert failed_physical[0]["logicalPosition"] == logical_position
    assert failed_physical[0]["availabilitySummary"]["reasonCodeCounts"]
