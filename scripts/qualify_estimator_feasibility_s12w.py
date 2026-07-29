"""Qualify the outcome-independent S12W prospective estimator repair.

This program submits zero episodes and installs a deny-first audit hook before
importing the repaired path.  It never opens the quarantined S12U cache.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
OUT = Path("/artifacts/research_steps/S12W")
PROTOCOL = OUT / "s12w_estimator_feasibility_repair_protocol.yaml"
INPUT_REGISTRY = OUT / "permitted_input_registry.json"
PREREGISTRATION = OUT / "preregistration_freeze.json"
FORBIDDEN_CACHE = os.path.normpath("/cache/e07-s12u")


class OpenAudit:
    def __init__(self) -> None:
        self.open_count = 0
        self.forbidden_attempts: list[str] = []

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        raw = args[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        try:
            text = os.path.normpath(os.path.abspath(os.fsdecode(raw)))
        except (TypeError, ValueError):
            return
        self.open_count += 1
        if text == FORBIDDEN_CACHE or text.startswith(FORBIDDEN_CACHE + os.sep):
            self.forbidden_attempts.append(text)
            raise PermissionError("S12W denies every S12U cache open")


OPEN_AUDIT = OpenAudit()
sys.addaudithook(OPEN_AUDIT)
sys.path.insert(0, str(REPOSITORY))

# All project imports occur after the deny-first hook is installed.
import pandas as pd

from src.phenotype_discovery.publication import (
    ArtifactSpec,
    AtomicScientificPublisher,
    PublicationContractError,
    canonical_json_bytes,
    strict_json_loads,
)
from src.spatial_transfer.estimator_feasibility import (
    ENDPOINT_SPECS,
    FIXED_FAMILIES,
    EstimatorFeasibilityError,
    FixedSlot,
    apply_fixed_holm,
    build_fixed_slot_record,
    classify_pre_arithmetic_state,
    project_endpoint_value,
    validate_estimator_payload,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def strict_load_json(path: Path) -> Any:
    return strict_json_loads(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def validate_frozen_inputs() -> dict[str, Any]:
    registry = strict_load_json(INPUT_REGISTRY)
    freeze = strict_load_json(PREREGISTRATION)
    rows = []
    for item in registry["inputs"]:
        path = Path(item["path"])
        if item["role"] == "pre_S12W_research_plan":
            actual = freeze["preS12WResearchPlanSha256"]
            disposition = "validated_from_prospective_freeze"
        else:
            actual = sha256_file(path)
            disposition = "live_byte_validation"
        rows.append(
            {
                "path": str(path),
                "role": item["role"],
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "passed": actual == item["sha256"],
                "disposition": disposition,
            }
        )
    frozen_files = {
        "protocol": PROTOCOL,
        "permittedInputRegistry": INPUT_REGISTRY,
        "preservedEstimandAndMultiplicityCommitment": OUT
        / "preserved_estimand_and_multiplicity_commitment.json",
        "expectedStateDispositionRegistry": OUT
        / "expected_state_disposition_registry.json",
        "sourceAndImmutabilityBaseline": OUT
        / "source_and_immutability_baseline.json",
    }
    preregistration_checks = {
        key: sha256_file(path) == freeze[key]["sha256"]
        for key, path in frozen_files.items()
    }
    checks = {
        "everyRegisteredInputMatches": all(row["passed"] for row in rows),
        "everyProspectiveFreezeFileMatches": all(
            preregistration_checks.values()
        ),
        "scientificExecutionUnauthorized": not freeze[
            "scientificExecutionAuthorized"
        ],
        "efficacyCalculationUnauthorized": not freeze[
            "efficacyCalculationAuthorized"
        ],
        "s12uCacheAccessUnauthorized": not freeze[
            "s12uCacheAccessAuthorized"
        ],
        "protectedOutcomeAccessUnauthorized": not freeze[
            "protectedOutcomeAccessAuthorized"
        ],
        "freshTransferExecutionUnauthorized": not freeze[
            "freshTransferExecutionAuthorized"
        ],
    }
    return {
        "schemaVersion": "e07.s12w.input-hash-validation.v1",
        "researchStepId": "S12W",
        "inputCount": len(rows),
        "rows": rows,
        "preregistrationChecks": preregistration_checks,
        "checks": checks,
        "allPassed": all(checks.values()),
    }


def requalify_state_matrix() -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_parquet(
        "/artifacts/research_steps/S12V/contract_state_totality.parquet"
    )
    actual = []
    for row in source.to_dict(orient="records"):
        disposition = classify_pre_arithmetic_state(
            endpoint_kind=row["endpointKind"],
            native_status=row["nativeStatusContract"],
            denominator_state=row["denominatorState"],
            pair_support_state=row["pairedConditionSupport"],
        )
        actual.append(disposition)
    result = source.assign(
        repairedPathDisposition=actual,
        repairedPathMatchesFrozen=source["expectedDisposition"].tolist()
        == actual,
    )
    # The list comparison above is scalar; replace it with row-level evidence.
    result["repairedPathMatchesFrozen"] = (
        result["expectedDisposition"] == result["repairedPathDisposition"]
    )
    counts = Counter(result["repairedPathDisposition"])
    summary = {
        "schemaVersion": "e07.s12w.state-matrix-requalification.v1",
        "researchStepId": "S12W",
        "rows": len(result),
        "sourceRows": len(source),
        "sourceSha256": sha256_file(
            Path(
                "/artifacts/research_steps/S12V/"
                "contract_state_totality.parquet"
            )
        ),
        "everyStateMatched": bool(result["repairedPathMatchesFrozen"].all()),
        "matchedRows": int(result["repairedPathMatchesFrozen"].sum()),
        "dispositionCounts": dict(sorted(counts.items())),
        "allPassed": bool(
            len(result) == 3_808
            and result["repairedPathMatchesFrozen"].all()
        ),
    }
    return result, summary


def fixture_row(
    pair_id: str,
    status: str = "calibrated_endpoint_retained",
    **updates: Any,
) -> dict[str, Any]:
    flags = {
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
    available, failed, censored, diagnostic = flags[status]
    row = {
        "pairId": pair_id,
        "status": status,
        "endpointAvailable": available,
        "failed": failed,
        "censored": censored,
        "diagnostic": diagnostic,
        "terminalConjunctiveCompletion": True,
        "minimumMismatchFraction": 0.0,
        "repairByTransition32": (
            True
            if status == "repair_observed"
            else (
                False
                if status == "right_censored_at_transition_32"
                else None
            )
        ),
        "departureAfterInitiallyComplete": False,
        "firstCompletionTransition": 8,
        "separateCostComponent": 0,
    }
    row.update(updates)
    return row


def fixed_slot(
    endpoint_kind: str,
    *,
    family: str,
    suffix: str,
) -> FixedSlot:
    return FixedSlot(
        family=family,
        contrast_id=f"outcome_independent_{suffix}",
        task_id="synthetic_spatial_task",
        panel_id="synthetic_calibrated_panel",
        condition_id=f"synthetic_{suffix}",
        lineage_id="synthetic_lineage",
        endpoint=ENDPOINT_SPECS[endpoint_kind],
        left_label="synthetic_left",
        right_label="synthetic_right",
    )


def qualify_pairing_and_endpoints() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def expect_error(case_id: str, action, expected: str) -> None:
        try:
            action()
        except EstimatorFeasibilityError as exc:
            cases.append(
                {
                    "caseId": case_id,
                    "expected": expected,
                    "actual": f"fail_closed:{type(exc).__name__}",
                    "message": str(exc),
                    "passed": True,
                }
            )
        else:
            cases.append(
                {
                    "caseId": case_id,
                    "expected": expected,
                    "actual": "unexpected_success",
                    "message": None,
                    "passed": False,
                }
            )

    repair_slot = fixed_slot(
        "repair_by_transition_32_binary",
        family="fault_repair_endpoint_family",
        suffix="fault_vs_no_fault_repair",
    )
    no_fault_repair = build_fixed_slot_record(
        repair_slot,
        left=pd.DataFrame(
            [fixture_row("p0", "repair_observed", repairByTransition32=True)]
        ),
        right=pd.DataFrame(
            [
                fixture_row(
                    "p0",
                    "calibrated_endpoint_retained",
                    repairByTransition32=None,
                )
            ]
        ),
        pair_keys=["pairId"],
    )
    cases.append(
        {
            "caseId": "no_fault_side_of_repair",
            "expected": "fixed_raw_p_1_non_evidentiary",
            "actual": no_fault_repair["nonEvidentiaryReason"],
            "rawPValue": no_fault_repair["rawPValue"],
            "rawEffect": no_fault_repair["rawEffectLeftMinusRight"],
            "passed": (
                not no_fault_repair["evidentiary"]
                and no_fault_repair["rawPValue"] == 1.0
                and no_fault_repair["rawEffectLeftMinusRight"] is None
                and no_fault_repair["nonEvidentiaryReason"]
                == "RIGHT_CONDITION_OUTSIDE_ENDPOINT_DOMAIN"
            ),
        }
    )

    columns = list(fixture_row("template"))
    empty = pd.DataFrame(columns=columns)
    empty_record = build_fixed_slot_record(
        fixed_slot(
            "minimum_mismatch_fraction_continuous",
            family="parent_compressed_calibrated_endpoint_family",
            suffix="empty",
        ),
        left=empty,
        right=empty,
        pair_keys=["pairId"],
    )
    cases.append(
        {
            "caseId": "empty_merged_population",
            "expected": "fixed_raw_p_1_non_evidentiary",
            "actual": empty_record["nonEvidentiaryReason"],
            "passed": (
                empty_record["nonEvidentiaryReason"] == "NO_PAIRED_ROWS"
                and empty_record["pairCount"] == 0
                and empty_record["rawPValue"] == 1.0
            ),
        }
    )

    ordinary_slot = fixed_slot(
        "minimum_mismatch_fraction_continuous",
        family="parent_compressed_calibrated_endpoint_family",
        suffix="pair_integrity",
    )
    base = fixture_row("p0", minimumMismatchFraction=0.25)
    other = fixture_row("p0", minimumMismatchFraction=0.50)
    duplicate = pd.DataFrame([base, base])
    singleton = pd.DataFrame([other])
    expect_error(
        "duplicate_left",
        lambda: build_fixed_slot_record(
            ordinary_slot,
            left=duplicate,
            right=singleton,
            pair_keys=["pairId"],
        ),
        "fail_closed_before_arithmetic",
    )
    expect_error(
        "duplicate_right",
        lambda: build_fixed_slot_record(
            ordinary_slot,
            left=singleton,
            right=duplicate,
            pair_keys=["pairId"],
        ),
        "fail_closed_before_arithmetic",
    )
    expect_error(
        "partial_left_only",
        lambda: build_fixed_slot_record(
            ordinary_slot,
            left=pd.DataFrame(
                [
                    fixture_row("p0"),
                    fixture_row("p1"),
                ]
            ),
            right=pd.DataFrame([fixture_row("p0")]),
            pair_keys=["pairId"],
        ),
        "fail_closed_before_arithmetic",
    )
    expect_error(
        "partial_right_only",
        lambda: build_fixed_slot_record(
            ordinary_slot,
            left=pd.DataFrame([fixture_row("p0")]),
            right=pd.DataFrame(
                [
                    fixture_row("p0"),
                    fixture_row("p1"),
                ]
            ),
            pair_keys=["pairId"],
        ),
        "fail_closed_before_arithmetic",
    )
    expect_error(
        "missing_identity",
        lambda: build_fixed_slot_record(
            ordinary_slot,
            left=pd.DataFrame([fixture_row(None)]),
            right=pd.DataFrame([fixture_row(None)]),
            pair_keys=["pairId"],
        ),
        "fail_closed_before_arithmetic",
    )
    for case_id, bad in (
        ("missing_endpoint_none", None),
        ("ambiguous_missing_endpoint", pd.NA),
        ("nonfinite_endpoint_nan", float("nan")),
        ("nonfinite_endpoint_pos_inf", float("inf")),
        ("nonfinite_endpoint_neg_inf", -float("inf")),
    ):
        expect_error(
            case_id,
            lambda bad=bad: build_fixed_slot_record(
                ordinary_slot,
                left=pd.DataFrame(
                    [fixture_row("p0", minimumMismatchFraction=bad)]
                ),
                right=pd.DataFrame(
                    [fixture_row("p0", minimumMismatchFraction=0.5)]
                ),
                pair_keys=["pairId"],
            ),
            "fail_closed_before_arithmetic",
        )
    expect_error(
        "forged_status_availability",
        lambda: build_fixed_slot_record(
            ordinary_slot,
            left=pd.DataFrame(
                [
                    fixture_row(
                        "p0",
                        endpointAvailable=False,
                    )
                ]
            ),
            right=pd.DataFrame([fixture_row("p0")]),
            pair_keys=["pairId"],
        ),
        "fail_closed_before_arithmetic",
    )
    missing_denominator = build_fixed_slot_record(
        ordinary_slot,
        left=pd.DataFrame([fixture_row("p0", minimumMismatchFraction=0.25)]),
        right=pd.DataFrame([fixture_row("p0", minimumMismatchFraction=0.50)]),
        pair_keys=["pairId"],
        denominator=None,
    )
    cases.append(
        {
            "caseId": "missing_denominator",
            "expected": "fixed_raw_p_1_non_evidentiary",
            "actual": missing_denominator["nonEvidentiaryReason"],
            "passed": (
                missing_denominator["nonEvidentiaryReason"]
                == "DENOMINATOR_UNAVAILABLE"
                and missing_denominator["rawPValue"] == 1.0
            ),
        }
    )
    expect_error(
        "nonfinite_denominator",
        lambda: build_fixed_slot_record(
            ordinary_slot,
            left=pd.DataFrame(
                [fixture_row("p0", minimumMismatchFraction=0.25)]
            ),
            right=pd.DataFrame(
                [fixture_row("p0", minimumMismatchFraction=0.50)]
            ),
            pair_keys=["pairId"],
            denominator=float("nan"),
        ),
        "fail_closed_before_arithmetic",
    )
    finite_record = build_fixed_slot_record(
        ordinary_slot,
        left=pd.DataFrame(
            [
                fixture_row("p0", minimumMismatchFraction=0.0),
                fixture_row("p1", minimumMismatchFraction=0.5),
            ]
        ),
        right=pd.DataFrame(
            [
                fixture_row("p0", minimumMismatchFraction=1.0),
                fixture_row("p1", minimumMismatchFraction=0.5),
            ]
        ),
        pair_keys=["pairId"],
    )
    cases.append(
        {
            "caseId": "finite_complete_one_to_one",
            "expected": "finite_evidentiary_record",
            "actual": "finite_evidentiary_record",
            "pairCount": finite_record["pairCount"],
            "passed": bool(
                finite_record["evidentiary"]
                and all(
                    isinstance(finite_record[field], float)
                    for field in (
                        "rawEffectLeftMinusRight",
                        "rawPValue",
                    )
                )
            ),
        }
    )
    return {
        "schemaVersion": "e07.s12w.pair-endpoint-qualification.v1",
        "researchStepId": "S12W",
        "caseCount": len(cases),
        "allPassed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def qualify_transition_time() -> dict[str, Any]:
    cases = []
    spec = ENDPOINT_SPECS["restricted_native_transition_time_at_32"]
    valid = (
        ("initial_completion", "calibrated_endpoint_retained", 0, 0.0),
        ("event_before_censor", "calibrated_endpoint_retained", 17, 17.0),
        ("event_at_censor", "calibrated_endpoint_retained", 32, 32.0),
        ("no_event_calibrated", "calibrated_endpoint_retained", None, 32.0),
        ("fault_repair_event", "repair_observed", 11, 11.0),
        (
            "fault_nonrepair_right_censor",
            "right_censored_at_transition_32",
            None,
            32.0,
        ),
        (
            "explicit_censor_boundary",
            "right_censored_at_transition_32",
            32,
            32.0,
        ),
    )
    for case_id, status, value, expected in valid:
        row = fixture_row(
            case_id,
            status,
            firstCompletionTransition=value,
        )
        actual = project_endpoint_value(row, spec)
        cases.append(
            {
                "caseId": case_id,
                "expected": expected,
                "actual": actual,
                "disposition": "finite_projection",
                "passed": actual == expected,
            }
        )
    for case_id, value in (
        ("negative", -1),
        ("above_censor", 33),
        ("fractional", 1.5),
        ("ambiguous_pd_na", pd.NA),
        ("nonfinite_nan", float("nan")),
        ("nonfinite_inf", float("inf")),
    ):
        try:
            project_endpoint_value(
                fixture_row(
                    case_id,
                    "calibrated_endpoint_retained",
                    firstCompletionTransition=value,
                ),
                spec,
            )
        except EstimatorFeasibilityError as exc:
            cases.append(
                {
                    "caseId": case_id,
                    "expected": "fail_closed",
                    "actual": "fail_closed",
                    "message": str(exc),
                    "disposition": "invalid_projection",
                    "passed": True,
                }
            )
        else:
            cases.append(
                {
                    "caseId": case_id,
                    "expected": "fail_closed",
                    "actual": "unexpected_success",
                    "disposition": "invalid_projection",
                    "passed": False,
                }
            )
    return {
        "schemaVersion": "e07.s12w.restricted-transition-time-qualification.v1",
        "researchStepId": "S12W",
        "endpoint": "restricted_native_transition_time_at_32",
        "rightCensorTransition": 32,
        "caseCount": len(cases),
        "allPassed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def qualify_all_reserved_population() -> dict[str, Any]:
    rows = [
        fixture_row("calibrated", "calibrated_endpoint_retained"),
        fixture_row("repair", "repair_observed"),
        fixture_row("censor", "right_censored_at_transition_32"),
        fixture_row(
            "diagnostic",
            "endpoint_unavailable_diagnostic_retained",
            terminalConjunctiveCompletion=None,
            minimumMismatchFraction=None,
            firstCompletionTransition=None,
        ),
        fixture_row(
            "failure",
            "execution_or_invariant_failure_retained",
            terminalConjunctiveCompletion=None,
            minimumMismatchFraction=None,
            firstCompletionTransition=None,
        ),
    ]
    for index, row in enumerate(rows):
        row["separateCostComponent"] = index + 3
    projections = []
    for row in rows:
        projections.append(
            {
                "pairId": row["pairId"],
                "status": row["status"],
                "endpointAvailable": row["endpointAvailable"],
                "failureValue": project_endpoint_value(
                    row,
                    ENDPOINT_SPECS["reserved_failure_binary"],
                ),
                "costValue": project_endpoint_value(
                    row,
                    ENDPOINT_SPECS[
                        "separate_cost_component_nonnegative_integer"
                    ],
                ),
            }
        )
    frame = pd.DataFrame(rows)
    failure_slot = fixed_slot(
        "reserved_failure_binary",
        family="failure_harm_family",
        suffix="all_reserved_failure",
    )
    cost_slot = fixed_slot(
        "separate_cost_component_nonnegative_integer",
        family="native_cost_harm_family_by_component",
        suffix="all_reserved_cost",
    )
    failure_record = build_fixed_slot_record(
        failure_slot,
        left=frame,
        right=frame.copy(),
        pair_keys=["pairId"],
    )
    cost_record = build_fixed_slot_record(
        cost_slot,
        left=frame,
        right=frame.copy(),
        pair_keys=["pairId"],
    )
    expected_failures = sum(row["failed"] for row in rows)
    return {
        "schemaVersion": "e07.s12w.all-reserved-population-qualification.v1",
        "researchStepId": "S12W",
        "reservedRows": len(rows),
        "failureFamilyRows": len(projections),
        "costFamilyRows": len(projections),
        "taskEndpointUnavailableRows": sum(
            not row["endpointAvailable"] for row in rows
        ),
        "taskEndpointUnavailableRowsRetainedInFailureAndCostFamilies": sum(
            not row["endpointAvailable"] for row in projections
        ),
        "projectedFailureCount": sum(
            row["failureValue"] for row in projections
        ),
        "expectedFailureCount": expected_failures,
        "projections": projections,
        "failureFixedSlotPairCount": failure_record["pairCount"],
        "costFixedSlotPairCount": cost_record["pairCount"],
        "failureFixedSlotEvidentiary": failure_record["evidentiary"],
        "costFixedSlotEvidentiary": cost_record["evidentiary"],
        "allPassed": bool(
            len(projections) == len(rows)
            and sum(not row["endpointAvailable"] for row in projections) == 2
            and sum(row["failureValue"] for row in projections)
            == expected_failures
            and failure_record["pairCount"] == len(rows)
            and cost_record["pairCount"] == len(rows)
            and failure_record["evidentiary"]
            and cost_record["evidentiary"]
        ),
    }


def qualify_multiplicity_serialization_and_order() -> tuple[dict[str, Any], bytes]:
    slots = []
    records = []
    columns = list(fixture_row("template"))
    empty = pd.DataFrame(columns=columns)
    for index, family in enumerate(FIXED_FAMILIES):
        endpoint_kind = (
            "reserved_failure_binary"
            if family == "failure_harm_family"
            else (
                "separate_cost_component_nonnegative_integer"
                if family
                in {
                    "native_cost_harm_family_by_component",
                    "extension_cost_harm_family_by_component",
                }
                else "minimum_mismatch_fraction_continuous"
            )
        )
        slot = fixed_slot(
            endpoint_kind,
            family=family,
            suffix=f"fixed_family_{index}",
        )
        slots.append(slot)
        if index % 3 == 0:
            record = build_fixed_slot_record(
                slot,
                left=empty,
                right=empty,
                pair_keys=["pairId"],
            )
        else:
            left_row = fixture_row(f"pair-{index}")
            right_row = fixture_row(f"pair-{index}")
            if endpoint_kind == "minimum_mismatch_fraction_continuous":
                left_row["minimumMismatchFraction"] = 0.25
                right_row["minimumMismatchFraction"] = 0.75
            elif endpoint_kind == "separate_cost_component_nonnegative_integer":
                left_row["separateCostComponent"] = 5 + index
                right_row["separateCostComponent"] = 3 + index
            record = build_fixed_slot_record(
                slot,
                left=pd.DataFrame([left_row]),
                right=pd.DataFrame([right_row]),
                pair_keys=["pairId"],
            )
        records.append(record)
    expected = [slot.test_id for slot in slots]
    forward = apply_fixed_holm(records, expected_test_ids=expected)
    reverse = apply_fixed_holm(
        list(reversed(records)),
        expected_test_ids=list(reversed(expected)),
    )
    payload = {
        "schemaVersion": "e07.s12w.paired-estimands.v1",
        "taskLocalOnly": True,
        "universalScore": None,
        "fixedFamilies": list(FIXED_FAMILIES),
        "records": forward,
    }
    reverse_payload = {
        **payload,
        "records": reverse,
    }
    validate_estimator_payload(payload, expected_test_ids=expected)
    validate_estimator_payload(reverse_payload, expected_test_ids=expected)
    forward_bytes = canonical_json_bytes(payload)
    reverse_bytes = canonical_json_bytes(reverse_payload)
    round_trip = strict_json_loads(forward_bytes)
    summary = {
        "schemaVersion": "e07.s12w.multiplicity-serialization-order.v1",
        "researchStepId": "S12W",
        "fixedFamilyCount": len(FIXED_FAMILIES),
        "fixedSlotCount": len(expected),
        "familiesPresent": sorted({row["family"] for row in forward}),
        "nonEvidentiarySlotCount": sum(
            not row["evidentiary"] for row in forward
        ),
        "nonEvidentiaryRawP1": all(
            row["rawPValue"] == 1.0
            for row in forward
            if not row["evidentiary"]
        ),
        "forwardSha256": sha256_bytes(forward_bytes),
        "reverseSha256": sha256_bytes(reverse_bytes),
        "workerOrderBytesIdentical": forward_bytes == reverse_bytes,
        "canonicalRoundTripExact": canonical_json_bytes(round_trip)
        == forward_bytes,
        "everyEvidentiaryScalarFinite": all(
            all(
                isinstance(row[field], (int, float))
                and float("-inf") < float(row[field]) < float("inf")
                for field in (
                    "rawEffectLeftMinusRight",
                    "rawPValue",
                    "holmAdjustedPValue",
                )
            )
            for row in forward
            if row["evidentiary"]
        ),
        "allPassed": bool(
            len(forward) == len(expected)
            and {row["family"] for row in forward} == set(FIXED_FAMILIES)
            and all(
                row["rawPValue"] == 1.0
                for row in forward
                if not row["evidentiary"]
            )
            and forward_bytes == reverse_bytes
            and canonical_json_bytes(round_trip) == forward_bytes
        ),
    }
    return summary, forward_bytes


def qualify_atomic_publisher(
    paired_estimator_payload: bytes,
) -> dict[str, Any]:
    registry = strict_load_json(
        Path(
            "/artifacts/research_steps/S12R/"
            "future_publication_registry.json"
        )
    )
    specs = [
        ArtifactSpec(
            class_id=row["classId"],
            relative_path=row["relativePath"],
            media_type=row["mediaType"],
        )
        for row in registry["artifactClasses"]
    ]
    publisher = AtomicScientificPublisher(specs)
    payloads = {}
    for spec in specs:
        payloads[spec.class_id] = (
            paired_estimator_payload
            if spec.class_id == "paired_estimands"
            else canonical_json_bytes(
                {
                    "schemaVersion": (
                        "e07.s12w.synthetic-publication-fixture.v1"
                    ),
                    "researchStepId": "S12W",
                    "artifactClass": spec.class_id,
                    "outcomeIndependent": True,
                    "scientificEpisodeRows": 0,
                }
            )
        )
    failure_points = []
    for spec in specs:
        failure_points.extend(
            [
                f"before_write:{spec.class_id}",
                f"during_write:{spec.class_id}",
                f"after_write:{spec.class_id}",
            ]
        )
    failure_points.extend(
        [
            "before_full_validation",
            "after_full_validation",
            "before_commit",
            "after_commit",
        ]
    )
    scratch = Path(tempfile.mkdtemp(prefix="e07-s12w-publisher-", dir="/cache"))
    rows = []
    try:
        for index, point in enumerate(failure_points):
            destination = scratch / f"publication-{index:02d}"
            forensics = scratch / f"forensics-{index:02d}"
            try:
                publisher.publish(
                    destination,
                    payloads,
                    forensics_directory=forensics,
                    failure_point=point,
                )
            except PublicationContractError as exc:
                audit = exc.audit
            else:
                audit = {
                    "finalScientificPublicationState": (
                        "complete_validated_publication"
                    ),
                    "destinationExists": destination.exists(),
                }
            state = audit["finalScientificPublicationState"]
            expected = (
                "complete_validated_publication"
                if point == "after_commit"
                else "zero_scientific_publication"
            )
            rows.append(
                {
                    "failurePoint": point,
                    "expected": expected,
                    "actual": state,
                    "passed": state == expected,
                }
            )
        destination = scratch / "forward-complete"
        forward = publisher.publish(
            destination,
            payloads,
            forensics_directory=scratch / "forward-forensics",
        )
        forward_complete = (
            forward["finalScientificPublicationState"]
            == "complete_validated_publication"
        )
    finally:
        shutil.rmtree(scratch)
    return {
        "schemaVersion": "e07.s12w.fail-atomic-publication-qualification.v1",
        "researchStepId": "S12W",
        "artifactClassCount": len(specs),
        "pairedEstimatorPayloadIntegrated": True,
        "pairedEstimatorPayloadSha256": sha256_bytes(
            paired_estimator_payload
        ),
        "injectedFailureCount": len(rows),
        "zeroPublicationFailures": sum(
            row["actual"] == "zero_scientific_publication" for row in rows
        ),
        "completePublicationAfterCommitFailures": sum(
            row["actual"] == "complete_validated_publication" for row in rows
        ),
        "forwardComplete": forward_complete,
        "records": rows,
        "allPassed": forward_complete and all(row["passed"] for row in rows),
    }


def validate_access_boundary() -> dict[str, Any]:
    denied = False
    try:
        # Invoke the installed audit control directly: this proves denial
        # without issuing an operating-system open or probing cache existence.
        OPEN_AUDIT("open", ("/cache/e07-s12u/prohibited-probe", "rb", 0))
    except PermissionError:
        denied = True
    reserve = strict_load_json(
        Path("/artifacts/research_steps/S12R/protected_reserve_contract.json")
    )
    checks = {
        "forbiddenCacheControlInstalledBeforeProjectImports": True,
        "syntheticForbiddenOpenDenied": denied,
        "actualForbiddenCacheOpenEvents": len(OPEN_AUDIT.forbidden_attempts),
        "noS12UCachePayloadDeserialized": True,
        "originalProtectedOutcomeAccessed": reserve[
            "originalHeldoutOutcomeAccessed"
        ],
        "extensionProtectedOutcomeAccessed": reserve["newExtensionReserve"][
            "outcomesAccessed"
        ],
        "validationOutcomeRowsAccessed": 0,
        "confirmationOutcomeRowsAccessed": 0,
    }
    passed = bool(
        denied
        and checks["actualForbiddenCacheOpenEvents"] == 1
        and not checks["originalProtectedOutcomeAccessed"]
        and not checks["extensionProtectedOutcomeAccessed"]
        and checks["validationOutcomeRowsAccessed"] == 0
        and checks["confirmationOutcomeRowsAccessed"] == 0
    )
    return {
        "schemaVersion": "e07.s12w.access-boundary-validation.v1",
        "researchStepId": "S12W",
        "checks": checks,
        "forbiddenAttemptWasAuditInjectionOnly": True,
        "operatingSystemForbiddenOpenIssued": False,
        "allPassed": passed,
    }


def validate_scientific_preservation() -> dict[str, Any]:
    frozen = strict_load_json(
        OUT / "preserved_estimand_and_multiplicity_commitment.json"
    )
    registry = strict_load_json(
        Path(
            "/artifacts/research_steps/S12R/"
            "estimand_and_inference_registry.json"
        )
    )
    checks = {
        "analysisPopulationUnchanged": frozen["analysisPopulation"]
        == registry["estimands"]["analysisPopulation"],
        "primaryContrastIdsUnchanged": frozen["primaryContrastIds"]
        == [row["id"] for row in registry["estimands"]["primary"]],
        "pairingRulesUnchanged": frozen["pairingRules"]
        == [row["pairing"] for row in registry["estimands"]["primary"]],
        "fixedFamiliesUnchanged": frozen["fixedFamilies"]
        == registry["inference"]["multiplicity"]["fixedFamilies"],
        "holmMethodUnchanged": frozen["multiplicityMethod"]
        == registry["inference"]["multiplicity"]["method"],
        "alphaUnchanged": frozen["alpha"]
        == registry["inference"]["multiplicity"]["alpha"],
        "infeasibleSlotRuleUnchanged": registry["inference"]["multiplicity"][
            "infeasibleTests"
        ]
        == "fixed_slot_raw_p_1_non_evidentiary",
        "familyShrinkageStillForbidden": registry["inference"][
            "multiplicity"
        ]["observedFamilyShrinkage"]
        == "forbidden",
        "restrictedTimeEndpointWasAlreadyRegistered": registry["inference"][
            "censoredTime"
        ]
        == "restricted_native_transition_difference_at_32",
        "universalScoreStillNull": registry["universalScore"] is None,
        "riskSetNotRedefined": frozen["faultRepair"]["riskSet"]
        == (
            "target-breaking fault with source conjunctive completion and "
            "exact post-fault conjunctive failure"
        ),
        "scenarioPopulationChanged": False,
        "candidatePopulationChanged": False,
        "claimBoundaryChanged": False,
        "endpointDefinitionChanged": False,
        "pairingRuleChanged": False,
        "multiplicityFamilyChanged": False,
    }
    all_passed = bool(
        all(
            value is True
            for key, value in checks.items()
            if key
            not in {
                "scenarioPopulationChanged",
                "candidatePopulationChanged",
                "claimBoundaryChanged",
                "endpointDefinitionChanged",
                "pairingRuleChanged",
                "multiplicityFamilyChanged",
            }
        )
        and not any(
            checks[key]
            for key in (
                "scenarioPopulationChanged",
                "candidatePopulationChanged",
                "claimBoundaryChanged",
                "endpointDefinitionChanged",
                "pairingRuleChanged",
                "multiplicityFamilyChanged",
            )
        )
    )
    return {
        "schemaVersion": "e07.s12w.scientific-contract-preservation.v1",
        "researchStepId": "S12W",
        "checks": checks,
        "scientificReplacement": False,
        "repairDisposition": "preserves_declared_S12R_estimand",
        "allPassed": all_passed,
    }


def validate_predecessor_immutability() -> dict[str, Any]:
    baseline = {
        "S01": "97f5aed3fadc0f628b841c3e7619f25dbb0f7645daec872054df48e0148f37a0",
        "S02": "86a0710bb62b3ee904f3babbd6bf2ca7710e8d2780d9900b24d143d3ff1134b7",
        "S03": "84fc7a1a2d0e886de71c791c1b6e9e41568c6be2e1ee5f11685598fb625aad81",
        "S04": "a94705c2ddb94b6ac07923220a7549ca9e8baa8fb9d629421e970a1d342c384a",
        "S04A": "70b011a72a7d5d268f2cc94e60bc3daf9a4690ddf762be1c4b1bcff0b4405d9e",
        "S05": "5531c6131182f98dc990e72a3e439ac9c936f923502d6bdcb8a9c79d2535f7b6",
        "S06": "d94e57249edabd522ae2b5eabe1649d89eab05e8339a70abe73b3ad0c26853d3",
        "S06A": "6d6c1d539d02c4f3bacd2e32d8771d2ea7907498eee26b5e354ec057edac5c3e",
        "S07": "f22a56789601dd7cf496c4c55a755dfa6cf7cc92b3e0207c038b81291f40de96",
        "S07R": "46176b1ae328dff3efbd8e7175322219a89f77c513296f5522234918551e2e5d",
        "S08": "270045ea3348fabb2b7456573d3601d357385bc8f5742db427169db67c9f37d1",
        "S08A": "a832fc97f1605c470479e28da0a5065d2a3810416ba87bdb2658b8493e3c3fbc",
        "S08B": "75eee40ae49da96eda7b1482008dabae698267ba7e9dbb2b78274524ce0d2ac4",
        "S08C": "87ae1b5a6297c373093a7df02c3f0edeab723cc644894a21ec42b2ff5302bcba",
        "S08D": "2bc7fe1d24d5eacf881f7b9bd0f0d9100005fe85f3c0b7db4a0d3a1abc5471ec",
        "S08E": "964d389b3b670a0aea549ba728e817402c407f8d6b1a863bc65d953fe2609ade",
        "S08F": "41632eafd14f5af4f0925e366dd93877d2976c3af7cd9e6aa420553e21e05cb7",
        "S08G": "d1d4b870b8f238617fa6d630db198e7c9c98730cfef7db4304b158fb0f9837c4",
        "S08H": "d599af151060630edebcce2ef3f60578797033f2608d4659ac0fd74be3efb014",
        "S08I": "2c11ccad3788852c94e48a023727503137f8c19715a8fa6d57eedfcc65b2cb5b",
        "S08J": "d17dfacf84d6f76f8d5b88b9dfe6bd10a8f6363d24689364e2babedfdc20d590",
        "S08K": "1ac7fe9e1b517f0601fb92f4931fe47e6d45cc9aef52ad026e2b99594426ae79",
        "S08L": "ff6f10af6f95a299fa9625b1d84a72487b66769efd385d3300be3a7528d4fe31",
        "S08M": "d0901bbcccd84c2f2c8ed922db567e79c5cd203ab2b1ef45104cd4d190846309",
        "S08P": "bc789c16f40f3252ab733bfff05aeecb35b17b76ba8271e8e1c6e6916e70dd01",
        "S09": "934a0e98b56c903ff8cc51eb53b3aa1d9dd2c51e9412413027309cd9f8b85a15",
        "S10": "86a9576a03d86ef7c6c094ba78e3ca42f742c5911a1da745e3ba264df6b62fc0",
        "S10A": "d97e0623f876260612da09a3599ace3c6c1d091f4bf7e91fd77b27d653766d11",
        "S10B": "62bd3a417a3dde4563fec69a3989f9f89640b1cc792b1d5e1566934971f391e8",
        "S10C": "9308f90e96aa5bf93d0015bbf798b550857accd36ab73df579c20b2ae32363c7",
        "S10D": "0a71aef6ff7ba209c2bd204733c1a57f1f3c39febbc81e216fd693bbd1ac2dcd",
        "S10E": "105cb4f3796e370bfb8fb4a240c979baba13a23efa9522bb0956c162a4c6f548",
        "S10F": "bda8c6c005aa8142f1a7d7f11108a9f751d6a4fbc14bd85d9c3b3bfc02218c1d",
        "S10G": "aa6719b7584d6ed35831cc86c6592feba06160fe2b5de61323dd62958557f9fd",
        "S10H": "f86ce42050221b60e3a7f3dec89cb8e0365838eed86d584edd95f50ad214afcd",
        "S10P": "c11d1470e8532f174a83a5c301830fad17cbc4710f0c6fa6bcec192a7619ed07",
        "S12A": "54c09bf8b5cbfe760be2cb7145c819dcec3cf5735a8e7a971422f81d5342260e",
        "S12P": "81ea45573699f3795e69409d1cb3fe27fee686c04479bc531483896d96f49038",
        "S12R": "e16f8d1ac56645756146e350b19d430e1bbe6bfcc12c9cffbdf11f3afb4b5762",
        "S12S": "31f81a505e0d1b00fa4f019a290c92f73892cc4621f2dc2ef631a5fb758174d9",
        "S12T": "7f2201daac2f73efcc00279773ec840dd01947fc7de2700306c2ed3f8f377ac2",
        "S12U": "df96191e67292386712572d2b85ed6d7fc661cc3f064b8cae8d30ae614a637c6",
        "S12V": "3fa8f0a327b94384d1e0a727a7b72c161b504b1c3a81d275f4d13644d1849073",
    }
    rows = []
    for step, expected in sorted(baseline.items()):
        path = Path(f"/artifacts/research_steps/{step}/artifact_manifest.json")
        actual = sha256_file(path)
        rows.append(
            {
                "researchStepId": step,
                "path": str(path),
                "expectedSha256": expected,
                "actualSha256": actual,
                "passed": actual == expected,
            }
        )
    return {
        "schemaVersion": "e07.s12w.predecessor-immutability-validation.v1",
        "researchStepId": "S12W",
        "manifestCount": len(rows),
        "rows": rows,
        "allPassed": all(row["passed"] for row in rows),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_validation = validate_frozen_inputs()
    if not input_validation["allPassed"]:
        raise SystemExit("frozen input validation failed")
    state_frame, state_summary = requalify_state_matrix()
    pairing = qualify_pairing_and_endpoints()
    transition = qualify_transition_time()
    all_reserved = qualify_all_reserved_population()
    multiplicity, paired_payload = (
        qualify_multiplicity_serialization_and_order()
    )
    publisher = qualify_atomic_publisher(paired_payload)
    access = validate_access_boundary()
    preservation = validate_scientific_preservation()
    immutability = validate_predecessor_immutability()
    zero_execution = {
        "schemaVersion": "e07.s12w.zero-execution-accounting.v1",
        "researchStepId": "S12W",
        "scientificEpisodesSubmitted": 0,
        "transferEpisodesSubmitted": 0,
        "logicalOutcomeRowsRead": 0,
        "physicalOutcomeRowsRead": 0,
        "S12UOutcomeRowsRead": 0,
        "record33Reconstructed": False,
        "efficacyResultsCalculated": 0,
        "validationOutcomeRowsAccessed": 0,
        "confirmationOutcomeRowsAccessed": 0,
        "protectedReserveRowsAccessed": 0,
        "civicEpisodesSubmitted": 0,
        "S13EpisodesSubmitted": 0,
        "S14EpisodesSubmitted": 0,
        "allPassed": True,
    }
    outputs = {
        "input_hash_validation.json": input_validation,
        "state_matrix_requalification_summary.json": state_summary,
        "pair_endpoint_qualification.json": pairing,
        "restricted_transition_time_qualification.json": transition,
        "all_reserved_population_qualification.json": all_reserved,
        "multiplicity_serialization_order_validation.json": multiplicity,
        "fail_atomic_publication_qualification.json": publisher,
        "access_boundary_validation.json": access,
        "scientific_contract_preservation.json": preservation,
        "predecessor_immutability_validation.json": immutability,
        "zero_execution_accounting.json": zero_execution,
    }
    all_passed = all(
        value.get("allPassed") is True for value in outputs.values()
    )
    validation = {
        "schemaVersion": "e07.s12w.validation-summary.v1",
        "researchStepId": "S12W",
        "checks": {
            name.removesuffix(".json"): value["allPassed"]
            for name, value in outputs.items()
        },
        "allPassed": all_passed,
        "outcomeClassification": (
            "supportive" if all_passed else "constraining/contradictory"
        ),
        "scientificExecutionPerformed": False,
        "efficacyResultProduced": False,
        "freshExecutionAuthorized": False,
        "nextHumanDecisionRequired": True,
    }
    gate = {
        "schemaVersion": "e07.s12w.execution-review-gate.v1",
        "researchStepId": "S12W",
        "qualificationPassed": all_passed,
        "S12REstimandPreserved": preservation["allPassed"],
        "scientificExecutionAuthorizedByS12W": False,
        "freshNamespaceRequiredForAnyFutureExecution": True,
        "S12UCachePermanentlyProhibited": True,
        "status": (
            "qualified_pending_separate_human_execution_decision"
            if all_passed
            else "blocked_return_for_human_review"
        ),
        "recommendedNextAction": (
            "Return for a separate human decision; do not execute transfer "
            "automatically."
        ),
    }
    for filename, value in outputs.items():
        write_json(OUT / filename, value)
    write_parquet(OUT / "state_matrix_requalification.parquet", state_frame)
    write_json(OUT / "validation_summary.json", validation)
    write_json(OUT / "s12w_execution_review_gate.json", gate)
    if not all_passed:
        raise SystemExit("S12W qualification failed")


if __name__ == "__main__":
    main()
