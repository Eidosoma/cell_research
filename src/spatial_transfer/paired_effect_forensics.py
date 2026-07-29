"""Outcome-independent S12V qualification helpers.

This module is diagnostic-only.  It does not repair the frozen S12U
estimator, read an S12U execution cache, or construct a scientific result.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np

ENDPOINT_KINDS = (
    "terminal_conjunctive_completion_binary",
    "minimum_mismatch_fraction_continuous",
    "repair_by_transition_32_binary",
    "departure_after_initial_completion_binary",
    "reserved_failure_binary",
    "separate_cost_component_nonnegative_integer",
    "restricted_native_transition_time_at_32",
)
DENOMINATOR_STATES = ("positive_finite", "zero", "missing", "nonfinite")
PAIR_SUPPORT_STATES = (
    "complete_one_to_one",
    "empty",
    "partial_left_only",
    "partial_right_only",
    "duplicate_left",
    "duplicate_right",
    "endpoint_missing",
    "endpoint_nonfinite",
)

STATUS_CONTRACTS = {
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

TASK_ENDPOINTS = frozenset(
    {
        "terminal_conjunctive_completion_binary",
        "minimum_mismatch_fraction_continuous",
        "repair_by_transition_32_binary",
        "departure_after_initial_completion_binary",
        "restricted_native_transition_time_at_32",
    }
)
ALL_RESERVED_ENDPOINTS = frozenset(
    {
        "reserved_failure_binary",
        "separate_cost_component_nonnegative_integer",
    }
)


@dataclass(frozen=True)
class ContractState:
    endpointKind: str
    endpointAvailable: bool
    failed: bool
    censored: bool
    diagnostic: bool
    denominatorState: str
    pairedConditionSupport: str
    nativeStatusContract: str


def matching_status_contracts(
    *,
    endpoint_available: bool,
    failed: bool,
    censored: bool,
    diagnostic: bool,
) -> list[str]:
    expected = {
        "endpointAvailable": endpoint_available,
        "failed": failed,
        "censored": censored,
        "diagnostic": diagnostic,
    }
    matches = [
        name for name, contract in STATUS_CONTRACTS.items() if contract == expected
    ]
    return matches or ["INVALID_STATUS_COMBINATION"]


def endpoint_applies(endpoint_kind: str, native_status: str) -> bool:
    if endpoint_kind in ALL_RESERVED_ENDPOINTS:
        return native_status != "INVALID_STATUS_COMBINATION"
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
    raise ValueError(f"unknown endpoint kind: {endpoint_kind}")


def expected_disposition(state: ContractState) -> tuple[str, str]:
    """Return the frozen-contract disposition and root-cause class."""

    support = state.pairedConditionSupport
    if state.nativeStatusContract == "INVALID_STATUS_COMBINATION":
        return "fail_closed_invalid_status_combination", "C03_PROJECTION_FEASIBILITY_DEFECT"
    if support in {"duplicate_left", "duplicate_right"}:
        return "fail_closed_ambiguous_pair_identity", "C03_PROJECTION_FEASIBILITY_DEFECT"
    if support == "endpoint_missing":
        return "fail_closed_missing_endpoint_before_arithmetic", "C03_PROJECTION_FEASIBILITY_DEFECT"
    if support == "endpoint_nonfinite":
        return "fail_closed_nonfinite_endpoint_before_arithmetic", "C03_PROJECTION_FEASIBILITY_DEFECT"
    if state.denominatorState == "nonfinite":
        return "fail_closed_nonfinite_denominator", "C03_PROJECTION_FEASIBILITY_DEFECT"
    if not endpoint_applies(state.endpointKind, state.nativeStatusContract):
        return (
            "fixed_slot_non_evidentiary_endpoint_unavailable_or_inapplicable",
            "C02_EXPLICIT_UNDEFINED_NON_EVIDENTIARY",
        )
    if support == "empty":
        return (
            "fixed_slot_non_evidentiary_no_paired_rows",
            "C02_EXPLICIT_UNDEFINED_NON_EVIDENTIARY",
        )
    if support in {"partial_left_only", "partial_right_only"}:
        return "fail_closed_incomplete_pair_support", "C03_PROJECTION_FEASIBILITY_DEFECT"
    if state.denominatorState in {"zero", "missing"}:
        return (
            "fixed_slot_non_evidentiary_denominator_unavailable",
            "C02_EXPLICIT_UNDEFINED_NON_EVIDENTIARY",
        )
    if support != "complete_one_to_one":
        raise AssertionError(f"unclassified support: {support}")
    if state.denominatorState != "positive_finite":
        raise AssertionError(
            f"unclassified denominator: {state.denominatorState}"
        )
    return "finite_evidentiary_record", "C01_FINITE_CONTRACT_VALID"


def enumerate_contract_states() -> list[dict[str, Any]]:
    """Exhaust the preregistered Boolean/status/denominator/support product."""

    records: list[dict[str, Any]] = []
    for (
        endpoint_kind,
        endpoint_available,
        failed,
        censored,
        diagnostic,
        denominator_state,
        pair_support,
    ) in product(
        ENDPOINT_KINDS,
        (False, True),
        (False, True),
        (False, True),
        (False, True),
        DENOMINATOR_STATES,
        PAIR_SUPPORT_STATES,
    ):
        statuses = matching_status_contracts(
            endpoint_available=endpoint_available,
            failed=failed,
            censored=censored,
            diagnostic=diagnostic,
        )
        for status in statuses:
            state = ContractState(
                endpointKind=endpoint_kind,
                endpointAvailable=endpoint_available,
                failed=failed,
                censored=censored,
                diagnostic=diagnostic,
                denominatorState=denominator_state,
                pairedConditionSupport=pair_support,
                nativeStatusContract=status,
            )
            disposition, cause = expected_disposition(state)
            records.append(
                {
                    **asdict(state),
                    "expectedDisposition": disposition,
                    "expectedCauseClass": cause,
                }
            )
    return records


def arithmetic_bound_evidence() -> dict[str, Any]:
    """Prove finite means for the frozen bounded endpoint domains."""

    logical_budget = 195_072
    signed_int64_max = (1 << 63) - 1
    domain_bounds = {
        "terminal_conjunctive_completion_binary": [0.0, 1.0],
        "minimum_mismatch_fraction_continuous": [0.0, 1.0],
        "repair_by_transition_32_binary": [0.0, 1.0],
        "departure_after_initial_completion_binary": [0.0, 1.0],
        "reserved_failure_binary": [0.0, 1.0],
        "separate_cost_component_nonnegative_integer": [
            0.0,
            float(signed_int64_max),
        ],
        "restricted_native_transition_time_at_32": [0.0, 32.0],
    }
    rows = []
    for endpoint, (lower, upper) in domain_bounds.items():
        # A constant extreme is enough to exercise the largest absolute
        # frozen-domain mean without allocating a bootstrap tensor.
        left = np.full(logical_budget, upper, dtype=np.float64)
        right = np.full(logical_budget, lower, dtype=np.float64)
        difference = left - right
        effect = float(difference.mean())
        rows.append(
            {
                "endpointKind": endpoint,
                "lowerBound": lower,
                "upperBound": upper,
                "maximumAbsoluteDifference": upper - lower,
                "logicalBudget": logical_budget,
                "maximumAbsoluteFiniteSumBound": logical_budget
                * (upper - lower),
                "meanAtExtreme": effect,
                "differenceFinite": bool(np.isfinite(difference).all()),
                "meanFinite": math.isfinite(effect),
            }
        )
    return {
        "schemaVersion": "e07.s12v.arithmetic-bound-evidence.v1",
        "logicalBudget": logical_budget,
        "binaryAndMismatchDifferenceBound": [-1.0, 1.0],
        "restrictedTimeDifferenceBound": [-32.0, 32.0],
        "costConservativeInputBound": "signed_int64_nonnegative",
        "float64Maximum": float(np.finfo(np.float64).max),
        "rows": rows,
        "allFinite": all(row["differenceFinite"] and row["meanFinite"] for row in rows),
        "conclusion": (
            "Contract-valid finite bounded inputs cannot create a non-finite "
            "raw paired mean at the frozen logical budget."
        ),
    }


def disposition_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["expectedDisposition"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
