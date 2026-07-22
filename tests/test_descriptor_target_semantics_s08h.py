from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.environment_suite.e05_semantics import (
    TARGET_CHANGE_SEMANTICS_VERSION,
    target_change_terminal_transition,
    validate_target_change_result_semantics,
)
from src.quality_diversity.core import (
    E05_REPAIR_DESCRIPTOR_DOMAIN_VERSION,
    E05_REPAIR_DISTANCE_EDGES,
    _aggregate_descriptors,
    _bin,
    e05_repair_descriptor_domain_record,
)


E05 = "e07_s02_regeneration_1d"


def _outcome(initial: int, final: int, coordinate: float) -> dict:
    movement = {
        "acceptedNativeActionFraction": 0.25,
        "committedDisplacementFraction": 0.125,
    }
    return {
        "initialDistance": initial,
        "finalDistance": final,
        "sourceTerminal": False,
        "nativeMovementDescriptorsByPhase": {
            phase: movement
            for phase in (
                "development",
                "stabilization",
                "recovery",
                "memoryResetRecovery",
                "robustnessFault",
                "transfer",
            )
        },
        "descriptorsByAxis": {
            "robustness": {
                "pairedCompletionDelta": 0.0,
                "pairedResidualDelta": 0.0,
            },
            "repair": {
                "distanceRestorationFraction": coordinate,
                "restrictedRecoveryTimeFraction": 1.0,
                "recoveryCensored": True,
            },
            "memory": {
                "historyInterventionEffect": 0.0,
                "resetInterventionEffect": 0.0,
            },
            "transfer": {
                "frozenStratumSuccessFraction": 0.0,
                "frozenStratumResidual": 1.0,
            },
        },
    }


def _row(family: int, outcome: dict) -> dict:
    return {
        "taskId": E05,
        "scenarioFamilyOrdinal": family,
        "scenarioId": f"s08h:fixture:{family}",
        "stopReason": "phase_event_budget",
        "failed": False,
        "censored": True,
        "outcome": outcome,
    }


def _target(
    stop: str,
    *,
    phase: int,
    hit: int | None,
    probe: int,
    retained: bool,
) -> dict:
    completed = hit is not None
    return {
        "stopReason": stop,
        "targetCompleted": completed,
        "adaptationTime": hit,
        "adaptationCensored": not completed,
        "overshootCensored": not completed,
        "phaseActivationCount": phase,
        "postHitProbeOpportunities": probe,
        "postHitProbeRetained": retained,
    }


def test_protocol_is_outcome_free_and_fails_closed() -> None:
    protocol = yaml.safe_load(
        Path("configs/portfolio/s08h_descriptor_target_semantics.yaml").read_text()
    )
    assert protocol["researchStepId"] == "S08H"
    assert protocol["scope"]["qualificationOnly"] is True
    assert protocol["scope"]["freshFrozenSmokeRows"] == 0
    assert protocol["scope"]["portfolioExecutionRows"] == 0
    assert protocol["scope"]["archiveConstructionCalls"] == 0
    assert protocol["scope"]["validationOutcomeAccesses"] == 0
    assert protocol["scope"]["confirmationOutcomeAccesses"] == 0
    assert (
        protocol["descriptorDomain"]["version"] == E05_REPAIR_DESCRIPTOR_DOMAIN_VERSION
    )
    assert (
        protocol["targetChangeSemantics"]["version"] == TARGET_CHANGE_SEMANTICS_VERSION
    )


def test_complete_algebraic_domain_includes_maximum_finite_deterioration() -> None:
    minimum = e05_repair_descriptor_domain_record(_outcome(1, 496, -495.0))
    maximum = e05_repair_descriptor_domain_record(_outcome(496, 0, 1.0))
    assert minimum["status"] == maximum["status"] == "valid"
    assert E05_REPAIR_DISTANCE_EDGES[0] == -495.0
    assert E05_REPAIR_DISTANCE_EDGES[-1] == 1.0
    assert _bin(-495.0, E05_REPAIR_DISTANCE_EDGES) == 0
    assert _bin(1.0, E05_REPAIR_DISTANCE_EDGES) == len(E05_REPAIR_DISTANCE_EDGES) - 2


def test_zero_initial_distance_is_explicitly_unavailable_not_imputed() -> None:
    record = e05_repair_descriptor_domain_record(_outcome(0, 4, 1.0))
    assert record["status"] == "unavailable"
    assert record["reason"] == "zero_initial_distance_undefined_repair_coordinate"
    assert record["rawAdapterValueRetained"] == 1.0


def test_negative_panel_is_cell_eligible_and_keeps_raw_value() -> None:
    rows = [_row(index, _outcome(4, 12, -2.0)) for index in range(4)]
    repair = next(
        item
        for item in _aggregate_descriptors(E05, rows)
        if item["archiveId"] == "phenotype:e05:repair"
    )
    assert repair["cellEligible"] is True
    assert repair["values"]["distanceRestorationFraction"] == -2.0
    assert repair["descriptorDomainVersion"] == E05_REPAIR_DESCRIPTOR_DOMAIN_VERSION
    assert repair["support"]["repairDomainRecords"][0]["status"] == "valid"


def test_inconsistent_present_repair_coordinate_fails_closed() -> None:
    bad = _outcome(4, 12, -1.5)
    with pytest.raises(ValueError, match="invalid present E05 repair coordinate"):
        _aggregate_descriptors(E05, [_row(index, bad) for index in range(4)])


@pytest.mark.parametrize(
    ("summary", "classification"),
    [
        (
            _target(
                "post_adaptation_probe_complete",
                phase=260,
                hit=100,
                probe=160,
                retained=True,
            ),
            "valid_completed_probe",
        ),
        (
            _target(
                "post_adaptation_probe_complete",
                phase=6560,
                hit=6400,
                probe=160,
                retained=True,
            ),
            "valid_completed_probe",
        ),
        (
            _target(
                "controller_quiescent",
                phase=320,
                hit=None,
                probe=0,
                retained=True,
            ),
            "valid_retained_quiescent_censor",
        ),
        (
            _target(
                "phase_event_budget",
                phase=6400,
                hit=None,
                probe=0,
                retained=True,
            ),
            "valid_retained_phase_budget_censor",
        ),
    ],
)
def test_authoritative_target_change_branches_are_retained(
    summary: dict, classification: str
) -> None:
    audit = validate_target_change_result_semantics(
        summary, adaptation_budget=6400, probe_budget=160
    )
    assert audit["validNativeContract"] is True
    assert audit["classification"] == classification
    assert audit["retainedOutcome"] is True


@pytest.mark.parametrize(
    "summary",
    [
        _target(
            "phase_event_budget",
            phase=6560,
            hit=6500,
            probe=60,
            retained=False,
        ),
        _target(
            "post_adaptation_probe_complete",
            phase=6561,
            hit=6401,
            probe=160,
            retained=True,
        ),
        {
            **_target(
                "invariant_error",
                phase=10,
                hit=None,
                probe=0,
                retained=False,
            ),
            "adaptationCensored": True,
            "overshootCensored": True,
        },
    ],
)
def test_partial_probe_late_hit_and_invariant_are_adapter_failures(
    summary: dict,
) -> None:
    audit = validate_target_change_result_semantics(
        summary, adaptation_budget=6400, probe_budget=160
    )
    assert audit["validNativeContract"] is False
    assert audit["adapterFailure"] is True
    assert audit["classification"] == "adapter_failure"


def test_failure_censor_status_is_not_relabelled() -> None:
    summary = _target(
        "phase_event_budget",
        phase=6400,
        hit=None,
        probe=0,
        retained=True,
    )
    original = deepcopy(summary)
    audit = validate_target_change_result_semantics(
        summary, adaptation_budget=6400, probe_budget=160
    )
    assert summary == original
    assert audit["nativeStopReasonPreserved"] == "phase_event_budget"
    assert audit["adaptationCensoredPreserved"] is True
    assert audit["overshootCensoredPreserved"] is True


def test_terminal_transition_enforces_deadline_then_exact_probe() -> None:
    hit, terminal = target_change_terminal_transition(
        elapsed=6400,
        distance=0,
        first_hit=None,
        quiescent=False,
        invariant_error=False,
        adaptation_budget=6400,
        probe_budget=160,
    )
    assert (hit, terminal) == (6400, None)
    hit, terminal = target_change_terminal_transition(
        elapsed=6560,
        distance=0,
        first_hit=hit,
        quiescent=False,
        invariant_error=False,
        adaptation_budget=6400,
        probe_budget=160,
    )
    assert (hit, terminal) == (6400, "post_adaptation_probe_complete")
    with pytest.raises(ValueError, match="after the adaptation deadline"):
        target_change_terminal_transition(
            elapsed=6401,
            distance=0,
            first_hit=None,
            quiescent=False,
            invariant_error=False,
            adaptation_budget=6400,
            probe_budget=160,
        )
