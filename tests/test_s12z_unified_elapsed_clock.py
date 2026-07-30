from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from src.environment_suite.dsl_adapters import (
    dsl_action,
    run_spatial_dsl_episode,
)
from src.morph2d.baseline import TargetMetricTracker, load_baseline_assets
from src.morph2d.elapsed_clock import (
    ElapsedClockAuthenticator,
    ElapsedClockValidationError,
    build_first_completion_projection,
    parse_and_validate_elapsed_clock_record,
    validate_elapsed_clock_summary,
    validate_first_completion_projection,
)
from src.morph2d.engine import EpisodeDefinition, run_cpu_episode
from src.morph2d.movements import MovementState, initial_movement_state
from src.spatial_transfer.estimator_feasibility import (
    EstimatorFeasibilityError,
    project_authenticated_restricted_time,
)

HORIZON = 32
SCENARIO = "s12z-outcome-independent-clock-fixture"


def _state(elapsed: int) -> MovementState:
    return MovementState(
        environment_id="s12z-synthetic-environment",
        environment_sha256="0" * 64,
        transition_index=elapsed,
        occupancy=(),
    )


def _clock(*, raw_labels: list[int | str | None] | None = None):
    issuer = ElapsedClockAuthenticator(
        scenario_id=SCENARIO,
        horizon_transitions=HORIZON,
    )
    labels = raw_labels or [-1, *range(HORIZON)]
    records = [
        issuer.issue(
            elapsed_transition=elapsed,
            raw_observation_label=labels[elapsed],
            state=_state(elapsed),
        ).to_mapping()
        for elapsed in range(HORIZON + 1)
    ]
    return records, issuer.finalize()


def _projection(first: int | None):
    records, summary = _clock()
    commitment = None if first is None else records[first]["recordCommitmentSha256"]
    projection = build_first_completion_projection(
        clock_summary=summary,
        first_completion_transition=first,
        first_completion_record_commitment_sha256=commitment,
    )
    return records, summary, projection


def _status_row(
    status: str,
    *,
    summary: dict,
    projection: dict,
) -> dict:
    flags = {
        "calibrated_endpoint_retained": (True, False, False, False),
        "repair_observed": (True, False, False, False),
        "right_censored_at_transition_32": (True, False, True, False),
    }
    available, failed, censored, diagnostic = flags[status]
    return {
        "schemaVersion": "e07.s12z.physical-result-projection.v2",
        "scenarioFamilyId": SCENARIO,
        "status": status,
        "endpointAvailable": available,
        "failed": failed,
        "censored": censored,
        "diagnostic": diagnostic,
        "firstCompletionTransition": projection["firstCompletionTransition"],
        "authenticatedElapsedClockSummaryJson": json.dumps(
            summary, sort_keys=True, separators=(",", ":")
        ),
        "authenticatedFirstCompletionProjectionJson": json.dumps(
            projection, sort_keys=True, separators=(",", ":")
        ),
        "rawObservationLabelsUsedAsScientificTime": False,
    }


@pytest.mark.parametrize("first", [*range(HORIZON + 1), None])
def test_every_completion_time_and_noncompletion_are_authenticated(first) -> None:
    _records, summary, projection = _projection(first)
    assert (
        validate_elapsed_clock_summary(
            summary,
            expected_scenario_id=SCENARIO,
            expected_horizon=HORIZON,
        )
        == summary
    )
    assert (
        validate_first_completion_projection(
            projection,
            expected_scenario_id=SCENARIO,
            expected_horizon=HORIZON,
            clock_summary=summary,
        )
        == projection
    )
    assert projection["firstCompletionTransition"] == first
    assert projection["noncompletionRightCensorTransition"] == HORIZON


def test_raw_labels_are_provenance_and_cannot_move_scientific_time() -> None:
    labels = ["initial-provenance", *[-999] * HORIZON]
    records, summary = _clock(raw_labels=labels)
    projection = build_first_completion_projection(
        clock_summary=summary,
        first_completion_transition=7,
        first_completion_record_commitment_sha256=records[7]["recordCommitmentSha256"],
    )
    assert projection["firstCompletionTransition"] == 7
    assert projection["scientificTimeSource"] == "authenticatedElapsedTransition"
    assert projection["rawObservationLabelsUsed"] is False
    assert summary["rawObservationLabelsScientificTime"] is False


@pytest.mark.parametrize(
    "bad",
    [-1, 33, 1.5, True, float("nan"), float("inf"), -float("inf")],
)
def test_issuer_rejects_invalid_elapsed_values(bad) -> None:
    issuer = ElapsedClockAuthenticator(
        scenario_id=SCENARIO,
        horizon_transitions=HORIZON,
    )
    with pytest.raises(ElapsedClockValidationError):
        issuer.issue(
            elapsed_transition=bad,
            raw_observation_label=None,
            state=_state(0),
        )


def test_record_validation_rejects_missing_forged_and_contradictory_metadata() -> None:
    records, _summary = _clock()
    valid = records[12]
    previous = records[11]["recordCommitmentSha256"]
    cases = []
    missing = deepcopy(valid)
    missing.pop("recordCommitmentSha256")
    cases.append(missing)
    for key, value in (
        ("recordCommitmentSha256", "f" * 64),
        ("authenticatedElapsedTransition", 11),
        ("movementStateTransitionIndex", 11),
        ("phase", "initial_state"),
        ("previousCommitmentSha256", "e" * 64),
        ("sequenceOrdinal", 11),
    ):
        case = deepcopy(valid)
        case[key] = value
        cases.append(case)
    for case in cases:
        with pytest.raises(ElapsedClockValidationError):
            parse_and_validate_elapsed_clock_record(
                case,
                expected_scenario_id=SCENARIO,
                expected_horizon=HORIZON,
                expected_ordinal=12,
                expected_elapsed=12,
                expected_previous_commitment=previous,
                state=_state(12),
            )


def test_summary_and_projection_reject_missing_or_forged_authentication() -> None:
    records, summary, projection = _projection(9)
    forged_summary = deepcopy(summary)
    forged_summary["recordCommitmentsByElapsed"][9] = "f" * 64
    with pytest.raises(ElapsedClockValidationError):
        validate_elapsed_clock_summary(forged_summary)
    forged_projection = deepcopy(projection)
    forged_projection["firstCompletionRecordCommitmentSha256"] = "e" * 64
    with pytest.raises(ElapsedClockValidationError):
        validate_first_completion_projection(
            forged_projection,
            clock_summary=summary,
        )
    with pytest.raises(ElapsedClockValidationError):
        build_first_completion_projection(
            clock_summary=summary,
            first_completion_transition=9,
            first_completion_record_commitment_sha256=records[8][
                "recordCommitmentSha256"
            ],
        )


@pytest.mark.parametrize(
    ("status", "first", "expected"),
    [
        ("calibrated_endpoint_retained", 0, 0.0),
        ("calibrated_endpoint_retained", 32, 32.0),
        ("calibrated_endpoint_retained", None, 32.0),
        ("repair_observed", 17, 17.0),
        ("right_censored_at_transition_32", None, 32.0),
    ],
)
def test_s12w_projection_uses_only_authenticated_elapsed_time(
    status, first, expected
) -> None:
    _records, summary, projection = _projection(first)
    row = _status_row(status, summary=summary, projection=projection)
    assert project_authenticated_restricted_time(row) == expected
    forged = deepcopy(row)
    forged["firstCompletionTransition"] = 6
    with pytest.raises(EstimatorFeasibilityError):
        project_authenticated_restricted_time(forged)


def test_native_and_dsl_paths_audit_the_same_elapsed_domain() -> None:
    context, targets, grammars, environments = load_baseline_assets()
    target = targets["stripes_alternating_three_band"]
    grammar = next(
        item for item in grammars.values() if item.target_id == target.target_id
    )
    environment = environments[target.target_id]
    initial = initial_movement_state(environment)
    base_definition = EpisodeDefinition(
        scenario_id="s12z-native-dsl-parity-fixture",
        environment_id=environment.environment_id,
        policy_id="greedy_neighbor_satisfaction_v1",
        relation_grammar_id=grammar.grammar_id,
        channel_mode="none",
        transitions=HORIZON,
        actor_batch_size=4,
        parameters={},
    )
    native_tracker = TargetMetricTracker(environment, target, grammar)
    native = run_cpu_episode(
        context,
        base_definition,
        include_selected_traces=False,
        initial_state_override=initial,
        authenticated_state_audit=native_tracker.observe_authenticated,
        authenticated_elapsed_clock=True,
    )
    with open(
        "src/policy_dsl/baselines/spatial_greedy_local_v1.json",
        encoding="utf-8",
    ) as handle:
        dsl_document = json.load(handle)
    dsl_tracker = TargetMetricTracker(environment, target, grammar)
    dsl = run_spatial_dsl_episode(
        context,
        replace(
            base_definition,
            policy_id="spatial_greedy_local_v1",
        ),
        dsl_action([dsl_document]),
        target_id=target.target_id,
        initial_state_override=initial,
        offline_tracker=dsl_tracker,
        authenticated_elapsed_clock=True,
    )
    native_summary = validate_elapsed_clock_summary(
        native["authenticatedElapsedClockAudit"],
        expected_scenario_id=base_definition.scenario_id,
        expected_horizon=HORIZON,
    )
    dsl_summary = validate_elapsed_clock_summary(
        dsl["authenticatedElapsedClockAudit"],
        expected_scenario_id=base_definition.scenario_id,
        expected_horizon=HORIZON,
    )
    assert (
        native_summary["initialElapsedTransition"],
        native_summary["finalElapsedTransition"],
        native_summary["recordCount"],
    ) == (0, 32, 33)
    assert (
        dsl_summary["initialElapsedTransition"],
        dsl_summary["finalElapsedTransition"],
        dsl_summary["recordCount"],
    ) == (0, 32, 33)
    native_metrics = native_tracker.finalize()
    dsl_metrics = dsl_tracker.finalize()
    assert (
        native_metrics["authenticatedFirstCompletionProjection"][
            "firstCompletionTransition"
        ]
        == dsl_metrics["authenticatedFirstCompletionProjection"][
            "firstCompletionTransition"
        ]
        == 0
    )
