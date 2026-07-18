"""E05 S10 full-population repeated-injury longitudinal runtime.

The runtime preserves the original S01 Scenario and counter-addressed scheduler.
It extends the *phase-local* S01 budget to two injury episodes without changing
the Scenario ID, carries native and S04 fatigue state across episodes, and
records an unobservable second episode as a competing terminal instead of
dropping or substituting the case.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Mapping

import jsonschema

from causal_simulator.action_interface import CommonActionInterface
from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
)
from reference_simulator.engine import evaluate_terminal
from reference_simulator.model import (
    FaultMode,
    RunState,
    Scenario,
    canonical_json_bytes,
    sha256_json,
    state_hash,
)
from reference_simulator.scheduler import scheduled_actor, scheduled_side
from reference_simulator.transition_primitives import (
    commit_proposal,
    cost_delta,
    ledger_identity,
    validate_proposal,
)

from .dynamic_faults import (
    PROCESS_STREAMS,
    DynamicFaultContract,
    DynamicProcessController,
    DynamicProfile,
)
from .lesions import LesionState, apply_lesion, validate_application
from .tasks import (
    Checkpoint,
    _checkpoint_hash,
    absorbing_occupancy_certificate,
    occupancy_values,
    strict_unequal_inversions,
)


BENCHMARK_VERSION = "E05-repeated-injuries-v1"
REPEATED_INJURY_SPEC_SCHEMA_VERSION = "e05.s10.repeated-injury-spec.v1"
REPEATED_INJURY_RUN_SCHEMA_VERSION = "e05.s10.repeated-injury-run.v1"
SEQUENCE_ASSIGNMENT_STREAM = "repeated_injury_sequence_assignment_s10_v1"


class RepeatedArm(str, Enum):
    PRIOR = "prior_injury_retained_fatigue"
    NAIVE = "naive_opportunity_matched"
    FATIGUE_MATCHED = "naive_fatigue_state_matched"
    NO_FATIGUE = "prior_injury_fatigue_disabled"
    NO_SECOND = "prior_injury_no_second"


class Spacing(str, Enum):
    IMMEDIATE = "immediate_v1"
    REST_20N = "rest_20n_v1"
    REST_100N = "rest_100n_v1"

    def opportunities(self, n: int) -> int:
        return {
            Spacing.IMMEDIATE: 0,
            Spacing.REST_20N: 20 * n,
            Spacing.REST_100N: 100 * n,
        }[self]


SEQUENCE_OPERATORS = {
    "repeat_segment_reversal_v1": "segment_reversal_central_v1",
    "repeat_block_transposition_v1": "block_transposition_adjacent_equal_v1",
}


REPEATED_INJURY_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s10/repeated-injury-spec.schema.json",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenAtUtc",
        "frozenQuestion",
        "inherits",
        "population",
        "sequenceAssignment",
        "spacing",
        "postRecoveryStabilization",
        "eligibleControllers",
        "arms",
        "stateContract",
        "outcomes",
        "estimands",
        "multiplicity",
        "validationPanel",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": REPEATED_INJURY_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S10"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "arms": {"type": "array", "minItems": 5, "maxItems": 5},
        "spacing": {"type": "array", "minItems": 3, "maxItems": 3},
    },
}


def validate_repeated_injury_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(REPEATED_INJURY_SPEC_SCHEMA).validate(specification)
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "lesionSpecSchemaVersion": "e05.s03.lesion-spec.v1",
        "dynamicSpecSchemaVersion": "e05.s04.dynamic-fault-spec.v1",
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "continuation": "skip_and_continue",
        "retry": "no_retry",
        "informationPermission": "policy_native_local",
        "target": "direction-aware strict unequal inversion distance",
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"inherited S01-S04 contract changed: {key}")
    sequence = specification["sequenceAssignment"]
    if sequence.get("stream") != SEQUENCE_ASSIGNMENT_STREAM:
        raise ValueError("S10 sequence-assignment stream changed")
    rows = {item["sequenceId"]: item for item in sequence["sequences"]}
    if set(rows) != set(SEQUENCE_OPERATORS):
        raise ValueError("S10 repeated sequence set changed")
    for sequence_id, operator in SEQUENCE_OPERATORS.items():
        if rows[sequence_id]["episode1Operator"] != operator:
            raise ValueError("S10 first injury no longer repeats the assigned operator")
        if rows[sequence_id]["episode2Operator"] != operator:
            raise ValueError("S10 second injury no longer repeats the assigned operator")
    if {item["spacingId"] for item in specification["spacing"]} != {
        item.value for item in Spacing
    }:
        raise ValueError("S10 spacing set changed")
    if {item["armId"] for item in specification["arms"]} != {
        item.value for item in RepeatedArm
    }:
        raise ValueError("S10 arm set changed")
    panel = specification["validationPanel"]
    expected_counts = {
        "plannedCaseCount": 384,
        "plannedSequenceCount": 2,
        "plannedSpacingCount": 3,
        "plannedArmCount": 5,
        "plannedRunCount": 5760,
        "plannedReplayCount": 5760,
        "plannedPrimaryPairedContrasts": 2304,
        "plannedSelectedTraceCount": 24,
        "sequenceAssignmentsPerSequence": 192,
    }
    for key, value in expected_counts.items():
        if panel.get(key) != value:
            raise ValueError(f"S10 planned panel changed: {key}")
    if specification["population"].get("survivorOnlyAnalysis") != "prohibited":
        raise ValueError("S10 survivor-only prohibition was removed")


@dataclass(slots=True)
class _TraceDigest:
    digest: bytes
    count: int
    head: list[dict[str, Any]]
    tail: deque[dict[str, Any]]

    @classmethod
    def create(cls) -> "_TraceDigest":
        return cls(
            hashlib.sha256(b"E05/S10/native-opportunity/v1").digest(),
            0,
            [],
            deque(maxlen=5),
        )

    def add(self, record: dict[str, Any], retain: bool) -> None:
        self.count += 1
        if retain:
            self.digest = hashlib.sha256(
                self.digest + canonical_json_bytes(record)
            ).digest()
            if len(self.head) < 5:
                self.head.append(record)
            self.tail.append(record)


def _delta(final: Mapping[str, int], initial: Mapping[str, int]) -> dict[str, int]:
    keys = set(final) | set(initial)
    return {key: int(final.get(key, 0)) - int(initial.get(key, 0)) for key in keys}


def _distance(scenario: Scenario, state: RunState) -> int:
    return strict_unequal_inversions(
        occupancy_values(scenario, state.occupancy), scenario.cells[0].direction
    )


def _terminal_without_single_repair_envelope(
    scenario: Scenario, state: RunState
) -> str | None:
    terminal = evaluate_terminal(scenario, state)
    return None if terminal == "event_budget" else terminal


def _execute_opportunity(
    scenario: Scenario,
    state: RunState,
    router: ArchitectureProposalRouter,
    controller: DynamicProcessController,
    trace: _TraceDigest,
    *,
    retain_trace: bool,
) -> tuple[bool, str]:
    """Execute one exact native opportunity without the old one-repair envelope."""

    event_index = state.activation_count
    actor_id, _, actor_consumed = scheduled_actor(
        scenario, event_index, include_draws=False
    )
    state.stream_counters["actor_activation"] = (
        state.stream_counters.get("actor_activation", 0) + actor_consumed
    )
    actor = scenario.cell_map[actor_id]
    side = None
    if actor.policy.value == "Bubble" and actor.fault == FaultMode.NORMAL:
        side, _ = scheduled_side(scenario, event_index)
        state.stream_counters["bubble_side"] = (
            state.stream_counters.get("bubble_side", 0) + 1
        )
    proposal = router.proposal_for(scenario, state, actor_id, side=side)
    proposal, consumption = controller.prepare(proposal, event_index)
    for stream, count in consumption:
        state.stream_counters[stream] = state.stream_counters.get(stream, 0) + count
    validation = controller.outcome(
        proposal, validate_proposal(scenario, state, proposal), event_index
    )
    decision = "accepted" if validation.eligible_for_commit else validation.decision
    snapshot = state.clone()
    changed = commit_proposal(state, snapshot, proposal, decision)
    for key, value in cost_delta(state.ledger, proposal, decision).items():
        state.ledger[key] += value
    state.activation_count += 1
    state.terminal = (
        _terminal_without_single_repair_envelope(scenario, state)
        if changed
        else None
    )
    controller.after_batch(
        scenario,
        state,
        (proposal,),
        {proposal.ordinal: decision},
        event_index,
    )
    if retain_trace:
        trace.add(
            {
                "eventIndex": event_index,
                "actorIdentityId": actor_id,
                "proposalKind": proposal.kind.value,
                "decision": decision,
                "stateChanged": changed,
                "terminal": state.terminal,
            },
            True,
        )
    else:
        trace.count += 1
    return decision == "accepted" and proposal.kind.value == "Swap", actor_id


def _checkpoint_from_state(scenario: Scenario, state: RunState) -> Checkpoint:
    state.terminal = None
    return Checkpoint(
        occupancy=tuple(state.occupancy),
        selection_cursors=tuple(sorted(state.selection_cursors.items())),
        activation_count=state.activation_count,
        stream_counters=tuple(sorted(state.stream_counters.items())),
        ledger=tuple(sorted(state.ledger.items())),
        distance=_distance(scenario, state),
        state_hash=_checkpoint_hash(
            scenario.scenario_id,
            state.occupancy,
            state.selection_cursors,
            state.activation_count,
            state.stream_counters,
            state.ledger,
        ),
    )


def _apply_episode_lesion(
    scenario: Scenario,
    state: RunState,
    *,
    operator_id: str,
    pairing_id: str,
    timing_condition_id: str,
    injury_seed: int,
) -> dict[str, Any]:
    checkpoint = _checkpoint_from_state(scenario, state)
    application = apply_lesion(
        operator_id,
        LesionState.from_checkpoint(scenario, checkpoint),
        s01_pairing_block_id=pairing_id,
        timing_condition_id=timing_condition_id,
        injury_seed=injury_seed,
    )
    audit = validate_application(application)
    if not audit["success"]:
        raise AssertionError("S10 lesion failed its inherited S03 invariant audit")
    state.occupancy = list(application.post_state.occupancy)
    state.terminal = _terminal_without_single_repair_envelope(scenario, state)
    return {
        "operatorId": operator_id,
        "preStateHash": application.pre_state.state_hash,
        "postStateHash": application.post_state.state_hash,
        "preOccupancySha256": sha256_json(list(application.pre_state.occupancy)),
        "postOccupancySha256": sha256_json(list(application.post_state.occupancy)),
        "parameters": dict(application.parameters),
        "severity": dict(application.severity),
        "validation": dict(audit),
    }


def _run_recovery(
    scenario: Scenario,
    state: RunState,
    router: ArchitectureProposalRouter,
    controller: DynamicProcessController,
    *,
    budget: int,
    trace: _TraceDigest,
    retain_trace: bool,
) -> dict[str, Any]:
    start_event = state.activation_count
    start_ledger = dict(state.ledger)
    start_process = controller.process_ledger()
    start_distance = _distance(scenario, state)
    current_distance = start_distance
    distance_auc = 0
    stop_reason: str | None = None
    while state.activation_count - start_event < budget:
        terminal = state.terminal
        if terminal is not None:
            stop_reason = terminal
            break
        swapped, _ = _execute_opportunity(
            scenario,
            state,
            router,
            controller,
            trace,
            retain_trace=retain_trace,
        )
        if swapped:
            current_distance = _distance(scenario, state)
        distance_auc += current_distance
    if stop_reason is None:
        terminal = state.terminal
        stop_reason = terminal if terminal is not None else "phase_event_budget"
    duration = state.activation_count - start_event
    if stop_reason == "phase_event_budget":
        state.terminal = "phase_event_budget"
    completed = stop_reason == "complete"
    return {
        "startEventIndex": start_event,
        "endEventIndex": state.activation_count,
        "durationOpportunities": duration,
        "budget": budget,
        "stopReason": stop_reason,
        "completed": completed,
        "censored": not completed,
        "initialDistance": start_distance,
        "finalDistance": current_distance,
        "distanceAuc": distance_auc,
        "ledgerDelta": _delta(dict(state.ledger), start_ledger),
        "processLedgerDelta": _delta(controller.process_ledger(), start_process),
    }


def _run_matched_naive_exposure(
    scenario: Scenario,
    state: RunState,
    router: ArchitectureProposalRouter,
    controller: DynamicProcessController,
    *,
    duration: int,
    trace: _TraceDigest,
    retain_trace: bool,
) -> dict[str, Any]:
    start_event = state.activation_count
    start_ledger = dict(state.ledger)
    first_target_duration: int | None = 0 if _distance(scenario, state) == 0 else None
    current_distance = _distance(scenario, state)
    target_departures = 0
    stop_reason = "matched_exposure_boundary"
    for _ in range(duration):
        terminal = state.terminal
        if terminal in {"invariant_error", "quiescent"}:
            stop_reason = terminal
            break
        was_complete = current_distance == 0
        if was_complete:
            state.terminal = None
        swapped, _ = _execute_opportunity(
            scenario,
            state,
            router,
            controller,
            trace,
            retain_trace=retain_trace,
        )
        if swapped:
            current_distance = _distance(scenario, state)
        now_complete = current_distance == 0
        if was_complete and not now_complete:
            target_departures += 1
        if now_complete and first_target_duration is None:
            first_target_duration = state.activation_count - start_event
    achieved = current_distance == 0 and first_target_duration is not None
    if achieved and stop_reason == "matched_exposure_boundary":
        stop_reason = "matched_boundary_complete"
        state.terminal = "complete"
    elif stop_reason == "matched_exposure_boundary":
        stop_reason = "target_not_reached_by_matched_boundary"
        state.terminal = stop_reason
    return {
        "startEventIndex": start_event,
        "endEventIndex": state.activation_count,
        "durationOpportunities": state.activation_count - start_event,
        "assignedDurationOpportunities": duration,
        "stopReason": stop_reason,
        "completed": achieved,
        "censored": not achieved,
        "firstTargetDuration": first_target_duration,
        "targetDepartures": target_departures,
        "initialDistance": None,
        "finalDistance": current_distance,
        "distanceAuc": None,
        "ledgerDelta": _delta(dict(state.ledger), start_ledger),
        "processLedgerDelta": {},
    }


def _run_target_stable_exposure(
    scenario: Scenario,
    state: RunState,
    router: ArchitectureProposalRouter,
    controller: DynamicProcessController,
    *,
    opportunities: int,
    trace: _TraceDigest,
    retain_trace: bool,
) -> dict[str, Any]:
    start_event = state.activation_count
    start_hash = sha256_json(list(state.occupancy))
    current_distance = _distance(scenario, state)
    departures = 0
    for _ in range(opportunities):
        if current_distance != 0:
            departures += 1
            break
        state.terminal = None
        swapped, _ = _execute_opportunity(
            scenario,
            state,
            router,
            controller,
            trace,
            retain_trace=retain_trace,
        )
        if swapped:
            current_distance = _distance(scenario, state)
            if current_distance != 0:
                departures += 1
                break
    stable = departures == 0 and current_distance == 0
    state.terminal = "complete" if stable else "target_stability_failure"
    return {
        "startEventIndex": start_event,
        "endEventIndex": state.activation_count,
        "plannedOpportunities": opportunities,
        "chargedOpportunities": state.activation_count - start_event,
        "targetDepartures": departures,
        "stable": stable,
        "preOccupancySha256": start_hash,
        "postOccupancySha256": sha256_json(list(state.occupancy)),
    }


def _run_s01_stabilization(
    scenario: Scenario,
    state: RunState,
    router: ArchitectureProposalRouter,
    controller: DynamicProcessController,
    *,
    trace: _TraceDigest,
    retain_trace: bool,
    minimum_per_actor: int = 2,
    cap_multiplier: int = 20,
) -> dict[str, Any]:
    """Run S01's achieved-state certificate inside the retained S04 overlay."""

    if _distance(scenario, state) != 0:
        raise ValueError("S10 stabilization requires an achieved target")
    start_event = state.activation_count
    certificate = absorbing_occupancy_certificate(
        scenario, _checkpoint_from_state(scenario, state)
    )
    counts: Counter[str] = Counter()
    actor_ids = tuple(cell.cell_id for cell in scenario.cells)
    cap = cap_multiplier * len(actor_ids)
    accepted_movement_count = 0
    invalid_terminal: str | None = None
    for _ in range(cap):
        state.terminal = None
        swapped, actor_id = _execute_opportunity(
            scenario,
            state,
            router,
            controller,
            trace,
            retain_trace=retain_trace,
        )
        counts[actor_id] += 1
        accepted_movement_count += int(swapped)
        if state.terminal not in {None, "complete"}:
            invalid_terminal = state.terminal
            break
        if swapped:
            break
        if min(counts.get(identity, 0) for identity in actor_ids) >= minimum_per_actor:
            break
    minimum_observed = min(counts.get(identity, 0) for identity in actor_ids)
    coverage = minimum_observed >= minimum_per_actor
    success = bool(
        certificate["success"]
        and coverage
        and accepted_movement_count == 0
        and _distance(scenario, state) == 0
        and invalid_terminal is None
    )
    state.terminal = "complete" if success else "stabilization_failure"
    return {
        "startEventIndex": start_event,
        "endEventIndex": state.activation_count,
        "opportunities": state.activation_count - start_event,
        "cap": cap,
        "minimumPerActor": minimum_per_actor,
        "minimumObserved": minimum_observed,
        "coveragePass": coverage,
        "acceptedMovementCount": accepted_movement_count,
        "absorbingTargetCertificate": certificate["success"],
        "absorbingProposalCount": certificate["proposalCount"],
        "eligibleOccupancySwapCount": certificate["eligibleOccupancySwapCount"],
        "invalidTerminal": invalid_terminal,
        "success": success,
    }


def fatigue_snapshot(
    controller: DynamicProcessController, event_index: int
) -> dict[str, Any]:
    return {
        "fatigueLoad": dict(sorted(controller.fatigue_load.items())),
        "residualCooldown": {
            identity: max(0, int(until) - event_index)
            for identity, until in sorted(controller.fatigue_until.items())
        },
    }


def inject_fatigue_snapshot(
    controller: DynamicProcessController,
    snapshot: Mapping[str, Any],
    event_index: int,
) -> dict[str, Any]:
    if controller.contract.profile != DynamicProfile.FATIGUE:
        raise ValueError("fatigue matching requires the S04 fatigue profile")
    before = fatigue_snapshot(controller, event_index)
    loads = {str(key): int(value) for key, value in snapshot["fatigueLoad"].items()}
    if set(loads) != set(controller.fatigue_load):
        raise ValueError("fatigue donor identities do not match the recipient")
    controller.fatigue_load = loads
    controller.fatigue_until = {
        str(identity): event_index + int(residual)
        for identity, residual in snapshot["residualCooldown"].items()
        if int(residual) > 0
    }
    after = fatigue_snapshot(controller, event_index)
    return {
        "eventIndex": event_index,
        "beforeSha256": sha256_json(before),
        "donorSha256": sha256_json(snapshot),
        "afterSha256": sha256_json(after),
        "exactMatch": sha256_json(after) == sha256_json(snapshot),
    }


def _internal_state(scenario: Scenario, state: RunState) -> dict[str, Any]:
    return {
        "occupancySha256": sha256_json(list(state.occupancy)),
        "selectionCursorsSha256": sha256_json(dict(state.selection_cursors)),
        "streamCountersSha256": sha256_json(dict(state.stream_counters)),
        "ledgerSha256": sha256_json(dict(state.ledger)),
        "stateHash": state_hash(scenario.scenario_id, state),
        "globalEventIndex": state.activation_count,
    }


def _run_arm(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    pairing_id: str,
    timing_condition_id: str,
    sequence_id: str,
    spacing: Spacing,
    arm: RepeatedArm,
    injury_seed: int,
    recovery_budget: int,
    donor_duration: int | None = None,
    donor_fatigue_snapshot: Mapping[str, Any] | None = None,
    retain_trace: bool = False,
) -> dict[str, Any]:
    operator_id = SEQUENCE_OPERATORS[sequence_id]
    state = checkpoint.to_run_state()
    initial_state = _internal_state(scenario, state)
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    profile = (
        DynamicProfile.SHAM
        if arm == RepeatedArm.NO_FATIGUE
        else DynamicProfile.FATIGUE
    )
    selected = state.occupancy[(len(state.occupancy) - 1) // 2]
    controller = DynamicProcessController(
        DynamicFaultContract(profile),
        scenario,
        selected,
        state.activation_count,
        retain_audits=retain_trace,
    )
    router = ArchitectureProposalRouter(
        ArchitectureExecutionContract.distributed_local(),
        action_interface=CommonActionInterface(),
    )
    trace = _TraceDigest.create()
    episode1_lesion: dict[str, Any] | None = None
    if arm in {RepeatedArm.NAIVE, RepeatedArm.FATIGUE_MATCHED}:
        if donor_duration is None:
            raise ValueError("naive S10 controls require the paired donor duration")
        episode1 = _run_matched_naive_exposure(
            scenario,
            state,
            router,
            controller,
            duration=donor_duration,
            trace=trace,
            retain_trace=retain_trace,
        )
        episode1_injury_administered = False
    else:
        episode1_lesion = _apply_episode_lesion(
            scenario,
            state,
            operator_id=operator_id,
            pairing_id=pairing_id,
            timing_condition_id=timing_condition_id,
            injury_seed=injury_seed,
        )
        episode1 = _run_recovery(
            scenario,
            state,
            router,
            controller,
            budget=recovery_budget,
            trace=trace,
            retain_trace=retain_trace,
        )
        episode1_injury_administered = True

    first_stage_eligible = bool(episode1["completed"])
    stabilization = None
    if first_stage_eligible:
        stabilization = _run_s01_stabilization(
            scenario,
            state,
            router,
            controller,
            trace=trace,
            retain_trace=retain_trace,
        )
        first_stage_eligible = bool(stabilization["success"])
    rest = None
    if first_stage_eligible:
        rest = _run_target_stable_exposure(
            scenario,
            state,
            router,
            controller,
            opportunities=spacing.opportunities(len(state.occupancy)),
            trace=trace,
            retain_trace=retain_trace,
        )
        first_stage_eligible = bool(rest["stable"])
    boundary_state = _internal_state(scenario, state)
    boundary_fatigue = fatigue_snapshot(controller, state.activation_count)
    injection = None
    if arm == RepeatedArm.FATIGUE_MATCHED and first_stage_eligible:
        if donor_fatigue_snapshot is None:
            raise ValueError("fatigue-matched S10 control requires a donor snapshot")
        injection = inject_fatigue_snapshot(
            controller, donor_fatigue_snapshot, state.activation_count
        )
        boundary_fatigue = fatigue_snapshot(controller, state.activation_count)

    episode2_lesion: dict[str, Any] | None = None
    episode2: dict[str, Any] | None = None
    stability_probe: dict[str, Any] | None = None
    episode2_administered = False
    if first_stage_eligible and arm == RepeatedArm.NO_SECOND:
        stability_probe = _run_target_stable_exposure(
            scenario,
            state,
            router,
            controller,
            opportunities=20 * len(state.occupancy),
            trace=trace,
            retain_trace=retain_trace,
        )
    elif first_stage_eligible:
        episode2_lesion = _apply_episode_lesion(
            scenario,
            state,
            operator_id=operator_id,
            pairing_id=pairing_id,
            timing_condition_id=f"s10_episode2::{spacing.value}",
            injury_seed=injury_seed,
        )
        episode2_administered = True
        episode2 = _run_recovery(
            scenario,
            state,
            router,
            controller,
            budget=recovery_budget,
            trace=trace,
            retain_trace=retain_trace,
        )

    if arm == RepeatedArm.NO_SECOND:
        cause = "no_second_injury_control" if first_stage_eligible else "episode1_competing_terminal"
    elif not first_stage_eligible:
        cause = "episode1_competing_terminal"
    elif episode2 is None:
        cause = "episode2_missing_internal_error"
    elif episode2["completed"]:
        cause = "episode2_complete"
    elif episode2["stopReason"] == "quiescent":
        cause = "episode2_quiescent"
    elif episode2["stopReason"] == "phase_event_budget":
        cause = "episode2_phase_event_budget"
    else:
        cause = f"episode2_{episode2['stopReason']}"

    final_ledger = dict(state.ledger)
    final_streams = dict(state.stream_counters)
    ledger_delta = _delta(final_ledger, initial_ledger)
    stream_delta = _delta(final_streams, initial_streams)
    process_ledger = controller.process_ledger()
    episode2_complete = bool(episode2 and episode2["completed"])
    comparable_arm = arm != RepeatedArm.NO_SECOND
    restricted_second = (
        None
        if not comparable_arm
        else (
            int(episode2["durationOpportunities"])
            if episode2_complete and episode2 is not None
            else recovery_budget
        )
    )
    joint_success = (
        None
        if not comparable_arm
        else bool(first_stage_eligible and episode2_complete)
    )
    validations = {
        **ledger_identity(ledger_delta),
        "activationLedgerMatchesClock": ledger_delta["activations"]
        == state.activation_count - checkpoint.activation_count,
        "processOpportunityMatchesClock": process_ledger["chargedOpportunities"]
        == state.activation_count - checkpoint.activation_count,
        "noS10RuntimeStream": SEQUENCE_ASSIGNMENT_STREAM not in stream_delta,
        "noS04ExogenousProcessStream": all(stream_delta.get(item, 0) == 0 for item in PROCESS_STREAMS),
        "fatigueIsStreamFree": profile != DynamicProfile.FATIGUE
        or all(stream_delta.get(item, 0) == 0 for item in PROCESS_STREAMS),
        "fatigueMovementIdentity": profile != DynamicProfile.FATIGUE
        or process_ledger["fatigueMovementParticipations"]
        == 2 * ledger_delta["acceptedSwaps"],
        "firstBudgetRespected": int(episode1["durationOpportunities"])
        <= recovery_budget,
        "secondBudgetRespected": episode2 is None
        or int(episode2["durationOpportunities"]) <= recovery_budget,
        "noSecondAfterCompetingTerminal": first_stage_eligible
        or not episode2_administered,
        "episode2OnlyFromExactTarget": not episode2_administered
        or episode2_lesion is not None
        and int(episode2_lesion["severity"]["validPostTargetOrderDistanceBefore"])
        == 0,
        "fatigueInjectionExact": injection is None or bool(injection["exactMatch"]),
        "stabilizationAccounting": stabilization is None
        or int(stabilization["opportunities"]) <= int(stabilization["cap"]),
        "episode2RequiresStabilization": not episode2_administered
        or stabilization is not None
        and bool(stabilization["success"]),
        "targetStableRest": rest is None or bool(rest["stable"]),
        "noSecondProbeStable": stability_probe is None or bool(stability_probe["stable"]),
        "phaseLocalEnvelopeUsed": True,
    }
    result = {
        "schemaVersion": REPEATED_INJURY_RUN_SCHEMA_VERSION,
        "benchmarkVersion": BENCHMARK_VERSION,
        "arm": arm.value,
        "sequenceId": sequence_id,
        "operatorId": operator_id,
        "spacingId": spacing.value,
        "spacingOpportunities": spacing.opportunities(len(state.occupancy)),
        "sourceScenarioId": scenario.scenario_id,
        "sourceCheckpointHash": checkpoint.state_hash,
        "initialState": initial_state,
        "episode1InjuryAdministered": episode1_injury_administered,
        "episode1Lesion": episode1_lesion,
        "episode1": episode1,
        "firstStageEligible": first_stage_eligible,
        "postRecoveryStabilization": stabilization,
        "rest": rest,
        "preEpisode2State": boundary_state,
        "preEpisode2Fatigue": boundary_fatigue,
        "fatigueInjection": injection,
        "episode2InjuryAdministered": episode2_administered,
        "episode2Lesion": episode2_lesion,
        "episode2": episode2,
        "noSecondStabilityProbe": stability_probe,
        "episode2OutcomeCause": cause,
        "episode2Observed": episode2_administered,
        "jointSequenceSuccess": joint_success,
        "restrictedEpisode2Time": restricted_second,
        "finalState": _internal_state(scenario, state),
        "finalFatigue": fatigue_snapshot(controller, state.activation_count),
        "nativeLedgerDelta": ledger_delta,
        "streamCounterDelta": stream_delta,
        "processLedger": process_ledger,
        "processAuditDigest": controller.audit_digest.hex(),
        "processAuditCount": controller.audit_count,
        "processTransitions": list(controller.transitions),
        "nativeEventDigest": trace.digest.hex(),
        "nativeEventCount": trace.count,
        "traceHead": trace.head,
        "traceTail": list(trace.tail),
        "validation": validations,
    }
    result["runDigest"] = sha256_json(result)
    return result


def run_repeated_case(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    pairing_id: str,
    timing_condition_id: str,
    sequence_id: str,
    spacing: Spacing,
    injury_seed: int,
    recovery_budget: int,
    retain_trace: bool = False,
) -> dict[str, dict[str, Any]]:
    """Execute all five frozen arms for one case/spacing dependency group."""

    if sequence_id not in SEQUENCE_OPERATORS:
        raise ValueError("unknown S10 sequence")
    prior = _run_arm(
        scenario,
        checkpoint,
        pairing_id=pairing_id,
        timing_condition_id=timing_condition_id,
        sequence_id=sequence_id,
        spacing=spacing,
        arm=RepeatedArm.PRIOR,
        injury_seed=injury_seed,
        recovery_budget=recovery_budget,
        retain_trace=retain_trace,
    )
    donor_duration = int(prior["episode1"]["durationOpportunities"])
    donor_snapshot = prior["preEpisode2Fatigue"]
    answer = {RepeatedArm.PRIOR.value: prior}
    for arm in (
        RepeatedArm.NAIVE,
        RepeatedArm.FATIGUE_MATCHED,
        RepeatedArm.NO_FATIGUE,
        RepeatedArm.NO_SECOND,
    ):
        answer[arm.value] = _run_arm(
            scenario,
            checkpoint,
            pairing_id=pairing_id,
            timing_condition_id=timing_condition_id,
            sequence_id=sequence_id,
            spacing=spacing,
            arm=arm,
            injury_seed=injury_seed,
            recovery_budget=recovery_budget,
            donor_duration=donor_duration,
            donor_fatigue_snapshot=donor_snapshot,
            retain_trace=retain_trace,
        )
    return answer


def exact_replay_repeated_case(
    result: Mapping[str, Mapping[str, Any]],
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    pairing_id: str,
    timing_condition_id: str,
    sequence_id: str,
    spacing: Spacing,
    injury_seed: int,
    recovery_budget: int,
    retain_trace: bool = False,
) -> dict[str, dict[str, Any]]:
    replay = run_repeated_case(
        scenario,
        checkpoint,
        pairing_id=pairing_id,
        timing_condition_id=timing_condition_id,
        sequence_id=sequence_id,
        spacing=spacing,
        injury_seed=injury_seed,
        recovery_budget=recovery_budget,
        retain_trace=retain_trace,
    )
    if canonical_json_bytes(replay) != canonical_json_bytes(result):
        raise AssertionError("S10 repeated-injury replay mismatch")
    return replay
