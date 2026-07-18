"""E05 S11 factorial structural/native/fatigue state-reset runtime.

The intervention forks the exact S10 prior-history trajectory immediately after
the matched second lesion.  All first-stage competing terminals remain in every
assigned arm; no reset or second recovery is invented for them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import jsonschema

from causal_simulator.action_interface import CommonActionInterface
from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
)
from reference_simulator.model import (
    RunState,
    Scenario,
    canonical_json_bytes,
    sha256_json,
    state_hash,
)
from reference_simulator.rng import bounded
from reference_simulator.transition_primitives import ledger_identity

from .dynamic_faults import (
    PROCESS_STREAMS,
    DynamicFaultContract,
    DynamicProcessController,
    DynamicProfile,
)
from .repeated_injuries import (
    SEQUENCE_OPERATORS,
    Spacing,
    _TraceDigest,
    _apply_episode_lesion,
    _delta,
    _distance,
    _internal_state,
    _run_recovery,
    _run_s01_stabilization,
    _run_target_stable_exposure,
    fatigue_snapshot,
)
from .tasks import Checkpoint, occupancy_values, strict_unequal_inversions


BENCHMARK_VERSION = "E05-structural-internal-memory-reset-v1"
STATE_MEMORY_RESET_SPEC_SCHEMA_VERSION = "e05.s11.state-memory-reset-spec.v1"
STATE_MEMORY_RESET_RUN_SCHEMA_VERSION = "e05.s11.state-memory-reset-run.v1"
SCRAMBLE_STREAM = "state_reset_distance_matched_scramble_s11_v1"
ENGINEERED_EMPTY_STATE: dict[str, Any] = {}


class ResetArm(str, Enum):
    A0_N0_F0 = "factorial_a0_n0_f0"
    A0_N0_F1 = "factorial_a0_n0_f1"
    A0_N1_F0 = "factorial_a0_n1_f0"
    A0_N1_F1 = "factorial_a0_n1_f1"
    A1_N0_F0 = "factorial_a1_n0_f0"
    A1_N0_F1 = "factorial_a1_n0_f1"
    A1_N1_F0 = "factorial_a1_n1_f0"
    A1_N1_F1 = "factorial_a1_n1_f1"
    SHAM_ARRANGEMENT = "sham_arrangement_roundtrip"
    SHAM_INTERNAL = "sham_internal_roundtrip"

    @property
    def arrangement_reset(self) -> bool:
        return self.value.startswith("factorial_a1")

    @property
    def native_reset(self) -> bool:
        return self.value.startswith("factorial_") and "_n1_" in self.value

    @property
    def fatigue_reset(self) -> bool:
        return self.value.startswith("factorial_") and self.value.endswith("_f1")

    @property
    def is_factorial(self) -> bool:
        return self.value.startswith("factorial_")


STATE_MEMORY_RESET_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s11/state-memory-reset-spec.schema.json",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenAtUtc",
        "frozenBeforeConfirmatoryExecution",
        "frozenQuestion",
        "inherits",
        "population",
        "injuryContract",
        "statePartitions",
        "arms",
        "resetGateway",
        "scrambleFeasibility",
        "estimands",
        "multiplicity",
        "validationPanel",
        "outcomeRule",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": STATE_MEMORY_RESET_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S11"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "frozenBeforeConfirmatoryExecution": {"const": True},
        "arms": {"type": "array", "minItems": 10, "maxItems": 10},
    },
}


def validate_state_memory_reset_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(STATE_MEMORY_RESET_SPEC_SCHEMA).validate(
        specification
    )
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "lesionSpecSchemaVersion": "e05.s03.lesion-spec.v1",
        "dynamicSpecSchemaVersion": "e05.s04.dynamic-fault-spec.v1",
        "repeatedInjurySpecSchemaVersion": "e05.s10.repeated-injury-spec.v1",
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "continuation": "skip_and_continue",
        "retry": "no_retry",
        "informationPermission": "policy_native_local",
        "target": "direction-aware strict unequal inversion distance",
        "spacing": Spacing.REST_20N.value,
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"inherited S01-S10 contract changed: {key}")
    arms = {item["armId"]: item for item in specification["arms"]}
    if set(arms) != {item.value for item in ResetArm}:
        raise ValueError("S11 arm set changed")
    for arm in ResetArm:
        row = arms[arm.value]
        expected_factors = (
            arm.arrangement_reset,
            arm.native_reset,
            arm.fatigue_reset,
        )
        observed = (
            bool(row["arrangementReset"]),
            bool(row["nativeStateReset"]),
            bool(row["fatigueReset"]),
        )
        if observed != expected_factors:
            raise ValueError(f"S11 arm factor coding changed: {arm.value}")
    panel = specification["validationPanel"]
    expected_counts = {
        "plannedCaseCount": 384,
        "plannedSequenceCount": 2,
        "plannedFactorialArmCount": 8,
        "plannedShamArmCount": 2,
        "plannedArmCount": 10,
        "plannedRunCount": 3840,
        "plannedReplayCount": 3840,
        "plannedResetAuditCount": 3840,
        "plannedFactorialCaseContrastCount": 2688,
        "plannedFactorialEffectCount": 14,
        "plannedShamPairCount": 768,
        "plannedSelectedTraceGroupCount": 16,
    }
    for key, value in expected_counts.items():
        if panel.get(key) != value:
            raise ValueError(f"S11 planned panel changed: {key}")
    if specification["population"].get("survivorOnlyAnalysis") != "prohibited":
        raise ValueError("S11 survivor-only prohibition was removed")
    if specification["statePartitions"]["engineeredState"].get("fields") != []:
        raise ValueError("S11 silently promoted engineered state")


def _target_identity_order(scenario: Scenario) -> list[str]:
    reverse = scenario.cells[0].direction.value == "descending"
    values = [cell.value for cell in scenario.cells]
    if len(values) != len(set(values)):
        raise ValueError("S11 exact matched scramble requires unique S02 values")
    return [
        cell.cell_id
        for cell in sorted(scenario.cells, key=lambda item: item.value, reverse=reverse)
    ]


def _permutation_from_lehmer(digits: list[int]) -> list[int]:
    choices = list(range(len(digits)))
    answer: list[int] = []
    for digit in digits:
        answer.append(choices.pop(digit))
    return answer


def distance_matched_scramble(
    scenario: Scenario,
    occupancy: list[str] | tuple[str, ...],
    *,
    case_address: int,
    attempt_cap: int = 64,
) -> dict[str, Any]:
    """Generate a non-identical exact-inversion permutation without runtime draws."""

    before = list(occupancy)
    n = len(before)
    target = _target_identity_order(scenario)
    distance = strict_unequal_inversions(
        occupancy_values(scenario, before), scenario.cells[0].direction
    )
    maximum = n * (n - 1) // 2
    if distance in {0, maximum}:
        return {
            "feasible": False,
            "reason": "unique_distance_class",
            "targetDistance": distance,
            "maximumDistance": maximum,
            "attempts": 0,
            "drawBlocksConsumed": 0,
            "stream": SCRAMBLE_STREAM,
            "candidateOccupancy": None,
        }
    total_consumed = 0
    for attempt in range(attempt_cap):
        remaining = distance
        digits: list[int] = []
        cursor = attempt * 1024
        for position in range(n):
            remaining_slots = n - position
            maximum_digit = remaining_slots - 1
            maximum_rest = (remaining_slots - 1) * (remaining_slots - 2) // 2
            low = max(0, remaining - maximum_rest)
            high = min(maximum_digit, remaining)
            offset, consumed = bounded(
                scenario.seed,
                scenario.scenario_id,
                SCRAMBLE_STREAM,
                case_address,
                high - low + 1,
                cursor,
            )
            cursor += consumed
            total_consumed += consumed
            digit = low + offset
            digits.append(digit)
            remaining -= digit
        if remaining != 0:
            raise AssertionError("constrained Lehmer sampler left inversion mass")
        ranks = _permutation_from_lehmer(digits)
        candidate = [target[index] for index in ranks]
        if candidate == before:
            continue
        candidate_distance = strict_unequal_inversions(
            occupancy_values(scenario, candidate), scenario.cells[0].direction
        )
        if candidate_distance != distance:
            raise AssertionError("matched scramble changed target distance")
        before_position = {identity: index for index, identity in enumerate(before)}
        displacement = sum(
            abs(index - before_position[identity])
            for index, identity in enumerate(candidate)
        )
        return {
            "feasible": True,
            "reason": None,
            "targetDistance": distance,
            "maximumDistance": maximum,
            "attempts": attempt + 1,
            "drawBlocksConsumed": total_consumed,
            "stream": SCRAMBLE_STREAM,
            "candidateOccupancy": candidate,
            "candidateOccupancySha256": sha256_json(candidate),
            "sourceOccupancySha256": sha256_json(before),
            "identitySetConserved": set(candidate) == set(before),
            "identityDisplacementL1": displacement,
            "movedIdentityCount": sum(a != b for a, b in zip(before, candidate)),
        }
    return {
        "feasible": False,
        "reason": "attempt_cap_exhausted",
        "targetDistance": distance,
        "maximumDistance": maximum,
        "attempts": attempt_cap,
        "drawBlocksConsumed": total_consumed,
        "stream": SCRAMBLE_STREAM,
        "candidateOccupancy": None,
    }


def _partition_snapshot(
    scenario: Scenario,
    state: RunState,
    controller: DynamicProcessController,
) -> dict[str, Any]:
    selection = dict(sorted(state.selection_cursors.items()))
    fatigue = fatigue_snapshot(controller, state.activation_count)
    protected = {
        "globalEventIndex": state.activation_count,
        "streamCountersSha256": sha256_json(dict(state.stream_counters)),
        "nativeLedgerSha256": sha256_json(dict(state.ledger)),
        "processLedgerSha256": sha256_json(controller.process_ledger()),
        "processAuditDigest": controller.audit_digest.hex(),
        "processAuditCount": controller.audit_count,
    }
    return {
        "arrangementSha256": sha256_json(list(state.occupancy)),
        "nativeSelectionCursorsSha256": sha256_json(selection),
        "engineeredStateSha256": sha256_json(ENGINEERED_EMPTY_STATE),
        "fatigueSha256": sha256_json(fatigue),
        "protectedExecutionSha256": sha256_json(protected),
        "stateHash": state_hash(scenario.scenario_id, state),
        "targetDistance": _distance(scenario, state),
        "selectionCursorCount": len(selection),
        "fatigueLoadSum": sum(int(value) for value in fatigue["fatigueLoad"].values()),
        "fatiguedIdentityCount": len(fatigue["residualCooldown"]),
        "protected": protected,
    }


def _zero_fatigue(controller: DynamicProcessController) -> None:
    controller.fatigue_load = {
        identity: 0 for identity in sorted(controller.fatigue_load)
    }
    controller.fatigue_until = {}


def _apply_reset(
    scenario: Scenario,
    state: RunState,
    controller: DynamicProcessController,
    checkpoint: Checkpoint,
    arm: ResetArm,
    scramble: Mapping[str, Any],
) -> dict[str, Any]:
    before = _partition_snapshot(scenario, state, controller)
    before_occupancy = list(state.occupancy)
    before_selection = dict(state.selection_cursors)
    before_fatigue = fatigue_snapshot(controller, state.activation_count)
    reset_feasible = True
    reset_reason = None
    structural_invoked = arm.arrangement_reset or arm == ResetArm.SHAM_ARRANGEMENT
    internal_invoked = (
        arm.native_reset or arm.fatigue_reset or arm == ResetArm.SHAM_INTERNAL
    )
    structural_writes = 0
    native_writes = 0
    fatigue_writes = 0
    if arm.arrangement_reset:
        if not bool(scramble["feasible"]):
            reset_feasible = False
            reset_reason = f"reset_infeasible_{scramble['reason']}"
        else:
            state.occupancy = list(scramble["candidateOccupancy"])
            state.terminal = None
            structural_writes = len(state.occupancy)
    elif arm == ResetArm.SHAM_ARRANGEMENT:
        state.occupancy = list(before_occupancy)
        structural_writes = len(state.occupancy)
    if arm.native_reset:
        state.selection_cursors = dict(checkpoint.selection_cursors)
        native_writes = len(
            set(before_selection) | set(state.selection_cursors)
        )
    if arm.fatigue_reset:
        _zero_fatigue(controller)
        fatigue_writes = len(controller.fatigue_load) + len(
            before_fatigue["residualCooldown"]
        )
    if arm == ResetArm.SHAM_INTERNAL:
        state.selection_cursors = dict(before_selection)
        controller.fatigue_load = dict(before_fatigue["fatigueLoad"])
        controller.fatigue_until = {
            identity: state.activation_count + int(residual)
            for identity, residual in before_fatigue["residualCooldown"].items()
            if int(residual) > 0
        }
        native_writes = len(before_selection)
        fatigue_writes = len(controller.fatigue_load) + len(
            before_fatigue["residualCooldown"]
        )
    after = _partition_snapshot(scenario, state, controller)
    baseline_selection_hash = sha256_json(dict(checkpoint.selection_cursors))
    baseline_fatigue = {
        "fatigueLoad": {
            identity: 0 for identity in sorted(controller.fatigue_load)
        },
        "residualCooldown": {},
    }
    validations = {
        "protectedExecutionStateUnchanged": before["protectedExecutionSha256"]
        == after["protectedExecutionSha256"],
        "engineeredStateEmptyBefore": before["engineeredStateSha256"]
        == sha256_json({}),
        "engineeredStateEmptyAfter": after["engineeredStateSha256"]
        == sha256_json({}),
        "arrangementTargetDistanceMatched": not arm.arrangement_reset
        or not reset_feasible
        or before["targetDistance"] == after["targetDistance"],
        "arrangementIdentityConserved": not arm.arrangement_reset
        or not reset_feasible
        or bool(scramble["identitySetConserved"]),
        "arrangementResetAppliedExactly": not arm.arrangement_reset
        or not reset_feasible
        or before["arrangementSha256"] != after["arrangementSha256"],
        "arrangementPreservedExactly": arm.arrangement_reset
        or before["arrangementSha256"] == after["arrangementSha256"],
        "nativeResetExact": not arm.native_reset
        or after["nativeSelectionCursorsSha256"] == baseline_selection_hash,
        "nativePreservedExact": arm.native_reset
        or before["nativeSelectionCursorsSha256"]
        == after["nativeSelectionCursorsSha256"],
        "fatigueResetExact": not arm.fatigue_reset
        or after["fatigueSha256"] == sha256_json(baseline_fatigue),
        "fatiguePreservedExact": arm.fatigue_reset
        or before["fatigueSha256"] == after["fatigueSha256"],
        "shamArrangementExact": arm != ResetArm.SHAM_ARRANGEMENT
        or before["stateHash"] == after["stateHash"],
        "shamInternalExact": arm != ResetArm.SHAM_INTERNAL
        or before["stateHash"] == after["stateHash"]
        and before["fatigueSha256"] == after["fatigueSha256"],
        "zeroOpportunityAndRuntimeDrawCost": before["protected"]["globalEventIndex"]
        == after["protected"]["globalEventIndex"]
        and before["protected"]["streamCountersSha256"]
        == after["protected"]["streamCountersSha256"],
        "zeroNativeAndProcessLedgerCost": before["protected"]["nativeLedgerSha256"]
        == after["protected"]["nativeLedgerSha256"]
        and before["protected"]["processLedgerSha256"]
        == after["protected"]["processLedgerSha256"],
    }
    audit = {
        "arm": arm.value,
        "resetApplicable": True,
        "resetFeasible": reset_feasible,
        "resetReason": reset_reason,
        "arrangementResetAssigned": arm.arrangement_reset,
        "nativeStateResetAssigned": arm.native_reset,
        "fatigueResetAssigned": arm.fatigue_reset,
        "structuralGatewayInvoked": structural_invoked,
        "internalGatewayInvoked": internal_invoked,
        "engineeredStateFieldCount": 0,
        "supplementalStructuralWrites": structural_writes,
        "supplementalNativeWrites": native_writes,
        "supplementalFatigueWrites": fatigue_writes,
        "before": before,
        "after": after,
        "scramble": dict(scramble),
        "validation": validations,
    }
    audit["auditDigest"] = sha256_json(audit)
    return audit


@dataclass(slots=True)
class _Prefix:
    state: RunState
    controller: DynamicProcessController
    trace: _TraceDigest
    initial_ledger: dict[str, int]
    initial_streams: dict[str, int]
    initial_state: dict[str, Any]
    episode1_lesion: dict[str, Any]
    episode1: dict[str, Any]
    stabilization: dict[str, Any] | None
    rest: dict[str, Any] | None
    first_stage_eligible: bool
    episode2_lesion: dict[str, Any] | None
    pre_reset: dict[str, Any]
    scramble: dict[str, Any]


def _prepare_prefix(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    pairing_id: str,
    timing_condition_id: str,
    sequence_id: str,
    injury_seed: int,
    recovery_budget: int,
    case_address: int,
    retain_trace: bool,
) -> _Prefix:
    operator_id = SEQUENCE_OPERATORS[sequence_id]
    state = checkpoint.to_run_state()
    initial_state = _internal_state(scenario, state)
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    selected = state.occupancy[(len(state.occupancy) - 1) // 2]
    controller = DynamicProcessController(
        DynamicFaultContract(DynamicProfile.FATIGUE),
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
    eligible = bool(episode1["completed"])
    stabilization = None
    if eligible:
        stabilization = _run_s01_stabilization(
            scenario,
            state,
            router,
            controller,
            trace=trace,
            retain_trace=retain_trace,
        )
        eligible = bool(stabilization["success"])
    rest = None
    if eligible:
        rest = _run_target_stable_exposure(
            scenario,
            state,
            router,
            controller,
            opportunities=Spacing.REST_20N.opportunities(len(state.occupancy)),
            trace=trace,
            retain_trace=retain_trace,
        )
        eligible = bool(rest["stable"])
    episode2_lesion = None
    if eligible:
        episode2_lesion = _apply_episode_lesion(
            scenario,
            state,
            operator_id=operator_id,
            pairing_id=pairing_id,
            timing_condition_id="s11_episode2::rest_20n_v1",
            injury_seed=injury_seed,
        )
    pre_reset = _partition_snapshot(scenario, state, controller)
    scramble = (
        distance_matched_scramble(
            scenario,
            state.occupancy,
            case_address=case_address,
        )
        if eligible
        else {
            "feasible": False,
            "reason": "episode1_competing_terminal",
            "targetDistance": None,
            "maximumDistance": None,
            "attempts": 0,
            "drawBlocksConsumed": 0,
            "stream": SCRAMBLE_STREAM,
            "candidateOccupancy": None,
        }
    )
    return _Prefix(
        state,
        controller,
        trace,
        initial_ledger,
        initial_streams,
        initial_state,
        episode1_lesion,
        episode1,
        stabilization,
        rest,
        eligible,
        episode2_lesion,
        pre_reset,
        scramble,
    )


def _run_arm_from_prefix(
    scenario: Scenario,
    checkpoint: Checkpoint,
    prefix: _Prefix,
    *,
    arm: ResetArm,
    sequence_id: str,
    recovery_budget: int,
    retain_trace: bool,
) -> dict[str, Any]:
    state = prefix.state.clone()
    controller = copy.deepcopy(prefix.controller)
    trace = copy.deepcopy(prefix.trace)
    router = ArchitectureProposalRouter(
        ArchitectureExecutionContract.distributed_local(),
        action_interface=CommonActionInterface(),
    )
    if prefix.first_stage_eligible:
        reset_audit = _apply_reset(
            scenario,
            state,
            controller,
            checkpoint,
            arm,
            prefix.scramble,
        )
    else:
        reset_audit = {
            "arm": arm.value,
            "resetApplicable": False,
            "resetFeasible": False,
            "resetReason": "episode1_competing_terminal",
            "arrangementResetAssigned": arm.arrangement_reset,
            "nativeStateResetAssigned": arm.native_reset,
            "fatigueResetAssigned": arm.fatigue_reset,
            "structuralGatewayInvoked": False,
            "internalGatewayInvoked": False,
            "engineeredStateFieldCount": 0,
            "supplementalStructuralWrites": 0,
            "supplementalNativeWrites": 0,
            "supplementalFatigueWrites": 0,
            "before": prefix.pre_reset,
            "after": prefix.pre_reset,
            "scramble": dict(prefix.scramble),
            "validation": {"unobservableResetRetained": True},
        }
        reset_audit["auditDigest"] = sha256_json(reset_audit)
    episode2 = None
    if prefix.first_stage_eligible and reset_audit["resetFeasible"]:
        episode2 = _run_recovery(
            scenario,
            state,
            router,
            controller,
            budget=recovery_budget,
            trace=trace,
            retain_trace=retain_trace,
        )
    if not prefix.first_stage_eligible:
        cause = "episode1_competing_terminal"
    elif not reset_audit["resetFeasible"]:
        cause = str(reset_audit["resetReason"])
    elif episode2 is not None and episode2["completed"]:
        cause = "episode2_complete"
    elif episode2 is not None and episode2["stopReason"] == "quiescent":
        cause = "episode2_quiescent"
    elif episode2 is not None and episode2["stopReason"] == "phase_event_budget":
        cause = "episode2_phase_event_budget"
    elif episode2 is not None:
        cause = f"episode2_{episode2['stopReason']}"
    else:
        cause = "episode2_missing_internal_error"
    episode2_complete = bool(episode2 and episode2["completed"])
    joint_success = bool(prefix.first_stage_eligible and episode2_complete)
    restricted_time = (
        int(episode2["durationOpportunities"])
        if episode2_complete and episode2 is not None
        else recovery_budget
    )
    ledger_delta = _delta(dict(state.ledger), prefix.initial_ledger)
    stream_delta = _delta(dict(state.stream_counters), prefix.initial_streams)
    process_ledger = controller.process_ledger()
    validations = {
        **ledger_identity(ledger_delta),
        "activationLedgerMatchesClock": ledger_delta["activations"]
        == state.activation_count - checkpoint.activation_count,
        "processOpportunityMatchesClock": process_ledger["chargedOpportunities"]
        == state.activation_count - checkpoint.activation_count,
        "noS11ConstructionStreamAtRuntime": SCRAMBLE_STREAM not in stream_delta,
        "noS04ExogenousProcessStream": all(
            stream_delta.get(item, 0) == 0 for item in PROCESS_STREAMS
        ),
        "fatigueIsStreamFree": all(
            stream_delta.get(item, 0) == 0 for item in PROCESS_STREAMS
        ),
        "fatigueMovementIdentity": process_ledger["fatigueMovementParticipations"]
        == 2 * ledger_delta["acceptedSwaps"],
        "firstBudgetRespected": int(prefix.episode1["durationOpportunities"])
        <= recovery_budget,
        "secondBudgetRespected": episode2 is None
        or int(episode2["durationOpportunities"]) <= recovery_budget,
        "noResetAfterCompetingTerminal": prefix.first_stage_eligible
        or not reset_audit["resetApplicable"],
        "episode2OnlyAfterStabilization": not prefix.first_stage_eligible
        or prefix.stabilization is not None
        and bool(prefix.stabilization["success"]),
        "targetStableRest": prefix.rest is None or bool(prefix.rest["stable"]),
        "preResetSharedPrefixPreserved": prefix.pre_reset
        == reset_audit["before"],
        "resetIsolationPass": all(reset_audit["validation"].values()),
        "phaseLocalEnvelopeUsed": True,
    }
    result = {
        "schemaVersion": STATE_MEMORY_RESET_RUN_SCHEMA_VERSION,
        "benchmarkVersion": BENCHMARK_VERSION,
        "arm": arm.value,
        "isFactorialArm": arm.is_factorial,
        "arrangementResetAssigned": arm.arrangement_reset,
        "nativeStateResetAssigned": arm.native_reset,
        "fatigueResetAssigned": arm.fatigue_reset,
        "sequenceId": sequence_id,
        "operatorId": SEQUENCE_OPERATORS[sequence_id],
        "spacingId": Spacing.REST_20N.value,
        "sourceScenarioId": scenario.scenario_id,
        "sourceCheckpointHash": checkpoint.state_hash,
        "initialState": prefix.initial_state,
        "episode1Lesion": prefix.episode1_lesion,
        "episode1": prefix.episode1,
        "firstStageEligible": prefix.first_stage_eligible,
        "postRecoveryStabilization": prefix.stabilization,
        "rest": prefix.rest,
        "episode2Lesion": prefix.episode2_lesion,
        "episode2InjuryAdministered": prefix.episode2_lesion is not None,
        "preResetState": prefix.pre_reset,
        "resetAudit": reset_audit,
        "resetBoundaryObserved": prefix.first_stage_eligible,
        "resetApplied": bool(prefix.first_stage_eligible and reset_audit["resetFeasible"]),
        "episode2": episode2,
        "episode2RecoveryObserved": episode2 is not None,
        "episode2OutcomeCause": cause,
        "jointSequenceSuccess": joint_success,
        "restrictedEpisode2Time": restricted_time,
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


def run_state_memory_reset_case(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    pairing_id: str,
    timing_condition_id: str,
    sequence_id: str,
    injury_seed: int,
    recovery_budget: int,
    case_address: int,
    retain_trace: bool = False,
) -> dict[str, dict[str, Any]]:
    """Execute the complete ten-arm S11 group from one exact shared prefix."""

    if sequence_id not in SEQUENCE_OPERATORS:
        raise ValueError("unknown S11 repeated sequence")
    prefix = _prepare_prefix(
        scenario,
        checkpoint,
        pairing_id=pairing_id,
        timing_condition_id=timing_condition_id,
        sequence_id=sequence_id,
        injury_seed=injury_seed,
        recovery_budget=recovery_budget,
        case_address=case_address,
        retain_trace=retain_trace,
    )
    return {
        arm.value: _run_arm_from_prefix(
            scenario,
            checkpoint,
            prefix,
            arm=arm,
            sequence_id=sequence_id,
            recovery_budget=recovery_budget,
            retain_trace=retain_trace,
        )
        for arm in ResetArm
    }


def exact_replay_state_memory_reset_case(
    result: Mapping[str, Mapping[str, Any]],
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    pairing_id: str,
    timing_condition_id: str,
    sequence_id: str,
    injury_seed: int,
    recovery_budget: int,
    case_address: int,
    retain_trace: bool = False,
) -> dict[str, dict[str, Any]]:
    replay = run_state_memory_reset_case(
        scenario,
        checkpoint,
        pairing_id=pairing_id,
        timing_condition_id=timing_condition_id,
        sequence_id=sequence_id,
        injury_seed=injury_seed,
        recovery_budget=recovery_budget,
        case_address=case_address,
        retain_trace=retain_trace,
    )
    if canonical_json_bytes(replay) != canonical_json_bytes(result):
        raise AssertionError("S11 state-memory reset replay mismatch")
    return replay
