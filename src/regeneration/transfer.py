"""E05 S12 immutable held-out transfer semantics.

The module composes already-frozen E01/E02 transition, scheduler, S04 process,
S06 rescue, and S09 target-code surfaces.  It deliberately contains no tuning
or controller-selection API: callers supply a fully assigned scenario row.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Any, Mapping, Sequence

import jsonschema

from causal_simulator.action_interface import CommonActionInterface
from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
)
from causal_simulator.schedulers import PERMUTATION_STREAM, SchedulerFamily
from reference_simulator.engine import EMPTY_DIGEST, evaluate_terminal, execute_batch
from reference_simulator.model import (
    Direction,
    FaultMode,
    Policy,
    Proposal,
    ProposalKind,
    RunState,
    Scenario,
    canonical_json_bytes,
    sha256_json,
    state_hash,
)
from reference_simulator.rng import bounded, u64
from reference_simulator.scheduler import ScheduledOpportunity, scheduled_actor, scheduled_side
from reference_simulator.transition_primitives import ledger_identity

from .assisted_rescue import (
    AssistedRescueContract,
    AssistedRescueController,
    RescueArm,
)
from .dynamic_faults import DynamicFaultContract, DynamicProcessController, DynamicProfile
from .target_change import (
    SignalPermission,
    TargetArm,
    TargetChange,
    TargetChangeContract,
    TargetChangeController,
    TargetDefinition,
    build_target_definition,
)
from .tasks import Checkpoint, occupancy_values, strict_unequal_inversions


BENCHMARK_VERSION = "E05-transfer-v1"
TRANSFER_SPEC_SCHEMA_VERSION = "e05.s12.transfer-spec.v1"
TRANSFER_RESULT_SCHEMA_VERSION = "e05.s12.transfer-result.v1"


class TransferTargetChange(str, Enum):
    ADJACENT_PAIR_SWAP = "adjacent_pair_swap_total_order_v1"
    QUARTILE_ROTATION = "quartile_rotation_2_0_3_1_v1"


TRANSFER_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s12/transfer-spec.schema.json",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenAtUtc",
        "implementationStateAtFreeze",
        "frozenQuestion",
        "priorEvidenceConstraints",
        "inherits",
        "immutableSplit",
        "damageContract",
        "goalContract",
        "runtimeProcesses",
        "estimands",
        "inference",
        "validationPanel",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": TRANSFER_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S12"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
    },
}


def validate_transfer_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(TRANSFER_SPEC_SCHEMA).validate(specification)
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "lesionSpecSchemaVersion": "e05.s03.lesion-spec.v1",
        "dynamicSpecSchemaVersion": "e05.s04.dynamic-fault-spec.v1",
        "rescueSpecSchemaVersion": "e05.s06.assisted-rescue-spec.v1",
        "targetChangeSpecSchemaVersion": "e05.s09.target-change-spec.v1",
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "developmentOpportunities": "100*n^2",
        "postInterventionOpportunities": "100*n^2",
        "postHitProbeOpportunities": "20*n",
        "architecture": "distributed_local",
        "continuation": "skip_and_continue",
        "retry": "no_retry",
        "nativeInformationPermission": "policy_native_local",
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"S12 inherited contract changed: {key}")
    split = specification["immutableSplit"]
    holdout = split["heldOutAxes"]
    calibration = split["calibration"]
    if set(calibration["sizes"]) & set(holdout["size"]):
        raise ValueError("calibration and held-out sizes overlap")
    if calibration["scheduler"] in holdout["scheduler"]:
        raise ValueError("calibration scheduler leaked into scheduler holdout")
    if calibration["faultProcess"] in holdout["faultProcess"]:
        raise ValueError("calibration fault process leaked into fault holdout")
    if set(calibration["targetChanges"]) & set(holdout["targetChange"]):
        raise ValueError("calibration and held-out targets overlap")
    panel = specification["validationPanel"]
    if sum(
        value
        for key, value in panel["damageScenarioCounts"].items()
        if key != "total"
    ) != panel["damageScenarioCounts"]["total"]:
        raise ValueError("damage scenario counts do not sum")
    if sum(
        value
        for key, value in panel["goalScenarioCounts"].items()
        if key != "total"
    ) != panel["goalScenarioCounts"]["total"]:
        raise ValueError("goal scenario counts do not sum")
    planned = (
        panel["plannedDamageRuns"]
        + panel["plannedGoalChangeRuns"]
        + panel["plannedStabilityRuns"]
    )
    if planned != panel["plannedRunCount"]:
        raise ValueError("S12 planned run accounting changed")
    if panel["plannedTotalTrajectoryExecutions"] != 2 * planned:
        raise ValueError("S12 exact replay accounting changed")


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def lesion_window_length(n: int) -> int:
    if n < 5:
        raise ValueError("S12 transfer lesions require at least five identities")
    return max(2, _round_half_up(0.4 * n))


def lesion_window(n: int, location: str) -> tuple[int, int]:
    length = lesion_window_length(n)
    if location == "central":
        start = (n - length) // 2
    elif location == "left_off_center":
        start = max(0, (n - length) // 4)
    elif location == "right_off_center":
        start = min(n - length, n - length - (n - length) // 4)
    else:
        raise ValueError(f"unknown transfer lesion location: {location}")
    return start, start + length


def _sattolo_window(
    occupancy: Sequence[str],
    start: int,
    end: int,
    *,
    seed: int,
    scenario_id: str,
) -> tuple[str, ...]:
    result = list(occupancy)
    window = result[start:end]
    draw_cursor = 0
    for index in range(len(window) - 1, 0, -1):
        selected, consumed = bounded(
            seed,
            scenario_id,
            "lesion_local_scramble_s03_v1",
            0,
            index,
            draw_cursor,
        )
        draw_cursor += consumed
        window[index], window[selected] = window[selected], window[index]
    result[start:end] = window
    return tuple(result)


def apply_transfer_lesion(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    lesion_type: str,
    location: str,
) -> dict[str, Any]:
    """Apply an instantaneous identity-conserving S03-compatible lesion."""

    before = tuple(checkpoint.occupancy)
    start, end = lesion_window(len(before), location)
    if lesion_type == "segment_reversal_central_v1":
        after = before[:start] + tuple(reversed(before[start:end])) + before[end:]
    elif lesion_type == "local_scramble_sattolo_v1":
        after = _sattolo_window(
            before,
            start,
            end,
            seed=scenario.seed,
            scenario_id=scenario.scenario_id,
        )
    else:
        raise ValueError(f"unsupported S12 lesion type: {lesion_type}")
    if set(after) != set(before) or len(after) != len(before):
        raise AssertionError("S12 transfer lesion changed identity count")
    before_distance = strict_unequal_inversions(
        occupancy_values(scenario, before), scenario.cells[0].direction
    )
    after_distance = strict_unequal_inversions(
        occupancy_values(scenario, after), scenario.cells[0].direction
    )
    content = {
        "lesionType": lesion_type,
        "location": location,
        "windowStart": start,
        "windowEndExclusive": end,
        "windowLength": end - start,
        "preOccupancy": list(before),
        "postOccupancy": list(after),
        "preDistance": before_distance,
        "postDistance": after_distance,
        "identityConserved": True,
        "activationCountPreserved": True,
        "streamCountersPreserved": True,
        "ledgerPreserved": True,
    }
    return {**content, "lesionStateHash": sha256_json(content)}


class TransferSchedule:
    """State-blind serial schedule starting at an inherited global boundary."""

    __slots__ = ("scenario", "family", "start_event", "actor_ids", "audits")

    def __init__(
        self,
        scenario: Scenario,
        family: SchedulerFamily,
        start_event: int,
    ) -> None:
        if family not in {
            SchedulerFamily.UNIFORM_RANDOM_ACTIVATION,
            SchedulerFamily.DETERMINISTIC_SCAN,
            SchedulerFamily.RANDOM_PERMUTATION_SWEEP,
        }:
            raise ValueError("S12 uses only frozen serial scheduler families")
        self.scenario = scenario
        self.family = family
        self.start_event = start_event
        self.actor_ids = tuple(sorted(scenario.cell_map))
        self.audits: list[dict[str, Any]] = []

    def _permutation(self, sweep: int) -> tuple[tuple[str, ...], tuple[tuple[str, int, int, int], ...]]:
        result = list(self.actor_ids)
        draws: list[tuple[str, int, int, int]] = []
        cursor = 0
        for index in range(len(result) - 1, 0, -1):
            selected, consumed = bounded(
                self.scenario.seed,
                self.scenario.scenario_id,
                PERMUTATION_STREAM,
                sweep,
                index + 1,
                cursor,
            )
            for draw_index in range(cursor, cursor + consumed):
                draws.append(
                    (
                        PERMUTATION_STREAM,
                        sweep,
                        draw_index,
                        u64(
                            self.scenario.seed,
                            self.scenario.scenario_id,
                            PERMUTATION_STREAM,
                            sweep,
                            draw_index,
                        ),
                    )
                )
            cursor += consumed
            result[index], result[selected] = result[selected], result[index]
        return tuple(result), tuple(draws)

    def __call__(
        self, event_index: int, remaining_opportunities: int
    ) -> tuple[ScheduledOpportunity, ...]:
        if remaining_opportunities < 1:
            return ()
        local = event_index - self.start_event
        if local != len(self.audits):
            raise ValueError("S12 schedule requested outside its contiguous phase")
        draws: tuple[tuple[str, int, int, int], ...] = ()
        consumption: tuple[tuple[str, int], ...] = ()
        if self.family == SchedulerFamily.UNIFORM_RANDOM_ACTIVATION:
            actor, draws, consumed = scheduled_actor(
                self.scenario, event_index, include_draws=True
            )
            consumption = (("actor_activation", consumed),)
            reason = "uniform_with_replacement_global_event_clock"
        elif self.family == SchedulerFamily.DETERMINISTIC_SCAN:
            actor = self.actor_ids[local % len(self.actor_ids)]
            reason = "canonical_scan_phase_local_index"
        else:
            sweep, offset = divmod(local, len(self.actor_ids))
            permutation, sweep_draws = self._permutation(sweep)
            actor = permutation[offset]
            if offset == 0:
                draws = sweep_draws
                consumption = ((PERMUTATION_STREAM, len(draws)),) if draws else ()
            reason = "counter_addressed_phase_local_permutation"
        self.audits.append(
            {
                "localOpportunityIndex": local,
                "globalEventIndex": event_index,
                "actorId": actor,
                "family": self.family.value,
                "reason": reason,
                "drawCount": len(draws),
            }
        )
        return (ScheduledOpportunity(actor, draws, consumption),)


def _old_rank_codes(scenario: Scenario) -> dict[str, int]:
    direction = scenario.cells[0].direction
    ordered = sorted(
        scenario.cells,
        key=lambda cell: cell.value,
        reverse=direction == Direction.DESCENDING,
    )
    return {cell.cell_id: rank for rank, cell in enumerate(ordered)}


def _maximum_distance(codes: Mapping[str, int]) -> int:
    counts = Counter(codes.values())
    ordered = sorted(counts)
    return sum(
        counts[left] * counts[right]
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    )


def build_transfer_target_definition(
    scenario: Scenario,
    target_change: TargetChange | TransferTargetChange,
    *,
    no_change: bool = False,
) -> TargetDefinition:
    if isinstance(target_change, TargetChange):
        return build_target_definition(scenario, target_change, no_change=no_change)
    old = _old_rank_codes(scenario)
    n = len(old)
    if no_change:
        target = dict(old)
    elif target_change == TransferTargetChange.ADJACENT_PAIR_SWAP:
        target = {
            identity: (rank + 1 if rank % 2 == 0 and rank + 1 < n else rank - 1)
            if rank % 2 or rank + 1 < n
            else rank
            for identity, rank in old.items()
        }
    elif target_change == TransferTargetChange.QUARTILE_ROTATION:
        rotation = {0: 2, 1: 0, 2: 3, 3: 1}
        target = {
            identity: rotation[min(3, (4 * rank) // n)]
            for identity, rank in old.items()
        }
    else:  # pragma: no cover - exhaustive Enum guard
        raise AssertionError(target_change)
    direction = scenario.cells[0].direction
    sign = 1 if direction == Direction.ASCENDING else -1
    policy = {identity: sign * code for identity, code in target.items()}
    maximum = _maximum_distance(target)
    if maximum < 1:
        raise ValueError("S12 target lacks unequal-pair support")
    target_hash = sha256_json(
        {
            "scenarioId": scenario.scenario_id,
            "targetChangeId": target_change.value,
            "noChange": no_change,
            "oldCodes": old,
            "targetCodes": target,
            "policyCodes": policy,
        }
    )
    return TargetDefinition(
        target_change,  # type: ignore[arg-type]
        no_change,
        tuple(sorted(old.items())),
        tuple(sorted(target.items())),
        tuple(sorted(policy.items())),
        maximum,
        target_hash,
    )


def transfer_target_correspondence_rows(
    scenario: Scenario, definition: TargetDefinition
) -> list[dict[str, Any]]:
    codes = definition.target_code_map
    old = definition.old_code_map
    counts = Counter(codes.values())
    starts: dict[int, int] = {}
    cursor = 0
    for code in sorted(counts):
        starts[code] = cursor
        cursor += counts[code]
    return [
        {
            "scenarioId": scenario.scenario_id,
            "targetChangeId": definition.target_change.value,
            "targetHash": definition.target_hash,
            "identityId": identity,
            "oldTargetRank": old[identity],
            "newTargetCode": codes[identity],
            "policyCode": definition.policy_code_map[identity],
            "allowedPositionStart": starts[codes[identity]],
            "allowedPositionEnd": starts[codes[identity]] + counts[codes[identity]] - 1,
            "identityConserved": True,
            "targetFeasible": True,
        }
        for identity in sorted(codes)
    ]


def _dynamic_controller(
    scenario: Scenario,
    process_id: str,
    selected_identity: str,
    start_event: int,
) -> DynamicProcessController | None:
    mapping = {
        "none": None,
        DynamicProfile.INTERMITTENT.value: DynamicProfile.INTERMITTENT,
        DynamicProfile.FATIGUE.value: DynamicProfile.FATIGUE,
    }
    if process_id not in mapping:
        raise ValueError(f"unknown S12 runtime process: {process_id}")
    profile = mapping[process_id]
    if profile is None:
        return None
    return DynamicProcessController(
        DynamicFaultContract(profile),
        scenario,
        selected_identity,
        start_event,
        retain_audits=False,
    )


class DamageCompositeInterceptor:
    __slots__ = ("dynamic", "rescue")

    def __init__(
        self,
        dynamic: DynamicProcessController | None,
        rescue: AssistedRescueController,
    ) -> None:
        self.dynamic = dynamic
        self.rescue = rescue

    def prepare(self, proposal: Proposal, event_index: int):
        consumption: list[tuple[str, int]] = []
        if self.dynamic is not None:
            proposal, dynamic_consumption = self.dynamic.prepare(proposal, event_index)
            consumption.extend(dynamic_consumption)
        proposal, rescue_consumption = self.rescue.prepare(proposal, event_index)
        consumption.extend(rescue_consumption)
        return proposal, tuple(consumption)

    def outcome(self, proposal: Proposal, validation: Any, event_index: int):
        if self.dynamic is not None:
            validation = self.dynamic.outcome(proposal, validation, event_index)
        return self.rescue.outcome(proposal, validation, event_index)

    def after_batch(
        self,
        scenario: Scenario,
        state: RunState,
        proposals: tuple[Proposal, ...],
        decisions: Mapping[int, str],
        batch_start_index: int,
    ) -> None:
        if self.dynamic is not None:
            self.dynamic.after_batch(
                scenario, state, proposals, decisions, batch_start_index
            )
        self.rescue.after_batch(
            scenario, state, proposals, decisions, batch_start_index
        )


class TargetCompositeGateway:
    __slots__ = ("target", "dynamic", "pending", "last_base_eligible")

    def __init__(
        self,
        target: TargetChangeController,
        dynamic: DynamicProcessController | None,
    ) -> None:
        self.target = target
        self.dynamic = dynamic
        self.pending: dict[str, Any] | None = None
        self.last_base_eligible = False

    def proposal_for(
        self,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        side: str | None = None,
    ) -> Proposal:
        del scenario
        if self.pending is not None:
            raise AssertionError("S12 target gateway retained a prior opportunity")
        surface = self.target.proposal_for(state, actor_id, side=side)
        self.pending = {
            "surface": surface,
            "actorId": actor_id,
            "actorPosition": state.occupancy.index(actor_id),
            "side": side,
        }
        return surface.emitted

    def prepare(self, proposal: Proposal, event_index: int):
        if self.dynamic is None:
            return proposal, ()
        return self.dynamic.prepare(proposal, event_index)

    def outcome(self, proposal: Proposal, validation: Any, event_index: int):
        # Preserve the authoritative mechanical decision before an S04 process
        # may only turn an eligible movement off.  Intermittence and fatigue can
        # never create a proposal or make a mechanically ineligible proposal
        # eligible, so this is the sound quiescence certificate surface.
        self.last_base_eligible = bool(validation.eligible_for_commit)
        if self.dynamic is None:
            return validation
        return self.dynamic.outcome(proposal, validation, event_index)

    def after_batch(
        self,
        scenario: Scenario,
        state: RunState,
        proposals: tuple[Proposal, ...],
        decisions: Mapping[int, str],
        batch_start_index: int,
    ) -> None:
        if len(proposals) != 1 or self.pending is None:
            raise AssertionError("S12 target gateway is serial")
        if self.dynamic is not None:
            self.dynamic.after_batch(
                scenario, state, proposals, decisions, batch_start_index
            )
        proposal = proposals[0]
        decision = decisions[proposal.ordinal]
        self.target.record_outcome(
            batch_start_index,
            self.pending["actorId"],
            self.pending["actorPosition"],
            self.pending["side"],
            self.pending["surface"],
            decision,
            decision == "accepted",
        )
        self.pending = None


def _delta(final: Mapping[str, int], initial: Mapping[str, int]) -> dict[str, int]:
    return {key: int(final.get(key, 0)) - int(initial.get(key, 0)) for key in final}


def _digest_result(result: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(result)).hexdigest()


def run_damage_transfer(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    lesion: Mapping[str, Any],
    scheduler: SchedulerFamily,
    process_id: str,
    arm: RescueArm,
    assigned_duration: int | None,
    recovery_budget: int,
    retain_trace: bool = False,
) -> dict[str, Any]:
    if arm not in {RescueArm.ACTIVE, RescueArm.MATCHED_COST}:
        raise ValueError("S12 damage transfer freezes active and matched-cost arms")
    if (arm == RescueArm.MATCHED_COST) != (assigned_duration is not None):
        raise ValueError("only the S12 matched arm has an assigned duration")
    occupancy = tuple(lesion["postOccupancy"])
    state = checkpoint.to_run_state(occupancy=occupancy)
    start = state.activation_count
    selected = occupancy[(len(occupancy) - 1) // 2]
    rescue_contract = AssistedRescueContract(arm, assigned_duration=assigned_duration)
    rescue = AssistedRescueController(
        rescue_contract,
        scenario,
        selected,
        state.occupancy.index(selected),
        start,
        retain_audits=False,
    )
    dynamic = _dynamic_controller(scenario, process_id, selected, start)
    interceptor = DamageCompositeInterceptor(dynamic, rescue)
    schedule = TransferSchedule(scenario, scheduler, start)
    router = ArchitectureProposalRouter(
        ArchitectureExecutionContract.distributed_local(),
        action_interface=CommonActionInterface(),
    )
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    initial_hash = state_hash(scenario.scenario_id, state)
    initial_distance = strict_unequal_inversions(
        occupancy_values(scenario, state.occupancy), scenario.cells[0].direction
    )
    distance = initial_distance
    distance_auc = 0
    trace: list[Mapping[str, Any]] = []
    digest = bytes.fromhex(EMPTY_DIGEST)
    state.terminal = evaluate_terminal(scenario, state)
    while state.terminal is None and state.activation_count - start < recovery_budget:
        events, encoded = execute_batch(
            scenario,
            state,
            retain_events=retain_trace,
            emit_event_records=retain_trace,
            proposal_factory=router.proposal_for,
            schedule_factory=schedule,
            execution_interceptor=interceptor,
        )
        trace.extend(events)
        for item in encoded:
            digest = hashlib.sha256(digest + item).digest()
        if state.ledger["acceptedSwaps"] != initial_ledger["acceptedSwaps"]:
            distance = strict_unequal_inversions(
                occupancy_values(scenario, state.occupancy),
                scenario.cells[0].direction,
            )
        distance_auc += distance
    if state.terminal is None:
        state.terminal = "phase_event_budget"
    phase = state.activation_count - start
    native = _delta(dict(state.ledger), initial_ledger)
    streams = _delta(dict(state.stream_counters), initial_streams)
    rescue_ledger = rescue.process_ledger()
    dynamic_ledger = {} if dynamic is None else dynamic.process_ledger()
    result: dict[str, Any] = {
        "schemaVersion": TRANSFER_RESULT_SCHEMA_VERSION,
        "benchmarkVersion": BENCHMARK_VERSION,
        "mechanismId": "s06_active_local_repair_v1",
        "arm": arm.value,
        "sourceScenarioId": scenario.scenario_id,
        "sourceCheckpointHash": checkpoint.state_hash,
        "lesionStateHash": lesion["lesionStateHash"],
        "scheduler": scheduler.value,
        "processId": process_id,
        "selectedIdentityId": selected,
        "assignedDuration": assigned_duration,
        "startEventIndex": start,
        "endEventIndex": state.activation_count,
        "initialStateHash": initial_hash,
        "finalStateHash": state_hash(scenario.scenario_id, state),
        "stopReason": state.terminal,
        "success": state.terminal == "complete",
        "phaseActivationCount": phase,
        "restrictedTime": phase if state.terminal == "complete" else recovery_budget + 1,
        "recoveryBudget": recovery_budget,
        "initialDistance": initial_distance,
        "finalDistance": distance,
        "distanceAuc": distance_auc,
        "recoveryObserved": rescue.recovery_event is not None,
        "recoveryDuration": None if rescue.recovery_event is None else rescue.recovery_event - start,
        "recoveryCensored": rescue.recovery_event is None,
        "nativeLedgerDelta": native,
        "streamCounterDelta": streams,
        "rescueLedger": rescue_ledger,
        "dynamicLedger": dynamic_ledger,
        "rescueState": rescue.state_dict(),
        "dynamicState": None if dynamic is None else dynamic.state_dict(),
        "rescueAuditDigest": rescue.audit_digest.hex(),
        "dynamicAuditDigest": None if dynamic is None else dynamic.audit_digest.hex(),
        "schedulerAuditDigest": sha256_json(schedule.audits),
        "schedulerAuditCount": len(schedule.audits),
        "eventDigest": digest.hex() if retain_trace else EMPTY_DIGEST,
        "events": trace,
        "validation": {
            **ledger_identity(native),
            "phaseBudgetRespected": phase <= recovery_budget,
            "schedulerAuditCountMatchesPhase": len(schedule.audits) == phase,
            "rescueOpportunityCountMatchesPhase": rescue_ledger["chargedOpportunities"] == phase,
            "dynamicOpportunityCountMatchesPhase": dynamic is None or dynamic_ledger["chargedOpportunities"] == phase,
            "actionBudgetConserved": rescue_ledger["actionUnitsSpent"] + rescue.action_remaining == 1,
            "energyBudgetConserved": rescue_ledger["energyUnitsSpent"] + rescue.energy_remaining == 1,
            "atMostOneIntervention": rescue_ledger["nativeOpportunitiesSuppressed"] <= 1,
            "identityCountPreserved": len(state.occupancy) == len(scenario.cells),
        },
    }
    result["resultDigest"] = _digest_result(result)
    return result


def _expected_contexts(scenario: Scenario) -> set[tuple[str, str | None]]:
    answer: set[tuple[str, str | None]] = set()
    for cell in scenario.cells:
        if cell.policy == Policy.BUBBLE and cell.fault == FaultMode.NORMAL:
            answer.add((cell.cell_id, "left"))
            answer.add((cell.cell_id, "right"))
        else:
            answer.add((cell.cell_id, None))
    return answer


def run_goal_transfer(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    target_change: TargetChange | TransferTargetChange,
    scheduler: SchedulerFamily,
    process_id: str,
    target_aware: bool,
    adaptation_budget: int,
    probe_budget: int,
    stability: bool = False,
    retain_trace: bool = False,
) -> dict[str, Any]:
    state = checkpoint.to_run_state()
    state.terminal = None
    start = state.activation_count
    definition = build_transfer_target_definition(
        scenario, target_change, no_change=stability
    )
    arm = (
        TargetArm.STABILITY_AWARE
        if stability and target_aware
        else TargetArm.STABILITY_NONADAPTIVE
        if stability
        else TargetArm.CHANGED_AWARE
        if target_aware
        else TargetArm.CHANGED_NONADAPTIVE
    )
    contract = TargetChangeContract(arm, target_change, SignalPermission.GRADIENT)  # type: ignore[arg-type]
    contract.validate(scenario)
    target = TargetChangeController(
        contract, definition, scenario, start, retain_audits=False
    )
    selected = state.occupancy[(len(state.occupancy) - 1) // 2]
    dynamic = _dynamic_controller(scenario, process_id, selected, start)
    gateway = TargetCompositeGateway(target, dynamic)
    schedule = TransferSchedule(scenario, scheduler, start)
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    initial_hash = state_hash(scenario.scenario_id, state)
    old_distance = definition.old_distance(state.occupancy)
    new_distance = definition.distance(state.occupancy)
    initial_old_distance = old_distance
    initial_new_distance = new_distance
    max_new_distance = new_distance
    new_auc = 0
    old_auc = 0
    hit_time: int | None = 0 if not stability and new_distance == 0 else None
    post_hit = 0
    post_hit_departure = False
    stability_departure = False
    coverage: set[tuple[str, str | None]] = set()
    expected = _expected_contexts(scenario)
    stop_reason: str | None = None
    trace: list[Mapping[str, Any]] = []
    digest = bytes.fromhex(EMPTY_DIGEST)
    run_limit = probe_budget if stability else adaptation_budget + probe_budget
    while state.activation_count - start < run_limit:
        if not stability and hit_time is not None and post_hit >= probe_budget:
            stop_reason = "target_hit_probe_complete"
            break
        state.terminal = None
        before_occupancy = tuple(state.occupancy)
        before_cursors = dict(state.selection_cursors)
        events, encoded = execute_batch(
            scenario,
            state,
            retain_events=retain_trace,
            emit_event_records=retain_trace,
            proposal_factory=gateway.proposal_for,
            schedule_factory=schedule,
            execution_interceptor=gateway,
        )
        trace.extend(events)
        for item in encoded:
            digest = hashlib.sha256(digest + item).digest()
        audit = schedule.audits[-1]
        actor = scenario.cell_map[audit["actorId"]]
        side: str | None = None
        if actor.policy == Policy.BUBBLE:
            side, _ = scheduled_side(scenario, audit["globalEventIndex"])
        changed = (
            tuple(state.occupancy) != before_occupancy
            or state.selection_cursors != before_cursors
        )
        if changed:
            old_distance = definition.old_distance(state.occupancy)
            new_distance = definition.distance(state.occupancy)
            coverage.clear()
        elif not gateway.last_base_eligible:
            coverage.add((audit["actorId"], side))
        elapsed = state.activation_count - start
        old_auc += old_distance
        new_auc += new_distance
        max_new_distance = max(max_new_distance, new_distance)
        stability_departure = stability_departure or (stability and new_distance > 0)
        hit_now = False
        if not stability and hit_time is None and new_distance == 0:
            hit_time = elapsed
            hit_now = True
        if not stability and hit_time is not None and not hit_now:
            post_hit += 1
            post_hit_departure = post_hit_departure or new_distance > 0
        if (
            not stability
            and hit_time is None
            and coverage == expected
        ):
            stop_reason = "target_quiescent"
            break
    if stop_reason is None:
        if stability:
            stop_reason = "stability_probe_complete"
        elif hit_time is not None:
            stop_reason = "target_hit_probe_complete"
        else:
            stop_reason = "phase_event_budget"
    phase = state.activation_count - start
    native = _delta(dict(state.ledger), initial_ledger)
    streams = _delta(dict(state.stream_counters), initial_streams)
    target_ledger = target.process_ledger()
    dynamic_ledger = {} if dynamic is None else dynamic.process_ledger()
    result: dict[str, Any] = {
        "schemaVersion": TRANSFER_RESULT_SCHEMA_VERSION,
        "benchmarkVersion": BENCHMARK_VERSION,
        "mechanismId": "s09_gradient_target_code_controller_v1",
        "arm": arm.value,
        "targetAware": target_aware,
        "stability": stability,
        "targetChangeId": target_change.value,
        "targetHash": definition.target_hash,
        "targetMaximumDistance": definition.maximum_distance,
        "sourceScenarioId": scenario.scenario_id,
        "sourceCheckpointHash": checkpoint.state_hash,
        "scheduler": scheduler.value,
        "processId": process_id,
        "startEventIndex": start,
        "endEventIndex": state.activation_count,
        "initialStateHash": initial_hash,
        "finalStateHash": state_hash(scenario.scenario_id, state),
        "stopReason": stop_reason,
        "success": (new_distance == 0) if stability else hit_time is not None,
        "targetCompleted": hit_time is not None,
        "phaseActivationCount": phase,
        "adaptationBudget": adaptation_budget,
        "probeBudget": probe_budget,
        "adaptationTime": hit_time,
        "adaptationCensored": not stability and hit_time is None,
        "restrictedTime": hit_time if hit_time is not None else adaptation_budget + 1,
        "initialOldTargetDistance": initial_old_distance,
        "initialNewTargetDistance": initial_new_distance,
        "finalOldTargetDistance": old_distance,
        "finalNewTargetDistance": new_distance,
        "initialNormalizedNewTargetDistance": initial_new_distance / definition.maximum_distance,
        "finalNormalizedNewTargetDistance": new_distance / definition.maximum_distance,
        "maximumNewTargetDistance": max_new_distance,
        "oldTargetDistanceAuc": old_auc,
        "newTargetDistanceAuc": new_auc,
        "postHitProbeOpportunities": post_hit,
        "postHitAnyDeparture": post_hit_departure,
        "noChangeAnyTargetDeparture": stability_departure,
        "noChangeFinalTargetRetained": stability and new_distance == 0,
        "nativeLedgerDelta": native,
        "streamCounterDelta": streams,
        "targetLedger": target_ledger,
        "dynamicLedger": dynamic_ledger,
        "targetAuditDigest": target.audit_digest.hex(),
        "dynamicAuditDigest": None if dynamic is None else dynamic.audit_digest.hex(),
        "schedulerAuditDigest": sha256_json(schedule.audits),
        "schedulerAuditCount": len(schedule.audits),
        "eventDigest": digest.hex() if retain_trace else EMPTY_DIGEST,
        "events": trace,
        "validation": {
            **ledger_identity(native),
            "phaseBudgetRespected": phase <= run_limit,
            "schedulerAuditCountMatchesPhase": len(schedule.audits) == phase,
            "targetOpportunityCountMatchesPhase": target_ledger["chargedOpportunities"] == phase,
            "dynamicOpportunityCountMatchesPhase": dynamic is None or dynamic_ledger["chargedOpportunities"] == phase,
            "oneProposalPerOpportunity": native["proposals"] == phase,
            "signalCostIdentity": target_ledger["signalQueries"] == phase,
            "targetFeasible": definition.maximum_distance > 0,
            "identityCountPreserved": len(state.occupancy) == len(scenario.cells),
            "stabilityBudgetExact": not stability or phase == probe_budget,
        },
    }
    result["resultDigest"] = _digest_result(result)
    return result


def exact_replay_result(run_factory: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    replay = run_factory()
    if canonical_json_bytes(replay) != canonical_json_bytes(expected):
        raise AssertionError("S12 exact replay mismatch")
    return replay
