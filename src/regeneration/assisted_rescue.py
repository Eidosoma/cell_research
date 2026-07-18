"""E05 S06 costed neighbor-assisted rescue over the S04/S05 runtime overlay.

The native cell-view policy always constructs exactly one ordinary proposal.
An explicitly engineered, local intervention gateway may replace that already
charged proposal with one repair action.  The gateway is stream-free, has a
one-action/one-energy budget, and does not alter native policy observations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Mapping, Sequence

import jsonschema

from causal_simulator.action_interface import CommonActionInterface
from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
)
from reference_simulator.engine import EMPTY_DIGEST, evaluate_terminal, execute_batch
from reference_simulator.model import (
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
from reference_simulator.scheduler import (
    ScheduledOpportunity,
    scheduled_actor,
    scheduled_side,
)
from reference_simulator.transition_primitives import (
    ValidationDecision,
    commit_proposal,
    cost_delta,
    ledger_identity,
    validate_proposal,
)

from .tasks import Checkpoint, occupancy_values, strict_unequal_inversions


BENCHMARK_VERSION = "E05-neighbor-assisted-rescue-v1"
ASSISTED_SPEC_SCHEMA_VERSION = "e05.s06.assisted-rescue-spec.v1"
ASSISTED_RUN_SCHEMA_VERSION = "e05.s06.assisted-rescue-run.v1"
CONTROL_ASSIGNMENT_STREAM = "assisted_rescue_control_assignment_s06_v1"


class RescueArm(str, Enum):
    ACTIVE = "active_assisted_rescue"
    PASSIVE = "passive_permanent_freeze"
    SPONTANEOUS = "spontaneous_time_matched"
    SHAM = "sham_local_repair"
    MATCHED_COST = "matched_time_opportunity_energy"


class MatchingChoice(str, Enum):
    PRIMARY_N_POLICY = "empirical_bijection_n_policy_v1"
    SENSITIVITY_N = "empirical_bijection_n_v1"
    SENSITIVITY_POOLED = "empirical_bijection_pooled_v1"


ASSISTED_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s06/assisted-rescue-spec.schema.json",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenAtUtc",
        "frozenQuestion",
        "inherits",
        "repairProposal",
        "costsAndBudgets",
        "controls",
        "matching",
        "tradeoffMetrics",
        "validationPanel",
        "successCriteria",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": ASSISTED_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S06"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "controls": {"type": "array", "minItems": 4, "maxItems": 4},
        "matching": {"type": "object"},
        "validationPanel": {"type": "object"},
    },
}


def validate_assisted_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(ASSISTED_SPEC_SCHEMA).validate(specification)
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "lesionSpecSchemaVersion": "e05.s03.lesion-spec.v1",
        "dynamicSpecSchemaVersion": "e05.s04.dynamic-fault-spec.v1",
        "nudgeSpecSchemaVersion": "e05.s05.nudge-recovery-spec.v1",
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "runtimeAnchorLesion": "segment_reversal_central_v1",
        "runtimeScenarioBoundary": "original_s01_scenario_id_runtime_overlay",
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"inherited S01-S05 contract changed: {key}")
    proposal = specification["repairProposal"]
    if proposal.get("proposalId") != "local_adjacent_repair_once_v1":
        raise ValueError("S06 repair-proposal semantics changed")
    if proposal.get("attemptLimit") != 1 or proposal.get("runtimeStreams") != []:
        raise ValueError("S06 proposal must remain one-shot and stream-free")
    costs = specification["costsAndBudgets"]
    for key in (
        "actionCostPerProposal",
        "energyCostPerProposal",
        "collectiveRepairActionBudget",
        "collectiveEnergyBudget",
        "perHelperEnergyBudget",
    ):
        if costs.get(key) != 1:
            raise ValueError(f"S06 minimal unit budget changed: {key}")
    control_arms = {item["arm"] for item in specification["controls"]}
    if control_arms != {
        RescueArm.PASSIVE.value,
        RescueArm.SPONTANEOUS.value,
        RescueArm.SHAM.value,
        RescueArm.MATCHED_COST.value,
    }:
        raise ValueError("S06 controls changed")
    matching = specification["matching"]
    if matching.get("calibrationReplicates") != [0, 1]:
        raise ValueError("S06 calibration split changed")
    if matching.get("confirmatoryReplicates") != [2, 3]:
        raise ValueError("S06 confirmatory split changed")
    if set(matching.get("choices", [])) != {item.value for item in MatchingChoice}:
        raise ValueError("S06 matching choices changed")
    panel = specification["validationPanel"]
    if panel.get("plannedCheckpointCount") != 384:
        raise ValueError("S06 must retain all 384 checkpoints")
    if panel.get("plannedRunCount") != 1920:
        raise ValueError("S06 planned run count changed")
    if specification.get("claimBoundary", "").startswith("S06 tests an engineered") is False:
        raise ValueError("S06 engineered-intervention claim boundary is missing")


@dataclass(frozen=True, slots=True)
class AssistedRescueContract:
    arm: RescueArm
    assigned_duration: int | None = None
    action_budget: int = 1
    energy_budget: int = 1
    per_helper_energy_budget: int = 1
    action_cost: int = 1
    energy_cost: int = 1
    information_permission: str = "policy_native_local"
    native_legal_primitives: tuple[str, ...] = ("NoOp", "Swap", "MemoryUpdate")
    scheduler: str = "uniform_random_activation"
    continuation: str = "skip_and_continue"
    retry: str = "no_retry"

    def validate(self, scenario: Scenario, selected_identity: str) -> None:
        ArchitectureExecutionContract.distributed_local().validate_policy_action_boundary(
            scenario
        )
        if selected_identity not in scenario.cell_map:
            raise ValueError("S06 selected an unknown identity")
        if any(cell.fault != FaultMode.NORMAL for cell in scenario.cells):
            raise ValueError("S06 requires the original normal S01 scenario")
        if self.information_permission != "policy_native_local":
            raise ValueError("S06 cannot change native policy information")
        if self.native_legal_primitives != ("NoOp", "Swap", "MemoryUpdate"):
            raise ValueError("S06 cannot change native legal primitives")
        if self.scheduler != "uniform_random_activation":
            raise ValueError("S06 preserves uniform random activation")
        if self.continuation != "skip_and_continue" or self.retry != "no_retry":
            raise ValueError("S06 preserves continuation and no-retry")
        if (
            self.action_budget,
            self.energy_budget,
            self.per_helper_energy_budget,
            self.action_cost,
            self.energy_cost,
        ) != (1, 1, 1, 1, 1):
            raise ValueError("S06 unit costs and budgets are frozen")
        if self.arm in {RescueArm.SPONTANEOUS, RescueArm.MATCHED_COST}:
            if self.assigned_duration is not None and self.assigned_duration < 1:
                raise ValueError("a finite assigned duration must be positive")
        elif self.assigned_duration is not None:
            raise ValueError("only scheduled controls may have assigned durations")

    @property
    def owns_runtime_streams(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "assignedDurationOpportunities": self.assigned_duration,
            "collectiveActionBudget": self.action_budget,
            "collectiveEnergyBudget": self.energy_budget,
            "perHelperEnergyBudget": self.per_helper_energy_budget,
            "actionCost": self.action_cost,
            "energyCost": self.energy_cost,
            "energyInterpretation": "abstract_intervention_accounting_unit",
            "runtimeStreams": list(self.owns_runtime_streams),
            "nativeInformationPermission": self.information_permission,
            "nativeLegalPrimitives": list(self.native_legal_primitives),
            "scheduler": self.scheduler,
            "continuation": self.continuation,
            "retry": self.retry,
        }


class AssistedRescueController:
    """Frozen gate, one-shot local intervention, and supplementary cost ledger."""

    __slots__ = (
        "contract",
        "scenario",
        "selected_identity",
        "selected_position",
        "start_event_index",
        "frozen",
        "recovery_event",
        "recovery_reason",
        "recovery_pending",
        "intervention_pending",
        "intervention_kind",
        "intervention_event",
        "intervention_actor",
        "intervention_native_kind",
        "intervention_native_eligible",
        "action_remaining",
        "energy_remaining",
        "helper_energy_spent",
        "ledger",
        "audit_digest",
        "audit_count",
        "retained_audits",
        "retain_audits",
        "transitions",
        "proposal_kinds",
    )

    def __init__(
        self,
        contract: AssistedRescueContract,
        scenario: Scenario,
        selected_identity: str,
        selected_position: int,
        start_event_index: int,
        *,
        retain_audits: bool = False,
    ) -> None:
        contract.validate(scenario, selected_identity)
        self.contract = contract
        self.scenario = scenario
        self.selected_identity = selected_identity
        self.selected_position = selected_position
        self.start_event_index = start_event_index
        self.frozen = True
        self.recovery_event: int | None = None
        self.recovery_reason: str | None = None
        self.recovery_pending = False
        self.intervention_pending = False
        self.intervention_kind: str | None = None
        self.intervention_event: int | None = None
        self.intervention_actor: str | None = None
        self.intervention_native_kind: str | None = None
        self.intervention_native_eligible: bool | None = None
        self.action_remaining = contract.action_budget
        self.energy_remaining = contract.energy_budget
        self.helper_energy_spent: Counter[str] = Counter()
        self.ledger: Counter[str] = Counter()
        self.audit_digest = hashlib.sha256(b"E05/S06/assisted-rescue-audit/v1").digest()
        self.audit_count = 0
        self.retained_audits: list[dict[str, Any]] = []
        self.retain_audits = retain_audits
        self.transitions: list[dict[str, Any]] = []
        self.proposal_kinds: set[str] = set()

    def _audit(self, kind: str, content: Mapping[str, Any]) -> None:
        record = {"kind": kind, **content}
        self.audit_digest = hashlib.sha256(
            self.audit_digest + canonical_json_bytes(record)
        ).digest()
        self.audit_count += 1
        if self.retain_audits:
            self.retained_audits.append(record)

    def _transition(self, kind: str, event_index: int, **content: Any) -> None:
        record = {"transition": kind, "eventIndex": event_index, **content}
        self.transitions.append(record)
        self._audit("transition", record)

    def _recover(self, event_index: int, reason: str, duration: int) -> None:
        if not self.frozen:
            raise AssertionError("S06 recovery attempted twice")
        self.frozen = False
        self.recovery_event = event_index
        self.recovery_reason = reason
        self.ledger["recoveries"] += 1
        self._transition(
            "selected_identity_recovered",
            event_index,
            reason=reason,
            durationOpportunities=duration,
        )

    def _local_helper_eligible(self, proposal: Proposal) -> bool:
        if not self.frozen or proposal.actor_id == self.selected_identity:
            return False
        actor = self.scenario.cell_map[proposal.actor_id]
        return (
            actor.fault == FaultMode.NORMAL
            and abs(proposal.actor_pos - self.selected_position) == 1
            and self.action_remaining >= self.contract.action_cost
            and self.energy_remaining >= self.contract.energy_cost
            and self.helper_energy_spent[proposal.actor_id]
            + self.contract.energy_cost
            <= self.contract.per_helper_energy_budget
        )

    def prepare(
        self, proposal: Proposal, event_index: int
    ) -> tuple[Proposal, tuple[tuple[str, int], ...]]:
        self.ledger["chargedOpportunities"] += 1
        self.proposal_kinds.add(proposal.kind.value)
        self.recovery_pending = False
        self.intervention_pending = False
        self.intervention_kind = None

        if (
            self.contract.arm == RescueArm.SPONTANEOUS
            and self.frozen
            and self.contract.assigned_duration is not None
            and event_index >= self.start_event_index + self.contract.assigned_duration
        ):
            self.ledger["spontaneousScheduleEvents"] += 1
            self._recover(
                event_index,
                "calibration_time_matched_spontaneous",
                self.contract.assigned_duration,
            )

        if self.frozen:
            self.ledger["freezeExposures"] += 1

        if self.contract.arm in {RescueArm.ACTIVE, RescueArm.SHAM}:
            adjacent = (
                proposal.actor_id != self.selected_identity
                and abs(proposal.actor_pos - self.selected_position) == 1
            )
            self.ledger["localAdjacencySignals"] += int(adjacent and self.frozen)
            if self._local_helper_eligible(proposal):
                self.intervention_pending = True
                self.intervention_kind = "local_repair"
        elif (
            self.contract.arm == RescueArm.MATCHED_COST
            and self.frozen
            and self.contract.assigned_duration is not None
            and event_index
            == self.start_event_index + self.contract.assigned_duration - 1
        ):
            self.intervention_pending = True
            self.intervention_kind = "matched_cost"
        return proposal, ()

    def _spend(
        self,
        proposal: Proposal,
        validation: ValidationDecision,
        event_index: int,
    ) -> None:
        if self.action_remaining < 1 or self.energy_remaining < 1:
            raise AssertionError("S06 intervention overspent its collective budget")
        self.action_remaining -= 1
        self.energy_remaining -= 1
        self.helper_energy_spent[proposal.actor_id] += 1
        self.intervention_event = event_index
        self.intervention_actor = proposal.actor_id
        self.intervention_native_kind = proposal.kind.value
        self.intervention_native_eligible = validation.eligible_for_commit
        self.ledger["actionUnitsSpent"] += 1
        self.ledger["energyUnitsSpent"] += 1
        self.ledger["nativeOpportunitiesSuppressed"] += 1
        self.ledger["forgoneEligibleNativeChanges"] += int(
            validation.eligible_for_commit
        )
        self.ledger[f"forgoneNative{proposal.kind.value}"] += 1

    def outcome(
        self,
        proposal: Proposal,
        validation: ValidationDecision,
        event_index: int,
    ) -> ValidationDecision:
        if self.intervention_pending:
            self._spend(proposal, validation, event_index)
            if self.intervention_kind == "local_repair":
                self.ledger["repairProposals"] += 1
                success = self.contract.arm == RescueArm.ACTIVE
                self.ledger["repairSuccesses"] += int(success)
                self.ledger["shamRepairFailures"] += int(not success)
                self.recovery_pending = success
                self._audit(
                    "local_repair_proposal",
                    {
                        "eventIndex": event_index,
                        "helperIdentityId": proposal.actor_id,
                        "helperPosition": proposal.actor_pos,
                        "selectedIdentityId": self.selected_identity,
                        "selectedPosition": self.selected_position,
                        "nativeProposalKind": proposal.kind.value,
                        "nativeProposalEligible": validation.eligible_for_commit,
                        "success": success,
                        "actionRemaining": self.action_remaining,
                        "energyRemaining": self.energy_remaining,
                    },
                )
                decision = (
                    "s06_repair_success_pending"
                    if success
                    else "s06_sham_repair_no_effect"
                )
            else:
                self.ledger["matchedCostEvents"] += 1
                self.recovery_pending = True
                self._audit(
                    "matched_opportunity_energy_event",
                    {
                        "eventIndex": event_index,
                        "scheduledActorId": proposal.actor_id,
                        "nativeProposalKind": proposal.kind.value,
                        "nativeProposalEligible": validation.eligible_for_commit,
                        "assignedDuration": self.contract.assigned_duration,
                    },
                )
                decision = "s06_matched_cost_recovery_pending"
            return ValidationDecision(decision, False)

        if not validation.eligible_for_commit or not self.frozen:
            return validation
        target_id = (
            proposal.observed_target_id if proposal.kind == ProposalKind.SWAP else None
        )
        if proposal.actor_id == self.selected_identity:
            self.ledger["freezeActorBlocks"] += 1
            self._audit(
                "freeze_block",
                {
                    "eventIndex": event_index,
                    "reason": "s06_actor_frozen",
                    "identityId": proposal.actor_id,
                },
            )
            return ValidationDecision("rejected_s06_actor_frozen", False)
        if target_id == self.selected_identity:
            self.ledger["freezeTargetBlocks"] += 1
            self._audit(
                "freeze_block",
                {
                    "eventIndex": event_index,
                    "reason": "s06_target_frozen",
                    "identityId": target_id,
                },
            )
            return ValidationDecision("rejected_s06_target_frozen", False)
        return validation

    def after_batch(
        self,
        scenario: Scenario,
        state: RunState,
        proposals: tuple[Proposal, ...],
        decisions: Mapping[int, str],
        batch_start_index: int,
    ) -> None:
        if len(proposals) != 1:
            raise ValueError("S06 assisted-rescue panel is serial")
        if self.recovery_pending and self.frozen:
            duration = batch_start_index - self.start_event_index + 1
            reason = (
                "active_neighbor_repair"
                if self.contract.arm == RescueArm.ACTIVE
                else "matched_time_opportunity_energy"
            )
            self._recover(batch_start_index + 1, reason, duration)
        self.recovery_pending = False
        self.intervention_pending = False
        self.intervention_kind = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "arm": self.contract.arm.value,
            "selectedIdentityId": self.selected_identity,
            "selectedPosition": self.selected_position,
            "startEventIndex": self.start_event_index,
            "frozen": self.frozen,
            "recoveryEventIndex": self.recovery_event,
            "recoveryReason": self.recovery_reason,
            "interventionEventIndex": self.intervention_event,
            "interventionActorId": self.intervention_actor,
            "interventionNativeProposalKind": self.intervention_native_kind,
            "interventionNativeProposalEligible": self.intervention_native_eligible,
            "actionBudgetInitial": self.contract.action_budget,
            "actionBudgetRemaining": self.action_remaining,
            "energyBudgetInitial": self.contract.energy_budget,
            "energyBudgetRemaining": self.energy_remaining,
            "helperEnergySpent": dict(sorted(self.helper_energy_spent.items())),
        }

    def process_ledger(self) -> dict[str, int]:
        fields = (
            "chargedOpportunities",
            "freezeExposures",
            "freezeActorBlocks",
            "freezeTargetBlocks",
            "localAdjacencySignals",
            "repairProposals",
            "repairSuccesses",
            "shamRepairFailures",
            "matchedCostEvents",
            "spontaneousScheduleEvents",
            "nativeOpportunitiesSuppressed",
            "forgoneEligibleNativeChanges",
            "forgoneNativeNoOp",
            "forgoneNativeSwap",
            "forgoneNativeMemoryUpdate",
            "actionUnitsSpent",
            "energyUnitsSpent",
            "recoveries",
        )
        return {field: int(self.ledger[field]) for field in fields}


@dataclass(frozen=True, slots=True)
class AssistedRescueRun:
    contract: AssistedRescueContract
    source_scenario_id: str
    source_checkpoint_hash: str
    anchor_lesion_state_hash: str
    selected_identity: str
    start_event_index: int
    end_event_index: int
    initial_state_hash: str
    final_state_hash: str
    final_state: Mapping[str, Any]
    summary: Mapping[str, Any]
    event_digest: str
    events: tuple[Mapping[str, Any], ...]
    process_initial_state: Mapping[str, Any]
    process_final_state: Mapping[str, Any]
    process_ledger: Mapping[str, int]
    process_audit_digest: str
    process_audit_count: int
    process_transitions: tuple[Mapping[str, Any], ...]
    retained_process_audits: tuple[Mapping[str, Any], ...]
    opportunity_validation: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": ASSISTED_RUN_SCHEMA_VERSION,
            "benchmarkVersion": BENCHMARK_VERSION,
            "contract": self.contract.to_dict(),
            "sourceScenarioId": self.source_scenario_id,
            "sourceCheckpointHash": self.source_checkpoint_hash,
            "anchorLesionStateHash": self.anchor_lesion_state_hash,
            "selectedIdentityId": self.selected_identity,
            "startEventIndex": self.start_event_index,
            "endEventIndex": self.end_event_index,
            "initialStateHash": self.initial_state_hash,
            "finalStateHash": self.final_state_hash,
            "finalState": dict(self.final_state),
            "summary": dict(self.summary),
            "eventDigest": self.event_digest,
            "events": list(self.events),
            "processInitialState": dict(self.process_initial_state),
            "processFinalState": dict(self.process_final_state),
            "processLedger": dict(self.process_ledger),
            "processAuditDigest": self.process_audit_digest,
            "processAuditCount": self.process_audit_count,
            "processTransitions": list(self.process_transitions),
            "retainedProcessAudits": list(self.retained_process_audits),
            "opportunityValidation": dict(self.opportunity_validation),
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def _uniform_schedule(scenario: Scenario):
    def schedule(
        event_index: int, remaining_opportunities: int
    ) -> tuple[ScheduledOpportunity, ...]:
        if remaining_opportunities < 1:
            return ()
        actor_id, draws, consumed = scheduled_actor(
            scenario, event_index, include_draws=True
        )
        return (
            ScheduledOpportunity(
                actor_id,
                draws,
                (("actor_activation", consumed),),
            ),
        )

    return schedule


def _delta(final: Mapping[str, int], initial: Mapping[str, int]) -> dict[str, int]:
    return {key: int(final.get(key, 0)) - int(initial.get(key, 0)) for key in final}


def _execute_summary_opportunity(
    scenario: Scenario,
    state: RunState,
    router: ArchitectureProposalRouter,
    controller: AssistedRescueController,
) -> None:
    """Exact serial S06 transition projection without trace/snapshot overhead.

    This path preserves the authoritative proposal router, mechanical
    validation, cost rules, controller ordering, and counter-addressed scheduler.
    A snapshot is only needed when an accepted native state change is committed.
    Terminal state can change after such a commit or at the event-budget boundary;
    a rejected/no-op opportunity changes neither occupancy nor Selection memory.
    Full-event execution remains the validation authority.
    """

    if scenario.batch_width != 1 or state.terminal is not None:
        raise ValueError("S06 summary projection requires one live serial opportunity")
    event_index = state.activation_count
    actor_id, _, actor_consumed = scheduled_actor(
        scenario, event_index, include_draws=False
    )
    state.stream_counters["actor_activation"] = (
        state.stream_counters.get("actor_activation", 0) + actor_consumed
    )
    actor = scenario.cell_map[actor_id]
    side = None
    if actor.policy == Policy.BUBBLE and actor.fault == FaultMode.NORMAL:
        side, _ = scheduled_side(scenario, event_index)
        state.stream_counters["bubble_side"] = (
            state.stream_counters.get("bubble_side", 0) + 1
        )
    proposal = router.proposal_for(scenario, state, actor_id, side=side)
    proposal, consumption = controller.prepare(proposal, event_index)
    for stream, count in consumption:
        state.stream_counters[stream] = state.stream_counters.get(stream, 0) + count
    validation = validate_proposal(scenario, state, proposal)
    validation = controller.outcome(proposal, validation, event_index)
    decision = "accepted" if validation.eligible_for_commit else validation.decision
    delta = cost_delta(state.ledger, proposal, decision)
    changed = False
    if validation.eligible_for_commit:
        snapshot = state.clone()
        changed = commit_proposal(state, snapshot, proposal, decision)
    for key, value in delta.items():
        state.ledger[key] += value
    state.activation_count += 1
    if changed or state.activation_count >= scenario.max_activations:
        state.terminal = evaluate_terminal(scenario, state)
    controller.after_batch(
        scenario,
        state,
        (proposal,),
        {proposal.ordinal: decision},
        event_index,
    )


def run_assisted_rescue_phase(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    post_anchor_occupancy: Sequence[str],
    anchor_lesion_state_hash: str,
    selected_identity: str,
    contract: AssistedRescueContract,
    recovery_budget: int,
    trace_mode: str = "digest",
    retain_process_audits: bool = False,
) -> AssistedRescueRun:
    """Resume an exact S02/S03 state and execute one S06 arm."""

    if trace_mode not in {"full", "digest"}:
        raise ValueError("S06 supports full or digest trace modes")
    if recovery_budget < 1:
        raise ValueError("recovery budget must be positive")
    contract.validate(scenario, selected_identity)
    state = checkpoint.to_run_state(occupancy=post_anchor_occupancy)
    selected_position = state.occupancy.index(selected_identity)
    start_event = state.activation_count
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    initial_hash = state_hash(scenario.scenario_id, state)
    controller = AssistedRescueController(
        contract,
        scenario,
        selected_identity,
        selected_position,
        start_event,
        retain_audits=retain_process_audits,
    )
    process_initial = controller.state_dict()
    router = ArchitectureProposalRouter(
        ArchitectureExecutionContract.distributed_local(),
        action_interface=CommonActionInterface(),
    )
    retained: list[Mapping[str, Any]] = []
    digest = bytes.fromhex(EMPTY_DIGEST)
    distance_auc = 0
    # Distance depends only on occupancy.  Most permanent-freeze opportunities
    # are rejected/no-op transitions, so retain the exact opportunity-clock AUC
    # while avoiding an O(n^2) inversion recount when no swap was accepted.
    values = occupancy_values(scenario, state.occupancy)
    current_distance = strict_unequal_inversions(
        values, scenario.cells[0].direction
    )
    state.terminal = evaluate_terminal(scenario, state)
    while (
        state.terminal is None
        and state.activation_count - start_event < recovery_budget
    ):
        accepted_swaps_before = state.ledger["acceptedSwaps"]
        if trace_mode == "full":
            events, encoded = execute_batch(
                scenario,
                state,
                retain_events=True,
                emit_event_records=True,
                proposal_factory=router.proposal_for,
                schedule_factory=_uniform_schedule(scenario),
                execution_interceptor=controller,
            )
        else:
            _execute_summary_opportunity(scenario, state, router, controller)
            events, encoded = (), ()
        for item in encoded:
            digest = hashlib.sha256(digest + item).digest()
        retained.extend(events)
        if state.ledger["acceptedSwaps"] != accepted_swaps_before:
            values = occupancy_values(scenario, state.occupancy)
            current_distance = strict_unequal_inversions(
                values, scenario.cells[0].direction
            )
        distance_auc += current_distance
    if state.terminal is None:
        state.terminal = "phase_event_budget"
    phase_activations = state.activation_count - start_event
    ledger_delta = _delta(dict(state.ledger), initial_ledger)
    stream_delta = _delta(dict(state.stream_counters), initial_streams)
    process_ledger = controller.process_ledger()
    native_checks = ledger_identity(ledger_delta)
    recovery_duration = (
        None
        if controller.recovery_event is None
        else controller.recovery_event - start_event
    )
    state_record = controller.state_dict()
    helper_spent = sum(state_record["helperEnergySpent"].values())
    validation = {
        **native_checks,
        "activationDeltaMatchesPhase": ledger_delta["activations"]
        == phase_activations,
        "processOpportunityCountMatchesPhase": process_ledger[
            "chargedOpportunities"
        ]
        == phase_activations,
        "noS06RuntimeStreamsConsumed": not any(
            stream.startswith("assisted_rescue_") for stream in stream_delta
        ),
        "actionBudgetConserved": process_ledger["actionUnitsSpent"]
        + state_record["actionBudgetRemaining"]
        == state_record["actionBudgetInitial"],
        "energyBudgetConserved": process_ledger["energyUnitsSpent"]
        + state_record["energyBudgetRemaining"]
        == state_record["energyBudgetInitial"],
        "helperEnergyConserved": helper_spent
        == process_ledger["energyUnitsSpent"],
        "costIdentity": process_ledger["actionUnitsSpent"]
        == process_ledger["energyUnitsSpent"]
        == process_ledger["nativeOpportunitiesSuppressed"]
        == process_ledger["repairProposals"] + process_ledger["matchedCostEvents"],
        "repairOutcomePartition": process_ledger["repairProposals"]
        == process_ledger["repairSuccesses"]
        + process_ledger["shamRepairFailures"],
        "atMostOneIntervention": process_ledger["nativeOpportunitiesSuppressed"]
        <= 1,
        "atMostOneRecovery": process_ledger["recoveries"] <= 1,
        "activeSuccessRecoveryIdentity": contract.arm != RescueArm.ACTIVE
        or process_ledger["repairSuccesses"] == process_ledger["recoveries"],
        "shamNeverRecovers": contract.arm != RescueArm.SHAM
        or process_ledger["recoveries"] == 0,
        "passiveNeverRecovers": contract.arm != RescueArm.PASSIVE
        or process_ledger["recoveries"] == 0,
        "recoveryDurationConsistent": controller.recovery_event is None
        or recovery_duration is not None
        and recovery_duration >= 1,
        "scheduledDurationExact": contract.assigned_duration is None
        or controller.recovery_event is None
        or recovery_duration == contract.assigned_duration,
        "legalNativePrimitivesPreserved": all(
            kind in contract.native_legal_primitives for kind in controller.proposal_kinds
        ),
        "phaseBudgetRespected": phase_activations <= recovery_budget,
    }
    values = occupancy_values(scenario, state.occupancy)
    scheduled_unrecovered = (
        contract.arm in {RescueArm.SPONTANEOUS, RescueArm.MATCHED_COST}
        and contract.assigned_duration is None
    )
    summary = {
        "arm": contract.arm.value,
        "stopReason": state.terminal,
        "completed": state.terminal == "complete",
        "phaseActivationCount": phase_activations,
        "globalStartEventIndex": start_event,
        "globalEndEventIndex": state.activation_count,
        "recoveryBudget": recovery_budget,
        "finalDistance": strict_unequal_inversions(
            values, scenario.cells[0].direction
        ),
        "distanceAuc": distance_auc,
        "recoveryObserved": controller.recovery_event is not None,
        "recoveryDuration": recovery_duration,
        "recoveryCensored": controller.recovery_event is None,
        "scheduledUnrecovered": scheduled_unrecovered,
        "assignedDuration": contract.assigned_duration,
        "recoveryReason": controller.recovery_reason,
        "ledgerDelta": ledger_delta,
        "streamCounterDelta": stream_delta,
        "traceMode": trace_mode,
        "retainedEventCount": len(retained),
    }
    return AssistedRescueRun(
        contract=contract,
        source_scenario_id=scenario.scenario_id,
        source_checkpoint_hash=checkpoint.state_hash,
        anchor_lesion_state_hash=anchor_lesion_state_hash,
        selected_identity=selected_identity,
        start_event_index=start_event,
        end_event_index=state.activation_count,
        initial_state_hash=initial_hash,
        final_state_hash=state_hash(scenario.scenario_id, state),
        final_state=state.to_dict(),
        summary=summary,
        event_digest=digest.hex(),
        events=tuple(retained),
        process_initial_state=process_initial,
        process_final_state=state_record,
        process_ledger=process_ledger,
        process_audit_digest=controller.audit_digest.hex(),
        process_audit_count=controller.audit_count,
        process_transitions=tuple(controller.transitions),
        retained_process_audits=tuple(controller.retained_audits),
        opportunity_validation=validation,
    )


def exact_replay_assisted_rescue(
    run_result: AssistedRescueRun,
    scenario: Scenario,
    checkpoint: Checkpoint,
    post_anchor_occupancy: Sequence[str],
    recovery_budget: int,
) -> AssistedRescueRun:
    replayed = run_assisted_rescue_phase(
        scenario,
        checkpoint,
        post_anchor_occupancy=post_anchor_occupancy,
        anchor_lesion_state_hash=run_result.anchor_lesion_state_hash,
        selected_identity=run_result.selected_identity,
        contract=run_result.contract,
        recovery_budget=recovery_budget,
        trace_mode=run_result.summary["traceMode"],
        retain_process_audits=bool(run_result.retained_process_audits),
    )
    if replayed.to_json_bytes() != run_result.to_json_bytes():
        raise AssertionError("S06 assisted-rescue replay mismatch")
    return replayed


def assisted_case_id(
    s01_pairing_block_id: str,
    timing_condition_id: str,
    anchor_lesion_state_hash: str,
) -> str:
    return "e05ac6:" + sha256_json(
        {
            "s01PairingBlockId": s01_pairing_block_id,
            "timingConditionId": timing_condition_id,
            "anchorLesionStateHash": anchor_lesion_state_hash,
        }
    )
