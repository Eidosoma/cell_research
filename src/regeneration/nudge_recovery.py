"""E05 S05 contact-mediated unfreezing over the frozen S04 runtime overlay.

The selected identity remains part of the original immutable S01 scenario.  A
typed runtime overlay blocks its eligible state changes while frozen, preserving
the S04 scenario/RNG boundary.  Contact counters are endogenous and consume no
random stream; matched spontaneous schedules are constructed outside the run by
an isolated, counter-addressed assignment stream.
"""

from __future__ import annotations

from collections import Counter, deque
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
    Proposal,
    ProposalKind,
    RunState,
    Scenario,
    canonical_json_bytes,
    sha256_json,
    state_hash,
)
from reference_simulator.scheduler import ScheduledOpportunity, scheduled_actor
from reference_simulator.transition_primitives import ValidationDecision, ledger_identity

from .tasks import Checkpoint, occupancy_values, strict_unequal_inversions


BENCHMARK_VERSION = "E05-nudge-dependent-unfreezing-v1"
NUDGE_SPEC_SCHEMA_VERSION = "e05.s05.nudge-recovery-spec.v1"
NUDGE_RUN_SCHEMA_VERSION = "e05.s05.nudge-recovery-run.v1"
MATCHING_ASSIGNMENT_STREAM = "nudge_spontaneous_matching_assignment_s05_v1"


class NudgeMechanism(str, Enum):
    ATTEMPTED_CONTACT = "attempted_contact_k3_v1"
    PRESSURE_PROXY = "accumulated_displacement_pressure_k3_v1"
    DISTINCT_NEIGHBOR = "distinct_neighbor_k2_v1"
    LOCAL_QUORUM = "local_two_sided_quorum_window_16_v1"


class RecoveryMode(str, Enum):
    CONTACT_DEPENDENT = "contact_dependent"
    SPONTANEOUS_SCHEDULED = "spontaneous_scheduled"


class MatchingChoice(str, Enum):
    PRIMARY_N_POLICY = "empirical_bijection_n_policy_v1"
    SENSITIVITY_N = "empirical_bijection_n_v1"
    SENSITIVITY_POOLED = "empirical_bijection_pooled_v1"


NUDGE_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s05/nudge-recovery-spec.schema.json",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenQuestion",
        "inherits",
        "contactEvent",
        "mechanisms",
        "matching",
        "validationPanel",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": NUDGE_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S05"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "mechanisms": {"type": "array", "minItems": 4, "maxItems": 4},
        "matching": {"type": "object"},
        "validationPanel": {"type": "object"},
    },
}


@dataclass(frozen=True, slots=True)
class NudgeRecoveryContract:
    mechanism: NudgeMechanism
    mode: RecoveryMode = RecoveryMode.CONTACT_DEPENDENT
    spontaneous_duration: int | None = None
    contact_threshold: int = 3
    pressure_threshold: int = 3
    distinct_neighbor_threshold: int = 2
    quorum_window: int = 16
    information_permission: str = "policy_native_local"
    legal_primitives: tuple[str, ...] = ("NoOp", "Swap", "MemoryUpdate")
    scheduler: str = "uniform_random_activation"
    continuation: str = "skip_and_continue"
    retry: str = "no_retry"

    def validate(self, scenario: Scenario, selected_identity: str) -> None:
        ArchitectureExecutionContract.distributed_local().validate_policy_action_boundary(
            scenario
        )
        if selected_identity not in scenario.cell_map:
            raise ValueError("S05 selected an unknown identity")
        if any(cell.fault != FaultMode.NORMAL for cell in scenario.cells):
            raise ValueError("S05 runtime overlays require the original normal S01 scenario")
        if self.information_permission != "policy_native_local":
            raise ValueError("S05 cannot change policy information permission")
        if self.legal_primitives != ("NoOp", "Swap", "MemoryUpdate"):
            raise ValueError("S05 legal primitive set is frozen")
        if self.scheduler != "uniform_random_activation":
            raise ValueError("S05 preserves uniform random activation")
        if self.continuation != "skip_and_continue" or self.retry != "no_retry":
            raise ValueError("S05 preserves continuation and no-retry contracts")
        if min(
            self.contact_threshold,
            self.pressure_threshold,
            self.distinct_neighbor_threshold,
            self.quorum_window,
        ) < 1:
            raise ValueError("S05 thresholds and quorum window must be positive")
        if self.mode == RecoveryMode.CONTACT_DEPENDENT:
            if self.spontaneous_duration is not None:
                raise ValueError("contact-dependent recovery cannot have a schedule")
        elif self.spontaneous_duration is not None and self.spontaneous_duration < 1:
            raise ValueError("a finite spontaneous duration must be positive")

    @property
    def owns_runtime_streams(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanismId": self.mechanism.value,
            "mode": self.mode.value,
            "spontaneousDurationOpportunities": self.spontaneous_duration,
            "contactThreshold": self.contact_threshold,
            "pressureThreshold": self.pressure_threshold,
            "distinctNeighborThreshold": self.distinct_neighbor_threshold,
            "quorumWindowOpportunities": self.quorum_window,
            "runtimeStreams": list(self.owns_runtime_streams),
            "informationPermission": self.information_permission,
            "legalPrimitives": list(self.legal_primitives),
            "scheduler": self.scheduler,
            "continuation": self.continuation,
            "retry": self.retry,
        }


def validate_nudge_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(NUDGE_SPEC_SCHEMA).validate(specification)
    mechanism_ids = [item["mechanismId"] for item in specification["mechanisms"]]
    if len(set(mechanism_ids)) != 4 or set(mechanism_ids) != {
        item.value for item in NudgeMechanism
    }:
        raise ValueError("S05 must define every mechanism exactly once")
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "lesionSpecSchemaVersion": "e05.s03.lesion-spec.v1",
        "dynamicSpecSchemaVersion": "e05.s04.dynamic-fault-spec.v1",
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "runtimeAnchorLesion": "segment_reversal_central_v1",
        "runtimeScenarioBoundary": "original_s01_scenario_id_runtime_overlay",
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"inherited S01-S04 contract changed: {key}")
    matching = specification["matching"]
    if matching.get("calibrationReplicates") != [0, 1]:
        raise ValueError("S05 calibration split changed")
    if matching.get("confirmatoryReplicates") != [2, 3]:
        raise ValueError("S05 confirmatory split changed")
    if set(matching.get("choices", [])) != {item.value for item in MatchingChoice}:
        raise ValueError("S05 matching sensitivity choices changed")
    panel = specification["validationPanel"]
    if panel.get("plannedCheckpointCount") != 384:
        raise ValueError("S05 must retain all 384 S02 checkpoints")
    if panel.get("plannedRunCount") != 3840:
        raise ValueError("S05 planned run count changed")
    if specification.get("pressureInterpretation") != "abstract_event_count_proxy":
        raise ValueError("S05 pressure must remain an abstract event-count proxy")


@dataclass(frozen=True, slots=True)
class ContactEvidence:
    event_index: int
    actor_id: str
    other_identity_id: str
    selected_role: str
    side: str
    inbound: bool
    actor_position: int
    target_position: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventIndex": self.event_index,
            "actorId": self.actor_id,
            "otherIdentityId": self.other_identity_id,
            "selectedRole": self.selected_role,
            "side": self.side,
            "inbound": self.inbound,
            "actorPosition": self.actor_position,
            "targetPosition": self.target_position,
        }


class NudgeRecoveryController:
    """Frozen-identity gate, local evidence counters, and supplemental ledger."""

    __slots__ = (
        "contract",
        "selected_identity",
        "start_event_index",
        "frozen",
        "recovery_event",
        "recovery_reason",
        "recovery_pending",
        "contact_count",
        "pressure_units",
        "distinct_neighbors",
        "quorum_contacts",
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
        contract: NudgeRecoveryContract,
        scenario: Scenario,
        selected_identity: str,
        start_event_index: int,
        *,
        retain_audits: bool = False,
    ) -> None:
        contract.validate(scenario, selected_identity)
        self.contract = contract
        self.selected_identity = selected_identity
        self.start_event_index = start_event_index
        self.frozen = True
        self.recovery_event: int | None = None
        self.recovery_reason: str | None = None
        self.recovery_pending = False
        self.contact_count = 0
        self.pressure_units = 0
        self.distinct_neighbors: set[str] = set()
        self.quorum_contacts: deque[tuple[int, str, str]] = deque()
        self.ledger: Counter[str] = Counter()
        self.audit_digest = hashlib.sha256(b"E05/S05/nudge-audit/v1").digest()
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

    def prepare(
        self, proposal: Proposal, event_index: int
    ) -> tuple[Proposal, tuple[tuple[str, int], ...]]:
        self.ledger["chargedOpportunities"] += 1
        self.proposal_kinds.add(proposal.kind.value)
        self.recovery_pending = False
        if (
            self.contract.mode == RecoveryMode.SPONTANEOUS_SCHEDULED
            and self.frozen
            and self.contract.spontaneous_duration is not None
            and event_index
            >= self.start_event_index + self.contract.spontaneous_duration
        ):
            self._recover(
                event_index,
                "matched_spontaneous_schedule",
                self.contract.spontaneous_duration,
            )
        if self.frozen:
            self.ledger["freezeExposures"] += 1
        return proposal, ()

    def _contact_evidence(
        self, proposal: Proposal, event_index: int
    ) -> ContactEvidence | None:
        if proposal.kind != ProposalKind.SWAP or proposal.observed_target_id is None:
            return None
        actor_selected = proposal.actor_id == self.selected_identity
        target_selected = proposal.observed_target_id == self.selected_identity
        if actor_selected == target_selected:
            return None
        if proposal.target_pos is None or abs(proposal.actor_pos - proposal.target_pos) != 1:
            return None
        if actor_selected:
            other_id = proposal.observed_target_id
            selected_pos = proposal.actor_pos
            other_pos = proposal.target_pos
            selected_role = "actor"
        else:
            other_id = proposal.actor_id
            selected_pos = proposal.target_pos
            other_pos = proposal.actor_pos
            selected_role = "target"
        return ContactEvidence(
            event_index=event_index,
            actor_id=proposal.actor_id,
            other_identity_id=other_id,
            selected_role=selected_role,
            side="left" if other_pos < selected_pos else "right",
            inbound=target_selected,
            actor_position=proposal.actor_pos,
            target_position=proposal.target_pos,
        )

    def _record_contact(self, evidence: ContactEvidence) -> bool:
        self.ledger["qualifyingContactEvents"] += 1
        self.ledger["inboundContactEvents"] += int(evidence.inbound)
        self.contact_count += 1
        new_neighbor = evidence.other_identity_id not in self.distinct_neighbors
        self.distinct_neighbors.add(evidence.other_identity_id)
        self.ledger["distinctNeighborAdds"] += int(new_neighbor)
        if evidence.inbound:
            self.pressure_units += 1
            self.ledger["pressureProxyUnits"] += 1
            self.quorum_contacts.append(
                (evidence.event_index, evidence.other_identity_id, evidence.side)
            )
        cutoff = evidence.event_index - self.contract.quorum_window + 1
        while self.quorum_contacts and self.quorum_contacts[0][0] < cutoff:
            self.quorum_contacts.popleft()
            self.ledger["quorumEvidenceExpirations"] += 1
        left = {identity for _, identity, side in self.quorum_contacts if side == "left"}
        right = {
            identity for _, identity, side in self.quorum_contacts if side == "right"
        }
        quorum = any(left_id != right_id for left_id in left for right_id in right)
        trigger = {
            NudgeMechanism.ATTEMPTED_CONTACT: self.contact_count
            >= self.contract.contact_threshold,
            NudgeMechanism.PRESSURE_PROXY: self.pressure_units
            >= self.contract.pressure_threshold,
            NudgeMechanism.DISTINCT_NEIGHBOR: len(self.distinct_neighbors)
            >= self.contract.distinct_neighbor_threshold,
            NudgeMechanism.LOCAL_QUORUM: quorum,
        }[self.contract.mechanism]
        self.ledger["mechanismTriggerEvents"] += int(trigger)
        self._audit(
            "qualifying_contact",
            {
                **evidence.to_dict(),
                "contactCount": self.contact_count,
                "pressureProxyUnits": self.pressure_units,
                "distinctNeighborCount": len(self.distinct_neighbors),
                "quorumLeftIdentities": sorted(left),
                "quorumRightIdentities": sorted(right),
                "triggered": trigger,
            },
        )
        return trigger

    def outcome(
        self,
        proposal: Proposal,
        validation: ValidationDecision,
        event_index: int,
    ) -> ValidationDecision:
        if not validation.eligible_for_commit:
            return validation
        target_id = (
            proposal.observed_target_id if proposal.kind == ProposalKind.SWAP else None
        )
        if self.frozen:
            evidence = self._contact_evidence(proposal, event_index)
            if (
                evidence is not None
                and self.contract.mode == RecoveryMode.CONTACT_DEPENDENT
                and self._record_contact(evidence)
            ):
                self.recovery_pending = True
            if proposal.actor_id == self.selected_identity:
                self.ledger["freezeActorBlocks"] += 1
                self._audit(
                    "block",
                    {
                        "eventIndex": event_index,
                        "reason": "nudge_actor_frozen",
                        "identityId": proposal.actor_id,
                    },
                )
                return ValidationDecision("rejected_nudge_actor_frozen", False)
            if target_id == self.selected_identity:
                self.ledger["freezeTargetBlocks"] += 1
                self._audit(
                    "block",
                    {
                        "eventIndex": event_index,
                        "reason": "nudge_target_frozen",
                        "identityId": target_id,
                    },
                )
                return ValidationDecision("rejected_nudge_target_frozen", False)
        return validation

    def _recover(self, event_index: int, reason: str, duration: int) -> None:
        if not self.frozen:
            raise AssertionError("S05 recovery attempted twice")
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

    def after_batch(
        self,
        scenario: Scenario,
        state: RunState,
        proposals: tuple[Proposal, ...],
        decisions: Mapping[int, str],
        batch_start_index: int,
    ) -> None:
        if len(proposals) != 1:
            raise ValueError("S05 regeneration nudge panel is serial")
        if self.recovery_pending and self.frozen:
            duration = batch_start_index - self.start_event_index + 1
            self._recover(
                batch_start_index + 1,
                self.contract.mechanism.value,
                duration,
            )
        self.recovery_pending = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "mechanismId": self.contract.mechanism.value,
            "mode": self.contract.mode.value,
            "selectedIdentityId": self.selected_identity,
            "startEventIndex": self.start_event_index,
            "frozen": self.frozen,
            "recoveryEventIndex": self.recovery_event,
            "recoveryReason": self.recovery_reason,
            "contactCount": self.contact_count,
            "pressureProxyUnits": self.pressure_units,
            "distinctNeighborIds": sorted(self.distinct_neighbors),
            "quorumEvidence": [list(item) for item in self.quorum_contacts],
        }

    def process_ledger(self) -> dict[str, int]:
        fields = (
            "chargedOpportunities",
            "freezeExposures",
            "freezeActorBlocks",
            "freezeTargetBlocks",
            "qualifyingContactEvents",
            "inboundContactEvents",
            "pressureProxyUnits",
            "distinctNeighborAdds",
            "quorumEvidenceExpirations",
            "mechanismTriggerEvents",
            "recoveries",
        )
        return {field: int(self.ledger[field]) for field in fields}


@dataclass(frozen=True, slots=True)
class NudgeRun:
    contract: NudgeRecoveryContract
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
            "schemaVersion": NUDGE_RUN_SCHEMA_VERSION,
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


def run_nudge_phase(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    post_anchor_occupancy: Sequence[str],
    anchor_lesion_state_hash: str,
    selected_identity: str,
    contract: NudgeRecoveryContract,
    recovery_budget: int,
    trace_mode: str = "digest",
    retain_process_audits: bool = False,
) -> NudgeRun:
    """Resume an exact S02/S03 state and execute one S05 recovery arm."""

    if trace_mode not in {"full", "digest"}:
        raise ValueError("S05 supports full or digest trace modes")
    if recovery_budget < 1:
        raise ValueError("recovery budget must be positive")
    contract.validate(scenario, selected_identity)
    state = checkpoint.to_run_state(occupancy=post_anchor_occupancy)
    start_event = state.activation_count
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    initial_hash = state_hash(scenario.scenario_id, state)
    controller = NudgeRecoveryController(
        contract,
        scenario,
        selected_identity,
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
    state.terminal = evaluate_terminal(scenario, state)
    while (
        state.terminal is None
        and state.activation_count - start_event < recovery_budget
    ):
        events, encoded = execute_batch(
            scenario,
            state,
            retain_events=trace_mode == "full",
            proposal_factory=router.proposal_for,
            schedule_factory=_uniform_schedule(scenario),
            execution_interceptor=controller,
        )
        for item in encoded:
            digest = hashlib.sha256(digest + item).digest()
        retained.extend(events)
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
    validation = {
        **native_checks,
        "activationDeltaMatchesPhase": ledger_delta["activations"]
        == phase_activations,
        "processOpportunityCountMatchesPhase": process_ledger[
            "chargedOpportunities"
        ]
        == phase_activations,
        "noS05RuntimeStreamsConsumed": not any(
            stream.startswith("nudge_") for stream in stream_delta
        ),
        "pressureIsInboundEventSubset": process_ledger["pressureProxyUnits"]
        == process_ledger["inboundContactEvents"]
        <= process_ledger["qualifyingContactEvents"],
        "distinctNeighborAddsBounded": process_ledger["distinctNeighborAdds"]
        <= process_ledger["qualifyingContactEvents"],
        "atMostOneRecovery": process_ledger["recoveries"] <= 1,
        "recoveryDurationConsistent": controller.recovery_event is None
        or recovery_duration is not None
        and recovery_duration >= 1,
        "legalPrimitivesPreserved": all(
            kind in contract.legal_primitives for kind in controller.proposal_kinds
        ),
        "phaseBudgetRespected": phase_activations <= recovery_budget,
    }
    values = occupancy_values(scenario, state.occupancy)
    scheduled_unrecovered = (
        contract.mode == RecoveryMode.SPONTANEOUS_SCHEDULED
        and contract.spontaneous_duration is None
    )
    summary = {
        "mechanismId": contract.mechanism.value,
        "mode": contract.mode.value,
        "stopReason": state.terminal,
        "completed": state.terminal == "complete",
        "phaseActivationCount": phase_activations,
        "globalStartEventIndex": start_event,
        "globalEndEventIndex": state.activation_count,
        "recoveryBudget": recovery_budget,
        "finalDistance": strict_unequal_inversions(
            values, scenario.cells[0].direction
        ),
        "recoveryObserved": controller.recovery_event is not None,
        "recoveryDuration": recovery_duration,
        "recoveryCensored": controller.recovery_event is None,
        "scheduledUnrecovered": scheduled_unrecovered,
        "assignedSpontaneousDuration": contract.spontaneous_duration,
        "recoveryReason": controller.recovery_reason,
        "ledgerDelta": ledger_delta,
        "streamCounterDelta": stream_delta,
        "traceMode": trace_mode,
        "retainedEventCount": len(retained),
    }
    return NudgeRun(
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
        process_final_state=controller.state_dict(),
        process_ledger=process_ledger,
        process_audit_digest=controller.audit_digest.hex(),
        process_audit_count=controller.audit_count,
        process_transitions=tuple(controller.transitions),
        retained_process_audits=tuple(controller.retained_audits),
        opportunity_validation=validation,
    )


def exact_replay_nudge(
    run_result: NudgeRun,
    scenario: Scenario,
    checkpoint: Checkpoint,
    post_anchor_occupancy: Sequence[str],
    recovery_budget: int,
) -> NudgeRun:
    replayed = run_nudge_phase(
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
        raise AssertionError("S05 nudge recovery replay mismatch")
    return replayed


def nudge_case_id(
    s01_pairing_block_id: str,
    timing_condition_id: str,
    anchor_lesion_state_hash: str,
    mechanism: NudgeMechanism,
) -> str:
    return "e05nc5:" + sha256_json(
        {
            "s01PairingBlockId": s01_pairing_block_id,
            "timingConditionId": timing_condition_id,
            "anchorLesionStateHash": anchor_lesion_state_hash,
            "mechanismId": mechanism.value,
        }
    )
