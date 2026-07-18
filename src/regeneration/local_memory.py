"""E05 S07 bounded local-memory variants over the S04-S06 runtime overlay.

The native policy still constructs exactly one ordinary proposal per charged
opportunity.  Active S07 variants maintain a finite private state using only
the scheduled actor's prior local outcomes and the S06 adjacent-frozen signal.
When prior state is ready, the controller replaces that already-built proposal
with the same unit-cost deterministic repair used as an engineered benchmark
in S06.  Memoryless controls never instantiate the private-state bank.
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


BENCHMARK_VERSION = "E05-minimal-local-memory-v1"
LOCAL_MEMORY_SPEC_SCHEMA_VERSION = "e05.s07.local-memory-spec.v1"
LOCAL_MEMORY_RUN_SCHEMA_VERSION = "e05.s07.local-memory-run.v1"
CONTROL_ASSIGNMENT_STREAM = "local_memory_control_assignment_s07_v1"

FAILED = "failed_move_counter_cap3_v1"
BLOCKED = "blocked_direction_mask_v1"
RECENT = "recent_neighbor_state_ttl3_v1"
TIMER = "time_since_local_progress_cap7_v1"


class MemoryArm(str, Enum):
    ACTIVE = "active_local_memory"
    PERMANENT = "memoryless_permanent_freeze"
    IMMEDIATE = "memoryless_immediate_neighbor_repair_s06_reference"
    MATCHED = "memoryless_matched_time_opportunity_energy"


class MemoryVariant(str, Enum):
    FAILED_ONLY = "failed_move_counter_only_v1"
    BLOCKED_ONLY = "blocked_direction_only_v1"
    RECENT_ONLY = "recent_neighbor_only_v1"
    TIMER_ONLY = "time_since_local_progress_only_v1"
    ALL = "all_components_union_v1"


class MatchingChoice(str, Enum):
    PRIMARY_N_POLICY = "empirical_bijection_n_policy_v1"
    SENSITIVITY_N = "empirical_bijection_n_v1"
    SENSITIVITY_POOLED = "empirical_bijection_pooled_v1"


VARIANT_COMPONENTS: dict[MemoryVariant, tuple[str, ...]] = {
    MemoryVariant.FAILED_ONLY: (FAILED,),
    MemoryVariant.BLOCKED_ONLY: (BLOCKED,),
    MemoryVariant.RECENT_ONLY: (RECENT,),
    MemoryVariant.TIMER_ONLY: (TIMER,),
    MemoryVariant.ALL: (FAILED, BLOCKED, RECENT, TIMER),
}
COMPONENT_BITS = {FAILED: 2, BLOCKED: 2, RECENT: 4, TIMER: 3}


LOCAL_MEMORY_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s07/local-memory-spec.schema.json",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenAtUtc",
        "frozenQuestion",
        "s06Handoff",
        "inherits",
        "localObservation",
        "memoryComponents",
        "variants",
        "repairAction",
        "resetSemantics",
        "controls",
        "matching",
        "structuralConfoundingAudit",
        "complexityCosts",
        "successCriteria",
        "validationPanel",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": LOCAL_MEMORY_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S07"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "memoryComponents": {"type": "array", "minItems": 4, "maxItems": 4},
        "variants": {"type": "array", "minItems": 5, "maxItems": 5},
        "controls": {"type": "array", "minItems": 3, "maxItems": 3},
    },
}


def validate_local_memory_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(LOCAL_MEMORY_SPEC_SCHEMA).validate(specification)
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "lesionSpecSchemaVersion": "e05.s03.lesion-spec.v1",
        "dynamicSpecSchemaVersion": "e05.s04.dynamic-fault-spec.v1",
        "nudgeSpecSchemaVersion": "e05.s05.nudge-recovery-spec.v1",
        "rescueSpecSchemaVersion": "e05.s06.assisted-rescue-spec.v1",
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "runtimeAnchorLesion": "segment_reversal_central_v1",
        "runtimeScenarioBoundary": "original_s01_scenario_id_runtime_overlay",
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"inherited S01-S06 contract changed: {key}")
    component_rows = {item["componentId"]: item for item in specification["memoryComponents"]}
    if set(component_rows) != set(COMPONENT_BITS):
        raise ValueError("S07 component set changed")
    for component, bits in COMPONENT_BITS.items():
        if component_rows[component].get("storageBitsPerIdentity") != bits:
            raise ValueError(f"S07 component complexity changed: {component}")
    variant_rows = {item["variantId"]: item for item in specification["variants"]}
    if set(variant_rows) != {item.value for item in MemoryVariant}:
        raise ValueError("S07 variant set changed")
    for variant, components in VARIANT_COMPONENTS.items():
        row = variant_rows[variant.value]
        if tuple(row["enabledComponents"]) != components:
            raise ValueError(f"S07 enabled components changed: {variant.value}")
        if row["storageBitsPerIdentity"] != sum(COMPONENT_BITS[item] for item in components):
            raise ValueError(f"S07 storage cost changed: {variant.value}")
    if specification["repairAction"].get("attemptLimit") != 1:
        raise ValueError("S07 repair must remain one-shot")
    for key in (
        "actionCostPerProposal",
        "energyCostPerProposal",
        "collectiveActionBudget",
        "collectiveEnergyBudget",
        "perHelperEnergyBudget",
    ):
        if specification["repairAction"].get(key) != 1:
            raise ValueError(f"S07 unit repair budget changed: {key}")
    if specification["localObservation"].get("runtimeStreams") != []:
        raise ValueError("S07 active memory must remain stream-free")
    if set(specification["matching"].get("choices", [])) != {
        item.value for item in MatchingChoice
    }:
        raise ValueError("S07 matching choices changed")
    panel = specification["validationPanel"]
    expected_counts = {
        "plannedCheckpointCount": 384,
        "variantCount": 5,
        "plannedCalibrationActiveRuns": 960,
        "plannedConfirmatoryActiveRuns": 960,
        "plannedConfirmatoryMatchedControlRuns": 2880,
        "plannedConfirmatorySharedMemorylessRuns": 192,
        "plannedConfirmatoryImmediateReferenceRuns": 192,
        "plannedRunCount": 5184,
        "plannedPairwiseContrasts": 4800,
        "plannedControlAssignments": 2880,
    }
    for key, value in expected_counts.items():
        if panel.get(key) != value:
            raise ValueError(f"S07 planned panel changed: {key}")


@dataclass(frozen=True, slots=True)
class ActorMemory:
    failed_count: int = 0
    blocked_mask: int = 0
    recent_valid: bool = False
    recent_side: int = -1
    recent_age: int = 0
    local_no_progress: int = 0

    def validate(self) -> None:
        if not 0 <= self.failed_count <= 3:
            raise AssertionError("failed-move counter escaped 0..3")
        if not 0 <= self.blocked_mask <= 3:
            raise AssertionError("blocked-direction mask escaped two bits")
        if self.recent_side not in {-1, 0, 1}:
            raise AssertionError("recent-neighbor side escaped {-1,0,1}")
        if not 0 <= self.recent_age <= 3:
            raise AssertionError("recent-neighbor age escaped 0..3")
        if not self.recent_valid and (self.recent_side, self.recent_age) != (-1, 0):
            raise AssertionError("invalid recent-neighbor state is not canonical")
        if not 0 <= self.local_no_progress <= 7:
            raise AssertionError("local-progress timer escaped 0..7")

    def to_dict(self, enabled: Sequence[str]) -> dict[str, Any]:
        answer: dict[str, Any] = {}
        if FAILED in enabled:
            answer["failedMoveCount"] = self.failed_count
        if BLOCKED in enabled:
            answer["blockedDirectionMask"] = self.blocked_mask
        if RECENT in enabled:
            answer["recentNeighborValid"] = self.recent_valid
            answer["recentNeighborSide"] = self.recent_side
            answer["recentNeighborAge"] = self.recent_age
        if TIMER in enabled:
            answer["timeSinceLocalProgress"] = self.local_no_progress
        return answer


@dataclass(frozen=True, slots=True)
class LocalObservation:
    """Complete information passed to the private state bank."""

    actor_id: str
    adjacent_side: int | None


@dataclass(frozen=True, slots=True)
class LocalOutcome:
    """Only an actor's own local disposition can update its private state."""

    actor_id: str
    adjacent_side: int | None
    freeze_target_blocked: bool
    blocked_direction_side: int | None
    native_progress: bool


class LocalMemoryBank:
    """Finite private state with no scenario, occupancy, clock, or target access."""

    __slots__ = ("variant", "enabled", "states", "ledger")

    def __init__(self, variant: MemoryVariant, actor_ids: Sequence[str]) -> None:
        self.variant = variant
        self.enabled = VARIANT_COMPONENTS[variant]
        self.states = {actor_id: ActorMemory() for actor_id in actor_ids}
        self.ledger: Counter[str] = Counter()

    @property
    def storage_bits_per_identity(self) -> int:
        return sum(COMPONENT_BITS[item] for item in self.enabled)

    def ready(self, observation: LocalObservation) -> tuple[bool, str | None]:
        if observation.adjacent_side not in {0, 1}:
            return False, None
        state = self.states[observation.actor_id]
        state.validate()
        self.ledger["componentStateReads"] += len(self.enabled)
        self.ledger["readinessComparisons"] += len(self.enabled)
        signals = {
            FAILED: state.failed_count == 3,
            BLOCKED: bool(state.blocked_mask & (1 << observation.adjacent_side)),
            RECENT: state.recent_valid
            and state.recent_side == observation.adjacent_side
            and state.recent_age <= 3,
            TIMER: state.local_no_progress == 7,
        }
        for component in self.enabled:
            if signals[component]:
                self.ledger[f"readySignal::{component}"] += 1
                return True, component
        return False, None

    def update(self, outcome: LocalOutcome) -> None:
        before = self.states[outcome.actor_id]
        before.validate()
        self.ledger["memoryUpdateOpportunities"] += 1
        self.ledger["componentStateReads"] += len(self.enabled)
        values = {
            "failed_count": before.failed_count,
            "blocked_mask": before.blocked_mask,
            "recent_valid": before.recent_valid,
            "recent_side": before.recent_side,
            "recent_age": before.recent_age,
            "local_no_progress": before.local_no_progress,
        }
        if outcome.native_progress:
            for component in self.enabled:
                if before.to_dict((component,)) != ActorMemory().to_dict((component,)):
                    self.ledger["componentResets"] += 1
            after = ActorMemory()
        else:
            if FAILED in self.enabled and outcome.freeze_target_blocked:
                if values["failed_count"] == 3:
                    self.ledger["saturatingIncrements"] += 1
                values["failed_count"] = min(3, values["failed_count"] + 1)
            if BLOCKED in self.enabled and outcome.freeze_target_blocked:
                if outcome.blocked_direction_side not in {0, 1}:
                    raise AssertionError("a target freeze block lacked local side")
                values["blocked_mask"] |= 1 << outcome.blocked_direction_side
            if RECENT in self.enabled:
                if outcome.adjacent_side in {0, 1}:
                    values["recent_valid"] = True
                    values["recent_side"] = outcome.adjacent_side
                    values["recent_age"] = 0
                elif values["recent_valid"]:
                    if values["recent_age"] == 3:
                        values["recent_valid"] = False
                        values["recent_side"] = -1
                        values["recent_age"] = 0
                    else:
                        values["recent_age"] += 1
            if TIMER in self.enabled:
                if values["local_no_progress"] == 7:
                    self.ledger["saturatingIncrements"] += 1
                values["local_no_progress"] = min(7, values["local_no_progress"] + 1)
            after = ActorMemory(**values)
        after.validate()
        before_dict = before.to_dict(self.enabled)
        after_dict = after.to_dict(self.enabled)
        for component in self.enabled:
            if before.to_dict((component,)) != after.to_dict((component,)):
                self.ledger["componentStateWrites"] += 1
                self.ledger[f"componentWrites::{component}"] += 1
        if before_dict != after_dict:
            self.states[outcome.actor_id] = after

    def reset(self) -> None:
        for actor_id, before in self.states.items():
            for component in self.enabled:
                if before.to_dict((component,)) != ActorMemory().to_dict((component,)):
                    self.ledger["explicitResetComponents"] += 1
            self.states[actor_id] = ActorMemory()

    def validate_bounds(self) -> bool:
        for state in self.states.values():
            state.validate()
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "variantId": self.variant.value,
            "enabledComponents": list(self.enabled),
            "storageBitsPerIdentity": self.storage_bits_per_identity,
            "actors": {
                actor_id: self.states[actor_id].to_dict(self.enabled)
                for actor_id in sorted(self.states)
            },
        }


@dataclass(frozen=True, slots=True)
class LocalMemoryContract:
    arm: MemoryArm
    variant: MemoryVariant | None = None
    assigned_duration: int | None = None
    action_budget: int = 1
    energy_budget: int = 1
    per_helper_energy_budget: int = 1
    information_permission: str = "policy_native_local"
    native_legal_primitives: tuple[str, ...] = ("NoOp", "Swap", "MemoryUpdate")
    scheduler: str = "uniform_random_activation"
    continuation: str = "skip_and_continue"
    retry: str = "no_retry"

    def validate(self, scenario: Scenario, selected_identity: str) -> None:
        ArchitectureExecutionContract.distributed_local().validate_policy_action_boundary(scenario)
        if selected_identity not in scenario.cell_map:
            raise ValueError("S07 selected an unknown identity")
        if any(cell.fault != FaultMode.NORMAL for cell in scenario.cells):
            raise ValueError("S07 requires the original normal S01 scenario")
        if self.information_permission != "policy_native_local":
            raise ValueError("S07 cannot change native policy information")
        if self.native_legal_primitives != ("NoOp", "Swap", "MemoryUpdate"):
            raise ValueError("S07 cannot change native legal primitives")
        if self.scheduler != "uniform_random_activation":
            raise ValueError("S07 preserves uniform activation")
        if self.continuation != "skip_and_continue" or self.retry != "no_retry":
            raise ValueError("S07 preserves continuation and no-retry")
        if (self.action_budget, self.energy_budget, self.per_helper_energy_budget) != (1, 1, 1):
            raise ValueError("S07 preserves unit action/energy budgets")
        if self.arm == MemoryArm.ACTIVE and self.variant is None:
            raise ValueError("an active S07 arm requires a memory variant")
        if self.arm == MemoryArm.MATCHED and self.variant is None:
            raise ValueError("a matched S07 arm requires its donor variant")
        if self.arm in {MemoryArm.PERMANENT, MemoryArm.IMMEDIATE} and self.variant is not None:
            raise ValueError("shared memoryless arms cannot carry a variant")
        if self.arm == MemoryArm.MATCHED:
            if self.assigned_duration is not None and self.assigned_duration < 1:
                raise ValueError("a finite assigned duration must be positive")
        elif self.assigned_duration is not None:
            raise ValueError("only the scheduled matched arm may have a duration")

    @property
    def owns_runtime_streams(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "variantId": None if self.variant is None else self.variant.value,
            "assignedDurationOpportunities": self.assigned_duration,
            "collectiveActionBudget": self.action_budget,
            "collectiveEnergyBudget": self.energy_budget,
            "perHelperEnergyBudget": self.per_helper_energy_budget,
            "actionCost": 1,
            "energyCost": 1,
            "energyInterpretation": "abstract_intervention_accounting_unit",
            "runtimeStreams": list(self.owns_runtime_streams),
            "nativeInformationPermission": self.information_permission,
            "nativeLegalPrimitives": list(self.native_legal_primitives),
            "scheduler": self.scheduler,
            "continuation": self.continuation,
            "retry": self.retry,
        }


class LocalMemoryController:
    """Trusted runtime overlay around a separately capability-bounded bank."""

    __slots__ = (
        "contract",
        "selected_identity",
        "selected_position",
        "normal_identities",
        "start_event_index",
        "frozen",
        "recovery_event",
        "recovery_reason",
        "recovery_pending",
        "intervention_pending",
        "intervention_event",
        "intervention_actor",
        "intervention_native_kind",
        "intervention_native_eligible",
        "trigger_component",
        "action_remaining",
        "energy_remaining",
        "helper_energy_spent",
        "bank",
        "pending_observation",
        "pending_freeze_target_block",
        "pending_blocked_direction_side",
        "pending_intervention",
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
        contract: LocalMemoryContract,
        scenario: Scenario,
        selected_identity: str,
        selected_position: int,
        start_event_index: int,
        *,
        retain_audits: bool = False,
    ) -> None:
        contract.validate(scenario, selected_identity)
        self.contract = contract
        self.selected_identity = selected_identity
        self.selected_position = selected_position
        self.normal_identities = frozenset(
            cell.cell_id for cell in scenario.cells if cell.fault == FaultMode.NORMAL
        )
        self.start_event_index = start_event_index
        self.frozen = True
        self.recovery_event: int | None = None
        self.recovery_reason: str | None = None
        self.recovery_pending = False
        self.intervention_pending = False
        self.intervention_event: int | None = None
        self.intervention_actor: str | None = None
        self.intervention_native_kind: str | None = None
        self.intervention_native_eligible: bool | None = None
        self.trigger_component: str | None = None
        self.action_remaining = contract.action_budget
        self.energy_remaining = contract.energy_budget
        self.helper_energy_spent: Counter[str] = Counter()
        self.bank = (
            LocalMemoryBank(contract.variant, sorted(self.normal_identities))
            if contract.arm == MemoryArm.ACTIVE and contract.variant is not None
            else None
        )
        self.pending_observation: LocalObservation | None = None
        self.pending_freeze_target_block = False
        self.pending_blocked_direction_side: int | None = None
        self.pending_intervention = False
        self.ledger: Counter[str] = Counter()
        self.audit_digest = hashlib.sha256(b"E05/S07/local-memory-audit/v1").digest()
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
            raise AssertionError("S07 recovery attempted twice")
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

    def _local_side(self, proposal: Proposal) -> int | None:
        if not self.frozen or proposal.actor_id == self.selected_identity:
            return None
        difference = self.selected_position - proposal.actor_pos
        if difference == -1:
            return 0
        if difference == 1:
            return 1
        return None

    def _budget_eligible(self, proposal: Proposal) -> bool:
        return (
            proposal.actor_id in self.normal_identities
            and proposal.actor_id != self.selected_identity
            and self.action_remaining >= 1
            and self.energy_remaining >= 1
            and self.helper_energy_spent[proposal.actor_id] < 1
        )

    def prepare(
        self, proposal: Proposal, event_index: int
    ) -> tuple[Proposal, tuple[tuple[str, int], ...]]:
        self.ledger["chargedOpportunities"] += 1
        self.proposal_kinds.add(proposal.kind.value)
        self.recovery_pending = False
        self.intervention_pending = False
        self.pending_intervention = False
        self.pending_freeze_target_block = False
        self.pending_blocked_direction_side = None
        side = self._local_side(proposal)
        self.pending_observation = LocalObservation(proposal.actor_id, side)
        self.ledger["localAdjacencySignals"] += int(side is not None)
        if self.frozen:
            self.ledger["freezeExposures"] += 1

        if (
            self.contract.arm == MemoryArm.MATCHED
            and self.frozen
            and self.contract.assigned_duration is not None
            and event_index == self.start_event_index + self.contract.assigned_duration - 1
        ):
            self.intervention_pending = True
            self.pending_intervention = True
            self.trigger_component = "assigned_elapsed_schedule"
        elif (
            self.contract.arm == MemoryArm.IMMEDIATE
            and side is not None
            and self._budget_eligible(proposal)
        ):
            self.intervention_pending = True
            self.pending_intervention = True
            self.trigger_component = "s06_immediate_adjacency"
        elif (
            self.contract.arm == MemoryArm.ACTIVE
            and side is not None
            and self._budget_eligible(proposal)
        ):
            if self.bank is None:
                raise AssertionError("active S07 controller lacks private state")
            ready, component = self.bank.ready(self.pending_observation)
            if ready:
                self.intervention_pending = True
                self.pending_intervention = True
                self.trigger_component = component
        return proposal, ()

    def _spend(
        self,
        proposal: Proposal,
        validation: ValidationDecision,
        event_index: int,
    ) -> None:
        if self.action_remaining < 1 or self.energy_remaining < 1:
            raise AssertionError("S07 intervention overspent its budget")
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
        self.ledger["forgoneEligibleNativeChanges"] += int(validation.eligible_for_commit)
        self.ledger[f"forgoneNative{proposal.kind.value}"] += 1

    def outcome(
        self,
        proposal: Proposal,
        validation: ValidationDecision,
        event_index: int,
    ) -> ValidationDecision:
        if self.intervention_pending:
            self._spend(proposal, validation, event_index)
            self.recovery_pending = True
            if self.contract.arm == MemoryArm.ACTIVE:
                self.ledger["memoryGatedRepairProposals"] += 1
                decision = "rejected_s07_memory_repair_success_pending"
            elif self.contract.arm == MemoryArm.IMMEDIATE:
                self.ledger["immediateReferenceRepairProposals"] += 1
                decision = "rejected_s07_immediate_reference_success_pending"
            else:
                self.ledger["matchedScheduleEvents"] += 1
                decision = "rejected_s07_matched_schedule_recovery_pending"
            self.ledger["repairSuccesses"] += 1
            self._audit(
                "repair_proposal",
                {
                    "eventIndex": event_index,
                    "arm": self.contract.arm.value,
                    "variantId": None
                    if self.contract.variant is None
                    else self.contract.variant.value,
                    "triggerComponent": self.trigger_component,
                    "helperIdentityId": proposal.actor_id,
                    "relativeSide": None
                    if self.pending_observation is None
                    else self.pending_observation.adjacent_side,
                    "nativeProposalKind": proposal.kind.value,
                    "nativeProposalEligible": validation.eligible_for_commit,
                },
            )
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
                    "reason": "s07_actor_frozen",
                    "identityId": proposal.actor_id,
                },
            )
            return ValidationDecision("rejected_s07_actor_frozen", False)
        if target_id == self.selected_identity:
            self.ledger["freezeTargetBlocks"] += 1
            self.pending_freeze_target_block = True
            if proposal.target_pos is None or proposal.target_pos == proposal.actor_pos:
                raise AssertionError("a frozen Swap target lacked a direction")
            self.pending_blocked_direction_side = int(
                proposal.target_pos > proposal.actor_pos
            )
            self._audit(
                "freeze_block",
                {
                    "eventIndex": event_index,
                    "reason": "s07_target_frozen",
                    "identityId": target_id,
                    "relativeSide": None
                    if self.pending_observation is None
                    else self.pending_observation.adjacent_side,
                },
            )
            return ValidationDecision("rejected_s07_target_frozen", False)
        return validation

    def after_batch(
        self,
        scenario: Scenario,
        state: RunState,
        proposals: tuple[Proposal, ...],
        decisions: Mapping[int, str],
        batch_start_index: int,
    ) -> None:
        del scenario, state
        if len(proposals) != 1:
            raise ValueError("S07 local-memory panel is serial")
        proposal = proposals[0]
        decision = decisions[proposal.ordinal]
        if self.bank is not None:
            if self.pending_observation is None:
                raise AssertionError("S07 opportunity lacked local observation")
            native_progress = (
                not self.pending_intervention
                and decision == "accepted"
                and proposal.kind in {ProposalKind.SWAP, ProposalKind.MEMORY_UPDATE}
            )
            before_local = self.bank.states[proposal.actor_id].to_dict(self.bank.enabled)
            self.bank.update(
                LocalOutcome(
                    actor_id=proposal.actor_id,
                    adjacent_side=self.pending_observation.adjacent_side,
                    freeze_target_blocked=self.pending_freeze_target_block,
                    blocked_direction_side=self.pending_blocked_direction_side,
                    native_progress=native_progress,
                )
            )
            after_local = self.bank.states[proposal.actor_id].to_dict(self.bank.enabled)
            if before_local != after_local or self.pending_freeze_target_block:
                self._audit(
                    "private_state_update",
                    {
                        "eventIndex": batch_start_index,
                        "actorIdentityId": proposal.actor_id,
                        "nativeProgress": native_progress,
                        "freezeTargetBlocked": self.pending_freeze_target_block,
                        "stateChanged": before_local != after_local,
                        "priorStateSha256": sha256_json(before_local),
                        "postStateSha256": sha256_json(after_local),
                    },
                )
        if self.recovery_pending and self.frozen:
            duration = batch_start_index - self.start_event_index + 1
            reasons = {
                MemoryArm.ACTIVE: "active_local_memory_repair",
                MemoryArm.IMMEDIATE: "s06_immediate_neighbor_reference",
                MemoryArm.MATCHED: "matched_time_opportunity_energy",
            }
            self._recover(batch_start_index + 1, reasons[self.contract.arm], duration)
        self.recovery_pending = False
        self.intervention_pending = False
        self.pending_intervention = False
        self.pending_freeze_target_block = False
        self.pending_blocked_direction_side = None
        self.pending_observation = None

    def state_dict(self) -> dict[str, Any]:
        bank_state = None if self.bank is None else self.bank.state_dict()
        return {
            "arm": self.contract.arm.value,
            "variantId": None
            if self.contract.variant is None
            else self.contract.variant.value,
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
            "triggerComponent": self.trigger_component,
            "actionBudgetInitial": self.contract.action_budget,
            "actionBudgetRemaining": self.action_remaining,
            "energyBudgetInitial": self.contract.energy_budget,
            "energyBudgetRemaining": self.energy_remaining,
            "helperEnergySpent": dict(sorted(self.helper_energy_spent.items())),
            "privateMemory": bank_state,
            "privateMemorySha256": None if bank_state is None else sha256_json(bank_state),
        }

    def process_ledger(self) -> dict[str, int]:
        fields = (
            "chargedOpportunities",
            "freezeExposures",
            "freezeActorBlocks",
            "freezeTargetBlocks",
            "localAdjacencySignals",
            "memoryGatedRepairProposals",
            "immediateReferenceRepairProposals",
            "matchedScheduleEvents",
            "repairSuccesses",
            "nativeOpportunitiesSuppressed",
            "forgoneEligibleNativeChanges",
            "forgoneNativeNoOp",
            "forgoneNativeSwap",
            "forgoneNativeMemoryUpdate",
            "actionUnitsSpent",
            "energyUnitsSpent",
            "recoveries",
        )
        answer = {field: int(self.ledger[field]) for field in fields}
        complexity_fields = (
            "memoryUpdateOpportunities",
            "componentStateReads",
            "componentStateWrites",
            "saturatingIncrements",
            "componentResets",
            "explicitResetComponents",
        )
        for field in complexity_fields:
            answer[field] = 0 if self.bank is None else int(self.bank.ledger[field])
        for component in (FAILED, BLOCKED, RECENT, TIMER):
            answer[f"readySignal::{component}"] = (
                0 if self.bank is None else int(self.bank.ledger[f"readySignal::{component}"])
            )
            answer[f"componentWrites::{component}"] = (
                0 if self.bank is None else int(self.bank.ledger[f"componentWrites::{component}"])
            )
        return answer


@dataclass(frozen=True, slots=True)
class LocalMemoryRun:
    contract: LocalMemoryContract
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
            "schemaVersion": LOCAL_MEMORY_RUN_SCHEMA_VERSION,
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
    controller: LocalMemoryController,
) -> None:
    if scenario.batch_width != 1 or state.terminal is not None:
        raise ValueError("S07 summary projection requires one live serial opportunity")
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


def run_local_memory_phase(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    post_anchor_occupancy: Sequence[str],
    anchor_lesion_state_hash: str,
    selected_identity: str,
    contract: LocalMemoryContract,
    recovery_budget: int,
    trace_mode: str = "digest",
    retain_process_audits: bool = False,
) -> LocalMemoryRun:
    """Resume an exact S02/S03 state and execute one S07 arm."""

    if trace_mode not in {"full", "digest"}:
        raise ValueError("S07 supports full or digest trace modes")
    if recovery_budget < 1:
        raise ValueError("recovery budget must be positive")
    contract.validate(scenario, selected_identity)
    state = checkpoint.to_run_state(occupancy=post_anchor_occupancy)
    selected_position = state.occupancy.index(selected_identity)
    start_event = state.activation_count
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    initial_hash = state_hash(scenario.scenario_id, state)
    controller = LocalMemoryController(
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
    memory_enabled = contract.arm == MemoryArm.ACTIVE
    active_repairs = process_ledger["memoryGatedRepairProposals"]
    immediate_repairs = process_ledger["immediateReferenceRepairProposals"]
    matched_events = process_ledger["matchedScheduleEvents"]
    validation = {
        **native_checks,
        "activationDeltaMatchesPhase": ledger_delta["activations"] == phase_activations,
        "processOpportunityCountMatchesPhase": process_ledger["chargedOpportunities"]
        == phase_activations,
        "noS07RuntimeStreamsConsumed": not any(
            stream.startswith("local_memory_") for stream in stream_delta
        ),
        "actionBudgetConserved": process_ledger["actionUnitsSpent"]
        + state_record["actionBudgetRemaining"]
        == state_record["actionBudgetInitial"],
        "energyBudgetConserved": process_ledger["energyUnitsSpent"]
        + state_record["energyBudgetRemaining"]
        == state_record["energyBudgetInitial"],
        "helperEnergyConserved": helper_spent == process_ledger["energyUnitsSpent"],
        "costIdentity": process_ledger["actionUnitsSpent"]
        == process_ledger["energyUnitsSpent"]
        == process_ledger["nativeOpportunitiesSuppressed"]
        == active_repairs + immediate_repairs + matched_events,
        "repairSuccessIdentity": process_ledger["repairSuccesses"]
        == active_repairs + immediate_repairs + matched_events
        == process_ledger["recoveries"],
        "atMostOneIntervention": process_ledger["nativeOpportunitiesSuppressed"] <= 1,
        "atMostOneRecovery": process_ledger["recoveries"] <= 1,
        "permanentNeverRecovers": contract.arm != MemoryArm.PERMANENT
        or process_ledger["recoveries"] == 0,
        "activeOwnsMemory": (state_record["privateMemory"] is not None) == memory_enabled,
        "memoryBounds": controller.bank is None or controller.bank.validate_bounds(),
        "memoryOpportunityUpdates": not memory_enabled
        or process_ledger["memoryUpdateOpportunities"] == phase_activations,
        "memorylessZeroStateCost": memory_enabled
        or process_ledger["componentStateReads"]
        + process_ledger["componentStateWrites"]
        == 0,
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
        contract.arm == MemoryArm.MATCHED and contract.assigned_duration is None
    )
    summary = {
        "arm": contract.arm.value,
        "variantId": None if contract.variant is None else contract.variant.value,
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
        "storageBitsPerIdentity": 0
        if controller.bank is None
        else controller.bank.storage_bits_per_identity,
        "totalStorageBits": 0
        if controller.bank is None
        else controller.bank.storage_bits_per_identity * len(scenario.cells),
        "ledgerDelta": ledger_delta,
        "streamCounterDelta": stream_delta,
        "traceMode": trace_mode,
        "retainedEventCount": len(retained),
    }
    return LocalMemoryRun(
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


def exact_replay_local_memory(
    run_result: LocalMemoryRun,
    scenario: Scenario,
    checkpoint: Checkpoint,
    post_anchor_occupancy: Sequence[str],
    recovery_budget: int,
) -> LocalMemoryRun:
    replayed = run_local_memory_phase(
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
        raise AssertionError("S07 local-memory replay mismatch")
    return replayed


def local_memory_case_id(
    s01_pairing_block_id: str,
    timing_condition_id: str,
    anchor_lesion_state_hash: str,
) -> str:
    return "e05mc7:" + sha256_json(
        {
            "s01PairingBlockId": s01_pairing_block_id,
            "timingConditionId": timing_condition_id,
            "anchorLesionStateHash": anchor_lesion_state_hash,
        }
    )
