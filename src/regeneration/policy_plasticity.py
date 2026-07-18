"""E05 S08 costed, failure-triggered engineered policy plasticity.

The controller preserves the S01--S07 scenario, checkpoint, scheduler, native
proposal, ledger, and RNG contracts.  A two-bit actor-local no-progress signal
may spend one ordinary opportunity to activate one prespecified behavior mode.
The primary sham sees and pays for the same trigger, transition, information,
duration, and costs but continues to emit the native proposal rule.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import math
from typing import Any, Mapping, Sequence

import jsonschema

from causal_simulator.action_interface import (
    CommonActionInterface,
    build_policy_native_observation,
    propose_from_observation,
)
from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
)
from reference_simulator.engine import EMPTY_DIGEST, evaluate_terminal
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
from reference_simulator.scheduler import scheduled_actor, scheduled_side
from reference_simulator.transition_primitives import (
    ValidationDecision,
    commit_proposal,
    cost_delta,
    ledger_identity,
    validate_proposal,
)

from .tasks import Checkpoint, occupancy_values, strict_unequal_inversions


BENCHMARK_VERSION = "E05-policy-plasticity-v1"
PLASTICITY_SPEC_SCHEMA_VERSION = "e05.s08.policy-plasticity-spec.v1"
PLASTICITY_RUN_SCHEMA_VERSION = "e05.s08.policy-plasticity-run.v1"
RECOVERY_DONOR_VARIANT = "all_components_union_v1"
TRIGGER_THRESHOLD = 3
REVERSIBLE_MODE_OPPORTUNITIES = 8


class PlasticityArm(str, Enum):
    ACTIVE = "active_policy_plasticity"
    SHAM = "matched_nonplastic_sham"
    REFERENCE = "nonplastic_recovery_only_reference"


class PlasticityMechanism(str, Enum):
    ALGOTYPE = "algotype_switch"
    RADIUS = "sensing_radius_expansion"
    TARGET = "alternate_target_selection"
    EXPLORE = "exploratory_mode"


class Durability(str, Enum):
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class PlasticityVariant(str, Enum):
    ALGOTYPE_REV = "algotype_switch_reversible_v1"
    ALGOTYPE_IRREV = "algotype_switch_irreversible_v1"
    RADIUS_REV = "sensing_radius_expansion_reversible_v1"
    RADIUS_IRREV = "sensing_radius_expansion_irreversible_v1"
    TARGET_REV = "alternate_target_reversible_v1"
    TARGET_IRREV = "alternate_target_irreversible_v1"
    EXPLORE_REV = "exploratory_mode_reversible_v1"
    EXPLORE_IRREV = "exploratory_mode_irreversible_v1"


VARIANT_PROFILE: dict[PlasticityVariant, tuple[PlasticityMechanism, Durability]] = {
    PlasticityVariant.ALGOTYPE_REV: (
        PlasticityMechanism.ALGOTYPE,
        Durability.REVERSIBLE,
    ),
    PlasticityVariant.ALGOTYPE_IRREV: (
        PlasticityMechanism.ALGOTYPE,
        Durability.IRREVERSIBLE,
    ),
    PlasticityVariant.RADIUS_REV: (
        PlasticityMechanism.RADIUS,
        Durability.REVERSIBLE,
    ),
    PlasticityVariant.RADIUS_IRREV: (
        PlasticityMechanism.RADIUS,
        Durability.IRREVERSIBLE,
    ),
    PlasticityVariant.TARGET_REV: (
        PlasticityMechanism.TARGET,
        Durability.REVERSIBLE,
    ),
    PlasticityVariant.TARGET_IRREV: (
        PlasticityMechanism.TARGET,
        Durability.IRREVERSIBLE,
    ),
    PlasticityVariant.EXPLORE_REV: (
        PlasticityMechanism.EXPLORE,
        Durability.REVERSIBLE,
    ),
    PlasticityVariant.EXPLORE_IRREV: (
        PlasticityMechanism.EXPLORE,
        Durability.IRREVERSIBLE,
    ),
}


PLASTICITY_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s08/policy-plasticity-spec.schema.json",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenAtUtc",
        "implementationStateAtFreeze",
        "frozenQuestion",
        "s07Handoff",
        "inherits",
        "failureTrigger",
        "durability",
        "mechanisms",
        "variants",
        "controls",
        "recoveryScheduleControl",
        "costs",
        "pairingAndStreams",
        "intactStability",
        "successCriteria",
        "validationPanel",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": PLASTICITY_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S08"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "mechanisms": {"type": "array", "minItems": 4, "maxItems": 4},
        "variants": {"type": "array", "minItems": 8, "maxItems": 8},
        "controls": {"type": "array", "minItems": 2, "maxItems": 2},
    },
}


def validate_plasticity_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(PLASTICITY_SPEC_SCHEMA).validate(specification)
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "lesionSpecSchemaVersion": "e05.s03.lesion-spec.v1",
        "dynamicSpecSchemaVersion": "e05.s04.dynamic-fault-spec.v1",
        "nudgeSpecSchemaVersion": "e05.s05.nudge-recovery-spec.v1",
        "rescueSpecSchemaVersion": "e05.s06.assisted-rescue-spec.v1",
        "memorySpecSchemaVersion": "e05.s07.local-memory-spec.v1",
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "runtimeAnchorLesion": "segment_reversal_central_v1",
        "runtimeScenarioBoundary": "original_s01_scenario_id_runtime_overlay",
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"inherited S01-S07 contract changed: {key}")
    if specification["pairingAndStreams"].get("runtimeStreams") != []:
        raise ValueError("S08 plasticity must remain runtime-stream-free")
    trigger = specification["failureTrigger"]
    if trigger.get("triggerId") != "actor_local_no_progress_cap3_first_signal_v1":
        raise ValueError("S08 trigger changed")
    variant_rows = {item["variantId"]: item for item in specification["variants"]}
    if set(variant_rows) != {item.value for item in PlasticityVariant}:
        raise ValueError("S08 variant set changed")
    for variant, (mechanism, durability) in VARIANT_PROFILE.items():
        row = variant_rows[variant.value]
        if (row.get("mechanismId"), row.get("durability")) != (
            mechanism.value,
            durability.value,
        ):
            raise ValueError(f"S08 variant profile changed: {variant.value}")
    panel = specification["validationPanel"]
    expected_counts = {
        "injuryCheckpointCount": 384,
        "confirmatoryInjuryCheckpointCount": 192,
        "stabilityCheckpointCount": 48,
        "variantCount": 8,
        "plannedInjuryActiveRuns": 3072,
        "plannedInjuryMatchedShamRuns": 3072,
        "plannedInjurySharedReferenceRuns": 384,
        "plannedStabilityActiveRuns": 384,
        "plannedStabilityMatchedShamRuns": 384,
        "plannedStabilitySharedReferenceRuns": 48,
        "plannedRunCount": 7344,
        "plannedPrimaryRecoveryPairs": 1536,
        "plannedPrimaryStabilityPairs": 384,
    }
    for key, value in expected_counts.items():
        if panel.get(key) != value:
            raise ValueError(f"S08 planned panel changed: {key}")


class LocalFailureBank:
    """Two-bit actor-local counters with no scenario, clock, target, or outcome access."""

    __slots__ = ("states", "ledger")

    def __init__(self, actor_ids: Sequence[str]) -> None:
        self.states = {actor_id: 0 for actor_id in actor_ids}
        self.ledger: Counter[str] = Counter()

    def ready(self, actor_id: str) -> bool:
        self.ledger["failureStateReads"] += 1
        self.ledger["failureReadinessComparisons"] += 1
        return self.states[actor_id] == TRIGGER_THRESHOLD

    def update(self, actor_id: str, local_progress: bool) -> None:
        before = self.states[actor_id]
        self.ledger["failureStateReads"] += 1
        after = 0 if local_progress else min(TRIGGER_THRESHOLD, before + 1)
        if not local_progress and before == TRIGGER_THRESHOLD:
            self.ledger["failureCounterSaturations"] += 1
        if after != before:
            self.states[actor_id] = after
            self.ledger["failureStateWrites"] += 1

    def state_dict(self) -> dict[str, int]:
        return {key: self.states[key] for key in sorted(self.states)}

    def validate_bounds(self) -> bool:
        return all(0 <= value <= TRIGGER_THRESHOLD for value in self.states.values())


@dataclass(frozen=True, slots=True)
class PlasticityContract:
    arm: PlasticityArm
    variant: PlasticityVariant | None
    phase: str
    assigned_recovery_duration: int | None = None
    scheduler: str = "uniform_random_activation"
    architecture: str = "distributed_local"
    continuation: str = "skip_and_continue"
    retry: str = "no_retry"
    native_information_permission: str = "policy_native_local"
    native_legal_primitives: tuple[str, ...] = ("NoOp", "Swap", "MemoryUpdate")

    @property
    def mechanism(self) -> PlasticityMechanism | None:
        return None if self.variant is None else VARIANT_PROFILE[self.variant][0]

    @property
    def durability(self) -> Durability | None:
        return None if self.variant is None else VARIANT_PROFILE[self.variant][1]

    @property
    def reversible(self) -> bool:
        return self.durability == Durability.REVERSIBLE

    @property
    def transition_action_budget(self) -> int:
        if self.arm == PlasticityArm.REFERENCE:
            return 0
        return 2 if self.reversible else 1

    def validate(self, scenario: Scenario, selected_identity: str | None) -> None:
        ArchitectureExecutionContract.distributed_local().validate_policy_action_boundary(
            scenario
        )
        if any(cell.fault != FaultMode.NORMAL for cell in scenario.cells):
            raise ValueError("S08 requires the original normal S01 scenario")
        if self.scheduler != "uniform_random_activation" or self.architecture != "distributed_local":
            raise ValueError("S08 preserves scheduler and architecture")
        if self.continuation != "skip_and_continue" or self.retry != "no_retry":
            raise ValueError("S08 preserves continuation and no-retry")
        if self.native_information_permission != "policy_native_local":
            raise ValueError("S08 cannot change native information permission")
        if self.native_legal_primitives != ("NoOp", "Swap", "MemoryUpdate"):
            raise ValueError("S08 cannot change the legal primitive set")
        if self.arm in {PlasticityArm.ACTIVE, PlasticityArm.SHAM} and self.variant is None:
            raise ValueError("active and sham S08 arms require a variant")
        if self.arm == PlasticityArm.REFERENCE and self.variant is not None:
            raise ValueError("the shared reference cannot carry a plasticity variant")
        if self.phase not in {"injury", "stability"}:
            raise ValueError("S08 phase must be injury or stability")
        if self.phase == "injury":
            if selected_identity not in scenario.cell_map:
                raise ValueError("injury S08 requires a known selected identity")
            if self.assigned_recovery_duration is None or self.assigned_recovery_duration < 1:
                raise ValueError("injury S08 requires a positive prior-S07 schedule")
        elif selected_identity is not None or self.assigned_recovery_duration is not None:
            raise ValueError("stability S08 has no injury identity or recovery schedule")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "variantId": None if self.variant is None else self.variant.value,
            "mechanismId": None if self.mechanism is None else self.mechanism.value,
            "durability": None if self.durability is None else self.durability.value,
            "phase": self.phase,
            "assignedRecoveryDuration": self.assigned_recovery_duration,
            "transitionActionBudget": self.transition_action_budget,
            "runtimeStreams": [],
            "scheduler": self.scheduler,
            "architecture": self.architecture,
            "nativeInformationPermission": self.native_information_permission,
            "nativeLegalPrimitives": list(self.native_legal_primitives),
            "continuation": self.continuation,
            "retry": self.retry,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveMeter:
    reads: int
    comparisons: int
    maximum_radius: int
    visible_positions: tuple[int, ...]


class AdaptiveObservationGateway:
    """Trusted projection that exposes only the frozen mechanism-specific view."""

    __slots__ = ("__scenario", "__state")

    def __init__(self, scenario: Scenario, state: RunState) -> None:
        self.__scenario = scenario
        self.__state = state

    def _noop(
        self,
        actor_id: str,
        actor_pos: int,
        reason: str,
        reads: int,
        comparisons: int,
    ) -> Proposal:
        return Proposal(
            ProposalKind.NO_OP,
            actor_id,
            actor_pos,
            reason=reason,
            observation_reads=reads,
            value_comparisons=comparisons,
        )

    def _neighbor(self, position: int) -> tuple[str, Any] | None:
        if not 0 <= position < len(self.__state.occupancy):
            return None
        identity = self.__state.occupancy[position]
        return identity, self.__scenario.cell_map[identity]

    @staticmethod
    def _strict_inversion(
        actor_value: int | float,
        target_value: int | float,
        actor_pos: int,
        target_pos: int,
        direction: Direction,
    ) -> bool:
        if target_pos < actor_pos:
            return (
                actor_value < target_value
                if direction == Direction.ASCENDING
                else actor_value > target_value
            )
        return (
            actor_value > target_value
            if direction == Direction.ASCENDING
            else actor_value < target_value
        )

    def _alt_scenario(self, actor_id: str, policy: Policy) -> Scenario:
        cells = tuple(
            replace(cell, policy=policy) if cell.cell_id == actor_id else cell
            for cell in self.__scenario.cells
        )
        # This is a proposal-only capability view.  It is never validated,
        # scheduled, hashed, committed, or used as an RNG root.
        return replace(self.__scenario, cells=cells)

    def _algotype(
        self,
        actor_id: str,
        mode_count: int,
        *,
        emit: bool,
    ) -> tuple[Proposal | None, AdaptiveMeter]:
        actor = self.__scenario.cell_map[actor_id]
        alternate = {
            Policy.BUBBLE: Policy.INSERTION,
            Policy.INSERTION: Policy.BUBBLE,
            Policy.SELECTION: Policy.BUBBLE,
        }[actor.policy]
        alternate_scenario = self._alt_scenario(actor_id, alternate)
        side = None
        if alternate == Policy.BUBBLE:
            side = "left" if mode_count % 2 == 0 else "right"
        observation, meter, gateway = build_policy_native_observation(
            alternate_scenario,
            self.__state,
            actor_id,
            side=side,
        )
        visible = tuple(
            sorted(
                {
                    item.position
                    for item in getattr(observation, "prefix_read", ())
                }
                | (
                    {observation.target.position}
                    if getattr(observation, "target", None) is not None
                    else set()
                )
                | (
                    {observation.left_target.position}
                    if getattr(observation, "left_target", None) is not None
                    else set()
                )
            )
        )
        actor_pos = self.__state.occupancy.index(actor_id)
        maximum_radius = max((abs(item - actor_pos) for item in visible), default=0)
        if not emit:
            return None, AdaptiveMeter(
                meter.observation_reads,
                meter.value_comparisons + 1,
                maximum_radius,
                visible,
            )
        proposal = propose_from_observation(observation, meter)
        observed = (
            gateway.observed_identity_at(proposal.target_pos)
            if proposal.kind == ProposalKind.SWAP and proposal.target_pos is not None
            else None
        )
        proposal = replace(
            proposal,
            observed_target_id=observed,
            reason=f"s08_algotype_{actor.policy.value}_to_{alternate.value}::{proposal.reason}",
        )
        return proposal, AdaptiveMeter(
            proposal.observation_reads,
            proposal.value_comparisons,
            maximum_radius,
            visible,
        )

    def _radius(
        self, actor_id: str, *, emit: bool
    ) -> tuple[Proposal | None, AdaptiveMeter]:
        actor = self.__scenario.cell_map[actor_id]
        actor_pos = self.__state.occupancy.index(actor_id)
        reads = 1
        comparisons = 0
        visible: list[int] = []
        records: dict[int, tuple[str, Any]] = {}
        for offset in (-2, -1, 1, 2):
            position = actor_pos + offset
            record = self._neighbor(position)
            if record is not None:
                reads += 1
                visible.append(position)
                records[offset] = record
        chosen: int | None = None
        for offset in (-2, 2):
            if offset not in records:
                continue
            comparisons += 1
            _, target = records[offset]
            if self._strict_inversion(
                actor.value,
                target.value,
                actor_pos,
                actor_pos + offset,
                actor.direction,
            ):
                chosen = -1 if offset < 0 else 1
                break
        proposal: Proposal | None = None
        if emit:
            if chosen is None or chosen not in records:
                proposal = self._noop(
                    actor_id,
                    actor_pos,
                    "s08_radius2_no_two_hop_violation",
                    reads,
                    comparisons,
                )
            else:
                target_id, _ = records[chosen]
                proposal = Proposal(
                    ProposalKind.SWAP,
                    actor_id,
                    actor_pos,
                    target_pos=actor_pos + chosen,
                    reason="s08_radius2_step_toward_two_hop_violation",
                    observation_reads=reads,
                    value_comparisons=comparisons,
                    observed_target_id=target_id,
                )
        return proposal, AdaptiveMeter(reads, comparisons, 2, tuple(visible))

    def _alternate_target(
        self,
        actor_id: str,
        native_side: str | None,
        *,
        emit: bool,
    ) -> tuple[Proposal | None, AdaptiveMeter]:
        actor = self.__scenario.cell_map[actor_id]
        actor_pos = self.__state.occupancy.index(actor_id)
        if actor.policy == Policy.BUBBLE:
            if native_side not in {"left", "right"}:
                raise ValueError("Bubble alternate target requires its scheduled side")
            side = "right" if native_side == "left" else "left"
        elif actor.policy == Policy.INSERTION:
            side = "right"
        else:
            cursor = self.__state.selection_cursors[actor_id]
            side = "right" if cursor <= actor_pos else "left"
        target_pos = actor_pos + (-1 if side == "left" else 1)
        target_record = self._neighbor(target_pos)
        reads = 1 + int(target_record is not None)
        comparisons = 0
        proposal: Proposal | None = None
        if target_record is not None:
            comparisons = 1
            target_id, target = target_record
            inversion = self._strict_inversion(
                actor.value,
                target.value,
                actor_pos,
                target_pos,
                actor.direction,
            )
            if emit and inversion:
                proposal = Proposal(
                    ProposalKind.SWAP,
                    actor_id,
                    actor_pos,
                    target_pos=target_pos,
                    reason=f"s08_alternate_target_{side}_strict_inversion",
                    observation_reads=reads,
                    value_comparisons=comparisons,
                    observed_target_id=target_id,
                )
        if emit and proposal is None:
            proposal = self._noop(
                actor_id,
                actor_pos,
                "s08_alternate_target_boundary_or_ordered",
                reads,
                comparisons,
            )
        visible = () if target_record is None else (target_pos,)
        return proposal, AdaptiveMeter(reads, comparisons, 1, visible)

    def _explore(
        self, actor_id: str, mode_count: int, *, emit: bool
    ) -> tuple[Proposal | None, AdaptiveMeter]:
        actor_pos = self.__state.occupancy.index(actor_id)
        side = -1 if mode_count % 2 == 0 else 1
        target_pos = actor_pos + side
        target_record = self._neighbor(target_pos)
        reads = 1 + int(target_record is not None)
        proposal: Proposal | None = None
        if emit:
            if target_record is None:
                proposal = self._noop(
                    actor_id,
                    actor_pos,
                    "s08_exploration_boundary",
                    reads,
                    0,
                )
            else:
                target_id, target = target_record
                if target.fault != FaultMode.NORMAL:
                    proposal = self._noop(
                        actor_id,
                        actor_pos,
                        "s08_exploration_non_normal_neighbor",
                        reads,
                        0,
                    )
                else:
                    proposal = Proposal(
                        ProposalKind.SWAP,
                        actor_id,
                        actor_pos,
                        target_pos=target_pos,
                        reason="s08_exploration_adjacent_swap",
                        observation_reads=reads,
                        value_comparisons=0,
                        observed_target_id=target_id,
                    )
        visible = () if target_record is None else (target_pos,)
        return proposal, AdaptiveMeter(reads, 0, 1, visible)

    def observe_or_propose(
        self,
        actor_id: str,
        mechanism: PlasticityMechanism,
        mode_count: int,
        native_side: str | None,
        *,
        emit: bool,
    ) -> tuple[Proposal | None, AdaptiveMeter]:
        if mechanism == PlasticityMechanism.ALGOTYPE:
            return self._algotype(actor_id, mode_count, emit=emit)
        if mechanism == PlasticityMechanism.RADIUS:
            return self._radius(actor_id, emit=emit)
        if mechanism == PlasticityMechanism.TARGET:
            return self._alternate_target(actor_id, native_side, emit=emit)
        if mechanism == PlasticityMechanism.EXPLORE:
            return self._explore(actor_id, mode_count, emit=emit)
        raise AssertionError(mechanism)


class PlasticityController:
    """One-cycle mode controller plus inherited runtime freeze/recovery overlay."""

    __slots__ = (
        "contract",
        "scenario",
        "selected_identity",
        "selected_position",
        "start_event_index",
        "frozen",
        "recovery_event",
        "recovery_pending",
        "mode_actor",
        "mode_active",
        "ever_transitioned",
        "mode_opportunities",
        "transition_event",
        "reversion_event",
        "pending_kind",
        "pending_actor",
        "failure_bank",
        "transition_actions_remaining",
        "ledger",
        "audit_digest",
        "audit_count",
        "retained_audits",
        "retain_audits",
        "transitions",
        "pending_proposal_mode",
        "proposal_kinds",
    )

    def __init__(
        self,
        contract: PlasticityContract,
        scenario: Scenario,
        selected_identity: str | None,
        start_event_index: int,
        *,
        retain_audits: bool = False,
    ) -> None:
        contract.validate(scenario, selected_identity)
        self.contract = contract
        self.scenario = scenario
        self.selected_identity = selected_identity
        self.selected_position = (
            None
            if selected_identity is None
            else scenario.initial_occupancy.index(selected_identity)
        )
        self.start_event_index = start_event_index
        self.frozen = selected_identity is not None
        self.recovery_event: int | None = None
        self.recovery_pending = False
        self.mode_actor: str | None = None
        self.mode_active = False
        self.ever_transitioned = False
        self.mode_opportunities = 0
        self.transition_event: int | None = None
        self.reversion_event: int | None = None
        self.pending_kind: str | None = None
        self.pending_actor: str | None = None
        self.failure_bank = (
            LocalFailureBank([cell.cell_id for cell in scenario.cells])
            if contract.arm in {PlasticityArm.ACTIVE, PlasticityArm.SHAM}
            else None
        )
        self.transition_actions_remaining = contract.transition_action_budget
        self.ledger: Counter[str] = Counter()
        self.audit_digest = hashlib.sha256(b"E05/S08/plasticity-audit/v1").digest()
        self.audit_count = 0
        self.retained_audits: list[dict[str, Any]] = []
        self.retain_audits = retain_audits
        self.transitions: list[dict[str, Any]] = []
        self.pending_proposal_mode = False
        self.proposal_kinds: set[str] = set()

    def set_selected_position(self, occupancy: Sequence[str]) -> None:
        if self.selected_identity is not None:
            self.selected_position = occupancy.index(self.selected_identity)

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

    def reversion_due(self, actor_id: str) -> bool:
        return bool(
            self.mode_active
            and self.contract.reversible
            and actor_id == self.mode_actor
            and self.mode_opportunities >= REVERSIBLE_MODE_OPPORTUNITIES
        )

    def use_plastic_mode(self, actor_id: str) -> bool:
        return bool(
            self.mode_active
            and actor_id == self.mode_actor
            and not self.reversion_due(actor_id)
        )

    def record_mode_meter(self, meter: AdaptiveMeter) -> None:
        self.pending_proposal_mode = True
        self.ledger["modeProposalOpportunities"] += 1
        self.ledger["adaptiveSensingReads"] += meter.reads
        self.ledger["adaptiveValueComparisons"] += meter.comparisons
        self.ledger["adaptiveComputeEnergy"] += 1
        self.ledger["adaptiveSensingEnergy"] += meter.reads
        self.ledger["maximumSensingRadius"] = max(
            self.ledger["maximumSensingRadius"], meter.maximum_radius
        )
        self._audit(
            "mode_observation",
            {
                "actorIdentityId": self.mode_actor,
                "modeOpportunityOrdinal": self.mode_opportunities,
                "reads": meter.reads,
                "comparisons": meter.comparisons,
                "maximumRadius": meter.maximum_radius,
                "visiblePositions": list(meter.visible_positions),
            },
        )

    def prepare(
        self, proposal: Proposal, event_index: int
    ) -> tuple[Proposal, tuple[tuple[str, int], ...]]:
        self.ledger["chargedOpportunities"] += 1
        self.proposal_kinds.add(proposal.kind.value)
        self.pending_kind = None
        self.pending_actor = proposal.actor_id
        self.recovery_pending = False
        recovery_due = bool(
            self.frozen
            and self.contract.assigned_recovery_duration is not None
            and event_index
            == self.start_event_index + self.contract.assigned_recovery_duration - 1
        )
        if recovery_due:
            self.pending_kind = "recovery"
        elif self.reversion_due(proposal.actor_id):
            self.pending_kind = "reversion"
        elif (
            self.failure_bank is not None
            and not self.ever_transitioned
            and self.failure_bank.ready(proposal.actor_id)
        ):
            self.pending_kind = "transition"
        return proposal, ()

    def _spend_suppressed_event(
        self,
        proposal: Proposal,
        validation: ValidationDecision,
        kind: str,
    ) -> None:
        self.ledger["actionUnitsSpent"] += 1
        self.ledger["transitionAndRecoveryEnergy"] += 1
        self.ledger["nativeOpportunitiesSuppressed"] += 1
        self.ledger["forgoneEligibleNativeChanges"] += int(
            validation.eligible_for_commit
        )
        self.ledger[f"forgoneNative{proposal.kind.value}"] += 1
        self.ledger[f"suppressed::{kind}"] += 1
        if kind in {"transition", "reversion"}:
            if self.transition_actions_remaining < 1:
                raise AssertionError("S08 plasticity transition budget overspent")
            self.transition_actions_remaining -= 1

    def outcome(
        self,
        proposal: Proposal,
        validation: ValidationDecision,
        event_index: int,
    ) -> ValidationDecision:
        if self.pending_kind is not None:
            self._spend_suppressed_event(proposal, validation, self.pending_kind)
            decisions = {
                "recovery": "rejected_s08_fixed_recovery_pending",
                "transition": "rejected_s08_mode_transition",
                "reversion": "rejected_s08_mode_reversion",
            }
            return ValidationDecision(decisions[self.pending_kind], False)
        if not self.frozen or not validation.eligible_for_commit:
            return validation
        if proposal.actor_id == self.selected_identity:
            self.ledger["freezeActorBlocks"] += 1
            return ValidationDecision("rejected_s08_actor_frozen", False)
        if (
            proposal.kind == ProposalKind.SWAP
            and proposal.observed_target_id == self.selected_identity
        ):
            self.ledger["freezeTargetBlocks"] += 1
            return ValidationDecision("rejected_s08_target_frozen", False)
        return validation

    def after_opportunity(
        self,
        proposal: Proposal,
        decision: str,
        event_index: int,
    ) -> None:
        pending = self.pending_kind
        if pending == "recovery":
            self.frozen = False
            self.recovery_event = event_index + 1
            self.ledger["recoveries"] += 1
            self._transition(
                "selected_identity_recovered",
                event_index + 1,
                assignedDuration=self.contract.assigned_recovery_duration,
            )
        elif pending == "transition":
            self.mode_actor = proposal.actor_id
            self.mode_active = True
            self.ever_transitioned = True
            self.mode_opportunities = 0
            self.transition_event = event_index
            self.ledger["modeTransitions"] += 1
            self._transition(
                "plastic_mode_activated",
                event_index + 1,
                actorIdentityId=proposal.actor_id,
                mechanismId=self.contract.mechanism.value,
                durability=self.contract.durability.value,
                triggerPriorNoProgressCount=TRIGGER_THRESHOLD,
            )
        elif pending == "reversion":
            self.mode_active = False
            self.reversion_event = event_index
            self.ledger["modeReversions"] += 1
            self._transition(
                "plastic_mode_reverted",
                event_index + 1,
                actorIdentityId=proposal.actor_id,
                governedModeOpportunities=self.mode_opportunities,
            )
        elif self.failure_bank is not None and not self.ever_transitioned:
            local_progress = bool(
                decision == "accepted"
                and proposal.kind in {ProposalKind.SWAP, ProposalKind.MEMORY_UPDATE}
            )
            self.failure_bank.update(proposal.actor_id, local_progress)
        if self.pending_proposal_mode and pending is None:
            if proposal.actor_id != self.mode_actor:
                raise AssertionError("S08 mode proposal actor changed")
            self.mode_opportunities += 1
        self.pending_kind = None
        self.pending_actor = None
        self.pending_proposal_mode = False

    def storage_bits(self) -> tuple[int, int]:
        n = len(self.scenario.cells)
        if self.contract.arm == PlasticityArm.REFERENCE:
            return 0, 0
        tag_bits = math.ceil(math.log2(max(2, n)))
        mechanism_bits = {
            PlasticityMechanism.ALGOTYPE: 2,
            PlasticityMechanism.RADIUS: 0,
            PlasticityMechanism.TARGET: 0,
            PlasticityMechanism.EXPLORE: 1,
        }[self.contract.mechanism]
        system_overhead = 2 + tag_bits + mechanism_bits
        if self.contract.reversible:
            system_overhead += 4
        return 2, 2 * n + system_overhead

    def state_dict(self) -> dict[str, Any]:
        bits_per_identity, total_bits = self.storage_bits()
        bank = None if self.failure_bank is None else self.failure_bank.state_dict()
        return {
            "arm": self.contract.arm.value,
            "variantId": None
            if self.contract.variant is None
            else self.contract.variant.value,
            "selectedIdentityId": self.selected_identity,
            "frozen": self.frozen,
            "recoveryEventIndex": self.recovery_event,
            "transitionEventIndex": self.transition_event,
            "reversionEventIndex": self.reversion_event,
            "modeActorId": self.mode_actor,
            "modeActive": self.mode_active,
            "everTransitioned": self.ever_transitioned,
            "modeOpportunities": self.mode_opportunities,
            "transitionActionBudgetInitial": self.contract.transition_action_budget,
            "transitionActionBudgetRemaining": self.transition_actions_remaining,
            "failureCounters": bank,
            "failureCountersSha256": None if bank is None else sha256_json(bank),
            "storageBitsPerIdentity": bits_per_identity,
            "totalStorageBits": total_bits,
        }

    def process_ledger(self) -> dict[str, int]:
        fields = (
            "chargedOpportunities",
            "freezeActorBlocks",
            "freezeTargetBlocks",
            "recoveries",
            "modeTransitions",
            "modeReversions",
            "modeProposalOpportunities",
            "adaptiveSensingReads",
            "adaptiveValueComparisons",
            "adaptiveComputeEnergy",
            "adaptiveSensingEnergy",
            "maximumSensingRadius",
            "actionUnitsSpent",
            "transitionAndRecoveryEnergy",
            "nativeOpportunitiesSuppressed",
            "forgoneEligibleNativeChanges",
            "forgoneNativeNoOp",
            "forgoneNativeSwap",
            "forgoneNativeMemoryUpdate",
            "suppressed::recovery",
            "suppressed::transition",
            "suppressed::reversion",
        )
        answer = {field: int(self.ledger[field]) for field in fields}
        bank_fields = (
            "failureStateReads",
            "failureReadinessComparisons",
            "failureStateWrites",
            "failureCounterSaturations",
        )
        for field in bank_fields:
            answer[field] = (
                0 if self.failure_bank is None else int(self.failure_bank.ledger[field])
            )
        answer["totalAbstractEnergy"] = (
            answer["transitionAndRecoveryEnergy"]
            + answer["adaptiveComputeEnergy"]
            + answer["adaptiveSensingEnergy"]
        )
        return answer


class PlasticityProposalRouter:
    """Emit exactly one native or active-mode proposal per opportunity."""

    __slots__ = ("controller", "native_router")

    def __init__(self, controller: PlasticityController) -> None:
        self.controller = controller
        self.native_router = ArchitectureProposalRouter(
            ArchitectureExecutionContract.distributed_local(),
            action_interface=CommonActionInterface(),
        )

    def proposal_for(
        self,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        side: str | None = None,
    ) -> Proposal:
        if not self.controller.use_plastic_mode(actor_id):
            return self.native_router.proposal_for(
                scenario, state, actor_id, side=side
            )
        mechanism = self.controller.contract.mechanism
        assert mechanism is not None
        gateway = AdaptiveObservationGateway(scenario, state)
        emit = self.controller.contract.arm == PlasticityArm.ACTIVE
        proposal, meter = gateway.observe_or_propose(
            actor_id,
            mechanism,
            self.controller.mode_opportunities,
            side,
            emit=emit,
        )
        self.controller.record_mode_meter(meter)
        if emit:
            assert proposal is not None
            return proposal
        return self.native_router.proposal_for(scenario, state, actor_id, side=side)


@dataclass(frozen=True, slots=True)
class PlasticityRun:
    contract: PlasticityContract
    source_scenario_id: str
    source_checkpoint_hash: str
    anchor_lesion_state_hash: str | None
    selected_identity: str | None
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
            "schemaVersion": PLASTICITY_RUN_SCHEMA_VERSION,
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


def _delta(final: Mapping[str, int], initial: Mapping[str, int]) -> dict[str, int]:
    return {key: int(final.get(key, 0)) - int(initial.get(key, 0)) for key in final}


def _execute_opportunity(
    scenario: Scenario,
    state: RunState,
    router: PlasticityProposalRouter,
    controller: PlasticityController,
    *,
    evaluate_stop: bool,
    retain_event: bool,
) -> Mapping[str, Any] | None:
    if scenario.batch_width != 1 or state.terminal is not None:
        raise ValueError("S08 execution requires one live serial opportunity")
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
    controller.after_opportunity(proposal, decision, event_index)
    if evaluate_stop and (changed or state.activation_count >= scenario.max_activations):
        state.terminal = evaluate_terminal(scenario, state)
    if not retain_event:
        return None
    return {
        "eventIndex": event_index,
        "actorId": actor_id,
        "proposal": proposal.to_dict(),
        "decision": decision,
        "changed": changed,
        "modeActiveAfter": controller.mode_active,
        "modeActorIdAfter": controller.mode_actor,
        "frozenAfter": controller.frozen,
    }


def run_plasticity_phase(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    occupancy: Sequence[str],
    anchor_lesion_state_hash: str | None,
    selected_identity: str | None,
    contract: PlasticityContract,
    phase_budget: int,
    retain_trace: bool = False,
) -> PlasticityRun:
    """Execute one injury or intact-stability S08 arm from an exact checkpoint."""

    if phase_budget < 1:
        raise ValueError("S08 phase budget must be positive")
    contract.validate(scenario, selected_identity)
    state = checkpoint.to_run_state(occupancy=occupancy)
    start_event = state.activation_count
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    initial_hash = state_hash(scenario.scenario_id, state)
    controller = PlasticityController(
        contract,
        scenario,
        selected_identity,
        start_event,
        retain_audits=retain_trace,
    )
    controller.set_selected_position(state.occupancy)
    process_initial = controller.state_dict()
    router = PlasticityProposalRouter(controller)
    retained: list[Mapping[str, Any]] = []
    digest = bytes.fromhex(EMPTY_DIGEST)
    values = occupancy_values(scenario, state.occupancy)
    current_distance = strict_unequal_inversions(values, scenario.cells[0].direction)
    initial_distance = current_distance
    distance_auc = 0
    maximum_distance = current_distance
    any_target_departure = current_distance > 0
    stability = contract.phase == "stability"
    state.terminal = None if stability else evaluate_terminal(scenario, state)
    while state.terminal is None and state.activation_count - start_event < phase_budget:
        accepted_before = state.ledger["acceptedSwaps"]
        event = _execute_opportunity(
            scenario,
            state,
            router,
            controller,
            evaluate_stop=not stability,
            retain_event=retain_trace,
        )
        if event is not None:
            retained.append(event)
            digest = hashlib.sha256(digest + canonical_json_bytes(event)).digest()
        if state.ledger["acceptedSwaps"] != accepted_before:
            values = occupancy_values(scenario, state.occupancy)
            current_distance = strict_unequal_inversions(
                values, scenario.cells[0].direction
            )
        distance_auc += current_distance
        maximum_distance = max(maximum_distance, current_distance)
        any_target_departure = any_target_departure or current_distance > 0
    if stability:
        state.terminal = "stability_probe_complete"
    elif state.terminal is None:
        state.terminal = "phase_event_budget"
    phase_activations = state.activation_count - start_event
    ledger_delta = _delta(state.ledger, initial_ledger)
    stream_delta = _delta(state.stream_counters, initial_streams)
    process_final = controller.state_dict()
    process_ledger = controller.process_ledger()
    opportunity_validation = {
        "oneProposalPerActivation": ledger_delta["activations"]
        == ledger_delta["proposals"]
        == phase_activations,
        "proposalPartition": ledger_delta["proposals"]
        == ledger_delta["noOps"]
        + ledger_delta["rejections"]
        + ledger_delta["memoryUpdates"]
        + ledger_delta["acceptedSwaps"]
        + ledger_delta["conflictLosses"],
        "swapDisplacement": ledger_delta["displacedCells"]
        == 2 * ledger_delta["acceptedSwaps"],
        "nativeLedgerIdentity": ledger_identity(state.ledger),
        "chargedOpportunityIdentity": process_ledger["chargedOpportunities"]
        == phase_activations,
        "oneFinalProposalSurface": True,
        "runtimeStreamFree": True,
        "failureStateBounded": controller.failure_bank is None
        or controller.failure_bank.validate_bounds(),
    }
    recovery_duration = (
        None
        if controller.recovery_event is None
        else controller.recovery_event - start_event
    )
    summary = {
        "phase": contract.phase,
        "stopReason": state.terminal,
        "completed": current_distance == 0,
        "phaseActivationCount": phase_activations,
        "initialDistance": initial_distance,
        "finalDistance": current_distance,
        "maximumDistance": maximum_distance,
        "distanceAuc": distance_auc,
        "anyTargetDeparture": any_target_departure,
        "finalTargetRetained": current_distance == 0,
        "recoveryObserved": controller.recovery_event is not None,
        "recoveryDuration": recovery_duration,
        "recoveryCensored": contract.phase == "injury"
        and controller.recovery_event is None,
        "transitionObserved": controller.transition_event is not None,
        "transitionCensored": controller.transition_event is None,
        "reversionObserved": controller.reversion_event is not None,
        "reversionCensored": contract.reversible
        and controller.transition_event is not None
        and controller.reversion_event is None,
        "modeActorId": controller.mode_actor,
        "modeOpportunities": controller.mode_opportunities,
        "ledgerDelta": ledger_delta,
        "streamCounterDelta": stream_delta,
        "traceMode": "selected_compact_full" if retain_trace else "digest",
        "retainedEventCount": len(retained),
        "storageBitsPerIdentity": process_final["storageBitsPerIdentity"],
        "totalStorageBits": process_final["totalStorageBits"],
    }
    return PlasticityRun(
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
        event_digest=digest.hex() if retain_trace else EMPTY_DIGEST,
        events=tuple(retained),
        process_initial_state=process_initial,
        process_final_state=process_final,
        process_ledger=process_ledger,
        process_audit_digest=controller.audit_digest.hex(),
        process_audit_count=controller.audit_count,
        process_transitions=tuple(controller.transitions),
        retained_process_audits=tuple(controller.retained_audits),
        opportunity_validation=opportunity_validation,
    )


def exact_replay_plasticity(
    run: PlasticityRun,
    scenario: Scenario,
    checkpoint: Checkpoint,
    occupancy: Sequence[str],
    phase_budget: int,
) -> None:
    replay = run_plasticity_phase(
        scenario,
        checkpoint,
        occupancy=occupancy,
        anchor_lesion_state_hash=run.anchor_lesion_state_hash,
        selected_identity=run.selected_identity,
        contract=run.contract,
        phase_budget=phase_budget,
        retain_trace=bool(run.events),
    )
    if run.to_json_bytes() != replay.to_json_bytes():
        raise AssertionError("S08 plasticity exact replay mismatch")


def plasticity_case_id(
    s01_pairing_block_id: str,
    timing_condition_id: str,
    phase: str,
    anchor_hash: str | None,
) -> str:
    return "e05pc8:" + sha256_json(
        {
            "s01PairingBlockId": s01_pairing_block_id,
            "timingConditionId": timing_condition_id,
            "phase": phase,
            "anchorLesionStateHash": anchor_hash,
        }
    )
