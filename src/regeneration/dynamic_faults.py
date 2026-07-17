"""E05 S04 dynamic fault processes over exact regeneration checkpoints.

Dynamic process state is deliberately outside immutable ``Scenario.cells``.
That preserves the original executable scenario ID and its base RNG root while
allowing recovery.  Static S03 stuck reconstruction remains a semantic
reference, not a byte-equivalent execution path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Any, Mapping, Sequence

import jsonschema

from causal_simulator.action_interface import (
    ActorView,
    CommonActionInterface,
    ObservationRecord,
)
from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
)
from reference_simulator.engine import EMPTY_DIGEST, evaluate_terminal, execute_batch
from reference_simulator.model import (
    Cell,
    FaultMode,
    Proposal,
    ProposalKind,
    RunState,
    Scenario,
    canonical_json_bytes,
    sha256_json,
    state_hash,
)
from reference_simulator.rng import u64
from reference_simulator.scheduler import ScheduledOpportunity, scheduled_actor
from reference_simulator.transition_primitives import ValidationDecision, ledger_identity

from .tasks import Checkpoint, occupancy_values, strict_unequal_inversions


BENCHMARK_VERSION = "E05-dynamic-fault-processes-v1"
DYNAMIC_SPEC_SCHEMA_VERSION = "e05.s04.dynamic-fault-spec.v1"
DYNAMIC_RUN_SCHEMA_VERSION = "e05.s04.dynamic-fault-run.v1"

TEMPORARY_RECOVERY_STREAM = "temporary_freeze_recovery_hazard_s04_v1"
INTERMITTENT_TRANSITION_STREAM = "intermittent_movement_transition_s04_v1"
SENSING_VALUE_STREAM = "probabilistic_sensing_value_s04_v1"
SENSING_STATUS_STREAM = "probabilistic_sensing_status_s04_v1"
PROCESS_STREAMS = (
    TEMPORARY_RECOVERY_STREAM,
    INTERMITTENT_TRANSITION_STREAM,
    SENSING_VALUE_STREAM,
    SENSING_STATUS_STREAM,
)


class DynamicProfile(str, Enum):
    SHAM = "matched_dynamic_process_sham_v1"
    TEMPORARY_FIXED = "temporary_freeze_fixed_16_v1"
    TEMPORARY_GEOMETRIC = "temporary_freeze_geometric_1_16_v1"
    INTERMITTENT = "intermittent_movement_markov_1_16_1_4_v1"
    SENSING = "probabilistic_sensing_value_status_1_8_v1"
    FATIGUE = "movement_fatigue_threshold_3_cooldown_8_v1"


ACTIVE_PROFILES = tuple(item for item in DynamicProfile if item != DynamicProfile.SHAM)


DYNAMIC_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s04/dynamic-fault-spec.schema.json",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenQuestion",
        "inherits",
        "runtimeOverlay",
        "profiles",
        "shamProfile",
        "streamRegistry",
        "pairingContract",
        "calibration",
        "validationPanel",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": DYNAMIC_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S04"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "frozenQuestion": {"type": "string", "minLength": 20},
        "inherits": {"type": "object"},
        "runtimeOverlay": {"type": "object"},
        "profiles": {"type": "array", "minItems": 5, "maxItems": 5},
        "shamProfile": {"type": "object"},
        "streamRegistry": {"type": "array", "minItems": 4, "maxItems": 4},
        "pairingContract": {"type": "object"},
        "calibration": {"type": "object"},
        "validationPanel": {"type": "object"},
        "claimBoundary": {"type": "string", "minLength": 30},
    },
}


@dataclass(frozen=True, slots=True)
class DynamicFaultContract:
    profile: DynamicProfile
    fixed_duration: int = 16
    recovery_hazard_numerator: int = 1
    recovery_hazard_denominator: int = 16
    intermittent_fail_numerator: int = 1
    intermittent_fail_denominator: int = 16
    intermittent_recover_numerator: int = 1
    intermittent_recover_denominator: int = 4
    sensing_error_numerator: int = 1
    sensing_error_denominator: int = 8
    fatigue_threshold: int = 3
    fatigue_cooldown: int = 8
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
            raise ValueError("dynamic process selected an unknown identity")
        if any(cell.fault != FaultMode.NORMAL for cell in scenario.cells):
            raise ValueError("S04 runtime overlays require the original normal S01 scenario")
        if self.information_permission != "policy_native_local":
            raise ValueError("S04 cannot change policy information permission")
        if self.legal_primitives != ("NoOp", "Swap", "MemoryUpdate"):
            raise ValueError("S04 legal primitive set is frozen")
        if self.scheduler != "uniform_random_activation":
            raise ValueError("S04 regeneration panel preserves uniform scheduling")
        if self.continuation != "skip_and_continue" or self.retry != "no_retry":
            raise ValueError("S04 preserves continuation and no-retry contracts")
        for numerator, denominator in (
            (self.recovery_hazard_numerator, self.recovery_hazard_denominator),
            (self.intermittent_fail_numerator, self.intermittent_fail_denominator),
            (
                self.intermittent_recover_numerator,
                self.intermittent_recover_denominator,
            ),
            (self.sensing_error_numerator, self.sensing_error_denominator),
        ):
            _validate_dyadic(numerator, denominator)
        if self.fixed_duration < 1:
            raise ValueError("fixed freeze duration must be positive")
        if self.fatigue_threshold < 1 or self.fatigue_cooldown < 1:
            raise ValueError("fatigue threshold and cooldown must be positive")

    @property
    def classification(self) -> str:
        if self.profile == DynamicProfile.FATIGUE:
            return "endogenous_mediator"
        if self.profile == DynamicProfile.SHAM:
            return "control"
        return "exogenous_fault"

    @property
    def family(self) -> str:
        return {
            DynamicProfile.SHAM: "none",
            DynamicProfile.TEMPORARY_FIXED: "temporary_freezing",
            DynamicProfile.TEMPORARY_GEOMETRIC: "temporary_freezing",
            DynamicProfile.INTERMITTENT: "intermittent_movement_failure",
            DynamicProfile.SENSING: "probabilistic_sensing_failure",
            DynamicProfile.FATIGUE: "movement_dependent_fatigue",
        }[self.profile]

    @property
    def streams(self) -> tuple[str, ...]:
        return {
            DynamicProfile.SHAM: (),
            DynamicProfile.TEMPORARY_FIXED: (),
            DynamicProfile.TEMPORARY_GEOMETRIC: (TEMPORARY_RECOVERY_STREAM,),
            DynamicProfile.INTERMITTENT: (INTERMITTENT_TRANSITION_STREAM,),
            DynamicProfile.SENSING: (SENSING_VALUE_STREAM, SENSING_STATUS_STREAM),
            DynamicProfile.FATIGUE: (),
        }[self.profile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profileId": self.profile.value,
            "family": self.family,
            "classification": self.classification,
            "fixedDurationOpportunities": self.fixed_duration,
            "recoveryHazard": {
                "numerator": self.recovery_hazard_numerator,
                "denominator": self.recovery_hazard_denominator,
            },
            "intermittentAvailableToFailed": {
                "numerator": self.intermittent_fail_numerator,
                "denominator": self.intermittent_fail_denominator,
            },
            "intermittentFailedToAvailable": {
                "numerator": self.intermittent_recover_numerator,
                "denominator": self.intermittent_recover_denominator,
            },
            "sensingError": {
                "numerator": self.sensing_error_numerator,
                "denominator": self.sensing_error_denominator,
            },
            "fatigueThreshold": self.fatigue_threshold,
            "fatigueCooldownOpportunities": self.fatigue_cooldown,
            "streams": list(self.streams),
            "informationPermission": self.information_permission,
            "legalPrimitives": list(self.legal_primitives),
            "scheduler": self.scheduler,
            "continuation": self.continuation,
            "retry": self.retry,
        }


def validate_dynamic_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(DYNAMIC_SPEC_SCHEMA).validate(specification)
    profile_ids = {item["profileId"] for item in specification["profiles"]}
    if profile_ids != {item.value for item in ACTIVE_PROFILES}:
        raise ValueError("S04 specification must define each active profile exactly once")
    if specification["shamProfile"].get("profileId") != DynamicProfile.SHAM.value:
        raise ValueError("S04 matched sham profile changed")
    stream_names = [item["name"] for item in specification["streamRegistry"]]
    if len(set(stream_names)) != len(stream_names) or set(stream_names) != set(
        PROCESS_STREAMS
    ):
        raise ValueError("S04 process stream registry must be unique and complete")
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "lesionSpecSchemaVersion": "e05.s03.lesion-spec.v1",
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "runtimeAnchorLesion": "segment_reversal_central_v1",
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"inherited S01-S03 contract changed: {key}")
    panel = specification["validationPanel"]
    if panel.get("plannedCheckpointCount") != 384:
        raise ValueError("S04 must retain all 384 S02 checkpoints")
    if panel.get("activeProfilesPerCheckpoint") != 5:
        raise ValueError("S04 must retain all five active profiles")
    if panel.get("plannedRunCount") != 3840:
        raise ValueError("S04 must retain every sham/active run")


def _validate_dyadic(numerator: int, denominator: int) -> None:
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError("probability denominator must be a positive power of two")
    if not 0 <= numerator <= denominator:
        raise ValueError("probability numerator must be inside [0, denominator]")


def dyadic_hit(raw: int, numerator: int, denominator: int) -> bool:
    _validate_dyadic(numerator, denominator)
    return raw < ((1 << 64) * numerator // denominator)


@dataclass(frozen=True, slots=True)
class DynamicSensingAudit:
    event_index: int
    read_ordinal: int
    field: str
    stream: str
    raw_uint64: int
    applied: bool
    actual: int | float | str
    visible: int | float | str

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventIndex": self.event_index,
            "readOrdinal": self.read_ordinal,
            "field": self.field,
            "stream": self.stream,
            "rawUint64": self.raw_uint64,
            "applied": self.applied,
            "actual": self.actual,
            "visible": self.visible,
        }


class DynamicSensingTransformer:
    """S04 stream-isolated corruption of authorized observation records."""

    __slots__ = ("contract", "seed", "scenario_id", "controller")

    def __init__(
        self,
        contract: DynamicFaultContract,
        scenario: Scenario,
        controller: "DynamicProcessController",
    ) -> None:
        self.contract = contract
        self.seed = scenario.seed
        self.scenario_id = scenario.scenario_id
        self.controller = controller

    def __call__(
        self,
        record: ObservationRecord,
        *,
        event_index: int,
        read_ordinal: int,
    ) -> ObservationRecord:
        raw_value = u64(
            self.seed,
            self.scenario_id,
            SENSING_VALUE_STREAM,
            event_index,
            read_ordinal,
        )
        value_hit = dyadic_hit(
            raw_value,
            self.contract.sensing_error_numerator,
            self.contract.sensing_error_denominator,
        )
        visible_value = record.value
        if value_hit:
            visible_value = record.value + (1 if ((raw_value >> 60) & 1) else -1)
        self.controller.record_sensing(
            DynamicSensingAudit(
                event_index,
                read_ordinal,
                "value",
                SENSING_VALUE_STREAM,
                raw_value,
                value_hit,
                record.value,
                visible_value,
            )
        )
        if isinstance(record, ActorView):
            # Actor self fault and every protected field remain ground truth.
            from dataclasses import replace

            return replace(record, value=visible_value)

        raw_status = u64(
            self.seed,
            self.scenario_id,
            SENSING_STATUS_STREAM,
            event_index,
            read_ordinal,
        )
        status_hit = dyadic_hit(
            raw_status,
            self.contract.sensing_error_numerator,
            self.contract.sensing_error_denominator,
        )
        visible_fault = record.fault
        if status_hit:
            alternatives = tuple(item for item in FaultMode if item != record.fault)
            visible_fault = alternatives[(raw_status >> 60) & 1]
        self.controller.record_sensing(
            DynamicSensingAudit(
                event_index,
                read_ordinal,
                "target_status",
                SENSING_STATUS_STREAM,
                raw_status,
                status_hit,
                record.fault.value,
                visible_fault.value,
            )
        )
        from dataclasses import replace

        return replace(record, value=visible_value, fault=visible_fault)


class DynamicProcessController:
    """Process state, post-validation gates, audit digest, and supplemental ledger."""

    __slots__ = (
        "contract",
        "seed",
        "scenario_id",
        "selected_identity",
        "start_event_index",
        "temporary_frozen",
        "temporary_recovery_event",
        "temporary_recovery_pending",
        "intermittent_failed",
        "fatigue_load",
        "fatigue_until",
        "ledger",
        "audit_digest",
        "audit_count",
        "retained_audits",
        "retain_audits",
        "transitions",
        "pending_sensing_draws",
        "proposal_kinds",
    )

    def __init__(
        self,
        contract: DynamicFaultContract,
        scenario: Scenario,
        selected_identity: str,
        start_event_index: int,
        *,
        retain_audits: bool = False,
    ) -> None:
        contract.validate(scenario, selected_identity)
        self.contract = contract
        self.seed = scenario.seed
        self.scenario_id = scenario.scenario_id
        self.selected_identity = selected_identity
        self.start_event_index = start_event_index
        self.temporary_frozen = contract.profile in {
            DynamicProfile.TEMPORARY_FIXED,
            DynamicProfile.TEMPORARY_GEOMETRIC,
        }
        self.temporary_recovery_event: int | None = None
        self.temporary_recovery_pending = False
        self.intermittent_failed = False
        self.fatigue_load = {cell.cell_id: 0 for cell in scenario.cells}
        self.fatigue_until: dict[str, int] = {}
        self.ledger: Counter[str] = Counter()
        self.audit_digest = hashlib.sha256(b"E05/S04/process-audit/v1").digest()
        self.audit_count = 0
        self.retained_audits: list[dict[str, Any]] = []
        self.retain_audits = retain_audits
        self.transitions: list[dict[str, Any]] = []
        self.pending_sensing_draws: list[DynamicSensingAudit] = []
        self.proposal_kinds: set[str] = set()

    def _audit(self, kind: str, content: Mapping[str, Any]) -> None:
        record = {"kind": kind, **content}
        encoded = canonical_json_bytes(record)
        self.audit_digest = hashlib.sha256(self.audit_digest + encoded).digest()
        self.audit_count += 1
        if self.retain_audits:
            self.retained_audits.append(record)

    def _transition(self, kind: str, event_index: int, **content: Any) -> None:
        record = {"transition": kind, "eventIndex": event_index, **content}
        self.transitions.append(record)
        self._audit("transition", record)

    def record_sensing(self, audit: DynamicSensingAudit) -> None:
        item = audit.to_dict()
        self.pending_sensing_draws.append(audit)
        if audit.field == "value":
            self.ledger["sensingValueDraws"] += 1
            self.ledger["sensingValueErrors"] += int(audit.applied)
        else:
            self.ledger["sensingStatusDraws"] += 1
            self.ledger["sensingStatusErrors"] += int(audit.applied)
        self.ledger["sensingErrorsApplied"] += int(audit.applied)
        self.ledger["sensingErrorHandlingOperations"] += int(audit.applied)
        self._audit("sensing", item)

    def _refresh_fixed_and_fatigue(self, event_index: int) -> None:
        if (
            self.contract.profile == DynamicProfile.TEMPORARY_FIXED
            and self.temporary_frozen
            and event_index >= self.start_event_index + self.contract.fixed_duration
        ):
            self.temporary_frozen = False
            self.temporary_recovery_event = event_index
            self.ledger["temporaryFreezeRecoveries"] += 1
            self._transition(
                "temporary_freeze_recovered_fixed",
                event_index,
                durationOpportunities=event_index - self.start_event_index,
            )
        recovered = [
            cell_id
            for cell_id, until in self.fatigue_until.items()
            if event_index >= until
        ]
        for cell_id in sorted(recovered):
            del self.fatigue_until[cell_id]
            self.ledger["fatigueRecoveries"] += 1
            self._transition("fatigue_recovered", event_index, identityId=cell_id)

    def prepare(
        self,
        proposal: Proposal,
        event_index: int,
    ) -> tuple[Proposal, tuple[tuple[str, int], ...]]:
        from dataclasses import replace

        self.ledger["chargedOpportunities"] += 1
        self.proposal_kinds.add(proposal.kind.value)
        self._refresh_fixed_and_fatigue(event_index)
        draws: list[tuple[str, int, int, int]] = []
        consumption: list[tuple[str, int]] = []
        self.temporary_recovery_pending = False
        if (
            self.contract.profile == DynamicProfile.TEMPORARY_GEOMETRIC
            and self.temporary_frozen
        ):
            raw = u64(
                self.seed,
                self.scenario_id,
                TEMPORARY_RECOVERY_STREAM,
                event_index,
                0,
            )
            hit = dyadic_hit(
                raw,
                self.contract.recovery_hazard_numerator,
                self.contract.recovery_hazard_denominator,
            )
            self.temporary_recovery_pending = hit
            self.ledger["temporaryRecoveryHazardDraws"] += 1
            draws.append((TEMPORARY_RECOVERY_STREAM, event_index, 0, raw))
            consumption.append((TEMPORARY_RECOVERY_STREAM, 1))
            self._audit(
                "temporary_recovery_hazard",
                {"eventIndex": event_index, "rawUint64": raw, "hit": hit},
            )
        if self.contract.profile == DynamicProfile.INTERMITTENT:
            raw = u64(
                self.seed,
                self.scenario_id,
                INTERMITTENT_TRANSITION_STREAM,
                event_index,
                0,
            )
            before = self.intermittent_failed
            if before:
                if dyadic_hit(
                    raw,
                    self.contract.intermittent_recover_numerator,
                    self.contract.intermittent_recover_denominator,
                ):
                    self.intermittent_failed = False
                    self.ledger["intermittentRecoverTransitions"] += 1
            elif dyadic_hit(
                raw,
                self.contract.intermittent_fail_numerator,
                self.contract.intermittent_fail_denominator,
            ):
                self.intermittent_failed = True
                self.ledger["intermittentFailureTransitions"] += 1
            self.ledger["intermittentTransitionDraws"] += 1
            self.ledger["intermittentFailedExposures"] += int(
                self.intermittent_failed
            )
            draws.append((INTERMITTENT_TRANSITION_STREAM, event_index, 0, raw))
            consumption.append((INTERMITTENT_TRANSITION_STREAM, 1))
            self._audit(
                "intermittent_transition",
                {
                    "eventIndex": event_index,
                    "rawUint64": raw,
                    "beforeFailed": before,
                    "afterFailed": self.intermittent_failed,
                },
            )
        if self.contract.profile == DynamicProfile.SENSING:
            if any(item.event_index != event_index for item in self.pending_sensing_draws):
                raise AssertionError("pending sensing draws crossed an event boundary")
            for item in self.pending_sensing_draws:
                draws.append(
                    (item.stream, item.event_index, item.read_ordinal, item.raw_uint64)
                )
            counts = Counter(item.stream for item in self.pending_sensing_draws)
            consumption.extend(sorted(counts.items()))
            self.pending_sensing_draws.clear()
        elif self.pending_sensing_draws:
            raise AssertionError("sensing draws appeared outside the sensing profile")
        if self.temporary_frozen:
            self.ledger["temporaryFreezeExposures"] += 1
        if self.fatigue_until:
            self.ledger["fatigueExposureOpportunities"] += 1
        if draws:
            proposal = replace(
                proposal, random_draws=proposal.random_draws + tuple(draws)
            )
        return proposal, tuple(consumption)

    def outcome(
        self,
        proposal: Proposal,
        validation: ValidationDecision,
        event_index: int,
    ) -> ValidationDecision:
        if not validation.eligible_for_commit:
            return validation
        target_id = (
            proposal.observed_target_id
            if proposal.kind == ProposalKind.SWAP
            else None
        )
        if self.temporary_frozen:
            if proposal.actor_id == self.selected_identity:
                self.ledger["temporaryFreezeActorBlocks"] += 1
                self._audit(
                    "block",
                    {
                        "eventIndex": event_index,
                        "reason": "temporary_actor_frozen",
                        "identityId": proposal.actor_id,
                    },
                )
                return ValidationDecision("rejected_temporary_actor_frozen", False)
            if target_id == self.selected_identity:
                self.ledger["temporaryFreezeTargetBlocks"] += 1
                self._audit(
                    "block",
                    {
                        "eventIndex": event_index,
                        "reason": "temporary_target_frozen",
                        "identityId": target_id,
                    },
                )
                return ValidationDecision("rejected_temporary_target_frozen", False)
        if (
            self.contract.profile == DynamicProfile.INTERMITTENT
            and self.intermittent_failed
            and proposal.kind == ProposalKind.SWAP
        ):
            self.ledger["intermittentMovementBlocks"] += 1
            self._audit(
                "block",
                {
                    "eventIndex": event_index,
                    "reason": "intermittent_movement_failure",
                    "identityId": proposal.actor_id,
                },
            )
            return ValidationDecision("rejected_intermittent_movement_failure", False)
        if self.contract.profile == DynamicProfile.FATIGUE:
            if proposal.actor_id in self.fatigue_until:
                self.ledger["fatigueActorBlocks"] += 1
                self._audit(
                    "block",
                    {
                        "eventIndex": event_index,
                        "reason": "fatigued_actor",
                        "identityId": proposal.actor_id,
                    },
                )
                return ValidationDecision("rejected_fatigued_actor", False)
            if target_id in self.fatigue_until:
                self.ledger["fatigueTargetBlocks"] += 1
                self._audit(
                    "block",
                    {
                        "eventIndex": event_index,
                        "reason": "fatigued_target",
                        "identityId": target_id,
                    },
                )
                return ValidationDecision("rejected_fatigued_target", False)
        return validation

    def _increment_fatigue(self, cell_id: str, event_index: int) -> None:
        self.fatigue_load[cell_id] += 1
        self.ledger["fatigueMovementParticipations"] += 1
        if self.fatigue_load[cell_id] >= self.contract.fatigue_threshold:
            self.fatigue_load[cell_id] = 0
            until = event_index + 1 + self.contract.fatigue_cooldown
            self.fatigue_until[cell_id] = until
            self.ledger["fatigueThresholdTriggers"] += 1
            self._transition(
                "fatigue_threshold_reached",
                event_index,
                identityId=cell_id,
                fatigueUntilExclusive=until,
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
            raise ValueError("S04 regeneration dynamic panel is serial")
        proposal = proposals[0]
        event_index = batch_start_index
        if (
            self.contract.profile == DynamicProfile.FATIGUE
            and decisions[proposal.ordinal] == "accepted"
            and proposal.kind == ProposalKind.SWAP
        ):
            if proposal.observed_target_id is None:
                raise AssertionError("accepted Swap lacks trusted target identity")
            self._increment_fatigue(proposal.actor_id, event_index)
            self._increment_fatigue(proposal.observed_target_id, event_index)
        if (
            self.contract.profile == DynamicProfile.TEMPORARY_GEOMETRIC
            and self.temporary_frozen
            and self.temporary_recovery_pending
        ):
            self.temporary_frozen = False
            self.temporary_recovery_event = event_index + 1
            self.ledger["temporaryFreezeRecoveries"] += 1
            self._transition(
                "temporary_freeze_recovered_geometric",
                event_index + 1,
                durationOpportunities=event_index - self.start_event_index + 1,
            )
        self.temporary_recovery_pending = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "profileId": self.contract.profile.value,
            "selectedIdentityId": self.selected_identity,
            "startEventIndex": self.start_event_index,
            "temporaryFrozen": self.temporary_frozen,
            "temporaryRecoveryEventIndex": self.temporary_recovery_event,
            "intermittentFailed": self.intermittent_failed,
            "fatigueLoad": dict(sorted(self.fatigue_load.items())),
            "fatigueUntilExclusive": dict(sorted(self.fatigue_until.items())),
        }

    def process_ledger(self) -> dict[str, int]:
        fields = (
            "chargedOpportunities",
            "temporaryFreezeExposures",
            "temporaryRecoveryHazardDraws",
            "temporaryFreezeRecoveries",
            "temporaryFreezeActorBlocks",
            "temporaryFreezeTargetBlocks",
            "intermittentTransitionDraws",
            "intermittentFailureTransitions",
            "intermittentRecoverTransitions",
            "intermittentFailedExposures",
            "intermittentMovementBlocks",
            "sensingValueDraws",
            "sensingStatusDraws",
            "sensingValueErrors",
            "sensingStatusErrors",
            "sensingErrorsApplied",
            "sensingErrorHandlingOperations",
            "fatigueMovementParticipations",
            "fatigueThresholdTriggers",
            "fatigueExposureOpportunities",
            "fatigueRecoveries",
            "fatigueActorBlocks",
            "fatigueTargetBlocks",
        )
        return {field: int(self.ledger[field]) for field in fields}


@dataclass(frozen=True, slots=True)
class DynamicRun:
    contract: DynamicFaultContract
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
            "schemaVersion": DYNAMIC_RUN_SCHEMA_VERSION,
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


def run_dynamic_phase(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    post_anchor_occupancy: Sequence[str],
    anchor_lesion_state_hash: str,
    selected_identity: str,
    contract: DynamicFaultContract,
    recovery_budget: int,
    trace_mode: str = "digest",
    retain_process_audits: bool = False,
) -> DynamicRun:
    """Resume the exact global state and execute one dynamic process arm."""

    if trace_mode not in {"full", "digest"}:
        raise ValueError("S04 dynamic phase supports full or digest trace modes")
    if recovery_budget < 1:
        raise ValueError("recovery budget must be positive")
    contract.validate(scenario, selected_identity)
    state = checkpoint.to_run_state(occupancy=post_anchor_occupancy)
    start_event = state.activation_count
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    initial_hash = state_hash(scenario.scenario_id, state)
    controller = DynamicProcessController(
        contract,
        scenario,
        selected_identity,
        start_event,
        retain_audits=retain_process_audits,
    )
    process_initial = controller.state_dict()
    transformer = (
        DynamicSensingTransformer(contract, scenario, controller)
        if contract.profile == DynamicProfile.SENSING
        else None
    )
    interface = CommonActionInterface(transformer)
    router = ArchitectureProposalRouter(
        ArchitectureExecutionContract.distributed_local(),
        action_interface=interface,
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
    final_ledger = dict(state.ledger)
    final_streams = dict(state.stream_counters)
    ledger_delta = _delta(final_ledger, initial_ledger)
    stream_delta = _delta(final_streams, initial_streams)
    process_ledger = controller.process_ledger()
    native_checks = ledger_identity(ledger_delta)
    process_draws = {
        TEMPORARY_RECOVERY_STREAM: process_ledger["temporaryRecoveryHazardDraws"],
        INTERMITTENT_TRANSITION_STREAM: process_ledger[
            "intermittentTransitionDraws"
        ],
        SENSING_VALUE_STREAM: process_ledger["sensingValueDraws"],
        SENSING_STATUS_STREAM: process_ledger["sensingStatusDraws"],
    }
    validation = {
        **native_checks,
        "activationDeltaMatchesPhase": ledger_delta["activations"]
        == phase_activations,
        "processOpportunityCountMatchesPhase": process_ledger[
            "chargedOpportunities"
        ]
        == phase_activations,
        "processStreamCounterIdentity": all(
            stream_delta.get(stream, 0) == count
            for stream, count in process_draws.items()
        ),
        "noUndeclaredProcessStream": all(
            stream in contract.streams or count == 0
            for stream, count in process_draws.items()
        ),
        "sensingErrorIdentity": process_ledger["sensingErrorsApplied"]
        == process_ledger["sensingValueErrors"]
        + process_ledger["sensingStatusErrors"]
        == process_ledger["sensingErrorHandlingOperations"],
        "fatigueHasNoExogenousDraws": contract.profile != DynamicProfile.FATIGUE
        or sum(process_draws.values()) == 0,
        "fixedDurationHasNoExogenousDraws": contract.profile
        != DynamicProfile.TEMPORARY_FIXED
        or sum(process_draws.values()) == 0,
        "legalPrimitivesPreserved": all(
            kind in contract.legal_primitives for kind in controller.proposal_kinds
        ),
        "phaseBudgetRespected": phase_activations <= recovery_budget,
    }
    values = occupancy_values(scenario, state.occupancy)
    temporary_duration = (
        None
        if controller.temporary_recovery_event is None
        else controller.temporary_recovery_event - start_event
    )
    summary = {
        "profileId": contract.profile.value,
        "family": contract.family,
        "classification": contract.classification,
        "stopReason": state.terminal,
        "completed": state.terminal == "complete",
        "phaseActivationCount": phase_activations,
        "globalStartEventIndex": start_event,
        "globalEndEventIndex": state.activation_count,
        "recoveryBudget": recovery_budget,
        "finalDistance": strict_unequal_inversions(values, scenario.cells[0].direction),
        "temporaryRecoveryObserved": controller.temporary_recovery_event is not None,
        "temporaryRecoveryDuration": temporary_duration,
        "temporaryRecoveryCensored": contract.profile
        in {DynamicProfile.TEMPORARY_FIXED, DynamicProfile.TEMPORARY_GEOMETRIC}
        and controller.temporary_recovery_event is None,
        "ledgerDelta": ledger_delta,
        "streamCounterDelta": stream_delta,
        "traceMode": trace_mode,
        "retainedEventCount": len(retained),
    }
    return DynamicRun(
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


def exact_replay_dynamic(
    run_result: DynamicRun,
    scenario: Scenario,
    checkpoint: Checkpoint,
    post_anchor_occupancy: Sequence[str],
    recovery_budget: int,
) -> DynamicRun:
    replayed = run_dynamic_phase(
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
        raise AssertionError("dynamic fault run replay mismatch")
    return replayed


def rebuild_static_stuck_scenario(
    scenario: Scenario, selected_identity: str
) -> Scenario:
    """Construct only the S03 comparison root; never use it for S04 execution."""

    if selected_identity not in scenario.cell_map:
        raise KeyError(selected_identity)
    cells = []
    for cell in scenario.cells:
        if cell.cell_id == selected_identity:
            cell = Cell(
                cell_id=cell.cell_id,
                value=cell.value,
                policy=cell.policy,
                direction=cell.direction,
                fault=FaultMode.STUCK,
                analysis_label=cell.analysis_label,
            )
        cells.append(cell)
    return Scenario.create(
        cells,
        initial_occupancy=scenario.initial_occupancy,
        initial_selection_cursors=dict(scenario.initial_selection_cursors),
        seed=scenario.seed,
        max_activations=scenario.max_activations,
        architecture=scenario.architecture,
        scheduler=scenario.scheduler,
        batch_width=scenario.batch_width,
        traditional_policy=scenario.traditional_policy,
        generation_key=scenario.generation_key,
        fault_placement="explicit",
        requested_fault_count=1,
        rng_profile=scenario.rng_profile,
        goal_profile=scenario.goal_profile,
        metric_profile=scenario.metric_profile,
    )


def rng_coupling_audit(
    scenario: Scenario,
    selected_identity: str,
    event_index: int,
) -> dict[str, Any]:
    rebuilt = rebuild_static_stuck_scenario(scenario, selected_identity)
    streams = ("actor_activation", "bubble_side", *PROCESS_STREAMS)
    rows = []
    for stream in streams:
        original = u64(scenario.seed, scenario.scenario_id, stream, event_index, 0)
        reconstructed = u64(
            scenario.seed, rebuilt.scenario_id, stream, event_index, 0
        )
        rows.append(
            {
                "stream": stream,
                "eventIndex": event_index,
                "originalRootValue": str(original),
                "reconstructedRootValue": str(reconstructed),
                "valuesEqual": original == reconstructed,
            }
        )
    return {
        "sourceScenarioId": scenario.scenario_id,
        "reconstructedStaticStuckScenarioId": rebuilt.scenario_id,
        "scenarioIdChanged": rebuilt.scenario_id != scenario.scenario_id,
        "selectedIdentityId": selected_identity,
        "classification": "scenario_paired_rng_unpaired",
        "resolution": "execute dynamic process overlays on the original scenario root",
        "streams": rows,
        "basePairingStatus": "shared_prefix_until_arm_terminal_or_path_divergence",
    }


def dynamic_pair_id(
    s01_pairing_block_id: str,
    timing_condition_id: str,
    anchor_lesion_state_hash: str,
    profile: DynamicProfile,
) -> str:
    return "e05dp4:" + sha256_json(
        {
            "s01PairingBlockId": s01_pairing_block_id,
            "timingConditionId": timing_condition_id,
            "anchorLesionStateHash": anchor_lesion_state_hash,
            "profileId": profile.value,
        }
    )


def probability_z(observed_hits: int, trials: int, probability: float) -> float:
    if trials < 1 or not 0.0 < probability < 1.0:
        raise ValueError("z calibration requires trials and an interior probability")
    expected = trials * probability
    standard = math.sqrt(trials * probability * (1.0 - probability))
    return (observed_hits - expected) / standard
