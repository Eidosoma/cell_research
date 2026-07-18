"""E05 S09 collective-target changes with explicit signal permissions.

S09 keeps the immutable E01 scenario and exact S02 checkpoint untouched.  The
collective objective is an external target-code overlay.  A target-aware arm
may transform only records already authorized by the native Algotype; its
matched nonadaptive control receives and pays for the same signal and shadow
candidate but emits the ordinary native proposal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
from typing import Any, Mapping, Sequence

import jsonschema

from causal_simulator.action_interface import (
    ActorView,
    CommonActionInterface,
    ObservationRecord,
    VisibleCell,
    build_policy_native_observation,
    propose_from_observation,
)
from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
)
from reference_simulator.engine import EMPTY_DIGEST
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
    commit_proposal,
    cost_delta,
    ledger_identity,
    validate_proposal,
)

from .tasks import Checkpoint


BENCHMARK_VERSION = "E05-collective-target-change-v1"
TARGET_CHANGE_SPEC_SCHEMA_VERSION = "e05.s09.target-change-spec.v1"
TARGET_CHANGE_RUN_SCHEMA_VERSION = "e05.s09.target-change-run.v1"


class TargetChange(str, Enum):
    REVERSE = "reverse_total_order_v1"
    PARTIAL = "binary_precedence_partial_order_v1"
    CLASSES = "cyclic_tertile_value_classes_v1"


class SignalPermission(str, Enum):
    NONE = "none_negative_control_v1"
    LOCAL = "local_boundary_relay_v1"
    GRADIENT = "gradient_target_code_v1"
    GLOBAL = "global_target_broadcast_v1"


class TargetArm(str, Enum):
    CHANGED_AWARE = "changed_target_aware"
    CHANGED_NONADAPTIVE = "changed_nonadaptive_matched"
    STABILITY_AWARE = "no_change_target_aware_stability"
    STABILITY_NONADAPTIVE = "no_change_nonadaptive_matched_stability"


TARGET_CHANGE_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s09/target-change-spec.schema.json",
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
        "targetChanges",
        "targetCorrespondenceContract",
        "changeTiming",
        "signalPermissions",
        "forbiddenControllerInputs",
        "controllerConditions",
        "controllerAndCostContract",
        "controlsAndPairing",
        "metrics",
        "successCriteria",
        "validationPanel",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": TARGET_CHANGE_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S09"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "targetChanges": {"type": "array", "minItems": 3, "maxItems": 3},
        "signalPermissions": {"type": "array", "minItems": 4, "maxItems": 4},
        "controllerConditions": {"type": "array", "minItems": 4, "maxItems": 4},
    },
}


def validate_target_change_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(TARGET_CHANGE_SPEC_SCHEMA).validate(specification)
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "lesionSpecSchemaVersion": "e05.s03.lesion-spec.v1",
        "dynamicSpecSchemaVersion": "e05.s04.dynamic-fault-spec.v1",
        "nudgeSpecSchemaVersion": "e05.s05.nudge-recovery-spec.v1",
        "rescueSpecSchemaVersion": "e05.s06.assisted-rescue-spec.v1",
        "memorySpecSchemaVersion": "e05.s07.local-memory-spec.v1",
        "plasticitySpecSchemaVersion": "e05.s08.policy-plasticity-spec.v1",
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "runtimeScenarioBoundary": "original_s01_scenario_id_external_target_overlay",
        "targetChangeLesion": "none; target change is an instantaneous objective intervention on the exact S02 checkpoint, before any S03 lesion",
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"inherited S01-S08 contract changed: {key}")
    targets = {item["targetChangeId"] for item in specification["targetChanges"]}
    signals = {item["signalId"] for item in specification["signalPermissions"]}
    arms = {item["arm"] for item in specification["controllerConditions"]}
    if targets != {item.value for item in TargetChange}:
        raise ValueError("S09 target-change set changed")
    if signals != {item.value for item in SignalPermission}:
        raise ValueError("S09 signal-permission set changed")
    if arms != {item.value for item in TargetArm}:
        raise ValueError("S09 controller-arm set changed")
    if specification["controllerAndCostContract"].get("runtimeStreams") != []:
        raise ValueError("S09 must own no runtime stream")
    if specification["priorEvidenceConstraints"].get("controllerBoundary") is None:
        raise ValueError("S07/S08 constraints must remain explicit")
    panel = specification["validationPanel"]
    expected_counts = {
        "sourceBlockCount": 48,
        "changedCheckpointCount": 384,
        "stabilityCheckpointCount": 48,
        "targetChangeCount": 3,
        "signalPermissionCount": 4,
        "plannedChangedRuns": 9216,
        "plannedNoChangeStabilityRuns": 1152,
        "plannedRunCount": 10368,
        "plannedExactReplayCount": 10368,
        "plannedTotalTrajectoryExecutions": 20736,
        "plannedPrimaryAdaptationPairs": 4608,
        "plannedPrimaryStabilityPairs": 576,
        "plannedPairwiseContrasts": 5184,
        "selectedTraceRuns": 24,
    }
    for key, value in expected_counts.items():
        if panel.get(key) != value:
            raise ValueError(f"S09 planned panel changed: {key}")


def _strict_code_inversions(codes: Sequence[int]) -> int:
    return sum(
        left > right
        for index, left in enumerate(codes)
        for right in codes[index + 1 :]
    )


def _applied_swap_inversion_delta(
    occupancy_after: Sequence[str],
    codes: Mapping[str, int],
    first_position: int,
    second_position: int,
) -> int:
    """Return exact post-minus-pre inversion distance for an applied swap.

    ``occupancy_after`` is the state after exchanging the two positions. Only
    the exchanged identities' pair relations can change, so this calculation
    is O(distance between positions) and is algebraically identical to a full
    inversion recount. Equal target codes remain incomparable and contribute
    zero in both states.
    """

    if first_position == second_position:
        return 0
    left = min(first_position, second_position)
    right = max(first_position, second_position)
    post_left = codes[occupancy_after[left]]
    post_right = codes[occupancy_after[right]]
    pre_left = post_right
    pre_right = post_left
    delta = int(post_left > post_right) - int(pre_left > pre_right)
    for position in range(left + 1, right):
        middle = codes[occupancy_after[position]]
        post = int(post_left > middle) + int(middle > post_right)
        pre = int(pre_left > middle) + int(middle > pre_right)
        delta += post - pre
    return delta


def _maximum_code_distance(codes: Mapping[str, int]) -> int:
    counts = Counter(codes.values())
    values = sorted(counts)
    return sum(counts[left] * counts[right] for i, left in enumerate(values) for right in values[i + 1 :])


@dataclass(frozen=True, slots=True)
class TargetDefinition:
    target_change: TargetChange
    no_change: bool
    old_codes: tuple[tuple[str, int], ...]
    target_codes: tuple[tuple[str, int], ...]
    policy_codes: tuple[tuple[str, int], ...]
    maximum_distance: int
    target_hash: str

    @property
    def old_code_map(self) -> dict[str, int]:
        return dict(self.old_codes)

    @property
    def target_code_map(self) -> dict[str, int]:
        return dict(self.target_codes)

    @property
    def policy_code_map(self) -> dict[str, int]:
        return dict(self.policy_codes)

    def distance(self, occupancy: Sequence[str]) -> int:
        mapping = self.target_code_map
        return _strict_code_inversions([mapping[item] for item in occupancy])

    def old_distance(self, occupancy: Sequence[str]) -> int:
        mapping = self.old_code_map
        return _strict_code_inversions([mapping[item] for item in occupancy])

    def to_dict(self) -> dict[str, Any]:
        return {
            "targetChangeId": self.target_change.value,
            "noChange": self.no_change,
            "oldCodes": dict(self.old_codes),
            "targetCodes": dict(self.target_codes),
            "policyCodes": dict(self.policy_codes),
            "maximumDistance": self.maximum_distance,
            "targetHash": self.target_hash,
        }


def build_target_definition(
    scenario: Scenario,
    target_change: TargetChange,
    *,
    no_change: bool = False,
) -> TargetDefinition:
    values = [cell.value for cell in scenario.cells]
    if len(values) != len(set(values)):
        raise ValueError("S09 frozen correspondence requires unique source values")
    direction = scenario.cells[0].direction
    if any(cell.direction != direction for cell in scenario.cells):
        raise ValueError("S09 requires the inherited homogeneous direction")
    ordered = sorted(
        scenario.cells,
        key=lambda cell: cell.value,
        reverse=direction == Direction.DESCENDING,
    )
    old = {cell.cell_id: rank for rank, cell in enumerate(ordered)}
    n = len(ordered)
    if no_change:
        target = dict(old)
    elif target_change == TargetChange.REVERSE:
        target = {identity: n - 1 - rank for identity, rank in old.items()}
    elif target_change == TargetChange.PARTIAL:
        target = {identity: rank % 2 for identity, rank in old.items()}
    elif target_change == TargetChange.CLASSES:
        rotation = {0: 2, 1: 0, 2: 1}
        target = {
            identity: rotation[min(2, (3 * rank) // n)]
            for identity, rank in old.items()
        }
    else:  # pragma: no cover - exhaustive Enum guard
        raise AssertionError(target_change)
    sign = 1 if direction == Direction.ASCENDING else -1
    policy = {identity: sign * code for identity, code in target.items()}
    maximum = _maximum_code_distance(target)
    if maximum < 1:
        raise ValueError("S09 target must have positive unequal-pair support")
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
        target_change,
        no_change,
        tuple(sorted(old.items())),
        tuple(sorted(target.items())),
        tuple(sorted(policy.items())),
        maximum,
        target_hash,
    )


def target_correspondence_rows(
    scenario: Scenario,
    definition: TargetDefinition,
) -> list[dict[str, Any]]:
    target = definition.target_code_map
    old = definition.old_code_map
    counts = Counter(target.values())
    starts: dict[int, int] = {}
    cursor = 0
    for code in sorted(counts):
        starts[code] = cursor
        cursor += counts[code]
    rows: list[dict[str, Any]] = []
    for identity in sorted(target):
        code = target[identity]
        start = starts[code]
        end = start + counts[code] - 1
        rows.append(
            {
                "scenarioId": scenario.scenario_id,
                "targetChangeId": definition.target_change.value,
                "noChange": definition.no_change,
                "targetHash": definition.target_hash,
                "identityId": identity,
                "oldTargetRank": old[identity],
                "newTargetCode": code,
                "policyCode": definition.policy_code_map[identity],
                "allowedPositionStart": start,
                "allowedPositionEnd": end,
                "exactPositionRequired": start == end,
                "identityConserved": True,
                "targetFeasible": True,
            }
        )
    return rows


@dataclass(frozen=True, slots=True)
class TargetChangeContract:
    arm: TargetArm
    target_change: TargetChange
    signal: SignalPermission

    @property
    def phase(self) -> str:
        return "stability" if self.arm in {
            TargetArm.STABILITY_AWARE,
            TargetArm.STABILITY_NONADAPTIVE,
        } else "changed"

    @property
    def target_aware(self) -> bool:
        return self.arm in {TargetArm.CHANGED_AWARE, TargetArm.STABILITY_AWARE}

    @property
    def no_change(self) -> bool:
        return self.phase == "stability"

    def validate(self, scenario: Scenario) -> None:
        if scenario.batch_width != 1:
            raise ValueError("S09 requires the inherited serial scheduler")
        if scenario.scheduler not in {"serial_counter_addressed", "uniform_random_activation"}:
            raise ValueError("S09 requires inherited uniform activation")
        if scenario.architecture.value != "cell_view":
            raise ValueError("S09 requires the inherited cell-view scenario")
        if self.signal == SignalPermission.NONE and self.target_aware:
            # Allowed only as the prespecified exact negative control.
            return

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "targetChangeId": self.target_change.value,
            "signalPermissionId": self.signal.value,
            "phase": self.phase,
            "targetAware": self.target_aware,
            "noChange": self.no_change,
        }


class TargetCodeTransformer:
    """Trusted target projection over records already authorized by E02."""

    __slots__ = ("codes", "occupancy", "record_count", "positions", "identities")

    def __init__(self, codes: Mapping[str, int], occupancy: Sequence[str]) -> None:
        self.codes = dict(codes)
        self.occupancy = tuple(occupancy)
        self.record_count = 0
        self.positions: list[int] = []
        self.identities: list[str] = []

    def __call__(
        self,
        record: ObservationRecord,
        *,
        event_index: int,
        read_ordinal: int,
    ) -> ObservationRecord:
        del event_index, read_ordinal
        self.record_count += 1
        if isinstance(record, ActorView):
            identity = record.cell_id
            self.positions.append(record.position)
            self.identities.append(identity)
            return replace(record, value=self.codes[identity])
        if isinstance(record, VisibleCell):
            identity = self.occupancy[record.position]
            self.positions.append(record.position)
            self.identities.append(identity)
            return replace(record, value=self.codes[identity])
        raise TypeError(type(record))


@dataclass(frozen=True, slots=True)
class ProposalSurface:
    native: Proposal
    candidate: Proposal | None
    emitted: Proposal
    signal_available: bool
    authorized_positions: tuple[int, ...]
    authorized_identities: tuple[str, ...]
    target_record_reads: int
    target_comparisons: int


class TargetChangeController:
    """Matched signal/candidate gateway; only final proposal selection differs."""

    __slots__ = (
        "contract",
        "definition",
        "scenario",
        "start_event",
        "native_router",
        "ledger",
        "audit_digest",
        "audit_count",
        "retained_audits",
        "retain_audits",
    )

    LEDGER_FIELDS = (
        "chargedOpportunities",
        "signalQueries",
        "signalDeliveries",
        "signalPayloadUnits",
        "localRelayAvailableOpportunities",
        "gradientFieldOpportunities",
        "globalBroadcastUnits",
        "targetRecordReads",
        "shadowComparisons",
        "controllerComputations",
        "targetAwareCandidates",
        "targetAwareEmitted",
        "nativeEmitted",
        "abstractEnergyUnits",
    )

    def __init__(
        self,
        contract: TargetChangeContract,
        definition: TargetDefinition,
        scenario: Scenario,
        start_event: int,
        *,
        retain_audits: bool,
    ) -> None:
        self.contract = contract
        self.definition = definition
        self.scenario = scenario
        self.start_event = start_event
        self.native_router = ArchitectureProposalRouter(
            ArchitectureExecutionContract.distributed_local(),
            action_interface=CommonActionInterface(),
        )
        self.ledger: Counter[str] = Counter({field: 0 for field in self.LEDGER_FIELDS})
        self.audit_digest = bytes.fromhex(EMPTY_DIGEST)
        self.audit_count = 0
        self.retained_audits: list[Mapping[str, Any]] = []
        self.retain_audits = retain_audits
        if contract.signal == SignalPermission.GLOBAL:
            n = len(scenario.cells)
            self.ledger["globalBroadcastUnits"] += n
            self.ledger["signalPayloadUnits"] += n
            self.ledger["abstractEnergyUnits"] += n

    def signal_available(self, state: RunState, actor_id: str) -> bool:
        signal = self.contract.signal
        if signal == SignalPermission.NONE:
            return False
        actor_position = state.occupancy.index(actor_id)
        elapsed = state.activation_count - self.start_event
        if signal == SignalPermission.LOCAL:
            self.ledger["signalQueries"] += 1
            self.ledger["abstractEnergyUnits"] += 1
            available = min(actor_position, len(state.occupancy) - 1 - actor_position) <= elapsed
            if available:
                self.ledger["signalDeliveries"] += 1
                self.ledger["signalPayloadUnits"] += 1
                self.ledger["localRelayAvailableOpportunities"] += 1
                self.ledger["abstractEnergyUnits"] += 1
            return available
        if signal == SignalPermission.GRADIENT:
            self.ledger["signalQueries"] += 1
            self.ledger["gradientFieldOpportunities"] += 1
            self.ledger["abstractEnergyUnits"] += 1
            return True
        if signal == SignalPermission.GLOBAL:
            return True
        raise AssertionError(signal)

    def proposal_for(
        self,
        state: RunState,
        actor_id: str,
        *,
        side: str | None,
    ) -> ProposalSurface:
        native = self.native_router.proposal_for(
            self.scenario, state, actor_id, side=side
        )
        available = self.signal_available(state, actor_id)
        candidate: Proposal | None = None
        positions: tuple[int, ...] = ()
        identities: tuple[str, ...] = ()
        target_reads = 0
        target_comparisons = 0
        if available:
            transformer = TargetCodeTransformer(
                self.definition.policy_code_map, state.occupancy
            )
            observation, meter, gateway = build_policy_native_observation(
                self.scenario,
                state,
                actor_id,
                side=side,
                record_transformer=transformer,
            )
            candidate = propose_from_observation(observation, meter)
            observed = (
                gateway.observed_identity_at(candidate.target_pos)
                if candidate.kind == ProposalKind.SWAP and candidate.target_pos is not None
                else None
            )
            candidate = replace(
                candidate,
                observed_target_id=observed,
                reason=f"s09_target_code::{candidate.reason}",
                observation_reads=native.observation_reads,
                value_comparisons=native.value_comparisons,
            )
            positions = tuple(transformer.positions)
            identities = tuple(transformer.identities)
            target_reads = transformer.record_count
            target_comparisons = meter.value_comparisons
            self.ledger["targetRecordReads"] += target_reads
            self.ledger["shadowComparisons"] += target_comparisons
            self.ledger["controllerComputations"] += 1
            self.ledger["targetAwareCandidates"] += 1
            self.ledger["abstractEnergyUnits"] += target_reads + 1
        if self.contract.target_aware and candidate is not None:
            emitted = candidate
            self.ledger["targetAwareEmitted"] += 1
        else:
            emitted = native
            self.ledger["nativeEmitted"] += 1
        self.ledger["chargedOpportunities"] += 1
        return ProposalSurface(
            native,
            candidate,
            emitted,
            available,
            positions,
            identities,
            target_reads,
            target_comparisons,
        )

    def record_outcome(
        self,
        event_index: int,
        actor_id: str,
        actor_position: int,
        side: str | None,
        surface: ProposalSurface,
        decision: str,
        changed: bool,
    ) -> None:
        audit = {
            "eventIndex": event_index,
            "actorId": actor_id,
            "actorPosition": actor_position,
            "nativeSide": side,
            "signalPermissionId": self.contract.signal.value,
            "signalAvailable": surface.signal_available,
            "authorizedPositions": list(surface.authorized_positions),
            "authorizedIdentityCount": len(surface.authorized_identities),
            "targetRecordReads": surface.target_record_reads,
            "targetComparisons": surface.target_comparisons,
            "nativeKind": surface.native.kind.value,
            "nativeTargetPosition": surface.native.target_pos,
            "candidateKind": None if surface.candidate is None else surface.candidate.kind.value,
            "candidateTargetPosition": None if surface.candidate is None else surface.candidate.target_pos,
            "emittedKind": surface.emitted.kind.value,
            "emittedTargetPosition": surface.emitted.target_pos,
            "decision": decision,
            "changed": changed,
        }
        self.audit_digest = hashlib.sha256(
            self.audit_digest + canonical_json_bytes(audit)
        ).digest()
        self.audit_count += 1
        if self.retain_audits:
            self.retained_audits.append(audit)

    def process_ledger(self) -> dict[str, int]:
        return {field: int(self.ledger[field]) for field in self.LEDGER_FIELDS}

    def state_dict(self) -> dict[str, Any]:
        return {
            "signalPermissionId": self.contract.signal.value,
            "targetHash": self.definition.target_hash,
            "targetAware": self.contract.target_aware,
            "noChange": self.contract.no_change,
            "runtimeStreams": [],
        }


@dataclass(frozen=True, slots=True)
class TargetChangeRun:
    contract: TargetChangeContract
    target_definition: TargetDefinition
    source_scenario_id: str
    source_checkpoint_hash: str
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
    retained_process_audits: tuple[Mapping[str, Any], ...]
    opportunity_validation: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": TARGET_CHANGE_RUN_SCHEMA_VERSION,
            "benchmarkVersion": BENCHMARK_VERSION,
            "contract": self.contract.to_dict(),
            "targetDefinition": self.target_definition.to_dict(),
            "sourceScenarioId": self.source_scenario_id,
            "sourceCheckpointHash": self.source_checkpoint_hash,
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
            "retainedProcessAudits": list(self.retained_process_audits),
            "opportunityValidation": dict(self.opportunity_validation),
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def _delta(final: Mapping[str, int], initial: Mapping[str, int]) -> dict[str, int]:
    return {key: int(final.get(key, 0)) - int(initial.get(key, 0)) for key in final}


def _semantic_proposal(proposal: Proposal) -> tuple[Any, ...]:
    return (
        proposal.kind.value,
        proposal.actor_id,
        proposal.actor_pos,
        proposal.target_pos,
        proposal.new_cursor,
        proposal.observed_target_id,
    )


def _expected_contexts(scenario: Scenario) -> set[tuple[str, str | None]]:
    answer: set[tuple[str, str | None]] = set()
    for cell in scenario.cells:
        if cell.policy == Policy.BUBBLE and cell.fault == FaultMode.NORMAL:
            answer.add((cell.cell_id, "left"))
            answer.add((cell.cell_id, "right"))
        else:
            answer.add((cell.cell_id, None))
    return answer


def _signal_surface_stable(signal: SignalPermission, elapsed: int, n: int) -> bool:
    if signal != SignalPermission.LOCAL:
        return True
    return elapsed >= (n - 1) // 2


def _execute_opportunity(
    scenario: Scenario,
    state: RunState,
    controller: TargetChangeController,
    *,
    retain_event: bool,
) -> tuple[Mapping[str, Any] | None, bool, tuple[str, str | None], ProposalSurface]:
    if scenario.batch_width != 1 or state.terminal is not None:
        raise ValueError("S09 execution requires one live serial opportunity")
    event_index = state.activation_count
    actor_id, _, actor_consumed = scheduled_actor(
        scenario, event_index, include_draws=False
    )
    state.stream_counters["actor_activation"] = (
        state.stream_counters.get("actor_activation", 0) + actor_consumed
    )
    actor = scenario.cell_map[actor_id]
    side: str | None = None
    if actor.policy == Policy.BUBBLE and actor.fault == FaultMode.NORMAL:
        side, _ = scheduled_side(scenario, event_index)
        state.stream_counters["bubble_side"] = (
            state.stream_counters.get("bubble_side", 0) + 1
        )
    actor_position = state.occupancy.index(actor_id)
    surface = controller.proposal_for(state, actor_id, side=side)
    proposal = surface.emitted
    validation = validate_proposal(scenario, state, proposal)
    decision = "accepted" if validation.eligible_for_commit else validation.decision
    ledger_delta = cost_delta(state.ledger, proposal, decision)
    changed = False
    if validation.eligible_for_commit:
        snapshot = state.clone()
        changed = commit_proposal(state, snapshot, proposal, decision)
    for key, value in ledger_delta.items():
        state.ledger[key] += value
    state.activation_count += 1
    controller.record_outcome(
        event_index,
        actor_id,
        actor_position,
        side,
        surface,
        decision,
        changed,
    )
    event: Mapping[str, Any] | None = None
    if retain_event:
        event = {
            "eventIndex": event_index,
            "actorId": actor_id,
            "nativeSide": side,
            "signalAvailable": surface.signal_available,
            "nativeProposal": surface.native.to_dict(),
            "candidateProposal": None if surface.candidate is None else surface.candidate.to_dict(),
            "proposal": proposal.to_dict(),
            "decision": decision,
            "changed": changed,
        }
    return event, changed, (actor_id, side), surface


def run_target_change_phase(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    contract: TargetChangeContract,
    adaptation_budget: int,
    probe_budget: int,
    retain_trace: bool = False,
) -> TargetChangeRun:
    """Execute one target-change or paired no-change stability arm."""

    if adaptation_budget < 1 or probe_budget < 1:
        raise ValueError("S09 budgets must be positive")
    contract.validate(scenario)
    definition = build_target_definition(
        scenario, contract.target_change, no_change=contract.no_change
    )
    state = checkpoint.to_run_state()
    state.terminal = None
    start_event = state.activation_count
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    initial_hash = state_hash(scenario.scenario_id, state)
    controller = TargetChangeController(
        contract,
        definition,
        scenario,
        start_event,
        retain_audits=retain_trace,
    )
    process_initial = controller.state_dict()
    retained: list[Mapping[str, Any]] = []
    digest = bytes.fromhex(EMPTY_DIGEST)

    old_distance = definition.old_distance(state.occupancy)
    new_distance = definition.distance(state.occupancy)
    initial_old_distance = old_distance
    initial_new_distance = new_distance
    old_auc = 0
    new_auc = 0
    maximum_new_distance = new_distance
    crossover_time: int | None = 0 if new_distance < old_distance else None
    target_hit_time: int | None = 0 if new_distance == 0 else None
    post_hit_count = 0
    post_hit_auc = 0
    post_hit_maximum = 0
    post_hit_departure = False
    coverage: set[tuple[str, str | None]] = set()
    expected = _expected_contexts(scenario)
    quiescent = False

    if contract.phase == "stability":
        run_limit = probe_budget
    else:
        run_limit = adaptation_budget + probe_budget

    while state.activation_count - start_event < run_limit:
        elapsed_before = state.activation_count - start_event
        if contract.phase == "changed" and target_hit_time is not None and post_hit_count >= probe_budget:
            break
        event, changed, context, surface = _execute_opportunity(
            scenario,
            state,
            controller,
            retain_event=retain_trace,
        )
        if event is not None:
            retained.append(event)
            digest = hashlib.sha256(digest + canonical_json_bytes(event)).digest()
        if changed:
            if surface.emitted.kind == ProposalKind.SWAP:
                assert surface.emitted.target_pos is not None
                old_distance += _applied_swap_inversion_delta(
                    state.occupancy,
                    definition.old_code_map,
                    surface.emitted.actor_pos,
                    surface.emitted.target_pos,
                )
                new_distance += _applied_swap_inversion_delta(
                    state.occupancy,
                    definition.target_code_map,
                    surface.emitted.actor_pos,
                    surface.emitted.target_pos,
                )
            coverage.clear()
        elapsed = state.activation_count - start_event
        old_auc += old_distance
        new_auc += new_distance
        maximum_new_distance = max(maximum_new_distance, new_distance)
        if crossover_time is None and new_distance < old_distance:
            crossover_time = elapsed

        hit_this_opportunity = False
        if contract.phase == "changed" and target_hit_time is None and new_distance == 0:
            target_hit_time = elapsed
            hit_this_opportunity = True
        if contract.phase == "changed" and target_hit_time is not None and not hit_this_opportunity:
            post_hit_count += 1
            post_hit_auc += new_distance
            post_hit_maximum = max(post_hit_maximum, new_distance)
            post_hit_departure = post_hit_departure or new_distance > 0

        if contract.phase == "changed" and target_hit_time is None:
            if changed:
                coverage.clear()
            elif _signal_surface_stable(contract.signal, elapsed_before, len(state.occupancy)):
                coverage.add(context)
            else:
                coverage.clear()
            if coverage == expected:
                quiescent = True
                break

    phase_activations = state.activation_count - start_event
    if contract.phase == "stability":
        state.terminal = "no_change_stability_probe_complete"
    elif target_hit_time is not None:
        state.terminal = "post_adaptation_probe_complete"
    elif quiescent:
        state.terminal = "controller_quiescent"
    else:
        state.terminal = "phase_event_budget"

    ledger_delta = _delta(state.ledger, initial_ledger)
    stream_delta = _delta(state.stream_counters, initial_streams)
    process_final = controller.state_dict()
    process_ledger = controller.process_ledger()
    negative_control_exact = not (
        contract.signal == SignalPermission.NONE
        and contract.target_aware
        and process_ledger["targetAwareCandidates"] != 0
    )
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
        "negativeControlNoCandidate": negative_control_exact,
    }
    summary = {
        "phase": contract.phase,
        "stopReason": state.terminal,
        "targetCompleted": target_hit_time is not None,
        "phaseActivationCount": phase_activations,
        "adaptationBudget": adaptation_budget,
        "probeBudget": probe_budget,
        "initialOldTargetDistance": initial_old_distance,
        "initialNewTargetDistance": initial_new_distance,
        "initialNormalizedNewTargetDistance": initial_new_distance / definition.maximum_distance,
        "finalOldTargetDistance": old_distance,
        "finalNewTargetDistance": new_distance,
        "finalNormalizedNewTargetDistance": new_distance / definition.maximum_distance,
        "maximumNewTargetDistance": maximum_new_distance,
        "oldTargetDistanceAuc": old_auc,
        "newTargetDistanceAuc": new_auc,
        "crossoverTime": crossover_time,
        "crossoverCensored": crossover_time is None,
        "adaptationTime": target_hit_time,
        "adaptationCensored": target_hit_time is None,
        "restrictedAdaptationTime": target_hit_time if target_hit_time is not None else adaptation_budget + 1,
        "overshootCensored": target_hit_time is None,
        "postHitProbeOpportunities": post_hit_count,
        "postHitAnyDeparture": post_hit_departure,
        "postHitMaximumDistance": post_hit_maximum,
        "postHitDistanceAuc": post_hit_auc,
        "postHitFinalTargetRetained": target_hit_time is not None and new_distance == 0,
        "noChangeAnyTargetDeparture": contract.phase == "stability" and maximum_new_distance > 0,
        "noChangeFinalTargetRetained": contract.phase == "stability" and new_distance == 0,
        "quiescenceContextCount": len(coverage),
        "quiescenceContextExpected": len(expected),
        "ledgerDelta": ledger_delta,
        "streamCounterDelta": stream_delta,
        "traceMode": "selected_compact_full" if retain_trace else "digest",
        "retainedEventCount": len(retained),
    }
    return TargetChangeRun(
        contract=contract,
        target_definition=definition,
        source_scenario_id=scenario.scenario_id,
        source_checkpoint_hash=checkpoint.state_hash,
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
        retained_process_audits=tuple(controller.retained_audits),
        opportunity_validation=opportunity_validation,
    )


def exact_replay_target_change(
    run: TargetChangeRun,
    scenario: Scenario,
    checkpoint: Checkpoint,
    adaptation_budget: int,
    probe_budget: int,
) -> None:
    replay = run_target_change_phase(
        scenario,
        checkpoint,
        contract=run.contract,
        adaptation_budget=adaptation_budget,
        probe_budget=probe_budget,
        retain_trace=bool(run.events),
    )
    if run.to_json_bytes() != replay.to_json_bytes():
        raise AssertionError("S09 target-change exact replay mismatch")


def target_change_case_id(
    s01_pairing_block_id: str,
    timing_condition_id: str,
    target_change: TargetChange,
    phase: str,
) -> str:
    return "e05tc9:" + sha256_json(
        {
            "s01PairingBlockId": s01_pairing_block_id,
            "timingConditionId": timing_condition_id,
            "targetChangeId": target_change.value,
            "phase": phase,
        }
    )


def semantic_proposal_equal(left: Proposal, right: Proposal) -> bool:
    """Compare behavior while ignoring reason and accounting annotations."""

    return _semantic_proposal(left) == _semantic_proposal(right)
