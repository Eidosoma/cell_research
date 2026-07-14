"""S03 matched architecture contracts over the S02 action interface.

The weak coordinator is deliberately split from the trusted signal encoder.
Its ``decide`` method receives only a two-field :class:`CoordinatorSignal`,
never a scenario, run state, action envelope, cell value, or identity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, Literal

from reference_simulator.engine import run
from reference_simulator.model import (
    Architecture,
    Proposal,
    ProposalKind,
    RunResult,
    RunState,
    Scenario,
    canonical_json_bytes,
)
from reference_simulator.transition_primitives import ledger_identity

from .action_interface import (
    ActionEnvelope,
    CommonActionInterface,
    ControlTopology,
    InformationPermission,
)


ARCHITECTURE_INTERFACE_VERSION = "E02-matched-architectures-v1"
PRESPECIFICATION_SHA256 = "c87356a66bbd4f4c072ad23513611701047cca0178a3ec533478e01194b2c15f"


class ControlArchitecture(str, Enum):
    CENTRAL_GLOBAL_LEGACY = "central_global_legacy"
    CENTRAL_LOCAL_PROPOSAL_K1 = "central_local_proposal_k1"
    DISTRIBUTED_LOCAL = "distributed_local"
    DISTRIBUTED_WEAK_COORDINATOR = "distributed_weak_coordinator"


class CoordinatorProfile(str, Enum):
    LEGACY_GLOBAL = "legacy_global"
    COMMON_VALIDATOR_ONLY = "common_validator_only"
    NONE = "none"
    WEAK_FROZEN_BUDGET = "weak_frozen_budget"


@dataclass(frozen=True, slots=True)
class WeakCoordinatorParameters:
    """Typed bandwidth/frequency parameters for the frozen weak rule.

    The S03 evidence uses the prespecified p8/phase7 values.  Period and phase
    remain explicit parameters so later authorized factorial designs can vary
    intervention frequency without changing information or action semantics.
    Boolean payloads require exactly one bit in each direction.
    """

    period_opportunities: int = 8
    phase_zero_based: int = 7
    inbound_payload_bits: int = 1
    outbound_payload_bits: int = 1

    def __post_init__(self) -> None:
        if self.period_opportunities < 1:
            raise ValueError("coordinator period must be positive")
        if not 0 <= self.phase_zero_based < self.period_opportunities:
            raise ValueError("coordinator phase must be inside its period")
        if self.inbound_payload_bits != 1 or self.outbound_payload_bits != 1:
            raise ValueError("the frozen Boolean signal and response require one bit each")

    def eligible(self, event_index: int) -> bool:
        if event_index < 0:
            raise ValueError("event index must be nonnegative")
        return event_index % self.period_opportunities == self.phase_zero_based

    @property
    def messages_per_eligible_opportunity(self) -> int:
        return 2

    @property
    def bits_per_eligible_opportunity(self) -> int:
        return self.inbound_payload_bits + self.outbound_payload_bits

    @property
    def is_prespecified_s03_profile(self) -> bool:
        return self == WeakCoordinatorParameters()

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "periodOpportunities": self.period_opportunities,
            "phaseZeroBased": self.phase_zero_based,
            "inboundPayloadBits": self.inbound_payload_bits,
            "outboundPayloadBits": self.outbound_payload_bits,
            "messagesPerEligibleOpportunity": self.messages_per_eligible_opportunity,
            "bitsPerEligibleOpportunity": self.bits_per_eligible_opportunity,
            "isPrespecifiedS03Profile": self.is_prespecified_s03_profile,
        }


@dataclass(frozen=True, slots=True)
class CoordinatorSignal:
    """The complete input surface visible to the weak coordinator."""

    event_index: int
    is_nonlocal_swap: bool


@dataclass(frozen=True, slots=True)
class CoordinatorDecision:
    """The complete one-bit output surface owned by the weak coordinator."""

    allow: bool


class FrozenWeakCoordinator:
    """A stateless coordinator that can inspect only one encoded Boolean."""

    __slots__ = ("parameters",)

    def __init__(self, parameters: WeakCoordinatorParameters) -> None:
        self.parameters = parameters

    def decide(self, signal: CoordinatorSignal) -> CoordinatorDecision:
        if not isinstance(signal, CoordinatorSignal):
            raise TypeError("weak coordinator accepts CoordinatorSignal only")
        if not self.parameters.eligible(signal.event_index):
            raise ValueError("weak coordinator may decide only on eligible opportunities")
        return CoordinatorDecision(allow=not signal.is_nonlocal_swap)


def encode_coordinator_signal(envelope: ActionEnvelope, event_index: int) -> CoordinatorSignal:
    """Trusted one-bit projection from an already-built S02 envelope."""

    if not isinstance(envelope, ActionEnvelope):
        raise TypeError("signal encoder accepts ActionEnvelope only")
    proposal = envelope.proposal
    nonlocal_swap = (
        proposal.kind == ProposalKind.SWAP
        and proposal.target_pos is not None
        and abs(proposal.target_pos - proposal.actor_pos) > 1
    )
    return CoordinatorSignal(event_index=event_index, is_nonlocal_swap=nonlocal_swap)


@dataclass(frozen=True, slots=True)
class ArchitectureExecutionContract:
    architecture: ControlArchitecture
    coordinator_profile: CoordinatorProfile
    weak_parameters: WeakCoordinatorParameters | None = None
    continuation_policy: str = "skip_and_continue"
    retry_policy: str = "no_retry"
    action_failure: str = "none"
    sensing_error: str = "exact"
    scheduler: str = "uniform_random_activation"
    legal_primitives: tuple[str, ...] = ("NoOp", "Swap", "MemoryUpdate")

    @classmethod
    def central_global_legacy(cls) -> "ArchitectureExecutionContract":
        return cls(
            ControlArchitecture.CENTRAL_GLOBAL_LEGACY,
            CoordinatorProfile.LEGACY_GLOBAL,
            scheduler="traditional_controller",
        )

    @classmethod
    def central_local_k1(cls) -> "ArchitectureExecutionContract":
        return cls(
            ControlArchitecture.CENTRAL_LOCAL_PROPOSAL_K1,
            CoordinatorProfile.COMMON_VALIDATOR_ONLY,
        )

    @classmethod
    def distributed_local(cls) -> "ArchitectureExecutionContract":
        return cls(ControlArchitecture.DISTRIBUTED_LOCAL, CoordinatorProfile.NONE)

    @classmethod
    def distributed_weak(
        cls,
        *,
        enabled: bool = True,
        parameters: WeakCoordinatorParameters | None = None,
    ) -> "ArchitectureExecutionContract":
        if not enabled and parameters is not None:
            raise ValueError("disabled coordinator cannot have weak parameters")
        return cls(
            ControlArchitecture.DISTRIBUTED_WEAK_COORDINATOR,
            CoordinatorProfile.WEAK_FROZEN_BUDGET if enabled else CoordinatorProfile.NONE,
            parameters or (WeakCoordinatorParameters() if enabled else None),
        )

    @property
    def matched_contrast_eligible(self) -> bool:
        return self.architecture != ControlArchitecture.CENTRAL_GLOBAL_LEGACY

    @property
    def information_permission(self) -> InformationPermission:
        return (
            InformationPermission.FULL_GLOBAL_STATE
            if self.architecture == ControlArchitecture.CENTRAL_GLOBAL_LEGACY
            else InformationPermission.POLICY_NATIVE_LOCAL
        )

    @property
    def proposal_candidates_per_opportunity(self) -> int | None:
        return None if not self.matched_contrast_eligible else 1

    def validate(self, scenario: Scenario) -> None:
        self.validate_policy_action_boundary(scenario)
        if self.matched_contrast_eligible and self.scheduler != "uniform_random_activation":
            raise ValueError("matched S03 architecture cannot alter actor scheduling")

    def validate_policy_action_boundary(self, scenario: Scenario) -> None:
        """Validate S03's frozen boundary independently of an S04 treatment.

        S04 owns the out-of-band scheduler assignment.  This method preserves
        all S03 information, action, architecture, fault, retry, and base
        scenario requirements without pretending that the S03 baseline
        scheduler label is the active S04 scheduler family.
        """
        if self.continuation_policy != "skip_and_continue":
            raise ValueError("unsupported continuation policy")
        if self.retry_policy != "no_retry":
            raise ValueError("unsupported retry policy")
        if self.action_failure != "none" or self.sensing_error != "exact":
            raise ValueError("S03 supports only no action failure and exact sensing")
        if self.legal_primitives != ("NoOp", "Swap", "MemoryUpdate"):
            raise ValueError("S03 legal primitive set is frozen by S02")

        if self.architecture == ControlArchitecture.CENTRAL_GLOBAL_LEGACY:
            if self.coordinator_profile != CoordinatorProfile.LEGACY_GLOBAL:
                raise ValueError("legacy global architecture requires legacy_global profile")
            if self.weak_parameters is not None:
                raise ValueError("legacy global architecture cannot use weak parameters")
            if scenario.architecture != Architecture.TRADITIONAL:
                raise ValueError("legacy global contract requires a traditional scenario")
            if self.scheduler != "traditional_controller":
                raise ValueError("legacy global contract requires its declared controller scheduler")
            return

        if scenario.architecture != Architecture.CELL_VIEW:
            raise ValueError("matched architectures reject full-global traditional scenarios")
        if scenario.scheduler != "serial_counter_addressed" or scenario.batch_width != 1:
            raise ValueError("matched S03 architectures require one serial opportunity")
        expected_profiles = {
            ControlArchitecture.CENTRAL_LOCAL_PROPOSAL_K1: {
                CoordinatorProfile.COMMON_VALIDATOR_ONLY
            },
            ControlArchitecture.DISTRIBUTED_LOCAL: {CoordinatorProfile.NONE},
            ControlArchitecture.DISTRIBUTED_WEAK_COORDINATOR: {
                CoordinatorProfile.NONE,
                CoordinatorProfile.WEAK_FROZEN_BUDGET,
            },
        }
        if self.coordinator_profile not in expected_profiles[self.architecture]:
            raise ValueError("coordinator profile is incompatible with architecture")
        if self.coordinator_profile == CoordinatorProfile.WEAK_FROZEN_BUDGET:
            if self.weak_parameters is None:
                raise ValueError("weak coordinator requires explicit parameters")
        elif self.weak_parameters is not None:
            raise ValueError("only the weak coordinator may have weak parameters")

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture.value,
            "coordinatorProfile": self.coordinator_profile.value,
            "matchedContrastEligible": self.matched_contrast_eligible,
            "informationPermission": self.information_permission.value,
            "proposalCandidatesPerOpportunity": self.proposal_candidates_per_opportunity,
            "weakParameters": self.weak_parameters.to_dict() if self.weak_parameters else None,
            "continuationPolicy": self.continuation_policy,
            "retryPolicy": self.retry_policy,
            "actionFailure": self.action_failure,
            "sensingError": self.sensing_error,
            "scheduler": self.scheduler,
            "legalPrimitives": list(self.legal_primitives),
            "prespecificationSha256": PRESPECIFICATION_SHA256,
        }


@dataclass(frozen=True, slots=True)
class CoordinationAudit:
    event_index: int
    incoming_bit: bool
    outgoing_allow_bit: bool
    original_proposal_kind: ProposalKind
    selected_proposal_kind: ProposalKind
    original_reason: str
    selected_reason: str
    intervened: bool
    message_count: int = 2
    message_bits: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventIndex": self.event_index,
            "incomingBitIsNonlocalSwap": self.incoming_bit,
            "outgoingBitAllow": self.outgoing_allow_bit,
            "originalProposalKind": self.original_proposal_kind.value,
            "selectedProposalKind": self.selected_proposal_kind.value,
            "originalReason": self.original_reason,
            "selectedReason": self.selected_reason,
            "intervened": self.intervened,
            "messageCount": self.message_count,
            "messageBits": self.message_bits,
        }


def _coordinator_veto(proposal: Proposal) -> Proposal:
    return replace(
        proposal,
        kind=ProposalKind.NO_OP,
        target_pos=None,
        new_cursor=None,
        reason="coordinator_veto_nonlocal_swap",
        observed_target_id=None,
    )


class ArchitectureProposalRouter:
    """Routes one S02 proposal while keeping coordinator access separately typed."""

    __slots__ = ("interface", "contract", "coordinator", "audits")

    def __init__(
        self,
        contract: ArchitectureExecutionContract,
        *,
        action_interface: CommonActionInterface | None = None,
    ) -> None:
        if not contract.matched_contrast_eligible:
            raise ValueError("legacy global control does not use the matched proposal router")
        self.interface = action_interface or CommonActionInterface()
        self.contract = contract
        self.coordinator = (
            FrozenWeakCoordinator(contract.weak_parameters)
            if contract.coordinator_profile == CoordinatorProfile.WEAK_FROZEN_BUDGET
            and contract.weak_parameters is not None
            else None
        )
        self.audits: list[CoordinationAudit] = []

    def proposal_for(
        self,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        side: Literal["left", "right"] | None = None,
    ) -> Proposal:
        topology = (
            ControlTopology.CENTRAL_LOCAL_PROPOSAL_K1
            if self.contract.architecture == ControlArchitecture.CENTRAL_LOCAL_PROPOSAL_K1
            else ControlTopology.DISTRIBUTED_LOCAL
        )
        envelope = self.interface.envelope_for(topology, scenario, state, actor_id, side=side)
        event_index = state.activation_count
        return self._apply_coordinator(envelope, event_index)

    def route_existing_envelope(
        self,
        envelope: ActionEnvelope,
        event_index: int,
    ) -> Proposal:
        """Route one already-charged proposal through the frozen coordinator.

        S05 deferred retries use this path to preserve the global coordinator
        phase and ownership rule without constructing a second policy proposal.
        """
        if not isinstance(envelope, ActionEnvelope):
            raise TypeError("existing proposal route accepts ActionEnvelope only")
        if self.contract.architecture == ControlArchitecture.CENTRAL_LOCAL_PROPOSAL_K1:
            envelope = self.interface.central_relay.forward_one((envelope,))
        return self._apply_coordinator(envelope, event_index)

    def _apply_coordinator(
        self,
        envelope: ActionEnvelope,
        event_index: int,
    ) -> Proposal:
        if self.coordinator is None:
            return envelope.proposal
        parameters = self.coordinator.parameters
        if not parameters.eligible(event_index):
            return envelope.proposal
        signal = encode_coordinator_signal(envelope, event_index)
        decision = self.coordinator.decide(signal)
        selected = envelope.proposal if decision.allow else _coordinator_veto(envelope.proposal)
        self.audits.append(
            CoordinationAudit(
                event_index=event_index,
                incoming_bit=signal.is_nonlocal_swap,
                outgoing_allow_bit=decision.allow,
                original_proposal_kind=envelope.proposal.kind,
                selected_proposal_kind=selected.kind,
                original_reason=envelope.proposal.reason,
                selected_reason=selected.reason,
                intervened=not decision.allow,
                message_bits=parameters.bits_per_eligible_opportunity,
            )
        )
        return selected

    def architecture_ledger(self) -> dict[str, int]:
        return {
            "coordinatorMessages": sum(item.message_count for item in self.audits),
            "coordinatorMessageBits": sum(item.message_bits for item in self.audits),
            "coordinatorEligibleDecisions": len(self.audits),
            "coordinatorInterventions": sum(item.intervened for item in self.audits),
        }


def _zero_architecture_ledger() -> dict[str, int]:
    return {
        "coordinatorMessages": 0,
        "coordinatorMessageBits": 0,
        "coordinatorEligibleDecisions": 0,
        "coordinatorInterventions": 0,
    }


@dataclass(frozen=True, slots=True)
class ArchitectureRun:
    contract: ArchitectureExecutionContract
    result: RunResult
    coordination_audit: tuple[CoordinationAudit, ...] = ()
    architecture_ledger: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.architecture_ledger is None:
            object.__setattr__(self, "architecture_ledger", _zero_architecture_ledger())

    def extended_cost_ledger(self) -> dict[str, int]:
        return {
            **dict(self.result.summary["ledger"]),
            **dict(self.architecture_ledger or _zero_architecture_ledger()),
        }

    def opportunity_validation(self) -> dict[str, bool]:
        native = dict(self.result.summary["ledger"])
        architecture = dict(self.architecture_ledger or _zero_architecture_ledger())
        checks = dict(ledger_identity(native))
        checks.update(
            {
                "oneCandidatePerMatchedOpportunity": (
                    not self.contract.matched_contrast_eligible
                    or self.contract.proposal_candidates_per_opportunity == 1
                ),
                "eligibleAuditCountMatchesLedger": len(self.coordination_audit)
                == architecture["coordinatorEligibleDecisions"],
                "messageCountMatchesEligibleDecisions": architecture["coordinatorMessages"]
                == 2 * architecture["coordinatorEligibleDecisions"],
                "interventionsDoNotExceedEligibleDecisions": architecture[
                    "coordinatorInterventions"
                ]
                <= architecture["coordinatorEligibleDecisions"],
            }
        )
        if self.contract.weak_parameters is not None:
            checks["messageBitsMatchBudget"] = architecture["coordinatorMessageBits"] == (
                self.contract.weak_parameters.bits_per_eligible_opportunity
                * architecture["coordinatorEligibleDecisions"]
            )
            checks["auditIndicesMatchFrequency"] = all(
                self.contract.weak_parameters.eligible(item.event_index)
                for item in self.coordination_audit
            )
        else:
            checks["messageBitsMatchBudget"] = architecture["coordinatorMessageBits"] == 0
            checks["auditIndicesMatchFrequency"] = not self.coordination_audit
        return checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "E02.architecture-run.v1",
            "architectureInterfaceVersion": ARCHITECTURE_INTERFACE_VERSION,
            "contract": self.contract.to_dict(),
            "nativeResult": self.result.to_dict(),
            "coordinationAudit": [item.to_dict() for item in self.coordination_audit],
            "architectureLedger": dict(self.architecture_ledger or _zero_architecture_ledger()),
            "extendedCostLedger": self.extended_cost_ledger(),
            "opportunityValidation": self.opportunity_validation(),
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def run_architecture(
    scenario: Scenario,
    contract: ArchitectureExecutionContract,
    *,
    trace_mode: Literal["full", "digest", "none"] = "digest",
) -> ArchitectureRun:
    contract.validate(scenario)
    if contract.architecture == ControlArchitecture.CENTRAL_GLOBAL_LEGACY:
        result = run(scenario, trace_mode=trace_mode)
        return ArchitectureRun(contract, result)
    router = ArchitectureProposalRouter(contract)
    result = run(scenario, trace_mode=trace_mode, proposal_factory=router.proposal_for)
    return ArchitectureRun(
        contract,
        result,
        tuple(router.audits),
        router.architecture_ledger(),
    )


def exact_replay_architecture(run_result: ArchitectureRun) -> ArchitectureRun:
    replayed = run_architecture(
        run_result.result.scenario,
        run_result.contract,
        trace_mode=run_result.result.summary["traceMode"],
    )
    if replayed.to_json_bytes() != run_result.to_json_bytes():
        raise AssertionError("architecture replay mismatch")
    return replayed


def coordinator_input_fields() -> tuple[str, ...]:
    """Stable introspection surface used by permission validation."""

    return tuple(item.name for item in fields(CoordinatorSignal))
