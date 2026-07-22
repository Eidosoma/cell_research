"""S05 matched fault primitives composed over the frozen S02--S04 contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal

from reference_simulator.engine import run
from reference_simulator.model import (
    FaultMode,
    Proposal,
    ProposalKind,
    RunState,
    Scenario,
    canonical_json_bytes,
)
from reference_simulator.rng import u64
from reference_simulator.transition_primitives import (
    ValidationDecision,
    ledger_identity,
)

from .action_interface import (
    ActionEnvelope,
    ActorView,
    CommonActionInterface,
    InformationPermission,
    ObservationRecord,
)
from .architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
    ArchitectureRun,
    ControlArchitecture,
)
from .schedulers import (
    FrozenSchedulerController,
    SchedulerExecutionContract,
    SchedulerRun,
)


FAULT_INTERFACE_VERSION = "E02-matched-faults-v1"
BERNOULLI_FAILURE_STREAM = "action_failure_bernoulli_s05_v1"
TRANSIENT_FAILURE_STREAM = "action_failure_transient_s05_v1"
SENSING_VALUE_STREAM = "sensing_value_error_s05_v1"
SENSING_STATUS_STREAM = "sensing_status_error_s05_v1"


class MobilityProfile(str, Enum):
    NORMAL = "normal"
    PASSIVE = "passive"
    STUCK = "stuck"


class ContinuationPolicy(str, Enum):
    SKIP_AND_CONTINUE = "skip_and_continue"
    STOP_ON_FIRST_BLOCKING_FAILURE = "stop_on_first_blocking_failure"


class RetryPolicy(str, Enum):
    NO_RETRY = "no_retry"
    RETRY_LATER_BOUNDED = "retry_later_bounded"


class ActionFailureProfile(str, Enum):
    NONE = "none"
    BERNOULLI_P = "bernoulli_p"
    TRANSIENT_MARKOV = "transient_markov"


class SensingProfile(str, Enum):
    EXACT = "exact"
    NOISY_VALUE_OR_STATUS = "noisy_value_or_status"


@dataclass(frozen=True, slots=True)
class FaultExecutionContract:
    mobility: MobilityProfile = MobilityProfile.NORMAL
    continuation: ContinuationPolicy = ContinuationPolicy.SKIP_AND_CONTINUE
    retry: RetryPolicy = RetryPolicy.NO_RETRY
    action_failure: ActionFailureProfile = ActionFailureProfile.NONE
    sensing: SensingProfile = SensingProfile.EXACT
    retry_minimum_intervening_opportunities: int = 1
    retry_maximum_attempts: int = 2
    legal_primitives: tuple[str, ...] = ("NoOp", "Swap", "MemoryUpdate")
    information_permission: str = "policy_native_local"
    proposal_candidates_per_opportunity: int = 1

    def validate(
        self,
        scenario: Scenario,
        architecture: ArchitectureExecutionContract,
        scheduler: SchedulerExecutionContract,
    ) -> None:
        scheduler.validate(scenario, architecture)
        if not architecture.matched_contrast_eligible:
            raise ValueError("central_global_legacy is non-comparable in S05")
        if self.retry_minimum_intervening_opportunities != 1:
            raise ValueError("S05 retry delay is frozen at one intervening opportunity")
        if self.retry_maximum_attempts != 2:
            raise ValueError("S05 retry cap is frozen at two attempts")
        if self.legal_primitives != ("NoOp", "Swap", "MemoryUpdate"):
            raise ValueError("S05 legal primitive set is frozen by S02")
        if self.information_permission != "policy_native_local":
            raise ValueError("S05 cannot change policy information permission")
        if self.proposal_candidates_per_opportunity != 1:
            raise ValueError("S05 forbids free proposal candidates")
        observed = {
            cell.fault for cell in scenario.cells if cell.fault != FaultMode.NORMAL
        }
        if self.mobility == MobilityProfile.NORMAL and observed:
            raise ValueError("normal mobility profile requires a no-fault scenario")
        if self.mobility != MobilityProfile.NORMAL:
            required = FaultMode(self.mobility.value)
            if not observed or observed != {required}:
                raise ValueError(
                    f"{self.mobility.value} profile requires at least one exclusively "
                    f"{self.mobility.value} scenario fault"
                )

    @property
    def degenerate_retry_combination(self) -> bool:
        return (
            self.continuation == ContinuationPolicy.STOP_ON_FIRST_BLOCKING_FAILURE
            and self.retry == RetryPolicy.RETRY_LATER_BOUNDED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mobility": self.mobility.value,
            "continuation": self.continuation.value,
            "retry": self.retry.value,
            "actionFailure": self.action_failure.value,
            "sensing": self.sensing.value,
            "retryMinimumInterveningOpportunities": self.retry_minimum_intervening_opportunities,
            "retryMaximumAttempts": self.retry_maximum_attempts,
            "legalPrimitives": list(self.legal_primitives),
            "informationPermission": self.information_permission,
            "proposalCandidatesPerOpportunity": self.proposal_candidates_per_opportunity,
            "degenerateRetryCombination": self.degenerate_retry_combination,
        }


def _dyadic_hit(raw: int, numerator: int, denominator: int) -> bool:
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError("S05 probability denominator must be a positive power of two")
    return raw < ((1 << 64) * numerator // denominator)


@dataclass(frozen=True, slots=True)
class SensingAudit:
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


class FrozenSensingTransformer:
    """Counter-addressed corruption of already-authorized observation fields."""

    __slots__ = ("profile", "seed", "scenario_id", "audits", "draws_by_event")

    def __init__(self, contract: FaultExecutionContract, scenario: Scenario) -> None:
        self.profile = contract.sensing
        self.seed = scenario.seed
        self.scenario_id = scenario.scenario_id
        self.audits: list[SensingAudit] = []
        self.draws_by_event: dict[int, list[tuple[str, int, int, int]]] = {}

    def _draw(self, stream: str, event_index: int, read_ordinal: int) -> int:
        raw = u64(self.seed, self.scenario_id, stream, event_index, read_ordinal)
        self.draws_by_event.setdefault(event_index, []).append(
            (stream, event_index, read_ordinal, raw)
        )
        return raw

    def __call__(
        self,
        record: ObservationRecord,
        *,
        event_index: int,
        read_ordinal: int,
    ) -> ObservationRecord:
        if self.profile == SensingProfile.EXACT:
            return record
        raw_value = self._draw(SENSING_VALUE_STREAM, event_index, read_ordinal)
        value_hit = _dyadic_hit(raw_value, 1, 8)
        visible_value = record.value
        if value_hit:
            visible_value = record.value + (1 if ((raw_value >> 60) & 1) else -1)
        self.audits.append(
            SensingAudit(
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
            # Actor self fault is protected so S04's side-draw contract cannot
            # be altered by a sensing treatment.
            return replace(record, value=visible_value)

        raw_status = self._draw(SENSING_STATUS_STREAM, event_index, read_ordinal)
        status_hit = _dyadic_hit(raw_status, 1, 8)
        visible_fault = record.fault
        if status_hit:
            alternatives = tuple(item for item in FaultMode if item != record.fault)
            visible_fault = alternatives[(raw_status >> 60) & 1]
        self.audits.append(
            SensingAudit(
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
        return replace(record, value=visible_value, fault=visible_fault)

    def take_draws(self, event_index: int) -> tuple[tuple[str, int, int, int], ...]:
        return tuple(self.draws_by_event.pop(event_index, ()))

    def ledger(self) -> dict[str, int]:
        value = [item for item in self.audits if item.field == "value"]
        status = [item for item in self.audits if item.field == "target_status"]
        return {
            "sensingValueDraws": len(value),
            "sensingStatusDraws": len(status),
            "sensingValueErrors": sum(item.applied for item in value),
            "sensingStatusErrors": sum(item.applied for item in status),
            "sensingErrorsApplied": sum(item.applied for item in self.audits),
            "sensingErrorHandlingOperations": sum(item.applied for item in self.audits),
        }


@dataclass(frozen=True, slots=True)
class RetryAudit:
    event_index: int
    actor_id: str
    disposition: str
    retry_attempt: int
    eligible_event_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventIndex": self.event_index,
            "actorId": self.actor_id,
            "disposition": self.disposition,
            "retryAttempt": self.retry_attempt,
            "eligibleEventIndex": self.eligible_event_index,
        }


@dataclass(frozen=True, slots=True)
class _PendingRetry:
    proposal: Proposal
    retry_attempt: int
    eligible_event_index: int
    original_reason: str


class FaultProposalRouter:
    """Return one fresh or deferred proposal for the scheduler-selected actor."""

    __slots__ = (
        "contract",
        "architecture_router",
        "sensing",
        "pending",
        "attempt_by_event",
        "audits",
    )

    def __init__(
        self,
        contract: FaultExecutionContract,
        architecture_router: ArchitectureProposalRouter,
        sensing: FrozenSensingTransformer,
    ) -> None:
        self.contract = contract
        self.architecture_router = architecture_router
        self.sensing = sensing
        self.pending: dict[str, _PendingRetry] = {}
        self.attempt_by_event: dict[int, int] = {}
        self.audits: list[RetryAudit] = []

    def proposal_for(
        self,
        scenario: Scenario,
        state: RunState,
        actor_id: str,
        *,
        side: Literal["left", "right"] | None = None,
    ) -> Proposal:
        event_index = state.activation_count
        pending = self.pending.get(actor_id)
        if (
            self.contract.retry == RetryPolicy.RETRY_LATER_BOUNDED
            and pending is not None
            and event_index >= pending.eligible_event_index
        ):
            del self.pending[actor_id]
            proposal = replace(
                pending.proposal,
                reason=(
                    f"retry_later_bounded_attempt_{pending.retry_attempt}:"
                    f"{pending.original_reason}"
                ),
                observation_reads=0,
                value_comparisons=0,
                ordinal=0,
                priority=0,
                random_draws=(),
            )
            envelope = ActionEnvelope(
                proposal=proposal,
                observed_target_id=proposal.observed_target_id,
                permission=InformationPermission.POLICY_NATIVE_LOCAL,
                used_capabilities=(),
            )
            proposal = self.architecture_router.route_existing_envelope(
                envelope, event_index
            )
            self.attempt_by_event[event_index] = pending.retry_attempt
            self.audits.append(
                RetryAudit(
                    event_index,
                    actor_id,
                    "attempted",
                    pending.retry_attempt,
                    pending.eligible_event_index,
                )
            )
            return proposal

        proposal = self.architecture_router.proposal_for(
            scenario, state, actor_id, side=side
        )
        sensing_draws = self.sensing.take_draws(event_index)
        if sensing_draws:
            proposal = replace(
                proposal,
                random_draws=proposal.random_draws + sensing_draws,
            )
        return proposal

    def observe_action_failure(self, proposal: Proposal, event_index: int) -> None:
        if self.contract.retry != RetryPolicy.RETRY_LATER_BOUNDED:
            return
        prior_attempt = self.attempt_by_event.get(event_index, 0)
        next_attempt = prior_attempt + 1
        if next_attempt > self.contract.retry_maximum_attempts:
            self.audits.append(
                RetryAudit(event_index, proposal.actor_id, "exhausted", prior_attempt)
            )
            return
        if proposal.actor_id in self.pending:
            self.audits.append(
                RetryAudit(
                    event_index, proposal.actor_id, "queue_collision", next_attempt
                )
            )
            return
        base_reason = proposal.reason
        if prior_attempt:
            marker = f"retry_later_bounded_attempt_{prior_attempt}:"
            if base_reason.startswith(marker):
                base_reason = base_reason[len(marker) :]
        eligible = (
            event_index + self.contract.retry_minimum_intervening_opportunities + 1
        )
        stripped = replace(
            proposal,
            reason=base_reason,
            ordinal=0,
            priority=0,
            random_draws=(),
        )
        self.pending[proposal.actor_id] = _PendingRetry(
            stripped, next_attempt, eligible, base_reason
        )
        self.audits.append(
            RetryAudit(event_index, proposal.actor_id, "queued", next_attempt, eligible)
        )

    def ledger(self) -> dict[str, int]:
        return {
            "retryQueued": sum(item.disposition == "queued" for item in self.audits),
            "retryAttempts": sum(
                item.disposition == "attempted" for item in self.audits
            ),
            "retryExhausted": sum(
                item.disposition == "exhausted" for item in self.audits
            ),
            "retryQueueCollisions": sum(
                item.disposition == "queue_collision" for item in self.audits
            ),
            "retryPendingAtStop": len(self.pending),
        }


@dataclass(slots=True)
class ActionFailureAudit:
    event_index: int
    stream: str | None
    raw_uint64: int | None
    transient_state_failed: bool
    mechanically_eligible: bool = False
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventIndex": self.event_index,
            "stream": self.stream,
            "rawUint64": self.raw_uint64,
            "transientStateFailed": self.transient_state_failed,
            "mechanicallyEligible": self.mechanically_eligible,
            "applied": self.applied,
        }


@dataclass(frozen=True, slots=True)
class BlockingFailureAudit:
    event_index: int
    actor_id: str
    disposition: str
    caused_stop: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventIndex": self.event_index,
            "actorId": self.actor_id,
            "disposition": self.disposition,
            "causedStop": self.caused_stop,
        }


class FaultExecutionInterceptor:
    """Post-construction failure, retry observation, and continuation hook."""

    __slots__ = (
        "contract",
        "seed",
        "scenario_id",
        "retry_router",
        "transient_failed",
        "action_audits",
        "action_by_event",
        "blocking_audits",
    )

    def __init__(
        self,
        contract: FaultExecutionContract,
        scenario: Scenario,
        retry_router: FaultProposalRouter,
    ) -> None:
        self.contract = contract
        self.seed = scenario.seed
        self.scenario_id = scenario.scenario_id
        self.retry_router = retry_router
        self.transient_failed = False
        self.action_audits: list[ActionFailureAudit] = []
        self.action_by_event: dict[int, ActionFailureAudit] = {}
        self.blocking_audits: list[BlockingFailureAudit] = []

    def prepare(
        self,
        proposal: Proposal,
        event_index: int,
    ) -> tuple[Proposal, tuple[tuple[str, int], ...]]:
        profile = self.contract.action_failure
        stream: str | None = None
        raw: int | None = None
        active = False
        if profile == ActionFailureProfile.BERNOULLI_P:
            stream = BERNOULLI_FAILURE_STREAM
            raw = u64(self.seed, self.scenario_id, stream, event_index, 0)
            active = _dyadic_hit(raw, 1, 4)
        elif profile == ActionFailureProfile.TRANSIENT_MARKOV:
            stream = TRANSIENT_FAILURE_STREAM
            raw = u64(self.seed, self.scenario_id, stream, event_index, 0)
            if self.transient_failed:
                if _dyadic_hit(raw, 1, 2):
                    self.transient_failed = False
            elif _dyadic_hit(raw, 1, 8):
                self.transient_failed = True
            active = self.transient_failed
        audit = ActionFailureAudit(event_index, stream, raw, active)
        self.action_audits.append(audit)
        self.action_by_event[event_index] = audit
        if stream is None or raw is None:
            return proposal, ()
        return (
            replace(
                proposal,
                random_draws=proposal.random_draws + ((stream, event_index, 0, raw),),
            ),
            ((stream, 1),),
        )

    def outcome(
        self,
        proposal: Proposal,
        validation: ValidationDecision,
        event_index: int,
    ) -> ValidationDecision:
        audit = self.action_by_event[event_index]
        audit.mechanically_eligible = validation.eligible_for_commit
        if audit.transient_state_failed and validation.eligible_for_commit:
            audit.applied = True
            return ValidationDecision("action_failure", False)
        return validation

    @staticmethod
    def _blocking_disposition(
        scenario: Scenario,
        proposal: Proposal,
        decision: str,
    ) -> str | None:
        if decision in {
            "action_failure",
            "rejected_actor_stuck",
            "rejected_target_stuck",
        }:
            return decision
        if (
            proposal.kind == ProposalKind.NO_OP
            and proposal.reason == "actor_fault"
            and scenario.cell_map[proposal.actor_id].fault != FaultMode.NORMAL
        ):
            return "actor_fault"
        return None

    def after_batch(
        self,
        scenario: Scenario,
        state: RunState,
        proposals: tuple[Proposal, ...],
        decisions: dict[int, str],
        batch_start_index: int,
    ) -> None:
        blocking: list[tuple[int, Proposal, str]] = []
        for proposal in proposals:
            event_index = batch_start_index + proposal.ordinal
            decision = decisions[proposal.ordinal]
            if decision == "action_failure":
                self.retry_router.observe_action_failure(proposal, event_index)
            disposition = self._blocking_disposition(scenario, proposal, decision)
            if disposition is not None:
                blocking.append((event_index, proposal, disposition))
        will_stop = (
            bool(blocking)
            and self.contract.continuation
            == ContinuationPolicy.STOP_ON_FIRST_BLOCKING_FAILURE
            and state.terminal is None
        )
        for index, proposal, disposition in blocking:
            self.blocking_audits.append(
                BlockingFailureAudit(
                    index,
                    proposal.actor_id,
                    disposition,
                    will_stop and index == blocking[0][0],
                )
            )
        if will_stop:
            state.terminal = "blocking_failure"

    def ledger(self) -> dict[str, int]:
        draws = sum(item.stream is not None for item in self.action_audits)
        return {
            "actionFailureDraws": draws,
            "actionFailureExposures": sum(
                item.transient_state_failed for item in self.action_audits
            ),
            "actionFailures": sum(item.applied for item in self.action_audits),
            "blockingFailures": len(self.blocking_audits),
            "continuationStops": sum(item.caused_stop for item in self.blocking_audits),
        }


@dataclass(frozen=True, slots=True)
class FaultRun:
    contract: FaultExecutionContract
    scheduler_run: SchedulerRun
    sensing_audit: tuple[SensingAudit, ...]
    action_failure_audit: tuple[ActionFailureAudit, ...]
    retry_audit: tuple[RetryAudit, ...]
    blocking_failure_audit: tuple[BlockingFailureAudit, ...]
    fault_ledger: dict[str, int]

    @property
    def result(self):
        return self.scheduler_run.result

    def extended_cost_ledger(self) -> dict[str, int]:
        return {
            **self.scheduler_run.architecture_run.extended_cost_ledger(),
            **self.fault_ledger,
        }

    def opportunity_validation(self) -> dict[str, bool]:
        checks = dict(self.scheduler_run.opportunity_validation())
        native = dict(self.result.summary["ledger"])
        fault = self.fault_ledger
        stream_counters = dict(self.result.final_state["streamCounters"])
        sensing_draws = fault["sensingValueDraws"] + fault["sensingStatusDraws"]
        action_draws = fault["actionFailureDraws"]
        retry_attempts = [
            item for item in self.retry_audit if item.disposition == "attempted"
        ]
        checks.update(ledger_identity(native))
        checks.update(
            {
                "legalPrimitivesPreserved": all(
                    event["proposal"]["kind"] in {"NoOp", "Swap", "MemoryUpdate"}
                    for event in self.result.events
                ),
                "sensingDrawIdentity": sensing_draws
                == sum(
                    stream_counters.get(stream, 0)
                    for stream in (SENSING_VALUE_STREAM, SENSING_STATUS_STREAM)
                ),
                "sensingErrorIdentity": fault["sensingErrorsApplied"]
                == fault["sensingValueErrors"] + fault["sensingStatusErrors"]
                == fault["sensingErrorHandlingOperations"],
                "actionFailureDrawIdentity": action_draws
                == sum(
                    stream_counters.get(stream, 0)
                    for stream in (BERNOULLI_FAILURE_STREAM, TRANSIENT_FAILURE_STREAM)
                ),
                "actionFailureClassification": fault["actionFailures"]
                == sum(item.applied for item in self.action_failure_audit),
                "retryAttemptsAreCharged": fault["retryAttempts"]
                <= native["activations"],
                "retryDelayRespected": all(
                    item.eligible_event_index is not None
                    and item.event_index >= item.eligible_event_index
                    for item in retry_attempts
                ),
                "continuationStopIdentity": fault["continuationStops"]
                == (self.result.summary["stopReason"] == "blocking_failure"),
                "oneFaultDrawTotalIdentity": fault["faultRngDraws"]
                == sensing_draws + action_draws,
            }
        )
        return checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "E02.fault-run.v1",
            "faultInterfaceVersion": FAULT_INTERFACE_VERSION,
            "contract": self.contract.to_dict(),
            "schedulerRun": self.scheduler_run.to_dict(),
            "sensingAudit": [item.to_dict() for item in self.sensing_audit],
            "actionFailureAudit": [
                item.to_dict() for item in self.action_failure_audit
            ],
            "retryAudit": [item.to_dict() for item in self.retry_audit],
            "blockingFailureAudit": [
                item.to_dict() for item in self.blocking_failure_audit
            ],
            "faultLedger": dict(self.fault_ledger),
            "extendedCostLedger": self.extended_cost_ledger(),
            "opportunityValidation": self.opportunity_validation(),
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def run_faulted_architecture(
    scenario: Scenario,
    architecture_contract: ArchitectureExecutionContract,
    scheduler_contract: SchedulerExecutionContract,
    fault_contract: FaultExecutionContract,
    *,
    trace_mode: Literal["full", "digest", "none"] = "digest",
    action_interface: CommonActionInterface | None = None,
    after_batch_observer: Any | None = None,
) -> FaultRun:
    fault_contract.validate(scenario, architecture_contract, scheduler_contract)
    controller = FrozenSchedulerController(
        scheduler_contract,
        seed=scenario.seed,
        scenario_id=scenario.scenario_id,
        actor_ids=tuple(cell.cell_id for cell in scenario.cells),
    )
    sensing = FrozenSensingTransformer(fault_contract, scenario)
    if action_interface is not None and fault_contract.sensing != SensingProfile.EXACT:
        raise ValueError(
            "custom action interfaces cannot bypass the frozen sensing transformer"
        )
    interface = action_interface or CommonActionInterface(
        None if fault_contract.sensing == SensingProfile.EXACT else sensing
    )
    architecture_router = ArchitectureProposalRouter(
        architecture_contract, action_interface=interface
    )
    proposal_router = FaultProposalRouter(fault_contract, architecture_router, sensing)
    interceptor = FaultExecutionInterceptor(fault_contract, scenario, proposal_router)

    class _ObservedInterceptor:
        def prepare(self, proposal, event_index):
            return interceptor.prepare(proposal, event_index)

        def outcome(self, proposal, validation, event_index):
            return interceptor.outcome(proposal, validation, event_index)

        def after_batch(
            self, active_scenario, state, proposals, decisions, batch_start_index
        ):
            interceptor.after_batch(
                active_scenario, state, proposals, decisions, batch_start_index
            )
            if after_batch_observer is not None:
                after_batch_observer.after_batch(
                    active_scenario, state, proposals, decisions, batch_start_index
                )

    active_interceptor = (
        interceptor if after_batch_observer is None else _ObservedInterceptor()
    )

    def _adapter_terminal(active_scenario, state):
        # The fault interceptor may set a native competing terminal during its
        # after-batch commit.  Adapter quiescence must never overwrite it.
        if state.terminal == "blocking_failure":
            return "blocking_failure"
        return after_batch_observer.evaluate_terminal(active_scenario, state)

    result = run(
        scenario,
        trace_mode=trace_mode,
        proposal_factory=proposal_router.proposal_for,
        schedule_factory=controller,
        execution_interceptor=active_interceptor,
        terminal_evaluator=(
            None if after_batch_observer is None else _adapter_terminal
        ),
    )
    architecture_run = ArchitectureRun(
        architecture_contract,
        result,
        tuple(architecture_router.audits),
        architecture_router.architecture_ledger(),
    )
    scheduler_run = SchedulerRun(
        scheduler_contract, architecture_run, tuple(controller.audits)
    )
    fault_ledger = {
        **sensing.ledger(),
        **interceptor.ledger(),
        **proposal_router.ledger(),
    }
    fault_ledger["faultRngDraws"] = (
        fault_ledger["sensingValueDraws"]
        + fault_ledger["sensingStatusDraws"]
        + fault_ledger["actionFailureDraws"]
    )
    return FaultRun(
        fault_contract,
        scheduler_run,
        tuple(sensing.audits),
        tuple(interceptor.action_audits),
        tuple(proposal_router.audits),
        tuple(interceptor.blocking_audits),
        fault_ledger,
    )


def exact_replay_fault(run_result: FaultRun) -> FaultRun:
    replayed = run_faulted_architecture(
        run_result.result.scenario,
        run_result.scheduler_run.architecture_run.contract,
        run_result.scheduler_run.contract,
        run_result.contract,
        trace_mode=run_result.result.summary["traceMode"],
    )
    if replayed.to_json_bytes() != run_result.to_json_bytes():
        raise AssertionError("fault-run replay mismatch")
    return replayed


def comparability_classification(
    architecture: ControlArchitecture,
    contract: FaultExecutionContract,
) -> tuple[str, str]:
    if architecture == ControlArchitecture.CENTRAL_GLOBAL_LEGACY:
        return (
            "non_comparable",
            "full-global legacy control has unmatched information, scheduling, and opportunity semantics",
        )
    if contract.degenerate_retry_combination:
        return (
            "comparable_degenerate",
            "first action failure stops the run, so queued retries cannot activate",
        )
    return "comparable", "matched S02-S04 contracts preserved"
