"""S04 scheduler interventions over the frozen S03 action boundary.

Schedulers select immutable actor identities.  They never receive the dynamic
scenario state and never construct or inspect action proposals.  The common
engine then charges exactly one activation and one policy proposal for every
selected identity, including synchronous conflict losers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from reference_simulator.engine import run
from reference_simulator.model import RunResult, Scenario, canonical_json_bytes
from reference_simulator.rng import bounded, u64
from reference_simulator.scheduler import RandomDraw, ScheduledOpportunity
from reference_simulator.transition_primitives import ledger_identity

from .architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
    ArchitectureRun,
    ControlArchitecture,
)


SCHEDULER_INTERFACE_VERSION = "E02-scheduler-families-v1"
SCHEDULER_PRESPECIFICATION_SHA256 = (
    "2fb3e94908c5fa40f529abe1a3a21ad3e5d8d8390105dcb958102aae6a81720e"
)
PERMUTATION_STREAM = "scheduler_permutation_s04_v1"


class SchedulerFamily(str, Enum):
    DETERMINISTIC_SCAN = "deterministic_scan"
    UNIFORM_RANDOM_ACTIVATION = "uniform_random_activation"
    RANDOM_PERMUTATION_SWEEP = "random_permutation_sweep"
    SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT = (
        "synchronous_batch_deterministic_conflict"
    )
    FAIR_ADVERSARIAL = "fair_adversarial"


@dataclass(frozen=True, slots=True)
class SchedulerExecutionContract:
    family: SchedulerFamily
    opportunity_unit: str = "one_actor_one_policy_proposal"
    proposal_candidates_per_opportunity: int = 1
    retry_policy: str = "no_retry"
    policy_information_permission: str = "policy_native_local"
    legal_primitives: tuple[str, ...] = ("NoOp", "Swap", "MemoryUpdate")
    fairness_window_multiplier: int = 2
    adversarial_lookahead_n_multiplier: int = 1

    def validate(
        self,
        scenario: Scenario,
        architecture_contract: ArchitectureExecutionContract,
    ) -> None:
        if not architecture_contract.matched_contrast_eligible:
            raise ValueError("legacy global control is unsupported in matched S04 scheduling")
        architecture_contract.validate_policy_action_boundary(scenario)
        if self.opportunity_unit != "one_actor_one_policy_proposal":
            raise ValueError("S04 opportunity unit is frozen")
        if self.proposal_candidates_per_opportunity != 1:
            raise ValueError("S04 forbids free policy proposal candidates")
        if self.retry_policy != "no_retry":
            raise ValueError("S04 forbids same-opportunity retries")
        if self.policy_information_permission != "policy_native_local":
            raise ValueError("S04 cannot change policy information permission")
        if self.legal_primitives != ("NoOp", "Swap", "MemoryUpdate"):
            raise ValueError("S04 legal primitive set is frozen by S02/S03")
        if self.fairness_window_multiplier != 2:
            raise ValueError("S04 fair-adversarial window is frozen at W=2n")
        if self.adversarial_lookahead_n_multiplier != 1:
            raise ValueError("S04 adversarial lookahead is frozen at L=n")

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "opportunityUnit": self.opportunity_unit,
            "proposalCandidatesPerOpportunity": self.proposal_candidates_per_opportunity,
            "retryPolicy": self.retry_policy,
            "policyInformationPermission": self.policy_information_permission,
            "legalPrimitives": list(self.legal_primitives),
            "fairnessWindowMultiplier": self.fairness_window_multiplier,
            "adversarialLookaheadNMultiplier": self.adversarial_lookahead_n_multiplier,
            "prespecificationSha256": SCHEDULER_PRESPECIFICATION_SHA256,
        }


@dataclass(frozen=True, slots=True)
class SchedulerAudit:
    event_index: int
    batch_start_index: int
    batch_ordinal: int
    batch_width: int
    actor_id: str
    selection_reason: str
    actor_selection_stream_blocks: int
    identities_scored: int
    proposal_candidates_inspected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventIndex": self.event_index,
            "batchStartIndex": self.batch_start_index,
            "batchOrdinal": self.batch_ordinal,
            "batchWidth": self.batch_width,
            "actorId": self.actor_id,
            "selectionReason": self.selection_reason,
            "actorSelectionStreamBlocks": self.actor_selection_stream_blocks,
            "identitiesScored": self.identities_scored,
            "proposalCandidatesInspected": self.proposal_candidates_inspected,
        }


class FrozenSchedulerController:
    """State-independent scheduler surface plus scheduler-owned history only."""

    __slots__ = (
        "contract",
        "seed",
        "scenario_id",
        "actor_ids",
        "audits",
        "last_activation",
        "cached_sweep_index",
        "cached_permutation",
        "cached_permutation_draws",
    )

    def __init__(
        self,
        contract: SchedulerExecutionContract,
        *,
        seed: int,
        scenario_id: str,
        actor_ids: tuple[str, ...],
    ) -> None:
        if not actor_ids or tuple(sorted(actor_ids)) != actor_ids:
            raise ValueError("scheduler actor identities must be nonempty and canonical")
        if len(set(actor_ids)) != len(actor_ids):
            raise ValueError("scheduler actor identities must be unique")
        self.contract = contract
        self.seed = seed
        self.scenario_id = scenario_id
        self.actor_ids = actor_ids
        self.audits: list[SchedulerAudit] = []
        self.last_activation = {actor_id: -1 for actor_id in actor_ids}
        self.cached_sweep_index = -1
        self.cached_permutation: tuple[str, ...] = ()
        self.cached_permutation_draws: tuple[RandomDraw, ...] = ()

    def __call__(
        self,
        event_index: int,
        remaining_opportunities: int,
    ) -> tuple[ScheduledOpportunity, ...]:
        if event_index != len(self.audits):
            raise ValueError("scheduler opportunities must be requested sequentially")
        if remaining_opportunities < 1:
            return ()
        if self.contract.family == SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT:
            return self._synchronous(event_index, remaining_opportunities)
        return (self._serial(event_index),)

    def _serial(self, event_index: int) -> ScheduledOpportunity:
        family = self.contract.family
        n = len(self.actor_ids)
        draws: tuple[RandomDraw, ...] = ()
        consumption: tuple[tuple[str, int], ...] = ()
        blocks = 0
        identities_scored = 1
        if family == SchedulerFamily.DETERMINISTIC_SCAN:
            actor_id = self.actor_ids[event_index % n]
            reason = "canonical_scan"
        elif family == SchedulerFamily.UNIFORM_RANDOM_ACTIVATION:
            selected, blocks = bounded(
                self.seed,
                self.scenario_id,
                "actor_activation",
                event_index,
                n,
            )
            actor_id = self.actor_ids[selected]
            draws = tuple(
                (
                    "actor_activation",
                    event_index,
                    draw_index,
                    u64(
                        self.seed,
                        self.scenario_id,
                        "actor_activation",
                        event_index,
                        draw_index,
                    ),
                )
                for draw_index in range(blocks)
            )
            consumption = (("actor_activation", blocks),)
            reason = "uniform_with_replacement"
        elif family == SchedulerFamily.RANDOM_PERMUTATION_SWEEP:
            sweep = event_index // n
            offset = event_index % n
            self._ensure_permutation(sweep)
            actor_id = self.cached_permutation[offset]
            if offset == 0:
                draws = self.cached_permutation_draws
                blocks = len(draws)
                consumption = ((PERMUTATION_STREAM, blocks),) if blocks else ()
            reason = "counter_addressed_permutation"
        elif family == SchedulerFamily.FAIR_ADVERSARIAL:
            actor_id, reason, identities_scored = self._adversarial_actor(event_index)
        else:
            raise AssertionError(f"unhandled serial scheduler {family}")
        self.last_activation[actor_id] = event_index
        self.audits.append(
            SchedulerAudit(
                event_index=event_index,
                batch_start_index=event_index,
                batch_ordinal=0,
                batch_width=1,
                actor_id=actor_id,
                selection_reason=reason,
                actor_selection_stream_blocks=blocks,
                identities_scored=identities_scored,
            )
        )
        return ScheduledOpportunity(actor_id, draws, consumption)

    def _ensure_permutation(self, sweep_index: int) -> None:
        if self.cached_sweep_index == sweep_index:
            return
        if sweep_index != self.cached_sweep_index + 1:
            raise ValueError("permutation sweeps must be requested sequentially")
        result = list(self.actor_ids)
        draws: list[RandomDraw] = []
        draw_cursor = 0
        for index in range(len(result) - 1, 0, -1):
            selected, consumed = bounded(
                self.seed,
                self.scenario_id,
                PERMUTATION_STREAM,
                sweep_index,
                index + 1,
                draw_cursor,
            )
            for draw_index in range(draw_cursor, draw_cursor + consumed):
                draws.append(
                    (
                        PERMUTATION_STREAM,
                        sweep_index,
                        draw_index,
                        u64(
                            self.seed,
                            self.scenario_id,
                            PERMUTATION_STREAM,
                            sweep_index,
                            draw_index,
                        ),
                    )
                )
            draw_cursor += consumed
            result[index], result[selected] = result[selected], result[index]
        self.cached_sweep_index = sweep_index
        self.cached_permutation = tuple(result)
        self.cached_permutation_draws = tuple(draws)

    def _deadline_continuation_is_feasible(
        self,
        actor_id: str,
        event_index: int,
    ) -> bool:
        n = len(self.actor_ids)
        window = self.contract.fairness_window_multiplier * n
        lookahead = self.contract.adversarial_lookahead_n_multiplier * n
        simulated = dict(self.last_activation)
        simulated[actor_id] = event_index
        for future_index in range(event_index + 1, event_index + lookahead + 1):
            if any(future_index - last > window for last in simulated.values()):
                return False
            earliest = min(self.actor_ids, key=lambda item: (simulated[item], item))
            if future_index - simulated[earliest] >= window:
                simulated[earliest] = future_index
        return True

    def _adversarial_actor(self, event_index: int) -> tuple[str, str, int]:
        n = len(self.actor_ids)
        if event_index < n:
            return self.actor_ids[event_index], "canonical_fairness_warmup", 1
        window = self.contract.fairness_window_multiplier * n
        due = [
            actor_id
            for actor_id in self.actor_ids
            if event_index - self.last_activation[actor_id] >= window
        ]
        feasible = [
            actor_id
            for actor_id in self.actor_ids
            if self._deadline_continuation_is_feasible(actor_id, event_index)
        ]
        if not feasible:
            raise RuntimeError("fair-adversarial deadline contract is infeasible")
        if due:
            actor_id = min(due, key=lambda item: (self.last_activation[item], item))
            if actor_id not in feasible:
                raise RuntimeError("due actor failed bounded-lookahead feasibility")
            return actor_id, "fairness_deadline_override", n
        # Repeating the most recently activated feasible identity maximizes
        # short-horizon clustering without observing state or proposals.
        actor_id = min(feasible, key=lambda item: (-self.last_activation[item], item))
        return actor_id, "bounded_lookahead_max_clustering", n

    def _synchronous(
        self,
        event_index: int,
        remaining_opportunities: int,
    ) -> tuple[ScheduledOpportunity, ...]:
        width = min(len(self.actor_ids), remaining_opportunities)
        selected = self.actor_ids[:width]
        result: list[ScheduledOpportunity] = []
        for ordinal, actor_id in enumerate(selected):
            slot_index = event_index + ordinal
            self.last_activation[actor_id] = slot_index
            self.audits.append(
                SchedulerAudit(
                    event_index=slot_index,
                    batch_start_index=event_index,
                    batch_ordinal=ordinal,
                    batch_width=width,
                    actor_id=actor_id,
                    selection_reason=(
                        "canonical_full_synchronous_batch"
                        if width == len(self.actor_ids)
                        else "canonical_budget_truncated_batch"
                    ),
                    actor_selection_stream_blocks=0,
                    identities_scored=1,
                )
            )
            result.append(ScheduledOpportunity(actor_id))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class SchedulerRun:
    contract: SchedulerExecutionContract
    architecture_run: ArchitectureRun
    scheduler_audit: tuple[SchedulerAudit, ...]

    @property
    def result(self) -> RunResult:
        return self.architecture_run.result

    def opportunity_validation(self) -> dict[str, bool]:
        ledger = dict(self.result.summary["ledger"])
        checks = dict(ledger_identity(ledger))
        checks.update(self.architecture_run.opportunity_validation())
        count = int(self.result.summary["activationCount"])
        checks.update(
            {
                "auditCountMatchesActivations": len(self.scheduler_audit) == count,
                "auditIndicesContiguous": [item.event_index for item in self.scheduler_audit]
                == list(range(count)),
                "zeroPolicyProposalCandidatesInspected": all(
                    item.proposal_candidates_inspected == 0 for item in self.scheduler_audit
                ),
                "onePolicyProposalPerActivation": ledger["proposals"] == count,
                "knownActorsOnly": all(
                    item.actor_id in self.result.scenario.cell_map
                    for item in self.scheduler_audit
                ),
            }
        )
        if self.result.events:
            checks["auditActorsMatchTrace"] = [
                item.actor_id for item in self.scheduler_audit
            ] == [str(event["actorId"]) for event in self.result.events]
            checks["auditBatchesMatchTrace"] = all(
                item.batch_width == int(event["batchWidth"])
                and item.batch_ordinal == int(event["batchOrdinal"])
                for item, event in zip(self.scheduler_audit, self.result.events)
            )
        return checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "E02.scheduler-run.v1",
            "schedulerInterfaceVersion": SCHEDULER_INTERFACE_VERSION,
            "contract": self.contract.to_dict(),
            "architectureRun": self.architecture_run.to_dict(),
            "schedulerAudit": [item.to_dict() for item in self.scheduler_audit],
            "opportunityValidation": self.opportunity_validation(),
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def run_scheduled_architecture(
    scenario: Scenario,
    architecture_contract: ArchitectureExecutionContract,
    scheduler_contract: SchedulerExecutionContract,
    *,
    trace_mode: Literal["full", "digest", "none"] = "digest",
) -> SchedulerRun:
    scheduler_contract.validate(scenario, architecture_contract)
    if architecture_contract.architecture == ControlArchitecture.CENTRAL_GLOBAL_LEGACY:
        raise ValueError("legacy global control is structurally unsupported in S04")
    controller = FrozenSchedulerController(
        scheduler_contract,
        seed=scenario.seed,
        scenario_id=scenario.scenario_id,
        actor_ids=tuple(cell.cell_id for cell in scenario.cells),
    )
    router = ArchitectureProposalRouter(architecture_contract)
    result = run(
        scenario,
        trace_mode=trace_mode,
        proposal_factory=router.proposal_for,
        schedule_factory=controller,
    )
    architecture_run = ArchitectureRun(
        architecture_contract,
        result,
        tuple(router.audits),
        router.architecture_ledger(),
    )
    return SchedulerRun(scheduler_contract, architecture_run, tuple(controller.audits))


def exact_replay_scheduler(run_result: SchedulerRun) -> SchedulerRun:
    replayed = run_scheduled_architecture(
        run_result.result.scenario,
        run_result.architecture_run.contract,
        run_result.contract,
        trace_mode=run_result.result.summary["traceMode"],
    )
    if replayed.to_json_bytes() != run_result.to_json_bytes():
        raise AssertionError("scheduler replay mismatch")
    return replayed


def scheduler_controller_fields() -> tuple[str, ...]:
    """Stable source-level audit of all controller-held fields."""

    return tuple(FrozenSchedulerController.__slots__)
