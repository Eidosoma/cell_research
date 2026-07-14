"""S09 complete, loss-explicit cost accounting over frozen S02--S08 runs.

This module is deliberately a read-only projection over :class:`FaultRun`.
It does not participate in proposal construction, scheduling, validation,
commit, retry, RNG addressing, or pairing.  Timing wraps an unchanged run and
is excluded from deterministic replay and every algorithmic cost projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

from reference_simulator.model import canonical_json_bytes

from .architectures import CoordinatorProfile
from .faults import FaultRun, exact_replay_fault, run_faulted_architecture
from .architectures import ArchitectureExecutionContract
from .faults import FaultExecutionContract
from .schedulers import SchedulerExecutionContract
from reference_simulator.model import Scenario


COST_SCHEMA_VERSION = "E02.complete-cost-ledger.v1"
COST_SCHEMA_SHA256 = "cd065da9294ee4ca0838ab584f89fe43572336dcde9aa034d3efb704e7bc2104"

S01_ADDITIVE_FIELDS = (
    "activations",
    "observationRecordReads",
    "valueComparisons",
    "targetCalculations",
    "proposals",
    "noOps",
    "rejections",
    "memoryUpdates",
    "acceptedSwaps",
    "displacedCells",
    "conflictLosses",
    "coordinatorMessages",
)

ALIAS_FIELDS = (
    "valueReads",
    "statusReads",
    "policyCandidateConstructions",
    "deferredRetryEnvelopeReuses",
    "failedProposals",
    "coordinatorEligibleDecisions",
    "coordinatorInterventions",
)

SCHEDULER_OVERHEAD_FIELDS = (
    "schedulerActorSelectionRngBlocks",
    "schedulerIdentitiesScored",
    "schedulerCandidateInspections",
)


def validate_cost_schema(path: str | Path) -> dict[str, Any]:
    """Require the frozen schema bytes before accounting a run."""

    selected = Path(path)
    payload = selected.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    return {
        "path": str(selected),
        "expectedSha256": COST_SCHEMA_SHA256,
        "observedSha256": observed,
        "byteCount": len(payload),
        "success": observed == COST_SCHEMA_SHA256,
    }


def _nonnegative_ints(values: dict[str, int | float]) -> bool:
    return all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for name, value in values.items()
        if name != "wallTimeSeconds"
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True, slots=True)
class CompleteCostLedger:
    """Canonical vector ledger plus named projections and validation checks."""

    scenario_id: str
    n: int
    architecture: str
    coordinator_profile: str
    scheduler: str
    fault_profile: dict[str, str]
    costs: dict[str, int | float]
    projections: dict[str, int]
    normalizations: dict[str, float | None]
    validation: dict[str, bool]

    @property
    def success(self) -> bool:
        return all(self.validation.values())

    def to_dict(self, *, include_wall_time: bool = True) -> dict[str, Any]:
        costs = dict(self.costs)
        if not include_wall_time:
            costs.pop("wallTimeSeconds", None)
        return {
            "schemaVersion": COST_SCHEMA_VERSION,
            "schemaSha256": COST_SCHEMA_SHA256,
            "scenarioId": self.scenario_id,
            "n": self.n,
            "architecture": self.architecture,
            "coordinatorProfile": self.coordinator_profile,
            "scheduler": self.scheduler,
            "faultProfile": dict(self.fault_profile),
            "costs": costs,
            "projections": dict(self.projections),
            "normalizations": dict(self.normalizations),
            "validation": dict(self.validation),
            "success": self.success,
        }

    def deterministic_bytes(self) -> bytes:
        """Stable cost serialization with host wall time deliberately removed."""

        return canonical_json_bytes(self.to_dict(include_wall_time=False))

    def to_flat_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schemaVersion": COST_SCHEMA_VERSION,
            "schemaSha256": COST_SCHEMA_SHA256,
            "scenarioId": self.scenario_id,
            "n": self.n,
            "architecture": self.architecture,
            "coordinatorProfile": self.coordinator_profile,
            "scheduler": self.scheduler,
            **{f"fault_{key}": value for key, value in self.fault_profile.items()},
            **self.costs,
            **self.projections,
            **self.normalizations,
            "validationPass": self.success,
        }
        return row


def build_complete_cost_ledger(
    run: FaultRun,
    *,
    wall_time_seconds: float = 0.0,
) -> CompleteCostLedger:
    """Project a completed frozen-contract run into the S09 ledger.

    Fresh candidate construction is inferred exactly from the S05 retry audit:
    a natural retry activation submits a proposal but reuses an envelope, so
    ``targetCalculations = proposals - retryAttempts``.
    """

    if not math.isfinite(wall_time_seconds) or wall_time_seconds < 0:
        raise ValueError("wall_time_seconds must be finite and nonnegative")
    native = dict(run.result.summary["ledger"])
    architecture_run = run.scheduler_run.architecture_run
    architecture_ledger = dict(architecture_run.architecture_ledger or {})
    fault = dict(run.fault_ledger)
    scheduler_audit = run.scheduler_run.scheduler_audit
    retry_attempts = int(fault["retryAttempts"])
    target_calculations = int(native["proposals"]) - retry_attempts
    if target_calculations < 0:
        raise ValueError("retry attempts cannot exceed proposals")
    active_weak = (
        architecture_run.contract.coordinator_profile
        == CoordinatorProfile.WEAK_FROZEN_BUDGET
    )
    eligible = int(architecture_ledger["coordinatorEligibleDecisions"])

    costs: dict[str, int | float] = {
        "activations": int(native["activations"]),
        "observationRecordReads": int(native["observationReads"]),
        "valueReads": int(native["observationReads"]),
        "statusReads": int(native["observationReads"]),
        "valueComparisons": int(native["valueComparisons"]),
        "targetCalculations": target_calculations,
        "policyCandidateConstructions": target_calculations,
        "deferredRetryEnvelopeReuses": retry_attempts,
        "proposals": int(native["proposals"]),
        "failedProposals": int(native["rejections"] + native["conflictLosses"]),
        "noOps": int(native["noOps"]),
        "rejections": int(native["rejections"]),
        "memoryUpdates": int(native["memoryUpdates"]),
        "acceptedSwaps": int(native["acceptedSwaps"]),
        "displacedCells": int(native["displacedCells"]),
        "conflictLosses": int(native["conflictLosses"]),
        "coordinatorMessages": int(architecture_ledger["coordinatorMessages"]),
        "coordinatorMessageBits": int(architecture_ledger["coordinatorMessageBits"]),
        "coordinatorClockChecks": int(native["activations"]) if active_weak else 0,
        "coordinatorCandidateEvaluations": eligible,
        "coordinatorEligibleDecisions": eligible,
        "coordinatorInterventions": int(architecture_ledger["coordinatorInterventions"]),
        "sensingValueDraws": int(fault["sensingValueDraws"]),
        "sensingStatusDraws": int(fault["sensingStatusDraws"]),
        "sensingValueErrors": int(fault["sensingValueErrors"]),
        "sensingStatusErrors": int(fault["sensingStatusErrors"]),
        "sensingErrorsApplied": int(fault["sensingErrorsApplied"]),
        "sensingErrorHandlingOperations": int(fault["sensingErrorHandlingOperations"]),
        "actionFailureDraws": int(fault["actionFailureDraws"]),
        "actionFailureExposures": int(fault["actionFailureExposures"]),
        "actionFailures": int(fault["actionFailures"]),
        "blockingFailures": int(fault["blockingFailures"]),
        "continuationStops": int(fault["continuationStops"]),
        "retryQueued": int(fault["retryQueued"]),
        "retryAttempts": retry_attempts,
        "retryExhausted": int(fault["retryExhausted"]),
        "retryQueueCollisions": int(fault["retryQueueCollisions"]),
        "retryPendingAtStop": int(fault["retryPendingAtStop"]),
        "faultRngDraws": int(fault["faultRngDraws"]),
        "schedulerActorSelectionRngBlocks": sum(
            item.actor_selection_stream_blocks for item in scheduler_audit
        ),
        "schedulerIdentitiesScored": sum(item.identities_scored for item in scheduler_audit),
        "schedulerCandidateInspections": sum(
            item.proposal_candidates_inspected for item in scheduler_audit
        ),
        "wallTimeSeconds": float(wall_time_seconds),
    }

    s01 = sum(int(costs[name]) for name in S01_ADDITIVE_FIELDS)
    controller = (
        s01
        + int(costs["coordinatorClockChecks"])
        + int(costs["coordinatorCandidateEvaluations"])
    )
    mechanism = (
        controller
        + int(costs["sensingErrorHandlingOperations"])
        + int(costs["faultRngDraws"])
    )
    projections = {
        "s01UnitWeightFullCost": s01,
        "controllerExpandedSensitivity": controller,
        "mechanismExpandedSensitivity": mechanism,
        "zeroCostControl": 0,
    }
    activations = int(costs["activations"])
    normalizations = {
        "s01UnitWeightPerOpportunity": _ratio(s01, activations),
        "s01UnitWeightPerCell": s01 / len(run.result.scenario.cells),
        "s01UnitWeightPerN2": s01 / (len(run.result.scenario.cells) ** 2),
        "failedProposalsPerOpportunity": _ratio(
            int(costs["failedProposals"]), activations
        ),
        "displacedCellsPerOpportunity": _ratio(
            int(costs["displacedCells"]), activations
        ),
        "coordinatorMessagesPerOpportunity": _ratio(
            int(costs["coordinatorMessages"]), activations
        ),
    }

    opportunity_checks = run.opportunity_validation()
    contract = run.contract
    scheduler_contract = run.scheduler_run.contract
    architecture_contract = architecture_run.contract
    validation = {
        "inheritedOpportunityValidation": all(opportunity_checks.values()),
        "integerCountUnits": _nonnegative_ints(costs),
        "wallTimeUnit": isinstance(costs["wallTimeSeconds"], float)
        and math.isfinite(costs["wallTimeSeconds"])
        and costs["wallTimeSeconds"] >= 0,
        "activationProposalIdentity": costs["activations"] == costs["proposals"],
        "proposalPartitionIdentity": costs["proposals"]
        == costs["noOps"]
        + costs["rejections"]
        + costs["memoryUpdates"]
        + costs["acceptedSwaps"]
        + costs["conflictLosses"],
        "swapDisplacementIdentity": costs["displacedCells"]
        == 2 * costs["acceptedSwaps"],
        "readFieldAliasIdentity": costs["valueReads"]
        == costs["statusReads"]
        == costs["observationRecordReads"],
        "candidateConstructionIdentity": costs["targetCalculations"]
        == costs["policyCandidateConstructions"]
        == costs["proposals"] - costs["retryAttempts"],
        "deferredRetryIdentity": costs["deferredRetryEnvelopeReuses"]
        == costs["retryAttempts"],
        "failedProposalIdentity": costs["failedProposals"]
        == costs["rejections"] + costs["conflictLosses"],
        "actionFailureSubsetIdentity": costs["actionFailures"] <= costs["rejections"],
        "coordinatorMessageIdentity": costs["coordinatorMessages"]
        == 2 * costs["coordinatorEligibleDecisions"],
        "coordinatorCandidateIdentity": costs["coordinatorCandidateEvaluations"]
        == costs["coordinatorEligibleDecisions"],
        "coordinatorClockIdentity": costs["coordinatorClockChecks"]
        == (costs["activations"] if active_weak else 0),
        "schedulerCandidateBoundary": costs["schedulerCandidateInspections"] == 0,
        "s01ProjectionIdentity": projections["s01UnitWeightFullCost"]
        == sum(int(costs[name]) for name in S01_ADDITIVE_FIELDS),
        "aliasFieldsExcludedFromS01": not set(ALIAS_FIELDS).intersection(
            S01_ADDITIVE_FIELDS
        ),
        "schedulerOverheadExcludedFromS01": not set(
            SCHEDULER_OVERHEAD_FIELDS
        ).intersection(S01_ADDITIVE_FIELDS),
        "wallTimeExcludedFromAlgorithmicCost": "wallTimeSeconds"
        not in S01_ADDITIVE_FIELDS,
        "zeroCostProjectionIdentity": projections["zeroCostControl"] == 0,
        "legalPrimitivesPreserved": contract.legal_primitives
        == scheduler_contract.legal_primitives
        == ("NoOp", "Swap", "MemoryUpdate"),
        "informationPermissionPreserved": contract.information_permission
        == scheduler_contract.policy_information_permission
        == architecture_contract.information_permission.value
        == "policy_native_local",
        "opportunityContractPreserved": contract.proposal_candidates_per_opportunity
        == scheduler_contract.proposal_candidates_per_opportunity
        == architecture_contract.proposal_candidates_per_opportunity
        == 1,
    }
    return CompleteCostLedger(
        scenario_id=run.result.scenario.scenario_id,
        n=len(run.result.scenario.cells),
        architecture=architecture_contract.architecture.value,
        coordinator_profile=architecture_contract.coordinator_profile.value,
        scheduler=scheduler_contract.family.value,
        fault_profile={
            "mobility": contract.mobility.value,
            "continuation": contract.continuation.value,
            "retry": contract.retry.value,
            "actionFailure": contract.action_failure.value,
            "sensing": contract.sensing.value,
        },
        costs=costs,
        projections=projections,
        normalizations=normalizations,
        validation=validation,
    )


@dataclass(frozen=True, slots=True)
class CostedFaultRun:
    """An unchanged S05 run paired with its S09 observational projection."""

    fault_run: FaultRun
    ledger: CompleteCostLedger

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "E02.costed-fault-run.v1",
            "faultRun": self.fault_run.to_dict(),
            "completeCostLedger": self.ledger.to_dict(),
        }


def run_costed_faulted_architecture(
    scenario: Scenario,
    architecture_contract: ArchitectureExecutionContract,
    scheduler_contract: SchedulerExecutionContract,
    fault_contract: FaultExecutionContract,
    *,
    trace_mode: Literal["full", "digest", "none"] = "digest",
) -> CostedFaultRun:
    """Time the unchanged S05 executor and attach a separate S09 ledger."""

    start = perf_counter_ns()
    fault_run = run_faulted_architecture(
        scenario,
        architecture_contract,
        scheduler_contract,
        fault_contract,
        trace_mode=trace_mode,
    )
    elapsed = (perf_counter_ns() - start) / 1_000_000_000
    return CostedFaultRun(
        fault_run,
        build_complete_cost_ledger(fault_run, wall_time_seconds=elapsed),
    )


def exact_replay_costed(run: CostedFaultRun) -> CostedFaultRun:
    """Require exact S05 bytes and deterministic cost bytes; ignore host timing."""

    replayed_fault = exact_replay_fault(run.fault_run)
    replayed_ledger = build_complete_cost_ledger(replayed_fault, wall_time_seconds=0.0)
    if replayed_ledger.deterministic_bytes() != run.ledger.deterministic_bytes():
        raise AssertionError("complete-cost replay mismatch")
    return CostedFaultRun(replayed_fault, replayed_ledger)
