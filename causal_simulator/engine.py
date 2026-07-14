"""Matched execution contracts layered over the E01 transition kernel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Literal, Mapping

from reference_simulator.engine import run
from reference_simulator.model import Architecture, FaultMode, RunResult, Scenario, canonical_json_bytes

from .action_interface import CommonActionInterface, ControlTopology, InformationPermission


class ContinuationPolicy(str, Enum):
    SKIP_AND_CONTINUE = "skip_and_continue"


class RetryPolicy(str, Enum):
    NO_RETRY = "no_retry"


class ActionFailure(str, Enum):
    NONE = "none"


@dataclass(frozen=True, slots=True)
class MatchedExecutionContract:
    topology: ControlTopology
    information_permission: InformationPermission = InformationPermission.POLICY_NATIVE_LOCAL
    continuation_policy: ContinuationPolicy = ContinuationPolicy.SKIP_AND_CONTINUE
    retry_policy: RetryPolicy = RetryPolicy.NO_RETRY
    action_failure: ActionFailure = ActionFailure.NONE
    sensing_error: str = "exact"
    coordinator_budget: str = "common_validator_only"
    scheduler: str = "uniform_random_activation"
    decision_rule: str = "policy_native_rules_v1"

    def validate(self, scenario: Scenario) -> None:
        if scenario.architecture != Architecture.CELL_VIEW:
            raise ValueError("matched action interface excludes central_global_legacy")
        if scenario.scheduler != "serial_counter_addressed" or scenario.batch_width != 1:
            raise ValueError("S02 contract requires one uniform random actor per opportunity")
        if self.information_permission != InformationPermission.POLICY_NATIVE_LOCAL:
            raise ValueError("S02 implements only policy_native_local information")
        if self.continuation_policy != ContinuationPolicy.SKIP_AND_CONTINUE:
            raise ValueError("unsupported continuation policy")
        if self.retry_policy != RetryPolicy.NO_RETRY:
            raise ValueError("unsupported retry policy")
        if self.action_failure != ActionFailure.NONE:
            raise ValueError("unsupported action failure profile")
        if self.sensing_error != "exact":
            raise ValueError("unsupported sensing profile")
        if self.coordinator_budget != "common_validator_only":
            raise ValueError("unsupported coordinator budget")
        if self.scheduler != "uniform_random_activation":
            raise ValueError("unsupported scheduler intervention")
        if self.decision_rule != "policy_native_rules_v1":
            raise ValueError("unsupported decision rule")

    def to_dict(self) -> dict[str, str]:
        return {
            "topology": self.topology.value,
            "informationPermission": self.information_permission.value,
            "continuationPolicy": self.continuation_policy.value,
            "retryPolicy": self.retry_policy.value,
            "actionFailure": self.action_failure.value,
            "sensingError": self.sensing_error,
            "coordinatorBudget": self.coordinator_budget,
            "scheduler": self.scheduler,
            "decisionRule": self.decision_rule,
        }


@dataclass(frozen=True, slots=True)
class TopologyRun:
    contract: MatchedExecutionContract
    result: RunResult

    def transition_projection(self) -> dict[str, Any]:
        return {
            "scenarioId": self.result.scenario.scenario_id,
            "initialStateHash": self.result.initial_state_hash,
            "finalStateHash": self.result.final_state_hash,
            "finalState": dict(self.result.final_state),
            "stopReason": self.result.summary["stopReason"],
            "eventDigest": self.result.event_digest,
            "events": list(self.result.events),
            "ledger": dict(self.result.summary["ledger"]),
        }

    def transition_bytes(self) -> bytes:
        return canonical_json_bytes(self.transition_projection())


def run_with_contract(
    scenario: Scenario,
    contract: MatchedExecutionContract,
    *,
    trace_mode: Literal["full", "digest", "none"] = "digest",
) -> TopologyRun:
    contract.validate(scenario)
    interface = CommonActionInterface()
    factory = partial(interface.proposal_for, contract.topology)
    result = run(scenario, trace_mode=trace_mode, proposal_factory=factory)
    return TopologyRun(contract, result)


@dataclass(frozen=True, slots=True)
class ExactParityResult:
    success: bool
    scenario_id: str
    comparisons: Mapping[str, bool]
    distributed: TopologyRun
    central_local: TopologyRun

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimandId": "E02-S01-E02",
            "success": self.success,
            "scenarioId": self.scenario_id,
            "comparisons": dict(self.comparisons),
            "contracts": {
                "distributed": self.distributed.contract.to_dict(),
                "centralLocal": self.central_local.contract.to_dict(),
            },
            "eventDigest": self.distributed.result.event_digest,
            "stopReason": self.distributed.result.summary["stopReason"],
            "ledger": dict(self.distributed.result.summary["ledger"]),
        }


def evaluate_no_fault_parity(
    scenario: Scenario,
    *,
    trace_mode: Literal["full", "digest"] = "full",
) -> ExactParityResult:
    if any(cell.fault != FaultMode.NORMAL for cell in scenario.cells):
        raise ValueError("E02-S01-E02 parity population requires no faults")
    distributed = run_with_contract(
        scenario,
        MatchedExecutionContract(ControlTopology.DISTRIBUTED_LOCAL),
        trace_mode=trace_mode,
    )
    central = run_with_contract(
        scenario,
        MatchedExecutionContract(ControlTopology.CENTRAL_LOCAL_PROPOSAL_K1),
        trace_mode=trace_mode,
    )
    comparisons = {
        "resultBytes": distributed.result.to_json_bytes() == central.result.to_json_bytes(),
        "transitionBytes": distributed.transition_bytes() == central.transition_bytes(),
        "finalState": distributed.result.final_state == central.result.final_state,
        "stopReason": distributed.result.summary["stopReason"] == central.result.summary["stopReason"],
        "eventDigest": distributed.result.event_digest == central.result.event_digest,
        "events": distributed.result.events == central.result.events,
        "ledger": distributed.result.summary["ledger"] == central.result.summary["ledger"],
    }
    return ExactParityResult(
        success=all(comparisons.values()),
        scenario_id=scenario.scenario_id,
        comparisons=comparisons,
        distributed=distributed,
        central_local=central,
    )
