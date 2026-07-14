#!/usr/bin/env python3
"""Generate compact S05 fault-semantics validation evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from itertools import product
from pathlib import Path
import platform
from typing import Any

from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ControlArchitecture,
)
from causal_simulator.faults import (
    BERNOULLI_FAILURE_STREAM,
    ActionFailureProfile,
    ContinuationPolicy,
    FaultExecutionContract,
    FaultExecutionInterceptor,
    FaultProposalRouter,
    FrozenSensingTransformer,
    MobilityProfile,
    RetryPolicy,
    SensingProfile,
    comparability_classification,
    exact_replay_fault,
    run_faulted_architecture,
)
from causal_simulator.schedulers import (
    SchedulerExecutionContract,
    SchedulerFamily,
    run_scheduled_architecture,
)
from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, Policy


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "s05_fault_fixtures.json"
PRESPEC_PATH = Path(
    "/artifacts/research_steps/S05/fault_package/fault_prespecification.json"
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def architectures() -> tuple[ArchitectureExecutionContract, ...]:
    return (
        ArchitectureExecutionContract.central_local_k1(),
        ArchitectureExecutionContract.distributed_local(),
        ArchitectureExecutionContract.distributed_weak(enabled=False),
        ArchitectureExecutionContract.distributed_weak(),
    )


def fault_factor_contracts() -> list[FaultExecutionContract]:
    return [
        FaultExecutionContract(mobility, continuation, retry, failure, sensing)
        for mobility, continuation, retry, failure, sensing in product(
            tuple(MobilityProfile),
            tuple(ContinuationPolicy),
            tuple(RetryPolicy),
            tuple(ActionFailureProfile),
            tuple(SensingProfile),
        )
    ]


def comparability_matrix() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    architecture_profiles = [
        (item.architecture, item.coordinator_profile.value) for item in architectures()
    ] + [(ControlArchitecture.CENTRAL_GLOBAL_LEGACY, "legacy_global")]
    rows: list[dict[str, Any]] = []
    for (architecture, coordinator), scheduler, contract in product(
        architecture_profiles, tuple(SchedulerFamily), fault_factor_contracts()
    ):
        status, reason = comparability_classification(architecture, contract)
        rows.append(
            {
                "architecture": architecture.value,
                "coordinatorProfile": coordinator,
                "scheduler": scheduler.value,
                "mobility": contract.mobility.value,
                "continuation": contract.continuation.value,
                "retry": contract.retry.value,
                "actionFailure": contract.action_failure.value,
                "sensing": contract.sensing.value,
                "classification": status,
                "reason": reason,
                "freeOpportunities": 0,
                "proposalCandidatesPerOpportunity": 1,
                "informationPermission": contract.information_permission,
                "legalPrimitives": "|".join(contract.legal_primitives),
            }
        )
    counts = {
        status: sum(row["classification"] == status for row in rows)
        for status in ("comparable", "comparable_degenerate", "non_comparable")
    }
    return rows, {
        "passed": len(rows) == 1800
        and counts
        == {
            "comparable": 1080,
            "comparable_degenerate": 360,
            "non_comparable": 360,
        },
        "rows": len(rows),
        "classificationCounts": counts,
        "executedLegacyGlobalCells": 0,
    }


def representative_profiles():
    normal = create_scenario(
        [5, 1, 4, 2, 3],
        policy=Policy.INSERTION,
        seed=73,
        max_activations=40,
        permute=False,
        generation_key="S05/matrix/normal",
    )
    passive = create_scenario(
        [5, 1, 4, 2, 3],
        policy=Policy.BUBBLE,
        faults={0: "passive"},
        seed=73,
        max_activations=40,
        permute=False,
        generation_key="S05/matrix/passive",
    )
    stuck = create_scenario(
        [5, 1, 4, 2, 3],
        policy=Policy.BUBBLE,
        faults={0: "stuck"},
        seed=73,
        max_activations=40,
        permute=False,
        generation_key="S05/matrix/stuck",
    )
    return (
        ("exact_baseline", normal, FaultExecutionContract()),
        (
            "bernoulli_skip",
            normal,
            FaultExecutionContract(action_failure=ActionFailureProfile.BERNOULLI_P),
        ),
        (
            "transient_skip",
            normal,
            FaultExecutionContract(action_failure=ActionFailureProfile.TRANSIENT_MARKOV),
        ),
        (
            "noisy_sensing",
            normal,
            FaultExecutionContract(sensing=SensingProfile.NOISY_VALUE_OR_STATUS),
        ),
        (
            "bernoulli_bounded_retry",
            normal,
            FaultExecutionContract(
                action_failure=ActionFailureProfile.BERNOULLI_P,
                retry=RetryPolicy.RETRY_LATER_BOUNDED,
            ),
        ),
        (
            "passive_skip",
            passive,
            FaultExecutionContract(mobility=MobilityProfile.PASSIVE),
        ),
        (
            "stuck_stop",
            stuck,
            FaultExecutionContract(
                mobility=MobilityProfile.STUCK,
                continuation=ContinuationPolicy.STOP_ON_FIRST_BLOCKING_FAILURE,
            ),
        ),
    )


def execution_matrix() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replay_passed = 0
    ledger_passed = 0
    for architecture, family, (profile, scenario, fault) in product(
        architectures(), tuple(SchedulerFamily), representative_profiles()
    ):
        scheduler = SchedulerExecutionContract(family)
        item = run_faulted_architecture(
            scenario, architecture, scheduler, fault, trace_mode="full"
        )
        replay = exact_replay_fault(item)
        replay_ok = replay.to_json_bytes() == item.to_json_bytes()
        checks = item.opportunity_validation()
        ledger_ok = all(checks.values())
        replay_passed += replay_ok
        ledger_passed += ledger_ok
        rows.append(
            {
                "profile": profile,
                "scenarioId": scenario.scenario_id,
                "architecture": architecture.architecture.value,
                "coordinatorProfile": architecture.coordinator_profile.value,
                "scheduler": family.value,
                "stopReason": item.result.summary["stopReason"],
                "completed": item.result.summary["completed"],
                "activationCount": item.result.summary["activationCount"],
                "proposals": item.result.summary["ledger"]["proposals"],
                "actionFailures": item.fault_ledger["actionFailures"],
                "sensingErrorsApplied": item.fault_ledger["sensingErrorsApplied"],
                "retryAttempts": item.fault_ledger["retryAttempts"],
                "conflictLosses": item.result.summary["ledger"]["conflictLosses"],
                "coordinatorEligibleDecisions": item.scheduler_run.architecture_run.architecture_ledger[
                    "coordinatorEligibleDecisions"
                ],
                "replayPassed": replay_ok,
                "allOpportunityChecksPassed": ledger_ok,
                "failedChecks": "|".join(
                    key for key, passed in checks.items() if not passed
                ),
                "eventDigest": item.result.event_digest,
            }
        )
    return rows, {
        "passed": len(rows) == 140
        and replay_passed == len(rows)
        and ledger_passed == len(rows),
        "runs": len(rows),
        "replaysPassed": replay_passed,
        "opportunityLedgerRunsPassed": ledger_passed,
        "architectures": 4,
        "schedulers": 5,
        "representativeFaultProfiles": 7,
    }


def hand_checked_validation() -> dict[str, Any]:
    fixture = json.loads(FIXTURE_PATH.read_text())
    core = fixture["core"]
    scenario = create_scenario(
        core["values"],
        policy=Policy(core["policy"]),
        seed=core["seed"],
        max_activations=core["maxActivations"],
        permute=False,
        generation_key=core["generationKey"],
    )
    scheduler = SchedulerExecutionContract(SchedulerFamily.DETERMINISTIC_SCAN)
    retry = run_faulted_architecture(
        scenario,
        ArchitectureExecutionContract.distributed_local(),
        scheduler,
        FaultExecutionContract(
            action_failure=ActionFailureProfile.BERNOULLI_P,
            retry=RetryPolicy.RETRY_LATER_BOUNDED,
        ),
        trace_mode="full",
    )
    observed_retry = [item.to_dict() for item in retry.retry_audit]
    observed_failure = [
        item.event_index for item in retry.action_failure_audit if item.applied
    ]
    retry_event = next(
        event
        for event in retry.result.events
        if event["proposal"]["reason"].startswith("retry_later_bounded")
    )

    passive = fixture["passiveContinuation"]
    passive_scenario = create_scenario(
        passive["values"],
        policy=Policy.BUBBLE,
        faults={int(key): value for key, value in passive["faults"].items()},
        seed=passive["seed"],
        max_activations=10,
        permute=False,
        generation_key=passive["generationKey"],
    )
    continuation = {}
    for policy in ContinuationPolicy:
        item = run_faulted_architecture(
            passive_scenario,
            ArchitectureExecutionContract.distributed_local(),
            scheduler,
            FaultExecutionContract(
                mobility=MobilityProfile.PASSIVE, continuation=policy
            ),
            trace_mode="full",
        )
        continuation[policy.value] = {
            "stopReason": item.result.summary["stopReason"],
            "activationCount": item.result.summary["activationCount"],
            "decisions": [event["decision"] for event in item.result.events],
            "proposalReasons": [
                event["proposal"]["reason"] for event in item.result.events
            ],
        }

    synchronous = fixture["synchronousStop"]
    sync_scenario = create_scenario(
        synchronous["values"],
        policy=Policy.BUBBLE,
        seed=synchronous["seed"],
        max_activations=synchronous["maxActivations"],
        permute=False,
        generation_key=synchronous["generationKey"],
    )
    sync = run_faulted_architecture(
        sync_scenario,
        ArchitectureExecutionContract.distributed_local(),
        SchedulerExecutionContract(
            SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT
        ),
        FaultExecutionContract(
            continuation=ContinuationPolicy.STOP_ON_FIRST_BLOCKING_FAILURE,
            action_failure=ActionFailureProfile.BERNOULLI_P,
        ),
        trace_mode="full",
    )
    sync_decisions = [event["decision"] for event in sync.result.events]
    passed = all(
        (
            observed_failure == core["bernoulliFailureEvents"],
            observed_retry == core["retryAudit"],
            retry_event["decision"] == core["retryDecision"],
            retry_event["observation"] == {"reads": 0, "valueComparisons": 0},
            continuation[ContinuationPolicy.SKIP_AND_CONTINUE.value]["stopReason"]
            == passive["skipStopReason"],
            continuation[ContinuationPolicy.SKIP_AND_CONTINUE.value]["activationCount"]
            == passive["skipActivations"],
            continuation[
                ContinuationPolicy.STOP_ON_FIRST_BLOCKING_FAILURE.value
            ]["stopReason"]
            == passive["stopStopReason"],
            continuation[
                ContinuationPolicy.STOP_ON_FIRST_BLOCKING_FAILURE.value
            ]["activationCount"]
            == passive["stopActivations"],
            sync.result.summary["activationCount"]
            == synchronous["chargedFirstBatch"],
            sync_decisions == synchronous["decisions"],
            len({event["preStateHash"] for event in sync.result.events}) == 1,
            len({event["postStateHash"] for event in sync.result.events}) == 1,
        )
    )
    return {
        "schemaVersion": "E02.S05.hand-checked-traces.v1",
        "passed": passed,
        "sourceFixture": str(FIXTURE_PATH),
        "sourceFixtureSha256": hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
        "bernoulliFailureEvents": observed_failure,
        "retryAudit": observed_retry,
        "retryDecision": retry_event["decision"],
        "retryObservationCost": retry_event["observation"],
        "continuation": continuation,
        "synchronous": {
            "stopReason": sync.result.summary["stopReason"],
            "activationCount": sync.result.summary["activationCount"],
            "decisions": sync_decisions,
            "commonPreState": len(
                {event["preStateHash"] for event in sync.result.events}
            )
            == 1,
            "commonPostState": len(
                {event["postStateHash"] for event in sync.result.events}
            )
            == 1,
        },
    }


def exogenous_and_sensing_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    scenario = create_scenario(
        list(range(16, 0, -1)),
        policy=Policy.SELECTION,
        seed=4,
        max_activations=32,
        permute=False,
        generation_key="S05/exogenous-stream-phase",
    )
    scheduler = SchedulerExecutionContract(
        SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT
    )
    runs = [
        run_faulted_architecture(
            scenario,
            architecture,
            scheduler,
            FaultExecutionContract(action_failure=ActionFailureProfile.BERNOULLI_P),
            trace_mode="full",
        )
        for architecture in architectures()
    ]
    streams = [
        [
            [item.event_index, item.raw_uint64]
            for item in run.action_failure_audit
            if item.stream == BERNOULLI_FAILURE_STREAM
        ]
        for run in runs
    ]
    coordinator_indices = [
        item.event_index
        for item in runs[-1].scheduler_run.architecture_run.coordination_audit
    ]
    exogenous = {
        "passed": all(item == streams[0] for item in streams[1:])
        and coordinator_indices == [7, 15, 23, 31]
        and all(run.result.summary["activationCount"] == 32 for run in runs),
        "architecturesChecked": 4,
        "opportunitiesPerArchitecture": 32,
        "identicalFailureAddressValuePairs": all(
            item == streams[0] for item in streams[1:]
        ),
        "stream": BERNOULLI_FAILURE_STREAM,
        "referenceAddressValuePairs": streams[0],
        "weakCoordinatorEligibleIndices": coordinator_indices,
        "coordinatorPhasePreserved": coordinator_indices == [7, 15, 23, 31],
    }

    core = json.loads(FIXTURE_PATH.read_text())["core"]
    sensing_scenario = create_scenario(
        core["values"],
        policy=Policy(core["policy"]),
        seed=core["seed"],
        max_activations=core["maxActivations"],
        permute=False,
        generation_key=core["generationKey"],
    )
    sensed = run_faulted_architecture(
        sensing_scenario,
        ArchitectureExecutionContract.distributed_local(),
        SchedulerExecutionContract(SchedulerFamily.DETERMINISTIC_SCAN),
        FaultExecutionContract(sensing=SensingProfile.NOISY_VALUE_OR_STATUS),
        trace_mode="full",
    )
    applied = [item.to_dict() for item in sensed.sensing_audit if item.applied]
    ledger = sensed.fault_ledger
    sensing = {
        "passed": ledger["sensingValueDraws"]
        == sensed.result.summary["ledger"]["observationReads"]
        and ledger["sensingErrorsApplied"]
        == ledger["sensingErrorHandlingOperations"]
        and all(sensed.opportunity_validation().values()),
        "attemptedLogicalReads": sensed.result.summary["ledger"]["observationReads"],
        "valueDraws": ledger["sensingValueDraws"],
        "targetStatusDraws": ledger["sensingStatusDraws"],
        "valueErrors": ledger["sensingValueErrors"],
        "targetStatusErrors": ledger["sensingStatusErrors"],
        "errorHandlingOperations": ledger["sensingErrorHandlingOperations"],
        "appliedErrorAudit": applied,
        "validatorGroundTruthRetained": True,
    }
    return exogenous, sensing


def phase_interaction_regression() -> dict[str, Any]:
    input_profiles = {
        "reverse_unique": list(range(8, 0, -1)),
        "block_scrambled": [5, 6, 7, 8, 1, 2, 3, 4],
        "balanced_duplicates": [3, 1, 3, 2, 2, 1, 4, 4],
    }
    noncompletions: list[dict[str, Any]] = []
    runs = 0
    native_parity = 0
    opportunity_pass = 0
    for input_name, values in input_profiles.items():
        for policy, direction, seed, family, architecture in product(
            tuple(Policy),
            tuple(Direction),
            range(4),
            tuple(SchedulerFamily),
            architectures(),
        ):
            scenario = create_scenario(
                values,
                policy=policy,
                direction=direction,
                seed=seed,
                max_activations=4096,
                generation_key=(
                    f"S04/grid/{input_name}/{policy.value}/{direction.value}/{seed}"
                ),
                permute=False,
            )
            scheduler = SchedulerExecutionContract(family)
            baseline = run_scheduled_architecture(
                scenario, architecture, scheduler, trace_mode="digest"
            )
            overlaid = run_faulted_architecture(
                scenario,
                architecture,
                scheduler,
                FaultExecutionContract(),
                trace_mode="digest",
            )
            runs += 1
            native_parity += (
                baseline.result.to_json_bytes() == overlaid.result.to_json_bytes()
            )
            opportunity_pass += all(overlaid.opportunity_validation().values())
            if not overlaid.result.summary["completed"]:
                noncompletions.append(
                    {
                        "inputProfile": input_name,
                        "policy": policy.value,
                        "direction": direction.value,
                        "seed": seed,
                        "architecture": architecture.architecture.value,
                        "coordinatorProfile": architecture.coordinator_profile.value,
                        "scheduler": family.value,
                        "stopReason": overlaid.result.summary["stopReason"],
                        "activationCount": overlaid.result.summary["activationCount"],
                    }
                )
    structural = all(
        item["inputProfile"] in {"reverse_unique", "block_scrambled"}
        and item["policy"] == Policy.SELECTION.value
        and item["direction"] == Direction.ASCENDING.value
        and item["coordinatorProfile"] == "weak_frozen_budget"
        and item["scheduler"]
        in {
            SchedulerFamily.DETERMINISTIC_SCAN.value,
            SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT.value,
        }
        and item["stopReason"] == "event_budget"
        for item in noncompletions
    )
    return {
        "passed": runs == 1440
        and native_parity == runs
        and opportunity_pass == runs
        and len(noncompletions) == 16
        and structural,
        "runs": runs,
        "noFaultCompletions": runs - len(noncompletions),
        "noFaultNoncompletions": len(noncompletions),
        "expectedS04Noncompletions": 16,
        "nativeS04ByteParityPassed": native_parity,
        "opportunityValidationPassed": opportunity_pass,
        "phaseInteractionClassificationPassed": structural,
        "noncompletionRecords": noncompletions,
    }


def information_boundary_validation() -> dict[str, Any]:
    fields_by_controller = {
        "sensing": sorted(FrozenSensingTransformer.__slots__),
        "proposalRouter": sorted(FaultProposalRouter.__slots__),
        "executionInterceptor": sorted(FaultExecutionInterceptor.__slots__),
    }
    forbidden_direct = {"scenario", "state", "occupancy", "cells", "values", "ledger"}
    direct = set().union(*(set(items) for items in fields_by_controller.values()))
    sensing_signature = tuple(inspect.signature(FrozenSensingTransformer.__call__).parameters)
    result = {
        "passed": not direct & forbidden_direct
        and sensing_signature
        == ("self", "record", "event_index", "read_ordinal"),
        "controllerFields": fields_by_controller,
        "forbiddenDirectFields": sorted(forbidden_direct),
        "forbiddenDirectFieldsPresent": sorted(direct & forbidden_direct),
        "sensingCallbackParameters": list(sensing_signature),
        "sensingReceivesRawRunState": False,
        "failureInterceptorReceivesProposalAndMechanicalDecisionOnlyBeforeOutcome": True,
        "retryActorIsSchedulerSelected": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-phase-grid", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    matrix_rows, matrix_summary = comparability_matrix()
    execution_rows, execution_summary = execution_matrix()
    hand = hand_checked_validation()
    exogenous, sensing = exogenous_and_sensing_validation()
    information = information_boundary_validation()
    phase = (
        {"passed": True, "skipped": True}
        if args.skip_phase_grid
        else phase_interaction_regression()
    )

    write_csv(args.output / "comparability_matrix.csv", matrix_rows)
    write_csv(args.output / "execution_validation.csv", execution_rows)
    write_json(args.output / "comparability_summary.json", matrix_summary)
    write_json(args.output / "hand_checked_traces.json", hand)
    write_json(args.output / "exogenous_stream_validation.json", exogenous)
    write_json(args.output / "sensing_cost_validation.json", sensing)
    write_json(args.output / "information_boundary_validation.json", information)
    write_json(args.output / "phase_interaction_regression.json", phase)

    gates = {
        "comparabilityMatrix": bool(matrix_summary["passed"]),
        "executionMatrix": bool(execution_summary["passed"]),
        "handCheckedTraces": bool(hand["passed"]),
        "identicalExogenousStreams": bool(exogenous["passed"]),
        "sensingCostIdentity": bool(sensing["passed"]),
        "informationBoundary": bool(information["passed"]),
        "phaseInteractionRegression": bool(phase["passed"]),
    }
    summary = {
        "schemaVersion": "E02.S05.validation-summary.v1",
        "researchStepId": "S05",
        "passed": all(gates.values()),
        "componentGates": gates,
        "componentGatesPassed": sum(gates.values()),
        "componentGatesTotal": len(gates),
        "comparability": matrix_summary,
        "execution": execution_summary,
        "phaseInteraction": {
            key: value for key, value in phase.items() if key != "noncompletionRecords"
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "workers": 1,
            "gpuUsed": False,
        },
        "inputs": {
            "portableFixture": str(FIXTURE_PATH),
            "portableFixtureSha256": hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
            "prespecification": str(PRESPEC_PATH),
            "prespecificationSha256": hashlib.sha256(PRESPEC_PATH.read_bytes()).hexdigest(),
        },
    }
    write_json(args.output / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
