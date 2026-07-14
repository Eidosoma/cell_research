#!/usr/bin/env python3
"""Generate the compact S03 architecture validation evidence package."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import inspect
from itertools import permutations, product
import json
from pathlib import Path
from typing import Any

from causal_simulator.action_interface import (
    CommonActionInterface,
    ControlTopology,
    ForbiddenInformationError,
    OperationMeter,
    PolicyNativeReadGateway,
    ReadCapability,
)
from causal_simulator.architectures import (
    PRESPECIFICATION_SHA256,
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
    ControlArchitecture,
    CoordinatorSignal,
    FrozenWeakCoordinator,
    WeakCoordinatorParameters,
    coordinator_input_fields,
    exact_replay_architecture,
    run_architecture,
)
from reference_simulator.api import create_scenario
from reference_simulator.engine import initial_state
from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    ProposalKind,
    Scenario,
    canonical_json_bytes,
)
from reference_simulator.policies import cell_view_proposal


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "s03_architecture_fixtures.json"
PROFILES = {
    "reverse8": [8, 7, 6, 5, 4, 3, 2, 1],
    "alternating8": [8, 1, 7, 2, 6, 3, 5, 4],
    "duplicates8": [3, 1, 3, 2, 2, 1, 4, 4],
    "rotated8": [4, 5, 6, 7, 8, 1, 2, 3],
}
SEEDS = (3, 17, 101, 2026, 20260714)


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def contract_for(architecture: str, coordinator: str) -> ArchitectureExecutionContract:
    contracts = {
        ("distributed_local", "none"): ArchitectureExecutionContract.distributed_local(),
        (
            "central_local_proposal_k1",
            "common_validator_only",
        ): ArchitectureExecutionContract.central_local_k1(),
        (
            "distributed_weak_coordinator",
            "none",
        ): ArchitectureExecutionContract.distributed_weak(enabled=False),
        (
            "distributed_weak_coordinator",
            "weak_frozen_budget",
        ): ArchitectureExecutionContract.distributed_weak(),
    }
    return contracts[(architecture, coordinator)]


def validate_prespecification(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    decoded = json.loads(payload)
    checks = {
        "sha256MatchesImplementation": digest == PRESPECIFICATION_SHA256,
        "researchStepIdIsS03": decoded.get("researchStepId") == "S03",
        "statusFrozenBeforeImplementation": decoded.get("status")
        == "frozen_before_implementation",
        "profileMatches": decoded["weakCoordinator"]["profileId"]
        == "weak_nonlocal_veto_p8_v1",
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"path": str(path), "sha256": digest, "checks": checks}


def validate_hand_fixtures() -> dict[str, Any]:
    cases = json.loads(FIXTURES.read_text())["cases"]
    records: list[dict[str, Any]] = []
    for item in cases:
        policy = Policy(item["policy"])
        direction = Direction(item["direction"])
        scenario = Scenario.create(
            tuple(Cell(cell_id, value, policy, direction) for cell_id, value in item["cells"]),
            initial_occupancy=tuple(cell_id for cell_id, _ in item["cells"]),
            max_activations=100,
            generation_key="E02-S03/validation-fixture/" + item["id"],
        )
        state = initial_state(scenario)
        state.activation_count = item["activationCount"]
        router = ArchitectureProposalRouter(
            contract_for(item["architecture"], item["coordinator"])
        )
        proposal = router.proposal_for(
            scenario,
            state,
            item["actor"],
            side=item["side"],
        )
        ledger = router.architecture_ledger()
        checks = {
            "kind": proposal.kind.value == item["expectedKind"],
            "reason": proposal.reason == item["expectedReason"],
            "eligibleDecisions": ledger["coordinatorEligibleDecisions"]
            == item["expectedEligibleDecisions"],
            "interventions": ledger["coordinatorInterventions"]
            == item["expectedInterventions"],
            "messages": ledger["coordinatorMessages"]
            == 2 * item["expectedEligibleDecisions"],
            "messageBits": ledger["coordinatorMessageBits"]
            == 2 * item["expectedEligibleDecisions"],
        }
        records.append(
            {
                "fixtureId": item["id"],
                "success": all(checks.values()),
                "checks": checks,
                "selectedProposal": proposal.to_dict(),
                "architectureLedger": ledger,
                "coordinationAudit": [audit.to_dict() for audit in router.audits],
            }
        )
    if not all(record["success"] for record in records):
        raise AssertionError("hand architecture fixture failure")
    return {
        "schemaVersion": "E02.S03.architecture-fixture-results.v1",
        "sourceFixturePath": str(FIXTURES),
        "fixtureCount": len(records),
        "passed": sum(record["success"] for record in records),
        "records": records,
    }


def validate_information_boundaries() -> dict[str, Any]:
    denied_policy_reads = 0
    for policy in Policy:
        scenario = Scenario.create(
            (
                Cell("actor", 2, policy, analysis_label="forbidden-actor-label"),
                Cell("target", 1, Policy.BUBBLE, analysis_label="forbidden-target-label"),
            ),
            initial_occupancy=("actor", "target"),
            generation_key="E02-S03/permission/" + policy.value,
        )
        gateway = PolicyNativeReadGateway(
            scenario, initial_state(scenario), "actor", OperationMeter()
        )
        for capability in set(ReadCapability) - gateway.allowed_capabilities:
            try:
                gateway.read(capability)
            except ForbiddenInformationError:
                denied_policy_reads += 1
            else:
                raise AssertionError(f"forbidden read returned: {policy}/{capability}")

    method_source = inspect.getsource(FrozenWeakCoordinator.decide)
    forbidden_source_tokens = (
        "scenario",
        "run_state",
        "envelope",
        "occupancy",
        "cell_map",
        "analysis_label",
        "ledger",
        "random_draw",
    )
    source_token_checks = {
        token: token not in method_source for token in forbidden_source_tokens
    }
    coordinator = FrozenWeakCoordinator(WeakCoordinatorParameters())
    forbidden_attributes = ("scenario", "state", "envelope", "occupancy", "cells", "ledger")
    attribute_checks = {
        name: not hasattr(coordinator, name) for name in forbidden_attributes
    }
    signal_fields = tuple(item.name for item in fields(CoordinatorSignal))
    checks = {
        "allPolicyForbiddenReadsDenied": denied_policy_reads == 21,
        "coordinatorSignalFieldsExact": signal_fields
        == ("event_index", "is_nonlocal_swap"),
        "stableCoordinatorInputFieldsExact": coordinator_input_fields() == signal_fields,
        "coordinatorMethodHasOnlySignalArgument": tuple(
            inspect.signature(FrozenWeakCoordinator.decide).parameters
        )
        == ("self", "signal"),
        "coordinatorSourceHasNoForbiddenTokens": all(source_token_checks.values()),
        "coordinatorHasNoForbiddenAttributes": all(attribute_checks.values()),
        "matchedContractsRemainPolicyNativeLocal": all(
            contract.information_permission.value == "policy_native_local"
            for contract in (
                ArchitectureExecutionContract.central_local_k1(),
                ArchitectureExecutionContract.distributed_local(),
                ArchitectureExecutionContract.distributed_weak(enabled=False),
                ArchitectureExecutionContract.distributed_weak(),
            )
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {
        "success": True,
        "checks": checks,
        "deniedPolicyReadProbes": denied_policy_reads,
        "coordinatorSignalFields": list(signal_fields),
        "sourceTokenChecks": source_token_checks,
        "attributeChecks": attribute_checks,
    }


def validate_exhaustive_primitives() -> dict[str, Any]:
    checked = 0
    interventions = 0
    legal_outputs = 0
    for policy, direction, order, fault_values in product(
        tuple(Policy),
        tuple(Direction),
        permutations((0, 1, 2)),
        product(tuple(FaultMode), repeat=3),
    ):
        cells = tuple(
            Cell(f"c{index}", value, policy, direction, fault)
            for index, (value, fault) in enumerate(zip(order, fault_values))
        )
        scenario = Scenario.create(
            cells,
            initial_occupancy=("c0", "c1", "c2"),
            generation_key=f"E02-S03/exhaustive/{policy}/{direction}/{order}/{fault_values}",
        )
        state = initial_state(scenario)
        state.activation_count = 7
        for actor_id in state.occupancy:
            sides = ("left", "right") if policy == Policy.BUBBLE else (None,)
            for side in sides:
                reference = cell_view_proposal(scenario, state, actor_id, side=side)
                direct = ArchitectureProposalRouter(
                    ArchitectureExecutionContract.distributed_local()
                ).proposal_for(scenario, state, actor_id, side=side)
                central = ArchitectureProposalRouter(
                    ArchitectureExecutionContract.central_local_k1()
                ).proposal_for(scenario, state, actor_id, side=side)
                weak_none = ArchitectureProposalRouter(
                    ArchitectureExecutionContract.distributed_weak(enabled=False)
                ).proposal_for(scenario, state, actor_id, side=side)
                weak_router = ArchitectureProposalRouter(
                    ArchitectureExecutionContract.distributed_weak()
                )
                weak = weak_router.proposal_for(scenario, state, actor_id, side=side)
                if not (
                    direct.to_dict()
                    == central.to_dict()
                    == weak_none.to_dict()
                    == reference.to_dict()
                ):
                    raise AssertionError("matched primitive differential failure")
                is_nonlocal_swap = (
                    reference.kind == ProposalKind.SWAP
                    and reference.target_pos is not None
                    and abs(reference.target_pos - reference.actor_pos) > 1
                )
                if is_nonlocal_swap:
                    interventions += 1
                    if not (
                        weak.kind == ProposalKind.NO_OP
                        and weak.reason == "coordinator_veto_nonlocal_swap"
                        and weak.target_pos is None
                        and weak.new_cursor is None
                        and weak.observation_reads == reference.observation_reads
                        and weak.value_comparisons == reference.value_comparisons
                    ):
                        raise AssertionError("weak veto rule failure")
                elif weak.to_dict() != reference.to_dict():
                    raise AssertionError("weak forwarding changed a legal primitive")
                if weak.kind in set(ProposalKind):
                    legal_outputs += 1
                if weak_router.architecture_ledger()["coordinatorEligibleDecisions"] != 1:
                    raise AssertionError("eligible decision accounting failure")
                checked += 1
    if checked != 3888 or legal_outputs != checked or interventions == 0:
        raise AssertionError((checked, legal_outputs, interventions))
    return {
        "success": True,
        "opportunitiesChecked": checked,
        "matchedReferenceEqual": checked,
        "legalWeakOutputs": legal_outputs,
        "weakInterventionFixtures": interventions,
    }


def make_legacy_scenario(cell_scenario: Scenario, generation_key: str) -> Scenario:
    policies = {cell.policy for cell in cell_scenario.cells}
    if len(policies) != 1:
        raise ValueError("legacy validation requires homogeneous policy")
    return Scenario.create(
        cell_scenario.cells,
        initial_occupancy=cell_scenario.initial_occupancy,
        seed=cell_scenario.seed,
        max_activations=cell_scenario.max_activations,
        architecture=Architecture.TRADITIONAL,
        traditional_policy=next(iter(policies)),
        generation_key=generation_key,
    )


def validate_run_grid() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    base_scenarios = 0
    matched_runs = 0
    legacy_runs = 0
    replay_passed = 0
    completion_passed = 0
    opportunity_passed = 0
    central_exact = 0
    weak_none_exact = 0
    weak_active_native_exact = 0
    weak_active_final_values_equal = 0
    weak_eligible_decisions = 0
    weak_interventions = 0

    for profile_name, values in PROFILES.items():
        for policy, direction, seed in product(tuple(Policy), tuple(Direction), SEEDS):
            base_scenarios += 1
            key = f"E02-S03/grid/{profile_name}/{policy.value}/{direction.value}/{seed}"
            scenario = create_scenario(
                values,
                policy=policy,
                direction=direction,
                seed=seed,
                max_activations=200_000,
                generation_key=key + "/cell",
                permute=True,
            )
            contracts = (
                ArchitectureExecutionContract.distributed_local(),
                ArchitectureExecutionContract.central_local_k1(),
                ArchitectureExecutionContract.distributed_weak(enabled=False),
                ArchitectureExecutionContract.distributed_weak(),
            )
            runs = [
                run_architecture(scenario, contract, trace_mode="digest")
                for contract in contracts
            ]
            matched_runs += len(runs)
            for architecture_run in runs:
                completed = bool(architecture_run.result.summary["completed"])
                replay = (
                    exact_replay_architecture(architecture_run).to_json_bytes()
                    == architecture_run.to_json_bytes()
                )
                opportunity = all(architecture_run.opportunity_validation().values())
                completion_passed += completed
                replay_passed += replay
                opportunity_passed += opportunity
                if not (completed and replay and opportunity):
                    raise AssertionError(
                        {
                            "contract": architecture_run.contract.to_dict(),
                            "completed": completed,
                            "replay": replay,
                            "opportunity": architecture_run.opportunity_validation(),
                        }
                    )

            direct, central, weak_none, weak_active = runs
            central_equal = direct.result.to_json_bytes() == central.result.to_json_bytes()
            weak_none_equal = direct.result.to_json_bytes() == weak_none.result.to_json_bytes()
            weak_active_equal = direct.result.to_json_bytes() == weak_active.result.to_json_bytes()
            final_values_equal = (
                direct.result.summary["finalValues"] == weak_active.result.summary["finalValues"]
            )
            central_exact += central_equal
            weak_none_exact += weak_none_equal
            weak_active_native_exact += weak_active_equal
            weak_active_final_values_equal += final_values_equal
            weak_eligible_decisions += weak_active.architecture_ledger[
                "coordinatorEligibleDecisions"
            ]
            weak_interventions += weak_active.architecture_ledger[
                "coordinatorInterventions"
            ]
            if not (central_equal and weak_none_equal and final_values_equal):
                raise AssertionError("matched parity/final-task validation failed")

            legacy_scenario = make_legacy_scenario(scenario, key + "/legacy")
            legacy = run_architecture(
                legacy_scenario,
                ArchitectureExecutionContract.central_global_legacy(),
                trace_mode="digest",
            )
            legacy_runs += 1
            legacy_completed = bool(legacy.result.summary["completed"])
            legacy_replay = (
                exact_replay_architecture(legacy).to_json_bytes() == legacy.to_json_bytes()
            )
            legacy_opportunity = all(legacy.opportunity_validation().values())
            completion_passed += legacy_completed
            replay_passed += legacy_replay
            opportunity_passed += legacy_opportunity
            if not (legacy_completed and legacy_replay and legacy_opportunity):
                raise AssertionError("legacy descriptive validation failed")

            records.append(
                {
                    "scenarioKey": key,
                    "profile": profile_name,
                    "policy": policy.value,
                    "direction": direction.value,
                    "seed": str(seed),
                    "scenarioId": scenario.scenario_id,
                    "centralLocalExactNativeParity": central_equal,
                    "weakNoneExactNativeParity": weak_none_equal,
                    "weakActiveExactNativeParity": weak_active_equal,
                    "weakActiveFinalValuesEqual": final_values_equal,
                    "distributedActivations": direct.result.summary["activationCount"],
                    "weakActivations": weak_active.result.summary["activationCount"],
                    "weakEligibleDecisions": weak_active.architecture_ledger[
                        "coordinatorEligibleDecisions"
                    ],
                    "weakInterventions": weak_active.architecture_ledger[
                        "coordinatorInterventions"
                    ],
                    "legacyActivations": legacy.result.summary["activationCount"],
                    "allFiveCompleted": all(
                        run_item.result.summary["completed"] for run_item in (*runs, legacy)
                    ),
                }
            )

    total_runs = matched_runs + legacy_runs
    summary = {
        "success": True,
        "baseNoFaultScenarios": base_scenarios,
        "matchedArchitectureRuns": matched_runs,
        "legacyDescriptiveRuns": legacy_runs,
        "totalRuns": total_runs,
        "completionPassed": completion_passed,
        "deterministicReplayPassed": replay_passed,
        "opportunityAndLedgerValidationPassed": opportunity_passed,
        "centralLocalExactNativeParity": central_exact,
        "weakNoneExactNativeParity": weak_none_exact,
        "weakActiveExactNativeParity": weak_active_native_exact,
        "weakActiveFinalValuesEqual": weak_active_final_values_equal,
        "weakCoordinatorEligibleDecisions": weak_eligible_decisions,
        "weakCoordinatorInterventions": weak_interventions,
    }
    if not (
        completion_passed == total_runs
        and replay_passed == total_runs
        and opportunity_passed == total_runs
        and central_exact == base_scenarios
        and weak_none_exact == base_scenarios
        and weak_active_final_values_equal == base_scenarios
    ):
        raise AssertionError(summary)

    residual_rows = [
        {
            "contrast": "central_local_proposal_k1 vs distributed_local",
            "matchedContrastEligible": "true",
            "informationAsymmetry": "none",
            "legalPrimitiveAsymmetry": "none",
            "opportunityAsymmetry": "none",
            "coordinationCostAsymmetry": "none",
            "observedNativeExactPairs": f"{central_exact}/{base_scenarios}",
            "residualAsymmetry": "ownership/routing label only; contract is out-of-band",
        },
        {
            "contrast": "distributed_weak_coordinator:none vs distributed_local",
            "matchedContrastEligible": "true",
            "informationAsymmetry": "none",
            "legalPrimitiveAsymmetry": "none",
            "opportunityAsymmetry": "none",
            "coordinationCostAsymmetry": "none",
            "observedNativeExactPairs": f"{weak_none_exact}/{base_scenarios}",
            "residualAsymmetry": "architecture ownership label only; contract is out-of-band",
        },
        {
            "contrast": "distributed_weak_coordinator:weak vs none",
            "matchedContrastEligible": "true",
            "informationAsymmetry": "coordinator receives one derived bit on eligible opportunities; policy unchanged",
            "legalPrimitiveAsymmetry": "none; veto selects existing NoOp",
            "opportunityAsymmetry": "none; one actor and one final proposal",
            "coordinationCostAsymmetry": "two messages/two bits per eligible opportunity",
            "observedNativeExactPairs": f"{weak_active_native_exact}/{base_scenarios}",
            "residualAsymmetry": "prespecified information, intervention, and charged message costs",
        },
        {
            "contrast": "central_global_legacy vs distributed_local",
            "matchedContrastEligible": "false",
            "informationAsymmetry": "full global scan vs policy-native local",
            "legalPrimitiveAsymmetry": "global controller-owned nonlocal actions vs actor proposal",
            "opportunityAsymmetry": "controller step vs scheduled actor activation",
            "coordinationCostAsymmetry": "not comparable under matched ledger",
            "observedNativeExactPairs": "not tested as parity",
            "residualAsymmetry": "declared S01 descriptive bundle",
        },
    ]
    return summary, records, residual_rows


def information_access_rows() -> list[dict[str, str]]:
    legal = "NoOp|Swap|MemoryUpdate"
    return [
        {
            "architecture": "central_global_legacy",
            "coordinatorProfile": "legacy_global",
            "matchedContrastEligible": "false",
            "policyEndpointInformation": "not applicable; controller owns proposal",
            "coordinatorInformation": "full global state",
            "rawGlobalRead": "declared",
            "proposalCandidatesPerOpportunity": "not comparable",
            "legalPrimitiveSet": "global traditional primitives (unmatched)",
            "scheduler": "traditional_controller",
        },
        {
            "architecture": "central_local_proposal_k1",
            "coordinatorProfile": "common_validator_only",
            "matchedContrastEligible": "true",
            "policyEndpointInformation": "policy_native_local",
            "coordinatorInformation": "one ActionEnvelope; no selection state",
            "rawGlobalRead": "forbidden",
            "proposalCandidatesPerOpportunity": "1",
            "legalPrimitiveSet": legal,
            "scheduler": "uniform_random_activation",
        },
        {
            "architecture": "distributed_local",
            "coordinatorProfile": "none",
            "matchedContrastEligible": "true",
            "policyEndpointInformation": "policy_native_local",
            "coordinatorInformation": "none",
            "rawGlobalRead": "forbidden",
            "proposalCandidatesPerOpportunity": "1",
            "legalPrimitiveSet": legal,
            "scheduler": "uniform_random_activation",
        },
        {
            "architecture": "distributed_weak_coordinator",
            "coordinatorProfile": "none",
            "matchedContrastEligible": "true",
            "policyEndpointInformation": "policy_native_local",
            "coordinatorInformation": "none",
            "rawGlobalRead": "forbidden",
            "proposalCandidatesPerOpportunity": "1",
            "legalPrimitiveSet": legal,
            "scheduler": "uniform_random_activation",
        },
        {
            "architecture": "distributed_weak_coordinator",
            "coordinatorProfile": "weak_frozen_budget",
            "matchedContrastEligible": "true",
            "policyEndpointInformation": "policy_native_local",
            "coordinatorInformation": "event_index + one is_nonlocal_swap bit",
            "rawGlobalRead": "forbidden",
            "proposalCandidatesPerOpportunity": "1",
            "legalPrimitiveSet": legal,
            "scheduler": "uniform_random_activation",
        },
    ]


def capability_rows() -> list[dict[str, str]]:
    return [
        {
            "architecture": item.value,
            "implemented": "true",
            "matchedContrastEligible": str(
                item != ControlArchitecture.CENTRAL_GLOBAL_LEGACY
            ).lower(),
            "noFaultCompletionValidated": "true",
            "deterministicReplayValidated": "true",
            "ledgerValidated": "true",
            "kGreaterThanOneSupported": "false",
            "notes": (
                "existing E01 descriptive adapter; declared unmatched"
                if item == ControlArchitecture.CENTRAL_GLOBAL_LEGACY
                else "S02 boundaries preserved"
            ),
        }
        for item in ControlArchitecture
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows must be nonempty")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prespec", type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    package = output / "architecture_package"
    package.mkdir(parents=True, exist_ok=True)
    prespec = args.prespec or package / "weak_coordinator_prespecification.json"

    prespec_result = validate_prespecification(prespec)
    fixture_results = validate_hand_fixtures()
    information_result = validate_information_boundaries()
    primitive_result = validate_exhaustive_primitives()
    run_grid_result, run_records, residual_rows = validate_run_grid()

    write_json(package / "architecture_fixture_results.json", fixture_results)
    write_json(output / "architecture_fixtures.json", fixture_results)
    write_json(package / "information_boundary_validation.json", information_result)
    write_json(package / "primitive_differential_validation.json", primitive_result)
    write_json(package / "no_fault_run_records.json", {
        "schemaVersion": "E02.S03.no-fault-run-records.v1",
        "records": run_records,
    })
    write_csv(package / "information_access_table.csv", information_access_rows())
    write_csv(package / "architecture_capability_matrix.csv", capability_rows())
    write_csv(package / "residual_asymmetry_table.csv", residual_rows)

    validation = {
        "schemaVersion": "E02.S03.architecture-validation.v1",
        "researchStepId": "S03",
        "success": True,
        "prespecification": prespec_result,
        "handFixtures": {
            "passed": fixture_results["passed"],
            "total": fixture_results["fixtureCount"],
        },
        "informationBoundaries": information_result,
        "primitiveDifferential": primitive_result,
        "runGrid": run_grid_result,
        "escalationTriggerReached": False,
    }
    write_json(package / "validation_summary.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
