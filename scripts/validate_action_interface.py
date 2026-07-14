#!/usr/bin/env python3
"""Generate the compact E02 S02 parity and interface-validation evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from causal_simulator.engine import MatchedExecutionContract, evaluate_no_fault_parity, run_with_contract
from reference_simulator.api import create_scenario, exact_replay
from reference_simulator.engine import initial_state, run
from reference_simulator.model import Cell, Direction, FaultMode, Policy, RunResult, Scenario
from reference_simulator.policies import cell_view_proposal
from reference_simulator.transition_primitives import (
    commit_proposal,
    cost_delta,
    ledger_identity,
    validate_proposal,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "s02_action_interface_fixtures.json"
ARCHIVED_E01_SMOKE = Path("/previous-artifacts/E01/research_steps/S05/smoke_sample_result.json")


def canonical_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fixture_scenario(item: dict[str, Any]) -> tuple[Scenario, Any]:
    default_policy = Policy(item["policy"])
    direction = Direction(item["direction"])
    cells = [
        Cell(
            raw[0],
            raw[1],
            Policy(raw[3]) if len(raw) == 4 else default_policy,
            direction,
            FaultMode(raw[2]),
            analysis_label="forbidden-label",
        )
        for raw in item["cells"]
    ]
    scenario = Scenario.create(
        cells,
        initial_occupancy=[raw[0] for raw in item["cells"]],
        max_activations=50,
        generation_key="E02-S02/validation-fixture/" + item["id"],
    )
    state = initial_state(scenario)
    state.selection_cursors.update(item.get("stateCursors", {}))
    return scenario, state


def validate_fixtures() -> dict[str, Any]:
    source = json.loads(FIXTURES.read_text())
    interface = CommonActionInterface()
    rows = []
    for item in source["fixtures"]:
        scenario, state = fixture_scenario(item)
        side = item.get("side")
        reference = cell_view_proposal(scenario, state, item["actor"], side=side)
        distributed = interface.envelope_for(
            ControlTopology.DISTRIBUTED_LOCAL,
            scenario,
            state,
            item["actor"],
            side=side,
        )
        central = interface.envelope_for(
            ControlTopology.CENTRAL_LOCAL_PROPOSAL_K1,
            scenario,
            state,
            item["actor"],
            side=side,
        )
        expected = item["expected"]
        validation = validate_proposal(scenario, state, distributed.proposal)
        decision = "accepted" if validation.eligible_for_commit else validation.decision
        delta = cost_delta(state.ledger, distributed.proposal, decision)
        snapshot = state.clone()
        commit_proposal(state, snapshot, distributed.proposal, decision)
        for key, value in delta.items():
            state.ledger[key] += value
        checks = {
            "topologyEnvelope": distributed == central,
            "e01Proposal": distributed.proposal.to_dict() == reference.to_dict(),
            "kind": distributed.proposal.kind.value == expected["kind"],
            "reason": distributed.proposal.reason == expected["reason"],
            "reads": distributed.proposal.observation_reads == expected["reads"],
            "comparisons": distributed.proposal.value_comparisons == expected["comparisons"],
            "decision": decision == expected["decision"],
            "postOccupancy": state.occupancy == expected["postOccupancy"],
            "cursor": "newCursor" not in expected
            or state.selection_cursors[item["actor"]] == expected["newCursor"],
            "ledgerIdentity": all(ledger_identity(state.ledger).values()),
        }
        rows.append(
            {
                "fixtureId": item["id"],
                "success": all(checks.values()),
                "checks": checks,
                "proposal": distributed.proposal.to_dict(),
                "decision": decision,
                "ledgerDelta": delta,
                "usedCapabilities": [value.value for value in distributed.used_capabilities],
            }
        )
    return {
        "schema": "E02.S02.differential-fixture-results.v1",
        "success": all(row["success"] for row in rows),
        "fixtureCount": len(rows),
        "fixtureSource": str(FIXTURES.relative_to(ROOT)),
        "results": rows,
    }


def validate_exhaustive_primitives() -> dict[str, Any]:
    interface = CommonActionInterface()
    failures: list[dict[str, Any]] = []
    scenario_count = 0
    opportunity_count = 0
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
            generation_key=(
                f"E02-S02/exhaustive-validation/{policy.value}/{direction.value}/"
                f"{order}/{tuple(item.value for item in fault_values)}"
            ),
        )
        scenario_count += 1
        state = initial_state(scenario)
        for actor_id in state.occupancy:
            sides = ("left", "right") if policy == Policy.BUBBLE else (None,)
            for side in sides:
                reference = cell_view_proposal(scenario, state, actor_id, side=side)
                distributed = interface.proposal_for(
                    ControlTopology.DISTRIBUTED_LOCAL,
                    scenario,
                    state,
                    actor_id,
                    side=side,
                )
                central = interface.proposal_for(
                    ControlTopology.CENTRAL_LOCAL_PROPOSAL_K1,
                    scenario,
                    state,
                    actor_id,
                    side=side,
                )
                opportunity_count += 1
                if not (
                    reference.to_dict() == distributed.to_dict() == central.to_dict()
                ):
                    failures.append(
                        {
                            "scenarioId": scenario.scenario_id,
                            "actorId": actor_id,
                            "side": side,
                            "reference": reference.to_dict(),
                            "distributed": distributed.to_dict(),
                            "centralLocal": central.to_dict(),
                        }
                    )
    return {
        "schema": "E02.S02.exhaustive-primitive-parity.v1",
        "success": not failures,
        "scenarioCount": scenario_count,
        "opportunityCount": opportunity_count,
        "policies": [item.value for item in Policy],
        "directions": [item.value for item in Direction],
        "faultModes": [item.value for item in FaultMode],
        "failureCount": len(failures),
        "failures": failures,
    }


def validate_forbidden_information() -> dict[str, Any]:
    rows = []
    for policy in Policy:
        scenario = Scenario.create(
            (
                Cell("actor", 2, policy, analysis_label="secret-actor"),
                Cell("target", 1, Policy.BUBBLE, analysis_label="secret-target"),
            ),
            initial_occupancy=("actor", "target"),
            generation_key="E02-S02/forbidden-validation/" + policy.value,
        )
        gateway = PolicyNativeReadGateway(
            scenario, initial_state(scenario), "actor", OperationMeter()
        )
        for capability in sorted(
            set(ReadCapability) - gateway.allowed_capabilities, key=lambda item: item.value
        ):
            denied = False
            try:
                gateway.read(capability)
            except ForbiddenInformationError:
                denied = True
            rows.append(
                {
                    "policy": policy.value,
                    "capability": capability.value,
                    "deniedBeforeRead": denied,
                }
            )
    return {
        "schema": "E02.S02.forbidden-information-validation.v1",
        "success": all(row["deniedBeforeRead"] for row in rows),
        "probeCount": len(rows),
        "probes": rows,
        "centralRelayInput": "ActionEnvelope only; no Scenario, RunState, or PolicyObservation",
    }


def validate_parity_matrix() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    profiles = {
        "singleton": [1],
        "sorted_unique": [1, 2, 3, 4],
        "reverse_unique": [4, 3, 2, 1],
        "duplicates": [2, 1, 2, 1],
        "nearly_sorted": [1, 3, 2, 4],
    }
    rows: list[dict[str, Any]] = []
    replay_rows = []
    ledger_rows = []
    for policy, direction, (profile, values) in product(
        tuple(Policy), tuple(Direction), profiles.items()
    ):
        scenario = create_scenario(
            values,
            policy=policy,
            direction=direction,
            seed=20260714,
            max_activations=500,
            generation_key=f"E02-S01-E02/validation/{policy.value}/{direction.value}/{profile}",
            permute=False,
        )
        parity = evaluate_no_fault_parity(scenario, trace_mode="full")
        repeated = [
            run_with_contract(
                scenario, MatchedExecutionContract(topology), trace_mode="full"
            )
            for topology in ControlTopology
        ]
        replay_checks = {
            topology.value: repeated[index].result.to_json_bytes()
            == (
                parity.distributed.result.to_json_bytes()
                if topology == ControlTopology.DISTRIBUTED_LOCAL
                else parity.central_local.result.to_json_bytes()
            )
            and exact_replay(repeated[index].result).to_json_bytes()
            == repeated[index].result.to_json_bytes()
            for index, topology in enumerate(ControlTopology)
        }
        ledger_checks = ledger_identity(parity.distributed.result.summary["ledger"])
        rows.append(
            {
                "scenarioId": scenario.scenario_id,
                "profile": profile,
                "policy": policy.value,
                "direction": direction.value,
                "success": parity.success,
                **parity.comparisons,
                "stopReason": parity.distributed.result.summary["stopReason"],
                "eventCount": len(parity.distributed.result.events),
                "eventDigest": parity.distributed.result.event_digest,
            }
        )
        replay_rows.append(
            {
                "scenarioId": scenario.scenario_id,
                "checks": replay_checks,
                "success": all(replay_checks.values()),
            }
        )
        ledger_rows.append(
            {
                "scenarioId": scenario.scenario_id,
                "checks": ledger_checks,
                "success": all(ledger_checks.values()),
                "ledger": dict(parity.distributed.result.summary["ledger"]),
            }
        )
    replay = {
        "schema": "E02.S02.deterministic-replay-validation.v1",
        "success": all(row["success"] for row in replay_rows),
        "scenarioCount": len(replay_rows),
        "results": replay_rows,
    }
    ledger = {
        "schema": "E02.S02.cost-ledger-validation.v1",
        "success": all(row["success"] for row in ledger_rows),
        "scenarioCount": len(ledger_rows),
        "identities": [
            "activations == proposals",
            "proposals == noOps + rejections + memoryUpdates + acceptedSwaps + conflictLosses",
            "displacedCells == 2 * acceptedSwaps",
        ],
        "results": ledger_rows,
    }
    return rows, replay, ledger


def validate_fault_diagnostics() -> dict[str, Any]:
    rows = []
    for policy, fault_mode in product(tuple(Policy), (FaultMode.PASSIVE, FaultMode.STUCK)):
        scenario = create_scenario(
            [4, 1, 3, 2],
            policy=policy,
            faults={1: fault_mode},
            seed=731,
            max_activations=150,
            generation_key=f"E02-S02/fault-diagnostic/{policy.value}/{fault_mode.value}",
            permute=False,
        )
        runs = [
            run_with_contract(
                scenario, MatchedExecutionContract(topology), trace_mode="full"
            )
            for topology in ControlTopology
        ]
        checks = {
            "transitionBytes": runs[0].transition_bytes() == runs[1].transition_bytes(),
            "resultBytes": runs[0].result.to_json_bytes() == runs[1].result.to_json_bytes(),
            "ledgerIdentity": all(ledger_identity(runs[0].result.summary["ledger"]).values()),
        }
        rows.append(
            {
                "scenarioId": scenario.scenario_id,
                "policy": policy.value,
                "faultMode": fault_mode.value,
                "success": all(checks.values()),
                "checks": checks,
                "stopReason": runs[0].result.summary["stopReason"],
            }
        )
    return {
        "schema": "E02.S02.fault-diagnostic-parity.v1",
        "scope": "diagnostic only; the E02-S01-E02 primary gate is no-fault",
        "success": all(row["success"] for row in rows),
        "scenarioCount": len(rows),
        "results": rows,
    }


def validate_archived_e01() -> dict[str, Any]:
    if not ARCHIVED_E01_SMOKE.exists():
        return {
            "schema": "E02.S02.e01-regression-validation.v1",
            "success": False,
            "status": "blocked_missing_archived_smoke",
            "path": str(ARCHIVED_E01_SMOKE),
        }
    source_bytes = ARCHIVED_E01_SMOKE.read_bytes()
    archived = RunResult.from_json_bytes(source_bytes)
    current = run(archived.scenario, trace_mode=archived.summary["traceMode"])
    behavior_equal = current.to_json_bytes() == archived.to_json_bytes()
    return {
        "schema": "E02.S02.e01-regression-validation.v1",
        "success": behavior_equal,
        "status": "pass" if behavior_equal else "mismatch",
        "path": str(ARCHIVED_E01_SMOKE),
        "inputSha256": hashlib.sha256(source_bytes).hexdigest(),
        "sourceFileUsesCanonicalBytesWithoutTrailingWhitespace": source_bytes
        == archived.to_json_bytes(),
        "canonicalResultBytesEqual": behavior_equal,
        "scenarioId": archived.scenario.scenario_id,
        "eventCount": len(archived.events),
        "eventDigest": archived.event_digest,
    }


def write_parity_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    fixture_results = validate_fixtures()
    primitive_results = validate_exhaustive_primitives()
    forbidden_results = validate_forbidden_information()
    parity_rows, replay_results, ledger_results = validate_parity_matrix()
    fault_results = validate_fault_diagnostics()
    e01_results = validate_archived_e01()

    write_parity_csv(args.output / "parity_matrix.csv", parity_rows)
    canonical_write(args.output / "differential_fixture_results.json", fixture_results)
    canonical_write(args.output / "primitive_parity_validation.json", primitive_results)
    canonical_write(args.output / "forbidden_information_validation.json", forbidden_results)
    canonical_write(args.output / "deterministic_replay_validation.json", replay_results)
    canonical_write(args.output / "ledger_validation.json", ledger_results)
    canonical_write(args.output / "fault_diagnostic_validation.json", fault_results)
    canonical_write(args.output / "e01_regression_validation.json", e01_results)

    checks = {
        "e02S01E02NoFaultExactParity": all(row["success"] for row in parity_rows),
        "primitiveParity": primitive_results["success"],
        "differentialFixtures": fixture_results["success"],
        "forbiddenInformation": forbidden_results["success"],
        "costLedgerIdentity": ledger_results["success"],
        "deterministicReplay": replay_results["success"],
        "faultDiagnostics": fault_results["success"],
        "archivedE01Regression": e01_results["success"],
    }
    summary = {
        "schema": "E02.S02.validation-summary.v1",
        "researchStepId": "S02",
        "stepNumber": 2,
        "success": all(checks.values()),
        "status": "complete" if all(checks.values()) else "failed",
        "outcomeClassification": "supportive" if all(checks.values()) else "constraining/contradictory",
        "estimandId": "E02-S01-E02",
        "checks": checks,
        "counts": {
            "noFaultParityScenarios": len(parity_rows),
            "noFaultParityArms": 2 * len(parity_rows),
            "primitiveScenarios": primitive_results["scenarioCount"],
            "primitiveOpportunities": primitive_results["opportunityCount"],
            "handCheckedFixtures": fixture_results["fixtureCount"],
            "forbiddenReadProbes": forbidden_results["probeCount"],
            "faultDiagnosticScenarios": fault_results["scenarioCount"],
        },
        "escalationTriggerReached": not all(row["success"] for row in parity_rows),
        "caveatsOrBlockers": [],
        "recommendedNextAction": (
            "Return control to the Chief Scientist; S03 may be authorized separately."
            if all(checks.values())
            else "Stop for design review; do not start S03."
        ),
    }
    canonical_write(args.output / "validation_summary.json", summary)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
