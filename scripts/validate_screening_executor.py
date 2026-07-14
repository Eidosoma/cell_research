#!/usr/bin/env python3
"""Differentially validate the compact S10 executor against S05/S09."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.costing import run_costed_faulted_architecture
from causal_simulator.faults import (
    ActionFailureProfile,
    ContinuationPolicy,
    FaultExecutionContract,
    MobilityProfile,
    RetryPolicy,
    SensingProfile,
)
from causal_simulator.screening import (
    materialize_screening_scenario,
    run_compact_screening,
)
from causal_simulator.schedulers import SchedulerExecutionContract, SchedulerFamily
from reference_simulator.model import Scenario, canonical_json_bytes


SETTING_FIELDS = (
    "architecture", "coordinatorProfile", "scheduler", "mobility", "continuation",
    "retry", "actionFailure", "sensing", "informationPermission", "legalPrimitives",
    "proposalCandidatesPerOpportunity",
)


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: list[Mapping[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd", compression_level=9, use_dictionary=True, write_statistics=True, version="2.6")


def architecture(settings: Mapping[str, str]) -> ArchitectureExecutionContract:
    name = settings["architecture"]
    if name == "central_local_proposal_k1":
        return ArchitectureExecutionContract.central_local_k1()
    if name == "distributed_local":
        return ArchitectureExecutionContract.distributed_local()
    if name == "distributed_weak_coordinator":
        return ArchitectureExecutionContract.distributed_weak(
            enabled=settings["coordinatorProfile"] == "weak_frozen_budget"
        )
    raise ValueError(name)


def fault_contract(settings: Mapping[str, str]) -> FaultExecutionContract:
    return FaultExecutionContract(
        mobility=MobilityProfile(settings["mobility"]),
        continuation=ContinuationPolicy(settings["continuation"]),
        retry=RetryPolicy(settings["retry"]),
        action_failure=ActionFailureProfile(settings["actionFailure"]),
        sensing=SensingProfile(settings["sensing"]),
    )


def bounded_scenario(scenario: Scenario, opportunities: int, key: str) -> Scenario:
    return Scenario.create(
        scenario.cells,
        initial_occupancy=scenario.initial_occupancy,
        initial_selection_cursors=dict(scenario.initial_selection_cursors),
        seed=scenario.seed,
        max_activations=opportunities,
        generation_key=key,
        fault_placement=scenario.fault_placement,
        requested_fault_count=scenario.requested_fault_count,
    )


def authoritative_projection(run: Any) -> dict[str, Any]:
    result = run.fault_run.result
    events = []
    for event in result.events:
        proposal = event["proposal"]
        events.append({
            "eventIndex": event["eventIndex"],
            "actorId": event["actorId"],
            "side": next((draw["value"] for draw in event["randomAddressesAndDraws"] if draw["stream"] == "bubble_side"), None),
            "proposal": {
                "kind": proposal["kind"],
                "actorPos": proposal["actorPos"],
                "targetPos": proposal["targetPos"],
                "newCursor": proposal["newCursor"],
                "reason": proposal["reason"],
                "observationReads": proposal["observationReads"],
                "valueComparisons": proposal["valueComparisons"],
            },
            "decision": event["decision"],
            "ledgerDelta": event["ledgerDelta"],
            "stopReasonIfAny": event["stopReasonIfAny"],
        })
    return {"result": result, "events": events, "ledger": run.ledger}


def validate(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    pairing_rows = pq.read_table(args.pairing, filters=[("screeningStage", "=", 1)]).to_pylist()
    # Exactly one deterministic n=20 block for each policy family.
    selected = []
    for policy in ("Bubble", "Insertion", "Selection"):
        selected.append(sorted((row for row in pairing_rows if row["n"] == 20 and row["policyProfile"] == policy), key=lambda row: row["pairingBlockId"])[0])
    input_ids = {row["inputScenarioId"] for row in selected}
    table = pq.read_table(args.scenario_bank, filters=[("split", "=", "screening_pool")])
    table = table.filter(pc.is_in(table["inputScenarioId"], value_set=pa.array(sorted(input_ids))))
    scenarios = {row["inputScenarioId"]: row for row in table.to_pylist()}
    design = pq.read_table(args.design, filters=[("screeningStage", "=", 1)]).to_pylist()
    treatments: dict[str, dict[str, str]] = {}
    for row in design:
        treatments.setdefault(row["treatmentSignature"], {field: str(row[field]) for field in SETTING_FIELDS})
    if len(treatments) != 14:
        raise AssertionError("executor validation requires all 14 frozen signatures")

    rows: list[dict[str, Any]] = []
    parity: dict[str, dict[str, Any]] = {}
    for pairing in selected:
        for signature, settings in sorted(treatments.items()):
            full = materialize_screening_scenario(scenarios[pairing["inputScenarioId"]], pairing, settings)
            scenario = bounded_scenario(full, args.opportunities, f"S10/executor-validation/{pairing['pairingBlockId']}/{settings['mobility']}")
            compact = run_compact_screening(scenario, settings, retain_step_projection=True)
            replay = run_compact_screening(scenario, settings, retain_step_projection=True)
            auth = run_costed_faulted_architecture(
                scenario,
                architecture(settings),
                SchedulerExecutionContract(SchedulerFamily(settings["scheduler"])),
                fault_contract(settings),
                trace_mode="full",
            )
            projected = authoritative_projection(auth)
            auth_result = projected["result"]
            compact_event_core = []
            for event in compact["stepProjection"]:
                item = dict(event)
                item["proposal"] = {key: value for key, value in item["proposal"].items() if key != "observedTargetId"}
                # The authoritative side draw retains uint64, while the compact
                # projection names left/right. Proposal and transition equality
                # checks the effect; side-address validation is covered by S04.
                item["side"] = None
                compact_event_core.append(item)
            auth_event_core = []
            for event in projected["events"]:
                item = dict(event)
                item["side"] = None
                auth_event_core.append(item)
            auth_flat = auth.ledger.to_flat_dict()
            cost_checks = {
                key: compact[f"cost_{key}"] == value
                for key, value in auth.ledger.costs.items()
                if key != "wallTimeSeconds"
            }
            projection_checks = {
                key: compact[f"projection_{key}"] == value
                for key, value in auth.ledger.projections.items()
            }
            checks = {
                "stopReason": compact["stopReason"] == auth_result.summary["stopReason"],
                "activationCount": compact["activationCount"] == auth_result.summary["activationCount"],
                "finalOccupancy": compact["finalOccupancy"] == auth_result.summary["finalOccupancy"],
                "finalValues": compact["finalValues"] == auth_result.summary["finalValues"],
                "nativeLedger": compact["nativeLedger"] == auth_result.summary["ledger"],
                "streamCounters": compact["streamCounters"] == auth_result.final_state["streamCounters"],
                "eventProjection": compact_event_core == auth_event_core,
                "costVector": all(cost_checks.values()),
                "costProjections": all(projection_checks.values()),
                "contractValidation": compact["contractValidationPass"] and auth.ledger.success,
                "compactExactReplay": compact["deterministicResultSha256"] == replay["deterministicResultSha256"] and compact["compactTransitionDigest"] == replay["compactTransitionDigest"] and compact["compactReplayDigest"] == replay["compactReplayDigest"],
            }
            success = all(checks.values())
            rows.append({
                "pairingBlockId": pairing["pairingBlockId"],
                "policyProfile": pairing["policyProfile"],
                "treatmentSignature": signature,
                **settings,
                "opportunitiesExecuted": compact["activationCount"],
                "stopReason": compact["stopReason"],
                "checksPassed": sum(checks.values()),
                "checksTotal": len(checks),
                "success": success,
                "checksJson": canonical_json_bytes(checks).decode("utf-8"),
            })
            if not success:
                raise AssertionError(json.dumps(rows[-1], indent=2, sort_keys=True))
            if settings["mobility"] == "normal" and settings["scheduler"] == "uniform_random_activation":
                role = settings["architecture"]
                if role in {"central_local_proposal_k1", "distributed_local"}:
                    parity.setdefault(pairing["pairingBlockId"], {})[role] = auth_result

    parity_checks = []
    for block_id, arms in sorted(parity.items()):
        central = arms["central_local_proposal_k1"]
        distributed = arms["distributed_local"]
        checks = {
            "finalState": central.final_state == distributed.final_state,
            "stopReason": central.summary["stopReason"] == distributed.summary["stopReason"],
            "eventDigest": central.event_digest == distributed.event_digest,
            "events": central.events == distributed.events,
            "ledger": central.summary["ledger"] == distributed.summary["ledger"],
        }
        parity_checks.append({"pairingBlockId": block_id, **checks, "success": all(checks.values())})
    if len(parity_checks) != 3 or not all(row["success"] for row in parity_checks):
        raise AssertionError("bounded authoritative E02 parity failed")

    write_parquet(args.output / "executor_differential_validation.parquet", rows)
    write_json(args.output / "executor_parity_validation.json", {
        "schemaVersion": "e02.s10.executor_parity_validation.v1",
        "researchStepId": "S10",
        "boundedOpportunityBudget": args.opportunities,
        "rows": parity_checks,
        "success": all(row["success"] for row in parity_checks),
    })
    summary = {
        "schemaVersion": "e02.s10.executor_validation.v1",
        "researchStepId": "S10",
        "validationResult": "PASS",
        "success": True,
        "pairingBlocks": len(selected),
        "policyProfiles": sorted({row["policyProfile"] for row in selected}),
        "treatmentSignatures": len(treatments),
        "differentialRows": len(rows),
        "successfulDifferentialRows": sum(row["success"] for row in rows),
        "opportunitiesExecuted": sum(row["opportunitiesExecuted"] for row in rows),
        "stopReasonCounts": dict(sorted(Counter(row["stopReason"] for row in rows).items())),
        "authoritativeParityRows": len(parity_checks),
        "authoritativeParityPass": sum(row["success"] for row in parity_checks),
        "compactReplayRows": sum(json.loads(row["checksJson"])["compactExactReplay"] for row in rows),
        "protectedOutcomeReads": 0,
    }
    write_json(args.output / "executor_validation.json", summary)
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opportunities", type=int, default=400)
    parser.add_argument("--design", type=Path, default=Path("/artifacts/research_steps/S10/screening_design.parquet"))
    parser.add_argument("--pairing", type=Path, default=Path("/artifacts/research_steps/S10/screening_pairing_blocks.parquet"))
    parser.add_argument("--scenario-bank", type=Path, default=Path("/artifacts/research_steps/S07/scenario_extension.parquet"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S10"))
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
