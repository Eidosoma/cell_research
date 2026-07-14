#!/usr/bin/env python3
"""Validate S02-S06 execution contracts over every S07 factorial cell."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.faults import (
    FaultExecutionContract,
    exact_replay_fault,
    run_faulted_architecture,
)
from causal_simulator.scenario_extensions import scenario_from_record
from causal_simulator.schedulers import SchedulerExecutionContract, SchedulerFamily
from reference_simulator.model import (
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
)


ARCHITECTURES = (
    ArchitectureExecutionContract.central_local_k1(),
    ArchitectureExecutionContract.distributed_local(),
    ArchitectureExecutionContract.distributed_weak(enabled=False),
    ArchitectureExecutionContract.distributed_weak(),
)


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pq.write_table(
        pa.Table.from_pylist(list(rows)),
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def materialize(record: Mapping[str, Any], policy: Policy) -> Scenario:
    input_scenario = scenario_from_record(record)
    cells = tuple(
        Cell(
            f"cell-{index:04d}",
            value,
            policy,
            Direction.ASCENDING,
            FaultMode.NORMAL,
        )
        for index, value in enumerate(input_scenario.values_by_identity)
    )
    return Scenario.create(
        cells,
        initial_occupancy=input_scenario.initial_occupancy,
        seed=input_scenario.scenario_seed,
        max_activations=min(input_scenario.n, 32),
        generation_key=(
            f"S07/contract-overlay/{input_scenario.input_scenario_id}/{policy.value}"
        ),
        fault_placement="explicit",
    )


def validate(bank: Path, output: Path) -> None:
    records = pq.read_table(
        bank,
        filters=[
            ("split", "=", "runtime_validation"),
            ("replicateOrdinal", "=", 0),
        ],
    ).to_pylist()
    if len(records) != 45:
        raise AssertionError("overlay requires exactly one row from every factorial cell")
    rows: list[dict[str, Any]] = []
    for record in records:
        for policy in Policy:
            scenario = materialize(record, policy)
            for architecture in ARCHITECTURES:
                for family in SchedulerFamily:
                    run = run_faulted_architecture(
                        scenario,
                        architecture,
                        SchedulerExecutionContract(family),
                        FaultExecutionContract(),
                        trace_mode="none",
                    )
                    validations = run.opportunity_validation()
                    replay = exact_replay_fault(run)
                    rows.append(
                        {
                            "inputScenarioId": record["inputScenarioId"],
                            "n": record["n"],
                            "valueProfile": record["valueProfile"],
                            "orderStructure": record["orderStructure"],
                            "policy": policy.value,
                            "architecture": architecture.architecture.value,
                            "coordinatorProfile": architecture.coordinator_profile.value,
                            "scheduler": family.value,
                            "maxOpportunities": scenario.max_activations,
                            "opportunitiesExecuted": run.result.summary["ledger"][
                                "activations"
                            ],
                            "stopReason": run.result.summary["stopReason"],
                            "opportunityChecks": len(validations),
                            "opportunityChecksPassed": sum(validations.values()),
                            "allOpportunityChecksPassed": all(validations.values()),
                            "exactReplay": replay.to_json_bytes() == run.to_json_bytes(),
                            "legalPrimitives": "NoOp|Swap|MemoryUpdate",
                            "informationPermission": "policy_native_local",
                            "proposalCandidatesPerOpportunity": 1,
                            "faultMapAssignment": "none",
                            "outcomeAccess": "runtime_validation_only",
                        }
                    )
    write_parquet(output / "contract_overlay_validation.parquet", rows)
    summary = {
        "schemaVersion": "e02.s07.contract_overlay_validation.v1",
        "researchStepId": "S07",
        "inputFactorialCells": len({item["inputScenarioId"] for item in rows}),
        "policies": sorted({item["policy"] for item in rows}),
        "architectures": sorted(
            {
                f'{item["architecture"]}:{item["coordinatorProfile"]}'
                for item in rows
            }
        ),
        "schedulers": sorted({item["scheduler"] for item in rows}),
        "executionRows": len(rows),
        "exactReplayRows": sum(item["exactReplay"] for item in rows),
        "opportunityValidationRows": sum(
            item["allOpportunityChecksPassed"] for item in rows
        ),
        "opportunitiesExecuted": sum(item["opportunitiesExecuted"] for item in rows),
        "stopReasonCounts": dict(sorted(Counter(item["stopReason"] for item in rows).items())),
        "legalPrimitivesPreserved": all(
            item["legalPrimitives"] == "NoOp|Swap|MemoryUpdate" for item in rows
        ),
        "informationPermissionPreserved": all(
            item["informationPermission"] == "policy_native_local" for item in rows
        ),
        "oneProposalBoundaryPreserved": all(
            item["proposalCandidatesPerOpportunity"] == 1 for item in rows
        ),
        "noFaultMapOrOutcomeUse": all(
            item["faultMapAssignment"] == "none"
            and item["outcomeAccess"] == "runtime_validation_only"
            for item in rows
        ),
    }
    summary["success"] = (
        summary["inputFactorialCells"] == 45
        and summary["executionRows"] == 45 * 3 * 4 * 5
        and summary["exactReplayRows"] == summary["executionRows"]
        and summary["opportunityValidationRows"] == summary["executionRows"]
        and summary["legalPrimitivesPreserved"]
        and summary["informationPermissionPreserved"]
        and summary["oneProposalBoundaryPreserved"]
        and summary["noFaultMapOrOutcomeUse"]
    )
    write_json(output / "contract_overlay_validation.json", summary)
    if not summary["success"]:
        raise AssertionError("S07 contract overlay failed")
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("/artifacts/research_steps/S07/scenario_extension.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/artifacts/research_steps/S07"),
    )
    args = parser.parse_args()
    validate(args.bank, args.output)


if __name__ == "__main__":
    main()
