#!/usr/bin/env python3
"""Build the compact S09 complete-ledger validation and benchmark package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import pandas as pd

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.costing import (
    ALIAS_FIELDS,
    COST_SCHEMA_SHA256,
    COST_SCHEMA_VERSION,
    S01_ADDITIVE_FIELDS,
    SCHEDULER_OVERHEAD_FIELDS,
    build_complete_cost_ledger,
    exact_replay_costed,
    run_costed_faulted_architecture,
    validate_cost_schema,
)
from causal_simulator.faults import (
    ActionFailureProfile,
    ContinuationPolicy,
    FaultExecutionContract,
    RetryPolicy,
    SensingProfile,
    run_faulted_architecture,
)
from causal_simulator.schedulers import SchedulerExecutionContract, SchedulerFamily
from reference_simulator.api import create_scenario
from reference_simulator.model import Policy, canonical_json_bytes


ROOT = Path(__file__).parents[1]
WORKSPACE = ROOT.parent
SCHEMA_PATH = ROOT / "design" / "s09" / "cost_schema.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "s09_cost_ledger_fixtures.json"
S07_BANK = Path("/artifacts/research_steps/S07/scenario_extension.parquet")
S08_ROOT = Path("/artifacts/research_steps/S08")

ARCHITECTURES = {
    "central_local_k1": ArchitectureExecutionContract.central_local_k1,
    "distributed_local": ArchitectureExecutionContract.distributed_local,
    "distributed_weak_disabled": lambda: ArchitectureExecutionContract.distributed_weak(
        enabled=False
    ),
    "distributed_weak_active": ArchitectureExecutionContract.distributed_weak,
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fault(spec: dict[str, str]) -> FaultExecutionContract:
    return FaultExecutionContract(
        continuation=ContinuationPolicy(
            spec.get("continuation", ContinuationPolicy.SKIP_AND_CONTINUE.value)
        ),
        retry=RetryPolicy(spec.get("retry", RetryPolicy.NO_RETRY.value)),
        action_failure=ActionFailureProfile(
            spec.get("actionFailure", ActionFailureProfile.NONE.value)
        ),
        sensing=SensingProfile(spec.get("sensing", SensingProfile.EXACT.value)),
    )


def _scenario_from_fixture(item: dict[str, Any]):
    return create_scenario(
        item["values"],
        policy=Policy(item["policy"]),
        seed=item["seed"],
        max_activations=item["maxActivations"],
        generation_key=item["generationKey"],
        permute=False,
    )


def _validate_fixtures(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in _json(FIXTURE_PATH)["fixtures"]:
        run = run_faulted_architecture(
            _scenario_from_fixture(item),
            ARCHITECTURES[item["architecture"]](),
            SchedulerExecutionContract(SchedulerFamily(item["scheduler"])),
            _fault(item["fault"]),
            trace_mode="full",
        )
        ledger = build_complete_cost_ledger(run)
        combined = {**ledger.costs, **ledger.projections}
        comparisons: list[dict[str, Any]] = []
        if item.get("expectedAllAlgorithmicZero"):
            for name in S01_ADDITIVE_FIELDS:
                comparisons.append(
                    {"field": name, "expected": 0, "observed": combined[name], "passed": combined[name] == 0}
                )
            for name, value in ledger.projections.items():
                comparisons.append(
                    {"field": name, "expected": 0, "observed": value, "passed": value == 0}
                )
        else:
            for name, expected in item["expectedCosts"].items():
                comparisons.append(
                    {"field": name, "expected": expected, "observed": combined[name], "passed": combined[name] == expected}
                )
        passed = ledger.success and all(item["passed"] for item in comparisons)
        if not passed:
            failures.append(item["name"])
        records.append(
            {
                "name": item["name"],
                "scenarioId": run.result.scenario.scenario_id,
                "eventCount": len(run.result.events),
                "stopReason": run.result.summary["stopReason"],
                "expectedComparisons": comparisons,
                "completeCostLedger": ledger.to_dict(),
                "passed": passed,
            }
        )
    document = {
        "schemaVersion": "e02.s09.hand_counted_trace_validation.v1",
        "researchStepId": "S09",
        "fixtureCount": len(records),
        "passedCount": sum(item["passed"] for item in records),
        "failedFixtures": failures,
        "fixtures": records,
        "success": not failures,
    }
    _write_json(output / "ledger_fixtures.json", document)
    zero = next(item for item in records if item["name"] == "zero_opportunity_control")
    zero_document = {
        "schemaVersion": "e02.s09.zero_cost_validation.v1",
        "researchStepId": "S09",
        "fixture": zero["name"],
        "algorithmicFieldsChecked": list(S01_ADDITIVE_FIELDS),
        "normalizationAtZeroOpportunities": zero["completeCostLedger"]["normalizations"]["s01UnitWeightPerOpportunity"],
        "passed": zero["passed"],
        "success": zero["passed"],
    }
    _write_json(output / "zero_cost_validation.json", zero_document)
    return records, zero_document


def _runtime_inputs() -> pd.DataFrame:
    bank = pd.read_parquet(S07_BANK)
    selected = bank[
        (bank["split"] == "runtime_validation")
        & (bank["replicateOrdinal"] == 0)
    ].copy()
    selected = selected.sort_values(["n", "valueProfile", "orderStructure"])
    expected = {
        (n, value, order)
        for n in (20, 50, 100, 200, 500)
        for value in ("unique", "balanced_duplicate", "uneven_duplicate")
        for order in ("nearly_sorted", "reverse", "block_scrambled")
    }
    observed = set(
        zip(selected["n"], selected["valueProfile"], selected["orderStructure"])
    )
    if observed != expected or len(selected) != 45:
        raise AssertionError("S07 runtime-validation coverage is incomplete")
    if selected["protected"].any() or set(selected["outcomeAccess"]) != {"none"}:
        raise AssertionError("S09 runtime inputs must be unprotected and outcome-blind")
    return selected


def _runtime_scenario(row: Any):
    n = int(row.n)
    return create_scenario(
        [value.item() if hasattr(value, "item") else value for value in row.initialValues],
        policy=Policy.BUBBLE,
        seed=int(row.scenarioSeed),
        # One complete fairness warmup at every scale. S04 already validates
        # long-horizon fairness, while S09's separate n=4 mechanism matrix
        # exercises scored adversarial decisions. At n=500, repeating the
        # bounded-lookahead feasibility search would benchmark an intentionally
        # clear Python implementation rather than improve ledger validation.
        max_activations=n,
        generation_key=f"S09/runtime-validation/{row.inputScenarioId}",
        permute=False,
    )


def _benchmark(output: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    comparable_groups: list[dict[str, Any]] = []
    replay_records: list[dict[str, Any]] = []
    replayed_sizes: set[int] = set()
    for row in _runtime_inputs().itertuples(index=False):
        scenario = _runtime_scenario(row)
        for family in SchedulerFamily:
            costed_by_architecture = {}
            for name, factory in ARCHITECTURES.items():
                costed = run_costed_faulted_architecture(
                    scenario,
                    factory(),
                    SchedulerExecutionContract(family),
                    FaultExecutionContract(),
                    trace_mode="none",
                )
                costed_by_architecture[name] = costed
                records.append(
                    {
                        "benchmarkId": f"{row.inputScenarioId}/{family.value}/{name}",
                        "inputScenarioId": row.inputScenarioId,
                        "split": row.split,
                        "protected": bool(row.protected),
                        "outcomeAccess": row.outcomeAccess,
                        "valueProfile": row.valueProfile,
                        "orderStructure": row.orderStructure,
                        "replicateOrdinal": int(row.replicateOrdinal),
                        "eventBudget": scenario.max_activations,
                        "stopReason": costed.fault_run.result.summary["stopReason"],
                        **costed.ledger.to_flat_dict(),
                    }
                )
            native = {
                name: value.fault_run.result.to_json_bytes()
                for name, value in costed_by_architecture.items()
            }
            exact_native = len(set(native.values())) == 1
            baseline = costed_by_architecture["distributed_local"].ledger.costs
            noncoordinator = set(baseline) - {
                "coordinatorMessages",
                "coordinatorMessageBits",
                "coordinatorClockChecks",
                "coordinatorCandidateEvaluations",
                "coordinatorEligibleDecisions",
                "coordinatorInterventions",
                "wallTimeSeconds",
            }
            base_equal = all(
                all(
                    value.ledger.costs[field] == baseline[field]
                    for field in noncoordinator
                )
                for value in costed_by_architecture.values()
            )
            weak = costed_by_architecture["distributed_weak_active"].ledger.costs
            weak_identity = (
                weak["coordinatorMessages"] == 2 * weak["coordinatorEligibleDecisions"]
                and weak["coordinatorClockChecks"] == weak["activations"]
            )
            comparable_groups.append(
                {
                    "inputScenarioId": row.inputScenarioId,
                    "n": int(row.n),
                    "scheduler": family.value,
                    "architectures": list(ARCHITECTURES),
                    "nativeResultBytesExact": exact_native,
                    "nonCoordinatorCostsExact": base_equal,
                    "weakSupplementIdentity": weak_identity,
                    "passed": exact_native and base_equal and weak_identity,
                }
            )
            if int(row.n) not in replayed_sizes and family == SchedulerFamily.UNIFORM_RANDOM_ACTIVATION:
                for name, costed in costed_by_architecture.items():
                    replayed = exact_replay_costed(costed)
                    replay_records.append(
                        {
                            "inputScenarioId": row.inputScenarioId,
                            "n": int(row.n),
                            "architecture": name,
                            "scheduler": family.value,
                            "faultBytesExact": replayed.fault_run.to_json_bytes()
                            == costed.fault_run.to_json_bytes(),
                            "deterministicCostBytesExact": replayed.ledger.deterministic_bytes()
                            == costed.ledger.deterministic_bytes(),
                            "wallTimeExcluded": replayed.ledger.costs["wallTimeSeconds"] == 0,
                        }
                    )
                replayed_sizes.add(int(row.n))

    benchmark = pd.DataFrame(records).sort_values("benchmarkId").reset_index(drop=True)
    benchmark.to_parquet(output / "cost_benchmark.parquet", index=False)
    benchmark_summary = (
        benchmark.groupby(["n", "architecture", "scheduler"], as_index=False)
        .agg(
            runs=("benchmarkId", "count"),
            meanActivations=("activations", "mean"),
            meanS01Cost=("s01UnitWeightFullCost", "mean"),
            meanS01PerOpportunity=("s01UnitWeightPerOpportunity", "mean"),
            meanSchedulerRngBlocks=("schedulerActorSelectionRngBlocks", "mean"),
            meanSchedulerIdentitiesScored=("schedulerIdentitiesScored", "mean"),
            medianWallTimeSeconds=("wallTimeSeconds", "median"),
            maxWallTimeSeconds=("wallTimeSeconds", "max"),
        )
        .sort_values(["n", "architecture", "scheduler"])
    )
    benchmark_summary.to_csv(output / "benchmark_summary.csv", index=False)

    architecture_document = {
        "schemaVersion": "e02.s09.architecture_comparability_validation.v1",
        "researchStepId": "S09",
        "groupCount": len(comparable_groups),
        "passedCount": sum(item["passed"] for item in comparable_groups),
        "groups": comparable_groups,
        "legacyGlobalExcluded": True,
        "success": all(item["passed"] for item in comparable_groups),
    }
    _write_json(output / "architecture_comparability_validation.json", architecture_document)
    replay_document = {
        "schemaVersion": "e02.s09.deterministic_replay_validation.v1",
        "researchStepId": "S09",
        "replayCount": len(replay_records),
        "sizes": sorted(replayed_sizes),
        "records": replay_records,
        "wallTimeComparisonRule": "excluded_from_deterministic_bytes",
        "success": all(
            item["faultBytesExact"]
            and item["deterministicCostBytesExact"]
            and item["wallTimeExcluded"]
            for item in replay_records
        ),
    }
    _write_json(output / "deterministic_replay_validation.json", replay_document)
    return benchmark, architecture_document, replay_document


def _mechanism_matrix(output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    profiles = {
        "exact": FaultExecutionContract(),
        "noisy": FaultExecutionContract(sensing=SensingProfile.NOISY_VALUE_OR_STATUS),
        "bernoulli_retry": FaultExecutionContract(
            action_failure=ActionFailureProfile.BERNOULLI_P,
            retry=RetryPolicy.RETRY_LATER_BOUNDED,
        ),
    }
    records: list[dict[str, Any]] = []
    for family in SchedulerFamily:
        for architecture_name, architecture_factory in ARCHITECTURES.items():
            for profile_name, fault in profiles.items():
                scenario = create_scenario(
                    [4, 1, 3, 2],
                    policy=Policy.BUBBLE,
                    seed=7,
                    max_activations=32,
                    generation_key=(
                        f"S09/mechanism/{family.value}/{profile_name}"
                    ),
                    permute=False,
                )
                costed = run_costed_faulted_architecture(
                    scenario,
                    architecture_factory(),
                    SchedulerExecutionContract(family),
                    fault,
                    trace_mode="digest",
                )
                records.append(
                    {
                        "mechanismId": f"{family.value}/{architecture_name}/{profile_name}",
                        "profile": profile_name,
                        **costed.ledger.to_flat_dict(),
                    }
                )
    table = pd.DataFrame(records).sort_values("mechanismId").reset_index(drop=True)
    table.to_parquet(output / "mechanism_cost_validation.parquet", index=False)
    validation = {
        "schemaVersion": "e02.s09.mechanism_cost_validation.v1",
        "researchStepId": "S09",
        "rowCount": len(table),
        "architectureCount": table["architecture"].nunique(),
        "schedulerCount": table["scheduler"].nunique(),
        "faultProfileCount": table["profile"].nunique(),
        "allLedgersValid": bool(table["validationPass"].all()),
        "retryCandidateIdentity": bool(
            (
                table["targetCalculations"]
                == table["proposals"] - table["retryAttempts"]
            ).all()
        ),
        "sensingDrawsNotReads": bool(
            (
                table.loc[table["profile"] == "noisy", "sensingValueDraws"]
                == table.loc[table["profile"] == "noisy", "observationRecordReads"]
            ).all()
        ),
        "schedulerCandidatesRemainZero": bool(
            (table["schedulerCandidateInspections"] == 0).all()
        ),
    }
    validation["success"] = all(
        value
        for key, value in validation.items()
        if key in {
            "allLedgersValid",
            "retryCandidateIdentity",
            "sensingDrawsNotReads",
            "schedulerCandidatesRemainZero",
        }
    )
    _write_json(output / "mechanism_cost_validation.json", validation)
    return table, validation


def _accounting_validations(
    output: Path,
    fixtures: list[dict[str, Any]],
    benchmark: pd.DataFrame,
    mechanism: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_rows = pd.concat([benchmark, mechanism], ignore_index=True, sort=False)
    identity_checks = {
        "activationProposal": bool((all_rows["activations"] == all_rows["proposals"]).all()),
        "proposalPartition": bool(
            (
                all_rows["proposals"]
                == all_rows[["noOps", "rejections", "memoryUpdates", "acceptedSwaps", "conflictLosses"]].sum(axis=1)
            ).all()
        ),
        "swapDisplacement": bool(
            (all_rows["displacedCells"] == 2 * all_rows["acceptedSwaps"]).all()
        ),
        "readAliases": bool(
            (all_rows["valueReads"] == all_rows["observationRecordReads"]).all()
            and (all_rows["statusReads"] == all_rows["observationRecordReads"]).all()
        ),
        "candidateConstruction": bool(
            (
                all_rows["targetCalculations"]
                == all_rows["proposals"] - all_rows["retryAttempts"]
            ).all()
            and (
                all_rows["policyCandidateConstructions"]
                == all_rows["targetCalculations"]
            ).all()
        ),
        "failedProposal": bool(
            (
                all_rows["failedProposals"]
                == all_rows["rejections"] + all_rows["conflictLosses"]
            ).all()
        ),
        "actionFailureSubset": bool(
            (all_rows["actionFailures"] <= all_rows["rejections"]).all()
        ),
        "coordinatorMessages": bool(
            (
                all_rows["coordinatorMessages"]
                == 2 * all_rows["coordinatorEligibleDecisions"]
            ).all()
        ),
        "schedulerCandidatesZero": bool(
            (all_rows["schedulerCandidateInspections"] == 0).all()
        ),
        "rowValidation": bool(all_rows["validationPass"].all()),
    }
    identity_document = {
        "schemaVersion": "e02.s09.ledger_identity_validation.v1",
        "researchStepId": "S09",
        "rowCount": len(all_rows),
        "fixtureCount": len(fixtures),
        "checks": identity_checks,
        "success": all(identity_checks.values()) and all(item["passed"] for item in fixtures),
    }
    _write_json(output / "ledger_identity_validation.json", identity_document)

    count_fields = [
        name
        for name in benchmark.columns
        if name in set(S01_ADDITIVE_FIELDS)
        | set(ALIAS_FIELDS)
        | set(SCHEDULER_OVERHEAD_FIELDS)
        | {
            "valueReads", "statusReads", "failedProposals", "policyCandidateConstructions",
            "deferredRetryEnvelopeReuses", "coordinatorMessageBits", "coordinatorClockChecks",
            "coordinatorCandidateEvaluations", "sensingValueDraws", "sensingStatusDraws",
            "sensingValueErrors", "sensingStatusErrors", "sensingErrorsApplied",
            "sensingErrorHandlingOperations", "actionFailureDraws", "actionFailureExposures",
            "actionFailures", "blockingFailures", "continuationStops", "retryQueued",
            "retryAttempts", "retryExhausted", "retryQueueCollisions", "retryPendingAtStop",
            "faultRngDraws"
        }
    ]
    unit_checks = {
        "countFieldsInteger": all(
            pd.api.types.is_integer_dtype(all_rows[field]) for field in count_fields
        ),
        "countFieldsNonnegative": bool((all_rows[count_fields] >= 0).all().all()),
        "wallTimeFloatSeconds": pd.api.types.is_float_dtype(all_rows["wallTimeSeconds"]),
        "wallTimeNonnegative": bool((all_rows["wallTimeSeconds"] >= 0).all()),
        "normalizationsFloat": all(
            pd.api.types.is_float_dtype(all_rows[field])
            for field in (
                "s01UnitWeightPerOpportunity", "s01UnitWeightPerCell", "s01UnitWeightPerN2"
            )
        ),
    }
    units_document = {
        "schemaVersion": "e02.s09.units_validation.v1",
        "researchStepId": "S09",
        "fieldCount": len(count_fields),
        "countFields": count_fields,
        "checks": unit_checks,
        "success": all(unit_checks.values()),
    }
    _write_json(output / "units_validation.json", units_document)

    double_checks = {
        "aliasFieldsExcludedFromS01": not set(ALIAS_FIELDS).intersection(S01_ADDITIVE_FIELDS),
        "schedulerOverheadExcludedFromS01": not set(SCHEDULER_OVERHEAD_FIELDS).intersection(S01_ADDITIVE_FIELDS),
        "wallTimeExcludedFromS01": "wallTimeSeconds" not in S01_ADDITIVE_FIELDS,
        "candidateAliasNotAdditive": "policyCandidateConstructions" not in S01_ADDITIVE_FIELDS,
        "failedProposalAliasNotAdditive": "failedProposals" not in S01_ADDITIVE_FIELDS,
        "readFieldAliasesNotAdditive": not {"valueReads", "statusReads"}.intersection(S01_ADDITIVE_FIELDS),
        "s01ReprojectionExact": bool(
            (
                all_rows["s01UnitWeightFullCost"]
                == all_rows[list(S01_ADDITIVE_FIELDS)].sum(axis=1)
            ).all()
        ),
        "zeroProjectionExact": bool((all_rows["zeroCostControl"] == 0).all()),
    }
    double_document = {
        "schemaVersion": "e02.s09.double_counting_validation.v1",
        "researchStepId": "S09",
        "s01AdditiveFields": list(S01_ADDITIVE_FIELDS),
        "aliasFields": list(ALIAS_FIELDS),
        "schedulerOverheadFields": list(SCHEDULER_OVERHEAD_FIELDS),
        "checks": double_checks,
        "success": all(double_checks.values()),
    }
    _write_json(output / "double_counting_validation.json", double_document)
    return identity_document, units_document, double_document


def _cross_scale(output: Path, benchmark: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "sizesComplete": sorted(benchmark["n"].unique().tolist()) == [20, 50, 100, 200, 500],
        "structureCellsComplete": benchmark[["n", "valueProfile", "orderStructure"]].drop_duplicates().shape[0] == 45,
        "architectureSchedulerCellsComplete": benchmark[
            ["n", "architecture", "coordinatorProfile", "scheduler"]
        ].drop_duplicates().shape[0] == 100,
        "perCellExact": bool(
            ((benchmark["s01UnitWeightPerCell"] - benchmark["s01UnitWeightFullCost"] / benchmark["n"]).abs() < 1e-12).all()
        ),
        "perN2Exact": bool(
            ((benchmark["s01UnitWeightPerN2"] - benchmark["s01UnitWeightFullCost"] / (benchmark["n"] ** 2)).abs() < 1e-12).all()
        ),
        "perOpportunityExact": bool(
            ((benchmark["s01UnitWeightPerOpportunity"] - benchmark["s01UnitWeightFullCost"] / benchmark["activations"]).abs() < 1e-12).all()
        ),
        "rawCountsRetained": all(name in benchmark for name in S01_ADDITIVE_FIELDS),
    }
    by_size = (
        benchmark.groupby("n", as_index=False)
        .agg(
            runs=("benchmarkId", "count"),
            minActivations=("activations", "min"),
            maxActivations=("activations", "max"),
            meanS01PerOpportunity=("s01UnitWeightPerOpportunity", "mean"),
            minS01PerOpportunity=("s01UnitWeightPerOpportunity", "min"),
            maxS01PerOpportunity=("s01UnitWeightPerOpportunity", "max"),
        )
        .to_dict(orient="records")
    )
    document = {
        "schemaVersion": "e02.s09.cross_scale_normalization_validation.v1",
        "researchStepId": "S09",
        "checks": checks,
        "bySize": by_size,
        "success": all(checks.values()),
    }
    _write_json(output / "cross_scale_normalization_validation.json", document)
    return document


def _s08_preservation(output: Path) -> dict[str, Any]:
    manifest = _json(S08_ROOT / "artifact_manifest.json")
    expected = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    selected = (
        "pairing_manifest.parquet",
        "contrast_arm_catalog.json",
        "pairing_prespecification.json",
        "semantic_random_stream_specification.json",
        "treatment_arm_marginals.parquet",
    )
    checks = []
    for name in selected:
        observed = _sha256(S08_ROOT / name)
        checks.append(
            {
                "path": str(S08_ROOT / name),
                "expectedSha256": expected[name],
                "observedSha256": observed,
                "passed": expected[name] == observed,
            }
        )
    summary = _json(S08_ROOT / "validation_summary.json")
    document = {
        "schemaVersion": "e02.s09.contract_preservation_validation.v1",
        "researchStepId": "S09",
        "hashedS08Artifacts": checks,
        "pairingRows": summary["pairingRows"],
        "executableArms": summary["executableArms"],
        "protectedOutcomeReads": 0,
        "searchDerivedAssignments": 0,
        "legalPrimitives": ["NoOp", "Swap", "MemoryUpdate"],
        "informationPermission": "policy_native_local",
        "proposalCandidatesPerOpportunity": 1,
        "costProjectionIsPostRunReadOnly": True,
        "success": all(item["passed"] for item in checks),
    }
    _write_json(output / "contract_preservation_validation.json", document)
    return document


def _provenance(output: Path) -> None:
    inputs = [
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "FULL_PLAN.md",
        WORKSPACE / "RESEARCH_PLAN.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        WORKSPACE / "input-attachments" / "MANIFEST.json",
        WORKSPACE / "input-attachments" / "21c2278b-9950-4e39-a2c8-df578a2508ec" / "_metadata" / "ATTACHMENT.md",
        WORKSPACE / "input-attachments" / "21c2278b-9950-4e39-a2c8-df578a2508ec" / "pdf-markdown.md",
        Path("/previous-artifacts/E01/release/baseline/schemas.json"),
        Path("/previous-artifacts/E01/research_steps/S06/event_schema.json"),
        Path("/previous-artifacts/E01/research_steps/S06/schema_documentation.md"),
        Path("/previous-artifacts/E01/research_steps/S10/research_step_full_results.md"),
        S07_BANK,
        S08_ROOT / "artifact_manifest.json",
        SCHEMA_PATH,
        FIXTURE_PATH,
    ]
    inputs.extend(
        Path(f"/artifacts/research_steps/S{step:02d}/research_step_full_results.md")
        for step in range(1, 9)
    )
    records = []
    for path in inputs:
        records.append(
            {
                "path": str(path),
                "present": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path) if path.is_file() else None,
            }
        )
    _write_json(
        output / "input_provenance.json",
        {
            "schemaVersion": "e02.s09.input_provenance.v1",
            "researchStepId": "S09",
            "inputs": records,
            "datasetManifestsPresent": False,
            "capabilityManifestsPresent": False,
            "originalPdfMaterialized": False,
            "doclingMarkdownUsedAsContext": True,
        },
    )
    source_paths = [
        ROOT / "causal_simulator" / "costing.py",
        ROOT / "scripts" / "validate_cost_ledger.py",
        ROOT / "tests" / "test_costing.py",
        FIXTURE_PATH,
        SCHEMA_PATH,
    ]
    _write_json(
        output / "source_package_manifest.json",
        {
            "schemaVersion": "e02.s09.source_package_manifest.v1",
            "researchStepId": "S09",
            "repositoryRoot": str(ROOT),
            "branch": "eidosoma/groups/28",
            "files": [
                {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in source_paths
            ],
        },
    )
    _write_json(
        output / "environment_provenance.json",
        {
            "schemaVersion": "e02.s09.environment.v1",
            "researchStepId": "S09",
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workers": 1,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
            "dependencies": {
                "pandas": pd.__version__,
            },
            "newDependenciesInstalled": [],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    schema_check = validate_cost_schema(SCHEMA_PATH)
    if not schema_check["success"]:
        raise AssertionError("frozen cost schema hash mismatch")
    shutil.copyfile(SCHEMA_PATH, output / "cost_schema.json")
    _write_json(output / "schema_validation.json", {"researchStepId": "S09", **schema_check})

    fixtures, zero = _validate_fixtures(output)
    benchmark, architectures, replay = _benchmark(output)
    mechanism, mechanism_validation = _mechanism_matrix(output)
    identities, units, double_counting = _accounting_validations(
        output, fixtures, benchmark, mechanism
    )
    cross_scale = _cross_scale(output, benchmark)
    preservation = _s08_preservation(output)
    _provenance(output)

    validation = {
        "schemaVersion": "e02.s09.validation_summary.v1",
        "researchStepId": "S09",
        "schemaSha256": COST_SCHEMA_SHA256,
        "componentGatesTotal": 10,
        "componentGatesPassed": sum(
            (
                schema_check["success"],
                all(item["passed"] for item in fixtures),
                zero["success"],
                identities["success"],
                units["success"],
                double_counting["success"],
                cross_scale["success"],
                replay["success"],
                architectures["success"],
                preservation["success"] and mechanism_validation["success"],
            )
        ),
        "handCountedFixtures": len(fixtures),
        "benchmarkRows": len(benchmark),
        "mechanismRows": len(mechanism),
        "ledgerRowsValidated": len(benchmark) + len(mechanism),
        "architectureComparabilityGroups": architectures["groupCount"],
        "deterministicReplays": replay["replayCount"],
        "sizes": sorted(benchmark["n"].unique().tolist()),
        "structureCells": benchmark[["n", "valueProfile", "orderStructure"]].drop_duplicates().shape[0],
        "protectedOutcomeReads": 0,
        "s08ArmsOrPairingChanged": False,
        "schedulerCandidateInspections": int(benchmark["schedulerCandidateInspections"].sum() + mechanism["schedulerCandidateInspections"].sum()),
        "validationResult": "PASS",
    }
    validation["success"] = validation["componentGatesPassed"] == validation["componentGatesTotal"]
    if not validation["success"]:
        validation["validationResult"] = "FAIL"
    _write_json(output / "validation_summary.json", validation)
    if not validation["success"]:
        raise AssertionError("S09 validation failed")


if __name__ == "__main__":
    main()
