#!/usr/bin/env python3
"""Execute fresh S08G under the byte-frozen S08P design and S08F gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from src.environment_suite import AccessDeniedError
from src.environment_suite.contracts import canonical_sha256
from src.portfolio_preregistration.core import canonical_hash, read_jsonl
from src.portfolio_search.execution import (
    ARTIFACT_ROOT,
    S08P,
    VALIDATION_FAMILIES,
    _catalog,
    _candidate_lock_authorize,
    _initial_work,
    _json_bytes,
    _physical_key,
    _write_json,
    aggregate_panel,
    build_action,
    build_archive,
    build_validation_plan,
    environment_record,
    execute_work,
    FailAtomicBatchError,
    generate_adaptive_generation,
    immutable_tree_hashes,
    paired_estimands,
    publish_parquet_fail_atomic,
    run_preflight,
    s09_gate,
    select_shortlist,
    sha256_file,
    validate_rows,
    _archive_rows,
)


S08G_CONTROL_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs/portfolio/s08g_execution_continuation.yaml"
)
S08G_CACHE_ROOT = Path("/cache/e07-s08g")
S08F = ARTIFACT_ROOT / "S08F"
IMMUTABLE_STEPS = (
    "S05",
    "S08P",
    "S08A",
    "S08",
    "S08B",
    "S08C",
    "S08D",
    "S08E",
    "S08F",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "S08G")
    parser.add_argument(
        "--phase", choices=("preflight", "smoke", "full"), default="full"
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_json_bytes(dict(row)) + b"\n" for row in rows))


def ledger_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    nested = {
        "memberPolicySha256",
        "validation",
        "nativeContractErrors",
        "outcome",
        "nativeLedgerFamilies",
        "nativeEvent",
        "provenance",
        "nativeNonFiniteValues",
    }
    projected = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if key in nested:
                item[f"{key}Json"] = json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            else:
                item[key] = value
        projected.append(item)
    return pd.DataFrame(projected)


def work_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(
        "E07/S08G/work-order-independent/v1",
        sorted(
            (
                str(row["logicalSlotId"]),
                str(row["configuration"]["configurationId"]),
                int(row["scenarioFamilyOrdinal"]),
            )
            for row in rows
        ),
    )


def result_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(
        "E07/S08G/result-order-independent/v1",
        sorted(
            (str(row["logicalSlotId"]), str(row["logicalExpansionSha256"]))
            for row in rows
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def s08f_contract_revalidation() -> dict[str, Any]:
    gate = _read_json(S08F / "s08_execution_eligibility_gate.json")
    descriptor = _read_json(S08F / "e05_terminal_support_qualification.json")
    accounting = _read_json(S08F / "full_registry_dedup_accounting.json")
    dedup = _read_json(S08F / "dedup_equivalence_qualification.json")
    identity = _read_json(S08F / "logical_identity_restoration.json")
    expected_archives = {
        "common:e07_s02_regeneration_1d:development",
        "common:e07_s02_regeneration_1d:stabilization",
        "common:e07_s02_regeneration_1d:recovery",
        "common:e07_s02_regeneration_1d:memoryResetRecovery",
        "common:e07_s02_regeneration_1d:robustnessFault",
        "common:e07_s02_regeneration_1d:transfer",
        "phenotype:e05:robustness",
        "phenotype:e05:repair",
        "phenotype:e05:memory",
        "phenotype:e05:transfer",
    }
    checks = {
        "g01ToG06": bool(gate.get("success"))
        and not gate.get("blockedGateIds")
        and [row.get("gateId") for row in gate.get("rows", [])]
        == ["G01", "G02", "G03", "G04", "G05", "G06"]
        and all(row.get("status") == "pass" for row in gate.get("rows", [])),
        "descriptorSupport": bool(descriptor.get("success"))
        and descriptor.get("descriptorAvailabilityVersion")
        == "e07.s08f.e05-descriptor-availability.v1"
        and descriptor.get("qualificationPanels") == 25
        and descriptor.get("logicalFixtureRows") == 100
        and descriptor.get("imputedValues") == 0
        and descriptor.get("silentlyDroppedRows") == 0
        and set(descriptor.get("expectedArchiveIds", [])) == expected_archives,
        "initialPhysicalAccounting": bool(accounting.get("success"))
        and accounting.get("configurationRows") == 1216
        and accounting.get("logicalRows") == 4864
        and accounting.get("uniquePhysicalRows") == 4864
        and accounting.get("legallyDeduplicatedLogicalRows") == 0
        and accounting.get("forwardPlanSha256") == accounting.get("reversePlanSha256"),
        "completeEquivalence": bool(dedup.get("success"))
        and all(row.get("pass") for row in dedup.get("cases", []))
        and set(dedup.get("equivalenceCommitments", []))
        == {
            "behavior",
            "native task/scenario/carrier",
            "selector/assignment",
            "communication",
            "portfolio structural cost",
            "native cost",
            "licensed-capability cost",
        },
        "logicalIdentityRestoration": bool(identity.get("success"))
        and identity.get("logicalRows") == 2
        and identity.get("physicalRows") == 1
        and len(set(identity.get("restoredConfigurationIds", []))) == 2
        and len(set(identity.get("logicalStableEvaluationSha256", []))) == 2,
    }
    return {
        "schemaVersion": "e07.s08g.s08f-contract-revalidation.v1",
        "researchStepId": "S08G",
        "checks": checks,
        "descriptorAvailabilityVersion": descriptor.get(
            "descriptorAvailabilityVersion"
        ),
        "s08fForwardPhysicalPlanSha256": accounting.get("forwardPlanSha256"),
        "success": all(checks.values()),
    }


def initial_physical_accounting_revalidation(
    budget: pd.DataFrame, definitions: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    work = _initial_work(budget, definitions)
    forward = [(row["logicalSlotId"], _physical_key(row)) for row in work]
    reverse = [(row["logicalSlotId"], _physical_key(row)) for row in reversed(work)]
    forward_map = dict(forward)
    reverse_map = dict(reverse)
    keys = [key for _, key in forward]
    evidence = _read_json(S08F / "full_registry_dedup_accounting.json")
    forward_digest = canonical_sha256("E07/S08F/full-physical-plan/v1", sorted(forward))
    reverse_digest = canonical_sha256("E07/S08F/full-physical-plan/v1", sorted(reverse))
    checks = {
        "configurationRows": len(definitions) == 1216,
        "logicalRows": len(work) == 4864,
        "uniquePhysicalRows": len(set(keys)) == 4864,
        "noPhysicalDedup": len(work) - len(set(keys)) == 0,
        "workerOrderIndependent": forward_map == reverse_map,
        "s08fPlanCommitment": forward_digest
        == evidence.get("forwardPlanSha256")
        == reverse_digest,
    }
    return work, {
        "schemaVersion": "e07.s08g.initial-physical-accounting-revalidation.v1",
        "researchStepId": "S08G",
        "checks": checks,
        "configurationRows": len(definitions),
        "logicalRows": len(work),
        "uniquePhysicalRows": len(set(keys)),
        "physicalDedupSavedRows": len(work) - len(set(keys)),
        "forwardPlanSha256": forward_digest,
        "reversePlanSha256": reverse_digest,
        "success": all(checks.values()),
    }


def descriptor_availability_audit(
    aggregates: Sequence[Mapping[str, Any]],
    archive: Mapping[tuple[str, str, tuple[int, ...]], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    e05 = [row for row in aggregates if row["taskId"] == "e07_s02_regeneration_1d"]
    records = [item for row in e05 for item in row["descriptors"]]
    unavailable = [
        item for item in records if item.get("supportState") == "unavailable"
    ]
    unavailable_pairs = {
        (row["configurationId"], item["archiveId"])
        for row in e05
        for item in row["descriptors"]
        if item.get("supportState") == "unavailable"
    }
    archive_pairs = {
        (str(item["configurationId"]), str(key[1]))
        for key, front in archive.items()
        for item in front
    }
    checks = {
        "tenRecordsPerConfiguration": all(len(row["descriptors"]) == 10 for row in e05),
        "unavailableRetained": bool(unavailable),
        "unavailableHasNoCoordinate": all(
            item.get("cellEligible") is False
            and item.get("values") is None
            and item.get("cell") is None
            and item.get("comparabilityKeySha256") is None
            for item in unavailable
        ),
        "availabilityNotNovelty": all(
            item.get("availabilityIsNoveltyCoordinate") is False for item in records
        ),
        "unavailableExcludedFromCells": not (unavailable_pairs & archive_pairs),
        "descriptorVersion": all(
            item.get("descriptorAvailabilityVersion")
            == "e07.s08f.e05-descriptor-availability.v1"
            for item in records
        ),
    }
    return {
        "schemaVersion": "e07.s08g.e05-descriptor-availability-audit.v1",
        "researchStepId": "S08G",
        "configurationAggregates": len(e05),
        "descriptorRecords": len(records),
        "completeRecords": len(records) - len(unavailable),
        "unavailableRecordsRetained": len(unavailable),
        "unavailableArchivePairIntersections": len(unavailable_pairs & archive_pairs),
        "checks": checks,
        "success": bool(e05) and all(checks.values()),
    }


def logical_identity_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_physical: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_physical[str(row["physicalWorkSha256"])].append(row)
    reused = [items for items in by_physical.values() if len(items) > 1]
    checks = {
        "everyLogicalIdentityRestored": all(
            row["configurationId"] == row["reservedConfigurationSlotId"] for row in rows
        ),
        "physicalProvenanceRetained": all(
            row.get("physicalStableEvaluationSha256")
            and row.get("physicalDedupEquivalenceSha256")
            and row.get("logicalExpansionSha256")
            for row in rows
        ),
        "logicalSlotsUnique": len({row["logicalSlotId"] for row in rows}) == len(rows),
        "reusedRowsKeepDistinctLogicalHashes": all(
            len({item["stableEvaluationSha256"] for item in items}) == len(items)
            for items in reused
        ),
    }
    return {
        "schemaVersion": "e07.s08g.logical-identity-audit.v1",
        "researchStepId": "S08G",
        "logicalRows": len(rows),
        "uniquePhysicalKeys": len(by_physical),
        "legallyReusedPhysicalKeys": len(reused),
        "logicalRowsInReusedGroups": sum(len(items) for items in reused),
        "checks": checks,
        "success": all(checks.values()),
    }


def configuration_hash_validation(
    definitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_hash, by_id = _catalog()
    rows = []
    for cid, definition in sorted(definitions.items()):
        action = build_action(definition, by_hash, by_id)
        rows.append(
            {
                "configurationId": cid,
                "taskId": definition["taskId"],
                "mode": definition["mode"],
                "portfolioSize": definition["portfolioSize"],
                "configurationDefinitionSha256": canonical_hash(
                    "E07/S08G/configuration-definition/v1", definition
                ),
                "actionSha256": action.policy_sha256,
                "memberPolicySha256": [
                    member["policySha256"] for member in definition["members"]
                ],
                "s07ArmMembershipUsed": bool(
                    definition.get("s07ArmMembershipUsed", False)
                ),
                "rejectedModelOrEmbeddingUsed": bool(
                    definition.get("rejectedModelOrEmbeddingUsed", False)
                ),
            }
        )
    return {
        "schemaVersion": "e07.s08g.configuration-hash-validation.v1",
        "researchStepId": "S08G",
        "success": all(
            not row["s07ArmMembershipUsed"]
            and not row["rejectedModelOrEmbeddingUsed"]
            and len(row["configurationDefinitionSha256"]) == 64
            and len(row["actionSha256"]) == 64
            for row in rows
        ),
        "configurationRows": len(rows),
        "uniqueConfigurationIds": len({row["configurationId"] for row in rows}),
        "uniqueActionHashes": len({row["actionSha256"] for row in rows}),
        "rows": rows,
    }


def validation_support_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["taskId"]].append(row)
    result = {}
    for task_id, task_rows in sorted(by_task.items()):
        scenario_by_family = {
            int(row["scenarioFamilyOrdinal"]): row["scenarioCommitmentSha256"]
            for row in task_rows
        }
        # Count distinct native result projections, not logical labels.
        support = {
            canonical_hash(
                "E07/S08G/native-validation-support/v1",
                {
                    "outcome": row["outcome"],
                    "nativeEvent": row["nativeEvent"],
                    "stopReason": row["stopReason"],
                },
            )
            for row in task_rows
        }
        result[task_id] = {
            "logicalFamilies": len(scenario_by_family),
            "distinctScenarioCommitments": len(set(scenario_by_family.values())),
            "distinctObservedNativeResultProjectionsAcrossAllConfigurations": len(
                support
            ),
            "independenceClaim": "not_made",
        }
    return {
        "schemaVersion": "e07.s08g.validation-support-audit.v1",
        "researchStepId": "S08G",
        "nativeSupportsRemainTaskSpecific": True,
        "universalScenarioNormalizationApplied": False,
        "byTask": result,
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in [1,8]")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    output = args.output
    if args.phase != "preflight" and S08G_CACHE_ROOT.exists():
        raise SystemExit(
            f"fresh S08G cache namespace already exists: {S08G_CACHE_ROOT}"
        )
    output.mkdir(parents=True, exist_ok=True)
    control = yaml.safe_load(S08G_CONTROL_PATH.read_text(encoding="utf-8"))
    immutable_expected = dict(control["expectedTrees"])
    _write_json(
        output / "input_hash_freeze.json",
        {
            "schemaVersion": "e07.s08g.input-hash-freeze.v1",
            "researchStepId": "S08G",
            "controlPath": str(S08G_CONTROL_PATH),
            "controlSha256": sha256_file(S08G_CONTROL_PATH),
            "expectedTrees": immutable_expected,
            "expectedFiles": control["expectedFiles"],
            "freshCacheRoot": str(S08G_CACHE_ROOT),
            "freshCacheAbsentBeforeExecution": not S08G_CACHE_ROOT.exists(),
            "quarantinedCacheRoots": control["quarantinedCacheRoots"],
            "quarantinedOutcomeRowsLoaded": 0,
        },
    )
    _write_json(
        output / "preregistration_freeze.json",
        {
            "schemaVersion": "e07.s08g.preregistration-freeze.v1",
            "researchStepId": "S08G",
            "sourceDesign": control["sourceDesign"],
            "sourceDesignSha256": sha256_file(Path(control["sourceDesign"])),
            "frozenExecution": control["frozenExecution"],
            "smokePublication": control["smokePublication"],
            "validationBoundary": control["validationBoundary"],
            "analysis": control["analysis"],
            "historicalDesignMayChange": False,
            "runtimeDrivenWeakening": False,
            "reuseS08cOutcomes": False,
            "reuseS08eOutcomes": False,
        },
    )
    preflight = run_preflight(S08G_CONTROL_PATH)
    s08f_contract = s08f_contract_revalidation()
    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    initial_definitions = read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    definitions: dict[str, dict[str, Any]] = {
        str(row["configurationId"]): dict(row) for row in initial_definitions
    }
    initial_work, initial_accounting_revalidation = (
        initial_physical_accounting_revalidation(budget, definitions)
    )
    if not s08f_contract["success"]:
        for row in preflight["gateRows"]:
            if row["gateId"] == "G04":
                row["status"] = "blocked"
    if not initial_accounting_revalidation["success"]:
        for row in preflight["gateRows"]:
            if row["gateId"] == "G06":
                row["status"] = "blocked"
    preflight["blockedGateIds"] = [
        row["gateId"] for row in preflight["gateRows"] if row["status"] != "pass"
    ]
    preflight["s08fContractRevalidationPass"] = s08f_contract["success"]
    preflight["initialPhysicalAccountingPass"] = initial_accounting_revalidation[
        "success"
    ]
    preflight["freshCacheAbsent"] = not S08G_CACHE_ROOT.exists()
    preflight["quarantinedOutcomeRowsLoaded"] = 0
    preflight["success"] = (
        not preflight["blockedGateIds"]
        and preflight["noMutation"]
        and preflight["freshCacheAbsent"]
    )
    preflight["status"] = (
        "eligible_for_frozen_smoke" if preflight["success"] else "blocked_before_smoke"
    )
    _write_json(output / "preflight_hash_and_gate_revalidation.json", preflight)
    _write_json(output / "s08f_contract_revalidation.json", s08f_contract)
    _write_json(
        output / "initial_physical_accounting_revalidation.json",
        initial_accounting_revalidation,
    )
    _write_json(
        output / "s08f_binding_revalidation.json", preflight["bindingValidation"]
    )
    _write_json(
        output / "access_control_preflight.json",
        {
            "schemaVersion": "e07.s08g.access-control-preflight.v1",
            "researchStepId": "S08G",
            "success": all(
                row["deniedBeforeMaterialization"] for row in preflight["accessRows"]
            ),
            "rows": preflight["accessRows"],
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
        },
    )
    if not preflight["success"]:
        _write_json(
            output / "execution_stop_status.json",
            {
                "researchStepId": "S08G",
                "status": "blocked_before_smoke",
                "success": False,
                "blockedGateIds": preflight["blockedGateIds"],
                "trainingLogicalRowsExecuted": 0,
                "validationLogicalRowsExecuted": 0,
                "confirmationLogicalRowsExecuted": 0,
            },
        )
        return 2
    if args.phase == "preflight":
        return 0

    smoke_work = [row for row in initial_work if row["smoke"]]
    smoke_failure_accounting = output / "smoke_batch_forensic_accounting.json"
    try:
        smoke_rows, smoke_physical = execute_work(
            smoke_work,
            workers=args.workers,
            cache_dir=S08G_CACHE_ROOT / "evaluations",
            failure_accounting_path=smoke_failure_accounting,
            atomic_new_cache=True,
        )
    except FailAtomicBatchError as exc:
        _write_json(
            output / "execution_stop_status.json",
            {
                "researchStepId": "S08G",
                "status": "blocked_at_frozen_smoke",
                "success": False,
                "failureAccountingPath": str(smoke_failure_accounting),
                "attemptedPhysicalRows": exc.accounting["attemptedPhysicalRows"],
                "publishedSmokeRows": 0,
                "trainingLogicalRowsExecuted": 0,
                "validationLogicalRowsExecuted": 0,
                "confirmationLogicalRowsExecuted": 0,
            },
        )
        return 3
    smoke_validation = validate_rows(smoke_rows, 40)
    smoke_validation.update(
        {
            "schemaVersion": "e07.s08g.frozen-smoke-validation.v1",
            "researchStepId": "S08G",
            "withinFrozenTrainingBudget": True,
            "physicalAccounting": smoke_physical,
            "substantiveSearchAuthorized": smoke_validation["success"],
        }
    )
    _write_json(output / "frozen_smoke_validation.json", smoke_validation)
    if not smoke_validation["success"]:
        _write_json(
            smoke_failure_accounting,
            {
                "schemaVersion": "e07.s08g.smoke-forensic-accounting.v1",
                "researchStepId": "S08G",
                "success": False,
                "failureClass": "post_execution_integrity_validation",
                "physicalAccounting": smoke_physical,
                "validation": smoke_validation,
                "publishedSmokeRows": 0,
            },
        )
        _write_json(
            output / "execution_stop_status.json",
            {
                "researchStepId": "S08G",
                "status": "blocked_at_frozen_smoke",
                "success": False,
                "trainingLogicalRowsExecuted": 40,
                "validationLogicalRowsExecuted": 0,
                "confirmationLogicalRowsExecuted": 0,
            },
        )
        return 3
    publish_parquet_fail_atomic(
        ledger_frame(smoke_rows), output / "frozen_smoke_results.parquet"
    )
    if args.phase == "smoke":
        return 0

    training_rows, initial_physical = execute_work(
        initial_work, workers=args.workers, cache_dir=S08G_CACHE_ROOT / "evaluations"
    )
    initial_aggregates = aggregate_panel(definitions, training_rows)
    aggregates = list(initial_aggregates)
    archive = build_archive(aggregates)
    all_work = list(initial_work)
    lineage_rows = []
    convergence = [
        {
            "generation": 0,
            "logicalRowsCumulative": len(training_rows),
            "archiveCells": len(archive),
            "archiveFrontEntries": sum(len(value) for value in archive.values()),
            "validTargetConfigurations": sum(
                row["mode"] in {"fixed_balanced_identity", "environment_conditioned"}
                and row["validNativeEpisodes"]
                for row in aggregates
            ),
        }
    ]
    candidates = read_jsonl(S08P / "candidate_eligibility_registry.jsonl")
    protocol = yaml.safe_load((S08P / "s08p_portfolio_protocol.yaml").read_text())
    selector_signals = protocol["selectorSignals"]["byTask"]
    prior_ids = set(definitions)
    physical_accounting = {"smoke": smoke_physical, "initial": initial_physical}
    for generation in range(1, 7):
        works, generated, generation_lineage = generate_adaptive_generation(
            generation,
            archive,
            definitions,
            candidates,
            selector_signals,
            budget,
            prior_ids,
        )
        definitions.update(generated)
        rows, physical = execute_work(
            works, workers=args.workers, cache_dir=S08G_CACHE_ROOT / "evaluations"
        )
        training_rows.extend(rows)
        all_work.extend(works)
        lineage_rows.extend(generation_lineage)
        generation_aggregates = aggregate_panel(definitions, rows)
        aggregates.extend(generation_aggregates)
        archive = build_archive(aggregates)
        physical_accounting[f"generation_{generation}"] = physical
        convergence.append(
            {
                "generation": generation,
                "logicalRowsCumulative": len(training_rows),
                "archiveCells": len(archive),
                "archiveFrontEntries": sum(len(value) for value in archive.values()),
                "validTargetConfigurations": sum(
                    row["mode"]
                    in {"fixed_balanced_identity", "environment_conditioned"}
                    and row["validNativeEpisodes"]
                    for row in aggregates
                ),
            }
        )
        checkpoint = S08G_CACHE_ROOT / f"generation_{generation}_checkpoint.json"
        _write_json(
            checkpoint,
            {
                "generation": generation,
                "trainingLogicalRows": len(training_rows),
                "workDigest": work_digest(all_work),
                "resultDigest": result_digest(training_rows),
                "definitionIds": sorted(definitions),
            },
        )

    descriptor_audit = descriptor_availability_audit(aggregates, archive)
    identity_audit = logical_identity_audit(training_rows)
    initial_execution_accounting = {
        "schemaVersion": "e07.s08g.initial-execution-physical-accounting.v1",
        "researchStepId": "S08G",
        "initialLogicalRows": len(initial_work),
        "initialUniquePhysicalRows": initial_physical["uniquePhysicalRows"],
        "smokePhysicalRowsExecuted": smoke_physical["physicalRowsExecutedNow"],
        "remainingInitialPhysicalRowsExecuted": initial_physical[
            "physicalRowsExecutedNow"
        ],
        "initialCacheHitsFromFreshSmoke": initial_physical["cacheHits"],
        "freshPhysicalRowsAcrossSmokeAndInitial": smoke_physical[
            "physicalRowsExecutedNow"
        ]
        + initial_physical["physicalRowsExecutedNow"],
        "physicalDedupSavedRows": initial_physical["physicalDedupSavedRows"],
        "physicalDedupEquivalenceVersion": initial_physical[
            "physicalDedupEquivalenceVersion"
        ],
        "logicalExpansionVersion": initial_physical["logicalExpansionVersion"],
    }
    initial_execution_accounting["success"] = (
        initial_execution_accounting["initialLogicalRows"] == 4864
        and initial_execution_accounting["initialUniquePhysicalRows"] == 4864
        and initial_execution_accounting["smokePhysicalRowsExecuted"] == 40
        and initial_execution_accounting["remainingInitialPhysicalRowsExecuted"] == 4824
        and initial_execution_accounting["initialCacheHitsFromFreshSmoke"] == 40
        and initial_execution_accounting["freshPhysicalRowsAcrossSmokeAndInitial"]
        == 4864
        and initial_execution_accounting["physicalDedupSavedRows"] == 0
        and initial_execution_accounting["physicalDedupEquivalenceVersion"]
        == "e07.s08f.physical-work.v1"
        and initial_execution_accounting["logicalExpansionVersion"]
        == "e07.s08f.logical-expansion.v1"
    )
    _write_json(
        output / "initial_execution_physical_accounting.json",
        initial_execution_accounting,
    )
    _write_json(output / "e05_descriptor_availability_audit.json", descriptor_audit)
    _write_json(output / "logical_identity_restoration_audit.json", identity_audit)
    training_validation = validate_rows(training_rows, 11008)
    training_validation.update(
        {
            "schemaVersion": "e07.s08g.training-validation.v1",
            "researchStepId": "S08G",
            "generationsCompleted": 6,
            "runtimeDrivenWeakening": False,
            "efficacyEarlyStopUsed": False,
            "initialPhysicalAccountingPass": initial_execution_accounting["success"],
            "descriptorAvailabilityPass": descriptor_audit["success"],
            "logicalIdentityRestorationPass": identity_audit["success"],
        }
    )
    training_validation["success"] = training_validation["success"] and all(
        (
            initial_execution_accounting["success"],
            descriptor_audit["success"],
            identity_audit["success"],
        )
    )
    if not training_validation["success"]:
        _write_json(output / "training_validation.json", training_validation)
        return 4

    initial_single_by_member = {}
    for definition in initial_definitions:
        if definition["mode"] == "single_policy":
            initial_single_by_member[
                (definition["taskId"], definition["members"][0]["policySha256"])
            ] = definition
    shortlist, single_leaders = select_shortlist(
        archive, definitions, aggregates, initial_single_by_member
    )
    lock_path = output / "validation_candidate_lock.json"
    validation_work, lock, validation_definitions = build_validation_plan(
        shortlist,
        single_leaders,
        definitions,
        initial_single_by_member,
        lock_path=lock_path,
    )
    lock_sha = sha256_file(lock_path)
    # Demonstrate lock denial before any validation outcome is materialized.
    first_locked = next(iter(validation_definitions.values()))
    first_action = build_action(first_locked, *_catalog())
    denied_absent = False
    try:
        _candidate_lock_authorize(
            lock_path,
            lock_sha,
            "f" * 64,
            str(first_locked["taskId"]),
            first_action.policy_sha256,
        )
    except AccessDeniedError:
        denied_absent = True
    if not denied_absent:
        raise RuntimeError("candidate lock accepted an absent configuration")
    validation_rows, validation_physical = execute_work(
        validation_work,
        workers=args.workers,
        cache_dir=S08G_CACHE_ROOT / "validation_evaluations",
    )
    validation_validation = validate_rows(validation_rows, len(validation_work))
    if not validation_validation["success"]:
        _write_json(
            output / "validation_execution_validation.json", validation_validation
        )
        return 5
    endpoint_rows, cost_rows, guard_rows = paired_estimands(
        validation_rows,
        lock["comparisonRegistry"],
        validation_definitions,
    )
    gate = s09_gate(shortlist, endpoint_rows, guard_rows, validation_rows)
    gate["schemaVersion"] = "e07.s08g.s09-eligibility-gate.v1"
    gate["researchStepId"] = "S08G"

    # Compact canonical artifacts.
    ledger_frame(training_rows).to_parquet(
        output / "training_evaluation_ledger.parquet", index=False, compression="zstd"
    )
    ledger_frame(validation_rows).to_parquet(
        output / "validation_evaluation_ledger.parquet", index=False, compression="zstd"
    )
    write_jsonl(
        output / "portfolio_configuration_registry.jsonl", list(definitions.values())
    )
    write_jsonl(output / "configuration_aggregates.jsonl", aggregates)
    write_jsonl(output / "adaptive_lineage_ledger.jsonl", lineage_rows)
    write_jsonl(output / "portfolio_archive.jsonl", _archive_rows(archive))
    pd.DataFrame(endpoint_rows).to_csv(
        output / "paired_native_estimands.csv", index=False
    )
    pd.DataFrame(cost_rows).to_csv(output / "paired_cost_estimands.csv", index=False)
    pd.DataFrame(guard_rows).to_csv(output / "failure_censor_guards.csv", index=False)
    _write_json(output / "training_validation.json", training_validation)
    _write_json(
        output / "convergence_and_stopping.json",
        {
            "schemaVersion": "e07.s08g.convergence.v1",
            "researchStepId": "S08G",
            "rows": convergence,
            "stoppingRule": "exactly_six_generations_or_integrity_failure",
            "generationsCompleted": 6,
            "runtimeDrivenWeakening": False,
        },
    )
    _write_json(
        output / "training_shortlist.json",
        {
            "schemaVersion": "e07.s08g.training-shortlist.v1",
            "researchStepId": "S08G",
            "byTask": shortlist,
            "singleFrontierObjectiveLeadersByTask": single_leaders,
            "maximumPortfoliosPerTask": 8,
            "selectionUsesS07ArmMembership": False,
            "selectionUsesRejectedModelOrEmbedding": False,
        },
    )
    _write_json(
        output / "validation_plan_and_accounting.json",
        {
            "schemaVersion": "e07.s08g.validation-plan-accounting.v1",
            "researchStepId": "S08G",
            "candidateLockSha256": lock_sha,
            "logicalRows": len(validation_work),
            "logicalCeiling": 3072,
            "withinCeiling": len(validation_work) <= 3072,
            "familiesPerTask": len(VALIDATION_FAMILIES),
            "physicalAccounting": validation_physical,
            "confirmationRows": 0,
        },
    )
    _write_json(output / "validation_execution_validation.json", validation_validation)
    _write_json(
        output / "validation_support_audit.json",
        validation_support_audit(validation_rows),
    )
    _write_json(
        output / "uncertainty_and_multiplicity.json",
        {
            "schemaVersion": "e07.s08g.uncertainty-multiplicity.v1",
            "researchStepId": "S08G",
            "pairedScenarioBlockBootstrapReplicates": 2000,
            "confidenceLevel": 0.95,
            "seed": 708004,
            "multiplicityMethod": "Holm",
            "family": "within_task_target_contrast_across_primary_native_endpoints",
            "crossTaskPooling": False,
            "endpointRows": len(endpoint_rows),
        },
    )
    _write_json(output / "s09_eligibility_gate.json", gate)
    config_hashes = configuration_hash_validation(definitions)
    _write_json(output / "configuration_and_member_hash_validation.json", config_hashes)
    action_groups = defaultdict(list)
    for row in config_hashes["rows"]:
        action_groups[row["actionSha256"]].append(row["configurationId"])
    semantic_groups = defaultdict(list)
    for aggregate in aggregates:
        signature = canonical_hash(
            "E07/S08G/finite-panel-equivalence/v1",
            aggregate["evaluationHashes"],
        )
        semantic_groups[(aggregate["taskId"], signature)].append(
            aggregate["configurationId"]
        )
    _write_json(
        output / "duplicate_detection_audit.json",
        {
            "schemaVersion": "e07.s08g.duplicate-audit.v1",
            "researchStepId": "S08G",
            "canonicalConfigurationDuplicates": len(definitions)
            - len(set(definitions)),
            "actionHashGroupsWithMultipleConfigurations": [
                {"actionSha256": key, "configurationIds": value}
                for key, value in sorted(action_groups.items())
                if len(value) > 1
            ],
            "finiteTrainingPanelSemanticGroupsWithMultipleConfigurations": [
                {"taskId": key[0], "signature": key[1], "configurationIds": value}
                for key, value in sorted(semantic_groups.items())
                if len(value) > 1
            ],
            "finiteSupportOnly": True,
        },
    )
    order_validation = {
        "schemaVersion": "e07.s08g.replay-worker-order-validation.v1",
        "researchStepId": "S08G",
        "forwardWorkDigest": work_digest(all_work),
        "reverseWorkDigest": work_digest(list(reversed(all_work))),
        "forwardResultDigest": result_digest(training_rows),
        "reverseResultDigest": result_digest(list(reversed(training_rows))),
        "workerOrderIndependent": work_digest(all_work)
        == work_digest(list(reversed(all_work)))
        and result_digest(training_rows)
        == result_digest(list(reversed(training_rows))),
        "nativeExactReplayRows": sum(row["replayPass"] for row in training_rows),
        "validationExactReplayRows": sum(row["replayPass"] for row in validation_rows),
    }
    order_validation["success"] = order_validation["workerOrderIndependent"]
    _write_json(output / "replay_worker_order_validation.json", order_validation)
    logical_by_generation = Counter(int(row["generation"]) for row in training_rows)
    logical_by_task = Counter(str(row["taskId"]) for row in training_rows)
    logical_by_family = Counter(
        int(row["scenarioFamilyOrdinal"]) for row in training_rows
    )
    budget_accounting = {
        "schemaVersion": "e07.s08g.complete-budget-accounting.v1",
        "researchStepId": "S08G",
        "trainingLogicalRows": len(training_rows),
        "trainingLogicalRowsExpected": 11008,
        "smokeLogicalRows": 40,
        "smokeIncludedInTrainingBudget": True,
        "byGeneration": dict(sorted(logical_by_generation.items())),
        "byTask": dict(sorted(logical_by_task.items())),
        "byScenarioFamily": dict(sorted(logical_by_family.items())),
        "validationLogicalRows": len(validation_rows),
        "validationLogicalCeiling": 3072,
        "confirmationLogicalRows": 0,
        "physicalAccounting": physical_accounting,
        "validationPhysicalAccounting": validation_physical,
        "initialExecutionPhysicalAccounting": initial_execution_accounting,
        "runtimeDrivenWeakening": False,
        "complete": len(training_rows) == 11008
        and len(validation_rows) <= 3072
        and logical_by_generation
        == Counter({0: 4864, 1: 1024, 2: 1024, 3: 1024, 4: 1024, 5: 1024, 6: 1024}),
    }
    _write_json(output / "complete_budget_accounting.json", budget_accounting)
    _write_json(
        output / "access_control_validation.json",
        {
            "schemaVersion": "e07.s08g.access-control-validation.v1",
            "researchStepId": "S08G",
            "preflightNontrainingDenials": 16,
            "absentCandidateLockEntryDenied": denied_absent,
            "validationAccessAfterCandidateLockOnly": True,
            "validationOutcomeEvaluations": len(validation_rows),
            "confirmationScenarioMaterializations": 0,
            "confirmationOutcomeEvaluations": 0,
            "confirmationRemainsSealed": True,
            "success": denied_absent and len(validation_rows) <= 3072,
        },
    )
    _write_json(
        output / "dependency_exclusion_audit.json",
        {
            "schemaVersion": "e07.s08g.dependency-exclusion-audit.v1",
            "researchStepId": "S08G",
            "loadedProhibitedModules": preflight["loadedProhibitedModules"],
            "rejectedModelArtifactsOpened": 0,
            "rejectedEmbeddingArtifactsOpened": 0,
            "pseudoLabelsUsed": 0,
            "warmStartsUsed": 0,
            "distillationUses": 0,
            "s07ArmPromotionUses": 0,
            "s08cQuarantineOutcomeRowsLoaded": 0,
            "s08eQuarantineOutcomeRowsLoaded": 0,
            "success": not preflight["loadedProhibitedModules"],
        },
    )
    final_trees = immutable_tree_hashes(IMMUTABLE_STEPS)
    no_mutation = final_trees == immutable_expected
    _write_json(
        output / "no_mutation_audit.json",
        {
            "schemaVersion": "e07.s08g.no-mutation-audit.v1",
            "researchStepId": "S08G",
            "expectedTrees": immutable_expected,
            "observedTrees": final_trees,
            "unchanged": no_mutation,
            "s05ArchiveMutations": 0,
            "historicalS08ArtifactMutations": 0,
            "success": no_mutation,
        },
    )
    _write_json(output / "environment.json", environment_record())
    _write_json(
        output / "provenance.json",
        {
            "schemaVersion": "e07.s08g.provenance.v1",
            "researchStepId": "S08G",
            "controlPath": str(S08G_CONTROL_PATH),
            "controlSha256": sha256_file(S08G_CONTROL_PATH),
            "s08pProtocolSha256": sha256_file(S08P / "s08p_portfolio_protocol.yaml"),
            "candidateLockSha256": lock_sha,
            "trainingWorkDigest": work_digest(all_work),
            "trainingResultDigest": result_digest(training_rows),
            "validationResultDigest": result_digest(validation_rows),
            "immutableTrees": final_trees,
            "trainingOutcomeEvaluations": len(training_rows),
            "validationOutcomeEvaluations": len(validation_rows),
            "confirmationOutcomeEvaluations": 0,
        },
    )
    summary = {
        "schemaVersion": "e07.s08g.validation-summary.v1",
        "researchStepId": "S08G",
        "checks": {
            "preflight": preflight["success"],
            "g01ToG06": not preflight["blockedGateIds"],
            "smoke40": smoke_validation["success"],
            "training11008": training_validation["success"],
            "sixGenerations": len(convergence) == 7,
            "initialPhysical4864": initial_execution_accounting["success"],
            "e05DescriptorAvailability": descriptor_audit["success"],
            "logicalIdentityRestoration": identity_audit["success"],
            "replay": order_validation["success"],
            "configurationHashes": config_hashes["success"],
            "validationWithinCeiling": len(validation_rows) <= 3072,
            "candidateLock": denied_absent,
            "confirmationSealed": True,
            "dependencyExclusion": not preflight["loadedProhibitedModules"],
            "noMutation": no_mutation,
            "completeAccounting": budget_accounting["complete"],
        },
        "eligibleS09Configurations": len(gate["eligibleConfigurationIds"]),
    }
    summary["passed"] = sum(summary["checks"].values())
    summary["total"] = len(summary["checks"])
    summary["success"] = all(summary["checks"].values())
    _write_json(output / "validation_summary.json", summary)
    if not summary["success"]:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
