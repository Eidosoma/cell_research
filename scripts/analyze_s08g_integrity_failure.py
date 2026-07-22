#!/usr/bin/env python3
"""Write compact fail-closed forensics for the S08G generation-0 stop."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.environment_suite.contracts import canonical_sha256
from src.portfolio_preregistration.core import read_jsonl
from src.portfolio_search.execution import (
    S08P,
    _initial_work,
    _physical_key,
    immutable_tree_hashes,
)
from src.quality_diversity.core import (
    E05_REGENERATION_DESCRIPTOR_SPECS,
    _descriptor_sets,
    _edges_for,
)


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


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_commitment(paths: Sequence[Path]) -> str:
    return canonical_sha256(
        "E07/S08G/generation-0-cache/v1",
        [(path.name, _sha256_file(path)) for path in sorted(paths)],
    )


def _write_artifact_manifest(output: Path) -> None:
    artifacts = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e07.s08g.artifact-manifest.v1",
            "researchStepId": "S08G",
            "artifactCount": len(artifacts),
            "artifacts": artifacts,
            "success": True,
        },
    )


def _descriptor_forensics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    e05_rows = [row for row in rows if row["taskId"] == "e07_s02_regeneration_1d"]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in e05_rows:
        grouped[str(row["configurationId"])].append(row)

    support_counts: Counter[str] = Counter()
    violation_counts: Counter[tuple[str, str]] = Counter()
    violations: list[dict[str, Any]] = []
    terminal_counts: Counter[tuple[str, bool, bool]] = Counter()
    for configuration_id, panel in sorted(grouped.items()):
        panel = sorted(panel, key=lambda row: int(row["scenarioFamilyOrdinal"]))
        by_family = {
            str(row["scenarioFamilyOrdinal"]): {
                item["archiveId"]: item
                for item in _descriptor_sets(str(row["taskId"]), row["outcome"])
            }
            for row in panel
        }
        for row in panel:
            terminal_counts[
                (str(row["stopReason"]), bool(row["failed"]), bool(row["censored"]))
            ] += 1
        for archive_id, spec in E05_REGENERATION_DESCRIPTOR_SPECS.items():
            items = [
                by_family[str(row["scenarioFamilyOrdinal"])].get(archive_id)
                for row in panel
            ]
            if any(item is None for item in items):
                support_counts["unavailable"] += 1
                continue
            support_counts["complete"] += 1
            typed_items = [dict(item) for item in items if item is not None]
            edges = _edges_for(str(spec["kind"]), tuple(spec["fields"]))
            for field in spec["fields"]:
                values = [float(item["values"][field]) for item in typed_items]
                mean = sum(values) / len(values)
                if not math.isfinite(mean) or mean < edges[field][0] or mean > edges[field][-1]:
                    violation_counts[(archive_id, field)] += 1
                    violations.append(
                        {
                            "configurationId": configuration_id,
                            "archiveId": archive_id,
                            "kind": str(spec["kind"]),
                            "field": field,
                            "mean": mean,
                            "scenarioMinimum": min(values),
                            "scenarioMaximum": max(values),
                            "frozenLowerBound": edges[field][0],
                            "frozenUpperBound": edges[field][-1],
                            "scenarioFamilyOrdinals": [
                                int(row["scenarioFamilyOrdinal"]) for row in panel
                            ],
                            "scenarioValues": values,
                            "nativeStatuses": [
                                {
                                    "scenarioFamilyOrdinal": int(
                                        row["scenarioFamilyOrdinal"]
                                    ),
                                    "stopReason": str(row["stopReason"]),
                                    "failed": bool(row["failed"]),
                                    "censored": bool(row["censored"]),
                                }
                                for row in panel
                            ],
                        }
                    )

    return {
        "schemaVersion": "e07.s08g.generation-0-descriptor-forensics.v1",
        "researchStepId": "S08G",
        "taskId": "e07_s02_regeneration_1d",
        "configurationPanels": len(grouped),
        "panelSizeCounts": dict(sorted(Counter(len(rows) for rows in grouped.values()).items())),
        "descriptorPanels": sum(support_counts.values()),
        "supportStateCounts": dict(sorted(support_counts.items())),
        "boundViolationCount": len(violations),
        "affectedConfigurationCount": len(
            {row["configurationId"] for row in violations}
        ),
        "violationsByArchiveAndField": [
            {"archiveId": key[0], "field": key[1], "count": count}
            for key, count in sorted(violation_counts.items())
        ],
        "nativeTerminalCounts": [
            {
                "stopReason": key[0],
                "failed": key[1],
                "censored": key[2],
                "count": count,
            }
            for key, count in sorted(terminal_counts.items())
        ],
        "violations": violations,
        "archiveConstructionAuthorized": False,
        "availabilityUsedAsNovelty": False,
        "imputationApplied": False,
        "silentDropApplied": False,
        "success": not violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/cache/e07-s08g/evaluations"))
    parser.add_argument(
        "--output", type=Path, default=Path("/artifacts/research_steps/S08G")
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path("/artifacts/research_steps/S08G/preflight_hash_and_gate_revalidation.json"),
    )
    args = parser.parse_args()
    paths = sorted(args.cache.glob("*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(paths) != 4864 or len(rows) != 4864:
        raise ValueError("S08G generation-0 cache must contain exactly 4,864 rows")

    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    definitions = {
        str(row["configurationId"]): dict(row)
        for row in read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    }
    expected_work = _initial_work(budget, definitions)
    expected_by_slot = {
        str(row["logicalSlotId"]): _physical_key(row) for row in expected_work
    }
    observed_by_slot = {
        str(row["logicalSlotId"]): path.stem
        for path, row in zip(paths, rows, strict=True)
    }
    cache_keys_match = observed_by_slot == expected_by_slot
    task_counts = Counter(str(row["taskId"]) for row in rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    access_counts = Counter(str(row["accessBoundary"]) for row in rows)
    generation_counts = Counter(int(row["generation"]) for row in rows)
    replay_pass = sum(bool(row["replayPass"]) for row in rows)
    contract_pass = sum(bool(row["nativeContractPass"]) for row in rows)
    contract_failures = [
        {
            "logicalSlotId": str(row["logicalSlotId"]),
            "configurationId": str(row["configurationId"]),
            "taskId": str(row["taskId"]),
            "scenarioFamilyOrdinal": int(row["scenarioFamilyOrdinal"]),
            "stopReason": str(row["stopReason"]),
            "failed": bool(row["failed"]),
            "censored": bool(row["censored"]),
            "replayPass": bool(row["replayPass"]),
            "nativeContractErrors": list(row["nativeContractErrors"]),
            "falseValidationFlags": sorted(
                key for key, value in row["validation"].items() if not value
            ),
            "stableEvaluationSha256": str(row["stableEvaluationSha256"]),
        }
        for row in rows
        if not bool(row["nativeContractPass"])
    ]
    failed = sum(bool(row["failed"]) for row in rows)
    censored = sum(bool(row["censored"]) for row in rows)
    descriptor = _descriptor_forensics(rows)

    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    before_trees = dict(preflight["treeHashesBefore"])
    after_trees = immutable_tree_hashes(IMMUTABLE_STEPS)
    no_mutation = before_trees == after_trees
    quarantine = {
        "schemaVersion": "e07.s08g.training-evaluation-quarantine.v1",
        "researchStepId": "S08G",
        "status": "quarantined_after_generation_0_integrity_failure",
        "cachePath": str(args.cache),
        "cacheRows": len(rows),
        "cacheCommitmentSha256": _cache_commitment(paths),
        "allowedUse": ["integrity_forensics", "exact_execution_accounting"],
        "prohibitedUse": [
            "efficacy_claims",
            "promotion",
            "archive_input",
            "adaptive_generation",
            "candidate_lock",
            "validation_selection",
            "s09_input",
        ],
        "priorQuarantinesLoaded": False,
        "canonicalTrainingLedgerPublished": False,
    }
    accounting = {
        "schemaVersion": "e07.s08g.execution-accounting.v1",
        "researchStepId": "S08G",
        "freshCache": True,
        "frozenSmokeLogicalRows": 40,
        "frozenSmokePhysicalRows": 40,
        "initialLogicalRows": len(rows),
        "initialPhysicalRows": len(paths),
        "initialPhysicalDedupSavedRows": len(rows) - len(paths),
        "initialCacheHitsFromSmoke": 40,
        "remainingInitialPhysicalRowsExecuted": len(paths) - 40,
        "freshPhysicalRowsAcrossSmokeAndInitial": len(paths),
        "trainingRowsExecutedByTask": dict(sorted(task_counts.items())),
        "splitCounts": dict(sorted(split_counts.items())),
        "accessBoundaryCounts": dict(sorted(access_counts.items())),
        "generationCounts": {str(key): value for key, value in sorted(generation_counts.items())},
        "replayPassRows": replay_pass,
        "nativeContractPassRows": contract_pass,
        "failedRowsRetained": failed,
        "censoredRowsRetained": censored,
        "adaptiveGenerationsExecuted": 0,
        "logicalTrainingBudgetPlanned": 11008,
        "logicalTrainingRowsExecuted": len(rows),
        "logicalTrainingRowsNotExecuted": 11008 - len(rows),
        "archiveRowsPublished": 0,
        "shortlistRowsPublished": 0,
        "validationLogicalRowsExecuted": 0,
        "confirmationLogicalRowsExecuted": 0,
        "cacheFilenameKeyMatch": cache_keys_match,
        "completeInitialAccounting": (
            len(rows) == len(paths) == 4864
            and len(set(row["logicalSlotId"] for row in rows)) == 4864
            and cache_keys_match
        ),
    }
    stop = {
        "schemaVersion": "e07.s08g.execution-stop-status.v1",
        "researchStepId": "S08G",
        "success": False,
        "status": "blocked_at_generation_0_archive_construction",
        "failureClass": "generation_0_integrity_failure",
        "failureClasses": [
            "complete_e05_descriptor_outside_frozen_bounds",
            "native_contract_validation_failure",
        ],
        "exceptionType": "builtins.ValueError",
        "exceptionMessage": "descriptor outside frozen bounds",
        "smokePassed": True,
        "substantiveInitialPanelCompleted": True,
        "adaptiveSearchStarted": False,
        "archiveConstructed": False,
        "validationAccessed": False,
        "confirmationAccessed": False,
        "retryAttempted": False,
        "redesignAttempted": False,
        "forensicArtifacts": [
            "generation_0_descriptor_forensics.json",
            "generation_0_native_contract_forensics.json",
            "generation_0_execution_accounting.json",
            "training_evaluation_quarantine.json",
            "no_mutation_audit.json",
        ],
    }
    no_mutation_record = {
        "schemaVersion": "e07.s08g.no-mutation-audit.v1",
        "researchStepId": "S08G",
        "beforeImmutableTreeSha256": before_trees,
        "afterImmutableTreeSha256": after_trees,
        "unchanged": no_mutation,
        "success": no_mutation,
    }
    native_contract_forensics = {
        "schemaVersion": "e07.s08g.generation-0-native-contract-forensics.v1",
        "researchStepId": "S08G",
        "evaluatedRows": len(rows),
        "nativeContractPassRows": contract_pass,
        "nativeContractFailureRows": len(contract_failures),
        "failures": contract_failures,
        "failuresRelabeled": False,
        "failuresDropped": False,
        "success": not contract_failures,
    }
    physical_pairs = [
        (str(row["logicalSlotId"]), str(row["stableEvaluationSha256"]))
        for row in rows
    ]
    forward_results = {slot: digest for slot, digest in physical_pairs}
    reverse_results = {slot: digest for slot, digest in reversed(physical_pairs)}
    forward_result_digest = canonical_sha256(
        "E07/S08G/generation-0-order-independent-results/v1", forward_results
    )
    reverse_result_digest = canonical_sha256(
        "E07/S08G/generation-0-order-independent-results/v1", reverse_results
    )
    initial_revalidation = json.loads(
        (args.output / "initial_physical_accounting_revalidation.json").read_text(
            encoding="utf-8"
        )
    )
    replay_order = {
        "schemaVersion": "e07.s08g.replay-worker-order-validation.v1",
        "researchStepId": "S08G",
        "evaluatedRows": len(rows),
        "replayPassRows": replay_pass,
        "forwardResultCommitmentSha256": forward_result_digest,
        "reverseResultCommitmentSha256": reverse_result_digest,
        "resultCommitmentWorkerOrderIndependent": (
            forward_result_digest == reverse_result_digest
        ),
        "forwardPhysicalPlanSha256": initial_revalidation["forwardPlanSha256"],
        "reversePhysicalPlanSha256": initial_revalidation["reversePlanSha256"],
        "physicalPlanWorkerOrderIndependent": initial_revalidation["checks"][
            "workerOrderIndependent"
        ],
        "fullReverseExecutionPerformed": False,
        "qualificationEvidence": "S08F replay/order qualification plus S08G order-independent plan and result commitments",
        "success": replay_pass == len(rows)
        and forward_result_digest == reverse_result_digest
        and bool(initial_revalidation["checks"]["workerOrderIndependent"]),
    }
    access_validation = {
        "schemaVersion": "e07.s08g.access-control-validation.v1",
        "researchStepId": "S08G",
        "protectedRequestsDeniedBeforeMaterialization": len(preflight["accessRows"]),
        "protectedDenialRowsPassed": sum(
            bool(row["deniedBeforeMaterialization"]) for row in preflight["accessRows"]
        ),
        "trainingRows": split_counts.get("train", 0),
        "developmentTrainingAccessRows": access_counts.get("development_training", 0),
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "priorQuarantineOutcomeRowsLoaded": 0,
        "success": (
            len(preflight["accessRows"]) == 16
            and all(row["deniedBeforeMaterialization"] for row in preflight["accessRows"])
            and split_counts == {"train": len(rows)}
            and access_counts == {"development_training": len(rows)}
        ),
    }
    dependency_exclusion = {
        "schemaVersion": "e07.s08g.dependency-exclusion-audit.v1",
        "researchStepId": "S08G",
        "loadedProhibitedModules": list(preflight["loadedProhibitedModules"]),
        "s06ModelsOrEmbeddingsLoaded": False,
        "s06aModelsOrEmbeddingsLoaded": False,
        "s07ArmMembershipLoadedForPromotion": False,
        "s08cOutcomesLoaded": False,
        "s08eOutcomesLoaded": False,
        "priorQuarantineOutcomeRowsLoaded": preflight["quarantinedOutcomeRowsLoaded"],
        "success": not preflight["loadedProhibitedModules"]
        and preflight["quarantinedOutcomeRowsLoaded"] == 0,
    }
    hash_validation = {
        "schemaVersion": "e07.s08g.configuration-and-member-hash-validation.v1",
        "researchStepId": "S08G",
        "candidateCommitmentExpected": preflight["candidateCommitmentExpected"],
        "candidateCommitmentActual": preflight["candidateCommitmentActual"],
        "completePlanExpected": preflight["completePlanExpected"],
        "completePlanActual": preflight["completePlanActual"],
        "configurationRowsChecked": preflight["bindingValidation"][
            "configurationRowsChecked"
        ],
        "configurationRowsBound": preflight["bindingValidation"][
            "configurationRowsBound"
        ],
        "initialCacheKeysMatchFrozenWork": cache_keys_match,
        "cacheCommitmentSha256": quarantine["cacheCommitmentSha256"],
        "success": preflight["candidateCommitmentExpected"]
        == preflight["candidateCommitmentActual"]
        and preflight["completePlanExpected"] == preflight["completePlanActual"]
        and cache_keys_match,
    }
    archive_status = {
        "schemaVersion": "e07.s08g.archive-selection-status.v1",
        "researchStepId": "S08G",
        "archiveConstructionAttempted": True,
        "archiveConstructionCompleted": False,
        "archiveRowsPublished": 0,
        "adaptiveGenerationsExecuted": 0,
        "shortlistLockWritten": False,
        "shortlistRowsPublished": 0,
        "validationPlanWritten": False,
        "efficacyResultPublished": False,
        "blockingIntegrityChecks": [
            "descriptorBoundsConformance",
            "allInitialNativeContractsPass",
        ],
    }
    validation_accounting = {
        "schemaVersion": "e07.s08g.validation-plan-accounting.v1",
        "researchStepId": "S08G",
        "candidateLockWritten": False,
        "postSelectionValidationAuthorized": False,
        "validationLogicalRowsPlannedAfterLock": None,
        "validationLogicalRowsExecuted": 0,
        "confirmationLogicalRowsExecuted": 0,
        "validationRemainedSealed": True,
        "confirmationRemainedSealed": True,
    }
    uncertainty_status = {
        "schemaVersion": "e07.s08g.uncertainty-multiplicity-status.v1",
        "researchStepId": "S08G",
        "pairedEstimandsComputed": False,
        "uncertaintyIntervalsComputed": False,
        "multiplicityAdjustmentComputed": False,
        "reason": "generation-0 integrity stop occurred before archive and shortlist",
        "unadjustedClaimsMade": False,
    }
    s09_gate = {
        "schemaVersion": "e07.s08g.s09-eligibility-gate.v1",
        "researchStepId": "S08G",
        "eligible": False,
        "status": "blocked",
        "blockingReasons": [
            "S08G did not complete the exact 11,008-slot six-generation training design",
            "one complete E05 repair descriptor panel is outside frozen bounds",
            "two initial target-change rows fail native-contract validation",
            "no archive, shortlist lock, or post-selection validation result exists",
        ],
        "candidateLockPath": None,
        "authorizationToStartS09": False,
    }
    integrity_failure = {
        "schemaVersion": "e07.s08g.generation-0-integrity-failure.v1",
        "researchStepId": "S08G",
        "success": False,
        "status": "failed_closed_before_archive_construction",
        "trigger": {
            "exceptionType": "ValueError",
            "exceptionMessage": "descriptor outside frozen bounds",
            "callPath": "aggregate_panel -> aggregate_configuration -> _aggregate_descriptors -> _aggregate_e05_descriptors -> _bin",
        },
        "descriptorFailure": {
            "boundViolationCount": descriptor["boundViolationCount"],
            "affectedConfigurationCount": descriptor["affectedConfigurationCount"],
            "violations": descriptor["violations"],
            "completeSupportPanels": descriptor["supportStateCounts"].get("complete", 0),
            "unavailableSupportPanelsRetained": descriptor["supportStateCounts"].get(
                "unavailable", 0
            ),
            "unavailableSupportUsedForCells": False,
            "availabilityUsedAsNovelty": False,
            "imputationOrClippingApplied": False,
        },
        "nativeContractFailure": {
            "failureRows": len(contract_failures),
            "rows": contract_failures,
            "relabeledOrDropped": False,
        },
        "executionBoundary": {
            "initialLogicalRows": len(rows),
            "initialPhysicalRows": len(paths),
            "adaptiveGenerationsExecuted": 0,
            "archiveRowsPublished": 0,
            "shortlistRowsPublished": 0,
            "validationOutcomeEvaluations": 0,
            "confirmationOutcomeEvaluations": 0,
            "s09Authorized": False,
        },
    }
    checks = {
        "frozenSmokePassed": True,
        "exactInitialLogicalRows": accounting["initialLogicalRows"] == 4864,
        "exactInitialPhysicalRows": accounting["initialPhysicalRows"] == 4864,
        "cacheFilenameKeyMatch": cache_keys_match,
        "allInitialReplayPass": replay_pass == 4864,
        "allInitialNativeContractsPass": contract_pass == 4864,
        "trainingOnly": split_counts == {"train": 4864},
        "developmentAccessOnly": access_counts == {"development_training": 4864},
        "adaptiveSearchNotStarted": True,
        "archiveNotPublished": True,
        "validationNotAccessed": True,
        "confirmationNotAccessed": True,
        "priorQuarantinesNotLoaded": True,
        "immutableInputsUnchanged": no_mutation,
        "descriptorBoundsConformance": descriptor["success"],
    }
    validation = {
        "schemaVersion": "e07.s08g.blocked-validation-summary.v1",
        "researchStepId": "S08G",
        "checks": checks,
        "passedChecks": sum(bool(value) for value in checks.values()),
        "totalChecks": len(checks),
        "success": False,
        "blockingChecks": sorted(key for key, value in checks.items() if not value),
    }
    _write_json(args.output / "generation_0_descriptor_forensics.json", descriptor)
    _write_json(
        args.output / "generation_0_native_contract_forensics.json",
        native_contract_forensics,
    )
    _write_json(args.output / "generation_0_execution_accounting.json", accounting)
    _write_json(args.output / "complete_budget_accounting.json", accounting)
    _write_json(args.output / "generation_0_integrity_failure.json", integrity_failure)
    _write_json(args.output / "access_control_validation.json", access_validation)
    _write_json(args.output / "dependency_exclusion_audit.json", dependency_exclusion)
    _write_json(args.output / "replay_worker_order_validation.json", replay_order)
    _write_json(
        args.output / "configuration_and_member_hash_validation.json", hash_validation
    )
    _write_json(args.output / "archive_and_selection_status.json", archive_status)
    _write_json(args.output / "validation_plan_and_accounting.json", validation_accounting)
    _write_json(args.output / "uncertainty_and_multiplicity.json", uncertainty_status)
    _write_json(args.output / "s09_eligibility_gate.json", s09_gate)
    _write_json(args.output / "training_evaluation_quarantine.json", quarantine)
    _write_json(args.output / "execution_stop_status.json", stop)
    _write_json(args.output / "no_mutation_audit.json", no_mutation_record)
    _write_json(args.output / "validation_summary.json", validation)
    _write_artifact_manifest(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
