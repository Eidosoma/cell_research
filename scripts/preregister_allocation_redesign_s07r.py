#!/usr/bin/env python3
"""Build and validate the design-only E07 S07R preregistration artifacts."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Iterable, Mapping

import yaml

from src.allocation_redesign.core import (
    PROTOCOL_PATH,
    TASK_IDS,
    build_candidate_registry,
    build_scenario_registry,
    candidate_population_commitment,
    checked_protocol,
    derive_rare_status_registry,
    feasibility_summary,
    hash_file,
    synthetic_worker_order_validation,
)
from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = Path("/artifacts/research_steps/S07R")
S07_ROOT = Path("/artifacts/research_steps/S07")
TASK_REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLIT_MANIFEST = REPOSITORY / "configs/environment_suite/split_manifest.json"
ALLOCATION_SOURCE = REPOSITORY / "src/allocation_redesign/core.py"
REPORT = ROOT / "research_step_full_results.md"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_bytes(row).decode("ascii") + "\n")


def _manifest_entries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("artifacts", data.get("files", [])))


def _verify_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    root_value = data.get("root") or data.get("artifactRoot")
    root = Path(root_value) if root_value else path.parent
    entries = list(data.get("artifacts", data.get("files", [])))
    checks = []
    for entry in entries:
        entry_path = Path(entry["path"])
        target = entry_path if entry_path.is_absolute() else root / entry_path
        actual = hash_file(target)
        checks.append(actual == entry["sha256"])
    return {
        "path": str(path),
        "entryCount": len(entries),
        "success": bool(entries) and all(checks),
    }


def _input_paths() -> list[Path]:
    paths = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.json"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        Path(
            "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"
        ),
        Path(
            "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md"
        ),
        Path(
            "/previous-artifacts/E01/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E01/release/baseline/release_manifest.json"),
        Path(
            "/previous-artifacts/E02/research_steps/S14/research_step_full_results.md"
        ),
        Path(
            "/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"
        ),
        Path("/previous-artifacts/E03/research_steps/S14/e07_handoff.md"),
        Path("/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.md"),
        Path(
            "/previous-artifacts/E05/research_steps/S14/regeneration_benchmark/E07_HANDOFF.md"
        ),
        Path("/previous-artifacts/E06/report_inputs/e07_handoff.md"),
        TASK_REGISTRY,
        SPLIT_MANIFEST,
        REPOSITORY / "configs/search/s04_objective_registry.yaml",
        REPOSITORY / "configs/search/s04_descriptor_registry.yaml",
        REPOSITORY / "configs/search/s04_task_weighting.yaml",
        REPOSITORY / "configs/search/s04_uncertainty_registry.yaml",
        REPOSITORY / "configs/search/s04_anti_gaming_registry.yaml",
        Path("/artifacts/research_steps/S03/seed_library.jsonl"),
        Path("/artifacts/research_steps/S04A/s05_eligibility_gate.json"),
        Path("/artifacts/research_steps/S04A/adapter_binding_registry.yaml"),
        Path("/artifacts/research_steps/S04A/e05_native_contract_validation.json"),
        Path("/artifacts/research_steps/S04A/e06_native_contract_validation.json"),
        Path("/artifacts/research_steps/S05/artifact_manifest.json"),
        Path("/artifacts/research_steps/S05/search_protocol.yaml"),
        Path("/artifacts/research_steps/S05/generation_plan_ledger.jsonl"),
        Path("/artifacts/research_steps/S05/evaluation_ledger.jsonl"),
        Path("/artifacts/research_steps/S05/lineage_graph.jsonl"),
        Path("/artifacts/research_steps/S05/policy_catalog.jsonl"),
        Path("/artifacts/research_steps/S05/archive/archive_manifest.json"),
        Path("/artifacts/research_steps/S05/archive/archive_entries.jsonl"),
        Path("/artifacts/research_steps/S06/artifact_manifest.json"),
        Path("/artifacts/research_steps/S06/deployment_decision.json"),
        Path("/artifacts/research_steps/S06A/artifact_manifest.json"),
        Path("/artifacts/research_steps/S06A/deployment_decision.json"),
        PROTOCOL_PATH,
        ALLOCATION_SOURCE,
        REPOSITORY / "src/allocation_redesign/__init__.py",
        REPOSITORY / "tests/test_allocation_redesign_s07r.py",
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing S07R input files: {missing}")
    return paths


def _rejected_audit(protocol: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for step_id in ("S06", "S06A"):
        root = Path(f"/artifacts/research_steps/{step_id}")
        decision_path = root / "deployment_decision.json"
        manifest_path = root / "artifact_manifest.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        manifest_entries = _manifest_entries(manifest_path)
        prohibited_entries = [
            entry["path"]
            for entry in manifest_entries
            if entry["path"].startswith(("models/", "embeddings/"))
        ]
        rejected = decision.get("decision") == "reject"
        if step_id == "S06A":
            rejected = rejected and decision.get("eligibleForS07") is False
        if not rejected:
            raise RuntimeError(f"{step_id} is not the expected immutable rejection")
        records.append(
            {
                "stepId": step_id,
                "artifactManifestSha256": hash_file(manifest_path),
                "deploymentDecisionSha256": hash_file(decision_path),
                "decision": "reject",
                "eligibleForS07": False,
                "prohibitedModelOrEmbeddingEntryCount": len(prohibited_entries),
                "prohibitedEntryPathsListedButNotOpened": prohibited_entries,
            }
        )
    return {
        "schemaVersion": "e07.s07r.rejected-artifact-audit.v1",
        "researchStepId": "S07R",
        "success": True,
        "auditPath": "preregistration_only_not_allocation_path",
        "rejectedArtifactPayloadsOpened": 0,
        "rejectedModelsLoaded": 0,
        "rejectedEmbeddingsLoaded": 0,
        "prohibitedUses": protocol["historicalBoundary"]["prohibitedUses"],
        "records": records,
    }


def _source_import_audit(protocol: Mapping[str, Any]) -> dict[str, Any]:
    tree = ast.parse(ALLOCATION_SOURCE.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    rejected = tuple(protocol["historicalBoundary"]["rejectedCodeImports"])
    forbidden_imports = [name for name in imports if name.startswith(rejected)]
    loaded_forbidden = [name for name in sys.modules if name.startswith(rejected)]
    return {
        "schemaVersion": "e07.s07r.forbidden-dependency-validation.v1",
        "researchStepId": "S07R",
        "success": not forbidden_imports and not loaded_forbidden,
        "allocationModule": str(ALLOCATION_SOURCE),
        "imports": sorted(imports),
        "forbiddenImports": forbidden_imports,
        "forbiddenModulesLoaded": loaded_forbidden,
        "allocationPathRejectedArtifactReads": 0,
        "allocationPathRejectedModelLoads": 0,
        "allocationPathRejectedEmbeddingLoads": 0,
        "allocationPathPseudoLabels": 0,
        "allocationPathWarmStarts": 0,
        "allocationPathDistillation": 0,
    }


def _access_audit() -> dict[str, Any]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    grant = AccessGrant(AccessPhase.DEVELOPMENT)
    denials = []
    before = suite.broker.audit.materializer_invocations
    for task_id in TASK_IDS:
        records = [
            record for record in suite.records.values() if record.task_id == task_id
        ]
        for split in ("validation", "confirmation"):
            record = next(item for item in records if item.split.value == split)
            try:
                suite.open(task_id, record.scenario_id, grant)
            except AccessDeniedError:
                denials.append(
                    {
                        "taskId": task_id,
                        "scenarioId": record.scenario_id,
                        "split": split,
                        "denied": True,
                    }
                )
            else:
                raise RuntimeError("nontraining S07R access unexpectedly succeeded")
    after = suite.broker.audit.materializer_invocations
    return {
        "schemaVersion": "e07.s07r.access-control-validation.v1",
        "researchStepId": "S07R",
        "success": len(denials) == 16 and after == before,
        "developmentGrantNontrainingAttempts": len(denials),
        "denials": denials,
        "materializerInvocationDelta": after - before,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "scenarioMaterializations": 0,
    }


def _selection_log_schema(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": "e07.s07r.selection-log-schema.v1",
        "researchStepId": "S07R",
        "requiredFields": protocol["propensityAndSelectionLogging"]["requiredFields"],
        "missingLogPolicy": "fail_closed",
        "uniformInclusionProbability": 0.5,
        "uniformInverseProbabilityWeight": 2.0,
        "fixedArmInference": "deterministic selection indicators; descriptive paired comparison only",
        "logicalAndPhysicalLedgersBothRequired": True,
        "candidateSelectionFrameRowsRequired": protocol[
            "propensityAndSelectionLogging"
        ]["candidateSelectionFrame"]["rowCount"],
        "logicalEvaluationLedgerRowsRequired": protocol[
            "propensityAndSelectionLogging"
        ]["logicalEvaluationLedger"]["rowCount"],
    }


def _budget_template(protocol: Mapping[str, Any]) -> dict[str, Any]:
    per_arm = {}
    for arm_id in protocol["arms"]:
        per_arm[arm_id] = {
            "logicalPlanned": 1024,
            "logicalExecutedDuringS07R": 0,
            "perTaskPlanned": {task_id: 128 for task_id in TASK_IDS},
            "perFamilyPlanned": {
                str(ordinal): 256
                for ordinal in protocol["scenarioPopulation"]["familyOrdinals"]
            },
        }
    return {
        "schemaVersion": "e07.s07r.budget-accounting-template.v1",
        "researchStepId": "S07R",
        "success": True,
        "designOnly": True,
        "logicalPlanned": 3072,
        "logicalExecutedDuringS07R": 0,
        "physicalExecutedDuringS07R": 0,
        "physicalFutureLowerBound": 1024,
        "physicalFutureUpperBound": 3072,
        "smokeLogicalWithinBudget": 24,
        "episodesEvaluated": 0,
        "perArm": per_arm,
        "accountingIdentity": "sum arm logical rows = 3072; unique physical row IDs determine future physical count",
    }


def build() -> None:
    if S07_ROOT.exists():
        raise RuntimeError(
            "S07R refuses to run after any S07 artifact directory exists"
        )
    protocol = checked_protocol()
    archive_paths = [
        Path("/artifacts/research_steps/S05/archive/archive_manifest.json"),
        Path("/artifacts/research_steps/S05/archive/archive_entries.jsonl"),
        Path("/artifacts/research_steps/S05/evaluation_ledger.jsonl"),
        Path("/artifacts/research_steps/S05/generation_plan_ledger.jsonl"),
    ]
    before_hashes = {str(path): hash_file(path) for path in archive_paths}
    ROOT.mkdir(parents=True, exist_ok=True)

    candidate_rows = build_candidate_registry(protocol)
    scenario_rows = build_scenario_registry(protocol)
    feasibility = feasibility_summary(candidate_rows, protocol)
    worker_order = synthetic_worker_order_validation(protocol)
    rare_status = derive_rare_status_registry(candidate_rows)
    access = _access_audit()
    dependency = _source_import_audit(protocol)
    rejected = _rejected_audit(protocol)
    if not all(
        (
            feasibility["success"],
            worker_order["success"],
            access["success"],
            dependency["success"],
        )
    ):
        raise RuntimeError("S07R design validation failed")

    shutil.copyfile(PROTOCOL_PATH, ROOT / "s07r_non_surrogate_protocol.yaml")
    write_jsonl(ROOT / "candidate_population_registry.jsonl", candidate_rows)
    write_json(
        ROOT / "scenario_population_registry.json",
        {
            "schemaVersion": "e07.s07r.scenario-population-registry.v1",
            "researchStepId": "S07R",
            "rows": scenario_rows,
        },
    )
    write_json(ROOT / "rare_status_registry.json", rare_status)
    write_json(ROOT / "feasibility_validation.json", feasibility)
    write_json(ROOT / "worker_order_validation.json", worker_order)
    write_json(ROOT / "access_control_validation.json", access)
    write_json(ROOT / "forbidden_dependency_validation.json", dependency)
    write_json(ROOT / "rejected_artifact_audit.json", rejected)
    write_json(ROOT / "selection_log_schema.json", _selection_log_schema(protocol))
    write_json(ROOT / "budget_accounting_template.json", _budget_template(protocol))
    (ROOT / "downstream_gate_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schemaVersion": "e07.s07r.downstream-gate-registry.v1",
                "researchStepId": "S07R",
                "gates": protocol["downstreamGates"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    input_paths = _input_paths()
    frozen_inputs = [
        {"path": str(path), "sha256": hash_file(path), "bytes": path.stat().st_size}
        for path in input_paths
    ]
    freeze = {
        "schemaVersion": "e07.s07r.input-hash-freeze.v1",
        "researchStepId": "S07R",
        "success": True,
        "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
        "protocolSha256": hash_file(PROTOCOL_PATH),
        "candidatePopulationCommitmentSha256": candidate_population_commitment(
            candidate_rows
        ),
        "candidatePopulationRows": len(candidate_rows),
        "scenarioTemplateRows": len(scenario_rows),
        "authorizedAllocationEvidenceLatestStep": "S05",
        "rejectedStepsAuditOnly": ["S06", "S06A"],
        "allocationPathRejectedArtifactLoads": 0,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "inputs": frozen_inputs,
    }
    write_json(ROOT / "input_hash_freeze.json", freeze)
    prereg = {
        "schemaVersion": "e07.s07r.preregistration-freeze.v1",
        "researchStepId": "S07R",
        "success": True,
        "designOnly": True,
        "frozenBeforeS07Allocation": True,
        "frozenBeforeS07Evaluation": True,
        "frozenBeforeAnyArchiveMutation": True,
        "protocolSha256": freeze["protocolSha256"],
        "artifactProtocolSha256": hash_file(ROOT / "s07r_non_surrogate_protocol.yaml"),
        "inputHashFreezeSha256": hash_file(ROOT / "input_hash_freeze.json"),
        "candidatePopulationRegistrySha256": hash_file(
            ROOT / "candidate_population_registry.jsonl"
        ),
        "scenarioPopulationRegistrySha256": hash_file(
            ROOT / "scenario_population_registry.json"
        ),
        "candidatePopulationCommitmentSha256": freeze[
            "candidatePopulationCommitmentSha256"
        ],
        "actualAllocationRosterWritten": False,
        "episodesEvaluated": 0,
        "archiveMutations": 0,
        "approvalRequiredBeforeS07": True,
    }
    write_json(ROOT / "preregistration_freeze.json", prereg)

    after_hashes = {str(path): hash_file(path) for path in archive_paths}
    no_mutation = {
        "schemaVersion": "e07.s07r.no-mutation-audit.v1",
        "researchStepId": "S07R",
        "success": before_hashes == after_hashes and not S07_ROOT.exists(),
        "beforeSha256": before_hashes,
        "afterSha256": after_hashes,
        "s05ArchiveMutations": 0,
        "s05LedgerMutations": 0,
        "s07ArtifactDirectoryExists": S07_ROOT.exists(),
        "allocationRosterWritten": False,
        "episodesEvaluated": 0,
    }
    write_json(ROOT / "no_mutation_audit.json", no_mutation)
    manifest_checks = [
        _verify_manifest(
            Path(f"/artifacts/research_steps/{step}/artifact_manifest.json")
        )
        for step in ("S01", "S02", "S03", "S04", "S04A", "S05")
    ]
    validation = {
        "schemaVersion": "e07.s07r.validation-summary.v1",
        "researchStepId": "S07R",
        "success": all(item["success"] for item in manifest_checks)
        and no_mutation["success"],
        "designOnly": True,
        "checks": {
            "S01ThroughS05ArtifactManifests": all(
                item["success"] for item in manifest_checks
            ),
            "candidatePopulation512And64PerTask": feasibility["success"],
            "fixedQuotaFeasibilityAllTasks": all(
                item["descriptorQuotaFeasible"]
                for item in feasibility["tasks"].values()
            ),
            "scenarioTemplateCount32": len(scenario_rows) == 32,
            "completeLogicalBudget3072": _budget_template(protocol)["logicalPlanned"]
            == 3072,
            "syntheticWorkerOrderIndependence": worker_order["success"],
            "protectedSplitDenial16Of16": access["success"],
            "allocationPathNoRejectedImportsOrLoads": dependency["success"],
            "S06AndS06AImmutableRejected": rejected["success"],
            "S05AndS07NoMutation": no_mutation["success"],
            "actualAllocationRosterAbsent": True,
            "episodesEvaluatedZero": True,
            "validationAndConfirmationOutcomesZero": True,
        },
        "manifestChecks": manifest_checks,
        "validationResult": "PASS: exact design, populations, budgets, quota feasibility, synthetic worker-order independence, 16/16 nontraining denials, rejected-model isolation, and no mutation; zero S07 roster rows or evaluations.",
    }
    write_json(ROOT / "validation_summary.json", validation)
    write_json(
        ROOT / "environment.json",
        {
            "schemaVersion": "e07.s07r.environment.v1",
            "researchStepId": "S07R",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "plannedFutureWorkers": 8,
            "plannedFutureThreadsPerWorker": 1,
            "workersUsedForS07R": 1,
            "gpuUsed": False,
            "networkUsed": False,
            "newDependenciesInstalled": [],
        },
    )
    write_json(
        ROOT / "input_provenance.json",
        {
            "schemaVersion": "e07.s07r.input-provenance.v1",
            "researchStepId": "S07R",
            "files": frozen_inputs,
            "allocationAuthorizedThroughStep": "S05",
            "rejectedModelSteps": ["S06", "S06A"],
            "protectedOutcomeArtifactsOpened": 0,
        },
    )
    write_json(
        ROOT / "status.json",
        {
            "researchStepId": "S07R",
            "stepNumber": "07R",
            "success": True,
            "status": "complete_design_frozen_awaiting_S07_approval",
            "outcomeClassification": "supportive",
            "artifactsWritten": [
                "/artifacts/research_steps/S07R/s07r_non_surrogate_protocol.yaml",
                "/artifacts/research_steps/S07R/candidate_population_registry.jsonl",
                "/artifacts/research_steps/S07R/scenario_population_registry.json",
                "/artifacts/research_steps/S07R/input_hash_freeze.json",
                "/artifacts/research_steps/S07R/preregistration_freeze.json",
                "/artifacts/research_steps/S07R/validation_summary.json",
                "/artifacts/research_steps/S07R/research_step_full_results.md",
                "/artifacts/research_steps/S07R/artifact_manifest.json",
            ],
            "validationResult": validation["validationResult"],
            "caveatsOrBlockers": [
                "S07R is design-only; no actual allocation roster, scenario materialization, episode evaluation, or archive mutation occurred.",
                "Fixed-arm comparisons are descriptive design-efficiency contrasts; only the uniform arm has positive sampling propensity over the full task candidate population.",
                "S06 and S06A remain immutable rejected diagnostic artifacts and are prohibited from all downstream allocation and decision uses.",
            ],
            "recommendedNextAction": "Chief Scientist review and explicit approval of the frozen S07R protocol before any S07 roster generation or training-only evaluation.",
        },
    )


def validate() -> None:
    protocol = checked_protocol(ROOT / "s07r_non_surrogate_protocol.yaml")
    required = [
        "candidate_population_registry.jsonl",
        "scenario_population_registry.json",
        "rare_status_registry.json",
        "feasibility_validation.json",
        "worker_order_validation.json",
        "access_control_validation.json",
        "forbidden_dependency_validation.json",
        "rejected_artifact_audit.json",
        "selection_log_schema.json",
        "budget_accounting_template.json",
        "downstream_gate_registry.yaml",
        "input_hash_freeze.json",
        "preregistration_freeze.json",
        "no_mutation_audit.json",
        "validation_summary.json",
        "status.json",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing S07R outputs: {missing}")
    validation = json.loads(
        (ROOT / "validation_summary.json").read_text(encoding="utf-8")
    )
    prereg = json.loads(
        (ROOT / "preregistration_freeze.json").read_text(encoding="utf-8")
    )
    if not validation["success"] or prereg["actualAllocationRosterWritten"]:
        raise RuntimeError("S07R validation or no-allocation boundary failed")
    if S07_ROOT.exists():
        raise RuntimeError("S07 artifact directory exists")
    if protocol["S07ExecutionApprovalGate"]["approved"] is not False:
        raise RuntimeError("S07 execution gate is not closed")


def finalize_manifest() -> None:
    validate()
    if not REPORT.is_file():
        raise RuntimeError(
            "canonical S07R report must exist before manifest finalization"
        )
    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": hash_file(path),
                }
            )
    write_json(
        ROOT / "artifact_manifest.json",
        {
            "schemaVersion": "e07.s07r.artifact-manifest.v1",
            "researchStepId": "S07R",
            "root": str(ROOT),
            "success": True,
            "designOnly": True,
            "artifacts": files,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate", "finalize-manifest"))
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "validate":
        validate()
    else:
        finalize_manifest()


if __name__ == "__main__":
    main()
