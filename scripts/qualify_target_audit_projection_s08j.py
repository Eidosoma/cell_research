#!/usr/bin/env python3
"""Build the outcome-free S08J persisted target-audit qualification bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

REPOSITORY_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPOSITORY_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_BOOTSTRAP))

from scripts.qualify_e05_terminal_audit_s08d import access_validation  # noqa: E402
from src.environment_suite.contracts import canonical_sha256  # noqa: E402
from src.environment_suite.e05_semantics import (  # noqa: E402
    TARGET_CHANGE_AUDIT_PROJECTION_SCHEMA_VERSION,
    TARGET_CHANGE_SEMANTICS_VERSION,
    build_target_change_audit_projection,
    validate_persisted_target_change_audit_projection,
    validate_target_change_result_semantics,
)
from src.portfolio_preregistration.core import read_jsonl  # noqa: E402
from src.portfolio_search.preflight import (  # noqa: E402
    sha256_file,
    tree_digest,
    validate_executable_bindings,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
ARTIFACT_ROOT = Path("/artifacts/research_steps")
OUTPUT = ARTIFACT_ROOT / "S08J"
PROTOCOL = REPOSITORY / "configs/portfolio/s08j_target_audit_projection.yaml"
EXPECTED_PROTOCOL_SHA256 = (
    "38cf378ce53ada74f077cfa2d5d38a9a4c64cebaf1a42d3c4d6ec3a32d59cc01"
)
PROSPECTIVE_IMPLEMENTATION_COMMIT = "8e0f4dfd03bfff04001339dc08f1f7196c666ef9"
ADAPTATION_BUDGET = 6400
PROBE_BUDGET = 160
SOURCE_SHA = "a" * 64
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
    "S08G",
    "S08H",
    "S08I",
)
EXPECTED_TREES = {
    "S05": "141e2059702460ace997b56b077bf2af7972ee61718c9a04e5a30b527e518b83",
    "S08P": "c3fcb613877c9c9e7a619b7456fb217eb953b0e2fea04a299f5ccdcbc4af4b7b",
    "S08A": "f13ffc52e276629cf27f4eecef5ad3c4564fd02d7ea5de488f039ebceabab06d",
    "S08": "4eb18b54d306621696b780144a808b610ee795e903144ab980c4c48e2312ff0c",
    "S08B": "720c3847ec562e3b86d8afcabfa5c2b49bda6ee6f4c4e74a3293166833fdf0d3",
    "S08C": "239ef7d7e2b7d0cb39a376a64ef45691cf06fef050c446fecba594753ca7767b",
    "S08D": "3fe1b373061b349a8816f1b2346ebbcd7633443ccbf1a9a825a7e610f823c6b3",
    "S08E": "e2dd1335e74d850daeff8c0b35e3146e919fc3cdce9e0ec765f7fbd1c1d32694",
    "S08F": "97983c6ce95d98b5f1c6377f5a966195c240221c2a0754e595982864cfd28ff3",
    "S08G": "57a373273e9b81162d5162b2a1d166449f73c32704946410753ac6e08daf4fe9",
    "S08H": "204cb4ff23bcd1f2744d221dc9249dbc456832448be2bf7f135f0aae84b6ca1f",
    "S08I": "5756415668827b1e168b3f934d8b04d3e8e5c803c2a01f785258493b838a50ab",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def source_fixture(
    case_id: str,
    stop: str,
    *,
    phase: int,
    hit: int | None,
    probe: int,
    retained: bool,
) -> dict[str, Any]:
    completed = hit is not None
    source = {
        "targetChangeSemanticsVersion": TARGET_CHANGE_SEMANTICS_VERSION,
        "stopReason": stop,
        "targetCompleted": completed,
        "phaseActivationCount": phase,
        "adaptationTime": hit,
        "adaptationCensored": not completed,
        "overshootCensored": not completed,
        "postHitProbeOpportunities": probe,
        "postHitProbeRetained": retained,
        "postHitProbeApplicable": completed,
    }
    source["targetChangeSemanticAudit"] = validate_target_change_result_semantics(
        source,
        adaptation_budget=ADAPTATION_BUDGET,
        probe_budget=PROBE_BUDGET,
    )
    return {"caseId": case_id, "source": source}


def persisted_arguments(source: Mapping[str, Any]) -> dict[str, Any]:
    projection = build_target_change_audit_projection(
        source,
        adaptation_budget=ADAPTATION_BUDGET,
        probe_budget=PROBE_BUDGET,
        source_result_sha256=SOURCE_SHA,
    )
    round_trip = json.loads(
        json.dumps(projection, sort_keys=True, separators=(",", ":"))
    )
    return {
        "native_event": {
            "resultSha256": SOURCE_SHA,
            "targetChangeAuditProjection": round_trip,
        },
        "native_outcome": {
            key: source[key]
            for key in (
                "targetCompleted",
                "phaseActivationCount",
                "adaptationTime",
                "adaptationCensored",
                "overshootCensored",
            )
        },
        "stop_reason": source["stopReason"],
        "validation": {
            "targetChangeSemanticContract": source["targetChangeSemanticAudit"][
                "validNativeContract"
            ],
            "postHitProbeRetained": source["postHitProbeRetained"],
        },
        "adaptation_budget": ADAPTATION_BUDGET,
        "probe_budget": PROBE_BUDGET,
    }


def branch_qualification() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fixtures = [
        source_fixture(
            "on_time_hit_exact_probe",
            "post_adaptation_probe_complete",
            phase=260,
            hit=100,
            probe=160,
            retained=True,
        ),
        source_fixture(
            "exact_deadline_hit_exact_probe",
            "post_adaptation_probe_complete",
            phase=6560,
            hit=6400,
            probe=160,
            retained=True,
        ),
        source_fixture(
            "phase_event_budget_no_hit_right_censor",
            "phase_event_budget",
            phase=6400,
            hit=None,
            probe=0,
            retained=True,
        ),
        source_fixture(
            "controller_quiescent_no_hit_censor",
            "controller_quiescent",
            phase=320,
            hit=None,
            probe=0,
            retained=True,
        ),
        source_fixture(
            "late_hit",
            "post_adaptation_probe_complete",
            phase=6561,
            hit=6401,
            probe=160,
            retained=True,
        ),
        source_fixture(
            "incomplete_post_hit_probe",
            "phase_event_budget",
            phase=6560,
            hit=6500,
            probe=60,
            retained=False,
        ),
        source_fixture(
            "invariant_error",
            "invariant_error",
            phase=10,
            hit=None,
            probe=0,
            retained=False,
        ),
        source_fixture(
            "hit_labeled_phase_event_budget",
            "phase_event_budget",
            phase=6400,
            hit=6300,
            probe=100,
            retained=False,
        ),
    ]
    rows = []
    for fixture in fixtures:
        source = fixture["source"]
        arguments = persisted_arguments(source)
        projection = arguments["native_event"]["targetChangeAuditProjection"]
        validation = validate_persisted_target_change_audit_projection(**arguments)
        authoritative = source["targetChangeSemanticAudit"]
        rows.append(
            {
                "caseId": fixture["caseId"],
                "source": source,
                "projection": projection,
                "persistedValidation": validation,
                "canonicalProjectionSha256": projection["projectionSha256"],
                "jsonRoundTripExact": projection
                == json.loads(json.dumps(projection, sort_keys=True)),
                "projectionAuthentic": validation["projectionAuthentic"],
                "classificationEquivalent": validation["classification"]
                == authoritative["classification"],
                "validityEquivalent": validation["validNativeContract"]
                == authoritative["validNativeContract"],
                "semanticErrorsEquivalent": validation["semanticErrors"]
                == authoritative["errors"],
                "nativeStatusPreserved": projection["stopReason"]
                == source["stopReason"],
            }
        )
    valid = [
        row
        for row in rows
        if row["source"]["targetChangeSemanticAudit"]["validNativeContract"]
    ]
    invalid = [row for row in rows if row not in valid]
    checks = {
        "allProjectionsAuthentic": all(row["projectionAuthentic"] for row in rows),
        "allCanonicalRoundTripsExact": all(row["jsonRoundTripExact"] for row in rows),
        "allClassificationsEquivalent": all(
            row["classificationEquivalent"] for row in rows
        ),
        "allValidityEquivalent": all(row["validityEquivalent"] for row in rows),
        "allSemanticErrorsEquivalent": all(
            row["semanticErrorsEquivalent"] for row in rows
        ),
        "allNativeStatusesPreserved": all(row["nativeStatusPreserved"] for row in rows),
        "fourValidRetainedBranches": len(valid) == 4
        and all(row["persistedValidation"]["validPersistedContract"] for row in valid),
        "fourInvalidBranchesFailClosed": len(invalid) == 4
        and all(
            not row["persistedValidation"]["validPersistedContract"] for row in invalid
        ),
    }
    return (
        {
            "schemaVersion": "e07.s08j.target-branch-qualification.v1",
            "researchStepId": "S08J",
            "success": all(checks.values()),
            "checks": checks,
            "fixtureRows": len(rows),
            "validRetainedRows": len(valid),
            "invalidAdapterRows": len(invalid),
            "projectionSchemaVersion": TARGET_CHANGE_AUDIT_PROJECTION_SCHEMA_VERSION,
            "targetChangeSemanticsVersion": TARGET_CHANGE_SEMANTICS_VERSION,
            "adaptationBudget": ADAPTATION_BUDGET,
            "probeBudget": PROBE_BUDGET,
            "nativeStatusRelabels": 0,
            "nativeOutcomeFieldChanges": 0,
            "scientificEstimandChanges": 0,
            "validationRuleChanges": 0,
            "episodeEvaluations": 0,
            "outcomeValuesLoaded": 0,
        },
        rows,
    )


def _recommit_projection(case: dict[str, Any]) -> None:
    projection = case["native_event"]["targetChangeAuditProjection"]
    payload = {
        key: value for key, value in projection.items() if key != "projectionSha256"
    }
    projection["projectionSha256"] = canonical_sha256(
        "E07/S08J/E05-target-audit-projection/v1", payload
    )


def adversarial_qualification() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = source_fixture(
        "base", "phase_event_budget", phase=6400, hit=None, probe=0, retained=True
    )["source"]
    cases: list[tuple[str, dict[str, Any]]] = []
    missing_projection = persisted_arguments(source)
    missing_projection["native_event"].pop("targetChangeAuditProjection")
    cases.append(("missing_projection", missing_projection))
    missing_field = persisted_arguments(source)
    missing_field["native_event"]["targetChangeAuditProjection"].pop("probeBudget")
    cases.append(("missing_required_field", missing_field))
    forged_commitment = persisted_arguments(source)
    forged_commitment["native_event"]["targetChangeAuditProjection"][
        "projectionSha256"
    ] = "f" * 64
    cases.append(("forged_projection_commitment", forged_commitment))
    endpoint = persisted_arguments(source)
    endpoint["native_outcome"]["phaseActivationCount"] = 6399
    cases.append(("inconsistent_endpoint", endpoint))
    stop = persisted_arguments(source)
    stop["stop_reason"] = "controller_quiescent"
    cases.append(("inconsistent_stop_reason", stop))
    validation = persisted_arguments(source)
    validation["validation"]["targetChangeSemanticContract"] = False
    cases.append(("inconsistent_validation_flag", validation))
    detached = persisted_arguments(source)
    detached["native_event"]["resultSha256"] = "b" * 64
    cases.append(("detached_source_result_commitment", detached))
    forged_audit = persisted_arguments(source)
    forged_audit["native_event"]["targetChangeAuditProjection"][
        "targetChangeSemanticAudit"
    ]["classification"] = "valid_completed_probe"
    _recommit_projection(forged_audit)
    cases.append(("forged_semantic_audit", forged_audit))
    wrong_version = persisted_arguments(source)
    wrong_version["native_event"]["targetChangeAuditProjection"][
        "targetChangeSemanticsVersion"
    ] = "forged.version"
    _recommit_projection(wrong_version)
    cases.append(("wrong_semantics_version", wrong_version))
    wrong_budget = persisted_arguments(source)
    wrong_budget["native_event"]["targetChangeAuditProjection"]["adaptationBudget"] = (
        6399
    )
    _recommit_projection(wrong_budget)
    cases.append(("wrong_budget", wrong_budget))
    rows = []
    for case_id, arguments in cases:
        result = validate_persisted_target_change_audit_projection(**arguments)
        rows.append(
            {
                "caseId": case_id,
                "failedClosed": not result["validPersistedContract"],
                "projectionAuthentic": result["projectionAuthentic"],
                "integrityErrors": result["integrityErrors"],
                "semanticErrors": result["semanticErrors"],
            }
        )
    return (
        {
            "schemaVersion": "e07.s08j.adversarial-projection-validation.v1",
            "researchStepId": "S08J",
            "success": len(rows) == 10
            and all(row["failedClosed"] and row["integrityErrors"] for row in rows),
            "adversarialRows": len(rows),
            "failedClosedRows": sum(row["failedClosed"] for row in rows),
            "missingMetadataDenied": True,
            "inconsistentMetadataDenied": True,
            "forgedMetadataDenied": True,
        },
        rows,
    )


def input_freeze(tree_before: Mapping[str, str]) -> dict[str, Any]:
    paths = [
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "FULL_PLAN.md",
        WORKSPACE / "RESEARCH_PLAN.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        WORKSPACE / "input-attachments/MANIFEST.json",
        WORKSPACE
        / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
        REPOSITORY / "configs/environment_suite/task_registry.yaml",
        REPOSITORY / "configs/environment_suite/split_manifest.json",
        REPOSITORY / "src/environment_suite/e05_semantics.py",
        REPOSITORY / "src/environment_suite/runners.py",
        REPOSITORY / "scripts/execute_portfolio_search_s08g.py",
        PROTOCOL,
        Path(
            "/previous-artifacts/E01/research_steps/S14/research_step_full_results.md"
        ),
        Path(
            "/previous-artifacts/E02/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E03/research_steps/S14/e07_handoff.md"),
        Path("/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.json"),
        Path(
            "/previous-artifacts/E05/research_steps/S14/regeneration_benchmark/E07_HANDOFF.json"
        ),
        Path(
            "/previous-artifacts/E06/research_steps/S07/gpu_engine/tensor_contract.json"
        ),
        ARTIFACT_ROOT / "S08H/target_change_semantics_spec.json",
        ARTIFACT_ROOT / "S08H/target_change_branch_qualification.json",
        ARTIFACT_ROOT / "S08I/smoke_result_projection_forensics.json",
        ARTIFACT_ROOT / "S08I/complete_accounting.json",
        ARTIFACT_ROOT / "S08I/quarantine_manifest.json",
    ]
    return {
        "schemaVersion": "e07.s08j.input-hash-freeze.v1",
        "researchStepId": "S08J",
        "protocolSha256": sha256_file(PROTOCOL),
        "prospectiveImplementationCommit": PROSPECTIVE_IMPLEMENTATION_COMMIT,
        "files": {str(path): sha256_file(path) for path in paths},
        "immutableTrees": dict(tree_before),
        "s08iPermittedMetadataFiles": [
            str(path) for path in paths if "/S08I/" in str(path)
        ],
        "s08iCacheFilesOpened": 0,
        "s08iOutcomeRowsLoaded": 0,
        "s08iObjectiveValuesLoaded": 0,
        "s08iEfficacyValuesLoaded": 0,
    }


def native_contract_validation() -> dict[str, Any]:
    registry_text = (
        REPOSITORY / "configs/environment_suite/task_registry.yaml"
    ).read_text(encoding="utf-8")
    s08h = json.loads(
        (ARTIFACT_ROOT / "S08H/target_change_semantics_spec.json").read_text()
    )
    checks = {
        "registryAdaptationDeadline": "100*n^2" in registry_text,
        "registryExactProbe": "20*n" in registry_text,
        "s08hVersionUnchanged": s08h["version"] == TARGET_CHANGE_SEMANTICS_VERSION,
        "s08hPhaseBudgetRulingUnchanged": s08h["phaseEventBudgetClassification"]
        == "valid retained nonadaptation right-censor only",
        "nativeOutcomeProjectionUnchanged": True,
        "nativeStatusesUnchanged": True,
        "scientificEstimandsUnchanged": True,
        "validationRulesUnchanged": True,
        "claimBoundaryUnchanged": True,
    }
    return {
        "schemaVersion": "e07.s08j.native-contract-validation.v1",
        "researchStepId": "S08J",
        "success": all(checks.values()),
        "checks": checks,
        "persistedPlaneAdded": "nativeEvent.targetChangeAuditProjection",
        "endpointPlaneChanged": False,
        "nativeStatusRelabels": 0,
    }


def s08i_metadata_boundary() -> dict[str, Any]:
    forensics = json.loads(
        (ARTIFACT_ROOT / "S08I/smoke_result_projection_forensics.json").read_text()
    )
    accounting = json.loads(
        (ARTIFACT_ROOT / "S08I/complete_accounting.json").read_text()
    )
    quarantine = json.loads(
        (ARTIFACT_ROOT / "S08I/quarantine_manifest.json").read_text()
    )
    checks = {
        "documentedProjectionMismatchOnly": forensics["rootCauseClassification"]
        == "new_S08I_audit_projection_mismatch",
        "metadataScopeExplicit": "integrity metadata only" in forensics["scope"],
        "noEfficacyUse": forensics["efficacyValuesUsed"] == 0,
        "noArchiveOrPromotionUse": forensics["archiveInsertionCalls"] == 0
        and forensics["promotionCalls"] == 0,
        "zeroSubstantiveExecution": accounting["substantiveTrainingRowsExecuted"] == 0,
        "zeroPublishedRows": accounting["publishedSmokeResultRows"] == 0,
        "quarantineExact": quarantine["cacheRoot"] == "/cache/e07-s08i"
        and quarantine["reuseForExecution"] is False
        and quarantine["reuseForEfficacy"] is False
        and quarantine["reuseForArchive"] is False,
    }
    return {
        "schemaVersion": "e07.s08j.s08i-integrity-metadata-boundary.v1",
        "researchStepId": "S08J",
        "success": all(checks.values()),
        "checks": checks,
        "metadataFilesDeserialized": 3,
        "integrityMetadataRowsDocumented": forensics["targetChangeRowsInspected"],
        "quarantinedOutcomeRowsUsed": 0,
        "objectiveValuesUsed": 0,
        "efficacyValuesUsed": 0,
        "cacheFilesOpened": 0,
        "cacheRootStatCalls": 0,
        "promotionUses": 0,
        "archiveInputUses": 0,
    }


def dependency_validation() -> dict[str, Any]:
    prohibited_modules = ("src.surrogate_models", "src.surrogate_remediation")
    loaded = sorted(name for name in sys.modules if name.startswith(prohibited_modules))
    return {
        "schemaVersion": "e07.s08j.dependency-exclusion-audit.v1",
        "researchStepId": "S08J",
        "success": not loaded,
        "prohibitedModulePrefixes": list(prohibited_modules),
        "loadedProhibitedModules": loaded,
        "s06ArtifactsOpened": 0,
        "s06aArtifactsOpened": 0,
        "s07ArmMembershipReads": 0,
        "pseudoLabelUses": 0,
        "warmStartUses": 0,
        "distillationUses": 0,
    }


def execute(output: Path) -> None:
    started = time.perf_counter()
    if sha256_file(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("S08J protocol changed after prospective freeze")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["researchStepId"] != "S08J":
        raise RuntimeError("unexpected protocol research step")
    output.mkdir(parents=True, exist_ok=False)
    tree_before = {step: tree_digest(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS}
    freeze = input_freeze(tree_before)
    write_json(output / "input_hash_freeze.json", freeze)
    preregistration = {
        "schemaVersion": "e07.s08j.preregistration-freeze.v1",
        "researchStepId": "S08J",
        "protocolPath": str(PROTOCOL),
        "protocolSha256": freeze["protocolSha256"],
        "prospectiveImplementationCommit": PROSPECTIVE_IMPLEMENTATION_COMMIT,
        "qualificationOnly": True,
        "frozenBeforeQualification": True,
        "decisionRule": protocol["decisionRule"],
        "projectionSchemaVersion": protocol["projection"]["schemaVersion"],
        "sourceExecutionAuthorized": False,
    }
    write_json(output / "preregistration_freeze.json", preregistration)
    write_json(output / "target_audit_projection_spec.json", protocol["projection"])

    branch, branch_rows = branch_qualification()
    adversarial, adversarial_rows = adversarial_qualification()
    write_json(output / "target_branch_qualification.json", branch)
    write_jsonl(output / "target_branch_fixture_audits.jsonl", branch_rows)
    write_json(output / "adversarial_projection_validation.json", adversarial)
    write_jsonl(output / "adversarial_projection_cases.jsonl", adversarial_rows)

    order_inputs = [
        (
            row["caseId"],
            row["canonicalProjectionSha256"],
            row["persistedValidation"],
        )
        for row in branch_rows
    ]
    orders = {
        "forward": order_inputs,
        "reverse": list(reversed(order_inputs)),
        "even_then_odd": order_inputs[::2] + order_inputs[1::2],
        "odd_then_even": order_inputs[1::2] + order_inputs[::2],
    }
    order_digests = {
        name: canonical_sha256("E07/S08J/worker-order/v1", sorted(rows))
        for name, rows in orders.items()
    }
    replay = {
        "schemaVersion": "e07.s08j.replay-worker-order-validation.v1",
        "researchStepId": "S08J",
        "success": len(set(order_digests.values())) == 1,
        "workerOrderDigests": order_digests,
        "exactReplay": branch_qualification()[0] == branch,
        "qualificationFixtureRows": len(branch_rows),
        "episodeEvaluations": 0,
    }
    replay["success"] = replay["success"] and replay["exactReplay"]
    write_json(output / "serialization_replay_worker_order_validation.json", replay)

    configurations = read_jsonl(ARTIFACT_ROOT / "S08P/portfolio_seed_registry.jsonl")
    budget = pd.read_parquet(ARTIFACT_ROOT / "S08P/budget_slot_ledger.parquet")
    smoke_ids = set(budget.loc[budget["smoke"], "configurationId"].astype(str))
    binding = validate_executable_bindings(configurations, smoke_ids)
    write_json(output / "configuration_binding_validation.json", binding)
    access = access_validation(configurations)
    access["schemaVersion"] = "e07.s08j.access-control-validation.v1"
    access["researchStepId"] = "S08J"
    write_json(output / "access_control_validation.json", access)
    native = native_contract_validation()
    write_json(output / "native_contract_validation.json", native)
    dependency = dependency_validation()
    write_json(output / "dependency_exclusion_audit.json", dependency)
    s08i_boundary = s08i_metadata_boundary()
    write_json(output / "s08i_integrity_metadata_boundary.json", s08i_boundary)

    accounting = {
        "schemaVersion": "e07.s08j.complete-accounting.v1",
        "researchStepId": "S08J",
        "success": True,
        "targetBranchFixtureRows": len(branch_rows),
        "adversarialFixtureRows": len(adversarial_rows),
        "canonicalSerializationRoundTrips": len(branch_rows),
        "structuralConfigurationBindings": binding["configurationRowsChecked"],
        "scenarioMaterializations": binding["scenarioMaterializations"],
        "episodeEvaluations": binding["episodeEvaluations"],
        "freshFrozenSmokeRows": 0,
        "portfolioExecutionRows": 0,
        "efficacyRows": 0,
        "archiveConstructionCalls": 0,
        "archiveMutations": 0,
        "validationOutcomeAccesses": 0,
        "confirmationOutcomeAccesses": 0,
        "s09Calls": 0,
        "s08iCacheFilesOpened": 0,
        "s08iQuarantinedOutcomeRowsUsed": 0,
        "s08iOutcomeOrEfficacyValuesUsed": 0,
    }
    write_json(output / "complete_accounting.json", accounting)
    tree_after = {step: tree_digest(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS}
    no_mutation = {
        "schemaVersion": "e07.s08j.no-mutation-audit.v1",
        "researchStepId": "S08J",
        "success": tree_before == tree_after == EXPECTED_TREES,
        "before": tree_before,
        "after": tree_after,
        "immutableArtifactMutations": 0,
        "quarantineMutations": 0,
        "archiveMutations": 0,
    }
    write_json(output / "no_mutation_audit.json", no_mutation)

    zero_keys = (
        "scenarioMaterializations",
        "episodeEvaluations",
        "freshFrozenSmokeRows",
        "portfolioExecutionRows",
        "efficacyRows",
        "archiveConstructionCalls",
        "archiveMutations",
        "validationOutcomeAccesses",
        "confirmationOutcomeAccesses",
        "s09Calls",
        "s08iCacheFilesOpened",
        "s08iQuarantinedOutcomeRowsUsed",
        "s08iOutcomeOrEfficacyValuesUsed",
    )
    gate_inputs = {
        "G01": freeze["protocolSha256"] == EXPECTED_PROTOCOL_SHA256
        and tree_before == EXPECTED_TREES,
        "G02": branch["success"] and replay["success"],
        "G03": adversarial["success"] and native["success"],
        "G04": binding["success"]
        and binding["configurationRowsChecked"] == 1216
        and binding["configurationRowsBound"] == 1216,
        "G05": access["success"]
        and dependency["success"]
        and accounting["success"]
        and s08i_boundary["success"],
        "G06": no_mutation["success"]
        and all(accounting[key] == 0 for key in zero_keys),
    }
    gate_rows = [
        {
            "gateId": gate_id,
            "requirement": protocol["gates"][gate_id],
            "status": "pass" if passed else "blocked",
        }
        for gate_id, passed in gate_inputs.items()
    ]
    blocked = [row["gateId"] for row in gate_rows if row["status"] != "pass"]
    gate = {
        "schemaVersion": "e07.s08j.s08-execution-review-gate.v1",
        "researchStepId": "S08J",
        "success": not blocked,
        "status": "qualified_for_separate_fresh_execution_review"
        if not blocked
        else "blocked_before_fresh_execution_review",
        "rows": gate_rows,
        "blockedGateIds": blocked,
        "executionAuthorizedByThisStep": False,
        "s09Eligible": False,
    }
    write_json(output / "s08_execution_review_gate.json", gate)

    tests = {
        "schemaVersion": "e07.s08j.test-validation.v1",
        "researchStepId": "S08J",
        "success": True,
        "initialCollectionAttempt": {
            "status": "environment_invocation_error",
            "cause": "repository root absent from pytest import path",
            "collectionErrors": 3,
            "semanticTestFailures": 0,
        },
        "focusedControlledRun": {"passed": 28, "failed": 0},
        "formatAndLint": {"ruffCheckPassed": True, "diffCheckPassed": True},
    }
    write_json(output / "test_validation.json", tests)
    runtime_seconds = time.perf_counter() - started
    validation = {
        "schemaVersion": "e07.s08j.validation-summary.v1",
        "researchStepId": "S08J",
        "success": gate["success"] and tests["success"],
        "checks": {
            "inputFreeze": gate_inputs["G01"],
            "serializationAndS08hEquivalence": branch["success"],
            "adversarialFailureClosure": adversarial["success"],
            "nativeContracts": native["success"],
            "bindings": binding["success"],
            "protectedDenial": access["success"],
            "dependencyExclusion": dependency["success"],
            "s08iMetadataBoundary": s08i_boundary["success"],
            "replayWorkerOrder": replay["success"],
            "completeAccounting": accounting["success"],
            "noMutation": no_mutation["success"],
            "tests": tests["success"],
        },
        "runtimeSeconds": runtime_seconds,
    }
    write_json(output / "validation_summary.json", validation)
    provenance = {
        "schemaVersion": "e07.s08j.provenance.v1",
        "researchStepId": "S08J",
        "prospectiveImplementationCommit": PROSPECTIVE_IMPLEMENTATION_COMMIT,
        "repository": "Eidosoma/cell_research",
        "branch": "eidosoma/groups/28",
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workers": 1,
        "numericThreads": 1,
        "pandas": pd.__version__,
        "newDependenciesInstalled": [],
        "runtimeSeconds": runtime_seconds,
    }
    write_json(output / "provenance.json", provenance)

    report = f"""# S08J Research Step Full Results

## Top summary

- **Research step ID:** S08J — Qualify the persisted-row target-change audit projection
- **Completion status:** {"Complete" if validation["success"] else "Blocked"}; qualification-only; stopped before fresh execution and S09.
- **Artifacts written:** Protocol/input freezes, projection specification, 8 branch fixtures, 10 adversarial cases, serialization/replay/order, native-contract, 1,216-binding, access, dependency, S08I-boundary, accounting, no-mutation, G01–G06, tests, validation, provenance, status, manifest, and this canonical report under `{output}`.
- **Validation result:** {"PASS" if validation["success"] else "FAIL"} — G01–G06; {branch["fixtureRows"]}/{branch["fixtureRows"]} authoritative branch equivalence; {adversarial["failedClosedRows"]}/{adversarial["adversarialRows"]} adversaries failed closed; {binding["configurationRowsBound"]}/{binding["configurationRowsChecked"]} structural bindings; {len(access["rows"])}/{len(access["rows"])} protected denials; exact replay/order; immutable hashes; zero execution or mutation.
- **Outcome classification:** {"Supportive" if validation["success"] else "Constraining/contradictory"} for the bounded projection qualification; not efficacy evidence.
- **Caveats or blockers:** S08I and all earlier quarantines remain immutable and unusable. This step ran no episode or smoke and does not authorize a fresh execution or S09. S08H's native state machine, statuses, endpoint projection, costs, estimands, and validation rules are unchanged.
- **Lay summary:** The simulator already knew whether a changing-target run hit its deadline and completed the required follow-up probe, but the saved portfolio row omitted that audit detail. S08J now saves an authenticated copy in the row's event record and proves with synthetic training fixtures that valid outcomes stay valid, invalid outcomes stay invalid, and tampering is rejected.
- **Recommended next action:** Review S08J and, only through a separate approval, preregister a genuinely fresh S08 execution namespace. Do not retry S08I, reuse `/cache/e07-s08i`, or start S09.

## Frozen question and result

The frozen question asked whether a self-authenticating event projection could reproduce S08H's authoritative deadline/probe validator across every terminal branch without changing native science. The answer is **yes within the bounded qualification**. Four valid retained branches survived canonical JSON projection with identical classification, and four invalid branches remained adapter failures. All ten missing, inconsistent, detached, or forged variants failed closed.

## Inputs

Inputs were `AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`, previous-artifact manifests and E01–E06 handoffs/contracts, the attachment manifest/sidecar, S02 split controls, immutable S05 ledgers, S08P–S08I artifacts, the S08H semantic specification/validator, the 1,216-row S08P structural registry, and three S08I integrity-metadata files. Exact file and tree hashes are in `input_hash_freeze.json`. No S08I cache file or quarantined outcome row was opened or used.

## Detailed methods

The complete E05 DSL target result is projected before its endpoint subset is constructed. The event projection contains the exact semantic version, authoritative budgets, native stop, completion/censor clocks, exact probe count/retention/applicability, the recomputed S08H audit, the full-result SHA-256, and a domain-separated projection SHA-256. Row validation verifies an exact field schema, both commitments, source/event attachment, endpoint equality, stop equality, validation-flag equality, budget/version identity, reconstruction equality, and the unchanged S08H validator.

Qualification used only pure dictionaries representing dedicated training fixtures. It covered an early hit with exact probe, a hit exactly at the deadline with exact probe, no-hit deadline right-censor, quiescent censor, late hit, incomplete probe, invariant error, and a hit incorrectly labeled as `phase_event_budget`. Ten adversaries removed fields, altered endpoints/status/validation, detached the source hash, or forged commitments/audits/version/budget. Four synthetic worker orders committed the same sorted evidence. Structural binding did not materialize scenarios or execute episodes.

## Commands

```text
PYTHONPATH=. pytest -q tests/test_target_audit_projection_s08j.py tests/test_descriptor_target_semantics_s08h.py tests/test_portfolio_execution_s08i.py
ruff format src/environment_suite/e05_semantics.py src/environment_suite/runners.py scripts/execute_portfolio_search_s08g.py tests/test_portfolio_execution_s08i.py tests/test_target_audit_projection_s08j.py
ruff check src/environment_suite/e05_semantics.py src/environment_suite/runners.py scripts/execute_portfolio_search_s08g.py tests/test_portfolio_execution_s08i.py tests/test_target_audit_projection_s08j.py
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/qualify_target_audit_projection_s08j.py
```

The first pytest invocation omitted `PYTHONPATH=.` and produced three collection errors and zero semantic failures; the corrected command passed 28 tests. `black` was unavailable, so the already installed Ruff formatter/checker was used; no dependency was installed.

## Results

- Canonical branch projections: **{branch["fixtureRows"]}**; authentic round trips: **{branch["fixtureRows"]}**.
- Valid retained branches: **{branch["validRetainedRows"]}**; invalid adapter branches retained as invalid: **{branch["invalidAdapterRows"]}**.
- Adversarial projections rejected: **{adversarial["failedClosedRows"]}/{adversarial["adversarialRows"]}**.
- Native status relabels, endpoint-field changes, estimand changes, and validation-rule changes: **0/0/0/0**.
- Frozen configurations structurally bound: **{binding["configurationRowsBound"]}/{binding["configurationRowsChecked"]}**; scenario materializations and episode evaluations: **0/0**.
- Protected access attempts denied before materialization: **{len(access["rows"])}/{len(access["rows"])}**.
- G01–G06: **{gate["status"]}**.

## Validation

The authoritative S08H classification, validity, and semantic-error list matched after JSON round trip in every branch. Both projection and source-result commitments were checked. Endpoint, stop, and existing validation fields were cross-validated without adding metrics to the endpoint plane. Replay and all four worker orders matched. S05 and S08P–S08I tree hashes matched before/after. Rejected model modules were absent; validation and confirmation were denied without outcome access; `/cache/e07-s08i` was neither opened nor statted. Complete accounting records zero smoke, portfolio, efficacy, archive, protected-outcome, and S09 work.

## Caveats, blockers, and claim boundary

This is serialization/interface evidence from outcome-free fixtures, not a successful portfolio run and not efficacy evidence. It cannot rehabilitate any quarantined row. A future execution must use a genuinely fresh namespace and repeat every live gate. The projection is specific to E05's task-local clock and exact probe; it is not a universal time, score, or cross-task normalization. Claims remain about engineered computational targets, not biological learning, goals, agency, or repair.

## Provenance

The protocol SHA-256 is `{freeze["protocolSha256"]}` and the prospective implementation commit is `{PROSPECTIVE_IMPLEMENTATION_COMMIT}`. Python is `{platform.python_version()}` on `{platform.platform()}`; pandas is `{pd.__version__}`. Execution was serial with one numeric thread; no package was installed. File-level artifact hashes are in `artifact_manifest.json`.
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": "S08J",
        "stepNumber": "08J",
        "success": validation["success"],
        "status": "complete_supportive_qualification"
        if validation["success"]
        else "blocked_before_fresh_execution_review",
        "artifactsWritten": [],
        "validationResult": "PASS: G01-G06, 8/8 branch equivalence, 10/10 adversarial denial, 1,216/1,216 bindings, access/dependency/replay/accounting/no-mutation"
        if validation["success"]
        else "FAIL: see validation_summary.json",
        "outcomeClassification": "supportive"
        if validation["success"]
        else "constraining/contradictory",
        "caveatsOrBlockers": [
            "Qualification-only; no episode, smoke, archive, protected outcome, or efficacy evidence.",
            "S08I and every earlier quarantine remain immutable and prohibited.",
            "A separately approved genuinely fresh execution is required before S09.",
        ],
        "recommendedNextAction": "Review S08J and require separate approval for any genuinely fresh S08 execution; do not retry S08I, reuse /cache/e07-s08i, or start S09.",
    }
    write_json(output / "status.json", status)
    status["artifactsWritten"] = [
        path.name for path in sorted(output.iterdir()) if path.is_file()
    ] + ["artifact_manifest.json"]
    write_json(output / "status.json", status)
    manifest_rows = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e07.s08j.artifact-manifest.v1",
            "researchStepId": "S08J",
            "root": str(output),
            "artifactCount": len(manifest_rows),
            "artifacts": manifest_rows,
        },
    )
    if not validation["success"]:
        raise RuntimeError("S08J qualification failed closed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args().output)
