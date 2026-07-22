#!/usr/bin/env python3
"""Build and validate the design-only E07 S08P preregistration."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
)
from src.portfolio_preregistration.core import (
    ARTIFACT_DIR,
    PROTOCOL_PATH,
    REPOSITORY,
    TASK_IDS,
    build_budget_slots,
    build_candidate_eligibility_registry,
    build_portfolio_seed_registry,
    candidate_commitment,
    canonical_hash,
    checked_protocol,
    hash_file,
    plan_digest,
    read_jsonl,
    validate_portfolio_registry,
    verify_frozen_inputs,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _input_provenance(protocol: Mapping[str, Any]) -> dict[str, Any]:
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
            "/previous-artifacts/E01/research_steps/S14/research_step_full_results.md"
        ),
        Path(
            "/previous-artifacts/E02/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E03/research_steps/S14/e07_handoff.md"),
        Path("/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.md"),
        Path(
            "/previous-artifacts/E05/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E06/report_inputs/e07_handoff.md"),
        PROTOCOL_PATH,
        REPOSITORY / "src/portfolio_preregistration/core.py",
        REPOSITORY / "scripts/preregister_portfolio_search_s08p.py",
        REPOSITORY / "tests/test_portfolio_preregistration_s08p.py",
    ]
    for step in (
        "S01",
        "S02",
        "S03",
        "S04",
        "S04A",
        "S05",
        "S06",
        "S06A",
        "S07R",
        "S07",
    ):
        paths.append(
            Path(f"/artifacts/research_steps/{step}/research_step_full_results.md")
        )
        manifest = Path(f"/artifacts/research_steps/{step}/artifact_manifest.json")
        if manifest.exists():
            paths.append(manifest)
    paths.extend(Path(spec["path"]) for spec in protocol["frozenInputs"].values())
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_file():
            raise RuntimeError(f"required provenance input is absent: {path}")
        unique.append(
            {
                "path": resolved,
                "sha256": hash_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schemaVersion": "e07.s08p.input-provenance.v1",
        "researchStepId": "S08P",
        "designOnly": True,
        "inputs": unique,
        "webSourcesUsed": False,
        "datasetsUsed": False,
        "protectedOutcomeFilesRead": False,
    }


def _access_audit() -> dict[str, Any]:
    suite = EnvironmentSuite(
        "/artifacts/research_steps/S02/environment_suite/task_registry.yaml",
        "/artifacts/research_steps/S02/environment_suite/split_manifest.json",
    )
    grant = AccessGrant(AccessPhase.DEVELOPMENT)
    attempts = []
    for task_id in TASK_IDS:
        for split in ("validation", "confirmation"):
            record = next(
                row
                for row in suite.records.values()
                if row.task_id == task_id and row.split.value == split
            )
            before = suite.broker.audit.materializer_invocations
            denied = False
            message = None
            try:
                suite.open(task_id, record.scenario_id, grant)
            except AccessDeniedError as exc:
                denied = True
                message = str(exc)
            after = suite.broker.audit.materializer_invocations
            attempts.append(
                {
                    "taskId": task_id,
                    "split": split,
                    "scenarioId": record.scenario_id,
                    "denied": denied,
                    "materializerDelta": after - before,
                    "message": message,
                }
            )
    success = len(attempts) == 16 and all(
        row["denied"] and row["materializerDelta"] == 0 for row in attempts
    )
    if not success:
        raise RuntimeError("protected/nontraining access audit failed")
    return {
        "schemaVersion": "e07.s08p.access-control-validation.v1",
        "researchStepId": "S08P",
        "success": True,
        "attempts": attempts,
        "audit": suite.broker.audit.to_dict(),
        "trainingScenarioMaterializations": 0,
        "validationScenarioMaterializations": 0,
        "confirmationScenarioMaterializations": 0,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
    }


def _rejected_dependency_audit(protocol: Mapping[str, Any]) -> dict[str, Any]:
    decisions = {}
    for key in ("s06RejectedDecision", "s06aRejectedDecision"):
        spec = protocol["frozenInputs"][key]
        decision = json.loads(Path(spec["path"]).read_text(encoding="utf-8"))
        eligibility = bool(
            decision.get("eligibleForS07", decision.get("deploymentEligible", False))
        )
        if eligibility:
            raise RuntimeError(f"{key} is no longer rejected")
        decisions[key] = {
            "path": spec["path"],
            "sha256": spec["sha256"],
            "eligible": False,
            "decisionReadOnly": True,
            "payloadModelOrEmbeddingOpened": False,
        }
    forbidden_modules = tuple(
        protocol["prohibitedDependencies"]["pythonModulePrefixes"]
    )
    loaded = sorted(name for name in sys.modules if name.startswith(forbidden_modules))
    source = (REPOSITORY / "src/portfolio_preregistration/core.py").read_text(
        encoding="utf-8"
    )
    forbidden_imports = [name for name in forbidden_modules if name in source]
    if loaded or forbidden_imports:
        raise RuntimeError("rejected model implementation entered S08P path")
    return {
        "schemaVersion": "e07.s08p.rejected-dependency-audit.v1",
        "researchStepId": "S08P",
        "success": True,
        "decisions": decisions,
        "forbiddenModulesLoaded": loaded,
        "forbiddenImports": forbidden_imports,
        "modelLoads": 0,
        "embeddingLoads": 0,
        "pseudoLabels": 0,
        "warmStarts": 0,
        "distillationUses": 0,
        "allocationUses": 0,
        "archiveMutationUses": 0,
        "decisionUses": 0,
    }


def _s07_exclusion_audit(
    candidates: list[dict[str, Any]],
    member_sets: list[dict[str, Any]],
    configs: list[dict[str, Any]],
) -> dict[str, Any]:
    forbidden = {
        "armId",
        "armMembership",
        "selected",
        "selectionRank",
        "inclusionProbability",
        "s07Outcome",
    }
    rows = [*candidates, *member_sets, *configs]
    hits = []
    for index, row in enumerate(rows):
        present = sorted(forbidden & set(row))
        if present:
            hits.append({"row": index, "fields": present})
    if hits or any(row.get("s07ArmMembershipUsed") is True for row in rows):
        raise RuntimeError("S07 allocation membership entered S08P planning")
    return {
        "schemaVersion": "e07.s08p.s07-arm-exclusion-audit.v1",
        "researchStepId": "S08P",
        "success": True,
        "candidateRows": len(candidates),
        "memberSetRows": len(member_sets),
        "configurationRows": len(configs),
        "forbiddenFieldHits": hits,
        "s07CandidateSelectionFrameRead": False,
        "s07LogicalOrPhysicalOutcomeLedgerRead": False,
        "s07ArmMembershipUses": 0,
        "s07OutcomePromotionUses": 0,
        "authorizedS07Reads": [
            "/artifacts/research_steps/S07/artifact_manifest.json",
            "/artifacts/research_steps/S07/downstream_gate_status.json",
        ],
    }


def _adapter_audit() -> dict[str, Any]:
    runners = (REPOSITORY / "src/environment_suite/runners.py").read_text(
        encoding="utf-8"
    )
    adapters = (REPOSITORY / "src/environment_suite/dsl_adapters.py").read_text(
        encoding="utf-8"
    )
    required_one_policy_guards = (
        "sorting accepts one DSL policy",
        "fault task accepts one DSL policy",
        "detour task accepts one DSL policy",
        "E05 core adapter accepts one DSL policy",
        "E05 target-change adapter accepts one DSL policy",
        "current E06 task contract accepts one DSL policy",
    )
    evidence = {
        text: (text in runners or text in adapters)
        for text in required_one_policy_guards
    }
    e04_binding = (
        "E04 DSL portfolio must bind every native Algotype explicitly" in runners
    )
    if not all(evidence.values()) or not e04_binding:
        raise RuntimeError(
            "adapter capability audit no longer matches the preregistration"
        )
    return {
        "schemaVersion": "e07.s08p.adapter-capability-audit.v1",
        "researchStepId": "S08P",
        "success": True,
        "singlePolicyGuardEvidence": evidence,
        "e04TwoCarrierBindingPresent": e04_binding,
        "identityLevelMultiPolicyAllEightTasksQualified": False,
        "conditionedSwitchingAllEightTasksQualified": False,
        "memberMemoryIsolationAllEightTasksQualified": False,
        "portfolioCostExtractionAllEightTasksQualified": False,
        "executionBlockerConfirmed": True,
    }


def _budget_accounting(
    protocol: Mapping[str, Any], slots: list[dict[str, Any]]
) -> dict[str, Any]:
    by_stage = Counter(row["stage"] for row in slots)
    by_task = Counter(row["taskId"] for row in slots)
    by_family = Counter(row["scenarioFamilyOrdinal"] for row in slots)
    by_generation = Counter(row["generation"] for row in slots)
    smoke = sum(bool(row["smoke"]) for row in slots)
    result = {
        "schemaVersion": "e07.s08p.budget-accounting.v1",
        "researchStepId": "S08P",
        "designOnly": True,
        "totalTrainingLogicalSlots": len(slots),
        "initialLogicalSlots": by_stage["initial"],
        "adaptiveLogicalSlots": by_stage["adaptive"],
        "smokeLogicalSlotsWithinBudget": smoke,
        "logicalSlotsByTask": dict(sorted(by_task.items())),
        "logicalSlotsByScenarioFamily": {
            str(key): value for key, value in sorted(by_family.items())
        },
        "logicalSlotsByGeneration": {
            str(key): value for key, value in sorted(by_generation.items())
        },
        "futureValidationLogicalMaximum": int(
            protocol["selectionBoundaries"]["validation"]["maximumLogicalBudget"]
        ),
        "confirmationLogicalBudget": 0,
        "evaluationsExecuted": 0,
        "physicalEvaluationsExecuted": 0,
        "runtimeDrivenWeakening": False,
        "nativeHorizonNormalizationApplied": False,
        "completeAccounting": True,
    }
    if (
        len(slots) != 11008
        or by_stage != Counter({"adaptive": 6144, "initial": 4864})
        or smoke != 40
        or set(by_task.values()) != {1376}
        or set(by_family.values()) != {2752}
    ):
        raise RuntimeError("S08P budget accounting mismatch")
    return result


def _paired_comparison_registry(protocol: Mapping[str, Any]) -> dict[str, Any]:
    objectives = yaml.safe_load(
        Path(protocol["frozenInputs"]["s04ObjectiveRegistry"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    task_registry = yaml.safe_load(
        Path(protocol["frozenInputs"]["s02TaskRegistry"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    contracts = {
        row["taskId"]: {
            "nativeUnit": row["horizon"]["nativeUnit"],
            "censoringRule": row["horizon"]["censoringRule"],
            "comparabilityGroup": row["horizon"]["comparabilityGroup"],
            "costContract": row["costContract"],
            "claimBoundary": row["claimBoundary"],
        }
        for row in task_registry["tasks"]
    }
    return {
        "schemaVersion": "e07.s08p.paired-native-comparison-registry.v1",
        "researchStepId": "S08P",
        "scenarioCoupling": protocol["scenarioCoupling"],
        "comparators": protocol["comparators"],
        "estimands": protocol["estimands"],
        "taskNativeObjectives": objectives["taskObjectives"],
        "taskNativeContracts": contracts,
        "crossTaskScalarizationPermitted": False,
        "universalNormalizedScorePermitted": False,
        "allocationArmMembershipPromotionPermitted": False,
    }


def _no_mutation_snapshot(protocol: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "s05EvaluationLedger",
        "s05PolicyCatalog",
        "s05LineageLedger",
        "s05ArchiveEntries",
        "s05ArchiveManifest",
        "s07ArtifactManifest",
    )
    return {name: hash_file(protocol["frozenInputs"][name]["path"]) for name in names}


def build() -> None:
    protocol = checked_protocol()
    before = _no_mutation_snapshot(protocol)
    frozen_checks = verify_frozen_inputs(protocol)
    source_rows = read_jsonl(protocol["frozenInputs"]["s07rCandidateRegistry"]["path"])
    candidates = build_candidate_eligibility_registry(
        protocol, source_order=source_rows
    )
    member_sets, configs = build_portfolio_seed_registry(protocol, candidates)
    legality = validate_portfolio_registry(protocol, candidates, member_sets, configs)
    slots = build_budget_slots(protocol, configs)
    digest = plan_digest(candidates, member_sets, configs, slots)

    order_digests = []
    orders = [
        source_rows,
        list(reversed(source_rows)),
        source_rows[137:] + source_rows[:137],
        sorted(
            source_rows,
            key=lambda row: canonical_hash(
                "E07/S08P/worker-order/v1", row["policySha256"]
            ),
        ),
    ]
    for order in orders:
        ordered_candidates = build_candidate_eligibility_registry(
            protocol, source_order=order
        )
        ordered_sets, ordered_configs = build_portfolio_seed_registry(
            protocol, ordered_candidates
        )
        ordered_slots = build_budget_slots(protocol, ordered_configs)
        order_digests.append(
            plan_digest(
                ordered_candidates, ordered_sets, ordered_configs, ordered_slots
            )
        )
    if set(order_digests) != {digest}:
        raise RuntimeError("portfolio plan depends on candidate/worker order")

    access = _access_audit()
    rejected = _rejected_dependency_audit(protocol)
    s07_exclusion = _s07_exclusion_audit(candidates, member_sets, configs)
    adapter = _adapter_audit()
    budget = _budget_accounting(protocol, slots)
    after = _no_mutation_snapshot(protocol)
    if before != after:
        raise RuntimeError("S05 or S07 immutable source changed during S08P")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROTOCOL_PATH, ARTIFACT_DIR / "s08p_portfolio_protocol.yaml")
    _write_jsonl(ARTIFACT_DIR / "candidate_eligibility_registry.jsonl", candidates)
    _write_jsonl(ARTIFACT_DIR / "portfolio_member_sets.jsonl", member_sets)
    _write_jsonl(ARTIFACT_DIR / "portfolio_seed_registry.jsonl", configs)
    pd.DataFrame(slots).to_parquet(
        ARTIFACT_DIR / "budget_slot_ledger.parquet", index=False
    )
    _write_json(ARTIFACT_DIR / "budget_accounting.json", budget)
    _write_json(ARTIFACT_DIR / "composition_legality_validation.json", legality)
    _write_yaml(
        ARTIFACT_DIR / "environment_signal_registry.yaml",
        {
            "schemaVersion": "e07.s08p.environment-signal-registry.v1",
            "researchStepId": "S08P",
            **protocol["selectorSignals"],
        },
    )
    _write_yaml(
        ARTIFACT_DIR / "portfolio_cost_registry.yaml",
        {
            "schemaVersion": "e07.s08p.portfolio-cost-registry.v1",
            "researchStepId": "S08P",
            **protocol["portfolioCosts"],
        },
    )
    _write_yaml(
        ARTIFACT_DIR / "comparator_registry.yaml",
        {
            "schemaVersion": "e07.s08p.comparator-registry.v1",
            "researchStepId": "S08P",
            **protocol["comparators"],
        },
    )
    _write_json(
        ARTIFACT_DIR / "paired_native_comparison_registry.json",
        _paired_comparison_registry(protocol),
    )
    _write_yaml(
        ARTIFACT_DIR / "s09_gate_registry.yaml",
        {
            "schemaVersion": "e07.s08p.s09-gate-registry.v1",
            "researchStepId": "S08P",
            **protocol["promotionAndS09Gate"],
        },
    )
    _write_json(ARTIFACT_DIR / "access_control_validation.json", access)
    _write_json(ARTIFACT_DIR / "rejected_dependency_audit.json", rejected)
    _write_json(ARTIFACT_DIR / "s07_arm_exclusion_audit.json", s07_exclusion)
    _write_json(ARTIFACT_DIR / "adapter_capability_audit.json", adapter)
    _write_json(
        ARTIFACT_DIR / "candidate_hash_validation.json",
        {
            "schemaVersion": "e07.s08p.candidate-hash-validation.v1",
            "researchStepId": "S08P",
            "success": True,
            "candidateRows": len(candidates),
            "authoritativeDocuments": len(candidates),
            "canonicalRecompiles": len(candidates),
            "hashMismatches": 0,
            "candidateEligibilityCommitmentSha256": candidate_commitment(candidates),
        },
    )
    _write_json(
        ARTIFACT_DIR / "deterministic_plan_validation.json",
        {
            "schemaVersion": "e07.s08p.deterministic-plan-validation.v1",
            "researchStepId": "S08P",
            "success": True,
            "syntheticWorkerOrders": len(order_digests),
            "uniqueDigests": len(set(order_digests)),
            "planSha256": digest,
            "digests": order_digests,
            "counterBasedRandomness": True,
            "workerOrderAsRandomSource": False,
        },
    )
    _write_json(
        ARTIFACT_DIR / "feasibility_validation.json",
        {
            "schemaVersion": "e07.s08p.feasibility-validation.v1",
            "researchStepId": "S08P",
            "success": True,
            "staticDesignFeasible": True,
            "substantiveExecutionEligible": False,
            "candidateRows": len(candidates),
            "candidateCountsByTask": dict(
                sorted(Counter(row["taskId"] for row in candidates).items())
            ),
            "carrierCountsByTask": {
                task: dict(
                    sorted(
                        Counter(
                            row["nativeCarrier"]
                            for row in candidates
                            if row["taskId"] == task
                        ).items()
                    )
                )
                for task in TASK_IDS
            },
            "memberSetRows": len(member_sets),
            "configurationRows": len(configs),
            "logicalBudgetRows": len(slots),
            "validationLogicalMaximum": 3072,
            "confirmationBudget": 0,
            "adapterPrerequisiteBlocked": True,
            "blocker": "multi-policy identity assignment and bounded switching are not qualified across all eight native runners",
        },
    )
    _write_json(
        ARTIFACT_DIR / "no_mutation_audit.json",
        {
            "schemaVersion": "e07.s08p.no-mutation-audit.v1",
            "researchStepId": "S08P",
            "success": True,
            "before": before,
            "after": after,
            "identical": before == after,
            "s05ArchiveMutations": 0,
            "s08PortfolioArchiveRows": 0,
            "episodeEvaluations": 0,
        },
    )
    _write_json(
        ARTIFACT_DIR / "input_hash_freeze.json",
        {
            "schemaVersion": "e07.s08p.input-hash-freeze.v1",
            "researchStepId": "S08P",
            "frozenAtUtc": _utc_now(),
            "inputs": frozen_checks,
        },
    )
    _write_json(ARTIFACT_DIR / "input_provenance.json", _input_provenance(protocol))
    _write_json(
        ARTIFACT_DIR / "preregistration_freeze.json",
        {
            "schemaVersion": "e07.s08p.preregistration-freeze.v1",
            "researchStepId": "S08P",
            "frozenAtUtc": _utc_now(),
            "designOnly": True,
            "protocolSha256": hash_file(PROTOCOL_PATH),
            "candidateEligibilityCommitmentSha256": candidate_commitment(candidates),
            "completePlanSha256": digest,
            "candidateRows": 512,
            "memberSetRows": 192,
            "initialConfigurationRows": 1216,
            "trainingLogicalBudgetRows": 11008,
            "smokeRowsWithinBudget": 40,
            "episodeEvaluations": 0,
            "portfolioArchiveMutations": 0,
            "substantiveExecutionEligible": False,
        },
    )
    gate_rows = [
        {
            "gateId": "G01",
            "requirement": "all frozen input hashes and rejection decisions unchanged",
            "status": "pass",
        },
        {
            "gateId": "G02",
            "requirement": "512 authoritative candidates recompile and exclude S07 arm efficacy",
            "status": "pass",
        },
        {
            "gateId": "G03",
            "requirement": "portfolio compositions, selectors, costs, and paired plan are legal",
            "status": "pass",
        },
        {
            "gateId": "G04",
            "requirement": "multi-policy identity assignment and switching adapter qualified on all eight tasks",
            "status": "blocked",
        },
        {
            "gateId": "G05",
            "requirement": "protected access denied and rejected models/embeddings isolated",
            "status": "pass",
        },
        {
            "gateId": "G06",
            "requirement": "smoke, training, validation ceiling, and stopping budgets completely accounted",
            "status": "pass",
        },
    ]
    _write_json(
        ARTIFACT_DIR / "s08_execution_eligibility_gate.json",
        {
            "schemaVersion": "e07.s08p.s08-execution-eligibility-gate.v1",
            "researchStepId": "S08P",
            "success": True,
            "designPreregistrationComplete": True,
            "s08SubstantiveExecutionEligible": False,
            "rows": gate_rows,
            "blockedGateIds": ["G04"],
            "evaluationMustFailClosed": True,
            "recommendedNextAction": "Run a bounded train-only portfolio-adapter qualification without evaluation search or archive mutation, then revalidate G01-G06 before substantive S08.",
        },
    )
    validation_checks = {
        "allFrozenInputsHashMatch": True,
        "candidateRows512And64PerTask": True,
        "candidateCanonicalRecompile512Of512": True,
        "s07ArmMembershipAndOutcomePromotionZero": True,
        "rejectedModelAndEmbeddingLoadsZero": True,
        "legalMemberSets192": True,
        "legalInitialConfigurations1216": True,
        "deterministicPlanFourOrdersOneDigest": True,
        "trainingLogicalSlots11008": True,
        "smokeSlots40WithinBudget": True,
        "protectedSplitDenial16Of16": True,
        "validationAndConfirmationOutcomesZero": True,
        "s05AndS07Immutable": True,
        "substantiveEpisodesZero": True,
        "portfolioArchiveMutationsZero": True,
        "multiPolicyAdapterAllTasksQualified": False,
        "failClosedExecutionGate": True,
    }
    _write_json(
        ARTIFACT_DIR / "validation_summary.json",
        {
            "schemaVersion": "e07.s08p.validation-summary.v1",
            "researchStepId": "S08P",
            "success": True,
            "designOnly": True,
            "checks": validation_checks,
            "validationResult": "PASS for the bounded preregistration: 512 candidates, 192 legal member sets, 1,216 deterministic initial configurations, 11,008 fully accounted training slots, 40-row in-budget smoke plan, 16/16 protected denials, zero outcomes/evaluations/mutations/rejected-model use; substantive S08 remains fail-closed at G04 pending multi-policy adapter qualification.",
        },
    )
    _write_json(
        ARTIFACT_DIR / "environment.json",
        {
            "schemaVersion": "e07.s08p.environment.v1",
            "researchStepId": "S08P",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "designWorkersUsed": 1,
            "futureEvaluationWorkers": 8,
            "gpuUsed": False,
            "networkUsed": False,
            "newDependenciesInstalled": [],
        },
    )


def validate() -> None:
    protocol = checked_protocol(ARTIFACT_DIR / "s08p_portfolio_protocol.yaml")
    verify_frozen_inputs(protocol)
    candidates = read_jsonl(ARTIFACT_DIR / "candidate_eligibility_registry.jsonl")
    member_sets = read_jsonl(ARTIFACT_DIR / "portfolio_member_sets.jsonl")
    configs = read_jsonl(ARTIFACT_DIR / "portfolio_seed_registry.jsonl")
    validate_portfolio_registry(protocol, candidates, member_sets, configs)
    slots = pd.read_parquet(ARTIFACT_DIR / "budget_slot_ledger.parquet").to_dict(
        "records"
    )
    if (
        len(slots) != 11008
        or int(sum(slots[index]["smoke"] for index in range(len(slots)))) != 40
    ):
        raise RuntimeError("persisted budget ledger failed validation")
    gate = json.loads(
        (ARTIFACT_DIR / "s08_execution_eligibility_gate.json").read_text()
    )
    if gate["s08SubstantiveExecutionEligible"] is not False or gate[
        "blockedGateIds"
    ] != ["G04"]:
        raise RuntimeError("persisted execution gate is not fail closed")
    if any(
        path.name.startswith(("portfolio_results", "evaluation_ledger"))
        for path in ARTIFACT_DIR.iterdir()
    ):
        raise RuntimeError("substantive result appeared in design-only directory")


def manifest() -> None:
    artifacts = []
    for path in sorted(ARTIFACT_DIR.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {"path": path.name, "sha256": hash_file(path), "bytes": path.stat().st_size}
        )
    _write_json(
        ARTIFACT_DIR / "artifact_manifest.json",
        {
            "schemaVersion": "e07.s08p.artifact-manifest.v1",
            "researchStepId": "S08P",
            "root": str(ARTIFACT_DIR),
            "designOnly": True,
            "success": True,
            "artifacts": artifacts,
        },
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"build", "validate", "manifest"}:
        raise SystemExit(
            "usage: preregister_portfolio_search_s08p.py {build|validate|manifest}"
        )
    {"build": build, "validate": validate, "manifest": manifest}[sys.argv[1]]()


if __name__ == "__main__":
    main()
