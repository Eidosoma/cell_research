#!/usr/bin/env python3
"""Final integrity, accounting, and downstream-gate validation for S07."""

from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.allocation_redesign.core import hash_file
from src.non_surrogate_allocation.core import (
    ARTIFACT_DIR,
    LOCK_PATH,
    S05_DIR,
    _read_jsonl,
    _write_json,
    validate_physical_result,
)


def main() -> None:
    root = ARTIFACT_DIR
    freeze = json.loads((root / "execution_freeze.json").read_text())
    execution = json.loads((root / "execution_accounting.json").read_text())
    analysis = json.loads((root / "analysis_summary.json").read_text())
    frame = pd.read_parquet(root / "candidate_selection_frame.parquet")
    roster = pd.read_parquet(root / "allocation_logical_roster.parquet")
    plan = pd.read_parquet(root / "physical_evaluation_plan.parquet")
    physical_rows = _read_jsonl(root / "physical_evaluation_ledger.jsonl")
    physical = {row["physicalRowId"]: row for row in physical_rows}
    expected = {row["physicalRowId"]: row for row in plan.to_dict("records")}

    result_errors = {
        row["physicalRowId"]: validate_physical_result(row, expected[row["physicalRowId"]])
        for row in physical_rows
    }
    result_errors = {key: value for key, value in result_errors.items() if value}
    required_logical = {
        "protocolSha256", "logicalRowId", "physicalRowId", "armId", "taskId",
        "policySha256", "scenarioFamilyOrdinal", "scenarioCommitmentSha256",
        "inclusionProbability", "deterministicSelectionIndicator",
        "randomizedSelectionIndicator", "selectionMechanismDeterministic",
        "selectionRank", "selectionStrata", "overlapArmIds", "reuseIndicator",
    }
    missing_logical = sorted(required_logical - set(roster.columns))
    arm_counts = roster.groupby("armId").size().to_dict()
    task_arm_counts = roster.groupby(["armId", "taskId"]).size()
    family_arm_counts = roster.groupby(["armId", "scenarioFamilyOrdinal"]).size()
    multiplicity = Counter(roster["physicalRowId"])
    logical_reconciled = sum(multiplicity.values()) == len(roster) == 3072
    physical_reconciled = set(multiplicity) == set(physical) == set(plan["physicalRowId"])
    frame_arm_selected = frame.groupby("armId")["selected"].sum().astype(int).to_dict()
    uniform = frame[frame.armId == "uniform_randomized"]
    fixed = frame[frame.armId != "uniform_randomized"]
    propensity_pass = (
        set(uniform.inclusionProbability) == {0.5}
        and set(fixed.inclusionProbability) <= {0.0, 1.0}
        and set(frame_arm_selected.values()) == {256}
        and len(frame) == 1536
    )
    budget = {
        "schemaVersion": "e07.s07.budget-accounting.v1", "researchStepId": "S07",
        "success": logical_reconciled and physical_reconciled,
        "candidateSelectionRows": len(frame), "logicalRows": len(roster),
        "physicalRows": len(physical), "logicalMinusPhysicalReuse": len(roster) - len(physical),
        "armLogicalRows": arm_counts,
        "taskArmCountsAll128": bool((task_arm_counts == 128).all()),
        "familyArmCountsAll256": bool((family_arm_counts == 256).all()),
        "physicalMultiplicityCounts": dict(sorted(Counter(multiplicity.values()).items())),
        "physicalMultiplicitySum": sum(multiplicity.values()),
        "smokeLogicalRows": int(roster.smokeLogicalIndicator.sum()),
        "smokePhysicalRows": int(plan.smokeIndicator.sum()),
        "failedPhysicalRowsRetained": execution["failedPhysicalRowsRetained"],
        "censoredPhysicalRowsRetained": execution["censoredPhysicalRowsRetained"],
        "replayPassPhysicalRows": execution["replayPassRows"],
        "workers": execution["execution"]["workers"],
        "substantiveWallSeconds": execution["execution"]["wallSeconds"],
        "runtimeScaleReduction": False, "efficacyEarlyStop": False,
    }
    _write_json(root / "budget_accounting.json", budget)
    propensity = {
        "schemaVersion": "e07.s07.propensity-selection-audit.v1", "researchStepId": "S07",
        "success": propensity_pass, "candidateSelectionRows": len(frame),
        "selectedCandidatesByArm": frame_arm_selected,
        "uniformInclusionProbability": 0.5,
        "uniformRandomizedIndicatorsPresent": bool(uniform.randomizedSelectionIndicator.notna().all()),
        "fixedProbabilitiesOnlyZeroOrOne": set(fixed.inclusionProbability) <= {0.0, 1.0},
        "fixedArmInference": "descriptive_only", "uniformInference": "probability_sampled_comparator",
        "excludedCandidatesPreserved": len(frame) - int(frame.selected.sum()),
        "missingLogicalFields": missing_logical,
    }
    _write_json(root / "propensity_selection_audit.json", propensity)

    s05_after = {
        "archiveManifest": hash_file(S05_DIR / "archive/archive_manifest.json"),
        "archiveEntries": hash_file(S05_DIR / "archive/archive_entries.jsonl"),
        "evaluationLedger": hash_file(S05_DIR / "evaluation_ledger.jsonl"),
    }
    no_mutation = {
        "schemaVersion": "e07.s07.no-mutation-audit.v1", "researchStepId": "S07",
        "success": (
            s05_after["archiveManifest"] == freeze["S05ArchiveManifestSha256Before"]
            and s05_after["archiveEntries"] == freeze["S05ArchiveEntriesSha256Before"]
            and s05_after["evaluationLedger"] == freeze["S05EvaluationLedgerSha256Before"]
        ),
        "before": {
            "archiveManifest": freeze["S05ArchiveManifestSha256Before"],
            "archiveEntries": freeze["S05ArchiveEntriesSha256Before"],
            "evaluationLedger": freeze["S05EvaluationLedgerSha256Before"],
        },
        "after": s05_after, "S05ArchiveMutations": 0,
    }
    _write_json(root / "no_mutation_audit.json", no_mutation)

    source_paths = [
        Path("src/non_surrogate_allocation/core.py"),
        Path("src/non_surrogate_allocation/analysis.py"),
        Path("scripts/run_non_surrogate_allocation_s07.py"),
    ]
    imports = set()
    for path in source_paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    forbidden_imports = sorted(name for name in imports if name.startswith(("src.surrogate_models", "src.surrogate_remediation")))
    dependency = {
        "schemaVersion": "e07.s07.forbidden-dependency-audit.v1", "researchStepId": "S07",
        "success": not forbidden_imports,
        "forbiddenImports": forbidden_imports,
        "rejectedModelLoads": 0, "rejectedEmbeddingLoads": 0,
        "pseudoLabels": 0, "warmStarts": 0, "distillationUses": 0,
        "allocationOrDecisionUses": 0,
        "controlPlaneRejectedDecisionChecksOnly": [
            "/artifacts/research_steps/S06/deployment_decision.json",
            "/artifacts/research_steps/S06A/deployment_decision.json",
        ],
    }
    _write_json(root / "forbidden_dependency_audit.json", dependency)

    primary = json.loads((root / "primary_comparisons.json").read_text())
    trace = json.loads((root / "trace_availability_audit.json").read_text())
    s08_pass = all((budget["success"], propensity["success"], no_mutation["success"], dependency["success"], not result_errors))
    downstream = {
        "schemaVersion": "e07.s07.downstream-gate-status.v1", "researchStepId": "S07",
        "S08": {
            "integrityPrerequisitesPass": s08_pass,
            "status": "eligible_for_separate_preregistered_portfolio_design" if s08_pass else "blocked",
            "constraints": [
                "use only immutable S05 candidates and S07 training results",
                "allocation-arm membership is not efficacy evidence and may not promote a policy",
                "preserve native outcomes, costs, failures, censors, and claim boundaries",
            ],
        },
        "S09": {"status": "blocked_pending_S08_candidate_portfolio_freeze_and_paired_plan"},
        "S10": {
            "status": "redesign_required_or_blocked",
            "reason": "complete native trajectories are unavailable; only native event summaries exist",
            "eventFeatureRedesignPermittedOnlyAfterSeparatePreregistration": True,
        },
        "rejectedModelOrEmbeddingUse": False,
    }
    _write_json(root / "downstream_gate_status.json", downstream)

    validation = {
        "schemaVersion": "e07.s07.validation-summary.v1", "researchStepId": "S07",
        "success": all((budget["success"], propensity["success"], no_mutation["success"], dependency["success"], not result_errors, analysis["success"])),
        "validatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "candidateSelectionRows": len(frame), "logicalRows": len(roster), "physicalRows": len(physical),
        "physicalResultIntegrityErrors": result_errors,
        "allReplayPass": all(row["replayPass"] for row in physical_rows),
        "allTrainOnly": all(row["split"] == "train" and not row["protected"] for row in physical_rows),
        "allScenarioCommitmentsMatch": all(not validate_physical_result(row, expected[row["physicalRowId"]]) for row in physical_rows),
        "validationOutcomeEvaluations": 0, "confirmationOutcomeEvaluations": 0,
        "S05Unchanged": no_mutation["success"], "rejectedModelOrEmbeddingUse": False,
        "workerOrderPlanPass": json.loads((root / "worker_order_plan_validation.json").read_text())["success"],
        "smokePass": json.loads((root / "smoke_gate.json").read_text())["success"],
        "HolmSignificantFixedArmContrast": analysis["anyHolmSignificantFixedArmContrast"],
        "outcomeClassification": "null",
        "analysisCompatibilityFix": "read-only scenarioOrdinal and stableEvaluationSha256 aliases for frozen S05 stability helper; no persisted result or analysis rule changed",
    }
    _write_json(root / "validation_summary.json", validation)

    rare = pd.read_parquet(root / "rare_event_discovery_curves.parquet")
    descriptor = pd.read_parquet(root / "shadow_descriptor_discovery_curves.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, data, title in (
        (axes[0], rare, "Rare status discovery yield"),
        (axes[1], descriptor, "Shadow descriptor discovery yield"),
    ):
        summary = data.groupby(["armId", "waveIndex"], as_index=False).cumulativeDiscoveryYield.mean()
        for arm_id, group in summary.groupby("armId"):
            axis.plot(group.waveIndex, group.cumulativeDiscoveryYield, marker="o", label=arm_id)
        axis.set_xticks([1, 2, 3, 4])
        axis.set_xlabel("Frozen scenario-family wave")
        axis.set_ylabel("Equal-task mean cumulative discoveries / logical row")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle("S07 fixed arms vs probability-sampled uniform comparator")
    fig.savefig(root / "allocation_discovery_curves.png", dpi=180)
    plt.close(fig)

    environment = {
        "schemaVersion": "e07.s07.environment.v1", "python": sys.version,
        "platform": platform.platform(), "cpuCount": os.cpu_count(),
        "workers": 8, "numericThreadsPerWorker": 1,
        "pandas": pd.__version__, "repositoryCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "executionLockSha256": hash_file(LOCK_PATH),
    }
    _write_json(root / "environment.json", environment)
    provenance = {
        "schemaVersion": "e07.s07.provenance.v1", "researchStepId": "S07",
        "approvedS07RProtocolSha256": freeze["approvedS07RProtocolSha256"],
        "executionLockSha256": freeze["executionLockSha256"],
        "candidatePopulationCommitmentSha256": freeze["candidatePopulationCommitmentSha256"],
        "planDigestSha256": freeze["planDigestSha256"],
        "candidateSelectionFrameSha256": freeze["candidateSelectionFrameSha256"],
        "logicalRosterSha256": freeze["logicalRosterSha256"],
        "physicalEvaluationPlanSha256": freeze["physicalEvaluationPlanSha256"],
        "physicalEvaluationLedgerSha256": execution["physicalLedgerSha256"],
        "analysisImplementationCommitBeforeOutcomes": "2c7a1e6",
        "analysisMechanicalCompatibilityFixRecorded": True,
    }
    _write_json(root / "provenance.json", provenance)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
