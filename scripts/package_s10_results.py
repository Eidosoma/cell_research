#!/usr/bin/env python3
"""Package S10 result/provenance manifests and validate the final handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from reference_simulator.model import canonical_json_bytes


OUTPUT = Path("/artifacts/research_steps/S10")
REPOSITORY = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True
    ).strip()


def build_result_summary() -> dict[str, object]:
    results = pd.read_parquet(OUTPUT / "action_suppression_results.parquet")
    pairs = pd.read_parquet(OUTPUT / "paired_effects.parquet")
    audits = pd.read_parquet(OUTPUT / "filter_audit.parquet")
    reach = pd.read_parquet(OUTPUT / "filtered_structural_reachability.parquet")
    budget = pd.read_parquet(OUTPUT / "event_budget_sensitivity.parquet")
    primary = pairs[pairs.filter_threshold == 0]
    metrics: dict[str, object] = {}
    for metric, group in primary.groupby("filter_metric", sort=True):
        metrics[metric] = {
            "pairs": len(group),
            "suppressionExposedPairs": int(group.suppressed_count.gt(0).sum()),
            "suppressedProposals": int(group.suppressed_count.sum()),
            "controlCompletions": int(group.control_completed.sum()),
            "filteredCompletions": int(group.filtered_completed.sum()),
            "completionDifference": int(group.filtered_completed.sum() - group.control_completed.sum()),
            "createdExactImpossibilityPairs": int(
                group.intervention_created_impossibility.sum()
            ),
            "controlCompleteFilterNoncomplete": int(
                (group.control_completed & ~group.filtered_completed).sum()
            ),
            "controlNoncompleteFilterComplete": int(
                (~group.control_completed & group.filtered_completed).sum()
            ),
            "bothCompletePairs": int(group.efficiency_comparable.sum()),
            "reachableCensoredPairs": int(group.completion_censored.sum()),
        }
    replicate_zero = pairs[pairs.replicate_index == 0]
    threshold = (
        replicate_zero.groupby(["filter_metric", "filter_threshold"])
        .agg(
            pairs=("pair_id", "size"),
            suppression_exposed=("suppressed_count", lambda x: int((x > 0).sum())),
            suppressed=("suppressed_count", "sum"),
            filtered_completion=("filtered_completed", "mean"),
            created_impossibility=("intervention_created_impossibility", "mean"),
        )
        .reset_index()
    )
    executed = budget.sensitivity_action.eq("extended_exact_reachable")
    validation = json.loads((OUTPUT / "validation_results.json").read_text())
    summary = {
        "schemaVersion": "e03.s10.result_summary.v1",
        "researchStepId": "S10",
        "completionStatus": "complete",
        "outcomeClassification": "supportive",
        "controls": int(results.condition.eq("unfiltered_control").sum()),
        "filteredRuns": int(results.condition.eq("metric_filtered").sum()),
        "primaryPairs": int(primary.shape[0]),
        "thresholdSensitivityPairs": int(
            pairs.filter_threshold.eq(1).sum()
        ),
        "structuralContexts": int(
            reach[
                [
                    "source_family_ordinal",
                    "source_state_ordinal",
                    "arm_family_ordinal",
                    "arm_state_ordinal",
                    "arm_variant",
                ]
            ].drop_duplicates().shape[0]
        ),
        "filterFalsePositives": int(audits.false_positive_count.sum()),
        "filterFalseNegatives": int(audits.false_negative_count.sum()),
        "metricResults": metrics,
        "thresholdSensitivity": threshold.to_dict("records"),
        "eventBudgetSensitivity": {
            "primaryReplicate0BudgetRuns": len(budget),
            "exactReachableExtended": int(executed.sum()),
            "exactImpossibleNotExtended": int((~executed).sum()),
            "extendedCompleted": int(budget.loc[executed, "extended_completed"].sum()),
            "extendedStillActive": int(
                budget.loc[executed, "extended_stop_reason"].eq("event_budget").sum()
            ),
            "extendedQuiescent": int(
                budget.loc[executed, "extended_stop_reason"].eq("quiescent").sum()
            ),
        },
        "validationGatesPassed": validation["gatesPassed"],
        "validationGateCount": validation["gateCount"],
        "deterministicReplays": validation["deterministicReplays"],
        "primaryFinding": "Strict metric-worsening suppression is causally harmful in retained barrier contexts when it removes required graph edges or blocks completion on the identical opportunity stream; exact impossibility, reachable censoring, and successful-run efficiency remain separate.",
        "recommendedNextAction": "Hand control back and await separate authorization for S11; use S10 exposure/reachability strata when constructing behavioral nulls.",
    }
    return summary


def main() -> None:
    required = [
        "action_suppression_results.parquet",
        "filter_audit.parquet",
        "paired_effects.parquet",
        "filtered_structural_reachability.parquet",
        "effect_summary.parquet",
        "event_budget_sensitivity.parquet",
        "representative_traces.parquet",
        "suppression_effects.png",
        "replay_validation.parquet",
        "validation_results.json",
        "input_immutability.json",
        "environment.json",
        "commands.log",
        "research_step_full_results.md",
    ]
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing final S10 artifacts: {missing}")
    summary = build_result_summary()
    write_json(OUTPUT / "result_summary.json", summary)

    source_paths = [
        REPOSITORY / "analysis/s10_action_suppression_contract.json",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "src/detours/action_suppression.py",
        REPOSITORY / "src/detours/barrier_interventions.py",
        REPOSITORY / "scripts/build_s10_action_suppression.py",
        REPOSITORY / "scripts/run_s10_budget_sensitivity.py",
        REPOSITORY / "scripts/build_s10_representative_traces.py",
        REPOSITORY / "scripts/validate_s10_action_suppression.py",
        REPOSITORY / "scripts/package_s10_results.py",
        REPOSITORY / "tests/test_action_suppression.py",
    ]
    immutability = json.loads((OUTPUT / "input_immutability.json").read_text())
    provenance = {
        "schemaVersion": "e03.s10.provenance_manifest.v1",
        "researchStepId": "S10",
        "repository": str(REPOSITORY),
        "branch": git_output("branch", "--show-current"),
        "repositoryCommitAtPackaging": git_output("rev-parse", "HEAD"),
        "previousArtifactMount": "/previous-artifacts/E01",
        "protocolContract": str(
            REPOSITORY / "analysis/s10_action_suppression_contract.json"
        ),
        "schedulerClaimBoundary": "paired deterministic S09 common opportunity streams; no scheduler-frequency, expected-value, or universal-scheduler claim",
        "sourceFiles": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        "executionInputs": immutability["inputs"],
        "runtimeCommands": str(OUTPUT / "commands.log"),
    }
    write_json(OUTPUT / "provenance_manifest.json", provenance)

    artifact_paths = sorted(
        path
        for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "schemaVersion": "e03.s10.artifact_manifest.v1",
        "researchStepId": "S10",
        "artifactCount": len(artifact_paths),
        "artifacts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(OUTPUT / "artifact_manifest.json", artifact_manifest)

    report = (OUTPUT / "research_step_full_results.md").read_text()
    handoff = {
        "schemaVersion": "e03.s10.handoff_validation.v1",
        "researchStepId": "S10",
        "success": all(
            phrase in report
            for phrase in (
                "Research step",
                "Completion status",
                "Artifacts written",
                "Validation result",
                "Outcome classification",
                "Caveats or blockers",
                "Lay summary",
                "Recommended next action",
                "Methods",
                "Commands",
                "Provenance",
            )
        )
        and not Path("/artifacts/research_steps/S11").exists(),
        "s11ArtifactsAbsent": not Path("/artifacts/research_steps/S11").exists(),
        "requiredArtifactsPresent": not missing,
    }
    write_json(OUTPUT / "handoff_validation.json", handoff)
    if not handoff["success"]:
        raise RuntimeError("S10 handoff validation failed")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
