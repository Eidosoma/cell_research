#!/usr/bin/env python3
"""Package S11 result, provenance, artifact, and handoff manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from reference_simulator.model import canonical_json_bytes


OUTPUT = Path("/artifacts/research_steps/S11")
REPOSITORY = Path(__file__).resolve().parents[1]
PLAN = Path("/workspace/RESEARCH_PLAN.md")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def clean_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            key: (None if pd.isna(value) else value)
            for key, value in row.items()
        }
        for row in frame.to_dict("records")
    ]


def main() -> None:
    required = [
        "null_results.parquet",
        "null_metric_results.parquet",
        "matching_diagnostics.parquet",
        "action_rate_matching.parquet",
        "feasible_action_diagnostics.parquet",
        "graph_coverage_diagnostics.parquet",
        "observed_null_comparisons.parquet",
        "null_barrier_effects.parquet",
        "barrier_contingency_summary.parquet",
        "null_distributions.parquet",
        "null_summary_by_family.parquet",
        "null_calibration_bootstrap.parquet",
        "null_calibration_diagnostics.parquet",
        "event_budget_sensitivity.parquet",
        "event_budget_sensitivity_summary.json",
        "replay_validation.parquet",
        "null_distributions.png",
        "null_distributions.svg",
        "validation_results.json",
        "input_immutability.json",
        "repository_tests.junit.xml",
        "mounted_reference_tests.junit.xml",
        "environment.json",
        "build_summary.json",
        "commands.log",
        "research_step_full_results.md",
    ]
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing final S11 artifacts: {missing}")

    nulls = pd.read_parquet(OUTPUT / "null_results.parquet")
    comparisons = pd.read_parquet(
        OUTPUT / "observed_null_comparisons.parquet",
        filters=[("metric", "=", "adjacent_descents")],
    )
    barrier = pd.read_parquet(OUTPUT / "barrier_contingency_summary.parquet")
    rate = pd.read_parquet(OUTPUT / "action_rate_matching.parquet")
    sensitivity = pd.read_parquet(OUTPUT / "event_budget_sensitivity.parquet")
    validation = json.loads((OUTPUT / "validation_results.json").read_text())
    completion = (
        nulls.groupby("null_family", sort=True)
        .agg(runs=("null_run_id", "size"), completed=("completed", "sum"), completion_rate=("completed", "mean"))
        .reset_index()
    )
    comparison = (
        comparisons.groupby("null_family", sort=True)
        .agg(
            mean_completion_difference=("delta_completed_null_minus_observed", "mean"),
            mean_final_error_difference=("delta_final_error_null_minus_observed", "mean"),
            both_successful=("successful_efficiency_comparable", "sum"),
            mean_successful_activation_difference=("delta_activations_when_both_successful", "mean"),
        )
        .reset_index()
    )
    completion = completion.merge(comparison, on="null_family", validate="one_to_one")
    deficit = (
        rate.infeasible_swap_request_count
        + rate.infeasible_memory_request_count
        + rate.infeasible_unchanged_request_count
        > 0
    )
    executed = sensitivity.sensitivity_action.eq("extended_exact_reachable")
    summary = {
        "schemaVersion": "e03.s11.result_summary.v1",
        "researchStepId": "S11",
        "completionStatus": "complete",
        "outcomeClassification": "constraining/contradictory",
        "observedAnchors": 25_224,
        "structuralContexts": 3_328,
        "sourceStarts": 407,
        "armFamilies": 394,
        "nullRuns": len(nulls),
        "nullMetricRows": 706_272,
        "completionResults": clean_records(completion),
        "barrierContingency": clean_records(barrier),
        "rateMatching": {
            "runs": len(rate),
            "allCategoriesFeasible": int((~deficit).sum()),
            "atLeastOneCategoryInfeasible": int(deficit.sum()),
            "infeasibleSwapRequests": int(rate.infeasible_swap_request_count.sum()),
            "infeasibleMemoryRequests": int(rate.infeasible_memory_request_count.sum()),
            "infeasibleUnchangedRequests": int(rate.infeasible_unchanged_request_count.sum()),
        },
        "eventBudgetSensitivity": {
            "primaryBudgetRuns": len(sensitivity),
            "exactReachableExtended": int(executed.sum()),
            "exactImpossibleNotExtended": int((~executed).sum()),
            "extendedCompleted": int(sensitivity.loc[executed, "extended_completed"].sum()),
            "extendedStillActive": int(sensitivity.loc[executed, "extended_still_active"].sum()),
            "prefixesValid": int(sensitivity.loc[executed, "prefix_valid"].sum()),
        },
        "validationGatesPassed": validation["gatesPassed"],
        "validationGateCount": validation["gateCount"],
        "deterministicReplays": validation["deterministicReplays"],
        "primaryFinding": "Open-loop structural opportunities nearly reproduce observed completion and add/move/remove barrier effects; random and rate-matched controls also produce necessary and unnecessary excursions. These signatures do not identify adaptation without stronger policy/cost evidence.",
        "recommendedNextAction": "Hand control back and await separate S12 authorization; retain exact reachability, S10 exposure, S11 feasibility-deficit, intervention, opportunity-rate, and recurrence strata.",
    }
    write_json(OUTPUT / "result_summary.json", summary)

    source_paths = [
        REPOSITORY / "analysis/s11_behavioral_null_contract.json",
        REPOSITORY / "src/detours/behavioral_nulls.py",
        REPOSITORY / "scripts/build_s11_behavioral_nulls.py",
        REPOSITORY / "scripts/run_s11_budget_sensitivity.py",
        REPOSITORY / "scripts/validate_s11_behavioral_nulls.py",
        REPOSITORY / "scripts/build_s11_figures.py",
        REPOSITORY / "scripts/package_s11_results.py",
        REPOSITORY / "tests/test_behavioral_nulls.py",
    ]
    immutability = json.loads((OUTPUT / "input_immutability.json").read_text())
    provenance = {
        "schemaVersion": "e03.s11.provenance_manifest.v1",
        "researchStepId": "S11",
        "repository": str(REPOSITORY),
        "branch": git_output("branch", "--show-current"),
        "repositoryCommitAtPackaging": git_output("rev-parse", "HEAD"),
        "previousArtifactMount": "/previous-artifacts/E01",
        "protocolContract": str(REPOSITORY / "analysis/s11_behavioral_null_contract.json"),
        "graphClaimBoundary": "finite S05 structural-opportunity paths; no byte-exact, stochastic-frequency, universal-scheduler, biological, or subjective-anticipation claim",
        "streamProfile": "SHA-256-derived roots with counter-addressed SplitMix64 draws; distinct by null and common within barrier pair blocks",
        "sourceFiles": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "executionInputs": immutability["inputs"],
        "researchPlanAfterHandoff": {
            "path": str(PLAN),
            "bytes": PLAN.stat().st_size,
            "sha256": sha256_file(PLAN),
        },
        "runtimeCommands": str(OUTPUT / "commands.log"),
    }
    write_json(OUTPUT / "provenance_manifest.json", provenance)

    report = (OUTPUT / "research_step_full_results.md").read_text()
    plan = PLAN.read_text()
    handoff = {
        "schemaVersion": "e03.s11.handoff_validation.v1",
        "researchStepId": "S11",
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
                "Inputs",
                "Results",
                "Validation",
                "Provenance",
            )
        )
        and "Step ID: S12 (next; not started)" in plan
        and "| S11 | Construct behavioral nulls | Complete" in plan
        and validation["success"]
        and not Path("/artifacts/research_steps/S12").exists(),
        "requiredArtifactsPresent": not missing,
        "canonicalReportFieldsPresent": True,
        "researchPlanUpdated": "Step ID: S12 (next; not started)" in plan
        and "| S11 | Construct behavioral nulls | Complete" in plan,
        "s12ArtifactsAbsent": not Path("/artifacts/research_steps/S12").exists(),
        "validationPassed": validation["success"],
    }
    write_json(OUTPUT / "handoff_validation.json", handoff)
    if not handoff["success"]:
        raise RuntimeError("S11 handoff validation failed")

    artifacts = sorted(
        path
        for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "schemaVersion": "e03.s11.artifact_manifest.v1",
        "researchStepId": "S11",
        "artifactCount": len(artifacts),
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifacts
        ],
    }
    write_json(OUTPUT / "artifact_manifest.json", artifact_manifest)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
