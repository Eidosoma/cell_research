#!/usr/bin/env python3
"""Finalize S13 from sealed trajectory tables after execution-only completion."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import pandas as pd

from scripts.build_regeneration_s13 import (
    E01_COMPOSITION,
    OUTPUT,
    REPOSITORY,
    S12_REPORT,
    S12_RESULTS,
    SPEC_PATH,
    WORKERS,
    _composition_contrasts,
    _controller_contrasts,
    _critical_thresholds,
    _pareto,
    _placement_sensitivity,
    _plots,
    _policy_summary,
    _terminal_hazards,
    _validation,
    canonical_hash,
    file_sha256,
    git_output,
    write_json,
)
from src.regeneration.chimeric import fault_count, fault_identity_ranking


def main() -> None:
    specification = json.loads(SPEC_PATH.read_text())
    sources = pd.read_parquet(OUTPUT / "source_checkpoints.parquet")
    results = pd.read_parquet(OUTPUT / "chimeric_recovery.parquet")
    assignments = pd.read_parquet(OUTPUT / "threshold_assignments.parquet")
    main_results = results[results["taskKind"] != "critical_fault_threshold"].copy()
    threshold = results[results["taskKind"] == "critical_fault_threshold"].copy()
    source_success = sources.set_index("sourceId")["sourceSuccess"].astype(bool).to_dict()

    # Populate outcome-blind fault assignments even for source competing
    # terminals, for which no faulted trajectory could be executed.
    for index, row in threshold.iterrows():
        identities = tuple(f"cell-{item:04d}" for item in range(int(row["n"])))
        ranking = fault_identity_ranking(
            identities,
            n=int(row["n"]),
            direction=str(row["direction"]),
            replicate=int(row["replicateOrdinal"]),
        )
        selected = tuple(ranking[: fault_count(int(row["n"]), float(row["faultFraction"]))])
        threshold.at[index, "faultIdentityIdsJson"] = json.dumps(sorted(selected))
        threshold.at[index, "faultIdentitySetSha256"] = canonical_hash(sorted(selected))
        if float(row["faultFraction"]) == 0.0 and not source_success[str(row["sourceId"])]:
            threshold.at[index, "executionStatus"] = "reused_main_source_competing_terminal"

    status_by_case = threshold.set_index("thresholdCaseId")["executionStatus"].to_dict()
    for index, row in assignments.iterrows():
        case_id = str(row["thresholdCaseId"])
        if case_id in status_by_case:
            assignments.at[index, "executionStatus"] = status_by_case[case_id]
    combined = pd.concat([main_results, threshold], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["taskKind", "taskId", "portfolioId", "placementMap", "faultFraction", "sourceId"]
    ).reset_index(drop=True)
    combined.to_parquet(OUTPUT / "chimeric_recovery.parquet", index=False)
    assignments.to_parquet(OUTPUT / "threshold_assignments.parquet", index=False)

    contrasts = _composition_contrasts(main_results)
    placement = _placement_sensitivity(main_results)
    controller = _controller_contrasts(main_results)
    policy_summary = _policy_summary(combined)
    threshold_curve, critical = _critical_thresholds(threshold)
    hazards = _terminal_hazards(combined)
    pareto = _pareto(main_results)
    injury = main_results[
        (main_results["taskKind"] == "injury")
        & (main_results["placementMap"] == 0)
        & main_results["trajectoryExecuted"].astype(bool)
    ]
    lesion_block_audit = (
        injury.groupby(["baseDrawPairingId", "taskId"], sort=True)
        .agg(
            representedPortfolios=("portfolioId", "nunique"),
            windowLengthUniqueCount=("lesionWindowLength", "nunique"),
            realizedDistanceUniqueCount=("lesionPostDistance", "nunique"),
            realizedDistanceMin=("lesionPostDistance", "min"),
            realizedDistanceMax=("lesionPostDistance", "max"),
        )
        .reset_index()
    )
    lesion_distribution = (
        injury.groupby(["taskId", "portfolioId"], sort=True)
        .agg(
            assignedExecutableRuns=("runId", "size"),
            meanPostDistance=("lesionPostDistance", "mean"),
            standardDeviationPostDistance=("lesionPostDistance", "std"),
            minimumPostDistance=("lesionPostDistance", "min"),
            maximumPostDistance=("lesionPostDistance", "max"),
            meanWindowLength=("lesionWindowLength", "mean"),
        )
        .reset_index()
    )
    lesion_block_audit["rngPairingStatus"] = lesion_block_audit["taskId"].map(
        {
            "segment_reversal_central_v1": "base_draw_paired_exact_severity",
            "local_scramble_sattolo_v1": "base_window_operator_paired_scenario_id_rng_unpaired",
        }
    )
    contrasts.to_parquet(OUTPUT / "composition_contrasts.parquet", index=False)
    placement.to_parquet(OUTPUT / "placement_sensitivity.parquet", index=False)
    controller.to_parquet(OUTPUT / "controller_contrasts.parquet", index=False)
    policy_summary.to_parquet(OUTPUT / "policy_family_summary.parquet", index=False)
    threshold_curve.to_parquet(OUTPUT / "threshold_curves.parquet", index=False)
    critical.to_parquet(OUTPUT / "critical_thresholds.parquet", index=False)
    hazards.to_parquet(OUTPUT / "terminal_hazards.parquet", index=False)
    pareto.to_parquet(OUTPUT / "pareto_summary.parquet", index=False)
    lesion_block_audit.to_parquet(OUTPUT / "lesion_pairing_audit.parquet", index=False)
    lesion_distribution.to_parquet(OUTPUT / "lesion_severity_distribution.parquet", index=False)
    figures = _plots(pareto, threshold_curve)

    validation = _validation(
        specification,
        sources,
        main_results,
        threshold,
        assignments,
        [],
    )
    write_json(OUTPUT / "validation_summary.json", validation)
    write_json(OUTPUT / "worker_failures.json", [])
    benefits = int(contrasts["primaryBenefit"].sum())
    harms = int(contrasts["primaryHarm"].sum())
    intact_departures = int(
        main_results[
            (main_results["taskKind"] == "intact_stability")
            & main_results["trajectoryExecuted"].astype(bool)
            & main_results["noChangeAnyTargetDeparture"].fillna(False)
        ].shape[0]
    )
    classification = (
        "constraining/contradictory"
        if harms or intact_departures or not validation["allPassed"]
        else "supportive"
        if benefits
        else "null"
    )
    ucb_history: list[dict[str, Any]] = []
    for fraction, group in threshold_curve.groupby("faultFraction", sort=True):
        level: dict[str, Any] = {"faultFraction": float(fraction)}
        level.update(
            {
                str(row["portfolioId"]): float(row["wilsonHigh"])
                for _, row in group.iterrows()
            }
        )
        ucb_history.append(level)
    elapsed = (
        max(path.stat().st_mtime for path in [OUTPUT / "chimeric_recovery.parquet", *figures])
        - (OUTPUT / "s13_specification.json").stat().st_mtime
    )
    accounting = {
        "schemaVersion": "e05.s13.run-accounting.v1",
        "plannedSources": specification["panel"]["plannedSourceCount"],
        "builtSourceObjects": len(sources),
        "formedAndStabilizedSources": int(sources["sourceSuccess"].sum()),
        "sourceCompetingTerminals": int((~sources["sourceSuccess"].astype(bool)).sum()),
        "sourceWorkerFailures": 0,
        "plannedMainLogicalRows": specification["panel"]["plannedMainRuns"],
        "mainLogicalRows": len(main_results),
        "executedMainTrajectories": int(main_results["trajectoryExecuted"].sum()),
        "sourceTerminalMainRows": int((~main_results["trajectoryExecuted"].astype(bool)).sum()),
        "completedMainExactReplays": int(main_results["trajectoryExecuted"].sum()),
        "thresholdPotentialRows": len(assignments),
        "thresholdLogicalResultRows": len(threshold),
        "thresholdReusedZeroRows": int(assignments["executionStatus"].str.startswith("reused_main").sum()),
        "thresholdNewExecutedTrajectories": int(assignments["executionStatus"].eq("executed").sum()),
        "thresholdSourceTerminalRows": int(assignments["executionStatus"].eq("source_competing_terminal").sum()),
        "thresholdPrespecifiedStopRows": int(assignments["executionStatus"].str.startswith("not_run").sum()),
        "thresholdNewExactReplays": int(assignments["executionStatus"].eq("executed").sum()),
        "allLogicalResultRows": len(combined),
        "newTrajectoryExecutionsIncludingReplays": 2
        * (
            int(main_results["trajectoryExecuted"].sum())
            + int(assignments["executionStatus"].eq("executed").sum())
        ),
        "sourceDevelopmentExecutionsIncludingReplays": 2 * len(sources),
        "workerFailures": 0,
        "thresholdUcbHistory": ucb_history,
        "globalThresholdStopAfterFraction": 0.2,
        "outcomeClassification": classification,
        "primaryBenefits": benefits,
        "primaryHarms": harms,
        "intactDepartureRuns": intact_departures,
        "executionElapsedSecondsFromArtifactTimestamps": elapsed,
        "workers": WORKERS,
    }
    write_json(OUTPUT / "run_accounting.json", accounting)
    write_json(
        OUTPUT / "finalization_summary.json",
        {
            "schemaVersion": "e05.s13.finalization-summary.v1",
            "researchStepId": "S13",
            "finalizedAtUtc": datetime.now(timezone.utc).isoformat(),
            "source": "sealed S13 parquet outputs written after all trajectory execution",
            "trajectoryRerunsDuringFinalization": 0,
            "correction": "populated fault assignment IDs for source competing-terminal rows and distinguished reused source terminals before validation",
            "validationPassed": validation["allPassed"],
            "outcomeClassification": classification,
        },
    )

    artifact_paths = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and path.name not in {"provenance_manifest.json", "research_step_full_results.md", "status.json"}
    )
    manifest = {
        "schemaVersion": "e05.s13.provenance-manifest.v1",
        "researchStepId": "S13",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "branch": git_output("branch", "--show-current"),
        "sourceCommitAtExecution": git_output("rev-parse", "HEAD"),
        "workingTreeDirtyAtExecution": True,
        "executionCommand": "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 E05_S13_WORKERS=8 python scripts/build_regeneration_s13.py",
        "finalizationCommand": "PYTHONPATH=. python scripts/finalize_regeneration_s13.py",
        "workers": WORKERS,
        "threadEnvironment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "s12Results": {"path": str(S12_RESULTS), "sha256": file_sha256(S12_RESULTS)},
            "s12Report": {"path": str(S12_REPORT), "sha256": file_sha256(S12_REPORT)},
            "e01Composition": {"path": str(E01_COMPOSITION), "sha256": file_sha256(E01_COMPOSITION)},
            "specificationSource": {"path": str(SPEC_PATH), "sha256": file_sha256(SPEC_PATH)},
        },
        "artifacts": [
            {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        ],
        "outcomeClassification": classification,
    }
    write_json(OUTPUT / "provenance_manifest.json", manifest)
    print(json.dumps({"validation": validation, "accounting": accounting}, sort_keys=True))


if __name__ == "__main__":
    main()
