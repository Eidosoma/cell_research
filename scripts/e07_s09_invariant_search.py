#!/usr/bin/env python3
"""Search for E07 S09 empirical invariants with cross-world validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
for thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_var, "1")

import numpy as np
import pandas as pd
import sklearn

from platonic_space.behavior_predictor import DEFAULT_TARGET_COLUMNS
from platonic_space.invariant_search import (
    INVARIANT_FEATURE_GROUPS,
    INVARIANT_SEARCH_CLAIM_BOUNDARY,
    INVARIANT_SEARCH_MODEL_VERSION,
    INVARIANT_SEARCH_SCHEMA_VERSION,
    build_invariant_feature_frame,
    confound_adjusted_associations,
    cross_world_holdout_search,
    apply_cross_world_signal_caveat,
    invariant_candidate_table,
    invariant_feature_columns,
    model_comparison_table,
    summarize_coefficients,
    summarize_cross_world_metrics,
    summarize_group_ablation,
    target_error_weights,
    validate_invariant_search,
)
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S09"
STEP_NUMBER = 9
STEP_TITLE = "Search for invariants"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_FRAME_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "modeling_frame.parquet"
DEFAULT_POLICY_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S02" / "policy_abstract_catalog.parquet"
DEFAULT_GOAL_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S03" / "goal_catalog.parquet"
DEFAULT_WORLD_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S01" / "normalized_world_catalog.parquet"
DEFAULT_ERROR_LIMITS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "error_limits.parquet"
DEFAULT_S05_MODEL_CARD_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "e07_behavior_predictor" / "model_card.json"
DEFAULT_S08_DISTANCES_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S08" / "platonic_distances.parquet"
DEFAULT_S08_ENTITIES_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S08" / "distance_entities.parquet"
FOCUSED_TESTS = ["tests.test_e07_invariant_search"]


def run_command(command: Sequence[str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(thread_var, "1")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    return {
        "command": list(command),
        "cwd": str(cwd),
        "returncode": int(completed.returncode),
        "success": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsedSeconds": round(time.perf_counter() - started, 6),
    }


def git_value(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def write_dataframe(df: pd.DataFrame, stem: Path, csv: bool = True) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if csv:
        csv_path = stem.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        paths.append(csv_path)
    parquet_path = stem.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)
    paths.append(parquet_path)
    return paths


def collect_artifacts(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(out, key=lambda row: row["path"])


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("\n", " ").replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def dataframe_markdown(df: pd.DataFrame, limit: int = 60) -> str:
    if df.empty:
        return "No rows."
    display = df.head(limit).copy()
    return markdown_table(display.astype(str).to_dict(orient="records"), [str(column) for column in display.columns])


def target_columns_from_model_card(path: Path) -> list[str]:
    if not path.exists():
        return list(DEFAULT_TARGET_COLUMNS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(column) for column in payload.get("targetColumns", list(DEFAULT_TARGET_COLUMNS))]


def performance_summary(
    feature_frame: pd.DataFrame,
    metrics: pd.DataFrame,
    model_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    candidate_count = int(candidates["evidenceTier"].isin(["candidate_invariant", "weak_or_contextual_candidate"]).sum()) if not candidates.empty else 0
    strict_count = int(candidates["evidenceTier"].eq("candidate_invariant").sum()) if not candidates.empty else 0
    return {
        "featureFrameRows": int(len(feature_frame)),
        "invariantFeatureCount": int(len(invariant_feature_columns(feature_frame))),
        "eligibleTargetCount": int(metrics["target"].nunique()) if not metrics.empty else 0,
        "crossWorldMetricRows": int(len(metrics)),
        "targetModelSummaryRows": int(len(model_summary)),
        "combinedBeatsGlobalTargetCount": int((comparisons["combinedMinusGlobalR2"] > 0).sum()) if not comparisons.empty else 0,
        "combinedBeatsConfoundTargetCount": int((comparisons["combinedMinusConfoundR2"] > 0).sum()) if not comparisons.empty else 0,
        "invariantBeatsGlobalTargetCount": int((comparisons["invariantMinusGlobalR2"] > 0).sum()) if not comparisons.empty else 0,
        "ablationFeatureGroupCount": int(ablation_summary["featureGroup"].nunique()) if not ablation_summary.empty else 0,
        "candidateOrContextualInvariantCount": candidate_count,
        "strictCandidateInvariantCount": strict_count,
        "topCandidateFeatureGroup": str(candidates["featureGroup"].iloc[0]) if not candidates.empty else "none",
        "topCandidateEvidenceTier": str(candidates["evidenceTier"].iloc[0]) if not candidates.empty else "none",
        "topCandidateScore": float(candidates["candidateInvariantScore"].iloc[0]) if not candidates.empty else float("nan"),
    }


def invariant_report_markdown(
    path: Path,
    status: Mapping[str, Any],
    perf: Mapping[str, Any],
    candidates: pd.DataFrame,
    comparisons: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    associations: pd.DataFrame,
) -> None:
    top_assoc = associations.sort_values("absConfoundAdjustedPearsonR", ascending=False).head(25) if not associations.empty else pd.DataFrame()
    lines = [
        f"# {STEP_ID} Invariant Candidate Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status['artifactsWritten'])} files, including invariant feature frame, cross-world holdout metrics, confound-aware model comparisons, group ablations, adjusted association tables, candidate table, validation report, status, and manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Claim Boundary",
        "",
        INVARIANT_SEARCH_CLAIM_BOUNDARY,
        "",
        "## Cross-World Model Result",
        "",
        f"Eligible targets: {perf['eligibleTargetCount']}; invariant-only ridge beat the global mean on {perf['invariantBeatsGlobalTargetCount']} targets, and the combined invariant+confound model beat confound-only metadata on {perf['combinedBeatsConfoundTargetCount']} targets.",
        "",
        "## Candidate Groups",
        "",
        dataframe_markdown(candidates, limit=20),
        "",
        "## Combined-Versus-Confound Comparison By Target",
        "",
        dataframe_markdown(comparisons.sort_values("combinedMinusConfoundR2", ascending=False), limit=40),
        "",
        "## Group Ablation Summary",
        "",
        dataframe_markdown(ablation_summary.sort_values("meanDeltaR2WhenRemoved", ascending=False), limit=20),
        "",
        "## Strongest Confound-Adjusted Associations",
        "",
        dataframe_markdown(top_assoc, limit=25),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def uncertainty_report_markdown(
    path: Path,
    status: Mapping[str, Any],
    eligibility: pd.DataFrame,
    target_weights: pd.DataFrame,
    model_summary: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    low_weight = target_weights.sort_values("s05ReliabilityWeight").head(16) if "s05ReliabilityWeight" in target_weights.columns else target_weights
    lines = [
        f"# {STEP_ID} Uncertainty And Caveat Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: invariant feature frame, target eligibility, S05 target weights, model metrics, uncertainty summaries, candidate table, validation report, summary, status, and manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Target Eligibility",
        "",
        dataframe_markdown(eligibility, limit=30),
        "",
        "## S05 Error-Limit Weights",
        "",
        dataframe_markdown(low_weight[["target", "availableRows", "robustScale", "s05MedianTestRmse", "s05ReliabilityWeight"]], limit=30),
        "",
        "## Cross-World R2 Uncertainty",
        "",
        dataframe_markdown(model_summary.sort_values(["target", "modelName"]), limit=80),
        "",
        "## Interpretation Caveats",
        "",
        "- Cross-world holdout performance tests transfer across observed S04 worlds, not future direct simulations.",
        "- Confound-aware comparisons use metadata controls for source experiment, policy family, goal family, substrate, and source kind, but cannot remove all artifact structure.",
        "- S08 Platonic position and sparse-coverage features can improve prediction by marking under-sampled or source-specific regimes; those signals are caveats, not mechanistic invariants.",
        "- Coefficients and ablations are computational associations. They do not prove causal mechanisms.",
        "",
        "## Candidate-Tier Caveat",
        "",
        dataframe_markdown(candidates[["featureGroup", "evidenceTier", "meanDeltaR2WhenRemoved", "positiveDeltaFraction", "meanAbsConfoundAdjustedCorrelation"]], limit=20),
        "",
        INVARIANT_SEARCH_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def validation_markdown(path: Path, validation: pd.DataFrame, status: Mapping[str, Any]) -> None:
    failed = validation[~validation["success"]]
    lines = [
        f"# {STEP_ID} Invariant Search Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        "- Artifacts written: invariant feature frame, centrality table, target eligibility, cross-world metrics, summaries, comparisons, group ablations, coefficients, adjusted associations, candidate table, model card, summary, status, and manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Validation Checks",
        "",
        dataframe_markdown(validation, limit=80),
        "",
        "## Failed Or Warning Checks",
        "",
        dataframe_markdown(failed, limit=80) if not failed.empty else "No failed validation checks.",
        "",
        INVARIANT_SEARCH_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_markdown(path: Path, status: Mapping[str, Any], perf: Mapping[str, Any]) -> None:
    lines = [
        f"# {STEP_ID} Status Summary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status['artifactsWritten'])} files; primary outputs are invariant feature frame, cross-world holdout metrics, model comparisons, group ablation summary, confound-adjusted associations, invariant candidate table, caveat report, validation report, model card, status, and manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Outcome classification: {status['outcomeClassification']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        "- Lay summary: S09 asks whether compact policy, world, goal, and S08 distance features can predict behavior when whole worlds are held out. It reports candidate computational regularities and flags where metadata or sparse coverage may explain the signal.",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        f"Anchor result: {perf['eligibleTargetCount']} targets evaluated with cross-world holdouts; combined invariant+confound model beat confound-only metadata on {perf['combinedBeatsConfoundTargetCount']} targets; top candidate group is `{perf['topCandidateFeatureGroup']}` ({perf['topCandidateEvidenceTier']}).",
        "",
        INVARIANT_SEARCH_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--modeling-frame-path", type=Path, default=DEFAULT_FRAME_PATH)
    parser.add_argument("--policy-catalog-path", type=Path, default=DEFAULT_POLICY_CATALOG_PATH)
    parser.add_argument("--goal-catalog-path", type=Path, default=DEFAULT_GOAL_CATALOG_PATH)
    parser.add_argument("--world-catalog-path", type=Path, default=DEFAULT_WORLD_CATALOG_PATH)
    parser.add_argument("--error-limits-path", type=Path, default=DEFAULT_ERROR_LIMITS_PATH)
    parser.add_argument("--s05-model-card-path", type=Path, default=DEFAULT_S05_MODEL_CARD_PATH)
    parser.add_argument("--s08-distances-path", type=Path, default=DEFAULT_S08_DISTANCES_PATH)
    parser.add_argument("--s08-entities-path", type=Path, default=DEFAULT_S08_ENTITIES_PATH)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    result_dir = args.artifacts_dir / "results"
    model_dir = args.artifacts_dir / "models" / "e07_invariant_search"
    provenance_dir = args.artifacts_dir / "provenance"
    for directory in (step_dir, result_dir, model_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.modeling_frame_path)
    policy_catalog = pd.read_parquet(args.policy_catalog_path)
    goal_catalog = pd.read_parquet(args.goal_catalog_path)
    world_catalog = pd.read_parquet(args.world_catalog_path)
    error_limits = pd.read_parquet(args.error_limits_path)
    s08_distances = pd.read_parquet(args.s08_distances_path)
    s08_entities = pd.read_parquet(args.s08_entities_path)
    target_columns = target_columns_from_model_card(args.s05_model_card_path)
    target_weights = target_error_weights(frame, error_limits, target_columns)

    feature_frame, centrality = build_invariant_feature_frame(frame, policy_catalog, goal_catalog, world_catalog, s08_distances, s08_entities)
    metrics, group_ablation, coefficients, eligibility = cross_world_holdout_search(
        feature_frame,
        target_columns=target_columns,
        max_targets=args.max_targets,
    )
    model_summary = summarize_cross_world_metrics(metrics, target_weights)
    comparisons = model_comparison_table(model_summary)
    ablation_summary = summarize_group_ablation(group_ablation)
    associations = confound_adjusted_associations(feature_frame, target_columns, max_targets=args.max_targets)
    coefficient_summary = summarize_coefficients(coefficients)
    candidates = invariant_candidate_table(ablation_summary, associations, coefficient_summary)
    candidates = apply_cross_world_signal_caveat(candidates, comparisons)
    validation = validate_invariant_search(feature_frame, metrics, model_summary, ablation_summary, candidates, associations)
    hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

    feature_paths = write_dataframe(feature_frame, step_dir / "invariant_feature_frame", csv=False)
    centrality_paths = write_dataframe(centrality, step_dir / "s08_entity_centrality")
    target_weight_paths = write_dataframe(target_weights, step_dir / "target_error_weights")
    eligibility_paths = write_dataframe(eligibility, step_dir / "target_eligibility")
    metric_paths = write_dataframe(metrics, step_dir / "cross_world_holdout_metrics")
    summary_paths = write_dataframe(model_summary, step_dir / "target_model_summary")
    comparison_paths = write_dataframe(comparisons, step_dir / "model_comparison")
    group_ablation_paths = write_dataframe(group_ablation, step_dir / "feature_group_ablation")
    ablation_summary_paths = write_dataframe(ablation_summary, step_dir / "feature_group_ablation_summary")
    coefficient_paths = write_dataframe(coefficients, step_dir / "ridge_coefficients", csv=False)
    coefficient_summary_paths = write_dataframe(coefficient_summary, step_dir / "ridge_coefficient_summary")
    association_paths = write_dataframe(associations, step_dir / "confound_adjusted_associations")
    candidate_paths = write_dataframe(candidates, step_dir / "invariant_candidates")
    validation_paths = write_dataframe(validation, step_dir / "invariant_validation")
    result_candidate_path = result_dir / "e07_invariant_search.parquet"
    candidates.to_parquet(result_candidate_path, index=False)
    result_metric_path = result_dir / "e07_invariant_search_cross_world_metrics.parquet"
    metrics.to_parquet(result_metric_path, index=False)

    test_command = None
    if not args.skip_tests:
        test_command = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS], cwd=REPO_ROOT)
    tests_ok = True if test_command is None else bool(test_command["success"])
    success = bool(hard_failures.empty and tests_ok)
    perf = performance_summary(feature_frame, metrics, model_summary, comparisons, ablation_summary, candidates)
    validation_result = (
        f"{'passed' if success else 'failed'}: {perf['featureFrameRows']} feature rows, "
        f"{perf['eligibleTargetCount']} eligible targets, {perf['crossWorldMetricRows']} cross-world metric rows, "
        f"{len(hard_failures)} hard validation failures; focused tests "
        f"{'skipped' if test_command is None else 'passed' if tests_ok else 'failed'}"
    )
    if not success:
        outcome = "constraining/contradictory"
    elif perf["combinedBeatsGlobalTargetCount"] == 0 and perf["combinedBeatsConfoundTargetCount"] == 0:
        outcome = "constraining/contradictory"
    elif perf["strictCandidateInvariantCount"] > 0 and perf["combinedBeatsConfoundTargetCount"] > 0:
        outcome = "supportive"
    else:
        outcome = "null"
    caveats = (
        "S09 identifies empirical computational regularities, not causal mechanisms. Cross-world holdouts test transfer "
        "across S04-observed worlds only; metadata confounds remain possible despite source/policy/world/goal controls. "
        "S08 Platonic centrality and sparse-coverage features can mark missingness or source-family structure, so any "
        "coverage-related signal is treated as a caveat. In this run, invariant and combined models did not beat global "
        "or confound baselines on any eligible target, so the candidate groups are not robust cross-world predictors. "
        "Rare morphology, homeostasis, dominance, and under-sampled goals remain lower-power axes."
    )
    recommended = "Stop for Chief Scientist review before S10; if accepted, use S09 candidate invariants, confound caveats, and S08 distances to define universality classes in S10."
    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "title": STEP_TITLE,
        "artifactsWritten": [],
        "validationResult": validation_result,
        "outcomeClassification": outcome,
        "caveatsOrBlockers": caveats if success else caveats + " Review failed validation/test checks before S10.",
        "recommendedNextAction": recommended,
        "performanceSummary": perf,
        "focusedTestCommand": test_command,
        "claimBoundary": INVARIANT_SEARCH_CLAIM_BOUNDARY,
    }

    validation_summary_path = step_dir / "invariant_validation_summary.json"
    write_json(
        validation_summary_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": success,
            "hardValidationFailureCount": int(len(hard_failures)),
            "warningFailureCount": int(len(validation[(validation["severity"].eq("warning")) & (~validation["success"])])),
            "performanceSummary": perf,
            "validationResult": validation_result,
            "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
            "recommendedNextAction": recommended,
        },
    )

    model_card = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": INVARIANT_SEARCH_SCHEMA_VERSION,
        "modelVersion": INVARIANT_SEARCH_MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "modelingFramePath": str(args.modeling_frame_path),
        "modelingFrameSha256": sha256_path(args.modeling_frame_path),
        "policyCatalogPath": str(args.policy_catalog_path),
        "policyCatalogSha256": sha256_path(args.policy_catalog_path),
        "goalCatalogPath": str(args.goal_catalog_path),
        "goalCatalogSha256": sha256_path(args.goal_catalog_path),
        "worldCatalogPath": str(args.world_catalog_path),
        "worldCatalogSha256": sha256_path(args.world_catalog_path),
        "s08DistancesPath": str(args.s08_distances_path),
        "s08DistancesSha256": sha256_path(args.s08_distances_path),
        "s08EntitiesPath": str(args.s08_entities_path),
        "s08EntitiesSha256": sha256_path(args.s08_entities_path),
        "targetColumns": target_columns,
        "invariantFeatureGroups": {key: list(value) for key, value in INVARIANT_FEATURE_GROUPS.items()},
        "performanceSummary": perf,
        "claimBoundary": INVARIANT_SEARCH_CLAIM_BOUNDARY,
        "sklearnVersion": sklearn.__version__,
    }
    model_card_path = model_dir / "model_card.json"
    write_json(model_card_path, model_card)

    invariant_report_path = step_dir / "invariant_candidate_report.md"
    uncertainty_report_path = step_dir / "uncertainty_caveat_report.md"
    validation_report_path = step_dir / "validation_report.md"
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    artifact_paths = [
        *feature_paths,
        *centrality_paths,
        *target_weight_paths,
        *eligibility_paths,
        *metric_paths,
        *summary_paths,
        *comparison_paths,
        *group_ablation_paths,
        *ablation_summary_paths,
        *coefficient_paths,
        *coefficient_summary_paths,
        *association_paths,
        *candidate_paths,
        *validation_paths,
        result_candidate_path,
        result_metric_path,
        model_card_path,
        validation_summary_path,
        invariant_report_path,
        uncertainty_report_path,
        validation_report_path,
        summary_path,
        status_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    status_payload["artifactsWritten"] = collect_artifacts(
        [
            path
            for path in artifact_paths
            if path
            not in {invariant_report_path, uncertainty_report_path, validation_report_path, summary_path, status_path, artifact_manifest_path, run_manifest_path}
        ]
    )
    invariant_report_markdown(invariant_report_path, status_payload, perf, candidates, comparisons, ablation_summary, associations)
    uncertainty_report_markdown(uncertainty_report_path, status_payload, eligibility, target_weights, model_summary, candidates)
    validation_markdown(validation_report_path, validation, status_payload)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": INVARIANT_SEARCH_SCHEMA_VERSION,
        "modelVersion": INVARIANT_SEARCH_MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "status": status_payload["status"],
        "artifactsWritten": collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path]),
        "validationResult": status_payload["validationResult"],
        "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "claimBoundary": INVARIANT_SEARCH_CLAIM_BOUNDARY,
    }
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    invariant_report_markdown(invariant_report_path, status_payload, perf, candidates, comparisons, ablation_summary, associations)
    uncertainty_report_markdown(uncertainty_report_path, status_payload, eligibility, target_weights, model_summary, candidates)
    validation_markdown(validation_report_path, validation, status_payload)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    run_manifest = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "startedAtApprox": datetime.now(UTC).isoformat(),
        "elapsedSeconds": round(time.perf_counter() - started, 6),
        "command": [sys.executable, *sys.argv],
        "repo": {
            "root": str(REPO_ROOT),
            "branch": git_value(["branch", "--show-current"]),
            "commit": git_value(["rev-parse", "HEAD"]),
            "dirtyStatus": git_value(["status", "--short"]),
        },
        "inputs": {
            "modelingFramePath": str(args.modeling_frame_path),
            "policyCatalogPath": str(args.policy_catalog_path),
            "goalCatalogPath": str(args.goal_catalog_path),
            "worldCatalogPath": str(args.world_catalog_path),
            "errorLimitsPath": str(args.error_limits_path),
            "s05ModelCardPath": str(args.s05_model_card_path),
            "s08DistancesPath": str(args.s08_distances_path),
            "s08EntitiesPath": str(args.s08_entities_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threading": "serial sklearn ridge; no GPU used",
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "researchSteps": {STEP_ID: status_payload},
    }
    write_json(run_manifest_path, run_manifest)
    manifest_payload["artifactsWritten"] = collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path])
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    invariant_report_markdown(invariant_report_path, status_payload, perf, candidates, comparisons, ablation_summary, associations)
    uncertainty_report_markdown(uncertainty_report_path, status_payload, eligibility, target_weights, model_summary, candidates)
    validation_markdown(validation_report_path, validation, status_payload)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
