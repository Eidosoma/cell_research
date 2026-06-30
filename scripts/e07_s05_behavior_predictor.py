#!/usr/bin/env python3
"""Train and evaluate the E07 S05 behavior predictor."""

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

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from platonic_space.behavior_predictor import (
    PREDICTOR_CLAIM_BOUNDARY,
    PREDICTOR_MODEL_VERSION,
    PREDICTOR_SCHEMA_VERSION,
    baseline_comparison,
    build_modeling_frame,
    compact_json,
    performance_summary,
    selected_targets,
    target_coverage,
    train_evaluate_predictor,
    validation_checks,
)
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S05"
STEP_NUMBER = 5
STEP_TITLE = "Train a behavior predictor"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_CORPUS_PATH = DEFAULT_ARTIFACTS_DIR / "data" / "e07_unified_behavior_corpus.parquet"
DEFAULT_WORLD_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S01" / "normalized_world_catalog.parquet"
DEFAULT_POLICY_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S02" / "policy_abstract_catalog.parquet"
DEFAULT_GOAL_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S03" / "goal_catalog.parquet"
FOCUSED_TESTS = ["tests.test_e07_behavior_predictor"]


def run_command(command: Sequence[str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
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


def write_dataframe(df: pd.DataFrame, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    parquet_path = stem.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


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


def json_ready_df(df: pd.DataFrame, json_columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for column in json_columns:
        if column in out.columns:
            out[column] = out[column].map(compact_json)
    return out


def plot_calibration(calibration: pd.DataFrame, path_png: Path, path_svg: Path) -> list[Path]:
    subset = calibration[
        calibration["modelName"].eq("sparse_ridge")
        & calibration["partition"].eq("test")
        & calibration["splitName"].isin(["heldout_policy", "heldout_world", "heldout_goal", "heldout_perturbation"])
        & calibration["target"].isin(["final_sortedness_percent", "completed", "aggregation", "delayed_gratification"])
    ].copy()
    if subset.empty:
        return []
    targets = list(dict.fromkeys(subset["target"].tolist()))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, target in zip(axes_flat, targets):
        target_rows = subset[subset["target"].eq(target)]
        for split_name, split_rows in target_rows.groupby("splitName"):
            ax.plot(split_rows["predictionMean"], split_rows["observedMean"], marker="o", linewidth=1, label=split_name)
        lo = min(float(target_rows["predictionMean"].min()), float(target_rows["observedMean"].min()))
        hi = max(float(target_rows["predictionMean"].max()), float(target_rows["observedMean"].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=0.8)
        ax.set_title(target)
        ax.set_xlabel("Predicted bin mean")
        ax.set_ylabel("Observed bin mean")
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(targets) :]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=7)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)
    return [path_png, path_svg]


def write_model_report(path: Path, summary: Mapping[str, Any], target_cov: pd.DataFrame, comparisons: pd.DataFrame, status: Mapping[str, Any]) -> None:
    selected = target_cov[target_cov["selectedForTraining"]].copy()
    test_comparisons = comparisons[comparisons["partition"].eq("test")] if not comparisons.empty else pd.DataFrame()
    lines = [
        f"# {STEP_ID} Behavior Predictor Model Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status['artifactsWritten'])} files, including model bundle, target coverage, split manifest, metrics, baseline comparison, calibration tables, residual sample, figures, validation, summary, and status artifacts.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Selected Targets",
        "",
        dataframe_markdown(selected[["target", "availableRows", "coverageFraction", "sourceExperimentCountsJson"]], limit=30),
        "",
        "## Baseline Comparison",
        "",
        dataframe_markdown(
            test_comparisons[
                [
                    "target",
                    "splitName",
                    "baselineName",
                    "modelRmse",
                    "baselineRmse",
                    "rmseImprovementFraction",
                    "modelBeatsBaseline",
                ]
            ].sort_values(["splitName", "target", "baselineName"]),
            limit=80,
        )
        if not test_comparisons.empty
        else "No evaluated test comparisons.",
        "",
        "## Error Limits",
        "",
        f"Median test RMSE across evaluated sparse-ridge target/split pairs: {summary['medianTestRmse']:.6g}. Median test R2: {summary['medianTestR2']:.6g}.",
        f"The model beat the global-mean baseline in {summary['beatsGlobalMeanCount']}/{summary['globalMeanComparisonCount']} held-out test comparisons and the source-group mean baseline in {summary['beatsSourceGroupMeanCount']}/{summary['sourceGroupMeanComparisonCount']} held-out test comparisons.",
        "",
        "## Claim Boundary",
        "",
        PREDICTOR_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_report(path: Path, validation: pd.DataFrame, status: Mapping[str, Any]) -> None:
    failed = validation[~validation["success"]]
    lines = [
        f"# {STEP_ID} Behavior Predictor Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        "- Artifacts written: model bundle, model card, target coverage, split manifest, evaluation metrics, baseline comparison, calibration table, residual sample, calibration figure, validation summary, summary, status, and artifact manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Failed Or Warning Checks",
        "",
        dataframe_markdown(failed, limit=80) if not failed.empty else "No failed validation checks.",
        "",
        "## Claim Boundary",
        "",
        PREDICTOR_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, status: Mapping[str, Any], perf: Mapping[str, Any]) -> None:
    lines = [
        f"# {STEP_ID} Status Summary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status['artifactsWritten'])} files; primary outputs are the behavior predictor model bundle, held-out split manifest, metrics, baseline comparison, calibration/error reports, and validation artifacts.",
        f"- Validation result: {status['validationResult']}",
        f"- Outcome classification: {status['outcomeClassification']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        "- Lay summary: S05 trained a sparse behavior predictor from the S04 corpus and evaluated it against simple baselines under held-out policy, world, goal, and perturbation splits.",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        f"Performance anchor: sparse ridge beat the global-mean baseline in {perf['beatsGlobalMeanCount']}/{perf['globalMeanComparisonCount']} held-out test comparisons and the source-group mean baseline in {perf['beatsSourceGroupMeanCount']}/{perf['sourceGroupMeanComparisonCount']} held-out test comparisons.",
        "",
        PREDICTOR_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--world-catalog-path", type=Path, default=DEFAULT_WORLD_CATALOG_PATH)
    parser.add_argument("--policy-catalog-path", type=Path, default=DEFAULT_POLICY_CATALOG_PATH)
    parser.add_argument("--goal-catalog-path", type=Path, default=DEFAULT_GOAL_CATALOG_PATH)
    parser.add_argument("--min-target-rows", type=int, default=1000)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    model_dir = args.artifacts_dir / "models" / "e07_behavior_predictor"
    result_dir = args.artifacts_dir / "results"
    figure_dir = args.artifacts_dir / "figures"
    provenance_dir = args.artifacts_dir / "provenance"
    for directory in (step_dir, model_dir, result_dir, figure_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = build_modeling_frame(args.corpus_path, args.world_catalog_path, args.policy_catalog_path, args.goal_catalog_path)
    coverage = target_coverage(frame)
    target_columns = selected_targets(coverage, min_rows=args.min_target_rows)
    metrics, calibration, residual_sample, model_bundle, split_manifest = train_evaluate_predictor(frame, target_columns)
    comparisons = baseline_comparison(metrics)
    perf = performance_summary(metrics, comparisons)

    model_bundle_path = model_dir / "model_bundle.joblib"
    joblib.dump(model_bundle, model_bundle_path, compress=3)
    validation = validation_checks(frame, coverage, metrics, split_manifest, model_bundle_path)
    hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

    modeling_frame_path = step_dir / "modeling_frame.parquet"
    frame.to_parquet(modeling_frame_path, index=False)
    coverage_paths = write_dataframe(json_ready_df(coverage, ["sourceExperimentCountsJson"]), step_dir / "target_coverage")
    split_paths = write_dataframe(json_ready_df(split_manifest, ["targetNonNullRowsJson"]), step_dir / "split_manifest")
    metrics_paths = write_dataframe(metrics, step_dir / "evaluation_metrics")
    result_metrics_path = result_dir / "e07_behavior_predictor_metrics.parquet"
    metrics.to_parquet(result_metrics_path, index=False)
    comparison_paths = write_dataframe(comparisons, step_dir / "baseline_comparison")
    calibration_paths = write_dataframe(calibration, step_dir / "calibration_table")
    residual_path = step_dir / "prediction_residuals_sample.parquet"
    residual_sample.to_parquet(residual_path, index=False)
    error_limits = metrics[
        metrics["modelName"].eq("sparse_ridge") & metrics["status"].eq("evaluated")
    ][
        [
            "target",
            "splitName",
            "partition",
            "n",
            "rmse",
            "mae",
            "absErrorP50",
            "absErrorP90",
            "absErrorP95",
            "calibrationSlope",
            "predictionCorrelation",
            "brierScore",
            "binaryEce10",
        ]
    ].copy()
    error_limit_paths = write_dataframe(error_limits, step_dir / "error_limits")
    validation_paths = write_dataframe(validation, step_dir / "predictor_validation")

    model_card = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "predictorSchemaVersion": PREDICTOR_SCHEMA_VERSION,
        "predictorModelVersion": PREDICTOR_MODEL_VERSION,
        "trainedAt": datetime.now(UTC).isoformat(),
        "corpusPath": str(args.corpus_path),
        "corpusSha256": sha256_path(args.corpus_path),
        "targetColumns": target_columns,
        "splitGroupColumns": model_bundle["splitGroupColumns"],
        "featureColumns": model_bundle["featureColumns"],
        "performanceSummary": perf,
        "claimBoundary": PREDICTOR_CLAIM_BOUNDARY,
        "sklearnVersion": sklearn.__version__,
    }
    model_card_path = model_dir / "model_card.json"
    write_json(model_card_path, model_card)

    calibration_fig_paths = plot_calibration(
        calibration,
        figure_dir / "e07_s05_calibration_error_limits.png",
        figure_dir / "e07_s05_calibration_error_limits.svg",
    )

    test_command = None
    if not args.skip_tests:
        test_command = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS], cwd=REPO_ROOT)

    tests_ok = True if test_command is None else bool(test_command["success"])
    validation_ok = hard_failures.empty
    success = bool(validation_ok and tests_ok)
    validation_result = (
        f"{'passed' if success else 'failed'}: {len(target_columns)} targets selected, "
        f"{len(metrics[metrics['status'].eq('evaluated')])} evaluated metric rows, "
        f"{len(hard_failures)} hard validation failures; focused tests "
        f"{'skipped' if test_command is None else 'passed' if tests_ok else 'failed'}"
    )
    global_count = max(1, perf["globalMeanComparisonCount"])
    supportive = perf["beatsGlobalMeanCount"] / global_count >= 0.5 and success
    outcome = "supportive" if supportive else ("constraining/contradictory" if success else "constraining/contradictory")
    caveats = (
        "The S05 surrogate is a sparse linear model over heterogeneous S04 metadata and JSON-derived summaries. "
        "It is useful for bounded proxy prediction and error diagnostics, but held-out world/goal/perturbation "
        "performance is limited by sparse cross-world coverage, summary rows without exact policies, and targets "
        "that exist only within one upstream experiment. Rare morphology, symmetry, homeostasis, and dominance axes "
        "remain underpowered for a general predictor."
    )
    recommended = "Stop for Chief Scientist review before S06; if accepted, proceed to S06 policy embeddings using S04/S05 features and documented error limits."
    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "title": STEP_TITLE,
        "artifactsWritten": [],
        "validationResult": validation_result,
        "outcomeClassification": outcome,
        "caveatsOrBlockers": caveats if success else caveats + " Review failed validation/test checks before S06.",
        "recommendedNextAction": recommended,
        "targetColumns": target_columns,
        "performanceSummary": perf,
        "focusedTestCommand": test_command,
        "claimBoundary": PREDICTOR_CLAIM_BOUNDARY,
    }

    validation_summary_path = step_dir / "predictor_validation_summary.json"
    write_json(
        validation_summary_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": success,
            "hardValidationFailureCount": int(len(hard_failures)),
            "warningFailureCount": int(len(validation[(validation["severity"].eq("warning")) & (~validation["success"])])),
            "selectedTargetCount": len(target_columns),
            "performanceSummary": perf,
            "validationResult": validation_result,
            "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
            "recommendedNextAction": recommended,
        },
    )

    model_report_path = step_dir / "model_report.md"
    validation_report_path = step_dir / "validation_report.md"
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    artifact_paths = [
        modeling_frame_path,
        *coverage_paths,
        *split_paths,
        *metrics_paths,
        result_metrics_path,
        *comparison_paths,
        *calibration_paths,
        residual_path,
        *error_limit_paths,
        *validation_paths,
        validation_summary_path,
        model_bundle_path,
        model_card_path,
        *calibration_fig_paths,
        model_report_path,
        validation_report_path,
        summary_path,
        status_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    status_payload["artifactsWritten"] = collect_artifacts(
        [path for path in artifact_paths if path not in {model_report_path, validation_report_path, summary_path, status_path, artifact_manifest_path, run_manifest_path}]
    )
    write_model_report(model_report_path, perf, json_ready_df(coverage, ["sourceExperimentCountsJson"]), comparisons, status_payload)
    write_validation_report(validation_report_path, validation, status_payload)
    write_summary(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": PREDICTOR_SCHEMA_VERSION,
        "modelVersion": PREDICTOR_MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "status": status_payload["status"],
        "artifactsWritten": collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path]),
        "validationResult": status_payload["validationResult"],
        "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "claimBoundary": PREDICTOR_CLAIM_BOUNDARY,
    }
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    write_model_report(model_report_path, perf, json_ready_df(coverage, ["sourceExperimentCountsJson"]), comparisons, status_payload)
    write_validation_report(validation_report_path, validation, status_payload)
    write_summary(summary_path, status_payload, perf)
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
            "corpusPath": str(args.corpus_path),
            "worldCatalogPath": str(args.world_catalog_path),
            "policyCatalogPath": str(args.policy_catalog_path),
            "goalCatalogPath": str(args.goal_catalog_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threading": "serial sklearn sparse-ridge fits; no GPU used",
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "researchSteps": {STEP_ID: status_payload},
    }
    write_json(run_manifest_path, run_manifest)
    manifest_payload["artifactsWritten"] = collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path])
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    write_model_report(model_report_path, perf, json_ready_df(coverage, ["sourceExperimentCountsJson"]), comparisons, status_payload)
    write_validation_report(validation_report_path, validation, status_payload)
    write_summary(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
