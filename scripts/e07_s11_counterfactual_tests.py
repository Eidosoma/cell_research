#!/usr/bin/env python3
"""Run E07 S11 counterfactual prediction tests with direct replay validation."""

from __future__ import annotations

import argparse
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from platonic_space.counterfactual_tests import (
    CLASS_STRATIFICATION_NOTE,
    COUNTERFACTUAL_CLAIM_BOUNDARY,
    COUNTERFACTUAL_MODEL_VERSION,
    COUNTERFACTUAL_SCHEMA_VERSION,
    DEFAULT_TEST_FAMILIES,
    attach_universality_classes,
    claim_boundary,
    class_coverage_summary,
    dataframe_json_columns,
    error_limit_table,
    performance_summary,
    replay_records_for_candidates,
    score_heldout_predictions,
    select_counterfactual_candidates,
    target_performance_summary,
    validate_candidates,
    validation_checks,
)
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S11"
STEP_NUMBER = 11
STEP_TITLE = "Run counterfactual prediction tests"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_MODELING_FRAME_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "modeling_frame.parquet"
DEFAULT_S05_METRICS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "evaluation_metrics.parquet"
DEFAULT_S05_ERROR_LIMITS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "error_limits.parquet"
DEFAULT_S04_CORPUS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S04" / "unified_behavior_corpus.parquet"
DEFAULT_S10_CLASSES_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S10" / "universality_classes.parquet"
FOCUSED_TESTS = ["tests.test_e07_counterfactual_tests"]


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


def write_dataframe(df: pd.DataFrame, stem: Path, csv: bool = True) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if csv:
        csv_path = stem.with_suffix(".csv")
        dataframe_json_columns(df).to_csv(csv_path, index=False)
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


def dataframe_markdown(df: pd.DataFrame, limit: int = 50) -> str:
    if df.empty:
        return "No rows."
    display = df.head(limit).copy()
    return markdown_table(display.astype(str).to_dict(orient="records"), [str(column) for column in display.columns])


def plot_prediction_observation(validation: pd.DataFrame, path_png: Path, path_svg: Path) -> list[Path]:
    if validation.empty:
        return []
    preferred_targets = ["final_sortedness_percent", "aggregation", "delayed_gratification", "completed"]
    targets = [target for target in preferred_targets if target in set(validation["target"].astype(str))]
    if not targets:
        targets = validation["target"].astype(str).value_counts().head(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, target in zip(axes_flat, targets):
        subset = validation[validation["target"].astype(str).eq(target)].copy()
        if subset.empty:
            ax.axis("off")
            continue
        for split_name, split_rows in subset.groupby("splitName", sort=True):
            ax.scatter(split_rows["predictionMean"], split_rows["observedMean"], s=22, alpha=0.75, label=split_name)
        lo = float(np.nanmin([subset["predictionMean"].min(), subset["observedMean"].min()]))
        hi = float(np.nanmax([subset["predictionMean"].max(), subset["observedMean"].max()]))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black", linewidth=0.8)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(target)
        ax.set_xlabel("Predicted mean")
        ax.set_ylabel("Observed replay mean")
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(targets) :]:
        ax.axis("off")
    if targets:
        axes_flat[0].legend(fontsize=7)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)
    return [path_png, path_svg]


def write_candidate_report(
    path: Path,
    status: Mapping[str, Any],
    selection_diagnostics: pd.DataFrame,
    candidates: pd.DataFrame,
    class_coverage: pd.DataFrame,
) -> None:
    display_cols = [
        "counterfactualFamily",
        "target",
        "splitName",
        "eligibleScenarioCount",
        "selectedScenarioCount",
        "selectedClassCount",
    ]
    candidate_cols = [
        "candidateId",
        "counterfactualFamily",
        "target",
        "splitName",
        "primaryPolicyId",
        "worldId",
        "universalityClassId",
        "predictionMean",
        "observedMean",
        "observedRecordCount",
    ]
    lines = [
        f"# {STEP_ID} Candidate Selection Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; primary candidate, prediction, validation, coverage, report, figure, model-card, status, manifest, and results artifacts are listed in `status.json`.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Class Use Boundary",
        "",
        CLASS_STRATIFICATION_NOTE,
        "",
        "## Selection Diagnostics",
        "",
        dataframe_markdown(selection_diagnostics[[column for column in display_cols if column in selection_diagnostics.columns]], limit=80),
        "",
        "## Selected Candidates",
        "",
        dataframe_markdown(candidates[[column for column in candidate_cols if column in candidates.columns]], limit=80),
        "",
        "## S10 Class Coverage",
        "",
        dataframe_markdown(class_coverage, limit=30),
        "",
        "## Claim Boundary",
        "",
        claim_boundary(),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_report(
    path: Path,
    status: Mapping[str, Any],
    validation: pd.DataFrame,
    target_summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    validation_cols = [
        "candidateId",
        "counterfactualFamily",
        "target",
        "splitName",
        "predictionMean",
        "observedMean",
        "absoluteError",
        "s05AbsErrorP95",
        "observedWithinS05P95",
        "directValidationMode",
        "sourceTableVerifiedFraction",
    ]
    failed = checks[~checks["success"]] if not checks.empty else pd.DataFrame()
    lines = [
        f"# {STEP_ID} Counterfactual Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; validation rows, replay records, prediction-observation comparisons, target summaries, validation checks, figure, status, summary, and manifest are listed in `status.json`.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Prediction Versus Replay",
        "",
        dataframe_markdown(validation[[column for column in validation_cols if column in validation.columns]], limit=80),
        "",
        "## Target Summary",
        "",
        dataframe_markdown(target_summary, limit=80),
        "",
        "## Failed Or Warning Checks",
        "",
        dataframe_markdown(failed, limit=80) if not failed.empty else "No failed checks.",
        "",
        "## Replay Boundary",
        "",
        "Direct validation here means the selected S04 behavior records were traced back to read-only upstream simulator result tables, their table hashes matched S04 provenance, and their source row indices were in bounds. Where raw trace links exist, they are recorded as replay links; candidates without trace links still validate against the upstream simulator result row that S04 normalized.",
        "",
        "## Claim Boundary",
        "",
        claim_boundary(),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, status: Mapping[str, Any], perf: Mapping[str, Any]) -> None:
    lines = [
        f"# {STEP_ID} Status Summary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; primary outputs are counterfactual candidates, held-out predictions, direct replay validation, prediction-observation comparison, coverage summaries, validation report, figure, model card, status, and manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Outcome classification: {status['outcomeClassification']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        "- Lay summary: S11 asked the S05-style behavior predictor to nominate held-out policy/world/goal/perturbation cases, then checked those predictions against upstream simulator replay rows while using S10 classes only to spread candidate coverage.",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        f"Anchor result: {perf['candidateCount']} candidates across {perf['targetCount']} targets, {perf['splitCount']} held-out split types, and {perf['classCount']} S10 strata were validated; {perf['overallWithinS05P95Fraction']:.4f} fell within S05 P95 error limits, and source-table replay verification averaged {perf['overallSourceTableVerifiedFraction']:.4f}.",
        "",
        CLASS_STRATIFICATION_NOTE,
        "",
        claim_boundary(),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--modeling-frame-path", type=Path, default=DEFAULT_MODELING_FRAME_PATH)
    parser.add_argument("--s05-metrics-path", type=Path, default=DEFAULT_S05_METRICS_PATH)
    parser.add_argument("--s05-error-limits-path", type=Path, default=DEFAULT_S05_ERROR_LIMITS_PATH)
    parser.add_argument("--s04-corpus-path", type=Path, default=DEFAULT_S04_CORPUS_PATH)
    parser.add_argument("--s10-classes-path", type=Path, default=DEFAULT_S10_CLASSES_PATH)
    parser.add_argument("--max-candidates-per-spec", type=int, default=24)
    parser.add_argument("--max-candidates-per-class", type=int, default=2)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    result_dir = args.artifacts_dir / "results"
    model_dir = args.artifacts_dir / "models" / "e07_counterfactual_tests"
    figure_dir = args.artifacts_dir / "figures"
    provenance_dir = args.artifacts_dir / "provenance"
    for directory in (step_dir, result_dir, model_dir, figure_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.modeling_frame_path)
    s05_metrics = pd.read_parquet(args.s05_metrics_path)
    s05_error_limits = error_limit_table(pd.read_parquet(args.s05_error_limits_path))
    s10_classes = pd.read_parquet(args.s10_classes_path)
    s04_corpus = pd.read_parquet(args.s04_corpus_path)

    targets = sorted({str(spec["target"]) for spec in DEFAULT_TEST_FAMILIES})
    predictions, diagnostics = score_heldout_predictions(frame, s05_metrics, target_columns=targets)
    predictions = attach_universality_classes(predictions, s10_classes)
    candidates, selection_diagnostics = select_counterfactual_candidates(
        predictions,
        DEFAULT_TEST_FAMILIES,
        max_candidates_per_spec=args.max_candidates_per_spec,
        max_candidates_per_class=args.max_candidates_per_class,
    )
    replay_records = replay_records_for_candidates(candidates, s04_corpus)
    validation = validate_candidates(candidates, replay_records, s05_error_limits)
    target_summary = target_performance_summary(validation)
    class_coverage = class_coverage_summary(validation)
    checks = validation_checks(predictions, candidates, validation, replay_records, diagnostics)
    perf = performance_summary(validation, target_summary, checks)

    test_command = None
    if not args.skip_tests:
        test_command = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS], cwd=REPO_ROOT)
    tests_ok = True if test_command is None else bool(test_command["success"])
    hard_failures = checks[(checks["severity"].eq("error")) & (~checks["success"])]
    success = bool(hard_failures.empty and tests_ok)
    if not success:
        outcome = "constraining/contradictory"
    elif perf["overallWithinS05P95Fraction"] >= 0.65 and perf["overallSourceTableVerifiedFraction"] >= 0.95:
        outcome = "supportive"
    elif perf["candidateCount"] > 0 and perf["overallSourceTableVerifiedFraction"] >= 0.95:
        outcome = "null"
    else:
        outcome = "constraining/contradictory"
    validation_result = (
        f"{'passed' if success else 'failed'}: {perf['candidateCount']} held-out counterfactual candidates, "
        f"{perf['targetCount']} targets, {perf['classCount']} S10 strata used for coverage only, "
        f"within-S05-P95 fraction {perf['overallWithinS05P95Fraction']:.4f}, "
        f"source-table replay verification {perf['overallSourceTableVerifiedFraction']:.4f}, "
        f"{perf['hardValidationFailureCount']} hard validation failures; focused tests "
        f"{'skipped' if test_command is None else 'passed' if tests_ok else 'failed'}"
    )
    caveats = (
        "Counterfactual candidates are selected from already observed S04 held-out test partitions, so validation is replay/holdout validation rather than prospective wet-lab or new biological validation. "
        f"{CLASS_STRATIFICATION_NOTE} S05 error intervals are empirical held-out surrogate limits and can be wide or poorly calibrated for sparse targets; single-row candidate groups lack observed replicate CIs. "
        "Direct replay means verified upstream simulator result rows and raw-trace links where available, not re-execution of every upstream simulator kernel."
    )
    if not tests_ok:
        caveats += " Focused tests failed and must be reviewed before S12."
    recommended = "Stop for Chief Scientist review before S12; if accepted, use S11 replay-validated successes/failures and caveats to decide whether S12 should attempt inverse design or narrow to better-supported target/split families."

    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "title": STEP_TITLE,
        "artifactsWritten": [],
        "validationResult": validation_result,
        "outcomeClassification": outcome,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
        "performanceSummary": perf,
        "focusedTestCommand": test_command,
        "claimBoundary": COUNTERFACTUAL_CLAIM_BOUNDARY,
    }

    prediction_paths = write_dataframe(predictions, step_dir / "heldout_counterfactual_predictions", csv=False)
    diagnostic_paths = write_dataframe(diagnostics, step_dir / "split_model_diagnostics")
    candidate_paths = write_dataframe(candidates, step_dir / "counterfactual_candidates")
    selection_paths = write_dataframe(selection_diagnostics, step_dir / "candidate_selection_diagnostics")
    replay_paths = write_dataframe(replay_records, step_dir / "direct_replay_records", csv=False)
    validation_paths = write_dataframe(validation, step_dir / "direct_validation_results")
    comparison_paths = write_dataframe(validation, step_dir / "prediction_observation_comparison")
    target_summary_paths = write_dataframe(target_summary, step_dir / "target_performance_summary")
    class_coverage_paths = write_dataframe(class_coverage, step_dir / "class_coverage")
    check_paths = write_dataframe(checks, step_dir / "counterfactual_validation")
    result_path = result_dir / "e07_counterfactual_tests.parquet"
    validation.to_parquet(result_path, index=False)
    result_summary_path = result_dir / "e07_counterfactual_summary.parquet"
    target_summary.to_parquet(result_summary_path, index=False)
    figure_paths = plot_prediction_observation(
        validation,
        figure_dir / "e07_s11_prediction_vs_observation.png",
        figure_dir / "e07_s11_prediction_vs_observation.svg",
    )

    validation_summary_path = step_dir / "counterfactual_validation_summary.json"
    model_card_path = model_dir / "model_card.json"
    candidate_report_path = step_dir / "candidate_selection_report.md"
    validation_report_path = step_dir / "validation_report.md"
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    write_json(
        validation_summary_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": success,
            "status": status_payload["status"],
            "validationResult": validation_result,
            "performanceSummary": perf,
            "hardValidationFailureCount": int(len(hard_failures)),
            "warningFailureCount": int(len(checks[(checks["severity"].eq("warning")) & (~checks["success"])])),
            "caveatsOrBlockers": caveats,
            "recommendedNextAction": recommended,
        },
    )
    write_json(
        model_card_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "schemaVersion": COUNTERFACTUAL_SCHEMA_VERSION,
            "modelVersion": COUNTERFACTUAL_MODEL_VERSION,
            "createdAt": datetime.now(UTC).isoformat(),
            "modelingFramePath": str(args.modeling_frame_path),
            "modelingFrameSha256": sha256_path(args.modeling_frame_path),
            "s05MetricsPath": str(args.s05_metrics_path),
            "s05MetricsSha256": sha256_path(args.s05_metrics_path),
            "s05ErrorLimitsPath": str(args.s05_error_limits_path),
            "s05ErrorLimitsSha256": sha256_path(args.s05_error_limits_path),
            "s04CorpusPath": str(args.s04_corpus_path),
            "s04CorpusSha256": sha256_path(args.s04_corpus_path),
            "s10ClassesPath": str(args.s10_classes_path),
            "s10ClassesSha256": sha256_path(args.s10_classes_path),
            "candidateFamilies": list(DEFAULT_TEST_FAMILIES),
            "performanceSummary": perf,
            "classUseBoundary": CLASS_STRATIFICATION_NOTE,
            "claimBoundary": claim_boundary(),
            "packageVersions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        },
    )

    artifact_paths = [
        *prediction_paths,
        *diagnostic_paths,
        *candidate_paths,
        *selection_paths,
        *replay_paths,
        *validation_paths,
        *comparison_paths,
        *target_summary_paths,
        *class_coverage_paths,
        *check_paths,
        result_path,
        result_summary_path,
        *figure_paths,
        validation_summary_path,
        model_card_path,
        candidate_report_path,
        validation_report_path,
        summary_path,
        status_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    status_payload["artifactsWritten"] = collect_artifacts(artifact_paths)
    write_candidate_report(candidate_report_path, status_payload, selection_diagnostics, candidates, class_coverage)
    write_validation_report(validation_report_path, status_payload, validation, target_summary, checks)
    write_summary(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    input_paths = [
        args.modeling_frame_path,
        args.s05_metrics_path,
        args.s05_error_limits_path,
        args.s04_corpus_path,
        args.s10_classes_path,
    ]
    final_artifacts = collect_artifacts(artifact_paths)
    status_payload["artifactsWritten"] = final_artifacts
    write_json(status_path, status_payload)
    write_candidate_report(candidate_report_path, status_payload, selection_diagnostics, candidates, class_coverage)
    write_validation_report(validation_report_path, status_payload, validation, target_summary, checks)
    write_summary(summary_path, status_payload, perf)

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": COUNTERFACTUAL_SCHEMA_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "status": status_payload["status"],
        "success": success,
        "artifactsWritten": final_artifacts,
        "inputArtifacts": [
            {"path": str(path), "exists": path.exists(), "sha256": sha256_path(path) if path.exists() and path.is_file() else None}
            for path in input_paths
        ],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
        "claimBoundary": claim_boundary(),
    }
    write_json(artifact_manifest_path, manifest_payload)

    run_manifest = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "argv": sys.argv[1:],
        "createdAt": datetime.now(UTC).isoformat(),
        "elapsedSeconds": round(time.perf_counter() - started, 6),
        "status": status_payload["status"],
        "success": success,
        "gitCommit": git_value(["rev-parse", "HEAD"]),
        "gitBranch": git_value(["branch", "--show-current"]),
        "gitStatusShort": git_value(["status", "--short"]),
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "packageVersions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "performanceSummary": perf,
        "artifactsWritten": collect_artifacts([*artifact_paths, artifact_manifest_path, status_path, summary_path, run_manifest_path]),
        "claimBoundary": claim_boundary(),
    }
    write_json(run_manifest_path, run_manifest)

    final_artifacts = collect_artifacts([*artifact_paths, artifact_manifest_path, status_path, summary_path, run_manifest_path])
    status_payload["artifactsWritten"] = final_artifacts
    write_json(status_path, status_payload)
    manifest_payload["artifactsWritten"] = final_artifacts
    write_json(artifact_manifest_path, manifest_payload)
    write_candidate_report(candidate_report_path, status_payload, selection_diagnostics, candidates, class_coverage)
    write_validation_report(validation_report_path, status_payload, validation, target_summary, checks)
    write_summary(summary_path, status_payload, perf)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
