#!/usr/bin/env python3
"""Learn and validate E07 S06 policy embeddings."""

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
import numpy as np
import pandas as pd
import sklearn

from platonic_space.behavior_predictor import DEFAULT_TARGET_COLUMNS
from platonic_space.policy_embeddings import (
    POLICY_EMBEDDING_CLAIM_BOUNDARY,
    POLICY_EMBEDDING_MODEL_VERSION,
    POLICY_EMBEDDING_SCHEMA_VERSION,
    baseline_comparison,
    build_policy_profiles,
    family_baseline_matrix,
    feature_columns,
    fit_behavior_embedding,
    heldout_behavior_retrieval_metrics,
    heldout_truth_columns,
    label_retrieval_metrics,
    matrix_from_profile,
    nearest_neighbor_table,
    performance_summary,
    random_baseline_matrix,
    source_code_baseline_matrix,
    split_behavior_frame,
    stability_metrics,
    target_error_weights,
    validate_policy_embeddings,
)
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S06"
STEP_NUMBER = 6
STEP_TITLE = "Learn policy embeddings"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_FRAME_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "modeling_frame.parquet"
DEFAULT_POLICY_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S02" / "policy_abstract_catalog.parquet"
DEFAULT_ERROR_LIMITS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "error_limits.parquet"
DEFAULT_MODEL_CARD_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "e07_behavior_predictor" / "model_card.json"
FOCUSED_TESTS = ["tests.test_e07_policy_embeddings"]


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
    columns = payload.get("targetColumns") or list(DEFAULT_TARGET_COLUMNS)
    return [str(column) for column in columns]


def plot_embeddings(embeddings: pd.DataFrame, path_png: Path, path_svg: Path) -> list[Path]:
    if embeddings.empty:
        return []
    source_order = sorted(embeddings["sourceExperimentId"].astype(str).unique())
    palette = plt.get_cmap("tab10")
    colors = {source: palette(index % 10) for index, source in enumerate(source_order)}
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for source, subset in embeddings.groupby("sourceExperimentId"):
        sizes = 18 + 8 * np.log1p(subset["rowCount"].to_numpy(dtype=float))
        ax.scatter(
            subset["embeddingX"],
            subset["embeddingY"],
            s=sizes,
            alpha=0.72,
            label=str(source),
            color=colors[str(source)],
            linewidths=0.25,
            edgecolors="black",
        )
    ax.set_xlabel("Behavior embedding PC1")
    ax.set_ylabel("Behavior embedding PC2")
    ax.set_title("E07 S06 policy embeddings")
    ax.grid(alpha=0.25)
    ax.legend(title="Source", fontsize=8)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)
    return [path_png, path_svg]


def coverage_markdown(path: Path, coverage: pd.DataFrame, target_weights: pd.DataFrame, status: Mapping[str, Any]) -> None:
    status_counts = coverage["embeddingStatus"].value_counts().rename_axis("embeddingStatus").reset_index(name="policyCount")
    lines = [
        f"# {STEP_ID} Policy Embedding Coverage Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status['artifactsWritten'])} files, including embedding tables, feature matrices, validation metrics, baseline comparisons, figures, model card, summary, status, and artifact manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Policy Coverage",
        "",
        dataframe_markdown(status_counts, limit=20),
        "",
        "## S05 Error-Limit Target Weights",
        "",
        dataframe_markdown(
            target_weights[["target", "availableRows", "robustScale", "s05MedianTestRmse", "s05ReliabilityWeight"]],
            limit=30,
        ),
        "",
        POLICY_EMBEDDING_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def validation_markdown(path: Path, validation: pd.DataFrame, retrieval: pd.DataFrame, stability: pd.DataFrame, comparisons: pd.DataFrame, status: Mapping[str, Any]) -> None:
    failed = validation[~validation["success"]]
    heldout = retrieval[retrieval["validationTask"].eq("heldout_behavior_profile_retrieval")]
    lines = [
        f"# {STEP_ID} Policy Embedding Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        "- Artifacts written: policy embeddings, feature/profile matrices, nearest neighbors, retrieval and stability metrics, baseline comparison, validation summary, model card, summary, status, and artifact manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Held-Out Behavior Retrieval",
        "",
        dataframe_markdown(heldout.sort_values(["k", "modelName"]), limit=40),
        "",
        "## Baseline Comparison",
        "",
        dataframe_markdown(comparisons.sort_values(["k", "baselineName"]), limit=40),
        "",
        "## Stability",
        "",
        dataframe_markdown(stability, limit=20),
        "",
        "## Failed Or Warning Checks",
        "",
        dataframe_markdown(failed, limit=80) if not failed.empty else "No failed validation checks.",
        "",
        POLICY_EMBEDDING_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def model_report_markdown(path: Path, status: Mapping[str, Any], perf: Mapping[str, Any], embeddings: pd.DataFrame, retrieval: pd.DataFrame) -> None:
    family_label = retrieval[
        retrieval["validationTask"].eq("label_neighbor_retrieval")
        & retrieval["label"].eq("policyFamily")
        & retrieval["k"].eq(5)
    ].copy()
    lines = [
        f"# {STEP_ID} Policy Embedding Model Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status['artifactsWritten'])} files, including the policy embeddings, S05 error-weighted feature matrix, source-code/family baselines, retrieval/stability validation, figure, model card, summary, and status artifacts.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Model",
        "",
        f"Weighted behavior PCA embedded {len(embeddings)} observed policies using S04 behavior records, S05 predictor metadata features, and S05 target error-limit weights.",
        f"At k=5, held-out behavior retrieval mean cosine was {perf['behaviorK5MeanCosine']:.6g}; the behavior embedding beat {perf['k5BaselineComparisonsWon']}/{perf['k5BaselineComparisonCount']} source/family/random baseline comparisons at k=5.",
        f"Split-half distance Spearman was {perf['splitHalfDistanceSpearman']:.6g}; mean top-neighbor Jaccard was {perf['splitHalfMeanNeighborJaccard']:.6g}.",
        "",
        "## Label Retrieval Context",
        "",
        dataframe_markdown(family_label.sort_values("modelName"), limit=20),
        "",
        POLICY_EMBEDDING_CLAIM_BOUNDARY,
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
        f"- Artifacts written: {len(status['artifactsWritten'])} files; primary outputs are the policy embedding table, feature/profile matrices, nearest-neighbor table, baseline comparison, retrieval/stability metrics, figure, model card, validation report, and status artifacts.",
        f"- Validation result: {status['validationResult']}",
        f"- Outcome classification: {status['outcomeClassification']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        "- Lay summary: S06 learned policy-level coordinates from observed behavior profiles and checked whether nearby policies retrieve similar held-out behavior better than source-code or family metadata baselines.",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        f"Performance anchor: k=5 held-out behavior retrieval mean cosine was {perf['behaviorK5MeanCosine']:.6g}; behavior embeddings beat {perf['k5BaselineComparisonsWon']}/{perf['k5BaselineComparisonCount']} compared baselines at k=5.",
        "",
        POLICY_EMBEDDING_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--modeling-frame-path", type=Path, default=DEFAULT_FRAME_PATH)
    parser.add_argument("--policy-catalog-path", type=Path, default=DEFAULT_POLICY_CATALOG_PATH)
    parser.add_argument("--error-limits-path", type=Path, default=DEFAULT_ERROR_LIMITS_PATH)
    parser.add_argument("--s05-model-card-path", type=Path, default=DEFAULT_MODEL_CARD_PATH)
    parser.add_argument("--min-policy-rows", type=int, default=10)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    result_dir = args.artifacts_dir / "results"
    figure_dir = args.artifacts_dir / "figures"
    model_dir = args.artifacts_dir / "models" / "e07_policy_embeddings"
    provenance_dir = args.artifacts_dir / "provenance"
    for directory in (step_dir, result_dir, figure_dir, model_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.modeling_frame_path)
    policy_catalog = pd.read_parquet(args.policy_catalog_path)
    error_limits = pd.read_parquet(args.error_limits_path)
    target_columns = target_columns_from_model_card(args.s05_model_card_path)

    target_weights = target_error_weights(frame, error_limits, target_columns)
    profiles, coverage = build_policy_profiles(frame, policy_catalog, target_columns, target_weights, min_policy_rows=args.min_policy_rows)
    embedded_profiles = profiles[profiles["embeddingStatus"].eq("embedded_behavior")].sort_values("abstractPolicyId").reset_index(drop=True)
    columns = feature_columns(embedded_profiles)
    embeddings, embedding_model, behavior_matrix = fit_behavior_embedding(embedded_profiles, columns, n_components=12)
    policy_ids = embeddings["abstractPolicyId"].astype(str).tolist()

    family_matrix, family_model = family_baseline_matrix(policy_catalog, policy_ids, n_components=12)
    source_matrix, source_model = source_code_baseline_matrix(policy_catalog, policy_ids, n_components=12)
    random_matrix = random_baseline_matrix(policy_ids, n_components=12)

    nearest = nearest_neighbor_table(policy_ids, behavior_matrix, "weighted_behavior_pca", k=10)
    label_metrics = pd.concat(
        [
            label_retrieval_metrics(policy_ids, behavior_matrix, embedded_profiles, "weighted_behavior_pca"),
            label_retrieval_metrics(policy_ids, family_matrix, embedded_profiles, "family_metadata_baseline"),
            label_retrieval_metrics(policy_ids, source_matrix, embedded_profiles, "source_code_baseline"),
            label_retrieval_metrics(policy_ids, random_matrix, embedded_profiles, "random_baseline"),
        ],
        ignore_index=True,
    )

    train_frame, holdout_frame = split_behavior_frame(frame)
    train_profiles, _ = build_policy_profiles(train_frame, policy_catalog, target_columns, target_weights, min_policy_rows=1)
    holdout_profiles, _ = build_policy_profiles(holdout_frame, policy_catalog, target_columns, target_weights, min_policy_rows=1)
    train_index = train_profiles.set_index("abstractPolicyId")
    holdout_index = holdout_profiles.set_index("abstractPolicyId")
    common_ids = [
        policy_id
        for policy_id in policy_ids
        if policy_id in train_index.index
        and policy_id in holdout_index.index
        and int(train_index.loc[policy_id, "rowCount"]) >= 5
        and int(holdout_index.loc[policy_id, "rowCount"]) >= 5
        and int(train_index.loc[policy_id, "observedTargetCount"]) > 0
        and int(holdout_index.loc[policy_id, "observedTargetCount"]) > 0
    ]
    train_common = train_index.loc[common_ids].reset_index()
    holdout_common = holdout_index.loc[common_ids].reset_index()
    _, _, behavior_train_matrix = fit_behavior_embedding(train_common, columns, n_components=12)
    _, _, behavior_holdout_matrix = fit_behavior_embedding(holdout_common, columns, n_components=12)
    family_common_matrix, _ = family_baseline_matrix(policy_catalog, common_ids, n_components=12)
    source_common_matrix, _ = source_code_baseline_matrix(policy_catalog, common_ids, n_components=12)
    random_common_matrix = random_baseline_matrix(common_ids, n_components=12)
    truth_columns = heldout_truth_columns(holdout_common)
    heldout_metrics = heldout_behavior_retrieval_metrics(
        common_ids,
        {
            "weighted_behavior_pca": behavior_train_matrix,
            "family_metadata_baseline": family_common_matrix,
            "source_code_baseline": source_common_matrix,
            "random_baseline": random_common_matrix,
        },
        holdout_common,
        truth_columns,
    )
    stability = stability_metrics(common_ids, behavior_train_matrix, behavior_holdout_matrix, k=10)
    retrieval_metrics = pd.concat([label_metrics, heldout_metrics], ignore_index=True)
    comparisons = baseline_comparison(retrieval_metrics)
    perf = performance_summary(retrieval_metrics, comparisons, stability)

    feature_matrix_df = pd.DataFrame(matrix_from_profile(embedded_profiles, columns), columns=columns)
    feature_matrix_df.insert(0, "abstractPolicyId", policy_ids)

    model_bundle = {
        "schemaVersion": POLICY_EMBEDDING_SCHEMA_VERSION,
        "modelVersion": POLICY_EMBEDDING_MODEL_VERSION,
        "embeddingModel": embedding_model,
        "familyBaselineModel": family_model,
        "sourceCodeBaselineModel": source_model,
        "targetWeights": target_weights.to_dict(orient="records"),
        "policyIds": policy_ids,
        "claimBoundary": POLICY_EMBEDDING_CLAIM_BOUNDARY,
    }
    model_path = model_dir / "embedding_model_bundle.joblib"
    joblib.dump(model_bundle, model_path, compress=3)
    validation = validate_policy_embeddings(embeddings, coverage, retrieval_metrics, stability, model_path)
    hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

    profile_paths = write_dataframe(profiles, step_dir / "policy_behavior_profiles", csv=False)
    coverage_paths = write_dataframe(coverage, step_dir / "policy_embedding_coverage")
    target_weight_paths = write_dataframe(target_weights, step_dir / "target_error_weights")
    feature_matrix_paths = write_dataframe(feature_matrix_df, step_dir / "policy_embedding_feature_matrix", csv=False)
    embedding_paths = write_dataframe(embeddings, step_dir / "policy_embeddings")
    result_embedding_path = result_dir / "e07_policy_embeddings.parquet"
    embeddings.to_parquet(result_embedding_path, index=False)
    nearest_paths = write_dataframe(nearest, step_dir / "nearest_neighbors")
    retrieval_paths = write_dataframe(retrieval_metrics, step_dir / "retrieval_metrics")
    result_retrieval_path = result_dir / "e07_policy_embedding_retrieval_metrics.parquet"
    retrieval_metrics.to_parquet(result_retrieval_path, index=False)
    comparison_paths = write_dataframe(comparisons, step_dir / "baseline_comparison")
    stability_paths = write_dataframe(stability, step_dir / "stability_metrics")
    validation_paths = write_dataframe(validation, step_dir / "embedding_validation")
    figure_paths = plot_embeddings(embeddings, figure_dir / "e07_s06_policy_embeddings.png", figure_dir / "e07_s06_policy_embeddings.svg")

    model_card = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": POLICY_EMBEDDING_SCHEMA_VERSION,
        "modelVersion": POLICY_EMBEDDING_MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "modelingFramePath": str(args.modeling_frame_path),
        "modelingFrameSha256": sha256_path(args.modeling_frame_path),
        "policyCatalogPath": str(args.policy_catalog_path),
        "policyCatalogSha256": sha256_path(args.policy_catalog_path),
        "s05ErrorLimitsPath": str(args.error_limits_path),
        "s05ErrorLimitsSha256": sha256_path(args.error_limits_path),
        "targetColumns": target_columns,
        "targetWeights": target_weights.to_dict(orient="records"),
        "embeddedPolicyCount": int(len(embeddings)),
        "observedPrimaryPolicyCount": int((coverage["rowCount"] > 0).sum()),
        "unobservedCatalogPolicyCount": int(coverage["embeddingStatus"].eq("unobserved_in_s04_primary_policy").sum()),
        "featureCount": int(len(columns)),
        "explainedVarianceRatio": embedding_model["explainedVarianceRatio"],
        "performanceSummary": perf,
        "claimBoundary": POLICY_EMBEDDING_CLAIM_BOUNDARY,
        "sklearnVersion": sklearn.__version__,
    }
    model_card_path = model_dir / "model_card.json"
    write_json(model_card_path, model_card)

    test_command = None
    if not args.skip_tests:
        test_command = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS], cwd=REPO_ROOT)

    tests_ok = True if test_command is None else bool(test_command["success"])
    validation_ok = hard_failures.empty
    success = bool(validation_ok and tests_ok)
    validation_result = (
        f"{'passed' if success else 'failed'}: {len(embeddings)} policies embedded, "
        f"{len(retrieval_metrics)} retrieval metric rows, {len(stability)} stability metric rows, "
        f"{len(hard_failures)} hard validation failures; focused tests "
        f"{'skipped' if test_command is None else 'passed' if tests_ok else 'failed'}"
    )
    supportive = bool(success and perf["k5BaselineComparisonCount"] > 0 and perf["k5BaselineComparisonsWon"] >= 2 and perf["splitHalfDistanceSpearman"] > 0)
    outcome = "supportive" if supportive else ("null" if success else "constraining/contradictory")
    caveats = (
        "S06 embeds only policies observed as primary policies in S04 behavior rows with enough behavior coverage. "
        "The remaining catalog-only policies are documented but not assigned learned behavior coordinates. "
        "Behavior profiles are heterogeneous summary proxies weighted by S05 surrogate error limits, so distances "
        "reflect available simulator measurements and missingness patterns rather than direct biological or causal equivalence."
    )
    recommended = "Stop for Chief Scientist review before S07; if accepted, proceed to S07 goal embeddings using S04 behavior profiles, S05 error limits, and S06 policy-neighborhood diagnostics."
    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "title": STEP_TITLE,
        "artifactsWritten": [],
        "validationResult": validation_result,
        "outcomeClassification": outcome,
        "caveatsOrBlockers": caveats if success else caveats + " Review failed validation/test checks before S07.",
        "recommendedNextAction": recommended,
        "performanceSummary": perf,
        "focusedTestCommand": test_command,
        "claimBoundary": POLICY_EMBEDDING_CLAIM_BOUNDARY,
    }

    validation_summary_path = step_dir / "embedding_validation_summary.json"
    write_json(
        validation_summary_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": success,
            "hardValidationFailureCount": int(len(hard_failures)),
            "warningFailureCount": int(len(validation[(validation["severity"].eq("warning")) & (~validation["success"])])),
            "embeddedPolicyCount": int(len(embeddings)),
            "performanceSummary": perf,
            "validationResult": validation_result,
            "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
            "recommendedNextAction": recommended,
        },
    )

    coverage_report_path = step_dir / "coverage_report.md"
    validation_report_path = step_dir / "validation_report.md"
    model_report_path = step_dir / "model_report.md"
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    artifact_paths = [
        *profile_paths,
        *coverage_paths,
        *target_weight_paths,
        *feature_matrix_paths,
        *embedding_paths,
        result_embedding_path,
        *nearest_paths,
        *retrieval_paths,
        result_retrieval_path,
        *comparison_paths,
        *stability_paths,
        *validation_paths,
        *figure_paths,
        model_path,
        model_card_path,
        validation_summary_path,
        coverage_report_path,
        validation_report_path,
        model_report_path,
        summary_path,
        status_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    status_payload["artifactsWritten"] = collect_artifacts(
        [path for path in artifact_paths if path not in {coverage_report_path, validation_report_path, model_report_path, summary_path, status_path, artifact_manifest_path, run_manifest_path}]
    )
    coverage_markdown(coverage_report_path, coverage, target_weights, status_payload)
    validation_markdown(validation_report_path, validation, retrieval_metrics, stability, comparisons, status_payload)
    model_report_markdown(model_report_path, status_payload, perf, embeddings, retrieval_metrics)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": POLICY_EMBEDDING_SCHEMA_VERSION,
        "modelVersion": POLICY_EMBEDDING_MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "status": status_payload["status"],
        "artifactsWritten": collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path]),
        "validationResult": status_payload["validationResult"],
        "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "claimBoundary": POLICY_EMBEDDING_CLAIM_BOUNDARY,
    }
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    coverage_markdown(coverage_report_path, coverage, target_weights, status_payload)
    validation_markdown(validation_report_path, validation, retrieval_metrics, stability, comparisons, status_payload)
    model_report_markdown(model_report_path, status_payload, perf, embeddings, retrieval_metrics)
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
            "errorLimitsPath": str(args.error_limits_path),
            "s05ModelCardPath": str(args.s05_model_card_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threading": "serial sklearn PCA/SVD; no GPU used",
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
    coverage_markdown(coverage_report_path, coverage, target_weights, status_payload)
    validation_markdown(validation_report_path, validation, retrieval_metrics, stability, comparisons, status_payload)
    model_report_markdown(model_report_path, status_payload, perf, embeddings, retrieval_metrics)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
