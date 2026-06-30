#!/usr/bin/env python3
"""Learn and validate E07 S07 goal embeddings."""

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
from platonic_space.goal_embeddings import (
    GOAL_EMBEDDING_CLAIM_BOUNDARY,
    GOAL_EMBEDDING_MODEL_VERSION,
    GOAL_EMBEDDING_SCHEMA_VERSION,
    baseline_comparison,
    build_goal_profiles,
    feature_columns,
    fit_goal_embedding,
    goal_metadata_baseline_matrix,
    heldout_goal_retrieval_metrics,
    heldout_truth_columns,
    label_retrieval_metrics,
    nearest_neighbor_table,
    performance_summary,
    policy_neighborhood_baseline_matrix,
    random_baseline_matrix,
    split_behavior_frame,
    stability_metrics,
    target_error_weights,
    validate_goal_embeddings,
)
from platonic_space.policy_embeddings import matrix_from_profile
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S07"
STEP_NUMBER = 7
STEP_TITLE = "Learn goal embeddings"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_FRAME_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "modeling_frame.parquet"
DEFAULT_CORPUS_PATH = DEFAULT_ARTIFACTS_DIR / "data" / "e07_unified_behavior_corpus.parquet"
DEFAULT_GOAL_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S03" / "goal_catalog.parquet"
DEFAULT_ERROR_LIMITS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "error_limits.parquet"
DEFAULT_S05_MODEL_CARD_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "e07_behavior_predictor" / "model_card.json"
DEFAULT_POLICY_EMBEDDINGS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S06" / "policy_embeddings.parquet"
DEFAULT_POLICY_NEIGHBORS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S06" / "nearest_neighbors.parquet"
FOCUSED_TESTS = ["tests.test_e07_goal_embeddings"]


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
    family_order = sorted(embeddings["goalFamily"].astype(str).unique())
    palette = plt.get_cmap("tab20")
    colors = {family: palette(index % 20) for index, family in enumerate(family_order)}
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for family, subset in embeddings.groupby("goalFamily"):
        sizes = 35 + 10 * np.log1p(subset["rowCount"].to_numpy(dtype=float))
        ax.scatter(
            subset["embeddingX"],
            subset["embeddingY"],
            s=sizes,
            alpha=0.78,
            label=str(family),
            color=colors[str(family)],
            linewidths=0.3,
            edgecolors="black",
        )
    ax.set_xlabel("Goal embedding PC1")
    ax.set_ylabel("Goal embedding PC2")
    ax.set_title("E07 S07 goal embeddings")
    ax.grid(alpha=0.25)
    ax.legend(title="Goal family", fontsize=7, ncols=2)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)
    return [path_png, path_svg]


def coverage_markdown(path: Path, coverage: pd.DataFrame, uncertainty: pd.DataFrame, target_weights: pd.DataFrame, status: Mapping[str, Any]) -> None:
    status_counts = coverage["embeddingStatus"].value_counts().rename_axis("embeddingStatus").reset_index(name="goalCount")
    uncertainty_counts = uncertainty["sparseUncertaintyLevel"].value_counts().rename_axis("sparseUncertaintyLevel").reset_index(name="goalCount")
    high = uncertainty[uncertainty["sparseUncertaintyLevel"].eq("high")][
        ["abstractGoalId", "rowCount", "uniquePolicyCount", "observedTargetCount", "sparseUncertaintyLevel", "uncertaintyReason"]
    ].sort_values(["rowCount", "abstractGoalId"])
    lines = [
        f"# {STEP_ID} Goal Embedding Coverage And Uncertainty Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status['artifactsWritten'])} files, including goal embeddings, profiles, S05 target weights, S06 policy-neighborhood feature aggregates, validation metrics, uncertainty table, model card, summary, status, and artifact manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Goal Coverage",
        "",
        dataframe_markdown(status_counts, limit=20),
        "",
        "## Sparse Uncertainty Levels",
        "",
        dataframe_markdown(uncertainty_counts, limit=20),
        "",
        "## High-Uncertainty Goals",
        "",
        dataframe_markdown(high, limit=80),
        "",
        "## S05 Error-Limit Target Weights",
        "",
        dataframe_markdown(target_weights[["target", "availableRows", "robustScale", "s05MedianTestRmse", "s05ReliabilityWeight"]], limit=30),
        "",
        GOAL_EMBEDDING_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def validation_markdown(path: Path, validation: pd.DataFrame, retrieval: pd.DataFrame, stability: pd.DataFrame, comparisons: pd.DataFrame, status: Mapping[str, Any]) -> None:
    failed = validation[~validation["success"]]
    heldout = retrieval[retrieval["validationTask"].eq("heldout_goal_behavior_profile_retrieval")]
    lines = [
        f"# {STEP_ID} Goal Embedding Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        "- Artifacts written: goal embeddings, feature/profile matrices, nearest neighbors, retrieval and stability metrics, sparse uncertainty table, baseline comparison, validation summary, model card, summary, status, and artifact manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Held-Out Goal Behavior Retrieval",
        "",
        dataframe_markdown(heldout.sort_values(["k", "modelName"]), limit=60),
        "",
        "## Baseline Comparison",
        "",
        dataframe_markdown(comparisons.sort_values(["k", "baselineName"]), limit=60),
        "",
        "## Stability",
        "",
        dataframe_markdown(stability, limit=20),
        "",
        "## Failed Or Warning Checks",
        "",
        dataframe_markdown(failed, limit=80) if not failed.empty else "No failed validation checks.",
        "",
        GOAL_EMBEDDING_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def model_report_markdown(path: Path, status: Mapping[str, Any], perf: Mapping[str, Any], embeddings: pd.DataFrame, retrieval: pd.DataFrame) -> None:
    family_label = retrieval[
        retrieval["validationTask"].eq("goal_label_neighbor_retrieval")
        & retrieval["label"].eq("goalFamily")
        & retrieval["k"].eq(3)
    ].copy()
    lines = [
        f"# {STEP_ID} Goal Embedding Model Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status['artifactsWritten'])} files, including the goal embeddings, S04/S05 behavior profile matrix, S06 policy-neighborhood aggregates, goal retrieval/stability validation, sparse uncertainty report, figure, model card, summary, and status artifacts.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Model",
        "",
        f"Weighted goal behavior PCA embedded {len(embeddings)} observed goals using S04 behavior records, S05 target error-limit weights, and S06 policy-neighborhood diagnostics.",
        f"At k=3, held-out goal behavior retrieval mean cosine was {perf['goalK3MeanCosine']:.6g}; the goal embedding beat {perf['k3BaselineComparisonsWon']}/{perf['k3BaselineComparisonCount']} metadata/policy/random baseline comparisons at k=3.",
        f"Split-half distance Spearman was {perf['splitHalfDistanceSpearman']:.6g}; mean top-neighbor Jaccard was {perf['splitHalfMeanNeighborJaccard']:.6g}.",
        "",
        "## Label Retrieval Context",
        "",
        dataframe_markdown(family_label.sort_values("modelName"), limit=20),
        "",
        GOAL_EMBEDDING_CLAIM_BOUNDARY,
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
        f"- Artifacts written: {len(status['artifactsWritten'])} files; primary outputs are the goal embedding table, feature/profile matrix, nearest-neighbor table, sparse uncertainty report, baseline comparison, retrieval/stability metrics, figure, model card, validation report, and status artifacts.",
        f"- Validation result: {status['validationResult']}",
        f"- Outcome classification: {status['outcomeClassification']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        "- Lay summary: S07 learned goal-level coordinates from observed behavior profiles, S05 target reliability weights, and S06 policy-neighborhood diagnostics, then checked whether nearby goals retrieve similar held-out behavior better than simple metadata or policy-neighborhood baselines.",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        f"Performance anchor: k=3 held-out goal behavior retrieval mean cosine was {perf['goalK3MeanCosine']:.6g}; goal embeddings beat {perf['k3BaselineComparisonsWon']}/{perf['k3BaselineComparisonCount']} compared baselines at k=3. High-uncertainty goals: {perf['highUncertaintyGoalCount']}.",
        "",
        GOAL_EMBEDDING_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--modeling-frame-path", type=Path, default=DEFAULT_FRAME_PATH)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--goal-catalog-path", type=Path, default=DEFAULT_GOAL_CATALOG_PATH)
    parser.add_argument("--error-limits-path", type=Path, default=DEFAULT_ERROR_LIMITS_PATH)
    parser.add_argument("--s05-model-card-path", type=Path, default=DEFAULT_S05_MODEL_CARD_PATH)
    parser.add_argument("--policy-embeddings-path", type=Path, default=DEFAULT_POLICY_EMBEDDINGS_PATH)
    parser.add_argument("--policy-neighbors-path", type=Path, default=DEFAULT_POLICY_NEIGHBORS_PATH)
    parser.add_argument("--min-goal-rows", type=int, default=10)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    result_dir = args.artifacts_dir / "results"
    figure_dir = args.artifacts_dir / "figures"
    model_dir = args.artifacts_dir / "models" / "e07_goal_embeddings"
    provenance_dir = args.artifacts_dir / "provenance"
    for directory in (step_dir, result_dir, figure_dir, model_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.modeling_frame_path)
    corpus = pd.read_parquet(args.corpus_path)
    goal_catalog = pd.read_parquet(args.goal_catalog_path)
    error_limits = pd.read_parquet(args.error_limits_path)
    policy_embeddings = pd.read_parquet(args.policy_embeddings_path)
    policy_neighbors = pd.read_parquet(args.policy_neighbors_path)
    target_columns = target_columns_from_model_card(args.s05_model_card_path)

    target_weights = target_error_weights(frame, error_limits, target_columns)
    profiles, coverage, uncertainty = build_goal_profiles(
        frame,
        goal_catalog,
        policy_embeddings,
        policy_neighbors,
        target_columns,
        target_weights,
        corpus=corpus,
        min_goal_rows=args.min_goal_rows,
    )
    embedded_profiles = profiles[profiles["embeddingStatus"].eq("embedded_behavior")].sort_values("abstractGoalId").reset_index(drop=True)
    columns = feature_columns(embedded_profiles)
    embeddings, embedding_model, behavior_matrix = fit_goal_embedding(embedded_profiles, columns, n_components=8)
    goal_ids = embeddings["abstractGoalId"].astype(str).tolist()

    metadata_matrix, metadata_model = goal_metadata_baseline_matrix(goal_catalog, goal_ids, n_components=8)
    policy_baseline_matrix, policy_baseline_model = policy_neighborhood_baseline_matrix(embedded_profiles, n_components=8)
    random_matrix = random_baseline_matrix(goal_ids, n_components=8)

    nearest = nearest_neighbor_table(goal_ids, behavior_matrix, "weighted_goal_behavior_pca", k=10)
    label_metrics = pd.concat(
        [
            label_retrieval_metrics(goal_ids, behavior_matrix, embedded_profiles, "weighted_goal_behavior_pca"),
            label_retrieval_metrics(goal_ids, metadata_matrix, embedded_profiles, "goal_metadata_baseline"),
            label_retrieval_metrics(goal_ids, policy_baseline_matrix, embedded_profiles, "policy_neighborhood_baseline"),
            label_retrieval_metrics(goal_ids, random_matrix, embedded_profiles, "random_baseline"),
        ],
        ignore_index=True,
    )

    train_frame, holdout_frame = split_behavior_frame(frame)
    train_profiles, _, _ = build_goal_profiles(
        train_frame, goal_catalog, policy_embeddings, policy_neighbors, target_columns, target_weights, corpus=corpus, min_goal_rows=1
    )
    holdout_profiles, _, _ = build_goal_profiles(
        holdout_frame, goal_catalog, policy_embeddings, policy_neighbors, target_columns, target_weights, corpus=corpus, min_goal_rows=1
    )
    train_index = train_profiles.set_index("abstractGoalId")
    holdout_index = holdout_profiles.set_index("abstractGoalId")
    common_ids = [
        goal_id
        for goal_id in goal_ids
        if goal_id in train_index.index
        and goal_id in holdout_index.index
        and int(train_index.loc[goal_id, "rowCount"]) >= 5
        and int(holdout_index.loc[goal_id, "rowCount"]) >= 5
        and int(train_index.loc[goal_id, "observedTargetCount"]) > 0
        and int(holdout_index.loc[goal_id, "observedTargetCount"]) > 0
    ]
    train_common = train_index.loc[common_ids].reset_index()
    holdout_common = holdout_index.loc[common_ids].reset_index()
    _, _, behavior_train_matrix = fit_goal_embedding(train_common, columns, n_components=8)
    _, _, behavior_holdout_matrix = fit_goal_embedding(holdout_common, columns, n_components=8)
    metadata_common_matrix, _ = goal_metadata_baseline_matrix(goal_catalog, common_ids, n_components=8)
    policy_common_matrix, _ = policy_neighborhood_baseline_matrix(train_common, n_components=8)
    random_common_matrix = random_baseline_matrix(common_ids, n_components=8)
    truth_columns = heldout_truth_columns(holdout_common)
    heldout_metrics = heldout_goal_retrieval_metrics(
        common_ids,
        {
            "weighted_goal_behavior_pca": behavior_train_matrix,
            "goal_metadata_baseline": metadata_common_matrix,
            "policy_neighborhood_baseline": policy_common_matrix,
            "random_baseline": random_common_matrix,
        },
        holdout_common,
        truth_columns,
    )
    stability = stability_metrics(common_ids, behavior_train_matrix, behavior_holdout_matrix, k=5)
    retrieval_metrics = pd.concat([label_metrics, heldout_metrics], ignore_index=True)
    comparisons = baseline_comparison(retrieval_metrics)
    perf = performance_summary(retrieval_metrics, comparisons, stability, uncertainty)

    feature_matrix_df = pd.DataFrame(matrix_from_profile(embedded_profiles, columns), columns=columns)
    feature_matrix_df.insert(0, "abstractGoalId", goal_ids)

    model_bundle = {
        "schemaVersion": GOAL_EMBEDDING_SCHEMA_VERSION,
        "modelVersion": GOAL_EMBEDDING_MODEL_VERSION,
        "embeddingModel": embedding_model,
        "goalMetadataBaselineModel": metadata_model,
        "policyNeighborhoodBaselineModel": policy_baseline_model,
        "targetWeights": target_weights.to_dict(orient="records"),
        "goalIds": goal_ids,
        "claimBoundary": GOAL_EMBEDDING_CLAIM_BOUNDARY,
    }
    model_path = model_dir / "embedding_model_bundle.joblib"
    joblib.dump(model_bundle, model_path, compress=3)
    validation = validate_goal_embeddings(embeddings, coverage, uncertainty, retrieval_metrics, stability, model_path)
    hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

    profile_paths = write_dataframe(profiles, step_dir / "goal_behavior_profiles", csv=False)
    coverage_paths = write_dataframe(coverage, step_dir / "goal_embedding_coverage")
    uncertainty_paths = write_dataframe(uncertainty, step_dir / "goal_uncertainty")
    target_weight_paths = write_dataframe(target_weights, step_dir / "target_error_weights")
    feature_matrix_paths = write_dataframe(feature_matrix_df, step_dir / "goal_embedding_feature_matrix", csv=False)
    embedding_paths = write_dataframe(embeddings, step_dir / "goal_embeddings")
    result_embedding_path = result_dir / "e07_goal_embeddings.parquet"
    embeddings.to_parquet(result_embedding_path, index=False)
    nearest_paths = write_dataframe(nearest, step_dir / "nearest_neighbors")
    retrieval_paths = write_dataframe(retrieval_metrics, step_dir / "retrieval_metrics")
    result_retrieval_path = result_dir / "e07_goal_embedding_retrieval_metrics.parquet"
    retrieval_metrics.to_parquet(result_retrieval_path, index=False)
    comparison_paths = write_dataframe(comparisons, step_dir / "baseline_comparison")
    stability_paths = write_dataframe(stability, step_dir / "stability_metrics")
    validation_paths = write_dataframe(validation, step_dir / "embedding_validation")
    figure_paths = plot_embeddings(embeddings, figure_dir / "e07_s07_goal_embeddings.png", figure_dir / "e07_s07_goal_embeddings.svg")

    model_card = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": GOAL_EMBEDDING_SCHEMA_VERSION,
        "modelVersion": GOAL_EMBEDDING_MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "modelingFramePath": str(args.modeling_frame_path),
        "modelingFrameSha256": sha256_path(args.modeling_frame_path),
        "corpusPath": str(args.corpus_path),
        "corpusSha256": sha256_path(args.corpus_path),
        "goalCatalogPath": str(args.goal_catalog_path),
        "goalCatalogSha256": sha256_path(args.goal_catalog_path),
        "s05ErrorLimitsPath": str(args.error_limits_path),
        "s05ErrorLimitsSha256": sha256_path(args.error_limits_path),
        "s06PolicyEmbeddingsPath": str(args.policy_embeddings_path),
        "s06PolicyEmbeddingsSha256": sha256_path(args.policy_embeddings_path),
        "s06PolicyNeighborsPath": str(args.policy_neighbors_path),
        "s06PolicyNeighborsSha256": sha256_path(args.policy_neighbors_path),
        "targetColumns": target_columns,
        "targetWeights": target_weights.to_dict(orient="records"),
        "embeddedGoalCount": int(len(embeddings)),
        "observedGoalCount": int((coverage["rowCount"] > 0).sum()),
        "unobservedCatalogGoalCount": int(coverage["embeddingStatus"].eq("unobserved_in_s04").sum()),
        "highUncertaintyGoalCount": int(uncertainty["sparseUncertaintyLevel"].eq("high").sum()),
        "featureCount": int(len(columns)),
        "explainedVarianceRatio": embedding_model["explainedVarianceRatio"],
        "performanceSummary": perf,
        "claimBoundary": GOAL_EMBEDDING_CLAIM_BOUNDARY,
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
        f"{'passed' if success else 'failed'}: {len(embeddings)} goals embedded, "
        f"{len(retrieval_metrics)} retrieval metric rows, {len(stability)} stability metric rows, "
        f"{len(hard_failures)} hard validation failures; focused tests "
        f"{'skipped' if test_command is None else 'passed' if tests_ok else 'failed'}"
    )
    supportive = bool(success and perf["k3BaselineComparisonCount"] > 0 and perf["k3BaselineComparisonsWon"] >= 2 and perf["splitHalfDistanceSpearman"] > 0)
    outcome = "supportive" if supportive else ("null" if success else "constraining/contradictory")
    caveats = (
        "S07 embeds only S03 goals observed in S04 with enough S05 target coverage. Five catalog goals are "
        "unobserved in S04 and remain documented without learned coordinates, and several low-row or low-policy "
        "goals carry high sparse-uncertainty labels. Goal distances are weighted summary-proxy relationships and "
        "can reflect missingness, source experiment structure, and policy availability rather than direct causal or biological equivalence."
    )
    recommended = "Stop for Chief Scientist review before S08; if accepted, proceed to S08 Platonic distance definitions using S06 policy embeddings, S07 goal embeddings, S05 error limits, and documented sparse-coverage caveats."
    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "title": STEP_TITLE,
        "artifactsWritten": [],
        "validationResult": validation_result,
        "outcomeClassification": outcome,
        "caveatsOrBlockers": caveats if success else caveats + " Review failed validation/test checks before S08.",
        "recommendedNextAction": recommended,
        "performanceSummary": perf,
        "focusedTestCommand": test_command,
        "claimBoundary": GOAL_EMBEDDING_CLAIM_BOUNDARY,
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
            "embeddedGoalCount": int(len(embeddings)),
            "performanceSummary": perf,
            "validationResult": validation_result,
            "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
            "recommendedNextAction": recommended,
        },
    )

    coverage_report_path = step_dir / "coverage_uncertainty_report.md"
    validation_report_path = step_dir / "validation_report.md"
    model_report_path = step_dir / "model_report.md"
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    artifact_paths = [
        *profile_paths,
        *coverage_paths,
        *uncertainty_paths,
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
    coverage_markdown(coverage_report_path, coverage, uncertainty, target_weights, status_payload)
    validation_markdown(validation_report_path, validation, retrieval_metrics, stability, comparisons, status_payload)
    model_report_markdown(model_report_path, status_payload, perf, embeddings, retrieval_metrics)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": GOAL_EMBEDDING_SCHEMA_VERSION,
        "modelVersion": GOAL_EMBEDDING_MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "status": status_payload["status"],
        "artifactsWritten": collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path]),
        "validationResult": status_payload["validationResult"],
        "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "claimBoundary": GOAL_EMBEDDING_CLAIM_BOUNDARY,
    }
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    coverage_markdown(coverage_report_path, coverage, uncertainty, target_weights, status_payload)
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
            "corpusPath": str(args.corpus_path),
            "goalCatalogPath": str(args.goal_catalog_path),
            "errorLimitsPath": str(args.error_limits_path),
            "s05ModelCardPath": str(args.s05_model_card_path),
            "policyEmbeddingsPath": str(args.policy_embeddings_path),
            "policyNeighborsPath": str(args.policy_neighbors_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threading": "serial sklearn PCA; no GPU used",
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
    coverage_markdown(coverage_report_path, coverage, uncertainty, target_weights, status_payload)
    validation_markdown(validation_report_path, validation, retrieval_metrics, stability, comparisons, status_payload)
    model_report_markdown(model_report_path, status_payload, perf, embeddings, retrieval_metrics)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
