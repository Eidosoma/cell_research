#!/usr/bin/env python3
"""Compute and validate E07 S07 goal embeddings."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import sklearn  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.goal_embedding_schema import (  # noqa: E402
    GOAL_EMBEDDING_SCHEMA_VERSION,
    conflict_separation_summary,
    conflict_separation_table,
    policy_support_weight,
    source_balance_weights,
    stable_fraction,
    validate_goal_embedding_artifacts,
    weighted_frequency_profile,
    weighted_numeric_means,
)
from src.e07.policy_embedding_schema import (  # noqa: E402
    cosine_neighbor_table,
    neighbor_overlap_at_k,
    pad_embedding,
    sampled_pairwise_distance_spearman,
    standardize_profile_matrix,
    weighted_mean_profile,
)


STEP_ID = "S07"
STEP_NUMBER = 7
EXPERIMENT_ID = "E07"
RANDOM_SEED = 20260703
EMBEDDING_DIMS = 8
NEIGHBOR_K = 5
STABILITY_SEEDS = (17, 29, 43)
METRIC_SUBSET_FRACTION = 0.70
MIN_POLICY_SUPPORT_ROWS = 10

GOAL_METADATA_COLUMNS = (
    "canonical_goal_id",
    "goal_uid",
    "source_experiment_id",
    "source_step_id",
    "source_goal_id",
    "display_name",
    "goal_family",
    "goal_kind",
    "abstraction_kind",
    "target_structure",
    "representation_status",
    "partial",
    "target_direction",
    "unit",
    "primary_metric_id",
    "metric_family",
    "target_value",
    "conflict_group_id",
    "conflicts_with_goal_ids_json",
    "compatible_with_goal_ids_json",
    "limitations_json",
    "representation_hash",
)

TARGET_MEAN_PROFILE_COLUMNS = (
    "metric_family",
    "world_family",
    "perturbation_type",
    "metric_direction",
    "metric_unit",
    "goal_metric_family",
    "goal_target_direction",
    "substrate_kind",
    "world_record_granularity",
    "policy_family",
    "policy_algorithm",
    "policy_representation_type",
)

FREQUENCY_PROFILE_COLUMNS = (
    "outcome_band",
    "metric_family",
    "world_family",
    "perturbation_type",
    "policy_family",
    "policy_source_experiment_id",
    "behavior_support_status",
)

POLICY_EMBEDDING_COLUMNS = tuple(f"behavior_embedding_dim_{index:02d}" for index in range(EMBEDDING_DIMS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--s03-goal-table", type=Path, default=artifacts_dir / "tables" / "e07_goal_representations.parquet")
    parser.add_argument("--s03-goal-conflicts", type=Path, default=artifacts_dir / "tables" / "e07_goal_conflicts.parquet")
    parser.add_argument("--s04-corpus", type=Path, default=artifacts_dir / "results" / "e07_unified_behavior_corpus.parquet")
    parser.add_argument("--s05-modeling-dataset", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_modeling_dataset.parquet")
    parser.add_argument("--s05-metrics", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_metrics.parquet")
    parser.add_argument("--s05-calibration", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_calibration.parquet")
    parser.add_argument("--s06-policy-embeddings", type=Path, default=artifacts_dir / "results" / "e07_policy_embeddings.parquet")
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = __import__("hashlib").sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_path(path),
        "sizeBytes": path.stat().st_size if path.is_file() else sum(child.stat().st_size for child in path.rglob("*") if child.is_file()),
        "artifactType": "directory" if path.is_dir() else "file",
    }


def self_referential_artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "artifactType": "file",
        "note": "Checksum omitted because this report or manifest contains the artifact list.",
    }


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(elapsed),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    frame = df.head(max_rows).copy()
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in frame.to_dict(orient="records"):
        values = [clean_text(record.get(column)).replace("|", "\\|").replace("\n", " ") for column in frame.columns]
        rows.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        rows.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(rows)


def load_goal_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = [column for column in GOAL_METADATA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"S03 goal table missing required S07 columns: {missing}")
    goals = frame[list(GOAL_METADATA_COLUMNS)].drop_duplicates("canonical_goal_id").reset_index(drop=True)
    for column in goals.columns:
        if goals[column].dtype == object:
            goals[column] = goals[column].fillna("").astype(str)
    return goals


def load_policy_support(path: Path) -> pd.DataFrame:
    columns = [
        "canonical_policy_id",
        "source_experiment_id",
        "policy_family",
        "algorithm",
        "representation_type",
        "behavior_support_status",
        "behavior_observation_count",
        "behavior_source_count",
        *POLICY_EMBEDDING_COLUMNS,
    ]
    frame = pd.read_parquet(path, columns=columns)
    return frame.rename(
        columns={
            "source_experiment_id": "s06_policy_source_experiment_id",
            "policy_family": "s06_policy_family",
            "algorithm": "s06_policy_algorithm",
            "representation_type": "s06_policy_representation_type",
        }
    )


def outcome_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.dropna().empty:
        return pd.Series("__missing__", index=values.index)
    q25 = float(numeric.quantile(0.25))
    q75 = float(numeric.quantile(0.75))

    def label(value: float) -> str:
        if not np.isfinite(value):
            return "__missing__"
        if value < 0:
            return "negative_or_failure_proxy"
        if value <= q25:
            return "lower_quartile_proxy"
        if value >= q75:
            return "upper_quartile_proxy"
        return "middle_range_proxy"

    return numeric.map(label)


def prepare_goal_frame(modeling: pd.DataFrame, policies: pd.DataFrame) -> pd.DataFrame:
    frame = modeling.copy()
    frame["canonical_goal_id"] = frame["canonical_goal_id"].fillna("__missing__").astype(str)
    frame = frame[frame["canonical_goal_id"] != "__missing__"].copy()
    frame["canonical_policy_id"] = frame["canonical_policy_id"].fillna("__missing__").astype(str)
    frame["target_transformed"] = pd.to_numeric(frame["target_transformed"], errors="coerce")
    frame = frame[np.isfinite(frame["target_transformed"].to_numpy(dtype=float))].reset_index(drop=True)
    frame = frame.merge(policies, on="canonical_policy_id", how="left")
    frame["behavior_support_status"] = frame["behavior_support_status"].fillna("policy_missing_or_unmatched")
    frame["behavior_observation_count"] = pd.to_numeric(frame["behavior_observation_count"], errors="coerce").fillna(0.0)
    for column in POLICY_EMBEDDING_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["s06_policy_source_experiment_id"] = frame["s06_policy_source_experiment_id"].fillna("__missing__").astype(str)
    frame["s06_policy_family"] = frame["s06_policy_family"].fillna("__missing__").astype(str)
    frame["s06_policy_algorithm"] = frame["s06_policy_algorithm"].fillna("__missing__").astype(str)
    frame["s06_policy_representation_type"] = frame["s06_policy_representation_type"].fillna("__missing__").astype(str)
    frame["policy_support_weight"] = [
        policy_support_weight(status, count, min_supported_rows=MIN_POLICY_SUPPORT_ROWS)
        for status, count in zip(frame["behavior_support_status"], frame["behavior_observation_count"], strict=False)
    ]
    frame["source_balance_weight"] = source_balance_weights(frame)
    frame["goal_uncertainty_weight"] = frame["source_balance_weight"] * frame["policy_support_weight"]
    frame["supported_only_weight"] = frame["source_balance_weight"] * (frame["behavior_support_status"] == "behavior_profile_supported").astype(float)
    frame["outcome_band"] = outcome_band(frame["target_transformed"])

    rename_map = {
        "s06_policy_family": "policy_family",
        "s06_policy_algorithm": "policy_algorithm",
        "s06_policy_representation_type": "policy_representation_type",
        "s06_policy_source_experiment_id": "policy_source_experiment_id",
    }
    for source, target in rename_map.items():
        if target not in frame.columns:
            frame[target] = frame[source]
    for column in set(TARGET_MEAN_PROFILE_COLUMNS) | set(FREQUENCY_PROFILE_COLUMNS):
        if column not in frame.columns:
            frame[column] = "__missing__"
        frame[column] = frame[column].fillna("__missing__").astype(str).map(lambda value: value if value else "__missing__")
    return frame


def build_goal_profile_matrix(
    frame: pd.DataFrame,
    goal_ids: Sequence[str],
    *,
    weight_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_blocks: list[pd.DataFrame] = []
    for column in TARGET_MEAN_PROFILE_COLUMNS:
        sub = frame[["canonical_goal_id", "target_transformed", weight_column, column]].copy()
        sub["profile_feature"] = f"mean_target:{column}=" + sub[column].fillna("__missing__").astype(str)
        block = weighted_mean_profile(
            sub,
            entity_column="canonical_goal_id",
            feature_column="profile_feature",
            value_column="target_transformed",
            weight_column=weight_column,
        )
        feature_blocks.append(block)
    for column in FREQUENCY_PROFILE_COLUMNS:
        sub = frame[["canonical_goal_id", weight_column, column]].copy()
        sub["profile_feature"] = f"freq:{column}=" + sub[column].fillna("__missing__").astype(str)
        block = weighted_frequency_profile(
            sub,
            entity_column="canonical_goal_id",
            feature_column="profile_feature",
            weight_column=weight_column,
        )
        feature_blocks.append(block)
    policy_means = weighted_numeric_means(
        frame,
        entity_column="canonical_goal_id",
        numeric_columns=POLICY_EMBEDDING_COLUMNS,
        weight_column=weight_column,
    ).rename(columns={column: f"mean_policy_{column}" for column in POLICY_EMBEDDING_COLUMNS})
    feature_blocks.append(policy_means)
    blocks = [block for block in feature_blocks if not block.empty]
    if blocks:
        profile = pd.concat(blocks, axis=1).fillna(0.0)
    else:
        profile = pd.DataFrame(index=goal_ids)
    profile = profile.reindex(goal_ids, fill_value=0.0)
    stats = {
        "weightColumn": weight_column,
        "profileFeatureColumns": int(profile.shape[1]),
        "nonzeroWeightedRows": int((pd.to_numeric(frame[weight_column], errors="coerce").fillna(0) > 0).sum()),
        "goalsWithNonzeroWeight": int(frame.loc[pd.to_numeric(frame[weight_column], errors="coerce").fillna(0) > 0, "canonical_goal_id"].nunique()),
    }
    return profile, stats


def fit_goal_embedding(profile: pd.DataFrame, *, dims: int = EMBEDDING_DIMS) -> tuple[np.ndarray, dict[str, Any]]:
    standardized, standardization_stats = standardize_profile_matrix(profile)
    if standardized.empty or standardized.shape[1] == 0:
        return np.zeros((len(profile), dims), dtype=float), {
            **standardization_stats,
            "pcaComponents": 0,
            "explainedVarianceRatio": [],
        }
    n_components = min(dims, standardized.shape[0], standardized.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    values = pca.fit_transform(standardized.to_numpy(dtype=float))
    return pad_embedding(values, dims), {
        **standardization_stats,
        "pcaComponents": int(n_components),
        "explainedVarianceRatio": [float(value) for value in pca.explained_variance_ratio_],
    }


def goal_coverage(goals: pd.DataFrame, frame: pd.DataFrame, s04: pd.DataFrame) -> pd.DataFrame:
    base = goals[["canonical_goal_id"]].copy()
    behavior = (
        frame.groupby("canonical_goal_id", observed=True)
        .agg(
            goal_behavior_row_count=("canonical_goal_id", "size"),
            goal_policy_count=("canonical_policy_id", "nunique"),
            supported_policy_row_count=("behavior_support_status", lambda values: int((values == "behavior_profile_supported").sum())),
            supported_policy_count=("canonical_policy_id", lambda values: int(frame.loc[values.index].query("behavior_support_status == 'behavior_profile_supported'")["canonical_policy_id"].nunique())),
            sparse_policy_row_count=("behavior_support_status", lambda values: int((values == "sparse_behavior_profile").sum())),
            missing_policy_row_count=("behavior_support_status", lambda values: int((values == "policy_missing_or_unmatched").sum())),
            weighted_evidence_row_sum=("goal_uncertainty_weight", "sum"),
            supported_only_row_sum=("supported_only_weight", "sum"),
            source_count=("source_experiment_id", "nunique"),
            metric_family_count=("metric_family", "nunique"),
            world_family_count=("world_family", "nunique"),
            perturbation_count=("perturbation_type", "nunique"),
            target_mean=("target_transformed", "mean"),
            target_std=("target_transformed", "std"),
        )
        .reset_index()
    )
    s04 = s04.copy()
    s04["canonical_goal_id"] = s04["canonical_goal_id"].fillna("__missing__").astype(str)
    s04 = s04[(s04["evidence_kind"].astype(str) == "metric_observation") & (s04["canonical_goal_id"] != "__missing__")]
    s04_summary = (
        s04.groupby("canonical_goal_id", observed=True)
        .agg(
            s04_metric_observation_count=("canonical_goal_id", "size"),
            s04_policy_count=("canonical_policy_id", "nunique"),
            s04_world_count=("world_id", "nunique"),
            s04_metric_family_count=("metric_family", "nunique"),
        )
        .reset_index()
    )
    coverage = base.merge(behavior, on="canonical_goal_id", how="left").merge(s04_summary, on="canonical_goal_id", how="left")
    count_cols = [
        "goal_behavior_row_count",
        "goal_policy_count",
        "supported_policy_row_count",
        "supported_policy_count",
        "sparse_policy_row_count",
        "missing_policy_row_count",
        "source_count",
        "metric_family_count",
        "world_family_count",
        "perturbation_count",
        "s04_metric_observation_count",
        "s04_policy_count",
        "s04_world_count",
        "s04_metric_family_count",
    ]
    for column in count_cols:
        coverage[column] = pd.to_numeric(coverage[column], errors="coerce").fillna(0).astype(int)
    for column in ("weighted_evidence_row_sum", "supported_only_row_sum", "target_mean", "target_std"):
        coverage[column] = pd.to_numeric(coverage[column], errors="coerce").fillna(0.0)
    coverage["goal_support_status"] = np.select(
        [
            coverage["supported_policy_row_count"] > 0,
            coverage["goal_behavior_row_count"] > 0,
        ],
        ["supported_policy_evidence", "modeled_without_supported_policy_evidence"],
        default="no_s05_goal_behavior_rows",
    )
    return coverage


def add_goal_labels(neighbors: pd.DataFrame, goals: pd.DataFrame) -> pd.DataFrame:
    if neighbors.empty:
        return neighbors.copy()
    labels = goals[["canonical_goal_id", "display_name", "goal_family", "source_goal_id", "target_direction"]].copy()
    query = labels.add_prefix("query_").rename(columns={"query_canonical_goal_id": "query_goal_id"})
    neighbor = labels.add_prefix("neighbor_").rename(columns={"neighbor_canonical_goal_id": "neighbor_goal_id"})
    return neighbors.merge(query, on="query_goal_id", how="left").merge(neighbor, on="neighbor_goal_id", how="left")


def goal_neighbor_table(embedding: np.ndarray, goal_ids: Sequence[str], goals: pd.DataFrame, *, embedding_name: str) -> pd.DataFrame:
    table = cosine_neighbor_table(embedding, goal_ids, k=NEIGHBOR_K, embedding_name=embedding_name).rename(
        columns={"query_policy_id": "query_goal_id", "neighbor_policy_id": "neighbor_goal_id"}
    )
    return add_goal_labels(table, goals)


def sensitivity_metrics(
    primary_embedding: np.ndarray,
    supported_embedding: np.ndarray,
    goal_ids: Sequence[str],
    eligible_goal_ids: Sequence[str],
) -> pd.DataFrame:
    id_to_index = {goal_id: index for index, goal_id in enumerate(goal_ids)}
    indices = [id_to_index[goal_id] for goal_id in eligible_goal_ids]
    if len(indices) < 3:
        return pd.DataFrame(
            [
                {"analysis_name": "supported_only_filter", "metric_name": "eligible_goal_count", "metric_value": float(len(indices)), "detail": "too few goals for distance sensitivity"}
            ]
        )
    primary = primary_embedding[indices]
    supported = supported_embedding[indices]
    primary_neighbors = cosine_neighbor_table(primary, eligible_goal_ids, k=3, embedding_name="primary")
    supported_neighbors = cosine_neighbor_table(supported, eligible_goal_ids, k=3, embedding_name="supported_only")
    shifts = np.linalg.norm(primary - supported, axis=1)
    rows = [
        {
            "analysis_name": "supported_only_filter",
            "metric_name": "eligible_goal_count",
            "metric_value": float(len(indices)),
            "detail": "goals with at least one behavior-supported policy row",
        },
        {
            "analysis_name": "supported_only_filter",
            "metric_name": "pairwise_distance_spearman",
            "metric_value": sampled_pairwise_distance_spearman(primary, supported, max_pairs=10_000, seed=RANDOM_SEED + 101),
            "detail": "primary uncertainty-weighted map vs supported-policy-only map",
        },
        {
            "analysis_name": "supported_only_filter",
            "metric_name": "neighbor_overlap_at_3",
            "metric_value": neighbor_overlap_at_k(primary_neighbors, supported_neighbors, k=3),
            "detail": "top-3 neighbor overlap after filtering to S06-supported policies",
        },
        {
            "analysis_name": "supported_only_filter",
            "metric_name": "median_coordinate_shift",
            "metric_value": float(np.median(shifts)),
            "detail": "median Euclidean coordinate shift under supported-only filtering",
        },
        {
            "analysis_name": "supported_only_filter",
            "metric_name": "max_coordinate_shift",
            "metric_value": float(np.max(shifts)),
            "detail": "maximum Euclidean coordinate shift under supported-only filtering",
        },
    ]
    return pd.DataFrame(rows)


def metric_subset_stability(
    frame: pd.DataFrame,
    goal_ids: Sequence[str],
    eligible_goal_ids: Sequence[str],
    main_embedding: np.ndarray,
    main_neighbors: pd.DataFrame,
) -> pd.DataFrame:
    id_to_index = {goal_id: index for index, goal_id in enumerate(goal_ids)}
    eligible_indices = [id_to_index[goal_id] for goal_id in eligible_goal_ids]
    main = main_embedding[eligible_indices]
    rows = []
    for seed in STABILITY_SEEDS:
        metric_mask = frame["source_metric_name"].astype(str).map(lambda value: stable_fraction(value, salt=f"s07-metric-subset-{seed}") < METRIC_SUBSET_FRACTION)
        subset = frame[metric_mask].reset_index(drop=True)
        profile, profile_stats = build_goal_profile_matrix(subset, goal_ids, weight_column="goal_uncertainty_weight")
        embedding, pca_stats = fit_goal_embedding(profile, dims=EMBEDDING_DIMS)
        candidate = embedding[eligible_indices]
        candidate_neighbors = cosine_neighbor_table(candidate, eligible_goal_ids, k=3, embedding_name=f"metric_subset_{seed}")
        rows.append(
            {
                "seed": int(seed),
                "metric_subset_fraction": float(METRIC_SUBSET_FRACTION),
                "retained_rows": int(len(subset)),
                "retained_source_metric_count": int(subset["source_metric_name"].nunique()),
                "profile_feature_count": int(profile_stats["profileFeatureColumns"]),
                "pca_components": int(pca_stats["pcaComponents"]),
                "pairwise_distance_spearman": sampled_pairwise_distance_spearman(main, candidate, max_pairs=10_000, seed=RANDOM_SEED + seed),
                "neighbor_overlap_at_3": neighbor_overlap_at_k(main_neighbors, candidate_neighbors, k=3),
            }
        )
    return pd.DataFrame(rows)


def plot_goal_map(embeddings: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    source_order = sorted(embeddings["source_experiment_id"].fillna("__missing__").astype(str).unique())
    cmap = plt.get_cmap("tab10")
    colors = {source: cmap(index % 10) for index, source in enumerate(source_order)}
    markers = {
        "supported_policy_evidence": "o",
        "modeled_without_supported_policy_evidence": "^",
        "no_s05_goal_behavior_rows": "x",
    }
    fig, ax = plt.subplots(figsize=(11, 8))
    for source in source_order:
        for status, marker in markers.items():
            subset = embeddings[
                (embeddings["source_experiment_id"].fillna("__missing__").astype(str) == source)
                & (embeddings["goal_support_status"] == status)
            ]
            if subset.empty:
                continue
            ax.scatter(
                subset["embedding_x"],
                subset["embedding_y"],
                s=70 if marker != "x" else 55,
                alpha=0.72 if status == "supported_policy_evidence" else 0.38,
                color=colors[source],
                marker=marker,
                label=f"{source} {status}",
                linewidths=0.9,
            )
    for row in embeddings.to_dict(orient="records"):
        if row["goal_support_status"] == "supported_policy_evidence":
            ax.annotate(str(row["goal_family"])[:18], (row["embedding_x"], row["embedding_y"]), fontsize=6, alpha=0.75)
    ax.set_title("E07 S07 goal embedding map")
    ax.set_xlabel("Goal embedding PC1")
    ax.set_ylabel("Goal embedding PC2")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)


def outcome_classification(validation: pd.DataFrame) -> str:
    if not bool(validation["success"].all()):
        return "constraining/contradictory"
    return "supportive"


def build_report(
    *,
    full_report_path: Path,
    artifacts_written: Sequence[str],
    goal_embeddings: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    stability: pd.DataFrame,
    sensitivity: pd.DataFrame,
    conflict_summary: pd.DataFrame,
    conflict_distances: pd.DataFrame,
    validation: pd.DataFrame,
    outcome: str,
    primary_profile_stats: Mapping[str, Any],
    primary_pca_stats: Mapping[str, Any],
    supported_profile_stats: Mapping[str, Any],
    supported_pca_stats: Mapping[str, Any],
    unit_test_result: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    validation_success = bool(validation["success"].all())
    support_counts = goal_embeddings["goal_support_status"].value_counts().to_dict()
    caveats = (
        "Goal embeddings carry forward S06 policy-support limits; the primary map is source-balanced and uncertainty-weighted by S06 behavior_support_status and coverage counts; "
        "goals lacking supported policy evidence are retained but low-confidence; conflict separation failure is treated as constraining."
    )
    test_line = "not run"
    if unit_test_result is not None:
        test_line = f"{'pass' if unit_test_result['success'] else 'fail'}: `{unit_test_result['command']}` return code {unit_test_result['returnCode']}"
    text = f"""# E07 S07 Full Results: Goal Embeddings

## Top Summary

- Research step ID: S07
- Completion status: complete
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: {"pass" if validation_success else "fail"} ({int(validation['success'].sum())}/{len(validation)} checks passed)
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Lay summary: S07 mapped the 38 abstract goals into a behavior-space using S05 goal-linked rows and S06 policy embeddings, while explicitly down-weighting sparse policy evidence and checking how the map changes when only S06-supported policies are used.
- Recommended next action: Chief review of S07 coverage, filtering sensitivity, and conflict-separation results; proceed to S08 only after accepting the low-confidence status of sparse goal regions.

## Frozen Question

Can goals be embedded by the policies they recruit, the failures they induce, and the trajectories they produce?

## Inputs

- S03 goal table: `{args.s03_goal_table}` (SHA-256 `{sha256_file(args.s03_goal_table)}`)
- S03 goal conflicts: `{args.s03_goal_conflicts}` (SHA-256 `{sha256_file(args.s03_goal_conflicts)}`)
- S04 unified behavior corpus: `{args.s04_corpus}` (SHA-256 `{sha256_file(args.s04_corpus)}`)
- S05 modeling dataset: `{args.s05_modeling_dataset}` (SHA-256 `{sha256_file(args.s05_modeling_dataset)}`)
- S05 metrics: `{args.s05_metrics}` (SHA-256 `{sha256_file(args.s05_metrics)}`)
- S05 calibration: `{args.s05_calibration}` (SHA-256 `{sha256_file(args.s05_calibration)}`)
- S06 policy embeddings: `{args.s06_policy_embeddings}` (SHA-256 `{sha256_file(args.s06_policy_embeddings)}`)

## Methods

The script represented every S03 `canonical_goal_id`. It joined S05 modeling rows to S06 policy embeddings by `canonical_policy_id`, then computed a policy support weight from S06 status and behavior coverage: `behavior_profile_supported = 1.0`, sparse policies receive `sqrt(behavior_observation_count / {MIN_POLICY_SUPPORT_ROWS})` clipped to `[0.10, 0.95]`, and no-behavior or unmatched policies receive `0.0`. Row weights are the product of that support weight and an inverse source-count balance weight.

Primary goal profiles combine three evidence families: weighted target means by metric/world/perturbation/policy axes, weighted frequency shares for outcome-band and recruitment distributions, and weighted mean S06 policy behavior coordinates for policies associated with each goal. The matrix is standardized and reduced by PCA to {EMBEDDING_DIMS} dimensions. `embedding_x` and `embedding_y` are the first two dimensions.

S06 filtering sensitivity was recorded by recomputing a supported-only map using only rows linked to behavior-supported S06 policies. Metric-subset stability was measured across three deterministic 70% source-metric subsets. Known conflict separation was tested using S03 conflict pairs against non-conflict reference pairs; a failure is a constraining result, not a parameter to tune around.

## Commands

- `python -m unittest tests.e07.test_goal_embedding_schema`
- `python scripts/e07_s07_goal_embeddings.py`

Unit-test result: {test_line}

## Dependencies and Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- numpy: {np.__version__}
- scikit-learn: {sklearn.__version__}
- matplotlib: {matplotlib.__version__}
- Worker count: single-process CPU analysis; no GPU model training in S07.
- Repository commit before S07 commit: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Branch: `{git_output(args.repo_dir, ['branch', '--show-current'])}`

## Results

Goal rows represented: {len(goal_embeddings)}

Goal support statuses:

{markdown_table(pd.DataFrame([{"goal_support_status": key, "goal_count": value} for key, value in support_counts.items()]))}

Primary profile statistics:

- Nonzero weighted rows: {primary_profile_stats['nonzeroWeightedRows']}
- Goals with nonzero primary weight: {primary_profile_stats['goalsWithNonzeroWeight']}
- Profile feature columns: {primary_profile_stats['profileFeatureColumns']}
- PCA retained features: {primary_pca_stats['retainedFeatureCount']}
- PCA components: {primary_pca_stats['pcaComponents']}
- PCA explained variance ratio: `{primary_pca_stats['explainedVarianceRatio']}`

Supported-only profile statistics:

- Nonzero weighted rows: {supported_profile_stats['nonzeroWeightedRows']}
- Goals with nonzero supported-only weight: {supported_profile_stats['goalsWithNonzeroWeight']}
- Profile feature columns: {supported_profile_stats['profileFeatureColumns']}
- PCA retained features: {supported_pca_stats['retainedFeatureCount']}
- PCA components: {supported_pca_stats['pcaComponents']}

### Coverage Summary

{markdown_table(coverage_summary, max_rows=50)}

### S06 Filtering Sensitivity

{markdown_table(sensitivity)}

### Metric-Subset Stability

{markdown_table(stability)}

### Known Conflict Separation

{markdown_table(conflict_summary)}

Conflict distance sample:

{markdown_table(conflict_distances.sort_values(['pair_type', 'distance']).head(20))}

## Validation

{markdown_table(validation)}

## Output Artifacts

{chr(10).join(f'- `{item}`' for item in artifacts_written)}

## Caveats, Blockers, and Limitations

- The result is a computational proxy embedding over prior simulation metrics, not biological or causal validation.
- S07 inherits S05 oriented signed-log target limits and S06 policy-support sparsity.
- Goals with `no_s05_goal_behavior_rows` are represented with finite coordinates for table completeness but are not behavior-supported.
- Conflict separation is evaluated on available S03 conflict pairs and the current proxy feature matrix; failure indicates a limitation of available data or representation, not a reason to tune thresholds.
- Source balancing reduces but cannot remove source-specific metric availability effects.
- The S07 map is a bounded input to S08, not a final distance ontology.

## Recommended Next Action

Chief review should decide whether S08 should use only supported-policy-evidence goals, uncertainty weights, or both. Stop here and do not start S08 until instructed.
"""
    write_text(full_report_path, text)


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e07"
    for directory in (step_dir, results_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_test_result = None
    if args.run_unit_tests:
        unit_test_result = run_command([sys.executable, "-m", "unittest", "tests.e07.test_goal_embedding_schema"], args.repo_dir)

    goals = load_goal_metadata(args.s03_goal_table)
    goal_ids = goals["canonical_goal_id"].astype(str).tolist()
    conflicts = pd.read_parquet(args.s03_goal_conflicts)
    policies = load_policy_support(args.s06_policy_embeddings)
    modeling = pd.read_parquet(args.s05_modeling_dataset)
    s04 = pd.read_parquet(
        args.s04_corpus,
        columns=[
            "canonical_goal_id",
            "canonical_policy_id",
            "metric_family",
            "world_id",
            "evidence_kind",
        ],
    )
    frame = prepare_goal_frame(modeling, policies)
    coverage = goal_coverage(goals, frame, s04)

    primary_profile, primary_profile_stats = build_goal_profile_matrix(frame, goal_ids, weight_column="goal_uncertainty_weight")
    primary_embedding, primary_pca_stats = fit_goal_embedding(primary_profile, dims=EMBEDDING_DIMS)
    supported_profile, supported_profile_stats = build_goal_profile_matrix(frame, goal_ids, weight_column="supported_only_weight")
    supported_embedding, supported_pca_stats = fit_goal_embedding(supported_profile, dims=EMBEDDING_DIMS)

    goal_embeddings = goals.merge(coverage, on="canonical_goal_id", how="left")
    for index in range(EMBEDDING_DIMS):
        goal_embeddings[f"goal_embedding_dim_{index:02d}"] = primary_embedding[:, index]
        goal_embeddings[f"supported_only_goal_embedding_dim_{index:02d}"] = supported_embedding[:, index]
    goal_embeddings["embedding_x"] = goal_embeddings["goal_embedding_dim_00"]
    goal_embeddings["embedding_y"] = goal_embeddings["goal_embedding_dim_01"]
    goal_embeddings["supported_only_embedding_x"] = goal_embeddings["supported_only_goal_embedding_dim_00"]
    goal_embeddings["supported_only_embedding_y"] = goal_embeddings["supported_only_goal_embedding_dim_01"]
    goal_embeddings["embedding_method"] = "source_balanced_policy_support_weighted_goal_profile_pca"
    goal_embeddings["s06_filtering_method"] = "uncertainty_weighted_primary_plus_supported_only_sensitivity"
    goal_embeddings["research_step_id"] = STEP_ID
    goal_embeddings["schema_version"] = GOAL_EMBEDDING_SCHEMA_VERSION

    eligible_goal_ids = goal_embeddings.loc[goal_embeddings["supported_policy_row_count"] > 0, "canonical_goal_id"].astype(str).tolist()
    main_neighbors_raw = cosine_neighbor_table(
        primary_embedding[[goal_ids.index(goal_id) for goal_id in eligible_goal_ids]],
        eligible_goal_ids,
        k=3,
        embedding_name="primary_uncertainty_weighted",
    )
    neighbor_table = goal_neighbor_table(primary_embedding, goal_ids, goals, embedding_name="primary_uncertainty_weighted")
    supported_neighbor_table = goal_neighbor_table(supported_embedding, goal_ids, goals, embedding_name="supported_only")
    neighbors = pd.concat([neighbor_table, supported_neighbor_table], ignore_index=True)
    sensitivity = sensitivity_metrics(primary_embedding, supported_embedding, goal_ids, eligible_goal_ids)
    stability = metric_subset_stability(frame, goal_ids, eligible_goal_ids, primary_embedding, main_neighbors_raw)
    conflict_distances = conflict_separation_table(
        goal_embeddings,
        conflicts,
        coordinate_columns=[f"goal_embedding_dim_{index:02d}" for index in range(EMBEDDING_DIMS)],
        eligible_goal_ids=eligible_goal_ids,
    )
    conflict_summary = conflict_separation_summary(conflict_distances)
    validation = validate_goal_embedding_artifacts(
        goal_embeddings,
        stability,
        sensitivity,
        conflict_summary,
        expected_goal_ids=goal_ids,
    )
    outcome = outcome_classification(validation)

    coverage_summary = (
        goal_embeddings.groupby(["source_experiment_id", "goal_support_status"], as_index=False)
        .agg(
            goal_count=("canonical_goal_id", "size"),
            s05_modeling_rows=("goal_behavior_row_count", "sum"),
            supported_policy_rows=("supported_policy_row_count", "sum"),
            supported_policy_count=("supported_policy_count", "sum"),
            s04_metric_rows=("s04_metric_observation_count", "sum"),
        )
        .sort_values(["source_experiment_id", "goal_support_status"], kind="mergesort")
    )

    goal_embeddings_path = results_dir / "e07_goal_embeddings.parquet"
    goal_embeddings_csv_path = tables_dir / "e07_goal_embeddings.csv"
    unified_embeddings_path = results_dir / "e07_embeddings.parquet"
    neighbors_path = results_dir / "e07_goal_embedding_neighbor_retrieval.parquet"
    neighbors_csv_path = tables_dir / "e07_goal_embedding_neighbor_retrieval_sample.csv"
    stability_path = results_dir / "e07_goal_embedding_stability.parquet"
    stability_csv_path = tables_dir / "e07_goal_embedding_stability.csv"
    sensitivity_path = tables_dir / "e07_goal_embedding_s06_filter_sensitivity.csv"
    conflict_path = tables_dir / "e07_goal_embedding_conflict_separation.csv"
    conflict_distances_path = results_dir / "e07_goal_embedding_conflict_distances.parquet"
    coverage_path = tables_dir / "e07_goal_embedding_coverage_summary.csv"
    validation_path = step_dir / "e07_s07_validation_checks.csv"
    config_path = step_dir / "s07_config.json"
    full_report_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    figure_path = figures_dir / "goal_embedding_map.png"

    goal_embeddings.to_parquet(goal_embeddings_path, index=False)
    goal_embeddings.to_csv(goal_embeddings_csv_path, index=False)
    policy_embeddings = pd.read_parquet(args.s06_policy_embeddings).copy()
    policy_embeddings.insert(0, "embedding_entity_type", "policy")
    goal_entity_rows = goal_embeddings.copy()
    goal_entity_rows.insert(0, "embedding_entity_type", "goal")
    pd.concat([policy_embeddings, goal_entity_rows], ignore_index=True, sort=False).to_parquet(unified_embeddings_path, index=False)
    neighbors.to_parquet(neighbors_path, index=False)
    neighbors.head(500).to_csv(neighbors_csv_path, index=False)
    stability.to_parquet(stability_path, index=False)
    stability.to_csv(stability_csv_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    conflict_summary.to_csv(conflict_path, index=False)
    conflict_distances.to_parquet(conflict_distances_path, index=False)
    coverage_summary.to_csv(coverage_path, index=False)
    validation.to_csv(validation_path, index=False)
    plot_goal_map(goal_embeddings, figure_path)
    write_json(
        config_path,
        {
            "schemaVersion": GOAL_EMBEDDING_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "randomSeed": RANDOM_SEED,
            "embeddingDims": EMBEDDING_DIMS,
            "neighborK": NEIGHBOR_K,
            "metricSubsetFraction": METRIC_SUBSET_FRACTION,
            "stabilitySeeds": list(STABILITY_SEEDS),
            "minPolicySupportRows": MIN_POLICY_SUPPORT_ROWS,
            "targetMeanProfileColumns": list(TARGET_MEAN_PROFILE_COLUMNS),
            "frequencyProfileColumns": list(FREQUENCY_PROFILE_COLUMNS),
            "policyEmbeddingColumns": list(POLICY_EMBEDDING_COLUMNS),
            "primaryProfileStats": primary_profile_stats,
            "primaryPcaStats": primary_pca_stats,
            "supportedOnlyProfileStats": supported_profile_stats,
            "supportedOnlyPcaStats": supported_pca_stats,
        },
    )

    artifacts_written = [
        str(full_report_path),
        str(goal_embeddings_path),
        str(goal_embeddings_csv_path),
        str(unified_embeddings_path),
        str(neighbors_path),
        str(neighbors_csv_path),
        str(stability_path),
        str(stability_csv_path),
        str(sensitivity_path),
        str(conflict_path),
        str(conflict_distances_path),
        str(coverage_path),
        str(figure_path),
        str(validation_path),
        str(config_path),
        str(artifact_manifest_path),
    ]
    build_report(
        full_report_path=full_report_path,
        artifacts_written=artifacts_written,
        goal_embeddings=goal_embeddings,
        coverage_summary=coverage_summary,
        stability=stability,
        sensitivity=sensitivity,
        conflict_summary=conflict_summary,
        conflict_distances=conflict_distances,
        validation=validation,
        outcome=outcome,
        primary_profile_stats=primary_profile_stats,
        primary_pca_stats=primary_pca_stats,
        supported_profile_stats=supported_profile_stats,
        supported_pca_stats=supported_pca_stats,
        unit_test_result=unit_test_result,
        args=args,
    )

    manifest_payload = {
        "schemaVersion": "eidosoma.e07.s07.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "success": bool(validation["success"].all()) and (unit_test_result is None or bool(unit_test_result["success"])),
        "status": "complete",
        "outcomeClassification": outcome,
        "validationResult": {
            "passed": int(validation["success"].sum()),
            "total": int(len(validation)),
            "allPassed": bool(validation["success"].all()),
        },
        "unitTestResult": unit_test_result,
        "inputs": {
            "s03GoalTable": {"path": str(args.s03_goal_table), "sha256": sha256_file(args.s03_goal_table)},
            "s03GoalConflicts": {"path": str(args.s03_goal_conflicts), "sha256": sha256_file(args.s03_goal_conflicts)},
            "s04Corpus": {"path": str(args.s04_corpus), "sha256": sha256_file(args.s04_corpus)},
            "s05ModelingDataset": {"path": str(args.s05_modeling_dataset), "sha256": sha256_file(args.s05_modeling_dataset)},
            "s05Metrics": {"path": str(args.s05_metrics), "sha256": sha256_file(args.s05_metrics)},
            "s05Calibration": {"path": str(args.s05_calibration), "sha256": sha256_file(args.s05_calibration)},
            "s06PolicyEmbeddings": {"path": str(args.s06_policy_embeddings), "sha256": sha256_file(args.s06_policy_embeddings)},
        },
        "coverage": {
            "goalRows": int(len(goal_embeddings)),
            "goalsWithS05Rows": int((goal_embeddings["goal_behavior_row_count"] > 0).sum()),
            "goalsWithSupportedPolicyRows": int((goal_embeddings["supported_policy_row_count"] > 0).sum()),
            "goalsWithoutS05Rows": int((goal_embeddings["goal_behavior_row_count"] == 0).sum()),
        },
        "artifacts": [
            self_referential_artifact_entry(full_report_path, artifacts_dir, "S07 full-results report."),
            artifact_entry(goal_embeddings_path, artifacts_dir, "Primary S07 goal embedding table."),
            artifact_entry(goal_embeddings_csv_path, artifacts_dir, "Goal embedding table in CSV format."),
            artifact_entry(unified_embeddings_path, artifacts_dir, "Combined E07 embedding table with policy and goal rows."),
            artifact_entry(neighbors_path, artifacts_dir, "Goal neighbor retrieval table."),
            artifact_entry(neighbors_csv_path, artifacts_dir, "Small CSV sample of goal neighbor retrieval rows."),
            artifact_entry(stability_path, artifacts_dir, "Metric-subset goal embedding stability metrics."),
            artifact_entry(stability_csv_path, artifacts_dir, "Metric-subset stability metrics in CSV format."),
            artifact_entry(sensitivity_path, artifacts_dir, "S06 policy-support filtering sensitivity table."),
            artifact_entry(conflict_path, artifacts_dir, "Known conflict separation summary."),
            artifact_entry(conflict_distances_path, artifacts_dir, "Pairwise distances for known conflict and reference goal pairs."),
            artifact_entry(coverage_path, artifacts_dir, "Goal embedding coverage summary."),
            artifact_entry(figure_path, artifacts_dir, "Two-dimensional goal embedding map."),
            artifact_entry(validation_path, artifacts_dir, "S07 validation checks."),
            artifact_entry(config_path, artifacts_dir, "S07 embedding configuration."),
            self_referential_artifact_entry(artifact_manifest_path, artifacts_dir, "S07 artifact manifest."),
        ],
    }
    write_json(artifact_manifest_path, manifest_payload)
    print(f"[S07] wrote {full_report_path}")
    print(f"[S07] validation {int(validation['success'].sum())}/{len(validation)} outcome={outcome}")


if __name__ == "__main__":
    main()
