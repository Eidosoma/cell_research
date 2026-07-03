#!/usr/bin/env python3
"""Compute and validate E07 S06 policy embeddings."""

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
from sklearn.decomposition import PCA, TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.policy_embedding_schema import (  # noqa: E402
    POLICY_EMBEDDING_SCHEMA_VERSION,
    add_neighbor_labels,
    cosine_neighbor_table,
    neighbor_overlap_at_k,
    pad_embedding,
    sampled_pairwise_distance_spearman,
    stable_fraction,
    standardize_profile_matrix,
    validate_policy_embedding_artifacts,
    weighted_mean_profile,
)


STEP_ID = "S06"
STEP_NUMBER = 6
EXPERIMENT_ID = "E07"
RANDOM_SEED = 20260703
EMBEDDING_DIMS = 8
NEIGHBOR_K = 5
STABILITY_SEEDS = (11, 23, 37)
BOOTSTRAP_FRACTION = 0.70

BEHAVIOR_PROFILE_COLUMNS = (
    "metric_family",
    "world_family",
    "goal_family",
    "perturbation_type",
    "metric_direction",
    "metric_unit",
    "goal_metric_family",
    "goal_target_direction",
    "substrate_kind",
    "world_record_granularity",
)

POLICY_METADATA_COLUMNS = (
    "canonical_policy_id",
    "policy_uid",
    "source_experiment_id",
    "source_step_id",
    "source_policy_id",
    "display_name",
    "source_category",
    "policy_family",
    "algorithm",
    "representation_type",
    "abstraction_kind",
    "interpretable",
    "metadata_only",
    "requires_memory",
    "requires_signaling",
    "uses_global_oracle",
    "information_access",
    "execution_backend",
    "direction_support",
    "observation_contract",
    "action_vocabulary_json",
    "state_variables_json",
    "parameter_keys_json",
    "representation_payload_json",
    "declared_dsl_sha256",
    "computed_dsl_canonical_sha256",
    "computed_dsl_source_sha256",
    "source_payload_sha256",
    "representation_hash",
    "hash_validation_status",
    "parser_validation_status",
    "limitations_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--s02-policy-table", type=Path, default=artifacts_dir / "tables" / "e07_policy_representations.parquet")
    parser.add_argument("--s04-corpus", type=Path, default=artifacts_dir / "results" / "e07_unified_behavior_corpus.parquet")
    parser.add_argument("--s05-modeling-dataset", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_modeling_dataset.parquet")
    parser.add_argument("--s05-metrics", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_metrics.parquet")
    parser.add_argument("--s05-calibration", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_calibration.parquet")
    parser.add_argument("--s05-model-manifest", type=Path, default=artifacts_dir / "models" / "e07_behavior_predictor" / "model_manifest.json")
    parser.add_argument("--min-behavior-rows", type=int, default=10)
    parser.add_argument("--max-syntax-features", type=int, default=50_000)
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


def extract_json_strings(value: Any, *, max_chars: int = 20_000) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text[:max_chars]

    strings: list[str] = []

    def walk(item: Any) -> None:
        if len(" ".join(strings)) > max_chars:
            return
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                strings.append(str(key))
                walk(child)
        elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            for child in item:
                walk(child)
        elif item is not None:
            strings.append(str(item))

    walk(payload)
    return " ".join(strings)[:max_chars]


def load_policy_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = [column for column in POLICY_METADATA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"S02 policy table missing required columns for S06: {missing}")
    policy = frame[list(POLICY_METADATA_COLUMNS)].drop_duplicates("canonical_policy_id").reset_index(drop=True)
    for column in policy.columns:
        if column.startswith("computed_") or column.endswith("_sha256"):
            policy[column] = policy[column].fillna("").astype(str)
    return policy


def source_balance_weights(frame: pd.DataFrame) -> pd.Series:
    counts = frame.groupby("source_experiment_id", observed=True).size().astype(float)
    median_count = float(counts.median()) if not counts.empty else 1.0
    weights = frame["source_experiment_id"].map(lambda value: median_count / max(1.0, float(counts.get(value, 1.0))))
    return weights.clip(lower=0.05, upper=20.0).astype(float)


def build_behavior_profile_matrix(
    modeling: pd.DataFrame,
    policy_ids: Sequence[str],
    *,
    feature_columns: Sequence[str] = BEHAVIOR_PROFILE_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid = modeling.copy()
    valid["canonical_policy_id"] = valid["canonical_policy_id"].fillna("__missing__").astype(str)
    valid = valid[valid["canonical_policy_id"] != "__missing__"].copy()
    valid["target_transformed"] = pd.to_numeric(valid["target_transformed"], errors="coerce")
    valid = valid[np.isfinite(valid["target_transformed"].to_numpy(dtype=float))].reset_index(drop=True)
    if valid.empty:
        return pd.DataFrame(index=policy_ids), valid, {"longRows": 0, "sourceBalanceWeights": {}}
    valid["source_balance_weight"] = source_balance_weights(valid)

    long_frames: list[pd.DataFrame] = []
    for column in feature_columns:
        if column not in valid.columns:
            valid[column] = "__missing__"
        sub = valid[["canonical_policy_id", "target_transformed", "source_balance_weight", column]].copy()
        sub[column] = sub[column].fillna("__missing__").astype(str).map(lambda value: value if value else "__missing__")
        sub["profile_feature"] = column + "=" + sub[column]
        long_frames.append(sub[["canonical_policy_id", "profile_feature", "target_transformed", "source_balance_weight"]])
    long = pd.concat(long_frames, ignore_index=True)
    profile = weighted_mean_profile(
        long,
        entity_column="canonical_policy_id",
        feature_column="profile_feature",
        value_column="target_transformed",
        weight_column="source_balance_weight",
    )
    profile = profile.reindex(policy_ids, fill_value=0.0)
    stats = {
        "validModelingRows": int(len(valid)),
        "longRows": int(len(long)),
        "profileFeatureColumns": int(profile.shape[1]),
        "sourceBalanceWeights": valid.groupby("source_experiment_id")["source_balance_weight"].first().astype(float).to_dict(),
    }
    return profile, valid, stats


def fit_pca_embedding(profile: pd.DataFrame, *, dims: int = EMBEDDING_DIMS) -> tuple[np.ndarray, dict[str, Any]]:
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
    embedding = pad_embedding(values, dims)
    return embedding, {
        **standardization_stats,
        "pcaComponents": int(n_components),
        "explainedVarianceRatio": [float(value) for value in pca.explained_variance_ratio_],
    }


def syntax_documents(policy: pd.DataFrame) -> list[str]:
    columns = [
        "source_policy_id",
        "display_name",
        "source_category",
        "policy_family",
        "algorithm",
        "representation_type",
        "abstraction_kind",
        "information_access",
        "execution_backend",
        "direction_support",
        "observation_contract",
        "action_vocabulary_json",
        "state_variables_json",
        "parameter_keys_json",
        "hash_validation_status",
        "parser_validation_status",
    ]
    docs = []
    for row in policy.to_dict(orient="records"):
        parts = [clean_text(row.get(column)) for column in columns]
        parts.append(extract_json_strings(row.get("representation_payload_json")))
        docs.append(" ".join(part for part in parts if part))
    return docs


def fit_syntax_embedding(policy: pd.DataFrame, *, dims: int, max_features: int) -> tuple[np.ndarray, dict[str, Any]]:
    docs = syntax_documents(policy)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=True,
        max_features=max_features,
    )
    syntax_matrix = vectorizer.fit_transform(docs)
    if syntax_matrix.shape[1] <= 1 or syntax_matrix.shape[0] <= 1:
        return np.zeros((len(docs), dims), dtype=float), {
            "syntaxFeatureCount": int(syntax_matrix.shape[1]),
            "syntaxSvdComponents": 0,
            "explainedVarianceRatio": [],
        }
    n_components = min(dims, syntax_matrix.shape[1] - 1, syntax_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    values = svd.fit_transform(syntax_matrix)
    return pad_embedding(values, dims), {
        "syntaxFeatureCount": int(syntax_matrix.shape[1]),
        "syntaxSvdComponents": int(n_components),
        "explainedVarianceRatio": [float(value) for value in svd.explained_variance_ratio_],
    }


def behavior_coverage(modeling_valid: pd.DataFrame, s04_corpus: pd.DataFrame, policy_ids: Sequence[str]) -> pd.DataFrame:
    s05 = (
        modeling_valid.groupby("canonical_policy_id", observed=True)
        .agg(
            behavior_observation_count=("target_transformed", "size"),
            behavior_source_count=("source_experiment_id", "nunique"),
            behavior_metric_family_count=("metric_family", "nunique"),
            behavior_world_family_count=("world_family", "nunique"),
            behavior_goal_family_count=("goal_family", "nunique"),
            behavior_perturbation_count=("perturbation_type", "nunique"),
            behavior_target_mean=("target_transformed", "mean"),
            behavior_target_std=("target_transformed", "std"),
        )
        .reset_index()
    )
    s04 = s04_corpus.copy()
    s04["canonical_policy_id"] = s04["canonical_policy_id"].fillna("").astype(str)
    s04 = s04[(s04["evidence_kind"].astype(str) == "metric_observation") & (s04["canonical_policy_id"] != "")].copy()
    s04 = s04[s04["canonical_policy_id"] != "__missing__"]
    s04_summary = (
        s04.groupby("canonical_policy_id", observed=True)
        .agg(
            s04_metric_observation_count=("canonical_policy_id", "size"),
            s04_source_count=("source_experiment_id", "nunique"),
            s04_metric_family_count=("metric_family", "nunique"),
            s04_world_count=("world_id", "nunique"),
            s04_goal_count=("canonical_goal_id", "nunique"),
            s04_perturbation_count=("perturbation_type", "nunique"),
        )
        .reset_index()
    )
    coverage = pd.DataFrame({"canonical_policy_id": list(policy_ids)})
    coverage = coverage.merge(s05, on="canonical_policy_id", how="left").merge(s04_summary, on="canonical_policy_id", how="left")
    count_columns = [column for column in coverage.columns if column.endswith("_count") or column.endswith("_observation_count")]
    for column in count_columns:
        coverage[column] = pd.to_numeric(coverage[column], errors="coerce").fillna(0).astype(int)
    for column in ("behavior_target_mean", "behavior_target_std"):
        coverage[column] = pd.to_numeric(coverage[column], errors="coerce").fillna(0.0)
    return coverage


def label_frame(policy: pd.DataFrame) -> pd.DataFrame:
    columns = ["canonical_policy_id", "policy_family", "algorithm", "source_experiment_id", "representation_type"]
    labels = policy[columns].copy()
    for column in columns:
        labels[column] = labels[column].fillna("__missing__").astype(str).map(lambda value: value if value else "__missing__")
    return labels


def random_label_agreement(neighbors: pd.DataFrame, labels: pd.DataFrame, label_column: str, *, seed: int) -> float:
    if neighbors.empty:
        return float("nan")
    rng = np.random.default_rng(seed)
    ids = labels["canonical_policy_id"].astype(str).to_numpy()
    values = labels[label_column].fillna("__missing__").astype(str).to_numpy()
    shuffled = rng.permutation(values)
    original_map = dict(zip(ids, values, strict=False))
    shuffled_map = dict(zip(ids, shuffled, strict=False))
    query_values = neighbors["query_policy_id"].astype(str).map(original_map).fillna("__missing__")
    neighbor_values = neighbors["neighbor_policy_id"].astype(str).map(shuffled_map).fillna("__missing__")
    valid = (query_values != "__missing__") & (neighbor_values != "__missing__")
    if not bool(valid.any()):
        return float("nan")
    return float((query_values[valid].to_numpy() == neighbor_values[valid].to_numpy()).mean())


def procrustes_coordinate_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape or ref.size == 0:
        return float("nan")
    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    cand_centered = cand - cand.mean(axis=0, keepdims=True)
    try:
        u, _, vt = np.linalg.svd(cand_centered.T @ ref_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    aligned = cand_centered @ (u @ vt)
    if np.std(ref_centered.ravel()) == 0 or np.std(aligned.ravel()) == 0:
        return float("nan")
    return float(np.corrcoef(ref_centered.ravel(), aligned.ravel())[0, 1])


def build_neighbors(
    embeddings: np.ndarray,
    policy_ids: Sequence[str],
    supported_ids: Sequence[str],
    labels: pd.DataFrame,
    *,
    embedding_name: str,
) -> pd.DataFrame:
    id_to_index = {policy_id: index for index, policy_id in enumerate(policy_ids)}
    supported_indices = [id_to_index[policy_id] for policy_id in supported_ids]
    values = embeddings[supported_indices]
    neighbors = cosine_neighbor_table(values, supported_ids, k=NEIGHBOR_K, embedding_name=embedding_name)
    return add_neighbor_labels(neighbors, labels)


def baseline_comparison(
    *,
    behavior_neighbors: pd.DataFrame,
    syntax_neighbors: pd.DataFrame,
    behavior_embedding: np.ndarray,
    syntax_embedding: np.ndarray,
    policy_ids: Sequence[str],
    supported_ids: Sequence[str],
    labels: pd.DataFrame,
) -> pd.DataFrame:
    id_to_index = {policy_id: index for index, policy_id in enumerate(policy_ids)}
    supported_indices = [id_to_index[policy_id] for policy_id in supported_ids]
    behavior_supported = behavior_embedding[supported_indices]
    syntax_supported = syntax_embedding[supported_indices]
    behavior_family = float(behavior_neighbors["same_policy_family"].astype(float).mean())
    syntax_family = float(syntax_neighbors["same_policy_family"].astype(float).mean())
    behavior_algorithm = float(behavior_neighbors["same_algorithm"].astype(float).mean())
    syntax_algorithm = float(syntax_neighbors["same_algorithm"].astype(float).mean())
    behavior_source = float(behavior_neighbors["same_source_experiment_id"].astype(float).mean())
    syntax_source = float(syntax_neighbors["same_source_experiment_id"].astype(float).mean())
    random_family = random_label_agreement(behavior_neighbors, labels, "policy_family", seed=RANDOM_SEED + 1)
    random_algorithm = random_label_agreement(behavior_neighbors, labels, "algorithm", seed=RANDOM_SEED + 2)
    overlap = neighbor_overlap_at_k(behavior_neighbors, syntax_neighbors, k=NEIGHBOR_K)
    distance_corr = sampled_pairwise_distance_spearman(
        behavior_supported,
        syntax_supported,
        max_pairs=50_000,
        seed=RANDOM_SEED + 3,
    )
    rows = [
        ("behavior_policy_family_agreement_at_5", behavior_family, "behavior_vs_label", "Share of behavior neighbors with same S02 policy_family."),
        ("syntax_policy_family_agreement_at_5", syntax_family, "syntax_vs_label", "Share of syntax neighbors with same S02 policy_family."),
        ("random_policy_family_agreement_at_5", random_family, "random_label_baseline", "Policy-family agreement after deterministic label permutation."),
        (
            "behavior_policy_family_agreement_lift_over_random_at_5",
            behavior_family - random_family,
            "behavior_minus_random",
            "Behavior neighbor policy-family agreement minus random-label agreement.",
        ),
        ("behavior_algorithm_agreement_at_5", behavior_algorithm, "behavior_vs_label", "Share of behavior neighbors with same algorithm label."),
        ("syntax_algorithm_agreement_at_5", syntax_algorithm, "syntax_vs_label", "Share of syntax neighbors with same algorithm label."),
        ("random_algorithm_agreement_at_5", random_algorithm, "random_label_baseline", "Algorithm agreement after deterministic label permutation."),
        (
            "behavior_algorithm_agreement_lift_over_random_at_5",
            behavior_algorithm - random_algorithm,
            "behavior_minus_random",
            "Behavior neighbor algorithm agreement minus random-label agreement.",
        ),
        ("behavior_source_experiment_agreement_at_5", behavior_source, "source_identity_check", "Share of behavior neighbors from same S02 source experiment."),
        ("syntax_source_experiment_agreement_at_5", syntax_source, "source_identity_check", "Share of syntax neighbors from same S02 source experiment."),
        ("behavior_syntax_neighbor_overlap_at_5", overlap, "behavior_vs_syntax", "Mean top-5 neighbor overlap between behavior and syntax embeddings."),
        ("behavior_syntax_pairwise_distance_spearman", distance_corr, "behavior_vs_syntax", "Spearman correlation of sampled pairwise distances."),
    ]
    return pd.DataFrame(
        [
            {
                "metric_name": name,
                "metric_value": float(value),
                "comparison_type": comparison_type,
                "detail": detail,
            }
            for name, value, comparison_type, detail in rows
        ]
    )


def stability_checks(
    *,
    modeling_valid: pd.DataFrame,
    policy_ids: Sequence[str],
    supported_ids: Sequence[str],
    main_embedding: np.ndarray,
    main_neighbors: pd.DataFrame,
) -> pd.DataFrame:
    id_to_index = {policy_id: index for index, policy_id in enumerate(policy_ids)}
    supported_indices = [id_to_index[policy_id] for policy_id in supported_ids]
    main_supported = main_embedding[supported_indices]
    rows = []
    for seed in STABILITY_SEEDS:
        mask = modeling_valid["corpus_row_id"].astype(str).map(lambda value: stable_fraction(value, salt=f"s06-stability-{seed}") < BOOTSTRAP_FRACTION)
        subset = modeling_valid[mask].reset_index(drop=True)
        profile, subset_valid, profile_stats = build_behavior_profile_matrix(subset, policy_ids)
        candidate_embedding, pca_stats = fit_pca_embedding(profile, dims=EMBEDDING_DIMS)
        candidate_supported = candidate_embedding[supported_indices]
        candidate_neighbors = cosine_neighbor_table(candidate_supported, supported_ids, k=NEIGHBOR_K, embedding_name=f"behavior_bootstrap_{seed}")
        rows.append(
            {
                "seed": int(seed),
                "bootstrap_fraction": float(BOOTSTRAP_FRACTION),
                "retained_rows": int(len(subset_valid)),
                "retained_policy_count": int(subset_valid["canonical_policy_id"].nunique()),
                "profile_feature_count": int(profile_stats["profileFeatureColumns"]),
                "pca_components": int(pca_stats["pcaComponents"]),
                "pairwise_distance_spearman": sampled_pairwise_distance_spearman(
                    main_supported,
                    candidate_supported,
                    max_pairs=50_000,
                    seed=RANDOM_SEED + seed,
                ),
                "aligned_coordinate_correlation": procrustes_coordinate_correlation(main_supported, candidate_supported),
                "neighbor_overlap_at_5": neighbor_overlap_at_k(main_neighbors, candidate_neighbors, k=NEIGHBOR_K),
            }
        )
    return pd.DataFrame(rows)


def plot_embedding_map(embeddings: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    source_order = sorted(embeddings["source_experiment_id"].fillna("__missing__").astype(str).unique())
    cmap = plt.get_cmap("tab10")
    colors = {source: cmap(index % 10) for index, source in enumerate(source_order)}
    fig, ax = plt.subplots(figsize=(11, 8))
    for source in source_order:
        subset = embeddings[embeddings["source_experiment_id"].fillna("__missing__").astype(str) == source]
        supported = subset["behavior_support_status"] == "behavior_profile_supported"
        sparse = subset["behavior_support_status"] != "behavior_profile_supported"
        ax.scatter(
            subset.loc[supported, "embedding_x"],
            subset.loc[supported, "embedding_y"],
            s=18,
            alpha=0.68,
            color=colors[source],
            label=source,
            linewidths=0,
        )
        if bool(sparse.any()):
            ax.scatter(
                subset.loc[sparse, "embedding_x"],
                subset.loc[sparse, "embedding_y"],
                s=18,
                alpha=0.25,
                color=colors[source],
                marker="x",
                linewidths=0.8,
            )
    ax.set_title("E07 S06 policy behavior embedding map")
    ax.set_xlabel("Behavior embedding PC1")
    ax.set_ylabel("Behavior embedding PC2")
    ax.grid(alpha=0.22)
    ax.legend(title="S02 source", fontsize=8, title_fontsize=9, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)


def outcome_classification(validation: pd.DataFrame, baseline: pd.DataFrame) -> str:
    if not bool(validation["success"].all()):
        return "constraining/contradictory"
    lookup = baseline.set_index("metric_name")["metric_value"].to_dict()
    family_lift = float(lookup.get("behavior_policy_family_agreement_lift_over_random_at_5", 0.0))
    source_agreement = float(lookup.get("behavior_source_experiment_agreement_at_5", 1.0))
    if family_lift > 0.02 and source_agreement < 0.95:
        return "supportive"
    return "null"


def s05_summary(metrics: pd.DataFrame, calibration: pd.DataFrame) -> dict[str, Any]:
    best = metrics.sort_values(["split_name", "mae"]).groupby("split_name", as_index=False).head(1)
    best = best[["split_name", "model_name", "mae", "calibration_ece", "calibration_slope"]]
    calibration_summary = calibration.groupby(["split_name", "model_name"], as_index=False).agg(mean_abs_gap=("abs_gap", "mean"))
    return {
        "bestModels": best.to_dict(orient="records"),
        "medianBestCalibrationEce": float(best["calibration_ece"].median()) if not best.empty else float("nan"),
        "meanCalibrationAbsGapByModel": calibration_summary.to_dict(orient="records"),
    }


def build_report(
    *,
    full_report_path: Path,
    artifacts_written: Sequence[str],
    embeddings: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    baseline: pd.DataFrame,
    stability: pd.DataFrame,
    validation: pd.DataFrame,
    outcome: str,
    behavior_stats: Mapping[str, Any],
    pca_stats: Mapping[str, Any],
    syntax_stats: Mapping[str, Any],
    s05_info: Mapping[str, Any],
    unit_test_result: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    validation_success = bool(validation["success"].all())
    supported_count = int((embeddings["behavior_support_status"] == "behavior_profile_supported").sum())
    sparse_count = int((embeddings["behavior_support_status"] == "sparse_behavior_profile").sum())
    no_behavior_count = int((embeddings["behavior_support_status"] == "no_behavior_observations").sum())
    caveats = (
        "Embeddings use observed S05 modeling targets aggregated into behavior profiles rather than causal policy effects; "
        "S05 calibration and source-imbalance limits remain; policies with sparse or absent behavior are retained with explicit support status; "
        "behavior profiles exclude source-experiment labels but can still reflect source-specific metric availability."
    )
    test_line = "not run"
    if unit_test_result is not None:
        test_line = f"{'pass' if unit_test_result['success'] else 'fail'}: `{unit_test_result['command']}` return code {unit_test_result['returnCode']}"
    s05_best = pd.DataFrame(s05_info.get("bestModels", []))
    text = f"""# E07 S06 Full Results: Policy Embeddings

## Top Summary

- Research step ID: S06
- Completion status: complete
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: {"pass" if validation_success else "fail"} ({int(validation['success'].sum())}/{len(validation)} checks passed)
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Lay summary: S06 placed policies into a behavior-space map by summarizing how each policy performs across S04/S05 metric, world, goal, and perturbation axes, then checked whether nearby policies are behaviorally meaningful rather than just syntactically similar.
- Recommended next action: Chief review of the S06 embedding stability and source-identity checks; proceed to S07 goal embeddings only after accepting these policy-map limitations.

## Frozen Question

Can policies be embedded so that behaviorally similar policies are close even when their source code is different?

## Inputs

- S02 policy representations: `{args.s02_policy_table}` (SHA-256 `{sha256_file(args.s02_policy_table)}`)
- S04 unified behavior corpus: `{args.s04_corpus}` (SHA-256 `{sha256_file(args.s04_corpus)}`)
- S05 balanced modeling dataset: `{args.s05_modeling_dataset}` (SHA-256 `{sha256_file(args.s05_modeling_dataset)}`)
- S05 predictor metrics: `{args.s05_metrics}` (SHA-256 `{sha256_file(args.s05_metrics)}`)
- S05 calibration bins: `{args.s05_calibration}` (SHA-256 `{sha256_file(args.s05_calibration)}`)
- S05 model manifest: `{args.s05_model_manifest}` (SHA-256 `{sha256_file(args.s05_model_manifest)}`)

## Methods

S06 used S02 canonical policy IDs as the row universe, so every policy-source record from S02 has one output embedding row. Behavior evidence came from the S05 deterministic balanced modeling dataset, which already capped rows per `(source_experiment_id, source_table, source_metric_name)` stratum. To account for remaining source imbalance, rows were aggregated with inverse source-count weights clipped to `[0.05, 20.0]`.

For each non-missing `canonical_policy_id`, the script built weighted mean competence profiles over behavior axes: `{', '.join(BEHAVIOR_PROFILE_COLUMNS)}`. Source experiment, source table, and source metric names were intentionally excluded from the behavior feature names. The resulting policy-by-feature matrix was standardized and reduced by PCA to {EMBEDDING_DIMS} behavior dimensions. `embedding_x` and `embedding_y` are the first two behavior dimensions.

The syntax/source-code baseline used S02 metadata plus DSL/source payload strings, encoded by character n-gram TF-IDF and reduced by truncated SVD to {EMBEDDING_DIMS} dimensions. Neighbor retrieval was compared between behavior and syntax embeddings.

Stability was measured with three deterministic 70% row resamples of the S05 modeling rows. For each resample, the behavior profile and PCA embedding were recomputed, then compared to the main map using sampled pairwise-distance Spearman correlation, Procrustes-aligned coordinate correlation, and top-5 neighbor overlap.

## Commands

- `python -m unittest tests.e07.test_policy_embedding_schema`
- `python scripts/e07_s06_policy_embeddings.py`

Unit-test result: {test_line}

## Dependencies and Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- numpy: {np.__version__}
- scikit-learn: {sklearn.__version__}
- matplotlib: {matplotlib.__version__}
- Worker count: single-process CPU analysis; thread environment OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'unset')}.
- Repository commit before S06 commit: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Branch: `{git_output(args.repo_dir, ['branch', '--show-current'])}`

## Results

Policy rows represented: {len(embeddings)}

Behavior support:

| support_status | policy_count |
| --- | --- |
| behavior_profile_supported | {supported_count} |
| sparse_behavior_profile | {sparse_count} |
| no_behavior_observations | {no_behavior_count} |

Behavior profile statistics:

- Valid S05 modeling rows used: {behavior_stats['validModelingRows']}
- Long profile rows after feature expansion: {behavior_stats['longRows']}
- Behavior profile feature columns: {behavior_stats['profileFeatureColumns']}
- PCA retained behavior features: {pca_stats['retainedFeatureCount']}
- PCA components: {pca_stats['pcaComponents']}
- PCA explained variance ratio: `{pca_stats['explainedVarianceRatio']}`

Syntax baseline statistics:

- TF-IDF features: {syntax_stats['syntaxFeatureCount']}
- SVD components: {syntax_stats['syntaxSvdComponents']}
- SVD explained variance ratio: `{syntax_stats['explainedVarianceRatio']}`

### S05 Calibration Context

S06 did not treat S05 predictions as direct ground truth. It used the S05 balanced modeling rows and carried the calibration caveat forward. Median calibration ECE among the best S05 models was `{s05_info.get('medianBestCalibrationEce')}`.

{markdown_table(s05_best)}

### Coverage Summary by Source

{markdown_table(coverage_summary)}

### Behavior vs Syntax Baseline

{markdown_table(baseline)}

### Stability

{markdown_table(stability)}

## Validation

{markdown_table(validation)}

## Output Artifacts

{chr(10).join(f'- `{item}`' for item in artifacts_written)}

## Caveats, Blockers, and Limitations

- The embeddings are computational proxy summaries over upstream simulated metrics; they do not establish biological mechanism or causal policy effects.
- The behavior target is the S05 oriented signed-log proxy target, so different physical units are deliberately compressed into one comparative behavior scale.
- S05 source imbalance is reduced by balanced sampling and source-balance aggregation weights, but source-specific metric availability still shapes the map.
- Sparse/no-behavior policies are retained with explicit `behavior_support_status`; their behavior coordinates are inferred from zero-filled missing profiles and should not be overinterpreted.
- The source-code/DSL baseline is a syntax proxy, not a formal semantic equivalence proof.
- The map is ready for goal/world analysis as a bounded representation artifact, not as a final ontology.

## Recommended Next Action

Chief review should inspect the validation checks, especially stability and source-experiment agreement, before authorizing S07 goal embeddings. Stop here and do not start S07 until instructed.
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
        unit_test_result = run_command([sys.executable, "-m", "unittest", "tests.e07.test_policy_embedding_schema"], args.repo_dir)

    policy = load_policy_metadata(args.s02_policy_table)
    policy_ids = policy["canonical_policy_id"].astype(str).tolist()
    modeling = pd.read_parquet(args.s05_modeling_dataset)
    metrics = pd.read_parquet(args.s05_metrics)
    calibration = pd.read_parquet(args.s05_calibration)
    s04_corpus = pd.read_parquet(
        args.s04_corpus,
        columns=[
            "canonical_policy_id",
            "source_experiment_id",
            "source_metric_name",
            "metric_family",
            "world_id",
            "canonical_goal_id",
            "perturbation_type",
            "evidence_kind",
        ],
    )

    behavior_profile, modeling_valid, behavior_stats = build_behavior_profile_matrix(modeling, policy_ids)
    behavior_embedding, pca_stats = fit_pca_embedding(behavior_profile, dims=EMBEDDING_DIMS)
    syntax_embedding, syntax_stats = fit_syntax_embedding(policy, dims=EMBEDDING_DIMS, max_features=args.max_syntax_features)
    coverage = behavior_coverage(modeling_valid, s04_corpus, policy_ids)

    embeddings = policy.copy()
    embeddings = embeddings.merge(coverage, on="canonical_policy_id", how="left")
    for index in range(EMBEDDING_DIMS):
        embeddings[f"behavior_embedding_dim_{index:02d}"] = behavior_embedding[:, index]
        embeddings[f"syntax_embedding_dim_{index:02d}"] = syntax_embedding[:, index]
    embeddings["embedding_x"] = embeddings["behavior_embedding_dim_00"]
    embeddings["embedding_y"] = embeddings["behavior_embedding_dim_01"]
    embeddings["syntax_embedding_x"] = embeddings["syntax_embedding_dim_00"]
    embeddings["syntax_embedding_y"] = embeddings["syntax_embedding_dim_01"]
    embeddings["embedding_method"] = "source_balanced_behavior_profile_pca"
    embeddings["syntax_baseline_method"] = "s02_metadata_payload_char_tfidf_svd"
    embeddings["research_step_id"] = STEP_ID
    embeddings["schema_version"] = POLICY_EMBEDDING_SCHEMA_VERSION
    embeddings["behavior_support_status"] = np.select(
        [
            embeddings["behavior_observation_count"] >= args.min_behavior_rows,
            embeddings["behavior_observation_count"] > 0,
        ],
        ["behavior_profile_supported", "sparse_behavior_profile"],
        default="no_behavior_observations",
    )

    labels = label_frame(policy)
    supported_ids = embeddings.loc[embeddings["behavior_observation_count"] >= args.min_behavior_rows, "canonical_policy_id"].astype(str).tolist()
    behavior_neighbors = build_neighbors(behavior_embedding, policy_ids, supported_ids, labels, embedding_name="behavior")
    syntax_neighbors = build_neighbors(syntax_embedding, policy_ids, supported_ids, labels, embedding_name="syntax")
    neighbors = pd.concat([behavior_neighbors, syntax_neighbors], ignore_index=True)
    baseline = baseline_comparison(
        behavior_neighbors=behavior_neighbors,
        syntax_neighbors=syntax_neighbors,
        behavior_embedding=behavior_embedding,
        syntax_embedding=syntax_embedding,
        policy_ids=policy_ids,
        supported_ids=supported_ids,
        labels=labels,
    )
    stability = stability_checks(
        modeling_valid=modeling_valid,
        policy_ids=policy_ids,
        supported_ids=supported_ids,
        main_embedding=behavior_embedding,
        main_neighbors=behavior_neighbors,
    )
    validation = validate_policy_embedding_artifacts(
        embeddings,
        neighbors,
        stability,
        baseline,
        expected_policy_ids=policy_ids,
        min_supported_policies=100,
    )
    outcome = outcome_classification(validation, baseline)

    coverage_summary = (
        embeddings.groupby("source_experiment_id", as_index=False)
        .agg(
            policy_count=("canonical_policy_id", "size"),
            behavior_supported_policies=("behavior_support_status", lambda values: int((values == "behavior_profile_supported").sum())),
            sparse_or_absent_policies=("behavior_support_status", lambda values: int((values != "behavior_profile_supported").sum())),
            s05_modeling_rows=("behavior_observation_count", "sum"),
            s04_metric_rows=("s04_metric_observation_count", "sum"),
        )
        .sort_values("source_experiment_id", kind="mergesort")
    )
    s05_info = s05_summary(metrics, calibration)

    policy_embeddings_path = results_dir / "e07_policy_embeddings.parquet"
    policy_embeddings_csv_path = tables_dir / "e07_policy_embeddings_sample.csv"
    unified_embeddings_path = results_dir / "e07_embeddings.parquet"
    neighbor_path = results_dir / "e07_policy_embedding_neighbor_retrieval.parquet"
    neighbor_csv_path = tables_dir / "e07_policy_embedding_neighbor_retrieval_sample.csv"
    stability_path = results_dir / "e07_policy_embedding_stability.parquet"
    stability_csv_path = tables_dir / "e07_policy_embedding_stability.csv"
    baseline_path = tables_dir / "e07_policy_embedding_baseline_comparison.csv"
    coverage_path = tables_dir / "e07_policy_embedding_coverage_summary.csv"
    validation_path = step_dir / "e07_s06_validation_checks.csv"
    figure_path = figures_dir / "policy_embedding_map.png"
    full_report_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    config_path = step_dir / "s06_config.json"

    embeddings.to_parquet(policy_embeddings_path, index=False)
    embeddings.head(200).to_csv(policy_embeddings_csv_path, index=False)
    unified_embeddings = embeddings.copy()
    unified_embeddings.insert(0, "embedding_entity_type", "policy")
    unified_embeddings.to_parquet(unified_embeddings_path, index=False)
    neighbors.to_parquet(neighbor_path, index=False)
    neighbors.head(500).to_csv(neighbor_csv_path, index=False)
    stability.to_parquet(stability_path, index=False)
    stability.to_csv(stability_csv_path, index=False)
    baseline.to_csv(baseline_path, index=False)
    coverage_summary.to_csv(coverage_path, index=False)
    validation.to_csv(validation_path, index=False)
    plot_embedding_map(embeddings, figure_path)
    write_json(
        config_path,
        {
            "schemaVersion": POLICY_EMBEDDING_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "randomSeed": RANDOM_SEED,
            "embeddingDims": EMBEDDING_DIMS,
            "neighborK": NEIGHBOR_K,
            "behaviorProfileColumns": list(BEHAVIOR_PROFILE_COLUMNS),
            "stabilitySeeds": list(STABILITY_SEEDS),
            "bootstrapFraction": BOOTSTRAP_FRACTION,
            "minBehaviorRows": args.min_behavior_rows,
            "maxSyntaxFeatures": args.max_syntax_features,
            "sourceBalanceWeights": behavior_stats["sourceBalanceWeights"],
            "s05CalibrationSummary": s05_info,
        },
    )

    artifacts_written = [
        str(full_report_path),
        str(policy_embeddings_path),
        str(policy_embeddings_csv_path),
        str(unified_embeddings_path),
        str(neighbor_path),
        str(neighbor_csv_path),
        str(stability_path),
        str(stability_csv_path),
        str(baseline_path),
        str(coverage_path),
        str(figure_path),
        str(validation_path),
        str(config_path),
        str(artifact_manifest_path),
    ]
    build_report(
        full_report_path=full_report_path,
        artifacts_written=artifacts_written,
        embeddings=embeddings,
        coverage_summary=coverage_summary,
        baseline=baseline,
        stability=stability,
        validation=validation,
        outcome=outcome,
        behavior_stats=behavior_stats,
        pca_stats=pca_stats,
        syntax_stats=syntax_stats,
        s05_info=s05_info,
        unit_test_result=unit_test_result,
        args=args,
    )

    manifest_payload = {
        "schemaVersion": "eidosoma.e07.s06.artifact_manifest.v1",
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
            "s02PolicyTable": {"path": str(args.s02_policy_table), "sha256": sha256_file(args.s02_policy_table)},
            "s04Corpus": {"path": str(args.s04_corpus), "sha256": sha256_file(args.s04_corpus)},
            "s05ModelingDataset": {"path": str(args.s05_modeling_dataset), "sha256": sha256_file(args.s05_modeling_dataset)},
            "s05Metrics": {"path": str(args.s05_metrics), "sha256": sha256_file(args.s05_metrics)},
            "s05Calibration": {"path": str(args.s05_calibration), "sha256": sha256_file(args.s05_calibration)},
            "s05ModelManifest": {"path": str(args.s05_model_manifest), "sha256": sha256_file(args.s05_model_manifest)},
        },
        "coverage": {
            "policyRows": int(len(embeddings)),
            "behaviorSupportedPolicies": int((embeddings["behavior_support_status"] == "behavior_profile_supported").sum()),
            "sparseBehaviorPolicies": int((embeddings["behavior_support_status"] == "sparse_behavior_profile").sum()),
            "noBehaviorPolicies": int((embeddings["behavior_support_status"] == "no_behavior_observations").sum()),
            "supportedPolicyIdsUsedForNeighborValidation": int(len(supported_ids)),
        },
        "artifacts": [
            self_referential_artifact_entry(full_report_path, artifacts_dir, "S06 full-results report."),
            artifact_entry(policy_embeddings_path, artifacts_dir, "Primary S06 policy embedding table."),
            artifact_entry(policy_embeddings_csv_path, artifacts_dir, "Small CSV sample of the policy embedding table."),
            artifact_entry(unified_embeddings_path, artifacts_dir, "Policy rows in shared E07 embedding table form."),
            artifact_entry(neighbor_path, artifacts_dir, "Behavior and syntax top-k neighbor retrieval table."),
            artifact_entry(neighbor_csv_path, artifacts_dir, "Small CSV sample of neighbor retrieval rows."),
            artifact_entry(stability_path, artifacts_dir, "Embedding stability metrics across deterministic row resamples."),
            artifact_entry(stability_csv_path, artifacts_dir, "Embedding stability metrics in CSV format."),
            artifact_entry(baseline_path, artifacts_dir, "Behavior-vs-syntax baseline comparison metrics."),
            artifact_entry(coverage_path, artifacts_dir, "Policy embedding coverage summary by source experiment."),
            artifact_entry(figure_path, artifacts_dir, "Two-dimensional policy behavior embedding map."),
            artifact_entry(validation_path, artifacts_dir, "S06 validation checks."),
            artifact_entry(config_path, artifacts_dir, "S06 embedding configuration and S05 calibration summary."),
            self_referential_artifact_entry(artifact_manifest_path, artifacts_dir, "S06 artifact manifest."),
        ],
    }
    write_json(artifact_manifest_path, manifest_payload)
    print(f"[S06] wrote {full_report_path}")
    print(f"[S06] validation {int(validation['success'].sum())}/{len(validation)} outcome={outcome}")


if __name__ == "__main__":
    main()
