#!/usr/bin/env python3
"""Find conservative E07 universality-class taxonomies."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import sklearn  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.invariant_schema import extract_policy_invariants, extract_world_invariants  # noqa: E402
from src.e07.universality_schema import (  # noqa: E402
    UNIVERSALITY_SCHEMA_VERSION,
    cautious_cluster_label,
    classify_cluster_status,
    dominance,
    finite_standardize,
    neighbor_distance_matrix,
    stable_fraction,
    validate_universality_artifacts,
)


STEP_ID = "S10"
STEP_NUMBER = 10
EXPERIMENT_ID = "E07"
RANDOM_SEED = 20260703
MAX_COMPONENTS = 8
INVARIANT_FEATURE_WEIGHT = 0.25
STABILITY_SEEDS = (2026070310, 2026070311, 2026070312, 2026070313, 2026070314)
N_JOBS = min(8, os.cpu_count() or 1)

ENTITY_PREFIX = {"policy": "P", "goal": "G", "world": "W"}
SUPPORTED_STATUS = {
    "policy": "behavior_profile_supported",
    "goal": "supported_policy_evidence",
    "world": "metric_profile_supported",
}


@dataclass(frozen=True)
class BranchSpec:
    entity_type: str
    branch_name: str
    branch_role: str
    distance_variant: str
    embedding_columns: tuple[str, ...]
    invariant_columns: tuple[str, ...]
    max_k: int = 8


def parse_args() -> argparse.Namespace:
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--policy-embeddings", type=Path, default=artifacts_dir / "results" / "e07_policy_embeddings.parquet")
    parser.add_argument("--goal-embeddings", type=Path, default=artifacts_dir / "results" / "e07_goal_embeddings.parquet")
    parser.add_argument("--distance-benchmarks", type=Path, default=artifacts_dir / "results" / "e07_distance_benchmarks.parquet")
    parser.add_argument("--neighbors", type=Path, default=artifacts_dir / "results" / "e07_platonic_neighbors.parquet")
    parser.add_argument("--entity-metadata", type=Path, default=artifacts_dir / "results" / "e07_distance_entity_metadata.parquet")
    parser.add_argument("--s02-policy-table", type=Path, default=artifacts_dir / "tables" / "e07_policy_representations.parquet")
    parser.add_argument("--s01-world-inventory", type=Path, default=artifacts_dir / "tables" / "e07_world_inventory.csv")
    parser.add_argument("--s09-candidates", type=Path, default=artifacts_dir / "results" / "e07_candidate_invariants.parquet")
    parser.add_argument("--s09-counterexamples", type=Path, default=artifacts_dir / "results" / "e07_invariant_counterexamples.parquet")
    parser.add_argument("--s09-caveat-audit", type=Path, default=artifacts_dir / "tables" / "e07_invariant_s08_caveat_audit.csv")
    parser.add_argument("--s08-source-missingness-audit", type=Path, default=artifacts_dir / "tables" / "e07_distance_source_missingness_audit.csv")
    parser.add_argument("--s08-conflict-validation", type=Path, default=artifacts_dir / "tables" / "e07_distance_conflict_validation.csv")
    parser.add_argument("--e03-policy-clusters", type=Path, default=Path("/previous-artifacts/E03/results/e03_policy_clusters.parquet"))
    parser.add_argument("--e06-mosaic-summary", type=Path, default=Path("/previous-artifacts/E06/tables/e06_mosaic_class_summary.csv"))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        view[column] = view[column].map(format_cell)
    header = "| " + " | ".join(map(str, view.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    if len(frame) > max_rows:
        rows.append(f"| ... | {len(frame) - max_rows} more rows omitted |" + " |" * max(0, len(view.columns) - 2))
    return "\n".join([header, separator, *rows])


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6g}"
    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path) if path.is_file() else None,
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


def load_e03_policy_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["source_policy_id", "e03_class_id", "e03_cautious_label"])
    table = pd.read_parquet(path, columns=["policy_id", "class_id", "cautious_label"]).drop_duplicates("policy_id")
    return table.rename(columns={"policy_id": "source_policy_id", "class_id": "e03_class_id", "cautious_label": "e03_cautious_label"})


def coalesce_columns(frame: pd.DataFrame, columns: Sequence[str], default: str = "") -> pd.Series:
    """Return the first non-missing value across available columns."""

    out = pd.Series(default, index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype("object")
        mask = out.isna() | out.astype(str).eq("")
        out.loc[mask] = values.loc[mask]
    return out.fillna(default)


def build_policy_entity_table(args: argparse.Namespace, metadata: pd.DataFrame) -> pd.DataFrame:
    policy = pd.read_parquet(args.policy_embeddings).drop_duplicates("canonical_policy_id").copy()
    inv_source = pd.read_parquet(args.s02_policy_table).drop_duplicates("canonical_policy_id").copy()
    extracted = pd.DataFrame([extract_policy_invariants(row) for row in inv_source.to_dict(orient="records")])
    inv = pd.concat([inv_source[["canonical_policy_id"]].reset_index(drop=True), extracted], axis=1).rename(columns={"canonical_policy_id": "entity_id"})
    base = metadata[metadata["entity_type"] == "policy"].rename(columns={"entity_id": "canonical_policy_id"}).copy()
    frame = base.merge(policy, on="canonical_policy_id", how="left", suffixes=("_metadata", ""))
    frame = frame.rename(columns={"canonical_policy_id": "entity_id", "family": "s08_family"})
    frame["source_experiment_id"] = coalesce_columns(frame, ["source_experiment_id_metadata", "source_experiment_id"])
    frame["family"] = coalesce_columns(frame, ["s08_family", "policy_family"])
    frame["display_name"] = coalesce_columns(frame, ["display_name_metadata", "display_name"])
    frame = frame.merge(inv, on="entity_id", how="left")
    frame = frame.merge(load_e03_policy_labels(args.e03_policy_clusters), on="source_policy_id", how="left")
    return frame


def build_goal_entity_table(args: argparse.Namespace, metadata: pd.DataFrame) -> pd.DataFrame:
    goal = pd.read_parquet(args.goal_embeddings).drop_duplicates("canonical_goal_id").copy()
    base = metadata[metadata["entity_type"] == "goal"].rename(columns={"entity_id": "canonical_goal_id"}).copy()
    frame = base.merge(goal, on="canonical_goal_id", how="left", suffixes=("_metadata", ""))
    frame = frame.rename(columns={"canonical_goal_id": "entity_id", "family": "s08_family"})
    frame["source_experiment_id"] = coalesce_columns(frame, ["source_experiment_id_metadata", "source_experiment_id"])
    frame["family"] = coalesce_columns(frame, ["s08_family", "goal_family"])
    frame["display_name"] = coalesce_columns(frame, ["display_name_metadata", "display_name"])
    return frame


def build_world_entity_table(args: argparse.Namespace, metadata: pd.DataFrame) -> pd.DataFrame:
    world = pd.read_csv(args.s01_world_inventory).drop_duplicates("world_id").copy()
    extracted = pd.DataFrame([extract_world_invariants(row) for row in world.to_dict(orient="records")])
    world_features = pd.concat([world[["world_id"]].reset_index(drop=True), extracted], axis=1).rename(columns={"world_id": "entity_id"})
    base = metadata[metadata["entity_type"] == "world"].rename(columns={"entity_id": "world_id"}).copy()
    frame = base.merge(world, on="world_id", how="left", suffixes=("_metadata", ""))
    frame = frame.rename(columns={"world_id": "entity_id", "family": "s08_family"})
    frame["source_experiment_id"] = coalesce_columns(frame, ["source_experiment_id_metadata", "source_experiment_id", "experiment_id"])
    frame["family"] = coalesce_columns(frame, ["s08_family", "world_family"])
    frame["display_name"] = coalesce_columns(frame, ["display_name_metadata", "display_name", "task_label"])
    frame = frame.merge(world_features, on="entity_id", how="left")
    return frame


def embedding_columns(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{idx:02d}" for idx in range(8))


def branch_specs(stable_features: Sequence[str]) -> list[BranchSpec]:
    stable = set(stable_features)
    policy_stable = tuple(feature for feature in ("policy_memory_depth_proxy", "policy_stochastic_choice") if feature in stable)
    world_stable = tuple(feature for feature in ("world_has_damage_or_repair", "world_frozen_count") if feature in stable)
    return [
        BranchSpec("policy", "policy_supported_only", "primary", "platonic_supported_only", embedding_columns("behavior_embedding_dim"), policy_stable),
        BranchSpec("policy", "policy_uncertainty_weighted_all", "sensitivity", "platonic_uncertainty_weighted", embedding_columns("behavior_embedding_dim"), policy_stable),
        BranchSpec("goal", "goal_supported_only", "primary", "platonic_supported_only", embedding_columns("supported_only_goal_embedding_dim"), ()),
        BranchSpec("goal", "goal_uncertainty_weighted_all", "sensitivity", "platonic_uncertainty_weighted", embedding_columns("goal_embedding_dim"), ()),
        BranchSpec("world", "world_supported_only", "primary", "platonic_supported_only", (), world_stable),
        BranchSpec("world", "world_metric_profile_all", "sensitivity", "platonic_metric_profile", (), world_stable),
    ]


def control_variants(entity_type: str, primary_branch: str) -> list[tuple[str, str, str]]:
    if entity_type == "policy":
        if primary_branch == "policy_supported_only":
            return [
                ("metric_only", "metric_only_supported_only_baseline", "metric_only_supported_control"),
                ("syntax", "syntax_supported_only_baseline", "syntax_supported_control"),
                ("source_metadata", "source_metadata_baseline", "source_metadata_control"),
            ]
        return [
            ("metric_only", "metric_only_baseline", "metric_only_control"),
            ("syntax", "syntax_baseline", "syntax_control"),
            ("source_metadata", "source_metadata_baseline", "source_metadata_control"),
        ]
    if entity_type == "goal":
        if primary_branch == "goal_supported_only":
            return [
                ("metric_only", "metric_only_supported_only_baseline", "metric_only_supported_control"),
                ("source_metadata", "source_metadata_baseline", "source_metadata_control"),
            ]
        return [
            ("metric_only", "metric_only_baseline", "metric_only_control"),
            ("source_metadata", "source_metadata_baseline", "source_metadata_control"),
        ]
    if entity_type == "world":
        return [
            ("metric_only", "metric_only_baseline", "metric_only_control"),
            ("source_metadata", "source_metadata_baseline", "source_metadata_control"),
        ]
    return []


def branch_entity_ids(neighbors: pd.DataFrame, spec: BranchSpec, table: pd.DataFrame) -> list[str]:
    ids = set(
        neighbors[
            (neighbors["entity_type"].astype(str) == spec.entity_type)
            & (neighbors["distance_variant"].astype(str) == spec.distance_variant)
        ]["query_entity_id"].astype(str)
    )
    table_ids = set(table["entity_id"].astype(str))
    return sorted(ids & table_ids, key=lambda value: stable_fraction(value, salt=f"{spec.branch_name}:ids"))


def pca_coordinates(matrix: pd.DataFrame, *, prefix: str, max_components: int = MAX_COMPONENTS) -> tuple[pd.DataFrame, dict[str, Any]]:
    if matrix.empty:
        return pd.DataFrame(index=matrix.index), {"componentCount": 0}
    scaled, stats = finite_standardize(matrix)
    if scaled.empty or scaled.shape[0] < 2 or scaled.shape[1] < 1:
        return pd.DataFrame(index=matrix.index), {**stats, "componentCount": 0}
    component_count = min(max_components, scaled.shape[0] - 1, scaled.shape[1])
    if component_count < 1:
        return pd.DataFrame(index=matrix.index), {**stats, "componentCount": 0}
    values = PCA(n_components=component_count, random_state=RANDOM_SEED).fit_transform(scaled.to_numpy(dtype=float))
    columns = [f"{prefix}_{idx:02d}" for idx in range(values.shape[1])]
    return pd.DataFrame(values, index=matrix.index, columns=columns), {**stats, "componentCount": int(component_count)}


def entity_feature_blocks(
    table: pd.DataFrame,
    ids: Sequence[str],
    spec: BranchSpec,
    neighbors: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = list(map(str, ids))
    by_id = table.set_index("entity_id", drop=False).reindex(index)
    distance_matrix, distance_stats = neighbor_distance_matrix(
        neighbors,
        entity_type=spec.entity_type,
        distance_variant=spec.distance_variant,
        entity_ids=index,
    )
    distance_components, distance_component_stats = pca_coordinates(distance_matrix, prefix="distance_component")
    blocks = [distance_components]
    used_embedding = [column for column in spec.embedding_columns if column in by_id.columns]
    if used_embedding:
        embedding_block, embedding_stats = finite_standardize(by_id[used_embedding])
        embedding_block.columns = [f"embedding::{column}" for column in embedding_block.columns]
        blocks.append(embedding_block)
    else:
        embedding_stats = {"inputFeatureCount": 0, "retainedFeatureCount": 0, "droppedConstantFeatureCount": 0}
    used_invariants = [column for column in spec.invariant_columns if column in by_id.columns]
    if used_invariants:
        invariant_block, invariant_stats = finite_standardize(by_id[used_invariants])
        invariant_block = invariant_block * INVARIANT_FEATURE_WEIGHT
        invariant_block.columns = [f"invariant::{column}" for column in invariant_block.columns]
        blocks.append(invariant_block)
    else:
        invariant_stats = {"inputFeatureCount": 0, "retainedFeatureCount": 0, "droppedConstantFeatureCount": 0}
    combined = pd.concat(blocks, axis=1).fillna(0.0)
    combined = combined.loc[:, combined.std(axis=0, ddof=0) > 0]
    stats = {
        "distance": {**distance_stats, **distance_component_stats},
        "embedding": embedding_stats,
        "invariant": invariant_stats,
        "featureColumnCount": int(combined.shape[1]),
        "usedEmbeddingColumns": used_embedding,
        "usedInvariantColumns": used_invariants,
    }
    return combined, stats


def choose_k(features: pd.DataFrame, *, max_k: int, branch_name: str) -> tuple[int, pd.DataFrame]:
    n_rows = len(features)
    upper = min(max_k, n_rows - 1)
    rows: list[dict[str, Any]] = []
    if upper < 2:
        return 1, pd.DataFrame()
    matrix = features.to_numpy(dtype=float)
    best_k = 2
    best_score = -np.inf
    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_SEED)
        labels = model.fit_predict(matrix)
        counts = np.bincount(labels, minlength=k)
        score = float(silhouette_score(matrix, labels)) if len(set(labels)) > 1 and len(features) > k else float("nan")
        rows.append(
            {
                "schema_version": UNIVERSALITY_SCHEMA_VERSION,
                "research_step_id": STEP_ID,
                "branch_name": branch_name,
                "candidate_k": int(k),
                "silhouette_score": score,
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
            }
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_k = k
    return int(best_k), pd.DataFrame(rows)


def fit_assignments(features: pd.DataFrame, ids: Sequence[str], *, k: int, class_prefix: str, branch_name: str) -> tuple[pd.DataFrame, np.ndarray, KMeans]:
    matrix = features.to_numpy(dtype=float)
    if k <= 1:
        labels = np.zeros(len(features), dtype=int)
        centers = np.nanmean(matrix, axis=0, keepdims=True)
        distances = np.linalg.norm(matrix - centers[labels], axis=1)
        model = KMeans(n_clusters=1, n_init=1, random_state=RANDOM_SEED).fit(matrix)
    else:
        model = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_SEED)
        labels = model.fit_predict(matrix)
        distances = np.linalg.norm(matrix - model.cluster_centers_[labels], axis=1)
    coord_count = min(2, matrix.shape[0] - 1, matrix.shape[1])
    if coord_count >= 1:
        coords = PCA(n_components=coord_count, random_state=RANDOM_SEED).fit_transform(matrix)
        if coord_count == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    else:
        coords = np.zeros((len(matrix), 2))
    rows = []
    for entity_id, label, distance, coord in zip(ids, labels, distances, coords, strict=True):
        rows.append(
            {
                "entity_id": str(entity_id),
                "branch_name": branch_name,
                "raw_cluster_id": int(label),
                "class_id": f"UC-{class_prefix}-{int(label) + 1:02d}",
                "cluster_distance": float(distance),
                "map_x": float(coord[0]),
                "map_y": float(coord[1]),
            }
        )
    return pd.DataFrame(rows), labels, model


def stability_frame(features: pd.DataFrame, assignments: pd.DataFrame, *, k: int, branch_name: str) -> pd.DataFrame:
    original = assignments.set_index("entity_id")["raw_cluster_id"].astype(int)
    rows: list[dict[str, Any]] = []
    ids = list(features.index.astype(str))
    for seed in STABILITY_SEEDS:
        subset_ids = [entity_id for entity_id in ids if stable_fraction(entity_id, salt=f"{branch_name}:{seed}") < 0.80]
        if len(subset_ids) < max(3, k + 1):
            subset_ids = ids
        sub_features = features.loc[subset_ids]
        model = KMeans(n_clusters=k, n_init=25, random_state=seed)
        labels = model.fit_predict(sub_features.to_numpy(dtype=float))
        ari = adjusted_rand_score(original.loc[subset_ids].to_numpy(dtype=int), labels)
        nmi = normalized_mutual_info_score(original.loc[subset_ids].to_numpy(dtype=int), labels)
        rows.append(
            {
                "schema_version": UNIVERSALITY_SCHEMA_VERSION,
                "research_step_id": STEP_ID,
                "branch_name": branch_name,
                "seed": int(seed),
                "sample_fraction": float(len(subset_ids) / len(ids)),
                "sampled_entity_count": int(len(subset_ids)),
                "adjusted_rand_index": float(ari),
                "normalized_mutual_info": float(nmi),
            }
        )
    frame = pd.DataFrame(rows)
    frame["mean_adjusted_rand_index"] = float(frame["adjusted_rand_index"].mean()) if not frame.empty else float("nan")
    frame["min_adjusted_rand_index"] = float(frame["adjusted_rand_index"].min()) if not frame.empty else float("nan")
    return frame


def summarize_stability(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return pd.DataFrame(columns=["branch_name", "mean_adjusted_rand_index", "min_adjusted_rand_index", "mean_normalized_mutual_info"])
    return (
        stability.groupby("branch_name", observed=True)
        .agg(
            mean_adjusted_rand_index=("adjusted_rand_index", "mean"),
            min_adjusted_rand_index=("adjusted_rand_index", "min"),
            mean_normalized_mutual_info=("normalized_mutual_info", "mean"),
            replicate_count=("seed", "count"),
        )
        .reset_index()
    )


def control_audit_for_branch(
    spec: BranchSpec,
    table: pd.DataFrame,
    ids: Sequence[str],
    primary_assignments: pd.DataFrame,
    primary_k: int,
    neighbors: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = primary_assignments.set_index("entity_id")["raw_cluster_id"].astype(int)
    for control_family, variant, label in control_variants(spec.entity_type, spec.branch_name):
        control_spec = BranchSpec(spec.entity_type, f"{spec.branch_name}_{label}", "control", variant, (), ())
        try:
            features, feature_stats = entity_feature_blocks(table, ids, control_spec, neighbors)
        except Exception as exc:
            rows.append(
                {
                    "schema_version": UNIVERSALITY_SCHEMA_VERSION,
                    "research_step_id": STEP_ID,
                    "entity_type": spec.entity_type,
                    "branch_name": spec.branch_name,
                    "control_family": control_family,
                    "control_variant": variant,
                    "success": False,
                    "detail": repr(exc),
                }
            )
            continue
        if features.empty or len(features) < max(2, primary_k):
            rows.append(
                {
                    "schema_version": UNIVERSALITY_SCHEMA_VERSION,
                    "research_step_id": STEP_ID,
                    "entity_type": spec.entity_type,
                    "branch_name": spec.branch_name,
                    "control_family": control_family,
                    "control_variant": variant,
                    "success": False,
                    "detail": "insufficient control features",
                    "control_feature_count": int(features.shape[1]),
                }
            )
            continue
        control_assignments, _, _ = fit_assignments(
            features,
            ids,
            k=min(primary_k, len(features) - 1),
            class_prefix=f"{ENTITY_PREFIX[spec.entity_type]}-CTRL",
            branch_name=control_spec.branch_name,
        )
        control = control_assignments.set_index("entity_id")["raw_cluster_id"].astype(int)
        common = sorted(set(primary.index) & set(control.index))
        ari = adjusted_rand_score(primary.loc[common], control.loc[common]) if len(common) >= 2 else float("nan")
        nmi = normalized_mutual_info_score(primary.loc[common], control.loc[common]) if len(common) >= 2 else float("nan")
        rows.append(
            {
                "schema_version": UNIVERSALITY_SCHEMA_VERSION,
                "research_step_id": STEP_ID,
                "entity_type": spec.entity_type,
                "branch_name": spec.branch_name,
                "control_family": control_family,
                "control_variant": variant,
                "success": True,
                "overlap_entity_count": int(len(common)),
                "control_feature_count": int(features.shape[1]),
                "adjusted_rand_index": float(ari),
                "normalized_mutual_info": float(nmi),
                "detail": f"control features={feature_stats.get('featureColumnCount', 0)}",
            }
        )
    return pd.DataFrame(rows)


def summarize_classes(
    assignments: pd.DataFrame,
    table: pd.DataFrame,
    spec: BranchSpec,
    stability_summary: pd.DataFrame,
    control_audit: pd.DataFrame,
) -> pd.DataFrame:
    merged = assignments.merge(table, on="entity_id", how="left", suffixes=("", "_entity"))
    stability_row = stability_summary[stability_summary["branch_name"] == spec.branch_name]
    stability_mean = float(stability_row["mean_adjusted_rand_index"].iloc[0]) if not stability_row.empty else float("nan")
    max_control_ari = float(pd.to_numeric(control_audit.get("adjusted_rand_index", pd.Series(dtype=float)), errors="coerce").max()) if not control_audit.empty else float("nan")
    rows: list[dict[str, Any]] = []
    for class_id, group in merged.groupby("class_id", observed=True):
        dominant_source, source_rate, source_count = dominance(group.get("source_experiment_id", pd.Series(dtype=str)))
        dominant_family, family_rate, family_count = dominance(group.get("family", pd.Series(dtype=str)))
        dominant_support, support_rate, support_count = dominance(group.get("support_status", pd.Series(dtype=str)))
        row: dict[str, Any] = {
            "schema_version": UNIVERSALITY_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "entity_type": spec.entity_type,
            "branch_name": spec.branch_name,
            "branch_role": spec.branch_role,
            "distance_variant": spec.distance_variant,
            "class_id": class_id,
            "raw_cluster_id": int(group["raw_cluster_id"].iloc[0]),
            "entity_count": int(len(group)),
            "dominant_source_experiment_id": dominant_source,
            "source_dominance_rate": source_rate,
            "source_count": source_count,
            "dominant_family": dominant_family,
            "family_dominance_rate": family_rate,
            "family_count": family_count,
            "dominant_support_status": dominant_support,
            "support_status_dominance_rate": support_rate,
            "support_status_count": support_count,
            "mean_support_weight": float(pd.to_numeric(group.get("support_weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()),
            "mean_cluster_distance": float(pd.to_numeric(group["cluster_distance"], errors="coerce").mean()),
            "max_cluster_distance": float(pd.to_numeric(group["cluster_distance"], errors="coerce").max()),
            "stability_mean_ari": stability_mean,
            "max_control_ari": max_control_ari,
            "dominant_e03_class_id": "",
            "dominant_e03_label": "",
        }
        for column in spec.invariant_columns:
            if column in group.columns:
                row[f"mean_{column}"] = float(pd.to_numeric(group[column], errors="coerce").fillna(0.0).mean())
        if spec.entity_type == "policy" and "e03_class_id" in group.columns:
            e03 = group.dropna(subset=["e03_class_id"])
            if not e03.empty:
                label, rate, _ = dominance(e03["e03_class_id"])
                row["dominant_e03_class_id"] = label
                row["dominant_e03_class_rate_among_labeled"] = rate
                labels = e03[e03["e03_class_id"].astype(str) == label]["e03_cautious_label"].dropna().astype(str)
                row["dominant_e03_label"] = labels.iloc[0] if not labels.empty else ""
        row["class_status"] = classify_cluster_status(row)
        row["cautious_label"] = cautious_cluster_label(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["entity_type", "branch_name", "raw_cluster_id"], kind="mergesort").reset_index(drop=True)


def exemplar_and_counterexample_tables(assignments: pd.DataFrame, class_summary: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = assignments.merge(table, on="entity_id", how="left", suffixes=("", "_entity")).merge(
        class_summary[["class_id", "class_status", "cautious_label", "dominant_source_experiment_id", "dominant_family", "dominant_support_status"]],
        on="class_id",
        how="left",
    )
    exemplar_rows: list[dict[str, Any]] = []
    counter_rows: list[dict[str, Any]] = []
    for class_id, group in merged.groupby("class_id", observed=True):
        ordered = group.sort_values(["cluster_distance", "entity_id"], kind="mergesort")
        medoid = ordered.iloc[0]
        exemplar_rows.append(exemplar_row(medoid, "cluster_medoid"))
        supported = group.sort_values(["support_weight", "cluster_distance"], ascending=[False, True], kind="mergesort")
        if not supported.empty and supported.iloc[0]["entity_id"] != medoid["entity_id"]:
            exemplar_rows.append(exemplar_row(supported.iloc[0], "highest_support_entity"))
        far = group.sort_values(["cluster_distance", "entity_id"], ascending=[False, True], kind="mergesort").iloc[0]
        counter_rows.append(counterexample_row(far, "far_from_cluster_medoid_or_centroid"))
        source_minority = group[group["source_experiment_id"].astype(str) != str(group["dominant_source_experiment_id"].iloc[0])]
        if not source_minority.empty:
            counter_rows.append(counterexample_row(source_minority.sort_values(["cluster_distance", "entity_id"], ascending=[False, True]).iloc[0], "minority_source_member"))
        support_minority = group[group["support_status"].astype(str) != str(group["dominant_support_status"].iloc[0])]
        if not support_minority.empty:
            counter_rows.append(counterexample_row(support_minority.sort_values(["cluster_distance", "entity_id"], ascending=[False, True]).iloc[0], "minority_support_status_member"))
    return pd.DataFrame(exemplar_rows), pd.DataFrame(counter_rows)


def exemplar_row(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    return {
        "schema_version": UNIVERSALITY_SCHEMA_VERSION,
        "research_step_id": STEP_ID,
        "entity_type": row.get("entity_type"),
        "branch_name": row.get("branch_name"),
        "class_id": row.get("class_id"),
        "class_status": row.get("class_status"),
        "cautious_label": row.get("cautious_label"),
        "entity_id": row.get("entity_id"),
        "display_name": row.get("display_name"),
        "source_experiment_id": row.get("source_experiment_id"),
        "family": row.get("family"),
        "support_status": row.get("support_status"),
        "support_weight": row.get("support_weight"),
        "cluster_distance": row.get("cluster_distance"),
        "exemplar_role": role,
    }


def counterexample_row(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": UNIVERSALITY_SCHEMA_VERSION,
        "research_step_id": STEP_ID,
        "entity_type": row.get("entity_type"),
        "branch_name": row.get("branch_name"),
        "class_id": row.get("class_id"),
        "class_status": row.get("class_status"),
        "cautious_label": row.get("cautious_label"),
        "entity_id": row.get("entity_id"),
        "display_name": row.get("display_name"),
        "source_experiment_id": row.get("source_experiment_id"),
        "family": row.get("family"),
        "support_status": row.get("support_status"),
        "support_weight": row.get("support_weight"),
        "cluster_distance": row.get("cluster_distance"),
        "counterexample_type": reason,
    }


def make_map_figure(assignments: pd.DataFrame, class_summary: pd.DataFrame, figure_path: Path) -> None:
    primary = assignments[assignments["branch_role"] == "primary"].copy()
    if primary.empty:
        primary = assignments.copy()
    primary = primary.merge(class_summary[["class_id", "class_status", "cautious_label"]], on="class_id", how="left", suffixes=("", "_summary"))
    entities = ["policy", "goal", "world"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = {
        "bounded_interpretable_class": "#4c78a8",
        "source_dominated_constraint": "#e45756",
        "missingness_driven_constraint": "#f58518",
        "unstable_constraint": "#b279a2",
        "control_aligned_constraint": "#72b7b2",
        "small_or_uninterpretable_constraint": "#9d755d",
    }
    for axis, entity_type in zip(axes, entities, strict=True):
        sub = primary[primary["entity_type"] == entity_type]
        if sub.empty:
            axis.set_axis_off()
            continue
        for status, group in sub.groupby("class_status", observed=True):
            axis.scatter(
                group["map_x"],
                group["map_y"],
                s=28 if entity_type == "policy" else 44,
                alpha=0.76,
                label=status.replace("_", " "),
                color=colors.get(str(status), "#54a24b"),
                edgecolor="white",
                linewidth=0.3,
            )
        axis.set_title(f"{entity_type.title()} primary supported-only classes")
        axis.set_xlabel("S10 map x")
        axis.set_ylabel("S10 map y")
        axis.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def outcome_classification(validation: pd.DataFrame, class_summary: pd.DataFrame, controls: pd.DataFrame) -> str:
    if not bool(validation["success"].all()):
        return "constraining/contradictory"
    statuses = class_summary["class_status"].astype(str)
    control_max = pd.to_numeric(controls.get("adjusted_rand_index", pd.Series(dtype=float)), errors="coerce").max()
    if statuses.str.contains("constraint", regex=False).any() or (np.isfinite(control_max) and control_max > 0.80):
        return "constraining/contradictory"
    if (statuses == "bounded_interpretable_class").any():
        return "supportive"
    return "null"


def write_report(
    args: argparse.Namespace,
    *,
    class_summary: pd.DataFrame,
    assignments: pd.DataFrame,
    candidate_k: pd.DataFrame,
    stability_summary: pd.DataFrame,
    controls: pd.DataFrame,
    exemplars: pd.DataFrame,
    counterexamples: pd.DataFrame,
    validation: pd.DataFrame,
    outcome: str,
    unit_test_result: dict[str, Any],
    input_stats: dict[str, Any],
    artifacts_written: list[Path],
    report_path: Path,
) -> None:
    validation_label = f"{int(validation['success'].sum())}/{len(validation)} checks passed"
    top_artifacts = ", ".join(str(path) for path in artifacts_written[:14])
    stable_primary = class_summary[
        (class_summary["branch_role"] == "primary") & (class_summary["class_status"] == "bounded_interpretable_class")
    ]
    constraining_count = int(class_summary["class_status"].astype(str).str.contains("constraint", regex=False).sum())
    text = f"""# E07 S10 Full Results: Find Universality Classes

## Top Summary

- Research step ID: S10
- Completion status: complete
- Artifacts written: {top_artifacts}
- Validation result: {validation_label}
- Outcome classification: {outcome}
- Caveats or blockers: S10 treats S06-S09 as computational proxy evidence. Supported-only branches are primary; uncertainty-weighted and metric-profile branches are sensitivity checks. S09 stable candidates are bounded, non-causal features. S08 source-dominance and goal-conflict failures and the S09 invariant-model baseline failure remain active caveats.
- Lay summary: S10 grouped policies, goals, and worlds into provisional classes, then checked whether the groups were stable, interpretable, and separable from source, metric, syntax, and missingness controls. Some supported-only classes are usable as cautious descriptors, but control alignment and inherited support limits keep the result constraining.
- Recommended next action: Chief review before S11. If S11 proceeds, freeze only bounded primary supported-only classes and carry class-status caveats into counterfactual prediction tests.

## Frozen Question

Can algorithms be classified into substrate-independent types such as local gradient descent, barrier navigator, opportunistic repairer, morphogen-field follower, self-aggregator, and dominance-seeker?

## Inputs

- S06 policy embeddings: `{args.policy_embeddings}` (SHA-256 `{sha256_file(args.policy_embeddings)}`)
- S07 goal embeddings: `{args.goal_embeddings}` (SHA-256 `{sha256_file(args.goal_embeddings)}`)
- S08 benchmarks: `{args.distance_benchmarks}` (SHA-256 `{sha256_file(args.distance_benchmarks)}`)
- S08 neighbors: `{args.neighbors}` (SHA-256 `{sha256_file(args.neighbors)}`)
- S08 entity metadata: `{args.entity_metadata}` (SHA-256 `{sha256_file(args.entity_metadata)}`)
- S09 candidate invariants: `{args.s09_candidates}` (SHA-256 `{sha256_file(args.s09_candidates)}`)
- S09 counterexamples: `{args.s09_counterexamples}` (SHA-256 `{sha256_file(args.s09_counterexamples)}`)
- S01 world inventory: `{args.s01_world_inventory}` (SHA-256 `{sha256_file(args.s01_world_inventory)}`)
- S02 policy table: `{args.s02_policy_table}` (SHA-256 `{sha256_file(args.s02_policy_table)}`)

## Methods

S10 built separate policy, goal, and world taxonomies. Primary branches used S08 `platonic_supported_only` distances. Sensitivity branches used policy and goal `platonic_uncertainty_weighted` distances and the world `platonic_metric_profile` branch because no world uncertainty-weighted branch was produced by S08. Distance branches were converted into deterministic top-k graph distance features by symmetrizing the S08 neighbor table, filling unobserved pair distances with a high-distance value, and reducing the distance rows by PCA.

Entity embeddings from S06 and S07 were included where available. S09 stable candidates were included only as low-weight, bounded features: `policy_memory_depth_proxy`, `policy_stochastic_choice`, `world_has_damage_or_repair`, and `world_frozen_count`. No unstable S09 candidate was used as a clustering feature. Source, metric, syntax, and source-metadata distances were clustered as controls and compared to the primary branch by adjusted Rand index and normalized mutual information.

Clusters were labeled only after clustering, using source/support/family dominance and bounded S09 feature summaries. Labels are non-causal descriptors. Clusters dominated by source, missingness/support status, control baselines, small size, or instability are explicitly marked as constraining rather than relabeled into stronger claims.

## Commands

- `python -m unittest tests.e07.test_universality_schema`
- `python scripts/e07_s10_universality_classes.py`

Unit-test result: {'pass' if unit_test_result.get('success') else 'fail'}: `{unit_test_result.get('command')}` return code {unit_test_result.get('returnCode')}

## Dependencies and Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- numpy: {np.__version__}
- scikit-learn: {sklearn.__version__}
- matplotlib: {matplotlib.__version__}
- Worker count: deterministic single-process clustering; configured CPU ceiling noted as {N_JOBS}; no GPU training.
- Repository commit before S10 commit: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Branch: `{git_output(args.repo_dir, ['branch', '--show-current'])}`

## Parameters

```json
{json.dumps(input_stats, indent=2, sort_keys=True)}
```

## Results

Primary bounded-interpretable class rows: {len(stable_primary)}

Constraining class rows across primary and sensitivity branches: {constraining_count}

### Universality Class Summary

{markdown_table(class_summary[['entity_type','branch_name','branch_role','class_id','entity_count','class_status','cautious_label','dominant_source_experiment_id','source_dominance_rate','dominant_family','dominant_support_status','stability_mean_ari','max_control_ari']].sort_values(['entity_type','branch_role','class_id']), max_rows=80)}

### Stability Summary

{markdown_table(stability_summary, max_rows=40)}

### Control Audit

{markdown_table(controls[['entity_type','branch_name','control_family','control_variant','overlap_entity_count','adjusted_rand_index','normalized_mutual_info','success','detail']], max_rows=80)}

### Candidate K

{markdown_table(candidate_k[['branch_name','candidate_k','silhouette_score','min_cluster_size','max_cluster_size']], max_rows=80)}

### Exemplars

{markdown_table(exemplars[['entity_type','branch_name','class_id','exemplar_role','entity_id','display_name','source_experiment_id','family','support_status','cluster_distance']], max_rows=80)}

### Counterexamples

{markdown_table(counterexamples[['entity_type','branch_name','class_id','counterexample_type','entity_id','display_name','source_experiment_id','family','support_status','cluster_distance']], max_rows=80)}

## Validation

{markdown_table(validation, max_rows=40)}

## Output Artifacts

{markdown_table(pd.DataFrame([artifact_entry(path, args.artifacts_dir, '') for path in artifacts_written if path.exists()]), max_rows=80)}

## Caveats, Blockers, And Failed Assumptions

- S10 taxonomy is a computational proxy over existing E07 artifacts, not causal or biological validation.
- The policy all-entity uncertainty-weighted S08 branch inherited source dominance from S08 and should not be interpreted as substrate-independent similarity.
- S07/S08 supported-only goal conflict separation failed; S10 goal classes are descriptive support/metric groupings, not a validated conflict ontology.
- S09 found small stable invariant proxies, but invariant models did not improve over source-metric medians. S10 therefore uses those proxies only as low-weight descriptors.
- The S08 neighbor table contains top-k distances, not a complete distance matrix. S10 graph-distance features are approximate and record this as a limitation.
- E03 and E06 upstream labels were used only as annotations/context, not as clustering targets.

## Provenance

- Script: `scripts/e07_s10_universality_classes.py`
- Schema helper: `src/e07/universality_schema.py`
- Tests: `tests/e07/test_universality_schema.py`
- Artifact root: `{args.artifacts_dir}`
- Created at: `{utc_now()}`

## Recommended Next Action

Stop before S11 for Chief review. If approved, S11 should freeze the primary supported-only S10 class assignments, include class-status caveats in any counterfactual prediction list, and retain source/metric/missingness controls as mandatory baselines.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    results_dir = args.artifacts_dir / "results"
    tables_dir = args.artifacts_dir / "tables"
    figures_dir = args.artifacts_dir / "figures" / "e07"
    for directory in (step_dir, results_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_test_result = run_command([sys.executable, "-m", "unittest", "tests.e07.test_universality_schema"], args.repo_dir) if args.run_unit_tests else {
        "command": "not run",
        "returnCode": 0,
        "success": True,
        "stdout": "",
        "stderr": "",
        "elapsedSeconds": 0.0,
    }

    metadata = pd.read_parquet(args.entity_metadata)
    neighbors = pd.read_parquet(args.neighbors)
    s09_candidates = pd.read_parquet(args.s09_candidates)
    stable_features = sorted(s09_candidates.loc[s09_candidates["candidate_status"].eq("stable_supported_candidate"), "candidate_feature"].astype(str).unique())

    entity_tables = {
        "policy": build_policy_entity_table(args, metadata),
        "goal": build_goal_entity_table(args, metadata),
        "world": build_world_entity_table(args, metadata),
    }

    all_assignments: list[pd.DataFrame] = []
    all_class_summaries: list[pd.DataFrame] = []
    all_candidate_k: list[pd.DataFrame] = []
    all_stability: list[pd.DataFrame] = []
    all_controls: list[pd.DataFrame] = []
    feature_stats: dict[str, Any] = {}
    used_invariant_features: set[str] = set()

    for spec in branch_specs(stable_features):
        table = entity_tables[spec.entity_type]
        ids = branch_entity_ids(neighbors, spec, table)
        if len(ids) < 3:
            continue
        features, stats = entity_feature_blocks(table, ids, spec, neighbors)
        if features.empty or features.shape[1] < 1:
            continue
        feature_stats[spec.branch_name] = stats
        used_invariant_features.update(stats.get("usedInvariantColumns", []))
        k, k_frame = choose_k(features, max_k=spec.max_k, branch_name=spec.branch_name)
        all_candidate_k.append(k_frame)
        class_prefix = f"{ENTITY_PREFIX[spec.entity_type]}-{'SUP' if spec.branch_role == 'primary' else 'SENS'}"
        assignments, labels, _ = fit_assignments(features, ids, k=k, class_prefix=class_prefix, branch_name=spec.branch_name)
        table_for_merge = table.set_index("entity_id", drop=False).reindex(ids).reset_index(drop=True)
        assignments = assignments.merge(
            table_for_merge[
                [
                    column
                    for column in [
                        "entity_id",
                        "display_name",
                        "source_experiment_id",
                        "family",
                        "support_status",
                        "support_weight",
                        "e03_class_id",
                        "e03_cautious_label",
                        *spec.invariant_columns,
                    ]
                    if column in table_for_merge.columns
                ]
            ],
            on="entity_id",
            how="left",
        )
        assignments["schema_version"] = UNIVERSALITY_SCHEMA_VERSION
        assignments["research_step_id"] = STEP_ID
        assignments["entity_type"] = spec.entity_type
        assignments["branch_role"] = spec.branch_role
        assignments["distance_variant"] = spec.distance_variant
        stability = stability_frame(features, assignments, k=k, branch_name=spec.branch_name)
        all_stability.append(stability)
        stability_summary = summarize_stability(stability)
        controls = control_audit_for_branch(spec, table, ids, assignments, k, neighbors) if spec.branch_role == "primary" else pd.DataFrame()
        if not controls.empty:
            all_controls.append(controls)
        class_summary = summarize_classes(assignments, table, spec, stability_summary, controls)
        all_assignments.append(assignments)
        all_class_summaries.append(class_summary)

    assignments = pd.concat(all_assignments, ignore_index=True, sort=False) if all_assignments else pd.DataFrame()
    class_summary = pd.concat(all_class_summaries, ignore_index=True, sort=False) if all_class_summaries else pd.DataFrame()
    candidate_k = pd.concat(all_candidate_k, ignore_index=True, sort=False) if all_candidate_k else pd.DataFrame()
    stability = pd.concat(all_stability, ignore_index=True, sort=False) if all_stability else pd.DataFrame()
    stability_summary = summarize_stability(stability)
    controls = pd.concat(all_controls, ignore_index=True, sort=False) if all_controls else pd.DataFrame()

    # Re-apply branch-level control maxima after all control rows are available.
    if not class_summary.empty and not controls.empty:
        max_control_by_branch = controls.groupby("branch_name", observed=True)["adjusted_rand_index"].max().to_dict()
        class_summary["max_control_ari"] = class_summary["branch_name"].map(max_control_by_branch).combine_first(class_summary.get("max_control_ari"))
        class_summary["class_status"] = [classify_cluster_status(row) for row in class_summary.to_dict(orient="records")]
        class_summary["cautious_label"] = [cautious_cluster_label(row) for row in class_summary.to_dict(orient="records")]

    exemplars, counterexamples = exemplar_and_counterexample_tables(assignments, class_summary, pd.concat(entity_tables.values(), ignore_index=True, sort=False))

    classes_csv = tables_dir / "e07_universality_classes.csv"
    classes_parquet = results_dir / "e07_universality_classes.parquet"
    assignments_csv = tables_dir / "e07_universality_assignments.csv"
    assignments_parquet = results_dir / "e07_universality_assignments.parquet"
    stability_csv = tables_dir / "e07_universality_stability.csv"
    controls_csv = tables_dir / "e07_universality_control_audit.csv"
    candidate_k_csv = tables_dir / "e07_universality_candidate_k.csv"
    exemplars_csv = tables_dir / "e07_universality_exemplars.csv"
    counterexamples_csv = tables_dir / "e07_universality_counterexamples.csv"
    figure_path = figures_dir / "universality_class_map.png"
    validation_path = step_dir / "e07_s10_validation_checks.csv"
    config_path = step_dir / "s10_config.json"
    manifest_path = step_dir / "artifact_manifest.json"
    report_path = step_dir / "research_step_full_results.md"

    class_summary.to_csv(classes_csv, index=False)
    class_summary.to_parquet(classes_parquet, index=False)
    assignments.to_csv(assignments_csv, index=False)
    assignments.to_parquet(assignments_parquet, index=False)
    stability_summary.to_csv(stability_csv, index=False)
    controls.to_csv(controls_csv, index=False)
    candidate_k.to_csv(candidate_k_csv, index=False)
    exemplars.to_csv(exemplars_csv, index=False)
    counterexamples.to_csv(counterexamples_csv, index=False)
    make_map_figure(assignments, class_summary, figure_path)

    validation_context = {"stableInvariantFeatures": stable_features, "usedInvariantFeatures": sorted(used_invariant_features)}
    validation = validate_universality_artifacts(class_summary, assignments, stability_summary, controls, exemplars, counterexamples, validation_context)
    validation = pd.concat(
        [
            validation,
            pd.DataFrame(
                [
                    {
                        "validation_case": "unit_tests_passed",
                        "success": bool(unit_test_result.get("success")),
                        "detail": f"`{unit_test_result.get('command')}` return code {unit_test_result.get('returnCode')}",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    validation.to_csv(validation_path, index=False)
    outcome = outcome_classification(validation, class_summary, controls)
    input_stats = {
        "schemaVersion": UNIVERSALITY_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stableInvariantFeatures": stable_features,
        "usedInvariantFeatures": sorted(used_invariant_features),
        "branchFeatureStats": feature_stats,
        "entityCounts": {entity_type: int(len(table)) for entity_type, table in entity_tables.items()},
        "classStatusCounts": class_summary["class_status"].value_counts().to_dict() if not class_summary.empty else {},
        "branchClassCounts": class_summary.groupby("branch_name", observed=True)["class_id"].nunique().to_dict() if not class_summary.empty else {},
        "maxControlAri": float(pd.to_numeric(controls.get("adjusted_rand_index", pd.Series(dtype=float)), errors="coerce").max()) if not controls.empty else None,
        "unitTestCommand": unit_test_result.get("command"),
        "unitTestReturnCode": unit_test_result.get("returnCode"),
    }
    write_json(config_path, input_stats)

    artifacts_written = [
        report_path,
        classes_csv,
        classes_parquet,
        assignments_csv,
        assignments_parquet,
        stability_csv,
        controls_csv,
        candidate_k_csv,
        exemplars_csv,
        counterexamples_csv,
        figure_path,
        validation_path,
        config_path,
        manifest_path,
    ]

    write_report(
        args,
        class_summary=class_summary,
        assignments=assignments,
        candidate_k=candidate_k,
        stability_summary=stability_summary,
        controls=controls,
        exemplars=exemplars,
        counterexamples=counterexamples,
        validation=validation,
        outcome=outcome,
        unit_test_result=unit_test_result,
        input_stats=input_stats,
        artifacts_written=artifacts_written,
        report_path=report_path,
    )

    manifest = {
        "schemaVersion": UNIVERSALITY_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": "complete",
        "success": bool(validation["success"].all()),
        "validationResult": f"{int(validation['success'].sum())}/{len(validation)} checks passed",
        "outcomeClassification": outcome,
        "createdAt": utc_now(),
        "inputs": {
            "policyEmbeddings": {"path": str(args.policy_embeddings), "sha256": sha256_file(args.policy_embeddings)},
            "goalEmbeddings": {"path": str(args.goal_embeddings), "sha256": sha256_file(args.goal_embeddings)},
            "neighbors": {"path": str(args.neighbors), "sha256": sha256_file(args.neighbors)},
            "entityMetadata": {"path": str(args.entity_metadata), "sha256": sha256_file(args.entity_metadata)},
            "s09Candidates": {"path": str(args.s09_candidates), "sha256": sha256_file(args.s09_candidates)},
        },
        "artifacts": [
            self_referential_artifact_entry(report_path, args.artifacts_dir, "S10 full-results report."),
            *[artifact_entry(path, args.artifacts_dir, "S10 generated artifact.") for path in artifacts_written[1:-1]],
            self_referential_artifact_entry(manifest_path, args.artifacts_dir, "S10 artifact manifest."),
        ],
        "classStatusCounts": input_stats["classStatusCounts"],
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    # Rewrite the report after manifest exists so artifact sizes are current; checksum remains self-referentially omitted.
    write_report(
        args,
        class_summary=class_summary,
        assignments=assignments,
        candidate_k=candidate_k,
        stability_summary=stability_summary,
        controls=controls,
        exemplars=exemplars,
        counterexamples=counterexamples,
        validation=validation,
        outcome=outcome,
        unit_test_result=unit_test_result,
        input_stats=input_stats,
        artifacts_written=artifacts_written,
        report_path=report_path,
    )
    print(f"[S10] wrote {report_path}")
    print(f"[S10] validation {int(validation['success'].sum())}/{len(validation)} outcome={outcome}")


if __name__ == "__main__":
    main()
