#!/usr/bin/env python3
"""Define and validate E07 S08 Platonic distances."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.distance_schema import (  # noqa: E402
    PLATONIC_DISTANCE_SCHEMA_VERSION,
    attach_neighbor_metadata,
    euclidean_neighbor_table,
    metadata_feature_matrix,
    sampled_pairwise_distance_spearman,
    support_weight_from_status,
    validate_platonic_distance_artifacts,
    weighted_mean_profile,
)
from src.e07.goal_embedding_schema import unique_conflict_pairs  # noqa: E402


STEP_ID = "S08"
STEP_NUMBER = 8
EXPERIMENT_ID = "E07"
RANDOM_SEED = 20260703
EMBEDDING_DIMS = 8
NEIGHBOR_K = 5
MIN_SUPPORTED_ROWS = 10
SOURCE_DOMINANCE_THRESHOLD = 0.95
UNCERTAINTY_LOW_SUPPORT_THRESHOLD = 0.80
SUPPORTED_LOW_SUPPORT_THRESHOLD = 0.05

POLICY_BEHAVIOR_COLUMNS = tuple(f"behavior_embedding_dim_{index:02d}" for index in range(EMBEDDING_DIMS))
POLICY_SYNTAX_COLUMNS = tuple(f"syntax_embedding_dim_{index:02d}" for index in range(EMBEDDING_DIMS))
GOAL_PRIMARY_COLUMNS = tuple(f"goal_embedding_dim_{index:02d}" for index in range(EMBEDDING_DIMS))
GOAL_SUPPORTED_COLUMNS = tuple(f"supported_only_goal_embedding_dim_{index:02d}" for index in range(EMBEDDING_DIMS))

POLICY_METRIC_FEATURES = (
    "metric_family",
    "world_family",
    "goal_family",
    "perturbation_type",
    "goal_metric_family",
    "goal_target_direction",
    "substrate_kind",
)
GOAL_METRIC_FEATURES = (
    "metric_family",
    "world_family",
    "perturbation_type",
    "policy_family",
    "policy_algorithm",
    "policy_representation_type",
    "substrate_kind",
)
WORLD_METRIC_FEATURES = (
    "metric_family",
    "goal_family",
    "perturbation_type",
    "policy_family",
    "policy_algorithm",
    "goal_metric_family",
    "substrate_kind",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--s01-world-inventory", type=Path, default=artifacts_dir / "tables" / "e07_world_inventory.csv")
    parser.add_argument("--s04-corpus", type=Path, default=artifacts_dir / "results" / "e07_unified_behavior_corpus.parquet")
    parser.add_argument("--s05-modeling-dataset", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_modeling_dataset.parquet")
    parser.add_argument("--s05-metrics", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_metrics.parquet")
    parser.add_argument("--s05-calibration", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_calibration.parquet")
    parser.add_argument("--s06-policy-embeddings", type=Path, default=artifacts_dir / "results" / "e07_policy_embeddings.parquet")
    parser.add_argument("--s07-goal-embeddings", type=Path, default=artifacts_dir / "results" / "e07_goal_embeddings.parquet")
    parser.add_argument("--s03-goal-conflicts", type=Path, default=artifacts_dir / "tables" / "e07_goal_conflicts.parquet")
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


def source_balance_weights(frame: pd.DataFrame, source_column: str = "source_experiment_id") -> pd.Series:
    if frame.empty or source_column not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)
    counts = frame.groupby(source_column, observed=True).size().astype(float)
    median_count = float(counts.median()) if not counts.empty else 1.0
    weights = frame[source_column].map(lambda value: median_count / max(1.0, float(counts.get(value, 1.0))))
    return weights.clip(lower=0.05, upper=20.0).astype(float)


def numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def policy_entities(policies: pd.DataFrame) -> pd.DataFrame:
    frame = policies.copy()
    frame["entity_type"] = "policy"
    frame["entity_id"] = frame["canonical_policy_id"].astype(str)
    frame["family"] = frame["policy_family"].fillna("__missing__").astype(str)
    frame["support_status"] = frame["behavior_support_status"].fillna("no_behavior_observations").astype(str)
    frame["support_weight"] = [
        support_weight_from_status(status, count, min_rows=MIN_SUPPORTED_ROWS)
        for status, count in zip(frame["support_status"], frame["behavior_observation_count"], strict=False)
    ]
    return frame[
        [
            "entity_type",
            "entity_id",
            "source_experiment_id",
            "family",
            "support_status",
            "support_weight",
            "display_name",
        ]
    ].copy()


def goal_entities(goals: pd.DataFrame) -> pd.DataFrame:
    frame = goals.copy()
    frame["entity_type"] = "goal"
    frame["entity_id"] = frame["canonical_goal_id"].astype(str)
    frame["family"] = frame["goal_family"].fillna("__missing__").astype(str)
    frame["support_status"] = frame["goal_support_status"].fillna("no_s05_goal_behavior_rows").astype(str)
    supported_rows = pd.to_numeric(frame.get("supported_policy_row_count", 0), errors="coerce").fillna(0.0)
    modeled_rows = pd.to_numeric(frame.get("goal_behavior_row_count", 0), errors="coerce").fillna(0.0)
    frame["support_weight"] = np.select(
        [supported_rows > 0, modeled_rows > 0],
        [1.0, 0.25],
        default=0.0,
    )
    return frame[
        [
            "entity_type",
            "entity_id",
            "source_experiment_id",
            "family",
            "support_status",
            "support_weight",
            "display_name",
        ]
    ].copy()


def world_entities(worlds: pd.DataFrame, modeling: pd.DataFrame) -> pd.DataFrame:
    counts = (
        modeling.assign(world_id=modeling["world_id"].fillna("__missing__").astype(str))
        .query("world_id != '__missing__'")
        .groupby("world_id", observed=True)
        .size()
        .rename("s05_modeling_row_count")
        .reset_index()
    )
    frame = worlds.copy()
    frame["entity_type"] = "world"
    frame["entity_id"] = frame["world_id"].astype(str)
    frame["source_experiment_id"] = frame["experiment_id"].fillna("__missing__").astype(str)
    frame["family"] = frame["world_family"].fillna("__missing__").astype(str)
    frame = frame.merge(counts, on="world_id", how="left")
    frame["s05_modeling_row_count"] = pd.to_numeric(frame["s05_modeling_row_count"], errors="coerce").fillna(0).astype(int)
    frame["support_status"] = np.select(
        [frame["s05_modeling_row_count"] >= MIN_SUPPORTED_ROWS, frame["s05_modeling_row_count"] > 0],
        ["metric_profile_supported", "sparse_metric_profile"],
        default="no_s05_world_rows",
    )
    frame["support_weight"] = [
        support_weight_from_status(status, count, supported_status="metric_profile_supported", min_rows=MIN_SUPPORTED_ROWS)
        for status, count in zip(frame["support_status"], frame["s05_modeling_row_count"], strict=False)
    ]
    frame["display_name"] = frame["task_label"].fillna(frame["world_id"]).astype(str)
    return frame[
        [
            "entity_type",
            "entity_id",
            "source_experiment_id",
            "family",
            "support_status",
            "support_weight",
            "display_name",
        ]
    ].copy()


def prepare_modeling(modeling: pd.DataFrame) -> pd.DataFrame:
    frame = modeling.copy()
    for column in [
        "canonical_policy_id",
        "canonical_goal_id",
        "world_id",
        "metric_family",
        "world_family",
        "goal_family",
        "perturbation_type",
        "goal_metric_family",
        "goal_target_direction",
        "substrate_kind",
        "policy_family",
        "policy_algorithm",
        "policy_representation_type",
        "source_experiment_id",
    ]:
        if column not in frame.columns:
            frame[column] = "__missing__"
        frame[column] = frame[column].fillna("__missing__").astype(str).replace("", "__missing__")
    frame["target_transformed"] = pd.to_numeric(frame["target_transformed"], errors="coerce")
    frame = frame[np.isfinite(frame["target_transformed"].to_numpy(dtype=float))].copy()
    frame["source_balance_weight"] = source_balance_weights(frame)
    return frame


def build_distance_branch(
    *,
    entity_type: str,
    distance_variant: str,
    baseline_family: str,
    ids: Sequence[str],
    coordinates: np.ndarray | pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    id_list = list(map(str, ids))
    if isinstance(coordinates, pd.DataFrame):
        coords = coordinates.reindex(id_list, fill_value=0.0).to_numpy(dtype=float)
    else:
        coords = np.asarray(coordinates, dtype=float)
    if len(id_list) < 2:
        return pd.DataFrame()
    neighbors = euclidean_neighbor_table(
        coords,
        id_list,
        k=NEIGHBOR_K,
        entity_type=entity_type,
        distance_variant=distance_variant,
        baseline_family=baseline_family,
    )
    return attach_neighbor_metadata(neighbors, metadata)


def build_all_neighbors(
    policies: pd.DataFrame,
    goals: pd.DataFrame,
    worlds: pd.DataFrame,
    modeling: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    policy_meta = policy_entities(policies)
    goal_meta = goal_entities(goals)
    world_meta = world_entities(worlds, modeling)
    metadata = pd.concat([policy_meta, goal_meta, world_meta], ignore_index=True)

    policy_ids = policies["canonical_policy_id"].astype(str).tolist()
    supported_policy_ids = policies.loc[policies["behavior_support_status"] == "behavior_profile_supported", "canonical_policy_id"].astype(str).tolist()
    goal_ids = goals["canonical_goal_id"].astype(str).tolist()
    supported_goal_ids = goals.loc[goals["goal_support_status"] == "supported_policy_evidence", "canonical_goal_id"].astype(str).tolist()
    world_ids = worlds["world_id"].astype(str).tolist()
    supported_world_ids = world_meta.loc[world_meta["support_status"] == "metric_profile_supported", "entity_id"].astype(str).tolist()

    policy_behavior = pd.DataFrame(numeric_matrix(policies, POLICY_BEHAVIOR_COLUMNS), index=policy_ids)
    policy_syntax = pd.DataFrame(numeric_matrix(policies, POLICY_SYNTAX_COLUMNS), index=policy_ids)
    goal_primary = pd.DataFrame(numeric_matrix(goals, GOAL_PRIMARY_COLUMNS), index=goal_ids)
    goal_supported = pd.DataFrame(numeric_matrix(goals, GOAL_SUPPORTED_COLUMNS), index=goal_ids)

    policy_metric, policy_metric_stats = weighted_mean_profile(
        modeling,
        entity_column="canonical_policy_id",
        entity_ids=policy_ids,
        feature_columns=POLICY_METRIC_FEATURES,
        value_column="target_transformed",
        weight_column="source_balance_weight",
    )
    goal_metric, goal_metric_stats = weighted_mean_profile(
        modeling,
        entity_column="canonical_goal_id",
        entity_ids=goal_ids,
        feature_columns=GOAL_METRIC_FEATURES,
        value_column="target_transformed",
        weight_column="source_balance_weight",
    )
    world_metric, world_metric_stats = weighted_mean_profile(
        modeling,
        entity_column="world_id",
        entity_ids=world_ids,
        feature_columns=WORLD_METRIC_FEATURES,
        value_column="target_transformed",
        weight_column="source_balance_weight",
    )
    policy_source, policy_source_stats = metadata_feature_matrix(
        policies,
        id_column="canonical_policy_id",
        entity_ids=policy_ids,
        categorical_columns=(
            "source_experiment_id",
            "policy_family",
            "algorithm",
            "representation_type",
            "abstraction_kind",
            "metadata_only",
            "requires_memory",
            "requires_signaling",
            "uses_global_oracle",
        ),
        numeric_columns=("behavior_observation_count",),
    )
    goal_source, goal_source_stats = metadata_feature_matrix(
        goals,
        id_column="canonical_goal_id",
        entity_ids=goal_ids,
        categorical_columns=(
            "source_experiment_id",
            "goal_family",
            "goal_kind",
            "target_direction",
            "metric_family",
            "conflict_group_id",
            "goal_support_status",
            "partial",
        ),
        numeric_columns=("goal_behavior_row_count", "supported_policy_row_count"),
    )
    world_source, world_source_stats = metadata_feature_matrix(
        worlds.rename(columns={"experiment_id": "source_experiment_id"}),
        id_column="world_id",
        entity_ids=world_ids,
        categorical_columns=(
            "source_experiment_id",
            "world_family",
            "substrate_kind",
            "record_granularity",
            "completeness",
        ),
        numeric_columns=(),
    )

    branches = [
        ("policy", "platonic_uncertainty_weighted", "platonic", policy_ids, policy_behavior, policy_meta),
        ("policy", "platonic_supported_only", "platonic", supported_policy_ids, policy_behavior, policy_meta),
        ("policy", "syntax_baseline", "syntax", policy_ids, policy_syntax, policy_meta),
        ("policy", "syntax_supported_only_baseline", "syntax", supported_policy_ids, policy_syntax, policy_meta),
        ("policy", "source_metadata_baseline", "source_metadata", policy_ids, policy_source, policy_meta),
        ("policy", "metric_only_baseline", "metric_only", policy_ids, policy_metric, policy_meta),
        ("policy", "metric_only_supported_only_baseline", "metric_only", supported_policy_ids, policy_metric, policy_meta),
        ("goal", "platonic_uncertainty_weighted", "platonic", goal_ids, goal_primary, goal_meta),
        ("goal", "platonic_supported_only", "platonic", supported_goal_ids, goal_supported, goal_meta),
        ("goal", "source_metadata_baseline", "source_metadata", goal_ids, goal_source, goal_meta),
        ("goal", "metric_only_baseline", "metric_only", goal_ids, goal_metric, goal_meta),
        ("goal", "metric_only_supported_only_baseline", "metric_only", supported_goal_ids, goal_metric, goal_meta),
        ("world", "platonic_metric_profile", "platonic", world_ids, world_metric, world_meta),
        ("world", "platonic_supported_only", "platonic", supported_world_ids, world_metric, world_meta),
        ("world", "source_metadata_baseline", "source_metadata", world_ids, world_source, world_meta),
        ("world", "metric_only_baseline", "metric_only", world_ids, world_metric, world_meta),
    ]
    neighbor_tables = [
        build_distance_branch(
            entity_type=entity_type,
            distance_variant=variant,
            baseline_family=baseline,
            ids=ids,
            coordinates=coords,
            metadata=meta,
        )
        for entity_type, variant, baseline, ids, coords, meta in branches
    ]
    neighbors = pd.concat([table for table in neighbor_tables if not table.empty], ignore_index=True)
    benchmarks = summarize_with_counts(neighbors, metadata)
    profile_stats = {
        "policyMetricProfile": policy_metric_stats,
        "goalMetricProfile": goal_metric_stats,
        "worldMetricProfile": world_metric_stats,
        "policySourceMetadataProfile": policy_source_stats,
        "goalSourceMetadataProfile": goal_source_stats,
        "worldSourceMetadataProfile": world_source_stats,
        "policySupportedEntityCount": len(supported_policy_ids),
        "goalSupportedEntityCount": len(supported_goal_ids),
        "worldSupportedEntityCount": len(supported_world_ids),
    }
    distance_matrices = {
        "goal_primary": goal_primary,
        "goal_supported": goal_supported,
        "goal_metric": goal_metric,
        "goal_source": goal_source,
        "policy_behavior": policy_behavior,
        "policy_syntax": policy_syntax,
        "policy_metric": policy_metric,
        "world_metric": world_metric,
    }
    return neighbors, benchmarks, metadata, {**profile_stats, "distanceMatrices": distance_matrices}


def summarize_with_counts(neighbors: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    from src.e07.distance_schema import summarize_neighbors

    summary = summarize_neighbors(neighbors, k=NEIGHBOR_K)
    if summary.empty:
        return summary
    status_counts = (
        metadata.groupby(["entity_type", "support_status"], observed=True)
        .size()
        .rename("entity_count")
        .reset_index()
        .sort_values(["entity_type", "support_status"], kind="mergesort")
    )
    status_json = {
        entity_type: group[["support_status", "entity_count"]].to_dict(orient="records")
        for entity_type, group in status_counts.groupby("entity_type", observed=True)
    }
    summary["entity_support_status_counts_json"] = summary["entity_type"].map(lambda value: json.dumps(status_json.get(value, []), sort_keys=True))
    summary["neighbor_k"] = NEIGHBOR_K
    summary["schema_version"] = PLATONIC_DISTANCE_SCHEMA_VERSION
    summary["research_step_id"] = STEP_ID
    return summary


def source_missingness_audit(benchmarks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in benchmarks.to_dict(orient="records"):
        same_source = float(record.get("same_source_rate_at_k", np.nan))
        low_support = float(record.get("low_support_neighbor_rate_at_k", np.nan))
        baseline_family = str(record.get("baseline_family", ""))
        variant = str(record.get("distance_variant", ""))
        entity_type = str(record.get("entity_type", ""))
        evaluated = baseline_family == "platonic"
        low_threshold = SUPPORTED_LOW_SUPPORT_THRESHOLD if "supported_only" in variant else UNCERTAINTY_LOW_SUPPORT_THRESHOLD
        source_ok = (not evaluated) or (np.isfinite(same_source) and same_source <= SOURCE_DOMINANCE_THRESHOLD)
        missingness_ok = (not evaluated) or (np.isfinite(low_support) and low_support <= low_threshold)
        rows.append(
            {
                "entity_type": entity_type,
                "distance_variant": variant,
                "baseline_family": baseline_family,
                "audit_case": "source_and_missingness_dominance",
                "evaluated_for_distance_pass": bool(evaluated),
                "same_source_rate_at_k": same_source,
                "same_source_threshold": SOURCE_DOMINANCE_THRESHOLD,
                "low_support_neighbor_rate_at_k": low_support,
                "low_support_threshold": low_threshold,
                "success": bool(source_ok and missingness_ok),
                "detail": (
                    "Pass/fail applies to Platonic distance branches only; baselines are retained as references."
                    if not evaluated
                    else f"source_ok={source_ok}; missingness_ok={missingness_ok}"
                ),
            }
        )
    return pd.DataFrame(rows)


def conflict_validation(
    goals: pd.DataFrame,
    conflicts: pd.DataFrame,
    matrices: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_table = unique_conflict_pairs(conflicts)
    supported_ids = set(goals.loc[goals["goal_support_status"] == "supported_policy_evidence", "canonical_goal_id"].astype(str))
    all_ids = set(goals["canonical_goal_id"].astype(str))
    branches = [
        ("platonic_uncertainty_weighted", "all_goals", matrices["goal_primary"], all_ids),
        ("platonic_uncertainty_weighted", "supported_only", matrices["goal_primary"], supported_ids),
        ("platonic_supported_only", "supported_only", matrices["goal_supported"], supported_ids),
        ("source_metadata_baseline", "all_goals", matrices["goal_source"], all_ids),
        ("source_metadata_baseline", "supported_only", matrices["goal_source"], supported_ids),
        ("metric_only_baseline", "all_goals", matrices["goal_metric"], all_ids),
        ("metric_only_supported_only_baseline", "supported_only", matrices["goal_metric"], supported_ids),
    ]
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant, eligible_mode, matrix, eligible_set in branches:
        available_ids = [goal_id for goal_id in matrix.index.astype(str).tolist() if goal_id in eligible_set]
        id_to_vec = {goal_id: matrix.loc[goal_id].to_numpy(dtype=float) for goal_id in available_ids}
        conflict_pairs = {
            tuple(sorted((str(row["goal_a"]), str(row["goal_b"]))))
            for row in pair_table.to_dict(orient="records")
            if str(row["goal_a"]) in id_to_vec and str(row["goal_b"]) in id_to_vec
        }
        for left, right in sorted(conflict_pairs):
            pair_rows.append(
                {
                    "distance_variant": variant,
                    "eligible_mode": eligible_mode,
                    "pair_type": "known_conflict",
                    "goal_a": left,
                    "goal_b": right,
                    "distance": float(np.linalg.norm(id_to_vec[left] - id_to_vec[right])),
                }
            )
        for left, right in combinations(sorted(id_to_vec), 2):
            key = tuple(sorted((left, right)))
            if key in conflict_pairs:
                continue
            pair_rows.append(
                {
                    "distance_variant": variant,
                    "eligible_mode": eligible_mode,
                    "pair_type": "non_conflict_reference",
                    "goal_a": left,
                    "goal_b": right,
                    "distance": float(np.linalg.norm(id_to_vec[left] - id_to_vec[right])),
                }
            )
        branch_pairs = pd.DataFrame([row for row in pair_rows if row["distance_variant"] == variant and row["eligible_mode"] == eligible_mode])
        conflict_dist = branch_pairs[branch_pairs["pair_type"] == "known_conflict"]["distance"] if not branch_pairs.empty else pd.Series(dtype=float)
        reference_dist = branch_pairs[branch_pairs["pair_type"] == "non_conflict_reference"]["distance"] if not branch_pairs.empty else pd.Series(dtype=float)
        conflict_median = float(conflict_dist.median()) if not conflict_dist.empty else float("nan")
        reference_median = float(reference_dist.median()) if not reference_dist.empty else float("nan")
        lift = conflict_median - reference_median if np.isfinite(conflict_median) and np.isfinite(reference_median) else float("nan")
        summary_rows.append(
            {
                "distance_variant": variant,
                "eligible_mode": eligible_mode,
                "eligible_goal_count": int(len(id_to_vec)),
                "known_conflict_pair_count": int(len(conflict_dist)),
                "reference_non_conflict_pair_count": int(len(reference_dist)),
                "known_conflict_distance_median": conflict_median,
                "reference_non_conflict_distance_median": reference_median,
                "known_conflict_distance_lift_over_reference_median": lift,
                "success": bool(np.isfinite(lift) and lift > 0.0),
                "detail": "positive lift means known conflict/opposite goals are farther apart than non-conflict references",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def robustness_metrics(matrices: Mapping[str, pd.DataFrame], policies: pd.DataFrame, goals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    supported_policy_ids = policies.loc[policies["behavior_support_status"] == "behavior_profile_supported", "canonical_policy_id"].astype(str).tolist()
    supported_goal_ids = goals.loc[goals["goal_support_status"] == "supported_policy_evidence", "canonical_goal_id"].astype(str).tolist()
    if len(supported_policy_ids) >= 3:
        behavior = matrices["policy_behavior"].loc[supported_policy_ids].to_numpy(dtype=float)
        syntax = matrices["policy_syntax"].loc[supported_policy_ids].to_numpy(dtype=float)
        metric = matrices["policy_metric"].loc[supported_policy_ids].to_numpy(dtype=float)
        rows.extend(
            [
                {
                    "entity_type": "policy",
                    "comparison": "platonic_supported_only_vs_syntax_baseline",
                    "pairwise_distance_spearman": sampled_pairwise_distance_spearman(behavior, syntax, seed=RANDOM_SEED + 1),
                    "eligible_entity_count": len(supported_policy_ids),
                },
                {
                    "entity_type": "policy",
                    "comparison": "platonic_supported_only_vs_metric_only_baseline",
                    "pairwise_distance_spearman": sampled_pairwise_distance_spearman(behavior, metric, seed=RANDOM_SEED + 2),
                    "eligible_entity_count": len(supported_policy_ids),
                },
            ]
        )
    if len(supported_goal_ids) >= 3:
        primary = matrices["goal_primary"].loc[supported_goal_ids].to_numpy(dtype=float)
        supported = matrices["goal_supported"].loc[supported_goal_ids].to_numpy(dtype=float)
        metric = matrices["goal_metric"].loc[supported_goal_ids].to_numpy(dtype=float)
        rows.extend(
            [
                {
                    "entity_type": "goal",
                    "comparison": "uncertainty_weighted_vs_supported_only",
                    "pairwise_distance_spearman": sampled_pairwise_distance_spearman(primary, supported, seed=RANDOM_SEED + 3),
                    "eligible_entity_count": len(supported_goal_ids),
                },
                {
                    "entity_type": "goal",
                    "comparison": "platonic_supported_only_vs_metric_only_baseline",
                    "pairwise_distance_spearman": sampled_pairwise_distance_spearman(supported, metric, seed=RANDOM_SEED + 4),
                    "eligible_entity_count": len(supported_goal_ids),
                },
            ]
        )
    return pd.DataFrame(rows)


def plot_benchmarks(benchmarks: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    if benchmarks.empty:
        return
    plot_frame = benchmarks.copy()
    plot_frame["label"] = plot_frame["entity_type"] + "\n" + plot_frame["distance_variant"].str.replace("_", " ", regex=False)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].bar(range(len(plot_frame)), plot_frame["same_source_rate_at_k"], color="#49759c")
    axes[0].axhline(SOURCE_DOMINANCE_THRESHOLD, color="#b0443c", linestyle="--", linewidth=1)
    axes[0].set_ylabel("same source @k")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(range(len(plot_frame)), plot_frame["low_support_neighbor_rate_at_k"], color="#609c72")
    axes[1].axhline(UNCERTAINTY_LOW_SUPPORT_THRESHOLD, color="#b0443c", linestyle="--", linewidth=1)
    axes[1].set_ylabel("low-support neighbors @k")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(range(len(plot_frame)))
    axes[1].set_xticklabels(plot_frame["label"], rotation=72, ha="right", fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("E07 S08 distance benchmark audit")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)


def outcome_classification(validation: pd.DataFrame) -> str:
    return "supportive" if bool(validation["success"].all()) else "constraining/contradictory"


def build_distance_spec(spec_path: Path, validation: pd.DataFrame, outcome: str) -> None:
    validation_success = bool(validation["success"].all())
    text = f"""# E07 Platonic Distance Specification

## Top Summary

- Research step ID: S08
- Completion status: complete
- Artifacts written: `{spec_path}`, `$ARTIFACTS_DIR/results/e07_distance_benchmarks.parquet`, `$ARTIFACTS_DIR/results/e07_platonic_neighbors.parquet`
- Validation result: {"pass" if validation_success else "fail"} ({int(validation['success'].sum())}/{len(validation)} checks passed)
- Outcome classification: {outcome}
- Caveats or blockers: S08 carries forward S06 policy-support sparsity and S07 conflict-separation failure; the supported-only branch remains the least speculative branch.
- Recommended next action: Chief review before S09; use supported-only Platonic distances only as bounded proxy inputs unless conflict and missingness limitations are accepted.

## Definition

The S08 Platonic distance is a computational proxy distance, not a metaphysical or biological claim. For entities represented by an S06/S07 or S05-derived profile vector `z`, the base distance is Euclidean:

```text
d(i, j) = ||z_i - z_j||_2
```

Two Platonic branches are preserved:

- `platonic_supported_only`: filters to entities with direct behavior support, using S06 behavior-supported policies and S07 supported-policy-evidence goals.
- `platonic_uncertainty_weighted`: retains all represented policies or goals and carries support weights into audits and reliability summaries. It does not add support status as a coordinate, because that would make missingness dominance tautological.

World distances use source-balanced S05 metric profiles because no S06/S07 world embedding exists yet. They are included as provisional `platonic_metric_profile` distances.

## Baselines

S08 benchmarks the Platonic branches against:

- `syntax_baseline`: policy DSL/source syntax coordinates from S06.
- `source_metadata_baseline`: one-hot source/family/representation metadata.
- `metric_only_baseline`: source-balanced S05 target profiles without S06/S07 embedding coordinates.

## Validation Rule

The distance is treated as constrained if either:

- proposed Platonic nearest neighbors are dominated by source identity or low-support/missing entities; or
- known opposite/conflicting S03 goal pairs are not farther apart than non-conflict reference goal pairs in the supported-only branch.

Both conditions are stress tests, not tuning targets.
"""
    write_text(spec_path, text)


def build_report(
    *,
    full_report_path: Path,
    artifacts_written: Sequence[str],
    benchmarks: pd.DataFrame,
    source_audit: pd.DataFrame,
    conflict_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    validation: pd.DataFrame,
    outcome: str,
    profile_stats: Mapping[str, Any],
    unit_test_result: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    validation_success = bool(validation["success"].all())
    caveats = (
        "S08 distances are bounded proxy distances. S06 behavior support is sparse, S07 known-conflict separation previously failed, "
        "and S08 treats continued missingness/source dominance or conflict-separation failure as constraining."
    )
    test_line = "not run"
    if unit_test_result is not None:
        test_line = f"{'pass' if unit_test_result['success'] else 'fail'}: `{unit_test_result['command']}` return code {unit_test_result['returnCode']}"
    text = f"""# E07 S08 Full Results: Define Platonic Distance

## Top Summary

- Research step ID: S08
- Completion status: complete
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: {"pass" if validation_success else "fail"} ({int(validation['success'].sum())}/{len(validation)} checks passed)
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Lay summary: S08 defined practical distance tables for policies, goals, and worlds, then stress-tested whether the nearest neighbors reflect behavior rather than just source identity, missingness, or metric availability. The supported-only filters and support-weighted audits are both present; the result remains constrained because the available evidence still fails at least one required stress test.
- Recommended next action: Chief review before S09. Use `platonic_supported_only` distances as cautious inputs and carry the conflict/source/missingness caveats into invariant discovery if S09 proceeds.

## Frozen Question

Can distances between algorithms, goals, and worlds be defined by behavioral equivalence across substrates rather than syntactic similarity?

## Inputs

- S01 world inventory: `{args.s01_world_inventory}` (SHA-256 `{sha256_file(args.s01_world_inventory)}`)
- S04 unified behavior corpus: `{args.s04_corpus}` (SHA-256 `{sha256_file(args.s04_corpus)}`)
- S05 modeling dataset: `{args.s05_modeling_dataset}` (SHA-256 `{sha256_file(args.s05_modeling_dataset)}`)
- S05 metrics: `{args.s05_metrics}` (SHA-256 `{sha256_file(args.s05_metrics)}`)
- S05 calibration: `{args.s05_calibration}` (SHA-256 `{sha256_file(args.s05_calibration)}`)
- S06 policy embeddings: `{args.s06_policy_embeddings}` (SHA-256 `{sha256_file(args.s06_policy_embeddings)}`)
- S07 goal embeddings: `{args.s07_goal_embeddings}` (SHA-256 `{sha256_file(args.s07_goal_embeddings)}`)
- S03 goal conflicts: `{args.s03_goal_conflicts}` (SHA-256 `{sha256_file(args.s03_goal_conflicts)}`)

## Methods

S08 defined Euclidean distance over compact entity-profile vectors. Policies use S06 behavior coordinates for Platonic branches and S06 syntax coordinates for a syntax/source-code baseline. Goals use S07 primary uncertainty-weighted coordinates and S07 supported-only coordinates. Worlds use S05 source-balanced metric profiles because no world embedding has been trained yet.

Support handling is explicit. `platonic_supported_only` filters to behavior-supported S06 policies, S07 goals with supported-policy evidence, and worlds with at least {MIN_SUPPORTED_ROWS} S05 rows. `platonic_uncertainty_weighted` retains all represented policies or goals while recording support weights and low-support neighbor rates. S08 does not hide low-support rows; it audits whether they dominate nearest neighbors.

Metric-only baselines were built from the S05 oriented signed-log target using source-balanced weighted means by metric, world, goal, perturbation, substrate, and policy axes. Source metadata baselines were built from one-hot source/family/representation descriptors. Known-conflict validation used S03 conflict pairs and compared their median distance to all non-conflict references under all-goal and supported-only branches.

## Commands

- `python -m unittest tests.e07.test_distance_schema`
- `python scripts/e07_s08_platonic_distance.py`

Unit-test result: {test_line}

## Dependencies and Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- numpy: {np.__version__}
- matplotlib: {matplotlib.__version__}
- Worker count: single-process CPU analysis; no GPU training in S08.
- Repository commit before S08 commit: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Branch: `{git_output(args.repo_dir, ['branch', '--show-current'])}`

## Results

### Distance Benchmarks

{markdown_table(benchmarks, max_rows=40)}

### Source and Missingness Audit

{markdown_table(source_audit, max_rows=50)}

### Conflict-Aware Validation

{markdown_table(conflict_summary, max_rows=40)}

### Distance Robustness and Baseline Correlation

{markdown_table(robustness, max_rows=20)}

### Profile Statistics

```json
{json.dumps({key: value for key, value in profile_stats.items() if key != 'distanceMatrices'}, indent=2, sort_keys=True, default=str)}
```

## Validation

{markdown_table(validation)}

## Output Artifacts

{chr(10).join(f'- `{item}`' for item in artifacts_written)}

## Caveats, Blockers, and Limitations

- "Platonic distance" is a modeling label for cross-world behavioral proxy distances, not a metaphysical claim.
- The S05 target is a heterogeneous oriented signed-log metric proxy; source-specific metric coverage remains uneven.
- S06 policy support is sparse: only the supported-only branch should be used for strong policy-neighbor claims.
- S07 conflict separation failed previously; S08 preserves that branch as a hard stress test rather than tuning it away.
- World distances are provisional metric-profile distances because world embeddings are not yet available.
- Source metadata and metric-only baselines can be informative but should not be mistaken for behavioral equivalence.

## Recommended Next Action

Stop before S09 for Chief review. If S09 proceeds, use the S08 supported-only distance artifacts as cautious inputs and explicitly propagate the constraining conflict/source/missingness flags.
"""
    write_text(full_report_path, text)


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    reports_dir = artifacts_dir / "reports"
    figures_dir = artifacts_dir / "figures" / "e07"
    for directory in (step_dir, results_dir, tables_dir, reports_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_test_result = None
    if args.run_unit_tests:
        unit_test_result = run_command([sys.executable, "-m", "unittest", "tests.e07.test_distance_schema"], args.repo_dir)

    policies = pd.read_parquet(args.s06_policy_embeddings)
    goals = pd.read_parquet(args.s07_goal_embeddings)
    worlds = pd.read_csv(args.s01_world_inventory)
    modeling = prepare_modeling(pd.read_parquet(args.s05_modeling_dataset))
    conflicts = pd.read_parquet(args.s03_goal_conflicts)

    neighbors, benchmarks, metadata, profile_stats = build_all_neighbors(policies, goals, worlds, modeling)
    source_audit = source_missingness_audit(benchmarks)
    matrices = profile_stats["distanceMatrices"]
    conflict_summary, conflict_pairs = conflict_validation(goals, conflicts, matrices)
    robustness = robustness_metrics(matrices, policies, goals)
    validation = validate_platonic_distance_artifacts(
        benchmarks,
        neighbors,
        source_audit,
        conflict_summary,
        required_variants=[
            "policy:platonic_uncertainty_weighted",
            "policy:platonic_supported_only",
            "policy:syntax_baseline",
            "policy:source_metadata_baseline",
            "policy:metric_only_baseline",
            "goal:platonic_uncertainty_weighted",
            "goal:platonic_supported_only",
            "goal:source_metadata_baseline",
            "goal:metric_only_baseline",
            "world:platonic_metric_profile",
            "world:source_metadata_baseline",
            "world:metric_only_baseline",
        ],
    )
    if unit_test_result is not None and not bool(unit_test_result["success"]):
        validation = pd.concat(
            [
                validation,
                pd.DataFrame(
                    [
                        {
                            "validation_case": "unit_tests_passed",
                            "success": False,
                            "detail": f"`{unit_test_result['command']}` failed with return code {unit_test_result['returnCode']}",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        validation = pd.concat(
            [
                validation,
                pd.DataFrame(
                    [
                        {
                            "validation_case": "unit_tests_passed",
                            "success": True,
                            "detail": f"`{unit_test_result['command']}` passed" if unit_test_result else "unit tests not requested",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    outcome = outcome_classification(validation)

    benchmarks_path = results_dir / "e07_distance_benchmarks.parquet"
    benchmarks_csv_path = tables_dir / "e07_distance_benchmarks.csv"
    neighbors_path = results_dir / "e07_platonic_neighbors.parquet"
    neighbors_sample_path = tables_dir / "e07_platonic_neighbors_sample.csv"
    source_audit_path = tables_dir / "e07_distance_source_missingness_audit.csv"
    conflict_summary_path = tables_dir / "e07_distance_conflict_validation.csv"
    conflict_pairs_path = results_dir / "e07_distance_conflict_pairs.parquet"
    robustness_path = tables_dir / "e07_distance_robustness.csv"
    metadata_path = results_dir / "e07_distance_entity_metadata.parquet"
    spec_path = reports_dir / "e07_platonic_distance_spec.md"
    figure_path = figures_dir / "platonic_distance_benchmark.png"
    validation_path = step_dir / "e07_s08_validation_checks.csv"
    config_path = step_dir / "s08_config.json"
    full_report_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"

    benchmarks.to_parquet(benchmarks_path, index=False)
    benchmarks.to_csv(benchmarks_csv_path, index=False)
    neighbors.to_parquet(neighbors_path, index=False)
    neighbors.head(1000).to_csv(neighbors_sample_path, index=False)
    source_audit.to_csv(source_audit_path, index=False)
    conflict_summary.to_csv(conflict_summary_path, index=False)
    conflict_pairs.to_parquet(conflict_pairs_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    metadata.to_parquet(metadata_path, index=False)
    validation.to_csv(validation_path, index=False)
    plot_benchmarks(benchmarks, figure_path)
    build_distance_spec(spec_path, validation, outcome)
    write_json(
        config_path,
        {
            "schemaVersion": PLATONIC_DISTANCE_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "randomSeed": RANDOM_SEED,
            "neighborK": NEIGHBOR_K,
            "embeddingDims": EMBEDDING_DIMS,
            "minSupportedRows": MIN_SUPPORTED_ROWS,
            "sourceDominanceThreshold": SOURCE_DOMINANCE_THRESHOLD,
            "uncertaintyLowSupportThreshold": UNCERTAINTY_LOW_SUPPORT_THRESHOLD,
            "supportedLowSupportThreshold": SUPPORTED_LOW_SUPPORT_THRESHOLD,
            "policyMetricFeatures": list(POLICY_METRIC_FEATURES),
            "goalMetricFeatures": list(GOAL_METRIC_FEATURES),
            "worldMetricFeatures": list(WORLD_METRIC_FEATURES),
            "profileStats": {key: value for key, value in profile_stats.items() if key != "distanceMatrices"},
        },
    )

    artifacts_written = [
        str(full_report_path),
        str(spec_path),
        str(benchmarks_path),
        str(benchmarks_csv_path),
        str(neighbors_path),
        str(neighbors_sample_path),
        str(source_audit_path),
        str(conflict_summary_path),
        str(conflict_pairs_path),
        str(robustness_path),
        str(metadata_path),
        str(figure_path),
        str(validation_path),
        str(config_path),
        str(artifact_manifest_path),
    ]
    build_report(
        full_report_path=full_report_path,
        artifacts_written=artifacts_written,
        benchmarks=benchmarks,
        source_audit=source_audit,
        conflict_summary=conflict_summary,
        robustness=robustness,
        validation=validation,
        outcome=outcome,
        profile_stats=profile_stats,
        unit_test_result=unit_test_result,
        args=args,
    )

    manifest_payload = {
        "schemaVersion": "eidosoma.e07.s08.artifact_manifest.v1",
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
            "s01WorldInventory": {"path": str(args.s01_world_inventory), "sha256": sha256_file(args.s01_world_inventory)},
            "s04Corpus": {"path": str(args.s04_corpus), "sha256": sha256_file(args.s04_corpus)},
            "s05ModelingDataset": {"path": str(args.s05_modeling_dataset), "sha256": sha256_file(args.s05_modeling_dataset)},
            "s05Metrics": {"path": str(args.s05_metrics), "sha256": sha256_file(args.s05_metrics)},
            "s05Calibration": {"path": str(args.s05_calibration), "sha256": sha256_file(args.s05_calibration)},
            "s06PolicyEmbeddings": {"path": str(args.s06_policy_embeddings), "sha256": sha256_file(args.s06_policy_embeddings)},
            "s07GoalEmbeddings": {"path": str(args.s07_goal_embeddings), "sha256": sha256_file(args.s07_goal_embeddings)},
            "s03GoalConflicts": {"path": str(args.s03_goal_conflicts), "sha256": sha256_file(args.s03_goal_conflicts)},
        },
        "coverage": {
            "policyRows": int(len(policies)),
            "goalRows": int(len(goals)),
            "worldRows": int(len(worlds)),
            "distanceBenchmarkRows": int(len(benchmarks)),
            "neighborRows": int(len(neighbors)),
            "conflictValidationRows": int(len(conflict_summary)),
        },
        "artifacts": [
            self_referential_artifact_entry(full_report_path, artifacts_dir, "S08 full-results report."),
            artifact_entry(spec_path, artifacts_dir, "Platonic distance specification."),
            artifact_entry(benchmarks_path, artifacts_dir, "Distance benchmark summary table."),
            artifact_entry(benchmarks_csv_path, artifacts_dir, "Distance benchmark summary CSV."),
            artifact_entry(neighbors_path, artifacts_dir, "Top-k Platonic and baseline neighbor table."),
            artifact_entry(neighbors_sample_path, artifacts_dir, "Small CSV sample of neighbor rows."),
            artifact_entry(source_audit_path, artifacts_dir, "Source and missingness dominance audit."),
            artifact_entry(conflict_summary_path, artifacts_dir, "Conflict-aware distance validation summary."),
            artifact_entry(conflict_pairs_path, artifacts_dir, "Pairwise goal distances for conflict validation."),
            artifact_entry(robustness_path, artifacts_dir, "Distance robustness and baseline correlation summary."),
            artifact_entry(metadata_path, artifacts_dir, "Entity metadata used to label distance neighbors."),
            artifact_entry(figure_path, artifacts_dir, "Distance benchmark audit figure."),
            artifact_entry(validation_path, artifacts_dir, "S08 validation checks."),
            artifact_entry(config_path, artifacts_dir, "S08 distance configuration."),
            self_referential_artifact_entry(artifact_manifest_path, artifacts_dir, "S08 artifact manifest."),
        ],
    }
    write_json(artifact_manifest_path, manifest_payload)
    print(f"[S08] wrote {full_report_path}")
    print(f"[S08] validation {int(validation['success'].sum())}/{len(validation)} outcome={outcome}")


if __name__ == "__main__":
    main()
