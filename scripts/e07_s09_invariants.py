#!/usr/bin/env python3
"""Search for E07 S09 candidate invariants with held-out checks."""

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
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import SGDRegressor  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.invariant_schema import (  # noqa: E402
    INVARIANT_SCHEMA_VERSION,
    deterministic_cap,
    extract_policy_invariants,
    extract_world_invariants,
    residualize_by_group_median,
    source_balance_weights,
    stable_fraction,
    validate_invariant_artifacts,
    weighted_correlation,
    weighted_mean,
)
from src.e07.predictor_schema import assign_group_holdout, metrics_for_predictions, transformed_target  # noqa: E402


STEP_ID = "S09"
STEP_NUMBER = 9
EXPERIMENT_ID = "E07"
RANDOM_SEED = 20260703
MAX_SOURCE_DOMINANCE_RATE = 0.85
MIN_ABS_EFFECT = 0.03
MIN_ACTIVE_ROWS = 200
MIN_SUPPORT_WEIGHT = 0.20
MAX_ROWS_PER_SOURCE_METRIC = 1500
MAX_LINEAR_TRAIN_ROWS = 60_000
MAX_LINEAR_TEST_ROWS = 60_000
N_JOBS = min(8, os.cpu_count() or 1)

CORPUS_COLUMNS = (
    "corpus_row_id",
    "source_experiment_id",
    "source_table",
    "source_metric_name",
    "metric_value",
    "metric_unit",
    "metric_direction",
    "metric_family",
    "world_id",
    "world_link_status",
    "canonical_policy_id",
    "policy_uid",
    "policy_link_status",
    "canonical_goal_id",
    "goal_uid",
    "goal_link_status",
    "perturbation_type",
    "evidence_kind",
)

POLICY_INVARIANT_FEATURES = (
    "policy_requires_memory",
    "policy_requires_signaling",
    "policy_uses_global_oracle",
    "policy_target_position_knowledge",
    "policy_stochastic_choice",
    "policy_feedback_or_adaptation",
    "policy_locality_horizon_score",
    "policy_memory_depth_proxy",
    "policy_feedback_strength_proxy",
    "policy_rule_count",
    "policy_condition_count",
    "policy_action_count",
    "policy_has_swap_action",
    "policy_has_compare_action",
    "policy_has_wait_action",
    "policy_has_signal_action",
    "policy_has_remember_action",
    "policy_interpretable",
)

WORLD_INVARIANT_FEATURES = (
    "world_dimension_proxy",
    "world_has_2d_or_graph_substrate",
    "world_has_frozen_perturbation",
    "world_frozen_count",
    "world_has_chimera_or_mixed_identity",
    "world_has_damage_or_repair",
    "world_has_homeostasis",
    "world_has_governance_or_dominance",
    "world_has_target_morphology",
    "world_has_null_or_control",
    "world_scheduler_cyclic",
    "world_scheduler_uniform_or_random",
    "world_timeout_guard",
    "world_nonlocal_baseline_context",
    "world_local_observation_context",
    "world_observation_richness_proxy",
    "world_action_count",
    "world_measurement_count",
)

MISSINGNESS_CONTROL_FEATURES = (
    "has_world_link",
    "has_policy_link",
    "has_goal_link",
    "policy_metadata_only",
    "goal_partial",
)

S08_PROXY_FEATURES = (
    "policy_s08_supported_available",
    "policy_s08_support_weight",
    "policy_s08_top1_distance",
    "policy_s08_mean_neighbor_distance",
    "policy_s08_neighbor_same_source_rate",
    "goal_s08_supported_available",
    "goal_s08_support_weight",
    "goal_s08_top1_distance",
    "goal_s08_mean_neighbor_distance",
    "goal_s08_neighbor_same_source_rate",
    "world_s08_supported_available",
    "world_s08_support_weight",
    "world_s08_top1_distance",
    "world_s08_mean_neighbor_distance",
    "world_s08_neighbor_same_source_rate",
)

TASK_CONTEXT_COLUMNS = (
    "metric_family",
    "metric_direction",
    "goal_family",
    "goal_target_direction",
    "goal_conflict_group_id",
    "perturbation_type",
)

SOURCE_CONTROL_COLUMNS = (
    "source_experiment_id",
    "source_table",
    "source_metric_name",
    "policy_source_experiment_id",
    "goal_source_experiment_id",
    "world_family",
    "policy_family",
    "goal_family",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--s04-corpus", type=Path, default=artifacts_dir / "results" / "e07_unified_behavior_corpus.parquet")
    parser.add_argument("--s01-world-inventory", type=Path, default=artifacts_dir / "tables" / "e07_world_inventory.csv")
    parser.add_argument("--s02-policy-table", type=Path, default=artifacts_dir / "tables" / "e07_policy_representations.parquet")
    parser.add_argument("--s03-goal-table", type=Path, default=artifacts_dir / "tables" / "e07_goal_representations.parquet")
    parser.add_argument("--s08-distance-benchmarks", type=Path, default=artifacts_dir / "results" / "e07_distance_benchmarks.parquet")
    parser.add_argument("--s08-neighbors", type=Path, default=artifacts_dir / "results" / "e07_platonic_neighbors.parquet")
    parser.add_argument("--s08-entity-metadata", type=Path, default=artifacts_dir / "results" / "e07_distance_entity_metadata.parquet")
    parser.add_argument("--s08-validation", type=Path, default=artifacts_dir / "research_steps" / "S08" / "e07_s08_validation_checks.csv")
    parser.add_argument("--s08-conflict-validation", type=Path, default=artifacts_dir / "tables" / "e07_distance_conflict_validation.csv")
    parser.add_argument("--max-rows-per-source-metric", type=int, default=MAX_ROWS_PER_SOURCE_METRIC)
    parser.add_argument("--max-linear-train-rows", type=int, default=MAX_LINEAR_TRAIN_ROWS)
    parser.add_argument("--max-linear-test-rows", type=int, default=MAX_LINEAR_TEST_ROWS)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


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


def deterministic_sample_by_source_metric(corpus: pd.DataFrame, max_per_stratum: int) -> pd.DataFrame:
    frame = corpus.copy()
    frame["_sample_hash"] = pd.util.hash_pandas_object(frame["corpus_row_id"].astype(str), index=False).astype("uint64")
    frame["_stratum"] = frame["source_experiment_id"].astype(str) + "|" + frame["source_table"].astype(str) + "|" + frame["source_metric_name"].astype(str)
    return (
        frame.sort_values(["_stratum", "_sample_hash"], kind="mergesort")
        .groupby("_stratum", sort=False, group_keys=False)
        .head(max_per_stratum)
        .drop(columns=["_sample_hash", "_stratum"])
        .reset_index(drop=True)
    )


def build_policy_feature_table(path: Path) -> pd.DataFrame:
    policy = pd.read_parquet(path).drop_duplicates("canonical_policy_id").copy()
    extracted = pd.DataFrame([extract_policy_invariants(row) for row in policy.to_dict(orient="records")])
    frame = pd.concat([policy.reset_index(drop=True), extracted], axis=1)
    keep = [
        "canonical_policy_id",
        "source_experiment_id",
        "policy_family",
        "algorithm",
        "representation_type",
        "abstraction_kind",
        "metadata_only",
        *POLICY_INVARIANT_FEATURES,
    ]
    frame = frame[[column for column in keep if column in frame.columns]]
    frame = frame.rename(
        columns={
            "source_experiment_id": "policy_source_experiment_id",
            "algorithm": "policy_algorithm",
            "representation_type": "policy_representation_type",
            "abstraction_kind": "policy_abstraction_kind",
            "metadata_only": "policy_metadata_only",
        }
    )
    return frame


def build_world_feature_table(path: Path) -> pd.DataFrame:
    world = pd.read_csv(path).drop_duplicates("world_id").copy()
    extracted = pd.DataFrame([extract_world_invariants(row) for row in world.to_dict(orient="records")])
    frame = pd.concat([world.reset_index(drop=True), extracted], axis=1)
    keep = [
        "world_id",
        "experiment_id",
        "world_family",
        "substrate_kind",
        "record_granularity",
        "completeness",
        *WORLD_INVARIANT_FEATURES,
    ]
    frame = frame[[column for column in keep if column in frame.columns]]
    frame = frame.rename(
        columns={
            "experiment_id": "world_source_experiment_id",
            "record_granularity": "world_record_granularity",
            "completeness": "world_completeness",
        }
    )
    return frame


def build_goal_table(path: Path) -> pd.DataFrame:
    goal = pd.read_parquet(path).drop_duplicates("canonical_goal_id").copy()
    keep = [
        "canonical_goal_id",
        "source_experiment_id",
        "goal_family",
        "goal_kind",
        "abstraction_kind",
        "target_direction",
        "unit",
        "metric_family",
        "partial",
        "conflict_group_id",
    ]
    goal = goal[[column for column in keep if column in goal.columns]]
    goal = goal.rename(
        columns={
            "source_experiment_id": "goal_source_experiment_id",
            "abstraction_kind": "goal_abstraction_kind",
            "target_direction": "goal_target_direction",
            "unit": "goal_unit",
            "metric_family": "goal_metric_family",
            "partial": "goal_partial",
            "conflict_group_id": "goal_conflict_group_id",
        }
    )
    return goal


def s08_neighbor_summary(neighbors: pd.DataFrame, entity_type: str, variant: str, prefix: str) -> pd.DataFrame:
    subset = neighbors[(neighbors["entity_type"] == entity_type) & (neighbors["distance_variant"] == variant)].copy()
    if subset.empty:
        return pd.DataFrame(columns=[f"{prefix}_id"])
    top1 = subset[subset["neighbor_rank"] == 1].set_index("query_entity_id")["distance"].rename(f"{prefix}_top1_distance")
    grouped = subset.groupby("query_entity_id", observed=True).agg(
        mean_neighbor_distance=("distance", "mean"),
        neighbor_same_source_rate=("same_source_experiment", "mean"),
        neighbor_low_support_rate=("low_support_pair", "mean"),
        mean_pair_support_weight=("pair_support_weight", "mean"),
    )
    grouped = grouped.rename(columns={column: f"{prefix}_{column}" for column in grouped.columns})
    out = grouped.join(top1).reset_index().rename(columns={"query_entity_id": f"{prefix}_id"})
    out[f"{prefix}_supported_available"] = 1.0
    return out


def add_s08_controls(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    s08_metadata = pd.read_parquet(args.s08_entity_metadata)
    neighbors = pd.read_parquet(args.s08_neighbors)
    validation = pd.read_csv(args.s08_validation)
    conflicts = pd.read_csv(args.s08_conflict_validation)

    for entity_type, id_column, prefix in [
        ("policy", "canonical_policy_id", "policy_s08"),
        ("goal", "canonical_goal_id", "goal_s08"),
        ("world", "world_id", "world_s08"),
    ]:
        meta = s08_metadata[s08_metadata["entity_type"] == entity_type][["entity_id", "support_status", "support_weight"]].copy()
        meta = meta.rename(
            columns={
                "entity_id": id_column,
                "support_status": f"{prefix}_support_status",
                "support_weight": f"{prefix}_support_weight",
            }
        )
        frame = frame.merge(meta, on=id_column, how="left")
    policy_neighbors = s08_neighbor_summary(neighbors, "policy", "platonic_supported_only", "policy_s08")
    goal_neighbors = s08_neighbor_summary(neighbors, "goal", "platonic_supported_only", "goal_s08")
    world_neighbors = s08_neighbor_summary(neighbors, "world", "platonic_supported_only", "world_s08")
    frame = frame.merge(policy_neighbors, left_on="canonical_policy_id", right_on="policy_s08_id", how="left").drop(columns=["policy_s08_id"], errors="ignore")
    frame = frame.merge(goal_neighbors, left_on="canonical_goal_id", right_on="goal_s08_id", how="left").drop(columns=["goal_s08_id"], errors="ignore")
    frame = frame.merge(world_neighbors, left_on="world_id", right_on="world_s08_id", how="left").drop(columns=["world_s08_id"], errors="ignore")
    for prefix in ("policy_s08", "goal_s08", "world_s08"):
        for suffix in (
            "supported_available",
            "support_weight",
            "top1_distance",
            "mean_neighbor_distance",
            "neighbor_same_source_rate",
            "neighbor_low_support_rate",
            "mean_pair_support_weight",
        ):
            column = f"{prefix}_{suffix}"
            if column not in frame.columns:
                frame[column] = 0.0
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        status_col = f"{prefix}_support_status"
        if status_col not in frame.columns:
            frame[status_col] = "__missing__"
        frame[status_col] = frame[status_col].fillna("__missing__").astype(str)

    failed = validation[~validation["success"].astype(bool)].copy()
    source_row = failed[failed["validation_case"] == "source_missingness_dominance_not_detected"]
    conflict_row = conflicts[(conflicts["distance_variant"] == "platonic_supported_only") & (conflicts["eligible_mode"] == "supported_only")]
    caveats = pd.DataFrame(
        [
            {
                "s08_caveat": "source_dominance_failure",
                "success": False,
                "detail": source_row["detail"].iloc[0] if not source_row.empty else "S08 source dominance failure not found",
                "metric_value": float("nan"),
            },
            {
                "s08_caveat": "goal_conflict_separation_failure",
                "success": False,
                "detail": "supported-only known goal conflicts were not farther than references",
                "metric_value": float(conflict_row["known_conflict_distance_lift_over_reference_median"].iloc[0]) if not conflict_row.empty else float("nan"),
            },
        ]
    )
    return frame, caveats


def load_analysis_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    corpus = pd.read_parquet(args.s04_corpus, columns=list(CORPUS_COLUMNS))
    corpus = corpus[corpus["evidence_kind"].astype(str) == "metric_observation"].copy()
    input_stats = {
        "s04Rows": int(len(corpus)),
        "s04Sha256": sha256_file(args.s04_corpus),
        "rowsBySourceExperiment": corpus.groupby("source_experiment_id").size().astype(int).to_dict(),
        "sourceMetricCount": int(corpus["source_metric_name"].nunique()),
        "policyGroupCount": int(corpus.loc[corpus["canonical_policy_id"].fillna("").astype(str) != "", "canonical_policy_id"].nunique()),
        "worldGroupCount": int(corpus.loc[corpus["world_id"].fillna("").astype(str) != "", "world_id"].nunique()),
        "goalGroupCount": int(corpus.loc[corpus["canonical_goal_id"].fillna("").astype(str) != "", "canonical_goal_id"].nunique()),
    }
    sampled = deterministic_sample_by_source_metric(corpus, args.max_rows_per_source_metric)
    input_stats["modelingRows"] = int(len(sampled))
    input_stats["modelingRowsBySourceExperiment"] = sampled.groupby("source_experiment_id").size().astype(int).to_dict()

    policy = build_policy_feature_table(args.s02_policy_table)
    world = build_world_feature_table(args.s01_world_inventory)
    goal = build_goal_table(args.s03_goal_table)
    frame = sampled.merge(world, on="world_id", how="left").merge(policy, on="canonical_policy_id", how="left").merge(goal, on="canonical_goal_id", how="left")
    frame["target_transformed"] = [
        transformed_target(value, direction)
        for value, direction in zip(frame["metric_value"], frame["metric_direction"], strict=False)
    ]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_transformed"]).reset_index(drop=True)
    for id_column in ("canonical_policy_id", "world_id", "canonical_goal_id"):
        frame[id_column] = frame[id_column].fillna("__missing__").astype(str).replace("", "__missing__")
    frame["has_world_link"] = (frame["world_id"] != "__missing__").astype(float)
    frame["has_policy_link"] = (frame["canonical_policy_id"] != "__missing__").astype(float)
    frame["has_goal_link"] = (frame["canonical_goal_id"] != "__missing__").astype(float)
    for column in [*POLICY_INVARIANT_FEATURES, *WORLD_INVARIANT_FEATURES, "policy_metadata_only", "goal_partial"]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in [
        "metric_family",
        "metric_direction",
        "goal_family",
        "goal_target_direction",
        "goal_conflict_group_id",
        "perturbation_type",
        "source_experiment_id",
        "source_table",
        "source_metric_name",
        "policy_source_experiment_id",
        "goal_source_experiment_id",
        "world_source_experiment_id",
        "world_family",
        "policy_family",
    ]:
        if column not in frame.columns:
            frame[column] = "__missing__"
        frame[column] = frame[column].fillna("__missing__").astype(str).replace("", "__missing__")
    frame["source_balance_weight"] = source_balance_weights(frame)
    frame["residual_vs_source_metric"] = residualize_by_group_median(
        frame,
        target_column="target_transformed",
        group_columns=["source_experiment_id", "source_metric_name", "metric_direction"],
    )
    frame["heldout_policy_mask"] = assign_group_holdout(
        frame,
        group_column="canonical_policy_id",
        split_name="s09_heldout_policy",
        test_fraction=0.2,
    ) & (frame["canonical_policy_id"] != "__missing__")
    frame["heldout_world_mask"] = assign_group_holdout(
        frame,
        group_column="world_id",
        split_name="s09_heldout_world",
        test_fraction=0.2,
    ) & (frame["world_id"] != "__missing__")
    frame, s08_caveats = add_s08_controls(frame, args)
    return frame, input_stats, s08_caveats


def effect_for_feature(frame: pd.DataFrame, feature: str, mask: pd.Series) -> tuple[float, str, int, float]:
    mask = mask.astype(bool)
    if not mask.any():
        return float("nan"), "none", 0, float("nan")
    values = pd.to_numeric(frame.loc[mask, feature], errors="coerce").fillna(0.0)
    residual = pd.to_numeric(frame.loc[mask, "residual_vs_source_metric"], errors="coerce")
    weights = pd.to_numeric(frame.loc[mask, "source_balance_weight"], errors="coerce").fillna(0.0)
    unique = set(values.dropna().round(8).unique().tolist())
    if unique <= {0.0, 1.0}:
        active = values > 0.5
        active_rows = int(active.sum())
        if active_rows < 1 or int((~active).sum()) < 1:
            return float("nan"), "binary_difference", active_rows, float(active.mean()) if len(active) else float("nan")
        effect = weighted_mean(residual[active], weights[active]) - weighted_mean(residual[~active], weights[~active])
        return float(effect), "binary_difference", active_rows, float(active.mean())
    active = values >= values.quantile(0.75)
    corr = weighted_correlation(values, residual, weights)
    return float(corr), "weighted_correlation", int(active.sum()), float(active.mean()) if len(active) else float("nan")


def source_dominance_for_feature(frame: pd.DataFrame, feature: str) -> tuple[float, float]:
    values = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
    if set(values.round(8).unique().tolist()) <= {0.0, 1.0}:
        active = values > 0.5
    else:
        active = values >= values.quantile(0.75)
    if not active.any():
        return float("nan"), float("nan")
    weights = pd.to_numeric(frame.loc[active, "source_balance_weight"], errors="coerce").fillna(0.0)
    source_values = frame.loc[active, "source_experiment_id"]
    weighted_source = pd.DataFrame({"source_experiment_id": source_values, "_weight": weights}).groupby("source_experiment_id", observed=True)["_weight"].sum()
    total = float(weighted_source.sum())
    if total <= 0:
        counts = source_values.value_counts(normalize=True)
        return float(counts.max()), float(counts.size)
    return float((weighted_source / total).max()), float(weighted_source.size)


def support_weight_for_feature(frame: pd.DataFrame, feature: str) -> float:
    values = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
    if set(values.round(8).unique().tolist()) <= {0.0, 1.0}:
        active = values > 0.5
    else:
        active = values >= values.quantile(0.75)
    if feature.startswith("policy_"):
        support = frame.loc[active, "policy_s08_support_weight"]
    elif feature.startswith("world_"):
        support = frame.loc[active, "world_s08_support_weight"]
    else:
        support = frame.loc[active, "goal_s08_support_weight"] if "goal_s08_support_weight" in frame else pd.Series(dtype=float)
    return float(pd.to_numeric(support, errors="coerce").fillna(0.0).mean()) if len(support) else float("nan")


def sign(value: float) -> int:
    if not np.isfinite(value) or abs(value) < 1e-12:
        return 0
    return 1 if value > 0 else -1


def classify_candidate(row: Mapping[str, Any]) -> str:
    if int(row.get("active_rows_full", 0)) < MIN_ACTIVE_ROWS:
        return "insufficient_coverage"
    signs = [sign(float(row.get(column, np.nan))) for column in ("effect_full", "effect_heldout_policy", "effect_heldout_world")]
    if 0 in signs or len(set(signs)) > 1:
        return "unstable_or_counterexample_rich"
    min_abs = min(abs(float(row.get(column, 0.0))) for column in ("effect_full", "effect_heldout_policy", "effect_heldout_world"))
    if min_abs < MIN_ABS_EFFECT:
        return "weak_effect"
    if float(row.get("source_dominance_rate", 1.0)) > MAX_SOURCE_DOMINANCE_RATE:
        return "source_dominated"
    if row.get("feature_group") in {"policy_invariant", "world_invariant"} and float(row.get("mean_s08_support_weight_active", 0.0)) < MIN_SUPPORT_WEIGHT:
        return "low_s08_support"
    return "stable_supported_candidate"


def candidate_associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_groups = {
        **{feature: "policy_invariant" for feature in POLICY_INVARIANT_FEATURES},
        **{feature: "world_invariant" for feature in WORLD_INVARIANT_FEATURES},
        **{feature: "missingness_control" for feature in MISSINGNESS_CONTROL_FEATURES},
        **{feature: "s08_proxy_control" for feature in S08_PROXY_FEATURES if feature in frame.columns},
    }
    masks = {
        "full": pd.Series(True, index=frame.index),
        "heldout_policy": frame["heldout_policy_mask"].astype(bool),
        "heldout_world": frame["heldout_world_mask"].astype(bool),
    }
    for feature, group in feature_groups.items():
        if feature not in frame.columns:
            continue
        full_effect, effect_type, active_rows, active_share = effect_for_feature(frame, feature, masks["full"])
        policy_effect, _, policy_active, _ = effect_for_feature(frame, feature, masks["heldout_policy"])
        world_effect, _, world_active, _ = effect_for_feature(frame, feature, masks["heldout_world"])
        source_dominance, active_source_count = source_dominance_for_feature(frame, feature)
        support_weight = support_weight_for_feature(frame, feature)
        row = {
            "schema_version": INVARIANT_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "candidate_feature": feature,
            "feature_group": group,
            "effect_type": effect_type,
            "effect_full": full_effect,
            "effect_heldout_policy": policy_effect,
            "effect_heldout_world": world_effect,
            "abs_min_heldout_effect": float(np.nanmin(np.abs([full_effect, policy_effect, world_effect]))),
            "active_rows_full": int(active_rows),
            "active_rows_heldout_policy": int(policy_active),
            "active_rows_heldout_world": int(world_active),
            "active_share_full": active_share,
            "source_dominance_rate": source_dominance,
            "active_source_count": active_source_count,
            "mean_s08_support_weight_active": support_weight,
        }
        row["sign_consistent"] = len({sign(row["effect_full"]), sign(row["effect_heldout_policy"]), sign(row["effect_heldout_world"])}) == 1 and sign(row["effect_full"]) != 0
        row["candidate_status"] = "control_not_candidate" if group.endswith("_control") else classify_candidate(row)
        penalty = 1.0
        if np.isfinite(source_dominance):
            penalty *= max(0.0, 1.0 - max(0.0, source_dominance - 0.5))
        if group in {"policy_invariant", "world_invariant"} and np.isfinite(support_weight):
            penalty *= max(0.0, min(1.0, support_weight))
        row["stability_score"] = float(row["abs_min_heldout_effect"] * penalty) if np.isfinite(row["abs_min_heldout_effect"]) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["candidate_status", "stability_score"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def source_metric_median_predictions(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    group_cols = ["source_experiment_id", "source_metric_name", "metric_direction"]
    global_median = float(train["target_transformed"].median())
    medians = train.groupby(group_cols, observed=True)["target_transformed"].median().to_dict()
    preds = []
    for row in test[group_cols].to_dict(orient="records"):
        key = tuple(row[column] for column in group_cols)
        preds.append(float(medians.get(key, global_median)))
    return np.asarray(preds, dtype=float)


def fit_linear_sgd_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    max_train_rows: int,
    salt: str,
) -> np.ndarray:
    train_fit = deterministic_cap(train, max_train_rows, id_column="corpus_row_id", salt=salt)
    numeric = [column for column in numeric_columns if column in train_fit.columns]
    categorical = [column for column in categorical_columns if column in train_fit.columns]
    train_features = train_fit[numeric + categorical].copy()
    test_features = test[numeric + categorical].copy()
    for column in numeric:
        train_features[column] = pd.to_numeric(train_features[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_features[column] = pd.to_numeric(test_features[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for column in categorical:
        train_features[column] = train_features[column].fillna("__missing__").astype(str).replace("", "__missing__")
        test_features[column] = test_features[column].fillna("__missing__").astype(str).replace("", "__missing__")
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(with_mean=False), numeric))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical))
    if not transformers:
        return np.repeat(float(train["target_transformed"].median()), len(test))
    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=1.0)
    x_train = preprocessor.fit_transform(train_features)
    x_test = preprocessor.transform(test_features)
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        max_iter=300,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        learning_rate="constant",
        eta0=1e-3,
        average=True,
        random_state=RANDOM_SEED,
    )
    y_train = train_fit["target_transformed"].to_numpy(dtype=float)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    lower, upper = np.nanquantile(y_train, [0.001, 0.999])
    margin = max(1.0, float(upper - lower) * 0.25)
    return np.clip(preds, lower - margin, upper + margin)


def model_metrics_for_splits(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_defs = {
        "heldout_policy": ("canonical_policy_id", "heldout_policy_mask"),
        "heldout_world": ("world_id", "heldout_world_mask"),
    }
    model_defs = {
        "invariant_only_linear_sgd": (list(POLICY_INVARIANT_FEATURES + WORLD_INVARIANT_FEATURES), []),
        "invariant_plus_task_context_linear_sgd": (list(POLICY_INVARIANT_FEATURES + WORLD_INVARIANT_FEATURES), list(TASK_CONTEXT_COLUMNS)),
        "source_missingness_control_linear_sgd": (
            list(MISSINGNESS_CONTROL_FEATURES + S08_PROXY_FEATURES),
            list(SOURCE_CONTROL_COLUMNS) + ["policy_s08_support_status", "goal_s08_support_status", "world_s08_support_status"],
        ),
        "invariant_plus_s08_supported_proxy_linear_sgd": (
            list(POLICY_INVARIANT_FEATURES + WORLD_INVARIANT_FEATURES + S08_PROXY_FEATURES),
            list(TASK_CONTEXT_COLUMNS) + ["policy_s08_support_status", "goal_s08_support_status", "world_s08_support_status"],
        ),
    }
    for split_name, (group_col, mask_col) in split_defs.items():
        eligible = frame[group_col].astype(str) != "__missing__"
        test_mask = eligible & frame[mask_col].astype(bool)
        train_mask = eligible & ~frame[mask_col].astype(bool)
        train = frame[train_mask].copy()
        test = frame[test_mask].copy()
        if train.empty or test.empty:
            continue
        test_eval = deterministic_cap(test, args.max_linear_test_rows, id_column="corpus_row_id", salt=f"{split_name}:test_eval")
        y_true = test_eval["target_transformed"].to_numpy(dtype=float)
        baseline = source_metric_median_predictions(train, test_eval)
        metric_payload = metrics_for_predictions(y_true, baseline)
        rows.append(
            {
                "schema_version": INVARIANT_SCHEMA_VERSION,
                "research_step_id": STEP_ID,
                "split_name": split_name,
                "model_name": "source_metric_median",
                "train_rows": int(len(train)),
                "test_rows": int(len(test_eval)),
                "uncapped_test_rows": int(len(test)),
                "train_groups": int(train[group_col].nunique()),
                "test_groups": int(test_eval[group_col].nunique()),
                **metric_payload,
            }
        )
        for model_name, (numeric_columns, categorical_columns) in model_defs.items():
            numeric = [column for column in numeric_columns if column in frame.columns]
            categorical = [column for column in categorical_columns if column in frame.columns]
            preds = fit_linear_sgd_model(
                train,
                test_eval,
                numeric_columns=numeric,
                categorical_columns=categorical,
                max_train_rows=args.max_linear_train_rows,
                salt=f"{split_name}:{model_name}",
            )
            metric_payload = metrics_for_predictions(y_true, preds)
            rows.append(
                {
                    "schema_version": INVARIANT_SCHEMA_VERSION,
                    "research_step_id": STEP_ID,
                    "split_name": split_name,
                    "model_name": model_name,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test_eval)),
                    "uncapped_test_rows": int(len(test)),
                    "train_groups": int(train[group_col].nunique()),
                    "test_groups": int(test_eval[group_col].nunique()),
                    **metric_payload,
                }
            )
    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        baseline_lookup = metrics[metrics["model_name"] == "source_metric_median"].set_index("split_name")["mae"].to_dict()
        metrics["mae_delta_vs_source_metric_median"] = [
            float(record["mae"] - baseline_lookup.get(record["split_name"], np.nan))
            for record in metrics.to_dict(orient="records")
        ]
    return metrics


def build_counterexamples(frame: pd.DataFrame, candidates: pd.DataFrame, max_features: int = 10) -> pd.DataFrame:
    selected = candidates[
        candidates["feature_group"].isin(["policy_invariant", "world_invariant"])
        & candidates["candidate_status"].isin(["stable_supported_candidate", "source_dominated", "unstable_or_counterexample_rich", "weak_effect"])
    ].sort_values("stability_score", ascending=False, kind="mergesort").head(max_features)
    rows: list[dict[str, Any]] = []
    for record in selected.to_dict(orient="records"):
        feature = record["candidate_feature"]
        effect = float(record.get("effect_full", np.nan))
        if not np.isfinite(effect) or feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
        residual = pd.to_numeric(frame["residual_vs_source_metric"], errors="coerce")
        if set(values.round(8).unique().tolist()) <= {0.0, 1.0}:
            active = values > 0.5
        else:
            active = values >= values.quantile(0.75)
        if effect >= 0:
            counter_mask = active & (residual <= residual.quantile(0.10))
            counter_type = "active_low_residual"
        else:
            counter_mask = active & (residual >= residual.quantile(0.90))
            counter_type = "active_high_residual_against_negative_effect"
        subset = frame[counter_mask].copy()
        if subset.empty:
            if effect >= 0:
                subset = frame[(~active) & (residual >= residual.quantile(0.90))].copy()
                counter_type = "inactive_high_residual"
            else:
                subset = frame[(~active) & (residual <= residual.quantile(0.10))].copy()
                counter_type = "inactive_low_residual_against_negative_effect"
        subset["_feature_value"] = values.loc[subset.index]
        subset["_abs_residual"] = residual.loc[subset.index].abs()
        subset = subset.sort_values("_abs_residual", ascending=False, kind="mergesort").head(20)
        for row in subset.to_dict(orient="records"):
            rows.append(
                {
                    "schema_version": INVARIANT_SCHEMA_VERSION,
                    "research_step_id": STEP_ID,
                    "candidate_feature": feature,
                    "candidate_status": record.get("candidate_status", ""),
                    "expected_effect_direction": "positive" if effect >= 0 else "negative",
                    "counterexample_type": counter_type,
                    "corpus_row_id": row.get("corpus_row_id", ""),
                    "source_experiment_id": row.get("source_experiment_id", ""),
                    "source_metric_name": row.get("source_metric_name", ""),
                    "metric_family": row.get("metric_family", ""),
                    "world_id": row.get("world_id", ""),
                    "canonical_policy_id": row.get("canonical_policy_id", ""),
                    "canonical_goal_id": row.get("canonical_goal_id", ""),
                    "feature_value": float(row.get("_feature_value", 0.0)),
                    "target_transformed": float(row.get("target_transformed", np.nan)),
                    "residual_vs_source_metric": float(row.get("residual_vs_source_metric", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def feature_coverage(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    return (
        candidates.groupby(["feature_group", "candidate_status"], observed=True)
        .agg(
            feature_count=("candidate_feature", "size"),
            max_stability_score=("stability_score", "max"),
            median_source_dominance_rate=("source_dominance_rate", "median"),
            median_s08_support_weight=("mean_s08_support_weight_active", "median"),
        )
        .reset_index()
        .sort_values(["feature_group", "candidate_status"], kind="mergesort")
    )


def add_model_improvement_validation(validation: pd.DataFrame, model_metrics: pd.DataFrame) -> pd.DataFrame:
    if model_metrics.empty:
        extra = pd.DataFrame(
            [
                {
                    "validation_case": "invariant_models_improve_over_source_metric_baseline",
                    "success": False,
                    "detail": "model metrics table empty",
                }
            ]
        )
        return pd.concat([validation, extra], ignore_index=True)
    rows = []
    for split_name, group in model_metrics.groupby("split_name", observed=True):
        baseline = group[group["model_name"] == "source_metric_median"]["mae"]
        candidate = group[group["model_name"].isin(["invariant_plus_task_context_linear_sgd", "invariant_plus_s08_supported_proxy_linear_sgd"])]["mae"]
        if baseline.empty or candidate.empty:
            rows.append((split_name, False, "missing baseline or invariant model"))
            continue
        delta = float(candidate.min() - baseline.iloc[0])
        rows.append((split_name, delta <= -0.005, f"best invariant-model MAE delta vs source metric median={delta:.4f}"))
    success = bool(rows) and all(item[1] for item in rows)
    detail = "; ".join(f"{split}: {msg}" for split, _, msg in rows)
    extra = pd.DataFrame(
        [
            {
                "validation_case": "invariant_models_improve_over_source_metric_baseline",
                "success": success,
                "detail": detail,
            }
        ]
    )
    return pd.concat([validation, extra], ignore_index=True)


def plot_feature_importance(candidates: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot = candidates[candidates["feature_group"].isin(["policy_invariant", "world_invariant"])].copy()
    plot = plot.sort_values("stability_score", ascending=False, kind="mergesort").head(18)
    if plot.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No invariant candidates", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=170)
        plt.close(fig)
        return
    colors = {
        "stable_supported_candidate": "#3f7d4b",
        "source_dominated": "#b36b2c",
        "low_s08_support": "#9863a8",
        "unstable_or_counterexample_rich": "#9c3f3f",
        "weak_effect": "#657a99",
        "insufficient_coverage": "#878787",
    }
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [name.replace("policy_", "pol_").replace("world_", "world_") for name in plot["candidate_feature"]]
    y = np.arange(len(plot))
    ax.barh(y, plot["stability_score"].fillna(0.0), color=[colors.get(status, "#777777") for status in plot["candidate_status"]])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("S09 stability score after source/support penalties")
    ax.set_title("Candidate invariant feature ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)


def outcome_classification(validation: pd.DataFrame) -> str:
    if bool(validation["success"].all()):
        return "supportive"
    stable_check = validation[validation["validation_case"] == "stable_supported_candidate_present"]
    if not stable_check.empty and not bool(stable_check["success"].iloc[0]):
        return "null"
    return "constraining/contradictory"


def build_report(
    *,
    report_path: Path,
    artifacts_written: Sequence[str],
    candidates: pd.DataFrame,
    model_metrics: pd.DataFrame,
    counterexamples: pd.DataFrame,
    feature_coverage_table: pd.DataFrame,
    s08_caveats: pd.DataFrame,
    validation: pd.DataFrame,
    outcome: str,
    input_stats: Mapping[str, Any],
    unit_test_result: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    validation_success = bool(validation["success"].all())
    stable = candidates[candidates["candidate_status"] == "stable_supported_candidate"]
    caveats = (
        "S09 residualizes targets against source-metric medians and uses S08 supported-only distances only as bounded controls. "
        "S08 source-dominance and supported goal-conflict failures remain active caveats."
    )
    test_line = "not run"
    if unit_test_result is not None:
        test_line = f"{'pass' if unit_test_result['success'] else 'fail'}: `{unit_test_result['command']}` return code {unit_test_result['returnCode']}"
    text = f"""# E07 S09 Full Results: Search for Invariants

## Top Summary

- Research step ID: S09
- Completion status: complete
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: {"pass" if validation_success else "fail"} ({int(validation['success'].sum())}/{len(validation)} checks passed)
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Lay summary: S09 searched for policy and world properties that predict behavior after subtracting source-metric baselines, then checked whether the apparent signals held up on held-out policies and held-out worlds. It also recorded counterexamples and kept S08 source/missingness and goal-conflict failures as active limits.
- Recommended next action: Chief review before S10. If S10 proceeds, use only stable, non-source-dominated candidates and supported-only S08 distances as cautious inputs; otherwise revise source-balanced evidence or conflict encodings first.

## Frozen Question

Do properties such as locality horizon, memory depth, feedback strength, stochasticity, or target-position knowledge predict competence across arrays, grids, graphs, chimeras, and homeostatic tasks?

## Inputs

- S04 unified behavior corpus: `{args.s04_corpus}` (SHA-256 `{sha256_file(args.s04_corpus)}`)
- S01 world inventory: `{args.s01_world_inventory}` (SHA-256 `{sha256_file(args.s01_world_inventory)}`)
- S02 policy table: `{args.s02_policy_table}` (SHA-256 `{sha256_file(args.s02_policy_table)}`)
- S03 goal table: `{args.s03_goal_table}` (SHA-256 `{sha256_file(args.s03_goal_table)}`)
- S08 benchmarks: `{args.s08_distance_benchmarks}` (SHA-256 `{sha256_file(args.s08_distance_benchmarks)}`)
- S08 neighbors: `{args.s08_neighbors}` (SHA-256 `{sha256_file(args.s08_neighbors)}`)
- S08 entity metadata: `{args.s08_entity_metadata}` (SHA-256 `{sha256_file(args.s08_entity_metadata)}`)
- S08 validation checks: `{args.s08_validation}` (SHA-256 `{sha256_file(args.s08_validation)}`)
- S08 conflict validation: `{args.s08_conflict_validation}` (SHA-256 `{sha256_file(args.s08_conflict_validation)}`)

## Methods

S09 used S04 metric-observation rows as the measurement layer. To keep the run deterministic and balanced, rows were capped to {args.max_rows_per_source_metric} per `(source_experiment_id, source_table, source_metric_name)` stratum. The target was the S05-style oriented signed-log metric value. Every row was residualized against the median target for its `(source_experiment_id, source_metric_name, metric_direction)` group before candidate invariant effects were estimated.

Policy invariant proxies were extracted from S02 metadata and DSL/source payloads: memory, signaling, global-oracle use, target-position knowledge, stochastic choice, feedback/adaptation, locality-horizon score, rule/condition/action counts, and action vocabulary flags. World invariant proxies were extracted from S01 state/action/transition/perturbation/measurement text: dimensionality, frozen perturbation, chimeric identity, damage/repair, homeostasis, governance/dominance, target morphology, null/control context, scheduler class, local/nonlocal observation context, action count, and measurement count.

S08 was used only as bounded proxy input: supported-only nearest-neighbor availability, distances, and support weights were carried as controls and reliability fields. The S08 source-dominance failure and supported-only goal-conflict separation failure were copied into the S09 caveat audit. Source, metric, and missingness baselines were evaluated as controls rather than interpreted as invariants.

Candidate effects were tested on the full balanced frame, the deterministic held-out policy split, and the deterministic held-out world split. Predictive checks compared source-metric medians, invariant-only linear SGD models, invariant plus task-context linear SGD models, source/missingness controls, and invariant plus S08-supported-proxy controls. Linear SGD checks scaled numeric features, used a small fixed learning rate with averaged weights, and clipped predictions to the training target envelope as a guard against optimizer outliers; they were used as coarse held-out controls, not as final predictive models.

## Commands

- `python -m unittest tests.e07.test_invariant_schema`
- `python scripts/e07_s09_invariants.py`

Unit-test result: {test_line}

## Dependencies and Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- numpy: {np.__version__}
- scikit-learn: {sklearn.__version__}
- matplotlib: {matplotlib.__version__}
- Worker count: deterministic single-process analysis; linear SGD models use sparse one-hot controls, scaled numeric features, deterministic train/test caps, no GPU training; configured CPU ceiling noted as {N_JOBS}.
- Repository commit before S09 commit: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Branch: `{git_output(args.repo_dir, ['branch', '--show-current'])}`

## Results

Input and sampling statistics:

```json
{json.dumps(input_stats, indent=2, sort_keys=True, default=str)}
```

Stable supported candidates: {len(stable)}

### Candidate Invariants

{markdown_table(candidates.sort_values("stability_score", ascending=False, kind="mergesort").head(30), max_rows=30)}

### Feature Coverage

{markdown_table(feature_coverage_table, max_rows=40)}

### Held-Out Model Checks

{markdown_table(model_metrics, max_rows=40)}

### S08 Caveat Audit

{markdown_table(s08_caveats)}

### Counterexamples

{markdown_table(counterexamples.head(40), max_rows=40)}

## Validation

{markdown_table(validation)}

## Output Artifacts

{chr(10).join(f'- `{item}`' for item in artifacts_written)}

## Caveats, Blockers, and Limitations

- This is a computational proxy analysis over S04 metrics, not causal or biological validation.
- Residualizing by source-metric medians removes a major source artifact but can also remove real source-specific behavior.
- S08 all-policy distances were source dominated, so S09 uses S08 only as supported-only proxy controls and support annotations.
- S08 supported-only goal conflict separation failed; S09 does not use goal-distance separation as evidence for an invariant.
- Metadata-derived features such as memory depth or feedback strength are proxies inferred from DSL/source text and representation metadata.
- Held-out policy/world tests reduce but do not eliminate confounding by metric family, source experiment, and missing upstream artifacts.
- Counterexamples are expected and are preserved as constraints on candidate laws.

## Recommended Next Action

Stop before S10 for Chief review. If the Chief accepts S09 as bounded evidence, S10 should use only candidates with `candidate_status = stable_supported_candidate` and include source/missingness controls in any universality-class clustering.
"""
    write_text(report_path, text)


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
        unit_test_result = run_command([sys.executable, "-m", "unittest", "tests.e07.test_invariant_schema"], args.repo_dir)

    frame, input_stats, s08_caveats = load_analysis_frame(args)
    candidates = candidate_associations(frame)
    model_metrics = model_metrics_for_splits(frame, args)
    counterexamples = build_counterexamples(frame, candidates)
    coverage = feature_coverage(candidates)
    validation = validate_invariant_artifacts(candidates, model_metrics, counterexamples, s08_caveats)
    validation = add_model_improvement_validation(validation, model_metrics)
    if unit_test_result is not None:
        validation = pd.concat(
            [
                validation,
                pd.DataFrame(
                    [
                        {
                            "validation_case": "unit_tests_passed",
                            "success": bool(unit_test_result["success"]),
                            "detail": f"`{unit_test_result['command']}` return code {unit_test_result['returnCode']}",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    outcome = outcome_classification(validation)

    candidates_path = results_dir / "e07_candidate_invariants.parquet"
    candidates_csv_path = tables_dir / "e07_candidate_invariants.csv"
    model_metrics_path = results_dir / "e07_invariant_model_metrics.parquet"
    model_metrics_csv_path = tables_dir / "e07_invariant_model_metrics.csv"
    counterexamples_path = results_dir / "e07_invariant_counterexamples.parquet"
    counterexamples_csv_path = tables_dir / "e07_invariant_counterexamples.csv"
    coverage_path = tables_dir / "e07_invariant_feature_coverage.csv"
    s08_caveat_path = tables_dir / "e07_invariant_s08_caveat_audit.csv"
    validation_path = step_dir / "e07_s09_validation_checks.csv"
    config_path = step_dir / "s09_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"
    figure_path = figures_dir / "invariant_feature_importance.png"

    candidates.to_parquet(candidates_path, index=False)
    candidates.to_csv(candidates_csv_path, index=False)
    model_metrics.to_parquet(model_metrics_path, index=False)
    model_metrics.to_csv(model_metrics_csv_path, index=False)
    counterexamples.to_parquet(counterexamples_path, index=False)
    counterexamples.to_csv(counterexamples_csv_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    s08_caveats.to_csv(s08_caveat_path, index=False)
    validation.to_csv(validation_path, index=False)
    plot_feature_importance(candidates, figure_path)
    write_json(
        config_path,
        {
            "schemaVersion": INVARIANT_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "randomSeed": RANDOM_SEED,
            "maxRowsPerSourceMetric": args.max_rows_per_source_metric,
            "maxLinearTrainRows": args.max_linear_train_rows,
            "maxLinearTestRows": args.max_linear_test_rows,
            "maxSourceDominanceRate": MAX_SOURCE_DOMINANCE_RATE,
            "minAbsEffect": MIN_ABS_EFFECT,
            "minActiveRows": MIN_ACTIVE_ROWS,
            "minSupportWeight": MIN_SUPPORT_WEIGHT,
            "policyInvariantFeatures": list(POLICY_INVARIANT_FEATURES),
            "worldInvariantFeatures": list(WORLD_INVARIANT_FEATURES),
            "missingnessControlFeatures": list(MISSINGNESS_CONTROL_FEATURES),
            "s08ProxyFeatures": list(S08_PROXY_FEATURES),
            "taskContextColumns": list(TASK_CONTEXT_COLUMNS),
            "sourceControlColumns": list(SOURCE_CONTROL_COLUMNS),
            "inputStats": input_stats,
        },
    )

    artifacts_written = [
        str(report_path),
        str(candidates_path),
        str(candidates_csv_path),
        str(model_metrics_path),
        str(model_metrics_csv_path),
        str(counterexamples_path),
        str(counterexamples_csv_path),
        str(coverage_path),
        str(s08_caveat_path),
        str(figure_path),
        str(validation_path),
        str(config_path),
        str(manifest_path),
    ]
    build_report(
        report_path=report_path,
        artifacts_written=artifacts_written,
        candidates=candidates,
        model_metrics=model_metrics,
        counterexamples=counterexamples,
        feature_coverage_table=coverage,
        s08_caveats=s08_caveats,
        validation=validation,
        outcome=outcome,
        input_stats=input_stats,
        unit_test_result=unit_test_result,
        args=args,
    )

    manifest_payload = {
        "schemaVersion": "eidosoma.e07.s09.artifact_manifest.v1",
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
            "s04Corpus": {"path": str(args.s04_corpus), "sha256": sha256_file(args.s04_corpus)},
            "s01WorldInventory": {"path": str(args.s01_world_inventory), "sha256": sha256_file(args.s01_world_inventory)},
            "s02PolicyTable": {"path": str(args.s02_policy_table), "sha256": sha256_file(args.s02_policy_table)},
            "s03GoalTable": {"path": str(args.s03_goal_table), "sha256": sha256_file(args.s03_goal_table)},
            "s08DistanceBenchmarks": {"path": str(args.s08_distance_benchmarks), "sha256": sha256_file(args.s08_distance_benchmarks)},
            "s08Neighbors": {"path": str(args.s08_neighbors), "sha256": sha256_file(args.s08_neighbors)},
            "s08EntityMetadata": {"path": str(args.s08_entity_metadata), "sha256": sha256_file(args.s08_entity_metadata)},
            "s08Validation": {"path": str(args.s08_validation), "sha256": sha256_file(args.s08_validation)},
            "s08ConflictValidation": {"path": str(args.s08_conflict_validation), "sha256": sha256_file(args.s08_conflict_validation)},
        },
        "coverage": {
            "analysisRows": int(len(frame)),
            "candidateRows": int(len(candidates)),
            "stableSupportedCandidates": int((candidates["candidate_status"] == "stable_supported_candidate").sum()),
            "modelMetricRows": int(len(model_metrics)),
            "counterexampleRows": int(len(counterexamples)),
        },
        "artifacts": [
            self_referential_artifact_entry(report_path, artifacts_dir, "S09 full-results report."),
            artifact_entry(candidates_path, artifacts_dir, "Candidate invariant table."),
            artifact_entry(candidates_csv_path, artifacts_dir, "Candidate invariant CSV table."),
            artifact_entry(model_metrics_path, artifacts_dir, "Held-out invariant model metrics."),
            artifact_entry(model_metrics_csv_path, artifacts_dir, "Held-out model metrics CSV."),
            artifact_entry(counterexamples_path, artifacts_dir, "Counterexample records for candidate invariants."),
            artifact_entry(counterexamples_csv_path, artifacts_dir, "Counterexample CSV table."),
            artifact_entry(coverage_path, artifacts_dir, "Feature coverage/status summary."),
            artifact_entry(s08_caveat_path, artifacts_dir, "S08 caveats carried into S09."),
            artifact_entry(figure_path, artifacts_dir, "Candidate invariant feature-importance figure."),
            artifact_entry(validation_path, artifacts_dir, "S09 validation checks."),
            artifact_entry(config_path, artifacts_dir, "S09 analysis configuration."),
            self_referential_artifact_entry(manifest_path, artifacts_dir, "S09 artifact manifest."),
        ],
    }
    write_json(manifest_path, manifest_payload)
    print(f"[S09] wrote {report_path}")
    print(f"[S09] validation {int(validation['success'].sum())}/{len(validation)} outcome={outcome}")


if __name__ == "__main__":
    main()
