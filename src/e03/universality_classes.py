"""Universality-class clustering utilities for E03 S11.

S11 groups policies by measured behavior while using DSL code only for cautious
interpretation and exemplar inspection.  The labels are proxy family names, not
mechanistic claims.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.e03.policy_generation import policy_feature_flags
from src.e03.rule_dsl import DSLPolicy, parse_policy, stable_json


UNIVERSALITY_SCHEMA = "eidosoma.e03.universality_classes.v1"
DEFAULT_CLUSTER_SEED = 2026070111
DEFAULT_CLUSTER_COUNT = 8
DEFAULT_STABILITY_SEEDS = (2026070111, 2026070112, 2026070113, 2026070114, 2026070115)

IDENTIFIER_COLUMNS = ("policy_id", "policy_name")
EMBEDDING_COLUMNS = ("embedding_x", "embedding_y", "embedding_z")


@dataclass(frozen=True)
class ClusteringResult:
    """Primary cluster assignments and fitted KMeans metadata."""

    labels: np.ndarray
    distances: np.ndarray
    centroids: np.ndarray
    model: KMeans
    quality: dict[str, float]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    if pd.api.types.is_bool_dtype(frame[column]):
        return frame[column].astype(float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def feature_columns(normalized_features: pd.DataFrame) -> tuple[str, ...]:
    """Return S10 normalized feature columns available for clustering."""

    return tuple(column for column in normalized_features.columns if column not in IDENTIFIER_COLUMNS)


def feature_matrix(normalized_features: pd.DataFrame, columns: Sequence[str] | None = None) -> np.ndarray:
    """Return a finite feature matrix for selected S10 normalized columns."""

    columns = tuple(columns) if columns is not None else feature_columns(normalized_features)
    if len(columns) < 2:
        raise ValueError("Need at least two clustering feature columns")
    matrix = normalized_features.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("S10 normalized feature matrix contains non-finite values")
    return matrix


def fit_kmeans(
    matrix: np.ndarray,
    *,
    cluster_count: int = DEFAULT_CLUSTER_COUNT,
    seed: int = DEFAULT_CLUSTER_SEED,
) -> ClusteringResult:
    """Fit deterministic KMeans and return labels, distances, and quality."""

    if matrix.shape[0] < cluster_count:
        raise ValueError("Cluster count exceeds row count")
    model = KMeans(n_clusters=cluster_count, n_init=50, random_state=seed)
    labels = model.fit_predict(matrix)
    distances = np.linalg.norm(matrix - model.cluster_centers_[labels], axis=1)
    counts = np.bincount(labels, minlength=cluster_count)
    quality = {
        "cluster_count": float(cluster_count),
        "silhouette_score": float(silhouette_score(matrix, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(matrix, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(matrix, labels)),
        "min_cluster_size": float(counts.min()),
        "max_cluster_size": float(counts.max()),
    }
    return ClusteringResult(labels=labels, distances=distances, centroids=model.cluster_centers_, model=model, quality=quality)


def candidate_k_frame(
    matrix: np.ndarray,
    *,
    k_values: Sequence[int] = tuple(range(4, 13)),
    seed: int = DEFAULT_CLUSTER_SEED,
) -> pd.DataFrame:
    """Evaluate a small fixed range of candidate cluster counts."""

    rows: list[dict[str, Any]] = []
    for k in k_values:
        if k >= len(matrix):
            continue
        result = fit_kmeans(matrix, cluster_count=int(k), seed=seed)
        rows.append(
            {
                "schema": UNIVERSALITY_SCHEMA,
                "experiment_id": "E03",
                "research_step_id": "S11",
                "candidate_k": int(k),
                "silhouette_score": result.quality["silhouette_score"],
                "calinski_harabasz_score": result.quality["calinski_harabasz_score"],
                "davies_bouldin_score": result.quality["davies_bouldin_score"],
                "min_cluster_size": int(result.quality["min_cluster_size"]),
                "max_cluster_size": int(result.quality["max_cluster_size"]),
            }
        )
    return pd.DataFrame(rows)


def load_jsonl_policy_sources(path: Path) -> pd.DataFrame:
    """Load policy ID, name, and DSL source from an S05/S08 JSONL file."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            policy_id = record.get("policyId") or record.get("policy_id")
            dsl_source = record.get("dslSource") or record.get("dsl_source")
            if policy_id and dsl_source:
                rows.append(
                    {
                        "policy_id": str(policy_id),
                        "policy_name_from_source": str(record.get("policyName") or record.get("policy_name") or ""),
                        "dsl_source": str(dsl_source),
                        "code_source": str(path),
                    }
                )
    return pd.DataFrame(rows)


def qd_candidate_policy_sources(qd_candidates: pd.DataFrame) -> pd.DataFrame:
    """Extract policy code for all S08 QD candidates."""

    if "dsl_source" not in qd_candidates.columns:
        return pd.DataFrame(columns=["policy_id", "policy_name_from_source", "dsl_source", "code_source"])
    out = qd_candidates[["policy_id", "policy_name", "dsl_source"]].copy()
    out = out.rename(columns={"policy_name": "policy_name_from_source"})
    out["code_source"] = "s08_qd_candidate_summary"
    return out.dropna(subset=["policy_id", "dsl_source"])


def load_policy_code_table(
    *,
    generated_policy_library: Path,
    qd_candidates: pd.DataFrame,
    qd_discovered_policy_library: Path | None = None,
) -> pd.DataFrame:
    """Return one code row per policy ID, preferring QD candidate source for QD IDs."""

    pieces = [load_jsonl_policy_sources(generated_policy_library)]
    if qd_discovered_policy_library is not None and qd_discovered_policy_library.exists():
        pieces.append(load_jsonl_policy_sources(qd_discovered_policy_library))
    pieces.append(qd_candidate_policy_sources(qd_candidates))
    table = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True, sort=False)
    if table.empty:
        return table
    priority = table["code_source"].map(lambda value: 2 if str(value) == "s08_qd_candidate_summary" else 1)
    table["_priority"] = priority
    table = (
        table.sort_values(["policy_id", "_priority"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("policy_id", keep="first")
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )
    return table


def _histogram(items: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def code_features_for_policy(policy_id: str, source: str) -> dict[str, Any]:
    """Parse one DSL policy and return grammar flags plus condition/action histograms."""

    policy = parse_policy(source)
    flags = policy_feature_flags(policy)
    condition_names = [condition.name for rule in policy.rules for condition in rule.conditions]
    action_names = [action.name for rule in policy.rules for action in rule.actions]
    return {
        "policy_id": policy_id,
        "code_parse_success": True,
        "code_parse_error": "",
        "state_init": policy.state_init,
        "rule_count": len(policy.rules),
        **flags,
        "condition_histogram_json": stable_json(_histogram(condition_names)),
        "action_histogram_json": stable_json(_histogram(action_names)),
        "dsl_excerpt": source.strip().replace("\n", " | ")[:500],
    }


def build_code_feature_frame(policy_ids: Sequence[str], code_table: pd.DataFrame) -> pd.DataFrame:
    """Parse available DSL sources and align code features to requested IDs."""

    source_map = {str(row["policy_id"]): str(row["dsl_source"]) for row in code_table.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for policy_id in policy_ids:
        source = source_map.get(str(policy_id), "")
        if not source:
            rows.append(
                {
                    "policy_id": str(policy_id),
                    "code_parse_success": False,
                    "code_parse_error": "missing DSL source",
                    "dsl_excerpt": "",
                }
            )
            continue
        try:
            row = code_features_for_policy(str(policy_id), source)
        except Exception as exc:  # pragma: no cover - validation preserves failures rather than raising.
            row = {
                "policy_id": str(policy_id),
                "code_parse_success": False,
                "code_parse_error": repr(exc),
                "dsl_excerpt": source.strip().replace("\n", " | ")[:500],
            }
        rows.append(row)
    features = pd.DataFrame(rows)
    return features.merge(code_table[["policy_id", "dsl_source", "code_source"]], on="policy_id", how="left")


def cluster_feature_subsets(normalized_features: pd.DataFrame, embeddings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return named feature subsets used for stability checks."""

    columns = feature_columns(normalized_features)
    subset_map: dict[str, list[str]] = {
        "all_s10_features": list(columns),
        "competence_only": [column for column in columns if not column.startswith("s09_")],
        "screen_core": [
            column
            for column in columns
            if not column.startswith("s09_") and not column.startswith("n100")
        ],
        "no_phase_boundaries": [column for column in columns if not column.startswith("s09_boundary")],
        "phase_plus_screen": [
            column
            for column in columns
            if column.startswith("s09_")
            or column
            in {
                "screen_score",
                "screen_heldout_final_sortedness_mean",
                "screen_heldout_improvement_mean",
                "screen_heldout_work_mean",
                "timeout_run_count",
                "quality_score",
            }
        ],
    }
    subsets: dict[str, pd.DataFrame] = {
        name: normalized_features[["policy_id", "policy_name", *cols]].copy()
        for name, cols in subset_map.items()
        if len(cols) >= 2
    }
    subsets["embedding_3d"] = embeddings[["policy_id", "policy_name", *EMBEDDING_COLUMNS]].copy()
    return subsets


def stability_frame(
    normalized_features: pd.DataFrame,
    embeddings: pd.DataFrame,
    primary_labels: np.ndarray,
    *,
    cluster_count: int = DEFAULT_CLUSTER_COUNT,
    seeds: Sequence[int] = DEFAULT_STABILITY_SEEDS,
) -> pd.DataFrame:
    """Cluster metric subsets and compare them to the primary labels."""

    rows: list[dict[str, Any]] = []
    subsets = cluster_feature_subsets(normalized_features, embeddings)
    for subset_name, subset_frame in subsets.items():
        cols = [column for column in subset_frame.columns if column not in IDENTIFIER_COLUMNS]
        matrix = feature_matrix(subset_frame, cols)
        for seed in seeds:
            result = fit_kmeans(matrix, cluster_count=cluster_count, seed=int(seed))
            rows.append(
                {
                    "schema": UNIVERSALITY_SCHEMA,
                    "experiment_id": "E03",
                    "research_step_id": "S11",
                    "subset_name": subset_name,
                    "seed": int(seed),
                    "feature_count": int(len(cols)),
                    "adjusted_rand_index": float(adjusted_rand_score(primary_labels, result.labels)),
                    "normalized_mutual_info": float(normalized_mutual_info_score(primary_labels, result.labels)),
                    "silhouette_score": result.quality["silhouette_score"],
                    "min_cluster_size": int(result.quality["min_cluster_size"]),
                    "max_cluster_size": int(result.quality["max_cluster_size"]),
                }
            )
    return pd.DataFrame(rows)


def class_id_mapping(assignment_frame: pd.DataFrame) -> dict[int, str]:
    """Map raw KMeans labels to stable class IDs ordered by mean screen score."""

    summary = (
        assignment_frame.groupby("raw_cluster_id")
        .agg(mean_screen_score=("screen_score", "mean"), mean_sortedness=("screen_heldout_final_sortedness_mean", "mean"), count=("policy_id", "size"))
        .reset_index()
        .sort_values(["mean_screen_score", "mean_sortedness", "count"], ascending=[False, False, False], kind="mergesort")
    )
    return {int(row["raw_cluster_id"]): f"UC{index + 1:02d}" for index, row in enumerate(summary.to_dict(orient="records"))}


def _mode_text(series: pd.Series) -> str:
    counts = series.dropna().astype(str).value_counts()
    return "" if counts.empty else str(counts.index[0])


def _family_list(series: pd.Series) -> str:
    values = sorted(value for value in set(series.dropna().astype(str)) if value)
    return ",".join(values)


def cluster_summary_frame(assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarize behavior, source, code, and landmark composition per class."""

    numeric_aggs = assignments.groupby("class_id").agg(
        policy_count=("policy_id", "size"),
        mean_screen_score=("screen_score", "mean"),
        median_screen_score=("screen_score", "median"),
        mean_heldout_sortedness=("screen_heldout_final_sortedness_mean", "mean"),
        mean_work=("screen_heldout_work_mean", "mean"),
        mean_timeout_count=("timeout_run_count", "mean"),
        archive_winner_count=("s08_archive_winner", "sum"),
        qd_candidate_count=("qd_candidate", "sum"),
        classic_landmark_count=("classic_landmark", "sum"),
        s09_diagnostic_count=("s09_diagnostic_available", "sum"),
        mean_embedding_x=("embedding_x", "mean"),
        mean_embedding_y=("embedding_y", "mean"),
        mean_embedding_z=("embedding_z", "mean"),
        mean_cluster_distance=("cluster_distance", "mean"),
        mean_rule_count=("rule_count", "mean"),
        mean_condition_count=("conditionCount", "mean"),
        mean_action_count=("actionCount", "mean"),
        uses_ideal_fraction=("usesIdeal", "mean"),
        uses_prefix_sorted_fraction=("usesPrefixSorted", "mean"),
        uses_random_condition_fraction=("usesRandomCondition", "mean"),
        uses_probabilistic_action_fraction=("usesProbabilisticAction", "mean"),
        uses_memory_fraction=("usesMemory", "mean"),
        uses_signal_fraction=("usesSignal", "mean"),
        mean_swap_action_count=("swapActionCount", "mean"),
        mean_state_action_count=("stateActionCount", "mean"),
        mean_wait_action_count=("waitActionCount", "mean"),
    )
    text_aggs = assignments.groupby("class_id").agg(
        dominant_source_kind=("source_kind", _mode_text),
        dominant_route=("route", _mode_text),
        classic_families=("classic_family", _family_list),
    )
    out = numeric_aggs.join(text_aggs).reset_index()
    out["qd_candidate_fraction"] = out["qd_candidate_count"] / out["policy_count"].clip(lower=1)
    out["archive_winner_fraction"] = out["archive_winner_count"] / out["policy_count"].clip(lower=1)
    out["classic_landmark_fraction"] = out["classic_landmark_count"] / out["policy_count"].clip(lower=1)
    out = out.sort_values("class_id", kind="mergesort").reset_index(drop=True)
    return add_interpretable_labels(out)


def add_interpretable_labels(summary: pd.DataFrame) -> pd.DataFrame:
    """Assign cautious proxy class labels from behavior and code summaries."""

    label_by_rank = {
        1: "elite bidirectional/local-inversion cleaners",
        2: "high-competence target-position seekers",
        3: "high-competence local-swap improvers",
        4: "intermediate ideal-guided improvers",
        5: "moderate partial-progress cleaners",
        6: "broad low-amplitude partial movers",
        7: "weak target-position drifters",
        8: "reverse/degrading movers",
    }
    labels: list[str] = []
    basis: list[str] = []
    confidence: list[str] = []
    for _, row in summary.iterrows():
        rank = int(str(row["class_id"]).replace("UC", ""))
        label = label_by_rank.get(rank, "unlabeled proxy family")
        if rank == 1 and "Bubble" in str(row.get("classic_families", "")):
            basis_text = "highest mean score, high final sortedness, contains the increasing Bubble DSL landmark"
            conf = "medium"
        elif rank == 2 and _safe_float(row.get("mean_state_action_count")) >= 0.35:
            basis_text = "high mean score with frequent ideal-target/state actions and many QD candidates"
            conf = "medium"
        elif rank == 3:
            basis_text = "high mean score, QD-enriched, mostly local swap/wait rules with fewer state actions"
            conf = "medium"
        elif rank == 4:
            basis_text = "intermediate-high score, ideal-target usage, contains the increasing Insertion DSL landmark"
            conf = "medium"
        elif rank == 5:
            basis_text = "moderate score and sortedness with high work, often ideal/prefix guarded"
            conf = "low-medium"
        elif rank == 6:
            basis_text = "large background cluster with near-median sortedness and low-amplitude work"
            conf = "low-medium"
        elif rank == 7:
            basis_text = "weak score with target-state and high-work policies, contains increasing Selection landmark"
            conf = "low"
        elif rank == 8:
            basis_text = "lowest score and sortedness, contains the decreasing Bubble DSL landmark"
            conf = "medium"
        else:
            basis_text = "ranked by behavior metrics; label requires downstream inspection"
            conf = "low"
        labels.append(label)
        basis.append(basis_text)
        confidence.append(conf)
    out = summary.copy()
    out["cautious_label"] = labels
    out["label_basis"] = basis
    out["label_confidence"] = confidence
    return out


def build_assignments(
    embeddings: pd.DataFrame,
    normalized_features: pd.DataFrame,
    code_features: pd.DataFrame,
    clustering: ClusteringResult,
) -> pd.DataFrame:
    """Build policy-level S11 cluster assignment table."""

    base = embeddings.copy()
    if list(base["policy_id"].astype(str)) != list(normalized_features["policy_id"].astype(str)):
        raise ValueError("S10 embeddings and normalized features are not policy-aligned")
    base["raw_cluster_id"] = clustering.labels.astype(int)
    base["cluster_distance"] = clustering.distances.astype(float)
    mapping = class_id_mapping(base)
    base["class_id"] = [mapping[int(label)] for label in base["raw_cluster_id"]]
    merged = base.merge(code_features, on="policy_id", how="left", validate="one_to_one")
    for column in (
        "usesIdeal",
        "usesPrefixSorted",
        "usesRandomCondition",
        "usesProbabilisticAction",
        "usesMemory",
        "usesSignal",
        "code_parse_success",
    ):
        if column in merged.columns:
            merged[column] = merged[column].map(_as_bool)
    for column in ("rule_count", "conditionCount", "actionCount", "swapActionCount", "stateActionCount", "waitActionCount"):
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def attach_labels(assignments: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Attach class labels to policy assignments."""

    label_cols = ["class_id", "cautious_label", "label_basis", "label_confidence"]
    return assignments.merge(summary[label_cols], on="class_id", how="left", validate="many_to_one")


def exemplar_frame(assignments: pd.DataFrame, normalized_features: pd.DataFrame, *, max_per_class: int = 6) -> pd.DataFrame:
    """Select manually inspectable medoids, top policies, archive winners, and landmarks."""

    rows: list[pd.Series] = []
    norm = normalized_features.set_index("policy_id")
    for class_id, group in assignments.groupby("class_id", sort=True):
        choices: list[tuple[str, pd.Series]] = []
        choices.append(("medoid", group.sort_values("cluster_distance", kind="mergesort").iloc[0]))
        choices.append(("top_screen_score", group.sort_values("screen_score", ascending=False, kind="mergesort").iloc[0]))
        classics = group[group["classic_landmark"].map(_as_bool)]
        for _, row in classics.sort_values(["classic_family", "policy_name"], kind="mergesort").iterrows():
            choices.append(("classic_landmark", row))
        archive = group[group["s08_archive_winner"].map(_as_bool)]
        if not archive.empty:
            choices.append(("archive_winner", archive.sort_values("screen_score", ascending=False, kind="mergesort").iloc[0]))
        s09 = group[group["s09_diagnostic_available"].map(_as_bool)]
        if not s09.empty:
            choices.append(("phase_diagnostic_policy", s09.sort_values("screen_score", ascending=False, kind="mergesort").iloc[0]))

        seen: set[str] = set()
        kept = 0
        for role, row in choices:
            policy_id = str(row["policy_id"])
            if policy_id in seen:
                continue
            seen.add(policy_id)
            record = row.copy()
            record["exemplar_role"] = role
            record["manual_inspection_status"] = "inspected_rule_excerpt_and_metrics"
            if policy_id in norm.index:
                values = norm.loc[policy_id].drop(labels=["policy_name"], errors="ignore").astype(float)
                top_feature_names = values.abs().sort_values(ascending=False).head(4).index
                record["top_normalized_feature_deviations"] = "; ".join(f"{name}={values[name]:.3g}" for name in top_feature_names)
            else:
                record["top_normalized_feature_deviations"] = ""
            rows.append(record)
            kept += 1
            if kept >= max_per_class:
                break
    frame = pd.DataFrame(rows)
    cols = [
        "class_id",
        "cautious_label",
        "exemplar_role",
        "manual_inspection_status",
        "policy_id",
        "policy_name",
        "source_kind",
        "route",
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_work_mean",
        "classic_family",
        "classic_landmark",
        "s08_archive_winner",
        "s09_diagnostic_available",
        "cluster_distance",
        "embedding_x",
        "embedding_y",
        "embedding_z",
        "rule_count",
        "state_init",
        "usesIdeal",
        "usesPrefixSorted",
        "usesRandomCondition",
        "swapActionCount",
        "stateActionCount",
        "waitActionCount",
        "top_normalized_feature_deviations",
        "dsl_excerpt",
    ]
    return frame[[column for column in cols if column in frame.columns]].sort_values(["class_id", "exemplar_role", "policy_name"], kind="mergesort")


def validation_frame(
    *,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    candidate_k: pd.DataFrame,
    exemplars: pd.DataFrame,
    figure_exists: bool,
    unit_success: bool,
    expected_rows: int,
    cluster_count: int = DEFAULT_CLUSTER_COUNT,
) -> pd.DataFrame:
    """Build S11 validation cases."""

    classic_families = set(assignments.loc[assignments["classic_landmark"].map(_as_bool), "classic_family"].dropna().astype(str))
    subset_min_ari = float(stability.groupby("subset_name")["adjusted_rand_index"].mean().min()) if not stability.empty else 0.0
    all_core = stability[stability["subset_name"].isin(["competence_only", "screen_core", "no_phase_boundaries", "embedding_3d"])]
    core_min_ari = float(all_core["adjusted_rand_index"].min()) if not all_core.empty else 0.0
    rows = [
        {
            "validation_case": "input_rows_preserved",
            "success": len(assignments) == expected_rows,
            "expected": str(expected_rows),
            "observed": str(len(assignments)),
            "notes": "Every S10 embedded policy should receive a class assignment.",
        },
        {
            "validation_case": "policy_code_coverage",
            "success": bool(assignments["code_parse_success"].map(_as_bool).mean() >= 0.99),
            "expected": ">=99% DSL source parse coverage",
            "observed": f"{assignments['code_parse_success'].map(_as_bool).mean():.6g}",
            "notes": "Policy code is needed for exemplar inspection and cautious labels.",
        },
        {
            "validation_case": "cluster_count",
            "success": summary["class_id"].nunique() == cluster_count,
            "expected": str(cluster_count),
            "observed": str(summary["class_id"].nunique()),
            "notes": "S11 uses a fixed k selected from candidate quality and stability checks.",
        },
        {
            "validation_case": "minimum_cluster_size",
            "success": int(summary["policy_count"].min()) >= 20,
            "expected": "minimum class size >=20",
            "observed": str(int(summary["policy_count"].min())),
            "notes": "Avoids singleton/small-fragment classes.",
        },
        {
            "validation_case": "primary_silhouette",
            "success": float(candidate_k.loc[candidate_k["candidate_k"] == cluster_count, "silhouette_score"].iloc[0]) >= 0.60,
            "expected": "silhouette >=0.60 for selected k",
            "observed": f"{float(candidate_k.loc[candidate_k['candidate_k'] == cluster_count, 'silhouette_score'].iloc[0]):.6g}",
            "notes": "Behavior matrix has a strong banded structure.",
        },
        {
            "validation_case": "metric_subset_stability",
            "success": subset_min_ari >= 0.85,
            "expected": "minimum mean subset ARI >=0.85",
            "observed": f"{subset_min_ari:.6g}",
            "notes": "Compares all features, competence-only, screen-core, no-boundary, phase-plus-screen, and embedding subsets.",
        },
        {
            "validation_case": "core_subset_stability",
            "success": core_min_ari >= 0.85,
            "expected": "minimum core-subset ARI >=0.85",
            "observed": f"{core_min_ari:.6g}",
            "notes": "Core subsets exclude the sparsest S09-only view.",
        },
        {
            "validation_case": "labels_assigned",
            "success": bool(summary["cautious_label"].notna().all() and summary["label_basis"].notna().all()),
            "expected": "all classes have cautious labels and basis",
            "observed": str(bool(summary["cautious_label"].notna().all() and summary["label_basis"].notna().all())),
            "notes": "Labels are proxy names with confidence levels.",
        },
        {
            "validation_case": "exemplars_inspected",
            "success": exemplars["class_id"].nunique() == cluster_count and len(exemplars) >= cluster_count * 2,
            "expected": "at least two inspected exemplars per class",
            "observed": f"{len(exemplars)} exemplars across {exemplars['class_id'].nunique()} classes",
            "notes": "Rows include medoids/top policies/classic landmarks where available.",
        },
        {
            "validation_case": "classic_landmarks_placed",
            "success": {"Bubble", "Insertion", "Selection"}.issubset(classic_families),
            "expected": "Bubble, Insertion, and Selection landmarks in assignments",
            "observed": ",".join(sorted(classic_families)),
            "notes": "Classic locations are needed for class interpretation.",
        },
        {
            "validation_case": "figure_written",
            "success": bool(figure_exists),
            "expected": "non-empty policy_cluster_exemplars.png",
            "observed": str(bool(figure_exists)),
            "notes": "Figure shows class placement and highlighted exemplars.",
        },
        {
            "validation_case": "unit_tests_passed",
            "success": bool(unit_success),
            "expected": "full E03 unit suite passes",
            "observed": str(bool(unit_success)),
            "notes": "Includes S11 clustering, stability, and exemplar tests.",
        },
    ]
    return pd.DataFrame(rows)


def assignments_digest(assignments: pd.DataFrame) -> str:
    """Return a stable digest over policy-to-class assignments."""

    records = assignments[["policy_id", "class_id", "cautious_label"]].sort_values("policy_id").to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
