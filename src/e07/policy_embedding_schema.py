"""Helpers for E07 S06 policy-embedding construction and validation."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


POLICY_EMBEDDING_SCHEMA_VERSION = "eidosoma.e07.policy_embedding.v1"


def stable_fraction(label: str, *, salt: str = "e07-s06") -> float:
    digest = hashlib.sha256(f"{salt}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def weighted_mean_profile(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    feature_column: str,
    value_column: str,
    weight_column: str,
) -> pd.DataFrame:
    """Return an entity x feature matrix of weighted mean values."""

    if frame.empty:
        return pd.DataFrame()
    required = [entity_column, feature_column, value_column, weight_column]
    work = frame[required].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work[weight_column] = pd.to_numeric(work[weight_column], errors="coerce").fillna(0.0)
    work = work.dropna(subset=[entity_column, feature_column, value_column])
    work = work[work[weight_column] > 0].copy()
    if work.empty:
        return pd.DataFrame()
    work["_weighted_value"] = work[value_column] * work[weight_column]
    grouped = work.groupby([entity_column, feature_column], observed=True).agg(
        weighted_sum=("_weighted_value", "sum"),
        weight_sum=(weight_column, "sum"),
    )
    grouped["profile_value"] = grouped["weighted_sum"] / grouped["weight_sum"].replace(0, np.nan)
    return grouped["profile_value"].unstack(feature_column).fillna(0.0)


def standardize_profile_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mean-center and variance-scale non-constant profile columns."""

    if matrix.empty:
        return matrix.copy(), {"featureCount": 0, "retainedFeatureCount": 0, "droppedConstantFeatureCount": 0}
    numeric = matrix.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    means = numeric.mean(axis=0)
    stds = numeric.std(axis=0, ddof=0)
    retained = stds > 0
    scaled = (numeric.loc[:, retained] - means.loc[retained]) / stds.loc[retained]
    stats = {
        "featureCount": int(numeric.shape[1]),
        "retainedFeatureCount": int(retained.sum()),
        "droppedConstantFeatureCount": int((~retained).sum()),
    }
    return scaled, stats


def pad_embedding(array: np.ndarray, dims: int) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim != 2:
        raise ValueError("embedding array must be two-dimensional")
    if values.shape[1] >= dims:
        return values[:, :dims]
    return np.pad(values, ((0, 0), (0, dims - values.shape[1])), constant_values=0.0)


def cosine_neighbor_table(
    embeddings: np.ndarray,
    ids: Iterable[str],
    *,
    k: int = 5,
    embedding_name: str = "behavior",
) -> pd.DataFrame:
    """Compute top-k cosine neighbors for every row in an embedding matrix."""

    id_list = list(ids)
    values = np.asarray(embeddings, dtype=float)
    if len(id_list) != values.shape[0]:
        raise ValueError("number of ids must match embedding rows")
    if values.size == 0 or len(id_list) == 0:
        return pd.DataFrame(
            columns=["embedding_name", "query_policy_id", "neighbor_rank", "neighbor_policy_id", "cosine_similarity"]
        )
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.where(norms == 0, 1.0, norms)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    max_k = min(k, max(0, len(id_list) - 1))
    rows: list[dict[str, Any]] = []
    for row_index, query_id in enumerate(id_list):
        if max_k == 0:
            continue
        order = np.argsort(-similarity[row_index])[:max_k]
        for rank, neighbor_index in enumerate(order, start=1):
            rows.append(
                {
                    "embedding_name": embedding_name,
                    "query_policy_id": query_id,
                    "neighbor_rank": int(rank),
                    "neighbor_policy_id": id_list[int(neighbor_index)],
                    "cosine_similarity": float(similarity[row_index, int(neighbor_index)]),
                }
            )
    return pd.DataFrame(rows)


def add_neighbor_labels(neighbors: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach query and neighbor policy metadata to a neighbor table."""

    if neighbors.empty:
        return neighbors.copy()
    label_frame = labels.copy()
    query = label_frame.add_prefix("query_").rename(columns={"query_canonical_policy_id": "query_policy_id"})
    neighbor = label_frame.add_prefix("neighbor_").rename(columns={"neighbor_canonical_policy_id": "neighbor_policy_id"})
    merged = neighbors.merge(query, on="query_policy_id", how="left").merge(neighbor, on="neighbor_policy_id", how="left")
    for column in ("policy_family", "algorithm", "source_experiment_id", "representation_type"):
        query_col = f"query_{column}"
        neighbor_col = f"neighbor_{column}"
        if query_col in merged.columns and neighbor_col in merged.columns:
            merged[f"same_{column}"] = (
                merged[query_col].fillna("__missing__").astype(str)
                == merged[neighbor_col].fillna("__missing__").astype(str)
            )
    return merged


def label_agreement(neighbors: pd.DataFrame, label_column: str) -> float:
    same_column = f"same_{label_column}"
    if same_column not in neighbors or neighbors.empty:
        return float("nan")
    return float(neighbors[same_column].astype(float).mean())


def neighbor_overlap_at_k(left: pd.DataFrame, right: pd.DataFrame, *, k: int = 5) -> float:
    """Mean Jaccard-like overlap fraction between two top-k neighbor tables."""

    if left.empty or right.empty:
        return float("nan")
    left_sets = (
        left[left["neighbor_rank"] <= k]
        .groupby("query_policy_id")["neighbor_policy_id"]
        .apply(lambda values: set(map(str, values)))
        .to_dict()
    )
    right_sets = (
        right[right["neighbor_rank"] <= k]
        .groupby("query_policy_id")["neighbor_policy_id"]
        .apply(lambda values: set(map(str, values)))
        .to_dict()
    )
    overlaps = []
    for query_id, left_set in left_sets.items():
        right_set = right_sets.get(query_id)
        if not right_set:
            continue
        overlaps.append(len(left_set & right_set) / max(1, min(k, len(left_set), len(right_set))))
    return float(np.mean(overlaps)) if overlaps else float("nan")


def sampled_pairwise_distance_spearman(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_pairs: int = 50_000,
    seed: int = 20260703,
) -> float:
    """Spearman correlation between sampled pairwise Euclidean distances."""

    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape[0] != right_values.shape[0]:
        raise ValueError("left and right matrices must have the same row count")
    n_rows = left_values.shape[0]
    if n_rows < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    max_possible = n_rows * (n_rows - 1) // 2
    pair_count = min(max_pairs, max_possible)
    if pair_count == max_possible:
        first, second = np.triu_indices(n_rows, k=1)
    else:
        first = rng.integers(0, n_rows, size=pair_count * 2)
        second = rng.integers(0, n_rows, size=pair_count * 2)
        mask = first != second
        first = first[mask][:pair_count]
        second = second[mask][:pair_count]
    if len(first) < 3:
        return float("nan")
    left_dist = np.linalg.norm(left_values[first] - left_values[second], axis=1)
    right_dist = np.linalg.norm(right_values[first] - right_values[second], axis=1)
    return float(pd.Series(left_dist).corr(pd.Series(right_dist), method="spearman"))


def validate_policy_embedding_artifacts(
    embeddings: pd.DataFrame,
    neighbors: pd.DataFrame,
    stability: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    *,
    expected_policy_ids: Iterable[str],
    min_supported_policies: int = 100,
    min_family_lift: float = 0.02,
    min_stability_distance_spearman: float = 0.65,
    min_neighbor_overlap: float = 0.25,
    max_source_identity_agreement: float = 0.95,
) -> pd.DataFrame:
    """Validate S06 embedding coverage, numerical integrity, and basic behavior signal."""

    checks: list[dict[str, Any]] = []

    required_columns = {"canonical_policy_id", "embedding_x", "embedding_y", "behavior_observation_count"}
    missing_columns = sorted(required_columns - set(embeddings.columns))
    checks.append(
        {
            "validation_case": "embedding_required_columns_present",
            "success": not missing_columns,
            "detail": "all required embedding columns present" if not missing_columns else f"missing columns: {missing_columns}",
        }
    )

    observed_ids = set(embeddings.get("canonical_policy_id", pd.Series(dtype=str)).astype(str))
    expected_ids = set(map(str, expected_policy_ids))
    missing_ids = sorted(expected_ids - observed_ids)
    checks.append(
        {
            "validation_case": "all_s02_policies_represented",
            "success": not missing_ids,
            "detail": f"represented {len(observed_ids & expected_ids)}/{len(expected_ids)} S02 canonical policies",
        }
    )

    coordinate_cols = [column for column in embeddings.columns if column.startswith("behavior_embedding_dim_") or column in {"embedding_x", "embedding_y"}]
    finite_coordinates = embeddings[coordinate_cols].replace([np.inf, -np.inf], np.nan).notna().all().all() if coordinate_cols else False
    checks.append(
        {
            "validation_case": "embedding_coordinates_finite",
            "success": bool(finite_coordinates),
            "detail": f"finite coordinate columns: {coordinate_cols[:6]}{'...' if len(coordinate_cols) > 6 else ''}",
        }
    )

    supported_count = int((pd.to_numeric(embeddings.get("behavior_observation_count", 0), errors="coerce").fillna(0) >= 10).sum())
    checks.append(
        {
            "validation_case": "sufficient_behavior_supported_policies",
            "success": supported_count >= min_supported_policies,
            "detail": f"policies with at least 10 S05 modeling rows: {supported_count}",
        }
    )

    checks.append(
        {
            "validation_case": "neighbor_table_populated",
            "success": not neighbors.empty and {"query_policy_id", "neighbor_policy_id", "embedding_name"} <= set(neighbors.columns),
            "detail": f"neighbor rows: {len(neighbors)}",
        }
    )

    mean_distance_corr = float(pd.to_numeric(stability.get("pairwise_distance_spearman", pd.Series(dtype=float)), errors="coerce").mean())
    mean_overlap = float(pd.to_numeric(stability.get("neighbor_overlap_at_5", pd.Series(dtype=float)), errors="coerce").mean())
    checks.append(
        {
            "validation_case": "embedding_stability_passes_threshold",
            "success": np.isfinite(mean_distance_corr) and mean_distance_corr >= min_stability_distance_spearman and np.isfinite(mean_overlap) and mean_overlap >= min_neighbor_overlap,
            "detail": f"mean distance Spearman={mean_distance_corr:.3f}; mean neighbor overlap@5={mean_overlap:.3f}",
        }
    )

    lookup = baseline_comparison.set_index("metric_name")["metric_value"].to_dict() if "metric_name" in baseline_comparison else {}
    family_lift = float(lookup.get("behavior_policy_family_agreement_lift_over_random_at_5", np.nan))
    source_agreement = float(lookup.get("behavior_source_experiment_agreement_at_5", np.nan))
    checks.append(
        {
            "validation_case": "behavior_neighbor_retrieval_exceeds_random_family_baseline",
            "success": np.isfinite(family_lift) and family_lift >= min_family_lift,
            "detail": f"policy-family agreement lift over random@5={family_lift:.3f}",
        }
    )
    checks.append(
        {
            "validation_case": "embedding_not_collapsed_to_source_identity",
            "success": np.isfinite(source_agreement) and source_agreement <= max_source_identity_agreement,
            "detail": f"same source-experiment neighbor agreement@5={source_agreement:.3f}",
        }
    )
    checks.append(
        {
            "validation_case": "syntax_baseline_compared",
            "success": "behavior_syntax_neighbor_overlap_at_5" in lookup and np.isfinite(float(lookup.get("behavior_syntax_neighbor_overlap_at_5", np.nan))),
            "detail": f"behavior/syntax neighbor overlap@5={lookup.get('behavior_syntax_neighbor_overlap_at_5', 'missing')}",
        }
    )
    return pd.DataFrame(checks)
