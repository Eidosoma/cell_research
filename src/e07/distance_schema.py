"""Helpers for E07 S08 Platonic-distance definitions and validation."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PLATONIC_DISTANCE_SCHEMA_VERSION = "eidosoma.e07.platonic_distance.v1"


def standardized_numeric_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mean-center and variance-scale numeric columns, dropping constants."""

    if frame.empty:
        return frame.copy(), {"featureCount": 0, "retainedFeatureCount": 0, "droppedConstantFeatureCount": 0}
    numeric = frame.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    means = numeric.mean(axis=0)
    stds = numeric.std(axis=0, ddof=0)
    retained = stds > 0
    scaled = (numeric.loc[:, retained] - means.loc[retained]) / stds.loc[retained]
    return scaled, {
        "featureCount": int(numeric.shape[1]),
        "retainedFeatureCount": int(retained.sum()),
        "droppedConstantFeatureCount": int((~retained).sum()),
    }


def metadata_feature_matrix(
    frame: pd.DataFrame,
    *,
    id_column: str,
    entity_ids: Sequence[str],
    categorical_columns: Iterable[str],
    numeric_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a deterministic one-hot metadata feature matrix."""

    columns = [id_column, *list(categorical_columns), *list(numeric_columns)]
    work = frame[[column for column in columns if column in frame.columns]].copy()
    work[id_column] = work[id_column].fillna("__missing__").astype(str)
    work = work.drop_duplicates(id_column).set_index(id_column)
    blocks: list[pd.DataFrame] = []
    used_categorical = []
    for column in categorical_columns:
        if column not in work.columns:
            continue
        values = work[column].fillna("__missing__").astype(str).replace("", "__missing__")
        blocks.append(pd.get_dummies(values, prefix=column, dtype=float))
        used_categorical.append(column)
    used_numeric = []
    for column in numeric_columns:
        if column not in work.columns:
            continue
        used_numeric.append(column)
        blocks.append(pd.to_numeric(work[column], errors="coerce").fillna(0.0).to_frame(column))
    if blocks:
        matrix = pd.concat(blocks, axis=1).fillna(0.0)
    else:
        matrix = pd.DataFrame(index=work.index)
    matrix = matrix.reindex(list(map(str, entity_ids)), fill_value=0.0)
    scaled, stats = standardized_numeric_frame(matrix)
    stats.update({"categoricalColumns": used_categorical, "numericColumns": used_numeric})
    return scaled, stats


def weighted_mean_profile(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    entity_ids: Sequence[str],
    feature_columns: Iterable[str],
    value_column: str,
    weight_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a standardized entity-by-feature matrix of weighted target means."""

    work = frame.copy()
    work[entity_column] = work[entity_column].fillna("__missing__").astype(str)
    work = work[work[entity_column] != "__missing__"].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work[weight_column] = pd.to_numeric(work[weight_column], errors="coerce").fillna(0.0)
    work = work[np.isfinite(work[value_column].to_numpy(dtype=float)) & (work[weight_column] > 0)].copy()
    blocks: list[pd.DataFrame] = []
    used_features = []
    for column in feature_columns:
        if column not in work.columns:
            continue
        sub = work[[entity_column, column, value_column, weight_column]].copy()
        sub[column] = sub[column].fillna("__missing__").astype(str).replace("", "__missing__")
        sub["profile_feature"] = f"{column}=" + sub[column]
        sub["_weighted_value"] = sub[value_column] * sub[weight_column]
        grouped = sub.groupby([entity_column, "profile_feature"], observed=True).agg(
            weighted_sum=("_weighted_value", "sum"),
            weight_sum=(weight_column, "sum"),
        )
        grouped["profile_value"] = grouped["weighted_sum"] / grouped["weight_sum"].replace(0, np.nan)
        blocks.append(grouped["profile_value"].unstack("profile_feature").fillna(0.0))
        used_features.append(column)
    if blocks:
        matrix = pd.concat(blocks, axis=1).fillna(0.0)
    else:
        matrix = pd.DataFrame(index=list(map(str, entity_ids)))
    matrix = matrix.reindex(list(map(str, entity_ids)), fill_value=0.0)
    scaled, stats = standardized_numeric_frame(matrix)
    stats.update(
        {
            "featureColumns": used_features,
            "nonzeroWeightedRows": int(len(work)),
            "entitiesWithNonzeroWeight": int(work[entity_column].nunique()) if not work.empty else 0,
        }
    )
    return scaled, stats


def euclidean_neighbor_table(
    coordinates: np.ndarray,
    ids: Sequence[str],
    *,
    k: int,
    entity_type: str,
    distance_variant: str,
    baseline_family: str,
) -> pd.DataFrame:
    """Compute top-k Euclidean neighbors for every row."""

    id_list = list(map(str, ids))
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2:
        raise ValueError("coordinates must be two-dimensional")
    if values.shape[0] != len(id_list):
        raise ValueError("number of ids must match coordinate rows")
    if len(id_list) < 2:
        return pd.DataFrame(
            columns=[
                "entity_type",
                "distance_variant",
                "baseline_family",
                "query_entity_id",
                "neighbor_rank",
                "neighbor_entity_id",
                "distance",
            ]
        )
    squared_norms = np.sum(values * values, axis=1)
    squared = squared_norms[:, None] + squared_norms[None, :] - 2.0 * values @ values.T
    distances = np.sqrt(np.maximum(squared, 0.0))
    np.fill_diagonal(distances, np.inf)
    max_k = min(k, len(id_list) - 1)
    rows: list[dict[str, Any]] = []
    for index, query_id in enumerate(id_list):
        order = np.argsort(distances[index])[:max_k]
        for rank, neighbor_index in enumerate(order, start=1):
            rows.append(
                {
                    "entity_type": entity_type,
                    "distance_variant": distance_variant,
                    "baseline_family": baseline_family,
                    "query_entity_id": query_id,
                    "neighbor_rank": int(rank),
                    "neighbor_entity_id": id_list[int(neighbor_index)],
                    "distance": float(distances[index, int(neighbor_index)]),
                }
            )
    return pd.DataFrame(rows)


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
    row_count = left_values.shape[0]
    if row_count < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    max_possible = row_count * (row_count - 1) // 2
    pair_count = min(max_pairs, max_possible)
    if pair_count == max_possible:
        first, second = np.triu_indices(row_count, k=1)
    else:
        first = rng.integers(0, row_count, size=pair_count * 2)
        second = rng.integers(0, row_count, size=pair_count * 2)
        mask = first != second
        first = first[mask][:pair_count]
        second = second[mask][:pair_count]
    if len(first) < 3:
        return float("nan")
    left_dist = np.linalg.norm(left_values[first] - left_values[second], axis=1)
    right_dist = np.linalg.norm(right_values[first] - right_values[second], axis=1)
    return float(pd.Series(left_dist).corr(pd.Series(right_dist), method="spearman"))


def support_weight_from_status(status: Any, count: Any = 0, *, supported_status: str = "behavior_profile_supported", min_rows: int = 10) -> float:
    """Map support-status labels and row counts to a bounded uncertainty weight."""

    text = "" if status is None else str(status)
    value = pd.to_numeric(pd.Series([count]), errors="coerce").fillna(0.0).iloc[0]
    if text == supported_status:
        return 1.0
    if value > 0:
        return float(np.clip(np.sqrt(float(value) / max(1.0, float(min_rows))), 0.10, 0.95))
    return 0.0


def attach_neighbor_metadata(neighbors: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach generic query/neighbor metadata to a distance-neighbor table."""

    if neighbors.empty:
        return neighbors.copy()
    required = {"entity_type", "entity_id", "source_experiment_id", "family", "support_status", "support_weight"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing columns: {missing}")
    labels = metadata.copy()
    labels["entity_id"] = labels["entity_id"].astype(str)
    query = labels.add_prefix("query_").rename(
        columns={"query_entity_id": "query_entity_id", "query_entity_type": "entity_type"}
    )
    neighbor = labels.add_prefix("neighbor_").rename(
        columns={"neighbor_entity_id": "neighbor_entity_id", "neighbor_entity_type": "entity_type"}
    )
    merged = neighbors.merge(query, on=["query_entity_id", "entity_type"], how="left")
    merged = merged.merge(neighbor, on=["neighbor_entity_id", "entity_type"], how="left")
    merged["same_source_experiment"] = (
        merged["query_source_experiment_id"].fillna("__missing__").astype(str)
        == merged["neighbor_source_experiment_id"].fillna("__missing__").astype(str)
    )
    merged["same_family"] = merged["query_family"].fillna("__missing__").astype(str) == merged["neighbor_family"].fillna("__missing__").astype(str)
    merged["pair_support_weight"] = np.minimum(
        pd.to_numeric(merged["query_support_weight"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["neighbor_support_weight"], errors="coerce").fillna(0.0),
    )
    merged["low_support_pair"] = merged["pair_support_weight"] < 0.5
    return merged


def weighted_bool_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = values.astype(float)
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    total = float(numeric_weights.sum())
    if total <= 0:
        return float(numeric_values.mean()) if len(numeric_values) else float("nan")
    return float((numeric_values * numeric_weights).sum() / total)


def summarize_neighbors(neighbors: pd.DataFrame, *, k: int) -> pd.DataFrame:
    """Summarize source, family, support, and distance behavior by variant."""

    if neighbors.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in neighbors[neighbors["neighbor_rank"] <= k].groupby(["entity_type", "distance_variant", "baseline_family"], observed=True):
        entity_type, distance_variant, baseline_family = keys
        top1 = group[group["neighbor_rank"] == 1]
        rows.append(
            {
                "entity_type": entity_type,
                "distance_variant": distance_variant,
                "baseline_family": baseline_family,
                "eligible_entity_count": int(group["query_entity_id"].nunique()),
                "neighbor_rows": int(len(group)),
                "mean_topk_distance": float(pd.to_numeric(group["distance"], errors="coerce").mean()),
                "median_top1_distance": float(pd.to_numeric(top1["distance"], errors="coerce").median()) if not top1.empty else float("nan"),
                "same_source_rate_at_k": float(group["same_source_experiment"].astype(float).mean()),
                "support_weighted_same_source_rate_at_k": weighted_bool_mean(group["same_source_experiment"], group["pair_support_weight"]),
                "same_family_rate_at_k": float(group["same_family"].astype(float).mean()),
                "low_support_neighbor_rate_at_k": float(group["low_support_pair"].astype(float).mean()),
                "mean_pair_support_weight_at_k": float(pd.to_numeric(group["pair_support_weight"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def validate_platonic_distance_artifacts(
    benchmarks: pd.DataFrame,
    neighbors: pd.DataFrame,
    source_missingness_audit: pd.DataFrame,
    conflict_validation: pd.DataFrame,
    *,
    required_variants: Iterable[str],
) -> pd.DataFrame:
    """Validate S08 distance artifacts and expected stress-test branches."""

    checks: list[dict[str, Any]] = []
    observed = set((benchmarks["entity_type"].astype(str) + ":" + benchmarks["distance_variant"].astype(str)).tolist()) if not benchmarks.empty else set()
    required = set(required_variants)
    missing = sorted(required - observed)
    checks.append(
        {
            "validation_case": "required_distance_variants_present",
            "success": not missing,
            "detail": "all required variants present" if not missing else f"missing variants: {missing}",
        }
    )
    checks.append(
        {
            "validation_case": "neighbor_table_populated",
            "success": not neighbors.empty and {"query_entity_id", "neighbor_entity_id", "distance_variant", "distance"} <= set(neighbors.columns),
            "detail": f"neighbor rows: {len(neighbors)}",
        }
    )
    finite_distances = not neighbors.empty and np.isfinite(pd.to_numeric(neighbors["distance"], errors="coerce").to_numpy(dtype=float)).all()
    checks.append(
        {
            "validation_case": "neighbor_distances_finite",
            "success": bool(finite_distances),
            "detail": "all neighbor distances finite" if finite_distances else "non-finite neighbor distances found or table empty",
        }
    )
    baseline_families = set(benchmarks.get("baseline_family", pd.Series(dtype=str)).astype(str))
    expected_baselines = {"platonic", "syntax", "source_metadata", "metric_only"}
    checks.append(
        {
            "validation_case": "syntax_source_metric_baselines_compared",
            "success": expected_baselines <= baseline_families,
            "detail": f"baseline families observed: {sorted(baseline_families)}",
        }
    )
    has_supported = any(benchmarks.get("distance_variant", pd.Series(dtype=str)).astype(str).str.contains("supported_only", regex=False))
    has_uncertainty = any(benchmarks.get("distance_variant", pd.Series(dtype=str)).astype(str).str.contains("uncertainty_weighted", regex=False))
    checks.append(
        {
            "validation_case": "supported_only_and_uncertainty_weighted_inputs_used",
            "success": bool(has_supported and has_uncertainty),
            "detail": f"supported_only={has_supported}; uncertainty_weighted={has_uncertainty}",
        }
    )
    audit_success = not source_missingness_audit.empty and bool(source_missingness_audit["success"].astype(bool).all())
    checks.append(
        {
            "validation_case": "source_missingness_dominance_not_detected",
            "success": bool(audit_success),
            "detail": "source/missingness audit passed" if audit_success else "source or missingness dominance detected in at least one branch",
        }
    )
    branch_present = not conflict_validation.empty and {"distance_variant", "eligible_mode", "known_conflict_distance_lift_over_reference_median"} <= set(conflict_validation.columns)
    checks.append(
        {
            "validation_case": "conflict_aware_validation_branch_present",
            "success": bool(branch_present),
            "detail": f"conflict validation rows: {len(conflict_validation)}",
        }
    )
    supported_branch = conflict_validation[
        (conflict_validation.get("distance_variant", pd.Series(dtype=str)).astype(str) == "platonic_supported_only")
        & (conflict_validation.get("eligible_mode", pd.Series(dtype=str)).astype(str) == "supported_only")
    ]
    conflict_success = bool(not supported_branch.empty and supported_branch["success"].astype(bool).all())
    detail = "supported-only conflict branch separated known conflicts" if conflict_success else "supported-only known conflicts were not farther than references"
    checks.append(
        {
            "validation_case": "supported_platonic_conflict_separation_passed",
            "success": conflict_success,
            "detail": detail,
        }
    )
    return pd.DataFrame(checks)
