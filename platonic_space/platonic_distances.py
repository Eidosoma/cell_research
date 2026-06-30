from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .behavior_predictor import DEFAULT_TARGET_COLUMNS
from .policy_embeddings import matrix_from_profile, target_error_weights
from .world_schema import compact_json


PLATONIC_DISTANCE_SCHEMA_VERSION = "e07_s08_platonic_distance.v1"
PLATONIC_DISTANCE_MODEL_VERSION = "e07_s08_empirical_metric.v1"
SPARSE_COVERAGE_PENALTY_WEIGHT = 0.10
PLATONIC_DISTANCE_CLAIM_BOUNDARY = (
    "Empirical computational distances over S04-S07 simulation-derived behavior summaries, policy embeddings, "
    "goal embeddings, and S05 error-limit weights only. These distances are proxy relationships among simulated "
    "policies, goals, and worlds; they are not metaphysical distances and do not validate biological "
    "morphogenesis, cognition, agency, clinical behavior, sentience, or living chimeras."
)


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()[:length]


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("embedding_"))


def _safe_label(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and math.isnan(value):
        return "unknown"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "unknown"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _count_quality(values: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    positive = numeric[numeric > 0]
    if positive.empty:
        return pd.Series(np.zeros(len(numeric), dtype=float), index=values.index)
    scale = float(positive.quantile(0.95))
    if not math.isfinite(scale) or scale <= 0:
        scale = float(positive.max())
    if column == "rowCount":
        numerator = np.log1p(numeric.to_numpy(dtype=float))
        denominator = max(math.log1p(scale), 1e-9)
    else:
        numerator = numeric.to_numpy(dtype=float)
        denominator = max(scale, 1e-9)
    return pd.Series(np.clip(numerator / denominator, 0.0, 1.0), index=values.index)


def policy_sparse_penalty(policy_embeddings: pd.DataFrame, policy_coverage: pd.DataFrame) -> pd.DataFrame:
    coverage = policy_coverage.set_index("abstractPolicyId", drop=False)
    rows = policy_embeddings[["abstractPolicyId", "rowCount", "observedTargetCount", "uniqueWorldCount", "uniqueGoalCount"]].copy()
    rows = rows.set_index("abstractPolicyId", drop=False)
    for column in ("rowCount", "observedTargetCount", "uniqueWorldCount", "uniqueGoalCount"):
        if column in coverage.columns:
            rows[column] = pd.to_numeric(coverage.reindex(rows.index)[column], errors="coerce").fillna(rows[column])
    row_quality = _count_quality(rows["rowCount"], "rowCount")
    target_quality = _count_quality(rows["observedTargetCount"], "observedTargetCount")
    world_quality = _count_quality(rows["uniqueWorldCount"], "uniqueWorldCount")
    goal_quality = _count_quality(rows["uniqueGoalCount"], "uniqueGoalCount")
    reliability = (0.45 * row_quality) + (0.30 * target_quality) + (0.15 * world_quality) + (0.10 * goal_quality)
    penalty = (1.0 - reliability).clip(0.0, 1.0)
    out = pd.DataFrame(
        {
            "abstractPolicyId": rows.index.astype(str),
            "sparseCoveragePenalty": penalty.to_numpy(dtype=float),
            "distanceUncertaintyLevel": [_level_from_penalty(value) for value in penalty],
        }
    )
    return out.reset_index(drop=True)


def goal_sparse_penalty(goal_embeddings: pd.DataFrame, goal_uncertainty: pd.DataFrame) -> pd.DataFrame:
    uncertainty = goal_uncertainty.set_index("abstractGoalId", drop=False)
    ids = goal_embeddings["abstractGoalId"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    max_score = float(pd.to_numeric(goal_uncertainty.get("sparseUncertaintyScore", pd.Series([1.0])), errors="coerce").max())
    if not math.isfinite(max_score) or max_score <= 0:
        max_score = 1.0
    for goal_id in ids:
        row = uncertainty.loc[goal_id].to_dict() if goal_id in uncertainty.index else {}
        level = _safe_label(row.get("sparseUncertaintyLevel"))
        score = _finite_float(row.get("sparseUncertaintyScore"), default=1.0)
        penalty = min(max(score / max_score, 0.0), 1.0)
        rows.append(
            {
                "abstractGoalId": goal_id,
                "sparseCoveragePenalty": float(penalty),
                "distanceUncertaintyLevel": level if level != "unknown" else _level_from_penalty(penalty),
            }
        )
    return pd.DataFrame(rows)


def _level_from_penalty(value: float) -> str:
    if value < 0.35:
        return "lower"
    if value < 0.65:
        return "moderate"
    return "high"


def _target_stats(target_weights: pd.DataFrame) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for row in target_weights.to_dict(orient="records"):
        target = str(row.get("target"))
        rows[target] = {
            "center": _finite_float(row.get("centerMedian"), default=math.nan),
            "scale": _finite_float(row.get("robustScale"), default=math.nan),
            "weight": _finite_float(row.get("s05ReliabilityWeight"), default=0.0),
        }
    return rows


def _weighted_z(values: pd.Series, stats: Mapping[str, float]) -> pd.Series:
    center = _finite_float(stats.get("center"), default=math.nan)
    scale = _finite_float(stats.get("scale"), default=math.nan)
    weight = _finite_float(stats.get("weight"), default=0.0)
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return ((pd.to_numeric(values, errors="coerce") - center) / scale) * weight


def build_world_profiles(
    frame: pd.DataFrame,
    world_catalog: pd.DataFrame,
    policy_embeddings: pd.DataFrame,
    goal_embeddings: pd.DataFrame,
    target_weights: pd.DataFrame,
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
    min_world_rows: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = _target_stats(target_weights)
    policy_cols = embedding_columns(policy_embeddings)
    goal_cols = embedding_columns(goal_embeddings)
    policy_index = policy_embeddings.set_index("abstractPolicyId")
    goal_index = goal_embeddings.set_index("abstractGoalId")
    world_lookup = world_catalog.set_index("worldId", drop=False)
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for world_id, subset in frame.groupby("worldId", dropna=False, sort=True):
        world_id = str(world_id)
        world = world_lookup.loc[world_id].to_dict() if world_id in world_lookup.index else {}
        row_count = int(len(subset))
        policy_ids = sorted(set(subset["primaryPolicyId"].dropna().astype(str)) & set(policy_index.index.astype(str)))
        goal_ids = sorted(set(subset["abstractGoalId"].dropna().astype(str)) & set(goal_index.index.astype(str)))
        profile: dict[str, Any] = {
            "worldId": world_id,
            "title": world.get("title", world_id),
            "experimentId": world.get("experimentId", subset["sourceExperimentId"].mode().iloc[0] if row_count else "unknown"),
            "worldFamily": world.get("worldFamily", subset["worldFamily"].mode().iloc[0] if row_count else "unknown"),
            "substrateClass": world.get("substrateClass", subset["substrateClass"].mode().iloc[0] if row_count else "unknown"),
            "replayability": world.get("replayability", "unknown"),
            "metadataCompleteness": world.get("metadataCompleteness", "unknown"),
            "rowCount": row_count,
            "uniquePolicyCount": int(subset["primaryPolicyId"].nunique()) if row_count else 0,
            "embeddedPolicyCount": len(policy_ids),
            "uniqueGoalCount": int(subset["abstractGoalId"].nunique()) if row_count else 0,
            "embeddedGoalCount": len(goal_ids),
            "uniquePerturbationCount": int(subset["perturbationHash"].nunique()) if row_count else 0,
            "uniqueSourceExperimentCount": int(subset["sourceExperimentId"].nunique()) if row_count else 0,
        }
        profile["feat::coverage::row_count_log1p"] = float(np.log1p(row_count))
        profile["feat::coverage::policy_count_log1p"] = float(np.log1p(profile["uniquePolicyCount"]))
        profile["feat::coverage::embedded_policy_count_log1p"] = float(np.log1p(len(policy_ids)))
        profile["feat::coverage::goal_count_log1p"] = float(np.log1p(profile["uniqueGoalCount"]))
        profile["feat::coverage::embedded_goal_count_log1p"] = float(np.log1p(len(goal_ids)))
        profile["feat::coverage::perturbation_count_log1p"] = float(np.log1p(profile["uniquePerturbationCount"]))

        observed_targets = 0
        target_nonnull_total = 0
        for target in target_columns:
            if target not in subset.columns:
                continue
            values = pd.to_numeric(subset[target], errors="coerce")
            mask = values.notna()
            count = int(mask.sum())
            profile[f"feat::target::{target}::coverage"] = float(count / row_count) if row_count else 0.0
            profile[f"feat::target::{target}::count_log1p"] = float(np.log1p(count))
            if count:
                observed_targets += 1
                target_nonnull_total += count
                z = _weighted_z(values[mask], stats.get(target, {}))
                profile[f"feat::target::{target}::mean_z"] = float(z.mean())
                profile[f"feat::target::{target}::std_z"] = float(z.std(ddof=0)) if count > 1 else 0.0
            else:
                profile[f"feat::target::{target}::mean_z"] = 0.0
                profile[f"feat::target::{target}::std_z"] = 0.0

        if policy_ids:
            policy_subset = policy_index.loc[policy_ids]
            for column in policy_cols:
                values = pd.to_numeric(policy_subset[column], errors="coerce").fillna(0.0)
                profile[f"feat::s06_policy::{column}::mean"] = float(values.mean())
                profile[f"feat::s06_policy::{column}::std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        else:
            for column in policy_cols:
                profile[f"feat::s06_policy::{column}::mean"] = 0.0
                profile[f"feat::s06_policy::{column}::std"] = 0.0

        if goal_ids:
            goal_subset = goal_index.loc[goal_ids]
            for column in goal_cols:
                values = pd.to_numeric(goal_subset[column], errors="coerce").fillna(0.0)
                profile[f"feat::s07_goal::{column}::mean"] = float(values.mean())
                profile[f"feat::s07_goal::{column}::std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        else:
            for column in goal_cols:
                profile[f"feat::s07_goal::{column}::mean"] = 0.0
                profile[f"feat::s07_goal::{column}::std"] = 0.0

        profile["observedTargetCount"] = int(observed_targets)
        profile["targetNonNullTotal"] = int(target_nonnull_total)
        if row_count < min_world_rows or observed_targets == 0:
            status = "insufficient_behavior_rows_or_targets"
        else:
            status = "embedded_behavior_profile"
        profile["embeddingStatus"] = status
        rows.append(profile)
        coverage_rows.append(
            {
                "worldId": world_id,
                "rowCount": row_count,
                "uniquePolicyCount": profile["uniquePolicyCount"],
                "embeddedPolicyCount": len(policy_ids),
                "uniqueGoalCount": profile["uniqueGoalCount"],
                "embeddedGoalCount": len(goal_ids),
                "observedTargetCount": observed_targets,
                "targetNonNullTotal": target_nonnull_total,
                "uniquePerturbationCount": profile["uniquePerturbationCount"],
                "embeddingStatus": status,
            }
        )

    profiles = pd.DataFrame(rows)
    catalog_missing = world_catalog[~world_catalog["worldId"].astype(str).isin(set(profiles["worldId"].astype(str)))]
    for _, world in catalog_missing.iterrows():
        coverage_rows.append(
            {
                "worldId": str(world["worldId"]),
                "rowCount": 0,
                "uniquePolicyCount": 0,
                "embeddedPolicyCount": 0,
                "uniqueGoalCount": 0,
                "embeddedGoalCount": 0,
                "observedTargetCount": 0,
                "targetNonNullTotal": 0,
                "uniquePerturbationCount": 0,
                "embeddingStatus": "unobserved_in_s04",
            }
        )
    return profiles, pd.DataFrame(coverage_rows)


def fit_world_embedding(profiles: pd.DataFrame, n_components: int = 8) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    embedded = profiles[profiles["embeddingStatus"].eq("embedded_behavior_profile")].sort_values("worldId").reset_index(drop=True)
    columns = sorted(column for column in embedded.columns if column.startswith("feat::"))
    x = matrix_from_profile(embedded, columns)
    if x.shape[0] < 2 or x.shape[1] == 0:
        raise ValueError("At least two worlds and one feature are required for S08 world distance profiles")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    components = int(min(n_components, x_scaled.shape[0] - 1, x_scaled.shape[1]))
    pca = PCA(n_components=components, random_state=0)
    matrix = pca.fit_transform(x_scaled)
    metadata_cols = [
        "worldId",
        "title",
        "experimentId",
        "worldFamily",
        "substrateClass",
        "replayability",
        "metadataCompleteness",
        "rowCount",
        "uniquePolicyCount",
        "embeddedPolicyCount",
        "uniqueGoalCount",
        "embeddedGoalCount",
        "uniquePerturbationCount",
        "observedTargetCount",
        "embeddingStatus",
    ]
    rows = embedded[metadata_cols].copy()
    for index in range(components):
        rows[f"embedding_{index:02d}"] = matrix[:, index]
    rows["embeddingX"] = matrix[:, 0]
    rows["embeddingY"] = matrix[:, 1] if components > 1 else 0.0
    rows["featureCount"] = int(len(columns))
    rows["embeddingModel"] = "s05_behavior_profile_plus_s06_s07_context_pca"
    model = {
        "schemaVersion": PLATONIC_DISTANCE_SCHEMA_VERSION,
        "modelVersion": PLATONIC_DISTANCE_MODEL_VERSION,
        "featureColumns": columns,
        "scaler": scaler,
        "pca": pca,
        "explainedVarianceRatio": [float(value) for value in pca.explained_variance_ratio_],
        "worldIds": rows["worldId"].astype(str).tolist(),
        "claimBoundary": PLATONIC_DISTANCE_CLAIM_BOUNDARY,
    }
    return rows, model, matrix


def world_sparse_penalty(world_embeddings: pd.DataFrame, world_coverage: pd.DataFrame) -> pd.DataFrame:
    coverage = world_coverage.set_index("worldId", drop=False)
    rows = world_embeddings[["worldId", "rowCount", "observedTargetCount", "embeddedPolicyCount", "embeddedGoalCount"]].copy()
    rows = rows.set_index("worldId", drop=False)
    for column in ("rowCount", "observedTargetCount", "embeddedPolicyCount", "embeddedGoalCount"):
        if column in coverage.columns:
            rows[column] = pd.to_numeric(coverage.reindex(rows.index)[column], errors="coerce").fillna(rows[column])
    row_quality = _count_quality(rows["rowCount"], "rowCount")
    target_quality = _count_quality(rows["observedTargetCount"], "observedTargetCount")
    policy_quality = _count_quality(rows["embeddedPolicyCount"], "embeddedPolicyCount")
    goal_quality = _count_quality(rows["embeddedGoalCount"], "embeddedGoalCount")
    reliability = (0.40 * row_quality) + (0.25 * target_quality) + (0.20 * policy_quality) + (0.15 * goal_quality)
    penalty = (1.0 - reliability).clip(0.0, 1.0)
    return pd.DataFrame(
        {
            "worldId": rows.index.astype(str),
            "sparseCoveragePenalty": penalty.to_numpy(dtype=float),
            "distanceUncertaintyLevel": [_level_from_penalty(value) for value in penalty],
        }
    ).reset_index(drop=True)


def entity_table(
    embeddings: pd.DataFrame,
    entity_type: str,
    id_column: str,
    label_columns: Sequence[str],
    penalty_frame: pd.DataFrame,
) -> pd.DataFrame:
    cols = embedding_columns(embeddings)
    penalties = penalty_frame.set_index(id_column, drop=False) if id_column in penalty_frame.columns else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in embeddings.iterrows():
        entity_id = str(row[id_column])
        penalty_row = penalties.loc[entity_id].to_dict() if not penalties.empty and entity_id in penalties.index else {}
        record: dict[str, Any] = {
            "entityType": entity_type,
            "entityId": entity_id,
            "entityLabel": _safe_label(row.get(label_columns[0], entity_id)) if label_columns else entity_id,
            "embeddingDimension": len(cols),
            "embeddingModel": _safe_label(row.get("embeddingModel")),
            "sparseCoveragePenalty": _finite_float(penalty_row.get("sparseCoveragePenalty"), default=0.5),
            "distanceUncertaintyLevel": _safe_label(penalty_row.get("distanceUncertaintyLevel", row.get("sparseUncertaintyLevel", "unknown"))),
            "rowCount": int(_finite_float(row.get("rowCount"), default=0.0)),
            "observedTargetCount": int(_finite_float(row.get("observedTargetCount"), default=0.0)),
            "embeddingVectorJson": compact_json([float(row[column]) for column in cols]),
        }
        for column in label_columns:
            record[column] = _safe_label(row.get(column))
        rows.append(record)
    return pd.DataFrame(rows)


def pairwise_distance_tables(
    embeddings: pd.DataFrame,
    entity_type: str,
    id_column: str,
    label_columns: Sequence[str],
    penalty_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    cols = embedding_columns(embeddings)
    if not cols:
        raise ValueError(f"No embedding columns found for {entity_type}")
    ordered = embeddings.sort_values(id_column).reset_index(drop=True)
    ids = ordered[id_column].astype(str).tolist()
    matrix = ordered[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw_euclidean = _euclidean_matrix(matrix)
    max_euclidean = float(np.max(raw_euclidean))
    if not math.isfinite(max_euclidean) or max_euclidean <= 0:
        max_euclidean = 1.0
    normalized_euclidean = raw_euclidean / max_euclidean
    cosine_distance = _cosine_distance_matrix(matrix)
    normalized_cosine = np.clip(cosine_distance / 2.0, 0.0, 1.0)
    penalty_lookup = penalty_frame.set_index(id_column, drop=False) if id_column in penalty_frame.columns else pd.DataFrame()
    penalties = np.asarray(
        [
            _finite_float(
                penalty_lookup.loc[entity_id].get("sparseCoveragePenalty") if not penalty_lookup.empty and entity_id in penalty_lookup.index else 0.5,
                default=0.5,
            )
            for entity_id in ids
        ],
        dtype=float,
    )
    penalty_matrix = (penalties[:, None] + penalties[None, :]) / 2.0
    primary = normalized_euclidean + (SPARSE_COVERAGE_PENALTY_WEIGHT * penalty_matrix)
    np.fill_diagonal(primary, 0.0)
    np.fill_diagonal(raw_euclidean, 0.0)
    np.fill_diagonal(normalized_euclidean, 0.0)
    np.fill_diagonal(cosine_distance, 0.0)
    np.fill_diagonal(normalized_cosine, 0.0)

    entities = entity_table(ordered, entity_type, id_column, label_columns, penalty_frame)
    metadata = ordered.set_index(id_column, drop=False)
    rows: list[dict[str, Any]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            row_a = metadata.loc[ids[i]]
            row_b = metadata.loc[ids[j]]
            record: dict[str, Any] = {
                "entityType": entity_type,
                "entityIdA": ids[i],
                "entityIdB": ids[j],
                "platonicDistance": float(primary[i, j]),
                "normalizedEuclideanDistance": float(normalized_euclidean[i, j]),
                "embeddingEuclideanDistance": float(raw_euclidean[i, j]),
                "normalizedCosineDistance": float(normalized_cosine[i, j]),
                "embeddingCosineDistance": float(cosine_distance[i, j]),
                "coveragePenalty": float(penalty_matrix[i, j]),
                "sparsePenaltyA": float(penalties[i]),
                "sparsePenaltyB": float(penalties[j]),
                "distanceDefinition": "normalized_embedding_euclidean_plus_sparse_coverage_penalty",
                "claimBoundary": PLATONIC_DISTANCE_CLAIM_BOUNDARY,
            }
            for column in label_columns:
                record[f"{column}A"] = _safe_label(row_a.get(column))
                record[f"{column}B"] = _safe_label(row_b.get(column))
                record[f"same_{column}"] = bool(_safe_label(row_a.get(column)) == _safe_label(row_b.get(column)))
            rows.append(record)
    return pd.DataFrame(rows), entities, primary


def _euclidean_matrix(matrix: np.ndarray) -> np.ndarray:
    sq = np.sum(matrix * matrix, axis=1, keepdims=True)
    distances = np.sqrt(np.maximum(sq + sq.T - (2.0 * matrix @ matrix.T), 0.0))
    return distances.astype(float)


def _cosine_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return np.clip(1.0 - (normalized @ normalized.T), 0.0, 2.0)


def nearest_neighbor_table(
    embeddings: pd.DataFrame,
    entity_type: str,
    id_column: str,
    label_columns: Sequence[str],
    distance_matrix: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    ordered = embeddings.sort_values(id_column).reset_index(drop=True)
    ids = ordered[id_column].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for i, entity_id in enumerate(ids):
        order = np.argsort(distance_matrix[i])
        order = [int(value) for value in order if int(value) != i][:k]
        for rank, j in enumerate(order, start=1):
            record: dict[str, Any] = {
                "entityType": entity_type,
                "queryEntityId": entity_id,
                "neighborRank": rank,
                "neighborEntityId": ids[j],
                "platonicDistance": float(distance_matrix[i, j]),
            }
            for column in label_columns:
                record[f"query_{column}"] = _safe_label(ordered.loc[i, column] if column in ordered.columns else None)
                record[f"neighbor_{column}"] = _safe_label(ordered.loc[j, column] if column in ordered.columns else None)
                record[f"same_{column}"] = bool(record[f"query_{column}"] == record[f"neighbor_{column}"])
            rows.append(record)
    return pd.DataFrame(rows)


def symmetry_triangle_diagnostics(
    entity_type: str,
    ids: Sequence[str],
    distance_matrix: np.ndarray,
    sample_limit: int = 25000,
    seed: int = 1708,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    n = len(ids)
    max_asymmetry = float(np.max(np.abs(distance_matrix - distance_matrix.T))) if n else math.nan
    max_diagonal = float(np.max(np.abs(np.diag(distance_matrix)))) if n else math.nan
    rng = np.random.default_rng(seed)
    total_combinations = math.comb(n, 3) if n >= 3 else 0
    sampled = total_combinations > sample_limit
    if n < 3:
        triples: list[tuple[int, int, int]] = []
    elif sampled:
        triples = []
        seen: set[tuple[int, int, int]] = set()
        while len(triples) < sample_limit:
            triple = tuple(sorted(int(value) for value in rng.choice(n, size=3, replace=False)))
            if triple in seen:
                continue
            seen.add(triple)
            triples.append(triple)
    else:
        triples = [(i, j, k) for i in range(n - 2) for j in range(i + 1, n - 1) for k in range(j + 1, n)]

    violations = 0
    checked = 0
    max_violation = 0.0
    for i, j, k in triples:
        checks = (
            float(distance_matrix[i, k] - distance_matrix[i, j] - distance_matrix[j, k]),
            float(distance_matrix[i, j] - distance_matrix[i, k] - distance_matrix[k, j]),
            float(distance_matrix[j, k] - distance_matrix[j, i] - distance_matrix[i, k]),
        )
        for value in checks:
            checked += 1
            if value > tolerance:
                violations += 1
                max_violation = max(max_violation, value)

    return pd.DataFrame(
        [
            {
                "entityType": entity_type,
                "entityCount": int(n),
                "maxSymmetryAbsError": max_asymmetry,
                "maxDiagonalAbsDistance": max_diagonal,
                "triangleCombinationCountAvailable": int(total_combinations),
                "triangleCombinationCountChecked": int(len(triples)),
                "triangleInequalityCountChecked": int(checked),
                "triangleViolationCount": int(violations),
                "triangleViolationRate": float(violations / checked) if checked else 0.0,
                "maxTriangleViolation": float(max_violation),
                "triangleDiagnosticsSampled": bool(sampled),
                "triangleDiagnosticSeed": int(seed),
                "tolerance": float(tolerance),
            }
        ]
    )


def nearest_neighbor_sanity(
    nearest: pd.DataFrame,
    entity_type: str,
    label_columns: Sequence[str],
    ks: Sequence[int] = (1, 5, 10),
) -> pd.DataFrame:
    subset = nearest[nearest["entityType"].eq(entity_type)].copy()
    rows: list[dict[str, Any]] = []
    for column in label_columns:
        same_col = f"same_{column}"
        if same_col not in subset.columns:
            continue
        for k in ks:
            top = subset[subset["neighborRank"].le(k)]
            grouped = top.groupby("queryEntityId", sort=False)[same_col]
            precision = grouped.mean()
            hit = grouped.any()
            rows.append(
                {
                    "entityType": entity_type,
                    "validationTask": "nearest_neighbor_label_sanity",
                    "label": column,
                    "k": int(k),
                    "eligibleQueries": int(len(precision)),
                    "precisionAtK": float(precision.mean()) if len(precision) else math.nan,
                    "hitAtK": float(hit.mean()) if len(hit) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def prior_neighbor_overlap(nearest: pd.DataFrame, prior: pd.DataFrame, entity_type: str, k_values: Sequence[int] = (5, 10)) -> pd.DataFrame:
    if prior.empty:
        return pd.DataFrame()
    if entity_type == "policy":
        query_col, neighbor_col = "queryPolicyId", "neighborPolicyId"
    elif entity_type == "goal":
        query_col, neighbor_col = "queryGoalId", "neighborGoalId"
    else:
        return pd.DataFrame()
    if query_col not in prior.columns or neighbor_col not in prior.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    s08 = nearest[nearest["entityType"].eq(entity_type)]
    for k in k_values:
        overlaps: list[float] = []
        for query_id, query_rows in s08[s08["neighborRank"].le(k)].groupby("queryEntityId"):
            prior_rows = prior[(prior[query_col].astype(str).eq(str(query_id))) & (prior["neighborRank"].le(k))]
            if prior_rows.empty:
                continue
            current_set = set(query_rows["neighborEntityId"].astype(str))
            prior_set = set(prior_rows[neighbor_col].astype(str))
            union = current_set | prior_set
            overlaps.append(len(current_set & prior_set) / len(union) if union else math.nan)
        rows.append(
            {
                "entityType": entity_type,
                "validationTask": "prior_embedding_neighbor_overlap",
                "label": "s06_or_s07_nearest_neighbor_table",
                "k": int(k),
                "eligibleQueries": int(len(overlaps)),
                "meanJaccard": float(np.nanmean(overlaps)) if overlaps else math.nan,
                "medianJaccard": float(np.nanmedian(overlaps)) if overlaps else math.nan,
            }
        )
    return pd.DataFrame(rows)


def distance_baseline_comparison(distances: pd.DataFrame, label_columns_by_type: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entity_type, labels in label_columns_by_type.items():
        subset = distances[distances["entityType"].eq(entity_type)]
        for label in labels:
            same_col = f"same_{label}"
            if same_col not in subset.columns:
                continue
            same = subset[subset[same_col].astype(bool)]["platonicDistance"]
            different = subset[~subset[same_col].astype(bool)]["platonicDistance"]
            label_distance = (~subset[same_col].astype(bool)).astype(float)
            corr = subset["platonicDistance"].corr(label_distance, method="spearman") if len(subset) > 1 else math.nan
            rows.append(
                {
                    "entityType": entity_type,
                    "metadataLabel": label,
                    "pairCount": int(len(subset)),
                    "sameLabelPairCount": int(len(same)),
                    "differentLabelPairCount": int(len(different)),
                    "meanPlatonicDistanceSameLabel": float(same.mean()) if len(same) else math.nan,
                    "meanPlatonicDistanceDifferentLabel": float(different.mean()) if len(different) else math.nan,
                    "sameLabelCloserDelta": float(different.mean() - same.mean()) if len(same) and len(different) else math.nan,
                    "spearmanWithBinaryMetadataDistance": float(corr) if corr == corr else math.nan,
                }
            )
    return pd.DataFrame(rows)


def validate_platonic_distances(
    distances: pd.DataFrame,
    entities: pd.DataFrame,
    diagnostics: pd.DataFrame,
    nearest_sanity: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    world_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(scope: str, check: str, success: bool, severity: str = "error", detail: str = "") -> None:
        checks.append({"scope": scope, "check": check, "success": bool(success), "severity": severity, "detail": detail})

    add("distances", "pairwise_distance_rows_present", len(distances) > 1000, detail=str(len(distances)))
    add("distances", "all_entity_types_present", {"policy", "goal", "world"}.issubset(set(distances["entityType"])), detail=compact_json(sorted(set(distances["entityType"]))))
    add("distances", "distances_finite_nonnegative", np.isfinite(distances["platonicDistance"].to_numpy(dtype=float)).all() and (distances["platonicDistance"] >= 0).all())
    add("entities", "distance_entities_present", {"policy", "goal", "world"}.issubset(set(entities["entityType"])), detail=compact_json(entities["entityType"].value_counts().to_dict()))
    add("worlds", "world_embeddings_present", len(world_embeddings) >= 20, detail=str(len(world_embeddings)))
    add(
        "diagnostics",
        "symmetry_and_diagonal_checks_pass",
        bool((diagnostics["maxSymmetryAbsError"] <= 1e-10).all() and (diagnostics["maxDiagonalAbsDistance"] <= 1e-10).all()),
        detail=diagnostics[["entityType", "maxSymmetryAbsError", "maxDiagonalAbsDistance"]].to_json(orient="records"),
    )
    add(
        "diagnostics",
        "triangle_diagnostics_pass",
        bool((diagnostics["triangleViolationRate"] <= 1e-8).all()),
        detail=diagnostics[["entityType", "triangleViolationRate", "maxTriangleViolation"]].to_json(orient="records"),
    )
    add("nearest_neighbors", "nearest_neighbor_sanity_present", not nearest_sanity.empty, detail=str(len(nearest_sanity)))
    add("baselines", "metadata_distance_comparison_present", not baseline_comparison.empty, detail=str(len(baseline_comparison)))
    overlap = nearest_sanity[nearest_sanity["validationTask"].eq("prior_embedding_neighbor_overlap")] if not nearest_sanity.empty else pd.DataFrame()
    add("nearest_neighbors", "prior_policy_goal_neighbor_overlap_reported", {"policy", "goal"}.issubset(set(overlap["entityType"])) if not overlap.empty else False, severity="warning", detail=str(len(overlap)))
    return pd.DataFrame(checks)
