from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from .behavior_predictor import DEFAULT_TARGET_COLUMNS
from .policy_catalog import safe_json_loads
from .policy_embeddings import (
    matrix_from_profile,
    neighbor_indices,
    random_baseline_matrix,
    stable_bucket,
    target_error_weights,
)
from .world_schema import compact_json, sha256_path


GOAL_EMBEDDING_SCHEMA_VERSION = "e07_s07_goal_embedding.v1"
GOAL_EMBEDDING_MODEL_VERSION = "e07_s07_weighted_goal_behavior_pca.v1"
GOAL_EMBEDDING_CLAIM_BOUNDARY = (
    "Computational goal embeddings learned from S04 simulation-derived behavior summaries, S05 surrogate "
    "error limits, and S06 policy-neighborhood diagnostics only. Goal distances are proxy relationships among "
    "simulated objectives and do not validate biological morphogenesis, cognition, agency, clinical behavior, "
    "sentience, or living chimeras."
)

GOAL_METADATA_COLUMNS = (
    "abstractGoalId",
    "goalLabel",
    "goalFamily",
    "goalKind",
    "representationType",
    "targetDimensionality",
    "localObservability",
    "rowCount",
    "uniquePolicyCount",
    "embeddedPolicyCount",
    "uniqueWorldCount",
    "uniquePerturbationCount",
    "observedTargetCount",
    "embeddingStatus",
    "sparseUncertaintyLevel",
)

CONTEXT_COLUMNS = ("worldFamily", "policyFamily", "sourceKind")


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _safe_label(value: Any) -> str:
    text = _safe_text(value).strip()
    return text if text and text.lower() != "nan" else "unknown"


def _safe_json(value: Any, default: Any) -> Any:
    parsed = safe_json_loads(value, default)
    return default if parsed is None else parsed


def _json_len(value: Any) -> int:
    parsed = _safe_json(value, {})
    if isinstance(parsed, (dict, list)):
        return len(parsed)
    return 0


def _json_list_label(value: Any) -> str:
    parsed = _safe_json(value, [])
    if isinstance(parsed, list):
        return "+".join(str(item) for item in parsed) or "unknown"
    return _safe_label(parsed)


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()[:length]


def _target_stats(target_weights: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(row["target"]): {
            "center": _float_or_nan(row.get("centerMedian")),
            "scale": _float_or_nan(row.get("robustScale")),
            "weight": _float_or_nan(row.get("s05ReliabilityWeight")),
        }
        for row in target_weights.to_dict(orient="records")
    }


def _z_values(values: pd.Series, stats: Mapping[str, float]) -> pd.Series:
    center = stats.get("center", math.nan)
    scale = stats.get("scale", math.nan)
    weight = stats.get("weight", 0.0)
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return ((pd.to_numeric(values, errors="coerce") - center) / scale) * float(weight)


def _goal_lookup(goal_catalog: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if goal_catalog.empty:
        return {}
    return {
        str(row["abstractGoalId"]): row
        for row in goal_catalog.to_dict(orient="records")
        if row.get("abstractGoalId") is not None
    }


def s04_record_summary(corpus: pd.DataFrame) -> pd.DataFrame:
    if corpus.empty or "behaviorRecordId" not in corpus.columns:
        return pd.DataFrame(columns=["behaviorRecordId"])
    rows: list[dict[str, Any]] = []
    for row in corpus.to_dict(orient="records"):
        rows.append(
            {
                "behaviorRecordId": row.get("behaviorRecordId"),
                "metricFieldCount": _json_len(row.get("metricValuesJson")),
                "trajectoryFieldCount": _json_len(row.get("trajectorySummaryJson")),
                "finalStateFieldCount": _json_len(row.get("finalStateJson")),
                "rawTraceLinkCount": _json_len(row.get("rawTraceLinksJson")),
                "missingnessFieldCount": _json_len(row.get("missingnessJson")),
            }
        )
    return pd.DataFrame(rows)


def _merge_s04_summary(frame: pd.DataFrame, corpus: pd.DataFrame | None) -> pd.DataFrame:
    out = frame.copy()
    if corpus is not None and not corpus.empty:
        summary = s04_record_summary(corpus)
        out = out.merge(summary, on="behaviorRecordId", how="left")
    for column in ("metricFieldCount", "trajectoryFieldCount", "finalStateFieldCount", "rawTraceLinkCount", "missingnessFieldCount"):
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out


def _policy_embedding_columns(policy_embeddings: pd.DataFrame) -> list[str]:
    return sorted(column for column in policy_embeddings.columns if column.startswith("embedding_"))


def _policy_neighborhood_stats(nearest_neighbors: pd.DataFrame, goal_policy_ids: set[str]) -> dict[str, float]:
    if nearest_neighbors.empty or not goal_policy_ids:
        return {"hitRate": 0.0, "meanSimilarity": 0.0, "queryCount": 0.0}
    subset = nearest_neighbors[
        nearest_neighbors["queryPolicyId"].astype(str).isin(goal_policy_ids)
        & nearest_neighbors["modelName"].astype(str).eq("weighted_behavior_pca")
    ].copy()
    if subset.empty:
        return {"hitRate": 0.0, "meanSimilarity": 0.0, "queryCount": 0.0}
    subset["neighborInGoal"] = subset["neighborPolicyId"].astype(str).isin(goal_policy_ids).astype(float)
    query = subset.groupby("queryPolicyId").agg(
        hitRate=("neighborInGoal", "mean"),
        meanSimilarity=("cosineSimilarity", "mean"),
    )
    return {
        "hitRate": float(query["hitRate"].mean()) if not query.empty else 0.0,
        "meanSimilarity": float(query["meanSimilarity"].mean()) if not query.empty else 0.0,
        "queryCount": float(len(query)),
    }


def _uncertainty_level(row_count: int, observed_target_count: int, unique_policy_count: int, embedded_policy_count: int) -> tuple[str, float]:
    row_term = 1.0 / math.sqrt(max(row_count, 1))
    target_term = 1.0 / max(observed_target_count, 1)
    policy_term = 1.0 / math.sqrt(max(unique_policy_count, 1))
    embedded_penalty = 0.25 if embedded_policy_count < max(2, min(unique_policy_count, 2)) else 0.0
    score = float(row_term + target_term + policy_term + embedded_penalty)
    if row_count < 50 or observed_target_count < 2 or unique_policy_count < 3 or embedded_policy_count < 2:
        return "high", score
    if row_count < 200 or observed_target_count < 4 or unique_policy_count < 5:
        return "moderate", score
    return "lower", score


def build_goal_profiles(
    frame: pd.DataFrame,
    goal_catalog: pd.DataFrame,
    policy_embeddings: pd.DataFrame,
    nearest_neighbors: pd.DataFrame,
    target_columns: Sequence[str],
    target_weights: pd.DataFrame,
    corpus: pd.DataFrame | None = None,
    min_goal_rows: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _merge_s04_summary(frame, corpus)
    stats = _target_stats(target_weights)
    goal_lookup = _goal_lookup(goal_catalog)
    policy_embedding_cols = _policy_embedding_columns(policy_embeddings)
    policy_emb_index = policy_embeddings.set_index("abstractPolicyId", drop=False) if not policy_embeddings.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []

    for goal_id in sorted(set(goal_catalog["abstractGoalId"].astype(str)) | set(frame["abstractGoalId"].astype(str))):
        goal = goal_lookup.get(goal_id, {})
        subset = frame[frame["abstractGoalId"].astype(str).eq(goal_id)].copy()
        row_count = int(len(subset))
        policy_ids = sorted(set(subset["primaryPolicyId"].astype(str))) if row_count else []
        embedded_policy_ids = [policy_id for policy_id in policy_ids if policy_id in policy_emb_index.index]
        embedded_policy_row_fraction = float(subset["primaryPolicyId"].astype(str).isin(embedded_policy_ids).mean()) if row_count else 0.0
        uncertainty_level, uncertainty_score = _uncertainty_level(row_count, 0, len(policy_ids), len(embedded_policy_ids))

        profile: dict[str, Any] = {
            "abstractGoalId": goal_id,
            "goalLabel": goal.get("goalLabel", goal_id),
            "goalFamily": goal.get("goalFamily", "unknown"),
            "goalKind": goal.get("goalKind", "unknown"),
            "representationType": goal.get("representationType", "unknown"),
            "targetDimensionality": _json_list_label(goal.get("targetDimensionality")),
            "localObservability": goal.get("localObservability", "unknown"),
            "rowCount": row_count,
            "uniquePolicyCount": int(len(policy_ids)),
            "embeddedPolicyCount": int(len(embedded_policy_ids)),
            "embeddedPolicyRowFraction": embedded_policy_row_fraction,
            "uniqueWorldCount": int(subset["worldId"].nunique()) if row_count else 0,
            "uniquePerturbationCount": int(subset["perturbationHash"].nunique()) if row_count else 0,
            "uniqueSourceExperimentCount": int(subset["sourceExperimentId"].nunique()) if row_count else 0,
            "linkedWorldCount": int(goal.get("linkedWorldCount", 0) or 0),
            "linkedPolicyCount": int(goal.get("linkedPolicyCount", 0) or 0),
        }
        profile["feat::coverage::row_count_log1p"] = float(np.log1p(row_count))
        profile["feat::coverage::policy_count_log1p"] = float(np.log1p(len(policy_ids)))
        profile["feat::coverage::embedded_policy_count_log1p"] = float(np.log1p(len(embedded_policy_ids)))
        profile["feat::coverage::world_count_log1p"] = float(np.log1p(profile["uniqueWorldCount"]))
        profile["feat::coverage::perturbation_count_log1p"] = float(np.log1p(profile["uniquePerturbationCount"]))
        profile["feat::coverage::embedded_policy_row_fraction"] = embedded_policy_row_fraction
        profile["feat::catalog::linked_world_count_log1p"] = float(np.log1p(profile["linkedWorldCount"]))
        profile["feat::catalog::linked_policy_count_log1p"] = float(np.log1p(profile["linkedPolicyCount"]))

        observed_targets = 0
        target_nonnull_total = 0
        mean_standard_error_terms: list[float] = []
        if row_count:
            for target in target_columns:
                if target not in subset.columns:
                    continue
                values = pd.to_numeric(subset[target], errors="coerce")
                mask = values.notna()
                count = int(mask.sum())
                profile[f"feat::target::{target}::count_log1p"] = float(np.log1p(count))
                profile[f"feat::target::{target}::coverage"] = float(count / row_count)
                if count:
                    observed_targets += 1
                    target_nonnull_total += count
                    z = _z_values(values[mask], stats[target])
                    profile[f"feat::target::{target}::mean_z"] = float(z.mean())
                    profile[f"feat::target::{target}::std_z"] = float(z.std(ddof=0)) if count > 1 else 0.0
                    mean_standard_error_terms.append(float(1.0 / math.sqrt(count)))
                    for context_col in CONTEXT_COLUMNS:
                        context_subset = subset.loc[mask, [context_col, target]].copy()
                        context_subset["_z"] = z.to_numpy()
                        for context_value, context_rows in context_subset.groupby(context_col, dropna=False):
                            value_label = _safe_label(context_value).replace("|", "_").replace("=", "_")
                            profile[f"feat::context::{context_col}={value_label}::{target}::mean_z"] = float(context_rows["_z"].mean())
                else:
                    profile[f"feat::target::{target}::mean_z"] = 0.0
                    profile[f"feat::target::{target}::std_z"] = 0.0

            for column in ("metricFieldCount", "trajectoryFieldCount", "finalStateFieldCount", "rawTraceLinkCount", "missingnessFieldCount"):
                profile[f"feat::s04::{column}::mean"] = float(pd.to_numeric(subset[column], errors="coerce").fillna(0.0).mean())

            for prefix_col in ("policyFamily", "worldFamily", "sourceExperimentId"):
                counts = subset[prefix_col].map(_safe_label).value_counts(normalize=True)
                for value, fraction in counts.items():
                    profile[f"feat::{prefix_col}::{value}::fraction"] = float(fraction)

        if embedded_policy_ids:
            emb_subset = policy_emb_index.loc[embedded_policy_ids]
            for column in policy_embedding_cols:
                values = pd.to_numeric(emb_subset[column], errors="coerce").fillna(0.0)
                profile[f"feat::s06_policy_embedding::{column}::mean"] = float(values.mean())
                profile[f"feat::s06_policy_embedding::{column}::std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        else:
            for column in policy_embedding_cols:
                profile[f"feat::s06_policy_embedding::{column}::mean"] = 0.0
                profile[f"feat::s06_policy_embedding::{column}::std"] = 0.0
        neighborhood = _policy_neighborhood_stats(nearest_neighbors, set(embedded_policy_ids))
        profile["policyNeighborhoodHitRateK10"] = neighborhood["hitRate"]
        profile["policyNeighborhoodMeanSimilarityK10"] = neighborhood["meanSimilarity"]
        profile["policyNeighborhoodQueryCount"] = neighborhood["queryCount"]
        profile["feat::s06_policy_neighborhood::hit_rate_k10"] = neighborhood["hitRate"]
        profile["feat::s06_policy_neighborhood::mean_similarity_k10"] = neighborhood["meanSimilarity"]
        profile["feat::s06_policy_neighborhood::query_count_log1p"] = float(np.log1p(neighborhood["queryCount"]))

        uncertainty_level, uncertainty_score = _uncertainty_level(row_count, observed_targets, len(policy_ids), len(embedded_policy_ids))
        profile["observedTargetCount"] = int(observed_targets)
        profile["targetNonNullTotal"] = int(target_nonnull_total)
        profile["meanTargetStandardErrorProxy"] = float(np.mean(mean_standard_error_terms)) if mean_standard_error_terms else math.nan
        profile["sparseUncertaintyLevel"] = uncertainty_level
        profile["sparseUncertaintyScore"] = uncertainty_score
        if row_count == 0:
            status = "unobserved_in_s04"
        elif row_count < min_goal_rows or observed_targets == 0:
            status = "insufficient_behavior_rows_or_targets"
        else:
            status = "embedded_behavior"
        profile["embeddingStatus"] = status
        rows.append(profile)
        coverage_rows.append(
            {
                "abstractGoalId": goal_id,
                "rowCount": row_count,
                "uniquePolicyCount": int(len(policy_ids)),
                "embeddedPolicyCount": int(len(embedded_policy_ids)),
                "embeddedPolicyRowFraction": embedded_policy_row_fraction,
                "observedTargetCount": int(observed_targets),
                "targetNonNullTotal": int(target_nonnull_total),
                "uniqueWorldCount": int(profile["uniqueWorldCount"]),
                "uniquePerturbationCount": int(profile["uniquePerturbationCount"]),
                "embeddingStatus": status,
            }
        )
        uncertainty_rows.append(
            {
                "abstractGoalId": goal_id,
                "rowCount": row_count,
                "uniquePolicyCount": int(len(policy_ids)),
                "embeddedPolicyCount": int(len(embedded_policy_ids)),
                "observedTargetCount": int(observed_targets),
                "targetNonNullTotal": int(target_nonnull_total),
                "meanTargetStandardErrorProxy": profile["meanTargetStandardErrorProxy"],
                "sparseUncertaintyScore": uncertainty_score,
                "sparseUncertaintyLevel": uncertainty_level,
                "uncertaintyReason": _uncertainty_reason(row_count, observed_targets, len(policy_ids), len(embedded_policy_ids)),
            }
        )

    profiles = pd.DataFrame(rows)
    text_columns = ("abstractGoalId", "goalLabel", "goalFamily", "goalKind", "representationType", "targetDimensionality", "localObservability", "embeddingStatus", "sparseUncertaintyLevel")
    profiles = profiles.fillna({column: "unknown" for column in text_columns if column in profiles.columns})
    return profiles, pd.DataFrame(coverage_rows), pd.DataFrame(uncertainty_rows)


def _uncertainty_reason(row_count: int, observed_target_count: int, unique_policy_count: int, embedded_policy_count: int) -> str:
    reasons: list[str] = []
    if row_count == 0:
        reasons.append("not_observed_in_s04")
    elif row_count < 50:
        reasons.append("few_behavior_rows")
    if observed_target_count < 2:
        reasons.append("few_s05_targets")
    if unique_policy_count < 3:
        reasons.append("few_policies")
    if embedded_policy_count < 2:
        reasons.append("few_s06_embedded_policies")
    return ",".join(reasons) if reasons else "adequate_for_s07_proxy_embedding"


def feature_columns(profile: pd.DataFrame, prefixes: Sequence[str] = ("feat::",)) -> list[str]:
    return sorted(column for column in profile.columns if any(column.startswith(prefix) for prefix in prefixes))


def fit_goal_embedding(
    profile: pd.DataFrame,
    columns: Sequence[str],
    n_components: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    x = matrix_from_profile(profile, columns)
    if x.shape[0] < 2 or x.shape[1] == 0:
        raise ValueError("At least two goals and one feature are required for S07 goal embedding")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    components = int(min(n_components, x_scaled.shape[0] - 1, x_scaled.shape[1]))
    pca = PCA(n_components=components, random_state=0)
    embedding = pca.fit_transform(x_scaled)
    rows = profile[list(GOAL_METADATA_COLUMNS)].copy()
    for index in range(components):
        rows[f"embedding_{index:02d}"] = embedding[:, index]
    rows["embeddingX"] = embedding[:, 0]
    rows["embeddingY"] = embedding[:, 1] if components > 1 else 0.0
    rows["featureCount"] = int(len(columns))
    rows["embeddingModel"] = "weighted_goal_behavior_pca"
    model = {
        "schemaVersion": GOAL_EMBEDDING_SCHEMA_VERSION,
        "modelVersion": GOAL_EMBEDDING_MODEL_VERSION,
        "featureColumns": list(columns),
        "scaler": scaler,
        "pca": pca,
        "explainedVarianceRatio": [float(value) for value in pca.explained_variance_ratio_],
        "goalIds": rows["abstractGoalId"].astype(str).tolist(),
        "claimBoundary": GOAL_EMBEDDING_CLAIM_BOUNDARY,
    }
    return rows, model, embedding


def goal_metadata_baseline_matrix(goal_catalog: pd.DataFrame, goal_ids: Sequence[str], n_components: int = 8) -> tuple[np.ndarray, dict[str, Any]]:
    indexed = goal_catalog.set_index("abstractGoalId", drop=False)
    records: list[dict[str, Any]] = []
    for goal_id in goal_ids:
        row = indexed.loc[goal_id].to_dict() if goal_id in indexed.index else {"abstractGoalId": goal_id}
        record: dict[str, Any] = {}
        for column in ("goalFamily", "goalKind", "representationType", "localObservability"):
            record[f"{column}={_safe_label(row.get(column))}"] = 1.0
        record[f"targetDimensionality={_json_list_label(row.get('targetDimensionality'))}"] = 1.0
        record[f"sourceExperimentIds={_json_list_label(row.get('sourceExperimentIds'))}"] = 1.0
        for column in ("linkedWorldCount", "linkedPolicyCount"):
            value = _float_or_nan(row.get(column))
            record[column] = 0.0 if math.isnan(value) else float(np.log1p(value))
        records.append(record)
    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform(records)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    components = int(min(n_components, x_scaled.shape[0] - 1, x_scaled.shape[1]))
    pca = PCA(n_components=components, random_state=0)
    matrix = pca.fit_transform(x_scaled)
    return matrix, {"vectorizer": vectorizer, "scaler": scaler, "pca": pca, "featureNames": vectorizer.get_feature_names_out().tolist()}


def policy_neighborhood_baseline_matrix(profile: pd.DataFrame, n_components: int = 8) -> tuple[np.ndarray, dict[str, Any]]:
    columns = feature_columns(profile, prefixes=("feat::s06_policy_embedding::", "feat::s06_policy_neighborhood::", "feat::policyFamily::"))
    x = matrix_from_profile(profile, columns)
    if x.shape[1] == 0:
        return np.zeros((len(profile), 1), dtype=float), {"featureColumns": []}
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    components = int(min(n_components, x_scaled.shape[0] - 1, x_scaled.shape[1]))
    pca = PCA(n_components=components, random_state=0)
    matrix = pca.fit_transform(x_scaled)
    return matrix, {"featureColumns": columns, "scaler": scaler, "pca": pca}


def nearest_neighbor_table(goal_ids: Sequence[str], matrix: np.ndarray, model_name: str, k: int = 10) -> pd.DataFrame:
    order, scores = neighbor_indices(matrix, k=k)
    ids = list(goal_ids)
    rows: list[dict[str, Any]] = []
    for query_index, goal_id in enumerate(ids):
        for rank, (neighbor_index, score) in enumerate(zip(order[query_index], scores[query_index]), start=1):
            rows.append(
                {
                    "modelName": model_name,
                    "queryGoalId": goal_id,
                    "neighborRank": rank,
                    "neighborGoalId": ids[int(neighbor_index)],
                    "cosineSimilarity": float(score),
                }
            )
    return pd.DataFrame(rows)


def label_retrieval_metrics(
    goal_ids: Sequence[str],
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    model_name: str,
    labels: Sequence[str] = ("goalFamily", "representationType", "targetDimensionality"),
    ks: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    order, _ = neighbor_indices(matrix, k=max(ks))
    meta = metadata.set_index("abstractGoalId")
    ids = list(goal_ids)
    rows: list[dict[str, Any]] = []
    for label in labels:
        values = meta.reindex(ids)[label].map(_safe_label).tolist()
        for k in ks:
            eligible = 0
            precisions: list[float] = []
            hits: list[float] = []
            for index, value in enumerate(values):
                if value == "unknown" or values.count(value) <= 1:
                    continue
                neighbors = order[index, :k]
                same = [1.0 if values[int(neighbor)] == value else 0.0 for neighbor in neighbors]
                eligible += 1
                precisions.append(float(np.mean(same)))
                hits.append(float(any(same)))
            rows.append(
                {
                    "validationTask": "goal_label_neighbor_retrieval",
                    "modelName": model_name,
                    "label": label,
                    "k": int(k),
                    "eligibleQueries": int(eligible),
                    "precisionAtK": float(np.mean(precisions)) if precisions else math.nan,
                    "hitAtK": float(np.mean(hits)) if hits else math.nan,
                }
            )
    return pd.DataFrame(rows)


def heldout_truth_columns(profile: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in profile.columns
        if column.startswith("feat::")
        and column.endswith("::mean_z")
        and (column.startswith("feat::target::") or column.startswith("feat::context::"))
    )


def heldout_goal_retrieval_metrics(
    goal_ids: Sequence[str],
    model_matrices: Mapping[str, np.ndarray],
    truth_profile: pd.DataFrame,
    truth_columns: Sequence[str],
    ks: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    ids = list(goal_ids)
    truth = truth_profile.set_index("abstractGoalId").reindex(ids)
    truth_matrix = matrix_from_profile(truth.reset_index(), truth_columns)
    truth_norms = np.linalg.norm(truth_matrix, axis=1)
    rows: list[dict[str, Any]] = []
    for model_name, matrix in model_matrices.items():
        order, _ = neighbor_indices(matrix, k=max(ks))
        for k in ks:
            scores: list[float] = []
            evaluated = 0
            for index in range(len(ids)):
                if truth_norms[index] == 0:
                    continue
                neighbor_truth = truth_matrix[order[index, :k]]
                pred = np.nanmean(neighbor_truth, axis=0)
                pred_norm = float(np.linalg.norm(pred))
                if pred_norm == 0:
                    continue
                score = float(np.dot(truth_matrix[index], pred) / (truth_norms[index] * pred_norm))
                if math.isfinite(score):
                    scores.append(score)
                    evaluated += 1
            rows.append(
                {
                    "validationTask": "heldout_goal_behavior_profile_retrieval",
                    "modelName": model_name,
                    "k": int(k),
                    "eligibleQueries": int(evaluated),
                    "meanCosineToHeldoutProfile": float(np.mean(scores)) if scores else math.nan,
                    "medianCosineToHeldoutProfile": float(np.median(scores)) if scores else math.nan,
                }
            )
    return pd.DataFrame(rows)


def split_behavior_frame(frame: pd.DataFrame, salt: str = "s07_goal_embedding") -> tuple[pd.DataFrame, pd.DataFrame]:
    buckets = frame["behaviorRecordId"].map(lambda value: stable_bucket(value, salt=salt, modulus=2))
    return frame[buckets.eq(0)].copy(), frame[buckets.eq(1)].copy()


def stability_metrics(
    goal_ids: Sequence[str],
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    k: int = 5,
) -> pd.DataFrame:
    ids = list(goal_ids)
    order_a, _ = neighbor_indices(embedding_a, k=k)
    order_b, _ = neighbor_indices(embedding_b, k=k)
    jaccards: list[float] = []
    for index in range(len(ids)):
        set_a = set(int(value) for value in order_a[index])
        set_b = set(int(value) for value in order_b[index])
        union = set_a | set_b
        jaccards.append(len(set_a & set_b) / len(union) if union else math.nan)
    norm_a = _normalize(embedding_a)
    norm_b = _normalize(embedding_b)
    dist_a = 1.0 - norm_a @ norm_a.T
    dist_b = 1.0 - norm_b @ norm_b.T
    pairs = np.transpose(np.triu_indices(len(ids), k=1))
    a_values = np.asarray([dist_a[i, j] for i, j in pairs], dtype=float)
    b_values = np.asarray([dist_b[i, j] for i, j in pairs], dtype=float)
    spearman = pd.Series(a_values).corr(pd.Series(b_values), method="spearman") if len(a_values) > 1 else math.nan
    return pd.DataFrame(
        [
            {
                "validationTask": "goal_split_half_embedding_stability",
                "modelName": "weighted_goal_behavior_pca",
                "goalCount": int(len(ids)),
                "neighborK": int(k),
                "meanNeighborJaccard": float(np.nanmean(jaccards)) if jaccards else math.nan,
                "medianNeighborJaccard": float(np.nanmedian(jaccards)) if jaccards else math.nan,
                "pairCount": int(len(a_values)),
                "pairwiseDistanceSpearman": float(spearman) if spearman == spearman else math.nan,
            }
        ]
    )


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def baseline_comparison(metrics: pd.DataFrame, primary_model: str = "weighted_goal_behavior_pca") -> pd.DataFrame:
    heldout = metrics[metrics["validationTask"].eq("heldout_goal_behavior_profile_retrieval")]
    primary = heldout[heldout["modelName"].eq(primary_model)]
    rows: list[dict[str, Any]] = []
    for _, primary_row in primary.iterrows():
        subset = heldout[heldout["k"].eq(primary_row["k"]) & ~heldout["modelName"].eq(primary_model)]
        for _, baseline in subset.iterrows():
            rows.append(
                {
                    "validationTask": primary_row["validationTask"],
                    "k": int(primary_row["k"]),
                    "baselineName": baseline["modelName"],
                    "goalEmbeddingMeanCosine": primary_row["meanCosineToHeldoutProfile"],
                    "baselineMeanCosine": baseline["meanCosineToHeldoutProfile"],
                    "cosineImprovement": primary_row["meanCosineToHeldoutProfile"] - baseline["meanCosineToHeldoutProfile"],
                    "goalEmbeddingBeatsBaseline": bool(primary_row["meanCosineToHeldoutProfile"] > baseline["meanCosineToHeldoutProfile"]),
                }
            )
    return pd.DataFrame(rows)


def validate_goal_embeddings(
    embeddings: pd.DataFrame,
    coverage: pd.DataFrame,
    uncertainty: pd.DataFrame,
    retrieval_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    model_path: str | Path,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(scope: str, check: str, success: bool, severity: str = "error", detail: str = "") -> None:
        checks.append({"scope": scope, "check": check, "success": bool(success), "severity": severity, "detail": detail})

    add("embeddings", "goal_embedding_rows_present", len(embeddings) >= 20, detail=str(len(embeddings)))
    add("embeddings", "embedding_coordinates_finite", np.isfinite(embeddings.filter(regex=r"^embedding_").to_numpy(dtype=float)).all(), detail=str(embeddings.filter(regex=r"^embedding_").shape))
    observed_count = int((coverage["rowCount"] > 0).sum()) if not coverage.empty else 0
    embedded_count = int(coverage["embeddingStatus"].eq("embedded_behavior").sum()) if not coverage.empty else 0
    add("coverage", "observed_goals_documented", observed_count >= len(embeddings), detail=str(observed_count))
    add("coverage", "embedded_goal_count_matches", embedded_count == len(embeddings), detail=f"{embedded_count} embedded, {len(embeddings)} rows")
    add("uncertainty", "all_catalog_goals_have_uncertainty_rows", len(uncertainty) >= len(coverage), detail=f"{len(uncertainty)} uncertainty rows, {len(coverage)} coverage rows")
    models = set(retrieval_metrics["modelName"].astype(str)) if not retrieval_metrics.empty else set()
    add("validation", "retrieval_metrics_include_goal_and_baselines", {"weighted_goal_behavior_pca", "goal_metadata_baseline", "policy_neighborhood_baseline"}.issubset(models), detail=compact_json(sorted(models)))
    heldout = retrieval_metrics[retrieval_metrics["validationTask"].eq("heldout_goal_behavior_profile_retrieval")] if not retrieval_metrics.empty else pd.DataFrame()
    add("validation", "heldout_goal_retrieval_present", not heldout.empty, detail=str(len(heldout)))
    add("validation", "stability_metrics_present", not stability.empty, detail=str(len(stability)))
    if not stability.empty:
        value = float(stability["pairwiseDistanceSpearman"].iloc[0])
        add("validation", "split_half_distance_correlation_positive", value > 0, severity="warning", detail=str(value))
    add("model", "embedding_model_written", Path(model_path).exists(), detail=str(model_path))
    return pd.DataFrame(checks)


def performance_summary(retrieval_metrics: pd.DataFrame, comparisons: pd.DataFrame, stability: pd.DataFrame, uncertainty: pd.DataFrame) -> dict[str, Any]:
    heldout = retrieval_metrics[retrieval_metrics["validationTask"].eq("heldout_goal_behavior_profile_retrieval")] if not retrieval_metrics.empty else pd.DataFrame()
    k3 = heldout[heldout["k"].eq(3)] if not heldout.empty else pd.DataFrame()
    behavior_k3 = k3[k3["modelName"].eq("weighted_goal_behavior_pca")]
    comparisons_k3 = comparisons[comparisons["k"].eq(3)] if not comparisons.empty else pd.DataFrame()
    return {
        "retrievalMetricRows": int(len(retrieval_metrics)),
        "heldoutGoalMetricRows": int(len(heldout)),
        "goalK3MeanCosine": float(behavior_k3["meanCosineToHeldoutProfile"].iloc[0]) if not behavior_k3.empty else math.nan,
        "k3BaselineComparisonsWon": int(comparisons_k3["goalEmbeddingBeatsBaseline"].sum()) if not comparisons_k3.empty else 0,
        "k3BaselineComparisonCount": int(len(comparisons_k3)),
        "splitHalfDistanceSpearman": float(stability["pairwiseDistanceSpearman"].iloc[0]) if not stability.empty else math.nan,
        "splitHalfMeanNeighborJaccard": float(stability["meanNeighborJaccard"].iloc[0]) if not stability.empty else math.nan,
        "highUncertaintyGoalCount": int(uncertainty["sparseUncertaintyLevel"].eq("high").sum()) if not uncertainty.empty else 0,
        "moderateUncertaintyGoalCount": int(uncertainty["sparseUncertaintyLevel"].eq("moderate").sum()) if not uncertainty.empty else 0,
    }


def artifact_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)}
