from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .behavior_corpus import compact_json
from .behavior_predictor import DEFAULT_TARGET_COLUMNS, NUMERIC_FEATURE_COLUMNS
from .policy_catalog import safe_json_loads
from .world_schema import sha256_path


POLICY_EMBEDDING_SCHEMA_VERSION = "e07_s06_policy_embedding.v1"
POLICY_EMBEDDING_MODEL_VERSION = "e07_s06_weighted_behavior_pca.v1"
POLICY_EMBEDDING_CLAIM_BOUNDARY = (
    "Computational policy embeddings learned from S04 simulation-derived behavior summaries and S05 surrogate "
    "error limits only. Distances are proxy relationships among algorithms in these simulated worlds; they do "
    "not validate biological morphogenesis, cognition, agency, clinical behavior, sentience, or living chimeras."
)

POLICY_METADATA_COLUMNS = (
    "abstractPolicyId",
    "sourcePolicyId",
    "policyLabel",
    "sourceExperimentId",
    "policyFamily",
    "policyKind",
    "representationType",
    "representationHash",
    "stochasticity",
    "memoryDepth",
    "signalingHorizon",
    "complexityScore",
    "dslAvailable",
    "automatonAvailable",
    "decisionGraphAvailable",
    "neuralMetadataAvailable",
    "sourceRowCount",
)

PROFILE_METADATA_COLUMNS = (
    "abstractPolicyId",
    "policyLabel",
    "sourceExperimentId",
    "policyFamily",
    "policyKind",
    "representationType",
    "stochasticity",
    "rowCount",
    "uniqueWorldCount",
    "uniqueGoalCount",
    "uniquePerturbationCount",
    "uniqueSourceExperimentCount",
    "observedTargetCount",
    "embeddingStatus",
)

CONTEXT_COLUMNS = ("worldFamily", "goalFamily", "sourceKind")


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()[:length]


def stable_bucket(value: Any, salt: str, modulus: int = 2) -> int:
    return int(hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:8], 16) % modulus


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


def _json_text(value: Any) -> str:
    parsed = safe_json_loads(value, value)
    return compact_json(parsed) if isinstance(parsed, (dict, list)) else _safe_text(parsed)


def target_error_weights(
    frame: pd.DataFrame,
    error_limits: pd.DataFrame,
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in target_columns:
        if target not in frame.columns:
            values = pd.Series(dtype=float)
        else:
            values = pd.to_numeric(frame[target], errors="coerce").dropna()
        if values.empty:
            center = math.nan
            scale = math.nan
            available = 0
        else:
            q25, q75 = values.quantile([0.25, 0.75])
            iqr = float(q75 - q25)
            std = float(values.std(ddof=0))
            scale = max(iqr, std, 1e-9)
            center = float(values.median())
            available = int(len(values))

        err = error_limits[error_limits["target"].astype(str).eq(target)] if not error_limits.empty else pd.DataFrame()
        test_err = err[err["partition"].astype(str).eq("test")] if "partition" in err.columns else err
        rmse_values = pd.to_numeric(test_err.get("rmse", pd.Series(dtype=float)), errors="coerce").dropna()
        median_rmse = float(rmse_values.median()) if not rmse_values.empty else math.nan
        if available == 0 or not math.isfinite(scale):
            weight = 0.0
        elif not math.isfinite(median_rmse):
            weight = 0.5
        else:
            weight = max(0.05, min(1.0, scale / (scale + max(median_rmse, 0.0))))
        rows.append(
            {
                "target": target,
                "availableRows": available,
                "centerMedian": center,
                "robustScale": scale,
                "s05MedianTestRmse": median_rmse,
                "s05ReliabilityWeight": float(weight),
            }
        )
    return pd.DataFrame(rows)


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


def _policy_lookup(policy_catalog: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if policy_catalog.empty:
        return {}
    return {
        str(row["abstractPolicyId"]): row
        for row in policy_catalog[list(policy_catalog.columns)].to_dict(orient="records")
        if row.get("abstractPolicyId") is not None
    }


def build_policy_profiles(
    frame: pd.DataFrame,
    policy_catalog: pd.DataFrame,
    target_columns: Sequence[str],
    target_weights: pd.DataFrame,
    min_policy_rows: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = _target_stats(target_weights)
    policy_lookup = _policy_lookup(policy_catalog)
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for policy_id, subset in frame.groupby("primaryPolicyId", sort=True):
        policy_id = str(policy_id)
        policy = policy_lookup.get(policy_id, {})
        profile: dict[str, Any] = {
            "abstractPolicyId": policy_id,
            "policyLabel": policy.get("policyLabel", policy_id),
            "sourceExperimentId": policy.get("sourceExperimentId", subset["sourceExperimentId"].mode().iloc[0] if len(subset) else "unknown"),
            "policyFamily": policy.get("policyFamily", subset["policyFamily"].mode().iloc[0] if len(subset) else "unknown"),
            "policyKind": policy.get("policyKind", subset["policyKind"].mode().iloc[0] if len(subset) else "unknown"),
            "representationType": policy.get("representationType", subset["policyRepresentationType"].mode().iloc[0] if len(subset) else "unknown"),
            "stochasticity": policy.get("stochasticity", subset["policyStochasticity"].mode().iloc[0] if len(subset) else "unknown"),
            "rowCount": int(len(subset)),
            "uniqueWorldCount": int(subset["worldId"].nunique()),
            "uniqueGoalCount": int(subset["abstractGoalId"].nunique()),
            "uniquePerturbationCount": int(subset["perturbationHash"].nunique()),
            "uniqueSourceExperimentCount": int(subset["sourceExperimentId"].nunique()),
        }
        profile["feat::coverage::row_count_log1p"] = float(np.log1p(len(subset)))
        profile["feat::coverage::world_count_log1p"] = float(np.log1p(profile["uniqueWorldCount"]))
        profile["feat::coverage::goal_count_log1p"] = float(np.log1p(profile["uniqueGoalCount"]))
        profile["feat::coverage::perturbation_count_log1p"] = float(np.log1p(profile["uniquePerturbationCount"]))
        profile["feat::coverage::source_experiment_count"] = float(profile["uniqueSourceExperimentCount"])

        observed_targets = 0
        target_nonnull_total = 0
        for target in target_columns:
            if target not in subset.columns:
                continue
            values = pd.to_numeric(subset[target], errors="coerce")
            mask = values.notna()
            count = int(mask.sum())
            if count:
                observed_targets += 1
                target_nonnull_total += count
                z = _z_values(values[mask], stats[target])
                profile[f"feat::target::{target}::count_log1p"] = float(np.log1p(count))
                profile[f"feat::target::{target}::coverage"] = float(count / len(subset))
                profile[f"feat::target::{target}::mean_z"] = float(z.mean())
                profile[f"feat::target::{target}::std_z"] = float(z.std(ddof=0)) if count > 1 else 0.0
                for context_col in CONTEXT_COLUMNS:
                    if context_col not in subset.columns:
                        continue
                    context_subset = subset.loc[mask, [context_col, target]].copy()
                    context_subset["_z"] = z.to_numpy()
                    for context_value, context_rows in context_subset.groupby(context_col, dropna=False):
                        value_label = _safe_label(context_value).replace("|", "_").replace("=", "_")
                        feature = f"feat::context::{context_col}={value_label}::{target}::mean_z"
                        profile[feature] = float(context_rows["_z"].mean())
            else:
                profile[f"feat::target::{target}::count_log1p"] = 0.0
                profile[f"feat::target::{target}::coverage"] = 0.0

        for column in NUMERIC_FEATURE_COLUMNS:
            if column in subset.columns:
                values = pd.to_numeric(subset[column], errors="coerce").dropna()
                profile[f"feat::s05::{column}::mean"] = float(values.mean()) if not values.empty else 0.0

        for column in ("memoryDepth", "signalingHorizon", "complexityScore", "sourceRowCount"):
            profile[f"feat::catalog::{column}"] = _float_or_nan(policy.get(column))
        for column in ("dslAvailable", "automatonAvailable", "decisionGraphAvailable", "neuralMetadataAvailable"):
            profile[f"feat::catalog::{column}"] = float(bool(policy.get(column))) if column in policy else 0.0

        profile["observedTargetCount"] = int(observed_targets)
        profile["targetNonNullTotal"] = int(target_nonnull_total)
        profile["embeddingStatus"] = "embedded_behavior" if len(subset) >= min_policy_rows and observed_targets > 0 else "insufficient_behavior_rows"
        rows.append(profile)
        coverage_rows.append(
            {
                "abstractPolicyId": policy_id,
                "rowCount": int(len(subset)),
                "observedTargetCount": int(observed_targets),
                "targetNonNullTotal": int(target_nonnull_total),
                "uniqueWorldCount": int(profile["uniqueWorldCount"]),
                "uniqueGoalCount": int(profile["uniqueGoalCount"]),
                "uniquePerturbationCount": int(profile["uniquePerturbationCount"]),
                "embeddingStatus": profile["embeddingStatus"],
            }
        )

    observed_ids = set(frame["primaryPolicyId"].astype(str).unique())
    for policy in policy_catalog.to_dict(orient="records"):
        policy_id = str(policy.get("abstractPolicyId"))
        if policy_id not in observed_ids:
            coverage_rows.append(
                {
                    "abstractPolicyId": policy_id,
                    "rowCount": 0,
                    "observedTargetCount": 0,
                    "targetNonNullTotal": 0,
                    "uniqueWorldCount": 0,
                    "uniqueGoalCount": 0,
                    "uniquePerturbationCount": 0,
                    "embeddingStatus": "unobserved_in_s04_primary_policy",
                }
            )

    profiles = pd.DataFrame(rows)
    text_columns = ("abstractPolicyId", "policyLabel", "sourceExperimentId", "policyFamily", "policyKind", "representationType", "stochasticity", "embeddingStatus")
    profiles = profiles.fillna({column: "unknown" for column in text_columns if column in profiles.columns})
    coverage = pd.DataFrame(coverage_rows)
    return profiles, coverage


def feature_columns(profile: pd.DataFrame) -> list[str]:
    return sorted(column for column in profile.columns if column.startswith("feat::"))


def matrix_from_profile(profile: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(profile), 0), dtype=float)
    aligned = profile.reindex(columns=columns, fill_value=0.0)
    return aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def fit_behavior_embedding(
    profile: pd.DataFrame,
    columns: Sequence[str],
    n_components: int = 12,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    x = matrix_from_profile(profile, columns)
    if x.shape[0] < 2 or x.shape[1] == 0:
        raise ValueError("At least two policies and one feature are required for S06 policy embedding")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    components = int(min(n_components, x_scaled.shape[0] - 1, x_scaled.shape[1]))
    pca = PCA(n_components=components, random_state=0)
    embedding = pca.fit_transform(x_scaled)
    rows = profile[list(PROFILE_METADATA_COLUMNS)].copy()
    for index in range(components):
        rows[f"embedding_{index:02d}"] = embedding[:, index]
    rows["embeddingX"] = embedding[:, 0]
    rows["embeddingY"] = embedding[:, 1] if components > 1 else 0.0
    rows["featureCount"] = int(len(columns))
    rows["embeddingModel"] = "weighted_behavior_pca"
    model = {
        "schemaVersion": POLICY_EMBEDDING_SCHEMA_VERSION,
        "modelVersion": POLICY_EMBEDDING_MODEL_VERSION,
        "featureColumns": list(columns),
        "scaler": scaler,
        "pca": pca,
        "explainedVarianceRatio": [float(value) for value in pca.explained_variance_ratio_],
        "policyIds": rows["abstractPolicyId"].astype(str).tolist(),
        "claimBoundary": POLICY_EMBEDDING_CLAIM_BOUNDARY,
    }
    return rows, model, embedding


def _metadata_records(policy_catalog: pd.DataFrame, policy_ids: Sequence[str]) -> list[dict[str, Any]]:
    indexed = policy_catalog.set_index("abstractPolicyId", drop=False)
    records: list[dict[str, Any]] = []
    for policy_id in policy_ids:
        row = indexed.loc[policy_id].to_dict() if policy_id in indexed.index else {"abstractPolicyId": policy_id}
        records.append(row)
    return records


def family_baseline_matrix(policy_catalog: pd.DataFrame, policy_ids: Sequence[str], n_components: int = 12) -> tuple[np.ndarray, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in _metadata_records(policy_catalog, policy_ids):
        record: dict[str, Any] = {}
        for column in ("sourceExperimentId", "policyFamily", "policyKind", "representationType", "stochasticity"):
            record[f"{column}={_safe_label(row.get(column))}"] = 1.0
        for column in ("memoryDepth", "signalingHorizon", "complexityScore", "sourceRowCount"):
            value = _float_or_nan(row.get(column))
            record[column] = 0.0 if math.isnan(value) else float(value)
        for column in ("dslAvailable", "automatonAvailable", "decisionGraphAvailable", "neuralMetadataAvailable"):
            record[column] = float(bool(row.get(column)))
        records.append(record)
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform(records)
    components = int(min(n_components, x.shape[0] - 1, max(1, x.shape[1] - 1)))
    svd = TruncatedSVD(n_components=components, random_state=0)
    matrix = svd.fit_transform(x)
    return matrix, {"vectorizer": vectorizer, "svd": svd, "featureNames": vectorizer.get_feature_names_out().tolist()}


def source_code_baseline_matrix(policy_catalog: pd.DataFrame, policy_ids: Sequence[str], n_components: int = 12) -> tuple[np.ndarray, dict[str, Any]]:
    texts: list[str] = []
    for row in _metadata_records(policy_catalog, policy_ids):
        text = " ".join(
            [
                _safe_text(row.get("sourcePolicyId")),
                _safe_text(row.get("policyLabel")),
                _safe_text(row.get("representationHash")),
                _json_text(row.get("representationJson")),
            ]
        )
        texts.append(text)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=512, min_df=1)
    x = vectorizer.fit_transform(texts)
    components = int(min(n_components, x.shape[0] - 1, max(1, x.shape[1] - 1)))
    svd = TruncatedSVD(n_components=components, random_state=0)
    matrix = svd.fit_transform(x)
    return matrix, {"vectorizer": vectorizer, "svd": svd, "featureCount": int(x.shape[1])}


def random_baseline_matrix(policy_ids: Sequence[str], n_components: int = 12) -> np.ndarray:
    rows: list[list[float]] = []
    for policy_id in policy_ids:
        digest = hashlib.sha256(f"e07_s06_random_baseline:{policy_id}".encode("utf-8")).digest()
        values = np.frombuffer(digest[: n_components * 2], dtype=np.uint16).astype(float)
        values = values[:n_components] if len(values) >= n_components else np.pad(values, (0, n_components - len(values)))
        rows.append((values / 65535.0) - 0.5)
    return np.asarray(rows, dtype=float)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def neighbor_indices(matrix: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    normalized = _normalize(np.asarray(matrix, dtype=float))
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    k = int(min(k, max(1, matrix.shape[0] - 1)))
    order = np.argsort(-similarity, axis=1)[:, :k]
    scores = np.take_along_axis(similarity, order, axis=1)
    return order, scores


def nearest_neighbor_table(policy_ids: Sequence[str], matrix: np.ndarray, model_name: str, k: int = 10) -> pd.DataFrame:
    order, scores = neighbor_indices(matrix, k=k)
    rows: list[dict[str, Any]] = []
    ids = list(policy_ids)
    for query_index, policy_id in enumerate(ids):
        for rank, (neighbor_index, score) in enumerate(zip(order[query_index], scores[query_index]), start=1):
            rows.append(
                {
                    "modelName": model_name,
                    "queryPolicyId": policy_id,
                    "neighborRank": rank,
                    "neighborPolicyId": ids[int(neighbor_index)],
                    "cosineSimilarity": float(score),
                }
            )
    return pd.DataFrame(rows)


def label_retrieval_metrics(
    policy_ids: Sequence[str],
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    model_name: str,
    labels: Sequence[str] = ("policyFamily", "policyKind", "sourceExperimentId"),
    ks: Sequence[int] = (1, 5, 10),
) -> pd.DataFrame:
    order, _ = neighbor_indices(matrix, k=max(ks))
    meta = metadata.set_index("abstractPolicyId")
    rows: list[dict[str, Any]] = []
    ids = list(policy_ids)
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
                    "validationTask": "label_neighbor_retrieval",
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


def heldout_behavior_retrieval_metrics(
    policy_ids: Sequence[str],
    model_matrices: Mapping[str, np.ndarray],
    truth_profile: pd.DataFrame,
    truth_columns: Sequence[str],
    ks: Sequence[int] = (1, 5, 10),
) -> pd.DataFrame:
    ids = list(policy_ids)
    truth = truth_profile.set_index("abstractPolicyId").reindex(ids)
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
                    "validationTask": "heldout_behavior_profile_retrieval",
                    "modelName": model_name,
                    "k": int(k),
                    "eligibleQueries": int(evaluated),
                    "meanCosineToHeldoutProfile": float(np.mean(scores)) if scores else math.nan,
                    "medianCosineToHeldoutProfile": float(np.median(scores)) if scores else math.nan,
                }
            )
    return pd.DataFrame(rows)


def split_behavior_frame(frame: pd.DataFrame, salt: str = "s06_policy_embedding") -> tuple[pd.DataFrame, pd.DataFrame]:
    buckets = frame["behaviorRecordId"].map(lambda value: stable_bucket(value, salt=salt, modulus=2))
    return frame[buckets.eq(0)].copy(), frame[buckets.eq(1)].copy()


def stability_metrics(
    policy_ids: Sequence[str],
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    k: int = 10,
    max_pairs: int = 20_000,
) -> pd.DataFrame:
    ids = list(policy_ids)
    order_a, _ = neighbor_indices(embedding_a, k=k)
    order_b, _ = neighbor_indices(embedding_b, k=k)
    jaccards: list[float] = []
    for index in range(len(ids)):
        set_a = set(int(value) for value in order_a[index])
        set_b = set(int(value) for value in order_b[index])
        union = set_a | set_b
        jaccards.append(len(set_a & set_b) / len(union) if union else math.nan)

    dist_a = 1.0 - (_normalize(embedding_a) @ _normalize(embedding_a).T)
    dist_b = 1.0 - (_normalize(embedding_b) @ _normalize(embedding_b).T)
    pairs = np.transpose(np.triu_indices(len(ids), k=1))
    if len(pairs) > max_pairs:
        chosen = []
        for pair in pairs:
            if stable_bucket(f"{ids[pair[0]]}:{ids[pair[1]]}", salt="s06_pair_sample", modulus=10_000) < int(max_pairs / len(pairs) * 10_000):
                chosen.append(pair)
            if len(chosen) >= max_pairs:
                break
        pairs = np.asarray(chosen, dtype=int)
    a_values = np.asarray([dist_a[i, j] for i, j in pairs], dtype=float)
    b_values = np.asarray([dist_b[i, j] for i, j in pairs], dtype=float)
    spearman = pd.Series(a_values).corr(pd.Series(b_values), method="spearman") if len(a_values) > 1 else math.nan
    return pd.DataFrame(
        [
            {
                "validationTask": "split_half_embedding_stability",
                "modelName": "weighted_behavior_pca",
                "policyCount": int(len(ids)),
                "neighborK": int(k),
                "meanNeighborJaccard": float(np.nanmean(jaccards)) if jaccards else math.nan,
                "medianNeighborJaccard": float(np.nanmedian(jaccards)) if jaccards else math.nan,
                "pairCount": int(len(a_values)),
                "pairwiseDistanceSpearman": float(spearman) if spearman == spearman else math.nan,
            }
        ]
    )


def baseline_comparison(metrics: pd.DataFrame, primary_model: str = "weighted_behavior_pca") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    heldout = metrics[metrics["validationTask"].eq("heldout_behavior_profile_retrieval")]
    primary = heldout[heldout["modelName"].eq(primary_model)]
    for _, primary_row in primary.iterrows():
        subset = heldout[
            heldout["k"].eq(primary_row["k"])
            & ~heldout["modelName"].eq(primary_model)
        ]
        for _, baseline in subset.iterrows():
            rows.append(
                {
                    "validationTask": primary_row["validationTask"],
                    "k": int(primary_row["k"]),
                    "baselineName": baseline["modelName"],
                    "behaviorMeanCosine": primary_row["meanCosineToHeldoutProfile"],
                    "baselineMeanCosine": baseline["meanCosineToHeldoutProfile"],
                    "cosineImprovement": primary_row["meanCosineToHeldoutProfile"] - baseline["meanCosineToHeldoutProfile"],
                    "behaviorBeatsBaseline": bool(primary_row["meanCosineToHeldoutProfile"] > baseline["meanCosineToHeldoutProfile"]),
                }
            )
    return pd.DataFrame(rows)


def validate_policy_embeddings(
    embeddings: pd.DataFrame,
    coverage: pd.DataFrame,
    retrieval_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    model_path: str | Path,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(scope: str, check: str, success: bool, severity: str = "error", detail: str = "") -> None:
        checks.append({"scope": scope, "check": check, "success": bool(success), "severity": severity, "detail": detail})

    add("embeddings", "policy_embedding_rows_present", len(embeddings) >= 100, detail=str(len(embeddings)))
    add("embeddings", "embedding_coordinates_finite", np.isfinite(embeddings.filter(regex=r"^embedding_").to_numpy(dtype=float)).all(), detail=str(embeddings.filter(regex=r"^embedding_").shape))
    observed_count = int((coverage["rowCount"] > 0).sum()) if not coverage.empty else 0
    embedded_count = int(coverage["embeddingStatus"].eq("embedded_behavior").sum()) if not coverage.empty else 0
    add("coverage", "observed_policies_documented", observed_count >= len(embeddings), detail=str(observed_count))
    add("coverage", "embedded_policy_count_matches", embedded_count == len(embeddings), detail=f"{embedded_count} embedded, {len(embeddings)} rows")
    models = set(retrieval_metrics["modelName"].astype(str)) if not retrieval_metrics.empty else set()
    add("validation", "retrieval_metrics_include_behavior_and_baselines", {"weighted_behavior_pca", "family_metadata_baseline", "source_code_baseline"}.issubset(models), detail=compact_json(sorted(models)))
    heldout = retrieval_metrics[retrieval_metrics["validationTask"].eq("heldout_behavior_profile_retrieval")] if not retrieval_metrics.empty else pd.DataFrame()
    add("validation", "heldout_behavior_retrieval_present", not heldout.empty, detail=str(len(heldout)))
    add("validation", "stability_metrics_present", not stability.empty, detail=str(len(stability)))
    if not stability.empty:
        value = float(stability["pairwiseDistanceSpearman"].iloc[0])
        add("validation", "split_half_distance_correlation_positive", value > 0, severity="warning", detail=str(value))
    add("model", "embedding_model_written", Path(model_path).exists(), detail=str(model_path))
    return pd.DataFrame(checks)


def performance_summary(retrieval_metrics: pd.DataFrame, comparisons: pd.DataFrame, stability: pd.DataFrame) -> dict[str, Any]:
    heldout = retrieval_metrics[retrieval_metrics["validationTask"].eq("heldout_behavior_profile_retrieval")] if not retrieval_metrics.empty else pd.DataFrame()
    k5 = heldout[heldout["k"].eq(5)] if not heldout.empty else pd.DataFrame()
    behavior_k5 = k5[k5["modelName"].eq("weighted_behavior_pca")]
    comparisons_k5 = comparisons[comparisons["k"].eq(5)] if not comparisons.empty else pd.DataFrame()
    return {
        "retrievalMetricRows": int(len(retrieval_metrics)),
        "heldoutBehaviorMetricRows": int(len(heldout)),
        "behaviorK5MeanCosine": float(behavior_k5["meanCosineToHeldoutProfile"].iloc[0]) if not behavior_k5.empty else math.nan,
        "k5BaselineComparisonsWon": int(comparisons_k5["behaviorBeatsBaseline"].sum()) if not comparisons_k5.empty else 0,
        "k5BaselineComparisonCount": int(len(comparisons_k5)),
        "splitHalfDistanceSpearman": float(stability["pairwiseDistanceSpearman"].iloc[0]) if not stability.empty else math.nan,
        "splitHalfMeanNeighborJaccard": float(stability["meanNeighborJaccard"].iloc[0]) if not stability.empty else math.nan,
    }


def artifact_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)}
