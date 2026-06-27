"""Behavior-embedding helpers for E03 S10.

S10 maps policies by measured behavior rather than by source-code similarity.
The feature table combines S07 competence vectors, S07/S09 trajectory summaries,
S08 elite metadata, and S09 phase-boundary involvement.  The embedding routines
use only preinstalled scientific Python dependencies and keep normalization
explicit for auditability.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, SpectralEmbedding

from .competence import canonical_json


BEHAVIOR_EMBEDDING_VERSION = "e03_s10_behavior_embeddings.v1"

COMPETENCE_METRIC_COLUMNS = (
    "completionSuccess",
    "finalSortednessScore",
    "finalMonotonicityScore",
    "finalKendallTauScore",
    "finalEarthMoverPositionScore",
    "swapEfficiencyScore",
    "comparisonEfficiencyScore",
    "activationEfficiencyScore",
    "energyScore",
    "blockedMoveRate",
    "frozenSwapAttemptRate",
    "robustnessScore",
    "delayedGratification",
    "delayedGratificationScore",
    "dgEventCount",
    "aggregationFinal",
    "aggregationPeak",
    "aggregationAuc",
    "aggregationPeakScore",
    "aggregationAucScore",
    "oscillationProxy",
    "oscillationScore",
    "failureFlag",
    "failureOscillationScore",
    "trajectoryExcessPathRatio",
    "pathDirectnessScore",
    "eventCount",
    "swapCount",
    "comparisonCount",
    "activationCount",
)

RUN_METRIC_COLUMNS = (
    "completedNumeric",
    "finalSortednessScore",
    "finalSortednessPercent",
    "finalMonotonicityError",
    "delayedGratification",
    "dgEventCount",
    "oscillationProxy",
    "initialAggregation",
    "finalAggregation",
    "robustnessScore",
    "swapCount",
    "comparisonCount",
    "activationCount",
    "eventCount",
    "blockedMoveAttempts",
    "frozenSwapAttempts",
    "traceStateCount",
    "runtimeSeconds",
)

STRUCTURAL_NUMERIC_COLUMNS = (
    "lineageDepth",
    "generationIndex",
    "ruleCount",
    "predicateCount",
    "updateCount",
    "stateKeyCount",
    "stochasticActionCount",
    "targetUpdateCount",
    "signalCount",
    "complexityScore",
)

EMBEDDING_METHODS = ("pca", "spectral", "mds")
EMBEDDING_SEEDS = (0, 1, 2)


def _safe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return default


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _prefix_columns(df: pd.DataFrame, prefix: str, skip: Sequence[str] = ("policyId",)) -> pd.DataFrame:
    renamed = {}
    skip_set = set(skip)
    for column in df.columns:
        if column not in skip_set:
            renamed[column] = f"{prefix}{column}"
    return df.rename(columns=renamed)


def _sd(values: pd.Series) -> float:
    numeric = _as_numeric(values).dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1))


def _series_json_unique(values: pd.Series) -> str:
    return canonical_json(sorted(set(map(str, values.dropna()))))


def _parse_action_counts(value: Any) -> dict[str, int]:
    raw = _safe_json(value, {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): int(val) for key, val in raw.items()}


def _trace_hash_count(value: Any) -> tuple[int, int]:
    raw = _safe_json(value, [])
    if not isinstance(raw, list):
        return (0, 0)
    hashes = [str(item) for item in raw]
    return (len(hashes), len(set(hashes)))


def corpus_policy_metadata(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per corpus policy with structural and role hints."""

    corpus = corpus_df.copy()
    corpus["policyId"] = corpus["policyId"].astype(str)
    rows: list[dict[str, Any]] = []
    for _, row in corpus.iterrows():
        action_counts = _parse_action_counts(row.get("actionCountsJson"))
        lineage = str(row.get("lineageId", ""))
        generation_method = str(row.get("generationMethod", ""))
        action_total = int(sum(action_counts.values()))
        swap_action_count = int(sum(action_counts.get(action, 0) for action in ["swap_left", "swap_right", "swap_target"]))
        is_null = "null" in lineage or (action_total > 0 and swap_action_count == 0 and action_counts.get("wait", 0) == action_total)
        is_classic = (
            generation_method == "classic_parameter_sweep"
            or lineage.startswith(("bubble_", "insertion_", "selection_"))
            or lineage in {"seed_bubble_template", "seed_insertion_template", "seed_selection_template"}
        )
        classic_family = "none"
        for key in ["bubble", "insertion", "selection"]:
            if key in lineage:
                classic_family = key
                break
        payload = {
            "policyId": str(row["policyId"]),
            "family": str(row.get("family", "unknown")),
            "generationMethod": generation_method,
            "lineageId": lineage,
            "description": str(row.get("description", "")),
            "isNullPolicy": bool(is_null),
            "isClassicPolicy": bool(is_classic),
            "classicFamily": classic_family,
            "actionTotalCount": action_total,
            "swapActionCount": swap_action_count,
            "waitActionCount": int(action_counts.get("wait", 0)),
            "rememberActionCount": int(action_counts.get("remember", 0)),
            "signalActionCountFromActions": int(action_counts.get("signal", 0)),
            "swapLeftActionCount": int(action_counts.get("swap_left", 0)),
            "swapRightActionCount": int(action_counts.get("swap_right", 0)),
            "swapTargetActionCount": int(action_counts.get("swap_target", 0)),
        }
        for column in STRUCTURAL_NUMERIC_COLUMNS:
            payload[column] = row.get(column, np.nan)
        rows.append(payload)
    return pd.DataFrame(rows)


def aggregate_competence_features(comp_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate competence-vector rows to policy-level behavior features."""

    if comp_df.empty or "policyId" not in comp_df.columns:
        return pd.DataFrame(columns=["policyId"])
    df = comp_df.copy()
    df["policyId"] = df["policyId"].astype(str)
    metrics = [column for column in COMPETENCE_METRIC_COLUMNS if column in df.columns]
    for column in metrics:
        df[column] = _as_numeric(df[column])
    grouped = df.groupby("policyId", dropna=False)
    frames: list[pd.DataFrame] = [
        grouped.agg(
            vectorRowCount=("policyId", "size"),
            taskCount=("taskId", pd.Series.nunique),
            taskPanelCount=("taskPanel", pd.Series.nunique),
            taskFamilyCount=("taskFamily", pd.Series.nunique),
            completedRunCount=("completed", lambda values: int(pd.Series(values).astype(bool).sum())),
            stopReasonsJson=("stopReason", _series_json_unique),
        ).reset_index()
    ]
    if metrics:
        summary = grouped[metrics].agg(["mean", _sd, "min", "max"]).reset_index()
        summary.columns = [
            "policyId"
            if column[0] == "policyId"
            else f"{column[0]}{''.join(part.capitalize() for part in column[1:])}"
            for column in summary.columns.to_flat_index()
        ]
        frames.append(summary)
    if "taskFamily" in df.columns and "completionSuccess" in df.columns:
        pivot = (
            df.pivot_table(
                index="policyId",
                columns="taskFamily",
                values="completionSuccess",
                aggfunc="mean",
            )
            .add_prefix("completionRateTaskFamily_")
            .reset_index()
        )
        frames.append(pivot)
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="policyId", how="left")
    return _prefix_columns(result, prefix)


def aggregate_run_features(run_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate run/trajectory summary rows to policy-level features."""

    if run_df.empty or "policyId" not in run_df.columns:
        return pd.DataFrame(columns=["policyId"])
    df = run_df.copy()
    df["policyId"] = df["policyId"].astype(str)
    if "completedNumeric" not in df.columns and "completed" in df.columns:
        df["completedNumeric"] = df["completed"].astype(bool).astype(float)
    if "traceStateCount" not in df.columns and "traceHashSequenceJson" in df.columns:
        counts = df["traceHashSequenceJson"].map(_trace_hash_count)
        df["traceStateCount"] = counts.map(lambda item: item[0])
        df["traceUniqueStateCount"] = counts.map(lambda item: item[1])
    elif "traceHashSequenceJson" in df.columns:
        counts = df["traceHashSequenceJson"].map(_trace_hash_count)
        df["traceUniqueStateCount"] = counts.map(lambda item: item[1])
    metrics = [column for column in RUN_METRIC_COLUMNS if column in df.columns]
    if "traceUniqueStateCount" in df.columns:
        metrics.append("traceUniqueStateCount")
    for column in metrics:
        df[column] = _as_numeric(df[column])
    grouped = df.groupby("policyId", dropna=False)
    task_column = "taskId" if "taskId" in df.columns else "validationTaskId" if "validationTaskId" in df.columns else "policyId"
    task_family_column = "taskFamily" if "taskFamily" in df.columns else "policyId"
    frames: list[pd.DataFrame] = [
        grouped.agg(
            runRowCount=("policyId", "size"),
            runTaskCount=(task_column, pd.Series.nunique),
            runTaskFamilyCount=(task_family_column, pd.Series.nunique),
            runStopReasonsJson=("stopReason", _series_json_unique),
        ).reset_index()
    ]
    if metrics:
        summary = grouped[metrics].agg(["mean", _sd, "min", "max"]).reset_index()
        summary.columns = [
            "policyId"
            if column[0] == "policyId"
            else f"{column[0]}{''.join(part.capitalize() for part in column[1:])}"
            for column in summary.columns.to_flat_index()
        ]
        frames.append(summary)
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="policyId", how="left")
    return _prefix_columns(result, prefix)


def aggregate_elite_features(elites_df: pd.DataFrame) -> pd.DataFrame:
    if elites_df.empty or "policyId" not in elites_df.columns:
        return pd.DataFrame(columns=["policyId"])
    columns = [
        "policyId",
        "eliteRank",
        "archiveRank",
        "qualityScore",
        "noveltyScore",
        "completionRate",
        "meanSortednessScore",
        "meanEnergyScore",
        "duplicateCompletionRate",
        "robustnessMean",
        "aggregationMean",
        "coverageScore",
        "missingBurdenNormalized",
        "cellOccupancy",
    ]
    available = [column for column in columns if column in elites_df.columns]
    df = elites_df[available].copy()
    df["policyId"] = df["policyId"].astype(str)
    df["isS08Elite"] = True
    return _prefix_columns(df, "s08Elite_")


def aggregate_phase_boundary_features(boundaries_df: pd.DataFrame) -> pd.DataFrame:
    if boundaries_df.empty:
        return pd.DataFrame(columns=["policyId"])
    transition_columns = [
        "sortingSuccessTransition",
        "sortednessTransition",
        "delayedGratificationTransition",
        "aggregationTransition",
        "oscillationTransition",
        "robustnessTransition",
    ]
    rows: list[dict[str, Any]] = []
    for _, row in boundaries_df.iterrows():
        for side, policy_col in [("left", "leftPolicyId"), ("right", "rightPolicyId")]:
            if policy_col not in row or pd.isna(row[policy_col]):
                continue
            payload = {
                "policyId": str(row[policy_col]),
                "boundarySide": side,
                "boundaryScore": row.get("boundaryScore", np.nan),
                "transitionKindCount": row.get("transitionKindCount", np.nan),
            }
            for column in transition_columns:
                payload[column] = bool(row.get(column, False))
            rows.append(payload)
    if not rows:
        return pd.DataFrame(columns=["policyId"])
    events = pd.DataFrame(rows)
    events["isLeftBoundarySide"] = events["boundarySide"].eq("left").astype(int)
    events["isRightBoundarySide"] = events["boundarySide"].eq("right").astype(int)
    grouped = events.groupby("policyId", dropna=False)
    agg = grouped.agg(
        phaseBoundaryInvolvementCount=("policyId", "size"),
        phaseBoundaryLeftCount=("isLeftBoundarySide", "sum"),
        phaseBoundaryRightCount=("isRightBoundarySide", "sum"),
        phaseBoundaryScoreMean=("boundaryScore", "mean"),
        phaseBoundaryScoreMax=("boundaryScore", "max"),
        phaseBoundaryTransitionKindMean=("transitionKindCount", "mean"),
        phaseBoundaryTransitionKindMax=("transitionKindCount", "max"),
    ).reset_index()
    for column in transition_columns:
        counts = grouped[column].sum().rename(f"{column}Count").reset_index()
        agg = agg.merge(counts, on="policyId", how="left")
    return agg


def aggregate_variance_features(variance_df: pd.DataFrame) -> pd.DataFrame:
    if variance_df.empty or "policyId" not in variance_df.columns:
        return pd.DataFrame(columns=["policyId"])
    df = variance_df.copy()
    df["policyId"] = df["policyId"].astype(str)
    metrics = [
        "completionRateSd",
        "finalSortednessScoreSd",
        "delayedGratificationSd",
        "aggregationFinalSd",
        "oscillationProxySd",
        "robustnessScoreSd",
    ]
    metrics = [column for column in metrics if column in df.columns]
    for column in metrics:
        df[column] = _as_numeric(df[column])
    grouped = df.groupby("policyId", dropna=False)
    result = grouped.agg(nearBoundaryVarianceRecordCount=("policyId", "size")).reset_index()
    if metrics:
        summary = grouped[metrics].agg(["mean", "max"]).reset_index()
        summary.columns = [
            "policyId"
            if column[0] == "policyId"
            else f"nearBoundary{column[0][0].upper()}{column[0][1:]}{''.join(part.capitalize() for part in column[1:])}"
            for column in summary.columns.to_flat_index()
        ]
        result = result.merge(summary, on="policyId", how="left")
    return result


def build_policy_feature_table(
    *,
    corpus_df: pd.DataFrame,
    s07_competence_df: pd.DataFrame,
    s07_run_df: pd.DataFrame,
    s08_elites_df: pd.DataFrame,
    s08_validation_df: pd.DataFrame,
    s09_competence_df: pd.DataFrame,
    s09_run_df: pd.DataFrame,
    s09_boundaries_df: pd.DataFrame,
    s09_variance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine S07-S09 evidence into one policy-level feature table."""

    ids: set[str] = set()
    for df in [
        corpus_df,
        s07_competence_df,
        s07_run_df,
        s08_elites_df,
        s08_validation_df,
        s09_competence_df,
        s09_run_df,
        s09_variance_df,
    ]:
        if "policyId" in df.columns:
            ids.update(map(str, df["policyId"].dropna()))
    for column in ["leftPolicyId", "rightPolicyId"]:
        if column in s09_boundaries_df.columns:
            ids.update(map(str, s09_boundaries_df[column].dropna()))
    base = pd.DataFrame({"policyId": sorted(ids)})
    metadata = corpus_policy_metadata(corpus_df)
    result = base.merge(metadata, on="policyId", how="left")
    aggregates = [
        aggregate_competence_features(s07_competence_df, "s07_"),
        aggregate_run_features(s07_run_df, "s07Trace_"),
        aggregate_elite_features(s08_elites_df),
        aggregate_run_features(s08_validation_df, "s08Validation_"),
        aggregate_competence_features(s09_competence_df, "s09_"),
        aggregate_run_features(s09_run_df, "s09Trace_"),
        aggregate_phase_boundary_features(s09_boundaries_df),
        aggregate_variance_features(s09_variance_df),
    ]
    for frame in aggregates:
        if not frame.empty:
            result = result.merge(frame, on="policyId", how="left")

    if "s08Elite_isS08Elite" in result.columns:
        result["isS08Elite"] = result["s08Elite_isS08Elite"].map(lambda value: bool(value) if pd.notna(value) else False)
    else:
        result["isS08Elite"] = False
    result["isPhaseBoundaryPolicy"] = result.get("phaseBoundaryInvolvementCount", 0).fillna(0).astype(float) > 0
    result["isNullPolicy"] = result["isNullPolicy"].map(lambda value: bool(value) if pd.notna(value) else False)
    result["isClassicPolicy"] = result["isClassicPolicy"].map(lambda value: bool(value) if pd.notna(value) else False)
    result["family"] = result["family"].fillna("not_in_s05_corpus")
    result["generationMethod"] = result["generationMethod"].fillna("not_in_s05_corpus")
    result["lineageId"] = result["lineageId"].fillna("not_in_s05_corpus")
    result["classicFamily"] = result["classicFamily"].fillna("none")

    s09_completion = pd.to_numeric(result.get("s09Trace_completedNumericMean", np.nan), errors="coerce")
    s09_oscillation = pd.to_numeric(result.get("s09Trace_oscillationProxyMean", np.nan), errors="coerce")
    s07_completion = pd.to_numeric(result.get("s07_completionSuccessMean", np.nan), errors="coerce")
    result["isPathologicalPolicy"] = (
        ((s09_completion.fillna(s07_completion).fillna(1.0) <= 0.10) & (s09_oscillation.fillna(0.0) >= 0.25))
        | ((s07_completion.fillna(1.0) <= 0.10) & result["isPhaseBoundaryPolicy"])
    )

    def primary_role(row: pd.Series) -> str:
        if bool(row["isNullPolicy"]):
            return "null"
        if bool(row["isClassicPolicy"]):
            return "classic"
        if bool(row["isS08Elite"]):
            return "elite"
        if bool(row["isPathologicalPolicy"]):
            return "pathological"
        if bool(row["isPhaseBoundaryPolicy"]):
            return "phase_boundary"
        return "generated"

    result["primaryRole"] = result.apply(primary_role, axis=1)
    evidence_counts = []
    for _, row in result.iterrows():
        count = 0
        for column in [
            "s07_vectorRowCount",
            "s07Trace_runRowCount",
            "s08Validation_runRowCount",
            "s09_vectorRowCount",
            "s09Trace_runRowCount",
            "phaseBoundaryInvolvementCount",
        ]:
            if float(row.get(column, 0) or 0) > 0:
                count += 1
        evidence_counts.append(count)
    result["evidenceLayerCount"] = evidence_counts
    result["behaviorEmbeddingVersion"] = BEHAVIOR_EMBEDDING_VERSION
    return result


def normalize_feature_table(
    feature_df: pd.DataFrame,
    *,
    clip_value: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median/IQR-normalize numeric features with explicit imputation records."""

    metadata_like = {
        "policyId",
        "family",
        "generationMethod",
        "lineageId",
        "description",
        "classicFamily",
        "primaryRole",
        "behaviorEmbeddingVersion",
    }
    numeric_candidates: list[str] = []
    prepared = feature_df.copy()
    for column in prepared.columns:
        if column in metadata_like or column.endswith("Json"):
            continue
        if prepared[column].dtype == bool:
            prepared[column] = prepared[column].astype(float)
            numeric_candidates.append(column)
            continue
        numeric = pd.to_numeric(prepared[column], errors="coerce")
        if numeric.notna().any():
            prepared[column] = numeric
            numeric_candidates.append(column)
    matrix_payload: dict[str, Any] = {"policyId": prepared["policyId"].astype(str)}
    spec_rows: list[dict[str, Any]] = []
    for column in sorted(set(numeric_candidates)):
        values = pd.to_numeric(prepared[column], errors="coerce")
        finite = values[np.isfinite(values)]
        missing_count = int(values.isna().sum() + (~np.isfinite(values.fillna(0))).sum())
        if finite.empty:
            impute = 0.0
        else:
            impute = float(finite.median())
        filled = values.replace([np.inf, -np.inf], np.nan).fillna(impute).astype(float)
        center = float(filled.median())
        q1 = float(filled.quantile(0.25))
        q3 = float(filled.quantile(0.75))
        iqr = q3 - q1
        std = float(filled.std(ddof=0))
        if math.isfinite(iqr) and iqr > 1e-12:
            scale = iqr
            scale_method = "median_iqr"
        elif math.isfinite(std) and std > 1e-12:
            scale = std
            scale_method = "median_std_fallback"
        else:
            scale = 1.0
            scale_method = "constant_feature_unit_scale"
        normalized = ((filled - center) / scale).clip(-clip_value, clip_value)
        matrix_payload[column] = normalized.astype(float)
        spec_rows.append(
            {
                "behaviorEmbeddingVersion": BEHAVIOR_EMBEDDING_VERSION,
                "featureName": column,
                "sourceFeature": column,
                "isMissingIndicator": False,
                "rawMissingCount": missing_count,
                "rawMissingFraction": float(missing_count / max(1, len(prepared))),
                "imputationValue": impute,
                "center": center,
                "scale": float(scale),
                "scaleMethod": scale_method,
                "clipValue": float(clip_value),
                "normalization": "fill_missing_with_median_then_center_by_median_scale_by_iqr_and_clip",
            }
        )
        if missing_count > 0:
            indicator_name = f"{column}__missing"
            indicator = values.isna().astype(float)
            matrix_payload[indicator_name] = indicator
            spec_rows.append(
                {
                    "behaviorEmbeddingVersion": BEHAVIOR_EMBEDDING_VERSION,
                    "featureName": indicator_name,
                    "sourceFeature": column,
                    "isMissingIndicator": True,
                    "rawMissingCount": 0,
                    "rawMissingFraction": 0.0,
                    "imputationValue": 0.0,
                    "center": 0.0,
                    "scale": 1.0,
                    "scaleMethod": "binary_missing_indicator",
                    "clipValue": 1.0,
                    "normalization": "1_if_source_feature_missing_else_0",
                }
            )
    matrix_df = pd.DataFrame(matrix_payload)
    spec_df = pd.DataFrame(spec_rows)
    return matrix_df, spec_df


def compute_embeddings(
    normalized_matrix_df: pd.DataFrame,
    *,
    methods: Sequence[str] = EMBEDDING_METHODS,
    seeds: Sequence[int] = EMBEDDING_SEEDS,
) -> pd.DataFrame:
    """Compute 2D/5D policy embeddings for configured methods and seeds."""

    policy_ids = normalized_matrix_df["policyId"].astype(str).tolist()
    feature_columns = [column for column in normalized_matrix_df.columns if column != "policyId"]
    x = normalized_matrix_df[feature_columns].to_numpy(dtype=float)
    if len(policy_ids) < 3:
        raise ValueError("at least three policies are required for S10 embeddings")
    rows: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            if method == "pca":
                n_components = max(2, min(5, x.shape[0], x.shape[1]))
                model = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
                coords = model.fit_transform(x)
                variance = list(map(float, model.explained_variance_ratio_))
                extra = {"explainedVarianceRatioJson": canonical_json(variance)}
            elif method == "spectral":
                n_neighbors = max(2, min(15, x.shape[0] - 1))
                model = SpectralEmbedding(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    affinity="nearest_neighbors",
                    random_state=int(seed),
                )
                coords = model.fit_transform(x)
                extra = {"nNeighbors": n_neighbors}
            elif method == "mds":
                model = MDS(
                    n_components=2,
                    metric_mds=True,
                    n_init=1,
                    init="random",
                    max_iter=160,
                    eps=1e-3,
                    random_state=int(seed),
                    metric="euclidean",
                    normalized_stress="auto",
                )
                coords = model.fit_transform(x)
                extra = {"stress": float(getattr(model, "stress_", np.nan))}
            else:
                raise ValueError(f"unknown embedding method: {method}")
            if coords.shape[1] < 5:
                coords = np.pad(coords, ((0, 0), (0, 5 - coords.shape[1])), constant_values=np.nan)
            for index, policy_id in enumerate(policy_ids):
                row = {
                    "behaviorEmbeddingVersion": BEHAVIOR_EMBEDDING_VERSION,
                    "policyId": policy_id,
                    "embeddingMethod": method,
                    "embeddingSeed": int(seed),
                    "embeddingDim1": float(coords[index, 0]),
                    "embeddingDim2": float(coords[index, 1]),
                    "embeddingDim3": float(coords[index, 2]) if math.isfinite(coords[index, 2]) else np.nan,
                    "embeddingDim4": float(coords[index, 3]) if math.isfinite(coords[index, 3]) else np.nan,
                    "embeddingDim5": float(coords[index, 4]) if math.isfinite(coords[index, 4]) else np.nan,
                }
                row.update(extra)
                rows.append(row)
    return pd.DataFrame(rows)


def _coords_for(embeddings_df: pd.DataFrame, method: str, seed: int) -> tuple[list[str], np.ndarray]:
    subset = embeddings_df[
        embeddings_df["embeddingMethod"].eq(method) & embeddings_df["embeddingSeed"].eq(int(seed))
    ].sort_values("policyId", kind="mergesort")
    coords = subset[["embeddingDim1", "embeddingDim2"]].to_numpy(dtype=float)
    return subset["policyId"].astype(str).tolist(), coords


def _neighbor_sets(coords: np.ndarray, k: int) -> list[set[int]]:
    distances = squareform(pdist(coords))
    result = []
    for index in range(len(coords)):
        order = np.argsort(distances[index])
        neighbors = [int(item) for item in order if int(item) != index][:k]
        result.append(set(neighbors))
    return result


def _mean_neighbor_jaccard(left: np.ndarray, right: np.ndarray, k: int) -> float:
    left_sets = _neighbor_sets(left, k)
    right_sets = _neighbor_sets(right, k)
    scores = []
    for a, b in zip(left_sets, right_sets):
        union = a | b
        scores.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(scores)) if scores else 0.0


def embedding_stability_table(
    embeddings_df: pd.DataFrame,
    *,
    neighbor_k: int = 10,
) -> pd.DataFrame:
    """Compare embedding stability across seeds and across methods."""

    methods = sorted(embeddings_df["embeddingMethod"].unique())
    seeds = sorted(map(int, embeddings_df["embeddingSeed"].unique()))
    rows: list[dict[str, Any]] = []
    for method in methods:
        for left_seed_index, left_seed in enumerate(seeds):
            for right_seed in seeds[left_seed_index + 1 :]:
                policy_ids_left, left = _coords_for(embeddings_df, method, left_seed)
                policy_ids_right, right = _coords_for(embeddings_df, method, right_seed)
                if policy_ids_left != policy_ids_right:
                    continue
                rows.append(_stability_row("within_method_seed", method, method, left_seed, right_seed, left, right, neighbor_k))
    if seeds:
        seed = seeds[0]
        for left_index, left_method in enumerate(methods):
            for right_method in methods[left_index + 1 :]:
                policy_ids_left, left = _coords_for(embeddings_df, left_method, seed)
                policy_ids_right, right = _coords_for(embeddings_df, right_method, seed)
                if policy_ids_left != policy_ids_right:
                    continue
                rows.append(_stability_row("cross_method_seed0", left_method, right_method, seed, seed, left, right, neighbor_k))
    return pd.DataFrame(rows)


def _stability_row(
    comparison_type: str,
    left_method: str,
    right_method: str,
    left_seed: int,
    right_seed: int,
    left: np.ndarray,
    right: np.ndarray,
    neighbor_k: int,
) -> dict[str, Any]:
    _, _, disparity = procrustes(left, right)
    left_dist = pdist(left)
    right_dist = pdist(right)
    if np.allclose(left_dist, left_dist[0]) or np.allclose(right_dist, right_dist[0]):
        distance_spearman = 0.0
    else:
        corr = spearmanr(left_dist, right_dist).statistic
        distance_spearman = float(corr) if math.isfinite(float(corr)) else 0.0
    return {
        "behaviorEmbeddingVersion": BEHAVIOR_EMBEDDING_VERSION,
        "comparisonType": comparison_type,
        "leftMethod": left_method,
        "rightMethod": right_method,
        "leftSeed": int(left_seed),
        "rightSeed": int(right_seed),
        "policyCount": int(left.shape[0]),
        "neighborK": int(neighbor_k),
        "procrustesDisparity": float(disparity),
        "pairwiseDistanceSpearman": distance_spearman,
        "meanNeighborJaccard": _mean_neighbor_jaccard(left, right, neighbor_k),
    }


def nearest_neighbors_table(
    normalized_matrix_df: pd.DataFrame,
    *,
    neighbor_k: int = 8,
) -> pd.DataFrame:
    """Return nearest behavioral neighbors in normalized feature space."""

    policy_ids = normalized_matrix_df["policyId"].astype(str).tolist()
    feature_columns = [column for column in normalized_matrix_df.columns if column != "policyId"]
    x = normalized_matrix_df[feature_columns].to_numpy(dtype=float)
    distances = squareform(pdist(x))
    rows: list[dict[str, Any]] = []
    for index, policy_id in enumerate(policy_ids):
        order = np.argsort(distances[index])
        rank = 1
        for neighbor_index in order:
            if int(neighbor_index) == index:
                continue
            rows.append(
                {
                    "behaviorEmbeddingVersion": BEHAVIOR_EMBEDDING_VERSION,
                    "policyId": policy_id,
                    "neighborPolicyId": policy_ids[int(neighbor_index)],
                    "neighborRank": rank,
                    "featureDistance": float(distances[index, int(neighbor_index)]),
                }
            )
            rank += 1
            if rank > int(neighbor_k):
                break
    return pd.DataFrame(rows)


def embedding_config_dict(
    *,
    methods: Sequence[str],
    seeds: Sequence[int],
    neighbor_k: int,
) -> dict[str, Any]:
    return {
        "behaviorEmbeddingVersion": BEHAVIOR_EMBEDDING_VERSION,
        "methods": list(methods),
        "seeds": [int(seed) for seed in seeds],
        "featureSources": [
            "S07 competence vectors",
            "S07 coarse trajectory summaries",
            "S08 QD elite metadata and CPU validation",
            "S09 competence vectors",
            "S09 repeated-seed trajectory summaries",
            "S09 phase-boundary and near-boundary variance records",
            "S05 corpus structural metadata for null/classic/generated policy role labels",
        ],
        "normalization": {
            "numericFeatures": "median imputation, median centering, IQR scaling with std/unit fallback, clipped to [-5,5]",
            "missingValues": "missing numeric values are imputed and accompanied by binary missing-indicator features",
            "categoricalUse": "categorical labels are retained as metadata; structural booleans are included as numeric features",
        },
        "stability": {
            "neighborK": int(neighbor_k),
            "withinMethod": "compare seed pairs for each method using Procrustes disparity, pairwise-distance Spearman, and neighbor Jaccard",
            "crossMethod": "compare seed-0 method pairs using the same criteria",
        },
    }


def validate_embedding_outputs(
    *,
    feature_df: pd.DataFrame,
    normalized_matrix_df: pd.DataFrame,
    normalization_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    upstream_statuses: Mapping[str, Mapping[str, Any]],
    repo_test_payload: Mapping[str, Any],
    figure_paths: Sequence[str],
    s11_dir_exists: bool,
) -> pd.DataFrame:
    policy_count = int(feature_df["policyId"].nunique()) if "policyId" in feature_df.columns else 0
    methods = set(map(str, embeddings_df.get("embeddingMethod", pd.Series(dtype=str))))
    seeds = set(map(int, embeddings_df.get("embeddingSeed", pd.Series(dtype=int)))) if len(embeddings_df) else set()
    expected_embedding_rows = policy_count * max(1, len(methods)) * max(1, len(seeds))
    checks = [
        {
            "checkId": "upstream_s07_s09_success",
            "success": all(bool(status.get("success")) for status in upstream_statuses.values()),
            "detail": canonical_json({step: status.get("success") for step, status in upstream_statuses.items()}),
        },
        {
            "checkId": "feature_table_policy_coverage",
            "success": policy_count >= 400 and feature_df["policyId"].is_unique,
            "detail": f"{policy_count} unique policies in S10 feature table",
        },
        {
            "checkId": "role_coverage",
            "success": all(
                bool(feature_df[column].fillna(False).astype(bool).any())
                for column in ["isNullPolicy", "isClassicPolicy", "isS08Elite", "isPhaseBoundaryPolicy", "isPathologicalPolicy"]
            ),
            "detail": canonical_json(
                {
                    column: int(feature_df[column].fillna(False).astype(bool).sum())
                    for column in ["isNullPolicy", "isClassicPolicy", "isS08Elite", "isPhaseBoundaryPolicy", "isPathologicalPolicy"]
                }
            ),
        },
        {
            "checkId": "normalized_matrix_finite",
            "success": len(normalized_matrix_df) == policy_count
            and normalized_matrix_df.shape[1] >= 25
            and np.isfinite(normalized_matrix_df.drop(columns=["policyId"]).to_numpy(dtype=float)).all(),
            "detail": f"{normalized_matrix_df.shape[1] - 1} normalized features for {len(normalized_matrix_df)} policies",
        },
        {
            "checkId": "normalization_spec_complete",
            "success": len(normalization_df) == normalized_matrix_df.shape[1] - 1
            and (pd.to_numeric(normalization_df["scale"], errors="coerce") > 0).all(),
            "detail": f"{len(normalization_df)} normalization records",
        },
        {
            "checkId": "embedding_rows_complete",
            "success": len(embeddings_df) == expected_embedding_rows
            and {"pca", "spectral", "mds"}.issubset(methods)
            and len(seeds) >= 3,
            "detail": f"{len(embeddings_df)} embedding rows; expected {expected_embedding_rows}",
        },
        {
            "checkId": "embedding_coordinates_finite",
            "success": bool(
                len(embeddings_df)
                and np.isfinite(embeddings_df[["embeddingDim1", "embeddingDim2"]].to_numpy(dtype=float)).all()
            ),
            "detail": "2D embedding coordinates are finite",
        },
        {
            "checkId": "stability_documented",
            "success": len(stability_df) >= 12
            and np.isfinite(
                stability_df[["procrustesDisparity", "pairwiseDistanceSpearman", "meanNeighborJaccard"]].to_numpy(dtype=float)
            ).all(),
            "detail": f"{len(stability_df)} seed/method stability comparisons",
        },
        {
            "checkId": "neighbors_documented",
            "success": len(neighbors_df) >= policy_count * 5,
            "detail": f"{len(neighbors_df)} nearest-neighbor rows",
        },
        {
            "checkId": "figures_written",
            "success": all(bool(path) for path in figure_paths),
            "detail": canonical_json(list(figure_paths)),
        },
        {
            "checkId": "repo_unit_tests",
            "success": bool(repo_test_payload.get("success")),
            "detail": f"{repo_test_payload.get('command')} returned {repo_test_payload.get('returnCode')}",
        },
        {
            "checkId": "s11_not_started",
            "success": not s11_dir_exists,
            "detail": "S11 artifact directory is absent",
        },
    ]
    return pd.DataFrame(checks)
