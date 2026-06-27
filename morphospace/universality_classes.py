"""Empirical universality-class helpers for E03 S11.

S11 assigns computational policy families from completed S10 behavior features,
embeddings, and neighbors.  The labels are empirical cluster descriptions over
bounded simulator evidence; they are not mathematical universality proofs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .competence import canonical_json


UNIVERSALITY_CLASS_VERSION = "e03_s11_universality_classes.v1"
DEFAULT_CANDIDATE_K = tuple(range(5, 13))
DEFAULT_CLUSTER_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_REDUCTION_COMPONENTS = 24


def _feature_columns(normalized_matrix_df: pd.DataFrame) -> list[str]:
    return [column for column in normalized_matrix_df.columns if column != "policyId"]


def _as_feature_array(normalized_matrix_df: pd.DataFrame, columns: Sequence[str] | None = None) -> np.ndarray:
    feature_columns = list(columns) if columns is not None else _feature_columns(normalized_matrix_df)
    if not feature_columns:
        raise ValueError("at least one numeric feature column is required")
    x = normalized_matrix_df[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("normalized feature matrix contains non-finite values")
    return x


def reduce_feature_array(
    x: np.ndarray,
    *,
    seed: int = 0,
    n_components: int = DEFAULT_REDUCTION_COMPONENTS,
) -> tuple[np.ndarray, int, float]:
    """Return a bounded PCA representation for clustering and its explained variance."""

    if x.ndim != 2:
        raise ValueError("feature array must be two-dimensional")
    if x.shape[0] < 3:
        raise ValueError("at least three policies are required")
    max_components = max(1, min(int(n_components), x.shape[0] - 1, x.shape[1]))
    if x.shape[1] <= max_components:
        return x.astype(float, copy=True), int(x.shape[1]), 1.0
    model = PCA(n_components=max_components, svd_solver="randomized", random_state=int(seed))
    reduced = model.fit_transform(x)
    return reduced.astype(float, copy=False), int(max_components), float(np.sum(model.explained_variance_ratio_))


def _fit_kmeans(x: np.ndarray, *, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if n_clusters >= x.shape[0]:
        raise ValueError("n_clusters must be smaller than the policy count")
    model = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init=20, algorithm="lloyd")
    labels = model.fit_predict(x)
    distances = model.transform(x)[np.arange(len(labels)), labels]
    if len(set(map(int, labels))) > 1 and len(set(map(int, labels))) < len(labels):
        sample_size = min(1000, len(labels))
        silhouette = float(silhouette_score(x, labels, sample_size=sample_size, random_state=int(seed)))
    else:
        silhouette = float("nan")
    return labels.astype(int), distances.astype(float), float(model.inertia_), silhouette


def candidate_cluster_table(
    normalized_matrix_df: pd.DataFrame,
    *,
    candidate_k: Iterable[int] = DEFAULT_CANDIDATE_K,
    seed: int = 0,
    n_components: int = DEFAULT_REDUCTION_COMPONENTS,
) -> pd.DataFrame:
    """Score candidate k values on the S10 normalized feature matrix."""

    x = _as_feature_array(normalized_matrix_df)
    reduced, component_count, explained = reduce_feature_array(x, seed=seed, n_components=n_components)
    rows: list[dict[str, Any]] = []
    for k in candidate_k:
        if int(k) < 2 or int(k) >= len(normalized_matrix_df):
            continue
        labels, _, inertia, silhouette = _fit_kmeans(reduced, n_clusters=int(k), seed=seed)
        counts = np.bincount(labels, minlength=int(k))
        rows.append(
            {
                "universalityClassVersion": UNIVERSALITY_CLASS_VERSION,
                "candidateK": int(k),
                "seed": int(seed),
                "pcaComponentCount": int(component_count),
                "pcaExplainedVariance": float(explained),
                "silhouetteScore": silhouette,
                "inertia": inertia,
                "minClusterSize": int(counts.min()),
                "medianClusterSize": float(np.median(counts)),
                "maxClusterSize": int(counts.max()),
                "clusterSizesJson": canonical_json([int(value) for value in counts]),
            }
        )
    if not rows:
        raise ValueError("no valid candidate cluster counts were available")
    table = pd.DataFrame(rows)
    table["selectedBySilhouette"] = False
    best_index = table["silhouetteScore"].astype(float).idxmax()
    table.loc[best_index, "selectedBySilhouette"] = True
    return table


def primary_cluster_assignments(
    normalized_matrix_df: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    candidate_k: Iterable[int] = DEFAULT_CANDIDATE_K,
    seed: int = 0,
    n_components: int = DEFAULT_REDUCTION_COMPONENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign every policy to a primary empirical class."""

    candidate_df = candidate_cluster_table(normalized_matrix_df, candidate_k=candidate_k, seed=seed, n_components=n_components)
    if n_clusters is None:
        selected = int(candidate_df.loc[candidate_df["selectedBySilhouette"], "candidateK"].iloc[0])
    else:
        selected = int(n_clusters)
    x = _as_feature_array(normalized_matrix_df)
    reduced, component_count, explained = reduce_feature_array(x, seed=seed, n_components=n_components)
    labels, distances, inertia, silhouette = _fit_kmeans(reduced, n_clusters=selected, seed=seed)
    label_counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    label_to_class = {int(label): f"UC{rank:02d}" for rank, label in enumerate(label_counts.index, start=1)}
    rows: list[dict[str, Any]] = []
    policy_ids = normalized_matrix_df["policyId"].astype(str).tolist()
    for index, (policy_id, label, distance) in enumerate(zip(policy_ids, labels, distances, strict=True)):
        payload = {
            "universalityClassVersion": UNIVERSALITY_CLASS_VERSION,
            "policyId": str(policy_id),
            "universalityClassId": label_to_class[int(label)],
            "kMeansLabel": int(label),
            "clusterSeed": int(seed),
            "selectedClusterCount": int(selected),
            "distanceToCentroid": float(distance),
            "primaryClusteringSilhouette": silhouette,
            "primaryClusteringInertia": inertia,
            "pcaComponentCount": int(component_count),
            "pcaExplainedVariance": float(explained),
        }
        for component_index in range(min(5, reduced.shape[1])):
            payload[f"primaryPcaComponent{component_index + 1}"] = float(reduced[index, component_index])
        rows.append(payload)
    assignments = pd.DataFrame(rows).sort_values(["universalityClassId", "distanceToCentroid", "policyId"], kind="mergesort")
    return assignments.reset_index(drop=True), candidate_df


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _coalesced_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for column in columns:
        if column in df.columns:
            result = result.fillna(pd.to_numeric(df[column], errors="coerce"))
    return result


def _mean_finite(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def _median_finite(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return float("nan")
    return float(numeric.median())


def _bool_rate(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    values = df[column]
    if values.dtype == bool:
        return float(values.mean())
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return float((numeric > 0).mean())


def _value_counts_json(df: pd.DataFrame, column: str, *, limit: int = 8) -> str:
    if column not in df.columns:
        return "{}"
    counts = df[column].fillna("missing").astype(str).value_counts().head(limit)
    return canonical_json({str(key): int(value) for key, value in counts.items()})


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
    return value


def _class_label(summary: Mapping[str, Any], global_metrics: Mapping[str, float]) -> tuple[str, str]:
    completion = float(summary.get("meanCompletionSuccess", math.nan))
    oscillation = float(summary.get("meanOscillationProxy", math.nan))
    aggregation = float(summary.get("meanAggregationFinal", math.nan))
    chimera = float(summary.get("meanChimeraCompletion", math.nan))
    target_updates = float(summary.get("meanTargetUpdateCount", math.nan))
    swap_target = float(summary.get("meanSwapTargetActionCount", math.nan))
    stochastic = float(summary.get("meanStochasticActionCount", math.nan))
    state_keys = float(summary.get("meanStateKeyCount", math.nan))
    phase_rate = float(summary.get("phaseBoundaryPolicyRate", 0.0))
    classic_rate = float(summary.get("classicPolicyRate", 0.0))
    null_rate = float(summary.get("nullPolicyRate", 0.0))
    elite_rate = float(summary.get("elitePolicyRate", 0.0))
    pathological_rate = float(summary.get("pathologicalPolicyRate", 0.0))
    wait_count = float(summary.get("meanWaitActionCount", math.nan))
    swap_count = float(summary.get("meanSwapActionCount", math.nan))
    global_aggregation = float(global_metrics.get("meanAggregationFinal", math.nan))

    if null_rate >= 0.35 or (wait_count >= max(0.75, swap_count * 1.25) and completion <= 0.10):
        return (
            "null_or_waiting_controls",
            "Policies dominated by wait/null behavior with little sorting completion in the available panels.",
        )
    if elite_rate >= 0.08 and completion >= 0.45:
        return (
            "elite_reliable_local_sorters",
            "QD elites and classic-like local swappers with comparatively high small-array completion and sortedness.",
        )
    if classic_rate >= 0.25 and completion >= 0.45:
        return (
            "classic_local_inversion_sorters",
            "Classic or generated local-inversion policies that usually move inversions toward sorted arrays.",
        )
    if classic_rate >= 0.25 and completion < 0.30:
        return (
            "classic_inversion_failure_modes",
            "Classic-template regions where direction or guard choices fail the bounded S07-S09 tasks.",
        )
    if pathological_rate >= 0.50 or (oscillation >= 0.25 and completion <= 0.20):
        return (
            "pathological_oscillators_or_failure_modes",
            "Low-completion policies with elevated oscillation or repeated phase-boundary failure behavior.",
        )
    if phase_rate >= 0.45 and (target_updates >= 0.45 or swap_target >= 0.30):
        return (
            "targeted_phase_boundary_navigators",
            "Target-aware policies concentrated near detected phase boundaries.",
        )
    if phase_rate >= 0.35 and chimera >= 0.25:
        return (
            "chimera_boundary_navigators",
            "Boundary policies with measurable same-goal chimera or aggregation-sensitive behavior.",
        )
    if phase_rate >= 0.25:
        return (
            "phase_boundary_bridge_policies",
            "Policies repeatedly appearing on S09 transition edges between measured behavioral regimes.",
        )
    if stochastic >= 0.60:
        return (
            "stochastic_memory_wanderers",
            "Generated policies with stochastic and stateful primitives but weak bounded sorting evidence.",
        )
    if target_updates >= 0.50 or swap_target >= 0.30:
        return (
            "target_position_seekers",
            "Policies using target-position primitives without strong phase-boundary concentration.",
        )
    if state_keys >= 0.80:
        return (
            "memory_stateful_local_rules",
            "Stateful local rules whose behavior is represented mainly by structural and trace-memory features.",
        )
    if math.isfinite(aggregation) and math.isfinite(global_aggregation) and aggregation >= global_aggregation + 0.05:
        return (
            "aggregation_sensitive_local_rules",
            "Policies with above-background aggregation summaries in the bounded S07-S09 panels.",
        )
    return (
        "mixed_generated_local_rules",
        "A broad generated-rule region with limited evidence for a sharper S11 behavioral label.",
    )


def summarize_universality_classes(assignments_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per empirical class with role, metric, and label summaries."""

    merged = assignments_df.merge(feature_df, on="policyId", how="left", validate="one_to_one")
    completion_series = _coalesced_numeric(
        merged,
        ["s09_completionSuccessMean", "s07_completionSuccessMean", "s09Trace_completedNumericMean", "s07Trace_completedNumericMean"],
    )
    sortedness_series = _coalesced_numeric(
        merged,
        ["s09_finalSortednessScoreMean", "s07_finalSortednessScoreMean", "s09Trace_finalSortednessScoreMean"],
    )
    oscillation_series = _coalesced_numeric(merged, ["s09Trace_oscillationProxyMean", "s09_oscillationProxyMean", "s07_oscillationProxyMean"])
    aggregation_series = _coalesced_numeric(
        merged,
        ["s09_aggregationFinalMean", "s07_aggregationFinalMean", "s09Trace_finalAggregationMean", "s07Trace_finalAggregationMean"],
    )
    global_metrics = {
        "meanCompletionSuccess": _mean_finite(completion_series),
        "meanSortednessScore": _mean_finite(sortedness_series),
        "meanOscillationProxy": _mean_finite(oscillation_series),
        "meanAggregationFinal": _mean_finite(aggregation_series),
    }
    rows: list[dict[str, Any]] = []
    for class_id, group in merged.groupby("universalityClassId", sort=True):
        class_completion = _coalesced_numeric(
            group,
            ["s09_completionSuccessMean", "s07_completionSuccessMean", "s09Trace_completedNumericMean", "s07Trace_completedNumericMean"],
        )
        class_sortedness = _coalesced_numeric(
            group,
            ["s09_finalSortednessScoreMean", "s07_finalSortednessScoreMean", "s09Trace_finalSortednessScoreMean"],
        )
        class_oscillation = _coalesced_numeric(group, ["s09Trace_oscillationProxyMean", "s09_oscillationProxyMean", "s07_oscillationProxyMean"])
        class_aggregation = _coalesced_numeric(
            group,
            ["s09_aggregationFinalMean", "s07_aggregationFinalMean", "s09Trace_finalAggregationMean", "s07Trace_finalAggregationMean"],
        )
        summary: dict[str, Any] = {
            "universalityClassVersion": UNIVERSALITY_CLASS_VERSION,
            "universalityClassId": str(class_id),
            "policyCount": int(len(group)),
            "policyFraction": float(len(group) / max(1, len(merged))),
            "kMeansLabelsJson": canonical_json(sorted(map(int, group["kMeansLabel"].dropna().unique()))),
            "medianDistanceToCentroid": _median_finite(group["distanceToCentroid"]),
            "meanDistanceToCentroid": _mean_finite(group["distanceToCentroid"]),
            "primaryRoleMode": str(group["primaryRole"].fillna("missing").mode().iloc[0]) if "primaryRole" in group.columns else "missing",
            "primaryRoleCountsJson": _value_counts_json(group, "primaryRole"),
            "familyCountsJson": _value_counts_json(group, "family"),
            "generationMethodCountsJson": _value_counts_json(group, "generationMethod"),
            "classicFamilyCountsJson": _value_counts_json(group, "classicFamily"),
            "nullPolicyRate": _bool_rate(group, "isNullPolicy"),
            "classicPolicyRate": _bool_rate(group, "isClassicPolicy"),
            "elitePolicyRate": _bool_rate(group, "isS08Elite"),
            "phaseBoundaryPolicyRate": _bool_rate(group, "isPhaseBoundaryPolicy"),
            "pathologicalPolicyRate": _bool_rate(group, "isPathologicalPolicy"),
            "meanCompletionSuccess": _mean_finite(class_completion),
            "meanSortednessScore": _mean_finite(class_sortedness),
            "meanOscillationProxy": _mean_finite(class_oscillation),
            "meanAggregationFinal": _mean_finite(class_aggregation),
            "meanRobustnessScore": _mean_finite(_coalesced_numeric(group, ["s09_robustnessScoreMean", "s07_robustnessScoreMean"])),
            "meanDelayedGratification": _mean_finite(
                _coalesced_numeric(group, ["s09_delayedGratificationMean", "s07_delayedGratificationMean"])
            ),
            "meanChimeraCompletion": _mean_finite(
                _coalesced_numeric(group, ["s09_completionRateTaskFamily_chimera", "s07_completionRateTaskFamily_chimera"])
            ),
            "meanFrozenCompletion": _mean_finite(
                _coalesced_numeric(group, ["s09_completionRateTaskFamily_frozen", "s07_completionRateTaskFamily_frozen"])
            ),
            "meanDuplicateCompletion": _mean_finite(
                _coalesced_numeric(
                    group,
                    ["s09_completionRateTaskFamily_sorting_duplicate_values", "s07_completionRateTaskFamily_sorting_duplicate_values"],
                )
            ),
            "meanPhaseBoundaryInvolvementCount": _mean_finite(_numeric(group, "phaseBoundaryInvolvementCount")),
            "maxPhaseBoundaryInvolvementCount": float(_numeric(group, "phaseBoundaryInvolvementCount").max(skipna=True))
            if "phaseBoundaryInvolvementCount" in group.columns
            else float("nan"),
            "meanNearBoundaryVarianceRecords": _mean_finite(_numeric(group, "nearBoundaryVarianceRecordCount")),
            "meanS08EliteQualityScore": _mean_finite(_numeric(group, "s08Elite_qualityScore")),
            "meanActionTotalCount": _mean_finite(_numeric(group, "actionTotalCount")),
            "meanSwapActionCount": _mean_finite(_numeric(group, "swapActionCount")),
            "meanWaitActionCount": _mean_finite(_numeric(group, "waitActionCount")),
            "meanSwapTargetActionCount": _mean_finite(_numeric(group, "swapTargetActionCount")),
            "meanStochasticActionCount": _mean_finite(_numeric(group, "stochasticActionCount")),
            "meanTargetUpdateCount": _mean_finite(_numeric(group, "targetUpdateCount")),
            "meanStateKeyCount": _mean_finite(_numeric(group, "stateKeyCount")),
            "meanSignalCount": _mean_finite(_numeric(group, "signalCount")),
            "meanComplexityScore": _mean_finite(_numeric(group, "complexityScore")),
            "evidenceLayerCountMean": _mean_finite(_numeric(group, "evidenceLayerCount")),
        }
        label, interpretation = _class_label(summary, global_metrics)
        summary["classLabel"] = label
        summary["classInterpretation"] = interpretation
        summary["empiricalCaveat"] = "Empirical computational label from S07-S10 proxy artifacts; not a mathematical universality proof."
        rows.append(summary)
    table = pd.DataFrame(rows).sort_values("universalityClassId", kind="mergesort").reset_index(drop=True)
    table["className"] = table["universalityClassId"] + "_" + table["classLabel"]
    return table


def add_class_metadata(assignments_df: pd.DataFrame, class_summary_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Attach class names and policy metadata to assignment rows."""

    metadata_columns = [
        "policyId",
        "primaryRole",
        "family",
        "generationMethod",
        "lineageId",
        "classicFamily",
        "isNullPolicy",
        "isClassicPolicy",
        "isS08Elite",
        "isPhaseBoundaryPolicy",
        "isPathologicalPolicy",
        "phaseBoundaryInvolvementCount",
        "s08Elite_qualityScore",
        "s09_completionSuccessMean",
        "s07_completionSuccessMean",
        "s09_finalSortednessScoreMean",
        "s07_finalSortednessScoreMean",
        "s09Trace_oscillationProxyMean",
        "s07_oscillationProxyMean",
    ]
    available_metadata = [column for column in metadata_columns if column in feature_df.columns]
    class_columns = [
        "universalityClassId",
        "className",
        "classLabel",
        "classInterpretation",
        "empiricalCaveat",
    ]
    enriched = assignments_df.merge(class_summary_df[class_columns], on="universalityClassId", how="left", validate="many_to_one")
    enriched = enriched.merge(feature_df[available_metadata], on="policyId", how="left", validate="one_to_one")
    return enriched.sort_values(["universalityClassId", "distanceToCentroid", "policyId"], kind="mergesort").reset_index(drop=True)


def named_feature_subsets(normalized_matrix_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return S11 feature subsets for robustness checks."""

    columns = _feature_columns(normalized_matrix_df)

    def lower_has(column: str, terms: Sequence[str]) -> bool:
        lowered = column.lower()
        return any(term in lowered for term in terms)

    subsets = {
        "s07_s09_competence": [
            column
            for column in columns
            if (column.startswith("s07_") or column.startswith("s09_")) and not column.startswith(("s07Trace_", "s09Trace_"))
        ],
        "trajectory_phase_boundary": [
            column
            for column in columns
            if column.startswith(("s07Trace_", "s08Validation_", "s09Trace_"))
            or lower_has(column, ["boundary", "nearboundary", "transition"])
        ],
        "rule_structure_roles": [
            column
            for column in columns
            if lower_has(
                column,
                [
                    "action",
                    "rulecount",
                    "predicatecount",
                    "updatecount",
                    "statekey",
                    "stochastic",
                    "target",
                    "signal",
                    "lineagedepth",
                    "generationindex",
                    "complexity",
                    "isclassic",
                    "isnull",
                    "isphase",
                    "ispathological",
                ],
            )
            and not column.startswith(("s07_", "s09_", "s07Trace_", "s09Trace_", "s08Validation_"))
        ],
        "qd_elite_features": [column for column in columns if column.startswith("s08Elite_") or column == "isS08Elite"],
        "task_family_completion": [column for column in columns if "completionratetaskfamily" in column.lower()],
    }
    return {name: sorted(set(cols)) for name, cols in subsets.items() if cols}


def _cluster_for_comparison(
    policy_ids: Sequence[str],
    x: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    n_components: int,
) -> tuple[np.ndarray, float, int, float]:
    if x.shape[0] != len(policy_ids):
        raise ValueError("policy id count must match feature array rows")
    if x.shape[1] < 1:
        raise ValueError("comparison feature matrix has no columns")
    if float(np.nanstd(x)) <= 1e-12:
        raise ValueError("comparison feature matrix has near-zero variance")
    reduced, component_count, explained = reduce_feature_array(x, seed=seed, n_components=n_components)
    labels, _, _, silhouette = _fit_kmeans(reduced, n_clusters=n_clusters, seed=seed)
    return labels, silhouette, component_count, explained


def cluster_robustness_table(
    normalized_matrix_df: pd.DataFrame,
    primary_assignments_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    *,
    cluster_seeds: Sequence[int] = DEFAULT_CLUSTER_SEEDS,
    n_components: int = DEFAULT_REDUCTION_COMPONENTS,
) -> pd.DataFrame:
    """Compare primary classes against seeds, feature subsets, and S10 layouts."""

    policy_ids = normalized_matrix_df["policyId"].astype(str).tolist()
    primary_labels = (
        primary_assignments_df.set_index("policyId").loc[policy_ids, "universalityClassId"].astype(str).to_numpy()
    )
    n_clusters = int(primary_assignments_df["universalityClassId"].nunique())
    rows: list[dict[str, Any]] = []

    def add_row(
        *,
        comparison_id: str,
        comparison_type: str,
        feature_choice: str,
        seed: int,
        n_features: int,
        labels: np.ndarray | None = None,
        silhouette: float = float("nan"),
        component_count: int = 0,
        explained: float = float("nan"),
        success: bool = True,
        missing_reason: str = "",
    ) -> None:
        if labels is not None:
            counts = pd.Series(labels).value_counts()
            ari = float(adjusted_rand_score(primary_labels, labels))
            nmi = float(normalized_mutual_info_score(primary_labels, labels))
            min_size = int(counts.min())
            median_size = float(counts.median())
            max_size = int(counts.max())
        else:
            ari = float("nan")
            nmi = float("nan")
            min_size = 0
            median_size = float("nan")
            max_size = 0
        rows.append(
            {
                "universalityClassVersion": UNIVERSALITY_CLASS_VERSION,
                "comparisonId": comparison_id,
                "comparisonType": comparison_type,
                "featureChoice": feature_choice,
                "randomSeed": int(seed),
                "nPolicies": int(len(policy_ids)),
                "nFeatures": int(n_features),
                "clusterCount": int(n_clusters),
                "pcaComponentCount": int(component_count),
                "pcaExplainedVariance": float(explained),
                "silhouetteScore": float(silhouette),
                "adjustedRandIndex": ari,
                "normalizedMutualInfo": nmi,
                "minClusterSize": min_size,
                "medianClusterSize": median_size,
                "maxClusterSize": max_size,
                "success": bool(success),
                "missingReason": str(missing_reason),
            }
        )

    all_features = _feature_columns(normalized_matrix_df)
    add_row(
        comparison_id="primary_baseline",
        comparison_type="primary",
        feature_choice="all_s10_normalized_features",
        seed=0,
        n_features=len(all_features),
        labels=primary_labels,
        silhouette=float(primary_assignments_df["primaryClusteringSilhouette"].iloc[0]),
        component_count=int(primary_assignments_df["pcaComponentCount"].iloc[0]),
        explained=float(primary_assignments_df["pcaExplainedVariance"].iloc[0]),
    )
    for seed in cluster_seeds:
        if int(seed) == 0:
            continue
        try:
            labels, silhouette, component_count, explained = _cluster_for_comparison(
                policy_ids,
                _as_feature_array(normalized_matrix_df),
                n_clusters=n_clusters,
                seed=int(seed),
                n_components=n_components,
            )
            add_row(
                comparison_id=f"seed_repeat_{seed}",
                comparison_type="seed_repeat",
                feature_choice="all_s10_normalized_features",
                seed=int(seed),
                n_features=len(all_features),
                labels=labels,
                silhouette=silhouette,
                component_count=component_count,
                explained=explained,
            )
        except Exception as exc:  # pragma: no cover - defensive artifact row
            add_row(
                comparison_id=f"seed_repeat_{seed}",
                comparison_type="seed_repeat",
                feature_choice="all_s10_normalized_features",
                seed=int(seed),
                n_features=len(all_features),
                success=False,
                missing_reason=str(exc),
            )

    for subset_name, columns in named_feature_subsets(normalized_matrix_df).items():
        try:
            labels, silhouette, component_count, explained = _cluster_for_comparison(
                policy_ids,
                _as_feature_array(normalized_matrix_df, columns),
                n_clusters=n_clusters,
                seed=0,
                n_components=min(n_components, max(1, len(columns))),
            )
            add_row(
                comparison_id=f"feature_subset_{subset_name}",
                comparison_type="feature_subset",
                feature_choice=subset_name,
                seed=0,
                n_features=len(columns),
                labels=labels,
                silhouette=silhouette,
                component_count=component_count,
                explained=explained,
            )
        except Exception as exc:  # pragma: no cover - defensive artifact row
            add_row(
                comparison_id=f"feature_subset_{subset_name}",
                comparison_type="feature_subset",
                feature_choice=subset_name,
                seed=0,
                n_features=len(columns),
                success=False,
                missing_reason=str(exc),
            )

    required_embedding_cols = ["embeddingDim1", "embeddingDim2", "embeddingDim3", "embeddingDim4", "embeddingDim5"]
    if not embeddings_df.empty:
        for method in sorted(map(str, embeddings_df["embeddingMethod"].dropna().unique())):
            subset = embeddings_df[embeddings_df["embeddingMethod"].eq(method) & embeddings_df["embeddingSeed"].eq(0)].copy()
            subset["policyId"] = subset["policyId"].astype(str)
            subset = subset.set_index("policyId").reindex(policy_ids).reset_index()
            available = [column for column in required_embedding_cols if column in subset.columns]
            try:
                x = subset[available].to_numpy(dtype=float)
                labels, silhouette, component_count, explained = _cluster_for_comparison(
                    policy_ids,
                    x,
                    n_clusters=n_clusters,
                    seed=0,
                    n_components=min(n_components, len(available)),
                )
                add_row(
                    comparison_id=f"embedding_coordinates_{method}_seed0",
                    comparison_type="embedding_coordinates",
                    feature_choice=f"{method}_seed0_dims",
                    seed=0,
                    n_features=len(available),
                    labels=labels,
                    silhouette=silhouette,
                    component_count=component_count,
                    explained=explained,
                )
            except Exception as exc:  # pragma: no cover - defensive artifact row
                add_row(
                    comparison_id=f"embedding_coordinates_{method}_seed0",
                    comparison_type="embedding_coordinates",
                    feature_choice=f"{method}_seed0_dims",
                    seed=0,
                    n_features=len(available),
                    success=False,
                    missing_reason=str(exc),
                )
    return pd.DataFrame(rows)


def neighbor_class_alignment_table(neighbors_df: pd.DataFrame, assignments_df: pd.DataFrame) -> pd.DataFrame:
    """Measure whether S10 nearest-neighbor edges stay inside S11 classes."""

    if neighbors_df.empty:
        return pd.DataFrame(
            columns=[
                "universalityClassVersion",
                "universalityClassId",
                "className",
                "neighborEdgeCount",
                "meanSameClassNeighborRate",
                "meanFeatureDistance",
                "medianFeatureDistance",
            ]
        )
    class_lookup = assignments_df.set_index("policyId")[["universalityClassId", "className"]]
    edges = neighbors_df.copy()
    edges["policyId"] = edges["policyId"].astype(str)
    edges["neighborPolicyId"] = edges["neighborPolicyId"].astype(str)
    edges = edges.merge(
        class_lookup.rename(columns={"universalityClassId": "policyClassId", "className": "policyClassName"}),
        left_on="policyId",
        right_index=True,
        how="left",
    )
    edges = edges.merge(
        class_lookup.rename(columns={"universalityClassId": "neighborClassId", "className": "neighborClassName"}),
        left_on="neighborPolicyId",
        right_index=True,
        how="left",
    )
    edges["sameClassNeighbor"] = edges["policyClassId"].eq(edges["neighborClassId"])
    rows: list[dict[str, Any]] = []
    for class_id, group in edges.groupby("policyClassId", dropna=False, sort=True):
        class_name = str(group["policyClassName"].dropna().iloc[0]) if group["policyClassName"].notna().any() else "missing"
        rows.append(
            {
                "universalityClassVersion": UNIVERSALITY_CLASS_VERSION,
                "universalityClassId": str(class_id),
                "className": class_name,
                "neighborEdgeCount": int(len(group)),
                "meanSameClassNeighborRate": float(group["sameClassNeighbor"].mean()),
                "meanFeatureDistance": _mean_finite(group["featureDistance"]) if "featureDistance" in group.columns else float("nan"),
                "medianFeatureDistance": _median_finite(group["featureDistance"]) if "featureDistance" in group.columns else float("nan"),
            }
        )
    if rows:
        rows.append(
            {
                "universalityClassVersion": UNIVERSALITY_CLASS_VERSION,
                "universalityClassId": "ALL",
                "className": "all_classes",
                "neighborEdgeCount": int(len(edges)),
                "meanSameClassNeighborRate": float(edges["sameClassNeighbor"].mean()),
                "meanFeatureDistance": _mean_finite(edges["featureDistance"]) if "featureDistance" in edges.columns else float("nan"),
                "medianFeatureDistance": _median_finite(edges["featureDistance"]) if "featureDistance" in edges.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _policy_neighbor_payload(policy_id: str, neighbors_df: pd.DataFrame, assignments_df: pd.DataFrame, *, limit: int = 5) -> str:
    if neighbors_df.empty:
        return "[]"
    class_lookup = assignments_df.set_index("policyId")[["universalityClassId", "className"]]
    subset = neighbors_df[neighbors_df["policyId"].astype(str).eq(str(policy_id))].sort_values("neighborRank").head(limit)
    payload = []
    for _, row in subset.iterrows():
        neighbor_id = str(row["neighborPolicyId"])
        class_info = class_lookup.loc[neighbor_id] if neighbor_id in class_lookup.index else None
        payload.append(
            {
                "neighborPolicyId": neighbor_id,
                "neighborRank": int(row.get("neighborRank", len(payload) + 1)),
                "featureDistance": float(row.get("featureDistance", math.nan)),
                "neighborClassId": str(class_info["universalityClassId"]) if class_info is not None else "missing",
                "neighborClassName": str(class_info["className"]) if class_info is not None else "missing",
            }
        )
    return canonical_json(payload)


def class_exemplar_table(
    assignments_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    *,
    exemplars_per_class: int = 5,
) -> pd.DataFrame:
    """Pick centroid-near exemplars and attach local S10 neighbor context."""

    metadata_columns = [
        "policyId",
        "primaryRole",
        "family",
        "generationMethod",
        "lineageId",
        "classicFamily",
        "description",
        "isNullPolicy",
        "isClassicPolicy",
        "isS08Elite",
        "isPhaseBoundaryPolicy",
        "isPathologicalPolicy",
        "s09_completionSuccessMean",
        "s07_completionSuccessMean",
        "s09_finalSortednessScoreMean",
        "s07_finalSortednessScoreMean",
        "s09Trace_oscillationProxyMean",
        "s07_oscillationProxyMean",
        "phaseBoundaryInvolvementCount",
        "s08Elite_qualityScore",
    ]
    available = [column for column in metadata_columns if column in feature_df.columns and column not in assignments_df.columns]
    if available:
        merged = assignments_df.merge(feature_df[["policyId", *available]], on="policyId", how="left", validate="one_to_one")
    else:
        merged = assignments_df.copy()
    rows: list[dict[str, Any]] = []
    for class_id, group in merged.groupby("universalityClassId", sort=True):
        ordered = group.sort_values(["distanceToCentroid", "policyId"], kind="mergesort").head(int(exemplars_per_class))
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            completion = _coalesced_numeric(pd.DataFrame([row]), ["s09_completionSuccessMean", "s07_completionSuccessMean"]).iloc[0]
            sortedness = _coalesced_numeric(
                pd.DataFrame([row]), ["s09_finalSortednessScoreMean", "s07_finalSortednessScoreMean"]
            ).iloc[0]
            oscillation = _coalesced_numeric(pd.DataFrame([row]), ["s09Trace_oscillationProxyMean", "s07_oscillationProxyMean"]).iloc[0]
            rows.append(
                {
                    "universalityClassVersion": UNIVERSALITY_CLASS_VERSION,
                    "universalityClassId": str(class_id),
                    "className": str(row["className"]),
                    "classLabel": str(row["classLabel"]),
                    "exemplarRank": int(rank),
                    "selectionReason": "nearest_to_primary_cluster_centroid",
                    "policyId": str(row["policyId"]),
                    "primaryRole": str(row.get("primaryRole", "missing")),
                    "family": str(row.get("family", "missing")),
                    "generationMethod": str(row.get("generationMethod", "missing")),
                    "lineageId": str(row.get("lineageId", "missing")),
                    "classicFamily": str(row.get("classicFamily", "none")),
                    "distanceToCentroid": float(row["distanceToCentroid"]),
                    "completionSuccessMean": float(completion) if pd.notna(completion) else float("nan"),
                    "finalSortednessScoreMean": float(sortedness) if pd.notna(sortedness) else float("nan"),
                    "oscillationProxyMean": float(oscillation) if pd.notna(oscillation) else float("nan"),
                    "phaseBoundaryInvolvementCount": float(row.get("phaseBoundaryInvolvementCount", math.nan)),
                    "s08EliteQualityScore": float(row.get("s08Elite_qualityScore", math.nan)),
                    "nearestNeighborsJson": _policy_neighbor_payload(str(row["policyId"]), neighbors_df, assignments_df),
                }
            )
    return pd.DataFrame(rows)


def classic_null_elite_placement_table(
    assignments_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
) -> pd.DataFrame:
    """Document placements for classic, null, and S08 elite policies."""

    flag_columns = ["isClassicPolicy", "isNullPolicy", "isS08Elite"]
    metadata_columns = [
        "policyId",
        "primaryRole",
        "family",
        "generationMethod",
        "lineageId",
        "classicFamily",
        "isNullPolicy",
        "isClassicPolicy",
        "isS08Elite",
        "isPhaseBoundaryPolicy",
        "isPathologicalPolicy",
        "phaseBoundaryInvolvementCount",
        "s08Elite_qualityScore",
    ]
    available = [column for column in metadata_columns if column in feature_df.columns and column not in assignments_df.columns]
    if available:
        merged = assignments_df.merge(feature_df[["policyId", *available]], on="policyId", how="left", validate="one_to_one")
    else:
        merged = assignments_df.copy()
    special_mask = pd.Series(False, index=merged.index)
    for column in flag_columns:
        if column in merged.columns:
            special_mask = special_mask | merged[column].fillna(False).astype(bool)
    special = merged[special_mask].copy()

    embed = embeddings_df[embeddings_df["embeddingSeed"].eq(0)].copy() if not embeddings_df.empty else pd.DataFrame()
    embed_columns = [column for column in ["embeddingDim1", "embeddingDim2", "embeddingDim3", "embeddingDim4", "embeddingDim5"] if column in embed.columns]
    for method in sorted(map(str, embed["embeddingMethod"].dropna().unique())) if not embed.empty else []:
        subset = embed[embed["embeddingMethod"].eq(method)][["policyId", *embed_columns]].copy()
        subset["policyId"] = subset["policyId"].astype(str)
        subset = subset.rename(columns={column: f"{method}_{column}" for column in embed_columns})
        special = special.merge(subset, on="policyId", how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for _, row in special.sort_values(["primaryRole", "policyId"], kind="mergesort").iterrows():
        placement_role = []
        for name, column in [("classic", "isClassicPolicy"), ("null", "isNullPolicy"), ("s08_elite", "isS08Elite")]:
            if bool(row.get(column, False)):
                placement_role.append(name)
        payload = {
            "universalityClassVersion": UNIVERSALITY_CLASS_VERSION,
            "policyId": str(row["policyId"]),
            "placementRole": ",".join(placement_role),
            "universalityClassId": str(row["universalityClassId"]),
            "className": str(row["className"]),
            "classLabel": str(row["classLabel"]),
            "primaryRole": str(row.get("primaryRole", "missing")),
            "family": str(row.get("family", "missing")),
            "generationMethod": str(row.get("generationMethod", "missing")),
            "lineageId": str(row.get("lineageId", "missing")),
            "classicFamily": str(row.get("classicFamily", "none")),
            "distanceToCentroid": float(row["distanceToCentroid"]),
            "phaseBoundaryInvolvementCount": float(row.get("phaseBoundaryInvolvementCount", math.nan)),
            "s08EliteQualityScore": float(row.get("s08Elite_qualityScore", math.nan)),
            "nearestNeighborsJson": _policy_neighbor_payload(str(row["policyId"]), neighbors_df, assignments_df),
        }
        for column, value in row.items():
            if column.startswith(("pca_embeddingDim", "spectral_embeddingDim", "mds_embeddingDim")):
                payload[column] = float(value) if pd.notna(value) else float("nan")
        rows.append(payload)
    return pd.DataFrame(rows)


def identify_universality_classes(
    *,
    feature_df: pd.DataFrame,
    normalized_matrix_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    candidate_k: Iterable[int] = DEFAULT_CANDIDATE_K,
    seed: int = 0,
    n_components: int = DEFAULT_REDUCTION_COMPONENTS,
    exemplars_per_class: int = 5,
) -> dict[str, pd.DataFrame]:
    """Run the full S11 taxonomy workflow on completed S10 artifacts."""

    assignments, candidate_df = primary_cluster_assignments(
        normalized_matrix_df,
        candidate_k=candidate_k,
        seed=seed,
        n_components=n_components,
    )
    class_summary = summarize_universality_classes(assignments, feature_df)
    enriched_assignments = add_class_metadata(assignments, class_summary, feature_df)
    robustness = cluster_robustness_table(normalized_matrix_df, enriched_assignments, embeddings_df, n_components=n_components)
    neighbor_alignment = neighbor_class_alignment_table(neighbors_df, enriched_assignments)
    exemplars = class_exemplar_table(enriched_assignments, feature_df, neighbors_df, exemplars_per_class=exemplars_per_class)
    placements = classic_null_elite_placement_table(enriched_assignments, feature_df, embeddings_df, neighbors_df)
    return {
        "candidate_clusters": candidate_df,
        "class_assignments": enriched_assignments,
        "class_summary": class_summary,
        "cluster_robustness": robustness,
        "neighbor_alignment": neighbor_alignment,
        "class_exemplars": exemplars,
        "classic_null_elite_placements": placements,
    }


def validate_universality_outputs(
    *,
    feature_df: pd.DataFrame,
    normalized_matrix_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    class_assignments_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    exemplars_df: pd.DataFrame,
    placements_df: pd.DataFrame,
    neighbor_alignment_df: pd.DataFrame,
    upstream_statuses: Mapping[str, Mapping[str, Any]],
    repo_test_payload: Mapping[str, Any],
    taxonomy_report_exists: bool,
    figure_paths: Sequence[str],
    s12_dir_exists: bool,
) -> pd.DataFrame:
    """Return pass/fail validation rows for S11."""

    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append({"checkId": check_id, "success": bool(success), "detail": str(detail)})

    for step in ["S07", "S08", "S09", "S10"]:
        status = upstream_statuses.get(step, {})
        add(f"upstream_{step.lower()}_success", bool(status.get("success")), f"{step} status={status.get('status')}")

    feature_ids = set(map(str, feature_df["policyId"]))
    matrix_ids = set(map(str, normalized_matrix_df["policyId"]))
    assignment_ids = set(map(str, class_assignments_df["policyId"]))
    add(
        "policy_coverage",
        feature_ids == matrix_ids == assignment_ids and len(assignment_ids) == len(class_assignments_df),
        f"features={len(feature_ids)} matrix={len(matrix_ids)} assignments={len(assignment_ids)} rows={len(class_assignments_df)}",
    )
    class_count = int(class_summary_df["universalityClassId"].nunique()) if not class_summary_df.empty else 0
    add("class_count_range", 5 <= class_count <= 12, f"class_count={class_count}")
    add(
        "class_summary_nonempty",
        not class_summary_df.empty and int(class_summary_df["policyCount"].sum()) == len(class_assignments_df),
        f"summary_rows={len(class_summary_df)} summarized_policies={int(class_summary_df['policyCount'].sum()) if not class_summary_df.empty else 0}",
    )
    finite_robustness = robustness_df[robustness_df["success"].astype(bool)].copy() if not robustness_df.empty else pd.DataFrame()
    add(
        "robustness_comparisons",
        not finite_robustness.empty
        and {"seed_repeat", "feature_subset", "embedding_coordinates"}.issubset(set(finite_robustness["comparisonType"]))
        and np.isfinite(finite_robustness["adjustedRandIndex"]).all()
        and np.isfinite(finite_robustness["normalizedMutualInfo"]).all(),
        f"rows={len(robustness_df)} successful={len(finite_robustness)} types={sorted(set(finite_robustness.get('comparisonType', [])))}",
    )
    add(
        "neighbor_alignment",
        not neighbor_alignment_df.empty
        and "ALL" in set(neighbor_alignment_df["universalityClassId"].astype(str))
        and np.isfinite(pd.to_numeric(neighbor_alignment_df["meanSameClassNeighborRate"], errors="coerce")).all(),
        f"rows={len(neighbor_alignment_df)} neighbors={len(neighbors_df)}",
    )
    expected_min_exemplars = class_count * min(3, int(class_assignments_df.groupby("universalityClassId").size().min() if class_count else 0))
    add(
        "class_exemplars",
        len(exemplars_df) >= expected_min_exemplars and not exemplars_df.empty,
        f"exemplars={len(exemplars_df)} expected_min={expected_min_exemplars}",
    )
    placement_roles = ",".join(sorted(set(",".join(placements_df.get("placementRole", pd.Series(dtype=str)).astype(str)).split(","))))
    add(
        "classic_null_elite_placements",
        not placements_df.empty
        and {"classic", "null", "s08_elite"}.issubset(set(placement_roles.split(",")))
        and len(placements_df["policyId"].unique()) == len(placements_df),
        f"placements={len(placements_df)} roles={placement_roles}",
    )
    add(
        "s10_embedding_inputs_used",
        not embeddings_df.empty and not neighbors_df.empty and {"embeddingMethod", "neighborPolicyId"}.issubset(
            set(embeddings_df.columns) | set(neighbors_df.columns)
        ),
        f"embedding_rows={len(embeddings_df)} neighbor_rows={len(neighbors_df)}",
    )
    add("taxonomy_report_exists", taxonomy_report_exists, f"taxonomy_report_exists={taxonomy_report_exists}")
    add(
        "figure_outputs",
        len(figure_paths) >= 2 and all(bool(path) for path in figure_paths),
        f"figures={len(figure_paths)}",
    )
    add(
        "repo_unit_tests",
        bool(repo_test_payload.get("success")),
        f"returnCode={repo_test_payload.get('returnCode')} command={repo_test_payload.get('command')}",
    )
    add("no_s12_artifacts", not s12_dir_exists, f"s12_dir_exists={s12_dir_exists}")
    return pd.DataFrame(rows)
