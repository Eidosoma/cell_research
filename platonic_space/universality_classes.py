from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

from .behavior_predictor import DEFAULT_TARGET_COLUMNS
from .platonic_distances import embedding_columns
from .world_schema import compact_json


UNIVERSALITY_CLASS_SCHEMA_VERSION = "e07_s10_universality_classes.v1"
UNIVERSALITY_CLASS_MODEL_VERSION = "e07_s10_behavior_coordinate_taxonomy.v1"
UNIVERSALITY_CLASS_CLAIM_BOUNDARY = (
    "Empirical computational taxonomy over S04-S09 simulation-derived behavior records, S06 policy embeddings, "
    "S07 goal embeddings, S08 empirical distances, and upstream E03/E06 labels only. Classes are substrate-spanning "
    "behavioral neighborhoods in this corpus, not mathematical universality proofs, causal mechanisms, biological "
    "validation, evidence about living chimeras, clinical advice, cognition, agency, or sentience."
)
S09_CONSTRAINING_CAVEAT = (
    "S09 found no robust cross-world invariant predictor from the tested feature groups. S10 therefore treats "
    "S08 distance position, sparse coverage, and S09 interpretable features as descriptive diagnostics and caveats, "
    "not as accepted invariant laws."
)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _safe_label(value: Any, default: str = "unknown") -> str:
    text = _safe_text(value).strip()
    return text if text and text.lower() != "nan" else default


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _top_counts(values: Sequence[Any], limit: int = 5) -> dict[str, int]:
    counts = Counter(_safe_label(value) for value in values if _safe_label(value) != "unknown")
    return dict(counts.most_common(limit))


def _mode(values: Sequence[Any], default: str = "unknown") -> str:
    counts = _top_counts(values, limit=1)
    if not counts:
        return default
    return next(iter(counts))


def _base_pc_id(value: Any) -> str:
    match = re.search(r"(pc_[0-9a-f]{16})", _safe_text(value))
    return match.group(1) if match else ""


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _standardized_block(frame: pd.DataFrame, columns: Sequence[str], weight: float = 1.0, clip: float = 6.0, l2: bool = False) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=float)
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    values = StandardScaler().fit_transform(values)
    if clip > 0:
        values = np.clip(values, -clip, clip)
    if l2 and values.shape[1] > 0:
        values = normalize(values)
    return values * float(weight)


def policy_distance_matrix(distances: pd.DataFrame, policy_ids: Sequence[str]) -> np.ndarray:
    ids = [str(policy_id) for policy_id in policy_ids]
    index = {policy_id: i for i, policy_id in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    policy_distances = distances[distances["entityType"].astype(str).eq("policy")]
    for row in policy_distances[["entityIdA", "entityIdB", "platonicDistance"]].itertuples(index=False):
        a = str(row.entityIdA)
        b = str(row.entityIdB)
        if a not in index or b not in index:
            continue
        value = _finite(row.platonicDistance, default=math.nan)
        if math.isfinite(value):
            matrix[index[a], index[b]] = value
            matrix[index[b], index[a]] = value
    np.fill_diagonal(matrix, 0.0)
    missing_mask = (matrix == 0.0) & (~np.eye(len(ids), dtype=bool))
    if missing_mask.any():
        observed = matrix[~missing_mask]
        fill = float(np.nanmax(observed)) if observed.size else 1.0
        matrix[missing_mask] = fill
    return (matrix + matrix.T) / 2.0


def policy_s09_feature_aggregates(feature_frame: pd.DataFrame) -> pd.DataFrame:
    inv_policy_columns = [column for column in feature_frame.columns if column.startswith("inv__policy__")]
    rows: list[dict[str, Any]] = []
    for policy_id, subset in feature_frame.groupby("primaryPolicyId", dropna=False, sort=True):
        row: dict[str, Any] = {
            "abstractPolicyId": str(policy_id),
            "s09BehaviorRowCount": int(len(subset)),
            "s09ObservedWorldCount": int(subset["worldId"].nunique()) if "worldId" in subset.columns else 0,
            "s09ObservedGoalCount": int(subset["abstractGoalId"].nunique()) if "abstractGoalId" in subset.columns else 0,
        }
        for column in inv_policy_columns:
            row[f"s09_mean__{column}"] = float(pd.to_numeric(subset[column], errors="coerce").fillna(0.0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def policy_goal_context(feature_frame: pd.DataFrame, goal_embeddings: pd.DataFrame) -> pd.DataFrame:
    goal_cols = embedding_columns(goal_embeddings)
    goal_lookup = goal_embeddings.set_index("abstractGoalId", drop=False)
    rows: list[dict[str, Any]] = []
    for policy_id, subset in feature_frame.groupby("primaryPolicyId", dropna=False, sort=True):
        counts = subset["abstractGoalId"].astype(str).value_counts()
        ids = [goal_id for goal_id in counts.index if goal_id in goal_lookup.index]
        row: dict[str, Any] = {
            "abstractPolicyId": str(policy_id),
            "goalContextObservedGoalCount": int(subset["abstractGoalId"].nunique()) if "abstractGoalId" in subset.columns else 0,
            "goalContextEmbeddedGoalCount": int(len(ids)),
            "goalContextTopGoalFamily": _mode(subset.get("goalFamily", pd.Series(dtype=str)).tolist()),
            "goalContextTopGoalKind": _mode(subset.get("goalKind", pd.Series(dtype=str)).tolist()),
        }
        if ids:
            weights = counts.loc[ids].to_numpy(dtype=float)
            weights = weights / max(float(weights.sum()), 1e-9)
            values = goal_lookup.loc[ids, goal_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            averaged = np.average(values, axis=0, weights=weights)
            levels = goal_lookup.loc[ids, "sparseUncertaintyLevel"].astype(str).str.lower() if "sparseUncertaintyLevel" in goal_lookup.columns else pd.Series(dtype=str)
            row["goalContextHighSparseGoalFraction"] = float(np.average((levels == "high").to_numpy(dtype=float), weights=weights)) if len(levels) else 0.0
        else:
            averaged = np.zeros(len(goal_cols), dtype=float)
            row["goalContextHighSparseGoalFraction"] = 1.0
        for column, value in zip(goal_cols, averaged):
            row[f"goalContext_{column}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _e06_mosaic_policy_roles(mosaic: pd.DataFrame) -> pd.DataFrame:
    if mosaic.empty:
        return pd.DataFrame(columns=["sourcePolicyId", "e06MosaicModalClass", "e06MosaicInvolvementCount", "e06MosaicWinnerRate"])
    role_rows: list[dict[str, Any]] = []
    policy_ids = sorted(
        set(mosaic.get("winnerPolicyId", pd.Series(dtype=str)).dropna().astype(str))
        | set(mosaic.get("loserPolicyId", pd.Series(dtype=str)).dropna().astype(str))
        | set(mosaic.get("policyAPanelPolicyId", pd.Series(dtype=str)).dropna().astype(str))
        | set(mosaic.get("policyBPanelPolicyId", pd.Series(dtype=str)).dropna().astype(str))
    )
    for policy_id in policy_ids:
        involved = mosaic[
            mosaic.get("winnerPolicyId", pd.Series(dtype=str)).astype(str).eq(policy_id)
            | mosaic.get("loserPolicyId", pd.Series(dtype=str)).astype(str).eq(policy_id)
            | mosaic.get("policyAPanelPolicyId", pd.Series(dtype=str)).astype(str).eq(policy_id)
            | mosaic.get("policyBPanelPolicyId", pd.Series(dtype=str)).astype(str).eq(policy_id)
        ]
        if involved.empty:
            continue
        wins = int(involved.get("winnerPolicyId", pd.Series(dtype=str)).astype(str).eq(policy_id).sum())
        role_rows.append(
            {
                "sourcePolicyId": policy_id,
                "e06MosaicModalClass": _mode(involved.get("s07MosaicClass", pd.Series(dtype=str)).tolist()),
                "e06ContextDependencyClass": _mode(involved.get("s06ContextDependencyClass", pd.Series(dtype=str)).tolist()),
                "e06MosaicInvolvementCount": int(len(involved)),
                "e06MosaicWinnerRate": float(wins / len(involved)),
            }
        )
    return pd.DataFrame(role_rows)


def build_upstream_label_table(
    policy_catalog: pd.DataFrame,
    embedded_policy_ids: Sequence[str],
    e03_assignments: pd.DataFrame | None = None,
    e06_panel: pd.DataFrame | None = None,
    e06_dominance: pd.DataFrame | None = None,
    e06_mosaic: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = policy_catalog[["abstractPolicyId", "sourcePolicyId", "sourceExperimentId", "policyFamily", "policyKind", "policyLabel"]].copy()
    base = base[base["abstractPolicyId"].astype(str).isin({str(policy_id) for policy_id in embedded_policy_ids})].copy()
    base["abstractPolicyId"] = base["abstractPolicyId"].astype(str)
    base["sourcePolicyId"] = base["sourcePolicyId"].astype(str)
    base["sourceExperimentId"] = base["sourceExperimentId"].astype(str)
    base["sourcePolicyBasePcId"] = base["sourcePolicyId"].map(_base_pc_id)
    audit_rows: list[dict[str, Any]] = []

    if e03_assignments is not None and not e03_assignments.empty:
        e03_cols = [
            "policyId",
            "universalityClassId",
            "className",
            "classLabel",
            "classInterpretation",
            "primaryRole",
            "isNullPolicy",
            "isClassicPolicy",
            "isPhaseBoundaryPolicy",
            "isPathologicalPolicy",
        ]
        e03 = e03_assignments[[column for column in e03_cols if column in e03_assignments.columns]].copy()
        e03["policyId"] = e03["policyId"].astype(str)
        direct = e03.add_prefix("e03Direct_").rename(columns={"e03Direct_policyId": "sourcePolicyId"})
        base = base.merge(direct, on="sourcePolicyId", how="left")
        base_join = e03.add_prefix("e03Base_").rename(columns={"e03Base_policyId": "sourcePolicyBasePcId"})
        base = base.merge(base_join, on="sourcePolicyBasePcId", how="left")
        direct_match = base["e03Direct_classLabel"].notna() if "e03Direct_classLabel" in base.columns else pd.Series(False, index=base.index)
        base_match = base["e03Base_classLabel"].notna() if "e03Base_classLabel" in base.columns else pd.Series(False, index=base.index)
        for suffix in ("universalityClassId", "className", "classLabel", "classInterpretation", "primaryRole"):
            base[f"e03{suffix[0].upper()}{suffix[1:]}"] = base.get(f"e03Direct_{suffix}", pd.Series(index=base.index, dtype=object)).where(
                direct_match,
                base.get(f"e03Base_{suffix}", pd.Series(index=base.index, dtype=object)),
            )
        base["e03LabelJoinType"] = np.select([direct_match, (~direct_match) & base_match], ["direct_source_policy", "base_pc_inherited"], default="unmatched")
        audit_rows.append(
            {
                "labelSource": "E03 universality classes",
                "sourcePath": "/previous-artifacts/E03/research_steps/S11/policy_class_assignments.parquet",
                "sourceRows": int(len(e03_assignments)),
                "embeddedPolicyMatches": int((direct_match | base_match).sum()),
                "joinKey": "sourcePolicyId direct, then base pc_* id inherited for ablations",
                "usableLabelColumn": "e03ClassLabel",
                "caveat": "Base-PC inherited labels are upstream E03 class labels for the source policy, not revalidated labels for every E07 ablation row.",
            }
        )
    else:
        base["e03ClassLabel"] = np.nan
        base["e03LabelJoinType"] = "missing_source_file"
        audit_rows.append(
            {
                "labelSource": "E03 universality classes",
                "sourcePath": "/previous-artifacts/E03/research_steps/S11/policy_class_assignments.parquet",
                "sourceRows": 0,
                "embeddedPolicyMatches": 0,
                "joinKey": "not_available",
                "usableLabelColumn": "none",
                "caveat": "Direct E03 class artifact was not available.",
            }
        )

    if e06_panel is not None and not e06_panel.empty:
        panel_cols = ["panelPolicyId", "panelGroup", "policyFamily", "classLabel", "readinessStatus"]
        panel = e06_panel[[column for column in panel_cols if column in e06_panel.columns]].copy()
        panel["panelPolicyId"] = panel["panelPolicyId"].astype(str)
        panel = panel.add_prefix("e06Panel_").rename(columns={"e06Panel_panelPolicyId": "sourcePolicyId"})
        base = base.merge(panel, on="sourcePolicyId", how="left")
        panel_matches = base["e06Panel_classLabel"].notna() if "e06Panel_classLabel" in base.columns else pd.Series(False, index=base.index)
        audit_rows.append(
            {
                "labelSource": "E06 algotype panel",
                "sourcePath": "/previous-artifacts/E06/research_steps/S01/algotype_panel.parquet",
                "sourceRows": int(len(e06_panel)),
                "embeddedPolicyMatches": int(panel_matches.sum()),
                "joinKey": "sourcePolicyId to panelPolicyId",
                "usableLabelColumn": "e06Panel_classLabel",
                "caveat": "Panel labels are E06 source roles before E07 cross-substrate re-embedding.",
            }
        )
    else:
        base["e06Panel_classLabel"] = np.nan

    if e06_dominance is not None and not e06_dominance.empty:
        dominance_cols = ["panelPolicyId", "classLabel", "winRate", "resolvedWinRate", "weightedWinScore"]
        dominance = e06_dominance[[column for column in dominance_cols if column in e06_dominance.columns]].copy()
        dominance["panelPolicyId"] = dominance["panelPolicyId"].astype(str)
        dominance = dominance.add_prefix("e06Dominance_").rename(columns={"e06Dominance_panelPolicyId": "sourcePolicyId"})
        base = base.merge(dominance, on="sourcePolicyId", how="left")
        dominance_matches = base["e06Dominance_classLabel"].notna() if "e06Dominance_classLabel" in base.columns else pd.Series(False, index=base.index)
        audit_rows.append(
            {
                "labelSource": "E06 dominance hierarchy",
                "sourcePath": "/previous-artifacts/E06/results/e06_dominance_hierarchy.parquet",
                "sourceRows": int(len(e06_dominance)),
                "embeddedPolicyMatches": int(dominance_matches.sum()),
                "joinKey": "sourcePolicyId to panelPolicyId",
                "usableLabelColumn": "e06Dominance_classLabel",
                "caveat": "Dominance labels are pair/context summaries and remain computational proxies.",
            }
        )
    else:
        base["e06Dominance_classLabel"] = np.nan

    mosaic_roles = _e06_mosaic_policy_roles(e06_mosaic if e06_mosaic is not None else pd.DataFrame())
    if not mosaic_roles.empty:
        base = base.merge(mosaic_roles, on="sourcePolicyId", how="left")
        mosaic_matches = base["e06MosaicModalClass"].notna()
        audit_rows.append(
            {
                "labelSource": "E06 mosaic roles",
                "sourcePath": "/previous-artifacts/E06/results/e06_mosaic_classifications.parquet",
                "sourceRows": int(len(e06_mosaic)) if e06_mosaic is not None else 0,
                "embeddedPolicyMatches": int(mosaic_matches.sum()),
                "joinKey": "sourcePolicyId to winner/loser/panel policy IDs",
                "usableLabelColumn": "e06MosaicModalClass",
                "caveat": "Mosaic role summaries aggregate policy involvement across contexts and are not intrinsic policy traits.",
            }
        )
    else:
        base["e06MosaicModalClass"] = np.nan
        base["e06ContextDependencyClass"] = np.nan
        base["e06MosaicInvolvementCount"] = 0
        base["e06MosaicWinnerRate"] = np.nan

    base["e06RoleLabel"] = base.get("e06Dominance_classLabel", pd.Series(index=base.index, dtype=object))
    base["e06RoleLabel"] = base["e06RoleLabel"].fillna(base.get("e06Panel_classLabel", pd.Series(index=base.index, dtype=object)))
    base["e06RoleLabel"] = base["e06RoleLabel"].fillna(base.get("e06MosaicModalClass", pd.Series(index=base.index, dtype=object)))
    base["upstreamLabelForAgreement"] = np.where(
        base["sourceExperimentId"].eq("E03"),
        base.get("e03ClassLabel", pd.Series(index=base.index, dtype=object)),
        np.where(base["sourceExperimentId"].eq("E06"), base["e06RoleLabel"], base["policyFamily"]),
    )
    base["upstreamLabelForAgreement"] = pd.Series(base["upstreamLabelForAgreement"]).map(lambda value: _safe_label(value))
    return base, pd.DataFrame(audit_rows)


def build_policy_taxonomy_features(
    policy_embeddings: pd.DataFrame,
    goal_embeddings: pd.DataFrame,
    feature_frame: pd.DataFrame,
    policy_catalog: pd.DataFrame,
    distance_entities: pd.DataFrame,
    upstream_labels: pd.DataFrame,
    policy_behavior_profiles: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, list[str]]]:
    policies = policy_embeddings.copy()
    policies["abstractPolicyId"] = policies["abstractPolicyId"].astype(str)
    catalog_cols = ["abstractPolicyId", "sourcePolicyId", "sourceStepIds", "sourceArtifactPaths", "caveatsOrBlockers"]
    policies = policies.merge(policy_catalog[[column for column in catalog_cols if column in policy_catalog.columns]], on="abstractPolicyId", how="left")
    entity_cols = ["entityId", "sparseCoveragePenalty", "distanceUncertaintyLevel", "rowCount", "observedTargetCount"]
    entities = distance_entities[distance_entities["entityType"].astype(str).eq("policy")][[column for column in entity_cols if column in distance_entities.columns]].copy()
    entities = entities.rename(columns={"entityId": "abstractPolicyId", "rowCount": "s08RowCount", "observedTargetCount": "s08ObservedTargetCount"})
    policies = policies.merge(entities, on="abstractPolicyId", how="left")
    policies = policies.merge(policy_s09_feature_aggregates(feature_frame), on="abstractPolicyId", how="left")
    policies = policies.merge(policy_goal_context(feature_frame, goal_embeddings), on="abstractPolicyId", how="left")
    policies = policies.merge(upstream_labels.drop(columns=[column for column in ("policyLabel", "policyFamily", "policyKind", "sourceExperimentId") if column in upstream_labels.columns], errors="ignore"), on="abstractPolicyId", how="left")
    if policy_behavior_profiles is not None and not policy_behavior_profiles.empty:
        keep = ["abstractPolicyId"]
        for target in DEFAULT_TARGET_COLUMNS:
            for suffix in ("mean_z", "coverage"):
                column = f"feat::target::{target}::{suffix}"
                if column in policy_behavior_profiles.columns:
                    keep.append(column)
        policies = policies.merge(policy_behavior_profiles[keep], on="abstractPolicyId", how="left")

    policy_cols = embedding_columns(policies)
    goal_cols = sorted(column for column in policies.columns if column.startswith("goalContext_embedding_"))
    s09_cols = sorted(
        column
        for column in policies.columns
        if column.startswith("s09_mean__inv__policy__")
        and any(token in column for token in ("memory", "signaling", "feedback", "target_position", "stochasticity", "locality", "s08_sparse"))
    )
    feature_blocks = [
        _standardized_block(policies, policy_cols, weight=1.0, clip=6.0, l2=True),
        _standardized_block(policies, goal_cols, weight=0.25, clip=6.0, l2=True),
        _standardized_block(policies, s09_cols, weight=0.10, clip=6.0, l2=False),
    ]
    matrix = np.hstack([block for block in feature_blocks if block.shape[1] > 0])
    return policies, matrix, {"s06PolicyEmbeddingColumns": policy_cols, "s07GoalContextColumns": goal_cols, "s09DescriptiveFeatureColumns": s09_cols}


def _kmeans_labels(matrix: np.ndarray, cluster_count: int, random_state: int = 1710) -> np.ndarray:
    return KMeans(n_clusters=int(cluster_count), random_state=int(random_state), n_init=50, max_iter=500).fit_predict(matrix)


def s08_neighbor_alignment(distance_matrix: np.ndarray, labels: np.ndarray, neighbor_count: int = 10) -> tuple[float, float]:
    if len(labels) <= 1:
        return math.nan, math.nan
    order = np.argsort(distance_matrix, axis=1)
    fractions: list[float] = []
    for i in range(len(labels)):
        neighbors = [j for j in order[i] if j != i][:neighbor_count]
        if neighbors:
            fractions.append(float(np.mean(labels[neighbors] == labels[i])))
    expected = float(sum((np.bincount(labels) / len(labels)) ** 2))
    return float(np.mean(fractions)) if fractions else math.nan, expected


def select_cluster_count(
    matrix: np.ndarray,
    s08_distance_matrix: np.ndarray,
    cluster_counts: Sequence[int] = tuple(range(6, 13)),
    min_class_size: int = 8,
    max_class_fraction: float = 0.60,
    random_state: int = 1710,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_score = -math.inf
    best_k = int(cluster_counts[0])
    best_labels: np.ndarray | None = None
    for cluster_count in cluster_counts:
        labels = _kmeans_labels(matrix, cluster_count, random_state=random_state)
        sizes = np.bincount(labels)
        silhouette = float(silhouette_score(matrix, labels, metric="euclidean")) if len(set(labels)) > 1 else math.nan
        same_neighbor, expected_neighbor = s08_neighbor_alignment(s08_distance_matrix, labels)
        min_size = int(sizes.min())
        max_share = float(sizes.max() / len(labels))
        balance_penalty = max(0.0, max_share - max_class_fraction) + max(0.0, (min_class_size - min_size) / max(min_class_size, 1))
        score = silhouette + (0.15 * (same_neighbor - expected_neighbor)) - (0.40 * balance_penalty)
        eligible = min_size >= min_class_size and max_share <= max_class_fraction
        rows.append(
            {
                "candidateClusterCount": int(cluster_count),
                "selectionScore": float(score),
                "silhouetteEuclidean": silhouette,
                "s08SameClassNeighborFractionAt10": same_neighbor,
                "s08RandomExpectedSameClassFraction": expected_neighbor,
                "minClassSize": min_size,
                "maxClassSize": int(sizes.max()),
                "maxClassFraction": max_share,
                "eligible": bool(eligible),
                "classSizesJson": compact_json({str(i): int(size) for i, size in enumerate(sizes)}),
            }
        )
        if eligible and score > best_score:
            best_score = score
            best_k = int(cluster_count)
            best_labels = labels
    if best_labels is None:
        scored = sorted(rows, key=lambda row: row["selectionScore"], reverse=True)[0]
        best_k = int(scored["candidateClusterCount"])
        best_labels = _kmeans_labels(matrix, best_k, random_state=random_state)
    return best_k, best_labels, pd.DataFrame(rows).sort_values(["eligible", "selectionScore"], ascending=[False, False]).reset_index(drop=True)


def _cluster_medoid_indices(distance_matrix: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    medoids: dict[int, int] = {}
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        sub = distance_matrix[np.ix_(indices, indices)]
        means = sub.mean(axis=1) if len(indices) > 1 else np.array([0.0])
        medoids[int(label)] = int(indices[int(np.argmin(means))])
    return medoids


def _count_json(summary: Mapping[str, Any], key: str) -> dict[str, int]:
    parsed = _json_loads(summary.get(key), {})
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, int] = {}
    for label, count in parsed.items():
        try:
            out[str(label)] = int(count)
        except (TypeError, ValueError):
            continue
    return out


def _dominant_label(counts: Mapping[str, int], total: int, min_fraction: float = 0.30) -> str:
    if not counts or total <= 0:
        return ""
    label, count = max(counts.items(), key=lambda item: item[1])
    return label if count / total >= min_fraction else ""


def _infer_class_label(summary: Mapping[str, Any]) -> str:
    policy_count = int(_finite(summary.get("policyCount"), default=0.0))
    e03_counts = _count_json(summary, "topE03ClassLabelsJson")
    e06_counts = _count_json(summary, "topE06RoleLabelsJson")
    family_counts = _count_json(summary, "topPolicyFamiliesJson")
    kind_counts = _count_json(summary, "topPolicyKindsJson")
    goal_counts = _count_json(summary, "topGoalFamiliesJson")

    dominant_e03 = _dominant_label(e03_counts, policy_count, min_fraction=0.40)
    e03_name_map = {
        "null_or_waiting_controls": "null_or_waiting_controls",
        "pathological_oscillators_or_failure_modes": "pathological_oscillators_or_failure_modes",
        "targeted_phase_boundary_navigators": "phase_boundary_navigators",
        "chimera_boundary_navigators": "self_aggregating_boundary_modulators",
        "classic_inversion_failure_modes": "classic_inversion_failure_modes",
        "stochastic_memory_wanderers": "stochastic_memory_wanderers",
        "elite_reliable_local_sorters": "local_gradient_orderers",
        "mixed_generated_local_rules": "mixed_generated_local_rules",
    }
    if dominant_e03 in e03_name_map:
        return e03_name_map[dominant_e03]

    if sum(e06_counts.values()) >= 5:
        return "chimeric_dominance_and_mosaic_roles"

    family_text = " ".join(family_counts)
    kind_text = " ".join(kind_counts)
    goal_text = " ".join(goal_counts)
    text = " ".join(
        _safe_text(summary.get(key, ""))
        for key in (
            "topPolicyFamiliesJson",
            "topPolicyKindsJson",
            "topGoalFamiliesJson",
            "topE03ClassLabelsJson",
            "topE06RoleLabelsJson",
            "topE06MosaicClassesJson",
            "exemplarPolicyLabel",
        )
    ).lower()
    if "target_morphology" in goal_text or "symmetry" in goal_text or "morphology" in family_text or "morphology" in kind_text:
        return "morphogen_field_followers"
    if _finite(summary.get("meanS09MemoryDepth"), default=0.0) >= 0.7 or _finite(summary.get("meanS09FeedbackStrength"), default=0.0) >= 0.6:
        return "memory_repair_signalers"
    if "null_model" in kind_text or "null" in family_text or "wait" in family_text:
        return "null_or_waiting_controls"
    if any(token in text for token in ("pathological", "oscillat", "failure")):
        return "pathological_oscillators_or_failure_modes"
    if any(token in text for token in ("dominance", "governance", "mosaic", "chimera", "frontier_pair")):
        return "chimeric_dominance_and_mosaic_roles"
    if any(token in text for token in ("memory", "signal", "repair", "homeostasis", "bounded_counter", "diffusive")):
        return "memory_repair_signalers"
    if any(token in text for token in ("phase_boundary", "targeted", "barrier", "selection")):
        return "phase_boundary_navigators"
    if any(token in text for token in ("morphology", "symmetry", "field", "2d", "shape")):
        return "morphogen_field_followers"
    if any(token in text for token in ("aggregation", "boundary", "patchy", "layered", "polarized")):
        return "self_aggregating_boundary_modulators"
    if any(token in text for token in ("classic", "bubble", "insertion", "local", "sorting", "inversion")):
        return "local_gradient_orderers"
    if any(token in text for token in ("stochastic", "random", "wander")):
        return "stochastic_memory_wanderers"
    return "behavioral_neighborhood"


def _target_signal_summary(subset: pd.DataFrame, limit: int = 6) -> str:
    rows: list[dict[str, Any]] = []
    for target in DEFAULT_TARGET_COLUMNS:
        column = f"feat::target::{target}::mean_z"
        coverage_column = f"feat::target::{target}::coverage"
        if column not in subset.columns:
            continue
        values = pd.to_numeric(subset[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        rows.append(
            {
                "target": target,
                "meanZ": float(values.mean()),
                "absMeanZ": float(abs(values.mean())),
                "meanCoverage": float(pd.to_numeric(subset.get(coverage_column, pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()),
            }
        )
    rows = sorted(rows, key=lambda row: row["absMeanZ"], reverse=True)[:limit]
    return compact_json([{key: row[key] for key in ("target", "meanZ", "meanCoverage")} for row in rows])


def build_taxonomy_tables(
    policies: pd.DataFrame,
    feature_matrix: np.ndarray,
    s08_distance_matrix: np.ndarray,
    labels: np.ndarray,
    selected_cluster_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_ids = policies["abstractPolicyId"].astype(str).tolist()
    medoids = _cluster_medoid_indices(s08_distance_matrix, labels)
    raw_summaries: list[dict[str, Any]] = []
    for raw_label in sorted(set(labels), key=lambda label: (-int(np.sum(labels == label)), int(label))):
        indices = np.flatnonzero(labels == raw_label)
        subset = policies.iloc[indices].copy()
        medoid_idx = medoids[int(raw_label)]
        medoid_row = policies.iloc[medoid_idx]
        if len(indices) > 1:
            within = s08_distance_matrix[np.ix_(indices, indices)]
            mean_within = float(within[np.triu_indices(len(indices), k=1)].mean()) if len(indices) > 1 else 0.0
        else:
            mean_within = 0.0
        outside = np.flatnonzero(labels != raw_label)
        nearest_outside = float(s08_distance_matrix[np.ix_(indices, outside)].min()) if len(outside) else math.nan
        row = {
            "rawClusterLabel": int(raw_label),
            "policyCount": int(len(indices)),
            "policyFraction": float(len(indices) / len(policies)),
            "selectedClusterCount": int(selected_cluster_count),
            "s08MeanWithinClassDistance": mean_within,
            "s08NearestOutsideClassDistance": nearest_outside,
            "s08SeparationRatio": float(nearest_outside / max(mean_within, 1e-9)) if math.isfinite(nearest_outside) else math.nan,
            "exemplarPolicyId": str(policy_ids[medoid_idx]),
            "exemplarPolicyLabel": _safe_label(medoid_row.get("policyLabel")),
            "exemplarSourceExperimentId": _safe_label(medoid_row.get("sourceExperimentId")),
            "exemplarPolicyFamily": _safe_label(medoid_row.get("policyFamily")),
            "exemplarPolicyKind": _safe_label(medoid_row.get("policyKind")),
            "topSourceExperimentsJson": compact_json(_top_counts(subset["sourceExperimentId"].tolist())),
            "topPolicyFamiliesJson": compact_json(_top_counts(subset["policyFamily"].tolist())),
            "topPolicyKindsJson": compact_json(_top_counts(subset["policyKind"].tolist())),
            "topGoalFamiliesJson": compact_json(_top_counts(subset.get("goalContextTopGoalFamily", pd.Series(dtype=str)).tolist())),
            "topE03ClassLabelsJson": compact_json(_top_counts(subset.get("e03ClassLabel", pd.Series(dtype=str)).tolist())),
            "topE06RoleLabelsJson": compact_json(_top_counts(subset.get("e06RoleLabel", pd.Series(dtype=str)).tolist())),
            "topE06MosaicClassesJson": compact_json(_top_counts(subset.get("e06MosaicModalClass", pd.Series(dtype=str)).tolist())),
            "meanSparseCoveragePenalty": float(pd.to_numeric(subset.get("sparseCoveragePenalty", pd.Series(dtype=float)), errors="coerce").fillna(1.0).mean()),
            "meanS09MemoryDepth": float(pd.to_numeric(subset.get("s09_mean__inv__policy__memory_depth", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()),
            "meanS09FeedbackStrength": float(pd.to_numeric(subset.get("s09_mean__inv__policy__feedback_strength", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()),
            "meanS09SignalingHorizon": float(pd.to_numeric(subset.get("s09_mean__inv__policy__signaling_horizon", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()),
            "meanS09TargetPositionKnowledge": float(pd.to_numeric(subset.get("s09_mean__inv__policy__target_position_knowledge", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()),
            "meanS09StochasticityFlag": float(pd.to_numeric(subset.get("s09_mean__inv__policy__stochasticity_flag", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()),
            "behaviorSignalSummaryJson": _target_signal_summary(subset),
            "classCaveat": S09_CONSTRAINING_CAVEAT,
        }
        raw_summaries.append(row)

    seen_names: Counter[str] = Counter()
    class_rows: list[dict[str, Any]] = []
    raw_to_class: dict[int, tuple[str, str, str]] = {}
    for i, row in enumerate(raw_summaries, start=1):
        class_id = f"S10_UC{i:02d}"
        class_label = _infer_class_label(row)
        seen_names[class_label] += 1
        unique_label = class_label if seen_names[class_label] == 1 else f"{class_label}_{seen_names[class_label]}"
        class_name = f"{class_id}_{unique_label}"
        interpretation = (
            f"Behavioral neighborhood centered on `{row['exemplarPolicyLabel']}` with dominant policy families "
            f"{row['topPolicyFamiliesJson']} and goal context {row['topGoalFamiliesJson']}. "
            "Interpretation is empirical and inherits the S09 no-robust-invariant caveat."
        )
        raw_to_class[int(row["rawClusterLabel"])] = (class_id, class_name, unique_label)
        class_rows.append({**row, "universalityClassId": class_id, "className": class_name, "classLabel": unique_label, "classInterpretation": interpretation})
    class_summary = pd.DataFrame(class_rows).drop(columns=["rawClusterLabel"]).sort_values(["policyCount", "universalityClassId"], ascending=[False, True]).reset_index(drop=True)

    exemplar_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    medoid_by_raw = {raw_label: medoids[raw_label] for raw_label in medoids}
    medoid_indices = list(medoid_by_raw.values())
    for i, (policy_id, raw_label) in enumerate(zip(policy_ids, labels)):
        class_id, class_name, class_label = raw_to_class[int(raw_label)]
        own_medoid = medoid_by_raw[int(raw_label)]
        own_distance = float(s08_distance_matrix[i, own_medoid])
        other_medoids = [idx for raw, idx in medoid_by_raw.items() if raw != int(raw_label)]
        nearest_other = float(np.min(s08_distance_matrix[i, other_medoids])) if other_medoids else math.nan
        confidence = float(max(0.0, min(1.0, (nearest_other - own_distance) / max(nearest_other, 1e-9)))) if math.isfinite(nearest_other) else 1.0
        row = policies.iloc[i].to_dict()
        assignment_rows.append(
            {
                "abstractPolicyId": policy_id,
                "policyLabel": _safe_label(row.get("policyLabel")),
                "sourceExperimentId": _safe_label(row.get("sourceExperimentId")),
                "policyFamily": _safe_label(row.get("policyFamily")),
                "policyKind": _safe_label(row.get("policyKind")),
                "sourcePolicyId": _safe_label(row.get("sourcePolicyId")),
                "universalityClassId": class_id,
                "className": class_name,
                "classLabel": class_label,
                "selectedClusterCount": int(selected_cluster_count),
                "distanceToClassMedoidS08": own_distance,
                "nearestOtherClassMedoidDistanceS08": nearest_other,
                "classAssignmentConfidence": confidence,
                "sparseCoveragePenalty": _finite(row.get("sparseCoveragePenalty"), default=math.nan),
                "distanceUncertaintyLevel": _safe_label(row.get("distanceUncertaintyLevel")),
                "goalContextTopGoalFamily": _safe_label(row.get("goalContextTopGoalFamily")),
                "e03ClassLabel": _safe_label(row.get("e03ClassLabel")),
                "e03LabelJoinType": _safe_label(row.get("e03LabelJoinType")),
                "e06RoleLabel": _safe_label(row.get("e06RoleLabel")),
                "e06MosaicModalClass": _safe_label(row.get("e06MosaicModalClass")),
                "claimBoundary": UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
            }
        )
    assignments = pd.DataFrame(assignment_rows)

    for class_row in class_rows:
        class_id = class_row["universalityClassId"]
        raw_label = next(raw for raw, values in raw_to_class.items() if values[0] == class_id)
        indices = np.flatnonzero(labels == raw_label)
        medoid_idx = medoid_by_raw[raw_label]
        dists = s08_distance_matrix[indices, medoid_idx]
        order = np.argsort(dists)[:5]
        for rank, order_idx in enumerate(order, start=1):
            idx = int(indices[int(order_idx)])
            row = assignments.iloc[idx]
            exemplar_rows.append(
                {
                    "universalityClassId": class_id,
                    "className": class_row["className"],
                    "classLabel": class_row["classLabel"],
                    "exemplarRank": int(rank),
                    "selectionReason": "nearest_to_s08_class_medoid",
                    "abstractPolicyId": row["abstractPolicyId"],
                    "policyLabel": row["policyLabel"],
                    "sourceExperimentId": row["sourceExperimentId"],
                    "policyFamily": row["policyFamily"],
                    "policyKind": row["policyKind"],
                    "distanceToClassMedoidS08": float(dists[int(order_idx)]),
                    "classAssignmentConfidence": float(row["classAssignmentConfidence"]),
                    "claimBoundary": UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
                }
            )
    exemplars = pd.DataFrame(exemplar_rows)
    return assignments, class_summary, exemplars


def stability_tables(
    matrix: np.ndarray,
    s08_distance_matrix: np.ndarray,
    labels: np.ndarray,
    selected_cluster_count: int,
    random_state: int = 1710,
    resamples: int = 24,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    same_neighbor, expected_neighbor = s08_neighbor_alignment(s08_distance_matrix, labels)
    rows.append(
        {
            "validationTask": "s08_nearest_neighbor_class_coherence",
            "metricName": "same_class_fraction_at_10",
            "value": same_neighbor,
            "baselineValue": expected_neighbor,
            "replicateCount": 1,
            "interpretation": "Mean fraction of each policy's ten nearest S08-distance neighbors assigned to the same S10 class.",
        }
    )
    subset_scores: list[float] = []
    for replicate in range(resamples):
        size = max(selected_cluster_count + 1, int(round(0.80 * len(labels))))
        subset = np.sort(rng.choice(len(labels), size=size, replace=False))
        subset_labels = _kmeans_labels(matrix[subset], selected_cluster_count, random_state=random_state + replicate + 1)
        subset_scores.append(float(adjusted_rand_score(labels[subset], subset_labels)))
    rows.append(
        {
            "validationTask": "subsample_cluster_stability",
            "metricName": "adjusted_rand_index_vs_full_labels",
            "value": float(np.mean(subset_scores)),
            "baselineValue": 0.0,
            "replicateCount": int(resamples),
            "valueStd": float(np.std(subset_scores, ddof=1)) if len(subset_scores) > 1 else 0.0,
            "interpretation": "ARI comparing 80% subsample reclustering with the full S10 class labels on shared policies.",
        }
    )
    seed_scores: list[float] = []
    for seed in range(random_state + 1, random_state + 17):
        seed_labels = _kmeans_labels(matrix, selected_cluster_count, random_state=seed)
        seed_scores.append(float(adjusted_rand_score(labels, seed_labels)))
    rows.append(
        {
            "validationTask": "seed_stability",
            "metricName": "adjusted_rand_index_vs_selected_seed",
            "value": float(np.mean(seed_scores)),
            "baselineValue": 0.0,
            "replicateCount": int(len(seed_scores)),
            "valueStd": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0,
            "interpretation": "ARI comparing KMeans labels across random initializations.",
        }
    )
    try:
        s08_labels = AgglomerativeClustering(n_clusters=selected_cluster_count, metric="precomputed", linkage="average").fit_predict(s08_distance_matrix)
        s08_ari = float(adjusted_rand_score(labels, s08_labels))
    except ValueError:
        s08_ari = math.nan
    rows.append(
        {
            "validationTask": "s08_distance_tree_modality_agreement",
            "metricName": "adjusted_rand_index_vs_s08_average_linkage",
            "value": s08_ari,
            "baselineValue": 0.0,
            "replicateCount": 1,
            "interpretation": "Agreement with direct average-linkage clustering on S08 distances; low values indicate S08 tree outlier structure rather than a failed S10 taxonomy.",
        }
    )
    return pd.DataFrame(rows)


def _purity_score(predicted: Sequence[Any], reference: Sequence[Any]) -> float:
    frame = pd.DataFrame({"predicted": list(predicted), "reference": [_safe_label(value) for value in reference]})
    frame = frame[~frame["reference"].isin(["unknown", "nan", ""])]
    if frame.empty:
        return math.nan
    total = 0
    for _, subset in frame.groupby("predicted"):
        total += int(subset["reference"].value_counts().max())
    return float(total / len(frame))


def upstream_label_agreement(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_specs = [
        ("all", "sourceExperimentId", "sourceExperimentId"),
        ("all", "policyFamily", "policyFamily"),
        ("all", "policyKind", "policyKind"),
        ("E03", "e03ClassLabel", "E03 universality class label"),
        ("E06", "e06RoleLabel", "E06 panel/dominance role label"),
        ("E06", "e06MosaicModalClass", "E06 mosaic modal class"),
    ]
    for subset_name, column, description in label_specs:
        subset = assignments.copy()
        if subset_name != "all":
            subset = subset[subset["sourceExperimentId"].eq(subset_name)].copy()
        if column not in subset.columns:
            continue
        reference = subset[column].map(_safe_label)
        mask = ~reference.isin(["unknown", "nan", ""])
        subset = subset[mask].copy()
        reference = reference[mask]
        if subset.empty:
            rows.append(
                {
                    "comparisonScope": subset_name,
                    "referenceLabelColumn": column,
                    "referenceDescription": description,
                    "policyCount": 0,
                    "uniqueReferenceLabelCount": 0,
                    "adjustedRandIndex": math.nan,
                    "normalizedMutualInformation": math.nan,
                    "purity": math.nan,
                    "interpretation": "No joinable labels available for this comparison.",
                }
            )
            continue
        class_labels = subset["universalityClassId"].astype(str)
        rows.append(
            {
                "comparisonScope": subset_name,
                "referenceLabelColumn": column,
                "referenceDescription": description,
                "policyCount": int(len(subset)),
                "uniqueReferenceLabelCount": int(reference.nunique()),
                "adjustedRandIndex": float(adjusted_rand_score(reference, class_labels)) if reference.nunique() > 1 else math.nan,
                "normalizedMutualInformation": float(normalized_mutual_info_score(reference, class_labels)) if reference.nunique() > 1 else math.nan,
                "purity": _purity_score(class_labels, reference),
                "interpretation": "Agreement is descriptive only; upstream labels are source-specific and not S10 ground truth.",
            }
        )
    return pd.DataFrame(rows)


def validation_table(
    assignments: pd.DataFrame,
    class_summary: pd.DataFrame,
    stability: pd.DataFrame,
    agreement: pd.DataFrame,
    cluster_selection: pd.DataFrame,
    s09_candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def add(check_id: str, severity: str, success: bool, value: Any, threshold: Any, interpretation: str) -> None:
        rows.append(
            {
                "checkId": check_id,
                "severity": severity,
                "success": bool(success),
                "value": value,
                "threshold": threshold,
                "interpretation": interpretation,
            }
        )

    add("assignment_row_count", "error", len(assignments) >= 500, int(len(assignments)), ">=500", "S10 should cover the S06 behavior-embedded policy set.")
    add("class_count", "error", class_summary["universalityClassId"].nunique() >= 4, int(class_summary["universalityClassId"].nunique()), ">=4", "Taxonomy should contain multiple classes.")
    add("minimum_class_size", "warning", int(class_summary["policyCount"].min()) >= 8, int(class_summary["policyCount"].min()), ">=8", "Very small classes are allowed but flagged as sparse neighborhoods.")
    add("class_names_present", "error", assignments["className"].map(_safe_label).ne("unknown").all(), int(assignments["className"].map(_safe_label).ne("unknown").sum()), "all assignments", "Every policy needs a class name.")
    same_row = stability[stability["validationTask"].eq("s08_nearest_neighbor_class_coherence")]
    same_value = float(same_row["value"].iloc[0]) if not same_row.empty else math.nan
    baseline = float(same_row["baselineValue"].iloc[0]) if not same_row.empty else math.nan
    add("s08_neighbor_coherence", "error", math.isfinite(same_value) and same_value > baseline, same_value, f">{baseline:.4f}" if math.isfinite(baseline) else "finite baseline", "S10 classes should be enriched among S08 nearest neighbors.")
    subsample = stability[stability["validationTask"].eq("subsample_cluster_stability")]
    subsample_value = float(subsample["value"].iloc[0]) if not subsample.empty else math.nan
    add("subsample_stability", "warning", math.isfinite(subsample_value) and subsample_value >= 0.40, subsample_value, ">=0.40 ARI", "Subsample stability is a warning threshold, not a proof of class truth.")
    e03 = agreement[(agreement["comparisonScope"].eq("E03")) & (agreement["referenceLabelColumn"].eq("e03ClassLabel"))]
    e06 = agreement[(agreement["comparisonScope"].eq("E06")) & (agreement["referenceLabelColumn"].eq("e06RoleLabel"))]
    add("e03_label_coverage", "error", (not e03.empty) and int(e03["policyCount"].iloc[0]) >= 400, int(e03["policyCount"].iloc[0]) if not e03.empty else 0, ">=400 policies", "Most embedded E03 policies should have upstream class labels.")
    add("e06_label_coverage", "warning", (not e06.empty) and int(e06["policyCount"].iloc[0]) >= 10, int(e06["policyCount"].iloc[0]) if not e06.empty else 0, ">=10 policies", "E06 role labels are sparse but should be present for panel policies.")
    add("selection_candidates", "error", not cluster_selection.empty, int(len(cluster_selection)), ">0", "Cluster-count selection diagnostics must be written.")
    s09_tiers = set(s09_candidates.get("evidenceTier", pd.Series(dtype=str)).astype(str))
    add(
        "s09_constraining_caveat_propagated",
        "error",
        "unstable_cross_world_signal" in s09_tiers or "coverage_caveat" in s09_tiers,
        ",".join(sorted(s09_tiers)),
        "contains S09 caveat tiers",
        "S10 must carry forward S09's constraining invariant result.",
    )
    return pd.DataFrame(rows)
