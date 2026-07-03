"""Helpers for E07 S10 universality-class taxonomy."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


UNIVERSALITY_SCHEMA_VERSION = "eidosoma.e07.universality_classes.v1"

SUPPORTED_STATUSES = {
    "behavior_profile_supported",
    "supported_policy_evidence",
    "metric_profile_supported",
}


def stable_fraction(label: str, *, salt: str = "e07-s10") -> float:
    """Return a deterministic pseudorandom fraction for stable ordering."""

    digest = hashlib.sha256(f"{salt}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def finite_standardize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a finite, z-scored numeric frame with constant columns removed."""

    numeric = frame.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if numeric.empty:
        return numeric.copy(), {"inputFeatureCount": 0, "retainedFeatureCount": 0, "droppedConstantFeatureCount": 0}
    means = numeric.mean(axis=0)
    stds = numeric.std(axis=0, ddof=0)
    retained = stds > 0
    scaled = (numeric.loc[:, retained] - means.loc[retained]) / stds.loc[retained]
    return scaled.fillna(0.0), {
        "inputFeatureCount": int(numeric.shape[1]),
        "retainedFeatureCount": int(retained.sum()),
        "droppedConstantFeatureCount": int((~retained).sum()),
    }


def dominance(values: Sequence[Any]) -> tuple[str, float, int]:
    """Return dominant label, dominant fraction, and unique-label count."""

    series = pd.Series(list(values), dtype="object").fillna("__missing__").astype(str).replace("", "__missing__")
    if series.empty:
        return "__missing__", float("nan"), 0
    counts = series.value_counts(dropna=False)
    return str(counts.index[0]), float(counts.iloc[0] / len(series)), int(counts.size)


def classify_cluster_status(row: Mapping[str, Any]) -> str:
    """Classify whether a cluster is bounded-interpretable or constraining."""

    entity_count = int(row.get("entity_count", 0) or 0)
    stability = pd.to_numeric(pd.Series([row.get("stability_mean_ari")]), errors="coerce").iloc[0]
    source_rate = pd.to_numeric(pd.Series([row.get("source_dominance_rate")]), errors="coerce").iloc[0]
    support_rate = pd.to_numeric(pd.Series([row.get("support_status_dominance_rate")]), errors="coerce").iloc[0]
    support_status = str(row.get("dominant_support_status", ""))
    control_max = pd.to_numeric(pd.Series([row.get("max_control_ari")]), errors="coerce").iloc[0]

    if entity_count < 3:
        return "small_or_uninterpretable_constraint"
    if np.isfinite(stability) and stability < 0.50:
        return "unstable_constraint"
    if np.isfinite(source_rate) and source_rate > 0.90:
        return "source_dominated_constraint"
    if support_status not in SUPPORTED_STATUSES and np.isfinite(support_rate) and support_rate > 0.80:
        return "missingness_driven_constraint"
    if np.isfinite(control_max) and control_max > 0.80:
        return "control_aligned_constraint"
    return "bounded_interpretable_class"


def cautious_cluster_label(row: Mapping[str, Any]) -> str:
    """Build a conservative, non-causal label from cluster metadata."""

    status = str(row.get("class_status", ""))
    entity_type = str(row.get("entity_type", "entity"))
    family = str(row.get("dominant_family", "mixed")).replace("_", " ")
    support_status = str(row.get("dominant_support_status", "")).replace("_", " ")

    if status == "source_dominated_constraint":
        return f"{entity_type} source-dominated cluster"
    if status == "missingness_driven_constraint":
        return f"{entity_type} support/missingness-driven cluster"
    if status == "unstable_constraint":
        return f"unstable {entity_type} cluster"
    if status == "control_aligned_constraint":
        return f"{entity_type} cluster aligned with control baseline"
    if status == "small_or_uninterpretable_constraint":
        return f"small {entity_type} cluster"
    if entity_type == "policy":
        memory = float(row.get("mean_policy_memory_depth_proxy", 0.0) or 0.0)
        stochastic = float(row.get("mean_policy_stochastic_choice", 0.0) or 0.0)
        if memory >= 1.0 and stochastic >= 0.25:
            return "supported policy group with memory and stochastic-choice proxies"
        if memory >= 1.0:
            return "supported policy group with memory-depth proxy"
        if stochastic >= 0.25:
            return "supported policy group with stochastic-choice proxy"
        return f"supported policy group dominated by {family}"
    if entity_type == "world":
        damage = float(row.get("mean_world_has_damage_or_repair", 0.0) or 0.0)
        frozen = float(row.get("mean_world_frozen_count", 0.0) or 0.0)
        if damage >= 0.5 and frozen > 0:
            return "supported world group with damage/repair and frozen perturbation proxies"
        if damage >= 0.5:
            return "supported world group with damage/repair proxy"
        if frozen > 0:
            return "supported world group with frozen perturbation proxy"
        return f"supported world group dominated by {family}"
    if entity_type == "goal":
        return f"supported goal group dominated by {family}"
    return f"{entity_type} group dominated by {family} ({support_status})"


def neighbor_distance_matrix(
    neighbors: pd.DataFrame,
    *,
    entity_type: str,
    distance_variant: str,
    entity_ids: Sequence[str],
    fill_multiplier: float = 1.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a symmetric approximate distance matrix from top-k neighbor rows."""

    ids = list(map(str, entity_ids))
    index = {entity_id: idx for idx, entity_id in enumerate(ids)}
    if not ids:
        return pd.DataFrame(), {"knownEdgeCount": 0, "fillDistance": float("nan"), "entityCount": 0}
    sub = neighbors[
        (neighbors["entity_type"].astype(str) == entity_type)
        & (neighbors["distance_variant"].astype(str) == distance_variant)
    ].copy()
    sub = sub[sub["query_entity_id"].astype(str).isin(index) & sub["neighbor_entity_id"].astype(str).isin(index)]
    finite = pd.to_numeric(sub.get("distance", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    fill_distance = float(finite.max() * fill_multiplier) if not finite.empty and finite.max() > 0 else 1.0
    matrix = np.full((len(ids), len(ids)), fill_distance, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    edge_count = 0
    for row in sub.to_dict(orient="records"):
        q = index.get(str(row.get("query_entity_id")))
        n = index.get(str(row.get("neighbor_entity_id")))
        if q is None or n is None:
            continue
        distance = pd.to_numeric(pd.Series([row.get("distance")]), errors="coerce").iloc[0]
        if not np.isfinite(distance):
            continue
        value = float(distance)
        matrix[q, n] = min(matrix[q, n], value)
        matrix[n, q] = min(matrix[n, q], value)
        edge_count += 1
    return pd.DataFrame(matrix, index=ids, columns=ids), {
        "knownEdgeCount": int(edge_count),
        "fillDistance": fill_distance,
        "entityCount": int(len(ids)),
    }


def validate_universality_artifacts(
    class_summary: pd.DataFrame,
    assignments: pd.DataFrame,
    stability: pd.DataFrame,
    controls: pd.DataFrame,
    exemplars: pd.DataFrame,
    counterexamples: pd.DataFrame,
    validation_context: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Validate S10 taxonomy artifacts and expected conservative controls."""

    validation_context = dict(validation_context or {})
    checks: list[dict[str, Any]] = []
    required_summary = {"entity_type", "branch_name", "class_id", "class_status", "entity_count", "cautious_label"}
    required_assignments = {"entity_type", "branch_name", "entity_id", "class_id", "map_x", "map_y"}
    checks.append(
        {
            "validation_case": "class_summary_required_columns_present",
            "success": required_summary <= set(class_summary.columns) and not class_summary.empty,
            "detail": f"rows={len(class_summary)} missing={sorted(required_summary - set(class_summary.columns))}",
        }
    )
    checks.append(
        {
            "validation_case": "assignments_required_columns_present",
            "success": required_assignments <= set(assignments.columns) and not assignments.empty,
            "detail": f"rows={len(assignments)} missing={sorted(required_assignments - set(assignments.columns))}",
        }
    )
    observed_primary = set(
        class_summary.loc[class_summary.get("branch_role", pd.Series(dtype=str)).astype(str).eq("primary"), "entity_type"].astype(str)
    )
    checks.append(
        {
            "validation_case": "primary_supported_branches_present",
            "success": {"policy", "goal", "world"} <= observed_primary,
            "detail": f"primary entity types={sorted(observed_primary)}",
        }
    )
    observed_roles = set(class_summary.get("branch_role", pd.Series(dtype=str)).astype(str))
    checks.append(
        {
            "validation_case": "sensitivity_branches_present",
            "success": "sensitivity" in observed_roles,
            "detail": f"branch roles={sorted(observed_roles)}",
        }
    )
    control_types = set(controls.get("control_family", pd.Series(dtype=str)).astype(str))
    checks.append(
        {
            "validation_case": "source_metric_missingness_controls_present",
            "success": {"source_metadata", "metric_only"} <= control_types and not controls.empty,
            "detail": f"control families={sorted(control_types)}",
        }
    )
    checks.append(
        {
            "validation_case": "stability_evaluated",
            "success": not stability.empty and {"mean_adjusted_rand_index", "branch_name"} <= set(stability.columns),
            "detail": f"stability rows={len(stability)}",
        }
    )
    finite_maps = not assignments.empty and np.isfinite(pd.to_numeric(assignments["map_x"], errors="coerce")).all() and np.isfinite(
        pd.to_numeric(assignments["map_y"], errors="coerce")
    ).all()
    checks.append(
        {
            "validation_case": "map_coordinates_finite",
            "success": bool(finite_maps),
            "detail": "all map coordinates finite" if finite_maps else "non-finite map coordinate found",
        }
    )
    stable_features = set(validation_context.get("stableInvariantFeatures", []))
    used_features = set(validation_context.get("usedInvariantFeatures", []))
    checks.append(
        {
            "validation_case": "only_s09_stable_invariants_used",
            "success": bool(used_features) and used_features <= stable_features,
            "detail": f"used={sorted(used_features)} stable={sorted(stable_features)}",
        }
    )
    constraining = class_summary.get("class_status", pd.Series(dtype=str)).astype(str).str.contains("constraint", regex=False).sum()
    checks.append(
        {
            "validation_case": "constraining_clusters_flagged",
            "success": int(constraining) > 0,
            "detail": f"constraining class rows={int(constraining)}",
        }
    )
    checks.append(
        {
            "validation_case": "exemplars_and_counterexamples_recorded",
            "success": not exemplars.empty and not counterexamples.empty,
            "detail": f"exemplar rows={len(exemplars)} counterexample rows={len(counterexamples)}",
        }
    )
    return pd.DataFrame(checks)
