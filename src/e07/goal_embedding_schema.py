"""Helpers for E07 S07 goal-embedding construction and validation."""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd


GOAL_EMBEDDING_SCHEMA_VERSION = "eidosoma.e07.goal_embedding.v1"


def stable_fraction(label: str, *, salt: str = "e07-s07") -> float:
    digest = hashlib.sha256(f"{salt}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def policy_support_weight(status: Any, observation_count: Any, *, min_supported_rows: int = 10) -> float:
    """Map S06 policy support status and behavior coverage to an uncertainty weight."""

    text = "" if status is None else str(status)
    try:
        count = float(observation_count)
    except (TypeError, ValueError):
        count = 0.0
    if text == "behavior_profile_supported":
        return 1.0
    if text == "sparse_behavior_profile" and count > 0:
        return float(np.clip(np.sqrt(count / max(1.0, float(min_supported_rows))), 0.10, 0.95))
    return 0.0


def source_balance_weights(frame: pd.DataFrame, source_column: str = "source_experiment_id") -> pd.Series:
    if frame.empty or source_column not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)
    counts = frame.groupby(source_column, observed=True).size().astype(float)
    median_count = float(counts.median()) if not counts.empty else 1.0
    weights = frame[source_column].map(lambda value: median_count / max(1.0, float(counts.get(value, 1.0))))
    return weights.clip(lower=0.05, upper=20.0).astype(float)


def weighted_frequency_profile(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    feature_column: str,
    weight_column: str,
) -> pd.DataFrame:
    """Return an entity x feature matrix of weighted frequency shares."""

    if frame.empty:
        return pd.DataFrame()
    required = [entity_column, feature_column, weight_column]
    work = frame[required].copy()
    work[weight_column] = pd.to_numeric(work[weight_column], errors="coerce").fillna(0.0)
    work = work.dropna(subset=[entity_column, feature_column])
    work = work[work[weight_column] > 0].copy()
    if work.empty:
        return pd.DataFrame()
    grouped = work.groupby([entity_column, feature_column], observed=True)[weight_column].sum()
    totals = grouped.groupby(level=0).sum().replace(0, np.nan)
    shares = grouped / totals
    return shares.unstack(feature_column).fillna(0.0)


def weighted_numeric_means(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    numeric_columns: Iterable[str],
    weight_column: str,
) -> pd.DataFrame:
    """Return weighted means for numeric columns by entity."""

    columns = list(numeric_columns)
    if frame.empty or not columns:
        return pd.DataFrame()
    work = frame[[entity_column, weight_column, *columns]].copy()
    work[weight_column] = pd.to_numeric(work[weight_column], errors="coerce").fillna(0.0)
    work = work[work[weight_column] > 0].copy()
    if work.empty:
        return pd.DataFrame()
    for column in columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        work[f"_weighted_{column}"] = work[column] * work[weight_column]
    grouped = work.groupby(entity_column, observed=True)
    weight_sums = grouped[weight_column].sum().replace(0, np.nan)
    out = pd.DataFrame(index=weight_sums.index)
    for column in columns:
        out[column] = grouped[f"_weighted_{column}"].sum() / weight_sums
    return out.fillna(0.0)


def unique_conflict_pairs(conflicts: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate directed S03 conflict rows into unordered canonical goal pairs."""

    required = {"canonical_goal_id_a", "canonical_goal_id_b", "conflict_type", "conflict_group_id"}
    if conflicts.empty or not required <= set(conflicts.columns):
        return pd.DataFrame(columns=["goal_a", "goal_b", "conflict_types", "conflict_group_ids"])
    rows: dict[tuple[str, str], dict[str, set[str]]] = {}
    for record in conflicts.to_dict(orient="records"):
        left = str(record.get("canonical_goal_id_a", ""))
        right = str(record.get("canonical_goal_id_b", ""))
        if not left or not right or left == right:
            continue
        key = tuple(sorted((left, right)))
        item = rows.setdefault(key, {"conflict_types": set(), "conflict_group_ids": set()})
        item["conflict_types"].add(str(record.get("conflict_type", "")))
        item["conflict_group_ids"].add(str(record.get("conflict_group_id", "")))
    return pd.DataFrame(
        [
            {
                "goal_a": left,
                "goal_b": right,
                "conflict_types": ",".join(sorted(values["conflict_types"])),
                "conflict_group_ids": ",".join(sorted(values["conflict_group_ids"])),
            }
            for (left, right), values in sorted(rows.items())
        ]
    )


def conflict_separation_table(
    embeddings: pd.DataFrame,
    conflicts: pd.DataFrame,
    *,
    coordinate_columns: Iterable[str],
    eligible_goal_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compare known conflict-pair distances against non-conflict goal-pair distances."""

    columns = list(coordinate_columns)
    if embeddings.empty or not columns:
        return pd.DataFrame()
    eligible = set(map(str, eligible_goal_ids)) if eligible_goal_ids is not None else set(embeddings["canonical_goal_id"].astype(str))
    frame = embeddings[embeddings["canonical_goal_id"].astype(str).isin(eligible)].copy()
    id_to_vec = {
        str(row["canonical_goal_id"]): np.asarray([float(row[column]) for column in columns], dtype=float)
        for row in frame.to_dict(orient="records")
    }
    pair_table = unique_conflict_pairs(conflicts)
    conflict_pairs = {
        tuple(sorted((str(row["goal_a"]), str(row["goal_b"]))))
        for row in pair_table.to_dict(orient="records")
        if str(row["goal_a"]) in id_to_vec and str(row["goal_b"]) in id_to_vec
    }
    rows: list[dict[str, Any]] = []
    for left, right in sorted(conflict_pairs):
        rows.append(
            {
                "pair_type": "known_conflict",
                "goal_a": left,
                "goal_b": right,
                "distance": float(np.linalg.norm(id_to_vec[left] - id_to_vec[right])),
            }
        )
    all_ids = sorted(id_to_vec)
    for left, right in combinations(all_ids, 2):
        key = tuple(sorted((left, right)))
        if key in conflict_pairs:
            continue
        rows.append(
            {
                "pair_type": "non_conflict_reference",
                "goal_a": left,
                "goal_b": right,
                "distance": float(np.linalg.norm(id_to_vec[left] - id_to_vec[right])),
            }
        )
    return pd.DataFrame(rows)


def conflict_separation_summary(pair_distances: pd.DataFrame) -> pd.DataFrame:
    if pair_distances.empty:
        return pd.DataFrame(
            [
                {
                    "metric_name": "known_conflict_distance_lift_over_reference_median",
                    "metric_value": float("nan"),
                    "detail": "no eligible conflict or reference pairs",
                }
            ]
        )
    conflict = pair_distances[pair_distances["pair_type"] == "known_conflict"]["distance"]
    reference = pair_distances[pair_distances["pair_type"] == "non_conflict_reference"]["distance"]
    conflict_median = float(conflict.median()) if not conflict.empty else float("nan")
    reference_median = float(reference.median()) if not reference.empty else float("nan")
    lift = conflict_median - reference_median if np.isfinite(conflict_median) and np.isfinite(reference_median) else float("nan")
    rows = [
        {
            "metric_name": "known_conflict_pair_count",
            "metric_value": float(len(conflict)),
            "detail": "eligible known conflict pairs with both goals represented",
        },
        {
            "metric_name": "reference_non_conflict_pair_count",
            "metric_value": float(len(reference)),
            "detail": "eligible non-conflict reference pairs",
        },
        {
            "metric_name": "known_conflict_distance_median",
            "metric_value": conflict_median,
            "detail": "median Euclidean distance among known conflict goal pairs",
        },
        {
            "metric_name": "reference_non_conflict_distance_median",
            "metric_value": reference_median,
            "detail": "median Euclidean distance among non-conflict reference pairs",
        },
        {
            "metric_name": "known_conflict_distance_lift_over_reference_median",
            "metric_value": lift,
            "detail": "positive values indicate known conflicts are more separated than typical non-conflicts",
        },
    ]
    return pd.DataFrame(rows)


def validate_goal_embedding_artifacts(
    embeddings: pd.DataFrame,
    stability: pd.DataFrame,
    sensitivity: pd.DataFrame,
    conflict_summary: pd.DataFrame,
    *,
    expected_goal_ids: Iterable[str],
    min_modeled_goals: int = 15,
    min_supported_goals: int = 10,
    min_stability_distance_spearman: float = 0.55,
    min_neighbor_overlap: float = 0.20,
) -> pd.DataFrame:
    """Validate S07 goal embedding coverage, stability, sensitivity, and conflict separation."""

    checks: list[dict[str, Any]] = []
    required = {"canonical_goal_id", "embedding_x", "embedding_y", "goal_behavior_row_count", "goal_support_status"}
    missing = sorted(required - set(embeddings.columns))
    checks.append(
        {
            "validation_case": "goal_embedding_required_columns_present",
            "success": not missing,
            "detail": "all required goal embedding columns present" if not missing else f"missing columns: {missing}",
        }
    )
    observed = set(embeddings.get("canonical_goal_id", pd.Series(dtype=str)).astype(str))
    expected = set(map(str, expected_goal_ids))
    checks.append(
        {
            "validation_case": "all_s03_goals_represented",
            "success": expected <= observed,
            "detail": f"represented {len(expected & observed)}/{len(expected)} S03 canonical goals",
        }
    )
    coordinate_cols = [column for column in embeddings.columns if column.startswith("goal_embedding_dim_") or column in {"embedding_x", "embedding_y"}]
    finite = embeddings[coordinate_cols].replace([np.inf, -np.inf], np.nan).notna().all().all() if coordinate_cols else False
    checks.append(
        {
            "validation_case": "goal_embedding_coordinates_finite",
            "success": bool(finite),
            "detail": f"finite coordinate columns: {coordinate_cols[:6]}{'...' if len(coordinate_cols) > 6 else ''}",
        }
    )
    modeled_goals = int((pd.to_numeric(embeddings.get("goal_behavior_row_count", 0), errors="coerce").fillna(0) > 0).sum())
    supported_goals = int((pd.to_numeric(embeddings.get("supported_policy_row_count", 0), errors="coerce").fillna(0) > 0).sum())
    checks.append(
        {
            "validation_case": "sufficient_modeled_goal_coverage",
            "success": modeled_goals >= min_modeled_goals,
            "detail": f"goals with S05 modeling rows: {modeled_goals}",
        }
    )
    checks.append(
        {
            "validation_case": "sufficient_s06_supported_goal_coverage",
            "success": supported_goals >= min_supported_goals,
            "detail": f"goals with at least one behavior-supported policy row: {supported_goals}",
        }
    )
    mean_distance_corr = float(pd.to_numeric(stability.get("pairwise_distance_spearman", pd.Series(dtype=float)), errors="coerce").mean())
    mean_overlap = float(pd.to_numeric(stability.get("neighbor_overlap_at_3", pd.Series(dtype=float)), errors="coerce").mean())
    checks.append(
        {
            "validation_case": "metric_subset_stability_passes_threshold",
            "success": np.isfinite(mean_distance_corr) and mean_distance_corr >= min_stability_distance_spearman and np.isfinite(mean_overlap) and mean_overlap >= min_neighbor_overlap,
            "detail": f"mean distance Spearman={mean_distance_corr:.3f}; mean neighbor overlap@3={mean_overlap:.3f}",
        }
    )
    sensitivity_recorded = not sensitivity.empty and {"analysis_name", "metric_name", "metric_value"} <= set(sensitivity.columns)
    checks.append(
        {
            "validation_case": "s06_policy_support_sensitivity_recorded",
            "success": bool(sensitivity_recorded),
            "detail": f"sensitivity rows: {len(sensitivity)}",
        }
    )
    lookup = conflict_summary.set_index("metric_name")["metric_value"].to_dict() if "metric_name" in conflict_summary else {}
    lift = float(lookup.get("known_conflict_distance_lift_over_reference_median", np.nan))
    checks.append(
        {
            "validation_case": "known_opposite_or_conflicting_goals_separated",
            "success": np.isfinite(lift) and lift > 0.0,
            "detail": f"known-conflict median distance lift={lift:.3f}",
        }
    )
    return pd.DataFrame(checks)
