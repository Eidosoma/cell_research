"""Classic-policy position analysis for E03 S13.

S13 compares the S03/S04 classic algorithms against the measured S10-S12 DSL
morphospace.  Exact public-method interface wrappers are retained as canonical
metric context, while the six S03 DSL classic landmarks are the measurable
points in the S07-S12 morphospace.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CLASSIC_POSITION_SCHEMA = "eidosoma.e03.classic_position.v1"
CLASSIC_ALGORITHMS = ("bubble", "insertion", "selection")
CLASSIC_DISPLAY = {"bubble": "Bubble", "insertion": "Insertion", "selection": "Selection"}
HIGH_COMPETENCE_CLASSES = {"UC01", "UC02", "UC03"}


@dataclass(frozen=True)
class FeatureMatrix:
    """A finite matrix and the feature names used to build it."""

    matrix: np.ndarray
    feature_names: tuple[str, ...]
    imputed_feature_names: tuple[str, ...]


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def load_s03_policy_library(path: Path) -> pd.DataFrame:
    """Load the S03 classic policy library JSON as a normalized frame."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    policies = payload.get("policies", [])
    if not policies:
        raise ValueError(f"S03 policy library contains no policies: {path}")
    rows: list[dict[str, Any]] = []
    for record in policies:
        rows.append(
            {
                "policy_id": str(record["policyId"]),
                "algorithm": str(record["algorithm"]),
                "algorithm_label": CLASSIC_DISPLAY.get(str(record["algorithm"]), str(record["algorithm"]).title()),
                "representation_type": str(record["representationType"]),
                "exactness": str(record["exactness"]),
                "direction": str(record["direction"]),
                "implementation_ref": str(record.get("implementationRef", "")),
                "dsl_sha256": record.get("dslSha256"),
                "has_dsl_source": bool(record.get("dslSource")),
                "known_deviations_json": stable_json(record.get("knownDeviations", [])),
                "s03_notes": str(record.get("notes", "")),
            }
        )
    out = pd.DataFrame(rows)
    expected = set(CLASSIC_ALGORITHMS)
    observed = set(out["algorithm"])
    if not expected.issubset(observed):
        raise ValueError(f"S03 policy library missing classic algorithms: {sorted(expected - observed)}")
    return out


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _claim_counts(group: pd.DataFrame) -> dict[str, int]:
    if group.empty or "claim_type" not in group.columns:
        return {}
    counts = group["claim_type"].dropna().astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def classic_alignment_frame(
    s03: pd.DataFrame,
    s04_vectors: pd.DataFrame,
    s07_competence: pd.DataFrame,
    s10_embeddings: pd.DataFrame,
    s11_clusters: pd.DataFrame,
    s12_claims: pd.DataFrame,
) -> pd.DataFrame:
    """Align classic IDs across S03/S04/S07/S10/S11/S12."""

    frame = s03.copy()
    id_sets = {
        "s04_present": set(s04_vectors["policy_id"].astype(str)),
        "s07_present": set(s07_competence["policy_id"].astype(str)),
        "s10_present": set(s10_embeddings["policy_id"].astype(str)),
        "s11_present": set(s11_clusters["policy_id"].astype(str)),
    }
    for column, ids in id_sets.items():
        frame[column] = frame["policy_id"].isin(ids)
    if not s12_claims.empty:
        s12_groups = s12_claims.groupby("original_policy_id")
        frame["s12_present"] = frame["policy_id"].isin(set(s12_claims["original_policy_id"].astype(str)))
        frame["s12_ablation_count"] = frame["policy_id"].map(lambda policy_id: int(len(s12_groups.get_group(policy_id))) if policy_id in s12_groups.groups else 0)
        frame["s12_claim_types_json"] = frame["policy_id"].map(
            lambda policy_id: stable_json(_claim_counts(s12_groups.get_group(policy_id))) if policy_id in s12_groups.groups else "{}"
        )
    else:
        frame["s12_present"] = False
        frame["s12_ablation_count"] = 0
        frame["s12_claim_types_json"] = "{}"

    s04_cols = [
        "policy_id",
        "canonical_classic_baseline",
        "metric_projection_scope",
        "metric_trust_level",
        "final_sortedness_score",
        "frozen_cell_robustness_score",
        "dg_tendency_score",
        "aggregation_tendency_score",
        "path_directness_score",
        "competence_vector_hash",
    ]
    frame = frame.merge(s04_vectors[[column for column in s04_cols if column in s04_vectors.columns]], on="policy_id", how="left")
    s10_cols = [
        "policy_id",
        "policy_name",
        "source_kind",
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_improvement_mean",
        "screen_heldout_work_mean",
        "n100_final_sortedness_mean",
        "n1000_final_sortedness_mean",
        "embedding_x",
        "embedding_y",
        "embedding_z",
        "classic_landmark",
        "classic_family",
        "s08_archive_winner",
        "qd_candidate",
    ]
    frame = frame.merge(s10_embeddings[[column for column in s10_cols if column in s10_embeddings.columns]], on="policy_id", how="left", suffixes=("", "_s10"))
    s11_cols = [
        "policy_id",
        "class_id",
        "cautious_label",
        "label_confidence",
        "cluster_distance",
        "classic_landmark",
        "classic_family",
    ]
    frame = frame.merge(s11_clusters[[column for column in s11_cols if column in s11_clusters.columns]], on="policy_id", how="left", suffixes=("", "_s11"))
    frame["morphospace_role"] = np.where(
        frame["representation_type"].eq("dsl") & frame["s10_present"] & frame["s11_present"],
        "measured_dsl_landmark",
        np.where(frame["representation_type"].eq("interface_wrapper"), "canonical_interface_context_not_embedded", "unembedded_classic_entry"),
    )
    return frame


def behavior_feature_columns(normalized_features: pd.DataFrame) -> tuple[str, ...]:
    """Return S10 behavior feature columns used for normalized/raw distances."""

    return tuple(column for column in normalized_features.columns if column not in {"policy_id", "policy_name"})


def finite_feature_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> FeatureMatrix:
    """Convert selected numeric columns to a finite matrix with median imputation."""

    columns = tuple(feature_names)
    if not columns:
        raise ValueError("At least one feature column is required")
    data = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    imputed: list[str] = []
    for column in columns:
        series = data[column]
        if series.isna().any():
            imputed.append(column)
        median = series.median(skipna=True)
        if pd.isna(median):
            median = 0.0
        data[column] = series.fillna(float(median))
    matrix = data.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values after imputation")
    return FeatureMatrix(matrix=matrix, feature_names=columns, imputed_feature_names=tuple(imputed))


def _distance_vector(matrix: np.ndarray, target_index: int) -> np.ndarray:
    diff = matrix - matrix[target_index]
    return np.sqrt(np.sum(diff * diff, axis=1))


def nearest_distance_percentiles(
    policy_table: pd.DataFrame,
    matrix: np.ndarray,
    *,
    exclude_classics: bool = True,
) -> pd.Series:
    """Return each row's nearest non-classic distance percentile."""

    classic_mask = policy_table["classic_landmark"].map(_truthy).to_numpy() if "classic_landmark" in policy_table.columns else np.zeros(len(policy_table), dtype=bool)
    mins: list[float] = []
    for idx in range(len(policy_table)):
        candidate_mask = np.ones(len(policy_table), dtype=bool)
        candidate_mask[idx] = False
        if exclude_classics:
            candidate_mask &= ~classic_mask
        distances = _distance_vector(matrix, idx)
        mins.append(float(np.min(distances[candidate_mask])))
    values = pd.Series(mins, index=policy_table["policy_id"].astype(str))
    ranks = values.rank(method="average", pct=True)
    return ranks


def nearest_neighbors(
    policy_table: pd.DataFrame,
    matrix: np.ndarray,
    classic_policy_ids: Sequence[str],
    *,
    metric_space: str,
    k: int = 20,
    exclude_classics: bool = True,
) -> pd.DataFrame:
    """Compute nearest non-classic neighbors for each classic DSL landmark."""

    ids = policy_table["policy_id"].astype(str).tolist()
    id_to_idx = {policy_id: idx for idx, policy_id in enumerate(ids)}
    classic_set = set(str(policy_id) for policy_id in classic_policy_ids)
    classic_mask = policy_table["classic_landmark"].map(_truthy).to_numpy() if "classic_landmark" in policy_table.columns else np.array([policy_id in classic_set for policy_id in ids])
    percentile = nearest_distance_percentiles(policy_table, matrix, exclude_classics=exclude_classics)
    rows: list[dict[str, Any]] = []
    for target_id in classic_policy_ids:
        target_id = str(target_id)
        if target_id not in id_to_idx:
            continue
        target_idx = id_to_idx[target_id]
        target_row = policy_table.iloc[target_idx]
        candidate_mask = np.ones(len(policy_table), dtype=bool)
        candidate_mask[target_idx] = False
        if exclude_classics:
            candidate_mask &= ~classic_mask
        distances = _distance_vector(matrix, target_idx)
        candidates = [
            (float(distances[idx]), str(policy_table.iloc[idx]["policy_id"]), idx)
            for idx in np.where(candidate_mask)[0]
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        for rank, (distance, neighbor_id, neighbor_idx) in enumerate(candidates[:k], start=1):
            neighbor = policy_table.iloc[neighbor_idx]
            rows.append(
                {
                    "schema": CLASSIC_POSITION_SCHEMA,
                    "experiment_id": "E03",
                    "research_step_id": "S13",
                    "metric_space": metric_space,
                    "target_policy_id": target_id,
                    "target_policy_name": str(target_row.get("policy_name", "")),
                    "target_algorithm": str(target_row.get("classic_family", "")),
                    "target_class_id": str(target_row.get("class_id", "")),
                    "nearest_distance_percentile": float(percentile.loc[target_id]),
                    "neighbor_rank": int(rank),
                    "neighbor_policy_id": neighbor_id,
                    "neighbor_policy_name": str(neighbor.get("policy_name", "")),
                    "neighbor_distance": distance,
                    "neighbor_class_id": str(neighbor.get("class_id", "")),
                    "neighbor_cautious_label": str(neighbor.get("cautious_label", "")),
                    "same_class": str(target_row.get("class_id", "")) == str(neighbor.get("class_id", "")),
                    "neighbor_screen_score": float(pd.to_numeric(pd.Series([neighbor.get("screen_score", np.nan)]), errors="coerce").iloc[0]),
                    "neighbor_source_kind": str(neighbor.get("source_kind", "")),
                    "neighbor_qd_candidate": bool(_truthy(neighbor.get("qd_candidate", False))),
                    "neighbor_archive_winner": bool(_truthy(neighbor.get("s08_archive_winner", False))),
                    "neighbor_classic_landmark": bool(_truthy(neighbor.get("classic_landmark", False))),
                }
            )
    return pd.DataFrame(rows)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def neighbor_robustness_frame(neighbors: pd.DataFrame, *, k: int = 10) -> pd.DataFrame:
    """Summarize top-k nearest-neighbor stability across metric spaces."""

    rows: list[dict[str, Any]] = []
    metric_spaces = sorted(neighbors["metric_space"].dropna().unique().tolist())
    for target_id, group in neighbors.groupby("target_policy_id", sort=True):
        row: dict[str, Any] = {
            "schema": CLASSIC_POSITION_SCHEMA,
            "experiment_id": "E03",
            "research_step_id": "S13",
            "target_policy_id": str(target_id),
            "target_policy_name": str(group["target_policy_name"].iloc[0]),
            "target_algorithm": str(group["target_algorithm"].iloc[0]),
            "target_class_id": str(group["target_class_id"].iloc[0]),
        }
        sets: dict[str, set[str]] = {}
        for metric_space in metric_spaces:
            subset = group[(group["metric_space"] == metric_space) & (group["neighbor_rank"] <= k)]
            sets[metric_space] = set(subset["neighbor_policy_id"].astype(str))
            row[f"{metric_space}_top{k}_same_class_fraction"] = float(subset["same_class"].mean()) if not subset.empty else np.nan
            row[f"{metric_space}_top{k}_qd_fraction"] = float(subset["neighbor_qd_candidate"].mean()) if not subset.empty else np.nan
            row[f"{metric_space}_top{k}_archive_fraction"] = float(subset["neighbor_archive_winner"].mean()) if not subset.empty else np.nan
            row[f"{metric_space}_nearest_distance"] = float(subset["neighbor_distance"].iloc[0]) if not subset.empty else np.nan
            row[f"{metric_space}_nearest_distance_percentile"] = float(subset["nearest_distance_percentile"].iloc[0]) if not subset.empty else np.nan
        row[f"normalized_raw_top{k}_jaccard"] = _jaccard(sets.get("normalized", set()), sets.get("raw", set()))
        row[f"normalized_embedding_top{k}_jaccard"] = _jaccard(sets.get("normalized", set()), sets.get("embedding", set()))
        row[f"raw_embedding_top{k}_jaccard"] = _jaccard(sets.get("raw", set()), sets.get("embedding", set()))
        nr = float(row.get(f"normalized_raw_top{k}_jaccard", 0.0))
        same_norm = float(row.get(f"normalized_top{k}_same_class_fraction", 0.0))
        same_raw = float(row.get(f"raw_top{k}_same_class_fraction", 0.0))
        if nr >= 0.5 and min(same_norm, same_raw) >= 0.5:
            stability = "robust"
        elif nr >= 0.2 or max(same_norm, same_raw) >= 0.5:
            stability = "moderate"
        else:
            stability = "unstable"
        row["nearest_neighbor_stability"] = stability
        rows.append(row)
    return pd.DataFrame(rows)


def classic_policy_position_frame(
    alignment: pd.DataFrame,
    robustness: pd.DataFrame,
    s12_claims: pd.DataFrame,
) -> pd.DataFrame:
    """Attach nearest-neighbor and S12 summaries to each aligned classic policy."""

    out = alignment.copy()
    out = out.merge(robustness, left_on="policy_id", right_on="target_policy_id", how="left", suffixes=("", "_robustness"))
    if not s12_claims.empty:
        s12_rows: list[dict[str, Any]] = []
        for policy_id, group in s12_claims.groupby("original_policy_id", sort=True):
            counts = _claim_counts(group)
            s12_rows.append(
                {
                    "policy_id": str(policy_id),
                    "s12_local_necessary_count": int(counts.get("local_necessary_candidate", 0)),
                    "s12_residual_sufficient_count": int(counts.get("residual_sufficient_candidate", 0)),
                    "s12_anti_feature_count": int(counts.get("anti_feature_candidate", 0)),
                    "s12_neutral_count": int(counts.get("locally_neutral_candidate", 0)),
                    "s12_weak_signal_count": int(
                        counts.get("weak_local_improvement_candidate", 0) + counts.get("weak_local_necessity_candidate", 0)
                    ),
                    "s12_mean_delta_final_sortedness": float(group["heldout_delta_final_sortedness"].mean()),
                    "s12_most_negative_delta": float(group["heldout_delta_final_sortedness"].min()),
                    "s12_most_positive_delta": float(group["heldout_delta_final_sortedness"].max()),
                }
            )
        out = out.merge(pd.DataFrame(s12_rows), on="policy_id", how="left")
    for column in (
        "s12_local_necessary_count",
        "s12_residual_sufficient_count",
        "s12_anti_feature_count",
        "s12_neutral_count",
        "s12_weak_signal_count",
    ):
        if column in out.columns:
            out[column] = out[column].fillna(0).astype(int)
    return out


def classify_algorithm(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return central/peripheral/accidental classification plus confidence/reason."""

    best_score = float(row.get("best_dsl_screen_score", np.nan))
    best_class = str(row.get("best_dsl_class_id", ""))
    best_policy = str(row.get("best_dsl_policy_name", ""))
    direction_split = bool(row.get("direction_split", False))
    stability = str(row.get("best_dsl_nearest_neighbor_stability", ""))
    normalized_raw_jaccard = float(row.get("best_dsl_normalized_raw_top10_jaccard", np.nan))
    local_necessary = int(row.get("s12_local_necessary_total", 0))
    anti_feature = int(row.get("s12_anti_feature_total", 0))

    if best_score >= 0.85 and best_class in HIGH_COMPETENCE_CLASSES:
        classification = "central"
        confidence = "medium"
        reason = f"Best DSL landmark `{best_policy}` lies in high-competence {best_class} with screen score {best_score:.3f}; S12 found {local_necessary} local necessary-feature signals."
        if direction_split:
            reason += " Directional shadows split strongly, so centrality applies to the competent direction rather than every parameterization."
    elif best_score >= 0.55:
        classification = "peripheral"
        confidence = "medium" if stability in {"robust", "moderate"} else "low-medium"
        reason = f"Best DSL landmark `{best_policy}` has intermediate screen score {best_score:.3f} in {best_class}; it has measurable neighbors but is outside the high-competence UC01-UC03 core."
        if direction_split:
            reason += " The opposite-direction DSL landmark falls into a weaker class."
    else:
        classification = "accidental"
        confidence = "low-medium"
        reason = f"Best DSL landmark `{best_policy}` remains low-scoring ({best_score:.3f}) outside high-competence classes; S12 found {anti_feature} anti-feature signals, so this classic appears incidental under the current DSL proxy screen."
    if np.isfinite(normalized_raw_jaccard):
        reason += f" Normalized/raw top-10 neighbor Jaccard for the best landmark is {normalized_raw_jaccard:.3f}."
    return classification, confidence, reason


def algorithm_position_frame(classic_positions: pd.DataFrame) -> pd.DataFrame:
    """Collapse DSL landmark placement into one cautious classification per algorithm."""

    rows: list[dict[str, Any]] = []
    for algorithm, group in classic_positions.groupby("algorithm", sort=True):
        dsl = group[group["representation_type"] == "dsl"].copy()
        iface = group[group["representation_type"] == "interface_wrapper"].copy()
        if dsl.empty:
            continue
        dsl["screen_score_numeric"] = pd.to_numeric(dsl["screen_score"], errors="coerce")
        dsl = dsl.sort_values(["screen_score_numeric", "policy_id"], ascending=[False, True], kind="mergesort")
        best = dsl.iloc[0]
        worst = dsl.iloc[-1]
        direction_split = bool(
            str(best.get("class_id", "")) != str(worst.get("class_id", ""))
            and float(best.get("screen_score_numeric", np.nan)) - float(worst.get("screen_score_numeric", np.nan)) >= 0.25
        )
        row: dict[str, Any] = {
            "schema": CLASSIC_POSITION_SCHEMA,
            "experiment_id": "E03",
            "research_step_id": "S13",
            "algorithm": str(algorithm),
            "algorithm_label": CLASSIC_DISPLAY.get(str(algorithm), str(algorithm).title()),
            "interface_policy_ids_json": stable_json(iface["policy_id"].astype(str).tolist()),
            "dsl_policy_ids_json": stable_json(dsl["policy_id"].astype(str).tolist()),
            "dsl_landmark_count": int(len(dsl)),
            "embedded_dsl_landmark_count": int(dsl["s10_present"].sum()),
            "best_dsl_policy_id": str(best["policy_id"]),
            "best_dsl_policy_name": str(best.get("policy_name", "")),
            "best_dsl_direction": str(best.get("direction", "")),
            "best_dsl_exactness": str(best.get("exactness", "")),
            "best_dsl_class_id": str(best.get("class_id", "")),
            "best_dsl_cautious_label": str(best.get("cautious_label", "")),
            "best_dsl_screen_score": float(best.get("screen_score_numeric", np.nan)),
            "best_dsl_heldout_sortedness": float(best.get("screen_heldout_final_sortedness_mean", np.nan)),
            "worst_dsl_policy_id": str(worst["policy_id"]),
            "worst_dsl_direction": str(worst.get("direction", "")),
            "worst_dsl_class_id": str(worst.get("class_id", "")),
            "worst_dsl_screen_score": float(worst.get("screen_score_numeric", np.nan)),
            "direction_split": direction_split,
            "s12_local_necessary_total": int(pd.to_numeric(dsl.get("s12_local_necessary_count", 0), errors="coerce").fillna(0).sum()),
            "s12_residual_sufficient_total": int(pd.to_numeric(dsl.get("s12_residual_sufficient_count", 0), errors="coerce").fillna(0).sum()),
            "s12_anti_feature_total": int(pd.to_numeric(dsl.get("s12_anti_feature_count", 0), errors="coerce").fillna(0).sum()),
            "best_dsl_nearest_neighbor_stability": str(best.get("nearest_neighbor_stability", "")),
            "best_dsl_normalized_raw_top10_jaccard": float(best.get("normalized_raw_top10_jaccard", np.nan)),
            "best_dsl_normalized_nearest_distance_percentile": float(best.get("normalized_nearest_distance_percentile", np.nan)),
            "best_dsl_raw_nearest_distance_percentile": float(best.get("raw_nearest_distance_percentile", np.nan)),
            "canonical_interface_final_sortedness_score": float(pd.to_numeric(iface.get("final_sortedness_score", pd.Series(dtype=float)), errors="coerce").mean())
            if not iface.empty
            else np.nan,
            "canonical_interface_dg_tendency_score": float(pd.to_numeric(iface.get("dg_tendency_score", pd.Series(dtype=float)), errors="coerce").mean())
            if not iface.empty
            else np.nan,
            "canonical_interface_aggregation_tendency_score": float(pd.to_numeric(iface.get("aggregation_tendency_score", pd.Series(dtype=float)), errors="coerce").mean())
            if not iface.empty
            else np.nan,
        }
        classification, confidence, reason = classify_algorithm(row)
        row["classic_position_classification"] = classification
        row["classification_confidence"] = confidence
        row["classification_basis"] = reason
        rows.append(row)
    order = {algorithm: idx for idx, algorithm in enumerate(CLASSIC_ALGORITHMS)}
    out = pd.DataFrame(rows)
    return out.sort_values("algorithm", key=lambda series: series.map(order), kind="mergesort").reset_index(drop=True)


def validation_frame(
    *,
    alignment: pd.DataFrame,
    classic_positions: pd.DataFrame,
    algorithm_positions: pd.DataFrame,
    neighbors: pd.DataFrame,
    robustness: pd.DataFrame,
    figure_exists: bool,
    unit_success: bool,
) -> pd.DataFrame:
    """Build S13 validation rows."""

    cases: list[dict[str, Any]] = []

    def add(name: str, success: bool, expected: str, observed: Any, notes: str) -> None:
        cases.append(
            {
                "validation_case": name,
                "success": bool(success),
                "expected": expected,
                "observed": str(observed),
                "notes": notes,
            }
        )

    add("s03_policy_count", len(alignment) == 9, "9 S03 classic entries", len(alignment), "S03 has 3 exact interface wrappers and 6 DSL landmarks.")
    add("s04_all_classics_aligned", bool(alignment["s04_present"].all()), "all S03 rows present in S04", int(alignment["s04_present"].sum()), "S04 competence context is aligned to S03 IDs.")
    embedded_dsl = alignment[(alignment["representation_type"] == "dsl") & alignment["s10_present"] & alignment["s11_present"]]
    add("dsl_landmarks_embedded", len(embedded_dsl) == 6, "6 DSL landmarks in S10/S11", len(embedded_dsl), "Only DSL classic landmarks are embedded in the E03 morphospace.")
    iface = alignment[alignment["representation_type"] == "interface_wrapper"]
    add("interfaces_not_embedded_documented", bool((~iface["s10_present"]).all()), "interface wrappers absent from S10/S11", int((~iface["s10_present"]).sum()), "Exact public wrappers are canonical context, not DSL morphospace points.")
    add("normalized_and_raw_neighbors", set(neighbors["metric_space"]) >= {"normalized", "raw"}, "normalized and raw metric spaces present", sorted(neighbors["metric_space"].unique()), "Nearest neighbors were computed in both requested spaces.")
    add("embedding_neighbors_present", "embedding" in set(neighbors["metric_space"]), "embedding metric space present", sorted(neighbors["metric_space"].unique()), "Embedding neighbors provide a visualization-space robustness check.")
    neighbor_counts = neighbors.groupby(["target_policy_id", "metric_space"]).size()
    expected_neighbor_groups = 6 * 3
    add(
        "neighbor_coverage",
        len(neighbor_counts) == expected_neighbor_groups and int(neighbor_counts.min()) > 0 and int(neighbor_counts.max()) <= 20,
        "6 DSL landmarks x 3 metric spaces, each with 1-20 neighbors",
        neighbor_counts.describe().to_dict() if not neighbor_counts.empty else {},
        "Top-20 neighbors per DSL landmark per metric space, capped by available non-classic policies.",
    )
    add("neighbor_robustness_rows", len(robustness) == 6, "6 robustness rows", len(robustness), "One nearest-neighbor stability row per DSL landmark.")
    add("algorithm_classifications_present", set(algorithm_positions["algorithm"]) == set(CLASSIC_ALGORITHMS), "Bubble/Insertion/Selection", sorted(algorithm_positions["algorithm"].tolist()), "One cautious position classification per classic algorithm.")
    add("classification_vocabulary", set(algorithm_positions["classic_position_classification"]).issubset({"central", "peripheral", "accidental"}), "central/peripheral/accidental only", sorted(algorithm_positions["classic_position_classification"].unique()), "S13 uses the requested classification vocabulary.")
    add("figure_written", figure_exists, "figure exists", figure_exists, "Classics-in-morphospace figure was written.")
    add("unit_tests_passed", unit_success, "unit tests pass", unit_success, "Full E03 unit suite status from the S13 script.")
    return pd.DataFrame(cases)


def classic_position_digest(frame: pd.DataFrame) -> str:
    cols = [
        "algorithm",
        "best_dsl_policy_id",
        "best_dsl_class_id",
        "best_dsl_screen_score",
        "classic_position_classification",
        "classification_confidence",
    ]
    payload = frame[[column for column in cols if column in frame.columns]].sort_values("algorithm", kind="mergesort")
    return hashlib.sha256(payload.to_json(orient="records", double_precision=12).encode("utf-8")).hexdigest()
