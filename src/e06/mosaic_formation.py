"""E06 S07 final-state and mosaic formation analysis."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.e06.mixture_ratios import EXPERIMENT_ID, stable_hash


STEP_ID = "S07"
MOSAIC_SCHEMA = "eidosoma.e06.s07_mosaic_run_class.v1"
CLASS_SUMMARY_SCHEMA = "eidosoma.e06.s07_mosaic_class_summary.v1"
CONTINUUM_SCHEMA = "eidosoma.e06.s07_metric_continuum.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s07_validation.v1"

CLASSIFIER_FEATURE_COLUMNS = (
    "final_target_quality",
    "reference_increasing_quality",
    "goal_conflict_index",
    "aggregation_delta_percent",
    "interface_density",
    "largest_block_fraction",
    "label_entropy",
    "position_bias_abs",
    "work_rate",
    "swap_rate",
    "frozen_block_rate",
    "sortedness_delta",
)

SPATIAL_FEATURE_COLUMNS = (
    "aggregation_delta_percent",
    "interface_density",
    "largest_block_fraction",
    "label_entropy",
    "position_bias_abs",
)


@dataclass(frozen=True)
class S07Config:
    """Configuration for S07 final-state classification and stability checks."""

    seed_stability_threshold: float = 0.75
    classifier_stability_threshold: float = 0.70
    classifier_min_pairwise_agreement: float = 0.55
    classifier_k_values: tuple[int, ...] = (4, 5, 6)
    classifier_random_states: tuple[int, ...] = (17, 23, 31)
    exemplar_count_per_label: int = 1


def _json_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), separators=(",", ":"), default=str)


def _json_object(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)


def _parse_json_list(text: Any) -> tuple[Any, ...]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return tuple()
    try:
        return tuple(json.loads(str(text)))
    except json.JSONDecodeError:
        return tuple()


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _stable_run_id(source_step: str, row: Mapping[str, Any]) -> str:
    return f"s07_{stable_hash({'sourceStep': source_step, 'condition': row.get('condition_id', ''), 'seed': row.get('seed', ''), 'orientation': row.get('orientation', ''), 'value': row.get('value_profile', ''), 'perturbation': row.get('perturbation_profile', '')})[:18]}"


def _ratio_label(ratios: Sequence[float]) -> str:
    return ":".join(str(int(round(float(value) * 100))) for value in ratios)


def _normalized_key_list(text: Any, *, floats: bool = False) -> str:
    values = _parse_json_list(text)
    if floats:
        values = tuple(round(float(value), 6) for value in values)
    return _json_list(values)


def _source_goal_defaults(source_step: str) -> tuple[str, str]:
    if source_step in {"S02", "S03"}:
        return "same_increasing", "same_goal"
    return "", ""


def harmonize_source_runs(source_step: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Convert one S02/S03/S04/S06 run table into the S07 run schema."""

    rows: list[dict[str, Any]] = []
    default_goal_profile, default_goal_class = _source_goal_defaults(source_step)
    for raw in frame.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in _parse_json_list(raw.get("policy_ids_json")))
        ratios = tuple(float(item) for item in _parse_json_list(raw.get("ratio_targets_json")))
        display_names = tuple(str(item) for item in _parse_json_list(raw.get("display_names_json")))
        source_categories = tuple(str(item) for item in _parse_json_list(raw.get("source_categories_json")))
        array_size = _safe_int(raw.get("array_size"), 100)
        event_cap = _safe_int(raw.get("event_cap"), 0)
        final_sortedness = _safe_float(raw.get("final_inversion_sortedness"))
        assigned_quality = _safe_float(raw.get("assigned_policy_mean_sortedness"))
        target_quality = assigned_quality if math.isfinite(assigned_quality) else final_sortedness
        reference_quality = _safe_float(raw.get("reference_increasing_score"), final_sortedness)
        goal_gap = _safe_float(raw.get("goal_alignment_gap"), 0.0)
        work_count = _safe_float(raw.get("work_count"), 0.0)
        swap_count = _safe_float(raw.get("swap_count"), 0.0)
        frozen_block_count = _safe_float(raw.get("frozen_block_count"), 0.0)
        condition_id = str(raw.get("condition_id", ""))
        goal_profile_id = str(raw.get("goal_profile_id", default_goal_profile) or default_goal_profile)
        goal_class = str(raw.get("goal_compatibility_class", default_goal_class) or default_goal_class)
        final_state_class = str(raw.get("final_state_class", ""))
        goal_state_class = str(raw.get("goal_state_class", ""))
        ratio_max = max(ratios) if ratios else math.nan
        interface_count = _safe_float(raw.get("interface_count"))
        denom = max(1.0, float(array_size - 1))
        row = {
            "schema": MOSAIC_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "research_step_id": STEP_ID,
            "source_research_step_id": source_step,
            "source_run_schema": str(raw.get("schema", "")),
            "run_id": _stable_run_id(source_step, raw),
            "source_condition_id": condition_id,
            "analysis_condition_id": condition_id,
            "seed": _safe_int(raw.get("seed")),
            "panel": str(raw.get("panel", "")),
            "condition_kind": str(raw.get("condition_kind", "pair")),
            "candidate_reason": str(raw.get("candidate_reason", "")),
            "arrangement": str(raw.get("arrangement", "")),
            "orientation": str(raw.get("orientation", "")),
            "value_profile": str(raw.get("value_profile", "")),
            "perturbation_profile": str(raw.get("perturbation_profile", "")),
            "goal_profile_id": goal_profile_id,
            "goal_compatibility_class": goal_class,
            "policy_ids_json": _json_list(policy_ids),
            "display_names_json": _json_list(display_names),
            "source_categories_json": _json_list(source_categories),
            "ratio_targets_json": _json_list(ratios),
            "ratio_label": _ratio_label(ratios) if ratios else "",
            "ratio_max_fraction": float(ratio_max) if math.isfinite(ratio_max) else math.nan,
            "array_size": int(array_size),
            "event_cap": int(event_cap),
            "events_executed": _safe_int(raw.get("events_executed"), event_cap),
            "final_target_quality": float(target_quality),
            "reference_increasing_quality": float(reference_quality),
            "assigned_goal_quality": float(assigned_quality) if math.isfinite(assigned_quality) else math.nan,
            "goal_conflict_index": float(goal_gap),
            "final_inversion_sortedness": float(final_sortedness),
            "initial_inversion_sortedness": _safe_float(raw.get("initial_inversion_sortedness")),
            "sortedness_delta": _safe_float(raw.get("inversion_sortedness_delta"), 0.0),
            "aggregation_delta_percent": _safe_float(raw.get("aggregation_delta_percent"), 0.0),
            "interface_count": float(interface_count),
            "interface_density": float(interface_count / denom) if math.isfinite(interface_count) else math.nan,
            "contiguous_run_count": _safe_float(raw.get("contiguous_run_count")),
            "largest_block_fraction": _safe_float(raw.get("largest_block_fraction"), 0.0),
            "label_entropy": _safe_float(raw.get("label_entropy"), 0.0),
            "position_bias_margin": _safe_float(raw.get("position_bias_margin"), 0.0),
            "position_bias_abs": abs(_safe_float(raw.get("position_bias_margin"), 0.0)),
            "work_count": float(work_count),
            "swap_count": float(swap_count),
            "frozen_block_count": float(frozen_block_count),
            "work_rate": float(work_count / max(1, event_cap)),
            "swap_rate": float(swap_count / max(1, event_cap)),
            "frozen_block_rate": float(frozen_block_count / max(1, event_cap)),
            "final_state_class": final_state_class,
            "goal_state_class": goal_state_class,
            "leftmost_policy_id": str(raw.get("leftmost_policy_id", "")),
            "rightmost_policy_id": str(raw.get("rightmost_policy_id", "")),
            "mean_position_by_policy_json": str(raw.get("mean_position_by_policy_json", "{}")),
            "final_labels_head_json": str(raw.get("final_labels_head_json", "[]")),
            "final_labels_tail_json": str(raw.get("final_labels_tail_json", "[]")),
            "final_values_head_json": str(raw.get("final_values_head_json", "[]")),
            "final_values_tail_json": str(raw.get("final_values_tail_json", "[]")),
            "base_contest_id": str(raw.get("base_contest_id", "")),
            "dominant_policy_id": str(raw.get("dominant_policy_id", "")),
            "dominated_policy_id": str(raw.get("dominated_policy_id", "")),
            "dominance_margin_abs": _safe_float(raw.get("dominance_margin_abs"), math.nan),
            "condition_context_json": _json_object(
                {
                    "sourceStep": source_step,
                    "arrangement": str(raw.get("arrangement", "")),
                    "goalProfileId": goal_profile_id,
                    "orientation": str(raw.get("orientation", "")),
                    "valueProfile": str(raw.get("value_profile", "")),
                    "perturbationProfile": str(raw.get("perturbation_profile", "")),
                }
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def harmonize_all_runs(frames_by_step: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [harmonize_source_runs(step, frame) for step, frame in frames_by_step.items()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _context_key(
    source_step: str,
    policy_ids_json: Any,
    ratio_targets_json: Any,
    arrangement: Any,
    goal_profile_id: Any,
) -> str:
    return _json_object(
        {
            "sourceStep": source_step,
            "policyIds": _parse_json_list(policy_ids_json),
            "ratios": tuple(round(float(value), 6) for value in _parse_json_list(ratio_targets_json)),
            "arrangement": str(arrangement),
            "goalProfileId": str(goal_profile_id),
        }
    )


def attach_explanatory_contexts(
    runs: pd.DataFrame,
    compatibility_scores: pd.DataFrame,
    dominance_hierarchy_df: pd.DataFrame,
    dominance_context_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach S05/S06 context columns without using them as classifier features."""

    enriched = runs.copy()
    enriched["s05_join_key"] = enriched.apply(
        lambda row: _context_key(
            str(row["source_research_step_id"]),
            row["policy_ids_json"],
            row["ratio_targets_json"],
            row["arrangement"],
            row["goal_profile_id"],
        ),
        axis=1,
    )
    if not compatibility_scores.empty:
        score_context = compatibility_scores.copy()
        score_context["s05_join_key"] = score_context.apply(
            lambda row: _context_key(
                str(row["source_research_step_id"]),
                row["policy_ids_json"],
                row["ratio_targets_json"],
                row["arrangement"],
                row["goal_profile_id"],
            ),
            axis=1,
        )
        score_columns = [
            "s05_join_key",
            "score_id",
            "primary_target_quality",
            "expected_pure_quality",
            "quality_synergy",
            "work_interference_log_ratio",
            "cooperative_efficiency_score",
            "integration_score",
            "interface_stability_score",
            "dominance_abs_margin",
            "goal_conflict_index",
            "metric_disagreement_count",
            "conflicting_dimensions_json",
            "compatibility_pattern_class",
        ]
        available = [column for column in score_columns if column in score_context.columns]
        score_context = score_context[available].drop_duplicates("s05_join_key", keep="first")
        rename = {column: f"s05_{column}" for column in available if column != "s05_join_key"}
        enriched = enriched.merge(score_context.rename(columns=rename), on="s05_join_key", how="left")

    if not dominance_context_df.empty and "base_contest_id" in dominance_context_df.columns:
        dom_context = dominance_context_df.copy()
        dom_context = dom_context.rename(
            columns={
                column: f"s06_context_{column}"
                for column in dom_context.columns
                if column != "base_contest_id"
            }
        )
        enriched = enriched.merge(dom_context, on="base_contest_id", how="left")

    hierarchy_map = {}
    if not dominance_hierarchy_df.empty:
        for row in dominance_hierarchy_df.to_dict(orient="records"):
            hierarchy_map[str(row["policy_id"])] = {
                "net": _safe_float(row.get("net_dominance_score")),
                "win_fraction": _safe_float(row.get("win_fraction")),
                "display": str(row.get("display_name", row.get("policy_id", ""))),
            }
    net_mins: list[float] = []
    net_maxes: list[float] = []
    net_means: list[float] = []
    net_spreads: list[float] = []
    top_ids: list[str] = []
    top_names: list[str] = []
    for text in enriched["policy_ids_json"].tolist():
        policy_ids = tuple(str(item) for item in _parse_json_list(text))
        nets = [(policy_id, hierarchy_map[policy_id]["net"]) for policy_id in policy_ids if policy_id in hierarchy_map and math.isfinite(hierarchy_map[policy_id]["net"])]
        if not nets:
            net_mins.append(math.nan)
            net_maxes.append(math.nan)
            net_means.append(math.nan)
            net_spreads.append(math.nan)
            top_ids.append("")
            top_names.append("")
            continue
        values = [value for _, value in nets]
        top_id, _top_value = max(nets, key=lambda item: item[1])
        net_mins.append(float(min(values)))
        net_maxes.append(float(max(values)))
        net_means.append(float(np.mean(values)))
        net_spreads.append(float(max(values) - min(values)))
        top_ids.append(top_id)
        top_names.append(hierarchy_map[top_id]["display"])
    enriched["s06_hierarchy_net_min"] = net_mins
    enriched["s06_hierarchy_net_max"] = net_maxes
    enriched["s06_hierarchy_net_mean"] = net_means
    enriched["s06_hierarchy_net_spread"] = net_spreads
    enriched["s06_hierarchy_top_policy_id"] = top_ids
    enriched["s06_hierarchy_top_display_name"] = top_names
    return enriched


def classify_mosaic_label(row: Mapping[str, Any]) -> str:
    """Assign a conservative rule-based final-state label from measured continua."""

    target = _safe_float(row.get("final_target_quality"), 0.0)
    goal_gap = _safe_float(row.get("goal_conflict_index"), 0.0)
    aggregation = _safe_float(row.get("aggregation_delta_percent"), 0.0)
    largest_block = _safe_float(row.get("largest_block_fraction"), 0.0)
    entropy = _safe_float(row.get("label_entropy"), 0.0)
    position_bias = _safe_float(row.get("position_bias_abs"), 0.0)
    interface_density = _safe_float(row.get("interface_density"), 1.0)
    work_rate = _safe_float(row.get("work_rate"), 0.0)
    frozen_rate = _safe_float(row.get("frozen_block_rate"), 0.0)
    ratio_max = _safe_float(row.get("ratio_max_fraction"), math.nan)
    if math.isfinite(ratio_max) and (ratio_max >= 0.98 or entropy <= 0.12):
        return "homogeneous_high_quality" if target >= 0.75 else "homogeneous_low_progress"
    if goal_gap >= 0.45 and target >= 0.60:
        return "polarized_goal_conflict"
    if goal_gap >= 0.25 and target >= 0.65:
        return "local_goal_tension_mosaic"
    if (frozen_rate >= 0.30 and work_rate <= 0.35) or (target < 0.45 and work_rate <= 0.20):
        return "frozen_or_low_activity"
    if target < 0.55:
        return "low_progress_diffuse"
    if aggregation >= 15.0 and largest_block >= 0.45:
        return "segregated_patch_mosaic"
    if position_bias >= 0.35 and largest_block >= 0.35:
        return "polarized_dominance_mosaic"
    if aggregation >= 8.0 and interface_density <= 0.30:
        return "patchy_mosaic"
    if target >= 0.88 and aggregation < 8.0 and position_bias < 0.25:
        return "integrated_high_quality"
    if target >= 0.75:
        return "mixed_partial_progress"
    return "diffuse_intermediate"


def _class_family(label: str) -> str:
    if label.startswith("homogeneous"):
        return "homogeneous"
    if "polarized" in label:
        return "polarized"
    if "patch" in label or "mosaic" in label:
        return "mosaic"
    if "low" in label or "frozen" in label:
        return "low_progress"
    if "integrated" in label:
        return "integrated"
    return "intermediate"


def add_rule_labels(runs: pd.DataFrame) -> pd.DataFrame:
    labeled = runs.copy()
    labeled["s07_rule_label"] = [classify_mosaic_label(row) for row in labeled.to_dict(orient="records")]
    labeled["s07_class_family"] = labeled["s07_rule_label"].map(_class_family)
    labeled["mosaic_continuum_score"] = (
        np.clip(pd.to_numeric(labeled["aggregation_delta_percent"], errors="coerce").fillna(0.0) / 50.0, 0.0, 1.0) * 0.40
        + np.clip(pd.to_numeric(labeled["largest_block_fraction"], errors="coerce").fillna(0.0), 0.0, 1.0) * 0.35
        + (1.0 - np.clip(pd.to_numeric(labeled["label_entropy"], errors="coerce").fillna(0.0), 0.0, 1.0)) * 0.25
    )
    labeled["conflict_continuum_score"] = (
        np.clip(pd.to_numeric(labeled["goal_conflict_index"], errors="coerce").fillna(0.0), 0.0, 1.0) * 0.55
        + np.clip(pd.to_numeric(labeled["position_bias_abs"], errors="coerce").fillna(0.0) / 0.50, 0.0, 1.0) * 0.45
    )
    return labeled


def _feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    for column in numeric.columns:
        values = numeric[column]
        median = values.median()
        numeric[column] = values.fillna(0.0 if not math.isfinite(float(median)) else float(median))
    scaler = StandardScaler()
    return scaler.fit_transform(numeric.to_numpy(dtype=float))


def add_continuum_coordinates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    matrix = _feature_matrix(result, CLASSIFIER_FEATURE_COLUMNS)
    n_components = min(3, matrix.shape[1], matrix.shape[0])
    pca = PCA(n_components=n_components, random_state=17)
    coords = pca.fit_transform(matrix)
    for idx in range(n_components):
        result[f"s07_continuum_pc{idx + 1}"] = coords[:, idx]
    for idx in range(n_components, 3):
        result[f"s07_continuum_pc{idx + 1}"] = 0.0
    explained = list(pca.explained_variance_ratio_)
    pca_summary = pd.DataFrame(
        [
            {
                "component": f"pc{idx + 1}",
                "explained_variance_ratio": float(value),
            }
            for idx, value in enumerate(explained)
        ]
    )
    return result, pca_summary


def _map_clusters_to_rule_labels(cluster_labels: np.ndarray, rule_labels: Sequence[str]) -> list[str]:
    mapped: dict[int, str] = {}
    rule = list(rule_labels)
    for cluster in sorted(set(int(value) for value in cluster_labels)):
        labels = [rule[idx] for idx, value in enumerate(cluster_labels) if int(value) == cluster]
        if labels:
            mapped[cluster] = Counter(labels).most_common(1)[0][0]
        else:
            mapped[cluster] = "unassigned"
    return [mapped[int(value)] for value in cluster_labels]


def classifier_feature_sets() -> dict[str, tuple[str, ...]]:
    return {
        "spatial_only": SPATIAL_FEATURE_COLUMNS,
        "spatial_target": (*SPATIAL_FEATURE_COLUMNS, "final_target_quality", "sortedness_delta"),
        "spatial_goal_work": CLASSIFIER_FEATURE_COLUMNS,
    }


def run_classifier_settings(
    frame: pd.DataFrame,
    config: S07Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run alternative unsupervised settings and map clusters to rule labels."""

    result = frame.copy()
    setting_rows: list[dict[str, Any]] = []
    prediction_columns: list[str] = []
    rule_labels = result["s07_rule_label"].astype(str).tolist()
    for feature_name, features in classifier_feature_sets().items():
        matrix = _feature_matrix(result, features)
        for k_value in config.classifier_k_values:
            if len(result) <= k_value:
                continue
            for random_state in config.classifier_random_states:
                model = KMeans(n_clusters=int(k_value), random_state=int(random_state), n_init=10)
                clusters = model.fit_predict(matrix)
                column = f"kmeans_{feature_name}_k{k_value}_rs{random_state}"
                mapped = _map_clusters_to_rule_labels(clusters, rule_labels)
                result[column] = mapped
                prediction_columns.append(column)
                setting_rows.append(
                    {
                        "setting_id": column,
                        "method": "kmeans",
                        "feature_set": feature_name,
                        "k": int(k_value),
                        "random_state": int(random_state),
                        "mapped_label_count": len(set(mapped)),
                        "agreement_with_rule_label": float(np.mean(np.asarray(mapped, dtype=object) == np.asarray(rule_labels, dtype=object))),
                        "label_counts_json": _json_object(Counter(mapped)),
                    }
                )
            model = AgglomerativeClustering(n_clusters=int(k_value), linkage="ward")
            clusters = model.fit_predict(matrix)
            column = f"agglomerative_{feature_name}_k{k_value}"
            mapped = _map_clusters_to_rule_labels(clusters, rule_labels)
            result[column] = mapped
            prediction_columns.append(column)
            setting_rows.append(
                {
                    "setting_id": column,
                    "method": "agglomerative",
                    "feature_set": feature_name,
                    "k": int(k_value),
                    "random_state": "",
                    "mapped_label_count": len(set(mapped)),
                    "agreement_with_rule_label": float(np.mean(np.asarray(mapped, dtype=object) == np.asarray(rule_labels, dtype=object))),
                    "label_counts_json": _json_object(Counter(mapped)),
                }
            )

    pairwise_rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(prediction_columns, 2):
        agreement = float(np.mean(result[left].astype(str).to_numpy() == result[right].astype(str).to_numpy()))
        pairwise_rows.append({"left_setting_id": left, "right_setting_id": right, "semantic_label_agreement": agreement})
    if prediction_columns:
        predictions = result[prediction_columns].astype(str)
        consensus_labels: list[str] = []
        consensus_fractions: list[float] = []
        for _, row in predictions.iterrows():
            counts = Counter(row.tolist())
            label, count = counts.most_common(1)[0]
            consensus_labels.append(label)
            consensus_fractions.append(count / max(1, len(prediction_columns)))
        result["s07_classifier_consensus_label"] = consensus_labels
        result["s07_classifier_consensus_fraction"] = consensus_fractions
        result["s07_label"] = np.where(
            result["s07_classifier_consensus_fraction"].astype(float) >= 0.50,
            result["s07_classifier_consensus_label"],
            result["s07_rule_label"],
        )
    else:
        result["s07_classifier_consensus_label"] = result["s07_rule_label"]
        result["s07_classifier_consensus_fraction"] = 1.0
        result["s07_label"] = result["s07_rule_label"]
    pairwise = pd.DataFrame(pairwise_rows)
    setting_df = pd.DataFrame(setting_rows)
    summary = {
        "setting_count": int(len(prediction_columns)),
        "mean_pairwise_semantic_agreement": float(pairwise["semantic_label_agreement"].mean()) if not pairwise.empty else math.nan,
        "min_pairwise_semantic_agreement": float(pairwise["semantic_label_agreement"].min()) if not pairwise.empty else math.nan,
        "mean_rule_agreement": float(setting_df["agreement_with_rule_label"].mean()) if not setting_df.empty else math.nan,
    }
    summary["classifier_settings_stable"] = bool(
        math.isfinite(summary["mean_pairwise_semantic_agreement"])
        and summary["mean_pairwise_semantic_agreement"] >= config.classifier_stability_threshold
        and summary["min_pairwise_semantic_agreement"] >= config.classifier_min_pairwise_agreement
    )
    return result, setting_df, pairwise, summary


def condition_seed_stability(frame: pd.DataFrame, config: S07Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_columns = ["source_research_step_id", "analysis_condition_id"]
    for key, group in frame.groupby(group_columns, sort=False):
        labels = group["s07_label"].astype(str).tolist()
        counts = Counter(labels)
        majority_label, majority_count = counts.most_common(1)[0]
        seed_count = group["seed"].nunique()
        majority_fraction = majority_count / max(1, len(group))
        rows.append(
            {
                "source_research_step_id": key[0],
                "analysis_condition_id": key[1],
                "run_count": int(len(group)),
                "seed_count": int(seed_count),
                "majority_label": majority_label,
                "majority_fraction": float(majority_fraction),
                "stable_across_seeds": bool(majority_fraction >= config.seed_stability_threshold),
                "label_counts_json": _json_object(counts),
            }
        )
    stability = pd.DataFrame(rows)
    summary = {
        "condition_group_count": int(len(stability)),
        "stable_condition_count": int(stability["stable_across_seeds"].sum()) if not stability.empty else 0,
        "stable_condition_fraction": float(stability["stable_across_seeds"].mean()) if not stability.empty else math.nan,
        "mean_majority_fraction": float(stability["majority_fraction"].mean()) if not stability.empty else math.nan,
    }
    summary["seed_labels_stable"] = bool(
        math.isfinite(summary["stable_condition_fraction"])
        and summary["stable_condition_fraction"] >= config.seed_stability_threshold
    )
    return stability, summary


def classification_mode(seed_summary: Mapping[str, Any], classifier_summary: Mapping[str, Any]) -> str:
    if bool(seed_summary.get("seed_labels_stable")) and bool(classifier_summary.get("classifier_settings_stable")):
        return "stable_discrete_taxonomy"
    return "metric_continuum_with_exemplars"


def summarize_classes(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby("s07_label", sort=False):
        rows.append(
            {
                "schema": CLASS_SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "s07_label": label,
                "class_family": group["s07_class_family"].mode().iloc[0],
                "run_count": int(len(group)),
                "condition_count": int(group["analysis_condition_id"].nunique()),
                "source_step_counts_json": _json_object(group["source_research_step_id"].value_counts().sort_index().to_dict()),
                "mean_final_target_quality": float(group["final_target_quality"].mean()),
                "mean_reference_increasing_quality": float(group["reference_increasing_quality"].mean()),
                "mean_goal_conflict_index": float(group["goal_conflict_index"].mean()),
                "mean_aggregation_delta_percent": float(group["aggregation_delta_percent"].mean()),
                "mean_interface_density": float(group["interface_density"].mean()),
                "mean_largest_block_fraction": float(group["largest_block_fraction"].mean()),
                "mean_label_entropy": float(group["label_entropy"].mean()),
                "mean_position_bias_abs": float(group["position_bias_abs"].mean()),
                "mean_work_rate": float(group["work_rate"].mean()),
                "mean_mosaic_continuum_score": float(group["mosaic_continuum_score"].mean()),
                "mean_conflict_continuum_score": float(group["conflict_continuum_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("run_count", ascending=False, kind="mergesort").reset_index(drop=True)


def metric_continua_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_target_quality",
        "reference_increasing_quality",
        "goal_conflict_index",
        "aggregation_delta_percent",
        "interface_density",
        "largest_block_fraction",
        "label_entropy",
        "position_bias_abs",
        "work_rate",
        "swap_rate",
        "frozen_block_rate",
        "mosaic_continuum_score",
        "conflict_continuum_score",
        "s07_continuum_pc1",
        "s07_continuum_pc2",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "schema": CONTINUUM_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "metric": metric,
                "run_count": int(len(values)),
                "min": float(values.min()),
                "p10": float(values.quantile(0.10)),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def select_exemplars(frame: pd.DataFrame, config: S07Config) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for label, group in frame.groupby("s07_label", sort=False):
        matrix = group[list(CLASSIFIER_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
        center = matrix.median(numeric_only=True).to_numpy(dtype=float)
        distances = np.linalg.norm(matrix.to_numpy(dtype=float) - center, axis=1)
        chosen = group.iloc[np.argsort(distances)[: config.exemplar_count_per_label]]
        rows.extend([row for _, row in chosen.iterrows()])
    if not rows:
        return pd.DataFrame()
    exemplars = pd.DataFrame(rows).copy()
    return exemplars.sort_values(["s07_label", "source_research_step_id", "analysis_condition_id"], kind="mergesort").reset_index(drop=True)


def validate_s07_outputs(
    mosaic_df: pd.DataFrame,
    input_counts: Mapping[str, int],
    class_summary: pd.DataFrame,
    continua: pd.DataFrame,
    condition_stability: pd.DataFrame,
    classifier_settings: pd.DataFrame,
    classifier_pairwise: pd.DataFrame,
    exemplars: pd.DataFrame,
    validation_context: Mapping[str, Any],
    config: S07Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    expected_rows = int(sum(input_counts.values()))
    required = set(CLASSIFIER_FEATURE_COLUMNS) | {"s07_label", "s07_rule_label", "source_research_step_id", "analysis_condition_id"}
    finite_metrics = False
    if required.issubset(mosaic_df.columns) and not mosaic_df.empty:
        finite_metrics = bool(
            mosaic_df[list(CLASSIFIER_FEATURE_COLUMNS)]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .all()
            .all()
        )
    s05_context_rows = mosaic_df[mosaic_df["source_research_step_id"].isin(["S02", "S03", "S04"])] if "source_research_step_id" in mosaic_df else pd.DataFrame()
    s05_join_rate = (
        float(s05_context_rows["s05_score_id"].notna().mean())
        if not s05_context_rows.empty and "s05_score_id" in s05_context_rows.columns
        else 0.0
    )
    s06_rows = mosaic_df[mosaic_df["source_research_step_id"] == "S06"] if "source_research_step_id" in mosaic_df else pd.DataFrame()
    s06_context_rate = (
        float(s06_rows["s06_context_context_dependent_dominance"].notna().mean())
        if not s06_rows.empty and "s06_context_context_dependent_dominance" in s06_rows.columns
        else 0.0
    )
    classifier_feature_s06_leak = any(column.startswith("s06_") for column in CLASSIFIER_FEATURE_COLUMNS)
    classification_mode_value = str(validation_context.get("classification_mode", ""))
    seed_stable = bool(validation_context.get("seed_labels_stable", False))
    classifier_stable = bool(validation_context.get("classifier_settings_stable", False))
    checks = [
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "input_run_tables_present",
            "success": bool(input_counts and all(count > 0 for count in input_counts.values())),
            "observed": _json_object(input_counts),
            "expected": "S02, S03, S04, and S06 run tables are present and non-empty",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "harmonized_rows_match_inputs",
            "success": bool(len(mosaic_df) == expected_rows and expected_rows > 0),
            "observed": f"mosaic_rows={len(mosaic_df)} expected_rows={expected_rows}",
            "expected": "one S07 row per input S02/S3/S04/S06 run",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "required_classifier_metrics_finite",
            "success": finite_metrics,
            "observed": f"metrics={len(CLASSIFIER_FEATURE_COLUMNS)} rows={len(mosaic_df)}",
            "expected": "all classifier and continuum metrics are finite",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "s05_context_joined_for_s02_s04",
            "success": bool(s05_join_rate >= 0.99),
            "observed": f"s05_join_rate={s05_join_rate:.3f}",
            "expected": "S02-S04 run rows join to S05 compatibility context by policy, ratio, arrangement, and goal",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "s06_dominance_context_kept_separate",
            "success": bool(s06_context_rate >= 0.99 and not classifier_feature_s06_leak),
            "observed": f"s06_context_rate={s06_context_rate:.3f}; s06_features_used={classifier_feature_s06_leak}",
            "expected": "S06 context-dependent dominance is attached as explanatory context but not used as a classifier feature",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "class_labels_assigned",
            "success": bool(mosaic_df["s07_label"].notna().all() and mosaic_df["s07_label"].nunique() >= 4 and not class_summary.empty),
            "observed": f"labels={sorted(mosaic_df['s07_label'].dropna().unique().tolist()) if 's07_label' in mosaic_df else []}",
            "expected": "S07 assigns multiple interpretable final-state labels",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "metric_continua_written",
            "success": bool(not continua.empty and {"metric", "median", "p10", "p90"}.issubset(continua.columns)),
            "observed": f"continua_rows={len(continua)}",
            "expected": "metric-continuum summary exists for fallback interpretation",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "seed_label_stability_evaluated",
            "success": bool(not condition_stability.empty and "stable_across_seeds" in condition_stability.columns),
            "observed": f"stable_fraction={validation_context.get('stable_condition_fraction', math.nan):.3f}; threshold={config.seed_stability_threshold}",
            "expected": "label stability across matched seeds is evaluated",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "classifier_setting_stability_evaluated",
            "success": bool(not classifier_settings.empty and not classifier_pairwise.empty),
            "observed": f"settings={len(classifier_settings)} mean_pairwise={validation_context.get('mean_pairwise_semantic_agreement', math.nan):.3f}",
            "expected": "label stability across classifier settings is evaluated",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "discrete_labels_stable_or_continuum_fallback",
            "success": bool((seed_stable and classifier_stable and classification_mode_value == "stable_discrete_taxonomy") or classification_mode_value == "metric_continuum_with_exemplars"),
            "observed": f"mode={classification_mode_value}; seed_stable={seed_stable}; classifier_stable={classifier_stable}",
            "expected": "use discrete taxonomy only when seed and classifier-setting stability pass; otherwise report metric continua with exemplars",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "exemplar_panel_written",
            "success": bool(figure_written and not exemplars.empty),
            "observed": f"figure_written={figure_written}; exemplars={len(exemplars)}",
            "expected": "mosaic/final-state exemplar figure and rows are written",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests_success),
            "observed": str(bool(unit_tests_success)),
            "expected": "focused E06 unit tests pass",
        },
    ]
    return pd.DataFrame(checks)
