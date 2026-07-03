"""E06 S05 compatibility metrics for chimeric mixture outputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e06.mixture_ratios import EXPERIMENT_ID, stable_hash


STEP_ID = "S05"
SCORE_SCHEMA = "eidosoma.e06.s05_compatibility_score.v1"
BASELINE_SCHEMA = "eidosoma.e06.s05_pure_baseline.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s05_validation.v1"


@dataclass(frozen=True)
class S05Config:
    """Configuration for compatibility metric construction."""

    array_size: int = 100
    event_cap: int = 4_000
    baseline_array_size: int = 8
    work_scaling_exponent: float = 2.0
    high_quality_threshold: float = 0.85
    low_quality_threshold: float = 0.70
    synergy_tolerance: float = 0.05
    interference_threshold: float = -0.20
    aggregation_conflict_threshold: float = 10.0
    dominance_threshold: float = 0.40
    goal_gap_threshold: float = 0.25


METRIC_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "metric": "primary_target_quality",
        "formula": "S02/S03 final inversion sortedness; S04 assigned-policy goal sortedness.",
        "interpretation": "How well the mixture achieved the target relevant to the current step.",
    },
    {
        "metric": "quality_synergy",
        "formula": "primary_target_quality - ratio-weighted S01 pure final sortedness.",
        "interpretation": "Positive values exceed pure-policy expectation; negative values indicate interference.",
    },
    {
        "metric": "work_interference_log_ratio",
        "formula": "log((observed mean work + 1) / (ratio-weighted scaled S01 pure work + 1)).",
        "interpretation": "Positive values mean the mixture used more work than scaled pure baselines.",
    },
    {
        "metric": "cooperative_efficiency_score",
        "formula": "primary_target_quality / (1 + observed mean work / event cap).",
        "interpretation": "Bounded proxy favoring high target quality with lower work cost.",
    },
    {
        "metric": "integration_score",
        "formula": "1 - clamp(max(aggregation delta percent, 0) / 100, 0, 1).",
        "interpretation": "Higher means less label aggregation beyond random expectation.",
    },
    {
        "metric": "interface_stability_score",
        "formula": "1 - clamp(interface count / (array size - 1), 0, 1).",
        "interpretation": "Higher means fewer label interfaces; interpret with aggregation, not alone.",
    },
    {
        "metric": "dominance_abs_margin",
        "formula": "abs(position bias margin) when available.",
        "interpretation": "Higher means stronger left/right positional dominance by one Algotype.",
    },
    {
        "metric": "goal_conflict_index",
        "formula": "S04 goal-alignment gap; zero for S02/S03 same-reference rows.",
        "interpretation": "Higher means relevant goals disagree more strongly at the final state.",
    },
)


def _json_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _json_object(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)


def _parse_json_list(text: Any) -> tuple[Any, ...]:
    return tuple(json.loads(str(text)))


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if math.isnan(value):
        return math.nan
    return float(min(high, max(low, value)))


def _ratio_label(ratios: Sequence[float]) -> str:
    return ":".join(str(int(round(float(value) * 100))) for value in ratios)


def build_pure_baselines(metadata: pd.DataFrame, validation_runs: pd.DataFrame) -> pd.DataFrame:
    """Build one S01 pure-policy baseline row per Algotype."""

    if metadata.empty:
        raise ValueError("S01 metadata table is empty")
    required_meta = {
        "algotype_id",
        "display_name",
        "source_category",
        "baseline_competence_score",
        "mean_final_inversion_sortedness",
        "mean_work_count",
        "mean_swap_count",
        "validation_run_count",
        "invalid_run_count",
    }
    missing = required_meta - set(metadata.columns)
    if missing:
        raise ValueError(f"S01 metadata missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    validation_summary = pd.DataFrame()
    if not validation_runs.empty and {"algotype_id", "final_inversion_sortedness", "work_count", "array_size"}.issubset(validation_runs.columns):
        validation_summary = (
            validation_runs.groupby("algotype_id", sort=False)
            .agg(
                validation_mean_final_sortedness=("final_inversion_sortedness", "mean"),
                validation_mean_work_count=("work_count", "mean"),
                validation_array_size=("array_size", "median"),
                validation_rows=("algotype_id", "size"),
                validation_invalid_fraction=("invalid", "mean"),
            )
            .reset_index()
        )
    merged = metadata.merge(validation_summary, on="algotype_id", how="left")
    for row in merged.to_dict(orient="records"):
        pure_quality = _safe_float(row.get("mean_final_inversion_sortedness"))
        validation_quality = _safe_float(row.get("validation_mean_final_sortedness"))
        pure_work = _safe_float(row.get("mean_work_count"))
        validation_work = _safe_float(row.get("validation_mean_work_count"))
        array_size = int(_safe_float(row.get("validation_array_size"), 8.0))
        rows.append(
            {
                "schema": BASELINE_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "algotype_id": str(row["algotype_id"]),
                "display_name": str(row["display_name"]),
                "source_category": str(row["source_category"]),
                "baseline_competence_score": _safe_float(row.get("baseline_competence_score")),
                "pure_final_inversion_sortedness": validation_quality if math.isfinite(validation_quality) else pure_quality,
                "pure_mean_work_count": validation_work if math.isfinite(validation_work) else pure_work,
                "pure_mean_swap_count": _safe_float(row.get("mean_swap_count")),
                "pure_validation_run_count": int(_safe_float(row.get("validation_rows"), _safe_float(row.get("validation_run_count"), 0.0))),
                "pure_array_size": array_size,
                "pure_invalid_fraction": _safe_float(row.get("validation_invalid_fraction"), _safe_float(row.get("invalid_run_count"), 0.0)),
                "pure_execution_success": bool(row.get("pure_execution_success", True)),
            }
        )
    result = pd.DataFrame(rows)
    if result["algotype_id"].duplicated().any():
        raise ValueError("duplicate Algotype IDs in pure baseline table")
    return result


def _baseline_maps(baselines: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["algotype_id"]): row for row in baselines.to_dict(orient="records")}


def _weighted_baseline(
    policy_ids: Sequence[str],
    ratios: Sequence[float],
    baselines: Mapping[str, Mapping[str, Any]],
    column: str,
) -> float:
    weights = np.asarray([float(value) for value in ratios], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return math.nan
    weights = weights / total
    values: list[float] = []
    for policy_id in policy_ids:
        if str(policy_id) not in baselines:
            return math.nan
        values.append(_safe_float(baselines[str(policy_id)].get(column)))
    if any(not math.isfinite(value) for value in values):
        return math.nan
    return float(np.dot(weights, np.asarray(values, dtype=float)))


def _scaled_work_count(
    policy_ids: Sequence[str],
    ratios: Sequence[float],
    baselines: Mapping[str, Mapping[str, Any]],
    *,
    target_array_size: int,
    exponent: float,
) -> float:
    weights = np.asarray([float(value) for value in ratios], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return math.nan
    weights = weights / total
    scaled: list[float] = []
    for policy_id in policy_ids:
        baseline = baselines.get(str(policy_id))
        if baseline is None:
            return math.nan
        work = _safe_float(baseline.get("pure_mean_work_count"))
        baseline_size = max(1.0, _safe_float(baseline.get("pure_array_size"), 8.0))
        scaled.append(work * (float(target_array_size) / baseline_size) ** exponent)
    if any(not math.isfinite(value) for value in scaled):
        return math.nan
    return float(np.dot(weights, np.asarray(scaled, dtype=float)))


def _display_from_policy_ids(policy_ids: Sequence[str], baselines: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(str(baselines.get(str(policy_id), {}).get("display_name", policy_id)) for policy_id in policy_ids)


def _categories_from_policy_ids(policy_ids: Sequence[str], baselines: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(str(baselines.get(str(policy_id), {}).get("source_category", "unknown")) for policy_id in policy_ids)


def _source_condition_id(source_step: str, row: Mapping[str, Any], policy_ids: Sequence[str], ratios: Sequence[float]) -> str:
    if source_step == "S02" and row.get("condition_id"):
        return str(row["condition_id"])
    payload = {
        "sourceStep": source_step,
        "policyIds": list(policy_ids),
        "ratios": [round(float(value), 6) for value in ratios],
        "arrangement": row.get("arrangement", ""),
        "goalProfileId": row.get("goal_profile_id", ""),
        "candidate": row.get("s02_candidate_id", row.get("s03_candidate_id", "")),
    }
    return f"{source_step.lower()}_{stable_hash(payload)[:16]}"


def _conflicting_dimensions(
    *,
    primary_quality: float,
    reference_quality: float,
    assigned_quality: float,
    quality_synergy: float,
    aggregation_delta: float,
    dominance_abs: float,
    goal_gap: float,
    work_interference: float,
    config: S05Config,
) -> tuple[str, ...]:
    conflicts: list[str] = []
    if primary_quality >= config.high_quality_threshold and aggregation_delta >= config.aggregation_conflict_threshold:
        conflicts.append("high_quality_high_aggregation")
    if assigned_quality >= config.high_quality_threshold and reference_quality <= 0.65 and goal_gap >= config.goal_gap_threshold:
        conflicts.append("local_goal_success_global_reference_loss")
    if primary_quality >= config.low_quality_threshold and dominance_abs >= config.dominance_threshold:
        conflicts.append("good_target_quality_with_dominance_skew")
    if quality_synergy >= -config.synergy_tolerance and work_interference >= 0.50:
        conflicts.append("baseline_quality_with_high_work_cost")
    if goal_gap >= config.goal_gap_threshold and primary_quality >= config.low_quality_threshold:
        conflicts.append("goal_alignment_conflict")
    return tuple(conflicts)


def _classify_pattern(
    *,
    control_type: str,
    primary_quality: float,
    assigned_quality: float,
    quality_synergy: float,
    aggregation_delta: float,
    dominance_abs: float,
    goal_gap: float,
    config: S05Config,
) -> str:
    if control_type:
        return "pure_or_dummy_label_control"
    if goal_gap >= config.goal_gap_threshold and assigned_quality >= config.low_quality_threshold:
        return "goal_tension_local_success"
    if primary_quality >= config.high_quality_threshold and aggregation_delta >= config.aggregation_conflict_threshold:
        return "high_quality_segregated"
    if (
        primary_quality >= config.high_quality_threshold
        and quality_synergy >= -config.synergy_tolerance
        and aggregation_delta < config.aggregation_conflict_threshold
        and dominance_abs < 0.25
        and goal_gap < 0.10
    ):
        return "cooperative_integrated"
    if quality_synergy <= config.interference_threshold and primary_quality < config.low_quality_threshold:
        return "mutual_interference"
    if dominance_abs >= config.dominance_threshold:
        return "dominance_skewed"
    if primary_quality >= config.low_quality_threshold:
        return "partial_or_contextual_compatibility"
    return "low_compatibility_or_unresolved"


def _score_record(
    *,
    source_step: str,
    source_condition_id: str,
    panel: str,
    condition_kind: str,
    candidate_reason: str,
    arrangement: str,
    goal_profile_id: str,
    goal_compatibility_class: str,
    goal_assignments_json: str,
    state_class: str,
    policy_ids: Sequence[str],
    display_names: Sequence[str],
    source_categories: Sequence[str],
    ratios: Sequence[float],
    seed_count: int,
    run_count: int,
    reference_quality: float,
    assigned_quality: float,
    shared_quality: float,
    best_goal_quality: float,
    goal_gap: float,
    aggregation_delta: float,
    interface_count: float,
    largest_block_fraction: float,
    position_bias_margin: float,
    mean_work_count: float,
    invalid_action_count: int,
    baselines: Mapping[str, Mapping[str, Any]],
    config: S05Config,
    control_type: str = "",
) -> dict[str, Any]:
    policy_ids = tuple(str(item) for item in policy_ids)
    display_names = tuple(str(item) for item in display_names)
    source_categories = tuple(str(item) for item in source_categories)
    ratios = tuple(float(item) for item in ratios)
    primary_quality = assigned_quality if math.isfinite(assigned_quality) else reference_quality
    expected_quality = _weighted_baseline(policy_ids, ratios, baselines, "pure_final_inversion_sortedness")
    expected_competence = _weighted_baseline(policy_ids, ratios, baselines, "baseline_competence_score")
    expected_work = _scaled_work_count(
        policy_ids,
        ratios,
        baselines,
        target_array_size=config.array_size,
        exponent=config.work_scaling_exponent,
    )
    quality_synergy = primary_quality - expected_quality if math.isfinite(expected_quality) else math.nan
    reference_synergy = reference_quality - expected_quality if math.isfinite(expected_quality) else math.nan
    shared_synergy = shared_quality - expected_quality if math.isfinite(shared_quality) and math.isfinite(expected_quality) else math.nan
    work_cost_index = mean_work_count / config.event_cap if config.event_cap > 0 and math.isfinite(mean_work_count) else math.nan
    work_interference = math.log((mean_work_count + 1.0) / (expected_work + 1.0)) if math.isfinite(expected_work) and expected_work >= 0.0 and math.isfinite(mean_work_count) else math.nan
    cooperative_efficiency = primary_quality / (1.0 + work_cost_index) if math.isfinite(primary_quality) and math.isfinite(work_cost_index) else math.nan
    integration_score = 1.0 - _clip(max(0.0, aggregation_delta) / 100.0)
    interface_stability = 1.0 - _clip(interface_count / max(1.0, config.array_size - 1.0)) if math.isfinite(interface_count) else math.nan
    dominance_abs = abs(position_bias_margin) if math.isfinite(position_bias_margin) else 0.0
    goal_conflict = goal_gap if math.isfinite(goal_gap) else 0.0
    conflicts = _conflicting_dimensions(
        primary_quality=primary_quality,
        reference_quality=reference_quality,
        assigned_quality=assigned_quality,
        quality_synergy=quality_synergy,
        aggregation_delta=aggregation_delta,
        dominance_abs=dominance_abs,
        goal_gap=goal_conflict,
        work_interference=work_interference,
        config=config,
    )
    pattern = _classify_pattern(
        control_type=control_type,
        primary_quality=primary_quality,
        assigned_quality=assigned_quality,
        quality_synergy=quality_synergy,
        aggregation_delta=aggregation_delta,
        dominance_abs=dominance_abs,
        goal_gap=goal_conflict,
        config=config,
    )
    score_id = f"s05_{stable_hash({'source': source_step, 'sourceCondition': source_condition_id, 'goalProfile': goal_profile_id, 'control': control_type})[:16]}"
    return {
        "schema": SCORE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "score_id": score_id,
        "source_research_step_id": source_step,
        "source_condition_id": source_condition_id,
        "control_type": control_type,
        "panel": panel,
        "condition_kind": condition_kind,
        "candidate_reason": candidate_reason,
        "arrangement": arrangement,
        "goal_profile_id": goal_profile_id,
        "goal_compatibility_class": goal_compatibility_class,
        "goal_assignments_json": goal_assignments_json,
        "source_state_class": state_class,
        "policy_ids_json": _json_list(policy_ids),
        "display_names_json": _json_list(display_names),
        "source_categories_json": _json_list(source_categories),
        "ratio_targets_json": _json_list(ratios),
        "ratio_label": _ratio_label(ratios),
        "seed_count": int(seed_count),
        "run_count": int(run_count),
        "primary_target_quality": float(primary_quality),
        "reference_increasing_quality": float(reference_quality),
        "assigned_goal_quality": float(assigned_quality),
        "shared_goal_quality": float(shared_quality) if math.isfinite(shared_quality) else math.nan,
        "best_goal_quality": float(best_goal_quality) if math.isfinite(best_goal_quality) else math.nan,
        "expected_pure_quality": float(expected_quality),
        "expected_pure_competence": float(expected_competence),
        "quality_synergy": float(quality_synergy),
        "reference_quality_synergy": float(reference_synergy),
        "shared_goal_synergy": float(shared_synergy) if math.isfinite(shared_synergy) else math.nan,
        "observed_work_count": float(mean_work_count),
        "expected_scaled_pure_work_count": float(expected_work),
        "work_cost_index": float(work_cost_index),
        "work_interference_log_ratio": float(work_interference),
        "cooperative_efficiency_score": float(cooperative_efficiency),
        "aggregation_delta_percent": float(aggregation_delta),
        "integration_score": float(integration_score),
        "interface_count": float(interface_count),
        "interface_stability_score": float(interface_stability),
        "largest_block_fraction": float(largest_block_fraction) if math.isfinite(largest_block_fraction) else math.nan,
        "position_bias_margin": float(position_bias_margin) if math.isfinite(position_bias_margin) else math.nan,
        "dominance_abs_margin": float(dominance_abs),
        "goal_conflict_index": float(goal_conflict),
        "invalid_action_count": int(invalid_action_count),
        "metric_disagreement_count": int(len(conflicts)),
        "conflicting_dimensions_json": _json_list(conflicts),
        "compatibility_pattern_class": pattern,
    }


def _supplement_summary_from_runs(summary: pd.DataFrame, runs: pd.DataFrame, source_step: str) -> pd.DataFrame:
    if summary.empty or runs.empty:
        return summary.copy()
    wanted = {
        "largest_block_fraction": "mean_largest_block_fraction",
        "position_bias_margin": "mean_position_bias_margin",
        "work_count": "mean_work_count",
        "interface_count": "mean_interface_count",
        "aggregation_delta_percent": "mean_aggregation_delta_percent",
    }
    present = {src: dst for src, dst in wanted.items() if src in runs.columns and dst not in summary.columns}
    if not present:
        return summary.copy()
    if source_step == "S02":
        keys = ["condition_id"]
    elif source_step == "S03":
        keys = ["s02_candidate_id", "ratio_targets_json", "arrangement"]
    elif source_step == "S04":
        keys = ["s03_candidate_id", "ratio_targets_json", "arrangement", "goal_profile_id"]
    else:
        return summary.copy()
    if not set(keys).issubset(runs.columns) or not set(keys).issubset(summary.columns):
        return summary.copy()
    aggregate = runs.groupby(keys, sort=False).agg(**{dst: (src, "mean") for src, dst in present.items()}).reset_index()
    return summary.merge(aggregate, on=keys, how="left")


def score_source_summary(
    summary: pd.DataFrame,
    baselines: pd.DataFrame,
    config: S05Config,
    *,
    source_step: str,
    runs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score one source-step summary table against S01 pure baselines."""

    if summary.empty:
        return pd.DataFrame()
    summary = _supplement_summary_from_runs(summary, runs if runs is not None else pd.DataFrame(), source_step)
    baseline_map = _baseline_maps(baselines)
    rows: list[dict[str, Any]] = []
    for raw in summary.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
        ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
        display_names = (
            tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
            if raw.get("display_names_json")
            else _display_from_policy_ids(policy_ids, baseline_map)
        )
        source_categories = (
            tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
            if raw.get("source_categories_json")
            else _categories_from_policy_ids(policy_ids, baseline_map)
        )
        source_condition = _source_condition_id(source_step, raw, policy_ids, ratios)
        if source_step == "S04":
            reference_quality = _safe_float(raw.get("mean_reference_increasing_score"))
            assigned_quality = _safe_float(raw.get("mean_assigned_policy_sortedness"))
            shared_quality = _safe_float(raw.get("mean_shared_goal_score"))
            best_quality = _safe_float(raw.get("mean_best_goal_score"))
            goal_gap = _safe_float(raw.get("mean_goal_alignment_gap"), 0.0)
            condition_kind = "pair"
            state_class = str(raw.get("goal_state_class_mode", ""))
        else:
            reference_quality = _safe_float(raw.get("mean_final_inversion_sortedness"))
            assigned_quality = reference_quality
            shared_quality = reference_quality
            best_quality = reference_quality
            goal_gap = 0.0
            condition_kind = str(raw.get("condition_kind", "pair"))
            state_class = str(raw.get("final_state_class_mode", ""))
        rows.append(
            _score_record(
                source_step=source_step,
                source_condition_id=source_condition,
                panel=str(raw.get("panel", "")),
                condition_kind=condition_kind,
                candidate_reason=str(raw.get("candidate_reason", "")),
                arrangement=str(raw.get("arrangement", "random_permutation_labels" if source_step == "S02" else "")),
                goal_profile_id=str(raw.get("goal_profile_id", "same_increasing" if source_step in {"S02", "S03"} else "")),
                goal_compatibility_class=str(raw.get("goal_compatibility_class", "same_goal" if source_step in {"S02", "S03"} else "")),
                goal_assignments_json=str(raw.get("goal_assignments_json", "")),
                state_class=state_class,
                policy_ids=policy_ids,
                display_names=display_names,
                source_categories=source_categories,
                ratios=ratios,
                seed_count=int(_safe_float(raw.get("seed_count"), 0.0)),
                run_count=int(_safe_float(raw.get("run_count"), 0.0)),
                reference_quality=reference_quality,
                assigned_quality=assigned_quality,
                shared_quality=shared_quality,
                best_goal_quality=best_quality,
                goal_gap=goal_gap,
                aggregation_delta=_safe_float(raw.get("mean_aggregation_delta_percent"), 0.0),
                interface_count=_safe_float(raw.get("mean_interface_count"), 0.0),
                largest_block_fraction=_safe_float(raw.get("mean_largest_block_fraction")),
                position_bias_margin=_safe_float(raw.get("mean_position_bias_margin"), 0.0),
                mean_work_count=_safe_float(raw.get("mean_work_count"), 0.0),
                invalid_action_count=int(_safe_float(raw.get("invalid_action_count"), 0.0)),
                baselines=baseline_map,
                config=config,
            )
        )
    return pd.DataFrame(rows)


def score_all_sources(
    *,
    s02_summary: pd.DataFrame,
    s03_summary: pd.DataFrame,
    s04_summary: pd.DataFrame,
    baselines: pd.DataFrame,
    config: S05Config,
    s02_runs: pd.DataFrame | None = None,
    s03_runs: pd.DataFrame | None = None,
    s04_runs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = [
        score_source_summary(s02_summary, baselines, config, source_step="S02", runs=s02_runs),
        score_source_summary(s03_summary, baselines, config, source_step="S03", runs=s03_runs),
        score_source_summary(s04_summary, baselines, config, source_step="S04", runs=s04_runs),
    ]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_control_scores(baselines: pd.DataFrame, config: S05Config, *, max_dummy_controls: int = 6) -> pd.DataFrame:
    """Build pure-policy and same-policy dummy-label controls for validation."""

    baseline_map = _baseline_maps(baselines)
    rows: list[dict[str, Any]] = []
    for row in baselines.to_dict(orient="records"):
        policy_id = str(row["algotype_id"])
        display = str(row["display_name"])
        quality = _safe_float(row["pure_final_inversion_sortedness"])
        scaled_work = _scaled_work_count((policy_id,), (1.0,), baseline_map, target_array_size=config.array_size, exponent=config.work_scaling_exponent)
        rows.append(
            _score_record(
                source_step="S01_CONTROL",
                source_condition_id=f"s01_pure_{stable_hash(policy_id)[:12]}",
                panel="pure_policy_control",
                condition_kind="pure",
                candidate_reason="pure_policy_baseline",
                arrangement="not_applicable",
                goal_profile_id="same_increasing",
                goal_compatibility_class="pure_control",
                goal_assignments_json=json.dumps({policy_id: "increasing"}, sort_keys=True, separators=(",", ":")),
                state_class="pure_policy_baseline",
                policy_ids=(policy_id,),
                display_names=(display,),
                source_categories=(str(row["source_category"]),),
                ratios=(1.0,),
                seed_count=int(row["pure_validation_run_count"]),
                run_count=int(row["pure_validation_run_count"]),
                reference_quality=quality,
                assigned_quality=quality,
                shared_quality=quality,
                best_goal_quality=quality,
                goal_gap=0.0,
                aggregation_delta=0.0,
                interface_count=0.0,
                largest_block_fraction=1.0,
                position_bias_margin=0.0,
                mean_work_count=scaled_work,
                invalid_action_count=0,
                baselines=baseline_map,
                config=config,
                control_type="pure_policy_baseline_control",
            )
        )
    candidates = baselines.sort_values(["source_category", "display_name"], kind="mergesort").head(max_dummy_controls)
    for row in candidates.to_dict(orient="records"):
        policy_id = str(row["algotype_id"])
        display = str(row["display_name"])
        quality = _safe_float(row["pure_final_inversion_sortedness"])
        scaled_work = _scaled_work_count((policy_id, policy_id), (0.5, 0.5), baseline_map, target_array_size=config.array_size, exponent=config.work_scaling_exponent)
        rows.append(
            _score_record(
                source_step="S01_CONTROL",
                source_condition_id=f"s01_dummy_{stable_hash(policy_id)[:12]}",
                panel="dummy_label_control",
                condition_kind="dummy_same_policy_pair",
                candidate_reason="same_algotype_two_labels",
                arrangement="dummy_balanced_labels",
                goal_profile_id="same_increasing",
                goal_compatibility_class="dummy_label_control",
                goal_assignments_json=json.dumps({f"{policy_id}#a": "increasing", f"{policy_id}#b": "increasing"}, sort_keys=True, separators=(",", ":")),
                state_class="dummy_label_control",
                policy_ids=(policy_id, policy_id),
                display_names=(f"{display}#a", f"{display}#b"),
                source_categories=(str(row["source_category"]), str(row["source_category"])),
                ratios=(0.5, 0.5),
                seed_count=int(row["pure_validation_run_count"]),
                run_count=int(row["pure_validation_run_count"]),
                reference_quality=quality,
                assigned_quality=quality,
                shared_quality=quality,
                best_goal_quality=quality,
                goal_gap=0.0,
                aggregation_delta=0.0,
                interface_count=0.0,
                largest_block_fraction=1.0,
                position_bias_margin=0.0,
                mean_work_count=scaled_work,
                invalid_action_count=0,
                baselines=baseline_map,
                config=config,
                control_type="dummy_label_same_policy_control",
            )
        )
    return pd.DataFrame(rows)


def summarize_dimensions(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouping = ["source_research_step_id", "goal_compatibility_class"]
    for keys, group in scores.groupby(grouping, dropna=False, sort=False):
        source_step, goal_class = keys
        rows.append(
            {
                "source_research_step_id": source_step,
                "goal_compatibility_class": goal_class,
                "score_row_count": int(len(group)),
                "mean_primary_target_quality": float(group["primary_target_quality"].mean()),
                "mean_quality_synergy": float(group["quality_synergy"].mean()),
                "mean_work_interference_log_ratio": float(group["work_interference_log_ratio"].mean()),
                "mean_cooperative_efficiency_score": float(group["cooperative_efficiency_score"].mean()),
                "mean_aggregation_delta_percent": float(group["aggregation_delta_percent"].mean()),
                "mean_integration_score": float(group["integration_score"].mean()),
                "mean_dominance_abs_margin": float(group["dominance_abs_margin"].mean()),
                "mean_goal_conflict_index": float(group["goal_conflict_index"].mean()),
                "mean_metric_disagreement_count": float(group["metric_disagreement_count"].mean()),
                "compatibility_pattern_counts_json": _json_object(group["compatibility_pattern_class"].value_counts().sort_index().to_dict()),
            }
        )
    return pd.DataFrame(rows)


def select_s06_candidate_contests(scores: pd.DataFrame, *, top_n: int = 24) -> pd.DataFrame:
    """Select candidate rows likely useful for S06 dominance/conflict contests."""

    if scores.empty:
        return pd.DataFrame()
    candidate = scores[scores["source_research_step_id"] == "S04"].copy()
    if candidate.empty:
        return pd.DataFrame()
    candidate["s06_priority_score"] = (
        candidate["goal_conflict_index"].fillna(0.0) * 2.0
        + candidate["dominance_abs_margin"].fillna(0.0)
        + (-candidate["quality_synergy"].fillna(0.0)).clip(lower=0.0)
        + candidate["metric_disagreement_count"].fillna(0.0) * 0.10
    )
    columns = [
        "score_id",
        "source_condition_id",
        "panel",
        "arrangement",
        "goal_profile_id",
        "goal_compatibility_class",
        "display_names_json",
        "ratio_targets_json",
        "primary_target_quality",
        "quality_synergy",
        "goal_conflict_index",
        "dominance_abs_margin",
        "aggregation_delta_percent",
        "compatibility_pattern_class",
        "conflicting_dimensions_json",
        "s06_priority_score",
    ]
    return candidate.sort_values("s06_priority_score", ascending=False, kind="mergesort")[columns].head(top_n).reset_index(drop=True)


def validate_s05_outputs(
    scores: pd.DataFrame,
    baselines: pd.DataFrame,
    control_scores: pd.DataFrame,
    *,
    expected_source_counts: Mapping[str, int],
    unit_tests_success: bool,
) -> pd.DataFrame:
    required_score_columns = {
        "score_id",
        "source_research_step_id",
        "policy_ids_json",
        "ratio_targets_json",
        "primary_target_quality",
        "expected_pure_quality",
        "quality_synergy",
        "cooperative_efficiency_score",
        "integration_score",
        "dominance_abs_margin",
        "goal_conflict_index",
        "compatibility_pattern_class",
    }
    has_columns = required_score_columns.issubset(scores.columns)
    source_counts = scores["source_research_step_id"].value_counts().to_dict() if has_columns and not scores.empty else {}
    count_success = bool(all(int(source_counts.get(step, 0)) == int(count) for step, count in expected_source_counts.items()))
    bounded_success = False
    finite_success = False
    no_single_aggregate = False
    disagreement_logged = False
    if has_columns and not scores.empty:
        bounded_columns = [
            "primary_target_quality",
            "reference_increasing_quality",
            "assigned_goal_quality",
            "expected_pure_quality",
            "cooperative_efficiency_score",
            "integration_score",
            "interface_stability_score",
        ]
        bounded_success = bool(all(((scores[col] >= -1e-12) & (scores[col] <= 1.0 + 1e-12)).all() for col in bounded_columns))
        finite_columns = [
            "primary_target_quality",
            "expected_pure_quality",
            "quality_synergy",
            "observed_work_count",
            "expected_scaled_pure_work_count",
            "work_interference_log_ratio",
            "aggregation_delta_percent",
            "dominance_abs_margin",
            "goal_conflict_index",
        ]
        finite_success = bool(np.isfinite(scores[finite_columns].to_numpy(dtype=float)).all())
        no_single_aggregate = not {"compatibility_score", "overall_compatibility_score", "aggregate_compatibility_score"}.intersection(scores.columns)
        disagreement_logged = bool((scores["metric_disagreement_count"] > 0).any())

    pure_controls = control_scores[control_scores["control_type"] == "pure_policy_baseline_control"] if not control_scores.empty else pd.DataFrame()
    dummy_controls = control_scores[control_scores["control_type"] == "dummy_label_same_policy_control"] if not control_scores.empty else pd.DataFrame()
    pure_success = bool(not pure_controls.empty and pure_controls["quality_synergy"].abs().max() <= 1e-12)
    dummy_success = bool(
        not dummy_controls.empty
        and dummy_controls["quality_synergy"].abs().max() <= 1e-12
        and dummy_controls["dominance_abs_margin"].max() <= 1e-12
        and dummy_controls["metric_disagreement_count"].max() == 0
    )
    baseline_success = bool(
        not baselines.empty
        and baselines["algotype_id"].is_unique
        and ((baselines["pure_final_inversion_sortedness"] >= 0.0) & (baselines["pure_final_inversion_sortedness"] <= 1.0)).all()
    )
    checks = [
        {
            "validation_case": "required_score_columns_present",
            "success": has_columns,
            "observed": f"columns={len(scores.columns) if not scores.empty else 0}",
            "expected": "all S05 compatibility dimensions present",
        },
        {
            "validation_case": "source_row_counts_match_inputs",
            "success": count_success,
            "observed": _json_object(source_counts),
            "expected": _json_object(expected_source_counts),
        },
        {
            "validation_case": "pure_baselines_valid",
            "success": baseline_success,
            "observed": f"baseline_rows={len(baselines)}",
            "expected": "unique S01 pure baselines with bounded pure sortedness",
        },
        {
            "validation_case": "bounded_metric_dimensions",
            "success": bounded_success,
            "observed": "bounded quality/integration dimensions" if bounded_success else "out-of-range bounded dimension",
            "expected": "quality-like dimensions are within [0, 1]",
        },
        {
            "validation_case": "finite_metric_dimensions",
            "success": finite_success,
            "observed": "finite required numeric dimensions" if finite_success else "non-finite required numeric dimension",
            "expected": "numeric dimensions required for S05 scoring are finite",
        },
        {
            "validation_case": "pure_policy_controls_center_synergy",
            "success": pure_success,
            "observed": f"pure_control_rows={len(pure_controls)} max_abs_synergy={pure_controls['quality_synergy'].abs().max() if not pure_controls.empty else 'missing'}",
            "expected": "pure-policy control quality synergy is exactly zero",
        },
        {
            "validation_case": "dummy_label_controls_do_not_create_interference",
            "success": dummy_success,
            "observed": f"dummy_control_rows={len(dummy_controls)}",
            "expected": "same-Algotype dummy labels retain baseline quality, no dominance, and no metric disagreement",
        },
        {
            "validation_case": "conflicting_dimensions_kept_separate",
            "success": no_single_aggregate and disagreement_logged,
            "observed": f"no_single_aggregate={no_single_aggregate}; rows_with_disagreements={int((scores['metric_disagreement_count'] > 0).sum()) if has_columns and not scores.empty else 0}",
            "expected": "no aggregate compatibility score and at least one row explicitly logs metric disagreement",
        },
        {
            "validation_case": "no_invalid_actions_in_scored_inputs",
            "success": bool(has_columns and int(scores["invalid_action_count"].sum()) == 0),
            "observed": str(int(scores["invalid_action_count"].sum())) if has_columns and not scores.empty else "missing",
            "expected": "zero invalid actions in scored S02-S04 summaries",
        },
        {
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests_success),
            "observed": str(bool(unit_tests_success)),
            "expected": "focused E06 unit tests pass",
        },
    ]
    return pd.DataFrame(checks)
