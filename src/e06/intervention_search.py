"""E06 S14 prospective minimal intervention search.

S14 uses S13 model failures and feature associations to choose a bounded set
of failure contexts, searches minimal transient/local interventions on new
seeds, then validates the best local options on separate held-out contexts and
seeds. Global-controller-like comparators are retained as a separate stratum.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.coarse_sweep import actor_schedule, sortedness_metrics
from src.e06.goal_compatibility import evaluate_goal_state, goal_assignments_for_profile, rank_values
from src.e06.governance_mechanisms import DEFAULT_GOVERNANCE_MECHANISMS, GovernanceMechanism, governance_decision
from src.e06.interface_rules import (
    DEFAULT_INTERFACE_RULES,
    InterfaceRule,
    frozen_indices_for_profile,
    goal_profile_for_id,
    initial_values_for_profile,
    interface_decision,
)
from src.e06.mixture_ratios import (
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    MixtureRuntime,
    aggregation_metrics,
    classify_final_state,
    counts_for_ratios,
    dominance_metrics,
    stable_hash,
)
from src.e06.spatial_arrangements import arrangement_metric_row, labels_for_arrangement


STEP_ID = "S14"
INTERVENTION_SCHEMA = "eidosoma.e06.s14_intervention_search_run.v1"
CONDITION_SCHEMA = "eidosoma.e06.s14_condition_matrix.v1"
CONTEXT_SCHEMA = "eidosoma.e06.s14_selected_failure_context.v1"
INTERVENTION_SPEC_SCHEMA = "eidosoma.e06.s14_intervention_spec.v1"
SUMMARY_SCHEMA = "eidosoma.e06.s14_intervention_summary.v1"
COMPARISON_SCHEMA = "eidosoma.e06.s14_baseline_comparison.v1"
RANKING_SCHEMA = "eidosoma.e06.s14_intervention_ranking.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s14_validation.v1"

SEARCH_STAGE = "search_design"
VALIDATION_STAGE = "prospective_holdout_validation"
BASELINE_SPEC_ID = "no_intervention_behavior_baseline"
DEFAULT_SEARCH_SEEDS = (2026071401, 2026071402)
DEFAULT_VALIDATION_SEEDS = (2026071491, 2026071492, 2026071493, 2026071494)
SOURCE_STEP_PRIORITY = ("S06", "S09", "S10", "S12", "S08", "S04", "S03", "S02")
FAILURE_CLASSES = {"segregated_low_sort", "low_progress", "polarized_goal_conflict", "mixed_goal_tension"}


@dataclass(frozen=True)
class InterventionSpec:
    """One minimal transient signal or local rule-change candidate."""

    intervention_spec_id: str
    mechanism_family: str
    schedule_family: str
    start_fraction: float
    duration_fraction: float
    interface_rule_id: str
    governance_mechanism_id: str
    access_stratum: str
    information_access_class: str
    influence_radius_cells: int | None
    explicit_recognition_used: bool
    global_controller_like: bool
    intended_effect: str
    rationale_features: tuple[str, ...]
    description: str

    @property
    def minimality_score(self) -> float:
        access_penalty = 0.0
        if self.access_stratum == "behavior_only":
            access_penalty = 0.0
        elif self.global_controller_like:
            access_penalty = 3.0
        elif self.explicit_recognition_used:
            access_penalty = 1.2
        else:
            access_penalty = 1.0
        radius_penalty = 0.0 if self.influence_radius_cells is None else 0.05 * float(self.influence_radius_cells)
        return float(access_penalty + self.duration_fraction + radius_penalty)

    @property
    def active_event_fraction(self) -> float:
        return float(max(0.0, min(1.0, self.duration_fraction)))


DEFAULT_INTERVENTION_SPECS = (
    InterventionSpec(
        intervention_spec_id=BASELINE_SPEC_ID,
        mechanism_family="behavior_only",
        schedule_family="none",
        start_fraction=0.0,
        duration_fraction=0.0,
        interface_rule_id="behavior_only",
        governance_mechanism_id="no_governance",
        access_stratum="behavior_only",
        information_access_class="actor_local_policy_only",
        influence_radius_cells=0,
        explicit_recognition_used=False,
        global_controller_like=False,
        intended_effect="matched no-intervention comparator",
        rationale_features=("intervention_stratum",),
        description="No transient signal and no local rule change.",
    ),
    InterventionSpec(
        intervention_spec_id="early_adhesion_pulse_15pct",
        mechanism_family="transient_interface_signal",
        schedule_family="early_short_pulse",
        start_fraction=0.10,
        duration_fraction=0.15,
        interface_rule_id="adhesion_like_preference",
        governance_mechanism_id="no_governance",
        access_stratum="local_explicit_interface",
        information_access_class="local_label_identity",
        influence_radius_cells=1,
        explicit_recognition_used=True,
        global_controller_like=False,
        intended_effect="stabilize local same-label contacts during early sorting",
        rationale_features=("interface_rule_id", "interface_access_stratum", "aggregation_delta_percent"),
        description="Early label-aware adhesion-like pulse, then return to behavior-only rules.",
    ),
    InterventionSpec(
        intervention_spec_id="early_permeability_barrier_15pct",
        mechanism_family="transient_interface_signal",
        schedule_family="early_short_pulse",
        start_fraction=0.10,
        duration_fraction=0.15,
        interface_rule_id="permeability_barrier",
        governance_mechanism_id="no_governance",
        access_stratum="local_explicit_interface",
        information_access_class="local_label_identity",
        influence_radius_cells=1,
        explicit_recognition_used=True,
        global_controller_like=False,
        intended_effect="briefly suppress heterotypic swaps when early mixing predicts failure",
        rationale_features=("interface_rule_id", "interface_access_stratum", "largest_block_fraction"),
        description="Early semipermeable interface pulse, then return to behavior-only rules.",
    ),
    InterventionSpec(
        intervention_spec_id="early_quorum_signal_radius2_15pct",
        mechanism_family="transient_local_signal",
        schedule_family="early_short_pulse",
        start_fraction=0.10,
        duration_fraction=0.15,
        interface_rule_id="behavior_only",
        governance_mechanism_id="quorum_signal_radius2",
        access_stratum="finite_radius_governance",
        information_access_class="local_label_density_radius_2",
        influence_radius_cells=2,
        explicit_recognition_used=False,
        global_controller_like=False,
        intended_effect="use local heterotypic-density signal to damp extra mixing",
        rationale_features=("governance_information_access_class", "ratio_imbalance", "arrangement"),
        description="Early radius-2 quorum signal, then return to no-governance behavior.",
    ),
    InterventionSpec(
        intervention_spec_id="early_conflict_resolution_radius2_15pct",
        mechanism_family="transient_local_signal",
        schedule_family="early_short_pulse",
        start_fraction=0.10,
        duration_fraction=0.15,
        interface_rule_id="behavior_only",
        governance_mechanism_id="conflict_resolution_radius2",
        access_stratum="finite_radius_governance",
        information_access_class="local_goal_proxy_radius_2_for_actor_and_target",
        influence_radius_cells=2,
        explicit_recognition_used=False,
        global_controller_like=False,
        intended_effect="briefly reject swaps that worsen local assigned-goal satisfaction",
        rationale_features=("goal_compatibility_class", "governance_information_access_class", "goal_conflict_index"),
        description="Early radius-2 conflict-resolution signal, then return to no-governance behavior.",
    ),
    InterventionSpec(
        intervention_spec_id="mid_conflict_resolution_radius2_15pct",
        mechanism_family="transient_local_signal",
        schedule_family="mid_short_pulse",
        start_fraction=0.35,
        duration_fraction=0.15,
        interface_rule_id="behavior_only",
        governance_mechanism_id="conflict_resolution_radius2",
        access_stratum="finite_radius_governance",
        information_access_class="local_goal_proxy_radius_2_for_actor_and_target",
        influence_radius_cells=2,
        explicit_recognition_used=False,
        global_controller_like=False,
        intended_effect="test whether delaying the same local conflict signal is more minimal",
        rationale_features=("history_introduction_event_fraction", "goal_compatibility_class"),
        description="Mid-run radius-2 conflict-resolution pulse, then return to no-governance behavior.",
    ),
    InterventionSpec(
        intervention_spec_id="early_organizer_patch_radius3_15pct",
        mechanism_family="transient_local_signal",
        schedule_family="early_short_pulse",
        start_fraction=0.10,
        duration_fraction=0.15,
        interface_rule_id="behavior_only",
        governance_mechanism_id="organizer_patch_radius3",
        access_stratum="finite_radius_governance",
        information_access_class="fixed_organizer_patch_local_values_and_labels_radius_3",
        influence_radius_cells=3,
        explicit_recognition_used=False,
        global_controller_like=False,
        intended_effect="brief sparse organizer-like local coordination",
        rationale_features=("arrangement", "governance_information_access_class"),
        description="Early sparse organizer signal with radius-3 local access, then return to no governance.",
    ),
    InterventionSpec(
        intervention_spec_id="full_local_recognition_gate",
        mechanism_family="local_rule_change",
        schedule_family="full_run_rule_change",
        start_fraction=0.0,
        duration_fraction=1.0,
        interface_rule_id="local_recognition_gate",
        governance_mechanism_id="no_governance",
        access_stratum="local_explicit_interface",
        information_access_class="local_label_identity_and_goal_proxy",
        influence_radius_cells=1,
        explicit_recognition_used=True,
        global_controller_like=False,
        intended_effect="heavier local explicit-recognition comparator for minimality",
        rationale_features=("interface_rule_id", "interface_access_stratum"),
        description="Full-run local recognition gate; included as a non-transient local rule-change comparator.",
    ),
    InterventionSpec(
        intervention_spec_id="global_reference_pulse_15pct",
        mechanism_family="transient_global_comparator",
        schedule_family="early_short_pulse",
        start_fraction=0.10,
        duration_fraction=0.15,
        interface_rule_id="behavior_only",
        governance_mechanism_id="global_reference_controller",
        access_stratum="global_controller_like",
        information_access_class="global_array_state_and_reference_goal",
        influence_radius_cells=None,
        explicit_recognition_used=False,
        global_controller_like=True,
        intended_effect="separate oracle-like global sortedness access from local mechanisms",
        rationale_features=("global_controller_like", "governance_information_access_class"),
        description="Early global-reference comparator pulse; explicitly not local-governance evidence.",
    ),
)


@dataclass(frozen=True)
class S14Config:
    """Configuration for S14 prospective intervention search."""

    array_size: int = 100
    event_cap: int = 4_000
    search_seeds: tuple[int, ...] = DEFAULT_SEARCH_SEEDS
    validation_seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS
    max_search_contexts: int = 4
    max_holdout_contexts: int = 2
    validation_local_intervention_count: int = 3
    intervention_specs: tuple[InterventionSpec, ...] = DEFAULT_INTERVENTION_SPECS
    scheduler: str = "cyclic_scan_seed_offset"
    worker_count: int = 1
    rescue_delta_threshold: float = 0.01
    minimality_penalty_weight: float = 0.01


@dataclass(frozen=True)
class FailureContext:
    """One pairwise context selected from S13 failures."""

    base_context_id: str
    evaluation_stage: str
    source_research_step_id: str
    source_condition_id: str
    source_run_condition_id: str
    condition_group_id: str
    selection_rank: int
    selection_reason: str
    s13_failure_score: float
    s13_final_state_class_mode: str
    s13_mean_final_target_quality: float
    s13_mean_goal_conflict_index: float
    s13_classification_error_rate: float
    s13_mean_quality_abs_error: float
    policy_ids: tuple[str, str]
    display_names: tuple[str, str]
    source_categories: tuple[str, str]
    ratio_targets: tuple[float, float]
    arrangement: str
    goal_profile_id: str
    goal_compatibility_class: str
    value_profile: str
    perturbation_profile: str
    panel: str
    candidate_reason: str
    feature_rationale: tuple[str, ...]


@dataclass(frozen=True)
class InterventionCondition:
    """One executable S14 condition before seed expansion."""

    condition_id: str
    context: FailureContext
    spec: InterventionSpec
    active_interface_rule: InterfaceRule
    active_governance: GovernanceMechanism


def _json_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), separators=(",", ":"), default=str)


def _json_object(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)


def _parse_json_list(text: Any) -> tuple[Any, ...]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return tuple()
    if isinstance(text, Sequence) and not isinstance(text, (str, bytes, bytearray)):
        return tuple(text)
    try:
        parsed = json.loads(str(text))
    except json.JSONDecodeError:
        return tuple()
    if isinstance(parsed, list):
        return tuple(parsed)
    return tuple()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mode(values: pd.Series) -> str:
    if values.empty:
        return ""
    mode = values.astype(str).mode()
    return str(mode.iloc[0]) if not mode.empty else str(values.iloc[0])


def _normalize_arrangement(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"", "nan", "not_applicable", "random_permutation_labels"}:
        return "random_permutation"
    return text


def _normalize_value_profile(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"", "nan", "not_applicable"}:
        return "random_permutation"
    return text


def _normalize_perturbation_profile(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"", "nan", "not_applicable"}:
        return "none"
    return text


def _ratios_from_label(label: Any, policy_count: int) -> tuple[float, ...]:
    parts = []
    for item in str(label or "").split(":"):
        try:
            parts.append(float(item) / 100.0)
        except ValueError:
            continue
    if len(parts) != int(policy_count) or sum(parts) <= 0:
        return tuple([1.0 / float(policy_count)] * int(policy_count))
    total = sum(parts)
    return tuple(float(part / total) for part in parts)


def _context_id(payload: Mapping[str, Any]) -> str:
    return f"s14_base_{stable_hash(payload)[:16]}"


def _condition_id(base_context_id: str, spec_id: str, stage: str) -> str:
    return f"s14_{stable_hash({'baseContextId': base_context_id, 'specId': spec_id, 'stage': stage})[:16]}"


def _interface_by_id() -> dict[str, InterfaceRule]:
    return {rule.rule_id: rule for rule in DEFAULT_INTERFACE_RULES}


def _governance_by_id() -> dict[str, GovernanceMechanism]:
    return {mechanism.mechanism_id: mechanism for mechanism in DEFAULT_GOVERNANCE_MECHANISMS}


def _feature_rationale(feature_screen: pd.DataFrame) -> tuple[str, ...]:
    if feature_screen.empty or "feature_name" not in feature_screen.columns:
        return ("s13_feature_screen_unavailable",)
    ranked = (
        feature_screen.groupby("feature_name", sort=False)["mean_importance"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .index.astype(str)
        .tolist()
    )
    return tuple(ranked) if ranked else ("s13_feature_screen_empty",)


def intervention_spec_table(
    config: S14Config | None = None,
    feature_screen: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return machine-readable intervention definitions and minimality fields."""

    config = config or S14Config()
    top_features = set(_feature_rationale(feature_screen if feature_screen is not None else pd.DataFrame()))
    rows: list[dict[str, Any]] = []
    for rank, spec in enumerate(config.intervention_specs, start=1):
        rationale = tuple(dict.fromkeys([*spec.rationale_features, *[item for item in top_features if item in spec.rationale_features]]))
        rows.append(
            {
                "schema": INTERVENTION_SPEC_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "intervention_spec_id": spec.intervention_spec_id,
                "spec_rank": rank,
                "mechanism_family": spec.mechanism_family,
                "schedule_family": spec.schedule_family,
                "start_fraction": float(spec.start_fraction),
                "duration_fraction": float(spec.duration_fraction),
                "active_event_fraction": spec.active_event_fraction,
                "interface_rule_id": spec.interface_rule_id,
                "governance_mechanism_id": spec.governance_mechanism_id,
                "access_stratum": spec.access_stratum,
                "information_access_class": spec.information_access_class,
                "influence_radius_cells": -1 if spec.influence_radius_cells is None else int(spec.influence_radius_cells),
                "explicit_recognition_used": bool(spec.explicit_recognition_used),
                "global_controller_like": bool(spec.global_controller_like),
                "minimality_score": spec.minimality_score,
                "intended_effect": spec.intended_effect,
                "rationale_features_json": _json_list(rationale),
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)


def _prediction_error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["prediction_row_id", "classification_error_rate", "final_target_quality_abs_error"])
    rows = pd.DataFrame({"prediction_row_id": sorted(set(predictions["prediction_row_id"].astype(str)))})
    rf = predictions[predictions["model_id"].astype(str) == "random_forest"].copy()
    if rf.empty:
        return rows.assign(classification_error_rate=0.0, final_target_quality_abs_error=0.0)
    cls = rf[rf["target_name"].astype(str) == "final_state_class"].copy()
    if not cls.empty:
        cls["classification_error"] = cls["observed_value"].astype(str) != cls["predicted_value"].astype(str)
        rows = rows.merge(
            cls.groupby("prediction_row_id", as_index=False)["classification_error"].mean().rename(
                columns={"classification_error": "classification_error_rate"}
            ),
            on="prediction_row_id",
            how="left",
        )
    else:
        rows["classification_error_rate"] = 0.0
    qual = rf[rf["target_name"].astype(str) == "final_target_quality"].copy()
    if not qual.empty:
        qual["final_target_quality_abs_error"] = (qual["observed_numeric"].astype(float) - qual["predicted_numeric"].astype(float)).abs()
        rows = rows.merge(
            qual.groupby("prediction_row_id", as_index=False)["final_target_quality_abs_error"].mean(),
            on="prediction_row_id",
            how="left",
        )
    else:
        rows["final_target_quality_abs_error"] = 0.0
    return rows.fillna({"classification_error_rate": 0.0, "final_target_quality_abs_error": 0.0})


def select_s14_failure_contexts(
    s13_matrix: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_screen: pd.DataFrame,
    records: Sequence[Mapping[str, Any]],
    config: S14Config | None = None,
) -> pd.DataFrame:
    """Select bounded search and held-out validation contexts from S13 failures."""

    config = config or S14Config()
    if s13_matrix.empty:
        raise ValueError("S13 model input matrix is empty")
    required = {
        "prediction_row_id",
        "source_research_step_id",
        "source_condition_id",
        "source_run_condition_id",
        "condition_group_id",
        "policy_count",
        "policy_signature",
        "display_signature",
        "source_category_signature",
        "ratio_label",
        "arrangement",
        "goal_profile_id",
        "goal_compatibility_class",
        "value_profile",
        "perturbation_profile",
        "final_state_class",
        "final_target_quality",
        "goal_conflict_index",
    }
    missing = required - set(s13_matrix.columns)
    if missing:
        raise ValueError(f"S13 model input matrix missing S14 columns: {sorted(missing)}")

    record_ids = {str(record["algotypeId"]) for record in records}
    errors = _prediction_error_table(predictions)
    matrix = s13_matrix.merge(errors, on="prediction_row_id", how="left")
    matrix["classification_error_rate"] = matrix["classification_error_rate"].fillna(0.0)
    matrix["final_target_quality_abs_error"] = matrix["final_target_quality_abs_error"].fillna(0.0)
    matrix = matrix[matrix["policy_count"].round().astype(int) == 2].copy()
    matrix["policy_ids_tuple"] = matrix["policy_signature"].map(lambda value: tuple(str(item) for item in _parse_json_list(value)))
    matrix = matrix[matrix["policy_ids_tuple"].map(lambda ids: len(ids) == 2 and set(ids).issubset(record_ids))].copy()
    matrix = matrix[matrix["source_research_step_id"].astype(str).isin(SOURCE_STEP_PRIORITY)].copy()
    if matrix.empty:
        raise ValueError("S14 could not find pairwise S13 failure contexts whose policies are in the S01 library")

    matrix["low_quality_gap"] = (0.85 - pd.to_numeric(matrix["final_target_quality"], errors="coerce")).clip(lower=0.0).fillna(0.0)
    matrix["conflict_pressure"] = pd.to_numeric(matrix["goal_conflict_index"], errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)
    matrix["bad_class_indicator"] = matrix["final_state_class"].astype(str).isin(FAILURE_CLASSES).astype(float)
    matrix["source_priority"] = matrix["source_research_step_id"].map(lambda value: SOURCE_STEP_PRIORITY.index(str(value)) if str(value) in SOURCE_STEP_PRIORITY else 99)
    matrix["s13_failure_score"] = (
        0.45 * matrix["low_quality_gap"]
        + 0.30 * matrix["conflict_pressure"]
        + 0.15 * matrix["classification_error_rate"]
        + 0.10 * matrix["bad_class_indicator"]
        + 0.10 * matrix["final_target_quality_abs_error"].clip(upper=1.0)
    )

    grouping = [
        "condition_group_id",
        "source_research_step_id",
        "source_condition_id",
        "source_run_condition_id",
        "policy_signature",
        "display_signature",
        "source_category_signature",
        "ratio_label",
        "arrangement",
        "goal_profile_id",
        "goal_compatibility_class",
        "value_profile",
        "perturbation_profile",
        "panel",
        "candidate_reason",
    ]
    grouped = (
        matrix.groupby(grouping, dropna=False)
        .agg(
            s13_failure_score=("s13_failure_score", "mean"),
            s13_final_state_class_mode=("final_state_class", _mode),
            s13_mean_final_target_quality=("final_target_quality", "mean"),
            s13_mean_goal_conflict_index=("goal_conflict_index", "mean"),
            s13_classification_error_rate=("classification_error_rate", "mean"),
            s13_mean_quality_abs_error=("final_target_quality_abs_error", "mean"),
            source_priority=("source_priority", "min"),
            run_row_count=("prediction_row_id", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(["s13_failure_score", "source_priority", "s13_mean_goal_conflict_index"], ascending=[False, True, False], kind="mergesort")

    total_needed = int(config.max_search_contexts + config.max_holdout_contexts)
    selected_indices: list[int] = []
    seen_sources: set[str] = set()
    for idx, row in grouped.iterrows():
        source = str(row["source_research_step_id"])
        if source not in seen_sources or len(selected_indices) >= len(SOURCE_STEP_PRIORITY):
            selected_indices.append(int(idx))
            seen_sources.add(source)
        if len(selected_indices) >= total_needed:
            break
    for idx in grouped.index:
        if len(selected_indices) >= total_needed:
            break
        if int(idx) not in selected_indices:
            selected_indices.append(int(idx))
    selected = grouped.loc[selected_indices[:total_needed]].copy()

    top_features = _feature_rationale(feature_screen)
    rows: list[dict[str, Any]] = []
    for rank, raw in enumerate(selected.to_dict(orient="records"), start=1):
        policies = tuple(str(item) for item in _parse_json_list(raw["policy_signature"]))
        displays = tuple(str(item) for item in _parse_json_list(raw["display_signature"]))
        categories = tuple(str(item) for item in _parse_json_list(raw["source_category_signature"]))
        ratios = tuple(float(item) for item in _ratios_from_label(raw["ratio_label"], len(policies)))
        payload = {
            "sourceStep": raw["source_research_step_id"],
            "conditionGroup": raw["condition_group_id"],
            "policyIds": policies,
            "ratios": [round(value, 6) for value in ratios],
            "arrangement": _normalize_arrangement(raw["arrangement"]),
            "goalProfileId": raw["goal_profile_id"],
        }
        evaluation_stage = SEARCH_STAGE if rank <= config.max_search_contexts else VALIDATION_STAGE
        rows.append(
            {
                "schema": CONTEXT_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "base_context_id": _context_id(payload),
                "evaluation_stage": evaluation_stage,
                "source_research_step_id": str(raw["source_research_step_id"]),
                "source_condition_id": str(raw["source_condition_id"]),
                "source_run_condition_id": str(raw["source_run_condition_id"]),
                "condition_group_id": str(raw["condition_group_id"]),
                "selection_rank": rank,
                "selection_reason": f"s13_failure_context:score={float(raw['s13_failure_score']):.4f}:stage={evaluation_stage}",
                "s13_failure_score": float(raw["s13_failure_score"]),
                "s13_final_state_class_mode": str(raw["s13_final_state_class_mode"]),
                "s13_mean_final_target_quality": float(raw["s13_mean_final_target_quality"]),
                "s13_mean_goal_conflict_index": float(raw["s13_mean_goal_conflict_index"]),
                "s13_classification_error_rate": float(raw["s13_classification_error_rate"]),
                "s13_mean_quality_abs_error": float(raw["s13_mean_quality_abs_error"]),
                "run_row_count": int(raw["run_row_count"]),
                "policy_ids_json": _json_list(policies),
                "display_names_json": _json_list(displays),
                "source_categories_json": _json_list(categories),
                "ratio_targets_json": _json_list(ratios),
                "arrangement": _normalize_arrangement(raw["arrangement"]),
                "goal_profile_id": str(raw["goal_profile_id"]) if str(raw["goal_profile_id"]) != "not_applicable" else "same_increasing",
                "goal_compatibility_class": str(raw["goal_compatibility_class"]) if str(raw["goal_compatibility_class"]) != "not_applicable" else "same_goal",
                "value_profile": _normalize_value_profile(raw.get("value_profile", "random_permutation")),
                "perturbation_profile": _normalize_perturbation_profile(raw.get("perturbation_profile", "none")),
                "panel": str(raw.get("panel", "not_available")),
                "candidate_reason": str(raw.get("candidate_reason", "s13_failure_context")),
                "feature_rationale_json": _json_list(top_features),
            }
        )
    return pd.DataFrame(rows)


def _context_from_row(raw: Mapping[str, Any]) -> FailureContext:
    policies = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
    displays = tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
    categories = tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
    ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
    return FailureContext(
        base_context_id=str(raw["base_context_id"]),
        evaluation_stage=str(raw["evaluation_stage"]),
        source_research_step_id=str(raw["source_research_step_id"]),
        source_condition_id=str(raw["source_condition_id"]),
        source_run_condition_id=str(raw["source_run_condition_id"]),
        condition_group_id=str(raw["condition_group_id"]),
        selection_rank=int(raw["selection_rank"]),
        selection_reason=str(raw["selection_reason"]),
        s13_failure_score=float(raw["s13_failure_score"]),
        s13_final_state_class_mode=str(raw["s13_final_state_class_mode"]),
        s13_mean_final_target_quality=_safe_float(raw["s13_mean_final_target_quality"]),
        s13_mean_goal_conflict_index=_safe_float(raw["s13_mean_goal_conflict_index"]),
        s13_classification_error_rate=_safe_float(raw["s13_classification_error_rate"]),
        s13_mean_quality_abs_error=_safe_float(raw["s13_mean_quality_abs_error"]),
        policy_ids=(policies[0], policies[1]),
        display_names=(displays[0], displays[1]),
        source_categories=(categories[0], categories[1]),
        ratio_targets=(ratios[0], ratios[1]),
        arrangement=_normalize_arrangement(raw["arrangement"]),
        goal_profile_id=str(raw["goal_profile_id"]),
        goal_compatibility_class=str(raw["goal_compatibility_class"]),
        value_profile=_normalize_value_profile(raw["value_profile"]),
        perturbation_profile=_normalize_perturbation_profile(raw["perturbation_profile"]),
        panel=str(raw["panel"]),
        candidate_reason=str(raw["candidate_reason"]),
        feature_rationale=tuple(str(item) for item in _parse_json_list(raw["feature_rationale_json"])),
    )


def build_s14_conditions(selected_contexts: pd.DataFrame, specs: Sequence[InterventionSpec], stage: str) -> list[InterventionCondition]:
    interfaces = _interface_by_id()
    governance = _governance_by_id()
    contexts = selected_contexts[selected_contexts["evaluation_stage"].astype(str) == str(stage)].copy()
    conditions: list[InterventionCondition] = []
    for raw in contexts.to_dict(orient="records"):
        context = _context_from_row(raw)
        for spec in specs:
            if spec.interface_rule_id not in interfaces:
                raise ValueError(f"S14 intervention references unknown interface rule: {spec.interface_rule_id}")
            if spec.governance_mechanism_id not in governance:
                raise ValueError(f"S14 intervention references unknown governance mechanism: {spec.governance_mechanism_id}")
            conditions.append(
                InterventionCondition(
                    condition_id=_condition_id(context.base_context_id, spec.intervention_spec_id, stage),
                    context=context,
                    spec=spec,
                    active_interface_rule=interfaces[spec.interface_rule_id],
                    active_governance=governance[spec.governance_mechanism_id],
                )
            )
    return conditions


def _memory_success(memory: dict[int, dict[str, Any]], cell_id: int) -> None:
    state = memory.setdefault(int(cell_id), {})
    state["last_move_success"] = True
    state["last_action_type"] = "swap"
    state["time_since_movement"] = 0
    state["local_frustration"] = max(0, int(state.get("local_frustration", 0)) - 1)


def _memory_failure(memory: dict[int, dict[str, Any]], cell_id: int, action_type: str) -> None:
    state = memory.setdefault(int(cell_id), {})
    state["last_move_success"] = False
    state["last_action_type"] = action_type
    state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)
    state["local_frustration"] = min(255, int(state.get("local_frustration", 0)) + 1)
    state["failed_swap_count"] = min(255, int(state.get("failed_swap_count", 0)) + 1)


def _memory_wait(memory: dict[int, dict[str, Any]], cell_id: int) -> None:
    state = memory.setdefault(int(cell_id), {})
    state["last_move_success"] = None
    state["last_action_type"] = "wait"
    state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)


def _active_window(spec: InterventionSpec, event_cap: int) -> tuple[int, int]:
    start = max(0, min(int(event_cap), int(round(float(spec.start_fraction) * event_cap))))
    duration = max(0, min(int(event_cap), int(round(float(spec.duration_fraction) * event_cap))))
    return start, min(int(event_cap), start + duration)


def simulate_intervention_condition(
    condition: InterventionCondition,
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: S14Config,
) -> dict[str, Any]:
    counts = counts_for_ratios(condition.context.ratio_targets, config.array_size)
    values = list(initial_values_for_profile(config.array_size, seed, condition.context.value_profile))
    labels = list(
        labels_for_arrangement(
            condition.context.policy_ids,
            counts,
            seed=seed,
            condition_id=condition.context.base_context_id,
            arrangement=condition.context.arrangement,
        )
    )
    initial_labels = tuple(labels)
    cell_ids = list(range(config.array_size))
    memory: dict[int, dict[str, Any]] = {}
    runtime = MixtureRuntime([records_by_id[policy_id] for policy_id in condition.context.policy_ids], seed)
    schedule = actor_schedule(config.array_size, config.event_cap, seed)
    rng = random.Random(seed ^ int(stable_hash({"s14": condition.condition_id, "spec": condition.spec.intervention_spec_id})[:12], 16))
    goal_profile = goal_profile_for_id(condition.context.goal_profile_id)
    goal_assignments = goal_assignments_for_profile(condition.context.policy_ids, goal_profile)
    frozen_indices = frozen_indices_for_profile(config.array_size, condition.context.perturbation_profile)
    active_start, active_end = _active_window(condition.spec, config.event_cap)
    baseline_interface = _interface_by_id()["behavior_only"]
    baseline_governance = _governance_by_id()["no_governance"]

    started = time.perf_counter()
    initial_metrics = sortedness_metrics(values)
    initial_arrangement = arrangement_metric_row(initial_labels, condition.context.policy_ids)
    initial_counts = Counter(labels)
    counters: Counter[str] = Counter()
    same_contact_delta_sum = 0.0
    local_goal_delta_sum = 0.0
    governance_local_goal_delta_sum = 0.0
    governance_heterotypic_edge_delta_sum = 0.0
    governance_block_reasons: Counter[str] = Counter()
    interface_block_reasons: Counter[str] = Counter()

    for event_index, actor_index in enumerate(schedule):
        actor_index = int(actor_index)
        active = bool(active_start <= int(event_index) < active_end and condition.spec.intervention_spec_id != BASELINE_SPEC_ID)
        rule = condition.active_interface_rule if active else baseline_interface
        mechanism = condition.active_governance if active else baseline_governance
        counters["intervention_active_event_count"] += int(active)
        policy_id = str(labels[actor_index])
        cell_id = int(cell_ids[actor_index])
        actor_goal = goal_assignments[policy_id]
        proxy_values = rank_values(values, actor_goal, config.array_size)
        action = runtime.action_for(policy_id, proxy_values, labels, actor_index, cell_id, memory, rng)
        counters["compare_count"] += int(bool(action.compare_counted))
        target = None if action.target_index is None else int(action.target_index)
        if target is not None and (actor_index in frozen_indices or target in frozen_indices):
            counters["frozen_block_count"] += 1
            counters["wait_count"] += 1
            _memory_failure(memory, cell_id, "frozen_block")
            continue
        if action.action_type == "swap" and target is not None:
            if 0 <= target < len(values):
                counters["raw_swap_proposal_count"] += 1
                pre_cross = str(labels[actor_index]) != str(labels[target])
                counters["pre_governance_cross_label_proposal_count"] += int(pre_cross)
                governance = governance_decision(
                    mechanism,
                    values=values,
                    labels=labels,
                    actor_index=actor_index,
                    target_index=target,
                    goal_assignments=goal_assignments,
                    policy_ids=condition.context.policy_ids,
                    array_size=config.array_size,
                    rng=rng,
                )
                if bool(governance["invoked"]):
                    counters["governance_invocation_count"] += 1
                    counters["governance_cost_units"] += int(governance["cost_units"])
                    counters["global_controller_invocation_count"] += int(mechanism.global_controller_like)
                    counters["active_governance_invocation_count"] += int(active)
                    if math.isfinite(float(governance["local_goal_before"])):
                        governance_local_goal_delta_sum += float(governance["local_goal_after"]) - float(governance["local_goal_before"])
                    if math.isfinite(float(governance["heterotypic_edge_before"])):
                        governance_heterotypic_edge_delta_sum += float(governance["heterotypic_edge_after"]) - float(governance["heterotypic_edge_before"])
                if not bool(governance["accepted"]):
                    counters["governance_block_count"] += 1
                    counters["governance_cross_label_block_count"] += int(pre_cross)
                    reason = str(governance["block_reason"])
                    governance_block_reasons[reason] += 1
                    counters["wait_count"] += 1
                    _memory_failure(memory, cell_id, f"governance_block:{reason}")
                    continue
                decision = interface_decision(
                    rule,
                    values=values,
                    labels=labels,
                    actor_index=actor_index,
                    target_index=target,
                    goal_assignments=goal_assignments,
                    array_size=config.array_size,
                    rng=rng,
                )
                if bool(decision["cross_label"]):
                    counters["cross_label_swap_proposal_count"] += 1
                else:
                    counters["same_label_swap_proposal_count"] += 1
                if bool(decision["explicit_invoked"]):
                    counters["interface_rule_invocation_count"] += 1
                    counters["active_interface_invocation_count"] += int(active)
                    if math.isfinite(float(decision["same_contact_before"])):
                        same_contact_delta_sum += float(decision["same_contact_after"]) - float(decision["same_contact_before"])
                    if math.isfinite(float(decision["local_goal_before"])):
                        local_goal_delta_sum += float(decision["local_goal_after"]) - float(decision["local_goal_before"])
                if not bool(decision["accepted"]):
                    counters["interface_block_count"] += 1
                    reason = str(decision["block_reason"])
                    interface_block_reasons[reason] += 1
                    counters["wait_count"] += 1
                    _memory_failure(memory, cell_id, f"interface_block:{reason}")
                    continue
                cross_swap = str(labels[actor_index]) != str(labels[target])
                values[actor_index], values[target] = values[target], values[actor_index]
                labels[actor_index], labels[target] = labels[target], labels[actor_index]
                cell_ids[actor_index], cell_ids[target] = cell_ids[target], cell_ids[actor_index]
                counters["swap_count"] += 1
                counters["cross_label_swap_count"] += int(cross_swap)
                counters["same_label_swap_count"] += int(not cross_swap)
                _memory_success(memory, cell_id)
            else:
                counters["invalid_action_count"] += 1
                counters["wait_count"] += 1
                _memory_failure(memory, cell_id, "invalid_swap")
        elif action.action_type == "update_state":
            counters["update_count"] += 1
            state = memory.setdefault(cell_id, {})
            if "ideal_position" in action.state_update:
                updated = action.state_update["ideal_position"]
                state["ideal_position"] = None if updated is None else int(updated)
            state["last_action_type"] = "update_state"
        else:
            counters["wait_count"] += 1
            _memory_wait(memory, cell_id)

    elapsed = time.perf_counter() - started
    final_metrics = sortedness_metrics(values)
    final_agg = aggregation_metrics(labels)
    final_counts = Counter(labels)
    dominance = dominance_metrics(labels, condition.context.policy_ids)
    expected_counts = dict(zip(condition.context.policy_ids, counts, strict=True))
    realized_ratios = tuple(final_counts.get(policy_id, 0) / config.array_size for policy_id in condition.context.policy_ids)
    goal_eval = evaluate_goal_state(
        values=values,
        labels=labels,
        goal_assignments=goal_assignments,
        profile=goal_profile,
        array_size=config.array_size,
        aggregation_delta_percent=float(final_agg["aggregation_delta_percent"]),
    )
    final_state_class = classify_final_state(
        float(final_metrics["inversion_sortedness"]),
        float(final_agg["aggregation_delta_percent"]),
        float(final_agg["largest_block_fraction"]),
        int(final_agg["interface_count"]),
    )
    return {
        "schema": INTERVENTION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "condition_id": condition.condition_id,
        "base_context_id": condition.context.base_context_id,
        "evaluation_stage": condition.context.evaluation_stage,
        "source_research_step_id": condition.context.source_research_step_id,
        "source_condition_id": condition.context.source_condition_id,
        "source_run_condition_id": condition.context.source_run_condition_id,
        "condition_group_id": condition.context.condition_group_id,
        "selection_rank": int(condition.context.selection_rank),
        "selection_reason": condition.context.selection_reason,
        "s13_failure_score": float(condition.context.s13_failure_score),
        "s13_final_state_class_mode": condition.context.s13_final_state_class_mode,
        "s13_mean_final_target_quality": float(condition.context.s13_mean_final_target_quality),
        "s13_mean_goal_conflict_index": float(condition.context.s13_mean_goal_conflict_index),
        "s13_classification_error_rate": float(condition.context.s13_classification_error_rate),
        "s13_mean_quality_abs_error": float(condition.context.s13_mean_quality_abs_error),
        "intervention_spec_id": condition.spec.intervention_spec_id,
        "mechanism_family": condition.spec.mechanism_family,
        "schedule_family": condition.spec.schedule_family,
        "intervention_start_event": int(active_start),
        "intervention_end_event": int(active_end),
        "intervention_duration_events": int(max(0, active_end - active_start)),
        "intervention_active_event_fraction": float(max(0, active_end - active_start) / max(1, config.event_cap)),
        "minimality_score": float(condition.spec.minimality_score),
        "access_stratum": condition.spec.access_stratum,
        "information_access_class": condition.spec.information_access_class,
        "active_interface_rule_id": condition.spec.interface_rule_id,
        "active_governance_mechanism_id": condition.spec.governance_mechanism_id,
        "influence_radius_cells": -1 if condition.spec.influence_radius_cells is None else int(condition.spec.influence_radius_cells),
        "explicit_recognition_used": bool(condition.spec.explicit_recognition_used),
        "global_controller_like": bool(condition.spec.global_controller_like),
        "intended_effect": condition.spec.intended_effect,
        "feature_rationale_json": _json_list(condition.context.feature_rationale),
        "array_size": int(config.array_size),
        "event_cap": int(config.event_cap),
        "events_executed": int(config.event_cap),
        "seed": int(seed),
        "scheduler": config.scheduler,
        "policy_ids_json": _json_list(condition.context.policy_ids),
        "display_names_json": _json_list(condition.context.display_names),
        "source_categories_json": _json_list(condition.context.source_categories),
        "ratio_targets_json": _json_list(condition.context.ratio_targets),
        "goal_profile_id": goal_profile.profile_id,
        "goal_compatibility_class": goal_profile.compatibility_class,
        "goal_assignments_json": _json_object(goal_assignments),
        "arrangement": condition.context.arrangement,
        "value_profile": condition.context.value_profile,
        "perturbation_profile": condition.context.perturbation_profile,
        "panel": condition.context.panel,
        "candidate_reason": condition.context.candidate_reason,
        "expected_counts_json": _json_object(expected_counts),
        "initial_counts_json": _json_object(dict(sorted(initial_counts.items()))),
        "final_counts_json": _json_object(dict(sorted(final_counts.items()))),
        "realized_ratios_json": _json_list(realized_ratios),
        "realized_ratio_max_abs_error": float(max(abs(realized_ratios[idx] - (counts[idx] / config.array_size)) for idx in range(len(counts)))),
        "count_preservation_success": dict(initial_counts) == dict(final_counts) == expected_counts,
        "frozen_indices_json": _json_list(sorted(frozen_indices)),
        "intervention_active_event_count": int(counters.get("intervention_active_event_count", 0)),
        "frozen_block_count": int(counters.get("frozen_block_count", 0)),
        "compare_count": int(counters.get("compare_count", 0)),
        "swap_count": int(counters.get("swap_count", 0)),
        "same_label_swap_count": int(counters.get("same_label_swap_count", 0)),
        "cross_label_swap_count": int(counters.get("cross_label_swap_count", 0)),
        "raw_swap_proposal_count": int(counters.get("raw_swap_proposal_count", 0)),
        "pre_governance_cross_label_proposal_count": int(counters.get("pre_governance_cross_label_proposal_count", 0)),
        "same_label_swap_proposal_count": int(counters.get("same_label_swap_proposal_count", 0)),
        "cross_label_swap_proposal_count": int(counters.get("cross_label_swap_proposal_count", 0)),
        "update_count": int(counters.get("update_count", 0)),
        "wait_count": int(counters.get("wait_count", 0)),
        "work_count": int(counters.get("compare_count", 0) + counters.get("swap_count", 0) + counters.get("update_count", 0)),
        "invalid_action_count": int(counters.get("invalid_action_count", 0)),
        "governance_invocation_count": int(counters.get("governance_invocation_count", 0)),
        "active_governance_invocation_count": int(counters.get("active_governance_invocation_count", 0)),
        "governance_block_count": int(counters.get("governance_block_count", 0)),
        "governance_cross_label_block_count": int(counters.get("governance_cross_label_block_count", 0)),
        "governance_cost_units": int(counters.get("governance_cost_units", 0)),
        "global_controller_invocation_count": int(counters.get("global_controller_invocation_count", 0)),
        "governance_block_fraction": float(counters.get("governance_block_count", 0) / max(1, counters.get("raw_swap_proposal_count", 0))),
        "governance_cost_per_event": float(counters.get("governance_cost_units", 0) / max(1, config.event_cap)),
        "governance_local_goal_delta_per_invocation": float(governance_local_goal_delta_sum / max(1, counters.get("governance_invocation_count", 0))),
        "governance_heterotypic_edge_delta_per_invocation": float(governance_heterotypic_edge_delta_sum / max(1, counters.get("governance_invocation_count", 0))),
        "governance_block_reasons_json": _json_object(governance_block_reasons),
        "interface_rule_invocation_count": int(counters.get("interface_rule_invocation_count", 0)),
        "active_interface_invocation_count": int(counters.get("active_interface_invocation_count", 0)),
        "interface_block_count": int(counters.get("interface_block_count", 0)),
        "interface_block_fraction": float(counters.get("interface_block_count", 0) / max(1, counters.get("cross_label_swap_proposal_count", 0))),
        "cross_label_swap_acceptance_fraction": float(counters.get("cross_label_swap_count", 0) / max(1, counters.get("cross_label_swap_proposal_count", 0))),
        "same_contact_delta_per_invocation": float(same_contact_delta_sum / max(1, counters.get("interface_rule_invocation_count", 0))),
        "local_goal_delta_per_invocation": float(local_goal_delta_sum / max(1, counters.get("interface_rule_invocation_count", 0))),
        "interface_block_reasons_json": _json_object(interface_block_reasons),
        "initial_inversion_count": int(initial_metrics["inversion_count"]),
        "final_inversion_count": int(final_metrics["inversion_count"]),
        "initial_inversion_sortedness": float(initial_metrics["inversion_sortedness"]),
        "final_inversion_sortedness": float(final_metrics["inversion_sortedness"]),
        "inversion_sortedness_delta": float(final_metrics["inversion_sortedness"] - initial_metrics["inversion_sortedness"]),
        "initial_adjacent_sortedness": float(initial_metrics["adjacent_sortedness"]),
        "final_adjacent_sortedness": float(final_metrics["adjacent_sortedness"]),
        "final_is_sorted": bool(final_metrics["is_sorted"]),
        "final_aggregation_left_neighbor_percent": float(final_agg["aggregation_left_neighbor_percent"]),
        "expected_random_left_neighbor_percent": float(final_agg["expected_random_left_neighbor_percent"]),
        "aggregation_delta_percent": float(final_agg["aggregation_delta_percent"]),
        "interface_count": int(final_agg["interface_count"]),
        "contiguous_run_count": int(final_agg["contiguous_run_count"]),
        "largest_block_fraction": float(final_agg["largest_block_fraction"]),
        "label_entropy": float(final_agg["label_entropy"]),
        "leftmost_policy_id": dominance["leftmost_policy_id"],
        "rightmost_policy_id": dominance["rightmost_policy_id"],
        "position_bias_margin": float(dominance["position_bias_margin"]),
        "position_bias_abs": abs(float(dominance["position_bias_margin"])),
        "mean_position_by_policy_json": dominance["mean_position_by_policy_json"],
        "final_state_class": final_state_class,
        "final_target_quality": float(goal_eval["assigned_policy_mean_sortedness"]),
        "goal_conflict_index": float(goal_eval["goal_alignment_gap"]),
        "initial_labels_head_json": _json_list(initial_labels[:20]),
        "initial_labels_tail_json": _json_list(initial_labels[-20:]),
        "final_values_head_json": _json_list(values[:20]),
        "final_values_tail_json": _json_list(values[-20:]),
        "final_labels_head_json": _json_list(labels[:20]),
        "final_labels_tail_json": _json_list(labels[-20:]),
        "elapsed_seconds": float(elapsed),
        **initial_arrangement,
        **goal_eval,
    }


def _simulate_task(task: tuple[InterventionCondition, Mapping[str, Mapping[str, Any]], int, S14Config]) -> dict[str, Any]:
    condition, records_by_id, seed, config = task
    return simulate_intervention_condition(condition, records_by_id, seed=int(seed), config=config)


def _run_conditions(
    conditions: Sequence[InterventionCondition],
    records_by_id: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    config: S14Config,
) -> pd.DataFrame:
    tasks = [(condition, records_by_id, int(seed), config) for condition in conditions for seed in seeds]
    if config.worker_count > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(config.worker_count)) as pool:
            rows = list(pool.map(_simulate_task, tasks))
    else:
        rows = [_simulate_task(task) for task in tasks]
    return pd.DataFrame(rows)


def condition_matrix(conditions: Sequence[InterventionCondition], config: S14Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        expected_counts = dict(zip(condition.context.policy_ids, counts_for_ratios(condition.context.ratio_targets, config.array_size), strict=True))
        rows.append(
            {
                "schema": CONDITION_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "condition_id": condition.condition_id,
                "base_context_id": condition.context.base_context_id,
                "evaluation_stage": condition.context.evaluation_stage,
                "source_research_step_id": condition.context.source_research_step_id,
                "source_condition_id": condition.context.source_condition_id,
                "condition_group_id": condition.context.condition_group_id,
                "selection_rank": int(condition.context.selection_rank),
                "selection_reason": condition.context.selection_reason,
                "intervention_spec_id": condition.spec.intervention_spec_id,
                "mechanism_family": condition.spec.mechanism_family,
                "schedule_family": condition.spec.schedule_family,
                "start_fraction": float(condition.spec.start_fraction),
                "duration_fraction": float(condition.spec.duration_fraction),
                "minimality_score": float(condition.spec.minimality_score),
                "access_stratum": condition.spec.access_stratum,
                "global_controller_like": bool(condition.spec.global_controller_like),
                "information_access_class": condition.spec.information_access_class,
                "active_interface_rule_id": condition.spec.interface_rule_id,
                "active_governance_mechanism_id": condition.spec.governance_mechanism_id,
                "policy_ids_json": _json_list(condition.context.policy_ids),
                "display_names_json": _json_list(condition.context.display_names),
                "source_categories_json": _json_list(condition.context.source_categories),
                "ratio_targets_json": _json_list(condition.context.ratio_targets),
                "expected_counts_json": _json_object(expected_counts),
                "arrangement": condition.context.arrangement,
                "goal_profile_id": condition.context.goal_profile_id,
                "goal_compatibility_class": condition.context.goal_compatibility_class,
                "value_profile": condition.context.value_profile,
                "perturbation_profile": condition.context.perturbation_profile,
            }
        )
    return pd.DataFrame(rows)


def summarize_s14_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["evaluation_stage", "base_context_id", "intervention_spec_id"]
    for _, group in run_df.groupby(group_columns, sort=False):
        first = group.iloc[0]
        class_counts = group["final_state_class"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "evaluation_stage": first["evaluation_stage"],
                "base_context_id": first["base_context_id"],
                "intervention_spec_id": first["intervention_spec_id"],
                "mechanism_family": first["mechanism_family"],
                "schedule_family": first["schedule_family"],
                "access_stratum": first["access_stratum"],
                "global_controller_like": bool(first["global_controller_like"]),
                "minimality_score": float(first["minimality_score"]),
                "intervention_active_event_fraction": float(first["intervention_active_event_fraction"]),
                "source_research_step_id": first["source_research_step_id"],
                "display_names_json": first["display_names_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "arrangement": first["arrangement"],
                "goal_profile_id": first["goal_profile_id"],
                "goal_compatibility_class": first["goal_compatibility_class"],
                "s13_failure_score": float(first["s13_failure_score"]),
                "seed_count": int(group["seed"].nunique()),
                "run_count": int(len(group)),
                "mean_final_target_quality": float(group["final_target_quality"].mean()),
                "mean_reference_increasing_score": float(group["reference_increasing_score"].mean()),
                "mean_goal_conflict_index": float(group["goal_conflict_index"].mean()),
                "mean_aggregation_delta_percent": float(group["aggregation_delta_percent"].mean()),
                "mean_interface_count": float(group["interface_count"].mean()),
                "mean_largest_block_fraction": float(group["largest_block_fraction"].mean()),
                "mean_position_bias_abs": float(group["position_bias_abs"].mean()),
                "mean_work_count": float(group["work_count"].mean()),
                "mean_swap_count": float(group["swap_count"].mean()),
                "mean_governance_invocation_count": float(group["governance_invocation_count"].mean()),
                "mean_governance_block_fraction": float(group["governance_block_fraction"].mean()),
                "mean_governance_cost_per_event": float(group["governance_cost_per_event"].mean()),
                "mean_global_controller_invocation_count": float(group["global_controller_invocation_count"].mean()),
                "mean_interface_rule_invocation_count": float(group["interface_rule_invocation_count"].mean()),
                "mean_interface_block_fraction": float(group["interface_block_fraction"].mean()),
                "mean_cross_label_swap_acceptance_fraction": float(group["cross_label_swap_acceptance_fraction"].mean()),
                "invalid_action_count": int(group["invalid_action_count"].sum()),
                "count_preservation_success": bool(group["count_preservation_success"].all()),
                "final_state_class_mode": _mode(group["final_state_class"]),
                "goal_state_class_mode": _mode(group["goal_state_class"]),
                "final_state_class_counts_json": _json_object(class_counts),
            }
        )
    return pd.DataFrame(rows)


def compare_to_baseline(summary: pd.DataFrame, config: S14Config | None = None) -> pd.DataFrame:
    config = config or S14Config()
    if summary.empty:
        return pd.DataFrame()
    metric_columns = [
        "mean_final_target_quality",
        "mean_reference_increasing_score",
        "mean_goal_conflict_index",
        "mean_aggregation_delta_percent",
        "mean_interface_count",
        "mean_largest_block_fraction",
        "mean_position_bias_abs",
        "mean_work_count",
        "mean_swap_count",
    ]
    baseline = summary[summary["intervention_spec_id"] == BASELINE_SPEC_ID].copy()
    baseline = baseline[["evaluation_stage", "base_context_id", *metric_columns, "final_state_class_mode"]].rename(
        columns={column: f"baseline_{column}" for column in [*metric_columns, "final_state_class_mode"]}
    )
    comparison = summary.merge(baseline, on=["evaluation_stage", "base_context_id"], how="left")
    for column in metric_columns:
        comparison[f"delta_{column.removeprefix('mean_')}"] = comparison[column] - comparison[f"baseline_{column}"]
    comparison["schema"] = COMPARISON_SCHEMA
    comparison["conflict_reduction"] = comparison["baseline_mean_goal_conflict_index"] - comparison["mean_goal_conflict_index"]
    comparison["work_increase_fraction"] = comparison["delta_work_count"] / comparison["baseline_mean_work_count"].replace(0, np.nan)
    comparison["work_increase_fraction"] = comparison["work_increase_fraction"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    comparison["unintended_consequence_score"] = (
        comparison["delta_final_target_quality"].clip(upper=0.0).abs()
        + comparison["conflict_reduction"].clip(upper=0.0).abs() * 0.25
        + comparison["work_increase_fraction"].clip(lower=0.0) * 0.02
        + comparison["mean_governance_block_fraction"] * 0.02
        + comparison["mean_interface_block_fraction"] * 0.02
        + comparison["global_controller_like"].astype(bool).astype(float) * 0.01
    )
    comparison["rescue_effect_score"] = (
        comparison["delta_final_target_quality"]
        + 0.25 * comparison["conflict_reduction"]
        - comparison["mean_governance_block_fraction"] * 0.02
        - comparison["mean_interface_block_fraction"] * 0.02
        - comparison["mean_governance_cost_per_event"] * 0.001
    )
    comparison["minimality_adjusted_rescue_score"] = comparison["rescue_effect_score"] - config.minimality_penalty_weight * comparison["minimality_score"]
    comparison["target_quality_rescued"] = comparison["delta_final_target_quality"] >= config.rescue_delta_threshold
    comparison["conflict_reduced"] = comparison["conflict_reduction"] >= config.rescue_delta_threshold
    comparison["state_class_changed_vs_baseline"] = comparison["final_state_class_mode"].astype(str) != comparison["baseline_final_state_class_mode"].astype(str)
    return comparison


def rank_interventions(comparison: pd.DataFrame, config: S14Config | None = None) -> pd.DataFrame:
    config = config or S14Config()
    if comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (stage, spec_id), group in comparison.groupby(["evaluation_stage", "intervention_spec_id"], sort=False):
        first = group.iloc[0]
        is_baseline = str(spec_id) == BASELINE_SPEC_ID
        rows.append(
            {
                "schema": RANKING_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "evaluation_stage": stage,
                "intervention_spec_id": spec_id,
                "mechanism_family": first["mechanism_family"],
                "schedule_family": first["schedule_family"],
                "access_stratum": first["access_stratum"],
                "global_controller_like": bool(first["global_controller_like"]),
                "minimality_score": float(first["minimality_score"]),
                "condition_count": int(group["base_context_id"].nunique()),
                "mean_delta_final_target_quality": float(group["delta_final_target_quality"].mean()),
                "mean_conflict_reduction": float(group["conflict_reduction"].mean()),
                "mean_delta_aggregation_delta_percent": float(group["delta_aggregation_delta_percent"].mean()),
                "mean_delta_largest_block_fraction": float(group["delta_largest_block_fraction"].mean()),
                "mean_governance_block_fraction": float(group["mean_governance_block_fraction"].mean()),
                "mean_interface_block_fraction": float(group["mean_interface_block_fraction"].mean()),
                "mean_governance_cost_per_event": float(group["mean_governance_cost_per_event"].mean()),
                "mean_unintended_consequence_score": float(group["unintended_consequence_score"].mean()),
                "mean_rescue_effect_score": float(group["rescue_effect_score"].mean()),
                "mean_minimality_adjusted_rescue_score": float(group["minimality_adjusted_rescue_score"].mean()),
                "target_quality_rescue_rate": 0.0 if is_baseline else float(group["target_quality_rescued"].mean()),
                "conflict_reduction_rate": 0.0 if is_baseline else float(group["conflict_reduced"].mean()),
                "state_class_change_rate": 0.0 if is_baseline else float(group["state_class_changed_vs_baseline"].mean()),
                "supportive_rescue_by_threshold": bool(
                    (not is_baseline)
                    and group["rescue_effect_score"].mean() >= config.rescue_delta_threshold
                    and group["target_quality_rescued"].mean() >= 0.50
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["evaluation_stage", "global_controller_like", "mean_minimality_adjusted_rescue_score", "minimality_score"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )


def choose_validation_specs(search_rankings: pd.DataFrame, config: S14Config) -> tuple[str, ...]:
    if search_rankings.empty:
        return (BASELINE_SPEC_ID,)
    non_baseline = search_rankings[
        (search_rankings["evaluation_stage"] == SEARCH_STAGE) & (search_rankings["intervention_spec_id"] != BASELINE_SPEC_ID)
    ].copy()
    selected = [BASELINE_SPEC_ID]
    local = non_baseline[~non_baseline["global_controller_like"].astype(bool)].sort_values(
        ["mean_minimality_adjusted_rescue_score", "minimality_score"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected.extend(local["intervention_spec_id"].head(config.validation_local_intervention_count).astype(str).tolist())
    global_like = non_baseline[non_baseline["global_controller_like"].astype(bool)].sort_values(
        ["mean_minimality_adjusted_rescue_score", "minimality_score"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected.extend(global_like["intervention_spec_id"].head(1).astype(str).tolist())
    return tuple(dict.fromkeys(selected))


def run_s14_search(
    records: Sequence[Mapping[str, Any]],
    s13_matrix: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_screen: pd.DataFrame,
    config: S14Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or S14Config()
    selected_contexts = select_s14_failure_contexts(s13_matrix, predictions, feature_screen, records, config)
    records_by_id = {str(record["algotypeId"]): dict(record) for record in records}
    search_conditions = build_s14_conditions(selected_contexts, config.intervention_specs, SEARCH_STAGE)
    missing = sorted({policy_id for condition in search_conditions for policy_id in condition.context.policy_ids} - set(records_by_id))
    if missing:
        raise ValueError(f"S14 selected search policies missing from S01 library: {missing}")
    search_runs = _run_conditions(search_conditions, records_by_id, config.search_seeds, config)
    search_summary = summarize_s14_runs(search_runs)
    search_comparison = compare_to_baseline(search_summary, config)
    search_rankings = rank_interventions(search_comparison, config)
    validation_spec_ids = choose_validation_specs(search_rankings, config)
    validation_specs = tuple(spec for spec in config.intervention_specs if spec.intervention_spec_id in validation_spec_ids)
    validation_conditions = build_s14_conditions(selected_contexts, validation_specs, VALIDATION_STAGE)
    missing = sorted({policy_id for condition in validation_conditions for policy_id in condition.context.policy_ids} - set(records_by_id))
    if missing:
        raise ValueError(f"S14 selected validation policies missing from S01 library: {missing}")
    validation_runs = _run_conditions(validation_conditions, records_by_id, config.validation_seeds, config)
    run_df = pd.concat([search_runs, validation_runs], ignore_index=True)
    all_conditions = [*search_conditions, *validation_conditions]
    condition_df = condition_matrix(all_conditions, config)
    specs = intervention_spec_table(config, feature_screen)
    summary = summarize_s14_runs(run_df)
    comparison = compare_to_baseline(summary, config)
    rankings = rank_interventions(comparison, config)
    selected_contexts = selected_contexts.copy()
    selected_contexts["validation_spec_ids_json"] = _json_list(validation_spec_ids)
    return run_df, condition_df, selected_contexts, specs, summary, comparison, rankings


def outcome_classification(rankings: pd.DataFrame, validation_passed: bool) -> str:
    if not validation_passed or rankings.empty:
        return "null"
    validation = rankings[rankings["evaluation_stage"] == VALIDATION_STAGE]
    local = validation[(validation["intervention_spec_id"] != BASELINE_SPEC_ID) & (~validation["global_controller_like"].astype(bool))]
    global_like = validation[validation["global_controller_like"].astype(bool)]
    if not local.empty and bool(local["supportive_rescue_by_threshold"].any()):
        return "supportive"
    if not global_like.empty and bool(global_like["supportive_rescue_by_threshold"].any()):
        return "constraining/contradictory"
    return "null"


def validate_s14_outputs(
    run_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    selected_contexts: pd.DataFrame,
    specs: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    rankings: pd.DataFrame,
    config: S14Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    search_seeds = set(int(seed) for seed in config.search_seeds)
    validation_seeds = set(int(seed) for seed in config.validation_seeds)
    previous_seeds = set(int(seed) for seed in DEFAULT_SEEDS)
    search_contexts = set(selected_contexts[selected_contexts["evaluation_stage"] == SEARCH_STAGE]["base_context_id"].astype(str))
    validation_contexts = set(selected_contexts[selected_contexts["evaluation_stage"] == VALIDATION_STAGE]["base_context_id"].astype(str))
    seed_sets = (
        run_df.groupby(["evaluation_stage", "base_context_id", "intervention_spec_id"])["seed"].agg(lambda values: set(int(value) for value in values)).tolist()
        if not run_df.empty
        else []
    )
    baseline_groups = (
        run_df.groupby(["evaluation_stage", "base_context_id"])["intervention_spec_id"].agg(lambda values: set(map(str, values))).tolist()
        if not run_df.empty
        else []
    )
    finite_columns = [
        "delta_final_target_quality",
        "conflict_reduction",
        "rescue_effect_score",
        "minimality_adjusted_rescue_score",
        "unintended_consequence_score",
    ]
    finite_comparison = bool(
        not comparison.empty
        and set(finite_columns).issubset(comparison.columns)
        and comparison[finite_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().all().all()
    )
    checks = [
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "selected_contexts_from_s13_failure_table",
            "success": bool(
                not selected_contexts.empty
                and selected_contexts["selection_reason"].astype(str).str.contains("s13_failure_context").all()
                and selected_contexts["s13_failure_score"].astype(float).gt(0).all()
            ),
            "observed": f"contexts={len(selected_contexts)}",
            "expected": "S14 contexts are selected from S13 failure/error contexts",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "s13_feature_associations_documented",
            "success": bool(not specs.empty and specs["rationale_features_json"].astype(str).str.len().gt(2).all()),
            "observed": sorted(specs["intervention_spec_id"].astype(str).tolist()) if not specs.empty else [],
            "expected": "each intervention spec records S13 feature-association rationale",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "prospective_heldout_seed_sets",
            "success": bool(
                search_seeds.isdisjoint(validation_seeds)
                and search_seeds.isdisjoint(previous_seeds)
                and validation_seeds.isdisjoint(previous_seeds)
                and seed_sets
                and all(seed_set == search_seeds or seed_set == validation_seeds for seed_set in seed_sets)
            ),
            "observed": f"search={sorted(search_seeds)} validation={sorted(validation_seeds)} previous={sorted(previous_seeds)}",
            "expected": "search and validation use new, mutually disjoint held-out seeds",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "heldout_conditions_separate_from_search_conditions",
            "success": bool(search_contexts and validation_contexts and search_contexts.isdisjoint(validation_contexts)),
            "observed": f"search_contexts={len(search_contexts)} validation_contexts={len(validation_contexts)} overlap={len(search_contexts & validation_contexts)}",
            "expected": "validation contexts are not used in search ranking",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "behavior_baseline_for_each_stage_context",
            "success": bool(baseline_groups and all(BASELINE_SPEC_ID in item for item in baseline_groups)),
            "observed": f"groups={len(baseline_groups)}",
            "expected": "every search and validation base context has a no-intervention baseline",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "minimality_documented_for_interventions",
            "success": bool(
                not specs.empty
                and {"minimality_score", "duration_fraction", "active_event_fraction", "access_stratum"}.issubset(specs.columns)
                and specs["minimality_score"].notna().all()
            ),
            "observed": f"spec_rows={len(specs)}",
            "expected": "intervention specs document active duration, access class, and minimality score",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "local_and_global_access_strata_explicit",
            "success": bool(
                not run_df.empty
                and "behavior_only" in set(run_df["access_stratum"].astype(str))
                and any(not value for value in run_df["global_controller_like"].astype(bool).unique())
                and any(value for value in run_df["global_controller_like"].astype(bool).unique())
            ),
            "observed": sorted(run_df["access_stratum"].astype(str).unique().tolist()) if not run_df.empty else [],
            "expected": "behavior-only, finite/local, and global-controller-like rows are labeled separately",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "realized_ratios_and_counts_preserved",
            "success": bool(
                not run_df.empty and run_df["count_preservation_success"].astype(bool).all() and run_df["realized_ratio_max_abs_error"].le(1e-12).all()
            ),
            "observed": f"max_ratio_error={run_df['realized_ratio_max_abs_error'].max() if not run_df.empty else math.nan}",
            "expected": "policy counts and realized ratios are preserved in every S14 run",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "no_invalid_actions",
            "success": bool(not run_df.empty and int(run_df["invalid_action_count"].sum()) == 0),
            "observed": str(int(run_df["invalid_action_count"].sum())) if not run_df.empty else "missing rows",
            "expected": "zero out-of-bounds actions",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "unintended_consequences_reported",
            "success": bool(finite_comparison and "mean_unintended_consequence_score" in rankings.columns),
            "observed": f"comparison_rows={len(comparison)} rankings_rows={len(rankings)}",
            "expected": "baseline comparison reports target, conflict, work, block, cost, and unintended-consequence metrics",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "figure_written",
            "success": bool(figure_written),
            "observed": str(bool(figure_written)),
            "expected": "intervention rescue figure exists and is non-empty",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests_success),
            "observed": str(bool(unit_tests_success)),
            "expected": "focused S14 unit tests pass",
        },
    ]
    return pd.DataFrame(checks)
