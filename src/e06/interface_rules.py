"""E06 S08 explicit interface-rule intervention sweeps."""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.coarse_sweep import actor_schedule, initial_values, sortedness_metrics
from src.e06.goal_compatibility import (
    DEFAULT_GOAL_PROFILES,
    GoalProfile,
    evaluate_goal_state,
    goal_assignments_for_profile,
    rank_values,
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


STEP_ID = "S08"
INTERVENTION_SCHEMA = "eidosoma.e06.s08_interface_rule_intervention.v1"
SUMMARY_SCHEMA = "eidosoma.e06.s08_interface_rule_summary.v1"
COMPARISON_SCHEMA = "eidosoma.e06.s08_interface_rule_baseline_comparison.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s08_validation.v1"
SAME_INCREASING_PROFILE = GoalProfile(
    profile_id="same_increasing",
    compatibility_class="same_goal",
    policy_goal_names=("increasing", "increasing"),
    shared_goal_name="increasing",
    relevant_goal_names=("increasing",),
    description="S08 replay default: both policies use standard increasing-order behavior.",
)
S08_OPPOSITE_PROFILE = GoalProfile(
    profile_id="s06_opposite_direction",
    compatibility_class="opposite_goal",
    policy_goal_names=("increasing", "decreasing"),
    shared_goal_name=None,
    relevant_goal_names=("increasing", "decreasing"),
    description="S08 replay of S06 opposite-goal conditions.",
)


@dataclass(frozen=True)
class InterfaceRule:
    """One interface intervention mechanism."""

    rule_id: str
    mechanism_family: str
    explicit_recognition_used: bool
    recognition_access_class: str
    permeability: float
    description: str


DEFAULT_INTERFACE_RULES = (
    InterfaceRule(
        rule_id="behavior_only",
        mechanism_family="behavior_only_baseline",
        explicit_recognition_used=False,
        recognition_access_class="none",
        permeability=1.0,
        description="Original local behavior only; no explicit access to same/different Algotype identity.",
    ),
    InterfaceRule(
        rule_id="adhesion_like_preference",
        mechanism_family="adhesion_like_preference",
        explicit_recognition_used=True,
        recognition_access_class="local_label_identity",
        permeability=0.20,
        description="Label-aware preference for swaps that preserve or increase same-Algotype local contacts.",
    ),
    InterfaceRule(
        rule_id="heterotypic_repulsion",
        mechanism_family="repulsion",
        explicit_recognition_used=True,
        recognition_access_class="local_label_identity",
        permeability=0.50,
        description="Label-aware repulsion that favors swaps reducing unlike-neighbor contacts.",
    ),
    InterfaceRule(
        rule_id="permeability_barrier",
        mechanism_family="permeability",
        explicit_recognition_used=True,
        recognition_access_class="local_label_identity",
        permeability=0.15,
        description="Semipermeable heterotypic interface; cross-Algotype swaps pass only at a low local probability.",
    ),
    InterfaceRule(
        rule_id="local_recognition_gate",
        mechanism_family="local_recognition",
        explicit_recognition_used=True,
        recognition_access_class="local_label_identity_and_goal_proxy",
        permeability=0.0,
        description="Explicit local-recognition gate allowing heterotypic swaps only when local label cohesion and goal satisfaction do not degrade.",
    ),
)


@dataclass(frozen=True)
class S08Config:
    """Configuration for bounded S08 interface-rule intervention tests."""

    array_size: int = 100
    event_cap: int = 4_000
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    max_base_conditions: int = 12
    metric_extreme_candidates_per_metric: int = 8
    interface_rules: tuple[InterfaceRule, ...] = DEFAULT_INTERFACE_RULES
    scheduler: str = "cyclic_scan_seed_offset"
    explicit_dominance_delta_threshold: float = 10.0
    explicit_dominance_block_threshold: float = 0.25


@dataclass(frozen=True)
class InterfaceBaseCondition:
    """One selected S07 condition prototype before intervention expansion."""

    base_condition_id: str
    source_s07_run_id: str
    source_research_step_id: str
    source_condition_id: str
    selection_reason: str
    s07_label: str
    s07_classification_mode: str
    panel: str
    candidate_reason: str
    policy_ids: tuple[str, str]
    display_names: tuple[str, str]
    source_categories: tuple[str, str]
    ratio_targets: tuple[float, float]
    original_arrangement: str
    arrangement: str
    goal_profile_id: str
    goal_compatibility_class: str
    value_profile: str
    perturbation_profile: str
    source_final_target_quality: float
    source_goal_conflict_index: float
    source_aggregation_delta_percent: float
    source_largest_block_fraction: float
    source_position_bias_abs: float
    source_mosaic_continuum_score: float
    source_conflict_continuum_score: float


@dataclass(frozen=True)
class InterfaceCondition:
    """One S08 executable intervention condition before seed expansion."""

    condition_id: str
    base: InterfaceBaseCondition
    rule: InterfaceRule


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _normalize_arrangement(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "random_permutation"
    arrangement = str(value or "").strip()
    if arrangement in {"", "random_permutation_labels"}:
        return "random_permutation"
    return arrangement


def _normalize_value_profile(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "random_permutation"
    text = str(value or "").strip()
    if text.lower() == "nan":
        return "random_permutation"
    return text if text else "random_permutation"


def _normalize_perturbation_profile(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "none"
    text = str(value or "").strip()
    if text.lower() == "nan":
        return "none"
    return text if text else "none"


def _condition_key(row: Mapping[str, Any]) -> str:
    return _json_object(
        {
            "policyIds": _parse_json_list(row.get("policy_ids_json")),
            "ratios": [round(float(value), 6) for value in _parse_json_list(row.get("ratio_targets_json"))],
            "arrangement": _normalize_arrangement(row.get("arrangement")),
            "goalProfileId": str(row.get("goal_profile_id", "same_increasing") or "same_increasing"),
            "valueProfile": _normalize_value_profile(row.get("value_profile")),
            "perturbationProfile": _normalize_perturbation_profile(row.get("perturbation_profile")),
        }
    )


def _base_condition_id(row: Mapping[str, Any]) -> str:
    return f"s08_base_{stable_hash(_condition_key(row))[:16]}"


def _condition_id(base_condition_id: str, rule_id: str) -> str:
    return f"s08_{stable_hash({'baseConditionId': base_condition_id, 'ruleId': rule_id})[:16]}"


def _reason_join(reasons: Sequence[str]) -> str:
    seen: list[str] = []
    for reason in reasons:
        if reason and reason not in seen:
            seen.append(str(reason))
    return ",".join(seen)


def _pairwise_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "policy_ids_json" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["policy_ids_json"].map(lambda text: len(_parse_json_list(text)) == 2)].copy()


def select_s08_base_conditions(
    mosaic: pd.DataFrame,
    exemplars: pd.DataFrame,
    config: S08Config,
) -> pd.DataFrame:
    """Select bounded pairwise S07 exemplar and metric-extreme conditions."""

    if mosaic.empty:
        raise ValueError("S07 mosaic class table is empty")
    required = {
        "run_id",
        "source_research_step_id",
        "source_condition_id",
        "policy_ids_json",
        "display_names_json",
        "source_categories_json",
        "ratio_targets_json",
        "arrangement",
        "goal_profile_id",
        "goal_compatibility_class",
        "s07_label",
        "s07_classification_mode",
        "final_target_quality",
        "goal_conflict_index",
        "aggregation_delta_percent",
        "largest_block_fraction",
        "position_bias_abs",
        "mosaic_continuum_score",
        "conflict_continuum_score",
    }
    missing = required - set(mosaic.columns)
    if missing:
        raise ValueError(f"S07 mosaic table missing columns: {sorted(missing)}")

    rows_by_key: dict[str, dict[str, Any]] = {}
    reasons_by_key: dict[str, list[str]] = {}

    def add_rows(rows: pd.DataFrame, reason: str) -> None:
        for raw in rows.to_dict(orient="records"):
            policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
            ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
            if len(policy_ids) != 2 or len(ratios) != 2:
                continue
            key = _condition_key(raw)
            if key not in rows_by_key:
                rows_by_key[key] = dict(raw)
                reasons_by_key[key] = []
            reasons_by_key[key].append(reason)
            if len(rows_by_key) >= config.max_base_conditions:
                return

    exemplar_pairs = _pairwise_rows(exemplars)
    if not exemplar_pairs.empty:
        exemplar_pairs = exemplar_pairs.sort_values(["s07_label", "source_research_step_id", "run_id"], kind="mergesort")
        for label, group in exemplar_pairs.groupby("s07_label", sort=True):
            add_rows(group.head(1), f"s07_exemplar:{label}")
            if len(rows_by_key) >= config.max_base_conditions:
                break

    pairwise = _pairwise_rows(mosaic)
    extreme_specs = [
        ("metric_extreme:high_mosaic_continuum", "mosaic_continuum_score", False),
        ("metric_extreme:high_conflict_continuum", "conflict_continuum_score", False),
        ("metric_extreme:high_aggregation", "aggregation_delta_percent", False),
        ("metric_extreme:high_goal_conflict", "goal_conflict_index", False),
        ("metric_extreme:high_position_bias", "position_bias_abs", False),
        ("metric_extreme:high_largest_block", "largest_block_fraction", False),
        ("metric_extreme:low_target_quality", "final_target_quality", True),
    ]
    for reason, metric, ascending in extreme_specs:
        if len(rows_by_key) >= config.max_base_conditions:
            break
        ranked = pairwise.copy()
        ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
        ranked = ranked.dropna(subset=[metric]).sort_values(metric, ascending=ascending, kind="mergesort")
        add_rows(ranked.head(config.metric_extreme_candidates_per_metric), reason)

    condition_rows: list[dict[str, Any]] = []
    for rank, (key, raw) in enumerate(rows_by_key.items(), start=1):
        policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
        display_names = tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
        categories = tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
        ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
        condition_rows.append(
            {
                "base_condition_id": _base_condition_id(raw),
                "selection_rank": int(rank),
                "selection_reason": _reason_join(reasons_by_key[key]),
                "source_s07_run_id": str(raw["run_id"]),
                "source_research_step_id": str(raw["source_research_step_id"]),
                "source_condition_id": str(raw["source_condition_id"]),
                "s07_label": str(raw["s07_label"]),
                "s07_classification_mode": str(raw["s07_classification_mode"]),
                "panel": str(raw.get("panel", "")),
                "candidate_reason": str(raw.get("candidate_reason", "")),
                "policy_ids_json": _json_list(policy_ids),
                "display_names_json": _json_list(display_names),
                "source_categories_json": _json_list(categories),
                "ratio_targets_json": _json_list(ratios),
                "original_arrangement": str(raw.get("arrangement", "")),
                "arrangement": _normalize_arrangement(raw.get("arrangement")),
                "goal_profile_id": str(raw.get("goal_profile_id", "same_increasing") or "same_increasing"),
                "goal_compatibility_class": str(raw.get("goal_compatibility_class", "same_goal") or "same_goal"),
                "value_profile": _normalize_value_profile(raw.get("value_profile")),
                "perturbation_profile": _normalize_perturbation_profile(raw.get("perturbation_profile")),
                "source_final_target_quality": float(raw["final_target_quality"]),
                "source_goal_conflict_index": float(raw["goal_conflict_index"]),
                "source_aggregation_delta_percent": float(raw["aggregation_delta_percent"]),
                "source_largest_block_fraction": float(raw["largest_block_fraction"]),
                "source_position_bias_abs": float(raw["position_bias_abs"]),
                "source_mosaic_continuum_score": float(raw["mosaic_continuum_score"]),
                "source_conflict_continuum_score": float(raw["conflict_continuum_score"]),
            }
        )
    selected = pd.DataFrame(condition_rows)
    if selected.empty:
        raise ValueError("S08 could not select pairwise S07 base conditions")
    return selected


def build_s08_conditions(selected: pd.DataFrame, config: S08Config) -> list[InterfaceCondition]:
    conditions: list[InterfaceCondition] = []
    for raw in selected.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
        display_names = tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
        categories = tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
        ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
        if len(policy_ids) != 2 or len(ratios) != 2:
            continue
        base = InterfaceBaseCondition(
            base_condition_id=str(raw["base_condition_id"]),
            source_s07_run_id=str(raw["source_s07_run_id"]),
            source_research_step_id=str(raw["source_research_step_id"]),
            source_condition_id=str(raw["source_condition_id"]),
            selection_reason=str(raw["selection_reason"]),
            s07_label=str(raw["s07_label"]),
            s07_classification_mode=str(raw["s07_classification_mode"]),
            panel=str(raw["panel"]),
            candidate_reason=str(raw["candidate_reason"]),
            policy_ids=(policy_ids[0], policy_ids[1]),
            display_names=(display_names[0], display_names[1]),
            source_categories=(categories[0], categories[1]),
            ratio_targets=(ratios[0], ratios[1]),
            original_arrangement=str(raw["original_arrangement"]),
            arrangement=str(raw["arrangement"]),
            goal_profile_id=str(raw["goal_profile_id"]),
            goal_compatibility_class=str(raw["goal_compatibility_class"]),
            value_profile=str(raw["value_profile"]),
            perturbation_profile=str(raw["perturbation_profile"]),
            source_final_target_quality=float(raw["source_final_target_quality"]),
            source_goal_conflict_index=float(raw["source_goal_conflict_index"]),
            source_aggregation_delta_percent=float(raw["source_aggregation_delta_percent"]),
            source_largest_block_fraction=float(raw["source_largest_block_fraction"]),
            source_position_bias_abs=float(raw["source_position_bias_abs"]),
            source_mosaic_continuum_score=float(raw["source_mosaic_continuum_score"]),
            source_conflict_continuum_score=float(raw["source_conflict_continuum_score"]),
        )
        for rule in config.interface_rules:
            conditions.append(InterfaceCondition(condition_id=_condition_id(base.base_condition_id, rule.rule_id), base=base, rule=rule))
    return conditions


def goal_profile_for_id(goal_profile_id: str) -> GoalProfile:
    profile_id = str(goal_profile_id or "same_increasing")
    if profile_id == "s06_opposite_direction":
        return S08_OPPOSITE_PROFILE
    for profile in DEFAULT_GOAL_PROFILES:
        if profile.profile_id == profile_id:
            return profile
    if profile_id == "same_increasing":
        return SAME_INCREASING_PROFILE
    return SAME_INCREASING_PROFILE


def initial_values_for_profile(array_size: int, seed: int, value_profile: str) -> tuple[int, ...]:
    profile = str(value_profile or "random_permutation")
    if profile == "random_permutation":
        return tuple(int(value) for value in initial_values(array_size, seed))
    if profile == "reverse_sorted":
        return tuple(range(array_size, 0, -1))
    raise ValueError(f"unknown S08 value profile: {value_profile}")


def frozen_indices_for_profile(array_size: int, perturbation_profile: str) -> frozenset[int]:
    profile = str(perturbation_profile or "none")
    if profile == "none":
        return frozenset()
    if profile == "frozen_edges_2":
        return frozenset({0, max(0, array_size - 1)})
    raise ValueError(f"unknown S08 perturbation profile: {perturbation_profile}")


def _affected_edges(n: int, actor_index: int, target_index: int) -> tuple[int, ...]:
    edges: set[int] = set()
    for idx in (int(actor_index), int(target_index)):
        for edge in (idx - 1, idx):
            if 0 <= edge < n - 1:
                edges.add(edge)
    return tuple(sorted(edges))


def _same_edge_count(labels: Sequence[str], edges: Sequence[int]) -> int:
    return sum(1 for edge in edges if str(labels[edge]) == str(labels[edge + 1]))


def _local_goal_satisfaction(
    values: Sequence[int],
    labels: Sequence[str],
    edges: Sequence[int],
    goal_assignments: Mapping[str, str],
    array_size: int,
) -> float:
    if not edges:
        return 1.0
    satisfied = 0
    total = 0
    for edge in edges:
        left_label = str(labels[edge])
        right_label = str(labels[edge + 1])
        for goal_name in (goal_assignments[left_label], goal_assignments[right_label]):
            ranks = rank_values((values[edge], values[edge + 1]), goal_name, array_size)
            satisfied += int(ranks[0] <= ranks[1])
            total += 1
    return float(satisfied / max(1, total))


def interface_decision(
    rule: InterfaceRule,
    *,
    values: Sequence[int],
    labels: Sequence[str],
    actor_index: int,
    target_index: int,
    goal_assignments: Mapping[str, str],
    array_size: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Return whether a proposed valid swap passes the S08 interface rule."""

    actor_label = str(labels[actor_index])
    target_label = str(labels[target_index])
    cross_label = actor_label != target_label
    if not rule.explicit_recognition_used:
        return {
            "accepted": True,
            "cross_label": cross_label,
            "explicit_invoked": False,
            "block_reason": "",
            "same_contact_before": math.nan,
            "same_contact_after": math.nan,
            "local_goal_before": math.nan,
            "local_goal_after": math.nan,
        }
    if not cross_label:
        return {
            "accepted": True,
            "cross_label": False,
            "explicit_invoked": False,
            "block_reason": "",
            "same_contact_before": math.nan,
            "same_contact_after": math.nan,
            "local_goal_before": math.nan,
            "local_goal_after": math.nan,
        }

    edges = _affected_edges(len(labels), actor_index, target_index)
    before_same = _same_edge_count(labels, edges)
    after_labels = list(labels)
    after_labels[actor_index], after_labels[target_index] = after_labels[target_index], after_labels[actor_index]
    after_same = _same_edge_count(after_labels, edges)
    before_hetero = len(edges) - before_same
    after_hetero = len(edges) - after_same
    after_values = list(values)
    after_values[actor_index], after_values[target_index] = after_values[target_index], after_values[actor_index]
    before_goal = _local_goal_satisfaction(values, labels, edges, goal_assignments, array_size)
    after_goal = _local_goal_satisfaction(after_values, after_labels, edges, goal_assignments, array_size)

    accepted = True
    block_reason = ""
    if rule.rule_id == "adhesion_like_preference":
        accepted = after_same >= before_same or rng.random() < rule.permeability
        block_reason = "" if accepted else "adhesion_same_contact_loss"
    elif rule.rule_id == "heterotypic_repulsion":
        accepted = after_hetero < before_hetero or (after_hetero == before_hetero and rng.random() < rule.permeability)
        block_reason = "" if accepted else "repulsion_heterotypic_contact_not_reduced"
    elif rule.rule_id == "permeability_barrier":
        accepted = rng.random() < rule.permeability
        block_reason = "" if accepted else "permeability_cross_label_barrier"
    elif rule.rule_id == "local_recognition_gate":
        accepted = (after_goal > before_goal + 1e-12 and after_same >= before_same - 1) or (
            after_goal >= before_goal - 1e-12 and after_same >= before_same
        )
        block_reason = "" if accepted else "local_recognition_goal_or_cohesion_loss"
    else:
        raise ValueError(f"unknown S08 interface rule: {rule.rule_id}")
    return {
        "accepted": bool(accepted),
        "cross_label": True,
        "explicit_invoked": True,
        "block_reason": block_reason,
        "same_contact_before": float(before_same),
        "same_contact_after": float(after_same),
        "local_goal_before": float(before_goal),
        "local_goal_after": float(after_goal),
    }


def _paper_scope(rule: InterfaceRule) -> str:
    if not rule.explicit_recognition_used:
        return "original_behavior_only_no_explicit_recognition"
    return "explicit_interface_intervention_beyond_original_no_recognition_claim"


def simulate_interface_condition(
    condition: InterfaceCondition,
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: S08Config,
) -> dict[str, Any]:
    counts = counts_for_ratios(condition.base.ratio_targets, config.array_size)
    values = list(initial_values_for_profile(config.array_size, seed, condition.base.value_profile))
    labels = list(
        labels_for_arrangement(
            condition.base.policy_ids,
            counts,
            seed=seed,
            condition_id=condition.base.base_condition_id,
            arrangement=condition.base.arrangement,
        )
    )
    cell_ids = list(range(config.array_size))
    memory: dict[int, dict[str, Any]] = {}
    runtime = MixtureRuntime([records_by_id[policy_id] for policy_id in condition.base.policy_ids], seed)
    schedule = actor_schedule(config.array_size, config.event_cap, seed)
    rng = random.Random(seed ^ int(stable_hash({"s08": condition.condition_id})[:12], 16))
    goal_profile = goal_profile_for_id(condition.base.goal_profile_id)
    goal_assignments = goal_assignments_for_profile(condition.base.policy_ids, goal_profile)
    frozen_indices = frozen_indices_for_profile(config.array_size, condition.base.perturbation_profile)

    compare_count = 0
    swap_count = 0
    same_label_swap_count = 0
    cross_label_swap_count = 0
    cross_label_swap_proposal_count = 0
    same_label_swap_proposal_count = 0
    update_count = 0
    wait_count = 0
    invalid_action_count = 0
    frozen_block_count = 0
    interface_rule_invocation_count = 0
    interface_block_count = 0
    adhesion_block_count = 0
    repulsion_block_count = 0
    permeability_block_count = 0
    recognition_block_count = 0
    same_contact_delta_sum = 0.0
    local_goal_delta_sum = 0.0
    block_reasons: Counter[str] = Counter()
    started = time.perf_counter()
    initial_metrics = sortedness_metrics(values)
    initial_labels = tuple(labels)
    initial_counts = Counter(labels)
    initial_arrangement = arrangement_metric_row(labels, condition.base.policy_ids)

    for actor_index in schedule:
        actor_index = int(actor_index)
        policy_id = str(labels[actor_index])
        cell_id = cell_ids[actor_index]
        actor_goal = goal_assignments[policy_id]
        proxy_values = rank_values(values, actor_goal, config.array_size)
        action = runtime.action_for(policy_id, proxy_values, labels, actor_index, cell_id, memory, rng)
        compare_count += int(bool(action.compare_counted))
        target = None if action.target_index is None else int(action.target_index)
        if target is not None and (actor_index in frozen_indices or target in frozen_indices):
            frozen_block_count += 1
            wait_count += 1
            state = memory.setdefault(cell_id, {})
            state["last_move_success"] = False
            state["last_action_type"] = "frozen_block"
            state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)
            continue
        if action.action_type == "swap" and target is not None:
            if 0 <= target < len(values):
                decision = interface_decision(
                    condition.rule,
                    values=values,
                    labels=labels,
                    actor_index=actor_index,
                    target_index=target,
                    goal_assignments=goal_assignments,
                    array_size=config.array_size,
                    rng=rng,
                )
                if bool(decision["cross_label"]):
                    cross_label_swap_proposal_count += 1
                else:
                    same_label_swap_proposal_count += 1
                if bool(decision["explicit_invoked"]):
                    interface_rule_invocation_count += 1
                    if math.isfinite(float(decision["same_contact_before"])):
                        same_contact_delta_sum += float(decision["same_contact_after"]) - float(decision["same_contact_before"])
                    if math.isfinite(float(decision["local_goal_before"])):
                        local_goal_delta_sum += float(decision["local_goal_after"]) - float(decision["local_goal_before"])
                if not bool(decision["accepted"]):
                    interface_block_count += 1
                    reason = str(decision["block_reason"])
                    block_reasons[reason] += 1
                    if condition.rule.rule_id == "adhesion_like_preference":
                        adhesion_block_count += 1
                    elif condition.rule.rule_id == "heterotypic_repulsion":
                        repulsion_block_count += 1
                    elif condition.rule.rule_id == "permeability_barrier":
                        permeability_block_count += 1
                    elif condition.rule.rule_id == "local_recognition_gate":
                        recognition_block_count += 1
                    wait_count += 1
                    state = memory.setdefault(cell_id, {})
                    state["last_move_success"] = False
                    state["last_action_type"] = f"interface_block:{reason}"
                    state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)
                    continue
                cross_swap = str(labels[actor_index]) != str(labels[target])
                values[actor_index], values[target] = values[target], values[actor_index]
                labels[actor_index], labels[target] = labels[target], labels[actor_index]
                cell_ids[actor_index], cell_ids[target] = cell_ids[target], cell_ids[actor_index]
                swap_count += 1
                if cross_swap:
                    cross_label_swap_count += 1
                else:
                    same_label_swap_count += 1
                state = memory.setdefault(cell_id, {})
                state["last_move_success"] = True
                state["last_action_type"] = "swap"
                state["time_since_movement"] = 0
            else:
                invalid_action_count += 1
                wait_count += 1
                state = memory.setdefault(cell_id, {})
                state["last_move_success"] = False
                state["last_action_type"] = "invalid_swap"
                state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)
        elif action.action_type == "update_state":
            update_count += 1
            state = memory.setdefault(cell_id, {})
            if "ideal_position" in action.state_update:
                updated = action.state_update["ideal_position"]
                state["ideal_position"] = None if updated is None else int(updated)
            state["last_action_type"] = "update_state"
        else:
            wait_count += 1
            state = memory.setdefault(cell_id, {})
            state["last_move_success"] = None
            state["last_action_type"] = "wait"
            state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)

    elapsed = time.perf_counter() - started
    final_metrics = sortedness_metrics(values)
    final_agg = aggregation_metrics(labels)
    final_counts = Counter(labels)
    dominance = dominance_metrics(labels, condition.base.policy_ids)
    realized_ratios = tuple(final_counts.get(policy_id, 0) / config.array_size for policy_id in condition.base.policy_ids)
    expected_counts = dict(zip(condition.base.policy_ids, counts, strict=True))
    final_state_class = classify_final_state(
        float(final_metrics["inversion_sortedness"]),
        float(final_agg["aggregation_delta_percent"]),
        float(final_agg["largest_block_fraction"]),
        int(final_agg["interface_count"]),
    )
    goal_eval = evaluate_goal_state(
        values=values,
        labels=labels,
        goal_assignments=goal_assignments,
        profile=goal_profile,
        array_size=config.array_size,
        aggregation_delta_percent=float(final_agg["aggregation_delta_percent"]),
    )
    return {
        "schema": INTERVENTION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "condition_id": condition.condition_id,
        "base_condition_id": condition.base.base_condition_id,
        "source_s07_run_id": condition.base.source_s07_run_id,
        "source_research_step_id": condition.base.source_research_step_id,
        "source_condition_id": condition.base.source_condition_id,
        "selection_reason": condition.base.selection_reason,
        "s07_label": condition.base.s07_label,
        "s07_classification_mode": condition.base.s07_classification_mode,
        "panel": condition.base.panel,
        "candidate_reason": condition.base.candidate_reason,
        "interface_rule_id": condition.rule.rule_id,
        "interface_mechanism_family": condition.rule.mechanism_family,
        "interface_rule_description": condition.rule.description,
        "explicit_recognition_used": bool(condition.rule.explicit_recognition_used),
        "recognition_access_class": condition.rule.recognition_access_class,
        "paper_claim_scope": _paper_scope(condition.rule),
        "no_recognition_baseline": not condition.rule.explicit_recognition_used,
        "array_size": int(config.array_size),
        "event_cap": int(config.event_cap),
        "events_executed": int(config.event_cap),
        "seed": int(seed),
        "scheduler": config.scheduler,
        "policy_ids_json": _json_list(condition.base.policy_ids),
        "display_names_json": _json_list(condition.base.display_names),
        "source_categories_json": _json_list(condition.base.source_categories),
        "ratio_targets_json": _json_list(condition.base.ratio_targets),
        "goal_profile_id": goal_profile.profile_id,
        "goal_compatibility_class": goal_profile.compatibility_class,
        "goal_assignments_json": _json_object(goal_assignments),
        "original_arrangement": condition.base.original_arrangement,
        "arrangement": condition.base.arrangement,
        "value_profile": condition.base.value_profile,
        "perturbation_profile": condition.base.perturbation_profile,
        "source_final_target_quality": condition.base.source_final_target_quality,
        "source_goal_conflict_index": condition.base.source_goal_conflict_index,
        "source_aggregation_delta_percent": condition.base.source_aggregation_delta_percent,
        "source_largest_block_fraction": condition.base.source_largest_block_fraction,
        "source_position_bias_abs": condition.base.source_position_bias_abs,
        "source_mosaic_continuum_score": condition.base.source_mosaic_continuum_score,
        "source_conflict_continuum_score": condition.base.source_conflict_continuum_score,
        "expected_counts_json": _json_object(expected_counts),
        "initial_counts_json": _json_object(dict(sorted(initial_counts.items()))),
        "final_counts_json": _json_object(dict(sorted(final_counts.items()))),
        "realized_ratios_json": _json_list(realized_ratios),
        "realized_ratio_max_abs_error": float(max(abs(realized_ratios[idx] - (counts[idx] / config.array_size)) for idx in range(len(counts)))),
        "count_preservation_success": dict(initial_counts) == dict(final_counts) == expected_counts,
        "frozen_indices_json": _json_list(sorted(frozen_indices)),
        "frozen_block_count": int(frozen_block_count),
        "compare_count": int(compare_count),
        "swap_count": int(swap_count),
        "same_label_swap_count": int(same_label_swap_count),
        "cross_label_swap_count": int(cross_label_swap_count),
        "same_label_swap_proposal_count": int(same_label_swap_proposal_count),
        "cross_label_swap_proposal_count": int(cross_label_swap_proposal_count),
        "update_count": int(update_count),
        "wait_count": int(wait_count),
        "work_count": int(compare_count + swap_count + update_count),
        "invalid_action_count": int(invalid_action_count),
        "interface_rule_invocation_count": int(interface_rule_invocation_count),
        "interface_block_count": int(interface_block_count),
        "adhesion_block_count": int(adhesion_block_count),
        "repulsion_block_count": int(repulsion_block_count),
        "permeability_block_count": int(permeability_block_count),
        "recognition_block_count": int(recognition_block_count),
        "block_reasons_json": _json_object(block_reasons),
        "interface_block_fraction": float(interface_block_count / max(1, cross_label_swap_proposal_count)),
        "cross_label_swap_acceptance_fraction": float(cross_label_swap_count / max(1, cross_label_swap_proposal_count)),
        "same_contact_delta_per_invocation": float(same_contact_delta_sum / max(1, interface_rule_invocation_count)),
        "local_goal_delta_per_invocation": float(local_goal_delta_sum / max(1, interface_rule_invocation_count)),
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


def run_s08_sweep(
    records: Sequence[Mapping[str, Any]],
    mosaic: pd.DataFrame,
    exemplars: pd.DataFrame,
    config: S08Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or S08Config()
    selected = select_s08_base_conditions(mosaic, exemplars, config)
    conditions = build_s08_conditions(selected, config)
    records_by_id = {str(record["algotypeId"]): dict(record) for record in records}
    missing = sorted({policy_id for condition in conditions for policy_id in condition.base.policy_ids} - set(records_by_id))
    if missing:
        raise ValueError(f"S08 selected policies missing from S01 library: {missing}")
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in config.seeds:
            rows.append(simulate_interface_condition(condition, records_by_id, seed=int(seed), config=config))
    run_df = pd.DataFrame(rows)
    condition_rows = []
    for condition in conditions:
        expected_counts = dict(zip(condition.base.policy_ids, counts_for_ratios(condition.base.ratio_targets, config.array_size), strict=True))
        condition_rows.append(
            {
                "condition_id": condition.condition_id,
                "base_condition_id": condition.base.base_condition_id,
                "source_s07_run_id": condition.base.source_s07_run_id,
                "source_research_step_id": condition.base.source_research_step_id,
                "source_condition_id": condition.base.source_condition_id,
                "selection_reason": condition.base.selection_reason,
                "s07_label": condition.base.s07_label,
                "interface_rule_id": condition.rule.rule_id,
                "interface_mechanism_family": condition.rule.mechanism_family,
                "explicit_recognition_used": bool(condition.rule.explicit_recognition_used),
                "recognition_access_class": condition.rule.recognition_access_class,
                "paper_claim_scope": _paper_scope(condition.rule),
                "policy_ids_json": _json_list(condition.base.policy_ids),
                "display_names_json": _json_list(condition.base.display_names),
                "source_categories_json": _json_list(condition.base.source_categories),
                "ratio_targets_json": _json_list(condition.base.ratio_targets),
                "expected_counts_json": _json_object(expected_counts),
                "arrangement": condition.base.arrangement,
                "goal_profile_id": condition.base.goal_profile_id,
                "value_profile": condition.base.value_profile,
                "perturbation_profile": condition.base.perturbation_profile,
            }
        )
    return run_df, pd.DataFrame(condition_rows), selected


def summarize_s08_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if run_df.empty:
        return pd.DataFrame()
    group_columns = ["base_condition_id", "interface_rule_id"]
    for _, group in run_df.groupby(group_columns, sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "base_condition_id": first["base_condition_id"],
                "interface_rule_id": first["interface_rule_id"],
                "interface_mechanism_family": first["interface_mechanism_family"],
                "explicit_recognition_used": bool(first["explicit_recognition_used"]),
                "recognition_access_class": first["recognition_access_class"],
                "paper_claim_scope": first["paper_claim_scope"],
                "selection_reason": first["selection_reason"],
                "s07_label": first["s07_label"],
                "source_research_step_id": first["source_research_step_id"],
                "display_names_json": first["display_names_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "arrangement": first["arrangement"],
                "goal_profile_id": first["goal_profile_id"],
                "value_profile": first["value_profile"],
                "perturbation_profile": first["perturbation_profile"],
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
                "mean_cross_label_swap_proposal_count": float(group["cross_label_swap_proposal_count"].mean()),
                "mean_cross_label_swap_count": float(group["cross_label_swap_count"].mean()),
                "mean_interface_rule_invocation_count": float(group["interface_rule_invocation_count"].mean()),
                "mean_interface_block_count": float(group["interface_block_count"].mean()),
                "mean_interface_block_fraction": float(group["interface_block_fraction"].mean()),
                "mean_cross_label_swap_acceptance_fraction": float(group["cross_label_swap_acceptance_fraction"].mean()),
                "mean_same_contact_delta_per_invocation": float(group["same_contact_delta_per_invocation"].mean()),
                "mean_local_goal_delta_per_invocation": float(group["local_goal_delta_per_invocation"].mean()),
                "invalid_action_count": int(group["invalid_action_count"].sum()),
                "count_preservation_success": bool(group["count_preservation_success"].all()),
                "final_state_class_mode": str(group["final_state_class"].mode().iloc[0]),
                "goal_state_class_mode": str(group["goal_state_class"].mode().iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def compare_to_behavior_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    baseline = summary[summary["interface_rule_id"] == "behavior_only"].copy()
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
    baseline_columns = ["base_condition_id", *metric_columns]
    baseline = baseline[baseline_columns].rename(columns={column: f"baseline_{column}" for column in metric_columns})
    comparison = summary.merge(baseline, on="base_condition_id", how="left")
    for column in metric_columns:
        comparison[f"delta_{column.removeprefix('mean_')}"] = comparison[column] - comparison[f"baseline_{column}"]
    comparison["schema"] = COMPARISON_SCHEMA
    comparison["abs_delta_aggregation_delta_percent"] = comparison["delta_aggregation_delta_percent"].abs()
    comparison["abs_delta_final_target_quality"] = comparison["delta_final_target_quality"].abs()
    return comparison


def mechanism_strata(comparison: pd.DataFrame, config: S08Config) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for rule_id, group in comparison.groupby("interface_rule_id", sort=False):
        first = group.iloc[0]
        is_baseline = str(rule_id) == "behavior_only"
        mean_abs_agg = float(group["abs_delta_aggregation_delta_percent"].mean())
        mean_block_fraction = float(group["mean_interface_block_fraction"].mean())
        explicit_dominates = bool(
            (not is_baseline)
            and (
                mean_abs_agg >= config.explicit_dominance_delta_threshold
                or mean_block_fraction >= config.explicit_dominance_block_threshold
            )
        )
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "interface_rule_id": rule_id,
                "interface_mechanism_family": first["interface_mechanism_family"],
                "explicit_recognition_used": bool(first["explicit_recognition_used"]),
                "condition_count": int(group["base_condition_id"].nunique()),
                "mean_delta_final_target_quality": float(group["delta_final_target_quality"].mean()),
                "mean_abs_delta_final_target_quality": float(group["abs_delta_final_target_quality"].mean()),
                "mean_delta_aggregation_delta_percent": float(group["delta_aggregation_delta_percent"].mean()),
                "mean_abs_delta_aggregation_delta_percent": mean_abs_agg,
                "mean_delta_interface_count": float(group["delta_interface_count"].mean()),
                "mean_delta_largest_block_fraction": float(group["delta_largest_block_fraction"].mean()),
                "mean_delta_goal_conflict_index": float(group["delta_goal_conflict_index"].mean()),
                "mean_interface_block_fraction": mean_block_fraction,
                "mean_cross_label_swap_acceptance_fraction": float(group["mean_cross_label_swap_acceptance_fraction"].mean()),
                "explicit_rule_dominates_baseline_behavior": explicit_dominates,
            }
        )
    return pd.DataFrame(rows)


def validate_s08_outputs(
    run_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    strata: pd.DataFrame,
    config: S08Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    seed_set = set(int(seed) for seed in config.seeds)
    expected_rules = {rule.rule_id for rule in config.interface_rules}
    explicit_rules = {rule.rule_id for rule in config.interface_rules if rule.explicit_recognition_used}
    base_rule_seed_counts = (
        run_df.groupby(["base_condition_id", "interface_rule_id"])["seed"].agg(lambda values: set(int(value) for value in values)).tolist()
        if not run_df.empty and {"base_condition_id", "interface_rule_id", "seed"}.issubset(run_df.columns)
        else []
    )
    base_rule_counts = run_df.groupby("base_condition_id")["interface_rule_id"].agg(lambda values: set(str(value) for value in values)).tolist() if not run_df.empty else []
    selection_reasons = ",".join(selected["selection_reason"].astype(str).tolist()) if "selection_reason" in selected else ""
    explicit_scope_ok = False
    baseline_scope_ok = False
    if not run_df.empty:
        baseline = run_df[run_df["interface_rule_id"] == "behavior_only"]
        explicit = run_df[run_df["interface_rule_id"].isin(explicit_rules)]
        baseline_scope_ok = bool(
            not baseline.empty
            and (baseline["explicit_recognition_used"] == False).all()  # noqa: E712
            and (baseline["recognition_access_class"].astype(str) == "none").all()
            and (baseline["paper_claim_scope"].astype(str) == "original_behavior_only_no_explicit_recognition").all()
            and (baseline["interface_rule_invocation_count"].astype(int) == 0).all()
            and (baseline["interface_block_count"].astype(int) == 0).all()
        )
        explicit_scope_ok = bool(
            not explicit.empty
            and explicit["explicit_recognition_used"].all()
            and (explicit["recognition_access_class"].astype(str) != "none").all()
            and explicit["paper_claim_scope"].astype(str).str.contains("beyond_original_no_recognition").all()
        )
    explicit_domination = bool(strata["explicit_rule_dominates_baseline_behavior"].any()) if "explicit_rule_dominates_baseline_behavior" in strata else False
    finite_comparison = False
    finite_columns = [
        "delta_final_target_quality",
        "delta_aggregation_delta_percent",
        "delta_interface_count",
        "delta_largest_block_fraction",
        "mean_interface_block_fraction",
    ]
    if not comparison.empty and set(finite_columns).issubset(comparison.columns):
        finite_comparison = bool(
            comparison[finite_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .all()
            .all()
        )
    checks = [
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "selected_conditions_use_s07_exemplars_and_metric_extremes",
            "success": bool(
                not selected.empty
                and selected["selection_reason"].astype(str).str.contains("s07_exemplar").any()
                and selected["selection_reason"].astype(str).str.contains("metric_extreme").any()
            ),
            "observed": selection_reasons,
            "expected": "bounded selection includes S07 exemplar rows and metric-continuum extremes",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "behavior_baseline_for_each_base_condition",
            "success": bool(base_rule_counts and all("behavior_only" in item for item in base_rule_counts)),
            "observed": f"base_conditions={len(base_rule_counts)}",
            "expected": "each selected base condition has a behavior-only no-recognition baseline",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "explicit_mechanism_families_present",
            "success": bool(expected_rules.issubset(set(run_df["interface_rule_id"].astype(str))) if not run_df.empty else False),
            "observed": sorted(run_df["interface_rule_id"].astype(str).unique().tolist()) if not run_df.empty else [],
            "expected": sorted(expected_rules),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "matched_seed_sets_by_base_and_rule",
            "success": bool(base_rule_seed_counts and all(item == seed_set for item in base_rule_seed_counts)),
            "observed": f"groups={len(base_rule_seed_counts)} expected_seeds={sorted(seed_set)}",
            "expected": "each base-condition/rule condition uses the same configured seed set",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "realized_ratios_preserved",
            "success": bool(not run_df.empty and (run_df["realized_ratio_max_abs_error"] <= 1e-12).all() and run_df["count_preservation_success"].all()),
            "observed": f"max_error={run_df['realized_ratio_max_abs_error'].max() if not run_df.empty else math.nan}",
            "expected": "integer policy counts are preserved under every intervention",
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
            "validation_case": "explicit_recognition_rules_separated_from_original_claims",
            "success": bool(baseline_scope_ok and explicit_scope_ok),
            "observed": f"baseline_scope_ok={baseline_scope_ok}; explicit_scope_ok={explicit_scope_ok}",
            "expected": "behavior-only rows carry no explicit recognition access; explicit mechanisms are labeled beyond original no-recognition claims",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "baseline_comparison_metrics_finite",
            "success": finite_comparison,
            "observed": f"comparison_rows={len(comparison)}",
            "expected": "baseline delta metrics are finite",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "explicit_rule_dominance_stratified_if_present",
            "success": bool((not explicit_domination) or (not strata.empty and "explicit_rule_dominates_baseline_behavior" in strata.columns)),
            "observed": f"explicit_domination={explicit_domination}; strata_rows={len(strata)}",
            "expected": "if explicit rules dominate, mechanism-specific strata are reported separately",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "figure_written",
            "success": bool(figure_written),
            "observed": str(bool(figure_written)),
            "expected": "interface-rule effects figure exists",
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
