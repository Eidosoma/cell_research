"""E06 S04 bounded goal-compatibility sweeps for chimeric mixtures."""

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


STEP_ID = "S04"
GOAL_SCHEMA = "eidosoma.e06.s04_goal_compatibility_run.v1"
SUMMARY_SCHEMA = "eidosoma.e06.s04_goal_compatibility_summary.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s04_validation.v1"
GOAL_NAMES = (
    "increasing",
    "decreasing",
    "high_half_first",
    "low_focus_increasing",
    "high_focus_increasing",
    "parity_even_first",
    "center_out",
)


@dataclass(frozen=True)
class GoalProfile:
    """A policy-goal assignment template for pairwise chimeras."""

    profile_id: str
    compatibility_class: str
    policy_goal_names: tuple[str, str]
    shared_goal_name: str | None
    relevant_goal_names: tuple[str, ...]
    description: str


DEFAULT_GOAL_PROFILES = (
    GoalProfile(
        profile_id="same_increasing",
        compatibility_class="same_goal",
        policy_goal_names=("increasing", "increasing"),
        shared_goal_name="increasing",
        relevant_goal_names=("increasing",),
        description="Both Algotypes optimize the standard increasing sort target.",
    ),
    GoalProfile(
        profile_id="opposite_direction",
        compatibility_class="opposite_goal",
        policy_goal_names=("increasing", "decreasing"),
        shared_goal_name=None,
        relevant_goal_names=("increasing", "decreasing"),
        description="First Algotype optimizes increasing order and second optimizes decreasing order.",
    ),
    GoalProfile(
        profile_id="partially_compatible_halves",
        compatibility_class="partially_compatible",
        policy_goal_names=("increasing", "high_half_first"),
        shared_goal_name=None,
        relevant_goal_names=("increasing", "high_half_first"),
        description="One Algotype preserves increasing order while the other prioritizes the high-value half first.",
    ),
    GoalProfile(
        profile_id="shared_global_focus",
        compatibility_class="local_goals_shared_global",
        policy_goal_names=("low_focus_increasing", "high_focus_increasing"),
        shared_goal_name="increasing",
        relevant_goal_names=("increasing", "low_focus_increasing", "high_focus_increasing"),
        description="Both local goals are monotone-compatible with the global increasing target but emphasize different value regions.",
    ),
    GoalProfile(
        profile_id="unrelated_parity_center",
        compatibility_class="unrelated_goal",
        policy_goal_names=("parity_even_first", "center_out"),
        shared_goal_name=None,
        relevant_goal_names=("parity_even_first", "center_out", "increasing"),
        description="Local targets use unrelated parity-first and center-out orderings.",
    ),
)


@dataclass(frozen=True)
class S04Config:
    """Configuration for the bounded S04 goal-compatibility sweep."""

    array_size: int = 100
    event_cap: int = 4_000
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    max_sensitive_rows: int = 6
    include_memory_contrast: bool = True
    goal_profiles: tuple[GoalProfile, ...] = DEFAULT_GOAL_PROFILES
    scheduler: str = "cyclic_scan_seed_offset"


@dataclass(frozen=True)
class GoalCondition:
    """One S04 candidate-ratio-arrangement-goal condition before seed expansion."""

    condition_id: str
    s03_candidate_id: str
    s03_candidate_rank: int
    panel: str
    candidate_reason: str
    policy_ids: tuple[str, str]
    display_names: tuple[str, str]
    ratio_targets: tuple[float, float]
    arrangement: str
    arrangement_role: str
    goal_profile: GoalProfile
    s03_sortedness_range: float
    s03_aggregation_delta_range: float
    s03_arrangement_sensitive: bool


def _json_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _condition_id(
    policy_ids: Sequence[str],
    ratios: Sequence[float],
    arrangement: str,
    goal_profile_id: str,
) -> str:
    payload = {
        "step": STEP_ID,
        "policyIds": list(policy_ids),
        "ratios": [round(float(value), 6) for value in ratios],
        "arrangement": arrangement,
        "goalProfileId": goal_profile_id,
    }
    return f"s04_{stable_hash(payload)[:16]}"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def rank_values(values: Sequence[int | float], goal_name: str, array_size: int | None = None) -> tuple[int, ...]:
    """Map actual values to integer ranks for a named goal ordering."""

    n = int(array_size or len(values))
    midpoint = (n + 1) / 2.0
    half = n // 2
    ranks: list[int] = []
    for raw in values:
        value = int(raw)
        if goal_name == "increasing":
            rank = value
        elif goal_name == "decreasing":
            rank = n + 1 - value
        elif goal_name == "high_half_first":
            rank = value - half if value > half else half + value
        elif goal_name == "low_focus_increasing":
            rank = value * 100 if value <= half else half * 100 + (value - half)
        elif goal_name == "high_focus_increasing":
            rank = value if value <= half else half + (value - half) * 100
        elif goal_name == "parity_even_first":
            rank = (0 if value % 2 == 0 else n) + value
        elif goal_name == "center_out":
            rank = int(abs(value - midpoint) * 1000 + value)
        else:
            raise ValueError(f"unknown goal name: {goal_name}")
        ranks.append(int(rank))
    return tuple(ranks)


def goal_sortedness(values: Sequence[int | float], goal_name: str, array_size: int | None = None) -> dict[str, Any]:
    ranks = rank_values(values, goal_name, array_size)
    metrics = sortedness_metrics(ranks)
    return {
        "goal_name": goal_name,
        "inversion_sortedness": float(metrics["inversion_sortedness"]),
        "adjacent_sortedness": float(metrics["adjacent_sortedness"]),
        "inversion_count": int(metrics["inversion_count"]),
        "is_sorted": bool(metrics["is_sorted"]),
    }


def goal_assignments_for_profile(policy_ids: Sequence[str], profile: GoalProfile) -> dict[str, str]:
    if len(policy_ids) != len(profile.policy_goal_names):
        raise ValueError("S04 currently expects pairwise policy IDs for goal assignment")
    assignments = dict(zip((str(item) for item in policy_ids), profile.policy_goal_names, strict=True))
    unknown = sorted(set(assignments.values()) - set(GOAL_NAMES))
    if unknown:
        raise ValueError(f"unknown S04 goal names in profile {profile.profile_id}: {unknown}")
    return assignments


def selected_goal_names(profile: GoalProfile, assignments: Mapping[str, str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for name in (*profile.relevant_goal_names, *assignments.values(), "increasing"):
        if name not in ordered:
            ordered.append(str(name))
    return tuple(ordered)


def assigned_subpopulation_scores(
    *,
    values: Sequence[int],
    labels: Sequence[str],
    goal_assignments: Mapping[str, str],
    array_size: int,
) -> dict[str, Any]:
    scores: dict[str, float | None] = {}
    weighted_sum = 0.0
    weighted_count = 0
    for policy_id, goal_name in goal_assignments.items():
        sub_values = [int(value) for value, label in zip(values, labels, strict=True) if str(label) == str(policy_id)]
        if len(sub_values) < 2:
            score = 1.0 if len(sub_values) == 1 else None
        else:
            score = float(goal_sortedness(sub_values, goal_name, array_size)["inversion_sortedness"])
        scores[policy_id] = score
        if score is not None:
            weighted_sum += float(score) * len(sub_values)
            weighted_count += len(sub_values)
    return {
        "assigned_policy_scores_json": json.dumps(scores, sort_keys=True, separators=(",", ":")),
        "assigned_policy_mean_sortedness": float(weighted_sum / weighted_count) if weighted_count else math.nan,
    }


def assigned_adjacent_satisfaction(
    *,
    values: Sequence[int],
    labels: Sequence[str],
    goal_assignments: Mapping[str, str],
    array_size: int,
) -> float:
    if len(values) < 2:
        return 1.0
    satisfied = 0
    total = 0
    for idx in range(len(values) - 1):
        left_goal = goal_assignments[str(labels[idx])]
        right_goal = goal_assignments[str(labels[idx + 1])]
        left_ranks = rank_values((values[idx], values[idx + 1]), left_goal, array_size)
        right_ranks = rank_values((values[idx], values[idx + 1]), right_goal, array_size)
        satisfied += int(left_ranks[0] <= left_ranks[1])
        satisfied += int(right_ranks[0] <= right_ranks[1])
        total += 2
    return float(satisfied / total)


def evaluate_goal_state(
    *,
    values: Sequence[int],
    labels: Sequence[str],
    goal_assignments: Mapping[str, str],
    profile: GoalProfile,
    array_size: int,
    aggregation_delta_percent: float,
) -> dict[str, Any]:
    goal_names = selected_goal_names(profile, goal_assignments)
    goal_scores = {name: goal_sortedness(values, name, array_size)["inversion_sortedness"] for name in goal_names}
    goal_adjacent = {name: goal_sortedness(values, name, array_size)["adjacent_sortedness"] for name in goal_names}
    assigned = assigned_subpopulation_scores(values=values, labels=labels, goal_assignments=goal_assignments, array_size=array_size)
    assigned_adjacent = assigned_adjacent_satisfaction(values=values, labels=labels, goal_assignments=goal_assignments, array_size=array_size)
    shared_score = goal_scores.get(profile.shared_goal_name) if profile.shared_goal_name else None
    reference_increasing = goal_scores.get("increasing", goal_sortedness(values, "increasing", array_size)["inversion_sortedness"])
    best_goal_name = max(goal_scores, key=lambda name: goal_scores[name])
    worst_goal_name = min(goal_scores, key=lambda name: goal_scores[name])
    goal_gap = float(goal_scores[best_goal_name] - goal_scores[worst_goal_name])
    assigned_mean = float(assigned["assigned_policy_mean_sortedness"])

    if shared_score is not None and shared_score >= 0.90 and assigned_mean >= 0.90:
        goal_state_class = "shared_and_local_success"
    elif shared_score is not None and shared_score >= 0.90:
        goal_state_class = "shared_success_local_tension"
    elif assigned_mean >= 0.90 and (shared_score is None or shared_score < 0.75):
        goal_state_class = "local_success_shared_failure"
    elif goal_gap >= 0.25 and aggregation_delta_percent >= 10.0:
        goal_state_class = "polarized_goal_conflict"
    elif max(goal_scores.values()) < 0.55 and assigned_mean < 0.65:
        goal_state_class = "low_goal_progress"
    elif assigned_mean >= 0.70 and max(goal_scores.values()) >= 0.70:
        goal_state_class = "mixed_goal_compromise"
    else:
        goal_state_class = "mixed_goal_tension"

    return {
        "relevant_goal_names_json": _json_list(goal_names),
        "goal_scores_json": json.dumps(goal_scores, sort_keys=True, separators=(",", ":")),
        "goal_adjacent_scores_json": json.dumps(goal_adjacent, sort_keys=True, separators=(",", ":")),
        "assigned_policy_scores_json": assigned["assigned_policy_scores_json"],
        "assigned_policy_mean_sortedness": assigned_mean,
        "assigned_adjacent_satisfaction": assigned_adjacent,
        "shared_goal_score": None if shared_score is None else float(shared_score),
        "reference_increasing_score": float(reference_increasing),
        "best_goal_name": best_goal_name,
        "best_goal_score": float(goal_scores[best_goal_name]),
        "worst_goal_name": worst_goal_name,
        "worst_goal_score": float(goal_scores[worst_goal_name]),
        "goal_alignment_gap": goal_gap,
        "goal_state_class": goal_state_class,
    }


def select_s04_candidates(s03_sensitivity: pd.DataFrame, config: S04Config) -> pd.DataFrame:
    if s03_sensitivity.empty:
        raise ValueError("S03 arrangement sensitivity table is empty")
    required = {
        "s02_candidate_id",
        "s02_candidate_rank",
        "panel",
        "candidate_reason",
        "policy_ids_json",
        "display_names_json",
        "ratio_targets_json",
        "sortedness_range_across_arrangements",
        "aggregation_delta_range_across_arrangements",
        "best_sortedness_arrangement",
        "worst_sortedness_arrangement",
        "arrangement_sensitive_candidate",
    }
    missing = required - set(s03_sensitivity.columns)
    if missing:
        raise ValueError(f"S03 sensitivity table missing columns: {sorted(missing)}")
    sensitivity = s03_sensitivity.copy()
    sensitivity["arrangement_sensitive_candidate"] = sensitivity["arrangement_sensitive_candidate"].map(_truthy)
    for column in ("sortedness_range_across_arrangements", "aggregation_delta_range_across_arrangements"):
        sensitivity[column] = pd.to_numeric(sensitivity[column], errors="coerce").fillna(0.0)

    selected_indices: list[int] = []

    def add(indexes: Sequence[int]) -> None:
        for idx in indexes:
            if idx not in selected_indices:
                selected_indices.append(int(idx))

    sensitive = sensitivity[sensitivity["arrangement_sensitive_candidate"]].sort_values(
        ["sortedness_range_across_arrangements", "aggregation_delta_range_across_arrangements"],
        ascending=[False, False],
        kind="mergesort",
    )
    add(sensitive.head(config.max_sensitive_rows).index.tolist())
    if config.include_memory_contrast:
        memory = sensitivity[
            sensitivity["panel"].astype(str).str.contains("memory", case=False, na=False)
            | sensitivity["display_names_json"].astype(str).str.contains("memory", case=False, na=False)
        ].sort_values(["sortedness_range_across_arrangements", "aggregation_delta_range_across_arrangements"], ascending=[False, False], kind="mergesort")
        add(memory.head(1).index.tolist())
    selected = sensitivity.loc[selected_indices].copy()
    selected["s04_candidate_rank"] = np.arange(1, len(selected) + 1)
    selected["selection_reason"] = np.where(
        selected["arrangement_sensitive_candidate"],
        "arrangement_sensitive",
        "memory_repair_contrast",
    )
    return selected.reset_index(drop=True)


def build_s04_conditions(selected_candidates: pd.DataFrame, config: S04Config) -> list[GoalCondition]:
    conditions: list[GoalCondition] = []
    for row in selected_candidates.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in json.loads(str(row["policy_ids_json"])))
        display_names = tuple(str(item) for item in json.loads(str(row["display_names_json"])))
        ratios = tuple(float(item) for item in json.loads(str(row["ratio_targets_json"])))
        if len(policy_ids) != 2 or len(ratios) != 2:
            continue
        arrangements = []
        for role, arrangement in (
            ("best_s03_sortedness", row["best_sortedness_arrangement"]),
            ("worst_s03_sortedness", row["worst_sortedness_arrangement"]),
        ):
            item = str(arrangement)
            if item not in [arr for _, arr in arrangements]:
                arrangements.append((role, item))
        for arrangement_role, arrangement in arrangements:
            for profile in config.goal_profiles:
                conditions.append(
                    GoalCondition(
                        condition_id=_condition_id(policy_ids, ratios, arrangement, profile.profile_id),
                        s03_candidate_id=str(row["s02_candidate_id"]),
                        s03_candidate_rank=int(row["s04_candidate_rank"]),
                        panel=str(row["panel"]),
                        candidate_reason=str(row["candidate_reason"]),
                        policy_ids=(policy_ids[0], policy_ids[1]),
                        display_names=(display_names[0], display_names[1]),
                        ratio_targets=(ratios[0], ratios[1]),
                        arrangement=arrangement,
                        arrangement_role=arrangement_role,
                        goal_profile=profile,
                        s03_sortedness_range=float(row["sortedness_range_across_arrangements"]),
                        s03_aggregation_delta_range=float(row["aggregation_delta_range_across_arrangements"]),
                        s03_arrangement_sensitive=bool(row["arrangement_sensitive_candidate"]),
                    )
                )
    return conditions


def simulate_goal_condition(
    condition: GoalCondition,
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: S04Config,
) -> dict[str, Any]:
    counts = counts_for_ratios(condition.ratio_targets, config.array_size)
    values = list(initial_values(config.array_size, seed))
    labels = list(
        labels_for_arrangement(
            condition.policy_ids,
            counts,
            seed=seed,
            condition_id=condition.condition_id,
            arrangement=condition.arrangement,
        )
    )
    cell_ids = list(range(config.array_size))
    memory: dict[int, dict[str, Any]] = {}
    runtime = MixtureRuntime([records_by_id[policy_id] for policy_id in condition.policy_ids], seed)
    schedule = actor_schedule(config.array_size, config.event_cap, seed)
    rng = random.Random(seed ^ int(stable_hash(condition.condition_id)[:12], 16))
    goal_assignments = goal_assignments_for_profile(condition.policy_ids, condition.goal_profile)
    compare_count = 0
    swap_count = 0
    update_count = 0
    wait_count = 0
    invalid_action_count = 0
    started = time.perf_counter()
    initial_metrics = sortedness_metrics(values)
    initial_counts = Counter(labels)
    initial_arrangement = arrangement_metric_row(labels, condition.policy_ids)

    for actor_index in schedule:
        actor_index = int(actor_index)
        policy_id = labels[actor_index]
        cell_id = cell_ids[actor_index]
        actor_goal = goal_assignments[str(policy_id)]
        proxy_values = rank_values(values, actor_goal, config.array_size)
        action = runtime.action_for(str(policy_id), proxy_values, labels, actor_index, cell_id, memory, rng)
        compare_count += int(bool(action.compare_counted))
        if action.action_type == "swap" and action.target_index is not None:
            target = int(action.target_index)
            if 0 <= target < len(values):
                values[actor_index], values[target] = values[target], values[actor_index]
                labels[actor_index], labels[target] = labels[target], labels[actor_index]
                cell_ids[actor_index], cell_ids[target] = cell_ids[target], cell_ids[actor_index]
                swap_count += 1
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
    dominance = dominance_metrics(labels, condition.policy_ids)
    realized_ratios = tuple(final_counts.get(policy_id, 0) / config.array_size for policy_id in condition.policy_ids)
    expected_counts = dict(zip(condition.policy_ids, counts, strict=True))
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
        profile=condition.goal_profile,
        array_size=config.array_size,
        aggregation_delta_percent=float(final_agg["aggregation_delta_percent"]),
    )
    categories = [str(records_by_id[policy_id]["sourceCategory"]) for policy_id in condition.policy_ids]
    row = {
        "schema": GOAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "condition_id": condition.condition_id,
        "s03_candidate_id": condition.s03_candidate_id,
        "s03_candidate_rank": int(condition.s03_candidate_rank),
        "panel": condition.panel,
        "candidate_reason": condition.candidate_reason,
        "arrangement": condition.arrangement,
        "arrangement_role": condition.arrangement_role,
        "goal_profile_id": condition.goal_profile.profile_id,
        "goal_compatibility_class": condition.goal_profile.compatibility_class,
        "goal_profile_description": condition.goal_profile.description,
        "array_size": int(config.array_size),
        "event_cap": int(config.event_cap),
        "events_executed": int(config.event_cap),
        "seed": int(seed),
        "scheduler": config.scheduler,
        "policy_ids_json": _json_list(condition.policy_ids),
        "display_names_json": _json_list(condition.display_names),
        "source_categories_json": _json_list(categories),
        "ratio_targets_json": _json_list(condition.ratio_targets),
        "goal_assignments_json": json.dumps(goal_assignments, sort_keys=True, separators=(",", ":")),
        "shared_goal_name": condition.goal_profile.shared_goal_name,
        "s03_sortedness_range": float(condition.s03_sortedness_range),
        "s03_aggregation_delta_range": float(condition.s03_aggregation_delta_range),
        "s03_arrangement_sensitive": bool(condition.s03_arrangement_sensitive),
        "expected_counts_json": json.dumps(expected_counts, sort_keys=True, separators=(",", ":")),
        "initial_counts_json": json.dumps(dict(sorted(initial_counts.items())), sort_keys=True, separators=(",", ":")),
        "final_counts_json": json.dumps(dict(sorted(final_counts.items())), sort_keys=True, separators=(",", ":")),
        "realized_ratios_json": _json_list(realized_ratios),
        "realized_ratio_max_abs_error": float(max(abs(realized_ratios[idx] - (counts[idx] / config.array_size)) for idx in range(len(counts)))),
        "count_preservation_success": dict(initial_counts) == dict(final_counts) == expected_counts,
        "compare_count": int(compare_count),
        "swap_count": int(swap_count),
        "update_count": int(update_count),
        "wait_count": int(wait_count),
        "work_count": int(compare_count + swap_count + update_count),
        "invalid_action_count": int(invalid_action_count),
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
        "mean_position_by_policy_json": dominance["mean_position_by_policy_json"],
        "final_state_class": final_state_class,
        "initial_labels_head_json": _json_list(labels[:20]),
        "initial_labels_tail_json": _json_list(labels[-20:]),
        "final_values_head_json": _json_list(values[:20]),
        "final_values_tail_json": _json_list(values[-20:]),
        "final_labels_head_json": _json_list(labels[:20]),
        "final_labels_tail_json": _json_list(labels[-20:]),
        "elapsed_seconds": float(elapsed),
    }
    row.update(initial_arrangement)
    row.update(goal_eval)
    return row


def run_s04_sweep(
    records: Sequence[Mapping[str, Any]],
    s03_sensitivity: pd.DataFrame,
    config: S04Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or S04Config()
    selected = select_s04_candidates(s03_sensitivity, config)
    conditions = build_s04_conditions(selected, config)
    records_by_id = {str(record["algotypeId"]): dict(record) for record in records}
    missing = sorted({policy_id for condition in conditions for policy_id in condition.policy_ids} - set(records_by_id))
    if missing:
        raise ValueError(f"S04 selected policies missing from S01 library: {missing}")
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in config.seeds:
            rows.append(simulate_goal_condition(condition, records_by_id, seed=int(seed), config=config))
    run_df = pd.DataFrame(rows)
    condition_rows = []
    for condition in conditions:
        goal_assignments = goal_assignments_for_profile(condition.policy_ids, condition.goal_profile)
        condition_rows.append(
            {
                "condition_id": condition.condition_id,
                "s03_candidate_id": condition.s03_candidate_id,
                "s03_candidate_rank": int(condition.s03_candidate_rank),
                "panel": condition.panel,
                "candidate_reason": condition.candidate_reason,
                "arrangement": condition.arrangement,
                "arrangement_role": condition.arrangement_role,
                "goal_profile_id": condition.goal_profile.profile_id,
                "goal_compatibility_class": condition.goal_profile.compatibility_class,
                "policy_ids_json": _json_list(condition.policy_ids),
                "display_names_json": _json_list(condition.display_names),
                "source_categories_json": _json_list([records_by_id[pid]["sourceCategory"] for pid in condition.policy_ids]),
                "ratio_targets_json": _json_list(condition.ratio_targets),
                "goal_assignments_json": json.dumps(goal_assignments, sort_keys=True, separators=(",", ":")),
                "shared_goal_name": condition.goal_profile.shared_goal_name,
                "relevant_goal_names_json": _json_list(selected_goal_names(condition.goal_profile, goal_assignments)),
                "s03_sortedness_range": float(condition.s03_sortedness_range),
                "s03_aggregation_delta_range": float(condition.s03_aggregation_delta_range),
                "s03_arrangement_sensitive": bool(condition.s03_arrangement_sensitive),
                "expected_counts_json": json.dumps(
                    dict(zip(condition.policy_ids, counts_for_ratios(condition.ratio_targets, config.array_size), strict=True)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return run_df, pd.DataFrame(condition_rows), selected


def summarize_s04_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["s03_candidate_id", "ratio_targets_json", "arrangement", "goal_profile_id"]
    for _, group in run_df.groupby(group_columns, sort=False):
        first = group.iloc[0]
        goal_counts = group["goal_state_class"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "s03_candidate_id": first["s03_candidate_id"],
                "s03_candidate_rank": int(first["s03_candidate_rank"]),
                "panel": first["panel"],
                "candidate_reason": first["candidate_reason"],
                "arrangement": first["arrangement"],
                "arrangement_role": first["arrangement_role"],
                "goal_profile_id": first["goal_profile_id"],
                "goal_compatibility_class": first["goal_compatibility_class"],
                "policy_ids_json": first["policy_ids_json"],
                "display_names_json": first["display_names_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "goal_assignments_json": first["goal_assignments_json"],
                "relevant_goal_names_json": first["relevant_goal_names_json"],
                "seed_count": int(group["seed"].nunique()),
                "run_count": int(len(group)),
                "mean_reference_increasing_score": float(group["reference_increasing_score"].mean()),
                "mean_assigned_policy_sortedness": float(group["assigned_policy_mean_sortedness"].mean()),
                "mean_assigned_adjacent_satisfaction": float(group["assigned_adjacent_satisfaction"].mean()),
                "mean_shared_goal_score": float(group["shared_goal_score"].dropna().mean()) if group["shared_goal_score"].notna().any() else math.nan,
                "mean_best_goal_score": float(group["best_goal_score"].mean()),
                "mean_goal_alignment_gap": float(group["goal_alignment_gap"].mean()),
                "mean_final_inversion_sortedness": float(group["final_inversion_sortedness"].mean()),
                "mean_aggregation_delta_percent": float(group["aggregation_delta_percent"].mean()),
                "mean_interface_count": float(group["interface_count"].mean()),
                "invalid_action_count": int(group["invalid_action_count"].sum()),
                "goal_state_class_mode": str(group["goal_state_class"].mode().iloc[0]),
                "goal_state_class_counts_json": json.dumps(goal_counts, sort_keys=True, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def goal_profile_sensitivity(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary_df.empty:
        return pd.DataFrame()
    group_columns = ["s03_candidate_id", "ratio_targets_json", "arrangement"]
    for _, group in summary_df.groupby(group_columns, sort=False):
        first = group.iloc[0]
        assigned = group["mean_assigned_policy_sortedness"].astype(float)
        reference = group["mean_reference_increasing_score"].astype(float)
        gap = group["mean_goal_alignment_gap"].astype(float)
        rows.append(
            {
                "s03_candidate_id": first["s03_candidate_id"],
                "s03_candidate_rank": int(first["s03_candidate_rank"]),
                "panel": first["panel"],
                "display_names_json": first["display_names_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "arrangement": first["arrangement"],
                "arrangement_role": first["arrangement_role"],
                "goal_profile_count": int(group["goal_profile_id"].nunique()),
                "assigned_policy_sortedness_range": float(assigned.max() - assigned.min()),
                "reference_increasing_range": float(reference.max() - reference.min()),
                "goal_alignment_gap_range": float(gap.max() - gap.min()),
                "best_assigned_profile": str(group.loc[assigned.idxmax(), "goal_profile_id"]),
                "best_assigned_score": float(assigned.max()),
                "worst_assigned_profile": str(group.loc[assigned.idxmin(), "goal_profile_id"]),
                "worst_assigned_score": float(assigned.min()),
                "goal_sensitive_candidate": bool((assigned.max() - assigned.min()) >= 0.05 or (gap.max() - gap.min()) >= 0.10),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["assigned_policy_sortedness_range", "goal_alignment_gap_range"],
        ascending=[False, False],
        kind="mergesort",
    )


def validate_s04_outputs(
    run_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    config: S04Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    seed_set = set(int(seed) for seed in config.seeds)
    profile_set = {profile.profile_id for profile in config.goal_profiles}
    required_run_columns = {
        "condition_id",
        "s03_candidate_id",
        "arrangement",
        "goal_profile_id",
        "seed",
        "policy_ids_json",
        "goal_assignments_json",
        "relevant_goal_names_json",
        "goal_scores_json",
        "assigned_policy_mean_sortedness",
        "assigned_adjacent_satisfaction",
        "realized_ratio_max_abs_error",
        "count_preservation_success",
        "invalid_action_count",
    }
    has_run_schema = required_run_columns.issubset(run_df.columns)
    seed_sets = []
    profile_sets = []
    assignments_valid = False
    final_eval_valid = False
    ratio_success = False
    invalid_success = False
    if has_run_schema and not run_df.empty:
        seed_sets = run_df.groupby(["s03_candidate_id", "ratio_targets_json", "arrangement", "goal_profile_id"])["seed"].agg(
            lambda values: set(int(value) for value in values)
        ).tolist()
        profile_sets = run_df.groupby(["s03_candidate_id", "ratio_targets_json", "arrangement", "seed"])["goal_profile_id"].agg(
            lambda values: set(map(str, values))
        ).tolist()
        assignment_flags = []
        eval_flags = []
        for row in run_df.to_dict(orient="records"):
            policy_ids = set(json.loads(row["policy_ids_json"]))
            assignments = json.loads(row["goal_assignments_json"])
            relevant = set(json.loads(row["relevant_goal_names_json"]))
            scores = json.loads(row["goal_scores_json"])
            assignment_flags.append(
                set(assignments) == policy_ids
                and set(assignments.values()).issubset(set(GOAL_NAMES))
                and set(assignments.values()).issubset(relevant)
            )
            eval_flags.append(
                relevant.issubset(set(scores))
                and all(math.isfinite(float(scores[name])) for name in relevant)
                and math.isfinite(float(row["assigned_policy_mean_sortedness"]))
                and math.isfinite(float(row["assigned_adjacent_satisfaction"]))
            )
        assignments_valid = bool(all(assignment_flags))
        final_eval_valid = bool(all(eval_flags))
        ratio_success = bool((run_df["realized_ratio_max_abs_error"] <= 1e-12).all() and run_df["count_preservation_success"].all())
        invalid_success = int(run_df["invalid_action_count"].sum()) == 0
    checks = [
        {
            "validation_case": "per_algotype_goal_assignments_valid",
            "success": assignments_valid,
            "observed": "all run rows have one recognized goal per policy ID" if assignments_valid else "invalid or missing goal assignments",
            "expected": "goal_assignments_json keys match policy_ids_json and values are registered S04 goals",
        },
        {
            "validation_case": "final_state_evaluated_under_relevant_goals",
            "success": final_eval_valid,
            "observed": "all relevant goal scores and assigned-goal scores finite" if final_eval_valid else "missing or non-finite relevant goal score",
            "expected": "every run row contains finite final-state scores for all relevant goals",
        },
        {
            "validation_case": "matched_seed_sets_by_condition",
            "success": bool(seed_sets and all(item == seed_set for item in seed_sets)),
            "observed": f"{len(seed_sets)} condition groups; expected_seeds={sorted(seed_set)}",
            "expected": "each candidate-arrangement-goal condition uses the same configured seed set",
        },
        {
            "validation_case": "matched_goal_profiles_by_seed",
            "success": bool(profile_sets and all(item == profile_set for item in profile_sets)),
            "observed": f"{len(profile_sets)} candidate-arrangement-seed groups; expected_goal_profiles={sorted(profile_set)}",
            "expected": "each candidate-arrangement-seed group includes every configured goal profile",
        },
        {
            "validation_case": "realized_ratios_and_counts_match",
            "success": ratio_success,
            "observed": f"max_error={run_df['realized_ratio_max_abs_error'].max():.3g}; count_failures={int((~run_df['count_preservation_success']).sum())}" if has_run_schema and not run_df.empty else "missing run rows",
            "expected": "zero realized-ratio error against integer counts and all counts preserved",
        },
        {
            "validation_case": "bounded_s03_candidate_set_used",
            "success": bool(not selected_candidates.empty and len(selected_candidates) <= config.max_sensitive_rows + int(config.include_memory_contrast)),
            "observed": f"selected_candidates={len(selected_candidates)} max={config.max_sensitive_rows + int(config.include_memory_contrast)}",
            "expected": "bounded nonempty S03-derived candidate set",
        },
        {
            "validation_case": "result_rows_complete",
            "success": bool(len(run_df) == len(condition_df) * len(seed_set) and not condition_df.empty),
            "observed": f"runs={len(run_df)} conditions={len(condition_df)} seeds={len(seed_set)}",
            "expected": "run rows equal condition count times seed count",
        },
        {
            "validation_case": "no_invalid_actions",
            "success": invalid_success,
            "observed": str(int(run_df["invalid_action_count"].sum())) if has_run_schema and not run_df.empty else "missing run rows",
            "expected": "zero out-of-bounds or invalid actions",
        },
        {
            "validation_case": "figure_written",
            "success": bool(figure_written),
            "observed": str(bool(figure_written)),
            "expected": "goal compatibility matrix figure exists",
        },
        {
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests_success),
            "observed": str(bool(unit_tests_success)),
            "expected": "focused E06 unit tests pass",
        },
    ]
    return pd.DataFrame(checks)
