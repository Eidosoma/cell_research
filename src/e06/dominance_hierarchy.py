"""E06 S06 opposite-goal dominance hierarchy sweeps."""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.coarse_sweep import actor_schedule, initial_values, sortedness_metrics
from src.e06.goal_compatibility import (
    GoalProfile,
    assigned_subpopulation_scores,
    evaluate_goal_state,
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


STEP_ID = "S06"
DOMINANCE_RUN_SCHEMA = "eidosoma.e06.s06_dominance_contest_run.v1"
SUMMARY_SCHEMA = "eidosoma.e06.s06_dominance_contest_summary.v1"
HIERARCHY_SCHEMA = "eidosoma.e06.s06_dominance_hierarchy.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s06_validation.v1"
OPPOSITE_PROFILE = GoalProfile(
    profile_id="s06_opposite_direction",
    compatibility_class="opposite_goal",
    policy_goal_names=("increasing", "decreasing"),
    shared_goal_name=None,
    relevant_goal_names=("increasing", "decreasing"),
    description="S06 direct contest: first policy optimizes increasing order and second optimizes decreasing order.",
)
VALUE_PROFILES = ("random_permutation", "reverse_sorted")
PERTURBATION_PROFILES = ("none", "frozen_edges_2")


@dataclass(frozen=True)
class S06Config:
    """Configuration for bounded S06 dominance contests."""

    array_size: int = 100
    event_cap: int = 4_000
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    max_s05_candidates: int = 18
    include_paper_original_controls: bool = True
    value_profiles: tuple[str, ...] = VALUE_PROFILES
    perturbation_profiles: tuple[str, ...] = PERTURBATION_PROFILES
    orientations: tuple[str, ...] = ("as_selected", "relabel_swap")
    scheduler: str = "cyclic_scan_seed_offset"
    dominance_tie_threshold: float = 0.03


@dataclass(frozen=True)
class DominanceBaseContest:
    """One S06 base contest before orientation, value profile, perturbation, and seed expansion."""

    base_contest_id: str
    source_score_id: str
    source_condition_id: str
    source_kind: str
    selection_rank: int
    panel: str
    policy_ids: tuple[str, str]
    display_names: tuple[str, str]
    source_categories: tuple[str, str]
    ratio_targets: tuple[float, float]
    arrangement: str
    s05_goal_profile_id: str
    s05_goal_compatibility_class: str
    s05_priority_score: float
    s05_goal_conflict_index: float
    s05_dominance_abs_margin: float


@dataclass(frozen=True)
class DominanceCondition:
    """One executable S06 condition before seed expansion."""

    condition_id: str
    symmetry_group_id: str
    base_contest: DominanceBaseContest
    orientation: str
    policy_ids: tuple[str, str]
    display_names: tuple[str, str]
    source_categories: tuple[str, str]
    ratio_targets: tuple[float, float]
    arrangement: str
    value_profile: str
    perturbation_profile: str


def _json_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _json_object(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)


def _condition_id(payload: Mapping[str, Any]) -> str:
    return f"s06_{stable_hash(payload)[:16]}"


def _parse_json_list(text: Any) -> tuple[Any, ...]:
    return tuple(json.loads(str(text)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def initial_values_for_profile(array_size: int, seed: int, value_profile: str) -> tuple[int, ...]:
    if value_profile == "random_permutation":
        return tuple(int(value) for value in initial_values(array_size, seed))
    if value_profile == "reverse_sorted":
        return tuple(range(array_size, 0, -1))
    raise ValueError(f"unknown S06 value profile: {value_profile}")


def frozen_indices_for_profile(array_size: int, perturbation_profile: str) -> frozenset[int]:
    if perturbation_profile == "none":
        return frozenset()
    if perturbation_profile == "frozen_edges_2":
        return frozenset({0, max(0, array_size - 1)})
    raise ValueError(f"unknown S06 perturbation profile: {perturbation_profile}")


def _metadata_maps(metadata: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    display_to_id = {str(row["display_name"]): str(row["algotype_id"]) for row in metadata.to_dict(orient="records")}
    id_to_display = {str(row["algotype_id"]): str(row["display_name"]) for row in metadata.to_dict(orient="records")}
    id_to_category = {str(row["algotype_id"]): str(row["source_category"]) for row in metadata.to_dict(orient="records")}
    return display_to_id, id_to_display, id_to_category


def select_s06_base_contests(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    config: S06Config,
) -> pd.DataFrame:
    """Select bounded S05 candidate contests and explicit paper-original controls."""

    if candidates.empty:
        raise ValueError("S05 S06 candidate-contest table is empty")
    if scores.empty:
        raise ValueError("S05 compatibility score table is empty")
    required = {"score_id", "s06_priority_score", "arrangement", "goal_profile_id", "goal_compatibility_class"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"S05 candidate table missing columns: {sorted(missing)}")
    score_required = {"score_id", "policy_ids_json", "display_names_json", "source_categories_json", "ratio_targets_json"}
    score_missing = score_required - set(scores.columns)
    if score_missing:
        raise ValueError(f"S05 score table missing columns: {sorted(score_missing)}")
    merged = candidates.merge(
        scores[
            [
                "score_id",
                "policy_ids_json",
                "display_names_json",
                "source_categories_json",
                "ratio_targets_json",
            ]
        ],
        on="score_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_score"),
    )
    if merged["policy_ids_json"].isna().any():
        raise ValueError("could not join all S06 candidates to S05 score rows")
    merged = merged.sort_values("s06_priority_score", ascending=False, kind="mergesort")
    merged["dedupe_key"] = merged.apply(
        lambda row: _json_object(
            {
                "policyIds": _parse_json_list(row["policy_ids_json"]),
                "ratios": _parse_json_list(row["ratio_targets_json"]),
                "arrangement": row["arrangement"],
            }
        ),
        axis=1,
    )
    selected = merged.drop_duplicates("dedupe_key", keep="first").head(config.max_s05_candidates).copy()
    selected["source_kind"] = "s05_candidate"
    selected["selection_rank"] = np.arange(1, len(selected) + 1)

    _display_to_id, id_to_display, id_to_category = _metadata_maps(metadata)
    rows: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in _parse_json_list(row["policy_ids_json"]))
        ratios = tuple(float(item) for item in _parse_json_list(row["ratio_targets_json"]))
        display_names = tuple(str(item) for item in _parse_json_list(row["display_names_json"]))
        categories = tuple(str(item) for item in _parse_json_list(row["source_categories_json"]))
        rows.append(
            {
                "base_contest_id": _condition_id(
                    {
                        "source": "s05",
                        "scoreId": row["score_id"],
                        "policyIds": policy_ids,
                        "ratios": ratios,
                        "arrangement": row["arrangement"],
                    }
                ),
                "source_score_id": str(row["score_id"]),
                "source_condition_id": str(row["source_condition_id"]),
                "source_kind": "s05_candidate",
                "selection_rank": int(row["selection_rank"]),
                "panel": str(row["panel"]),
                "policy_ids_json": _json_list(policy_ids),
                "display_names_json": _json_list(display_names),
                "source_categories_json": _json_list(categories),
                "ratio_targets_json": _json_list(ratios),
                "arrangement": str(row["arrangement"]),
                "s05_goal_profile_id": str(row["goal_profile_id"]),
                "s05_goal_compatibility_class": str(row["goal_compatibility_class"]),
                "s05_priority_score": float(row["s06_priority_score"]),
                "s05_goal_conflict_index": float(row["goal_conflict_index"]),
                "s05_dominance_abs_margin": float(row["dominance_abs_margin"]),
            }
        )

    if config.include_paper_original_controls:
        originals = metadata[metadata["source_category"] == "original"].sort_values("display_name")
        original_ids = originals["algotype_id"].astype(str).tolist()
        original_pairs = [
            ("original_bubble", "original_insertion"),
            ("original_bubble", "original_selection"),
            ("original_insertion", "original_selection"),
        ]
        display_to_id, _id_to_display, _id_to_category = _metadata_maps(metadata)
        next_rank = len(rows) + 1
        for left_name, right_name in original_pairs:
            if left_name not in display_to_id or right_name not in display_to_id:
                continue
            left = display_to_id[left_name]
            right = display_to_id[right_name]
            if left not in original_ids or right not in original_ids:
                continue
            policy_ids = (left, right)
            ratios = (0.5, 0.5)
            rows.append(
                {
                    "base_contest_id": _condition_id(
                        {"source": "paper_original_control", "policyIds": policy_ids, "ratios": ratios, "arrangement": "random_permutation"}
                    ),
                    "source_score_id": "",
                    "source_condition_id": "",
                    "source_kind": "paper_original_control",
                    "selection_rank": next_rank,
                    "panel": "paper_originals",
                    "policy_ids_json": _json_list(policy_ids),
                    "display_names_json": _json_list([id_to_display[policy_id] for policy_id in policy_ids]),
                    "source_categories_json": _json_list([id_to_category[policy_id] for policy_id in policy_ids]),
                    "ratio_targets_json": _json_list(ratios),
                    "arrangement": "random_permutation",
                    "s05_goal_profile_id": "paper_original_control",
                    "s05_goal_compatibility_class": "opposite_goal",
                    "s05_priority_score": 0.0,
                    "s05_goal_conflict_index": 0.0,
                    "s05_dominance_abs_margin": 0.0,
                }
            )
            next_rank += 1

    return pd.DataFrame(rows)


def build_s06_conditions(base_contests: pd.DataFrame, config: S06Config) -> list[DominanceCondition]:
    conditions: list[DominanceCondition] = []
    for raw in base_contests.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
        display_names = tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
        categories = tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
        ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
        if len(policy_ids) != 2 or len(ratios) != 2:
            continue
        base = DominanceBaseContest(
            base_contest_id=str(raw["base_contest_id"]),
            source_score_id=str(raw["source_score_id"]),
            source_condition_id=str(raw["source_condition_id"]),
            source_kind=str(raw["source_kind"]),
            selection_rank=int(raw["selection_rank"]),
            panel=str(raw["panel"]),
            policy_ids=(policy_ids[0], policy_ids[1]),
            display_names=(display_names[0], display_names[1]),
            source_categories=(categories[0], categories[1]),
            ratio_targets=(ratios[0], ratios[1]),
            arrangement=str(raw["arrangement"]),
            s05_goal_profile_id=str(raw["s05_goal_profile_id"]),
            s05_goal_compatibility_class=str(raw["s05_goal_compatibility_class"]),
            s05_priority_score=float(raw["s05_priority_score"]),
            s05_goal_conflict_index=float(raw["s05_goal_conflict_index"]),
            s05_dominance_abs_margin=float(raw["s05_dominance_abs_margin"]),
        )
        for value_profile in config.value_profiles:
            for perturbation_profile in config.perturbation_profiles:
                symmetry_group_id = _condition_id(
                    {
                        "baseContestId": base.base_contest_id,
                        "valueProfile": value_profile,
                        "perturbationProfile": perturbation_profile,
                    }
                )
                for orientation in config.orientations:
                    if orientation == "as_selected":
                        oriented_ids = base.policy_ids
                        oriented_names = base.display_names
                        oriented_categories = base.source_categories
                        oriented_ratios = base.ratio_targets
                    elif orientation == "relabel_swap":
                        oriented_ids = (base.policy_ids[1], base.policy_ids[0])
                        oriented_names = (base.display_names[1], base.display_names[0])
                        oriented_categories = (base.source_categories[1], base.source_categories[0])
                        oriented_ratios = base.ratio_targets
                    else:
                        raise ValueError(f"unknown S06 orientation: {orientation}")
                    condition_payload = {
                        "symmetryGroupId": symmetry_group_id,
                        "orientation": orientation,
                        "policyIds": oriented_ids,
                        "ratios": oriented_ratios,
                        "arrangement": base.arrangement,
                    }
                    conditions.append(
                        DominanceCondition(
                            condition_id=_condition_id(condition_payload),
                            symmetry_group_id=symmetry_group_id,
                            base_contest=base,
                            orientation=orientation,
                            policy_ids=(oriented_ids[0], oriented_ids[1]),
                            display_names=(oriented_names[0], oriented_names[1]),
                            source_categories=(oriented_categories[0], oriented_categories[1]),
                            ratio_targets=(oriented_ratios[0], oriented_ratios[1]),
                            arrangement=base.arrangement,
                            value_profile=str(value_profile),
                            perturbation_profile=str(perturbation_profile),
                        )
                    )
    return conditions


def _goal_assignments(policy_ids: Sequence[str]) -> dict[str, str]:
    if len(policy_ids) != 2:
        raise ValueError("S06 dominance contests require pairwise policies")
    return {str(policy_ids[0]): "increasing", str(policy_ids[1]): "decreasing"}


def _dominance_from_goal_scores(
    policy_ids: Sequence[str],
    goal_assignments: Mapping[str, str],
    goal_scores: Mapping[str, float],
    tie_threshold: float,
) -> dict[str, Any]:
    first, second = str(policy_ids[0]), str(policy_ids[1])
    first_goal = goal_assignments[first]
    second_goal = goal_assignments[second]
    first_score = float(goal_scores[first_goal])
    second_score = float(goal_scores[second_goal])
    margin = first_score - second_score
    if margin > tie_threshold:
        winner = first
        loser = second
    elif margin < -tie_threshold:
        winner = second
        loser = first
    else:
        winner = "tie"
        loser = "tie"
    return {
        "dominant_policy_id": winner,
        "dominated_policy_id": loser,
        "global_goal_margin_first_minus_second": float(margin),
        "dominance_margin_abs": float(abs(margin)),
        "first_global_goal_score": first_score,
        "second_global_goal_score": second_score,
    }


def _assigned_goal_margin(policy_ids: Sequence[str], assigned_scores_json: str) -> float:
    assigned = json.loads(assigned_scores_json)
    first = assigned.get(str(policy_ids[0]))
    second = assigned.get(str(policy_ids[1]))
    if first is None or second is None:
        return math.nan
    return float(first) - float(second)


def _apply_perturbation_block(
    actor_index: int,
    target_index: int | None,
    frozen_indices: frozenset[int],
) -> bool:
    if actor_index in frozen_indices:
        return True
    return target_index is not None and int(target_index) in frozen_indices


def simulate_dominance_condition(
    condition: DominanceCondition,
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: S06Config,
) -> dict[str, Any]:
    counts = counts_for_ratios(condition.ratio_targets, config.array_size)
    values = list(initial_values_for_profile(config.array_size, seed, condition.value_profile))
    labels = list(
        labels_for_arrangement(
            condition.policy_ids,
            counts,
            seed=seed,
            condition_id=condition.symmetry_group_id,
            arrangement=condition.arrangement,
        )
    )
    if condition.orientation == "relabel_swap":
        # Keep the same spatial pattern as the selected orientation, then relabel.
        base_labels = labels_for_arrangement(
            condition.base_contest.policy_ids,
            counts_for_ratios(condition.base_contest.ratio_targets, config.array_size),
            seed=seed,
            condition_id=condition.symmetry_group_id,
            arrangement=condition.arrangement,
        )
        label_map = {condition.base_contest.policy_ids[0]: condition.base_contest.policy_ids[1], condition.base_contest.policy_ids[1]: condition.base_contest.policy_ids[0]}
        labels = [label_map[str(label)] for label in base_labels]
    cell_ids = list(range(config.array_size))
    memory: dict[int, dict[str, Any]] = {}
    runtime = MixtureRuntime([records_by_id[policy_id] for policy_id in condition.policy_ids], seed)
    schedule = actor_schedule(config.array_size, config.event_cap, seed)
    rng = random.Random(seed ^ int(stable_hash(condition.symmetry_group_id)[:12], 16))
    goal_assignments = _goal_assignments(condition.policy_ids)
    frozen_indices = frozen_indices_for_profile(config.array_size, condition.perturbation_profile)
    compare_count = 0
    swap_count = 0
    update_count = 0
    wait_count = 0
    invalid_action_count = 0
    frozen_block_count = 0
    started = time.perf_counter()
    initial_metrics = sortedness_metrics(values)
    initial_labels = tuple(labels)
    initial_counts = Counter(labels)
    initial_arrangement = arrangement_metric_row(labels, condition.policy_ids)

    for actor_index in schedule:
        actor_index = int(actor_index)
        policy_id = str(labels[actor_index])
        cell_id = cell_ids[actor_index]
        actor_goal = goal_assignments[policy_id]
        proxy_values = rank_values(values, actor_goal, config.array_size)
        action = runtime.action_for(policy_id, proxy_values, labels, actor_index, cell_id, memory, rng)
        compare_count += int(bool(action.compare_counted))
        target = None if action.target_index is None else int(action.target_index)
        if _apply_perturbation_block(actor_index, target, frozen_indices):
            frozen_block_count += 1
            wait_count += 1
            state = memory.setdefault(cell_id, {})
            state["last_move_success"] = False
            state["last_action_type"] = "frozen_block"
            state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)
            continue
        if action.action_type == "swap" and target is not None:
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
        profile=OPPOSITE_PROFILE,
        array_size=config.array_size,
        aggregation_delta_percent=float(final_agg["aggregation_delta_percent"]),
    )
    goal_scores = json.loads(goal_eval["goal_scores_json"])
    dominance_goal = _dominance_from_goal_scores(condition.policy_ids, goal_assignments, goal_scores, config.dominance_tie_threshold)
    assigned_margin = _assigned_goal_margin(condition.policy_ids, goal_eval["assigned_policy_scores_json"])
    row = {
        "schema": DOMINANCE_RUN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "condition_id": condition.condition_id,
        "symmetry_group_id": condition.symmetry_group_id,
        "base_contest_id": condition.base_contest.base_contest_id,
        "source_score_id": condition.base_contest.source_score_id,
        "source_condition_id": condition.base_contest.source_condition_id,
        "source_kind": condition.base_contest.source_kind,
        "selection_rank": int(condition.base_contest.selection_rank),
        "panel": condition.base_contest.panel,
        "orientation": condition.orientation,
        "arrangement": condition.arrangement,
        "value_profile": condition.value_profile,
        "perturbation_profile": condition.perturbation_profile,
        "goal_profile_id": OPPOSITE_PROFILE.profile_id,
        "goal_compatibility_class": OPPOSITE_PROFILE.compatibility_class,
        "array_size": int(config.array_size),
        "event_cap": int(config.event_cap),
        "events_executed": int(config.event_cap),
        "seed": int(seed),
        "scheduler": config.scheduler,
        "policy_ids_json": _json_list(condition.policy_ids),
        "display_names_json": _json_list(condition.display_names),
        "source_categories_json": _json_list(condition.source_categories),
        "ratio_targets_json": _json_list(condition.ratio_targets),
        "goal_assignments_json": _json_object(goal_assignments),
        "s05_goal_profile_id": condition.base_contest.s05_goal_profile_id,
        "s05_goal_compatibility_class": condition.base_contest.s05_goal_compatibility_class,
        "s05_priority_score": float(condition.base_contest.s05_priority_score),
        "s05_goal_conflict_index": float(condition.base_contest.s05_goal_conflict_index),
        "s05_dominance_abs_margin": float(condition.base_contest.s05_dominance_abs_margin),
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
        "assigned_goal_margin_first_minus_second": float(assigned_margin),
        "initial_labels_head_json": _json_list(initial_labels[:20]),
        "initial_labels_tail_json": _json_list(initial_labels[-20:]),
        "final_values_head_json": _json_list(values[:20]),
        "final_values_tail_json": _json_list(values[-20:]),
        "final_labels_head_json": _json_list(labels[:20]),
        "final_labels_tail_json": _json_list(labels[-20:]),
        "elapsed_seconds": float(elapsed),
    }
    row.update(initial_arrangement)
    row.update(goal_eval)
    row.update(dominance_goal)
    return row


def run_s06_sweep(
    records: Sequence[Mapping[str, Any]],
    metadata: pd.DataFrame,
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    config: S06Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or S06Config()
    base_contests = select_s06_base_contests(candidates, scores, metadata, config)
    conditions = build_s06_conditions(base_contests, config)
    records_by_id = {str(record["algotypeId"]): dict(record) for record in records}
    missing = sorted({policy_id for condition in conditions for policy_id in condition.policy_ids} - set(records_by_id))
    if missing:
        raise ValueError(f"S06 selected policies missing from S01 library: {missing}")
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in config.seeds:
            rows.append(simulate_dominance_condition(condition, records_by_id, seed=int(seed), config=config))
    run_df = pd.DataFrame(rows)
    condition_rows = []
    for condition in conditions:
        goal_assignments = _goal_assignments(condition.policy_ids)
        expected_counts = dict(zip(condition.policy_ids, counts_for_ratios(condition.ratio_targets, config.array_size), strict=True))
        condition_rows.append(
            {
                "condition_id": condition.condition_id,
                "symmetry_group_id": condition.symmetry_group_id,
                "base_contest_id": condition.base_contest.base_contest_id,
                "source_score_id": condition.base_contest.source_score_id,
                "source_condition_id": condition.base_contest.source_condition_id,
                "source_kind": condition.base_contest.source_kind,
                "selection_rank": int(condition.base_contest.selection_rank),
                "panel": condition.base_contest.panel,
                "orientation": condition.orientation,
                "arrangement": condition.arrangement,
                "value_profile": condition.value_profile,
                "perturbation_profile": condition.perturbation_profile,
                "goal_profile_id": OPPOSITE_PROFILE.profile_id,
                "goal_compatibility_class": OPPOSITE_PROFILE.compatibility_class,
                "policy_ids_json": _json_list(condition.policy_ids),
                "display_names_json": _json_list(condition.display_names),
                "source_categories_json": _json_list(condition.source_categories),
                "ratio_targets_json": _json_list(condition.ratio_targets),
                "goal_assignments_json": _json_object(goal_assignments),
                "expected_counts_json": _json_object(expected_counts),
                "s05_goal_profile_id": condition.base_contest.s05_goal_profile_id,
                "s05_goal_compatibility_class": condition.base_contest.s05_goal_compatibility_class,
                "s05_priority_score": float(condition.base_contest.s05_priority_score),
            }
        )
    return run_df, pd.DataFrame(condition_rows), base_contests


def summarize_s06_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if run_df.empty:
        return pd.DataFrame()
    for condition_id, group in run_df.groupby("condition_id", sort=False):
        first = group.iloc[0]
        winner_counts = group["dominant_policy_id"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "symmetry_group_id": first["symmetry_group_id"],
                "base_contest_id": first["base_contest_id"],
                "source_kind": first["source_kind"],
                "selection_rank": int(first["selection_rank"]),
                "panel": first["panel"],
                "orientation": first["orientation"],
                "arrangement": first["arrangement"],
                "value_profile": first["value_profile"],
                "perturbation_profile": first["perturbation_profile"],
                "policy_ids_json": first["policy_ids_json"],
                "display_names_json": first["display_names_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "seed_count": int(group["seed"].nunique()),
                "run_count": int(len(group)),
                "mean_global_goal_margin_first_minus_second": float(group["global_goal_margin_first_minus_second"].mean()),
                "mean_dominance_margin_abs": float(group["dominance_margin_abs"].mean()),
                "dominant_policy_mode": str(group["dominant_policy_id"].mode().iloc[0]),
                "dominant_policy_counts_json": _json_object(winner_counts),
                "mean_assigned_goal_margin_first_minus_second": float(group["assigned_goal_margin_first_minus_second"].mean()),
                "mean_goal_alignment_gap": float(group["goal_alignment_gap"].mean()),
                "mean_reference_increasing_score": float(group["reference_increasing_score"].mean()),
                "mean_final_inversion_sortedness": float(group["final_inversion_sortedness"].mean()),
                "mean_aggregation_delta_percent": float(group["aggregation_delta_percent"].mean()),
                "mean_position_bias_margin": float(group["position_bias_margin"].mean()),
                "mean_frozen_block_count": float(group["frozen_block_count"].mean()),
                "mean_work_count": float(group["work_count"].mean()),
                "invalid_action_count": int(group["invalid_action_count"].sum()),
                "final_state_class_mode": str(group["final_state_class"].mode().iloc[0]),
                "goal_state_class_mode": str(group["goal_state_class"].mode().iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def dominance_hierarchy(run_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    id_to_display = {str(row["algotype_id"]): str(row["display_name"]) for row in metadata.to_dict(orient="records")}
    id_to_category = {str(row["algotype_id"]): str(row["source_category"]) for row in metadata.to_dict(orient="records")}
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "net": 0.0, "abs": [], "rows": 0})
    for row in run_df.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in _parse_json_list(row["policy_ids_json"]))
        winner = str(row["dominant_policy_id"])
        loser = str(row["dominated_policy_id"])
        margin = float(row["dominance_margin_abs"])
        for policy_id in policy_ids:
            stats[policy_id]["rows"] += 1
            stats[policy_id]["abs"].append(margin)
        if winner == "tie":
            for policy_id in policy_ids:
                stats[policy_id]["ties"] += 1
            continue
        stats[winner]["wins"] += 1
        stats[winner]["net"] += margin
        stats[loser]["losses"] += 1
        stats[loser]["net"] -= margin
    rows: list[dict[str, Any]] = []
    for policy_id, stat in stats.items():
        rows.append(
            {
                "schema": HIERARCHY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "policy_id": policy_id,
                "display_name": id_to_display.get(policy_id, policy_id),
                "source_category": id_to_category.get(policy_id, "unknown"),
                "contest_run_count": int(stat["rows"]),
                "dominance_wins": int(stat["wins"]),
                "dominance_losses": int(stat["losses"]),
                "dominance_ties": int(stat["ties"]),
                "win_fraction": float(stat["wins"] / max(1, stat["wins"] + stat["losses"] + stat["ties"])),
                "net_dominance_score": float(stat["net"]),
                "mean_abs_dominance_margin": float(np.mean(stat["abs"])) if stat["abs"] else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["net_dominance_score", "win_fraction"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def context_dependence(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary_df.empty:
        return pd.DataFrame()
    for base_contest_id, group in summary_df.groupby("base_contest_id", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "base_contest_id": base_contest_id,
                "source_kind": first["source_kind"],
                "panel": first["panel"],
                "display_names_json": first["display_names_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "condition_count": int(len(group)),
                "dominant_policy_modes_json": _json_object(group["dominant_policy_mode"].value_counts().sort_index().to_dict()),
                "dominance_margin_range": float(group["mean_global_goal_margin_first_minus_second"].max() - group["mean_global_goal_margin_first_minus_second"].min()),
                "abs_dominance_margin_range": float(group["mean_dominance_margin_abs"].max() - group["mean_dominance_margin_abs"].min()),
                "aggregation_delta_range": float(group["mean_aggregation_delta_percent"].max() - group["mean_aggregation_delta_percent"].min()),
                "position_bias_margin_range": float(group["mean_position_bias_margin"].max() - group["mean_position_bias_margin"].min()),
                "context_dependent_dominance": bool(group["dominant_policy_mode"].nunique() > 1 or (group["mean_global_goal_margin_first_minus_second"].max() - group["mean_global_goal_margin_first_minus_second"].min()) >= 0.25),
            }
        )
    return pd.DataFrame(rows).sort_values(["context_dependent_dominance", "dominance_margin_range"], ascending=[False, False], kind="mergesort")


def relabeling_metric_control() -> bool:
    labels = ("a", "a", "b", "b", "b")
    swapped = tuple("b" if label == "a" else "a" for label in labels)
    base = dominance_metrics(labels, ("a", "b"))
    mapped = dominance_metrics(swapped, ("b", "a"))
    return (
        base["leftmost_policy_id"] == "a"
        and base["rightmost_policy_id"] == "b"
        and mapped["leftmost_policy_id"] == "b"
        and mapped["rightmost_policy_id"] == "a"
        and abs(float(base["position_bias_margin"]) - float(mapped["position_bias_margin"])) <= 1e-12
    )


def validate_s06_outputs(
    run_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    base_contests: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    config: S06Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    seed_set = set(int(seed) for seed in config.seeds)
    required_columns = {
        "condition_id",
        "symmetry_group_id",
        "seed",
        "orientation",
        "value_profile",
        "perturbation_profile",
        "policy_ids_json",
        "goal_assignments_json",
        "relevant_goal_names_json",
        "goal_scores_json",
        "assigned_policy_mean_sortedness",
        "assigned_adjacent_satisfaction",
        "dominant_policy_id",
        "global_goal_margin_first_minus_second",
        "realized_ratio_max_abs_error",
        "count_preservation_success",
        "invalid_action_count",
    }
    has_columns = required_columns.issubset(run_df.columns)
    seed_success = False
    ratio_success = False
    opposite_success = False
    final_eval_success = False
    orientation_success = False
    relabel_run_success = False
    context_success = False
    originals_success = False
    if has_columns and not run_df.empty:
        seed_sets = run_df.groupby("condition_id")["seed"].agg(lambda values: set(int(value) for value in values))
        seed_success = bool((seed_sets == seed_set).all())
        ratio_success = bool((run_df["realized_ratio_max_abs_error"] <= 1e-12).all() and run_df["count_preservation_success"].all())
        opposite_success = bool(
            run_df["goal_assignments_json"].map(lambda text: set(json.loads(str(text)).values()) == {"increasing", "decreasing"}).all()
        )
        eval_flags = []
        for row in run_df.to_dict(orient="records"):
            relevant = set(json.loads(str(row["relevant_goal_names_json"])))
            scores = json.loads(str(row["goal_scores_json"]))
            eval_flags.append(
                relevant == {"increasing", "decreasing"}
                and relevant.issubset(set(scores))
                and all(math.isfinite(float(scores[name])) for name in relevant)
                and math.isfinite(float(row["assigned_policy_mean_sortedness"]))
                and math.isfinite(float(row["assigned_adjacent_satisfaction"]))
            )
        final_eval_success = bool(eval_flags and all(eval_flags))
        orientation_sets = condition_df.groupby(["symmetry_group_id"])["orientation"].agg(lambda values: set(map(str, values)))
        orientation_success = bool(not orientation_sets.empty and all(item == set(config.orientations) for item in orientation_sets))
        relabel_pair_checks = []
        for _, group in run_df.groupby(["symmetry_group_id", "seed"], sort=False):
            orientations = set(group["orientation"].astype(str))
            policy_sets = {frozenset(_parse_json_list(text)) for text in group["policy_ids_json"].tolist()}
            relabel_pair_checks.append(orientations == set(config.orientations) and len(policy_sets) == 1)
        relabel_run_success = bool(relabel_pair_checks and all(relabel_pair_checks))
        context_success = bool(set(config.value_profiles).issubset(set(condition_df["value_profile"])) and set(config.perturbation_profiles).issubset(set(condition_df["perturbation_profile"])))
        original_names = set()
        for text in condition_df["display_names_json"].tolist():
            original_names.update(str(item) for item in _parse_json_list(text) if str(item).startswith("original_"))
        originals_success = {"original_bubble", "original_insertion", "original_selection"}.issubset(original_names)
    checks = [
        {
            "validation_case": "required_run_columns_present",
            "success": has_columns,
            "observed": f"columns={len(run_df.columns) if not run_df.empty else 0}",
            "expected": "S06 run table contains dominance, goal, ratio, and symmetry fields",
        },
        {
            "validation_case": "bounded_s05_candidate_table_used",
            "success": bool(not base_contests.empty and len(base_contests) <= config.max_s05_candidates + 3),
            "observed": f"base_contests={len(base_contests)} max={config.max_s05_candidates + 3}",
            "expected": "bounded S05-selected contest set plus up to three paper-original controls",
        },
        {
            "validation_case": "paper_originals_included",
            "success": originals_success,
            "observed": "original Bubble/Insertion/Selection present" if originals_success else "missing at least one paper original",
            "expected": "paper original Bubble, Insertion, and Selection Algotypes are included",
        },
        {
            "validation_case": "opposite_goal_assignments_valid",
            "success": opposite_success,
            "observed": "all rows have one increasing and one decreasing assignment" if opposite_success else "invalid goal assignments",
            "expected": "every S06 contest is an opposite-goal contest",
        },
        {
            "validation_case": "final_state_evaluated_under_relevant_goals",
            "success": final_eval_success,
            "observed": "all relevant goal scores and assigned-goal scores finite" if final_eval_success else "missing or non-finite relevant goal score",
            "expected": "every S06 run row contains finite final-state scores for increasing and decreasing goals",
        },
        {
            "validation_case": "matched_seed_sets_by_condition",
            "success": seed_success,
            "observed": f"conditions={condition_df['condition_id'].nunique() if not condition_df.empty else 0}; expected_seeds={sorted(seed_set)}",
            "expected": "each condition uses the configured matched seed set",
        },
        {
            "validation_case": "matched_relabel_orientation_pairs",
            "success": orientation_success,
            "observed": f"symmetry_groups={condition_df['symmetry_group_id'].nunique() if not condition_df.empty else 0}",
            "expected": "each symmetry group has as-selected and relabel-swap orientations",
        },
        {
            "validation_case": "matched_relabel_run_pairs_by_seed",
            "success": relabel_run_success,
            "observed": f"symmetry_seed_groups={run_df.groupby(['symmetry_group_id', 'seed']).ngroups if has_columns and not run_df.empty else 0}",
            "expected": "each symmetry group and seed has matched orientation rows over the same policy set",
        },
        {
            "validation_case": "value_and_perturbation_contexts_present",
            "success": context_success,
            "observed": f"value_profiles={sorted(set(condition_df['value_profile'])) if not condition_df.empty else []}; perturbations={sorted(set(condition_df['perturbation_profile'])) if not condition_df.empty else []}",
            "expected": "configured value and perturbation context profiles are represented",
        },
        {
            "validation_case": "dominance_metric_relabeling_control",
            "success": relabeling_metric_control(),
            "observed": str(relabeling_metric_control()),
            "expected": "position-dominance metric maps left/right winners correctly under label relabeling",
        },
        {
            "validation_case": "realized_ratios_and_counts_match",
            "success": ratio_success,
            "observed": f"max_error={run_df['realized_ratio_max_abs_error'].max():.3g}; count_failures={int((~run_df['count_preservation_success']).sum())}" if has_columns and not run_df.empty else "missing run rows",
            "expected": "zero realized-ratio error and preserved counts",
        },
        {
            "validation_case": "no_invalid_actions",
            "success": bool(has_columns and int(run_df["invalid_action_count"].sum()) == 0),
            "observed": str(int(run_df["invalid_action_count"].sum())) if has_columns and not run_df.empty else "missing",
            "expected": "zero invalid actions; frozen perturbations are counted separately as blocked moves",
        },
        {
            "validation_case": "hierarchy_rows_written",
            "success": bool(not hierarchy_df.empty and {"policy_id", "net_dominance_score"}.issubset(hierarchy_df.columns)),
            "observed": f"hierarchy_rows={len(hierarchy_df)}",
            "expected": "policy-level dominance hierarchy exists",
        },
        {
            "validation_case": "figure_written",
            "success": bool(figure_written),
            "observed": str(bool(figure_written)),
            "expected": "dominance network figure exists",
        },
        {
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests_success),
            "observed": str(bool(unit_tests_success)),
            "expected": "focused E06 unit tests pass",
        },
    ]
    return pd.DataFrame(checks)
