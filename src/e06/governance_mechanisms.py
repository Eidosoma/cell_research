"""E06 S09 local governance-mechanism intervention sweeps."""

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
from src.e06.goal_compatibility import (
    GoalProfile,
    evaluate_goal_state,
    goal_assignments_for_profile,
    rank_values,
)
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


STEP_ID = "S09"
GOVERNANCE_SCHEMA = "eidosoma.e06.s09_governance_mechanism_run.v1"
SUMMARY_SCHEMA = "eidosoma.e06.s09_governance_summary.v1"
COMPARISON_SCHEMA = "eidosoma.e06.s09_governance_baseline_comparison.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s09_validation.v1"
CONFLICT_CLASSES = ("opposite_goal", "partially_compatible", "unrelated_goal")


@dataclass(frozen=True)
class GovernanceMechanism:
    """One governance overlay with explicit information-access metadata."""

    mechanism_id: str
    mechanism_family: str
    influence_radius_cells: int | None
    influence_radius_label: str
    information_access_class: str
    global_controller_like: bool
    uses_label_identity: bool
    uses_goal_proxy: bool
    description: str


DEFAULT_GOVERNANCE_MECHANISMS = (
    GovernanceMechanism(
        mechanism_id="no_governance",
        mechanism_family="behavior_baseline",
        influence_radius_cells=0,
        influence_radius_label="none",
        information_access_class="actor_local_policy_only",
        global_controller_like=False,
        uses_label_identity=False,
        uses_goal_proxy=False,
        description="No governance overlay; policy proposals pass directly to the S08 interface stratum.",
    ),
    GovernanceMechanism(
        mechanism_id="local_voting_radius1",
        mechanism_family="local_voting",
        influence_radius_cells=1,
        influence_radius_label="radius_1",
        information_access_class="local_labels_radius_1_and_actor_goal_proxy",
        global_controller_like=False,
        uses_label_identity=True,
        uses_goal_proxy=True,
        description="Radius-1 neighborhood vote gates swaps that reduce local label support unless actor-goal satisfaction improves.",
    ),
    GovernanceMechanism(
        mechanism_id="leader_cells_radius2",
        mechanism_family="leader_cells",
        influence_radius_cells=2,
        influence_radius_label="radius_2_sparse_fixed_leaders",
        information_access_class="fixed_leader_patch_local_values_radius_2",
        global_controller_like=False,
        uses_label_identity=False,
        uses_goal_proxy=True,
        description="Sparse fixed leader positions influence only nearby swaps using the leader policy's local goal proxy.",
    ),
    GovernanceMechanism(
        mechanism_id="pacemaker_cells_radius2",
        mechanism_family="pacemaker_cells",
        influence_radius_cells=2,
        influence_radius_label="radius_2_sparse_pacemakers",
        information_access_class="fixed_pacemaker_patch_local_values_radius_2",
        global_controller_like=False,
        uses_label_identity=False,
        uses_goal_proxy=True,
        description="Sparse local pacemakers encourage nearby swaps that improve the shared increasing-order proxy.",
    ),
    GovernanceMechanism(
        mechanism_id="quorum_signal_radius2",
        mechanism_family="quorum_signal",
        influence_radius_cells=2,
        influence_radius_label="radius_2",
        information_access_class="local_label_density_radius_2",
        global_controller_like=False,
        uses_label_identity=True,
        uses_goal_proxy=False,
        description="Radius-2 quorum signal activates in high heterotypic-density patches and blocks swaps that increase local mixing.",
    ),
    GovernanceMechanism(
        mechanism_id="conflict_resolution_radius2",
        mechanism_family="conflict_resolution_signal",
        influence_radius_cells=2,
        influence_radius_label="radius_2",
        information_access_class="local_goal_proxy_radius_2_for_actor_and_target",
        global_controller_like=False,
        uses_label_identity=False,
        uses_goal_proxy=True,
        description="Radius-2 conflict-resolution signal accepts swaps that preserve or improve local satisfaction of both assigned goals.",
    ),
    GovernanceMechanism(
        mechanism_id="organizer_patch_radius3",
        mechanism_family="limited_organizer_cells",
        influence_radius_cells=3,
        influence_radius_label="radius_3_sparse_organizers",
        information_access_class="fixed_organizer_patch_local_values_and_labels_radius_3",
        global_controller_like=False,
        uses_label_identity=True,
        uses_goal_proxy=True,
        description="Sparse organizer patches combine local shared-goal improvement with local interface-density limits.",
    ),
    GovernanceMechanism(
        mechanism_id="global_reference_controller",
        mechanism_family="global_controller_comparator",
        influence_radius_cells=None,
        influence_radius_label="global",
        information_access_class="global_array_state_and_reference_goal",
        global_controller_like=True,
        uses_label_identity=False,
        uses_goal_proxy=True,
        description="Explicitly labeled global comparator that can inspect whole-array increasing sortedness before accepting a swap.",
    ),
)


@dataclass(frozen=True)
class S09Config:
    """Configuration for bounded S09 governance-mechanism tests."""

    array_size: int = 100
    event_cap: int = 4_000
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    max_base_conditions: int = 8
    governance_mechanisms: tuple[GovernanceMechanism, ...] = DEFAULT_GOVERNANCE_MECHANISMS
    interface_rules: tuple[InterfaceRule, ...] = DEFAULT_INTERFACE_RULES
    scheduler: str = "cyclic_scan_seed_offset"
    rescue_delta_threshold: float = 0.01
    cost_units_per_global_invocation: int = 100
    worker_count: int = 1


@dataclass(frozen=True)
class GovernanceBaseCondition:
    """One S04 conflict case selected for governance replay."""

    base_condition_id: str
    source_s04_condition_id: str
    selection_rank: int
    selection_reason: str
    panel: str
    candidate_reason: str
    arrangement: str
    arrangement_role: str
    goal_profile_id: str
    goal_compatibility_class: str
    policy_ids: tuple[str, str]
    display_names: tuple[str, str]
    source_categories: tuple[str, str]
    ratio_targets: tuple[float, float]
    value_profile: str
    perturbation_profile: str
    source_mean_final_target_quality: float
    source_mean_reference_increasing_score: float
    source_mean_goal_conflict_index: float
    source_mean_aggregation_delta_percent: float
    source_goal_state_class_mode: str


@dataclass(frozen=True)
class GovernanceCondition:
    """One executable S09 condition before seed expansion."""

    condition_id: str
    base: GovernanceBaseCondition
    interface_rule: InterfaceRule
    governance: GovernanceMechanism


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


def _mode(values: pd.Series) -> str:
    if values.empty:
        return ""
    mode = values.astype(str).mode()
    return str(mode.iloc[0]) if not mode.empty else str(values.iloc[0])


def _base_condition_id(row: Mapping[str, Any]) -> str:
    payload = {
        "sourceS04ConditionId": str(row["condition_id"]),
        "policyIds": _parse_json_list(row["policy_ids_json"]),
        "ratios": [round(float(value), 6) for value in _parse_json_list(row["ratio_targets_json"])],
        "arrangement": str(row["arrangement"]),
        "goalProfileId": str(row["goal_profile_id"]),
    }
    return f"s09_base_{stable_hash(payload)[:16]}"


def _condition_id(base_condition_id: str, interface_rule_id: str, governance_id: str) -> str:
    payload = {
        "baseConditionId": base_condition_id,
        "interfaceRuleId": interface_rule_id,
        "governanceMechanismId": governance_id,
    }
    return f"s09_{stable_hash(payload)[:16]}"


def _selection_score(row: Mapping[str, Any]) -> float:
    goal_gap = _safe_float(row.get("source_mean_goal_conflict_index"))
    reference_loss = 1.0 - _safe_float(row.get("source_mean_reference_increasing_score"))
    aggregation = abs(_safe_float(row.get("source_mean_aggregation_delta_percent"))) / 100.0
    class_bonus = 0.15 if str(row.get("source_goal_state_class_mode")) in {"polarized_goal_conflict", "mixed_goal_tension"} else 0.0
    return float(goal_gap + 0.5 * reference_loss + 0.2 * aggregation + class_bonus)


def select_s09_conflict_cases(s04_runs: pd.DataFrame, config: S09Config) -> pd.DataFrame:
    """Select a bounded, diverse set of S04 conflict cases for governance replay."""

    required = {
        "condition_id",
        "panel",
        "candidate_reason",
        "arrangement",
        "arrangement_role",
        "goal_profile_id",
        "goal_compatibility_class",
        "policy_ids_json",
        "display_names_json",
        "source_categories_json",
        "ratio_targets_json",
        "assigned_policy_mean_sortedness",
        "reference_increasing_score",
        "goal_alignment_gap",
        "aggregation_delta_percent",
        "goal_state_class",
    }
    missing = required - set(s04_runs.columns)
    if missing:
        raise ValueError(f"S04 goal-compatibility table missing columns: {sorted(missing)}")
    conflict = s04_runs[s04_runs["goal_compatibility_class"].isin(CONFLICT_CLASSES)].copy()
    if conflict.empty:
        raise ValueError("S09 could not find S04 conflict-class rows")

    grouping = [
        "condition_id",
        "panel",
        "candidate_reason",
        "arrangement",
        "arrangement_role",
        "goal_profile_id",
        "goal_compatibility_class",
        "policy_ids_json",
        "display_names_json",
        "source_categories_json",
        "ratio_targets_json",
    ]
    summary = (
        conflict.groupby(grouping, dropna=False)
        .agg(
            source_mean_final_target_quality=("assigned_policy_mean_sortedness", "mean"),
            source_mean_reference_increasing_score=("reference_increasing_score", "mean"),
            source_mean_goal_conflict_index=("goal_alignment_gap", "mean"),
            source_mean_aggregation_delta_percent=("aggregation_delta_percent", "mean"),
            source_goal_state_class_mode=("goal_state_class", _mode),
            seed_count=("seed", "nunique"),
        )
        .reset_index()
    )
    summary["selection_score"] = summary.apply(_selection_score, axis=1)
    summary = summary.sort_values(["selection_score", "source_mean_goal_conflict_index"], ascending=[False, False], kind="mergesort")

    selected_indices: list[int] = []

    def add(indexes: Sequence[int]) -> None:
        for idx in indexes:
            if idx not in selected_indices:
                selected_indices.append(int(idx))
            if len(selected_indices) >= config.max_base_conditions:
                return

    for conflict_class in CONFLICT_CLASSES:
        class_rows = summary[summary["goal_compatibility_class"] == conflict_class]
        add(class_rows.head(2).index.tolist())
    add(summary.index.tolist())

    selected = summary.loc[selected_indices[: config.max_base_conditions]].copy()
    selected = selected.sort_values(["selection_score", "source_mean_goal_conflict_index"], ascending=[False, False], kind="mergesort")
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected["base_condition_id"] = selected.apply(_base_condition_id, axis=1)
    selected["selection_reason"] = selected.apply(
        lambda row: f"s04_conflict_case:{row['goal_compatibility_class']}:score={row['selection_score']:.4f}",
        axis=1,
    )
    selected["value_profile"] = "random_permutation"
    selected["perturbation_profile"] = "none"
    return selected.reset_index(drop=True)


def build_s09_conditions(selected: pd.DataFrame, config: S09Config) -> list[GovernanceCondition]:
    conditions: list[GovernanceCondition] = []
    for raw in selected.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
        display_names = tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
        categories = tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
        ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
        if len(policy_ids) != 2 or len(ratios) != 2:
            continue
        base = GovernanceBaseCondition(
            base_condition_id=str(raw["base_condition_id"]),
            source_s04_condition_id=str(raw["condition_id"]),
            selection_rank=int(raw["selection_rank"]),
            selection_reason=str(raw["selection_reason"]),
            panel=str(raw["panel"]),
            candidate_reason=str(raw["candidate_reason"]),
            arrangement=str(raw["arrangement"]),
            arrangement_role=str(raw["arrangement_role"]),
            goal_profile_id=str(raw["goal_profile_id"]),
            goal_compatibility_class=str(raw["goal_compatibility_class"]),
            policy_ids=(policy_ids[0], policy_ids[1]),
            display_names=(display_names[0], display_names[1]),
            source_categories=(categories[0], categories[1]),
            ratio_targets=(ratios[0], ratios[1]),
            value_profile=str(raw["value_profile"]),
            perturbation_profile=str(raw["perturbation_profile"]),
            source_mean_final_target_quality=float(raw["source_mean_final_target_quality"]),
            source_mean_reference_increasing_score=float(raw["source_mean_reference_increasing_score"]),
            source_mean_goal_conflict_index=float(raw["source_mean_goal_conflict_index"]),
            source_mean_aggregation_delta_percent=float(raw["source_mean_aggregation_delta_percent"]),
            source_goal_state_class_mode=str(raw["source_goal_state_class_mode"]),
        )
        for interface_rule in config.interface_rules:
            for governance in config.governance_mechanisms:
                conditions.append(
                    GovernanceCondition(
                        condition_id=_condition_id(base.base_condition_id, interface_rule.rule_id, governance.mechanism_id),
                        base=base,
                        interface_rule=interface_rule,
                        governance=governance,
                    )
                )
    return conditions


def _paper_scope(rule: InterfaceRule) -> str:
    if not rule.explicit_recognition_used:
        return "original_behavior_only_no_explicit_recognition"
    return "explicit_interface_intervention_beyond_original_no_recognition_claim"


def _window_indices(n: int, center: int, radius: int) -> tuple[int, ...]:
    start = max(0, int(center) - int(radius))
    stop = min(n, int(center) + int(radius) + 1)
    return tuple(range(start, stop))


def _neighborhood_indices(n: int, actor_index: int, target_index: int, radius: int) -> tuple[int, ...]:
    values = set(_window_indices(n, actor_index, radius))
    values.update(_window_indices(n, target_index, radius))
    return tuple(sorted(values))


def _edges_for_indices(n: int, indices: Sequence[int]) -> tuple[int, ...]:
    edges: set[int] = set()
    for idx in indices:
        for edge in (int(idx) - 1, int(idx)):
            if 0 <= edge < n - 1:
                edges.add(edge)
    return tuple(sorted(edges))


def _same_edge_count(labels: Sequence[str], edges: Sequence[int]) -> int:
    return sum(1 for edge in edges if str(labels[edge]) == str(labels[edge + 1]))


def _heterotypic_edge_count(labels: Sequence[str], edges: Sequence[int]) -> int:
    return sum(1 for edge in edges if str(labels[edge]) != str(labels[edge + 1]))


def _swap_sequence(values: Sequence[Any], actor_index: int, target_index: int) -> list[Any]:
    swapped = list(values)
    swapped[actor_index], swapped[target_index] = swapped[target_index], swapped[actor_index]
    return swapped


def _goal_edge_satisfaction(
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


def _goal_edge_satisfaction_for_goal(values: Sequence[int], edges: Sequence[int], goal_name: str, array_size: int) -> float:
    if not edges:
        return 1.0
    satisfied = 0
    for edge in edges:
        ranks = rank_values((values[edge], values[edge + 1]), goal_name, array_size)
        satisfied += int(ranks[0] <= ranks[1])
    return float(satisfied / len(edges))


def _global_inversion_sortedness_delta_for_swap(values: Sequence[int], actor_index: int, target_index: int) -> float:
    """Exact whole-array inversion-sortedness delta for swapping two positions."""

    i = int(actor_index)
    j = int(target_index)
    if i == j:
        return 0.0
    n = len(values)
    total_pairs = n * (n - 1) / 2
    if total_pairs <= 0:
        return 0.0
    before = 0
    after = 0
    swapped = _swap_sequence(values, i, j)
    affected_pairs: set[tuple[int, int]] = set()
    for idx in (i, j):
        for other in range(n):
            if other == idx:
                continue
            affected_pairs.add((min(idx, other), max(idx, other)))
    for left, right in affected_pairs:
        before += int(int(values[left]) > int(values[right]))
        after += int(int(swapped[left]) > int(swapped[right]))
    return float((before - after) / total_pairs)


def _local_label_share(labels: Sequence[str], indices: Sequence[int], label: str) -> float:
    if not indices:
        return 0.0
    return float(sum(1 for idx in indices if str(labels[idx]) == str(label)) / len(indices))


def _near_anchor(indices: Sequence[int], anchors: Sequence[int], radius: int) -> int | None:
    for anchor in anchors:
        if any(abs(int(idx) - int(anchor)) <= int(radius) for idx in indices):
            return int(anchor)
    return None


def _governance_cost_units(mechanism: GovernanceMechanism, array_size: int) -> int:
    if mechanism.global_controller_like:
        return int(array_size)
    if mechanism.influence_radius_cells is None:
        return int(array_size)
    return max(1, 2 * int(mechanism.influence_radius_cells) + 1)


def governance_decision(
    mechanism: GovernanceMechanism,
    *,
    values: Sequence[int],
    labels: Sequence[str],
    actor_index: int,
    target_index: int,
    goal_assignments: Mapping[str, str],
    policy_ids: Sequence[str],
    array_size: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Return whether a valid proposed swap passes the governance overlay."""

    actor_label = str(labels[actor_index])
    target_label = str(labels[target_index])
    cross_label = actor_label != target_label
    baseline = {
        "accepted": True,
        "invoked": False,
        "block_reason": "",
        "local_goal_before": math.nan,
        "local_goal_after": math.nan,
        "same_edge_before": math.nan,
        "same_edge_after": math.nan,
        "heterotypic_edge_before": math.nan,
        "heterotypic_edge_after": math.nan,
        "label_share_before": math.nan,
        "label_share_after": math.nan,
        "global_reference_before": math.nan,
        "global_reference_after": math.nan,
        "cross_label": cross_label,
        "cost_units": 0,
    }
    if mechanism.mechanism_id == "no_governance":
        return baseline

    n = len(values)
    radius = int(mechanism.influence_radius_cells or 0)
    indices = _neighborhood_indices(n, actor_index, target_index, radius) if radius > 0 else (actor_index, target_index)
    edges = _edges_for_indices(n, indices)
    after_values = _swap_sequence(values, actor_index, target_index)
    after_labels = _swap_sequence(labels, actor_index, target_index)
    before_goal = _goal_edge_satisfaction(values, labels, edges, goal_assignments, array_size)
    after_goal = _goal_edge_satisfaction(after_values, after_labels, edges, goal_assignments, array_size)
    before_same = _same_edge_count(labels, edges)
    after_same = _same_edge_count(after_labels, edges)
    before_hetero = _heterotypic_edge_count(labels, edges)
    after_hetero = _heterotypic_edge_count(after_labels, edges)
    before_share = _local_label_share(labels, indices, actor_label)
    after_share = _local_label_share(after_labels, indices, actor_label)
    accepted = True
    invoked = True
    block_reason = ""
    global_before = math.nan
    global_after = math.nan

    if mechanism.mechanism_id == "local_voting_radius1":
        accepted = (after_share >= before_share) or (after_goal > before_goal + 1e-12) or (rng.random() < 0.25)
        block_reason = "" if accepted else "local_vote_rejects_label_or_goal_loss"
    elif mechanism.mechanism_id == "leader_cells_radius2":
        anchor = _near_anchor(indices, tuple(range(0, n, 25)), radius)
        invoked = anchor is not None
        if invoked:
            leader_label = str(labels[anchor])
            leader_goal = goal_assignments.get(leader_label, goal_assignments[actor_label])
            before_leader = _goal_edge_satisfaction_for_goal(values, edges, leader_goal, array_size)
            after_leader = _goal_edge_satisfaction_for_goal(after_values, edges, leader_goal, array_size)
            before_goal, after_goal = before_leader, after_leader
            accepted = after_leader >= before_leader - 1e-12 or rng.random() < 0.15
            block_reason = "" if accepted else "leader_patch_goal_loss"
    elif mechanism.mechanism_id == "pacemaker_cells_radius2":
        anchors = (n // 4, n // 2, (3 * n) // 4)
        anchor = _near_anchor(indices, anchors, radius)
        invoked = anchor is not None
        if invoked:
            before_pace = _goal_edge_satisfaction_for_goal(values, edges, "increasing", array_size)
            after_pace = _goal_edge_satisfaction_for_goal(after_values, edges, "increasing", array_size)
            before_goal, after_goal = before_pace, after_pace
            accepted = after_pace >= before_pace - 1e-12 or rng.random() < 0.10
            block_reason = "" if accepted else "pacemaker_shared_goal_loss"
    elif mechanism.mechanism_id == "quorum_signal_radius2":
        quorum_fraction = before_hetero / max(1, len(edges))
        invoked = quorum_fraction >= 0.35
        if invoked:
            accepted = after_hetero <= before_hetero or rng.random() < 0.20
            block_reason = "" if accepted else "quorum_signal_blocks_extra_mixing"
    elif mechanism.mechanism_id == "conflict_resolution_radius2":
        accepted = after_goal >= before_goal - 1e-12 or rng.random() < 0.10
        block_reason = "" if accepted else "conflict_resolution_goal_loss"
    elif mechanism.mechanism_id == "organizer_patch_radius3":
        anchors = tuple(range(10, n, 20))
        anchor = _near_anchor(indices, anchors, radius)
        invoked = anchor is not None
        if invoked:
            before_shared = _goal_edge_satisfaction_for_goal(values, edges, "increasing", array_size)
            after_shared = _goal_edge_satisfaction_for_goal(after_values, edges, "increasing", array_size)
            before_goal, after_goal = before_shared, after_shared
            accepted = (after_shared >= before_shared - 1e-12 and after_hetero <= before_hetero + 1) or rng.random() < 0.10
            block_reason = "" if accepted else "organizer_patch_shared_goal_or_boundary_loss"
    elif mechanism.mechanism_id == "global_reference_controller":
        delta_sortedness = _global_inversion_sortedness_delta_for_swap(values, actor_index, target_index)
        global_before = 0.0
        global_after = float(delta_sortedness)
        before_goal, after_goal = global_before, global_after
        accepted = delta_sortedness >= -1e-12 or rng.random() < 0.05
        block_reason = "" if accepted else "global_reference_controller_blocks_reference_loss"
    else:
        raise ValueError(f"unknown S09 governance mechanism: {mechanism.mechanism_id}")

    if not invoked:
        accepted = True
        block_reason = ""

    return {
        "accepted": bool(accepted),
        "invoked": bool(invoked),
        "block_reason": block_reason,
        "local_goal_before": float(before_goal),
        "local_goal_after": float(after_goal),
        "same_edge_before": float(before_same),
        "same_edge_after": float(after_same),
        "heterotypic_edge_before": float(before_hetero),
        "heterotypic_edge_after": float(after_hetero),
        "label_share_before": float(before_share),
        "label_share_after": float(after_share),
        "global_reference_before": float(global_before),
        "global_reference_after": float(global_after),
        "cross_label": cross_label,
        "cost_units": _governance_cost_units(mechanism, array_size) if invoked else 0,
    }


def _block_state(memory: dict[int, dict[str, Any]], cell_id: int, action_type: str) -> None:
    state = memory.setdefault(cell_id, {})
    state["last_move_success"] = False
    state["last_action_type"] = action_type
    state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)


def simulate_governance_condition(
    condition: GovernanceCondition,
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: S09Config,
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
    rng = random.Random(seed ^ int(stable_hash({"s09": condition.condition_id})[:12], 16))
    goal_profile = goal_profile_for_id(condition.base.goal_profile_id)
    goal_assignments = goal_assignments_for_profile(condition.base.policy_ids, goal_profile)
    frozen_indices = frozen_indices_for_profile(config.array_size, condition.base.perturbation_profile)

    compare_count = 0
    swap_count = 0
    same_label_swap_count = 0
    cross_label_swap_count = 0
    raw_swap_proposal_count = 0
    pre_governance_cross_label_proposal_count = 0
    governance_invocation_count = 0
    governance_block_count = 0
    governance_cross_label_block_count = 0
    global_controller_invocation_count = 0
    governance_cost_units = 0
    same_label_swap_proposal_count = 0
    cross_label_swap_proposal_count = 0
    update_count = 0
    wait_count = 0
    invalid_action_count = 0
    frozen_block_count = 0
    interface_rule_invocation_count = 0
    interface_block_count = 0
    same_contact_delta_sum = 0.0
    local_goal_delta_sum = 0.0
    governance_local_goal_delta_sum = 0.0
    governance_same_edge_delta_sum = 0.0
    governance_heterotypic_edge_delta_sum = 0.0
    governance_block_reasons: Counter[str] = Counter()
    interface_block_reasons: Counter[str] = Counter()
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
            _block_state(memory, cell_id, "frozen_block")
            continue
        if action.action_type == "swap" and target is not None:
            if 0 <= target < len(values):
                raw_swap_proposal_count += 1
                pre_cross = str(labels[actor_index]) != str(labels[target])
                pre_governance_cross_label_proposal_count += int(pre_cross)
                governance = governance_decision(
                    condition.governance,
                    values=values,
                    labels=labels,
                    actor_index=actor_index,
                    target_index=target,
                    goal_assignments=goal_assignments,
                    policy_ids=condition.base.policy_ids,
                    array_size=config.array_size,
                    rng=rng,
                )
                if bool(governance["invoked"]):
                    governance_invocation_count += 1
                    governance_cost_units += int(governance["cost_units"])
                    if condition.governance.global_controller_like:
                        global_controller_invocation_count += 1
                    if math.isfinite(float(governance["local_goal_before"])):
                        governance_local_goal_delta_sum += float(governance["local_goal_after"]) - float(governance["local_goal_before"])
                    if math.isfinite(float(governance["same_edge_before"])):
                        governance_same_edge_delta_sum += float(governance["same_edge_after"]) - float(governance["same_edge_before"])
                    if math.isfinite(float(governance["heterotypic_edge_before"])):
                        governance_heterotypic_edge_delta_sum += float(governance["heterotypic_edge_after"]) - float(governance["heterotypic_edge_before"])
                if not bool(governance["accepted"]):
                    governance_block_count += 1
                    governance_cross_label_block_count += int(pre_cross)
                    reason = str(governance["block_reason"])
                    governance_block_reasons[reason] += 1
                    wait_count += 1
                    _block_state(memory, cell_id, f"governance_block:{reason}")
                    continue

                decision = interface_decision(
                    condition.interface_rule,
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
                    interface_block_reasons[reason] += 1
                    wait_count += 1
                    _block_state(memory, cell_id, f"interface_block:{reason}")
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
                _block_state(memory, cell_id, "invalid_swap")
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
        "schema": GOVERNANCE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "condition_id": condition.condition_id,
        "base_condition_id": condition.base.base_condition_id,
        "source_s04_condition_id": condition.base.source_s04_condition_id,
        "selection_rank": int(condition.base.selection_rank),
        "selection_reason": condition.base.selection_reason,
        "panel": condition.base.panel,
        "candidate_reason": condition.base.candidate_reason,
        "interface_rule_id": condition.interface_rule.rule_id,
        "interface_mechanism_family": condition.interface_rule.mechanism_family,
        "interface_rule_description": condition.interface_rule.description,
        "explicit_recognition_used": bool(condition.interface_rule.explicit_recognition_used),
        "recognition_access_class": condition.interface_rule.recognition_access_class,
        "paper_claim_scope": _paper_scope(condition.interface_rule),
        "no_recognition_baseline": not condition.interface_rule.explicit_recognition_used,
        "governance_mechanism_id": condition.governance.mechanism_id,
        "governance_mechanism_family": condition.governance.mechanism_family,
        "governance_description": condition.governance.description,
        "governance_influence_radius_cells": condition.governance.influence_radius_cells,
        "governance_influence_radius_label": condition.governance.influence_radius_label,
        "governance_information_access_class": condition.governance.information_access_class,
        "global_controller_like": bool(condition.governance.global_controller_like),
        "governance_uses_label_identity": bool(condition.governance.uses_label_identity),
        "governance_uses_goal_proxy": bool(condition.governance.uses_goal_proxy),
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
        "arrangement": condition.base.arrangement,
        "arrangement_role": condition.base.arrangement_role,
        "value_profile": condition.base.value_profile,
        "perturbation_profile": condition.base.perturbation_profile,
        "source_mean_final_target_quality": condition.base.source_mean_final_target_quality,
        "source_mean_reference_increasing_score": condition.base.source_mean_reference_increasing_score,
        "source_mean_goal_conflict_index": condition.base.source_mean_goal_conflict_index,
        "source_mean_aggregation_delta_percent": condition.base.source_mean_aggregation_delta_percent,
        "source_goal_state_class_mode": condition.base.source_goal_state_class_mode,
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
        "raw_swap_proposal_count": int(raw_swap_proposal_count),
        "pre_governance_cross_label_proposal_count": int(pre_governance_cross_label_proposal_count),
        "same_label_swap_proposal_count": int(same_label_swap_proposal_count),
        "cross_label_swap_proposal_count": int(cross_label_swap_proposal_count),
        "update_count": int(update_count),
        "wait_count": int(wait_count),
        "work_count": int(compare_count + swap_count + update_count),
        "invalid_action_count": int(invalid_action_count),
        "governance_invocation_count": int(governance_invocation_count),
        "governance_block_count": int(governance_block_count),
        "governance_cross_label_block_count": int(governance_cross_label_block_count),
        "governance_cost_units": int(governance_cost_units),
        "global_controller_invocation_count": int(global_controller_invocation_count),
        "governance_block_fraction": float(governance_block_count / max(1, raw_swap_proposal_count)),
        "governance_cost_per_event": float(governance_cost_units / max(1, config.event_cap)),
        "governance_local_goal_delta_per_invocation": float(governance_local_goal_delta_sum / max(1, governance_invocation_count)),
        "governance_same_edge_delta_per_invocation": float(governance_same_edge_delta_sum / max(1, governance_invocation_count)),
        "governance_heterotypic_edge_delta_per_invocation": float(governance_heterotypic_edge_delta_sum / max(1, governance_invocation_count)),
        "governance_block_reasons_json": _json_object(governance_block_reasons),
        "interface_rule_invocation_count": int(interface_rule_invocation_count),
        "interface_block_count": int(interface_block_count),
        "interface_block_reasons_json": _json_object(interface_block_reasons),
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


def run_s09_sweep(
    records: Sequence[Mapping[str, Any]],
    s04_runs: pd.DataFrame,
    s08_strata: pd.DataFrame,
    config: S09Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or S09Config()
    selected = select_s09_conflict_cases(s04_runs, config)
    observed_rules = set(s08_strata["interface_rule_id"].astype(str)) if "interface_rule_id" in s08_strata else set()
    missing_rules = {rule.rule_id for rule in config.interface_rules} - observed_rules
    if missing_rules:
        raise ValueError(f"S08 mechanism-strata table lacks required interface rules: {sorted(missing_rules)}")
    conditions = build_s09_conditions(selected, config)
    records_by_id = {str(record["algotypeId"]): dict(record) for record in records}
    missing = sorted({policy_id for condition in conditions for policy_id in condition.base.policy_ids} - set(records_by_id))
    if missing:
        raise ValueError(f"S09 selected policies missing from S01 library: {missing}")
    tasks = [(condition, records_by_id, int(seed), config) for condition in conditions for seed in config.seeds]
    if config.worker_count > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(config.worker_count)) as pool:
            rows = list(pool.map(_simulate_governance_task, tasks))
    else:
        rows = [_simulate_governance_task(task) for task in tasks]
    run_df = pd.DataFrame(rows)
    condition_rows = []
    for condition in conditions:
        expected_counts = dict(zip(condition.base.policy_ids, counts_for_ratios(condition.base.ratio_targets, config.array_size), strict=True))
        condition_rows.append(
            {
                "condition_id": condition.condition_id,
                "base_condition_id": condition.base.base_condition_id,
                "source_s04_condition_id": condition.base.source_s04_condition_id,
                "selection_rank": int(condition.base.selection_rank),
                "selection_reason": condition.base.selection_reason,
                "interface_rule_id": condition.interface_rule.rule_id,
                "interface_mechanism_family": condition.interface_rule.mechanism_family,
                "explicit_recognition_used": bool(condition.interface_rule.explicit_recognition_used),
                "recognition_access_class": condition.interface_rule.recognition_access_class,
                "governance_mechanism_id": condition.governance.mechanism_id,
                "governance_mechanism_family": condition.governance.mechanism_family,
                "governance_influence_radius_cells": condition.governance.influence_radius_cells,
                "governance_influence_radius_label": condition.governance.influence_radius_label,
                "governance_information_access_class": condition.governance.information_access_class,
                "global_controller_like": bool(condition.governance.global_controller_like),
                "policy_ids_json": _json_list(condition.base.policy_ids),
                "display_names_json": _json_list(condition.base.display_names),
                "source_categories_json": _json_list(condition.base.source_categories),
                "ratio_targets_json": _json_list(condition.base.ratio_targets),
                "expected_counts_json": _json_object(expected_counts),
                "arrangement": condition.base.arrangement,
                "arrangement_role": condition.base.arrangement_role,
                "goal_profile_id": condition.base.goal_profile_id,
                "goal_compatibility_class": condition.base.goal_compatibility_class,
                "value_profile": condition.base.value_profile,
                "perturbation_profile": condition.base.perturbation_profile,
            }
        )
    return run_df, pd.DataFrame(condition_rows), selected


def _simulate_governance_task(task: tuple[GovernanceCondition, Mapping[str, Mapping[str, Any]], int, S09Config]) -> dict[str, Any]:
    condition, records_by_id, seed, config = task
    return simulate_governance_condition(condition, records_by_id, seed=int(seed), config=config)


def summarize_s09_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if run_df.empty:
        return pd.DataFrame()
    group_columns = ["base_condition_id", "interface_rule_id", "governance_mechanism_id"]
    for _, group in run_df.groupby(group_columns, sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "base_condition_id": first["base_condition_id"],
                "source_s04_condition_id": first["source_s04_condition_id"],
                "interface_rule_id": first["interface_rule_id"],
                "interface_mechanism_family": first["interface_mechanism_family"],
                "explicit_recognition_used": bool(first["explicit_recognition_used"]),
                "recognition_access_class": first["recognition_access_class"],
                "governance_mechanism_id": first["governance_mechanism_id"],
                "governance_mechanism_family": first["governance_mechanism_family"],
                "governance_influence_radius_cells": first["governance_influence_radius_cells"],
                "governance_influence_radius_label": first["governance_influence_radius_label"],
                "governance_information_access_class": first["governance_information_access_class"],
                "global_controller_like": bool(first["global_controller_like"]),
                "selection_reason": first["selection_reason"],
                "display_names_json": first["display_names_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "arrangement": first["arrangement"],
                "goal_profile_id": first["goal_profile_id"],
                "goal_compatibility_class": first["goal_compatibility_class"],
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
                "mean_raw_swap_proposal_count": float(group["raw_swap_proposal_count"].mean()),
                "mean_pre_governance_cross_label_proposal_count": float(group["pre_governance_cross_label_proposal_count"].mean()),
                "mean_governance_invocation_count": float(group["governance_invocation_count"].mean()),
                "mean_governance_block_count": float(group["governance_block_count"].mean()),
                "mean_governance_block_fraction": float(group["governance_block_fraction"].mean()),
                "mean_governance_cost_units": float(group["governance_cost_units"].mean()),
                "mean_governance_cost_per_event": float(group["governance_cost_per_event"].mean()),
                "mean_global_controller_invocation_count": float(group["global_controller_invocation_count"].mean()),
                "mean_governance_local_goal_delta_per_invocation": float(group["governance_local_goal_delta_per_invocation"].mean()),
                "mean_governance_heterotypic_edge_delta_per_invocation": float(group["governance_heterotypic_edge_delta_per_invocation"].mean()),
                "mean_cross_label_swap_proposal_count": float(group["cross_label_swap_proposal_count"].mean()),
                "mean_cross_label_swap_count": float(group["cross_label_swap_count"].mean()),
                "mean_interface_rule_invocation_count": float(group["interface_rule_invocation_count"].mean()),
                "mean_interface_block_count": float(group["interface_block_count"].mean()),
                "mean_interface_block_fraction": float(group["interface_block_fraction"].mean()),
                "mean_cross_label_swap_acceptance_fraction": float(group["cross_label_swap_acceptance_fraction"].mean()),
                "invalid_action_count": int(group["invalid_action_count"].sum()),
                "count_preservation_success": bool(group["count_preservation_success"].all()),
                "final_state_class_mode": _mode(group["final_state_class"]),
                "goal_state_class_mode": _mode(group["goal_state_class"]),
            }
        )
    return pd.DataFrame(rows)


def compare_to_no_governance_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    baseline = summary[summary["governance_mechanism_id"] == "no_governance"].copy()
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
    baseline_columns = ["base_condition_id", "interface_rule_id", *metric_columns]
    baseline = baseline[baseline_columns].rename(columns={column: f"baseline_{column}" for column in metric_columns})
    comparison = summary.merge(baseline, on=["base_condition_id", "interface_rule_id"], how="left")
    for column in metric_columns:
        comparison[f"delta_{column.removeprefix('mean_')}"] = comparison[column] - comparison[f"baseline_{column}"]
    comparison["schema"] = COMPARISON_SCHEMA
    comparison["abs_delta_final_target_quality"] = comparison["delta_final_target_quality"].abs()
    comparison["abs_delta_goal_conflict_index"] = comparison["delta_goal_conflict_index"].abs()
    comparison["rescue_effect_score"] = (
        comparison["delta_final_target_quality"]
        - comparison["delta_goal_conflict_index"].clip(lower=0.0) * 0.25
        - comparison["mean_governance_block_fraction"] * 0.02
        - comparison["mean_governance_cost_per_event"] * 0.001
    )
    comparison["target_quality_rescued"] = comparison["delta_final_target_quality"] >= 0.01
    comparison["conflict_reduced"] = comparison["delta_goal_conflict_index"] <= -0.01
    return comparison


def governance_rankings(comparison: pd.DataFrame, config: S09Config) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for mechanism_id, group in comparison.groupby("governance_mechanism_id", sort=False):
        first = group.iloc[0]
        is_baseline = str(mechanism_id) == "no_governance"
        rescue_rate = float(group["target_quality_rescued"].mean()) if not is_baseline else 0.0
        conflict_reduction_rate = float(group["conflict_reduced"].mean()) if not is_baseline else 0.0
        mean_rescue_score = float(group["rescue_effect_score"].mean())
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "governance_mechanism_id": mechanism_id,
                "governance_mechanism_family": first["governance_mechanism_family"],
                "global_controller_like": bool(first["global_controller_like"]),
                "governance_influence_radius_label": first["governance_influence_radius_label"],
                "governance_information_access_class": first["governance_information_access_class"],
                "condition_count": int(group[["base_condition_id", "interface_rule_id"]].drop_duplicates().shape[0]),
                "mean_delta_final_target_quality": float(group["delta_final_target_quality"].mean()),
                "mean_delta_reference_increasing_score": float(group["delta_reference_increasing_score"].mean()),
                "mean_delta_goal_conflict_index": float(group["delta_goal_conflict_index"].mean()),
                "mean_delta_aggregation_delta_percent": float(group["delta_aggregation_delta_percent"].mean()),
                "mean_delta_interface_count": float(group["delta_interface_count"].mean()),
                "mean_delta_work_count": float(group["delta_work_count"].mean()),
                "mean_governance_block_fraction": float(group["mean_governance_block_fraction"].mean()),
                "mean_governance_cost_per_event": float(group["mean_governance_cost_per_event"].mean()),
                "mean_global_controller_invocation_count": float(group["mean_global_controller_invocation_count"].mean()),
                "target_quality_rescue_rate": rescue_rate,
                "conflict_reduction_rate": conflict_reduction_rate,
                "mean_rescue_effect_score": mean_rescue_score,
                "supportive_rescue_by_threshold": bool((not is_baseline) and mean_rescue_score >= config.rescue_delta_threshold),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["global_controller_like", "mean_rescue_effect_score", "mean_delta_final_target_quality"],
        ascending=[True, False, False],
        kind="mergesort",
    )


def information_access_strata(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    group_columns = [
        "global_controller_like",
        "governance_influence_radius_label",
        "governance_information_access_class",
        "interface_rule_id",
        "explicit_recognition_used",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in comparison.groupby(group_columns, sort=False):
        (
            global_like,
            radius_label,
            access_class,
            interface_rule_id,
            explicit_recognition,
        ) = keys
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "global_controller_like": bool(global_like),
                "governance_influence_radius_label": radius_label,
                "governance_information_access_class": access_class,
                "interface_rule_id": interface_rule_id,
                "explicit_recognition_used": bool(explicit_recognition),
                "mechanism_count": int(group["governance_mechanism_id"].nunique()),
                "condition_count": int(group[["base_condition_id", "governance_mechanism_id"]].drop_duplicates().shape[0]),
                "mean_delta_final_target_quality": float(group["delta_final_target_quality"].mean()),
                "mean_delta_goal_conflict_index": float(group["delta_goal_conflict_index"].mean()),
                "mean_rescue_effect_score": float(group["rescue_effect_score"].mean()),
                "mean_governance_block_fraction": float(group["mean_governance_block_fraction"].mean()),
                "mean_governance_cost_per_event": float(group["mean_governance_cost_per_event"].mean()),
            }
        )
    return pd.DataFrame(rows)


def validate_s09_outputs(
    run_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    rankings: pd.DataFrame,
    info_strata: pd.DataFrame,
    s08_strata: pd.DataFrame,
    config: S09Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    seed_set = set(int(seed) for seed in config.seeds)
    expected_interfaces = {rule.rule_id for rule in config.interface_rules}
    expected_governance = {mechanism.mechanism_id for mechanism in config.governance_mechanisms}
    base_interface_sets = (
        run_df.groupby("base_condition_id")["interface_rule_id"].agg(lambda values: set(map(str, values))).tolist() if not run_df.empty else []
    )
    base_interface_governance_sets = (
        run_df.groupby(["base_condition_id", "interface_rule_id"])["governance_mechanism_id"].agg(lambda values: set(map(str, values))).tolist()
        if not run_df.empty
        else []
    )
    seed_sets = (
        run_df.groupby(["base_condition_id", "interface_rule_id", "governance_mechanism_id"])["seed"].agg(lambda values: set(int(value) for value in values)).tolist()
        if not run_df.empty
        else []
    )
    finite_columns = [
        "delta_final_target_quality",
        "delta_goal_conflict_index",
        "rescue_effect_score",
        "mean_governance_block_fraction",
        "mean_governance_cost_per_event",
    ]
    finite_comparison = bool(
        not comparison.empty
        and set(finite_columns).issubset(comparison.columns)
        and comparison[finite_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().all().all()
    )
    local_mechanisms = [mechanism for mechanism in config.governance_mechanisms if not mechanism.global_controller_like]
    global_mechanisms = [mechanism for mechanism in config.governance_mechanisms if mechanism.global_controller_like]
    radius_documented = all(mechanism.influence_radius_label and (mechanism.influence_radius_cells is not None or mechanism.global_controller_like) for mechanism in config.governance_mechanisms)
    info_documented = all(mechanism.information_access_class for mechanism in config.governance_mechanisms)
    global_labeled = bool(
        global_mechanisms
        and not info_strata.empty
        and info_strata["global_controller_like"].isin([True]).any()
        and run_df[run_df["global_controller_like"]]["governance_information_access_class"].astype(str).str.contains("global").all()
        and not run_df[~run_df["global_controller_like"]]["governance_information_access_class"].astype(str).str.contains("global").any()
    )
    explicit_ok = False
    if not run_df.empty:
        baseline = run_df[run_df["interface_rule_id"] == "behavior_only"]
        explicit = run_df[run_df["interface_rule_id"] != "behavior_only"]
        explicit_ok = bool(
            not baseline.empty
            and not explicit.empty
            and (baseline["explicit_recognition_used"] == False).all()  # noqa: E712
            and explicit["explicit_recognition_used"].all()
        )
    checks = [
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "selected_s04_conflict_cases",
            "success": bool(not selected.empty and set(selected["goal_compatibility_class"]).issubset(set(CONFLICT_CLASSES))),
            "observed": sorted(selected["goal_compatibility_class"].unique().tolist()) if not selected.empty else [],
            "expected": f"selected S04 cases are drawn from {list(CONFLICT_CLASSES)}",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "s08_interface_strata_present",
            "success": bool(expected_interfaces.issubset(set(s08_strata["interface_rule_id"].astype(str))) and all(expected_interfaces == item for item in base_interface_sets)),
            "observed": sorted(s08_strata["interface_rule_id"].astype(str).unique().tolist()) if "interface_rule_id" in s08_strata else [],
            "expected": sorted(expected_interfaces),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "behavior_only_and_explicit_interface_strata_separated",
            "success": explicit_ok,
            "observed": "behavior-only and explicit-recognition rows present with separate flags" if explicit_ok else "missing or mixed interface flags",
            "expected": "behavior_only is no-recognition baseline; all non-baseline S08 strata are explicit-recognition interventions",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "no_governance_baseline_for_each_base_interface",
            "success": bool(base_interface_governance_sets and all("no_governance" in item for item in base_interface_governance_sets)),
            "observed": f"base/interface groups={len(base_interface_governance_sets)}",
            "expected": "each selected base condition and interface stratum has no_governance baseline",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "governance_mechanism_panel_complete",
            "success": bool(base_interface_governance_sets and all(expected_governance == item for item in base_interface_governance_sets)),
            "observed": sorted(run_df["governance_mechanism_id"].astype(str).unique().tolist()) if not run_df.empty else [],
            "expected": sorted(expected_governance),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "matched_seed_sets_by_base_interface_governance",
            "success": bool(seed_sets and all(item == seed_set for item in seed_sets)),
            "observed": f"groups={len(seed_sets)} expected_seeds={sorted(seed_set)}",
            "expected": "each condition uses the same configured seed set",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "realized_ratios_and_counts_match",
            "success": bool(not run_df.empty and (run_df["realized_ratio_max_abs_error"] <= 1e-12).all() and run_df["count_preservation_success"].all()),
            "observed": f"max_error={run_df['realized_ratio_max_abs_error'].max() if not run_df.empty else math.nan}",
            "expected": "policy counts are preserved for every governance run",
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
            "validation_case": "influence_radius_documented",
            "success": bool(radius_documented and not condition_df["governance_influence_radius_label"].isna().any()),
            "observed": sorted(condition_df["governance_influence_radius_label"].astype(str).unique().tolist()) if not condition_df.empty else [],
            "expected": "every governance mechanism has an influence-radius label or explicit global label",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "information_access_documented",
            "success": bool(info_documented and not condition_df["governance_information_access_class"].isna().any()),
            "observed": sorted(condition_df["governance_information_access_class"].astype(str).unique().tolist()) if not condition_df.empty else [],
            "expected": "every governance mechanism documents its information access",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "global_controller_access_labeled_and_stratified",
            "success": global_labeled,
            "observed": f"global_mechanisms={len(global_mechanisms)} local_mechanisms={len(local_mechanisms)} info_strata={len(info_strata)}",
            "expected": "global-controller-like rows are explicitly labeled and separated from finite-radius mechanisms",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "finite_baseline_comparisons_and_rankings",
            "success": bool(finite_comparison and not rankings.empty and rankings["mean_rescue_effect_score"].notna().all()),
            "observed": f"comparison_rows={len(comparison)} ranking_rows={len(rankings)}",
            "expected": "all rescue, cost, and baseline-delta metrics are finite",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "s04_traceability_preserved",
            "success": bool(not run_df.empty and run_df["source_s04_condition_id"].notna().all() and set(selected["condition_id"]).issubset(set(run_df["source_s04_condition_id"]))),
            "observed": f"selected={selected['condition_id'].nunique() if 'condition_id' in selected else 0} run_sources={run_df['source_s04_condition_id'].nunique() if not run_df.empty else 0}",
            "expected": "every S09 row traces to one selected S04 conflict condition",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "figure_written",
            "success": bool(figure_written),
            "observed": str(figure_written),
            "expected": "governance_effects.png written and non-empty",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests_success),
            "observed": str(unit_tests_success),
            "expected": "focused and regression unit tests pass",
        },
    ]
    return pd.DataFrame(checks)
