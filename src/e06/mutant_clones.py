"""E06 S11 cancer-like mutant clone experiments."""

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
from src.e03.policy_interface import PolicyAction
from src.e06.goal_compatibility import evaluate_goal_state, goal_assignments_for_profile, rank_values
from src.e06.governance_mechanisms import (
    DEFAULT_GOVERNANCE_MECHANISMS,
    DEFAULT_INTERFACE_RULES,
    GovernanceMechanism,
    governance_decision,
)
from src.e06.graft_experiments import (
    _block_state,
    _run_host_development,
    _success_state,
    _wait_state,
    governance_access_stratum,
    interface_access_stratum,
    intervention_stratum,
)
from src.e06.interface_rules import InterfaceRule, frozen_indices_for_profile, goal_profile_for_id, initial_values_for_profile, interface_decision
from src.e06.mixture_ratios import (
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    MixtureRuntime,
    aggregation_metrics,
    classify_final_state,
    dominance_metrics,
    stable_hash,
)
from src.e06.spatial_arrangements import arrangement_metric_row, labels_for_arrangement


STEP_ID = "S11"
MUTANT_SCHEMA = "eidosoma.e06.s11_mutant_clone_run.v1"
INITIAL_STATE_SCHEMA = "eidosoma.e06.s11_initial_clone_state.v1"
OBJECTIVE_SCHEMA = "eidosoma.e06.s11_selfish_objective.v1"
SUMMARY_SCHEMA = "eidosoma.e06.s11_mutant_clone_summary.v1"
COMPARISON_SCHEMA = "eidosoma.e06.s11_mutant_baseline_comparison.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s11_validation.v1"


@dataclass(frozen=True)
class SelfishObjective:
    """One explicitly defined computational selfish objective."""

    objective_id: str
    objective_family: str
    score_name: str
    allows_replication: bool
    gate_goal_proxy: str
    definition: str
    action_rule: str
    biological_analogy_caveat: str


DEFAULT_SELFISH_OBJECTIVES = (
    SelfishObjective(
        objective_id="edge_position_left",
        objective_family="position_exploitation",
        score_name="mean_left_edge_position_advantage",
        allows_replication=False,
        gate_goal_proxy="increasing",
        definition="Mutant cells maximize their own left-edge positional advantage, scored as 1 - index/(n-1), without optimizing host sortedness.",
        action_rule="A mutant actor proposes an adjacent left swap whenever possible, independent of global target quality.",
        biological_analogy_caveat="This is a computational selfish-position proxy, not evidence for biological polarity or migration mechanisms.",
    ),
    SelfishObjective(
        objective_id="clone_cohesion",
        objective_family="homotypic_clone_aggregation",
        score_name="mutant_mutant_contact_fraction",
        allows_replication=False,
        gate_goal_proxy="increasing",
        definition="Mutant cells maximize local homotypic contact with other mutant cells, scored by mutant-mutant neighbor contacts per possible mutant contact.",
        action_rule="A mutant actor proposes the adjacent swap that strictly increases local mutant-mutant edge count around the actor and target.",
        biological_analogy_caveat="This is a computational cohesion proxy, not a biological adhesion or self-recognition mechanism.",
    ),
    SelfishObjective(
        objective_id="replicative_takeover",
        objective_family="identity_replication",
        score_name="mutant_label_fraction",
        allows_replication=True,
        gate_goal_proxy="increasing",
        definition="Mutant cells maximize clone identity prevalence, scored as the fraction of array sites carrying the mutant label.",
        action_rule="A mutant actor adjacent to a host proposes identity conversion of one neighboring host site into the mutant label; values are not duplicated.",
        biological_analogy_caveat="This is a computational identity-takeover proxy, not biological cell division or cancer validation.",
    ),
)


@dataclass(frozen=True)
class CloneSpec:
    """One mutant clone introduction schedule and placement."""

    spec_id: str
    introduction_event: int
    initial_size: int
    initial_location: str
    placement_reference: str
    description: str


DEFAULT_CLONE_SPECS = (
    CloneSpec(
        spec_id="early_center_clone_05",
        introduction_event=1_000,
        initial_size=5,
        initial_location="center",
        placement_reference="graft_like_insertions",
        description="Early 5-cell center clone introduced after 1,000 host-development events using S03 graft-like placement.",
    ),
    CloneSpec(
        spec_id="late_left_edge_clone_10",
        introduction_event=2_400,
        initial_size=10,
        initial_location="left_edge",
        placement_reference="contiguous_patch",
        description="Late 10-cell left-edge clone introduced after 2,400 host-development events using S03 contiguous patch placement.",
    ),
)


@dataclass(frozen=True)
class S11Config:
    """Configuration for bounded S11 mutant-clone tests."""

    array_size: int = 100
    event_cap: int = 4_000
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    max_base_contexts: int = 3
    clone_specs: tuple[CloneSpec, ...] = DEFAULT_CLONE_SPECS
    selfish_objectives: tuple[SelfishObjective, ...] = DEFAULT_SELFISH_OBJECTIVES
    interface_rules: tuple[InterfaceRule, ...] = DEFAULT_INTERFACE_RULES
    governance_mechanisms: tuple[GovernanceMechanism, ...] = DEFAULT_GOVERNANCE_MECHANISMS
    scheduler: str = "cyclic_scan_seed_offset"
    worker_count: int = 1
    containment_effect_threshold: float = 0.02


@dataclass(frozen=True)
class MutantBaseContext:
    """One S10 graft-sensitive context converted into a mutant clone assay."""

    base_context_id: str
    source_s07_run_id: str
    source_research_step_id: str
    source_condition_id: str
    selection_rank: int
    selection_reason: str
    s11_selection_reason: str
    s07_label: str
    s07_classification_mode: str
    panel: str
    candidate_reason: str
    source_arrangement: str
    goal_profile_id: str
    goal_compatibility_class: str
    policy_ids: tuple[str, str]
    display_names: tuple[str, str]
    source_categories: tuple[str, str]
    ratio_targets: tuple[float, float]
    host_policy_id: str
    mutant_parent_policy_id: str
    host_display_name: str
    mutant_parent_display_name: str
    host_source_category: str
    mutant_parent_source_category: str
    value_profile: str
    perturbation_profile: str
    source_final_target_quality: float
    source_goal_conflict_index: float
    source_aggregation_delta_percent: float
    source_largest_block_fraction: float
    source_mosaic_continuum_score: float
    source_conflict_continuum_score: float
    s10_run_count: int
    s10_graft_outcome_classes: int
    s10_graft_outcome_counts_json: str
    s10_mean_target_delta_vs_pre_graft: float
    s10_max_abs_target_delta_vs_pre_graft: float
    s10_mean_graft_block_fraction: float
    s10_graft_sensitivity_score: float


@dataclass(frozen=True)
class MutantCondition:
    """One executable S11 condition before seed expansion."""

    condition_id: str
    base: MutantBaseContext
    clone_spec: CloneSpec
    objective: SelfishObjective
    interface_rule: InterfaceRule
    governance: GovernanceMechanism
    mutant_policy_id: str
    mutant_display_name: str


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
    mode = values.astype(str).mode()
    return str(mode.iloc[0]) if not mode.empty else ""


def objective_table(objectives: Sequence[SelfishObjective] = DEFAULT_SELFISH_OBJECTIVES) -> pd.DataFrame:
    return pd.DataFrame([{**objective.__dict__, "schema": OBJECTIVE_SCHEMA, "research_step_id": STEP_ID} for objective in objectives])


def _mutant_policy_id(parent_policy_id: str, objective_id: str) -> str:
    return f"s11_mutant_{stable_hash({'parentPolicyId': parent_policy_id, 'objectiveId': objective_id})[:16]}"


def _condition_id(base_context_id: str, clone_spec_id: str, objective_id: str, interface_rule_id: str, governance_id: str) -> str:
    payload = {
        "baseContextId": base_context_id,
        "cloneSpecId": clone_spec_id,
        "objectiveId": objective_id,
        "interfaceRuleId": interface_rule_id,
        "governanceMechanismId": governance_id,
    }
    return f"s11_{stable_hash(payload)[:16]}"


def _initial_state_id(base_context_id: str, clone_spec_id: str, objective_id: str, seed: int) -> str:
    payload = {"baseContextId": base_context_id, "cloneSpecId": clone_spec_id, "objectiveId": objective_id, "seed": int(seed)}
    return f"s11_state_{stable_hash(payload)[:18]}"


def select_s11_base_contexts(s10_selected: pd.DataFrame, s10_runs: pd.DataFrame, config: S11Config) -> pd.DataFrame:
    """Select bounded S10 graft-sensitive contexts for mutant clone tests."""

    required_selected = {
        "base_context_id",
        "selection_rank",
        "selection_reason",
        "source_s07_run_id",
        "source_research_step_id",
        "source_condition_id",
        "s07_label",
        "s07_classification_mode",
        "panel",
        "candidate_reason",
        "source_arrangement",
        "goal_profile_id",
        "goal_compatibility_class",
        "policy_ids_json",
        "display_names_json",
        "source_categories_json",
        "ratio_targets_json",
        "host_policy_id",
        "graft_policy_id",
        "host_display_name",
        "graft_display_name",
        "host_source_category",
        "graft_source_category",
        "value_profile",
        "perturbation_profile",
        "source_final_target_quality",
        "source_goal_conflict_index",
        "source_aggregation_delta_percent",
        "source_largest_block_fraction",
        "source_mosaic_continuum_score",
        "source_conflict_continuum_score",
    }
    required_runs = {
        "base_context_id",
        "research_step_id",
        "graft_outcome_class",
        "target_delta_vs_pre_graft",
        "graft_relative_largest_block_fraction",
    }
    missing_selected = required_selected - set(s10_selected.columns)
    missing_runs = required_runs - set(s10_runs.columns)
    if missing_selected:
        raise ValueError(f"S10 selected-context table missing columns for S11: {sorted(missing_selected)}")
    if missing_runs:
        raise ValueError(f"S10 graft run table missing columns for S11: {sorted(missing_runs)}")
    runs = s10_runs[s10_runs["research_step_id"].astype(str) == "S10"].copy()
    if runs.empty:
        raise ValueError("S11 requires non-empty S10 graft run rows")
    runs["target_delta_vs_pre_graft"] = pd.to_numeric(runs["target_delta_vs_pre_graft"], errors="coerce").fillna(0.0)
    runs["graft_relative_largest_block_fraction"] = pd.to_numeric(runs["graft_relative_largest_block_fraction"], errors="coerce").fillna(0.0)
    agg_rows: list[dict[str, Any]] = []
    for base_context_id, group in runs.groupby("base_context_id", sort=False):
        outcome_counts = group["graft_outcome_class"].astype(str).value_counts().sort_index().to_dict()
        outcome_class_count = len(outcome_counts)
        max_abs_delta = float(group["target_delta_vs_pre_graft"].abs().max())
        mean_delta = float(group["target_delta_vs_pre_graft"].mean())
        mean_block = float(group["graft_relative_largest_block_fraction"].mean())
        sensitivity_score = float(outcome_class_count / 5.0 + max_abs_delta + abs(mean_delta) + 0.25 * mean_block)
        agg_rows.append(
            {
                "base_context_id": str(base_context_id),
                "s10_run_count": int(len(group)),
                "s10_graft_outcome_classes": int(outcome_class_count),
                "s10_graft_outcome_counts_json": _json_object(outcome_counts),
                "s10_mean_target_delta_vs_pre_graft": mean_delta,
                "s10_max_abs_target_delta_vs_pre_graft": max_abs_delta,
                "s10_mean_graft_block_fraction": mean_block,
                "s10_graft_sensitivity_score": sensitivity_score,
            }
        )
    sensitivity = pd.DataFrame(agg_rows)
    selected = s10_selected.merge(sensitivity, on="base_context_id", how="inner")
    if selected.empty:
        raise ValueError("S11 could not match S10 selected contexts to S10 run sensitivity rows")
    selected = selected.sort_values(["s10_graft_sensitivity_score", "selection_rank"], ascending=[False, True], kind="mergesort").head(
        config.max_base_contexts
    )
    selected = selected.reset_index(drop=True)
    selected["s11_selection_rank"] = np.arange(1, len(selected) + 1)
    selected["s11_selection_reason"] = selected["selection_reason"].astype(str) + ",s10_graft_sensitive_context"
    return selected


def _base_from_row(raw: Mapping[str, Any]) -> MutantBaseContext:
    policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
    display_names = tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
    categories = tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
    ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
    return MutantBaseContext(
        base_context_id=str(raw["base_context_id"]),
        source_s07_run_id=str(raw["source_s07_run_id"]),
        source_research_step_id=str(raw["source_research_step_id"]),
        source_condition_id=str(raw["source_condition_id"]),
        selection_rank=int(raw.get("s11_selection_rank", raw["selection_rank"])),
        selection_reason=str(raw["selection_reason"]),
        s11_selection_reason=str(raw["s11_selection_reason"]),
        s07_label=str(raw["s07_label"]),
        s07_classification_mode=str(raw["s07_classification_mode"]),
        panel=str(raw["panel"]),
        candidate_reason=str(raw["candidate_reason"]),
        source_arrangement=str(raw["source_arrangement"]),
        goal_profile_id=str(raw["goal_profile_id"]),
        goal_compatibility_class=str(raw["goal_compatibility_class"]),
        policy_ids=(policy_ids[0], policy_ids[1]),
        display_names=(display_names[0], display_names[1]),
        source_categories=(categories[0], categories[1]),
        ratio_targets=(ratios[0], ratios[1]),
        host_policy_id=str(raw["host_policy_id"]),
        mutant_parent_policy_id=str(raw["graft_policy_id"]),
        host_display_name=str(raw["host_display_name"]),
        mutant_parent_display_name=str(raw["graft_display_name"]),
        host_source_category=str(raw["host_source_category"]),
        mutant_parent_source_category=str(raw["graft_source_category"]),
        value_profile=str(raw["value_profile"]),
        perturbation_profile=str(raw["perturbation_profile"]),
        source_final_target_quality=float(raw["source_final_target_quality"]),
        source_goal_conflict_index=float(raw["source_goal_conflict_index"]),
        source_aggregation_delta_percent=float(raw["source_aggregation_delta_percent"]),
        source_largest_block_fraction=float(raw["source_largest_block_fraction"]),
        source_mosaic_continuum_score=float(raw["source_mosaic_continuum_score"]),
        source_conflict_continuum_score=float(raw["source_conflict_continuum_score"]),
        s10_run_count=int(raw["s10_run_count"]),
        s10_graft_outcome_classes=int(raw["s10_graft_outcome_classes"]),
        s10_graft_outcome_counts_json=str(raw["s10_graft_outcome_counts_json"]),
        s10_mean_target_delta_vs_pre_graft=float(raw["s10_mean_target_delta_vs_pre_graft"]),
        s10_max_abs_target_delta_vs_pre_graft=float(raw["s10_max_abs_target_delta_vs_pre_graft"]),
        s10_mean_graft_block_fraction=float(raw["s10_mean_graft_block_fraction"]),
        s10_graft_sensitivity_score=float(raw["s10_graft_sensitivity_score"]),
    )


def build_s11_conditions(selected: pd.DataFrame, config: S11Config) -> list[MutantCondition]:
    conditions: list[MutantCondition] = []
    for raw in selected.to_dict(orient="records"):
        base = _base_from_row(raw)
        for clone_spec in config.clone_specs:
            for objective in config.selfish_objectives:
                mutant_policy_id = _mutant_policy_id(base.mutant_parent_policy_id, objective.objective_id)
                mutant_display_name = f"{base.mutant_parent_display_name}::{objective.objective_id}"
                for interface_rule in config.interface_rules:
                    for governance in config.governance_mechanisms:
                        conditions.append(
                            MutantCondition(
                                condition_id=_condition_id(
                                    base.base_context_id,
                                    clone_spec.spec_id,
                                    objective.objective_id,
                                    interface_rule.rule_id,
                                    governance.mechanism_id,
                                ),
                                base=base,
                                clone_spec=clone_spec,
                                objective=objective,
                                interface_rule=interface_rule,
                                governance=governance,
                                mutant_policy_id=mutant_policy_id,
                                mutant_display_name=mutant_display_name,
                            )
                        )
    return conditions


def clone_layout_labels(host_policy_id: str, mutant_policy_id: str, clone_spec: CloneSpec, array_size: int, seed: int, condition_id: str) -> tuple[str, ...]:
    host_count = int(array_size - clone_spec.initial_size)
    clone_count = int(clone_spec.initial_size)
    if clone_spec.initial_location == "center":
        return labels_for_arrangement(
            (host_policy_id, mutant_policy_id),
            (host_count, clone_count),
            seed=seed,
            condition_id=condition_id,
            arrangement="graft_like_insertions",
        )
    if clone_spec.initial_location == "left_edge":
        return labels_for_arrangement(
            (mutant_policy_id, host_policy_id),
            (clone_count, host_count),
            seed=seed,
            condition_id=condition_id,
            arrangement="contiguous_patch",
        )
    if clone_spec.initial_location == "right_edge":
        return labels_for_arrangement(
            (host_policy_id, mutant_policy_id),
            (host_count, clone_count),
            seed=seed,
            condition_id=condition_id,
            arrangement="contiguous_patch",
        )
    raise ValueError(f"unknown clone initial location: {clone_spec.initial_location}")


def apply_clone_initialization(
    labels: list[str],
    cell_ids: list[int],
    *,
    host_policy_id: str,
    mutant_policy_id: str,
    clone_spec: CloneSpec,
    seed: int,
    condition_id: str,
) -> dict[str, Any]:
    layout = clone_layout_labels(host_policy_id, mutant_policy_id, clone_spec, len(labels), seed, condition_id)
    clone_indices = [idx for idx, label in enumerate(layout) if str(label) == str(mutant_policy_id)]
    if len(clone_indices) != clone_spec.initial_size:
        raise RuntimeError("S03 clone layout produced the wrong clone size")
    for idx, label in enumerate(layout):
        labels[idx] = str(label)
    base_new_id = len(cell_ids)
    for offset, idx in enumerate(clone_indices):
        cell_ids[idx] = base_new_id + offset
    return {
        "clone_indices": tuple(clone_indices),
        "clone_start_index": int(min(clone_indices)),
        "clone_end_index": int(max(clone_indices)),
        "initial_clone_cell_ids": tuple(cell_ids[idx] for idx in clone_indices),
        "clone_layout_labels": tuple(layout),
    }


def _affected_edges(n: int, left: int, right: int) -> tuple[tuple[int, int], ...]:
    low = max(0, min(left, right) - 1)
    high = min(n - 2, max(left, right) + 1)
    return tuple((idx, idx + 1) for idx in range(low, high + 1))


def _mutant_edge_count(labels: Sequence[str], edges: Sequence[tuple[int, int]], mutant_policy_id: str) -> int:
    return sum(1 for left, right in edges if str(labels[left]) == str(labels[right]) == str(mutant_policy_id))


def selfish_objective_score(labels: Sequence[str], cell_ids: Sequence[int], mutant_policy_id: str, objective: SelfishObjective) -> float:
    positions = [idx for idx, label in enumerate(labels) if str(label) == str(mutant_policy_id)]
    if not positions:
        return 0.0
    n = len(labels)
    if objective.objective_id == "edge_position_left":
        denom = max(1, n - 1)
        return float(sum(1.0 - pos / denom for pos in positions) / len(positions))
    if objective.objective_id == "clone_cohesion":
        if len(positions) <= 1:
            return 0.0
        mutant_edges = sum(1 for idx in range(n - 1) if str(labels[idx]) == str(labels[idx + 1]) == str(mutant_policy_id))
        return float(mutant_edges / max(1, len(positions) - 1))
    if objective.objective_id == "replicative_takeover":
        return float(len(positions) / max(1, n))
    raise ValueError(f"unknown selfish objective: {objective.objective_id}")


def selfish_action(
    objective: SelfishObjective,
    *,
    labels: Sequence[str],
    actor_index: int,
    mutant_policy_id: str,
    rng: random.Random,
) -> PolicyAction:
    n = len(labels)
    if objective.objective_id == "edge_position_left":
        target = actor_index - 1
        if target >= 0:
            return PolicyAction("swap", target_index=target, compare_counted=True, metadata={"selfish_objective": objective.objective_id})
        return PolicyAction("wait", compare_counted=False, metadata={"selfish_objective": objective.objective_id})
    if objective.objective_id == "clone_cohesion":
        candidates: list[tuple[int, int, int]] = []
        for target in (actor_index - 1, actor_index + 1):
            if not 0 <= target < n:
                continue
            edges = _affected_edges(n, actor_index, target)
            before = _mutant_edge_count(labels, edges, mutant_policy_id)
            after_labels = list(labels)
            after_labels[actor_index], after_labels[target] = after_labels[target], after_labels[actor_index]
            after = _mutant_edge_count(after_labels, edges, mutant_policy_id)
            candidates.append((after - before, -abs(target - actor_index), target))
        if candidates:
            best_delta, _distance, best_target = max(candidates)
            if best_delta > 0:
                return PolicyAction("swap", target_index=best_target, compare_counted=True, metadata={"selfish_objective": objective.objective_id})
        return PolicyAction("wait", compare_counted=bool(candidates), metadata={"selfish_objective": objective.objective_id})
    if objective.objective_id == "replicative_takeover":
        targets = [target for target in (actor_index - 1, actor_index + 1) if 0 <= target < n and str(labels[target]) != str(mutant_policy_id)]
        if targets:
            target = rng.choice(sorted(targets))
            return PolicyAction("replicate", target_index=target, compare_counted=True, metadata={"selfish_objective": objective.objective_id})
        return PolicyAction("wait", compare_counted=False, metadata={"selfish_objective": objective.objective_id})
    raise ValueError(f"unknown selfish objective: {objective.objective_id}")


def clone_metrics(labels: Sequence[str], cell_ids: Sequence[int], mutant_policy_id: str, initial_clone_cell_ids: Sequence[int]) -> dict[str, Any]:
    clone_positions = [idx for idx, label in enumerate(labels) if str(label) == str(mutant_policy_id)]
    n = len(labels)
    clone_count = len(clone_positions)
    if clone_count == 0:
        return {
            "final_clone_count": 0,
            "final_clone_fraction": 0.0,
            "clone_positions_json": "[]",
            "clone_position_mean": math.nan,
            "clone_position_min": math.nan,
            "clone_position_max": math.nan,
            "clone_position_span": math.nan,
            "clone_edge_fraction": math.nan,
            "clone_largest_block_fraction": 0.0,
            "initial_clone_cell_retention_fraction": 0.0,
        }
    runs: list[int] = []
    current = 0
    for label in labels:
        if str(label) == str(mutant_policy_id):
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    edge_window = max(1, int(round(n * 0.15)))
    edge_count = sum(1 for pos in clone_positions if pos < edge_window or pos >= n - edge_window)
    initial_ids = {int(item) for item in initial_clone_cell_ids}
    retained = sum(1 for idx, cell_id in enumerate(cell_ids) if int(cell_id) in initial_ids and str(labels[idx]) == str(mutant_policy_id))
    return {
        "final_clone_count": int(clone_count),
        "final_clone_fraction": float(clone_count / max(1, n)),
        "clone_positions_json": _json_list(clone_positions),
        "clone_position_mean": float(sum(clone_positions) / clone_count),
        "clone_position_min": int(min(clone_positions)),
        "clone_position_max": int(max(clone_positions)),
        "clone_position_span": int(max(clone_positions) - min(clone_positions) + 1),
        "clone_edge_fraction": float(edge_count / clone_count),
        "clone_largest_block_fraction": float(max(runs) / clone_count) if runs else 0.0,
        "initial_clone_cell_retention_fraction": float(retained / max(1, len(initial_ids))),
    }


def classify_mutant_outcome(*, final_clone_fraction: float, clone_growth_delta: int, target_quality_damage: float, clone_largest_block_fraction: float) -> str:
    if final_clone_fraction >= 0.75 and target_quality_damage >= 0.05:
        return "damaging_takeover"
    if final_clone_fraction >= 0.50:
        return "takeover"
    if clone_growth_delta >= 10 or final_clone_fraction >= 0.25:
        return "local_invasion"
    if target_quality_damage >= 0.08:
        return "damaging_contained_clone"
    if clone_growth_delta <= 0 and clone_largest_block_fraction >= 0.70:
        return "contained_segregated_clone"
    if clone_growth_delta <= 0:
        return "contained_no_growth"
    return "partial_containment"


def _post_clone_loop(
    *,
    values: list[int],
    labels: list[str],
    cell_ids: list[int],
    memory: dict[int, dict[str, Any]],
    runtime: MixtureRuntime,
    schedule: Sequence[int],
    goal_assignments: Mapping[str, str],
    frozen_indices: frozenset[int],
    condition: MutantCondition,
    config: S11Config,
    rng: random.Random,
) -> dict[str, Any]:
    counts = Counter()
    same_contact_delta_sum = 0.0
    local_goal_delta_sum = 0.0
    governance_local_goal_delta_sum = 0.0
    governance_heterotypic_edge_delta_sum = 0.0
    governance_block_reasons: Counter[str] = Counter()
    interface_block_reasons: Counter[str] = Counter()
    next_cell_id = max(cell_ids) + 1
    clone_fraction_peak = labels.count(condition.mutant_policy_id) / max(1, len(labels))
    for actor_index in schedule:
        actor_index = int(actor_index)
        policy_id = str(labels[actor_index])
        cell_id = cell_ids[actor_index]
        if policy_id == condition.mutant_policy_id:
            action = selfish_action(condition.objective, labels=labels, actor_index=actor_index, mutant_policy_id=condition.mutant_policy_id, rng=rng)
            counts["selfish_action_count"] += 1
            counts[f"selfish_{action.action_type}_proposal_count"] += int(action.action_type in {"swap", "replicate"})
        else:
            actor_goal = goal_assignments[policy_id]
            proxy_values = rank_values(values, actor_goal, config.array_size)
            action = runtime.action_for(policy_id, proxy_values, labels, actor_index, cell_id, memory, rng)
            counts["host_policy_action_count"] += 1
        counts["compare_count"] += int(bool(action.compare_counted))
        target = None if action.target_index is None else int(action.target_index)
        if target is not None and (actor_index in frozen_indices or target in frozen_indices):
            counts["frozen_block_count"] += 1
            counts["wait_count"] += 1
            _block_state(memory, cell_id, "post_clone_frozen_block")
            continue
        if action.action_type in {"swap", "replicate"} and target is not None:
            if 0 <= target < len(values):
                counts["raw_swap_proposal_count"] += 1
                pre_cross = str(labels[actor_index]) != str(labels[target])
                counts["pre_governance_cross_label_proposal_count"] += int(pre_cross)
                governance = governance_decision(
                    condition.governance,
                    values=values,
                    labels=labels,
                    actor_index=actor_index,
                    target_index=target,
                    goal_assignments=goal_assignments,
                    policy_ids=(condition.base.host_policy_id, condition.mutant_policy_id),
                    array_size=config.array_size,
                    rng=rng,
                )
                if bool(governance["invoked"]):
                    counts["governance_invocation_count"] += 1
                    counts["governance_cost_units"] += int(governance["cost_units"])
                    counts["global_controller_invocation_count"] += int(condition.governance.global_controller_like)
                    if math.isfinite(float(governance["local_goal_before"])):
                        governance_local_goal_delta_sum += float(governance["local_goal_after"]) - float(governance["local_goal_before"])
                    if math.isfinite(float(governance["heterotypic_edge_before"])):
                        governance_heterotypic_edge_delta_sum += float(governance["heterotypic_edge_after"]) - float(governance["heterotypic_edge_before"])
                if not bool(governance["accepted"]):
                    counts["governance_block_count"] += 1
                    counts["governance_cross_label_block_count"] += int(pre_cross)
                    reason = str(governance["block_reason"])
                    governance_block_reasons[reason] += 1
                    counts["wait_count"] += 1
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
                    counts["cross_label_swap_proposal_count"] += 1
                else:
                    counts["same_label_swap_proposal_count"] += 1
                if bool(decision["explicit_invoked"]):
                    counts["interface_rule_invocation_count"] += 1
                    if math.isfinite(float(decision["same_contact_before"])):
                        same_contact_delta_sum += float(decision["same_contact_after"]) - float(decision["same_contact_before"])
                    if math.isfinite(float(decision["local_goal_before"])):
                        local_goal_delta_sum += float(decision["local_goal_after"]) - float(decision["local_goal_before"])
                if not bool(decision["accepted"]):
                    counts["interface_block_count"] += 1
                    reason = str(decision["block_reason"])
                    interface_block_reasons[reason] += 1
                    counts["wait_count"] += 1
                    _block_state(memory, cell_id, f"interface_block:{reason}")
                    continue
                if action.action_type == "replicate":
                    if str(labels[target]) != condition.mutant_policy_id:
                        labels[target] = condition.mutant_policy_id
                        cell_ids[target] = next_cell_id
                        next_cell_id += 1
                        memory.setdefault(cell_ids[target], {})
                        counts["replication_conversion_count"] += 1
                        counts["host_to_mutant_conversion_count"] += 1
                        clone_fraction_peak = max(clone_fraction_peak, labels.count(condition.mutant_policy_id) / max(1, len(labels)))
                    else:
                        counts["replication_noop_count"] += 1
                    _success_state(memory, cell_id)
                else:
                    cross_swap = str(labels[actor_index]) != str(labels[target])
                    values[actor_index], values[target] = values[target], values[actor_index]
                    labels[actor_index], labels[target] = labels[target], labels[actor_index]
                    cell_ids[actor_index], cell_ids[target] = cell_ids[target], cell_ids[actor_index]
                    counts["swap_count"] += 1
                    counts["cross_label_swap_count"] += int(cross_swap)
                    counts["same_label_swap_count"] += int(not cross_swap)
                    _success_state(memory, cell_id)
            else:
                counts["invalid_action_count"] += 1
                counts["wait_count"] += 1
                _block_state(memory, cell_id, "post_clone_invalid_action")
        elif action.action_type == "update_state":
            counts["update_count"] += 1
            state = memory.setdefault(cell_id, {})
            if "ideal_position" in action.state_update:
                updated = action.state_update["ideal_position"]
                state["ideal_position"] = None if updated is None else int(updated)
            state["last_action_type"] = "update_state"
        else:
            counts["wait_count"] += 1
            _wait_state(memory, cell_id)
    return {
        **{str(key): int(value) for key, value in counts.items()},
        "same_contact_delta_sum": float(same_contact_delta_sum),
        "local_goal_delta_sum": float(local_goal_delta_sum),
        "governance_local_goal_delta_sum": float(governance_local_goal_delta_sum),
        "governance_heterotypic_edge_delta_sum": float(governance_heterotypic_edge_delta_sum),
        "governance_block_reasons_json": _json_object(governance_block_reasons),
        "interface_block_reasons_json": _json_object(interface_block_reasons),
        "clone_fraction_peak": float(clone_fraction_peak),
    }


def simulate_mutant_condition(
    condition: MutantCondition,
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: S11Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    initial_values = list(initial_values_for_profile(config.array_size, seed, condition.base.value_profile))
    values = list(initial_values)
    labels = [condition.base.host_policy_id] * config.array_size
    cell_ids = list(range(config.array_size))
    memory: dict[int, dict[str, Any]] = {}
    runtime = MixtureRuntime([records_by_id[condition.base.host_policy_id]], seed)
    full_schedule = actor_schedule(config.array_size, config.event_cap, seed)
    intro_event = min(max(0, int(condition.clone_spec.introduction_event)), config.event_cap)
    pre_schedule = full_schedule[:intro_event]
    post_schedule = full_schedule[intro_event:]
    pre_rng = random.Random(
        seed
        ^ int(
            stable_hash({"s11PreClone": condition.base.base_context_id, "cloneSpecId": condition.clone_spec.spec_id})[:12],
            16,
        )
    )
    post_rng = random.Random(seed ^ int(stable_hash({"s11PostClone": condition.condition_id})[:12], 16))
    goal_profile = goal_profile_for_id(condition.base.goal_profile_id)
    original_assignments = goal_assignments_for_profile((condition.base.host_policy_id, condition.base.mutant_parent_policy_id), goal_profile)
    goal_assignments = {
        condition.base.host_policy_id: original_assignments[condition.base.host_policy_id],
        condition.mutant_policy_id: original_assignments[condition.base.mutant_parent_policy_id],
    }
    frozen_indices = frozen_indices_for_profile(config.array_size, condition.base.perturbation_profile)
    initial_metrics = sortedness_metrics(values)
    pre_counts = _run_host_development(
        values=values,
        labels=labels,
        cell_ids=cell_ids,
        memory=memory,
        runtime=runtime,
        schedule=pre_schedule,
        goal_assignments={condition.base.host_policy_id: goal_assignments[condition.base.host_policy_id]},
        frozen_indices=frozen_indices,
        array_size=config.array_size,
        rng=pre_rng,
    )
    pre_metrics = sortedness_metrics(values)
    layout = apply_clone_initialization(
        labels,
        cell_ids,
        host_policy_id=condition.base.host_policy_id,
        mutant_policy_id=condition.mutant_policy_id,
        clone_spec=condition.clone_spec,
        seed=seed,
        condition_id=condition.condition_id,
    )
    initial_clone_metrics = clone_metrics(labels, cell_ids, condition.mutant_policy_id, layout["initial_clone_cell_ids"])
    initial_selfish_score = selfish_objective_score(labels, cell_ids, condition.mutant_policy_id, condition.objective)
    initial_agg = aggregation_metrics(labels)
    initial_state_id = _initial_state_id(condition.base.base_context_id, condition.clone_spec.spec_id, condition.objective.objective_id, seed)
    initial_state_row = {
        "schema": INITIAL_STATE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "initial_clone_state_id": initial_state_id,
        "base_context_id": condition.base.base_context_id,
        "clone_spec_id": condition.clone_spec.spec_id,
        "selfish_objective_id": condition.objective.objective_id,
        "seed": int(seed),
        "host_policy_id": condition.base.host_policy_id,
        "mutant_policy_id": condition.mutant_policy_id,
        "mutant_parent_policy_id": condition.base.mutant_parent_policy_id,
        "clone_introduction_event": int(intro_event),
        "clone_initial_size": int(condition.clone_spec.initial_size),
        "clone_initial_location": condition.clone_spec.initial_location,
        "clone_placement_reference": condition.clone_spec.placement_reference,
        "clone_initial_indices_json": _json_list(layout["clone_indices"]),
        "clone_initial_cell_ids_json": _json_list(layout["initial_clone_cell_ids"]),
        "initial_values_json": _json_list(values),
        "initial_labels_json": _json_list(labels),
        "initial_cell_ids_json": _json_list(cell_ids),
        "pre_clone_inversion_sortedness": float(pre_metrics["inversion_sortedness"]),
        "initial_clone_fraction": float(initial_clone_metrics["final_clone_fraction"]),
        "initial_selfish_objective_score": float(initial_selfish_score),
    }
    post_intro_values = list(values)
    post_intro_labels = list(labels)
    post_intro_cell_ids = list(cell_ids)
    post_counts = _post_clone_loop(
        values=values,
        labels=labels,
        cell_ids=cell_ids,
        memory=memory,
        runtime=runtime,
        schedule=post_schedule,
        goal_assignments=goal_assignments,
        frozen_indices=frozen_indices,
        condition=condition,
        config=config,
        rng=post_rng,
    )
    elapsed = time.perf_counter() - started
    final_metrics = sortedness_metrics(values)
    final_agg = aggregation_metrics(labels)
    final_counts = Counter(labels)
    dominance = dominance_metrics(labels, (condition.base.host_policy_id, condition.mutant_policy_id))
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
    clone_final = clone_metrics(labels, cell_ids, condition.mutant_policy_id, layout["initial_clone_cell_ids"])
    final_selfish_score = selfish_objective_score(labels, cell_ids, condition.mutant_policy_id, condition.objective)
    clone_growth_delta = int(clone_final["final_clone_count"] - condition.clone_spec.initial_size)
    target_quality_damage = float(pre_metrics["inversion_sortedness"] - goal_eval["assigned_policy_mean_sortedness"])
    reference_damage = float(pre_metrics["inversion_sortedness"] - goal_eval["reference_increasing_score"])
    mutant_outcome = classify_mutant_outcome(
        final_clone_fraction=float(clone_final["final_clone_fraction"]),
        clone_growth_delta=clone_growth_delta,
        target_quality_damage=target_quality_damage,
        clone_largest_block_fraction=float(clone_final["clone_largest_block_fraction"]),
    )
    total_count_success = sum(final_counts.values()) == config.array_size
    nonrep_count_success = True
    if not condition.objective.allows_replication:
        nonrep_count_success = int(final_counts.get(condition.mutant_policy_id, 0)) == int(condition.clone_spec.initial_size)
    run_row = {
        "schema": MUTANT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "condition_id": condition.condition_id,
        "base_context_id": condition.base.base_context_id,
        "source_s07_run_id": condition.base.source_s07_run_id,
        "source_research_step_id": condition.base.source_research_step_id,
        "source_condition_id": condition.base.source_condition_id,
        "selection_rank": int(condition.base.selection_rank),
        "selection_reason": condition.base.selection_reason,
        "s11_selection_reason": condition.base.s11_selection_reason,
        "s07_label": condition.base.s07_label,
        "s07_classification_mode": condition.base.s07_classification_mode,
        "panel": condition.base.panel,
        "candidate_reason": condition.base.candidate_reason,
        "source_arrangement": condition.base.source_arrangement,
        "source_final_target_quality": float(condition.base.source_final_target_quality),
        "source_goal_conflict_index": float(condition.base.source_goal_conflict_index),
        "source_aggregation_delta_percent": float(condition.base.source_aggregation_delta_percent),
        "source_largest_block_fraction": float(condition.base.source_largest_block_fraction),
        "source_mosaic_continuum_score": float(condition.base.source_mosaic_continuum_score),
        "source_conflict_continuum_score": float(condition.base.source_conflict_continuum_score),
        "s10_run_count": int(condition.base.s10_run_count),
        "s10_graft_outcome_classes": int(condition.base.s10_graft_outcome_classes),
        "s10_graft_outcome_counts_json": condition.base.s10_graft_outcome_counts_json,
        "s10_mean_target_delta_vs_pre_graft": float(condition.base.s10_mean_target_delta_vs_pre_graft),
        "s10_max_abs_target_delta_vs_pre_graft": float(condition.base.s10_max_abs_target_delta_vs_pre_graft),
        "s10_mean_graft_block_fraction": float(condition.base.s10_mean_graft_block_fraction),
        "s10_graft_sensitivity_score": float(condition.base.s10_graft_sensitivity_score),
        "interface_rule_id": condition.interface_rule.rule_id,
        "interface_mechanism_family": condition.interface_rule.mechanism_family,
        "explicit_recognition_used": bool(condition.interface_rule.explicit_recognition_used),
        "recognition_access_class": condition.interface_rule.recognition_access_class,
        "interface_access_stratum": interface_access_stratum(condition.interface_rule),
        "governance_mechanism_id": condition.governance.mechanism_id,
        "governance_mechanism_family": condition.governance.mechanism_family,
        "governance_access_stratum": governance_access_stratum(condition.governance),
        "intervention_stratum": intervention_stratum(condition.interface_rule, condition.governance),
        "governance_influence_radius_cells": condition.governance.influence_radius_cells,
        "governance_influence_radius_label": condition.governance.influence_radius_label,
        "governance_information_access_class": condition.governance.information_access_class,
        "global_controller_like": bool(condition.governance.global_controller_like),
        "array_size": int(config.array_size),
        "event_cap": int(config.event_cap),
        "events_executed": int(config.event_cap),
        "seed": int(seed),
        "scheduler": config.scheduler,
        "host_policy_id": condition.base.host_policy_id,
        "host_display_name": condition.base.host_display_name,
        "host_source_category": condition.base.host_source_category,
        "mutant_parent_policy_id": condition.base.mutant_parent_policy_id,
        "mutant_parent_display_name": condition.base.mutant_parent_display_name,
        "mutant_parent_source_category": condition.base.mutant_parent_source_category,
        "mutant_policy_id": condition.mutant_policy_id,
        "mutant_display_name": condition.mutant_display_name,
        "policy_ids_json": _json_list((condition.base.host_policy_id, condition.mutant_policy_id)),
        "display_names_json": _json_list((condition.base.host_display_name, condition.mutant_display_name)),
        "source_categories_json": _json_list((condition.base.host_source_category, "selfish_mutant_proxy")),
        "s10_policy_ids_json": _json_list(condition.base.policy_ids),
        "s10_display_names_json": _json_list(condition.base.display_names),
        "ratio_targets_json": _json_list(condition.base.ratio_targets),
        "goal_profile_id": goal_profile.profile_id,
        "goal_compatibility_class": goal_profile.compatibility_class,
        "goal_assignments_json": _json_object(goal_assignments),
        "value_profile": condition.base.value_profile,
        "perturbation_profile": condition.base.perturbation_profile,
        "frozen_indices_json": _json_list(sorted(frozen_indices)),
        "selfish_objective_id": condition.objective.objective_id,
        "selfish_objective_family": condition.objective.objective_family,
        "selfish_objective_score_name": condition.objective.score_name,
        "selfish_objective_definition": condition.objective.definition,
        "selfish_action_rule": condition.objective.action_rule,
        "selfish_allows_replication": bool(condition.objective.allows_replication),
        "selfish_gate_goal_proxy": condition.objective.gate_goal_proxy,
        "selfish_biological_analogy_caveat": condition.objective.biological_analogy_caveat,
        "initial_clone_state_id": initial_state_id,
        "clone_spec_id": condition.clone_spec.spec_id,
        "clone_description": condition.clone_spec.description,
        "clone_introduction_event": int(intro_event),
        "post_clone_event_count": int(len(post_schedule)),
        "clone_initial_size": int(condition.clone_spec.initial_size),
        "clone_initial_fraction": float(condition.clone_spec.initial_size / config.array_size),
        "clone_initial_location": condition.clone_spec.initial_location,
        "clone_placement_reference": condition.clone_spec.placement_reference,
        "clone_initial_start_index": int(layout["clone_start_index"]),
        "clone_initial_end_index": int(layout["clone_end_index"]),
        "clone_initial_indices_json": _json_list(layout["clone_indices"]),
        "clone_initial_cell_ids_json": _json_list(layout["initial_clone_cell_ids"]),
        "clone_mode": "staged_identity_mutant_proxy_preserving_values",
        "initial_counts_json": _json_object(dict(sorted(Counter([condition.base.host_policy_id] * (config.array_size - condition.clone_spec.initial_size) + [condition.mutant_policy_id] * condition.clone_spec.initial_size).items()))),
        "final_counts_json": _json_object(dict(sorted(final_counts.items()))),
        "total_count_success": bool(total_count_success),
        "nonreplicative_clone_count_preserved": bool(nonrep_count_success),
        "initial_inversion_count": int(initial_metrics["inversion_count"]),
        "initial_inversion_sortedness": float(initial_metrics["inversion_sortedness"]),
        "pre_clone_inversion_sortedness": float(pre_metrics["inversion_sortedness"]),
        "pre_clone_adjacent_sortedness": float(pre_metrics["adjacent_sortedness"]),
        "pre_clone_is_sorted": bool(pre_metrics["is_sorted"]),
        "final_inversion_count": int(final_metrics["inversion_count"]),
        "final_inversion_sortedness": float(final_metrics["inversion_sortedness"]),
        "final_adjacent_sortedness": float(final_metrics["adjacent_sortedness"]),
        "final_is_sorted": bool(final_metrics["is_sorted"]),
        "final_target_quality": float(goal_eval["assigned_policy_mean_sortedness"]),
        "reference_increasing_score": float(goal_eval["reference_increasing_score"]),
        "target_quality_damage": float(target_quality_damage),
        "reference_increasing_damage": float(reference_damage),
        "target_delta_vs_pre_clone": float(goal_eval["assigned_policy_mean_sortedness"] - pre_metrics["inversion_sortedness"]),
        "reference_delta_vs_pre_clone": float(goal_eval["reference_increasing_score"] - pre_metrics["inversion_sortedness"]),
        "goal_conflict_index": float(goal_eval["goal_alignment_gap"]),
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
        "mutant_outcome_class": mutant_outcome,
        "initial_selfish_objective_score": float(initial_selfish_score),
        "final_selfish_objective_score": float(final_selfish_score),
        "selfish_objective_score_delta": float(final_selfish_score - initial_selfish_score),
        "clone_growth_delta": int(clone_growth_delta),
        "clone_expansion_multiple": float(clone_final["final_clone_count"] / max(1, condition.clone_spec.initial_size)),
        "clone_fraction_peak": float(post_counts.get("clone_fraction_peak", initial_clone_metrics["final_clone_fraction"])),
        "containment_success": bool(float(clone_final["final_clone_fraction"]) < 0.25 and target_quality_damage < 0.08),
        "takeover_success": bool(float(clone_final["final_clone_fraction"]) >= 0.50),
        "damaging_clone": bool(target_quality_damage >= 0.08),
        "pre_compare_count": int(pre_counts.get("compare_count", 0)),
        "pre_swap_count": int(pre_counts.get("swap_count", 0)),
        "pre_update_count": int(pre_counts.get("update_count", 0)),
        "pre_wait_count": int(pre_counts.get("wait_count", 0)),
        "pre_frozen_block_count": int(pre_counts.get("frozen_block_count", 0)),
        "pre_invalid_action_count": int(pre_counts.get("invalid_action_count", 0)),
        "post_compare_count": int(post_counts.get("compare_count", 0)),
        "post_swap_count": int(post_counts.get("swap_count", 0)),
        "same_label_swap_count": int(post_counts.get("same_label_swap_count", 0)),
        "cross_label_swap_count": int(post_counts.get("cross_label_swap_count", 0)),
        "raw_swap_proposal_count": int(post_counts.get("raw_swap_proposal_count", 0)),
        "pre_governance_cross_label_proposal_count": int(post_counts.get("pre_governance_cross_label_proposal_count", 0)),
        "same_label_swap_proposal_count": int(post_counts.get("same_label_swap_proposal_count", 0)),
        "cross_label_swap_proposal_count": int(post_counts.get("cross_label_swap_proposal_count", 0)),
        "post_update_count": int(post_counts.get("update_count", 0)),
        "post_wait_count": int(post_counts.get("wait_count", 0)),
        "post_frozen_block_count": int(post_counts.get("frozen_block_count", 0)),
        "post_invalid_action_count": int(post_counts.get("invalid_action_count", 0)),
        "compare_count": int(pre_counts.get("compare_count", 0) + post_counts.get("compare_count", 0)),
        "swap_count": int(pre_counts.get("swap_count", 0) + post_counts.get("swap_count", 0)),
        "update_count": int(pre_counts.get("update_count", 0) + post_counts.get("update_count", 0)),
        "wait_count": int(pre_counts.get("wait_count", 0) + post_counts.get("wait_count", 0)),
        "frozen_block_count": int(pre_counts.get("frozen_block_count", 0) + post_counts.get("frozen_block_count", 0)),
        "invalid_action_count": int(pre_counts.get("invalid_action_count", 0) + post_counts.get("invalid_action_count", 0)),
        "work_count": int(pre_counts.get("compare_count", 0) + post_counts.get("compare_count", 0) + pre_counts.get("swap_count", 0) + post_counts.get("swap_count", 0) + pre_counts.get("update_count", 0) + post_counts.get("update_count", 0)),
        "selfish_action_count": int(post_counts.get("selfish_action_count", 0)),
        "selfish_swap_proposal_count": int(post_counts.get("selfish_swap_proposal_count", 0)),
        "selfish_replicate_proposal_count": int(post_counts.get("selfish_replicate_proposal_count", 0)),
        "replication_conversion_count": int(post_counts.get("replication_conversion_count", 0)),
        "host_to_mutant_conversion_count": int(post_counts.get("host_to_mutant_conversion_count", 0)),
        "governance_invocation_count": int(post_counts.get("governance_invocation_count", 0)),
        "governance_block_count": int(post_counts.get("governance_block_count", 0)),
        "governance_cross_label_block_count": int(post_counts.get("governance_cross_label_block_count", 0)),
        "governance_cost_units": int(post_counts.get("governance_cost_units", 0)),
        "global_controller_invocation_count": int(post_counts.get("global_controller_invocation_count", 0)),
        "governance_block_fraction": float(post_counts.get("governance_block_count", 0) / max(1, post_counts.get("raw_swap_proposal_count", 0))),
        "governance_cost_per_event": float(post_counts.get("governance_cost_units", 0) / max(1, len(post_schedule))),
        "governance_local_goal_delta_per_invocation": float(post_counts.get("governance_local_goal_delta_sum", 0.0) / max(1, post_counts.get("governance_invocation_count", 0))),
        "governance_heterotypic_edge_delta_per_invocation": float(post_counts.get("governance_heterotypic_edge_delta_sum", 0.0) / max(1, post_counts.get("governance_invocation_count", 0))),
        "governance_block_reasons_json": str(post_counts.get("governance_block_reasons_json", "{}")),
        "interface_rule_invocation_count": int(post_counts.get("interface_rule_invocation_count", 0)),
        "interface_block_count": int(post_counts.get("interface_block_count", 0)),
        "interface_block_fraction": float(post_counts.get("interface_block_count", 0) / max(1, post_counts.get("cross_label_swap_proposal_count", 0))),
        "cross_label_swap_acceptance_fraction": float(post_counts.get("cross_label_swap_count", 0) / max(1, post_counts.get("cross_label_swap_proposal_count", 0))),
        "same_contact_delta_per_invocation": float(post_counts.get("same_contact_delta_sum", 0.0) / max(1, post_counts.get("interface_rule_invocation_count", 0))),
        "local_goal_delta_per_invocation": float(post_counts.get("local_goal_delta_sum", 0.0) / max(1, post_counts.get("interface_rule_invocation_count", 0))),
        "interface_block_reasons_json": str(post_counts.get("interface_block_reasons_json", "{}")),
        "initial_values_head_json": _json_list(initial_values[:20]),
        "post_intro_values_head_json": _json_list(post_intro_values[:20]),
        "post_intro_labels_head_json": _json_list(post_intro_labels[:20]),
        "post_intro_cell_ids_head_json": _json_list(post_intro_cell_ids[:20]),
        "final_values_head_json": _json_list(values[:20]),
        "final_values_tail_json": _json_list(values[-20:]),
        "final_labels_head_json": _json_list(labels[:20]),
        "final_labels_tail_json": _json_list(labels[-20:]),
        "elapsed_seconds": float(elapsed),
        **{f"post_intro_{key}": value for key, value in initial_agg.items()},
        **clone_final,
        **goal_eval,
    }
    return run_row, initial_state_row


def _simulate_mutant_task(task: tuple[MutantCondition, Mapping[str, Mapping[str, Any]], int, S11Config]) -> tuple[dict[str, Any], dict[str, Any]]:
    condition, records_by_id, seed, config = task
    return simulate_mutant_condition(condition, records_by_id, seed=int(seed), config=config)


def run_s11_sweep(
    records: Sequence[Mapping[str, Any]],
    s10_selected: pd.DataFrame,
    s10_runs: pd.DataFrame,
    s09_rankings: pd.DataFrame,
    config: S11Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or S11Config()
    selected = select_s11_base_contexts(s10_selected, s10_runs, config)
    observed_governance = set(s09_rankings["governance_mechanism_id"].astype(str)) if "governance_mechanism_id" in s09_rankings else set()
    expected_governance = {mechanism.mechanism_id for mechanism in config.governance_mechanisms}
    missing_governance = expected_governance - observed_governance
    if missing_governance:
        raise ValueError(f"S09 ranking table lacks required governance options: {sorted(missing_governance)}")
    conditions = build_s11_conditions(selected, config)
    records_by_id = {str(record["algotypeId"]): dict(record) for record in records}
    needed_policy_ids = {condition.base.host_policy_id for condition in conditions} | {condition.base.mutant_parent_policy_id for condition in conditions}
    missing = sorted(needed_policy_ids - set(records_by_id))
    if missing:
        raise ValueError(f"S11 selected policies missing from S01 library: {missing}")
    tasks = [(condition, records_by_id, int(seed), config) for condition in conditions for seed in config.seeds]
    if config.worker_count > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(config.worker_count)) as pool:
            outputs = list(pool.map(_simulate_mutant_task, tasks))
    else:
        outputs = [_simulate_mutant_task(task) for task in tasks]
    run_df = pd.DataFrame([item[0] for item in outputs])
    initial_state_df = pd.DataFrame([item[1] for item in outputs]).drop_duplicates(subset=["initial_clone_state_id"]).reset_index(drop=True)
    condition_rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_rows.append(
            {
                "condition_id": condition.condition_id,
                "base_context_id": condition.base.base_context_id,
                "selection_rank": int(condition.base.selection_rank),
                "s11_selection_reason": condition.base.s11_selection_reason,
                "clone_spec_id": condition.clone_spec.spec_id,
                "clone_introduction_event": int(min(condition.clone_spec.introduction_event, config.event_cap)),
                "clone_initial_size": int(condition.clone_spec.initial_size),
                "clone_initial_location": condition.clone_spec.initial_location,
                "clone_placement_reference": condition.clone_spec.placement_reference,
                "selfish_objective_id": condition.objective.objective_id,
                "selfish_objective_family": condition.objective.objective_family,
                "selfish_allows_replication": bool(condition.objective.allows_replication),
                "interface_rule_id": condition.interface_rule.rule_id,
                "interface_access_stratum": interface_access_stratum(condition.interface_rule),
                "governance_mechanism_id": condition.governance.mechanism_id,
                "governance_access_stratum": governance_access_stratum(condition.governance),
                "intervention_stratum": intervention_stratum(condition.interface_rule, condition.governance),
                "global_controller_like": bool(condition.governance.global_controller_like),
                "host_policy_id": condition.base.host_policy_id,
                "mutant_parent_policy_id": condition.base.mutant_parent_policy_id,
                "mutant_policy_id": condition.mutant_policy_id,
                "host_display_name": condition.base.host_display_name,
                "mutant_display_name": condition.mutant_display_name,
                "goal_profile_id": condition.base.goal_profile_id,
            }
        )
    return run_df, pd.DataFrame(condition_rows), selected, initial_state_df, objective_table(config.selfish_objectives)


def summarize_s11_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["base_context_id", "clone_spec_id", "selfish_objective_id", "interface_rule_id", "governance_mechanism_id"]
    for _, group in run_df.groupby(group_columns, sort=False):
        first = group.iloc[0]
        outcome_counts = group["mutant_outcome_class"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "base_context_id": first["base_context_id"],
                "clone_spec_id": first["clone_spec_id"],
                "clone_initial_size": int(first["clone_initial_size"]),
                "clone_initial_location": first["clone_initial_location"],
                "selfish_objective_id": first["selfish_objective_id"],
                "selfish_objective_family": first["selfish_objective_family"],
                "selfish_allows_replication": bool(first["selfish_allows_replication"]),
                "interface_rule_id": first["interface_rule_id"],
                "interface_access_stratum": first["interface_access_stratum"],
                "governance_mechanism_id": first["governance_mechanism_id"],
                "governance_access_stratum": first["governance_access_stratum"],
                "intervention_stratum": first["intervention_stratum"],
                "global_controller_like": bool(first["global_controller_like"]),
                "host_display_name": first["host_display_name"],
                "mutant_display_name": first["mutant_display_name"],
                "display_names_json": first["display_names_json"],
                "seed_count": int(group["seed"].nunique()),
                "run_count": int(len(group)),
                "mean_final_target_quality": float(group["final_target_quality"].mean()),
                "mean_reference_increasing_score": float(group["reference_increasing_score"].mean()),
                "mean_target_quality_damage": float(group["target_quality_damage"].mean()),
                "mean_reference_increasing_damage": float(group["reference_increasing_damage"].mean()),
                "mean_goal_conflict_index": float(group["goal_conflict_index"].mean()),
                "mean_final_clone_fraction": float(group["final_clone_fraction"].mean()),
                "mean_clone_growth_delta": float(group["clone_growth_delta"].mean()),
                "mean_clone_expansion_multiple": float(group["clone_expansion_multiple"].mean()),
                "mean_clone_fraction_peak": float(group["clone_fraction_peak"].mean()),
                "mean_clone_largest_block_fraction": float(group["clone_largest_block_fraction"].mean()),
                "mean_clone_edge_fraction": float(group["clone_edge_fraction"].mean()),
                "mean_selfish_objective_score_delta": float(group["selfish_objective_score_delta"].mean()),
                "takeover_rate": float(group["takeover_success"].mean()),
                "containment_success_rate": float(group["containment_success"].mean()),
                "damaging_clone_rate": float(group["damaging_clone"].mean()),
                "mean_replication_conversion_count": float(group["replication_conversion_count"].mean()),
                "mean_governance_block_fraction": float(group["governance_block_fraction"].mean()),
                "mean_governance_cost_per_event": float(group["governance_cost_per_event"].mean()),
                "mean_interface_block_fraction": float(group["interface_block_fraction"].mean()),
                "mean_work_count": float(group["work_count"].mean()),
                "invalid_action_count": int(group["invalid_action_count"].sum()),
                "total_count_success": bool(group["total_count_success"].all()),
                "nonreplicative_clone_count_preserved": bool(group["nonreplicative_clone_count_preserved"].all()),
                "mutant_outcome_class_mode": _mode(group["mutant_outcome_class"]),
                "mutant_outcome_counts_json": _json_object(outcome_counts),
            }
        )
    return pd.DataFrame(rows)


def compare_to_behavior_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    baseline = summary[(summary["interface_rule_id"] == "behavior_only") & (summary["governance_mechanism_id"] == "no_governance")].copy()
    metric_columns = [
        "mean_final_target_quality",
        "mean_reference_increasing_score",
        "mean_target_quality_damage",
        "mean_reference_increasing_damage",
        "mean_goal_conflict_index",
        "mean_final_clone_fraction",
        "mean_clone_growth_delta",
        "mean_clone_expansion_multiple",
        "mean_clone_fraction_peak",
        "mean_clone_largest_block_fraction",
        "mean_clone_edge_fraction",
        "mean_selfish_objective_score_delta",
        "takeover_rate",
        "containment_success_rate",
        "damaging_clone_rate",
        "mean_work_count",
    ]
    baseline = baseline[["base_context_id", "clone_spec_id", "selfish_objective_id", *metric_columns]].rename(
        columns={column: f"baseline_{column}" for column in metric_columns}
    )
    comparison = summary.merge(baseline, on=["base_context_id", "clone_spec_id", "selfish_objective_id"], how="left")
    for column in metric_columns:
        comparison[f"delta_{column.removeprefix('mean_')}"] = comparison[column] - comparison[f"baseline_{column}"]
    comparison["schema"] = COMPARISON_SCHEMA
    comparison["containment_effect_score"] = (
        (comparison["baseline_mean_final_clone_fraction"] - comparison["mean_final_clone_fraction"])
        + (comparison["baseline_mean_target_quality_damage"] - comparison["mean_target_quality_damage"])
        + (comparison["containment_success_rate"] - comparison["baseline_containment_success_rate"]) * 0.25
        - comparison["mean_governance_block_fraction"] * 0.02
        - comparison["mean_interface_block_fraction"] * 0.01
        - comparison["mean_governance_cost_per_event"] * 0.001
    )
    comparison["takeover_reduced"] = comparison["delta_takeover_rate"] < -0.01
    comparison["damage_reduced"] = comparison["delta_target_quality_damage"] < -0.01
    return comparison


def mutant_strata_summary(comparison: pd.DataFrame, config: S11Config) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for stratum, group in comparison.groupby("intervention_stratum", sort=False):
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "intervention_stratum": stratum,
                "interface_access_strata": ",".join(sorted(set(group["interface_access_stratum"].astype(str)))),
                "governance_access_strata": ",".join(sorted(set(group["governance_access_stratum"].astype(str)))),
                "condition_count": int(
                    group[["base_context_id", "clone_spec_id", "selfish_objective_id", "interface_rule_id", "governance_mechanism_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "mean_delta_final_clone_fraction": float(group["delta_final_clone_fraction"].mean()),
                "mean_delta_target_quality_damage": float(group["delta_target_quality_damage"].mean()),
                "mean_delta_takeover_rate": float(group["delta_takeover_rate"].mean()),
                "mean_delta_containment_success_rate": float(group["delta_containment_success_rate"].mean()),
                "mean_containment_effect_score": float(group["containment_effect_score"].mean()),
                "takeover_reduction_rate": float(group["takeover_reduced"].mean()),
                "damage_reduction_rate": float(group["damage_reduced"].mean()),
                "mean_governance_block_fraction": float(group["mean_governance_block_fraction"].mean()),
                "mean_interface_block_fraction": float(group["mean_interface_block_fraction"].mean()),
                "mean_governance_cost_per_event": float(group["mean_governance_cost_per_event"].mean()),
                "supportive_containment_by_threshold": bool(group["containment_effect_score"].mean() >= config.containment_effect_threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_containment_effect_score", ascending=False, kind="mergesort")


def validate_s11_outputs(
    run_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    selected: pd.DataFrame,
    initial_state_df: pd.DataFrame,
    objective_df: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    strata: pd.DataFrame,
    config: S11Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    seed_set = {int(seed) for seed in config.seeds}
    expected_objectives = {objective.objective_id for objective in config.selfish_objectives}
    expected_interfaces = {rule.rule_id for rule in config.interface_rules}
    expected_governance = {mechanism.mechanism_id for mechanism in config.governance_mechanisms}
    expected_clone_specs = {spec.spec_id for spec in config.clone_specs}
    base_clone_objective_sets = (
        run_df.groupby(["base_context_id", "clone_spec_id"])["selfish_objective_id"].agg(lambda values: set(map(str, values))).tolist()
        if not run_df.empty
        else []
    )
    base_clone_objective_interface_sets = (
        run_df.groupby(["base_context_id", "clone_spec_id", "selfish_objective_id"])["interface_rule_id"].agg(lambda values: set(map(str, values))).tolist()
        if not run_df.empty
        else []
    )
    base_clone_objective_interface_governance_sets = (
        run_df.groupby(["base_context_id", "clone_spec_id", "selfish_objective_id", "interface_rule_id"])["governance_mechanism_id"]
        .agg(lambda values: set(map(str, values)))
        .tolist()
        if not run_df.empty
        else []
    )
    condition_seed_sets = (
        run_df.groupby(["base_context_id", "clone_spec_id", "selfish_objective_id", "interface_rule_id", "governance_mechanism_id"])["seed"]
        .agg(lambda values: {int(value) for value in values})
        .tolist()
        if not run_df.empty
        else []
    )
    state_ids = set(initial_state_df["initial_clone_state_id"].astype(str)) if not initial_state_df.empty else set()
    run_state_ids = set(run_df["initial_clone_state_id"].astype(str)) if not run_df.empty else set()
    nonrep = run_df[~run_df["selfish_allows_replication"].astype(bool)] if not run_df.empty else pd.DataFrame()
    rep = run_df[run_df["selfish_allows_replication"].astype(bool)] if not run_df.empty else pd.DataFrame()
    finite_columns = ["delta_final_clone_fraction", "delta_target_quality_damage", "delta_takeover_rate", "containment_effect_score"]
    finite_comparison = bool(
        not comparison.empty
        and set(finite_columns).issubset(comparison.columns)
        and comparison[finite_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().all().all()
    )
    checks = [
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "selfish_objectives_explicitly_defined",
            "success": bool(
                set(objective_df["objective_id"].astype(str)) == expected_objectives
                and objective_df[["definition", "action_rule", "biological_analogy_caveat"]].astype(str).apply(lambda col: col.str.len().gt(0)).all().all()
            ),
            "observed": sorted(objective_df["objective_id"].astype(str).tolist()) if not objective_df.empty else [],
            "expected": sorted(expected_objectives),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "selected_contexts_trace_to_s10_graft_sensitive_rows",
            "success": bool(not selected.empty and selected["s10_run_count"].gt(0).all() and selected["s11_selection_reason"].astype(str).str.contains("s10_graft_sensitive").all()),
            "observed": selected[["base_context_id", "s10_graft_sensitivity_score"]].to_dict(orient="records") if not selected.empty else [],
            "expected": "S11 uses S10 selected contexts with computed graft-sensitivity scores",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "clone_initial_size_location_logged",
            "success": bool(
                not run_df.empty
                and run_df[
                    [
                        "clone_initial_size",
                        "clone_initial_location",
                        "clone_initial_start_index",
                        "clone_initial_end_index",
                        "clone_initial_indices_json",
                    ]
                ]
                .notna()
                .all()
                .all()
            ),
            "observed": sorted(run_df["clone_spec_id"].astype(str).unique().tolist()) if not run_df.empty else [],
            "expected": sorted(expected_clone_specs),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "initial_clone_states_saved",
            "success": bool(run_state_ids and run_state_ids.issubset(state_ids) and initial_state_df["initial_labels_json"].astype(str).str.startswith("[").all()),
            "observed": f"run_state_ids={len(run_state_ids)} saved_state_ids={len(state_ids)}",
            "expected": "every run references a saved full initial clone state",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "behavior_explicit_finite_global_strata_preserved",
            "success": bool(
                not run_df.empty
                and {"behavior_only", "explicit_interface"}.issubset(set(run_df["interface_access_stratum"].astype(str)))
                and {"no_governance", "finite_radius_governance", "global_controller_like"}.issubset(set(run_df["governance_access_stratum"].astype(str)))
                and "global_controller_like" in set(run_df["intervention_stratum"].astype(str))
            ),
            "observed": sorted(run_df["intervention_stratum"].astype(str).unique().tolist()) if not run_df.empty else [],
            "expected": "behavior-only, explicit-interface, finite-radius governance, and global-controller-like strata are separate",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "all_selfish_objectives_per_base_clone",
            "success": bool(base_clone_objective_sets and all(item == expected_objectives for item in base_clone_objective_sets)),
            "observed": f"base_clone_groups={len(base_clone_objective_sets)}",
            "expected": sorted(expected_objectives),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "all_interface_rules_per_base_clone_objective",
            "success": bool(base_clone_objective_interface_sets and all(item == expected_interfaces for item in base_clone_objective_interface_sets)),
            "observed": f"base_clone_objective_groups={len(base_clone_objective_interface_sets)}",
            "expected": sorted(expected_interfaces),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "all_governance_mechanisms_per_base_clone_objective_interface",
            "success": bool(
                base_clone_objective_interface_governance_sets
                and all(item == expected_governance for item in base_clone_objective_interface_governance_sets)
            ),
            "observed": f"base_clone_objective_interface_groups={len(base_clone_objective_interface_governance_sets)}",
            "expected": sorted(expected_governance),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "matched_seed_sets_by_condition",
            "success": bool(condition_seed_sets and all(item == seed_set for item in condition_seed_sets)),
            "observed": f"condition_groups={len(condition_seed_sets)} expected_seeds={sorted(seed_set)}",
            "expected": "each mutant clone condition uses the configured matched seed set",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "clone_counts_bounded_and_total_size_preserved",
            "success": bool(
                not run_df.empty
                and run_df["total_count_success"].all()
                and run_df["final_clone_count"].between(0, config.array_size).all()
                and (nonrep.empty or nonrep["nonreplicative_clone_count_preserved"].all())
            ),
            "observed": f"total_failures={int((~run_df['total_count_success']).sum()) if not run_df.empty else 'missing'} nonrep_count_failures={int((~nonrep['nonreplicative_clone_count_preserved']).sum()) if not nonrep.empty else 0}",
            "expected": "array size is preserved; non-replicative objectives keep clone count fixed",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "replicative_objective_changes_clone_count",
            "success": bool(not rep.empty and int(rep["replication_conversion_count"].sum()) > 0 and (rep["final_clone_count"] >= rep["clone_initial_size"]).all()),
            "observed": f"rep_rows={len(rep)} conversions={int(rep['replication_conversion_count'].sum()) if not rep.empty else 0}",
            "expected": "replicative takeover objective performs host-to-mutant identity conversions",
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
            "validation_case": "containment_takeover_categories_quantified",
            "success": bool(not summary.empty and summary["mutant_outcome_counts_json"].astype(str).str.startswith("{").all()),
            "observed": sorted(run_df["mutant_outcome_class"].astype(str).unique().tolist()) if not run_df.empty else [],
            "expected": "summary records per-condition mutant outcome class counts",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "finite_baseline_comparisons_and_strata",
            "success": bool(finite_comparison and not strata.empty and strata["mean_containment_effect_score"].notna().all()),
            "observed": f"comparison_rows={len(comparison)} strata_rows={len(strata)}",
            "expected": "all mutant baseline deltas and stratum summaries are finite",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "figure_written",
            "success": bool(figure_written),
            "observed": str(figure_written),
            "expected": "mutant_takeover_curves.png written and non-empty",
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
