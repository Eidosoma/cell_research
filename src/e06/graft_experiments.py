"""E06 S10 staged graft experiments for chimeric collectives."""

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
from src.e06.governance_mechanisms import (
    DEFAULT_GOVERNANCE_MECHANISMS,
    DEFAULT_INTERFACE_RULES,
    GovernanceMechanism,
    governance_decision,
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


STEP_ID = "S10"
GRAFT_SCHEMA = "eidosoma.e06.s10_graft_experiment_run.v1"
PRE_GRAFT_SCHEMA = "eidosoma.e06.s10_pre_graft_state.v1"
SUMMARY_SCHEMA = "eidosoma.e06.s10_graft_summary.v1"
COMPARISON_SCHEMA = "eidosoma.e06.s10_graft_baseline_comparison.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s10_validation.v1"


@dataclass(frozen=True)
class GraftSpec:
    """One staged graft schedule and placement."""

    spec_id: str
    graft_event: int
    graft_size: int
    graft_location: str
    arrangement_reference: str
    description: str


DEFAULT_GRAFT_SPECS = (
    GraftSpec(
        spec_id="early_center_patch_10",
        graft_event=1_000,
        graft_size=10,
        graft_location="center",
        arrangement_reference="graft_like_insertions",
        description="Early center graft after 1,000 host-development events, placed by S03 graft-like insertion layout.",
    ),
    GraftSpec(
        spec_id="late_left_edge_patch_15",
        graft_event=2_400,
        graft_size=15,
        graft_location="left_edge",
        arrangement_reference="contiguous_patch",
        description="Late edge graft after 2,400 host-development events, placed by S03 contiguous-patch layout.",
    ),
)


@dataclass(frozen=True)
class S10Config:
    """Configuration for bounded S10 staged graft tests."""

    array_size: int = 100
    event_cap: int = 4_000
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    max_base_contexts: int = 3
    graft_specs: tuple[GraftSpec, ...] = DEFAULT_GRAFT_SPECS
    interface_rules: tuple[InterfaceRule, ...] = DEFAULT_INTERFACE_RULES
    governance_mechanisms: tuple[GovernanceMechanism, ...] = DEFAULT_GOVERNANCE_MECHANISMS
    scheduler: str = "cyclic_scan_seed_offset"
    worker_count: int = 1
    graft_rescue_delta_threshold: float = 0.02


@dataclass(frozen=True)
class GraftBaseContext:
    """One S07 context converted into a host/graft staged experiment."""

    base_context_id: str
    source_s07_run_id: str
    source_research_step_id: str
    source_condition_id: str
    selection_rank: int
    selection_reason: str
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
    graft_policy_id: str
    host_display_name: str
    graft_display_name: str
    host_source_category: str
    graft_source_category: str
    host_policy_index: int
    graft_policy_index: int
    value_profile: str
    perturbation_profile: str
    source_final_target_quality: float
    source_goal_conflict_index: float
    source_aggregation_delta_percent: float
    source_largest_block_fraction: float
    source_mosaic_continuum_score: float
    source_conflict_continuum_score: float


@dataclass(frozen=True)
class GraftCondition:
    """One executable staged graft condition before seed expansion."""

    condition_id: str
    base: GraftBaseContext
    graft_spec: GraftSpec
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


def _normalize_text(value: Any, default: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return text


def _condition_key(row: Mapping[str, Any]) -> str:
    return _json_object(
        {
            "sourceRunId": str(row.get("run_id", "")),
            "sourceConditionId": str(row.get("source_condition_id", "")),
            "policyIds": _parse_json_list(row.get("policy_ids_json")),
            "ratios": [round(float(value), 6) for value in _parse_json_list(row.get("ratio_targets_json"))],
            "goalProfileId": _normalize_text(row.get("goal_profile_id"), "same_increasing"),
            "arrangement": _normalize_text(row.get("arrangement"), "random_permutation"),
            "valueProfile": _normalize_text(row.get("value_profile"), "random_permutation"),
            "perturbationProfile": _normalize_text(row.get("perturbation_profile"), "none"),
        }
    )


def _base_context_id(row: Mapping[str, Any]) -> str:
    return f"s10_base_{stable_hash(_condition_key(row))[:16]}"


def _condition_id(base_context_id: str, graft_spec_id: str, interface_rule_id: str, governance_id: str) -> str:
    payload = {
        "baseContextId": base_context_id,
        "graftSpecId": graft_spec_id,
        "interfaceRuleId": interface_rule_id,
        "governanceMechanismId": governance_id,
    }
    return f"s10_{stable_hash(payload)[:16]}"


def _state_id(condition_id: str, seed: int) -> str:
    return f"s10_state_{stable_hash({'conditionId': condition_id, 'seed': int(seed)})[:18]}"


def _mode(values: pd.Series) -> str:
    mode = values.astype(str).mode()
    return str(mode.iloc[0]) if not mode.empty else ""


def _pairwise_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "policy_ids_json" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["policy_ids_json"].map(lambda text: len(_parse_json_list(text)) == 2)].copy()


def _host_graft_indices(ratios: Sequence[float]) -> tuple[int, int]:
    if len(ratios) != 2:
        raise ValueError("S10 graft contexts require pairwise ratios")
    if abs(float(ratios[0]) - float(ratios[1])) < 1e-12:
        return 0, 1
    host_idx = 0 if float(ratios[0]) > float(ratios[1]) else 1
    return host_idx, 1 - host_idx


def select_s10_base_contexts(mosaic: pd.DataFrame, exemplars: pd.DataFrame, config: S10Config) -> pd.DataFrame:
    """Select bounded S07 exemplar and continuum contexts for staged grafts."""

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
        "mosaic_continuum_score",
        "conflict_continuum_score",
    }
    missing = required - set(mosaic.columns)
    if missing:
        raise ValueError(f"S07 mosaic table missing columns for S10: {sorted(missing)}")
    pairwise = _pairwise_rows(mosaic)
    exemplar_pairs = _pairwise_rows(exemplars)
    if pairwise.empty:
        raise ValueError("S10 could not find pairwise S07 contexts")

    rows_by_key: dict[str, dict[str, Any]] = {}
    reasons_by_key: dict[str, list[str]] = {}

    def add_rows(rows: pd.DataFrame, reason: str, *, limit: int | None = None) -> None:
        added = 0
        for raw in rows.to_dict(orient="records"):
            if len(rows_by_key) >= config.max_base_contexts:
                return
            policy_ids = _parse_json_list(raw.get("policy_ids_json"))
            ratios = _parse_json_list(raw.get("ratio_targets_json"))
            if len(policy_ids) != 2 or len(ratios) != 2:
                continue
            key = _condition_key(raw)
            if key not in rows_by_key:
                rows_by_key[key] = dict(raw)
                reasons_by_key[key] = []
                added += 1
            reasons_by_key[key].append(reason)
            if limit is not None and added >= limit:
                return

    if not exemplar_pairs.empty:
        exemplar_pairs = exemplar_pairs.sort_values(["s07_label", "source_research_step_id", "run_id"], kind="mergesort")
        add_rows(exemplar_pairs, "s07_exemplar", limit=1)

    extreme_specs = [
        ("metric_extreme:high_conflict_continuum", "conflict_continuum_score", False),
        ("metric_extreme:high_mosaic_continuum", "mosaic_continuum_score", False),
        ("metric_extreme:low_target_quality", "final_target_quality", True),
        ("metric_extreme:high_aggregation", "aggregation_delta_percent", False),
    ]
    for reason, metric, ascending in extreme_specs:
        if len(rows_by_key) >= config.max_base_contexts:
            break
        ranked = pairwise.copy()
        ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
        ranked = ranked.dropna(subset=[metric]).sort_values(metric, ascending=ascending, kind="mergesort")
        add_rows(ranked.head(8), reason, limit=1)

    if len(rows_by_key) < config.max_base_contexts:
        ranked = pairwise.copy()
        ranked["selection_score"] = (
            pd.to_numeric(ranked["conflict_continuum_score"], errors="coerce").fillna(0.0)
            + pd.to_numeric(ranked["mosaic_continuum_score"], errors="coerce").fillna(0.0)
            + (1.0 - pd.to_numeric(ranked["final_target_quality"], errors="coerce").fillna(0.0))
        )
        ranked = ranked.sort_values("selection_score", ascending=False, kind="mergesort")
        add_rows(ranked, "metric_extreme:combined_graft_relevance")

    rows: list[dict[str, Any]] = []
    for rank, (key, raw) in enumerate(rows_by_key.items(), start=1):
        policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
        display_names = tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
        categories = tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
        ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
        host_idx, graft_idx = _host_graft_indices(ratios)
        rows.append(
            {
                "base_context_id": _base_context_id(raw),
                "selection_rank": int(rank),
                "selection_reason": ",".join(dict.fromkeys(reasons_by_key[key])),
                "source_s07_run_id": str(raw["run_id"]),
                "source_research_step_id": str(raw["source_research_step_id"]),
                "source_condition_id": str(raw["source_condition_id"]),
                "s07_label": str(raw["s07_label"]),
                "s07_classification_mode": str(raw["s07_classification_mode"]),
                "panel": str(raw.get("panel", "")),
                "candidate_reason": str(raw.get("candidate_reason", "")),
                "source_arrangement": _normalize_text(raw.get("arrangement"), "random_permutation"),
                "goal_profile_id": _normalize_text(raw.get("goal_profile_id"), "same_increasing"),
                "goal_compatibility_class": _normalize_text(raw.get("goal_compatibility_class"), "same_goal"),
                "policy_ids_json": _json_list(policy_ids),
                "display_names_json": _json_list(display_names),
                "source_categories_json": _json_list(categories),
                "ratio_targets_json": _json_list(ratios),
                "host_policy_id": policy_ids[host_idx],
                "graft_policy_id": policy_ids[graft_idx],
                "host_display_name": display_names[host_idx],
                "graft_display_name": display_names[graft_idx],
                "host_source_category": categories[host_idx],
                "graft_source_category": categories[graft_idx],
                "host_policy_index": int(host_idx),
                "graft_policy_index": int(graft_idx),
                "value_profile": _normalize_text(raw.get("value_profile"), "random_permutation"),
                "perturbation_profile": _normalize_text(raw.get("perturbation_profile"), "none"),
                "source_final_target_quality": float(raw["final_target_quality"]),
                "source_goal_conflict_index": float(raw["goal_conflict_index"]),
                "source_aggregation_delta_percent": float(raw["aggregation_delta_percent"]),
                "source_largest_block_fraction": float(raw["largest_block_fraction"]),
                "source_mosaic_continuum_score": float(raw["mosaic_continuum_score"]),
                "source_conflict_continuum_score": float(raw["conflict_continuum_score"]),
            }
        )
    selected = pd.DataFrame(rows)
    if selected.empty:
        raise ValueError("S10 could not select graft contexts")
    return selected


def build_s10_conditions(selected: pd.DataFrame, config: S10Config) -> list[GraftCondition]:
    conditions: list[GraftCondition] = []
    for raw in selected.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
        display_names = tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
        categories = tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
        ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
        base = GraftBaseContext(
            base_context_id=str(raw["base_context_id"]),
            source_s07_run_id=str(raw["source_s07_run_id"]),
            source_research_step_id=str(raw["source_research_step_id"]),
            source_condition_id=str(raw["source_condition_id"]),
            selection_rank=int(raw["selection_rank"]),
            selection_reason=str(raw["selection_reason"]),
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
            graft_policy_id=str(raw["graft_policy_id"]),
            host_display_name=str(raw["host_display_name"]),
            graft_display_name=str(raw["graft_display_name"]),
            host_source_category=str(raw["host_source_category"]),
            graft_source_category=str(raw["graft_source_category"]),
            host_policy_index=int(raw["host_policy_index"]),
            graft_policy_index=int(raw["graft_policy_index"]),
            value_profile=str(raw["value_profile"]),
            perturbation_profile=str(raw["perturbation_profile"]),
            source_final_target_quality=float(raw["source_final_target_quality"]),
            source_goal_conflict_index=float(raw["source_goal_conflict_index"]),
            source_aggregation_delta_percent=float(raw["source_aggregation_delta_percent"]),
            source_largest_block_fraction=float(raw["source_largest_block_fraction"]),
            source_mosaic_continuum_score=float(raw["source_mosaic_continuum_score"]),
            source_conflict_continuum_score=float(raw["source_conflict_continuum_score"]),
        )
        for graft_spec in config.graft_specs:
            for interface_rule in config.interface_rules:
                for governance in config.governance_mechanisms:
                    conditions.append(
                        GraftCondition(
                            condition_id=_condition_id(base.base_context_id, graft_spec.spec_id, interface_rule.rule_id, governance.mechanism_id),
                            base=base,
                            graft_spec=graft_spec,
                            interface_rule=interface_rule,
                            governance=governance,
                        )
                    )
    return conditions


def interface_access_stratum(rule: InterfaceRule) -> str:
    return "behavior_only" if not rule.explicit_recognition_used else "explicit_interface"


def governance_access_stratum(mechanism: GovernanceMechanism) -> str:
    if mechanism.mechanism_id == "no_governance":
        return "no_governance"
    if mechanism.global_controller_like:
        return "global_controller_like"
    return "finite_radius_governance"


def intervention_stratum(rule: InterfaceRule, mechanism: GovernanceMechanism) -> str:
    interface = interface_access_stratum(rule)
    governance = governance_access_stratum(mechanism)
    if interface == "behavior_only" and governance == "no_governance":
        return "behavior_only_no_governance"
    if interface == "explicit_interface" and governance == "no_governance":
        return "explicit_interface_only"
    if governance == "global_controller_like":
        return "global_controller_like"
    if interface == "behavior_only":
        return "finite_radius_governance_behavior_only"
    return "finite_radius_governance_with_explicit_interface"


def graft_layout_labels(host_policy_id: str, graft_policy_id: str, graft_spec: GraftSpec, array_size: int, seed: int, condition_id: str) -> tuple[str, ...]:
    host_count = int(array_size - graft_spec.graft_size)
    graft_count = int(graft_spec.graft_size)
    if graft_spec.graft_location == "center":
        return labels_for_arrangement(
            (host_policy_id, graft_policy_id),
            (host_count, graft_count),
            seed=seed,
            condition_id=condition_id,
            arrangement="graft_like_insertions",
        )
    if graft_spec.graft_location == "left_edge":
        return labels_for_arrangement(
            (graft_policy_id, host_policy_id),
            (graft_count, host_count),
            seed=seed,
            condition_id=condition_id,
            arrangement="contiguous_patch",
        )
    if graft_spec.graft_location == "right_edge":
        return labels_for_arrangement(
            (host_policy_id, graft_policy_id),
            (host_count, graft_count),
            seed=seed,
            condition_id=condition_id,
            arrangement="contiguous_patch",
        )
    raise ValueError(f"unknown graft location: {graft_spec.graft_location}")


def apply_graft_patch(
    labels: list[str],
    cell_ids: list[int],
    *,
    host_policy_id: str,
    graft_policy_id: str,
    graft_spec: GraftSpec,
    seed: int,
    condition_id: str,
) -> dict[str, Any]:
    layout = graft_layout_labels(host_policy_id, graft_policy_id, graft_spec, len(labels), seed, condition_id)
    graft_indices = [idx for idx, label in enumerate(layout) if str(label) == str(graft_policy_id)]
    if len(graft_indices) != graft_spec.graft_size:
        raise RuntimeError("S03 graft layout produced the wrong graft size")
    for idx, label in enumerate(layout):
        labels[idx] = str(label)
    base_new_id = len(cell_ids)
    for offset, idx in enumerate(graft_indices):
        cell_ids[idx] = base_new_id + offset
    return {
        "graft_indices": tuple(graft_indices),
        "graft_start_index": int(min(graft_indices)),
        "graft_end_index": int(max(graft_indices)),
        "graft_layout_labels": tuple(layout),
    }


def _block_state(memory: dict[int, dict[str, Any]], cell_id: int, action_type: str) -> None:
    state = memory.setdefault(cell_id, {})
    state["last_move_success"] = False
    state["last_action_type"] = action_type
    state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)


def _success_state(memory: dict[int, dict[str, Any]], cell_id: int) -> None:
    state = memory.setdefault(cell_id, {})
    state["last_move_success"] = True
    state["last_action_type"] = "swap"
    state["time_since_movement"] = 0


def _wait_state(memory: dict[int, dict[str, Any]], cell_id: int) -> None:
    state = memory.setdefault(cell_id, {})
    state["last_move_success"] = None
    state["last_action_type"] = "wait"
    state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)


def _run_host_development(
    *,
    values: list[int],
    labels: list[str],
    cell_ids: list[int],
    memory: dict[int, dict[str, Any]],
    runtime: MixtureRuntime,
    schedule: Sequence[int],
    goal_assignments: Mapping[str, str],
    frozen_indices: frozenset[int],
    array_size: int,
    rng: random.Random,
) -> dict[str, int]:
    counts = Counter()
    for actor_index in schedule:
        actor_index = int(actor_index)
        policy_id = str(labels[actor_index])
        cell_id = cell_ids[actor_index]
        actor_goal = goal_assignments[policy_id]
        proxy_values = rank_values(values, actor_goal, array_size)
        action = runtime.action_for(policy_id, proxy_values, labels, actor_index, cell_id, memory, rng)
        counts["compare_count"] += int(bool(action.compare_counted))
        target = None if action.target_index is None else int(action.target_index)
        if target is not None and (actor_index in frozen_indices or target in frozen_indices):
            counts["frozen_block_count"] += 1
            counts["wait_count"] += 1
            _block_state(memory, cell_id, "pre_graft_frozen_block")
            continue
        if action.action_type == "swap" and target is not None:
            if 0 <= target < len(values):
                values[actor_index], values[target] = values[target], values[actor_index]
                labels[actor_index], labels[target] = labels[target], labels[actor_index]
                cell_ids[actor_index], cell_ids[target] = cell_ids[target], cell_ids[actor_index]
                counts["swap_count"] += 1
                _success_state(memory, cell_id)
            else:
                counts["invalid_action_count"] += 1
                counts["wait_count"] += 1
                _block_state(memory, cell_id, "pre_graft_invalid_swap")
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
    return {str(key): int(value) for key, value in counts.items()}


def _graft_cell_metrics(labels: Sequence[str], cell_ids: Sequence[int], graft_policy_id: str, graft_cell_ids: Sequence[int]) -> dict[str, Any]:
    graft_positions = [idx for idx, cell_id in enumerate(cell_ids) if int(cell_id) in set(int(item) for item in graft_cell_ids)]
    graft_label_positions = [idx for idx, label in enumerate(labels) if str(label) == str(graft_policy_id)]
    n = len(labels)
    graft_size = max(1, len(graft_cell_ids))
    if not graft_positions:
        return {
            "graft_cell_positions_json": "[]",
            "graft_position_mean": math.nan,
            "graft_position_min": math.nan,
            "graft_position_max": math.nan,
            "graft_position_span": math.nan,
            "graft_edge_fraction": math.nan,
            "graft_relative_largest_block_fraction": 0.0,
            "graft_label_position_mean": math.nan,
            "graft_cell_label_retention_fraction": 0.0,
        }
    runs: list[int] = []
    current = 0
    for label in labels:
        if str(label) == str(graft_policy_id):
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    edge_window = max(1, int(round(n * 0.15)))
    edge_count = sum(1 for pos in graft_positions if pos < edge_window or pos >= n - edge_window)
    retained = sum(1 for pos in graft_positions if str(labels[pos]) == str(graft_policy_id))
    return {
        "graft_cell_positions_json": _json_list(graft_positions),
        "graft_position_mean": float(sum(graft_positions) / len(graft_positions)),
        "graft_position_min": int(min(graft_positions)),
        "graft_position_max": int(max(graft_positions)),
        "graft_position_span": int(max(graft_positions) - min(graft_positions) + 1),
        "graft_edge_fraction": float(edge_count / len(graft_positions)),
        "graft_relative_largest_block_fraction": float(max(runs) / graft_size) if runs else 0.0,
        "graft_label_position_mean": float(sum(graft_label_positions) / len(graft_label_positions)) if graft_label_positions else math.nan,
        "graft_cell_label_retention_fraction": float(retained / len(graft_positions)),
    }


def classify_graft_outcome(*, final_target_quality: float, pre_graft_reference_score: float, graft_relative_largest_block_fraction: float, graft_edge_fraction: float, graft_position_span: float, graft_size: int, interface_count: int) -> str:
    target_delta = float(final_target_quality - pre_graft_reference_score)
    relative_span = float(graft_position_span / max(1, graft_size))
    if graft_relative_largest_block_fraction >= 0.70 and graft_edge_fraction >= 0.50:
        return "rejected_or_edge_segregated"
    if graft_relative_largest_block_fraction >= 0.70 and interface_count <= 4:
        return "stable_segregated_patch"
    if graft_relative_largest_block_fraction <= 0.35 and relative_span >= 3.0:
        return "absorbed_mixed"
    if target_delta >= 0.08 and relative_span >= 2.0:
        return "graft_reorganized_host"
    return "partial_mosaic_or_contained"


def _post_graft_intervention_loop(
    *,
    values: list[int],
    labels: list[str],
    cell_ids: list[int],
    memory: dict[int, dict[str, Any]],
    runtime: MixtureRuntime,
    schedule: Sequence[int],
    goal_assignments: Mapping[str, str],
    frozen_indices: frozenset[int],
    condition: GraftCondition,
    config: S10Config,
    rng: random.Random,
) -> dict[str, Any]:
    counts = Counter()
    same_contact_delta_sum = 0.0
    local_goal_delta_sum = 0.0
    governance_local_goal_delta_sum = 0.0
    governance_heterotypic_edge_delta_sum = 0.0
    governance_block_reasons: Counter[str] = Counter()
    interface_block_reasons: Counter[str] = Counter()
    for actor_index in schedule:
        actor_index = int(actor_index)
        policy_id = str(labels[actor_index])
        cell_id = cell_ids[actor_index]
        actor_goal = goal_assignments[policy_id]
        proxy_values = rank_values(values, actor_goal, config.array_size)
        action = runtime.action_for(policy_id, proxy_values, labels, actor_index, cell_id, memory, rng)
        counts["compare_count"] += int(bool(action.compare_counted))
        target = None if action.target_index is None else int(action.target_index)
        if target is not None and (actor_index in frozen_indices or target in frozen_indices):
            counts["frozen_block_count"] += 1
            counts["wait_count"] += 1
            _block_state(memory, cell_id, "post_graft_frozen_block")
            continue
        if action.action_type == "swap" and target is not None:
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
                    policy_ids=(condition.base.host_policy_id, condition.base.graft_policy_id),
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
                _block_state(memory, cell_id, "post_graft_invalid_swap")
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
    }


def simulate_graft_condition(
    condition: GraftCondition,
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: S10Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    initial_values = list(initial_values_for_profile(config.array_size, seed, condition.base.value_profile))
    values = list(initial_values)
    labels = [condition.base.host_policy_id] * config.array_size
    cell_ids = list(range(config.array_size))
    memory: dict[int, dict[str, Any]] = {}
    runtime = MixtureRuntime([records_by_id[condition.base.host_policy_id], records_by_id[condition.base.graft_policy_id]], seed)
    full_schedule = actor_schedule(config.array_size, config.event_cap, seed)
    graft_event = min(max(0, int(condition.graft_spec.graft_event)), config.event_cap)
    pre_schedule = full_schedule[:graft_event]
    post_schedule = full_schedule[graft_event:]
    pre_rng = random.Random(
        seed
        ^ int(
            stable_hash(
                {
                    "s10PreGraft": condition.base.base_context_id,
                    "graftSpecId": condition.graft_spec.spec_id,
                }
            )[:12],
            16,
        )
    )
    post_rng = random.Random(seed ^ int(stable_hash({"s10PostGraft": condition.condition_id})[:12], 16))
    goal_profile = goal_profile_for_id(condition.base.goal_profile_id)
    goal_assignments = goal_assignments_for_profile((condition.base.host_policy_id, condition.base.graft_policy_id), goal_profile)
    frozen_indices = frozen_indices_for_profile(config.array_size, condition.base.perturbation_profile)

    initial_metrics = sortedness_metrics(values)
    pre_counts = _run_host_development(
        values=values,
        labels=labels,
        cell_ids=cell_ids,
        memory=memory,
        runtime=runtime,
        schedule=pre_schedule,
        goal_assignments=goal_assignments,
        frozen_indices=frozen_indices,
        array_size=config.array_size,
        rng=pre_rng,
    )
    pre_metrics = sortedness_metrics(values)
    pre_labels = tuple(labels)
    pre_values = tuple(values)
    pre_cell_ids = tuple(cell_ids)
    pre_arrangement = arrangement_metric_row(pre_labels, (condition.base.host_policy_id, condition.base.graft_policy_id))
    state_id = _state_id(f"{condition.base.base_context_id}:{condition.graft_spec.spec_id}", seed)
    layout = apply_graft_patch(
        labels,
        cell_ids,
        host_policy_id=condition.base.host_policy_id,
        graft_policy_id=condition.base.graft_policy_id,
        graft_spec=condition.graft_spec,
        seed=seed,
        condition_id=condition.condition_id,
    )
    graft_cell_ids = tuple(cell_ids[idx] for idx in layout["graft_indices"])
    post_graft_counts = Counter(labels)
    post_counts = _post_graft_intervention_loop(
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
    dominance = dominance_metrics(labels, (condition.base.host_policy_id, condition.base.graft_policy_id))
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
    graft_metrics = _graft_cell_metrics(labels, cell_ids, condition.base.graft_policy_id, graft_cell_ids)
    graft_outcome = classify_graft_outcome(
        final_target_quality=float(goal_eval["assigned_policy_mean_sortedness"]),
        pre_graft_reference_score=float(pre_metrics["inversion_sortedness"]),
        graft_relative_largest_block_fraction=float(graft_metrics["graft_relative_largest_block_fraction"]),
        graft_edge_fraction=float(graft_metrics["graft_edge_fraction"]),
        graft_position_span=float(graft_metrics["graft_position_span"]),
        graft_size=condition.graft_spec.graft_size,
        interface_count=int(final_agg["interface_count"]),
    )
    expected_counts = {
        condition.base.host_policy_id: int(config.array_size - condition.graft_spec.graft_size),
        condition.base.graft_policy_id: int(condition.graft_spec.graft_size),
    }
    pre_state_row = {
        "schema": PRE_GRAFT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "pre_graft_state_id": state_id,
        "base_context_id": condition.base.base_context_id,
        "graft_spec_id": condition.graft_spec.spec_id,
        "seed": int(seed),
        "host_policy_id": condition.base.host_policy_id,
        "graft_policy_id": condition.base.graft_policy_id,
        "graft_event": int(graft_event),
        "graft_size": int(condition.graft_spec.graft_size),
        "graft_location": condition.graft_spec.graft_location,
        "pre_graft_values_json": _json_list(pre_values),
        "pre_graft_labels_json": _json_list(pre_labels),
        "pre_graft_cell_ids_json": _json_list(pre_cell_ids),
        "pre_graft_inversion_sortedness": float(pre_metrics["inversion_sortedness"]),
        "pre_graft_adjacent_sortedness": float(pre_metrics["adjacent_sortedness"]),
        "pre_graft_is_sorted": bool(pre_metrics["is_sorted"]),
        "pre_graft_counts_json": _json_object(dict(sorted(Counter(pre_labels).items()))),
    }
    run_row = {
        "schema": GRAFT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "condition_id": condition.condition_id,
        "base_context_id": condition.base.base_context_id,
        "source_s07_run_id": condition.base.source_s07_run_id,
        "source_research_step_id": condition.base.source_research_step_id,
        "source_condition_id": condition.base.source_condition_id,
        "selection_rank": int(condition.base.selection_rank),
        "selection_reason": condition.base.selection_reason,
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
        "policy_ids_json": _json_list(condition.base.policy_ids),
        "display_names_json": _json_list(condition.base.display_names),
        "source_categories_json": _json_list(condition.base.source_categories),
        "ratio_targets_json": _json_list(condition.base.ratio_targets),
        "host_policy_id": condition.base.host_policy_id,
        "graft_policy_id": condition.base.graft_policy_id,
        "host_display_name": condition.base.host_display_name,
        "graft_display_name": condition.base.graft_display_name,
        "host_source_category": condition.base.host_source_category,
        "graft_source_category": condition.base.graft_source_category,
        "goal_profile_id": goal_profile.profile_id,
        "goal_compatibility_class": goal_profile.compatibility_class,
        "goal_assignments_json": _json_object(goal_assignments),
        "value_profile": condition.base.value_profile,
        "perturbation_profile": condition.base.perturbation_profile,
        "frozen_indices_json": _json_list(sorted(frozen_indices)),
        "pre_graft_state_id": state_id,
        "graft_spec_id": condition.graft_spec.spec_id,
        "graft_description": condition.graft_spec.description,
        "graft_event": int(graft_event),
        "post_graft_event_count": int(len(post_schedule)),
        "graft_size": int(condition.graft_spec.graft_size),
        "graft_location": condition.graft_spec.graft_location,
        "graft_arrangement_reference": condition.graft_spec.arrangement_reference,
        "graft_mode": "in_place_relabel_patch_preserving_values",
        "graft_start_index": int(layout["graft_start_index"]),
        "graft_end_index": int(layout["graft_end_index"]),
        "graft_indices_json": _json_list(layout["graft_indices"]),
        "graft_cell_ids_json": _json_list(graft_cell_ids),
        "expected_counts_json": _json_object(expected_counts),
        "post_graft_counts_json": _json_object(dict(sorted(post_graft_counts.items()))),
        "final_counts_json": _json_object(dict(sorted(final_counts.items()))),
        "count_preservation_success": dict(final_counts) == expected_counts,
        "realized_host_fraction": float(final_counts.get(condition.base.host_policy_id, 0) / config.array_size),
        "realized_graft_fraction": float(final_counts.get(condition.base.graft_policy_id, 0) / config.array_size),
        "initial_inversion_count": int(initial_metrics["inversion_count"]),
        "initial_inversion_sortedness": float(initial_metrics["inversion_sortedness"]),
        "pre_graft_inversion_sortedness": float(pre_metrics["inversion_sortedness"]),
        "pre_graft_adjacent_sortedness": float(pre_metrics["adjacent_sortedness"]),
        "pre_graft_is_sorted": bool(pre_metrics["is_sorted"]),
        "final_inversion_count": int(final_metrics["inversion_count"]),
        "final_inversion_sortedness": float(final_metrics["inversion_sortedness"]),
        "final_adjacent_sortedness": float(final_metrics["adjacent_sortedness"]),
        "final_is_sorted": bool(final_metrics["is_sorted"]),
        "final_target_quality": float(goal_eval["assigned_policy_mean_sortedness"]),
        "reference_increasing_score": float(goal_eval["reference_increasing_score"]),
        "target_delta_vs_pre_graft": float(goal_eval["assigned_policy_mean_sortedness"] - pre_metrics["inversion_sortedness"]),
        "reference_delta_vs_pre_graft": float(goal_eval["reference_increasing_score"] - pre_metrics["inversion_sortedness"]),
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
        "graft_outcome_class": graft_outcome,
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
        "pre_graft_values_head_json": _json_list(pre_values[:20]),
        "pre_graft_values_tail_json": _json_list(pre_values[-20:]),
        "pre_graft_labels_head_json": _json_list(pre_labels[:20]),
        "pre_graft_labels_tail_json": _json_list(pre_labels[-20:]),
        "final_values_head_json": _json_list(values[:20]),
        "final_values_tail_json": _json_list(values[-20:]),
        "final_labels_head_json": _json_list(labels[:20]),
        "final_labels_tail_json": _json_list(labels[-20:]),
        "elapsed_seconds": float(elapsed),
        **pre_arrangement,
        **graft_metrics,
        **goal_eval,
    }
    return run_row, pre_state_row


def _simulate_graft_task(task: tuple[GraftCondition, Mapping[str, Mapping[str, Any]], int, S10Config]) -> tuple[dict[str, Any], dict[str, Any]]:
    condition, records_by_id, seed, config = task
    return simulate_graft_condition(condition, records_by_id, seed=int(seed), config=config)


def run_s10_sweep(
    records: Sequence[Mapping[str, Any]],
    mosaic: pd.DataFrame,
    exemplars: pd.DataFrame,
    s09_rankings: pd.DataFrame,
    config: S10Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or S10Config()
    selected = select_s10_base_contexts(mosaic, exemplars, config)
    observed_governance = set(s09_rankings["governance_mechanism_id"].astype(str)) if "governance_mechanism_id" in s09_rankings else set()
    expected_governance = {mechanism.mechanism_id for mechanism in config.governance_mechanisms}
    missing_governance = expected_governance - observed_governance
    if missing_governance:
        raise ValueError(f"S09 ranking table lacks required governance options: {sorted(missing_governance)}")
    conditions = build_s10_conditions(selected, config)
    records_by_id = {str(record["algotypeId"]): dict(record) for record in records}
    needed_policy_ids = {condition.base.host_policy_id for condition in conditions} | {condition.base.graft_policy_id for condition in conditions}
    missing = sorted(needed_policy_ids - set(records_by_id))
    if missing:
        raise ValueError(f"S10 selected policies missing from S01 library: {missing}")
    tasks = [(condition, records_by_id, int(seed), config) for condition in conditions for seed in config.seeds]
    if config.worker_count > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(config.worker_count)) as pool:
            outputs = list(pool.map(_simulate_graft_task, tasks))
    else:
        outputs = [_simulate_graft_task(task) for task in tasks]
    run_rows = [item[0] for item in outputs]
    state_rows = [item[1] for item in outputs]
    run_df = pd.DataFrame(run_rows)
    pre_state_df = pd.DataFrame(state_rows).drop_duplicates(subset=["pre_graft_state_id"]).reset_index(drop=True)
    condition_rows: list[dict[str, Any]] = []
    for condition in conditions:
        expected_counts = {
            condition.base.host_policy_id: int(config.array_size - condition.graft_spec.graft_size),
            condition.base.graft_policy_id: int(condition.graft_spec.graft_size),
        }
        condition_rows.append(
            {
                "condition_id": condition.condition_id,
                "base_context_id": condition.base.base_context_id,
                "source_s07_run_id": condition.base.source_s07_run_id,
                "selection_rank": int(condition.base.selection_rank),
                "selection_reason": condition.base.selection_reason,
                "s07_label": condition.base.s07_label,
                "graft_spec_id": condition.graft_spec.spec_id,
                "graft_event": int(min(condition.graft_spec.graft_event, config.event_cap)),
                "graft_size": int(condition.graft_spec.graft_size),
                "graft_location": condition.graft_spec.graft_location,
                "graft_arrangement_reference": condition.graft_spec.arrangement_reference,
                "interface_rule_id": condition.interface_rule.rule_id,
                "interface_access_stratum": interface_access_stratum(condition.interface_rule),
                "governance_mechanism_id": condition.governance.mechanism_id,
                "governance_access_stratum": governance_access_stratum(condition.governance),
                "intervention_stratum": intervention_stratum(condition.interface_rule, condition.governance),
                "global_controller_like": bool(condition.governance.global_controller_like),
                "policy_ids_json": _json_list(condition.base.policy_ids),
                "display_names_json": _json_list(condition.base.display_names),
                "host_policy_id": condition.base.host_policy_id,
                "graft_policy_id": condition.base.graft_policy_id,
                "host_display_name": condition.base.host_display_name,
                "graft_display_name": condition.base.graft_display_name,
                "goal_profile_id": condition.base.goal_profile_id,
                "expected_counts_json": _json_object(expected_counts),
            }
        )
    return run_df, pd.DataFrame(condition_rows), selected, pre_state_df


def summarize_s10_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["base_context_id", "graft_spec_id", "interface_rule_id", "governance_mechanism_id"]
    for _, group in run_df.groupby(group_columns, sort=False):
        first = group.iloc[0]
        outcome_counts = group["graft_outcome_class"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "base_context_id": first["base_context_id"],
                "graft_spec_id": first["graft_spec_id"],
                "interface_rule_id": first["interface_rule_id"],
                "interface_access_stratum": first["interface_access_stratum"],
                "governance_mechanism_id": first["governance_mechanism_id"],
                "governance_access_stratum": first["governance_access_stratum"],
                "intervention_stratum": first["intervention_stratum"],
                "global_controller_like": bool(first["global_controller_like"]),
                "s07_label": first["s07_label"],
                "host_display_name": first["host_display_name"],
                "graft_display_name": first["graft_display_name"],
                "policy_ids_json": first["policy_ids_json"],
                "display_names_json": first["display_names_json"],
                "source_categories_json": first["source_categories_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "graft_event": int(first["graft_event"]),
                "graft_size": int(first["graft_size"]),
                "graft_location": first["graft_location"],
                "seed_count": int(group["seed"].nunique()),
                "run_count": int(len(group)),
                "mean_final_target_quality": float(group["final_target_quality"].mean()),
                "mean_reference_increasing_score": float(group["reference_increasing_score"].mean()),
                "mean_target_delta_vs_pre_graft": float(group["target_delta_vs_pre_graft"].mean()),
                "mean_goal_conflict_index": float(group["goal_conflict_index"].mean()),
                "mean_aggregation_delta_percent": float(group["aggregation_delta_percent"].mean()),
                "mean_interface_count": float(group["interface_count"].mean()),
                "mean_graft_relative_largest_block_fraction": float(group["graft_relative_largest_block_fraction"].mean()),
                "mean_graft_edge_fraction": float(group["graft_edge_fraction"].mean()),
                "mean_graft_position_span": float(group["graft_position_span"].mean()),
                "mean_governance_block_fraction": float(group["governance_block_fraction"].mean()),
                "mean_governance_cost_per_event": float(group["governance_cost_per_event"].mean()),
                "mean_interface_block_fraction": float(group["interface_block_fraction"].mean()),
                "mean_work_count": float(group["work_count"].mean()),
                "invalid_action_count": int(group["invalid_action_count"].sum()),
                "count_preservation_success": bool(group["count_preservation_success"].all()),
                "graft_outcome_class_mode": _mode(group["graft_outcome_class"]),
                "graft_outcome_counts_json": _json_object(outcome_counts),
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
        "mean_target_delta_vs_pre_graft",
        "mean_goal_conflict_index",
        "mean_aggregation_delta_percent",
        "mean_interface_count",
        "mean_graft_relative_largest_block_fraction",
        "mean_graft_edge_fraction",
        "mean_graft_position_span",
        "mean_work_count",
    ]
    baseline = baseline[["base_context_id", "graft_spec_id", *metric_columns]].rename(
        columns={column: f"baseline_{column}" for column in metric_columns}
    )
    comparison = summary.merge(baseline, on=["base_context_id", "graft_spec_id"], how="left")
    for column in metric_columns:
        comparison[f"delta_{column.removeprefix('mean_')}"] = comparison[column] - comparison[f"baseline_{column}"]
    comparison["schema"] = COMPARISON_SCHEMA
    comparison["graft_rescue_score"] = (
        comparison["delta_final_target_quality"]
        - comparison["delta_goal_conflict_index"].clip(lower=0.0) * 0.20
        - comparison["mean_governance_block_fraction"] * 0.02
        - comparison["mean_interface_block_fraction"] * 0.01
        - comparison["mean_governance_cost_per_event"] * 0.001
    )
    comparison["target_quality_rescued"] = comparison["delta_final_target_quality"] >= 0.02
    comparison["graft_mixing_shift"] = -comparison["delta_graft_relative_largest_block_fraction"]
    return comparison


def graft_strata_summary(comparison: pd.DataFrame, config: S10Config) -> pd.DataFrame:
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
                "condition_count": int(group[["base_context_id", "graft_spec_id", "interface_rule_id", "governance_mechanism_id"]].drop_duplicates().shape[0]),
                "mean_delta_final_target_quality": float(group["delta_final_target_quality"].mean()),
                "mean_delta_goal_conflict_index": float(group["delta_goal_conflict_index"].mean()),
                "mean_delta_graft_relative_largest_block_fraction": float(group["delta_graft_relative_largest_block_fraction"].mean()),
                "mean_delta_graft_edge_fraction": float(group["delta_graft_edge_fraction"].mean()),
                "mean_graft_rescue_score": float(group["graft_rescue_score"].mean()),
                "target_quality_rescue_rate": float(group["target_quality_rescued"].mean()),
                "mean_governance_block_fraction": float(group["mean_governance_block_fraction"].mean()),
                "mean_interface_block_fraction": float(group["mean_interface_block_fraction"].mean()),
                "mean_governance_cost_per_event": float(group["mean_governance_cost_per_event"].mean()),
                "supportive_graft_shift_by_threshold": bool(group["graft_rescue_score"].mean() >= config.graft_rescue_delta_threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_graft_rescue_score", ascending=False, kind="mergesort")


def validate_s10_outputs(
    run_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    selected: pd.DataFrame,
    pre_state_df: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    strata: pd.DataFrame,
    config: S10Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    seed_set = set(int(seed) for seed in config.seeds)
    expected_interfaces = {rule.rule_id for rule in config.interface_rules}
    expected_governance = {mechanism.mechanism_id for mechanism in config.governance_mechanisms}
    expected_graft_specs = {spec.spec_id for spec in config.graft_specs}
    condition_seed_sets = (
        run_df.groupby(["base_context_id", "graft_spec_id", "interface_rule_id", "governance_mechanism_id"])["seed"].agg(lambda values: set(int(value) for value in values)).tolist()
        if not run_df.empty
        else []
    )
    base_graft_interface_sets = (
        run_df.groupby(["base_context_id", "graft_spec_id"])["interface_rule_id"].agg(lambda values: set(map(str, values))).tolist()
        if not run_df.empty
        else []
    )
    base_graft_interface_governance_sets = (
        run_df.groupby(["base_context_id", "graft_spec_id", "interface_rule_id"])["governance_mechanism_id"].agg(lambda values: set(map(str, values))).tolist()
        if not run_df.empty
        else []
    )
    state_ids = set(pre_state_df["pre_graft_state_id"].astype(str)) if not pre_state_df.empty else set()
    run_state_ids = set(run_df["pre_graft_state_id"].astype(str)) if not run_df.empty else set()
    finite_columns = ["delta_final_target_quality", "delta_goal_conflict_index", "graft_rescue_score", "mean_graft_relative_largest_block_fraction"]
    finite_comparison = bool(
        not comparison.empty
        and set(finite_columns).issubset(comparison.columns)
        and comparison[finite_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().all().all()
    )
    checks = [
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "selected_contexts_use_s07_exemplars_and_continua",
            "success": bool(
                not selected.empty
                and selected["selection_reason"].astype(str).str.contains("s07_exemplar").any()
                and selected["selection_reason"].astype(str).str.contains("metric_extreme").any()
            ),
            "observed": ",".join(selected["selection_reason"].astype(str).tolist()) if not selected.empty else "none",
            "expected": "bounded selection includes at least one S07 exemplar and metric-continuum context",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "graft_timing_size_location_logged",
            "success": bool(not run_df.empty and run_df[["graft_event", "graft_size", "graft_location", "graft_start_index", "graft_end_index"]].notna().all().all()),
            "observed": sorted(run_df["graft_spec_id"].astype(str).unique().tolist()) if not run_df.empty else [],
            "expected": sorted(expected_graft_specs),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "host_and_graft_identity_logged",
            "success": bool(not run_df.empty and run_df[["host_policy_id", "graft_policy_id", "host_display_name", "graft_display_name"]].notna().all().all()),
            "observed": f"contexts={run_df['base_context_id'].nunique() if not run_df.empty else 0}",
            "expected": "every run logs host and graft policy identity",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "host_pre_graft_state_saved",
            "success": bool(run_state_ids and run_state_ids.issubset(state_ids) and pre_state_df["pre_graft_values_json"].astype(str).str.startswith("[").all()),
            "observed": f"run_state_ids={len(run_state_ids)} saved_state_ids={len(state_ids)}",
            "expected": "every run references a saved full host pre-graft state",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "s03_arrangement_code_used_for_graft_placement",
            "success": bool(not run_df.empty and run_df["graft_arrangement_reference"].isin({"graft_like_insertions", "contiguous_patch"}).all() and (run_df["graft_indices_json"].astype(str) != "[]").all()),
            "observed": sorted(run_df["graft_arrangement_reference"].astype(str).unique().tolist()) if not run_df.empty else [],
            "expected": "graft patches are placed through S03 arrangement families",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "behavior_explicit_governance_global_strata_separated",
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
            "validation_case": "all_interface_rules_per_base_graft",
            "success": bool(base_graft_interface_sets and all(item == expected_interfaces for item in base_graft_interface_sets)),
            "observed": f"base_graft_groups={len(base_graft_interface_sets)}",
            "expected": sorted(expected_interfaces),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "all_governance_mechanisms_per_base_graft_interface",
            "success": bool(base_graft_interface_governance_sets and all(item == expected_governance for item in base_graft_interface_governance_sets)),
            "observed": f"base_graft_interface_groups={len(base_graft_interface_governance_sets)}",
            "expected": sorted(expected_governance),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "matched_seed_sets_by_condition",
            "success": bool(condition_seed_sets and all(item == seed_set for item in condition_seed_sets)),
            "observed": f"condition_groups={len(condition_seed_sets)} expected_seeds={sorted(seed_set)}",
            "expected": "each staged graft condition uses the configured matched seed set",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "post_graft_counts_preserved",
            "success": bool(not run_df.empty and run_df["count_preservation_success"].all()),
            "observed": f"count_failures={int((~run_df['count_preservation_success']).sum()) if not run_df.empty else 'missing'}",
            "expected": "host/graft counts remain fixed after graft insertion",
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
            "validation_case": "graft_outcome_categories_quantified",
            "success": bool(not summary.empty and summary["graft_outcome_counts_json"].astype(str).str.startswith("{").all()),
            "observed": sorted(run_df["graft_outcome_class"].astype(str).unique().tolist()) if not run_df.empty else [],
            "expected": "summary records per-condition graft outcome class counts",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "finite_baseline_comparisons_and_strata",
            "success": bool(finite_comparison and not strata.empty and strata["mean_graft_rescue_score"].notna().all()),
            "observed": f"comparison_rows={len(comparison)} strata_rows={len(strata)}",
            "expected": "all graft baseline deltas and stratum summaries are finite",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "figure_written",
            "success": bool(figure_written),
            "observed": str(figure_written),
            "expected": "graft_outcome_examples.png written and non-empty",
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
