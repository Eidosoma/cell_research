"""E06 S12 developmental-history experiments."""

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
from src.e06.graft_experiments import _run_host_development, governance_access_stratum, interface_access_stratum, intervention_stratum
from src.e06.interface_rules import InterfaceRule, frozen_indices_for_profile, goal_profile_for_id, initial_values_for_profile, interface_decision
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


STEP_ID = "S12"
HISTORY_SCHEMA = "eidosoma.e06.s12_developmental_history_run.v1"
HISTORY_SPEC_SCHEMA = "eidosoma.e06.s12_history_spec.v1"
SUMMARY_SCHEMA = "eidosoma.e06.s12_developmental_history_summary.v1"
COMPARISON_SCHEMA = "eidosoma.e06.s12_history_baseline_comparison.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s12_validation.v1"


@dataclass(frozen=True)
class HistorySpec:
    """One developmental-history encoding."""

    history_id: str
    history_family: str
    start_mode: str
    description: str
    introduction_event: int | None = None
    staged_arrangement_reference: str | None = None
    transient_start_event: int | None = None
    transient_end_event: int | None = None
    transient_profile: str | None = None
    prior_exposure_events: int = 0
    reset_after_exposure: bool = False
    preserves_memory_across_reset: bool = False


DEFAULT_HISTORY_SPECS = (
    HistorySpec(
        history_id="simultaneous_mixed_start",
        history_family="simultaneous",
        start_mode="matched_mixture_from_event0",
        description="All policies are mixed from event 0 using the target final composition and source arrangement.",
    ),
    HistorySpec(
        history_id="staged_early_patch_intro",
        history_family="staged_introduction",
        start_mode="host_predevelopment_then_patch",
        introduction_event=1_000,
        staged_arrangement_reference="graft_like_insertions",
        description="Host policy develops alone for 1,000 events, then the partner policy is introduced with S10/S03 graft-like placement at the matched final composition.",
    ),
    HistorySpec(
        history_id="staged_late_patch_intro",
        history_family="staged_introduction",
        start_mode="host_predevelopment_then_patch",
        introduction_event=2_400,
        staged_arrangement_reference="contiguous_patch",
        description="Host policy develops alone for 2,400 events, then the partner policy is introduced with S10/S03 contiguous-patch placement at the matched final composition.",
    ),
    HistorySpec(
        history_id="early_transient_center_freeze",
        history_family="transient_perturbation",
        start_mode="matched_mixture_from_event0",
        transient_start_event=600,
        transient_end_event=1_200,
        transient_profile="center_block_10",
        description="Matched mixture from event 0 with a transient 10-cell center frozen block during early sorting events 600-1,199.",
    ),
    HistorySpec(
        history_id="late_transient_center_freeze",
        history_family="transient_perturbation",
        start_mode="matched_mixture_from_event0",
        transient_start_event=2_400,
        transient_end_event=3_000,
        transient_profile="center_block_10",
        description="Matched mixture from event 0 with a transient 10-cell center frozen block during late sorting events 2,400-2,999.",
    ),
    HistorySpec(
        history_id="prior_chimeric_exposure_memory_reset",
        history_family="prior_exposure_memory",
        start_mode="prior_mixed_exposure_then_reset",
        prior_exposure_events=800,
        reset_after_exposure=True,
        preserves_memory_across_reset=True,
        description="Run an 800-event mixed exposure, reset values, labels, and cell positions to the matched initial composition, then continue while preserving cell memory state.",
    ),
)


@dataclass(frozen=True)
class S12Config:
    """Configuration for bounded S12 developmental-history tests."""

    array_size: int = 100
    event_cap: int = 4_000
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    max_s10_contexts: int = 3
    max_memory_contexts: int = 2
    history_specs: tuple[HistorySpec, ...] = DEFAULT_HISTORY_SPECS
    interface_rules: tuple[InterfaceRule, ...] = DEFAULT_INTERFACE_RULES
    governance_mechanisms: tuple[GovernanceMechanism, ...] = DEFAULT_GOVERNANCE_MECHANISMS
    scheduler: str = "cyclic_scan_seed_offset"
    worker_count: int = 1
    history_effect_threshold: float = 0.025


@dataclass(frozen=True)
class HistoryBaseContext:
    """One base pair/context for developmental-history comparisons."""

    base_context_id: str
    source_context_family: str
    source_s10_base_context_id: str
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
    partner_policy_id: str
    host_display_name: str
    partner_display_name: str
    host_source_category: str
    partner_source_category: str
    value_profile: str
    perturbation_profile: str
    source_final_target_quality: float
    source_goal_conflict_index: float
    source_aggregation_delta_percent: float
    source_largest_block_fraction: float
    source_mosaic_continuum_score: float
    source_conflict_continuum_score: float
    memory_policy_present: bool
    memory_policy_ids: tuple[str, ...]
    e04_memory_input_reason: str
    source_s10_graft_sensitivity_score: float
    source_s11_mutant_takeover_rate: float


@dataclass(frozen=True)
class HistoryCondition:
    """One executable S12 condition before seed expansion."""

    condition_id: str
    base: HistoryBaseContext
    history: HistorySpec
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
    mode = values.astype(str).mode()
    return str(mode.iloc[0]) if not mode.empty else ""


def _normalize_arrangement(arrangement: Any) -> str:
    text = str(arrangement or "").strip()
    if text == "random_permutation_labels":
        return "random_permutation"
    if text in {"random_permutation", "contiguous_patch", "alternating", "clustered_islands", "gradient", "graft_like_insertions"}:
        return text
    return "random_permutation"


def history_spec_table(history_specs: Sequence[HistorySpec] = DEFAULT_HISTORY_SPECS) -> pd.DataFrame:
    return pd.DataFrame([{**spec.__dict__, "schema": HISTORY_SPEC_SCHEMA, "research_step_id": STEP_ID} for spec in history_specs])


def _condition_id(base_context_id: str, history_id: str, interface_rule_id: str, governance_id: str) -> str:
    payload = {
        "baseContextId": base_context_id,
        "historyId": history_id,
        "interfaceRuleId": interface_rule_id,
        "governanceMechanismId": governance_id,
    }
    return f"s12_{stable_hash(payload)[:16]}"


def _base_context_id(payload: Mapping[str, Any]) -> str:
    return f"s12_base_{stable_hash(payload)[:16]}"


def _metadata_row(metadata: pd.DataFrame, display_name: str) -> dict[str, Any] | None:
    rows = metadata[metadata["display_name"].astype(str) == display_name]
    if rows.empty:
        return None
    return dict(rows.iloc[0])


def _metadata_by_id(metadata: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["algotype_id"]): dict(row) for row in metadata.to_dict(orient="records")}


def _source_categories_have_memory(categories: Sequence[str]) -> bool:
    return any(str(category) == "memory_repair" for category in categories)


def _memory_ids(policy_ids: Sequence[str], metadata_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    ids: list[str] = []
    for policy_id in policy_ids:
        if str(metadata_by_id.get(str(policy_id), {}).get("source_category", "")) == "memory_repair":
            ids.append(str(policy_id))
    return tuple(ids)


def select_s12_base_contexts(
    s10_selected: pd.DataFrame,
    s11_runs: pd.DataFrame,
    metadata: pd.DataFrame,
    records: Sequence[Mapping[str, Any]],
    config: S12Config,
) -> pd.DataFrame:
    """Select S10 staged contexts plus E04 memory-policy contexts."""

    required_s10 = {
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
    missing = required_s10 - set(s10_selected.columns)
    if missing:
        raise ValueError(f"S10 selected-context table missing S12 columns: {sorted(missing)}")
    metadata_by_id = _metadata_by_id(metadata)
    record_ids = {str(record["algotypeId"]) for record in records}

    s11_takeover = pd.DataFrame()
    if not s11_runs.empty and {"base_context_id", "takeover_success"}.issubset(s11_runs.columns):
        s11_takeover = (
            s11_runs.groupby("base_context_id", sort=False)
            .agg(source_s11_mutant_takeover_rate=("takeover_success", "mean"))
            .reset_index()
        )

    rows: list[dict[str, Any]] = []
    selected_s10 = s10_selected.head(config.max_s10_contexts).copy()
    if not s11_takeover.empty:
        selected_s10 = selected_s10.merge(s11_takeover, on="base_context_id", how="left")
    else:
        selected_s10["source_s11_mutant_takeover_rate"] = 0.0
    for raw in selected_s10.to_dict(orient="records"):
        policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
        display_names = tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
        categories = tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
        ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
        source_memory_ids = _memory_ids(policy_ids, metadata_by_id)
        rows.append(
            {
                "base_context_id": f"s12_from_{raw['base_context_id']}",
                "source_context_family": "s10_graft_sensitive",
                "source_s10_base_context_id": str(raw["base_context_id"]),
                "source_s07_run_id": str(raw["source_s07_run_id"]),
                "source_research_step_id": str(raw["source_research_step_id"]),
                "source_condition_id": str(raw["source_condition_id"]),
                "selection_rank": int(len(rows) + 1),
                "selection_reason": str(raw["selection_reason"]) + ",s12_s10_staged_context",
                "s07_label": str(raw["s07_label"]),
                "s07_classification_mode": str(raw["s07_classification_mode"]),
                "panel": str(raw["panel"]),
                "candidate_reason": str(raw["candidate_reason"]),
                "source_arrangement": _normalize_arrangement(raw["source_arrangement"]),
                "goal_profile_id": str(raw["goal_profile_id"]),
                "goal_compatibility_class": str(raw["goal_compatibility_class"]),
                "policy_ids_json": _json_list(policy_ids),
                "display_names_json": _json_list(display_names),
                "source_categories_json": _json_list(categories),
                "ratio_targets_json": _json_list(ratios),
                "host_policy_id": str(raw["host_policy_id"]),
                "partner_policy_id": str(raw["graft_policy_id"]),
                "host_display_name": str(raw["host_display_name"]),
                "partner_display_name": str(raw["graft_display_name"]),
                "host_source_category": str(raw["host_source_category"]),
                "partner_source_category": str(raw["graft_source_category"]),
                "value_profile": str(raw["value_profile"]),
                "perturbation_profile": str(raw["perturbation_profile"]),
                "source_final_target_quality": float(raw["source_final_target_quality"]),
                "source_goal_conflict_index": float(raw["source_goal_conflict_index"]),
                "source_aggregation_delta_percent": float(raw["source_aggregation_delta_percent"]),
                "source_largest_block_fraction": float(raw["source_largest_block_fraction"]),
                "source_mosaic_continuum_score": float(raw["source_mosaic_continuum_score"]),
                "source_conflict_continuum_score": float(raw["source_conflict_continuum_score"]),
                "memory_policy_present": bool(source_memory_ids),
                "memory_policy_ids_json": _json_list(source_memory_ids),
                "e04_memory_input_reason": "no_memory_policy_in_s10_context",
                "source_s10_graft_sensitivity_score": _safe_float(raw.get("s10_graft_sensitivity_score"), 0.0),
                "source_s11_mutant_takeover_rate": _safe_float(raw.get("source_s11_mutant_takeover_rate"), 0.0),
            }
        )

    memory_meta = metadata[(metadata["source_category"].astype(str) == "memory_repair") & (metadata["algotype_id"].astype(str).isin(record_ids))]
    if memory_meta.empty:
        raise ValueError("S12 requires at least one E04 memory_repair policy in the S01 library")
    anchor_specs = [
        ("original_bubble", (0.50, 0.50), "same_increasing", "memory_vs_original", "random_permutation"),
        ("e03_frontier_rank_01", (0.75, 0.25), "same_increasing", "memory_vs_frontier", "graft_like_insertions"),
    ]
    memory_rows = memory_meta.sort_values("display_name", kind="mergesort").head(config.max_memory_contexts).to_dict(orient="records")
    for idx, mem in enumerate(memory_rows):
        anchor_name, ratios, goal_profile_id, panel, arrangement = anchor_specs[min(idx, len(anchor_specs) - 1)]
        anchor = _metadata_row(metadata, anchor_name) or _metadata_row(metadata, "original_bubble")
        if anchor is None:
            raise ValueError("S12 could not find an anchor policy for memory context construction")
        policy_ids = (str(anchor["algotype_id"]), str(mem["algotype_id"]))
        if not set(policy_ids).issubset(record_ids):
            continue
        display_names = (str(anchor["display_name"]), str(mem["display_name"]))
        categories = (str(anchor["source_category"]), str(mem["source_category"]))
        payload = {
            "source": "s12_memory_policy_input",
            "policyIds": policy_ids,
            "ratios": ratios,
            "goalProfileId": goal_profile_id,
            "arrangement": arrangement,
        }
        rows.append(
            {
                "base_context_id": _base_context_id(payload),
                "source_context_family": "e04_memory_policy_input",
                "source_s10_base_context_id": "",
                "source_s07_run_id": "",
                "source_research_step_id": "S01/E04",
                "source_condition_id": f"s12_memory_input_{idx + 1:02d}",
                "selection_rank": int(len(rows) + 1),
                "selection_reason": f"e04_memory_policy_input:{display_names[1]}",
                "s07_label": "memory_policy_input",
                "s07_classification_mode": "not_from_s07",
                "panel": panel,
                "candidate_reason": "include_e04_memory_policy_input",
                "source_arrangement": arrangement,
                "goal_profile_id": goal_profile_id,
                "goal_compatibility_class": "same_goal",
                "policy_ids_json": _json_list(policy_ids),
                "display_names_json": _json_list(display_names),
                "source_categories_json": _json_list(categories),
                "ratio_targets_json": _json_list(ratios),
                "host_policy_id": policy_ids[0],
                "partner_policy_id": policy_ids[1],
                "host_display_name": display_names[0],
                "partner_display_name": display_names[1],
                "host_source_category": categories[0],
                "partner_source_category": categories[1],
                "value_profile": "random_permutation",
                "perturbation_profile": "none",
                "source_final_target_quality": math.nan,
                "source_goal_conflict_index": math.nan,
                "source_aggregation_delta_percent": math.nan,
                "source_largest_block_fraction": math.nan,
                "source_mosaic_continuum_score": math.nan,
                "source_conflict_continuum_score": math.nan,
                "memory_policy_present": True,
                "memory_policy_ids_json": _json_list((policy_ids[1],)),
                "e04_memory_input_reason": "explicit_s12_memory_context",
                "source_s10_graft_sensitivity_score": math.nan,
                "source_s11_mutant_takeover_rate": math.nan,
            }
        )

    selected = pd.DataFrame(rows)
    if selected.empty:
        raise ValueError("S12 could not select developmental-history contexts")
    return selected.reset_index(drop=True)


def _base_from_row(raw: Mapping[str, Any]) -> HistoryBaseContext:
    policy_ids = tuple(str(item) for item in _parse_json_list(raw["policy_ids_json"]))
    display_names = tuple(str(item) for item in _parse_json_list(raw["display_names_json"]))
    categories = tuple(str(item) for item in _parse_json_list(raw["source_categories_json"]))
    ratios = tuple(float(item) for item in _parse_json_list(raw["ratio_targets_json"]))
    memory_ids = tuple(str(item) for item in _parse_json_list(raw["memory_policy_ids_json"]))
    return HistoryBaseContext(
        base_context_id=str(raw["base_context_id"]),
        source_context_family=str(raw["source_context_family"]),
        source_s10_base_context_id=str(raw.get("source_s10_base_context_id", "")),
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
        partner_policy_id=str(raw["partner_policy_id"]),
        host_display_name=str(raw["host_display_name"]),
        partner_display_name=str(raw["partner_display_name"]),
        host_source_category=str(raw["host_source_category"]),
        partner_source_category=str(raw["partner_source_category"]),
        value_profile=str(raw["value_profile"]),
        perturbation_profile=str(raw["perturbation_profile"]),
        source_final_target_quality=_safe_float(raw.get("source_final_target_quality"), math.nan),
        source_goal_conflict_index=_safe_float(raw.get("source_goal_conflict_index"), math.nan),
        source_aggregation_delta_percent=_safe_float(raw.get("source_aggregation_delta_percent"), math.nan),
        source_largest_block_fraction=_safe_float(raw.get("source_largest_block_fraction"), math.nan),
        source_mosaic_continuum_score=_safe_float(raw.get("source_mosaic_continuum_score"), math.nan),
        source_conflict_continuum_score=_safe_float(raw.get("source_conflict_continuum_score"), math.nan),
        memory_policy_present=bool(raw["memory_policy_present"]),
        memory_policy_ids=memory_ids,
        e04_memory_input_reason=str(raw["e04_memory_input_reason"]),
        source_s10_graft_sensitivity_score=_safe_float(raw.get("source_s10_graft_sensitivity_score"), math.nan),
        source_s11_mutant_takeover_rate=_safe_float(raw.get("source_s11_mutant_takeover_rate"), math.nan),
    )


def build_s12_conditions(selected: pd.DataFrame, config: S12Config) -> list[HistoryCondition]:
    conditions: list[HistoryCondition] = []
    for raw in selected.to_dict(orient="records"):
        base = _base_from_row(raw)
        for history in config.history_specs:
            for interface_rule in config.interface_rules:
                for governance in config.governance_mechanisms:
                    conditions.append(
                        HistoryCondition(
                            condition_id=_condition_id(base.base_context_id, history.history_id, interface_rule.rule_id, governance.mechanism_id),
                            base=base,
                            history=history,
                            interface_rule=interface_rule,
                            governance=governance,
                        )
                    )
    return conditions


def _memory_digest(memory: Mapping[int, Mapping[str, Any]]) -> str:
    payload = {str(key): dict(value) for key, value in sorted(memory.items())}
    return stable_hash(payload)


def _memory_stats(memory: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    if not memory:
        return {
            "memory_state_count": 0,
            "memory_success_count": 0,
            "memory_failure_count": 0,
            "memory_total_frustration": 0,
            "memory_mean_time_since_movement": 0.0,
            "memory_failed_swap_count": 0,
            "memory_neighbor_identity_count": 0,
        }
    states = list(memory.values())
    return {
        "memory_state_count": int(len(states)),
        "memory_success_count": int(sum(1 for state in states if state.get("last_move_success") is True)),
        "memory_failure_count": int(sum(1 for state in states if state.get("last_move_success") is False)),
        "memory_total_frustration": int(sum(int(state.get("local_frustration", 0)) for state in states)),
        "memory_mean_time_since_movement": float(np.mean([int(state.get("time_since_movement", 0)) for state in states])),
        "memory_failed_swap_count": int(sum(int(state.get("failed_swap_count", 0)) for state in states)),
        "memory_neighbor_identity_count": int(sum(int(state.get("neighbor_identity_count", 0)) for state in states)),
    }


def _update_success(memory: dict[int, dict[str, Any]], cell_id: int, action_type: str = "swap") -> None:
    state = memory.setdefault(int(cell_id), {})
    state["last_move_success"] = True
    state["last_action_type"] = action_type
    state["time_since_movement"] = 0
    state["local_frustration"] = max(0, int(state.get("local_frustration", 0)) - 1)


def _update_failure(memory: dict[int, dict[str, Any]], cell_id: int, action_type: str) -> None:
    state = memory.setdefault(int(cell_id), {})
    state["last_move_success"] = False
    state["last_action_type"] = action_type
    state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)
    state["local_frustration"] = min(255, int(state.get("local_frustration", 0)) + 1)
    state["failed_swap_count"] = min(255, int(state.get("failed_swap_count", 0)) + 1)


def _update_wait(memory: dict[int, dict[str, Any]], cell_id: int, action_type: str = "wait") -> None:
    state = memory.setdefault(int(cell_id), {})
    state["last_move_success"] = None
    state["last_action_type"] = action_type
    state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)


def _transient_indices(profile: str | None, array_size: int) -> frozenset[int]:
    if profile is None:
        return frozenset()
    if profile == "center_block_10":
        width = min(10, array_size)
        start = max(0, (array_size - width) // 2)
        return frozenset(range(start, start + width))
    if profile == "edge_pairs":
        return frozenset({0, 1, array_size - 2, array_size - 1})
    return frozenset()


def _counts_and_expected(base: HistoryBaseContext, array_size: int) -> tuple[tuple[int, int], dict[str, int]]:
    counts = counts_for_ratios(base.ratio_targets, array_size)
    expected = {str(policy_id): int(count) for policy_id, count in zip(base.policy_ids, counts, strict=True)}
    return (int(counts[0]), int(counts[1])), expected


def _target_labels(base: HistoryBaseContext, counts: Sequence[int], seed: int, condition_id: str, arrangement: str | None = None) -> tuple[str, ...]:
    return labels_for_arrangement(
        base.policy_ids,
        counts,
        seed=seed,
        condition_id=condition_id,
        arrangement=_normalize_arrangement(arrangement or base.source_arrangement),
    )


def _apply_staged_layout(
    labels: list[str],
    cell_ids: list[int],
    *,
    base: HistoryBaseContext,
    counts: Sequence[int],
    history: HistorySpec,
    seed: int,
    condition_id: str,
) -> dict[str, Any]:
    arrangement = history.staged_arrangement_reference or "graft_like_insertions"
    layout = _target_labels(base, counts, seed=seed, condition_id=condition_id, arrangement=arrangement)
    partner_indices = [idx for idx, label in enumerate(layout) if str(label) == base.partner_policy_id]
    expected_partner_count = int(dict(zip(base.policy_ids, counts, strict=True)).get(base.partner_policy_id, 0))
    if len(partner_indices) != expected_partner_count:
        raise RuntimeError("S12 staged layout produced the wrong partner count")
    for idx, label in enumerate(layout):
        labels[idx] = str(label)
    base_new_id = len(cell_ids)
    for offset, idx in enumerate(partner_indices):
        cell_ids[idx] = base_new_id + offset
    return {
        "staged_partner_indices": tuple(partner_indices),
        "staged_partner_start_index": int(min(partner_indices)) if partner_indices else -1,
        "staged_partner_end_index": int(max(partner_indices)) if partner_indices else -1,
        "staged_partner_cell_ids": tuple(cell_ids[idx] for idx in partner_indices),
        "staged_layout_labels": tuple(layout),
    }


def _history_encoding(history: HistorySpec) -> str:
    return _json_object(history.__dict__)


def _run_intervention_loop(
    *,
    values: list[int],
    labels: list[str],
    cell_ids: list[int],
    memory: dict[int, dict[str, Any]],
    runtime: MixtureRuntime,
    schedule: Sequence[int],
    absolute_start_event: int,
    goal_assignments: Mapping[str, str],
    base_frozen_indices: frozenset[int],
    condition: HistoryCondition,
    config: S12Config,
    rng: random.Random,
    phase_label: str,
) -> dict[str, Any]:
    counts = Counter()
    same_contact_delta_sum = 0.0
    local_goal_delta_sum = 0.0
    governance_local_goal_delta_sum = 0.0
    governance_heterotypic_edge_delta_sum = 0.0
    governance_block_reasons: Counter[str] = Counter()
    interface_block_reasons: Counter[str] = Counter()
    transient_indices = _transient_indices(condition.history.transient_profile, config.array_size)
    transient_start = condition.history.transient_start_event
    transient_end = condition.history.transient_end_event
    for offset, actor_index in enumerate(schedule):
        event_index = int(absolute_start_event + offset)
        actor_index = int(actor_index)
        active_transient = bool(
            transient_indices
            and transient_start is not None
            and transient_end is not None
            and int(transient_start) <= event_index < int(transient_end)
        )
        active_frozen = base_frozen_indices | transient_indices if active_transient else base_frozen_indices
        counts[f"{phase_label}_transient_active_event_count"] += int(active_transient)
        policy_id = str(labels[actor_index])
        cell_id = int(cell_ids[actor_index])
        actor_goal = goal_assignments[policy_id]
        proxy_values = rank_values(values, actor_goal, config.array_size)
        action = runtime.action_for(policy_id, proxy_values, labels, actor_index, cell_id, memory, rng)
        counts["compare_count"] += int(bool(action.compare_counted))
        target = None if action.target_index is None else int(action.target_index)
        if target is not None and (actor_index in active_frozen or target in active_frozen):
            counts["frozen_block_count"] += 1
            counts["transient_frozen_block_count"] += int(active_transient and (actor_index in transient_indices or target in transient_indices))
            counts["wait_count"] += 1
            _update_failure(memory, cell_id, f"{phase_label}_frozen_block")
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
                    policy_ids=condition.base.policy_ids,
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
                    _update_failure(memory, cell_id, f"governance_block:{reason}")
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
                    _update_failure(memory, cell_id, f"interface_block:{reason}")
                    continue
                cross_swap = str(labels[actor_index]) != str(labels[target])
                values[actor_index], values[target] = values[target], values[actor_index]
                labels[actor_index], labels[target] = labels[target], labels[actor_index]
                cell_ids[actor_index], cell_ids[target] = cell_ids[target], cell_ids[actor_index]
                counts["swap_count"] += 1
                counts["cross_label_swap_count"] += int(cross_swap)
                counts["same_label_swap_count"] += int(not cross_swap)
                _update_success(memory, cell_id)
            else:
                counts["invalid_action_count"] += 1
                counts["wait_count"] += 1
                _update_failure(memory, cell_id, f"{phase_label}_invalid_swap")
        elif action.action_type == "update_state":
            counts["update_count"] += 1
            state = memory.setdefault(cell_id, {})
            if "ideal_position" in action.state_update:
                updated = action.state_update["ideal_position"]
                state["ideal_position"] = None if updated is None else int(updated)
            state["last_action_type"] = "update_state"
        else:
            counts["wait_count"] += 1
            _update_wait(memory, cell_id, f"{phase_label}_wait")
    return {
        **{str(key): int(value) for key, value in counts.items()},
        "same_contact_delta_sum": float(same_contact_delta_sum),
        "local_goal_delta_sum": float(local_goal_delta_sum),
        "governance_local_goal_delta_sum": float(governance_local_goal_delta_sum),
        "governance_heterotypic_edge_delta_sum": float(governance_heterotypic_edge_delta_sum),
        "governance_block_reasons_json": _json_object(governance_block_reasons),
        "interface_block_reasons_json": _json_object(interface_block_reasons),
    }


def _sum_counts(*items: Mapping[str, Any]) -> Counter:
    total: Counter[str] = Counter()
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, np.integer)):
                total[str(key)] += int(value)
    return total


def simulate_history_condition(
    condition: HistoryCondition,
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: S12Config,
) -> dict[str, Any]:
    started = time.perf_counter()
    counts, expected_counts = _counts_and_expected(condition.base, config.array_size)
    initial_values = list(initial_values_for_profile(config.array_size, seed, condition.base.value_profile))
    target_initial_labels = list(_target_labels(condition.base, counts, seed=seed, condition_id=condition.condition_id))
    values = list(initial_values)
    labels = list(target_initial_labels)
    cell_ids = list(range(config.array_size))
    memory: dict[int, dict[str, Any]] = {}
    runtime = MixtureRuntime([records_by_id[policy_id] for policy_id in condition.base.policy_ids], seed)
    full_schedule = actor_schedule(config.array_size, config.event_cap, seed)
    goal_profile = goal_profile_for_id(condition.base.goal_profile_id)
    goal_assignments = goal_assignments_for_profile(condition.base.policy_ids, goal_profile)
    base_frozen_indices = frozen_indices_for_profile(config.array_size, condition.base.perturbation_profile)
    rng = random.Random(seed ^ int(stable_hash({"s12": condition.condition_id, "history": condition.history.history_id})[:12], 16))
    initial_target_arrangement = arrangement_metric_row(target_initial_labels, condition.base.policy_ids)
    initial_target_counts = Counter(target_initial_labels)
    initial_metrics = sortedness_metrics(values)

    pre_counts: dict[str, Any] = {}
    exposure_counts: dict[str, Any] = {}
    main_counts: dict[str, Any] = {}
    stage_layout = {
        "staged_partner_indices": tuple(),
        "staged_partner_start_index": -1,
        "staged_partner_end_index": -1,
        "staged_partner_cell_ids": tuple(),
    }
    pre_history_metrics = sortedness_metrics(values)
    pre_history_counts = Counter(labels)
    exposure_memory_digest = ""
    memory_digest_before_main = _memory_digest(memory)

    if condition.history.history_family == "staged_introduction":
        labels = [condition.base.host_policy_id] * config.array_size
        cell_ids = list(range(config.array_size))
        intro_event = min(max(0, int(condition.history.introduction_event or 0)), config.event_cap)
        pre_schedule = full_schedule[:intro_event]
        post_schedule = full_schedule[intro_event:]
        pre_rng = random.Random(seed ^ int(stable_hash({"s12PreStage": condition.condition_id})[:12], 16))
        pre_counts = _run_host_development(
            values=values,
            labels=labels,
            cell_ids=cell_ids,
            memory=memory,
            runtime=runtime,
            schedule=pre_schedule,
            goal_assignments={condition.base.host_policy_id: goal_assignments[condition.base.host_policy_id]},
            frozen_indices=base_frozen_indices,
            array_size=config.array_size,
            rng=pre_rng,
        )
        pre_history_metrics = sortedness_metrics(values)
        pre_history_counts = Counter(labels)
        stage_layout = _apply_staged_layout(
            labels,
            cell_ids,
            base=condition.base,
            counts=counts,
            history=condition.history,
            seed=seed,
            condition_id=condition.condition_id,
        )
        memory_digest_before_main = _memory_digest(memory)
        main_counts = _run_intervention_loop(
            values=values,
            labels=labels,
            cell_ids=cell_ids,
            memory=memory,
            runtime=runtime,
            schedule=post_schedule,
            absolute_start_event=intro_event,
            goal_assignments=goal_assignments,
            base_frozen_indices=base_frozen_indices,
            condition=condition,
            config=config,
            rng=rng,
            phase_label="post_stage",
        )
    elif condition.history.history_family == "prior_exposure_memory":
        exposure_schedule = full_schedule[: int(condition.history.prior_exposure_events)]
        reset_values = list(values)
        reset_labels = list(labels)
        reset_cell_ids = list(cell_ids)
        exposure_counts = _run_intervention_loop(
            values=values,
            labels=labels,
            cell_ids=cell_ids,
            memory=memory,
            runtime=runtime,
            schedule=exposure_schedule,
            absolute_start_event=0,
            goal_assignments=goal_assignments,
            base_frozen_indices=base_frozen_indices,
            condition=condition,
            config=config,
            rng=rng,
            phase_label="prior_exposure",
        )
        exposure_memory_digest = _memory_digest(memory)
        if condition.history.reset_after_exposure:
            values = list(reset_values)
            labels = list(reset_labels)
            cell_ids = list(reset_cell_ids)
        memory_digest_before_main = _memory_digest(memory)
        main_counts = _run_intervention_loop(
            values=values,
            labels=labels,
            cell_ids=cell_ids,
            memory=memory,
            runtime=runtime,
            schedule=full_schedule,
            absolute_start_event=0,
            goal_assignments=goal_assignments,
            base_frozen_indices=base_frozen_indices,
            condition=condition,
            config=config,
            rng=rng,
            phase_label="main",
        )
    else:
        main_counts = _run_intervention_loop(
            values=values,
            labels=labels,
            cell_ids=cell_ids,
            memory=memory,
            runtime=runtime,
            schedule=full_schedule,
            absolute_start_event=0,
            goal_assignments=goal_assignments,
            base_frozen_indices=base_frozen_indices,
            condition=condition,
            config=config,
            rng=rng,
            phase_label="main",
        )

    elapsed = time.perf_counter() - started
    combined_counts = _sum_counts(pre_counts, exposure_counts, main_counts)
    final_metrics = sortedness_metrics(values)
    final_agg = aggregation_metrics(labels)
    final_counts = Counter(labels)
    dominance = dominance_metrics(labels, condition.base.policy_ids)
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
    final_composition_matched = dict(final_counts) == expected_counts
    memory_stats = _memory_stats(memory)
    transient_indices = _transient_indices(condition.history.transient_profile, config.array_size)
    memory_policy_ids = tuple(policy_id for policy_id in condition.base.memory_policy_ids)
    total_events_executed = int(config.event_cap + (condition.history.prior_exposure_events if condition.history.history_family == "prior_exposure_memory" else 0))
    run_row = {
        "schema": HISTORY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "condition_id": condition.condition_id,
        "base_context_id": condition.base.base_context_id,
        "source_context_family": condition.base.source_context_family,
        "source_s10_base_context_id": condition.base.source_s10_base_context_id,
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
        "source_s10_graft_sensitivity_score": float(condition.base.source_s10_graft_sensitivity_score),
        "source_s11_mutant_takeover_rate": float(condition.base.source_s11_mutant_takeover_rate),
        "history_id": condition.history.history_id,
        "history_family": condition.history.history_family,
        "history_start_mode": condition.history.start_mode,
        "history_description": condition.history.description,
        "history_encoding_json": _history_encoding(condition.history),
        "introduction_event": -1 if condition.history.introduction_event is None else int(condition.history.introduction_event),
        "staged_arrangement_reference": str(condition.history.staged_arrangement_reference or ""),
        "transient_start_event": -1 if condition.history.transient_start_event is None else int(condition.history.transient_start_event),
        "transient_end_event": -1 if condition.history.transient_end_event is None else int(condition.history.transient_end_event),
        "transient_profile": str(condition.history.transient_profile or ""),
        "transient_indices_json": _json_list(sorted(transient_indices)),
        "prior_exposure_events": int(condition.history.prior_exposure_events),
        "reset_after_exposure": bool(condition.history.reset_after_exposure),
        "preserves_memory_across_reset": bool(condition.history.preserves_memory_across_reset),
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
        "events_executed": total_events_executed,
        "seed": int(seed),
        "scheduler": config.scheduler,
        "policy_ids_json": _json_list(condition.base.policy_ids),
        "display_names_json": _json_list(condition.base.display_names),
        "source_categories_json": _json_list(condition.base.source_categories),
        "ratio_targets_json": _json_list(condition.base.ratio_targets),
        "host_policy_id": condition.base.host_policy_id,
        "partner_policy_id": condition.base.partner_policy_id,
        "host_display_name": condition.base.host_display_name,
        "partner_display_name": condition.base.partner_display_name,
        "host_source_category": condition.base.host_source_category,
        "partner_source_category": condition.base.partner_source_category,
        "memory_policy_present": bool(condition.base.memory_policy_present),
        "memory_policy_ids_json": _json_list(memory_policy_ids),
        "e04_memory_input_reason": condition.base.e04_memory_input_reason,
        "goal_profile_id": goal_profile.profile_id,
        "goal_compatibility_class": goal_profile.compatibility_class,
        "goal_assignments_json": _json_object(goal_assignments),
        "value_profile": condition.base.value_profile,
        "perturbation_profile": condition.base.perturbation_profile,
        "persistent_frozen_indices_json": _json_list(sorted(base_frozen_indices)),
        "expected_counts_json": _json_object(expected_counts),
        "target_initial_counts_json": _json_object(dict(sorted(initial_target_counts.items()))),
        "pre_history_counts_json": _json_object(dict(sorted(pre_history_counts.items()))),
        "final_counts_json": _json_object(dict(sorted(final_counts.items()))),
        "matched_final_composition_possible": True,
        "final_composition_matched": bool(final_composition_matched),
        "initial_inversion_count": int(initial_metrics["inversion_count"]),
        "initial_inversion_sortedness": float(initial_metrics["inversion_sortedness"]),
        "pre_history_inversion_sortedness": float(pre_history_metrics["inversion_sortedness"]),
        "final_inversion_count": int(final_metrics["inversion_count"]),
        "final_inversion_sortedness": float(final_metrics["inversion_sortedness"]),
        "final_adjacent_sortedness": float(final_metrics["adjacent_sortedness"]),
        "final_is_sorted": bool(final_metrics["is_sorted"]),
        "final_target_quality": float(goal_eval["assigned_policy_mean_sortedness"]),
        "reference_increasing_score": float(goal_eval["reference_increasing_score"]),
        "target_delta_vs_initial": float(goal_eval["assigned_policy_mean_sortedness"] - initial_metrics["inversion_sortedness"]),
        "target_delta_vs_pre_history": float(goal_eval["assigned_policy_mean_sortedness"] - pre_history_metrics["inversion_sortedness"]),
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
        "staged_partner_start_index": int(stage_layout["staged_partner_start_index"]),
        "staged_partner_end_index": int(stage_layout["staged_partner_end_index"]),
        "staged_partner_indices_json": _json_list(stage_layout["staged_partner_indices"]),
        "staged_partner_cell_ids_json": _json_list(stage_layout["staged_partner_cell_ids"]),
        "exposure_memory_digest": exposure_memory_digest,
        "memory_digest_before_main": memory_digest_before_main,
        "memory_digest_final": _memory_digest(memory),
        **memory_stats,
        "pre_compare_count": int(pre_counts.get("compare_count", 0)),
        "pre_swap_count": int(pre_counts.get("swap_count", 0)),
        "pre_update_count": int(pre_counts.get("update_count", 0)),
        "pre_wait_count": int(pre_counts.get("wait_count", 0)),
        "pre_invalid_action_count": int(pre_counts.get("invalid_action_count", 0)),
        "exposure_compare_count": int(exposure_counts.get("compare_count", 0)),
        "exposure_swap_count": int(exposure_counts.get("swap_count", 0)),
        "exposure_wait_count": int(exposure_counts.get("wait_count", 0)),
        "post_compare_count": int(main_counts.get("compare_count", 0)),
        "post_swap_count": int(main_counts.get("swap_count", 0)),
        "same_label_swap_count": int(combined_counts.get("same_label_swap_count", 0)),
        "cross_label_swap_count": int(combined_counts.get("cross_label_swap_count", 0)),
        "raw_swap_proposal_count": int(combined_counts.get("raw_swap_proposal_count", 0)),
        "pre_governance_cross_label_proposal_count": int(combined_counts.get("pre_governance_cross_label_proposal_count", 0)),
        "same_label_swap_proposal_count": int(combined_counts.get("same_label_swap_proposal_count", 0)),
        "cross_label_swap_proposal_count": int(combined_counts.get("cross_label_swap_proposal_count", 0)),
        "post_update_count": int(main_counts.get("update_count", 0)),
        "post_wait_count": int(main_counts.get("wait_count", 0)),
        "compare_count": int(combined_counts.get("compare_count", 0)),
        "swap_count": int(combined_counts.get("swap_count", 0)),
        "update_count": int(combined_counts.get("update_count", 0)),
        "wait_count": int(combined_counts.get("wait_count", 0)),
        "frozen_block_count": int(combined_counts.get("frozen_block_count", 0)),
        "transient_frozen_block_count": int(combined_counts.get("transient_frozen_block_count", 0)),
        "transient_active_event_count": int(
            sum(value for key, value in combined_counts.items() if str(key).endswith("_transient_active_event_count"))
        ),
        "invalid_action_count": int(combined_counts.get("invalid_action_count", 0)),
        "work_count": int(combined_counts.get("compare_count", 0) + combined_counts.get("swap_count", 0) + combined_counts.get("update_count", 0)),
        "governance_invocation_count": int(combined_counts.get("governance_invocation_count", 0)),
        "governance_block_count": int(combined_counts.get("governance_block_count", 0)),
        "governance_cross_label_block_count": int(combined_counts.get("governance_cross_label_block_count", 0)),
        "governance_cost_units": int(combined_counts.get("governance_cost_units", 0)),
        "global_controller_invocation_count": int(combined_counts.get("global_controller_invocation_count", 0)),
        "governance_block_fraction": float(combined_counts.get("governance_block_count", 0) / max(1, combined_counts.get("raw_swap_proposal_count", 0))),
        "governance_cost_per_event": float(combined_counts.get("governance_cost_units", 0) / max(1, total_events_executed)),
        "governance_local_goal_delta_per_invocation": float(main_counts.get("governance_local_goal_delta_sum", 0.0) / max(1, combined_counts.get("governance_invocation_count", 0))),
        "governance_heterotypic_edge_delta_per_invocation": float(main_counts.get("governance_heterotypic_edge_delta_sum", 0.0) / max(1, combined_counts.get("governance_invocation_count", 0))),
        "governance_block_reasons_json": str(main_counts.get("governance_block_reasons_json", "{}")),
        "interface_rule_invocation_count": int(combined_counts.get("interface_rule_invocation_count", 0)),
        "interface_block_count": int(combined_counts.get("interface_block_count", 0)),
        "interface_block_fraction": float(combined_counts.get("interface_block_count", 0) / max(1, combined_counts.get("cross_label_swap_proposal_count", 0))),
        "cross_label_swap_acceptance_fraction": float(combined_counts.get("cross_label_swap_count", 0) / max(1, combined_counts.get("cross_label_swap_proposal_count", 0))),
        "same_contact_delta_per_invocation": float(main_counts.get("same_contact_delta_sum", 0.0) / max(1, combined_counts.get("interface_rule_invocation_count", 0))),
        "local_goal_delta_per_invocation": float(main_counts.get("local_goal_delta_sum", 0.0) / max(1, combined_counts.get("interface_rule_invocation_count", 0))),
        "interface_block_reasons_json": str(main_counts.get("interface_block_reasons_json", "{}")),
        "initial_values_head_json": _json_list(initial_values[:20]),
        "target_initial_labels_head_json": _json_list(target_initial_labels[:20]),
        "final_values_head_json": _json_list(values[:20]),
        "final_values_tail_json": _json_list(values[-20:]),
        "final_labels_head_json": _json_list(labels[:20]),
        "final_labels_tail_json": _json_list(labels[-20:]),
        "elapsed_seconds": float(elapsed),
        **initial_target_arrangement,
        **goal_eval,
    }
    return run_row


def _simulate_history_task(task: tuple[HistoryCondition, Mapping[str, Mapping[str, Any]], int, S12Config]) -> dict[str, Any]:
    condition, records_by_id, seed, config = task
    return simulate_history_condition(condition, records_by_id, seed=int(seed), config=config)


def run_s12_sweep(
    records: Sequence[Mapping[str, Any]],
    metadata: pd.DataFrame,
    s10_selected: pd.DataFrame,
    s11_runs: pd.DataFrame,
    s09_rankings: pd.DataFrame,
    config: S12Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or S12Config()
    selected = select_s12_base_contexts(s10_selected, s11_runs, metadata, records, config)
    observed_governance = set(s09_rankings["governance_mechanism_id"].astype(str)) if "governance_mechanism_id" in s09_rankings else set()
    expected_governance = {mechanism.mechanism_id for mechanism in config.governance_mechanisms}
    missing_governance = expected_governance - observed_governance
    if missing_governance:
        raise ValueError(f"S09 ranking table lacks required governance options: {sorted(missing_governance)}")
    conditions = build_s12_conditions(selected, config)
    records_by_id = {str(record["algotypeId"]): dict(record) for record in records}
    needed_policy_ids = {policy_id for condition in conditions for policy_id in condition.base.policy_ids}
    missing = sorted(needed_policy_ids - set(records_by_id))
    if missing:
        raise ValueError(f"S12 selected policies missing from S01 library: {missing}")
    tasks = [(condition, records_by_id, int(seed), config) for condition in conditions for seed in config.seeds]
    if config.worker_count > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(config.worker_count)) as pool:
            run_rows = list(pool.map(_simulate_history_task, tasks))
    else:
        run_rows = [_simulate_history_task(task) for task in tasks]
    run_df = pd.DataFrame(run_rows)
    condition_rows: list[dict[str, Any]] = []
    for condition in conditions:
        counts, expected_counts = _counts_and_expected(condition.base, config.array_size)
        condition_rows.append(
            {
                "condition_id": condition.condition_id,
                "base_context_id": condition.base.base_context_id,
                "source_context_family": condition.base.source_context_family,
                "selection_rank": int(condition.base.selection_rank),
                "selection_reason": condition.base.selection_reason,
                "history_id": condition.history.history_id,
                "history_family": condition.history.history_family,
                "history_encoding_json": _history_encoding(condition.history),
                "interface_rule_id": condition.interface_rule.rule_id,
                "interface_access_stratum": interface_access_stratum(condition.interface_rule),
                "governance_mechanism_id": condition.governance.mechanism_id,
                "governance_access_stratum": governance_access_stratum(condition.governance),
                "intervention_stratum": intervention_stratum(condition.interface_rule, condition.governance),
                "global_controller_like": bool(condition.governance.global_controller_like),
                "policy_ids_json": _json_list(condition.base.policy_ids),
                "display_names_json": _json_list(condition.base.display_names),
                "ratio_targets_json": _json_list(condition.base.ratio_targets),
                "expected_counts_json": _json_object(expected_counts),
                "memory_policy_present": bool(condition.base.memory_policy_present),
                "memory_policy_ids_json": _json_list(condition.base.memory_policy_ids),
                "target_counts_json": _json_list(counts),
            }
        )
    return run_df, pd.DataFrame(condition_rows), selected, history_spec_table(config.history_specs)


def summarize_s12_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["base_context_id", "history_id", "interface_rule_id", "governance_mechanism_id"]
    for _, group in run_df.groupby(group_columns, sort=False):
        first = group.iloc[0]
        state_counts = group["final_state_class"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "base_context_id": first["base_context_id"],
                "source_context_family": first["source_context_family"],
                "history_id": first["history_id"],
                "history_family": first["history_family"],
                "history_start_mode": first["history_start_mode"],
                "interface_rule_id": first["interface_rule_id"],
                "interface_access_stratum": first["interface_access_stratum"],
                "governance_mechanism_id": first["governance_mechanism_id"],
                "governance_access_stratum": first["governance_access_stratum"],
                "intervention_stratum": first["intervention_stratum"],
                "global_controller_like": bool(first["global_controller_like"]),
                "display_names_json": first["display_names_json"],
                "policy_ids_json": first["policy_ids_json"],
                "source_categories_json": first["source_categories_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "memory_policy_present": bool(first["memory_policy_present"]),
                "memory_policy_ids_json": first["memory_policy_ids_json"],
                "seed_count": int(group["seed"].nunique()),
                "run_count": int(len(group)),
                "mean_final_target_quality": float(group["final_target_quality"].mean()),
                "mean_reference_increasing_score": float(group["reference_increasing_score"].mean()),
                "mean_target_delta_vs_initial": float(group["target_delta_vs_initial"].mean()),
                "mean_goal_conflict_index": float(group["goal_conflict_index"].mean()),
                "mean_aggregation_delta_percent": float(group["aggregation_delta_percent"].mean()),
                "mean_interface_count": float(group["interface_count"].mean()),
                "mean_largest_block_fraction": float(group["largest_block_fraction"].mean()),
                "mean_position_bias_abs": float(group["position_bias_abs"].mean()),
                "mean_memory_state_count": float(group["memory_state_count"].mean()),
                "mean_memory_total_frustration": float(group["memory_total_frustration"].mean()),
                "mean_memory_failed_swap_count": float(group["memory_failed_swap_count"].mean()),
                "mean_governance_block_fraction": float(group["governance_block_fraction"].mean()),
                "mean_governance_cost_per_event": float(group["governance_cost_per_event"].mean()),
                "mean_interface_block_fraction": float(group["interface_block_fraction"].mean()),
                "mean_work_count": float(group["work_count"].mean()),
                "invalid_action_count": int(group["invalid_action_count"].sum()),
                "final_composition_matched": bool(group["final_composition_matched"].all()),
                "final_state_class_mode": _mode(group["final_state_class"]),
                "final_state_class_counts_json": _json_object(state_counts),
            }
        )
    return pd.DataFrame(rows)


def compare_to_simultaneous_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    baseline = summary[summary["history_id"] == "simultaneous_mixed_start"].copy()
    metric_columns = [
        "mean_final_target_quality",
        "mean_reference_increasing_score",
        "mean_target_delta_vs_initial",
        "mean_goal_conflict_index",
        "mean_aggregation_delta_percent",
        "mean_interface_count",
        "mean_largest_block_fraction",
        "mean_position_bias_abs",
        "mean_memory_state_count",
        "mean_memory_total_frustration",
        "mean_memory_failed_swap_count",
        "mean_work_count",
    ]
    baseline = baseline[["base_context_id", "interface_rule_id", "governance_mechanism_id", *metric_columns, "final_state_class_mode"]].rename(
        columns={column: f"baseline_{column}" for column in [*metric_columns, "final_state_class_mode"]}
    )
    comparison = summary.merge(baseline, on=["base_context_id", "interface_rule_id", "governance_mechanism_id"], how="left")
    for column in metric_columns:
        comparison[f"delta_{column.removeprefix('mean_')}"] = comparison[column] - comparison[f"baseline_{column}"]
    comparison["schema"] = COMPARISON_SCHEMA
    comparison["final_state_class_changed_vs_simultaneous"] = comparison["final_state_class_mode"].astype(str) != comparison[
        "baseline_final_state_class_mode"
    ].astype(str)
    comparison["history_effect_magnitude"] = (
        comparison["delta_final_target_quality"].abs()
        + comparison["delta_goal_conflict_index"].abs() * 0.10
        + comparison["delta_aggregation_delta_percent"].abs() * 0.005
        + comparison["delta_largest_block_fraction"].abs() * 0.05
    )
    comparison["target_quality_history_shift"] = comparison["delta_final_target_quality"].abs() >= 0.02
    return comparison


def history_strata_summary(comparison: pd.DataFrame, config: S12Config) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (history_id, stratum), group in comparison.groupby(["history_id", "intervention_stratum"], sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "history_id": history_id,
                "history_family": first["history_family"],
                "intervention_stratum": stratum,
                "interface_access_strata": ",".join(sorted(set(group["interface_access_stratum"].astype(str)))),
                "governance_access_strata": ",".join(sorted(set(group["governance_access_stratum"].astype(str)))),
                "condition_count": int(group[["base_context_id", "history_id", "interface_rule_id", "governance_mechanism_id"]].drop_duplicates().shape[0]),
                "memory_context_fraction": float(group["memory_policy_present"].mean()),
                "mean_delta_final_target_quality": float(group["delta_final_target_quality"].mean()),
                "mean_abs_delta_final_target_quality": float(group["delta_final_target_quality"].abs().mean()),
                "mean_delta_goal_conflict_index": float(group["delta_goal_conflict_index"].mean()),
                "mean_abs_delta_goal_conflict_index": float(group["delta_goal_conflict_index"].abs().mean()),
                "mean_delta_aggregation_delta_percent": float(group["delta_aggregation_delta_percent"].mean()),
                "mean_abs_delta_aggregation_delta_percent": float(group["delta_aggregation_delta_percent"].abs().mean()),
                "mean_delta_largest_block_fraction": float(group["delta_largest_block_fraction"].mean()),
                "mean_history_effect_magnitude": float(group["history_effect_magnitude"].mean()),
                "state_class_change_rate": float(group["final_state_class_changed_vs_simultaneous"].mean()),
                "target_quality_shift_rate": float(group["target_quality_history_shift"].mean()),
                "mean_governance_block_fraction": float(group["mean_governance_block_fraction"].mean()),
                "mean_interface_block_fraction": float(group["mean_interface_block_fraction"].mean()),
                "mean_governance_cost_per_event": float(group["mean_governance_cost_per_event"].mean()),
                "supportive_history_effect_by_threshold": bool(
                    history_id != "simultaneous_mixed_start" and group["history_effect_magnitude"].mean() >= config.history_effect_threshold
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_history_effect_magnitude", "history_id"], ascending=[False, True], kind="mergesort")


def outcome_classification(strata: pd.DataFrame, validation_passed: bool) -> str:
    if not validation_passed or strata.empty:
        return "null"
    non_baseline = strata[strata["history_id"] != "simultaneous_mixed_start"]
    if non_baseline.empty:
        return "null"
    local = non_baseline[~non_baseline["governance_access_strata"].astype(str).str.contains("global", na=False)]
    global_like = non_baseline[non_baseline["governance_access_strata"].astype(str).str.contains("global", na=False)]
    local_support = bool((local["supportive_history_effect_by_threshold"] == True).any())  # noqa: E712
    global_support = bool((global_like["supportive_history_effect_by_threshold"] == True).any())  # noqa: E712
    if local_support:
        return "supportive"
    if global_support:
        return "constraining/contradictory"
    return "null"


def validate_s12_outputs(
    run_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    selected: pd.DataFrame,
    history_specs: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    strata: pd.DataFrame,
    config: S12Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    seed_set = {int(seed) for seed in config.seeds}
    expected_histories = {spec.history_id for spec in config.history_specs}
    expected_interfaces = {rule.rule_id for rule in config.interface_rules}
    expected_governance = {mechanism.mechanism_id for mechanism in config.governance_mechanisms}
    condition_seed_sets = (
        run_df.groupby(["base_context_id", "history_id", "interface_rule_id", "governance_mechanism_id"])["seed"]
        .agg(lambda values: {int(value) for value in values})
        .tolist()
        if not run_df.empty
        else []
    )
    base_history_sets = (
        run_df.groupby("base_context_id")["history_id"].agg(lambda values: set(map(str, values))).tolist() if not run_df.empty else []
    )
    base_history_interface_sets = (
        run_df.groupby(["base_context_id", "history_id"])["interface_rule_id"].agg(lambda values: set(map(str, values))).tolist()
        if not run_df.empty
        else []
    )
    base_history_interface_governance_sets = (
        run_df.groupby(["base_context_id", "history_id", "interface_rule_id"])["governance_mechanism_id"]
        .agg(lambda values: set(map(str, values)))
        .tolist()
        if not run_df.empty
        else []
    )
    finite_columns = ["delta_final_target_quality", "delta_goal_conflict_index", "history_effect_magnitude"]
    finite_comparison = bool(
        not comparison.empty
        and set(finite_columns).issubset(comparison.columns)
        and comparison[finite_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().all().all()
    )
    prior_rows = run_df[run_df["history_family"] == "prior_exposure_memory"] if not run_df.empty else pd.DataFrame()
    transient_rows = run_df[run_df["history_family"] == "transient_perturbation"] if not run_df.empty else pd.DataFrame()
    staged_rows = run_df[run_df["history_family"] == "staged_introduction"] if not run_df.empty else pd.DataFrame()
    checks = [
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "histories_encoded_in_config",
            "success": bool(set(history_specs["history_id"].astype(str)) == expected_histories and history_specs["history_encoding_json" if "history_encoding_json" in history_specs.columns else "description"].notna().all()),
            "observed": sorted(history_specs["history_id"].astype(str).tolist()) if not history_specs.empty else [],
            "expected": sorted(expected_histories),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "s10_and_e04_memory_contexts_selected",
            "success": bool(
                not selected.empty
                and "s10_graft_sensitive" in set(selected["source_context_family"].astype(str))
                and "e04_memory_policy_input" in set(selected["source_context_family"].astype(str))
                and selected["memory_policy_present"].astype(bool).any()
            ),
            "observed": selected[["base_context_id", "source_context_family", "memory_policy_present"]].to_dict(orient="records") if not selected.empty else [],
            "expected": "S12 uses S10 staged contexts and at least one E04 memory policy input context",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "matched_final_composition_where_possible",
            "success": bool(not run_df.empty and run_df["matched_final_composition_possible"].all() and run_df["final_composition_matched"].all()),
            "observed": f"composition_failures={int((~run_df['final_composition_matched']).sum()) if not run_df.empty else 'missing'}",
            "expected": "all S12 histories preserve matched final policy counts",
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
            "validation_case": "all_histories_per_base_context",
            "success": bool(base_history_sets and all(item == expected_histories for item in base_history_sets)),
            "observed": f"base_context_groups={len(base_history_sets)}",
            "expected": sorted(expected_histories),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "all_interface_rules_per_base_history",
            "success": bool(base_history_interface_sets and all(item == expected_interfaces for item in base_history_interface_sets)),
            "observed": f"base_history_groups={len(base_history_interface_sets)}",
            "expected": sorted(expected_interfaces),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "all_governance_mechanisms_per_base_history_interface",
            "success": bool(
                base_history_interface_governance_sets and all(item == expected_governance for item in base_history_interface_governance_sets)
            ),
            "observed": f"base_history_interface_groups={len(base_history_interface_governance_sets)}",
            "expected": sorted(expected_governance),
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "matched_seed_sets_by_condition",
            "success": bool(condition_seed_sets and all(item == seed_set for item in condition_seed_sets)),
            "observed": f"condition_groups={len(condition_seed_sets)} expected_seeds={sorted(seed_set)}",
            "expected": "each condition uses the configured matched seed set",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "staged_introduction_logged",
            "success": bool(not staged_rows.empty and staged_rows["introduction_event"].ge(0).all() and (staged_rows["staged_partner_indices_json"].astype(str) != "[]").all()),
            "observed": sorted(staged_rows["history_id"].astype(str).unique().tolist()) if not staged_rows.empty else [],
            "expected": "staged histories log introduction events and partner placement indices",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "transient_perturbation_windows_encoded",
            "success": bool(
                not transient_rows.empty
                and transient_rows["transient_start_event"].ge(0).all()
                and transient_rows["transient_end_event"].gt(transient_rows["transient_start_event"]).all()
                and transient_rows["transient_active_event_count"].gt(0).all()
            ),
            "observed": sorted(transient_rows["history_id"].astype(str).unique().tolist()) if not transient_rows.empty else [],
            "expected": "early and late transient perturbation windows are encoded and activated",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "prior_exposure_preserves_memory_across_reset",
            "success": bool(
                not prior_rows.empty
                and prior_rows["prior_exposure_events"].gt(0).all()
                and prior_rows["reset_after_exposure"].astype(bool).all()
                and prior_rows["preserves_memory_across_reset"].astype(bool).all()
                and prior_rows["memory_state_count"].gt(0).all()
            ),
            "observed": f"prior_rows={len(prior_rows)} memory_context_rows={int(prior_rows['memory_policy_present'].sum()) if not prior_rows.empty else 0}",
            "expected": "prior exposure rows preserve non-empty memory state through reset",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "finite_history_baseline_comparisons_and_strata",
            "success": bool(finite_comparison and not strata.empty and strata["mean_history_effect_magnitude"].notna().all()),
            "observed": f"comparison_rows={len(comparison)} strata_rows={len(strata)}",
            "expected": "all history-vs-simultaneous deltas and stratum summaries are finite",
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
            "validation_case": "figure_written",
            "success": bool(figure_written),
            "observed": str(figure_written),
            "expected": "history_dependence_summary.png written and non-empty",
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
