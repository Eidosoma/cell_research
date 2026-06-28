"""E06 S10 staged graft experiments for chimeric collectives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e02_deterministic_simulator.metrics import aggregation, sortedness_percent, state_hash

from .dominance import (
    DEFAULT_S05_BOUNDARY_CAVEATS_PATH,
    artifact_records,
    load_boundary_caveats,
    predefined_winner_criteria,
    score_dominance_results,
)
from .goals import augment_goal_results, condition_goal_metadata, policy_goal_specs, reverse_directions_for_assignment
from .governance import (
    DEFAULT_S06_PAIR_SUMMARY_PATH,
    DEFAULT_S06_WINNER_CRITERIA_PATH,
    GOVERNANCE_VERSION,
    GovernanceConfig,
    GovernanceEventSimulator,
)
from .interfaces import INTERFACE_RULE_VERSION, interface_rule_config_by_variant
from .mixtures import (
    DEFAULT_PANEL_PATH,
    compact_json,
    compact_result_csv,
    git_value,
    initial_values,
    load_ready_panel,
    parse_json_maybe,
    simulator_backend,
    write_json,
)
from .mosaics import MOSAIC_CLASSIFIER_VERSION, REQUESTED_MOSAIC_CLASSES, classify_mosaic_formations, safe_float
from .panel import CLAIM_BOUNDARY, instantiate_panel_policy, json_ready


STEP_ID = "S10"
STEP_NUMBER = 10
GRAFT_VERSION = "e06_s10_graft_experiments.v1"
DEFAULT_S09_RESULTS_PATH = Path("/artifacts/results/e06_governance_mechanisms.parquet")
DEFAULT_S09_EFFECT_SUMMARY_PATH = Path("/artifacts/research_steps/S09/governance_effect_summary.parquet")
DEFAULT_S09_AUDIT_PATH = Path("/artifacts/research_steps/S09/governance_information_audit.parquet")
GRAFT_SIZES: tuple[int, ...] = (10, 20)
GRAFT_POSITIONS: tuple[str, ...] = ("left_edge_patch", "center_patch")
GRAFT_TIMINGS: tuple[tuple[str, int], ...] = (("early", 500), ("late", 1500))
GRAFT_INTERFACE_VARIANTS: tuple[str, ...] = ("disabled_baseline", "combined_self_boundary")
_PANEL_LOOKUP: dict[str, dict[str, Any]] = {}


def load_winner_criteria(path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return predefined_winner_criteria(load_boundary_caveats(DEFAULT_S05_BOUNDARY_CAVEATS_PATH))


def load_s09_results(path: Path = DEFAULT_S09_RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"S09 governance results missing: {path}")
    df = pd.read_parquet(path)
    required = {
        "conditionId",
        "s09ContextIndex",
        "governanceVariant",
        "goalMode",
        "leftPanelPolicyId",
        "rightPanelPolicyId",
        "policyIdsJson",
        "valueSeed",
        "schedulerSeed",
        "tieBreakerSeed",
        "runSucceeded",
        "metricBoundary",
        "s07MosaicClass",
        "finalTargetQualityScore",
        "minPolicyGoalScore",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"S09 governance results missing required columns: {missing}")
    if df.empty:
        raise ValueError("S09 governance results are empty")
    return df


def selected_governance_variants(effect_summary: pd.DataFrame, *, max_local: int = 2) -> tuple[str, ...]:
    base = ["no_governance_control"]
    if effect_summary.empty:
        local = ["conflict_resolution_range1", "local_voting_range1"]
    else:
        broad_flags = (
            effect_summary["broadControlLike"].map(bool)
            if "broadControlLike" in effect_summary.columns
            else pd.Series(False, index=effect_summary.index)
        )
        eligible = effect_summary[
            ~effect_summary["governanceVariant"].astype(str).isin({"no_governance_control", "organizer_global_upper_bound"})
            & ~broad_flags
        ].copy()
        ranked = (
            eligible.groupby("governanceVariant", dropna=False)["rankScore"]
            .mean()
            .sort_values(ascending=False, kind="mergesort")
        )
        local = [str(item) for item in ranked.head(int(max_local)).index]
    for fallback in ["conflict_resolution_range1", "local_voting_range1", "leader_cells_range2"]:
        if len(local) >= int(max_local):
            break
        if fallback not in local:
            local.append(fallback)
    out = [*base, *local[: int(max_local)], "organizer_global_upper_bound"]
    return tuple(dict.fromkeys(out))


def select_s10_base_contexts(
    s09_results: pd.DataFrame,
    effect_summary: pd.DataFrame,
    *,
    max_contexts: int = 4,
) -> pd.DataFrame:
    no_gov = s09_results[
        (s09_results["governanceVariant"].astype(str) == "no_governance_control")
        & (s09_results["goalMode"].astype(str).isin({"same_goal", "opposite_goal", "partially_compatible"}))
        & s09_results["runSucceeded"].map(bool)
    ].copy()
    if no_gov.empty:
        raise ValueError("S10 requires S09 no_governance_control rows")
    if not effect_summary.empty:
        local_effect = (
            effect_summary[
                ~effect_summary["governanceVariant"].astype(str).isin({"no_governance_control", "organizer_global_upper_bound"})
            ]
            .groupby("goalMode", dropna=False)
            .agg(bestLocalGovernanceRankScore=("rankScore", "max"), bestLocalGovernanceRescueRate=("rescueSuccessProxyRate", "max"))
            .reset_index()
        )
        no_gov = no_gov.merge(local_effect, on="goalMode", how="left")
    no_gov["bestLocalGovernanceRankScore"] = pd.to_numeric(no_gov.get("bestLocalGovernanceRankScore", 0.0), errors="coerce").fillna(0.0)
    no_gov["bestLocalGovernanceRescueRate"] = pd.to_numeric(no_gov.get("bestLocalGovernanceRescueRate", 0.0), errors="coerce").fillna(0.0)
    no_gov["s10StressScore"] = (
        (100.0 - pd.to_numeric(no_gov["minPolicyGoalScore"], errors="coerce").fillna(100.0)) / 100.0
        + (1.0 - pd.to_numeric(no_gov["finalTargetQualityScore"], errors="coerce").fillna(1.0))
        + pd.to_numeric(no_gov.get("s09InterfaceSensitiveScore", 0.0), errors="coerce").fillna(0.0) / 5.0
        + no_gov["bestLocalGovernanceRankScore"].clip(lower=0.0)
    )
    selected_ids: list[str] = []
    reasons: dict[str, list[str]] = {}

    def add(row: Mapping[str, Any], reason: str) -> None:
        cid = str(row["conditionId"])
        if cid not in selected_ids:
            selected_ids.append(cid)
        reasons.setdefault(cid, []).append(reason)

    for goal_mode in ["same_goal", "opposite_goal", "partially_compatible"]:
        rows = no_gov[no_gov["goalMode"].astype(str) == goal_mode].sort_values(
            ["s10StressScore", "s09ContextIndex", "conditionId"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        if not rows.empty:
            add(rows.iloc[0], f"goal_anchor_{goal_mode}")
    remaining = no_gov.sort_values(
        ["s10StressScore", "s09ContextIndex", "conditionId"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    for row in remaining.to_dict(orient="records"):
        if len(selected_ids) >= int(max_contexts):
            break
        add(row, "high_stress_s09_context")
    selected = no_gov[no_gov["conditionId"].astype(str).isin(selected_ids)].copy()
    selected["_order"] = selected["conditionId"].map({cid: index for index, cid in enumerate(selected_ids)})
    selected = selected.sort_values("_order", kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)
    selected.insert(0, "s10BaseContextIndex", np.arange(len(selected), dtype=int))
    selected["s10SelectionReason"] = selected["conditionId"].map(lambda value: "+".join(sorted(set(reasons.get(str(value), [])))))
    selected["s10SourceS09ConditionId"] = selected["conditionId"].astype(str)
    return selected


def graft_positions(n: int, graft_size: int, position_name: str) -> list[int]:
    n = int(n)
    graft_size = int(graft_size)
    if graft_size <= 0 or graft_size > n:
        raise ValueError("graft_size must be in [1, n]")
    if position_name == "left_edge_patch":
        start = 0
    elif position_name == "center_patch":
        start = (n - graft_size) // 2
    elif position_name == "right_edge_patch":
        start = n - graft_size
    else:
        raise ValueError(f"unknown graft position: {position_name}")
    return list(range(start, start + graft_size))


def _counts_json(values: Sequence[str]) -> str:
    return compact_json(dict(Counter(str(value) for value in values)))


def _label_assignment(policy_ids: Sequence[str], assigned_ids: Sequence[str]) -> list[int]:
    lookup = {str(pid): index for index, pid in enumerate(policy_ids)}
    return [int(lookup[str(pid)]) for pid in assigned_ids]


def _pre_state_id(
    base: Mapping[str, Any],
    *,
    timing_label: str,
    activation_cap: int,
    governance_variant: str,
    interface_variant: str,
    host_policy_id: str,
) -> str:
    payload = {
        "source": str(base["conditionId"]),
        "timing": timing_label,
        "cap": int(activation_cap),
        "governance": governance_variant,
        "interface": interface_variant,
        "host": host_policy_id,
    }
    digest = hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()[:16]
    return f"s10_pre_{digest}"


def _control_condition_id(pre_state_id: str, governance_variant: str, interface_variant: str) -> str:
    digest = hashlib.sha256(f"{pre_state_id}|{governance_variant}|{interface_variant}|control".encode("utf-8")).hexdigest()[:12]
    return f"s10_control_{digest}"


def _graft_condition_id(pre_state_id: str, graft_size: int, position_name: str, governance_variant: str, interface_variant: str) -> str:
    digest = hashlib.sha256(
        f"{pre_state_id}|{graft_size}|{position_name}|{governance_variant}|{interface_variant}|graft".encode("utf-8")
    ).hexdigest()[:12]
    return f"s10_graft_{digest}"


def build_graft_condition_matrix(
    selected_contexts: pd.DataFrame,
    *,
    governance_variants: Sequence[str],
    graft_sizes: Sequence[int] = GRAFT_SIZES,
    graft_position_names: Sequence[str] = GRAFT_POSITIONS,
    graft_timings: Sequence[tuple[str, int]] = GRAFT_TIMINGS,
    interface_variants: Sequence[str] = GRAFT_INTERFACE_VARIANTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pre_rows: dict[str, dict[str, Any]] = {}
    insertion_rows: list[dict[str, Any]] = []
    control_rows: dict[str, dict[str, Any]] = {}
    for base in selected_contexts.to_dict(orient="records"):
        policy_ids = [str(item) for item in parse_json_maybe(base["policyIdsJson"], [])]
        if len(policy_ids) != 2:
            policy_ids = [str(base["leftPanelPolicyId"]), str(base["rightPanelPolicyId"])]
        host_policy_id = str(base["leftPanelPolicyId"])
        donor_policy_id = str(base["rightPanelPolicyId"])
        ordered_policy_ids = [host_policy_id, donor_policy_id]
        n = int(base["n"])
        initial_values_json = str(base.get("initialValuesJson", ""))
        initial_values_for_hash = parse_json_maybe(initial_values_json, [])
        for timing_label, pre_cap in graft_timings:
            for governance_variant in governance_variants:
                for interface_variant in interface_variants:
                    governance_config = GovernanceConfig.from_payload(str(governance_variant))
                    interface_config = interface_rule_config_by_variant(str(interface_variant))
                    pre_state_id = _pre_state_id(
                        base,
                        timing_label=str(timing_label),
                        activation_cap=int(pre_cap),
                        governance_variant=str(governance_variant),
                        interface_variant=str(interface_variant),
                        host_policy_id=host_policy_id,
                    )
                    host_assignment = [host_policy_id] * n
                    goal_specs = policy_goal_specs(ordered_policy_ids, str(base["goalMode"]))
                    reverse = reverse_directions_for_assignment(host_assignment, goal_specs)
                    pre_payload = {
                        **{key: json_ready(value) for key, value in base.items()},
                        "preGraftStateId": pre_state_id,
                        "researchStepId": STEP_ID,
                        "sourceResearchStepId": "S09",
                        "s10Version": GRAFT_VERSION,
                        "hostPolicyId": host_policy_id,
                        "donorPolicyId": donor_policy_id,
                        "preGraftTimingLabel": str(timing_label),
                        "preGraftActivationCap": int(pre_cap),
                        "governanceVariant": str(governance_variant),
                        "governanceFamily": governance_config.family,
                        "governanceConfigJson": compact_json(governance_config.to_dict()),
                        "influenceRange": int(governance_config.influence_range),
                        "informationAccess": governance_config.information_access,
                        "localInformationOnly": bool(governance_config.local_information_only),
                        "usesGlobalState": bool(governance_config.uses_global_state),
                        "usesTargetMap": bool(governance_config.uses_target_map),
                        "usesOrganizer": bool(governance_config.uses_organizer),
                        "broadControlLike": bool(governance_config.broad_control_like),
                        "interfaceRuleVariant": str(interface_variant),
                        "interfaceRuleFamily": interface_config.rule_family,
                        "interfaceRuleEnabled": bool(interface_config.enabled),
                        "interfaceRuleConfigJson": compact_json(interface_config.to_dict()),
                        "policyIdsJson": compact_json(ordered_policy_ids),
                        "preGraftInitialValuesJson": compact_json(initial_values_for_hash),
                        "preGraftInitialPanelPolicyIdsJson": compact_json(host_assignment),
                        "preGraftInitialNumericLabelsJson": compact_json(_label_assignment(ordered_policy_ids, host_assignment)),
                        "preGraftGoalReverseDirectionsJson": compact_json(reverse),
                        "preGraftConditionHash": hashlib.sha256(
                            compact_json(
                                {
                                    "preGraftStateId": pre_state_id,
                                    "initialValues": initial_values_for_hash,
                                    "host": host_policy_id,
                                    "goalMode": base["goalMode"],
                                    "governance": governance_variant,
                                    "interface": interface_variant,
                                }
                            ).encode("utf-8")
                        ).hexdigest()[:16],
                    }
                    pre_rows[pre_state_id] = pre_payload
                    control_id = _control_condition_id(pre_state_id, str(governance_variant), str(interface_variant))
                    if control_id not in control_rows:
                        control_assignment = list(host_assignment)
                        control_payload = {
                            **pre_payload,
                            "conditionId": control_id,
                            "s10ConditionKind": "matched_no_graft_control",
                            "graftApplied": False,
                            "matchedNoGraftConditionId": control_id,
                            "graftSize": 0,
                            "graftPositionName": "none",
                            "graftStartIndex": -1,
                            "graftEndExclusive": -1,
                            "graftPositionsJson": compact_json([]),
                            "graftInsertionDescriptorJson": compact_json(
                                {
                                    "schema": GRAFT_VERSION,
                                    "graftApplied": False,
                                    "hostPolicyId": host_policy_id,
                                    "donorPolicyId": donor_policy_id,
                                    "description": "Matched no-graft control starts from the same saved pre-graft state.",
                                }
                            ),
                            "postGraftInitialPanelPolicyIdsJson": compact_json(control_assignment),
                            "postGraftInitialNumericLabelsJson": compact_json(_label_assignment(ordered_policy_ids, control_assignment)),
                            "targetCountsJson": _counts_json(control_assignment),
                            "conditionHash": hashlib.sha256(f"{control_id}|{pre_state_id}".encode("utf-8")).hexdigest()[:16],
                        }
                        control_rows[control_id] = control_payload
                        rows.append(control_payload)
                    for graft_size in graft_sizes:
                        for position_name in graft_position_names:
                            positions = graft_positions(n, int(graft_size), str(position_name))
                            start = min(positions)
                            end = max(positions) + 1
                            graft_assignment = list(host_assignment)
                            for pos in positions:
                                graft_assignment[int(pos)] = donor_policy_id
                            condition_id = _graft_condition_id(
                                pre_state_id,
                                int(graft_size),
                                str(position_name),
                                str(governance_variant),
                                str(interface_variant),
                            )
                            descriptor = {
                                "schema": GRAFT_VERSION,
                                "graftApplied": True,
                                "hostPolicyId": host_policy_id,
                                "donorPolicyId": donor_policy_id,
                                "graftSize": int(graft_size),
                                "graftPositionName": str(position_name),
                                "graftStartIndex": int(start),
                                "graftEndExclusive": int(end),
                                "graftPositions": positions,
                                "graftProtocol": "fixed-size policy-identity patch inserted after saved host-only pre-graft run",
                                "claimBoundary": CLAIM_BOUNDARY,
                            }
                            insertion_rows.append(
                                {
                                    "conditionId": condition_id,
                                    "preGraftStateId": pre_state_id,
                                    "matchedNoGraftConditionId": control_id,
                                    **descriptor,
                                    "graftPositionsJson": compact_json(positions),
                                    "graftInsertionDescriptorJson": compact_json(descriptor),
                                }
                            )
                            rows.append(
                                {
                                    **pre_payload,
                                    "conditionId": condition_id,
                                    "s10ConditionKind": "graft",
                                    "graftApplied": True,
                                    "matchedNoGraftConditionId": control_id,
                                    "graftSize": int(graft_size),
                                    "graftPositionName": str(position_name),
                                    "graftStartIndex": int(start),
                                    "graftEndExclusive": int(end),
                                    "graftPositionsJson": compact_json(positions),
                                    "graftInsertionDescriptorJson": compact_json(descriptor),
                                    "postGraftInitialPanelPolicyIdsJson": compact_json(graft_assignment),
                                    "postGraftInitialNumericLabelsJson": compact_json(_label_assignment(ordered_policy_ids, graft_assignment)),
                                    "targetCountsJson": _counts_json(graft_assignment),
                                    "conditionHash": hashlib.sha256(
                                        compact_json(
                                            {
                                                "conditionId": condition_id,
                                                "preGraftStateId": pre_state_id,
                                                "positions": positions,
                                                "governance": governance_variant,
                                                "interface": interface_variant,
                                            }
                                        ).encode("utf-8")
                                    ).hexdigest()[:16],
                                }
                            )
    condition_df = pd.DataFrame(rows).reset_index(drop=True)
    pre_df = pd.DataFrame(pre_rows.values()).reset_index(drop=True)
    insertion_df = pd.DataFrame(insertion_rows).reset_index(drop=True)
    return condition_df, pre_df, insertion_df


def _policy_records(policy_ids: Sequence[str], panel_lookup: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(panel_lookup[str(pid)]) for pid in policy_ids]


def _run_simulation(
    *,
    values: Sequence[int],
    assigned_ids: Sequence[str],
    policy_ids: Sequence[str],
    reverse_directions: Sequence[bool],
    panel_lookup: Mapping[str, Mapping[str, Any]],
    condition_id: str,
    governance_variant: str,
    interface_variant: str,
    scheduler_seed: int,
    tie_breaker_seed: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[GovernanceEventSimulator, Any, str]:
    label_by_id = {str(pid): index for index, pid in enumerate(policy_ids)}
    numeric_labels = [int(label_by_id[str(pid)]) for pid in assigned_ids]
    policy_cache = {pid: instantiate_panel_policy(panel_lookup[str(pid)]) for pid in policy_ids}
    policies = [policy_cache[str(pid)] for pid in assigned_ids]
    records = _policy_records(policy_ids, panel_lookup)
    source_backend = simulator_backend(records)
    first_signal = next((policy for policy in policy_cache.values() if hasattr(policy, "signal_config")), None)
    signal_config = getattr(first_signal, "signal_config", "no_signal")
    simulator = GovernanceEventSimulator(
        [int(value) for value in values],
        policies,
        labels=numeric_labels,
        reverse_directions=[bool(value) for value in reverse_directions],
        scheduler_seed=int(scheduler_seed),
        tie_breaker_seed=int(tie_breaker_seed),
        condition_id=str(condition_id),
        research_step_id=STEP_ID,
        signal_config=signal_config,
        auto_wrap_policies=False,
        trace_signal_activations=False,
        trace_memory_activations=False,
        interface_rule_config=str(interface_variant),
        governance_config=str(governance_variant),
        implementation="e06_s10_graft",
    )
    result = simulator.run(
        max_activations=int(max_activations),
        max_swaps=int(max_swaps),
        max_comparisons=int(max_comparisons),
        no_move_check_interval=len(values),
    )
    return simulator, result, source_backend


def run_pre_graft_state(
    pre_condition: Mapping[str, Any],
    panel_lookup: Mapping[str, Mapping[str, Any]],
    *,
    max_swaps: int = 1200,
    max_comparisons: int = 12000,
) -> dict[str, Any]:
    started = time.perf_counter()
    policy_ids = [str(item) for item in parse_json_maybe(pre_condition["policyIdsJson"], [])]
    assigned = [str(item) for item in parse_json_maybe(pre_condition["preGraftInitialPanelPolicyIdsJson"], [])]
    values = [int(item) for item in parse_json_maybe(pre_condition["preGraftInitialValuesJson"], [])]
    reverse = [bool(item) for item in parse_json_maybe(pre_condition["preGraftGoalReverseDirectionsJson"], [])]
    try:
        simulator, result, source_backend = _run_simulation(
            values=values,
            assigned_ids=assigned,
            policy_ids=policy_ids,
            reverse_directions=reverse,
            panel_lookup=panel_lookup,
            condition_id=str(pre_condition["preGraftStateId"]),
            governance_variant=str(pre_condition["governanceVariant"]),
            interface_variant=str(pre_condition["interfaceRuleVariant"]),
            scheduler_seed=int(pre_condition["schedulerSeed"]) + 101,
            tie_breaker_seed=int(pre_condition["tieBreakerSeed"]) + 101,
            max_activations=int(pre_condition["preGraftActivationCap"]),
            max_swaps=max_swaps,
            max_comparisons=max_comparisons,
        )
        final_values = simulator.current_values()
        final_ids = [policy_ids[int(cell.label)] for cell in simulator.cells]
        return {
            **{key: json_ready(value) for key, value in pre_condition.items()},
            "preGraftRunSucceeded": True,
            "preGraftRunError": "",
            "sourceSimulationBackend": source_backend,
            "preGraftFinalValuesJson": compact_json(final_values),
            "preGraftFinalPanelPolicyIdsJson": compact_json(final_ids),
            "preGraftFinalStateHash": state_hash(final_values),
            "preGraftFinalPolicyAssignmentHash": hashlib.sha256(compact_json(final_ids).encode("utf-8")).hexdigest(),
            "preGraftStopReason": result.stop_reason,
            "preGraftSwapCount": int(result.swap_count),
            "preGraftComparisonCount": int(result.comparison_count),
            "preGraftActivationCount": int(result.activation_count),
            "preGraftCompleted": bool(result.completed),
            "preGraftSortednessPercent": sortedness_percent(final_values),
            "preGraftAggregation": aggregation(final_ids),
            "preGraftRuntimeSeconds": float(time.perf_counter() - started),
            "claimBoundary": CLAIM_BOUNDARY,
        }
    except Exception as exc:  # pragma: no cover - artifact-level failure accounting
        return {
            **{key: json_ready(value) for key, value in pre_condition.items()},
            "preGraftRunSucceeded": False,
            "preGraftRunError": f"{type(exc).__name__}: {exc}",
            "preGraftRuntimeSeconds": float(time.perf_counter() - started),
            "claimBoundary": CLAIM_BOUNDARY,
        }


def _pre_worker_init(panel_records: Sequence[Mapping[str, Any]]) -> None:
    global _PANEL_LOOKUP
    _PANEL_LOOKUP = {str(row["panelPolicyId"]): dict(row) for row in panel_records}


def _run_pre_worker(record: Mapping[str, Any], max_swaps: int, max_comparisons: int) -> dict[str, Any]:
    return run_pre_graft_state(record, _PANEL_LOOKUP, max_swaps=max_swaps, max_comparisons=max_comparisons)


def run_pre_graft_states(
    pre_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    workers: int = 1,
    max_swaps: int = 1200,
    max_comparisons: int = 12000,
) -> pd.DataFrame:
    records = [dict(row) for row in pre_df.to_dict(orient="records")]
    panel_records = [dict(row) for row in panel_df.to_dict(orient="records")]
    if int(workers) <= 1:
        lookup = {str(row["panelPolicyId"]): row for row in panel_records}
        return pd.DataFrame([run_pre_graft_state(record, lookup, max_swaps=max_swaps, max_comparisons=max_comparisons) for record in records])
    with ProcessPoolExecutor(max_workers=int(workers), initializer=_pre_worker_init, initargs=(panel_records,)) as pool:
        futures = [pool.submit(_run_pre_worker, record, int(max_swaps), int(max_comparisons)) for record in records]
        return pd.DataFrame([future.result() for future in futures])


def run_graft_condition(
    condition: Mapping[str, Any],
    pre_state: Mapping[str, Any],
    panel_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    max_activations: int = 3500,
    max_swaps: int = 2500,
    max_comparisons: int = 20000,
) -> dict[str, Any]:
    panel_lookup = panel_lookup or _PANEL_LOOKUP
    started = time.perf_counter()
    policy_ids = [str(item) for item in parse_json_maybe(condition["policyIdsJson"], [])]
    assigned = [str(item) for item in parse_json_maybe(condition["postGraftInitialPanelPolicyIdsJson"], [])]
    pre_values = [int(item) for item in parse_json_maybe(pre_state["preGraftFinalValuesJson"], [])]
    specs = policy_goal_specs(policy_ids, str(condition["goalMode"]))
    reverse = reverse_directions_for_assignment(assigned, specs)
    try:
        simulator, result, source_backend = _run_simulation(
            values=pre_values,
            assigned_ids=assigned,
            policy_ids=policy_ids,
            reverse_directions=reverse,
            panel_lookup=panel_lookup,
            condition_id=str(condition["conditionId"]),
            governance_variant=str(condition["governanceVariant"]),
            interface_variant=str(condition["interfaceRuleVariant"]),
            scheduler_seed=int(condition["schedulerSeed"]) + 1009,
            tie_breaker_seed=int(condition["tieBreakerSeed"]) + 1009,
            max_activations=max_activations,
            max_swaps=max_swaps,
            max_comparisons=max_comparisons,
        )
        final_values = simulator.current_values()
        final_labels = [int(cell.label) for cell in simulator.cells]
        final_ids = [policy_ids[label] for label in final_labels]
        initial_ids = list(assigned)
        initial_counts = Counter(initial_ids)
        final_counts = Counter(final_ids)
        host_id = str(condition["hostPolicyId"])
        donor_id = str(condition["donorPolicyId"])
        graft_positions_list = [int(pos) for pos in parse_json_maybe(condition.get("graftPositionsJson", "[]"), [])]
        final_donor_positions = [idx for idx, pid in enumerate(final_ids) if pid == donor_id]
        donor_spread = (
            (max(final_donor_positions) - min(final_donor_positions) + 1) / max(1, len(final_ids))
            if final_donor_positions
            else 0.0
        )
        donor_mean = float(np.mean(final_donor_positions) / max(1, len(final_ids) - 1)) if final_donor_positions else float("nan")
        initial_mean = float(np.mean(graft_positions_list) / max(1, len(final_ids) - 1)) if graft_positions_list else float("nan")
        final_donor_aggregation = aggregation(["donor" if pid == donor_id else "host" for pid in final_ids]) if final_ids else float("nan")
        initial_donor_aggregation = aggregation(["donor" if pid == donor_id else "host" for pid in initial_ids]) if initial_ids else float("nan")
        interface_summary = simulator.interface_summary()
        governance_summary = simulator.governance_summary()
        completed = sortedness_percent(final_values) >= 100.0 - 1e-9
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": True,
            "runError": "",
            "simulationBackend": "graft_governance_event",
            "sourceSimulationBackend": source_backend,
            "preGraftFinalValuesJson": str(pre_state["preGraftFinalValuesJson"]),
            "preGraftFinalPanelPolicyIdsJson": str(pre_state["preGraftFinalPanelPolicyIdsJson"]),
            "initialValuesJson": compact_json(pre_values),
            "finalValuesJson": compact_json(final_values),
            "initialStateHash": state_hash(pre_values),
            "finalStateHash": state_hash(final_values),
            "initialPanelPolicyIdsJson": compact_json(initial_ids),
            "finalPanelPolicyIdsJson": compact_json(final_ids),
            "actualCountsJson": compact_json(dict(initial_counts)),
            "finalCountsJson": compact_json(dict(final_counts)),
            "actualRatiosMatchTarget": compact_json(dict(initial_counts)) == str(condition["targetCountsJson"]),
            "valueCountsConserved": Counter(pre_values) == Counter(final_values),
            "policyCountsConserved": initial_counts == final_counts,
            "completed": bool(completed),
            "stopReason": result.stop_reason,
            "finalStateClass": "sorted" if completed else ("resource_capped_partial" if str(result.stop_reason).startswith("max_") else "partial"),
            "initialSortednessPercent": sortedness_percent(pre_values),
            "finalSortednessPercent": sortedness_percent(final_values),
            "sortednessGain": sortedness_percent(final_values) - sortedness_percent(pre_values),
            "initialAggregation": aggregation(initial_ids),
            "finalAggregation": aggregation(final_ids),
            "aggregationDelta": aggregation(final_ids) - aggregation(initial_ids),
            "swapCount": int(result.swap_count),
            "comparisonCount": int(result.comparison_count),
            "activationCount": int(result.activation_count),
            "eventCount": int(result.event_count),
            "blockedMoveAttempts": int(result.blocked_move_attempts),
            "frozenSwapAttempts": int(result.frozen_swap_attempts),
            "graftDonorInitialCount": int(initial_counts.get(donor_id, 0)),
            "graftDonorFinalCount": int(final_counts.get(donor_id, 0)),
            "graftDonorRetentionFraction": float(final_counts.get(donor_id, 0) / max(1, initial_counts.get(donor_id, 0))),
            "graftDonorInitialPositionsJson": compact_json(graft_positions_list),
            "graftDonorFinalPositionsJson": compact_json(final_donor_positions),
            "graftDonorInitialMeanPosition": initial_mean,
            "graftDonorFinalMeanPosition": donor_mean,
            "graftDonorMeanPositionShift": float(donor_mean - initial_mean) if np.isfinite(donor_mean) and np.isfinite(initial_mean) else float("nan"),
            "graftDonorFinalSpreadFraction": float(donor_spread),
            "graftDonorInitialAggregation": float(initial_donor_aggregation),
            "graftDonorFinalAggregation": float(final_donor_aggregation),
            "graftDonorAggregationDelta": float(final_donor_aggregation - initial_donor_aggregation),
            "hostPolicyFinalCount": int(final_counts.get(host_id, 0)),
            "interfaceRuleEvaluations": int(interface_summary["interfaceRuleEvaluations"]),
            "interfaceRuleAllowedSwaps": int(interface_summary["interfaceRuleAllowedSwaps"]),
            "interfaceRuleBlockedSwaps": int(interface_summary["interfaceRuleBlockedSwaps"]),
            "interfaceCrossBoundaryAttempts": int(interface_summary["interfaceCrossBoundaryAttempts"]),
            "interfaceRuleBlockReasonCountsJson": compact_json(interface_summary["interfaceRuleBlockReasonCounts"]),
            "governanceEvaluations": int(governance_summary["governanceEvaluations"]),
            "governanceInterventions": int(governance_summary["governanceInterventions"]),
            "governanceInjectedSwaps": int(governance_summary["governanceInjectedSwaps"]),
            "governanceRedirectedSwaps": int(governance_summary["governanceRedirectedSwaps"]),
            "governanceVetoedSwaps": int(governance_summary["governanceVetoedSwaps"]),
            "governanceGlobalInformationReads": int(governance_summary["governanceGlobalInformationReads"]),
            "governanceBroadControlInterventions": int(governance_summary["governanceBroadControlInterventions"]),
            "governanceReasonCountsJson": compact_json(governance_summary["governanceReasonCounts"]),
            "runtimeSeconds": float(time.perf_counter() - started),
            "claimBoundary": CLAIM_BOUNDARY,
        }
    except Exception as exc:  # pragma: no cover - artifact-level failure accounting
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": False,
            "runError": f"{type(exc).__name__}: {exc}",
            "simulationBackend": "graft_governance_event",
            "runtimeSeconds": float(time.perf_counter() - started),
            "claimBoundary": CLAIM_BOUNDARY,
        }


def _post_worker_init(panel_records: Sequence[Mapping[str, Any]], pre_records: Sequence[Mapping[str, Any]]) -> None:
    global _PANEL_LOOKUP, _PRE_LOOKUP
    _PANEL_LOOKUP = {str(row["panelPolicyId"]): dict(row) for row in panel_records}
    _PRE_LOOKUP = {str(row["preGraftStateId"]): dict(row) for row in pre_records}


_PRE_LOOKUP: dict[str, dict[str, Any]] = {}


def _run_post_worker(record: Mapping[str, Any], max_activations: int, max_swaps: int, max_comparisons: int) -> dict[str, Any]:
    return run_graft_condition(
        record,
        _PRE_LOOKUP[str(record["preGraftStateId"])],
        _PANEL_LOOKUP,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )


def run_graft_conditions(
    condition_df: pd.DataFrame,
    pre_results: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    workers: int = 1,
    max_activations: int = 3500,
    max_swaps: int = 2500,
    max_comparisons: int = 20000,
) -> pd.DataFrame:
    records = [dict(row) for row in condition_df.to_dict(orient="records")]
    pre_records = [dict(row) for row in pre_results.to_dict(orient="records")]
    panel_records = [dict(row) for row in panel_df.to_dict(orient="records")]
    if int(workers) <= 1:
        panel_lookup = {str(row["panelPolicyId"]): row for row in panel_records}
        pre_lookup = {str(row["preGraftStateId"]): row for row in pre_records}
        return pd.DataFrame(
            [
                run_graft_condition(
                    record,
                    pre_lookup[str(record["preGraftStateId"])],
                    panel_lookup,
                    max_activations=max_activations,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                )
                for record in records
            ]
        )
    with ProcessPoolExecutor(max_workers=int(workers), initializer=_post_worker_init, initargs=(panel_records, pre_records)) as pool:
        futures = [
            pool.submit(_run_post_worker, record, int(max_activations), int(max_swaps), int(max_comparisons))
            for record in records
        ]
        return pd.DataFrame([future.result() for future in futures])


def attach_matched_no_graft_deltas(scored: pd.DataFrame) -> pd.DataFrame:
    controls = scored[~scored["graftApplied"].map(bool)].copy()
    cols = [
        "conditionId",
        "finalStateHash",
        "finalValuesJson",
        "finalPanelPolicyIdsJson",
        "s07MosaicClass",
        "finalTargetQualityScore",
        "minPolicyGoalScore",
        "meanPolicyGoalScore",
        "finalInterfaceDensityS07",
        "finalAggregationS07",
        "dominanceProxyScore",
        "stopReason",
        "swapCount",
        "activationCount",
    ]
    renamed = controls[[col for col in cols if col in controls.columns]].rename(
        columns={
            "conditionId": "matchedNoGraftConditionId",
            "finalStateHash": "noGraftFinalStateHash",
            "finalValuesJson": "noGraftFinalValuesJson",
            "finalPanelPolicyIdsJson": "noGraftFinalPanelPolicyIdsJson",
            "s07MosaicClass": "noGraftMosaicClass",
            "finalTargetQualityScore": "noGraftFinalTargetQualityScore",
            "minPolicyGoalScore": "noGraftMinPolicyGoalScore",
            "meanPolicyGoalScore": "noGraftMeanPolicyGoalScore",
            "finalInterfaceDensityS07": "noGraftFinalInterfaceDensityS07",
            "finalAggregationS07": "noGraftFinalAggregationS07",
            "dominanceProxyScore": "noGraftDominanceProxyScore",
            "stopReason": "noGraftStopReason",
            "swapCount": "noGraftSwapCount",
            "activationCount": "noGraftActivationCount",
        }
    )
    out = scored.merge(renamed, on="matchedNoGraftConditionId", how="left")
    out["deltaVsNoGraftFinalTargetQualityScore"] = pd.to_numeric(out["finalTargetQualityScore"], errors="coerce") - pd.to_numeric(
        out["noGraftFinalTargetQualityScore"], errors="coerce"
    )
    out["deltaVsNoGraftMinPolicyGoalScore"] = pd.to_numeric(out["minPolicyGoalScore"], errors="coerce") - pd.to_numeric(
        out["noGraftMinPolicyGoalScore"], errors="coerce"
    )
    out["deltaVsNoGraftMeanPolicyGoalScore"] = pd.to_numeric(out["meanPolicyGoalScore"], errors="coerce") - pd.to_numeric(
        out["noGraftMeanPolicyGoalScore"], errors="coerce"
    )
    out["deltaVsNoGraftFinalInterfaceDensity"] = pd.to_numeric(out["finalInterfaceDensityS07"], errors="coerce") - pd.to_numeric(
        out["noGraftFinalInterfaceDensityS07"], errors="coerce"
    )
    out["deltaVsNoGraftFinalAggregation"] = pd.to_numeric(out["finalAggregationS07"], errors="coerce") - pd.to_numeric(
        out["noGraftFinalAggregationS07"], errors="coerce"
    )
    out["graftChangedMosaicClass"] = out["s07MosaicClass"].astype(str) != out["noGraftMosaicClass"].astype(str)
    return out


def classify_graft_outcomes(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        if not bool(row.get("graftApplied", False)):
            outcome = "matched_no_graft_control"
        elif safe_float(row.get("deltaVsNoGraftFinalTargetQualityScore"), 0.0) >= 0.05 and safe_float(row.get("graftDonorFinalAggregation"), 0.0) >= 0.60:
            outcome = "host_reorganized_by_graft"
        elif safe_float(row.get("graftDonorFinalAggregation"), 0.0) >= 0.85 and safe_float(row.get("graftDonorFinalSpreadFraction"), 1.0) <= 0.35:
            outcome = "segregated_patch"
        elif str(row.get("graftPositionName", "")) == "center_patch" and (
            safe_float(row.get("graftDonorFinalMeanPosition"), 0.5) <= 0.15
            or safe_float(row.get("graftDonorFinalMeanPosition"), 0.5) >= 0.85
        ):
            outcome = "rejected_to_edge"
        elif safe_float(row.get("graftDonorFinalAggregation"), 1.0) <= 0.45 or safe_float(row.get("graftDonorFinalSpreadFraction"), 0.0) >= 0.50:
            outcome = "absorbed_or_mixed"
        else:
            outcome = "equilibrated_mosaic"
        rows.append({**row, "graftOutcomeClass": outcome, "graftOutcomeClassifierVersion": GRAFT_VERSION})
    return pd.DataFrame(rows)


def summarize_graft_effects(scored: pd.DataFrame) -> pd.DataFrame:
    grafts = scored[scored["graftApplied"].map(bool)].copy()
    group_cols = ["governanceVariant", "interfaceRuleVariant", "goalMode", "graftSize", "graftPositionName", "preGraftTimingLabel"]
    rows: list[dict[str, Any]] = []
    for keys, group in grafts.groupby(group_cols, dropna=False, sort=False):
        payload = dict(zip(group_cols, keys, strict=True))
        rows.append(
            {
                **payload,
                "conditionCount": int(len(group)),
                "runSuccessRate": float(group["runSucceeded"].map(bool).mean()),
                "meanDeltaTargetQualityVsNoGraft": float(group["deltaVsNoGraftFinalTargetQualityScore"].mean()),
                "meanDeltaMinGoalScoreVsNoGraft": float(group["deltaVsNoGraftMinPolicyGoalScore"].mean()),
                "mosaicClassChangedRate": float(group["graftChangedMosaicClass"].map(bool).mean()),
                "meanDonorFinalAggregation": float(pd.to_numeric(group["graftDonorFinalAggregation"], errors="coerce").mean()),
                "meanDonorSpreadFraction": float(pd.to_numeric(group["graftDonorFinalSpreadFraction"], errors="coerce").mean()),
                "outcomeCountsJson": compact_json(dict(Counter(group["graftOutcomeClass"].astype(str)))),
                "broadControlLike": bool(group["broadControlLike"].map(bool).any()) if "broadControlLike" in group.columns else False,
            }
        )
    return pd.DataFrame(rows)


def summarize_graft_outcomes(scored: pd.DataFrame) -> pd.DataFrame:
    grafts = scored[scored["graftApplied"].map(bool)].copy()
    if grafts.empty:
        return pd.DataFrame()
    return (
        grafts.groupby(["graftOutcomeClass", "goalMode"], dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            meanDeltaTargetQualityVsNoGraft=("deltaVsNoGraftFinalTargetQualityScore", "mean"),
            meanDeltaMinGoalScoreVsNoGraft=("deltaVsNoGraftMinPolicyGoalScore", "mean"),
            meanDonorFinalAggregation=("graftDonorFinalAggregation", "mean"),
            meanDonorSpreadFraction=("graftDonorFinalSpreadFraction", "mean"),
        )
        .reset_index()
    )


def validation_checks(
    selected_contexts: pd.DataFrame,
    condition_df: pd.DataFrame,
    pre_results: pd.DataFrame,
    insertion_df: pd.DataFrame,
    scored: pd.DataFrame,
    governance_variants: Sequence[str],
) -> pd.DataFrame:
    grafts = scored[scored["graftApplied"].map(bool)] if "graftApplied" in scored.columns else pd.DataFrame()
    controls = scored[~scored["graftApplied"].map(bool)] if "graftApplied" in scored.columns else pd.DataFrame()
    matched_control_ids = set(controls["conditionId"].astype(str)) if not controls.empty else set()
    checks = [
        {
            "checkId": "s09_contexts_loaded",
            "success": bool(not selected_contexts.empty and selected_contexts["s10SourceS09ConditionId"].astype(str).str.len().gt(0).all()),
            "detail": f"{len(selected_contexts)} S09 no-governance contexts selected",
        },
        {
            "checkId": "condition_matrix_coverage",
            "success": bool(
                set(GRAFT_SIZES) <= set(condition_df["graftSize"].astype(int))
                and set(GRAFT_POSITIONS) <= set(condition_df["graftPositionName"].astype(str))
                and set(label for label, _cap in GRAFT_TIMINGS) <= set(condition_df["preGraftTimingLabel"].astype(str))
                and set(GRAFT_INTERFACE_VARIANTS) <= set(condition_df["interfaceRuleVariant"].astype(str))
                and set(governance_variants) <= set(condition_df["governanceVariant"].astype(str))
            ),
            "detail": f"{len(condition_df)} S10 conditions with graft size, position, timing, governance, and interface variation",
        },
        {
            "checkId": "pre_graft_states_saved",
            "success": bool(
                not pre_results.empty
                and pre_results["preGraftRunSucceeded"].map(bool).all()
                and pre_results["preGraftFinalValuesJson"].astype(str).str.len().gt(2).all()
                and pre_results["preGraftStateId"].is_unique
            ),
            "detail": f"{len(pre_results)} saved pre-graft states",
        },
        {
            "checkId": "graft_insertions_documented",
            "success": bool(
                not insertion_df.empty
                and insertion_df["graftInsertionDescriptorJson"].astype(str).str.contains("policy-identity patch").all()
                and insertion_df["graftPositionsJson"].astype(str).str.len().gt(2).all()
            ),
            "detail": f"{len(insertion_df)} graft insertion descriptors written",
        },
        {
            "checkId": "matched_no_graft_controls_included",
            "success": bool(
                not grafts.empty
                and not controls.empty
                and set(grafts["matchedNoGraftConditionId"].astype(str)) <= matched_control_ids
                and controls["s10ConditionKind"].astype(str).eq("matched_no_graft_control").all()
            ),
            "detail": f"{len(controls)} matched no-graft controls for {len(grafts)} graft rows",
        },
        {
            "checkId": "broad_organizer_flag_retained",
            "success": bool(
                scored[scored["governanceVariant"].astype(str) == "organizer_global_upper_bound"]["broadControlLike"].map(bool).all()
                and scored[scored["governanceVariant"].astype(str) == "organizer_global_upper_bound"]["usesTargetMap"].map(bool).all()
            ),
            "detail": "broad organizer upper-bound rows retain target-map/global-control flags",
        },
        {
            "checkId": "one_result_per_condition",
            "success": bool(len(scored) == len(condition_df) and scored["conditionId"].is_unique),
            "detail": f"{len(scored)} result rows for {len(condition_df)} S10 conditions",
        },
        {
            "checkId": "runs_succeeded_and_counts_conserved",
            "success": bool(
                scored["runSucceeded"].fillna(False).map(bool).all()
                and scored["valueCountsConserved"].fillna(False).map(bool).all()
                and scored["policyCountsConserved"].fillna(False).map(bool).all()
                and scored["actualRatiosMatchTarget"].fillna(False).map(bool).all()
            ),
            "detail": "all S10 runs succeeded and conserved value/policy counts",
        },
        {
            "checkId": "goal_modes_covered",
            "success": bool({"same_goal", "opposite_goal", "partially_compatible"} <= set(scored["goalMode"].astype(str))),
            "detail": f"goal modes observed: {sorted(set(scored['goalMode'].astype(str)))}",
        },
        {
            "checkId": "graft_outcomes_assigned",
            "success": bool(grafts["graftOutcomeClass"].astype(str).isin({
                "host_reorganized_by_graft",
                "segregated_patch",
                "rejected_to_edge",
                "absorbed_or_mixed",
                "equilibrated_mosaic",
            }).all()),
            "detail": f"graft outcomes: {dict(Counter(grafts['graftOutcomeClass'].astype(str))) if not grafts.empty else {}}",
        },
        {
            "checkId": "metric_boundary_caveats_retained",
            "success": bool(
                scored["metricBoundary"].astype(str).str.len().gt(0).all()
                and scored["claimBoundary"].astype(str).str.contains("Computational").all()
            ),
            "detail": "metric-boundary caveats and computational claim boundary retained",
        },
    ]
    return pd.DataFrame(checks)


def write_graft_plots(effect_summary: pd.DataFrame, outcome_summary: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not effect_summary.empty:
        plot = (
            effect_summary.groupby(["governanceVariant", "interfaceRuleVariant"], dropna=False)
            .agg(meanDelta=("meanDeltaTargetQualityVsNoGraft", "mean"))
            .reset_index()
        )
        pivot = plot.pivot_table(index="governanceVariant", columns="interfaceRuleVariant", values="meanDelta", fill_value=0.0)
        fig, ax = plt.subplots(figsize=(10, 5))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_title("E06 S10 graft target-quality delta vs matched no-graft")
        fig.colorbar(image, ax=ax, label="Mean delta")
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s10_graft_effect_heatmap.png",
            figure_dir / "e06_s10_graft_effect_heatmap.pdf",
            step_dir / "graft_effect_heatmap.png",
            step_dir / "graft_effect_heatmap.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    if not outcome_summary.empty:
        pivot = outcome_summary.pivot_table(index="graftOutcomeClass", columns="goalMode", values="conditionCount", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(len(pivot.columns))
        for outcome in pivot.index:
            values = pivot.loc[outcome].to_numpy(dtype=float)
            ax.bar(pivot.columns, values, bottom=bottom, label=outcome)
            bottom += values
        ax.set_ylabel("Graft conditions")
        ax.set_title("E06 S10 graft outcome classes")
        ax.legend(fontsize=8)
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s10_graft_outcomes.png",
            figure_dir / "e06_s10_graft_outcomes.pdf",
            step_dir / "graft_outcomes.png",
            step_dir / "graft_outcomes.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    return paths


def write_markdown_report(
    *,
    step_dir: Path,
    status: Mapping[str, Any],
    artifacts_written: Sequence[str],
    effect_summary: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Path:
    top = effect_summary.sort_values("meanDeltaTargetQualityVsNoGraft", ascending=False, kind="mergesort").head(8) if not effect_summary.empty else pd.DataFrame()
    effect_lines = "\n".join(
        f"- `{row.governanceVariant}` / `{row.interfaceRuleVariant}` / `{row.goalMode}` / size {row.graftSize} / `{row.graftPositionName}` / `{row.preGraftTimingLabel}`: "
        f"target-quality delta {row.meanDeltaTargetQualityVsNoGraft:.3f}, min-goal delta {row.meanDeltaMinGoalScoreVsNoGraft:.1f}, "
        f"mosaic changed {row.mosaicClassChangedRate:.2f}"
        for row in top.itertuples(index=False)
    ) or "- No graft effect summaries available."
    outcome_lines = "\n".join(
        f"- `{row.graftOutcomeClass}` / `{row.goalMode}`: {int(row.conditionCount)} conditions"
        for row in outcome_summary.itertuples(index=False)
    ) or "- No graft outcomes available."
    text = f"""# Research Step S10: Run graft experiments

## Completion status

Research step ID: `S10`. {status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Ran {status['conditionCount']} post-graft or matched no-graft conditions from {status['preGraftStateCount']} saved pre-graft states. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

Graft terminology is a computational analogy. S10 uses fixed-size policy-identity patches in a one-dimensional array; no biological cells, tissues, growth, or rejection mechanisms are modeled. Policy states are reset at graft insertion while the pre-graft value arrangement is preserved. Broad-organizer rows remain flagged as target-map/global-control-like upper-bound baselines.

## Lay summary

S10 saved host-only pre-graft states, inserted donor-policy patches at matched sizes and positions, and compared each graft with a no-graft control that started from the exact same pre-graft state.

## Top graft effects

{effect_lines}

## Outcome summary

{outcome_lines}

## Recommended next action

{status['recommendedNextAction']}
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s10_graft_experiments(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    panel_path: Path = DEFAULT_PANEL_PATH,
    s09_results_path: Path = DEFAULT_S09_RESULTS_PATH,
    s09_effect_summary_path: Path = DEFAULT_S09_EFFECT_SUMMARY_PATH,
    s09_audit_path: Path = DEFAULT_S09_AUDIT_PATH,
    s06_pair_summary_path: Path = DEFAULT_S06_PAIR_SUMMARY_PATH,
    winner_criteria_path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH,
    workers: int | None = None,
    max_contexts: int = 4,
    pre_max_swaps: int = 1200,
    pre_max_comparisons: int = 12000,
    post_max_activations: int = 3500,
    post_max_swaps: int = 2500,
    post_max_comparisons: int = 20000,
) -> dict[str, Any]:
    artifacts_dir = artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    workers = int(workers if workers is not None else min(8, os.cpu_count() or 1))
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e06"
    provenance_dir = artifacts_dir / "provenance"
    for path in [step_dir, results_dir, figures_dir, provenance_dir]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ready_panel = load_ready_panel(panel_path)
    s09_results = load_s09_results(s09_results_path)
    s09_effect_summary = pd.read_parquet(s09_effect_summary_path) if s09_effect_summary_path.exists() else pd.DataFrame()
    s09_audit = pd.read_parquet(s09_audit_path) if s09_audit_path.exists() else pd.DataFrame()
    governance_variants = selected_governance_variants(s09_effect_summary)
    selected_contexts = select_s10_base_contexts(s09_results, s09_effect_summary, max_contexts=max_contexts)
    condition_df, pre_df, insertion_df = build_graft_condition_matrix(selected_contexts, governance_variants=governance_variants)
    criteria = load_winner_criteria(winner_criteria_path)
    pair_summary = pd.read_parquet(s06_pair_summary_path) if s06_pair_summary_path.exists() else pd.DataFrame()

    pre_results = run_pre_graft_states(
        pre_df,
        ready_panel,
        workers=workers,
        max_swaps=pre_max_swaps,
        max_comparisons=pre_max_comparisons,
    )
    raw = run_graft_conditions(
        condition_df,
        pre_results,
        ready_panel,
        workers=workers,
        max_activations=post_max_activations,
        max_swaps=post_max_swaps,
        max_comparisons=post_max_comparisons,
    )
    goal = augment_goal_results(raw)
    scored = score_dominance_results(goal, criteria)
    classified = classify_mosaic_formations(scored, pair_summary if not pair_summary.empty else None)
    classified = attach_matched_no_graft_deltas(classified)
    classified = classify_graft_outcomes(classified)
    classified["researchStepId"] = STEP_ID
    classified["sourceResearchStepId"] = "S10"
    classified["interventionStepId"] = STEP_ID
    classified["interventionType"] = "graft_experiment"
    effect_summary = summarize_graft_effects(classified)
    outcome_summary = summarize_graft_outcomes(classified)
    validation_df = validation_checks(selected_contexts, condition_df, pre_results, insertion_df, classified, governance_variants)

    selected_csv = step_dir / "selected_s09_graft_contexts.csv"
    selected_parquet = step_dir / "selected_s09_graft_contexts.parquet"
    condition_csv = step_dir / "graft_condition_matrix.csv"
    condition_parquet = step_dir / "graft_condition_matrix.parquet"
    pre_csv = step_dir / "pre_graft_states.csv"
    pre_parquet = step_dir / "pre_graft_states.parquet"
    insertion_csv = step_dir / "graft_insertion_descriptors.csv"
    insertion_parquet = step_dir / "graft_insertion_descriptors.parquet"
    runs_csv = step_dir / "graft_runs.csv"
    runs_parquet = step_dir / "graft_runs.parquet"
    result_csv = results_dir / "e06_graft_experiments.csv"
    result_parquet = results_dir / "e06_graft_experiments.parquet"
    combined_csv = results_dir / "e06_governance_interventions.csv"
    combined_parquet = results_dir / "e06_governance_interventions.parquet"
    effect_csv = step_dir / "graft_effect_summary.csv"
    effect_parquet = step_dir / "graft_effect_summary.parquet"
    outcome_csv = step_dir / "graft_outcome_summary.csv"
    outcome_parquet = step_dir / "graft_outcome_summary.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    for path, df in [
        (selected_csv, compact_result_csv(selected_contexts)),
        (condition_csv, compact_result_csv(condition_df)),
        (pre_csv, compact_result_csv(pre_results)),
        (insertion_csv, insertion_df),
        (runs_csv, compact_result_csv(classified)),
        (result_csv, compact_result_csv(classified)),
        (effect_csv, effect_summary),
        (outcome_csv, outcome_summary),
        (validation_csv, validation_df),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (selected_parquet, selected_contexts),
        (condition_parquet, condition_df),
        (pre_parquet, pre_results),
        (insertion_parquet, insertion_df),
        (runs_parquet, classified),
        (result_parquet, classified),
        (effect_parquet, effect_summary),
        (outcome_parquet, outcome_summary),
        (validation_parquet, validation_df),
    ]:
        df.to_parquet(path, index=False)

    if combined_parquet.exists():
        previous_combined = pd.read_parquet(combined_parquet)
        previous_combined = previous_combined[previous_combined.get("interventionStepId", "").astype(str) != STEP_ID] if "interventionStepId" in previous_combined.columns else previous_combined
        combined = pd.concat([previous_combined, classified], ignore_index=True, sort=False)
    else:
        combined = classified.copy()
    compact_result_csv(combined).to_csv(combined_csv, index=False)
    combined.to_parquet(combined_parquet, index=False)

    figure_paths = write_graft_plots(effect_summary, outcome_summary, figures_dir, step_dir)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() else "failed"
    graft_rows = classified[classified["graftApplied"].map(bool)]
    outcome = "supportive" if validation_result == "passed" and not graft_rows.empty else "constraining/contradictory"
    artifacts = [
        selected_csv,
        selected_parquet,
        condition_csv,
        condition_parquet,
        pre_csv,
        pre_parquet,
        insertion_csv,
        insertion_parquet,
        runs_csv,
        runs_parquet,
        result_csv,
        result_parquet,
        combined_csv,
        combined_parquet,
        effect_csv,
        effect_parquet,
        outcome_csv,
        outcome_parquet,
        validation_csv,
        validation_parquet,
        *figure_paths,
        step_dir / "summary.md",
        status_path,
        manifest_path,
        run_manifest_path,
    ]
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_result == "passed",
        "status": "completed" if validation_result == "passed" else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifacts],
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            "Graft terminology is a computational analogy for staged policy-identity patch insertion.",
            "S10 does not model biological tissue, growth, immune rejection, or cell death.",
            "Policy internal states are reset at graft insertion; the saved pre-graft value state and policy identities are preserved.",
            "The organizer_global_upper_bound variant remains a flagged target-map/global-control-like upper-bound baseline.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then proceed to S11 cancer-like mutant experiments using S10 matched no-graft controls, "
            "graft outcome classes, and governance/interface contexts; do not start S11 until review is complete."
        ),
        "laySummary": (
            "S10 saved host-only pre-graft states, inserted fixed-size donor-policy patches, and compared each graft to a matched no-graft control "
            "under selected S09 local governance and broad-organizer settings."
        ),
        "outcomeClassification": outcome,
        "selectedContextCount": int(len(selected_contexts)),
        "preGraftStateCount": int(len(pre_results)),
        "conditionCount": int(len(condition_df)),
        "graftConditionCount": int(graft_rows.shape[0]),
        "matchedNoGraftControlCount": int((~classified["graftApplied"].map(bool)).sum()),
        "governanceVariants": list(governance_variants),
        "interfaceRuleVariants": list(GRAFT_INTERFACE_VARIANTS),
        "graftSizes": list(GRAFT_SIZES),
        "graftPositions": list(GRAFT_POSITIONS),
        "graftTimings": [{"label": label, "activationCap": int(cap)} for label, cap in GRAFT_TIMINGS],
        "broadControlRunCount": int(classified["broadControlLike"].map(bool).sum()) if "broadControlLike" in classified.columns else 0,
        "workerCount": int(workers),
        "preMaxSwaps": int(pre_max_swaps),
        "preMaxComparisons": int(pre_max_comparisons),
        "postMaxActivations": int(post_max_activations),
        "postMaxSwaps": int(post_max_swaps),
        "postMaxComparisons": int(post_max_comparisons),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
        "newDependenciesInstalled": [],
        "sourceCodeLocation": str(repo_root),
        "sourceCodeArtifactPolicy": "repository-backed source was committed to git; source files were not copied into artifacts per workspace instructions",
    }
    summary_path = write_markdown_report(
        step_dir=step_dir,
        status=status,
        artifacts_written=[str(path) for path in artifacts],
        effect_summary=effect_summary,
        outcome_summary=outcome_summary,
        validation_df=validation_df,
    )
    if summary_path not in artifacts:
        artifacts.append(summary_path)
    status["artifactsWritten"] = [str(path) for path in artifacts]
    write_json(status_path, status)

    manifest_payload = {
        "schema": "eidosoma.step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "graftVersion": GRAFT_VERSION,
        "inputArtifacts": {
            "s09Results": str(s09_results_path),
            "s09EffectSummary": str(s09_effect_summary_path),
            "s09InformationAudit": str(s09_audit_path),
            "s06PairSummary": str(s06_pair_summary_path),
            "winnerCriteria": str(winner_criteria_path),
            "panel": str(panel_path),
        },
        "s09InformationAuditRows": int(len(s09_audit)),
        "artifacts": artifact_records(artifacts),
    }
    write_json(manifest_path, manifest_payload)

    existing_manifest: dict[str, Any] = {}
    if run_manifest_path.exists():
        try:
            existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}
    research_steps = dict(existing_manifest.get("researchSteps", {}))
    research_steps[STEP_ID] = status
    run_manifest = {
        **existing_manifest,
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": "E06",
        "lastResearchStepId": STEP_ID,
        "lastStepNumber": STEP_NUMBER,
        "updatedAt": completed_at,
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
            "commit": git_value(["rev-parse", "HEAD"], repo_root),
            "dirtyStatus": git_value(["status", "--short"], repo_root),
            "remote": git_value(["remote", "-v"], repo_root),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": int(workers),
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
                if os.environ.get(key) is not None
            },
            "newDependenciesInstalled": [],
        },
        "researchSteps": research_steps,
        "artifacts": artifact_records(artifacts),
    }
    write_json(run_manifest_path, run_manifest)
    manifest_payload["artifacts"] = artifact_records(artifacts)
    write_json(manifest_path, manifest_payload)
    return {
        "status": status,
        "selectedContexts": selected_contexts,
        "conditions": condition_df,
        "preGraftStates": pre_results,
        "insertions": insertion_df,
        "results": classified,
        "effectSummary": effect_summary,
        "outcomeSummary": outcome_summary,
        "validation": validation_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E06 S10 staged graft experiments")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--s09-results-path", type=Path, default=DEFAULT_S09_RESULTS_PATH)
    parser.add_argument("--s09-effect-summary-path", type=Path, default=DEFAULT_S09_EFFECT_SUMMARY_PATH)
    parser.add_argument("--s09-audit-path", type=Path, default=DEFAULT_S09_AUDIT_PATH)
    parser.add_argument("--s06-pair-summary-path", type=Path, default=DEFAULT_S06_PAIR_SUMMARY_PATH)
    parser.add_argument("--winner-criteria-path", type=Path, default=DEFAULT_S06_WINNER_CRITERIA_PATH)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-contexts", type=int, default=4)
    parser.add_argument("--pre-max-swaps", type=int, default=1200)
    parser.add_argument("--pre-max-comparisons", type=int, default=12000)
    parser.add_argument("--post-max-activations", type=int, default=3500)
    parser.add_argument("--post-max-swaps", type=int, default=2500)
    parser.add_argument("--post-max-comparisons", type=int, default=20000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_s10_graft_experiments(
        artifacts_dir=args.artifacts_dir,
        panel_path=args.panel_path,
        s09_results_path=args.s09_results_path,
        s09_effect_summary_path=args.s09_effect_summary_path,
        s09_audit_path=args.s09_audit_path,
        s06_pair_summary_path=args.s06_pair_summary_path,
        winner_criteria_path=args.winner_criteria_path,
        workers=args.workers,
        max_contexts=args.max_contexts,
        pre_max_swaps=args.pre_max_swaps,
        pre_max_comparisons=args.pre_max_comparisons,
        post_max_activations=args.post_max_activations,
        post_max_swaps=args.post_max_swaps,
        post_max_comparisons=args.post_max_comparisons,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['conditionCount']} graft/control conditions, "
        f"{status['preGraftStateCount']} pre-graft states, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
