"""E06 S12 developmental-history experiments for computational chimeras.

Developmental-history, clone, governance, and interface terms in this module
are computational analogies over one-dimensional local-policy arrays. The S12
clone perturbations reuse S11 fixed-size lineages: clone size is conserved and
division/growth is disabled.
"""

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
from .governance import DEFAULT_S06_PAIR_SUMMARY_PATH, DEFAULT_S06_WINNER_CRITERIA_PATH, GOVERNANCE_VERSION, GovernanceConfig
from .interfaces import INTERFACE_RULE_VERSION, interface_rule_config_by_variant
from .mixtures import (
    DEFAULT_PANEL_PATH,
    compact_json,
    compact_result_csv,
    git_value,
    load_ready_panel,
    parse_json_maybe,
    simulator_backend,
    write_json,
)
from .mosaics import classify_mosaic_formations, safe_float
from .mutants import (
    CANCER_ANALOGY_CAVEAT,
    MUTANT_VERSION,
    MutantCloneConfig,
    MutantCloneEventSimulator,
    mutant_clone_config_by_variant,
)
from .panel import CLAIM_BOUNDARY, instantiate_panel_policy, json_ready


STEP_ID = "S12"
STEP_NUMBER = 12
HISTORY_VERSION = "e06_s12_developmental_history.v1"
DEFAULT_S11_RESULTS_PATH = Path("/artifacts/results/e06_mutant_clone_experiments.parquet")
DEFAULT_S11_OUTCOME_SUMMARY_PATH = Path("/artifacts/research_steps/S11/mutant_outcome_summary.parquet")
NO_MUTANT_HISTORY_PROTOCOL = "matched_no_mutant_replay"
SIMULTANEOUS_HISTORY_PROTOCOL = "simultaneous_clone_reference"
HISTORY_ANALOGY_CAVEAT = (
    "Developmental-history, clone, governance, and interface terms are computational analogies over "
    "fixed-size one-dimensional local-policy arrays; they are not biological development, cancer, or tissue experiments."
)
FIXED_SIZE_CLONE_CAVEAT = (
    "S12 reuses S11 fixed-size computational clones with division and growth disabled; clone persistence and "
    "takeover are bounded spatial/metric proxies only."
)
NO_TRUE_MEMORY_CONTEXT_CAVEAT = (
    "The S11-derived S12 contexts contain no E04 memory-policy rows, so reset/carryover comparisons are policy-state "
    "history probes rather than direct evidence for E04 memory wrappers."
)
_PANEL_LOOKUP: dict[str, dict[str, Any]] = {}


def history_protocol_table() -> pd.DataFrame:
    """Return the fixed S12 protocol set."""

    rows = [
        {
            "historyProtocolIndex": 0,
            "historyProtocol": NO_MUTANT_HISTORY_PROTOCOL,
            "historyConditionKind": "matched_no_mutant_control",
            "historyTimingLabel": "none",
            "clonePresentAtStart": False,
            "cloneIntroducedDuringRun": False,
            "preExposureActivationCap": 0,
            "cloneIntroductionActivation": -1,
            "transientPerturbationStartActivation": -1,
            "transientPerturbationEndActivation": -1,
            "memoryCarryoverMode": "not_applicable_no_clone",
            "policyStateCarryoverMode": "single_stage_replay",
            "signalFieldCarryoverMode": "single_stage_replay",
            "description": "Replay the S11 matched no-mutant control from the S10 no-graft simulation identifier.",
        },
        {
            "historyProtocolIndex": 1,
            "historyProtocol": SIMULTANEOUS_HISTORY_PROTOCOL,
            "historyConditionKind": "simultaneous_clone",
            "historyTimingLabel": "simultaneous",
            "clonePresentAtStart": True,
            "cloneIntroducedDuringRun": False,
            "preExposureActivationCap": 0,
            "cloneIntroductionActivation": 0,
            "transientPerturbationStartActivation": -1,
            "transientPerturbationEndActivation": -1,
            "memoryCarryoverMode": "single_stage_reference",
            "policyStateCarryoverMode": "single_stage_reference",
            "signalFieldCarryoverMode": "single_stage_reference",
            "description": "Exact S11 clone replay with the clone present from activation zero.",
        },
        {
            "historyProtocolIndex": 2,
            "historyProtocol": "early_staged_introduction_reset",
            "historyConditionKind": "staged_clone_introduction",
            "historyTimingLabel": "early",
            "clonePresentAtStart": False,
            "cloneIntroducedDuringRun": True,
            "preExposureActivationCap": 500,
            "cloneIntroductionActivation": 500,
            "transientPerturbationStartActivation": -1,
            "transientPerturbationEndActivation": -1,
            "memoryCarryoverMode": "state_reset_at_clone_insertion",
            "policyStateCarryoverMode": "all_policy_states_reset_at_clone_insertion",
            "signalFieldCarryoverMode": "signal_fields_reset_at_clone_insertion",
            "description": "Run host-only early pre-exposure, insert the fixed-size clone, reset policy states, then continue.",
        },
        {
            "historyProtocolIndex": 3,
            "historyProtocol": "late_staged_introduction_reset",
            "historyConditionKind": "staged_clone_introduction",
            "historyTimingLabel": "late",
            "clonePresentAtStart": False,
            "cloneIntroducedDuringRun": True,
            "preExposureActivationCap": 1500,
            "cloneIntroductionActivation": 1500,
            "transientPerturbationStartActivation": -1,
            "transientPerturbationEndActivation": -1,
            "memoryCarryoverMode": "state_reset_at_clone_insertion",
            "policyStateCarryoverMode": "all_policy_states_reset_at_clone_insertion",
            "signalFieldCarryoverMode": "signal_fields_reset_at_clone_insertion",
            "description": "Run host-only late pre-exposure, insert the fixed-size clone, reset policy states, then continue.",
        },
        {
            "historyProtocolIndex": 4,
            "historyProtocol": "early_staged_introduction_carryover",
            "historyConditionKind": "staged_clone_introduction",
            "historyTimingLabel": "early",
            "clonePresentAtStart": False,
            "cloneIntroducedDuringRun": True,
            "preExposureActivationCap": 500,
            "cloneIntroductionActivation": 500,
            "transientPerturbationStartActivation": -1,
            "transientPerturbationEndActivation": -1,
            "memoryCarryoverMode": "policy_state_carryover_within_simulator",
            "policyStateCarryoverMode": "host_policy_states_carried_clone_cells_reinitialized_as_donor",
            "signalFieldCarryoverMode": "signal_fields_carried_within_simulator",
            "description": "Run host-only early pre-exposure in the same simulator, insert the clone, and carry host state forward.",
        },
        {
            "historyProtocolIndex": 5,
            "historyProtocol": "late_staged_introduction_carryover",
            "historyConditionKind": "staged_clone_introduction",
            "historyTimingLabel": "late",
            "clonePresentAtStart": False,
            "cloneIntroducedDuringRun": True,
            "preExposureActivationCap": 1500,
            "cloneIntroductionActivation": 1500,
            "transientPerturbationStartActivation": -1,
            "transientPerturbationEndActivation": -1,
            "memoryCarryoverMode": "policy_state_carryover_within_simulator",
            "policyStateCarryoverMode": "host_policy_states_carried_clone_cells_reinitialized_as_donor",
            "signalFieldCarryoverMode": "signal_fields_carried_within_simulator",
            "description": "Run host-only late pre-exposure in the same simulator, insert the clone, and carry host state forward.",
        },
        {
            "historyProtocolIndex": 6,
            "historyProtocol": "early_transient_rule_release_reset",
            "historyConditionKind": "transient_rule_release",
            "historyTimingLabel": "early",
            "clonePresentAtStart": True,
            "cloneIntroducedDuringRun": False,
            "preExposureActivationCap": 0,
            "cloneIntroductionActivation": 0,
            "transientPerturbationStartActivation": 0,
            "transientPerturbationEndActivation": 500,
            "memoryCarryoverMode": "state_reset_after_transient",
            "policyStateCarryoverMode": "all_policy_states_reset_after_transient",
            "signalFieldCarryoverMode": "signal_fields_reset_after_transient",
            "description": "Release governance/interface rules for the first 500 activations, reset states, then restore baseline rules.",
        },
        {
            "historyProtocolIndex": 7,
            "historyProtocol": "early_transient_rule_release_carryover",
            "historyConditionKind": "transient_rule_release",
            "historyTimingLabel": "early",
            "clonePresentAtStart": True,
            "cloneIntroducedDuringRun": False,
            "preExposureActivationCap": 0,
            "cloneIntroductionActivation": 0,
            "transientPerturbationStartActivation": 0,
            "transientPerturbationEndActivation": 500,
            "memoryCarryoverMode": "policy_state_carryover_within_simulator",
            "policyStateCarryoverMode": "policy_states_carried_after_transient",
            "signalFieldCarryoverMode": "signal_fields_carried_after_transient",
            "description": "Release governance/interface rules for the first 500 activations, keep states, then restore baseline rules.",
        },
        {
            "historyProtocolIndex": 8,
            "historyProtocol": "late_transient_rule_release_carryover",
            "historyConditionKind": "transient_rule_release",
            "historyTimingLabel": "late",
            "clonePresentAtStart": True,
            "cloneIntroducedDuringRun": False,
            "preExposureActivationCap": 1500,
            "cloneIntroductionActivation": 0,
            "transientPerturbationStartActivation": 1500,
            "transientPerturbationEndActivation": 2000,
            "memoryCarryoverMode": "policy_state_carryover_within_simulator",
            "policyStateCarryoverMode": "policy_states_carried_through_late_transient",
            "signalFieldCarryoverMode": "signal_fields_carried_through_late_transient",
            "description": "Run baseline clone exposure, release rules for activations 1500-2000, then restore baseline rules.",
        },
        {
            "historyProtocolIndex": 9,
            "historyProtocol": "late_transient_rule_release_reset",
            "historyConditionKind": "transient_rule_release",
            "historyTimingLabel": "late",
            "clonePresentAtStart": True,
            "cloneIntroducedDuringRun": False,
            "preExposureActivationCap": 1500,
            "cloneIntroductionActivation": 0,
            "transientPerturbationStartActivation": 1500,
            "transientPerturbationEndActivation": 2000,
            "memoryCarryoverMode": "state_reset_after_transient",
            "policyStateCarryoverMode": "all_policy_states_reset_after_late_transient",
            "signalFieldCarryoverMode": "signal_fields_reset_after_late_transient",
            "description": "Run baseline clone exposure, release rules for activations 1500-2000, reset states, then restore baseline rules.",
        },
    ]
    return pd.DataFrame(rows)


def _sha_short(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: int(length)]


def load_s11_results(path: Path = DEFAULT_S11_RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"S11 mutant clone results missing: {path}")
    df = pd.read_parquet(path)
    required = {
        "conditionId",
        "s11ConditionKind",
        "matchedNoMutantConditionId",
        "mutantOutcomeClass",
        "mutantBehaviorVariant",
        "mutantCloneSize",
        "mutantClonePositionName",
        "hostPolicyId",
        "donorPolicyId",
        "governanceVariant",
        "interfaceRuleVariant",
        "graftOutcomeClass",
        "initialValuesJson",
        "initialPanelPolicyIdsJson",
        "mutantLineageCellIdsJson",
        "mutantInitialPositionsJson",
        "mutantCloneConfigJson",
        "policyIdsJson",
        "goalMode",
        "finalStateHash",
        "finalValuesJson",
        "finalPanelPolicyIdsJson",
        "noMutantFinalStateHash",
        "schedulerSeed",
        "tieBreakerSeed",
        "n",
        "runSucceeded",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"S11 results missing required columns: {missing}")
    if df.empty:
        raise ValueError("S11 results are empty")
    return df


def _memory_eligible(row: Mapping[str, Any]) -> bool:
    text = " ".join(str(row.get(key, "")) for key in ["hostPolicyId", "donorPolicyId", "policyIdsJson", "sourceSimulationBackend", "simulationBackend"])
    return "memory" in text.lower() or "e04_memory" in text.lower()


def select_s12_contexts(s11_results: pd.DataFrame, *, max_contexts: int = 12) -> pd.DataFrame:
    """Select representative S11 clone outcomes for S12 history protocols."""

    clones = s11_results[
        (s11_results["s11ConditionKind"].astype(str) == "mutant_clone")
        & s11_results["runSucceeded"].fillna(False).map(bool)
        & s11_results["matchedNoMutantConditionId"].astype(str).str.len().gt(0)
    ].copy()
    if clones.empty:
        raise ValueError("S12 requires successful S11 mutant_clone rows with matched no-mutant controls")
    clones["s12PriorityScore"] = (
        2.0 * pd.to_numeric(clones.get("takeoverProxyScore", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(clones.get("containmentScore", 0.0), errors="coerce").fillna(0.0)
        + 3.0 * pd.to_numeric(clones.get("deltaVsNoMutantFinalTargetQualityScore", 0.0), errors="coerce").abs().fillna(0.0)
        + clones.get("mutantChangedMosaicClass", pd.Series(False, index=clones.index)).map(bool).astype(float)
    )
    selected_ids: list[str] = []
    reasons: dict[str, list[str]] = {}
    limit = max(1, int(max_contexts))

    def add(row: Mapping[str, Any], reason: str) -> None:
        cid = str(row["conditionId"])
        if cid not in selected_ids:
            if len(selected_ids) >= limit:
                return
            selected_ids.append(cid)
        reasons.setdefault(cid, []).append(reason)

    for outcome_class in sorted(clones["mutantOutcomeClass"].astype(str).unique()):
        rows = clones[clones["mutantOutcomeClass"].astype(str) == outcome_class].sort_values(
            ["s12PriorityScore", "conditionId"],
            ascending=[False, True],
            kind="mergesort",
        )
        if not rows.empty:
            add(rows.iloc[0], f"s11_outcome_anchor_{outcome_class}")
    for behavior in ["neutral_clone_control", "selfish_aggregation", "selfish_position_center", "selfish_disorder"]:
        rows = clones[clones["mutantBehaviorVariant"].astype(str) == behavior].sort_values(
            ["s12PriorityScore", "conditionId"],
            ascending=[False, True],
            kind="mergesort",
        )
        if not rows.empty:
            add(rows.iloc[0], f"mutant_behavior_anchor_{behavior}")
    for governance in ["no_governance_control", "conflict_resolution_range1", "local_voting_range1", "organizer_global_upper_bound"]:
        rows = clones[clones["governanceVariant"].astype(str) == governance].sort_values(
            ["s12PriorityScore", "conditionId"],
            ascending=[False, True],
            kind="mergesort",
        )
        if not rows.empty:
            add(rows.iloc[0], f"governance_anchor_{governance}")
    for interface in ["disabled_baseline", "combined_self_boundary"]:
        rows = clones[clones["interfaceRuleVariant"].astype(str) == interface].sort_values(
            ["s12PriorityScore", "conditionId"],
            ascending=[False, True],
            kind="mergesort",
        )
        if not rows.empty:
            add(rows.iloc[0], f"interface_anchor_{interface}")
    for graft_outcome in sorted(clones["graftOutcomeClass"].astype(str).unique()):
        rows = clones[clones["graftOutcomeClass"].astype(str) == graft_outcome].sort_values(
            ["s12PriorityScore", "conditionId"],
            ascending=[False, True],
            kind="mergesort",
        )
        if not rows.empty:
            add(rows.iloc[0], f"s10_graft_outcome_anchor_{graft_outcome}")
    remaining = clones.sort_values(["s12PriorityScore", "conditionId"], ascending=[False, True], kind="mergesort")
    for row in remaining.to_dict(orient="records"):
        if len(selected_ids) >= limit:
            break
        add(row, "high_s11_history_sensitivity_candidate")

    selected = clones[clones["conditionId"].astype(str).isin(selected_ids)].copy()
    selected["_order"] = selected["conditionId"].map({cid: index for index, cid in enumerate(selected_ids)})
    selected = selected.sort_values("_order", kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)
    selected.insert(0, "s12ContextIndex", np.arange(len(selected), dtype=int))
    selected["s12SourceS11ConditionId"] = selected["conditionId"].astype(str)
    selected["s12MatchedNoMutantConditionId"] = selected["matchedNoMutantConditionId"].astype(str)
    selected["s12SelectionReason"] = selected["conditionId"].map(lambda value: "+".join(sorted(set(reasons.get(str(value), [])))))
    selected["memoryEligiblePolicyPresent"] = selected.apply(_memory_eligible, axis=1)
    return selected


def _label_assignment(policy_ids: Sequence[str], assigned_ids: Sequence[str]) -> list[int]:
    lookup = {str(pid): index for index, pid in enumerate(policy_ids)}
    return [int(lookup[str(pid)]) for pid in assigned_ids]


def _counts_json(values: Sequence[str]) -> str:
    return compact_json(dict(Counter(str(value) for value in values)))


def _protocol_timeline(protocol: Mapping[str, Any], baseline_governance: str, baseline_interface: str) -> list[dict[str, Any]]:
    name = str(protocol["historyProtocol"])
    if name in {NO_MUTANT_HISTORY_PROTOCOL, SIMULTANEOUS_HISTORY_PROTOCOL}:
        return [
            {
                "stageIndex": 0,
                "stageName": "single_stage_replay",
                "startActivation": 0,
                "endActivation": "history_total_activation_cap",
                "clonePresent": name == SIMULTANEOUS_HISTORY_PROTOCOL,
                "event": "replay",
                "governanceVariant": baseline_governance,
                "interfaceRuleVariant": baseline_interface,
            }
        ]
    if "staged_introduction" in name:
        pre = int(protocol["preExposureActivationCap"])
        return [
            {
                "stageIndex": 0,
                "stageName": "host_pre_exposure",
                "startActivation": 0,
                "endActivation": pre,
                "clonePresent": False,
                "event": "host_only_pre_exposure",
                "governanceVariant": baseline_governance,
                "interfaceRuleVariant": baseline_interface,
            },
            {
                "stageIndex": 1,
                "stageName": "post_clone_history",
                "startActivation": pre,
                "endActivation": "history_total_activation_cap",
                "clonePresent": True,
                "event": "fixed_size_clone_inserted",
                "governanceVariant": baseline_governance,
                "interfaceRuleVariant": baseline_interface,
            },
        ]
    if "transient_rule_release" in name:
        start = int(protocol["transientPerturbationStartActivation"])
        end = int(protocol["transientPerturbationEndActivation"])
        if start == 0:
            return [
                {
                    "stageIndex": 0,
                    "stageName": "early_rule_release",
                    "startActivation": 0,
                    "endActivation": end,
                    "clonePresent": True,
                    "event": "governance_and_interface_temporarily_disabled",
                    "governanceVariant": "no_governance_control",
                    "interfaceRuleVariant": "disabled_baseline",
                },
                {
                    "stageIndex": 1,
                    "stageName": "restored_baseline_rules",
                    "startActivation": end,
                    "endActivation": "history_total_activation_cap",
                    "clonePresent": True,
                    "event": "baseline_rules_restored",
                    "governanceVariant": baseline_governance,
                    "interfaceRuleVariant": baseline_interface,
                },
            ]
        return [
            {
                "stageIndex": 0,
                "stageName": "baseline_pre_transient",
                "startActivation": 0,
                "endActivation": start,
                "clonePresent": True,
                "event": "baseline_pre_exposure",
                "governanceVariant": baseline_governance,
                "interfaceRuleVariant": baseline_interface,
            },
            {
                "stageIndex": 1,
                "stageName": "late_rule_release",
                "startActivation": start,
                "endActivation": end,
                "clonePresent": True,
                "event": "governance_and_interface_temporarily_disabled",
                "governanceVariant": "no_governance_control",
                "interfaceRuleVariant": "disabled_baseline",
            },
            {
                "stageIndex": 2,
                "stageName": "restored_baseline_rules",
                "startActivation": end,
                "endActivation": "history_total_activation_cap",
                "clonePresent": True,
                "event": "baseline_rules_restored",
                "governanceVariant": baseline_governance,
                "interfaceRuleVariant": baseline_interface,
            },
        ]
    raise ValueError(f"unknown S12 history protocol: {name}")


def build_developmental_history_condition_matrix(
    selected_contexts: pd.DataFrame,
    *,
    protocols: pd.DataFrame | None = None,
    total_activation_cap: int = 3500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocols = history_protocol_table() if protocols is None else protocols.copy()
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for context in selected_contexts.to_dict(orient="records"):
        n = int(context["n"])
        host = str(context["hostPolicyId"])
        donor = str(context["donorPolicyId"])
        policy_ids = [str(item) for item in parse_json_maybe(context["policyIdsJson"], [host, donor])]
        if host not in policy_ids:
            policy_ids.insert(0, host)
        if donor not in policy_ids:
            policy_ids.append(donor)
        base_values = [int(value) for value in parse_json_maybe(context["initialValuesJson"], [])]
        clone_assignment = [str(item) for item in parse_json_maybe(context["initialPanelPolicyIdsJson"], [])]
        clone_positions = [int(value) for value in parse_json_maybe(context["mutantInitialPositionsJson"], [])]
        if not clone_positions:
            clone_positions = [int(value) for value in parse_json_maybe(context.get("mutantLineageCellIdsJson", "[]"), [])]
        if len(base_values) != n or len(clone_assignment) != n:
            raise ValueError("S11 context values or assignment length does not match n")
        host_assignment = [host] * n
        clone_size = int(context["mutantCloneSize"])
        if clone_size > 0 and len(clone_positions) != clone_size:
            raise ValueError("S11 mutant clone size does not match initial clone positions")
        goal_mode = str(context["goalMode"])
        specs = policy_goal_specs(policy_ids, goal_mode)
        goal_meta = condition_goal_metadata(policy_ids, goal_mode)
        baseline_governance = str(context["governanceVariant"])
        baseline_interface = str(context["interfaceRuleVariant"])
        for protocol in protocols.to_dict(orient="records"):
            name = str(protocol["historyProtocol"])
            has_clone = name != NO_MUTANT_HISTORY_PROTOCOL
            clone_at_start = bool(protocol["clonePresentAtStart"]) and has_clone
            initial_assignment = clone_assignment if clone_at_start else host_assignment
            post_history_assignment = host_assignment if not has_clone else clone_assignment
            reverse = reverse_directions_for_assignment(initial_assignment, specs)
            post_reverse = reverse_directions_for_assignment(post_history_assignment, specs)
            digest = _sha_short(f"{context['conditionId']}|{name}|{total_activation_cap}")
            condition_id = f"s12_hist_{digest}"
            if name == NO_MUTANT_HISTORY_PROTOCOL:
                simulation_condition_id = str(context.get("s11MatchedNoGraftConditionId", context.get("matchedNoGraftConditionId", condition_id)))
                mutant_config_json = compact_json(
                    {
                        "schema": MUTANT_VERSION,
                        "variant": "matched_no_mutant_control",
                        "enabled": False,
                        "divisionEnabled": False,
                        "analogyCaveat": CANCER_ANALOGY_CAVEAT,
                        "claimBoundary": CLAIM_BOUNDARY,
                    }
                )
                mutant_ids: list[int] = []
            elif name == SIMULTANEOUS_HISTORY_PROTOCOL:
                simulation_condition_id = str(context["conditionId"])
                mutant_config_json = str(context["mutantCloneConfigJson"])
                mutant_ids = clone_positions
            else:
                simulation_condition_id = condition_id
                mutant_config_json = str(context["mutantCloneConfigJson"])
                mutant_ids = clone_positions if clone_at_start else []
            timeline = _protocol_timeline(protocol, baseline_governance, baseline_interface)
            for item in timeline:
                planned_end = item["endActivation"]
                planned_end_label = str(planned_end)
                planned_end_value = int(total_activation_cap) if isinstance(planned_end, str) else int(planned_end)
                event_rows.append(
                    {
                        "conditionId": condition_id,
                        "s12SourceS11ConditionId": str(context["conditionId"]),
                        "historyProtocol": name,
                        "historyProtocolIndex": int(protocol["historyProtocolIndex"]),
                        "plannedStageIndex": int(item["stageIndex"]),
                        "plannedStageName": str(item["stageName"]),
                        "plannedStartActivation": int(item["startActivation"]),
                        "plannedEndActivation": planned_end_value,
                        "plannedEndActivationLabel": planned_end_label,
                        "plannedEvent": str(item["event"]),
                        "plannedGovernanceVariant": str(item["governanceVariant"]),
                        "plannedInterfaceRuleVariant": str(item["interfaceRuleVariant"]),
                        "plannedClonePresent": bool(item["clonePresent"]),
                        "historyVersion": HISTORY_VERSION,
                    }
                )
            descriptor = {
                "schema": HISTORY_VERSION,
                "sourceS11ConditionId": str(context["conditionId"]),
                "matchedNoMutantConditionId": str(context["matchedNoMutantConditionId"]),
                "historyProtocol": name,
                "timingLabel": str(protocol["historyTimingLabel"]),
                "memoryCarryoverMode": str(protocol["memoryCarryoverMode"]),
                "plannedTimeline": timeline,
                "totalActivationCap": int(total_activation_cap),
                "fixedSizeCloneCaveat": FIXED_SIZE_CLONE_CAVEAT,
                "claimBoundary": CLAIM_BOUNDARY,
            }
            row = {
                **{key: json_ready(value) for key, value in context.items()},
                "conditionId": condition_id,
                "researchStepId": STEP_ID,
                "sourceResearchStepId": "S11",
                "implementationPrefix": "e06_s12",
                "historyVersion": HISTORY_VERSION,
                "s12ContextIndex": int(context["s12ContextIndex"]),
                "s12SourceS11ConditionId": str(context["conditionId"]),
                "s12MatchedNoMutantConditionId": str(context["matchedNoMutantConditionId"]),
                "s12SelectionReason": str(context["s12SelectionReason"]),
                "s11MutantOutcomeClass": str(context["mutantOutcomeClass"]),
                "s11ReferenceFinalStateHash": str(context["finalStateHash"]),
                "s11ReferenceFinalValuesJson": str(context["finalValuesJson"]),
                "s11ReferenceFinalPanelPolicyIdsJson": str(context["finalPanelPolicyIdsJson"]),
                "s11NoMutantFinalStateHash": str(context["noMutantFinalStateHash"]),
                "s11NoMutantFinalValuesJson": str(context.get("noMutantFinalValuesJson", "")),
                "s11NoMutantFinalPanelPolicyIdsJson": str(context.get("noMutantFinalPanelPolicyIdsJson", "")),
                "simulationConditionId": simulation_condition_id,
                "historyProtocol": name,
                "historyProtocolIndex": int(protocol["historyProtocolIndex"]),
                "historyConditionKind": str(protocol["historyConditionKind"]),
                "historyTimingLabel": str(protocol["historyTimingLabel"]),
                "clonePresentAtStart": clone_at_start,
                "cloneIntroducedDuringRun": bool(protocol["cloneIntroducedDuringRun"]) and has_clone,
                "preExposureActivationCap": int(protocol["preExposureActivationCap"]),
                "cloneIntroductionActivation": int(protocol["cloneIntroductionActivation"]) if has_clone else -1,
                "transientPerturbationStartActivation": int(protocol["transientPerturbationStartActivation"]),
                "transientPerturbationEndActivation": int(protocol["transientPerturbationEndActivation"]),
                "memoryCarryoverMode": str(protocol["memoryCarryoverMode"]),
                "policyStateCarryoverMode": str(protocol["policyStateCarryoverMode"]),
                "signalFieldCarryoverMode": str(protocol["signalFieldCarryoverMode"]),
                "memoryEligiblePolicyPresent": bool(context.get("memoryEligiblePolicyPresent", False)),
                "historyTotalActivationCap": int(total_activation_cap),
                "historyPlannedTimelineJson": compact_json(timeline),
                "historyDescriptorJson": compact_json(descriptor),
                "policyIdsJson": compact_json(policy_ids),
                "initialValuesJson": compact_json(base_values),
                "initialPanelPolicyIdsJson": compact_json(initial_assignment),
                "initialNumericLabelsJson": compact_json(_label_assignment(policy_ids, initial_assignment)),
                "postHistoryTargetPanelPolicyIdsJson": compact_json(post_history_assignment),
                "postHistoryTargetCountsJson": _counts_json(post_history_assignment),
                "postHistoryTargetProportionsJson": compact_json(
                    {pid: Counter(post_history_assignment).get(pid, 0) / n for pid in policy_ids}
                ),
                "targetCountsJson": _counts_json(post_history_assignment),
                "targetProportionsJson": compact_json({pid: Counter(post_history_assignment).get(pid, 0) / n for pid in policy_ids}),
                "goalCompatibilityClass": str(goal_meta["compatibilityClass"]),
                "conditionGoalMetadataJson": compact_json(goal_meta),
                "policyGoalMetadataJson": compact_json(specs),
                "goalReverseDirectionsJson": compact_json(reverse),
                "postHistoryGoalReverseDirectionsJson": compact_json(post_reverse),
                "goalDirectionByPolicyJson": compact_json({pid: specs[pid]["targetDirection"] for pid in policy_ids}),
                "goalBehaviorDirectionByPolicyJson": compact_json({pid: specs[pid]["behaviorDirection"] for pid in policy_ids}),
                "governanceVersion": GOVERNANCE_VERSION,
                "baselineGovernanceVariant": baseline_governance,
                "governanceVariant": baseline_governance,
                "governanceConfigJson": compact_json(GovernanceConfig.from_payload(baseline_governance).to_dict()),
                "interfaceRuleVersion": INTERFACE_RULE_VERSION,
                "baselineInterfaceRuleVariant": baseline_interface,
                "interfaceRuleVariant": baseline_interface,
                "interfaceRuleConfigJson": compact_json(interface_rule_config_by_variant(baseline_interface).to_dict()),
                "mutantVersion": MUTANT_VERSION,
                "mutantCloneConfigJson": mutant_config_json,
                "mutantLineageCellIdsJson": compact_json(mutant_ids),
                "mutantInitialPositionsJson": compact_json(clone_positions if has_clone else []),
                "mutantCloneSize": clone_size if has_clone else 0,
                "divisionEnabled": False,
                "lineageTracking": "cell_id_fixed_size_lineage_position_remapped_after_reset",
                "conditionHash": hashlib.sha256(compact_json(descriptor).encode("utf-8")).hexdigest()[:16],
                "analogyCaveat": HISTORY_ANALOGY_CAVEAT,
                "fixedSizeCloneCaveat": FIXED_SIZE_CLONE_CAVEAT,
                "claimBoundary": CLAIM_BOUNDARY,
            }
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True), pd.DataFrame(event_rows).reset_index(drop=True)


def _clone_positions_from_cells(cells: Sequence[Any], clone_ids: set[int]) -> list[int]:
    return [index for index, cell in enumerate(cells) if int(cell.cell_id) in clone_ids]


def _clone_boundary_count(n: int, clone_positions: Sequence[int]) -> int:
    clone_set = set(int(pos) for pos in clone_positions)
    return int(sum(1 for left in range(max(0, int(n) - 1)) if (left in clone_set) != ((left + 1) in clone_set)))


def _lineage_aggregation(n: int, positions: Sequence[int]) -> float:
    labels = ["clone" if index in set(int(pos) for pos in positions) else "host" for index in range(int(n))]
    return float(aggregation(labels))


def _policy_ids_from_labels(policy_ids: Sequence[str], cells: Sequence[Any]) -> list[str]:
    return [str(policy_ids[int(cell.label)]) for cell in cells]


def _make_simulator(
    *,
    values: Sequence[int],
    assigned_ids: Sequence[str],
    policy_ids: Sequence[str],
    reverse_directions: Sequence[bool],
    panel_lookup: Mapping[str, Mapping[str, Any]],
    condition_id: str,
    governance_variant: str,
    interface_variant: str,
    mutant_config: MutantCloneConfig,
    mutant_cell_ids: Sequence[int],
    scheduler_seed: int,
    tie_breaker_seed: int,
) -> tuple[MutantCloneEventSimulator, dict[str, Any], str]:
    label_by_id = {str(pid): index for index, pid in enumerate(policy_ids)}
    numeric_labels = [int(label_by_id[str(pid)]) for pid in assigned_ids]
    policy_cache = {str(pid): instantiate_panel_policy(panel_lookup[str(pid)]) for pid in policy_ids}
    policies = [policy_cache[str(pid)] for pid in assigned_ids]
    records = [dict(panel_lookup[str(pid)]) for pid in policy_ids]
    source_backend = simulator_backend(records)
    first_signal = next((policy for policy in policy_cache.values() if hasattr(policy, "signal_config")), None)
    signal_config = getattr(first_signal, "signal_config", "no_signal")
    simulator = MutantCloneEventSimulator(
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
        mutant_clone_config=mutant_config,
        mutant_cell_ids=mutant_cell_ids,
        implementation="e06_s12_developmental_history",
    )
    return simulator, policy_cache, source_backend


def _set_rule_configs(simulator: MutantCloneEventSimulator, governance_variant: str, interface_variant: str) -> None:
    simulator.governance_config = GovernanceConfig.from_payload(governance_variant)
    simulator.interface_rule_config = interface_rule_config_by_variant(interface_variant)
    if hasattr(simulator, "_select_leader_cell_ids"):
        simulator._leader_cell_ids = simulator._select_leader_cell_ids()
    if hasattr(simulator, "_select_pacemaker_cell_ids"):
        simulator._pacemaker_cell_ids = simulator._select_pacemaker_cell_ids()


def _insert_clone_carryover(
    simulator: MutantCloneEventSimulator,
    *,
    positions: Sequence[int],
    donor_policy_id: str,
    policy_ids: Sequence[str],
    policy_cache: Mapping[str, Any],
    goal_specs: Mapping[str, Mapping[str, Any]],
) -> list[int]:
    label_by_id = {str(pid): index for index, pid in enumerate(policy_ids)}
    clone_ids: list[int] = []
    n = len(simulator.cells)
    for pos in [int(value) for value in positions]:
        cell = simulator.cells[pos]
        donor_policy = policy_cache[str(donor_policy_id)]
        cell.policy = donor_policy
        cell.label = int(label_by_id[str(donor_policy_id)])
        cell.reverse_direction = str(goal_specs[str(donor_policy_id)]["behaviorDirection"]) == "decreasing"
        cell.state = donor_policy.initial_state(
            cell_id=int(cell.cell_id),
            position=int(pos),
            value=int(cell.value),
            n=n,
            reverse_direction=bool(cell.reverse_direction),
        )
        clone_ids.append(int(cell.cell_id))
    for cell in simulator.cells:
        pid = str(policy_ids[int(cell.label)])
        cell.reverse_direction = str(goal_specs[pid]["behaviorDirection"]) == "decreasing"
    simulator.mutant_cell_ids = set(clone_ids)
    return clone_ids


def _snapshot_assignment(simulator: MutantCloneEventSimulator, policy_ids: Sequence[str]) -> list[str]:
    return _policy_ids_from_labels(policy_ids, simulator.cells)


def _summaries(simulator: MutantCloneEventSimulator) -> dict[str, int]:
    interface = simulator.interface_summary()
    governance = simulator.governance_summary()
    mutant = simulator.mutant_summary()
    return {
        "swapCount": int(simulator.swap_count),
        "comparisonCount": int(simulator.comparison_count),
        "activationCount": int(simulator.activation_count),
        "eventCount": int(len(simulator.trace_rows)),
        "blockedMoveAttempts": int(simulator.blocked_move_attempts),
        "frozenSwapAttempts": int(simulator.frozen_swap_attempts),
        "interfaceRuleEvaluations": int(interface["interfaceRuleEvaluations"]),
        "interfaceRuleAllowedSwaps": int(interface["interfaceRuleAllowedSwaps"]),
        "interfaceRuleBlockedSwaps": int(interface["interfaceRuleBlockedSwaps"]),
        "interfaceCrossBoundaryAttempts": int(interface["interfaceCrossBoundaryAttempts"]),
        "governanceEvaluations": int(governance["governanceEvaluations"]),
        "governanceInterventions": int(governance["governanceInterventions"]),
        "governanceInjectedSwaps": int(governance["governanceInjectedSwaps"]),
        "governanceRedirectedSwaps": int(governance["governanceRedirectedSwaps"]),
        "governanceVetoedSwaps": int(governance["governanceVetoedSwaps"]),
        "governanceGlobalInformationReads": int(governance["governanceGlobalInformationReads"]),
        "governanceBroadControlInterventions": int(governance["governanceBroadControlInterventions"]),
        "mutantEvaluations": int(mutant["mutantEvaluations"]),
        "mutantSelfishProposals": int(mutant["mutantSelfishProposals"]),
        "mutantInjectedSwaps": int(mutant["mutantInjectedSwaps"]),
        "mutantRedirectedSwaps": int(mutant["mutantRedirectedSwaps"]),
    }


def _delta(after: Mapping[str, int], before: Mapping[str, int], key: str) -> int:
    return int(after.get(key, 0) - before.get(key, 0))


def _run_stage(
    simulator: MutantCloneEventSimulator,
    *,
    stage_index: int,
    stage_name: str,
    stage_event: str,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
    governance_variant: str,
    interface_variant: str,
) -> dict[str, Any]:
    _set_rule_configs(simulator, governance_variant, interface_variant)
    before = _summaries(simulator)
    start_activation = int(simulator.activation_count)
    result = simulator.run(
        max_activations=int(max_activations),
        max_swaps=int(max_swaps),
        max_comparisons=int(max_comparisons),
        no_move_check_interval=len(simulator.cells),
    )
    after = _summaries(simulator)
    clone_positions = _clone_positions_from_cells(simulator.cells, set(simulator.mutant_cell_ids))
    return {
        "stageIndex": int(stage_index),
        "stageName": stage_name,
        "stageEvent": stage_event,
        "governanceVariant": governance_variant,
        "interfaceRuleVariant": interface_variant,
        "startActivation": int(start_activation),
        "endActivation": int(simulator.activation_count),
        "maxActivationCap": int(max_activations),
        "stopReason": str(result.stop_reason),
        "swapCountDelta": _delta(after, before, "swapCount"),
        "comparisonCountDelta": _delta(after, before, "comparisonCount"),
        "activationCountDelta": _delta(after, before, "activationCount"),
        "eventCountDelta": _delta(after, before, "eventCount"),
        "blockedMoveAttemptsDelta": _delta(after, before, "blockedMoveAttempts"),
        "interfaceRuleEvaluationsDelta": _delta(after, before, "interfaceRuleEvaluations"),
        "governanceInterventionsDelta": _delta(after, before, "governanceInterventions"),
        "mutantSelfishProposalsDelta": _delta(after, before, "mutantSelfishProposals"),
        "cloneCellIdsJson": compact_json(sorted(int(value) for value in simulator.mutant_cell_ids)),
        "clonePositionsJson": compact_json(clone_positions),
        "valueStateHash": state_hash(simulator.current_values()),
    }


def _accumulate(stage_records: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(sum(int(record.get(f"{key}Delta", 0)) for record in stage_records))


def _make_stage_reset_simulator(
    simulator: MutantCloneEventSimulator,
    *,
    policy_ids: Sequence[str],
    panel_lookup: Mapping[str, Mapping[str, Any]],
    condition_id: str,
    governance_variant: str,
    interface_variant: str,
    mutant_config: MutantCloneConfig,
    mutant_positions: Sequence[int],
    scheduler_seed: int,
    tie_breaker_seed: int,
    goal_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[MutantCloneEventSimulator, dict[str, Any], str]:
    values = simulator.current_values()
    assigned = _snapshot_assignment(simulator, policy_ids)
    clone_set = set(int(pos) for pos in mutant_positions)
    for pos in clone_set:
        assigned[int(pos)] = str(policy_ids[int(simulator.cells[int(pos)].label)])
    reverse = reverse_directions_for_assignment(assigned, goal_specs)
    new_sim, cache, backend = _make_simulator(
        values=values,
        assigned_ids=assigned,
        policy_ids=policy_ids,
        reverse_directions=reverse,
        panel_lookup=panel_lookup,
        condition_id=condition_id,
        governance_variant=governance_variant,
        interface_variant=interface_variant,
        mutant_config=mutant_config,
        mutant_cell_ids=sorted(clone_set),
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_breaker_seed,
    )
    return new_sim, cache, backend


def _final_result_payload(
    condition: Mapping[str, Any],
    simulator: MutantCloneEventSimulator,
    *,
    policy_ids: Sequence[str],
    initial_values: Sequence[int],
    initial_assignment: Sequence[str],
    stage_records: Sequence[Mapping[str, Any]],
    source_backend: str,
    started: float,
) -> dict[str, Any]:
    final_values = simulator.current_values()
    final_ids = _snapshot_assignment(simulator, policy_ids)
    expected_assignment = [str(item) for item in parse_json_maybe(condition["postHistoryTargetPanelPolicyIdsJson"], [])]
    initial_counts = Counter(str(value) for value in initial_assignment)
    final_counts = Counter(str(value) for value in final_ids)
    expected_counts = Counter(str(value) for value in expected_assignment)
    clone_ids = set(int(value) for value in simulator.mutant_cell_ids)
    clone_positions = _clone_positions_from_cells(simulator.cells, clone_ids)
    initial_clone_positions = [int(pos) for pos in parse_json_maybe(condition.get("mutantInitialPositionsJson", "[]"), [])]
    clone_initial_count = int(condition.get("mutantCloneSize", 0) or 0)
    clone_final_count = len(clone_positions)
    clone_spread = (
        (max(clone_positions) - min(clone_positions) + 1) / max(1, len(final_ids))
        if clone_positions
        else 0.0
    )
    clone_mean = float(np.mean(clone_positions) / max(1, len(final_ids) - 1)) if clone_positions else float("nan")
    clone_initial_mean = float(np.mean(initial_clone_positions) / max(1, len(final_ids) - 1)) if initial_clone_positions else float("nan")
    clone_boundary = _clone_boundary_count(len(final_ids), clone_positions)
    clone_final_agg = _lineage_aggregation(len(final_ids), clone_positions) if clone_final_count else float("nan")
    clone_initial_agg = _lineage_aggregation(len(final_ids), initial_clone_positions) if clone_initial_count else float("nan")
    completed = sortedness_percent(final_values) >= 100.0 - 1e-9
    final_stage_stop = str(stage_records[-1]["stopReason"]) if stage_records else "not_run"
    interface = simulator.interface_summary()
    governance = simulator.governance_summary()
    mutant = simulator.mutant_summary()
    return {
        **{key: json_ready(value) for key, value in condition.items()},
        "runSucceeded": True,
        "runError": "",
        "simulationBackend": "developmental_history_mutant_clone_governance_event",
        "sourceSimulationBackend": source_backend,
        "historyActualTimelineJson": compact_json(list(stage_records)),
        "historyActualStageCount": int(len(stage_records)),
        "historyEventTimingStored": True,
        "finalValuesJson": compact_json(final_values),
        "initialStateHash": state_hash(initial_values),
        "finalStateHash": state_hash(final_values),
        "finalPanelPolicyIdsJson": compact_json(final_ids),
        "actualCountsJson": compact_json(dict(initial_counts)),
        "finalCountsJson": compact_json(dict(final_counts)),
        "expectedFinalCountsJson": compact_json(dict(expected_counts)),
        "actualRatiosMatchTarget": final_counts == expected_counts,
        "valueCountsConserved": Counter(int(value) for value in initial_values) == Counter(final_values),
        "policyCountsConserved": final_counts == expected_counts,
        "completed": bool(completed),
        "stopReason": final_stage_stop,
        "finalStateClass": "sorted" if completed else ("resource_capped_partial" if final_stage_stop.startswith("max_") else "partial"),
        "initialSortednessPercent": sortedness_percent(initial_values),
        "finalSortednessPercent": sortedness_percent(final_values),
        "sortednessGain": sortedness_percent(final_values) - sortedness_percent(initial_values),
        "initialAggregation": aggregation(initial_assignment),
        "finalAggregation": aggregation(final_ids),
        "aggregationDelta": aggregation(final_ids) - aggregation(initial_assignment),
        "swapCount": _accumulate(stage_records, "swapCount"),
        "comparisonCount": _accumulate(stage_records, "comparisonCount"),
        "activationCount": _accumulate(stage_records, "activationCount"),
        "eventCount": _accumulate(stage_records, "eventCount"),
        "blockedMoveAttempts": _accumulate(stage_records, "blockedMoveAttempts"),
        "frozenSwapAttempts": int(simulator.frozen_swap_attempts),
        "mutantCloneInitialCount": clone_initial_count,
        "mutantCloneFinalCount": clone_final_count,
        "mutantCloneSizeConserved": bool(clone_initial_count == clone_final_count),
        "mutantCloneFinalPositionsJson": compact_json(clone_positions),
        "mutantCloneInitialMeanPosition": clone_initial_mean,
        "mutantCloneFinalMeanPosition": clone_mean,
        "mutantCloneMeanPositionShift": float(clone_mean - clone_initial_mean) if np.isfinite(clone_mean) and np.isfinite(clone_initial_mean) else float("nan"),
        "mutantCloneInitialAggregation": float(clone_initial_agg),
        "mutantCloneFinalAggregation": float(clone_final_agg),
        "mutantCloneAggregationDelta": float(clone_final_agg - clone_initial_agg)
        if np.isfinite(clone_final_agg) and np.isfinite(clone_initial_agg)
        else float("nan"),
        "mutantCloneBoundaryCount": int(clone_boundary),
        "mutantCloneInterfaceDensity": float(clone_boundary / max(1, len(final_ids) - 1)),
        "mutantCloneSpreadFraction": float(clone_spread),
        "interfaceRuleEvaluations": int(interface["interfaceRuleEvaluations"]),
        "interfaceRuleAllowedSwaps": int(interface["interfaceRuleAllowedSwaps"]),
        "interfaceRuleBlockedSwaps": int(interface["interfaceRuleBlockedSwaps"]),
        "interfaceCrossBoundaryAttempts": int(interface["interfaceCrossBoundaryAttempts"]),
        "interfaceRuleBlockReasonCountsJson": compact_json(interface["interfaceRuleBlockReasonCounts"]),
        "governanceEvaluations": int(governance["governanceEvaluations"]),
        "governanceInterventions": int(governance["governanceInterventions"]),
        "governanceInjectedSwaps": int(governance["governanceInjectedSwaps"]),
        "governanceRedirectedSwaps": int(governance["governanceRedirectedSwaps"]),
        "governanceVetoedSwaps": int(governance["governanceVetoedSwaps"]),
        "governanceGlobalInformationReads": int(governance["governanceGlobalInformationReads"]),
        "governanceBroadControlInterventions": int(governance["governanceBroadControlInterventions"]),
        "governanceReasonCountsJson": compact_json(governance["governanceReasonCounts"]),
        "mutantEvaluations": int(mutant["mutantEvaluations"]),
        "mutantSelfishProposals": int(mutant["mutantSelfishProposals"]),
        "mutantInjectedSwaps": int(mutant["mutantInjectedSwaps"]),
        "mutantRedirectedSwaps": int(mutant["mutantRedirectedSwaps"]),
        "mutantReasonCountsJson": compact_json(mutant["mutantReasonCounts"]),
        "runtimeSeconds": float(time.perf_counter() - started),
        "analogyCaveat": HISTORY_ANALOGY_CAVEAT,
        "fixedSizeCloneCaveat": FIXED_SIZE_CLONE_CAVEAT,
        "claimBoundary": CLAIM_BOUNDARY,
    }


def run_history_condition(
    condition: Mapping[str, Any],
    panel_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    max_activations: int = 3500,
    max_swaps: int = 2500,
    max_comparisons: int = 20000,
) -> dict[str, Any]:
    panel_lookup = panel_lookup or _PANEL_LOOKUP
    started = time.perf_counter()
    policy_ids = [str(item) for item in parse_json_maybe(condition["policyIdsJson"], [])]
    initial_assignment = [str(item) for item in parse_json_maybe(condition["initialPanelPolicyIdsJson"], [])]
    initial_values = [int(item) for item in parse_json_maybe(condition["initialValuesJson"], [])]
    goal_specs = parse_json_maybe(condition["policyGoalMetadataJson"], {})
    if not isinstance(goal_specs, Mapping):
        goal_specs = policy_goal_specs(policy_ids, str(condition["goalMode"]))
    reverse = [bool(item) for item in parse_json_maybe(condition["goalReverseDirectionsJson"], [])]
    mutant_config = MutantCloneConfig.from_payload(condition.get("mutantCloneConfigJson", "neutral_clone_control"))
    protocol = str(condition["historyProtocol"])
    baseline_governance = str(condition["baselineGovernanceVariant"])
    baseline_interface = str(condition["baselineInterfaceRuleVariant"])
    seed_offset = 1009
    scheduler_seed = int(condition["schedulerSeed"]) + seed_offset
    tie_seed = int(condition["tieBreakerSeed"]) + seed_offset
    stage_records: list[dict[str, Any]] = []
    source_backend = ""

    try:
        simulator, policy_cache, source_backend = _make_simulator(
            values=initial_values,
            assigned_ids=initial_assignment,
            policy_ids=policy_ids,
            reverse_directions=reverse,
            panel_lookup=panel_lookup,
            condition_id=str(condition.get("simulationConditionId", condition["conditionId"])),
            governance_variant=baseline_governance,
            interface_variant=baseline_interface,
            mutant_config=mutant_config,
            mutant_cell_ids=[int(item) for item in parse_json_maybe(condition.get("mutantLineageCellIdsJson", "[]"), [])],
            scheduler_seed=scheduler_seed,
            tie_breaker_seed=tie_seed,
        )
        if protocol in {NO_MUTANT_HISTORY_PROTOCOL, SIMULTANEOUS_HISTORY_PROTOCOL}:
            stage_records.append(
                _run_stage(
                    simulator,
                    stage_index=0,
                    stage_name="single_stage_replay",
                    stage_event="s11_reference_replay",
                    max_activations=max_activations,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                    governance_variant=baseline_governance,
                    interface_variant=baseline_interface,
                )
            )
        elif "staged_introduction" in protocol:
            pre_cap = int(condition["preExposureActivationCap"])
            stage_records.append(
                _run_stage(
                    simulator,
                    stage_index=0,
                    stage_name="host_pre_exposure",
                    stage_event="host_only_pre_exposure",
                    max_activations=pre_cap,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                    governance_variant=baseline_governance,
                    interface_variant=baseline_interface,
                )
            )
            clone_positions = [int(pos) for pos in parse_json_maybe(condition["mutantInitialPositionsJson"], [])]
            donor = str(condition["donorPolicyId"])
            if "carryover" in protocol:
                _insert_clone_carryover(
                    simulator,
                    positions=clone_positions,
                    donor_policy_id=donor,
                    policy_ids=policy_ids,
                    policy_cache=policy_cache,
                    goal_specs=goal_specs,
                )
            else:
                assigned = _snapshot_assignment(simulator, policy_ids)
                for pos in clone_positions:
                    assigned[int(pos)] = donor
                reverse_after = reverse_directions_for_assignment(assigned, goal_specs)
                simulator, policy_cache, source_backend = _make_simulator(
                    values=simulator.current_values(),
                    assigned_ids=assigned,
                    policy_ids=policy_ids,
                    reverse_directions=reverse_after,
                    panel_lookup=panel_lookup,
                    condition_id=str(condition["conditionId"]),
                    governance_variant=baseline_governance,
                    interface_variant=baseline_interface,
                    mutant_config=mutant_config,
                    mutant_cell_ids=clone_positions,
                    scheduler_seed=scheduler_seed + 2001,
                    tie_breaker_seed=tie_seed + 2001,
                )
            stage_records.append(
                _run_stage(
                    simulator,
                    stage_index=1,
                    stage_name="post_clone_history",
                    stage_event="fixed_size_clone_inserted",
                    max_activations=max_activations if "carryover" in protocol else max(0, max_activations - pre_cap),
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                    governance_variant=baseline_governance,
                    interface_variant=baseline_interface,
                )
            )
        elif "transient_rule_release" in protocol:
            if str(condition["historyTimingLabel"]) == "early":
                stage_records.append(
                    _run_stage(
                        simulator,
                        stage_index=0,
                        stage_name="early_rule_release",
                        stage_event="governance_and_interface_temporarily_disabled",
                        max_activations=int(condition["transientPerturbationEndActivation"]),
                        max_swaps=max_swaps,
                        max_comparisons=max_comparisons,
                        governance_variant="no_governance_control",
                        interface_variant="disabled_baseline",
                    )
                )
                if "reset" in protocol:
                    clone_positions = _clone_positions_from_cells(simulator.cells, set(simulator.mutant_cell_ids))
                    simulator, policy_cache, source_backend = _make_stage_reset_simulator(
                        simulator,
                        policy_ids=policy_ids,
                        panel_lookup=panel_lookup,
                        condition_id=str(condition["conditionId"]),
                        governance_variant=baseline_governance,
                        interface_variant=baseline_interface,
                        mutant_config=mutant_config,
                        mutant_positions=clone_positions,
                        scheduler_seed=scheduler_seed + 3001,
                        tie_breaker_seed=tie_seed + 3001,
                        goal_specs=goal_specs,
                    )
                stage_records.append(
                    _run_stage(
                        simulator,
                        stage_index=1,
                        stage_name="restored_baseline_rules",
                        stage_event="baseline_rules_restored",
                        max_activations=max_activations
                        if "carryover" in protocol
                        else max(0, max_activations - int(condition["transientPerturbationEndActivation"])),
                        max_swaps=max_swaps,
                        max_comparisons=max_comparisons,
                        governance_variant=baseline_governance,
                        interface_variant=baseline_interface,
                    )
                )
            else:
                start = int(condition["transientPerturbationStartActivation"])
                end = int(condition["transientPerturbationEndActivation"])
                stage_records.append(
                    _run_stage(
                        simulator,
                        stage_index=0,
                        stage_name="baseline_pre_transient",
                        stage_event="baseline_pre_exposure",
                        max_activations=start,
                        max_swaps=max_swaps,
                        max_comparisons=max_comparisons,
                        governance_variant=baseline_governance,
                        interface_variant=baseline_interface,
                    )
                )
                stage_records.append(
                    _run_stage(
                        simulator,
                        stage_index=1,
                        stage_name="late_rule_release",
                        stage_event="governance_and_interface_temporarily_disabled",
                        max_activations=end,
                        max_swaps=max_swaps,
                        max_comparisons=max_comparisons,
                        governance_variant="no_governance_control",
                        interface_variant="disabled_baseline",
                    )
                )
                if "reset" in protocol:
                    clone_positions = _clone_positions_from_cells(simulator.cells, set(simulator.mutant_cell_ids))
                    simulator, policy_cache, source_backend = _make_stage_reset_simulator(
                        simulator,
                        policy_ids=policy_ids,
                        panel_lookup=panel_lookup,
                        condition_id=str(condition["conditionId"]),
                        governance_variant=baseline_governance,
                        interface_variant=baseline_interface,
                        mutant_config=mutant_config,
                        mutant_positions=clone_positions,
                        scheduler_seed=scheduler_seed + 4001,
                        tie_breaker_seed=tie_seed + 4001,
                        goal_specs=goal_specs,
                    )
                stage_records.append(
                    _run_stage(
                        simulator,
                        stage_index=2,
                        stage_name="restored_baseline_rules",
                        stage_event="baseline_rules_restored",
                        max_activations=max_activations if "carryover" in protocol else max(0, max_activations - end),
                        max_swaps=max_swaps,
                        max_comparisons=max_comparisons,
                        governance_variant=baseline_governance,
                        interface_variant=baseline_interface,
                    )
                )
        else:
            raise ValueError(f"unknown history protocol: {protocol}")

        return _final_result_payload(
            condition,
            simulator,
            policy_ids=policy_ids,
            initial_values=initial_values,
            initial_assignment=initial_assignment,
            stage_records=stage_records,
            source_backend=source_backend,
            started=started,
        )
    except Exception as exc:  # pragma: no cover - artifact-level failure accounting
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": False,
            "runError": f"{type(exc).__name__}: {exc}",
            "simulationBackend": "developmental_history_mutant_clone_governance_event",
            "runtimeSeconds": float(time.perf_counter() - started),
            "historyActualTimelineJson": compact_json(stage_records),
            "historyEventTimingStored": bool(stage_records),
            "analogyCaveat": HISTORY_ANALOGY_CAVEAT,
            "fixedSizeCloneCaveat": FIXED_SIZE_CLONE_CAVEAT,
            "claimBoundary": CLAIM_BOUNDARY,
        }


def _worker_init(panel_records: Sequence[Mapping[str, Any]]) -> None:
    global _PANEL_LOOKUP
    _PANEL_LOOKUP = {str(row["panelPolicyId"]): dict(row) for row in panel_records}


def _run_worker(record: Mapping[str, Any], max_activations: int, max_swaps: int, max_comparisons: int) -> dict[str, Any]:
    return run_history_condition(
        record,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )


def run_history_conditions(
    condition_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    workers: int = 1,
    max_activations: int = 3500,
    max_swaps: int = 2500,
    max_comparisons: int = 20000,
) -> pd.DataFrame:
    records = [dict(row) for row in condition_df.to_dict(orient="records")]
    panel_records = [dict(row) for row in panel_df.to_dict(orient="records")]
    if int(workers) <= 1:
        lookup = {str(row["panelPolicyId"]): row for row in panel_records}
        return pd.DataFrame(
            [
                run_history_condition(
                    record,
                    lookup,
                    max_activations=max_activations,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                )
                for record in records
            ]
        )
    with ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(panel_records,)) as pool:
        futures = [pool.submit(_run_worker, record, int(max_activations), int(max_swaps), int(max_comparisons)) for record in records]
        return pd.DataFrame([future.result() for future in futures])


def timeline_rows_from_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in results.to_dict(orient="records"):
        timeline = parse_json_maybe(row.get("historyActualTimelineJson", "[]"), [])
        if not isinstance(timeline, Sequence) or isinstance(timeline, str):
            continue
        for item in timeline:
            if not isinstance(item, Mapping):
                continue
            rows.append(
                {
                    "conditionId": str(row["conditionId"]),
                    "s12SourceS11ConditionId": str(row.get("s12SourceS11ConditionId", "")),
                    "historyProtocol": str(row.get("historyProtocol", "")),
                    "historyTimingLabel": str(row.get("historyTimingLabel", "")),
                    "memoryCarryoverMode": str(row.get("memoryCarryoverMode", "")),
                    **{key: json_ready(value) for key, value in item.items()},
                    "historyVersion": HISTORY_VERSION,
                }
            )
    return pd.DataFrame(rows)


def attach_history_reference_deltas(scored: pd.DataFrame) -> pd.DataFrame:
    keys = ["s12SourceS11ConditionId"]
    simultaneous = scored[scored["historyProtocol"].astype(str) == SIMULTANEOUS_HISTORY_PROTOCOL].copy()
    no_mutant = scored[scored["historyProtocol"].astype(str) == NO_MUTANT_HISTORY_PROTOCOL].copy()
    metric_cols = [
        "finalStateHash",
        "finalValuesJson",
        "finalPanelPolicyIdsJson",
        "s07MosaicClass",
        "finalTargetQualityScore",
        "minPolicyGoalScore",
        "meanPolicyGoalScore",
        "finalSortednessPercent",
        "finalAggregation",
        "dominanceProxyScore",
        "stopReason",
        "swapCount",
        "activationCount",
        "mutantCloneFinalAggregation",
        "mutantCloneSpreadFraction",
    ]
    sim_ref = simultaneous[keys + [col for col in metric_cols if col in simultaneous.columns]].rename(
        columns={col: f"simultaneous{col[0].upper()}{col[1:]}" for col in metric_cols if col in simultaneous.columns}
    )
    no_ref = no_mutant[keys + [col for col in metric_cols if col in no_mutant.columns]].rename(
        columns={col: f"noHistory{col[0].upper()}{col[1:]}" for col in metric_cols if col in no_mutant.columns}
    )
    out = scored.merge(sim_ref, on=keys, how="left").merge(no_ref, on=keys, how="left")
    out["deltaVsSimultaneousFinalTargetQualityScore"] = pd.to_numeric(out["finalTargetQualityScore"], errors="coerce") - pd.to_numeric(
        out["simultaneousFinalTargetQualityScore"], errors="coerce"
    )
    out["deltaVsSimultaneousMinPolicyGoalScore"] = pd.to_numeric(out["minPolicyGoalScore"], errors="coerce") - pd.to_numeric(
        out["simultaneousMinPolicyGoalScore"], errors="coerce"
    )
    out["deltaVsSimultaneousFinalAggregation"] = pd.to_numeric(out["finalAggregation"], errors="coerce") - pd.to_numeric(
        out["simultaneousFinalAggregation"], errors="coerce"
    )
    out["deltaVsNoHistoryFinalTargetQualityScore"] = pd.to_numeric(out["finalTargetQualityScore"], errors="coerce") - pd.to_numeric(
        out["noHistoryFinalTargetQualityScore"], errors="coerce"
    )
    out["deltaVsNoHistoryMinPolicyGoalScore"] = pd.to_numeric(out["minPolicyGoalScore"], errors="coerce") - pd.to_numeric(
        out["noHistoryMinPolicyGoalScore"], errors="coerce"
    )
    out["deltaVsNoHistoryFinalAggregation"] = pd.to_numeric(out["finalAggregation"], errors="coerce") - pd.to_numeric(
        out["noHistoryFinalAggregation"], errors="coerce"
    )
    out["historyChangedMosaicClassVsSimultaneous"] = out["s07MosaicClass"].astype(str) != out["simultaneousS07MosaicClass"].astype(str)
    out["historyChangedMosaicClassVsNoHistory"] = out["s07MosaicClass"].astype(str) != out["noHistoryS07MosaicClass"].astype(str)
    out["historyChangedStateHashVsSimultaneous"] = out["finalStateHash"].astype(str) != out["simultaneousFinalStateHash"].astype(str)
    out["simultaneousReplayMatchesS11Reference"] = out["simultaneousFinalStateHash"].astype(str) == out["s11ReferenceFinalStateHash"].astype(str)
    out["noHistoryReplayMatchesS11NoMutant"] = out["noHistoryFinalStateHash"].astype(str) == out["s11NoMutantFinalStateHash"].astype(str)
    return out


def classify_history_outcomes(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        protocol = str(row.get("historyProtocol", ""))
        if protocol == NO_MUTANT_HISTORY_PROTOCOL:
            outcome = "matched_no_mutant_replay"
        elif protocol == SIMULTANEOUS_HISTORY_PROTOCOL:
            outcome = "simultaneous_clone_replay"
        elif not bool(row.get("mutantCloneSizeConserved", False)):
            outcome = "lineage_tracking_failure"
        else:
            target_delta = safe_float(row.get("deltaVsSimultaneousFinalTargetQualityScore"), 0.0)
            min_delta = safe_float(row.get("deltaVsSimultaneousMinPolicyGoalScore"), 0.0)
            changed_class = bool(row.get("historyChangedMosaicClassVsSimultaneous", False))
            if target_delta >= 0.05 and min_delta >= -5.0:
                outcome = "history_rescue_proxy"
            elif target_delta <= -0.05 or min_delta <= -10.0:
                outcome = "history_worsened_proxy"
            elif changed_class:
                outcome = "timing_sensitive_mosaic_shift"
            else:
                outcome = "history_neutral_within_threshold"
        target_delta = safe_float(row.get("deltaVsSimultaneousFinalTargetQualityScore"), 0.0)
        min_delta = safe_float(row.get("deltaVsSimultaneousMinPolicyGoalScore"), 0.0)
        agg_delta = safe_float(row.get("deltaVsSimultaneousFinalAggregation"), 0.0)
        class_changed = bool(row.get("historyChangedMosaicClassVsSimultaneous", False))
        stop_changed = str(row.get("stopReason", "")) != str(row.get("simultaneousStopReason", ""))
        clone_agg_delta = abs(
            safe_float(row.get("mutantCloneFinalAggregation"), 0.0)
            - safe_float(row.get("simultaneousMutantCloneFinalAggregation"), 0.0)
        )
        effect = max(
            0.0,
            min(
                1.0,
                2.0 * abs(target_delta)
                + abs(min_delta) / 100.0
                + abs(agg_delta)
                + 0.25 * float(class_changed)
                + 0.10 * float(stop_changed)
                + 0.25 * clone_agg_delta,
            ),
        )
        disruption = max(
            0.0,
            min(
                1.0,
                2.0 * max(0.0, -safe_float(row.get("deltaVsNoHistoryFinalTargetQualityScore"), 0.0))
                + max(0.0, -safe_float(row.get("deltaVsNoHistoryMinPolicyGoalScore"), 0.0) / 100.0)
                + 0.25 * float(bool(row.get("historyChangedMosaicClassVsNoHistory", False))),
            ),
        )
        rows.append(
            {
                **row,
                "historyOutcomeClass": outcome,
                "historyOutcomeClassifierVersion": HISTORY_VERSION,
                "historyEffectMagnitudeScore": float(effect),
                "historyDisruptionProxyScore": float(disruption),
            }
        )
    return pd.DataFrame(rows)


def summarize_history_effects(scored: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["historyProtocol", "historyTimingLabel", "memoryCarryoverMode", "governanceVariant", "interfaceRuleVariant", "s11MutantOutcomeClass"]
    rows: list[dict[str, Any]] = []
    for keys, group in scored.groupby(group_cols, dropna=False, sort=False):
        payload = dict(zip(group_cols, keys, strict=True))
        non_control = ~group["historyProtocol"].astype(str).isin({NO_MUTANT_HISTORY_PROTOCOL, SIMULTANEOUS_HISTORY_PROTOCOL})
        rows.append(
            {
                **payload,
                "conditionCount": int(len(group)),
                "runSuccessRate": float(group["runSucceeded"].map(bool).mean()),
                "meanDeltaTargetQualityVsSimultaneous": float(pd.to_numeric(group["deltaVsSimultaneousFinalTargetQualityScore"], errors="coerce").mean()),
                "meanDeltaMinGoalVsSimultaneous": float(pd.to_numeric(group["deltaVsSimultaneousMinPolicyGoalScore"], errors="coerce").mean()),
                "meanHistoryEffectMagnitudeScore": float(pd.to_numeric(group["historyEffectMagnitudeScore"], errors="coerce").mean()),
                "meanHistoryDisruptionProxyScore": float(pd.to_numeric(group["historyDisruptionProxyScore"], errors="coerce").mean()),
                "mosaicShiftRateVsSimultaneous": float(group["historyChangedMosaicClassVsSimultaneous"].map(bool).mean()),
                "stateHashShiftRateVsSimultaneous": float(group["historyChangedStateHashVsSimultaneous"].map(bool).mean()),
                "historySensitiveConditionCount": int((group.loc[non_control, "historyEffectMagnitudeScore"] >= 0.15).sum()) if non_control.any() else 0,
                "outcomeCountsJson": compact_json(dict(Counter(group["historyOutcomeClass"].astype(str)))),
                "memoryEligiblePolicyPresent": bool(group["memoryEligiblePolicyPresent"].map(bool).any()),
                "broadControlLike": bool(group.get("broadControlLike", pd.Series(False, index=group.index)).map(bool).any()),
            }
        )
    return pd.DataFrame(rows)


def summarize_history_outcomes(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["historyOutcomeClass", "historyProtocol"], dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            meanHistoryEffectMagnitudeScore=("historyEffectMagnitudeScore", "mean"),
            meanHistoryDisruptionProxyScore=("historyDisruptionProxyScore", "mean"),
            meanDeltaTargetQualityVsSimultaneous=("deltaVsSimultaneousFinalTargetQualityScore", "mean"),
            mosaicShiftRateVsSimultaneous=("historyChangedMosaicClassVsSimultaneous", "mean"),
        )
        .reset_index()
    )


def validation_checks(
    selected_contexts: pd.DataFrame,
    condition_df: pd.DataFrame,
    planned_timeline_df: pd.DataFrame,
    actual_timeline_df: pd.DataFrame,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    protocols = set(condition_df["historyProtocol"].astype(str))
    simultaneous = scored[scored["historyProtocol"].astype(str) == SIMULTANEOUS_HISTORY_PROTOCOL]
    no_history = scored[scored["historyProtocol"].astype(str) == NO_MUTANT_HISTORY_PROTOCOL]
    clone_rows = scored[~scored["historyProtocol"].astype(str).eq(NO_MUTANT_HISTORY_PROTOCOL)].copy()
    sim_replay_ok = bool(
        not simultaneous.empty
        and (simultaneous["finalStateHash"].astype(str) == simultaneous["s11ReferenceFinalStateHash"].astype(str)).all()
    )
    no_history_replay_ok = bool(
        not no_history.empty
        and (no_history["finalStateHash"].astype(str) == no_history["s11NoMutantFinalStateHash"].astype(str)).all()
    )
    has_reset = condition_df["memoryCarryoverMode"].astype(str).str.contains("reset").any()
    has_carry = condition_df["memoryCarryoverMode"].astype(str).str.contains("carryover").any()
    memory_eligible = bool(selected_contexts["memoryEligiblePolicyPresent"].map(bool).any()) if not selected_contexts.empty else False
    checks = [
        {
            "checkId": "s11_clone_contexts_loaded",
            "success": bool(
                not selected_contexts.empty
                and selected_contexts["s12MatchedNoMutantConditionId"].astype(str).str.len().gt(0).all()
                and selected_contexts["s11MutantOutcomeClass" if "s11MutantOutcomeClass" in selected_contexts.columns else "mutantOutcomeClass"].astype(str).nunique() >= 2
            ),
            "detail": f"{len(selected_contexts)} S11 clone contexts selected with matched no-mutant controls",
        },
        {
            "checkId": "protocol_matrix_coverage",
            "success": bool(
                len(protocols) >= 10
                and NO_MUTANT_HISTORY_PROTOCOL in protocols
                and SIMULTANEOUS_HISTORY_PROTOCOL in protocols
                and any("staged_introduction" in item for item in protocols)
                and any("transient_rule_release" in item for item in protocols)
                and {"early", "late", "simultaneous", "none"} <= set(condition_df["historyTimingLabel"].astype(str))
            ),
            "detail": f"{len(condition_df)} S12 conditions across protocols {sorted(protocols)}",
        },
        {
            "checkId": "event_timing_stored",
            "success": bool(
                not planned_timeline_df.empty
                and not actual_timeline_df.empty
                and scored["historyEventTimingStored"].fillna(False).map(bool).all()
                and actual_timeline_df["startActivation"].notna().all()
                and actual_timeline_df["endActivation"].notna().all()
            ),
            "detail": f"{len(planned_timeline_df)} planned and {len(actual_timeline_df)} actual stage-event rows written",
        },
        {
            "checkId": "memory_reset_and_carryover_separated",
            "success": bool(has_reset and has_carry and "memoryCarryoverMode" in scored.columns),
            "detail": f"reset modes present={has_reset}; carryover modes present={has_carry}; true E04 memory-policy context present={memory_eligible}",
        },
        {
            "checkId": "matched_no_mutant_controls_replayed",
            "success": no_history_replay_ok,
            "detail": f"{int((no_history['finalStateHash'].astype(str) == no_history['s11NoMutantFinalStateHash'].astype(str)).sum()) if not no_history.empty else 0}/{len(no_history)} no-history controls matched S11 no-mutant hashes",
        },
        {
            "checkId": "simultaneous_clone_references_replayed",
            "success": sim_replay_ok,
            "detail": f"{int((simultaneous['finalStateHash'].astype(str) == simultaneous['s11ReferenceFinalStateHash'].astype(str)).sum()) if not simultaneous.empty else 0}/{len(simultaneous)} simultaneous clone references matched S11 hashes",
        },
        {
            "checkId": "fixed_size_clone_conserved",
            "success": bool(not clone_rows.empty and clone_rows["mutantCloneSizeConserved"].fillna(False).map(bool).all()),
            "detail": "all S12 clone-present rows conserved fixed-size lineage counts",
        },
        {
            "checkId": "runs_succeeded_and_counts_valid",
            "success": bool(
                scored["runSucceeded"].fillna(False).map(bool).all()
                and scored["valueCountsConserved"].fillna(False).map(bool).all()
                and scored["actualRatiosMatchTarget"].fillna(False).map(bool).all()
            ),
            "detail": "all S12 runs succeeded and conserved values; final policy counts matched protocol targets",
        },
        {
            "checkId": "governance_interface_contexts_retained",
            "success": bool(
                scored["governanceVariant"].astype(str).nunique() >= 2
                and scored["interfaceRuleVariant"].astype(str).nunique() >= 2
                and scored["graftOutcomeClass"].astype(str).nunique() >= 2
            ),
            "detail": (
                f"governance={sorted(scored['governanceVariant'].astype(str).unique())}; "
                f"interfaces={sorted(scored['interfaceRuleVariant'].astype(str).unique())}; "
                f"S10 graft outcomes={sorted(scored['graftOutcomeClass'].astype(str).unique())}"
            ),
        },
        {
            "checkId": "history_outcomes_assigned",
            "success": bool(scored["historyOutcomeClass"].astype(str).str.len().gt(0).all()),
            "detail": f"history outcomes: {dict(Counter(scored['historyOutcomeClass'].astype(str)))}",
        },
        {
            "checkId": "analogy_and_fixed_size_caveats_retained",
            "success": bool(
                scored["analogyCaveat"].astype(str).str.contains("computational analog").all()
                and scored["fixedSizeCloneCaveat"].astype(str).str.contains("fixed-size").all()
                and scored["claimBoundary"].astype(str).str.contains("Computational").all()
            ),
            "detail": "developmental-history and fixed-size clone caveats retained as computational analogies",
        },
    ]
    return pd.DataFrame(checks)


def write_history_plots(effect_summary: pd.DataFrame, outcome_summary: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not effect_summary.empty:
        plot_source = effect_summary[
            ~effect_summary["historyProtocol"].astype(str).isin({NO_MUTANT_HISTORY_PROTOCOL, SIMULTANEOUS_HISTORY_PROTOCOL})
        ].copy()
        if plot_source.empty:
            plot_source = effect_summary.copy()
        plot = (
            plot_source.groupby("historyProtocol", dropna=False)
            .agg(meanEffect=("meanHistoryEffectMagnitudeScore", "mean"), meanTargetDelta=("meanDeltaTargetQualityVsSimultaneous", "mean"))
            .reset_index()
            .sort_values("meanEffect", ascending=False, kind="mergesort")
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(plot["historyProtocol"], plot["meanEffect"], color="#4c78a8")
        ax.set_ylabel("Mean history-effect magnitude")
        ax.set_title("E06 S12 developmental-history sensitivity by protocol")
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s12_history_effects.png",
            figure_dir / "e06_s12_history_effects.pdf",
            step_dir / "history_effects.png",
            step_dir / "history_effects.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    if not outcome_summary.empty:
        pivot = outcome_summary.pivot_table(index="historyOutcomeClass", columns="historyProtocol", values="conditionCount", fill_value=0, aggfunc="sum")
        fig, ax = plt.subplots(figsize=(12, 5))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_title("E06 S12 developmental-history outcome classes")
        fig.colorbar(image, ax=ax, label="Condition count")
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s12_history_outcomes.png",
            figure_dir / "e06_s12_history_outcomes.pdf",
            step_dir / "history_outcomes.png",
            step_dir / "history_outcomes.pdf",
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
    ranked = (
        effect_summary[
            ~effect_summary["historyProtocol"].astype(str).isin({NO_MUTANT_HISTORY_PROTOCOL, SIMULTANEOUS_HISTORY_PROTOCOL})
        ].copy()
        if not effect_summary.empty
        else pd.DataFrame()
    )
    top = ranked.sort_values("meanHistoryEffectMagnitudeScore", ascending=False, kind="mergesort").head(8) if not ranked.empty else pd.DataFrame()
    effect_lines = "\n".join(
        f"- `{row.historyProtocol}` / `{row.governanceVariant}` / `{row.interfaceRuleVariant}` / S11 `{row.s11MutantOutcomeClass}`: "
        f"effect {row.meanHistoryEffectMagnitudeScore:.3f}, target delta {row.meanDeltaTargetQualityVsSimultaneous:.3f}, "
        f"mosaic shift {row.mosaicShiftRateVsSimultaneous:.2f}"
        for row in top.itertuples(index=False)
    ) or "- No history effects available."
    outcome_lines = "\n".join(
        f"- `{row.historyOutcomeClass}` / `{row.historyProtocol}`: {int(row.conditionCount)} conditions"
        for row in outcome_summary.itertuples(index=False)
    ) or "- No history outcomes available."
    text = f"""# Research Step S12: Run developmental-history experiments

## Completion status

Research step ID: `S12`. {status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Ran {status['conditionCount']} developmental-history conditions from {status['selectedContextCount']} S11 clone contexts. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

{HISTORY_ANALOGY_CAVEAT} {FIXED_SIZE_CLONE_CAVEAT} {NO_TRUE_MEMORY_CONTEXT_CAVEAT if not status.get('memoryEligiblePolicyPresent') else 'At least one memory-eligible policy context was present.'}

## Lay summary

S12 replayed selected S11 clone outcomes under simultaneous, staged, transient early/late, reset, and carryover protocols. It measured whether final morphology, target-quality, dominance, aggregation, and fixed-size clone placement changed when the same clone/control context had a different computational history.

## Anchor history effects

{effect_lines}

## Outcome summary

{outcome_lines}

## Recommended next action

{status['recommendedNextAction']}
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def load_winner_criteria(path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return predefined_winner_criteria(load_boundary_caveats(DEFAULT_S05_BOUNDARY_CAVEATS_PATH))


def run_s12_developmental_history_experiments(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    panel_path: Path = DEFAULT_PANEL_PATH,
    s11_results_path: Path = DEFAULT_S11_RESULTS_PATH,
    s11_outcome_summary_path: Path = DEFAULT_S11_OUTCOME_SUMMARY_PATH,
    s06_pair_summary_path: Path = DEFAULT_S06_PAIR_SUMMARY_PATH,
    winner_criteria_path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH,
    workers: int | None = None,
    max_contexts: int = 12,
    max_activations: int = 3500,
    max_swaps: int = 2500,
    max_comparisons: int = 20000,
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
    s11_results = load_s11_results(s11_results_path)
    s11_outcome_summary = pd.read_parquet(s11_outcome_summary_path) if s11_outcome_summary_path.exists() else pd.DataFrame()
    selected_contexts = select_s12_contexts(s11_results, max_contexts=max_contexts)
    protocol_df = history_protocol_table()
    condition_df, planned_timeline_df = build_developmental_history_condition_matrix(
        selected_contexts,
        protocols=protocol_df,
        total_activation_cap=max_activations,
    )
    criteria = load_winner_criteria(winner_criteria_path)
    pair_summary = pd.read_parquet(s06_pair_summary_path) if s06_pair_summary_path.exists() else pd.DataFrame()

    raw = run_history_conditions(
        condition_df,
        ready_panel,
        workers=workers,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )
    actual_timeline_df = timeline_rows_from_results(raw)
    goal = augment_goal_results(raw)
    scored = score_dominance_results(goal, criteria)
    classified = classify_mosaic_formations(scored, pair_summary if not pair_summary.empty else None)
    classified = attach_history_reference_deltas(classified)
    classified = classify_history_outcomes(classified)
    classified["researchStepId"] = STEP_ID
    classified["sourceResearchStepId"] = "S11"
    classified["interventionStepId"] = STEP_ID
    classified["interventionType"] = "developmental_history_experiment"
    classified["historyVersion"] = HISTORY_VERSION
    effect_summary = summarize_history_effects(classified)
    outcome_summary = summarize_history_outcomes(classified)
    validation_df = validation_checks(selected_contexts, condition_df, planned_timeline_df, actual_timeline_df, classified)

    selected_csv = step_dir / "selected_s11_history_contexts.csv"
    selected_parquet = step_dir / "selected_s11_history_contexts.parquet"
    protocol_csv = step_dir / "history_protocol_parameters.csv"
    protocol_parquet = step_dir / "history_protocol_parameters.parquet"
    condition_csv = step_dir / "developmental_history_condition_matrix.csv"
    condition_parquet = step_dir / "developmental_history_condition_matrix.parquet"
    planned_timeline_csv = step_dir / "planned_history_event_timeline.csv"
    planned_timeline_parquet = step_dir / "planned_history_event_timeline.parquet"
    actual_timeline_csv = step_dir / "history_event_timeline.csv"
    actual_timeline_parquet = step_dir / "history_event_timeline.parquet"
    runs_csv = step_dir / "developmental_history_runs.csv"
    runs_parquet = step_dir / "developmental_history_runs.parquet"
    result_csv = results_dir / "e06_developmental_history.csv"
    result_parquet = results_dir / "e06_developmental_history.parquet"
    combined_csv = results_dir / "e06_governance_interventions.csv"
    combined_parquet = results_dir / "e06_governance_interventions.parquet"
    effect_csv = step_dir / "history_effect_summary.csv"
    effect_parquet = step_dir / "history_effect_summary.parquet"
    outcome_csv = step_dir / "history_outcome_summary.csv"
    outcome_parquet = step_dir / "history_outcome_summary.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    for path, df in [
        (selected_csv, compact_result_csv(selected_contexts)),
        (protocol_csv, protocol_df),
        (condition_csv, compact_result_csv(condition_df)),
        (planned_timeline_csv, planned_timeline_df),
        (actual_timeline_csv, actual_timeline_df),
        (runs_csv, compact_result_csv(classified)),
        (result_csv, compact_result_csv(classified)),
        (effect_csv, effect_summary),
        (outcome_csv, outcome_summary),
        (validation_csv, validation_df),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (selected_parquet, selected_contexts),
        (protocol_parquet, protocol_df),
        (condition_parquet, condition_df),
        (planned_timeline_parquet, planned_timeline_df),
        (actual_timeline_parquet, actual_timeline_df),
        (runs_parquet, classified),
        (result_parquet, classified),
        (effect_parquet, effect_summary),
        (outcome_parquet, outcome_summary),
        (validation_parquet, validation_df),
    ]:
        df.to_parquet(path, index=False)

    if combined_parquet.exists():
        previous_combined = pd.read_parquet(combined_parquet)
        previous_combined = (
            previous_combined[previous_combined.get("interventionStepId", "").astype(str) != STEP_ID]
            if "interventionStepId" in previous_combined.columns
            else previous_combined
        )
        combined = pd.concat([previous_combined, classified], ignore_index=True, sort=False)
    else:
        combined = classified.copy()
    compact_result_csv(combined).to_csv(combined_csv, index=False)
    combined.to_parquet(combined_parquet, index=False)

    figure_paths = write_history_plots(effect_summary, outcome_summary, figures_dir, step_dir)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() else "failed"
    non_reference = classified[
        ~classified["historyProtocol"].astype(str).isin({NO_MUTANT_HISTORY_PROTOCOL, SIMULTANEOUS_HISTORY_PROTOCOL})
    ]
    outcome = "supportive" if validation_result == "passed" and not non_reference.empty else "constraining/contradictory"
    artifacts = [
        selected_csv,
        selected_parquet,
        protocol_csv,
        protocol_parquet,
        condition_csv,
        condition_parquet,
        planned_timeline_csv,
        planned_timeline_parquet,
        actual_timeline_csv,
        actual_timeline_parquet,
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
    memory_eligible = bool(selected_contexts["memoryEligiblePolicyPresent"].map(bool).any())
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_result == "passed",
        "status": "completed" if validation_result == "passed" else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifacts],
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            HISTORY_ANALOGY_CAVEAT,
            FIXED_SIZE_CLONE_CAVEAT,
            NO_TRUE_MEMORY_CONTEXT_CAVEAT if not memory_eligible else "S12 included at least one memory-eligible policy context.",
            "History effects are final-state computational proxies bounded by S11 fixed-size clone and E06 metric assumptions.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then S13 causal/predictive models should use S12 history variables with leakage checks; "
            "do not start S13 until review is complete."
        ),
        "laySummary": (
            "S12 compared simultaneous, staged, transient early/late, reset, and carryover histories for selected S11 fixed-size clone contexts "
            "against matched no-mutant and simultaneous clone replays."
        ),
        "outcomeClassification": outcome,
        "selectedContextCount": int(len(selected_contexts)),
        "conditionCount": int(len(condition_df)),
        "protocolCount": int(protocol_df["historyProtocol"].nunique()),
        "actualTimelineRowCount": int(len(actual_timeline_df)),
        "historySensitiveConditionCount": int((non_reference["historyEffectMagnitudeScore"] >= 0.15).sum()) if not non_reference.empty else 0,
        "meanHistoryEffectMagnitudeScore": float(pd.to_numeric(non_reference["historyEffectMagnitudeScore"], errors="coerce").mean()) if not non_reference.empty else float("nan"),
        "memoryEligiblePolicyPresent": memory_eligible,
        "governanceVariants": sorted(selected_contexts["governanceVariant"].astype(str).unique()),
        "interfaceRuleVariants": sorted(selected_contexts["interfaceRuleVariant"].astype(str).unique()),
        "s11MutantOutcomeClasses": sorted(selected_contexts["mutantOutcomeClass"].astype(str).unique()),
        "workerCount": int(workers),
        "maxActivations": int(max_activations),
        "maxSwaps": int(max_swaps),
        "maxComparisons": int(max_comparisons),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
        "newDependenciesInstalled": [],
        "sourceCodeLocation": str(repo_root),
        "sourceCodeArtifactPolicy": "repository-backed source is committed to git; source files are not copied into artifacts per workspace instructions",
        "s11OutcomeSummaryRows": int(len(s11_outcome_summary)),
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
        "historyVersion": HISTORY_VERSION,
        "inputArtifacts": {
            "s11Results": str(s11_results_path),
            "s11OutcomeSummary": str(s11_outcome_summary_path),
            "s06PairSummary": str(s06_pair_summary_path),
            "winnerCriteria": str(winner_criteria_path),
            "panel": str(panel_path),
        },
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
        "protocols": protocol_df,
        "conditions": condition_df,
        "plannedTimeline": planned_timeline_df,
        "actualTimeline": actual_timeline_df,
        "results": classified,
        "effectSummary": effect_summary,
        "outcomeSummary": outcome_summary,
        "validation": validation_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E06 S12 developmental-history experiments")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--s11-results-path", type=Path, default=DEFAULT_S11_RESULTS_PATH)
    parser.add_argument("--s11-outcome-summary-path", type=Path, default=DEFAULT_S11_OUTCOME_SUMMARY_PATH)
    parser.add_argument("--s06-pair-summary-path", type=Path, default=DEFAULT_S06_PAIR_SUMMARY_PATH)
    parser.add_argument("--winner-criteria-path", type=Path, default=DEFAULT_S06_WINNER_CRITERIA_PATH)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-contexts", type=int, default=12)
    parser.add_argument("--max-activations", type=int, default=3500)
    parser.add_argument("--max-swaps", type=int, default=2500)
    parser.add_argument("--max-comparisons", type=int, default=20000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_s12_developmental_history_experiments(
        artifacts_dir=args.artifacts_dir,
        panel_path=args.panel_path,
        s11_results_path=args.s11_results_path,
        s11_outcome_summary_path=args.s11_outcome_summary_path,
        s06_pair_summary_path=args.s06_pair_summary_path,
        winner_criteria_path=args.winner_criteria_path,
        workers=args.workers,
        max_contexts=args.max_contexts,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['conditionCount']} developmental-history conditions, "
        f"{status['selectedContextCount']} S11 contexts, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
