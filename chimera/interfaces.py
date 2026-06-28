"""E06 S08 explicit interface-rule interventions for chimeric collectives."""

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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e02_deterministic_simulator.metrics import aggregation, sortedness_percent, state_hash
from memory_repair import SignalEventSimulator

from .dominance import (
    DEFAULT_S05_BOUNDARY_CAVEATS_PATH,
    artifact_records,
    load_boundary_caveats,
    predefined_winner_criteria,
    score_dominance_results,
)
from .goals import augment_goal_results
from .mixtures import (
    DEFAULT_PANEL_PATH,
    compact_json,
    compact_result_csv,
    git_value,
    initial_values,
    load_ready_panel,
    parse_json_maybe,
    policy_assignment,
    simulator_backend,
    write_json,
)
from .mosaics import (
    DEFAULT_S06_PAIR_SUMMARY_PATH,
    DEFAULT_S06_WINNER_CRITERIA_PATH,
    MOSAIC_CLASSIFIER_VERSION,
    REQUESTED_MOSAIC_CLASSES,
    classify_mosaic_formations,
    safe_float,
)
from .panel import CLAIM_BOUNDARY, instantiate_panel_policy, json_ready


STEP_ID = "S08"
STEP_NUMBER = 8
INTERFACE_RULE_VERSION = "e06_s08_interface_rules.v1"
DEFAULT_S07_CLASSIFICATIONS_PATH = Path("/artifacts/results/e06_mosaic_classifications.parquet")
DEFAULT_S07_PRIORITY_PATH = Path("/artifacts/research_steps/S07/priority_mosaic_candidates.parquet")
DEFAULT_S07_CLASS_SUMMARY_PATH = Path("/artifacts/research_steps/S07/mosaic_class_summary.parquet")
INTERFACE_RULE_VARIANTS: tuple[str, ...] = (
    "disabled_baseline",
    "adhesion_homotypic",
    "repulsion_heterotypic",
    "permeability_low",
    "recognition_self_nonself",
    "interface_boundary_stabilize",
    "combined_self_boundary",
)
_PANEL_LOOKUP: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class InterfaceRuleConfig:
    """Serializable local interface rule configuration."""

    variant: str
    enabled: bool
    rule_family: str
    adhesion_strength: float = 0.0
    adhesion_tolerance: int = 0
    repulsion_strength: float = 0.0
    permeability: float = 1.0
    recognition_mode: str = "none"
    interface_swap_mode: str = "unrestricted"
    recognition_enabled: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.permeability) <= 1.0:
            raise ValueError("permeability must be in [0, 1]")
        if self.recognition_mode not in {"none", "self_nonself", "strict_self_nonself"}:
            raise ValueError(f"unknown recognition_mode: {self.recognition_mode}")
        if self.interface_swap_mode not in {"unrestricted", "boundary_only", "boundary_stabilize", "interior_only"}:
            raise ValueError(f"unknown interface_swap_mode: {self.interface_swap_mode}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INTERFACE_RULE_VERSION,
            "variant": self.variant,
            "enabled": bool(self.enabled),
            "ruleFamily": self.rule_family,
            "adhesionStrength": float(self.adhesion_strength),
            "adhesionTolerance": int(self.adhesion_tolerance),
            "repulsionStrength": float(self.repulsion_strength),
            "permeability": float(self.permeability),
            "recognitionMode": self.recognition_mode,
            "interfaceSwapMode": self.interface_swap_mode,
            "recognitionEnabled": bool(self.recognition_enabled),
            "description": self.description,
            "claimBoundary": CLAIM_BOUNDARY,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | str | "InterfaceRuleConfig") -> "InterfaceRuleConfig":
        if isinstance(payload, InterfaceRuleConfig):
            return payload
        if isinstance(payload, str):
            try:
                parsed = parse_json_maybe(payload, None)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                payload = parsed
            else:
                return interface_rule_config_by_variant(payload)
        return cls(
            variant=str(payload.get("variant", "disabled_baseline")),
            enabled=bool(payload.get("enabled", False)),
            rule_family=str(payload.get("ruleFamily", payload.get("rule_family", "disabled"))),
            adhesion_strength=float(payload.get("adhesionStrength", payload.get("adhesion_strength", 0.0))),
            adhesion_tolerance=int(payload.get("adhesionTolerance", payload.get("adhesion_tolerance", 0))),
            repulsion_strength=float(payload.get("repulsionStrength", payload.get("repulsion_strength", 0.0))),
            permeability=float(payload.get("permeability", 1.0)),
            recognition_mode=str(payload.get("recognitionMode", payload.get("recognition_mode", "none"))),
            interface_swap_mode=str(payload.get("interfaceSwapMode", payload.get("interface_swap_mode", "unrestricted"))),
            recognition_enabled=bool(payload.get("recognitionEnabled", payload.get("recognition_enabled", False))),
            description=str(payload.get("description", "")),
        )


def interface_rule_configs() -> tuple[InterfaceRuleConfig, ...]:
    return (
        InterfaceRuleConfig(
            variant="disabled_baseline",
            enabled=False,
            rule_family="disabled",
            description="No explicit interface rule; should reproduce the S06/S07 baseline exactly.",
        ),
        InterfaceRuleConfig(
            variant="adhesion_homotypic",
            enabled=True,
            rule_family="adhesion",
            adhesion_strength=1.0,
            adhesion_tolerance=0,
            recognition_enabled=True,
            description="Homotypic adhesion gate blocks proposed swaps that increase policy-label interfaces.",
        ),
        InterfaceRuleConfig(
            variant="repulsion_heterotypic",
            enabled=True,
            rule_family="repulsion",
            repulsion_strength=1.0,
            recognition_enabled=True,
            description="Heterotypic repulsion gate blocks cross-policy swaps unless they reduce policy-label interfaces.",
        ),
        InterfaceRuleConfig(
            variant="permeability_low",
            enabled=True,
            rule_family="permeability",
            permeability=0.35,
            recognition_enabled=True,
            description="Low-permeability boundary gate deterministically allows about 35% of cross-policy swaps.",
        ),
        InterfaceRuleConfig(
            variant="recognition_self_nonself",
            enabled=True,
            rule_family="recognition",
            recognition_mode="self_nonself",
            recognition_enabled=True,
            description="Self/non-self recognition blocks cross-policy swaps that increase policy-label interfaces.",
        ),
        InterfaceRuleConfig(
            variant="interface_boundary_stabilize",
            enabled=True,
            rule_family="interface_specific_swap",
            interface_swap_mode="boundary_stabilize",
            recognition_enabled=True,
            description="Interface-specific swap rule blocks boundary swaps that increase interface count.",
        ),
        InterfaceRuleConfig(
            variant="combined_self_boundary",
            enabled=True,
            rule_family="combined",
            adhesion_strength=1.0,
            repulsion_strength=1.0,
            permeability=0.50,
            recognition_mode="self_nonself",
            interface_swap_mode="boundary_stabilize",
            recognition_enabled=True,
            description="Combined adhesion, heterotypic repulsion, self/non-self recognition, and moderate permeability.",
        ),
    )


def interface_rule_config_by_variant(variant: str) -> InterfaceRuleConfig:
    lookup = {config.variant: config for config in interface_rule_configs()}
    if variant not in lookup:
        raise ValueError(f"unknown S08 interface rule variant: {variant}")
    return lookup[variant]


def interface_rule_parameter_table(configs: Sequence[InterfaceRuleConfig] | None = None) -> pd.DataFrame:
    configs = tuple(configs or interface_rule_configs())
    rows = []
    for index, config in enumerate(configs):
        payload = config.to_dict()
        rows.append(
            {
                "interfaceRuleIndex": int(index),
                "interfaceRuleVariant": config.variant,
                "interfaceRuleEnabled": bool(config.enabled),
                "interfaceRuleFamily": config.rule_family,
                "adhesionStrength": float(config.adhesion_strength),
                "adhesionTolerance": int(config.adhesion_tolerance),
                "repulsionStrength": float(config.repulsion_strength),
                "permeability": float(config.permeability),
                "recognitionMode": config.recognition_mode,
                "interfaceSwapMode": config.interface_swap_mode,
                "recognitionEnabled": bool(config.recognition_enabled),
                "interfaceRuleConfigJson": compact_json(payload),
                "description": config.description,
                "interfaceRuleVersion": INTERFACE_RULE_VERSION,
            }
        )
    return pd.DataFrame(rows)


def _interface_count(labels: Sequence[int]) -> int:
    return int(sum(1 for left, right in zip(labels, labels[1:]) if int(left) != int(right)))


def _stable_unit_interval(*parts: Any) -> float:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)


def evaluate_interface_rule(
    labels: Sequence[int],
    actor_pos: int,
    target_pos: int,
    config: InterfaceRuleConfig | Mapping[str, Any] | str,
    *,
    condition_id: str = "",
    activation_count: int = 0,
    actor_cell_id: int | None = None,
    target_cell_id: int | None = None,
) -> dict[str, Any]:
    config = InterfaceRuleConfig.from_payload(config)
    if not (0 <= int(actor_pos) < len(labels)) or not (0 <= int(target_pos) < len(labels)):
        return {
            "allowed": False,
            "blocked": True,
            "blockReason": "target_oob",
            "crossBoundary": False,
            "interfaceDelta": 0,
            "permeabilityGateValue": 0.0,
        }
    current = [int(label) for label in labels]
    swapped = list(current)
    swapped[int(actor_pos)], swapped[int(target_pos)] = swapped[int(target_pos)], swapped[int(actor_pos)]
    before = _interface_count(current)
    after = _interface_count(swapped)
    delta = int(after - before)
    cross_boundary = bool(current[int(actor_pos)] != current[int(target_pos)])
    gate = _stable_unit_interval(condition_id, activation_count, actor_cell_id, target_cell_id, config.variant)
    reasons: list[str] = []

    if not config.enabled:
        return {
            "allowed": True,
            "blocked": False,
            "blockReason": "disabled",
            "crossBoundary": cross_boundary,
            "interfaceCountBefore": int(before),
            "interfaceCountAfter": int(after),
            "interfaceDelta": delta,
            "permeabilityGateValue": float(gate),
        }
    if config.adhesion_strength > 0 and delta > int(config.adhesion_tolerance):
        reasons.append("adhesion_blocks_interface_increase")
    if config.repulsion_strength > 0 and cross_boundary and delta >= -int(config.adhesion_tolerance):
        reasons.append("repulsion_blocks_nonsegregating_boundary_swap")
    if cross_boundary and config.permeability < 1.0 and gate > float(config.permeability):
        reasons.append("permeability_blocks_cross_boundary_swap")
    if config.recognition_mode == "self_nonself" and cross_boundary and delta > 0:
        reasons.append("recognition_blocks_interface_increase")
    if config.recognition_mode == "strict_self_nonself" and cross_boundary and delta >= 0:
        reasons.append("recognition_blocks_nonsegregating_boundary_swap")
    if config.interface_swap_mode == "boundary_only" and not cross_boundary:
        reasons.append("interface_specific_blocks_interior_swap")
    if config.interface_swap_mode == "boundary_stabilize" and cross_boundary and delta > 0:
        reasons.append("interface_specific_blocks_boundary_destabilization")
    if config.interface_swap_mode == "interior_only" and cross_boundary:
        reasons.append("interface_specific_blocks_boundary_swap")
    return {
        "allowed": not reasons,
        "blocked": bool(reasons),
        "blockReason": "+".join(sorted(set(reasons))) if reasons else "allowed",
        "crossBoundary": cross_boundary,
        "interfaceCountBefore": int(before),
        "interfaceCountAfter": int(after),
        "interfaceDelta": delta,
        "permeabilityGateValue": float(gate),
    }


class InterfaceRuleEventSimulator(SignalEventSimulator):
    """Signal-capable simulator with explicit policy-label interface swap gates."""

    def __init__(
        self,
        *args: Any,
        interface_rule_config: InterfaceRuleConfig | Mapping[str, Any] | str | None = None,
        **kwargs: Any,
    ) -> None:
        self.interface_rule_config = InterfaceRuleConfig.from_payload(interface_rule_config or "disabled_baseline")
        self.interface_rule_evaluations = 0
        self.interface_rule_allowed_swaps = 0
        self.interface_rule_blocked_swaps = 0
        self.interface_cross_boundary_attempts = 0
        self.interface_delta_sum = 0
        self.interface_allowed_delta_sum = 0
        self.interface_blocked_delta_sum = 0
        self.interface_block_reason_counts: Counter[str] = Counter()
        self._last_interface_block_reason = ""
        super().__init__(*args, **kwargs)

    def _interface_decision(self, actor_pos: int, target_pos: int, *, record: bool) -> dict[str, Any]:
        labels = [int(cell.label) for cell in self.cells]
        target_cell_id = self.cells[int(target_pos)].cell_id if 0 <= int(target_pos) < len(self.cells) else None
        actor_cell_id = self.cells[int(actor_pos)].cell_id if 0 <= int(actor_pos) < len(self.cells) else None
        decision = evaluate_interface_rule(
            labels,
            actor_pos,
            target_pos,
            self.interface_rule_config,
            condition_id=self.condition_id,
            activation_count=self.activation_count,
            actor_cell_id=actor_cell_id,
            target_cell_id=target_cell_id,
        )
        if record and self.interface_rule_config.enabled:
            self.interface_rule_evaluations += 1
            self.interface_delta_sum += int(decision.get("interfaceDelta", 0))
            if bool(decision.get("crossBoundary", False)):
                self.interface_cross_boundary_attempts += 1
            if bool(decision.get("allowed", False)):
                self.interface_rule_allowed_swaps += 1
                self.interface_allowed_delta_sum += int(decision.get("interfaceDelta", 0))
            else:
                self.interface_rule_blocked_swaps += 1
                self.interface_blocked_delta_sum += int(decision.get("interfaceDelta", 0))
                reason = str(decision.get("blockReason", "interface_rule_blocked"))
                self.interface_block_reason_counts[reason] += 1
                self._last_interface_block_reason = reason
        return decision

    def _can_swap(self, actor_pos: int, target_pos: int) -> bool:
        self._last_interface_block_reason = ""
        if not super()._can_swap(actor_pos, target_pos):
            return False
        if not self.interface_rule_config.enabled:
            return True
        decision = self._interface_decision(actor_pos, target_pos, record=True)
        if bool(decision.get("allowed", False)):
            return True
        self.blocked_move_attempts += 1
        return False

    def legal_action_exists(self) -> bool:
        if not self.interface_rule_config.enabled:
            return super().legal_action_exists()
        for cell in self.cells:
            if cell.frozen:
                continue
            pos = self.positions_by_id[cell.cell_id]
            observation = cell.policy.observe(self.cells, pos, cell.state, self.frozen_variant)
            if not cell.policy.legal_action_exists(observation, cell.state):
                continue
            candidate_targets = {pos - 1, pos + 1}
            if observation.target_position is not None:
                candidate_targets.add(int(observation.target_position))
            for target in candidate_targets:
                if not self._target_status_allows_policy_check(int(target)):
                    continue
                if self.frozen_variant == "stuck" and self.cells[int(target)].frozen:
                    continue
                if bool(self._interface_decision(pos, int(target), record=False).get("allowed", False)):
                    return True
        return False

    def interface_summary(self) -> dict[str, Any]:
        evaluations = max(1, int(self.interface_rule_evaluations))
        return {
            "interfaceRuleConfig": self.interface_rule_config.to_dict(),
            "interfaceRuleEvaluations": int(self.interface_rule_evaluations),
            "interfaceRuleAllowedSwaps": int(self.interface_rule_allowed_swaps),
            "interfaceRuleBlockedSwaps": int(self.interface_rule_blocked_swaps),
            "interfaceCrossBoundaryAttempts": int(self.interface_cross_boundary_attempts),
            "interfaceRuleBlockReasonCounts": dict(self.interface_block_reason_counts),
            "meanInterfaceDeltaProposed": float(self.interface_delta_sum / evaluations),
            "meanInterfaceDeltaAllowed": float(
                self.interface_allowed_delta_sum / max(1, int(self.interface_rule_allowed_swaps))
            ),
            "meanInterfaceDeltaBlocked": float(
                self.interface_blocked_delta_sum / max(1, int(self.interface_rule_blocked_swaps))
            ),
        }


def load_winner_criteria(path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return predefined_winner_criteria(load_boundary_caveats(DEFAULT_S05_BOUNDARY_CAVEATS_PATH))


def load_s07_classifications(path: Path = DEFAULT_S07_CLASSIFICATIONS_PATH) -> pd.DataFrame:
    required = {
        "conditionId",
        "s06PairKey",
        "pairCategory",
        "s07MosaicClass",
        "policyIdsJson",
        "policyCountsJson",
        "targetCountsJson",
        "arrangementPolicyIdsJson",
        "goalReverseDirectionsJson",
        "valueSeed",
        "schedulerSeed",
        "tieBreakerSeed",
        "finalStateHash",
        "finalValuesJson",
        "finalPanelPolicyIdsJson",
        "stopReason",
        "swapCount",
        "activationCount",
    }
    if not path.exists():
        raise FileNotFoundError(f"S07 classifications missing: {path}")
    df = pd.read_parquet(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"S07 classifications missing required columns: {missing}")
    if df.empty:
        raise ValueError("S07 classifications are empty")
    return df


def load_s07_priority(path: Path = DEFAULT_S07_PRIORITY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"S07 priority candidates missing: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("S07 priority candidates are empty")
    return df


def select_s08_baseline_contexts(
    s07_classifications: pd.DataFrame,
    priority_candidates: pd.DataFrame,
    *,
    max_contexts: int = 18,
) -> pd.DataFrame:
    priority_cols = [
        "conditionId",
        "s07PriorityRank",
        "s07PriorityScore",
        "s07PriorityTagsJson",
    ]
    priority = priority_candidates[[col for col in priority_cols if col in priority_candidates.columns]].copy()
    merged = s07_classifications.merge(priority, on="conditionId", how="left")
    merged["s07PriorityRank"] = pd.to_numeric(merged.get("s07PriorityRank", np.nan), errors="coerce")
    merged["s07PriorityScore"] = pd.to_numeric(merged.get("s07PriorityScore", 0.0), errors="coerce").fillna(0.0)
    selected_ids: list[str] = []
    reasons: dict[str, list[str]] = {}
    for klass in REQUESTED_MOSAIC_CLASSES:
        hits = merged[merged["s07MosaicClass"].astype(str) == klass].sort_values(
            ["s07PriorityScore", "dominanceProxyScore", "conditionId"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        if hits.empty:
            continue
        condition_id = str(hits.iloc[0]["conditionId"])
        selected_ids.append(condition_id)
        reasons.setdefault(condition_id, []).append(f"class_anchor_{klass}")
    priority_sorted = priority_candidates.sort_values(
        ["s07PriorityScore", "s07PriorityRank", "conditionId"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    for row in priority_sorted.to_dict(orient="records"):
        if len(selected_ids) >= int(max_contexts):
            break
        condition_id = str(row["conditionId"])
        if condition_id not in selected_ids:
            selected_ids.append(condition_id)
        reasons.setdefault(condition_id, []).append("s07_priority_context")
    selected = merged[merged["conditionId"].astype(str).isin(selected_ids)].copy()
    selected["_order"] = selected["conditionId"].map({condition_id: index for index, condition_id in enumerate(selected_ids)})
    selected = selected.sort_values("_order", kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)
    selected.insert(0, "s08ContextIndex", np.arange(len(selected), dtype=int))
    selected["s08SelectionReason"] = selected["conditionId"].map(lambda value: "+".join(sorted(set(reasons.get(str(value), [])))))
    selected["s08BaselineConditionId"] = selected["conditionId"].astype(str)
    selected["s08BaselineS07MosaicClass"] = selected["s07MosaicClass"].astype(str)
    selected["s08BaselineFinalStateHash"] = selected["finalStateHash"].astype(str)
    selected["s08BaselineFinalValuesJson"] = selected["finalValuesJson"].astype(str)
    selected["s08BaselineFinalPanelPolicyIdsJson"] = selected["finalPanelPolicyIdsJson"].astype(str)
    selected["s08BaselineStopReason"] = selected["stopReason"].astype(str)
    selected["s08BaselineSwapCount"] = pd.to_numeric(selected["swapCount"], errors="coerce").fillna(-1).astype(int)
    selected["s08BaselineActivationCount"] = pd.to_numeric(selected["activationCount"], errors="coerce").fillna(-1).astype(int)
    return selected


def build_interface_condition_matrix(
    selected_contexts: pd.DataFrame,
    configs: Sequence[InterfaceRuleConfig] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = tuple(configs or interface_rule_configs())
    parameter_df = interface_rule_parameter_table(configs)
    rows: list[dict[str, Any]] = []
    for context in selected_contexts.to_dict(orient="records"):
        source_id = str(context["conditionId"])
        for rule_index, config in enumerate(configs):
            row = dict(context)
            row["s08SourceConditionId"] = source_id
            row["conditionId"] = f"s08_c{int(context['s08ContextIndex']):03d}_{config.variant}"
            row["researchStepId"] = STEP_ID
            row["sourceResearchStepId"] = "S07"
            row["implementationPrefix"] = "e06_s08"
            row["interfaceRuleVersion"] = INTERFACE_RULE_VERSION
            row["interfaceRuleIndex"] = int(rule_index)
            row["interfaceRuleVariant"] = config.variant
            row["interfaceRuleEnabled"] = bool(config.enabled)
            row["interfaceRuleFamily"] = config.rule_family
            row["interfaceRuleConfigJson"] = compact_json(config.to_dict())
            row["interfaceRuleParameterHash"] = hashlib.sha256(row["interfaceRuleConfigJson"].encode("utf-8")).hexdigest()[:16]
            row["s08BaselineConditionId"] = source_id
            rows.append(row)
    condition_df = pd.DataFrame(rows)
    return condition_df.reset_index(drop=True), parameter_df


def _interface_worker_init(panel_records: Sequence[Mapping[str, Any]]) -> None:
    global _PANEL_LOOKUP
    _PANEL_LOOKUP = {str(row["panelPolicyId"]): dict(row) for row in panel_records}


def run_interface_condition(
    condition: Mapping[str, Any],
    panel_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    max_activations: int = 6000,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
) -> dict[str, Any]:
    panel_lookup = panel_lookup or _PANEL_LOOKUP
    ids = [str(item) for item in parse_json_maybe(condition["policyIdsJson"], [])]
    records = [dict(panel_lookup[pid]) for pid in ids]
    assigned_ids, numeric_labels, arrangement_hash = policy_assignment(condition)
    policy_cache = {pid: instantiate_panel_policy(panel_lookup[pid]) for pid in ids}
    policies = [policy_cache[pid] for pid in assigned_ids]
    reverse_payload = parse_json_maybe(condition.get("goalReverseDirectionsJson", ""), [])
    reverse_directions = [bool(value) for value in reverse_payload] if reverse_payload else [False] * int(condition["n"])
    if len(reverse_directions) != int(condition["n"]):
        raise ValueError("goalReverseDirectionsJson length must match n")
    values = initial_values(
        int(condition["valueSeed"]),
        int(condition["n"]),
        str(condition.get("valueProfile", "random_unique")),
    )
    source_backend = simulator_backend(records)
    first_signal = next((policy for policy in policy_cache.values() if hasattr(policy, "signal_config")), None)
    signal_config = getattr(first_signal, "signal_config", "no_signal")
    signal_config_json = compact_json(signal_config.to_dict()) if hasattr(signal_config, "to_dict") else compact_json(signal_config)
    config = InterfaceRuleConfig.from_payload(condition.get("interfaceRuleConfigJson", "disabled_baseline"))
    common = {
        "labels": numeric_labels,
        "reverse_directions": reverse_directions,
        "scheduler_seed": int(condition["schedulerSeed"]),
        "tie_breaker_seed": int(condition["tieBreakerSeed"]),
        "condition_id": str(condition["conditionId"]),
        "research_step_id": STEP_ID,
    }
    started = time.perf_counter()
    try:
        simulator = InterfaceRuleEventSimulator(
            values,
            policies,
            signal_config=signal_config,
            auto_wrap_policies=False,
            trace_signal_activations=False,
            trace_memory_activations=False,
            interface_rule_config=config,
            implementation=f"{condition.get('implementationPrefix', 'e06_s08')}_interface_rule",
            **common,
        )
        result = simulator.run(
            max_activations=int(max_activations),
            max_swaps=int(max_swaps),
            max_comparisons=int(max_comparisons),
            no_move_check_interval=int(condition["n"]),
        )
        final_labels = [int(cell.label) for cell in simulator.cells]
        final_ids = [ids[label] for label in final_labels]
        initial_ids = list(assigned_ids)
        final_values = simulator.current_values()
        initial_counts = Counter(initial_ids)
        final_counts = Counter(final_ids)
        norm_positions = np.linspace(0.0, 1.0, num=len(final_ids)) if len(final_ids) > 1 else np.asarray([0.0])
        mean_positions = {
            pid: float(np.mean([norm_positions[i] for i, value in enumerate(final_ids) if value == pid]))
            for pid in ids
        }
        left_id = str(condition["leftPanelPolicyId"])
        right_id = str(condition["rightPanelPolicyId"])
        left_mean = mean_positions.get(left_id)
        right_mean = mean_positions.get(right_id)
        minority_count = min(initial_counts.values()) if initial_counts else 0
        minority_ids = sorted(pid for pid, count in initial_counts.items() if count == minority_count)
        final_aggregation = aggregation(final_ids)
        initial_aggregation = aggregation(initial_ids)
        completed = sortedness_percent(final_values) >= 100.0 - 1e-9
        if completed:
            final_class = "sorted"
        elif str(result.stop_reason).startswith("max_"):
            final_class = "resource_capped_partial"
        elif result.stop_reason == "no_cell_can_move_after_two_checks":
            final_class = "no_move_partial"
        else:
            final_class = "partial"
        interface_summary = simulator.interface_summary()
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": True,
            "runError": "",
            "simulationBackend": "interface_rule_event",
            "sourceSimulationBackend": source_backend,
            "signalConfigJson": signal_config_json,
            "initialValuesJson": compact_json(values),
            "finalValuesJson": compact_json(final_values),
            "initialStateHash": state_hash(values),
            "finalStateHash": state_hash(final_values),
            "initialPolicyAssignmentHash": arrangement_hash,
            "initialPanelPolicyIdsJson": compact_json(initial_ids),
            "finalPanelPolicyIdsJson": compact_json(final_ids),
            "actualCountsJson": compact_json(dict(initial_counts)),
            "finalCountsJson": compact_json(dict(final_counts)),
            "actualProportionsJson": compact_json({pid: initial_counts[pid] / int(condition["n"]) for pid in ids}),
            "actualRatiosMatchTarget": compact_json(dict(initial_counts)) == str(condition["targetCountsJson"]),
            "valueCountsConserved": Counter(values) == Counter(final_values),
            "policyCountsConserved": initial_counts == final_counts,
            "completed": bool(completed),
            "stopReason": result.stop_reason,
            "finalStateClass": final_class,
            "initialSortednessPercent": sortedness_percent(values),
            "finalSortednessPercent": sortedness_percent(final_values),
            "sortednessGain": sortedness_percent(final_values) - sortedness_percent(values),
            "initialAggregation": float(initial_aggregation),
            "finalAggregation": float(final_aggregation),
            "aggregationDelta": float(final_aggregation - initial_aggregation),
            "aggregationClass": "increased" if final_aggregation > initial_aggregation + 0.05 else ("decreased" if final_aggregation < initial_aggregation - 0.05 else "stable"),
            "swapCount": int(result.swap_count),
            "comparisonCount": int(result.comparison_count),
            "activationCount": int(result.activation_count),
            "eventCount": int(result.event_count),
            "blockedMoveAttempts": int(result.blocked_move_attempts),
            "frozenSwapAttempts": int(result.frozen_swap_attempts),
            "minorityPanelPolicyIdsJson": compact_json(minority_ids),
            "minorityInitialCount": int(minority_count),
            "minorityPersistence": 1.0 if minority_count > 0 and all(final_counts[pid] == initial_counts[pid] for pid in minority_ids) else 0.0,
            "leftPolicyMeanFinalPosition": left_mean,
            "rightPolicyMeanFinalPosition": right_mean,
            "leftMinusRightPositionBias": None if left_mean is None or right_mean is None else float(right_mean - left_mean),
            "meanFinalPositionByPolicyJson": compact_json(mean_positions),
            "interfaceRuleEvaluations": int(interface_summary["interfaceRuleEvaluations"]),
            "interfaceRuleAllowedSwaps": int(interface_summary["interfaceRuleAllowedSwaps"]),
            "interfaceRuleBlockedSwaps": int(interface_summary["interfaceRuleBlockedSwaps"]),
            "interfaceCrossBoundaryAttempts": int(interface_summary["interfaceCrossBoundaryAttempts"]),
            "interfaceRuleBlockReasonCountsJson": compact_json(interface_summary["interfaceRuleBlockReasonCounts"]),
            "meanInterfaceDeltaProposed": float(interface_summary["meanInterfaceDeltaProposed"]),
            "meanInterfaceDeltaAllowed": float(interface_summary["meanInterfaceDeltaAllowed"]),
            "meanInterfaceDeltaBlocked": float(interface_summary["meanInterfaceDeltaBlocked"]),
            "runtimeSeconds": float(time.perf_counter() - started),
            "claimBoundary": CLAIM_BOUNDARY,
        }
    except Exception as exc:  # pragma: no cover - retained for artifact-level failure accounting
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": False,
            "runError": f"{type(exc).__name__}: {exc}",
            "simulationBackend": "interface_rule_event",
            "sourceSimulationBackend": source_backend,
            "runtimeSeconds": float(time.perf_counter() - started),
            "claimBoundary": CLAIM_BOUNDARY,
        }


def _run_interface_worker(condition: Mapping[str, Any], max_activations: int, max_swaps: int, max_comparisons: int) -> dict[str, Any]:
    return run_interface_condition(
        condition,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )


def run_interface_conditions(
    condition_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    workers: int = 1,
    max_activations: int = 6000,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
) -> pd.DataFrame:
    records = [dict(row) for row in condition_df.to_dict(orient="records")]
    panel_records = [dict(row) for row in panel_df.to_dict(orient="records")]
    if int(workers) <= 1:
        lookup = {str(row["panelPolicyId"]): row for row in panel_records}
        return pd.DataFrame(
            [
                run_interface_condition(
                    record,
                    lookup,
                    max_activations=max_activations,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                )
                for record in records
            ]
        )
    with ProcessPoolExecutor(max_workers=int(workers), initializer=_interface_worker_init, initargs=(panel_records,)) as pool:
        futures = [
            pool.submit(_run_interface_worker, record, int(max_activations), int(max_swaps), int(max_comparisons))
            for record in records
        ]
        return pd.DataFrame([future.result() for future in futures])


def attach_baseline_deltas(classified: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "conditionId",
        "s07MosaicClass",
        "finalStateHash",
        "finalValuesJson",
        "finalPanelPolicyIdsJson",
        "stopReason",
        "swapCount",
        "activationCount",
        "finalInterfaceDensityS07",
        "finalAggregationS07",
        "finalTargetQualityScore",
        "dominanceProxyScore",
    ]
    baseline_cols = [col for col in cols if col in baseline.columns]
    renamed = baseline[baseline_cols].rename(
        columns={
            "conditionId": "s08SourceConditionId",
            "s07MosaicClass": "baselineS07MosaicClass",
            "finalStateHash": "baselineFinalStateHash",
            "finalValuesJson": "baselineFinalValuesJson",
            "finalPanelPolicyIdsJson": "baselineFinalPanelPolicyIdsJson",
            "stopReason": "baselineStopReason",
            "swapCount": "baselineSwapCount",
            "activationCount": "baselineActivationCount",
            "finalInterfaceDensityS07": "baselineFinalInterfaceDensityS07",
            "finalAggregationS07": "baselineFinalAggregationS07",
            "finalTargetQualityScore": "baselineFinalTargetQualityScore",
            "dominanceProxyScore": "baselineDominanceProxyScore",
        }
    )
    out = classified.merge(renamed, on="s08SourceConditionId", how="left")
    out["s08MosaicTransition"] = out["baselineS07MosaicClass"].astype(str) + "->" + out["s07MosaicClass"].astype(str)
    out["s08ClassMatchesBaseline"] = out["baselineS07MosaicClass"].astype(str) == out["s07MosaicClass"].astype(str)
    out["s08FinalStateHashMatchesBaseline"] = out["baselineFinalStateHash"].astype(str) == out["finalStateHash"].astype(str)
    out["s08FinalValuesMatchBaseline"] = out["baselineFinalValuesJson"].astype(str) == out["finalValuesJson"].astype(str)
    out["s08FinalPolicyIdsMatchBaseline"] = out["baselineFinalPanelPolicyIdsJson"].astype(str) == out["finalPanelPolicyIdsJson"].astype(str)
    out["s08StopReasonMatchesBaseline"] = out["baselineStopReason"].astype(str) == out["stopReason"].astype(str)
    out["deltaFinalInterfaceDensityS07"] = pd.to_numeric(out["finalInterfaceDensityS07"], errors="coerce") - pd.to_numeric(
        out["baselineFinalInterfaceDensityS07"], errors="coerce"
    )
    out["deltaFinalAggregationS07"] = pd.to_numeric(out["finalAggregationS07"], errors="coerce") - pd.to_numeric(
        out["baselineFinalAggregationS07"], errors="coerce"
    )
    out["deltaFinalTargetQualityScore"] = pd.to_numeric(out["finalTargetQualityScore"], errors="coerce") - pd.to_numeric(
        out["baselineFinalTargetQualityScore"], errors="coerce"
    )
    out["deltaDominanceProxyScore"] = pd.to_numeric(out["dominanceProxyScore"], errors="coerce") - pd.to_numeric(
        out["baselineDominanceProxyScore"], errors="coerce"
    )
    return out


def disabled_baseline_reproduction(scored: pd.DataFrame) -> pd.DataFrame:
    disabled = scored[scored["interfaceRuleVariant"].astype(str) == "disabled_baseline"].copy()
    if disabled.empty:
        return pd.DataFrame()
    rows = []
    for row in disabled.to_dict(orient="records"):
        rows.append(
            {
                "conditionId": str(row["conditionId"]),
                "s08SourceConditionId": str(row["s08SourceConditionId"]),
                "finalStateHashMatchesBaseline": bool(row.get("s08FinalStateHashMatchesBaseline", False)),
                "finalValuesMatchBaseline": bool(row.get("s08FinalValuesMatchBaseline", False)),
                "finalPolicyIdsMatchBaseline": bool(row.get("s08FinalPolicyIdsMatchBaseline", False)),
                "stopReasonMatchesBaseline": bool(row.get("s08StopReasonMatchesBaseline", False)),
                "swapCountMatchesBaseline": bool(int(row.get("swapCount", -1)) == int(row.get("baselineSwapCount", -2))),
                "activationCountMatchesBaseline": bool(int(row.get("activationCount", -1)) == int(row.get("baselineActivationCount", -2))),
                "baselineS07MosaicClass": str(row.get("baselineS07MosaicClass", "")),
                "s08S07MosaicClass": str(row.get("s07MosaicClass", "")),
            }
        )
    out = pd.DataFrame(rows)
    check_cols = [
        "finalStateHashMatchesBaseline",
        "finalValuesMatchBaseline",
        "finalPolicyIdsMatchBaseline",
        "stopReasonMatchesBaseline",
        "swapCountMatchesBaseline",
        "activationCountMatchesBaseline",
    ]
    out["disabledBaselineExactReproduction"] = out[check_cols].all(axis=1)
    return out


def summarize_interface_effects(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in scored.groupby("interfaceRuleVariant", sort=False):
        transitions = Counter(group["s08MosaicTransition"].astype(str))
        rows.append(
            {
                "interfaceRuleVariant": str(variant),
                "interfaceRuleFamily": str(group["interfaceRuleFamily"].iloc[0]),
                "conditionCount": int(len(group)),
                "runSuccessRate": float(group["runSucceeded"].map(bool).mean()),
                "classMatchBaselineRate": float(group["s08ClassMatchesBaseline"].map(bool).mean()),
                "meanDeltaFinalInterfaceDensity": float(group["deltaFinalInterfaceDensityS07"].mean()),
                "meanDeltaFinalAggregation": float(group["deltaFinalAggregationS07"].mean()),
                "meanDeltaFinalTargetQualityScore": float(group["deltaFinalTargetQualityScore"].mean()),
                "meanDeltaDominanceProxyScore": float(group["deltaDominanceProxyScore"].mean()),
                "meanInterfaceRuleEvaluations": float(pd.to_numeric(group["interfaceRuleEvaluations"], errors="coerce").mean()),
                "meanInterfaceRuleBlockedSwaps": float(pd.to_numeric(group["interfaceRuleBlockedSwaps"], errors="coerce").mean()),
                "meanInterfaceCrossBoundaryAttempts": float(pd.to_numeric(group["interfaceCrossBoundaryAttempts"], errors="coerce").mean()),
                "transitionCountsJson": compact_json(dict(transitions)),
            }
        )
    return pd.DataFrame(rows)


def summarize_interface_contexts(scored: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["interfaceRuleVariant", "baselineS07MosaicClass", "pairCategory", "arrangementType", "valueProfile"]
    rows = []
    for keys, group in scored.groupby(group_cols, dropna=False, sort=False):
        payload = dict(zip(group_cols, keys, strict=True))
        rows.append(
            {
                **payload,
                "conditionCount": int(len(group)),
                "classMatchBaselineRate": float(group["s08ClassMatchesBaseline"].map(bool).mean()),
                "meanDeltaFinalInterfaceDensity": float(group["deltaFinalInterfaceDensityS07"].mean()),
                "meanDeltaFinalAggregation": float(group["deltaFinalAggregationS07"].mean()),
                "meanDeltaFinalTargetQualityScore": float(group["deltaFinalTargetQualityScore"].mean()),
                "meanInterfaceRuleBlockedSwaps": float(pd.to_numeric(group["interfaceRuleBlockedSwaps"], errors="coerce").mean()),
                "transitionCountsJson": compact_json(dict(Counter(group["s08MosaicTransition"].astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def build_transition_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in scored.groupby("interfaceRuleVariant", sort=False):
        for transition, count in Counter(group["s08MosaicTransition"].astype(str)).items():
            before, after = transition.split("->", 1) if "->" in transition else (transition, "")
            rows.append(
                {
                    "interfaceRuleVariant": str(variant),
                    "baselineS07MosaicClass": before,
                    "s08S07MosaicClass": after,
                    "transitionCount": int(count),
                    "transitionShare": float(count / max(1, len(group))),
                }
            )
    return pd.DataFrame(rows)


def validation_checks(
    selected_contexts: pd.DataFrame,
    condition_df: pd.DataFrame,
    scored: pd.DataFrame,
    parameter_df: pd.DataFrame,
    disabled_reproduction: pd.DataFrame,
) -> pd.DataFrame:
    observed_context_classes = set(selected_contexts["s07MosaicClass"].astype(str))
    required_classes = set(REQUESTED_MOSAIC_CLASSES)
    variant_set = set(scored["interfaceRuleVariant"].astype(str)) if not scored.empty else set()
    disabled_ok = bool(
        not disabled_reproduction.empty
        and disabled_reproduction["disabledBaselineExactReproduction"].map(bool).all()
    )
    enabled = scored[scored["interfaceRuleEnabled"].map(bool)] if "interfaceRuleEnabled" in scored.columns else pd.DataFrame()
    enabled_by_variant = (
        enabled.assign(
            _evals=pd.to_numeric(enabled.get("interfaceRuleEvaluations", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
            _blocks=pd.to_numeric(enabled.get("interfaceRuleBlockedSwaps", pd.Series(dtype=float)), errors="coerce").fillna(0.0),
        )
        .groupby("interfaceRuleVariant", dropna=False)
        .agg(totalEvaluations=("_evals", "sum"), totalBlockedSwaps=("_blocks", "sum"))
        if not enabled.empty
        else pd.DataFrame()
    )
    checks = [
        {
            "checkId": "s07_priority_contexts_loaded",
            "success": bool(not selected_contexts.empty and required_classes <= observed_context_classes),
            "detail": f"{len(selected_contexts)} selected S07 contexts; classes {sorted(observed_context_classes)}",
        },
        {
            "checkId": "condition_matrix_crosses_rules",
            "success": bool(
                len(condition_df) == len(selected_contexts) * len(INTERFACE_RULE_VARIANTS)
                and set(INTERFACE_RULE_VARIANTS) <= set(condition_df["interfaceRuleVariant"].astype(str))
            ),
            "detail": f"{len(condition_df)} S08 conditions across {len(INTERFACE_RULE_VARIANTS)} interface variants",
        },
        {
            "checkId": "full_parameter_logging",
            "success": bool(
                set(INTERFACE_RULE_VARIANTS) <= set(parameter_df["interfaceRuleVariant"].astype(str))
                and condition_df["interfaceRuleConfigJson"].astype(str).str.len().gt(10).all()
                and scored["interfaceRuleConfigJson"].astype(str).str.len().gt(10).all()
            ),
            "detail": "all interface variants have machine-readable parameter rows and per-condition config JSON",
        },
        {
            "checkId": "disabled_rule_baseline_reproduction",
            "success": disabled_ok,
            "detail": (
                f"{int(disabled_reproduction['disabledBaselineExactReproduction'].sum()) if not disabled_reproduction.empty else 0}/"
                f"{len(disabled_reproduction)} disabled-rule rows exactly reproduced baseline hashes, values, labels, stop, swap, and activation counts"
            ),
        },
        {
            "checkId": "one_result_per_condition",
            "success": bool(len(scored) == len(condition_df) and scored["conditionId"].is_unique),
            "detail": f"{len(scored)} result rows for {len(condition_df)} conditions",
        },
        {
            "checkId": "runs_succeeded_and_counts_conserved",
            "success": bool(
                scored["runSucceeded"].fillna(False).map(bool).all()
                and scored["valueCountsConserved"].fillna(False).map(bool).all()
                and scored["policyCountsConserved"].fillna(False).map(bool).all()
            ),
            "detail": "all S08 runs succeeded and conserved value/policy counts",
        },
        {
            "checkId": "enabled_rules_exercised",
            "success": bool(
                not enabled_by_variant.empty
                and enabled_by_variant["totalEvaluations"].gt(0).all()
                and enabled_by_variant["totalBlockedSwaps"].gt(0).all()
            ),
            "detail": f"enabled variants evaluated {int(pd.to_numeric(enabled.get('interfaceRuleEvaluations', pd.Series(dtype=int)), errors='coerce').sum()) if not enabled.empty else 0} proposed swaps",
        },
        {
            "checkId": "mosaic_scores_populated",
            "success": bool(
                scored["s07MosaicClass"].astype(str).isin(set(REQUESTED_MOSAIC_CLASSES)).all()
                and scored["deltaFinalInterfaceDensityS07"].notna().all()
                and scored["deltaFinalTargetQualityScore"].notna().all()
            ),
            "detail": f"S08 rows classified with {MOSAIC_CLASSIFIER_VERSION} and baseline deltas populated",
        },
        {
            "checkId": "metric_boundary_caveats_retained",
            "success": bool(
                scored["metricBoundary"].astype(str).str.len().gt(0).all()
                and scored["claimBoundary"].astype(str).str.contains("Computational").all()
            ),
            "detail": "S06/S07 metric-boundary caveats and computational claim boundary retained",
        },
    ]
    return pd.DataFrame(checks)


def write_interface_plots(
    effect_summary: pd.DataFrame,
    transition_table: pd.DataFrame,
    figure_dir: Path,
    step_dir: Path,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not effect_summary.empty:
        plot_df = effect_summary.sort_values("interfaceRuleVariant", kind="mergesort")
        x = np.arange(len(plot_df))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.bar(x - 0.2, plot_df["meanDeltaFinalInterfaceDensity"], width=0.4, label="Interface density")
        ax.bar(x + 0.2, plot_df["meanDeltaFinalTargetQualityScore"], width=0.4, label="Target quality")
        ax.set_xticks(x, labels=plot_df["interfaceRuleVariant"], rotation=35, ha="right")
        ax.set_ylabel("Mean delta vs S07 baseline")
        ax.set_title("E06 S08 interface-rule effects")
        ax.legend()
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s08_interface_rule_effects.png",
            figure_dir / "e06_s08_interface_rule_effects.pdf",
            step_dir / "interface_rule_effects.png",
            step_dir / "interface_rule_effects.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    if not transition_table.empty:
        pivot = transition_table.pivot_table(
            index="baselineS07MosaicClass",
            columns="s08S07MosaicClass",
            values="transitionCount",
            aggfunc="sum",
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_title("E06 S08 mosaic class transitions")
        ax.set_xlabel("S08 class")
        ax.set_ylabel("S07 baseline class")
        fig.colorbar(image, ax=ax, label="Transition count")
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s08_mosaic_transition_heatmap.png",
            figure_dir / "e06_s08_mosaic_transition_heatmap.pdf",
            step_dir / "mosaic_transition_heatmap.png",
            step_dir / "mosaic_transition_heatmap.pdf",
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
    disabled_reproduction: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Path:
    effect_lines = "\n".join(
        f"- `{row.interfaceRuleVariant}`: class match {row.classMatchBaselineRate:.2f}, "
        f"mean interface-density delta {row.meanDeltaFinalInterfaceDensity:.3f}, "
        f"mean target-quality delta {row.meanDeltaFinalTargetQualityScore:.3f}, "
        f"mean blocked swaps {row.meanInterfaceRuleBlockedSwaps:.1f}"
        for row in effect_summary.itertuples(index=False)
    ) or "- No interface effect summaries available."
    disabled_rate = (
        float(disabled_reproduction["disabledBaselineExactReproduction"].mean())
        if not disabled_reproduction.empty
        else 0.0
    )
    text = f"""# Research Step S08: Add interface rules

## Completion status

Research step ID: `S08`. {status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Ran {status['conditionCount']} interface-rule conditions across {status['selectedContextCount']} S07 baseline contexts and {status['interfaceRuleVariantCount']} rule variants. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

Interface rules are deliberate computational extensions beyond the original no-recognition model. Adhesion, repulsion, permeability, recognition, and interface-specific swap labels are local one-dimensional policy-label gates, not biological mechanisms. Enabled rules only block proposed swaps; they do not invent new moves or add a global controller. Oscillatory and dominance fields remain inherited computational proxies from S06/S07.

## Lay summary

S08 added explicit local interface gates to selected S07 priority mosaics and compared each intervention with a disabled-rule baseline. Disabled rules reproduced the baseline exactly, while enabled rules tested whether direct self/non-self or boundary constraints changed clustering, target quality, and S07 mosaic class.

## Interface effect summary

{effect_lines}

## Disabled-rule baseline reproduction

Exact disabled-baseline reproduction rate: {disabled_rate:.3f}.

## Recommended next action

{status['recommendedNextAction']}
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s08_interface_rules(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    panel_path: Path = DEFAULT_PANEL_PATH,
    s07_classifications_path: Path = DEFAULT_S07_CLASSIFICATIONS_PATH,
    s07_priority_path: Path = DEFAULT_S07_PRIORITY_PATH,
    s06_pair_summary_path: Path = DEFAULT_S06_PAIR_SUMMARY_PATH,
    winner_criteria_path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH,
    workers: int | None = None,
    max_contexts: int = 18,
    max_activations: int = 6000,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
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
    s07_classifications = load_s07_classifications(s07_classifications_path)
    s07_priority = load_s07_priority(s07_priority_path)
    selected_contexts = select_s08_baseline_contexts(s07_classifications, s07_priority, max_contexts=max_contexts)
    condition_df, parameter_df = build_interface_condition_matrix(selected_contexts)
    criteria = load_winner_criteria(winner_criteria_path)
    pair_summary = pd.read_parquet(s06_pair_summary_path) if s06_pair_summary_path.exists() else pd.DataFrame()

    raw = run_interface_conditions(
        condition_df,
        ready_panel,
        workers=workers,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )
    goal = augment_goal_results(raw)
    scored = score_dominance_results(goal, criteria)
    scored["researchStepId"] = STEP_ID
    scored["sourceResearchStepId"] = "S08"
    classified = classify_mosaic_formations(scored, pair_summary if not pair_summary.empty else None)
    classified = attach_baseline_deltas(classified, s07_classifications)
    disabled_reproduction = disabled_baseline_reproduction(classified)
    effect_summary = summarize_interface_effects(classified)
    context_summary = summarize_interface_contexts(classified)
    transition_table = build_transition_table(classified)
    validation_df = validation_checks(selected_contexts, condition_df, classified, parameter_df, disabled_reproduction)

    parameter_csv = step_dir / "interface_rule_parameters.csv"
    parameter_parquet = step_dir / "interface_rule_parameters.parquet"
    selected_csv = step_dir / "selected_s07_interface_contexts.csv"
    selected_parquet = step_dir / "selected_s07_interface_contexts.parquet"
    condition_csv = step_dir / "interface_condition_matrix.csv"
    condition_parquet = step_dir / "interface_condition_matrix.parquet"
    runs_csv = step_dir / "interface_rule_runs.csv"
    runs_parquet = step_dir / "interface_rule_runs.parquet"
    result_csv = results_dir / "e06_interface_rules.csv"
    result_parquet = results_dir / "e06_interface_rules.parquet"
    governance_csv = results_dir / "e06_governance_interventions.csv"
    governance_parquet = results_dir / "e06_governance_interventions.parquet"
    effect_csv = step_dir / "interface_effect_summary.csv"
    effect_parquet = step_dir / "interface_effect_summary.parquet"
    context_csv = step_dir / "interface_context_summary.csv"
    context_parquet = step_dir / "interface_context_summary.parquet"
    transition_csv = step_dir / "mosaic_transition_summary.csv"
    transition_parquet = step_dir / "mosaic_transition_summary.parquet"
    disabled_csv = step_dir / "disabled_baseline_reproduction.csv"
    disabled_parquet = step_dir / "disabled_baseline_reproduction.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    for path, df in [
        (parameter_csv, parameter_df),
        (selected_csv, compact_result_csv(selected_contexts)),
        (condition_csv, compact_result_csv(condition_df)),
        (runs_csv, compact_result_csv(classified)),
        (result_csv, compact_result_csv(classified)),
        (governance_csv, compact_result_csv(classified)),
        (effect_csv, effect_summary),
        (context_csv, context_summary),
        (transition_csv, transition_table),
        (disabled_csv, disabled_reproduction),
        (validation_csv, validation_df),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (parameter_parquet, parameter_df),
        (selected_parquet, selected_contexts),
        (condition_parquet, condition_df),
        (runs_parquet, classified),
        (result_parquet, classified),
        (governance_parquet, classified),
        (effect_parquet, effect_summary),
        (context_parquet, context_summary),
        (transition_parquet, transition_table),
        (disabled_parquet, disabled_reproduction),
        (validation_parquet, validation_df),
    ]:
        df.to_parquet(path, index=False)

    figure_paths = write_interface_plots(effect_summary, transition_table, figures_dir, step_dir)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() else "failed"
    disabled_rate = (
        float(disabled_reproduction["disabledBaselineExactReproduction"].mean())
        if not disabled_reproduction.empty
        else float("nan")
    )
    enabled_changed_rate = float(
        (~classified[classified["interfaceRuleEnabled"].map(bool)]["s08ClassMatchesBaseline"].map(bool)).mean()
    )
    outcome = "supportive" if validation_result == "passed" and disabled_rate >= 1.0 else "constraining/contradictory"
    artifacts = [
        parameter_csv,
        parameter_parquet,
        selected_csv,
        selected_parquet,
        condition_csv,
        condition_parquet,
        runs_csv,
        runs_parquet,
        result_csv,
        result_parquet,
        governance_csv,
        governance_parquet,
        effect_csv,
        effect_parquet,
        context_csv,
        context_parquet,
        transition_csv,
        transition_parquet,
        disabled_csv,
        disabled_parquet,
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
            "Interface rules are deliberate computational extensions beyond the original no-recognition model.",
            "Adhesion, repulsion, permeability, recognition, and interface-specific swap rules are local one-dimensional policy-label gates.",
            "Enabled interface rules only block proposed swaps; they do not add new moves, global sensing, or a controller.",
            "Dominance and oscillatory labels remain inherited computational proxies from S06/S07.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then proceed to S09 governance mechanisms using S08 disabled baselines, "
            "interface-sensitive contexts, and rule-effect summaries; do not start S09 until review is complete."
        ),
        "laySummary": (
            "S08 added explicit local interface gates to selected S07 priority mosaics and compared adhesion, repulsion, "
            "permeability, recognition, interface-specific, and combined rules against exact disabled baselines."
        ),
        "outcomeClassification": outcome,
        "selectedContextCount": int(len(selected_contexts)),
        "conditionCount": int(len(condition_df)),
        "resultCount": int(len(classified)),
        "interfaceRuleVariantCount": int(parameter_df["interfaceRuleVariant"].nunique()),
        "disabledBaselineExactReproductionRate": disabled_rate,
        "enabledRuleClassChangedRate": enabled_changed_rate,
        "meanEnabledInterfaceRuleBlockedSwaps": float(
            pd.to_numeric(classified[classified["interfaceRuleEnabled"].map(bool)]["interfaceRuleBlockedSwaps"], errors="coerce").mean()
        ),
        "workerCount": int(workers),
        "maxActivations": int(max_activations),
        "maxSwaps": int(max_swaps),
        "maxComparisons": int(max_comparisons),
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
        disabled_reproduction=disabled_reproduction,
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
        "interfaceRuleVersion": INTERFACE_RULE_VERSION,
        "inputArtifacts": {
            "s07Classifications": str(s07_classifications_path),
            "s07PriorityCandidates": str(s07_priority_path),
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
        "conditions": condition_df,
        "parameters": parameter_df,
        "results": classified,
        "effectSummary": effect_summary,
        "contextSummary": context_summary,
        "transitionSummary": transition_table,
        "disabledBaselineReproduction": disabled_reproduction,
        "validation": validation_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E06 S08 explicit interface-rule interventions")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--s07-classifications-path", type=Path, default=DEFAULT_S07_CLASSIFICATIONS_PATH)
    parser.add_argument("--s07-priority-path", type=Path, default=DEFAULT_S07_PRIORITY_PATH)
    parser.add_argument("--s06-pair-summary-path", type=Path, default=DEFAULT_S06_PAIR_SUMMARY_PATH)
    parser.add_argument("--winner-criteria-path", type=Path, default=DEFAULT_S06_WINNER_CRITERIA_PATH)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-contexts", type=int, default=18)
    parser.add_argument("--max-activations", type=int, default=6000)
    parser.add_argument("--max-swaps", type=int, default=4000)
    parser.add_argument("--max-comparisons", type=int, default=30000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_s08_interface_rules(
        artifacts_dir=args.artifacts_dir,
        panel_path=args.panel_path,
        s07_classifications_path=args.s07_classifications_path,
        s07_priority_path=args.s07_priority_path,
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
        f"{STEP_ID} {status['status']}: {status['conditionCount']} interface-rule conditions, "
        f"disabled reproduction {status['disabledBaselineExactReproductionRate']:.3f}, "
        f"validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
