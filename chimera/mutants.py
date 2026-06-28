"""E06 S11 fixed-size computational mutant-clone experiments.

The "cancer-like" label in this module is only a computational analogy. The
simulator tracks small selfish policy-lineage clones in a fixed-size one-
dimensional local-policy array; it does not model biological cells, tumors,
division, growth, mutation biology, immune response, or tissue mechanics.
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
from morphospace import ProposedAction

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
    _candidate_positions,
)
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
from .panel import CLAIM_BOUNDARY, instantiate_panel_policy, json_ready


STEP_ID = "S11"
STEP_NUMBER = 11
MUTANT_VERSION = "e06_s11_mutant_clone_experiments.v1"
DEFAULT_S10_RESULTS_PATH = Path("/artifacts/results/e06_graft_experiments.parquet")
DEFAULT_S10_OUTCOME_SUMMARY_PATH = Path("/artifacts/research_steps/S10/graft_outcome_summary.parquet")
MUTANT_CLONE_SIZES: tuple[int, ...] = (3, 5, 10)
MUTANT_POSITION_NAMES: tuple[str, ...] = ("left_edge_clone", "center_clone", "s10_graft_site")
MUTANT_BEHAVIOR_VARIANTS: tuple[str, ...] = (
    "neutral_clone_control",
    "selfish_position_center",
    "selfish_aggregation",
    "selfish_disorder",
)
NO_MUTANT_CONTROL = "matched_no_mutant_control"
CANCER_ANALOGY_CAVEAT = (
    '"cancer-like mutant" is a computational analogy for selfish local-policy clone behavior; '
    "it is not a biological cancer model."
)
_PANEL_LOOKUP: dict[str, dict[str, Any]] = {}
_CONTROL_LOOKUP: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class MutantCloneConfig:
    """Serializable fixed-size computational clone behavior."""

    variant: str
    enabled: bool
    objective_family: str
    description: str
    division_enabled: bool = False
    lineage_tracking: str = "cell_id_fixed_size_lineage"
    information_access: str = "actor clone identity plus adjacent values and clone labels"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MUTANT_VERSION,
            "variant": self.variant,
            "enabled": bool(self.enabled),
            "objectiveFamily": self.objective_family,
            "divisionEnabled": bool(self.division_enabled),
            "lineageTracking": self.lineage_tracking,
            "informationAccess": self.information_access,
            "description": self.description,
            "analogyCaveat": CANCER_ANALOGY_CAVEAT,
            "claimBoundary": CLAIM_BOUNDARY,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | str | "MutantCloneConfig") -> "MutantCloneConfig":
        if isinstance(payload, MutantCloneConfig):
            return payload
        if isinstance(payload, str):
            try:
                parsed = parse_json_maybe(payload, None)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                payload = parsed
            else:
                return mutant_clone_config_by_variant(payload)
        return cls(
            variant=str(payload.get("variant", "neutral_clone_control")),
            enabled=bool(payload.get("enabled", False)),
            objective_family=str(payload.get("objectiveFamily", payload.get("objective_family", "neutral"))),
            description=str(payload.get("description", "")),
            division_enabled=bool(payload.get("divisionEnabled", payload.get("division_enabled", False))),
            lineage_tracking=str(payload.get("lineageTracking", payload.get("lineage_tracking", "cell_id_fixed_size_lineage"))),
            information_access=str(
                payload.get("informationAccess", payload.get("information_access", "actor clone identity plus adjacent values and clone labels"))
            ),
        )


def mutant_clone_configs() -> tuple[MutantCloneConfig, ...]:
    return (
        MutantCloneConfig(
            variant="neutral_clone_control",
            enabled=False,
            objective_family="neutral_donor_policy",
            description="Small donor-policy clone without selfish proposal override; controls for clone insertion alone.",
        ),
        MutantCloneConfig(
            variant="selfish_position_center",
            enabled=True,
            objective_family="self_position",
            description="Clone cells locally prefer adjacent swaps that move their own lineage toward the array center.",
        ),
        MutantCloneConfig(
            variant="selfish_aggregation",
            enabled=True,
            objective_family="self_aggregation",
            description="Clone cells locally prefer adjacent swaps that reduce clone/non-clone boundary count.",
        ),
        MutantCloneConfig(
            variant="selfish_disorder",
            enabled=True,
            objective_family="avoid_global_sortedness",
            description="Clone cells locally prefer adjacent swaps that reduce global increasing Sortedness.",
        ),
    )


def mutant_clone_config_by_variant(variant: str) -> MutantCloneConfig:
    lookup = {config.variant: config for config in mutant_clone_configs()}
    if variant not in lookup:
        raise ValueError(f"unknown S11 mutant clone variant: {variant}")
    return lookup[variant]


def mutant_clone_parameter_table(configs: Sequence[MutantCloneConfig] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(tuple(configs or mutant_clone_configs())):
        payload = config.to_dict()
        rows.append(
            {
                "mutantBehaviorIndex": int(index),
                "mutantBehaviorVariant": config.variant,
                "mutantBehaviorEnabled": bool(config.enabled),
                "mutantObjectiveFamily": config.objective_family,
                "divisionEnabled": bool(config.division_enabled),
                "lineageTracking": config.lineage_tracking,
                "informationAccess": config.information_access,
                "mutantCloneConfigJson": compact_json(payload),
                "description": config.description,
                "mutantVersion": MUTANT_VERSION,
                "analogyCaveat": CANCER_ANALOGY_CAVEAT,
                "claimBoundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def load_winner_criteria(path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return predefined_winner_criteria(load_boundary_caveats(DEFAULT_S05_BOUNDARY_CAVEATS_PATH))


def load_s10_results(path: Path = DEFAULT_S10_RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"S10 graft results missing: {path}")
    df = pd.read_parquet(path)
    required = {
        "conditionId",
        "matchedNoGraftConditionId",
        "s10ConditionKind",
        "graftApplied",
        "graftOutcomeClass",
        "hostPolicyId",
        "donorPolicyId",
        "governanceVariant",
        "interfaceRuleVariant",
        "goalMode",
        "policyIdsJson",
        "preGraftFinalValuesJson",
        "preGraftFinalPanelPolicyIdsJson",
        "postGraftInitialPanelPolicyIdsJson",
        "finalStateHash",
        "finalValuesJson",
        "finalPanelPolicyIdsJson",
        "finalTargetQualityScore",
        "minPolicyGoalScore",
        "s07MosaicClass",
        "runSucceeded",
        "metricBoundary",
        "n",
        "schedulerSeed",
        "tieBreakerSeed",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"S10 graft results missing required columns: {missing}")
    if df.empty:
        raise ValueError("S10 graft results are empty")
    return df


def select_s11_contexts(s10_results: pd.DataFrame, *, max_contexts: int = 8) -> pd.DataFrame:
    controls = s10_results[~s10_results["graftApplied"].map(bool)].copy()
    control_ids = set(controls["conditionId"].astype(str))
    grafts = s10_results[
        s10_results["graftApplied"].map(bool)
        & s10_results["runSucceeded"].map(bool)
        & s10_results["matchedNoGraftConditionId"].astype(str).isin(control_ids)
    ].copy()
    if grafts.empty:
        raise ValueError("S11 requires successful S10 graft rows with matched no-graft controls")
    grafts["s11ContextPriorityScore"] = (
        5.0 * pd.to_numeric(grafts.get("deltaVsNoGraftFinalTargetQualityScore", 0.0), errors="coerce").abs().fillna(0.0)
        + pd.to_numeric(grafts.get("deltaVsNoGraftMinPolicyGoalScore", 0.0), errors="coerce").abs().fillna(0.0) / 100.0
        + pd.to_numeric(grafts.get("graftDonorFinalAggregation", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(grafts.get("graftDonorFinalSpreadFraction", 0.0), errors="coerce").fillna(0.0)
        + grafts.get("graftChangedMosaicClass", pd.Series(False, index=grafts.index)).map(bool).astype(float)
    )
    selected_ids: list[str] = []
    reasons: dict[str, list[str]] = {}

    def add(row: Mapping[str, Any], reason: str) -> None:
        condition_id = str(row["conditionId"])
        if condition_id not in selected_ids:
            selected_ids.append(condition_id)
        reasons.setdefault(condition_id, []).append(reason)

    for outcome_class in ["segregated_patch", "absorbed_or_mixed", "equilibrated_mosaic", "host_reorganized_by_graft", "rejected_to_edge"]:
        rows = grafts[grafts["graftOutcomeClass"].astype(str) == outcome_class].sort_values(
            ["s11ContextPriorityScore", "conditionId"],
            ascending=[False, True],
            kind="mergesort",
        )
        if not rows.empty:
            add(rows.iloc[0], f"graft_outcome_anchor_{outcome_class}")
    for governance_variant in ["no_governance_control", "conflict_resolution_range1", "local_voting_range1", "organizer_global_upper_bound"]:
        rows = grafts[grafts["governanceVariant"].astype(str) == governance_variant].sort_values(
            ["s11ContextPriorityScore", "conditionId"],
            ascending=[False, True],
            kind="mergesort",
        )
        if not rows.empty:
            add(rows.iloc[0], f"governance_anchor_{governance_variant}")
    for interface_variant in ["disabled_baseline", "combined_self_boundary"]:
        rows = grafts[grafts["interfaceRuleVariant"].astype(str) == interface_variant].sort_values(
            ["s11ContextPriorityScore", "conditionId"],
            ascending=[False, True],
            kind="mergesort",
        )
        if not rows.empty:
            add(rows.iloc[0], f"interface_anchor_{interface_variant}")
    remaining = grafts.sort_values(["s11ContextPriorityScore", "conditionId"], ascending=[False, True], kind="mergesort")
    for row in remaining.to_dict(orient="records"):
        if len(selected_ids) >= int(max_contexts):
            break
        add(row, "high_s10_mutant_stress_context")

    selected = grafts[grafts["conditionId"].astype(str).isin(selected_ids)].copy()
    selected["_order"] = selected["conditionId"].map({condition_id: index for index, condition_id in enumerate(selected_ids)})
    selected = selected.sort_values("_order", kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)
    control_cols = [
        "conditionId",
        "finalStateHash",
        "finalValuesJson",
        "finalPanelPolicyIdsJson",
        "finalTargetQualityScore",
        "minPolicyGoalScore",
        "meanPolicyGoalScore",
        "finalSortednessPercent",
        "s07MosaicClass",
        "stopReason",
        "swapCount",
        "activationCount",
    ]
    control_lookup = controls[[col for col in control_cols if col in controls.columns]].rename(
        columns={
            "conditionId": "matchedNoGraftConditionId",
            "finalStateHash": "s10MatchedNoGraftFinalStateHash",
            "finalValuesJson": "s10MatchedNoGraftFinalValuesJson",
            "finalPanelPolicyIdsJson": "s10MatchedNoGraftFinalPanelPolicyIdsJson",
            "finalTargetQualityScore": "s10MatchedNoGraftFinalTargetQualityScore",
            "minPolicyGoalScore": "s10MatchedNoGraftMinPolicyGoalScore",
            "meanPolicyGoalScore": "s10MatchedNoGraftMeanPolicyGoalScore",
            "finalSortednessPercent": "s10MatchedNoGraftFinalSortednessPercent",
            "s07MosaicClass": "s10MatchedNoGraftMosaicClass",
            "stopReason": "s10MatchedNoGraftStopReason",
            "swapCount": "s10MatchedNoGraftSwapCount",
            "activationCount": "s10MatchedNoGraftActivationCount",
        }
    )
    selected = selected.merge(control_lookup, on="matchedNoGraftConditionId", how="left")
    selected.insert(0, "s11ContextIndex", np.arange(len(selected), dtype=int))
    selected["s11SourceS10GraftConditionId"] = selected["conditionId"].astype(str)
    selected["s11MatchedNoGraftConditionId"] = selected["matchedNoGraftConditionId"].astype(str)
    selected["s11SelectionReason"] = selected["conditionId"].map(lambda value: "+".join(sorted(set(reasons.get(str(value), [])))))
    return selected


def mutant_clone_positions(
    n: int,
    clone_size: int,
    position_name: str,
    *,
    source_start: int | None = None,
) -> list[int]:
    n = int(n)
    clone_size = int(clone_size)
    if clone_size <= 0 or clone_size > n:
        raise ValueError("clone_size must be in [1, n]")
    if position_name == "left_edge_clone":
        start = 0
    elif position_name == "center_clone":
        start = (n - clone_size) // 2
    elif position_name == "s10_graft_site":
        raw_start = int(source_start) if source_start is not None and int(source_start) >= 0 else (n - clone_size) // 2
        start = min(max(0, raw_start), n - clone_size)
    else:
        raise ValueError(f"unknown mutant clone position: {position_name}")
    return list(range(start, start + clone_size))


def _counts_json(values: Sequence[str]) -> str:
    return compact_json(dict(Counter(str(value) for value in values)))


def _label_assignment(policy_ids: Sequence[str], assigned_ids: Sequence[str]) -> list[int]:
    lookup = {str(pid): index for index, pid in enumerate(policy_ids)}
    return [int(lookup[str(pid)]) for pid in assigned_ids]


def _control_condition_id(context: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(f"{context['conditionId']}|s11|no_mutant".encode("utf-8")).hexdigest()[:12]
    return f"s11_control_{digest}"


def _mutant_condition_id(context: Mapping[str, Any], clone_size: int, position_name: str, behavior_variant: str) -> str:
    digest = hashlib.sha256(
        f"{context['conditionId']}|s11|{clone_size}|{position_name}|{behavior_variant}".encode("utf-8")
    ).hexdigest()[:12]
    return f"s11_mutant_{digest}"


def build_mutant_condition_matrix(
    selected_contexts: pd.DataFrame,
    *,
    clone_sizes: Sequence[int] = MUTANT_CLONE_SIZES,
    position_names: Sequence[str] = MUTANT_POSITION_NAMES,
    behavior_variants: Sequence[str] = MUTANT_BEHAVIOR_VARIANTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    parameter_df = mutant_clone_parameter_table([mutant_clone_config_by_variant(variant) for variant in behavior_variants])
    for context in selected_contexts.to_dict(orient="records"):
        n = int(context["n"])
        host_policy_id = str(context["hostPolicyId"])
        donor_policy_id = str(context["donorPolicyId"])
        policy_ids = [host_policy_id, donor_policy_id]
        base_values = [int(value) for value in parse_json_maybe(context["preGraftFinalValuesJson"], [])]
        if len(base_values) != n:
            raise ValueError("S10 preGraftFinalValuesJson length does not match n")
        host_assignment = [host_policy_id] * n
        specs = policy_goal_specs(policy_ids, str(context["goalMode"]))
        reverse = reverse_directions_for_assignment(host_assignment, specs)
        goal_meta = condition_goal_metadata(policy_ids, str(context["goalMode"]))
        governance_config = GovernanceConfig.from_payload(str(context["governanceVariant"]))
        interface_config = interface_rule_config_by_variant(str(context["interfaceRuleVariant"]))
        control_id = _control_condition_id(context)
        common = {
            **{key: json_ready(value) for key, value in context.items()},
            "conditionId": control_id,
            "researchStepId": STEP_ID,
            "sourceResearchStepId": "S10",
            "implementationPrefix": "e06_s11",
            "mutantVersion": MUTANT_VERSION,
            "s11ConditionKind": NO_MUTANT_CONTROL,
            "matchedNoMutantConditionId": control_id,
            "simulationConditionId": str(context["matchedNoGraftConditionId"]),
            "s11SourceS10GraftConditionId": str(context["conditionId"]),
            "s11MatchedNoGraftConditionId": str(context["matchedNoGraftConditionId"]),
            "hostPolicyId": host_policy_id,
            "donorPolicyId": donor_policy_id,
            "policyIdsJson": compact_json(policy_ids),
            "initialValuesJson": compact_json(base_values),
            "initialPanelPolicyIdsJson": compact_json(host_assignment),
            "initialNumericLabelsJson": compact_json(_label_assignment(policy_ids, host_assignment)),
            "targetCountsJson": _counts_json(host_assignment),
            "targetProportionsJson": compact_json({host_policy_id: 1.0, donor_policy_id: 0.0}),
            "goalCompatibilityClass": str(goal_meta["compatibilityClass"]),
            "conditionGoalMetadataJson": compact_json(goal_meta),
            "policyGoalMetadataJson": compact_json(specs),
            "goalReverseDirectionsJson": compact_json(reverse),
            "goalDirectionByPolicyJson": compact_json({pid: specs[pid]["targetDirection"] for pid in policy_ids}),
            "goalBehaviorDirectionByPolicyJson": compact_json({pid: specs[pid]["behaviorDirection"] for pid in policy_ids}),
            "governanceVersion": GOVERNANCE_VERSION,
            "governanceConfigJson": compact_json(governance_config.to_dict()),
            "governanceFamily": governance_config.family,
            "influenceRange": int(governance_config.influence_range),
            "informationAccess": governance_config.information_access,
            "localInformationOnly": bool(governance_config.local_information_only),
            "usesGlobalState": bool(governance_config.uses_global_state),
            "usesTargetMap": bool(governance_config.uses_target_map),
            "usesOrganizer": bool(governance_config.uses_organizer),
            "broadControlLike": bool(governance_config.broad_control_like),
            "interfaceRuleVersion": INTERFACE_RULE_VERSION,
            "interfaceRuleConfigJson": compact_json(interface_config.to_dict()),
            "interfaceRuleFamily": interface_config.rule_family,
            "interfaceRuleEnabled": bool(interface_config.enabled),
            "mutantBehaviorVariant": NO_MUTANT_CONTROL,
            "mutantObjectiveFamily": "none",
            "mutantCloneSize": 0,
            "mutantClonePositionName": "none",
            "mutantCloneStartIndex": -1,
            "mutantCloneEndExclusive": -1,
            "mutantLineageCellIdsJson": compact_json([]),
            "mutantInitialPositionsJson": compact_json([]),
            "mutantCloneConfigJson": compact_json(
                {
                    "schema": MUTANT_VERSION,
                    "variant": NO_MUTANT_CONTROL,
                    "enabled": False,
                    "divisionEnabled": False,
                    "analogyCaveat": CANCER_ANALOGY_CAVEAT,
                    "claimBoundary": CLAIM_BOUNDARY,
                }
            ),
            "divisionEnabled": False,
            "lineageTracking": "cell_id_fixed_size_lineage",
            "analogyCaveat": CANCER_ANALOGY_CAVEAT,
            "conditionHash": hashlib.sha256(f"{control_id}|{context['conditionId']}".encode("utf-8")).hexdigest()[:16],
        }
        rows.append(common)
        source_start = int(context.get("graftStartIndex", -1)) if pd.notna(context.get("graftStartIndex", -1)) else -1
        for clone_size in clone_sizes:
            for position_name in position_names:
                positions = mutant_clone_positions(n, int(clone_size), str(position_name), source_start=source_start)
                start = min(positions)
                end = max(positions) + 1
                for behavior_variant in behavior_variants:
                    config = mutant_clone_config_by_variant(str(behavior_variant))
                    assignment = list(host_assignment)
                    for pos in positions:
                        assignment[int(pos)] = donor_policy_id
                    reverse = reverse_directions_for_assignment(assignment, specs)
                    condition_id = _mutant_condition_id(context, int(clone_size), str(position_name), str(behavior_variant))
                    descriptor = {
                        "schema": MUTANT_VERSION,
                        "conditionId": condition_id,
                        "matchedNoMutantConditionId": control_id,
                        "sourceS10GraftConditionId": str(context["conditionId"]),
                        "sourceS10GraftOutcomeClass": str(context["graftOutcomeClass"]),
                        "matchedS10NoGraftConditionId": str(context["matchedNoGraftConditionId"]),
                        "hostPolicyId": host_policy_id,
                        "donorPolicyId": donor_policy_id,
                        "mutantBehaviorVariant": config.variant,
                        "mutantObjectiveFamily": config.objective_family,
                        "cloneSize": int(clone_size),
                        "clonePositionName": str(position_name),
                        "cloneStartIndex": int(start),
                        "cloneEndExclusive": int(end),
                        "initialLineageCellIds": positions,
                        "initialLineagePositions": positions,
                        "divisionEnabled": False,
                        "lineageTracking": config.lineage_tracking,
                        "analogyCaveat": CANCER_ANALOGY_CAVEAT,
                        "claimBoundary": CLAIM_BOUNDARY,
                    }
                    rows.append(
                        {
                            **common,
                            "conditionId": condition_id,
                            "s11ConditionKind": "mutant_clone",
                            "simulationConditionId": condition_id,
                            "mutantBehaviorVariant": config.variant,
                            "mutantObjectiveFamily": config.objective_family,
                            "mutantCloneSize": int(clone_size),
                            "mutantClonePositionName": str(position_name),
                            "mutantCloneStartIndex": int(start),
                            "mutantCloneEndExclusive": int(end),
                            "mutantLineageCellIdsJson": compact_json(positions),
                            "mutantInitialPositionsJson": compact_json(positions),
                            "mutantCloneConfigJson": compact_json(config.to_dict()),
                            "initialPanelPolicyIdsJson": compact_json(assignment),
                            "initialNumericLabelsJson": compact_json(_label_assignment(policy_ids, assignment)),
                            "targetCountsJson": _counts_json(assignment),
                            "targetProportionsJson": compact_json(
                                {pid: Counter(assignment).get(pid, 0) / n for pid in policy_ids}
                            ),
                            "goalReverseDirectionsJson": compact_json(reverse),
                            "divisionEnabled": False,
                            "lineageTracking": config.lineage_tracking,
                            "mutantCloneDescriptorJson": compact_json(descriptor),
                            "conditionHash": hashlib.sha256(compact_json(descriptor).encode("utf-8")).hexdigest()[:16],
                        }
                    )
                    lineage_rows.append(descriptor)
    return pd.DataFrame(rows).reset_index(drop=True), parameter_df.reset_index(drop=True), pd.DataFrame(lineage_rows).reset_index(drop=True)


def _swap_copy(values: Sequence[Any], a: int, b: int) -> list[Any]:
    out = list(values)
    out[int(a)], out[int(b)] = out[int(b)], out[int(a)]
    return out


def _clone_positions_from_cells(cells: Sequence[Any], clone_ids: set[int]) -> list[int]:
    return [index for index, cell in enumerate(cells) if int(cell.cell_id) in clone_ids]


def _clone_boundary_count_from_positions(n: int, clone_positions: Sequence[int]) -> int:
    clone_set = set(int(pos) for pos in clone_positions)
    return int(sum(1 for left in range(max(0, int(n) - 1)) if (left in clone_set) != ((left + 1) in clone_set)))


def _lineage_aggregation(n: int, positions: Sequence[int]) -> float:
    labels = ["clone" if index in set(int(pos) for pos in positions) else "host" for index in range(int(n))]
    return float(aggregation(labels))


class MutantCloneEventSimulator(GovernanceEventSimulator):
    """Governance simulator with an optional fixed-size selfish clone hook."""

    def __init__(
        self,
        *args: Any,
        mutant_clone_config: MutantCloneConfig | Mapping[str, Any] | str | None = None,
        mutant_cell_ids: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self.mutant_clone_config = MutantCloneConfig.from_payload(mutant_clone_config or "neutral_clone_control")
        self.mutant_cell_ids = set(int(cell_id) for cell_id in (mutant_cell_ids or []))
        self.mutant_evaluations = 0
        self.mutant_selfish_proposals = 0
        self.mutant_injected_swaps = 0
        self.mutant_redirected_swaps = 0
        self.mutant_reason_counts: Counter[str] = Counter()
        super().__init__(*args, **kwargs)

    def _clone_boundaries_after_swap(self, actor_pos: int, target_pos: int) -> int:
        clone_positions = _clone_positions_from_cells(self.cells, self.mutant_cell_ids)
        after_positions = _swap_copy(
            [1 if index in set(clone_positions) else 0 for index in range(len(self.cells))],
            int(actor_pos),
            int(target_pos),
        )
        return int(sum(1 for left, right in zip(after_positions, after_positions[1:]) if int(left) != int(right)))

    def _score_mutant_candidate(self, actor_pos: int, target_pos: int) -> float:
        if not self._target_status_allows_policy_check(int(target_pos)):
            return 0.0
        config = self.mutant_clone_config
        family = config.objective_family
        if family == "self_position":
            center = (len(self.cells) - 1) / 2.0
            before = abs(int(actor_pos) - center)
            after = abs(int(target_pos) - center)
            return float(before - after)
        if family == "self_aggregation":
            before = _clone_boundary_count_from_positions(len(self.cells), _clone_positions_from_cells(self.cells, self.mutant_cell_ids))
            after = self._clone_boundaries_after_swap(actor_pos, target_pos)
            return float(before - after)
        if family == "avoid_global_sortedness":
            values = self.current_values()
            before = sortedness_percent(values, direction="increasing")
            after = sortedness_percent(_swap_copy(values, actor_pos, target_pos), direction="increasing")
            return float(before - after)
        return 0.0

    def _mutant_adjust_proposal(self, actor_pos: int, proposed: ProposedAction) -> ProposedAction:
        actor = self.cells[int(actor_pos)]
        config = self.mutant_clone_config
        if not config.enabled or int(actor.cell_id) not in self.mutant_cell_ids:
            return proposed
        self.mutant_evaluations += 1
        best_target: int | None = None
        best_score = 0.0
        for target in _candidate_positions(actor_pos, len(self.cells)):
            if not self._target_status_allows_policy_check(target):
                continue
            score = self._score_mutant_candidate(actor_pos, target)
            if score > best_score + 1e-12:
                best_target = int(target)
                best_score = float(score)
        current_score = 0.0
        if proposed.action == "swap" and proposed.target_position is not None and self._target_in_bounds(int(proposed.target_position)):
            current_score = self._score_mutant_candidate(actor_pos, int(proposed.target_position))
        if best_target is None or best_score <= max(0.0, current_score) + 1e-12:
            return proposed
        self.mutant_selfish_proposals += 1
        if proposed.action == "swap":
            self.mutant_redirected_swaps += 1
            reason = "mutant_redirected_swap"
        else:
            self.mutant_injected_swaps += 1
            reason = "mutant_injected_swap"
        self.mutant_reason_counts[f"{config.variant}:{reason}"] += 1
        return ProposedAction.swap(
            int(best_target),
            comparison_delta=int(proposed.comparison_delta),
            state_updates=proposed.state_updates,
            swapped_reason=f"{reason}:{config.variant}",
            blocked_reason=f"mutant_swap_blocked:{config.variant}",
        )

    def _governance_adjust_proposal(self, actor_pos: int, proposed: ProposedAction) -> ProposedAction:
        mutant_proposed = self._mutant_adjust_proposal(actor_pos, proposed)
        return super()._governance_adjust_proposal(actor_pos, mutant_proposed)

    def mutant_summary(self) -> dict[str, Any]:
        final_positions = _clone_positions_from_cells(self.cells, self.mutant_cell_ids)
        return {
            "mutantCloneConfig": self.mutant_clone_config.to_dict(),
            "mutantLineageCellIds": sorted(self.mutant_cell_ids),
            "mutantEvaluations": int(self.mutant_evaluations),
            "mutantSelfishProposals": int(self.mutant_selfish_proposals),
            "mutantInjectedSwaps": int(self.mutant_injected_swaps),
            "mutantRedirectedSwaps": int(self.mutant_redirected_swaps),
            "mutantReasonCounts": dict(self.mutant_reason_counts),
            "mutantFinalPositions": final_positions,
        }


def _run_simulation(
    *,
    values: Sequence[int],
    assigned_ids: Sequence[str],
    policy_ids: Sequence[str],
    reverse_directions: Sequence[bool],
    panel_lookup: Mapping[str, Mapping[str, Any]],
    condition_id: str,
    simulation_condition_id: str,
    governance_variant: str,
    interface_variant: str,
    mutant_config: MutantCloneConfig,
    mutant_cell_ids: Sequence[int],
    scheduler_seed: int,
    tie_breaker_seed: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[MutantCloneEventSimulator, Any, str]:
    label_by_id = {str(pid): index for index, pid in enumerate(policy_ids)}
    numeric_labels = [int(label_by_id[str(pid)]) for pid in assigned_ids]
    policy_cache = {pid: instantiate_panel_policy(panel_lookup[str(pid)]) for pid in policy_ids}
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
        condition_id=str(simulation_condition_id),
        research_step_id=STEP_ID,
        signal_config=signal_config,
        auto_wrap_policies=False,
        trace_signal_activations=False,
        trace_memory_activations=False,
        interface_rule_config=str(interface_variant),
        governance_config=str(governance_variant),
        mutant_clone_config=mutant_config,
        mutant_cell_ids=mutant_cell_ids,
        implementation="e06_s11_mutant_clone",
    )
    result = simulator.run(
        max_activations=int(max_activations),
        max_swaps=int(max_swaps),
        max_comparisons=int(max_comparisons),
        no_move_check_interval=len(values),
    )
    simulator.condition_id = str(condition_id)
    return simulator, result, source_backend


def run_mutant_condition(
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
    assigned = [str(item) for item in parse_json_maybe(condition["initialPanelPolicyIdsJson"], [])]
    values = [int(item) for item in parse_json_maybe(condition["initialValuesJson"], [])]
    reverse = [bool(item) for item in parse_json_maybe(condition["goalReverseDirectionsJson"], [])]
    mutant_cell_ids = [int(item) for item in parse_json_maybe(condition.get("mutantLineageCellIdsJson", "[]"), [])]
    mutant_config = MutantCloneConfig.from_payload(condition.get("mutantCloneConfigJson", "neutral_clone_control"))
    try:
        simulator, result, source_backend = _run_simulation(
            values=values,
            assigned_ids=assigned,
            policy_ids=policy_ids,
            reverse_directions=reverse,
            panel_lookup=panel_lookup,
            condition_id=str(condition["conditionId"]),
            simulation_condition_id=str(condition.get("simulationConditionId", condition["conditionId"])),
            governance_variant=str(condition["governanceVariant"]),
            interface_variant=str(condition["interfaceRuleVariant"]),
            mutant_config=mutant_config,
            mutant_cell_ids=mutant_cell_ids,
            scheduler_seed=int(condition["schedulerSeed"]) + 1009,
            tie_breaker_seed=int(condition["tieBreakerSeed"]) + 1009,
            max_activations=max_activations,
            max_swaps=max_swaps,
            max_comparisons=max_comparisons,
        )
        final_values = simulator.current_values()
        final_labels = [int(cell.label) for cell in simulator.cells]
        final_ids = [policy_ids[label] for label in final_labels]
        initial_counts = Counter(assigned)
        final_counts = Counter(final_ids)
        final_clone_positions = _clone_positions_from_cells(simulator.cells, set(mutant_cell_ids))
        initial_clone_positions = [int(pos) for pos in parse_json_maybe(condition.get("mutantInitialPositionsJson", "[]"), [])]
        clone_size_initial = int(len(mutant_cell_ids))
        clone_size_final = int(len(final_clone_positions))
        clone_spread = (
            (max(final_clone_positions) - min(final_clone_positions) + 1) / max(1, len(final_ids))
            if final_clone_positions
            else 0.0
        )
        clone_mean = float(np.mean(final_clone_positions) / max(1, len(final_ids) - 1)) if final_clone_positions else float("nan")
        clone_initial_mean = float(np.mean(initial_clone_positions) / max(1, len(final_ids) - 1)) if initial_clone_positions else float("nan")
        clone_boundary_count = _clone_boundary_count_from_positions(len(final_ids), final_clone_positions)
        clone_interface_density = clone_boundary_count / max(1, len(final_ids) - 1)
        clone_final_aggregation = _lineage_aggregation(len(final_ids), final_clone_positions) if clone_size_final else float("nan")
        clone_initial_aggregation = _lineage_aggregation(len(assigned), initial_clone_positions) if clone_size_initial else float("nan")
        interface_summary = simulator.interface_summary()
        governance_summary = simulator.governance_summary()
        mutant_summary = simulator.mutant_summary()
        completed = sortedness_percent(final_values) >= 100.0 - 1e-9
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": True,
            "runError": "",
            "simulationBackend": "mutant_clone_governance_event",
            "sourceSimulationBackend": source_backend,
            "finalValuesJson": compact_json(final_values),
            "initialStateHash": state_hash(values),
            "finalStateHash": state_hash(final_values),
            "finalPanelPolicyIdsJson": compact_json(final_ids),
            "actualCountsJson": compact_json(dict(initial_counts)),
            "finalCountsJson": compact_json(dict(final_counts)),
            "actualRatiosMatchTarget": compact_json(dict(initial_counts)) == str(condition["targetCountsJson"]),
            "valueCountsConserved": Counter(values) == Counter(final_values),
            "policyCountsConserved": initial_counts == final_counts,
            "completed": bool(completed),
            "stopReason": result.stop_reason,
            "finalStateClass": "sorted" if completed else ("resource_capped_partial" if str(result.stop_reason).startswith("max_") else "partial"),
            "initialSortednessPercent": sortedness_percent(values),
            "finalSortednessPercent": sortedness_percent(final_values),
            "sortednessGain": sortedness_percent(final_values) - sortedness_percent(values),
            "initialAggregation": aggregation(assigned),
            "finalAggregation": aggregation(final_ids),
            "aggregationDelta": aggregation(final_ids) - aggregation(assigned),
            "swapCount": int(result.swap_count),
            "comparisonCount": int(result.comparison_count),
            "activationCount": int(result.activation_count),
            "eventCount": int(result.event_count),
            "blockedMoveAttempts": int(result.blocked_move_attempts),
            "frozenSwapAttempts": int(result.frozen_swap_attempts),
            "mutantCloneInitialCount": clone_size_initial,
            "mutantCloneFinalCount": clone_size_final,
            "mutantCloneSizeConserved": bool(clone_size_initial == clone_size_final),
            "mutantCloneFinalPositionsJson": compact_json(final_clone_positions),
            "mutantCloneInitialMeanPosition": clone_initial_mean,
            "mutantCloneFinalMeanPosition": clone_mean,
            "mutantCloneMeanPositionShift": float(clone_mean - clone_initial_mean) if np.isfinite(clone_mean) and np.isfinite(clone_initial_mean) else float("nan"),
            "mutantCloneInitialAggregation": float(clone_initial_aggregation),
            "mutantCloneFinalAggregation": float(clone_final_aggregation),
            "mutantCloneAggregationDelta": float(clone_final_aggregation - clone_initial_aggregation)
            if np.isfinite(clone_final_aggregation) and np.isfinite(clone_initial_aggregation)
            else float("nan"),
            "mutantCloneBoundaryCount": int(clone_boundary_count),
            "mutantCloneInterfaceDensity": float(clone_interface_density),
            "mutantCloneSpreadFraction": float(clone_spread),
            "mutantEvaluations": int(mutant_summary["mutantEvaluations"]),
            "mutantSelfishProposals": int(mutant_summary["mutantSelfishProposals"]),
            "mutantInjectedSwaps": int(mutant_summary["mutantInjectedSwaps"]),
            "mutantRedirectedSwaps": int(mutant_summary["mutantRedirectedSwaps"]),
            "mutantReasonCountsJson": compact_json(mutant_summary["mutantReasonCounts"]),
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
            "analogyCaveat": CANCER_ANALOGY_CAVEAT,
            "claimBoundary": CLAIM_BOUNDARY,
        }
    except Exception as exc:  # pragma: no cover - artifact-level failure accounting
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": False,
            "runError": f"{type(exc).__name__}: {exc}",
            "simulationBackend": "mutant_clone_governance_event",
            "runtimeSeconds": float(time.perf_counter() - started),
            "analogyCaveat": CANCER_ANALOGY_CAVEAT,
            "claimBoundary": CLAIM_BOUNDARY,
        }


def _worker_init(panel_records: Sequence[Mapping[str, Any]]) -> None:
    global _PANEL_LOOKUP
    _PANEL_LOOKUP = {str(row["panelPolicyId"]): dict(row) for row in panel_records}


def _run_worker(record: Mapping[str, Any], max_activations: int, max_swaps: int, max_comparisons: int) -> dict[str, Any]:
    return run_mutant_condition(
        record,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )


def run_mutant_conditions(
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
                run_mutant_condition(
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


def attach_matched_no_mutant_deltas(scored: pd.DataFrame) -> pd.DataFrame:
    controls = scored[scored["s11ConditionKind"].astype(str) == NO_MUTANT_CONTROL].copy()
    cols = [
        "conditionId",
        "finalStateHash",
        "finalValuesJson",
        "finalPanelPolicyIdsJson",
        "s07MosaicClass",
        "finalTargetQualityScore",
        "minPolicyGoalScore",
        "meanPolicyGoalScore",
        "finalSortednessPercent",
        "dominanceProxyScore",
        "stopReason",
        "swapCount",
        "activationCount",
    ]
    renamed = controls[[col for col in cols if col in controls.columns]].rename(
        columns={
            "conditionId": "matchedNoMutantConditionId",
            "finalStateHash": "noMutantFinalStateHash",
            "finalValuesJson": "noMutantFinalValuesJson",
            "finalPanelPolicyIdsJson": "noMutantFinalPanelPolicyIdsJson",
            "s07MosaicClass": "noMutantMosaicClass",
            "finalTargetQualityScore": "noMutantFinalTargetQualityScore",
            "minPolicyGoalScore": "noMutantMinPolicyGoalScore",
            "meanPolicyGoalScore": "noMutantMeanPolicyGoalScore",
            "finalSortednessPercent": "noMutantFinalSortednessPercent",
            "dominanceProxyScore": "noMutantDominanceProxyScore",
            "stopReason": "noMutantStopReason",
            "swapCount": "noMutantSwapCount",
            "activationCount": "noMutantActivationCount",
        }
    )
    out = scored.merge(renamed, on="matchedNoMutantConditionId", how="left")
    out["deltaVsNoMutantFinalTargetQualityScore"] = pd.to_numeric(out["finalTargetQualityScore"], errors="coerce") - pd.to_numeric(
        out["noMutantFinalTargetQualityScore"], errors="coerce"
    )
    out["deltaVsNoMutantMinPolicyGoalScore"] = pd.to_numeric(out["minPolicyGoalScore"], errors="coerce") - pd.to_numeric(
        out["noMutantMinPolicyGoalScore"], errors="coerce"
    )
    out["deltaVsNoMutantMeanPolicyGoalScore"] = pd.to_numeric(out["meanPolicyGoalScore"], errors="coerce") - pd.to_numeric(
        out["noMutantMeanPolicyGoalScore"], errors="coerce"
    )
    out["deltaVsNoMutantFinalSortednessPercent"] = pd.to_numeric(out["finalSortednessPercent"], errors="coerce") - pd.to_numeric(
        out["noMutantFinalSortednessPercent"], errors="coerce"
    )
    out["mutantChangedMosaicClass"] = out["s07MosaicClass"].astype(str) != out["noMutantMosaicClass"].astype(str)
    return out


def classify_mutant_outcomes(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        condition_kind = str(row.get("s11ConditionKind", ""))
        if condition_kind == NO_MUTANT_CONTROL:
            outcome = "matched_no_mutant_control"
        elif str(row.get("mutantBehaviorVariant", "")) == "neutral_clone_control":
            outcome = "neutral_clone_reference"
        elif not bool(row.get("mutantCloneSizeConserved", False)):
            outcome = "lineage_tracking_failure"
        else:
            target_delta = safe_float(row.get("deltaVsNoMutantFinalTargetQualityScore"), 0.0)
            min_delta = safe_float(row.get("deltaVsNoMutantMinPolicyGoalScore"), 0.0)
            spread = safe_float(row.get("mutantCloneSpreadFraction"), 1.0)
            clone_agg = safe_float(row.get("mutantCloneFinalAggregation"), 0.0)
            class_changed = bool(row.get("mutantChangedMosaicClass", False))
            if target_delta <= -0.08 or min_delta <= -15.0:
                outcome = "disruptive_takeover_proxy"
            elif clone_agg >= 0.85 and spread <= 0.35:
                outcome = "segregated_clone"
            elif target_delta >= -0.03 and clone_agg >= 0.70 and spread <= 0.45 and not class_changed:
                outcome = "contained_clone"
            else:
                outcome = "mixed_or_equilibrated_clone"
        target_delta = safe_float(row.get("deltaVsNoMutantFinalTargetQualityScore"), 0.0)
        min_delta = safe_float(row.get("deltaVsNoMutantMinPolicyGoalScore"), 0.0)
        spread = safe_float(row.get("mutantCloneSpreadFraction"), 0.0)
        clone_agg = safe_float(row.get("mutantCloneFinalAggregation"), 0.0)
        class_changed = bool(row.get("mutantChangedMosaicClass", False))
        target_preservation = max(0.0, min(1.0, 1.0 + target_delta))
        containment_score = (
            0.35 * clone_agg
            + 0.25 * max(0.0, min(1.0, 1.0 - spread))
            + 0.25 * target_preservation
            + 0.15 * (0.0 if class_changed else 1.0)
        )
        takeover_proxy_score = max(
            0.0,
            min(
                1.0,
                2.0 * max(0.0, -target_delta)
                + 0.25 * float(class_changed)
                + max(0.0, spread - 0.35)
                + max(0.0, -min_delta / 100.0),
            ),
        )
        rows.append(
            {
                **row,
                "mutantOutcomeClass": outcome,
                "mutantOutcomeClassifierVersion": MUTANT_VERSION,
                "containmentScore": float(containment_score),
                "takeoverProxyScore": float(takeover_proxy_score),
            }
        )
    return pd.DataFrame(rows)


def summarize_mutant_effects(scored: pd.DataFrame) -> pd.DataFrame:
    mutants = scored[scored["s11ConditionKind"].astype(str) == "mutant_clone"].copy()
    if mutants.empty:
        return pd.DataFrame()
    group_cols = ["mutantBehaviorVariant", "governanceVariant", "interfaceRuleVariant", "graftOutcomeClass", "mutantCloneSize", "mutantClonePositionName"]
    rows: list[dict[str, Any]] = []
    for keys, group in mutants.groupby(group_cols, dropna=False, sort=False):
        payload = dict(zip(group_cols, keys, strict=True))
        rows.append(
            {
                **payload,
                "conditionCount": int(len(group)),
                "runSuccessRate": float(group["runSucceeded"].map(bool).mean()),
                "meanDeltaTargetQualityVsNoMutant": float(pd.to_numeric(group["deltaVsNoMutantFinalTargetQualityScore"], errors="coerce").mean()),
                "meanDeltaMinGoalScoreVsNoMutant": float(pd.to_numeric(group["deltaVsNoMutantMinPolicyGoalScore"], errors="coerce").mean()),
                "meanContainmentScore": float(pd.to_numeric(group["containmentScore"], errors="coerce").mean()),
                "meanTakeoverProxyScore": float(pd.to_numeric(group["takeoverProxyScore"], errors="coerce").mean()),
                "meanCloneFinalAggregation": float(pd.to_numeric(group["mutantCloneFinalAggregation"], errors="coerce").mean()),
                "meanCloneSpreadFraction": float(pd.to_numeric(group["mutantCloneSpreadFraction"], errors="coerce").mean()),
                "mosaicClassChangedRate": float(group["mutantChangedMosaicClass"].map(bool).mean()),
                "outcomeCountsJson": compact_json(dict(Counter(group["mutantOutcomeClass"].astype(str)))),
                "broadControlLike": bool(group["broadControlLike"].map(bool).any()) if "broadControlLike" in group.columns else False,
            }
        )
    return pd.DataFrame(rows)


def summarize_mutant_outcomes(scored: pd.DataFrame) -> pd.DataFrame:
    mutants = scored[scored["s11ConditionKind"].astype(str) == "mutant_clone"].copy()
    if mutants.empty:
        return pd.DataFrame()
    return (
        mutants.groupby(["mutantOutcomeClass", "mutantBehaviorVariant"], dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            meanContainmentScore=("containmentScore", "mean"),
            meanTakeoverProxyScore=("takeoverProxyScore", "mean"),
            meanDeltaTargetQualityVsNoMutant=("deltaVsNoMutantFinalTargetQualityScore", "mean"),
            meanCloneFinalAggregation=("mutantCloneFinalAggregation", "mean"),
            meanCloneSpreadFraction=("mutantCloneSpreadFraction", "mean"),
        )
        .reset_index()
    )


def validation_checks(
    selected_contexts: pd.DataFrame,
    condition_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    mutants = scored[scored["s11ConditionKind"].astype(str) == "mutant_clone"] if "s11ConditionKind" in scored.columns else pd.DataFrame()
    controls = scored[scored["s11ConditionKind"].astype(str) == NO_MUTANT_CONTROL] if "s11ConditionKind" in scored.columns else pd.DataFrame()
    matched_control_ids = set(controls["conditionId"].astype(str)) if not controls.empty else set()
    no_mutant_replay = controls.copy()
    if not no_mutant_replay.empty and "s10MatchedNoGraftFinalStateHash" in no_mutant_replay.columns:
        replay_matches = (
            no_mutant_replay["finalStateHash"].astype(str) == no_mutant_replay["s10MatchedNoGraftFinalStateHash"].astype(str)
        )
        replay_match_count = int(replay_matches.sum())
        replay_ok = bool(replay_match_count == len(no_mutant_replay))
    else:
        replay_match_count = 0
        replay_ok = False
    checks = [
        {
            "checkId": "s10_contexts_loaded",
            "success": bool(
                not selected_contexts.empty
                and selected_contexts["s11MatchedNoGraftConditionId"].astype(str).str.len().gt(0).all()
                and selected_contexts["graftOutcomeClass"].astype(str).str.len().gt(0).all()
            ),
            "detail": f"{len(selected_contexts)} S10 graft contexts selected with matched no-graft controls",
        },
        {
            "checkId": "condition_matrix_coverage",
            "success": bool(
                not condition_df.empty
                and {NO_MUTANT_CONTROL, "mutant_clone"} <= set(condition_df["s11ConditionKind"].astype(str))
                and set(MUTANT_BEHAVIOR_VARIANTS) <= set(condition_df["mutantBehaviorVariant"].astype(str))
                and set(MUTANT_CLONE_SIZES) <= set(condition_df["mutantCloneSize"].astype(int))
                and set(MUTANT_POSITION_NAMES) <= set(condition_df["mutantClonePositionName"].astype(str))
            ),
            "detail": f"{len(condition_df)} S11 conditions covering controls, clone sizes, positions, and behavior variants",
        },
        {
            "checkId": "matched_no_mutant_controls_included",
            "success": bool(
                not mutants.empty
                and not controls.empty
                and set(mutants["matchedNoMutantConditionId"].astype(str)) <= matched_control_ids
                and controls["s11ConditionKind"].astype(str).eq(NO_MUTANT_CONTROL).all()
            ),
            "detail": f"{len(controls)} no-mutant controls matched to {len(mutants)} mutant-clone rows",
        },
        {
            "checkId": "s10_matched_controls_reproduced",
            "success": bool(replay_ok),
            "detail": f"{replay_match_count}/{len(controls)} exact S10 matched no-graft control replays matched final-state hashes",
        },
        {
            "checkId": "lineage_metadata_written",
            "success": bool(
                not lineage_df.empty
                and lineage_df["initialLineageCellIds"].map(lambda value: len(value) > 0 if isinstance(value, list) else True).all()
                and lineage_df["analogyCaveat"].astype(str).str.contains("computational analogy").all()
            ),
            "detail": f"{len(lineage_df)} mutant lineage descriptors written",
        },
        {
            "checkId": "clone_size_and_lineage_tracked",
            "success": bool(
                not mutants.empty
                and mutants["mutantCloneInitialCount"].astype(int).gt(0).all()
                and mutants["mutantCloneFinalCount"].astype(int).gt(0).all()
                and mutants["mutantCloneSizeConserved"].map(bool).all()
            ),
            "detail": "all fixed-size mutant clone runs tracked and conserved lineage counts",
        },
        {
            "checkId": "runs_succeeded_and_counts_conserved",
            "success": bool(
                scored["runSucceeded"].fillna(False).map(bool).all()
                and scored["valueCountsConserved"].fillna(False).map(bool).all()
                and scored["policyCountsConserved"].fillna(False).map(bool).all()
                and scored["actualRatiosMatchTarget"].fillna(False).map(bool).all()
            ),
            "detail": "all S11 runs succeeded and conserved value/policy counts",
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
            "checkId": "mutant_outcomes_assigned",
            "success": bool(
                mutants["mutantOutcomeClass"].astype(str).isin(
                    {
                        "neutral_clone_reference",
                        "lineage_tracking_failure",
                        "disruptive_takeover_proxy",
                        "segregated_clone",
                        "contained_clone",
                        "mixed_or_equilibrated_clone",
                    }
                ).all()
            ),
            "detail": f"mutant outcomes: {dict(Counter(mutants['mutantOutcomeClass'].astype(str))) if not mutants.empty else {}}",
        },
        {
            "checkId": "analogy_caveats_retained",
            "success": bool(
                scored["analogyCaveat"].astype(str).str.contains("computational analogy").all()
                and scored["claimBoundary"].astype(str).str.contains("Computational").all()
            ),
            "detail": "cancer-like language and biological claim boundaries retained as computational analogies",
        },
    ]
    return pd.DataFrame(checks)


def write_mutant_plots(effect_summary: pd.DataFrame, outcome_summary: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not effect_summary.empty:
        plot = (
            effect_summary.groupby(["mutantBehaviorVariant", "governanceVariant"], dropna=False)
            .agg(meanTakeover=("meanTakeoverProxyScore", "mean"))
            .reset_index()
        )
        pivot = plot.pivot_table(index="mutantBehaviorVariant", columns="governanceVariant", values="meanTakeover", fill_value=0.0)
        fig, ax = plt.subplots(figsize=(11, 5))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0.0, vmax=max(1.0, float(pivot.max().max())))
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_title("E06 S11 mutant-clone takeover-proxy score")
        fig.colorbar(image, ax=ax, label="Mean takeover-proxy score")
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s11_mutant_takeover_heatmap.png",
            figure_dir / "e06_s11_mutant_takeover_heatmap.pdf",
            step_dir / "mutant_takeover_heatmap.png",
            step_dir / "mutant_takeover_heatmap.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    if not outcome_summary.empty:
        pivot = outcome_summary.pivot_table(index="mutantOutcomeClass", columns="mutantBehaviorVariant", values="conditionCount", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(pivot.columns))
        for outcome in pivot.index:
            values = pivot.loc[outcome].to_numpy(dtype=float)
            ax.bar(pivot.columns, values, bottom=bottom, label=outcome)
            bottom += values
        ax.set_ylabel("Mutant-clone conditions")
        ax.set_title("E06 S11 mutant-clone outcome classes")
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s11_mutant_outcomes.png",
            figure_dir / "e06_s11_mutant_outcomes.pdf",
            step_dir / "mutant_outcomes.png",
            step_dir / "mutant_outcomes.pdf",
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
    top = effect_summary.sort_values("meanTakeoverProxyScore", ascending=False, kind="mergesort").head(8) if not effect_summary.empty else pd.DataFrame()
    effect_lines = "\n".join(
        f"- `{row.mutantBehaviorVariant}` / `{row.governanceVariant}` / `{row.interfaceRuleVariant}` / S10 `{row.graftOutcomeClass}` "
        f"/ size {row.mutantCloneSize} / `{row.mutantClonePositionName}`: takeover proxy {row.meanTakeoverProxyScore:.3f}, "
        f"containment {row.meanContainmentScore:.3f}, target-quality delta {row.meanDeltaTargetQualityVsNoMutant:.3f}"
        for row in top.itertuples(index=False)
    ) or "- No mutant effect summaries available."
    outcome_lines = "\n".join(
        f"- `{row.mutantOutcomeClass}` / `{row.mutantBehaviorVariant}`: {int(row.conditionCount)} conditions"
        for row in outcome_summary.itertuples(index=False)
    ) or "- No mutant outcomes available."
    text = f"""# Research Step S11: Run cancer-like mutant experiments

## Completion status

Research step ID: `S11`. {status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Ran {status['conditionCount']} fixed-size mutant-clone or matched no-mutant conditions from {status['selectedContextCount']} S10 contexts. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

{CANCER_ANALOGY_CAVEAT} S11 uses fixed-size lineages in a one-dimensional policy array; division and growth are disabled, clone size is conserved by design, and takeover is a spatial/metric disruption proxy only. Broad-organizer rows remain flagged as target-map/global-control-like upper-bound baselines.

## Lay summary

S11 introduced small fixed-size donor-policy clones into S10 matched no-graft control states, compared each clone condition to a no-mutant replay control, and tested whether governance/interface contexts contained, segregated, or allowed disruptive clone behavior under explicitly computational definitions.

## Top takeover-proxy effects

{effect_lines}

## Outcome summary

{outcome_lines}

## Recommended next action

{status['recommendedNextAction']}
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s11_mutant_clone_experiments(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    panel_path: Path = DEFAULT_PANEL_PATH,
    s10_results_path: Path = DEFAULT_S10_RESULTS_PATH,
    s10_outcome_summary_path: Path = DEFAULT_S10_OUTCOME_SUMMARY_PATH,
    s06_pair_summary_path: Path = DEFAULT_S06_PAIR_SUMMARY_PATH,
    winner_criteria_path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH,
    workers: int | None = None,
    max_contexts: int = 8,
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
    s10_results = load_s10_results(s10_results_path)
    s10_outcome_summary = pd.read_parquet(s10_outcome_summary_path) if s10_outcome_summary_path.exists() else pd.DataFrame()
    selected_contexts = select_s11_contexts(s10_results, max_contexts=max_contexts)
    condition_df, parameter_df, lineage_df = build_mutant_condition_matrix(selected_contexts)
    criteria = load_winner_criteria(winner_criteria_path)
    pair_summary = pd.read_parquet(s06_pair_summary_path) if s06_pair_summary_path.exists() else pd.DataFrame()

    raw = run_mutant_conditions(
        condition_df,
        ready_panel,
        workers=workers,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )
    goal = augment_goal_results(raw)
    scored = score_dominance_results(goal, criteria)
    classified = classify_mosaic_formations(scored, pair_summary if not pair_summary.empty else None)
    classified = attach_matched_no_mutant_deltas(classified)
    classified = classify_mutant_outcomes(classified)
    classified["researchStepId"] = STEP_ID
    classified["sourceResearchStepId"] = "S11"
    classified["interventionStepId"] = STEP_ID
    classified["interventionType"] = "mutant_clone_experiment"
    effect_summary = summarize_mutant_effects(classified)
    outcome_summary = summarize_mutant_outcomes(classified)
    validation_df = validation_checks(selected_contexts, condition_df, lineage_df, classified)

    selected_csv = step_dir / "selected_s10_mutant_contexts.csv"
    selected_parquet = step_dir / "selected_s10_mutant_contexts.parquet"
    condition_csv = step_dir / "mutant_condition_matrix.csv"
    condition_parquet = step_dir / "mutant_condition_matrix.parquet"
    parameter_csv = step_dir / "mutant_behavior_parameters.csv"
    parameter_parquet = step_dir / "mutant_behavior_parameters.parquet"
    lineage_csv = step_dir / "mutant_lineage_metadata.csv"
    lineage_parquet = step_dir / "mutant_lineage_metadata.parquet"
    runs_csv = step_dir / "mutant_clone_runs.csv"
    runs_parquet = step_dir / "mutant_clone_runs.parquet"
    result_csv = results_dir / "e06_mutant_clone_experiments.csv"
    result_parquet = results_dir / "e06_mutant_clone_experiments.parquet"
    combined_csv = results_dir / "e06_governance_interventions.csv"
    combined_parquet = results_dir / "e06_governance_interventions.parquet"
    effect_csv = step_dir / "mutant_effect_summary.csv"
    effect_parquet = step_dir / "mutant_effect_summary.parquet"
    outcome_csv = step_dir / "mutant_outcome_summary.csv"
    outcome_parquet = step_dir / "mutant_outcome_summary.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    for path, df in [
        (selected_csv, compact_result_csv(selected_contexts)),
        (condition_csv, compact_result_csv(condition_df)),
        (parameter_csv, parameter_df),
        (lineage_csv, lineage_df),
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
        (parameter_parquet, parameter_df),
        (lineage_parquet, lineage_df),
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

    figure_paths = write_mutant_plots(effect_summary, outcome_summary, figures_dir, step_dir)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() else "failed"
    mutant_rows = classified[classified["s11ConditionKind"].astype(str) == "mutant_clone"]
    outcome = "supportive" if validation_result == "passed" and not mutant_rows.empty else "constraining/contradictory"
    artifacts = [
        selected_csv,
        selected_parquet,
        condition_csv,
        condition_parquet,
        parameter_csv,
        parameter_parquet,
        lineage_csv,
        lineage_parquet,
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
            CANCER_ANALOGY_CAVEAT,
            "S11 uses fixed-size policy-lineage clones; division and growth are disabled.",
            "Clone takeover is a bounded spatial/metric disruption proxy, not biological invasion or proliferation.",
            "The organizer_global_upper_bound variant remains a flagged target-map/global-control-like upper-bound baseline.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then proceed to S12 developmental-history experiments using S11 clone outcomes, "
            "matched no-mutant controls, governance/interface contexts, and fixed-size clone caveats; do not start S12 until review is complete."
        ),
        "laySummary": (
            "S11 introduced small fixed-size computational clone perturbations into S10 matched no-graft control states and compared them with "
            "matched no-mutant replays under selected governance and interface contexts."
        ),
        "outcomeClassification": outcome,
        "selectedContextCount": int(len(selected_contexts)),
        "conditionCount": int(len(condition_df)),
        "mutantConditionCount": int(mutant_rows.shape[0]),
        "matchedNoMutantControlCount": int((classified["s11ConditionKind"].astype(str) == NO_MUTANT_CONTROL).sum()),
        "mutantBehaviorVariants": list(MUTANT_BEHAVIOR_VARIANTS),
        "mutantCloneSizes": list(MUTANT_CLONE_SIZES),
        "mutantClonePositions": list(MUTANT_POSITION_NAMES),
        "s10GraftOutcomeClasses": sorted(selected_contexts["graftOutcomeClass"].astype(str).unique()),
        "governanceVariants": sorted(selected_contexts["governanceVariant"].astype(str).unique()),
        "interfaceRuleVariants": sorted(selected_contexts["interfaceRuleVariant"].astype(str).unique()),
        "broadControlRunCount": int(classified["broadControlLike"].map(bool).sum()) if "broadControlLike" in classified.columns else 0,
        "divisionEnabled": False,
        "workerCount": int(workers),
        "maxActivations": int(max_activations),
        "maxSwaps": int(max_swaps),
        "maxComparisons": int(max_comparisons),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
        "newDependenciesInstalled": [],
        "sourceCodeLocation": str(repo_root),
        "sourceCodeArtifactPolicy": "repository-backed source is committed to git; source files are not copied into artifacts per workspace instructions",
        "s10OutcomeSummaryRows": int(len(s10_outcome_summary)),
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
        "mutantVersion": MUTANT_VERSION,
        "inputArtifacts": {
            "s10Results": str(s10_results_path),
            "s10OutcomeSummary": str(s10_outcome_summary_path),
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
        "lineage": lineage_df,
        "results": classified,
        "effectSummary": effect_summary,
        "outcomeSummary": outcome_summary,
        "validation": validation_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E06 S11 fixed-size computational mutant-clone experiments")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--s10-results-path", type=Path, default=DEFAULT_S10_RESULTS_PATH)
    parser.add_argument("--s10-outcome-summary-path", type=Path, default=DEFAULT_S10_OUTCOME_SUMMARY_PATH)
    parser.add_argument("--s06-pair-summary-path", type=Path, default=DEFAULT_S06_PAIR_SUMMARY_PATH)
    parser.add_argument("--winner-criteria-path", type=Path, default=DEFAULT_S06_WINNER_CRITERIA_PATH)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-contexts", type=int, default=8)
    parser.add_argument("--max-activations", type=int, default=3500)
    parser.add_argument("--max-swaps", type=int, default=2500)
    parser.add_argument("--max-comparisons", type=int, default=20000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_s11_mutant_clone_experiments(
        artifacts_dir=args.artifacts_dir,
        panel_path=args.panel_path,
        s10_results_path=args.s10_results_path,
        s10_outcome_summary_path=args.s10_outcome_summary_path,
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
        f"{STEP_ID} {status['status']}: {status['conditionCount']} clone/control conditions, "
        f"{status['selectedContextCount']} S10 contexts, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
