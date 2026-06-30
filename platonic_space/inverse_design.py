from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e02_deterministic_simulator.metrics import aggregation
from morphospace import (
    DSL_VERSION,
    DSLPolicy,
    FrontierValidationTask,
    NullPolicy,
    PolicyEventSimulator,
    bubble_template,
    compute_competence_vector,
    generate_policy_corpus,
    insertion_template,
    local_inversion_program,
    null_program,
    parse_rule_program,
    selection_template,
)
from morphospace.competence import delayed_gratification_from_sortedness, sortedness_sign_change_count
from morphospace.policy_corpus import structure_hash

from .world_schema import compact_json


INVERSE_DESIGN_SCHEMA_VERSION = "e07_s12_inverse_design.v1"
INVERSE_DESIGN_MODEL_VERSION = "e07_s12_direct_simulation_search.v1"
INVERSE_DESIGN_CLAIM_BOUNDARY = (
    "Empirical inverse design over DSL policies in deterministic simulator worlds. Designed policies are "
    "computational artifacts validated only by direct replay on held-out simulator seeds; they are not biological "
    "designs, causal invariants, clinical advice, or evidence about living systems."
)

SUPPORTED_S11_FAMILIES = (
    "high_aggregation",
    "high_completion",
    "high_final_sortedness",
    "low_activation_count",
    "unusual_transfer_final_sortedness",
)
DEFERRED_S11_FAMILIES = (
    "high_error_reduction",
    "high_compatibility",
    "high_repair_success",
    "low_swap_count",
)
PROFILE_IDS = ("balanced_sorting_low_cost", "robust_frozen_sorting", "aggregation_chimera")
PROFILE_SCORE_COLUMNS = (
    "sortingCompletionMean",
    "sortingFinalSortednessMean",
    "sortingEnergyMean",
    "sortingActivationEfficiencyMean",
    "frozenFinalSortednessMean",
    "frozenEnergyMean",
    "frozenBlockedMoveScoreMean",
    "chimeraAggregationAucMean",
    "chimeraAggregationPeakMean",
    "chimeraFinalSortednessMean",
    "chimeraEnergyMean",
    "transferScoreMean",
)
PANEL_VECTOR_COLUMNS = (
    "completionSuccess",
    "finalSortednessScore",
    "energyScore",
    "activationEfficiencyScore",
    "swapEfficiencyScore",
    "robustnessScore",
    "blockedMoveRate",
    "aggregationAucScore",
    "aggregationPeakScore",
    "transferScore",
    "compatibilityScore",
    "oscillationScore",
)


@dataclass(frozen=True)
class DesignRunPanel:
    tasks: tuple[FrontierValidationTask, ...]
    seed_count: int
    stage: str


def claim_boundary() -> str:
    return INVERSE_DESIGN_CLAIM_BOUNDARY


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else math.nan


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _stable_id(prefix: str, payload: Mapping[str, Any], length: int = 14) -> str:
    digest = hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def dataframe_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
            out[column] = out[column].map(
                lambda value: compact_json(sorted(value) if isinstance(value, set) else value)
                if isinstance(value, (dict, list, tuple, set))
                else value
            )
        if out[column].dtype == "object":
            non_null = out[column].dropna()
            observed_types = {type(value) for value in non_null}
            if len(observed_types) > 1:
                out[column] = out[column].map(lambda value: None if value is None or pd.isna(value) else str(value))
    return out


def s11_support_table(target_performance: pd.DataFrame, direct_validation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarize which S11 target families are suitable for S12 inverse design."""

    rows: list[dict[str, Any]] = []
    direct_validation = direct_validation if direct_validation is not None else pd.DataFrame()
    families = sorted(set(target_performance.get("counterfactualFamily", pd.Series(dtype=str)).astype(str)))
    for family in families:
        subset = target_performance[target_performance["counterfactualFamily"].astype(str).eq(family)].copy()
        direct = (
            direct_validation[direct_validation["counterfactualFamily"].astype(str).eq(family)].copy()
            if not direct_validation.empty and "counterfactualFamily" in direct_validation.columns
            else pd.DataFrame()
        )
        min_candidates = int(pd.to_numeric(subset.get("candidateCount", pd.Series(dtype=float)), errors="coerce").min())
        median_within = _mean(subset.get("withinS05P95Fraction", pd.Series(dtype=float)))
        median_source_verified = _mean(subset.get("sourceTableVerifiedFraction", pd.Series(dtype=float)))
        direct_within = _mean(direct.get("observedWithinS05P95", pd.Series(dtype=float)).astype(float)) if not direct.empty else math.nan
        mae = _mean(subset.get("mae", pd.Series(dtype=float)))
        rmse = _mean(subset.get("rmse", pd.Series(dtype=float)))
        if family in SUPPORTED_S11_FAMILIES and min_candidates >= 2 and median_within >= 0.85:
            decision = "selected_as_design_substrate"
        elif family in DEFERRED_S11_FAMILIES:
            decision = "deferred_or_secondary_caveat"
        else:
            decision = "not_selected_low_support_or_sparse"
        if family == "high_error_reduction":
            caveat = "High-scale target with S11 failures and large prediction errors; excluded from S12 design targets."
        elif family in {"high_compatibility", "high_repair_success"}:
            caveat = "Sparse/domain-specific S11 support; documented but not used as a primary S12 target."
        elif family == "low_swap_count":
            caveat = "S11 low-work predictions had large absolute errors; S12 treats work cost as a direct held-out measurement only."
        else:
            caveat = "S11 replay validation was sufficiently stable for use as profile support, with simulator-proxy limits."
        rows.append(
            {
                "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
                "counterfactualFamily": family,
                "target": ",".join(sorted(set(subset.get("target", pd.Series(dtype=str)).astype(str)))),
                "splitCount": int(subset["splitName"].nunique()) if "splitName" in subset else 0,
                "candidateCountMin": min_candidates,
                "withinS05P95FractionMean": median_within,
                "directWithinS05P95FractionMean": direct_within,
                "sourceTableVerifiedFractionMean": median_source_verified,
                "maeMean": mae,
                "rmseMean": rmse,
                "s12Decision": decision,
                "caveat": caveat,
            }
        )
    return pd.DataFrame(rows).sort_values(["s12Decision", "counterfactualFamily"], kind="mergesort").reset_index(drop=True)


def build_inverse_design_target_profiles(
    target_performance: pd.DataFrame,
    direct_validation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    support = s11_support_table(target_performance, direct_validation)
    support_lookup = {str(row.counterfactualFamily): row._asdict() for row in support.itertuples(index=False)}

    def s11_rows(families: Sequence[str]) -> list[dict[str, Any]]:
        return [support_lookup[family] for family in families if family in support_lookup]

    profiles = [
        {
            "profileId": "balanced_sorting_low_cost",
            "profileName": "Balanced sorting with low activation cost",
            "profileFamily": "sorting_efficiency",
            "targetDescription": "High held-out sorting completion/final sortedness with direct low activation and energy proxies.",
            "s11SupportFamilies": ["high_final_sortedness", "high_completion", "low_activation_count", "unusual_transfer_final_sortedness"],
            "deferredOrSecondaryFamilies": ["low_swap_count", "high_error_reduction"],
            "targetMetricValues": {
                "sortingCompletionMean": 0.90,
                "sortingFinalSortednessMean": 0.95,
                "sortingEnergyMean": 0.40,
                "sortingActivationEfficiencyMean": 0.40,
                "transferScoreMean": 0.85,
            },
            "selectionWeights": {
                "sortingCompletionMean": 0.30,
                "sortingFinalSortednessMean": 0.30,
                "sortingEnergyMean": 0.20,
                "sortingActivationEfficiencyMean": 0.10,
                "transferScoreMean": 0.10,
            },
            "validationThresholds": {
                "heldoutCompositeScore": 0.78,
                "sortingCompletionMean": 0.80,
                "sortingFinalSortednessMean": 0.92,
                "sortingEnergyMean": 0.35,
            },
            "caveat": "Low swap count is not optimized from the S11 surrogate because its S11 errors were high; it is measured after design only.",
        },
        {
            "profileId": "robust_frozen_sorting",
            "profileName": "Robust frozen-cell sorting",
            "profileFamily": "perturbation_robustness",
            "targetDescription": "Maintain high final sortedness under stuck-frozen perturbations while limiting blocked-move and energy costs.",
            "s11SupportFamilies": ["high_completion", "high_final_sortedness", "low_activation_count"],
            "deferredOrSecondaryFamilies": ["high_repair_success", "high_error_reduction"],
            "targetMetricValues": {
                "frozenFinalSortednessMean": 0.75,
                "frozenEnergyMean": 0.40,
                "frozenBlockedMoveScoreMean": 0.80,
                "sortingCompletionMean": 0.80,
            },
            "selectionWeights": {
                "frozenFinalSortednessMean": 0.45,
                "frozenEnergyMean": 0.20,
                "frozenBlockedMoveScoreMean": 0.20,
                "sortingCompletionMean": 0.15,
            },
            "validationThresholds": {
                "heldoutCompositeScore": 0.68,
                "frozenFinalSortednessMean": 0.70,
                "frozenBlockedMoveScoreMean": 0.65,
            },
            "caveat": "The direct stuck-frozen panel can make perfect completion impossible; validation uses final sortedness and blocked-move rate, not completion alone.",
        },
        {
            "profileId": "aggregation_chimera",
            "profileName": "High aggregation in candidate/null chimera",
            "profileFamily": "chimera_aggregation",
            "targetDescription": "High candidate/null algotype aggregation while retaining useful sorting behavior on mixed-policy arrays.",
            "s11SupportFamilies": ["high_aggregation", "high_final_sortedness"],
            "deferredOrSecondaryFamilies": ["high_compatibility", "high_repair_success"],
            "targetMetricValues": {
                "chimeraAggregationAucMean": 0.45,
                "chimeraAggregationPeakMean": 0.75,
                "chimeraFinalSortednessMean": 0.75,
                "chimeraEnergyMean": 0.25,
            },
            "selectionWeights": {
                "chimeraAggregationAucMean": 0.40,
                "chimeraAggregationPeakMean": 0.25,
                "chimeraFinalSortednessMean": 0.20,
                "chimeraEnergyMean": 0.15,
            },
            "validationThresholds": {
                "heldoutCompositeScore": 0.60,
                "chimeraAggregationAucMean": 0.40,
                "chimeraAggregationPeakMean": 0.70,
            },
            "caveat": "Compatibility and repair targets remain sparse/highly contextual; S12 validates aggregation only in the E03 candidate/null simulator panel.",
        },
    ]
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        row = dict(profile)
        support_families = list(profile["s11SupportFamilies"])
        deferred_families = list(profile["deferredOrSecondaryFamilies"])
        row.update(
            {
                "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
                "selectedForS12": True,
                "s11SupportFamiliesJson": compact_json(support_families),
                "deferredOrSecondaryFamiliesJson": compact_json(deferred_families),
                "s11SupportEvidenceJson": compact_json(s11_rows(support_families)),
                "deferredOrSecondaryEvidenceJson": compact_json(s11_rows(deferred_families)),
                "targetMetricValuesJson": compact_json(profile["targetMetricValues"]),
                "selectionWeightsJson": compact_json(profile["selectionWeights"]),
                "validationThresholdsJson": compact_json(profile["validationThresholds"]),
                "claimBoundary": INVERSE_DESIGN_CLAIM_BOUNDARY,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def training_panel() -> DesignRunPanel:
    return DesignRunPanel(
        stage="training_search",
        seed_count=2,
        tasks=(
            FrontierValidationTask(
                task_id="s12_train_unique_n6_zigzag",
                task_family="sorting",
                task_panel="e07_s12_training_search",
                input_profile="unique_n6_zigzag",
                initial_values=(6, 1, 5, 2, 4, 3),
                scheduler_seed_base=12000,
                tie_seed_base=22000,
                max_activations=1200,
                max_swaps=1200,
                max_comparisons=4800,
            ),
            FrontierValidationTask(
                task_id="s12_train_duplicate_n6",
                task_family="sorting_duplicate_values",
                task_panel="e07_s12_training_search",
                input_profile="duplicate_n6",
                initial_values=(3, 1, 3, 2, 1, 2),
                scheduler_seed_base=12100,
                tie_seed_base=22100,
                max_activations=1200,
                max_swaps=1200,
                max_comparisons=4800,
            ),
            FrontierValidationTask(
                task_id="s12_train_stuck_frozen_n6",
                task_family="frozen",
                task_panel="e07_s12_training_search",
                input_profile="unique_n6_stuck_frozen",
                initial_values=(6, 1, 5, 2, 4, 3),
                scheduler_seed_base=12200,
                tie_seed_base=22200,
                frozen_variant="stuck",
                frozen_positions=(2,),
                max_activations=1400,
                max_swaps=1400,
                max_comparisons=5600,
            ),
            FrontierValidationTask(
                task_id="s12_train_transfer_n8",
                task_family="transfer",
                task_panel="e07_s12_training_search",
                input_profile="unique_n8_transfer_train",
                initial_values=(8, 1, 7, 2, 6, 3, 5, 4),
                scheduler_seed_base=12300,
                tie_seed_base=22300,
                max_activations=1800,
                max_swaps=1800,
                max_comparisons=7200,
            ),
            FrontierValidationTask(
                task_id="s12_train_chimera_null_n8",
                task_family="chimera",
                task_panel="e07_s12_training_search",
                input_profile="alternating_dsl_null_chimera_n8_train",
                initial_values=(8, 1, 7, 2, 6, 3, 5, 4),
                scheduler_seed_base=12400,
                tie_seed_base=22400,
                chimera_with_null=True,
                candidate_positions=(0, 2, 4, 6),
                max_activations=1800,
                max_swaps=1800,
                max_comparisons=7200,
            ),
        ),
    )


def holdout_panel() -> DesignRunPanel:
    return DesignRunPanel(
        stage="heldout_validation",
        seed_count=4,
        tasks=(
            FrontierValidationTask(
                task_id="s12_holdout_unique_n8",
                task_family="sorting",
                task_panel="e07_s12_heldout_validation",
                input_profile="unique_n8_interleaved_holdout",
                initial_values=(8, 1, 7, 2, 6, 3, 5, 4),
                scheduler_seed_base=181000,
                tie_seed_base=191000,
                max_activations=1800,
                max_swaps=1800,
                max_comparisons=7200,
            ),
            FrontierValidationTask(
                task_id="s12_holdout_duplicate_n8",
                task_family="sorting_duplicate_values",
                task_panel="e07_s12_heldout_validation",
                input_profile="duplicate_n8_holdout",
                initial_values=(4, 2, 4, 1, 3, 1, 2, 3),
                scheduler_seed_base=181100,
                tie_seed_base=191100,
                max_activations=1800,
                max_swaps=1800,
                max_comparisons=7200,
            ),
            FrontierValidationTask(
                task_id="s12_holdout_stuck_frozen_n8",
                task_family="frozen",
                task_panel="e07_s12_heldout_validation",
                input_profile="unique_n8_stuck_frozen_holdout",
                initial_values=(8, 1, 7, 2, 6, 3, 5, 4),
                scheduler_seed_base=181200,
                tie_seed_base=191200,
                frozen_variant="stuck",
                frozen_positions=(3,),
                max_activations=2200,
                max_swaps=2200,
                max_comparisons=8800,
            ),
            FrontierValidationTask(
                task_id="s12_holdout_transfer_n10",
                task_family="transfer",
                task_panel="e07_s12_heldout_validation",
                input_profile="unique_n10_transfer_holdout",
                initial_values=(10, 1, 9, 2, 8, 3, 7, 4, 6, 5),
                scheduler_seed_base=181300,
                tie_seed_base=191300,
                max_activations=3000,
                max_swaps=3000,
                max_comparisons=12000,
            ),
            FrontierValidationTask(
                task_id="s12_holdout_chimera_null_n10",
                task_family="chimera",
                task_panel="e07_s12_heldout_validation",
                input_profile="alternating_dsl_null_chimera_n10_holdout",
                initial_values=(10, 1, 9, 2, 8, 3, 7, 4, 6, 5),
                scheduler_seed_base=181400,
                tie_seed_base=191400,
                chimera_with_null=True,
                candidate_positions=(0, 2, 4, 6, 8),
                max_activations=3000,
                max_swaps=3000,
                max_comparisons=12000,
            ),
        ),
    )


def _predicate(op: str, operator: str | None = None, *, key: str | None = None, value: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"op": op}
    if operator is not None:
        payload["operator"] = operator
    if key is not None:
        payload["key"] = key
    if value is not None:
        payload["value"] = value
    return payload


def _action(action: str, *, reason: str | None = None, updates: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": action}
    if reason is not None:
        payload["reason"] = reason
    if updates:
        payload["updates"] = [dict(update) for update in updates]
    return payload


def _rule(name: str, when: Sequence[Mapping[str, Any]], then: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": name, "when": [dict(item) for item in when], "then": dict(then)}


def _payload(name: str, rules: Sequence[Mapping[str, Any]], initial_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": DSL_VERSION,
        "name": name,
        "initial_state": dict(initial_state or {}),
        "rules": [dict(rule) for rule in rules],
        "default": {"action": "wait", "reason": "default_wait"},
    }


def _program_from_payload(payload: Mapping[str, Any]) -> tuple[str, str]:
    program_id = _stable_id("s12", payload, length=14)
    named = dict(payload)
    named["policy_id"] = program_id
    named["name"] = str(named.get("name", program_id))[:120]
    program = parse_rule_program(named)
    return program.policy_id, program.to_json()


def _strict_sort_rules(
    *,
    priority: str = "left_first",
    left_updates: Sequence[Mapping[str, Any]] = (),
    right_updates: Sequence[Mapping[str, Any]] = (),
    left_guards: Sequence[Mapping[str, Any]] = (),
    right_guards: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    left_rule = _rule(
        "swap_left_if_local_inversion",
        [*left_guards, _predicate("compare_left", "<")],
        _action("swap_left", updates=left_updates),
    )
    right_rule = _rule(
        "swap_right_if_local_inversion",
        [*right_guards, _predicate("compare_right", ">")],
        _action("swap_right", updates=right_updates),
    )
    return [left_rule, right_rule] if priority == "left_first" else [right_rule, left_rule]


def existing_policy_structure_lookup(reference_size: int = 512) -> dict[str, str]:
    records = generate_policy_corpus(target_size=reference_size)
    lookup = {record.structure_hash: record.policy_id for record in records}
    for program in (
        local_inversion_program("s12_ref_local_inversion"),
        null_program("s12_ref_null"),
        bubble_template(policy_id="s12_ref_bubble_left"),
        bubble_template(neighbor_priority="right_first", policy_id="s12_ref_bubble_right"),
        insertion_template(policy_id="s12_ref_insertion"),
        selection_template(policy_id="s12_ref_selection"),
    ):
        lookup.setdefault(structure_hash(program.to_dict()), program.policy_id)
    return lookup


def generate_inverse_design_pool(reference_size: int = 512) -> pd.DataFrame:
    """Generate deterministic DSL mutation/search candidates for S12."""

    existing_lookup = existing_policy_structure_lookup(reference_size)
    payloads: list[dict[str, Any]] = []

    for priority in ("left_first", "right_first"):
        payloads.append(_payload(f"S12 strict local inversion {priority}", _strict_sort_rules(priority=priority)))
        payloads.append(
            _payload(
                f"S12 counted local inversion {priority}",
                _strict_sort_rules(
                    priority=priority,
                    left_updates=({"op": "increment", "key": "move_count", "amount": 1, "min": 0, "max": 9},),
                    right_updates=({"op": "increment", "key": "move_count", "amount": 1, "min": 0, "max": 9},),
                ),
                initial_state={"move_count": 0},
            )
        )
        payloads.append(
            _payload(
                f"S12 signal local inversion {priority}",
                _strict_sort_rules(
                    priority=priority,
                    left_updates=({"op": "signal", "channel": "s12_left_move", "value": 1},),
                    right_updates=({"op": "signal", "channel": "s12_right_move", "value": 1},),
                ),
            )
        )
        payloads.append(
            _payload(
                f"S12 cooldown local inversion {priority}",
                [
                    _rule(
                        "cooldown_wait",
                        [_predicate("state_compare", ">", key="cooldown", value=0)],
                        _action("wait", updates=({"op": "decrement", "key": "cooldown", "amount": 1, "min": 0, "max": 1},)),
                    ),
                    *_strict_sort_rules(
                        priority=priority,
                        left_updates=({"op": "set", "key": "cooldown", "value": 1},),
                        right_updates=({"op": "set", "key": "cooldown", "value": 1},),
                    ),
                ],
                initial_state={"cooldown": 0},
            )
        )

    for split in (1, 2, 3, 4, 5, 6):
        payloads.append(
            _payload(
                f"S12 left-half right-priority split {split}",
                [
                    _rule(
                        "right_first_in_left_region",
                        [_predicate("position_compare", "<", value=split), _predicate("compare_right", ">")],
                        _action("swap_right"),
                    ),
                    *_strict_sort_rules(priority="left_first"),
                ],
            )
        )
        payloads.append(
            _payload(
                f"S12 right-half left-priority split {split}",
                [
                    _rule(
                        "left_first_in_right_region",
                        [_predicate("position_compare", ">=", value=split), _predicate("compare_left", "<")],
                        _action("swap_left"),
                    ),
                    *_strict_sort_rules(priority="right_first"),
                ],
            )
        )
        payloads.append(
            _payload(
                f"S12 interior guarded inversion split {split}",
                _strict_sort_rules(
                    priority="left_first",
                    left_guards=(_predicate("position_compare", ">=", value=1),),
                    right_guards=(_predicate("position_compare", "<", value=split + 2),),
                ),
            )
        )

    one_direction_rules = [
        ("left_only", [_rule("left_only", [_predicate("compare_left", "<")], _action("swap_left"))]),
        ("right_only", [_rule("right_only", [_predicate("compare_right", ">")], _action("swap_right"))]),
        (
            "left_then_equal_right",
            [
                _rule("left_strict", [_predicate("compare_left", "<")], _action("swap_left")),
                _rule("right_equal_or_inversion", [_predicate("compare_right", ">=")], _action("swap_right")),
            ],
        ),
        (
            "right_then_equal_left",
            [
                _rule("right_strict", [_predicate("compare_right", ">")], _action("swap_right")),
                _rule("left_equal_or_inversion", [_predicate("compare_left", "<=")], _action("swap_left")),
            ],
        ),
    ]
    for name, rules in one_direction_rules:
        payloads.append(_payload(f"S12 exploratory {name}", rules))

    for max_count in (1, 2, 3, 4):
        payloads.append(
            _payload(
                f"S12 limited burst local inversion max {max_count}",
                [
                    _rule(
                        "reset_if_locally_ordered",
                        [{"op": "always"}],
                        _action("wait", updates=({"op": "decrement", "key": "burst_count", "amount": 1, "min": 0, "max": max_count},)),
                    ),
                    *_strict_sort_rules(
                        priority="left_first",
                        left_guards=(_predicate("state_compare", "<", key="burst_count", value=max_count),),
                        right_guards=(_predicate("state_compare", "<", key="burst_count", value=max_count),),
                        left_updates=({"op": "increment", "key": "burst_count", "amount": 1, "min": 0, "max": max_count},),
                        right_updates=({"op": "increment", "key": "burst_count", "amount": 1, "min": 0, "max": max_count},),
                    ),
                ],
                initial_state={"burst_count": 0},
            )
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for generation_index, payload in enumerate(payloads):
        try:
            policy_id, dsl_json = _program_from_payload(payload)
            program = parse_rule_program(dsl_json)
        except Exception as exc:  # pragma: no cover - defensive validation path
            records.append(
                {
                    "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
                    "policyId": f"invalid_{generation_index}",
                    "poolStatus": "invalid_dsl",
                    "invalidReason": str(exc),
                    "generationIndex": generation_index,
                }
            )
            continue
        digest = structure_hash(program.to_dict())
        if digest in seen:
            continue
        seen.add(digest)
        rules = program.rules
        action_counts = Counter(rule.then.action for rule in rules)
        exact_existing = existing_lookup.get(digest)
        records.append(
            {
                "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
                "policyId": policy_id,
                "poolStatus": "valid",
                "invalidReason": "",
                "generationIndex": generation_index,
                "generationMethod": "deterministic_dsl_mutation_search",
                "designFamily": "local_rule_dsl",
                "structureHash": digest,
                "exactDuplicateExistingPolicyId": exact_existing,
                "isExactDuplicateOfExisting": exact_existing is not None,
                "ruleCount": len(rules),
                "predicateCount": sum(len(rule.when) for rule in rules),
                "stateKeyCount": len(program.initial_state),
                "actionCountsJson": compact_json(dict(sorted(action_counts.items()))),
                "dslProgramJson": dsl_json,
                "programPretty": program.pretty(),
                "claimBoundary": INVERSE_DESIGN_CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(records).sort_values(["poolStatus", "generationIndex"], kind="mergesort").reset_index(drop=True)


def make_policies_for_task(program_json: str, task: FrontierValidationTask):
    program = parse_rule_program(program_json)
    if not task.chimera_with_null:
        return DSLPolicy(program)
    candidate_positions = set(map(int, task.candidate_positions))
    return [DSLPolicy(program) if position in candidate_positions else NullPolicy() for position in range(len(task.initial_values))]


def _trace_sortedness(result: Any) -> list[float]:
    return [float(row["sortedness_percent"]) for row in result.trace_rows]


def _trace_hashes(result: Any) -> list[str]:
    return [str(row["state_hash"]) for row in result.trace_rows]


def _trace_values(result: Any) -> list[list[int]]:
    values: list[list[int]] = []
    for row in result.trace_rows:
        state_hash = str(row["state_hash"])
        if state_hash:
            values.append([])
    return values


def _value_counts_conserved(initial_values: Sequence[int], final_values: Sequence[int]) -> bool:
    return Counter(map(int, initial_values)) == Counter(map(int, final_values))


def run_policy_on_task(
    policy_row: Mapping[str, Any],
    task: FrontierValidationTask,
    seed_index: int,
    *,
    stage: str,
    implementation: str = "e07_s12_cpu_inverse_design",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy_id = str(policy_row["policyId"])
    scheduler_seed = int(task.scheduler_seed_base + seed_index)
    tie_seed = int(task.tie_seed_base + seed_index)
    condition_id = f"S12::{stage}::{task.task_id}::{policy_id}::seed{scheduler_seed}"
    result = PolicyEventSimulator(
        list(task.initial_values),
        make_policies_for_task(str(policy_row["dslProgramJson"]), task),
        frozen_positions=task.frozen_positions,
        frozen_variant=task.frozen_variant,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_seed,
        condition_id=condition_id,
        implementation=implementation,
        research_step_id="S12",
    ).run(max_activations=task.max_activations, max_swaps=task.max_swaps, max_comparisons=task.max_comparisons)

    sortedness_values = _trace_sortedness(result)
    dg = delayed_gratification_from_sortedness(sortedness_values)
    oscillation_proxy = (
        sortedness_sign_change_count(sortedness_values) / max(1, len(sortedness_values) - 1)
        if len(sortedness_values) > 1
        else 0.0
    )
    final_sortedness_score = float(result.final_sortedness_percent / 100.0)
    initial_aggregation = float(aggregation(result.initial_algotypes))
    final_aggregation = float(result.final_aggregation)
    run_record = {
        "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
        "modelVersion": INVERSE_DESIGN_MODEL_VERSION,
        "researchStepId": "S12",
        "stage": stage,
        "policyId": policy_id,
        "designProfileId": str(policy_row.get("designProfileId", "")),
        "designRankWithinProfile": policy_row.get("designRankWithinProfile"),
        "taskId": task.task_id,
        "taskFamily": task.task_family,
        "taskPanel": task.task_panel,
        "inputProfile": task.input_profile,
        "seedIndex": int(seed_index),
        "schedulerSeed": scheduler_seed,
        "tieBreakerSeed": tie_seed,
        "initialValuesJson": compact_json(list(task.initial_values)),
        "finalValuesJson": compact_json(result.final_values),
        "frozenVariant": task.frozen_variant,
        "frozenPositionsJson": compact_json(list(task.frozen_positions)),
        "chimeraWithNull": bool(task.chimera_with_null),
        "candidatePositionsJson": compact_json(list(task.candidate_positions)),
        "completed": bool(result.completed),
        "completionSuccess": 1.0 if result.completed else 0.0,
        "stopReason": result.stop_reason,
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "activationCount": int(result.activation_count),
        "eventCount": int(result.event_count),
        "blockedMoveAttempts": int(result.blocked_move_attempts),
        "blockedMoveRate": float(result.blocked_move_attempts / max(1, result.activation_count)),
        "frozenSwapAttempts": int(result.frozen_swap_attempts),
        "finalSortednessPercent": float(result.final_sortedness_percent),
        "finalSortednessScore": final_sortedness_score,
        "finalMonotonicityError": int(result.final_monotonicity_error),
        "delayedGratification": float(dg["delayedGratification"]),
        "dgEventCount": int(dg["dgEventCount"]),
        "oscillationProxy": float(oscillation_proxy),
        "initialAggregation": initial_aggregation,
        "finalAggregation": final_aggregation,
        "robustnessScore": None if task.frozen_variant == "none" else float((1.0 if result.completed else 0.0) * final_sortedness_score),
        "traceStateCount": int(len(result.trace_rows)),
        "traceHashSequenceSha256": hashlib.sha256(compact_json(_trace_hashes(result)).encode("utf-8")).hexdigest(),
        "valueCountsConserved": bool(_value_counts_conserved(task.initial_values, result.final_values)),
        "claimBoundary": INVERSE_DESIGN_CLAIM_BOUNDARY,
    }
    vector = compute_competence_vector(
        result,
        policy_id=policy_id,
        policy_family=str(policy_row.get("designFamily", policy_row.get("policyFamily", "e07_s12_inverse_design"))),
        task_id=task.task_id,
        task_family=task.task_family,
        task_panel=task.task_panel,
        input_profile=task.input_profile,
        frozen_variant=task.frozen_variant,
        frozen_count=len(task.frozen_positions),
        replicate_index=seed_index,
        source_metric_source=f"{stage}_direct_cpu_simulation",
        source_artifact_path=None,
        transfer_score=final_sortedness_score if task.task_family == "transfer" else None,
        compatibility_score=final_aggregation if task.task_family == "chimera" else None,
    )
    vector.update(
        {
            "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
            "modelVersion": INVERSE_DESIGN_MODEL_VERSION,
            "stage": stage,
            "designProfileId": str(policy_row.get("designProfileId", "")),
            "designRankWithinProfile": policy_row.get("designRankWithinProfile"),
            "schedulerSeed": scheduler_seed,
            "tieBreakerSeed": tie_seed,
            "backend": "cpu_reference",
            "claimBoundary": INVERSE_DESIGN_CLAIM_BOUNDARY,
        }
    )
    trace_record = {
        "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
        "researchStepId": "S12",
        "stage": stage,
        "policyId": policy_id,
        "designProfileId": str(policy_row.get("designProfileId", "")),
        "taskId": task.task_id,
        "seedIndex": int(seed_index),
        "schedulerSeed": scheduler_seed,
        "tieBreakerSeed": tie_seed,
        "traceStateCount": int(len(result.trace_rows)),
        "traceHashSequenceJson": compact_json(_trace_hashes(result)),
        "traceSortednessSequenceJson": compact_json(sortedness_values),
        "traceHashSequenceSha256": run_record["traceHashSequenceSha256"],
    }
    return run_record, vector, trace_record


def run_design_panel(policy_df: pd.DataFrame, panel: DesignRunPanel, *, include_traces: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for _, row in policy_df.reset_index(drop=True).iterrows():
        for task in panel.tasks:
            for seed_index in range(panel.seed_count):
                run_record, vector, trace_record = run_policy_on_task(row, task, seed_index, stage=panel.stage)
                run_rows.append(run_record)
                vector_rows.append(vector)
                if include_traces:
                    trace_rows.append(trace_record)
    return pd.DataFrame(run_rows), pd.DataFrame(vector_rows), pd.DataFrame(trace_rows)


def _policy_metric_summary(vectors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if vectors.empty:
        return pd.DataFrame()
    for policy_id, group in vectors.groupby("policyId", sort=True):
        family = group["taskFamily"].astype(str)
        sorting = group[family.isin(["sorting", "sorting_duplicate_values"])]
        frozen = group[family.eq("frozen")]
        chimera = group[family.eq("chimera")]
        transfer = group[family.eq("transfer")]
        rows.append(
            {
                "policyId": str(policy_id),
                "runCount": int(len(group)),
                "taskCount": int(group["taskId"].nunique()),
                "completionMean": _mean(group["completionSuccess"]),
                "finalSortednessMean": _mean(group["finalSortednessScore"]),
                "energyMean": _mean(group["energyScore"]),
                "activationEfficiencyMean": _mean(group["activationEfficiencyScore"]),
                "swapEfficiencyMean": _mean(group["swapEfficiencyScore"]),
                "sortingCompletionMean": _mean(sorting["completionSuccess"]) if not sorting.empty else math.nan,
                "sortingFinalSortednessMean": _mean(sorting["finalSortednessScore"]) if not sorting.empty else math.nan,
                "sortingEnergyMean": _mean(sorting["energyScore"]) if not sorting.empty else math.nan,
                "sortingActivationEfficiencyMean": _mean(sorting["activationEfficiencyScore"]) if not sorting.empty else math.nan,
                "frozenFinalSortednessMean": _mean(frozen["finalSortednessScore"]) if not frozen.empty else math.nan,
                "frozenEnergyMean": _mean(frozen["energyScore"]) if not frozen.empty else math.nan,
                "frozenBlockedMoveScoreMean": _mean(1.0 - pd.to_numeric(frozen["blockedMoveRate"], errors="coerce").clip(0.0, 1.0))
                if not frozen.empty and "blockedMoveRate" in frozen
                else math.nan,
                "frozenRobustnessMean": _mean(frozen["robustnessScore"]) if not frozen.empty else math.nan,
                "chimeraAggregationAucMean": _mean(chimera["aggregationAucScore"]) if not chimera.empty else math.nan,
                "chimeraAggregationPeakMean": _mean(chimera["aggregationPeakScore"]) if not chimera.empty else math.nan,
                "chimeraFinalSortednessMean": _mean(chimera["finalSortednessScore"]) if not chimera.empty else math.nan,
                "chimeraEnergyMean": _mean(chimera["energyScore"]) if not chimera.empty else math.nan,
                "chimeraCompatibilityMean": _mean(chimera["compatibilityScore"]) if not chimera.empty else math.nan,
                "transferScoreMean": _mean(transfer["transferScore"]) if not transfer.empty else math.nan,
                "oscillationScoreMean": _mean(group["oscillationScore"]),
            }
        )
    return pd.DataFrame(rows)


def _profile_score(summary: Mapping[str, Any], profile: Mapping[str, Any]) -> tuple[float, float]:
    weights = _json_loads(profile.get("selectionWeightsJson"), default=profile.get("selectionWeights") or {})
    targets = _json_loads(profile.get("targetMetricValuesJson"), default=profile.get("targetMetricValues") or {})
    weighted = 0.0
    weight_total = 0.0
    distance_terms: list[float] = []
    for metric, weight in dict(weights).items():
        value = _finite(summary.get(metric), default=0.0)
        weighted += float(weight) * max(0.0, min(1.0, value))
        weight_total += float(weight)
    for metric, target in dict(targets).items():
        value = _finite(summary.get(metric), default=0.0)
        distance_terms.append((max(0.0, min(1.0, value)) - max(0.0, min(1.0, float(target)))) ** 2)
    return (
        float(weighted / weight_total) if weight_total > 0 else math.nan,
        float(math.sqrt(sum(distance_terms) / max(1, len(distance_terms)))) if distance_terms else math.nan,
    )


def score_candidate_pool(pool: pd.DataFrame, training_vectors: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = _policy_metric_summary(training_vectors)
    if summary.empty:
        return pd.DataFrame(), summary
    meta_cols = [
        "policyId",
        "generationIndex",
        "generationMethod",
        "designFamily",
        "structureHash",
        "isExactDuplicateOfExisting",
        "exactDuplicateExistingPolicyId",
        "ruleCount",
        "predicateCount",
        "stateKeyCount",
        "actionCountsJson",
        "dslProgramJson",
        "programPretty",
    ]
    merged = summary.merge(pool[[column for column in meta_cols if column in pool.columns]], on="policyId", how="left")
    rows: list[dict[str, Any]] = []
    for _, profile in profiles.iterrows():
        for _, row in merged.iterrows():
            score, distance = _profile_score(row, profile)
            out = row.to_dict()
            out.update(
                {
                    "designProfileId": str(profile["profileId"]),
                    "profileName": str(profile["profileName"]),
                    "profileFamily": str(profile["profileFamily"]),
                    "profileScore": score,
                    "targetProfileDistance": distance,
                    "s11SupportFamiliesJson": profile["s11SupportFamiliesJson"],
                    "profileCaveat": profile["caveat"],
                }
            )
            rows.append(out)
    scored = pd.DataFrame(rows).sort_values(
        ["designProfileId", "profileScore", "targetProfileDistance", "isExactDuplicateOfExisting", "generationIndex"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )
    return scored.reset_index(drop=True), summary


def select_designed_policies(scored: pd.DataFrame, per_profile: int = 3) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    for profile_id in PROFILE_IDS:
        subset = scored[scored["designProfileId"].astype(str).eq(profile_id)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["isExactDuplicateOfExisting", "profileScore", "targetProfileDistance", "generationIndex"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        profile_rows: list[dict[str, Any]] = []
        for _, row in subset.iterrows():
            digest = str(row.get("structureHash", ""))
            if digest in selected_hashes:
                continue
            profile_rows.append(row.to_dict())
            selected_hashes.add(digest)
            if len(profile_rows) >= per_profile:
                break
        if len(profile_rows) < per_profile:
            for _, row in subset.iterrows():
                digest = str(row.get("structureHash", ""))
                if any(str(item.get("policyId")) == str(row["policyId"]) and item.get("designProfileId") == profile_id for item in profile_rows):
                    continue
                profile_rows.append(row.to_dict())
                selected_hashes.add(digest)
                if len(profile_rows) >= per_profile:
                    break
        for rank, row in enumerate(profile_rows, start=1):
            row["designRankWithinProfile"] = rank
            row["designSelectionStatus"] = "selected_for_heldout_direct_validation"
            row["selectedPolicyId"] = row["policyId"]
            row["designCaveat"] = (
                "Exact structure exists in upstream/reference corpus; selected only because the profile pool lacked a better novel candidate."
                if bool(row.get("isExactDuplicateOfExisting", False))
                else "Novel DSL structure relative to generated/reference corpus hash lookup; behavior may still match a baseline on the held-out panel."
            )
            selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


def summarize_profile_validation(designs: pd.DataFrame, holdout_vectors: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_summary = _policy_metric_summary(holdout_vectors)
    if metric_summary.empty or designs.empty:
        return pd.DataFrame(), pd.DataFrame()
    meta = designs[
        [
            "policyId",
            "designProfileId",
            "designRankWithinProfile",
            "profileName",
            "profileFamily",
            "profileScore",
            "targetProfileDistance",
            "isExactDuplicateOfExisting",
            "exactDuplicateExistingPolicyId",
            "structureHash",
            "designCaveat",
            "dslProgramJson",
        ]
    ].drop_duplicates("policyId")
    profile_lookup = {str(row.profileId): row._asdict() for row in profiles.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for _, row in metric_summary.merge(meta, on="policyId", how="inner").iterrows():
        profile = profile_lookup[str(row["designProfileId"])]
        composite, distance = _profile_score(row, profile)
        thresholds = _json_loads(profile.get("validationThresholdsJson"), default={})
        threshold_checks = []
        for metric, threshold in thresholds.items():
            value = composite if metric == "heldoutCompositeScore" else _finite(row.get(metric), default=0.0)
            threshold_checks.append(value >= float(threshold))
        validated = bool(threshold_checks and all(threshold_checks))
        out = row.to_dict()
        out.update(
            {
                "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
                "heldoutCompositeScore": composite,
                "heldoutTargetProfileDistance": distance,
                "profileValidated": validated,
                "validationThresholdsJson": profile.get("validationThresholdsJson"),
                "validationDecision": "validated" if validated else "not_validated_or_partial",
                "validationCaveat": str(profile.get("caveat", "")),
                "claimBoundary": INVERSE_DESIGN_CLAIM_BOUNDARY,
            }
        )
        rows.append(out)
    design_validation = pd.DataFrame(rows).sort_values(
        ["designProfileId", "profileValidated", "heldoutCompositeScore", "designRankWithinProfile"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    profile_rows: list[dict[str, Any]] = []
    for profile_id, group in design_validation.groupby("designProfileId", sort=True):
        profile_rows.append(
            {
                "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
                "profileId": profile_id,
                "profileName": str(group["profileName"].iloc[0]),
                "selectedDesignCount": int(len(group)),
                "validatedDesignCount": int(group["profileValidated"].sum()),
                "bestHeldoutCompositeScore": float(pd.to_numeric(group["heldoutCompositeScore"], errors="coerce").max()),
                "meanHeldoutCompositeScore": _mean(group["heldoutCompositeScore"]),
                "bestHeldoutTargetProfileDistance": float(pd.to_numeric(group["heldoutTargetProfileDistance"], errors="coerce").min()),
                "validatedPolicyIdsJson": compact_json(sorted(group.loc[group["profileValidated"], "policyId"].astype(str).tolist())),
                "profileStatus": "validated" if bool(group["profileValidated"].any()) else "not_validated",
                "caveat": str(group["validationCaveat"].iloc[0]),
            }
        )
    return design_validation.reset_index(drop=True), pd.DataFrame(profile_rows)


def reference_policy_table(reference_size: int = 48) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference_programs = [
        ("ref_local_inversion", "classic_dsl_reference", local_inversion_program("s12_reference_local_inversion")),
        ("ref_null", "classic_dsl_reference", null_program("s12_reference_null")),
        ("ref_bubble_left", "classic_dsl_reference", bubble_template(policy_id="s12_reference_bubble_left")),
        (
            "ref_bubble_right",
            "classic_dsl_reference",
            bubble_template(neighbor_priority="right_first", policy_id="s12_reference_bubble_right"),
        ),
        ("ref_insertion", "classic_dsl_reference", insertion_template(policy_id="s12_reference_insertion")),
        ("ref_selection", "classic_dsl_reference", selection_template(policy_id="s12_reference_selection")),
    ]
    for label, family, program in reference_programs:
        rows.append(
            {
                "policyId": label,
                "referenceSource": family,
                "designFamily": family,
                "structureHash": structure_hash(program.to_dict()),
                "dslProgramJson": program.to_json(),
                "programPretty": program.pretty(),
            }
        )
    generated = generate_policy_corpus(target_size=max(64, reference_size * 4))
    for record in generated:
        complexity = record.complexity()
        if int(complexity.get("stochasticActionCount", 0)) > 0:
            continue
        rows.append(
            {
                "policyId": f"ref_{record.policy_id}",
                "referenceSource": "generated_policy_corpus",
                "designFamily": "generated_policy_corpus",
                "structureHash": record.structure_hash,
                "dslProgramJson": record.program.to_json(),
                "programPretty": record.program.pretty(),
            }
        )
        if len(rows) >= reference_size:
            break
    return pd.DataFrame(rows).drop_duplicates("structureHash").reset_index(drop=True)


def nearest_existing_policy_comparison(
    design_validation: pd.DataFrame,
    reference_vectors: pd.DataFrame,
) -> pd.DataFrame:
    design_summary = _policy_metric_summary(reference_vectors[reference_vectors["policyId"].isin(set(design_validation["policyId"].astype(str)))])
    reference_summary = _policy_metric_summary(reference_vectors[~reference_vectors["policyId"].isin(set(design_validation["policyId"].astype(str)))])
    if design_summary.empty or reference_summary.empty:
        return pd.DataFrame()
    metric_cols = [column for column in PANEL_VECTOR_COLUMNS if column in reference_vectors.columns]
    summary_cols = [
        "completionMean",
        "finalSortednessMean",
        "energyMean",
        "activationEfficiencyMean",
        "swapEfficiencyMean",
        "frozenFinalSortednessMean",
        "frozenEnergyMean",
        "frozenBlockedMoveScoreMean",
        "chimeraAggregationAucMean",
        "chimeraAggregationPeakMean",
        "chimeraFinalSortednessMean",
        "chimeraEnergyMean",
        "transferScoreMean",
        "oscillationScoreMean",
    ]
    summary_cols = [column for column in summary_cols if column in design_summary.columns and column in reference_summary.columns]
    rows: list[dict[str, Any]] = []
    design_meta = design_validation.set_index("policyId").to_dict(orient="index")
    for _, design in design_summary.iterrows():
        distances: list[tuple[float, pd.Series]] = []
        for _, reference in reference_summary.iterrows():
            terms = []
            for column in summary_cols:
                left = _finite(design.get(column), default=0.0)
                right = _finite(reference.get(column), default=0.0)
                terms.append((left - right) ** 2)
            distances.append((float(math.sqrt(sum(terms) / max(1, len(terms)))), reference))
        distances.sort(key=lambda item: item[0])
        distance, nearest = distances[0]
        meta = design_meta.get(str(design["policyId"]), {})
        rows.append(
            {
                "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
                "policyId": str(design["policyId"]),
                "designProfileId": str(meta.get("designProfileId", "")),
                "nearestExistingPolicyId": str(nearest["policyId"]),
                "behaviorDistanceOnHeldoutPanel": distance,
                "distanceMetricColumnsJson": compact_json(summary_cols),
                "exactBehaviorMatchOnPanel": bool(distance <= 1.0e-12),
                "metricColumnCount": len(metric_cols),
                "comparisonCaveat": "Nearest-neighbor comparison is direct-panel behavior only, not proof of behavioral identity outside this panel.",
            }
        )
    return pd.DataFrame(rows).sort_values(["designProfileId", "behaviorDistanceOnHeldoutPanel"], kind="mergesort").reset_index(drop=True)


def validation_checks(
    profiles: pd.DataFrame,
    pool: pd.DataFrame,
    designs: pd.DataFrame,
    holdout_runs: pd.DataFrame,
    design_validation: pd.DataFrame,
    policy_files: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        {
            "checkId": "target_profiles_selected",
            "severity": "error",
            "success": set(profiles["profileId"].astype(str)) == set(PROFILE_IDS),
            "observed": ",".join(sorted(profiles["profileId"].astype(str))) if not profiles.empty else "",
            "expected": ",".join(PROFILE_IDS),
        },
        {
            "checkId": "deferred_sparse_targets_not_primary",
            "severity": "error",
            "success": not profiles["s11SupportFamiliesJson"].astype(str).str.contains("high_error_reduction|high_compatibility|high_repair_success").any(),
            "observed": "primary support families exclude sparse/high-scale targets",
            "expected": "no sparse/high-scale family in primary support",
        },
        {
            "checkId": "candidate_pool_valid_nonempty",
            "severity": "error",
            "success": not pool[pool["poolStatus"].astype(str).eq("valid")].empty,
            "observed": int(pool["poolStatus"].astype(str).eq("valid").sum()) if "poolStatus" in pool else 0,
            "expected": ">0",
        },
        {
            "checkId": "designed_policies_selected",
            "severity": "error",
            "success": len(designs) >= len(PROFILE_IDS),
            "observed": int(len(designs)),
            "expected": f">={len(PROFILE_IDS)}",
        },
        {
            "checkId": "heldout_direct_runs_complete",
            "severity": "error",
            "success": not holdout_runs.empty and holdout_runs["valueCountsConserved"].astype(bool).all(),
            "observed": int(len(holdout_runs)),
            "expected": "nonempty and all value counts conserved",
        },
        {
            "checkId": "policy_file_roundtrip",
            "severity": "error",
            "success": not policy_files.empty and policy_files["roundtripSuccess"].astype(bool).all(),
            "observed": int(policy_files["roundtripSuccess"].astype(bool).sum()) if not policy_files.empty else 0,
            "expected": int(len(policy_files)),
        },
        {
            "checkId": "at_least_one_validated_design",
            "severity": "warning",
            "success": not design_validation.empty and bool(design_validation["profileValidated"].astype(bool).any()),
            "observed": int(design_validation["profileValidated"].astype(bool).sum()) if not design_validation.empty else 0,
            "expected": ">=1",
        },
        {
            "checkId": "all_profiles_have_direct_decision",
            "severity": "error",
            "success": set(design_validation["designProfileId"].astype(str)) == set(PROFILE_IDS) if not design_validation.empty else False,
            "observed": ",".join(sorted(set(design_validation["designProfileId"].astype(str)))) if not design_validation.empty else "",
            "expected": ",".join(PROFILE_IDS),
        },
    ]
    return pd.DataFrame(checks)


def policy_outcome_classification(profile_summary: pd.DataFrame) -> str:
    if profile_summary.empty:
        return "null"
    validated_profiles = int(profile_summary["validatedDesignCount"].astype(int).gt(0).sum())
    validated_designs = int(profile_summary["validatedDesignCount"].astype(int).sum())
    if validated_profiles >= 2 and validated_designs >= 3:
        return "supportive"
    if validated_designs > 0:
        return "constraining"
    return "null"


def write_designed_policy_files(designs: pd.DataFrame, directory: Path) -> pd.DataFrame:
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, row in designs.sort_values(["designProfileId", "designRankWithinProfile"], kind="mergesort").iterrows():
        program = parse_rule_program(row["dslProgramJson"])
        filename = f"{row['designProfileId']}_{int(row['designRankWithinProfile']):02d}_{row['policyId']}.json"
        path = directory / filename
        path.write_text(json.dumps(program.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        roundtrip = compact_json(parse_rule_program(path.read_text(encoding="utf-8")).to_dict()) == compact_json(program.to_dict())
        rows.append(
            {
                "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
                "policyId": str(row["policyId"]),
                "designProfileId": str(row["designProfileId"]),
                "designRankWithinProfile": int(row["designRankWithinProfile"]),
                "dslPath": str(path),
                "structureHash": structure_hash(program.to_dict()),
                "roundtripSuccess": bool(roundtrip),
            }
        )
    return pd.DataFrame(rows)
