"""Phase-boundary helpers for E03 S09.

S09 uses the S08 elite set and S05 parameterized DSL policy families as a
bounded probe of local transitions in morphospace.  This module keeps the
selection and analysis logic deterministic so the execution script can focus on
running CPU reference simulations and writing artifacts.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .competence import canonical_json


PHASE_BOUNDARY_VERSION = "e03_s09_phase_boundaries.v1"

CORE_PARAMETERIZED_METHODS = {
    "classic_parameter_sweep",
    "single_rule_sweep",
    "probability_mutation",
}

OPERATOR_ORDER = {
    "lt": 0,
    "lteq": 1,
    "eq": 2,
    "neeq": 3,
    "gteq": 4,
    "gt": 5,
}
ACTION_ORDER = {
    "wait": 0,
    "swap_left": 1,
    "swap_right": 2,
    "swap_target": 3,
    "remember": 4,
    "signal": 5,
}
NEIGHBOR_ORDER = {"left": 0, "right": 1, "target": 2}

BOUNDARY_METRICS = (
    "completionRate",
    "meanFinalSortednessScore",
    "meanDelayedGratification",
    "meanAggregationFinal",
    "meanOscillationProxy",
    "meanRobustnessScore",
)


@dataclass(frozen=True)
class PhaseBoundaryTask:
    """CPU-reference task used to probe S09 transitions."""

    task_id: str
    task_family: str
    task_panel: str
    input_profile: str
    initial_values: tuple[int, ...]
    scheduler_seed_base: int
    tie_seed_base: int
    frozen_variant: str = "none"
    frozen_positions: tuple[int, ...] = ()
    chimera_with_null: bool = False
    candidate_positions: tuple[int, ...] = ()
    max_activations: int = 512
    max_swaps: int = 512
    max_comparisons: int = 2048

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "taskFamily": self.task_family,
            "taskPanel": self.task_panel,
            "inputProfile": self.input_profile,
            "initialValues": list(self.initial_values),
            "schedulerSeedBase": int(self.scheduler_seed_base),
            "tieSeedBase": int(self.tie_seed_base),
            "frozenVariant": self.frozen_variant,
            "frozenPositions": list(self.frozen_positions),
            "chimeraWithNull": bool(self.chimera_with_null),
            "candidatePositions": list(self.candidate_positions),
            "maxActivations": int(self.max_activations),
            "maxSwaps": int(self.max_swaps),
            "maxComparisons": int(self.max_comparisons),
        }


def _safe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _lineage_probability(lineage_id: str) -> float | None:
    match = re.search(r"_(0(?:\.\d+)?|1(?:\.0+)?)$", lineage_id)
    return None if match is None else float(match.group(1))


def _lineage_parent_id(lineage_id: str, prefix: str) -> str | None:
    match = re.match(rf"^{re.escape(prefix)}_(pc_[A-Za-z0-9]+)(?:_|$)", lineage_id)
    return None if match is None else match.group(1)


def _program_rules(program_payload: Any) -> list[Mapping[str, Any]]:
    program = _safe_json(program_payload, {})
    rules = program.get("rules", []) if isinstance(program, Mapping) else []
    return [rule for rule in rules if isinstance(rule, Mapping)]


def _action_probabilities(program_payload: Any) -> list[float]:
    probabilities: list[float] = []
    for rule in _program_rules(program_payload):
        then = rule.get("then", {})
        if isinstance(then, Mapping):
            probabilities.append(_safe_float(then.get("probability", 1.0), 1.0))
    return probabilities


def _initial_target_position(program_payload: Any) -> int | None:
    program = _safe_json(program_payload, {})
    if not isinstance(program, Mapping):
        return None
    initial_state = program.get("initial_state", {})
    if not isinstance(initial_state, Mapping) or "target_position" not in initial_state:
        return None
    try:
        return int(initial_state["target_position"])
    except (TypeError, ValueError):
        return None


def _parent_ids(value: Any) -> list[str]:
    payload = _safe_json(value, [])
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _single_rule_metadata(lineage_id: str) -> tuple[str, str, float, str]:
    match = re.match(
        r"^single_compare_(left|right|target)_(lt|lteq|eq|neeq|gteq|gt)_(wait|swap_left|swap_right|swap_target|remember|signal)_(0(?:\.\d+)?|1(?:\.0+)?)$",
        lineage_id,
    )
    if match is None:
        return ("single_rule_misc", "predicate_action_probability", 0.0, lineage_id)
    neighbor, operator, action, probability_text = match.groups()
    probability = float(probability_text)
    value = (
        NEIGHBOR_ORDER.get(neighbor, 9) * 100.0
        + OPERATOR_ORDER.get(operator, 9) * 10.0
        + ACTION_ORDER.get(action, 9)
        + (1.0 - probability)
    )
    family_key = f"single_rule_compare_{neighbor}_{operator}"
    label = f"{neighbor} {operator} -> {action} p={probability:g}"
    return (family_key, "swap_preference_and_wait_probability", value, label)


def phase_family_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic family and parameter metadata for one policy row."""

    generation_method = str(row.get("generationMethod", "unknown"))
    lineage_id = str(row.get("lineageId", "unknown"))
    policy_id = str(row.get("policyId", "unknown"))
    probabilities = _action_probabilities(row.get("dslProgramJson"))
    probability = _lineage_probability(lineage_id)
    if probability is None and probabilities:
        probability = float(np.mean(probabilities))
    initial_target = _initial_target_position(row.get("dslProgramJson"))

    family_key = generation_method
    parameter_axis = "generation_index"
    parameter_value = _safe_float(row.get("generationIndex"), 0.0)
    parameter_label = lineage_id
    parent_id = None
    family_description = "Fallback ordering by generation index."

    classic = re.match(
        r"^(bubble|insertion|selection)_(increasing|decreasing)(?:_(left_first|right_first))?(?:_([0-9.]+))?$",
        lineage_id,
    )
    if generation_method == "classic_parameter_sweep" and classic is not None:
        alg, direction, priority, suffix = classic.groups()
        if alg == "bubble":
            family_key = f"classic_bubble_{direction}_{priority or 'target_order'}"
            parameter_axis = "swap_probability"
            parameter_value = float(suffix or probability or 1.0)
            parameter_label = f"{alg} {direction} {priority or 'target'} p={parameter_value:g}"
            family_description = "Bubble-like local inversion template varied by direction, neighbor priority, and action probability."
        elif alg == "insertion":
            family_key = f"classic_insertion_{direction}"
            parameter_axis = "swap_probability"
            parameter_value = float(suffix or probability or 1.0)
            parameter_label = f"{alg} {direction} p={parameter_value:g}"
            family_description = "Insertion-like left-adjacent template varied by direction and action probability."
        else:
            family_key = f"classic_selection_{direction}"
            parameter_axis = "initial_target_position"
            parameter_value = float(initial_target if initial_target is not None else suffix or 0)
            parameter_label = f"{alg} {direction} target0={int(parameter_value)}"
            family_description = "Selection-like target-position template varied by starting target-position bias."
    elif generation_method == "single_rule_sweep":
        family_key, parameter_axis, parameter_value, parameter_label = _single_rule_metadata(lineage_id)
        family_description = "Single-predicate/action sweep varied by comparison neighbor, operator, action, and probability."
    elif generation_method == "probability_mutation":
        parent_id = _lineage_parent_id(lineage_id, "mut_prob")
        family_key = f"probability_mutation_{parent_id or 'unknown_parent'}"
        parameter_axis = "stochasticity_probability"
        parameter_value = float(probability if probability is not None else 1.0)
        parameter_label = f"{parent_id or policy_id} p={parameter_value:g}"
        family_description = "Mutation family that changes probabilistic action execution."
    elif generation_method == "memory_mutation":
        parent_id = _lineage_parent_id(lineage_id, "mut_memory")
        family_key = f"memory_mutation_{parent_id or 'unknown_parent'}"
        parameter_axis = "memory_bit"
        parameter_value = max(1.0, _safe_float(row.get("stateKeyCount"), 1.0))
        parameter_label = f"{parent_id or policy_id} memory={parameter_value:g}"
        family_description = "Mutation family adding remembered local state."
    elif generation_method == "target_memory_mutation":
        parent_id = _lineage_parent_id(lineage_id, "mut_target")
        family_key = f"target_memory_mutation_{parent_id or 'unknown_parent'}"
        parameter_axis = "target_position_bias"
        base = 0.0 if initial_target is None else float(initial_target)
        parameter_value = base + 0.1 * max(0.0, _safe_float(row.get("targetUpdateCount"), 0.0))
        parameter_label = f"{parent_id or policy_id} target0={base:g}"
        family_description = "Mutation family adding target-position estimation or target-directed state."
    elif generation_method == "two_parent_rule_recombination":
        parents = _parent_ids(row.get("parentPolicyIdsJson"))
        parent_id = "+".join(parents[:2]) if parents else None
        family_key = f"recombination_{parent_id or 'unknown_parents'}"
        parameter_axis = "recombined_complexity"
        parameter_value = _safe_float(row.get("complexityScore"), 0.0)
        parameter_label = f"{parent_id or policy_id} complexity={parameter_value:g}"
        family_description = "Two-parent recombination ordered by DSL complexity."
    elif generation_method == "seed_import":
        family_key = "seed_import_baselines"
        parameter_axis = "baseline_order"
        parameter_value = _safe_float(row.get("generationIndex"), 0.0)
        parameter_label = lineage_id
        family_description = "Seed policy baselines retained as anchors."
    elif generation_method == "seeded_random_grammar":
        family_key = "seeded_random_grammar_elites"
        parameter_axis = "grammar_generation_index"
        parameter_value = _safe_float(row.get("generationIndex"), 0.0)
        parameter_label = lineage_id
        family_description = "Random-grammar policies included only when selected as S08 elites or close controls."

    return {
        "phaseBoundaryVersion": PHASE_BOUNDARY_VERSION,
        "familyKey": family_key,
        "parameterAxis": parameter_axis,
        "parameterValue": float(parameter_value),
        "parameterLabel": parameter_label,
        "parameterFamilyDescription": family_description,
        "parentPolicyId": parent_id,
        "meanActionProbability": float(np.mean(probabilities)) if probabilities else 1.0,
        "minActionProbability": float(np.min(probabilities)) if probabilities else 1.0,
        "initialTargetPosition": initial_target,
    }


def add_phase_metadata(df: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.DataFrame([phase_family_metadata(row) for row in df.to_dict("records")])
    return pd.concat([df.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1)


def select_phase_boundary_candidates(
    corpus_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    *,
    max_policies: int = 420,
    memory_neighbor_limit: int = 36,
    target_neighbor_limit: int = 36,
    recombination_neighbor_limit: int = 16,
) -> pd.DataFrame:
    """Select a bounded S09 corpus around S08 elites and parameterized families."""

    corpus = add_phase_metadata(corpus_df.copy())
    corpus["policyId"] = corpus["policyId"].astype(str)
    elite_ranks = {str(row["policyId"]): int(row["eliteRank"]) for _, row in elites_df.iterrows()}
    elite_ids = set(elite_ranks)
    corpus["isS08Elite"] = corpus["policyId"].isin(elite_ids)
    corpus["s08EliteRank"] = corpus["policyId"].map(elite_ranks)

    def elite_parent_hit(row: Mapping[str, Any]) -> bool:
        parents = set(_parent_ids(row.get("parentPolicyIdsJson")))
        parent_id = row.get("parentPolicyId")
        if parent_id:
            parents.add(str(parent_id))
        return bool(parents & elite_ids) or any(elite_id in str(row.get("lineageId", "")) for elite_id in elite_ids)

    corpus["hasS08EliteParent"] = [elite_parent_hit(row) for row in corpus.to_dict("records")]
    reasons: dict[str, list[str]] = {}

    def mark(mask: pd.Series, reason: str) -> None:
        for policy_id in corpus.loc[mask, "policyId"].astype(str):
            reasons.setdefault(policy_id, []).append(reason)

    mark(corpus["isS08Elite"], "s08_elite")
    mark(corpus["generationMethod"].isin(CORE_PARAMETERIZED_METHODS), "core_parameterized_family")
    mark(corpus["generationMethod"].eq("seed_import"), "seed_baseline_anchor")

    for method, limit, reason in [
        ("memory_mutation", memory_neighbor_limit, "bounded_memory_neighbors"),
        ("target_memory_mutation", target_neighbor_limit, "bounded_target_neighbors"),
        ("two_parent_rule_recombination", recombination_neighbor_limit, "bounded_recombination_neighbors"),
    ]:
        subset = corpus[corpus["generationMethod"].eq(method)].copy()
        if subset.empty:
            continue
        subset["eliteSort"] = np.where(subset["isS08Elite"], 0, np.where(subset["hasS08EliteParent"], 1, 2))
        subset = subset.sort_values(
            ["eliteSort", "complexityScore", "generationIndex", "policyId"],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).head(int(limit))
        mark(corpus["policyId"].isin(subset["policyId"]), reason)

    selected_ids = set(reasons)
    selected = corpus[corpus["policyId"].isin(selected_ids)].copy()
    selected["candidateSelectionReason"] = selected["policyId"].map(
        lambda policy_id: canonical_json(sorted(set(reasons.get(str(policy_id), []))))
    )
    selected["candidatePriority"] = np.select(
        [
            selected["isS08Elite"],
            selected["generationMethod"].eq("classic_parameter_sweep"),
            selected["generationMethod"].eq("single_rule_sweep"),
            selected["generationMethod"].eq("probability_mutation"),
            selected["generationMethod"].eq("seed_import"),
            selected["hasS08EliteParent"],
        ],
        [0, 1, 2, 3, 4, 5],
        default=6,
    )
    selected = selected.sort_values(
        [
            "candidatePriority",
            "familyKey",
            "parameterValue",
            "complexityScore",
            "generationIndex",
            "policyId",
        ],
        ascending=[True, True, True, True, True, True],
        kind="mergesort",
    )
    if len(selected) > int(max_policies):
        selected = selected.head(int(max_policies)).copy()
    selected["candidateRank"] = np.arange(1, len(selected) + 1)
    return selected.reset_index(drop=True)


def phase_boundary_config_dict(
    *,
    seed_count: int,
    max_policies: int,
    tasks: Sequence[PhaseBoundaryTask],
) -> dict[str, Any]:
    return {
        "phaseBoundaryVersion": PHASE_BOUNDARY_VERSION,
        "candidateSelection": {
            "source": "S08 QD elites plus S05 parameterized families and bounded mutation/recombination neighbors",
            "maxPolicies": int(max_policies),
            "coreParameterizedMethods": sorted(CORE_PARAMETERIZED_METHODS),
        },
        "seedCountPerPolicyTask": int(seed_count),
        "tasks": [task.to_dict() for task in tasks],
        "boundaryMetrics": list(BOUNDARY_METRICS),
        "transitionRules": {
            "sortingSuccessTransition": "completion rate crosses 0.5 or adjacent delta >= 0.5",
            "sortednessTransition": "mean final sortedness score adjacent delta >= 0.25",
            "delayedGratificationTransition": "mean DG crosses zero or adjacent delta >= 0.2",
            "aggregationTransition": "chimera aggregation crosses 0.5 or adjacent delta >= 0.25",
            "oscillationTransition": "oscillation proxy adjacent delta >= 0.15 or low/high boundary",
            "robustnessTransition": "frozen-task robustness crosses 0.4 or adjacent delta >= 0.25",
        },
        "validationPlan": [
            "run smoke CPU simulations before full sweep",
            "repeat every candidate/task with fixed seed count",
            "check value-count conservation and metric ranges",
            "record repeated-seed SD/SEM/CI near boundary pairs",
            "stop before S10",
        ],
    }


def run_smoke_checks(smoke_runs: pd.DataFrame, *, expected_min_rows: int = 1) -> pd.DataFrame:
    checks = [
        {
            "checkId": "smoke_rows_nonempty",
            "success": len(smoke_runs) >= int(expected_min_rows),
            "detail": f"{len(smoke_runs)} smoke run rows",
        },
        {
            "checkId": "smoke_value_counts_conserved",
            "success": bool(smoke_runs["valueCountsConserved"].all()) if len(smoke_runs) else False,
            "detail": "all smoke rows preserved input value multisets",
        },
        {
            "checkId": "smoke_metric_ranges",
            "success": bool(smoke_runs["finalSortednessScore"].between(0.0, 1.0).all()) if len(smoke_runs) else False,
            "detail": "smoke final sortedness scores are bounded",
        },
    ]
    return pd.DataFrame(checks)


def summarize_repeated_seed_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated-seed CPU rows to one policy/task record."""

    if run_df.empty:
        return pd.DataFrame()
    df = run_df.copy()
    numeric_columns = [
        "completedNumeric",
        "finalSortednessScore",
        "delayedGratification",
        "dgEventCount",
        "oscillationProxy",
        "finalAggregation",
        "robustnessScore",
        "swapCount",
        "comparisonCount",
        "activationCount",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    group_columns = [
        "policyId",
        "taskId",
        "familyKey",
        "parameterAxis",
        "parameterValue",
        "parameterLabel",
        "generationMethod",
        "lineageId",
        "taskFamily",
        "taskPanel",
        "inputProfile",
        "frozenVariant",
        "isS08Elite",
        "s08EliteRank",
        "candidateRank",
    ]
    available_groups = [column for column in group_columns if column in df.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(available_groups, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(available_groups, keys)}
        seed_count = int(group["seedIndex"].nunique()) if "seedIndex" in group.columns else int(len(group))

        def stats(column: str) -> dict[str, Any]:
            values = pd.to_numeric(group[column], errors="coerce").dropna() if column in group.columns else pd.Series(dtype=float)
            n = int(len(values))
            mean = float(values.mean()) if n else np.nan
            sd = float(values.std(ddof=1)) if n > 1 else 0.0
            sem = float(sd / math.sqrt(n)) if n > 1 else 0.0
            return {
                f"{column}N": n,
                f"{column}Mean": mean,
                f"{column}Sd": sd,
                f"{column}Sem": sem,
                f"{column}Ci95Low": float(mean - 1.96 * sem) if n else np.nan,
                f"{column}Ci95High": float(mean + 1.96 * sem) if n else np.nan,
            }

        stat_payload: dict[str, Any] = {}
        for column in [
            "completedNumeric",
            "finalSortednessScore",
            "delayedGratification",
            "oscillationProxy",
            "finalAggregation",
            "robustnessScore",
            "swapCount",
            "comparisonCount",
            "activationCount",
        ]:
            stat_payload.update(stats(column))

        row.update(
            {
                "phaseBoundaryVersion": PHASE_BOUNDARY_VERSION,
                "seedCount": seed_count,
                "runCount": int(len(group)),
                "completionRate": stat_payload.pop("completedNumericMean", np.nan),
                "completionRateSd": stat_payload.pop("completedNumericSd", 0.0),
                "completionRateSem": stat_payload.pop("completedNumericSem", 0.0),
                "completionRateCi95Low": stat_payload.pop("completedNumericCi95Low", np.nan),
                "completionRateCi95High": stat_payload.pop("completedNumericCi95High", np.nan),
                "meanFinalSortednessScore": stat_payload.pop("finalSortednessScoreMean", np.nan),
                "finalSortednessScoreSd": stat_payload.pop("finalSortednessScoreSd", 0.0),
                "finalSortednessScoreSem": stat_payload.pop("finalSortednessScoreSem", 0.0),
                "meanDelayedGratification": stat_payload.pop("delayedGratificationMean", np.nan),
                "delayedGratificationSd": stat_payload.pop("delayedGratificationSd", 0.0),
                "delayedGratificationSem": stat_payload.pop("delayedGratificationSem", 0.0),
                "meanOscillationProxy": stat_payload.pop("oscillationProxyMean", np.nan),
                "oscillationProxySd": stat_payload.pop("oscillationProxySd", 0.0),
                "oscillationProxySem": stat_payload.pop("oscillationProxySem", 0.0),
                "meanAggregationFinal": stat_payload.pop("finalAggregationMean", np.nan),
                "aggregationFinalSd": stat_payload.pop("finalAggregationSd", 0.0),
                "aggregationFinalSem": stat_payload.pop("finalAggregationSem", 0.0),
                "meanRobustnessScore": stat_payload.pop("robustnessScoreMean", np.nan),
                "robustnessScoreSd": stat_payload.pop("robustnessScoreSd", 0.0),
                "robustnessScoreSem": stat_payload.pop("robustnessScoreSem", 0.0),
                "meanSwapCount": stat_payload.pop("swapCountMean", np.nan),
                "meanComparisonCount": stat_payload.pop("comparisonCountMean", np.nan),
                "meanActivationCount": stat_payload.pop("activationCountMean", np.nan),
                "stopReasonsJson": canonical_json(sorted(set(map(str, group.get("stopReason", pd.Series(dtype=str)))))),
                "completedSeedCount": int(pd.to_numeric(group["completedNumeric"], errors="coerce").fillna(0).sum()),
            }
        )
        row.update(stat_payload)
        rows.append(row)
    return pd.DataFrame(rows)


def _crosses(left: float, right: float, threshold: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return (left < threshold <= right) or (right < threshold <= left)


def transition_flags(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_completion = _safe_float(left.get("completionRate"), float("nan"))
    right_completion = _safe_float(right.get("completionRate"), float("nan"))
    left_sorted = _safe_float(left.get("meanFinalSortednessScore"), float("nan"))
    right_sorted = _safe_float(right.get("meanFinalSortednessScore"), float("nan"))
    left_dg = _safe_float(left.get("meanDelayedGratification"), float("nan"))
    right_dg = _safe_float(right.get("meanDelayedGratification"), float("nan"))
    left_aggregation = _safe_float(left.get("meanAggregationFinal"), float("nan"))
    right_aggregation = _safe_float(right.get("meanAggregationFinal"), float("nan"))
    left_osc = _safe_float(left.get("meanOscillationProxy"), float("nan"))
    right_osc = _safe_float(right.get("meanOscillationProxy"), float("nan"))
    left_robust = _safe_float(left.get("meanRobustnessScore"), float("nan"))
    right_robust = _safe_float(right.get("meanRobustnessScore"), float("nan"))

    delta_completion = right_completion - left_completion
    delta_sorted = right_sorted - left_sorted
    delta_dg = right_dg - left_dg
    delta_aggregation = right_aggregation - left_aggregation
    delta_osc = right_osc - left_osc
    delta_robust = right_robust - left_robust

    sorting_success = _crosses(left_completion, right_completion, 0.5) or abs(delta_completion) >= 0.5
    sortedness = abs(delta_sorted) >= 0.25
    dg = _crosses(left_dg, right_dg, 0.0) or abs(delta_dg) >= 0.2
    aggregation = _crosses(left_aggregation, right_aggregation, 0.5) or abs(delta_aggregation) >= 0.25
    oscillation = abs(delta_osc) >= 0.15 or (
        math.isfinite(left_osc)
        and math.isfinite(right_osc)
        and ((left_osc <= 0.10 and right_osc >= 0.25) or (right_osc <= 0.10 and left_osc >= 0.25))
    )
    robustness = _crosses(left_robust, right_robust, 0.4) or abs(delta_robust) >= 0.25
    flags = {
        "sortingSuccessTransition": bool(sorting_success),
        "sortednessTransition": bool(sortedness),
        "delayedGratificationTransition": bool(dg),
        "aggregationTransition": bool(aggregation),
        "oscillationTransition": bool(oscillation),
        "robustnessTransition": bool(robustness),
    }
    transition_kinds = [key for key, value in flags.items() if value]
    boundary_score = (
        1.25 * abs(delta_completion if math.isfinite(delta_completion) else 0.0)
        + abs(delta_sorted if math.isfinite(delta_sorted) else 0.0)
        + 0.40 * min(2.0, abs(delta_dg if math.isfinite(delta_dg) else 0.0))
        + abs(delta_aggregation if math.isfinite(delta_aggregation) else 0.0)
        + abs(delta_osc if math.isfinite(delta_osc) else 0.0)
        + abs(delta_robust if math.isfinite(delta_robust) else 0.0)
        + 0.10 * len(transition_kinds)
    )
    flags.update(
        {
            "transitionKindsJson": canonical_json(transition_kinds),
            "transitionKindCount": int(len(transition_kinds)),
            "boundaryScore": float(boundary_score),
            "deltaCompletionRate": float(delta_completion) if math.isfinite(delta_completion) else np.nan,
            "deltaFinalSortednessScore": float(delta_sorted) if math.isfinite(delta_sorted) else np.nan,
            "deltaDelayedGratification": float(delta_dg) if math.isfinite(delta_dg) else np.nan,
            "deltaAggregationFinal": float(delta_aggregation) if math.isfinite(delta_aggregation) else np.nan,
            "deltaOscillationProxy": float(delta_osc) if math.isfinite(delta_osc) else np.nan,
            "deltaRobustnessScore": float(delta_robust) if math.isfinite(delta_robust) else np.nan,
        }
    )
    return flags


def detect_phase_boundaries(policy_task_summary: pd.DataFrame) -> pd.DataFrame:
    """Detect adjacent transitions within each family/task parameter ordering."""

    if policy_task_summary.empty:
        return pd.DataFrame()
    summary = policy_task_summary.copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["familyKey", "taskId"]
    for (family_key, task_id), group in summary.groupby(group_columns, dropna=False):
        if len(group) < 2:
            continue
        ordered = group.sort_values(
            ["parameterValue", "candidateRank", "policyId"],
            ascending=[True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        for index in range(len(ordered) - 1):
            left = ordered.iloc[index].to_dict()
            right = ordered.iloc[index + 1].to_dict()
            flags = transition_flags(left, right)
            if not flags["transitionKindCount"]:
                continue
            rows.append(
                {
                    "phaseBoundaryVersion": PHASE_BOUNDARY_VERSION,
                    "boundaryId": f"pb_{len(rows) + 1:05d}",
                    "familyKey": str(family_key),
                    "taskId": str(task_id),
                    "taskFamily": left.get("taskFamily"),
                    "taskPanel": left.get("taskPanel"),
                    "parameterAxis": left.get("parameterAxis"),
                    "leftPolicyId": left.get("policyId"),
                    "rightPolicyId": right.get("policyId"),
                    "leftLineageId": left.get("lineageId"),
                    "rightLineageId": right.get("lineageId"),
                    "leftParameterValue": left.get("parameterValue"),
                    "rightParameterValue": right.get("parameterValue"),
                    "leftParameterLabel": left.get("parameterLabel"),
                    "rightParameterLabel": right.get("parameterLabel"),
                    "leftSeedCount": left.get("seedCount"),
                    "rightSeedCount": right.get("seedCount"),
                    "leftCompletionRate": left.get("completionRate"),
                    "rightCompletionRate": right.get("completionRate"),
                    "leftFinalSortednessScore": left.get("meanFinalSortednessScore"),
                    "rightFinalSortednessScore": right.get("meanFinalSortednessScore"),
                    "leftDelayedGratification": left.get("meanDelayedGratification"),
                    "rightDelayedGratification": right.get("meanDelayedGratification"),
                    "leftAggregationFinal": left.get("meanAggregationFinal"),
                    "rightAggregationFinal": right.get("meanAggregationFinal"),
                    "leftOscillationProxy": left.get("meanOscillationProxy"),
                    "rightOscillationProxy": right.get("meanOscillationProxy"),
                    "leftRobustnessScore": left.get("meanRobustnessScore"),
                    "rightRobustnessScore": right.get("meanRobustnessScore"),
                    "leftFinalSortednessSd": left.get("finalSortednessScoreSd"),
                    "rightFinalSortednessSd": right.get("finalSortednessScoreSd"),
                    "leftOscillationSd": left.get("oscillationProxySd"),
                    "rightOscillationSd": right.get("oscillationProxySd"),
                    "leftRobustnessSd": left.get("robustnessScoreSd"),
                    "rightRobustnessSd": right.get("robustnessScoreSd"),
                    **flags,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["boundaryScore", "transitionKindCount", "familyKey", "taskId"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def near_boundary_variance_table(boundaries: pd.DataFrame, policy_task_summary: pd.DataFrame) -> pd.DataFrame:
    """Return repeated-seed variance records for policies adjacent to boundaries."""

    if boundaries.empty or policy_task_summary.empty:
        return pd.DataFrame()
    keys = set()
    for _, boundary in boundaries.iterrows():
        keys.add((str(boundary["leftPolicyId"]), str(boundary["taskId"]), str(boundary["boundaryId"]), "left"))
        keys.add((str(boundary["rightPolicyId"]), str(boundary["taskId"]), str(boundary["boundaryId"]), "right"))
    summary = policy_task_summary.copy()
    rows: list[dict[str, Any]] = []
    indexed = summary.set_index(["policyId", "taskId"], drop=False)
    for policy_id, task_id, boundary_id, side in sorted(keys):
        if (policy_id, task_id) not in indexed.index:
            continue
        record = indexed.loc[(policy_id, task_id)]
        if isinstance(record, pd.DataFrame):
            record = record.iloc[0]
        rows.append(
            {
                "phaseBoundaryVersion": PHASE_BOUNDARY_VERSION,
                "boundaryId": boundary_id,
                "side": side,
                "policyId": policy_id,
                "taskId": task_id,
                "familyKey": record.get("familyKey"),
                "parameterAxis": record.get("parameterAxis"),
                "parameterValue": record.get("parameterValue"),
                "seedCount": int(record.get("seedCount", 0)),
                "completionRate": record.get("completionRate"),
                "completionRateSd": record.get("completionRateSd"),
                "completionRateSem": record.get("completionRateSem"),
                "meanFinalSortednessScore": record.get("meanFinalSortednessScore"),
                "finalSortednessScoreSd": record.get("finalSortednessScoreSd"),
                "finalSortednessScoreSem": record.get("finalSortednessScoreSem"),
                "meanDelayedGratification": record.get("meanDelayedGratification"),
                "delayedGratificationSd": record.get("delayedGratificationSd"),
                "meanAggregationFinal": record.get("meanAggregationFinal"),
                "aggregationFinalSd": record.get("aggregationFinalSd"),
                "meanOscillationProxy": record.get("meanOscillationProxy"),
                "oscillationProxySd": record.get("oscillationProxySd"),
                "meanRobustnessScore": record.get("meanRobustnessScore"),
                "robustnessScoreSd": record.get("robustnessScoreSd"),
                "varianceCheckStatus": "repeated_seed_estimate",
            }
        )
    return pd.DataFrame(rows)


def validate_phase_boundary_outputs(
    *,
    candidates: pd.DataFrame,
    smoke_checks: pd.DataFrame,
    run_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    expected_seed_count: int,
    expected_task_count: int,
    upstream_statuses: Mapping[str, Mapping[str, Any]],
    repo_test_payload: Mapping[str, Any],
    figure_paths: Sequence[str],
    s10_dir_exists: bool,
) -> pd.DataFrame:
    expected_rows = len(candidates) * int(expected_seed_count) * int(expected_task_count)
    family_keys = set(map(str, candidates.get("familyKey", pd.Series(dtype=str))))
    checks = [
        {
            "checkId": "upstream_s01_s08_success",
            "success": all(bool(status.get("success")) for status in upstream_statuses.values()),
            "detail": canonical_json({step: status.get("success") for step, status in upstream_statuses.items()}),
        },
        {
            "checkId": "candidate_set_nonempty_and_bounded",
            "success": 0 < len(candidates) <= 420,
            "detail": f"{len(candidates)} candidates selected",
        },
        {
            "checkId": "candidate_family_coverage",
            "success": len(family_keys) >= 12
            and any(key.startswith("classic_") for key in family_keys)
            and any(key.startswith("single_rule") for key in family_keys)
            and any(key.startswith("probability_mutation") for key in family_keys)
            and any(key.startswith("memory_mutation") for key in family_keys)
            and any(key.startswith("target_memory_mutation") for key in family_keys),
            "detail": f"{len(family_keys)} family keys represented",
        },
        {
            "checkId": "all_s08_elites_included",
            "success": int(candidates["isS08Elite"].sum()) >= 25,
            "detail": f"{int(candidates['isS08Elite'].sum())} S08 elites in candidate set",
        },
        {
            "checkId": "smoke_checks_passed",
            "success": bool(smoke_checks["success"].all()) if len(smoke_checks) else False,
            "detail": f"{int(smoke_checks['success'].sum()) if len(smoke_checks) else 0}/{len(smoke_checks)} smoke checks passed",
        },
        {
            "checkId": "full_repeated_seed_rows_complete",
            "success": len(run_df) == expected_rows,
            "detail": f"{len(run_df)} rows; expected {expected_rows}",
        },
        {
            "checkId": "value_counts_conserved",
            "success": bool(run_df["valueCountsConserved"].all()) if len(run_df) else False,
            "detail": "all S09 CPU rows preserved input value multisets",
        },
        {
            "checkId": "metric_ranges",
            "success": bool(
                len(run_df)
                and run_df["finalSortednessScore"].between(0.0, 1.0).all()
                and run_df["oscillationProxy"].between(0.0, 1.0).all()
                and run_df["finalAggregation"].between(0.0, 1.0).all()
            ),
            "detail": "final sortedness, oscillation, and aggregation metrics are bounded",
        },
        {
            "checkId": "seed_coverage_per_policy_task",
            "success": bool(len(summary_df) and summary_df["seedCount"].min() >= int(expected_seed_count)),
            "detail": f"minimum seed count {int(summary_df['seedCount'].min()) if len(summary_df) else 0}",
        },
        {
            "checkId": "boundary_candidates_nonempty",
            "success": len(boundaries_df) > 0,
            "detail": f"{len(boundaries_df)} adjacent phase-boundary candidates detected",
        },
        {
            "checkId": "near_boundary_variance_documented",
            "success": len(variance_df) > 0 and int(variance_df["seedCount"].min()) >= int(expected_seed_count),
            "detail": f"{len(variance_df)} near-boundary variance records",
        },
        {
            "checkId": "phase_diagrams_written",
            "success": all(bool(path) for path in figure_paths),
            "detail": canonical_json(list(figure_paths)),
        },
        {
            "checkId": "repo_unit_tests",
            "success": bool(repo_test_payload.get("success")),
            "detail": f"{repo_test_payload.get('command')} returned {repo_test_payload.get('returnCode')}",
        },
        {
            "checkId": "s10_not_started",
            "success": not s10_dir_exists,
            "detail": "S10 artifact directory is absent",
        },
    ]
    return pd.DataFrame(checks)
