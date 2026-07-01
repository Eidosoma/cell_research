"""Phase-boundary sweep utilities for E03 S09.

S09 is a focused follow-up to the S07/S08 proxy screens.  It keeps the same
DSL local-step semantics but records compact trajectories so transitions,
stalls, and oscillation candidates can be detected around promising archive
regions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.coarse_sweep import actor_schedule, initial_values as coarse_initial_values, sortedness_metrics
from src.e03.gpu_batch_simulator import compatibility_record_for_policy, load_generated_policy_records
from src.e03.policy_generation import policy_feature_flags, semantic_hash
from src.e03.rule_dsl import DSLAction, DSLArrayState, DSLCondition, DSLInterpreter, DSLPolicy, DSLRule, parse_policy, validate_policy


PHASE_SCHEMA = "eidosoma.e03.phase_boundary_sweep.v1"
DEFAULT_PHASE_SEED = 2026070109
COMPETENCE_THRESHOLD = 0.90

PRIMARY_SEEDS = (7101, 7102, 7103)
REPLICATE_SEEDS = (9101, 9102, 9103)
EVENT_CAP_VALUES = (8, 16, 24, 32, 48, 64, 96, 128)
ARRAY_SIZE_VALUES = (8, 12, 16, 24, 32)
INPUT_PROFILE_VALUES = ("nearly_sorted_1swap", "nearly_sorted_3swap", "random_permutation", "reverse")
PROBABILITY_VALUES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


@dataclass(frozen=True)
class PhasePolicy:
    """One policy selected for S09 boundary sweeps."""

    base_policy_id: str
    policy_name: str
    source_kind: str
    dsl_source: str
    route: str
    quality_score: float
    archive_winner: bool
    nonclassic_cell: bool
    selection_reason: str

    @property
    def policy(self) -> DSLPolicy:
        return parse_policy(self.dsl_source)


@dataclass(frozen=True)
class PhaseConfig:
    """One S09 sweep condition."""

    config_id: str
    axis_name: str
    axis_value: str
    axis_numeric: float
    array_size: int
    event_cap: int
    seed: int
    split: str
    input_profile: str = "random_permutation"
    probability_override: float | None = None


def format_probability(value: float) -> str:
    text = f"{float(value):.6g}"
    return "0" if text == "-0" else text


def stable_digest(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    payload: list[dict[str, Any]] = []
    for record in records:
        item: dict[str, Any] = {}
        for column in columns:
            value = record.get(column)
            if isinstance(value, float) and not pd.isna(value):
                value = round(float(value), 12)
            item[column] = value
        payload.append(item)
    payload = sorted(payload, key=lambda item: json.dumps(item, sort_keys=True))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def policy_has_probability(policy: DSLPolicy) -> bool:
    for rule in policy.rules:
        if any(condition.name == "random_lt" for condition in rule.conditions):
            return True
        if any(action.name == "choose" for action in rule.actions):
            return True
    return False


def set_policy_probability(policy: DSLPolicy, probability: float) -> DSLPolicy:
    """Replace all random guards and choose probabilities with one value."""

    prob = format_probability(probability)
    rules: list[DSLRule] = []
    for rule in policy.rules:
        conditions = tuple(
            DSLCondition(condition.name, (prob,) if condition.name == "random_lt" else condition.args)
            for condition in rule.conditions
        )
        actions = tuple(
            DSLAction(action.name, (prob, *action.args[1:]) if action.name == "choose" else action.args)
            for action in rule.actions
        )
        rules.append(DSLRule(conditions=conditions, actions=actions, is_else=rule.is_else))
    name = f"s09_prob_{policy.sha256[:8]}_{prob.replace('.', 'p')}"
    variant = DSLPolicy(name=name, version=policy.version, state_init=policy.state_init, rules=tuple(rules))
    validate_policy(variant)
    return variant


def initial_values(array_size: int, seed: int, input_profile: str) -> tuple[int, ...]:
    """Return deterministic arrays for phase-boundary stress profiles."""

    if input_profile == "random_permutation":
        return coarse_initial_values(array_size, seed, input_profile)
    if input_profile == "reverse":
        return tuple(range(array_size, 0, -1))
    if input_profile.startswith("nearly_sorted_") and input_profile.endswith("swap"):
        count_text = input_profile.removeprefix("nearly_sorted_").removesuffix("swap")
        swap_count = int(count_text)
        values = list(range(1, array_size + 1))
        rng = np.random.default_rng(seed)
        for _ in range(swap_count):
            left, right = rng.choice(array_size, size=2, replace=False)
            values[int(left)], values[int(right)] = values[int(right)], values[int(left)]
        return tuple(int(value) for value in values)
    raise ValueError(f"Unsupported input_profile: {input_profile}")


def rng_for_policy_config(policy: DSLPolicy, config: PhaseConfig) -> random.Random:
    return random.Random((config.seed * 1000003) ^ int(policy.sha256[:12], 16))


def classify_failure(
    *,
    final_is_sorted: bool,
    final_sortedness: float,
    initial_sortedness: float,
    swap_count: int,
    update_count: int,
    event_cap: int,
    no_change_event_count: int,
    cycle_detected: bool,
    two_cycle_count: int,
) -> str:
    if final_is_sorted:
        return "sorted"
    if cycle_detected and swap_count > 4 and (two_cycle_count / max(event_cap, 1)) >= 0.05:
        return "timeout_oscillation_candidate"
    if (swap_count + update_count) == 0 or (no_change_event_count / max(event_cap, 1)) >= 0.90:
        return "timeout_stalled"
    if final_sortedness < initial_sortedness - 0.05:
        return "timeout_degraded"
    if final_sortedness > initial_sortedness + 0.05:
        return "timeout_partial_progress"
    return "timeout_no_clear_progress"


def run_phase_config(phase_policy: PhasePolicy, config: PhaseConfig) -> dict[str, Any]:
    """Run one policy/config with compact trajectory diagnostics."""

    base_policy = phase_policy.policy
    policy = set_policy_probability(base_policy, config.probability_override) if config.probability_override is not None else base_policy
    compatibility = compatibility_record_for_policy(policy)
    compiles_for_batch = bool(compatibility.get("compiles_for_batch", False))
    requires_cpu_fallback = bool(compatibility.get("requires_cpu_fallback", True))
    route = "jax_batch" if compiles_for_batch and not requires_cpu_fallback else "cpu_fallback"
    values = initial_values(config.array_size, config.seed, config.input_profile)
    initial = sortedness_metrics(values)
    interpreter = DSLInterpreter(policy)
    rng = rng_for_policy_config(policy, config)
    schedule = actor_schedule(config.array_size, config.event_cap, config.seed)
    statuses = tuple("ACTIVE" for _ in values)
    ideal_position: int | None = None
    current_values = values
    compare_count = 0
    swap_count = 0
    update_count = 0
    wait_count = 0
    first_sorted_event: int | None = None
    min_sortedness = float(initial["inversion_sortedness"])
    max_sortedness = float(initial["inversion_sortedness"])
    no_change_event_count = 0
    seen_states: dict[tuple[tuple[int, ...], int | None], int] = {(current_values, ideal_position): 0}
    cycle_detected = False
    cycle_period: int | None = None
    two_cycle_count = 0
    history: list[tuple[int, tuple[int, ...], int | None]] = [(0, current_values, ideal_position)]
    error_message = ""
    execution_status = "ok"
    try:
        for event_index, actor_index in enumerate(schedule, start=1):
            state = DSLArrayState(
                values=current_values,
                statuses=statuses,
                actor_index=int(actor_index),
                ideal_position=ideal_position,
            )
            before_key = (state.values, state.ideal_position)
            result = interpreter.step_state(state, rng)
            current_values = result.state_after.values
            statuses = tuple(result.state_after.statuses or statuses)
            ideal_position = result.state_after.ideal_position
            after_key = (current_values, ideal_position)
            compare_count += int(bool(result.action.compare_counted))
            swap_count += int(result.action.action_type == "swap")
            update_count += int(result.action.action_type == "update_state")
            wait_count += int(result.action.action_type == "wait")
            if before_key == after_key:
                no_change_event_count += 1
            if len(history) >= 2 and after_key == (history[-2][1], history[-2][2]) and result.action.action_type == "swap":
                two_cycle_count += 1
            if after_key in seen_states and result.action.action_type in {"swap", "update_state"} and not cycle_detected:
                cycle_detected = True
                cycle_period = event_index - seen_states[after_key]
            seen_states.setdefault(after_key, event_index)
            history.append((event_index, current_values, ideal_position))
            sortedness = float(sortedness_metrics(current_values)["inversion_sortedness"])
            min_sortedness = min(min_sortedness, sortedness)
            max_sortedness = max(max_sortedness, sortedness)
            if first_sorted_event is None and sortedness_metrics(current_values)["is_sorted"]:
                first_sorted_event = event_index
    except Exception as exc:  # pragma: no cover - script records invalid rows
        execution_status = "invalid"
        error_message = repr(exc)

    final = sortedness_metrics(current_values)
    final_sortedness = float(final["inversion_sortedness"])
    initial_sortedness = float(initial["inversion_sortedness"])
    dg_drop = max(0.0, initial_sortedness - min_sortedness)
    dg_present = bool(dg_drop >= 0.05 and final_sortedness >= initial_sortedness + 0.10)
    competent = bool(final_sortedness >= COMPETENCE_THRESHOLD)
    failure_mode = "invalid" if execution_status != "ok" else classify_failure(
        final_is_sorted=bool(final["is_sorted"]),
        final_sortedness=final_sortedness,
        initial_sortedness=initial_sortedness,
        swap_count=swap_count,
        update_count=update_count,
        event_cap=config.event_cap,
        no_change_event_count=no_change_event_count,
        cycle_detected=cycle_detected,
        two_cycle_count=two_cycle_count,
    )
    row = {
        "schema": PHASE_SCHEMA,
        "experiment_id": "E03",
        "research_step_id": "S09",
        "base_policy_id": phase_policy.base_policy_id,
        "base_policy_name": phase_policy.policy_name,
        "base_source_kind": phase_policy.source_kind,
        "selection_reason": phase_policy.selection_reason,
        "archive_winner": bool(phase_policy.archive_winner),
        "nonclassic_cell": bool(phase_policy.nonclassic_cell),
        "s08_quality_score": float(phase_policy.quality_score) if not pd.isna(phase_policy.quality_score) else np.nan,
        "variant_policy_id": policy.policy_id,
        "variant_policy_name": policy.name,
        "variant_dsl_sha256": policy.sha256,
        "variant_semantic_hash": semantic_hash(policy),
        "route": route,
        "compiles_for_batch": compiles_for_batch,
        "requires_cpu_fallback": requires_cpu_fallback,
        "fallback_reasons_json": str(compatibility.get("fallback_reasons_json", "[]")),
        "config_id": config.config_id,
        "axis_name": config.axis_name,
        "axis_value": config.axis_value,
        "axis_numeric": float(config.axis_numeric),
        "array_size": int(config.array_size),
        "event_cap": int(config.event_cap),
        "seed": int(config.seed),
        "split": config.split,
        "input_profile": config.input_profile,
        "probability_override": np.nan if config.probability_override is None else float(config.probability_override),
        "execution_status": execution_status,
        "error_message": error_message[:500],
        "events_executed": int(config.event_cap if execution_status == "ok" else 0),
        "compare_count": int(compare_count),
        "swap_count": int(swap_count),
        "update_count": int(update_count),
        "wait_count": int(wait_count),
        "work_count": int(compare_count + swap_count + update_count),
        "initial_inversion_count": int(initial["inversion_count"]),
        "final_inversion_count": int(final["inversion_count"]),
        "initial_inversion_sortedness": initial_sortedness,
        "final_inversion_sortedness": final_sortedness,
        "inversion_sortedness_delta": final_sortedness - initial_sortedness,
        "max_inversion_sortedness": float(max_sortedness),
        "min_inversion_sortedness": float(min_sortedness),
        "dg_drop": float(dg_drop),
        "dg_present": dg_present,
        "competent_at_threshold": competent,
        "final_is_sorted": bool(final["is_sorted"]),
        "first_sorted_event": -1 if first_sorted_event is None else int(first_sorted_event),
        "timed_out": bool(execution_status == "ok" and not final["is_sorted"]),
        "failure_mode": failure_mode,
        "cycle_detected": bool(cycle_detected),
        "cycle_period": -1 if cycle_period is None else int(cycle_period),
        "two_cycle_count": int(two_cycle_count),
        "oscillation_score": float(two_cycle_count / max(config.event_cap, 1)),
        "no_change_event_fraction": float(no_change_event_count / max(config.event_cap, 1)),
        "distinct_state_fraction": float(len(seen_states) / max(config.event_cap + 1, 1)),
        "initial_values_json": json.dumps(list(values), separators=(",", ":")),
        "final_values_head_json": json.dumps(list(current_values)[:20], separators=(",", ":")),
        "final_values_tail_json": json.dumps(list(current_values)[-20:], separators=(",", ":")),
        "variant_dsl_source": policy.to_source(),
    }
    return row


def run_phase_sweep(policies: Sequence[PhasePolicy], configs_by_policy: Mapping[str, Sequence[PhaseConfig]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for config in configs_by_policy.get(policy.base_policy_id, ()):
            rows.append(run_phase_config(policy, config))
    return pd.DataFrame(rows)


def load_discovered_policies(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def select_phase_policies(
    *,
    policy_library: Path,
    discovered_jsonl: Path,
    candidate_count: int = 10,
) -> list[PhasePolicy]:
    """Select classic landmarks plus S08 QD policies for S09 sweeps."""

    selected: list[PhasePolicy] = []
    seen: set[str] = set()

    def add(policy: PhasePolicy) -> None:
        if policy.base_policy_id not in seen:
            selected.append(policy)
            seen.add(policy.base_policy_id)

    library_records = load_generated_policy_records(policy_library)
    classics = [record for record in library_records if record.get("sourceKind") == "classic_dsl_seed"]
    for record in classics:
        name = str(record["policyName"])
        if "increasing" not in name:
            continue
        policy = parse_policy(record["dslSource"])
        comp = compatibility_record_for_policy(policy)
        add(
            PhasePolicy(
                base_policy_id=str(record["policyId"]),
                policy_name=name,
                source_kind="classic_dsl_seed",
                dsl_source=str(record["dslSource"]),
                route="jax_batch" if comp.get("compiles_for_batch", False) and not comp.get("requires_cpu_fallback", True) else "cpu_fallback",
                quality_score=float("nan"),
                archive_winner=False,
                nonclassic_cell=False,
                selection_reason="classic_increasing_landmark",
            )
        )
        if len([item for item in selected if item.source_kind == "classic_dsl_seed"]) >= 3:
            break

    discovered = load_discovered_policies(discovered_jsonl)
    discovered_df = pd.DataFrame(discovered)
    if not discovered_df.empty:
        discovered_df = discovered_df.sort_values(["archiveWinner", "nonclassicCell", "qualityScore"], ascending=[False, False, False])
        for record in discovered_df.to_dict(orient="records"):
            add(
                PhasePolicy(
                    base_policy_id=str(record["policyId"]),
                    policy_name=str(record["policyName"]),
                    source_kind=str(record["sourceKind"]),
                    dsl_source=str(record["dslSource"]),
                    route=str(record["route"]),
                    quality_score=float(record["qualityScore"]),
                    archive_winner=bool(record["archiveWinner"]),
                    nonclassic_cell=bool(record["nonclassicCell"]),
                    selection_reason="s08_archive_winner_or_top_discovered",
                )
            )
            if len(selected) >= candidate_count:
                break
        stochastic = discovered_df[discovered_df["dslSource"].str.contains("random_lt|choose", regex=True, na=False)]
        for record in stochastic.to_dict(orient="records"):
            add(
                PhasePolicy(
                    base_policy_id=str(record["policyId"]),
                    policy_name=str(record["policyName"]),
                    source_kind=str(record["sourceKind"]),
                    dsl_source=str(record["dslSource"]),
                    route=str(record["route"]),
                    quality_score=float(record["qualityScore"]),
                    archive_winner=bool(record["archiveWinner"]),
                    nonclassic_cell=bool(record["nonclassicCell"]),
                    selection_reason="s08_stochastic_probability_axis",
                )
            )
            if len([item for item in selected if policy_has_probability(item.policy)]) >= 3:
                break
    return selected[:candidate_count]


def primary_configs_for_policy(policy: PhasePolicy) -> list[PhaseConfig]:
    configs: list[PhaseConfig] = []
    for event_cap in EVENT_CAP_VALUES:
        for seed in PRIMARY_SEEDS:
            configs.append(
                PhaseConfig(
                    config_id=f"{policy.base_policy_id.replace(':', '_')}_event_cap_{event_cap}_seed{seed}",
                    axis_name="event_cap",
                    axis_value=str(event_cap),
                    axis_numeric=float(event_cap),
                    array_size=16,
                    event_cap=int(event_cap),
                    seed=int(seed),
                    split="primary",
                )
            )
    for array_size in ARRAY_SIZE_VALUES:
        for seed in PRIMARY_SEEDS:
            configs.append(
                PhaseConfig(
                    config_id=f"{policy.base_policy_id.replace(':', '_')}_array_size_{array_size}_seed{seed}",
                    axis_name="array_size",
                    axis_value=str(array_size),
                    axis_numeric=float(array_size),
                    array_size=int(array_size),
                    event_cap=int(array_size * 4),
                    seed=int(seed + array_size),
                    split="primary",
                )
            )
    for index, profile in enumerate(INPUT_PROFILE_VALUES):
        for seed in PRIMARY_SEEDS:
            configs.append(
                PhaseConfig(
                    config_id=f"{policy.base_policy_id.replace(':', '_')}_input_{profile}_seed{seed}",
                    axis_name="input_disorder",
                    axis_value=profile,
                    axis_numeric=float(index),
                    array_size=16,
                    event_cap=64,
                    seed=int(seed + 200 + index),
                    split="primary",
                    input_profile=profile,
                )
            )
    if policy_has_probability(policy.policy):
        for probability in PROBABILITY_VALUES:
            for seed in PRIMARY_SEEDS:
                configs.append(
                    PhaseConfig(
                        config_id=f"{policy.base_policy_id.replace(':', '_')}_prob_{format_probability(probability).replace('.', 'p')}_seed{seed}",
                        axis_name="probability",
                        axis_value=format_probability(probability),
                        axis_numeric=float(probability),
                        array_size=16,
                        event_cap=64,
                        seed=int(seed + 400),
                        split="primary",
                        probability_override=float(probability),
                    )
                )
    return configs


def build_primary_configs(policies: Sequence[PhasePolicy]) -> dict[str, list[PhaseConfig]]:
    return {policy.base_policy_id: primary_configs_for_policy(policy) for policy in policies}


def aggregate_axis(frame: pd.DataFrame, *, split: str = "primary") -> pd.DataFrame:
    data = frame[frame["split"] == split].copy()
    if data.empty:
        return pd.DataFrame()
    grouped = data.groupby(["base_policy_id", "base_policy_name", "axis_name", "axis_value", "axis_numeric"], dropna=False)
    return (
        grouped.agg(
            run_count=("config_id", "size"),
            mean_final_sortedness=("final_inversion_sortedness", "mean"),
            mean_delta=("inversion_sortedness_delta", "mean"),
            success_fraction=("competent_at_threshold", "mean"),
            sorted_fraction=("final_is_sorted", "mean"),
            dg_fraction=("dg_present", "mean"),
            oscillation_fraction=("cycle_detected", "mean"),
            mean_oscillation_score=("oscillation_score", "mean"),
            dominant_failure_mode=("failure_mode", lambda values: values.value_counts().idxmax()),
        )
        .reset_index()
        .sort_values(["base_policy_id", "axis_name", "axis_numeric"], kind="mergesort")
    )


def detect_boundaries(axis_summary: pd.DataFrame, *, max_per_axis: int = 2) -> pd.DataFrame:
    """Detect adjacent value pairs with competence, DG, or oscillation changes."""

    rows: list[dict[str, Any]] = []
    ordered_axes = {"event_cap", "array_size", "probability"}
    for (policy_id, axis_name), group in axis_summary.groupby(["base_policy_id", "axis_name"], dropna=False):
        group = group.sort_values("axis_numeric", kind="mergesort").reset_index(drop=True)
        if axis_name in ordered_axes and len(group) >= 2:
            candidates: list[dict[str, Any]] = []
            for idx in range(len(group) - 1):
                left = group.iloc[idx]
                right = group.iloc[idx + 1]
                success_delta = float(right["success_fraction"] - left["success_fraction"])
                sortedness_delta = float(right["mean_final_sortedness"] - left["mean_final_sortedness"])
                dg_delta = float(right["dg_fraction"] - left["dg_fraction"])
                osc_delta = float(right["oscillation_fraction"] - left["oscillation_fraction"])
                crosses_success = (left["success_fraction"] < 0.5 <= right["success_fraction"]) or (left["success_fraction"] >= 0.5 > right["success_fraction"])
                crosses_dg = (left["dg_fraction"] < 0.5 <= right["dg_fraction"]) or (left["dg_fraction"] >= 0.5 > right["dg_fraction"])
                crosses_osc = (left["oscillation_fraction"] < 0.5 <= right["oscillation_fraction"]) or (left["oscillation_fraction"] >= 0.5 > right["oscillation_fraction"])
                strength = max(abs(success_delta), abs(sortedness_delta), abs(dg_delta), abs(osc_delta))
                if crosses_success:
                    kind = "competence_transition"
                elif crosses_dg:
                    kind = "dg_transition"
                elif crosses_osc:
                    kind = "oscillation_transition"
                elif strength >= 0.15:
                    kind = "gradient_transition"
                else:
                    kind = "weak_gradient"
                candidates.append(
                    {
                        "base_policy_id": policy_id,
                        "base_policy_name": left["base_policy_name"],
                        "axis_name": axis_name,
                        "lower_axis_value": left["axis_value"],
                        "upper_axis_value": right["axis_value"],
                        "lower_axis_numeric": float(left["axis_numeric"]),
                        "upper_axis_numeric": float(right["axis_numeric"]),
                        "boundary_kind": kind,
                        "transition_strength": float(strength),
                        "lower_success_fraction": float(left["success_fraction"]),
                        "upper_success_fraction": float(right["success_fraction"]),
                        "lower_mean_final_sortedness": float(left["mean_final_sortedness"]),
                        "upper_mean_final_sortedness": float(right["mean_final_sortedness"]),
                        "lower_dg_fraction": float(left["dg_fraction"]),
                        "upper_dg_fraction": float(right["dg_fraction"]),
                        "lower_oscillation_fraction": float(left["oscillation_fraction"]),
                        "upper_oscillation_fraction": float(right["oscillation_fraction"]),
                    }
                )
            ranked = sorted(
                candidates,
                key=lambda item: (
                    item["boundary_kind"] == "weak_gradient",
                    -item["transition_strength"],
                    item["lower_axis_numeric"],
                ),
            )
            rows.extend(ranked[:max_per_axis])
        elif axis_name == "input_disorder" and len(group) >= 2:
            ranked = group.sort_values("mean_final_sortedness", kind="mergesort").reset_index(drop=True)
            low = ranked.iloc[0]
            high = ranked.iloc[-1]
            rows.append(
                {
                    "base_policy_id": policy_id,
                    "base_policy_name": low["base_policy_name"],
                    "axis_name": axis_name,
                    "lower_axis_value": low["axis_value"],
                    "upper_axis_value": high["axis_value"],
                    "lower_axis_numeric": float(low["axis_numeric"]),
                    "upper_axis_numeric": float(high["axis_numeric"]),
                    "boundary_kind": "input_disorder_sensitivity",
                    "transition_strength": float(high["mean_final_sortedness"] - low["mean_final_sortedness"]),
                    "lower_success_fraction": float(low["success_fraction"]),
                    "upper_success_fraction": float(high["success_fraction"]),
                    "lower_mean_final_sortedness": float(low["mean_final_sortedness"]),
                    "upper_mean_final_sortedness": float(high["mean_final_sortedness"]),
                    "lower_dg_fraction": float(low["dg_fraction"]),
                    "upper_dg_fraction": float(high["dg_fraction"]),
                    "lower_oscillation_fraction": float(low["oscillation_fraction"]),
                    "upper_oscillation_fraction": float(high["oscillation_fraction"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["transition_strength", "axis_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _replicate_config(policy_id: str, axis_name: str, axis_value: str, axis_numeric: float, seed: int) -> PhaseConfig:
    safe_id = policy_id.replace(":", "_")
    if axis_name == "event_cap":
        event_cap = int(float(axis_value))
        return PhaseConfig(f"{safe_id}_rep_event_cap_{event_cap}_seed{seed}", axis_name, str(event_cap), float(event_cap), 16, event_cap, seed, "replicate")
    if axis_name == "array_size":
        array_size = int(float(axis_value))
        return PhaseConfig(f"{safe_id}_rep_array_size_{array_size}_seed{seed}", axis_name, str(array_size), float(array_size), array_size, array_size * 4, seed + array_size, "replicate")
    if axis_name == "probability":
        probability = float(axis_value)
        label = format_probability(probability)
        return PhaseConfig(f"{safe_id}_rep_prob_{label.replace('.', 'p')}_seed{seed}", axis_name, label, probability, 16, 64, seed + 400, "replicate", probability_override=probability)
    if axis_name == "input_disorder":
        profile = str(axis_value)
        numeric = float(axis_numeric)
        return PhaseConfig(f"{safe_id}_rep_input_{profile}_seed{seed}", axis_name, profile, numeric, 16, 64, seed + 200 + int(numeric), "replicate", input_profile=profile)
    raise ValueError(f"Unsupported axis_name: {axis_name}")


def replication_configs_from_boundaries(boundaries: pd.DataFrame, *, max_boundaries: int = 48) -> dict[str, list[PhaseConfig]]:
    configs: dict[str, list[PhaseConfig]] = {}
    selected = boundaries.head(max_boundaries)
    for row in selected.to_dict(orient="records"):
        policy_id = str(row["base_policy_id"])
        configs.setdefault(policy_id, [])
        for value_key, numeric_key in (("lower_axis_value", "lower_axis_numeric"), ("upper_axis_value", "upper_axis_numeric")):
            for seed in REPLICATE_SEEDS:
                configs[policy_id].append(
                    _replicate_config(
                        policy_id,
                        str(row["axis_name"]),
                        str(row[value_key]),
                        float(row[numeric_key]),
                        int(seed),
                    )
                )
    # Deduplicate configs when adjacent boundaries share an endpoint.
    deduped: dict[str, list[PhaseConfig]] = {}
    for policy_id, items in configs.items():
        seen: set[str] = set()
        deduped[policy_id] = []
        for item in items:
            key = f"{item.axis_name}|{item.axis_value}|{item.seed}|{item.input_profile}|{item.probability_override}"
            if key not in seen:
                deduped[policy_id].append(item)
                seen.add(key)
    return deduped


def annotate_boundary_replication(boundaries: pd.DataFrame, all_rows: pd.DataFrame) -> pd.DataFrame:
    if boundaries.empty:
        return boundaries.copy()
    rows: list[dict[str, Any]] = []
    replicate = aggregate_axis(all_rows, split="replicate")
    for record in boundaries.to_dict(orient="records"):
        policy_id = str(record["base_policy_id"])
        axis_name = str(record["axis_name"])
        lower = str(record["lower_axis_value"])
        upper = str(record["upper_axis_value"])
        subset = replicate[(replicate["base_policy_id"] == policy_id) & (replicate["axis_name"] == axis_name)]
        lower_rep = subset[subset["axis_value"].astype(str) == lower]
        upper_rep = subset[subset["axis_value"].astype(str) == upper]
        out = dict(record)
        out["replicate_lower_success_fraction"] = float(lower_rep["success_fraction"].iloc[0]) if not lower_rep.empty else np.nan
        out["replicate_upper_success_fraction"] = float(upper_rep["success_fraction"].iloc[0]) if not upper_rep.empty else np.nan
        out["replicate_lower_mean_final_sortedness"] = float(lower_rep["mean_final_sortedness"].iloc[0]) if not lower_rep.empty else np.nan
        out["replicate_upper_mean_final_sortedness"] = float(upper_rep["mean_final_sortedness"].iloc[0]) if not upper_rep.empty else np.nan
        primary_direction = math.copysign(1.0, float(record["upper_mean_final_sortedness"] - record["lower_mean_final_sortedness"])) if float(record["upper_mean_final_sortedness"] - record["lower_mean_final_sortedness"]) != 0 else 0.0
        replicate_delta = out["replicate_upper_mean_final_sortedness"] - out["replicate_lower_mean_final_sortedness"]
        replicate_direction = math.copysign(1.0, replicate_delta) if not pd.isna(replicate_delta) and replicate_delta != 0 else 0.0
        out["replication_direction_match"] = bool(primary_direction == replicate_direction) if replicate_direction != 0 else False
        out["replication_available"] = bool(not lower_rep.empty and not upper_rep.empty)
        rows.append(out)
    return pd.DataFrame(rows)


def result_digest(frame: pd.DataFrame) -> str:
    cols = [
        "base_policy_id",
        "variant_policy_id",
        "config_id",
        "split",
        "axis_name",
        "axis_value",
        "seed",
        "failure_mode",
        "final_inversion_count",
        "compare_count",
        "swap_count",
        "update_count",
        "wait_count",
    ]
    records = frame[cols].sort_values(["base_policy_id", "config_id", "split"]).to_dict(orient="records")
    return stable_digest(records, cols)


def validation_frame(
    *,
    run_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    selected_policy_count: int,
    figure_exists: bool,
    unit_success: bool,
) -> pd.DataFrame:
    cases: list[dict[str, Any]] = []

    def add(name: str, success: bool, expected: str, observed: Any, notes: str) -> None:
        cases.append(
            {
                "validation_case": name,
                "success": bool(success),
                "expected": str(expected),
                "observed": str(observed),
                "notes": notes,
            }
        )

    add("selected_policies_present", selected_policy_count > 0, "> 0 selected policies", selected_policy_count, "S09 uses S07/S08 candidates.")
    add("primary_rows_present", int((run_df["split"] == "primary").sum()) > 0, "primary sweep rows > 0", int((run_df["split"] == "primary").sum()), "Primary phase sweep ran.")
    add("replicate_rows_present", int((run_df["split"] == "replicate").sum()) > 0, "replicate rows > 0", int((run_df["split"] == "replicate").sum()), "Boundary points replicated on independent seeds.")
    add("required_axes_present", {"event_cap", "array_size", "input_disorder"}.issubset(set(run_df["axis_name"])), "event_cap, array_size, input_disorder", sorted(run_df["axis_name"].unique()), "Environment phase axes are present.")
    add("probability_axis_if_available", "probability" in set(run_df["axis_name"]), "probability axis present for stochastic candidates", sorted(run_df["axis_name"].unique()), "S08 stochastic policies support a DSL parameter axis.")
    add("no_invalid_runs", int((run_df["execution_status"] != "ok").sum()) == 0, "0 invalid runs", int((run_df["execution_status"] != "ok").sum()), "All S09 rows should execute under the CPU DSL interpreter.")
    add("boundary_candidates_present", len(boundary_df) > 0, "> 0 boundary candidates", len(boundary_df), "Transitions or strongest gradients were documented.")
    add("replicated_boundaries_available", bool(boundary_df["replication_available"].any()) if not boundary_df.empty else False, "at least one boundary row has replicate endpoints", int(boundary_df["replication_available"].sum()) if not boundary_df.empty else 0, "Independent seed endpoints exist for the capped replicated boundary subset.")
    add("some_replication_direction_match", bool(boundary_df["replication_direction_match"].any()) if not boundary_df.empty else False, "at least one replicated direction matches", int(boundary_df["replication_direction_match"].sum()) if not boundary_df.empty else 0, "At least one primary transition direction reproduces on independent seeds.")
    add("failure_modes_classified", bool(run_df["failure_mode"].notna().all()), "all rows have failure_mode", int(run_df["failure_mode"].notna().sum()), "Every row is classified as sorted, stalled, partial, degraded, or oscillatory.")
    add("oscillation_metric_present", "oscillation_score" in run_df.columns, "oscillation score column present", "oscillation_score" in run_df.columns, "Trajectory repeat diagnostics were recorded.")
    add("figure_written", figure_exists, "phase-boundary figure exists and is non-empty", figure_exists, "S09 figure artifact.")
    add("unit_tests_passed", unit_success, "E03 unit tests pass", unit_success, "Repository validation before S09 run.")
    return pd.DataFrame(cases)
