"""Targeted rule-component ablations for E03 S12.

The functions in this module keep causal language deliberately local: they
verify that each source-code transform touches only its declared DSL component,
then compare original and ablated policies under matched S07 screen configs.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.coarse_sweep import (
    PolicyRecord,
    SweepConfig,
    _base_row,
    _finalize_row,
    actor_schedule,
    initial_values,
)
from src.e03.policy_generation import semantic_hash
from src.e03.rule_dsl import (
    DSLAction,
    DSLArrayState,
    DSLCondition,
    DSLInterpreter,
    DSLPolicy,
    DSLRule,
    parse_action,
    stable_json,
    validate_policy,
)


ABLATION_SCHEMA = "eidosoma.e03.rule_ablation.v1"
DEFAULT_ABLATION_SEED = 2026070112

STATE_ACTIONS = {"set_ideal", "estimate_target_position"}
MEMORY_SIGNAL_ACTIONS = {"remember", "signal"}
TARGET_ARGS = {"left", "right", "ideal"}


@dataclass(frozen=True)
class AblationSpec:
    """One declared S12 ablation transform."""

    ablation_id: str
    component: str
    description: str


@dataclass(frozen=True)
class AblationVariant:
    """One ablated policy plus source-change verification."""

    original_policy_id: str
    original_policy_name: str
    class_id: str
    cautious_label: str
    exemplar_roles: str
    selection_reason: str
    code_source: str
    ablation_id: str
    ablation_component: str
    ablation_description: str
    policy: DSLPolicy
    variant_policy_id: str
    variant_policy_name: str
    variant_kind: str
    source_verification: Mapping[str, Any]


def _histogram(items: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def _safe_name(text: str, *, fallback: str = "policy") -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    if not text:
        text = fallback
    if not re.match(r"^[A-Za-z_]", text):
        text = f"p_{text}"
    return text[:96]


def policy_with_name(policy: DSLPolicy, name: str) -> DSLPolicy:
    renamed = DSLPolicy(
        name=_safe_name(name),
        version=policy.version,
        state_init=policy.state_init,
        rules=policy.rules,
        metadata=dict(policy.metadata),
    )
    validate_policy(renamed)
    return renamed


def _iter_action_tree(action: DSLAction) -> Iterable[DSLAction]:
    yield action
    if action.name == "choose":
        for nested_source in action.args[1:]:
            nested = parse_action(nested_source)
            yield from _iter_action_tree(nested)


def iter_policy_actions(policy: DSLPolicy, *, include_nested_choose: bool = True) -> list[DSLAction]:
    actions: list[DSLAction] = []
    for rule in policy.rules:
        for action in rule.actions:
            if include_nested_choose:
                actions.extend(_iter_action_tree(action))
            else:
                actions.append(action)
    return actions


def component_profile(policy: DSLPolicy) -> dict[str, Any]:
    """Return compact source-code component counts for ablation verification."""

    conditions = [condition for rule in policy.rules for condition in rule.conditions]
    actions_top = iter_policy_actions(policy, include_nested_choose=False)
    actions_all = iter_policy_actions(policy, include_nested_choose=True)
    target_refs: list[str] = []
    for item in [*conditions, *actions_all]:
        target_refs.extend(arg for arg in item.args if arg in TARGET_ARGS)
    condition_hist = _histogram(condition.name for condition in conditions)
    action_hist = _histogram(action.name for action in actions_all)
    return {
        "state_init": policy.state_init,
        "rule_count": len(policy.rules),
        "condition_count": len(conditions),
        "action_count": len(actions_all),
        "top_level_action_count": len(actions_top),
        "condition_histogram": condition_hist,
        "action_histogram": action_hist,
        "target_histogram": _histogram(target_refs),
        "random_guard_count": condition_hist.get("random_lt", 0),
        "prefix_guard_count": condition_hist.get("prefix_sorted", 0),
        "probabilistic_action_count": action_hist.get("choose", 0),
        "state_action_count": sum(action_hist.get(name, 0) for name in STATE_ACTIONS),
        "memory_signal_action_count": sum(action_hist.get(name, 0) for name in MEMORY_SIGNAL_ACTIONS),
        "compare_action_count": action_hist.get("compare", 0),
        "swap_action_count": action_hist.get("swap", 0),
        "wait_action_count": action_hist.get("wait", 0),
        "ideal_target_reference_count": _histogram(target_refs).get("ideal", 0),
        "source_sha256": policy.sha256,
    }


def flattened_component_counts(policy: DSLPolicy) -> dict[str, Any]:
    """Flatten profile counts into a deterministic diff-friendly mapping."""

    profile = component_profile(policy)
    flat: dict[str, Any] = {
        "state_init": profile["state_init"],
        "rule_count": profile["rule_count"],
        "condition_count": profile["condition_count"],
        "action_count": profile["action_count"],
        "top_level_action_count": profile["top_level_action_count"],
    }
    for key, value in profile["condition_histogram"].items():
        flat[f"condition:{key}"] = int(value)
    for key, value in profile["action_histogram"].items():
        flat[f"action:{key}"] = int(value)
    for key, value in profile["target_histogram"].items():
        flat[f"target:{key}"] = int(value)
    return flat


def component_delta(original: DSLPolicy, variant: DSLPolicy) -> dict[str, tuple[Any, Any]]:
    left = flattened_component_counts(original)
    right = flattened_component_counts(variant)
    keys = sorted(set(left) | set(right))
    return {key: (left.get(key, 0), right.get(key, 0)) for key in keys if left.get(key, 0) != right.get(key, 0)}


def _drop_conditions(policy: DSLPolicy, predicate: Callable[[DSLCondition], bool]) -> DSLPolicy:
    rules: list[DSLRule] = []
    for rule in policy.rules:
        if rule.is_else:
            rules.append(rule)
            continue
        kept = tuple(condition for condition in rule.conditions if not predicate(condition))
        if not kept:
            kept = (DSLCondition("always"),)
        rules.append(DSLRule(conditions=kept, actions=rule.actions, is_else=False))
    out = DSLPolicy(name=policy.name, version=policy.version, state_init=policy.state_init, rules=tuple(rules), metadata=dict(policy.metadata))
    validate_policy(out)
    return out


def _transform_action(action: DSLAction, predicate: Callable[[DSLAction], bool]) -> DSLAction | None:
    if action.name == "choose" and not predicate(action):
        true_action = _transform_action(parse_action(action.args[1]), predicate) or DSLAction("wait")
        false_action = _transform_action(parse_action(action.args[2]), predicate) or DSLAction("wait")
        return DSLAction("choose", (action.args[0], true_action.to_source(), false_action.to_source()))
    if predicate(action):
        return None
    return action


def _drop_actions(policy: DSLPolicy, predicate: Callable[[DSLAction], bool], *, state_init: str | None = None) -> DSLPolicy:
    rules: list[DSLRule] = []
    for rule in policy.rules:
        actions = tuple(action for action in (_transform_action(action, predicate) for action in rule.actions) if action is not None)
        if not actions:
            actions = (DSLAction("wait"),)
        rules.append(DSLRule(conditions=rule.conditions, actions=actions, is_else=rule.is_else))
    out = DSLPolicy(
        name=policy.name,
        version=policy.version,
        state_init=policy.state_init if state_init is None else state_init,
        rules=tuple(rules),
        metadata=dict(policy.metadata),
    )
    validate_policy(out)
    return out


def _determinize_choose_first(policy: DSLPolicy) -> DSLPolicy:
    rules: list[DSLRule] = []
    for rule in policy.rules:
        actions: list[DSLAction] = []
        for action in rule.actions:
            if action.name == "choose":
                actions.append(parse_action(action.args[1]))
            else:
                actions.append(action)
        rules.append(DSLRule(conditions=rule.conditions, actions=tuple(actions), is_else=rule.is_else))
    out = DSLPolicy(name=policy.name, version=policy.version, state_init=policy.state_init, rules=tuple(rules), metadata=dict(policy.metadata))
    validate_policy(out)
    return out


ABLATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec("drop_random_guards", "stochastic_condition", "Remove `random_lt` rule guards and replace empty guards with `always`."),
    AblationSpec("determinize_choose_first", "probabilistic_action", "Replace `choose(p, a, b)` actions with their first branch."),
    AblationSpec("drop_prefix_sorted_guards", "prefix_sorted_guard", "Remove `prefix_sorted` rule guards and replace empty guards with `always`."),
    AblationSpec("disable_target_position_state", "target_position_state", "Set initial ideal-position state to `none` and remove target-position update actions."),
    AblationSpec("drop_memory_signal_actions", "memory_signal_stub", "Remove `remember` and `signal` actions, preserving other actions."),
    AblationSpec("drop_compare_actions", "neighbor_compare", "Remove explicit `compare` actions while preserving swap/wait/state actions."),
    AblationSpec("drop_swap_actions", "neighbor_exchange", "Remove `swap` actions while preserving comparison and state actions."),
)


def applicable_ablation_specs(policy: DSLPolicy) -> tuple[AblationSpec, ...]:
    profile = component_profile(policy)
    checks = {
        "drop_random_guards": profile["random_guard_count"] > 0,
        "determinize_choose_first": profile["probabilistic_action_count"] > 0,
        "drop_prefix_sorted_guards": profile["prefix_guard_count"] > 0,
        "disable_target_position_state": profile["state_init"] != "none" or profile["state_action_count"] > 0,
        "drop_memory_signal_actions": profile["memory_signal_action_count"] > 0,
        "drop_compare_actions": profile["compare_action_count"] > 0,
        "drop_swap_actions": profile["swap_action_count"] > 0,
    }
    return tuple(spec for spec in ABLATION_SPECS if checks[spec.ablation_id])


def apply_ablation(policy: DSLPolicy, ablation_id: str) -> DSLPolicy:
    if ablation_id == "drop_random_guards":
        out = _drop_conditions(policy, lambda condition: condition.name == "random_lt")
    elif ablation_id == "determinize_choose_first":
        out = _determinize_choose_first(policy)
    elif ablation_id == "drop_prefix_sorted_guards":
        out = _drop_conditions(policy, lambda condition: condition.name == "prefix_sorted")
    elif ablation_id == "disable_target_position_state":
        out = _drop_actions(policy, lambda action: action.name in STATE_ACTIONS, state_init="none")
    elif ablation_id == "drop_memory_signal_actions":
        out = _drop_actions(policy, lambda action: action.name in MEMORY_SIGNAL_ACTIONS)
    elif ablation_id == "drop_compare_actions":
        out = _drop_actions(policy, lambda action: action.name == "compare")
    elif ablation_id == "drop_swap_actions":
        out = _drop_actions(policy, lambda action: action.name == "swap")
    else:
        raise ValueError(f"Unknown ablation_id: {ablation_id}")
    renamed = policy_with_name(out, f"s12_{policy.sha256[:8]}_{ablation_id}")
    return renamed


def _changed_keys_with_prefix(delta: Mapping[str, tuple[Any, Any]], prefixes: Sequence[str]) -> set[str]:
    return {key for key in delta if any(key == prefix or key.startswith(prefix) for prefix in prefixes)}


def verify_ablation_change(original: DSLPolicy, variant: DSLPolicy, ablation_id: str) -> dict[str, Any]:
    """Check that a transform changed only its declared component family."""

    delta = component_delta(original, variant)
    before = component_profile(original)
    after = component_profile(variant)
    success = bool(delta)
    notes: list[str] = []

    condition_delta = _changed_keys_with_prefix(delta, ("condition:", "condition_count"))
    action_delta = _changed_keys_with_prefix(delta, ("action:", "action_count", "top_level_action_count"))
    target_delta = _changed_keys_with_prefix(delta, ("target:",))
    state_delta = {"state_init"} if "state_init" in delta else set()
    rule_delta = {"rule_count"} if "rule_count" in delta else set()

    if ablation_id == "drop_random_guards":
        success = success and after["random_guard_count"] == 0 and before["random_guard_count"] > 0
        success = success and not action_delta and not state_delta and not rule_delta and not target_delta
        unintended = sorted(set(delta) - condition_delta)
    elif ablation_id == "drop_prefix_sorted_guards":
        success = success and after["prefix_guard_count"] == 0 and before["prefix_guard_count"] > 0
        success = success and not action_delta and not state_delta and not rule_delta and not target_delta
        unintended = sorted(set(delta) - condition_delta)
    elif ablation_id == "determinize_choose_first":
        success = success and after["probabilistic_action_count"] == 0 and before["probabilistic_action_count"] > 0
        success = success and not condition_delta and not state_delta and not rule_delta
        unintended = sorted(set(delta) - action_delta - target_delta)
        notes.append("Target counts may change because a concrete choose branch becomes top-level code.")
    elif ablation_id == "disable_target_position_state":
        success = success and after["state_init"] == "none" and after["state_action_count"] == 0
        success = success and before["state_init"] != "none" or before["state_action_count"] > 0
        success = bool(success and not condition_delta and not rule_delta)
        allowed = state_delta | action_delta | target_delta
        unintended = sorted(set(delta) - allowed)
    elif ablation_id == "drop_memory_signal_actions":
        success = success and after["memory_signal_action_count"] == 0 and before["memory_signal_action_count"] > 0
        success = success and not condition_delta and not state_delta and not rule_delta
        allowed = action_delta | target_delta
        unintended = sorted(set(delta) - allowed)
    elif ablation_id == "drop_compare_actions":
        success = success and after["compare_action_count"] == 0 and before["compare_action_count"] > 0
        success = success and not condition_delta and not state_delta and not rule_delta
        allowed = action_delta | target_delta
        unintended = sorted(set(delta) - allowed)
    elif ablation_id == "drop_swap_actions":
        success = success and after["swap_action_count"] == 0 and before["swap_action_count"] > 0
        success = success and not condition_delta and not state_delta and not rule_delta
        allowed = action_delta | target_delta
        unintended = sorted(set(delta) - allowed)
    else:
        success = False
        unintended = sorted(delta)
        notes.append("Unknown ablation type.")

    if unintended:
        success = False
        notes.append(f"Unintended changed keys: {unintended}")
    return {
        "ablation_id": ablation_id,
        "success": bool(success),
        "changed_key_count": len(delta),
        "changed_keys_json": stable_json(sorted(delta)),
        "component_delta_json": stable_json({key: list(value) for key, value in delta.items()}),
        "before_profile_json": stable_json(before),
        "after_profile_json": stable_json(after),
        "notes": " ".join(notes),
    }


def build_ablation_variants(
    *,
    policy: DSLPolicy,
    original_policy_id: str,
    original_policy_name: str,
    class_id: str,
    cautious_label: str,
    exemplar_roles: str,
    selection_reason: str,
    code_source: str,
) -> list[AblationVariant]:
    variants: list[AblationVariant] = []
    for spec in applicable_ablation_specs(policy):
        variant = apply_ablation(policy, spec.ablation_id)
        verification = verify_ablation_change(policy, variant, spec.ablation_id)
        variants.append(
            AblationVariant(
                original_policy_id=original_policy_id,
                original_policy_name=original_policy_name,
                class_id=class_id,
                cautious_label=cautious_label,
                exemplar_roles=exemplar_roles,
                selection_reason=selection_reason,
                code_source=code_source,
                ablation_id=spec.ablation_id,
                ablation_component=spec.component,
                ablation_description=spec.description,
                policy=variant,
                variant_policy_id=variant.policy_id,
                variant_policy_name=variant.name,
                variant_kind="ablation",
                source_verification=verification,
            )
        )
    return variants


def policy_record_for_variant(
    policy: DSLPolicy,
    *,
    policy_id: str | None = None,
    policy_name: str | None = None,
    source_kind: str = "s12_ablation",
) -> PolicyRecord:
    return PolicyRecord(
        policy=policy,
        policy_id=policy_id or policy.policy_id,
        policy_name=policy_name or policy.name,
        source_kind=source_kind,
        semantic_hash=semantic_hash(policy),
        dsl_sha256=policy.sha256,
        features=component_profile(policy),
        compiles_for_batch=False,
        requires_cpu_fallback=True,
        fallback_reasons=("s12_matched_cpu_reference",),
        route="cpu_fallback",
    )


def matched_rng_seed(config_seed: int, original_policy_id: str, seed: int = DEFAULT_ABLATION_SEED) -> int:
    digest = hashlib.sha256(f"{seed}|{config_seed}|{original_policy_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def run_cpu_policy_config_matched(
    record: PolicyRecord,
    config: SweepConfig,
    *,
    original_policy_id: str,
    variant_kind: str,
    ablation_id: str,
    ablation_component: str,
    seed: int = DEFAULT_ABLATION_SEED,
) -> dict[str, Any]:
    """Run a DSL policy with stochastic draws matched by original policy ID."""

    values = initial_values(config.array_size, config.seed, config.input_profile)
    schedule = actor_schedule(config.array_size, config.event_cap, config.seed)
    row = _base_row(record, config, values)
    row.update(
        {
            "schema": ABLATION_SCHEMA,
            "research_step_id": "S12",
            "original_policy_id": original_policy_id,
            "variant_policy_id": record.policy_id,
            "variant_kind": variant_kind,
            "ablation_id": ablation_id,
            "ablation_component": ablation_component,
            "matched_rng_seed": int(matched_rng_seed(config.seed, original_policy_id, seed)),
        }
    )
    started = time.perf_counter()
    try:
        interpreter = DSLInterpreter(record.policy)
        rng = random.Random(int(row["matched_rng_seed"]))
        statuses: tuple[str, ...] = tuple("ACTIVE" for _ in values)
        ideal_position: int | None = None
        compare_count = 0
        swap_count = 0
        update_count = 0
        wait_count = 0
        current_values = values
        for actor_index in schedule:
            state = DSLArrayState(
                values=current_values,
                statuses=statuses,
                actor_index=int(actor_index),
                ideal_position=ideal_position,
            )
            result = interpreter.step_state(state, rng)
            current_values = result.state_after.values
            statuses = tuple(result.state_after.statuses or statuses)
            ideal_position = result.state_after.ideal_position
            compare_count += int(bool(result.action.compare_counted))
            swap_count += int(result.action.action_type == "swap")
            update_count += int(result.action.action_type == "update_state")
            wait_count += int(result.action.action_type == "wait")
        elapsed = time.perf_counter() - started
        return _finalize_row(
            row,
            final_values=current_values,
            compare_count=compare_count,
            swap_count=swap_count,
            update_count=update_count,
            wait_count=wait_count,
            elapsed_seconds=elapsed,
            backend="cpu_matched",
            device="cpu",
        )
    except Exception as exc:  # pragma: no cover - captured as validation evidence.
        elapsed = time.perf_counter() - started
        return _finalize_row(
            row,
            final_values=values,
            compare_count=0,
            swap_count=0,
            update_count=0,
            wait_count=0,
            elapsed_seconds=elapsed,
            execution_status="invalid",
            error_message=repr(exc),
            backend="cpu_matched",
            device="cpu",
        )


def variant_metadata_frame(variants: Sequence[AblationVariant]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        rows.append(
            {
                "schema": ABLATION_SCHEMA,
                "experiment_id": "E03",
                "research_step_id": "S12",
                "original_policy_id": variant.original_policy_id,
                "original_policy_name": variant.original_policy_name,
                "variant_policy_id": variant.variant_policy_id,
                "variant_policy_name": variant.variant_policy_name,
                "variant_kind": variant.variant_kind,
                "class_id": variant.class_id,
                "cautious_label": variant.cautious_label,
                "exemplar_roles": variant.exemplar_roles,
                "selection_reason": variant.selection_reason,
                "code_source": variant.code_source,
                "ablation_id": variant.ablation_id,
                "ablation_component": variant.ablation_component,
                "ablation_description": variant.ablation_description,
                "source_verification_success": bool(variant.source_verification["success"]),
                "source_verification_notes": str(variant.source_verification.get("notes", "")),
                "changed_key_count": int(variant.source_verification["changed_key_count"]),
                "changed_keys_json": str(variant.source_verification["changed_keys_json"]),
                "component_delta_json": str(variant.source_verification["component_delta_json"]),
                "before_profile_json": str(variant.source_verification["before_profile_json"]),
                "after_profile_json": str(variant.source_verification["after_profile_json"]),
                "dsl_sha256": variant.policy.sha256,
                "dsl_source": variant.policy.to_source(),
            }
        )
    return pd.DataFrame(rows)


def aggregate_ablation_pairs(run_df: pd.DataFrame, variant_meta: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched original-versus-ablation run rows."""

    baseline = run_df[run_df["variant_kind"] == "original"].copy()
    ablated = run_df[run_df["variant_kind"] == "ablation"].copy()
    base_cols = [
        "original_policy_id",
        "config_id",
        "seed",
        "array_size",
        "split",
        "heldout",
        "matched_rng_seed",
        "final_inversion_sortedness",
        "inversion_sortedness_delta",
        "final_is_sorted",
        "work_count",
        "compare_count",
        "swap_count",
        "update_count",
        "wait_count",
        "invalid",
        "timed_out",
    ]
    merged = ablated.merge(
        baseline[base_cols],
        on=["original_policy_id", "config_id", "seed", "array_size", "split", "heldout", "matched_rng_seed"],
        how="left",
        suffixes=("", "_original"),
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    group_cols = [
        "original_policy_id",
        "variant_policy_id",
        "ablation_id",
        "ablation_component",
    ]
    for keys, group in merged.groupby(group_cols, sort=False):
        original_policy_id, variant_policy_id, ablation_id, ablation_component = keys
        heldout = group[group["heldout"] == True]  # noqa: E712
        basis = heldout if not heldout.empty else group
        delta_final = group["final_inversion_sortedness"] - group["final_inversion_sortedness_original"]
        delta_improve = group["inversion_sortedness_delta"] - group["inversion_sortedness_delta_original"]
        delta_work = group["work_count"] - group["work_count_original"]
        row = {
            "schema": ABLATION_SCHEMA,
            "experiment_id": "E03",
            "research_step_id": "S12",
            "original_policy_id": str(original_policy_id),
            "variant_policy_id": str(variant_policy_id),
            "ablation_id": str(ablation_id),
            "ablation_component": str(ablation_component),
            "run_count": int(len(group)),
            "heldout_run_count": int(len(heldout)),
            "matched_config_count": int(group["config_id"].nunique()),
            "invalid_run_count": int(group["invalid"].sum()),
            "timeout_run_count": int(group["timed_out"].sum()),
            "original_invalid_run_count": int(group["invalid_original"].sum()),
            "original_timeout_run_count": int(group["timed_out_original"].sum()),
            "original_mean_final_sortedness": float(group["final_inversion_sortedness_original"].mean()),
            "ablated_mean_final_sortedness": float(group["final_inversion_sortedness"].mean()),
            "delta_final_sortedness": float(delta_final.mean()),
            "heldout_original_mean_final_sortedness": float(basis["final_inversion_sortedness_original"].mean()),
            "heldout_ablated_mean_final_sortedness": float(basis["final_inversion_sortedness"].mean()),
            "heldout_delta_final_sortedness": float((basis["final_inversion_sortedness"] - basis["final_inversion_sortedness_original"]).mean()),
            "original_mean_improvement": float(group["inversion_sortedness_delta_original"].mean()),
            "ablated_mean_improvement": float(group["inversion_sortedness_delta"].mean()),
            "delta_improvement": float(delta_improve.mean()),
            "original_sorted_run_fraction": float(group["final_is_sorted_original"].mean()),
            "ablated_sorted_run_fraction": float(group["final_is_sorted"].mean()),
            "delta_sorted_run_fraction": float(group["final_is_sorted"].mean() - group["final_is_sorted_original"].mean()),
            "original_work_mean": float(group["work_count_original"].mean()),
            "ablated_work_mean": float(group["work_count"].mean()),
            "delta_work_mean": float(delta_work.mean()),
            "delta_compare_mean": float((group["compare_count"] - group["compare_count_original"]).mean()),
            "delta_swap_mean": float((group["swap_count"] - group["swap_count_original"]).mean()),
            "delta_update_mean": float((group["update_count"] - group["update_count_original"]).mean()),
            "delta_wait_mean": float((group["wait_count"] - group["wait_count_original"]).mean()),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    meta_cols = [
        "original_policy_id",
        "variant_policy_id",
        "original_policy_name",
        "variant_policy_name",
        "class_id",
        "cautious_label",
        "exemplar_roles",
        "selection_reason",
        "ablation_description",
        "source_verification_success",
        "source_verification_notes",
        "changed_key_count",
    ]
    return out.merge(
        variant_meta[[column for column in meta_cols if column in variant_meta.columns]],
        on=["original_policy_id", "variant_policy_id"],
        how="left",
        validate="one_to_one",
    )


def classify_claim(row: Mapping[str, Any]) -> dict[str, str]:
    """Assign cautious local causal-claim wording from matched ablation effect."""

    original = float(row.get("heldout_original_mean_final_sortedness", np.nan))
    ablated = float(row.get("heldout_ablated_mean_final_sortedness", np.nan))
    delta = float(row.get("heldout_delta_final_sortedness", np.nan))
    invalid = int(row.get("invalid_run_count", 0)) + int(row.get("original_invalid_run_count", 0))
    component = str(row.get("ablation_component", "component"))
    if invalid:
        claim_type = "indeterminate_invalid"
        claim = f"{component} could not be assessed because at least one matched run was invalid."
    elif np.isnan(delta) or np.isnan(original):
        claim_type = "indeterminate_missing"
        claim = f"{component} could not be assessed because matched aggregate metrics were missing."
    elif original >= 0.75 and delta <= -0.10:
        claim_type = "local_necessary_candidate"
        claim = f"Removing {component} caused a large matched sortedness drop; this is a local necessary-feature candidate for this tested policy family."
    elif original >= 0.75 and ablated >= 0.75 and delta >= -0.03:
        claim_type = "residual_sufficient_candidate"
        claim = f"Competence remained high after removing {component}; the remaining components are a local sufficient-feature-set candidate under these tests."
    elif delta >= 0.10:
        claim_type = "anti_feature_candidate"
        claim = f"Removing {component} improved matched sortedness; this component is an anti-feature candidate in this local context."
    elif abs(delta) < 0.03:
        claim_type = "locally_neutral_candidate"
        claim = f"Removing {component} changed matched sortedness little; no local necessity signal was detected."
    elif delta <= -0.03:
        claim_type = "weak_local_necessity_candidate"
        claim = f"Removing {component} reduced matched sortedness modestly; this is a weak local necessity candidate."
    else:
        claim_type = "weak_local_improvement_candidate"
        claim = f"Removing {component} modestly improved matched sortedness; this is a weak anti-feature signal."
    return {"claim_type": claim_type, "claim_statement": claim}


def claim_frame(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return aggregate.copy()
    claims = pd.DataFrame([classify_claim(record) for record in aggregate.to_dict(orient="records")])
    out = pd.concat([aggregate.reset_index(drop=True), claims], axis=1)
    return out.sort_values(
        ["claim_type", "heldout_delta_final_sortedness", "class_id", "original_policy_id"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def component_summary_frame(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (class_id, component), group in claims.groupby(["class_id", "ablation_component"], sort=True):
        rows.append(
            {
                "schema": ABLATION_SCHEMA,
                "experiment_id": "E03",
                "research_step_id": "S12",
                "class_id": str(class_id),
                "ablation_component": str(component),
                "tested_policy_count": int(group["original_policy_id"].nunique()),
                "tested_variant_count": int(len(group)),
                "mean_heldout_delta_final_sortedness": float(group["heldout_delta_final_sortedness"].mean()),
                "median_heldout_delta_final_sortedness": float(group["heldout_delta_final_sortedness"].median()),
                "min_heldout_delta_final_sortedness": float(group["heldout_delta_final_sortedness"].min()),
                "max_heldout_delta_final_sortedness": float(group["heldout_delta_final_sortedness"].max()),
                "large_drop_count": int((group["heldout_delta_final_sortedness"] <= -0.10).sum()),
                "survived_high_count": int((group["claim_type"] == "residual_sufficient_candidate").sum()),
                "improved_count": int((group["heldout_delta_final_sortedness"] >= 0.10).sum()),
                "claim_types_json": stable_json(_histogram(str(value) for value in group["claim_type"])),
            }
        )
    return pd.DataFrame(rows)


def matched_seed_validation(run_df: pd.DataFrame) -> bool:
    """Return True if every ablated variant has the same config/seed set as its original."""

    baseline = run_df[run_df["variant_kind"] == "original"]
    ablated = run_df[run_df["variant_kind"] == "ablation"]
    base_sets = {
        policy_id: set(zip(group["config_id"], group["seed"], group["matched_rng_seed"], strict=False))
        for policy_id, group in baseline.groupby("original_policy_id")
    }
    for (_, variant_id), group in ablated.groupby(["original_policy_id", "variant_policy_id"]):
        policy_id = str(group["original_policy_id"].iloc[0])
        observed = set(zip(group["config_id"], group["seed"], group["matched_rng_seed"], strict=False))
        if observed != base_sets.get(policy_id, set()):
            return False
    return True


def ablation_result_digest(frame: pd.DataFrame) -> str:
    cols = [
        "original_policy_id",
        "variant_policy_id",
        "ablation_id",
        "matched_config_count",
        "heldout_delta_final_sortedness",
        "delta_work_mean",
        "claim_type",
    ]
    payload = frame[[column for column in cols if column in frame.columns]].sort_values(
        ["original_policy_id", "variant_policy_id"], kind="mergesort"
    )
    return hashlib.sha256(payload.to_json(orient="records", double_precision=12).encode("utf-8")).hexdigest()
