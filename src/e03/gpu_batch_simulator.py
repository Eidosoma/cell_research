"""GPU-friendly batched simulator for S02/S05 DSL policies.

The S06 simulator keeps :class:`src.e03.rule_dsl.DSLInterpreter` as the
reference and compiles DSL policies into fixed-width arrays that can be
evaluated across a JAX batch.  It intentionally models one local actor step at
a time; S07 can wrap it with scheduler logic for larger sweeps.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.rule_dsl import (
    DSLAction,
    DSLArrayState,
    DSLInterpreter,
    DSLPolicy,
    DSLValidationError,
    parse_action,
    parse_policy,
    stable_json,
)


TARGET_NONE = -1
TARGET_LEFT = 0
TARGET_RIGHT = 1
TARGET_IDEAL = 2
TARGET_CODES = {"left": TARGET_LEFT, "right": TARGET_RIGHT, "ideal": TARGET_IDEAL}

STATE_NONE = -1
STATE_LEFT_BOUNDARY = -2
STATE_RIGHT_BOUNDARY = -3
STATE_CURRENT = -4
STATE_NEXT = -5
STATE_CODES = {
    "none": STATE_NONE,
    "left_boundary": STATE_LEFT_BOUNDARY,
    "right_boundary": STATE_RIGHT_BOUNDARY,
    "current": STATE_CURRENT,
    "next": STATE_NEXT,
}

COND_NONE = 0
COND_ALWAYS = 1
COND_TARGET_EXISTS = 2
COND_TARGET_ACTIVE = 3
COND_TARGET_MOVABLE = 4
COND_SELF_LT = 5
COND_SELF_GT = 6
COND_SELF_LE = 7
COND_SELF_GE = 8
COND_AT_IDEAL = 9
COND_NOT_AT_IDEAL = 10
COND_PREFIX_SORTED = 11
COND_RANDOM_LT = 12
CONDITION_CODES = {
    "always": COND_ALWAYS,
    "target_exists": COND_TARGET_EXISTS,
    "target_active": COND_TARGET_ACTIVE,
    "target_movable": COND_TARGET_MOVABLE,
    "self_lt": COND_SELF_LT,
    "self_gt": COND_SELF_GT,
    "self_le": COND_SELF_LE,
    "self_ge": COND_SELF_GE,
    "at_ideal": COND_AT_IDEAL,
    "not_at_ideal": COND_NOT_AT_IDEAL,
    "prefix_sorted": COND_PREFIX_SORTED,
    "random_lt": COND_RANDOM_LT,
}

ACTION_NONE = 0
ACTION_COMPARE = 1
ACTION_SWAP = 2
ACTION_SET_IDEAL = 3
ACTION_ESTIMATE_TARGET_POSITION = 4
ACTION_WAIT = 5
ACTION_REMEMBER = 6
ACTION_SIGNAL = 7
ACTION_CHOOSE = 8
ACTION_CODES = {
    "compare": ACTION_COMPARE,
    "swap": ACTION_SWAP,
    "set_ideal": ACTION_SET_IDEAL,
    "estimate_target_position": ACTION_ESTIMATE_TARGET_POSITION,
    "wait": ACTION_WAIT,
    "remember": ACTION_REMEMBER,
    "signal": ACTION_SIGNAL,
    "choose": ACTION_CHOOSE,
}

STATUS_OTHER = 0
STATUS_ACTIVE = 1
STATUS_FREEZE = 2
STATUS_ACTIVE_NAMES = {"ACTIVE", "CellStatus.ACTIVE"}
STATUS_FREEZE_NAMES = {"FREEZE", "CellStatus.FREEZE"}

DEFAULT_MAX_RULES = 6
DEFAULT_MAX_CONDITIONS = 5
DEFAULT_MAX_ACTIONS = 3


@dataclass(frozen=True)
class CompiledPolicy:
    """Fixed-width numeric representation of one DSL policy."""

    policy_id: str
    policy_name: str
    dsl_sha256: str
    max_rules: int
    max_conditions: int
    max_actions: int
    rule_mask: np.ndarray
    rule_is_else: np.ndarray
    condition_code: np.ndarray
    condition_target: np.ndarray
    condition_probability: np.ndarray
    action_code: np.ndarray
    action_target: np.ndarray
    action_state_value: np.ndarray
    action_probability: np.ndarray
    choose_true_code: np.ndarray
    choose_true_target: np.ndarray
    choose_true_state_value: np.ndarray
    choose_false_code: np.ndarray
    choose_false_target: np.ndarray
    choose_false_state_value: np.ndarray
    state_init_code: int
    requires_cpu_fallback: bool
    fallback_reasons: tuple[str, ...]

    def compatibility_record(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "dsl_sha256": self.dsl_sha256,
            "compiles_for_batch": True,
            "requires_cpu_fallback": bool(self.requires_cpu_fallback),
            "fallback_reasons_json": json.dumps(list(self.fallback_reasons), separators=(",", ":")),
            "rule_count": int(self.rule_mask.sum()),
            "max_rules": int(self.max_rules),
            "max_conditions": int(self.max_conditions),
            "max_actions": int(self.max_actions),
        }


@dataclass(frozen=True)
class BatchSimulationResult:
    """Final state and simple metrics from a batched JAX run."""

    values: np.ndarray
    statuses: np.ndarray
    actor_index: np.ndarray
    ideal_position: np.ndarray
    compare_count: np.ndarray
    swap_count: np.ndarray
    update_count: np.ndarray
    wait_count: np.ndarray
    backend: str
    device: str


def _require_jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - exercised only without JAX
        raise RuntimeError("JAX is required for the S06 batched simulator") from exc
    return jax, jnp


def jax_backend_summary() -> dict[str, Any]:
    """Return compact JAX backend metadata for reports and manifests."""

    try:
        jax, _jnp = _require_jax()
        devices = jax.devices()
        return {
            "available": True,
            "version": getattr(jax, "__version__", "unknown"),
            "defaultBackend": jax.default_backend(),
            "devices": [str(device) for device in devices],
            "gpuDeviceCount": len(jax.devices("gpu")) if devices else 0,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "available": False,
            "version": None,
            "defaultBackend": None,
            "devices": [],
            "gpuDeviceCount": 0,
            "error": repr(exc),
        }


def status_to_code(status: str) -> int:
    if str(status) in STATUS_ACTIVE_NAMES:
        return STATUS_ACTIVE
    if str(status) in STATUS_FREEZE_NAMES:
        return STATUS_FREEZE
    return STATUS_OTHER


def status_from_code(code: int) -> str:
    if int(code) == STATUS_ACTIVE:
        return "ACTIVE"
    if int(code) == STATUS_FREEZE:
        return "FREEZE"
    return "OTHER"


def _state_code(value: str) -> int:
    if value in STATE_CODES:
        return STATE_CODES[value]
    return int(value)


def _target_code(value: str | None) -> int:
    if value is None:
        return TARGET_NONE
    return TARGET_CODES[value]


def _compile_nested_action(action: DSLAction) -> tuple[int, int, int]:
    if action.name == "choose":
        raise DSLValidationError("Nested choose actions are not supported by the S06 batch compiler")
    code = ACTION_CODES[action.name]
    target = TARGET_NONE
    state_value = STATE_NONE
    if action.name in {"compare", "swap"}:
        target = _target_code(action.args[0])
    elif action.name in {"set_ideal", "estimate_target_position"}:
        state_value = _state_code(action.args[0])
    return code, target, state_value


def _is_stochastic_probability(probability: float) -> bool:
    return 0.0 < float(probability) < 1.0


def compile_policy(
    policy: DSLPolicy,
    *,
    max_rules: int = DEFAULT_MAX_RULES,
    max_conditions: int = DEFAULT_MAX_CONDITIONS,
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> CompiledPolicy:
    """Compile one DSL policy into padded numeric arrays."""

    if len(policy.rules) > max_rules:
        raise DSLValidationError(f"Policy has {len(policy.rules)} rules; max_rules is {max_rules}")

    rule_mask = np.zeros(max_rules, dtype=np.bool_)
    rule_is_else = np.zeros(max_rules, dtype=np.bool_)
    condition_code = np.zeros((max_rules, max_conditions), dtype=np.int32)
    condition_target = np.full((max_rules, max_conditions), TARGET_NONE, dtype=np.int32)
    condition_probability = np.zeros((max_rules, max_conditions), dtype=np.float32)
    action_code = np.zeros((max_rules, max_actions), dtype=np.int32)
    action_target = np.full((max_rules, max_actions), TARGET_NONE, dtype=np.int32)
    action_state_value = np.full((max_rules, max_actions), STATE_NONE, dtype=np.int32)
    action_probability = np.zeros((max_rules, max_actions), dtype=np.float32)
    choose_true_code = np.zeros((max_rules, max_actions), dtype=np.int32)
    choose_true_target = np.full((max_rules, max_actions), TARGET_NONE, dtype=np.int32)
    choose_true_state_value = np.full((max_rules, max_actions), STATE_NONE, dtype=np.int32)
    choose_false_code = np.zeros((max_rules, max_actions), dtype=np.int32)
    choose_false_target = np.full((max_rules, max_actions), TARGET_NONE, dtype=np.int32)
    choose_false_state_value = np.full((max_rules, max_actions), STATE_NONE, dtype=np.int32)

    fallback_reasons: list[str] = []
    for rule_index, rule in enumerate(policy.rules):
        if len(rule.conditions) > max_conditions:
            raise DSLValidationError(
                f"Rule {rule_index} has {len(rule.conditions)} conditions; max_conditions is {max_conditions}"
            )
        if len(rule.actions) > max_actions:
            raise DSLValidationError(f"Rule {rule_index} has {len(rule.actions)} actions; max_actions is {max_actions}")
        rule_mask[rule_index] = True
        rule_is_else[rule_index] = bool(rule.is_else)
        for condition_index, condition in enumerate(rule.conditions):
            condition_code[rule_index, condition_index] = CONDITION_CODES[condition.name]
            if condition.name in {
                "target_exists",
                "target_active",
                "target_movable",
                "self_lt",
                "self_gt",
                "self_le",
                "self_ge",
            }:
                condition_target[rule_index, condition_index] = _target_code(condition.args[0])
            elif condition.name == "random_lt":
                probability = float(condition.args[0])
                condition_probability[rule_index, condition_index] = probability
                if _is_stochastic_probability(probability):
                    fallback_reasons.append("stochastic random_lt stream requires CPU reference for exact replay")
        for action_index, action in enumerate(rule.actions):
            action_code[rule_index, action_index] = ACTION_CODES[action.name]
            if action.name in {"compare", "swap"}:
                action_target[rule_index, action_index] = _target_code(action.args[0])
            elif action.name in {"set_ideal", "estimate_target_position"}:
                action_state_value[rule_index, action_index] = _state_code(action.args[0])
            elif action.name in {"remember", "signal"}:
                fallback_reasons.append(f"{action.name} side effects are metadata/stub-only in the batch kernel")
            elif action.name == "choose":
                probability = float(action.args[0])
                action_probability[rule_index, action_index] = probability
                if _is_stochastic_probability(probability):
                    fallback_reasons.append("stochastic choose stream requires CPU reference for exact replay")
                true_code, true_target, true_state = _compile_nested_action(parse_action(action.args[1]))
                false_code, false_target, false_state = _compile_nested_action(parse_action(action.args[2]))
                choose_true_code[rule_index, action_index] = true_code
                choose_true_target[rule_index, action_index] = true_target
                choose_true_state_value[rule_index, action_index] = true_state
                choose_false_code[rule_index, action_index] = false_code
                choose_false_target[rule_index, action_index] = false_target
                choose_false_state_value[rule_index, action_index] = false_state

    return CompiledPolicy(
        policy_id=policy.policy_id,
        policy_name=policy.name,
        dsl_sha256=policy.sha256,
        max_rules=max_rules,
        max_conditions=max_conditions,
        max_actions=max_actions,
        rule_mask=rule_mask,
        rule_is_else=rule_is_else,
        condition_code=condition_code,
        condition_target=condition_target,
        condition_probability=condition_probability,
        action_code=action_code,
        action_target=action_target,
        action_state_value=action_state_value,
        action_probability=action_probability,
        choose_true_code=choose_true_code,
        choose_true_target=choose_true_target,
        choose_true_state_value=choose_true_state_value,
        choose_false_code=choose_false_code,
        choose_false_target=choose_false_target,
        choose_false_state_value=choose_false_state_value,
        state_init_code=_state_code(policy.state_init),
        requires_cpu_fallback=bool(fallback_reasons),
        fallback_reasons=tuple(sorted(set(fallback_reasons))),
    )


def compatibility_record_for_policy(
    policy: DSLPolicy,
    *,
    max_rules: int = DEFAULT_MAX_RULES,
    max_conditions: int = DEFAULT_MAX_CONDITIONS,
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> dict[str, Any]:
    """Return one row describing whether a policy can use the batch path."""

    try:
        compiled = compile_policy(
            policy,
            max_rules=max_rules,
            max_conditions=max_conditions,
            max_actions=max_actions,
        )
    except Exception as exc:
        return {
            "policy_id": policy.policy_id,
            "policy_name": policy.name,
            "dsl_sha256": policy.sha256,
            "compiles_for_batch": False,
            "requires_cpu_fallback": True,
            "fallback_reasons_json": json.dumps([repr(exc)], separators=(",", ":")),
            "rule_count": len(policy.rules),
            "max_rules": max_rules,
            "max_conditions": max_conditions,
            "max_actions": max_actions,
        }
    return compiled.compatibility_record()


def load_generated_policy_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_generated_policies(path: Path) -> list[DSLPolicy]:
    return [parse_policy(record["dslSource"]) for record in load_generated_policy_records(path)]


def _stack_compiled(compiled: Sequence[CompiledPolicy], jnp: Any) -> dict[str, Any]:
    def stack(attr: str, dtype: Any | None = None) -> Any:
        arr = np.stack([getattr(item, attr) for item in compiled], axis=0)
        return jnp.asarray(arr, dtype=dtype) if dtype is not None else jnp.asarray(arr)

    return {
        "rule_mask": stack("rule_mask"),
        "rule_is_else": stack("rule_is_else"),
        "condition_code": stack("condition_code", jnp.int32),
        "condition_target": stack("condition_target", jnp.int32),
        "condition_probability": stack("condition_probability", jnp.float32),
        "action_code": stack("action_code", jnp.int32),
        "action_target": stack("action_target", jnp.int32),
        "action_state_value": stack("action_state_value", jnp.int32),
        "action_probability": stack("action_probability", jnp.float32),
        "choose_true_code": stack("choose_true_code", jnp.int32),
        "choose_true_target": stack("choose_true_target", jnp.int32),
        "choose_true_state_value": stack("choose_true_state_value", jnp.int32),
        "choose_false_code": stack("choose_false_code", jnp.int32),
        "choose_false_target": stack("choose_false_target", jnp.int32),
        "choose_false_state_value": stack("choose_false_state_value", jnp.int32),
        "state_init_code": jnp.asarray([item.state_init_code for item in compiled], dtype=jnp.int32),
    }


def _gather_row(values: Any, index: Any, jnp: Any) -> Any:
    n = values.shape[1]
    clipped = jnp.clip(index, 0, n - 1)
    return jnp.take_along_axis(values, clipped[:, None], axis=1)[:, 0]


def _resolve_target(target_code: Any, actor_index: Any, ideal_position: Any, jnp: Any) -> Any:
    return jnp.where(
        target_code == TARGET_LEFT,
        actor_index - 1,
        jnp.where(target_code == TARGET_RIGHT, actor_index + 1, jnp.where(target_code == TARGET_IDEAL, ideal_position, -1)),
    )


def _resolve_state_value(
    state_code: Any,
    actor_index: Any,
    ideal_position: Any,
    left_boundary: Any,
    right_boundary: Any,
    reverse_direction: Any,
    jnp: Any,
) -> Any:
    next_position = jnp.where(reverse_direction, ideal_position - 1, ideal_position + 1)
    return jnp.where(
        state_code >= 0,
        state_code,
        jnp.where(
            state_code == STATE_LEFT_BOUNDARY,
            left_boundary,
            jnp.where(
                state_code == STATE_RIGHT_BOUNDARY,
                right_boundary,
                jnp.where(
                    state_code == STATE_CURRENT,
                    actor_index,
                    jnp.where(state_code == STATE_NEXT, next_position, -1),
                ),
            ),
        ),
    )


def _target_exists_index(index: Any, left_boundary: Any, right_boundary: Any, jnp: Any) -> Any:
    return (index >= left_boundary) & (index <= right_boundary)


def _prefix_sorted(values: Any, statuses: Any, actor_index: Any, left_boundary: Any, reverse_direction: Any, jnp: Any) -> Any:
    batch_size, n = values.shape
    prev = jnp.where(reverse_direction, 100000, -1).astype(jnp.int32)
    ok = jnp.ones(batch_size, dtype=bool)
    for position in range(n):
        in_prefix = (position >= left_boundary) & (position < actor_index)
        frozen = statuses[:, position] == STATUS_FREEZE
        value = values[:, position]
        fails = in_prefix & (~frozen) & jnp.where(reverse_direction, value > prev, value < prev)
        ok = ok & (~fails)
        prev = jnp.where(in_prefix & frozen, -1, jnp.where(in_prefix & (~frozen), value, prev))
    return ok


def _condition_pass(
    code: Any,
    target_code: Any,
    probability: Any,
    roll: Any,
    values: Any,
    statuses: Any,
    actor_index: Any,
    ideal_position: Any,
    left_boundary: Any,
    right_boundary: Any,
    reverse_direction: Any,
    jnp: Any,
) -> Any:
    target_index = _resolve_target(target_code, actor_index, ideal_position, jnp)
    exists = _target_exists_index(target_index, left_boundary, right_boundary, jnp)
    target_status = _gather_row(statuses, target_index, jnp)
    target_value = _gather_row(values, target_index, jnp)
    actor_value = _gather_row(values, actor_index, jnp)
    movable = (target_status == STATUS_ACTIVE) | (target_status == STATUS_FREEZE)
    prefix_ok = _prefix_sorted(values, statuses, actor_index, left_boundary, reverse_direction, jnp)
    passed = jnp.ones_like(actor_index, dtype=bool)
    passed = jnp.where(code == COND_NONE, True, passed)
    passed = jnp.where(code == COND_ALWAYS, True, passed)
    passed = jnp.where(code == COND_TARGET_EXISTS, exists, passed)
    passed = jnp.where(code == COND_TARGET_ACTIVE, target_status == STATUS_ACTIVE, passed)
    passed = jnp.where(code == COND_TARGET_MOVABLE, movable, passed)
    passed = jnp.where(code == COND_SELF_LT, exists & (actor_value < target_value), passed)
    passed = jnp.where(code == COND_SELF_GT, exists & (actor_value > target_value), passed)
    passed = jnp.where(code == COND_SELF_LE, exists & (actor_value <= target_value), passed)
    passed = jnp.where(code == COND_SELF_GE, exists & (actor_value >= target_value), passed)
    passed = jnp.where(code == COND_AT_IDEAL, (ideal_position >= 0) & (actor_index == ideal_position), passed)
    passed = jnp.where(code == COND_NOT_AT_IDEAL, (ideal_position >= 0) & (actor_index != ideal_position), passed)
    passed = jnp.where(code == COND_PREFIX_SORTED, prefix_ok, passed)
    passed = jnp.where(code == COND_RANDOM_LT, roll < probability, passed)
    return passed


def _apply_policy_state_init(
    arrays: dict[str, Any],
    actor_index: Any,
    ideal_position: Any,
    left_boundary: Any,
    right_boundary: Any,
    reverse_direction: Any,
    jnp: Any,
) -> Any:
    state_init = arrays["state_init_code"]
    resolved = _resolve_state_value(
        state_init,
        actor_index,
        ideal_position,
        left_boundary,
        right_boundary,
        reverse_direction,
        jnp,
    )
    needs_init = (ideal_position < 0) & (state_init != STATE_NONE)
    return jnp.where(needs_init, resolved, ideal_position)


def _selected_rule_field(field: Any, selected_rule: Any, slot: int, jnp: Any) -> Any:
    selected_safe = jnp.clip(selected_rule, 0, field.shape[1] - 1)
    return jnp.take_along_axis(field[:, :, slot], selected_safe[:, None], axis=1)[:, 0]


def _apply_swap(
    *,
    active: Any,
    target_code: Any,
    values: Any,
    statuses: Any,
    actor_index: Any,
    ideal_position: Any,
    left_boundary: Any,
    right_boundary: Any,
    jnp: Any,
) -> tuple[Any, Any, Any, Any]:
    target_index = _resolve_target(target_code, actor_index, ideal_position, jnp)
    exists = _target_exists_index(target_index, left_boundary, right_boundary, jnp)
    target_status = _gather_row(statuses, target_index, jnp)
    target_movable = (target_status == STATUS_ACTIVE) | (target_status == STATUS_FREEZE)
    do_swap = active & exists & target_movable
    positions = jnp.arange(values.shape[1])[None, :]
    actor_mask = positions == actor_index[:, None]
    target_mask = positions == target_index[:, None]
    actor_value = _gather_row(values, actor_index, jnp)
    target_value = _gather_row(values, target_index, jnp)
    actor_status = _gather_row(statuses, actor_index, jnp)
    values_after = jnp.where(
        do_swap[:, None] & actor_mask,
        target_value[:, None],
        jnp.where(do_swap[:, None] & target_mask, actor_value[:, None], values),
    )
    statuses_after = jnp.where(
        do_swap[:, None] & actor_mask,
        target_status[:, None],
        jnp.where(do_swap[:, None] & target_mask, actor_status[:, None], statuses),
    )
    actor_after = jnp.where(do_swap, target_index, actor_index)
    return values_after, statuses_after, actor_after, do_swap


def _apply_update(
    *,
    active: Any,
    state_code: Any,
    actor_index: Any,
    ideal_position: Any,
    left_boundary: Any,
    right_boundary: Any,
    reverse_direction: Any,
    jnp: Any,
) -> Any:
    resolved = _resolve_state_value(
        state_code,
        actor_index,
        ideal_position,
        left_boundary,
        right_boundary,
        reverse_direction,
        jnp,
    )
    return jnp.where(active, resolved, ideal_position)


def _step_batch(
    arrays: dict[str, Any],
    values: Any,
    statuses: Any,
    actor_index: Any,
    ideal_position: Any,
    left_boundary: Any,
    right_boundary: Any,
    reverse_direction: Any,
    condition_rolls: Any,
    choice_rolls: Any,
    compare_count: Any,
    swap_count: Any,
    update_count: Any,
    wait_count: Any,
    jnp: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    ideal_position = _apply_policy_state_init(
        arrays,
        actor_index,
        ideal_position,
        left_boundary,
        right_boundary,
        reverse_direction,
        jnp,
    )
    batch_size = values.shape[0]
    selected_rule = jnp.full(batch_size, -1, dtype=jnp.int32)
    for rule_index in range(arrays["rule_mask"].shape[1]):
        rule_present = arrays["rule_mask"][:, rule_index]
        rule_pass = jnp.ones(batch_size, dtype=bool)
        for condition_index in range(arrays["condition_code"].shape[2]):
            code = arrays["condition_code"][:, rule_index, condition_index]
            target = arrays["condition_target"][:, rule_index, condition_index]
            probability = arrays["condition_probability"][:, rule_index, condition_index]
            roll = condition_rolls[:, rule_index, condition_index]
            rule_pass = rule_pass & _condition_pass(
                code,
                target,
                probability,
                roll,
                values,
                statuses,
                actor_index,
                ideal_position,
                left_boundary,
                right_boundary,
                reverse_direction,
                jnp,
            )
        rule_pass = jnp.where(arrays["rule_is_else"][:, rule_index], True, rule_pass)
        choose_rule = (selected_rule < 0) & rule_present & rule_pass
        selected_rule = jnp.where(choose_rule, rule_index, selected_rule)

    terminal = jnp.zeros(batch_size, dtype=bool)
    compare_seen = jnp.zeros(batch_size, dtype=bool)
    for action_slot in range(arrays["action_code"].shape[2]):
        code = _selected_rule_field(arrays["action_code"], selected_rule, action_slot, jnp)
        target = _selected_rule_field(arrays["action_target"], selected_rule, action_slot, jnp)
        state_value = _selected_rule_field(arrays["action_state_value"], selected_rule, action_slot, jnp)
        probability = _selected_rule_field(arrays["action_probability"], selected_rule, action_slot, jnp)
        active = (selected_rule >= 0) & (~terminal) & (code != ACTION_NONE)

        is_compare = active & (code == ACTION_COMPARE)
        compare_seen = compare_seen | is_compare

        is_swap = active & (code == ACTION_SWAP)
        values_after, statuses_after, actor_after, did_swap = _apply_swap(
            active=is_swap,
            target_code=target,
            values=values,
            statuses=statuses,
            actor_index=actor_index,
            ideal_position=ideal_position,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            jnp=jnp,
        )
        values = values_after
        statuses = statuses_after
        actor_index = actor_after
        swap_count = swap_count + did_swap.astype(jnp.int32)
        wait_count = wait_count + (is_swap & (~did_swap)).astype(jnp.int32)
        terminal = terminal | is_swap

        is_update = active & ((code == ACTION_SET_IDEAL) | (code == ACTION_ESTIMATE_TARGET_POSITION))
        ideal_position = _apply_update(
            active=is_update,
            state_code=state_value,
            actor_index=actor_index,
            ideal_position=ideal_position,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            reverse_direction=reverse_direction,
            jnp=jnp,
        )
        update_count = update_count + is_update.astype(jnp.int32)
        terminal = terminal | is_update

        is_wait = active & (code == ACTION_WAIT)
        wait_count = wait_count + is_wait.astype(jnp.int32)
        terminal = terminal | is_wait

        is_choose = active & (code == ACTION_CHOOSE)
        choice_roll = _selected_rule_field(choice_rolls, selected_rule, action_slot, jnp)
        choice = choice_roll < probability
        true_code = _selected_rule_field(arrays["choose_true_code"], selected_rule, action_slot, jnp)
        true_target = _selected_rule_field(arrays["choose_true_target"], selected_rule, action_slot, jnp)
        true_state = _selected_rule_field(arrays["choose_true_state_value"], selected_rule, action_slot, jnp)
        false_code = _selected_rule_field(arrays["choose_false_code"], selected_rule, action_slot, jnp)
        false_target = _selected_rule_field(arrays["choose_false_target"], selected_rule, action_slot, jnp)
        false_state = _selected_rule_field(arrays["choose_false_state_value"], selected_rule, action_slot, jnp)
        nested_code = jnp.where(choice, true_code, false_code)
        nested_target = jnp.where(choice, true_target, false_target)
        nested_state = jnp.where(choice, true_state, false_state)

        nested_compare = is_choose & (nested_code == ACTION_COMPARE)
        compare_seen = compare_seen | nested_compare
        nested_swap = is_choose & (nested_code == ACTION_SWAP)
        values_after, statuses_after, actor_after, nested_did_swap = _apply_swap(
            active=nested_swap,
            target_code=nested_target,
            values=values,
            statuses=statuses,
            actor_index=actor_index,
            ideal_position=ideal_position,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            jnp=jnp,
        )
        values = values_after
        statuses = statuses_after
        actor_index = actor_after
        swap_count = swap_count + nested_did_swap.astype(jnp.int32)
        wait_count = wait_count + (nested_swap & (~nested_did_swap)).astype(jnp.int32)

        nested_update = is_choose & (
            (nested_code == ACTION_SET_IDEAL) | (nested_code == ACTION_ESTIMATE_TARGET_POSITION)
        )
        ideal_position = _apply_update(
            active=nested_update,
            state_code=nested_state,
            actor_index=actor_index,
            ideal_position=ideal_position,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            reverse_direction=reverse_direction,
            jnp=jnp,
        )
        update_count = update_count + nested_update.astype(jnp.int32)
        nested_wait = is_choose & (
            (nested_code == ACTION_WAIT)
            | (nested_code == ACTION_COMPARE)
            | (nested_code == ACTION_REMEMBER)
            | (nested_code == ACTION_SIGNAL)
            | (nested_code == ACTION_NONE)
        )
        wait_count = wait_count + nested_wait.astype(jnp.int32)
        terminal = terminal | is_choose

    implicit_wait = ~terminal
    wait_count = wait_count + implicit_wait.astype(jnp.int32)
    compare_count = compare_count + compare_seen.astype(jnp.int32)
    return values, statuses, actor_index, ideal_position, compare_count, swap_count, update_count, wait_count


def simulate_compiled_batch(
    compiled: Sequence[CompiledPolicy],
    initial_states: Sequence[DSLArrayState],
    *,
    steps: int,
    seed: int = 0,
    backend: str | None = None,
    allow_cpu_fallback: bool = True,
) -> BatchSimulationResult:
    """Run one fixed-length batch of compiled policies with JAX.

    Each row pairs ``compiled[i]`` with ``initial_states[i]``.  All rows must
    share the same array length; S07 can group larger sweeps by length.
    """

    if len(compiled) != len(initial_states):
        raise ValueError("compiled and initial_states must have the same length")
    if not compiled:
        raise ValueError("at least one policy is required")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    n = len(initial_states[0].values)
    if any(len(state.values) != n for state in initial_states):
        raise ValueError("all initial states in a batch must share one array length")

    jax, jnp = _require_jax()
    requested_backend = backend
    try:
        devices = jax.devices(requested_backend) if requested_backend else jax.devices()
    except RuntimeError:
        if not allow_cpu_fallback:
            raise
        devices = jax.devices("cpu")
    device = devices[0]
    arrays = _stack_compiled(compiled, jnp)
    arrays = {key: jax.device_put(value, device) for key, value in arrays.items()}

    values = jax.device_put(jnp.asarray([state.values for state in initial_states], dtype=jnp.int32), device)
    statuses = jax.device_put(
        jnp.asarray(
            [[status_to_code(status) for status in (state.statuses or ())] for state in initial_states],
            dtype=jnp.int32,
        ),
        device,
    )
    actor_index = jax.device_put(jnp.asarray([state.actor_index for state in initial_states], dtype=jnp.int32), device)
    ideal_position = jax.device_put(
        jnp.asarray([-1 if state.ideal_position is None else state.ideal_position for state in initial_states], dtype=jnp.int32),
        device,
    )
    left_boundary = jax.device_put(jnp.asarray([state.left_boundary for state in initial_states], dtype=jnp.int32), device)
    right_boundary = jax.device_put(jnp.asarray([state.right_boundary for state in initial_states], dtype=jnp.int32), device)
    reverse_direction = jax.device_put(
        jnp.asarray([state.reverse_direction for state in initial_states], dtype=bool),
        device,
    )
    compare_count = jax.device_put(jnp.zeros(len(initial_states), dtype=jnp.int32), device)
    swap_count = jax.device_put(jnp.zeros(len(initial_states), dtype=jnp.int32), device)
    update_count = jax.device_put(jnp.zeros(len(initial_states), dtype=jnp.int32), device)
    wait_count = jax.device_put(jnp.zeros(len(initial_states), dtype=jnp.int32), device)

    rng = np.random.default_rng(seed)
    condition_rolls = jax.device_put(
        jnp.asarray(
            rng.random((steps, len(initial_states), compiled[0].max_rules, compiled[0].max_conditions), dtype=np.float32),
            dtype=jnp.float32,
        ),
        device,
    )
    choice_rolls = jax.device_put(
        jnp.asarray(
            rng.random((steps, len(initial_states), compiled[0].max_rules, compiled[0].max_actions), dtype=np.float32),
            dtype=jnp.float32,
        ),
        device,
    )

    for step_index in range(steps):
        values, statuses, actor_index, ideal_position, compare_count, swap_count, update_count, wait_count = _step_batch(
            arrays,
            values,
            statuses,
            actor_index,
            ideal_position,
            left_boundary,
            right_boundary,
            reverse_direction,
            condition_rolls[step_index],
            choice_rolls[step_index],
            compare_count,
            swap_count,
            update_count,
            wait_count,
            jnp,
        )

    values.block_until_ready()
    actual_backend = getattr(device, "platform", jax.default_backend())
    return BatchSimulationResult(
        values=np.asarray(jax.device_get(values)),
        statuses=np.asarray(jax.device_get(statuses)),
        actor_index=np.asarray(jax.device_get(actor_index)),
        ideal_position=np.asarray(jax.device_get(ideal_position)),
        compare_count=np.asarray(jax.device_get(compare_count)),
        swap_count=np.asarray(jax.device_get(swap_count)),
        update_count=np.asarray(jax.device_get(update_count)),
        wait_count=np.asarray(jax.device_get(wait_count)),
        backend=str(actual_backend),
        device=str(device),
    )


def run_cpu_reference(policy: DSLPolicy, initial_state: DSLArrayState, *, steps: int, seed: int = 0) -> dict[str, Any]:
    """Run the S02 interpreter as the CPU reference for a fixed step count."""

    interpreter = DSLInterpreter(policy)
    rng = random.Random(seed)
    state = initial_state
    compare_count = 0
    swap_count = 0
    update_count = 0
    wait_count = 0
    actions: list[str] = []
    for _step in range(steps):
        result = interpreter.step_state(state, rng)
        state = result.state_after
        action = result.action
        actions.append(action.action_type)
        compare_count += int(bool(action.compare_counted))
        swap_count += int(action.action_type == "swap")
        update_count += int(action.action_type == "update_state")
        wait_count += int(action.action_type == "wait")
    return {
        "values": tuple(state.values),
        "statuses": tuple(state.statuses or ()),
        "actor_index": int(state.actor_index),
        "ideal_position": -1 if state.ideal_position is None else int(state.ideal_position),
        "compare_count": int(compare_count),
        "swap_count": int(swap_count),
        "update_count": int(update_count),
        "wait_count": int(wait_count),
        "actions": tuple(actions),
    }


def compare_batch_to_cpu(
    policies: Sequence[DSLPolicy],
    initial_states: Sequence[DSLArrayState],
    *,
    steps: int,
    seed: int = 0,
    backend: str | None = None,
) -> pd.DataFrame:
    """Return row-wise CPU/JAX agreement records for deterministic fixtures."""

    compiled = [compile_policy(policy) for policy in policies]
    batch = simulate_compiled_batch(compiled, initial_states, steps=steps, seed=seed, backend=backend)
    rows: list[dict[str, Any]] = []
    for idx, (policy, state) in enumerate(zip(policies, initial_states, strict=True)):
        cpu = run_cpu_reference(policy, state, steps=steps, seed=seed)
        gpu_values = tuple(int(value) for value in batch.values[idx].tolist())
        gpu_statuses = tuple(status_from_code(code) for code in batch.statuses[idx].tolist())
        gpu_ideal = int(batch.ideal_position[idx])
        metrics_match = (
            cpu["compare_count"] == int(batch.compare_count[idx])
            and cpu["swap_count"] == int(batch.swap_count[idx])
            and cpu["update_count"] == int(batch.update_count[idx])
            and cpu["wait_count"] == int(batch.wait_count[idx])
        )
        state_match = (
            tuple(cpu["values"]) == gpu_values
            and tuple(cpu["statuses"]) == gpu_statuses
            and int(cpu["actor_index"]) == int(batch.actor_index[idx])
            and int(cpu["ideal_position"]) == gpu_ideal
        )
        rows.append(
            {
                "row_index": idx,
                "policy_id": policy.policy_id,
                "policy_name": policy.name,
                "steps": int(steps),
                "initial_values_json": json.dumps(list(state.values), separators=(",", ":")),
                "initial_actor_index": int(state.actor_index),
                "initial_ideal_position": -1 if state.ideal_position is None else int(state.ideal_position),
                "cpu_final_values_json": json.dumps(list(cpu["values"]), separators=(",", ":")),
                "gpu_final_values_json": json.dumps(list(gpu_values), separators=(",", ":")),
                "cpu_final_statuses_json": json.dumps(list(cpu["statuses"]), separators=(",", ":")),
                "gpu_final_statuses_json": json.dumps(list(gpu_statuses), separators=(",", ":")),
                "cpu_actor_index": int(cpu["actor_index"]),
                "gpu_actor_index": int(batch.actor_index[idx]),
                "cpu_ideal_position": int(cpu["ideal_position"]),
                "gpu_ideal_position": gpu_ideal,
                "cpu_compare_count": int(cpu["compare_count"]),
                "gpu_compare_count": int(batch.compare_count[idx]),
                "cpu_swap_count": int(cpu["swap_count"]),
                "gpu_swap_count": int(batch.swap_count[idx]),
                "cpu_update_count": int(cpu["update_count"]),
                "gpu_update_count": int(batch.update_count[idx]),
                "cpu_wait_count": int(cpu["wait_count"]),
                "gpu_wait_count": int(batch.wait_count[idx]),
                "final_state_match": bool(state_match),
                "metrics_match": bool(metrics_match),
                "within_tolerance": bool(state_match and metrics_match),
                "backend": batch.backend,
                "device": batch.device,
            }
        )
    return pd.DataFrame(rows)


def compatibility_frame(policies: Sequence[DSLPolicy]) -> pd.DataFrame:
    return pd.DataFrame([compatibility_record_for_policy(policy) for policy in policies])


def compatibility_summary(frame: pd.DataFrame) -> pd.DataFrame:
    total = len(frame)
    compiled = int(frame["compiles_for_batch"].sum()) if total else 0
    fallback = int(frame["requires_cpu_fallback"].sum()) if total else 0
    ready = int(((frame["compiles_for_batch"]) & (~frame["requires_cpu_fallback"])).sum()) if total else 0
    return pd.DataFrame(
        [
            {"metric": "policies_total", "value": total, "detail": "Policies loaded for compatibility classification."},
            {"metric": "policies_compile_for_batch", "value": compiled, "detail": "Policies that fit the S06 padded DSL arrays."},
            {
                "metric": "policies_ready_for_exact_batch_validation",
                "value": ready,
                "detail": "Compiled policies without stochastic-stream or memory/signal fallback reasons.",
            },
            {
                "metric": "policies_requiring_cpu_fallback_for_exact_replay",
                "value": fallback,
                "detail": "Policies with stochastic replay or memory/signal semantics not exact in S06.",
            },
        ]
    )


def deterministic_validation_policies(library_path: Path | None = None) -> list[DSLPolicy]:
    """Return a small deterministic policy set for CPU/JAX parity checks."""

    sources = [
        """policy s06_swap_left v1
state ideal_position=none
rule if target_exists(left) and target_movable(left) and self_lt(left) then compare(left), swap(left)
rule else wait
end
""",
        """policy s06_swap_right v1
state ideal_position=none
rule if target_exists(right) and target_movable(right) and self_gt(right) then compare(right), swap(right)
rule else wait
end
""",
        """policy s06_prefix_left v1
state ideal_position=none
rule if prefix_sorted and target_exists(left) and target_active(left) and self_lt(left) then compare(left), swap(left)
rule else wait
end
""",
        """policy s06_selection_target v1
state ideal_position=left_boundary
rule if target_exists(ideal) and not_at_ideal and target_movable(ideal) and self_lt(ideal) then compare(ideal), swap(ideal)
rule if target_exists(ideal) and not_at_ideal and target_movable(ideal) and self_ge(ideal) then compare(ideal), set_ideal(next)
rule else wait
end
""",
        """policy s06_deterministic_random_choose v1
state ideal_position=none
rule if random_lt(1.0) and target_exists(right) and target_movable(right) and self_gt(right) then compare(right), choose(1.0, swap(right), wait)
rule else wait
end
""",
        """policy s06_reverse_selection_update v1
state ideal_position=right_boundary
rule if target_exists(ideal) and not_at_ideal and target_movable(ideal) and self_ge(ideal) then compare(ideal), set_ideal(next)
rule else wait
end
""",
    ]
    policies = [parse_policy(source) for source in sources]
    if library_path and library_path.exists():
        records = load_generated_policy_records(library_path)
        for record in records:
            features = record.get("features", {})
            if features.get("usesMemory") or features.get("usesSignal"):
                continue
            if features.get("usesRandomCondition") or features.get("usesProbabilisticAction"):
                continue
            if len(policies) >= 10:
                break
            policies.append(parse_policy(record["dslSource"]))
    return policies


def validation_fixtures(row_count: int) -> list[DSLArrayState]:
    """Return fixed-length fixtures repeated to match a validation batch."""

    fixtures = [
        DSLArrayState(values=(4, 2, 1, 3, 5), actor_index=2),
        DSLArrayState(values=(2, 4, 3, 1, 5), actor_index=1),
        DSLArrayState(values=(1, 3, 2, 4, 5), actor_index=2),
        DSLArrayState(values=(5, 1, 4, 2, 3), actor_index=3, ideal_position=None),
        DSLArrayState(values=(2, 1, 5, 4, 3), actor_index=0),
        DSLArrayState(values=(5, 4, 3, 2, 1), actor_index=2, reverse_direction=True),
        DSLArrayState(values=(3, 1, 2, 5, 4), actor_index=2, statuses=("ACTIVE", "FREEZE", "ACTIVE", "ACTIVE", "ACTIVE")),
        DSLArrayState(values=(1, 2, 5, 3, 4), actor_index=3),
    ]
    return [fixtures[idx % len(fixtures)] for idx in range(row_count)]


def comparison_digest(frame: pd.DataFrame) -> str:
    import hashlib

    payload = frame.sort_values(["row_index"]).to_dict(orient="records")
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
