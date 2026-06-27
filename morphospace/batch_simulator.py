"""JAX batch simulator for the S06 stateless adjacent DSL subset.

This module is deliberately narrower than the full S01/S02 interpreter. It is
the first GPU-friendly path for coarse sweeps, so it favors exact CPU agreement
for a small fixed DSL subset over partial support for stateful or stochastic
programs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised in environments with JAX installed.
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - defensive import path for docs tooling.
    jax = None
    jnp = None

from e02_deterministic_simulator.metrics import monotonicity_error, sortedness_percent, sortedness_raw, state_hash

from .policies import PolicyEventSimulator
from .rule_dsl import DSLPolicy, RuleProgram, parse_rule_program


BATCH_SIMULATOR_VERSION = "e03_s06_batch_simulator.v1"

PRED_ALWAYS = 0
PRED_LEFT_EXISTS = 1
PRED_RIGHT_EXISTS = 2
PRED_COMPARE_LEFT = 3
PRED_COMPARE_RIGHT = 4
PRED_POSITION_COMPARE = 5

OP_LT = 0
OP_LE = 1
OP_GT = 2
OP_GE = 3
OP_EQ = 4
OP_NE = 5

ACTION_WAIT = 0
ACTION_SWAP_LEFT = 1
ACTION_SWAP_RIGHT = 2

STOP_RUNNING = 0
STOP_SORTED = 1
STOP_MAX_ACTIVATION = 2
STOP_MAX_SWAP = 3
STOP_MAX_COMPARISON = 4
STOP_NO_MOVE = 5

PREDICATE_CODES = {
    "always": PRED_ALWAYS,
    "left_exists": PRED_LEFT_EXISTS,
    "right_exists": PRED_RIGHT_EXISTS,
    "compare_left": PRED_COMPARE_LEFT,
    "compare_right": PRED_COMPARE_RIGHT,
    "position_compare": PRED_POSITION_COMPARE,
}
OPERATOR_CODES = {"<": OP_LT, "<=": OP_LE, ">": OP_GT, ">=": OP_GE, "==": OP_EQ, "!=": OP_NE}
ACTION_CODES = {"wait": ACTION_WAIT, "swap_left": ACTION_SWAP_LEFT, "swap_right": ACTION_SWAP_RIGHT}
STOP_REASON_BY_CODE = {
    STOP_RUNNING: "running",
    STOP_SORTED: "sorted",
    STOP_MAX_ACTIVATION: "max_activation_cap",
    STOP_MAX_SWAP: "max_step_cap",
    STOP_MAX_COMPARISON: "max_comparison_cap",
    STOP_NO_MOVE: "no_cell_can_move_after_two_checks",
}
SUPPORTED_PREDICATES = tuple(PREDICATE_CODES)
SUPPORTED_ACTIONS = tuple(ACTION_CODES)


@dataclass(frozen=True)
class BatchSupportReport:
    policy_id: str
    supported: bool
    reasons: tuple[str, ...]
    rule_count: int
    predicate_count: int
    action_names: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        return {
            "policyId": self.policy_id,
            "batchSupported": bool(self.supported),
            "supportReasonsJson": json.dumps(list(self.reasons), separators=(",", ":")),
            "ruleCount": int(self.rule_count),
            "predicateCount": int(self.predicate_count),
            "actionNamesJson": json.dumps(list(self.action_names), separators=(",", ":")),
            "batchSimulatorVersion": BATCH_SIMULATOR_VERSION,
        }


@dataclass(frozen=True)
class CompiledBatchPrograms:
    policy_ids: tuple[str, ...]
    predicate_ops: np.ndarray
    predicate_operators: np.ndarray
    predicate_values: np.ndarray
    predicate_mask: np.ndarray
    rule_mask: np.ndarray
    action_codes: np.ndarray

    @property
    def batch_size(self) -> int:
        return len(self.policy_ids)

    @property
    def max_rules(self) -> int:
        return int(self.rule_mask.shape[1])

    @property
    def max_predicates(self) -> int:
        return int(self.predicate_mask.shape[2])


@dataclass(frozen=True)
class BatchSimulationResult:
    policy_ids: tuple[str, ...]
    initial_values: np.ndarray
    final_values: np.ndarray
    swap_count: np.ndarray
    comparison_count: np.ndarray
    activation_count: np.ndarray
    stop_reason_codes: np.ndarray
    final_sortedness_raw_count: np.ndarray
    final_sortedness_percent: np.ndarray
    final_monotonicity_error: np.ndarray
    values_trace: np.ndarray
    swapped_trace: np.ndarray
    scheduler_seeds: tuple[int, ...]
    device: str
    batch_simulator_version: str = BATCH_SIMULATOR_VERSION

    @property
    def stop_reasons(self) -> tuple[str, ...]:
        return tuple(STOP_REASON_BY_CODE[int(code)] for code in self.stop_reason_codes)

    @property
    def completed(self) -> np.ndarray:
        return self.stop_reason_codes == STOP_SORTED

    def swap_state_hashes(self, row_index: int) -> list[str]:
        hashes = [state_hash(self.values_trace[0, row_index].tolist())]
        for step_index, swapped in enumerate(self.swapped_trace[:, row_index], start=1):
            if bool(swapped):
                hashes.append(state_hash(self.values_trace[step_index, row_index].tolist()))
        return hashes

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, policy_id in enumerate(self.policy_ids):
            rows.append(
                {
                    "policyId": policy_id,
                    "batchSimulatorVersion": self.batch_simulator_version,
                    "backendDevice": self.device,
                    "schedulerSeed": int(self.scheduler_seeds[index]),
                    "initialValuesJson": json.dumps(self.initial_values[index].astype(int).tolist(), separators=(",", ":")),
                    "finalValuesJson": json.dumps(self.final_values[index].astype(int).tolist(), separators=(",", ":")),
                    "completed": bool(self.completed[index]),
                    "stopReason": STOP_REASON_BY_CODE[int(self.stop_reason_codes[index])],
                    "swapCount": int(self.swap_count[index]),
                    "comparisonCount": int(self.comparison_count[index]),
                    "activationCount": int(self.activation_count[index]),
                    "eventCount": int(self.swap_count[index]) + 1,
                    "finalSortednessRawCount": int(self.final_sortedness_raw_count[index]),
                    "finalSortednessPercent": float(self.final_sortedness_percent[index]),
                    "finalMonotonicityError": int(self.final_monotonicity_error[index]),
                    "traceHashSequenceJson": json.dumps(self.swap_state_hashes(index), separators=(",", ":")),
                }
            )
        return rows


def _ensure_jax_available() -> None:
    if jax is None or jnp is None:
        raise RuntimeError("JAX is required for the S06 batch simulator")


def available_jax_devices() -> tuple[str, ...]:
    _ensure_jax_available()
    return tuple(str(device) for device in jax.devices())


def preferred_jax_device():
    _ensure_jax_available()
    devices = jax.devices()
    for device in devices:
        if device.platform == "gpu":
            return device
    return devices[0]


def _program_id(program: RuleProgram) -> str:
    return str(program.policy_id)


def batch_support_report(program: RuleProgram | Mapping[str, Any] | str) -> BatchSupportReport:
    program = parse_rule_program(program)
    reasons: list[str] = []
    action_names: list[str] = [rule.then.action for rule in program.rules] + [program.default.action]
    predicate_count = 0

    if program.initial_state:
        reasons.append("initial_state_not_supported")

    for rule in program.rules:
        if rule.then.action not in ACTION_CODES:
            reasons.append(f"unsupported_action:{rule.then.action}")
        if rule.then.probability != 1.0:
            reasons.append("stochastic_action_not_supported")
        if rule.then.updates:
            reasons.append("state_updates_not_supported")
        for predicate in rule.when:
            predicate_count += 1
            if predicate.op not in PREDICATE_CODES:
                reasons.append(f"unsupported_predicate:{predicate.op}")
            if predicate.op in {"compare_left", "compare_right", "position_compare"} and predicate.operator not in OPERATOR_CODES:
                reasons.append(f"unsupported_operator:{predicate.operator}")
            if predicate.op == "position_compare":
                try:
                    int(predicate.value)
                except Exception:
                    reasons.append("non_integer_position_compare_value")

    if program.default.action not in ACTION_CODES:
        reasons.append(f"unsupported_default_action:{program.default.action}")
    if program.default.probability != 1.0:
        reasons.append("stochastic_default_action_not_supported")
    if program.default.updates:
        reasons.append("default_state_updates_not_supported")

    unique_reasons = tuple(sorted(set(reasons)))
    return BatchSupportReport(
        policy_id=_program_id(program),
        supported=not unique_reasons,
        reasons=unique_reasons,
        rule_count=len(program.rules),
        predicate_count=predicate_count,
        action_names=tuple(action_names),
    )


def is_batch_supported(program: RuleProgram | Mapping[str, Any] | str) -> bool:
    return batch_support_report(program).supported


def compile_batch_programs(programs: Sequence[RuleProgram | Mapping[str, Any] | str]) -> CompiledBatchPrograms:
    parsed = [parse_rule_program(program) for program in programs]
    if not parsed:
        raise ValueError("at least one program is required")
    reports = [batch_support_report(program) for program in parsed]
    unsupported = [report for report in reports if not report.supported]
    if unsupported:
        detail = "; ".join(f"{report.policy_id}:{','.join(report.reasons)}" for report in unsupported[:5])
        raise ValueError(f"unsupported programs for S06 batch simulator: {detail}")

    max_rules = max(len(program.rules) for program in parsed)
    max_predicates = max(max(len(rule.when) for rule in program.rules) for program in parsed)
    count = len(parsed)
    predicate_ops = np.full((count, max_rules, max_predicates), PRED_ALWAYS, dtype=np.int32)
    predicate_operators = np.full((count, max_rules, max_predicates), OP_EQ, dtype=np.int32)
    predicate_values = np.zeros((count, max_rules, max_predicates), dtype=np.int32)
    predicate_mask = np.zeros((count, max_rules, max_predicates), dtype=bool)
    rule_mask = np.zeros((count, max_rules), dtype=bool)
    action_codes = np.full((count, max_rules + 1), ACTION_WAIT, dtype=np.int32)

    for program_index, program in enumerate(parsed):
        for rule_index, rule in enumerate(program.rules):
            rule_mask[program_index, rule_index] = True
            action_codes[program_index, rule_index] = ACTION_CODES[rule.then.action]
            for predicate_index, predicate in enumerate(rule.when):
                predicate_mask[program_index, rule_index, predicate_index] = True
                predicate_ops[program_index, rule_index, predicate_index] = PREDICATE_CODES[predicate.op]
                if predicate.operator is not None:
                    predicate_operators[program_index, rule_index, predicate_index] = OPERATOR_CODES[predicate.operator]
                if predicate.value is not None:
                    predicate_values[program_index, rule_index, predicate_index] = int(predicate.value)
        action_codes[program_index, max_rules] = ACTION_CODES[program.default.action]

    return CompiledBatchPrograms(
        policy_ids=tuple(_program_id(program) for program in parsed),
        predicate_ops=predicate_ops,
        predicate_operators=predicate_operators,
        predicate_values=predicate_values,
        predicate_mask=predicate_mask,
        rule_mask=rule_mask,
        action_codes=action_codes,
    )


def actor_schedule(n: int, max_activations: int, scheduler_seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(scheduler_seed))
    return rng.choice(np.arange(n, dtype=np.int32), size=int(max_activations) + 1).astype(np.int32)


def actor_schedules(n: int, max_activations: int, scheduler_seeds: Sequence[int]) -> np.ndarray:
    return np.stack([actor_schedule(n, max_activations, seed) for seed in scheduler_seeds], axis=0)


def _to_device_arrays(compiled: CompiledBatchPrograms, initial_values: np.ndarray, schedules: np.ndarray, device):
    return (
        jax.device_put(np.asarray(initial_values, dtype=np.int32), device),
        jax.device_put(np.asarray(schedules, dtype=np.int32), device),
        jax.device_put(compiled.predicate_ops.astype(np.int32), device),
        jax.device_put(compiled.predicate_operators.astype(np.int32), device),
        jax.device_put(compiled.predicate_values.astype(np.int32), device),
        jax.device_put(compiled.predicate_mask.astype(bool), device),
        jax.device_put(compiled.rule_mask.astype(bool), device),
        jax.device_put(compiled.action_codes.astype(np.int32), device),
    )


def _sortedness_raw_jax(values):
    return jnp.sum(values[:-1] <= values[1:])


def _compare_jax(left, operator, right):
    return jnp.select(
        [
            operator == OP_LT,
            operator == OP_LE,
            operator == OP_GT,
            operator == OP_GE,
            operator == OP_EQ,
            operator == OP_NE,
        ],
        [left < right, left <= right, left > right, left >= right, left == right, left != right],
        default=False,
    )


def _eval_predicate_jax(values, actor_pos, pred_op, operator, pred_value):
    n = values.shape[0]
    actor_value = values[actor_pos]
    left_exists = actor_pos > 0
    right_exists = actor_pos < n - 1
    left_value = values[jnp.maximum(actor_pos - 1, 0)]
    right_value = values[jnp.minimum(actor_pos + 1, n - 1)]
    return jnp.select(
        [
            pred_op == PRED_ALWAYS,
            pred_op == PRED_LEFT_EXISTS,
            pred_op == PRED_RIGHT_EXISTS,
            pred_op == PRED_COMPARE_LEFT,
            pred_op == PRED_COMPARE_RIGHT,
            pred_op == PRED_POSITION_COMPARE,
        ],
        [
            True,
            left_exists,
            right_exists,
            left_exists & _compare_jax(actor_value, operator, left_value),
            right_exists & _compare_jax(actor_value, operator, right_value),
            _compare_jax(actor_pos, operator, pred_value),
        ],
        default=False,
    ), jnp.select(
        [pred_op == PRED_COMPARE_LEFT, pred_op == PRED_COMPARE_RIGHT],
        [jnp.where(left_exists, 1, 0), jnp.where(right_exists, 1, 0)],
        default=0,
    )


def _select_action_one_jax(values, actor_pos, predicate_ops, predicate_operators, predicate_values, predicate_mask, rule_mask, action_codes):
    selected_action = action_codes[-1]
    matched = jnp.asarray(False)
    comparison_delta = jnp.asarray(0, dtype=jnp.int32)
    max_rules = rule_mask.shape[0]
    max_predicates = predicate_mask.shape[1]
    for rule_index in range(max_rules):
        checking = rule_mask[rule_index] & ~matched
        for predicate_index in range(max_predicates):
            should_eval = checking & predicate_mask[rule_index, predicate_index]
            predicate_ok, predicate_delta = _eval_predicate_jax(
                values,
                actor_pos,
                predicate_ops[rule_index, predicate_index],
                predicate_operators[rule_index, predicate_index],
                predicate_values[rule_index, predicate_index],
            )
            comparison_delta = comparison_delta + jnp.where(should_eval, predicate_delta, 0)
            checking = jnp.where(should_eval, checking & predicate_ok, checking)
        rule_match = checking & rule_mask[rule_index] & ~matched
        selected_action = jnp.where(rule_match, action_codes[rule_index], selected_action)
        matched = matched | rule_match
    return selected_action, comparison_delta


def _legal_action_exists_one_jax(values, cell_ids, predicate_ops, predicate_operators, predicate_values, predicate_mask, rule_mask, action_codes):
    positions = jnp.arange(values.shape[0], dtype=jnp.int32)

    def legal_at_position(position):
        action, _ = _select_action_one_jax(
            values,
            position,
            predicate_ops,
            predicate_operators,
            predicate_values,
            predicate_mask,
            rule_mask,
            action_codes,
        )
        return ((action == ACTION_SWAP_LEFT) & (position > 0)) | ((action == ACTION_SWAP_RIGHT) & (position < values.shape[0] - 1))

    return jnp.any(jax.vmap(legal_at_position)(positions))


def _step_one_jax(carry, scheduled_actor_id):
    (
        values,
        cell_ids,
        swap_count,
        comparison_count,
        activation_count,
        no_move_checks,
        stop_reason,
        predicate_ops,
        predicate_operators,
        predicate_values,
        predicate_mask,
        rule_mask,
        action_codes,
        max_activations,
        max_swaps,
        max_comparisons,
        no_move_checks_required,
        no_move_check_interval,
    ) = carry
    active = stop_reason == STOP_RUNNING
    sorted_now = _sortedness_raw_jax(values) == values.shape[0] - 1
    stop_reason = jnp.where(active & sorted_now, STOP_SORTED, stop_reason)
    active = stop_reason == STOP_RUNNING
    stop_reason = jnp.where(active & (activation_count >= max_activations), STOP_MAX_ACTIVATION, stop_reason)
    active = stop_reason == STOP_RUNNING
    stop_reason = jnp.where(active & (swap_count >= max_swaps), STOP_MAX_SWAP, stop_reason)
    active = stop_reason == STOP_RUNNING
    stop_reason = jnp.where(active & (comparison_count >= max_comparisons), STOP_MAX_COMPARISON, stop_reason)
    active = stop_reason == STOP_RUNNING

    check_due = active & ((activation_count % no_move_check_interval) == 0)
    legal_exists = _legal_action_exists_one_jax(
        values,
        cell_ids,
        predicate_ops,
        predicate_operators,
        predicate_values,
        predicate_mask,
        rule_mask,
        action_codes,
    )
    new_no_move_checks = jnp.where(check_due & ~legal_exists, no_move_checks + 1, jnp.where(check_due & legal_exists, 0, no_move_checks))
    stop_no_move = check_due & ~legal_exists & (new_no_move_checks >= no_move_checks_required)
    stop_reason = jnp.where(stop_no_move, STOP_NO_MOVE, stop_reason)
    no_move_checks = new_no_move_checks
    active = stop_reason == STOP_RUNNING

    actor_pos = jnp.argmax(cell_ids == scheduled_actor_id).astype(jnp.int32)
    action, comparison_delta = _select_action_one_jax(
        values,
        actor_pos,
        predicate_ops,
        predicate_operators,
        predicate_values,
        predicate_mask,
        rule_mask,
        action_codes,
    )
    target_pos = jnp.where(action == ACTION_SWAP_LEFT, actor_pos - 1, jnp.where(action == ACTION_SWAP_RIGHT, actor_pos + 1, actor_pos))
    legal_swap = active & (((action == ACTION_SWAP_LEFT) & (actor_pos > 0)) | ((action == ACTION_SWAP_RIGHT) & (actor_pos < values.shape[0] - 1)))
    target_clipped = jnp.clip(target_pos, 0, values.shape[0] - 1)
    indices = jnp.arange(values.shape[0], dtype=jnp.int32)
    actor_value = values[actor_pos]
    target_value = values[target_clipped]
    actor_cell_id = cell_ids[actor_pos]
    target_cell_id = cell_ids[target_clipped]
    swapped_values = jnp.where(indices == actor_pos, target_value, jnp.where(indices == target_clipped, actor_value, values))
    swapped_cell_ids = jnp.where(indices == actor_pos, target_cell_id, jnp.where(indices == target_clipped, actor_cell_id, cell_ids))
    values = jnp.where(legal_swap, swapped_values, values)
    cell_ids = jnp.where(legal_swap, swapped_cell_ids, cell_ids)
    swap_count = swap_count + jnp.where(legal_swap, 1, 0)
    comparison_count = comparison_count + jnp.where(active, comparison_delta, 0)
    activation_count = activation_count + jnp.where(active, 1, 0)

    new_carry = (
        values,
        cell_ids,
        swap_count,
        comparison_count,
        activation_count,
        no_move_checks,
        stop_reason,
        predicate_ops,
        predicate_operators,
        predicate_values,
        predicate_mask,
        rule_mask,
        action_codes,
        max_activations,
        max_swaps,
        max_comparisons,
        no_move_checks_required,
        no_move_check_interval,
    )
    trace = {
        "values": values,
        "swapped": legal_swap,
        "stop_reason": stop_reason,
    }
    return new_carry, trace


def _run_batch_jax(
    initial_values,
    schedules,
    predicate_ops,
    predicate_operators,
    predicate_values,
    predicate_mask,
    rule_mask,
    action_codes,
    max_activations,
    max_swaps,
    max_comparisons,
    no_move_checks_required,
    no_move_check_interval,
):
    batch_size, n = initial_values.shape
    initial_cell_ids = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), initial_values.shape)
    carry = (
        initial_values,
        initial_cell_ids,
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.zeros((batch_size,), dtype=jnp.int32),
    )
    programs = (predicate_ops, predicate_operators, predicate_values, predicate_mask, rule_mask, action_codes)
    constants = (
        jnp.asarray(max_activations, dtype=jnp.int32),
        jnp.asarray(max_swaps, dtype=jnp.int32),
        jnp.asarray(max_comparisons, dtype=jnp.int32),
        jnp.asarray(no_move_checks_required, dtype=jnp.int32),
        jnp.asarray(no_move_check_interval, dtype=jnp.int32),
    )

    def step_one(state, scheduled_actor_id, program):
        full_carry = state + program + constants
        new_full_carry, trace = _step_one_jax(full_carry, scheduled_actor_id)
        return new_full_carry[:7], trace

    mapped_step = jax.vmap(step_one, in_axes=(0, 0, (0, 0, 0, 0, 0, 0)), out_axes=(0, 0))

    def scan_step(carry_state, scheduled_ids):
        return mapped_step(carry_state, scheduled_ids, programs)

    final_carry, trace = jax.lax.scan(scan_step, carry, schedules.T)
    final_values = final_carry[0]
    stop_reason_codes = final_carry[6]
    raw = jnp.sum(final_values[:, :-1] <= final_values[:, 1:], axis=1)
    return {
        "final_values": final_values,
        "swap_count": final_carry[2],
        "comparison_count": final_carry[3],
        "activation_count": final_carry[4],
        "stop_reason_codes": stop_reason_codes,
        "final_sortedness_raw_count": raw,
        "final_sortedness_percent": jnp.where(n > 1, 100.0 * raw / (n - 1), 100.0),
        "final_monotonicity_error": jnp.maximum(0, n - 1) - raw,
        "values_trace": jnp.concatenate([initial_values[None, :, :], trace["values"]], axis=0),
        "swapped_trace": trace["swapped"],
    }


_RUN_BATCH_JIT = jax.jit(_run_batch_jax, static_argnames=("max_activations", "max_swaps", "max_comparisons", "no_move_checks_required", "no_move_check_interval")) if jax is not None else None


def run_batch_simulator(
    programs: Sequence[RuleProgram | Mapping[str, Any] | str],
    initial_values: Sequence[Sequence[int]],
    scheduler_seeds: Sequence[int],
    *,
    max_activations: int = 128,
    max_swaps: int = 128,
    max_comparisons: int = 512,
    no_move_checks_required: int = 2,
    no_move_check_interval: int | None = None,
    device=None,
) -> BatchSimulationResult:
    _ensure_jax_available()
    compiled = compile_batch_programs(programs)
    values = np.asarray(initial_values, dtype=np.int32)
    if values.ndim != 2:
        raise ValueError("initial_values must be a two-dimensional array-like")
    if values.shape[0] != compiled.batch_size:
        raise ValueError("program count and initial_values row count must match")
    if len(scheduler_seeds) != compiled.batch_size:
        raise ValueError("scheduler_seeds length must match program count")
    interval = int(no_move_check_interval or values.shape[1])
    schedules = actor_schedules(values.shape[1], max_activations, scheduler_seeds)
    selected_device = preferred_jax_device() if device is None else device
    device_values = _to_device_arrays(compiled, values, schedules, selected_device)
    output = _RUN_BATCH_JIT(
        *device_values,
        max_activations=int(max_activations),
        max_swaps=int(max_swaps),
        max_comparisons=int(max_comparisons),
        no_move_checks_required=int(no_move_checks_required),
        no_move_check_interval=interval,
    )
    materialized = {key: np.asarray(value) for key, value in output.items()}
    return BatchSimulationResult(
        policy_ids=compiled.policy_ids,
        initial_values=values,
        final_values=materialized["final_values"],
        swap_count=materialized["swap_count"],
        comparison_count=materialized["comparison_count"],
        activation_count=materialized["activation_count"],
        stop_reason_codes=materialized["stop_reason_codes"],
        final_sortedness_raw_count=materialized["final_sortedness_raw_count"],
        final_sortedness_percent=materialized["final_sortedness_percent"],
        final_monotonicity_error=materialized["final_monotonicity_error"],
        values_trace=materialized["values_trace"],
        swapped_trace=materialized["swapped_trace"],
        scheduler_seeds=tuple(int(seed) for seed in scheduler_seeds),
        device=str(selected_device),
    )


def run_cpu_reference_case(
    program: RuleProgram | Mapping[str, Any] | str,
    initial_values: Sequence[int],
    scheduler_seed: int,
    *,
    max_activations: int = 128,
    max_swaps: int = 128,
    max_comparisons: int = 512,
    no_move_checks_required: int = 2,
    no_move_check_interval: int | None = None,
) -> dict[str, Any]:
    program = parse_rule_program(program)
    result = PolicyEventSimulator(
        list(initial_values),
        DSLPolicy(program),
        scheduler_seed=int(scheduler_seed),
        tie_breaker_seed=0,
        condition_id=f"S06_cpu_ref_{program.policy_id}_{scheduler_seed}",
        implementation="e03_s06_cpu_reference",
        research_step_id="S06",
    ).run(
        max_activations=int(max_activations),
        max_swaps=int(max_swaps),
        max_comparisons=int(max_comparisons),
        no_move_checks_required=int(no_move_checks_required),
        no_move_check_interval=no_move_check_interval,
    )
    return {
        "policyId": program.policy_id,
        "schedulerSeed": int(scheduler_seed),
        "initialValuesJson": json.dumps(list(initial_values), separators=(",", ":")),
        "finalValuesJson": json.dumps(result.final_values, separators=(",", ":")),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "activationCount": int(result.activation_count),
        "eventCount": int(result.event_count),
        "finalSortednessRawCount": int(result.final_sortedness_raw_count),
        "finalSortednessPercent": float(result.final_sortedness_percent),
        "finalMonotonicityError": int(result.final_monotonicity_error),
        "traceHashSequenceJson": json.dumps([row["state_hash"] for row in result.trace_rows], separators=(",", ":")),
    }


def compare_batch_to_cpu(cpu_rows: Sequence[Mapping[str, Any]], batch_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for cpu_row, batch_row in zip(cpu_rows, batch_rows):
        fields = [
            "finalValuesJson",
            "completed",
            "stopReason",
            "swapCount",
            "comparisonCount",
            "activationCount",
            "eventCount",
            "finalSortednessRawCount",
            "finalMonotonicityError",
            "traceHashSequenceJson",
        ]
        mismatches = [field for field in fields if cpu_row[field] != batch_row[field]]
        percent_match = abs(float(cpu_row["finalSortednessPercent"]) - float(batch_row["finalSortednessPercent"])) < 1e-9
        if not percent_match:
            mismatches.append("finalSortednessPercent")
        comparisons.append(
            {
                "policyId": cpu_row["policyId"],
                "schedulerSeed": int(cpu_row["schedulerSeed"]),
                "initialValuesJson": cpu_row["initialValuesJson"],
                "agreement": not mismatches,
                "mismatchedFieldsJson": json.dumps(mismatches, separators=(",", ":")),
                "cpuStopReason": cpu_row["stopReason"],
                "batchStopReason": batch_row["stopReason"],
                "cpuFinalValuesJson": cpu_row["finalValuesJson"],
                "batchFinalValuesJson": batch_row["finalValuesJson"],
                "cpuSwapCount": int(cpu_row["swapCount"]),
                "batchSwapCount": int(batch_row["swapCount"]),
                "cpuComparisonCount": int(cpu_row["comparisonCount"]),
                "batchComparisonCount": int(batch_row["comparisonCount"]),
                "cpuActivationCount": int(cpu_row["activationCount"]),
                "batchActivationCount": int(batch_row["activationCount"]),
            }
        )
    return comparisons


def summarize_support(reports: Sequence[BatchSupportReport]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for report in reports:
        for reason in report.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "batchSimulatorVersion": BATCH_SIMULATOR_VERSION,
        "policyCount": len(reports),
        "supportedPolicyCount": sum(1 for report in reports if report.supported),
        "unsupportedPolicyCount": sum(1 for report in reports if not report.supported),
        "unsupportedReasonCounts": dict(sorted(reason_counts.items())),
        "supportedPredicates": list(SUPPORTED_PREDICATES),
        "supportedActions": list(SUPPORTED_ACTIONS),
    }
