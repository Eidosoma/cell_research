"""Small JSON rule DSL for S02 local morphospace policies.

The DSL is deliberately compact and deterministic. Programs are JSON objects
that compile to ``LocalRulePolicy`` instances from the S01 interface. S02 keeps
the interpreter narrow enough to validate and search, while leaving classic
algorithm parameterization to S03.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .policies import LocalObservation, LocalRulePolicy, ProposedAction


DSL_VERSION = "e03_s02_rule_dsl.v1"
ALLOWED_PREDICATES = {
    "always",
    "left_exists",
    "right_exists",
    "target_exists",
    "compare_left",
    "compare_right",
    "compare_target",
    "state_equals",
    "state_compare",
    "position_compare",
}
ALLOWED_ACTIONS = {
    "wait",
    "swap_left",
    "swap_right",
    "swap_target",
    "remember",
    "signal",
}
ALLOWED_OPERATORS = {"<", "<=", ">", ">=", "==", "!="}
ALLOWED_UPDATE_OPS = {"set", "increment", "decrement", "estimate_target_position", "signal"}
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class RuleDslError(ValueError):
    """Raised when a rule DSL program is invalid or non-executable."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def stable_program_id(payload: Mapping[str, Any]) -> str:
    stripped = {key: value for key, value in payload.items() if key != "policy_id"}
    digest = hashlib.sha256(canonical_json(_json_ready(stripped)).encode("utf-8")).hexdigest()[:16]
    return f"dsl_{digest}"


def validate_key(value: Any, field_name: str = "key") -> str:
    key = str(value)
    if not KEY_RE.match(key):
        raise RuleDslError(f"invalid {field_name}: {value!r}")
    return key


def validate_operator(operator: Any) -> str:
    op = str(operator)
    if op not in ALLOWED_OPERATORS:
        raise RuleDslError(f"invalid operator: {operator!r}")
    return op


def compare_values(left: Any, operator: str, right: Any) -> bool:
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    raise RuleDslError(f"invalid operator: {operator!r}")


def expression_value(expr: Any, observation: LocalObservation, state: Mapping[str, Any]) -> Any:
    if not isinstance(expr, Mapping):
        return expr
    kind = expr.get("expr")
    if kind == "const":
        return expr.get("value")
    if kind == "state":
        return state.get(validate_key(expr.get("key", ""), "state key"), expr.get("default", 0))
    if kind == "actor_value":
        return observation.actor.value
    if kind == "actor_position":
        return observation.actor_position
    if kind == "n":
        return observation.n
    if kind == "left_value":
        return None if observation.left is None else observation.left.value
    if kind == "right_value":
        return None if observation.right is None else observation.right.value
    if kind == "target_value":
        return None if observation.target is None else observation.target.value
    if kind == "target_position":
        return observation.target_position
    raise RuleDslError(f"unknown expression: {kind!r}")


@dataclass(frozen=True)
class Predicate:
    op: str
    operator: str | None = None
    value: Any = None
    key: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Predicate":
        op = str(payload.get("op", ""))
        if op not in ALLOWED_PREDICATES:
            raise RuleDslError(f"invalid predicate op: {op!r}")
        operator = validate_operator(payload.get("operator")) if "operator" in payload else None
        key = validate_key(payload["key"]) if "key" in payload else None
        return cls(op=op, operator=operator, value=payload.get("value"), key=key)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op}
        if self.operator is not None:
            result["operator"] = self.operator
        if self.value is not None:
            result["value"] = self.value
        if self.key is not None:
            result["key"] = self.key
        return _json_ready(result)


@dataclass(frozen=True)
class StateUpdate:
    op: str
    key: str | None = None
    value: Any = None
    amount: int = 1
    minimum: int | None = None
    maximum: int | None = None
    direction: str = "increasing"
    channel: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateUpdate":
        op = str(payload.get("op", ""))
        if op not in ALLOWED_UPDATE_OPS:
            raise RuleDslError(f"invalid update op: {op!r}")
        key = validate_key(payload["key"]) if "key" in payload else None
        if op != "signal" and key is None:
            raise RuleDslError(f"update {op!r} requires a key")
        channel = validate_key(payload["channel"], "channel") if "channel" in payload else None
        if op == "signal" and channel is None:
            raise RuleDslError("signal update requires a channel")
        direction = str(payload.get("direction", "increasing"))
        if direction not in {"increasing", "decreasing"}:
            raise RuleDslError(f"invalid target-estimate direction: {direction!r}")
        return cls(
            op=op,
            key=key,
            value=payload.get("value"),
            amount=int(payload.get("amount", 1)),
            minimum=None if "min" not in payload else int(payload["min"]),
            maximum=None if "max" not in payload else int(payload["max"]),
            direction=direction,
            channel=channel,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op}
        if self.key is not None:
            result["key"] = self.key
        if self.value is not None:
            result["value"] = self.value
        if self.op in {"increment", "decrement"}:
            result["amount"] = int(self.amount)
        if self.minimum is not None:
            result["min"] = int(self.minimum)
        if self.maximum is not None:
            result["max"] = int(self.maximum)
        if self.op == "estimate_target_position":
            result["direction"] = self.direction
        if self.channel is not None:
            result["channel"] = self.channel
        return _json_ready(result)


@dataclass(frozen=True)
class ActionSpec:
    action: str
    probability: float = 1.0
    reason: str | None = None
    target_key: str = "target_position"
    updates: tuple[StateUpdate, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | str) -> "ActionSpec":
        if isinstance(payload, str):
            payload = {"action": payload}
        action = str(payload.get("action", ""))
        if action not in ALLOWED_ACTIONS:
            raise RuleDslError(f"invalid action: {action!r}")
        probability = float(payload.get("probability", 1.0))
        if not 0.0 <= probability <= 1.0:
            raise RuleDslError("action probability must be between 0 and 1")
        target_key = validate_key(payload.get("target_key", "target_position"), "target key")
        updates = tuple(StateUpdate.from_dict(update) for update in payload.get("updates", []))
        return cls(
            action=action,
            probability=probability,
            reason=None if payload.get("reason") is None else str(payload.get("reason")),
            target_key=target_key,
            updates=updates,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"action": self.action}
        if self.probability != 1.0:
            result["probability"] = self.probability
        if self.reason is not None:
            result["reason"] = self.reason
        if self.action == "swap_target" and self.target_key != "target_position":
            result["target_key"] = self.target_key
        if self.updates:
            result["updates"] = [update.to_dict() for update in self.updates]
        return _json_ready(result)


@dataclass(frozen=True)
class Rule:
    name: str
    when: tuple[Predicate, ...]
    then: ActionSpec

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], index: int) -> "Rule":
        name = str(payload.get("name", f"rule_{index}"))
        validate_key(name, "rule name")
        if "then" not in payload:
            raise RuleDslError(f"rule {name!r} missing then action")
        when = tuple(Predicate.from_dict(item) for item in payload.get("when", [{"op": "always"}]))
        if not when:
            raise RuleDslError(f"rule {name!r} has no predicates")
        return cls(name=name, when=when, then=ActionSpec.from_dict(payload["then"]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "when": [predicate.to_dict() for predicate in self.when],
            "then": self.then.to_dict(),
        }


@dataclass(frozen=True)
class RuleProgram:
    policy_id: str
    name: str
    rules: tuple[Rule, ...]
    default: ActionSpec = field(default_factory=lambda: ActionSpec("wait", reason="default_wait"))
    initial_state: dict[str, Any] = field(default_factory=dict)
    version: str = DSL_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuleProgram":
        version = str(payload.get("version", DSL_VERSION))
        if version != DSL_VERSION:
            raise RuleDslError(f"unsupported DSL version: {version!r}")
        rules = tuple(Rule.from_dict(rule, index) for index, rule in enumerate(payload.get("rules", [])))
        if not rules:
            raise RuleDslError("rule program requires at least one rule")
        policy_id = str(payload.get("policy_id") or stable_program_id(payload))
        validate_key(policy_id, "policy id")
        name = str(payload.get("name", policy_id))
        default = ActionSpec.from_dict(payload.get("default", {"action": "wait", "reason": "default_wait"}))
        initial_state = dict(payload.get("initial_state", {}))
        for key in initial_state:
            validate_key(key, "initial state key")
        return cls(
            policy_id=policy_id,
            name=name,
            rules=rules,
            default=default,
            initial_state=initial_state,
            version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policy_id": self.policy_id,
            "name": self.name,
            "initial_state": _json_ready(self.initial_state),
            "rules": [rule.to_dict() for rule in self.rules],
            "default": self.default.to_dict(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def pretty(self) -> str:
        lines = [f"policy {self.policy_id}: {self.name}"]
        for rule in self.rules:
            predicates = " and ".join(predicate_to_text(predicate) for predicate in rule.when)
            lines.append(f"  {rule.name}: when {predicates} -> {action_to_text(rule.then)}")
        lines.append(f"  default -> {action_to_text(self.default)}")
        return "\n".join(lines)


def predicate_to_text(predicate: Predicate) -> str:
    if predicate.op in {"always", "left_exists", "right_exists", "target_exists"}:
        return predicate.op
    if predicate.op in {"compare_left", "compare_right", "compare_target"}:
        return f"actor.value {predicate.operator} {predicate.op.removeprefix('compare_')}.value"
    if predicate.op == "state_equals":
        return f"state.{predicate.key} == {predicate.value!r}"
    if predicate.op == "state_compare":
        return f"state.{predicate.key} {predicate.operator} {predicate.value!r}"
    if predicate.op == "position_compare":
        return f"position {predicate.operator} {predicate.value!r}"
    return predicate.op


def action_to_text(action: ActionSpec) -> str:
    text = action.action
    if action.probability != 1.0:
        text += f" p={action.probability:g}"
    if action.updates:
        text += " updates=[" + ", ".join(update.op for update in action.updates) + "]"
    return text


def parse_rule_program(payload: str | Mapping[str, Any] | RuleProgram) -> RuleProgram:
    if isinstance(payload, RuleProgram):
        return payload
    if isinstance(payload, str):
        return RuleProgram.from_dict(json.loads(payload))
    return RuleProgram.from_dict(payload)


def evaluate_predicate(predicate: Predicate, observation: LocalObservation, state: Mapping[str, Any]) -> tuple[bool, int]:
    if predicate.op == "always":
        return True, 0
    if predicate.op == "left_exists":
        return observation.left is not None, 0
    if predicate.op == "right_exists":
        return observation.right is not None, 0
    if predicate.op == "target_exists":
        target_pos = observation.target_position
        return target_pos is not None and 0 <= int(target_pos) < observation.n, 0
    if predicate.op == "compare_left":
        return (
            observation.left is not None
            and compare_values(observation.actor.value, predicate.operator or "==", observation.left.value),
            1 if observation.left is not None else 0,
        )
    if predicate.op == "compare_right":
        return (
            observation.right is not None
            and compare_values(observation.actor.value, predicate.operator or "==", observation.right.value),
            1 if observation.right is not None else 0,
        )
    if predicate.op == "compare_target":
        return (
            observation.target is not None
            and compare_values(observation.actor.value, predicate.operator or "==", observation.target.value),
            1 if observation.target is not None else 0,
        )
    if predicate.op == "state_equals":
        return state.get(predicate.key or "", 0) == predicate.value, 0
    if predicate.op == "state_compare":
        return compare_values(state.get(predicate.key or "", 0), predicate.operator or "==", predicate.value), 0
    if predicate.op == "position_compare":
        return compare_values(observation.actor_position, predicate.operator or "==", predicate.value), 0
    raise RuleDslError(f"unknown predicate op: {predicate.op!r}")


def estimate_target_position(observation: LocalObservation, direction: str) -> int:
    if direction == "increasing":
        return max(0, min(observation.n - 1, int(observation.actor.value) - 1))
    if direction == "decreasing":
        return max(0, min(observation.n - 1, observation.n - int(observation.actor.value)))
    raise RuleDslError(f"invalid target-estimate direction: {direction!r}")


def apply_updates(
    updates: tuple[StateUpdate, ...],
    observation: LocalObservation,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    new_state = dict(state)
    for update in updates:
        if update.op == "set":
            new_state[update.key or ""] = expression_value(update.value, observation, new_state)
        elif update.op == "increment":
            value = int(new_state.get(update.key or "", 0)) + int(update.amount)
            if update.minimum is not None:
                value = max(update.minimum, value)
            if update.maximum is not None:
                value = min(update.maximum, value)
            new_state[update.key or ""] = value
        elif update.op == "decrement":
            value = int(new_state.get(update.key or "", 0)) - int(update.amount)
            if update.minimum is not None:
                value = max(update.minimum, value)
            if update.maximum is not None:
                value = min(update.maximum, value)
            new_state[update.key or ""] = value
        elif update.op == "estimate_target_position":
            new_state[update.key or ""] = estimate_target_position(observation, update.direction)
        elif update.op == "signal":
            new_state[f"signal_{update.channel}"] = expression_value(update.value, observation, new_state)
        else:
            raise RuleDslError(f"unknown update op: {update.op!r}")
    return new_state


class DSLPolicy(LocalRulePolicy):
    """LocalRulePolicy interpreter for RuleProgram objects."""

    family = "dsl"
    algotype = "dsl"
    version = DSL_VERSION

    def __init__(self, program: RuleProgram | Mapping[str, Any] | str) -> None:
        self.program = parse_rule_program(program)
        self.policy_id = self.program.policy_id
        super().__init__(program=self.program.to_dict())

    def to_spec(self):  # type: ignore[override]
        from .policies import PolicySpec

        return PolicySpec(
            policy_id=self.policy_id,
            family=self.family,
            algotype=self.algotype,
            version=self.version,
            parameters={"program": self.program.to_dict()},
        )

    def initial_state(
        self,
        *,
        cell_id: int,
        position: int,
        value: int,
        n: int,
        reverse_direction: bool = False,
    ) -> dict[str, Any]:
        return dict(self.program.initial_state)

    def observe(self, cells, actor_position: int, state: Mapping[str, Any], frozen_variant: str) -> LocalObservation:
        base = super().observe(cells, actor_position, state, frozen_variant)
        target_position = state.get("target_position")
        target = None
        if target_position is not None and 0 <= int(target_position) < len(cells):
            target = cells[int(target_position)].snapshot()
        return LocalObservation(
            actor=base.actor,
            actor_position=base.actor_position,
            n=base.n,
            left=base.left,
            right=base.right,
            left_context=base.left_context,
            target_position=None if target_position is None else int(target_position),
            target=target,
            frozen_variant=frozen_variant,
        )

    def _select_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
    ) -> tuple[ActionSpec, int, str]:
        comparison_delta = 0
        for rule in self.program.rules:
            matched = True
            for predicate in rule.when:
                predicate_ok, predicate_delta = evaluate_predicate(predicate, observation, state)
                comparison_delta += predicate_delta
                if not predicate_ok:
                    matched = False
                    break
            if matched:
                return rule.then, comparison_delta, rule.name
        return self.program.default, comparison_delta, "default"

    def propose_action(
        self,
        observation: LocalObservation,
        state: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        forced_direction: int | None = None,
    ) -> ProposedAction:
        action, comparison_delta, rule_name = self._select_action(observation, state)
        updated_state = apply_updates(action.updates, observation, state)
        state_updates = {key: value for key, value in updated_state.items() if state.get(key) != value}
        if action.probability < 1.0 and rng.random() > action.probability:
            return ProposedAction.wait(
                action.reason or f"{rule_name}_probability_wait",
                comparison_delta=comparison_delta,
                state_updates=state_updates,
            )

        if action.action == "wait":
            return ProposedAction.wait(
                action.reason or f"{rule_name}_wait",
                comparison_delta=comparison_delta,
                state_updates=state_updates,
            )
        if action.action in {"remember", "signal"}:
            return ProposedAction.wait(
                action.reason or action.action,
                comparison_delta=comparison_delta,
                state_updates=state_updates,
            )
        if action.action == "swap_left":
            return ProposedAction.swap(
                observation.actor_position - 1,
                comparison_delta=comparison_delta,
                state_updates=state_updates,
            )
        if action.action == "swap_right":
            return ProposedAction.swap(
                observation.actor_position + 1,
                comparison_delta=comparison_delta,
                state_updates=state_updates,
            )
        if action.action == "swap_target":
            target_position = updated_state.get(action.target_key)
            if target_position is None:
                return ProposedAction.wait(
                    action.reason or "target_missing",
                    comparison_delta=comparison_delta,
                    state_updates=state_updates,
                )
            return ProposedAction.swap(
                int(target_position),
                comparison_delta=comparison_delta,
                state_updates=state_updates,
            )
        raise RuleDslError(f"unknown action: {action.action!r}")

    def legal_action_exists(self, observation: LocalObservation, state: Mapping[str, Any]) -> bool:
        action, _, _ = self._select_action(observation, state)
        if action.action == "swap_left":
            target = observation.left
        elif action.action == "swap_right":
            target = observation.right
        elif action.action == "swap_target":
            target_position = state.get(action.target_key)
            target = observation.target if target_position is not None else None
        else:
            return False
        return target is not None and (observation.frozen_variant != "stuck" or not target.frozen)


def policy_from_dsl_json(payload: str) -> DSLPolicy:
    return DSLPolicy(parse_rule_program(payload))


def local_inversion_program(policy_id: str = "dsl_local_inversion_cleaner") -> RuleProgram:
    """Return a simple executable sorter-like policy for S02 validation."""

    return parse_rule_program(
        {
            "version": DSL_VERSION,
            "policy_id": policy_id,
            "name": "local inversion cleaner",
            "rules": [
                {
                    "name": "move_left_if_smaller",
                    "when": [{"op": "compare_left", "operator": "<"}],
                    "then": {"action": "swap_left"},
                },
                {
                    "name": "move_right_if_larger",
                    "when": [{"op": "compare_right", "operator": ">"}],
                    "then": {"action": "swap_right"},
                },
            ],
            "default": {"action": "wait", "reason": "locally_ordered"},
        }
    )


def null_program(policy_id: str = "dsl_null_wait") -> RuleProgram:
    return parse_rule_program(
        {
            "version": DSL_VERSION,
            "policy_id": policy_id,
            "name": "null wait",
            "rules": [{"name": "always_wait", "when": [{"op": "always"}], "then": {"action": "wait"}}],
            "default": {"action": "wait"},
        }
    )


def stochastic_right_program(policy_id: str = "dsl_stochastic_right_swap") -> RuleProgram:
    return parse_rule_program(
        {
            "version": DSL_VERSION,
            "policy_id": policy_id,
            "name": "stochastic right swap",
            "rules": [
                {
                    "name": "maybe_swap_right",
                    "when": [{"op": "right_exists"}],
                    "then": {"action": "swap_right", "probability": 0.5},
                }
            ],
            "default": {"action": "wait"},
        }
    )
