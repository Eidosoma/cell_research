"""Small auditable rule DSL for E03 local sorting policies."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.e03.policy_interface import ConstraintCheck, PolicyAction, PolicyObservation


TARGETS = ("left", "right", "ideal")
STATUS_ACTIVE = {"ACTIVE", "CellStatus.ACTIVE"}
STATUS_FROZEN = {"FREEZE", "CellStatus.FREEZE"}
STATUS_MOVABLE = STATUS_ACTIVE | STATUS_FROZEN
STATE_INIT_VALUES = ("none", "left_boundary", "right_boundary", "current")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
HEADER_RE = re.compile(r"^policy\s+([A-Za-z_][A-Za-z0-9_-]*)\s+v([0-9]+)$")
STATE_RE = re.compile(r"^state\s+ideal_position=([A-Za-z0-9_-]+)$")
FUNCTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\((.*)\))?$")


class DSLValidationError(ValueError):
    """Raised when a DSL source or object violates the frozen S02 grammar."""


@dataclass(frozen=True)
class DSLCondition:
    """One boolean predicate in a rule guard."""

    name: str
    args: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": list(self.args)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DSLCondition":
        return cls(name=str(data["name"]), args=tuple(str(item) for item in data.get("args", ())))

    def to_source(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(self.args)})"


@dataclass(frozen=True)
class DSLAction:
    """One action primitive in a rule body."""

    name: str
    args: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": list(self.args)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DSLAction":
        return cls(name=str(data["name"]), args=tuple(str(item) for item in data.get("args", ())))

    def to_source(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(self.args)})"


@dataclass(frozen=True)
class DSLRule:
    """One guarded action sequence."""

    conditions: tuple[DSLCondition, ...]
    actions: tuple[DSLAction, ...]
    is_else: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_else": bool(self.is_else),
            "conditions": [condition.to_dict() for condition in self.conditions],
            "actions": [action.to_dict() for action in self.actions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DSLRule":
        return cls(
            conditions=tuple(DSLCondition.from_dict(item) for item in data.get("conditions", ())),
            actions=tuple(DSLAction.from_dict(item) for item in data.get("actions", ())),
            is_else=bool(data.get("is_else", False)),
        )

    def to_source(self) -> str:
        actions = ", ".join(action.to_source() for action in self.actions)
        if self.is_else:
            return f"rule else {actions}"
        conditions = " and ".join(condition.to_source() for condition in self.conditions)
        return f"rule if {conditions} then {actions}"


@dataclass(frozen=True)
class DSLPolicy:
    """A complete DSL policy program."""

    name: str
    version: int
    state_init: str = "none"
    rules: tuple[DSLRule, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": int(self.version),
            "state_init": self.state_init,
            "rules": [rule.to_dict() for rule in self.rules],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DSLPolicy":
        policy = cls(
            name=str(data["name"]),
            version=int(data["version"]),
            state_init=str(data.get("state_init", "none")),
            rules=tuple(DSLRule.from_dict(item) for item in data.get("rules", ())),
            metadata=dict(data.get("metadata", {})),
        )
        validate_policy(policy)
        return policy

    def to_json(self) -> str:
        return stable_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "DSLPolicy":
        return cls.from_dict(json.loads(payload))

    def to_source(self) -> str:
        lines = [f"policy {self.name} v{self.version}", f"state ideal_position={self.state_init}"]
        lines.extend(rule.to_source() for rule in self.rules)
        lines.append("end")
        return "\n".join(lines) + "\n"

    @property
    def sha256(self) -> str:
        return policy_sha256(self)

    @property
    def policy_id(self) -> str:
        return f"dsl:{self.sha256[:16]}"


@dataclass(frozen=True)
class DSLArrayState:
    """Small immutable array snapshot used for reference DSL execution."""

    values: tuple[int, ...]
    labels: tuple[str, ...] | None = None
    statuses: tuple[str, ...] | None = None
    actor_index: int = 0
    reverse_direction: bool = False
    ideal_position: int | None = None
    left_boundary: int = 0
    right_boundary: int | None = None
    actor_label: str = "dsl"
    behavior: str = "dsl"

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("values must not be empty")
        right_boundary = len(self.values) - 1 if self.right_boundary is None else int(self.right_boundary)
        if self.left_boundary < 0 or right_boundary >= len(self.values) or self.left_boundary > right_boundary:
            raise ValueError("invalid boundaries")
        if self.actor_index < self.left_boundary or self.actor_index > right_boundary:
            raise ValueError("actor_index out of bounds")
        labels = self.labels or tuple("dsl" for _ in self.values)
        statuses = self.statuses or tuple("ACTIVE" for _ in self.values)
        if len(labels) != len(self.values) or len(statuses) != len(self.values):
            raise ValueError("labels and statuses must match values length")
        object.__setattr__(self, "values", tuple(int(value) for value in self.values))
        object.__setattr__(self, "labels", tuple(str(label) for label in labels))
        object.__setattr__(self, "statuses", tuple(str(status) for status in statuses))
        object.__setattr__(self, "right_boundary", right_boundary)

    def with_policy_state(self, policy: DSLPolicy) -> "DSLArrayState":
        if self.ideal_position is not None or policy.state_init == "none":
            return self
        if policy.state_init == "left_boundary":
            ideal = self.left_boundary
        elif policy.state_init == "right_boundary":
            ideal = int(self.right_boundary)
        elif policy.state_init == "current":
            ideal = self.actor_index
        else:
            ideal = int(policy.state_init)
        return DSLArrayState(
            values=self.values,
            labels=self.labels,
            statuses=self.statuses,
            actor_index=self.actor_index,
            reverse_direction=self.reverse_direction,
            ideal_position=ideal,
            left_boundary=self.left_boundary,
            right_boundary=self.right_boundary,
            actor_label=self.actor_label,
            behavior=self.behavior,
        )

    def to_observation(self) -> PolicyObservation:
        return PolicyObservation(
            actor_index=int(self.actor_index),
            actor_thread_id=int(self.actor_index + 1),
            actor_value=int(self.values[self.actor_index]),
            actor_label=str(self.labels[self.actor_index]),
            actor_status=str(self.statuses[self.actor_index]),
            behavior=str(self.behavior),
            values=tuple(self.values),
            labels=tuple(self.labels),
            statuses=tuple(self.statuses),
            left_boundary=int(self.left_boundary),
            right_boundary=int(self.right_boundary),
            reverse_direction=bool(self.reverse_direction),
            ideal_position=self.ideal_position,
            group_status="ACTIVE",
        )


@dataclass(frozen=True)
class DSLStepResult:
    """Reference execution result for a single DSL action proposal."""

    policy_id: str
    state_before: DSLArrayState
    action: PolicyAction
    state_after: DSLArrayState


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def policy_sha256(policy: DSLPolicy) -> str:
    return hashlib.sha256(policy.to_json().encode("utf-8")).hexdigest()


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise DSLValidationError(f"Unbalanced parentheses in {text!r}")
        if char == sep and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(char)
    if depth != 0:
        raise DSLValidationError(f"Unbalanced parentheses in {text!r}")
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def parse_call(text: str) -> tuple[str, tuple[str, ...]]:
    text = text.strip()
    aliases = {
        "compare-left": ("compare", ("left",)),
        "compare-right": ("compare", ("right",)),
        "swap-left": ("swap", ("left",)),
        "swap-right": ("swap", ("right",)),
        "wait": ("wait", ()),
    }
    if text in aliases:
        return aliases[text]
    match = FUNCTION_RE.match(text)
    if not match:
        raise DSLValidationError(f"Invalid primitive syntax: {text!r}")
    name = match.group(1).replace("-", "_").lower()
    arg_text = match.group(2)
    args = tuple(split_top_level(arg_text or ""))
    return name, args


def parse_condition(text: str) -> DSLCondition:
    name, args = parse_call(text)
    condition = DSLCondition(name=name, args=args)
    validate_condition(condition)
    return condition


def parse_action(text: str) -> DSLAction:
    name, args = parse_call(text)
    if name == "prob":
        name = "choose"
    action = DSLAction(name=name, args=args)
    validate_action(action)
    return action


def parse_conditions(text: str) -> tuple[DSLCondition, ...]:
    chunks = [chunk.strip() for chunk in re.split(r"\s+and\s+", text.strip()) if chunk.strip()]
    if not chunks:
        raise DSLValidationError("Rule guard must contain at least one condition")
    if len(chunks) == 1 and chunks[0] == "always":
        return (DSLCondition("always"),)
    return tuple(parse_condition(chunk) for chunk in chunks)


def parse_actions(text: str) -> tuple[DSLAction, ...]:
    actions = tuple(parse_action(chunk) for chunk in split_top_level(text))
    if not actions:
        raise DSLValidationError("Rule body must contain at least one action")
    return actions


def parse_policy(source: str) -> DSLPolicy:
    """Parse the S02 line-oriented policy DSL."""

    lines = [strip_comment(line) for line in source.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < 3:
        raise DSLValidationError("Policy source is too short")
    header_match = HEADER_RE.match(lines[0])
    if not header_match:
        raise DSLValidationError("First non-comment line must be 'policy <name> v<version>'")
    name = header_match.group(1)
    version = int(header_match.group(2))
    idx = 1
    state_init = "none"
    state_match = STATE_RE.match(lines[idx])
    if state_match:
        state_init = state_match.group(1)
        idx += 1
    rules: list[DSLRule] = []
    while idx < len(lines):
        line = lines[idx]
        if line == "end":
            if idx != len(lines) - 1:
                raise DSLValidationError("'end' must be the final line")
            policy = DSLPolicy(name=name, version=version, state_init=state_init, rules=tuple(rules))
            validate_policy(policy)
            return policy
        if not line.startswith("rule "):
            raise DSLValidationError(f"Expected rule or end, got {line!r}")
        if line.startswith("rule else "):
            rules.append(DSLRule(conditions=(), actions=parse_actions(line[len("rule else ") :]), is_else=True))
        else:
            match = re.match(r"^rule\s+if\s+(.+)\s+then\s+(.+)$", line)
            if not match:
                raise DSLValidationError(f"Invalid rule syntax: {line!r}")
            rules.append(
                DSLRule(
                    conditions=parse_conditions(match.group(1)),
                    actions=parse_actions(match.group(2)),
                    is_else=False,
                )
            )
        idx += 1
    raise DSLValidationError("Policy source must end with 'end'")


def validate_target(target: str) -> None:
    if target not in TARGETS:
        raise DSLValidationError(f"Unsupported target {target!r}; expected one of {TARGETS}")


def validate_probability(text: str) -> None:
    try:
        probability = float(text)
    except ValueError as exc:
        raise DSLValidationError(f"Invalid probability {text!r}") from exc
    if not 0.0 <= probability <= 1.0:
        raise DSLValidationError(f"Probability must be in [0, 1], got {text!r}")


def validate_condition(condition: DSLCondition) -> None:
    if condition.name == "always":
        if condition.args:
            raise DSLValidationError("always takes no arguments")
        return
    if condition.name in {"target_exists", "target_active", "target_movable", "self_lt", "self_gt", "self_le", "self_ge"}:
        if len(condition.args) != 1:
            raise DSLValidationError(f"{condition.name} takes exactly one target")
        validate_target(condition.args[0])
        return
    if condition.name in {"at_ideal", "not_at_ideal", "prefix_sorted"}:
        if condition.args:
            raise DSLValidationError(f"{condition.name} takes no arguments")
        return
    if condition.name == "random_lt":
        if len(condition.args) != 1:
            raise DSLValidationError("random_lt takes one probability")
        validate_probability(condition.args[0])
        return
    raise DSLValidationError(f"Unsupported condition primitive: {condition.name}")


def validate_action(action: DSLAction) -> None:
    if action.name in {"compare", "swap"}:
        if len(action.args) != 1:
            raise DSLValidationError(f"{action.name} takes exactly one target")
        validate_target(action.args[0])
        return
    if action.name in {"set_ideal", "estimate_target_position"}:
        if len(action.args) != 1:
            raise DSLValidationError(f"{action.name} takes one target-state argument")
        validate_state_value(action.args[0], allow_next=True)
        return
    if action.name == "wait":
        if action.args:
            raise DSLValidationError("wait takes no arguments")
        return
    if action.name in {"remember", "signal"}:
        if not action.args:
            raise DSLValidationError(f"{action.name} takes at least one argument")
        return
    if action.name == "choose":
        if len(action.args) != 3:
            raise DSLValidationError("choose takes probability, true-action, false-action")
        validate_probability(action.args[0])
        parse_action(action.args[1])
        parse_action(action.args[2])
        return
    raise DSLValidationError(f"Unsupported action primitive: {action.name}")


def validate_state_value(value: str, *, allow_next: bool = False) -> None:
    if value in STATE_INIT_VALUES or (allow_next and value == "next"):
        return
    try:
        int(value)
    except ValueError as exc:
        raise DSLValidationError(f"Invalid state value {value!r}") from exc


def validate_policy(policy: DSLPolicy) -> None:
    if not IDENTIFIER_RE.match(policy.name):
        raise DSLValidationError(f"Invalid policy name: {policy.name!r}")
    if policy.version != 1:
        raise DSLValidationError("Only DSL version 1 is supported in S02")
    validate_state_value(policy.state_init)
    if not policy.rules:
        raise DSLValidationError("Policy must contain at least one rule")
    else_seen = False
    for idx, rule in enumerate(policy.rules):
        if rule.is_else:
            if else_seen:
                raise DSLValidationError("Only one else rule is allowed")
            if rule.conditions:
                raise DSLValidationError("Else rule must not have conditions")
            if idx != len(policy.rules) - 1:
                raise DSLValidationError("Else rule must be last")
            else_seen = True
        else:
            if not rule.conditions:
                raise DSLValidationError("Non-else rules must have conditions")
            for condition in rule.conditions:
                validate_condition(condition)
        for action in rule.actions:
            validate_action(action)


def target_index(observation: PolicyObservation, target: str) -> int | None:
    validate_target(target)
    if target == "left":
        return observation.actor_index - 1
    if target == "right":
        return observation.actor_index + 1
    return observation.ideal_position


def target_exists(observation: PolicyObservation, target: str) -> bool:
    idx = target_index(observation, target)
    return idx is not None and observation.left_boundary <= idx <= observation.right_boundary


def target_status(observation: PolicyObservation, target: str) -> str | None:
    idx = target_index(observation, target)
    if idx is None or idx < 0 or idx >= len(observation.statuses):
        return None
    return observation.statuses[idx]


def target_value(observation: PolicyObservation, target: str) -> int | None:
    idx = target_index(observation, target)
    if idx is None or idx < 0 or idx >= len(observation.values):
        return None
    return int(observation.values[idx])


def next_ideal_position(observation: PolicyObservation) -> int | None:
    if observation.ideal_position is None:
        return None
    if observation.reverse_direction:
        return int(observation.ideal_position) - 1
    return int(observation.ideal_position) + 1


def resolve_state_value(observation: PolicyObservation, value: str) -> int | None:
    if value == "none":
        return None
    if value == "left_boundary":
        return int(observation.left_boundary)
    if value == "right_boundary":
        return int(observation.right_boundary)
    if value == "current":
        return int(observation.actor_index)
    if value == "next":
        return next_ideal_position(observation)
    return int(value)


def prefix_sorted(observation: PolicyObservation) -> bool:
    prev = 100000 if observation.reverse_direction else -1
    for idx in range(observation.left_boundary, observation.actor_index):
        if observation.statuses[idx] in STATUS_FROZEN:
            prev = -1
            continue
        value = observation.values[idx]
        if observation.reverse_direction and value > prev:
            return False
        if not observation.reverse_direction and value < prev:
            return False
        prev = value
    return True


def evaluate_condition(
    condition: DSLCondition,
    observation: PolicyObservation,
    rng: random.Random,
) -> ConstraintCheck:
    name = condition.name
    args = condition.args
    passed: bool
    detail = ""
    if name == "always":
        passed = True
    elif name == "target_exists":
        passed = target_exists(observation, args[0])
        detail = f"target={args[0]}"
    elif name == "target_active":
        status = target_status(observation, args[0])
        passed = status in STATUS_ACTIVE
        detail = f"status={status}"
    elif name == "target_movable":
        status = target_status(observation, args[0])
        passed = status in STATUS_MOVABLE
        detail = f"status={status}"
    elif name in {"self_lt", "self_gt", "self_le", "self_ge"}:
        value = target_value(observation, args[0])
        if value is None:
            passed = False
            detail = "missing_target"
        elif name == "self_lt":
            passed = observation.actor_value < value
        elif name == "self_gt":
            passed = observation.actor_value > value
        elif name == "self_le":
            passed = observation.actor_value <= value
        else:
            passed = observation.actor_value >= value
        detail = f"actor={observation.actor_value};target_value={value}"
    elif name == "at_ideal":
        passed = observation.ideal_position is not None and observation.actor_index == observation.ideal_position
        detail = f"ideal={observation.ideal_position}"
    elif name == "not_at_ideal":
        passed = observation.ideal_position is not None and observation.actor_index != observation.ideal_position
        detail = f"ideal={observation.ideal_position}"
    elif name == "prefix_sorted":
        passed = prefix_sorted(observation)
    elif name == "random_lt":
        roll = rng.random()
        threshold = float(args[0])
        passed = roll < threshold
        detail = f"roll={roll:.17g};threshold={threshold:.17g}"
    else:  # pragma: no cover - guarded by validation
        raise DSLValidationError(f"Unsupported condition primitive: {name}")
    return ConstraintCheck(name=condition.to_source(), passed=bool(passed), detail=detail)


class DSLInterpreter:
    """Evaluate S02 DSL policies against S01 observations."""

    def __init__(self, policy: DSLPolicy) -> None:
        validate_policy(policy)
        self.policy = policy

    def propose(self, observation: PolicyObservation, rng: random.Random | None = None) -> PolicyAction:
        rng = rng or random.Random(0)
        for rule in self.policy.rules:
            if rule.is_else:
                return self._run_actions(rule.actions, observation, rng, (ConstraintCheck("else", True),))
            checks = tuple(evaluate_condition(condition, observation, rng) for condition in rule.conditions)
            if all(check.passed for check in checks):
                return self._run_actions(rule.actions, observation, rng, checks)
        return PolicyAction(
            action_type="wait",
            constraints=(ConstraintCheck("no_rule_matched", True),),
            metadata={"policy_id": self.policy.policy_id},
        )

    def _run_actions(
        self,
        actions: Sequence[DSLAction],
        observation: PolicyObservation,
        rng: random.Random,
        constraints: tuple[ConstraintCheck, ...],
    ) -> PolicyAction:
        compare_counted = False
        state_update: dict[str, Any] = {}
        metadata: dict[str, Any] = {"policy_id": self.policy.policy_id, "policy_name": self.policy.name}
        for action in actions:
            if action.name == "compare":
                compare_counted = True
                target = action.args[0]
                constraints = (*constraints, ConstraintCheck(f"compare_target_exists({target})", target_exists(observation, target)))
            elif action.name == "swap":
                target = action.args[0]
                idx = target_index(observation, target)
                action_constraints = (
                    ConstraintCheck(f"swap_target_exists({target})", target_exists(observation, target), f"index={idx}"),
                    ConstraintCheck(
                        f"swap_target_movable({target})",
                        target_status(observation, target) in STATUS_MOVABLE,
                        f"status={target_status(observation, target)}",
                    ),
                )
                all_constraints = (*constraints, *action_constraints)
                if all(check.passed for check in action_constraints):
                    return PolicyAction(
                        action_type="swap",
                        target_index=idx,
                        compare_counted=compare_counted,
                        constraints=all_constraints,
                        state_update=state_update,
                        metadata=metadata | {"terminal_action": action.to_source()},
                    )
                return PolicyAction(
                    action_type="wait",
                    target_index=idx,
                    compare_counted=compare_counted,
                    constraints=all_constraints,
                    state_update=state_update,
                    metadata=metadata | {"terminal_action": action.to_source(), "blocked_action": "swap"},
                )
            elif action.name in {"set_ideal", "estimate_target_position"}:
                new_ideal = resolve_state_value(observation, action.args[0])
                state_update = state_update | {"ideal_position": new_ideal}
                return PolicyAction(
                    action_type="update_state",
                    target_index=observation.ideal_position,
                    compare_counted=compare_counted,
                    constraints=constraints,
                    state_update=state_update,
                    metadata=metadata | {"terminal_action": action.to_source()},
                )
            elif action.name == "remember":
                if len(action.args) == 1:
                    key, value = action.args[0], "true"
                else:
                    key, value = action.args[0], ",".join(action.args[1:])
                state_update = state_update | {f"memory.{key}": value}
            elif action.name == "signal":
                metadata = metadata | {"signal": ",".join(action.args)}
            elif action.name == "choose":
                probability = float(action.args[0])
                roll = rng.random()
                chosen_source = action.args[1] if roll < probability else action.args[2]
                chosen_action = parse_action(chosen_source)
                metadata = metadata | {
                    "choice_probability": probability,
                    "choice_roll": roll,
                    "choice_action": chosen_action.to_source(),
                }
                nested = self._run_actions((chosen_action,), observation, rng, constraints)
                return PolicyAction(
                    action_type=nested.action_type,
                    target_index=nested.target_index,
                    compare_counted=compare_counted or nested.compare_counted,
                    constraints=nested.constraints,
                    state_update=state_update | dict(nested.state_update),
                    metadata=metadata | dict(nested.metadata),
                )
            elif action.name == "wait":
                return PolicyAction(
                    action_type="wait",
                    compare_counted=compare_counted,
                    constraints=constraints,
                    state_update=state_update,
                    metadata=metadata | {"terminal_action": action.to_source()},
                )
            else:  # pragma: no cover - guarded by validation
                raise DSLValidationError(f"Unsupported action primitive: {action.name}")
        return PolicyAction(
            action_type="wait",
            compare_counted=compare_counted,
            constraints=constraints,
            state_update=state_update,
            metadata=metadata | {"terminal_action": "implicit_wait"},
        )

    def step_state(self, state: DSLArrayState, rng: random.Random | None = None) -> DSLStepResult:
        state_before = state.with_policy_state(self.policy)
        action = self.propose(state_before.to_observation(), rng)
        state_after = apply_action_to_state(state_before, action)
        return DSLStepResult(
            policy_id=self.policy.policy_id,
            state_before=state_before,
            action=action,
            state_after=state_after,
        )


def apply_action_to_state(state: DSLArrayState, action: PolicyAction) -> DSLArrayState:
    values = list(state.values)
    labels = list(state.labels or ())
    statuses = list(state.statuses or ())
    actor_index = state.actor_index
    ideal_position = state.ideal_position
    if "ideal_position" in action.state_update:
        ideal_position = None if action.state_update["ideal_position"] is None else int(action.state_update["ideal_position"])
    if action.action_type == "swap" and action.target_index is not None:
        target = int(action.target_index)
        if target < state.left_boundary or target > int(state.right_boundary):
            raise DSLValidationError(f"Cannot swap out-of-bounds target {target}")
        values[actor_index], values[target] = values[target], values[actor_index]
        labels[actor_index], labels[target] = labels[target], labels[actor_index]
        statuses[actor_index], statuses[target] = statuses[target], statuses[actor_index]
        actor_index = target
    return DSLArrayState(
        values=tuple(values),
        labels=tuple(labels),
        statuses=tuple(statuses),
        actor_index=actor_index,
        reverse_direction=state.reverse_direction,
        ideal_position=ideal_position,
        left_boundary=state.left_boundary,
        right_boundary=state.right_boundary,
        actor_label=state.actor_label,
        behavior=state.behavior,
    )


def parse_and_render_round_trip(source: str) -> tuple[DSLPolicy, str, DSLPolicy]:
    first = parse_policy(source)
    rendered = first.to_source()
    second = parse_policy(rendered)
    if first.to_dict() != second.to_dict():
        raise DSLValidationError("Round-trip parse/render changed policy semantics")
    return first, rendered, second


SIMPLE_SWAP_LEFT_POLICY = """policy compare_swap_left v1
state ideal_position=none
rule if target_exists(left) and target_movable(left) and self_lt(left) then compare(left), swap(left)
rule else wait
end
"""


SIMPLE_SWAP_RIGHT_POLICY = """policy compare_swap_right v1
state ideal_position=none
rule if target_exists(right) and target_movable(right) and self_gt(right) then compare(right), swap(right)
rule else wait
end
"""


SELECTION_TARGET_POLICY = """policy selection_target_update v1
state ideal_position=left_boundary
rule if target_exists(ideal) and not_at_ideal and target_movable(ideal) and self_lt(ideal) then compare(ideal), swap(ideal)
rule if target_exists(ideal) and not_at_ideal and target_movable(ideal) and self_ge(ideal) then compare(ideal), set_ideal(next)
rule else wait
end
"""
