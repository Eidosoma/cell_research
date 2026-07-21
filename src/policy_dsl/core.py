"""Safe compiler and total interpreter for the E07 compact policy DSL.

The language is deliberately a small JSON abstract syntax tree.  Compilation
performs all name, type, permission, shape, and worst-case operation checks.
Execution walks immutable bounded structures directly; it never invokes Python
``eval``/``exec``, imports policy-selected code, or performs I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence


DSL_VERSION = "e07.policy-dsl.v1"
HASH_DOMAIN = b"E07/S01/policy-dsl/v1\x00"
BASELINE_DIRECTORY = Path(__file__).with_name("baselines")

ENVIRONMENTS = {"line1d.v1", "spatial2d.v1"}
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
POLICY_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
COMPARATORS = {"eq", "ne", "lt", "le", "gt", "ge"}
ORDERED_COMPARATORS = {"lt", "le", "gt", "ge"}
MOVEMENT_KINDS = {
    "adjacent_swap",
    "vacancy_move",
    "short_exchange",
    "rotation",
}
CANDIDATE_FIELDS = {
    "candidate.local_relation_delta": "int",
    "candidate.natural_boundary_delta": "int",
    "candidate.gradient_delta": "int",
    "candidate.lagged_conflict_count": "int",
    "candidate.movement_cost": "int",
    "candidate.kind": "symbol",
}

OBSERVATION_TYPES: dict[str, dict[str, str]] = {
    "line1d.v1": {
        "activation.side": "symbol",
        "own.value": "int",
        "own.position": "int",
        "own.direction": "symbol",
        "neighbor.left.exists": "bool",
        "neighbor.left.value": "int",
        "neighbor.left.movable": "bool",
        "neighbor.right.exists": "bool",
        "neighbor.right.value": "int",
        "neighbor.right.movable": "bool",
        "line.prefix_ordered": "bool",
        "selection.cursor_in_bounds": "bool",
        "selection.cursor_at_actor": "bool",
        "selection.target.value": "int",
        "selection.target.stuck": "bool",
        "last_action.rejected": "bool",
        "repair.nudge_count": "int",
        "signal.neighbor_sum_u8": "int",
        "counter.choice_u8": "int",
    },
    "spatial2d.v1": {
        "own.token": "symbol",
        "own.local_relation_utility": "int",
        "natural.boundary_signal": "int",
        "gradient.current_u8": "int",
        "candidate.count": "int",
        "last_action.rejected": "bool",
        "signal.neighbor_sum_u8": "int",
        "counter.choice_u8": "int",
        **CANDIDATE_FIELDS,
    },
}

ACTION_ENVIRONMENTS = {
    "noop": ENVIRONMENTS,
    "set_memory": ENVIRONMENTS,
    "emit_signal": ENVIRONMENTS,
    "swap_relative": {"line1d.v1"},
    "swap_cursor": {"line1d.v1"},
    "advance_cursor": {"line1d.v1"},
    "move_candidate": {"spatial2d.v1"},
}

FORBIDDEN_TOKENS = {
    "analysislabel",
    "completion",
    "conflictpriority",
    "futurestate",
    "global",
    "holdout",
    "occupantid",
    "outcome",
    "proposalid",
    "route",
    "scenarioid",
    "siteid",
    "splitlabel",
    "statesha256",
    "targetpattern",
    "targetsite",
}


class PolicyValidationError(ValueError):
    """Raised for any malformed, unsafe, or out-of-contract policy input."""


@dataclass(frozen=True, slots=True)
class MemorySpec:
    name: str
    bits: int
    initial: int


@dataclass(frozen=True, slots=True)
class Complexity:
    rule_count: int
    branch_count: int
    expression_nodes: int
    action_count: int
    persistent_memory_bits: int
    outbound_signal_bits: int
    max_movement_radius: int
    max_candidates: int
    worst_case_operations: int
    canonical_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "ruleCount": self.rule_count,
            "branchCount": self.branch_count,
            "expressionNodes": self.expression_nodes,
            "actionCount": self.action_count,
            "persistentMemoryBits": self.persistent_memory_bits,
            "outboundSignalBits": self.outbound_signal_bits,
            "maxMovementRadius": self.max_movement_radius,
            "maxCandidates": self.max_candidates,
            "worstCaseOperations": self.worst_case_operations,
            "canonicalBytes": self.canonical_bytes,
        }


@dataclass(frozen=True, slots=True)
class CompiledPolicy:
    policy_id: str
    environment: str
    permissions: frozenset[str]
    memory: tuple[MemorySpec, ...]
    signal_channels: int
    signal_bits_per_channel: int
    max_operations: int
    max_candidates: int
    max_movement_radius: int
    rules: tuple[tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]], ...]
    default_actions: tuple[Mapping[str, Any], ...]
    canonical_json: bytes
    policy_sha256: str
    complexity: Complexity


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    policy_id: str
    policy_sha256: str
    matched_rule: int | None
    actions: tuple[Mapping[str, Any], ...]
    memory: Mapping[str, int]
    emitted_signals: Mapping[int, int]
    operation_count: int


def _reject_constant(value: str) -> None:
    raise PolicyValidationError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_loads(raw: str | bytes) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except PolicyValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise PolicyValidationError(f"invalid JSON: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PolicyValidationError(
            f"policy is not canonical JSON data: {exc}"
        ) from exc


def _expect_keys(value: Any, required: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyValidationError(f"{context} must be an object")
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise PolicyValidationError(
            f"{context} keys mismatch; missing={missing}, extra={extra}"
        )
    if any(not isinstance(key, str) for key in value):
        raise PolicyValidationError(f"{context} keys must be strings")
    return value


def _int(value: Any, minimum: int, maximum: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyValidationError(f"{context} must be an integer")
    if not minimum <= value <= maximum:
        raise PolicyValidationError(f"{context} must be in [{minimum}, {maximum}]")
    return value


def _literal_type(value: Any, context: str) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        if not -(2**31) <= value <= 2**31 - 1:
            raise PolicyValidationError(f"{context} integer is outside int32")
        return "int"
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 64:
            raise PolicyValidationError(f"{context} string exceeds 64 UTF-8 bytes")
        return "symbol"
    raise PolicyValidationError(f"{context} must be bool, int32, or string")


def _normalize_forbidden(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _validate_name(name: Any, pattern: re.Pattern[str], context: str) -> str:
    if not isinstance(name, str) or not pattern.fullmatch(name):
        raise PolicyValidationError(f"invalid {context}: {name!r}")
    normalized = _normalize_forbidden(name)
    if any(token in normalized for token in FORBIDDEN_TOKENS):
        raise PolicyValidationError(f"forbidden {context}: {name}")
    return name


def _memory_map(memory: Sequence[MemorySpec]) -> dict[str, MemorySpec]:
    return {item.name: item for item in memory}


def _validate_operand(
    value: Any,
    *,
    environment: str,
    permissions: frozenset[str],
    memories: Mapping[str, MemorySpec],
    context: str,
) -> str:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise PolicyValidationError(f"{context} operand must have exactly one source")
    source, payload = next(iter(value.items()))
    if source == "const":
        return _literal_type(payload, context)
    if source == "obs":
        if not isinstance(payload, str) or payload not in permissions:
            raise PolicyValidationError(
                f"{context} uses undeclared observation permission: {payload!r}"
            )
        if payload.startswith("candidate."):
            raise PolicyValidationError(
                f"{context} cannot use candidate-vector fields as scalar operands"
            )
        return OBSERVATION_TYPES[environment][payload]
    if source == "mem":
        if not isinstance(payload, str) or payload not in memories:
            raise PolicyValidationError(f"{context} uses unknown memory: {payload!r}")
        return "int"
    raise PolicyValidationError(f"{context} has unknown operand source: {source!r}")


def _validate_expression(
    expression: Any,
    *,
    environment: str,
    permissions: frozenset[str],
    memories: Mapping[str, MemorySpec],
    context: str,
) -> tuple[int, int]:
    if not isinstance(expression, Mapping) or len(expression) != 1:
        if isinstance(expression, Mapping) and set(expression) == {
            "op",
            "left",
            "right",
        }:
            pass
        else:
            raise PolicyValidationError(f"{context} must contain one expression form")
    keys = set(expression)
    if keys == {"const"}:
        if not isinstance(expression["const"], bool):
            raise PolicyValidationError(f"{context} condition constant must be boolean")
        return 1, 0
    if keys in ({"all"}, {"any"}):
        operator = next(iter(keys))
        children = expression[operator]
        if not isinstance(children, (list, tuple)) or not 1 <= len(children) <= 8:
            raise PolicyValidationError(f"{context}.{operator} must have 1-8 children")
        nodes = 1
        branches = 1
        for index, child in enumerate(children):
            child_nodes, child_branches = _validate_expression(
                child,
                environment=environment,
                permissions=permissions,
                memories=memories,
                context=f"{context}.{operator}[{index}]",
            )
            nodes += child_nodes
            branches += child_branches
        return nodes, branches
    if keys == {"not"}:
        nodes, branches = _validate_expression(
            expression["not"],
            environment=environment,
            permissions=permissions,
            memories=memories,
            context=f"{context}.not",
        )
        return nodes + 1, branches + 1
    if keys == {"op", "left", "right"}:
        operator = expression["op"]
        if operator not in COMPARATORS:
            raise PolicyValidationError(
                f"{context} has invalid comparator: {operator!r}"
            )
        left = _validate_operand(
            expression["left"],
            environment=environment,
            permissions=permissions,
            memories=memories,
            context=f"{context}.left",
        )
        right = _validate_operand(
            expression["right"],
            environment=environment,
            permissions=permissions,
            memories=memories,
            context=f"{context}.right",
        )
        if left != right:
            raise PolicyValidationError(
                f"{context} compares incompatible types: {left} and {right}"
            )
        if operator in ORDERED_COMPARATORS and left not in {"int", "symbol"}:
            raise PolicyValidationError(
                f"{context} ordered comparator requires int or symbol operands"
            )
        return 1, 1
    raise PolicyValidationError(
        f"{context} has unknown expression form: {sorted(keys)}"
    )


def _action_cost(action: Mapping[str, Any], max_candidates: int) -> int:
    return 1 + (max_candidates if action["kind"] == "move_candidate" else 0)


def _validate_action(
    action: Any,
    *,
    environment: str,
    permissions: frozenset[str],
    memories: Mapping[str, MemorySpec],
    signal_channels: int,
    signal_bits: int,
    max_movement_radius: int,
    context: str,
) -> None:
    if not isinstance(action, Mapping) or "kind" not in action:
        raise PolicyValidationError(f"{context} must be an action object")
    kind = action["kind"]
    if kind not in ACTION_ENVIRONMENTS:
        raise PolicyValidationError(f"{context} has unsupported action kind: {kind!r}")
    if environment not in ACTION_ENVIRONMENTS[kind]:
        raise PolicyValidationError(
            f"{context} action {kind} is invalid for {environment}"
        )

    if kind == "noop":
        _expect_keys(action, {"kind"}, context)
    elif kind == "swap_relative":
        _expect_keys(action, {"kind", "offset"}, context)
        offset = _int(action["offset"], -1, 1, f"{context}.offset")
        if offset == 0 or abs(offset) > max_movement_radius:
            raise PolicyValidationError(f"{context} exceeds movement radius")
    elif kind == "swap_cursor":
        _expect_keys(action, {"kind"}, context)
        if max_movement_radius < 2:
            raise PolicyValidationError(
                f"{context} requires the licensed cursor range declaration"
            )
    elif kind == "advance_cursor":
        _expect_keys(action, {"kind", "direction"}, context)
        if action["direction"] != "own_direction":
            raise PolicyValidationError(f"{context}.direction must be own_direction")
        if "own.direction" not in permissions:
            raise PolicyValidationError(f"{context} requires own.direction permission")
    elif kind == "set_memory":
        _expect_keys(action, {"kind", "register", "mode", "value"}, context)
        register = action["register"]
        if register not in memories:
            raise PolicyValidationError(
                f"{context} uses unknown register: {register!r}"
            )
        mode = action["mode"]
        if mode not in {
            "assign",
            "increment_saturating",
            "decrement_saturating",
            "toggle",
        }:
            raise PolicyValidationError(f"{context} has invalid memory mode: {mode!r}")
        if mode == "assign":
            value_type = _validate_operand(
                action["value"],
                environment=environment,
                permissions=permissions,
                memories=memories,
                context=f"{context}.value",
            )
            if value_type != "int":
                raise PolicyValidationError(f"{context}.value must be integer")
        elif action["value"] is not None:
            raise PolicyValidationError(f"{context}.value must be null for {mode}")
        if mode == "toggle" and memories[register].bits != 1:
            raise PolicyValidationError(f"{context} toggle requires a one-bit register")
    elif kind == "emit_signal":
        _expect_keys(action, {"kind", "channel", "value"}, context)
        if signal_channels == 0:
            raise PolicyValidationError(
                f"{context} emits from a policy with no channels"
            )
        _int(action["channel"], 0, signal_channels - 1, f"{context}.channel")
        value_type = _validate_operand(
            action["value"],
            environment=environment,
            permissions=permissions,
            memories=memories,
            context=f"{context}.value",
        )
        if value_type != "int":
            raise PolicyValidationError(f"{context}.value must be integer")
        if signal_bits <= 0:
            raise PolicyValidationError(f"{context} signal width must be positive")
    elif kind == "move_candidate":
        _expect_keys(
            action,
            {"kind", "selector", "allowedMovementKinds"},
            context,
        )
        selector = _expect_keys(
            action["selector"],
            {"mode", "field", "requirePositive", "indexObservation"},
            f"{context}.selector",
        )
        if selector["mode"] not in {"argmax", "argmin", "index"}:
            raise PolicyValidationError(f"{context} has invalid selector mode")
        field = selector["field"]
        if selector["mode"] in {"argmax", "argmin"}:
            if field not in CANDIDATE_FIELDS or field not in permissions:
                raise PolicyValidationError(
                    f"{context} selector field lacks permission: {field!r}"
                )
            if CANDIDATE_FIELDS[field] != "int":
                raise PolicyValidationError(f"{context} selector field must be integer")
            if selector["indexObservation"] is not None:
                raise PolicyValidationError(
                    f"{context} arg selector indexObservation must be null"
                )
        else:
            if field is not None or selector["requirePositive"] is not False:
                raise PolicyValidationError(
                    f"{context} index selector has incompatible field/threshold"
                )
            index_obs = selector["indexObservation"]
            if (
                index_obs not in permissions
                or OBSERVATION_TYPES[environment].get(index_obs) != "int"
            ):
                raise PolicyValidationError(
                    f"{context} index selector lacks integer observation permission"
                )
        if not isinstance(selector["requirePositive"], bool):
            raise PolicyValidationError(f"{context}.requirePositive must be boolean")
        kinds = action["allowedMovementKinds"]
        if (
            not isinstance(kinds, list)
            or not kinds
            or len(kinds) != len(set(kinds))
            or not set(kinds) <= MOVEMENT_KINDS
        ):
            raise PolicyValidationError(
                f"{context} has invalid movement-kind allowlist"
            )
        required_radius = max(2 if kind == "short_exchange" else 1 for kind in kinds)
        if "rotation" in kinds:
            required_radius = max(required_radius, 1)
        if required_radius > max_movement_radius:
            raise PolicyValidationError(f"{context} exceeds movement radius")


def parse_policy(source: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Parse and fully validate a policy, returning detached JSON data."""

    if isinstance(source, Mapping):
        raw = _json_loads(_canonical_bytes(source))
    elif isinstance(source, (str, bytes)):
        raw = _json_loads(source)
    else:
        raise PolicyValidationError(
            "policy source must be JSON text/bytes or a mapping"
        )

    document = _expect_keys(
        raw,
        {
            "schemaVersion",
            "policyId",
            "environment",
            "permissions",
            "memory",
            "signals",
            "limits",
            "rules",
            "default",
        },
        "policy",
    )
    if document["schemaVersion"] != DSL_VERSION:
        raise PolicyValidationError("unsupported policy schemaVersion")
    _validate_name(document["policyId"], POLICY_ID, "policyId")
    environment = document["environment"]
    if environment not in ENVIRONMENTS:
        raise PolicyValidationError(f"unsupported environment: {environment!r}")

    permissions_raw = document["permissions"]
    if (
        not isinstance(permissions_raw, list)
        or len(permissions_raw) != len(set(permissions_raw))
        or not all(isinstance(item, str) for item in permissions_raw)
    ):
        raise PolicyValidationError("permissions must be a unique string list")
    if permissions_raw != sorted(permissions_raw):
        raise PolicyValidationError("permissions must be lexicographically sorted")
    allowed_observations = OBSERVATION_TYPES[environment]
    unknown = sorted(set(permissions_raw) - set(allowed_observations))
    if unknown:
        raise PolicyValidationError(f"unknown or forbidden observations: {unknown}")
    for permission in permissions_raw:
        normalized = _normalize_forbidden(permission)
        if any(token in normalized for token in FORBIDDEN_TOKENS):
            raise PolicyValidationError(
                f"forbidden observation permission: {permission}"
            )
    permissions = frozenset(permissions_raw)

    memory_raw = document["memory"]
    if not isinstance(memory_raw, list) or len(memory_raw) > 8:
        raise PolicyValidationError("memory must be a list of at most eight registers")
    memory: list[MemorySpec] = []
    for index, raw_register in enumerate(memory_raw):
        register = _expect_keys(
            raw_register, {"name", "bits", "initial"}, f"memory[{index}]"
        )
        name = _validate_name(register["name"], IDENTIFIER, f"memory[{index}].name")
        bits = _int(register["bits"], 1, 16, f"memory[{index}].bits")
        initial = _int(register["initial"], 0, 2**bits - 1, f"memory[{index}].initial")
        memory.append(MemorySpec(name, bits, initial))
    if [item.name for item in memory] != sorted(item.name for item in memory):
        raise PolicyValidationError("memory registers must be sorted by name")
    if len({item.name for item in memory}) != len(memory):
        raise PolicyValidationError("memory register names must be unique")
    if sum(item.bits for item in memory) > 32:
        raise PolicyValidationError("persistent memory exceeds 32 bits")
    memories = _memory_map(memory)

    signals = _expect_keys(
        document["signals"], {"channels", "bitsPerChannel"}, "signals"
    )
    signal_channels = _int(signals["channels"], 0, 4, "signals.channels")
    signal_bits = _int(signals["bitsPerChannel"], 0, 8, "signals.bitsPerChannel")
    if (signal_channels == 0) != (signal_bits == 0):
        raise PolicyValidationError(
            "zero signal channels and width must be declared together"
        )
    if signal_channels * signal_bits > 32:
        raise PolicyValidationError("outbound signal state exceeds 32 bits")

    limits = _expect_keys(
        document["limits"],
        {
            "maxRules",
            "maxExpressionNodes",
            "maxActionsPerActivation",
            "maxOperationsPerActivation",
            "maxMovementRadius",
            "maxCandidates",
        },
        "limits",
    )
    max_rules = _int(limits["maxRules"], 1, 32, "limits.maxRules")
    max_expression_nodes = _int(
        limits["maxExpressionNodes"], 1, 128, "limits.maxExpressionNodes"
    )
    max_actions = _int(
        limits["maxActionsPerActivation"], 1, 4, "limits.maxActionsPerActivation"
    )
    max_operations = _int(
        limits["maxOperationsPerActivation"],
        1,
        256,
        "limits.maxOperationsPerActivation",
    )
    max_movement_radius = _int(
        limits["maxMovementRadius"], 0, 65_535, "limits.maxMovementRadius"
    )
    max_candidates = _int(limits["maxCandidates"], 0, 16, "limits.maxCandidates")
    if environment == "spatial2d.v1" and max_candidates == 0:
        raise PolicyValidationError(
            "spatial2d policies require a positive candidate bound"
        )
    if environment == "line1d.v1" and max_candidates != 0:
        raise PolicyValidationError("line1d policies must set maxCandidates to zero")

    rules = document["rules"]
    if not isinstance(rules, list) or not 1 <= len(rules) <= max_rules:
        raise PolicyValidationError("rules must be a nonempty bounded list")
    expression_nodes = 0
    branch_count = 0
    action_count = 0
    cumulative_condition_cost = 0
    worst_case_operations = 0
    for rule_index, rule_raw in enumerate(rules):
        rule = _expect_keys(rule_raw, {"when", "actions"}, f"rules[{rule_index}]")
        nodes, branches = _validate_expression(
            rule["when"],
            environment=environment,
            permissions=permissions,
            memories=memories,
            context=f"rules[{rule_index}].when",
        )
        if nodes > max_expression_nodes:
            raise PolicyValidationError(
                f"rules[{rule_index}] exceeds expression-node limit"
            )
        expression_nodes += nodes
        branch_count += branches
        cumulative_condition_cost += nodes
        actions = rule["actions"]
        if not isinstance(actions, list) or not 1 <= len(actions) <= max_actions:
            raise PolicyValidationError(
                f"rules[{rule_index}].actions exceeds action bound"
            )
        for action_index, action in enumerate(actions):
            _validate_action(
                action,
                environment=environment,
                permissions=permissions,
                memories=memories,
                signal_channels=signal_channels,
                signal_bits=signal_bits,
                max_movement_radius=max_movement_radius,
                context=f"rules[{rule_index}].actions[{action_index}]",
            )
        action_count += len(actions)
        worst_case_operations = max(
            worst_case_operations,
            cumulative_condition_cost
            + sum(_action_cost(action, max_candidates) for action in actions),
        )

    default = _expect_keys(document["default"], {"actions"}, "default")
    default_actions = default["actions"]
    if (
        not isinstance(default_actions, list)
        or not 1 <= len(default_actions) <= max_actions
    ):
        raise PolicyValidationError("default.actions exceeds action bound")
    for action_index, action in enumerate(default_actions):
        _validate_action(
            action,
            environment=environment,
            permissions=permissions,
            memories=memories,
            signal_channels=signal_channels,
            signal_bits=signal_bits,
            max_movement_radius=max_movement_radius,
            context=f"default.actions[{action_index}]",
        )
    action_count += len(default_actions)
    worst_case_operations = max(
        worst_case_operations,
        cumulative_condition_cost
        + sum(_action_cost(action, max_candidates) for action in default_actions),
    )
    if worst_case_operations > max_operations:
        raise PolicyValidationError(
            f"static worst-case operation count {worst_case_operations} exceeds budget {max_operations}"
        )

    movement_actions = [
        action
        for rule in rules
        for action in rule["actions"]
        if action["kind"] in {"swap_relative", "swap_cursor", "move_candidate"}
    ] + [
        action
        for action in default_actions
        if action["kind"] in {"swap_relative", "swap_cursor", "move_candidate"}
    ]
    for action_list in [rule["actions"] for rule in rules] + [default_actions]:
        if (
            sum(
                action["kind"]
                in {
                    "noop",
                    "swap_relative",
                    "swap_cursor",
                    "advance_cursor",
                    "move_candidate",
                }
                for action in action_list
            )
            != 1
        ):
            raise PolicyValidationError(
                "each action bundle must contain exactly one terminal movement/noop action"
            )
    if not movement_actions and max_movement_radius != 0:
        raise PolicyValidationError(
            "nonmoving policy must declare zero movement radius"
        )

    # Return a detached JSON-only copy after validation.
    return _json_loads(_canonical_bytes(document))


def canonical_policy_bytes(source: str | bytes | Mapping[str, Any]) -> bytes:
    return _canonical_bytes(parse_policy(source))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def compile_policy(source: str | bytes | Mapping[str, Any]) -> CompiledPolicy:
    """Validate and deterministically compile a policy into immutable IR."""

    document = parse_policy(source)
    canonical = _canonical_bytes(document)
    digest = hashlib.sha256(HASH_DOMAIN + canonical).hexdigest()
    limits = document["limits"]
    memory = tuple(MemorySpec(**item) for item in document["memory"])
    rules = tuple(
        (_freeze(rule["when"]), tuple(_freeze(action) for action in rule["actions"]))
        for rule in document["rules"]
    )
    default_actions = tuple(
        _freeze(action) for action in document["default"]["actions"]
    )

    expression_nodes = 0
    branch_count = 0
    cumulative = 0
    worst = 0
    for expression, actions in rules:
        nodes, branches = _validate_expression(
            expression,
            environment=document["environment"],
            permissions=frozenset(document["permissions"]),
            memories=_memory_map(memory),
            context="compiled",
        )
        expression_nodes += nodes
        branch_count += branches
        cumulative += nodes
        worst = max(
            worst,
            cumulative
            + sum(_action_cost(action, limits["maxCandidates"]) for action in actions),
        )
    worst = max(
        worst,
        cumulative
        + sum(
            _action_cost(action, limits["maxCandidates"]) for action in default_actions
        ),
    )
    action_count = sum(len(actions) for _, actions in rules) + len(default_actions)
    complexity = Complexity(
        rule_count=len(rules),
        branch_count=branch_count,
        expression_nodes=expression_nodes,
        action_count=action_count,
        persistent_memory_bits=sum(item.bits for item in memory),
        outbound_signal_bits=document["signals"]["channels"]
        * document["signals"]["bitsPerChannel"],
        max_movement_radius=limits["maxMovementRadius"],
        max_candidates=limits["maxCandidates"],
        worst_case_operations=worst,
        canonical_bytes=len(canonical),
    )
    return CompiledPolicy(
        policy_id=document["policyId"],
        environment=document["environment"],
        permissions=frozenset(document["permissions"]),
        memory=memory,
        signal_channels=document["signals"]["channels"],
        signal_bits_per_channel=document["signals"]["bitsPerChannel"],
        max_operations=limits["maxOperationsPerActivation"],
        max_candidates=limits["maxCandidates"],
        max_movement_radius=limits["maxMovementRadius"],
        rules=rules,
        default_actions=default_actions,
        canonical_json=canonical,
        policy_sha256=digest,
        complexity=complexity,
    )


def load_policy(path: str | Path) -> CompiledPolicy:
    return compile_policy(Path(path).read_bytes())


def policy_to_dict(policy: CompiledPolicy) -> dict[str, Any]:
    return _json_loads(policy.canonical_json)


def _validate_scalar(value: Any, expected: str, context: str) -> None:
    actual = _literal_type(value, context)
    if actual != expected:
        raise PolicyValidationError(f"{context} expected {expected}, got {actual}")


def validate_observation(
    policy: CompiledPolicy, observation: Mapping[str, Any]
) -> None:
    envelope = _expect_keys(observation, {"scalars", "candidates"}, "observation")
    scalars = envelope["scalars"]
    candidates = envelope["candidates"]
    if not isinstance(scalars, Mapping):
        raise PolicyValidationError("observation.scalars must be an object")
    scalar_permissions = {
        item for item in policy.permissions if not item.startswith("candidate.")
    }
    if set(scalars) != scalar_permissions:
        raise PolicyValidationError(
            "observation scalar keys must exactly equal the declared permissions"
        )
    for name, value in scalars.items():
        _validate_scalar(
            value, OBSERVATION_TYPES[policy.environment][name], f"observation.{name}"
        )
        if name.endswith("_u8") and not 0 <= value <= 255:
            raise PolicyValidationError(f"observation.{name} is outside uint8")
        if name == "own.direction" and value not in {"ascending", "descending"}:
            raise PolicyValidationError("own.direction has an invalid symbol")
        if name == "activation.side" and value not in {"left", "right", "none"}:
            raise PolicyValidationError("activation.side has an invalid symbol")
    if not isinstance(candidates, list) or len(candidates) > policy.max_candidates:
        raise PolicyValidationError("observation candidates exceed the declared bound")
    if "candidate.count" in scalars and scalars["candidate.count"] != len(candidates):
        raise PolicyValidationError(
            "candidate.count does not match the candidate vector"
        )
    candidate_permissions = sorted(
        item for item in policy.permissions if item.startswith("candidate.")
    )
    if candidates and not candidate_permissions:
        raise PolicyValidationError("candidate data supplied without permission")
    seen_keys: set[str] = set()
    expected_keys = {"key", *candidate_permissions}
    for index, candidate in enumerate(candidates):
        item = _expect_keys(
            candidate, expected_keys, f"observation.candidates[{index}]"
        )
        key = item["key"]
        if not isinstance(key, str) or not key or len(key.encode("utf-8")) > 96:
            raise PolicyValidationError(
                "candidate key must be a bounded nonempty opaque string"
            )
        if key in seen_keys:
            raise PolicyValidationError("candidate keys must be unique")
        seen_keys.add(key)
        for field in candidate_permissions:
            _validate_scalar(item[field], CANDIDATE_FIELDS[field], f"candidate.{field}")
            if field == "candidate.kind" and item[field] not in MOVEMENT_KINDS:
                raise PolicyValidationError("candidate movement kind is unsupported")


def _operand_value(
    operand: Mapping[str, Any], scalars: Mapping[str, Any], memory: Mapping[str, int]
) -> Any:
    source, payload = next(iter(operand.items()))
    if source == "const":
        return payload
    if source == "obs":
        return scalars[payload]
    return memory[payload]


def _eval_expression(
    expression: Mapping[str, Any],
    scalars: Mapping[str, Any],
    memory: Mapping[str, int],
) -> tuple[bool, int]:
    keys = set(expression)
    if keys == {"const"}:
        return bool(expression["const"]), 1
    if keys == {"not"}:
        value, operations = _eval_expression(expression["not"], scalars, memory)
        return not value, operations + 1
    if keys == {"all"}:
        operations = 1
        for child in expression["all"]:
            value, child_operations = _eval_expression(child, scalars, memory)
            operations += child_operations
            if not value:
                return False, operations
        return True, operations
    if keys == {"any"}:
        operations = 1
        for child in expression["any"]:
            value, child_operations = _eval_expression(child, scalars, memory)
            operations += child_operations
            if value:
                return True, operations
        return False, operations
    left = _operand_value(expression["left"], scalars, memory)
    right = _operand_value(expression["right"], scalars, memory)
    operator = expression["op"]
    result = {
        "eq": left == right,
        "ne": left != right,
        "lt": left < right,
        "le": left <= right,
        "gt": left > right,
        "ge": left >= right,
    }[operator]
    return result, 1


def _select_candidate(
    action: Mapping[str, Any],
    scalars: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], int]:
    allowed = set(action["allowedMovementKinds"])
    eligible = [item for item in candidates if item.get("candidate.kind") in allowed]
    selector = action["selector"]
    if not eligible:
        return MappingProxyType(
            {"kind": "noop", "reason": "no_eligible_candidate"}
        ), len(candidates)
    if selector["mode"] == "index":
        index = int(scalars[selector["indexObservation"]]) % len(eligible)
        selected = eligible[index]
        score = None
    else:
        field = selector["field"]
        reverse = selector["mode"] == "argmax"
        selected = sorted(
            eligible,
            key=lambda item: (
                -int(item[field]) if reverse else int(item[field]),
                str(item["key"]),
            ),
        )[0]
        score = int(selected[field])
        if selector["requirePositive"] and score <= 0:
            return MappingProxyType(
                {"kind": "noop", "reason": "selector_threshold"}
            ), len(candidates)
    return MappingProxyType(
        {
            "kind": "move_candidate",
            "candidateKey": selected["key"],
            "movementKind": selected["candidate.kind"],
            "score": score,
        }
    ), len(candidates)


def execute_policy(
    policy: CompiledPolicy,
    observation: Mapping[str, Any],
    memory: Mapping[str, int] | None = None,
) -> ExecutionResult:
    """Execute one bounded activation and return declarative effects."""

    validate_observation(policy, observation)
    scalars = observation["scalars"]
    candidates = observation["candidates"]
    specs = _memory_map(policy.memory)
    state = {item.name: item.initial for item in policy.memory}
    if memory is not None:
        if set(memory) != set(specs):
            raise PolicyValidationError(
                "runtime memory keys do not match policy registers"
            )
        for name, value in memory.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 2 ** specs[name].bits
            ):
                raise PolicyValidationError(f"runtime memory {name} is out of range")
            state[name] = value

    operations = 0
    matched_rule: int | None = None
    selected_actions = policy.default_actions
    for index, (condition, actions) in enumerate(policy.rules):
        matched, cost = _eval_expression(condition, scalars, state)
        operations += cost
        if operations > policy.max_operations:
            raise PolicyValidationError(
                "runtime operation budget exceeded in condition"
            )
        if matched:
            matched_rule = index
            selected_actions = actions
            break

    materialized: list[Mapping[str, Any]] = []
    emitted: dict[int, int] = {}
    for action in selected_actions:
        kind = action["kind"]
        operations += 1
        if kind == "noop":
            materialized.append(MappingProxyType({"kind": "noop"}))
        elif kind == "swap_relative":
            materialized.append(
                MappingProxyType({"kind": kind, "offset": action["offset"]})
            )
        elif kind == "swap_cursor":
            materialized.append(MappingProxyType({"kind": kind}))
        elif kind == "advance_cursor":
            delta = 1 if scalars["own.direction"] == "ascending" else -1
            materialized.append(MappingProxyType({"kind": kind, "delta": delta}))
        elif kind == "set_memory":
            name = action["register"]
            maximum = 2 ** specs[name].bits - 1
            mode = action["mode"]
            if mode == "assign":
                value = int(_operand_value(action["value"], scalars, state))
                state[name] = min(max(value, 0), maximum)
            elif mode == "increment_saturating":
                state[name] = min(maximum, state[name] + 1)
            elif mode == "decrement_saturating":
                state[name] = max(0, state[name] - 1)
            else:
                state[name] ^= 1
            materialized.append(
                MappingProxyType({"kind": kind, "register": name, "value": state[name]})
            )
        elif kind == "emit_signal":
            channel = int(action["channel"])
            maximum = 2**policy.signal_bits_per_channel - 1
            value = int(_operand_value(action["value"], scalars, state))
            emitted[channel] = min(max(value, 0), maximum)
            materialized.append(
                MappingProxyType(
                    {"kind": kind, "channel": channel, "value": emitted[channel]}
                )
            )
        else:
            decision, scan_cost = _select_candidate(action, scalars, candidates)
            operations += scan_cost
            materialized.append(decision)
        if operations > policy.max_operations:
            raise PolicyValidationError("runtime operation budget exceeded in actions")

    return ExecutionResult(
        policy_id=policy.policy_id,
        policy_sha256=policy.policy_sha256,
        matched_rule=matched_rule,
        actions=tuple(materialized),
        memory=MappingProxyType(dict(sorted(state.items()))),
        emitted_signals=MappingProxyType(dict(sorted(emitted.items()))),
        operation_count=operations,
    )
