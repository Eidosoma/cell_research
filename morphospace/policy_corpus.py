"""Deterministic DSL policy corpus generation for E03 S05.

The corpus is intentionally broad rather than selective. S05 filters invalid or
non-executable programs, but keeps policies that execute and fail to sort so
later competence mapping can represent failure regions instead of only winners.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .classic_templates import bubble_template, insertion_template, selection_template
from .rule_dsl import DSL_VERSION, RuleProgram, local_inversion_program, null_program, parse_rule_program, stochastic_right_program


POLICY_CORPUS_VERSION = "e03_s05_policy_corpus.v1"
DEFAULT_CORPUS_SIZE = 2048
OPERATORS = ("<", ">", "<=", ">=", "==", "!=")
COMPARE_PREDICATES = ("compare_left", "compare_right", "compare_target")
EXISTS_PREDICATES = ("left_exists", "right_exists", "target_exists")
ACTION_NAMES = ("wait", "swap_left", "swap_right", "swap_target", "remember", "signal")
PROBABILITIES = (1.0, 0.75, 0.5, 0.25)


def canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def structure_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return behavior-relevant payload used for deduplication."""

    stripped = {key: value for key, value in payload.items() if key not in {"policy_id", "name"}}
    return json.loads(canonical_json(stripped))


def structure_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(structure_payload(payload)).encode("utf-8")).hexdigest()[:16]


def policy_id_from_hash(digest: str) -> str:
    return f"pc_{digest}"


def _action(action: str, *, probability: float = 1.0, updates: list[dict[str, Any]] | None = None, reason: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": action}
    if probability != 1.0:
        payload["probability"] = float(probability)
    if updates:
        payload["updates"] = updates
    if reason is not None:
        payload["reason"] = reason
    return payload


def _set_update(key: str, expr: Any) -> dict[str, Any]:
    return {"op": "set", "key": key, "value": expr}


def _counter_update(key: str, *, op: str = "increment", amount: int = 1, maximum: int = 3) -> dict[str, Any]:
    return {"op": op, "key": key, "amount": int(amount), "min": 0, "max": int(maximum)}


def _target_update(direction: str = "increasing") -> dict[str, Any]:
    return {"op": "estimate_target_position", "key": "target_position", "direction": direction}


def _signal_update(channel: str, value: Any) -> dict[str, Any]:
    return {"op": "signal", "channel": channel, "value": value}


def _predicate(op: str, operator: str | None = None, *, key: str | None = None, value: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"op": op}
    if operator is not None:
        payload["operator"] = operator
    if key is not None:
        payload["key"] = key
    if value is not None:
        payload["value"] = value
    return payload


def _program_payload(
    *,
    name: str,
    rules: list[dict[str, Any]],
    default: dict[str, Any] | None = None,
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": DSL_VERSION,
        "name": name,
        "initial_state": dict(initial_state or {}),
        "rules": rules,
        "default": default or {"action": "wait", "reason": "default_wait"},
    }


@dataclass(frozen=True)
class PolicyCorpusRecord:
    policy_id: str
    corpus_version: str
    dsl_version: str
    family: str
    generation_method: str
    lineage_id: str
    lineage_depth: int
    parent_policy_ids: tuple[str, ...]
    mutation_operators: tuple[str, ...]
    recombination_parents: tuple[str, ...]
    program: RuleProgram
    structure_hash: str
    generation_index: int
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def complexity(self) -> dict[str, Any]:
        rule_count = len(self.program.rules)
        predicate_count = sum(len(rule.when) for rule in self.program.rules)
        update_count = sum(len(rule.then.updates) for rule in self.program.rules) + len(self.program.default.updates)
        action_counts = Counter(rule.then.action for rule in self.program.rules)
        stochastic_action_count = sum(1 for rule in self.program.rules if rule.then.probability != 1.0)
        state_keys = set(self.program.initial_state)
        signal_count = 0
        target_update_count = 0
        for action in [rule.then for rule in self.program.rules] + [self.program.default]:
            for update in action.updates:
                if update.key:
                    state_keys.add(update.key)
                if update.op == "signal":
                    signal_count += 1
                if update.op == "estimate_target_position":
                    target_update_count += 1
        complexity_score = rule_count + predicate_count + update_count + stochastic_action_count + len(state_keys)
        return {
            "ruleCount": int(rule_count),
            "predicateCount": int(predicate_count),
            "updateCount": int(update_count),
            "stateKeyCount": int(len(state_keys)),
            "stochasticActionCount": int(stochastic_action_count),
            "targetUpdateCount": int(target_update_count),
            "signalCount": int(signal_count),
            "complexityScore": int(complexity_score),
            "actionCountsJson": canonical_json(dict(sorted(action_counts.items()))),
        }

    def observation_requirements(self) -> tuple[str, ...]:
        requirements: set[str] = set()
        for rule in self.program.rules:
            for predicate in rule.when:
                if predicate.op in {"left_exists", "compare_left"}:
                    requirements.add("left_neighbor")
                elif predicate.op in {"right_exists", "compare_right"}:
                    requirements.add("right_neighbor")
                elif predicate.op in {"target_exists", "compare_target"}:
                    requirements.add("target_cell")
                elif predicate.op in {"state_equals", "state_compare"}:
                    requirements.add("policy_state")
                elif predicate.op == "position_compare":
                    requirements.add("actor_position")
        for action in [rule.then for rule in self.program.rules] + [self.program.default]:
            if action.action == "swap_target":
                requirements.add("target_position_state")
            if action.probability != 1.0:
                requirements.add("tie_breaker_rng")
            for update in action.updates:
                if update.op == "estimate_target_position":
                    requirements.add("actor_value")
                    requirements.add("target_position_state")
                elif update.op in {"increment", "decrement", "set"}:
                    requirements.add("policy_state")
                elif update.op == "signal":
                    requirements.add("local_signal_placeholder")
        return tuple(sorted(requirements))

    def to_row(self, *, dsl_relpath: str | None = None) -> dict[str, Any]:
        complexity = self.complexity()
        return {
            "policyId": self.policy_id,
            "corpusVersion": self.corpus_version,
            "dslVersion": self.dsl_version,
            "family": self.family,
            "generationMethod": self.generation_method,
            "lineageId": self.lineage_id,
            "lineageDepth": int(self.lineage_depth),
            "parentPolicyIdsJson": canonical_json(list(self.parent_policy_ids)),
            "mutationOperatorsJson": canonical_json(list(self.mutation_operators)),
            "recombinationParentsJson": canonical_json(list(self.recombination_parents)),
            "structureHash": self.structure_hash,
            "generationIndex": int(self.generation_index),
            "description": self.description,
            "tagsJson": canonical_json(list(self.tags)),
            "observationRequirementsJson": canonical_json(list(self.observation_requirements())),
            "dslProgramJson": self.program.to_json(),
            "dslRelativePath": dsl_relpath,
            **complexity,
        }

    def lineage_row(self) -> dict[str, Any]:
        return {
            "policyId": self.policy_id,
            "lineageId": self.lineage_id,
            "generationMethod": self.generation_method,
            "lineageDepth": int(self.lineage_depth),
            "parentPolicyIdsJson": canonical_json(list(self.parent_policy_ids)),
            "mutationOperatorsJson": canonical_json(list(self.mutation_operators)),
            "recombinationParentsJson": canonical_json(list(self.recombination_parents)),
            "structureHash": self.structure_hash,
            "description": self.description,
        }


def _record_from_payload(
    payload: Mapping[str, Any],
    *,
    family: str,
    generation_method: str,
    generation_index: int,
    lineage_id: str,
    lineage_depth: int = 0,
    parent_policy_ids: Sequence[str] = (),
    mutation_operators: Sequence[str] = (),
    recombination_parents: Sequence[str] = (),
    description: str = "",
    tags: Sequence[str] = (),
) -> PolicyCorpusRecord:
    digest = structure_hash(payload)
    policy_id = policy_id_from_hash(digest)
    named_payload = dict(payload)
    named_payload["policy_id"] = policy_id
    named_payload["name"] = str(payload.get("name", f"corpus policy {policy_id}"))[:120]
    program = parse_rule_program(named_payload)
    return PolicyCorpusRecord(
        policy_id=program.policy_id,
        corpus_version=POLICY_CORPUS_VERSION,
        dsl_version=program.version,
        family=family,
        generation_method=generation_method,
        lineage_id=lineage_id,
        lineage_depth=lineage_depth,
        parent_policy_ids=tuple(parent_policy_ids),
        mutation_operators=tuple(mutation_operators),
        recombination_parents=tuple(recombination_parents),
        program=program,
        structure_hash=digest,
        generation_index=int(generation_index),
        description=description,
        tags=tuple(tags),
    )


def _base_seed_payloads() -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    seeds = [
        ("null_wait", null_program().to_dict(), "null"),
        ("local_inversion_cleaner", local_inversion_program().to_dict(), "local_inversion"),
        ("stochastic_right_swap", stochastic_right_program().to_dict(), "stochastic"),
        ("bubble_template", bubble_template().to_dict(), "classic_template"),
        ("insertion_template", insertion_template().to_dict(), "classic_template"),
        ("selection_template", selection_template().to_dict(), "classic_template"),
    ]
    for name, payload, tag in seeds:
        payload = {key: value for key, value in payload.items() if key != "policy_id"}
        yield payload, {
            "family": "seed",
            "generation_method": "seed_import",
            "lineage_id": f"seed_{name}",
            "description": f"S01-S04 seed policy: {name}",
            "tags": ("seed", tag),
        }


def _hand_designed_payloads() -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for orientation, priority, probability in itertools.product(("increasing", "decreasing"), ("left_first", "right_first"), PROBABILITIES):
        payload = bubble_template(orientation=orientation, neighbor_priority=priority, action_probability=probability).to_dict()
        payload.pop("policy_id", None)
        yield payload, {
            "family": "hand_designed",
            "generation_method": "classic_parameter_sweep",
            "lineage_id": f"bubble_{orientation}_{priority}_{probability:g}",
            "description": "Bubble-like local inversion parameter sweep.",
            "tags": ("classic_region", "bubble_like", orientation, priority),
        }
    for orientation, probability in itertools.product(("increasing", "decreasing"), PROBABILITIES):
        payload = insertion_template(orientation=orientation, action_probability=probability).to_dict()
        payload.pop("policy_id", None)
        yield payload, {
            "family": "hand_designed",
            "generation_method": "classic_parameter_sweep",
            "lineage_id": f"insertion_{orientation}_{probability:g}",
            "description": "Insertion-like adjacent-left parameter sweep.",
            "tags": ("classic_region", "insertion_like", orientation),
        }
    for orientation, initial_target in itertools.product(("increasing", "decreasing"), range(4)):
        payload = selection_template(orientation=orientation, initial_target_position=initial_target).to_dict()
        payload.pop("policy_id", None)
        yield payload, {
            "family": "hand_designed",
            "generation_method": "classic_parameter_sweep",
            "lineage_id": f"selection_{orientation}_{initial_target}",
            "description": "Selection-like value-rank target-seeking parameter sweep.",
            "tags": ("classic_region", "selection_like", orientation),
        }

    compare_to_action = {
        "compare_left": "swap_left",
        "compare_right": "swap_right",
        "compare_target": "swap_target",
    }
    for compare_op, operator, action_name, probability in itertools.product(COMPARE_PREDICATES, OPERATORS, ACTION_NAMES[:4], (1.0, 0.5)):
        updates = []
        initial_state = {}
        if compare_op == "compare_target" or action_name == "swap_target":
            updates = [_target_update("increasing")]
            initial_state = {"target_position": 0}
        payload = _program_payload(
            name=f"single predicate {compare_op} {operator} {action_name}",
            initial_state=initial_state,
            rules=[
                {
                    "name": "single_rule",
                    "when": [_predicate(compare_op, operator)],
                    "then": _action(action_name, probability=probability, updates=updates),
                }
            ],
            default={"action": "wait", "reason": "single_rule_default"},
        )
        yield payload, {
            "family": "hand_designed",
            "generation_method": "single_rule_sweep",
            "lineage_id": f"single_{compare_op}_{operator}_{action_name}_{probability:g}".replace("<", "lt").replace(">", "gt").replace("=", "eq").replace("!", "ne"),
            "description": "Single-predicate/action sweep across current DSL primitives.",
            "tags": ("single_rule", compare_op, action_name, compare_to_action.get(compare_op, "mixed_action")),
        }


def _mutated_payloads(base_records: Sequence[PolicyCorpusRecord]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for record in base_records:
        parent = record.program.to_dict()
        parent.pop("policy_id", None)

        if len(parent["rules"]) > 1:
            payload = dict(parent)
            payload["rules"] = list(reversed(parent["rules"]))
            payload["name"] = f"{parent.get('name', record.policy_id)} reversed priority"
            yield payload, {
                "family": "mutation",
                "generation_method": "priority_reversal",
                "lineage_id": f"mut_reverse_{record.policy_id}",
                "lineage_depth": record.lineage_depth + 1,
                "parent_policy_ids": (record.policy_id,),
                "mutation_operators": ("reverse_rule_order",),
                "description": "Rule-priority reversal mutation.",
                "tags": ("mutation", "priority"),
            }

        for probability in (0.25, 0.5, 0.75):
            payload = json.loads(json.dumps(parent))
            changed = False
            for rule in payload["rules"]:
                if rule["then"]["action"].startswith("swap"):
                    rule["then"]["probability"] = probability
                    changed = True
            if changed:
                payload["name"] = f"{parent.get('name', record.policy_id)} p{probability:g}"
                yield payload, {
                    "family": "mutation",
                    "generation_method": "probability_mutation",
                    "lineage_id": f"mut_prob_{record.policy_id}_{probability:g}",
                    "lineage_depth": record.lineage_depth + 1,
                    "parent_policy_ids": (record.policy_id,),
                    "mutation_operators": (f"set_swap_probability_{probability:g}",),
                    "description": "Swap-probability mutation.",
                    "tags": ("mutation", "stochastic"),
                }

        payload = json.loads(json.dumps(parent))
        payload.setdefault("initial_state", {})["memory"] = 0
        for rule in payload["rules"]:
            rule["then"].setdefault("updates", []).append(_counter_update("memory", maximum=4))
        payload["name"] = f"{parent.get('name', record.policy_id)} memory counter"
        yield payload, {
            "family": "mutation",
            "generation_method": "memory_mutation",
            "lineage_id": f"mut_memory_{record.policy_id}",
            "lineage_depth": record.lineage_depth + 1,
            "parent_policy_ids": (record.policy_id,),
            "mutation_operators": ("add_bounded_memory_counter",),
            "description": "Adds a bounded memory counter update to selected actions.",
            "tags": ("mutation", "memory"),
        }

        payload = json.loads(json.dumps(parent))
        payload.setdefault("initial_state", {})["target_position"] = 0
        for rule in payload["rules"]:
            rule["then"].setdefault("updates", []).append(_target_update("increasing"))
        payload["name"] = f"{parent.get('name', record.policy_id)} target estimate"
        yield payload, {
            "family": "mutation",
            "generation_method": "target_memory_mutation",
            "lineage_id": f"mut_target_{record.policy_id}",
            "lineage_depth": record.lineage_depth + 1,
            "parent_policy_ids": (record.policy_id,),
            "mutation_operators": ("add_target_estimate_update",),
            "description": "Adds value-rank target-position estimation to selected actions.",
            "tags": ("mutation", "target"),
        }


def _recombined_payloads(parent_records: Sequence[PolicyCorpusRecord], limit: int = 512) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    yielded = 0
    for left, right in itertools.combinations(parent_records, 2):
        if yielded >= limit:
            break
        left_rules = [rule.to_dict() for rule in left.program.rules]
        right_rules = [rule.to_dict() for rule in right.program.rules]
        if not left_rules or not right_rules:
            continue
        rules = [left_rules[0], right_rules[-1]]
        for index, rule in enumerate(rules):
            rule["name"] = f"rx_{index}_{rule['name'][:24]}"
        initial_state = dict(left.program.initial_state)
        initial_state.update(right.program.initial_state)
        payload = _program_payload(
            name=f"recombined {left.policy_id} {right.policy_id}",
            initial_state=initial_state,
            rules=rules,
            default=left.program.default.to_dict(),
        )
        yielded += 1
        yield payload, {
            "family": "recombined",
            "generation_method": "two_parent_rule_recombination",
            "lineage_id": f"rx_{left.policy_id}_{right.policy_id}",
            "lineage_depth": max(left.lineage_depth, right.lineage_depth) + 1,
            "parent_policy_ids": (left.policy_id, right.policy_id),
            "recombination_parents": (left.policy_id, right.policy_id),
            "description": "Two-parent recombination using the first rule of one parent and last rule of another.",
            "tags": ("recombination",),
        }


def _random_predicate(rng: np.random.Generator) -> dict[str, Any]:
    choice = int(rng.integers(0, 9))
    if choice == 0:
        return {"op": "always"}
    if choice in {1, 2, 3}:
        return {"op": EXISTS_PREDICATES[choice - 1]}
    if choice in {4, 5, 6}:
        return _predicate(COMPARE_PREDICATES[choice - 4], str(rng.choice(OPERATORS)))
    if choice == 7:
        return _predicate("state_compare", str(rng.choice(("<", ">", "<=", ">=", "=="))), key="memory", value=int(rng.integers(0, 4)))
    return _predicate("position_compare", str(rng.choice(("<", ">", "<=", ">=", "=="))), value=int(rng.integers(0, 4)))


def _random_action(rng: np.random.Generator) -> dict[str, Any]:
    action_name = str(rng.choice(ACTION_NAMES))
    probability = float(rng.choice(PROBABILITIES))
    updates: list[dict[str, Any]] = []
    if rng.random() < 0.35:
        updates.append(_counter_update("memory", op=str(rng.choice(("increment", "decrement"))), maximum=int(rng.integers(2, 6))))
    if rng.random() < 0.25 or action_name == "swap_target":
        updates.append(_target_update(str(rng.choice(("increasing", "decreasing")))))
    if rng.random() < 0.15:
        updates.append(_set_update("seen_value", {"expr": "actor_value"}))
    if rng.random() < 0.1 or action_name == "signal":
        updates.append(_signal_update("marker", "seen"))
    if action_name in {"wait", "remember", "signal"}:
        probability = 1.0
    return _action(action_name, probability=probability, updates=updates)


def _random_payloads(seed: int = 905, attempts: int = 4096) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    for index in range(attempts):
        rule_count = int(rng.integers(1, 5))
        initial_state: dict[str, Any] = {}
        if rng.random() < 0.55:
            initial_state["memory"] = int(rng.integers(0, 3))
        if rng.random() < 0.55:
            initial_state["target_position"] = int(rng.integers(0, 4))
        rules = []
        for rule_index in range(rule_count):
            pred_count = int(rng.integers(1, 4))
            predicates = [_random_predicate(rng) for _ in range(pred_count)]
            if not any(predicate["op"] == "always" for predicate in predicates):
                predicates = predicates[:2]
            rules.append(
                {
                    "name": f"r{rule_index}",
                    "when": predicates,
                    "then": _random_action(rng),
                }
            )
        default_action = _action(str(rng.choice(("wait", "remember"))), updates=[] if rng.random() < 0.8 else [_counter_update("memory")])
        payload = _program_payload(
            name=f"random grammar sample {index}",
            initial_state=initial_state,
            rules=rules,
            default=default_action,
        )
        yield payload, {
            "family": "random_generated",
            "generation_method": "seeded_random_grammar",
            "lineage_id": f"random_{seed}_{index}",
            "description": "Seeded random sample from the S02 DSL grammar.",
            "tags": ("random",),
        }


def generate_policy_corpus(target_size: int = DEFAULT_CORPUS_SIZE, random_seed: int = 905) -> tuple[PolicyCorpusRecord, ...]:
    """Generate a deterministic, structurally deduplicated DSL policy corpus."""

    records_by_hash: dict[str, PolicyCorpusRecord] = {}
    generation_index = 0

    def add(payload: Mapping[str, Any], meta: Mapping[str, Any]) -> None:
        nonlocal generation_index
        digest = structure_hash(payload)
        if digest in records_by_hash:
            return
        record = _record_from_payload(
            payload,
            family=str(meta.get("family", "generated")),
            generation_method=str(meta.get("generation_method", "unknown")),
            generation_index=generation_index,
            lineage_id=str(meta.get("lineage_id", f"lineage_{generation_index}")),
            lineage_depth=int(meta.get("lineage_depth", 0)),
            parent_policy_ids=tuple(str(value) for value in meta.get("parent_policy_ids", ())),
            mutation_operators=tuple(str(value) for value in meta.get("mutation_operators", ())),
            recombination_parents=tuple(str(value) for value in meta.get("recombination_parents", ())),
            description=str(meta.get("description", "")),
            tags=tuple(str(value) for value in meta.get("tags", ())),
        )
        records_by_hash[digest] = record
        generation_index += 1

    for payload, meta in _base_seed_payloads():
        add(payload, meta)
    for payload, meta in _hand_designed_payloads():
        add(payload, meta)
    seed_records = tuple(records_by_hash.values())
    for payload, meta in _mutated_payloads(seed_records):
        add(payload, meta)
    mutation_records = tuple(records_by_hash.values())
    for payload, meta in _recombined_payloads(mutation_records, limit=max(256, target_size // 3)):
        add(payload, meta)
    for payload, meta in _random_payloads(seed=random_seed, attempts=max(4096, target_size * 4)):
        add(payload, meta)
        if len(records_by_hash) >= target_size:
            break

    return tuple(sorted(records_by_hash.values(), key=lambda record: record.generation_index))


def policy_corpus_table(records: Sequence[PolicyCorpusRecord], *, dsl_relpaths: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    dsl_relpaths = dsl_relpaths or {}
    return [record.to_row(dsl_relpath=dsl_relpaths.get(record.policy_id)) for record in records]


def policy_lineage_table(records: Sequence[PolicyCorpusRecord]) -> list[dict[str, Any]]:
    return [record.lineage_row() for record in records]


def corpus_summary(records: Sequence[PolicyCorpusRecord]) -> dict[str, Any]:
    family_counts = Counter(record.family for record in records)
    method_counts = Counter(record.generation_method for record in records)
    complexity = [record.complexity()["complexityScore"] for record in records]
    return {
        "corpusVersion": POLICY_CORPUS_VERSION,
        "policyCount": int(len(records)),
        "familyCounts": dict(sorted(family_counts.items())),
        "generationMethodCounts": dict(sorted(method_counts.items())),
        "minComplexityScore": int(min(complexity)) if complexity else 0,
        "maxComplexityScore": int(max(complexity)) if complexity else 0,
        "meanComplexityScore": float(np.mean(complexity)) if complexity else 0.0,
        "dslVersion": DSL_VERSION,
    }


def validate_corpus_records(records: Sequence[PolicyCorpusRecord]) -> dict[str, Any]:
    policy_ids = [record.policy_id for record in records]
    structure_hashes = [record.structure_hash for record in records]
    parsed = 0
    errors: list[str] = []
    for record in records:
        try:
            restored = parse_rule_program(record.program.to_json())
            if restored.to_dict() != record.program.to_dict():
                errors.append(f"round_trip_mismatch:{record.policy_id}")
            else:
                parsed += 1
        except Exception as exc:  # pragma: no cover - defensive validation
            errors.append(f"{record.policy_id}:{exc!r}")
    return {
        "policyCount": int(len(records)),
        "uniquePolicyIds": int(len(set(policy_ids))),
        "uniqueStructureHashes": int(len(set(structure_hashes))),
        "duplicatePolicyIds": sorted(policy_id for policy_id, count in Counter(policy_ids).items() if count > 1),
        "duplicateStructureHashes": sorted(digest for digest, count in Counter(structure_hashes).items() if count > 1),
        "parseRoundTripCount": int(parsed),
        "parseErrors": errors,
        "success": len(records) > 0
        and len(policy_ids) == len(set(policy_ids))
        and len(structure_hashes) == len(set(structure_hashes))
        and parsed == len(records)
        and not errors,
    }
