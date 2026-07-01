"""Policy generation utilities for E03 S05."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.e03.classic_policies import CLASSIC_DSL_SOURCES
from src.e03.rule_dsl import (
    DSLAction,
    DSLArrayState,
    DSLCondition,
    DSLInterpreter,
    DSLPolicy,
    DSLRule,
    DSLValidationError,
    parse_policy,
    stable_json,
    validate_policy,
)


GENERATION_SCHEMA = "eidosoma.e03.generated_policy_library.v1"
DEFAULT_GENERATION_SEED = 2026070105
DEFAULT_TARGET_UNIQUE = 2500
DEFAULT_MAX_ATTEMPTS = 30000

TARGETS = ("left", "right", "ideal")
STATE_INITS = ("none", "left_boundary", "right_boundary", "current")
COMPARATORS = ("self_lt", "self_gt", "self_le", "self_ge")
PROBABILITIES = ("0.1", "0.25", "0.33", "0.5", "0.67", "0.75", "0.9")
MEMORY_KEYS = ("last_move", "blocked", "side_bias", "target_seen", "phase")
SIGNALS = ("cluster", "ready", "blocked", "left", "right")


@dataclass(frozen=True)
class CandidatePolicy:
    """One policy candidate before final duplicate filtering."""

    source: str
    source_kind: str
    generator_index: int
    generator_seed: int
    parent_policy_ids: tuple[str, ...] = ()
    mutation_operator: str | None = None


@dataclass(frozen=True)
class AcceptedPolicy:
    """One accepted generated policy plus validation metadata."""

    policy: DSLPolicy
    semantic_hash: str
    source_kind: str
    generator_index: int
    generator_seed: int
    parent_policy_ids: tuple[str, ...]
    mutation_operator: str | None
    features: Mapping[str, Any]
    tiny_execution_signature: str
    tiny_execution_action_counts: Mapping[str, int]

    def to_json_record(self) -> dict[str, Any]:
        return {
            "schema": GENERATION_SCHEMA,
            "experimentId": "E03",
            "researchStepId": "S05",
            "policyId": self.policy.policy_id,
            "policyName": self.policy.name,
            "dslSha256": self.policy.sha256,
            "semanticHash": self.semantic_hash,
            "sourceKind": self.source_kind,
            "generatorIndex": self.generator_index,
            "generatorSeed": self.generator_seed,
            "parentPolicyIds": list(self.parent_policy_ids),
            "mutationOperator": self.mutation_operator,
            "stateInit": self.policy.state_init,
            "ruleCount": len(self.policy.rules),
            "features": dict(self.features),
            "tinyExecutionSignature": self.tiny_execution_signature,
            "tinyExecutionActionCounts": dict(self.tiny_execution_action_counts),
            "dslSource": self.policy.to_source(),
        }


@dataclass(frozen=True)
class RejectedPolicy:
    """Compact audit record for a rejected candidate."""

    source_kind: str
    generator_index: int
    generator_seed: int
    reason: str
    detail: str
    duplicate_of_policy_id: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "generator_index": self.generator_index,
            "generator_seed": self.generator_seed,
            "reason": self.reason,
            "detail": self.detail[:500],
            "duplicate_of_policy_id": self.duplicate_of_policy_id,
        }


def semantic_payload(policy: DSLPolicy) -> dict[str, Any]:
    """Return policy semantics excluding its name and metadata."""

    return {
        "version": policy.version,
        "state_init": policy.state_init,
        "rules": [rule.to_dict() for rule in policy.rules],
    }


def semantic_hash(policy: DSLPolicy) -> str:
    return hashlib.sha256(stable_json(semantic_payload(policy)).encode("utf-8")).hexdigest()


def policy_with_name(policy: DSLPolicy, name: str) -> DSLPolicy:
    renamed = DSLPolicy(name=name, version=policy.version, state_init=policy.state_init, rules=policy.rules)
    validate_policy(renamed)
    return renamed


def canonical_generated_policy(policy: DSLPolicy) -> DSLPolicy:
    digest = semantic_hash(policy)
    return policy_with_name(policy, f"s05_{digest[:16]}")


def policy_feature_flags(policy: DSLPolicy) -> dict[str, Any]:
    conditions = [condition for rule in policy.rules for condition in rule.conditions]
    actions = [action for rule in policy.rules for action in rule.actions]
    target_args = [
        arg
        for item in [*conditions, *actions]
        for arg in item.args
        if arg in TARGETS
    ]
    return {
        "conditionCount": len(conditions),
        "actionCount": len(actions),
        "usesLeft": "left" in target_args,
        "usesRight": "right" in target_args,
        "usesIdeal": "ideal" in target_args or policy.state_init != "none",
        "usesPrefixSorted": any(condition.name == "prefix_sorted" for condition in conditions),
        "usesRandomCondition": any(condition.name == "random_lt" for condition in conditions),
        "usesProbabilisticAction": any(action.name == "choose" for action in actions),
        "usesMemory": any(action.name == "remember" for action in actions),
        "usesSignal": any(action.name == "signal" for action in actions),
        "swapActionCount": sum(action.name == "swap" for action in actions),
        "stateActionCount": sum(action.name in {"set_ideal", "estimate_target_position"} for action in actions),
        "waitActionCount": sum(action.name == "wait" for action in actions),
    }


def parse_round_trip_policy(source: str) -> DSLPolicy:
    first = parse_policy(source)
    rendered = first.to_source()
    second = parse_policy(rendered)
    if first.to_dict() != second.to_dict():
        raise DSLValidationError("Round-trip parse/render changed policy semantics")
    return first


def tiny_execution_fixtures() -> tuple[DSLArrayState, ...]:
    return (
        DSLArrayState(values=(2, 1), actor_index=0),
        DSLArrayState(values=(2, 1), actor_index=1),
        DSLArrayState(values=(2, 1, 3), actor_index=1),
        DSLArrayState(values=(3, 1, 2), actor_index=1),
        DSLArrayState(values=(1, 2, 3), actor_index=1, reverse_direction=True),
        DSLArrayState(values=(3, 1, 2), actor_index=0, statuses=("ACTIVE", "FREEZE", "ACTIVE")),
        DSLArrayState(values=(4, 1, 3, 2), actor_index=2, ideal_position=0),
    )


def tiny_execution_signature(policy: DSLPolicy, seed: int) -> tuple[str, dict[str, int]]:
    interpreter = DSLInterpreter(policy)
    records: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    for idx, state in enumerate(tiny_execution_fixtures()):
        result = interpreter.step_state(state, random.Random(seed + idx))
        action_type = result.action.action_type
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
        records.append(
            {
                "fixture": idx,
                "before": result.state_before.values,
                "after": result.state_after.values,
                "actor_before": result.state_before.actor_index,
                "actor_after": result.state_after.actor_index,
                "ideal_before": result.state_before.ideal_position,
                "ideal_after": result.state_after.ideal_position,
                "action": action_type,
                "target": result.action.target_index,
                "compare": result.action.compare_counted,
                "state_update": dict(result.action.state_update),
            }
        )
    signature = hashlib.sha256(stable_json(records).encode("utf-8")).hexdigest()
    return signature, action_counts


def rule(conditions: Iterable[DSLCondition], actions: Iterable[DSLAction], *, is_else: bool = False) -> DSLRule:
    return DSLRule(conditions=tuple(conditions), actions=tuple(actions), is_else=is_else)


def cond(name: str, *args: str) -> DSLCondition:
    return DSLCondition(name=name, args=tuple(args))


def act(name: str, *args: str) -> DSLAction:
    return DSLAction(name=name, args=tuple(args))


def policy_from_parts(name: str, state_init: str, rules: Iterable[DSLRule]) -> DSLPolicy:
    policy = DSLPolicy(name=name, version=1, state_init=state_init, rules=tuple(rules))
    validate_policy(policy)
    return policy


def base_rule_templates() -> list[tuple[str, str, tuple[DSLCondition, ...], tuple[DSLAction, ...]]]:
    templates: list[tuple[str, str, tuple[DSLCondition, ...], tuple[DSLAction, ...]]] = []
    for target in TARGETS:
        for comparator in COMPARATORS:
            templates.append(
                (
                    f"{target}_{comparator}_movable_swap",
                    "none",
                    (cond("target_exists", target), cond("target_movable", target), cond(comparator, target)),
                    (act("compare", target), act("swap", target)),
                )
            )
            templates.append(
                (
                    f"{target}_{comparator}_active_swap",
                    "none",
                    (cond("target_exists", target), cond("target_active", target), cond(comparator, target)),
                    (act("compare", target), act("swap", target)),
                )
            )
        templates.append(
            (
                f"{target}_blocked_signal",
                "none",
                (cond("target_exists", target),),
                (act("signal", f"seen_{target}"), act("wait")),
            )
        )
    for comparator in COMPARATORS:
        templates.append(
            (
                f"prefix_left_{comparator}",
                "none",
                (cond("prefix_sorted"), cond("target_exists", "left"), cond("target_active", "left"), cond(comparator, "left")),
                (act("compare", "left"), act("swap", "left")),
            )
        )
        templates.append(
            (
                f"ideal_{comparator}_advance",
                "left_boundary",
                (cond("target_exists", "ideal"), cond("not_at_ideal"), cond(comparator, "ideal")),
                (act("compare", "ideal"), act("set_ideal", "next")),
            )
        )
    return templates


def hand_designed_candidates(seed: int) -> list[CandidatePolicy]:
    candidates: list[CandidatePolicy] = []
    index = 0
    for key, source in CLASSIC_DSL_SOURCES.items():
        candidates.append(CandidatePolicy(source=source, source_kind="classic_dsl_seed", generator_index=index, generator_seed=seed))
        index += 1
    templates = base_rule_templates()
    for name, state_init, conditions, actions in templates:
        policy = policy_from_parts(
            f"hand_{name}",
            state_init,
            (
                rule(conditions, actions),
                rule((), (act("wait"),), is_else=True),
            ),
        )
        candidates.append(CandidatePolicy(policy.to_source(), "hand_designed", index, seed))
        index += 1
    for left_cmp, right_cmp in (("self_lt", "self_gt"), ("self_gt", "self_lt"), ("self_le", "self_ge"), ("self_ge", "self_le")):
        policy = policy_from_parts(
            f"hand_bidirectional_{left_cmp}_{right_cmp}",
            "none",
            (
                rule((cond("target_exists", "left"), cond("target_movable", "left"), cond(left_cmp, "left")), (act("compare", "left"), act("swap", "left"))),
                rule((cond("target_exists", "right"), cond("target_movable", "right"), cond(right_cmp, "right")), (act("compare", "right"), act("swap", "right"))),
                rule((), (act("wait"),), is_else=True),
            ),
        )
        candidates.append(CandidatePolicy(policy.to_source(), "hand_designed", index, seed))
        index += 1
    for probability in ("0.25", "0.5", "0.75"):
        policy = policy_from_parts(
            f"hand_probabilistic_right_{probability.replace('.', '_')}",
            "none",
            (
                rule(
                    (cond("target_exists", "right"), cond("target_movable", "right"), cond("self_gt", "right")),
                    (act("compare", "right"), act("choose", probability, "swap(right)", "wait")),
                ),
                rule((), (act("wait"),), is_else=True),
            ),
        )
        candidates.append(CandidatePolicy(policy.to_source(), "hand_designed", index, seed))
        index += 1
    return candidates


def random_guard(rng: random.Random, state_init: str) -> tuple[DSLCondition, ...]:
    target = rng.choice(TARGETS)
    comparator = rng.choice(COMPARATORS)
    pattern = rng.choice(
        (
            "target_compare",
            "active_compare",
            "movable_compare",
            "prefix_left",
            "ideal_compare",
            "random_compare",
            "always",
        )
    )
    if pattern == "always":
        return (cond("always"),)
    if pattern == "prefix_left":
        return (
            cond("prefix_sorted"),
            cond("target_exists", "left"),
            cond(rng.choice(("target_active", "target_movable")), "left"),
            cond(comparator, "left"),
        )
    if pattern == "ideal_compare" or (target == "ideal" and state_init != "none"):
        return (
            cond("target_exists", "ideal"),
            cond(rng.choice(("not_at_ideal", "at_ideal"))),
            cond(rng.choice(("target_active", "target_movable")), "ideal"),
            cond(comparator, "ideal"),
        )
    guards: list[DSLCondition] = [cond("target_exists", target)]
    if pattern in {"active_compare", "movable_compare"}:
        guards.append(cond("target_active" if pattern == "active_compare" else "target_movable", target))
    if pattern == "random_compare":
        guards.insert(0, cond("random_lt", rng.choice(PROBABILITIES)))
    guards.append(cond(comparator, target))
    return tuple(guards)


def random_actions(rng: random.Random, state_init: str) -> tuple[DSLAction, ...]:
    target = rng.choice(TARGETS)
    pattern = rng.choice(
        (
            "compare_swap",
            "swap_only",
            "compare_choose_swap",
            "advance_ideal",
            "estimate_ideal",
            "remember_wait",
            "signal_wait",
            "wait",
        )
    )
    if pattern == "compare_swap":
        return (act("compare", target), act("swap", target))
    if pattern == "swap_only":
        return (act("swap", target),)
    if pattern == "compare_choose_swap":
        return (act("compare", target), act("choose", rng.choice(PROBABILITIES), f"swap({target})", "wait"))
    if pattern == "advance_ideal":
        return (act("compare", "ideal"), act("set_ideal", rng.choice(("next", "left_boundary", "right_boundary", "current"))))
    if pattern == "estimate_ideal":
        return (act("estimate_target_position", rng.choice(("next", "left_boundary", "right_boundary", "current"))),)
    if pattern == "remember_wait":
        return (act("remember", rng.choice(MEMORY_KEYS), rng.choice(("left", "right", "ideal", "blocked"))), act("wait"))
    if pattern == "signal_wait":
        return (act("signal", rng.choice(SIGNALS)), act("wait"))
    return (act("wait"),)


def random_policy_candidate(rng: random.Random, index: int, seed: int) -> CandidatePolicy:
    state_init = rng.choice(STATE_INITS)
    rule_count = rng.randint(1, 4)
    rules: list[DSLRule] = []
    for _ in range(rule_count):
        rules.append(rule(random_guard(rng, state_init), random_actions(rng, state_init)))
    rules.append(rule((), (act("wait"),), is_else=True))
    policy = policy_from_parts(f"random_{index}", state_init, rules)
    return CandidatePolicy(policy.to_source(), "random_expansion", index, seed)


def mutate_policy(policy: DSLPolicy, rng: random.Random, index: int, seed: int) -> CandidatePolicy:
    rules = [DSLRule(rule.conditions, rule.actions, rule.is_else) for rule in policy.rules]
    operator = rng.choice(("flip_comparator", "retarget", "change_probability", "insert_rule", "drop_rule", "change_state"))
    if operator == "flip_comparator":
        replacements = {"self_lt": "self_gt", "self_gt": "self_lt", "self_le": "self_ge", "self_ge": "self_le"}
        rules = [
            DSLRule(
                tuple(DSLCondition(replacements.get(c.name, c.name), c.args) for c in item.conditions),
                item.actions,
                item.is_else,
            )
            for item in rules
        ]
    elif operator == "retarget":
        old = rng.choice(TARGETS)
        new = rng.choice([target for target in TARGETS if target != old])
        rules = [
            DSLRule(
                tuple(DSLCondition(c.name, tuple(new if arg == old else arg for arg in c.args)) for c in item.conditions),
                tuple(DSLAction(a.name, tuple(new if arg == old else arg for arg in a.args)) for a in item.actions),
                item.is_else,
            )
            for item in rules
        ]
    elif operator == "change_probability":
        rules = [
            DSLRule(
                tuple(DSLCondition(c.name, (rng.choice(PROBABILITIES),) if c.name == "random_lt" else c.args) for c in item.conditions),
                tuple(DSLAction(a.name, (rng.choice(PROBABILITIES), *a.args[1:]) if a.name == "choose" else a.args) for a in item.actions),
                item.is_else,
            )
            for item in rules
        ]
    elif operator == "insert_rule":
        non_else_count = sum(not item.is_else for item in rules)
        insert_at = rng.randint(0, non_else_count)
        state_init = policy.state_init
        rules.insert(insert_at, rule(random_guard(rng, state_init), random_actions(rng, state_init)))
    elif operator == "drop_rule":
        non_else_indices = [idx for idx, item in enumerate(rules) if not item.is_else]
        if len(non_else_indices) > 1:
            del rules[rng.choice(non_else_indices)]
    elif operator == "change_state":
        pass
    state_init = rng.choice(STATE_INITS) if operator == "change_state" else policy.state_init
    mutated = policy_from_parts(f"mutation_{index}", state_init, rules)
    return CandidatePolicy(mutated.to_source(), "mutation", index, seed, (policy.policy_id,), operator)


def recombine_policies(left: DSLPolicy, right: DSLPolicy, rng: random.Random, index: int, seed: int) -> CandidatePolicy:
    left_rules = [rule for rule in left.rules if not rule.is_else]
    right_rules = [rule for rule in right.rules if not rule.is_else]
    chosen: list[DSLRule] = []
    if left_rules:
        chosen.extend(rng.sample(left_rules, k=rng.randint(1, min(len(left_rules), 2))))
    if right_rules:
        chosen.extend(rng.sample(right_rules, k=rng.randint(1, min(len(right_rules), 2))))
    rng.shuffle(chosen)
    chosen = chosen[:4]
    chosen.append(rule((), (act("wait"),), is_else=True))
    state_init = rng.choice((left.state_init, right.state_init, rng.choice(STATE_INITS)))
    child = policy_from_parts(f"recombination_{index}", state_init, chosen)
    return CandidatePolicy(child.to_source(), "recombination", index, seed, (left.policy_id, right.policy_id), "rule_recombination")


def _accept_candidate(
    candidate: CandidatePolicy,
    seen_semantics: dict[str, str],
) -> tuple[AcceptedPolicy | None, RejectedPolicy | None]:
    try:
        parsed = parse_round_trip_policy(candidate.source)
        canonical = parsed if candidate.source_kind == "classic_dsl_seed" else canonical_generated_policy(parsed)
        reparsed = parse_round_trip_policy(canonical.to_source())
        sem_hash = semantic_hash(reparsed)
        duplicate = seen_semantics.get(sem_hash)
        if duplicate:
            return None, RejectedPolicy(candidate.source_kind, candidate.generator_index, candidate.generator_seed, "duplicate", sem_hash, duplicate)
        signature, action_counts = tiny_execution_signature(reparsed, candidate.generator_seed)
        features = policy_feature_flags(reparsed)
    except Exception as exc:
        return None, RejectedPolicy(candidate.source_kind, candidate.generator_index, candidate.generator_seed, "invalid_or_execution_failed", repr(exc))
    seen_semantics[sem_hash] = reparsed.policy_id
    return (
        AcceptedPolicy(
            policy=reparsed,
            semantic_hash=sem_hash,
            source_kind=candidate.source_kind,
            generator_index=candidate.generator_index,
            generator_seed=candidate.generator_seed,
            parent_policy_ids=candidate.parent_policy_ids,
            mutation_operator=candidate.mutation_operator,
            features=features,
            tiny_execution_signature=signature,
            tiny_execution_action_counts=action_counts,
        ),
        None,
    )


def generate_policy_library(
    *,
    target_unique: int = DEFAULT_TARGET_UNIQUE,
    seed: int = DEFAULT_GENERATION_SEED,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[list[AcceptedPolicy], pd.DataFrame]:
    """Generate, validate, and deduplicate an S05 policy library."""

    if target_unique < 1:
        raise ValueError("target_unique must be positive")
    rng = random.Random(seed)
    accepted: list[AcceptedPolicy] = []
    rejected: list[RejectedPolicy] = []
    seen_semantics: dict[str, str] = {}
    attempts = 0

    def process(candidate: CandidatePolicy) -> None:
        nonlocal attempts
        attempts += 1
        ok, bad = _accept_candidate(candidate, seen_semantics)
        if ok is not None:
            accepted.append(ok)
        if bad is not None:
            rejected.append(bad)

    for candidate in hand_designed_candidates(seed):
        process(candidate)

    index = len(hand_designed_candidates(seed))
    while len(accepted) < target_unique and attempts < max_attempts:
        if len(accepted) < 50 or rng.random() < 0.70:
            candidate = random_policy_candidate(rng, index, seed)
        elif rng.random() < 0.65:
            parent = rng.choice(accepted).policy
            candidate = mutate_policy(parent, rng, index, seed)
        else:
            left, right = rng.sample(accepted, 2)
            candidate = recombine_policies(left.policy, right.policy, rng, index, seed)
        process(candidate)
        index += 1

    audit = pd.DataFrame(
        [
            {
                "accepted": True,
                "source_kind": item.source_kind,
                "generator_index": item.generator_index,
                "generator_seed": item.generator_seed,
                "policy_id": item.policy.policy_id,
                "semantic_hash": item.semantic_hash,
                "reason": "accepted",
                "detail": "parse/hash/tiny-array execution passed",
                "duplicate_of_policy_id": None,
            }
            for item in accepted
        ]
        + [
            {"accepted": False, "policy_id": None, "semantic_hash": None, **item.to_record()}
            for item in rejected
        ]
    )
    return accepted, audit


def write_policy_jsonl(path: Path, policies: list[AcceptedPolicy]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for policy in policies:
            handle.write(json.dumps(policy.to_json_record(), sort_keys=True) + "\n")


def accepted_policy_frame(policies: list[AcceptedPolicy]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_id": policy.policy.policy_id,
                "policy_name": policy.policy.name,
                "dsl_sha256": policy.policy.sha256,
                "semantic_hash": policy.semantic_hash,
                "source_kind": policy.source_kind,
                "state_init": policy.policy.state_init,
                "rule_count": len(policy.policy.rules),
                "generator_index": policy.generator_index,
                "parent_policy_ids_json": json.dumps(list(policy.parent_policy_ids), separators=(",", ":")),
                "mutation_operator": policy.mutation_operator,
                **dict(policy.features),
                "tiny_execution_signature": policy.tiny_execution_signature,
                "tiny_swap_actions": int(policy.tiny_execution_action_counts.get("swap", 0)),
                "tiny_wait_actions": int(policy.tiny_execution_action_counts.get("wait", 0)),
                "tiny_update_state_actions": int(policy.tiny_execution_action_counts.get("update_state", 0)),
            }
            for policy in policies
        ]
    )


def generation_summary_frame(policies: list[AcceptedPolicy], audit: pd.DataFrame) -> pd.DataFrame:
    accepted_df = accepted_policy_frame(policies)
    rows: list[dict[str, Any]] = [
        {"metric": "target_unique_policies", "value": len(policies), "detail": "Accepted unique policies in final JSONL library."},
        {"metric": "candidate_attempts", "value": len(audit), "detail": "All accepted plus rejected candidate records."},
        {
            "metric": "duplicate_rejections",
            "value": int((audit["reason"] == "duplicate").sum()) if not audit.empty else 0,
            "detail": "Rejected candidates with duplicate semantic hashes.",
        },
        {
            "metric": "invalid_or_execution_rejections",
            "value": int((audit["reason"] == "invalid_or_execution_failed").sum()) if not audit.empty else 0,
            "detail": "Rejected candidates that failed parse, round trip, or tiny-array execution.",
        },
        {
            "metric": "unique_policy_ids",
            "value": int(accepted_df["policy_id"].nunique()) if not accepted_df.empty else 0,
            "detail": "Unique DSL policy IDs in final library.",
        },
        {
            "metric": "unique_semantic_hashes",
            "value": int(accepted_df["semantic_hash"].nunique()) if not accepted_df.empty else 0,
            "detail": "Unique name-independent semantic hashes in final library.",
        },
        {
            "metric": "policies_with_swap_action",
            "value": int((accepted_df["swapActionCount"] > 0).sum()) if not accepted_df.empty else 0,
            "detail": "Accepted policies containing at least one swap action.",
        },
        {
            "metric": "policies_with_ideal_state",
            "value": int(accepted_df["usesIdeal"].sum()) if not accepted_df.empty else 0,
            "detail": "Accepted policies using target-position/ideal state.",
        },
        {
            "metric": "policies_with_randomness",
            "value": int((accepted_df["usesRandomCondition"] | accepted_df["usesProbabilisticAction"]).sum()) if not accepted_df.empty else 0,
            "detail": "Accepted policies using random guards or probabilistic actions.",
        },
        {
            "metric": "policies_with_memory_or_signal",
            "value": int((accepted_df["usesMemory"] | accepted_df["usesSignal"]).sum()) if not accepted_df.empty else 0,
            "detail": "Accepted policies using parsed memory or signal primitives.",
        },
    ]
    for source_kind, count in accepted_df["source_kind"].value_counts().sort_index().items():
        rows.append({"metric": f"accepted_source_kind_{source_kind}", "value": int(count), "detail": "Accepted policies by source kind."})
    return pd.DataFrame(rows)


def validate_generated_library(policies: list[AcceptedPolicy], target_unique: int) -> pd.DataFrame:
    frame = accepted_policy_frame(policies)
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

    add("target_count_met", len(policies) >= target_unique, f">= {target_unique}", len(policies), "Final library contains thousands of unique policies.")
    add(
        "policy_ids_unique",
        frame["policy_id"].nunique() == len(frame),
        "one unique policy_id per row",
        frame["policy_id"].nunique(),
        "Policy ID uniqueness after semantic duplicate filtering.",
    )
    add(
        "semantic_hashes_unique",
        frame["semantic_hash"].nunique() == len(frame),
        "one unique semantic hash per row",
        frame["semantic_hash"].nunique(),
        "Deduplication ignores policy names and metadata.",
    )
    add(
        "all_tiny_execution_signatures_present",
        bool(frame["tiny_execution_signature"].notna().all()),
        "non-null tiny execution signature for every row",
        int(frame["tiny_execution_signature"].notna().sum()),
        "Tiny-array execution validated every accepted policy.",
    )
    required_kinds = {"hand_designed", "random_expansion", "mutation", "recombination"}
    observed_kinds = set(frame["source_kind"])
    add(
        "required_generation_modes_present",
        required_kinds.issubset(observed_kinds),
        sorted(required_kinds),
        sorted(observed_kinds),
        "S05 requires hand-designed, random, mutation, and recombination sources.",
    )
    add(
        "swap_capable_policy_count",
        int((frame["swapActionCount"] > 0).sum()) >= int(0.5 * len(frame)),
        "at least half contain swap actions",
        int((frame["swapActionCount"] > 0).sum()),
        "Most policies should be executable movement candidates, not pure wait rules.",
    )
    add(
        "ideal_state_policy_count",
        int(frame["usesIdeal"].sum()) > 0,
        "some policies use target-position state",
        int(frame["usesIdeal"].sum()),
        "Selection-style target-position state remains represented in generated space.",
    )
    return pd.DataFrame(cases)
