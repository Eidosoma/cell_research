"""Parameterized DSL templates for the classic S01 policies.

S03 does not widen the S02 interpreter. Instead it records where Bubble,
Insertion, and Selection land inside the current DSL and states the minimal
extensions needed for exact S01 replay when the current primitive set is too
narrow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .rule_dsl import DSL_VERSION, RuleProgram, parse_rule_program


ORIENTATIONS = {"increasing", "decreasing"}
NEIGHBOR_PRIORITIES = {"left_first", "right_first"}
SELECTION_MODES = {"value_rank_swap"}


def _validate_choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _left_operator(orientation: str) -> str:
    return "<" if orientation == "increasing" else ">"


def _right_operator(orientation: str) -> str:
    return ">" if orientation == "increasing" else "<"


def _direction_name(orientation: str) -> str:
    return "smaller" if orientation == "increasing" else "larger"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


@dataclass(frozen=True)
class ClassicTemplateRecord:
    """Machine-readable bridge from a named classic policy to DSL primitives."""

    algorithm: str
    template_id: str
    policy_id: str
    representation_type: str
    exact_behavior: bool
    dsl_program: RuleProgram
    parameter_defaults: dict[str, Any] = field(default_factory=dict)
    parameter_space: dict[str, Any] = field(default_factory=dict)
    approximate_behavior: str = ""
    minimal_dsl_extension: list[str] = field(default_factory=list)
    validation_scope: str = ""
    caveats: list[str] = field(default_factory=list)
    linked_s01_policy_id: str = ""

    def to_dict(self, *, include_program: bool = True) -> dict[str, Any]:
        result = {
            "algorithm": self.algorithm,
            "templateId": self.template_id,
            "policyId": self.policy_id,
            "representationType": self.representation_type,
            "exactBehavior": self.exact_behavior,
            "parameterDefaults": _json_ready(self.parameter_defaults),
            "parameterSpace": _json_ready(self.parameter_space),
            "approximateBehavior": self.approximate_behavior,
            "minimalDslExtension": list(self.minimal_dsl_extension),
            "validationScope": self.validation_scope,
            "caveats": list(self.caveats),
            "linkedS01PolicyId": self.linked_s01_policy_id,
            "dslVersion": self.dsl_program.version,
        }
        if include_program:
            result["dslProgram"] = self.dsl_program.to_dict()
        return result


def bubble_template(
    *,
    orientation: str = "increasing",
    neighbor_priority: str = "left_first",
    action_probability: float = 1.0,
    policy_id: str | None = None,
) -> RuleProgram:
    """Return a Bubble-like local-inversion DSL region point.

    The S01 Bubble wrapper samples a direction before comparing. The S02 DSL is
    first-match, so this template chooses a deterministic priority when both
    neighbors are swappable.
    """

    orientation = _validate_choice(orientation, ORIENTATIONS, "orientation")
    neighbor_priority = _validate_choice(neighbor_priority, NEIGHBOR_PRIORITIES, "neighbor_priority")
    policy_id = policy_id or f"dsl_template_bubble_{orientation}_{neighbor_priority}"
    left_rule = {
        "name": f"move_left_if_{_direction_name(orientation)}_than_left",
        "when": [{"op": "compare_left", "operator": _left_operator(orientation)}],
        "then": {"action": "swap_left", "probability": float(action_probability)},
    }
    right_rule = {
        "name": f"move_right_if_{'larger' if orientation == 'increasing' else 'smaller'}_than_right",
        "when": [{"op": "compare_right", "operator": _right_operator(orientation)}],
        "then": {"action": "swap_right", "probability": float(action_probability)},
    }
    rules = [left_rule, right_rule] if neighbor_priority == "left_first" else [right_rule, left_rule]
    return parse_rule_program(
        {
            "version": DSL_VERSION,
            "policy_id": policy_id,
            "name": f"Bubble-like local inversion cleaner ({orientation}, {neighbor_priority})",
            "rules": rules,
            "default": {"action": "wait", "reason": "locally_ordered"},
        }
    )


def insertion_template(
    *,
    orientation: str = "increasing",
    action_probability: float = 1.0,
    policy_id: str | None = None,
) -> RuleProgram:
    """Return the current-DSL adjacent-left Insertion template.

    Exact S01 Insertion also checks that the left prefix is already sorted, with
    Frozen Cell reset semantics. That prefix guard is not an S02 predicate.
    """

    orientation = _validate_choice(orientation, ORIENTATIONS, "orientation")
    policy_id = policy_id or f"dsl_template_insertion_adjacent_{orientation}"
    return parse_rule_program(
        {
            "version": DSL_VERSION,
            "policy_id": policy_id,
            "name": f"Insertion-like adjacent-left mover ({orientation})",
            "rules": [
                {
                    "name": f"move_left_if_{_direction_name(orientation)}_than_left",
                    "when": [{"op": "compare_left", "operator": _left_operator(orientation)}],
                    "then": {"action": "swap_left", "probability": float(action_probability)},
                }
            ],
            "default": {"action": "wait", "reason": "left_prefix_unchecked_or_ordered"},
        }
    )


def selection_template(
    *,
    orientation: str = "increasing",
    mode: str = "value_rank_swap",
    initial_target_position: int = 0,
    policy_id: str | None = None,
) -> RuleProgram:
    """Return an executable Selection-like target-seeking DSL template.

    ``value_rank_swap`` uses the S02 value-rank target estimator and
    ``swap_target``. This is a nearby target-position policy, not exact S01
    Selection's per-cell scan-and-advance state machine.
    """

    orientation = _validate_choice(orientation, ORIENTATIONS, "orientation")
    mode = _validate_choice(mode, SELECTION_MODES, "mode")
    policy_id = policy_id or f"dsl_template_selection_{mode}_{orientation}"
    direction = "increasing" if orientation == "increasing" else "decreasing"
    return parse_rule_program(
        {
            "version": DSL_VERSION,
            "policy_id": policy_id,
            "name": f"Selection-like value-rank target swap ({orientation})",
            "initial_state": {"target_position": int(initial_target_position)},
            "rules": [
                {
                    "name": "estimate_target_then_swap",
                    "when": [{"op": "always"}],
                    "then": {
                        "action": "swap_target",
                        "updates": [
                            {
                                "op": "estimate_target_position",
                                "key": "target_position",
                                "direction": direction,
                            }
                        ],
                    },
                }
            ],
            "default": {"action": "wait", "reason": "target_missing"},
        }
    )


def classic_template_records() -> tuple[ClassicTemplateRecord, ...]:
    """Return the S03 base records for the three classic policy families."""

    bubble = bubble_template()
    insertion = insertion_template()
    selection = selection_template()
    return (
        ClassicTemplateRecord(
            algorithm="bubble",
            template_id="classic_bubble_priority_local_inversion_region",
            policy_id=bubble.policy_id,
            representation_type="dsl_record",
            exact_behavior=False,
            dsl_program=bubble,
            parameter_defaults={
                "orientation": "increasing",
                "neighbor_priority": "left_first",
                "tie_handling": "wait_on_equal",
                "action_probability": 1.0,
            },
            parameter_space={
                "orientation": sorted(ORIENTATIONS),
                "neighbor_priority": sorted(NEIGHBOR_PRIORITIES),
                "tie_handling": ["wait_on_equal"],
                "action_probability": {"min": 0.0, "max": 1.0},
            },
            approximate_behavior=(
                "Moves an actor left when it is smaller than the left neighbor or right when it is "
                "larger than the right neighbor; first-match priority replaces S01's random direction draw."
            ),
            minimal_dsl_extension=[
                "A direction-choice primitive that samples or accepts forced_direction before predicates are evaluated.",
                "Direction-scoped compare/action clauses so the off-direction neighbor is ignored on that activation.",
                "A predicate or expression for actor.reverse_direction if reverse-goal cells must be encoded by one program.",
                "Optional wait target/reason propagation if trace-level StepOutcome equality is required.",
            ],
            validation_scope=(
                "Matches S01 Bubble on forced single-neighbor inversion steps; intentionally diverges on two-sided "
                "inversions because S02 has deterministic priority instead of random direction selection."
            ),
            caveats=["Not exact for stochastic direction choice, per-cell reverse direction, or trace wait reasons."],
            linked_s01_policy_id="classic_bubble",
        ),
        ClassicTemplateRecord(
            algorithm="insertion",
            template_id="classic_insertion_adjacent_left_region",
            policy_id=insertion.policy_id,
            representation_type="parameterized_template",
            exact_behavior=False,
            dsl_program=insertion,
            parameter_defaults={
                "orientation": "increasing",
                "prefix_guard": "not_available_in_s02",
                "tie_handling": "wait_on_equal",
                "action_probability": 1.0,
            },
            parameter_space={
                "orientation": sorted(ORIENTATIONS),
                "prefix_guard": ["none", "requires_prefix_sorted_left_extension"],
                "tie_handling": ["wait_on_equal"],
                "action_probability": {"min": 0.0, "max": 1.0},
            },
            approximate_behavior=(
                "Moves an actor left across a larger neighbor. It is a local adjacent-left insertion fragment "
                "without S01's sorted-prefix enablement check."
            ),
            minimal_dsl_extension=[
                "A prefix_sorted_left predicate with the same orientation and Frozen Cell reset semantics as S01.",
                "A predicate or expression for actor.reverse_direction if reverse-goal cells must be encoded by one program.",
                "Optional wait target/reason propagation if trace-level StepOutcome equality is required.",
            ],
            validation_scope=(
                "Matches S01 Insertion on enabled adjacent-left inversion steps; intentionally diverges when the "
                "left-prefix guard is false but the immediate left inversion is true."
            ),
            caveats=["Not exact for unsorted left prefixes, Frozen-prefix reset behavior, or per-cell reverse direction."],
            linked_s01_policy_id="classic_insertion",
        ),
        ClassicTemplateRecord(
            algorithm="selection",
            template_id="classic_selection_value_rank_target_region",
            policy_id=selection.policy_id,
            representation_type="parameterized_template",
            exact_behavior=False,
            dsl_program=selection,
            parameter_defaults={
                "orientation": "increasing",
                "mode": "value_rank_swap",
                "initial_target_position": 0,
                "target_position_estimate": "value_rank",
            },
            parameter_space={
                "orientation": sorted(ORIENTATIONS),
                "mode": sorted(SELECTION_MODES),
                "initial_target_position": "integer",
                "target_position_estimate": ["value_rank"],
            },
            approximate_behavior=(
                "Estimates the actor's target position from its value and swaps directly with that target. "
                "This is an executable target-seeking proxy near Selection, not S01's scan-and-advance policy."
            ),
            minimal_dsl_extension=[
                "A dynamic actor_position_equals_state_target predicate to implement already-at-ideal waits.",
                "A target_frozen predicate and frozen-target branch matching S01's state-advance and blocked-swap semantics.",
                "A bounded advance_target_position update whose limits can depend on n and reverse_direction.",
                "A legal_state_progress action class so non-swap target-advance steps keep the simulator from stopping early.",
                "A predicate or expression for actor.reverse_direction and exact initial target state n-1 for reverse cells.",
                "Optional StepOutcome reason and comparison-count controls for trace-level equality.",
            ],
            validation_scope=(
                "Matches S01 Selection on a simple min-to-target swap at the action/result level; intentionally "
                "diverges for actors already at their ideal target because current DSL lacks dynamic state-position tests."
            ),
            caveats=[
                "The executable template is a value-rank target proxy, not an exact Selection replay.",
                "Comparison accounting differs from S01 Selection because target estimation is not a compare_target predicate.",
            ],
            linked_s01_policy_id="classic_selection",
        ),
    )


def classic_template_programs() -> dict[str, RuleProgram]:
    """Return base DSL programs keyed by classic algorithm name."""

    return {record.algorithm: record.dsl_program for record in classic_template_records()}


def classic_template_table() -> list[dict[str, Any]]:
    """Return flattened metadata rows suitable for CSV or Parquet."""

    return [record.to_dict(include_program=False) for record in classic_template_records()]

