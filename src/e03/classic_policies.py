"""Classic Bubble, Insertion, and Selection policy library for E03 S03."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from src.e02.deterministic_simulator import EventTracingStatusProbe, SimulatorConfig, _build_cells
from src.e03.policy_interface import OriginalCellPolicyWrapper, cells_signature, observe_cell, policy_for_cell
from src.e03.rule_dsl import DSLArrayState, DSLInterpreter, DSLPolicy, parse_policy, stable_json


BUBBLE_INCREASING_SHADOW_DSL = """policy classic_bubble_bidirectional_increasing_shadow v1
state ideal_position=none
rule if random_lt(0.5) and target_exists(right) and target_movable(right) and self_gt(right) then compare(right), swap(right)
rule if target_exists(left) and target_movable(left) and self_lt(left) then compare(left), swap(left)
rule else wait
end
"""

BUBBLE_DECREASING_SHADOW_DSL = """policy classic_bubble_bidirectional_decreasing_shadow v1
state ideal_position=none
rule if random_lt(0.5) and target_exists(right) and target_movable(right) and self_lt(right) then compare(right), swap(right)
rule if target_exists(left) and target_movable(left) and self_gt(left) then compare(left), swap(left)
rule else wait
end
"""

INSERTION_INCREASING_DSL = """policy classic_insertion_active_left_increasing v1
state ideal_position=none
rule if prefix_sorted and target_exists(left) and target_active(left) and self_lt(left) then compare(left), swap(left)
rule else wait
end
"""

INSERTION_DECREASING_DSL = """policy classic_insertion_active_left_decreasing v1
state ideal_position=none
rule if prefix_sorted and target_exists(left) and target_active(left) and self_gt(left) then compare(left), swap(left)
rule else wait
end
"""

SELECTION_INCREASING_DSL = """policy classic_selection_target_increasing v1
state ideal_position=left_boundary
rule if target_exists(ideal) and not_at_ideal and target_active(ideal) and self_lt(ideal) then compare(ideal), swap(ideal)
rule if target_exists(ideal) and not_at_ideal and target_active(ideal) and self_ge(ideal) then compare(ideal), set_ideal(next)
rule else wait
end
"""

SELECTION_DECREASING_DSL = """policy classic_selection_target_decreasing v1
state ideal_position=right_boundary
rule if target_exists(ideal) and not_at_ideal and target_active(ideal) and self_lt(ideal) then compare(ideal), swap(ideal)
rule if target_exists(ideal) and not_at_ideal and target_active(ideal) and self_ge(ideal) then compare(ideal), set_ideal(next)
rule else wait
end
"""


CLASSIC_DSL_SOURCES = {
    "bubble_increasing_shadow": BUBBLE_INCREASING_SHADOW_DSL,
    "bubble_decreasing_shadow": BUBBLE_DECREASING_SHADOW_DSL,
    "insertion_increasing_active": INSERTION_INCREASING_DSL,
    "insertion_decreasing_active": INSERTION_DECREASING_DSL,
    "selection_increasing_active": SELECTION_INCREASING_DSL,
    "selection_decreasing_active": SELECTION_DECREASING_DSL,
}


@dataclass(frozen=True)
class ClassicMapping:
    """One classic algorithm mapping entry."""

    policy_id: str
    algorithm: str
    representation_type: str
    exactness: str
    direction: str
    implementation_ref: str
    dsl_source: str | None
    dsl_sha256: str | None
    compatible_conditions: tuple[str, ...]
    known_deviations: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "policyId": self.policy_id,
            "algorithm": self.algorithm,
            "representationType": self.representation_type,
            "exactness": self.exactness,
            "direction": self.direction,
            "implementationRef": self.implementation_ref,
            "dslSource": self.dsl_source,
            "dslSha256": self.dsl_sha256,
            "compatibleConditions": list(self.compatible_conditions),
            "knownDeviations": list(self.known_deviations),
            "notes": self.notes,
        }


def stable_policy_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = json_hash(payload)
    return f"{prefix}:{digest[:16]}"


def json_hash(payload: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def interface_mapping(algorithm: str) -> ClassicMapping:
    payload = {
        "schema": "e03.classic_policy_mapping.v1",
        "representationType": "interface_wrapper",
        "algorithm": algorithm,
        "binding": f"OriginalCellPolicyWrapper({algorithm})",
        "source": "src/e03/policy_interface.py",
    }
    return ClassicMapping(
        policy_id=stable_policy_id("iface", payload),
        algorithm=algorithm,
        representation_type="interface_wrapper",
        exactness="exact_public_method",
        direction="parameterized_by_cell_reverse_direction",
        implementation_ref=f"src.e03.policy_interface.OriginalCellPolicyWrapper({algorithm!r})",
        dsl_source=None,
        dsl_sha256=None,
        compatible_conditions=(
            "Public repository cell object with S01/E02 cell construction",
            "Supports increasing and decreasing direction through cell.reverse_direction",
            "Supports public passive/stuck/dynamic frozen behavior through E02 simulator wrappers",
        ),
        known_deviations=(),
        notes="Exact behavior-preserving mapping because S01 delegates action application to the public move() method.",
    )


def dsl_mapping(key: str, algorithm: str, exactness: str, direction: str, compatible: tuple[str, ...], deviations: tuple[str, ...]) -> ClassicMapping:
    policy = parse_policy(CLASSIC_DSL_SOURCES[key])
    return ClassicMapping(
        policy_id=policy.policy_id,
        algorithm=algorithm,
        representation_type="dsl",
        exactness=exactness,
        direction=direction,
        implementation_ref=f"src.e03.rule_dsl.DSLPolicy({policy.name!r})",
        dsl_source=policy.to_source(),
        dsl_sha256=policy.sha256,
        compatible_conditions=compatible,
        known_deviations=deviations,
        notes="DSL mapping produced by S03 for classic-policy parameterization.",
    )


def classic_policy_mappings() -> list[ClassicMapping]:
    mappings = [
        interface_mapping("bubble"),
        interface_mapping("insertion"),
        interface_mapping("selection"),
        dsl_mapping(
            "bubble_increasing_shadow",
            "bubble",
            "approximate_shadow",
            "increasing",
            (
                "Local bidirectional inversion-cleaning proxy",
                "Useful as a DSL neighborhood point near Bubble behavior",
            ),
            (
                "Does not exactly preserve public Bubble's sampled side when the sampled side is blocked but the opposite side is swappable",
                "Comparison-count accounting is only target-local, while the public method counts if either active neighbor is disordered",
                "Frozen-target comparison semantics are not exact",
            ),
        ),
        dsl_mapping(
            "bubble_decreasing_shadow",
            "bubble",
            "approximate_shadow",
            "decreasing",
            (
                "Reverse-direction local bidirectional inversion-cleaning proxy",
                "Useful as a DSL neighborhood point near Bubble behavior",
            ),
            (
                "Same sampled-side and comparison-accounting limitations as the increasing Bubble shadow",
            ),
        ),
        dsl_mapping(
            "insertion_increasing_active",
            "insertion",
            "exact_active_unfrozen_subset",
            "increasing",
            (
                "Actor is active",
                "Left target exists and is active when swapping",
                "No frozen target interaction in the tested decision",
                "Increasing sort direction",
            ),
            (
                "Passive frozen left-target swaps are not exact because S02 DSL lacks a target_frozen guard and would otherwise overcount compare cost",
            ),
        ),
        dsl_mapping(
            "insertion_decreasing_active",
            "insertion",
            "exact_active_unfrozen_subset",
            "decreasing",
            (
                "Actor is active",
                "Left target exists and is active when swapping",
                "No frozen target interaction in the tested decision",
                "Decreasing sort direction",
            ),
            (
                "Passive frozen left-target swaps are not exact for the same reason as increasing Insertion",
            ),
        ),
        dsl_mapping(
            "selection_increasing_active",
            "selection",
            "exact_active_unfrozen_subset",
            "increasing",
            (
                "Ideal target exists and is active",
                "No frozen target interaction in the tested decision",
                "Increasing sort direction",
            ),
            (
                "Frozen ideal-target branch is not exact because the public method can update ideal_position and attempt a frozen swap in one call",
            ),
        ),
        dsl_mapping(
            "selection_decreasing_active",
            "selection",
            "exact_active_unfrozen_subset",
            "decreasing",
            (
                "Ideal target exists and is active",
                "No frozen target interaction in the tested decision",
                "Decreasing sort direction",
            ),
            (
                "Frozen ideal-target branch is not exact for the same reason as increasing Selection",
            ),
        ),
    ]
    ids = [mapping.policy_id for mapping in mappings]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Classic policy mappings must have unique policy IDs")
    return mappings


def classic_policy_library() -> dict[str, Any]:
    mappings = classic_policy_mappings()
    return {
        "schema": "eidosoma.e03.classic_policy_library.v1",
        "experimentId": "E03",
        "researchStepId": "S03",
        "policyCount": len(mappings),
        "policies": [mapping.to_dict() for mapping in mappings],
    }


def _build_public_cells(config: SimulatorConfig) -> tuple[list[Any], EventTracingStatusProbe]:
    probe = EventTracingStatusProbe()
    cells, _cell_status = _build_cells(config, probe)
    return cells, probe


def _probe_counts(probe: EventTracingStatusProbe) -> tuple[int, int, int]:
    return (int(probe.compare_and_swap_count), int(probe.swap_count), int(probe.frozen_swap_attempts))


def validate_interface_mapping(case_name: str, config: SimulatorConfig, actor_index: int, seed: int) -> dict[str, Any]:
    direct_cells, direct_probe = _build_public_cells(config)
    wrapper_cells, wrapper_probe = _build_public_cells(config)
    random.seed(seed)
    direct_cells[actor_index].move()
    direct_signature = cells_signature(direct_cells)
    random.seed(seed)
    wrapper = policy_for_cell(wrapper_cells[actor_index], config.label_to_behavior)
    result = wrapper.step(wrapper_cells[actor_index])
    wrapper_signature = cells_signature(wrapper_cells)
    success = direct_signature == wrapper_signature and _probe_counts(direct_probe) == _probe_counts(wrapper_probe)
    return {
        "validation_case": case_name,
        "algorithm": wrapper.behavior,
        "representation_type": "interface_wrapper",
        "exactness": "exact_public_method",
        "policy_id": interface_mapping(wrapper.behavior).policy_id,
        "success": bool(success),
        "expected_match": True,
        "observed_match": bool(success),
        "original_action": result.applied_action.action_type,
        "mapped_action": result.applied_action.action_type,
        "original_target_index": result.applied_action.target_index,
        "mapped_target_index": result.applied_action.target_index,
        "original_compare_counted": result.applied_action.compare_counted,
        "mapped_compare_counted": result.applied_action.compare_counted,
        "detail": "Interface wrapper matched direct public move() signature and counters.",
    }


def _dsl_action_for_config(source: str, config: SimulatorConfig, actor_index: int, seed: int) -> tuple[Any, Any]:
    cells, _probe = _build_public_cells(config)
    observation = observe_cell(cells[actor_index], config.label_to_behavior)
    policy = parse_policy(source)
    action = DSLInterpreter(policy).propose(observation, random.Random(seed))
    return policy, action


def _original_action_for_config(config: SimulatorConfig, actor_index: int, seed: int) -> Any:
    cells, _probe = _build_public_cells(config)
    random.seed(seed)
    wrapper = OriginalCellPolicyWrapper.from_cell(cells[actor_index], config.label_to_behavior)
    return wrapper.step(cells[actor_index]).proposed_action


def validate_dsl_exact_case(
    case_name: str,
    source: str,
    config: SimulatorConfig,
    actor_index: int,
    seed: int,
    expected_algorithm: str,
) -> dict[str, Any]:
    policy, mapped = _dsl_action_for_config(source, config, actor_index, seed)
    original = _original_action_for_config(config, actor_index, seed)
    target_matches = original.target_index == mapped.target_index or (
        original.action_type == "wait" and mapped.action_type == "wait"
    )
    observed_match = (
        original.action_type == mapped.action_type
        and target_matches
        and original.compare_counted == mapped.compare_counted
        and dict(original.state_update) == dict(mapped.state_update)
    )
    return {
        "validation_case": case_name,
        "algorithm": expected_algorithm,
        "representation_type": "dsl",
        "exactness": "exact_active_unfrozen_subset",
        "policy_id": policy.policy_id,
        "success": bool(observed_match),
        "expected_match": True,
        "observed_match": bool(observed_match),
        "original_action": original.action_type,
        "mapped_action": mapped.action_type,
        "original_target_index": original.target_index,
        "mapped_target_index": mapped.target_index,
        "original_compare_counted": original.compare_counted,
        "mapped_compare_counted": mapped.compare_counted,
        "detail": "DSL action matched S01 original wrapper on an active/no-frozen fixture.",
    }


def validate_documented_deviation(
    case_name: str,
    source: str,
    config: SimulatorConfig,
    actor_index: int,
    seed: int,
    expected_algorithm: str,
    detail: str,
) -> dict[str, Any]:
    policy, mapped = _dsl_action_for_config(source, config, actor_index, seed)
    original = _original_action_for_config(config, actor_index, seed)
    observed_match = (
        original.action_type == mapped.action_type
        and original.target_index == mapped.target_index
        and original.compare_counted == mapped.compare_counted
        and dict(original.state_update) == dict(mapped.state_update)
    )
    return {
        "validation_case": case_name,
        "algorithm": expected_algorithm,
        "representation_type": "dsl",
        "exactness": "documented_deviation",
        "policy_id": policy.policy_id,
        "success": not observed_match,
        "expected_match": False,
        "observed_match": bool(observed_match),
        "original_action": original.action_type,
        "mapped_action": mapped.action_type,
        "original_target_index": original.target_index,
        "mapped_target_index": mapped.target_index,
        "original_compare_counted": original.compare_counted,
        "mapped_compare_counted": mapped.compare_counted,
        "detail": detail,
    }


def validation_cases() -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        validate_interface_mapping("interface_bubble_public_parity", SimulatorConfig(values=(2, 1), algorithm="bubble"), 0, 1),
        validate_interface_mapping("interface_insertion_public_parity", SimulatorConfig(values=(2, 1, 3), algorithm="insertion"), 1, 11),
        validate_interface_mapping("interface_selection_public_parity", SimulatorConfig(values=(2, 1, 3), algorithm="selection"), 1, 21),
        validate_dsl_exact_case(
            "dsl_insertion_increasing_swap",
            INSERTION_INCREASING_DSL,
            SimulatorConfig(values=(2, 1, 3), algorithm="insertion"),
            1,
            101,
            "insertion",
        ),
        validate_dsl_exact_case(
            "dsl_insertion_increasing_wait",
            INSERTION_INCREASING_DSL,
            SimulatorConfig(values=(1, 2, 3), algorithm="insertion"),
            1,
            102,
            "insertion",
        ),
        validate_dsl_exact_case(
            "dsl_insertion_decreasing_swap",
            INSERTION_DECREASING_DSL,
            SimulatorConfig(values=(1, 2, 3), algorithm="insertion", reverse_directions=(True, True, True)),
            1,
            103,
            "insertion",
        ),
        validate_dsl_exact_case(
            "dsl_selection_increasing_swap",
            SELECTION_INCREASING_DSL,
            SimulatorConfig(values=(2, 1, 3), algorithm="selection"),
            1,
            104,
            "selection",
        ),
        validate_dsl_exact_case(
            "dsl_selection_increasing_update",
            SELECTION_INCREASING_DSL,
            SimulatorConfig(values=(1, 2, 3), algorithm="selection"),
            1,
            105,
            "selection",
        ),
        validate_dsl_exact_case(
            "dsl_selection_decreasing_swap",
            SELECTION_DECREASING_DSL,
            SimulatorConfig(values=(2, 1, 3), algorithm="selection", reverse_directions=(True, True, True)),
            1,
            106,
            "selection",
        ),
        validate_documented_deviation(
            "dsl_bubble_sampled_side_deviation",
            BUBBLE_INCREASING_SHADOW_DSL,
            SimulatorConfig(values=(3, 1, 2), algorithm="bubble"),
            1,
            1,
            "bubble",
            "With seed 1 the public Bubble method samples the right side and waits; the DSL shadow falls through to a left swap, documenting why exact Bubble remains interface-level.",
        ),
    ]
    return pd.DataFrame(rows)
