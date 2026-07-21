from __future__ import annotations

import ast
from copy import deepcopy
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from reference_simulator.engine import initial_state
from reference_simulator.model import (
    Cell,
    Direction,
    FaultMode,
    Policy,
    ProposalKind,
    Scenario,
)
from reference_simulator.policies import _prefix_is_ordered, cell_view_proposal
from src.policy_dsl import (
    BASELINE_DIRECTORY,
    PolicyValidationError,
    canonical_policy_bytes,
    compile_policy,
    execute_policy,
    load_policy,
    parse_policy,
    policy_to_dict,
)


BASELINES = {
    path.stem: load_policy(path) for path in sorted(BASELINE_DIRECTORY.glob("*.json"))
}


def _line_observation(policy, scenario, state, actor_id, *, side="none"):
    cells = scenario.cell_map
    actor = cells[actor_id]
    position = state.occupancy.index(actor_id)
    left = cells[state.occupancy[position - 1]] if position > 0 else None
    right = (
        cells[state.occupancy[position + 1]]
        if position + 1 < len(state.occupancy)
        else None
    )
    prefix, _, _ = _prefix_is_ordered(scenario, state, position, actor.direction)
    cursor = state.selection_cursors.get(actor_id, -1)
    cursor_in_bounds = 0 <= cursor < len(state.occupancy)
    target = cells[state.occupancy[cursor]] if cursor_in_bounds else None
    available = {
        "activation.side": side,
        "own.value": int(actor.value),
        "own.position": position,
        "own.direction": actor.direction.value,
        "neighbor.left.exists": left is not None,
        "neighbor.left.value": int(left.value) if left is not None else 0,
        "neighbor.left.movable": left is not None and left.fault != FaultMode.STUCK,
        "neighbor.right.exists": right is not None,
        "neighbor.right.value": int(right.value) if right is not None else 0,
        "neighbor.right.movable": right is not None and right.fault != FaultMode.STUCK,
        "line.prefix_ordered": prefix,
        "selection.cursor_in_bounds": cursor_in_bounds,
        "selection.cursor_at_actor": cursor == position,
        "selection.target.value": int(target.value) if target is not None else 0,
        "selection.target.stuck": target is not None
        and target.fault == FaultMode.STUCK,
        "last_action.rejected": False,
        "repair.nudge_count": 0,
        "signal.neighbor_sum_u8": 0,
        "counter.choice_u8": 0,
    }
    return {
        "scalars": {
            name: available[name]
            for name in sorted(policy.permissions)
            if not name.startswith("candidate.")
        },
        "candidates": [],
    }


def _assert_equivalent(reference, dsl, state, actor_id):
    terminal = [
        action
        for action in dsl.actions
        if action["kind"]
        in {
            "noop",
            "swap_relative",
            "swap_cursor",
            "advance_cursor",
            "move_candidate",
        }
    ]
    assert len(terminal) == 1
    action = terminal[0]
    if reference.kind == ProposalKind.NO_OP:
        assert action["kind"] == "noop"
    elif reference.kind == ProposalKind.SWAP:
        if action["kind"] == "swap_relative":
            assert (
                state.occupancy.index(actor_id) + action["offset"]
                == reference.target_pos
            )
        else:
            assert action["kind"] == "swap_cursor"
            assert state.selection_cursors[actor_id] == reference.target_pos
    else:
        assert reference.kind == ProposalKind.MEMORY_UPDATE
        assert action["kind"] == "advance_cursor"
        assert (
            state.selection_cursors[actor_id] + action["delta"] == reference.new_cursor
        )


def test_all_baselines_round_trip_compile_and_have_stable_hashes():
    assert set(BASELINES) == {
        "bubble_cell_view_v1",
        "insertion_cell_view_v1",
        "nudge_signal_repair_v1",
        "selection_cell_view_v1",
        "spatial_greedy_local_v1",
        "spatial_memory_repair_v1",
    }
    for path in sorted(BASELINE_DIRECTORY.glob("*.json")):
        first = load_policy(path)
        decoded = policy_to_dict(first)
        reversed_keys = {key: decoded[key] for key in reversed(decoded)}
        second = compile_policy(reversed_keys)
        third = compile_policy(first.canonical_json)
        assert first.canonical_json == canonical_policy_bytes(decoded)
        assert first.canonical_json == second.canonical_json == third.canonical_json
        assert first.policy_sha256 == second.policy_sha256 == third.policy_sha256
        assert first.complexity.worst_case_operations <= first.max_operations


def test_validation_cli_emits_machine_readable_hashes():
    paths = sorted(BASELINE_DIRECTORY.glob("*.json"))
    completed = subprocess.run(
        [sys.executable, "-m", "src.policy_dsl", "validate", *map(str, paths)],
        check=True,
        capture_output=True,
        text=True,
    )
    records = json.loads(completed.stdout)
    assert [item["policyId"] for item in records] == [
        load_policy(path).policy_id for path in paths
    ]
    assert all(item["valid"] and len(item["policySha256"]) == 64 for item in records)


@pytest.mark.parametrize(
    ("policy_name", "reference_policy"),
    [
        ("bubble_cell_view_v1", Policy.BUBBLE),
        ("insertion_cell_view_v1", Policy.INSERTION),
        ("selection_cell_view_v1", Policy.SELECTION),
    ],
)
def test_seeded_1d_policies_match_reference_on_2400_randomized_activations(
    policy_name, reference_policy
):
    rng = random.Random(0xE0701 + list(Policy).index(reference_policy))
    dsl_policy = BASELINES[policy_name]
    for case in range(800):
        size = rng.randint(2, 10)
        direction = rng.choice(tuple(Direction))
        actor_index = rng.randrange(size)
        actor_id = f"c{actor_index}"
        values = [rng.randint(-4, 12) for _ in range(size)]
        cells = []
        for index, value in enumerate(values):
            fault = (
                FaultMode.NORMAL
                if index == actor_index
                else rng.choice(tuple(FaultMode))
            )
            cells.append(
                Cell(
                    f"c{index}",
                    value,
                    reference_policy
                    if index == actor_index
                    else rng.choice(tuple(Policy)),
                    direction,
                    fault,
                )
            )
        scenario = Scenario.create(
            cells,
            initial_occupancy=tuple(f"c{index}" for index in range(size)),
            generation_key=f"E07/S01/equivalence/{policy_name}/{case}",
        )
        state = initial_state(scenario)
        if reference_policy == Policy.SELECTION:
            state.selection_cursors[actor_id] = rng.randrange(size)
            if case % 11 == 0:
                state.selection_cursors[actor_id] = -1 if case % 22 == 0 else size
        side = (
            rng.choice(("left", "right"))
            if reference_policy == Policy.BUBBLE
            else "none"
        )
        reference = cell_view_proposal(
            scenario,
            state,
            actor_id,
            side=side if reference_policy == Policy.BUBBLE else None,
        )
        observation = _line_observation(
            dsl_policy, scenario, state, actor_id, side=side
        )
        result = execute_policy(dsl_policy, observation)
        _assert_equivalent(reference, result, state, actor_id)


def test_spatial_selector_is_bounded_positive_and_tie_deterministic():
    policy = BASELINES["spatial_greedy_local_v1"]
    candidates = [
        {
            "key": "opaque-z",
            "candidate.kind": "adjacent_swap",
            "candidate.local_relation_delta": 4,
        },
        {
            "key": "opaque-a",
            "candidate.kind": "vacancy_move",
            "candidate.local_relation_delta": 4,
        },
        {
            "key": "opaque-r",
            "candidate.kind": "rotation",
            "candidate.local_relation_delta": 99,
        },
    ]
    result = execute_policy(policy, {"scalars": {}, "candidates": candidates})
    assert result.actions[0] == {
        "kind": "move_candidate",
        "candidateKey": "opaque-a",
        "movementKind": "vacancy_move",
        "score": 4,
    }
    assert result.operation_count == 5  # condition + action + three bounded scans
    nonpositive = deepcopy(candidates[:2])
    for item in nonpositive:
        item["candidate.local_relation_delta"] = 0
    assert (
        execute_policy(policy, {"scalars": {}, "candidates": nonpositive}).actions[0][
            "kind"
        ]
        == "noop"
    )


def test_memory_and_signal_updates_are_saturating_and_explicit():
    policy = BASELINES["nudge_signal_repair_v1"]
    observation = {
        "scalars": {
            "last_action.rejected": True,
            "repair.nudge_count": 0,
            "signal.neighbor_sum_u8": 0,
        },
        "candidates": [],
    }
    state = {"frustration": 3}
    result = execute_policy(policy, observation, state)
    assert result.memory == {"frustration": 3}
    assert result.emitted_signals == {0: 3}
    assert [item["kind"] for item in result.actions] == [
        "set_memory",
        "emit_signal",
        "noop",
    ]


def test_permission_gate_rejects_hidden_or_extra_fields_and_candidate_routes():
    bubble = BASELINES["bubble_cell_view_v1"]
    raw = policy_to_dict(bubble)
    raw["permissions"].append("global.completion_score")
    raw["permissions"].sort()
    with pytest.raises(PolicyValidationError, match="unknown or forbidden"):
        compile_policy(raw)

    scenario = Scenario.create(
        [Cell("a", 2, Policy.BUBBLE), Cell("b", 1, Policy.BUBBLE)],
        generation_key="E07/S01/permission",
    )
    state = initial_state(scenario)
    observation = _line_observation(bubble, scenario, state, "a", side="right")
    observation["scalars"]["analysisLabel"] = "protected"
    with pytest.raises(PolicyValidationError, match="exactly equal"):
        execute_policy(bubble, observation)

    spatial = BASELINES["spatial_greedy_local_v1"]
    candidate = {
        "key": "opaque",
        "candidate.kind": "adjacent_swap",
        "candidate.local_relation_delta": 1,
        "route": ["secret-site-a", "secret-site-b"],
    }
    with pytest.raises(PolicyValidationError, match="keys mismatch"):
        execute_policy(spatial, {"scalars": {}, "candidates": [candidate]})


def test_parser_rejects_duplicate_keys_nonfinite_numbers_and_dynamic_action_names():
    with pytest.raises(PolicyValidationError, match="duplicate JSON key"):
        parse_policy('{"schemaVersion":"x","schemaVersion":"y"}')
    with pytest.raises(PolicyValidationError, match="non-finite"):
        parse_policy('{"x":NaN}')
    raw = policy_to_dict(BASELINES["bubble_cell_view_v1"])
    raw["rules"][0]["actions"][0] = {"kind": "python_exec", "source": "pass"}
    with pytest.raises(PolicyValidationError, match="unsupported action"):
        compile_policy(raw)


def test_runtime_has_no_dynamic_code_or_process_execution_calls():
    source = Path("src/policy_dsl/core.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"eval", "exec", "compile", "__import__", "open", "system", "popen"}
    direct_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (direct_calls & forbidden)


def test_static_and_runtime_operation_and_candidate_budgets_are_enforced():
    spatial = BASELINES["spatial_greedy_local_v1"]
    raw = policy_to_dict(spatial)
    raw["limits"]["maxOperationsPerActivation"] = 17
    with pytest.raises(PolicyValidationError, match="static worst-case"):
        compile_policy(raw)

    candidate = {
        "key": "k00",
        "candidate.kind": "adjacent_swap",
        "candidate.local_relation_delta": 1,
    }
    sixteen = [{**candidate, "key": f"k{index:02d}"} for index in range(16)]
    result = execute_policy(spatial, {"scalars": {}, "candidates": sixteen})
    assert result.operation_count == spatial.complexity.worst_case_operations == 18
    seventeen = [{**candidate, "key": f"k{index:02d}"} for index in range(17)]
    with pytest.raises(PolicyValidationError, match="candidate"):
        execute_policy(spatial, {"scalars": {}, "candidates": seventeen})


def test_deterministic_mutation_fuzz_rejects_5000_malformed_policies():
    rng = random.Random(0xF022E07)
    base = policy_to_dict(BASELINES["bubble_cell_view_v1"])
    mutations = ("permission", "operations", "action", "memory", "expression")
    rejected = 0
    for index in range(5000):
        raw = deepcopy(base)
        mutation = mutations[index % len(mutations)]
        if mutation == "permission":
            raw["permissions"].append(f"holdout.outcome_{rng.randrange(1_000_000)}")
            raw["permissions"].sort()
        elif mutation == "operations":
            raw["limits"]["maxOperationsPerActivation"] = rng.choice((-1, 0, 999))
        elif mutation == "action":
            raw["rules"][rng.randrange(len(raw["rules"]))]["actions"][0] = {
                "kind": rng.choice(("exec", "import", "shell", "network"))
            }
        elif mutation == "memory":
            raw["memory"] = [
                {"name": "state", "bits": rng.randint(17, 99), "initial": 0}
            ]
        else:
            raw["rules"][0]["when"] = {
                "op": "eq",
                "left": {"obs": "analysis.label"},
                "right": {"const": "x"},
            }
        with pytest.raises(PolicyValidationError):
            compile_policy(raw)
        rejected += 1
    assert rejected == 5000


def test_total_interpreter_terminates_over_10000_bounded_activations():
    policy = BASELINES["spatial_greedy_local_v1"]
    candidates = [
        {
            "key": f"opaque-{index:02d}",
            "candidate.kind": "adjacent_swap",
            "candidate.local_relation_delta": index - 8,
        }
        for index in range(16)
    ]
    signatures = set()
    for _ in range(10_000):
        result = execute_policy(policy, {"scalars": {}, "candidates": candidates})
        assert result.operation_count <= policy.max_operations
        signatures.add((result.actions[0]["candidateKey"], result.operation_count))
    assert signatures == {("opaque-15", 18)}


def test_canonical_hashes_do_not_collapse_distinct_rule_order():
    raw = policy_to_dict(BASELINES["bubble_cell_view_v1"])
    swapped = deepcopy(raw)
    swapped["rules"][0], swapped["rules"][1] = swapped["rules"][1], swapped["rules"][0]
    assert compile_policy(raw).policy_sha256 != compile_policy(swapped).policy_sha256


def test_every_policy_bundle_has_one_terminal_effect():
    terminal = {
        "noop",
        "swap_relative",
        "swap_cursor",
        "advance_cursor",
        "move_candidate",
    }
    for policy in BASELINES.values():
        raw = policy_to_dict(policy)
        for actions in [rule["actions"] for rule in raw["rules"]] + [
            raw["default"]["actions"]
        ]:
            assert sum(action["kind"] in terminal for action in actions) == 1


def test_no_baseline_observation_names_overlap_protected_vocabulary():
    forbidden_fragments = {
        "analysis",
        "completion",
        "global",
        "holdout",
        "outcome",
        "scenario",
        "site",
        "split",
    }
    for policy in BASELINES.values():
        assert not any(
            fragment in permission.lower()
            for permission, fragment in itertools.product(
                policy.permissions, forbidden_fragments
            )
        )
