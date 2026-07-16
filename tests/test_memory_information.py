from __future__ import annotations

import inspect

import numpy as np

from src.detours.context_sensitivity import (
    result_dict as s12_result_dict,
    simulate_population as simulate_s12,
)
from src.detours.memory_information import (
    ARM_NAMES,
    KIND_NOOP,
    RECENT_LEFT,
    arm_flags,
    capability_proposal,
    complexity_rows,
    result_dict,
    simulate_population,
)


def _one_run(arm: str, *, occupancy: tuple[int, ...] = (3, 2, 1, 0)):
    n = len(occupancy)
    arm_code = ARM_NAMES.index(arm)
    return result_dict(
        simulate_population(
            np.asarray([n], dtype=np.int16),
            np.asarray([[0] * n], dtype=np.int8),
            np.zeros((1, n), dtype=np.int8),
            np.asarray([occupancy], dtype=np.int16),
            np.full((1, n), -32768, dtype=np.int16),
            np.asarray([0], dtype=np.int8),
            np.asarray([1], dtype=np.int8),
            np.asarray([1234567], dtype=np.uint64),
            np.asarray([256], dtype=np.int32),
            np.asarray([arm_code], dtype=np.int8),
            np.asarray([0], dtype=np.int32),
        )
    )


def test_complexity_ledger_matches_frozen_arm_product() -> None:
    rows = {row["arm"]: row for row in complexity_rows()}
    assert set(rows) == set(ARM_NAMES)
    assert rows["native"]["mutable_state_count_per_actor"] == 1
    assert rows["failure_bit"]["declared_stored_bits_per_actor"] == 1
    assert rows["counter_2bit"]["mutable_state_count_per_actor"] == 4
    assert rows["recent_direction"]["mutable_state_count_per_actor"] == 3
    assert rows["full"]["mutable_state_count_per_actor"] == 24
    assert rows["full"]["minimum_bits_per_actor"] == 5
    assert rows["full"]["declared_stored_bits_per_actor"] == 5
    assert rows["radius2"]["added_sensing_radius"] == 2


def test_native_arm_matches_s12_kernel_exactly() -> None:
    n = 5
    ns = np.asarray([n], dtype=np.int16)
    policies = np.asarray([[0, 1, 2, 0, 1]], dtype=np.int8)
    faults = np.asarray([[0, 0, 1, 0, 0]], dtype=np.int8)
    occupancy = np.asarray([[4, 0, 3, 1, 2]], dtype=np.int16)
    cursors = np.full((1, n), -32768, dtype=np.int16)
    directions = np.asarray([0], dtype=np.int8)
    widths = np.asarray([2], dtype=np.int8)
    roots = np.asarray([987654321], dtype=np.uint64)
    budgets = np.asarray([200], dtype=np.int32)
    checkpoints = np.asarray([0], dtype=np.int32)
    s12 = s12_result_dict(
        simulate_s12(
            ns,
            policies,
            faults,
            occupancy,
            directions,
            widths,
            roots,
            budgets,
            np.asarray([-1], dtype=np.int32),
            np.asarray([-1], dtype=np.int32),
            np.asarray([-1], dtype=np.int32),
            checkpoints,
        )
    )
    s13 = result_dict(
        simulate_population(
            ns,
            policies,
            faults,
            occupancy,
            cursors,
            directions,
            widths,
            roots,
            budgets,
            np.asarray([0], dtype=np.int8),
            checkpoints,
        )
    )
    for key in (
        "stop",
        "events",
        "ledger",
        "initial",
        "final",
        "peak",
        "episode_count",
        "recovered_count",
        "open_episode",
        "max_depth",
        "max_duration",
        "worsening_actions",
        "physical_fingerprint",
        "final_occupancy",
        "final_cursors",
    ):
        s12_key = "fingerprint" if key == "physical_fingerprint" else key
        assert np.array_equal(s13[key], s12[s12_key]), key


def test_state_reset_and_run_order_independence() -> None:
    first = _one_run("full")
    _one_run("counter_2bit", occupancy=(2, 3, 0, 1))
    second = _one_run("full")
    for key in (
        "stop",
        "events",
        "ledger",
        "capability_audit",
        "full_fingerprint",
        "final_failure",
        "final_counter",
        "final_recent",
    ):
        assert np.array_equal(first[key], second[key]), key


def test_radius_two_overlay_ignores_outside_sentinels() -> None:
    occupancy_a = np.asarray([6, 4, 2, 3, 1, 5, 0], dtype=np.int16)
    occupancy_b = np.asarray([0, 4, 2, 3, 1, 5, 6], dtype=np.int16)
    n = len(occupancy_a)
    policies = np.zeros(n, dtype=np.int8)
    faults = np.zeros(n, dtype=np.int8)
    cursors = np.full(n, -32768, dtype=np.int16)
    zeros = np.zeros(n, dtype=np.uint8)
    actor = 3
    args = (cursors, policies, faults, zeros, zeros, zeros, n, 0, actor, 0, 0, 0, 0, 1)
    proposal_a = capability_proposal.py_func(occupancy_a, *args)
    proposal_b = capability_proposal.py_func(occupancy_b, *args)
    assert proposal_a == proposal_b


def test_recent_direction_veto_is_actor_local() -> None:
    occupancy = np.asarray([0, 2, 1, 3], dtype=np.int16)
    n = len(occupancy)
    policies = np.zeros(n, dtype=np.int8)
    faults = np.zeros(n, dtype=np.int8)
    cursors = np.full(n, -32768, dtype=np.int16)
    failure = np.zeros(n, dtype=np.uint8)
    counter = np.zeros(n, dtype=np.uint8)
    recent = np.zeros(n, dtype=np.uint8)
    actor = 2  # identity 2 is at position 1 and can move right into identity 1
    recent[actor] = RECENT_LEFT
    proposal = capability_proposal.py_func(
        occupancy,
        cursors,
        policies,
        faults,
        failure,
        counter,
        recent,
        n,
        0,
        actor,
        1,
        0,
        0,
        1,
        0,
    )
    assert proposal[0] == KIND_NOOP
    assert proposal[9] == 1


def test_failure_bit_flips_only_bubble_side() -> None:
    occupancy = np.asarray([2, 0, 1], dtype=np.int16)
    n = len(occupancy)
    policies = np.zeros(n, dtype=np.int8)
    faults = np.zeros(n, dtype=np.int8)
    cursors = np.full(n, -32768, dtype=np.int16)
    failure = np.zeros(n, dtype=np.uint8)
    counter = np.zeros(n, dtype=np.uint8)
    recent = np.zeros(n, dtype=np.uint8)
    actor = 0  # at position 1; raw right is ordered, flipped left is inverted
    failure[actor] = 1
    proposal = capability_proposal.py_func(
        occupancy,
        cursors,
        policies,
        faults,
        failure,
        counter,
        recent,
        n,
        0,
        actor,
        1,
        1,
        0,
        0,
        0,
    )
    assert proposal[0] != KIND_NOOP
    assert proposal[6] == 1


def test_capability_gateway_has_no_outcome_inputs() -> None:
    signature = set(inspect.signature(capability_proposal.py_func).parameters)
    forbidden_parameters = {
        "metrics",
        "metric_levels",
        "reachability",
        "terminal",
        "analysis_labels",
        "future_draws",
        "goal_ranks",
    }
    assert signature.isdisjoint(forbidden_parameters)
    source = inspect.getsource(capability_proposal.py_func)
    for forbidden in (
        "metric_profile",
        "running_min",
        "reachability",
        "terminal_code",
        "future",
    ):
        assert forbidden not in source


def test_arm_flags_are_stable() -> None:
    assert arm_flags("native") == (0, 0, 0, 0)
    assert arm_flags("full") == (1, 1, 1, 1)
    assert arm_flags("full_minus_radius2") == (1, 1, 1, 0)
