from __future__ import annotations

import numpy as np

from analysis.conflict_phase_map import (
    TRACE_COLUMNS,
    _initial_order_score,
    build_conditions,
    classify_trace,
    materialize_arrays,
    schedule_choice,
    simulate_phase_kernel,
    wilson_interval,
)


def test_factorial_is_exact_and_opposing() -> None:
    conditions = build_conditions()
    assert len(conditions) == 4500
    assert len({condition.condition_id for condition in conditions}) == 4500
    assert all(condition.first_direction != condition.second_direction for condition in conditions)
    assert {condition.first_count for condition in conditions} == {20, 35, 50, 65, 80}
    assert {condition.activation_ratio for condition in conditions} == {0.25, 0.5, 1, 2, 4}
    assert {condition.fault_count for condition in conditions} == {0, 5, 10}


def test_materialization_counts_association_and_disorder() -> None:
    conditions = build_conditions()
    examples = {}
    for condition in conditions:
        if (
            condition.input_profile == "unique_1_100"
            and condition.first_count == 35
            and condition.activation_ratio == 1
            and condition.fault_count == 10
            and condition.first_policy == "Bubble"
            and condition.second_policy == "Insertion"
            and condition.first_direction == "ascending"
        ):
            examples[condition.state_profile] = condition
    assert len(examples) == 5
    scores = {}
    for profile, condition in examples.items():
        static, arrays = materialize_arrays(condition, 0)
        values, policy, _, directions, faults, occupancy, _, p1_ids, p2_ids = arrays
        assert len(p1_ids) == 35 and len(p2_ids) == 65
        assert faults.sum() == 10
        assert sorted(occupancy.tolist()) == list(range(100))
        assert set(directions.tolist()) == {-1, 1}
        scores[profile] = _initial_order_score(values, occupancy)
        if profile == "random_positive":
            assert static["policy_value_spearman_rho"] > 0.5
        elif profile == "random_negative":
            assert static["policy_value_spearman_rho"] < -0.5
    assert scores["ascending_blocks_absent"] > 0.75
    assert scores["descending_blocks_absent"] < -0.75
    assert abs(scores["random_absent"]) < 0.25


def test_counter_schedule_prefix_and_frequency() -> None:
    condition = next(
        item
        for item in build_conditions()
        if item.first_count == 35 and item.activation_ratio == 2
    )
    static, arrays = materialize_arrays(condition, 1)
    p1_ids, p2_ids = arrays[-2], arrays[-1]
    prefix = [
        schedule_choice(int(static["schedule_seed"]), event, p1_ids, p2_ids, 2)[0]
        for event in range(20_000)
    ]
    repeated = [
        schedule_choice(int(static["schedule_seed"]), event, p1_ids, p2_ids, 2)[0]
        for event in range(20_000)
    ]
    assert prefix == repeated
    achieved = np.mean(np.isin(prefix, p1_ids))
    assert abs(achieved - static["expected_first_activation_share"]) < 0.015


def test_kernel_determinism_conservation_and_stuck_positions() -> None:
    condition = next(
        item
        for item in build_conditions()
        if item.input_profile == "unique_1_100"
        and item.first_policy == "Bubble"
        and item.second_policy == "Selection"
        and item.first_count == 50
        and item.activation_ratio == 1
        and item.fault_count == 10
        and item.state_profile == "random_positive"
    )
    static, arrays = materialize_arrays(condition, 2)
    result1 = simulate_phase_kernel(
        *arrays,
        condition.activation_ratio,
        np.uint64(int(static["schedule_seed"])),
        10_000,
    )
    result2 = simulate_phase_kernel(
        *arrays,
        condition.activation_ratio,
        np.uint64(int(static["schedule_seed"])),
        10_000,
    )
    for left, right in zip(result1, result2):
        assert np.array_equal(left, right)
    trace, final_occupancy, _, _ = result1
    initial_occupancy = arrays[5]
    faults = arrays[4]
    initial_positions = np.argsort(initial_occupancy)
    final_positions = np.argsort(final_occupancy)
    assert np.array_equal(initial_positions[faults == 1], final_positions[faults == 1])
    assert abs(np.sum(trace[:, 12:14])) < 1e-12


def test_regime_classifier_fixtures() -> None:
    base = np.zeros((101, len(TRACE_COLUMNS)))
    assert classify_trace(base, True)["regime"] == "fixed_quiescence"
    dominant = base.copy()
    dominant[:71, 0] = np.linspace(0, 0.9, 71)
    dominant[71:, 0] = 0.9
    dominant[:, 7] = 0.002
    assert classify_trace(dominant, False)["regime"] == "active_dominance"
    equilibrium = base.copy()
    equilibrium[:, 0] = 0.1
    equilibrium[:, 2] = -0.1
    equilibrium[:, 7] = 0.002
    assert classify_trace(equilibrium, False)["regime"] == "dynamic_equilibrium"
    oscillation = base.copy()
    oscillation[:, 0] = 0.25 * np.sin(2 * np.pi * np.arange(101) / 10)
    oscillation[:, 7] = 0.002
    assert classify_trace(oscillation, False)["regime"] == "oscillation"
    metastable = base.copy()
    metastable[:, 7] = 0.003
    metastable[25:45, 7] = 0.0002
    metastable[45:, 0] = 0.35
    assert classify_trace(metastable, False)["regime"] == "metastability"


def test_wilson_interval_contains_observed_fraction() -> None:
    lower, upper = wilson_interval(8, 12)
    assert lower < 8 / 12 < upper
