from __future__ import annotations

import numpy as np

from analysis.dynamic_nulls import (
    NULL_FAMILIES,
    TOTAL_DRAWS,
    _bh_adjust,
    _cycle_signature,
    _label_curves,
    _permuted_labels,
    _policy_neutral_curves,
    _value_strata,
    mobility_strata,
    trajectory_outcomes,
)


def test_piecewise_linear_duration_and_positive_area() -> None:
    rising = np.linspace(-1.0, 1.0, 101)
    falling = -rising
    zero = np.zeros(101)
    result = trajectory_outcomes(np.stack([rising, falling, zero]))
    np.testing.assert_allclose(result["peak"], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(result["duration"], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(result["positive_area"], [0.25, 0.25, 0.0])


def test_global_and_stratified_permutations_preserve_counts() -> None:
    labels = np.repeat(np.arange(3, dtype=np.uint8), [34, 33, 33])
    strata = np.repeat(np.arange(5, dtype=np.uint8), 20)
    for frozen_strata in (None, strata):
        rng = np.random.Generator(np.random.PCG64DXSM(12345))
        draws = _permuted_labels(labels, frozen_strata, rng)
        assert draws.shape == (TOTAL_DRAWS, 100)
        for code, count in enumerate((34, 33, 33)):
            np.testing.assert_array_equal(np.sum(draws == code, axis=1), count)
        if frozen_strata is not None:
            for stratum in range(5):
                index = np.flatnonzero(strata == stratum)
                target = [np.sum(labels[index] == code) for code in range(3)]
                for code in range(3):
                    np.testing.assert_array_equal(
                        np.sum(draws[:, index] == code, axis=1), target[code]
                    )


def test_label_curve_matches_hand_count() -> None:
    occupancy = np.tile(np.arange(4, dtype=np.uint8), (101, 1))
    assignments = np.tile(np.asarray([0, 0, 1, 1], dtype=np.uint8), (TOTAL_DRAWS, 1))
    curves = _label_curves(assignments, occupancy, baseline=0.25)
    # Two of three open edges match; publication denominator is four.
    np.testing.assert_allclose(curves, 0.25)


def test_mobility_strata_are_equal_size_and_deterministic() -> None:
    occupancy = np.tile(np.arange(100, dtype=np.uint8), (101, 1))
    for grid_index in range(1, 101):
        occupancy[grid_index] = np.roll(occupancy[grid_index], grid_index % 7)
    first = mobility_strata(occupancy)
    second = mobility_strata(occupancy)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(np.bincount(first[0]), np.repeat(20, 5))
    assert np.all(first[1] >= 0)
    assert np.all(first[2] >= 0)


def test_frozen_value_strata_match_s03_profile_names() -> None:
    repeated = np.repeat(np.arange(1, 11), 10)
    unique = np.arange(1, 101)
    np.testing.assert_array_equal(
        _value_strata(repeated, "repeated_1_10_x10"), repeated
    )
    np.testing.assert_array_equal(
        np.bincount(_value_strata(unique, "unique_1_100")), np.repeat(10, 10)
    )


def test_policy_neutral_transport_is_deterministic_and_starts_observed() -> None:
    n = 8
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    occupancy = np.empty((101, n), dtype=np.uint8)
    for grid_index in range(101):
        occupancy[grid_index] = np.roll(np.arange(n, dtype=np.uint8), grid_index % n)
    rng_a = np.random.Generator(np.random.PCG64DXSM(54321))
    rng_b = np.random.Generator(np.random.PCG64DXSM(54321))
    first, first_cycles, first_moved = _policy_neutral_curves(
        labels, occupancy, 0.375, rng_a
    )
    second, second_cycles, second_moved = _policy_neutral_curves(
        labels, occupancy, 0.375, rng_b
    )
    np.testing.assert_array_equal(first, second)
    assert first_cycles == second_cycles == 0
    assert first_moved == second_moved == 0
    initial_raw = np.sum(labels[:-1] == labels[1:]) / n
    np.testing.assert_allclose(first[:, 0], initial_raw - 0.375)


def test_cycle_signature_is_conjugacy_invariant() -> None:
    mapping = np.asarray([1, 2, 0, 4, 3, 5])
    permutation = np.asarray([5, 2, 1, 3, 0, 4])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    conjugated = permutation[mapping[inverse]]
    assert _cycle_signature(mapping) == (1, 2, 3)
    assert _cycle_signature(conjugated) == _cycle_signature(mapping)


def test_bh_adjustment_is_monotone_and_bounded() -> None:
    p = np.asarray([0.01, 0.04, 0.03, 0.002, np.nan])
    q = _bh_adjust(p)
    assert np.isnan(q[-1])
    assert np.all((q[:-1] >= p[:-1]) & (q[:-1] <= 1))
    order = np.argsort(p[:-1])
    assert np.all(np.diff(q[:-1][order]) >= 0)


def test_frozen_family_set_is_complete() -> None:
    assert NULL_FAMILIES == (
        "label_permuted_global",
        "label_permuted_value_stratified",
        "mobility_matched",
        "policy_neutral_transport",
    )
