from __future__ import annotations

from collections import defaultdict
import itertools
import json
import math
from pathlib import Path

import pytest

from causal_simulator.scenario_extensions import minimum_duplicate_aware_footrule
from src.detours.distances import (
    adjacent_descents,
    distance_profile,
    duplicate_aware_earth_movers_distance,
    inversion_count,
    maximum_inversion_count,
    maximum_possible_rank_error,
    maximum_rank_error,
    maximum_spearman_footrule,
    normalized_adjacent_descents,
    normalized_kendall_distance,
    paper_sortedness_strict,
    spearman_footrule,
)


FIXTURES = Path(__file__).parent / "fixtures" / "s01_distance_fixtures.json"


def _brute_inversions(values: tuple[int, ...], direction: str) -> int:
    if direction == "ascending":
        return sum(
            values[left] > values[right]
            for left in range(len(values))
            for right in range(left + 1, len(values))
        )
    return sum(
        values[left] < values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


def _brute_goal_displacements(
    values: tuple[int, ...], direction: str
) -> tuple[int, int]:
    current: dict[int, tuple[int, ...]] = {}
    target: dict[int, tuple[int, ...]] = {}
    goal = tuple(sorted(values, reverse=direction == "descending"))
    for value in set(values):
        current[value] = tuple(index for index, item in enumerate(values) if item == value)
        target[value] = tuple(index for index, item in enumerate(goal) if item == value)

    per_class_assignments = [
        tuple(itertools.permutations(target[value])) for value in sorted(current)
    ]
    best_sum = math.inf
    best_max = math.inf
    for assignment_set in itertools.product(*per_class_assignments):
        displacements = [
            abs(position - goal_position)
            for value, assignment in zip(sorted(current), assignment_set)
            for position, goal_position in zip(current[value], assignment)
        ]
        best_sum = min(best_sum, sum(displacements))
        best_max = min(best_max, max(displacements, default=0))
    return int(best_sum), int(best_max)


def _cdf_transport(values: tuple[int, ...], direction: str) -> int:
    goal = tuple(sorted(values, reverse=direction == "descending"))
    total = 0
    for value in set(values):
        current_prefix = target_prefix = 0
        for boundary in range(len(values) - 1):
            current_prefix += values[boundary] == value
            target_prefix += goal[boundary] == value
            total += abs(current_prefix - target_prefix)
    return total


def test_hand_calculated_fixtures() -> None:
    payload = json.loads(FIXTURES.read_text())
    assert payload["schemaVersion"] == "E03.S01.distance-fixtures.v1"
    assert len(payload["fixtures"]) == 9
    for fixture in payload["fixtures"]:
        observed = distance_profile(
            fixture["values"], direction=fixture["direction"]
        ).to_dict()
        assert observed["n"] == len(fixture["values"])
        assert observed["direction"] == fixture["direction"]
        for field, expected in fixture["expected"].items():
            if isinstance(expected, float):
                assert observed[field] == pytest.approx(expected), (
                    fixture["fixtureId"],
                    field,
                )
            else:
                assert observed[field] == expected, (fixture["fixtureId"], field)


def test_exhaustive_unique_permutations_through_n8() -> None:
    count = 0
    for n in range(1, 9):
        maximum = n * (n - 1) // 2
        for values in itertools.permutations(range(n)):
            count += 1
            for direction in ("ascending", "descending"):
                profile = distance_profile(values, direction=direction)
                brute_inversions = _brute_inversions(values, direction)
                goal = tuple(sorted(values, reverse=direction == "descending"))
                goal_position = {value: position for position, value in enumerate(goal)}
                brute_footrule = sum(
                    abs(position - goal_position[value])
                    for position, value in enumerate(values)
                )
                brute_rank_error = max(
                    (
                        abs(position - goal_position[value])
                        for position, value in enumerate(values)
                    ),
                    default=0,
                )
                assert profile.inversion_count == brute_inversions
                assert profile.maximum_inversion_count == maximum
                assert profile.normalized_kendall_distance == pytest.approx(
                    brute_inversions / maximum if maximum else 0.0
                )
                assert profile.spearman_footrule == brute_footrule
                assert profile.maximum_rank_error == brute_rank_error
                assert 0.0 <= profile.normalized_spearman_footrule <= 1.0
                assert 0.0 <= profile.normalized_maximum_rank_error <= 1.0
                assert profile.normalized_duplicate_aware_earth_movers_distance == (
                    pytest.approx(profile.normalized_spearman_footrule)
                )
    assert count == sum(math.factorial(n) for n in range(1, 9)) == 46_233


def test_exhaustive_duplicate_sequences_through_n7() -> None:
    count = 0
    observed_maxima: dict[tuple[int, ...], tuple[int, int]] = {}
    for n in range(1, 8):
        for values in itertools.product(range(3), repeat=n):
            count += 1
            brute_maximum_inversions = sum(
                left != right
                for index, left in enumerate(values)
                for right in values[index + 1 :]
            )
            for direction in ("ascending", "descending"):
                brute_footrule, brute_rank_error = _brute_goal_displacements(
                    values, direction
                )
                assert inversion_count(values, direction=direction) == _brute_inversions(
                    values, direction
                )
                assert maximum_inversion_count(values) == brute_maximum_inversions
                assert spearman_footrule(values, direction=direction) == brute_footrule
                assert maximum_rank_error(values, direction=direction) == brute_rank_error
                assert duplicate_aware_earth_movers_distance(
                    values, direction=direction
                ) == pytest.approx(_cdf_transport(values, direction) / n)
                profile = distance_profile(values, direction=direction)
                assert 0.0 <= profile.normalized_kendall_distance <= 1.0
                assert 0.0 <= profile.normalized_spearman_footrule <= 1.0
                assert 0.0 <= profile.normalized_maximum_rank_error <= 1.0

            multiset = tuple(sorted(values))
            prior_footrule, prior_rank = observed_maxima.get(multiset, (0, 0))
            observed_maxima[multiset] = (
                max(prior_footrule, spearman_footrule(values)),
                max(prior_rank, maximum_rank_error(values)),
            )

    for multiset, (footrule_maximum, rank_maximum) in observed_maxima.items():
        assert maximum_spearman_footrule(multiset) == footrule_maximum
        assert maximum_possible_rank_error(multiset) == rank_maximum
    assert count == sum(3**n for n in range(1, 8)) == 3_279


def test_direction_reversal_and_kendall_complement() -> None:
    for values in itertools.product(range(3), repeat=6):
        reverse = tuple(reversed(values))
        ascending = distance_profile(values, direction="ascending")
        descending_reverse = distance_profile(reverse, direction="descending")
        assert ascending.to_dict() | {"direction": "descending"} == (
            descending_reverse.to_dict()
        )
        assert inversion_count(values, direction="ascending") + inversion_count(
            values, direction="descending"
        ) == maximum_inversion_count(values)


def test_monotone_adjacent_swap_paths() -> None:
    paths = (
        (5, 1, 4, 2, 3),
        (4, 3, 2, 1),
        (3, 1, 2, 1, 3),
    )
    saw_local_regression = False
    for initial in paths:
        values = list(initial)
        previous = distance_profile(values)
        while any(left > right for left, right in zip(values, values[1:])):
            index = next(
                index
                for index, (left, right) in enumerate(zip(values, values[1:]))
                if left > right
            )
            values[index], values[index + 1] = values[index + 1], values[index]
            current = distance_profile(values)
            assert current.inversion_count == previous.inversion_count - 1
            assert current.spearman_footrule <= previous.spearman_footrule
            assert previous.spearman_footrule - current.spearman_footrule in {0, 2}
            assert current.maximum_rank_error <= previous.maximum_rank_error
            saw_local_regression |= current.adjacent_descents > previous.adjacent_descents
            previous = current
        assert previous.inversion_count == previous.spearman_footrule == 0
    assert saw_local_regression


def test_duplicate_goal_identity_and_paper_proxy_boundary() -> None:
    values = (1, 1, 2, 2)
    profile = distance_profile(values)
    assert profile.inversion_count == 0
    assert profile.spearman_footrule == 0
    assert profile.maximum_rank_error == 0
    assert profile.duplicate_aware_earth_movers_distance == 0
    assert paper_sortedness_strict(values) == 0.5
    assert adjacent_descents(values) == 0


def test_prior_e02_duplicate_footrule_contract_remains_compatible() -> None:
    for values in itertools.product(range(3), repeat=5):
        assert spearman_footrule(values) == minimum_duplicate_aware_footrule(values)
        assert spearman_footrule(values, direction="descending") == (
            minimum_duplicate_aware_footrule(values, descending=True)
        )


@pytest.mark.parametrize(
    "values,error",
    (
        ((), ValueError),
        ((1, float("nan")), ValueError),
        ((1, float("inf")), ValueError),
        ((1, True), TypeError),
        ("123", TypeError),
    ),
)
def test_invalid_value_domains_fail_closed(values, error) -> None:
    with pytest.raises(error):
        distance_profile(values)


def test_invalid_direction_fails_closed() -> None:
    with pytest.raises(ValueError, match="direction"):
        distance_profile((1, 2), direction="sideways")
    with pytest.raises(ValueError, match="direction"):
        normalized_adjacent_descents((1,), direction="sideways")
    with pytest.raises(ValueError, match="direction"):
        normalized_kendall_distance((1, 1), direction="sideways")


def test_public_functions_agree_with_profile() -> None:
    values = (3, 1, 3, 2, 1)
    profile = distance_profile(values)
    assert profile.adjacent_descents == adjacent_descents(values)
    assert profile.inversion_count == inversion_count(values)
    assert profile.maximum_inversion_count == maximum_inversion_count(values)
    assert profile.normalized_kendall_distance == normalized_kendall_distance(values)
    assert profile.spearman_footrule == spearman_footrule(values)
    assert profile.maximum_rank_error == maximum_rank_error(values)
