from __future__ import annotations

from itertools import permutations

from analysis.s11_postfreeze_audit import positional_inversion_count


def test_positional_inversion_count_matches_known_patterns() -> None:
    assert positional_inversion_count([1, 2, 3, 4]) == 0
    assert positional_inversion_count([4, 3, 2, 1]) == 6
    assert positional_inversion_count([2, 1, 3, 1]) == 3
    assert positional_inversion_count([1, 1, 1]) == 0


def test_positional_inversion_count_varies_across_permutations() -> None:
    observed = {positional_inversion_count(row) for row in permutations([1, 2, 3])}
    assert observed == {0, 1, 2, 3}
