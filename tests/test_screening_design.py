from __future__ import annotations

from collections import Counter

from scripts.freeze_adaptive_screening_design import (
    FAULT_PROFILES,
    POLICY_DIRECTIONS,
    _balanced_permutation,
)
from scripts.validate_screening_replay import sample_key, scalar_equal


def test_balanced_permutation_is_deterministic_and_bounded() -> None:
    first = _balanced_permutation(FAULT_PROFILES, 50, address="test/fault")
    second = _balanced_permutation(FAULT_PROFILES, 50, address="test/fault")
    assert first == second
    assert len(first) == 50
    counts = Counter(first)
    assert set(counts) == set(FAULT_PROFILES)
    assert max(counts.values()) - min(counts.values()) == 1


def test_policy_direction_allocation_has_all_levels() -> None:
    values = _balanced_permutation(POLICY_DIRECTIONS, 50, address="test/pd")
    counts = Counter(values)
    assert set(counts) == set(POLICY_DIRECTIONS)
    assert sorted(counts.values()) == [8, 8, 8, 8, 9, 9]


def test_replay_sample_key_is_outcome_blind_and_deterministic() -> None:
    first = sample_key("S10-RUN-000001")
    assert first == sample_key("S10-RUN-000001")
    assert first != sample_key("S10-RUN-000002")
    assert len(first) == 64


def test_replay_scalar_equality_normalizes_parquet_nulls_only() -> None:
    assert scalar_equal(float("nan"), None)
    assert scalar_equal(None, float("nan"))
    assert scalar_equal(3.0, 3.0)
    assert not scalar_equal(3.0, 4.0)
