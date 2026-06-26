"""Metric helpers shared by the E02 deterministic simulator."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def sortedness_raw(values: Sequence[int], direction: str = "increasing") -> int:
    """Return the count of adjacent pairs consistent with the target order."""
    if direction == "increasing":
        return sum(1 for left, right in zip(values, values[1:]) if left <= right)
    if direction == "decreasing":
        return sum(1 for left, right in zip(values, values[1:]) if left >= right)
    raise ValueError(f"unknown direction: {direction}")


def sortedness_percent(values: Sequence[int], direction: str = "increasing") -> float:
    """Return E01's canonical percent Sortedness metric."""
    if len(values) < 2:
        return 100.0
    return 100.0 * sortedness_raw(values, direction=direction) / (len(values) - 1)


def monotonicity_error(values: Sequence[int], direction: str = "increasing") -> int:
    """Return E01 monotonicity error: inconsistent adjacent-pair count."""
    return max(0, len(values) - 1) - sortedness_raw(values, direction=direction)


def aggregation(algotypes: Sequence[str]) -> float:
    """Return deterministic adjacent-pair same-Algotype aggregation."""
    if len(algotypes) < 2:
        return 1.0
    same = sum(1 for left, right in zip(algotypes, algotypes[1:]) if left == right)
    return same / (len(algotypes) - 1)


def state_hash(values: Sequence[int]) -> str:
    """Hash a value state using the compact E01 int16 convention."""
    arr = np.asarray(list(values), dtype=np.int16)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def initial_values_from_seed(seed: int, n: int = 100, profile: str = "unique_1_100") -> list[int]:
    """Generate E01 baseline input arrays from a frozen seed."""
    rng = np.random.default_rng(int(seed))
    if profile == "unique_1_100" or profile.startswith("unique_"):
        values = np.arange(1, n + 1, dtype=np.int16)
    elif profile == "duplicate_1_10_x10":
        values = np.repeat(np.arange(1, 11, dtype=np.int16), 10)
        if len(values) != n:
            raise ValueError(f"duplicate_1_10_x10 requires n=100, got n={n}")
    else:
        raise ValueError(f"unsupported input profile: {profile}")
    rng.shuffle(values)
    return [int(value) for value in values]
