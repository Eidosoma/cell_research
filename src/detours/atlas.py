"""Sign, episode, and clustered-uncertainty helpers for E03 S03."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np


IMPROVEMENT = "improvement"
NEUTRAL = "neutral"
WORSENING = "worsening"


def delta_sign(value: float | int, *, tolerance: float = 1e-12) -> str:
    """Classify a distance delta, where positive means farther from goal."""

    numeric = float(value)
    if numeric > tolerance:
        return WORSENING
    if numeric < -tolerance:
        return IMPROVEMENT
    return NEUTRAL


@dataclass(frozen=True)
class EpisodeBoundary:
    """One positive paper-distance run and its optional immediate recovery."""

    start: int
    worsening_end: int
    end: int
    paired_recovery: bool


@dataclass
class _Run:
    sign: str
    start: int
    end: int


def paper_episode_boundaries(
    signs: Sequence[str],
    *,
    bridge_neutral: bool,
    include_first_transition: bool,
) -> list[EpisodeBoundary]:
    """Segment paper-distance signs under a frozen S03 boundary rule.

    Leading improvements are naturally ignored because only worsening runs
    start episodes. When ``bridge_neutral`` is true, neutral events are
    removed before adjacent same-sign runs are merged. Otherwise a neutral
    event breaks the run and prevents pairing across the plateau.
    """

    offset = 0 if include_first_transition else 1
    runs: list[_Run] = []
    current: _Run | None = None
    for index in range(offset, len(signs)):
        sign = signs[index]
        if sign not in {IMPROVEMENT, NEUTRAL, WORSENING}:
            raise ValueError(f"unknown sign label {sign!r}")
        if sign == NEUTRAL:
            if not bridge_neutral:
                current = None
            continue
        if current is not None and current.sign == sign:
            current.end = index
            continue
        current = _Run(sign=sign, start=index, end=index)
        runs.append(current)

    boundaries: list[EpisodeBoundary] = []
    for run_index, run in enumerate(runs):
        if run.sign != WORSENING:
            continue
        recovery: _Run | None = None
        if run_index + 1 < len(runs) and runs[run_index + 1].sign == IMPROVEMENT:
            candidate = runs[run_index + 1]
            if bridge_neutral or candidate.start == run.end + 1:
                recovery = candidate
        boundaries.append(
            EpisodeBoundary(
                start=run.start,
                worsening_end=run.end,
                end=recovery.end if recovery is not None else run.end,
                paired_recovery=recovery is not None,
            )
        )
    return boundaries


def stable_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic 32-bit bootstrap seed from a group label."""

    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int(base_seed) ^ int.from_bytes(digest[:4], "big")) & 0xFFFFFFFF


def clustered_percentile_interval(
    contributions: Sequence[tuple[float, float]],
    *,
    draws: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    """Return ratio-of-sums estimate and trace-cluster percentile interval."""

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    if not contributions:
        return None, None, None
    values = np.asarray(contributions, dtype=np.float64)
    denominator = float(values[:, 1].sum())
    if denominator <= 0:
        return None, None, None
    estimate = float(values[:, 0].sum() / denominator)
    rng = np.random.default_rng(seed)
    cluster_count = len(values)
    samples = np.empty(draws, dtype=np.float64)
    batch_size = 1000
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        selected = values[indices]
        numerator = selected[:, :, 0].sum(axis=1)
        sampled_denominator = selected[:, :, 1].sum(axis=1)
        samples[start:stop] = np.divide(
            numerator,
            sampled_denominator,
            out=np.full(stop - start, np.nan),
            where=sampled_denominator > 0,
        )
    valid = samples[np.isfinite(samples)]
    if not len(valid):
        return estimate, None, None
    lower, upper = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return estimate, float(lower), float(upper)


def trace_contributions(
    rows: Iterable[Mapping[str, object]],
    *,
    trace_field: str,
    numerator_field: str,
    denominator_field: str,
) -> list[tuple[float, float]]:
    """Aggregate event or episode contributions within logical traces."""

    grouped: dict[str, list[float]] = {}
    for row in rows:
        trace = str(row[trace_field])
        counts = grouped.setdefault(trace, [0.0, 0.0])
        counts[0] += float(row[numerator_field])
        counts[1] += float(row[denominator_field])
    return [(values[0], values[1]) for _, values in sorted(grouped.items())]
