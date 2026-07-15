"""Reusable S02 helpers for loss-explicit trajectory distance alignment."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from reference_simulator.model import canonical_json_bytes

from .distances import distance_profile


def canonical_hash(value: Any) -> str:
    """Hash JSON with the reference simulator's canonical encoding."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def values_hash(values: Sequence[int | float]) -> str:
    """Hash one ordered value snapshot (without the shared-event scope tag)."""

    return canonical_hash(list(values))


def goal_directions(directions: Iterable[str]) -> tuple[str, ...]:
    """Select the S01 goal directions required for one scenario.

    Homogeneous scenarios have one native goal. Mixed-direction scenarios do
    not have a single consensus sorting goal, so both frozen S01 directions
    are retained as bounded candidate-goal projections.
    """

    unique = set(directions)
    if unique == {"ascending"}:
        return ("ascending",)
    if unique == {"descending"}:
        return ("descending",)
    if unique == {"ascending", "descending"}:
        return ("ascending", "descending")
    raise ValueError(f"unsupported direction set: {sorted(unique)!r}")


def profile_rows(
    values: Sequence[int | float], directions: Iterable[str]
) -> list[dict[str, int | float | str]]:
    """Calculate the complete frozen S01 profile for each selected goal."""

    return [
        distance_profile(values, direction=direction).to_dict()
        for direction in goal_directions(directions)
    ]


def assert_numeric_series_equal(
    observed: Sequence[int | float],
    expected: Sequence[int | float],
    *,
    label: str,
    absolute_tolerance: float = 1e-12,
) -> None:
    """Fail with an indexed diagnostic when two retained curves differ."""

    if len(observed) != len(expected):
        raise AssertionError(
            f"{label} length mismatch: observed {len(observed)}, expected {len(expected)}"
        )
    for index, (left, right) in enumerate(zip(observed, expected)):
        if not math.isclose(float(left), float(right), abs_tol=absolute_tolerance):
            raise AssertionError(
                f"{label} differs at index {index}: {left!r} != {right!r}"
            )


def stable_trace_id(source_artifact: str, scenario_id: str) -> str:
    """Return a compact source-specific logical trace identifier."""

    material = json.dumps(
        [source_artifact, scenario_id], separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "s02t:" + hashlib.sha256(material).hexdigest()


def scheduler_span_total(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count the scheduler checkpoints represented by run-length rows."""

    return sum(int(row["scheduler_checkpoint_count"]) for row in rows)
