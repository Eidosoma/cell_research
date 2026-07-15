from __future__ import annotations

import pytest

from src.detours.replay import (
    assert_numeric_series_equal,
    goal_directions,
    profile_rows,
    scheduler_span_total,
    stable_trace_id,
)


def test_goal_directions_preserves_native_or_both_mixed_candidates() -> None:
    assert goal_directions(["ascending"] * 3) == ("ascending",)
    assert goal_directions(["descending"] * 2) == ("descending",)
    assert goal_directions(["ascending", "descending"]) == (
        "ascending",
        "descending",
    )
    with pytest.raises(ValueError):
        goal_directions([])


def test_profile_rows_emits_complete_s01_profiles() -> None:
    rows = profile_rows([2, 1, 1], ["ascending", "descending"])
    assert [row["direction"] for row in rows] == ["ascending", "descending"]
    assert rows[0]["inversion_count"] == 2
    assert rows[1]["inversion_count"] == 0
    assert rows[0]["normalized_duplicate_aware_earth_movers_distance"] == rows[0][
        "normalized_spearman_footrule"
    ]


def test_curve_comparison_and_scheduler_accounting() -> None:
    assert_numeric_series_equal([0, 0.5, 1], [0.0, 0.5, 1.0], label="fixture")
    with pytest.raises(AssertionError, match="index 1"):
        assert_numeric_series_equal([0, 0.4], [0, 0.5], label="fixture")
    assert scheduler_span_total(
        [{"scheduler_checkpoint_count": 2}, {"scheduler_checkpoint_count": 7}]
    ) == 9


def test_stable_trace_id_is_source_specific_and_deterministic() -> None:
    first = stable_trace_id("S09/selected", "scenario-1")
    assert first == stable_trace_id("S09/selected", "scenario-1")
    assert first != stable_trace_id("S11/summaries", "scenario-1")
