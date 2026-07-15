from __future__ import annotations

import pytest

from src.detours.atlas import (
    IMPROVEMENT,
    NEUTRAL,
    WORSENING,
    clustered_percentile_interval,
    delta_sign,
    paper_episode_boundaries,
    stable_seed,
)
from src.detours.distances import distance_profile


def test_delta_sign_and_s01_witness_fixture() -> None:
    assert delta_sign(1e-13) == NEUTRAL
    assert delta_sign(-1e-6) == IMPROVEMENT
    assert delta_sign(3) == WORSENING
    before = distance_profile([1, 3, 0, 2]).to_dict()
    after = distance_profile([1, 0, 3, 2]).to_dict()
    assert delta_sign(after["paper_sortedness_distance"] - before["paper_sortedness_distance"]) == WORSENING
    for metric in ("inversion_count", "spearman_footrule", "maximum_rank_error"):
        assert delta_sign(after[metric] - before[metric]) == IMPROVEMENT


def test_episode_boundaries_bridge_or_break_plateaus() -> None:
    signs = [IMPROVEMENT, WORSENING, NEUTRAL, WORSENING, NEUTRAL, IMPROVEMENT]
    bridged = paper_episode_boundaries(
        signs, bridge_neutral=True, include_first_transition=True
    )
    assert [(x.start, x.worsening_end, x.end, x.paired_recovery) for x in bridged] == [
        (1, 3, 5, True)
    ]
    broken = paper_episode_boundaries(
        signs, bridge_neutral=False, include_first_transition=True
    )
    assert [(x.start, x.worsening_end, x.end, x.paired_recovery) for x in broken] == [
        (1, 1, 1, False),
        (3, 3, 3, False),
    ]


def test_post_swap_only_excludes_first_transition_and_keeps_terminal_worsening() -> None:
    signs = [WORSENING, IMPROVEMENT, WORSENING]
    included = paper_episode_boundaries(
        signs, bridge_neutral=True, include_first_transition=True
    )
    excluded = paper_episode_boundaries(
        signs, bridge_neutral=True, include_first_transition=False
    )
    assert len(included) == 2
    assert included[0].paired_recovery
    assert not included[1].paired_recovery
    assert [(x.start, x.end) for x in excluded] == [(2, 2)]


def test_cluster_bootstrap_is_deterministic_bounded_and_cluster_weighted() -> None:
    contributions = [(1, 2), (8, 8), (0, 10)]
    seed = stable_seed(123, "overall")
    first = clustered_percentile_interval(contributions, draws=2000, seed=seed)
    second = clustered_percentile_interval(contributions, draws=2000, seed=seed)
    assert first == second
    assert first[0] == pytest.approx(9 / 20)
    assert 0 <= first[1] <= first[0] <= first[2] <= 1
    assert clustered_percentile_interval([], draws=10, seed=1) == (None, None, None)
