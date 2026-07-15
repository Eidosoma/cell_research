from __future__ import annotations

import numpy as np

from src.detours.behavioral_nulls import (
    NULL_NAMES,
    behavior_classification,
    bounded,
    counter_draw,
    opportunity_integer_weights,
    opportunity_rate_bin,
    simulate_family,
    stream_root,
    systematic_request,
)


def test_streams_are_deterministic_and_null_specific() -> None:
    roots = [stream_root("1:2:0", name) for name in NULL_NAMES]
    assert len(set(map(int, roots))) == len(NULL_NAMES)
    assert roots == [stream_root("1:2:0", name) for name in NULL_NAMES]
    draws = [int(counter_draw(roots[0], event, 1)) for event in range(16)]
    assert draws == [int(counter_draw(roots[0], event, 1)) for event in range(16)]
    assert len(set(draws)) == 16
    assert all(0 <= bounded(np.uint64(value), 7) < 7 for value in draws)


def test_systematic_quota_matches_complete_block_and_prefix_bound() -> None:
    requests = [systematic_request(t, 17, 5, 3, 11, 2) for t in range(17)]
    assert requests.count(1) == 5
    assert requests.count(2) == 3
    for length in range(1, 18):
        changed = sum(value > 0 for value in requests[:length])
        assert abs(changed - length * 8 / 17) <= 1


def _tiny_graph() -> tuple[np.ndarray, ...]:
    # Two labelled opportunities per state.  State 0 must initially worsen to
    # reach state 2; a metric-greedy null instead chooses the neutral loop.
    successors = np.asarray([0, 1, 1, 2, 2, 2], dtype=np.int64)
    terminal = np.asarray([0, 0, 1], dtype=np.uint8)
    decision = np.asarray([0, 1, 0, 1, 0, 0], dtype=np.int8)
    changed = np.asarray([0, 1, 0, 1, 0, 0], dtype=np.int8)
    kind = np.asarray([0, 1, 0, 1, 0, 0], dtype=np.int8)
    costs = np.zeros((7, 6), dtype=np.int16)
    costs[2, [0, 2, 4, 5]] = 1
    costs[5, [1, 3]] = 1
    costs[6, [1, 3]] = 2
    deltas = np.zeros((4, 6), dtype=np.int16)
    deltas[:, 1] = 1
    deltas[:, 3] = -2
    return successors, terminal, decision, changed, kind, costs, deltas


def test_rate_matched_completes_tiny_required_detour_and_greedy_waits() -> None:
    successors, terminal, decision, changed, kind, costs, deltas = _tiny_graph()
    roots = np.asarray(
        [[stream_root("tiny", name) for name in NULL_NAMES]], dtype=np.uint64
    )
    result = simulate_family(
        np.asarray([0], dtype=np.int64),
        roots,
        np.asarray([2], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.ones((1, 4), dtype=np.int16),
        2,
        successors,
        terminal,
        np.asarray([1, 1], dtype=np.int16),
        decision,
        changed,
        kind,
        costs,
        deltas,
        8,
    )
    final_states, stops, events, _, peaks, finals, _, requested, deficits, selected, *_ = result
    rate = NULL_NAMES.index("rate_matched_random")
    assert final_states[0, rate] == 2
    assert stops[0, rate] == 1
    assert events[0, rate] == 2
    assert requested[0, rate].tolist() == [2, 0, 0]
    assert deficits[0, rate].tolist() == [0, 0, 0]
    assert selected[0, rate, 0] == 2
    greedy = NULL_NAMES.index("greedy_inversion_count")
    assert final_states[0, greedy] == 0
    assert stops[0, greedy] == 3
    assert peaks[0, rate, 1] == 2
    assert finals[0, rate, 1] == 0
    assert peaks[0, greedy, 1] == 1


def test_zero_rate_matched_policy_selects_unchanged_edges() -> None:
    successors, terminal, decision, changed, kind, costs, deltas = _tiny_graph()
    roots = np.asarray(
        [[stream_root("zero-rate", name) for name in NULL_NAMES]], dtype=np.uint64
    )
    result = simulate_family(
        np.asarray([0], dtype=np.int64), roots,
        np.asarray([8], dtype=np.int32), np.asarray([0], dtype=np.int32),
        np.asarray([0], dtype=np.int32), np.ones((1, 4), dtype=np.int16),
        2, successors, terminal, np.asarray([1, 1], dtype=np.int16),
        decision, changed, kind, costs, deltas, 8,
    )
    final_states, stops, events, _, _, _, _, requested, deficits, selected, *_ = result
    rate = NULL_NAMES.index("rate_matched_random")
    assert final_states[0, rate] == 0
    assert stops[0, rate] == 3
    assert events[0, rate] == 8
    assert requested[0, rate].tolist() == [0, 0, 8]
    assert deficits[0, rate].tolist() == [0, 0, 0]
    assert selected[0, rate].tolist() == [0, 0, 8]


def test_probability_weights_and_outcome_conventions() -> None:
    weights = opportunity_integer_weights(
        np.asarray([1, 1, 1], dtype=np.uint8),
        np.asarray([4, 4, 2], dtype=np.uint8),
    )
    assert weights.tolist() == [1, 1, 2]
    assert opportunity_rate_bin(0, 0) == "terminal_zero_events"
    assert opportunity_rate_bin(10, 0) == "zero_movement"
    assert opportunity_rate_bin(10, 2) == "high_le_0.30"
    assert behavior_classification(
        completed=False,
        stop_reason="event_budget",
        exact_classification="unreachable_active",
        excursion=0,
        minimum_excursion=-1,
    ) == "exact_impossibility"
    assert behavior_classification(
        completed=True,
        stop_reason="complete",
        exact_classification="necessary_detour",
        excursion=2,
        minimum_excursion=1,
    ) == "successful_necessary_detour"
