import numpy as np

from src.detours.context_sensitivity import (
    STOP_COMPLETE,
    STOP_EVENT_BUDGET,
    metric_profile,
    result_dict,
    simulate_population,
    stream_root,
    swap_metric_delta,
)


def _run(
    occupancy=(2, 0, 1, 3),
    policies=(0, 0, 0, 0),
    faults=(0, 0, 0, 0),
    direction=0,
    batch_width=1,
    budget=64,
):
    n = len(occupancy)
    result = simulate_population(
        np.asarray([n], dtype=np.int16),
        np.asarray([policies], dtype=np.int8),
        np.asarray([faults], dtype=np.int8),
        np.asarray([occupancy], dtype=np.int16),
        np.asarray([direction], dtype=np.int8),
        np.asarray([batch_width], dtype=np.int8),
        np.asarray([stream_root("fixture")], dtype=np.uint64),
        np.asarray([budget], dtype=np.int32),
        np.asarray([-1], dtype=np.int32),
        np.asarray([-1], dtype=np.int32),
        np.asarray([-1], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
    )
    return result_dict(result)


def test_metric_profile_and_swap_delta_match_recomputation():
    occupancy = np.asarray([3, 0, 4, 1, 2], dtype=np.int16)
    for direction in (0, 1):
        before = metric_profile(occupancy, 5, direction)
        for a in range(5):
            for b in range(a + 1, 5):
                delta = swap_metric_delta(occupancy, 5, direction, a, b, before)
                changed = occupancy.copy()
                changed[a], changed[b] = changed[b], changed[a]
                assert np.array_equal(before + delta, metric_profile(changed, 5, direction))


def test_sorted_state_is_complete_without_charged_opportunity():
    result = _run(occupancy=(0, 1, 2, 3))
    assert int(result["stop"][0]) == STOP_COMPLETE
    assert int(result["events"][0]) == 0
    assert int(result["ledger"][0].sum()) == 0


def test_deterministic_replay_and_ledger_partition():
    first = _run()
    second = _run()
    for key in first:
        assert np.array_equal(first[key], second[key]), key
    ledger = first["ledger"][0]
    assert ledger[0] == ledger[3] + ledger[4] + ledger[5] + ledger[6] + ledger[7]
    assert first["stop"][0] in (STOP_COMPLETE, STOP_EVENT_BUDGET)


def test_shadow_rate_audit_does_not_mutate_native_path():
    first = _run()
    n = 4
    audited = result_dict(
        simulate_population(
            np.asarray([n], dtype=np.int16),
            np.asarray([[0, 0, 0, 0]], dtype=np.int8),
            np.asarray([[0, 0, 0, 0]], dtype=np.int8),
            np.asarray([[2, 0, 1, 3]], dtype=np.int16),
            np.asarray([0], dtype=np.int8),
            np.asarray([1], dtype=np.int8),
            np.asarray([stream_root("fixture")], dtype=np.uint64),
            np.asarray([64], dtype=np.int32),
            np.asarray([first["ledger"][0, 6]], dtype=np.int32),
            np.asarray([first["ledger"][0, 5]], dtype=np.int32),
            np.asarray([first["events"][0]], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        )
    )
    for key in (
        "stop",
        "events",
        "ledger",
        "initial",
        "final",
        "peak",
        "episode_count",
        "max_depth",
        "fingerprint",
        "final_occupancy",
        "final_cursors",
    ):
        assert np.array_equal(first[key], audited[key]), key
    assert int(audited["requested"][0].sum()) == int(audited["events"][0])
