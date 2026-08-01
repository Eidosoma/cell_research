from __future__ import annotations

from dataclasses import replace

import pytest

from src.morph2d.elapsed_clock import ElapsedClockAuthenticator
from src.morph2d.episode_origin_clock import (
    EpisodeOriginClockAuthenticator,
    EpisodeOriginClockValidationError,
    parse_and_validate_episode_origin_clock_record,
    validate_episode_origin_clock_summary,
)
from src.morph2d.movements import MovementState, movement_state_sha256


def _state(index: int) -> MovementState:
    return MovementState(
        environment_id="s12ad-test",
        environment_sha256="0" * 64,
        transition_index=index,
        occupancy=(),
    )


@pytest.mark.parametrize("origin", [0, 1, 7, 32])
def test_episode_origin_clock_retains_lifetime_index_and_emits_elapsed_0_to_32(
    origin: int,
) -> None:
    initial = _state(origin)
    issuer = EpisodeOriginClockAuthenticator(
        scenario_id=f"origin-{origin}", horizon_transitions=32, initial_state=initial
    )
    for elapsed in range(33):
        record = issuer.issue(
            elapsed_transition=elapsed,
            raw_observation_label=elapsed - 1,
            state=replace(initial, transition_index=origin + elapsed),
        )
        assert record.elapsed_transition == elapsed
        assert record.movement_state_transition_index == origin + elapsed
        assert record.episode_origin_transition_index == origin
    summary = validate_episode_origin_clock_summary(
        issuer.finalize(),
        expected_scenario_id=f"origin-{origin}",
        expected_horizon=32,
        expected_origin_transition_index=origin,
    )
    assert summary["initialElapsedTransition"] == 0
    assert summary["finalElapsedTransition"] == 32
    assert summary["initialLifetimeTransitionIndex"] == origin
    assert summary["finalLifetimeTransitionIndex"] == origin + 32
    assert summary["rawLifetimeProvenanceRetained"] is True


def test_forged_origin_and_state_hash_fail_closed() -> None:
    initial = _state(1)
    record = EpisodeOriginClockAuthenticator(
        scenario_id="forgery", horizon_transitions=32, initial_state=initial
    ).issue(
        elapsed_transition=0, raw_observation_label=-1, state=initial
    ).to_mapping()
    for field, value in (
        ("episodeOriginTransitionIndex", 0),
        ("episodeOriginMovementStateSha256", "f" * 64),
        ("movementStateSha256", "f" * 64),
        ("movementStateTransitionIndex", 2),
    ):
        forged = dict(record)
        forged[field] = value
        with pytest.raises(EpisodeOriginClockValidationError):
            parse_and_validate_episode_origin_clock_record(forged, state=initial)


def test_episode_origin_clock_does_not_mutate_initial_state() -> None:
    initial = _state(5)
    before = movement_state_sha256(initial)
    issuer = EpisodeOriginClockAuthenticator(
        scenario_id="immutability", horizon_transitions=32, initial_state=initial
    )
    issuer.issue(elapsed_transition=0, raw_observation_label=-1, state=initial)
    assert initial.transition_index == 5
    assert movement_state_sha256(initial) == before


def test_legacy_zero_origin_clock_remains_available() -> None:
    initial = _state(0)
    record = ElapsedClockAuthenticator(
        scenario_id="legacy", horizon_transitions=32
    ).issue(elapsed_transition=0, raw_observation_label=-1, state=initial)
    assert record.elapsed_transition == 0
    assert record.movement_state_transition_index == 0
