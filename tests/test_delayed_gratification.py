from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from analysis.delayed_gratification import (
    PREREGISTRATION,
    PREREGISTRATION_SHA256,
    SwapErrorRecorder,
    dg_metrics,
    frozen_code_randomized_comparison,
    hand_calculated_fixtures,
    monotonicity_error,
    run_optimized_trajectory,
    run_ordinary_trajectory,
    sha256_file,
)
from reference_simulator.api import create_scenario


def test_frozen_preregistration_and_hand_calculated_metric_fixtures() -> None:
    assert sha256_file(PREREGISTRATION) == PREREGISTRATION_SHA256
    result = hand_calculated_fixtures()
    assert result["success"]
    assert result["fixtureCount"] == 9


def test_exact_public_code_terminal_quirk_is_named_and_separate() -> None:
    observed = dg_metrics([8, 6, 7, 5, 7, 3, 6])
    assert observed["terminalEarlyReturn"]
    assert observed["primary"] == pytest.approx(2.0)
    assert observed["reconciled"] == pytest.approx(1.0)
    assert observed["recoveryRatio"] == pytest.approx(2.0)


def test_clean_room_metric_matches_frozen_pure_function_randomly() -> None:
    result = frozen_code_randomized_comparison(samples=250)
    assert result["success"]
    assert result["failureCount"] == 0


def test_incremental_error_recorder_matches_full_recomputation() -> None:
    scenario = create_scenario(
        [4, 1, 3, 2, 5, 0],
        policy="Bubble",
        generation_key="S12/test/incremental-recorder",
        permute=False,
    )
    state = SimpleNamespace(occupancy=list(scenario.initial_occupancy))
    recorder = SwapErrorRecorder(scenario, state.occupancy)
    rng = random.Random(1207)
    for _ in range(100):
        left, right = rng.sample(range(len(state.occupancy)), 2)
        state.occupancy[left], state.occupancy[right] = (
            state.occupancy[right],
            state.occupancy[left],
        )
        recorder(state, left, right)
        values = [scenario.cell_map[cell_id].value for cell_id in state.occupancy]
        assert recorder.current == monotonicity_error(values)


@pytest.mark.parametrize("architecture", ["cell_view", "traditional"])
@pytest.mark.parametrize("policy", ["Bubble", "Insertion", "Selection"])
def test_optimized_trajectory_matches_ordinary_event_path(
    architecture: str, policy: str
) -> None:
    scenario = create_scenario(
        [6, 1, 5, 2, 4, 3],
        policy=policy,
        architecture=architecture,
        faults={2: "stuck"},
        generation_key=f"S12/test/path/{architecture}/{policy}",
        permute=False,
        seed=1212,
        max_activations=20_000,
    )
    optimized = run_optimized_trajectory(scenario)
    ordinary = run_ordinary_trajectory(scenario)
    assert optimized["trajectorySha256"] == ordinary["trajectorySha256"]
    assert optimized["metrics"] == ordinary["metrics"]
    assert optimized["finalStateHash"] == ordinary["finalStateHash"]
    assert optimized["ledger"] == ordinary["ledger"]
