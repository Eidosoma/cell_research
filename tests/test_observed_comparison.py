from __future__ import annotations

import pytest

from reference_simulator.engine import initial_state
from reference_simulator.model import Cell, Direction, FaultMode, Policy, Scenario
from src.detours.necessary_detour import (
    CLASS_COMPLETE_START,
    CLASS_NECESSARY_DETOUR,
    CLASS_QUIESCENT,
    CLASS_REACHABLE_NO_DETOUR,
    CLASS_UNREACHABLE_ACTIVE,
)
from src.detours.observed_comparison import (
    classify_observed_suffix,
    event_scheduler_label,
    lexicographic_relation,
    project_scenario,
)


def test_value_rank_projection_transports_identity_owned_semantics() -> None:
    scenario = Scenario.create(
        [
            Cell("native-a", 20, Policy.INSERTION, Direction.ASCENDING, FaultMode.STUCK),
            Cell("native-b", 10, Policy.BUBBLE, Direction.ASCENDING),
        ],
        initial_occupancy=["native-a", "native-b"],
    )
    projection = project_scenario(scenario)
    assert projection.native_to_canonical == {"native-b": 0, "native-a": 1}
    assert projection.family.policies == (Policy.BUBBLE, Policy.INSERTION)
    assert projection.family.faults == (FaultMode.NORMAL, FaultMode.STUCK)
    assert projection.state(initial_state(scenario)).occupancy == (1, 0)


def test_mixed_direction_and_ties_rejected() -> None:
    with pytest.raises(ValueError, match="mixed-direction"):
        project_scenario(
            Scenario.create(
                [
                    Cell("a", 0, Policy.BUBBLE, Direction.ASCENDING),
                    Cell("b", 1, Policy.BUBBLE, Direction.DESCENDING),
                ]
            )
        )
    with pytest.raises(ValueError, match="Tied-valued|tied-valued"):
        project_scenario(
            Scenario.create(
                [
                    Cell("a", 0, Policy.BUBBLE, Direction.ASCENDING),
                    Cell("b", 0, Policy.BUBBLE, Direction.ASCENDING),
                ]
            )
        )


def test_scheduler_label_uses_retained_side_draw() -> None:
    scenario = Scenario.create(
        [
            Cell("a", 0, Policy.BUBBLE, Direction.ASCENDING),
            Cell("b", 1, Policy.INSERTION, Direction.ASCENDING),
        ]
    )
    projection = project_scenario(scenario)
    event = {
        "actorId": "a",
        "randomAddressesAndDraws": [
            {"stream": "bubble_side", "eventIndex": 0, "drawIndex": 0, "value": 1 << 63}
        ],
    }
    opportunity, actor, side = event_scheduler_label(projection, event)
    assert (opportunity, actor, side) == (1, 0, 1)


@pytest.mark.parametrize(
    ("solver", "success", "excursion", "label"),
    [
        (CLASS_COMPLETE_START, True, 0, "complete_start"),
        (CLASS_QUIESCENT, False, 0, "quiescent_terminal"),
        (CLASS_REACHABLE_NO_DETOUR, True, 0, "successful_no_detour"),
        (CLASS_REACHABLE_NO_DETOUR, True, 1, "successful_unnecessary_detour"),
        (CLASS_NECESSARY_DETOUR, True, 1, "successful_necessary_detour"),
        (CLASS_NECESSARY_DETOUR, False, 0, "failed_where_detour_required"),
        (CLASS_REACHABLE_NO_DETOUR, False, 0, "failed_despite_no_required_detour"),
        (CLASS_UNREACHABLE_ACTIVE, False, 2, "failed_structurally_unreachable"),
    ],
)
def test_behavior_partition(solver: int, success: bool, excursion: int, label: str) -> None:
    assert classify_observed_suffix(
        solver_classification=solver,
        observed_success=success,
        observed_excursion=excursion,
    ) == label


def test_lexicographic_relation() -> None:
    assert lexicographic_relation((1, 9), (2, 0)) == "below_optimum"
    assert lexicographic_relation((2, 1), (2, 0)) == "above_optimum"
    assert lexicographic_relation((2, 0), (2, 0)) == "equal"
