from __future__ import annotations

from causal_simulator.action_interface import CommonActionInterface, ControlTopology
from causal_simulator.screening import build_insertion_prefix_cache, compact_policy_proposal
from scripts.finalize_adaptive_screening import direction
from reference_simulator.engine import initial_state
from reference_simulator.model import Cell, Direction, FaultMode, Policy, Scenario
from reference_simulator.scheduler import scheduled_side


def _scenario(policy: Policy, fault: FaultMode = FaultMode.NORMAL) -> Scenario:
    cells = tuple(
        Cell(
            f"cell-{index:04d}",
            value,
            policy,
            Direction.ASCENDING,
            fault if index == 2 else FaultMode.NORMAL,
        )
        for index, value in enumerate((3, 1, 4, 2))
    )
    return Scenario.create(
        cells,
        initial_occupancy=("cell-0000", "cell-0001", "cell-0002", "cell-0003"),
        seed=19,
        max_activations=40,
        generation_key=f"screening-test/{policy.value}/{fault.value}",
    )


def test_compact_policy_projection_matches_s02_interface() -> None:
    interface = CommonActionInterface()
    for policy in Policy:
        for fault in (FaultMode.NORMAL, FaultMode.PASSIVE, FaultMode.STUCK):
            scenario = _scenario(policy, fault)
            state = initial_state(scenario)
            positions = {cell_id: index for index, cell_id in enumerate(state.occupancy)}
            for actor_id in positions:
                actor = scenario.cell_map[actor_id]
                side = None
                if actor.policy == Policy.BUBBLE and actor.fault == FaultMode.NORMAL:
                    side = scheduled_side(scenario, 0)[0]
                expected = interface.proposal_for(
                    ControlTopology.DISTRIBUTED_LOCAL,
                    scenario,
                    state,
                    actor_id,
                    side=side,
                )
                observed = compact_policy_proposal(
                    scenario,
                    state,
                    positions,
                    actor_id,
                    side=side,
                )
                assert observed == expected
                cached = compact_policy_proposal(
                    scenario,
                    state,
                    positions,
                    actor_id,
                    side=side,
                    insertion_prefix_cache=(
                        build_insertion_prefix_cache(scenario, state)
                        if policy == Policy.INSERTION
                        else None
                    ),
                )
                assert cached == expected


def test_screening_effect_direction_uses_frozen_active_minus_reference_sign() -> None:
    assert direction(-0.25) == "negative_active_minus_reference"
    assert direction(0.0) == "zero"
    assert direction(0.25) == "positive_active_minus_reference"
