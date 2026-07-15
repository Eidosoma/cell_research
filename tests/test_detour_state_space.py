from __future__ import annotations

import itertools
import math

import pytest

from reference_simulator.engine import initial_state
from reference_simulator.model import (
    Architecture,
    Direction,
    FaultMode,
    Policy,
    ProposalKind,
    state_hash,
)
from reference_simulator.policies import cell_view_proposal, traditional_proposal
from reference_simulator.transition_primitives import commit_proposal, validate_proposal
from src.detours.state_space import (
    FamilySpec,
    StructuralState,
    all_policy_assignments,
    all_policy_state_count,
    fault_maps,
    pure_policy,
    rank_permutation,
    structural_state_count,
    unrank_permutation,
)
from scripts.enumerate_s04_states import build_family_plan


def family(
    *,
    n: int = 4,
    architecture: Architecture = Architecture.CELL_VIEW,
    direction: Direction = Direction.ASCENDING,
    policy: Policy = Policy.BUBBLE,
    faults: tuple[FaultMode, ...] | None = None,
) -> FamilySpec:
    return FamilySpec(
        n=n,
        architecture=architecture,
        direction=direction,
        policies=pure_policy(policy, n),
        faults=faults or (FaultMode.NORMAL,) * n,
    )


def test_permutation_rank_round_trip_is_exhaustive_through_n9() -> None:
    checked = 0
    for n in range(1, 10):
        for expected_rank, permutation in enumerate(itertools.permutations(range(n))):
            assert rank_permutation(permutation) == expected_rank
            assert unrank_permutation(n, expected_rank) == permutation
            checked += 1
    assert checked == 409_113
    with pytest.raises(ValueError):
        rank_permutation((0, 0))
    with pytest.raises(ValueError):
        unrank_permutation(4, math.factorial(4))


def test_selection_cursor_code_round_trip_and_sentinels() -> None:
    policies = (Policy.SELECTION, Policy.BUBBLE, Policy.SELECTION, Policy.INSERTION)
    for direction in (Direction.ASCENDING, Direction.DESCENDING):
        spec = FamilySpec(
            n=4,
            architecture=Architecture.CELL_VIEW,
            direction=direction,
            policies=policies,
            faults=(FaultMode.NORMAL,) * 4,
        )
        domains = range(0, 5) if direction == Direction.ASCENDING else range(-1, 4)
        for left, right in itertools.product(domains, repeat=2):
            state = StructuralState.from_components(
                spec,
                (3, 1, 0, 2),
                {0: left, 2: right},
            )
            assert state.occupancy == (3, 1, 0, 2)
            assert state.selection_cursors == {0: left, 2: right}
            assert StructuralState.from_run_state(spec, state.to_run_state()) == state
        assert spec.cursor_state_count == 25


def test_count_formulas_cover_policy_memory_and_fault_maps() -> None:
    assert structural_state_count(4, 0) == 24
    assert structural_state_count(4, 4) == 15_000
    assert all_policy_state_count(4) == 57_624
    direct = sum(
        FamilySpec(
            4,
            Architecture.CELL_VIEW,
            Direction.ASCENDING,
            policies,
            (FaultMode.NORMAL,) * 4,
        ).state_count
        for policies in all_policy_assignments(4)
    )
    assert direct == all_policy_state_count(4)
    assert len(list(fault_maps(4))) == 1 + 2 * (math.comb(4, 1) + math.comb(4, 2) + math.comb(4, 3)) == 29


def test_encoder_fixture_and_family_semantic_sensitivity() -> None:
    spec = family()
    state = StructuralState(spec, 0, 0)
    assert spec.family_id == "s04f:af4f2c1f6dd2c1d8a203152436a379abb598243c691a5a6f9af88e2648578b8d"
    assert state.state_id == "s04s:3eb4215574b328a6c7e03590a00e8b5fa1ea896636566e130d62df694883fcdb"
    variants = {
        spec.family_digest,
        family(direction=Direction.DESCENDING).family_digest,
        family(policy=Policy.INSERTION).family_digest,
        family(faults=(FaultMode.STUCK,) + (FaultMode.NORMAL,) * 3).family_digest,
        family(architecture=Architecture.TRADITIONAL).family_digest,
    }
    assert len(variants) == 5
    assert FamilySpec.from_canonical_dict(spec.canonical_dict()) == spec


def test_direction_dual_is_an_involution() -> None:
    spec = FamilySpec(
        4,
        Architecture.CELL_VIEW,
        Direction.ASCENDING,
        (Policy.SELECTION, Policy.BUBBLE, Policy.INSERTION, Policy.SELECTION),
        (FaultMode.NORMAL, FaultMode.STUCK, FaultMode.PASSIVE, FaultMode.NORMAL),
    )
    state = StructuralState.from_components(spec, (2, 0, 3, 1), {0: 4, 3: 1})
    dual = state.dual()
    assert dual.family.direction == Direction.DESCENDING
    assert dual.occupancy == (1, 3, 0, 2)
    assert dual.selection_cursors == {0: 2, 3: -1}
    assert dual.dual() == state
    assert spec.dual().dual() == spec


def test_structural_projection_ignores_execution_history_but_native_hash_does_not() -> None:
    spec = family(policy=Policy.SELECTION)
    scenario = spec.scenario()
    state = initial_state(scenario)
    projected = StructuralState.from_run_state(spec, state)
    changed = state.clone()
    changed.activation_count = 17
    changed.stream_counters = {"actor_activation": 23}
    changed.ledger["activations"] = 17
    changed.ledger["proposals"] = 17
    assert StructuralState.from_run_state(spec, changed) == projected
    assert state_hash(scenario.scenario_id, state) != state_hash(scenario.scenario_id, changed)


def test_cell_view_successors_close_in_the_structural_domain() -> None:
    spec = FamilySpec(
        4,
        Architecture.CELL_VIEW,
        Direction.ASCENDING,
        (Policy.SELECTION, Policy.BUBBLE, Policy.INSERTION, Policy.SELECTION),
        (FaultMode.NORMAL,) * 4,
    )
    structural = StructuralState.from_components(spec, (3, 1, 2, 0), {0: 0, 3: 0})
    scenario = spec.scenario()
    for actor_id in structural.to_run_state().occupancy:
        actor = scenario.cell_map[actor_id]
        sides = ("left", "right") if actor.policy == Policy.BUBBLE else (None,)
        for side in sides:
            state = structural.to_run_state()
            proposal = cell_view_proposal(scenario, state, actor_id, side=side)
            decision = validate_proposal(scenario, state, proposal)
            if decision.eligible_for_commit:
                snapshot = state.clone()
                assert commit_proposal(state, snapshot, proposal, "accepted")
                successor = StructuralState.from_run_state(spec, state)
                assert 0 <= successor.occupancy_rank < spec.occupancy_state_count
                assert 0 <= successor.selection_cursor_code < spec.cursor_state_count


def test_stuck_target_rejects_and_passive_target_can_move() -> None:
    occupancy = (1, 0, 2, 3)
    for target_fault, expected in (
        (FaultMode.STUCK, "rejected_target_stuck"),
        (FaultMode.PASSIVE, "valid"),
    ):
        spec = family(
            faults=(target_fault, FaultMode.NORMAL, FaultMode.NORMAL, FaultMode.NORMAL)
        )
        structural = StructuralState.from_components(spec, occupancy, {})
        state = structural.to_run_state()
        proposal = cell_view_proposal(spec.scenario(), state, "c1", side="right")
        assert proposal.kind == ProposalKind.SWAP
        assert validate_proposal(spec.scenario(), state, proposal).decision == expected


def test_traditional_controller_successor_uses_same_occupancy_encoder() -> None:
    spec = family(architecture=Architecture.TRADITIONAL, policy=Policy.SELECTION)
    structural = StructuralState.from_components(spec, (3, 1, 2, 0), {})
    state = structural.to_run_state()
    scenario = spec.scenario()
    proposal = traditional_proposal(scenario, state)
    decision = validate_proposal(scenario, state, proposal)
    assert decision.eligible_for_commit
    snapshot = state.clone()
    assert commit_proposal(state, snapshot, proposal, "accepted")
    successor = StructuralState.from_run_state(spec, state)
    assert successor.family == spec
    assert successor.selection_cursor_code == 0


def test_frozen_materialization_plan_counts_and_limits() -> None:
    plan = build_family_plan()
    assert len(plan) == 7_984
    assert sum(item.family.state_count for item in plan) == 22_301_808
    assert max(item.family.state_count for item in plan) == 933_120
    assert {item.family.n for item in plan} == {4, 5, 6, 7, 8, 9}
    assert all(item.ordinal == index for index, item in enumerate(plan))
    assert len({item.family.family_id for item in plan}) == len(plan)
