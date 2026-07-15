from __future__ import annotations

from reference_simulator.engine import execute_batch, evaluate_terminal
from reference_simulator.model import Architecture, Direction, FaultMode, Policy
from src.detours.barrier_interventions import (
    _CoupledSchedule,
    compare_to_baseline,
    conditioned_seed,
    execute_intervention_arm,
    intervention_families,
    make_scenario,
)
from src.detours.state_space import FamilySpec, StructuralState, rank_permutation


def _family(*, stuck: int = 3) -> FamilySpec:
    faults = [FaultMode.NORMAL] * 4
    faults[stuck] = FaultMode.STUCK
    return FamilySpec(
        n=4,
        architecture=Architecture.CELL_VIEW,
        direction=Direction.ASCENDING,
        policies=(Policy.BUBBLE,) * 4,
        faults=tuple(faults),
    )


def test_intervention_family_changes_are_isolated_and_exhaustive() -> None:
    family = _family()
    arms = intervention_families(family, 3)
    assert [item[0] for item in arms].count("baseline") == 1
    assert [item[0] for item in arms].count("remove") == 1
    assert [item[0] for item in arms].count("activate") == 1
    assert [item[0] for item in arms].count("move") == 3
    assert [item[0] for item in arms].count("add") == 3
    for intervention, _, target, arm_family in arms:
        assert arm_family.policies == family.policies
        assert arm_family.direction == family.direction
        if intervention == "remove":
            assert arm_family.fault_count == 0
        elif intervention == "move":
            assert arm_family.fault_count == 1
            assert arm_family.faults[3] == FaultMode.NORMAL
            assert arm_family.faults[target] == FaultMode.STUCK
        elif intervention == "add":
            assert arm_family.fault_count == 2


def test_common_schedule_is_byte_exact_when_pair_key_is_scenario_id() -> None:
    family = FamilySpec(
        n=4,
        architecture=Architecture.CELL_VIEW,
        direction=Direction.ASCENDING,
        policies=(Policy.BUBBLE,) * 4,
        faults=(FaultMode.NORMAL,) * 4,
    )
    scenario = make_scenario(family, seed=17, max_activations=32, generation_key="test")
    state = StructuralState(family, rank_permutation((3, 2, 1, 0))).to_run_state()
    state.terminal = evaluate_terminal(scenario, state)
    native = state.clone()
    external = state.clone()
    native_events, native_bytes = execute_batch(scenario, native, retain_events=True)
    schedule = _CoupledSchedule(
        scenario, coupling_key=scenario.scenario_id, seed=scenario.seed, pulse_focal_id=None
    )
    external_events, external_bytes = execute_batch(
        scenario, external, retain_events=True, schedule_factory=schedule
    )
    assert native_events == external_events
    assert native_bytes == external_bytes
    assert native.to_dict() == external.to_dict()


def test_activation_pulse_and_baseline_share_prestate_and_tokens() -> None:
    family = _family(stuck=3)
    state_ordinal = rank_permutation((3, 2, 1, 0))
    coupling_key = "test-s09-pair"
    seed, attempts = conditioned_seed(coupling_key, 3, 4, 0)
    common = dict(
        source_family_ordinal=1,
        arm_family_ordinal=1,
        focal_index=3,
        target_index=None,
        replicate_index=0,
        coupling_key=coupling_key,
        coupling_seed=seed,
        seed_search_attempts=attempts,
        max_activations=128,
    )
    baseline_row, baseline_trace = execute_intervention_arm(
        family,
        state_ordinal,
        intervention_type="baseline",
        arm_variant="baseline",
        **common,
    )
    pulse_row, pulse_trace = execute_intervention_arm(
        family,
        state_ordinal,
        intervention_type="activate",
        arm_variant="activate:c3:event0",
        **common,
    )
    assert baseline_trace[0]["actor_id"] == "c3"
    assert pulse_trace[0]["actor_id"] == "c3"
    assert pulse_row["pulse_delivered"] is True
    comparison = compare_to_baseline(
        baseline_row, baseline_trace, pulse_row, pulse_trace
    )
    assert comparison["pre_dynamic_state_equal"] is True
    assert comparison["actor_token_common_prefix_equal"] is True
    assert comparison["side_token_common_prefix_equal"] is True
