from __future__ import annotations

from dataclasses import replace
import inspect
from itertools import permutations
import json
from pathlib import Path

import pytest

from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, Policy
from src.regeneration.target_change import (
    BENCHMARK_VERSION,
    TARGET_CHANGE_RUN_SCHEMA_VERSION,
    SignalPermission,
    TargetArm,
    TargetChange,
    TargetChangeContract,
    TargetChangeController,
    _applied_swap_inversion_delta,
    build_target_definition,
    exact_replay_target_change,
    run_target_change_phase,
    semantic_proposal_equal,
    target_correspondence_rows,
    validate_target_change_spec,
)
from src.regeneration.tasks import initial_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s09_collective_target_change.json"


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _scenario(
    *,
    policy: Policy = Policy.BUBBLE,
    direction: Direction = Direction.ASCENDING,
    sorted_values: bool = True,
    seed: int = 109,
):
    values = list(range(8))
    if direction == Direction.DESCENDING:
        values.reverse()
    if not sorted_values:
        values = [values[i] for i in (7, 0, 6, 1, 5, 2, 4, 3)]
    return create_scenario(
        values,
        policy=policy,
        direction=direction,
        seed=seed,
        max_activations=50_000,
        generation_key=f"test/s09/{policy.value}/{direction.value}/{seed}/{sorted_values}",
        permute=False,
    )


def test_spec_freezes_targets_signals_controls_and_complete_panel() -> None:
    specification = _spec()
    validate_target_change_spec(specification)
    assert specification["benchmarkVersion"] == BENCHMARK_VERSION
    assert {item["targetChangeId"] for item in specification["targetChanges"]} == {
        item.value for item in TargetChange
    }
    assert {item["signalId"] for item in specification["signalPermissions"]} == {
        item.value for item in SignalPermission
    }
    assert {item["arm"] for item in specification["controllerConditions"]} == {
        item.value for item in TargetArm
    }
    panel = specification["validationPanel"]
    assert panel["plannedRunCount"] == 10_368
    assert panel["plannedExactReplayCount"] == 10_368
    assert panel["plannedPairwiseContrasts"] == 5_184
    assert specification["controllerAndCostContract"]["runtimeStreams"] == []
    assert TARGET_CHANGE_RUN_SCHEMA_VERSION == "e05.s09.target-change-run.v1"


@pytest.mark.parametrize(
    "code_values",
    [
        (0, 1, 2, 3, 4),
        (0, 1, 0, 1, 0),
        (2, 0, 0, 1, 1),
    ],
)
def test_incremental_swap_distance_is_exact_for_unique_and_tied_targets(
    code_values: tuple[int, ...],
) -> None:
    identities = tuple(f"c{index}" for index in range(len(code_values)))
    codes = dict(zip(identities, code_values, strict=True))

    def distance(occupancy: tuple[str, ...]) -> int:
        values = [codes[identity] for identity in occupancy]
        return sum(
            left > right
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        )

    for before in permutations(identities):
        before_distance = distance(before)
        for left in range(len(before)):
            for right in range(left + 1, len(before)):
                after_list = list(before)
                after_list[left], after_list[right] = (
                    after_list[right],
                    after_list[left],
                )
                after = tuple(after_list)
                assert distance(after) - before_distance == _applied_swap_inversion_delta(
                    after, codes, left, right
                )


@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("target_change", list(TargetChange))
def test_target_correspondence_is_identity_conserving_feasible_and_exact(
    direction: Direction,
    target_change: TargetChange,
) -> None:
    scenario = _scenario(direction=direction)
    definition = build_target_definition(scenario, target_change)
    ordered = tuple(
        identity
        for identity, _ in sorted(
            definition.target_codes,
            key=lambda item: (item[1], item[0]),
        )
    )
    assert definition.distance(ordered) == 0
    assert definition.maximum_distance > 0
    rows = target_correspondence_rows(scenario, definition)
    assert len(rows) == len(scenario.cells)
    assert {row["identityId"] for row in rows} == set(scenario.cell_map)
    assert all(row["targetFeasible"] and row["identityConserved"] for row in rows)
    if target_change == TargetChange.REVERSE:
        assert all(row["exactPositionRequired"] for row in rows)
    else:
        assert any(not row["exactPositionRequired"] for row in rows)


@pytest.mark.parametrize("policy", list(Policy))
@pytest.mark.parametrize("direction", list(Direction))
def test_original_target_code_candidate_preserves_native_behavior(
    policy: Policy,
    direction: Direction,
) -> None:
    scenario = _scenario(policy=policy, direction=direction, sorted_values=False)
    state = initial_checkpoint(scenario).to_run_state()
    contract = TargetChangeContract(
        TargetArm.STABILITY_AWARE,
        TargetChange.REVERSE,
        SignalPermission.GLOBAL,
    )
    definition = build_target_definition(
        scenario, TargetChange.REVERSE, no_change=True
    )
    controller = TargetChangeController(
        contract, definition, scenario, state.activation_count, retain_audits=True
    )
    for actor in scenario.cells:
        sides = ("left", "right") if actor.policy == Policy.BUBBLE else (None,)
        for side in sides:
            surface = controller.proposal_for(state, actor.cell_id, side=side)
            assert surface.candidate is not None
            assert semantic_proposal_equal(surface.native, surface.candidate)
            actor_position = state.occupancy.index(actor.cell_id)
            assert all(
                0 <= position < len(state.occupancy)
                for position in surface.authorized_positions
            )
            if policy == Policy.BUBBLE:
                assert all(abs(position - actor_position) <= 1 for position in surface.authorized_positions)


def test_local_signal_respects_boundary_light_cone_without_clock_input() -> None:
    scenario = _scenario()
    checkpoint = initial_checkpoint(scenario)
    state = checkpoint.to_run_state()
    definition = build_target_definition(scenario, TargetChange.REVERSE)
    controller = TargetChangeController(
        TargetChangeContract(
            TargetArm.CHANGED_AWARE,
            TargetChange.REVERSE,
            SignalPermission.LOCAL,
        ),
        definition,
        scenario,
        state.activation_count,
        retain_audits=False,
    )
    boundary = state.occupancy[0]
    center = state.occupancy[3]
    assert controller.signal_available(state, boundary)
    assert not controller.signal_available(state, center)
    state.activation_count += 3
    assert controller.signal_available(state, center)
    assert "progress" not in inspect.getsource(TargetChangeController.signal_available)
    assert "distance" not in inspect.getsource(TargetChangeController.signal_available)


def test_no_signal_is_exact_behavioral_negative_control() -> None:
    scenario = _scenario(sorted_values=False)
    checkpoint = initial_checkpoint(scenario)
    active = run_target_change_phase(
        scenario,
        checkpoint,
        contract=TargetChangeContract(
            TargetArm.CHANGED_AWARE,
            TargetChange.CLASSES,
            SignalPermission.NONE,
        ),
        adaptation_budget=2_000,
        probe_budget=160,
    )
    control = run_target_change_phase(
        scenario,
        checkpoint,
        contract=replace(active.contract, arm=TargetArm.CHANGED_NONADAPTIVE),
        adaptation_budget=2_000,
        probe_budget=160,
    )
    assert active.final_state == control.final_state
    assert active.summary == control.summary
    assert active.process_ledger == control.process_ledger
    assert all(active.opportunity_validation.values())
    assert all(control.opportunity_validation.values())


@pytest.mark.parametrize("policy", list(Policy))
@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("target_change", list(TargetChange))
def test_global_target_signal_can_execute_every_native_policy_without_s08_modes(
    policy: Policy,
    direction: Direction,
    target_change: TargetChange,
) -> None:
    scenario = _scenario(policy=policy, direction=direction)
    checkpoint = initial_checkpoint(scenario)
    run = run_target_change_phase(
        scenario,
        checkpoint,
        contract=TargetChangeContract(
            TargetArm.CHANGED_AWARE,
            target_change,
            SignalPermission.GLOBAL,
        ),
        adaptation_budget=6_400,
        probe_budget=160,
    )
    assert run.summary["targetCompleted"]
    assert run.summary["postHitProbeOpportunities"] == 160
    assert run.process_ledger["targetAwareCandidates"] > 0
    assert run.process_final_state["runtimeStreams"] == []
    assert all(run.opportunity_validation.values())


def test_changed_and_stability_runs_replay_with_censors_and_costs_retained() -> None:
    scenario = _scenario(policy=Policy.INSERTION, sorted_values=False, seed=127)
    checkpoint = initial_checkpoint(scenario)
    changed = run_target_change_phase(
        scenario,
        checkpoint,
        contract=TargetChangeContract(
            TargetArm.CHANGED_NONADAPTIVE,
            TargetChange.REVERSE,
            SignalPermission.GRADIENT,
        ),
        adaptation_budget=2_000,
        probe_budget=160,
        retain_trace=True,
    )
    assert changed.summary["adaptationCensored"]
    assert changed.summary["stopReason"] in {"controller_quiescent", "phase_event_budget"}
    assert changed.process_ledger["targetAwareCandidates"] > 0
    exact_replay_target_change(changed, scenario, checkpoint, 2_000, 160)

    stable_scenario = _scenario(policy=Policy.SELECTION, seed=131)
    stable_checkpoint = initial_checkpoint(stable_scenario)
    stability = run_target_change_phase(
        stable_scenario,
        stable_checkpoint,
        contract=TargetChangeContract(
            TargetArm.STABILITY_AWARE,
            TargetChange.PARTIAL,
            SignalPermission.GLOBAL,
        ),
        adaptation_budget=6_400,
        probe_budget=160,
        retain_trace=True,
    )
    assert stability.summary["phaseActivationCount"] == 160
    assert not stability.summary["noChangeAnyTargetDeparture"]
    exact_replay_target_change(stability, stable_scenario, stable_checkpoint, 6_400, 160)


def test_observability_and_prior_evidence_boundaries_are_explicit() -> None:
    specification = _spec()
    forbidden = " ".join(specification["forbiddenControllerInputs"])
    assert "global occupancy" in forbidden
    assert "progress" in forbidden
    assert "future" in forbidden
    prior = specification["priorEvidenceConstraints"]
    assert "No bounded local-memory variant improved" in prior["s07"]
    assert "four harmed recovery" in prior["s08"]
    boundary = prior["s08"] + " " + prior["controllerBoundary"]
    for prohibited in (
        "Algotype-switch",
        "radius-expansion",
        "alternate-target",
        "exploratory",
    ):
        assert prohibited in boundary
