from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from causal_simulator.action_interface import ActorView, VisibleCell
from reference_simulator.api import create_scenario
from reference_simulator.model import (
    Direction,
    FaultMode,
    Policy,
    Proposal,
    ProposalKind,
)
from reference_simulator.rng import u64
from reference_simulator.transition_primitives import ValidationDecision
from src.regeneration.dynamic_faults import (
    ACTIVE_PROFILES,
    DYNAMIC_RUN_SCHEMA_VERSION,
    INTERMITTENT_TRANSITION_STREAM,
    PROCESS_STREAMS,
    SENSING_STATUS_STREAM,
    SENSING_VALUE_STREAM,
    TEMPORARY_RECOVERY_STREAM,
    DynamicFaultContract,
    DynamicProcessController,
    DynamicProfile,
    DynamicSensingTransformer,
    dyadic_hit,
    exact_replay_dynamic,
    rng_coupling_audit,
    run_dynamic_phase,
    validate_dynamic_spec,
)
from src.regeneration.lesions import LesionState, apply_lesion
from src.regeneration.tasks import initial_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s04_dynamic_faults.json"


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _scenario(*, policy: Policy = Policy.BUBBLE, seed: int = 17):
    return create_scenario(
        [7, 0, 6, 1, 5, 2, 4, 3],
        policy=policy,
        direction=Direction.ASCENDING,
        seed=seed,
        max_activations=20_000,
        generation_key=f"test/s04/{policy.value}/{seed}",
        permute=False,
    )


def _proposal(scenario, *, kind: ProposalKind = ProposalKind.NO_OP) -> Proposal:
    actor = scenario.cells[0]
    if kind == ProposalKind.SWAP:
        target = scenario.cells[1]
        return Proposal(
            kind,
            actor.cell_id,
            0,
            target_pos=1,
            observed_target_id=target.cell_id,
            reason="fixture",
        )
    if kind == ProposalKind.MEMORY_UPDATE:
        return Proposal(
            kind,
            actor.cell_id,
            0,
            new_cursor=1,
            reason="fixture",
        )
    return Proposal(kind, actor.cell_id, 0, reason="fixture")


def _controller(profile: DynamicProfile, *, seed: int = 17):
    scenario = _scenario(seed=seed)
    selected = scenario.cells[3].cell_id
    controller = DynamicProcessController(
        DynamicFaultContract(profile), scenario, selected, 0, retain_audits=True
    )
    return scenario, selected, controller


def test_spec_freezes_five_profiles_four_isolated_streams_and_fatigue_mediator() -> None:
    specification = _spec()
    validate_dynamic_spec(specification)
    assert {item["profileId"] for item in specification["profiles"]} == {
        profile.value for profile in ACTIVE_PROFILES
    }
    assert {item["name"] for item in specification["streamRegistry"]} == set(
        PROCESS_STREAMS
    )
    fatigue = DynamicFaultContract(DynamicProfile.FATIGUE)
    assert fatigue.classification == "endogenous_mediator"
    assert fatigue.streams == ()
    assert DYNAMIC_RUN_SCHEMA_VERSION == "e05.s04.dynamic-fault-run.v1"


def test_contract_rejects_scenario_faults_and_boundary_changes() -> None:
    scenario = _scenario()
    selected = scenario.cells[0].cell_id
    changed = replace(
        scenario,
        cells=(replace(scenario.cells[0], fault=FaultMode.STUCK), *scenario.cells[1:]),
    )
    with pytest.raises(ValueError, match="original normal"):
        DynamicFaultContract(DynamicProfile.SHAM).validate(changed, selected)
    with pytest.raises(ValueError, match="information"):
        replace(
            DynamicFaultContract(DynamicProfile.SHAM),
            information_permission="full_global_state",
        ).validate(scenario, selected)


def test_fixed_freeze_has_exact_16_opportunity_duration_without_rng() -> None:
    scenario, _, controller = _controller(DynamicProfile.TEMPORARY_FIXED)
    proposal = _proposal(scenario)
    for event_index in range(16):
        _, consumption = controller.prepare(proposal, event_index)
        assert controller.temporary_frozen
        assert consumption == ()
        controller.after_batch(scenario, initial_checkpoint(scenario).to_run_state(), (proposal,), {0: "no_op"}, event_index)
    controller.prepare(proposal, 16)
    assert not controller.temporary_frozen
    assert controller.temporary_recovery_event == 16
    assert controller.process_ledger()["temporaryFreezeExposures"] == 16


def test_geometric_recovery_is_first_hit_support_one_and_censorable() -> None:
    scenario, _, controller = _controller(DynamicProfile.TEMPORARY_GEOMETRIC)
    proposal = _proposal(scenario)
    expected = next(
        index + 1
        for index in range(1024)
        if dyadic_hit(
            u64(scenario.seed, scenario.scenario_id, TEMPORARY_RECOVERY_STREAM, index, 0),
            1,
            16,
        )
    )
    for event_index in range(expected):
        _, consumption = controller.prepare(proposal, event_index)
        assert consumption == ((TEMPORARY_RECOVERY_STREAM, 1),)
        controller.after_batch(scenario, initial_checkpoint(scenario).to_run_state(), (proposal,), {0: "no_op"}, event_index)
    assert controller.temporary_recovery_event == expected
    assert controller.process_ledger()["temporaryRecoveryHazardDraws"] == expected


def test_intermittent_transition_stream_and_gate_are_movement_specific() -> None:
    scenario, _, controller = _controller(DynamicProfile.INTERMITTENT)
    proposal = _proposal(scenario)
    _, consumption = controller.prepare(proposal, 0)
    assert consumption == ((INTERMITTENT_TRANSITION_STREAM, 1),)
    controller.intermittent_failed = True
    rejected = controller.outcome(
        _proposal(scenario, kind=ProposalKind.SWAP), ValidationDecision("valid", True), 1
    )
    permitted = controller.outcome(
        _proposal(scenario, kind=ProposalKind.MEMORY_UPDATE),
        ValidationDecision("valid", True),
        1,
    )
    assert rejected.decision == "rejected_intermittent_movement_failure"
    assert permitted.eligible_for_commit


def test_sensing_uses_field_specific_streams_and_protects_actor_status() -> None:
    scenario, _, controller = _controller(DynamicProfile.SENSING)
    transformer = DynamicSensingTransformer(controller.contract, scenario, controller)
    actor = scenario.cells[0]
    actor_view = ActorView(
        actor.cell_id, 0, actor.value, actor.policy, actor.direction, actor.fault, 8, None
    )
    visible_actor = transformer(actor_view, event_index=9, read_ordinal=0)
    assert visible_actor.cell_id == actor_view.cell_id
    assert visible_actor.position == actor_view.position
    assert visible_actor.policy == actor_view.policy
    assert visible_actor.direction == actor_view.direction
    assert visible_actor.fault == actor_view.fault
    target = VisibleCell(1, scenario.cells[1].value, scenario.cells[1].fault)
    transformer(target, event_index=9, read_ordinal=1)
    streams = {item["stream"] for item in controller.retained_audits if item["kind"] == "sensing"}
    assert streams == {SENSING_VALUE_STREAM, SENSING_STATUS_STREAM}
    ledger = controller.process_ledger()
    assert ledger["sensingValueDraws"] == 2
    assert ledger["sensingStatusDraws"] == 1


def test_fatigue_is_accepted_movement_mediated_and_recovers_on_global_clock() -> None:
    scenario, _, controller = _controller(DynamicProfile.FATIGUE)
    proposal = _proposal(scenario, kind=ProposalKind.SWAP)
    state = initial_checkpoint(scenario).to_run_state()
    for event_index in range(3):
        controller.prepare(proposal, event_index)
        controller.after_batch(scenario, state, (proposal,), {0: "accepted"}, event_index)
    assert controller.process_ledger()["fatigueMovementParticipations"] == 6
    assert controller.process_ledger()["fatigueThresholdTriggers"] == 2
    assert controller.fatigue_until[proposal.actor_id] == 11
    blocked = controller.outcome(proposal, ValidationDecision("valid", True), 3)
    assert blocked.decision == "rejected_fatigued_actor"
    for event_index in range(3, 11):
        controller.prepare(_proposal(scenario), event_index)
        assert proposal.actor_id in controller.fatigue_until
    controller.prepare(_proposal(scenario), 11)
    assert proposal.actor_id not in controller.fatigue_until
    assert DynamicFaultContract(DynamicProfile.FATIGUE).streams == ()


def test_dynamic_run_replays_exactly_and_preserves_checkpoint_clock_and_ledger() -> None:
    scenario = _scenario(policy=Policy.BUBBLE)
    checkpoint = initial_checkpoint(scenario)
    state = LesionState.from_checkpoint(scenario, checkpoint)
    anchor = apply_lesion(
        "segment_reversal_central_v1",
        state,
        s01_pairing_block_id="e05pb1:" + "a" * 64,
        timing_condition_id="initialization",
        injury_seed=123,
    )
    run = run_dynamic_phase(
        scenario,
        checkpoint,
        post_anchor_occupancy=anchor.post_state.occupancy,
        anchor_lesion_state_hash=anchor.post_state.state_hash,
        selected_identity=anchor.post_state.occupancy[3],
        contract=DynamicFaultContract(DynamicProfile.TEMPORARY_FIXED),
        recovery_budget=6400,
        trace_mode="full",
        retain_process_audits=True,
    )
    assert all(run.opportunity_validation.values()), run.opportunity_validation
    assert run.start_event_index == checkpoint.activation_count
    assert run.summary["phaseActivationCount"] == run.process_ledger["chargedOpportunities"]
    exact_replay_dynamic(
        run, scenario, checkpoint, anchor.post_state.occupancy, recovery_budget=6400
    )


def test_static_scenario_reconstruction_is_explicitly_rng_unpaired() -> None:
    scenario = _scenario()
    audit = rng_coupling_audit(scenario, scenario.cells[3].cell_id, 11)
    assert audit["scenarioIdChanged"]
    assert audit["classification"] == "scenario_paired_rng_unpaired"
    assert not any(item["valuesEqual"] for item in audit["streams"])
