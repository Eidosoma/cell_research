from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, FaultMode, Policy, Proposal, ProposalKind
from reference_simulator.transition_primitives import ValidationDecision
from src.regeneration.assisted_rescue import (
    ASSISTED_RUN_SCHEMA_VERSION,
    CONTROL_ASSIGNMENT_STREAM,
    AssistedRescueContract,
    AssistedRescueController,
    MatchingChoice,
    RescueArm,
    exact_replay_assisted_rescue,
    run_assisted_rescue_phase,
    validate_assisted_spec,
)
from src.regeneration.lesions import LesionState, apply_lesion
from src.regeneration.tasks import initial_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s06_assisted_rescue.json"


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _scenario(*, policy: Policy = Policy.BUBBLE, seed: int = 31):
    return create_scenario(
        [7, 0, 6, 1, 5, 2, 4, 3],
        policy=policy,
        direction=Direction.ASCENDING,
        seed=seed,
        max_activations=20_000,
        generation_key=f"test/s06/{policy.value}/{seed}",
        permute=False,
    )


def _proposal(scenario, actor_pos: int, kind: ProposalKind = ProposalKind.NO_OP):
    actor = scenario.cells[actor_pos]
    if kind == ProposalKind.SWAP:
        target_pos = actor_pos + 1
        return Proposal(
            kind,
            actor.cell_id,
            actor_pos,
            target_pos=target_pos,
            observed_target_id=scenario.cells[target_pos].cell_id,
            reason="fixture",
        )
    if kind == ProposalKind.MEMORY_UPDATE:
        return Proposal(
            kind,
            actor.cell_id,
            actor_pos,
            new_cursor=1,
            reason="fixture",
        )
    return Proposal(kind, actor.cell_id, actor_pos, reason="fixture")


def _controller(arm: RescueArm, *, assigned_duration: int | None = None):
    scenario = _scenario()
    selected = scenario.cells[3].cell_id
    controller = AssistedRescueController(
        AssistedRescueContract(arm, assigned_duration=assigned_duration),
        scenario,
        selected,
        3,
        10,
        retain_audits=True,
    )
    return scenario, selected, controller


def _offer(
    controller: AssistedRescueController,
    scenario,
    proposal: Proposal,
    event_index: int,
    *,
    eligible: bool = True,
):
    controller.prepare(proposal, event_index)
    decision = controller.outcome(
        proposal,
        ValidationDecision("fixture_valid" if eligible else "fixture_noop", eligible),
        event_index,
    )
    controller.after_batch(
        scenario,
        initial_checkpoint(scenario).to_run_state(),
        (proposal,),
        {proposal.ordinal: decision.decision},
        event_index,
    )
    return decision


def test_spec_freezes_active_intervention_controls_costs_and_matching() -> None:
    specification = _spec()
    validate_assisted_spec(specification)
    assert specification["repairProposal"]["attemptLimit"] == 1
    assert specification["costsAndBudgets"]["energyInterpretation"].startswith(
        "abstract"
    )
    assert set(specification["matching"]["choices"]) == {
        item.value for item in MatchingChoice
    }
    assert specification["validationPanel"]["plannedRunCount"] == 1920
    assert CONTROL_ASSIGNMENT_STREAM.endswith("_s06_v1")
    assert ASSISTED_RUN_SCHEMA_VERSION == "e05.s06.assisted-rescue-run.v1"


def test_contract_rejects_static_faults_and_native_boundary_changes() -> None:
    scenario = _scenario()
    selected = scenario.cells[3].cell_id
    changed = replace(
        scenario,
        cells=(replace(scenario.cells[0], fault=FaultMode.STUCK), *scenario.cells[1:]),
    )
    with pytest.raises(ValueError, match="original normal"):
        AssistedRescueContract(RescueArm.ACTIVE).validate(changed, selected)
    with pytest.raises(ValueError, match="information"):
        replace(
            AssistedRescueContract(RescueArm.ACTIVE),
            information_permission="full_global_state",
        ).validate(scenario, selected)
    with pytest.raises(ValueError, match="unit costs"):
        replace(AssistedRescueContract(RescueArm.ACTIVE), energy_cost=2).validate(
            scenario, selected
        )


def test_active_local_neighbor_spends_one_action_and_recovers_next_opportunity() -> None:
    scenario, _, controller = _controller(RescueArm.ACTIVE)
    proposal = _proposal(scenario, 2, ProposalKind.NO_OP)
    decision = _offer(controller, scenario, proposal, 10, eligible=False)
    assert decision.decision == "s06_repair_success_pending"
    assert controller.recovery_event == 11
    assert not controller.frozen
    ledger = controller.process_ledger()
    assert ledger["repairProposals"] == ledger["repairSuccesses"] == 1
    assert ledger["nativeOpportunitiesSuppressed"] == 1
    assert ledger["actionUnitsSpent"] == ledger["energyUnitsSpent"] == 1
    assert ledger["forgoneNativeNoOp"] == 1
    assert controller.state_dict()["helperEnergySpent"] == {
        proposal.actor_id: 1
    }


def test_local_channel_is_adjacency_only_and_does_not_add_a_free_proposal() -> None:
    scenario, _, controller = _controller(RescueArm.ACTIVE)
    nonneighbor = _proposal(scenario, 0, ProposalKind.SWAP)
    decision = _offer(controller, scenario, nonneighbor, 10)
    assert decision.eligible_for_commit
    assert controller.frozen
    assert controller.process_ledger()["repairProposals"] == 0
    neighbor = _proposal(scenario, 4, ProposalKind.MEMORY_UPDATE)
    decision = _offer(controller, scenario, neighbor, 11)
    assert not decision.eligible_for_commit
    ledger = controller.process_ledger()
    assert ledger["chargedOpportunities"] == 2
    assert ledger["repairProposals"] == 1
    assert ledger["nativeOpportunitiesSuppressed"] == 1


def test_sham_has_identical_one_shot_cost_but_never_recovers() -> None:
    scenario, _, controller = _controller(RescueArm.SHAM)
    first = _proposal(scenario, 2, ProposalKind.SWAP)
    _offer(controller, scenario, first, 10)
    second = _proposal(scenario, 4, ProposalKind.NO_OP)
    _offer(controller, scenario, second, 11, eligible=False)
    ledger = controller.process_ledger()
    assert controller.frozen
    assert controller.recovery_event is None
    assert ledger["repairProposals"] == ledger["shamRepairFailures"] == 1
    assert ledger["repairSuccesses"] == 0
    assert ledger["actionUnitsSpent"] == ledger["energyUnitsSpent"] == 1
    assert controller.state_dict()["actionBudgetRemaining"] == 0


def test_spontaneous_and_matched_cost_controls_share_duration_but_not_cost() -> None:
    scenario, _, spontaneous = _controller(
        RescueArm.SPONTANEOUS, assigned_duration=3
    )
    noop = _proposal(scenario, 0)
    for event_index in (10, 11, 12):
        spontaneous.prepare(noop, event_index)
        assert spontaneous.frozen
    spontaneous.prepare(noop, 13)
    assert spontaneous.recovery_event == 13
    assert spontaneous.process_ledger()["energyUnitsSpent"] == 0

    scenario, _, matched = _controller(
        RescueArm.MATCHED_COST, assigned_duration=3
    )
    for event_index in (10, 11):
        _offer(matched, scenario, noop, event_index, eligible=False)
        assert matched.frozen
    decision = _offer(matched, scenario, noop, 12, eligible=False)
    assert decision.decision == "s06_matched_cost_recovery_pending"
    assert matched.recovery_event == 13
    ledger = matched.process_ledger()
    assert ledger["matchedCostEvents"] == 1
    assert ledger["nativeOpportunitiesSuppressed"] == 1
    assert ledger["energyUnitsSpent"] == 1


def test_unrecovered_schedules_and_passive_freeze_are_retained_without_cost() -> None:
    scenario, _, spontaneous = _controller(
        RescueArm.SPONTANEOUS, assigned_duration=None
    )
    scenario2, _, matched = _controller(
        RescueArm.MATCHED_COST, assigned_duration=None
    )
    _, _, passive = _controller(RescueArm.PASSIVE)
    noop = _proposal(scenario, 0)
    noop2 = _proposal(scenario2, 0)
    for event_index in range(10, 30):
        _offer(spontaneous, scenario, noop, event_index, eligible=False)
        _offer(matched, scenario2, noop2, event_index, eligible=False)
        _offer(passive, scenario, noop, event_index, eligible=False)
    for controller in (spontaneous, matched, passive):
        assert controller.frozen
        assert controller.recovery_event is None
        assert controller.process_ledger()["energyUnitsSpent"] == 0


def test_assisted_phase_budget_ledger_and_exact_replay() -> None:
    scenario = _scenario(policy=Policy.BUBBLE)
    checkpoint = initial_checkpoint(scenario)
    anchor = apply_lesion(
        "segment_reversal_central_v1",
        LesionState.from_checkpoint(scenario, checkpoint),
        s01_pairing_block_id="e05pb1:" + "c" * 64,
        timing_condition_id="initialization",
        injury_seed=654,
    )
    selected = anchor.post_state.occupancy[3]
    run = run_assisted_rescue_phase(
        scenario,
        checkpoint,
        post_anchor_occupancy=anchor.post_state.occupancy,
        anchor_lesion_state_hash=anchor.post_state.state_hash,
        selected_identity=selected,
        contract=AssistedRescueContract(RescueArm.ACTIVE),
        recovery_budget=6400,
        trace_mode="full",
        retain_process_audits=True,
    )
    assert all(run.opportunity_validation.values()), run.opportunity_validation
    assert run.summary["phaseActivationCount"] == run.process_ledger[
        "chargedOpportunities"
    ]
    assert run.summary["distanceAuc"] >= 0
    assert CONTROL_ASSIGNMENT_STREAM not in run.summary["streamCounterDelta"]
    exact_replay_assisted_rescue(
        run,
        scenario,
        checkpoint,
        anchor.post_state.occupancy,
        recovery_budget=6400,
    )


def test_digest_summary_path_matches_full_transition_state_and_ledgers() -> None:
    scenario = _scenario(policy=Policy.BUBBLE, seed=37)
    checkpoint = initial_checkpoint(scenario)
    anchor = apply_lesion(
        "segment_reversal_central_v1",
        LesionState.from_checkpoint(scenario, checkpoint),
        s01_pairing_block_id="e05pb1:" + "d" * 64,
        timing_condition_id="initialization",
        injury_seed=987,
    )
    selected = anchor.post_state.occupancy[3]
    kwargs = {
        "post_anchor_occupancy": anchor.post_state.occupancy,
        "anchor_lesion_state_hash": anchor.post_state.state_hash,
        "selected_identity": selected,
        "contract": AssistedRescueContract(RescueArm.PASSIVE),
        "recovery_budget": 80,
    }
    full = run_assisted_rescue_phase(
        scenario,
        checkpoint,
        **kwargs,
        trace_mode="full",
        retain_process_audits=True,
    )
    digest = run_assisted_rescue_phase(
        scenario,
        checkpoint,
        **kwargs,
        trace_mode="digest",
        retain_process_audits=True,
    )
    assert digest.final_state == full.final_state
    assert digest.final_state_hash == full.final_state_hash
    assert digest.process_final_state == full.process_final_state
    assert digest.process_ledger == full.process_ledger
    assert digest.process_audit_digest == full.process_audit_digest
    assert digest.process_transitions == full.process_transitions
    assert digest.opportunity_validation == full.opportunity_validation
    for key in full.summary:
        if key not in {"traceMode", "retainedEventCount"}:
            assert digest.summary[key] == full.summary[key]
    assert full.events
    assert digest.events == ()
    exact_replay_assisted_rescue(
        digest,
        scenario,
        checkpoint,
        anchor.post_state.occupancy,
        recovery_budget=80,
    )
