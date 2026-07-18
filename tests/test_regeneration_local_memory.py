from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest

from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, FaultMode, Policy, Proposal, ProposalKind
from reference_simulator.transition_primitives import ValidationDecision
from src.regeneration.lesions import LesionState, apply_lesion
from src.regeneration.local_memory import (
    BLOCKED,
    FAILED,
    RECENT,
    TIMER,
    CONTROL_ASSIGNMENT_STREAM,
    LOCAL_MEMORY_RUN_SCHEMA_VERSION,
    LocalMemoryBank,
    LocalMemoryContract,
    LocalMemoryController,
    LocalObservation,
    LocalOutcome,
    MatchingChoice,
    MemoryArm,
    MemoryVariant,
    exact_replay_local_memory,
    run_local_memory_phase,
    validate_local_memory_spec,
)
from src.regeneration.tasks import initial_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s07_local_memory.json"


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _scenario(*, policy: Policy = Policy.BUBBLE, seed: int = 41):
    return create_scenario(
        [7, 0, 6, 1, 5, 2, 4, 3],
        policy=policy,
        direction=Direction.ASCENDING,
        seed=seed,
        max_activations=20_000,
        generation_key=f"test/s07/{policy.value}/{seed}",
        permute=False,
    )


def _proposal(scenario, actor_pos: int, *, target_pos: int | None = None):
    actor = scenario.cells[actor_pos]
    if target_pos is None:
        return Proposal(ProposalKind.NO_OP, actor.cell_id, actor_pos, reason="fixture")
    return Proposal(
        ProposalKind.SWAP,
        actor.cell_id,
        actor_pos,
        target_pos=target_pos,
        observed_target_id=scenario.cells[target_pos].cell_id,
        reason="fixture",
    )


def _offer(
    controller: LocalMemoryController,
    scenario,
    proposal: Proposal,
    event_index: int,
    *,
    eligible: bool,
):
    controller.prepare(proposal, event_index)
    decision = controller.outcome(
        proposal,
        ValidationDecision("fixture_valid" if eligible else "fixture_noop", eligible),
        event_index,
    )
    final_decision = "accepted" if decision.eligible_for_commit else decision.decision
    controller.after_batch(
        scenario,
        initial_checkpoint(scenario).to_run_state(),
        (proposal,),
        {proposal.ordinal: final_decision},
        event_index,
    )
    return decision


def test_spec_freezes_components_costs_controls_matching_and_panel() -> None:
    specification = _spec()
    validate_local_memory_spec(specification)
    assert {item["componentId"] for item in specification["memoryComponents"]} == {
        FAILED,
        BLOCKED,
        RECENT,
        TIMER,
    }
    assert {item["variantId"] for item in specification["variants"]} == {
        item.value for item in MemoryVariant
    }
    assert specification["validationPanel"]["plannedRunCount"] == 5184
    assert specification["validationPanel"]["plannedPairwiseContrasts"] == 4800
    assert set(specification["matching"]["choices"]) == {
        item.value for item in MatchingChoice
    }
    assert CONTROL_ASSIGNMENT_STREAM.endswith("_s07_v1")
    assert LOCAL_MEMORY_RUN_SCHEMA_VERSION == "e05.s07.local-memory-run.v1"


def test_contract_rejects_faulted_scenario_global_information_and_bad_roles() -> None:
    scenario = _scenario()
    selected = scenario.cells[3].cell_id
    changed = replace(
        scenario,
        cells=(replace(scenario.cells[0], fault=FaultMode.STUCK), *scenario.cells[1:]),
    )
    with pytest.raises(ValueError, match="original normal"):
        LocalMemoryContract(MemoryArm.ACTIVE, MemoryVariant.ALL).validate(
            changed, selected
        )
    with pytest.raises(ValueError, match="information"):
        replace(
            LocalMemoryContract(MemoryArm.ACTIVE, MemoryVariant.ALL),
            information_permission="full_global_state",
        ).validate(scenario, selected)
    with pytest.raises(ValueError, match="requires a memory variant"):
        LocalMemoryContract(MemoryArm.ACTIVE).validate(scenario, selected)
    with pytest.raises(ValueError, match="cannot carry a variant"):
        LocalMemoryContract(
            MemoryArm.PERMANENT, MemoryVariant.FAILED_ONLY
        ).validate(scenario, selected)


def test_private_bank_interface_has_no_global_clock_progress_or_scenario() -> None:
    bank_slots = set(LocalMemoryBank.__slots__)
    assert bank_slots == {"variant", "enabled", "states", "ledger"}
    forbidden = {"scenario", "occupancy", "event_index", "progress", "target", "ledger_total"}
    for cls in (LocalObservation, LocalOutcome):
        fields = set(cls.__dataclass_fields__)
        assert not fields.intersection(forbidden)
    assert set(inspect.signature(LocalMemoryBank.ready).parameters) == {
        "self",
        "observation",
    }
    assert set(inspect.signature(LocalMemoryBank.update).parameters) == {
        "self",
        "outcome",
    }


def test_failed_counter_is_saturating_prior_state_and_resets_on_local_progress() -> None:
    bank = LocalMemoryBank(MemoryVariant.FAILED_ONLY, ["a"])
    observation = LocalObservation("a", 1)
    assert bank.ready(observation) == (False, None)
    for _ in range(3):
        bank.update(LocalOutcome("a", 1, True, 1, False))
    assert bank.states["a"].failed_count == 3
    assert bank.ready(observation) == (True, FAILED)
    bank.update(LocalOutcome("a", 1, True, 1, False))
    assert bank.states["a"].failed_count == 3
    assert bank.ledger["saturatingIncrements"] == 1
    bank.update(LocalOutcome("a", None, False, None, True))
    assert bank.states["a"].failed_count == 0
    assert bank.validate_bounds()


def test_blocked_direction_is_side_specific_and_resettable() -> None:
    bank = LocalMemoryBank(MemoryVariant.BLOCKED_ONLY, ["a"])
    bank.update(LocalOutcome("a", 0, True, 0, False))
    assert bank.ready(LocalObservation("a", 0)) == (True, BLOCKED)
    assert bank.ready(LocalObservation("a", 1)) == (False, None)
    bank.update(LocalOutcome("a", 1, True, 1, False))
    assert bank.states["a"].blocked_mask == 3
    bank.update(LocalOutcome("a", None, False, None, True))
    assert bank.states["a"].blocked_mask == 0


def test_recent_neighbor_requires_a_prior_encounter_and_expires_after_ttl() -> None:
    bank = LocalMemoryBank(MemoryVariant.RECENT_ONLY, ["a"])
    assert bank.ready(LocalObservation("a", 0)) == (False, None)
    bank.update(LocalOutcome("a", 0, False, None, False))
    assert bank.ready(LocalObservation("a", 0)) == (True, RECENT)
    assert bank.ready(LocalObservation("a", 1)) == (False, None)
    for expected_age in (1, 2, 3):
        bank.update(LocalOutcome("a", None, False, None, False))
        assert bank.states["a"].recent_age == expected_age
    bank.update(LocalOutcome("a", None, False, None, False))
    assert not bank.states["a"].recent_valid
    assert bank.ready(LocalObservation("a", 0)) == (False, None)


def test_local_progress_timer_advances_only_on_that_actor_opportunities() -> None:
    bank = LocalMemoryBank(MemoryVariant.TIMER_ONLY, ["a", "b"])
    for _ in range(7):
        bank.update(LocalOutcome("a", None, False, None, False))
    assert bank.ready(LocalObservation("a", 1)) == (True, TIMER)
    assert bank.states["b"].local_no_progress == 0
    bank.update(LocalOutcome("a", None, False, None, True))
    assert bank.states["a"].local_no_progress == 0


def test_explicit_reset_clears_private_state_without_touching_native_state() -> None:
    scenario = _scenario()
    native = initial_checkpoint(scenario).to_run_state()
    native_before = native.to_dict()
    bank = LocalMemoryBank(MemoryVariant.ALL, [cell.cell_id for cell in scenario.cells])
    actor = scenario.cells[0].cell_id
    for _ in range(7):
        bank.update(LocalOutcome(actor, 1, True, 1, False))
    assert bank.state_dict() != LocalMemoryBank(
        MemoryVariant.ALL, [cell.cell_id for cell in scenario.cells]
    ).state_dict()
    bank.reset()
    fresh = LocalMemoryBank(
        MemoryVariant.ALL, [cell.cell_id for cell in scenario.cells]
    )
    assert bank.state_dict() == fresh.state_dict()
    assert native.to_dict() == native_before
    assert bank.validate_bounds()


def test_controller_uses_prior_state_then_repairs_with_one_costed_opportunity() -> None:
    scenario = _scenario()
    selected = scenario.cells[3].cell_id
    controller = LocalMemoryController(
        LocalMemoryContract(MemoryArm.ACTIVE, MemoryVariant.BLOCKED_ONLY),
        scenario,
        selected,
        3,
        10,
        retain_audits=True,
    )
    helper = _proposal(scenario, 2, target_pos=3)
    first = _offer(controller, scenario, helper, 10, eligible=True)
    assert first.decision == "rejected_s07_target_frozen"
    assert controller.frozen
    second = _offer(controller, scenario, helper, 11, eligible=True)
    assert second.decision == "rejected_s07_memory_repair_success_pending"
    assert controller.recovery_event == 12
    ledger = controller.process_ledger()
    assert ledger["freezeTargetBlocks"] == 1
    assert ledger["memoryGatedRepairProposals"] == 1
    assert ledger["actionUnitsSpent"] == ledger["energyUnitsSpent"] == 1
    assert ledger["nativeOpportunitiesSuppressed"] == 1


def test_memoryless_controls_preserve_cost_schedule_and_censor_semantics() -> None:
    scenario = _scenario()
    selected = scenario.cells[3].cell_id
    helper = _proposal(scenario, 2)
    immediate = LocalMemoryController(
        LocalMemoryContract(MemoryArm.IMMEDIATE), scenario, selected, 3, 10
    )
    decision = _offer(immediate, scenario, helper, 10, eligible=False)
    assert decision.decision == "rejected_s07_immediate_reference_success_pending"
    assert immediate.recovery_event == 11

    matched = LocalMemoryController(
        LocalMemoryContract(
            MemoryArm.MATCHED,
            MemoryVariant.ALL,
            assigned_duration=3,
        ),
        scenario,
        selected,
        3,
        10,
    )
    for event_index in (10, 11):
        _offer(matched, scenario, helper, event_index, eligible=False)
        assert matched.frozen
    decision = _offer(matched, scenario, helper, 12, eligible=False)
    assert decision.decision == "rejected_s07_matched_schedule_recovery_pending"
    assert matched.recovery_event == 13

    sentinel = LocalMemoryController(
        LocalMemoryContract(MemoryArm.MATCHED, MemoryVariant.ALL),
        scenario,
        selected,
        3,
        10,
    )
    permanent = LocalMemoryController(
        LocalMemoryContract(MemoryArm.PERMANENT), scenario, selected, 3, 10
    )
    for event_index in range(10, 30):
        _offer(sentinel, scenario, helper, event_index, eligible=False)
        _offer(permanent, scenario, helper, event_index, eligible=False)
    for controller in (sentinel, permanent):
        assert controller.frozen
        assert controller.recovery_event is None
        assert controller.process_ledger()["energyUnitsSpent"] == 0


def test_phase_budget_ledger_bounds_and_exact_replay() -> None:
    scenario = _scenario(policy=Policy.BUBBLE)
    checkpoint = initial_checkpoint(scenario)
    anchor = apply_lesion(
        "segment_reversal_central_v1",
        LesionState.from_checkpoint(scenario, checkpoint),
        s01_pairing_block_id="e05pb1:" + "e" * 64,
        timing_condition_id="initialization",
        injury_seed=765,
    )
    selected = anchor.post_state.occupancy[3]
    run = run_local_memory_phase(
        scenario,
        checkpoint,
        post_anchor_occupancy=anchor.post_state.occupancy,
        anchor_lesion_state_hash=anchor.post_state.state_hash,
        selected_identity=selected,
        contract=LocalMemoryContract(MemoryArm.ACTIVE, MemoryVariant.ALL),
        recovery_budget=6400,
        trace_mode="full",
        retain_process_audits=True,
    )
    assert all(run.opportunity_validation.values()), run.opportunity_validation
    assert run.summary["phaseActivationCount"] == run.process_ledger[
        "chargedOpportunities"
    ]
    assert run.summary["storageBitsPerIdentity"] == 11
    assert CONTROL_ASSIGNMENT_STREAM not in run.summary["streamCounterDelta"]
    exact_replay_local_memory(
        run,
        scenario,
        checkpoint,
        anchor.post_state.occupancy,
        recovery_budget=6400,
    )


@pytest.mark.parametrize("policy", [Policy.BUBBLE, Policy.INSERTION, Policy.SELECTION])
@pytest.mark.parametrize(
    ("arm", "variant", "duration"),
    [
        (MemoryArm.ACTIVE, MemoryVariant.FAILED_ONLY, None),
        (MemoryArm.ACTIVE, MemoryVariant.BLOCKED_ONLY, None),
        (MemoryArm.ACTIVE, MemoryVariant.RECENT_ONLY, None),
        (MemoryArm.ACTIVE, MemoryVariant.TIMER_ONLY, None),
        (MemoryArm.ACTIVE, MemoryVariant.ALL, None),
        (MemoryArm.PERMANENT, None, None),
        (MemoryArm.IMMEDIATE, None, None),
        (MemoryArm.MATCHED, MemoryVariant.ALL, 3),
        (MemoryArm.MATCHED, MemoryVariant.ALL, None),
    ],
)
def test_digest_summary_matches_full_state_ledgers_and_memory(
    policy: Policy,
    arm: MemoryArm,
    variant: MemoryVariant | None,
    duration: int | None,
) -> None:
    scenario = _scenario(policy=policy, seed=43)
    checkpoint = initial_checkpoint(scenario)
    anchor = apply_lesion(
        "segment_reversal_central_v1",
        LesionState.from_checkpoint(scenario, checkpoint),
        s01_pairing_block_id="e05pb1:" + "f" * 64,
        timing_condition_id="initialization",
        injury_seed=876,
    )
    selected = anchor.post_state.occupancy[3]
    kwargs = {
        "post_anchor_occupancy": anchor.post_state.occupancy,
        "anchor_lesion_state_hash": anchor.post_state.state_hash,
        "selected_identity": selected,
        "contract": LocalMemoryContract(arm, variant, assigned_duration=duration),
        "recovery_budget": 80,
    }
    full = run_local_memory_phase(
        scenario,
        checkpoint,
        **kwargs,
        trace_mode="full",
        retain_process_audits=True,
    )
    digest = run_local_memory_phase(
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
    assert all(digest.opportunity_validation.values()), digest.opportunity_validation
    for key in full.summary:
        if key not in {"traceMode", "retainedEventCount"}:
            assert digest.summary[key] == full.summary[key]
    assert full.events
    assert digest.events == ()
