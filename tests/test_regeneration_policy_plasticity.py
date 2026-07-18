from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest

from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, FaultMode, Policy, Proposal, ProposalKind
from reference_simulator.transition_primitives import ValidationDecision
from src.regeneration.policy_plasticity import (
    BENCHMARK_VERSION,
    PLASTICITY_RUN_SCHEMA_VERSION,
    REVERSIBLE_MODE_OPPORTUNITIES,
    AdaptiveMeter,
    AdaptiveObservationGateway,
    Durability,
    LocalFailureBank,
    PlasticityArm,
    PlasticityContract,
    PlasticityController,
    PlasticityMechanism,
    PlasticityProposalRouter,
    PlasticityVariant,
    VARIANT_PROFILE,
    exact_replay_plasticity,
    run_plasticity_phase,
    validate_plasticity_spec,
)
from src.regeneration.tasks import initial_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s08_policy_plasticity.json"


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _scenario(
    *,
    policy: Policy = Policy.BUBBLE,
    seed: int = 83,
    sorted_values: bool = False,
):
    values = list(range(8)) if sorted_values else [7, 0, 6, 1, 5, 2, 4, 3]
    return create_scenario(
        values,
        policy=policy,
        direction=Direction.ASCENDING,
        seed=seed,
        max_activations=20_000,
        generation_key=f"test/s08/{policy.value}/{seed}/{sorted_values}",
        permute=False,
    )


def _proposal(scenario, actor_pos: int, target_pos: int | None = None) -> Proposal:
    actor_id = scenario.initial_occupancy[actor_pos]
    if target_pos is None:
        return Proposal(ProposalKind.NO_OP, actor_id, actor_pos, reason="fixture")
    return Proposal(
        ProposalKind.SWAP,
        actor_id,
        actor_pos,
        target_pos=target_pos,
        observed_target_id=scenario.initial_occupancy[target_pos],
        reason="fixture",
    )


def _offer(
    controller: PlasticityController,
    proposal: Proposal,
    event_index: int,
    *,
    eligible: bool,
) -> str:
    controller.prepare(proposal, event_index)
    decision = controller.outcome(
        proposal,
        ValidationDecision("fixture_valid" if eligible else "fixture_noop", eligible),
        event_index,
    )
    final = "accepted" if decision.eligible_for_commit else decision.decision
    controller.after_opportunity(proposal, final, event_index)
    return final


def test_spec_freezes_variants_controls_costs_and_complete_panel() -> None:
    specification = _spec()
    validate_plasticity_spec(specification)
    assert specification["benchmarkVersion"] == BENCHMARK_VERSION
    assert {item["variantId"] for item in specification["variants"]} == {
        item.value for item in PlasticityVariant
    }
    assert {item["mechanismId"] for item in specification["mechanisms"]} == {
        item.value for item in PlasticityMechanism
    }
    assert {item["durability"] for item in specification["variants"]} == {
        item.value for item in Durability
    }
    assert specification["validationPanel"]["plannedRunCount"] == 7344
    assert specification["validationPanel"]["plannedPrimaryRecoveryPairs"] == 1536
    assert specification["validationPanel"]["plannedPrimaryStabilityPairs"] == 384
    assert specification["pairingAndStreams"]["runtimeStreams"] == []
    assert PLASTICITY_RUN_SCHEMA_VERSION == "e05.s08.policy-plasticity-run.v1"


def test_contract_rejects_faulted_scenario_global_information_and_wrong_phase() -> None:
    scenario = _scenario()
    selected = scenario.initial_occupancy[3]
    faulted = replace(
        scenario,
        cells=(replace(scenario.cells[0], fault=FaultMode.STUCK), *scenario.cells[1:]),
    )
    contract = PlasticityContract(
        PlasticityArm.ACTIVE,
        PlasticityVariant.ALGOTYPE_REV,
        "injury",
        4,
    )
    with pytest.raises(ValueError, match="original normal"):
        contract.validate(faulted, selected)
    with pytest.raises(ValueError, match="information"):
        replace(contract, native_information_permission="full_global_state").validate(
            scenario, selected
        )
    with pytest.raises(ValueError, match="positive prior-S07"):
        replace(contract, assigned_recovery_duration=None).validate(scenario, selected)
    with pytest.raises(ValueError, match="no injury identity"):
        replace(contract, phase="stability", assigned_recovery_duration=None).validate(
            scenario, selected
        )


def test_failure_trigger_is_prior_state_bounded_actor_local_and_resettable() -> None:
    bank = LocalFailureBank(["a", "b"])
    assert not bank.ready("a")
    for _ in range(3):
        bank.update("a", False)
    assert bank.ready("a")
    assert not bank.ready("b")
    bank.update("a", False)
    assert bank.states["a"] == 3
    assert bank.ledger["failureCounterSaturations"] == 1
    bank.update("a", True)
    assert bank.states["a"] == 0
    assert bank.validate_bounds()
    assert set(LocalFailureBank.__slots__) == {"states", "ledger"}


def test_transition_is_later_opportunity_one_cycle_and_reversible_after_eight() -> None:
    scenario = _scenario()
    contract = PlasticityContract(
        PlasticityArm.ACTIVE,
        PlasticityVariant.EXPLORE_REV,
        "stability",
    )
    controller = PlasticityController(contract, scenario, None, 0)
    proposal = _proposal(scenario, 3)
    for event_index in range(3):
        assert _offer(controller, proposal, event_index, eligible=False) == "fixture_noop"
        assert not controller.ever_transitioned
    assert _offer(controller, proposal, 3, eligible=False) == "rejected_s08_mode_transition"
    assert controller.ever_transitioned and controller.mode_active
    assert controller.transition_event == 3
    assert controller.mode_opportunities == 0
    for ordinal in range(REVERSIBLE_MODE_OPPORTUNITIES):
        controller.record_mode_meter(AdaptiveMeter(2, 0, 1, (2,)))
        _offer(controller, proposal, 4 + ordinal, eligible=False)
    assert controller.mode_opportunities == REVERSIBLE_MODE_OPPORTUNITIES
    assert controller.reversion_due(proposal.actor_id)
    assert _offer(controller, proposal, 12, eligible=False) == "rejected_s08_mode_reversion"
    assert not controller.mode_active
    assert controller.reversion_event == 12
    assert controller.transition_actions_remaining == 0
    for event_index in range(13, 20):
        _offer(controller, proposal, event_index, eligible=False)
    assert controller.ledger["modeTransitions"] == 1
    assert controller.ledger["modeReversions"] == 1


@pytest.mark.parametrize("policy", list(Policy))
def test_algotype_switch_mapping_uses_proposal_only_view_without_scenario_mutation(
    policy: Policy,
) -> None:
    scenario = _scenario(policy=policy)
    state = initial_checkpoint(scenario).to_run_state()
    actor_id = scenario.initial_occupancy[3]
    gateway = AdaptiveObservationGateway(scenario, state)
    proposal, meter = gateway.observe_or_propose(
        actor_id,
        PlasticityMechanism.ALGOTYPE,
        0,
        None,
        emit=True,
    )
    assert proposal is not None
    expected = {
        Policy.BUBBLE: "Bubble_to_Insertion",
        Policy.INSERTION: "Insertion_to_Bubble",
        Policy.SELECTION: "Selection_to_Bubble",
    }[policy]
    assert expected in proposal.reason
    assert scenario.cell_map[actor_id].policy == policy
    assert meter.reads >= 1


def test_radius_target_and_exploration_permissions_are_bounded() -> None:
    scenario = _scenario()
    state = initial_checkpoint(scenario).to_run_state()
    actor_id = scenario.initial_occupancy[3]
    gateway = AdaptiveObservationGateway(scenario, state)
    _, radius = gateway.observe_or_propose(
        actor_id,
        PlasticityMechanism.RADIUS,
        0,
        "left",
        emit=True,
    )
    target_proposal, target = gateway.observe_or_propose(
        actor_id,
        PlasticityMechanism.TARGET,
        0,
        "left",
        emit=True,
    )
    explore_proposal, explore = gateway.observe_or_propose(
        actor_id,
        PlasticityMechanism.EXPLORE,
        0,
        "left",
        emit=True,
    )
    assert radius.maximum_radius == 2
    assert all(abs(position - 3) <= 2 for position in radius.visible_positions)
    assert target.maximum_radius == 1
    assert target_proposal is not None
    assert explore.maximum_radius == 1 and explore.comparisons == 0
    assert explore_proposal is not None
    assert explore_proposal.kind == ProposalKind.SWAP


def test_active_and_sham_share_transition_and_cost_surface_but_not_behavior() -> None:
    scenario = _scenario(sorted_values=True, seed=97)
    checkpoint = initial_checkpoint(scenario)
    active_contract = PlasticityContract(
        PlasticityArm.ACTIVE,
        PlasticityVariant.EXPLORE_IRREV,
        "stability",
    )
    sham_contract = replace(active_contract, arm=PlasticityArm.SHAM)
    active = run_plasticity_phase(
        scenario,
        checkpoint,
        occupancy=checkpoint.occupancy,
        anchor_lesion_state_hash=None,
        selected_identity=None,
        contract=active_contract,
        phase_budget=20 * len(scenario.cells),
    )
    sham = run_plasticity_phase(
        scenario,
        checkpoint,
        occupancy=checkpoint.occupancy,
        anchor_lesion_state_hash=None,
        selected_identity=None,
        contract=sham_contract,
        phase_budget=20 * len(scenario.cells),
    )
    assert active.process_final_state["transitionEventIndex"] == sham.process_final_state[
        "transitionEventIndex"
    ]
    assert active.process_ledger["actionUnitsSpent"] == sham.process_ledger[
        "actionUnitsSpent"
    ]
    assert active.process_ledger["nativeOpportunitiesSuppressed"] == sham.process_ledger[
        "nativeOpportunitiesSuppressed"
    ]
    assert active.summary["anyTargetDeparture"]
    assert not sham.summary["anyTargetDeparture"]


def test_injury_recovery_schedule_costs_and_exact_replay() -> None:
    scenario = _scenario(seed=101)
    checkpoint = initial_checkpoint(scenario)
    selected = checkpoint.occupancy[3]
    post_anchor = list(checkpoint.occupancy)
    post_anchor[2:6] = reversed(post_anchor[2:6])
    contract = PlasticityContract(
        PlasticityArm.ACTIVE,
        PlasticityVariant.TARGET_REV,
        "injury",
        assigned_recovery_duration=4,
    )
    run = run_plasticity_phase(
        scenario,
        checkpoint,
        occupancy=post_anchor,
        anchor_lesion_state_hash="fixture-anchor",
        selected_identity=selected,
        contract=contract,
        phase_budget=2_000,
        retain_trace=True,
    )
    assert run.summary["recoveryObserved"]
    assert run.summary["recoveryDuration"] == 4
    assert run.process_ledger["suppressed::recovery"] == 1
    assert all(run.opportunity_validation.values())
    exact_replay_plasticity(run, scenario, checkpoint, post_anchor, 2_000)


def test_observability_surfaces_exclude_global_progress_target_and_clock() -> None:
    source = inspect.getsource(LocalFailureBank)
    forbidden = ("global_progress", "target_distance", "event_index", "occupancy")
    assert all(item not in source for item in forbidden)
    signature = inspect.signature(AdaptiveObservationGateway.observe_or_propose)
    assert set(signature.parameters) == {
        "self",
        "actor_id",
        "mechanism",
        "mode_count",
        "native_side",
        "emit",
    }
    assert set(PlasticityProposalRouter.__slots__) == {"controller", "native_router"}
    assert all(VARIANT_PROFILE[item][0] in PlasticityMechanism for item in PlasticityVariant)
