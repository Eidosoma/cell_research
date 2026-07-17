from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, FaultMode, Policy, Proposal, ProposalKind
from reference_simulator.transition_primitives import ValidationDecision
from src.regeneration.lesions import LesionState, apply_lesion
from src.regeneration.nudge_recovery import (
    MATCHING_ASSIGNMENT_STREAM,
    NUDGE_RUN_SCHEMA_VERSION,
    MatchingChoice,
    NudgeMechanism,
    NudgeRecoveryContract,
    NudgeRecoveryController,
    RecoveryMode,
    exact_replay_nudge,
    run_nudge_phase,
    validate_nudge_spec,
)
from src.regeneration.tasks import initial_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s05_nudge_recovery.json"


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _scenario(*, policy: Policy = Policy.BUBBLE, seed: int = 23):
    return create_scenario(
        [7, 0, 6, 1, 5, 2, 4, 3],
        policy=policy,
        direction=Direction.ASCENDING,
        seed=seed,
        max_activations=20_000,
        generation_key=f"test/s05/{policy.value}/{seed}",
        permute=False,
    )


def _swap(scenario, actor_pos: int, target_pos: int) -> Proposal:
    return Proposal(
        ProposalKind.SWAP,
        scenario.cells[actor_pos].cell_id,
        actor_pos,
        target_pos=target_pos,
        observed_target_id=scenario.cells[target_pos].cell_id,
        reason="fixture_contact",
    )


def _controller(mechanism: NudgeMechanism):
    scenario = _scenario()
    selected = scenario.cells[3].cell_id
    controller = NudgeRecoveryController(
        NudgeRecoveryContract(mechanism),
        scenario,
        selected,
        0,
        retain_audits=True,
    )
    return scenario, selected, controller


def _offer(controller, scenario, proposal: Proposal, event_index: int):
    controller.prepare(proposal, event_index)
    decision = controller.outcome(
        proposal, ValidationDecision("valid_fixture", True), event_index
    )
    controller.after_batch(
        scenario,
        initial_checkpoint(scenario).to_run_state(),
        (proposal,),
        {proposal.ordinal: decision.decision},
        event_index,
    )
    return decision


def test_spec_freezes_four_mechanisms_split_matching_and_pressure_proxy() -> None:
    specification = _spec()
    validate_nudge_spec(specification)
    assert {item["mechanismId"] for item in specification["mechanisms"]} == {
        item.value for item in NudgeMechanism
    }
    assert set(specification["matching"]["choices"]) == {
        item.value for item in MatchingChoice
    }
    assert specification["pressureInterpretation"] == "abstract_event_count_proxy"
    assert MATCHING_ASSIGNMENT_STREAM.endswith("_s05_v1")
    assert NUDGE_RUN_SCHEMA_VERSION == "e05.s05.nudge-recovery-run.v1"


def test_contract_preserves_s04_runtime_boundary() -> None:
    scenario = _scenario()
    selected = scenario.cells[3].cell_id
    changed = replace(
        scenario,
        cells=(replace(scenario.cells[0], fault=FaultMode.STUCK), *scenario.cells[1:]),
    )
    with pytest.raises(ValueError, match="original normal"):
        NudgeRecoveryContract(NudgeMechanism.ATTEMPTED_CONTACT).validate(
            changed, selected
        )
    with pytest.raises(ValueError, match="information"):
        replace(
            NudgeRecoveryContract(NudgeMechanism.ATTEMPTED_CONTACT),
            information_permission="full_global_state",
        ).validate(scenario, selected)


def test_attempted_contact_counts_only_eligible_adjacent_swaps_and_recovers_next() -> None:
    scenario, _, controller = _controller(NudgeMechanism.ATTEMPTED_CONTACT)
    nonlocal_swap = _swap(scenario, 3, 6)
    controller.prepare(nonlocal_swap, 0)
    controller.outcome(nonlocal_swap, ValidationDecision("valid", True), 0)
    assert controller.contact_count == 0
    adjacent = _swap(scenario, 3, 2)
    for event_index in (1, 2, 3):
        decision = _offer(controller, scenario, adjacent, event_index)
        assert not decision.eligible_for_commit
    assert controller.contact_count == 3
    assert controller.recovery_event == 4
    assert not controller.frozen
    assert controller.process_ledger()["qualifyingContactEvents"] == 3


def test_pressure_proxy_counts_inbound_event_units_only() -> None:
    scenario, _, controller = _controller(NudgeMechanism.PRESSURE_PROXY)
    outbound = _swap(scenario, 3, 2)
    _offer(controller, scenario, outbound, 0)
    assert controller.contact_count == 1
    assert controller.pressure_units == 0
    inbound = _swap(scenario, 2, 3)
    for event_index in (1, 2, 3):
        _offer(controller, scenario, inbound, event_index)
    assert controller.pressure_units == 3
    assert controller.recovery_event == 4
    ledger = controller.process_ledger()
    assert ledger["pressureProxyUnits"] == ledger["inboundContactEvents"] == 3


def test_distinct_neighbor_uses_immutable_identity_not_side_or_attempt_count() -> None:
    scenario, _, controller = _controller(NudgeMechanism.DISTINCT_NEIGHBOR)
    left = _swap(scenario, 2, 3)
    _offer(controller, scenario, left, 0)
    _offer(controller, scenario, left, 1)
    assert controller.frozen
    assert controller.process_ledger()["distinctNeighborAdds"] == 1
    right = _swap(scenario, 4, 3)
    _offer(controller, scenario, right, 2)
    assert not controller.frozen
    assert controller.recovery_event == 3
    assert controller.distinct_neighbors == {
        scenario.cells[2].cell_id,
        scenario.cells[4].cell_id,
    }


def test_quorum_requires_distinct_left_right_inbound_contacts_inside_window() -> None:
    scenario, _, controller = _controller(NudgeMechanism.LOCAL_QUORUM)
    _offer(controller, scenario, _swap(scenario, 2, 3), 0)
    _offer(controller, scenario, _swap(scenario, 4, 3), 15)
    assert controller.recovery_event == 16

    scenario, _, expired = _controller(NudgeMechanism.LOCAL_QUORUM)
    _offer(expired, scenario, _swap(scenario, 2, 3), 0)
    _offer(expired, scenario, _swap(scenario, 4, 3), 16)
    assert expired.frozen
    assert expired.process_ledger()["quorumEvidenceExpirations"] == 1


def test_spontaneous_schedule_has_exact_duration_and_explicit_unrecovered_state() -> None:
    scenario = _scenario()
    selected = scenario.cells[3].cell_id
    finite = NudgeRecoveryController(
        NudgeRecoveryContract(
            NudgeMechanism.ATTEMPTED_CONTACT,
            mode=RecoveryMode.SPONTANEOUS_SCHEDULED,
            spontaneous_duration=3,
        ),
        scenario,
        selected,
        10,
    )
    proposal = Proposal(ProposalKind.NO_OP, scenario.cells[0].cell_id, 0)
    for event_index in (10, 11, 12):
        finite.prepare(proposal, event_index)
        assert finite.frozen
    finite.prepare(proposal, 13)
    assert finite.recovery_event == 13
    assert finite.process_ledger()["freezeExposures"] == 3

    censored = NudgeRecoveryController(
        NudgeRecoveryContract(
            NudgeMechanism.ATTEMPTED_CONTACT,
            mode=RecoveryMode.SPONTANEOUS_SCHEDULED,
            spontaneous_duration=None,
        ),
        scenario,
        selected,
        0,
    )
    for event_index in range(20):
        censored.prepare(proposal, event_index)
    assert censored.recovery_event is None
    assert censored.process_ledger()["freezeExposures"] == 20


def test_nudge_phase_replays_exactly_and_consumes_no_runtime_recovery_stream() -> None:
    scenario = _scenario(policy=Policy.BUBBLE)
    checkpoint = initial_checkpoint(scenario)
    anchor = apply_lesion(
        "segment_reversal_central_v1",
        LesionState.from_checkpoint(scenario, checkpoint),
        s01_pairing_block_id="e05pb1:" + "b" * 64,
        timing_condition_id="initialization",
        injury_seed=321,
    )
    selected = anchor.post_state.occupancy[3]
    run = run_nudge_phase(
        scenario,
        checkpoint,
        post_anchor_occupancy=anchor.post_state.occupancy,
        anchor_lesion_state_hash=anchor.post_state.state_hash,
        selected_identity=selected,
        contract=NudgeRecoveryContract(NudgeMechanism.ATTEMPTED_CONTACT),
        recovery_budget=6400,
        trace_mode="full",
        retain_process_audits=True,
    )
    assert all(run.opportunity_validation.values()), run.opportunity_validation
    assert run.start_event_index == checkpoint.activation_count
    assert run.summary["phaseActivationCount"] == run.process_ledger[
        "chargedOpportunities"
    ]
    assert MATCHING_ASSIGNMENT_STREAM not in run.summary["streamCounterDelta"]
    exact_replay_nudge(
        run,
        scenario,
        checkpoint,
        anchor.post_state.occupancy,
        recovery_budget=6400,
    )
