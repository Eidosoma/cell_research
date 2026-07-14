from __future__ import annotations

from dataclasses import fields
import inspect
from itertools import permutations, product
import json
from pathlib import Path

import pytest

from causal_simulator.action_interface import CommonActionInterface, ControlTopology
from causal_simulator.architectures import (
    PRESPECIFICATION_SHA256,
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
    ControlArchitecture,
    CoordinatorProfile,
    CoordinatorSignal,
    FrozenWeakCoordinator,
    WeakCoordinatorParameters,
    coordinator_input_fields,
    exact_replay_architecture,
    run_architecture,
)
from reference_simulator.api import create_scenario
from reference_simulator.engine import initial_state
from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    ProposalKind,
    Scenario,
)
from reference_simulator.policies import cell_view_proposal


FIXTURES = Path(__file__).parent / "fixtures" / "s03_architecture_fixtures.json"


def contract_for(architecture: str, coordinator: str) -> ArchitectureExecutionContract:
    key = (architecture, coordinator)
    contracts = {
        ("distributed_local", "none"): ArchitectureExecutionContract.distributed_local(),
        (
            "central_local_proposal_k1",
            "common_validator_only",
        ): ArchitectureExecutionContract.central_local_k1(),
        (
            "distributed_weak_coordinator",
            "none",
        ): ArchitectureExecutionContract.distributed_weak(enabled=False),
        (
            "distributed_weak_coordinator",
            "weak_frozen_budget",
        ): ArchitectureExecutionContract.distributed_weak(),
    }
    return contracts[key]


def test_hand_checked_architecture_fixtures() -> None:
    cases = json.loads(FIXTURES.read_text())["cases"]
    for item in cases:
        policy = Policy(item["policy"])
        direction = Direction(item["direction"])
        scenario = Scenario.create(
            tuple(Cell(cell_id, value, policy, direction) for cell_id, value in item["cells"]),
            initial_occupancy=tuple(cell_id for cell_id, _ in item["cells"]),
            max_activations=100,
            generation_key="E02-S03/fixture/" + item["id"],
        )
        state = initial_state(scenario)
        state.activation_count = item["activationCount"]
        router = ArchitectureProposalRouter(
            contract_for(item["architecture"], item["coordinator"])
        )
        proposal = router.proposal_for(
            scenario,
            state,
            item["actor"],
            side=item["side"],
        )
        ledger = router.architecture_ledger()
        assert proposal.kind.value == item["expectedKind"], item["id"]
        assert proposal.reason == item["expectedReason"], item["id"]
        assert ledger["coordinatorEligibleDecisions"] == item["expectedEligibleDecisions"]
        assert ledger["coordinatorInterventions"] == item["expectedInterventions"]
        assert ledger["coordinatorMessages"] == 2 * item["expectedEligibleDecisions"]
        assert ledger["coordinatorMessageBits"] == 2 * item["expectedEligibleDecisions"]


def test_coordinator_has_only_the_prespecified_typed_input() -> None:
    assert coordinator_input_fields() == ("event_index", "is_nonlocal_swap")
    assert {item.name for item in fields(CoordinatorSignal)} == {
        "event_index",
        "is_nonlocal_swap",
    }
    signature = inspect.signature(FrozenWeakCoordinator.decide)
    assert tuple(signature.parameters) == ("self", "signal")
    coordinator = FrozenWeakCoordinator(WeakCoordinatorParameters())
    for forbidden in ("scenario", "state", "envelope", "occupancy", "cells", "ledger"):
        assert not hasattr(coordinator, forbidden)
    with pytest.raises(TypeError, match="CoordinatorSignal"):
        coordinator.decide(object())  # type: ignore[arg-type]


def test_frozen_contract_hash_and_frequency_parameter_validation() -> None:
    assert PRESPECIFICATION_SHA256 == (
        "c87356a66bbd4f4c072ad23513611701047cca0178a3ec533478e01194b2c15f"
    )
    default = WeakCoordinatorParameters()
    assert default.is_prespecified_s03_profile
    assert [index for index in range(24) if default.eligible(index)] == [7, 15, 23]
    alternate_frequency = WeakCoordinatorParameters(period_opportunities=4, phase_zero_based=1)
    assert not alternate_frequency.is_prespecified_s03_profile
    assert [index for index in range(12) if alternate_frequency.eligible(index)] == [1, 5, 9]
    with pytest.raises(ValueError, match="period"):
        WeakCoordinatorParameters(period_opportunities=0)
    with pytest.raises(ValueError, match="phase"):
        WeakCoordinatorParameters(period_opportunities=4, phase_zero_based=4)
    with pytest.raises(ValueError, match="one bit"):
        WeakCoordinatorParameters(inbound_payload_bits=2)


def test_contract_rejects_boundary_weakening_and_invalid_architecture_pairings() -> None:
    cell_view = create_scenario(
        [2, 1],
        policy=Policy.BUBBLE,
        generation_key="E02-S03/contract/cell",
        permute=False,
    )
    traditional = create_scenario(
        [2, 1],
        policy=Policy.BUBBLE,
        architecture=Architecture.TRADITIONAL,
        generation_key="E02-S03/contract/traditional",
        permute=False,
    )
    ArchitectureExecutionContract.distributed_local().validate(cell_view)
    ArchitectureExecutionContract.central_global_legacy().validate(traditional)
    with pytest.raises(ValueError, match="reject full-global"):
        ArchitectureExecutionContract.distributed_local().validate(traditional)
    with pytest.raises(ValueError, match="traditional scenario"):
        ArchitectureExecutionContract.central_global_legacy().validate(cell_view)
    with pytest.raises(ValueError, match="legal primitive"):
        ArchitectureExecutionContract(
            ControlArchitecture.DISTRIBUTED_LOCAL,
            CoordinatorProfile.NONE,
            legal_primitives=("Swap",),
        ).validate(cell_view)
    with pytest.raises(ValueError, match="incompatible"):
        ArchitectureExecutionContract(
            ControlArchitecture.CENTRAL_LOCAL_PROPOSAL_K1,
            CoordinatorProfile.WEAK_FROZEN_BUDGET,
            WeakCoordinatorParameters(),
        ).validate(cell_view)


def test_k1_selector_still_rejects_multiple_candidate_envelopes() -> None:
    scenario = create_scenario(
        [2, 1],
        policy=Policy.BUBBLE,
        generation_key="E02-S03/k1-selector",
        permute=False,
    )
    interface = CommonActionInterface()
    envelope = interface.envelope_for(
        ControlTopology.DISTRIBUTED_LOCAL,
        scenario,
        initial_state(scenario),
        "cell-0000",
        side="right",
    )
    with pytest.raises(ValueError, match="exactly one"):
        interface.central_relay.forward_one((envelope, envelope))


def test_exhaustive_small_state_matched_primitives_and_weak_veto_rule() -> None:
    interface = CommonActionInterface()
    checked = 0
    interventions = 0
    for policy, direction, order, fault_values in product(
        tuple(Policy),
        tuple(Direction),
        permutations((0, 1, 2)),
        product(tuple(FaultMode), repeat=3),
    ):
        cells = tuple(
            Cell(f"c{index}", value, policy, direction, fault)
            for index, (value, fault) in enumerate(zip(order, fault_values))
        )
        scenario = Scenario.create(
            cells,
            initial_occupancy=("c0", "c1", "c2"),
            generation_key=f"E02-S03/exhaustive/{policy}/{direction}/{order}/{fault_values}",
        )
        state = initial_state(scenario)
        state.activation_count = 7
        for actor_id in state.occupancy:
            sides = ("left", "right") if policy == Policy.BUBBLE else (None,)
            for side in sides:
                reference = cell_view_proposal(scenario, state, actor_id, side=side)
                direct = ArchitectureProposalRouter(
                    ArchitectureExecutionContract.distributed_local()
                ).proposal_for(scenario, state, actor_id, side=side)
                central = ArchitectureProposalRouter(
                    ArchitectureExecutionContract.central_local_k1()
                ).proposal_for(scenario, state, actor_id, side=side)
                weak_none = ArchitectureProposalRouter(
                    ArchitectureExecutionContract.distributed_weak(enabled=False)
                ).proposal_for(scenario, state, actor_id, side=side)
                weak_router = ArchitectureProposalRouter(
                    ArchitectureExecutionContract.distributed_weak()
                )
                weak = weak_router.proposal_for(scenario, state, actor_id, side=side)
                assert direct.to_dict() == reference.to_dict()
                assert central.to_dict() == reference.to_dict()
                assert weak_none.to_dict() == reference.to_dict()
                is_nonlocal_swap = (
                    reference.kind == ProposalKind.SWAP
                    and reference.target_pos is not None
                    and abs(reference.target_pos - reference.actor_pos) > 1
                )
                if is_nonlocal_swap:
                    interventions += 1
                    assert weak.kind == ProposalKind.NO_OP
                    assert weak.reason == "coordinator_veto_nonlocal_swap"
                    assert weak.observation_reads == reference.observation_reads
                    assert weak.value_comparisons == reference.value_comparisons
                    assert weak.target_pos is None and weak.new_cursor is None
                else:
                    assert weak.to_dict() == reference.to_dict()
                assert weak_router.architecture_ledger()["coordinatorEligibleDecisions"] == 1
                checked += 1
    assert checked == 3888
    assert interventions > 0


@pytest.mark.parametrize("policy", tuple(Policy))
@pytest.mark.parametrize("direction", tuple(Direction))
def test_no_fault_completion_replay_ledger_and_residual_parity(
    policy: Policy, direction: Direction
) -> None:
    values = [6, 1, 5, 2, 4, 3]
    scenario = create_scenario(
        values,
        policy=policy,
        direction=direction,
        seed=20260714,
        max_activations=20_000,
        generation_key=f"E02-S03/completion/{policy.value}/{direction.value}/cell",
        permute=False,
    )
    contracts = (
        ArchitectureExecutionContract.distributed_local(),
        ArchitectureExecutionContract.central_local_k1(),
        ArchitectureExecutionContract.distributed_weak(enabled=False),
        ArchitectureExecutionContract.distributed_weak(),
    )
    runs = [run_architecture(scenario, contract, trace_mode="full") for contract in contracts]
    for item in runs:
        assert item.result.summary["completed"]
        assert all(item.opportunity_validation().values()), item.opportunity_validation()
        assert exact_replay_architecture(item).to_json_bytes() == item.to_json_bytes()
        assert {
            event["proposal"]["kind"] for event in item.result.events
        } <= {"NoOp", "Swap", "MemoryUpdate"}
    assert runs[0].result.to_json_bytes() == runs[1].result.to_json_bytes()
    assert runs[0].result.to_json_bytes() == runs[2].result.to_json_bytes()
    if policy in {Policy.BUBBLE, Policy.INSERTION}:
        assert runs[0].result.to_json_bytes() == runs[3].result.to_json_bytes()

    traditional = create_scenario(
        values,
        policy=policy,
        architecture=Architecture.TRADITIONAL,
        direction=direction,
        seed=20260714,
        max_activations=20_000,
        generation_key=f"E02-S03/completion/{policy.value}/{direction.value}/traditional",
        permute=False,
    )
    legacy = run_architecture(
        traditional,
        ArchitectureExecutionContract.central_global_legacy(),
        trace_mode="full",
    )
    assert legacy.result.summary["completed"]
    assert not legacy.contract.matched_contrast_eligible
    assert all(legacy.opportunity_validation().values())
    assert exact_replay_architecture(legacy).to_json_bytes() == legacy.to_json_bytes()
