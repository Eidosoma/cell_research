from __future__ import annotations

from dataclasses import fields
from itertools import permutations, product
import json
from pathlib import Path

import pytest

from causal_simulator.action_interface import (
    ActionEnvelope,
    CommonActionInterface,
    ControlTopology,
    ForbiddenInformationError,
    InformationPermission,
    OperationMeter,
    PolicyNativeReadGateway,
    ReadCapability,
    VisibleCell,
)
from causal_simulator.engine import MatchedExecutionContract, run_with_contract
from reference_simulator.api import exact_replay
from reference_simulator.engine import initial_state
from reference_simulator.model import Architecture, Cell, Direction, FaultMode, Policy, Scenario
from reference_simulator.policies import cell_view_proposal
from reference_simulator.transition_primitives import (
    commit_proposal,
    cost_delta,
    ledger_identity,
    validate_proposal,
)


FIXTURES = Path(__file__).parent / "fixtures" / "s02_action_interface_fixtures.json"


def scenario_from_fixture(item: dict) -> tuple[Scenario, object]:
    default_policy = Policy(item["policy"])
    direction = Direction(item["direction"])
    cells = [
        Cell(
            cell_id=raw[0],
            value=raw[1],
            fault=FaultMode(raw[2]),
            policy=Policy(raw[3]) if len(raw) == 4 else default_policy,
            direction=direction,
            analysis_label="forbidden-label",
        )
        for raw in item["cells"]
    ]
    scenario = Scenario.create(
        cells,
        initial_occupancy=[raw[0] for raw in item["cells"]],
        generation_key="E02-S02/fixture/" + item["id"],
        max_activations=50,
    )
    state = initial_state(scenario)
    state.selection_cursors.update(item.get("stateCursors", {}))
    return scenario, state


def test_hand_checked_differential_fixtures() -> None:
    interface = CommonActionInterface()
    fixtures = json.loads(FIXTURES.read_text())["fixtures"]
    for item in fixtures:
        scenario, state = scenario_from_fixture(item)
        kwargs = {"side": item.get("side")}
        reference = cell_view_proposal(scenario, state, item["actor"], **kwargs)
        distributed = interface.envelope_for(
            ControlTopology.DISTRIBUTED_LOCAL, scenario, state, item["actor"], **kwargs
        )
        central = interface.envelope_for(
            ControlTopology.CENTRAL_LOCAL_PROPOSAL_K1,
            scenario,
            state,
            item["actor"],
            **kwargs,
        )
        expected = item["expected"]
        assert distributed == central, item["id"]
        assert distributed.proposal.to_dict() == reference.to_dict(), item["id"]
        proposal = distributed.proposal
        assert proposal.kind.value == expected["kind"], item["id"]
        assert proposal.reason == expected["reason"], item["id"]
        assert proposal.observation_reads == expected["reads"], item["id"]
        assert proposal.value_comparisons == expected["comparisons"], item["id"]

        validation = validate_proposal(scenario, state, proposal)
        decision = "accepted" if validation.eligible_for_commit else validation.decision
        assert decision == expected["decision"], item["id"]
        delta = cost_delta(state.ledger, proposal, decision)
        snapshot = state.clone()
        commit_proposal(state, snapshot, proposal, decision)
        for key, value in delta.items():
            state.ledger[key] += value
        assert state.occupancy == expected["postOccupancy"], item["id"]
        if "newCursor" in expected:
            assert state.selection_cursors[item["actor"]] == expected["newCursor"], item["id"]
        assert all(ledger_identity(state.ledger).values()), item["id"]


def test_exhaustive_small_state_primitive_and_topology_parity() -> None:
    interface = CommonActionInterface()
    checked = 0
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
            generation_key=f"E02-S02/exhaustive/{policy.value}/{direction.value}/{order}/{fault_values}",
        )
        state = initial_state(scenario)
        for actor_id in state.occupancy:
            sides = ("left", "right") if policy == Policy.BUBBLE else (None,)
            for side in sides:
                kwargs = {"side": side}
                reference = cell_view_proposal(scenario, state, actor_id, **kwargs)
                distributed = interface.proposal_for(
                    ControlTopology.DISTRIBUTED_LOCAL, scenario, state, actor_id, **kwargs
                )
                central = interface.proposal_for(
                    ControlTopology.CENTRAL_LOCAL_PROPOSAL_K1,
                    scenario,
                    state,
                    actor_id,
                    **kwargs,
                )
                assert distributed.to_dict() == reference.to_dict()
                assert central.to_dict() == reference.to_dict()
                checked += 1
    assert checked == 3888


@pytest.mark.parametrize("policy", tuple(Policy))
def test_forbidden_information_is_denied_before_read(policy: Policy) -> None:
    scenario = Scenario.create(
        (
            Cell("actor", 2, policy, analysis_label="secret-actor"),
            Cell("target", 1, Policy.BUBBLE, analysis_label="secret-target"),
        ),
        initial_occupancy=("actor", "target"),
        generation_key="E02-S02/permissions/" + policy.value,
    )
    gateway = PolicyNativeReadGateway(scenario, initial_state(scenario), "actor", OperationMeter())
    forbidden = set(ReadCapability) - gateway.allowed_capabilities
    for capability in forbidden:
        with pytest.raises(ForbiddenInformationError):
            gateway.read(capability)
    assert {field.name for field in fields(VisibleCell)} == {"position", "value", "fault"}
    assert not hasattr(gateway, "scenario")
    assert not hasattr(gateway, "state")


def test_central_relay_rejects_zero_or_multiple_proposals() -> None:
    scenario = Scenario.create(
        (Cell("a", 2, Policy.BUBBLE), Cell("b", 1, Policy.BUBBLE)),
        initial_occupancy=("a", "b"),
        generation_key="E02-S02/k1",
    )
    interface = CommonActionInterface()
    envelope = interface.envelope_for(
        ControlTopology.DISTRIBUTED_LOCAL,
        scenario,
        initial_state(scenario),
        "a",
        side="right",
    )
    with pytest.raises(ValueError, match="exactly one"):
        interface.central_relay.forward_one(())
    with pytest.raises(ValueError, match="exactly one"):
        interface.central_relay.forward_one((envelope, envelope))
    with pytest.raises(TypeError, match="ActionEnvelope"):
        interface.central_relay.forward_one((object(),))  # type: ignore[arg-type]


def test_common_validator_rejects_stale_observed_target_identity() -> None:
    scenario = Scenario.create(
        (
            Cell("a", 3, Policy.BUBBLE),
            Cell("b", 1, Policy.BUBBLE),
            Cell("c", 2, Policy.BUBBLE),
        ),
        initial_occupancy=("a", "b", "c"),
        generation_key="E02-S02/stale-target",
    )
    state = initial_state(scenario)
    envelope = CommonActionInterface().envelope_for(
        ControlTopology.CENTRAL_LOCAL_PROPOSAL_K1,
        scenario,
        state,
        "a",
        side="right",
    )
    assert envelope.observed_target_id == "b"
    state.occupancy[1], state.occupancy[2] = state.occupancy[2], state.occupancy[1]
    decision = validate_proposal(scenario, state, envelope.proposal)
    assert decision.decision == "rejected_stale_target"
    assert not decision.eligible_for_commit


def test_unimplemented_information_or_global_controller_cannot_enter_matched_run() -> None:
    cell_view = Scenario.create(
        (Cell("a", 2, Policy.BUBBLE), Cell("b", 1, Policy.BUBBLE)),
        initial_occupancy=("a", "b"),
        generation_key="E02-S02/contract/cell",
    )
    with pytest.raises(ValueError, match="policy_native_local"):
        run_with_contract(
            cell_view,
            MatchedExecutionContract(
                ControlTopology.CENTRAL_LOCAL_PROPOSAL_K1,
                information_permission=InformationPermission.FULL_GLOBAL_STATE,
            ),
        )
    traditional = Scenario.create(
        (Cell("a", 2, Policy.BUBBLE), Cell("b", 1, Policy.BUBBLE)),
        initial_occupancy=("a", "b"),
        architecture=Architecture.TRADITIONAL,
        traditional_policy=Policy.BUBBLE,
        generation_key="E02-S02/contract/traditional",
    )
    with pytest.raises(ValueError, match="central_global_legacy"):
        run_with_contract(
            traditional,
            MatchedExecutionContract(ControlTopology.DISTRIBUTED_LOCAL),
        )


@pytest.mark.parametrize("fault_mode", [FaultMode.PASSIVE, FaultMode.STUCK])
def test_fault_diagnostics_replay_and_ledger_identity(fault_mode: FaultMode) -> None:
    scenario = Scenario.create(
        (
            Cell("a", 4, Policy.BUBBLE),
            Cell("b", 1, Policy.BUBBLE, fault=fault_mode),
            Cell("c", 3, Policy.BUBBLE),
            Cell("d", 2, Policy.BUBBLE),
        ),
        initial_occupancy=("a", "b", "c", "d"),
        seed=99,
        max_activations=100,
        generation_key="E02-S02/fault-diagnostic/" + fault_mode.value,
    )
    runs = [
        run_with_contract(scenario, MatchedExecutionContract(topology), trace_mode="full")
        for topology in ControlTopology
    ]
    assert runs[0].transition_bytes() == runs[1].transition_bytes()
    assert runs[0].result.to_json_bytes() == runs[1].result.to_json_bytes()
    for item in runs:
        assert all(ledger_identity(item.result.summary["ledger"]).values())
        assert exact_replay(item.result).to_json_bytes() == item.result.to_json_bytes()
