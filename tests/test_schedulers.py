from __future__ import annotations

from collections import Counter, defaultdict
import inspect
import json
from pathlib import Path

import pytest

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.schedulers import (
    PERMUTATION_STREAM,
    SCHEDULER_PRESPECIFICATION_SHA256,
    FrozenSchedulerController,
    SchedulerExecutionContract,
    SchedulerFamily,
    exact_replay_scheduler,
    run_scheduled_architecture,
    scheduler_controller_fields,
)
from reference_simulator.api import create_scenario
from reference_simulator.model import Architecture, Direction, Policy
from reference_simulator.scheduler import ScheduledOpportunity


FIXTURES = Path(__file__).parent / "fixtures" / "s04_scheduler_fixtures.json"


def _controller(family: SchedulerFamily, n: int = 8, seed: int = 19):
    return FrozenSchedulerController(
        SchedulerExecutionContract(family),
        seed=seed,
        scenario_id="S04-controller-fixture",
        actor_ids=tuple(f"c{index:02d}" for index in range(n)),
    )


def _sequence(family: SchedulerFamily, opportunities: int, n: int = 8) -> tuple[str, ...]:
    controller = _controller(family, n=n)
    result: list[str] = []
    while len(result) < opportunities:
        slots = controller(len(result), opportunities - len(result))
        result.extend(slot.actor_id for slot in slots)
    return tuple(result)


def _gaps(sequence: tuple[str, ...]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, actor_id in enumerate(sequence):
        positions[actor_id].append(index)
    return {
        actor_id: [right - left for left, right in zip(values, values[1:])]
        for actor_id, values in positions.items()
    }


def test_prespecification_hash_and_narrow_controller_surface() -> None:
    assert SCHEDULER_PRESPECIFICATION_SHA256 == (
        "2fb3e94908c5fa40f529abe1a3a21ad3e5d8d8390105dcb958102aae6a81720e"
    )
    assert tuple(inspect.signature(FrozenSchedulerController.__call__).parameters) == (
        "self",
        "event_index",
        "remaining_opportunities",
    )
    fields = set(scheduler_controller_fields())
    assert fields == {
        "contract",
        "seed",
        "scenario_id",
        "actor_ids",
        "audits",
        "last_activation",
        "cached_sweep_index",
        "cached_permutation",
        "cached_permutation_draws",
    }
    for forbidden in (
        "scenario",
        "state",
        "occupancy",
        "cells",
        "values",
        "policies",
        "faults",
        "proposals",
        "ledger",
    ):
        assert forbidden not in fields


def test_frozen_portable_scheduler_sequences() -> None:
    fixture = json.loads(FIXTURES.read_text())
    for family_name, expected in fixture["expectedSequences"].items():
        controller = FrozenSchedulerController(
            SchedulerExecutionContract(SchedulerFamily(family_name)),
            seed=fixture["seed"],
            scenario_id=fixture["scenarioId"],
            actor_ids=tuple(fixture["actorIds"]),
        )
        observed: list[str] = []
        batches: list[list[str]] = []
        while len(observed) < fixture["opportunities"]:
            slots = controller(
                len(observed), fixture["opportunities"] - len(observed)
            )
            batch = [slot.actor_id for slot in slots]
            batches.append(batch)
            observed.extend(batch)
        assert observed == expected
        if family_name == "synchronous_batch_deterministic_conflict":
            assert batches == fixture["synchronousExpectedBatches"]


def test_contract_rejects_boundary_weakening_and_legacy_global() -> None:
    cell_view = create_scenario(
        [2, 1], policy=Policy.BUBBLE, permute=False, generation_key="S04/contract/cell"
    )
    architecture = ArchitectureExecutionContract.distributed_local()
    SchedulerExecutionContract(SchedulerFamily.DETERMINISTIC_SCAN).validate(
        cell_view, architecture
    )
    with pytest.raises(ValueError, match="free policy proposal"):
        SchedulerExecutionContract(
            SchedulerFamily.DETERMINISTIC_SCAN,
            proposal_candidates_per_opportunity=2,
        ).validate(cell_view, architecture)
    with pytest.raises(ValueError, match="policy information"):
        SchedulerExecutionContract(
            SchedulerFamily.DETERMINISTIC_SCAN,
            policy_information_permission="full_global_state",
        ).validate(cell_view, architecture)
    with pytest.raises(ValueError, match="legal primitive"):
        SchedulerExecutionContract(
            SchedulerFamily.DETERMINISTIC_SCAN,
            legal_primitives=("NoOp", "Swap"),
        ).validate(cell_view, architecture)
    traditional = create_scenario(
        [2, 1],
        policy=Policy.BUBBLE,
        architecture=Architecture.TRADITIONAL,
        permute=False,
        generation_key="S04/contract/global",
    )
    with pytest.raises(ValueError, match="legacy global"):
        SchedulerExecutionContract(SchedulerFamily.DETERMINISTIC_SCAN).validate(
            traditional, ArchitectureExecutionContract.central_global_legacy()
        )


def test_scheduled_opportunity_rejects_unaccounted_rng_draws() -> None:
    with pytest.raises(ValueError, match="draws and stream consumption"):
        ScheduledOpportunity(
            "c00",
            (("actor_activation", 0, 0, 7),),
            (),
        )
    with pytest.raises(ValueError, match="draws and stream consumption"):
        ScheduledOpportunity(
            "c00",
            (),
            (("actor_activation", 1),),
        )
    with pytest.raises(ValueError, match="require bubble_side"):
        ScheduledOpportunity(
            "c00",
            (("bubble_side", 0, 0, 7),),
            (("bubble_side", 1),),
        )
    with pytest.raises(ValueError, match="exactly one"):
        ScheduledOpportunity("c00", bubble_side="left")


def test_scheduled_opportunity_accepts_typed_external_bubble_side() -> None:
    opportunity = ScheduledOpportunity(
        "c00",
        (("actor_activation", 0, 0, 7), ("bubble_side", 0, 0, 9)),
        (("actor_activation", 1), ("bubble_side", 1)),
        bubble_side="left",
    )
    assert opportunity.bubble_side == "left"


def test_scan_permutation_synchronous_and_adversarial_fairness_contracts() -> None:
    n = 8
    opportunities = 8_000
    scan = _sequence(SchedulerFamily.DETERMINISTIC_SCAN, opportunities, n)
    for start in range(0, opportunities, n):
        assert set(scan[start : start + n]) == set(scan[:n])
    assert max(max(values) for values in _gaps(scan).values()) == n

    sweep = _sequence(SchedulerFamily.RANDOM_PERMUTATION_SWEEP, opportunities, n)
    actors = set(sweep[:n])
    for start in range(0, opportunities, n):
        assert set(sweep[start : start + n]) == actors
    assert max(max(values) for values in _gaps(sweep).values()) <= 2 * n - 1

    synchronous = _sequence(
        SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT,
        opportunities + 3,
        n,
    )
    for start in range(0, opportunities, n):
        assert set(synchronous[start : start + n]) == set(synchronous[:n])
    assert len(synchronous[-3:]) == 3

    adversarial_controller = _controller(SchedulerFamily.FAIR_ADVERSARIAL, n=n)
    adversarial = _sequence(SchedulerFamily.FAIR_ADVERSARIAL, opportunities, n)
    assert len(set(adversarial[:n])) == n
    assert max(max(values) for values in _gaps(adversarial).values()) <= 2 * n
    for start in range(n, opportunities - 2 * n + 1):
        assert set(adversarial[start : start + 2 * n]) == set(adversarial[:n])
    while len(adversarial_controller.audits) < 100:
        adversarial_controller(len(adversarial_controller.audits), 1)
    assert all(item.proposal_candidates_inspected == 0 for item in adversarial_controller.audits)
    assert max(item.identities_scored for item in adversarial_controller.audits) == n


def test_uniform_scheduler_is_exact_native_s03_baseline() -> None:
    from causal_simulator.architectures import run_architecture

    for policy in Policy:
        for direction in Direction:
            scenario = create_scenario(
                [6, 1, 5, 2, 4, 3],
                policy=policy,
                direction=direction,
                seed=20260714,
                max_activations=20_000,
                generation_key=f"S04/native-parity/{policy.value}/{direction.value}",
                permute=False,
            )
            architecture = ArchitectureExecutionContract.distributed_local()
            baseline = run_architecture(scenario, architecture, trace_mode="full")
            scheduled = run_scheduled_architecture(
                scenario,
                architecture,
                SchedulerExecutionContract(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION),
                trace_mode="full",
            )
            assert scheduled.result.to_json_bytes() == baseline.result.to_json_bytes()
            assert all(scheduled.opportunity_validation().values())


@pytest.mark.parametrize("family", tuple(SchedulerFamily))
def test_all_families_replay_ledger_and_matched_topology_parity(
    family: SchedulerFamily,
) -> None:
    scenario = create_scenario(
        [5, 1, 4, 2, 3],
        policy=Policy.INSERTION,
        seed=73,
        max_activations=20_000,
        generation_key=f"S04/all-families/{family.value}",
        permute=False,
    )
    scheduler = SchedulerExecutionContract(family)
    contracts = (
        ArchitectureExecutionContract.distributed_local(),
        ArchitectureExecutionContract.central_local_k1(),
        ArchitectureExecutionContract.distributed_weak(enabled=False),
        ArchitectureExecutionContract.distributed_weak(),
    )
    runs = [
        run_scheduled_architecture(scenario, contract, scheduler, trace_mode="full")
        for contract in contracts
    ]
    for item in runs:
        assert item.result.summary["completed"]
        assert all(item.opportunity_validation().values()), item.opportunity_validation()
        assert exact_replay_scheduler(item).to_json_bytes() == item.to_json_bytes()
        assert {
            event["proposal"]["kind"] for event in item.result.events
        } <= {"NoOp", "Swap", "MemoryUpdate"}
    assert runs[0].result.to_json_bytes() == runs[1].result.to_json_bytes()
    assert runs[0].result.to_json_bytes() == runs[2].result.to_json_bytes()


def test_permutation_rng_consumption_is_explicit_and_counter_addressed() -> None:
    scenario = create_scenario(
        [7, 6, 5, 4, 3, 2, 1, 0],
        policy=Policy.BUBBLE,
        seed=91,
        max_activations=17,
        generation_key="S04/permutation-rng",
        permute=False,
    )
    item = run_scheduled_architecture(
        scenario,
        ArchitectureExecutionContract.distributed_local(),
        SchedulerExecutionContract(SchedulerFamily.RANDOM_PERMUTATION_SWEEP),
        trace_mode="full",
    )
    blocks = sum(audit.actor_selection_stream_blocks for audit in item.scheduler_audit)
    assert blocks > 0
    assert item.result.final_state["streamCounters"][PERMUTATION_STREAM] == blocks
    traced_blocks = sum(
        draw["stream"] == PERMUTATION_STREAM
        for event in item.result.events
        for draw in event["randomAddressesAndDraws"]
    )
    assert traced_blocks == blocks
    assert all(
        audit.actor_selection_stream_blocks == 0
        for audit in item.scheduler_audit
        if audit.event_index % len(scenario.cells) != 0
    )


def test_synchronous_conflict_invariants_atomicity_and_charging() -> None:
    scenario = create_scenario(
        [5, 4, 3, 2, 1],
        policy=Policy.BUBBLE,
        seed=0,
        max_activations=5,
        generation_key="S04/synchronous-conflict",
        permute=False,
    )
    item = run_scheduled_architecture(
        scenario,
        ArchitectureExecutionContract.distributed_local(),
        SchedulerExecutionContract(
            SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT
        ),
        trace_mode="full",
    )
    events = item.result.events
    assert len(events) == len(scenario.cells)
    assert item.result.summary["ledger"]["activations"] == len(scenario.cells)
    assert item.result.summary["ledger"]["proposals"] == len(scenario.cells)
    assert item.result.summary["ledger"]["conflictLosses"] == 1
    assert len({event["preStateHash"] for event in events}) == 1
    assert len({event["postStateHash"] for event in events}) == 1
    assert all(event["batchWidth"] == len(scenario.cells) for event in events)
    assert all(
        any(draw["stream"] == "conflict_priority" for draw in event["randomAddressesAndDraws"])
        for event in events
    )

    changing = [
        event
        for event in events
        if event["decision"] in {"accepted", "conflict_loss"}
    ]
    ordered = sorted(
        changing,
        key=lambda event: (
            event["proposal"]["priority"],
            event["actorId"],
            event["proposal"]["targetPos"]
            if event["proposal"]["targetPos"] is not None
            else -1,
            event["batchOrdinal"],
        ),
    )
    reserved: set[tuple[str, str | int]] = set()
    expected_accepted: set[int] = set()
    for event in ordered:
        proposal = event["proposal"]
        resources: set[tuple[str, str | int]] = {("actor", event["actorId"])}
        if proposal["kind"] == "Swap":
            resources.update(
                {
                    ("position", proposal["actorPos"]),
                    ("position", proposal["targetPos"]),
                }
            )
        if not resources & reserved:
            expected_accepted.add(event["batchOrdinal"])
            reserved.update(resources)
    observed_accepted = {
        event["batchOrdinal"] for event in events if event["decision"] == "accepted"
    }
    assert observed_accepted == expected_accepted


def test_synchronous_weak_coordinator_uses_each_charged_event_index() -> None:
    scenario = create_scenario(
        list(range(16, 0, -1)),
        policy=Policy.SELECTION,
        seed=4,
        max_activations=32,
        generation_key="S04/synchronous-weak-clock",
        permute=False,
    )
    item = run_scheduled_architecture(
        scenario,
        ArchitectureExecutionContract.distributed_weak(),
        SchedulerExecutionContract(
            SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT
        ),
        trace_mode="full",
    )
    assert [audit.event_index for audit in item.architecture_run.coordination_audit] == [
        7,
        15,
        23,
        31,
    ]
    assert all(item.opportunity_validation().values())


def test_uniform_small_distribution_has_all_identities_and_reasonable_balance() -> None:
    sequence = _sequence(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION, 80_000, n=8)
    counts = Counter(sequence)
    assert len(counts) == 8
    assert max(abs(count - 10_000) for count in counts.values()) < 500
