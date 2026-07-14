from __future__ import annotations

from itertools import product
import json
from pathlib import Path

import pytest

from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ControlArchitecture,
)
from causal_simulator.faults import (
    BERNOULLI_FAILURE_STREAM,
    ActionFailureProfile,
    ContinuationPolicy,
    FaultExecutionContract,
    FrozenSensingTransformer,
    MobilityProfile,
    RetryPolicy,
    SensingProfile,
    comparability_classification,
    exact_replay_fault,
    run_faulted_architecture,
)
from causal_simulator.schedulers import (
    SchedulerExecutionContract,
    SchedulerFamily,
    run_scheduled_architecture,
)
from reference_simulator.api import create_scenario
from reference_simulator.model import Architecture, Policy


FIXTURES = Path(__file__).parent / "fixtures" / "s05_fault_fixtures.json"


def _architectures() -> tuple[ArchitectureExecutionContract, ...]:
    return (
        ArchitectureExecutionContract.central_local_k1(),
        ArchitectureExecutionContract.distributed_local(),
        ArchitectureExecutionContract.distributed_weak(enabled=False),
        ArchitectureExecutionContract.distributed_weak(),
    )


def _run(scenario, fault, family=SchedulerFamily.DETERMINISTIC_SCAN, architecture=None):
    return run_faulted_architecture(
        scenario,
        architecture or ArchitectureExecutionContract.distributed_local(),
        SchedulerExecutionContract(family),
        fault,
        trace_mode="full",
    )


def test_contract_rejects_boundary_weakening_and_classifies_noncomparables() -> None:
    scenario = create_scenario(
        [2, 1], policy=Policy.BUBBLE, permute=False, generation_key="S05/contract"
    )
    scheduler = SchedulerExecutionContract(SchedulerFamily.DETERMINISTIC_SCAN)
    architecture = ArchitectureExecutionContract.distributed_local()
    FaultExecutionContract().validate(scenario, architecture, scheduler)
    with pytest.raises(ValueError, match="free proposal"):
        FaultExecutionContract(proposal_candidates_per_opportunity=2).validate(
            scenario, architecture, scheduler
        )
    with pytest.raises(ValueError, match="information"):
        FaultExecutionContract(information_permission="full_global_state").validate(
            scenario, architecture, scheduler
        )
    with pytest.raises(ValueError, match="legal primitive"):
        FaultExecutionContract(legal_primitives=("NoOp", "Swap")).validate(
            scenario, architecture, scheduler
        )
    traditional = create_scenario(
        [2, 1],
        policy=Policy.BUBBLE,
        architecture=Architecture.TRADITIONAL,
        permute=False,
        generation_key="S05/contract/legacy",
    )
    with pytest.raises(ValueError):
        FaultExecutionContract().validate(
            traditional,
            ArchitectureExecutionContract.central_global_legacy(),
            scheduler,
        )
    status, _ = comparability_classification(
        ControlArchitecture.CENTRAL_GLOBAL_LEGACY, FaultExecutionContract()
    )
    assert status == "non_comparable"
    status, _ = comparability_classification(
        ControlArchitecture.DISTRIBUTED_LOCAL,
        FaultExecutionContract(
            continuation=ContinuationPolicy.STOP_ON_FIRST_BLOCKING_FAILURE,
            retry=RetryPolicy.RETRY_LATER_BOUNDED,
        ),
    )
    assert status == "comparable_degenerate"


def test_sensing_controller_has_no_state_or_global_scenario_surface() -> None:
    held = set(FrozenSensingTransformer.__slots__)
    assert held == {"profile", "seed", "scenario_id", "audits", "draws_by_event"}
    assert not held & {"scenario", "state", "occupancy", "cells", "ledger", "values"}


@pytest.mark.parametrize("architecture", _architectures())
@pytest.mark.parametrize("family", tuple(SchedulerFamily))
def test_exact_fault_overlay_is_byte_exact_s04(
    architecture: ArchitectureExecutionContract,
    family: SchedulerFamily,
) -> None:
    scenario = create_scenario(
        [6, 1, 5, 2, 4, 3],
        policy=Policy.INSERTION,
        seed=20260714,
        max_activations=100,
        permute=False,
        generation_key=f"S05/no-fault-parity/{architecture.architecture.value}/{family.value}",
    )
    scheduler = SchedulerExecutionContract(family)
    baseline = run_scheduled_architecture(
        scenario, architecture, scheduler, trace_mode="full"
    )
    faulted = run_faulted_architecture(
        scenario,
        architecture,
        scheduler,
        FaultExecutionContract(),
        trace_mode="full",
    )
    assert faulted.result.to_json_bytes() == baseline.result.to_json_bytes()
    assert all(faulted.opportunity_validation().values())


def test_hand_checked_passive_continuation_and_atomic_synchronous_stop() -> None:
    fixture = json.loads(FIXTURES.read_text())
    passive = fixture["passiveContinuation"]
    scenario = create_scenario(
        passive["values"],
        policy=Policy.BUBBLE,
        faults={int(key): value for key, value in passive["faults"].items()},
        seed=passive["seed"],
        max_activations=10,
        permute=False,
        generation_key=passive["generationKey"],
    )
    skip = _run(
        scenario,
        FaultExecutionContract(mobility=MobilityProfile.PASSIVE),
    )
    stop = _run(
        scenario,
        FaultExecutionContract(
            mobility=MobilityProfile.PASSIVE,
            continuation=ContinuationPolicy.STOP_ON_FIRST_BLOCKING_FAILURE,
        ),
    )
    assert (skip.result.summary["stopReason"], skip.result.summary["activationCount"]) == (
        passive["skipStopReason"],
        passive["skipActivations"],
    )
    assert (stop.result.summary["stopReason"], stop.result.summary["activationCount"]) == (
        passive["stopStopReason"],
        passive["stopActivations"],
    )
    assert stop.result.events[0]["proposal"]["reason"] == "actor_fault"

    synchronous = fixture["synchronousStop"]
    sync_scenario = create_scenario(
        synchronous["values"],
        policy=Policy.BUBBLE,
        seed=synchronous["seed"],
        max_activations=synchronous["maxActivations"],
        permute=False,
        generation_key=synchronous["generationKey"],
    )
    sync = _run(
        sync_scenario,
        FaultExecutionContract(
            continuation=ContinuationPolicy.STOP_ON_FIRST_BLOCKING_FAILURE,
            action_failure=ActionFailureProfile.BERNOULLI_P,
        ),
        SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT,
    )
    assert sync.result.summary["stopReason"] == "blocking_failure"
    assert sync.result.summary["activationCount"] == synchronous["chargedFirstBatch"]
    assert [event["decision"] for event in sync.result.events] == synchronous["decisions"]
    assert len({event["preStateHash"] for event in sync.result.events}) == 1
    assert len({event["postStateHash"] for event in sync.result.events}) == 1


def test_portable_failure_retry_and_sensing_fixtures() -> None:
    core = json.loads(FIXTURES.read_text())["core"]
    scenario = create_scenario(
        core["values"],
        policy=Policy(core["policy"]),
        seed=core["seed"],
        max_activations=core["maxActivations"],
        permute=False,
        generation_key=core["generationKey"],
    )
    retried = _run(
        scenario,
        FaultExecutionContract(
            action_failure=ActionFailureProfile.BERNOULLI_P,
            retry=RetryPolicy.RETRY_LATER_BOUNDED,
        ),
    )
    assert [
        item.event_index for item in retried.action_failure_audit if item.applied
    ] == core["bernoulliFailureEvents"]
    assert [item.to_dict() for item in retried.retry_audit] == core["retryAudit"]
    retry_event = next(
        event
        for event in retried.result.events
        if event["proposal"]["reason"].startswith("retry_later_bounded")
    )
    assert retry_event["decision"] == core["retryDecision"]
    assert retry_event["observation"] == {"reads": 0, "valueComparisons": 0}

    sensed = _run(
        scenario,
        FaultExecutionContract(sensing=SensingProfile.NOISY_VALUE_OR_STATUS),
    )
    applied = [
        {
            "eventIndex": item.event_index,
            "readOrdinal": item.read_ordinal,
            "field": item.field,
            "actual": item.actual,
            "visible": item.visible,
        }
        for item in sensed.sensing_audit
        if item.applied
    ]
    assert applied == core["sensingApplied"]
    assert sensed.fault_ledger["sensingValueDraws"] == sensed.result.summary["ledger"][
        "observationReads"
    ]
    assert sensed.fault_ledger["sensingErrorsApplied"] == sensed.fault_ledger[
        "sensingErrorHandlingOperations"
    ]


def test_identical_exogenous_failure_streams_and_weak_phase_interaction() -> None:
    scenario = create_scenario(
        list(range(16, 0, -1)),
        policy=Policy.SELECTION,
        seed=4,
        max_activations=32,
        permute=False,
        generation_key="S05/exogenous-stream-phase",
    )
    scheduler = SchedulerExecutionContract(
        SchedulerFamily.SYNCHRONOUS_BATCH_DETERMINISTIC_CONFLICT
    )
    runs = [
        run_faulted_architecture(
            scenario,
            architecture,
            scheduler,
            FaultExecutionContract(action_failure=ActionFailureProfile.BERNOULLI_P),
            trace_mode="full",
        )
        for architecture in _architectures()
    ]
    streams = [
        [
            (item.event_index, item.raw_uint64)
            for item in run.action_failure_audit
            if item.stream == BERNOULLI_FAILURE_STREAM
        ]
        for run in runs
    ]
    assert all(stream == streams[0] for stream in streams[1:])
    weak = runs[-1]
    assert [
        item.event_index
        for item in weak.scheduler_run.architecture_run.coordination_audit
    ] == [7, 15, 23, 31]
    assert all(run.result.summary["activationCount"] == 32 for run in runs)


def test_scheduler_architecture_fault_matrix_replay_and_accounting() -> None:
    normal = create_scenario(
        [5, 1, 4, 2, 3],
        policy=Policy.INSERTION,
        seed=73,
        max_activations=40,
        permute=False,
        generation_key="S05/matrix/normal",
    )
    passive = create_scenario(
        [5, 1, 4, 2, 3],
        policy=Policy.BUBBLE,
        faults={0: "passive"},
        seed=73,
        max_activations=40,
        permute=False,
        generation_key="S05/matrix/passive",
    )
    stuck = create_scenario(
        [5, 1, 4, 2, 3],
        policy=Policy.BUBBLE,
        faults={0: "stuck"},
        seed=73,
        max_activations=40,
        permute=False,
        generation_key="S05/matrix/stuck",
    )
    profiles = (
        (normal, FaultExecutionContract()),
        (
            normal,
            FaultExecutionContract(action_failure=ActionFailureProfile.BERNOULLI_P),
        ),
        (
            normal,
            FaultExecutionContract(action_failure=ActionFailureProfile.TRANSIENT_MARKOV),
        ),
        (
            normal,
            FaultExecutionContract(sensing=SensingProfile.NOISY_VALUE_OR_STATUS),
        ),
        (
            normal,
            FaultExecutionContract(
                action_failure=ActionFailureProfile.BERNOULLI_P,
                retry=RetryPolicy.RETRY_LATER_BOUNDED,
            ),
        ),
        (passive, FaultExecutionContract(mobility=MobilityProfile.PASSIVE)),
        (
            stuck,
            FaultExecutionContract(
                mobility=MobilityProfile.STUCK,
                continuation=ContinuationPolicy.STOP_ON_FIRST_BLOCKING_FAILURE,
            ),
        ),
    )
    checked = 0
    for architecture, family, (scenario, fault) in product(
        _architectures(), tuple(SchedulerFamily), profiles
    ):
        item = _run(scenario, fault, family, architecture)
        assert all(item.opportunity_validation().values()), item.opportunity_validation()
        assert exact_replay_fault(item).to_json_bytes() == item.to_json_bytes()
        checked += 1
    assert checked == 140
