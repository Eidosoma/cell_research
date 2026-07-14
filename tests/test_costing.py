from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.costing import (
    ALIAS_FIELDS,
    COST_SCHEMA_SHA256,
    S01_ADDITIVE_FIELDS,
    SCHEDULER_OVERHEAD_FIELDS,
    build_complete_cost_ledger,
    exact_replay_costed,
    run_costed_faulted_architecture,
    validate_cost_schema,
)
from causal_simulator.faults import (
    ActionFailureProfile,
    ContinuationPolicy,
    FaultExecutionContract,
    RetryPolicy,
    SensingProfile,
    run_faulted_architecture,
)
from causal_simulator.schedulers import SchedulerExecutionContract, SchedulerFamily
from reference_simulator.api import create_scenario
from reference_simulator.model import Policy


ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "s09_cost_ledger_fixtures.json"
SCHEMA = ROOT / "design" / "s09" / "cost_schema.json"


def _architecture(name: str) -> ArchitectureExecutionContract:
    return {
        "central_local_k1": ArchitectureExecutionContract.central_local_k1,
        "distributed_local": ArchitectureExecutionContract.distributed_local,
        "distributed_weak_disabled": lambda: ArchitectureExecutionContract.distributed_weak(
            enabled=False
        ),
        "distributed_weak_active": ArchitectureExecutionContract.distributed_weak,
    }[name]()


def _fault(spec: dict[str, str]) -> FaultExecutionContract:
    return FaultExecutionContract(
        continuation=ContinuationPolicy(
            spec.get("continuation", ContinuationPolicy.SKIP_AND_CONTINUE.value)
        ),
        retry=RetryPolicy(spec.get("retry", RetryPolicy.NO_RETRY.value)),
        action_failure=ActionFailureProfile(
            spec.get("actionFailure", ActionFailureProfile.NONE.value)
        ),
        sensing=SensingProfile(spec.get("sensing", SensingProfile.EXACT.value)),
    )


def _fixture_run(item: dict):
    scenario = create_scenario(
        item["values"],
        policy=Policy(item["policy"]),
        seed=item["seed"],
        max_activations=item["maxActivations"],
        generation_key=item["generationKey"],
        permute=False,
    )
    return run_faulted_architecture(
        scenario,
        _architecture(item["architecture"]),
        SchedulerExecutionContract(SchedulerFamily(item["scheduler"])),
        _fault(item["fault"]),
        trace_mode="full",
    )


def test_frozen_schema_hash_and_contract_boundary() -> None:
    check = validate_cost_schema(SCHEMA)
    assert check["success"]
    assert check["observedSha256"] == COST_SCHEMA_SHA256
    schema = json.loads(SCHEMA.read_text())
    assert schema["contractBoundary"]["changesActionOrInformationContract"] is False
    assert schema["contractBoundary"]["changesSchedulerOrFaultOpportunityContract"] is False
    assert schema["contractBoundary"]["changesS08ArmsOrPairing"] is False


@pytest.mark.parametrize(
    "item",
    json.loads(FIXTURES.read_text())["fixtures"],
    ids=lambda item: item["name"],
)
def test_hand_counted_complete_ledgers(item: dict) -> None:
    ledger = build_complete_cost_ledger(_fixture_run(item))
    assert ledger.success
    combined = {**ledger.costs, **ledger.projections}
    if item.get("expectedAllAlgorithmicZero"):
        assert all(combined[name] == 0 for name in S01_ADDITIVE_FIELDS)
        assert all(value == 0 for value in ledger.projections.values())
        assert ledger.normalizations["s01UnitWeightPerOpportunity"] is None
    else:
        for field, expected in item["expectedCosts"].items():
            assert combined[field] == expected, field


def test_double_counting_groups_and_named_projections() -> None:
    item = json.loads(FIXTURES.read_text())["fixtures"][5]
    ledger = build_complete_cost_ledger(_fixture_run(item))
    costs = ledger.costs
    assert not set(ALIAS_FIELDS).intersection(S01_ADDITIVE_FIELDS)
    assert not set(SCHEDULER_OVERHEAD_FIELDS).intersection(S01_ADDITIVE_FIELDS)
    assert "wallTimeSeconds" not in S01_ADDITIVE_FIELDS
    assert ledger.projections["s01UnitWeightFullCost"] == sum(
        costs[name] for name in S01_ADDITIVE_FIELDS
    )
    # Noisy sensing draws audit already-charged reads and do not inflate S01.
    assert costs["sensingValueDraws"] == costs["observationRecordReads"]
    assert ledger.projections["mechanismExpandedSensitivity"] == (
        ledger.projections["controllerExpandedSensitivity"]
        + costs["sensingErrorHandlingOperations"]
        + costs["faultRngDraws"]
    )


@pytest.mark.parametrize("family", tuple(SchedulerFamily))
def test_matched_architecture_cost_comparability(family: SchedulerFamily) -> None:
    scenario = create_scenario(
        [6, 1, 5, 2, 4, 3],
        policy=Policy.INSERTION,
        seed=20260714,
        max_activations=80,
        generation_key=f"S09/architecture-comparability/{family.value}",
        permute=False,
    )
    ledgers = []
    for architecture in (
        ArchitectureExecutionContract.central_local_k1(),
        ArchitectureExecutionContract.distributed_local(),
        ArchitectureExecutionContract.distributed_weak(enabled=False),
    ):
        run = run_faulted_architecture(
            scenario,
            architecture,
            SchedulerExecutionContract(family),
            FaultExecutionContract(),
            trace_mode="digest",
        )
        ledgers.append(build_complete_cost_ledger(run).costs)
    ignored = {"wallTimeSeconds"}
    projections = [
        {key: value for key, value in ledger.items() if key not in ignored}
        for ledger in ledgers
    ]
    assert projections[0] == projections[1] == projections[2]


def test_costed_replay_excludes_wall_time_but_not_deterministic_costs() -> None:
    scenario = create_scenario(
        [4, 1, 3, 2],
        policy=Policy.BUBBLE,
        seed=7,
        max_activations=20,
        generation_key="S09/costed-replay",
        permute=False,
    )
    costed = run_costed_faulted_architecture(
        scenario,
        ArchitectureExecutionContract.distributed_weak(),
        SchedulerExecutionContract(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION),
        FaultExecutionContract(
            retry=RetryPolicy.RETRY_LATER_BOUNDED,
            action_failure=ActionFailureProfile.BERNOULLI_P,
            sensing=SensingProfile.NOISY_VALUE_OR_STATUS,
        ),
        trace_mode="full",
    )
    assert costed.ledger.costs["wallTimeSeconds"] > 0
    replayed = exact_replay_costed(costed)
    assert replayed.ledger.costs["wallTimeSeconds"] == 0
    assert (
        replayed.ledger.deterministic_bytes()
        == costed.ledger.deterministic_bytes()
    )


def test_invalid_wall_units_fail_closed() -> None:
    item = json.loads(FIXTURES.read_text())["fixtures"][0]
    run = _fixture_run(item)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_complete_cost_ledger(run, wall_time_seconds=-1)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_complete_cost_ledger(run, wall_time_seconds=float("nan"))
