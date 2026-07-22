from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from reference_simulator.engine import run
from reference_simulator.model import Cell, Direction, Policy, Scenario
from src.environment_suite import (
    AccessGrant,
    AccessDeniedError,
    AccessPhase,
    EnvironmentSuite,
    EvaluationAction,
    SuiteValidationError,
)
from src.environment_suite.dsl_adapters import (
    dsl_action,
    line_replay_bytes,
    run_line_dsl_episode,
)


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "src/policy_dsl/baselines"
REGISTRY = ROOT / "configs/environment_suite/task_registry.yaml"
SPLITS = ROOT / "configs/environment_suite/split_manifest.json"


def _document(name: str) -> dict:
    return json.loads((POLICIES / f"{name}.json").read_text(encoding="utf-8"))


def _suite() -> EnvironmentSuite:
    return EnvironmentSuite(REGISTRY, SPLITS)


def _actions() -> dict[str, EvaluationAction]:
    bubble = _document("bubble_cell_view_v1")
    insertion = _document("insertion_cell_view_v1")
    greedy = _document("spatial_greedy_local_v1")
    memory = _document("spatial_memory_repair_v1")
    return {
        "e07_s02_sorting_1d": dsl_action([bubble]),
        "e07_s02_faults_1d": dsl_action([bubble]),
        "e07_s02_detour_1d": dsl_action([bubble]),
        "e07_s02_chimera_1d": dsl_action(
            [bubble, insertion],
            native_policy_bindings={
                "Bubble": "bubble_cell_view_v1",
                "Insertion": "insertion_cell_view_v1",
            },
        ),
        "e07_s02_regeneration_1d": dsl_action([bubble]),
        "e07_s02_target_change_1d": dsl_action([bubble]),
        "e07_s02_spatial2d_local": dsl_action([greedy]),
        "e07_s02_spatial2d_memory": dsl_action([memory]),
    }


TRAIN_SCENARIOS = {
    "e07_s02_sorting_1d": "e07s02:sorting:train:000",
    "e07_s02_faults_1d": "e07s02:faults:train:000",
    "e07_s02_detour_1d": "e07s02:detour:train:000",
    "e07_s02_chimera_1d": "e07s02:chimera:train:000",
    "e07_s02_regeneration_1d": "e07s02:regeneration:train:000",
    "e07_s02_target_change_1d": "e07s02:target-change:train:000",
    "e07_s02_spatial2d_local": "e07s02:spatial-local:train:000",
    "e07_s02_spatial2d_memory": "e07s02:spatial-memory:train:000",
}


@pytest.mark.parametrize(
    ("name", "carrier"),
    [
        ("bubble_cell_view_v1", Policy.BUBBLE),
        ("insertion_cell_view_v1", Policy.INSERTION),
        ("selection_cell_view_v1", Policy.SELECTION),
    ],
)
def test_faithful_line_policies_match_native_full_episode(name, carrier):
    values = [7, 1, 6, 2, 5, 3, 4, 0]
    scenario = Scenario.create(
        [
            Cell(f"c{index}", value, carrier, Direction.ASCENDING)
            for index, value in enumerate(values)
        ],
        seed=1729,
        max_activations=10_000,
        generation_key=f"S04A/parity/{name}",
    )
    native = run(scenario, trace_mode="full")
    adapted, runtime = run_line_dsl_episode(
        scenario, dsl_action([_document(name)]), trace_mode="full"
    )
    assert adapted.summary["finalValues"] == native.summary["finalValues"]
    assert adapted.summary["activationCount"] == native.summary["activationCount"]
    assert adapted.summary["stopReason"] == native.summary["stopReason"]
    assert adapted.summary["ledger"] == native.summary["ledger"]
    assert runtime.ledger["dslActivations"] == native.summary["activationCount"]
    if carrier == Policy.INSERTION:
        assert runtime.ledger["licensedPrefixPredicateEvaluations"] > 0
        assert runtime.ledger["licensedPrefixValueReads"] > 0
    if carrier == Policy.SELECTION:
        assert runtime.ledger["engineCursorStateReads"] > 0
        assert runtime.ledger["licensedLongRangeRequestedDistance"] > 0


def test_canonical_hash_tampering_is_rejected():
    action = dsl_action([_document("bubble_cell_view_v1")])
    with pytest.raises(SuiteValidationError, match="hash mismatch"):
        EvaluationAction(
            action.policy_id,
            "0" * 64,
            mode="dsl_episode",
            policy_documents=action.policy_documents,
        )


@pytest.mark.parametrize("task_id", sorted(TRAIN_SCENARIOS))
def test_dsl_episode_access_is_train_only(task_id):
    suite = _suite()
    validation_id = TRAIN_SCENARIOS[task_id].replace(":train:", ":validation:")
    with pytest.raises(AccessDeniedError, match="training split"):
        suite.open_for_action(
            task_id,
            validation_id,
            AccessGrant(AccessPhase.VALIDATION),
            _actions()[task_id],
        )
    assert suite.broker.audit.outcome_requests == 0
    assert suite.broker.audit.materializer_invocations == 0


@pytest.mark.parametrize("task_id", sorted(TRAIN_SCENARIOS))
def test_every_core_task_runs_a_policy_controlled_train_episode(task_id):
    suite = _suite()
    environment = suite.open(
        task_id,
        TRAIN_SCENARIOS[task_id],
        AccessGrant(AccessPhase.DEVELOPMENT),
    )
    step = environment.step(_actions()[task_id])
    outcome = environment.outcome().to_dict()
    assert step.terminal
    assert not step.failed
    assert step.event["replayPass"]
    assert all(step.event["validation"].values())
    assert step.cost["scalarCrossFamilyTotal"] is None
    assert step.cost["normalizationApplied"] is False
    assert outcome["claimBoundary"]
    if task_id == "e07_s02_sorting_1d":
        runtime = step.cost["nativeLedgerFamilies"]["dslRuntimeLedger"]
        descriptor = outcome["outcome"]["descriptors"]
        expected = runtime["committedNativeDisplacement"] / (
            runtime["committedNativeMovementActions"] * 7
        )
        assert descriptor["committedDisplacementFraction"] == expected
    if task_id == "e07_s02_spatial2d_local":
        movement = step.cost["nativeLedgerFamilies"]["e06MovementLedger"]
        descriptor = outcome["outcome"]["descriptors"]
        expected = movement["totalGraphDisplacement"] / (
            movement["acceptedMovements"] * 6
        )
        assert descriptor["committedDisplacementFraction"] == expected


def test_unauthorized_relative_direction_is_rejected_by_native_legality():
    document = {
        "schemaVersion": "e07.policy-dsl.v1",
        "policyId": "bubble_wrong_side_probe_v1",
        "environment": "line1d.v1",
        "permissions": ["activation.side"],
        "memory": [],
        "signals": {"channels": 0, "bitsPerChannel": 0},
        "limits": {
            "maxRules": 1,
            "maxExpressionNodes": 1,
            "maxActionsPerActivation": 1,
            "maxOperationsPerActivation": 8,
            "maxMovementRadius": 1,
            "maxCandidates": 0,
        },
        "rules": [
            {
                "when": {"const": True},
                "actions": [{"kind": "swap_relative", "offset": -1}],
            }
        ],
        "default": {"actions": [{"kind": "noop"}]},
    }
    scenario = Scenario.create(
        [
            Cell(f"c{index}", value, Policy.BUBBLE, Direction.ASCENDING)
            for index, value in enumerate([3, 2, 1, 0])
        ],
        seed=23,
        max_activations=64,
        generation_key="S04A/unauthorized-relative-direction",
    )
    first, runtime = run_line_dsl_episode(scenario, dsl_action([document]))
    second, replay_runtime = run_line_dsl_episode(scenario, dsl_action([document]))
    assert runtime.ledger["rejectedNativeActions"] > 0
    assert first.summary["ledger"]["rejections"] > 0
    assert line_replay_bytes(first, runtime) == line_replay_bytes(
        second, replay_runtime
    )


def test_peer_signals_are_delayed_costed_and_replayable_in_episode():
    document = {
        "schemaVersion": "e07.policy-dsl.v1",
        "policyId": "line_signal_ping_v1",
        "environment": "line1d.v1",
        "permissions": ["signal.neighbor_sum_u8"],
        "memory": [],
        "signals": {"channels": 1, "bitsPerChannel": 2},
        "limits": {
            "maxRules": 1,
            "maxExpressionNodes": 1,
            "maxActionsPerActivation": 2,
            "maxOperationsPerActivation": 8,
            "maxMovementRadius": 0,
            "maxCandidates": 0,
        },
        "rules": [
            {
                "when": {"const": True},
                "actions": [
                    {"kind": "emit_signal", "channel": 0, "value": {"const": 1}},
                    {"kind": "noop"},
                ],
            }
        ],
        "default": {"actions": [{"kind": "noop"}]},
    }
    scenario = Scenario.create(
        [
            Cell(f"c{index}", value, Policy.INSERTION, Direction.ASCENDING)
            for index, value in enumerate([3, 2, 1, 0])
        ],
        seed=29,
        max_activations=24,
        generation_key="S04A/communication",
    )
    first, runtime = run_line_dsl_episode(scenario, dsl_action([document]))
    second, replay_runtime = run_line_dsl_episode(scenario, dsl_action([document]))
    ledger = runtime.adapter_ledgers()["dslCommunicationLedger"]
    assert first.summary["stopReason"] == "event_budget"
    assert ledger["emittedSignalWrites"] == 24
    assert ledger["recipientDeliveries"] > 0
    assert ledger["consumedDeliveries"] > 0
    assert ledger["observableAggregateReads"] == 24
    assert line_replay_bytes(first, runtime) == line_replay_bytes(
        second, replay_runtime
    )


def test_line_authority_is_not_inferred_from_matching_field_names():
    document = deepcopy(_document("bubble_cell_view_v1"))
    document["policyId"] = "bubble_with_forbidden_cursor_v1"
    document["permissions"].append("selection.cursor_in_bounds")
    action = dsl_action([document])
    scenario = Scenario.create(
        [
            Cell(f"c{index}", value, Policy.BUBBLE, Direction.ASCENDING)
            for index, value in enumerate([2, 1, 0])
        ],
        seed=31,
        max_activations=32,
        generation_key="S04A/no-field-name-authority",
    )
    with pytest.raises(SuiteValidationError, match="outside .* authority"):
        run_line_dsl_episode(scenario, action)


def test_e06_channel_authority_is_not_inferred_from_dsl_permission():
    document = deepcopy(_document("spatial_greedy_local_v1"))
    document["policyId"] = "spatial_unlicensed_gradient_v1"
    document["permissions"].append("gradient.current_u8")
    action = dsl_action([document])
    suite = _suite()
    environment = suite.open_for_action(
        "e07_s02_spatial2d_local",
        TRAIN_SCENARIOS["e07_s02_spatial2d_local"],
        AccessGrant(AccessPhase.DEVELOPMENT),
        action,
    )
    with pytest.raises(SuiteValidationError, match="required native authority"):
        environment.step(action)
