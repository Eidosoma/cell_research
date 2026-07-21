from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.environment_suite import (
    DELIVERY_PROFILE,
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
    EvaluationAction,
    RecipientActivationMessageBus,
    SignalEmission,
    SuiteValidationError,
    baseline_policy_hash,
    load_split_manifest,
    load_task_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/environment_suite/task_registry.yaml"
SPLITS = ROOT / "configs/environment_suite/split_manifest.json"


def _suite() -> EnvironmentSuite:
    return EnvironmentSuite(REGISTRY, SPLITS)


def test_registry_covers_required_families_and_preserves_native_contracts():
    metadata, tasks = load_task_registry(REGISTRY)
    assert metadata["suiteVersion"] == "e07.s02.environment-suite.v1"
    assert {task.family for task in tasks.values()} == {
        "sorting",
        "faults",
        "detours",
        "chimeras",
        "regeneration",
        "changing_targets",
        "spatial2d",
    }
    assert len(tasks) == 8
    for task in tasks.values():
        assert task.horizon.cross_task_normalization == "forbidden"
        assert task.horizon.comparable_only_within_group
        assert task.cost_contract["scalarTotalPermitted"] is False
        assert task.outcome_contract["policyVisibleOnline"] is False
        assert (
            "protected_confirmation_outcome"
            in task.observation_contract["forbiddenFields"]
        )
        assert task.legality_authority
        assert task.event_contract["authority"]
        assert task.stopping_contract["states"]


def test_split_manifest_freezes_three_partitions_without_outcome_inputs():
    _, tasks = load_task_registry(REGISTRY)
    metadata, records = load_split_manifest(SPLITS, tasks)
    assert metadata["assignmentUsesOutcomes"] is False
    assert metadata["confirmationOutcomeAccess"] == "sealed"
    for task_id in tasks:
        task_records = [item for item in records.values() if item.task_id == task_id]
        assert {item.split.value for item in task_records} == {
            "train",
            "validation",
            "confirmation",
        }
        confirmation = next(item for item in task_records if item.protected)
        assert confirmation.materializer_id is None
        assert confirmation.outcome_access == "sealed"
        assert set(confirmation.public_parameters) == {
            "opaqueSetCommitmentSha256",
            "scenarioCount",
        }


def test_confirmation_denied_before_materialization_and_validation_is_gated():
    suite = _suite()
    before = suite.broker.audit.to_dict()
    with pytest.raises(AccessDeniedError):
        suite.open(
            "e07_s02_sorting_1d",
            "e07s02:sorting:confirmation:opaque",
            AccessGrant(AccessPhase.CONFIRMATION, "0" * 64),
        )
    after = suite.broker.audit.to_dict()
    assert after["scenarioDenials"] == before["scenarioDenials"] + 1
    assert after["materializerInvocations"] == before["materializerInvocations"]
    with pytest.raises(AccessDeniedError):
        suite.open(
            "e07_s02_sorting_1d",
            "e07s02:sorting:validation:000",
            AccessGrant(AccessPhase.DEVELOPMENT),
        )
    env = suite.open(
        "e07_s02_sorting_1d",
        "e07s02:sorting:validation:000",
        AccessGrant(AccessPhase.VALIDATION),
    )
    assert env.reset().split.value == "validation"


def test_task_and_policy_hash_tampering_are_rejected_before_execution():
    suite = _suite()
    with pytest.raises(SuiteValidationError):
        suite.open(
            "e07_s02_faults_1d",
            "e07s02:sorting:train:000",
            AccessGrant(AccessPhase.DEVELOPMENT),
        )
    env = suite.open(
        "e07_s02_sorting_1d",
        "e07s02:sorting:train:000",
        AccessGrant(AccessPhase.DEVELOPMENT),
    )
    with pytest.raises(SuiteValidationError):
        env.step(EvaluationAction("bubble_native_v1", "0" * 64))


def test_communication_delivery_timing_overwrite_saturation_and_costs():
    assert DELIVERY_PROFILE == "recipient_activation_lag_lww_sum_u8_v1"
    bus = RecipientActivationMessageBus(channels=2, bits_per_channel=3)
    topology = {"a": ("b",), "b": ("a", "c"), "c": ("b",)}
    bus.emit(SignalEmission("a", 0, {0: 5, 1: 2}), topology)
    bus.emit(SignalEmission("a", 1, {0: 6}), topology)
    bus.emit(SignalEmission("c", 1, {0: 4}), topology)
    with pytest.raises(SuiteValidationError):
        bus.observe("b", 1)
    observed = bus.observe("b", 2)
    assert observed.channel_sums == {0: 7, 1: 2}
    assert observed.consumed_sender_channel_pairs == 3
    assert bus.observe("b", 3).channel_sums == {0: 0, 1: 0}
    ledger = bus.ledger_snapshot()
    assert ledger == {
        "emittedSignalWrites": 4,
        "transmittedSignalBits": 12,
        "recipientDeliveries": 4,
        "bufferOverwrites": 1,
        "consumedDeliveries": 3,
        "observableAggregateReads": 4,
    }


def test_communication_uses_pretransition_neighbors_and_never_self_delivers():
    bus = RecipientActivationMessageBus(channels=1, bits_per_channel=8)
    pre = {"a": ("b",), "b": ("a",), "c": ()}
    bus.emit(SignalEmission("a", 4, {0: 9}), pre)
    assert bus.observe("c", 5).channel_sums == {0: 0}
    assert bus.observe("b", 5).channel_sums == {0: 9}
    with pytest.raises(SuiteValidationError):
        bus.emit(SignalEmission("a", 6, {0: 1}), {"a": ("a",)})


@pytest.mark.parametrize(
    ("task_id", "scenario_id"),
    [
        ("e07_s02_sorting_1d", "e07s02:sorting:train:000"),
        ("e07_s02_faults_1d", "e07s02:faults:train:000"),
        ("e07_s02_detour_1d", "e07s02:detour:train:000"),
        ("e07_s02_chimera_1d", "e07s02:chimera:train:000"),
        ("e07_s02_regeneration_1d", "e07s02:regeneration:train:000"),
        ("e07_s02_target_change_1d", "e07s02:target-change:train:000"),
        ("e07_s02_spatial2d_local", "e07s02:spatial-local:train:000"),
        ("e07_s02_spatial2d_memory", "e07s02:spatial-memory:train:000"),
    ],
)
def test_complete_native_baselines_execute_through_one_interface(task_id, scenario_id):
    suite = _suite()
    env = suite.open(task_id, scenario_id, AccessGrant(AccessPhase.DEVELOPMENT))
    reset = env.reset().to_dict()
    baseline_policy = suite.records[scenario_id].public_parameters["baselinePolicyId"]
    step = env.step(
        EvaluationAction(
            str(baseline_policy), baseline_policy_hash(str(baseline_policy))
        )
    ).to_dict()
    outcome = env.outcome().to_dict()
    assert reset["terminal"] is False
    assert step["terminal"] is True
    assert step["event"]["outcomeMetricsIncluded"] is False
    assert step["cost"]["scalarCrossFamilyTotal"] is None
    assert step["cost"]["normalizationApplied"] is False
    assert step["event"]["replayPass"] is True
    assert all(step["event"]["validation"].values())
    assert outcome["taskId"] == task_id
    assert outcome["claimBoundary"]
    with pytest.raises(SuiteValidationError):
        env.step(
            EvaluationAction(
                str(baseline_policy), baseline_policy_hash(str(baseline_policy))
            )
        )


def test_registry_rejects_metric_leakage_and_cross_task_normalization(tmp_path):
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    raw["tasks"][0]["outcomeContract"]["policyVisibleOnline"] = True
    poisoned = tmp_path / "poisoned.yaml"
    poisoned.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SuiteValidationError):
        load_task_registry(poisoned)
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    raw["tasks"][0]["horizon"]["crossTaskNormalization"] = "minmax"
    poisoned.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SuiteValidationError):
        load_task_registry(poisoned)


def test_split_rejects_confirmation_materializer_and_outcome_based_assignment(tmp_path):
    _, tasks = load_task_registry(REGISTRY)
    raw = json.loads(SPLITS.read_text(encoding="utf-8"))
    raw["assignmentUsesOutcomes"] = True
    poisoned = tmp_path / "poisoned.json"
    poisoned.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SuiteValidationError):
        load_split_manifest(poisoned, tasks)
    raw = json.loads(SPLITS.read_text(encoding="utf-8"))
    confirmation = next(item for item in raw["scenarios"] if item["protected"])
    confirmation["materializerId"] = "forbidden"
    poisoned.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SuiteValidationError):
        load_split_manifest(poisoned, tasks)
