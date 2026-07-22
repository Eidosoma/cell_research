from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from scripts import validate_surrogate_models_s06 as validator
from src.surrogate_models.core import (
    S05_ROOT,
    SurrogateMLP,
    TargetTransform,
    assign_grouped_splits,
    build_s05_freeze,
    build_target_registry,
    event_length,
    lineage_component,
    load_lineage,
    load_policy_catalog_with_plans,
    policy_feature_dict,
    read_jsonl,
    shared_event_features,
    target_arrays,
)


PROTOCOL = Path("configs/modeling/s06_modeling_protocol.yaml")


def settings():
    return yaml.safe_load(PROTOCOL.read_text())


def test_s05_hash_freeze_and_authoritative_plan_contract_pass():
    result = build_s05_freeze(settings())
    assert result["success"] is True
    assert result["selectedCanonicalRowCount"] == 2248
    assert result["recoveryOrphanRowCount"] == 189
    assert result["authoritativeGenerationPlanCount"] == 40
    assert result["persistedGenerationPlanIsAuthoritative"] is True
    assert result["validationOutcomeRows"] == 0
    assert result["confirmationOutcomeRows"] == 0
    assert all(item["pass"] for item in result["artifactHashChecks"])


def test_grouped_splits_are_lineage_and_scenario_disjoint():
    rows = read_jsonl(S05_ROOT / "evaluation_ledger.jsonl")
    selected = sorted(
        [row for row in rows if row["selectedForCanonicalSearch"]],
        key=lambda row: row["stableEvaluationSha256"],
    )
    assignments = assign_grouped_splits(selected, load_lineage(), settings())
    assert len(assignments) == 2248
    for task_id in sorted({item["taskId"] for item in assignments}):
        task = [item for item in assignments if item["taskId"] == task_id]
        component_partitions = {}
        scenario_partitions = {}
        for item in task:
            component_partitions.setdefault(item["lineageComponent"], set()).add(
                item["lineagePartition"]
            )
            scenario_partitions.setdefault(item["scenarioOrdinal"], set()).add(
                item["scenarioPartition"]
            )
        assert all(len(value) == 1 for value in component_partitions.values())
        assert all(len(value) == 1 for value in scenario_partitions.values())
        assert {item["lineagePartition"] for item in task} == {
            "fit",
            "calibration",
            "test",
        }


def test_lineage_component_retains_missing_native_parent_as_root_token():
    lineage = {
        "child": {"parents": ["native-parent"]},
        "grandchild": {"parents": ["child"]},
    }
    assert lineage_component("child", lineage) == lineage_component("grandchild", lineage)
    assert lineage_component("other", lineage) != lineage_component("child", lineage)


def test_policy_features_exclude_identifiers_outcomes_and_event_summaries():
    rows = read_jsonl(S05_ROOT / "evaluation_ledger.jsonl")
    row = next(item for item in rows if item["selectedForCanonicalSearch"])
    features = policy_feature_dict(row, load_policy_catalog_with_plans(), scenario=True)
    forbidden = ("hash", "policyid", "scenarioid", "outcome", "event", "cost", "elapsed")
    assert not any(any(token in key.lower() for token in forbidden) for key in features)
    assert "scenario::ordinal" in features


def test_target_registry_keeps_native_costs_and_licensed_capabilities_separate():
    rows = read_jsonl(S05_ROOT / "evaluation_ledger.jsonl")
    task_rows = [
        row
        for row in rows
        if row["selectedForCanonicalSearch"]
        and row["taskId"] == "e07_s02_sorting_1d"
    ]
    registry = build_target_registry("e07_s02_sorting_1d", task_rows)
    continuous, binary = target_arrays(task_rows, registry)
    assert continuous.shape[0] == len(task_rows)
    assert binary.shape[0] == len(task_rows)
    assert registry["universalScore"] is False
    assert any("licensedPrefix" in name for name in registry["licensedCapabilityCostTargets"])
    assert all(item["name"] != "cost::total" for item in registry["continuous"])


def test_target_transform_round_trip_with_missing_values():
    transform = TargetTransform(
        centers=np.asarray([1.0, 2.0, 0.0]),
        scales=np.asarray([2.0, 3.0, 1.0]),
        transforms=("identity", "log1p", "signed_log1p"),
    )
    values = np.asarray([[3.0, 7.0, -4.0], [np.nan, 0.0, 5.0]])
    restored = transform.inverse(transform.transform(values))
    assert np.allclose(restored, values, equal_nan=True)


def test_models_have_separate_continuous_and_binary_heads():
    model = SurrogateMLP(12, 5, 3)
    continuous, binary = model(torch.zeros((4, 12)))
    assert continuous.shape == (4, 5)
    assert binary.shape == (4, 3)


def test_event_summary_features_and_length_do_not_require_outcome_dropping():
    rows = read_jsonl(S05_ROOT / "evaluation_ledger.jsonl")
    failed = next(row for row in rows if row["failed"])
    features = shared_event_features(failed)
    assert features["status_failed"] == 1.0
    assert "status_censored" in features
    assert event_length(failed) >= 0


def test_recovery_orphans_are_disjoint_and_sensitivity_only():
    rows = read_jsonl(S05_ROOT / "evaluation_ledger.jsonl")
    selected = {
        row["stableEvaluationSha256"]
        for row in rows
        if row["selectedForCanonicalSearch"]
    }
    orphans = {
        row["stableEvaluationSha256"]
        for row in rows
        if not row["selectedForCanonicalSearch"]
    }
    assert len(orphans) == 189
    assert selected.isdisjoint(orphans)
    assert settings()["authorizedPopulation"]["recoveryOrphanUse"] == "frozen_model_sensitivity_only"


def test_protocol_predefines_fail_closed_thresholds():
    frozen = settings()["frozenThresholds"]
    assert frozen["deploymentRule"].startswith("all_")
    assert frozen["failureAction"] == "reject_models_for_S07_allocation_or_archive_mutation"
    assert settings()["surrogates"]["universalScore"] == "forbidden"


def test_postfit_gate_recalculation_is_fail_closed_and_complete():
    result = validator._gate_recalculation(settings())
    assert len(result["taskRows"]) == 8
    assert result["surrogateUtility"]["tasksPassingBoth"] == 3
    assert result["surrogateUtility"]["pass"] is False
    assert result["continuousCalibration"]["tasksPassingBothCoverageAndWidth"] == 2
    assert result["continuousCalibration"]["pass"] is False
    assert result["binaryCalibration"]["tasksPassingBothEceAndBrier"] == 1
    assert result["binaryCalibration"]["pass"] is False
    assert result["failureAndCensoring"]["pass"] is False


def test_fresh_model_double_load_is_bit_exact_and_hash_valid():
    result = validator._fresh_reload_validation()
    assert result["allFileHashesPass"] is True
    assert result["allStateHashesPass"] is True
    assert result["freshDoubleLoadCpuBitExactInferencePass"] is True
