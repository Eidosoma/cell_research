from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_surrogate_remediation_s06a import (
    _select_binary_blend,
    _select_continuous_blend,
)
from scripts.validate_surrogate_remediation_s06a import _native_failure_accounting
from src.environment_suite import Split, SuiteValidationError
from src.environment_suite.suite import EnvironmentSuite
from src.quality_diversity.core import BASE_SCENARIOS, TASK_IDS
from src.surrogate_remediation.core import (
    FAMILY_ORDINALS,
    SPLIT_MANIFEST,
    TASK_REGISTRY,
    build_additional_plan,
    build_s06a_target_registry,
    derive_additional_train_record,
    freeze_inputs,
    load_catalog,
    load_protocol,
    remediation_feature_dict,
)


S06_PROTOCOL = Path("/artifacts/research_steps/S06/modeling_preregistration.yaml")


def test_protocol_corrects_architecture_label_and_keeps_s06_thresholds() -> None:
    protocol = load_protocol()
    prior = yaml.safe_load(S06_PROTOCOL.read_text(encoding="utf-8"))
    observed = dict(protocol["frozenThresholds"])
    observed.pop("inheritedVerbatimFrom")
    observed.pop("inheritedSha256")
    observed["failureAction"] = "reject_models_for_S07_allocation_or_archive_mutation"
    assert observed == prior["frozenThresholds"]
    architecture = protocol["surrogates"]["architecture"]
    assert architecture["label"] == "targetwise_extra_trees_ridge_logistic_blend_v1"
    assert architecture["implementation"].startswith("independent_sklearn_target_heads")
    assert architecture["mismatchCorrected"] is True


def test_prefit_hash_freeze_verifies_s05_s06_and_sealed_access() -> None:
    result = freeze_inputs(load_protocol())
    assert result["success"] is True
    assert result["s05SelectedRows"] == 2248
    assert result["s05RecoveryOrphans"] == 189
    assert result["thresholdsUnchanged"] is True
    assert result["rejectedS06Decision"] == "reject"
    assert result["rejectedS06ModelsLoaded"] == 0
    assert result["validationOutcomeEvaluations"] == 0
    assert result["confirmationOutcomeEvaluations"] == 0


def test_additional_plan_is_exact_deterministic_and_train_only() -> None:
    protocol = load_protocol()
    first = build_additional_plan(protocol)
    second = build_additional_plan(protocol)
    assert first == second
    assert len(first) == 2560
    assert len({row["planRowSha256"] for row in first}) == 2560
    for task_id in TASK_IDS:
        task = [row for row in first if row["taskId"] == task_id]
        assert len(task) == 64 * len(FAMILY_ORDINALS)
        assert {row["scenarioOrdinal"] for row in task} == set(FAMILY_ORDINALS)
        assert len({row["policySha256"] for row in task}) == 64
        assert all(
            row["split"] == "train" and row["protected"] is False for row in task
        )
    e05 = [row for row in first if row["taskId"] == "e07_s02_regeneration_1d"]
    assert not {row["scenarioOrdinal"] for row in e05} & {4, 5, 6, 7}


def test_derivation_rejects_protected_or_unregistered_records() -> None:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    base = suite.records[BASE_SCENARIOS["e07_s02_sorting_1d"]]
    record = derive_additional_train_record(base, 100)
    assert record.split is Split.TRAIN
    assert record.protected is False
    assert record.public_parameters["seed"] != base.public_parameters["seed"]
    with pytest.raises(SuiteValidationError):
        derive_additional_train_record(base, 4)
    protected = next(row for row in suite.records.values() if row.protected)
    with pytest.raises(SuiteValidationError):
        derive_additional_train_record(protected, 100)


def test_features_exclude_identifiers_raw_seed_outcomes_costs_and_events() -> None:
    protocol = load_protocol()
    plan = build_additional_plan(protocol)
    selected = plan[0]
    row = {
        "policySha256": selected["policySha256"],
        "scenarioOrdinal": 100,
        "scenarioDerivation": {
            "sourceGeneration": "s06a",
            "withinSourceFamilyIndex": 0,
            "seed": 123456,
        },
        "outcome": {"shouldNotAppear": 1},
        "nativeLedgerFamilies": {"shouldNotAppear": 2},
        "nativeEvent": {"shouldNotAppear": 3},
        "elapsedSeconds": 4.0,
    }
    features = remediation_feature_dict(row, load_catalog(), scenario=True)
    forbidden = (
        "sha",
        "policyid",
        "scenarioid",
        "outcome",
        "nativeevent",
        "cost",
        "elapsed",
    )
    assert not any(any(token in key.lower() for token in forbidden) for key in features)
    assert "scenario::seed" not in features
    assert features["scenario::source::s06a"] == 1.0
    assert features["scenario::within_source_family_index"] == 0.0


def test_target_registry_retains_every_cost_and_separates_licensed_fields() -> None:
    from src.surrogate_remediation.core import read_jsonl, S05_ROOT

    rows = [
        row
        for row in read_jsonl(S05_ROOT / "evaluation_ledger.jsonl")
        if row["selectedForCanonicalSearch"] and row["taskId"] == "e07_s02_sorting_1d"
    ]
    registry = build_s06a_target_registry("e07_s02_sorting_1d", rows)
    names = {item["name"] for item in registry["continuous"]}
    assert any("licensedPrefix" in name for name in names)
    assert any("licensedLongRangeMaximumRequestedDistance" in name for name in names)
    assert all(item["name"] != "cost::total" for item in registry["continuous"])
    assert registry["universalScore"] is False


def test_blend_selection_is_deterministic_and_uses_only_supplied_panels() -> None:
    truth = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    ridge = np.asarray([0.2, 0.8, 0.4, 0.6, 0.3, 0.7])
    tree = np.asarray([0.1, 0.9, 0.2, 0.8, 0.1, 0.9])
    constant = np.full(6, 0.5)
    panels = [np.asarray([0, 1]), np.asarray([2, 3])]
    continuous = _select_continuous_blend(truth, ridge, tree, panels, [0.0, 0.5, 1.0])
    binary = _select_binary_blend(
        truth, ridge, tree, constant, panels, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]
    )
    assert continuous == _select_continuous_blend(
        truth, ridge, tree, panels, [0.0, 0.5, 1.0]
    )
    assert binary == _select_binary_blend(
        truth, ridge, tree, constant, panels, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]
    )


def test_s06_rejected_bundles_are_never_loaded_by_s06a_code() -> None:
    paths = [
        Path("src/surrogate_remediation/core.py"),
        Path("scripts/run_surrogate_remediation_s06a.py"),
        Path("scripts/validate_surrogate_remediation_s06a.py"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert 'S06_ROOT / "models"' not in source
    assert "torch.load" not in source


def test_native_validation_failures_are_retained_only_as_failed_outcomes() -> None:
    handled = _native_failure_accounting(
        [
            {
                "taskId": "e07_s02_regeneration_1d",
                "scenarioOrdinal": 100,
                "policySha256": "p",
                "failed": True,
                "validation": {
                    "exactReplay": True,
                    "developmentBudgetRespected": False,
                },
            }
        ]
    )
    inconsistent = _native_failure_accounting(
        [
            {
                "taskId": "e07_s02_regeneration_1d",
                "scenarioOrdinal": 100,
                "policySha256": "p",
                "failed": False,
                "validation": {
                    "exactReplay": True,
                    "developmentBudgetRespected": False,
                },
            }
        ]
    )
    assert handled["success"] is True
    assert handled["falseNativeValidationRowsMarkedFailed"] == 1
    assert inconsistent["success"] is False
