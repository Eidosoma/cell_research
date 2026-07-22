from __future__ import annotations

import importlib
from pathlib import Path
import sys

from src.environment_suite import Split
from src.environment_suite.suite import EnvironmentSuite
from src.non_surrogate_allocation.core import (
    FAMILY_ORDINALS,
    SPLIT_MANIFEST,
    TASK_REGISTRY,
    derive_train_record,
    validate_physical_result,
)
from src.quality_diversity.core import BASE_SCENARIOS
from src.non_surrogate_allocation.analysis import _auc, _holm


def test_s07_families_are_deterministic_train_only() -> None:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    for task_id, scenario_id in BASE_SCENARIOS.items():
        base = suite.records[scenario_id]
        rows = [derive_train_record(base, ordinal) for ordinal in FAMILY_ORDINALS]
        assert all(row.split is Split.TRAIN and not row.protected for row in rows)
        assert [row.public_dict() for row in rows] == [
            derive_train_record(base, ordinal).public_dict() for ordinal in FAMILY_ORDINALS
        ]


def test_result_validator_accepts_only_prespecified_e05_failure_flag() -> None:
    base = {
        "physicalRowId": "a" * 64, "policySha256": "b" * 64,
        "scenarioCommitmentSha256": "c" * 64, "split": "train", "protected": False,
        "replayPass": True, "taskId": "e07_s02_regeneration_1d", "failed": True,
        "censored": False, "stopReason": "source_terminal",
        "validation": {"developmentBudgetRespected": False, "exactReplay": True},
        "nativeLedgerFamilies": {"x": {"y": 1}}, "nativeEvent": {"kind": "x"},
    }
    from src.environment_suite.contracts import canonical_sha256
    stable = dict(base)
    stable["resultSha256"] = canonical_sha256("E07/S07/physical-evaluation/v1", stable)
    assert validate_physical_result(stable, base) == []
    stable["validation"]["exactReplay"] = False
    assert "native_validation" in validate_physical_result(stable, base)


def test_s07_module_loads_no_rejected_model_package() -> None:
    before = set(sys.modules)
    importlib.import_module("src.non_surrogate_allocation.core")
    loaded = set(sys.modules) - before
    assert not any(name.startswith(("src.surrogate_models", "src.surrogate_remediation")) for name in loaded)
    source = (Path(__file__).parents[1] / "src/non_surrogate_allocation/core.py").read_text()
    assert "from src.surrogate" not in source
    assert "import src.surrogate" not in source


def test_frozen_auc_and_holm_operationalization() -> None:
    assert _auc([1.0, 1.0, 1.0, 1.0]) == 0.875
    adjusted = _holm({"a": 0.01, "b": 0.02, "c": 0.5, "d": 0.8})
    assert adjusted["a"] == 0.04
    assert adjusted["b"] == 0.06
    assert all(0 <= value <= 1 for value in adjusted.values())
