from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import scripts.run_native_event_discovery_s10 as base
import scripts.run_native_event_discovery_s10b as s10b
import scripts.run_native_event_discovery_s10d as s10d


def test_s10d_control_preserves_exact_frozen_budget_and_scope() -> None:
    control = yaml.safe_load(s10d.CONFIG.read_text(encoding="utf-8"))
    assert control["researchStepId"] == "S10D"
    assert control["execution"]["cacheNamespace"] == "/cache/e07-s10d"
    assert control["execution"]["discoveryLogicalRows"] == 7_168
    assert control["execution"]["independentReproductionLogicalRows"] == 3_584
    assert control["execution"]["totalLogicalRows"] == 10_752
    assert control["execution"]["totalPhysicalReplays"] == 21_504
    assert control["execution"]["workers"] == 8
    assert control["execution"]["runtimeDrivenWeakening"] == "forbidden"
    assert control["missingness"]["minimumAvailability"] == 0.90
    assert control["missingness"]["comparison"] == "greater_than_or_equal"
    assert control["scientificDesign"]["population"] == "unchanged_from_S10P"
    assert control["authorization"]["validationOutcomeAccess"] == 0
    assert control["authorization"]["confirmationOutcomeAccess"] == 0


def test_s10d_callback_targets_install_on_the_actual_base_surface() -> None:
    targets = s10d._callback_overrides()
    before = {name: getattr(base, name) for name in targets}
    with s10b.installed_base_callbacks(overrides=targets):
        assert all(getattr(base, name) is target for name, target in targets.items())
    assert all(getattr(base, name) is before[name] for name in targets)


def test_s10d_preflight_and_freeze_dispatch_once_on_prepared_empty_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "fresh-cache"
    cache.mkdir()
    output = tmp_path / "output"
    monkeypatch.setattr(s10d, "CACHE", cache)
    monkeypatch.setattr(s10d, "OUTPUT", output)
    monkeypatch.setattr(s10d, "_FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY", True)
    monkeypatch.setattr(s10b, "STEP_ID", "S10D")
    monkeypatch.setattr(s10b, "CACHE", cache)
    monkeypatch.setattr(s10b, "OUTPUT", output)
    monkeypatch.setattr(s10b, "CONFIG", s10d.CONFIG)
    monkeypatch.setattr(s10b, "SCRIPT", s10d.SCRIPT)
    monkeypatch.setattr(s10b, "TEST", s10d.TEST)
    monkeypatch.setattr(s10b, "HISTORICAL", s10d.HISTORICAL)
    monkeypatch.setattr(base, "CACHE", cache)
    monkeypatch.setattr(base, "OUTPUT", output)
    monkeypatch.setattr(base, "CONFIG", s10d.CONFIG)
    monkeypatch.setattr(base, "SCRIPT", s10d.SCRIPT)
    monkeypatch.setattr(base, "TEST", s10d.TEST)
    monkeypatch.setattr(base, "HISTORICAL", s10d.HISTORICAL)
    s10d._CALLBACK_COUNTS.update(
        {"revalidate_frozen_inputs": 0, "prospective_freeze": 0}
    )
    try:
        with s10b.installed_base_callbacks(overrides=s10d._callback_overrides()):
            preflight = base.revalidate_frozen_inputs()
            freeze = base.prospective_freeze(preflight)
        assert preflight["allPass"]
        assert preflight["freshNamespaceGate"]["pass"]
        assert preflight["installedCallbackGate"]["pass"]
        assert freeze["installedCallbackIdentityPass"]
        assert freeze["installedCallbackInvocationCounts"] == {
            "revalidate_frozen_inputs": 1,
            "prospective_freeze": 1,
        }
        assert not any(cache.iterdir())
    finally:
        s10d._CALLBACK_COUNTS.update(
            {"revalidate_frozen_inputs": 0, "prospective_freeze": 0}
        )


def test_s10d_freeze_fails_before_delegate_on_wrong_callback_count() -> None:
    s10d._CALLBACK_COUNTS.update(
        {"revalidate_frozen_inputs": 0, "prospective_freeze": 0}
    )
    try:
        with pytest.raises(RuntimeError, match="callback execution gate"):
            s10d.prospective_freeze({"allPass": True})
    finally:
        s10d._CALLBACK_COUNTS.update(
            {"revalidate_frozen_inputs": 0, "prospective_freeze": 0}
        )


def test_s10d_all_five_predecessor_trees_match_the_prospective_freeze() -> None:
    record = s10d._full_immutable_tree_document()
    assert record["allPass"]
    assert [row["stepId"] for row in record["rows"]] == [
        "S10P",
        "S10",
        "S10A",
        "S10B",
        "S10C",
    ]
    assert all(row["pass"] for row in record["rows"])


def test_s10a_fail_atomic_gate_is_still_mandatory() -> None:
    gate = json.loads(
        (s10d.S10A / "fresh_s10_execution_gate.json").read_text(encoding="utf-8")
    )
    assert gate["qualificationPass"]
    assert gate["missingnessRemediationQualified"]
    assert gate["allRowAccountingQualified"]
    assert gate["mustPublishDispositionLedgerBeforeResultRows"]
    assert gate["mustStopFailClosedOnAnyIntegrityFailure"]
    assert not gate["failedS10CacheReusePermitted"]
    assert not gate["failedS10OutcomeReusePermitted"]


def test_s10d_manifest_is_step_scoped_and_excludes_self_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "one.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(s10d, "OUTPUT", tmp_path)
    manifest = s10d.manifest_for_output()
    assert manifest["researchStepId"] == "S10D"
    assert manifest["schemaVersion"] == "e07.s10d.artifact-manifest.v1"
    assert [Path(row["path"]).name for row in manifest["artifacts"]] == ["one.json"]
