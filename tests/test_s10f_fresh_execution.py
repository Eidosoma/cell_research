from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_native_event_discovery_s10 as base
import scripts.run_native_event_discovery_s10f as runner
from src.phenotype_discovery import feasible_search
from src.phenotype_discovery.method_feasibility import fold_for_family
from src.phenotype_discovery.native_features import build_feature_registry


def _families_covering_folds() -> list[str]:
    found: dict[int, str] = {}
    ordinal = 0
    while len(found) < 5:
        value = f"synthetic-family-{ordinal:04d}"
        found.setdefault(fold_for_family(value), value)
        ordinal += 1
    return [found[index] for index in range(5)]


def _row(candidate: str, family: str, value: float) -> dict:
    feature_ids = [
        spec.feature_id
        for spec in build_feature_registry()
        if spec.task_id == "e07_s02_spatial2d_local"
    ]
    features = {
        feature_id: value + index
        for index, feature_id in enumerate(feature_ids)
    }
    return {
        "candidateId": candidate,
        "scenarioFamilyId": family,
        "taskId": "e07_s02_spatial2d_local",
        "statusStratum": "synthetic_terminal",
        "analysisFeatures": features,
        "availability": {
            name: {
                "state": "observed_or_exact_native_derived",
                "reasonCode": None,
            }
            for name in features
        },
        "orderedEventSeries": {
            "proposalCount": [0] * 31,
            "acceptedCount": [0] * 31,
            "conflictLosses": [0] * 31,
            "stateHashChanged": [False] * 31,
        },
        "confounds": {
            "nativeHorizon": {"budget": 32},
            "observedEventLength": 31,
            "structuralScenarioSize": 8,
        },
    }


def test_incomplete_grid_and_short_series_are_non_evidentiary() -> None:
    families = _families_covering_folds()
    rows = [
        _row(f"candidate-{index:02d}", families[index % 5], float(index))
        for index in range(14)
    ]
    catalog, lock = feasible_search.discovery_analysis(rows)
    audit = feasible_search.discovery_audit()

    assert catalog["machineCandidates"] == []
    assert catalog["discoveryPassingCandidates"] == []
    assert catalog["machineMultiplicityFamilySize"] == 21
    assert len(lock["fixedHolmSlots"]) == 21
    assert audit["fixedHolmSlotCount"] == 21
    assert audit["fixedFamilyShrunk"] is False
    assert all(row["rawPValue"] == 1.0 for row in audit["holmSlots"])
    assert all(
        row["slotState"] == "non_evidentiary_infeasible"
        for row in audit["holmSlots"]
    )
    assert all(
        result["analysisState"]
        == "non_evidentiary_not_candidate_not_null"
        for result in catalog["results"]["changePoint"]
    )


def test_structural_projection_excludes_feature_values() -> None:
    row = _row("candidate-00", "family-00", 12345.0)
    projected = feasible_search.stratum_structure(
        [row], provisional_distinct_profiles=True
    )
    text = repr(projected)
    assert "12345" not in text
    assert len(projected["rows"][0]["observedFeatureIds"]) == 18


def test_installed_preflight_binds_actual_s10f_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "CACHE", tmp_path / "fresh")
    monkeypatch.setattr(runner, "_FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY", True)
    runner.CACHE.mkdir()
    runner._CALLBACK_COUNTS.update(
        {"revalidate_frozen_inputs": 0, "prospective_freeze": 0}
    )
    runner._configure_globals()
    monkeypatch.setattr(
        base, "discovery_analysis", feasible_search.discovery_analysis
    )
    monkeypatch.setattr(
        base, "reproduction_analysis", feasible_search.reproduction_analysis
    )
    with runner.s10b.installed_base_callbacks(overrides=runner._callbacks()):
        preflight = base.revalidate_frozen_inputs()
        assert preflight["allPass"]
        assert preflight["installedBindingGate"]["pass"]
        assert preflight["s10eExecutionReviewGatePass"]
        assert len(preflight["bindingRows"]) == 28
        assert preflight["preOutcomeCommitmentAudit"]["logicalReservations"] == 10_752
        assert (
            preflight["preOutcomeCommitmentAudit"][
                "physicalReplayCommitments"
            ]
            == 21_504
        )
        freeze = base.prospective_freeze(preflight)
        assert freeze["researchStepId"] == "S10F"
        assert freeze["fixedHolmSlotsPerRealizedStratum"] == 21
        assert freeze["fixedHolmFamilyShrinkPermitted"] is False
        assert freeze["failedOrQuarantineOutcomeRowsReusedBeforeFreeze"] == 0
