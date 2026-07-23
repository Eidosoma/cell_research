from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.phenotype_discovery.search import (
    _load_candidate_bundles,
    cluster_jaccard,
    configuration_profiles,
    fit_preprocessing,
    fold_for_family,
    holm_adjust,
    result_integrity,
    rows_digest,
)


def _rows() -> list[dict]:
    rows = []
    for family in range(10):
        for candidate in ("a", "b"):
            features = {
                f"feature_{index}": float(index + family + (candidate == "b"))
                for index in range(6)
            }
            rows.append(
                {
                    "logicalOrdinal": family * 2 + (candidate == "b"),
                    "logicalReservationId": f"reservation-{family}-{candidate}",
                    "logicalResultSha256": f"{family * 2 + (candidate == 'b'):064x}",
                    "candidateId": candidate,
                    "scenarioFamilyId": f"family-{family}",
                    "analysisFeatures": features,
                    "statusStratum": "transition_budget|failed=false|censored=true",
                    "confounds": {
                        "observedEventLength": 32,
                        "nativeHorizon": {"budget": 32},
                        "structuralScenarioSize": 64,
                    },
                    "replayPass": True,
                    "nativeContractPass": True,
                    "completeTrajectoryReconstructed": False,
                    "outcomePlaneReadForFeatures": False,
                }
            )
    return rows


def test_frozen_candidate_bundles_recompile() -> None:
    bundles = _load_candidate_bundles()
    assert len(bundles) == 14
    assert (
        sum(
            item["candidate"]["candidateRole"] == "s09_parent"
            for item in bundles.values()
        )
        == 7
    )
    assert (
        sum(
            item["candidate"]["candidateRole"] == "s09_compressed"
            for item in bundles.values()
        )
        == 7
    )


def test_fold_is_stable_and_bounded() -> None:
    assert fold_for_family("family-1") == fold_for_family("family-1")
    assert 0 <= fold_for_family("family-1") < 5


def test_preprocessing_and_profiles_are_deterministic() -> None:
    rows = _rows()
    first = fit_preprocessing(rows)
    second = fit_preprocessing(deepcopy(rows))
    a = first.pop("discoveryStandardized")
    b = second.pop("discoveryStandardized")
    assert first == second
    assert np.array_equal(a, b)
    ids, profiles = configuration_profiles(rows, a)
    assert ids == ["a", "b"]
    assert profiles.shape == (2, 6)


def test_cluster_jaccard_is_label_invariant() -> None:
    assert cluster_jaccard([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert cluster_jaccard([0, 0, 1, 1], [0, 1, 0, 1]) < 1.0


def test_holm_adjustment_uses_complete_candidate_family() -> None:
    candidates = [
        {"machineCandidateKey": "a", "rawPValue": 0.01},
        {"machineCandidateKey": "b", "rawPValue": 0.03},
        {"machineCandidateKey": "c", "rawPValue": 0.20},
    ]
    adjusted = holm_adjust(candidates)
    by_key = {row["machineCandidateKey"]: row for row in adjusted}
    assert by_key["a"]["holmAdjustedPValue"] == 0.03
    assert by_key["b"]["holmAdjustedPValue"] == 0.06
    assert by_key["c"]["holmAdjustedPValue"] == 0.20


def test_result_integrity_is_worker_order_independent() -> None:
    rows = _rows()
    audit = result_integrity(rows, len(rows))
    assert audit["pass"]
    assert audit["digestNatural"] == audit["digestReverse"]
    assert rows_digest(rows) == rows_digest(list(reversed(rows)))
