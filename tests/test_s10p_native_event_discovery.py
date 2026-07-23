from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from src.phenotype_discovery.native_features import (
    TASKS,
    FeatureExtractionError,
    build_feature_registry,
    exact_change_points,
    extract_native_event_features,
    registry_document,
    validate_ordered_transition_summaries,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _summaries() -> list[dict[str, object]]:
    current = _sha("initial")
    rows = []
    for index in range(32):
        changed = index >= 16
        post = _sha(f"state:{index}") if changed else current
        rows.append(
            {
                "transitionIndex": index,
                "scheduledActorCount": 4,
                "proposalCount": 0 if index < 16 else 3,
                "acceptedCount": 0 if index < 16 else 2,
                "conflictLosses": 0,
                "invalidProposals": 0,
                "preStateSha256": current,
                "postStateSha256": post,
                "transitionSha256": _sha(f"transition:{index}"),
            }
        )
        current = post
    return rows


def _spatial_payload() -> dict[str, object]:
    return {
        "event": {
            "schemaVersion": "e07.s04a.e06-dsl-episode.v1",
            "transitionCount": 32,
        },
        "costs": {
            "e06MovementLedger": {
                "submittedProposals": 64,
                "adjacentSwaps": 8,
                "vacancyMoves": 7,
                "shortExchanges": 6,
                "rotations": 5,
                "validProposals": 60,
                "conflictCandidates": 4,
                "reservedSiteClaims": 31,
                "totalGraphDisplacement": 42,
            }
        },
        "status": {
            "stopReason": "transition_budget",
            "failed": False,
            "censored": True,
        },
        "horizon": {"nativeUnit": "graph_transition", "budget": 32},
        "structuralScenarioSize": 64,
        "traceSelectionReason": "all_event_summaries",
        "orderedEventSummaries": _summaries(),
    }


def test_registry_is_task_local_and_excludes_efficacy_paths() -> None:
    registry = registry_document()
    assert registry["featureCount"] == len(build_feature_registry())
    assert set(registry["featureCountByTask"]) == set(TASKS)
    assert all(value >= 5 for value in registry["featureCountByTask"].values())
    assert registry["crossTaskPooling"] is False
    assert registry["universalScore"] is None
    for feature in registry["features"]:
        joined = f"{feature['source_path']} {feature['denominator_path']}".lower()
        assert "outcome" not in joined
        assert "objective" not in joined
        assert "descriptor" not in joined


def test_spatial_extraction_is_exact_replay_and_separates_confounds() -> None:
    payload = _spatial_payload()
    first = extract_native_event_features("e07_s02_spatial2d_local", payload)
    second = extract_native_event_features("e07_s02_spatial2d_local", deepcopy(payload))
    assert first == second
    assert len(first["analysisFeatures"]) >= 15
    assert first["confounds"]["taskId"] == "e07_s02_spatial2d_local"
    assert first["confounds"]["observedEventLength"] == 32
    assert first["confounds"]["traceAvailability"] == "authentic_ordered_event_summary"
    assert "stopReason" not in first["analysisFeatures"]
    assert first["provenance"]["outcomePlaneRead"] is False


def test_ordered_summary_validator_fails_closed_on_forgery() -> None:
    summaries = _summaries()
    assert validate_ordered_transition_summaries(summaries)["stateHashChainPass"]
    forged = deepcopy(summaries)
    forged[1]["preStateSha256"] = _sha("forged")
    with pytest.raises(FeatureExtractionError, match="chain"):
        validate_ordered_transition_summaries(forged)
    with pytest.raises(FeatureExtractionError, match="count"):
        validate_ordered_transition_summaries(summaries[:-1])


def test_outcomes_and_cross_contract_sequences_are_denied() -> None:
    payload = _spatial_payload()
    payload["outcome"] = {"completed": True}
    with pytest.raises(FeatureExtractionError, match="forbidden"):
        extract_native_event_features("e07_s02_spatial2d_local", payload)

    line = _spatial_payload()
    line["event"] = {
        "schemaVersion": "e07.s04a.line-dsl-event.v1",
        "activationCount": 16,
    }
    line["costs"] = {
        "e01ReferenceLedger": {
            "activations": 16,
            "observationReads": 31,
            "valueComparisons": 15,
            "proposals": 16,
            "noOps": 7,
            "rejections": 3,
            "memoryUpdates": 1,
            "acceptedSwaps": 5,
            "displacedCells": 10,
            "conflictLosses": 0,
        }
    }
    with pytest.raises(FeatureExtractionError, match="unauthorized"):
        extract_native_event_features("e07_s02_sorting_1d", line)


def test_missing_ordered_summaries_are_explicit_not_imputed() -> None:
    payload = _spatial_payload()
    del payload["orderedEventSummaries"]
    record = extract_native_event_features("e07_s02_spatial2d_memory", payload)
    ordered = {
        key: value
        for key, value in record["availability"].items()
        if ".ordered." in key
    }
    assert ordered
    assert all(value["state"] == "unavailable" for value in ordered.values())
    assert all(
        value["reason"] == "ordered_event_summary_unavailable"
        for value in ordered.values()
    )
    assert record["confounds"]["traceAvailability"] == "terminal_summary_only"


def test_exact_change_points_uses_prespecified_deterministic_tie_break() -> None:
    step = [[0.0, 0.0] for _ in range(16)] + [[3.0, 2.0] for _ in range(16)]
    assert exact_change_points(step) == (16,)
    assert exact_change_points([[1.0, 1.0] for _ in range(32)]) == ()
    assert exact_change_points(step) == exact_change_points(deepcopy(step))
