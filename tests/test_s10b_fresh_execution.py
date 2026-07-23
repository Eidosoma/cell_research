from __future__ import annotations

from copy import deepcopy

from scripts.run_native_event_discovery_s10 import availability_integrity
from scripts.run_native_event_discovery_s10b import (
    reviewer_instructions,
    tree_snapshot,
)
from src.phenotype_discovery.native_features import extract_native_event_features
from tests.test_s10p_native_event_discovery import _spatial_payload


def _row_with_explicit_unavailability() -> dict:
    payload = _spatial_payload()
    movement = payload["costs"]["e06MovementLedger"]
    for key in (
        "submittedProposals",
        "adjacentSwaps",
        "vacancyMoves",
        "shortExchanges",
        "rotations",
        "validProposals",
        "conflictCandidates",
        "reservedSiteClaims",
        "totalGraphDisplacement",
    ):
        movement[key] = 0
    record = extract_native_event_features("e07_s02_spatial2d_local", payload)
    return {
        "taskId": "e07_s02_spatial2d_local",
        "analysisFeatures": record["analysisFeatures"],
        "availability": record["availability"],
    }


def test_s10b_accepts_complete_availability_with_missing_analysis_values() -> None:
    row = _row_with_explicit_unavailability()
    assert len(row["analysisFeatures"]) < 18
    assert len(row["availability"]) == 18
    assert availability_integrity(row)


def test_s10b_availability_fails_closed_on_silent_drop() -> None:
    row = _row_with_explicit_unavailability()
    damaged = deepcopy(row)
    damaged["availability"].pop(next(iter(damaged["availability"])))
    assert not availability_integrity(damaged)


def test_size_aware_tree_snapshot_is_stable(tmp_path) -> None:
    (tmp_path / "a").write_text("one", encoding="utf-8")
    first = tree_snapshot(tmp_path)
    second = tree_snapshot(tmp_path)
    assert first == second
    (tmp_path / "a").write_text("two", encoding="utf-8")
    assert tree_snapshot(tmp_path)["treeSha256"] != first["treeSha256"]


def test_reviewer_packet_instructions_are_s10b_and_annotation_free() -> None:
    text = reviewer_instructions(2)
    assert "Research step ID | **S10B**" in text
    assert "reviewerAnnotation" in text
    assert "intentionally null" in text
