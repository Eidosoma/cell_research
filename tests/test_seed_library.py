from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.environment_suite.contracts import Split
from src.policy_dsl import compile_policy
from src.seed_library.core import (
    FAITHFUL_POLICY_MAP,
    build_candidate_seeds,
    build_s03_evidence,
    build_training_probes,
    load_training_records,
)


REGISTRY = Path("configs/environment_suite/task_registry.yaml")
SPLITS = Path("configs/environment_suite/split_manifest.json")


@pytest.fixture(scope="module")
def evidence():
    return build_s03_evidence(REGISTRY, SPLITS, probes_per_record=16)


def test_candidate_families_and_deterministic_generation():
    first = build_candidate_seeds()
    second = build_candidate_seeds()
    assert len(first) == len(second) == 31
    assert {candidate.family for candidate in first} == {
        "faithful",
        "simplified_variant",
        "random_finite_state",
        "repair",
    }
    first_rows = [
        (candidate.policy_id, candidate.compile().policy_sha256) for candidate in first
    ]
    second_rows = [
        (candidate.policy_id, candidate.compile().policy_sha256) for candidate in second
    ]
    assert first_rows == second_rows


def test_only_frozen_training_records_enter_probe_evaluation():
    _, _, training = load_training_records(REGISTRY, SPLITS)
    assert len(training) == 8
    assert all(record.split == Split.TRAIN for record in training)
    assert len({record.task_id for record in training}) == 8
    probes = build_training_probes(training, probes_per_record=4)
    assert len(probes) == 24  # four line + two spatial; E05 stays unbound
    validation = replace(training[0], split=Split.VALIDATION, protected=False)
    with pytest.raises(ValueError, match="non-training"):
        build_training_probes([validation], probes_per_record=1)


def test_faithful_hashes_behavior_and_licensed_costs(evidence):
    records = {record["policyId"]: record for record in evidence["seedRecords"]}
    evaluations = {row["policyId"]: row for row in evidence["evaluations"]}
    assert set(FAITHFUL_POLICY_MAP) <= set(records)
    for policy_id in FAITHFUL_POLICY_MAP:
        assert (
            compile_policy(records[policy_id]["canonicalPolicy"]).policy_sha256
            == records[policy_id]["policySha256"]
        )
        faithful = evaluations[policy_id]["faithfulReference"]
        assert faithful["matches"] == 64
        assert faithful["mismatches"] == 0

    insertion = evaluations["insertion_cell_view_v1"]["adapterProjectionLedger"]
    assert insertion["licensedPrefixPredicateEvaluations"] == 64
    assert insertion["licensedPrefixValueReads"] > 0
    assert insertion["licensedPrefixValueComparisons"] > 0

    selection = evaluations["selection_cell_view_v1"]["adapterProjectionLedger"]
    assert selection["engineCursorStateReads"] == 64
    assert selection["engineCursorTargetProjectionReads"] > 0
    assert selection["engineCursorAdvanceActions"] > 0
    assert selection["engineCursorSwapActions"] > 0
    assert selection["licensedLongRangeRequestedDistance"] > 0
    assert selection["licensedLongRangeMaximumRequestedDistance"] > 1


def test_simplified_variants_are_named_parented_and_valid(evidence):
    simplified = [
        record
        for record in evidence["seedRecords"]
        if record["family"] == "simplified_variant"
    ]
    assert len(simplified) == 6
    assert all(record["parents"] for record in simplified)
    assert all(
        record["variantValidation"].startswith("Validated") for record in simplified
    )
    assert all(
        compile_policy(record["canonicalPolicy"]).policy_sha256
        == record["policySha256"]
        for record in simplified
    )
    unlicensed = next(
        record
        for record in simplified
        if record["policyId"] == "insertion_adjacent_unlicensed_v1"
    )
    assert "line.prefix_ordered" not in unlicensed["permissions"]
    scan_only = next(
        record
        for record in simplified
        if record["policyId"] == "selection_scan_only_v1"
    )
    assert scan_only["complexity"]["maxMovementRadius"] == 0


def test_bounded_semantic_duplicate_is_detected_and_collapsed(evidence):
    assert evidence["validation"]["counts"]["candidateSeeds"] == 31
    assert evidence["validation"]["counts"]["retainedSeeds"] == 30
    groups = evidence["equivalence"]["duplicateGroups"]
    assert len(groups) == 1
    assert set(groups[0]["memberPolicyIds"]) == {
        "random_line_noop_guard_a_v1",
        "random_line_noop_guard_b_v1",
    }
    assert groups[0]["discardedPolicyIds"] == ["random_line_noop_guard_b_v1"]
    retained = {record["policyId"] for record in evidence["seedRecords"]}
    assert "random_line_noop_guard_a_v1" in retained
    assert "random_line_noop_guard_b_v1" not in retained


def test_task_bindings_preserve_e05_e06_representation_gaps(evidence):
    rows = {row["taskId"]: row for row in evidence["taskBindings"]["rows"]}
    assert rows["e07_s02_regeneration_1d"]["bindingStatus"] == "unbound_preserved"
    assert rows["e07_s02_target_change_1d"]["bindingStatus"] == "unbound_preserved"
    assert (
        rows["e07_s02_spatial2d_local"]["bindingStatus"]
        == "synthetic_typed_fixture_only"
    )
    assert (
        rows["e07_s02_spatial2d_memory"]["bindingStatus"]
        == "synthetic_typed_fixture_only"
    )
    assert evidence["taskBindings"]["arbitraryE05BindingsInferred"] == 0
    assert evidence["taskBindings"]["arbitraryE06BindingsInferred"] == 0
    assert evidence["taskBindings"]["seedEpisodeOutcomesRead"] == 0


def test_structural_coverage_does_not_freeze_s04_descriptors(evidence):
    coverage = evidence["coverage"]
    assert coverage["baselineStructuralCoveragePassed"]
    assert "not frozen S04 objectives" in coverage["scope"]
    assert set(coverage["familyCounts"]) == {
        "faithful",
        "simplified_variant",
        "random_finite_state",
        "repair",
    }
    assert set(coverage["environmentCounts"]) == {"line1d.v1", "spatial2d.v1"}


def test_full_evidence_replays_byte_identically():
    first = build_s03_evidence(REGISTRY, SPLITS, probes_per_record=4)
    second = build_s03_evidence(REGISTRY, SPLITS, probes_per_record=4)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["validation"]["success"]


def test_every_retained_policy_complexity_is_within_declared_limits(evidence):
    for record in evidence["seedRecords"]:
        assert (
            record["complexity"]["worstCaseOperations"]
            <= record["canonicalPolicy"]["limits"]["maxOperationsPerActivation"]
        )
    assert all(
        record["scalarTotalDefined"] is False
        and record["nativePredecessorCostLedgerReplaced"] is False
        for record in evidence["complexityRecords"]
    )
