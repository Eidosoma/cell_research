from __future__ import annotations

from copy import deepcopy

from src.portfolio_ablation.core import (
    SPATIAL_TASKS,
    build_edit_registry,
    checked_protocol,
    compare_paired_rows,
    compose_causal_edits,
    derive_training_record,
    load_parent_bundles,
    make_work_roster,
    validate_s09_inputs,
)
from src.environment_suite import EnvironmentSuite, Split


def test_s09_inputs_are_exactly_seven_frozen_spatial_parents() -> None:
    protocol = checked_protocol()
    validation = validate_s09_inputs(protocol)
    parents = load_parent_bundles(protocol)
    assert validation["success"]
    assert len(parents) == 7
    assert {row["taskId"] for row in parents} == set(SPATIAL_TASKS)
    assert sum(row["taskId"] == SPATIAL_TASKS[0] for row in parents) == 2
    assert sum(row["taskId"] == SPATIAL_TASKS[1] for row in parents) == 5


def test_edit_registry_is_structural_deterministic_and_compilable() -> None:
    parents = load_parent_bundles()
    first, coverage = build_edit_registry(parents)
    second, _ = build_edit_registry(list(reversed(parents)))
    assert coverage["success"]
    assert [row["editId"] for row in first] == [row["editId"] for row in second]
    assert all(not row["configuration"]["s07ArmMembershipUsed"] for row in first)
    assert all(
        not row["configuration"]["rejectedModelOrEmbeddingUsed"] for row in first
    )
    assert all(row["taskId"] in SPATIAL_TASKS for row in first)
    assert sum(row["operator"] == "exact_runtime_sham" for row in first) == 7
    assert coverage["signalAblationsFabricated"] == 0
    assert coverage["communicationAblationsFabricated"] == 0


def test_training_derivation_is_outcome_blind_and_spatial() -> None:
    suite = EnvironmentSuite(
        "configs/environment_suite/task_registry.yaml",
        "configs/environment_suite/split_manifest.json",
    )
    base = next(
        row
        for row in suite.records.values()
        if row.task_id == SPATIAL_TASKS[0] and row.split is Split.TRAIN
    )
    record = derive_training_record(base, 500, panel="screening")
    assert record.split is Split.TRAIN
    assert not record.protected
    assert "counterScheduleKey" in record.public_parameters
    assert "outcome" not in record.public_parameters


def test_roster_deduplicates_parent_conditions() -> None:
    parents = load_parent_bundles()
    edits, _ = build_edit_registry(parents)
    subset = [
        row
        for row in edits
        if row["parentConfigurationId"] == parents[0]["parentConfigurationId"]
    ][:3]
    roster = make_work_roster(parents, subset, panel="screening", families=[500, 501])
    assert len(roster) == 2 * (1 + len(subset))
    assert sum(row["conditionRole"] == "parent" for row in roster) == 2


def test_cumulative_composition_rebuilds_from_frozen_targets() -> None:
    parent = next(
        row for row in load_parent_bundles() if len(row["configuration"]["members"]) == 3
    )
    edits, _ = build_edit_registry([parent])
    candidates = [
        row
        for row in edits
        if row["causalEdit"] and row["operator"] != "member_delete"
    ][:2]
    composed = compose_causal_edits(parent, candidates)
    assert composed["componentEditIds"] == [
        row["editId"]
        for row in sorted(
            candidates,
            key=lambda row: (
                (
                    "member_delete",
                    "selector_remove_to_fixed_balanced",
                    "selector_branch_flip_control",
                    "fixed_assignment_rotation_control",
                    "rule_delete",
                    "memory_freeze_register",
                    "observation_predicate_erase",
                    "sensing_delta_to_counter_index",
                    "sensing_restrict_adjacent_radius_one",
                    "sensing_remove_positive_filter",
                    "exact_runtime_sham",
                ).index(row["operator"]),
                row["editId"],
            ),
        )
    ]
    assert composed["configuration"]["portfolioSize"] >= 2


def test_paired_effect_exact_sham() -> None:
    parent = load_parent_bundles()[0]
    edits, _ = build_edit_registry([parent])
    sham = next(row for row in edits if row["operator"] == "exact_runtime_sham")
    base = {
        "panel": "screening",
        "parentConfigurationId": parent["parentConfigurationId"],
        "taskId": parent["taskId"],
        "scenarioFamilyOrdinal": 500,
        "stopReason": "fixed_transition_budget",
        "failed": False,
        "censored": True,
        "replayPass": True,
        "nativeContractPass": True,
        "endpointProjection": {
            "conjunctive_completion": 0.0,
            "s01_global_mismatch": 0.2,
            "s02_local_acceptance": 1.0,
            "formation_success": None,
            "repair_success": 0.0,
        },
        "finalStateSha256": "a" * 64,
    }
    parent_row = {**deepcopy(base), "editId": None}
    edit_row = {**deepcopy(base), "editId": sham["editId"]}
    effect = compare_paired_rows(
        [parent_row, edit_row],
        [sham],
        bootstrap_replicates=10,
        sign_flip_replicates=10,
    )[0]
    assert effect["exactFunctionalEquivalence"]
    assert effect["boundedFunctionalEquivalence"]
