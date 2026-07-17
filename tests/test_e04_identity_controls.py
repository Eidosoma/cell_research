from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.dynamic_nulls import _label_curves, trajectory_outcomes
from analysis.identity_controls import (
    CALIBRATION_CHANNELS,
    CANDIDATE_CHANNELS,
    CHANNELS,
    CONTRACT_PATH,
    PRIMARY_CHANNEL,
    REFERENCE_CHANNELS,
    _bh_adjust,
    _labelled_copy,
    _source_policy_audit,
    ghost_assignments,
    native_parameters,
    value_block_assignment,
)
from analysis.kinetic_matching import execute_kinetic_run
from reference_simulator.model import Cell, Direction, FaultMode, Policy, Scenario


def _scenario() -> Scenario:
    values = (8, 3, 7, 1, 6, 2, 5, 4)
    policies = (Policy.BUBBLE, Policy.SELECTION)
    cells = tuple(
        Cell(
            f"cell-{index:04d}",
            value,
            policies[index % 2],
            Direction.ASCENDING,
            FaultMode.NORMAL,
        )
        for index, value in enumerate(values)
    )
    return Scenario.create(
        cells,
        initial_occupancy=tuple(cell.cell_id for cell in cells),
        seed=808808,
        max_activations=200_000,
        generation_key="E04/S08/test",
    )


def test_contract_freezes_three_controls_and_stops_before_s09() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["researchStepId"] == "S08"
    assert contract["scope"]["sourceRegime"] == "S07 native_control only"
    assert contract["scope"]["stopsBefore"] == "S09"
    assert "joint_equal" in contract["scope"]["excludedSourceRegimes"]
    assert {item["id"] for item in contract["controlFamilies"]} == {
        "distinct_labels_identical_policy",
        "same_label_distinct_policies",
        "action_irrelevant_ghost_labels",
    }
    assert contract["postHocArtifactRules"]["exclusionRule"].startswith(
        "Only channel 0"
    )


def test_channel_partition_is_frozen_and_disjoint() -> None:
    assert PRIMARY_CHANNEL == 0
    assert len(REFERENCE_CHANNELS) == 500
    assert len(CALIBRATION_CHANNELS) == 11
    assert len(CANDIDATE_CHANNELS) == 20
    assert (
        {PRIMARY_CHANNEL} | set(REFERENCE_CHANNELS) | set(CALIBRATION_CHANNELS)
    ) == set(range(CHANNELS))
    assert not (set(REFERENCE_CHANNELS) & set(CALIBRATION_CHANNELS))


def test_ghost_assignments_are_deterministic_exact_and_scenario_addressed() -> None:
    first = ghost_assignments("scenario-alpha")
    replay = ghost_assignments("scenario-alpha")
    second = ghost_assignments("scenario-beta")
    assert first.shape == (512, 100)
    assert np.array_equal(first, replay)
    assert np.all(np.sum(first == 0, axis=1) == 50)
    assert np.all(np.sum(first == 1, axis=1) == 50)
    assert not np.array_equal(first, second)


def test_value_block_assignment_is_exact_and_value_ordered() -> None:
    values = np.arange(100, dtype=np.int16)[::-1]
    labels = value_block_assignment(values, "value-test")
    assert np.sum(labels == 0) == np.sum(labels == 1) == 50
    assert values[labels == 0].max() < values[labels == 1].min()


def test_exact_composition_correction_handles_random_and_collapsed_labels() -> None:
    occupancy = np.broadcast_to(np.arange(100, dtype=np.uint8), (101, 100))
    grouped = np.repeat(np.arange(2, dtype=np.uint8), 50)
    grouped_curve = _label_curves(grouped[None, :], occupancy, 0.49)
    collapsed_curve = _label_curves(np.zeros((1, 100), dtype=np.uint8), occupancy, 0.99)
    assert np.allclose(grouped_curve, 0.49)
    assert np.array_equal(collapsed_curve, np.zeros((1, 101)))
    outcomes = trajectory_outcomes(collapsed_curve)
    assert outcomes["peak"][0] == outcomes["positive_area"][0] == 0


def test_analysis_labels_leave_transition_identical_with_fixed_runtime_key() -> None:
    scenario = _scenario()
    expected = execute_kinetic_run(scenario, native_parameters())
    labels = [f"ghost:{index % 2}" for index in range(len(scenario.cells))]
    labelled = _labelled_copy(scenario, labels)
    assert labelled.scenario_id == scenario.scenario_id
    assert [cell.analysis_label for cell in labelled.cells] == labels
    observed = execute_kinetic_run(
        labelled, native_parameters(), validate_scenario=False
    )
    for field in (
        "stop_reason",
        "activation_count",
        "successful_swap_count",
        "final_state_hash",
        "trajectory_sha256",
        "occupancy_b64",
        "corrected_curve_sha256",
        "ledger_json",
    ):
        assert observed[field] == expected[field]


def test_behavior_source_has_no_analysis_label_observation() -> None:
    audit = _source_policy_audit()
    assert audit["passed"]
    assert audit["analysisLabelAttributeAccesses"] == []
    assert all(Path(path).is_file() for path in audit["auditedFiles"])


def test_bh_adjustment_is_monotone_in_p_value_rank() -> None:
    p_values = np.asarray([0.04, 0.001, 0.2, 0.03])
    adjusted = _bh_adjust(p_values)
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all((adjusted >= p_values) & (adjusted <= 1))
