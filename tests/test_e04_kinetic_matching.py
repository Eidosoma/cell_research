from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from analysis.chimeric_replication import run_reference_summary
from analysis.kinetic_matching import (
    CALIBRATION_REPLICATES,
    CONTRACT_PATH,
    HOLDOUT_REPLICATES,
    POLICY_NAMES,
    RegimeParameters,
    _kinetic_values,
    execute_kinetic_run,
)
from reference_simulator.model import Cell, Direction, FaultMode, Policy, Scenario


def _scenario(policies: tuple[Policy, ...] = (Policy.BUBBLE, Policy.INSERTION)) -> Scenario:
    values = (8, 3, 7, 1, 6, 2, 5, 4)
    cells = tuple(
        Cell(
            f"cell-{index:04d}",
            value,
            policies[index % len(policies)],
            Direction.ASCENDING,
            FaultMode.NORMAL,
        )
        for index, value in enumerate(values)
    )
    return Scenario.create(
        cells,
        initial_occupancy=tuple(cell.cell_id for cell in cells),
        seed=912345,
        max_activations=200_000,
        generation_key="E04/S07/test",
    )


def _native() -> RegimeParameters:
    return RegimeParameters(
        "native_control",
        {policy: 1.0 for policy in POLICY_NAMES},
        {policy: 0.0 for policy in POLICY_NAMES},
        False,
    )


def test_frozen_split_is_disjoint_and_contract_names_s07() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["researchStepId"] == "S07"
    assert contract["scope"]["stopsBefore"] == "S08"
    assert not (set(CALIBRATION_REPLICATES) & set(HOLDOUT_REPLICATES))
    assert len(CALIBRATION_REPLICATES) == len(HOLDOUT_REPLICATES) == 25


def test_native_projection_matches_validated_reference_summary() -> None:
    scenario = _scenario()
    observed = execute_kinetic_run(scenario, _native())
    expected = run_reference_summary(
        scenario,
        family="s07_test_native",
        retain_raw_trace=False,
    )
    assert observed["stop_reason"] == expected["stop_reason"] == "complete"
    assert observed["activation_count"] == expected["activation_count"]
    assert observed["successful_swap_count"] == expected["successful_swap_count"]
    assert observed["final_state_hash"] == expected["final_state_hash"]
    assert observed["trajectory_sha256"] == expected["trajectory_sha256"]


def test_identity_cycle_completes_without_rewriting_policy_direction() -> None:
    scenario = _scenario((Policy.BUBBLE, Policy.SELECTION))
    parameters = RegimeParameters(
        "activation_equal",
        {policy: 1.0 for policy in POLICY_NAMES},
        {policy: 0.0 for policy in POLICY_NAMES},
        True,
    )
    result = execute_kinetic_run(scenario, parameters)
    assert result["completed"]
    assert result["final_consensus_ordered"]
    kinetics = json.loads(result["policy_kinetics_json"])
    values = [kinetics[policy]["activation_opportunity"] for policy in ("Bubble", "Selection")]
    # The terminal can occur partway through the final identity cycle.
    assert abs(values[0] - values[1]) <= 0.04


def test_policy_kinetic_structural_identity_is_exact() -> None:
    ledger = {
        policy: {
            "actor_activations": 100.0,
            "valid_swap_proposals": 20.0,
            "accepted_swaps": 10.0,
            "native_range_sum": 40.0,
            "accepted_range_sum": 30.0,
            "actor_displacement_sum": 30.0,
            "experienced_displacement_sum": 45.0,
            "gate_rejections": 10.0,
            "memory_updates": 0.0,
            "no_ops": 80.0,
        }
        for policy in ("Bubble", "Selection")
    }
    result = _kinetic_values(ledger, {"Bubble": 4, "Selection": 4}, 200)
    assert result["Bubble"]["structural_identity_error"] <= 1e-15
    assert result["Selection"]["structural_identity_error"] <= 1e-15
    assert np.isclose(result["Bubble"]["successful_action_rate"], 0.1)
    assert np.isclose(result["Bubble"]["executed_target_range"], 3.0)


def test_intervention_source_never_reads_analysis_label_attribute() -> None:
    source = Path(__file__).resolve().parents[1] / "analysis/kinetic_matching.py"
    tree = ast.parse(source.read_text())
    accessed = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "analysis_label"
    ]
    assert accessed == []


def test_secret_analysis_labels_do_not_change_transition_with_fixed_rng_key() -> None:
    scenario = _scenario((Policy.BUBBLE, Policy.SELECTION))
    expected = execute_kinetic_run(scenario, _native())
    secret_cells = tuple(
        replace(cell, analysis_label=f"secret-{index % 3}")
        for index, cell in enumerate(scenario.cells)
    )
    secret = Scenario.create(
        secret_cells,
        initial_occupancy=scenario.initial_occupancy,
        seed=scenario.seed,
        max_activations=scenario.max_activations,
        generation_key="E04/S07/test/secret",
    )
    object.__setattr__(secret, "scenario_id", scenario.scenario_id)
    observed = execute_kinetic_run(secret, _native(), validate_scenario=False)
    for field in (
        "stop_reason",
        "activation_count",
        "successful_swap_count",
        "final_state_hash",
        "trajectory_sha256",
        "policy_kinetics_json",
        "occupancy_b64",
    ):
        assert observed[field] == expected[field]
