from __future__ import annotations

import json

import numpy as np

from analysis.policy_label_switches import (
    ARM_ORDER,
    CONTRACT_PATH,
    GRID,
    LABEL_ASSIGNMENT,
    POLICY_ASSIGNMENT,
    RULE_BY_ARM,
    _run_branch,
    _scenario_with_assignments,
    corrected_adjacency,
    orthogonal_exchange,
)
from reference_simulator.engine import initial_state
from reference_simulator.model import Cell, Direction, FaultMode, Policy, Scenario, state_hash


def _scenario() -> Scenario:
    policies = [Policy.BUBBLE] * 50 + [Policy.SELECTION] * 50
    cells = tuple(
        Cell(
            f"cell-{index:04d}",
            100 - index,
            policies[index],
            Direction.ASCENDING,
            FaultMode.NORMAL,
            f"cohort:{policies[index].value}",
        )
        for index in range(100)
    )
    return Scenario.create(
        cells,
        initial_occupancy=tuple(cell.cell_id for cell in cells),
        seed=90909,
        max_activations=500,
        generation_key="E04/S09/test",
    )


def test_contract_freezes_scope_timing_rules_and_stops_before_s10() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["researchStepId"] == "S09"
    assert contract["scope"]["stopsBefore"] == "S10"
    assert contract["scope"]["sourceScenarioCount"] == 150
    assert contract["scope"]["continuations"] == 3150
    assert set(contract["timing"]["peakProgressByCondition"].values()) == {
        0.16,
        0.19,
        0.27,
        0.33,
        1.0,
    }
    assert contract["selectionInternalState"]["primaryRule"].startswith(
        "retain_old_reset_new"
    )
    assert contract["selectionInternalState"]["sensitivityRule"].startswith(
        "donor_transfer"
    )


def test_factorial_arm_definitions_are_complete() -> None:
    assert len(ARM_ORDER) == 7
    assert set(ARM_ORDER) == set(POLICY_ASSIGNMENT) == set(LABEL_ASSIGNMENT) == set(RULE_BY_ARM)
    assert POLICY_ASSIGNMENT["label_only"] == "original"
    assert LABEL_ASSIGNMENT["label_only"] == "switched"
    assert POLICY_ASSIGNMENT["policy_only_primary"] == "switched"
    assert LABEL_ASSIGNMENT["policy_only_primary"] == "original"
    assert np.array_equal(GRID, np.linspace(0, 1, 101))


def test_orthogonal_exchange_is_deterministic_exact_and_nontrivial() -> None:
    scenario = _scenario()
    first = orthogonal_exchange(scenario)
    replay = orthogonal_exchange(scenario)
    assert first == replay
    assert first["changedCount"] == 50
    assert set(first["crossTab"].values()) == {25}
    assert sorted(first["original"].values()).count("Bubble") == 50
    assert sorted(first["switched"].values()).count("Bubble") == 50
    original_membership = {
        frozenset(cell_id for cell_id, policy in first["original"].items() if policy == "Bubble"),
        frozenset(cell_id for cell_id, policy in first["original"].items() if policy == "Selection"),
    }
    switched_membership = {
        frozenset(cell_id for cell_id, policy in first["switched"].items() if policy == "Bubble"),
        frozenset(cell_id for cell_id, policy in first["switched"].items() if policy == "Selection"),
    }
    assert original_membership != switched_membership


def test_primary_selection_rule_retains_old_and_resets_new() -> None:
    source = _scenario()
    assignment = orthogonal_exchange(source)
    state = initial_state(source)
    for index, cell_id in enumerate(sorted(state.selection_cursors)):
        state.selection_cursors[cell_id] = index % 100
    observed, modified, audit = _scenario_with_assignments(
        source,
        assignment["switched"],
        assignment["original"],
        state,
        "retain_old_reset_new",
        assignment["donor"],
        "policy_only_primary",
    )
    for cell in observed.cells:
        if cell.policy != Policy.SELECTION:
            assert cell.cell_id not in modified.selection_cursors
        elif assignment["original"][cell.cell_id] == "Selection":
            assert modified.selection_cursors[cell.cell_id] == state.selection_cursors[cell.cell_id]
        else:
            assert modified.selection_cursors[cell.cell_id] == 0
    assert audit["runtime_key_preserved"]
    assert audit["occupancy_preserved"] and audit["activation_preserved"]
    assert audit["stream_counters_preserved"] and audit["ledger_preserved"]


def test_transfer_rule_moves_donor_selection_cursor() -> None:
    source = _scenario()
    assignment = orthogonal_exchange(source)
    state = initial_state(source)
    for index, cell_id in enumerate(sorted(state.selection_cursors)):
        state.selection_cursors[cell_id] = index % 100
    observed, modified, _ = _scenario_with_assignments(
        source,
        assignment["switched"],
        assignment["original"],
        state,
        "donor_transfer",
        assignment["donor"],
        "policy_only_transfer",
    )
    for cell in observed.cells:
        if cell.policy == Policy.SELECTION:
            donor = assignment["donor"][cell.cell_id]
            assert assignment["original"][donor] == "Selection"
            assert modified.selection_cursors[cell.cell_id] == state.selection_cursors[donor]


def test_label_only_is_transition_identical_under_source_runtime_key() -> None:
    source = _scenario()
    assignment = orthogonal_exchange(source)
    state = initial_state(source)
    donor = assignment["donor"]
    no_scenario, no_state, _ = _scenario_with_assignments(
        source,
        assignment["original"],
        assignment["original"],
        state,
        "identity_retain",
        donor,
        "no_switch",
    )
    label_scenario, label_state, _ = _scenario_with_assignments(
        source,
        assignment["original"],
        assignment["switched"],
        state,
        "identity_retain",
        donor,
        "label_only",
    )
    no_result = _run_branch(no_scenario, no_state, assignment["original"], assignment["switched"])
    label_result = _run_branch(label_scenario, label_state, assignment["original"], assignment["switched"])
    assert state_hash(no_scenario.scenario_id, no_result["state"]) == state_hash(
        label_scenario.scenario_id, label_result["state"]
    )
    for field in ("activations", "swaps", "original", "switched"):
        assert np.array_equal(no_result[field], label_result[field])


def test_corrected_adjacency_uses_exact_composition_baseline() -> None:
    occupancy = [f"cell-{index:04d}" for index in range(100)]
    blocked = {cell_id: ("A" if index < 50 else "B") for index, cell_id in enumerate(occupancy)}
    alternating = {cell_id: ("A" if index % 2 == 0 else "B") for index, cell_id in enumerate(occupancy)}
    assert np.isclose(corrected_adjacency(occupancy, blocked), 0.49)
    assert np.isclose(corrected_adjacency(occupancy, alternating), -0.49)
