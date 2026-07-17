from __future__ import annotations

from itertools import product
import json

import numpy as np

from analysis.declustering_intervention import (
    CONTRACT_PATH,
    GRID,
    PEAK_PROGRESS,
    _curve_outcomes,
    _same_edges,
    build_continuation,
    matched_optimum_patterns,
    optimal_policy_pattern,
    realize_identity_shuffle,
    run_branch,
    scheduler_address_signature,
)
from reference_simulator.engine import initial_state
from reference_simulator.model import (
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    state_hash,
)


def _scenario() -> Scenario:
    cells = tuple(
        Cell(
            f"cell-{index:04d}",
            index % 10,
            Policy.BUBBLE if index < 50 else Policy.SELECTION,
            Direction.ASCENDING,
            FaultMode.NORMAL,
            "cohort:Bubble" if index < 50 else "cohort:Selection",
        )
        for index in range(100)
    )
    return Scenario.create(
        cells,
        initial_occupancy=tuple(cell.cell_id for cell in cells),
        seed=101010,
        max_activations=20_000,
        generation_key="E04/S10/test",
    )


def test_contract_freezes_intervention_and_stops_before_s11() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["researchStepId"] == "S10"
    assert contract["scope"]["stopsBefore"] == "S11"
    assert contract["scope"]["sourceScenarioCount"] == 75
    assert contract["scope"]["uniqueFeasibilityScenarios"] == 75
    assert contract["scope"]["matchedNullChannels"] == 32
    assert contract["checkpoint"]["progressByCondition"] == PEAK_PROGRESS
    assert contract["feasibilityBoundary"]["uniqueInputs"].startswith(
        "With one identity per value"
    )
    assert contract["operationalAttractorCriteria"]["supportive"].startswith(
        "All operationalAttractorCriteria gates pass"
    )
    assert np.array_equal(GRID, np.linspace(0, 1, 101))


def test_exact_optimizer_matches_exhaustive_small_arrangements() -> None:
    case = 0
    for n in range(2, 9):
        for values in (
            tuple(index // 2 for index in range(n)),
            tuple(index % min(3, n) for index in range(n)),
        ):
            for policies_binary in product((0, 1), repeat=n):
                if len(set(policies_binary)) != 2:
                    continue
                policies = tuple("A" if item == 0 else "B" for item in policies_binary)
                observed = optimal_policy_pattern(values, policies, ("exhaustive", case))
                feasible_edges = []
                for candidate in product((0, 1), repeat=n):
                    if all(
                        sum(candidate[index] for index, value in enumerate(values) if value == stratum)
                        == sum(policies_binary[index] for index, value in enumerate(values) if value == stratum)
                        for stratum in set(values)
                    ):
                        feasible_edges.append(_same_edges(np.asarray(candidate, dtype=np.uint8)))
                assert observed["minimumSameEdges"] == min(feasible_edges)
                assert _same_edges(observed["binary"]) == min(feasible_edges)
                case += 1


def test_optimizer_and_matched_nulls_are_deterministic_exact_and_unique() -> None:
    values = [index % 10 for index in range(100)]
    policies = ["A" if index % 3 else "B" for index in range(100)]
    first = optimal_policy_pattern(values, policies, ("determinism", 1))
    replay = optimal_policy_pattern(values, policies, ("determinism", 1))
    assert np.array_equal(first["binary"], replay["binary"])
    physical, patterns, audit = matched_optimum_patterns(
        values, first, "test-null-support", channels=8
    )
    replay_physical, replay_patterns, replay_audit = matched_optimum_patterns(
        values, first, "test-null-support", channels=8
    )
    assert np.array_equal(physical, replay_physical)
    assert np.array_equal(patterns, replay_patterns)
    assert audit == replay_audit
    assert audit["uniquePatterns"] == 8
    assert audit["demonstratedDistinctAssignments"] == 9
    assert audit["allEdgeMatched"]
    assert _same_edges(physical) == audit["supportQualifiedSameEdges"]
    assert all(_same_edges(row) == audit["supportQualifiedSameEdges"] for row in patterns)
    assert audit["supportQualifiedSameEdges"] >= first["minimumSameEdges"]
    values_array = np.asarray(values)
    for row in patterns:
        for value, count in first["policyOneCounts"].items():
            assert int(row[values_array == value].sum()) == count


def test_identity_realization_preserves_value_sequence_and_owned_cursor() -> None:
    source = _scenario()
    checkpoint = initial_state(source)
    for index, cell_id in enumerate(sorted(checkpoint.selection_cursors)):
        checkpoint.selection_cursors[cell_id] = index
    values = [source.cell_map[cell_id].value for cell_id in checkpoint.occupancy]
    policies = [source.cell_map[cell_id].policy.value for cell_id in checkpoint.occupancy]
    optimum = optimal_policy_pattern(values, policies, (source.scenario_id, "test"))
    occupancy, realization = realize_identity_shuffle(source, checkpoint, optimum["pattern"])
    assert realization["occupancyChanged"]
    assert sorted(occupancy) == sorted(checkpoint.occupancy)
    assert [source.cell_map[cell_id].value for cell_id in occupancy] == values
    assert [source.cell_map[cell_id].policy.value for cell_id in occupancy] == list(
        optimum["pattern"]
    )
    observed, state, audit = build_continuation(
        source, checkpoint, occupancy, "test_primary", "identity_owned"
    )
    assert observed.scenario_id == source.scenario_id
    assert state.selection_cursors == checkpoint.selection_cursors
    assert audit["value_sequence_preserved"]
    assert audit["sortedness_preserved"]
    assert audit["cell_static_fields_preserved"]
    assert audit["policy_value_counts_preserved"]
    assert audit["identity_cursor_preserved"]


def test_position_cursor_sensitivity_is_confined_to_cursor_map() -> None:
    source = _scenario()
    checkpoint = initial_state(source)
    for index, cell_id in enumerate(sorted(checkpoint.selection_cursors)):
        checkpoint.selection_cursors[cell_id] = index + 1
    values = [source.cell_map[cell_id].value for cell_id in checkpoint.occupancy]
    policies = [source.cell_map[cell_id].policy.value for cell_id in checkpoint.occupancy]
    optimum = optimal_policy_pattern(values, policies, (source.scenario_id, "cursor"))
    occupancy, _ = realize_identity_shuffle(source, checkpoint, optimum["pattern"])
    primary_scenario, primary_state, _ = build_continuation(
        source, checkpoint, occupancy, "test_primary", "identity_owned"
    )
    sensitivity_scenario, sensitivity_state, audit = build_continuation(
        source,
        checkpoint,
        occupancy,
        "test_sensitivity",
        "position_retain_reset_new",
    )
    assert primary_scenario.scenario_id == sensitivity_scenario.scenario_id == source.scenario_id
    assert primary_state.occupancy == sensitivity_state.occupancy
    assert primary_state.activation_count == sensitivity_state.activation_count
    assert primary_state.stream_counters == sensitivity_state.stream_counters
    assert primary_state.ledger == sensitivity_state.ledger
    assert primary_state.selection_cursors != sensitivity_state.selection_cursors
    assert not audit["identity_cursor_preserved"]


def test_matched_labels_are_transition_inert_under_source_runtime_key() -> None:
    source = _scenario()
    checkpoint = initial_state(source)
    policy_labels = {cell.cell_id: cell.policy.value for cell in source.cells}
    alternate_labels = {
        cell_id: ("A" if index < 50 else "B")
        for index, cell_id in enumerate(checkpoint.occupancy)
    }
    no_scenario, no_state, _ = build_continuation(
        source, checkpoint, checkpoint.occupancy, "no", "identity_owned"
    )
    label_scenario, label_state, audit = build_continuation(
        source,
        checkpoint,
        checkpoint.occupancy,
        "label",
        "identity_owned",
        alternate_labels,
    )
    no_result = run_branch(no_scenario, no_state, policy_labels)
    label_result = run_branch(label_scenario, label_state, alternate_labels)
    assert audit["labels_changed"]
    assert state_hash(no_scenario.scenario_id, no_result["state"]) == state_hash(
        label_scenario.scenario_id, label_result["state"]
    )
    for field in ("activations", "swaps", "occupancies"):
        assert np.array_equal(no_result[field], label_result[field])
    assert scheduler_address_signature(no_scenario, 0, 1_000) == scheduler_address_signature(
        label_scenario, 0, 1_000
    )


def test_recovery_outcomes_use_extent_timing_rate_and_positive_area() -> None:
    curve = np.linspace(0.0, 1.0, 101)
    observed = _curve_outcomes(curve)
    assert observed["recovery_extent"] == 1.0
    assert np.isclose(observed["recovery_area"], 0.5)
    assert np.isclose(observed["recovery_rate"], 1.0)
    assert observed["time_to_half"] == 0.5
    assert observed["time_to_max"] == 1.0
    negative = _curve_outcomes(-curve)
    assert negative["recovery_area"] == 0.0
    assert np.isnan(negative["time_to_half"])
