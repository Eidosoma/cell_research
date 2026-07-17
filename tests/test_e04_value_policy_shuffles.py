from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.value_policy_shuffles import (
    ARMS,
    CONTRACT_PATH,
    GRID,
    PEAK_PROGRESS,
    build_continuation,
    donor_map,
    effect_trajectories,
    select_policy_pair,
    select_value_pair,
    value_rank_strata,
)
from reference_simulator.engine import initial_state
from reference_simulator.model import (
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
)


def _scenario(*, repeated: bool = True) -> Scenario:
    policies = [Policy.BUBBLE, Policy.SELECTION] * 50
    cells = tuple(
        Cell(
            f"cell-{index:04d}",
            index // 10 if repeated else index,
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
        seed=111011,
        max_activations=5000,
        generation_key=f"E04/S11/test/{repeated}",
    )


def test_contract_freezes_scope_boundaries_and_stops_before_s12() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["researchStepId"] == "S11"
    assert contract["scope"]["stopsBefore"] == "S12"
    assert contract["scope"]["sourceScenarioCount"] == 150
    assert contract["scope"]["supportBankDraws"] == 250_000
    assert contract["scope"]["requiredSupportedAssignments"] == 33
    assert contract["valueTargets"]["interpretation"].startswith(
        "Value shuffling changes explicit task progress"
    )
    assert contract["upstreamBoundaries"]["s10CompletionReference"].startswith(
        "S10 identity-owned primary completion was 76%"
    )
    assert len(ARMS) == 10
    assert len(PEAK_PROGRESS) == 6
    assert np.array_equal(GRID, np.linspace(0, 1, 101))


def test_rank_strata_are_exact_value_groups_for_repeated_and_deciles_for_unique() -> None:
    repeated = _scenario(repeated=True)
    unique = _scenario(repeated=False)
    repeated_strata = value_rank_strata(repeated)
    unique_strata = value_rank_strata(unique)
    assert all(
        repeated_strata[cell.cell_id] == int(cell.value)
        for cell in repeated.cells
    )
    assert all(
        unique_strata[cell.cell_id] == int(cell.value) // 10
        for cell in unique.cells
    )
    assert set(pd.Series(repeated_strata).value_counts()) == {10}
    assert set(pd.Series(unique_strata).value_counts()) == {10}


def test_policy_pair_is_supported_equal_edge_equal_hamming_and_stratum_exact() -> None:
    source = _scenario(repeated=True)
    state = initial_state(source)
    strata_by_id = value_rank_strata(source)
    strata = [strata_by_id[cell_id] for cell_id in state.occupancy]
    policies = [source.cell_map[cell_id].policy.value for cell_id in state.occupancy]
    result = select_policy_pair(strata, policies, source.scenario_id)
    original = result["originalBinary"]
    primary = result["primaryBinary"]
    matched = result["matchedBinary"]
    assert result["support"]["demonstratedDistinctAssignments"] == 33
    assert result["hamming"] == np.sum(primary != original)
    assert result["hamming"] == np.sum(matched != original)
    assert np.sum(primary[:-1] == primary[1:]) == result["support"][
        "supportQualifiedSameEdges"
    ]
    assert np.sum(matched[:-1] == matched[1:]) == result["support"][
        "supportQualifiedSameEdges"
    ]
    strata_array = np.asarray(strata)
    for stratum in range(10):
        expected = int(original[strata_array == stratum].sum())
        assert int(primary[strata_array == stratum].sum()) == expected
        assert int(matched[strata_array == stratum].sum()) == expected


def test_value_pair_preserves_multiset_and_matches_policy_support_distance() -> None:
    values = np.repeat(np.arange(10), 10)
    primary_support = np.asarray(
        [group * 10 + offset for group in range(10) for offset in range(4)]
    )
    matched_support = np.asarray(
        [group * 10 + offset for group in range(10) for offset in range(5, 9)]
    )
    result = select_value_pair(
        values.tolist(), primary_support, matched_support, "value-pair-test"
    )
    primary = result["primaryValues"]
    matched = result["matchedValues"]
    assert sorted(primary.tolist()) == sorted(values.tolist())
    assert sorted(matched.tolist()) == sorted(values.tolist())
    assert np.array_equal(np.flatnonzero(primary != values), primary_support)
    assert np.array_equal(np.flatnonzero(matched != values), matched_support)
    assert result["inversionDifference"] >= 0
    assert result["totalValueChangeDifference"] >= 0


def test_component_specific_state_preservation_and_cursor_rules() -> None:
    source = _scenario(repeated=True)
    checkpoint = initial_state(source)
    for index, cell_id in enumerate(sorted(checkpoint.selection_cursors)):
        checkpoint.selection_cursors[cell_id] = index
    original_values = {cell.cell_id: int(cell.value) for cell in source.cells}
    original_policies = {cell.cell_id: cell.policy.value for cell in source.cells}
    switched_policies = dict(original_policies)
    bubble_ids = [cell.cell_id for cell in source.cells if cell.policy == Policy.BUBBLE]
    selection_ids = [
        cell.cell_id for cell in source.cells if cell.policy == Policy.SELECTION
    ]
    for bubble, selection in zip(bubble_ids[:10], selection_ids[:10]):
        switched_policies[bubble] = Policy.SELECTION.value
        switched_policies[selection] = Policy.BUBBLE.value
    switched_values = dict(original_values)
    ids = [cell.cell_id for cell in source.cells[:20]]
    rotated = [original_values[cell_id] for cell_id in ids[10:]] + [
        original_values[cell_id] for cell_id in ids[:10]
    ]
    switched_values.update(dict(zip(ids, rotated)))

    value_scenario, value_state, value_audit = build_continuation(
        source,
        checkpoint,
        switched_values,
        original_policies,
        "value_primary",
        "identity_retain",
    )
    assert value_audit["policy_fields_preserved"]
    assert value_audit["global_value_multiset_preserved"]
    assert value_state.selection_cursors == checkpoint.selection_cursors
    assert any(
        value_scenario.cell_map[cell_id].value != source.cell_map[cell_id].value
        for cell_id in ids
    )

    policy_scenario, policy_state, policy_audit = build_continuation(
        source,
        checkpoint,
        original_values,
        switched_policies,
        "policy_primary",
        "retain_old_reset_new",
    )
    assert policy_audit["value_fields_preserved"]
    assert policy_audit["global_policy_multiset_preserved"]
    for cell in policy_scenario.cells:
        if cell.policy == Policy.SELECTION and original_policies[cell.cell_id] != "Selection":
            assert policy_state.selection_cursors[cell.cell_id] == 0

    donors = donor_map(source, switched_policies, source.scenario_id)
    _, transfer_state, _ = build_continuation(
        source,
        checkpoint,
        original_values,
        switched_policies,
        "policy_transfer",
        "donor_transfer",
    )
    for cell_id, donor_id in donors.items():
        assert transfer_state.selection_cursors[cell_id] == checkpoint.selection_cursors[donor_id]


def test_factorial_effect_identities_hold_exactly() -> None:
    rows = []
    arms = {
        "no_switch": np.asarray([0.0, 1.0]),
        "value_primary": np.asarray([0.0, 1.4]),
        "policy_primary": np.asarray([-0.2, 0.5]),
        "both_primary": np.asarray([-0.2, 1.1]),
        "value_matched": np.asarray([0.0, 1.3]),
        "policy_matched": np.asarray([-0.2, 0.4]),
    }
    for arm, values in arms.items():
        for grid_index, value in enumerate(values):
            rows.append(
                {
                    "condition_id": "fixture",
                    "input_profile": "unique_1_100",
                    "policy_set_label": "Bubble+Selection",
                    "replicate_ordinal": 1,
                    "scenario_id": "fixture-scenario",
                    "checkpoint_progress": 0.5,
                    "grid_index": grid_index,
                    "common_activation_progress": float(grid_index),
                    "arm": arm,
                    "corrected_aggregation": value,
                    "reference_sortedness": value + 0.5,
                }
            )
    effects = effect_trajectories(pd.DataFrame(rows))
    factorial = effects[
        (effects.metric == "corrected_aggregation")
        & (effects.effect_family == "primary_factorial")
    ].pivot(index="grid_index", columns="component", values="effect")
    assert np.allclose(
        factorial.value_shapley + factorial.policy_shapley, factorial.joint
    )
    expected_interaction = np.asarray([0.0, 0.2])
    assert np.allclose(factorial.interaction, expected_interaction)
