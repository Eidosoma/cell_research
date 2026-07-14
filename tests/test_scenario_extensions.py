from __future__ import annotations

from collections import Counter

import pytest

from causal_simulator.scenario_extensions import (
    FORBIDDEN_CONFIRMATORY_PLACEMENT_CLASS,
    PRESPECIFICATION_SHA256,
    STRUCTURAL_PLACEMENT_CLASSES,
    InputScenario,
    OrderStructure,
    ScenarioSplit,
    ValueProfile,
    assign_trace_selection,
    brute_force_duplicate_aware_footrule,
    generate_unique_input_scenario,
    maximum_strict_unequal_inversions,
    minimum_duplicate_aware_footrule,
    scenario_from_record,
    strict_unequal_inversions,
    trace_selection_count,
    value_counts,
)


@pytest.mark.parametrize("n", (20, 50, 100, 200, 500))
def test_exact_value_multiplicity_profiles(n: int) -> None:
    unique = value_counts(n, ValueProfile.UNIQUE)
    balanced = value_counts(n, ValueProfile.BALANCED_DUPLICATE)
    uneven = value_counts(n, ValueProfile.UNEVEN_DUPLICATE)
    assert unique == (1,) * n
    assert balanced == (n // 10,) * 10
    assert len(uneven) == 10 and sum(uneven) == n and min(uneven) >= 1
    assert len(set(uneven)) > 1


@pytest.mark.parametrize(
    "values,expected,maximum",
    (
        ((2, 1, 1), 2, 2),
        ((2, 2, 1, 1), 4, 4),
        ((1, 2, 1, 2), 1, 4),
        ((1, 1, 1), 0, 0),
    ),
)
def test_strict_duplicate_aware_inversion_fixtures(values, expected, maximum) -> None:
    assert strict_unequal_inversions(values) == expected
    assert maximum_strict_unequal_inversions(values) == maximum


@pytest.mark.parametrize(
    "values",
    (
        (2, 1, 2, 1),
        (3, 1, 2, 1, 3),
        (1, 1, 2, 3, 2, 1),
        (4, 2, 4, 2, 1, 1),
    ),
)
@pytest.mark.parametrize("descending", (False, True))
def test_duplicate_aware_footrule_matches_brute_force(values, descending: bool) -> None:
    assert minimum_duplicate_aware_footrule(
        values, descending=descending
    ) == brute_force_duplicate_aware_footrule(values, descending=descending)


@pytest.mark.parametrize("n", (20, 50, 100, 200, 500))
@pytest.mark.parametrize("profile", tuple(ValueProfile))
@pytest.mark.parametrize("structure", tuple(OrderStructure))
def test_generation_contracts_and_exact_reconstruction(n, profile, structure) -> None:
    seen: set[str] = set()
    scenario = generate_unique_input_scenario(
        ScenarioSplit.RUNTIME_VALIDATION,
        n,
        profile,
        structure,
        0,
        seen_initial_value_hashes=seen,
    )
    repeat = InputScenario.generate(
        ScenarioSplit.RUNTIME_VALIDATION,
        n,
        profile,
        structure,
        0,
        generation_attempt=scenario.generation_attempt,
    )
    assert scenario == repeat
    assert len(scenario.initial_values) == len(scenario.initial_occupancy_indices) == n
    assert sorted(scenario.initial_occupancy_indices) == list(range(n))
    assert all(scenario.target_feasibility().values())
    assert scenario.core_record()["outcomeAccess"] == "none"
    assert scenario.core_record()["prespecificationSha256"] == PRESPECIFICATION_SHA256
    assert scenario_from_record(scenario.core_record()) == scenario
    normalized = scenario.descriptors[
        "normalizedStrictUnequalInversionsAscending"
    ]
    if structure == OrderStructure.NEARLY_SORTED:
        assert 0 < normalized <= 0.15
    elif structure == OrderStructure.REVERSE:
        assert normalized == 1
    else:
        assert 0.15 <= normalized <= 0.85


def test_split_protection_trace_quota_and_confirmatory_fail_closed_fields() -> None:
    rows = [
        InputScenario.generate(
            ScenarioSplit.SCREENING_POOL,
            20,
            ValueProfile.BALANCED_DUPLICATE,
            OrderStructure.REVERSE,
            replicate,
        )
        for replicate in range(20)
    ]
    selected = assign_trace_selection(rows)
    # The function uses the frozen full-split quota; with a deliberately partial
    # test group every supplied row is still assigned deterministically.
    assert selected == assign_trace_selection(tuple(reversed(rows)))
    assert trace_selection_count(ScenarioSplit.SCREENING_POOL, 20) == 13
    record = rows[0].core_record()
    assert not record["protected"]
    assert record["confirmatoryEligiblePlacementClasses"] == list(
        STRUCTURAL_PLACEMENT_CLASSES
    )
    assert record["forbiddenConfirmatoryPlacementClass"] == (
        FORBIDDEN_CONFIRMATORY_PLACEMENT_CLASS
    )

def test_unique_sequences_across_semantic_splits_for_same_factorial_cell() -> None:
    seen: set[str] = set()
    scenarios = []
    for split in ScenarioSplit:
        for replicate in range(4):
            scenarios.append(
                generate_unique_input_scenario(
                    split,
                    20,
                    ValueProfile.UNEVEN_DUPLICATE,
                    OrderStructure.BLOCK_SCRAMBLED,
                    replicate,
                    seen_initial_value_hashes=seen,
                )
            )
    assert len(seen) == len(scenarios)
    assert len({item.input_scenario_id for item in scenarios}) == len(scenarios)
    assert not (
        {
            item.initial_values_sha256
            for item in scenarios
            if item.split == ScenarioSplit.CONFIRMATORY_HOLDOUT
        }
        & {
            item.initial_values_sha256
            for item in scenarios
            if item.split != ScenarioSplit.CONFIRMATORY_HOLDOUT
        }
    )
    assert Counter(item.split for item in scenarios) == Counter(
        {
            ScenarioSplit.SCREENING_POOL: 4,
            ScenarioSplit.CONFIRMATORY_HOLDOUT: 4,
            ScenarioSplit.RUNTIME_VALIDATION: 4,
        }
    )
