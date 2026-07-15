from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from reference_simulator.engine import evaluate_terminal
from reference_simulator.model import Architecture, Direction, FaultMode, Policy
from src.detours.state_space import FamilySpec, StructuralState
from src.detours.transition_graph import (
    EDGE_ARRAY_COLUMNS,
    classify_states,
    compare_edge_records,
    e01_outgoing_edges,
    edge_batch,
    metric_denominators,
    opportunity_arrays,
    reachable_states,
)


def family(
    policies: tuple[Policy, ...],
    *,
    direction: Direction = Direction.ASCENDING,
    faults: tuple[FaultMode, ...] | None = None,
    architecture: Architecture = Architecture.CELL_VIEW,
) -> FamilySpec:
    n = len(policies)
    return FamilySpec(
        n,
        architecture,
        direction,
        policies,
        faults or (FaultMode.NORMAL,) * n,
    )


def batch_records(spec: FamilySpec) -> tuple[np.ndarray, dict[int, list[dict[str, int]]]]:
    terminal = classify_states(spec)
    arrays = edge_batch(spec, terminal, 0, spec.state_count)
    grouped: dict[int, list[dict[str, int]]] = defaultdict(list)
    for row in range(len(arrays["source_state_ordinal"])):
        record = {column: int(arrays[column][row]) for column in EDGE_ARRAY_COLUMNS}
        grouped[record["source_state_ordinal"]].append(record)
    return terminal, grouped


def assert_exact_against_e01(spec: FamilySpec, state_ordinals: list[int] | None = None) -> None:
    terminal, grouped = batch_records(spec)
    selected = range(spec.state_count) if state_ordinals is None else state_ordinals
    for state_ordinal in selected:
        observed = grouped.get(state_ordinal, [])
        expected = e01_outgoing_edges(spec, state_ordinal)
        matched, detail = compare_edge_records(observed, expected)
        assert matched, (state_ordinal, detail)
        structural = StructuralState(
            spec,
            state_ordinal % spec.occupancy_state_count,
            state_ordinal // spec.occupancy_state_count,
        )
        native = evaluate_terminal(spec.scenario(), structural.to_run_state())
        expected_terminal = 1 if native == "complete" else 2 if native == "quiescent" else 0
        assert int(terminal[state_ordinal]) == expected_terminal


def test_scheduler_projection_handles_faulty_bubble_before_side_draw() -> None:
    spec = family(
        (Policy.BUBBLE, Policy.INSERTION, Policy.BUBBLE, Policy.SELECTION),
        faults=(FaultMode.PASSIVE, FaultMode.NORMAL, FaultMode.NORMAL, FaultMode.NORMAL),
    )
    opportunities = opportunity_arrays(spec)
    assert opportunities["actors"].tolist() == [0, 1, 2, 2, 3]
    assert opportunities["sides"].tolist() == [-1, -1, 0, 1, -1]
    assert sum(
        numerator / denominator
        for numerator, denominator in zip(
            opportunities["probability_numerators"],
            opportunities["probability_denominators"],
        )
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("direction", [Direction.ASCENDING, Direction.DESCENDING])
@pytest.mark.parametrize("policy", [Policy.BUBBLE, Policy.INSERTION])
def test_all_memoryless_n4_states_match_e01(direction: Direction, policy: Policy) -> None:
    assert_exact_against_e01(family((policy,) * 4, direction=direction))


@pytest.mark.parametrize("policy", [Policy.BUBBLE, Policy.INSERTION, Policy.SELECTION])
def test_all_traditional_n4_states_match_e01(policy: Policy) -> None:
    assert_exact_against_e01(
        family((policy,) * 4, architecture=Architecture.TRADITIONAL)
    )


def test_selection_memory_and_mixed_policy_states_match_e01() -> None:
    pure_selection = family((Policy.SELECTION,) * 4)
    # Exhaust the full 15,000-state Selection envelope in the independent oracle.
    assert_exact_against_e01(pure_selection)
    mixed = family(
        (Policy.SELECTION, Policy.BUBBLE, Policy.INSERTION, Policy.SELECTION),
        direction=Direction.DESCENDING,
        faults=(FaultMode.NORMAL, FaultMode.STUCK, FaultMode.PASSIVE, FaultMode.NORMAL),
    )
    selected = sorted({0, 1, 23, 24, 25, mixed.state_count // 2, mixed.state_count - 1})
    assert_exact_against_e01(mixed, selected)


@pytest.mark.parametrize("fault", [FaultMode.PASSIVE, FaultMode.STUCK])
def test_fault_action_and_cost_semantics_match_e01(fault: FaultMode) -> None:
    spec = family(
        (Policy.BUBBLE, Policy.INSERTION, Policy.BUBBLE, Policy.INSERTION),
        faults=(fault, FaultMode.NORMAL, FaultMode.NORMAL, FaultMode.NORMAL),
    )
    assert_exact_against_e01(spec)


def test_edge_key_partition_and_ledger_identities_are_exact() -> None:
    spec = family((Policy.SELECTION, Policy.BUBBLE, Policy.INSERTION, Policy.BUBBLE))
    terminal, grouped = batch_records(spec)
    slots = len(opportunity_arrays(spec)["actors"])
    keys: set[tuple[int, int]] = set()
    for source in range(spec.state_count):
        rows = grouped.get(source, [])
        if terminal[source] == 0:
            assert [row["opportunity_ordinal"] for row in rows] == list(range(slots))
        else:
            assert rows == []
        for row in rows:
            key = (source, row["opportunity_ordinal"])
            assert key not in keys
            keys.add(key)
            assert row["cost_no_ops"] + row["cost_rejections"] + row[
                "cost_memory_updates"
            ] + row["cost_accepted_swaps"] == 1
            assert row["cost_displaced_cells"] == 2 * row["cost_accepted_swaps"]
            assert row["changed"] == int(
                row["cost_memory_updates"] + row["cost_accepted_swaps"] == 1
            )


def test_anchor_reachability_is_direction_specific_and_exact() -> None:
    ascending = family((Policy.BUBBLE,) * 4)
    terminal_ascending = classify_states(ascending)
    assert reachable_states(ascending, terminal_ascending).sum() == 1
    descending = family((Policy.BUBBLE,) * 4, direction=Direction.DESCENDING)
    terminal_descending = classify_states(descending)
    assert reachable_states(descending, terminal_descending).sum() == 24


def test_unique_value_metric_denominators() -> None:
    assert metric_denominators(4) == {
        "normalized_adjacent_descents": 3,
        "paper_sortedness_distance": 4,
        "normalized_kendall_distance": 6,
        "normalized_spearman_footrule": 8,
        "normalized_maximum_rank_error": 3,
        "duplicate_aware_earth_movers_distance": 4,
        "normalized_duplicate_aware_earth_movers_distance": 8,
    }
