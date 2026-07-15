from __future__ import annotations

from collections.abc import Iterator
import itertools
import random

import numpy as np
import pytest

from reference_simulator.model import Architecture, Direction, FaultMode, Policy
from src.detours.necessary_detour import (
    CLASS_COMPLETE_START,
    CLASS_NECESSARY_DETOUR,
    CLASS_QUIESCENT,
    CLASS_REACHABLE_NO_DETOUR,
    CLASS_UNREACHABLE_ACTIVE,
    COST_PROFILES,
    UNREACHABLE,
    cost_matrix_from_s05,
    family_metric_levels,
    reconstruct_primary_witness,
    solve_lexicographic_witness,
    solve_primary_all_starts,
    tied_value_goal_mask,
    validate_goal_distance,
)
from src.detours.state_space import FamilySpec


def solve(
    levels: list[int],
    edges: list[tuple[int, int]],
    terminal: list[int],
):
    source = np.asarray([edge[0] for edge in edges], dtype=np.int64)
    target = np.asarray([edge[1] for edge in edges], dtype=np.int64)
    return source, target, solve_primary_all_starts(
        len(levels), source, target, levels, terminal
    )


def test_complete_start_uses_zero_length_success() -> None:
    source, target, result = solve([0], [], [1])
    assert result.bottleneck_levels.tolist() == [0]
    assert result.excursion_levels.tolist() == [0]
    assert result.classifications.tolist() == [CLASS_COMPLETE_START]
    assert reconstruct_primary_witness(0, source, target, [1], result) == ((0,), ())


def test_monotone_path_has_zero_excursion() -> None:
    source, target, result = solve(
        [3, 2, 1, 0], [(0, 1), (1, 2), (2, 3)], [0, 0, 0, 1]
    )
    assert result.bottleneck_levels.tolist() == [3, 2, 1, 0]
    assert result.excursion_levels.tolist() == [0, 0, 0, 0]
    assert result.classifications.tolist() == [
        CLASS_REACHABLE_NO_DETOUR,
        CLASS_REACHABLE_NO_DETOUR,
        CLASS_REACHABLE_NO_DETOUR,
        CLASS_COMPLETE_START,
    ]
    assert reconstruct_primary_witness(0, source, target, [0, 0, 0, 1], result) == (
        (0, 1, 2, 3),
        (0, 1, 2),
    )


def test_required_and_avoidable_detours_are_distinct() -> None:
    _, _, required = solve([2, 4, 0], [(0, 1), (1, 2)], [0, 0, 1])
    assert required.bottleneck_levels[0] == 4
    assert required.excursion_levels[0] == 2
    assert required.classifications[0] == CLASS_NECESSARY_DETOUR

    _, _, avoidable = solve(
        [2, 5, 2, 0], [(0, 1), (1, 3), (0, 2), (2, 3)], [0, 0, 0, 1]
    )
    assert avoidable.bottleneck_levels[0] == 2
    assert avoidable.excursion_levels[0] == 0
    assert avoidable.classifications[0] == CLASS_REACHABLE_NO_DETOUR


def test_unreachable_and_quiescent_are_not_vacuous_detours() -> None:
    _, _, result = solve(
        [1, 2, 0, 0], [(0, 1), (1, 0)], [0, 0, 1, 2]
    )
    assert result.bottleneck_levels[0] == UNREACHABLE
    assert result.excursion_levels[0] == UNREACHABLE
    assert result.classifications[0] == CLASS_UNREACHABLE_ACTIVE
    assert result.classifications[3] == CLASS_QUIESCENT
    with pytest.raises(ValueError, match="no successful path"):
        reconstruct_primary_witness(0, [0, 1], [1, 0], [0, 0, 1, 2], result)


def test_tied_identity_goals_are_a_set_and_proxy_must_be_goal_compatible() -> None:
    states = np.asarray(
        [[1, 1, 2], [1, 1, 2], [1, 2, 1], [2, 1, 1]], dtype=np.int64
    )
    goals = tied_value_goal_mask(states, Direction.ASCENDING)
    assert goals.tolist() == [True, True, False, False]
    validate_goal_distance([0, 0, 1, 2], goals)
    # Paper strict Sortedness distance is nonzero on tied sorted goals.
    with pytest.raises(ValueError, match="zero on every successful goal"):
        validate_goal_distance([1, 1, 1, 2], goals)


def test_self_loops_do_not_change_primary_and_parallel_edges_choose_cost() -> None:
    levels = [1, 0]
    terminal = [0, 1]
    edges = [(0, 0), (0, 1), (0, 1)]
    source, target, primary = solve(levels, edges, terminal)
    assert primary.excursion_levels[0] == 0
    costs = np.asarray([[0, 1], [5, 1], [2, 1]], dtype=np.int64)
    witness = solve_lexicographic_witness(
        0, 2, source, target, levels, terminal, costs, primary
    )
    assert witness.edges == (2,)
    assert witness.cost == (2, 1)


def test_secondary_solver_is_two_stage_not_combined_single_label() -> None:
    # At merge node 3, path A has (peak=1,cost=100), while path B has
    # (peak=2,cost=1).  Both later cross level 3, so path B is the true
    # lexicographic secondary optimum at the common primary peak.
    levels = [0, 1, 2, 0, 3, 0]
    terminal = [0, 0, 0, 0, 0, 1]
    edges = [(0, 1), (1, 3), (0, 2), (2, 3), (3, 4), (4, 5)]
    source, target, primary = solve(levels, edges, terminal)
    costs = np.asarray(
        [[99, 1], [1, 1], [0, 1], [1, 1], [0, 1], [0, 1]], dtype=np.int64
    )
    witness = solve_lexicographic_witness(
        0, len(levels), source, target, levels, terminal, costs, primary
    )
    assert witness.bottleneck_level == 3
    assert witness.edges == (2, 3, 4, 5)
    assert witness.cost == (1, 4)


def _simple_paths(
    start: int,
    edges: list[tuple[int, int]],
    goals: set[int],
    node_count: int,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    outgoing: list[list[int]] = [[] for _ in range(node_count)]
    for index, (source, _) in enumerate(edges):
        outgoing[source].append(index)
    stack = [(start, (start,), (), frozenset({start}))]
    while stack:
        node, nodes, path_edges, seen = stack.pop()
        if node in goals:
            yield nodes, path_edges
            continue
        for edge in reversed(outgoing[node]):
            target = edges[edge][1]
            if target in seen:
                continue
            stack.append(
                (target, nodes + (target,), path_edges + (edge,), seen | {target})
            )


def brute_force(
    start: int,
    levels: list[int],
    edges: list[tuple[int, int]],
    terminal: list[int],
    costs: np.ndarray,
) -> tuple[int, tuple[int, ...]] | None:
    goals = {node for node, code in enumerate(terminal) if code == 1}
    candidates: list[tuple[int, tuple[int, ...]]] = []
    for nodes, path_edges in _simple_paths(start, edges, goals, len(levels)):
        peak = max(levels[node] for node in nodes)
        cost = tuple(
            int(costs[list(path_edges), col].sum()) if path_edges else 0
            for col in range(costs.shape[1])
        )
        candidates.append((peak, cost))
    return min(candidates) if candidates else None


@pytest.mark.parametrize("seed", list(range(20)))
def test_tiny_random_multigraphs_match_independent_simple_path_bruteforce(seed: int) -> None:
    rng = random.Random(91_007 + seed)
    node_count = rng.randint(3, 7)
    goal = node_count - 1
    levels = [rng.randint(0, 4) for _ in range(node_count)]
    levels[goal] = 0
    terminal = [0] * node_count
    terminal[goal] = 1
    edges: list[tuple[int, int]] = []
    for source, target in itertools.product(range(node_count - 1), range(node_count)):
        if rng.random() < 0.28:
            edges.append((source, target))
            if rng.random() < 0.18:
                edges.append((source, target))
    costs = np.asarray(
        [[rng.randint(0, 3), 1] for _ in edges], dtype=np.int64
    ).reshape((-1, 2))
    source = np.asarray([edge[0] for edge in edges], dtype=np.int64)
    target = np.asarray([edge[1] for edge in edges], dtype=np.int64)
    primary = solve_primary_all_starts(
        node_count, source, target, levels, terminal
    )
    for start in range(node_count):
        expected = brute_force(start, levels, edges, terminal, costs)
        if expected is None:
            assert primary.bottleneck_levels[start] == UNREACHABLE
            continue
        peak, cost = expected
        assert primary.bottleneck_levels[start] == peak
        witness = solve_lexicographic_witness(
            start,
            node_count,
            source,
            target,
            levels,
            terminal,
            costs,
            primary,
        )
        assert witness.bottleneck_level == peak
        assert witness.cost == cost
        assert max(levels[node] for node in witness.nodes) == peak
        observed = tuple(
            int(costs[list(witness.edges), column].sum()) if witness.edges else 0
            for column in range(costs.shape[1])
        )
        assert observed == cost


def test_cost_profiles_include_positive_activation_tiebreak() -> None:
    fields = {
        "observation_reads": np.asarray([0, 2]),
        "value_comparisons": np.asarray([0, 1]),
        "cost_no_ops": np.asarray([1, 0]),
        "cost_rejections": np.asarray([0, 0]),
        "cost_memory_updates": np.asarray([0, 0]),
        "cost_accepted_swaps": np.asarray([0, 1]),
        "cost_displaced_cells": np.asarray([0, 2]),
    }
    for profile in COST_PROFILES:
        matrix = cost_matrix_from_s05(fields, profile)
        assert matrix.shape[0] == 2
        assert np.all(np.any(matrix > 0, axis=1))


def test_family_metric_levels_repeat_over_selection_memory_and_zero_at_goal() -> None:
    family = FamilySpec(
        n=4,
        architecture=Architecture.CELL_VIEW,
        direction=Direction.ASCENDING,
        policies=(Policy.SELECTION,) * 4,
        faults=(FaultMode.NORMAL,) * 4,
    )
    levels = family_metric_levels(family, "inversion_count")
    assert len(levels) == family.state_count == 15_000
    assert np.array_equal(levels[:24], levels[24:48])
    assert np.count_nonzero(levels == 0) == family.cursor_state_count
    assert levels.max() == 6


def test_terminal_outgoing_edges_are_rejected() -> None:
    with pytest.raises(ValueError, match="outdegree zero"):
        solve_primary_all_starts(2, [0], [1], [0, 1], [1, 0])
