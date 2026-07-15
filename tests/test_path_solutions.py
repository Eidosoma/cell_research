from __future__ import annotations

from collections.abc import Iterator
import itertools
import random

import numpy as np
import pytest

from src.detours.necessary_detour import UNREACHABLE
from src.detours.path_solutions import (
    PROFILE_ACCEPTED_SWAPS,
    PROFILE_ACTIVATIONS,
    PROFILE_FULL_LEDGER,
    PROFILE_OBSERVATION_READS,
    PROFILE_VALUE_COMPARISONS,
    reconstruct_successor_witness,
    serialized_labels,
    solve_all_starts_lexicographic,
)


def columns(costs: list[tuple[int, int, int, int, int, int, int]]) -> dict[str, np.ndarray]:
    names = (
        "observation_reads",
        "value_comparisons",
        "cost_no_ops",
        "cost_rejections",
        "cost_memory_updates",
        "cost_accepted_swaps",
        "cost_displaced_cells",
    )
    return {
        name: np.asarray([row[index] for row in costs], dtype=np.int64)
        for index, name in enumerate(names)
    }


def solve(profile: int, edges: list[tuple[int, int]], terminal: list[int], costs):
    source = np.asarray([edge[0] for edge in edges], dtype=np.int64)
    target = np.asarray([edge[1] for edge in edges], dtype=np.int64)
    result = solve_all_starts_lexicographic(
        len(terminal), source, target, terminal, columns(costs), profile
    )
    return source, target, result


def test_shortest_all_starts_and_failure_sentinels() -> None:
    edges = [(0, 1), (0, 2), (2, 3), (1, 3), (4, 4)]
    costs = [(0, 0, 1, 0, 0, 0, 0)] * len(edges)
    source, target, result = solve(
        PROFILE_ACTIVATIONS, edges, [0, 0, 0, 1, 0, 2], costs
    )
    assert result.labels[0].tolist() == [2, 1, 1, 0, UNREACHABLE, UNREACHABLE]
    assert reconstruct_successor_witness(
        0, source, target, [0, 0, 0, 1, 0, 2], result.successor_edges
    ) == ((0, 1, 3), (0, 3))
    assert serialized_labels(result.labels)[0].tolist() == [2, 1, 1, 0, -1, -1]


def test_parallel_edges_use_lexicographic_cost_and_physical_tie_break() -> None:
    edges = [(0, 1), (0, 1), (0, 1)]
    costs = [
        (5, 4, 0, 0, 0, 0, 0),
        (2, 8, 0, 0, 0, 0, 0),
        (2, 8, 0, 0, 0, 0, 0),
    ]
    source, target, result = solve(
        PROFILE_OBSERVATION_READS, edges, [0, 1], costs
    )
    assert result.labels[:, 0].tolist() == [2, 1]
    assert result.successor_edges[0] == 1
    assert reconstruct_successor_witness(
        0, source, target, [0, 1], result.successor_edges
    )[1] == (1,)


def test_edge_filter_and_full_ledger_order() -> None:
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    costs = [
        (0, 0, 0, 0, 1, 0, 0),
        (0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0),
        (0, 0, 1, 0, 0, 0, 0),
    ]
    source = np.asarray([x for x, _ in edges])
    target = np.asarray([y for _, y in edges])
    result = solve_all_starts_lexicographic(
        4,
        source,
        target,
        [0, 0, 0, 1],
        columns(costs),
        PROFILE_FULL_LEDGER,
        edge_allowed=[False, True, True, True],
    )
    assert result.labels[:, 0].tolist() == [2, 0, 0, 2, 0, 0, 0, 0]
    assert result.successor_edges[0] == 1


def _simple_paths(
    start: int,
    edges: list[tuple[int, int]],
    goals: set[int],
    node_count: int,
) -> Iterator[tuple[int, ...]]:
    outgoing = [[] for _ in range(node_count)]
    for edge, (source, _) in enumerate(edges):
        outgoing[source].append(edge)
    stack = [(start, (), frozenset({start}))]
    while stack:
        node, path, seen = stack.pop()
        if node in goals:
            yield path
            continue
        for edge in outgoing[node]:
            target = edges[edge][1]
            if target not in seen:
                stack.append((target, path + (edge,), seen | {target}))


@pytest.mark.parametrize("seed", list(range(16)))
def test_all_profiles_match_independent_simple_path_bruteforce(seed: int) -> None:
    rng = random.Random(7007 + seed)
    node_count = rng.randint(3, 7)
    terminal = [0] * node_count
    terminal[-1] = 1
    edges: list[tuple[int, int]] = []
    for source, target in itertools.product(range(node_count - 1), range(node_count)):
        if rng.random() < 0.27:
            edges.append((source, target))
            if rng.random() < 0.15:
                edges.append((source, target))
    cost_rows = []
    for _ in edges:
        swaps = rng.randint(0, 1)
        cost_rows.append(
            (
                rng.randint(0, 4),
                rng.randint(0, 3),
                rng.randint(0, 1),
                rng.randint(0, 1),
                rng.randint(0, 1),
                swaps,
                2 * swaps,
            )
        )
    source = np.asarray([edge[0] for edge in edges], dtype=np.int64)
    target = np.asarray([edge[1] for edge in edges], dtype=np.int64)
    edge_columns = columns(cost_rows)
    profiles = {
        PROFILE_ACTIVATIONS: lambda row: (1,),
        PROFILE_OBSERVATION_READS: lambda row: (row[0], 1),
        PROFILE_VALUE_COMPARISONS: lambda row: (row[1], 1),
        PROFILE_ACCEPTED_SWAPS: lambda row: (row[5], 1),
        PROFILE_FULL_LEDGER: lambda row: (1, *row),
    }
    for profile, cost_fn in profiles.items():
        result = solve_all_starts_lexicographic(
            node_count, source, target, terminal, edge_columns, profile
        )
        for start in range(node_count):
            candidates = []
            for path in _simple_paths(start, edges, {node_count - 1}, node_count):
                if path:
                    width = len(cost_fn(cost_rows[path[0]]))
                    label = tuple(
                        sum(cost_fn(cost_rows[edge])[coordinate] for edge in path)
                        for coordinate in range(width)
                    )
                else:
                    label = (0,) * result.labels.shape[0]
                candidates.append(label)
            if not candidates:
                assert result.labels[0, start] == UNREACHABLE
            else:
                assert tuple(result.labels[:, start]) == min(candidates)
