"""Exact necessary-detour quantities on frozen S05 structural opportunity graphs.

The primary estimand is existential over finite labelled graph paths.  It is not
a byte-exact replay, scheduler-robust guarantee, or stochastic hitting result.
All metric calculations use exact integer numerators within one graph family.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Mapping, Sequence

import numba as nb
import numpy as np

from reference_simulator.model import Direction
from src.detours.state_space import FamilySpec


SPEC_VERSION = "e03.s06.necessary_detour.v1"
UNREACHABLE = np.iinfo(np.int64).max

CLASS_COMPLETE_START = 0
CLASS_REACHABLE_NO_DETOUR = 1
CLASS_NECESSARY_DETOUR = 2
CLASS_UNREACHABLE_ACTIVE = 3
CLASS_QUIESCENT = 4
CLASS_LABELS = {
    CLASS_COMPLETE_START: "complete_start",
    CLASS_REACHABLE_NO_DETOUR: "reachable_no_detour",
    CLASS_NECESSARY_DETOUR: "necessary_detour",
    CLASS_UNREACHABLE_ACTIVE: "unreachable_active",
    CLASS_QUIESCENT: "quiescent",
}

METRIC_ALIASES = {
    "adjacent_descents": "adjacent_descents",
    "paper_sortedness_distance_numerator": "adjacent_descents",
    "inversion_count": "inversion_count",
    "spearman_footrule": "spearman_footrule",
    "emd_numerator": "spearman_footrule",
    "maximum_rank_error": "maximum_rank_error",
}

COST_PROFILES: dict[str, tuple[str, ...]] = {
    "activations": ("activations",),
    "observation_reads": ("observation_reads", "activations"),
    "value_comparisons": ("value_comparisons", "activations"),
    "accepted_swaps": ("cost_accepted_swaps", "activations"),
    "displaced_cells": ("cost_displaced_cells", "activations"),
    "full_ledger": (
        "activations",
        "observation_reads",
        "value_comparisons",
        "cost_no_ops",
        "cost_rejections",
        "cost_memory_updates",
        "cost_accepted_swaps",
        "cost_displaced_cells",
    ),
}


@dataclass(frozen=True, slots=True)
class PrimarySolution:
    """All-start minimum absolute bottlenecks and deterministic witnesses."""

    bottleneck_levels: np.ndarray
    excursion_levels: np.ndarray
    successor_edges: np.ndarray
    classifications: np.ndarray


@dataclass(frozen=True, slots=True)
class LexicographicWitness:
    """One deterministic secondary-optimal witness at the primary optimum."""

    start: int
    goal: int
    bottleneck_level: int
    excursion_level: int
    cost: tuple[int, ...]
    nodes: tuple[int, ...]
    edges: tuple[int, ...]


def _as_int64_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def validate_graph(
    node_count: int,
    sources: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if node_count < 1:
        raise ValueError("node_count must be positive")
    source = _as_int64_vector(sources, "sources")
    target = _as_int64_vector(targets, "targets")
    if len(source) != len(target):
        raise ValueError("sources and targets must have equal length")
    if len(source) and (
        int(source.min()) < 0
        or int(target.min()) < 0
        or int(source.max()) >= node_count
        or int(target.max()) >= node_count
    ):
        raise ValueError("edge endpoint outside node domain")
    return source, target


def goal_mask_from_terminal(terminal_codes: Sequence[int] | np.ndarray) -> np.ndarray:
    terminal = np.ascontiguousarray(terminal_codes, dtype=np.uint8)
    if terminal.ndim != 1 or np.any(terminal > 2):
        raise ValueError("terminal codes must be a one-dimensional 0/1/2 vector")
    return terminal == 1


def tied_value_goal_mask(
    value_states: Sequence[Sequence[float]] | np.ndarray,
    direction: str | Direction,
) -> np.ndarray:
    """Return the non-strict value-goal set without collapsing identities.

    Each row is one identity-distinct structural state.  Equal-valued identities
    remain separate rows, while every row with sorted values is marked as a goal.
    """

    values = np.asarray(value_states)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("value_states must be a nonempty two-dimensional array")
    selected = Direction(direction)
    if values.shape[1] == 1:
        return np.ones(values.shape[0], dtype=bool)
    if selected == Direction.ASCENDING:
        return np.all(values[:, :-1] <= values[:, 1:], axis=1)
    return np.all(values[:, :-1] >= values[:, 1:], axis=1)


def validate_goal_distance(
    distance_levels: Sequence[int] | np.ndarray,
    goal_mask: Sequence[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    levels = _as_int64_vector(distance_levels, "distance_levels")
    goals = np.ascontiguousarray(goal_mask, dtype=bool)
    if goals.ndim != 1 or len(goals) != len(levels):
        raise ValueError("goal_mask must align with distance_levels")
    if np.any(levels < 0):
        raise ValueError("distance levels must be nonnegative")
    if not np.any(goals):
        raise ValueError("the successful goal set is empty")
    if np.any(levels[goals] != 0):
        raise ValueError("a goal distance must be zero on every successful goal")
    return levels, goals


@nb.njit(cache=True)
def _reverse_csr(
    node_count: int, sources: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(node_count, dtype=np.int64)
    for target in targets:
        counts[target] += 1
    indptr = np.empty(node_count + 1, dtype=np.int64)
    indptr[0] = 0
    for node in range(node_count):
        indptr[node + 1] = indptr[node] + counts[node]
    cursor = indptr[:-1].copy()
    predecessors = np.empty(len(sources), dtype=np.int64)
    edge_indices = np.empty(len(sources), dtype=np.int64)
    for edge in range(len(sources)):
        target = targets[edge]
        index = cursor[target]
        predecessors[index] = sources[edge]
        edge_indices[index] = edge
        cursor[target] += 1
    return indptr, predecessors, edge_indices


@nb.njit(cache=True)
def _bucket_minimax(
    levels: np.ndarray,
    level_ranks: np.ndarray,
    unique_levels: np.ndarray,
    goals: np.ndarray,
    reverse_indptr: np.ndarray,
    predecessors: np.ndarray,
    reverse_edge_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = len(levels)
    best = np.full(node_count, UNREACHABLE, dtype=np.int64)
    successor_edge = np.full(node_count, -1, dtype=np.int64)
    heads = np.full(len(unique_levels), -1, dtype=np.int64)
    next_node = np.full(node_count, -1, dtype=np.int64)

    for node in range(node_count):
        if goals[node]:
            rank = level_ranks[node]
            best[node] = levels[node]
            next_node[node] = heads[rank]
            heads[rank] = node

    for bucket in range(len(unique_levels)):
        while heads[bucket] >= 0:
            node = heads[bucket]
            heads[bucket] = next_node[node]
            for position in range(reverse_indptr[node], reverse_indptr[node + 1]):
                predecessor = predecessors[position]
                if best[predecessor] != UNREACHABLE:
                    continue
                target_bucket = bucket
                if level_ranks[predecessor] > target_bucket:
                    target_bucket = level_ranks[predecessor]
                best[predecessor] = unique_levels[target_bucket]
                successor_edge[predecessor] = reverse_edge_indices[position]
                next_node[predecessor] = heads[target_bucket]
                heads[target_bucket] = predecessor
    return best, successor_edge


def solve_primary_all_starts(
    node_count: int,
    sources: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    distance_levels: Sequence[int] | np.ndarray,
    terminal_codes: Sequence[int] | np.ndarray,
) -> PrimarySolution:
    """Solve the existential minimum-peak excursion for every graph start.

    Complete nodes are the only goals.  Quiescent and active goal-unreachable
    starts retain an undefined excursion sentinel rather than being classified
    by a vacuous universal statement.
    """

    source, target = validate_graph(node_count, sources, targets)
    terminal = np.ascontiguousarray(terminal_codes, dtype=np.uint8)
    if terminal.ndim != 1 or len(terminal) != node_count or np.any(terminal > 2):
        raise ValueError("terminal_codes must align with the node domain")
    if len(source) and np.any(terminal[source] != 0):
        raise ValueError("complete and quiescent terminal nodes must have outdegree zero")
    levels, goals = validate_goal_distance(distance_levels, terminal == 1)
    unique_levels, level_ranks = np.unique(levels, return_inverse=True)
    level_ranks = np.ascontiguousarray(level_ranks, dtype=np.int64)
    reverse_indptr, predecessors, reverse_edges = _reverse_csr(
        node_count, source, target
    )
    bottleneck, successor_edges = _bucket_minimax(
        levels,
        level_ranks,
        np.ascontiguousarray(unique_levels, dtype=np.int64),
        np.ascontiguousarray(goals, dtype=np.bool_),
        reverse_indptr,
        predecessors,
        reverse_edges,
    )
    excursion = np.full(node_count, UNREACHABLE, dtype=np.int64)
    classification = np.empty(node_count, dtype=np.uint8)
    for node in range(node_count):
        if terminal[node] == 1:
            classification[node] = CLASS_COMPLETE_START
            excursion[node] = 0
        elif terminal[node] == 2:
            classification[node] = CLASS_QUIESCENT
        elif bottleneck[node] == UNREACHABLE:
            classification[node] = CLASS_UNREACHABLE_ACTIVE
        else:
            value = int(bottleneck[node] - levels[node])
            if value < 0:
                raise AssertionError("absolute bottleneck fell below start distance")
            excursion[node] = value
            classification[node] = (
                CLASS_NECESSARY_DETOUR if value > 0 else CLASS_REACHABLE_NO_DETOUR
            )
    return PrimarySolution(bottleneck, excursion, successor_edges, classification)


def reconstruct_primary_witness(
    start: int,
    sources: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    terminal_codes: Sequence[int] | np.ndarray,
    solution: PrimarySolution,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source = _as_int64_vector(sources, "sources")
    target = _as_int64_vector(targets, "targets")
    terminal = np.ascontiguousarray(terminal_codes, dtype=np.uint8)
    if not 0 <= start < len(solution.bottleneck_levels):
        raise ValueError("start outside node domain")
    if solution.bottleneck_levels[start] == UNREACHABLE:
        raise ValueError("start has no successful path")
    nodes = [start]
    edges: list[int] = []
    current = start
    seen: set[int] = set()
    while terminal[current] != 1:
        if current in seen:
            raise AssertionError("primary witness contains a cycle")
        seen.add(current)
        edge = int(solution.successor_edges[current])
        if edge < 0 or edge >= len(source) or int(source[edge]) != current:
            raise AssertionError("invalid primary successor edge")
        edges.append(edge)
        current = int(target[edge])
        nodes.append(current)
        if len(nodes) > len(solution.bottleneck_levels) + 1:
            raise AssertionError("primary witness exceeds simple-path bound")
    return tuple(nodes), tuple(edges)


def cost_matrix_from_s05(
    edge_columns: Mapping[str, Sequence[int] | np.ndarray], profile: str
) -> np.ndarray:
    if profile not in COST_PROFILES:
        raise ValueError(f"unsupported cost profile {profile!r}")
    lengths = {len(value) for value in edge_columns.values()}
    if len(lengths) != 1:
        raise ValueError("edge cost columns have inconsistent lengths")
    edge_count = lengths.pop() if lengths else 0
    columns: list[np.ndarray] = []
    for field in COST_PROFILES[profile]:
        if field == "activations":
            columns.append(np.ones(edge_count, dtype=np.int64))
        else:
            if field not in edge_columns:
                raise ValueError(f"missing S05 cost field {field}")
            values = _as_int64_vector(edge_columns[field], field)
            if np.any(values < 0):
                raise ValueError("secondary edge costs must be nonnegative")
            columns.append(values)
    result = np.column_stack(columns) if columns else np.empty((edge_count, 0), dtype=np.int64)
    if edge_count and np.any(np.all(result == 0, axis=1)):
        raise ValueError("every edge must have positive cost in at least one coordinate")
    return np.ascontiguousarray(result, dtype=np.int64)


def _forward_csr(
    node_count: int, sources: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(sources, minlength=node_count).astype(np.int64)
    indptr = np.empty(node_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    cursor = indptr[:-1].copy()
    edge_order = np.empty(len(sources), dtype=np.int64)
    for edge, source in enumerate(sources):
        position = cursor[source]
        edge_order[position] = edge
        cursor[source] += 1
    return indptr, edge_order


def solve_lexicographic_witness(
    start: int,
    node_count: int,
    sources: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    distance_levels: Sequence[int] | np.ndarray,
    terminal_codes: Sequence[int] | np.ndarray,
    edge_costs: Sequence[Sequence[int]] | np.ndarray,
    primary_solution: PrimarySolution | None = None,
) -> LexicographicWitness:
    """Minimize additive lexicographic cost after fixing the primary peak."""

    source, target = validate_graph(node_count, sources, targets)
    levels = _as_int64_vector(distance_levels, "distance_levels")
    terminal = np.ascontiguousarray(terminal_codes, dtype=np.uint8)
    costs = np.ascontiguousarray(edge_costs, dtype=np.int64)
    if len(levels) != node_count or len(terminal) != node_count:
        raise ValueError("node vectors must align with node_count")
    if costs.ndim != 2 or costs.shape[0] != len(source) or costs.shape[1] < 1:
        raise ValueError("edge_costs must have one or more columns and align with edges")
    if np.any(costs < 0) or (len(costs) and np.any(np.all(costs == 0, axis=1))):
        raise ValueError("edge costs must be nonnegative and lexicographically positive")
    solution = primary_solution or solve_primary_all_starts(
        node_count, source, target, levels, terminal
    )
    if not 0 <= start < node_count:
        raise ValueError("start outside node domain")
    if solution.bottleneck_levels[start] == UNREACHABLE:
        raise ValueError("start has no successful path")
    threshold = int(solution.bottleneck_levels[start])
    width = costs.shape[1]
    zero = (0,) * width
    labels: list[tuple[int, ...] | None] = [None] * node_count
    predecessor_edge = np.full(node_count, -1, dtype=np.int64)
    labels[start] = zero
    heap: list[tuple[tuple[int, ...], int]] = [(zero, start)]
    indptr, edge_order = _forward_csr(node_count, source)
    goal = -1
    while heap:
        cost, node = heapq.heappop(heap)
        if labels[node] != cost:
            continue
        if terminal[node] == 1:
            goal = node
            break
        for position in range(indptr[node], indptr[node + 1]):
            edge = int(edge_order[position])
            successor = int(target[edge])
            if levels[successor] > threshold:
                continue
            candidate = tuple(cost[index] + int(costs[edge, index]) for index in range(width))
            if labels[successor] is None or candidate < labels[successor]:
                labels[successor] = candidate
                predecessor_edge[successor] = edge
                heapq.heappush(heap, (candidate, successor))
    if goal < 0:
        raise AssertionError("primary sublevel set contains no secondary witness")
    reverse_nodes = [goal]
    reverse_edges: list[int] = []
    current = goal
    while current != start:
        edge = int(predecessor_edge[current])
        if edge < 0 or int(target[edge]) != current:
            raise AssertionError("invalid secondary predecessor")
        reverse_edges.append(edge)
        current = int(source[edge])
        reverse_nodes.append(current)
        if len(reverse_nodes) > node_count + 1:
            raise AssertionError("secondary witness exceeds simple-path bound")
    nodes = tuple(reversed(reverse_nodes))
    edges = tuple(reversed(reverse_edges))
    return LexicographicWitness(
        start=start,
        goal=goal,
        bottleneck_level=threshold,
        excursion_level=threshold - int(levels[start]),
        cost=labels[goal] or zero,
        nodes=nodes,
        edges=edges,
    )


@nb.njit(cache=True)
def _all_occupancy_metric_levels(n: int, descending: bool) -> np.ndarray:
    factorial = 1
    for value in range(2, n + 1):
        factorial *= value
    result = np.empty((4, factorial), dtype=np.int64)
    facts = np.empty(n + 1, dtype=np.int64)
    facts[0] = 1
    for value in range(1, n + 1):
        facts[value] = facts[value - 1] * value
    occupancy = np.empty(n, dtype=np.int64)
    available = np.empty(n, dtype=np.int64)
    for rank in range(factorial):
        for value in range(n):
            available[value] = value
        count = n
        remainder = rank
        for position in range(n):
            factor = facts[n - position - 1]
            choice = remainder // factor
            remainder %= factor
            occupancy[position] = available[choice]
            for shift in range(choice, count - 1):
                available[shift] = available[shift + 1]
            count -= 1
        adjacent = 0
        inversions = 0
        footrule = 0
        maximum_rank = 0
        for position in range(n):
            identity = occupancy[position]
            target = n - 1 - identity if descending else identity
            displacement = abs(position - target)
            footrule += displacement
            maximum_rank = max(maximum_rank, displacement)
            if position + 1 < n:
                adjacent += int(
                    occupancy[position] < occupancy[position + 1]
                    if descending
                    else occupancy[position] > occupancy[position + 1]
                )
            for right in range(position + 1, n):
                inversions += int(
                    identity < occupancy[right]
                    if descending
                    else identity > occupancy[right]
                )
        result[0, rank] = adjacent
        result[1, rank] = inversions
        result[2, rank] = footrule
        result[3, rank] = maximum_rank
    return result


def family_metric_levels(family: FamilySpec, metric: str) -> np.ndarray:
    """Return exact S01/S05 integer metric levels for every structural node."""

    if metric not in METRIC_ALIASES:
        raise ValueError(f"unsupported S05 metric {metric!r}")
    canonical = METRIC_ALIASES[metric]
    row = {
        "adjacent_descents": 0,
        "inversion_count": 1,
        "spearman_footrule": 2,
        "maximum_rank_error": 3,
    }[canonical]
    occupancy_levels = _all_occupancy_metric_levels(
        family.n, family.direction == Direction.DESCENDING
    )[row]
    return np.tile(occupancy_levels, family.cursor_state_count)


def estimated_primary_working_bytes(node_count: int, edge_count: int) -> int:
    """Conservative family-at-a-time peak excluding source Parquet buffering."""

    if node_count < 1 or edge_count < 0:
        raise ValueError("invalid graph size")
    # source/target, reverse predecessor/edge, CSR/cursors, levels/ranks/best,
    # successor/linked buckets/terminal plus conservative 25% allocator headroom.
    exact = edge_count * (8 + 8 + 8 + 8) + node_count * (8 * 8 + 2) + 16 * (node_count + 1)
    return math.ceil(exact * 1.25)
