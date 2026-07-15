"""All-start exact path solutions on frozen S05 directed multigraphs.

The graph quantification remains existential over finite structural opportunity
paths.  These routines do not define a stochastic scheduler or byte-exact E01
replay.  Lexicographic all-start labels use a deterministic reverse Dijkstra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numba as nb
import numpy as np

from src.detours.necessary_detour import UNREACHABLE, validate_graph


SPEC_VERSION = "e03.s07.path_solutions.v1"

PROFILE_ACTIVATIONS = 0
PROFILE_OBSERVATION_READS = 1
PROFILE_VALUE_COMPARISONS = 2
PROFILE_ACCEPTED_SWAPS = 3
PROFILE_FULL_LEDGER = 4

PROFILE_NAMES = {
    PROFILE_ACTIVATIONS: "activations",
    PROFILE_OBSERVATION_READS: "observation_reads",
    PROFILE_VALUE_COMPARISONS: "value_comparisons",
    PROFILE_ACCEPTED_SWAPS: "accepted_swaps",
    PROFILE_FULL_LEDGER: "full_ledger",
}
PROFILE_WIDTHS = {
    PROFILE_ACTIVATIONS: 1,
    PROFILE_OBSERVATION_READS: 2,
    PROFILE_VALUE_COMPARISONS: 2,
    PROFILE_ACCEPTED_SWAPS: 2,
    PROFILE_FULL_LEDGER: 8,
}


@dataclass(frozen=True, slots=True)
class AllStartLexicographicSolution:
    """Minimum cost labels and one deterministic successor edge per start."""

    labels: np.ndarray
    successor_edges: np.ndarray


def _int64(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def s05_cost_arrays(
    edge_columns: Mapping[str, Sequence[int] | np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Return exact cost columns in the order used by compiled solvers."""

    names = (
        "observation_reads",
        "value_comparisons",
        "cost_no_ops",
        "cost_rejections",
        "cost_memory_updates",
        "cost_accepted_swaps",
        "cost_displaced_cells",
    )
    missing = [name for name in names if name not in edge_columns]
    if missing:
        raise ValueError(f"missing S05 cost columns: {missing}")
    result = tuple(_int64(edge_columns[name], name) for name in names)
    lengths = {len(values) for values in result}
    if len(lengths) != 1:
        raise ValueError("S05 cost columns have inconsistent lengths")
    if any(np.any(values < 0) for values in result):
        raise ValueError("S05 edge costs must be nonnegative")
    if np.any(result[6] != 2 * result[5]):
        raise ValueError("displaced-cell cost must equal twice accepted swaps")
    return result


@nb.njit(cache=True)
def reverse_csr(
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
        position = cursor[target]
        predecessors[position] = sources[edge]
        edge_indices[position] = edge
        cursor[target] += 1
    return indptr, predecessors, edge_indices


@nb.njit(cache=True)
def _profile_cost(
    profile: int,
    coordinate: int,
    edge: int,
    reads: np.ndarray,
    comparisons: np.ndarray,
    no_ops: np.ndarray,
    rejections: np.ndarray,
    memory_updates: np.ndarray,
    accepted_swaps: np.ndarray,
    displaced_cells: np.ndarray,
) -> int:
    if profile == PROFILE_ACTIVATIONS:
        return 1
    if profile == PROFILE_OBSERVATION_READS:
        return reads[edge] if coordinate == 0 else 1
    if profile == PROFILE_VALUE_COMPARISONS:
        return comparisons[edge] if coordinate == 0 else 1
    if profile == PROFILE_ACCEPTED_SWAPS:
        return accepted_swaps[edge] if coordinate == 0 else 1
    if coordinate == 0:
        return 1
    if coordinate == 1:
        return reads[edge]
    if coordinate == 2:
        return comparisons[edge]
    if coordinate == 3:
        return no_ops[edge]
    if coordinate == 4:
        return rejections[edge]
    if coordinate == 5:
        return memory_updates[edge]
    if coordinate == 6:
        return accepted_swaps[edge]
    return displaced_cells[edge]


@nb.njit(cache=True)
def _node_less(left: int, right: int, labels: np.ndarray) -> bool:
    for coordinate in range(labels.shape[0]):
        if labels[coordinate, left] < labels[coordinate, right]:
            return True
        if labels[coordinate, left] > labels[coordinate, right]:
            return False
    return left < right


@nb.njit(cache=True)
def _sift_up(
    heap: np.ndarray, positions: np.ndarray, size: int, index: int, labels: np.ndarray
) -> None:
    while index > 0:
        parent = (index - 1) // 2
        if not _node_less(heap[index], heap[parent], labels):
            break
        left = heap[index]
        right = heap[parent]
        heap[parent] = left
        heap[index] = right
        positions[left] = parent
        positions[right] = index
        index = parent


@nb.njit(cache=True)
def _sift_down(
    heap: np.ndarray, positions: np.ndarray, size: int, index: int, labels: np.ndarray
) -> None:
    while True:
        left = 2 * index + 1
        if left >= size:
            return
        right = left + 1
        best = left
        if right < size and _node_less(heap[right], heap[left], labels):
            best = right
        if not _node_less(heap[best], heap[index], labels):
            return
        first = heap[index]
        second = heap[best]
        heap[index] = second
        heap[best] = first
        positions[second] = index
        positions[first] = best
        index = best


@nb.njit(cache=True)
def _candidate_relation(
    target_node: int,
    predecessor: int,
    edge: int,
    profile: int,
    labels: np.ndarray,
    reads: np.ndarray,
    comparisons: np.ndarray,
    no_ops: np.ndarray,
    rejections: np.ndarray,
    memory_updates: np.ndarray,
    accepted_swaps: np.ndarray,
    displaced_cells: np.ndarray,
) -> int:
    """Return -1 if candidate is lower, 0 equal, and 1 higher."""

    for coordinate in range(labels.shape[0]):
        candidate = labels[coordinate, target_node] + _profile_cost(
            profile,
            coordinate,
            edge,
            reads,
            comparisons,
            no_ops,
            rejections,
            memory_updates,
            accepted_swaps,
            displaced_cells,
        )
        if candidate < labels[coordinate, predecessor]:
            return -1
        if candidate > labels[coordinate, predecessor]:
            return 1
    return 0


@nb.njit(cache=True)
def _assign_candidate(
    target_node: int,
    predecessor: int,
    edge: int,
    profile: int,
    labels: np.ndarray,
    reads: np.ndarray,
    comparisons: np.ndarray,
    no_ops: np.ndarray,
    rejections: np.ndarray,
    memory_updates: np.ndarray,
    accepted_swaps: np.ndarray,
    displaced_cells: np.ndarray,
) -> None:
    for coordinate in range(labels.shape[0]):
        labels[coordinate, predecessor] = labels[coordinate, target_node] + _profile_cost(
            profile,
            coordinate,
            edge,
            reads,
            comparisons,
            no_ops,
            rejections,
            memory_updates,
            accepted_swaps,
            displaced_cells,
        )


@nb.njit(cache=True)
def _reverse_lexicographic_dijkstra(
    terminal: np.ndarray,
    reverse_indptr: np.ndarray,
    predecessors: np.ndarray,
    reverse_edges: np.ndarray,
    edge_allowed: np.ndarray,
    use_allowed: bool,
    profile: int,
    width: int,
    reads: np.ndarray,
    comparisons: np.ndarray,
    no_ops: np.ndarray,
    rejections: np.ndarray,
    memory_updates: np.ndarray,
    accepted_swaps: np.ndarray,
    displaced_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = len(terminal)
    labels = np.full((width, node_count), UNREACHABLE, dtype=np.int64)
    successor = np.full(node_count, -1, dtype=np.int64)
    positions = np.full(node_count, -1, dtype=np.int64)
    settled = np.zeros(node_count, dtype=np.uint8)
    heap = np.empty(node_count, dtype=np.int64)
    size = 0

    for node in range(node_count):
        if terminal[node] == 1:
            for coordinate in range(width):
                labels[coordinate, node] = 0
            heap[size] = node
            positions[node] = size
            size += 1
            _sift_up(heap, positions, size, size - 1, labels)

    while size > 0:
        node = heap[0]
        size -= 1
        positions[node] = -2
        settled[node] = 1
        if size > 0:
            replacement = heap[size]
            heap[0] = replacement
            positions[replacement] = 0
            _sift_down(heap, positions, size, 0, labels)
        for position in range(reverse_indptr[node], reverse_indptr[node + 1]):
            edge = reverse_edges[position]
            if use_allowed and not edge_allowed[edge]:
                continue
            predecessor = predecessors[position]
            if settled[predecessor]:
                continue
            relation = _candidate_relation(
                node,
                predecessor,
                edge,
                profile,
                labels,
                reads,
                comparisons,
                no_ops,
                rejections,
                memory_updates,
                accepted_swaps,
                displaced_cells,
            )
            if relation < 0:
                _assign_candidate(
                    node,
                    predecessor,
                    edge,
                    profile,
                    labels,
                    reads,
                    comparisons,
                    no_ops,
                    rejections,
                    memory_updates,
                    accepted_swaps,
                    displaced_cells,
                )
                successor[predecessor] = edge
                if positions[predecessor] < 0:
                    heap[size] = predecessor
                    positions[predecessor] = size
                    size += 1
                _sift_up(
                    heap,
                    positions,
                    size,
                    positions[predecessor],
                    labels,
                )
            elif relation == 0 and (
                successor[predecessor] < 0 or edge < successor[predecessor]
            ):
                successor[predecessor] = edge
    return labels, successor


def solve_all_starts_lexicographic(
    node_count: int,
    sources: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    terminal_codes: Sequence[int] | np.ndarray,
    edge_columns: Mapping[str, Sequence[int] | np.ndarray],
    profile: int,
    edge_allowed: Sequence[bool] | np.ndarray | None = None,
) -> AllStartLexicographicSolution:
    """Solve one exact S06 cost profile from every start to any complete goal."""

    source, target = validate_graph(node_count, sources, targets)
    terminal = np.ascontiguousarray(terminal_codes, dtype=np.uint8)
    if terminal.ndim != 1 or len(terminal) != node_count or np.any(terminal > 2):
        raise ValueError("terminal_codes must align with the node domain")
    if len(source) and np.any(terminal[source] != 0):
        raise ValueError("terminal nodes must have outdegree zero")
    if profile not in PROFILE_WIDTHS:
        raise ValueError(f"unsupported profile code {profile}")
    costs = s05_cost_arrays(edge_columns)
    if len(costs[0]) != len(source):
        raise ValueError("edge cost columns must align with graph edges")
    if edge_allowed is None:
        allowed = np.empty(0, dtype=np.bool_)
        use_allowed = False
    else:
        allowed = np.ascontiguousarray(edge_allowed, dtype=np.bool_)
        if allowed.ndim != 1 or len(allowed) != len(source):
            raise ValueError("edge_allowed must align with graph edges")
        use_allowed = True
    reverse_indptr, predecessors, reverse_edges = reverse_csr(
        node_count, source, target
    )
    labels, successor = _reverse_lexicographic_dijkstra(
        terminal,
        reverse_indptr,
        predecessors,
        reverse_edges,
        allowed,
        use_allowed,
        profile,
        PROFILE_WIDTHS[profile],
        *costs,
    )
    return AllStartLexicographicSolution(labels, successor)


def serialized_labels(labels: np.ndarray) -> np.ndarray:
    """Convert internal infinity to the S07 serialized -1 sentinel."""

    result = np.asarray(labels).copy()
    result[result == UNREACHABLE] = -1
    return result


def reconstruct_successor_witness(
    start: int,
    sources: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    terminal_codes: Sequence[int] | np.ndarray,
    successor_edges: Sequence[int] | np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Replay a deterministic all-start successor forest to a complete node."""

    source = _int64(sources, "sources")
    target = _int64(targets, "targets")
    terminal = np.ascontiguousarray(terminal_codes, dtype=np.uint8)
    successor = _int64(successor_edges, "successor_edges")
    if not 0 <= start < len(terminal) or len(successor) != len(terminal):
        raise ValueError("start/successor vector outside node domain")
    nodes = [start]
    edges: list[int] = []
    seen: set[int] = set()
    current = start
    while terminal[current] != 1:
        if current in seen:
            raise AssertionError("successor witness contains a cycle")
        seen.add(current)
        edge = int(successor[current])
        if edge < 0 or edge >= len(source) or int(source[edge]) != current:
            raise AssertionError("invalid successor edge")
        edges.append(edge)
        current = int(target[edge])
        nodes.append(current)
        if len(nodes) > len(terminal) + 1:
            raise AssertionError("witness exceeds simple-path bound")
    return tuple(nodes), tuple(edges)


def estimate_working_bytes(node_count: int, edge_count: int) -> int:
    """Conservative peak for the widest profile plus graph/heap/CSR arrays."""

    if node_count < 1 or edge_count < 0:
        raise ValueError("invalid family size")
    exact = (
        node_count * (8 * 8 + 8 * 5 + 2)
        + edge_count * (8 * 4 + 8 * 7 + 1)
        + (node_count + 1) * 8
    )
    return int(np.ceil(exact * 1.35))
