#!/usr/bin/env python3
"""Independent physical and mathematical validation for the complete S07 corpus."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Iterator
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
from typing import Any

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.detours.necessary_detour import (
    CLASS_COMPLETE_START,
    CLASS_NECESSARY_DETOUR,
    CLASS_QUIESCENT,
    CLASS_REACHABLE_NO_DETOUR,
    CLASS_UNREACHABLE_ACTIVE,
    UNREACHABLE,
    family_metric_levels,
    solve_primary_all_starts,
)
from src.detours.path_solutions import (
    PROFILE_ACCEPTED_SWAPS,
    PROFILE_ACTIVATIONS,
    PROFILE_FULL_LEDGER,
    PROFILE_OBSERVATION_READS,
    PROFILE_VALUE_COMPARISONS,
    reverse_csr,
    solve_all_starts_lexicographic,
)
from src.detours.state_space import FamilySpec


S04 = Path("/artifacts/research_steps/S04")
S05 = Path("/artifacts/research_steps/S05")
OUTPUT = Path("/artifacts/research_steps/S07")
CACHE = Path("/cache/e03_s07_solutions")
REPOSITORY = Path(__file__).resolve().parents[1]
WORKERS = 8
METRICS = (
    "adjacent_descents",
    "inversion_count",
    "spearman_footrule",
    "maximum_rank_error",
)
METRIC_CODES = {name: index for index, name in enumerate(METRICS)}
COST_COLUMNS = (
    "observation_reads",
    "value_comparisons",
    "cost_no_ops",
    "cost_rejections",
    "cost_memory_updates",
    "cost_accepted_swaps",
    "cost_displaced_cells",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def family_from_row(row: dict[str, Any]) -> FamilySpec:
    return FamilySpec.from_canonical_dict(json.loads(row["canonical_family_json"]))


def row_group_map(path: Path) -> tuple[pq.ParquetFile, dict[int, list[int]]]:
    parquet = pq.ParquetFile(path)
    column = parquet.schema_arrow.names.index("family_ordinal")
    result: dict[int, list[int]] = {}
    for row_group in range(parquet.num_row_groups):
        statistics = parquet.metadata.row_group(row_group).column(column).statistics
        if statistics is None or int(statistics.min) != int(statistics.max):
            raise AssertionError(f"row group {row_group} in {path} spans families")
        result.setdefault(int(statistics.min), []).append(row_group)
    return parquet, result


def read_groups(
    parquet: pq.ParquetFile,
    groups: dict[int, list[int]],
    ordinal: int,
    columns: list[str] | None = None,
) -> pa.Table:
    selected = groups.get(ordinal, [])
    if not selected:
        if columns is None:
            schema = parquet.schema_arrow
        else:
            schema = pa.schema([parquet.schema_arrow.field(name) for name in columns])
        return pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in schema], schema=schema
        )
    table = parquet.read_row_groups(selected, columns=columns).combine_chunks()
    if table.num_rows and np.any(table["family_ordinal"].to_numpy() != ordinal):
        raise AssertionError("family row-group map returned a foreign family")
    return table


def components_for_profile(
    profile: str, costs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    edge_count = len(costs["observation_reads"])
    ones = np.ones(edge_count, dtype=np.int64)
    if profile == "activations":
        return [ones]
    if profile == "observation_reads":
        return [costs["observation_reads"].astype(np.int64), ones]
    if profile == "value_comparisons":
        return [costs["value_comparisons"].astype(np.int64), ones]
    if profile == "accepted_swaps":
        return [costs["cost_accepted_swaps"].astype(np.int64), ones]
    if profile == "full_ledger":
        return [
            ones,
            costs["observation_reads"].astype(np.int64),
            costs["value_comparisons"].astype(np.int64),
            costs["cost_no_ops"].astype(np.int64),
            costs["cost_rejections"].astype(np.int64),
            costs["cost_memory_updates"].astype(np.int64),
            costs["cost_accepted_swaps"].astype(np.int64),
            costs["cost_displaced_cells"].astype(np.int64),
        ]
    raise ValueError(profile)


def validate_cost_bellman(
    labels: np.ndarray,
    successor: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    terminal: np.ndarray,
    components: list[np.ndarray],
) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    reachable = labels[0] >= 0
    goals = terminal == 1
    active = terminal == 0
    unsuccessful = ~reachable
    if np.any(labels[:, goals] != 0) or np.any(successor[goals] != -1):
        raise AssertionError("minimum-cost goal boundary mismatch")
    if np.any(labels[:, unsuccessful] != -1) or np.any(successor[unsuccessful] != -1):
        raise AssertionError("minimum-cost unsuccessful sentinel mismatch")
    valid_target = reachable[targets]
    if np.any(valid_target & ~reachable[sources]):
        raise AssertionError("unreachable source has edge to a reachable target")
    equal_prefix = valid_target.copy()
    any_better = np.zeros(len(sources), dtype=bool)
    for coordinate, edge_cost in enumerate(components):
        candidate = labels[coordinate, targets] + edge_cost
        any_better |= equal_prefix & (candidate < labels[coordinate, sources])
        equal_prefix &= candidate == labels[coordinate, sources]
    if np.any(any_better):
        raise AssertionError("lexicographic Bellman lower candidate exists")
    reachable_active = active & reachable
    if np.any(successor[reachable_active] < 0):
        raise AssertionError("reachable active state lacks a cost successor")
    nodes = np.flatnonzero(reachable_active)
    selected = successor[nodes].astype(np.int64)
    if len(selected):
        if np.any(selected >= len(sources)) or np.any(sources[selected] != nodes):
            raise AssertionError("cost witness successor source mismatch")
        for coordinate, edge_cost in enumerate(components):
            observed = labels[coordinate, targets[selected]] + edge_cost[selected]
            if np.any(observed != labels[coordinate, nodes]):
                raise AssertionError("cost witness Bellman equality mismatch")
    return {
        "reachable": int(reachable.sum()),
        "reachableActive": int(reachable_active.sum()),
        "unreachable": int(unsuccessful.sum()),
        "edgesChecked": len(sources),
    }


def validate_minimax_bellman(
    levels: np.ndarray,
    peak: np.ndarray,
    excursion: np.ndarray,
    classification: np.ndarray,
    successor: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    terminal: np.ndarray,
) -> dict[str, int]:
    reachable = peak >= 0
    goals = terminal == 1
    active = terminal == 0
    quiescent = terminal == 2
    if np.any(levels[goals] != 0) or np.any(peak[goals] != 0):
        raise AssertionError("minimax goal boundary mismatch")
    if np.any(excursion[goals] != 0) or np.any(classification[goals] != CLASS_COMPLETE_START):
        raise AssertionError("complete class mismatch")
    if np.any(successor[goals] != -1):
        raise AssertionError("complete minimax successor must be absent")
    if np.any(peak[quiescent] != -1) or np.any(excursion[quiescent] != -1):
        raise AssertionError("quiescent minimax sentinel mismatch")
    if np.any(classification[quiescent] != CLASS_QUIESCENT) or np.any(successor[quiescent] != -1):
        raise AssertionError("quiescent class/successor mismatch")
    valid_target = reachable[targets]
    if np.any(valid_target & ~reachable[sources]):
        raise AssertionError("minimax-unreachable source has reachable target")
    minimum = np.full(len(levels), np.iinfo(np.int64).max, dtype=np.int64)
    candidate = np.maximum(levels[sources[valid_target]], peak[targets[valid_target]])
    np.minimum.at(minimum, sources[valid_target], candidate)
    reachable_active = active & reachable
    unreachable_active = active & ~reachable
    if np.any(minimum[reachable_active] != peak[reachable_active]):
        raise AssertionError("minimax Bellman equation mismatch")
    if np.any(minimum[unreachable_active] != np.iinfo(np.int64).max):
        raise AssertionError("unreachable active state has a finite Bellman candidate")
    expected_excursion = peak[reachable_active] - levels[reachable_active]
    if np.any(expected_excursion != excursion[reachable_active]) or np.any(expected_excursion < 0):
        raise AssertionError("minimax excursion identity mismatch")
    expected_class = np.where(
        expected_excursion > 0, CLASS_NECESSARY_DETOUR, CLASS_REACHABLE_NO_DETOUR
    )
    if np.any(classification[reachable_active] != expected_class):
        raise AssertionError("minimax reachable classification mismatch")
    if np.any(classification[unreachable_active] != CLASS_UNREACHABLE_ACTIVE):
        raise AssertionError("unreachable active classification mismatch")
    if np.any(excursion[unreachable_active] != -1) or np.any(successor[unreachable_active] != -1):
        raise AssertionError("unreachable active sentinel mismatch")
    nodes = np.flatnonzero(reachable_active)
    selected = successor[nodes].astype(np.int64)
    if np.any(selected < 0) or np.any(selected >= len(sources)) or np.any(sources[selected] != nodes):
        raise AssertionError("primary successor source mismatch")
    selected_peak = np.maximum(levels[nodes], peak[targets[selected]])
    if np.any(selected_peak != peak[nodes]):
        raise AssertionError("primary successor minimax equality mismatch")
    return {
        "complete": int(goals.sum()),
        "reachableNoDetour": int((classification == CLASS_REACHABLE_NO_DETOUR).sum()),
        "necessaryDetour": int((classification == CLASS_NECESSARY_DETOUR).sum()),
        "unreachableActive": int(unreachable_active.sum()),
        "quiescent": int(quiescent.sum()),
        "edgesChecked": len(sources),
    }


@nb.njit(cache=True)
def threshold_oracle(
    levels: np.ndarray,
    terminal: np.ndarray,
    reverse_indptr: np.ndarray,
    predecessors: np.ndarray,
) -> np.ndarray:
    node_count = len(levels)
    result = np.full(node_count, -1, dtype=np.int64)
    queue = np.empty(node_count, dtype=np.int64)
    maximum = int(levels.max())
    for threshold in range(maximum + 1):
        reached = np.zeros(node_count, dtype=np.uint8)
        head = 0
        tail = 0
        for node in range(node_count):
            if terminal[node] == 1 and levels[node] <= threshold:
                reached[node] = 1
                queue[tail] = node
                tail += 1
        while head < tail:
            node = queue[head]
            head += 1
            for position in range(reverse_indptr[node], reverse_indptr[node + 1]):
                predecessor = predecessors[position]
                if not reached[predecessor] and levels[predecessor] <= threshold:
                    reached[predecessor] = 1
                    queue[tail] = predecessor
                    tail += 1
        for node in range(node_count):
            if reached[node] and result[node] < 0:
                result[node] = threshold
    return result


def validate_worker(shard: int) -> dict[str, Any]:
    started = time.perf_counter()
    edge_pf, edge_groups = row_group_map(S05 / f"graphs/edge_shard_{shard:02d}.parquet")
    node_pf, node_groups = row_group_map(S05 / f"graphs/node_shard_{shard:02d}.parquet")
    path_pf, path_groups = row_group_map(OUTPUT / "path_solutions.parquet")
    cost_pf, cost_groups = row_group_map(OUTPUT / "minimum_cost_solutions.parquet")
    family_frame = pd.read_parquet(S04 / "state_family_inventory.parquet")
    families = family_frame[family_frame.family_ordinal % WORKERS == shard].sort_values(
        "family_ordinal"
    )
    manifest = pd.read_parquet(OUTPUT / "solution_family_manifest.parquet").set_index(
        "family_ordinal"
    )
    graph_manifest = pd.read_parquet(S05 / "graph_family_manifest.parquet").set_index(
        "family_ordinal"
    )
    family_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    edge_columns = [
        "family_ordinal",
        "source_state_ordinal",
        "successor_state_ordinal",
        *COST_COLUMNS,
    ]
    for position, metadata in enumerate(families.to_dict("records")):
        ordinal = int(metadata["family_ordinal"])
        graph = graph_manifest.loc[ordinal]
        family = family_from_row(metadata)
        edges = read_groups(edge_pf, edge_groups, ordinal, edge_columns)
        nodes = read_groups(
            node_pf, node_groups, ordinal, ["family_ordinal", "state_ordinal", "terminal_code"]
        )
        cost_table = read_groups(cost_pf, cost_groups, ordinal).to_pandas().sort_values(
            "state_ordinal"
        )
        path_table = read_groups(path_pf, path_groups, ordinal).to_pandas().sort_values(
            ["metric_code", "state_ordinal"]
        )
        sources = edges["source_state_ordinal"].to_numpy().astype(np.int64, copy=False)
        targets = edges["successor_state_ordinal"].to_numpy().astype(np.int64, copy=False)
        costs = {name: edges[name].to_numpy() for name in COST_COLUMNS}
        terminal = nodes["terminal_code"].to_numpy()
        state_ordinals = nodes["state_ordinal"].to_numpy()
        node_count = len(terminal)
        if node_count != int(graph.node_count) or len(sources) != int(graph.edge_count):
            raise AssertionError(f"family {ordinal} S05 physical graph count mismatch")
        if len(cost_table) != node_count or len(path_table) != node_count * len(METRICS):
            raise AssertionError(f"family {ordinal} output row accounting mismatch")
        if not np.array_equal(state_ordinals, np.arange(node_count, dtype=np.uint32)):
            raise AssertionError("S05 state key order mismatch")
        if not np.array_equal(cost_table.state_ordinal.to_numpy(), state_ordinals):
            raise AssertionError("minimum-cost state key mismatch")
        if np.any(cost_table.family_ordinal.to_numpy() != ordinal):
            raise AssertionError("minimum-cost family key mismatch")

        activation_labels = cost_table[["shortest_activations"]].to_numpy().T
        activation_successor = cost_table.shortest_successor_edge_local_index.to_numpy()
        reads_labels = cost_table[
            ["minimum_observation_reads", "minimum_observation_reads_activations"]
        ].to_numpy().T
        reads_successor = cost_table.minimum_observation_reads_successor_edge_local_index.to_numpy()
        comparison_labels = cost_table[
            ["minimum_value_comparisons", "minimum_value_comparisons_activations"]
        ].to_numpy().T
        comparison_successor = cost_table.minimum_value_comparisons_successor_edge_local_index.to_numpy()
        swap_labels = cost_table[
            ["minimum_accepted_swaps", "minimum_accepted_swaps_activations"]
        ].to_numpy().T
        swap_successor = cost_table.minimum_accepted_swaps_successor_edge_local_index.to_numpy()
        full_columns = [
            f"minimum_full_ledger_{name}"
            for name in (
                "activations",
                "observation_reads",
                "value_comparisons",
                "no_ops",
                "rejections",
                "memory_updates",
                "accepted_swaps",
                "displaced_cells",
            )
        ]
        full_labels = cost_table[full_columns].to_numpy().T
        full_successor = cost_table.minimum_full_ledger_successor_edge_local_index.to_numpy()
        if not np.array_equal(cost_table.minimum_displaced_cells.to_numpy(), np.where(swap_labels[0] >= 0, 2 * swap_labels[0], -1)):
            raise AssertionError("displaced-cell label scale mismatch")
        if not np.array_equal(cost_table.minimum_displaced_cells_activations.to_numpy(), swap_labels[1]):
            raise AssertionError("displaced-cell activation tie-break mismatch")
        if not np.array_equal(cost_table.minimum_displaced_cells_successor_edge_local_index.to_numpy(), swap_successor):
            raise AssertionError("displaced-cell successor mismatch")
        cost_checks = {
            "activations": validate_cost_bellman(
                activation_labels, activation_successor, sources, targets, terminal, components_for_profile("activations", costs)
            ),
            "observation_reads": validate_cost_bellman(
                reads_labels, reads_successor, sources, targets, terminal, components_for_profile("observation_reads", costs)
            ),
            "value_comparisons": validate_cost_bellman(
                comparison_labels, comparison_successor, sources, targets, terminal, components_for_profile("value_comparisons", costs)
            ),
            "accepted_swaps": validate_cost_bellman(
                swap_labels, swap_successor, sources, targets, terminal, components_for_profile("accepted_swaps", costs)
            ),
            "full_ledger": validate_cost_bellman(
                full_labels, full_successor, sources, targets, terminal, components_for_profile("full_ledger", costs)
            ),
        }
        if not np.array_equal(activation_labels[0], full_labels[0]):
            raise AssertionError("full-ledger primary coordinate is not shortest")
        cost_digest = hash_arrays(
            terminal,
            activation_labels.astype(np.int32),
            activation_successor.astype(np.int32),
            reads_labels.astype(np.int32),
            reads_successor.astype(np.int32),
            comparison_labels.astype(np.int32),
            comparison_successor.astype(np.int32),
            swap_labels.astype(np.int32),
            swap_successor.astype(np.int32),
            full_labels.astype(np.int32),
            full_successor.astype(np.int32),
        )
        expected_manifest = manifest.loc[ordinal]
        if cost_digest != expected_manifest.cost_logical_digest:
            raise AssertionError(f"family {ordinal} cost digest mismatch")

        metric_results: dict[str, dict[str, int]] = {}
        reverse_indptr = predecessors = None
        if int(metadata["n"]) == 4:
            reverse_indptr, predecessors, _ = reverse_csr(node_count, sources, targets)
        for metric in METRICS:
            metric_code = METRIC_CODES[metric]
            physical = path_table[path_table.metric_code == metric_code].sort_values(
                "state_ordinal"
            )
            if len(physical) != node_count or not np.array_equal(
                physical.state_ordinal.to_numpy(), state_ordinals
            ):
                raise AssertionError("path solution state-metric key mismatch")
            levels = physical.metric_level.to_numpy().astype(np.int64)
            independent_levels = family_metric_levels(family, metric)
            if not np.array_equal(levels, independent_levels):
                raise AssertionError("stored metric levels mismatch S01/S06 implementation")
            peak = physical.minimum_peak.to_numpy().astype(np.int64)
            excursion = physical.minimum_excursion.to_numpy().astype(np.int64)
            classification = physical.classification_code.to_numpy().astype(np.uint8)
            successor = physical.primary_successor_edge_local_index.to_numpy().astype(np.int64)
            metric_results[metric] = validate_minimax_bellman(
                levels,
                peak,
                excursion,
                classification,
                successor,
                sources,
                targets,
                terminal,
            )
            digest = hash_arrays(
                terminal,
                levels.astype(np.int16),
                peak.astype(np.int16),
                excursion.astype(np.int16),
                classification,
                successor.astype(np.int32),
            )
            if digest != expected_manifest[f"path_{metric}_logical_digest"]:
                raise AssertionError(f"family {ordinal} {metric} digest mismatch")
            if int(metadata["n"]) == 4:
                oracle = threshold_oracle(levels, terminal, reverse_indptr, predecessors)
                if not np.array_equal(oracle, peak):
                    raise AssertionError(f"family {ordinal} {metric} threshold oracle mismatch")
                threshold_rows.append(
                    {
                        "family_ordinal": ordinal,
                        "metric": metric,
                        "states_checked": node_count,
                        "edges_checked_per_threshold": len(sources),
                        "threshold_count": int(levels.max()) + 1,
                        "matched": True,
                    }
                )
        family_rows.append(
            {
                "family_ordinal": ordinal,
                "n": int(metadata["n"]),
                "node_count": node_count,
                "edge_count": len(sources),
                "cost_profiles_checked": len(cost_checks),
                "metrics_checked": len(metric_results),
                "cost_digest_matched": True,
                "path_digests_matched": True,
                "cost_bellman_passed": True,
                "minimax_bellman_passed": True,
                "threshold_oracle_checked": int(metadata["n"]) == 4,
                "passed": True,
            }
        )
        if position % 100 == 0:
            print(json.dumps({"validationShard": shard, "position": position, "family": ordinal}), flush=True)
    family_path = CACHE / f"s07_validation_family_{shard:02d}.parquet"
    threshold_path = CACHE / f"s07_threshold_{shard:02d}.parquet"
    pq.write_table(pa.Table.from_pylist(family_rows), family_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(threshold_rows), threshold_path, compression="zstd")
    return {
        "shard": shard,
        "families": len(family_rows),
        "thresholdRows": len(threshold_rows),
        "runtimeSeconds": time.perf_counter() - started,
        "familyPath": str(family_path),
        "thresholdPath": str(threshold_path),
    }


def profile_cost(profile: str, row: tuple[int, ...]) -> tuple[int, ...]:
    reads, comparisons, no_ops, rejections, memory, swaps, displaced = row
    if profile == "activations":
        return (1,)
    if profile == "observation_reads":
        return (reads, 1)
    if profile == "value_comparisons":
        return (comparisons, 1)
    if profile == "accepted_swaps":
        return (swaps, 1)
    return (1, reads, comparisons, no_ops, rejections, memory, swaps, displaced)


def simple_paths(
    start: int, edges: list[tuple[int, int]], goals: set[int], node_count: int
) -> Iterator[tuple[int, ...]]:
    outgoing: list[list[int]] = [[] for _ in range(node_count)]
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


def tiny_crosscheck() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    profiles = {
        "activations": PROFILE_ACTIVATIONS,
        "observation_reads": PROFILE_OBSERVATION_READS,
        "value_comparisons": PROFILE_VALUE_COMPARISONS,
        "accepted_swaps": PROFILE_ACCEPTED_SWAPS,
        "full_ledger": PROFILE_FULL_LEDGER,
    }
    for seed in range(200):
        rng = random.Random(907_000 + seed)
        node_count = rng.randint(3, 7)
        goal = node_count - 1
        terminal = np.zeros(node_count, dtype=np.uint8)
        terminal[goal] = 1
        edges: list[tuple[int, int]] = []
        for source, target in itertools.product(range(node_count - 1), range(node_count)):
            if rng.random() < 0.28:
                edges.append((source, target))
                if rng.random() < 0.16:
                    edges.append((source, target))
        costs: list[tuple[int, ...]] = []
        for _ in edges:
            swaps = rng.randint(0, 1)
            costs.append(
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
        columns = {
            name: np.asarray([row[index] for row in costs], dtype=np.int64)
            for index, name in enumerate(COST_COLUMNS)
        }
        comparisons = 0
        for profile, code in profiles.items():
            solved = solve_all_starts_lexicographic(
                node_count, source, target, terminal, columns, code
            )
            for start in range(node_count):
                candidates: list[tuple[int, ...]] = []
                for path in simple_paths(start, edges, {goal}, node_count):
                    if path:
                        width = solved.labels.shape[0]
                        label = tuple(
                            sum(profile_cost(profile, costs[edge])[coordinate] for edge in path)
                            for coordinate in range(width)
                        )
                    else:
                        label = (0,) * solved.labels.shape[0]
                    candidates.append(label)
                observed = tuple(int(value) for value in solved.labels[:, start])
                expected = min(candidates) if candidates else (UNREACHABLE,) * solved.labels.shape[0]
                if observed != expected:
                    raise AssertionError(f"tiny graph {seed} {profile} start {start} mismatch")
                comparisons += 1
        rows.append(
            {
                "seed": seed,
                "node_count": node_count,
                "edge_count": len(edges),
                "profile_start_comparisons": comparisons,
                "matched": True,
            }
        )
    return pd.DataFrame(rows)


def witness_replay() -> pd.DataFrame:
    witnesses = pd.read_parquet(OUTPUT / "witness_paths.parquet")
    family_metadata = pd.read_parquet(S04 / "state_family_inventory.parquet").set_index(
        "family_ordinal"
    )
    graph_manifest = pd.read_parquet(S05 / "graph_family_manifest.parquet").set_index(
        "family_ordinal"
    )
    rows: list[dict[str, Any]] = []
    for ordinal, group in witnesses.groupby("family_ordinal", sort=True):
        ordinal = int(ordinal)
        graph = graph_manifest.loc[ordinal]
        edges = pq.read_table(
            S05 / graph.edge_shard,
            filters=[("family_ordinal", "=", ordinal)],
        ).combine_chunks()
        nodes = pq.read_table(
            S05 / graph.node_shard,
            filters=[("family_ordinal", "=", ordinal)],
        ).combine_chunks()
        source = edges["source_state_ordinal"].to_numpy()
        target = edges["successor_state_ordinal"].to_numpy()
        opportunity = edges["opportunity_ordinal"].to_numpy()
        terminal = nodes["terminal_code"].to_numpy()
        edge_costs = {name: edges[name].to_numpy() for name in COST_COLUMNS}
        family = family_from_row(family_metadata.loc[ordinal].to_dict())
        metric_levels = {
            metric: family_metric_levels(family, metric) for metric in METRICS
        }
        for row in group.itertuples(index=False):
            path_nodes = np.asarray(row.node_ordinals, dtype=np.int64)
            path_edges = np.asarray(row.edge_local_indices, dtype=np.int64)
            path_opportunities = np.asarray(row.opportunity_ordinals, dtype=np.int64)
            if len(path_nodes) != len(path_edges) + 1 or path_nodes[0] != row.state_ordinal:
                raise AssertionError("witness path shape/start mismatch")
            if len(path_edges):
                if np.any(path_edges < 0) or np.any(path_edges >= len(source)):
                    raise AssertionError("witness edge outside family")
                if np.any(source[path_edges] != path_nodes[:-1]) or np.any(
                    target[path_edges] != path_nodes[1:]
                ):
                    raise AssertionError("witness endpoint replay mismatch")
                if np.any(opportunity[path_edges] != path_opportunities):
                    raise AssertionError("witness opportunity key mismatch")
            if terminal[path_nodes[-1]] != 1 or int(path_nodes[-1]) != row.goal_state_ordinal:
                raise AssertionError("witness does not terminate at a complete goal")
            totals = [len(path_edges)]
            for name in COST_COLUMNS:
                totals.append(int(edge_costs[name][path_edges].sum()) if len(path_edges) else 0)
            recorded = [
                row.ledger_activations,
                row.ledger_observation_reads,
                row.ledger_value_comparisons,
                row.ledger_no_ops,
                row.ledger_rejections,
                row.ledger_memory_updates,
                row.ledger_accepted_swaps,
                row.ledger_displaced_cells,
            ]
            if totals != recorded:
                raise AssertionError("witness ledger replay mismatch")
            observed_peak = int(metric_levels[row.metric][path_nodes].max())
            if observed_peak != row.observed_path_peak:
                raise AssertionError("witness metric-peak replay mismatch")
            expected_label = json.loads(row.optimal_label_json)
            if row.cost_profile == "none":
                actual_label = [row.primary_minimum_peak]
            elif row.cost_profile == "activations":
                actual_label = [totals[0]]
            elif row.cost_profile == "observation_reads":
                actual_label = [totals[1], totals[0]]
            elif row.cost_profile == "value_comparisons":
                actual_label = [totals[2], totals[0]]
            elif row.cost_profile == "accepted_swaps":
                actual_label = [totals[6], totals[0]]
            elif row.cost_profile == "displaced_cells":
                actual_label = [totals[7], totals[0]]
            elif row.cost_profile == "full_ledger":
                actual_label = totals
            else:
                raise AssertionError("unknown witness cost profile")
            if actual_label != expected_label:
                raise AssertionError("witness optimal-label replay mismatch")
            if row.solution_scope in {"primary_minimax", "secondary_minimax"}:
                if observed_peak != row.primary_minimum_peak:
                    raise AssertionError("minimax witness peak is not primary-optimal")
            digest = hashlib.sha256()
            digest.update(np.asarray([ordinal], dtype=np.int32).tobytes())
            digest.update(path_nodes.astype(np.uint32).tobytes())
            digest.update(path_edges.astype(np.int32).tobytes())
            digest.update(path_opportunities.astype(np.uint8).tobytes())
            if digest.hexdigest() != row.path_logical_digest:
                raise AssertionError("witness logical digest mismatch")
            rows.append(
                {
                    "panel_case_id": row.panel_case_id,
                    "family_ordinal": ordinal,
                    "metric": row.metric,
                    "solution_scope": row.solution_scope,
                    "cost_profile": row.cost_profile,
                    "path_edge_count": len(path_edges),
                    "endpoint_replay_matched": True,
                    "opportunity_keys_matched": True,
                    "ledger_matched": True,
                    "metric_peak_matched": True,
                    "optimal_label_matched": True,
                    "goal_matched": True,
                    "digest_matched": True,
                    "passed": True,
                }
            )
    return pd.DataFrame(rows)


def deterministic_sample() -> pd.DataFrame:
    family_frame = pd.read_parquet(S04 / "state_family_inventory.parquet").set_index(
        "family_ordinal"
    )
    graph_manifest = pd.read_parquet(S05 / "graph_family_manifest.parquet").set_index(
        "family_ordinal"
    )
    solution_manifest = pd.read_parquet(OUTPUT / "solution_family_manifest.parquet").set_index(
        "family_ordinal"
    )
    selected = sorted(
        set(np.linspace(0, 7983, 28, dtype=int).tolist()) | {0, 2349, 5557, 7976}
    )
    rows: list[dict[str, Any]] = []
    for ordinal in selected:
        metadata = family_frame.loc[ordinal]
        graph = graph_manifest.loc[ordinal]
        family = family_from_row(metadata.to_dict())
        edges = pq.read_table(
            S05 / graph.edge_shard,
            filters=[("family_ordinal", "=", ordinal)],
        ).combine_chunks()
        nodes = pq.read_table(
            S05 / graph.node_shard,
            filters=[("family_ordinal", "=", ordinal)],
        ).combine_chunks()
        source = edges["source_state_ordinal"].to_numpy().astype(np.int64, copy=False)
        target = edges["successor_state_ordinal"].to_numpy().astype(np.int64, copy=False)
        terminal = nodes["terminal_code"].to_numpy()
        costs = {name: edges[name].to_numpy() for name in COST_COLUMNS}
        profiles = [
            PROFILE_ACTIVATIONS,
            PROFILE_OBSERVATION_READS,
            PROFILE_VALUE_COMPARISONS,
            PROFILE_ACCEPTED_SWAPS,
            PROFILE_FULL_LEDGER,
        ]
        solved = [
            solve_all_starts_lexicographic(
                len(terminal), source, target, terminal, costs, profile
            )
            for profile in profiles
        ]
        serialized = []
        for result in solved:
            labels = result.labels.copy()
            labels[labels == UNREACHABLE] = -1
            serialized.extend([labels.astype(np.int32), result.successor_edges.astype(np.int32)])
        cost_digest = hash_arrays(terminal, *serialized)
        if cost_digest != solution_manifest.loc[ordinal].cost_logical_digest:
            raise AssertionError(f"deterministic cost digest mismatch family {ordinal}")
        metric_matches = 0
        for metric in METRICS:
            levels = family_metric_levels(family, metric)
            result = solve_primary_all_starts(
                len(terminal), source, target, levels, terminal
            )
            peak = result.bottleneck_levels.copy()
            peak[peak == UNREACHABLE] = -1
            excursion = result.excursion_levels.copy()
            excursion[excursion == UNREACHABLE] = -1
            digest = hash_arrays(
                terminal,
                levels.astype(np.int16),
                peak.astype(np.int16),
                excursion.astype(np.int16),
                result.classifications.astype(np.uint8),
                result.successor_edges.astype(np.int32),
            )
            if digest != solution_manifest.loc[ordinal][f"path_{metric}_logical_digest"]:
                raise AssertionError(f"deterministic path digest mismatch family {ordinal} {metric}")
            metric_matches += 1
        rows.append(
            {
                "family_ordinal": ordinal,
                "n": int(metadata.n),
                "node_count": len(terminal),
                "edge_count": len(source),
                "cost_digest_matched": True,
                "metric_digests_matched": metric_matches,
                "passed": True,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    context = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        futures = [pool.submit(validate_worker, shard) for shard in range(WORKERS)]
        worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]
    worker_results.sort(key=lambda row: row["shard"])
    family_validation = pd.concat(
        [pd.read_parquet(row["familyPath"]) for row in worker_results], ignore_index=True
    ).sort_values("family_ordinal")
    threshold_validation = pd.concat(
        [pd.read_parquet(row["thresholdPath"]) for row in worker_results], ignore_index=True
    ).sort_values(["family_ordinal", "metric"])
    tiny_validation = tiny_crosscheck()
    witness_validation = witness_replay()
    deterministic_validation = deterministic_sample()
    pq.write_table(
        pa.Table.from_pandas(family_validation, preserve_index=False),
        OUTPUT / "bellman_validation_by_family.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pandas(threshold_validation, preserve_index=False),
        OUTPUT / "independent_threshold_crosscheck.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pandas(tiny_validation, preserve_index=False),
        OUTPUT / "independent_solver_crosscheck.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pandas(witness_validation, preserve_index=False),
        OUTPUT / "witness_replay_validation.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pandas(deterministic_validation, preserve_index=False),
        OUTPUT / "deterministic_reconstruction.parquet",
        compression="zstd",
    )

    before = json.loads((CACHE / "input_hashes_before.json").read_text())
    after = {path: sha256_file(Path(path)) for path in before}
    changed = [path for path in before if before[path] != after[path]]
    immutability = {
        "schemaVersion": "e03.s07.input_immutability.v1",
        "researchStepId": "S07",
        "success": not changed,
        "inputCount": len(before),
        "changedInputs": changed,
        "before": before,
        "after": after,
    }
    write_json(OUTPUT / "input_immutability.json", immutability)
    if changed:
        raise AssertionError(f"S07 inputs mutated: {changed}")

    path_pf = pq.ParquetFile(OUTPUT / "path_solutions.parquet")
    cost_pf = pq.ParquetFile(OUTPUT / "minimum_cost_solutions.parquet")
    prevalence = pd.read_parquet(OUTPUT / "necessity_prevalence_by_family.parquet")
    summary_counts = prevalence.groupby("metric")[
        [
            "node_count",
            "complete_start_count",
            "reachable_no_detour_count",
            "necessary_detour_count",
            "unreachable_active_count",
            "quiescent_count",
        ]
    ].sum()
    gates = {
        "family_count": len(family_validation) == 7984,
        "all_family_bellman": bool(family_validation.passed.all()),
        "state_count": cost_pf.metadata.num_rows == 22301808,
        "state_metric_count": path_pf.metadata.num_rows == 89207232,
        "prevalence_family_metric_count": len(prevalence) == 31936,
        "class_partition_each_metric": bool(
            (
                summary_counts[
                    [
                        "complete_start_count",
                        "reachable_no_detour_count",
                        "necessary_detour_count",
                        "unreachable_active_count",
                        "quiescent_count",
                    ]
                ].sum(axis=1)
                == summary_counts.node_count
            ).all()
        ),
        "threshold_n4_all_metrics": len(threshold_validation) > 0
        and bool(threshold_validation.matched.all()),
        "tiny_independent_solver": len(tiny_validation) == 200
        and bool(tiny_validation.matched.all()),
        "witness_replay": len(witness_validation) == 1157
        and bool(witness_validation.passed.all()),
        "deterministic_reconstruction": bool(deterministic_validation.passed.all()),
        "input_immutability": not changed,
        "artifact_budget": sum(
            path.stat().st_size for path in OUTPUT.rglob("*") if path.is_file()
        )
        <= 12 * 1024**3,
    }
    if not all(gates.values()):
        raise AssertionError({name: value for name, value in gates.items() if not value})
    result = {
        "schemaVersion": "e03.s07.validation_results.v1",
        "researchStepId": "S07",
        "success": True,
        "status": "complete",
        "gates": gates,
        "gateCount": len(gates),
        "workers": WORKERS,
        "workerResults": worker_results,
        "familiesValidated": len(family_validation),
        "statesValidated": int(cost_pf.metadata.num_rows),
        "stateMetricRowsValidated": int(path_pf.metadata.num_rows),
        "edgesValidatedPerBellmanProfile": 103599386,
        "allStateCostProfiles": 6,
        "allStateIndependentMetrics": 4,
        "n4ThresholdFamilyMetricCrosschecks": len(threshold_validation),
        "n4ThresholdStatesChecked": int(threshold_validation.states_checked.sum()),
        "tinyGraphs": len(tiny_validation),
        "tinyProfileStartComparisons": int(tiny_validation.profile_start_comparisons.sum()),
        "witnessRowsReplayed": len(witness_validation),
        "witnessEdgesReplayed": int(witness_validation.path_edge_count.sum()),
        "deterministicFamiliesRebuilt": len(deterministic_validation),
        "inputHashesChecked": len(before),
        "runtimeSeconds": time.perf_counter() - started,
    }
    write_json(OUTPUT / "validation_results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
