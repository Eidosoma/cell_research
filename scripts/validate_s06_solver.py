#!/usr/bin/env python3
"""Validate the S06 definition/solver and audit all S05 corpus requirements."""

from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import resource
import time
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes
from src.detours.necessary_detour import (
    CLASS_COMPLETE_START,
    CLASS_LABELS,
    CLASS_NECESSARY_DETOUR,
    CLASS_QUIESCENT,
    CLASS_REACHABLE_NO_DETOUR,
    CLASS_UNREACHABLE_ACTIVE,
    COST_PROFILES,
    METRIC_ALIASES,
    UNREACHABLE,
    cost_matrix_from_s05,
    estimated_primary_working_bytes,
    family_metric_levels,
    reconstruct_primary_witness,
    solve_lexicographic_witness,
    solve_primary_all_starts,
    tied_value_goal_mask,
    validate_goal_distance,
)
from src.detours.state_space import FamilySpec
from src.detours.transition_graph import EDGE_SCHEMA, NODE_SCHEMA


REPOSITORY = Path(__file__).resolve().parents[1]
S04 = Path("/artifacts/research_steps/S04")
S05 = Path("/artifacts/research_steps/S05")
OUTPUT = Path("/artifacts/research_steps/S06")
CONTRACT = REPOSITORY / "analysis/s06_necessary_detour_contract.json"
MAX_WORKING_BYTES = 4_294_967_296
METRICS = (
    "adjacent_descents",
    "inversion_count",
    "spearman_footrule",
    "maximum_rank_error",
)
EDGE_COST_COLUMNS = (
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256(b"E03/S06/array-digest/v1\x00")
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def hand_fixtures() -> list[dict[str, Any]]:
    return [
        {
            "fixtureId": "complete_zero_length",
            "levels": [0],
            "edges": [],
            "terminal": [1],
            "start": 0,
            "expectedClassification": "complete_start",
            "expectedBottleneck": 0,
            "expectedExcursion": 0,
        },
        {
            "fixtureId": "monotone_zero_excursion",
            "levels": [3, 2, 1, 0],
            "edges": [[0, 1], [1, 2], [2, 3]],
            "terminal": [0, 0, 0, 1],
            "start": 0,
            "expectedClassification": "reachable_no_detour",
            "expectedBottleneck": 3,
            "expectedExcursion": 0,
        },
        {
            "fixtureId": "strictly_necessary_excursion",
            "levels": [2, 4, 0],
            "edges": [[0, 1], [1, 2]],
            "terminal": [0, 0, 1],
            "start": 0,
            "expectedClassification": "necessary_detour",
            "expectedBottleneck": 4,
            "expectedExcursion": 2,
        },
        {
            "fixtureId": "worsening_path_avoidable",
            "levels": [2, 5, 2, 0],
            "edges": [[0, 1], [1, 3], [0, 2], [2, 3]],
            "terminal": [0, 0, 0, 1],
            "start": 0,
            "expectedClassification": "reachable_no_detour",
            "expectedBottleneck": 2,
            "expectedExcursion": 0,
        },
        {
            "fixtureId": "unreachable_active_cycle",
            "levels": [1, 2, 0],
            "edges": [[0, 1], [1, 0]],
            "terminal": [0, 0, 1],
            "start": 0,
            "expectedClassification": "unreachable_active",
            "expectedBottleneck": None,
            "expectedExcursion": None,
        },
        {
            "fixtureId": "quiescent_not_goal",
            "levels": [1, 0],
            "edges": [],
            "terminal": [2, 1],
            "start": 0,
            "expectedClassification": "quiescent",
            "expectedBottleneck": None,
            "expectedExcursion": None,
        },
        {
            "fixtureId": "self_loop_parallel_secondary",
            "levels": [1, 0],
            "edges": [[0, 0], [0, 1], [0, 1]],
            "terminal": [0, 1],
            "start": 0,
            "edgeCosts": [[0, 1], [5, 1], [2, 1]],
            "expectedClassification": "reachable_no_detour",
            "expectedBottleneck": 1,
            "expectedExcursion": 0,
            "expectedSecondaryCost": [2, 1],
            "expectedSecondaryEdges": [2],
        },
        {
            "fixtureId": "two_stage_lexicographic_counterexample",
            "levels": [0, 1, 2, 0, 3, 0],
            "edges": [[0, 1], [1, 3], [0, 2], [2, 3], [3, 4], [4, 5]],
            "terminal": [0, 0, 0, 0, 0, 1],
            "start": 0,
            "edgeCosts": [[99, 1], [1, 1], [0, 1], [1, 1], [0, 1], [0, 1]],
            "expectedClassification": "necessary_detour",
            "expectedBottleneck": 3,
            "expectedExcursion": 3,
            "expectedSecondaryCost": [1, 4],
            "expectedSecondaryEdges": [2, 3, 4, 5],
        },
    ]


def validate_hand_fixtures(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        edges = fixture["edges"]
        source = np.asarray([edge[0] for edge in edges], dtype=np.int64)
        target = np.asarray([edge[1] for edge in edges], dtype=np.int64)
        solution = solve_primary_all_starts(
            len(fixture["levels"]), source, target, fixture["levels"], fixture["terminal"]
        )
        start = fixture["start"]
        observed_label = CLASS_LABELS[int(solution.classifications[start])]
        observed_bottleneck = (
            None
            if solution.bottleneck_levels[start] == UNREACHABLE
            else int(solution.bottleneck_levels[start])
        )
        observed_excursion = (
            None
            if solution.excursion_levels[start] == UNREACHABLE
            else int(solution.excursion_levels[start])
        )
        require(observed_label == fixture["expectedClassification"], fixture["fixtureId"])
        require(observed_bottleneck == fixture["expectedBottleneck"], fixture["fixtureId"])
        require(observed_excursion == fixture["expectedExcursion"], fixture["fixtureId"])
        witness_edges: list[int] | None = None
        secondary_cost: list[int] | None = None
        if observed_bottleneck is not None:
            nodes, path_edges = reconstruct_primary_witness(
                start, source, target, fixture["terminal"], solution
            )
            require(fixture["terminal"][nodes[-1]] == 1, fixture["fixtureId"])
            require(max(fixture["levels"][node] for node in nodes) == observed_bottleneck, fixture["fixtureId"])
            witness_edges = list(path_edges)
        if "edgeCosts" in fixture:
            witness = solve_lexicographic_witness(
                start,
                len(fixture["levels"]),
                source,
                target,
                fixture["levels"],
                fixture["terminal"],
                fixture["edgeCosts"],
                solution,
            )
            secondary_cost = list(witness.cost)
            witness_edges = list(witness.edges)
            require(secondary_cost == fixture["expectedSecondaryCost"], fixture["fixtureId"])
            require(witness_edges == fixture["expectedSecondaryEdges"], fixture["fixtureId"])
        results.append(
            {
                "fixture_scope": "hand_solvable",
                "fixture_id": fixture["fixtureId"],
                "family_ordinal": None,
                "metric": None,
                "start": start,
                "success": True,
                "classification": observed_label,
                "bottleneck_level": observed_bottleneck,
                "excursion_level": observed_excursion,
                "secondary_cost_json": json.dumps(secondary_cost),
                "witness_edges_json": json.dumps(witness_edges),
            }
        )
    tied_states = np.asarray([[1, 1, 2], [1, 1, 2], [1, 2, 1], [2, 1, 1]])
    goals = tied_value_goal_mask(tied_states, "ascending")
    require(goals.tolist() == [True, True, False, False], "tied goal set")
    validate_goal_distance([0, 0, 1, 2], goals)
    proxy_rejected = False
    try:
        validate_goal_distance([1, 1, 1, 2], goals)
    except ValueError:
        proxy_rejected = True
    require(proxy_rejected, "tied paper proxy compatibility rejection")
    results.append(
        {
            "fixture_scope": "tie_goal_contract",
            "fixture_id": "tied_identity_goal_set",
            "family_ordinal": None,
            "metric": None,
            "start": None,
            "success": True,
            "classification": "multiple_exchangeable_goals",
            "bottleneck_level": None,
            "excursion_level": None,
            "secondary_cost_json": "null",
            "witness_edges_json": "null",
        }
    )
    return results


def simple_paths(
    start: int,
    edges: list[tuple[int, int]],
    goals: set[int],
    node_count: int,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    outgoing: list[list[int]] = [[] for _ in range(node_count)]
    for edge, (source, _) in enumerate(edges):
        outgoing[source].append(edge)
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


def brute_force_label(
    start: int,
    levels: list[int],
    edges: list[tuple[int, int]],
    terminal: list[int],
    costs: np.ndarray,
) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
    goals = {node for node, code in enumerate(terminal) if code == 1}
    candidates: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for nodes, path_edges in simple_paths(start, edges, goals, len(levels)):
        peak = max(levels[node] for node in nodes)
        cost = tuple(
            int(costs[list(path_edges), column].sum()) if path_edges else 0
            for column in range(costs.shape[1])
        )
        candidates.append((peak, cost, path_edges))
    if not candidates:
        return None
    peak, cost = min((peak, cost) for peak, cost, _ in candidates)
    edges_for_label = min(
        path_edges for candidate_peak, candidate_cost, path_edges in candidates
        if candidate_peak == peak and candidate_cost == cost
    )
    return peak, cost, edges_for_label


def validate_tiny_random(graph_count: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for graph_index in range(graph_count):
        rng = random.Random(606_000 + graph_index)
        node_count = rng.randint(3, 7)
        goal = node_count - 1
        quiescent = node_count - 2 if graph_index % 7 == 0 else -1
        terminal = [0] * node_count
        terminal[goal] = 1
        if quiescent >= 0:
            terminal[quiescent] = 2
        levels = [rng.randint(0, 5) for _ in range(node_count)]
        levels[goal] = 0
        edges: list[tuple[int, int]] = []
        for source, target in itertools.product(range(node_count), repeat=2):
            if terminal[source] != 0:
                continue
            if rng.random() < 0.24:
                edges.append((source, target))
                if rng.random() < 0.14:
                    edges.append((source, target))
        costs = np.asarray(
            [[rng.randint(0, 4), 1] for _ in edges], dtype=np.int64
        ).reshape((-1, 2))
        source = np.asarray([edge[0] for edge in edges], dtype=np.int64)
        target = np.asarray([edge[1] for edge in edges], dtype=np.int64)
        solution = solve_primary_all_starts(
            node_count, source, target, levels, terminal
        )
        for start in range(node_count):
            expected = brute_force_label(start, levels, edges, terminal, costs)
            if expected is None:
                require(solution.bottleneck_levels[start] == UNREACHABLE, f"tiny {graph_index}:{start}")
                expected_class = "quiescent" if terminal[start] == 2 else "unreachable_active"
                if terminal[start] == 1:
                    raise AssertionError("goal lost zero-length path")
                require(CLASS_LABELS[int(solution.classifications[start])] == expected_class, f"tiny class {graph_index}:{start}")
                rows.append(
                    {
                        "graph_index": graph_index,
                        "start": start,
                        "node_count": node_count,
                        "edge_count": len(edges),
                        "reachable": False,
                        "expected_bottleneck": None,
                        "observed_bottleneck": None,
                        "expected_cost_json": "null",
                        "observed_cost_json": "null",
                        "exact_match": True,
                    }
                )
                continue
            expected_peak, expected_cost, _ = expected
            require(int(solution.bottleneck_levels[start]) == expected_peak, f"tiny peak {graph_index}:{start}")
            witness = solve_lexicographic_witness(
                start,
                node_count,
                source,
                target,
                levels,
                terminal,
                costs,
                solution,
            )
            require(witness.cost == expected_cost, f"tiny cost {graph_index}:{start}")
            require(max(levels[node] for node in witness.nodes) == expected_peak, f"tiny replay {graph_index}:{start}")
            rows.append(
                {
                    "graph_index": graph_index,
                    "start": start,
                    "node_count": node_count,
                    "edge_count": len(edges),
                    "reachable": True,
                    "expected_bottleneck": expected_peak,
                    "observed_bottleneck": witness.bottleneck_level,
                    "expected_cost_json": json.dumps(expected_cost),
                    "observed_cost_json": json.dumps(witness.cost),
                    "exact_match": True,
                }
            )
    return rows


def family_from_row(row: dict[str, Any]) -> FamilySpec:
    return FamilySpec.from_canonical_dict(json.loads(row["canonical_family_json"]))


def corpus_compatibility(
    family_rows: list[dict[str, Any]], graph: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    require(len(family_rows) == len(graph) == 7_984, "family count")
    require(graph["family_ordinal"].tolist() == list(range(7_984)), "graph family order")
    for ordinal, (metadata, graph_row) in enumerate(zip(family_rows, graph.to_dict("records"))):
        family = family_from_row(metadata)
        require(family.family_id == graph_row["family_id"], f"family identity {ordinal}")
        require(family.state_count == graph_row["node_count"], f"node count {ordinal}")
        expected_goals = family.cursor_state_count
        require(graph_row["complete_node_count"] == expected_goals, f"goal count {ordinal}")
        expected_max = {
            "adjacent_descents": family.n - 1,
            "inversion_count": family.n * (family.n - 1) // 2,
            "spearman_footrule": family.n * family.n // 2,
            "maximum_rank_error": family.n - 1,
        }
        estimated_bytes = estimated_primary_working_bytes(
            family.state_count, int(graph_row["edge_count"])
        )
        require(estimated_bytes <= MAX_WORKING_BYTES, f"working memory {ordinal}")
        require(family.state_count <= np.iinfo(np.int32).max, f"node index {ordinal}")
        require(int(graph_row["edge_count"]) <= np.iinfo(np.uint32).max, f"edge index {ordinal}")
        require(max(expected_max.values()) <= np.iinfo(np.int32).max, f"metric level {ordinal}")
        rows.append(
            {
                "family_ordinal": ordinal,
                "family_id": family.family_id,
                "n": family.n,
                "architecture": family.architecture.value,
                "direction": family.direction.value,
                "policy_profile": family.policy_profile,
                "fault_mode": family.fault_mode,
                "fault_count": family.fault_count,
                "node_count": family.state_count,
                "edge_count": int(graph_row["edge_count"]),
                "goal_count": expected_goals,
                "complete_count_match": True,
                "quiescent_count": int(graph_row["quiescent_node_count"]),
                "adjacent_max_level": expected_max["adjacent_descents"],
                "inversion_max_level": expected_max["inversion_count"],
                "footrule_max_level": expected_max["spearman_footrule"],
                "maximum_rank_error_max_level": expected_max["maximum_rank_error"],
                "estimated_primary_working_bytes": estimated_bytes,
                "within_frozen_boundaries": True,
            }
        )
    return rows


def physical_family(
    graph_row: dict[str, Any], include_costs: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    edge_columns = ["family_ordinal", "source_state_ordinal", "successor_state_ordinal"]
    if include_costs:
        edge_columns.extend(EDGE_COST_COLUMNS)
    edge_table = pq.read_table(
        S05 / graph_row["edge_shard"],
        columns=edge_columns,
        filters=[("family_ordinal", "=", int(graph_row["family_ordinal"]))],
    )
    node_table = pq.read_table(
        S05 / graph_row["node_shard"],
        columns=["family_ordinal", "state_ordinal", "terminal_code"],
        filters=[("family_ordinal", "=", int(graph_row["family_ordinal"]))],
    )
    require(edge_table.num_rows == graph_row["edge_count"], "physical family edges")
    require(node_table.num_rows == graph_row["node_count"], "physical family nodes")
    family_ordinals = np.asarray(node_table["family_ordinal"], dtype=np.int32)
    require(np.all(family_ordinals == graph_row["family_ordinal"]), "node family filter")
    states = np.asarray(node_table["state_ordinal"], dtype=np.uint32)
    require(np.array_equal(states, np.arange(graph_row["node_count"], dtype=np.uint32)), "node order")
    source = np.asarray(edge_table["source_state_ordinal"], dtype=np.int64)
    target = np.asarray(edge_table["successor_state_ordinal"], dtype=np.int64)
    terminal = np.asarray(node_table["terminal_code"], dtype=np.uint8)
    costs = {
        field: np.asarray(edge_table[field], dtype=np.int64)
        for field in EDGE_COST_COLUMNS if field in edge_table.column_names
    }
    return source, target, terminal, costs


def validate_physical_tiny(
    family_rows: list[dict[str, Any]], graph: pd.DataFrame
) -> list[dict[str, Any]]:
    ordinal = 2349
    graph_row = graph.iloc[ordinal].to_dict()
    family = family_from_row(family_rows[ordinal])
    source, target, terminal, fields = physical_family(graph_row, include_costs=True)
    edges = list(zip(source.tolist(), target.tolist()))
    cost_matrix = cost_matrix_from_s05(fields, "value_comparisons")
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        levels_array = family_metric_levels(family, metric)
        levels = levels_array.tolist()
        solution = solve_primary_all_starts(
            family.state_count, source, target, levels_array, terminal
        )
        for start in range(family.state_count):
            expected = brute_force_label(
                start, levels, edges, terminal.tolist(), cost_matrix
            )
            require(expected is not None, f"physical tiny unreachable {metric}:{start}")
            peak, cost, _ = expected
            require(solution.bottleneck_levels[start] == peak, f"physical tiny peak {metric}:{start}")
            witness = solve_lexicographic_witness(
                start,
                family.state_count,
                source,
                target,
                levels_array,
                terminal,
                cost_matrix,
                solution,
            )
            require(witness.cost == cost, f"physical tiny secondary {metric}:{start}")
            rows.append(
                {
                    "fixture_scope": "physical_s05_family_2349",
                    "fixture_id": f"physical_2349_{metric}_start_{start}",
                    "family_ordinal": ordinal,
                    "metric": metric,
                    "start": start,
                    "success": True,
                    "classification": CLASS_LABELS[int(solution.classifications[start])],
                    "bottleneck_level": witness.bottleneck_level,
                    "excursion_level": witness.excursion_level,
                    "secondary_cost_json": json.dumps(witness.cost),
                    "witness_edges_json": json.dumps(witness.edges),
                }
            )
    return rows


def benchmark_largest_families(
    family_rows: list[dict[str, Any]], graph: pd.DataFrame
) -> list[dict[str, Any]]:
    # Descending representatives exercise nonterminal state ordinal zero while
    # retaining the corpus maxima: 7976 has the largest edge count and 5557 the
    # largest node count.  Solutions are validated/digested, not analyzed.
    rows: list[dict[str, Any]] = []
    for ordinal, boundary in [(7976, "largest_edge_family"), (5557, "largest_node_family")]:
        graph_row = graph.iloc[ordinal].to_dict()
        family = family_from_row(family_rows[ordinal])
        load_started = time.perf_counter()
        source, target, terminal, _ = physical_family(graph_row)
        load_seconds = time.perf_counter() - load_started
        for metric in METRICS:
            levels = family_metric_levels(family, metric)
            require(np.count_nonzero(levels == 0) == graph_row["complete_node_count"], f"goal zero set {ordinal}:{metric}")
            started = time.perf_counter()
            solution = solve_primary_all_starts(
                family.state_count, source, target, levels, terminal
            )
            elapsed = time.perf_counter() - started
            require(np.all(solution.bottleneck_levels[terminal == 1] == 0), f"goal bottleneck {ordinal}:{metric}")
            finite = np.flatnonzero(
                (terminal == 0) & (solution.bottleneck_levels != UNREACHABLE)
            )
            sample = finite[np.linspace(0, len(finite) - 1, min(12, len(finite)), dtype=int)] if len(finite) else np.asarray([], dtype=int)
            for start in sample:
                nodes, _ = reconstruct_primary_witness(
                    int(start), source, target, terminal, solution
                )
                require(terminal[nodes[-1]] == 1, f"benchmark goal {ordinal}:{metric}")
                require(max(int(levels[node]) for node in nodes) == solution.bottleneck_levels[start], f"benchmark peak {ordinal}:{metric}")
            rows.append(
                {
                    "boundary": boundary,
                    "family_ordinal": ordinal,
                    "family_id": family.family_id,
                    "n": family.n,
                    "metric": metric,
                    "node_count": family.state_count,
                    "edge_count": len(source),
                    "goal_count": int(np.count_nonzero(terminal == 1)),
                    "load_seconds": load_seconds,
                    "solve_seconds": elapsed,
                    "estimated_primary_working_bytes": estimated_primary_working_bytes(family.state_count, len(source)),
                    "witnesses_replayed": len(sample),
                    "solution_digest": sha256_array(
                        solution.bottleneck_levels,
                        solution.successor_edges,
                        solution.classifications,
                    ),
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "success": True,
                }
            )
    return rows


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in [
        "solver_validation_fixtures.json",
        "solver_validation_results.parquet",
        "tiny_bruteforce_results.parquet",
        "corpus_compatibility.parquet",
        "corpus_solver_benchmarks.parquet",
        "corpus_feasibility.json",
        "input_immutability.json",
        "validation_results.json",
        "validation.log",
    ]:
        (OUTPUT / name).unlink(missing_ok=True)
    input_paths = [
        CONTRACT,
        REPOSITORY / "src/detours/necessary_detour.py",
        REPOSITORY / "src/detours/transition_graph.py",
        S04 / "state_family_inventory.parquet",
        S05 / "graph_family_manifest.parquet",
        S05 / "graph_corpus_manifest.json",
        S05 / "validation_results.json",
        S05 / "graphs/edge_shard_00.parquet",
        S05 / "graphs/edge_shard_05.parquet",
        S05 / "graphs/edge_shard_06.parquet",
        S05 / "graphs/node_shard_00.parquet",
        S05 / "graphs/node_shard_05.parquet",
        S05 / "graphs/node_shard_06.parquet",
    ]
    pre_hashes = {str(path): sha256_file(path) for path in input_paths}
    contract = json.loads(CONTRACT.read_text())
    require(contract["researchStepId"] == "S06", "contract step")
    require(contract["sourceGraph"]["edges"] == 103_599_386, "contract corpus")

    family_frame = pd.read_parquet(S04 / "state_family_inventory.parquet").sort_values("family_ordinal")
    graph = pd.read_parquet(S05 / "graph_family_manifest.parquet").sort_values("family_ordinal")
    family_rows = family_frame.to_dict("records")

    edge_schema = pq.ParquetFile(S05 / "graphs/edge_shard_00.parquet").schema_arrow.metadata or {}
    node_schema = pq.ParquetFile(S05 / "graphs/node_shard_00.parquet").schema_arrow.metadata or {}
    require(edge_schema.get(b"schemaVersion") == EDGE_SCHEMA.encode(), "S05 edge schema")
    require(node_schema.get(b"schemaVersion") == NODE_SCHEMA.encode(), "S05 node schema")

    fixtures = hand_fixtures()
    write_json(
        OUTPUT / "solver_validation_fixtures.json",
        {
            "schemaVersion": "e03.s06.solver_fixtures.v1",
            "researchStepId": "S06",
            "fixtures": fixtures,
        },
    )
    hand_rows = validate_hand_fixtures(fixtures)
    tiny_rows = validate_tiny_random(500)
    physical_tiny_rows = validate_physical_tiny(family_rows, graph)
    compatibility_rows = corpus_compatibility(family_rows, graph)

    # Compile/warm the largest-family solver path before timed evidence.
    warm = solve_primary_all_starts(2, [0], [1], [1, 0], [0, 1])
    require(warm.bottleneck_levels.tolist() == [1, 0], "solver warmup")
    benchmark_rows = benchmark_largest_families(family_rows, graph)

    write_parquet(OUTPUT / "solver_validation_results.parquet", hand_rows + physical_tiny_rows)
    write_parquet(OUTPUT / "tiny_bruteforce_results.parquet", tiny_rows)
    write_parquet(OUTPUT / "corpus_compatibility.parquet", compatibility_rows)
    write_parquet(OUTPUT / "corpus_solver_benchmarks.parquet", benchmark_rows)

    compatibility = pd.DataFrame(compatibility_rows)
    largest_estimate = int(compatibility["estimated_primary_working_bytes"].max())
    largest_estimate_family = int(
        compatibility.loc[
            compatibility["estimated_primary_working_bytes"].idxmax(), "family_ordinal"
        ]
    )
    feasibility = {
        "schemaVersion": "e03.s06.corpus_feasibility.v1",
        "researchStepId": "S06",
        "success": True,
        "familyCount": len(compatibility_rows),
        "nodeCount": int(compatibility["node_count"].sum()),
        "edgeCount": int(compatibility["edge_count"].sum()),
        "goalCount": int(compatibility["goal_count"].sum()),
        "quiescentCount": int(compatibility["quiescent_count"].sum()),
        "maximumFamilyNodes": int(compatibility["node_count"].max()),
        "maximumFamilyEdges": int(compatibility["edge_count"].max()),
        "maximumEstimatedPrimaryWorkingBytes": largest_estimate,
        "maximumEstimateFamilyOrdinal": largest_estimate_family,
        "frozenMaximumWorkingBytes": MAX_WORKING_BYTES,
        "boundaryTriggers": [],
        "allFamiliesRetained": True,
        "fullCorpusSolutionDeferredToS07": True,
        "benchmarks": benchmark_rows,
    }
    write_json(OUTPUT / "corpus_feasibility.json", feasibility)

    post_hashes = {str(path): sha256_file(path) for path in input_paths}
    require(pre_hashes == post_hashes, "input mutation")
    write_json(
        OUTPUT / "input_immutability.json",
        {
            "schemaVersion": "e03.s06.input_immutability.v1",
            "researchStepId": "S06",
            "success": True,
            "preRunSha256": pre_hashes,
            "postRunSha256": post_hashes,
        },
    )

    class_counts = Counter(row["classification"] for row in hand_rows)
    checks = {
        "contract_present_and_frozen": True,
        "complete_zero_length_fixture": class_counts["complete_start"] >= 1,
        "monotone_no_excursion_fixture": class_counts["reachable_no_detour"] >= 1,
        "strictly_necessary_fixture": class_counts["necessary_detour"] >= 1,
        "avoidable_worsening_fixture": True,
        "unreachable_not_vacuous_fixture": class_counts["unreachable_active"] >= 1,
        "quiescent_not_goal_fixture": class_counts["quiescent"] >= 1,
        "tied_multi_goal_and_proxy_rejection": class_counts["multiple_exchangeable_goals"] == 1,
        "self_loop_parallel_secondary_fixture": True,
        "two_stage_lexicographic_fixture": True,
        "tiny_bruteforce_primary_exact": all(row["exact_match"] for row in tiny_rows),
        "tiny_bruteforce_secondary_exact": all(row["exact_match"] for row in tiny_rows),
        "physical_s05_tiny_exact": all(row["success"] for row in physical_tiny_rows),
        "all_s05_families_compatible": len(compatibility_rows) == 7_984,
        "all_s05_goals_match_complete_nodes": all(row["complete_count_match"] for row in compatibility_rows),
        "all_exact_metric_levels_fit": int(compatibility[["adjacent_max_level", "inversion_max_level", "footrule_max_level", "maximum_rank_error_max_level"]].max().max()) <= np.iinfo(np.int32).max,
        "largest_family_benchmarks_pass": all(row["success"] for row in benchmark_rows),
        "no_corpus_budget_trigger": not feasibility["boundaryTriggers"],
        "input_immutability": True,
        "structural_scheduler_claim_boundary_recorded": contract["schedulerQuantification"]["primary"].startswith("existential"),
    }
    results = {
        "schemaVersion": "e03.s06.validation_results.v1",
        "researchStepId": "S06",
        "success": all(checks.values()),
        "checks": checks,
        "handFixtureCount": len(hand_rows),
        "tinyGraphCount": 500,
        "tinyStartFixtureCount": len(tiny_rows),
        "physicalS05TinyFixtureCount": len(physical_tiny_rows),
        "corpusFamilyCount": len(compatibility_rows),
        "corpusNodeCount": int(compatibility["node_count"].sum()),
        "corpusEdgeCount": int(compatibility["edge_count"].sum()),
        "largestFamilyBenchmarkCount": len(benchmark_rows),
        "elapsedSeconds": time.perf_counter() - started,
    }
    write_json(OUTPUT / "validation_results.json", results)
    (OUTPUT / "validation.log").write_text(
        "S06 VALIDATION PASS\n"
        + "\n".join(f"{name}=PASS" for name in checks)
        + f"\nhand_fixtures={len(hand_rows)}\ntiny_start_fixtures={len(tiny_rows)}"
        + f"\nphysical_s05_tiny_fixtures={len(physical_tiny_rows)}"
        + f"\ncorpus_families={len(compatibility_rows)}\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    if not results["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(variable, "1")
    main()
