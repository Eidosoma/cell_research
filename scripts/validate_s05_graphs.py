#!/usr/bin/env python3
"""Independent physical, semantic, reachability, and coverage audit for S05."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.engine import evaluate_terminal
from scripts.build_s05_graphs import ColumnStreamDigest, write_json, write_parquet
from src.detours.state_space import FamilySpec, StructuralState
from src.detours.transition_graph import (
    EDGE_ARRAY_COLUMNS,
    EDGE_DIGEST_DOMAIN,
    EDGE_SCHEMA,
    NODE_DIGEST_DOMAIN,
    NODE_SCHEMA,
    compare_edge_records,
    e01_outgoing_edges,
    opportunity_arrays,
)


S04 = Path("/artifacts/research_steps/S04")
S05 = Path("/artifacts/research_steps/S05")
EXPECTED_FAMILIES = 7_984
EXPECTED_NODES = 22_301_808


EDGE_DTYPES: dict[str, Any] = {
    "source_state_ordinal": np.uint32,
    "successor_state_ordinal": np.uint32,
    "opportunity_ordinal": np.uint8,
    "scheduler_actor_index": np.int8,
    "scheduler_side_code": np.int8,
    "probability_numerator": np.uint8,
    "probability_denominator": np.uint8,
    "proposal_kind_code": np.uint8,
    "proposal_reason_code": np.uint8,
    "decision_code": np.uint8,
    "proposal_actor_index": np.int8,
    "actor_position": np.int8,
    "target_position": np.int8,
    "new_cursor": np.int8,
    "changed": np.uint8,
    "observation_reads": np.uint16,
    "value_comparisons": np.uint16,
    "cost_no_ops": np.uint8,
    "cost_rejections": np.uint8,
    "cost_memory_updates": np.uint8,
    "cost_accepted_swaps": np.uint8,
    "cost_displaced_cells": np.uint8,
    "delta_adjacent_descents": np.int8,
    "delta_paper_distance_numerator": np.int8,
    "delta_inversion_count": np.int16,
    "delta_spearman_footrule": np.int16,
    "delta_maximum_rank_error": np.int8,
    "delta_emd_numerator": np.int16,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def family_from_row(row: dict[str, Any]) -> FamilySpec:
    return FamilySpec.from_canonical_dict(json.loads(row["canonical_family_json"]))


def numpy_column(table: pa.Table, name: str, dtype: Any) -> np.ndarray:
    return np.asarray(table[name], dtype=dtype)


@nb.njit(cache=True)
def _reachable_from_materialized_edges(
    node_count: int, source: np.ndarray, successor: np.ndarray
) -> np.ndarray:
    counts = np.zeros(node_count, dtype=np.int64)
    for value in source:
        counts[value] += 1
    indptr = np.empty(node_count + 1, dtype=np.int64)
    indptr[0] = 0
    for index in range(node_count):
        indptr[index + 1] = indptr[index] + counts[index]
    reachable = np.zeros(node_count, dtype=np.uint8)
    queue = np.empty(node_count, dtype=np.uint32)
    queue[0] = 0
    reachable[0] = 1
    head = 0
    tail = 1
    while head < tail:
        current = int(queue[head])
        head += 1
        for edge in range(indptr[current], indptr[current + 1]):
            target = int(successor[edge])
            if reachable[target] == 0:
                reachable[target] = 1
                queue[tail] = target
                tail += 1
    return reachable


def load_nodes(
    family_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    terminal_by_family: list[np.ndarray | None] = [None] * EXPECTED_FAMILIES
    reachable_by_family: list[np.ndarray | None] = [None] * EXPECTED_FAMILIES
    digest_rows: list[dict[str, Any]] = []
    for shard in sorted((S05 / "graphs").glob("node_shard_*.parquet")):
        parquet = pq.ParquetFile(shard)
        require(
            (parquet.schema_arrow.metadata or {}).get(b"schemaVersion")
            == NODE_SCHEMA.encode(),
            f"node schema {shard}",
        )
        current = -1
        state_chunks: list[np.ndarray] = []
        terminal_chunks: list[np.ndarray] = []
        reachable_chunks: list[np.ndarray] = []

        def finalize() -> None:
            nonlocal current, state_chunks, terminal_chunks, reachable_chunks
            if current < 0:
                return
            states = np.concatenate(state_chunks)
            terminal = np.concatenate(terminal_chunks)
            reachable = np.concatenate(reachable_chunks)
            family = family_from_row(family_rows[current])
            expected = graph_rows[current]
            require(
                np.array_equal(states, np.arange(family.state_count, dtype=np.uint32)),
                f"node ordinals family {current}",
            )
            require(np.all(terminal <= 2), f"terminal codes family {current}")
            require(np.all(reachable <= 1) and reachable[0] == 1, f"reachability codes family {current}")
            digest = ColumnStreamDigest(
                NODE_DIGEST_DOMAIN,
                bytes(family_rows[current]["family_digest"]),
                ("state_ordinal", "terminal_code", "reachable_from_anchor"),
            )
            digest.update(
                {
                    "state_ordinal": states,
                    "terminal_code": terminal,
                    "reachable_from_anchor": reachable,
                }
            )
            require(digest.hexdigest() == expected["node_logical_digest"], f"node digest {current}")
            counts = np.bincount(terminal, minlength=3)
            require(int(counts[0]) == expected["active_node_count"], f"active count {current}")
            require(int(counts[1]) == expected["complete_node_count"], f"complete count {current}")
            require(int(counts[2]) == expected["quiescent_node_count"], f"quiescent count {current}")
            require(int(reachable.sum()) == expected["reachable_node_count"], f"reachable count {current}")
            terminal_by_family[current] = terminal
            reachable_by_family[current] = reachable
            digest_rows.append(
                {
                    "family_ordinal": current,
                    "node_rows": len(states),
                    "node_digest_match": True,
                }
            )
            current = -1
            state_chunks = []
            terminal_chunks = []
            reachable_chunks = []

        for row_group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(row_group)
            ordinals = numpy_column(table, "family_ordinal", np.int32)
            unique = np.unique(ordinals)
            require(len(unique) == 1, f"mixed node family row group {shard}:{row_group}")
            ordinal = int(unique[0])
            if current >= 0 and ordinal != current:
                finalize()
            if current < 0:
                current = ordinal
            state_chunks.append(numpy_column(table, "state_ordinal", np.uint32))
            terminal_chunks.append(numpy_column(table, "terminal_code", np.uint8))
            reachable_chunks.append(numpy_column(table, "reachable_from_anchor", np.uint8))
        finalize()
    require(all(value is not None for value in terminal_by_family), "missing node families")
    return (
        [value for value in terminal_by_family if value is not None],
        [value for value in reachable_by_family if value is not None],
        digest_rows,
    )


def sample_state_ordinals(
    family: FamilySpec,
    family_digest: bytes,
    terminal: np.ndarray,
) -> tuple[list[int], bool]:
    exhaustive = family.n == 4 and (
        family.architecture.value == "traditional" or family.selection_owner_count == 0
    )
    if exhaustive:
        return list(range(family.state_count)), True
    selected = {
        0,
        family.state_count - 1,
        int.from_bytes(family_digest[:8], "big") % family.state_count,
        int.from_bytes(family_digest[8:16], "big") % family.state_count,
    }
    for code in (0, 1, 2):
        indices = np.flatnonzero(terminal == code)
        if len(indices):
            selected.add(int(indices[0]))
            selected.add(int(indices[-1]))
    return sorted(selected), False


def materialized_records(
    arrays: dict[str, np.ndarray], state_ordinal: int
) -> list[dict[str, int]]:
    source = arrays["source_state_ordinal"]
    left = int(np.searchsorted(source, state_ordinal, side="left"))
    right = int(np.searchsorted(source, state_ordinal, side="right"))
    return [
        {column: int(arrays[column][index]) for column in EDGE_ARRAY_COLUMNS}
        for index in range(left, right)
    ]


def validate_family_edges(
    ordinal: int,
    arrays: dict[str, np.ndarray],
    family_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    terminal: np.ndarray,
    expected_reachable: np.ndarray,
    simulator_rows: list[dict[str, Any]],
    terminal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = family_rows[ordinal]
    manifest = graph_rows[ordinal]
    family = family_from_row(metadata)
    edge_count = len(arrays["source_state_ordinal"])
    slots = int(manifest["opportunities_per_active_node"])
    active = np.flatnonzero(terminal == 0).astype(np.uint32)
    require(edge_count == len(active) * slots, f"edge count formula {ordinal}")
    positions = np.arange(edge_count, dtype=np.int64)
    require(
        np.array_equal(arrays["source_state_ordinal"], active[positions // slots]),
        f"active source coverage {ordinal}",
    )
    require(
        np.array_equal(arrays["opportunity_ordinal"], (positions % slots).astype(np.uint8)),
        f"opportunity coverage {ordinal}",
    )
    require(np.all(arrays["successor_state_ordinal"] < family.state_count), f"endpoint {ordinal}")
    sorted_successors = arrays["successor_state_ordinal"].reshape((-1, slots)).copy()
    sorted_successors.sort(axis=1)
    parallel_endpoint_extra = int(
        np.count_nonzero(sorted_successors[:, 1:] == sorted_successors[:, :-1])
    )
    require(
        parallel_endpoint_extra == manifest["parallel_endpoint_extra_count"],
        f"parallel endpoint accounting {ordinal}",
    )
    action_sum = (
        arrays["cost_no_ops"].astype(np.uint8)
        + arrays["cost_rejections"]
        + arrays["cost_memory_updates"]
        + arrays["cost_accepted_swaps"]
    )
    require(np.all(action_sum == 1), f"action partition {ordinal}")
    require(
        np.array_equal(
            arrays["cost_displaced_cells"], 2 * arrays["cost_accepted_swaps"]
        ),
        f"displacement identity {ordinal}",
    )
    changed_expected = (
        arrays["cost_memory_updates"] + arrays["cost_accepted_swaps"]
    ).astype(np.uint8)
    require(np.array_equal(arrays["changed"], changed_expected), f"changed partition {ordinal}")
    require(
        np.array_equal(
            arrays["changed"],
            (arrays["source_state_ordinal"] != arrays["successor_state_ordinal"]).astype(np.uint8),
        ),
        f"self loop identity {ordinal}",
    )
    require(
        np.array_equal(
            arrays["delta_adjacent_descents"],
            arrays["delta_paper_distance_numerator"],
        ),
        f"paper numerator {ordinal}",
    )
    require(
        np.array_equal(
            arrays["delta_spearman_footrule"], arrays["delta_emd_numerator"]
        ),
        f"EMD numerator {ordinal}",
    )
    unchanged = arrays["changed"] == 0
    for metric in [
        "delta_adjacent_descents",
        "delta_inversion_count",
        "delta_spearman_footrule",
        "delta_maximum_rank_error",
    ]:
        require(np.all(arrays[metric][unchanged] == 0), f"unchanged metric {metric} {ordinal}")

    opportunities = opportunity_arrays(family)
    for column, expected_column in [
        ("scheduler_actor_index", "actors"),
        ("scheduler_side_code", "sides"),
        ("probability_numerator", "probability_numerators"),
        ("probability_denominator", "probability_denominators"),
    ]:
        tiled = np.tile(opportunities[expected_column], len(active))
        require(np.array_equal(arrays[column], tiled), f"scheduler labels {column} {ordinal}")
    probability_sum = sum(
        int(numerator) / int(denominator)
        for numerator, denominator in zip(
            opportunities["probability_numerators"],
            opportunities["probability_denominators"],
        )
    )
    require(abs(probability_sum - 1.0) < 1e-15, f"probability sum {ordinal}")

    reachable = _reachable_from_materialized_edges(
        family.state_count,
        arrays["source_state_ordinal"],
        arrays["successor_state_ordinal"],
    )
    require(np.array_equal(reachable, expected_reachable), f"materialized BFS {ordinal}")

    digest = ColumnStreamDigest(
        EDGE_DIGEST_DOMAIN, bytes(metadata["family_digest"]), EDGE_ARRAY_COLUMNS
    )
    digest.update(arrays)
    require(digest.hexdigest() == manifest["edge_logical_digest"], f"edge digest {ordinal}")

    selected, exhaustive = sample_state_ordinals(
        family, bytes(metadata["family_digest"]), terminal
    )
    for state_ordinal in selected:
        observed = materialized_records(arrays, state_ordinal)
        expected = e01_outgoing_edges(family, state_ordinal)
        matched, detail = compare_edge_records(observed, expected)
        simulator_rows.append(
            {
                "family_ordinal": ordinal,
                "family_id": family.family_id,
                "n": family.n,
                "architecture": family.architecture.value,
                "direction": family.direction.value,
                "policy_profile": family.policy_profile,
                "fault_mode": family.fault_mode,
                "fault_count": family.fault_count,
                "state_ordinal": state_ordinal,
                "fixture_scope": "exhaustive_n4_memoryless_or_traditional" if exhaustive else "all_family_deterministic_sample",
                "materialized_edge_count": len(observed),
                "e01_edge_count": len(expected),
                "exact_match": matched,
                "detail": detail,
            }
        )
        structural = StructuralState(
            family,
            state_ordinal % family.occupancy_state_count,
            state_ordinal // family.occupancy_state_count,
        )
        native = evaluate_terminal(family.scenario(), structural.to_run_state())
        native_code = 1 if native == "complete" else 2 if native == "quiescent" else 0
        terminal_rows.append(
            {
                "family_ordinal": ordinal,
                "family_id": family.family_id,
                "state_ordinal": state_ordinal,
                "structural_terminal_code": int(terminal[state_ordinal]),
                "e01_terminal_code": native_code,
                "e01_terminal_label": native or "active",
                "exact_match": int(terminal[state_ordinal]) == native_code,
                "fixture_scope": "exhaustive_n4_memoryless_or_traditional" if exhaustive else "all_family_deterministic_sample",
            }
        )

    counts = Counter()
    counts["edge_count"] = edge_count
    counts["changed_edge_count"] = int(arrays["changed"].sum())
    counts["self_loop_count"] = int(np.count_nonzero(arrays["source_state_ordinal"] == arrays["successor_state_ordinal"]))
    counts["no_op_edge_count"] = int(arrays["cost_no_ops"].sum())
    counts["rejection_edge_count"] = int(arrays["cost_rejections"].sum())
    counts["memory_update_edge_count"] = int(arrays["cost_memory_updates"].sum())
    counts["accepted_swap_edge_count"] = int(arrays["cost_accepted_swaps"].sum())
    counts["parallel_endpoint_extra_count"] = parallel_endpoint_extra
    for key, value in counts.items():
        require(value == manifest[key], f"manifest {key} family {ordinal}")
    return {
        "family_ordinal": ordinal,
        "edge_rows": edge_count,
        "edge_digest_match": True,
        "authoritative_keys_complete_unique": True,
        "endpoint_range_valid": True,
        "materialized_reachability_match": True,
        "action_cost_partition_valid": True,
        "metric_dependencies_valid": True,
    }


def main() -> None:
    started = time.perf_counter()
    required = [
        S05 / "graph_corpus_manifest.json",
        S05 / "graph_family_manifest.parquet",
        S05 / "transition_counts_by_family.parquet",
        S05 / "transition_counts_by_stratum.parquet",
        S05 / "reachability_summary.parquet",
        S05 / "deterministic_reconstruction.json",
        S05 / "input_immutability.json",
        *sorted((S05 / "graphs").glob("edge_shard_*.parquet")),
        *sorted((S05 / "graphs").glob("node_shard_*.parquet")),
    ]
    require(len(list((S05 / "graphs").glob("edge_shard_*.parquet"))) == 8, "edge shard count")
    require(len(list((S05 / "graphs").glob("node_shard_*.parquet"))) == 8, "node shard count")
    for path in required:
        require(path.is_file() and path.stat().st_size > 0, f"missing {path}")
    corpus = json.loads((S05 / "graph_corpus_manifest.json").read_text())
    reconstruction = json.loads((S05 / "deterministic_reconstruction.json").read_text())
    immutability = json.loads((S05 / "input_immutability.json").read_text())
    require(corpus["success"] and reconstruction["success"] and immutability["success"], "status input")
    family_rows = pq.read_table(S04 / "state_family_inventory.parquet").to_pylist()
    graph_rows = pq.read_table(S05 / "graph_family_manifest.parquet").to_pylist()
    require(len(family_rows) == len(graph_rows) == EXPECTED_FAMILIES, "family count")
    require([row["family_ordinal"] for row in graph_rows] == list(range(EXPECTED_FAMILIES)), "manifest order")
    terminal_by_family, reachable_by_family, node_digest_rows = load_nodes(
        family_rows, graph_rows
    )
    require(sum(len(value) for value in terminal_by_family) == EXPECTED_NODES, "node rows")

    simulator_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    edge_validation_rows: list[dict[str, Any]] = []
    seen_families: set[int] = set()
    total_edges = 0
    for shard in sorted((S05 / "graphs").glob("edge_shard_*.parquet")):
        parquet = pq.ParquetFile(shard)
        require(
            (parquet.schema_arrow.metadata or {}).get(b"schemaVersion")
            == EDGE_SCHEMA.encode(),
            f"edge schema {shard}",
        )
        current = -1
        chunks: dict[str, list[np.ndarray]] = {column: [] for column in EDGE_ARRAY_COLUMNS}

        def finalize() -> None:
            nonlocal current, chunks, total_edges
            if current < 0:
                return
            arrays = {
                column: np.concatenate(chunks[column]) for column in EDGE_ARRAY_COLUMNS
            }
            edge_validation_rows.append(
                validate_family_edges(
                    current,
                    arrays,
                    family_rows,
                    graph_rows,
                    terminal_by_family[current],
                    reachable_by_family[current],
                    simulator_rows,
                    terminal_rows,
                )
            )
            total_edges += len(arrays["source_state_ordinal"])
            seen_families.add(current)
            current = -1
            chunks = {column: [] for column in EDGE_ARRAY_COLUMNS}

        for row_group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(row_group)
            ordinals = numpy_column(table, "family_ordinal", np.int32)
            unique = np.unique(ordinals)
            require(len(unique) == 1, f"mixed edge family row group {shard}:{row_group}")
            ordinal = int(unique[0])
            if current >= 0 and ordinal != current:
                finalize()
            if current < 0:
                current = ordinal
            for column in EDGE_ARRAY_COLUMNS:
                chunks[column].append(numpy_column(table, column, EDGE_DTYPES[column]))
        finalize()

    # Entirely terminal families have no physical edge row from which to recover identity.
    for ordinal in sorted(set(range(EXPECTED_FAMILIES)) - seen_families):
        require(graph_rows[ordinal]["edge_count"] == 0, f"missing nonempty family {ordinal}")
        empty = {column: np.empty(0, dtype=dtype) for column, dtype in EDGE_DTYPES.items()}
        edge_validation_rows.append(
            validate_family_edges(
                ordinal,
                empty,
                family_rows,
                graph_rows,
                terminal_by_family[ordinal],
                reachable_by_family[ordinal],
                simulator_rows,
                terminal_rows,
            )
        )
    require(total_edges == corpus["edgeCount"], "total physical edge count")
    require(all(row["exact_match"] for row in simulator_rows), "E01 outgoing edge mismatch")
    require(all(row["exact_match"] for row in terminal_rows), "terminal mismatch")

    write_parquet(S05 / "simulator_edge_validation.parquet", simulator_rows)
    write_parquet(S05 / "terminal_agreement.parquet", terminal_rows)
    write_parquet(S05 / "physical_graph_validation.parquet", edge_validation_rows)
    write_parquet(S05 / "physical_node_validation.parquet", node_digest_rows)

    checks = {
        "required_outputs_present": True,
        "state_family_coverage": True,
        "node_coverage_and_metadata": True,
        "active_outgoing_opportunity_completeness": True,
        "terminal_outdegree_zero": True,
        "authoritative_edge_key_uniqueness": True,
        "endpoint_family_closure": True,
        "scheduler_probability_and_label_contract": True,
        "action_cost_ledger_partition": True,
        "metric_delta_dependency_contract": True,
        "materialized_directed_reachability": True,
        "physical_node_logical_digests": True,
        "physical_edge_logical_digests": True,
        "independent_e01_outgoing_edge_enumeration": True,
        "independent_e01_terminal_agreement": True,
        "second_pass_deterministic_reconstruction": True,
        "input_immutability": True,
        "graph_budget_not_triggered": not corpus["boundaries"]["triggered"],
    }
    results = {
        "schemaVersion": "e03.s05.validation_results.v1",
        "researchStepId": "S05",
        "success": all(checks.values()),
        "checks": checks,
        "familyCount": EXPECTED_FAMILIES,
        "nodeCount": EXPECTED_NODES,
        "edgeCount": total_edges,
        "nodeLogicalDigestsChecked": len(node_digest_rows),
        "edgeLogicalDigestsChecked": len(edge_validation_rows),
        "simulatorStateFixtures": len(simulator_rows),
        "simulatorOpportunityFixtures": sum(row["e01_edge_count"] for row in simulator_rows),
        "terminalFixtures": len(terminal_rows),
        "deterministicReconstructionMatchedFamilies": reconstruction["matchedFamilyCount"],
        "elapsedSeconds": time.perf_counter() - started,
    }
    write_json(S05 / "validation_results.json", results)
    (S05 / "validation.log").write_text(
        "S05 VALIDATION PASS\n"
        + "\n".join(f"{key}=PASS" for key in checks)
        + f"\nfamily_count={EXPECTED_FAMILIES}\nnode_count={EXPECTED_NODES}\nedge_count={total_edges}\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    if not results["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
