#!/usr/bin/env python3
"""Solve every frozen S05 family for S07 and materialize compact all-state tables."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from typing import Any

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
    PROFILE_NAMES,
    PROFILE_OBSERVATION_READS,
    PROFILE_VALUE_COMPARISONS,
    estimate_working_bytes,
    solve_all_starts_lexicographic,
)
from src.detours.state_space import FamilySpec


S04 = Path("/artifacts/research_steps/S04")
S05 = Path("/artifacts/research_steps/S05")
S06 = Path("/artifacts/research_steps/S06")
OUTPUT = Path("/artifacts/research_steps/S07")
CACHE = Path("/cache/e03_s07_solutions")
REPOSITORY = Path(__file__).resolve().parents[1]
WORKERS = 8
FINAL_BUDGET = 12 * 1024**3
SCRATCH_BUDGET = 48 * 1024**3
FAMILY_BUDGET = 4 * 1024**3

METRICS = (
    "adjacent_descents",
    "inversion_count",
    "spearman_footrule",
    "maximum_rank_error",
)
METRIC_CODES = {name: index for index, name in enumerate(METRICS)}
METRIC_DENOMINATORS = {
    "adjacent_descents": "normalized_adjacent_descents_denominator",
    "inversion_count": "normalized_kendall_distance_denominator",
    "spearman_footrule": "normalized_spearman_footrule_denominator",
    "maximum_rank_error": "normalized_maximum_rank_error_denominator",
}
COST_COLUMNS = (
    "observation_reads",
    "value_comparisons",
    "cost_no_ops",
    "cost_rejections",
    "cost_memory_updates",
    "cost_accepted_swaps",
    "cost_displaced_cells",
)
PROFILES = (
    PROFILE_ACTIVATIONS,
    PROFILE_OBSERVATION_READS,
    PROFILE_VALUE_COMPARISONS,
    PROFILE_ACCEPTED_SWAPS,
    PROFILE_FULL_LEDGER,
)

PATH_SCHEMA = pa.schema(
    [
        pa.field("family_ordinal", pa.int32(), nullable=False),
        pa.field("state_ordinal", pa.uint32(), nullable=False),
        pa.field("metric_code", pa.uint8(), nullable=False),
        pa.field("terminal_code", pa.uint8(), nullable=False),
        pa.field("metric_level", pa.int16(), nullable=False),
        pa.field("minimum_peak", pa.int16(), nullable=False),
        pa.field("minimum_excursion", pa.int16(), nullable=False),
        pa.field("classification_code", pa.uint8(), nullable=False),
        pa.field("primary_successor_edge_local_index", pa.int32(), nullable=False),
    ],
    metadata={
        b"schemaVersion": b"e03.s07.path_solutions.v1",
        b"researchStepId": b"S07",
        b"metricCodes": json.dumps(METRIC_CODES, sort_keys=True).encode(),
        b"classCodes": b'{"0":"complete_start","1":"reachable_no_detour","2":"necessary_detour","3":"unreachable_active","4":"quiescent"}',
        b"undefinedSentinel": b"-1",
        b"schedulerQuantification": b"existential finite S05 structural opportunity path",
    },
)

COST_FIELDS = [
    pa.field("family_ordinal", pa.int32(), nullable=False),
    pa.field("state_ordinal", pa.uint32(), nullable=False),
    pa.field("terminal_code", pa.uint8(), nullable=False),
    pa.field("shortest_activations", pa.int32(), nullable=False),
    pa.field("shortest_successor_edge_local_index", pa.int32(), nullable=False),
    pa.field("minimum_observation_reads", pa.int32(), nullable=False),
    pa.field("minimum_observation_reads_activations", pa.int32(), nullable=False),
    pa.field("minimum_observation_reads_successor_edge_local_index", pa.int32(), nullable=False),
    pa.field("minimum_value_comparisons", pa.int32(), nullable=False),
    pa.field("minimum_value_comparisons_activations", pa.int32(), nullable=False),
    pa.field("minimum_value_comparisons_successor_edge_local_index", pa.int32(), nullable=False),
    pa.field("minimum_accepted_swaps", pa.int32(), nullable=False),
    pa.field("minimum_accepted_swaps_activations", pa.int32(), nullable=False),
    pa.field("minimum_accepted_swaps_successor_edge_local_index", pa.int32(), nullable=False),
    pa.field("minimum_displaced_cells", pa.int32(), nullable=False),
    pa.field("minimum_displaced_cells_activations", pa.int32(), nullable=False),
    pa.field("minimum_displaced_cells_successor_edge_local_index", pa.int32(), nullable=False),
]
for name in (
    "activations",
    "observation_reads",
    "value_comparisons",
    "no_ops",
    "rejections",
    "memory_updates",
    "accepted_swaps",
    "displaced_cells",
):
    COST_FIELDS.append(pa.field(f"minimum_full_ledger_{name}", pa.int32(), nullable=False))
COST_FIELDS.append(
    pa.field("minimum_full_ledger_successor_edge_local_index", pa.int32(), nullable=False)
)
COST_SCHEMA = pa.schema(
    COST_FIELDS,
    metadata={
        b"schemaVersion": b"e03.s07.minimum_cost_solutions.v1",
        b"researchStepId": b"S07",
        b"undefinedSentinel": b"-1",
        b"costOrder": b"lexicographic",
        b"schedulerQuantification": b"existential finite S05 structural opportunity path",
    },
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


def read_family(
    parquet: pq.ParquetFile,
    groups: dict[int, list[int]],
    ordinal: int,
    columns: list[str],
) -> pa.Table:
    selected = groups.get(ordinal, [])
    if not selected:
        return pa.table({name: pa.array([], type=parquet.schema_arrow.field(name).type) for name in columns})
    table = parquet.read_row_groups(selected, columns=columns).combine_chunks()
    if table.num_rows and np.any(table["family_ordinal"].to_numpy() != ordinal):
        raise AssertionError("family row-group map returned a foreign family")
    return table


def family_from_row(row: dict[str, Any]) -> FamilySpec:
    return FamilySpec.from_canonical_dict(json.loads(row["canonical_family_json"]))


def serialize_int32(values: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(values, dtype=np.int64)
    result = source.copy()
    result[result == UNREACHABLE] = -1
    finite = result[result >= 0]
    if finite.size and int(finite.max()) > np.iinfo(np.int32).max:
        raise OverflowError(f"{name} exceeds int32 output domain")
    return np.ascontiguousarray(result, dtype=np.int32)


def arrow_table(schema: pa.Schema, values: dict[str, np.ndarray]) -> pa.Table:
    arrays = [pa.array(values[field.name], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def solve_worker(shard: int) -> dict[str, Any]:
    started = time.perf_counter()
    edge_path = S05 / f"graphs/edge_shard_{shard:02d}.parquet"
    node_path = S05 / f"graphs/node_shard_{shard:02d}.parquet"
    edge_pf, edge_groups = row_group_map(edge_path)
    node_pf, node_groups = row_group_map(node_path)
    family_table = pq.read_table(S04 / "state_family_inventory.parquet").to_pandas()
    graph_table = pq.read_table(S05 / "graph_family_manifest.parquet").to_pandas()
    families = family_table[family_table.family_ordinal % WORKERS == shard].sort_values(
        "family_ordinal"
    )
    graph_rows = graph_table.set_index("family_ordinal").to_dict("index")

    scratch_path = CACHE / f"path_solution_shard_{shard:02d}.parquet"
    scratch_cost = CACHE / f"cost_solution_shard_{shard:02d}.parquet"
    path_writer = pq.ParquetWriter(
        scratch_path,
        PATH_SCHEMA,
        compression="zstd",
        compression_level=7,
        use_dictionary=["metric_code", "terminal_code", "classification_code"],
        write_statistics=True,
    )
    cost_writer = pq.ParquetWriter(
        scratch_cost,
        COST_SCHEMA,
        compression="zstd",
        compression_level=7,
        use_dictionary=["terminal_code"],
        write_statistics=True,
    )
    summaries: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    cost_row_offset = 0
    path_row_offset = 0
    edge_columns = ["family_ordinal", "source_state_ordinal", "successor_state_ordinal", *COST_COLUMNS]
    node_columns = ["family_ordinal", "state_ordinal", "terminal_code"]
    try:
        for family_position, metadata in enumerate(families.to_dict("records")):
            ordinal = int(metadata["family_ordinal"])
            graph = graph_rows[ordinal]
            family = family_from_row(metadata)
            nodes = read_family(node_pf, node_groups, ordinal, node_columns)
            edges = read_family(edge_pf, edge_groups, ordinal, edge_columns)
            state_ordinals = nodes["state_ordinal"].to_numpy(zero_copy_only=False)
            terminal = nodes["terminal_code"].to_numpy(zero_copy_only=False)
            sources = edges["source_state_ordinal"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            targets = edges["successor_state_ordinal"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            costs = {name: edges[name].to_numpy(zero_copy_only=False) for name in COST_COLUMNS}
            node_count = len(terminal)
            edge_count = len(sources)
            if node_count != int(graph["node_count"]) or edge_count != int(graph["edge_count"]):
                raise AssertionError(f"family {ordinal} physical count mismatch")
            if not np.array_equal(state_ordinals, np.arange(node_count, dtype=np.uint32)):
                raise AssertionError(f"family {ordinal} state ordinal order mismatch")
            estimate = estimate_working_bytes(node_count, edge_count)
            if estimate > FAMILY_BUDGET:
                raise MemoryError(f"family {ordinal} projected {estimate} bytes exceeds budget")

            family_started = time.perf_counter()
            cost_solutions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for profile in PROFILES:
                solved = solve_all_starts_lexicographic(
                    node_count, sources, targets, terminal, costs, profile
                )
                labels = serialize_int32(solved.labels, f"family {ordinal} profile {profile}")
                successor = serialize_int32(
                    solved.successor_edges, f"family {ordinal} profile {profile} successor"
                )
                cost_solutions[profile] = (labels, successor)
            reachable_masks = [values[0][0] >= 0 for values in cost_solutions.values()]
            if any(not np.array_equal(reachable_masks[0], mask) for mask in reachable_masks[1:]):
                raise AssertionError(f"family {ordinal} cost-profile reachability mismatch")
            activation_labels, activation_successor = cost_solutions[PROFILE_ACTIVATIONS]
            reads_labels, reads_successor = cost_solutions[PROFILE_OBSERVATION_READS]
            comparisons_labels, comparisons_successor = cost_solutions[PROFILE_VALUE_COMPARISONS]
            swaps_labels, swaps_successor = cost_solutions[PROFILE_ACCEPTED_SWAPS]
            full_labels, full_successor = cost_solutions[PROFILE_FULL_LEDGER]
            if not np.array_equal(full_labels[0], activation_labels[0]):
                raise AssertionError(f"family {ordinal} full-ledger activations are not shortest")
            cost_values = {
                "family_ordinal": np.full(node_count, ordinal, dtype=np.int32),
                "state_ordinal": state_ordinals,
                "terminal_code": terminal,
                "shortest_activations": activation_labels[0],
                "shortest_successor_edge_local_index": activation_successor,
                "minimum_observation_reads": reads_labels[0],
                "minimum_observation_reads_activations": reads_labels[1],
                "minimum_observation_reads_successor_edge_local_index": reads_successor,
                "minimum_value_comparisons": comparisons_labels[0],
                "minimum_value_comparisons_activations": comparisons_labels[1],
                "minimum_value_comparisons_successor_edge_local_index": comparisons_successor,
                "minimum_accepted_swaps": swaps_labels[0],
                "minimum_accepted_swaps_activations": swaps_labels[1],
                "minimum_accepted_swaps_successor_edge_local_index": swaps_successor,
                "minimum_displaced_cells": np.where(swaps_labels[0] >= 0, 2 * swaps_labels[0], -1).astype(np.int32),
                "minimum_displaced_cells_activations": swaps_labels[1],
                "minimum_displaced_cells_successor_edge_local_index": swaps_successor,
            }
            full_names = (
                "activations",
                "observation_reads",
                "value_comparisons",
                "no_ops",
                "rejections",
                "memory_updates",
                "accepted_swaps",
                "displaced_cells",
            )
            for index, name in enumerate(full_names):
                cost_values[f"minimum_full_ledger_{name}"] = full_labels[index]
            cost_values["minimum_full_ledger_successor_edge_local_index"] = full_successor
            cost_table = arrow_table(COST_SCHEMA, cost_values)
            cost_writer.write_table(cost_table, row_group_size=max(1, node_count))
            cost_digest = hash_arrays(
                terminal,
                activation_labels,
                activation_successor,
                reads_labels,
                reads_successor,
                comparisons_labels,
                comparisons_successor,
                swaps_labels,
                swaps_successor,
                full_labels,
                full_successor,
            )
            del cost_values, cost_table, cost_solutions

            path_digests: dict[str, str] = {}
            metric_offsets: dict[str, int] = {}
            for metric in METRICS:
                levels = family_metric_levels(family, metric)
                primary = solve_primary_all_starts(
                    node_count, sources, targets, levels, terminal
                )
                bottleneck = serialize_int32(primary.bottleneck_levels, "minimum_peak")
                excursion = serialize_int32(primary.excursion_levels, "minimum_excursion")
                successor = serialize_int32(primary.successor_edges, "primary successor")
                classification = np.asarray(primary.classifications, dtype=np.uint8)
                if not np.array_equal(bottleneck >= 0, reachable_masks[0]):
                    raise AssertionError(f"family {ordinal} {metric} reachability disagrees with cost solver")
                if np.any(levels > np.iinfo(np.int16).max):
                    raise OverflowError("metric level exceeds int16")
                path_values = {
                    "family_ordinal": np.full(node_count, ordinal, dtype=np.int32),
                    "state_ordinal": state_ordinals,
                    "metric_code": np.full(node_count, METRIC_CODES[metric], dtype=np.uint8),
                    "terminal_code": terminal,
                    "metric_level": levels.astype(np.int16),
                    "minimum_peak": bottleneck.astype(np.int16),
                    "minimum_excursion": excursion.astype(np.int16),
                    "classification_code": classification,
                    "primary_successor_edge_local_index": successor,
                }
                path_table = arrow_table(PATH_SCHEMA, path_values)
                path_writer.write_table(path_table, row_group_size=max(1, node_count))
                metric_offsets[metric] = path_row_offset
                path_row_offset += node_count
                path_digests[metric] = hash_arrays(
                    terminal,
                    levels.astype(np.int16),
                    bottleneck.astype(np.int16),
                    excursion.astype(np.int16),
                    classification,
                    successor,
                )
                class_counts = np.bincount(classification, minlength=5)
                reachable_active = int(
                    class_counts[CLASS_REACHABLE_NO_DETOUR]
                    + class_counts[CLASS_NECESSARY_DETOUR]
                )
                defined_excursion = excursion[
                    (classification == CLASS_REACHABLE_NO_DETOUR)
                    | (classification == CLASS_NECESSARY_DETOUR)
                ]
                necessary_indices = np.flatnonzero(
                    classification == CLASS_NECESSARY_DETOUR
                )
                no_detour_indices = np.flatnonzero(
                    classification == CLASS_REACHABLE_NO_DETOUR
                )
                unreachable_indices = np.flatnonzero(
                    classification == CLASS_UNREACHABLE_ACTIVE
                )
                quiescent_indices = np.flatnonzero(classification == CLASS_QUIESCENT)
                deepest_state = -1
                if len(necessary_indices):
                    depths = excursion[necessary_indices]
                    deepest_state = int(necessary_indices[int(np.argmax(depths))])
                denominator = int(graph[METRIC_DENOMINATORS[metric]])
                summaries.append(
                    {
                        "family_ordinal": ordinal,
                        "metric": metric,
                        "metric_code": METRIC_CODES[metric],
                        "materialization_tier": metadata["materialization_tier"],
                        "n": int(metadata["n"]),
                        "architecture": metadata["architecture"],
                        "direction": metadata["direction"],
                        "policy_code": int(metadata["policy_code"]),
                        "policy_profile": metadata["policy_profile"],
                        "selection_owner_count": int(metadata["selection_owner_count"]),
                        "fault_code": int(metadata["fault_code"]),
                        "fault_mode": metadata["fault_mode"],
                        "fault_count": int(metadata["fault_count"]),
                        "fault_identity_ids": ",".join(metadata["fault_identity_ids"]),
                        "node_count": node_count,
                        "complete_start_count": int(class_counts[CLASS_COMPLETE_START]),
                        "reachable_no_detour_count": int(class_counts[CLASS_REACHABLE_NO_DETOUR]),
                        "necessary_detour_count": int(class_counts[CLASS_NECESSARY_DETOUR]),
                        "unreachable_active_count": int(class_counts[CLASS_UNREACHABLE_ACTIVE]),
                        "quiescent_count": int(class_counts[CLASS_QUIESCENT]),
                        "goal_reachable_count": int(
                            class_counts[CLASS_COMPLETE_START] + reachable_active
                        ),
                        "reachable_active_count": reachable_active,
                        "necessary_prevalence_reachable_active": (
                            float(class_counts[CLASS_NECESSARY_DETOUR] / reachable_active)
                            if reachable_active
                            else float("nan")
                        ),
                        "mean_excursion_reachable_active": (
                            float(defined_excursion.mean()) if len(defined_excursion) else float("nan")
                        ),
                        "maximum_excursion": int(defined_excursion.max()) if len(defined_excursion) else -1,
                        "maximum_normalized_excursion": (
                            float(defined_excursion.max() / denominator)
                            if len(defined_excursion)
                            else float("nan")
                        ),
                        "metric_denominator": denominator,
                        "anchor_classification_code": int(classification[0]),
                        "anchor_minimum_excursion": int(excursion[0]),
                        "first_reachable_no_detour_state": int(no_detour_indices[0]) if len(no_detour_indices) else -1,
                        "first_necessary_detour_state": int(necessary_indices[0]) if len(necessary_indices) else -1,
                        "deepest_necessary_detour_state": deepest_state,
                        "first_unreachable_active_state": int(unreachable_indices[0]) if len(unreachable_indices) else -1,
                        "first_quiescent_state": int(quiescent_indices[0]) if len(quiescent_indices) else -1,
                    }
                )
                del primary, bottleneck, excursion, successor, classification, path_values, path_table

            manifests.append(
                {
                    "family_ordinal": ordinal,
                    "worker_shard": shard,
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "cost_row_offset_in_worker": cost_row_offset,
                    "path_row_offset_in_worker": metric_offsets[METRICS[0]],
                    "cost_logical_digest": cost_digest,
                    **{f"path_{metric}_logical_digest": path_digests[metric] for metric in METRICS},
                    "estimated_working_bytes": estimate,
                    "family_runtime_seconds": time.perf_counter() - family_started,
                }
            )
            cost_row_offset += node_count
            if family_position % 100 == 0:
                print(
                    json.dumps(
                        {
                            "shard": shard,
                            "familyPosition": family_position,
                            "familyOrdinal": ordinal,
                            "nodes": node_count,
                            "edges": edge_count,
                        }
                    ),
                    flush=True,
                )
    finally:
        path_writer.close()
        cost_writer.close()

    summary_path = CACHE / f"prevalence_shard_{shard:02d}.parquet"
    manifest_path = CACHE / f"manifest_shard_{shard:02d}.parquet"
    pq.write_table(pa.Table.from_pylist(summaries), summary_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(manifests), manifest_path, compression="zstd")
    return {
        "shard": shard,
        "families": len(manifests),
        "costRows": cost_row_offset,
        "pathRows": path_row_offset,
        "runtimeSeconds": time.perf_counter() - started,
        "path": str(scratch_path),
        "cost": str(scratch_cost),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
    }


def merge_parquets(inputs: list[Path], output: Path, schema: pa.Schema) -> None:
    writer = pq.ParquetWriter(
        output,
        schema,
        compression="zstd",
        compression_level=7,
        use_dictionary=True,
        write_statistics=True,
    )
    try:
        for path in inputs:
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                writer.write_table(parquet.read_row_group(row_group))
    finally:
        writer.close()


def aggregate_strata(frame: pd.DataFrame) -> pd.DataFrame:
    count_columns = [
        "node_count",
        "complete_start_count",
        "reachable_no_detour_count",
        "necessary_detour_count",
        "unreachable_active_count",
        "quiescent_count",
        "goal_reachable_count",
        "reachable_active_count",
    ]
    specifications: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("n", ["n"]),
        ("materialization_tier", ["materialization_tier"]),
        ("architecture", ["architecture"]),
        ("direction", ["direction"]),
        ("policy_profile", ["policy_profile"]),
        ("policy_assignment", ["n", "policy_code"]),
        ("selection_owner_count", ["selection_owner_count"]),
        ("fault_mode", ["fault_mode"]),
        ("fault_count", ["fault_count"]),
        ("fault_placement", ["n", "fault_code"]),
        ("fault_identity_ids", ["n", "fault_identity_ids"]),
        ("architecture_policy", ["architecture", "policy_profile"]),
        ("policy_fault", ["policy_profile", "fault_mode", "fault_count"]),
        (
            "planned_regime",
            [
                "n",
                "architecture",
                "direction",
                "policy_profile",
                "fault_mode",
                "fault_count",
            ],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for stratum_type, columns in specifications:
        keys = ["metric", *columns]
        grouped = frame.groupby(keys, dropna=False, sort=True) if columns else frame.groupby(["metric"], sort=True)
        for key, group in grouped:
            key_values = key if isinstance(key, tuple) else (key,)
            metric = key_values[0]
            values = key_values[1:]
            totals = group[count_columns].sum()
            reachable_active = int(totals["reachable_active_count"])
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "stratum_value": (
                        "all"
                        if not columns
                        else "|".join(f"{name}={value}" for name, value in zip(columns, values, strict=True))
                    ),
                    "metric": metric,
                    "family_count": int(len(group)),
                    **{name: int(totals[name]) for name in count_columns},
                    "necessary_prevalence_reachable_active": (
                        float(totals["necessary_detour_count"] / reachable_active)
                        if reachable_active
                        else float("nan")
                    ),
                    "maximum_excursion": int(group["maximum_excursion"].max()),
                    "maximum_normalized_excursion": float(group["maximum_normalized_excursion"].max()),
                }
            )
    return pd.DataFrame(rows)


def direct_inputs() -> list[Path]:
    return [
        S04 / "state_family_inventory.parquet",
        S05 / "graph_family_manifest.parquet",
        S05 / "graph_corpus_manifest.json",
        S06 / "necessary_detour_spec.md",
        S06 / "solver_validation_fixtures.json",
        REPOSITORY / "analysis/s06_necessary_detour_contract.json",
        REPOSITORY / "analysis/s07_path_solution_contract.json",
        REPOSITORY / "src/detours/necessary_detour.py",
        REPOSITORY / "src/detours/path_solutions.py",
        *[S05 / f"graphs/edge_shard_{index:02d}.parquet" for index in range(WORKERS)],
        *[S05 / f"graphs/node_shard_{index:02d}.parquet" for index in range(WORKERS)],
    ]


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"refusing to mix a new S07 build with {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True)
    missing = [str(path) for path in direct_inputs() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    input_hashes = {str(path): sha256_file(path) for path in direct_inputs()}
    write_json(CACHE / "input_hashes_before.json", input_hashes)

    family_frame = pq.read_table(S05 / "graph_family_manifest.parquet").to_pandas()
    projected_rows = int(family_frame.node_count.sum()) * len(METRICS)
    projected_working = int(
        max(estimate_working_bytes(int(row.node_count), int(row.edge_count)) for row in family_frame.itertuples())
    )
    preflight = {
        "schemaVersion": "e03.s07.preflight.v1",
        "researchStepId": "S07",
        "families": int(len(family_frame)),
        "states": int(family_frame.node_count.sum()),
        "stateMetricRows": projected_rows,
        "edges": int(family_frame.edge_count.sum()),
        "workers": WORKERS,
        "maximumEstimatedPerFamilyWorkingBytes": projected_working,
        "perFamilyBudgetBytes": FAMILY_BUDGET,
        "finalBudgetBytes": FINAL_BUDGET,
        "scratchBudgetBytes": SCRATCH_BUDGET,
        "passed": projected_working <= FAMILY_BUDGET,
    }
    write_json(OUTPUT / "preflight.json", preflight)
    if not preflight["passed"]:
        raise MemoryError(preflight)

    # Compile both search kernels before forking so all workers inherit them.
    tiny_edges = np.asarray([0], dtype=np.int64)
    tiny_terminal = np.asarray([0, 1], dtype=np.uint8)
    tiny_costs = {name: np.asarray([0], dtype=np.int64) for name in COST_COLUMNS}
    solve_all_starts_lexicographic(
        2, tiny_edges, np.asarray([1], dtype=np.int64), tiny_terminal, tiny_costs, PROFILE_FULL_LEDGER
    )

    started = time.perf_counter()
    context = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        futures = [pool.submit(solve_worker, shard) for shard in range(WORKERS)]
        worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]
    worker_results.sort(key=lambda row: row["shard"])
    scratch_bytes = sum(
        Path(row[key]).stat().st_size
        for row in worker_results
        for key in ("path", "cost", "summary", "manifest")
    )
    if scratch_bytes > SCRATCH_BUDGET:
        raise RuntimeError(f"scratch output {scratch_bytes} exceeds frozen S07 budget")

    merge_parquets(
        [Path(row["path"]) for row in worker_results],
        OUTPUT / "path_solutions.parquet",
        PATH_SCHEMA,
    )
    merge_parquets(
        [Path(row["cost"]) for row in worker_results],
        OUTPUT / "minimum_cost_solutions.parquet",
        COST_SCHEMA,
    )
    summary_tables = [pq.read_table(row["summary"]) for row in worker_results]
    summary = pa.concat_tables(summary_tables).to_pandas().sort_values(
        ["family_ordinal", "metric_code"]
    )
    pq.write_table(
        pa.Table.from_pandas(summary, preserve_index=False),
        OUTPUT / "necessity_prevalence_by_family.parquet",
        compression="zstd",
        compression_level=7,
    )
    strata = aggregate_strata(summary)
    pq.write_table(
        pa.Table.from_pandas(strata, preserve_index=False),
        OUTPUT / "necessity_prevalence_by_stratum.parquet",
        compression="zstd",
        compression_level=7,
    )

    manifest_tables = [pq.read_table(row["manifest"]) for row in worker_results]
    manifest_frame = pa.concat_tables(manifest_tables).to_pandas().sort_values("family_ordinal")
    cost_bases: dict[int, int] = {}
    path_bases: dict[int, int] = {}
    cost_running = 0
    path_running = 0
    for row in worker_results:
        cost_bases[int(row["shard"])] = cost_running
        path_bases[int(row["shard"])] = path_running
        cost_running += int(row["costRows"])
        path_running += int(row["pathRows"])
    manifest_frame["cost_row_offset"] = manifest_frame.apply(
        lambda row: cost_bases[int(row.worker_shard)] + int(row.cost_row_offset_in_worker), axis=1
    )
    manifest_frame["path_row_offset"] = manifest_frame.apply(
        lambda row: path_bases[int(row.worker_shard)] + int(row.path_row_offset_in_worker), axis=1
    )
    manifest_frame["path_row_count"] = manifest_frame.node_count * len(METRICS)
    manifest_frame["cost_row_count"] = manifest_frame.node_count
    pq.write_table(
        pa.Table.from_pandas(manifest_frame, preserve_index=False),
        OUTPUT / "solution_family_manifest.parquet",
        compression="zstd",
        compression_level=7,
    )

    final_paths = [
        OUTPUT / "path_solutions.parquet",
        OUTPUT / "minimum_cost_solutions.parquet",
        OUTPUT / "necessity_prevalence_by_family.parquet",
        OUTPUT / "necessity_prevalence_by_stratum.parquet",
        OUTPUT / "solution_family_manifest.parquet",
        OUTPUT / "preflight.json",
    ]
    final_bytes = sum(path.stat().st_size for path in final_paths)
    if final_bytes > FINAL_BUDGET:
        raise RuntimeError(f"S07 partial final output {final_bytes} exceeds frozen budget")
    build_result = {
        "schemaVersion": "e03.s07.solution_build.v1",
        "researchStepId": "S07",
        "success": True,
        "workers": WORKERS,
        "families": int(len(manifest_frame)),
        "states": int(cost_running),
        "stateMetricRows": int(path_running),
        "edges": int(family_frame.edge_count.sum()),
        "workerResults": worker_results,
        "scratchBytes": scratch_bytes,
        "finalBytesBeforeWitnessAndReports": final_bytes,
        "runtimeSeconds": time.perf_counter() - started,
        "inputHashCount": len(input_hashes),
        "metricCodes": METRIC_CODES,
        "profileCodes": PROFILE_NAMES,
    }
    write_json(OUTPUT / "solution_build.json", build_result)
    print(json.dumps(build_result, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    main()
