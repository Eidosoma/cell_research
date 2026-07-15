#!/usr/bin/env python3
"""Materialize complete E03 S05 directed legal-opportunity graphs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes
from src.detours.state_space import FamilySpec
from src.detours.transition_graph import (
    DECISION_LABELS,
    EDGE_ARRAY_COLUMNS,
    EDGE_DIGEST_DOMAIN,
    EDGE_SCHEMA,
    NODE_DIGEST_DOMAIN,
    NODE_SCHEMA,
    PROPOSAL_KIND_LABELS,
    REASON_LABELS,
    TERMINAL_LABELS,
    classify_states,
    edge_batch,
    metric_denominators,
    opportunity_arrays,
    reachable_states,
    warm_graph_kernels,
)


OUTPUT = Path("/artifacts/research_steps/S05")
SCRATCH = Path("/cache/e03_s05_graphs")
S04 = Path("/artifacts/research_steps/S04")
REPOSITORY = Path(__file__).resolve().parents[1]
E01 = Path("/previous-artifacts/E01")
CONTRACT = REPOSITORY / "analysis/s05_graph_contract.json"
STARTING_COMMIT = "87f59d788fd9a9405c53d15e1a60152611b79b78"
WORKERS = 8
STATE_BATCH = 32_768
ROW_GROUP = 65_536
MAX_EDGE_ROWS = 150_000_000
MAX_FAMILY_EDGES = 10_000_000
MAX_GRAPH_BYTES = 16_106_127_360


EDGE_SCHEMA_ARROW = pa.schema(
    [
        pa.field("family_ordinal", pa.int32(), nullable=False),
        pa.field("source_state_ordinal", pa.uint32(), nullable=False),
        pa.field("successor_state_ordinal", pa.uint32(), nullable=False),
        pa.field("opportunity_ordinal", pa.uint8(), nullable=False),
        pa.field("scheduler_actor_index", pa.int8(), nullable=False),
        pa.field("scheduler_side_code", pa.int8(), nullable=False),
        pa.field("probability_numerator", pa.uint8(), nullable=False),
        pa.field("probability_denominator", pa.uint8(), nullable=False),
        pa.field("proposal_kind_code", pa.uint8(), nullable=False),
        pa.field("proposal_reason_code", pa.uint8(), nullable=False),
        pa.field("decision_code", pa.uint8(), nullable=False),
        pa.field("proposal_actor_index", pa.int8(), nullable=False),
        pa.field("actor_position", pa.int8(), nullable=False),
        pa.field("target_position", pa.int8(), nullable=False),
        pa.field("new_cursor", pa.int8(), nullable=False),
        pa.field("changed", pa.uint8(), nullable=False),
        pa.field("observation_reads", pa.uint16(), nullable=False),
        pa.field("value_comparisons", pa.uint16(), nullable=False),
        pa.field("cost_no_ops", pa.uint8(), nullable=False),
        pa.field("cost_rejections", pa.uint8(), nullable=False),
        pa.field("cost_memory_updates", pa.uint8(), nullable=False),
        pa.field("cost_accepted_swaps", pa.uint8(), nullable=False),
        pa.field("cost_displaced_cells", pa.uint8(), nullable=False),
        pa.field("delta_adjacent_descents", pa.int8(), nullable=False),
        pa.field("delta_paper_distance_numerator", pa.int8(), nullable=False),
        pa.field("delta_inversion_count", pa.int16(), nullable=False),
        pa.field("delta_spearman_footrule", pa.int16(), nullable=False),
        pa.field("delta_maximum_rank_error", pa.int8(), nullable=False),
        pa.field("delta_emd_numerator", pa.int16(), nullable=False),
    ],
    metadata={
        b"schemaVersion": EDGE_SCHEMA.encode(),
        b"researchStepId": b"S05",
        b"authoritativeKey": b"family_ordinal,source_state_ordinal,opportunity_ordinal",
        b"stateJoin": b"state_ordinal=selection_cursor_code*factorial(n)+occupancy_rank",
        b"schedulerProjection": b"fair_serial_legal_opportunity_set_v1",
    },
)

NODE_SCHEMA_ARROW = pa.schema(
    [
        pa.field("family_ordinal", pa.int32(), nullable=False),
        pa.field("state_ordinal", pa.uint32(), nullable=False),
        pa.field("terminal_code", pa.uint8(), nullable=False),
        pa.field("reachable_from_anchor", pa.uint8(), nullable=False),
    ],
    metadata={
        b"schemaVersion": NODE_SCHEMA.encode(),
        b"researchStepId": b"S05",
        b"terminalCodes": json.dumps(TERMINAL_LABELS, sort_keys=True).encode(),
        b"anchor": b"S04 scenario anchor state ordinal 0",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: Sequence[dict[str, Any]] | pd.DataFrame) -> None:
    table = (
        pa.Table.from_pandas(rows, preserve_index=False)
        if isinstance(rows, pd.DataFrame)
        else pa.Table.from_pylist(list(rows))
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


class ColumnStreamDigest:
    """Chunk-boundary-independent logical digest for one family table."""

    def __init__(self, domain: bytes, family_digest: bytes, columns: Sequence[str]) -> None:
        self.domain = domain
        self.family_digest = family_digest
        self.columns = tuple(columns)
        self.hashers = {column: hashlib.sha256() for column in self.columns}
        self.rows = 0

    def update(self, arrays: dict[str, np.ndarray]) -> None:
        lengths = {len(arrays[column]) for column in self.columns}
        if len(lengths) != 1:
            raise AssertionError("digest arrays have inconsistent lengths")
        count = lengths.pop()
        self.rows += count
        for column in self.columns:
            self.hashers[column].update(np.ascontiguousarray(arrays[column]).tobytes())

    def hexdigest(self) -> str:
        digest = hashlib.sha256(self.domain + self.family_digest + struct.pack(">Q", self.rows))
        for column in self.columns:
            encoded = column.encode()
            digest.update(struct.pack(">H", len(encoded)))
            digest.update(encoded)
            digest.update(self.hashers[column].digest())
        return digest.hexdigest()


def family_from_row(row: dict[str, Any]) -> FamilySpec:
    return FamilySpec.from_canonical_dict(json.loads(row["canonical_family_json"]))


def edge_table(family_ordinal: int, arrays: dict[str, np.ndarray]) -> pa.Table:
    count = len(arrays["source_state_ordinal"])
    values = [
        pa.array(np.full(count, family_ordinal, dtype=np.int32), type=pa.int32())
    ]
    for field_index in range(1, len(EDGE_SCHEMA_ARROW)):
        field = EDGE_SCHEMA_ARROW.field(field_index)
        values.append(pa.array(arrays[field.name], type=field.type))
    return pa.Table.from_arrays(values, schema=EDGE_SCHEMA_ARROW)


def node_table(
    family_ordinal: int,
    start: int,
    terminal: np.ndarray,
    reachable: np.ndarray,
) -> pa.Table:
    count = len(terminal)
    arrays = [
        pa.array(np.full(count, family_ordinal, dtype=np.int32), type=pa.int32()),
        pa.array(np.arange(start, start + count, dtype=np.uint32), type=pa.uint32()),
        pa.array(terminal, type=pa.uint8()),
        pa.array(reachable, type=pa.uint8()),
    ]
    return pa.Table.from_arrays(arrays, schema=NODE_SCHEMA_ARROW)


def _parallel_endpoint_extra(successors: np.ndarray, slots: int) -> int:
    if len(successors) == 0:
        return 0
    if len(successors) % slots:
        raise AssertionError("active-state edge groups are incomplete")
    reshaped = successors.reshape((-1, slots)).copy()
    reshaped.sort(axis=1)
    return int(np.count_nonzero(reshaped[:, 1:] == reshaped[:, :-1]))


def _worker(args: tuple[int, list[dict[str, Any]], str]) -> dict[str, Any]:
    worker, family_rows, scratch_string = args
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = "1"
    scratch = Path(scratch_string)
    edge_path = scratch / f"edge_shard_{worker:02d}.parquet"
    node_path = scratch / f"node_shard_{worker:02d}.parquet"
    started = time.perf_counter()
    manifest_rows: list[dict[str, Any]] = []
    edge_offset = 0
    node_offset = 0
    with pq.ParquetWriter(
        edge_path,
        EDGE_SCHEMA_ARROW,
        compression="zstd",
        compression_level=3,
        use_dictionary=[
            "family_ordinal",
            "opportunity_ordinal",
            "proposal_kind_code",
            "proposal_reason_code",
            "decision_code",
        ],
        write_statistics=True,
        version="2.6",
    ) as edge_writer, pq.ParquetWriter(
        node_path,
        NODE_SCHEMA_ARROW,
        compression="zstd",
        compression_level=6,
        use_dictionary=["family_ordinal", "terminal_code", "reachable_from_anchor"],
        write_statistics=True,
        version="2.6",
    ) as node_writer:
        for metadata in family_rows:
            family = family_from_row(metadata)
            family_ordinal = int(metadata["family_ordinal"])
            family_digest = bytes(metadata["family_digest"])
            opportunities = opportunity_arrays(family)
            slots = len(opportunities["actors"])
            terminal = classify_states(family)
            reachable = reachable_states(family, terminal)
            terminal_counts = np.bincount(terminal, minlength=3)
            reachable_terminal_counts = np.bincount(
                terminal[reachable.astype(bool)], minlength=3
            )
            node_digest = ColumnStreamDigest(
                NODE_DIGEST_DOMAIN,
                family_digest,
                ("state_ordinal", "terminal_code", "reachable_from_anchor"),
            )
            for start in range(0, family.state_count, ROW_GROUP):
                stop = min(start + ROW_GROUP, family.state_count)
                node_arrays = {
                    "state_ordinal": np.arange(start, stop, dtype=np.uint32),
                    "terminal_code": terminal[start:stop],
                    "reachable_from_anchor": reachable[start:stop],
                }
                node_digest.update(node_arrays)
                node_writer.write_table(
                    node_table(
                        family_ordinal,
                        start,
                        terminal[start:stop],
                        reachable[start:stop],
                    ),
                    row_group_size=ROW_GROUP,
                )

            edge_digest = ColumnStreamDigest(
                EDGE_DIGEST_DOMAIN, family_digest, EDGE_ARRAY_COLUMNS
            )
            counts: Counter[str] = Counter()
            metric_signs: Counter[str] = Counter()
            for start in range(0, family.state_count, STATE_BATCH):
                stop = min(start + STATE_BATCH, family.state_count)
                arrays = edge_batch(family, terminal, start, stop)
                edge_digest.update(arrays)
                edge_count = len(arrays["source_state_ordinal"])
                if edge_count == 0:
                    continue
                edge_writer.write_table(
                    edge_table(family_ordinal, arrays), row_group_size=ROW_GROUP
                )
                counts["edges"] += edge_count
                counts["changed_edges"] += int(arrays["changed"].sum())
                counts["self_loops"] += int(
                    np.count_nonzero(
                        arrays["source_state_ordinal"]
                        == arrays["successor_state_ordinal"]
                    )
                )
                counts["parallel_endpoint_extra"] += _parallel_endpoint_extra(
                    arrays["successor_state_ordinal"], slots
                )
                for column in [
                    "cost_no_ops",
                    "cost_rejections",
                    "cost_memory_updates",
                    "cost_accepted_swaps",
                    "cost_displaced_cells",
                    "observation_reads",
                    "value_comparisons",
                ]:
                    counts[column] += int(arrays[column].sum())
                for metric in [
                    "delta_adjacent_descents",
                    "delta_inversion_count",
                    "delta_spearman_footrule",
                    "delta_maximum_rank_error",
                ]:
                    values = arrays[metric]
                    metric_signs[f"{metric}_improving"] += int(np.count_nonzero(values < 0))
                    metric_signs[f"{metric}_neutral"] += int(np.count_nonzero(values == 0))
                    metric_signs[f"{metric}_worsening"] += int(np.count_nonzero(values > 0))

            expected_edges = int(terminal_counts[0]) * slots
            if counts["edges"] != expected_edges:
                raise AssertionError(
                    f"family {family_ordinal}: {counts['edges']} != {expected_edges}"
                )
            if counts["edges"] > MAX_FAMILY_EDGES:
                raise RuntimeError(f"family edge escalation: {family_ordinal}")
            if (
                counts["cost_no_ops"]
                + counts["cost_rejections"]
                + counts["cost_memory_updates"]
                + counts["cost_accepted_swaps"]
                != counts["edges"]
            ):
                raise AssertionError("action-cost partition failure")
            denominators = metric_denominators(family.n)
            manifest_rows.append(
                {
                    "schema_version": "e03.s05.graph_family_manifest.v1",
                    "research_step_id": "S05",
                    "family_ordinal": family_ordinal,
                    "family_id": metadata["family_id"],
                    "family_digest": family_digest,
                    "materialization_tier": metadata["materialization_tier"],
                    "n": family.n,
                    "architecture": family.architecture.value,
                    "direction": family.direction.value,
                    "policy_profile": family.policy_profile,
                    "fault_mode": family.fault_mode,
                    "fault_count": family.fault_count,
                    "node_count": family.state_count,
                    "active_node_count": int(terminal_counts[0]),
                    "complete_node_count": int(terminal_counts[1]),
                    "quiescent_node_count": int(terminal_counts[2]),
                    "reachable_node_count": int(reachable.sum()),
                    "reachable_active_node_count": int(reachable_terminal_counts[0]),
                    "reachable_complete_node_count": int(reachable_terminal_counts[1]),
                    "reachable_quiescent_node_count": int(reachable_terminal_counts[2]),
                    "unreachable_node_count": int(family.state_count - reachable.sum()),
                    "anchor_terminal_code": int(terminal[0]),
                    "opportunities_per_active_node": slots,
                    "edge_count": counts["edges"],
                    "changed_edge_count": counts["changed_edges"],
                    "self_loop_count": counts["self_loops"],
                    "parallel_endpoint_extra_count": counts["parallel_endpoint_extra"],
                    "no_op_edge_count": counts["cost_no_ops"],
                    "rejection_edge_count": counts["cost_rejections"],
                    "memory_update_edge_count": counts["cost_memory_updates"],
                    "accepted_swap_edge_count": counts["cost_accepted_swaps"],
                    "displaced_cell_cost_total": counts["cost_displaced_cells"],
                    "observation_read_cost_total": counts["observation_reads"],
                    "value_comparison_cost_total": counts["value_comparisons"],
                    **dict(metric_signs),
                    **{
                        f"{key}_denominator": value
                        for key, value in denominators.items()
                    },
                    "edge_logical_digest": edge_digest.hexdigest(),
                    "node_logical_digest": node_digest.hexdigest(),
                    "edge_shard": f"graphs/edge_shard_{worker:02d}.parquet",
                    "node_shard": f"graphs/node_shard_{worker:02d}.parquet",
                    "edge_shard_row_offset": edge_offset,
                    "node_shard_row_offset": node_offset,
                    "source_state_schema": metadata["schema_version"],
                    "scheduler_projection": metadata["scheduler_projection"],
                }
            )
            edge_offset += counts["edges"]
            node_offset += family.state_count
    return {
        "worker": worker,
        "edgePath": str(edge_path),
        "nodePath": str(node_path),
        "edgeRows": edge_offset,
        "nodeRows": node_offset,
        "edgeBytes": edge_path.stat().st_size,
        "nodeBytes": node_path.stat().st_size,
        "familyRows": manifest_rows,
        "elapsedSeconds": time.perf_counter() - started,
    }


def aggregate_strata(frame: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "materialization_tier",
        "n",
        "architecture",
        "direction",
        "policy_profile",
        "fault_mode",
        "fault_count",
    ]
    measures = [
        column
        for column in frame.columns
        if (
            column.endswith("_count")
            or column.endswith("_total")
            or column.startswith("delta_")
        )
        and column not in dimensions
    ]
    records: list[pd.DataFrame] = []
    for dimension in ["overall", *dimensions]:
        if dimension == "overall":
            values = frame[measures].sum().to_frame().T
            values.insert(0, "stratum_value", "all")
        else:
            values = frame.groupby(dimension, dropna=False)[measures].sum().reset_index()
            values = values.rename(columns={dimension: "stratum_value"})
            values["stratum_value"] = values["stratum_value"].astype(str)
        values.insert(0, "stratum_name", dimension)
        values.insert(0, "research_step_id", "S05")
        records.append(values)
    result = pd.concat(records, ignore_index=True)
    result[measures] = result[measures].astype(np.int64)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8]")
    family_table = pq.read_table(S04 / "state_family_inventory.parquet")
    family_rows = family_table.to_pylist()
    upper = 0
    maximum_family_upper = 0
    for row in family_rows:
        family = family_from_row(row)
        family_upper = family.state_count * len(opportunity_arrays(family)["actors"])
        upper += family_upper
        maximum_family_upper = max(maximum_family_upper, family_upper)
    preflight = {
        "researchStepId": "S05",
        "familyCount": len(family_rows),
        "nodeCount": sum(int(row["state_count"]) for row in family_rows),
        "allStateOpportunityUpperBound": upper,
        "maximumFamilyOpportunityUpperBound": maximum_family_upper,
        "edgeBoundary": MAX_EDGE_ROWS,
        "familyEdgeBoundary": MAX_FAMILY_EDGES,
        "passes": upper <= MAX_EDGE_ROWS and maximum_family_upper <= MAX_FAMILY_EDGES,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    if not preflight["passes"]:
        raise RuntimeError(f"S05 preflight escalation: {preflight}")

    input_paths = [
        CONTRACT,
        S04 / "state_inventory.parquet",
        S04 / "state_family_inventory.parquet",
        S04 / "canonical_state_encoder.md",
        S04 / "research_step_full_results.md",
        Path("/artifacts/research_steps/S01/research_step_full_results.md"),
        Path("/artifacts/research_steps/S02/research_step_full_results.md"),
        Path("/artifacts/research_steps/S03/research_step_full_results.md"),
        REPOSITORY / "src/detours/distances.py",
        REPOSITORY / "src/detours/state_space.py",
        REPOSITORY / "src/detours/transition_graph.py",
        REPOSITORY / "reference_simulator/model.py",
        REPOSITORY / "reference_simulator/policies.py",
        REPOSITORY / "reference_simulator/transition_primitives.py",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/scheduler.py",
        E01 / "research_steps/S03/state_diagrams.md",
        E01 / "research_steps/S05/semantic_decisions.json",
        E01 / "release/reference_simulator/release_manifest.json",
        Path("/workspace/input-attachments/MANIFEST.json"),
    ]
    pre_hashes = {str(path): sha256_file(path) for path in input_paths}
    if args.scratch.exists():
        shutil.rmtree(args.scratch)
    args.scratch.mkdir(parents=True)
    if args.output.exists():
        shutil.rmtree(args.output)
    (args.output / "graphs").mkdir(parents=True)
    warm_graph_kernels()
    started = time.perf_counter()
    assignments = [
        [row for row in family_rows if int(row["family_ordinal"]) % args.workers == worker]
        for worker in range(args.workers)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                _worker,
                [(worker, assignments[worker], str(args.scratch)) for worker in range(args.workers)],
            )
        )
    manifest_rows = sorted(
        [row for result in results for row in result["familyRows"]],
        key=lambda row: row["family_ordinal"],
    )
    for result in results:
        shutil.move(
            result["edgePath"], args.output / "graphs" / Path(result["edgePath"]).name
        )
        shutil.move(
            result["nodePath"], args.output / "graphs" / Path(result["nodePath"]).name
        )
    total_edges = sum(int(row["edge_count"]) for row in manifest_rows)
    total_nodes = sum(int(row["node_count"]) for row in manifest_rows)
    if total_edges > MAX_EDGE_ROWS:
        raise RuntimeError(f"S05 actual edge escalation: {total_edges}")
    if len(manifest_rows) != 7_984 or total_nodes != 22_301_808:
        raise AssertionError("S04 family/node coverage mismatch")
    manifest_frame = pd.DataFrame(manifest_rows)
    write_parquet(args.output / "graph_family_manifest.parquet", manifest_frame)
    count_columns = [
        "research_step_id",
        "family_ordinal",
        "family_id",
        "n",
        "architecture",
        "direction",
        "policy_profile",
        "fault_mode",
        "fault_count",
        "node_count",
        "active_node_count",
        "complete_node_count",
        "quiescent_node_count",
        "edge_count",
        "changed_edge_count",
        "self_loop_count",
        "parallel_endpoint_extra_count",
        "no_op_edge_count",
        "rejection_edge_count",
        "memory_update_edge_count",
        "accepted_swap_edge_count",
        "observation_read_cost_total",
        "value_comparison_cost_total",
    ] + [column for column in manifest_frame if column.startswith("delta_")]
    write_parquet(
        args.output / "transition_counts_by_family.parquet",
        manifest_frame[count_columns],
    )
    write_parquet(
        args.output / "transition_counts_by_stratum.parquet",
        aggregate_strata(manifest_frame),
    )
    reachability_columns = [
        "research_step_id",
        "family_ordinal",
        "family_id",
        "n",
        "architecture",
        "direction",
        "policy_profile",
        "fault_mode",
        "fault_count",
        "node_count",
        "active_node_count",
        "complete_node_count",
        "quiescent_node_count",
        "reachable_node_count",
        "reachable_active_node_count",
        "reachable_complete_node_count",
        "reachable_quiescent_node_count",
        "unreachable_node_count",
        "anchor_terminal_code",
    ]
    write_parquet(
        args.output / "reachability_summary.parquet",
        manifest_frame[reachability_columns],
    )

    post_hashes = {str(path): sha256_file(path) for path in input_paths}
    if pre_hashes != post_hashes:
        raise AssertionError("S05 input mutation")
    graph_paths = sorted((args.output / "graphs").glob("*.parquet"))
    graph_bytes = sum(path.stat().st_size for path in graph_paths)
    if graph_bytes > MAX_GRAPH_BYTES:
        raise RuntimeError(f"S05 graph byte escalation: {graph_bytes}")
    summary = {
        "schemaVersion": "e03.s05.graph_corpus_manifest.v1",
        "researchStepId": "S05",
        "success": True,
        "sourceStateSchema": "e03.s04.structural_state.v1",
        "edgeSchema": EDGE_SCHEMA,
        "nodeAnnotationSchema": NODE_SCHEMA,
        "schedulerProjection": "fair_serial_legal_opportunity_set_v1",
        "familyCount": len(manifest_rows),
        "nodeCount": total_nodes,
        "edgeCount": total_edges,
        "allStateOpportunityUpperBound": upper,
        "terminalSuppressedOpportunityCount": upper - total_edges,
        "activeNodeCount": int(manifest_frame["active_node_count"].sum()),
        "completeNodeCount": int(manifest_frame["complete_node_count"].sum()),
        "quiescentNodeCount": int(manifest_frame["quiescent_node_count"].sum()),
        "reachableNodeCountAcrossFamilyAnchors": int(manifest_frame["reachable_node_count"].sum()),
        "changedEdgeCount": int(manifest_frame["changed_edge_count"].sum()),
        "selfLoopCount": int(manifest_frame["self_loop_count"].sum()),
        "parallelEndpointExtraCount": int(manifest_frame["parallel_endpoint_extra_count"].sum()),
        "actionCounts": {
            "noOp": int(manifest_frame["no_op_edge_count"].sum()),
            "rejection": int(manifest_frame["rejection_edge_count"].sum()),
            "memoryUpdate": int(manifest_frame["memory_update_edge_count"].sum()),
            "acceptedSwap": int(manifest_frame["accepted_swap_edge_count"].sum()),
        },
        "graphBytes": graph_bytes,
        "elapsedSeconds": time.perf_counter() - started,
        "workerCount": args.workers,
        "stateBatch": STATE_BATCH,
        "rowGroup": ROW_GROUP,
        "workers": [
            {key: value for key, value in result.items() if key != "familyRows"}
            for result in results
        ],
        "edgeCodes": {
            "proposalKind": PROPOSAL_KIND_LABELS,
            "proposalReason": REASON_LABELS,
            "decision": DECISION_LABELS,
            "terminal": TERMINAL_LABELS,
            "schedulerSide": {"-1": "not_applicable", "0": "left", "1": "right"},
        },
        "boundaries": {
            "maximumEdgeRows": MAX_EDGE_ROWS,
            "maximumSingleFamilyEdges": MAX_FAMILY_EDGES,
            "maximumFinalGraphBytes": MAX_GRAPH_BYTES,
            "triggered": [],
        },
        "startingCommit": STARTING_COMMIT,
        "inputsUnchanged": True,
        "graphFiles": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "rowGroups": pq.ParquetFile(path).metadata.num_row_groups,
            }
            for path in graph_paths
        ],
    }
    write_json(args.output / "graph_corpus_manifest.json", summary)
    write_json(
        args.output / "input_immutability.json",
        {
            "schemaVersion": "e03.s05.input_immutability.v1",
            "researchStepId": "S05",
            "success": True,
            "preRunSha256": pre_hashes,
            "postRunSha256": post_hashes,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
