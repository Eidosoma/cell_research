#!/usr/bin/env python3
"""Second-pass deterministic logical reconstruction of every S05 family graph."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes
from scripts.build_s05_graphs import ColumnStreamDigest, family_from_row
from src.detours.transition_graph import (
    EDGE_ARRAY_COLUMNS,
    EDGE_DIGEST_DOMAIN,
    NODE_DIGEST_DOMAIN,
    classify_states,
    edge_batch,
    reachable_states,
    warm_graph_kernels,
)


S04_FAMILIES = Path("/artifacts/research_steps/S04/state_family_inventory.parquet")
S05 = Path("/artifacts/research_steps/S05")
STATE_BATCH = 32_768
ROW_GROUP = 65_536


def _worker(args: tuple[int, list[dict[str, Any]], dict[int, dict[str, Any]]]) -> dict[str, Any]:
    worker, rows, expected = args
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = "1"
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    reconstructed: list[dict[str, Any]] = []
    for metadata in rows:
        ordinal = int(metadata["family_ordinal"])
        family = family_from_row(metadata)
        family_digest = bytes(metadata["family_digest"])
        terminal = classify_states(family)
        reachable = reachable_states(family, terminal)
        node_digest = ColumnStreamDigest(
            NODE_DIGEST_DOMAIN,
            family_digest,
            ("state_ordinal", "terminal_code", "reachable_from_anchor"),
        )
        for start in range(0, family.state_count, ROW_GROUP):
            stop = min(start + ROW_GROUP, family.state_count)
            node_digest.update(
                {
                    "state_ordinal": np.arange(start, stop, dtype=np.uint32),
                    "terminal_code": terminal[start:stop],
                    "reachable_from_anchor": reachable[start:stop],
                }
            )
        edge_digest = ColumnStreamDigest(
            EDGE_DIGEST_DOMAIN, family_digest, EDGE_ARRAY_COLUMNS
        )
        edge_count = 0
        for start in range(0, family.state_count, STATE_BATCH):
            stop = min(start + STATE_BATCH, family.state_count)
            arrays = edge_batch(family, terminal, start, stop)
            edge_digest.update(arrays)
            edge_count += len(arrays["source_state_ordinal"])
        terminal_counts = np.bincount(terminal, minlength=3)
        actual = {
            "familyOrdinal": ordinal,
            "nodeCount": family.state_count,
            "activeNodeCount": int(terminal_counts[0]),
            "completeNodeCount": int(terminal_counts[1]),
            "quiescentNodeCount": int(terminal_counts[2]),
            "reachableNodeCount": int(reachable.sum()),
            "edgeCount": edge_count,
            "edgeLogicalDigest": edge_digest.hexdigest(),
            "nodeLogicalDigest": node_digest.hexdigest(),
        }
        reconstructed.append(actual)
        target = expected[ordinal]
        comparisons = {
            "nodeCount": target["node_count"],
            "activeNodeCount": target["active_node_count"],
            "completeNodeCount": target["complete_node_count"],
            "quiescentNodeCount": target["quiescent_node_count"],
            "reachableNodeCount": target["reachable_node_count"],
            "edgeCount": target["edge_count"],
            "edgeLogicalDigest": target["edge_logical_digest"],
            "nodeLogicalDigest": target["node_logical_digest"],
        }
        differing = [key for key, value in comparisons.items() if actual[key] != value]
        if differing:
            failures.append(
                {"familyOrdinal": ordinal, "differingFields": differing}
            )
    return {
        "worker": worker,
        "families": len(rows),
        "failures": failures,
        "reconstructed": reconstructed,
        "elapsedSeconds": time.perf_counter() - started,
    }


def main() -> None:
    families = pq.read_table(S04_FAMILIES).to_pylist()
    expected_rows = pq.read_table(S05 / "graph_family_manifest.parquet").to_pylist()
    expected = {int(row["family_ordinal"]): row for row in expected_rows}
    if len(families) != 7_984 or len(expected) != 7_984:
        raise AssertionError("family denominator mismatch")
    warm_graph_kernels()
    started = time.perf_counter()
    assignments = [
        [row for row in families if int(row["family_ordinal"]) % 8 == worker]
        for worker in range(8)
    ]
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                _worker,
                [(worker, assignments[worker], expected) for worker in range(8)],
            )
        )
    failures = [failure for result in results for failure in result["failures"]]
    reconstructed = [row for result in results for row in result["reconstructed"]]
    output = {
        "schemaVersion": "e03.s05.deterministic_reconstruction.v1",
        "researchStepId": "S05",
        "success": not failures,
        "familyCount": len(reconstructed),
        "edgeCount": sum(row["edgeCount"] for row in reconstructed),
        "nodeCount": sum(row["nodeCount"] for row in reconstructed),
        "matchedFamilyCount": len(reconstructed) - len(failures),
        "mismatchCount": len(failures),
        "failures": failures,
        "method": "independent second full kernel pass; compare node/terminal/reachability counts and chunk-independent logical edge/node digests",
        "workerCount": 8,
        "elapsedSeconds": time.perf_counter() - started,
        "workers": [
            {key: value for key, value in result.items() if key not in {"reconstructed", "failures"}}
            for result in results
        ],
    }
    (S05 / "deterministic_reconstruction.json").write_bytes(
        canonical_json_bytes(output) + b"\n"
    )
    print(json.dumps(output, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
