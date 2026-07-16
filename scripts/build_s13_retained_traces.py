#!/usr/bin/env python3
"""Materialize full metric-checkpoint traces for the outcome-blind S13 trace panel."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.detours import memory_information as mi
from reference_simulator.model import canonical_json_bytes
from scripts import build_s13_memory_information as build


OUTPUT = Path("/artifacts/research_steps/S13")
CACHE = Path("/cache/e03_s13/primary_results.npz")


def write_parquet(path: Path, frame: pd.DataFrame, schema: str) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema.encode(), b"researchStepId": b"S13"}
    )
    pq.write_table(table, path, compression="zstd", compression_level=9,
                   use_dictionary=True, version="2.6")


def traced_kernel() -> tuple[object, str]:
    """Derive an instrumented function without changing the accepted kernel."""

    source = inspect.getsource(mi._simulate_one.py_func)
    source = source.replace("@nb.njit(cache=True)\n", "@nb.njit(cache=False)\n", 1)
    source = source.replace("def _simulate_one(\n", "def _simulate_one_trace(\n", 1)
    source = source.replace(
        "    checkpoint_event: int,\n) -> tuple:",
        "    checkpoint_event: int,\n"
        "    trace_events: np.ndarray,\n"
        "    trace_levels: np.ndarray,\n"
        "    trace_physical: np.ndarray,\n"
        "    trace_full: np.ndarray,\n"
        "    trace_ledger: np.ndarray,\n"
        "    trace_audit: np.ndarray,\n"
        ") -> tuple:",
        1,
    )
    source = source.replace(
        "    state_checkpoints = 0\n    stop = _terminal_code(",
        "    state_checkpoints = 0\n"
        "    trace_count = 1\n"
        "    trace_events[0] = 0\n"
        "    trace_levels[0, :] = levels\n"
        "    trace_physical[0] = _state_fingerprint(root, occupancy, cursors, n, 0)\n"
        "    trace_full[0] = _full_state_fingerprint(root, occupancy, cursors, failure_memory, stagnation_counter, recent_direction, n, 0, arm_code)\n"
        "    trace_ledger[0, :] = ledger\n"
        "    trace_audit[0, :] = capability_audit\n"
        "    stop = _terminal_code(",
        1,
    )
    source = source.replace(
        "\n        if any_change:\n",
        "\n        if any_swap:\n"
        "            trace_events[trace_count] = events\n"
        "            trace_levels[trace_count, :] = levels\n"
        "            trace_physical[trace_count] = _state_fingerprint(root, occupancy, cursors, n, events)\n"
        "            trace_full[trace_count] = _full_state_fingerprint(root, occupancy, cursors, failure_memory, stagnation_counter, recent_direction, n, events, arm_code)\n"
        "            trace_ledger[trace_count, :] = ledger\n"
        "            trace_audit[trace_count, :] = capability_audit\n"
        "            trace_count += 1\n"
        "\n        if any_change:\n",
        1,
    )
    source = source.replace(
        "\n    if stop == 0:\n",
        "\n    if trace_events[trace_count - 1] != events:\n"
        "        trace_events[trace_count] = events\n"
        "        trace_levels[trace_count, :] = levels\n"
        "        trace_physical[trace_count] = _state_fingerprint(root, occupancy, cursors, n, events)\n"
        "        trace_full[trace_count] = _full_state_fingerprint(root, occupancy, cursors, failure_memory, stagnation_counter, recent_direction, n, events, arm_code)\n"
        "        trace_ledger[trace_count, :] = ledger\n"
        "        trace_audit[trace_count, :] = capability_audit\n"
        "        trace_count += 1\n"
        "\n    if stop == 0:\n",
        1,
    )
    source = source.replace(
        "        checkpoint_levels,\n    )\n",
        "        checkpoint_levels,\n        trace_count,\n    )\n",
        1,
    )
    namespace = dict(mi.__dict__)
    namespace["nb"] = nb
    exec(compile(source, "<source-derived-s13-trace-kernel>", "exec"), namespace)
    return namespace["_simulate_one_trace"], hashlib.sha256(source.encode()).hexdigest()


def main() -> None:
    kernel, instrumented_sha = traced_kernel()
    design, arrays, _ = build.build_design()
    with np.load(CACHE, allow_pickle=False) as archive:
        primary = {name: archive[name] for name in archive.files}
    selected = np.flatnonzero(design.trace_retained.to_numpy())
    rows: list[dict] = []
    equivalence = 0
    for index in selected:
        budget = int(arrays["budgets"][index])
        width = int(arrays["batch_widths"][index])
        capacity = budget // width + 3
        trace_events = np.full(capacity, -1, dtype=np.int32)
        trace_levels = np.full((capacity, 4), -1, dtype=np.int32)
        trace_physical = np.zeros(capacity, dtype=np.uint64)
        trace_full = np.zeros(capacity, dtype=np.uint64)
        trace_ledger = np.full((capacity, 8), -1, dtype=np.int64)
        trace_audit = np.full((capacity, 10), -1, dtype=np.int64)
        result = kernel(
            int(arrays["ns"][index]), arrays["policies"][index], arrays["faults"][index],
            arrays["occupancies"][index], arrays["cursors"][index],
            int(arrays["directions"][index]), width, arrays["roots"][index], budget,
            int(arrays["arm_codes"][index]), 0, trace_events, trace_levels,
            trace_physical, trace_full, trace_ledger, trace_audit,
        )
        checks = (
            int(result[0]) == int(primary["stop"][index]),
            int(result[1]) == int(primary["events"][index]),
            np.array_equal(result[2], primary["ledger"][index]),
            np.array_equal(result[3], primary["capability_audit"][index]),
            np.array_equal(result[4], primary["initial"][index]),
            np.array_equal(result[5], primary["final"][index]),
            np.array_equal(result[7], primary["episode_count"][index]),
            np.array_equal(result[10], primary["max_depth"][index]),
            int(result[19]) == int(primary["state_checkpoints"][index]),
            int(result[20]) == int(primary["trajectory_digest"][index]),
            int(result[21]) == int(primary["physical_fingerprint"][index]),
            int(result[22]) == int(primary["full_fingerprint"][index]),
        )
        if not all(checks):
            raise AssertionError(f"instrumented kernel drift for {design.at[index, 'run_id']}")
        equivalence += 1
        count = int(result[-1])
        expected_state_checkpoints = int(primary["state_checkpoints"][index])
        terminal_duplicate = int(trace_events[count - 1] != trace_events[count - 2]) if count > 1 else 0
        if count not in (expected_state_checkpoints + 1, expected_state_checkpoints + 2):
            raise AssertionError("trace checkpoint accounting drift")
        for checkpoint in range(count):
            role = "accepted_swap_checkpoint"
            if checkpoint == 0:
                role = "initial"
            elif checkpoint == count - 1 and (
                checkpoint > expected_state_checkpoints or trace_events[checkpoint] == int(result[1])
            ):
                role = "terminal"
            row = {
                "run_id": design.at[index, "run_id"],
                "evidence_tier": design.at[index, "evidence_tier"],
                "capability_arm": design.at[index, "capability_arm"],
                "metric_checkpoint_index": checkpoint,
                "checkpoint_role": role,
                "event_count": int(trace_events[checkpoint]),
                "physical_state_fingerprint_u64": f"{int(trace_physical[checkpoint]):016x}",
                "full_state_fingerprint_u64": f"{int(trace_full[checkpoint]):016x}",
            }
            for metric_index, metric in enumerate(mi.METRIC_NAMES):
                row[f"distance_{metric}"] = int(trace_levels[checkpoint, metric_index])
            for ledger_index, name in enumerate(mi.LEDGER_NAMES):
                row[f"ledger_{name}"] = int(trace_ledger[checkpoint, ledger_index])
            row["capability_memory_writes"] = int(trace_audit[checkpoint, 7])
            row["capability_bit_writes"] = int(trace_audit[checkpoint, 8])
            rows.append(row)
    frame = pd.DataFrame(rows)
    write_parquet(OUTPUT / "retained_metric_traces.parquet", frame, "e03.s13.retained_metric_traces.v1")
    summary = {
        "researchStepId": "S13", "success": True,
        "selectionFrozenBeforeOutcomes": True, "traceRuns": len(selected),
        "largeTraceRuns": int(design.loc[selected].evidence_tier.eq("empirical_large_n").sum()),
        "smallTraceRuns": int(design.loc[selected].evidence_tier.eq("exact_small_n_anchor_empirical_path").sum()),
        "traceRows": len(frame), "instrumentedKernelEquivalentRuns": equivalence,
        "acceptedSwapCheckpoints": int(primary["state_checkpoints"][selected].sum()),
        "acceptedKernelSourceSha256": hashlib.sha256(inspect.getsource(mi._simulate_one.py_func).encode()).hexdigest(),
        "instrumentedKernelSourceSha256": instrumented_sha,
    }
    (OUTPUT / "retained_trace_build_summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
