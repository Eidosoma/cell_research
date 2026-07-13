#!/usr/bin/env python3
"""One-process-per-run S09 wrapper around the source-external S04 adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from historical_backend.legacy_adapter import run_historical_toy  # noqa: E402


GRID = tuple(index / 200 for index in range(201))


def sortedness(values):
    return 100.0 * (1 + sum(b > a for a, b in zip(values, values[1:]))) / len(values)


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def interpolate(points):
    final = points[-1][0]
    if final == 0:
        return [points[-1][1]] * len(GRID)
    output = []
    cursor = 0
    for target_fraction in GRID:
        target = target_fraction * final
        while cursor + 1 < len(points) and points[cursor + 1][0] < target:
            cursor += 1
        if cursor + 1 == len(points):
            output.append(points[-1][1])
            continue
        left_x, left_y = points[cursor]
        right_x, right_y = points[cursor + 1]
        if right_x == left_x:
            output.append(right_y)
        else:
            weight = (target - left_x) / (right_x - left_x)
            output.append(left_y + weight * (right_y - left_y))
    return output


def main():
    task = json.load(sys.stdin)
    result = run_historical_toy(
        source_root=Path(task["source_root"]),
        checksum_manifest=Path(task["checksum_manifest"]),
        values=task["values"],
        policy=task["policy"],
        stop_mode="is_sorted",
        timeout_seconds=float(task["timeout_seconds"]),
        seed=int(task["seed"]),
    )
    points = [[0, sortedness(task["values"])]]
    points.extend(
        [index, sortedness(values)]
        for index, values in enumerate(result["sortingSteps"], start=1)
    )
    final_values = result["finalValues"]
    metrics = result["metrics"]
    output = {
        "backend_profile": "C-frozen-public-commit",
        "publication_snapshot_claimed": False,
        "scheduler": "CPython_OS_threads_lock_serialized_unknown_order",
        "sequence_basis": "recorded_swap",
        "historical_random_stream_status": "unavailable_not_invented",
        "stop_reason": result["stopReason"],
        "completed": result["stopReason"] == "historical_is_sorted" and result["nonStrictlySortedAtShutdown"],
        "timed_out": result["timedOut"],
        "activation_count": None,
        "successful_swap_count": metrics["recordedSortingSteps"],
        "initial_sortedness_percent": points[0][1],
        "final_sortedness_percent": sortedness(final_values),
        "final_nonstrictly_sorted": result["nonStrictlySortedAtShutdown"],
        "value_multiset_conserved": result["valueMultisetConserved"],
        "initial_values_sha256": canonical_hash(task["values"]),
        "final_values_sha256": canonical_hash(final_values),
        "final_state_hash": None,
        "observable_final_hash": canonical_hash(final_values),
        "trace_sha256": result["traceSha256"],
        "elapsed_seconds": result["elapsedSeconds"],
        "adapter_safety_timeout_seconds": result["adapterSafetyTimeoutSeconds"],
        "thread_shutdown_clean": result["threadShutdownClean"],
        "curve_grid": interpolate(points),
        "raw_curve": points if task["retain_raw_curve"] or result["timedOut"] else None,
        "ledger_activations": None,
        "ledger_observationReads": None,
        "ledger_valueComparisons": metrics["compareAndSwapCount"],
        "ledger_proposals": None,
        "ledger_noOps": None,
        "ledger_rejections": None,
        "ledger_memoryUpdates": None,
        "ledger_acceptedSwaps": None,
        "ledger_displacedCells": 2 * metrics["recordedSortingSteps"],
        "ledger_conflictLosses": None,
        "historical_swap_probe": metrics["swapCount"],
        "historical_frozen_swap_attempts": metrics["frozenSwapAttempts"],
        "source_precheck_valid": result["sourcePrecheck"]["contentValid"],
        "runtime_python": result["runtime"]["python"],
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
