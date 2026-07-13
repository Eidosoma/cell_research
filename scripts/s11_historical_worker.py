#!/usr/bin/env python3
"""One-process S11 wrapper for the source-external frozen public backend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from historical_backend.legacy_adapter import run_historical_toy  # noqa: E402


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def monotonicity_error(values):
    return sum(right < left for left, right in zip(values, values[1:]))


def main():
    task = json.load(sys.stdin)
    result = run_historical_toy(
        source_root=Path(task["source_root"]),
        checksum_manifest=Path(task["checksum_manifest"]),
        values=task["values"],
        policy=task["policy"],
        stop_mode="no_cells_should_move",
        timeout_seconds=float(task["timeout_seconds"]),
        seed=int(task["seed"]),
        frozen_indices=task["frozen_positions"],
    )
    final_values = result["finalValues"]
    metrics = result["metrics"]
    fault_final_positions = [final_values.index(value) for value in task["fault_values"]]
    output = {
        "schema_version": "e01.s11.frozen_cell_run.v1",
        "research_step_id": "S11",
        "backend_profile": "C-frozen-public-commit",
        "publication_snapshot_claimed": False,
        "evidence_layer": "frozen_public_commit_nondeterministic_comparator",
        "scheduler": "CPython_OS_threads_lock_serialized_unknown_order",
        "sequence_basis": "recorded_swap",
        "historical_random_stream_status": "unavailable_not_invented_bank_seed_override",
        "stop_reason": result["stopReason"],
        "completed": bool(result["nonStrictlySortedAtShutdown"]),
        "censored": bool(result["timedOut"]),
        "timed_out": bool(result["timedOut"]),
        "activation_count": None,
        "monotonicity_error": monotonicity_error(final_values),
        "final_nonstrictly_sorted": bool(result["nonStrictlySortedAtShutdown"]),
        "value_multiset_conserved": bool(result["valueMultisetConserved"]),
        "initial_values_sha256": canonical_hash(task["values"]),
        "final_values_sha256": canonical_hash(final_values),
        "initial_state_hash": None,
        "final_state_hash": None,
        "observable_final_hash": canonical_hash(final_values),
        "trace_sha256": result["traceSha256"],
        "elapsed_seconds": result["elapsedSeconds"],
        "adapter_safety_timeout_seconds": result["adapterSafetyTimeoutSeconds"],
        "thread_shutdown_clean": bool(result["threadShutdownClean"]),
        "final_values": final_values,
        "fault_cell_ids": task["fault_cell_ids"],
        "fault_values": task["fault_values"],
        "fault_positions_initial": task["frozen_positions"],
        "fault_positions_final": fault_final_positions,
        "ledger_activations": None,
        "ledger_observationReads": None,
        "ledger_valueComparisons": None,
        "ledger_proposals": None,
        "ledger_noOps": None,
        "ledger_rejections": None,
        "ledger_memoryUpdates": None,
        "ledger_acceptedSwaps": None,
        "ledger_displacedCells": 2 * metrics["recordedSortingSteps"],
        "ledger_conflictLosses": None,
        "swap_only_cost": metrics["recordedSortingSteps"],
        "publication_swap_plus_comparison_cost": metrics["recordedSortingSteps"] + metrics["compareAndSwapCount"],
        "target_calculations": None,
        "rejected_actions": None,
        "unit_weight_full_ledger": None,
        "historical_compare_and_swap_probe": metrics["compareAndSwapCount"],
        "historical_frozen_swap_attempts": metrics["frozenSwapAttempts"],
        "cost_profile": "lossy_C_recorded_swap_plus_source_probe_nonparity",
        "source_precheck_valid": bool(result["sourcePrecheck"]["contentValid"]),
        "runtime_python": result["runtime"]["python"],
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

