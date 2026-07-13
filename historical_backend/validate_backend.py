#!/usr/bin/env python3
"""Validation harness for the S04 source-external historical adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Sequence
import warnings

import numpy as np

from legacy_adapter import run_historical_toy, sha256_file, verify_source_tree, write_run_outputs


TOYS = [
    {"fixture": "C-T01", "s03Relation": "T01", "policy": "bubble", "values": [2, 1], "expected": [1, 2], "expectedSwaps": 1},
    {"fixture": "C-T02", "s03Relation": "T02/T15", "policy": "bubble", "values": [1, 1], "expected": [1, 1], "expectedSwaps": 0},
    {"fixture": "C-T05", "s03Relation": "T05", "policy": "insertion", "values": [2, 1], "expected": [1, 2], "expectedSwaps": 1},
    {"fixture": "C-T10", "s03Relation": "T10", "policy": "selection", "values": [2, 1], "expected": [1, 2], "expectedSwaps": 1},
    {"fixture": "C-ORDERED", "s03Relation": "completion subset", "policy": "insertion", "values": [1, 2, 3], "expected": [1, 2, 3], "expectedSwaps": 0},
    {"fixture": "C-DUP", "s03Relation": "T15 completion subset", "policy": "selection", "values": [2, 1, 1], "expected": [1, 1, 2], "expectedSwaps": None},
]


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run(source: Path, checksums: Path, toy: dict[str, Any], seed: int = 1729) -> dict[str, Any]:
    return run_historical_toy(
        source_root=source,
        checksum_manifest=checksums,
        values=toy["values"],
        policy=toy["policy"],
        seed=seed,
        timeout_seconds=10.0,
    )


def validate(source: Path, checksums: Path, output: Path, repetitions: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    precheck = verify_source_tree(source, checksums)
    if not precheck["contentValid"]:
        raise RuntimeError(f"pre-execution source validation failed: {precheck}")

    toy_rows: list[dict[str, Any]] = []
    raw_fixture_result: dict[str, Any] | None = None
    for toy in TOYS:
        result = _run(source, checksums, toy)
        checks = {
            "finalValues": result["finalValues"] == toy["expected"],
            "valueMultisetConserved": result["valueMultisetConserved"],
            "notTimedOut": not result["timedOut"],
            "threadShutdownClean": result["threadShutdownClean"],
            "probeLengthsAgree": (
                result["metrics"]["swapCount"]
                == result["metrics"]["recordedSortingSteps"]
                == result["metrics"]["recordedCellTypeSteps"]
            ),
            "expectedSwapCount": (
                toy["expectedSwaps"] is None
                or result["metrics"]["swapCount"] == toy["expectedSwaps"]
            ),
        }
        toy_rows.append(
            {
                **toy,
                "actualFinalValues": result["finalValues"],
                "stopReason": result["stopReason"],
                "metrics": result["metrics"],
                "traceSha256": result["traceSha256"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        if toy["fixture"] == "C-T01":
            raw_fixture_result = result

    if raw_fixture_result is None:
        raise AssertionError("raw fixture was not selected")
    raw_dir = output / "generated_raw_smoke"
    raw_paths = write_run_outputs(raw_fixture_result, raw_dir)
    loaded_steps = np.load(raw_dir / "sorting_steps.npy", allow_pickle=False)
    loaded_types = np.load(raw_dir / "cell_types.npy", allow_pickle=False)
    raw_readability = {
        "classification": "S04-generated smoke output; not historical raw data",
        "sortingSteps": {
            "path": str(raw_dir / "sorting_steps.npy"),
            "shape": list(loaded_steps.shape),
            "dtype": str(loaded_steps.dtype),
            "allow_pickle": False,
            "equalsProbe": loaded_steps.tolist() == raw_fixture_result["sortingSteps"],
        },
        "cellTypes": {
            "path": str(raw_dir / "cell_types.npy"),
            "shape": list(loaded_types.shape),
            "dtype": str(loaded_types.dtype),
            "allow_pickle": False,
            "equalsProbe": loaded_types.tolist() == raw_fixture_result["cellTypes"],
        },
        "files": raw_paths,
    }

    # The frozen no-fault driver batches variable-length trajectories with a
    # bare np.array(...) call. NumPy 1.23 preserves the resulting object-array
    # behavior (with a warning); NumPy 1.24+ raises instead. Exercise that exact
    # compatibility boundary and keep the pickled probe in cache only.
    ragged_runs = [
        run_historical_toy(
            source_root=source,
            checksum_manifest=checksums,
            values=values,
            policy="bubble",
            seed=1729,
            timeout_seconds=10.0,
        )["sortingSteps"]
        for values in ([3, 2, 1], [2, 1, 3])
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        historical_style_batch = np.array(ragged_runs)
    ragged_path = raw_dir / "historical_style_ragged_batch.npy"
    np.save(ragged_path, historical_style_batch)
    untrusted_load_rejected = False
    untrusted_load_error = None
    try:
        np.load(ragged_path, allow_pickle=False)
    except ValueError as error:
        untrusted_load_rejected = True
        untrusted_load_error = str(error)
    trusted_loaded = np.load(ragged_path, allow_pickle=True)
    raw_readability["historicalStyleRaggedBatch"] = {
        "path": str(ragged_path),
        "sha256": sha256_file(ragged_path),
        "dtype": str(historical_style_batch.dtype),
        "shape": list(historical_style_batch.shape),
        "conversionWarnings": [str(item.message) for item in caught],
        "untrustedLoadRejected": untrusted_load_rejected,
        "untrustedLoadError": untrusted_load_error,
        "trustedAllowPickleLoadEqualsProbe": trusted_loaded.tolist() == ragged_runs,
        "classification": "S04-generated trusted compatibility probe; cache-only and not historical raw data",
    }

    scheduler_input = [7, 1, 6, 2, 5, 3, 4, 0]
    scheduler_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        result = run_historical_toy(
            source_root=source,
            checksum_manifest=checksums,
            values=scheduler_input,
            policy="bubble",
            seed=24680,
            timeout_seconds=10.0,
        )
        scheduler_rows.append(
            {
                "repetition": repetition,
                "traceSha256": result["traceSha256"],
                "swapCount": result["metrics"]["swapCount"],
                "compareAndSwapCount": result["metrics"]["compareAndSwapCount"],
                "elapsedSeconds": result["elapsedSeconds"],
                "stopReason": result["stopReason"],
                "finalValues": result["finalValues"],
                "timedOut": result["timedOut"],
                "threadShutdownClean": result["threadShutdownClean"],
            }
        )

    trace_counts = Counter(row["traceSha256"] for row in scheduler_rows)
    swap_counts = Counter(row["swapCount"] for row in scheduler_rows)
    compare_counts = Counter(row["compareAndSwapCount"] for row in scheduler_rows)
    scheduler_summary = {
        "policy": "bubble",
        "inputValues": scheduler_input,
        "fixedSeedOverride": 24680,
        "repetitions": repetitions,
        "uniqueTraceHashes": len(trace_counts),
        "traceHashFrequencies": dict(sorted(trace_counts.items())),
        "uniqueSwapCounts": len(swap_counts),
        "swapCountFrequencies": {str(key): value for key, value in sorted(swap_counts.items())},
        "uniqueCompareAndSwapCounts": len(compare_counts),
        "compareAndSwapCountFrequencies": {
            str(key): value for key, value in sorted(compare_counts.items())
        },
        "schedulerVariabilityDetected": len(trace_counts) > 1 or len(compare_counts) > 1,
        "allCompleted": all(not row["timedOut"] for row in scheduler_rows),
        "allFinalStatesSorted": all(row["finalValues"] == sorted(scheduler_input) for row in scheduler_rows),
        "elapsedSeconds": {
            "min": min(row["elapsedSeconds"] for row in scheduler_rows),
            "max": max(row["elapsedSeconds"] for row in scheduler_rows),
            "mean": sum(row["elapsedSeconds"] for row in scheduler_rows) / repetitions,
        },
        "rowsSha256": _canonical_hash(scheduler_rows),
    }

    postcheck = verify_source_tree(source, checksums)
    source_unchanged = (
        precheck["contentValid"]
        and postcheck["contentValid"]
        and precheck["filesChecked"] == postcheck["filesChecked"]
        and not postcheck["mismatches"]
    )
    validation_checks = {
        "sourcePrecheck": precheck["contentValid"],
        "allKnownToys": all(row["passed"] for row in toy_rows),
        "rawOutputsReadableWithoutPickle": (
            raw_readability["sortingSteps"]["equalsProbe"]
            and raw_readability["cellTypes"]["equalsProbe"]
        ),
        "historicalStyleRaggedOutputReadableWithTrustedPickle": (
            raw_readability["historicalStyleRaggedBatch"]["dtype"] == "object"
            and raw_readability["historicalStyleRaggedBatch"]["untrustedLoadRejected"]
            and raw_readability["historicalStyleRaggedBatch"]["trustedAllowPickleLoadEqualsProbe"]
        ),
        "repeatedRunsComplete": scheduler_summary["allCompleted"],
        "repeatedRunsSorted": scheduler_summary["allFinalStatesSorted"],
        "sourcePostcheck": postcheck["contentValid"],
        "sourceUnchanged": source_unchanged,
    }
    summary = {
        "schema": "e01-s04-validation-v1",
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "executable": sys.executable,
        },
        "precheck": precheck,
        "knownToys": toy_rows,
        "rawOutputReadability": raw_readability,
        "schedulerSummary": scheduler_summary,
        "schedulerRuns": scheduler_rows,
        "postcheck": postcheck,
        "checks": validation_checks,
        "success": all(validation_checks.values()),
    }
    (output / "validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args(argv)
    result = validate(args.source, args.checksums, args.output, args.repetitions)
    print(json.dumps({"success": result["success"], "checks": result["checks"], "schedulerSummary": result["schedulerSummary"]}, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
