#!/usr/bin/env python3
"""Replay an outcome-blind, scale-stratified sample of completed S10 runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.screening import (
    flat_screening_row,
    materialize_screening_scenario,
    run_compact_screening,
)
from reference_simulator.model import canonical_json_bytes


SETTING_FIELDS = (
    "architecture", "coordinatorProfile", "scheduler", "mobility",
    "continuation", "retry", "actionFailure", "sensing",
    "informationPermission", "legalPrimitives",
    "proposalCandidatesPerOpportunity",
)
COMPARE_FIELDS = (
    "scenarioId", "stopReason", "activationCount", "normalizedResidualError",
    "strictAdjacentResidualCount", "successByBudget", "completionOpportunity",
    "completionObserved", "quiescentCompetingFailure",
    "deterministicResultSha256", "compactReplayDigest",
    "finalOccupancySha256", "selectionCursorsSha256", "nativeLedgerJson",
    "streamCountersJson", "contractValidationPass", "contractValidationJson",
)


def execute(payload: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    design, pairing, scenario_record = payload
    settings = {field: str(design[field]) for field in SETTING_FIELDS}
    scenario = materialize_screening_scenario(scenario_record, pairing, settings)
    return flat_screening_row(design, run_compact_screening(scenario, settings))


def sample_key(run_design_id: str) -> str:
    return hashlib.sha256(f"E02/S10/replay-sample/v1/{run_design_id}".encode()).hexdigest()


def scalar_equal(left: Any, right: Any) -> bool:
    """Compare Parquet null scalars and regenerated Python nulls identically."""

    left_null = left is None or (isinstance(left, float) and pd.isna(left))
    right_null = right is None or (isinstance(right, float) and pd.isna(right))
    if left_null or right_null:
        return left_null and right_null
    return bool(left == right)


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def validate(args: argparse.Namespace) -> None:
    stage_paths = [args.cache / f"stage_{stage:02d}_screening_runs.parquet" for stage in range(1, args.look + 1)]
    if not all(path.exists() for path in stage_paths):
        raise FileNotFoundError("one or more requested completed S10 stage files are absent")
    runs = pd.concat([pd.read_parquet(path) for path in stage_paths], ignore_index=True)
    selected_parts = []
    for n, part in runs.groupby("n", sort=True):
        part = part.copy()
        part["_sampleKey"] = part.runDesignId.map(sample_key)
        selected_parts.append(part.sort_values(["_sampleKey", "runDesignId"]).head(args.per_scale))
    selected = pd.concat(selected_parts, ignore_index=True).sort_values(["n", "_sampleKey"]).reset_index(drop=True)
    expected = args.per_scale * runs.n.nunique()
    if len(selected) != expected:
        raise AssertionError(f"replay sample accounting mismatch: {len(selected)} != {expected}")

    ids = set(selected.runDesignId)
    design_rows = {
        row["runDesignId"]: row
        for row in pq.read_table(args.design).to_pylist()
        if row["runDesignId"] in ids
    }
    pairing_ids = set(selected.pairingBlockId)
    pairing_rows = {
        row["pairingBlockId"]: row
        for row in pq.read_table(args.pairing).to_pylist()
        if row["pairingBlockId"] in pairing_ids
    }
    input_ids = {row["inputScenarioId"] for row in pairing_rows.values()}
    scenario_table = pq.read_table(args.scenario_bank, filters=[("split", "=", "screening_pool")])
    scenario_table = scenario_table.filter(pc.is_in(scenario_table["inputScenarioId"], value_set=pa.array(sorted(input_ids))))
    scenarios = {row["inputScenarioId"]: row for row in scenario_table.to_pylist()}
    if len(design_rows) != expected or len(scenarios) != len(input_ids):
        raise AssertionError("replay input join is incomplete")

    payloads = []
    for row in selected.itertuples(index=False):
        design = design_rows[row.runDesignId]
        pairing = pairing_rows[row.pairingBlockId]
        payloads.append((design, pairing, scenarios[pairing["inputScenarioId"]]))
    if args.workers == 1:
        replays = [execute(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            replays = list(pool.map(execute, payloads, chunksize=1))
    replay_by_id = {row["runDesignId"]: row for row in replays}

    dynamic_fields = sorted(
        column for column in runs.columns
        if column.startswith(("cost_", "projection_", "normalization_"))
        and column != "cost_wallTimeSeconds"
    )
    compare_fields = (*COMPARE_FIELDS, *dynamic_fields)
    evidence = []
    for original in selected.to_dict(orient="records"):
        replay = replay_by_id[original["runDesignId"]]
        checks = {field: scalar_equal(original[field], replay[field]) for field in compare_fields}
        evidence.append({
            "runDesignId": original["runDesignId"],
            "pairingBlockId": original["pairingBlockId"],
            "n": int(original["n"]),
            "treatmentSignature": original["treatmentSignature"],
            "sampleKey": original["_sampleKey"],
            "fieldsChecked": len(checks),
            "fieldsPassed": sum(checks.values()),
            "success": all(checks.values()),
            "checksJson": canonical_json_bytes(checks).decode("utf-8"),
        })
    pq.write_table(
        pa.Table.from_pylist(evidence),
        args.output / "replay_validation_samples.parquet",
        compression="zstd", compression_level=9, use_dictionary=True,
        write_statistics=True, version="2.6",
    )
    summary = {
        "schemaVersion": "e02.s10.replay_validation.v1",
        "researchStepId": "S10",
        "selectionRule": "four smallest SHA-256 outcome-blind runDesignId keys per scale",
        "look": args.look,
        "samples": len(evidence),
        "samplesPerScale": {str(n): int(sum(row["n"] == n for row in evidence)) for n in sorted(runs.n.unique())},
        "fieldsPerSample": len(compare_fields),
        "successfulSamples": sum(row["success"] for row in evidence),
        "protectedOutcomeReads": 0,
        "validationResult": "PASS",
        "success": True,
    }
    write_json(args.output / "replay_validation.json", summary)
    print(json.dumps(summary, sort_keys=True))
    if not all(row["success"] for row in evidence):
        failures = [row for row in evidence if not row["success"]]
        raise AssertionError(f"one or more deterministic replay sample checks failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--look", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--per-scale", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--cache", type=Path, default=Path("/cache/s10"))
    parser.add_argument("--design", type=Path, default=Path("/artifacts/research_steps/S10/screening_design.parquet"))
    parser.add_argument("--pairing", type=Path, default=Path("/artifacts/research_steps/S10/screening_pairing_blocks.parquet"))
    parser.add_argument("--scenario-bank", type=Path, default=Path("/artifacts/research_steps/S07/scenario_extension.parquet"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S10"))
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
