#!/usr/bin/env python3
"""Replay an outcome-blind, scale-by-treatment S11 sample exactly."""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.screening import materialize_screening_scenario, run_compact_screening
from reference_simulator.model import canonical_json_bytes
from scripts.run_confirmatory_stage import SETTING_FIELDS


def sample_key(run_id: str) -> str:
    return hashlib.sha256(f"S11/replay/{run_id}".encode()).hexdigest()


def load_arrow_records(
    pairing_path: Path, scenario_path: Path, input_ids: list[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load nested replay records without pandas' ndarray coercion."""

    pairing = {
        str(row["pairingBlockId"]): row
        for row in pq.read_table(pairing_path).to_pylist()
    }
    scenario_table = pq.read_table(
        scenario_path, filters=[("split", "=", "confirmatory_holdout")]
    )
    scenario_table = scenario_table.filter(
        pc.is_in(scenario_table["inputScenarioId"], value_set=pa.array(input_ids))
    )
    scenarios = {
        str(row["inputScenarioId"]): row for row in scenario_table.to_pylist()
    }
    return pairing, scenarios


def replay_one(
    payload: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
) -> dict[str, Any]:
    row, pairing_row, scenario_row = payload
    settings = {field: str(row[field]) for field in SETTING_FIELDS}
    first = run_compact_screening(
        materialize_screening_scenario(scenario_row, pairing_row, settings), settings
    )
    second = run_compact_screening(
        materialize_screening_scenario(scenario_row, pairing_row, settings), settings
    )
    observed = {
        "deterministicResultSha256": first["deterministicResultSha256"],
        "compactReplayDigest": first["compactReplayDigest"],
        "scenarioId": first["scenarioId"],
        "stopReason": first["stopReason"],
        "activationCount": first["activationCount"],
        "nativeLedgerJson": canonical_json_bytes(first["nativeLedger"]).decode(),
        "finalOccupancySha256": __import__(
            "reference_simulator.model", fromlist=["sha256_json"]
        ).sha256_json(first["finalOccupancy"]),
        "streamCountersJson": canonical_json_bytes(first["streamCounters"]).decode(),
        "projection_s01UnitWeightFullCost": first["projections"]["s01UnitWeightFullCost"],
    }
    comparisons = (
        "deterministicResultSha256", "compactReplayDigest", "scenarioId", "stopReason",
        "activationCount", "nativeLedgerJson", "finalOccupancySha256",
        "streamCountersJson", "projection_s01UnitWeightFullCost",
    )
    matches = {name: bool(observed[name] == row[name]) for name in comparisons}
    repeat_equal = bool(
        first["deterministicResultSha256"] == second["deterministicResultSha256"]
    )
    return {
        "runDesignId": str(row["runDesignId"]), "n": int(row["n"]),
        "treatmentSignature": str(row["treatmentSignature"]),
        "comparisonPass": matches, "repeatReplayPass": repeat_equal,
        "success": bool(all(matches.values()) and repeat_equal),
    }


def validate(args: argparse.Namespace) -> None:
    runs = pd.read_parquet(args.results)
    input_ids = sorted(set(runs.inputScenarioId.astype(str)))
    pairing, scenarios = load_arrow_records(args.pairing, args.scenario_bank, input_ids)
    samples = []
    for (_, _), group in runs.groupby(["n", "treatmentSignature"], sort=True):
        samples.append(group.assign(_key=group.runDesignId.map(sample_key)).sort_values("_key").iloc[0])
    args.cache.mkdir(parents=True, exist_ok=True)
    payloads = []
    records = []
    for sample in samples:
        row = sample.drop(labels=["_key"]).to_dict()
        checkpoint = args.cache / f"{sample_key(str(sample.runDesignId))}.json"
        if checkpoint.is_file():
            record = json.loads(checkpoint.read_text())
            if record["runDesignId"] != str(sample.runDesignId):
                raise AssertionError("replay checkpoint address collision")
            records.append(record)
        else:
            payloads.append((
                row,
                pairing[str(sample.pairingBlockId)],
                scenarios[str(sample.inputScenarioId)],
            ))
    if payloads:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(replay_one, payload) for payload in payloads]
            for future in as_completed(futures):
                record = future.result()
                checkpoint = args.cache / f"{sample_key(record['runDesignId'])}.json"
                checkpoint.write_bytes(canonical_json_bytes(record) + b"\n")
                records.append(record)
                print(json.dumps({
                    "runDesignId": record["runDesignId"], "n": record["n"],
                    "success": record["success"],
                }, sort_keys=True), flush=True)
    records.sort(key=lambda item: item["runDesignId"])
    summary = {
        "schemaVersion": "e02.s11.deterministic_replay_validation.v1",
        "researchStepId": "S11", "selectionRule": "lowest SHA256(S11/replay/runDesignId) within n-by-treatmentSignature",
        "sampleRows": len(records), "scaleCount": runs.n.nunique(),
        "treatmentSignatureCount": runs.treatmentSignature.nunique(),
        "records": records, "success": all(item["success"] for item in records),
    }
    args.output.write_bytes(canonical_json_bytes(summary) + b"\n")
    print(json.dumps({"sampleRows": len(records), "success": summary["success"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("/artifacts/research_steps/S11/confirmatory_results.parquet"))
    parser.add_argument("--pairing", type=Path, default=Path("/artifacts/research_steps/S11/confirmatory_pairing_blocks.parquet"))
    parser.add_argument("--scenario-bank", type=Path, default=Path("/artifacts/research_steps/S07/scenario_extension.parquet"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S11/deterministic_replay_validation.json"))
    parser.add_argument("--cache", type=Path, default=Path("/cache/s11/replay_validation"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
