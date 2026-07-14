#!/usr/bin/env python3
"""Execute one frozen S10 nested screening increment with restartable chunks."""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

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


SETTING_FIELDS = (
    "architecture",
    "coordinatorProfile",
    "scheduler",
    "mobility",
    "continuation",
    "retry",
    "actionFailure",
    "sensing",
    "informationPermission",
    "legalPrimitives",
    "proposalCandidatesPerOpportunity",
)


def write_parquet(path: Path, rows: list[Mapping[str, Any]]) -> None:
    pq.write_table(
        pa.Table.from_pylist(list(rows)),
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def execute(payload: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    run_design, pairing, scenario_record = payload
    settings = {field: str(run_design[field]) for field in SETTING_FIELDS}
    scenario = materialize_screening_scenario(scenario_record, pairing, settings)
    result = run_compact_screening(scenario, settings)
    return flat_screening_row(run_design, result)


def run_stage(args: argparse.Namespace) -> None:
    if args.stage not in range(1, 6):
        raise ValueError("stage must be in [1,5]")
    args.cache.mkdir(parents=True, exist_ok=True)
    design = pq.read_table(
        args.design,
        filters=[("screeningStage", "=", args.stage)],
    ).to_pylist()
    pairing_rows = pq.read_table(
        args.pairing,
        filters=[("screeningStage", "=", args.stage)],
    ).to_pylist()
    pairing = {str(row["pairingBlockId"]): row for row in pairing_rows}
    input_ids = {str(row["inputScenarioId"]) for row in pairing_rows}
    scenario_table = pq.read_table(
        args.scenario_bank,
        filters=[("split", "=", "screening_pool")],
    )
    mask = pc.is_in(
        scenario_table["inputScenarioId"],
        value_set=pa.array(sorted(input_ids)),
    )
    scenarios = {
        str(row["inputScenarioId"]): row
        for row in scenario_table.filter(mask).to_pylist()
    }
    if len(design) != 700 or len(pairing) != 50 or len(scenarios) != 50:
        raise AssertionError(
            f"stage accounting mismatch: design={len(design)}, pairing={len(pairing)}, scenarios={len(scenarios)}"
        )
    design.sort(key=lambda row: row["runDesignId"])
    stage_paths = sorted(args.cache.glob(f"stage_{args.stage:02d}_chunk_*.parquet"))
    existing_rows = []
    for path in stage_paths:
        existing_rows.extend(pq.read_table(path).to_pylist())
    existing_ids = [row["runDesignId"] for row in existing_rows]
    if len(existing_ids) != len(set(existing_ids)):
        raise AssertionError("duplicate runDesignId across restart checkpoints")
    valid_ids = {row["runDesignId"] for row in design}
    if not set(existing_ids).issubset(valid_ids):
        raise AssertionError("restart checkpoint contains a run outside this stage")
    if existing_rows:
        print(json.dumps({"stage": args.stage, "status": "resume_existing", "checkpoints": len(stage_paths), "rows": len(existing_rows)}, sort_keys=True), flush=True)

    pending = [row for row in design if row["runDesignId"] not in set(existing_ids)]
    payloads = [
            (
                dict(row),
                dict(pairing[str(row["pairingBlockId"])]),
                dict(scenarios[str(row["inputScenarioId"])]),
            )
            for row in pending
    ]
    next_checkpoint = max(
        (int(path.stem.rsplit("_", 1)[1]) for path in stage_paths),
        default=-1,
    ) + 1
    buffer: list[dict[str, Any]] = []

    def checkpoint(*, final: bool = False) -> None:
        nonlocal next_checkpoint, buffer
        while len(buffer) >= args.chunk_size or (final and buffer):
            count = args.chunk_size if len(buffer) >= args.chunk_size else len(buffer)
            rows = sorted(buffer[:count], key=lambda row: row["runDesignId"])
            del buffer[:count]
            if not all(row["contractValidationPass"] for row in rows):
                raise AssertionError("stage checkpoint failed contract validation")
            target = args.cache / f"stage_{args.stage:02d}_chunk_{next_checkpoint:03d}.parquet"
            if target.exists():
                raise AssertionError(f"refusing to overwrite restart checkpoint: {target}")
            write_parquet(target, rows)
            print(json.dumps({"stage": args.stage, "chunk": next_checkpoint, "status": "written_dynamic", "rows": len(rows), "wallTimeSeconds": sum(float(row["wallTimeSeconds"]) for row in rows)}, sort_keys=True), flush=True)
            next_checkpoint += 1

    if args.workers == 1:
        for payload in payloads:
            buffer.append(execute(payload))
            checkpoint()
    elif payloads:
        # Submit the whole pending increment once so long n=500 runs cannot leave
        # seven workers idle at arbitrary file-checkpoint boundaries. Completion-
        # ordered persistence changes neither run bytes nor scientific ordering.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute, payload) for payload in payloads]
            for future in as_completed(futures):
                buffer.append(future.result())
                checkpoint()
    checkpoint(final=True)

    stage_paths = sorted(args.cache.glob(f"stage_{args.stage:02d}_chunk_*.parquet"))
    tables = [pq.read_table(path) for path in stage_paths]
    combined = pa.concat_tables(tables).sort_by([("runDesignId", "ascending")])
    if combined.num_rows != 700 or len(set(combined["runDesignId"].to_pylist())) != 700:
        raise AssertionError("combined stage run accounting failed")
    stage_output = args.cache / f"stage_{args.stage:02d}_screening_runs.parquet"
    pq.write_table(
        combined,
        stage_output,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    print(json.dumps({"stage": args.stage, "status": "complete", "rows": combined.num_rows, "output": str(stage_output)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=28)
    parser.add_argument("--design", type=Path, default=Path("/artifacts/research_steps/S10/screening_design.parquet"))
    parser.add_argument("--pairing", type=Path, default=Path("/artifacts/research_steps/S10/screening_pairing_blocks.parquet"))
    parser.add_argument("--scenario-bank", type=Path, default=Path("/artifacts/research_steps/S07/scenario_extension.parquet"))
    parser.add_argument("--cache", type=Path, default=Path("/cache/s10"))
    run_stage(parser.parse_args())


if __name__ == "__main__":
    main()
