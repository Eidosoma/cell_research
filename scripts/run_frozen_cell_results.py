#!/usr/bin/env python3
"""Execute restartable S11 stages and materialize compact evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.frozen_cell_results import (  # noqa: E402
    C_PROFILE,
    CONFIRM_SPLIT,
    HISTORICAL_SOURCE,
    PAPER_SPLIT,
    R_PROFILE,
    S02_CHECKSUMS,
    analyze,
    environment_record,
    hand_counted_fault_toys,
    load_tasks,
    preregistration_sha256,
    run_exact_replay_samples,
    run_historical_population,
    run_reference_population,
    sha256_file,
    verify_source_tree,
    write_json,
    write_parquet,
)


CACHE = Path("/cache/e01_s11")
OUTPUT = Path("/artifacts/research_steps/S11")


def load_all_checkpoints():
    rows = []
    for name in ("reference_paper.jsonl", "reference_confirmatory.jsonl", "historical_paper.jsonl"):
        path = CACHE / name
        if not path.exists():
            raise FileNotFoundError(f"missing completed S11 checkpoint: {path}")
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "stage",
        choices=("validate", "reference-paper", "reference-confirmatory", "historical-paper", "replay", "analyze"),
    )
    value.add_argument("--workers", type=int, default=8)
    return value


def main():
    args = parser().parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prereg = preregistration_sha256()
    if args.stage == "validate":
        toys = hand_counted_fault_toys()
        write_json(OUTPUT / "fault_semantics_toys.json", toys)
        source = verify_source_tree(HISTORICAL_SOURCE, S02_CHECKSUMS)
        write_json(OUTPUT / "source_checksum_pre_run.json", source)
        write_json(OUTPUT / "environment.json", environment_record({"reference": 8, "historical": 8, "replay": 8}))
        print(json.dumps({"preregistrationSha256": prereg, "toys": toys, "source": source["contentValid"]}, indent=2))
        return 0 if toys["success"] and source["contentValid"] else 1
    if args.stage == "reference-paper":
        tasks = load_tasks(PAPER_SPLIT, "reference")
        rows = run_reference_population(tasks, CACHE / "reference_paper.jsonl", workers=args.workers)
        print(json.dumps({"backend": R_PROFILE, "split": PAPER_SPLIT, "runs": len(rows)}))
        return 0
    if args.stage == "reference-confirmatory":
        tasks = load_tasks(CONFIRM_SPLIT, "reference", preregistration_hash=prereg)
        rows = run_reference_population(tasks, CACHE / "reference_confirmatory.jsonl", workers=args.workers)
        print(json.dumps({"backend": R_PROFILE, "split": CONFIRM_SPLIT, "runs": len(rows)}))
        return 0
    if args.stage == "historical-paper":
        tasks = load_tasks(PAPER_SPLIT, "historical")
        rows = run_historical_population(tasks, CACHE / "historical_paper.jsonl", workers=min(args.workers, 8))
        source = verify_source_tree(HISTORICAL_SOURCE, S02_CHECKSUMS)
        write_json(OUTPUT / "source_checksum_post_run.json", source)
        print(json.dumps({"backend": C_PROFILE, "split": PAPER_SPLIT, "runs": len(rows), "sourceValid": source["contentValid"]}))
        return 0 if source["contentValid"] else 1
    if args.stage == "replay":
        tasks = load_tasks(PAPER_SPLIT, "reference")
        rows = run_exact_replay_samples(tasks, workers=args.workers)
        write_json(OUTPUT / "exact_replay_samples.json", rows)
        print(json.dumps({"samples": len(rows), "success": all(row["byteExactDigestReplay"] and row["fastSummaryMatchesDigest"] for row in rows)}))
        return 0 if all(row["byteExactDigestReplay"] and row["fastSummaryMatchesDigest"] for row in rows) else 1
    rows = load_all_checkpoints()
    write_parquet(rows, OUTPUT / "frozen_cell_results.parquet")
    import pandas as pd

    frame = pd.DataFrame(rows)
    result = analyze(frame, OUTPUT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["validationSuccess"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
