#!/usr/bin/env python3
"""Execute restartable S12 metric, replay, analysis, and validation stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.delayed_gratification import (  # noqa: E402
    CACHE_DIR,
    FIGURE6_RUNS,
    FIGURE7_RUNS,
    HISTORICAL_SOURCE,
    OUTPUT_DIR,
    PREREGISTRATION,
    PREREGISTRATION_SHA256,
    S02_CHECKSUMS,
    analyze,
    environment_record,
    frozen_code_randomized_comparison,
    hand_calculated_fixtures,
    load_figure6_tasks,
    load_figure7_tasks,
    replay_validation,
    run_population,
    s09_no_fault_path_comparison,
    sha256_file,
    verify_source_tree,
    write_json,
)


FIGURE7_CHECKPOINT = CACHE_DIR / "figure7_paper_scale.jsonl"
FIGURE6_CHECKPOINT = CACHE_DIR / "figure6_ambiguity_envelope.jsonl"


def _load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "stage",
        choices=("validate", "figure7", "figure6", "replay", "analyze", "final-validation"),
    )
    value.add_argument("--workers", type=int, default=8)
    return value


def main() -> int:
    args = parser().parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8]")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise PermissionError("S12 preregistration hash mismatch")

    if args.stage == "validate":
        fixtures = hand_calculated_fixtures()
        comparison = frozen_code_randomized_comparison()
        source = verify_source_tree(HISTORICAL_SOURCE, S02_CHECKSUMS)
        write_json(OUTPUT_DIR / "hand_calculated_metric_fixtures.json", fixtures)
        write_json(OUTPUT_DIR / "frozen_code_metric_comparison.json", comparison)
        write_json(OUTPUT_DIR / "source_checksum_pre_run.json", source)
        write_json(OUTPUT_DIR / "environment.json", environment_record(args.workers))
        success = fixtures["success"] and comparison["success"] and source["contentValid"]
        print(json.dumps({"success": success, "fixtures": fixtures["fixtureCount"], "randomComparisons": comparison["samples"]}, indent=2))
        return 0 if success else 1

    if args.stage == "figure7":
        tasks = load_figure7_tasks()
        rows = run_population(tasks, FIGURE7_CHECKPOINT, workers=args.workers)
        print(json.dumps({"population": "figure7_paper_scale", "runs": len(rows)}))
        return 0 if len(rows) == FIGURE7_RUNS else 1

    if args.stage == "figure6":
        tasks = load_figure6_tasks()
        rows = run_population(tasks, FIGURE6_CHECKPOINT, workers=args.workers)
        print(json.dumps({"population": "figure6_ambiguity_envelope", "runs": len(rows)}))
        return 0 if len(rows) == FIGURE6_RUNS else 1

    if args.stage == "replay":
        tasks = load_figure7_tasks()
        result = replay_validation(tasks)
        write_json(OUTPUT_DIR / "exact_replay_samples.json", result)
        print(json.dumps({"success": result["success"], "samples": result["samples"]}))
        return 0 if result["success"] else 1

    figure7 = _load_checkpoint(FIGURE7_CHECKPOINT)
    figure6 = _load_checkpoint(FIGURE6_CHECKPOINT)
    if args.stage == "analyze":
        result = analyze(figure7, figure6, OUTPUT_DIR)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["populationValidation"] else 1

    import pandas as pd

    frame = pd.read_parquet(OUTPUT_DIR / "original_dg_results.parquet")
    path_comparison = s09_no_fault_path_comparison(frame)
    source = verify_source_tree(HISTORICAL_SOURCE, S02_CHECKSUMS)
    fixtures = json.loads((OUTPUT_DIR / "hand_calculated_metric_fixtures.json").read_text())
    frozen = json.loads((OUTPUT_DIR / "frozen_code_metric_comparison.json").read_text())
    replay = json.loads((OUTPUT_DIR / "exact_replay_samples.json").read_text())
    population = json.loads((OUTPUT_DIR / "population_validation.json").read_text())
    write_json(OUTPUT_DIR / "s09_no_fault_path_comparison.json", path_comparison)
    write_json(OUTPUT_DIR / "source_checksum_post_run.json", source)
    checks = {
        "preregistrationHash": sha256_file(PREREGISTRATION) == PREREGISTRATION_SHA256,
        "handFixtures": fixtures["success"],
        "frozenCodeRandomized": frozen["success"],
        "exactReplay24": replay["success"] and replay["samples"] == 24,
        "s09NoFaultCodePath": path_comparison["success"],
        "population": population["success"],
        "sourcePre": json.loads((OUTPUT_DIR / "source_checksum_pre_run.json").read_text())["contentValid"],
        "sourcePost": source["contentValid"],
    }
    validation = {
        "schemaVersion": "e01.s12.validation.v1",
        "researchStepId": "S12",
        "success": all(checks.values()),
        "checks": checks,
        "runAccounting": {"figure7": len(figure7), "figure6": len(figure6), "total": len(figure7) + len(figure6)},
        "unexplainedFailures": [],
    }
    write_json(OUTPUT_DIR / "validation_summary.json", validation)
    print(json.dumps(validation, indent=2))
    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
