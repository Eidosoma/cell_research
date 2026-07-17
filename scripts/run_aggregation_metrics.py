#!/usr/bin/env python3
"""Restartable E04 S04 multi-metric workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregation_metrics import (  # noqa: E402
    analyze_metrics,
    assert_frozen,
    build_metric_tables,
    build_tasks,
    freeze_design,
    run_population,
    run_validation,
    validate_artifacts,
    write_artifact_manifest,
    write_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "static-validation",
            "run",
            "analyze",
            "validate",
            "provenance",
            "manifest",
        ),
    )
    parser.add_argument("--cache", type=Path, default=Path("/cache/e04_s04"))
    parser.add_argument(
        "--output", type=Path, default=Path("/artifacts/research_steps/S04")
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.command == "freeze":
        result = freeze_design(args.output, args.cache)
    elif args.command == "static-validation":
        assert_frozen(args.output)
        result = run_validation(args.output, workers=args.workers)
    elif args.command == "run":
        assert_frozen(args.output)
        result = run_population(
            build_tasks(), args.cache / "aggregation_metrics.jsonl", workers=args.workers
        )
    elif args.command == "analyze":
        assert_frozen(args.output)
        checkpoint = args.cache / "aggregation_metrics.jsonl"
        table_result = build_metric_tables(checkpoint, args.output)
        analysis_result = analyze_metrics(args.output)
        result = {**table_result, **analysis_result}
    elif args.command == "validate":
        assert_frozen(args.output)
        result = validate_artifacts(args.output)
    elif args.command == "provenance":
        assert_frozen(args.output)
        result = write_provenance(args.output)
    elif args.command == "manifest":
        assert_frozen(args.output)
        result = write_artifact_manifest(args.output)
    else:
        raise NotImplementedError(f"{args.command} is not yet wired")
    print(json.dumps({"command": args.command, **result}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
