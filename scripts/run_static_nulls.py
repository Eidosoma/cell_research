#!/usr/bin/env python3
"""Restartable E04 S05 static permutation-null workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.static_nulls import (  # noqa: E402
    analyze_static_nulls,
    assert_frozen,
    build_conditioned_null_artifacts,
    build_full_pointwise_global_deviations,
    build_global_null_corpus,
    build_snapshot_artifact,
    build_snapshot_tasks,
    freeze_design,
    replay_snapshot_population,
    run_conditioned_nulls,
    run_uniformity_validation,
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
            "uniformity",
            "global-null",
            "snapshot-run",
            "snapshot-package",
            "conditioned-null",
            "pointwise-global",
            "conditioned-package",
            "analyze",
            "validate",
            "provenance",
            "manifest",
        ),
    )
    parser.add_argument("--cache", type=Path, default=Path("/cache/e04_s05"))
    parser.add_argument(
        "--output", type=Path, default=Path("/artifacts/research_steps/S05")
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.command == "freeze":
        result = freeze_design(args.output, args.cache)
    elif args.command == "uniformity":
        assert_frozen(args.output)
        result = run_uniformity_validation(args.output)
    elif args.command == "global-null":
        assert_frozen(args.output)
        result = build_global_null_corpus(
            args.output, args.cache, workers=args.workers
        )
    elif args.command == "snapshot-run":
        assert_frozen(args.output)
        result = replay_snapshot_population(
            build_snapshot_tasks(),
            args.cache / "snapshot_states.jsonl",
            workers=args.workers,
        )
    elif args.command == "snapshot-package":
        assert_frozen(args.output)
        result = build_snapshot_artifact(
            args.cache / "snapshot_states.jsonl", args.output
        )
    elif args.command == "conditioned-null":
        assert_frozen(args.output)
        result = run_conditioned_nulls(
            args.output / "snapshot_states.parquet",
            args.cache / "value_conditioned_nulls.jsonl",
            workers=args.workers,
        )
    elif args.command == "pointwise-global":
        assert_frozen(args.output)
        result = build_full_pointwise_global_deviations(args.output)
    elif args.command == "conditioned-package":
        assert_frozen(args.output)
        result = build_conditioned_null_artifacts(
            args.cache / "value_conditioned_nulls.jsonl", args.output
        )
    elif args.command == "analyze":
        assert_frozen(args.output)
        result = analyze_static_nulls(args.output)
    elif args.command == "validate":
        assert_frozen(args.output)
        result = validate_artifacts(args.output)
    elif args.command == "provenance":
        assert_frozen(args.output)
        result = write_provenance(args.output, args.cache)
    elif args.command == "manifest":
        assert_frozen(args.output)
        result = write_artifact_manifest(args.output)
    else:
        raise NotImplementedError(args.command)
    print(json.dumps({"command": args.command, **result}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
