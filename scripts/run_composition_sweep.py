#!/usr/bin/env python3
"""Restartable E04 S03 composition-factorial workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.composition_sweep import (  # noqa: E402
    assert_frozen,
    build_artifacts,
    build_tasks,
    freeze_design,
    replay_selected,
    run_population,
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
            "run",
            "replay",
            "analyze",
            "validate",
            "provenance",
            "manifest",
        ),
    )
    parser.add_argument("--cache", type=Path, default=Path("/cache/e04_s03"))
    parser.add_argument(
        "--output", type=Path, default=Path("/artifacts/research_steps/S03")
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.cache / "composition_sweep.jsonl"

    if args.command == "freeze":
        result = freeze_design(args.output, args.cache)
    elif args.command == "run":
        assert_frozen(args.output)
        result = run_population(build_tasks(), checkpoint, workers=args.workers)
    elif args.command == "replay":
        assert_frozen(args.output)
        result = replay_selected(
            build_tasks(), checkpoint, args.output, workers=args.workers
        )
    elif args.command == "analyze":
        result = build_artifacts(checkpoint, args.output)
    elif args.command == "validate":
        result = validate_artifacts(args.output)
    elif args.command == "provenance":
        result = write_provenance(args.output)
    else:
        result = write_artifact_manifest(args.output)
    print(json.dumps({"command": args.command, **result}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
