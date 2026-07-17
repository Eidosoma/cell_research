#!/usr/bin/env python3
"""Command-line driver for the frozen E04 S06 dynamic-null workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dynamic_nulls import (  # noqa: E402
    analyze_dynamic_nulls,
    assert_frozen,
    build_replay_tasks,
    freeze_design,
    package_dynamic_nulls,
    package_replay_audit,
    run_condition_nulls,
    run_deterministic_audit,
    run_replay,
    validate_artifacts,
    write_artifact_manifest,
    write_provenance,
)


OUTPUT = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "research_steps/S06"
CACHE = Path("/cache/e04_s06")
REPLAY_CHECKPOINT = CACHE / "trajectory_replay.jsonl"
NULL_DIR = CACHE / "condition_nulls"
DETERMINISTIC_DIR = CACHE / "deterministic_rerun"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "replay",
            "replay-package",
            "nulls",
            "deterministic-audit",
            "package",
            "analyze",
            "provenance",
            "validate",
            "manifest",
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_design(OUTPUT, CACHE)
    elif args.command == "replay":
        assert_frozen(OUTPUT)
        result = run_replay(
            build_replay_tasks(), REPLAY_CHECKPOINT, workers=args.workers
        )
    elif args.command == "replay-package":
        assert_frozen(OUTPUT)
        result = package_replay_audit(REPLAY_CHECKPOINT, OUTPUT)
    elif args.command == "nulls":
        assert_frozen(OUTPUT)
        result = run_condition_nulls(
            REPLAY_CHECKPOINT, NULL_DIR, workers=args.workers
        )
    elif args.command == "deterministic-audit":
        assert_frozen(OUTPUT)
        result = run_deterministic_audit(
            REPLAY_CHECKPOINT, NULL_DIR, DETERMINISTIC_DIR, OUTPUT
        )
    elif args.command == "package":
        result = package_dynamic_nulls(NULL_DIR, OUTPUT)
    elif args.command == "analyze":
        result = analyze_dynamic_nulls(OUTPUT)
    elif args.command == "provenance":
        result = write_provenance(OUTPUT, CACHE)
    elif args.command == "validate":
        result = validate_artifacts(OUTPUT)
    elif args.command == "manifest":
        result = write_artifact_manifest(OUTPUT)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
