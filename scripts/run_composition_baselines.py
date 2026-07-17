#!/usr/bin/env python3
"""Restartable E04 S02 composition-baseline workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregation_baselines import (  # noqa: E402
    validate_artifacts,
    write_artifact_manifest,
    write_corrected_artifacts,
    write_exhaustive_artifacts,
    write_monte_carlo_artifacts,
    write_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "exhaustive",
            "monte-carlo",
            "correct",
            "validate",
            "provenance",
            "manifest",
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/artifacts/research_steps/S02")
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.command == "exhaustive":
        result = write_exhaustive_artifacts(args.output)
    elif args.command == "monte-carlo":
        result = write_monte_carlo_artifacts(args.output, args.workers)
    elif args.command == "correct":
        result = write_corrected_artifacts(args.output)
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
