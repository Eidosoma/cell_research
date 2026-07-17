#!/usr/bin/env python3
"""Restartable command-line driver for E04 S07 kinetic matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from analysis.kinetic_matching import (  # noqa: E402
    OUTPUT_DIR,
    analyze_results,
    calibrate_interventions,
    freeze_design,
    package_results,
    run_dynamic_nulls,
    run_holdout,
    validate_artifacts,
    write_artifact_manifest,
    write_commands,
    write_environment,
    write_provenance,
)


DEFAULT_CACHE = Path("/cache/e04_s07")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "command",
        choices=(
            "freeze",
            "calibrate",
            "holdout",
            "nulls",
            "package",
            "analyze",
            "environment",
            "commands",
            "provenance",
            "validate",
            "manifest",
        ),
    )
    value.add_argument("--output", type=Path, default=OUTPUT_DIR)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--workers", type=int, default=8)
    return value


def main() -> None:
    args = parser().parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be in [1, 8]")
    if args.command == "freeze":
        result = freeze_design(args.output, args.cache)
    elif args.command == "calibrate":
        result = calibrate_interventions(args.output, args.cache, args.workers)
    elif args.command == "holdout":
        result = run_holdout(args.output, args.cache, args.workers)
    elif args.command == "nulls":
        result = run_dynamic_nulls(args.output, args.cache, args.workers)
    elif args.command == "package":
        result = package_results(args.output, args.cache)
    elif args.command == "analyze":
        result = analyze_results(args.output, args.cache)
    elif args.command == "environment":
        result = write_environment(args.output, args.workers)
    elif args.command == "commands":
        result = write_commands(args.output)
    elif args.command == "provenance":
        result = write_provenance(args.output, args.cache)
    elif args.command == "validate":
        result = validate_artifacts(args.output)
    else:
        result = write_artifact_manifest(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
