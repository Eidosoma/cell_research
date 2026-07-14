#!/usr/bin/env python3
"""Build or audit the E01 S14 baseline release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.baseline_release import (  # noqa: E402
    build_release,
    finalize_artifact_manifest,
    regenerate_figures,
    reproduce,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("build", "figures", "reproduce", "finalize"))
    value.add_argument("--artifact-root", type=Path, default=Path("/artifacts"))
    value.add_argument("--output", type=Path)
    value.add_argument("--release", type=Path)
    value.add_argument("--report-inputs", type=Path)
    value.add_argument("--scratch", type=Path, default=Path("/cache/e01_s14/reproduction"))
    value.add_argument("--commit")
    value.add_argument("--write-collectible-audits", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    output = args.output or args.artifact_root / "research_steps" / "S14"
    release = args.release or args.artifact_root / "release" / "baseline"
    report_inputs = args.report_inputs or args.artifact_root / "report_inputs"
    if args.command == "build":
        result = build_release(output, release, report_inputs, args.commit)
    elif args.command == "figures":
        result = regenerate_figures(args.scratch)
    elif args.command == "reproduce":
        result = reproduce(
            args.artifact_root,
            args.scratch,
            write_collectible_audits=output if args.write_collectible_audits else None,
        )
    else:
        result = finalize_artifact_manifest(output, release, report_inputs)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("success", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
