"""Small command-line interface for stable scenario/result JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .api import run_scenario
from .model import Scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="canonical scenario JSON")
    parser.add_argument("--trace-mode", choices=("full", "digest", "none"), default="digest")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    scenario = Scenario.from_json_bytes(args.scenario.read_bytes())
    result = run_scenario(scenario, trace_mode=args.trace_mode)
    payload = result.to_json_bytes() + b"\n"
    if args.output:
        args.output.write_bytes(payload)
    else:
        sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
