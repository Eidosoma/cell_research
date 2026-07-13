#!/usr/bin/env python3
"""CLI for the S08 immutable paired scenario bank."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenario_bank.builder import build_bank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stable-output", type=Path)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    build_bank(
        args.claim_registry.resolve(), args.output.resolve(), repository=ROOT,
        stable_output=args.stable_output.resolve() if args.stable_output else None,
        compare_to=args.compare_to.resolve() if args.compare_to else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
