#!/usr/bin/env python3
"""Build or validate the E04 S14 aggregation synthesis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.aggregation_classification import build_all, validate_existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--artifact-root", type=Path, default=Path("/artifacts"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--repo", type=Path, default=Path("/workspace/cell-research"))
    args = parser.parse_args()
    if args.command == "build":
        result = build_all(
            artifact_root=args.artifact_root,
            workspace=args.workspace,
            repo=args.repo,
        )
    else:
        result = validate_existing(
            artifact_root=args.artifact_root,
            workspace=args.workspace,
            repo=args.repo,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
