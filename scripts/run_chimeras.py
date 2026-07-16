#!/usr/bin/env python3
"""Restartable E04 S01 chimeric baseline runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregation import (  # noqa: E402
    build_artifacts,
    load_tasks,
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
        choices=("run", "replay", "analyze", "validate", "provenance", "manifest"),
    )
    parser.add_argument("--cache", type=Path, default=Path("/cache/e04_s01"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S01"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.command == "validate":
        result = validate_artifacts(args.output)
        print(json.dumps({"command": "validate", "success": result["success"], "checks": result["checkCount"]}))
        return 0
    if args.command == "provenance":
        write_provenance(args.output)
        print(json.dumps({"command": "provenance", "success": True}))
        return 0
    if args.command == "manifest":
        result = write_artifact_manifest(args.output)
        print(json.dumps({"command": "manifest", "artifacts": result["artifactCount"]}))
        return 0

    tasks = load_tasks()
    if args.command == "run":
        results = run_population(tasks, args.cache / "original_mixtures.jsonl", workers=args.workers)
        print(json.dumps({"command": "run", "rows": len(results)}))
        return 0
    if args.command == "replay":
        rows = replay_selected(tasks, workers=args.workers)
        (args.cache / "deterministic_replays.json").write_text(
            json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"command": "replay", "rows": len(rows)}))
        return 0

    def read_jsonl(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    results = read_jsonl(args.cache / "original_mixtures.jsonl")
    replays = json.loads((args.cache / "deterministic_replays.json").read_text())
    built = build_artifacts(results, replays, args.output)
    print(json.dumps({"command": "analyze", **built}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
