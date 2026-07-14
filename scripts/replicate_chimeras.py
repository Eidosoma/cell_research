#!/usr/bin/env python3
"""Restartable S13 orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.chimeric_replication import (  # noqa: E402
    build_artifacts,
    exact_replay_selected,
    load_tasks,
    ordinary_engine_comparisons,
    run_population,
    validate_artifacts,
    write_artifact_manifest,
    write_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("run", "replay", "ordinary", "analyze", "validate", "provenance", "manifest"),
    )
    parser.add_argument("--cache", type=Path, default=Path("/cache/e01_s13"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S13"))
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
        results = run_population(tasks, args.cache / "reference_paper_scale.jsonl", workers=args.workers)
        print(json.dumps({"command": "run", "rows": len(results)}))
        return 0
    if args.command == "replay":
        rows = exact_replay_selected(tasks, workers=args.workers)
        (args.cache / "selected_replays.json").write_text(
            json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        print(json.dumps({"command": "replay", "rows": len(rows)}))
        return 0
    if args.command == "ordinary":
        rows = ordinary_engine_comparisons(tasks)
        (args.cache / "ordinary_comparisons.json").write_text(
            json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        print(json.dumps({"command": "ordinary", "rows": len(rows)}))
        return 0

    def read_jsonl(path: Path):
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    results = read_jsonl(args.cache / "reference_paper_scale.jsonl")
    replays = json.loads((args.cache / "selected_replays.json").read_text())
    ordinary = json.loads((args.cache / "ordinary_comparisons.json").read_text())
    built = build_artifacts(results, replays, ordinary, args.output)
    (args.output / "preregistration.json").write_bytes(
        (ROOT / "analysis" / "s13_chimera_preregistration.json").read_bytes()
    )
    print(json.dumps({"command": "analyze", **built}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
