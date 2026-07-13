#!/usr/bin/env python3
"""Command-line orchestration for E01 research step S09."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.no_fault_sorting import (  # noqa: E402
    PREREGISTRATION,
    build_artifacts,
    load_tasks,
    run_exact_replay_samples,
    run_historical_population,
    run_historical_variability,
    run_reference_population,
    sha256_file,
    validate_artifacts,
    write_artifact_manifest,
    write_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=(
            "reference", "historical", "replay", "variability", "analyze",
            "validate", "provenance", "manifest",
        )
    )
    parser.add_argument("--split", choices=("paper_scale", "confirmatory_holdout"))
    parser.add_argument("--cache", type=Path, default=Path("/cache/e01_s09"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S09"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preregistration-sha256")
    args = parser.parse_args()
    prereg_hash = sha256_file(PREREGISTRATION)

    def tasks(split: str):
        return load_tasks(
            split,
            preregistration_sha256=(prereg_hash if split == "confirmatory_holdout" else None),
        )

    args.cache.mkdir(parents=True, exist_ok=True)
    if args.command == "validate":
        result = validate_artifacts(args.output)
        print(json.dumps({"command": "validate", "passed": result["passed"], "checks": result["checkCount"]}))
        return 0
    if args.command == "provenance":
        result = write_provenance(args.output)
        print(json.dumps({"command": "provenance", **result}))
        return 0
    if args.command == "manifest":
        result = write_artifact_manifest(args.output)
        print(json.dumps({"command": "manifest", "artifacts": result["artifactCount"]}))
        return 0
    if args.command in {"reference", "historical"}:
        if args.split is None:
            parser.error("--split is required")
        selected = load_tasks(
            args.split,
            preregistration_sha256=args.preregistration_sha256,
        )
        if args.command == "reference":
            rows = run_reference_population(
                selected, args.cache / f"reference_{args.split}.jsonl", workers=args.workers
            )
        else:
            rows = run_historical_population(
                selected,
                args.cache / f"historical_{args.split}.jsonl",
                workers=args.workers,
                timeout_seconds=300.0,
            )
        print(json.dumps({"command": args.command, "split": args.split, "rows": len(rows)}))
        return 0

    paper = tasks("paper_scale")
    confirm = tasks("confirmatory_holdout")
    if args.command == "replay":
        rows = run_exact_replay_samples(paper, confirm, workers=args.workers)
        (args.cache / "exact_replay_samples.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"command": "replay", "rows": len(rows)}))
        return 0
    if args.command == "variability":
        rows = run_historical_variability(paper)
        (args.cache / "historical_scheduler_variability.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"command": "variability", "rows": len(rows)}))
        return 0

    def read_jsonl(path: Path):
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    reference = read_jsonl(args.cache / "reference_paper_scale.jsonl") + read_jsonl(
        args.cache / "reference_confirmatory_holdout.jsonl"
    )
    historical = read_jsonl(args.cache / "historical_paper_scale.jsonl") + read_jsonl(
        args.cache / "historical_confirmatory_holdout.jsonl"
    )
    replay = json.loads((args.cache / "exact_replay_samples.json").read_text())
    variability = json.loads((args.cache / "historical_scheduler_variability.json").read_text())
    summary = build_artifacts(reference, historical, replay, variability, args.output)
    (args.output / "confirmatory_preregistration.json").write_bytes(PREREGISTRATION.read_bytes())
    print(json.dumps({"command": "analyze", "finalValidation": summary["finalValidation"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
