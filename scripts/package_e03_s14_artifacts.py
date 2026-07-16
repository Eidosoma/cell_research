#!/usr/bin/env python3
"""Create the compact final artifact manifest for E03 S14."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S14"))
    parser.add_argument("--report-inputs", type=Path, default=Path("/artifacts/report_inputs"))
    args = parser.parse_args()
    mutable_exclusions = {"artifact_manifest.json", "validation_results.json", "validation.log"}
    files = []
    for root, role in ((args.output, "research_step"), (args.report_inputs, "report_input")):
        for path in sorted(candidate for candidate in root.iterdir() if candidate.is_file()):
            if role == "research_step" and path.name in mutable_exclusions:
                continue
            files.append(
                {
                    "role": role,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    value = {
        "schemaVersion": "e03.s14.artifact_manifest.v1",
        "researchStepId": "S14",
        "status": "complete",
        "fileCount": len(files),
        "files": files,
        "mutableExclusions": sorted(mutable_exclusions),
        "exclusionReason": "The manifest excludes itself and validator outputs that are rewritten after packaging; validation_results.json records their current status.",
    }
    (args.output / "artifact_manifest.json").write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"researchStepId": "S14", "fileCount": len(files)}, indent=2))


if __name__ == "__main__":
    main()
