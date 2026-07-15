#!/usr/bin/env python3
"""Finalize S08 provenance and the collectible artifact manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path("/artifacts/research_steps/S08")
REPOSITORY = Path(__file__).resolve().parents[1]
SOURCES = (
    "analysis/s08_comparison_contract.json",
    "scripts/build_s08_comparison.py",
    "scripts/validate_s08_comparison.py",
    "scripts/package_s08_results.py",
    "src/detours/observed_comparison.py",
    "tests/test_observed_comparison.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    provenance_path = ROOT / "provenance_manifest.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["repositoryCommitAtHandoff"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    provenance["repositorySourcesAtHandoff"] = [
        {
            "path": str(REPOSITORY / relative),
            "bytes": (REPOSITORY / relative).stat().st_size,
            "sha256": sha256_file(REPOSITORY / relative),
        }
        for relative in SOURCES
    ]
    write_json(provenance_path, provenance)
    artifacts = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(ROOT.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    write_json(
        ROOT / "artifact_manifest.json",
        {
            "schemaVersion": "e03.s08.artifact_manifest.v1",
            "researchStepId": "S08",
            "artifactCount": len(artifacts),
            "artifacts": artifacts,
        },
    )
    print(json.dumps({"artifactCount": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
