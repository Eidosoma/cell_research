#!/usr/bin/env python3
"""Write compact E03 S03 environment, provenance, and artifact manifests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess


REPOSITORY_FILES = [
    "analysis/s03_atlas_contract.json",
    "scripts/build_s03_atlas.py",
    "scripts/validate_s03_atlas.py",
    "scripts/package_s03_evidence.py",
    "src/detours/atlas.py",
    "src/detours/distances.py",
    "tests/test_detour_atlas.py",
    "tests/test_detour_replay.py",
    "tests/test_distances.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/artifacts/research_steps/S03"),
    )
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    repository = Path(__file__).resolve().parents[1]
    immutability = json.loads((root / "input_immutability.json").read_text())
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "matplotlib", "pytest"):
        packages[name] = importlib.metadata.version(name)
    environment = {
        "schemaVersion": "e03.s03.environment.v1",
        "researchStepId": "S03",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workerCount": 1,
        "bootstrapDrawsPerEstimate": 10_000,
        "packages": packages,
        "threadEnvironment": {
            name: os.environ.get(name, "1")
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    write_json(root / "environment.json", environment)
    code_hashes = {
        name: sha256_file(repository / name)
        for name in REPOSITORY_FILES
        if (repository / name).is_file()
    }
    provenance = {
        "schemaVersion": "e03.s03.provenance.v1",
        "researchStepId": "S03",
        "repository": "Eidosoma/cell_research",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "repositoryCommit": git("rev-parse", "HEAD"),
        "startingCommit": "576ee2edbf85c0f2dbf4944d00ff81ae0be0fabe",
        "contract": str(repository / "analysis/s03_atlas_contract.json"),
        "primaryInput": "/artifacts/research_steps/S02/replayed_distances.parquet",
        "scenarioBank": "/previous-artifacts/E01/scenarios/paired_scenario_bank.parquet",
        "sourceArtifactSha256": immutability["preRunSha256"],
        "sourceArtifactsUnchanged": immutability["success"],
        "repositoryCodeSha256": code_hashes,
        "commands": [
            "PYTHONPATH=. /cache/e03-s01-venv/bin/python -m pytest -q tests/test_detour_atlas.py tests/test_detour_replay.py tests/test_distances.py",
            "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/build_s03_atlas.py",
            "PYTHONPATH=. /cache/e03-s01-venv/bin/python scripts/validate_s03_atlas.py",
            "PYTHONPATH=. /cache/e03-s01-venv/bin/python -m compileall -q src/detours scripts/build_s03_atlas.py scripts/validate_s03_atlas.py scripts/package_s03_evidence.py tests/test_detour_atlas.py",
            "git diff --check",
        ],
        "analysis": {
            "bootstrapUnit": "logical trace cluster",
            "bootstrapDrawsPerEstimate": 10_000,
            "bootstrapSeed": 3_761_897_473,
            "episodeRules": [
                "plateau_bridge_initial_included",
                "plateau_break_initial_included",
                "plateau_bridge_post_swap_only",
                "plateau_break_post_swap_only",
            ],
        },
    }
    write_json(root / "provenance.json", provenance)
    artifacts = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    write_json(
        root / "artifact_manifest.json",
        {
            "schemaVersion": "e03.s03.artifact_manifest.v1",
            "researchStepId": "S03",
            "artifacts": artifacts,
        },
    )
    print(f"packaged {len(artifacts)} S03 artifacts")


if __name__ == "__main__":
    main()
