#!/usr/bin/env python3
"""Package compact environment, provenance, and artifact manifests for E03 S02."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import pyarrow

from reference_simulator.model import canonical_json_bytes


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = Path("/artifacts/research_steps/S02")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def main() -> None:
    immutability = json.loads((OUTPUT / "source_immutability.json").read_text())
    diagnostics = json.loads((OUTPUT / "replay_diagnostics.json").read_text())
    environment = {
        "schemaVersion": "e03.s02.environment.v1",
        "researchStepId": "S02",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workerCount": diagnostics["workerCount"],
        "threadEnvironment": {
            key: os.environ.get(key, "1 (run command)")
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "packages": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "pyarrow": pyarrow.__version__,
            "pytest": version("pytest"),
        },
    }
    write_json(OUTPUT / "environment.json", environment)
    code_paths = [
        REPOSITORY / "analysis/s02_replay_contract.json",
        REPOSITORY / "src/detours/distances.py",
        REPOSITORY / "src/detours/replay.py",
        REPOSITORY / "scripts/replay_e01_distances.py",
        REPOSITORY / "scripts/validate_s02_replays.py",
        REPOSITORY / "scripts/run_s02_upstream_regressions.py",
        REPOSITORY / "scripts/package_s02_evidence.py",
        REPOSITORY / "tests/test_detour_replay.py",
        REPOSITORY / "tests/test_distances.py",
    ]
    provenance = {
        "schemaVersion": "e03.s02.provenance.v1",
        "researchStepId": "S02",
        "repository": "Eidosoma/cell_research",
        "branch": git("branch", "--show-current"),
        "startingCommit": git("rev-parse", "HEAD"),
        "referenceSemantics": "E01-reference-v1",
        "primaryTraceContract": "/previous-artifacts/E01/release/baseline/selectedTraces.json",
        "scenarioBank": "/previous-artifacts/E01/scenarios/paired_scenario_bank.parquet",
        "sourceArtifactSha256": immutability["preRunSha256"],
        "sourceArtifactsUnchanged": immutability["success"],
        "repositoryCodeSha256": {str(path.relative_to(REPOSITORY)): sha256_file(path) for path in code_paths},
        "commands": [
            "PYTHONPATH=. /cache/e03-s01-venv/bin/python -m pytest -q tests/test_detour_replay.py tests/test_distances.py",
            "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/replay_e01_distances.py --workers 8",
            "PYTHONPATH=. /cache/e03-s01-venv/bin/python scripts/validate_s02_replays.py",
            "PYTHONPATH=. /cache/e03-s01-venv/bin/python scripts/run_s02_upstream_regressions.py",
            "PYTHONPATH=. /cache/e03-s01-venv/bin/python -m compileall -q src/detours scripts/replay_e01_distances.py scripts/validate_s02_replays.py tests/test_detour_replay.py",
            "git diff --check",
        ],
    }
    write_json(OUTPUT / "provenance.json", provenance)
    excluded = {"artifact_manifest.json"}
    artifacts = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name not in excluded:
            artifacts.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        OUTPUT / "artifact_manifest.json",
        {
            "schemaVersion": "e03.s02.artifact_manifest.v1",
            "researchStepId": "S02",
            "artifacts": artifacts,
        },
    )


if __name__ == "__main__":
    main()
