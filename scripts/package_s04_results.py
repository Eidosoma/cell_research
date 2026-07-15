#!/usr/bin/env python3
"""Write compact environment, provenance, command, and artifact manifests for S04."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


OUTPUT = Path("/artifacts/research_steps/S04")
REPOSITORY = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True
    ).strip()


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    environment = {
        "schemaVersion": "e03.s04.environment.v1",
        "researchStepId": "S04",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyarrow", "pytest"]
        },
        "cpuCount": os.cpu_count(),
        "enumerationWorkers": 8,
        "threadEnvironment": {
            name: "1"
            for name in [
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
        "gpuUsed": False,
        "networkUsed": False,
        "dependenciesInstalled": [],
    }
    write_json(OUTPUT / "environment.json", environment)

    commands = """S04 reproducible commands (working directory /workspace/cell-research)
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/enumerate_s04_states.py --preflight-only
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/enumerate_s04_states.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/validate_s04_enumeration.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python -m pytest -q tests/test_detour_state_space.py tests/test_distances.py tests/test_schedulers.py tests/test_faults.py tests/test_architectures.py
E01 reference tests: tests/test_reference_simulator.py, with module.S03_FIXTURES redirected read-only to /previous-artifacts/E01/research_steps/S03/toy_fixtures.json
"""
    (OUTPUT / "commands.log").write_text(commands, encoding="utf-8")

    input_paths = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.json"),
        Path("/workspace/DATASETS.md"),
        Path("/workspace/DATASET_CATALOG.json"),
        Path("/workspace/DATASET_AVAILABILITY.json"),
        Path("/workspace/CAPABILITIES.md"),
        Path("/workspace/CAPABILITY_AVAILABILITY.json"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"),
        Path("/artifacts/research_steps/S01/research_step_full_results.md"),
        Path("/artifacts/research_steps/S02/research_step_full_results.md"),
        Path("/artifacts/research_steps/S03/research_step_full_results.md"),
        Path("/previous-artifacts/E01/research_steps/S03/state_diagrams.md"),
        Path("/previous-artifacts/E01/research_steps/S05/semantic_decisions.json"),
        Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    ]
    source_paths = [
        REPOSITORY / "analysis/s04_enumeration_contract.json",
        REPOSITORY / "src/detours/state_space.py",
        REPOSITORY / "scripts/enumerate_s04_states.py",
        REPOSITORY / "scripts/validate_s04_enumeration.py",
        REPOSITORY / "scripts/package_s04_results.py",
        REPOSITORY / "tests/test_detour_state_space.py",
        REPOSITORY / "reference_simulator/model.py",
        REPOSITORY / "reference_simulator/policies.py",
        REPOSITORY / "reference_simulator/scheduler.py",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/transition_primitives.py",
    ]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(missing_inputs)
    provenance = {
        "schemaVersion": "e03.s04.provenance.v1",
        "researchStepId": "S04",
        "stepNumber": 4,
        "success": True,
        "status": "complete",
        "repository": {
            "path": str(REPOSITORY),
            "branch": git("branch", "--show-current"),
            "startingCommit": "ed583cc98d335b4156f1019c638bc29700a5975d",
            "headBeforeS04Commit": git("rev-parse", "HEAD"),
        },
        "inputs": [record(path) for path in input_paths],
        "repositorySources": [record(path) for path in source_paths],
        "previousArtifactMount": "/previous-artifacts/E01",
        "datasetInputsUsed": [],
        "networkInputsUsed": [],
    }
    write_json(OUTPUT / "provenance_manifest.json", provenance)

    excluded = {"artifact_manifest.json"}
    artifacts = [
        record(path)
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name not in excluded
    ]
    manifest = {
        "schemaVersion": "e03.s04.artifact_manifest.v1",
        "researchStepId": "S04",
        "stepNumber": 4,
        "success": True,
        "status": "complete",
        "artifactsWritten": [item["path"] for item in artifacts]
        + [str(OUTPUT / "artifact_manifest.json")],
        "artifacts": artifacts,
        "validationResult": "pass: all 13 independent artifact gates",
        "caveatsOrBlockers": [
            "tier-specific achieved n",
            "Selection cursor state explosion",
            "finite legal-opportunity scheduler projection",
            "syntactically valid rather than reachability-filtered inventory",
        ],
        "recommendedNextAction": "Await separate authorization for S05; do not start automatically.",
    }
    write_json(OUTPUT / "artifact_manifest.json", manifest)
    print(json.dumps({"artifacts": len(artifacts) + 1, "success": True}, indent=2))


if __name__ == "__main__":
    main()
