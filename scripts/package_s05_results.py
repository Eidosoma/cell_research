#!/usr/bin/env python3
"""Write compact environment, provenance, command, and artifact manifests for S05."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


OUTPUT = Path("/artifacts/research_steps/S05")
REPOSITORY = Path(__file__).resolve().parents[1]
STARTING_COMMIT = "87f59d788fd9a9405c53d15e1a60152611b79b78"


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
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    environment = {
        "schemaVersion": "e03.s05.environment.v1",
        "researchStepId": "S05",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyarrow", "numba", "pytest"]
        },
        "cpuCount": os.cpu_count(),
        "graphWorkers": 8,
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

    commands = """S05 reproducible commands (working directory /workspace/cell-research)
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/build_s05_graphs.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/reconstruct_s05_graphs.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/validate_s05_graphs.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python -m pytest -q tests/test_detour_transition_graph.py tests/test_detour_state_space.py tests/test_distances.py tests/test_schedulers.py tests/test_faults.py tests/test_architectures.py

Non-authoritative environment probes retained for transparency:
pytest -q ... -> exit 127, pytest executable absent from shell PATH
python3 -m pytest -q ... -> exit 1, system Python has no pytest module

Clean-restart history:
Two earlier build invocations completed graph materialization but failed only while forming the aggregate stratum report (fault_count grouping/measure collision, then mixed int/string stratum labels). Each partial output was deleted and rebuilt from the frozen contract. The third invocation wrote the final corpus; the independent reconstruction and physical validator passed afterward.
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
        Path("/artifacts/research_steps/S04/research_step_full_results.md"),
        Path("/artifacts/research_steps/S04/canonical_state_encoder.md"),
        Path("/artifacts/research_steps/S04/state_family_inventory.parquet"),
        Path("/artifacts/research_steps/S04/state_inventory.parquet"),
        Path("/previous-artifacts/E01/research_steps/S03/state_diagrams.md"),
        Path("/previous-artifacts/E01/research_steps/S05/semantic_decisions.json"),
        Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    ]
    source_paths = [
        REPOSITORY / "analysis/s05_graph_contract.json",
        REPOSITORY / "src/detours/distances.py",
        REPOSITORY / "src/detours/state_space.py",
        REPOSITORY / "src/detours/transition_graph.py",
        REPOSITORY / "scripts/build_s05_graphs.py",
        REPOSITORY / "scripts/reconstruct_s05_graphs.py",
        REPOSITORY / "scripts/validate_s05_graphs.py",
        REPOSITORY / "scripts/package_s05_results.py",
        REPOSITORY / "tests/test_detour_transition_graph.py",
        REPOSITORY / "reference_simulator/model.py",
        REPOSITORY / "reference_simulator/policies.py",
        REPOSITORY / "reference_simulator/scheduler.py",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/transition_primitives.py",
    ]
    missing = [str(path) for path in input_paths + source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    provenance = {
        "schemaVersion": "e03.s05.provenance.v1",
        "researchStepId": "S05",
        "stepNumber": 5,
        "success": True,
        "status": "complete",
        "repository": {
            "path": str(REPOSITORY),
            "branch": git("branch", "--show-current"),
            "startingCommit": STARTING_COMMIT,
            "headBeforeS05Commit": git("rev-parse", "HEAD"),
        },
        "inputs": [record(path) for path in input_paths],
        "repositorySources": [record(path) for path in source_paths],
        "previousArtifactMount": "/previous-artifacts/E01",
        "datasetInputsUsed": [],
        "networkInputsUsed": [],
    }
    write_json(OUTPUT / "provenance_manifest.json", provenance)

    excluded = {OUTPUT / "artifact_manifest.json"}
    artifacts = [
        record(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path not in excluded
    ]
    manifest = {
        "schemaVersion": "e03.s05.artifact_manifest.v1",
        "researchStepId": "S05",
        "stepNumber": 5,
        "success": True,
        "status": "complete",
        "artifactsWritten": [item["path"] for item in artifacts]
        + [str(OUTPUT / "artifact_manifest.json")],
        "artifacts": artifacts,
        "validationResult": (
            "pass: all 18 independent graph gates; 7,984/7,984 deterministic "
            "reconstructions; 75,700 E01 state and 169,045 opportunity fixtures; "
            "91 relevant repository tests"
        ),
        "caveatsOrBlockers": [
            "finite nondeterministic scheduler-opportunity projection, not byte-exact replay",
            "reachability is relative to S04 state-ordinal-zero anchor",
            "self-loops and parallel endpoint edges are intentional charged opportunities",
            "family coverage inherits S04 tier, unique-value, homogeneous-direction, and fault limits",
            "no graph-budget boundary was triggered",
        ],
        "recommendedNextAction": (
            "Await separate authorization for S06; then define the metric- and "
            "cost-relative necessary-detour quantity without changing S05 graph semantics."
        ),
    }
    write_json(OUTPUT / "artifact_manifest.json", manifest)
    print(json.dumps({"artifacts": len(artifacts) + 1, "success": True}, indent=2))


if __name__ == "__main__":
    main()
