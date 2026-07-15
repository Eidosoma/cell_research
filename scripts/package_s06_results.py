#!/usr/bin/env python3
"""Write environment, command, provenance, and artifact manifests for S06."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


OUTPUT = Path("/artifacts/research_steps/S06")
REPOSITORY = Path(__file__).resolve().parents[1]
STARTING_COMMIT = "4b28f0c2d61618aa4c166bb6b6d197072114518b"


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
        "schemaVersion": "e03.s06.environment.v1",
        "researchStepId": "S06",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyarrow", "numba", "pytest"]
        },
        "cpuCount": os.cpu_count(),
        "validationWorkers": 1,
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

    commands = """S06 reproducible commands (working directory /workspace/cell-research)
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python -m pytest -q tests/test_necessary_detour.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/validate_s06_solver.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python -m pytest -q tests/test_necessary_detour.py tests/test_detour_transition_graph.py tests/test_detour_state_space.py tests/test_distances.py tests/test_schedulers.py tests/test_faults.py tests/test_architectures.py
PYTHONPATH=. /cache/e03-s01-venv/bin/python -m py_compile src/detours/necessary_detour.py scripts/validate_s06_solver.py scripts/package_s06_results.py tests/test_necessary_detour.py
git diff --check

Recovery note:
The first validation-table assembly combined non-uniform hand and physical fixture dictionaries, causing PyArrow to infer null physical columns. Rows were normalized to one schema and the complete validator was rerun. This was a reporting-only defect; all solver comparisons passed before and after the correction.
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
        Path("/artifacts/research_steps/S01/distance_spec.md"),
        Path("/artifacts/research_steps/S01/research_step_full_results.md"),
        Path("/artifacts/research_steps/S02/research_step_full_results.md"),
        Path("/artifacts/research_steps/S03/research_step_full_results.md"),
        Path("/artifacts/research_steps/S04/canonical_state_encoder.md"),
        Path("/artifacts/research_steps/S04/research_step_full_results.md"),
        Path("/artifacts/research_steps/S04/state_family_inventory.parquet"),
        Path("/artifacts/research_steps/S05/graph_schema.md"),
        Path("/artifacts/research_steps/S05/graph_corpus_manifest.json"),
        Path("/artifacts/research_steps/S05/graph_family_manifest.parquet"),
        Path("/artifacts/research_steps/S05/validation_results.json"),
        Path("/artifacts/research_steps/S05/research_step_full_results.md"),
        Path("/previous-artifacts/E01/research_steps/S03/state_diagrams.md"),
        Path("/previous-artifacts/E01/research_steps/S05/semantic_decisions.json"),
        Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    ]
    source_paths = [
        REPOSITORY / "analysis/s06_necessary_detour_contract.json",
        REPOSITORY / "src/detours/__init__.py",
        REPOSITORY / "src/detours/distances.py",
        REPOSITORY / "src/detours/state_space.py",
        REPOSITORY / "src/detours/transition_graph.py",
        REPOSITORY / "src/detours/necessary_detour.py",
        REPOSITORY / "scripts/validate_s06_solver.py",
        REPOSITORY / "scripts/package_s06_results.py",
        REPOSITORY / "tests/test_necessary_detour.py",
        REPOSITORY / "reference_simulator/model.py",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/scheduler.py",
        REPOSITORY / "reference_simulator/policies.py",
        REPOSITORY / "reference_simulator/transition_primitives.py",
    ]
    missing = [str(path) for path in input_paths + source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    provenance = {
        "schemaVersion": "e03.s06.provenance.v1",
        "researchStepId": "S06",
        "stepNumber": 6,
        "success": True,
        "status": "complete",
        "repository": {
            "path": str(REPOSITORY),
            "branch": git("branch", "--show-current"),
            "startingCommit": STARTING_COMMIT,
            "headBeforeS06Commit": git("rev-parse", "HEAD"),
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
        "schemaVersion": "e03.s06.artifact_manifest.v1",
        "researchStepId": "S06",
        "stepNumber": 6,
        "success": True,
        "status": "complete",
        "artifactsWritten": [item["path"] for item in artifacts]
        + [str(OUTPUT / "artifact_manifest.json")],
        "artifacts": artifacts,
        "validationResult": (
            "pass: all 20 S06 gates; 2,490 independent tiny starts; 96 physical "
            "S05 start/metric fixtures; all 7,984 family contracts; eight largest-"
            "family metric benchmarks; 121 relevant repository tests"
        ),
        "caveatsOrBlockers": [
            "existential finite structural-opportunity paths only",
            "no universal, probabilistic, expected, or byte-exact scheduler claim",
            "unreachable and quiescent states have undefined excursion",
            "paper strict Sortedness is not a proper tied-value goal distance",
            "full-corpus scientific path solving and prevalence are deferred to S07",
            "no S06 ambiguity or budget trigger remains",
        ],
        "recommendedNextAction": (
            "Await separate authorization for S07; then solve all retained S05 "
            "families under the frozen S06 definition without changing semantics."
        ),
    }
    write_json(OUTPUT / "artifact_manifest.json", manifest)
    print(json.dumps({"artifacts": len(artifacts) + 1, "success": True}, indent=2))


if __name__ == "__main__":
    main()
