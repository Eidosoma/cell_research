#!/usr/bin/env python3
"""Finalize S09 provenance, environment, result, and artifact manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path("/artifacts/research_steps/S09")
REPOSITORY = Path(__file__).resolve().parents[1]
SOURCES = (
    "analysis/s09_barrier_intervention_contract.json",
    "reference_simulator/engine.py",
    "reference_simulator/scheduler.py",
    "src/detours/barrier_interventions.py",
    "scripts/build_s09_interventions.py",
    "scripts/run_s09_budget_sensitivity.py",
    "scripts/refine_s09_effect_tables.py",
    "scripts/validate_s09_interventions.py",
    "scripts/package_s09_results.py",
    "tests/test_barrier_interventions.py",
    "tests/test_schedulers.py",
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
    environment = json.loads((ROOT / "environment.json").read_text())
    environment.update(
        {
            "sensitivityWorkers": 8,
            "validationWorkers": 8,
            "clusterBootstrapResamples": 10_000,
            "primaryHorizon": 2_048,
            "sensitivityHorizon": 32_768,
            "relevantRepositoryTestsPassed": 141,
            "repositoryWideDiagnostic": {
                "passed": 369,
                "skipped": 2,
                "failed": 13,
                "errors": 15,
                "boundary": "artifact-dependent tests for other experiment layouts with unavailable or namespace-colliding /artifacts inputs",
            },
        }
    )
    write_json(ROOT / "environment.json", environment)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPOSITORY, text=True
    ).strip()
    immutability = json.loads((ROOT / "input_immutability.json").read_text())
    provenance = {
        "schemaVersion": "e03.s09.provenance_manifest.v1",
        "researchStepId": "S09",
        "repository": str(REPOSITORY),
        "branch": branch,
        "executionBaseCommit": "016b326f1cec49000e52b4e45cbc0651b71a8106",
        "repositoryCommitAtHandoff": commit,
        "previousArtifactMount": "/previous-artifacts/E01",
        "sourceFiles": [
            {
                "path": str(REPOSITORY / relative),
                "bytes": (REPOSITORY / relative).stat().st_size,
                "sha256": sha256_file(REPOSITORY / relative),
            }
            for relative in SOURCES
        ],
        "executionInputs": immutability["files"],
        "upstreamArtifacts": [
            {
                "path": path,
                "sha256": sha256_file(Path(path)),
            }
            for path in (
                "/artifacts/research_steps/S04/state_family_inventory.parquet",
                "/artifacts/research_steps/S05/graph_corpus_manifest.json",
                "/artifacts/research_steps/S06/necessary_detour_spec.md",
                "/artifacts/research_steps/S07/path_solutions.parquet",
                "/artifacts/research_steps/S08/behavior_necessity_comparison.parquet",
            )
        ],
        "runtimeCommands": str(ROOT / "commands.log"),
        "protocolContract": str(REPOSITORY / "analysis/s09_barrier_intervention_contract.json"),
        "randomProfile": "sha256_counter_E01_v1_common_pair_key_conditioned_focal_event0",
        "schedulerClaimBoundary": "paired deterministic cell-view common streams; deterministic traditional controller; no frequency claim",
    }
    write_json(ROOT / "provenance_manifest.json", provenance)

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
            "schemaVersion": "e03.s09.artifact_manifest.v1",
            "researchStepId": "S09",
            "artifactCount": len(artifacts),
            "artifacts": artifacts,
        },
    )
    print(json.dumps({"artifactCount": len(artifacts), "commit": commit}, indent=2))


if __name__ == "__main__":
    main()
