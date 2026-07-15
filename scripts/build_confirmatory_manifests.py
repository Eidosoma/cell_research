#!/usr/bin/env python3
"""Write compact S11 artifact, input, source, and environment provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/artifacts/research_steps/S11")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    label = str(path.relative_to(root)) if root is not None else str(path)
    return {"path": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def existing_inputs() -> list[Path]:
    paths = [
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
        Path("/artifacts/research_steps/S01/estimand_registry.yaml"),
        Path("/artifacts/research_steps/S01/research_step_full_results.md"),
        Path("/artifacts/research_steps/S02/research_step_full_results.md"),
        Path("/artifacts/research_steps/S03/research_step_full_results.md"),
        Path("/artifacts/research_steps/S04/research_step_full_results.md"),
        Path("/artifacts/research_steps/S05/research_step_full_results.md"),
        Path("/artifacts/research_steps/S06/structural_lock.json"),
        Path("/artifacts/research_steps/S06/research_step_full_results.md"),
        Path("/artifacts/research_steps/S07/scenario_extension.parquet"),
        Path("/artifacts/research_steps/S07/research_step_full_results.md"),
        Path("/artifacts/research_steps/S08/pairing_manifest.parquet"),
        Path("/artifacts/research_steps/S08/contrast_arm_catalog.json"),
        Path("/artifacts/research_steps/S08/coupling_matrix.csv"),
        Path("/artifacts/research_steps/S08/research_step_full_results.md"),
        Path("/artifacts/research_steps/S09/cost_schema.json"),
        Path("/artifacts/research_steps/S09/research_step_full_results.md"),
        Path("/artifacts/research_steps/S10/s11_frozen_selections.json"),
        Path("/artifacts/research_steps/S10/research_step_full_results.md"),
        Path("/previous-artifacts/E01/specification/transition_spec.md"),
        Path("/previous-artifacts/E01/report_inputs/report_bundle_manifest.json"),
        REPOSITORY / "design/s09/cost_schema.json",
        REPOSITORY / "design/s11/confirmatory_prespecification.json",
    ]
    metadata = sorted(Path("/workspace/input-attachments").glob("**/_metadata/ATTACHMENT.md"))
    reports = [
        Path(f"/artifacts/research_steps/S{step:02d}/validation_summary.json")
        for step in range(1, 11)
    ]
    return [path for path in paths + metadata + reports if path.is_file()]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def build(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((args.output / "confirmatory_freeze_manifest.json").read_text())
    sequential = json.loads((args.output / "sequential_sampling_log.json").read_text())

    input_records = [file_record(path) for path in existing_inputs()]
    canonical_write(args.output / "input_provenance.json", {
        "schemaVersion": "e02.s11.input_provenance.v1",
        "researchStepId": "S11",
        "inputs": input_records,
        "inputCount": len(input_records),
        "frozenSelectionContentSha256": freeze["hashes"]["s10SelectionContentSha256"],
        "designFreezeSha256": freeze["designFreezeSha256"],
        "protectedOutcomeReadsBeforeFreeze": freeze["protectedOutcomeReadsBeforeFreeze"],
        "terminalLook": sequential["terminalLook"],
    })

    source_paths = [
        REPOSITORY / "design/s11/confirmatory_prespecification.json",
        REPOSITORY / "scripts/freeze_confirmatory_design.py",
        REPOSITORY / "scripts/run_confirmatory_stage.py",
        REPOSITORY / "scripts/analyze_confirmatory_look.py",
        REPOSITORY / "scripts/finalize_confirmatory.py",
        REPOSITORY / "scripts/validate_confirmatory_replay.py",
        REPOSITORY / "scripts/validate_confirmatory_results.py",
        REPOSITORY / "scripts/build_confirmatory_manifests.py",
        REPOSITORY / "tests/test_confirmatory_design.py",
    ]
    canonical_write(args.output / "source_package_manifest.json", {
        "schemaVersion": "e02.s11.source_package_manifest.v1",
        "researchStepId": "S11",
        "repository": "Eidosoma/cell_research",
        "branch": git("branch", "--show-current"),
        "gitHead": git("rev-parse", "HEAD"),
        "gitStatusShort": git("status", "--short"),
        "files": [file_record(path, root=REPOSITORY) for path in source_paths],
    })

    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "scipy", "pytest"):
        packages[name] = importlib.metadata.version(name)
    canonical_write(args.output / "environment_provenance.json", {
        "schemaVersion": "e02.s11.environment_provenance.v1",
        "researchStepId": "S11",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logicalCpuCount": os.cpu_count(),
        "scientificRunWorkers": 8,
        "threadEnvironmentDuringExecution": {
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        },
        "packages": packages,
        "dependenciesInstalledForS11": [],
    })

    excluded = {"artifact_manifest.json"}
    artifact_paths = sorted(
        path for path in args.output.rglob("*")
        if path.is_file() and path.relative_to(args.output).as_posix() not in excluded
    )
    canonical_write(args.output / "artifact_manifest.json", {
        "schemaVersion": "e02.s11.artifact_manifest.v1",
        "researchStepId": "S11",
        "root": str(args.output),
        "files": [file_record(path, root=args.output) for path in artifact_paths],
        "fileCountExcludingThisManifest": len(artifact_paths),
        "cacheNotCollectible": "/cache/s11",
    })
    print(json.dumps({
        "inputs": len(input_records), "artifacts": len(artifact_paths),
        "gitHead": git("rev-parse", "HEAD"),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
