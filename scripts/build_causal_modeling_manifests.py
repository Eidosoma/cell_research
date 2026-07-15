#!/usr/bin/env python3
"""Package compact S12 model outputs, provenance, commands, and manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.causal_modeling import canonical_json_bytes, sha256_file  # noqa: E402


DEFAULT_OUTPUT = Path("/artifacts/research_steps/S12")
MODEL_FILES = (
    "model_coefficients.parquet",
    "model_diagnostics.parquet",
    "model_standardized_effects.parquet",
    "calibration_diagnostics.parquet",
    "survival_diagnostics.parquet",
    "piecewise_completion_rates.parquet",
    "model_fit_summary.json",
)
SOURCE_FILES = (
    "causal_simulator/causal_modeling.py",
    "design/s12/modeling_prespecification.json",
    "scripts/freeze_causal_modeling.py",
    "scripts/run_causal_effects.py",
    "scripts/fit_causal_models.py",
    "scripts/plot_causal_effects.py",
    "scripts/validate_causal_modeling.py",
    "scripts/validate_causal_regeneration.py",
    "scripts/build_causal_modeling_manifests.py",
    "scripts/summarize_s12_tests.py",
    "tests/test_causal_modeling.py",
)


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True
    ).strip()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def input_paths() -> list[Path]:
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
        Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md"),
        Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
        Path("/previous-artifacts/E01/release/baseline/release_manifest.json"),
        Path("/previous-artifacts/E01/report_inputs/provenance_manifest.json"),
        Path("/artifacts/research_steps/S01/causal_dag.json"),
        Path("/artifacts/research_steps/S01/estimand_registry.yaml"),
        Path("/artifacts/research_steps/S01/analysis_population_specification.md"),
        Path("/artifacts/research_steps/S02/action_interface_spec.md"),
        Path("/artifacts/research_steps/S02/validation_summary.json"),
        Path("/artifacts/research_steps/S03/architecture_specification.json"),
        Path("/artifacts/research_steps/S03/validation_summary.json"),
        Path("/artifacts/research_steps/S04/scheduler_specification.json"),
        Path("/artifacts/research_steps/S04/validation_summary.json"),
        Path("/artifacts/research_steps/S05/fault_semantics_specification.json"),
        Path("/artifacts/research_steps/S05/validation_summary.json"),
        Path("/artifacts/research_steps/S06/placement_prespecification.json"),
        Path("/artifacts/research_steps/S06/validation_summary.json"),
        Path("/artifacts/research_steps/S07/scenario_suite_manifest.json"),
        Path("/artifacts/research_steps/S07/validation_summary.json"),
        Path("/artifacts/research_steps/S08/pairing_prespecification.json"),
        Path("/artifacts/research_steps/S08/semantic_random_stream_specification.json"),
        Path("/artifacts/research_steps/S08/contrast_arm_catalog.json"),
        Path("/artifacts/research_steps/S08/validation_summary.json"),
        Path("/artifacts/research_steps/S09/cost_schema.json"),
        Path("/artifacts/research_steps/S09/validation_summary.json"),
        Path("/artifacts/research_steps/S10/s11_frozen_selections.json"),
        Path("/artifacts/research_steps/S10/screening_model_specification.json"),
        Path("/artifacts/research_steps/S10/validation_summary.json"),
        Path("/artifacts/research_steps/S11/confirmatory_prespecification.json"),
        Path("/artifacts/research_steps/S11/confirmatory_freeze_manifest.json"),
        Path("/artifacts/research_steps/S11/confirmatory_results.parquet"),
        Path("/artifacts/research_steps/S11/paired_effects.parquet"),
        Path("/artifacts/research_steps/S11/cost_ledger.parquet"),
        Path("/artifacts/research_steps/S11/validation_summary.json"),
    ]
    paths.extend(Path("/workspace/input-attachments").glob("*/_metadata/ATTACHMENT.md"))
    return sorted({path for path in paths if path.exists()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    model_output = output / "model_results"
    model_output.mkdir(exist_ok=True)
    copied = []
    for name in MODEL_FILES:
        source = output / name
        if source.exists():
            destination = model_output / name
            shutil.copy2(source, destination)
            copied.append(destination)
    write_json(model_output / "manifest.json", {
        "schemaVersion": "e02.s12.model_result_manifest.v1",
        "researchStepId": "S12",
        "files": [record(path) for path in copied],
    })

    inputs = input_paths()
    write_json(output / "input_provenance.json", {
        "schemaVersion": "e02.s12.input_provenance.v1",
        "researchStepId": "S12",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "inputs": [record(path) for path in inputs],
        "datasetInputs": [],
        "datasetContext": "No mounted dataset was required; S12 analyzes the frozen S11 simulator results.",
        "previousArtifactContext": "E01 release/previous-artifact context was refreshed as design provenance; no prior artifact was mutated.",
    })
    packages = {}
    for package in ("numpy", "pandas", "scipy", "statsmodels", "patsy", "pyarrow", "matplotlib"):
        packages[package] = importlib.metadata.version(package)
    write_json(output / "environment_provenance.json", {
        "schemaVersion": "e02.s12.environment_provenance.v1",
        "researchStepId": "S12",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "workerPolicy": "Design-based bootstrap used one process with OPENBLAS_NUM_THREADS=8; model fitting used one process and one BLAS thread; no nested parallelism.",
        "gpuUsed": False,
    })
    write_json(output / "source_package_manifest.json", {
        "schemaVersion": "e02.s12.source_package_manifest.v1",
        "researchStepId": "S12",
        "repository": str(REPOSITORY),
        "branch": git("branch", "--show-current"),
        "headAtPackaging": git("rev-parse", "HEAD"),
        "remote": git("remote", "get-url", "origin"),
        "sourceFiles": [record(REPOSITORY / relative) for relative in SOURCE_FILES],
    })
    commands = """S12 execution commands (UTC 2026-07-15)

PYTHONPATH=. python scripts/freeze_causal_modeling.py --output /artifacts/research_steps/S12
PYTHONPATH=. pytest -q tests/test_causal_modeling.py tests/test_confirmatory_design.py
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_causal_effects.py --output /artifacts/research_steps/S12 --bootstrap-replicates 2000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python scripts/fit_causal_models.py --output /artifacts/research_steps/S12
python scripts/plot_causal_effects.py --output /artifacts/research_steps/S12
PYTHONPATH=. python scripts/validate_causal_regeneration.py --reference /artifacts/research_steps/S12 --regenerated /cache/s12_regenerated
PYTHONPATH=. python scripts/validate_causal_modeling.py --output /artifacts/research_steps/S12
PYTHONPATH=. pytest -q --junitxml=/artifacts/research_steps/S12/focused_tests.junit.xml tests/test_causal_modeling.py tests/test_confirmatory_design.py tests/test_costing.py
PYTHONPATH=. pytest -q --junitxml=/artifacts/research_steps/S12/full_repository_tests.junit.xml
python scripts/build_causal_modeling_manifests.py --output /artifacts/research_steps/S12
"""
    (output / "execution_commands.log").write_text(commands)
    write_json(output / "repository_release.json", {
        "schemaVersion": "e02.s12.repository_release.v1",
        "researchStepId": "S12",
        "branch": git("branch", "--show-current"),
        "commitAtPackaging": git("rev-parse", "HEAD"),
        "remote": git("remote", "get-url", "origin"),
    })
    artifacts = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    write_json(output / "artifact_manifest.json", {
        "schemaVersion": "e02.s12.artifact_manifest.v1",
        "researchStepId": "S12",
        "artifactCount": len(artifacts),
        "artifacts": [record(path) for path in artifacts],
    })
    print(json.dumps({
        "modelsPackaged": len(copied), "inputs": len(inputs),
        "artifacts": len(artifacts),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
