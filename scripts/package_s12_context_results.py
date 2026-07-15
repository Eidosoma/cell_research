#!/usr/bin/env python3
"""Package S12 provenance, environment, commands, tests, and artifact manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

from reference_simulator.model import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/artifacts/research_steps/S12")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, value) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        name: sum(int(float(suite.attrib.get(name, 0))) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def main() -> None:
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "numba", "scipy", "statsmodels", "scikit-learn", "matplotlib", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = {
        "researchStepId": "S12",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "declaredMaximumWorkers": 8,
        "scientificKernelWorkers": 1,
        "referenceParityWorkers": 8,
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "packages": packages,
        "newDependenciesInstalled": [],
    }
    write_json(OUTPUT / "environment.json", environment)

    immutability = json.loads((OUTPUT / "input_immutability.json").read_text())
    provenance = {
        "schemaVersion": "e03.s12.provenance.v1",
        "researchStepId": "S12",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E03",
        "repository": str(ROOT),
        "branch": git("branch", "--show-current"),
        "headAtPackaging": git("rev-parse", "HEAD"),
        "startingCommit": "c67d363c59eb51821738626e22581dd4753e326a",
        "previousArtifactMount": "/previous-artifacts/E01",
        "datasetUse": "No mounted dataset required; DATASET_AVAILABILITY is not_required.",
        "simulatorSemantics": "E01-reference-v1 transition semantics with the declared state-blind external counter-addressed scheduler projection",
        "inputs": immutability["inputs"],
        "sourceCode": [
            "analysis/s12_context_sensitivity_contract.json",
            "src/detours/context_sensitivity.py",
            "scripts/build_s12_context_sensitivity.py",
            "scripts/fit_s12_context_models.py",
            "scripts/package_s12_identifiability_stop.py",
            "scripts/summarize_s12_context_results.py",
            "scripts/build_s12_context_figures.py",
            "scripts/validate_s12_context_sensitivity.py",
            "scripts/run_s12_mounted_reference_tests.py",
            "scripts/package_s12_context_results.py",
            "tests/test_context_sensitivity.py"
        ],
        "inputImmutabilityPassed": immutability["success"],
    }
    write_json(OUTPUT / "provenance_manifest.json", provenance)

    commands = """S12 final execution commands (UTC 2026-07-15)
Environment prefix for scientific Python commands:
OPENSSL_FORCE_FIPS_MODE=0 OPENSSL_FIPS_MODE_SWITCH_PATH=/dev/null NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.

/cache/e03-s01-venv/bin/python -m pytest -q tests/test_context_sensitivity.py
/cache/e03-s01-venv/bin/python scripts/build_s12_context_sensitivity.py
/cache/e03-s01-venv/bin/python scripts/fit_s12_context_models.py
# The preceding command stopped as preregistered after writing context_models/identifiability_failure.json.
/cache/e03-s01-venv/bin/python scripts/package_s12_identifiability_stop.py
/cache/e03-s01-venv/bin/python scripts/summarize_s12_context_results.py
/cache/e03-s01-venv/bin/python scripts/build_s12_context_figures.py
/cache/e03-s01-venv/bin/python scripts/validate_s12_context_sensitivity.py
/cache/e03-s01-venv/bin/python -m pytest -q --junitxml=/artifacts/research_steps/S12/repository_tests.junit.xml [focused S01-S12 regression modules]
/cache/e03-s01-venv/bin/python scripts/run_s12_mounted_reference_tests.py
/cache/e03-s01-venv/bin/python scripts/package_s12_context_results.py

Recovery note: one pre-outcome kernel launch was interrupted to add explicit open-episode time at risk. A later completed corpus was regenerated from scratch after adding builder-source immutability and terminal trace markers. No interrupted output is retained as evidence.
"""
    (OUTPUT / "commands.log").write_text(commands)

    tests = {}
    for name in ("repository_tests.junit.xml", "mounted_reference_tests.junit.xml"):
        path = OUTPUT / name
        if path.exists():
            tests[name] = junit_counts(path)
    test_success = bool(tests) and all(value["failures"] == 0 and value["errors"] == 0 for value in tests.values())
    write_json(
        OUTPUT / "test_summary.json",
        {
            "researchStepId": "S12",
            "success": test_success,
            "junit": tests,
        },
    )

    paths = sorted(
        path for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    write_json(
        OUTPUT / "artifact_manifest.json",
        {
            "schemaVersion": "e03.s12.artifact_manifest.v1",
            "researchStepId": "S12",
            "artifactCount": len(paths),
            "artifacts": [file_record(path) for path in paths],
        },
    )
    print(json.dumps({"artifacts": len(paths), "tests": tests, "testSuccess": test_success}, indent=2))


if __name__ == "__main__":
    main()
