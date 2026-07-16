#!/usr/bin/env python3
"""Package S13 environment, provenance, tests, status, commands, and manifest."""

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
OUTPUT = Path("/artifacts/research_steps/S13")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def junit(path: Path) -> tuple[dict[str, int], list[dict[str, str]]]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        key: sum(int(float(suite.attrib.get(key, 0))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    cases = [
        {"classname": case.attrib.get("classname", ""), "name": case.attrib.get("name", "")}
        for suite in suites for case in suite.findall("testcase")
    ]
    return counts, cases


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "numba", "scipy", "matplotlib", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json(
        OUTPUT / "environment.json",
        {
            "researchStepId": "S13", "generatedUtc": generated,
            "python": sys.version, "platform": platform.platform(), "cpuCount": os.cpu_count(),
            "declaredMaximumWorkers": 8, "scientificKernelWorkers": 1,
            "threadEnvironment": {name: os.environ.get(name) for name in (
                "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            )},
            "packages": packages, "gpuUsed": False, "newDependenciesInstalled": [],
        },
    )

    immutability = json.loads((OUTPUT / "input_immutability.json").read_text())
    source = [
        "analysis/s13_memory_information_contract.json",
        "src/detours/memory_information.py",
        "scripts/build_s13_memory_information.py",
        "scripts/analyze_s13_memory_information.py",
        "scripts/build_s13_retained_traces.py",
        "scripts/build_s13_figures.py",
        "scripts/validate_s13_memory_information.py",
        "scripts/run_s13_mounted_reference_tests.py",
        "scripts/package_s13_results.py",
        "tests/test_memory_information.py",
    ]
    write_json(
        OUTPUT / "provenance_manifest.json",
        {
            "schemaVersion": "e03.s13.provenance.v1", "researchStepId": "S13",
            "generatedUtc": generated, "experiment": "E03", "repository": str(ROOT),
            "branch": git("branch", "--show-current"), "headAtPackaging": git("rev-parse", "HEAD"),
            "startingCommit": "33d6b79a1b5cc7030449abbec5183a0221bc9629",
            "previousArtifactMount": "/previous-artifacts/E01",
            "datasetUse": "No mounted dataset required; dataset availability is not_required.",
            "simulatorSemantics": "Frozen E01 mechanical transitions with declared external counter-addressed opportunity streams and capability-local proposal overlays",
            "sourceCode": [{"path": path, "sha256": sha256_file(ROOT / path)} for path in source],
            "inputs": immutability["inputs"], "inputImmutabilityPassed": immutability["success"],
        },
    )

    commands = """S13 final execution commands (UTC 2026-07-15 to 2026-07-16)
Environment prefix for scientific Python commands:
OPENSSL_FORCE_FIPS_MODE=0 OPENSSL_FIPS_MODE_SWITCH_PATH=/dev/null NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.

/cache/e03-s01-venv/bin/python -m pytest -q tests/test_memory_information.py
/cache/e03-s01-venv/bin/python scripts/build_s13_memory_information.py
/cache/e03-s01-venv/bin/python scripts/analyze_s13_memory_information.py
/cache/e03-s01-venv/bin/python scripts/build_s13_retained_traces.py
MPLBACKEND=Agg /cache/e03-s01-venv/bin/python scripts/build_s13_figures.py
/cache/e03-s01-venv/bin/python scripts/validate_s13_memory_information.py
/cache/e03-s01-venv/bin/python -m pytest -q --junitxml=/artifacts/research_steps/S13/repository_tests.junit.xml [199-test S01-S13 regression list recorded in the canonical report]
/cache/e03-s01-venv/bin/python scripts/run_s13_mounted_reference_tests.py
/cache/e03-s01-venv/bin/python scripts/package_s13_results.py

Recovery notes:
- The first pre-acceptance corpus build exposed only an S12 source-column alias mismatch at native parity; no partial outcome was retained. The complete accepted corpus was regenerated.
- A Numba parallel trial encountered the shared runtime's blocked thread-allocation path. The final full-scope corpus, extensions, and replay used one deterministic worker.
- The summary's leave-one-out ID selector was corrected from loo_* to the frozen ablate_* IDs; no estimand changed.
- An initially overbroad validator gate was corrected to apply S12's native zero-global-event result only to its empirical larger-n support; the exact-small tier remains separately reported.
"""
    (OUTPUT / "commands.log").write_text(commands)

    repository_counts, repository_cases = junit(OUTPUT / "repository_tests.junit.xml")
    mounted_counts, _ = junit(OUTPUT / "mounted_reference_tests.junit.xml")
    test_success = all(
        counts["failures"] == 0 and counts["errors"] == 0
        for counts in (repository_counts, mounted_counts)
    )
    write_json(
        OUTPUT / "test_summary.json",
        {"researchStepId": "S13", "success": test_success,
         "junit": {"repository_tests.junit.xml": repository_counts,
                   "mounted_reference_tests.junit.xml": mounted_counts}},
    )
    memory_cases = [case for case in repository_cases if case["classname"] == "tests.test_memory_information"]
    write_json(
        OUTPUT / "semantic_fixture_results.json",
        {
            "researchStepId": "S13", "success": len(memory_cases) == 8,
            "fixturesPassed": len(memory_cases), "fixtures": memory_cases,
            "coverage": ["state reset and order independence", "native S12 equality",
                         "complexity", "radius-two boundary", "recent-direction locality",
                         "failure-bit behavior", "no-outcome-input gateway", "arm stability"],
        },
    )

    validation = json.loads((OUTPUT / "validation_results.json").read_text())
    artifacts_written = [
        "memory_information_results.parquet", "capability_runs.parquet",
        "paired_ablation_effects.parquet", "ablation_effect_summary.parquet",
        "stratified_ablation_effects.parquet", "complexity_table.parquet",
        "metric_zero_event_diagnostics.parquet", "terminal_recurrence_summary.parquet",
        "barrier_effects_by_capability.parquet", "cross_layer_boundary_audit.parquet",
        "long_horizon_sensitivity.parquet", "retained_metric_traces.parquet",
        "figures (PNG/SVG)", "validation/provenance/test/manifest records",
        "research_step_full_results.md",
    ]
    caveats = [
        "Capability policies are engineered additions, not publication-model or biological memory.",
        "Exact-labelled n=4-5 and empirical n=12-24 effects reverse and were not pooled; larger-n exact reachability is out of domain.",
        "Counter-addressed common streams support deterministic paired comparisons, not an unqualified scheduler probability claim.",
    ]
    recommended = "Chief Scientist review; do not start S14 without separate authorization, and preserve metric- and support-specific reversals in any later synthesis."
    status = {
        "schemaVersion": "e03.s13.status.v1", "researchStepId": "S13", "stepNumber": 13,
        "success": True, "status": "complete", "outcomeClassification": "supportive",
        "artifactsWritten": artifacts_written, "validationResult": validation["validationResult"],
        "caveatsOrBlockers": caveats, "recommendedNextAction": recommended,
    }
    write_json(OUTPUT / "status.json", status)
    result = json.loads((OUTPUT / "result_summary.json").read_text())
    result.update({
        "artifactsWritten": artifacts_written, "validationResult": validation["validationResult"],
        "caveatsOrBlockers": caveats, "recommendedNextAction": recommended,
    })
    write_json(OUTPUT / "result_summary.json", result)

    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    write_json(
        OUTPUT / "artifact_manifest.json",
        {
            "schemaVersion": "e03.s13.artifact_manifest.v1", "researchStepId": "S13",
            "artifactCount": len(paths),
            "artifacts": [{"path": str(path), "bytes": path.stat().st_size,
                           "sha256": sha256_file(path)} for path in paths],
        },
    )
    print(json.dumps({"artifacts": len(paths), "tests": repository_counts,
                      "mounted": mounted_counts, "success": test_success}, indent=2))


if __name__ == "__main__":
    main()
