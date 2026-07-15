#!/usr/bin/env python3
"""Write S07 environment, result, command, provenance, and artifact manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import pandas as pd


OUTPUT = Path("/artifacts/research_steps/S07")
REPOSITORY = Path(__file__).resolve().parents[1]
STARTING_COMMIT = "a7f61294eb424e59474b50f50b7fe0cac1c2b99b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
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


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def main() -> None:
    validation = json.loads((OUTPUT / "validation_results.json").read_text())
    summary_validation = json.loads((OUTPUT / "summary_validation.json").read_text())
    focused = junit_counts(OUTPUT / "focused_tests.junit.xml")
    repository_tests = junit_counts(OUTPUT / "repository_tests.junit.xml")
    if focused != {"tests": 49, "failures": 0, "errors": 0, "skipped": 0}:
        raise AssertionError(focused)
    if repository_tests != {"tests": 140, "failures": 0, "errors": 0, "skipped": 0}:
        raise AssertionError(repository_tests)

    prevalence = pd.read_parquet(OUTPUT / "necessity_prevalence_by_family.parquet")
    totals = prevalence.groupby("metric")[
        [
            "complete_start_count",
            "reachable_no_detour_count",
            "necessary_detour_count",
            "unreachable_active_count",
            "quiescent_count",
            "reachable_active_count",
        ]
    ].sum()
    signatures = pd.read_parquet(OUTPUT / "necessity_signature_counts.parquet")
    overall_signatures = signatures[
        (signatures.stratum_type == "overall")
        & (signatures.scope == "reachable_active_only")
    ]
    signature_counts = {
        int(row.signature_code): int(row.state_count)
        for row in overall_signatures.itertuples()
    }
    reachable_active = int(totals.iloc[0].reachable_active_count)
    result_summary = {
        "schemaVersion": "e03.s07.result_summary.v1",
        "researchStepId": "S07",
        "stepNumber": 7,
        "success": True,
        "status": "complete",
        "outcomeClassification": "supportive_with_constraining_scope",
        "families": 7984,
        "states": 22301808,
        "stateMetricRows": 89207232,
        "edges": 103599386,
        "completeStarts": 207594,
        "goalReachableActive": reachable_active,
        "unreachableActive": 10387749,
        "quiescent": 2492408,
        "metrics": {
            metric: {
                "reachableNoDetour": int(row.reachable_no_detour_count),
                "necessaryDetour": int(row.necessary_detour_count),
                "necessaryPrevalenceReachableActive": float(
                    row.necessary_detour_count / row.reachable_active_count
                ),
            }
            for metric, row in totals.iterrows()
        },
        "crossMetric": {
            "noneNecessary": signature_counts[0],
            "anyNecessary": reachable_active - signature_counts[0],
            "adjacentOnly": signature_counts[1],
            "globalOnly": sum(
                count
                for code, count in signature_counts.items()
                if code != 0 and (code & 1) == 0
            ),
            "allFour": signature_counts[15],
        },
        "validation": {
            "corpusGates": validation["gateCount"],
            "n4ThresholdFamilyMetrics": validation["n4ThresholdFamilyMetricCrosschecks"],
            "tinyProfileStartComparisons": validation["tinyProfileStartComparisons"],
            "witnessRows": validation["witnessRowsReplayed"],
            "witnessEdges": validation["witnessEdgesReplayed"],
            "deterministicFamilies": validation["deterministicFamiliesRebuilt"],
            "inputHashes": validation["inputHashesChecked"],
            "focusedTests": focused["tests"],
            "repositoryTests": repository_tests["tests"],
            "summaryValidation": summary_validation["success"],
        },
        "caveatsOrBlockers": [
            "existential finite S05 structural-opportunity paths only",
            "state-weighted structural prevalence is not observed behavior",
            "tiered unique-value n=4–9 scope cannot be extrapolated to omitted families",
            "unreachable and quiescent starts have undefined excursion",
            "start-conditioned minimax secondary optima are represented by a stratified exact witness panel",
            "no S07 budget, omission, semantic, solver, witness, or mutation blocker remains",
        ],
        "recommendedNextAction": (
            "Await separate authorization for S08; then compare observed E01 behavior "
            "with exact S07 starts, edges, metric peaks, and cost labels."
        ),
    }
    write_json(OUTPUT / "result_summary.json", result_summary)

    environment = {
        "schemaVersion": "e03.s07.environment.v1",
        "researchStepId": "S07",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyarrow", "numba", "pytest"]
        },
        "cpuCount": os.cpu_count(),
        "constructionWorkers": 8,
        "validationWorkers": 8,
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

    commands = """S07 reproducible commands (working directory /workspace/cell-research)
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python -m pytest -q tests/test_path_solutions.py tests/test_necessary_detour.py --junitxml=/artifacts/research_steps/S07/focused_tests.junit.xml
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/build_s07_solutions.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/build_s07_witnesses.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/validate_s07_solutions.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python -m pytest -q tests/test_path_solutions.py tests/test_necessary_detour.py tests/test_detour_transition_graph.py tests/test_detour_state_space.py tests/test_distances.py tests/test_schedulers.py tests/test_faults.py tests/test_architectures.py --junitxml=/artifacts/research_steps/S07/repository_tests.junit.xml
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /cache/e03-s01-venv/bin/python scripts/summarize_s07_solutions.py
PYTHONPATH=. /cache/e03-s01-venv/bin/python -m py_compile src/detours/path_solutions.py scripts/build_s07_solutions.py scripts/build_s07_witnesses.py scripts/validate_s07_solutions.py scripts/summarize_s07_solutions.py scripts/package_s07_results.py tests/test_path_solutions.py
PYTHONPATH=. /cache/e03-s01-venv/bin/python scripts/package_s07_results.py
git diff --check

Execution notes:
- Five unique lexicographic searches cover six frozen cost profiles because displaced cells equal exactly twice accepted swaps and share their witness/tie-break order.
- The complete build and validator both used eight family-shard workers with nested numerical threads disabled.
- All complete computations succeeded; no failed partial S07 corpus or waived validation is retained.
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
        *[
            Path(f"/artifacts/research_steps/S0{step}/research_step_full_results.md")
            for step in range(1, 7)
        ],
        Path("/artifacts/research_steps/S01/distance_spec.md"),
        Path("/artifacts/research_steps/S04/canonical_state_encoder.md"),
        Path("/artifacts/research_steps/S04/state_family_inventory.parquet"),
        Path("/artifacts/research_steps/S05/graph_schema.md"),
        Path("/artifacts/research_steps/S05/graph_corpus_manifest.json"),
        Path("/artifacts/research_steps/S05/graph_family_manifest.parquet"),
        Path("/artifacts/research_steps/S05/validation_results.json"),
        Path("/artifacts/research_steps/S06/necessary_detour_spec.md"),
        Path("/artifacts/research_steps/S06/proof_sketches.md"),
        Path("/artifacts/research_steps/S06/solver_validation_fixtures.json"),
        Path("/artifacts/research_steps/S06/validation_results.json"),
        Path("/previous-artifacts/E01/research_steps/S03/state_diagrams.md"),
        Path("/previous-artifacts/E01/research_steps/S05/semantic_decisions.json"),
        Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    ]
    source_paths = [
        REPOSITORY / "analysis/s06_necessary_detour_contract.json",
        REPOSITORY / "analysis/s07_path_solution_contract.json",
        REPOSITORY / "src/detours/necessary_detour.py",
        REPOSITORY / "src/detours/path_solutions.py",
        REPOSITORY / "scripts/build_s07_solutions.py",
        REPOSITORY / "scripts/build_s07_witnesses.py",
        REPOSITORY / "scripts/validate_s07_solutions.py",
        REPOSITORY / "scripts/summarize_s07_solutions.py",
        REPOSITORY / "scripts/package_s07_results.py",
        REPOSITORY / "tests/test_necessary_detour.py",
        REPOSITORY / "tests/test_path_solutions.py",
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
        "schemaVersion": "e03.s07.provenance.v1",
        "researchStepId": "S07",
        "stepNumber": 7,
        "success": True,
        "status": "complete",
        "repository": {
            "path": str(REPOSITORY),
            "branch": git("branch", "--show-current"),
            "startingCommit": STARTING_COMMIT,
            "headBeforeS07Commit": git("rev-parse", "HEAD"),
        },
        "inputs": [record(path) for path in input_paths],
        "repositorySources": [record(path) for path in source_paths],
        "previousArtifactMount": "/previous-artifacts/E01",
        "datasetInputsUsed": [],
        "networkInputsUsed": [],
        "directInputImmutability": json.loads(
            (OUTPUT / "input_immutability.json").read_text()
        ),
    }
    write_json(OUTPUT / "provenance_manifest.json", provenance)

    excluded = {OUTPUT / "artifact_manifest.json"}
    artifacts = [
        record(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path not in excluded
    ]
    manifest = {
        "schemaVersion": "e03.s07.artifact_manifest.v1",
        "researchStepId": "S07",
        "stepNumber": 7,
        "success": True,
        "status": "complete",
        "artifactsWritten": [item["path"] for item in artifacts]
        + [str(OUTPUT / "artifact_manifest.json")],
        "artifacts": artifacts,
        "validationResult": (
            "pass: 12 corpus gates; all 7,984 families/22,301,808 states/"
            "89,207,232 state-metric rows; 19,488 n=4 threshold cases; 4,760 "
            "tiny profile-start comparisons; 1,157 witnesses/9,760 edges; 31 "
            "deterministic families; 25 input hashes; 140 repository tests"
        ),
        "outcomeClassification": "supportive_with_constraining_scope",
        "caveatsOrBlockers": result_summary["caveatsOrBlockers"],
        "recommendedNextAction": result_summary["recommendedNextAction"],
    }
    write_json(OUTPUT / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "artifacts": len(artifacts) + 1,
                "artifactBytes": sum(int(item["bytes"]) for item in artifacts),
                "success": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
