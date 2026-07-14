#!/usr/bin/env python3
"""Package compact S12 validation, provenance, and artifact records."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.delayed_gratification import (  # noqa: E402
    CACHE_DIR,
    FIGURE6_RUNS,
    FIGURE7_RUNS,
    OUTPUT_DIR,
    PREREGISTRATION,
    PREREGISTRATION_SHA256,
    sha256_file,
    write_json,
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def file_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    results = pd.read_parquet(OUTPUT_DIR / "original_dg_results.parquet")
    envelope = pd.read_parquet(OUTPUT_DIR / "figure6_ambiguity_envelope.parquet")
    claims = pd.read_csv(OUTPUT_DIR / "figure6_7_claim_classifications.csv")
    fixture = json.loads((OUTPUT_DIR / "hand_calculated_metric_fixtures.json").read_text())
    frozen = json.loads((OUTPUT_DIR / "frozen_code_metric_comparison.json").read_text())
    replay = json.loads((OUTPUT_DIR / "exact_replay_samples.json").read_text())
    path_check = json.loads((OUTPUT_DIR / "s09_no_fault_path_comparison.json").read_text())
    population = json.loads((OUTPUT_DIR / "population_validation.json").read_text())
    pre = json.loads((OUTPUT_DIR / "source_checksum_pre_run.json").read_text())
    post = json.loads((OUTPUT_DIR / "source_checksum_post_run.json").read_text())
    junit_root = ET.parse(OUTPUT_DIR / "repository_tests.junit.xml").getroot()
    junit = junit_root if junit_root.tag == "testsuite" else junit_root.find("testsuite")
    if junit is None:
        raise ValueError("JUnit report has no testsuite")

    checks = {
        "preregistrationHash": sha256_file(PREREGISTRATION) == PREREGISTRATION_SHA256,
        "figure7RunAccounting": len(results) == FIGURE7_RUNS,
        "figure6RunAccounting": len(envelope) == FIGURE6_RUNS,
        "zeroCensoring": int(results.censored.sum() + envelope.censored.sum()) == 0,
        "handFixtures9": fixture["success"] and fixture["fixtureCount"] == 9,
        "frozenCodeRandomized1000": frozen["success"] and frozen["samples"] == 1_000,
        "exactReplay24": replay["success"] and replay["samples"] == 24,
        "s09ReferencePath6": path_check["success"] and path_check["samples"] == 6,
        "populationValidation": population["success"],
        "sourceChecksumPre": pre["contentValid"] and pre["filesChecked"] == 91,
        "sourceChecksumPost": post["contentValid"] and post["filesChecked"] == 91,
        "claimClassifications32": len(claims) == 32 and claims.claim_id.nunique() == 32,
        "repositoryTests78": int(junit.attrib.get("tests", 0)) == 78
        and int(junit.attrib.get("failures", 0)) == 0
        and int(junit.attrib.get("errors", 0)) == 0,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    validation = {
        "schemaVersion": "e01.s12.validation.v1",
        "researchStepId": "S12",
        "success": all(checks.values()),
        "checks": checks,
        "runAccounting": {
            "figure7": len(results),
            "figure6": len(envelope),
            "total": len(results) + len(envelope),
        },
        "repositoryTests": {
            "tests": int(junit.attrib.get("tests", 0)),
            "failures": int(junit.attrib.get("failures", 0)),
            "errors": int(junit.attrib.get("errors", 0)),
        },
        "unexplainedFailures": [],
    }
    write_json(OUTPUT_DIR / "validation_summary.json", validation)

    missing = {
        "researchStepId": "S12",
        "frozenPublicCommit": "1fd2bd5921c1f6b423a71f691d5189106a8a1020",
        "publicationSnapshotClaimed": False,
        "missingEvidence": [
            "exact publication commit/tag/release",
            "original Figure 6 event stream, algorithm label, scheduler, and omitted intermediate states",
            "original Figure 7 replicate arrays, random streams, fault maps, and error-bar construction",
            "historical traditional and stuck-fault generators",
            "historical activation-level observations and state hashes",
            "portable historical timing and scheduler environment",
        ],
        "nonSubstitutionDecision": "Clean-room trajectories remain labeled regenerated; no result is represented as missing historical bytes or streams.",
    }
    write_json(OUTPUT_DIR / "missing_evidence.json", missing)

    inputs = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"),
        Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md"),
        Path("/artifacts/research_steps/S01/claim_registry.parquet"),
        Path("/artifacts/research_steps/S03/transition_contract.json"),
        Path("/artifacts/research_steps/S04/patch_ledger.json"),
        Path("/artifacts/research_steps/S05/reference_simulator_release.json"),
        Path("/artifacts/research_steps/S06/event_schema.json"),
        Path("/artifacts/research_steps/S07/invariant_coverage_matrix.csv"),
        Path("/artifacts/research_steps/S08/paired_scenario_bank.parquet"),
        Path("/artifacts/research_steps/S09/no_fault_runs.parquet"),
        Path("/artifacts/research_steps/S10/cost_ledger.parquet"),
        Path("/artifacts/research_steps/S11/frozen_cell_results.parquet"),
        PREREGISTRATION,
    ]
    checkpoints = [
        CACHE_DIR / "figure7_paper_scale.jsonl",
        CACHE_DIR / "figure6_ambiguity_envelope.jsonl",
    ]
    provenance = {
        "schemaVersion": "e01.s12.provenance.v1",
        "researchStepId": "S12",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "repositoryHeadAtPackaging": git("rev-parse", "HEAD"),
        "repositoryBranch": git("branch", "--show-current"),
        "preregistrationSha256": sha256_file(PREREGISTRATION),
        "inputs": [file_record(path) for path in inputs if path.exists()],
        "cacheCheckpoints": [file_record(path) for path in checkpoints],
        "sourceCode": [
            "analysis/delayed_gratification.py",
            "analysis/s12_delayed_gratification_preregistration.json",
            "analysis/frozen_cell_results.py",
            "reference_simulator/engine.py",
            "scripts/replicate_delayed_gratification.py",
            "scripts/package_s12_evidence.py",
            "tests/test_delayed_gratification.py",
        ],
    }
    write_json(OUTPUT_DIR / "provenance.json", provenance)

    commands = {
        "researchStepId": "S12",
        "threadEnvironment": "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "commands": [
            "python scripts/replicate_delayed_gratification.py validate --workers 8",
            "python scripts/replicate_delayed_gratification.py figure7 --workers 8",
            "python scripts/replicate_delayed_gratification.py figure6 --workers 8",
            "python scripts/replicate_delayed_gratification.py replay --workers 8",
            "python scripts/replicate_delayed_gratification.py analyze --workers 8",
            "python scripts/replicate_delayed_gratification.py final-validation --workers 8",
            "pytest -q --junitxml=/artifacts/research_steps/S12/repository_tests.junit.xml",
            "python scripts/package_s12_evidence.py",
        ],
    }
    write_json(OUTPUT_DIR / "commands.json", commands)
    write_json(
        OUTPUT_DIR / "repository_release.json",
        {
            "researchStepId": "S12",
            "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "remote": git("remote", "get-url", "origin"),
            "publicationSnapshotClaimed": False,
        },
    )
    paths = sorted(
        path for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    write_json(
        OUTPUT_DIR / "artifact_manifest.json",
        {
            "schemaVersion": "e01.s12.artifact_manifest.v1",
            "researchStepId": "S12",
            "artifactCount": len(paths),
            "artifacts": [file_record(path) for path in paths],
        },
    )
    print(json.dumps({"success": validation["success"], "runs": len(results) + len(envelope), "claims": len(claims)}, indent=2))
    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
