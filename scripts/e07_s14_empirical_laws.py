#!/usr/bin/env python3
"""Derive E07 S14 bounded empirical computational laws from S01-S13 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
for thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_var, "1")

import pandas as pd

from platonic_space.empirical_laws import (
    EMPIRICAL_LAW_CLAIM_BOUNDARY,
    EMPIRICAL_LAW_MODEL_VERSION,
    EMPIRICAL_LAW_SCHEMA_VERSION,
    build_empirical_law_synthesis,
    dataframe_json_columns,
    law_outcome_classification,
    validation_checks,
)
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S14"
STEP_NUMBER = 14
STEP_TITLE = "Derive empirical laws"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_PREVIOUS_ARTIFACTS_DIR = Path("/previous-artifacts")
FOCUSED_TESTS = ["tests.test_e07_empirical_laws"]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def run_command(command: Sequence[str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    return {
        "command": list(command),
        "cwd": str(cwd),
        "returncode": int(completed.returncode),
        "success": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsedSeconds": round(time.perf_counter() - started, 6),
    }


def git_value(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def write_dataframe(df: pd.DataFrame, stem: Path, csv: bool = True) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    out = dataframe_json_columns(df)
    if csv:
        csv_path = stem.with_suffix(".csv")
        out.to_csv(csv_path, index=False)
        paths.append(csv_path)
    parquet_path = stem.with_suffix(".parquet")
    out.to_parquet(parquet_path, index=False)
    paths.append(parquet_path)
    return paths


def collect_artifacts(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(out, key=lambda row: row["path"])


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("\n", " ").replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def dataframe_markdown(df: pd.DataFrame, limit: int = 40) -> str:
    if df.empty:
        return "No rows."
    display = dataframe_json_columns(df.head(limit).copy())
    return markdown_table(display.astype(str).to_dict(orient="records"), [str(column) for column in display.columns])


def write_empirical_laws_report(
    path: Path,
    *,
    status: Mapping[str, Any],
    laws: pd.DataFrame,
    evidence: pd.DataFrame,
    counters: pd.DataFrame,
    uncertainty: pd.DataFrame,
    falsification: pd.DataFrame,
) -> None:
    law_cols = ["lawId", "title", "lawFamily", "supportLevel", "outcomeClassification", "lawStatement", "scope", "s13ConstraintRole"]
    evidence_cols = ["lawId", "evidenceRole", "artifactPath", "metricName", "metricValue", "interpretation"]
    lines = [
        f"# {STEP_ID} Empirical Computational Laws",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; exact paths and hashes are in `artifact_manifest.json`.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Laws",
        "",
        dataframe_markdown(laws[[column for column in law_cols if column in laws.columns]], limit=20),
        "",
        "## Evidence Links",
        "",
        dataframe_markdown(evidence[[column for column in evidence_cols if column in evidence.columns]], limit=80),
        "",
        "## Counterexamples",
        "",
        dataframe_markdown(counters, limit=80),
        "",
        "## Uncertainty",
        "",
        dataframe_markdown(uncertainty, limit=80),
        "",
        "## Falsification Conditions",
        "",
        dataframe_markdown(falsification, limit=80),
        "",
        "## S13 Constraint",
        "",
        (
            "S13 is treated as constraining evidence against a simple S08-distance transfer-success law. "
            "The transfer law in this catalog requires documented observation/action/goal mappings, controls, and direct replay."
        ),
        "",
        "## Claim Boundary",
        "",
        EMPIRICAL_LAW_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_report(path: Path, status: Mapping[str, Any], checks: pd.DataFrame) -> None:
    failed = checks[(checks["severity"].eq("error")) & (~checks["success"])] if not checks.empty else pd.DataFrame()
    lines = [
        f"# {STEP_ID} Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; exact paths and hashes are in `artifact_manifest.json`.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Validation Checks",
        "",
        dataframe_markdown(checks, limit=80),
        "",
        "## Hard Failures",
        "",
        "None." if failed.empty else dataframe_markdown(failed, limit=40),
        "",
        "## Claim Boundary",
        "",
        EMPIRICAL_LAW_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, status: Mapping[str, Any], laws: pd.DataFrame, checks: pd.DataFrame) -> None:
    support_counts = laws["supportLevel"].value_counts().to_dict() if "supportLevel" in laws else {}
    lines = [
        f"# {STEP_ID} Summary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; status and manifest enumerate exact paths.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        f"Outcome classification: {status['outcomeClassification']}.",
        "",
        (
            f"S14 derived {len(laws)} bounded empirical computational laws from S01-S13 and selected upstream E04/E06 summaries. "
            "The catalog includes supportive local-map, surrogate-screening, mechanism, chimeric-context, taxonomy, and inverse-design laws, "
            "plus constraining laws for broad invariants and substrate transfer."
        ),
        "",
        f"Support-level counts: `{json.dumps(support_counts, sort_keys=True)}`.",
        f"Hard validation failures: {len(checks[(checks['severity'].eq('error')) & (~checks['success'])]) if not checks.empty else 0}.",
        "",
        "S13 is explicitly recorded as constraining evidence against a simple S08-distance transfer-success law.",
        "",
        "## Law Snapshot",
        "",
        dataframe_markdown(laws[["lawId", "title", "supportLevel", "outcomeClassification", "recommendedUse"]], limit=20),
        "",
        "## Claim Boundary",
        "",
        EMPIRICAL_LAW_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--previous-artifacts-dir", type=Path, default=DEFAULT_PREVIOUS_ARTIFACTS_DIR)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts_dir: Path = args.artifacts_dir
    previous_artifacts_dir: Path = args.previous_artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    reports_dir = artifacts_dir / "reports"
    model_dir = artifacts_dir / "models" / "e07_empirical_laws"
    bundle_dir = artifacts_dir / "report_bundle_inputs"
    for directory in (step_dir, results_dir, reports_dir, model_dir, bundle_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    synthesis = build_empirical_law_synthesis(artifacts_dir, previous_artifacts_dir)
    checks = validation_checks(synthesis, artifacts_dir=artifacts_dir, previous_artifacts_dir=previous_artifacts_dir)

    test_result = None
    if not args.skip_tests:
        test_result = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS])
        checks = pd.concat(
            [
                checks,
                pd.DataFrame(
                    [
                        {
                            "checkId": "focused_unit_tests",
                            "severity": "error",
                            "success": bool(test_result["success"]),
                            "observed": f"returncode={test_result['returncode']}",
                            "expected": "returncode=0",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    hard_failures = checks[checks["severity"].eq("error") & ~checks["success"]]
    outcome = law_outcome_classification(checks)
    status_state = "completed" if hard_failures.empty else "completed_with_validation_errors"
    validation_result = (
        f"passed: {len(synthesis.law_catalog)} bounded empirical laws, {len(synthesis.evidence_links)} evidence links, "
        f"{len(synthesis.counterexamples)} counterexamples, {len(synthesis.uncertainty_register)} uncertainty rows, "
        f"{len(synthesis.falsification_register)} falsification conditions, 0 hard validation failures"
        if hard_failures.empty
        else f"failed: {len(hard_failures)} hard validation failures"
    )
    caveats = (
        "Candidate laws are empirical computational regularities over S01-S13 and selected E04/E06 summary artifacts only; "
        "S09 remains constraining against broad invariant laws; S13 constrains a simple S08-distance transfer-success law; "
        "all laws require scoped use, counterexamples, uncertainty, and falsification conditions."
    )
    recommended_next = "Chief Scientist review S14 empirical-law catalog, evidence links, counterexamples, uncertainty, and falsification conditions before authorizing S15."

    artifact_paths: list[Path] = []
    artifact_paths += write_dataframe(synthesis.law_catalog, step_dir / "empirical_law_catalog")
    artifact_paths += write_dataframe(synthesis.evidence_links, step_dir / "law_evidence_links")
    artifact_paths += write_dataframe(synthesis.counterexamples, step_dir / "law_counterexamples")
    artifact_paths += write_dataframe(synthesis.uncertainty_register, step_dir / "law_uncertainty_register")
    artifact_paths += write_dataframe(synthesis.falsification_register, step_dir / "law_falsification_register")
    artifact_paths += write_dataframe(checks, step_dir / "law_validation_checks")
    artifact_paths += write_dataframe(synthesis.law_catalog, results_dir / "e07_empirical_laws")
    artifact_paths += write_dataframe(synthesis.evidence_links, results_dir / "e07_empirical_law_evidence")

    anchor_path = step_dir / "law_anchor_metrics.json"
    write_json(anchor_path, dict(synthesis.anchor_metrics))
    artifact_paths.append(anchor_path)

    model_card = {
        "schemaVersion": EMPIRICAL_LAW_SCHEMA_VERSION,
        "modelVersion": EMPIRICAL_LAW_MODEL_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "createdUtc": utc_now(),
        "method": "Deterministic synthesis of bounded law records from S01-S13 artifact metrics and selected previous E04/E06 summary artifacts.",
        "lawCount": int(len(synthesis.law_catalog)),
        "evidenceLinkCount": int(len(synthesis.evidence_links)),
        "counterexampleCount": int(len(synthesis.counterexamples)),
        "uncertaintyRowCount": int(len(synthesis.uncertainty_register)),
        "falsificationConditionCount": int(len(synthesis.falsification_register)),
        "s13Constraint": "S13 is treated as constraining evidence against a simple S08-distance transfer-success law.",
        "claimBoundary": EMPIRICAL_LAW_CLAIM_BOUNDARY,
        "platform": {"python": sys.version, "platform": platform.platform()},
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }
    model_card_path = model_dir / "law_synthesis_card.json"
    write_json(model_card_path, model_card)
    artifact_paths.append(model_card_path)

    bundle_payload = {
        "schemaVersion": EMPIRICAL_LAW_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "lawCatalogPath": str(step_dir / "empirical_law_catalog.parquet"),
        "evidenceLinksPath": str(step_dir / "law_evidence_links.parquet"),
        "reportPath": str(reports_dir / "e07_empirical_laws.md"),
        "s13Constraint": "Treat S13 as constraining evidence against simple S08-distance transfer-success claims.",
        "claimBoundary": EMPIRICAL_LAW_CLAIM_BOUNDARY,
    }
    bundle_path = bundle_dir / "e07_s14_empirical_laws.json"
    write_json(bundle_path, bundle_payload)
    artifact_paths.append(bundle_path)

    test_log_path = step_dir / "focused_tests.json"
    if test_result is not None:
        write_json(test_log_path, test_result)
        artifact_paths.append(test_log_path)

    summary_path = step_dir / "summary.md"
    validation_report_path = step_dir / "validation_report.md"
    step_report_path = step_dir / "empirical_laws_report.md"
    root_report_path = reports_dir / "e07_empirical_laws.md"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    artifact_paths += [summary_path, validation_report_path, step_report_path, root_report_path, status_path, manifest_path]

    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(hard_failures.empty),
        "status": status_state,
        "artifactsWritten": [str(path) for path in artifact_paths],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next,
        "experimentId": EXPERIMENT_ID,
        "title": STEP_TITLE,
        "outcomeClassification": outcome,
        "completedUtc": utc_now(),
        "runtimeSeconds": round(time.perf_counter() - started, 3),
        "lawCount": int(len(synthesis.law_catalog)),
        "evidenceLinkCount": int(len(synthesis.evidence_links)),
        "counterexampleCount": int(len(synthesis.counterexamples)),
        "uncertaintyRowCount": int(len(synthesis.uncertainty_register)),
        "falsificationConditionCount": int(len(synthesis.falsification_register)),
        "repositoryCodePaths": [
            "platonic_space/empirical_laws.py",
            "scripts/e07_s14_empirical_laws.py",
            "tests/test_e07_empirical_laws.py",
        ],
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
            "statusShort": git_value(["status", "--short"]),
            "remote": git_value(["remote", "get-url", "origin"]),
        },
        "claimBoundary": EMPIRICAL_LAW_CLAIM_BOUNDARY,
    }

    write_summary(summary_path, status, synthesis.law_catalog, checks)
    write_validation_report(validation_report_path, status, checks)
    write_empirical_laws_report(
        step_report_path,
        status=status,
        laws=synthesis.law_catalog,
        evidence=synthesis.evidence_links,
        counters=synthesis.counterexamples,
        uncertainty=synthesis.uncertainty_register,
        falsification=synthesis.falsification_register,
    )
    write_empirical_laws_report(
        root_report_path,
        status=status,
        laws=synthesis.law_catalog,
        evidence=synthesis.evidence_links,
        counters=synthesis.counterexamples,
        uncertainty=synthesis.uncertainty_register,
        falsification=synthesis.falsification_register,
    )
    write_json(status_path, status)
    manifest = {
        "schemaVersion": EMPIRICAL_LAW_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdUtc": utc_now(),
        "artifacts": collect_artifacts(artifact_paths),
        "repositoryCodePaths": status["repositoryCodePaths"],
        "git": status["git"],
        "claimBoundary": EMPIRICAL_LAW_CLAIM_BOUNDARY,
    }
    write_json(manifest_path, manifest)

    print(json.dumps({key: status[key] for key in ("researchStepId", "success", "status", "validationResult", "outcomeClassification")}, indent=2))
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

