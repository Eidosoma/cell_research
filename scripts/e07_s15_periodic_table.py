#!/usr/bin/env python3
"""Publish the E07 periodic table atlas and final synthesis for S15."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
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

from platonic_space.periodic_table import (
    PERIODIC_TABLE_CLAIM_BOUNDARY,
    PERIODIC_TABLE_MODEL_VERSION,
    PERIODIC_TABLE_SCHEMA_VERSION,
    S13_S14_CAVEAT,
    build_claim_boundary_audit,
    build_periodic_table_bundle,
    dataframe_json_columns,
    periodic_table_outcome_classification,
    render_periodic_table_html,
    validate_periodic_table_outputs,
)
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S15"
STEP_NUMBER = 15
STEP_TITLE = "Publish the periodic table"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
FOCUSED_TESTS = ["tests.test_e07_periodic_table"]


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
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(rows, key=lambda row: row["path"])


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("\n", " ").replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def dataframe_markdown(df: pd.DataFrame, limit: int = 30) -> str:
    if df.empty:
        return "No rows."
    display = dataframe_json_columns(df.head(limit).copy())
    return markdown_table(display.astype(str).to_dict(orient="records"), [str(column) for column in display.columns])


def file_load_smoke(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(path.as_uri(), timeout=5) as response:
            payload = response.read(4096).decode("utf-8", errors="replace")
        return {
            "path": str(path),
            "uri": path.as_uri(),
            "success": "E07 Periodic Table Atlas" in payload,
            "bytesRead": len(payload.encode("utf-8")),
            "elapsedSeconds": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:  # pragma: no cover - surfaced in status JSON
        return {
            "path": str(path),
            "uri": path.as_uri(),
            "success": False,
            "error": repr(exc),
            "elapsedSeconds": round(time.perf_counter() - started, 6),
        }


def write_synthesis_report(
    path: Path,
    *,
    status: Mapping[str, Any],
    bundle_payload: Mapping[str, Any],
    step_summary: pd.DataFrame,
    law_catalog: pd.DataFrame,
    model_card_index: pd.DataFrame,
    figure_index: pd.DataFrame,
) -> None:
    entity_counts = bundle_payload.get("entityCounts", {})
    selected_steps = step_summary[
        step_summary["researchStepId"].isin(["S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"])
    ][["researchStepId", "outcomeClassification", "validationResult", "caveatsOrBlockers"]]
    law_cols = ["entityId", "displayName", "supportLevel", "outcomeClassification", "lawStatement", "scope", "s13ConstraintRole"]
    lines = [
        "# E07 Platonic Space Final Synthesis",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; exact paths and hashes are in `artifact_manifest.json`.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Claim Boundary",
        "",
        PERIODIC_TABLE_CLAIM_BOUNDARY,
        "",
        "## S13 And S14 Caveats",
        "",
        S13_S14_CAVEAT,
        "",
        "## Final Atlas Contents",
        "",
        (
            f"The final atlas packages {entity_counts.get('policies')} behavior-embedded policy entries, "
            f"{entity_counts.get('goals')} behavior-embedded goal entries, {entity_counts.get('worlds')} world entries, "
            f"and {entity_counts.get('laws')} bounded empirical laws into a standalone HTML report plus durable static tables."
        ),
        "",
        "## Anchor Results",
        "",
        dataframe_markdown(selected_steps, limit=20),
        "",
        "## Empirical Computational Laws",
        "",
        dataframe_markdown(law_catalog[[column for column in law_cols if column in law_catalog.columns]], limit=12),
        "",
        "## Model And Card Links",
        "",
        dataframe_markdown(model_card_index[["researchStepId", "title", "modelCardPath", "modelVersion"]], limit=30),
        "",
        "## Figure Links",
        "",
        dataframe_markdown(figure_index[["figureName", "figurePath", "suffix"]], limit=40),
        "",
        "## Interpretation",
        "",
        (
            "The supportive portions of E07 concern representation quality, local-neighbor retrieval, bounded surrogate screening, "
            "replay-validated counterfactual checks, inverse-design profile matching, and scoped empirical-law synthesis. The constraining "
            "portions are equally central: S09 did not find a robust broad invariant predictor, S13 did not support a simple S08-distance "
            "transfer-success rule, and S14 laws are not universal biological laws."
        ),
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
        dataframe_markdown(checks, limit=100),
        "",
        "## Hard Failures",
        "",
        "None." if failed.empty else dataframe_markdown(failed, limit=40),
        "",
        "## Claim Boundary",
        "",
        PERIODIC_TABLE_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, status: Mapping[str, Any], bundle_payload: Mapping[str, Any], checks: pd.DataFrame) -> None:
    counts = bundle_payload.get("entityCounts", {})
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
            "S15 published the final standalone E07 periodic table atlas and synthesis package. "
            f"The durable catalog contains {counts.get('combined')} total entries across policies, goals, worlds, and empirical laws."
        ),
        "",
        f"Hard validation failures: {len(checks[(checks['severity'].eq('error')) & (~checks['success'])]) if not checks.empty else 0}.",
        "",
        "S13 and S14 caveats are preserved in the atlas, synthesis report, and report-bundle input.",
        "",
        "## Claim Boundary",
        "",
        PERIODIC_TABLE_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts_dir: Path = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    reports_dir = artifacts_dir / "reports"
    model_dir = artifacts_dir / "models" / "e07_periodic_table_atlas"
    bundle_dir = artifacts_dir / "report_bundle_inputs"
    for directory in (step_dir, results_dir, reports_dir, model_dir, bundle_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    created_utc = utc_now()
    bundle = build_periodic_table_bundle(artifacts_dir)

    artifact_paths: list[Path] = []
    artifact_paths += write_dataframe(bundle.policy_catalog, step_dir / "periodic_table_policy_catalog")
    artifact_paths += write_dataframe(bundle.goal_catalog, step_dir / "periodic_table_goal_catalog")
    artifact_paths += write_dataframe(bundle.world_catalog, step_dir / "periodic_table_world_catalog")
    artifact_paths += write_dataframe(bundle.law_catalog, step_dir / "periodic_table_empirical_laws")
    artifact_paths += write_dataframe(bundle.combined_catalog, step_dir / "periodic_table_combined_catalog")
    artifact_paths += write_dataframe(bundle.step_summary, step_dir / "step_summary")
    artifact_paths += write_dataframe(bundle.artifact_index, step_dir / "final_artifact_index")
    artifact_paths += write_dataframe(bundle.model_card_index, step_dir / "model_card_index")
    artifact_paths += write_dataframe(bundle.figure_index, step_dir / "figure_index")

    artifact_paths += write_dataframe(bundle.combined_catalog, results_dir / "e07_periodic_table_catalog")
    artifact_paths += write_dataframe(bundle.artifact_index, results_dir / "e07_final_artifact_index")

    html_path = reports_dir / "e07_periodic_table.html"
    html_path.write_text(render_periodic_table_html(bundle=bundle, created_utc=created_utc), encoding="utf-8")
    artifact_paths.append(html_path)

    bundle_payload = dict(bundle.report_bundle_payload)
    bundle_payload["createdUtc"] = created_utc
    bundle_payload["stepSummaryPath"] = str(step_dir / "step_summary.parquet")
    bundle_payload["validationReportPath"] = str(step_dir / "validation_report.md")
    bundle_payload["finalStatusPath"] = str(step_dir / "status.json")
    bundle_path = bundle_dir / "e07_s15_periodic_table.json"
    write_json(bundle_path, bundle_payload)
    artifact_paths.append(bundle_path)

    model_card = {
        "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
        "modelVersion": PERIODIC_TABLE_MODEL_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "createdUtc": created_utc,
        "method": "Static deterministic atlas and final synthesis packaging from S01-S14 E07 artifacts.",
        "entityCounts": bundle_payload["entityCounts"],
        "claimBoundary": PERIODIC_TABLE_CLAIM_BOUNDARY,
        "s13S14Caveat": S13_S14_CAVEAT,
        "platform": {"python": sys.version, "platform": platform.platform()},
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }
    model_card_path = model_dir / "atlas_card.json"
    write_json(model_card_path, model_card)
    artifact_paths.append(model_card_path)

    load_smoke = file_load_smoke(html_path)
    load_smoke_path = step_dir / "local_load_smoke.json"
    write_json(load_smoke_path, load_smoke)
    artifact_paths.append(load_smoke_path)

    static_export_paths = [
        step_dir / "periodic_table_combined_catalog.csv",
        step_dir / "periodic_table_combined_catalog.parquet",
        results_dir / "e07_periodic_table_catalog.csv",
        results_dir / "e07_periodic_table_catalog.parquet",
        results_dir / "e07_final_artifact_index.csv",
        results_dir / "e07_final_artifact_index.parquet",
        bundle_path,
    ]
    caveats = (
        "Final atlas is a computational synthesis over completed simulator artifacts only; it preserves S09's no-broad-invariant result, "
        "S13's constraint against a simple S08-distance transfer-success law, and S14's bounded empirical-law scope."
    )
    recommended_next = "Chief Scientist review the final E07 atlas, static exports, bundle inputs, and caveats before any external report-bundle publication."
    synthesis_report_path = reports_dir / "e07_platonic_space_report.md"
    preliminary_status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": False,
        "status": "pending_validation",
        "artifactsWritten": [str(path) for path in [*artifact_paths, synthesis_report_path]],
        "validationResult": "pending validation",
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next,
        "outcomeClassification": "pending",
    }
    write_synthesis_report(
        synthesis_report_path,
        status=preliminary_status,
        bundle_payload=bundle_payload,
        step_summary=bundle.step_summary,
        law_catalog=bundle.law_catalog,
        model_card_index=bundle.model_card_index,
        figure_index=bundle.figure_index,
    )
    artifact_paths.append(synthesis_report_path)

    checks = validate_periodic_table_outputs(
        artifacts_dir=artifacts_dir,
        bundle=bundle,
        html_path=html_path,
        report_path=reports_dir / "e07_platonic_space_report.md",
        bundle_path=bundle_path,
        static_export_paths=static_export_paths,
    )
    checks = pd.concat(
        [
            checks,
            pd.DataFrame(
                [
                    {
                        "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
                        "checkId": "local_file_url_smoke",
                        "severity": "error",
                        "success": bool(load_smoke["success"]),
                        "observed": json.dumps(load_smoke, sort_keys=True),
                        "expected": "atlas opens through file:// URL and contains title",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    test_result = None
    if not args.skip_tests:
        test_result = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS])
        checks = pd.concat(
            [
                checks,
                pd.DataFrame(
                    [
                        {
                            "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
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

    claim_audit = build_claim_boundary_audit(
        bundle.step_summary,
        bundle.law_catalog,
        reports=(reports_dir / "e07_periodic_table.html", reports_dir / "e07_platonic_space_report.md"),
    )
    report_claim_checks = claim_audit[claim_audit["checkId"].astype(str).str.startswith("claim_scan::")]
    checks = pd.concat([checks, report_claim_checks], ignore_index=True)
    artifact_paths += write_dataframe(claim_audit, step_dir / "claim_boundary_audit")
    artifact_paths += write_dataframe(checks, step_dir / "atlas_validation_checks")

    hard_failures = checks[checks["severity"].eq("error") & ~checks["success"]]
    outcome = periodic_table_outcome_classification(checks)
    status_state = "completed" if hard_failures.empty else "completed_with_validation_errors"
    validation_result = (
        f"passed: standalone atlas loaded locally, {len(bundle.combined_catalog)} combined catalog entries, "
        f"{len(bundle.artifact_index)} indexed upstream artifacts, {len(bundle.model_card_index)} model/card links, "
        f"{len(bundle.figure_index)} figure links, 0 hard validation failures"
        if hard_failures.empty
        else f"failed: {len(hard_failures)} hard validation failures"
    )
    summary_path = step_dir / "summary.md"
    validation_report_path = step_dir / "validation_report.md"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    test_log_path = step_dir / "focused_tests.json"
    if test_result is not None:
        write_json(test_log_path, test_result)
        artifact_paths.append(test_log_path)
    artifact_paths += [summary_path, validation_report_path, status_path, manifest_path]

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
        "entityCounts": bundle_payload["entityCounts"],
        "artifactIndexCount": int(len(bundle.artifact_index)),
        "modelCardLinkCount": int(len(bundle.model_card_index)),
        "figureLinkCount": int(len(bundle.figure_index)),
        "localAtlasUrl": html_path.as_uri(),
        "laySummary": "The final E07 package is a local, browsable computational atlas linking policies, goals, worlds, validation results, empirical laws, model cards, figures, and caveats.",
        "repositoryCodePaths": [
            "platonic_space/periodic_table.py",
            "scripts/e07_s15_periodic_table.py",
            "tests/test_e07_periodic_table.py",
        ],
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
            "statusShort": git_value(["status", "--short"]),
            "remote": git_value(["remote", "get-url", "origin"]),
        },
        "claimBoundary": PERIODIC_TABLE_CLAIM_BOUNDARY,
        "s13S14Caveat": S13_S14_CAVEAT,
    }

    write_synthesis_report(
        synthesis_report_path,
        status=status,
        bundle_payload=bundle_payload,
        step_summary=bundle.step_summary,
        law_catalog=bundle.law_catalog,
        model_card_index=bundle.model_card_index,
        figure_index=bundle.figure_index,
    )
    write_summary(summary_path, status, bundle_payload, checks)
    write_validation_report(validation_report_path, status, checks)
    write_json(status_path, status)
    manifest = {
        "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdUtc": utc_now(),
        "artifacts": collect_artifacts(path for path in artifact_paths if path != manifest_path),
        "repositoryCodePaths": status["repositoryCodePaths"],
        "git": status["git"],
        "claimBoundary": PERIODIC_TABLE_CLAIM_BOUNDARY,
        "s13S14Caveat": S13_S14_CAVEAT,
    }
    write_json(manifest_path, manifest)

    print(json.dumps({key: status[key] for key in ("researchStepId", "success", "status", "validationResult", "outcomeClassification")}, indent=2))
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
