#!/usr/bin/env python3
"""Build E07 S03 abstract goal catalog artifacts."""

from __future__ import annotations

import argparse
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

import pandas as pd

from platonic_space.goal_catalog import (
    GOAL_CLAIM_BOUNDARY,
    GOAL_CATALOG_VERSION,
    GOAL_SCHEMA_VERSION,
    build_goal_catalog,
    catalog_to_dataframe,
    compact_json,
    coverage_summary,
    goal_schema_document,
    goal_world_links_dataframe,
    validate_goal_catalog,
)
from platonic_space.world_schema import sha256_path, write_json


STEP_ID = "S03"
STEP_NUMBER = 3
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_PREVIOUS_ARTIFACTS_DIR = Path("/previous-artifacts")
DEFAULT_WORLD_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S01" / "normalized_world_catalog.parquet"
DEFAULT_POLICY_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S02" / "policy_abstract_catalog.parquet"
FOCUSED_TESTS = ["tests.test_e07_goal_catalog"]


def run_command(command: Sequence[str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
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


def write_dataframe(df: pd.DataFrame, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    parquet_path = stem.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(compact_json(record) + "\n")
    return path


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


def dataframe_markdown(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    rows = df.astype(str).to_dict(orient="records")
    return markdown_table(rows, columns)


def write_validation_report(path: Path, summary: Mapping[str, Any], validation: pd.DataFrame) -> None:
    failed = validation[~validation["success"]]
    family_rows = [{"goalFamily": key, "records": value} for key, value in sorted(summary["goalFamilyCounts"].items())]
    kind_rows = [{"goalKind": key, "records": value} for key, value in sorted(summary["goalKindCounts"].items())]
    representation_rows = [
        {"representationType": key, "records": value} for key, value in sorted(summary["representationTypeCounts"].items())
    ]
    source_rows = [
        {"sourceExperimentId": key, "records": value} for key, value in sorted(summary["sourceExperimentCounts"].items())
    ]
    policy_rows = [{"policyLinkMode": key, "records": value} for key, value in sorted(summary["policyLinkModeCounts"].items())]
    lines = [
        f"# {STEP_ID} Goal Catalog Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {'completed' if summary['allHardChecksPassed'] else 'completed_with_validation_failures'}",
        "- Artifacts written: `goal_schema.json`, `goal_catalog.{csv,parquet,jsonl}`, `goal_world_links.{csv,parquet}`, `goal_catalog_validation.{csv,parquet}`, `goal_catalog_validation_summary.json`, `validation_report.md`, `summary.md`, `status.json`, `artifact_manifest.json`, and `$ARTIFACTS_DIR/data/e07_goal_catalog.{csv,parquet}`.",
        f"- Validation result: {'passed' if summary['allHardChecksPassed'] else 'failed'}; {summary['validationCheckCount']} checks across {summary['goalRecordCount']} goal records.",
        f"- Caveats or blockers: {summary['warningFailureCount']} warning checks; some goals are emergent metric tendencies or report-facing proxy objectives rather than explicit local policy objectives.",
        "- Recommended next action: build the unified behavior dataset in S04 after Chief Scientist review; do not start S04 from this run.",
        "",
        "## Coverage By Source Experiment",
        "",
        markdown_table(source_rows, ["sourceExperimentId", "records"]),
        "",
        "## Coverage By Goal Family",
        "",
        markdown_table(family_rows, ["goalFamily", "records"]),
        "",
        "## Coverage By Goal Kind",
        "",
        markdown_table(kind_rows, ["goalKind", "records"]),
        "",
        "## Coverage By Representation",
        "",
        markdown_table(representation_rows, ["representationType", "records"]),
        "",
        "## Policy Link Modes",
        "",
        markdown_table(policy_rows, ["policyLinkMode", "records"]),
        "",
        "## Failed Checks",
        "",
        dataframe_markdown(failed) if not failed.empty else "No failed validation checks.",
        "",
        "## Claim Boundary",
        "",
        GOAL_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, status_payload: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    lines = [
        f"# {STEP_ID} Status Summary",
        "",
        f"- Step ID: {STEP_ID}",
        f"- Completion status: {status_payload['status']}",
        f"- Artifacts written: {len(status_payload['artifactsWritten'])} files; primary outputs are `goal_schema.json`, `goal_catalog.*`, `goal_world_links.*`, and `validation_report.md`.",
        f"- Validation result: {status_payload['validationResult']}",
        f"- Outcome classification: {status_payload['outcomeClassification']}",
        f"- Caveats or blockers: {status_payload['caveatsOrBlockers']}",
        "- Lay summary: E07 now has a common goal table linking monotonic order, target morphology, repair, symmetry, aggregation, homeostasis, and chimeric conflict goals to the S01/S02 substrates.",
        f"- Recommended next action: {status_payload['recommendedNextAction']}",
        "",
        f"Catalog coverage: {summary['goalRecordCount']} goal records linking {summary['linkedWorldCount']} S01 worlds and {summary['linkedPolicyCount']} S02 policies or source-family policy candidates.",
        "",
        GOAL_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--previous-artifacts-dir", type=Path, default=DEFAULT_PREVIOUS_ARTIFACTS_DIR)
    parser.add_argument("--world-catalog-path", type=Path, default=DEFAULT_WORLD_CATALOG_PATH)
    parser.add_argument("--policy-catalog-path", type=Path, default=DEFAULT_POLICY_CATALOG_PATH)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    data_dir = args.artifacts_dir / "data"
    provenance_dir = args.artifacts_dir / "provenance"
    step_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    records = build_goal_catalog(args.previous_artifacts_dir, args.world_catalog_path, args.policy_catalog_path)
    catalog = catalog_to_dataframe(records)
    goal_links = goal_world_links_dataframe(records, args.world_catalog_path)
    validation = validate_goal_catalog(records, args.world_catalog_path, args.policy_catalog_path)
    summary = coverage_summary(records, validation)

    schema_path = step_dir / "goal_schema.json"
    write_json(schema_path, goal_schema_document())
    catalog_paths = write_dataframe(catalog, step_dir / "goal_catalog")
    catalog_jsonl_path = write_jsonl(records, step_dir / "goal_catalog.jsonl")
    data_catalog_paths = write_dataframe(catalog, data_dir / "e07_goal_catalog")
    link_paths = write_dataframe(goal_links, step_dir / "goal_world_links")
    validation_paths = write_dataframe(validation, step_dir / "goal_catalog_validation")
    validation_summary_path = step_dir / "goal_catalog_validation_summary.json"
    write_json(validation_summary_path, summary)
    validation_report_path = step_dir / "validation_report.md"
    write_validation_report(validation_report_path, summary, validation)

    test_command = None
    if not args.skip_tests:
        test_command = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS], cwd=REPO_ROOT)

    validation_ok = bool(summary["allHardChecksPassed"])
    tests_ok = True if test_command is None else bool(test_command["success"])
    success = validation_ok and tests_ok
    validation_result = (
        f"passed: {summary['goalRecordCount']} goal records, {summary['validationCheckCount']} validation checks, "
        f"{summary['hardValidationFailureCount']} hard failures; focused tests {'skipped' if test_command is None else ('passed' if tests_ok else 'failed')}"
    )
    if not success:
        validation_result = (
            f"failed: {summary['goalRecordCount']} goal records, {summary['validationCheckCount']} validation checks, "
            f"{summary['hardValidationFailureCount']} hard failures; focused tests {'skipped' if test_command is None else ('passed' if tests_ok else 'failed')}"
        )

    status_path = step_dir / "status.json"
    summary_path = step_dir / "summary.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"
    caveats = (
        "Some goals are emergent tendencies, control distributions, benchmark objectives, or report-facing proxy scores "
        "rather than explicit objectives optimized by every local policy; target morphology and chimeric-control labels "
        "remain computational analogies only."
    )
    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": [],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Proceed to S04 to build the unified behavior dataset after Chief Scientist review; do not start S04 from this run.",
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "goalRecordCount": summary["goalRecordCount"],
        "linkedWorldCount": summary["linkedWorldCount"],
        "linkedPolicyCount": summary["linkedPolicyCount"],
        "validationSummary": summary,
        "focusedTestCommand": test_command,
        "claimBoundary": GOAL_CLAIM_BOUNDARY,
    }

    artifact_paths = [
        schema_path,
        *catalog_paths,
        catalog_jsonl_path,
        *data_catalog_paths,
        *link_paths,
        *validation_paths,
        validation_summary_path,
        validation_report_path,
        summary_path,
        status_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    status_payload["artifactsWritten"] = collect_artifacts([path for path in artifact_paths if path not in {summary_path, status_path, artifact_manifest_path, run_manifest_path}])
    write_summary(summary_path, status_payload, summary)
    write_json(status_path, status_payload)
    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": GOAL_SCHEMA_VERSION,
        "catalogVersion": GOAL_CATALOG_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "status": status_payload["status"],
        "artifactsWritten": collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path]),
        "validationResult": status_payload["validationResult"],
        "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "claimBoundary": GOAL_CLAIM_BOUNDARY,
    }
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    write_summary(summary_path, status_payload, summary)
    write_json(status_path, status_payload)
    manifest_payload["artifactsWritten"] = collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path])
    write_json(artifact_manifest_path, manifest_payload)

    run_manifest = {
        "experimentId": "E07",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "startedAtApprox": datetime.now(UTC).isoformat(),
        "elapsedSeconds": round(time.perf_counter() - started, 6),
        "command": [sys.executable, *sys.argv],
        "repo": {
            "root": str(REPO_ROOT),
            "branch": git_value(["branch", "--show-current"]),
            "commit": git_value(["rev-parse", "HEAD"]),
            "dirtyStatus": git_value(["status", "--short"]),
        },
        "inputs": {
            "worldCatalogPath": str(args.world_catalog_path),
            "policyCatalogPath": str(args.policy_catalog_path),
            "previousArtifactsDir": str(args.previous_artifacts_dir),
            "goalSchemaVersion": GOAL_SCHEMA_VERSION,
            "goalCatalogVersion": GOAL_CATALOG_VERSION,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threading": "serial metadata extraction; no GPU used",
            "pandas": pd.__version__,
        },
        "researchSteps": {STEP_ID: status_payload},
    }
    write_json(run_manifest_path, run_manifest)
    manifest_payload["artifactsWritten"] = collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path])
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    write_summary(summary_path, status_payload, summary)
    write_json(status_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
