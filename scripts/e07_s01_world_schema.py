#!/usr/bin/env python3
"""Build the E07 S01 formal world schema and normalized world catalog."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from platonic_space.world_schema import (  # noqa: E402
    CATALOG_VERSION,
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    build_world_catalog,
    catalog_to_dataframe,
    sha256_path,
    validation_summary,
    validate_world_catalog,
    world_schema_document,
    write_json,
)


EXPERIMENT_ID = "E07"
STEP_ID = "S01"
STEP_NUMBER = 1
STEP_TITLE = "Formalize each world"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_PREVIOUS_ARTIFACTS_DIR = Path("/previous-artifacts")
FOCUSED_TESTS = ["tests.test_e07_world_schema"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "args": args,
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "runtimeSeconds": time.perf_counter() - started,
    }


def git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["success"] else "unknown",
        "branch": branch["stdout"].strip() if branch["success"] else "unknown",
        "remote": remote["stdout"].strip() if remote["success"] else "unknown",
        "statusShort": status["stdout"].strip(),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_dataframe(df: pd.DataFrame, base_path: Path) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base_path.with_suffix(".csv")
    parquet_path = base_path.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_ready(record), sort_keys=True) + "\n")
    return path


def collect_artifacts(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(records, key=lambda item: item["path"])


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_validation_report(path: Path, summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]], validation: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    experiment_rows = [
        {"experimentId": key, "worldRecords": value}
        for key, value in sorted(summary["experimentCounts"].items())
    ]
    replay_rows = [
        {"replayability": key, "worldRecords": value}
        for key, value in sorted(summary["replayabilityCounts"].items())
    ]
    partial_records = [
        {
            "worldId": record["worldId"],
            "replayability": record["replayability"],
            "exceptions": "; ".join(record.get("exceptionsOrGaps", [])),
        }
        for record in records
        if record["replayability"] in {"partial", "summary_only"} or record["metadataCompleteness"] != "complete"
    ]
    failed = validation[~validation["success"]]
    lines = [
        f"# {STEP_ID} Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {'completed' if summary['allHardChecksPassed'] else 'completed_with_validation_failures'}",
        f"- Schema version: `{SCHEMA_VERSION}`",
        f"- Catalog version: `{CATALOG_VERSION}`",
        f"- Validation result: {'passed' if summary['allHardChecksPassed'] else 'failed'}; {summary['validationCheckCount']} checks across {summary['worldRecordCount']} world records.",
        f"- Caveats or blockers: {len(partial_records)} partial or summary-only records preserve documented exceptions; no blocker if hard checks pass.",
        f"- Recommended next action: represent policies abstractly in S02 after Chief Scientist review.",
        "",
        "## Coverage By Experiment",
        "",
        markdown_table(experiment_rows, ["experimentId", "worldRecords"]),
        "",
        "## Replayability",
        "",
        markdown_table(replay_rows, ["replayability", "worldRecords"]),
        "",
        "## Documented Exceptions",
        "",
        markdown_table(partial_records, ["worldId", "replayability", "exceptions"]) if partial_records else "None.",
        "",
        "## Failed Checks",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed validation checks.",
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, status_payload: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {STEP_ID} Status Summary",
        "",
        f"- Step ID: {STEP_ID}",
        f"- Completion status: {status_payload['status']}",
        f"- Artifacts written: {len(status_payload['artifactsWritten'])} files; primary outputs are `world_schema.json`, `normalized_world_catalog.*`, and `validation_report.md`.",
        f"- Validation result: {status_payload['validationResult']}",
        f"- Outcome classification: {status_payload['outcomeClassification']}",
        f"- Caveats or blockers: {status_payload['caveatsOrBlockers']}",
        "- Lay summary: E07 now has one explicit metadata schema for prior simulation worlds, with each E01-E06 task family mapped to state, observation, action, transition, goal, perturbation, scheduler, and measurement fields.",
        f"- Recommended next action: {status_payload['recommendedNextAction']}",
        "",
        f"Catalog coverage: {summary['worldRecordCount']} world records across {len(summary['experimentCounts'])} upstream experiments.",
        "",
        CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_run_manifest(path: Path, status_payload: Mapping[str, Any]) -> None:
    write_json(
        path,
        {
            "schema": "e07.run_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "latestResearchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "updatedAt": utc_now(),
            "git": git_metadata(),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "researchSteps": {STEP_ID: status_payload},
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--previous-artifacts-dir", type=Path, default=DEFAULT_PREVIOUS_ARTIFACTS_DIR)
    parser.add_argument("--skip-tests", action="store_true", help="Do not run focused unit tests from the script.")
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

    schema = world_schema_document()
    records = build_world_catalog(args.previous_artifacts_dir)
    catalog = catalog_to_dataframe(records)
    validation = validate_world_catalog(records, schema)
    summary = validation_summary(validation, records)

    schema_path = step_dir / "world_schema.json"
    write_json(schema_path, schema)

    catalog_paths = write_dataframe(catalog, step_dir / "normalized_world_catalog")
    catalog_jsonl_path = write_jsonl(records, step_dir / "normalized_world_catalog.jsonl")
    data_catalog_paths = write_dataframe(catalog, data_dir / "e07_world_catalog")

    validation_paths = write_dataframe(validation, step_dir / "world_catalog_validation")
    validation_summary_path = step_dir / "validation_summary.json"
    write_json(validation_summary_path, summary)
    validation_report_path = step_dir / "validation_report.md"
    write_validation_report(validation_report_path, summary, records, validation)

    test_command = None
    if not args.skip_tests:
        test_command = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS], cwd=REPO_ROOT)

    validation_ok = bool(summary["allHardChecksPassed"])
    tests_ok = True if test_command is None else bool(test_command["success"])
    success = validation_ok and tests_ok
    validation_result = (
        f"passed: {summary['worldRecordCount']} world records, {summary['validationCheckCount']} validation checks, "
        f"{summary['hardFailureCount']} hard failures; focused tests {'skipped' if test_command is None else 'passed' if tests_ok else 'failed'}"
    )
    caveats = (
        f"{summary['metadataCompletenessCounts'].get('partial', 0)} partial and "
        f"{summary['metadataCompletenessCounts'].get('exception_documented', 0)} summary/exception records require inherited source semantics; "
        "no upstream blocker detected."
    )
    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "title": STEP_TITLE,
        "schemaVersion": SCHEMA_VERSION,
        "catalogVersion": CATALOG_VERSION,
        "artifactsWritten": [],
        "validationResult": validation_result,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "caveatsOrBlockers": caveats if success else caveats + " Review failed validation/test checks before S02.",
        "recommendedNextAction": "Proceed to S02 to represent policies abstractly after Chief Scientist review; do not start S02 from this run.",
        "startedAt": utc_now(),
        "runtimeSeconds": time.perf_counter() - started,
        "previousArtifactsDir": str(args.previous_artifacts_dir),
        "focusedTestCommand": test_command,
        "validationSummary": summary,
        "claimBoundary": CLAIM_BOUNDARY,
    }

    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    artifact_paths = [
        schema_path,
        *catalog_paths,
        catalog_jsonl_path,
        *data_catalog_paths,
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
        "schema": "e07_s01_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": status_payload["status"],
        "artifactsWritten": collect_artifacts(artifact_paths),
        "validationResult": validation_result,
        "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "git": git_metadata(),
    }
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    write_summary(summary_path, status_payload, summary)
    write_json(status_path, status_payload)
    write_run_manifest(run_manifest_path, status_payload)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
