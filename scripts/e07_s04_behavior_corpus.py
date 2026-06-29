#!/usr/bin/env python3
"""Build the E07 S04 unified behavior corpus."""

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
sys.dont_write_bytecode = True

import pandas as pd

from platonic_space.behavior_corpus import (
    BEHAVIOR_CLAIM_BOUNDARY,
    BEHAVIOR_CORPUS_VERSION,
    BEHAVIOR_SCHEMA_VERSION,
    build_behavior_corpus,
    compact_json,
    corpus_dictionary_document,
    corpus_to_dataframe,
    coverage_summary,
    missingness_report,
    validate_behavior_corpus,
)
from platonic_space.behavior_corpus import artifact_index_to_dataframe
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S04"
STEP_NUMBER = 4
STEP_TITLE = "Build a unified dataset"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_PREVIOUS_ARTIFACTS_DIR = Path("/previous-artifacts")
DEFAULT_WORLD_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S01" / "normalized_world_catalog.parquet"
DEFAULT_POLICY_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S02" / "policy_abstract_catalog.parquet"
DEFAULT_GOAL_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S03" / "goal_catalog.parquet"
FOCUSED_TESTS = ["tests.test_e07_behavior_corpus"]


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


def write_dataframe(df: pd.DataFrame, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    parquet_path = stem.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


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


def dataframe_markdown(df: pd.DataFrame, limit: int = 50) -> str:
    if df.empty:
        return "No rows."
    display = df.head(limit).copy()
    return markdown_table(display.astype(str).to_dict(orient="records"), [str(column) for column in display.columns])


def write_dictionary_markdown(path: Path, dictionary: Mapping[str, Any], status_payload: Mapping[str, Any]) -> None:
    field_rows = dictionary["fields"]
    lines = [
        f"# {STEP_ID} Corpus Dictionary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status_payload['status']}",
        f"- Artifacts written: {len(status_payload['artifactsWritten'])} files; primary outputs include `unified_behavior_corpus.parquet`, `corpus_dictionary.json`, source artifact index, validation, and missingness reports.",
        f"- Validation result: {status_payload['validationResult']}",
        f"- Caveats or blockers: {status_payload['caveatsOrBlockers']}",
        f"- Recommended next action: {status_payload['recommendedNextAction']}",
        "",
        "## Fields",
        "",
        markdown_table(field_rows, ["name", "description"]),
        "",
        "## Claim Boundary",
        "",
        BEHAVIOR_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_report(path: Path, summary: Mapping[str, Any], validation: pd.DataFrame, status_payload: Mapping[str, Any]) -> None:
    failed = validation[~validation["success"]]
    source_rows = [
        {"sourceExperimentId": key, "records": value} for key, value in sorted(summary["sourceExperimentCounts"].items())
    ]
    policy_rows = [
        {"policyResolutionStatus": key, "records": value}
        for key, value in sorted(summary["policyResolutionStatusCounts"].items())
    ]
    lines = [
        f"# {STEP_ID} Unified Behavior Corpus Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status_payload['status']}",
        "- Artifacts written: `unified_behavior_corpus.{parquet,jsonl}`, `corpus_dictionary.{json,md}`, `source_artifact_index.{csv,parquet,json}`, `corpus_validation.{csv,parquet}`, `missingness_report.{csv,parquet,md}`, `validation_report.md`, `summary.md`, `status.json`, `artifact_manifest.json`, and `$ARTIFACTS_DIR/data/e07_unified_behavior_corpus.parquet`.",
        f"- Validation result: {status_payload['validationResult']}",
        f"- Caveats or blockers: {status_payload['caveatsOrBlockers']}",
        f"- Recommended next action: {status_payload['recommendedNextAction']}",
        "",
        "## Row Counts By Upstream Experiment",
        "",
        markdown_table(source_rows, ["sourceExperimentId", "records"]),
        "",
        "## Policy Resolution",
        "",
        markdown_table(policy_rows, ["policyResolutionStatus", "records"]),
        "",
        "## Source Table Coverage",
        "",
        f"- Available source tables: {summary['availableSourceTableCount']}",
        f"- Integrated source tables: {summary['integratedSourceTableCount']}",
        f"- Total behavior records: {summary['recordCount']}",
        "",
        "## Failed Or Warning Checks",
        "",
        dataframe_markdown(failed, limit=80) if not failed.empty else "No failed validation checks.",
        "",
        "## Claim Boundary",
        "",
        BEHAVIOR_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_missingness_markdown(path: Path, missingness: pd.DataFrame, status_payload: Mapping[str, Any]) -> None:
    lines = [
        f"# {STEP_ID} Missingness Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status_payload['status']}",
        "- Artifacts written: `missingness_report.{csv,parquet,md}` plus the S04 corpus, dictionary, artifact index, validation report, summary, and status artifacts.",
        f"- Validation result: {status_payload['validationResult']}",
        f"- Caveats or blockers: {status_payload['caveatsOrBlockers']}",
        f"- Recommended next action: {status_payload['recommendedNextAction']}",
        "",
        "## Missingness By Field",
        "",
        dataframe_markdown(missingness, limit=40),
        "",
        "Rows without exact row-level policy IDs are retained with `candidate_from_world_or_goal` status where possible, because several upstream summary/statistical tables aggregate over policies or report family-level effects.",
        "",
        "## Claim Boundary",
        "",
        BEHAVIOR_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, status_payload: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    lines = [
        f"# {STEP_ID} Status Summary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status_payload['status']}",
        f"- Artifacts written: {len(status_payload['artifactsWritten'])} files; primary outputs are `unified_behavior_corpus.parquet`, `corpus_dictionary.*`, `source_artifact_index.*`, `corpus_validation.*`, and `missingness_report.*`.",
        f"- Validation result: {status_payload['validationResult']}",
        f"- Outcome classification: {status_payload['outcomeClassification']}",
        f"- Caveats or blockers: {status_payload['caveatsOrBlockers']}",
        f"- Lay summary: E07 now has one indexed behavior corpus linking E01-E06 metric and trajectory-summary rows to the S01 world catalog, S02 policy catalog, and S03 goal catalog.",
        f"- Recommended next action: {status_payload['recommendedNextAction']}",
        "",
        f"Corpus coverage: {summary['recordCount']} behavior records from {summary['integratedSourceTableCount']} upstream result tables across {len(summary['sourceExperimentCounts'])} experiments, with {summary['exactPolicyLinkedRowCount']} rows carrying exact row-level S02 policy links.",
        "",
        BEHAVIOR_CLAIM_BOUNDARY,
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
    parser.add_argument("--goal-catalog-path", type=Path, default=DEFAULT_GOAL_CATALOG_PATH)
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

    records, artifact_index = build_behavior_corpus(
        previous_artifacts_dir=args.previous_artifacts_dir,
        world_catalog_path=args.world_catalog_path,
        policy_catalog_path=args.policy_catalog_path,
        goal_catalog_path=args.goal_catalog_path,
    )
    corpus = corpus_to_dataframe(records)
    artifact_index_out = artifact_index_to_dataframe(artifact_index)
    validation = validate_behavior_corpus(
        records,
        artifact_index,
        world_catalog_path=args.world_catalog_path,
        policy_catalog_path=args.policy_catalog_path,
        goal_catalog_path=args.goal_catalog_path,
    )
    missingness = missingness_report(records)
    if "sourceExperimentBreakdownJson" in missingness:
        missingness = missingness.copy()
        missingness["sourceExperimentBreakdownJson"] = missingness["sourceExperimentBreakdownJson"].map(compact_json)
    summary = coverage_summary(records, validation, artifact_index)

    corpus_parquet_path = write_parquet(corpus, step_dir / "unified_behavior_corpus.parquet")
    corpus_jsonl_path = write_jsonl(records, step_dir / "unified_behavior_corpus.jsonl")
    data_corpus_parquet_path = write_parquet(corpus, data_dir / "e07_unified_behavior_corpus.parquet")
    artifact_index_paths = write_dataframe(artifact_index_out, step_dir / "source_artifact_index")
    artifact_index_json_path = step_dir / "source_artifact_index.json"
    write_json(artifact_index_json_path, {"researchStepId": STEP_ID, "stepNumber": STEP_NUMBER, "rows": artifact_index.to_dict(orient="records")})
    validation_paths = write_dataframe(validation, step_dir / "corpus_validation")
    validation_summary_path = step_dir / "corpus_validation_summary.json"
    write_json(validation_summary_path, summary)
    missingness_paths = write_dataframe(missingness, step_dir / "missingness_report")
    dictionary = corpus_dictionary_document()
    dictionary_json_path = step_dir / "corpus_dictionary.json"
    write_json(dictionary_json_path, dictionary)

    test_command = None
    if not args.skip_tests:
        test_command = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS], cwd=REPO_ROOT)

    validation_ok = bool(summary["allHardChecksPassed"])
    tests_ok = True if test_command is None else bool(test_command["success"])
    success = validation_ok and tests_ok
    validation_result = (
        f"passed: {summary['recordCount']} behavior records, {summary['validationCheckCount']} validation checks, "
        f"{summary['hardValidationFailureCount']} hard failures; focused tests "
        f"{'skipped' if test_command is None else 'passed' if tests_ok else 'failed'}"
    )
    if not success:
        validation_result = (
            f"failed: {summary['recordCount']} behavior records, {summary['validationCheckCount']} validation checks, "
            f"{summary['hardValidationFailureCount']} hard failures; focused tests "
            f"{'skipped' if test_command is None else 'passed' if tests_ok else 'failed'}"
        )

    caveats = (
        "Trace formats and metric schemas differ across E01-E06, so S04 stores normalized JSON metric and behavior-vector "
        "summaries and links raw traces rather than duplicating them. Some summary/statistical rows have candidate policy "
        "links instead of exact row-level S02 policy IDs."
    )
    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": "completed" if success else "completed_with_validation_failures",
        "title": STEP_TITLE,
        "behaviorSchemaVersion": BEHAVIOR_SCHEMA_VERSION,
        "behaviorCorpusVersion": BEHAVIOR_CORPUS_VERSION,
        "artifactsWritten": [],
        "validationResult": validation_result,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "caveatsOrBlockers": caveats if success else caveats + " Review failed validation/test checks before S05.",
        "recommendedNextAction": "Stop for Chief Scientist review before S05; if accepted, proceed to S05 behavior predictor training using the S04 corpus.",
        "recordCount": summary["recordCount"],
        "sourceExperimentCounts": summary["sourceExperimentCounts"],
        "sourceTableCount": summary["sourceTableCount"],
        "integratedSourceTableCount": summary["integratedSourceTableCount"],
        "policyResolutionStatusCounts": summary["policyResolutionStatusCounts"],
        "exactPolicyLinkedRowCount": summary["exactPolicyLinkedRowCount"],
        "validationSummary": summary,
        "focusedTestCommand": test_command,
        "claimBoundary": BEHAVIOR_CLAIM_BOUNDARY,
    }

    validation_report_path = step_dir / "validation_report.md"
    missingness_md_path = step_dir / "missingness_report.md"
    dictionary_md_path = step_dir / "corpus_dictionary.md"
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    artifact_paths = [
        corpus_parquet_path,
        corpus_jsonl_path,
        data_corpus_parquet_path,
        *artifact_index_paths,
        artifact_index_json_path,
        *validation_paths,
        validation_summary_path,
        *missingness_paths,
        missingness_md_path,
        dictionary_json_path,
        dictionary_md_path,
        validation_report_path,
        summary_path,
        status_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    status_payload["artifactsWritten"] = collect_artifacts(
        [path for path in artifact_paths if path not in {summary_path, status_path, artifact_manifest_path, run_manifest_path}]
    )
    write_dictionary_markdown(dictionary_md_path, dictionary, status_payload)
    write_validation_report(validation_report_path, summary, validation, status_payload)
    write_missingness_markdown(missingness_md_path, missingness, status_payload)
    write_summary(summary_path, status_payload, summary)
    write_json(status_path, status_payload)

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": BEHAVIOR_SCHEMA_VERSION,
        "corpusVersion": BEHAVIOR_CORPUS_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "status": status_payload["status"],
        "artifactsWritten": collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path]),
        "validationResult": status_payload["validationResult"],
        "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "claimBoundary": BEHAVIOR_CLAIM_BOUNDARY,
    }
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    write_dictionary_markdown(dictionary_md_path, dictionary, status_payload)
    write_validation_report(validation_report_path, summary, validation, status_payload)
    write_missingness_markdown(missingness_md_path, missingness, status_payload)
    write_summary(summary_path, status_payload, summary)
    write_json(status_path, status_payload)

    run_manifest = {
        "experimentId": EXPERIMENT_ID,
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
            "goalCatalogPath": str(args.goal_catalog_path),
            "previousArtifactsDir": str(args.previous_artifacts_dir),
            "behaviorSchemaVersion": BEHAVIOR_SCHEMA_VERSION,
            "behaviorCorpusVersion": BEHAVIOR_CORPUS_VERSION,
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
    write_dictionary_markdown(dictionary_md_path, dictionary, status_payload)
    write_validation_report(validation_report_path, summary, validation, status_payload)
    write_missingness_markdown(missingness_md_path, missingness, status_payload)
    write_summary(summary_path, status_payload, summary)
    write_json(status_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
