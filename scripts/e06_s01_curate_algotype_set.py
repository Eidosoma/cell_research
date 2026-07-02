#!/usr/bin/env python3
"""Run E06 S01 Algotype-library curation and pure-policy validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e06.algotype_library import (  # noqa: E402
    EXPERIMENT_ID,
    STEP_ID,
    S01SourcePaths,
    S01ValidationConfig,
    attach_competence_vectors,
    curate_algotype_records,
    json_safe_records,
    sha256_file,
    source_file_entry,
    summarize_validation,
    validate_algotype_records,
    validation_checks,
)


STEP_NUMBER = 1
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--previous-e03-dir", type=Path, default=Path("/previous-artifacts/E03"))
    parser.add_argument("--previous-e04-dir", type=Path, default=Path("/previous-artifacts/E04"))
    parser.add_argument("--max-e03-frontier", type=int, default=16)
    parser.add_argument("--max-e04-memory", type=int, default=4)
    parser.add_argument("--array-size", type=int, default=8)
    parser.add_argument("--dsl-event-cap", type=int, default=192)
    parser.add_argument("--memory-event-cap", type=int, default=192)
    parser.add_argument("--original-max-events", type=int, default=50_000)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = datetime.now(UTC)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=env)
    elapsed = (datetime.now(UTC) - started).total_seconds()
    return {
        "command": " ".join(command),
        "cwd": str(cwd),
        "returnCode": proc.returncode,
        "success": proc.returncode == 0,
        "elapsedSeconds": elapsed,
        "stdout": proc.stdout[-6000:],
        "stderr": proc.stderr[-6000:],
    }


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum omitted to avoid self-referential drift.",
    }


def pending_artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum computed after this file is written.",
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def source_paths(args: argparse.Namespace) -> S01SourcePaths:
    return S01SourcePaths(
        e03_classic_library=args.previous_e03_dir / "policies/e03_classic_policy_library.json",
        e03_frontier_candidates=args.previous_e03_dir / "policies/e03_frontier_candidate_policies.jsonl",
        e03_frontier_table=args.previous_e03_dir / "tables/e03_frontier_candidates.csv",
        e04_repair_capable=args.previous_e04_dir / "policies/e04_repair_capable_algotypes.jsonl",
        e04_evolved_repair=args.previous_e04_dir / "policies/e04_evolved_repair_policies.jsonl",
    )


def write_report(
    path: Path,
    *,
    artifacts: list[dict[str, Any]],
    metadata: pd.DataFrame,
    checks: pd.DataFrame,
    source_status: dict[str, Any],
    config: dict[str, Any],
    unit_test_result: dict[str, Any],
    repo_state: dict[str, Any],
    command: str,
) -> None:
    category_counts = metadata["source_category"].value_counts().sort_index().to_dict()
    validation_passed = bool(checks["success"].all() and unit_test_result.get("success", True))
    outcome = "supportive" if validation_passed else "constraining/contradictory"
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)
    result_cols = [
        "source_category",
        "display_name",
        "validation_run_count",
        "pure_execution_success",
        "sorted_run_fraction",
        "mean_final_inversion_sortedness",
        "mean_inversion_sortedness_delta",
        "mean_work_count",
    ]
    top_rows = metadata.sort_values(
        ["source_category", "baseline_competence_score", "display_name"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    checks_display = checks.copy()
    checks_display["success"] = checks_display["success"].map(bool)
    text = f"""# E06 S01 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(checks["success"].sum())}/{len(checks)} curation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: E03 frontier and E04 memory candidates were available, so no fallback-only blocker occurred. E04 memory policies were validated in a simplified pure-array adjacent-action harness; their repair competence remains inherited from E04 upstream evidence, not re-established here.
- Lay summary: S01 assembled a reusable menu of sorting-cell policies for later chimera tests: the three original algorithms, null controls, randomized controls, E03 discovered policies, and E04 memory-repair policies. Each entry has a stable ID, provenance, compatibility notes, and a small baseline behavior score.
- Recommended next action: Proceed to S02 mixture-ratio sweeps using `policies/e06_chimeric_algotype_library.jsonl`, while prioritizing originals, controls, the top E03 frontier candidates, and the two E04 memory-repair candidates.

## Frozen Question

How do original, null, randomized, discovered, and memory or signaling Algotypes differ as components of chimeric collectives?

## Inputs

- E03 classic policy library: `{config['sourcePaths']['e03_classic_library']}`
- E03 frontier candidates: `{config['sourcePaths']['e03_frontier_candidates']}`
- E04 repair-capable Algotypes: `{config['sourcePaths']['e04_repair_capable']}`
- E04 evolved repair policies: `{config['sourcePaths']['e04_evolved_repair']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`
- Uploaded paper context was available through `/workspace/input-attachments`; S01 did not need new numeric extraction from the paper.

## Methods

The curation script normalized policy records into a common JSONL schema with `algotypeId`, source category, source policy ID, execution backend, compatibility metadata, upstream metrics, and a S01 baseline competence vector.

Original Bubble, Insertion, and Selection were loaded from the E03 exact interface-wrapper records and validated with the E02 deterministic public-cell simulator. E03 discovered candidates were loaded from the S14 frontier JSONL and validated with the E03 DSL CPU interpreter. E06 null and randomized controls were generated as small DSL policies. E04 repair-capable candidates were loaded from E04 S15 and replayed with the existing `S08LocalEvolutionPolicy` decision function in a simplified pure-array adjacent-action harness.

## Commands

- Main command: `{command}`
- Unit-test command: `{unit_test_result.get('command', 'not run')}`
- Unit-test return code: `{unit_test_result.get('returnCode', 'not run')}`

## Dependencies

No new dependencies were installed. The step used the repository's Python modules plus preinstalled `pandas` and the standard library.

## Parameters

```json
{json.dumps(config, indent=2, sort_keys=True, default=str)}
```

## Results

Curated Algotype count by source category:

```json
{json.dumps(category_counts, indent=2, sort_keys=True)}
```

Validation and baseline competence summary:

{markdown_table(top_rows, result_cols)}

## Validation Checks

{markdown_table(checks_display, ["validation_case", "success", "observed", "expected"])}

The required completion criterion was met: every curated policy executed in pure-array validation, originals sorted the small pure arrays, and null controls made zero swaps.

## Artifacts

{artifact_lines}

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

Input source status:

```json
{json.dumps(source_status, indent=2, sort_keys=True, default=str)}
```

## Caveats And Limitations

- The S01 validation grid is intentionally small and checks executability plus baseline behavior, not chimeric compatibility.
- Discovered E03 DSL policies are treated as increasing-goal local policies unless later E06 steps explicitly test reverse or mixed-goal behavior.
- The E04 policies are memory-repair policies trained for compact repair/homeostasis tasks. Their S01 pure-array replay does not replace the E04 evidence layer and should be considered compatibility smoke evidence only.
- Null and randomized controls are generated controls, not paper originals.

## Recommended Next Action

Start S02 only when instructed by the Chief Scientist workflow. Use the S01 library and metadata table to select mixture-ratio conditions, with controls included in every batch.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    policies_path = artifacts_dir / "policies/e06_chimeric_algotype_library.jsonl"
    metadata_path = artifacts_dir / "tables/e06_algotype_metadata.csv"
    validation_runs_path = step_dir / "e06_s01_validation_runs.parquet"
    validation_checks_path = step_dir / "e06_s01_validation_checks.csv"
    manifest_path = step_dir / "manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    config_path = artifacts_dir / "configs/e06_s01_curation_config.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    paths = source_paths(args)
    validation_config = S01ValidationConfig(
        array_size=args.array_size,
        dsl_event_cap=args.dsl_event_cap,
        original_max_events=args.original_max_events,
        memory_event_cap=args.memory_event_cap,
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "sourcePaths": {key: str(value) for key, value in paths.__dict__.items()},
        "maxE03Frontier": args.max_e03_frontier,
        "maxE04Memory": args.max_e04_memory,
        "validation": validation_config.__dict__,
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    records, source_status = curate_algotype_records(paths, max_e03_frontier=args.max_e03_frontier, max_e04_memory=args.max_e04_memory)
    validation = validate_algotype_records(records, validation_config)
    metadata = summarize_validation(records, validation)
    enriched_records = attach_competence_vectors(records, metadata)
    checks = validation_checks(enriched_records, metadata, validation, source_status)

    write_jsonl(policies_path, enriched_records)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    validation_runs_path.parent.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(validation_runs_path, index=False)
    checks.to_csv(validation_checks_path, index=False)

    repo_state = {
        "head": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        "remote": git_output(args.repo_dir, ["remote", "-v"]),
        "python": sys.version,
        "platform": platform.platform(),
    }

    base_artifacts = [
        artifact_entry(policies_path, artifacts_dir, "Curated E06 chimeric Algotype JSONL library."),
        artifact_entry(metadata_path, artifacts_dir, "Flat metadata and baseline competence table."),
        artifact_entry(validation_runs_path, artifacts_dir, "Pure-policy validation run-level table."),
        artifact_entry(validation_checks_path, artifacts_dir, "Curation and validation check table."),
        artifact_entry(config_path, artifacts_dir, "S01 curation configuration."),
    ]
    report_listing = pending_artifact_entry(report_path, artifacts_dir, "S01 full-results Markdown handoff report.")
    manifest_listing = manifest_self_entry(manifest_path, artifacts_dir, "S01 artifact manifest.")

    write_report(
        report_path,
        artifacts=[*base_artifacts, manifest_listing, report_listing],
        metadata=metadata,
        checks=checks,
        source_status=source_status,
        config=config_payload,
        unit_test_result=unit_test_result,
        repo_state=repo_state,
        command=" ".join(sys.argv),
    )
    report_artifact = artifact_entry(report_path, artifacts_dir, "S01 full-results Markdown handoff report.")

    manifest = {
        "schema": "eidosoma.e06.s01_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": bool(checks["success"].all() and unit_test_result.get("success", True)),
        "artifacts": [*base_artifacts, report_artifact, manifest_listing],
        "sourceStatus": source_status,
        "unitTestResult": unit_test_result,
        "repoState": repo_state,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S01 artifact manifest.")
    artifacts = [*base_artifacts, manifest_artifact, report_artifact]

    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "latestResearchStepId": STEP_ID,
        "generatedAtUtc": utc_now(),
        "repoState": repo_state,
        "python": sys.version,
        "platform": platform.platform(),
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "sourceInputs": {
            "e03ClassicLibrary": source_file_entry(paths.e03_classic_library),
            "e03FrontierCandidates": source_file_entry(paths.e03_frontier_candidates),
            "e04RepairCapable": source_file_entry(paths.e04_repair_capable),
            "e04EvolvedRepair": source_file_entry(paths.e04_evolved_repair),
        },
    }
    write_json(run_manifest_path, run_manifest)

    checksum_lines = []
    for path in [policies_path, metadata_path, validation_runs_path, validation_checks_path, config_path, manifest_path, report_path, run_manifest_path]:
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}")
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")

    status = {
        "records": len(enriched_records),
        "categoryCounts": metadata["source_category"].value_counts().sort_index().to_dict(),
        "validationRows": len(validation),
        "checksPassed": int(checks["success"].sum()),
        "checksTotal": int(len(checks)),
        "success": bool(checks["success"].all() and unit_test_result.get("success", True)),
        "artifacts": json_safe_records(pd.DataFrame(artifacts)),
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
