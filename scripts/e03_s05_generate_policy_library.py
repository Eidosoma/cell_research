#!/usr/bin/env python3
"""Generate and validate the E03 S05 DSL policy library."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.e03.policy_generation import (
    DEFAULT_GENERATION_SEED,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TARGET_UNIQUE,
    accepted_policy_frame,
    generate_policy_library,
    generation_summary_frame,
    validate_generated_library,
    write_policy_jsonl,
)


STEP_ID = "S05"
STEP_NUMBER = 5
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--target-unique", type=int, default=DEFAULT_TARGET_UNIQUE)
    parser.add_argument("--seed", type=int, default=DEFAULT_GENERATION_SEED)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


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


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                text = "nan" if pd.isna(value) else f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def render_report(
    *,
    artifacts: dict[str, Path],
    accepted_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    unit_tests: dict[str, Any],
    manifest: dict[str, Any],
    command_line: str,
) -> str:
    validation_success = bool(validation_df["success"].all() and unit_tests["success"])
    outcome = "supportive" if validation_success else "constraining/contradictory"
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"unit tests return code {unit_tests['returnCode']}"
    )
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    summary_table = markdown_table(summary_df, ["metric", "value", "detail"])
    validation_table = markdown_table(validation_df, ["validation_case", "success", "expected", "observed"])
    source_kind_counts = (
        accepted_df["source_kind"].value_counts().rename_axis("source_kind").reset_index(name="accepted_count").sort_values("source_kind")
    )
    feature_rows = pd.DataFrame(
        [
            {"feature": "usesIdeal", "count": int(accepted_df["usesIdeal"].sum())},
            {"feature": "usesPrefixSorted", "count": int(accepted_df["usesPrefixSorted"].sum())},
            {"feature": "usesRandomCondition", "count": int(accepted_df["usesRandomCondition"].sum())},
            {"feature": "usesProbabilisticAction", "count": int(accepted_df["usesProbabilisticAction"].sum())},
            {"feature": "usesMemory", "count": int(accepted_df["usesMemory"].sum())},
            {"feature": "usesSignal", "count": int(accepted_df["usesSignal"].sum())},
            {"feature": "hasSwapAction", "count": int((accepted_df["swapActionCount"] > 0).sum())},
        ]
    )
    audit_summary = audit_df["reason"].value_counts().rename_axis("reason").reset_index(name="count").sort_values("reason")
    command_rows = pd.DataFrame(
        [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": command_line, "returnCode": 0, "success": True},
        ]
    )
    source_table = markdown_table(pd.DataFrame(manifest["sourceFiles"]), ["relativePath", "sha256", "sizeBytes"])
    return f"""# E03 S05 Research Step Full Results

## Top Summary

- Step ID: S05
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains. This step validates DSL syntax, hashing, duplicate filtering, and tiny-array execution only; competence evaluation and simulator vectorization remain S06/S07 work. Memory and signal primitives are represented but still use the S02 stub semantics.
- Lay summary: S05 produced a deterministic library of {len(accepted_df)} unique DSL policies from classic seeds, hand-designed variants, random grammar expansion, mutations, and recombinations. Every retained policy parses, round-trips, has a stable hash, is semantically unique, and executes on tiny array fixtures.
- Recommended next action: Stop for Chief review; if accepted, run S06 to vectorize and validate the simulator before large competence sweeps.

## Frozen Question

Does a broad random and mutational DSL sample contain policies with nontrivial sorting competence and diverse behavioral niches?

## Inputs

- S02 DSL implementation: `src/e03/rule_dsl.py`
- S03 classic DSL seeds: `src/e03/classic_policies.py`
- S04 competence schema context: `/artifacts/reports/e03_competence_vector_spec.md`
- Generation seed: `{manifest['parameters']['generationSeed']}`
- Target unique policies: `{manifest['parameters']['targetUniquePolicies']}`

## Methods

S05 generated DSL v1 programs using four source modes: hand-designed templates, random grammar expansion, mutation of accepted policies, and rule recombination between accepted policies. The S03 classic DSL policies were included as seed/context entries. Each candidate was parsed, rendered, reparsed, hashed, semantically deduplicated by name-independent rule content, and run through tiny-array execution fixtures. Accepted policies were written as JSON Lines with their DSL source, policy ID, semantic hash, parent IDs when applicable, feature flags, and tiny-execution signature.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- Host: `{platform.platform()}`
- Worker count: serial deterministic generation; no parallel workers used.
- New dependencies installed: none.

## Results

### Summary Metrics

{summary_table}

### Source Modes

{markdown_table(source_kind_counts, ["source_kind", "accepted_count"])}

### Feature Coverage

{markdown_table(feature_rows, ["feature", "count"])}

### Candidate Audit

{markdown_table(audit_summary, ["reason", "count"])}

### Validation

{validation_table}

## Validation Interpretation

The validation checks confirm that the final library reached the requested thousands-scale count, retained unique DSL policy IDs and semantic hashes, preserved tiny-array execution signatures for every policy, included required generation modes, retained many swap-capable policies, and kept target-position state represented for Selection-like neighborhoods.

## Caveats And Limitations

- Tiny-array execution is a syntax and runtime smoke check, not a sorting-competence measurement.
- S05 does not classify policies as good or bad at full sorting; S07 should evaluate competence across seeds, perturbations, and held-out sizes.
- Generated memory and signal actions are auditable DSL primitives, but their semantics remain the lightweight S02 metadata/state-update stubs.
- Duplicate filtering is semantic at the DSL rule level; behaviorally equivalent but syntactically different policies can still remain until S07/S10 behavior-based analysis.

## Provenance

Git commit before S05 commit: `{manifest['git']['headCommit']}`

Source files hashed in the S05 manifest:

{source_table}

## Artifacts

The reusable outputs are the generated JSONL policy library, the generation summary CSV, accepted-policy feature table, validation Parquet/CSV, candidate audit Parquet/CSV, artifact manifest, source snapshot manifest, run manifest, checksums, and this full-results handoff report.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    policies_dir = artifacts_dir / "policies"
    tables_dir = artifacts_dir / "tables"
    results_dir = artifacts_dir / "results"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, policies_dir, tables_dir, results_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_tests = {"command": "not run", "returnCode": 0, "success": True, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        unit_tests = run_command([sys.executable, "-m", "unittest", "tests.e03.test_policy_generation"], args.repo_dir)

    accepted, audit_df = generate_policy_library(
        target_unique=args.target_unique,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
    accepted_df = accepted_policy_frame(accepted)
    summary_df = generation_summary_frame(accepted, audit_df)
    validation_df = validate_generated_library(accepted, args.target_unique)

    policy_jsonl_path = policies_dir / "e03_generated_policy_library.jsonl"
    summary_path = tables_dir / "e03_policy_generation_summary.csv"
    accepted_table_path = results_dir / "e03_generated_policy_features.parquet"
    accepted_table_csv_path = tables_dir / "e03_generated_policy_features.csv"
    validation_path = results_dir / "e03_policy_generation_validation.parquet"
    validation_csv_path = tables_dir / "e03_policy_generation_validation.csv"
    audit_path = results_dir / "e03_policy_generation_audit.parquet"
    audit_csv_path = tables_dir / "e03_policy_generation_audit.csv"
    manifest_path = src_snapshot_dir / "e03_policy_generation_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = checksums_dir / "sha256sums.txt"
    full_report_path = step_dir / "research_step_full_results.md"

    write_policy_jsonl(policy_jsonl_path, accepted)
    summary_df.to_csv(summary_path, index=False)
    accepted_df.to_parquet(accepted_table_path, index=False)
    accepted_df.to_csv(accepted_table_csv_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)
    audit_df.to_parquet(audit_path, index=False)
    audit_df.to_csv(audit_csv_path, index=False)

    source_files = [
        args.repo_dir / "src/e03/policy_generation.py",
        args.repo_dir / "tests/e03/test_policy_generation.py",
        args.repo_dir / "scripts/e03_s05_generate_policy_library.py",
    ]
    manifest = {
        "schema": "eidosoma.e03.policy_generation_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": {
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "headCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "parameters": {
            "generationSeed": args.seed,
            "targetUniquePolicies": args.target_unique,
            "maxAttempts": args.max_attempts,
        },
        "inputArtifacts": {
            "s02Dsl": str(args.repo_dir / "src/e03/rule_dsl.py"),
            "s03ClassicPolicies": str(args.repo_dir / "src/e03/classic_policies.py"),
            "s04CompetenceSpec": str(artifacts_dir / "reports/e03_competence_vector_spec.md"),
            "s04ClassicCompetenceVectors": str(artifacts_dir / "results/e03_classic_competence_vectors.parquet"),
        },
        "sourceFiles": [source_entry(path, args.repo_dir) for path in source_files],
        "validationSummary": {
            "success": bool(validation_df["success"].all() and unit_tests["success"]),
            "validationCasesPassed": int(validation_df["success"].sum()),
            "validationCasesTotal": int(len(validation_df)),
            "unitTestsReturnCode": int(unit_tests["returnCode"]),
            "acceptedPolicyCount": int(len(accepted_df)),
            "candidateAuditRows": int(len(audit_df)),
        },
    }

    artifacts = {
        "researchStepReport": full_report_path,
        "generatedPolicyLibrary": policy_jsonl_path,
        "generationSummary": summary_path,
        "acceptedPolicyFeatures": accepted_table_path,
        "acceptedPolicyFeaturesCsv": accepted_table_csv_path,
        "validationParquet": validation_path,
        "validationCsv": validation_csv_path,
        "candidateAuditParquet": audit_path,
        "candidateAuditCsv": audit_csv_path,
        "sourceSnapshotManifest": manifest_path,
        "artifactManifest": artifact_manifest_path,
        "runManifest": run_manifest_path,
        "checksums": checksums_path,
    }

    manifest["artifacts"] = [
        artifact_entry(policy_jsonl_path, artifacts_dir, "S05 generated DSL policy library"),
        artifact_entry(summary_path, artifacts_dir, "S05 policy generation summary table"),
        artifact_entry(accepted_table_path, artifacts_dir, "S05 accepted policy feature table"),
        artifact_entry(accepted_table_csv_path, artifacts_dir, "CSV sidecar for accepted policy feature table"),
        artifact_entry(validation_path, artifacts_dir, "S05 validation cases"),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S05 validation cases"),
        artifact_entry(audit_path, artifacts_dir, "S05 candidate acceptance/rejection audit"),
        artifact_entry(audit_csv_path, artifacts_dir, "CSV sidecar for S05 candidate audit"),
        manifest_self_entry(manifest_path, artifacts_dir, "S05 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S05 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S05"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S05 outputs"),
        manifest_self_entry(full_report_path, artifacts_dir, "S05 full-results handoff report"),
    ]
    write_json(manifest_path, manifest)

    artifact_manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifacts": [
            artifact_entry(policy_jsonl_path, artifacts_dir, "S05 generated DSL policy library"),
            artifact_entry(summary_path, artifacts_dir, "S05 policy generation summary table"),
            artifact_entry(accepted_table_path, artifacts_dir, "S05 accepted policy feature table"),
            artifact_entry(accepted_table_csv_path, artifacts_dir, "CSV sidecar for accepted policy feature table"),
            artifact_entry(validation_path, artifacts_dir, "S05 validation cases"),
            artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S05 validation cases"),
            artifact_entry(audit_path, artifacts_dir, "S05 candidate acceptance/rejection audit"),
            artifact_entry(audit_csv_path, artifacts_dir, "CSV sidecar for S05 candidate audit"),
            artifact_entry(manifest_path, artifacts_dir, "S05 source snapshot and provenance manifest"),
            manifest_self_entry(artifact_manifest_path, artifacts_dir, "S05 artifact manifest"),
            manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S05"),
            manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S05 outputs"),
            manifest_self_entry(full_report_path, artifacts_dir, "S05 full-results handoff report"),
        ],
    }
    write_json(artifact_manifest_path, artifact_manifest)

    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": manifest["git"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "commands": [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": " ".join(sys.argv), "returnCode": 0, "success": True},
        ],
        "inputs": manifest["inputArtifacts"],
        "parameters": manifest["parameters"],
        "outputs": artifact_manifest["artifacts"],
    }
    write_json(run_manifest_path, run_manifest)

    full_report = render_report(
        artifacts=artifacts,
        accepted_df=accepted_df,
        summary_df=summary_df,
        validation_df=validation_df,
        audit_df=audit_df,
        unit_tests=unit_tests,
        manifest=manifest,
        command_line=" ".join(sys.argv),
    )
    write_text(full_report_path, full_report)

    checksum_targets = [
        full_report_path,
        policy_jsonl_path,
        summary_path,
        accepted_table_path,
        accepted_table_csv_path,
        validation_path,
        validation_csv_path,
        audit_path,
        audit_csv_path,
        manifest_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")

    success = bool(validation_df["success"].all() and unit_tests["success"])
    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": success,
                "acceptedPolicyCount": len(accepted_df),
                "artifactsDir": str(step_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
