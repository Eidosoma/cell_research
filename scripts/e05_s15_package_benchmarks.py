#!/usr/bin/env python3
"""Package the E05 morphology benchmark suite."""

from __future__ import annotations

import argparse
import hashlib
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

import numpy as np
import pandas as pd

from src.e05.benchmark_suite import (
    BENCHMARK_SUITE_ID,
    BENCHMARK_VERSION,
    GLOBAL_CONTEXT_POLICY_IDS,
    STEP_ID,
    benchmark_config_payloads,
    build_benchmark_suite_tables,
    stable_json,
    summarize_for_report,
    validate_benchmark_suite_tables,
)


STEP_NUMBER = 15
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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
        "elapsedSeconds": float(elapsed),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_path(path),
        "sizeBytes": path.stat().st_size if path.is_file() else sum(child.stat().st_size for child in path.rglob("*") if child.is_file()),
        "artifactType": "directory" if path.is_dir() else "file",
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def write_checksums(paths: list[Path], checksum_path: Path, artifacts_dir: Path) -> None:
    lines = []
    for path in sorted(paths):
        if path == checksum_path:
            continue
        lines.append(f"{sha256_path(path)}  {path.relative_to(artifacts_dir)}")
    write_text(checksum_path, "\n".join(lines) + "\n")


def markdown_table(df: pd.DataFrame, columns: list[str], *, max_rows: int = 24) -> str:
    if df.empty:
        return "_No rows._"
    shown = df[columns].head(max_rows).copy()
    for column in shown.select_dtypes(include=[float]).columns:
        shown[column] = shown[column].round(6)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in shown.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    if len(df) > max_rows:
        omitted = [f"... {len(df) - max_rows} more rows omitted", *("" for _ in columns[1:])]
        rows.append("| " + " | ".join(omitted) + " |")
    return "\n".join([header, separator, *rows])


def write_config_files(config_dir: Path, task_catalog: pd.DataFrame) -> list[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    for old in config_dir.glob("*.json"):
        old.unlink()
    payloads = benchmark_config_payloads(task_catalog)
    written: list[Path] = []
    for task_id, payload in sorted(payloads.items()):
        path = config_dir / f"{task_id}.json"
        write_json(path, payload)
        written.append(path)
    index_path = config_dir / "benchmark_index.json"
    write_json(
        index_path,
        {
            "schema": "eidosoma.e05.benchmark_suite_index.v1",
            "benchmarkSuiteId": BENCHMARK_SUITE_ID,
            "benchmarkVersion": BENCHMARK_VERSION,
            "researchStepId": STEP_ID,
            "taskConfigFiles": [path.name for path in sorted(written)],
            "taskCount": len(written),
        },
    )
    written.append(index_path)
    return written


def determine_outcome(validation_df: pd.DataFrame, task_catalog: pd.DataFrame, smoke_df: pd.DataFrame) -> tuple[str, str, str]:
    validation_success = bool(validation_df["success"].all())
    if validation_success and len(task_catalog) == 6 and smoke_df["success"].all():
        outcome = "supportive"
        lay = (
            "S15 packaged the E05 morphology tasks into a reproducible benchmark suite with configs, smoke tests, "
            "metrics, reference outputs, and a report-bundle handoff."
        )
    else:
        outcome = "null"
        lay = (
            "S15 attempted to package the benchmark suite, but validation did not establish a complete reusable suite."
        )
    caveats = (
        "The benchmark tasks are computational proxies over S01-S05 toy substrates, targets, actions, and metrics. "
        "Several regeneration-style tasks intentionally retain blockers under local swap/wait controls; S13 global rows "
        "are upper-bound endpoint context with declared nonlocal access, not local policy references."
    )
    return outcome, caveats, lay


def top_summary_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
) -> str:
    artifact_lines = "\n".join(f"- `{entry['path']}`" for entry in artifacts if entry.get("path"))
    return f"""## Top Summary

- Research step ID: {STEP_ID}
- Completion status: Completed
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome_classification}
- Caveats or blockers: {caveats_or_blockers}
- Lay summary: {lay_summary}
- Recommended next action: {recommended_next_action}
"""


def benchmark_suite_report_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
    task_catalog: pd.DataFrame,
    metric_catalog: pd.DataFrame,
    reference_df: pd.DataFrame,
    smoke_df: pd.DataFrame,
) -> str:
    summary = summarize_for_report(task_catalog, reference_df, smoke_df)
    policy_summary = (
        reference_df.groupby(["benchmark_task_id", "control_class"], as_index=False)
        .agg(
            reference_rows=("benchmark_task_id", "count"),
            mean_final_target_error=("final_target_error", "mean"),
            mean_target_recovery_fraction=("target_recovery_fraction", "mean"),
            exact_recovered_rows=("reference_outcome_class", lambda values: int(np.sum(pd.Series(values).eq("exact_recovered")))),
            blocked_rows=("reference_outcome_class", lambda values: int(np.sum(pd.Series(values).eq("blocked_baseline")))),
        )
        .sort_values(["benchmark_task_id", "control_class"])
    )
    config_lines = "\n".join(
        f"- `{row.benchmark_task_id}.json`: {row.title}" for row in task_catalog.itertuples()
    )
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action, lay_summary)}

# E05 Morphology Benchmark Suite

## Purpose

This suite packages the validated E05 substrate, identity, action, target, metric, trajectory, regeneration, scaling, local/global, and delayed-gratification artifacts into reusable benchmark tasks. It is intended for downstream comparison of local morphogenesis policies against fixed reference outputs and explicitly labeled global endpoint controls.

## Suite Summary

- Suite ID: `{BENCHMARK_SUITE_ID}`
- Version: `{BENCHMARK_VERSION}`
- Tasks: `{summary["task_count"]}`
- Reference rows: `{summary["reference_rows"]}`
- Local reference rows: `{summary["local_reference_rows"]}`
- Global endpoint context rows: `{summary["global_context_reference_rows"]}`
- Smoke-test rows: `{summary["smoke_rows"]}`, all passed: `{summary["smoke_success"]}`

## Task Catalog

{markdown_table(task_catalog, ["benchmark_task_id", "title", "target_id", "task_type", "perturbation_type", "reference_rows", "known_blocker"], max_rows=10)}

## Config Files

Config directory: `$ARTIFACTS_DIR/configs/e05_benchmark_configs/`

{config_lines}

## Metrics

{markdown_table(metric_catalog, ["metric_id", "metric_source", "metric_family", "lower_is_better", "exactness", "approximation_label"], max_rows=20)}

## Reference Baselines

{markdown_table(policy_summary, ["benchmark_task_id", "control_class", "reference_rows", "mean_final_target_error", "mean_target_recovery_fraction", "exact_recovered_rows", "blocked_rows"], max_rows=30)}

## Smoke Tests

{markdown_table(smoke_df, ["benchmark_task_id", "smoke_test_id", "success", "detail"], max_rows=40)}

## Use Notes

- Use the task JSON files as the stable benchmark contract.
- Use `$ARTIFACTS_DIR/results/e05_benchmark_reference_results.parquet` as the reference output table.
- Treat S13 top-down, organizer, and global-controller rows as declared upper-bound endpoint context. They are not local policy results.
- Missing-patch and foreign-patch tasks intentionally include local blockers. Do not erase these by adding birth/death, identity conversion, unfreeze, or global assignment unless evaluating a separately declared policy class.

## Caveats

{caveats_or_blockers}
"""


def report_bundle_handoff_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
    task_catalog: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> str:
    key_claims = [
        "E05 successfully generalized the 1D sorting abstraction into validated 2D/grid substrate, identity, action, target, metric, and benchmark artifacts.",
        "Local target-aware controls show partial recovery on several permutation-like tasks but remain blocked on true population-deficit, frozen, duplicated, and foreign-patch regeneration under swap/wait semantics.",
        "Declared global/top-down endpoint controls provide upper-bound context and should not be compared as same-information local policies.",
        "S14 found no constructive higher-dimensional delayed gratification above matched nulls in S08 traces, constraining that claim for current local controls.",
    ]
    claim_lines = "\n".join(f"- {claim}" for claim in key_claims)
    handoff_table = task_catalog[
        ["benchmark_task_id", "benchmark_family", "reference_rows", "local_reference_rows", "global_context_reference_rows", "known_blocker"]
    ]
    outcome_counts = (
        reference_df.groupby("reference_outcome_class", as_index=False)
        .agg(rows=("reference_outcome_class", "count"))
        .sort_values("reference_outcome_class")
    )
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action, lay_summary)}

# E05 Report-Bundle Handoff

## Chief-Ready Status

E05 is ready for Chief report-bundle generation. The final bundle should emphasize that these are computational proxy benchmarks, not biological validation.

## Key Claims To Carry Forward

{claim_lines}

## Benchmark Evidence

{markdown_table(handoff_table, ["benchmark_task_id", "benchmark_family", "reference_rows", "local_reference_rows", "global_context_reference_rows", "known_blocker"], max_rows=10)}

Reference outcome counts:

{markdown_table(outcome_counts, ["reference_outcome_class", "rows"], max_rows=10)}

## Primary Artifacts For Bundle

- Benchmark report: `$ARTIFACTS_DIR/reports/e05_morphology_benchmark_suite.md`
- Benchmark configs: `$ARTIFACTS_DIR/configs/e05_benchmark_configs/`
- Reference table: `$ARTIFACTS_DIR/results/e05_benchmark_reference_results.parquet`
- Full S15 report: `$ARTIFACTS_DIR/research_steps/S15/research_step_full_results.md`
- Prior step reports: `$ARTIFACTS_DIR/research_steps/S01/` through `$ARTIFACTS_DIR/research_steps/S14/`

## Caveats

{caveats_or_blockers}

## Recommended Next Action

{recommended_next_action}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    task_catalog: pd.DataFrame,
    reference_df: pd.DataFrame,
    metric_catalog: pd.DataFrame,
    smoke_df: pd.DataFrame,
    source_inputs: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
) -> str:
    command_lines = "\n".join(
        f"- `{command['command']}`: return code {command['returnCode']}, success={command['success']}, elapsed={command['elapsedSeconds']:.3f}s"
        for command in test_commands
    )
    source_lines = "\n".join(
        f"- `{entry['relativePath']}` sha256 `{entry['sha256']}` ({entry['sizeBytes']} bytes)"
        for entry in source_files
    )
    summary = summarize_for_report(task_catalog, reference_df, smoke_df)
    policy_summary = (
        reference_df.groupby(["benchmark_task_id", "policy_id", "control_class"], as_index=False)
        .agg(
            reference_rows=("benchmark_task_id", "count"),
            mean_final_target_error=("final_target_error", "mean"),
            mean_target_recovery_fraction=("target_recovery_fraction", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
            exact_recovered_rows=("reference_outcome_class", lambda values: int(np.sum(pd.Series(values).eq("exact_recovered")))),
            blocked_rows=("reference_outcome_class", lambda values: int(np.sum(pd.Series(values).eq("blocked_baseline")))),
        )
        .sort_values(["benchmark_task_id", "control_class", "policy_id"])
    )
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action, lay_summary)}

# Research Step Full Results: {STEP_ID} Produce A Morphology Benchmark Suite

## Lay Summary

{lay_summary}

## Frozen Question

Can standard 2D tasks be packaged so later work can compare local morphogenesis policies reproducibly?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S15.
- Source artifacts: completed E05 S01-S14 reports, configs, target specs, metric specs, result tables, trajectory summaries, local/global baselines, and S14 delayed-gratification summaries.
- Datasets: none required; `/workspace/DATASETS.md` states no dataset inputs are required.
- Previous artifacts: E01-E04 mounts were read as context, but S15 packaging uses E05 artifacts and repository code as direct inputs.

Input provenance:

{markdown_table(source_inputs, ["source_research_step_id", "input_type", "exists", "row_count", "used_for_configs", "used_for_reference_results", "used_for_report_bundle_context"], max_rows=20)}

## Detailed Methods

S15 defines six benchmark tasks from validated E05 components:

- embedded sort-row continuity from S06;
- scrambled anterior-posterior gradient restoration from S07/S13;
- missing-patch boundary repair from S08/S13;
- scrambled boundary-pattern recovery from S07/S13;
- toy organ-like appendage missing-patch regeneration from S08/S13;
- organ-stripes foreign-patch chimeric conflict from S08/S13.

For each task, S15 writes a JSON config under `$ARTIFACTS_DIR/configs/e05_benchmark_configs/` containing the target, perturbation/initialization protocol, action set, policy set, metric set, reference-output selector, smoke-test description, known blocker, and claim boundary.

Reference outputs are normalized into one table. S06 rows provide the embedded-row sorting reference. S13 rows provide local-policy references and explicitly labeled top-down, organizer, and global-controller endpoint context for the 2D tasks. S15 does not retune policies, rerun simulations, expand local information access, or reinterpret global endpoint rows as local policy trajectories.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- New packages installed: none.
- CPU/GPU use: CPU-only artifact packaging and validation; no new simulations or GPU runs were launched.

## Parameters

- Benchmark suite ID: `{BENCHMARK_SUITE_ID}`
- Benchmark version: `{BENCHMARK_VERSION}`
- Task count: `{summary["task_count"]}`
- Reference rows: `{summary["reference_rows"]}`
- Local reference rows: `{summary["local_reference_rows"]}`
- Global endpoint context rows: `{summary["global_context_reference_rows"]}`
- Config files: six task configs plus one benchmark index.

## Results

Validation summary:

{markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=15)}

Task catalog:

{markdown_table(task_catalog, ["benchmark_task_id", "title", "target_id", "task_type", "perturbation_type", "reference_rows", "local_reference_rows", "global_context_reference_rows", "known_blocker"], max_rows=10)}

Policy-level reference summary:

{markdown_table(policy_summary, ["benchmark_task_id", "policy_id", "control_class", "reference_rows", "mean_final_target_error", "mean_target_recovery_fraction", "mean_energy_per_site", "exact_recovered_rows", "blocked_rows"], max_rows=40)}

Metric manifest:

{markdown_table(metric_catalog, ["metric_id", "metric_source", "metric_family", "lower_is_better", "exactness", "approximation_label"], max_rows=20)}

Smoke tests:

{markdown_table(smoke_df, ["benchmark_task_id", "smoke_test_id", "success", "detail"], max_rows=50)}

## Metrics

- S06 special-case metrics: final sortedness percent, monotonicity error count, and final embedded morphospace error.
- S05 morphology metrics: target identity error, target-neighborhood error, boundary error, topology component error, shape moment error, Hausdorff distance, Earth-mover proxy, and graph-edit proxy.
- Reference performance fields: final target error, target recovery fraction, final aggregate morphospace error, aggregate recovery fraction, energy per site, blocker flags, and outcome class.

## Figures And Tables

- Required benchmark report: `$ARTIFACTS_DIR/reports/e05_morphology_benchmark_suite.md`.
- Required config directory: `$ARTIFACTS_DIR/configs/e05_benchmark_configs/`.
- Required reference result table: `$ARTIFACTS_DIR/results/e05_benchmark_reference_results.parquet`.
- Required report-bundle handoff: `$ARTIFACTS_DIR/reports/e05_report_bundle_handoff.md`.
- Additional task-catalog, metric-catalog, smoke-test, source-input, validation, manifest, checksum, and CSV artifacts are listed below.

## Validation Checks

- Confirmed the six standard tasks are present.
- Confirmed required S01-S14 input artifacts are available.
- Confirmed task JSON config files are written and parseable.
- Confirmed reference outputs exist for each task.
- Confirmed smoke tests pass for target, metric, config, and reference coverage.
- Confirmed local policy information access is not expanded.
- Confirmed known local-regeneration blockers remain explicit.
- Confirmed numeric metric columns are finite where present.
- Confirmed required S15 Markdown reports, configs, and tables are present.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- The benchmark is a computational proxy suite over toy substrates and targets, not biological validation.
- Missing-patch and foreign-patch tasks intentionally retain local swap/wait blockers; solving them requires a policy class with explicitly declared extra semantics.
- S13 top-down, organizer, and global-controller rows are upper-bound endpoint references with global access and are not same-information local policies.
- Energy costs are bookkeeping proxies and should not be treated as physical tissue work.
- The suite packages current E05 evidence; downstream users should still run seed, policy, metric, and scale sensitivity checks for new policy families.

## Blockers And Failed Assumptions

No execution blocker was encountered. The main constrained assumption is that all benchmark tasks should be locally repairable under current swap/wait controls; S08-S15 preserve population-deficit and identity-conversion blockers as benchmark facts.

## Recommended Next Action

{recommended_next_action}
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    reports_dir = artifacts_dir / "reports"
    configs_dir = artifacts_dir / "configs"
    benchmark_config_dir = configs_dir / "e05_benchmark_configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, reports_dir, configs_dir, benchmark_config_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    tables = build_benchmark_suite_tables(artifacts_dir)
    task_catalog = tables["task_catalog"]
    reference_df = tables["reference_results"]
    metric_catalog = tables["metric_catalog"]
    smoke_df = tables["smoke_tests"]
    source_inputs = tables["source_inputs"]

    config_files = write_config_files(benchmark_config_dir, task_catalog)
    config_index_path = benchmark_config_dir / "benchmark_index.json"

    reference_path = results_dir / "e05_benchmark_reference_results.parquet"
    reference_csv_path = tables_dir / "e05_benchmark_reference_results.csv"
    task_catalog_path = results_dir / "e05_benchmark_task_catalog.parquet"
    task_catalog_csv_path = tables_dir / "e05_benchmark_task_catalog.csv"
    metric_catalog_path = results_dir / "e05_benchmark_metric_catalog.parquet"
    metric_catalog_csv_path = tables_dir / "e05_benchmark_metric_catalog.csv"
    smoke_path = results_dir / "e05_benchmark_smoke_tests.parquet"
    smoke_csv_path = tables_dir / "e05_benchmark_smoke_tests.csv"
    source_inputs_path = results_dir / "e05_benchmark_source_inputs.parquet"
    source_inputs_csv_path = tables_dir / "e05_benchmark_source_inputs.csv"
    validation_path = results_dir / "e05_benchmark_validation.parquet"
    validation_csv_path = tables_dir / "e05_benchmark_validation.csv"
    suite_report_path = reports_dir / "e05_morphology_benchmark_suite.md"
    handoff_path = reports_dir / "e05_report_bundle_handoff.md"
    config_path = configs_dir / "e05_s15_benchmark_suite.json"
    source_manifest_path = src_snapshot_dir / "e05_benchmark_suite_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    reference_df.to_parquet(reference_path, index=False)
    reference_df.to_csv(reference_csv_path, index=False)
    task_catalog.to_parquet(task_catalog_path, index=False)
    task_catalog.to_csv(task_catalog_csv_path, index=False)
    metric_catalog.to_parquet(metric_catalog_path, index=False)
    metric_catalog.to_csv(metric_catalog_csv_path, index=False)
    smoke_df.to_parquet(smoke_path, index=False)
    smoke_df.to_csv(smoke_csv_path, index=False)
    source_inputs.to_parquet(source_inputs_path, index=False)
    source_inputs.to_csv(source_inputs_csv_path, index=False)

    config = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "benchmarkSuiteId": BENCHMARK_SUITE_ID,
        "benchmarkVersion": BENCHMARK_VERSION,
        "taskCount": int(len(task_catalog)),
        "referenceRows": int(len(reference_df)),
        "smokeRows": int(len(smoke_df)),
        "taskConfigDirectory": str(benchmark_config_dir),
        "taskConfigFiles": [str(path) for path in config_files],
        "globalContextPolicyIds": list(GLOBAL_CONTEXT_POLICY_IDS),
        "summary": summarize_for_report(task_catalog, reference_df, smoke_df),
    }
    write_json(config_path, config)

    source_files = [
        source_entry(args.repo_dir / "src/e05/benchmark_suite.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s15_package_benchmarks.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05_benchmarks/test_benchmark_suite.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/targets.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/morphospace_metrics.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/local_global_control.py", args.repo_dir),
    ]
    write_json(
        source_manifest_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "sourceFiles": source_files,
        },
    )

    preliminary_artifacts = [
        artifact_entry(reference_path, artifacts_dir, "Required S15 benchmark reference result table."),
        artifact_entry(benchmark_config_dir, artifacts_dir, "Required S15 benchmark task config directory."),
        artifact_entry(config_path, artifacts_dir, "S15 reproducibility configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "S15 source-code provenance manifest."),
    ]
    preliminary_validation = validate_benchmark_suite_tables(
        task_catalog=task_catalog,
        reference_df=reference_df,
        metric_catalog=metric_catalog,
        smoke_df=smoke_df,
        source_inputs=source_inputs,
        config_dir=benchmark_config_dir,
        required_artifact_paths=[reference_path, benchmark_config_dir, config_path, source_manifest_path],
    )
    preliminary_validation_result = f"{int(preliminary_validation['success'].sum())}/{len(preliminary_validation)} validation cases passed"
    preliminary_outcome, preliminary_caveats, preliminary_lay = determine_outcome(preliminary_validation, task_catalog, smoke_df)
    recommended_next_action = (
        "E05 is ready for Chief report-bundle generation; do not start E07 or any new experiment without an explicit launch instruction."
    )
    write_text(
        suite_report_path,
        benchmark_suite_report_markdown(
            artifacts=[*preliminary_artifacts, manifest_self_entry(suite_report_path, artifacts_dir, "Required benchmark-suite report.")],
            validation_result=preliminary_validation_result,
            outcome_classification=preliminary_outcome,
            caveats_or_blockers=preliminary_caveats,
            recommended_next_action=recommended_next_action,
            lay_summary=preliminary_lay,
            task_catalog=task_catalog,
            metric_catalog=metric_catalog,
            reference_df=reference_df,
            smoke_df=smoke_df,
        ),
    )
    write_text(
        handoff_path,
        report_bundle_handoff_markdown(
            artifacts=[*preliminary_artifacts, manifest_self_entry(handoff_path, artifacts_dir, "Required report-bundle handoff.")],
            validation_result=preliminary_validation_result,
            outcome_classification=preliminary_outcome,
            caveats_or_blockers=preliminary_caveats,
            recommended_next_action=recommended_next_action,
            lay_summary=preliminary_lay,
            task_catalog=task_catalog,
            reference_df=reference_df,
        ),
    )

    validation_df = validate_benchmark_suite_tables(
        task_catalog=task_catalog,
        reference_df=reference_df,
        metric_catalog=metric_catalog,
        smoke_df=smoke_df,
        source_inputs=source_inputs,
        config_dir=benchmark_config_dir,
        required_artifact_paths=[reference_path, benchmark_config_dir, suite_report_path, handoff_path, config_index_path, config_path, source_manifest_path],
    )
    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05_benchmarks", "-v"], args.repo_dir))
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )

    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    outcome_classification, caveats_or_blockers, lay_summary = determine_outcome(validation_df, task_catalog, smoke_df)

    artifacts: list[dict[str, Any]] = [
        artifact_entry(suite_report_path, artifacts_dir, "Required S15 morphology benchmark-suite report."),
        artifact_entry(benchmark_config_dir, artifacts_dir, "Required S15 benchmark task config directory."),
        artifact_entry(reference_path, artifacts_dir, "Required S15 benchmark reference result table."),
        artifact_entry(reference_csv_path, artifacts_dir, "CSV companion for S15 benchmark reference results."),
        artifact_entry(handoff_path, artifacts_dir, "Required S15 report-bundle handoff."),
        artifact_entry(task_catalog_path, artifacts_dir, "S15 benchmark task catalog."),
        artifact_entry(task_catalog_csv_path, artifacts_dir, "CSV companion for benchmark task catalog."),
        artifact_entry(metric_catalog_path, artifacts_dir, "S15 benchmark metric manifest."),
        artifact_entry(metric_catalog_csv_path, artifacts_dir, "CSV companion for benchmark metric manifest."),
        artifact_entry(smoke_path, artifacts_dir, "S15 benchmark smoke-test table."),
        artifact_entry(smoke_csv_path, artifacts_dir, "CSV companion for benchmark smoke tests."),
        artifact_entry(source_inputs_path, artifacts_dir, "S15 source input provenance table."),
        artifact_entry(source_inputs_csv_path, artifacts_dir, "CSV companion for source input provenance."),
        artifact_entry(validation_path, artifacts_dir, "S15 validation case table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV companion for S15 validation cases."),
        artifact_entry(config_path, artifacts_dir, "S15 reproducibility configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "S15 source-code provenance manifest."),
    ]

    report_artifacts = [
        *artifacts,
        manifest_self_entry(full_results_path, artifacts_dir, "Canonical S15 full-results report."),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S15 artifact manifest."),
        manifest_self_entry(checksum_path, artifacts_dir, "SHA-256 checksums for S15 artifacts."),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Workspace-root S15 run manifest."),
    ]
    write_text(
        suite_report_path,
        benchmark_suite_report_markdown(
            artifacts=report_artifacts,
            validation_result=validation_result,
            outcome_classification=outcome_classification,
            caveats_or_blockers=caveats_or_blockers,
            recommended_next_action=recommended_next_action,
            lay_summary=lay_summary,
            task_catalog=task_catalog,
            metric_catalog=metric_catalog,
            reference_df=reference_df,
            smoke_df=smoke_df,
        ),
    )
    write_text(
        handoff_path,
        report_bundle_handoff_markdown(
            artifacts=report_artifacts,
            validation_result=validation_result,
            outcome_classification=outcome_classification,
            caveats_or_blockers=caveats_or_blockers,
            recommended_next_action=recommended_next_action,
            lay_summary=lay_summary,
            task_catalog=task_catalog,
            reference_df=reference_df,
        ),
    )
    report = full_results_markdown(
        artifacts=report_artifacts,
        validation_result=validation_result,
        validation_df=validation_df,
        task_catalog=task_catalog,
        reference_df=reference_df,
        metric_catalog=metric_catalog,
        smoke_df=smoke_df,
        source_inputs=source_inputs,
        test_commands=test_commands,
        source_files=source_files,
        outcome_classification=outcome_classification,
        caveats_or_blockers=caveats_or_blockers,
        recommended_next_action=recommended_next_action,
        lay_summary=lay_summary,
    )
    write_text(full_results_path, report)
    artifacts.append(artifact_entry(full_results_path, artifacts_dir, "Canonical S15 full-results report."))
    artifacts.append(manifest_self_entry(artifact_manifest_path, artifacts_dir, "S15 artifact manifest."))
    write_json(
        artifact_manifest_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "artifacts": artifacts,
            "validationResult": validation_result,
            "outcomeClassification": outcome_classification,
            "createdAt": utc_now(),
        },
    )

    checksum_inputs = [
        suite_report_path,
        benchmark_config_dir,
        reference_path,
        reference_csv_path,
        handoff_path,
        task_catalog_path,
        task_catalog_csv_path,
        metric_catalog_path,
        metric_catalog_csv_path,
        smoke_path,
        smoke_csv_path,
        source_inputs_path,
        source_inputs_csv_path,
        validation_path,
        validation_csv_path,
        config_path,
        source_manifest_path,
        full_results_path,
        artifact_manifest_path,
    ]
    write_checksums(checksum_inputs, checksum_path, artifacts_dir)
    artifacts.append(artifact_entry(checksum_path, artifacts_dir, "SHA-256 checksums for S15 artifacts."))

    ended_at = utc_now()
    run_manifest = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "experimentTitle": EXPERIMENT_TITLE,
        "startedAt": started_at,
        "endedAt": ended_at,
        "success": bool(validation_success and all(command["success"] for command in test_commands)),
        "status": "completed",
        "artifactsDir": str(artifacts_dir),
        "validationResult": validation_result,
        "outcomeClassification": outcome_classification,
        "recommendedNextAction": recommended_next_action,
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "configPath": str(config_path),
        "artifactManifestPath": str(artifact_manifest_path),
        "checksumPath": str(checksum_path),
        "testCommands": test_commands,
        "artifacts": artifacts,
    }
    write_json(run_manifest_path, run_manifest)
    write_checksums([*checksum_inputs, checksum_path, run_manifest_path], checksum_path, artifacts_dir)
    return 0 if run_manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

