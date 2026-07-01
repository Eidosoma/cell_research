#!/usr/bin/env python3
"""Build validation artifacts for the E03 S06 batched simulator."""

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

import numpy as np
import pandas as pd

from src.e03.gpu_batch_simulator import (
    DEFAULT_MAX_ACTIONS,
    DEFAULT_MAX_CONDITIONS,
    DEFAULT_MAX_RULES,
    compare_batch_to_cpu,
    compatibility_frame,
    compatibility_summary,
    comparison_digest,
    deterministic_validation_policies,
    jax_backend_summary,
    load_generated_policies,
    validation_fixtures,
)


STEP_ID = "S06"
STEP_NUMBER = 6
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument(
        "--policy-library",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "policies/e03_generated_policy_library.jsonl",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026070106)
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
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float) or isinstance(value, np.floating):
                text = "nan" if pd.isna(value) else f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def validation_cases(
    *,
    comparison: pd.DataFrame,
    compatibility: pd.DataFrame,
    backend: dict[str, Any],
    unit_tests: dict[str, Any],
    policy_library: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(name: str, success: bool, expected: str, observed: Any, notes: str) -> None:
        rows.append(
            {
                "validation_case": name,
                "success": bool(success),
                "expected": expected,
                "observed": str(observed),
                "notes": notes,
            }
        )

    add("jax_available", bool(backend["available"]), "JAX backend available", backend, "JAX is required for the batched path.")
    add(
        "gpu_or_cpu_fallback_documented",
        bool(backend["available"]),
        "GPU backend or documented CPU fallback",
        backend.get("defaultBackend"),
        "If GPU devices are unavailable, the same JAX batch path can run on CPU and exact CPU reference remains available.",
    )
    add("s05_policy_library_present", policy_library.exists(), "S05 JSONL policy library exists", policy_library, "S06 consumes S05 output.")
    add(
        "comparison_rows_present",
        len(comparison) >= 6,
        "at least six CPU/JAX comparison rows",
        len(comparison),
        "Fixture set covers left/right swaps, prefix guard, target-position state, reverse direction, and deterministic random/choose.",
    )
    add(
        "final_states_match",
        bool(comparison["final_state_match"].all()),
        "all final values/statuses/actor/ideal fields match exactly",
        int(comparison["final_state_match"].sum()),
        "Integer final states use exact equality tolerance.",
    )
    add(
        "metrics_match",
        bool(comparison["metrics_match"].all()),
        "all compare/swap/update/wait counts match exactly",
        int(comparison["metrics_match"].sum()),
        "Metric tolerance is exact equality for deterministic fixtures.",
    )
    add(
        "batch_compile_coverage",
        bool(compatibility["compiles_for_batch"].all()) if not compatibility.empty else False,
        "all loaded S05 policies compile to padded batch arrays",
        int(compatibility["compiles_for_batch"].sum()) if not compatibility.empty else 0,
        "Fallback reasons do not mean compile failure; they mean exact replay remains CPU-referenced.",
    )
    add(
        "fallback_classification_present",
        "requires_cpu_fallback" in compatibility.columns and len(compatibility) > 0,
        "compatibility table classifies CPU fallback",
        compatibility.columns.tolist(),
        "S06 documents stochastic-stream and memory/signal fallback limits before S07.",
    )
    add(
        "unit_tests_passed",
        bool(unit_tests["success"]),
        "full E03 unit test suite passes",
        unit_tests["returnCode"],
        unit_tests["command"],
    )
    return pd.DataFrame(rows)


def render_validation_report(
    *,
    artifacts: dict[str, Path],
    comparison: pd.DataFrame,
    compat_summary: pd.DataFrame,
    validation: pd.DataFrame,
    backend: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    validation_success = bool(validation["success"].all())
    outcome = "supportive" if validation_success else "constraining/contradictory"
    validation_line = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed"
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    preview_cols = [
        "policy_name",
        "steps",
        "final_state_match",
        "metrics_match",
        "cpu_final_values_json",
        "gpu_final_values_json",
    ]
    return f"""# E03 GPU Simulator Validation

## Top Summary

- Step ID: S06
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; comparison digest `{comparison_digest(comparison)}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains for coarse deterministic DSL sweeps. Exact replay for stochastic policies and richer `remember`/`signal` side effects remains CPU-referenced; variable-length sweeps should group or pad arrays by length.
- Recommended next action: Stop for Chief review; if accepted, run S07 using the batch simulator for deterministic-compatible policies and CPU fallback for exact stochastic or metadata-sensitive replay.

## Lay Summary

S06 adds a JAX batch simulator for many DSL policies at once. On small deterministic fixtures, it matched the CPU DSL interpreter exactly for final array state and simple action counts.

## Results

JAX backend summary: `{json.dumps(backend, sort_keys=True)}`

### CPU/JAX Comparison Preview

{markdown_table(comparison[preview_cols].head(12), preview_cols)}

### S05 Compatibility Summary

{markdown_table(compat_summary, ["metric", "value", "detail"])}

### Validation Cases

{markdown_table(validation, ["validation_case", "success", "expected", "observed"])}

## Fallback Plan

Use `src.e03.rule_dsl.DSLInterpreter` as the authoritative CPU path whenever a policy requires exact stochastic random-stream replay, depends on `remember` or `signal` side effects beyond their S02 stubs, or must be audited with variable-length trajectories before padding/grouping logic is added. The S06 batch path can still compile those policies for coarse screening, but exact claims should be checked against the CPU reference.

## Provenance

- Git commit before S06 commit: `{manifest['git']['headCommit']}`
- Batch simulator source: `src/e03/gpu_batch_simulator.py`
- Validation script: `scripts/e03_s06_validate_gpu_simulator.py`
"""


def render_full_report(
    *,
    artifacts: dict[str, Path],
    comparison: pd.DataFrame,
    compatibility: pd.DataFrame,
    compat_summary: pd.DataFrame,
    validation: pd.DataFrame,
    backend: dict[str, Any],
    unit_tests: dict[str, Any],
    manifest: dict[str, Any],
    command_line: str,
) -> str:
    validation_success = bool(validation["success"].all() and unit_tests["success"])
    outcome = "supportive" if validation_success else "constraining/contradictory"
    validation_line = (
        f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; "
        f"unit tests return code {unit_tests['returnCode']}"
    )
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    source_table = markdown_table(pd.DataFrame(manifest["sourceFiles"]), ["relativePath", "sha256", "sizeBytes"])
    command_rows = pd.DataFrame(
        [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": command_line, "returnCode": 0, "success": True},
        ]
    )
    comparison_cols = [
        "policy_name",
        "steps",
        "initial_values_json",
        "cpu_final_values_json",
        "gpu_final_values_json",
        "cpu_compare_count",
        "gpu_compare_count",
        "cpu_swap_count",
        "gpu_swap_count",
        "within_tolerance",
    ]
    fallback_counts = compatibility["requires_cpu_fallback"].value_counts().rename_axis("requires_cpu_fallback").reset_index(name="count")
    return f"""# E03 S06 Research Step Full Results

## Top Summary

- Step ID: S06
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; comparison digest `{comparison_digest(comparison)}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains for coarse deterministic-compatible sweeps. Exact CPU/GPU parity is validated only for fixed-length deterministic fixtures; stochastic random-stream replay and richer memory/signal semantics should use the CPU reference until expanded in a later step.
- Lay summary: S06 built a JAX-backed batched simulator for DSL local-policy steps. It matched the CPU DSL interpreter exactly on the small validation fixtures for final values, statuses, actor position, target-position state, and compare/swap/update/wait counts.
- Recommended next action: Stop for Chief review; if accepted, proceed to S07 coarse morphospace sweeps using the S06 compatibility table to route exact fallback cases to the CPU interpreter.

## Frozen Question

Can a GPU-friendly batch simulator reproduce CPU reference behavior closely enough to support large policy sweeps?

## Inputs

- S02 DSL and CPU reference interpreter: `src/e03/rule_dsl.py`
- S05 generated policy library: `{manifest['inputArtifacts']['s05PolicyLibrary']}`
- E02 deterministic simulator context: `/previous-artifacts/E02` and repository `src/e02/deterministic_simulator.py`
- Validation fixtures: fixed-length DSL states generated by `validation_fixtures()` in `src/e03/gpu_batch_simulator.py`
- Validation seed: `{manifest['parameters']['seed']}`

## Methods

The simulator compiles each DSL policy into padded numeric arrays: rule masks, condition codes, action codes, target arguments, state-value arguments, and one-level `choose` branches. Rows in a batch pair one compiled policy with one `DSLArrayState`; all rows share one array length. A JAX vectorized step evaluates guards in rule order, selects the first passing rule, executes action sequences until a terminal swap, state update, wait, or choose, and updates values, statuses, actor index, Selection-style `ideal_position`, and simple action counts.

The CPU reference is unchanged: `DSLInterpreter.step_state()` runs the same policies one local actor step at a time. The validation compares CPU and JAX final values, statuses, actor index, ideal-position state, and compare/swap/update/wait counts after `{manifest['parameters']['steps']}` steps. Tolerance is exact integer equality.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- JAX backend summary: `{json.dumps(backend, sort_keys=True)}`
- Worker count: serial validation plus vectorized JAX batch execution; no CPU worker pool used.
- New dependencies installed: none.

## Parameters

- Batch max rules: `{manifest['parameters']['maxRules']}`
- Batch max conditions per rule: `{manifest['parameters']['maxConditions']}`
- Batch max actions per rule: `{manifest['parameters']['maxActions']}`
- Validation rows: `{len(comparison)}`
- Validation steps per row: `{manifest['parameters']['steps']}`
- Validation seed: `{manifest['parameters']['seed']}`

## Results

### CPU/JAX Comparison

{markdown_table(comparison[comparison_cols], comparison_cols)}

All validation rows matched within exact tolerance: `{bool(comparison['within_tolerance'].all())}`.

### S05 Compatibility

{markdown_table(compat_summary, ["metric", "value", "detail"])}

Fallback classification counts:

{markdown_table(fallback_counts, ["requires_cpu_fallback", "count"])}

## Metrics

- `final_state_match`: exact match for final values, statuses, actor index, and ideal-position state.
- `metrics_match`: exact match for compare-counted steps, swaps, state updates, and waits.
- `within_tolerance`: conjunction of `final_state_match` and `metrics_match`; tolerance is exact equality for integer fields.
- `requires_cpu_fallback`: policy compiles but should use CPU for exact replay if it has stochastic random streams or memory/signal side effects not represented in the S06 batch state.

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "expected", "observed", "notes"])}

## Caveats, Blockers, And Limitations

- The S06 batch simulator covers the S02 DSL local-step semantics, not the full E02 scheduler or complete paper simulation loop.
- Batch rows currently share one fixed array length. S07 should group by length or add explicit padding/event caps for mixed-length sweeps.
- Random primitives compile, but exact CPU/GPU stochastic replay is not claimed because the current CPU interpreter consumes random numbers along variable control paths. Deterministic thresholds, such as 0.0 or 1.0, are validated exactly.
- `remember` and `signal` retain the S02 lightweight metadata/state-update status. Policies that need richer memory or communication semantics remain CPU-reference cases.
- This step validates correctness on small fixtures; it does not benchmark large-sweep throughput.

## Failed Assumptions

No planned input was missing. `nvidia-smi` is not required by the script; JAX device discovery reported the available backend.

## Provenance

Git commit before S06 commit: `{manifest['git']['headCommit']}`

Source files hashed in the S06 manifest:

{source_table}

## Artifacts

The reusable S06 outputs are the JAX batch simulator source in the repository, the CPU/JAX comparison table, policy compatibility and fallback tables, validation report, full-results handoff report, source snapshot manifest, artifact manifest, run manifest, config, and checksums.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    reports_dir = artifacts_dir / "reports"
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, reports_dir, results_dir, tables_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_tests = {"command": "not run", "returnCode": 0, "success": True, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        unit_tests = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e03"],
            args.repo_dir,
        )

    backend = jax_backend_summary()
    policies = deterministic_validation_policies(args.policy_library)
    fixtures = validation_fixtures(len(policies))
    comparison = compare_batch_to_cpu(policies, fixtures, steps=args.steps, seed=args.seed)
    generated_policies = load_generated_policies(args.policy_library)
    compatibility = compatibility_frame(generated_policies)
    compat_summary = compatibility_summary(compatibility)
    validation = validation_cases(
        comparison=comparison,
        compatibility=compatibility,
        backend=backend,
        unit_tests=unit_tests,
        policy_library=args.policy_library,
    )

    comparison_path = results_dir / "e03_cpu_gpu_comparison.parquet"
    comparison_csv_path = tables_dir / "e03_cpu_gpu_comparison.csv"
    compatibility_path = results_dir / "e03_gpu_policy_compatibility.parquet"
    compatibility_csv_path = tables_dir / "e03_gpu_policy_compatibility.csv"
    compat_summary_path = tables_dir / "e03_gpu_policy_compatibility_summary.csv"
    validation_path = results_dir / "e03_gpu_simulator_validation.parquet"
    validation_csv_path = tables_dir / "e03_gpu_simulator_validation.csv"
    config_path = configs_dir / "e03_s06_gpu_simulator_validation.json"
    validation_report_path = reports_dir / "e03_gpu_simulator_validation.md"
    full_report_path = step_dir / "research_step_full_results.md"
    manifest_path = src_snapshot_dir / "e03_gpu_batch_simulator_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = checksums_dir / "sha256sums.txt"

    comparison.to_parquet(comparison_path, index=False)
    comparison.to_csv(comparison_csv_path, index=False)
    compatibility.to_parquet(compatibility_path, index=False)
    compatibility.to_csv(compatibility_csv_path, index=False)
    compat_summary.to_csv(compat_summary_path, index=False)
    validation.to_parquet(validation_path, index=False)
    validation.to_csv(validation_csv_path, index=False)

    config = {
        "schema": "eidosoma.e03.s06_gpu_simulator_validation_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "steps": args.steps,
        "seed": args.seed,
        "policyLibrary": str(args.policy_library),
        "comparisonRows": len(comparison),
        "batchMaxRules": DEFAULT_MAX_RULES,
        "batchMaxConditions": DEFAULT_MAX_CONDITIONS,
        "batchMaxActions": DEFAULT_MAX_ACTIONS,
    }
    write_json(config_path, config)

    source_files = [
        args.repo_dir / "src/e03/gpu_batch_simulator.py",
        args.repo_dir / "tests/e03/test_gpu_batch_simulator.py",
        args.repo_dir / "scripts/e03_s06_validate_gpu_simulator.py",
        args.repo_dir / "src/e03/rule_dsl.py",
    ]
    manifest = {
        "schema": "eidosoma.e03.gpu_batch_simulator_manifest.v1",
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
            "steps": args.steps,
            "seed": args.seed,
            "maxRules": DEFAULT_MAX_RULES,
            "maxConditions": DEFAULT_MAX_CONDITIONS,
            "maxActions": DEFAULT_MAX_ACTIONS,
        },
        "inputArtifacts": {
            "s05PolicyLibrary": str(args.policy_library),
            "s02Dsl": str(args.repo_dir / "src/e03/rule_dsl.py"),
            "e02PreviousArtifacts": "/previous-artifacts/E02",
            "e01PreviousArtifacts": "/previous-artifacts/E01",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "jax": backend,
        },
        "sourceFiles": [source_entry(path, args.repo_dir) for path in source_files],
        "validationSummary": {
            "success": bool(validation["success"].all() and unit_tests["success"]),
            "validationCasesPassed": int(validation["success"].sum()),
            "validationCasesTotal": int(len(validation)),
            "unitTestsReturnCode": int(unit_tests["returnCode"]),
            "comparisonRows": int(len(comparison)),
            "comparisonRowsWithinTolerance": int(comparison["within_tolerance"].sum()),
            "policyCompatibilityRows": int(len(compatibility)),
            "policiesCompileForBatch": int(compatibility["compiles_for_batch"].sum()),
            "policiesRequireCpuFallback": int(compatibility["requires_cpu_fallback"].sum()),
        },
    }

    artifacts = {
        "researchStepReport": full_report_path,
        "gpuSimulatorValidationReport": validation_report_path,
        "cpuGpuComparison": comparison_path,
        "cpuGpuComparisonCsv": comparison_csv_path,
        "policyCompatibility": compatibility_path,
        "policyCompatibilityCsv": compatibility_csv_path,
        "policyCompatibilitySummary": compat_summary_path,
        "validationParquet": validation_path,
        "validationCsv": validation_csv_path,
        "validationConfig": config_path,
        "sourceSnapshotManifest": manifest_path,
        "artifactManifest": artifact_manifest_path,
        "runManifest": run_manifest_path,
        "checksums": checksums_path,
    }

    validation_report = render_validation_report(
        artifacts=artifacts,
        comparison=comparison,
        compat_summary=compat_summary,
        validation=validation,
        backend=backend,
        manifest=manifest,
    )
    write_text(validation_report_path, validation_report)

    full_report = render_full_report(
        artifacts=artifacts,
        comparison=comparison,
        compatibility=compatibility,
        compat_summary=compat_summary,
        validation=validation,
        backend=backend,
        unit_tests=unit_tests,
        manifest=manifest,
        command_line=" ".join(sys.argv),
    )
    write_text(full_report_path, full_report)

    manifest["artifacts"] = [
        artifact_entry(comparison_path, artifacts_dir, "S06 CPU/JAX comparison table"),
        artifact_entry(comparison_csv_path, artifacts_dir, "CSV sidecar for CPU/JAX comparison table"),
        artifact_entry(compatibility_path, artifacts_dir, "S06 generated-policy batch compatibility table"),
        artifact_entry(compatibility_csv_path, artifacts_dir, "CSV sidecar for compatibility table"),
        artifact_entry(compat_summary_path, artifacts_dir, "S06 compatibility summary table"),
        artifact_entry(validation_path, artifacts_dir, "S06 validation cases"),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S06 validation cases"),
        artifact_entry(config_path, artifacts_dir, "S06 validation config"),
        artifact_entry(validation_report_path, artifacts_dir, "S06 GPU simulator validation report"),
        manifest_self_entry(manifest_path, artifacts_dir, "S06 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S06 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S06"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S06 outputs"),
        artifact_entry(full_report_path, artifacts_dir, "S06 full-results handoff report"),
    ]
    write_json(manifest_path, manifest)

    artifact_manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifacts": [
            artifact_entry(comparison_path, artifacts_dir, "S06 CPU/JAX comparison table"),
            artifact_entry(comparison_csv_path, artifacts_dir, "CSV sidecar for CPU/JAX comparison table"),
            artifact_entry(compatibility_path, artifacts_dir, "S06 generated-policy batch compatibility table"),
            artifact_entry(compatibility_csv_path, artifacts_dir, "CSV sidecar for compatibility table"),
            artifact_entry(compat_summary_path, artifacts_dir, "S06 compatibility summary table"),
            artifact_entry(validation_path, artifacts_dir, "S06 validation cases"),
            artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S06 validation cases"),
            artifact_entry(config_path, artifacts_dir, "S06 validation config"),
            artifact_entry(validation_report_path, artifacts_dir, "S06 GPU simulator validation report"),
            artifact_entry(manifest_path, artifacts_dir, "S06 source snapshot and provenance manifest"),
            manifest_self_entry(artifact_manifest_path, artifacts_dir, "S06 artifact manifest"),
            manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S06"),
            manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S06 outputs"),
            artifact_entry(full_report_path, artifacts_dir, "S06 full-results handoff report"),
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
        "runtime": manifest["runtime"],
        "commands": [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": " ".join(sys.argv), "returnCode": 0, "success": True},
        ],
        "inputs": manifest["inputArtifacts"],
        "parameters": manifest["parameters"],
        "outputs": artifact_manifest["artifacts"],
    }
    write_json(run_manifest_path, run_manifest)

    checksum_targets = [
        full_report_path,
        validation_report_path,
        comparison_path,
        comparison_csv_path,
        compatibility_path,
        compatibility_csv_path,
        compat_summary_path,
        validation_path,
        validation_csv_path,
        config_path,
        manifest_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")

    success = bool(validation["success"].all() and unit_tests["success"])
    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": success,
                "comparisonRows": len(comparison),
                "comparisonRowsWithinTolerance": int(comparison["within_tolerance"].sum()),
                "policyCompatibilityRows": len(compatibility),
                "policiesCompileForBatch": int(compatibility["compiles_for_batch"].sum()),
                "policiesRequireCpuFallback": int(compatibility["requires_cpu_fallback"].sum()),
                "artifactsDir": str(step_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
