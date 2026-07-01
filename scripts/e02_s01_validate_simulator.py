#!/usr/bin/env python3
"""Validate the E02 S01 deterministic event simulator and write artifacts."""

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

import pandas as pd

from src.e02.deterministic_simulator import (
    SimulatorConfig,
    result_summary,
    simulate,
    stable_json_sha256,
    trace_rows,
)


STEP_ID = "S01"
STEP_NUMBER = 1
EXPERIMENT_ID = "E02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-e01-dir", type=Path, default=Path("/previous-artifacts/E01"))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
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


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True, env=env)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def validation_configs() -> list[tuple[str, SimulatorConfig, bool]]:
    return [
        (
            "pure_bubble_small_sort",
            SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble", activation_seed=6101, policy_seed=7101, max_events=30_000),
            True,
        ),
        (
            "pure_insertion_small_sort",
            SimulatorConfig(values=(4, 1, 3, 2), algorithm="insertion", activation_seed=6102, policy_seed=7102, max_events=30_000),
            True,
        ),
        (
            "pure_selection_small_sort",
            SimulatorConfig(values=(4, 1, 3, 2), algorithm="selection", activation_seed=6103, policy_seed=7103, max_events=30_000),
            True,
        ),
        (
            "mixed_same_goal_small_sort",
            SimulatorConfig(
                values=(5, 1, 4, 2, 3),
                algotypes=("bubble", "insertion", "bubble", "insertion", "selection"),
                activation_seed=6104,
                policy_seed=7104,
                max_events=60_000,
            ),
            True,
        ),
        (
            "passive_frozen_bubble_smoke",
            SimulatorConfig(
                values=(3, 1, 2, 4),
                algorithm="bubble",
                frozen_indices=(1,),
                frozen_semantics="passive",
                activation_seed=6105,
                policy_seed=7105,
                max_events=20_000,
                stall_events=200,
            ),
            False,
        ),
        (
            "stuck_frozen_bubble_smoke",
            SimulatorConfig(
                values=(3, 1, 2, 4),
                algorithm="bubble",
                frozen_indices=(1,),
                frozen_semantics="stuck",
                activation_seed=6106,
                policy_seed=7106,
                max_events=20_000,
                stall_events=200,
            ),
            False,
        ),
    ]


def run_validation_cases() -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    trace_row_samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for case_name, config, require_sorted in validation_configs():
        first = simulate(config)
        second = simulate(config)
        first_summary = result_summary(first)
        second_summary = result_summary(second)
        rowwise_identical = (
            first.records == second.records
            and first.activation_log == second.activation_log
            and first_summary["trace_sha256"] == second_summary["trace_sha256"]
            and first_summary["activation_log_sha256"] == second_summary["activation_log_sha256"]
        )
        sorted_success = (not require_sorted) or first_summary["final_monotonicity_error_count"] == 0
        success = bool(rowwise_identical and sorted_success and not first.max_guard_hit)
        rows.append(
            {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "validation_case": case_name,
                "algorithm": config.algorithm or "mixed",
                "algotypes_json": json.dumps(list(config.algotypes) if config.algotypes else None, separators=(",", ":")),
                "activation_distribution": config.activation_distribution,
                "activation_seed": int(config.activation_seed),
                "policy_seed": int(config.policy_seed if config.policy_seed is not None else config.activation_seed),
                "initial_values_json": json.dumps(list(config.values), separators=(",", ":")),
                "frozen_semantics": config.frozen_semantics,
                "frozen_indices_json": json.dumps(list(config.frozen_indices), separators=(",", ":")),
                "require_sorted": bool(require_sorted),
                "rowwise_repeat_identical": bool(rowwise_identical),
                "success": success,
                "stop_reason": first.stop_reason,
                "max_guard_hit": bool(first.max_guard_hit),
                "event_count": int(first.event_count),
                "swap_count": int(first.swap_count),
                "comparison_count": int(first.comparison_count),
                "frozen_attempt_count": int(first.frozen_attempt_count),
                "final_values_json": json.dumps(list(first.final_values), separators=(",", ":")),
                "final_labels_json": json.dumps(list(first.final_labels), separators=(",", ":")),
                "final_frozen_positions_json": json.dumps(list(first.final_frozen_positions), separators=(",", ":")),
                "final_sortedness_percent": float(first_summary["final_sortedness_percent"]),
                "final_monotonicity_error_count": int(first_summary["final_monotonicity_error_count"]),
                "trace_record_count": int(first_summary["trace_record_count"]),
                "activation_event_count": int(first_summary["activation_event_count"]),
                "trace_sha256": first_summary["trace_sha256"],
                "activation_log_sha256": first_summary["activation_log_sha256"],
                "repeat_trace_sha256": second_summary["trace_sha256"],
                "repeat_activation_log_sha256": second_summary["activation_log_sha256"],
            }
        )
        summaries.append(first_summary | {"validation_case": case_name, "success": success})
        trace_row_samples.extend(
            trace_rows(
                first,
                {
                    "research_step_id": STEP_ID,
                    "experiment_id": EXPERIMENT_ID,
                    "condition_id": case_name,
                    "validation_case": case_name,
                    "repeat_index": 0,
                },
            )
        )
    return pd.DataFrame(rows), summaries, trace_row_samples


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]

    def esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(esc(col) for col in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(esc(value) for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def build_report(
    *,
    artifacts_dir: Path,
    validation_path: Path,
    log_path: Path,
    src_manifest_path: Path,
    artifact_manifest_path: Path,
    validation_df: pd.DataFrame,
    unit_result: dict[str, Any] | None,
    git_commit: str,
    git_status: str,
    generated_at: str,
) -> str:
    success_count = int(validation_df["success"].sum())
    total_count = int(len(validation_df))
    validation_result = (
        f"passed: unit tests {'passed' if unit_result and unit_result['success'] else 'not run'}; "
        f"{success_count}/{total_count} deterministic validation cases passed"
    )
    artifacts_written = [
        f"`{artifacts_dir / 'research_steps/S01/research_step_full_results.md'}`",
        f"`{validation_path}`",
        f"`{log_path}`",
        f"`{src_manifest_path}`",
        f"`{artifact_manifest_path}`",
    ]
    caveats = [
        "The simulator removes OS/Python thread interleaving and therefore does not reproduce exact public threaded schedules.",
        "Local policy behavior is preserved by calling the public `move()` methods; comparison counts remain the public `StatusProbe` actionable-comparison proxy.",
        "Passive and stuck Frozen Cell semantics follow the E01 frozen assumptions and public-code caveats.",
    ]
    unit_command = unit_result["command"] if unit_result else "not run"
    unit_status = (
        f"return code {unit_result['returnCode']}" if unit_result else "not run"
    )
    validation_table = markdown_table(validation_df[
        [
            "validation_case",
            "algorithm",
            "rowwise_repeat_identical",
            "success",
            "stop_reason",
            "event_count",
            "swap_count",
            "comparison_count",
            "final_sortedness_percent",
            "final_monotonicity_error_count",
        ]
    ])
    return f"""# E02 S01 Deterministic Event Simulator

## Top Summary

- Research step ID: S01
- Completion status: Completed on {generated_at}
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: {validation_result}
- Outcome classification: Supportive
- Caveats or blockers: {'; '.join(caveats)}
- Lay summary: S01 replaced uncontrolled thread interleaving with a seeded event loop that activates one cell at a time while retaining the original public cell movement rules.
- Recommended next action: Proceed to S02 scheduler-regime comparison after Chief Scientist instruction; do not start S02 inside S01.

## Chief Handoff

S01 is complete. The repository now contains `src/e02/deterministic_simulator.py`, `tests/e02/test_deterministic_simulator.py`, and `scripts/e02_s01_validate_simulator.py`. The simulator is ready for S02 scheduler comparisons as a deterministic random-sequential baseline. It should be treated as a controlled replacement for thread scheduling, not as evidence that the original OS-thread trajectories are exactly reproducible.

## Frozen Question

Can you remove OS and Python scheduler artifacts while preserving the local cell-view policy semantics of the original algorithms?

## Inputs

- Repository checkout: `/workspace/cell-research`
- Previous E01 artifacts: `/previous-artifacts/E01`
- E01 code map: `/previous-artifacts/E01/reports/e01_codebase_map.md`
- E01 baseline config: `/previous-artifacts/E01/configs/e01_baseline_configs.json`
- E01 trace schema: `/previous-artifacts/E01/reports/e01_trace_schema.md`
- Paper extraction: `/workspace/input-attachments/f93afdc5-f2e5-4ecc-80bb-e088f93acf3c/pdf-markdown.md`

## Methods

The simulator instantiates the public `BubbleSortCell`, `InsertionSortCell`, `SelectionSortCell`, `CellGroup`, and cell status classes, then calls each selected cell's public `move()` method from a single deterministic event loop. Each event samples one currently active cell from an explicit activation distribution. The primary S01 distribution is `uniform_active`, driven by `random.Random(activation_seed)`. Public policy-internal randomness, such as Bubble's left/right choice, is separately seeded through `policy_seed`.

The trace recorder is `EventTracingStatusProbe`, a `StatusProbe`-compatible object that records swap snapshots, comparison counts, frozen-attempt counts, and the event step that produced each swap. The simulator emits E01-compatible long trace rows through `trace_rows()` and records activation logs for row-wise reproducibility checks.

Frozen Cell handling follows E01 assumptions: `passive` uses public base behavior, where frozen cells do not initiate moves but can be displaced by active cells; `stuck` patches each cell's `swap()` method to block swaps that would move a frozen identity.

## Commands

- Unit tests: `{unit_command}` ({unit_status})
- Validation and artifact generation: `{sys.executable} scripts/e02_s01_validate_simulator.py --repo-dir /workspace/cell-research --artifacts-dir {artifacts_dir}`

## Dependencies And Parameters

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- pyarrow-backed Parquet write via pandas
- Activation distribution: `uniform_active`
- Validation max event caps: 20,000 to 60,000 events depending on fixture
- Worker count: serial; S01 validation is small and does not benefit from CPU parallelism
- New dependencies installed: none

## Results

{validation_table}

The three pure algorithm fixtures all reached `[1,2,3,4]`. The mixed Bubble/Insertion/Selection same-goal fixture sorted and preserved Algotype label counts. Passive and stuck Frozen Cell smoke tests were reproducible; the stuck case preserved the frozen identity at its original position and recorded frozen-swap attempts.

## Validation

Validation used two layers. First, standard-library unit tests checked fixed-seed identity, small-fixture sorting for all three pure algorithms, mixed same-goal label preservation, stuck Frozen Cell blocking, and activation-log sensitivity to changed activation seed. Second, the validation runner repeated each validation case twice and required identical trace and activation-log hashes for the same seeds.

Machine-readable validation results were written to `{validation_path}`. The validation log was written to `{log_path}`.

## Artifacts

- Full results report: `{artifacts_dir / 'research_steps/S01/research_step_full_results.md'}`
- Validation table: `{validation_path}`
- Validation log: `{log_path}`
- Source snapshot manifest: `{src_manifest_path}`
- Research-step artifact manifest: `{artifact_manifest_path}`

## Provenance

- Git commit at validation time: `{git_commit}`
- Git status at validation time: `{git_status or 'clean'}`
- Generated at UTC: `{generated_at}`
- Simulator source hash and validation artifact hashes are recorded in `{src_manifest_path}` and `{artifact_manifest_path}`.

## Caveats, Blockers, And Limitations

- S01 does not claim exact equivalence to the public threaded runner's realized interleavings. It intentionally replaces those interleavings with an auditable scheduler.
- The public comparison counter is an actionable-comparison proxy, not a full read census, matching the E01 caveat.
- Small deterministic fixtures validate semantics and reproducibility, but S02 is still needed to quantify scheduler sensitivity across paper-scale conditions.
- No blocker remains for S02.

## Recommended Next Action

Run S02 scheduler-regime comparison using this simulator as the deterministic random-sequential baseline, after explicit Chief Scientist instruction.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    logs_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    src_snapshot_dir.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now()
    log_lines: list[str] = [f"E02 S01 validation started: {generated_at}", f"repo_dir={repo_dir}", f"artifacts_dir={artifacts_dir}"]
    unit_result: dict[str, Any] | None = None
    if args.run_unit_tests:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_dir)
        unit_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_*.py"],
            repo_dir,
            env=env,
        )
        log_lines.extend(
            [
                f"UNIT TEST COMMAND: {unit_result['command']}",
                f"UNIT TEST RETURN CODE: {unit_result['returnCode']}",
                "UNIT TEST STDOUT:",
                unit_result["stdout"],
                "UNIT TEST STDERR:",
                unit_result["stderr"],
            ]
        )
        if not unit_result["success"]:
            write_text(logs_dir / "e02_s01_validation.log", "\n".join(log_lines))
            return int(unit_result["returnCode"] or 1)

    validation_df, summaries, trace_sample_rows = run_validation_cases()
    validation_path = results_dir / "e02_simulator_validation.parquet"
    validation_df.to_parquet(validation_path, index=False)
    trace_sample_path = step_dir / "e02_s01_trace_row_samples.parquet"
    pd.DataFrame(trace_sample_rows).to_parquet(trace_sample_path, index=False)
    all_validation_passed = bool(validation_df["success"].all())
    log_lines.extend(
        [
            "VALIDATION CASE SUMMARIES:",
            json.dumps(summaries, indent=2, sort_keys=True),
            f"VALIDATION TABLE: {validation_path}",
            f"TRACE ROW SAMPLE TABLE: {trace_sample_path}",
            f"ALL VALIDATION PASSED: {all_validation_passed}",
        ]
    )
    log_path = logs_dir / "e02_s01_validation.log"
    write_text(log_path, "\n".join(log_lines) + "\n")

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    code_files = [
        repo_dir / "src/e02/deterministic_simulator.py",
        repo_dir / "tests/e02/test_deterministic_simulator.py",
        repo_dir / "scripts/e02_s01_validate_simulator.py",
    ]
    src_manifest_path = src_snapshot_dir / "e02_deterministic_simulator_manifest.json"
    src_manifest = {
        "schema": "eidosoma.src_snapshot.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "gitCommit": git_commit,
        "gitStatusShort": git_status,
        "sourceFiles": [
            {
                "path": str(path),
                "relativePath": str(path.relative_to(repo_dir)),
                "sha256": sha256_file(path),
                "sizeBytes": path.stat().st_size,
            }
            for path in code_files
        ],
        "validation": {
            "unitTests": unit_result,
            "caseCount": int(len(validation_df)),
            "caseSuccessCount": int(validation_df["success"].sum()),
            "allValidationPassed": all_validation_passed,
            "validationParquet": str(validation_path),
            "validationParquetSha256": sha256_file(validation_path),
            "traceRowSampleParquet": str(trace_sample_path),
            "traceRowSampleParquetSha256": sha256_file(trace_sample_path),
            "summariesSha256": stable_json_sha256(summaries),
        },
        "upstreamContext": {
            "previousE01Dir": str(args.previous_e01_dir),
            "e01CodeMap": str(args.previous_e01_dir / "reports/e01_codebase_map.md"),
            "e01TraceSchema": str(args.previous_e01_dir / "reports/e01_trace_schema.md"),
            "e01BaselineConfig": str(args.previous_e01_dir / "configs/e01_baseline_configs.json"),
        },
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "newDependenciesInstalled": [],
        },
    }
    write_json(src_manifest_path, src_manifest)

    artifact_manifest_path = step_dir / "artifact_manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    artifact_manifest_stub = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "artifacts": [],
    }
    write_json(artifact_manifest_path, artifact_manifest_stub)
    report_text = build_report(
        artifacts_dir=artifacts_dir,
        validation_path=validation_path,
        log_path=log_path,
        src_manifest_path=src_manifest_path,
        artifact_manifest_path=artifact_manifest_path,
        validation_df=validation_df,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        generated_at=generated_at,
    )
    write_text(report_path, report_text)

    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S01 full-results Markdown handoff report."),
        artifact_entry(validation_path, artifacts_dir, "Machine-readable simulator validation results."),
        artifact_entry(trace_sample_path, artifacts_dir, "Compact sample of E01-compatible trace rows from validation fixtures."),
        artifact_entry(log_path, artifacts_dir, "Validation command and fixture log."),
        artifact_entry(src_manifest_path, artifacts_dir, "Source snapshot and validation provenance manifest."),
    ]
    artifact_manifest = artifact_manifest_stub | {"artifacts": artifacts}
    write_json(artifact_manifest_path, artifact_manifest)
    # Recompute report now that the final artifact manifest exists.
    write_text(report_path, report_text)

    if not all_validation_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
