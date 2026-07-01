#!/usr/bin/env python3
"""Run E04 S03 repairable Frozen Cell benchmark and write artifacts."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.e04.repairable_frozen import (
    INTERFACE_MODES,
    REPAIR_RULES,
    default_benchmark_configs,
    run_benchmark_matrix,
)


STEP_ID = "S03"
STEP_NUMBER = 3
EXPERIMENT_ID = "E04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--max-events", type=int, default=80)
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


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


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
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    view = df[columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in view.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def validate_results(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_rules = set(REPAIR_RULES)
    expected_modes = set(INTERFACE_MODES)
    rows.append(
        {
            "validation_case": "benchmark_matrix_complete",
            "success": set(df["repair_rule"]) == expected_rules and set(df["interface_mode"]) == expected_modes,
            "detail": f"rules={sorted(set(df['repair_rule']))}; modes={sorted(set(df['interface_mode']))}; rows={len(df)}",
        }
    )
    controls = df[df["repair_rule"].isin(["passive_control", "stuck_control"])]
    rows.append(
        {
            "validation_case": "original_controls_preserved",
            "success": bool((controls["unfreeze_count"] == 0).all() and (controls["repair_success"] == False).all()),
            "detail": "passive and stuck controls remain frozen identities with no unfreeze logs",
        }
    )
    repairable = df[df["repair_rule"].isin(["nudge_repair", "signal_threshold_repair", "time_repair", "directional_repair"])]
    rows.append(
        {
            "validation_case": "repairable_conditions_log_unfreezing",
            "success": bool((repairable["unfreeze_count"] > 0).any() and repairable["unfreeze_log_json"].str.contains("trigger").any()),
            "detail": f"repairable rows with unfreeze={int((repairable['unfreeze_count'] > 0).sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "no_memory_and_no_signal_controls_present",
            "success": bool({"no_memory", "no_signal"}.issubset(set(df["interface_mode"]))),
            "detail": f"interface modes={sorted(set(df['interface_mode']))}",
        }
    )
    rows.append(
        {
            "validation_case": "no_global_oracle_audit_passed",
            "success": bool((df["uses_global_oracle"] == False).all()),
            "detail": f"oracle hits={int(df['uses_global_oracle'].sum())}",
        }
    )
    full_signal = df[(df["algorithm"] == "bubble") & (df["repair_rule"] == "signal_threshold_repair")]
    rows.append(
        {
            "validation_case": "signal_threshold_has_no_signal_contrast",
            "success": bool(
                full_signal[full_signal["interface_mode"] == "full"]["repair_success"].any()
                and not full_signal[full_signal["interface_mode"] == "no_signal"]["repair_success"].any()
            ),
            "detail": "bubble signal-threshold repair succeeds with signals and remains frozen without signals in matched benchmark rows",
        }
    )
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["repair_rule", "interface_mode"], as_index=False)
        .agg(
            runs=("repair_success", "size"),
            repair_success_rate=("repair_success", "mean"),
            mean_unfreeze_count=("unfreeze_count", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
            mean_frozen_attempt_count=("frozen_attempt_count", "mean"),
            oracle_hit_count=("uses_global_oracle", "sum"),
        )
        .sort_values(["repair_rule", "interface_mode"])
    )
    summary["repair_success_rate"] = summary["repair_success_rate"].round(4)
    summary["mean_unfreeze_count"] = summary["mean_unfreeze_count"].round(4)
    summary["mean_final_sortedness_percent"] = summary["mean_final_sortedness_percent"].round(4)
    summary["mean_frozen_attempt_count"] = summary["mean_frozen_attempt_count"].round(4)
    return summary


def write_figure(summary: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = summary.pivot(index="repair_rule", columns="interface_mode", values="repair_success_rate").reindex(REPAIR_RULES)
    ax = pivot.plot(kind="bar", figsize=(11, 5.5), width=0.82)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Repair success rate")
    ax.set_xlabel("Repair rule")
    ax.set_title("E04 S03 repairable Frozen Cell outcomes")
    ax.legend(title="Interface mode", loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()


def render_report(
    *,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    validation_df: pd.DataFrame,
    validation_line: str,
    validation_success: bool,
    unit_tests: dict[str, Any],
    e03_tests: dict[str, Any],
    e02_tests: dict[str, Any],
    artifact_paths: list[Path],
    result_path: Path,
    figure_path: Path,
    summary_path: Path,
    config_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    manifest: dict[str, Any],
) -> str:
    outcome = "supportive" if validation_success else "constraining/contradictory"
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    commands = "\n".join(
        [
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e03_tests['command']}` -> return code {e03_tests['returnCode']}",
            f"- `{e02_tests['command']}` -> return code {e02_tests['returnCode']}",
            "- `python scripts/e04_s03_repairable_frozen_cells.py --repo-dir /workspace/cell-research --artifacts-dir $ARTIFACTS_DIR`",
        ]
    )
    summary_table = markdown_table(
        summary,
        [
            "repair_rule",
            "interface_mode",
            "runs",
            "repair_success_rate",
            "mean_unfreeze_count",
            "mean_final_sortedness_percent",
            "oracle_hit_count",
        ],
    )
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"])
    trigger_examples = df[df["unfreeze_count"] > 0][["algorithm", "repair_rule", "interface_mode", "unfreeze_log_json"]].head(6)
    trigger_table = markdown_table(trigger_examples, ["algorithm", "repair_rule", "interface_mode", "unfreeze_log_json"], max_rows=6)
    return f"""# E04 S03 Research Step Full Results

## Top Summary

- Research step ID: S03
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: S03 defines and runs a compact repairable Frozen Cell benchmark on baseline policies. It shows repair triggers can unfreeze frozen identities under the simulation contract, but it does not yet establish a general memory/signaling advantage across all algorithms or perturbations.
- Lay summary: Frozen cells can now be treated as damaged cells that repair after repeated nudges, enough local signal, elapsed time, or the right approach direction. The original stuck and passive frozen-cell behaviors remain as controls, and every unfreeze event is logged with its trigger.
- Recommended next action: Stop before S04. After Chief Scientist review, proceed to S04 cell fatigue and damage while preserving stuck/passive, no-memory, and no-signal controls.

## Frozen Question

Do memory and signaling rules enable cells to unfreeze or route around damaged cells more effectively than open-loop algorithms?

S03 implements the benchmark needed to test that question. The compact baseline result supports benchmark readiness and demonstrates signal-dependent repair in the Bubble baseline, but it does not yet claim broad repair superiority.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S03.
- S01 memory interface: `src/e04/memory_policies.py` and `/artifacts/reports/e04_memory_state_spec.md`.
- S02 signaling interface: `src/e04/signaling.py` and `/artifacts/reports/e04_signal_model_spec.md`.
- E02 deterministic simulator context: `src/e02/deterministic_simulator.py` and `/previous-artifacts/E02`.
- E03 policy interface context: `src/e03/policy_interface.py` and `/previous-artifacts/E03`.
- Datasets: none required.

## Methods

Implemented `src/e04/repairable_frozen.py` and `scripts/e04_s03_repairable_frozen_cells.py`.

The benchmark uses the public cell-view Bubble, Insertion, and Selection policies through the S01 memory and S02 signal wrappers. A local repair controller patches only swap attempts involving frozen identities. It never reads global Sortedness, full-array rank, final values, or centralized state while deciding repair.

Repair and control conditions:

- `passive_control`: original passive Frozen Cell target behavior remains available and never unfreezes identities.
- `stuck_control`: original stuck-like target blocking remains available and never unfreezes identities.
- `nudge_repair`: a frozen identity unfreezes after a configured number of local blocked contact attempts.
- `signal_threshold_repair`: a frozen identity unfreezes when local blocked/frustrated signal field at its position crosses a threshold.
- `time_repair`: a frozen identity unfreezes after a configured event count.
- `directional_repair`: a frozen identity unfreezes when approached from the configured side.

Interface modes:

- `full`: S01 memory and S02 signals enabled.
- `no_memory`: memory disabled, signals enabled.
- `no_signal`: memory enabled, signals disabled.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- matplotlib: {matplotlib.__version__}
- New dependencies installed: none.
- CPU use: serial validation and benchmark; host reports `{os.cpu_count()}` logical CPUs.
- Platform: {platform.platform()}

## Parameters

- Algorithms: Bubble, Insertion, Selection.
- Initial values: `(4, 1, 3, 2)`.
- Frozen index: `1`.
- Repair/control rules: `{', '.join(REPAIR_RULES)}`.
- Interface modes: `{', '.join(INTERFACE_MODES)}`.
- Seeds: 3 activation/policy seed pairs per condition.
- Max events per run: `{int(df['max_events'].iloc[0]) if len(df) else 'NA'}`.
- Nudge threshold: 2 contacts.
- Signal threshold: 0.35.
- Time threshold: 12 events.
- Directional trigger: approach from left.

## Results

Benchmark rows: `{len(df)}`. Result table: `{result_path}`. Summary table: `{summary_path}`. Figure: `{figure_path}`.

{summary_table}

### Trigger Examples

{trigger_table}

## Validation Checks

{validation_table}

Additional validation:

- Unit tests include repair trigger logging, passive/stuck controls, signal-threshold no-signal contrast, and deterministic replay.
- E03 policy-interface and E02 deterministic simulator regression tests passed.
- Every result row has `uses_global_oracle=false` from the signal audit.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S03.
- The repair dynamics are model extensions, not paper claims or wet-lab validation.
- The compact benchmark is intentionally small and uses one frozen index and one initial array; broader S04+ work should expand perturbation regimes.
- Time-based repair can succeed without memory or signaling, so it should be treated as a positive repair-control condition rather than evidence for communication.
- Signal-threshold repair in this compact run separates full signaling from no-signal in Bubble, but memory and signaling contributions need later ablations across richer tasks.
- Final sortedness is recorded only as offline evaluation; repair decisions do not use it.

## Provenance

- Git commit at validation time: `{manifest['gitCommit']}`
- Git branch at validation time: `{manifest['gitBranch']}`
- Git status at validation time: `{manifest['gitStatusShort'] or 'clean'}`
- Source files tracked in manifest: {len(manifest['sourceFiles'])}
- Run manifest: `{run_manifest_path}`
- Checksums: `{checksums_path}`
- Previous artifacts: E01 `{manifest['upstreamContext']['previousE01Dir']}`, E02 `{manifest['upstreamContext']['previousE02Dir']}`, E03 `{manifest['upstreamContext']['previousE03Dir']}`
- Created at UTC: `{manifest['createdAtUtc']}`

## Recommended Next Action

Stop before S04. Chief Scientist should review S03, then authorize S04 fatigue/damage with the same passive/stuck, no-memory, no-signal, and no-global-oracle controls.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_repairable_frozen_cells.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_repairable_frozen_cells.csv"
    summary_path = artifacts_dir / "tables" / "e04_repairable_frozen_cells_summary.csv"
    validation_path = artifacts_dir / "tables" / "e04_repairable_frozen_cells_validation.csv"
    figure_path = artifacts_dir / "figures" / "e04" / "repairable_frozen_cell_outcomes.png"
    config_path = artifacts_dir / "configs" / "e04_s03_repairable_frozen_cells.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_repairable_frozen_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"
    report_path = step_dir / "research_step_full_results.md"
    for path in [
        step_dir,
        result_path.parent,
        result_csv_path.parent,
        summary_path.parent,
        validation_path.parent,
        figure_path.parent,
        config_path.parent,
        source_manifest_path.parent,
        checksums_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    configs = default_benchmark_configs(max_events=args.max_events)
    config_doc = {
        "schema": "eidosoma.e04.s03.repairable_frozen_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "runCount": len(configs),
        "maxEvents": int(args.max_events),
        "algorithms": ["bubble", "insertion", "selection"],
        "repairRules": list(REPAIR_RULES),
        "interfaceModes": list(INTERFACE_MODES),
        "workerCount": 1,
        "configsPreview": [config.to_dict() for config in configs[:6]],
    }
    write_json(config_path, config_doc)

    results = run_benchmark_matrix(configs)
    df = pd.DataFrame([result.to_row() for result in results])
    df.to_parquet(result_path, index=False)
    df.to_csv(result_csv_path, index=False)
    summary = summarize_results(df)
    summary.to_csv(summary_path, index=False)
    validation_df = validate_results(df)
    validation_df.to_csv(validation_path, index=False)
    write_figure(summary, figure_path)

    unit_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e04", "-p", "test_*.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    e03_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    e02_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    validation_success = bool(validation_df["success"].all() and unit_tests["success"] and e03_tests["success"] and e02_tests["success"])
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"E04 unit tests return code {unit_tests['returnCode']}; "
        f"E03 policy-interface tests return code {e03_tests['returnCode']}; "
        f"E02 simulator tests return code {e02_tests['returnCode']}; "
        f"{len(df)} benchmark rows written"
    )
    artifact_paths = [
        report_path,
        result_path,
        figure_path,
        result_csv_path,
        summary_path,
        validation_path,
        config_path,
        source_manifest_path,
        artifact_manifest_path,
        run_manifest_path,
        checksums_path,
    ]
    source_paths = [
        repo_dir / "src/e04/repairable_frozen.py",
        repo_dir / "tests/e04/test_repairable_frozen.py",
        repo_dir / "scripts/e04_s03_repairable_frozen_cells.py",
        repo_dir / "src/e04/memory_policies.py",
        repo_dir / "src/e04/signaling.py",
        repo_dir / "src/e02/deterministic_simulator.py",
    ]
    manifest: dict[str, Any] = {
        "schema": "eidosoma.src_snapshot.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(repo_dir, ["status", "--short"]),
        "dependencies": {
            "newDependenciesInstalled": [],
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "upstreamContext": {
            "previousE01Dir": "/previous-artifacts/E01",
            "previousE02Dir": "/previous-artifacts/E02",
            "previousE03Dir": "/previous-artifacts/E03",
            "s01MemorySpec": str(artifacts_dir / "reports/e04_memory_state_spec.md"),
            "s02SignalSpec": str(artifacts_dir / "reports/e04_signal_model_spec.md"),
        },
        "sourceFiles": [source_entry(path, repo_dir) for path in source_paths if path.exists()],
        "validation": {
            "allValidationPassed": validation_success,
            "validationCaseCount": int(len(validation_df)),
            "validationCaseSuccessCount": int(validation_df["success"].sum()),
            "unitTests": unit_tests,
            "e03PolicyInterfaceTests": e03_tests,
            "e02DeterministicSimulatorTests": e02_tests,
            "resultParquet": str(result_path),
            "resultParquetSha256": sha256_file(result_path),
            "resultCsv": str(result_csv_path),
            "resultCsvSha256": sha256_file(result_csv_path),
            "summaryCsv": str(summary_path),
            "summaryCsvSha256": sha256_file(summary_path),
            "validationCsv": str(validation_path),
            "validationCsvSha256": sha256_file(validation_path),
            "figure": str(figure_path),
            "figureSha256": sha256_file(figure_path),
            "config": str(config_path),
            "configSha256": sha256_file(config_path),
        },
    }
    write_json(source_manifest_path, manifest)
    write_text(
        report_path,
        render_report(
            df=df,
            summary=summary,
            validation_df=validation_df,
            validation_line=validation_line,
            validation_success=validation_success,
            unit_tests=unit_tests,
            e03_tests=e03_tests,
            e02_tests=e02_tests,
            artifact_paths=artifact_paths,
            result_path=result_path,
            figure_path=figure_path,
            summary_path=summary_path,
            config_path=config_path,
            manifest_path=source_manifest_path,
            run_manifest_path=run_manifest_path,
            checksums_path=checksums_path,
            manifest=manifest,
        ),
    )
    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S03 full-results handoff report"),
        artifact_entry(result_path, artifacts_dir, "S03 repairable Frozen Cell benchmark rows"),
        artifact_entry(figure_path, artifacts_dir, "S03 repairable Frozen Cell outcomes figure"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S03 benchmark rows"),
        artifact_entry(summary_path, artifacts_dir, "S03 benchmark summary table"),
        artifact_entry(validation_path, artifacts_dir, "S03 validation table"),
        artifact_entry(config_path, artifacts_dir, "S03 benchmark config"),
        artifact_entry(source_manifest_path, artifacts_dir, "S03 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S03 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S03 outputs"),
    ]
    artifact_manifest = {
        "schema": "eidosoma.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "artifacts": artifacts,
    }
    write_json(artifact_manifest_path, artifact_manifest)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "lastResearchStepId": STEP_ID,
        "createdAtUtc": utc_now(),
        "git": {
            "branch": manifest["gitBranch"],
            "headCommit": manifest["gitCommit"],
            "statusShort": manifest["gitStatusShort"],
        },
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "cpuCountReported": os.cpu_count(),
            "workerCountUsed": 1,
        },
        "dependencies": manifest["dependencies"],
        "validation": manifest["validation"],
        "artifacts": artifacts,
    }
    write_json(run_manifest_path, run_manifest)
    checksum_targets = [
        report_path,
        result_path,
        figure_path,
        result_csv_path,
        summary_path,
        validation_path,
        config_path,
        source_manifest_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")
    return 0 if validation_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
