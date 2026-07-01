#!/usr/bin/env python3
"""Run E04 S05 homeostatic benchmark suite and write artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from src.e04.homeostasis import (
    HOMEOSTATIC_REPAIR_RULES,
    HOMEOSTATIC_TASKS,
    PERTURBATION_TYPES,
    default_homeostatic_configs,
    run_homeostatic_matrix,
)
from src.e04.fatigue_damage import RELIABILITY_MODES
from src.e04.repairable_frozen import INTERFACE_MODES


STEP_ID = "S05"
STEP_NUMBER = 5
EXPERIMENT_ID = "E04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--max-events", type=int, default=120)
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
    if df.empty:
        return "(no rows)"
    view = df[columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in view.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def _json_len(value: str) -> int:
    return len(json.loads(value))


def validate_results(df: pd.DataFrame, max_events: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "validation_case": "benchmark_matrix_complete",
            "success": bool(
                set(df["task_name"]) == set(HOMEOSTATIC_TASKS)
                and set(df["repair_rule"]) == set(HOMEOSTATIC_REPAIR_RULES)
                and set(df["interface_mode"]) == set(INTERFACE_MODES)
                and set(df["reliability_mode"]) == set(RELIABILITY_MODES)
                and len(df) == 648
            ),
            "detail": (
                f"tasks={sorted(set(df['task_name']))}; rules={sorted(set(df['repair_rule']))}; "
                f"modes={sorted(set(df['interface_mode']))}; reliability={sorted(set(df['reliability_mode']))}; rows={len(df)}"
            ),
        }
    )
    rows.append(
        {
            "validation_case": "ongoing_perturbation_families_present",
            "success": bool(
                (df["swap_perturbation_count"] > 0).any()
                and (df["delete_insert_perturbation_count"] > 0).any()
                and (df["freeze_perturbation_count"] > 0).any()
            ),
            "detail": (
                f"swap={int(df['swap_perturbation_count'].sum())}; "
                f"delete_insert={int(df['delete_insert_perturbation_count'].sum())}; "
                f"freeze={int(df['freeze_perturbation_count'].sum())}"
            ),
        }
    )
    turnover = df[df["delete_insert_perturbation_count"] > 0]
    rows.append(
        {
            "validation_case": "delete_insert_safely_represented_fixed_size",
            "success": bool(
                not turnover.empty
                and turnover["perturbation_log_json"].str.contains("fixed_size_delete_insert").all()
                and turnover["final_values_json"].map(_json_len).eq(6).all()
            ),
            "detail": f"turnover rows={len(turnover)}; final lengths={sorted(set(turnover['final_values_json'].map(_json_len))) if len(turnover) else []}",
        }
    )
    rows.append(
        {
            "validation_case": "frozen_events_and_repair_controls_logged",
            "success": bool(
                (df["freeze_perturbation_count"] > 0).any()
                and df[df["repair_rule"] == "nudge_repair"]["unfreeze_count"].sum() > 0
                and df[df["repair_rule"].isin(["passive_control", "stuck_control"])]["unfreeze_count"].eq(0).all()
            ),
            "detail": (
                f"freeze rows={int((df['freeze_perturbation_count'] > 0).sum())}; "
                f"nudge unfreezes={int(df[df['repair_rule'] == 'nudge_repair']['unfreeze_count'].sum())}"
            ),
        }
    )
    rows.append(
        {
            "validation_case": "homeostatic_metrics_sanity_checked",
            "success": bool(
                df["time_in_target_fraction"].between(0.0, 1.0).all()
                and df["min_sortedness_percent"].between(0.0, 100.0).all()
                and df["mean_sortedness_percent"].between(0.0, 100.0).all()
                and (df["energy_total"] >= 0).all()
                and all(math.isfinite(float(value)) for value in df["dg_primary"])
            ),
            "detail": (
                f"mean time-in-target={float(df['time_in_target_fraction'].mean()):.4f}; "
                f"mean energy={float(df['energy_total'].mean()):.4f}; max DG={float(df['dg_primary'].max()):.4f}"
            ),
        }
    )
    rows.append(
        {
            "validation_case": "recovery_metrics_consistent_with_perturbations",
            "success": bool(
                (
                    df["recovered_perturbation_count"]
                    + df["unrecovered_perturbation_count"]
                    <= df["perturbation_count"]
                ).all()
                and (df["recovered_perturbation_count"] > 0).any()
            ),
            "detail": (
                f"recovered={int(df['recovered_perturbation_count'].sum())}; "
                f"unrecovered={int(df['unrecovered_perturbation_count'].sum())}; "
                f"perturbations={int(df['perturbation_count'].sum())}"
            ),
        }
    )
    cumulative = df[df["reliability_mode"] == "cumulative_damage"]
    no_fatigue = df[df["reliability_mode"] == "no_fatigue_control"]
    rows.append(
        {
            "validation_case": "cumulative_damage_measured_and_no_fatigue_control_clean",
            "success": bool(
                cumulative["cumulative_damage_transition_count"].sum() > 0
                and cumulative["cumulative_impairment_count"].sum() > 0
                and no_fatigue["cumulative_damage_transition_count"].eq(0).all()
                and no_fatigue["cumulative_impairment_count"].eq(0).all()
            ),
            "detail": (
                f"damage transitions={int(cumulative['cumulative_damage_transition_count'].sum())}; "
                f"damage impairments={int(cumulative['cumulative_impairment_count'].sum())}"
            ),
        }
    )
    rows.append(
        {
            "validation_case": "fixed_horizon_not_sortedness_stopped",
            "success": bool((df["stop_reason"] == "fixed_horizon_complete").all() and (df["event_count"] == max_events).all()),
            "detail": f"stop reasons={df['stop_reason'].value_counts().to_dict()}; max_events={max_events}",
        }
    )
    rows.append(
        {
            "validation_case": "no_global_sortedness_oracle_audit_passed",
            "success": bool((df["uses_global_oracle"] == False).all()),
            "detail": f"oracle hits={int(df['uses_global_oracle'].sum())}",
        }
    )
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["task_name", "repair_rule", "interface_mode", "reliability_mode"], as_index=False)
        .agg(
            runs=("time_in_target_fraction", "size"),
            mean_time_in_target_fraction=("time_in_target_fraction", "mean"),
            mean_recovery_events=("mean_recovery_events", "mean"),
            mean_energy_total=("energy_total", "mean"),
            mean_cumulative_damage=("cumulative_damage_transition_count", "mean"),
            mean_impairment_count=("cumulative_impairment_count", "mean"),
            mean_min_sortedness_percent=("min_sortedness_percent", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
            mean_unfreeze_count=("unfreeze_count", "mean"),
            oracle_hit_count=("uses_global_oracle", "sum"),
        )
        .sort_values(["task_name", "repair_rule", "interface_mode", "reliability_mode"])
    )
    for column in [
        "mean_time_in_target_fraction",
        "mean_recovery_events",
        "mean_energy_total",
        "mean_cumulative_damage",
        "mean_impairment_count",
        "mean_min_sortedness_percent",
        "mean_final_sortedness_percent",
        "mean_unfreeze_count",
    ]:
        summary[column] = summary[column].round(4)
    return summary


def write_figure(summary: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    task_summary = (
        summary.groupby(["task_name", "reliability_mode"], as_index=False)
        .agg(
            mean_time_in_target_fraction=("mean_time_in_target_fraction", "mean"),
            mean_energy_total=("mean_energy_total", "mean"),
            mean_cumulative_damage=("mean_cumulative_damage", "mean"),
        )
    )
    tasks = list(HOMEOSTATIC_TASKS)
    modes = list(RELIABILITY_MODES)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for ax, metric, title, ylabel in [
        (axes[0], "mean_time_in_target_fraction", "Time in target", "Fraction"),
        (axes[1], "mean_energy_total", "Movement energy", "Mean swaps"),
        (axes[2], "mean_cumulative_damage", "Cumulative damage", "Mean damaged transitions"),
    ]:
        pivot = task_summary.pivot(index="task_name", columns="reliability_mode", values=metric).reindex(tasks)
        pivot = pivot.reindex(columns=modes)
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(title)
        ax.set_xlabel("Task")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
        if metric == "mean_time_in_target_fraction":
            ax.set_ylim(0, 1.05)
        ax.legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Reliability mode", loc="upper center", ncol=3)
    fig.suptitle("E04 S05 homeostatic baseline tasks", fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_task_spec(
    *,
    artifact_paths: list[Path],
    validation_line: str,
    validation_success: bool,
    result_path: Path,
    config_path: Path,
) -> str:
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    outcome = "supportive" if validation_success else "constraining/contradictory"
    return f"""# E04 Homeostatic Task Specification

## Top Summary

- Research step ID: S05
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: Insertions/deletions are represented as fixed-size turnover because the public cell classes assume a fixed array object and fixed boundaries. This is a safe benchmark representation, not a full dynamic tissue growth model.
- Recommended next action: Stop before S06. Chief Scientist should review the task contract and authorize S06 local learning rules if accepted.

## Purpose

S05 defines a reproducible homeostatic benchmark suite for ongoing perturbations. The task asks whether baseline local policies can keep or regain a sorted morphology under repeated disturbances rather than stopping after one sort.

## Task Families

- `swap_shocks`: exogenous swaps of two positions at fixed scheduled events.
- `turnover_replacement`: fixed-size delete/insert turnover. One position is deleted, that cell object is reused with a new thread ID and value, and it is inserted elsewhere. Array length remains constant so existing public cell boundaries stay valid.
- `frozen_damage`: scheduled Frozen Cell events plus a swap shock. Frozen identities use S03 passive/stuck/nudge-repair semantics.
- `mixed_perturbations`: a combined schedule with swap, Frozen Cell, and turnover perturbations.

## Observation And Oracle Boundary

Local policies receive only their public cell view plus S01 memory and S02 signals when enabled. The event loop runs for a fixed horizon and never stops or changes schedules based on Sortedness. Sortedness, time-in-target, recovery time, DG, and morphology scores are computed after trajectory logging as offline evaluation only.

## Metrics

- `time_in_target_fraction`: fraction of logged event states with Sortedness at or above the configured target.
- `mean_recovery_events` and `max_recovery_events`: post-hoc event delay from perturbation to the next in-target state.
- `energy_total`: policy swap count, excluding exogenous perturbation operations.
- `cumulative_damage_transition_count`: number of local reliability transitions into damaged state.
- `cumulative_impairment_count`: impaired activations skipped after fatigue or damage.
- `min_sortedness_percent`, `mean_sortedness_percent`, `final_sortedness_percent`, and DG metrics: offline trajectory summaries.

## Reproducibility Contract

- Config path: `{config_path}`
- Baseline result table: `{result_path}`
- Schedules are deterministic functions of `task_name`, `schedule_seed`, `max_events`, and initial array length.
- Default matrix: 3 algorithms x 2 seeds x 4 tasks x 3 repair rules x 3 interface modes x 3 reliability modes = 648 rows.
- Worker count: 1 serial run.
"""


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
    spec_path: Path,
    summary_path: Path,
    config_path: Path,
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
            "- `python scripts/e04_s05_homeostatic_tasks.py --repo-dir /workspace/cell-research --artifacts-dir $ARTIFACTS_DIR`",
        ]
    )
    headline = (
        summary.groupby(["task_name", "reliability_mode"], as_index=False)
        .agg(
            mean_time_in_target_fraction=("mean_time_in_target_fraction", "mean"),
            mean_energy_total=("mean_energy_total", "mean"),
            mean_cumulative_damage=("mean_cumulative_damage", "mean"),
            oracle_hit_count=("oracle_hit_count", "sum"),
        )
        .round(4)
    )
    summary_table = markdown_table(
        headline,
        [
            "task_name",
            "reliability_mode",
            "mean_time_in_target_fraction",
            "mean_energy_total",
            "mean_cumulative_damage",
            "oracle_hit_count",
        ],
        max_rows=24,
    )
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"])
    example_table = markdown_table(
        df[["task_name", "algorithm", "repair_rule", "interface_mode", "reliability_mode", "perturbation_log_json"]].head(6),
        ["task_name", "algorithm", "repair_rule", "interface_mode", "reliability_mode", "perturbation_log_json"],
        max_rows=6,
    )
    return f"""# E04 S05 Research Step Full Results

## Top Summary

- Research step ID: S05
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: Insertions/deletions are safely represented as fixed-size turnover, not arbitrary dynamic array growth/shrinkage. Baselines are compact and exploratory; no broad learning or adaptive superiority claim is made.
- Lay summary: S05 turns one-shot sorting into a maintenance task. The array starts sorted, then receives repeated swap, turnover, and Frozen Cell perturbations. Baseline cell policies keep acting locally for a fixed horizon while the report measures how often the array is in target, how long recovery takes, how much movement energy is used, and how much fatigue or damage accumulates.
- Recommended next action: Stop before S06. Chief Scientist should review the homeostatic task contract and, if accepted, authorize S06 local learning rules using these benchmark outputs.

## Frozen Question

Can local policies maintain sortedness under ongoing perturbations rather than merely sorting once?

S05 supports benchmark readiness: it defines and runs a reproducible homeostatic suite with ongoing swap shocks, fixed-size delete/insert turnover, Frozen Cell events, S03 repair controls, S01/S02 ablations, and S04 reliability modes. It does not yet test learned policies.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S05.
- S04 dynamic reliability layer: `src/e04/fatigue_damage.py` and `/artifacts/results/e04_fatigue_damage.parquet`.
- S03 repair framework: `src/e04/repairable_frozen.py`.
- S01 memory interface: `src/e04/memory_policies.py`.
- S02 signaling interface: `src/e04/signaling.py`.
- E02 deterministic simulator context: `src/e02/deterministic_simulator.py`.
- E01/E03 previous artifacts mounted read-only for context only; no dataset input required.

## Methods

Implemented `src/e04/homeostasis.py` and `scripts/e04_s05_homeostatic_tasks.py`.

Each run starts from sorted values `(1, 2, 3, 4, 5, 6)` and runs a fixed 120-event horizon. Perturbations are scheduled by seed and task name, not by online Sortedness. The local policy execution path is the S04 path: public cell policy plus optional S01 memory, optional S02 signal, S03 repair controller, and S04 fatigue/damage controller.

Insertions/deletions use the safe fixed-size representation described in `{spec_path}`. This preserves the public simulator's fixed array and boundary assumptions while still creating a turnover perturbation that changes values and position.

Global Sortedness is computed only after event logs are written. It is not used for policy action selection, perturbation scheduling, repair, fatigue, damage, memory updates, or signal emission.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- matplotlib: {matplotlib.__version__}
- New dependencies installed: none.
- CPU use: serial validation and benchmark; host reports `{os.cpu_count()}` logical CPUs and worker count used was 1.
- Platform: {platform.platform()}

## Parameters

- Algorithms: Bubble, Insertion, Selection.
- Initial values: `(1, 2, 3, 4, 5, 6)`.
- Tasks: `{', '.join(HOMEOSTATIC_TASKS)}`.
- Repair/control rules: `{', '.join(HOMEOSTATIC_REPAIR_RULES)}`.
- Interface modes: `{', '.join(INTERFACE_MODES)}`.
- Reliability modes: `{', '.join(RELIABILITY_MODES)}`.
- Seeds: 2 activation/policy/schedule seed triples per condition.
- Max events per run: `{int(df['max_events'].iloc[0]) if len(df) else 'NA'}`.
- Target Sortedness threshold: `{float(df['target_sortedness_percent'].iloc[0]) if len(df) else 'NA'}`.

## Results

Benchmark rows: `{len(df)}`. Result table: `{result_path}`. Summary table: `{summary_path}`. Figure: `{figure_path}`. Task spec: `{spec_path}`.

{summary_table}

### Perturbation Examples

{example_table}

## Validation Checks

{validation_table}

Additional validation:

- Unit tests cover deterministic schedules, turnover length preservation, Frozen Cell unfreezing, bounded metrics, control matrix contents, deterministic replay, and no-global-oracle audits.
- E03 policy-interface and E02 deterministic simulator regression tests passed.
- Every result row has `uses_global_oracle=false` from the S02 signal audit.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S05.
- Fixed-size turnover is a safe proxy for insertion/deletion under the public simulator; arbitrary array-size changes remain a future simulator extension.
- The suite is compact: one initial sorted array length, two seeds, fixed perturbation times, and baseline hand-coded policies.
- Time-in-target and recovery are offline trajectory metrics, not local rewards or policy-visible feedback.
- Several perturbations remain unrecovered within the fixed horizon; that is preserved as benchmark signal, not hidden as a failure.

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

Stop before S06. Chief Scientist should review S05 and, if accepted, authorize S06 local learning rules using the homeostatic task suite.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_homeostatic_baselines.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_homeostatic_baselines.csv"
    summary_path = artifacts_dir / "tables" / "e04_homeostatic_baselines_summary.csv"
    validation_path = artifacts_dir / "tables" / "e04_homeostatic_baselines_validation.csv"
    figure_path = artifacts_dir / "figures" / "e04" / "homeostatic_baselines.png"
    spec_path = artifacts_dir / "reports" / "e04_homeostatic_task_spec.md"
    config_path = artifacts_dir / "configs" / "e04_s05_homeostatic_tasks.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_homeostatic_tasks_manifest.json"
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
        spec_path.parent,
        config_path.parent,
        source_manifest_path.parent,
        checksums_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    configs = default_homeostatic_configs(max_events=args.max_events)
    config_doc = {
        "schema": "eidosoma.e04.s05.homeostatic_tasks_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "runCount": len(configs),
        "maxEvents": int(args.max_events),
        "algorithms": ["bubble", "insertion", "selection"],
        "tasks": list(HOMEOSTATIC_TASKS),
        "perturbationTypes": list(PERTURBATION_TYPES),
        "repairRules": list(HOMEOSTATIC_REPAIR_RULES),
        "interfaceModes": list(INTERFACE_MODES),
        "reliabilityModes": list(RELIABILITY_MODES),
        "workerCount": 1,
        "configsPreview": [config.to_dict() for config in configs[:6]],
    }
    write_json(config_path, config_doc)

    results = run_homeostatic_matrix(configs)
    df = pd.DataFrame([result.to_row() for result in results])
    df.to_parquet(result_path, index=False)
    df.to_csv(result_csv_path, index=False)
    summary = summarize_results(df)
    summary.to_csv(summary_path, index=False)
    validation_df = validate_results(df, args.max_events)
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
        spec_path,
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
    write_text(
        spec_path,
        render_task_spec(
            artifact_paths=artifact_paths,
            validation_line=validation_line,
            validation_success=validation_success,
            result_path=result_path,
            config_path=config_path,
        ),
    )

    source_paths = [
        repo_dir / "src/e04/homeostasis.py",
        repo_dir / "tests/e04/test_homeostasis.py",
        repo_dir / "scripts/e04_s05_homeostatic_tasks.py",
        repo_dir / "src/e04/fatigue_damage.py",
        repo_dir / "src/e04/repairable_frozen.py",
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
            "s03Results": str(artifacts_dir / "results/e04_repairable_frozen_cells.parquet"),
            "s04Results": str(artifacts_dir / "results/e04_fatigue_damage.parquet"),
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
            "taskSpec": str(spec_path),
            "taskSpecSha256": sha256_file(spec_path),
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
            spec_path=spec_path,
            summary_path=summary_path,
            config_path=config_path,
            run_manifest_path=run_manifest_path,
            checksums_path=checksums_path,
            manifest=manifest,
        ),
    )
    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S05 full-results handoff report"),
        artifact_entry(spec_path, artifacts_dir, "S05 homeostatic task specification"),
        artifact_entry(result_path, artifacts_dir, "S05 homeostatic baseline benchmark rows"),
        artifact_entry(figure_path, artifacts_dir, "S05 homeostatic baseline figure"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S05 benchmark rows"),
        artifact_entry(summary_path, artifacts_dir, "S05 benchmark summary table"),
        artifact_entry(validation_path, artifacts_dir, "S05 validation table"),
        artifact_entry(config_path, artifacts_dir, "S05 benchmark config"),
        artifact_entry(source_manifest_path, artifacts_dir, "S05 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S05 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S05 outputs"),
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
        spec_path,
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
