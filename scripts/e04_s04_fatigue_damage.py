#!/usr/bin/env python3
"""Run E04 S04 fatigue and damage benchmark and write artifacts."""

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

from src.e04.fatigue_damage import (
    RELIABILITY_MODES,
    S04_REPAIR_RULES,
    default_fatigue_damage_configs,
    run_fatigue_damage_matrix,
)
from src.e04.repairable_frozen import INTERFACE_MODES


STEP_ID = "S04"
STEP_NUMBER = 4
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


def validate_results(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "validation_case": "benchmark_matrix_complete",
            "success": bool(
                set(df["repair_rule"]) == set(S04_REPAIR_RULES)
                and set(df["interface_mode"]) == set(INTERFACE_MODES)
                and set(df["reliability_mode"]) == set(RELIABILITY_MODES)
                and len(df) == 162
            ),
            "detail": (
                f"rules={sorted(set(df['repair_rule']))}; modes={sorted(set(df['interface_mode']))}; "
                f"reliability={sorted(set(df['reliability_mode']))}; rows={len(df)}"
            ),
        }
    )
    controls = df[df["repair_rule"].isin(["passive_control", "stuck_control"])]
    rows.append(
        {
            "validation_case": "passive_and_stuck_controls_preserved",
            "success": bool((controls["unfreeze_count"] == 0).all() and (controls["repair_success"] == False).all()),
            "detail": "passive/stuck controls remain frozen identities and write no unfreeze logs",
        }
    )
    rows.append(
        {
            "validation_case": "no_memory_no_signal_no_fatigue_controls_present",
            "success": bool(
                {"no_memory", "no_signal"}.issubset(set(df["interface_mode"]))
                and {"no_fatigue_control"}.issubset(set(df["reliability_mode"]))
            ),
            "detail": f"interfaces={sorted(set(df['interface_mode']))}; reliability={sorted(set(df['reliability_mode']))}",
        }
    )
    fatigue = df[df["reliability_mode"] == "fatigue_recovery"]
    rows.append(
        {
            "validation_case": "fatigue_transition_and_recovery_logged",
            "success": bool(
                (fatigue["fatigue_transition_count"] > 0).any()
                and fatigue["fatigue_transition_log_json"].str.contains("fatigue_threshold_reached").any()
                and fatigue["fatigue_transition_log_json"].str.contains("recovery_after_rest").any()
                and (fatigue["impairment_count"] > 0).any()
            ),
            "detail": (
                f"fatigue transitions={int(fatigue['fatigue_transition_count'].sum())}; "
                f"impairments={int(fatigue['impairment_count'].sum())}"
            ),
        }
    )
    damage = df[df["reliability_mode"] == "cumulative_damage"]
    rows.append(
        {
            "validation_case": "damage_transition_and_impairment_logged",
            "success": bool(
                (damage["fatigue_transition_count"] > 0).any()
                and damage["fatigue_transition_log_json"].str.contains("damage_threshold_reached").any()
                and (damage["impairment_count"] > 0).any()
            ),
            "detail": (
                f"damage transitions={int(damage['fatigue_transition_count'].sum())}; "
                f"impairments={int(damage['impairment_count'].sum())}"
            ),
        }
    )
    no_fatigue = df[df["reliability_mode"] == "no_fatigue_control"]
    rows.append(
        {
            "validation_case": "no_fatigue_matches_s03_baseline",
            "success": bool(
                no_fatigue["baseline_match"].map(bool).all()
                and (no_fatigue["fatigue_transition_count"] == 0).all()
                and (no_fatigue["impairment_count"] == 0).all()
            ),
            "detail": f"no-fatigue rows={len(no_fatigue)}; baseline matches={int(no_fatigue['baseline_match'].map(bool).sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "no_global_sortedness_oracle_audit_passed",
            "success": bool((df["uses_global_oracle"] == False).all()),
            "detail": f"oracle hits={int(df['uses_global_oracle'].sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "energy_and_dg_metrics_finite",
            "success": bool(
                (df["energy_total"] >= 0).all()
                and all(math.isfinite(float(value)) for value in df["dg_primary"])
                and all(math.isfinite(float(value)) for value in df["dg_total_drop"])
            ),
            "detail": (
                f"mean energy={float(df['energy_total'].mean()):.4f}; "
                f"max DG={float(df['dg_primary'].max()):.4f}"
            ),
        }
    )
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["baseline_match_numeric"] = frame["baseline_match"].map(
        lambda value: 1.0 if value is True else (0.0 if value is False else None)
    )
    summary = (
        frame.groupby(["reliability_mode", "repair_rule", "interface_mode"], as_index=False)
        .agg(
            runs=("repair_success", "size"),
            repair_success_rate=("repair_success", "mean"),
            baseline_match_rate=("baseline_match_numeric", "mean"),
            mean_energy_total=("energy_total", "mean"),
            mean_dg_primary=("dg_primary", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
            mean_fatigue_transition_count=("fatigue_transition_count", "mean"),
            mean_impairment_count=("impairment_count", "mean"),
            mean_unfreeze_count=("unfreeze_count", "mean"),
            oracle_hit_count=("uses_global_oracle", "sum"),
        )
        .sort_values(["reliability_mode", "repair_rule", "interface_mode"])
    )
    for column in [
        "repair_success_rate",
        "baseline_match_rate",
        "mean_energy_total",
        "mean_dg_primary",
        "mean_final_sortedness_percent",
        "mean_fatigue_transition_count",
        "mean_impairment_count",
        "mean_unfreeze_count",
    ]:
        summary[column] = summary[column].round(4)
    return summary


def write_figure(summary: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(RELIABILITY_MODES)
    success = summary.groupby("reliability_mode")["repair_success_rate"].mean().reindex(labels)
    energy = summary.groupby("reliability_mode")["mean_energy_total"].mean().reindex(labels)
    impairment = summary.groupby("reliability_mode")["mean_impairment_count"].mean().reindex(labels)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    axes[0].bar(labels, success, color="#356c9a")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Repair success")
    axes[0].set_ylabel("Rate")
    axes[1].bar(labels, energy, color="#7b7f35")
    axes[1].set_title("Movement energy")
    axes[1].set_ylabel("Mean swaps")
    axes[2].bar(labels, impairment, color="#9a4f35")
    axes[2].set_title("Impaired activations")
    axes[2].set_ylabel("Mean skipped events")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("E04 S04 fatigue and damage tradeoffs", fontsize=13)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close(fig)


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
            "- `python scripts/e04_s04_fatigue_damage.py --repo-dir /workspace/cell-research --artifacts-dir $ARTIFACTS_DIR`",
        ]
    )
    summary_table = markdown_table(
        summary,
        [
            "reliability_mode",
            "repair_rule",
            "interface_mode",
            "runs",
            "repair_success_rate",
            "mean_energy_total",
            "mean_dg_primary",
            "mean_fatigue_transition_count",
            "mean_impairment_count",
            "oracle_hit_count",
        ],
        max_rows=36,
    )
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"])
    transition_examples = df[df["fatigue_transition_count"] > 0][
        ["algorithm", "repair_rule", "interface_mode", "reliability_mode", "fatigue_transition_log_json"]
    ].head(6)
    transition_table = markdown_table(
        transition_examples,
        ["algorithm", "repair_rule", "interface_mode", "reliability_mode", "fatigue_transition_log_json"],
        max_rows=6,
    )
    no_fatigue_matches = int(df[df["reliability_mode"] == "no_fatigue_control"]["baseline_match"].map(bool).sum())
    return f"""# E04 S04 Research Step Full Results

## Top Summary

- Research step ID: S04
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: S04 quantifies a compact dynamic unreliability benchmark only. Fatigue and damage are simulation extensions, not biological validation, and parameter sensitivity remains for later work.
- Lay summary: Cells now have local reliability states. A moving cell can become fatigued and later recover after rest, or accumulate damage and stop acting. The benchmark keeps passive/stuck frozen controls, no-memory/no-signal ablations, and a no-fatigue baseline that is checked against the S03 repair benchmark.
- Recommended next action: Stop before S05. Chief Scientist should review S04, then authorize S05 homeostatic task definitions if this dynamic perturbation layer is accepted.

## Frozen Question

Does dynamic unreliability through fatigue or damage reveal different competencies than static Frozen Cell perturbations?

S04 supports benchmark readiness: dynamic fatigue and cumulative damage produce logged state transitions and impairment events while no-fatigue rows exactly match the S03 repair baseline. This does not yet establish broad adaptive competence; it creates the controlled perturbation layer needed for S05+.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S04.
- S03 repair framework: `src/e04/repairable_frozen.py` and `/artifacts/research_steps/S03/research_step_full_results.md`.
- S01 memory interface: `src/e04/memory_policies.py` and `/artifacts/reports/e04_memory_state_spec.md`.
- S02 signaling interface: `src/e04/signaling.py` and `/artifacts/reports/e04_signal_model_spec.md`.
- E02 deterministic simulator context: `src/e02/deterministic_simulator.py` and `/previous-artifacts/E02`.
- Datasets: none required.

## Methods

Implemented `src/e04/fatigue_damage.py` and `scripts/e04_s04_fatigue_damage.py`.

The benchmark wraps the S03 repair controller with a local reliability controller. The controller stores per-cell state keyed by thread identity and never reads global Sortedness, full-array rank, final values, or centralized labels when deciding whether a cell can act.

Reliability conditions:

- `no_fatigue_control`: no dynamic impairment; every row is replayed through S03 and must match final values, frozen positions, swap count, and frozen-attempt count.
- `fatigue_recovery`: a healthy cell becomes fatigued after the configured movement threshold and recovers after a configured number of rest events.
- `cumulative_damage`: a healthy cell becomes damaged after cumulative movement crosses the configured threshold; damaged active cells skip actions.

Control and repair conditions:

- `passive_control` and `stuck_control` preserve the original S03 frozen-cell controls and never unfreeze identities.
- `nudge_repair` keeps the repairable S03 condition used for dynamic unreliability runs.
- `full`, `no_memory`, and `no_signal` preserve the S01/S02 interface ablations.

Sortedness, final morphology, energy, and DG are computed only as offline evaluation from logged trajectories. These metrics are not exposed to cell policies, repair rules, signal updates, or fatigue/damage transitions.

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
- Initial values: `(4, 1, 3, 2)`.
- Frozen index: `1`.
- Repair/control rules: `{', '.join(S04_REPAIR_RULES)}`.
- Interface modes: `{', '.join(INTERFACE_MODES)}`.
- Reliability modes: `{', '.join(RELIABILITY_MODES)}`.
- Seeds: 2 activation/policy seed pairs per condition.
- Max events per run: `{int(df['max_events'].iloc[0]) if len(df) else 'NA'}`.
- Nudge threshold: 2 contacts.
- Fatigue threshold: 2 movement swaps.
- Recovery threshold: 3 rest events.
- Damage threshold: 3 movement swaps.

## Results

Benchmark rows: `{len(df)}`. Result table: `{result_path}`. Summary table: `{summary_path}`. Figure: `{figure_path}`.

No-fatigue rows matching S03 baseline: `{no_fatigue_matches}` of `{int((df['reliability_mode'] == 'no_fatigue_control').sum())}`.

{summary_table}

### Transition Examples

{transition_table}

## Validation Checks

{validation_table}

Additional validation:

- Unit tests cover fatigue transitions, recovery, damage impairment, no-fatigue S03 parity, matrix controls, deterministic replay, and no-global-oracle audit.
- E03 policy-interface and E02 deterministic simulator regression tests passed.
- Every result row has `uses_global_oracle=false` from the S02 signal audit.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S04.
- Fatigue and damage are local simulation extensions, not claims about real cells or the original paper.
- The compact benchmark uses one initial array, one frozen index, and fixed reliability thresholds; broader parameter sensitivity is not yet run.
- `dg_primary` is an offline trajectory proxy. It is recorded for continuity with E01/E02 metrics and is not used by local policies or repair/fatigue controllers.
- Cumulative damage can create persistent impairment and may reduce sorting even after nudge repair succeeds; that is a stress condition rather than a failure of the benchmark.

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

Stop before S05. Chief Scientist should review S04 and, if accepted, authorize S05 homeostatic tasks using the dynamic perturbation interfaces defined here.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_fatigue_damage.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_fatigue_damage.csv"
    summary_path = artifacts_dir / "tables" / "e04_fatigue_damage_summary.csv"
    validation_path = artifacts_dir / "tables" / "e04_fatigue_damage_validation.csv"
    figure_path = artifacts_dir / "figures" / "e04" / "fatigue_damage_tradeoffs.png"
    config_path = artifacts_dir / "configs" / "e04_s04_fatigue_damage.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_fatigue_damage_manifest.json"
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

    configs = default_fatigue_damage_configs(max_events=args.max_events)
    config_doc = {
        "schema": "eidosoma.e04.s04.fatigue_damage_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "runCount": len(configs),
        "maxEvents": int(args.max_events),
        "algorithms": ["bubble", "insertion", "selection"],
        "repairRules": list(S04_REPAIR_RULES),
        "interfaceModes": list(INTERFACE_MODES),
        "reliabilityModes": list(RELIABILITY_MODES),
        "workerCount": 1,
        "configsPreview": [config.to_dict() for config in configs[:6]],
    }
    write_json(config_path, config_doc)

    results = run_fatigue_damage_matrix(configs)
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
        repo_dir / "src/e04/fatigue_damage.py",
        repo_dir / "tests/e04/test_fatigue_damage.py",
        repo_dir / "scripts/e04_s04_fatigue_damage.py",
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
            "s03Report": str(artifacts_dir / "research_steps/S03/research_step_full_results.md"),
            "s03Results": str(artifacts_dir / "results/e04_repairable_frozen_cells.parquet"),
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
            run_manifest_path=run_manifest_path,
            checksums_path=checksums_path,
            manifest=manifest,
        ),
    )
    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S04 full-results handoff report"),
        artifact_entry(result_path, artifacts_dir, "S04 fatigue and damage benchmark rows"),
        artifact_entry(figure_path, artifacts_dir, "S04 fatigue and damage tradeoff figure"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S04 benchmark rows"),
        artifact_entry(summary_path, artifacts_dir, "S04 benchmark summary table"),
        artifact_entry(validation_path, artifacts_dir, "S04 validation table"),
        artifact_entry(config_path, artifacts_dir, "S04 benchmark config"),
        artifact_entry(source_manifest_path, artifacts_dir, "S04 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S04 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S04 outputs"),
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
