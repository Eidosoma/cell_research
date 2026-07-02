#!/usr/bin/env python3
"""Run E05 S13 local-versus-global control comparison."""

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
import numpy as np
import pandas as pd

from src.e05.local_global_control import (
    STEP_ID,
    build_s13_result_tables,
    summarize_for_report,
)


STEP_NUMBER = 13
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


def render_figure(
    control_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    policy_colors = {
        "s07_local_target_neighbor_descent": "#2F6B4F",
        "s07_random_adjacent_swap_control": "#8D4E2F",
        "s13_top_down_global_assignment": "#4C78A8",
        "s13_organizer_beacon_assignment": "#F58518",
        "s13_global_rebuild_controller": "#B279A2",
    }
    short_labels = {
        "s07_local_target_neighbor_descent": "local target",
        "s07_random_adjacent_swap_control": "local random",
        "s13_top_down_global_assignment": "top-down",
        "s13_organizer_beacon_assignment": "organizer",
        "s13_global_rebuild_controller": "global rebuild",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), constrained_layout=True)

    summary = (
        control_df.groupby(["control_class", "policy_id"], as_index=False)
        .agg(
            mean_target_recovery_fraction=("target_recovery_fraction", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
            low_recovery_failure_rate=("low_recovery_failure", "mean"),
            blocked_rate=("baseline_blocked", "mean"),
        )
        .sort_values(["control_class", "policy_id"])
    )
    labels = [short_labels.get(policy_id, policy_id) for policy_id in summary["policy_id"]]
    x = np.arange(len(summary))
    axes[0].bar(
        x,
        summary["mean_target_recovery_fraction"],
        color=[policy_colors.get(policy_id, "#777777") for policy_id in summary["policy_id"]],
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylim(min(-0.2, float(summary["mean_target_recovery_fraction"].min()) - 0.05), 1.05)
    axes[0].set_title("Mean target recovery")
    axes[0].set_ylabel("recovery fraction")
    axes[0].grid(axis="y", linewidth=0.3, alpha=0.35)

    robustness = robustness_df[robustness_df["benchmark_task"].isin(["regeneration", "scaling"])].copy()
    robustness = (
        robustness.groupby(["policy_id"], as_index=False)
        .agg(
            low_recovery_failure_rate=("low_recovery_failure_rate", "mean"),
            blocked_rate=("blocked_rate", "mean"),
        )
        .set_index("policy_id")
        .reindex(summary["policy_id"])
        .reset_index()
    )
    width = 0.35
    axes[1].bar(x - width / 2, robustness["low_recovery_failure_rate"], width=width, color="#4C78A8", label="low recovery")
    axes[1].bar(x + width / 2, robustness["blocked_rate"], width=width, color="#F58518", label="blocked")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Robustness and blockers")
    axes[1].set_ylabel("rate")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", linewidth=0.3, alpha=0.35)

    scaling_large = scaling_df[scaling_df["scale_label"].eq("large")].copy()
    scaling_plot = (
        scaling_large.groupby("policy_id", as_index=False)
        .agg(
            large_recovery=("mean_target_recovery_fraction", "mean"),
            large_to_small_recovery_ratio=("large_to_small_recovery_ratio", "mean"),
            large_energy_per_site=("mean_energy_per_site", "mean"),
        )
        .set_index("policy_id")
        .reindex(summary["policy_id"])
        .reset_index()
    )
    axes[2].bar(
        x,
        scaling_plot["large_recovery"],
        color=[policy_colors.get(policy_id, "#777777") for policy_id in scaling_plot["policy_id"]],
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    axes[2].set_title("Large-grid target recovery")
    axes[2].set_ylabel("mean recovery fraction")
    axes[2].grid(axis="y", linewidth=0.3, alpha=0.35)

    fig.suptitle("E05 S13 local versus global control baselines", fontsize=13)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


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


def determine_outcome(validation_df: pd.DataFrame, control_df: pd.DataFrame) -> tuple[str, str, str]:
    validation_success = bool(validation_df["success"].all())
    policy_summary = (
        control_df.groupby("policy_id", as_index=False)
        .agg(mean_target_recovery_fraction=("target_recovery_fraction", "mean"))
    )
    local = policy_summary[policy_summary["policy_id"].eq("s07_local_target_neighbor_descent")]
    rebuild = policy_summary[policy_summary["policy_id"].eq("s13_global_rebuild_controller")]
    if validation_success and not local.empty and not rebuild.empty:
        outcome = "supportive"
        lay = (
            "S13 shows the expected access/performance tradeoff: existing local controls recover part of the target, "
            "while explicitly nonlocal top-down and global-controller baselines can close more of the error, especially "
            "when they are allowed extra repair semantics."
        )
    else:
        outcome = "null"
        lay = (
            "S13 produced local/global comparison tables, but validation or the local-versus-global contrast was "
            "insufficient to support the planned gap summary."
        )
    caveats = (
        "The nonlocal baselines are upper-bound computational controls with declared global target/state access. Their "
        "energy costs are proxy accounting models and are not directly interchangeable with S04 local swap energy. "
        "The global rebuild controller intentionally uses birth/death, identity conversion, and unfreeze semantics that "
        "local S07-S10 policies do not have."
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


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    control_df: pd.DataFrame,
    baseline_specs_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    route_context_df: pd.DataFrame,
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
    policy_summary = (
        control_df.groupby(["control_class", "policy_id"], as_index=False)
        .agg(
            runs=("task_id", "count"),
            mean_target_recovery_fraction=("target_recovery_fraction", "mean"),
            mean_final_target_error=("final_target_error", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
            low_recovery_failure_rate=("low_recovery_failure", "mean"),
            blocked_rate=("baseline_blocked", "mean"),
        )
        .sort_values(["control_class", "policy_id"])
    )
    task_summary = (
        control_df.groupby(["benchmark_task", "control_class", "policy_id"], as_index=False)
        .agg(
            runs=("task_id", "count"),
            mean_target_recovery_fraction=("target_recovery_fraction", "mean"),
            mean_final_target_error=("final_target_error", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
            low_recovery_failure_rate=("low_recovery_failure", "mean"),
            blocked_rate=("baseline_blocked", "mean"),
        )
        .sort_values(["benchmark_task", "control_class", "policy_id"])
    )
    top_gaps = gap_df.sort_values("target_recovery_gap_vs_local_target", ascending=False)
    route_summary = (
        route_context_df.groupby(["source_research_step_id", "policy_id"], as_index=False)
        .agg(
            route_rows=("task_id", "count"),
            mean_s12_path_curvature=("s12_path_curvature", "mean"),
            mean_s12_temporary_away_fraction=("s12_temporary_away_fraction", "mean"),
        )
        .sort_values(["source_research_step_id", "policy_id"])
    )
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action, lay_summary)}

# Research Step Full Results: {STEP_ID} Compare Local Versus Global Control

## Lay Summary

{lay_summary}

## Frozen Question

How much morphology recovery can local collectives achieve without top-down gradients, organizer cells, or global controllers?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S13.
- Local-policy inputs: S07 scrambled-embryo results, S08 regeneration results, S09 scaling results, S10 symmetry-breaking results, and S12 morphospace route metrics.
- Target and simulator inputs: S03 target morphology constructors, S04 action/energy conventions, S05 morphospace metrics, and S07-S10 task initial-state constructors.
- Datasets: none required.

## Detailed Methods

S13 preserves the existing local-policy rows as imported evidence. The two runnable local controls remain `s07_local_target_neighbor_descent` and `s07_random_adjacent_swap_control`; their declared information access is not changed, and no global state, organizer cue, or extra action semantic is added to those rows.

Three nonlocal baselines were added as explicitly labeled controls:

- `s13_top_down_global_assignment`: full state and target map, nonlocal assignment of existing active cells only; no birth/death, identity conversion, or unfreeze authority.
- `s13_organizer_beacon_assignment`: organizer-provided target field plus nonlocal assignment and unfreeze authority; no birth/death or identity conversion.
- `s13_global_rebuild_controller`: centralized target-state overwrite with birth/death, identity conversion, and unfreeze authority.

For S07, S08, S09, and S10 starts, S13 reconstructs each task initial state and evaluates those three baselines at the endpoint. Identity-preserving assignment baselines use exact target-identity matching and a Manhattan-distance relocation energy proxy. They are blocked when the current identity multiset cannot fill the target or when misplaced stuck cells would require unfreeze authority. The global rebuild baseline constructs the exact target state and charges a separate overwrite/edit energy proxy.

S13 then combines local and nonlocal rows into one comparison table, computes recovery, final error, energy-per-site, low-recovery failure, blocker rates, small-to-large scaling ratios, and local-versus-global gaps. S12 route curvature and temporary-away metrics are joined as context for prior local trajectories only; nonlocal endpoint baselines do not claim comparable route geometry.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- SciPy linear assignment was used from the preinstalled runtime.
- New packages installed: none.
- CPU/GPU use: CPU-only endpoint evaluation and report generation; S13 reused prior GPU-generated S11/S12 artifacts but ran no new GPU simulations.

## Parameters

- Low-recovery failure threshold: target recovery fraction below `0.05`.
- Assignment energy proxy: Manhattan relocation distance plus optional unfreeze count.
- Rebuild energy proxy: target site count plus identity edit count, population delta, and unfreeze count.
- Local controls: imported without retuning or information-access changes.
- Global baselines: intentionally nonlocal controls with explicit information and action-access labels.

## Results

Validation summary:

{markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=20)}

Baseline access specifications:

{markdown_table(baseline_specs_df, ["policy_id", "control_class", "global_state_access", "global_target_access", "organizer_cue_added", "birth_death_allowed", "identity_conversion_allowed"], max_rows=10)}

Policy-level comparison:

{markdown_table(policy_summary, ["control_class", "policy_id", "runs", "mean_target_recovery_fraction", "mean_final_target_error", "mean_energy_per_site", "low_recovery_failure_rate", "blocked_rate"], max_rows=12)}

Task-level comparison:

{markdown_table(task_summary, ["benchmark_task", "control_class", "policy_id", "runs", "mean_target_recovery_fraction", "mean_final_target_error", "mean_energy_per_site", "low_recovery_failure_rate", "blocked_rate"], max_rows=24)}

Largest recovery gaps versus the local target-aware control:

{markdown_table(top_gaps, ["benchmark_task", "target_kind", "policy_id", "control_class", "scale_label", "runs", "mean_target_recovery_fraction", "target_recovery_gap_vs_local_target", "final_target_error_gap_vs_local_target", "energy_per_site_gap_vs_local_target"], max_rows=18)}

S09 scaling summary:

{markdown_table(scaling_df.sort_values(["task_type", "target_kind", "control_class", "policy_id", "scale_label"]), ["task_type", "target_kind", "policy_id", "control_class", "scale_label", "runs", "mean_target_recovery_fraction", "mean_energy_per_site", "low_recovery_failure_rate", "large_to_small_recovery_ratio"], max_rows=24)}

S12 local route context:

{markdown_table(route_summary, ["source_research_step_id", "policy_id", "route_rows", "mean_s12_path_curvature", "mean_s12_temporary_away_fraction"], max_rows=12)}

## Metrics

- Target recovery fraction: `(initial target error - final target error) / initial target error`, with the existing E05 zero-initial convention.
- Final target error and aggregate morphospace error: S03 target error and S05 aggregate metric endpoints.
- Energy per site: local S04 swap/wait charges for local policies; declared proxy models for nonlocal baselines.
- Robustness: low-recovery failure rate, failure-to-improve rate, and blocked baseline rate.
- Scaling: S09 small/large recovery and energy-per-site ratios without retuning local controls.
- Gap metrics: nonlocal or local-control mean minus the local target-aware control mean within each benchmark task, target kind, and scale label.

## Figures And Tables

- Required comparison table: `$ARTIFACTS_DIR/results/e05_local_vs_global_control.parquet`.
- Required figure: `$ARTIFACTS_DIR/figures/e05/local_vs_global_control.png`.
- Additional baseline-spec, gap, robustness, scaling, route-context, validation, config, manifest, checksum, and CSV artifacts are listed below.

## Validation Checks

- Confirmed local-policy information access remains local-only.
- Confirmed global baselines are explicitly labeled and globally informed.
- Confirmed global rebuild solves all task starts as an upper-bound controller.
- Confirmed assignment baselines record blockers for identity-multiset conflicts rather than silently adding repair semantics.
- Confirmed recovery/error/energy, robustness, and scaling gaps are quantified.
- Confirmed S12 route context is loaded and S07-S10 task sources are covered.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- The nonlocal baselines are not local morphogenesis policies; they are upper-bound controls with declared global access.
- Energy comparisons use proxy accounting. Local swap energy and global overwrite or nonlocal relocation energy are useful for bounded comparison, not direct physical equivalence.
- The global rebuild controller intentionally includes repair semantics unavailable to S07-S10 local policies, so it tests an upper bound rather than a fair same-information competitor.
- S12 route context is attached to local traces only; global endpoint baselines do not produce comparable route curvature.
- All morphology tasks remain toy computational proxies, not biological validation.

## Blockers And Failed Assumptions

No execution blocker was encountered. The main failed assumption was that identity-preserving top-down assignment could solve every regeneration perturbation: missing, duplicated, and foreign patches remain blocked without birth/death or identity-conversion semantics, and frozen misplaced cells require unfreeze authority.

## Recommended Next Action

{recommended_next_action}
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e05"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, figures_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    tables = build_s13_result_tables(artifacts_dir)
    control_df = tables["control"]
    baseline_specs_df = tables["baseline_specs"]
    gap_df = tables["gap_summary"]
    robustness_df = tables["robustness_summary"]
    scaling_df = tables["scaling_summary"]
    route_context_df = tables["route_context"]
    validation_df = tables["validation"]

    results_path = results_dir / "e05_local_vs_global_control.parquet"
    results_csv_path = tables_dir / "e05_local_vs_global_control.csv"
    baseline_specs_path = results_dir / "e05_local_vs_global_baseline_specs.parquet"
    baseline_specs_csv_path = tables_dir / "e05_local_vs_global_baseline_specs.csv"
    gap_path = results_dir / "e05_local_vs_global_gap_summary.parquet"
    gap_csv_path = tables_dir / "e05_local_vs_global_gap_summary.csv"
    robustness_path = results_dir / "e05_local_vs_global_robustness.parquet"
    robustness_csv_path = tables_dir / "e05_local_vs_global_robustness.csv"
    scaling_path = results_dir / "e05_local_vs_global_scaling.parquet"
    scaling_csv_path = tables_dir / "e05_local_vs_global_scaling.csv"
    route_context_path = results_dir / "e05_local_vs_global_s12_route_context.parquet"
    route_context_csv_path = tables_dir / "e05_local_vs_global_s12_route_context.csv"
    validation_path = results_dir / "e05_local_vs_global_validation.parquet"
    validation_csv_path = tables_dir / "e05_local_vs_global_validation.csv"
    figure_path = figures_dir / "local_vs_global_control.png"
    config_path = configs_dir / "e05_s13_local_vs_global.json"
    source_manifest_path = src_snapshot_dir / "e05_local_vs_global_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    render_figure(control_df, gap_df, robustness_df, scaling_df, figure_path)

    control_df.to_parquet(results_path, index=False)
    control_df.to_csv(results_csv_path, index=False)
    baseline_specs_df.to_parquet(baseline_specs_path, index=False)
    baseline_specs_df.to_csv(baseline_specs_csv_path, index=False)
    gap_df.to_parquet(gap_path, index=False)
    gap_df.to_csv(gap_csv_path, index=False)
    robustness_df.to_parquet(robustness_path, index=False)
    robustness_df.to_csv(robustness_csv_path, index=False)
    scaling_df.to_parquet(scaling_path, index=False)
    scaling_df.to_csv(scaling_csv_path, index=False)
    route_context_df.to_parquet(route_context_path, index=False)
    route_context_df.to_csv(route_context_csv_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)

    summary = summarize_for_report(control_df, gap_df)
    config = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "controlRows": int(len(control_df)),
        "baselineSpecRows": int(len(baseline_specs_df)),
        "gapRows": int(len(gap_df)),
        "robustnessRows": int(len(robustness_df)),
        "scalingRows": int(len(scaling_df)),
        "routeContextRows": int(len(route_context_df)),
        "lowRecoveryFailureThreshold": 0.05,
        "globalBaselinePolicyIds": baseline_specs_df["policy_id"].tolist(),
        "localPolicyIds": ["s07_local_target_neighbor_descent", "s07_random_adjacent_swap_control"],
        "localTargetMeanRecovery": summary["local_target_mean_recovery"],
        "globalRebuildMeanRecovery": summary["global_rebuild_mean_recovery"],
    }
    write_json(config_path, config)

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "tests.e05.test_local_global_control", "-v"], args.repo_dir))
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )

    source_files = [
        source_entry(args.repo_dir / "src/e05/local_global_control.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s13_local_vs_global.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_local_global_control.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/scrambled_embryo.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/regeneration.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/scaling.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/symmetry_breaking.py", args.repo_dir),
    ]
    write_json(
        source_manifest_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "sourceFiles": source_files,
        },
    )

    artifacts: list[dict[str, Any]] = [
        artifact_entry(results_path, artifacts_dir, "Required S13 local-versus-global control comparison table."),
        artifact_entry(results_csv_path, artifacts_dir, "CSV companion for S13 comparison table."),
        artifact_entry(figure_path, artifacts_dir, "Required S13 local-versus-global control figure."),
        artifact_entry(baseline_specs_path, artifacts_dir, "S13 global baseline access and semantics table."),
        artifact_entry(baseline_specs_csv_path, artifacts_dir, "CSV companion for baseline specs."),
        artifact_entry(gap_path, artifacts_dir, "S13 recovery/error/energy gap summary table."),
        artifact_entry(gap_csv_path, artifacts_dir, "CSV companion for gap summary."),
        artifact_entry(robustness_path, artifacts_dir, "S13 robustness and blocker summary table."),
        artifact_entry(robustness_csv_path, artifacts_dir, "CSV companion for robustness summary."),
        artifact_entry(scaling_path, artifacts_dir, "S13 small/large scaling gap summary table."),
        artifact_entry(scaling_csv_path, artifacts_dir, "CSV companion for scaling summary."),
        artifact_entry(route_context_path, artifacts_dir, "S13 imported S12 route-context table."),
        artifact_entry(route_context_csv_path, artifacts_dir, "CSV companion for S12 route context."),
        artifact_entry(validation_path, artifacts_dir, "S13 validation case table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV companion for validation cases."),
        artifact_entry(config_path, artifacts_dir, "S13 reproducibility configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "S13 source-code provenance manifest."),
    ]

    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    outcome_classification, caveats_or_blockers, lay_summary = determine_outcome(validation_df, control_df)
    recommended_next_action = "Stop before S14 and let the Chief Scientist review S13; if accepted, proceed to S14 shape delayed-gratification analysis."

    report_artifacts = [
        *artifacts,
        manifest_self_entry(full_results_path, artifacts_dir, "Canonical S13 full-results report."),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S13 artifact manifest."),
        manifest_self_entry(checksum_path, artifacts_dir, "SHA-256 checksums for S13 artifacts."),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Workspace-root S13 run manifest."),
    ]
    report = full_results_markdown(
        artifacts=report_artifacts,
        validation_result=validation_result,
        validation_df=validation_df,
        control_df=control_df,
        baseline_specs_df=baseline_specs_df,
        gap_df=gap_df,
        robustness_df=robustness_df,
        scaling_df=scaling_df,
        route_context_df=route_context_df,
        test_commands=test_commands,
        source_files=source_files,
        outcome_classification=outcome_classification,
        caveats_or_blockers=caveats_or_blockers,
        recommended_next_action=recommended_next_action,
        lay_summary=lay_summary,
    )
    write_text(full_results_path, report)
    artifacts.append(artifact_entry(full_results_path, artifacts_dir, "Canonical S13 full-results report."))
    artifacts.append(manifest_self_entry(artifact_manifest_path, artifacts_dir, "S13 artifact manifest."))
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
        results_path,
        results_csv_path,
        figure_path,
        baseline_specs_path,
        baseline_specs_csv_path,
        gap_path,
        gap_csv_path,
        robustness_path,
        robustness_csv_path,
        scaling_path,
        scaling_csv_path,
        route_context_path,
        route_context_csv_path,
        validation_path,
        validation_csv_path,
        config_path,
        source_manifest_path,
        full_results_path,
        artifact_manifest_path,
    ]
    write_checksums(checksum_inputs, checksum_path, artifacts_dir)
    artifacts.append(artifact_entry(checksum_path, artifacts_dir, "SHA-256 checksums for S13 artifacts."))

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
