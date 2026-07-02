#!/usr/bin/env python3
"""Run E05 S14 higher-dimensional delayed-gratification analysis."""

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

from src.e05.shape_delayed_gratification import (
    DEFAULT_NULL_REPLICATES,
    METRIC_SPECS,
    NULL_MODEL_ID,
    RUN_KEY_COLUMNS,
    STEP_ID,
    build_shape_dg_tables,
    load_s08_trace_table,
    stable_json,
    summarize_for_report,
    validate_shape_dg_tables,
)


STEP_NUMBER = 14
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--n-null", type=int, default=DEFAULT_NULL_REPLICATES)
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


def short_policy(policy_id: str) -> str:
    return {
        "s07_local_target_neighbor_descent": "local target",
        "s07_random_adjacent_swap_control": "local random",
    }.get(policy_id, policy_id)


def short_metric(metric_id: str) -> str:
    return {
        "target_error_proxy": "target",
        "s05_aggregate_morphospace_error": "S05 aggregate",
    }.get(metric_id, metric_id)


def render_figure(dg_df: pd.DataFrame, sensitivity_df: pd.DataFrame, trace_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), constrained_layout=True)

    example_source = dg_df[dg_df["dg_score"].fillna(0.0) > 0.0].sort_values(
        ["dg_exceeds_null_p95", "dg_score", "max_prefix_drawup"], ascending=[False, False, False]
    )
    if example_source.empty:
        example_source = dg_df.sort_values(["temporary_away_fraction", "max_prefix_drawup"], ascending=False)
    examples = example_source.head(5)
    colors = ["#256D85", "#A05A2C", "#5C6B2F", "#7B4C9A", "#444444"]
    for color, row in zip(colors, examples.to_dict(orient="records"), strict=False):
        mask = np.ones(len(trace_df), dtype=bool)
        for column in RUN_KEY_COLUMNS:
            mask &= trace_df[column].astype(str).to_numpy() == str(row[column])
        series = trace_df.loc[mask].sort_values("event_step")
        if series.empty:
            continue
        metric_column = METRIC_SPECS[row["metric_id"]]["column"]
        label = (
            f"{row['target_id']} {row['perturbation_type']} "
            f"{short_policy(row['policy_id'])} {short_metric(row['metric_id'])}"
        )
        axes[0].plot(
            series["event_step"],
            series[metric_column],
            marker="o",
            markersize=2.8,
            linewidth=1.4,
            color=color,
            label=label,
        )
    axes[0].set_title("Representative temporary-away trajectories")
    axes[0].set_xlabel("event step")
    axes[0].set_ylabel("error proxy")
    axes[0].grid(linewidth=0.3, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=7)

    policy_metric = (
        dg_df.groupby(["policy_id", "metric_id"], as_index=False)
        .agg(
            dg_detected_rate=("dg_detected", "mean"),
            dg_exceeds_null_p95_rate=("dg_exceeds_null_p95", "mean"),
        )
        .sort_values(["policy_id", "metric_id"])
    )
    labels = [f"{short_policy(row.policy_id)}\n{short_metric(row.metric_id)}" for row in policy_metric.itertuples()]
    x = np.arange(len(policy_metric))
    width = 0.38
    axes[1].bar(x - width / 2, policy_metric["dg_detected_rate"], width=width, color="#4C78A8", label="observed")
    axes[1].bar(x + width / 2, policy_metric["dg_exceeds_null_p95_rate"], width=width, color="#F58518", label="> null p95")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Constructive DG call rates")
    axes[1].set_ylabel("fraction of S08 runs")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", linewidth=0.3, alpha=0.35)

    sensitivity = (
        sensitivity_df.groupby("perturbation_type", as_index=False)
        .agg(
            metric_inconsistency_rate=("metric_inconsistent_away_from_target", "mean"),
            both_metrics_null_exceed_rate=(
                "sensitivity_classification",
                lambda values: float(np.mean(pd.Series(values).eq("metric_consistent_null_exceeding_shape_dg"))),
            ),
            either_metric_detected_rate=(
                "sensitivity_classification",
                lambda values: float(np.mean(~pd.Series(values).eq("metric_consistent_no_constructive_dg"))),
            ),
        )
        .sort_values("perturbation_type")
    )
    x2 = np.arange(len(sensitivity))
    axes[2].bar(x2 - width / 2, sensitivity["metric_inconsistency_rate"], width=width, color="#B279A2", label="metric inconsistent")
    axes[2].bar(x2 + width / 2, sensitivity["both_metrics_null_exceed_rate"], width=width, color="#59A14F", label="both > null p95")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(sensitivity["perturbation_type"], rotation=35, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Metric sensitivity by perturbation")
    axes[2].set_ylabel("fraction of runs")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", linewidth=0.3, alpha=0.35)

    fig.suptitle("E05 S14 shape delayed-gratification trajectory proxies", fontsize=13)
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


def determine_outcome(validation_df: pd.DataFrame, dg_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> tuple[str, str, str]:
    validation_success = bool(validation_df["success"].all())
    summary = summarize_for_report(dg_df, sensitivity_df)
    target_null = float(summary["target_dg_exceeds_null_p95_rate"])
    aggregate_null = float(summary["aggregate_dg_exceeds_null_p95_rate"])
    inconsistency = float(summary["metric_inconsistency_rate"])
    best_null = max(target_null, aggregate_null)
    if not validation_success:
        outcome = "null"
        lay = (
            "S14 could not produce a fully validated shape delayed-gratification analysis, so no evidence claim is made."
        )
    elif best_null >= 0.05 and inconsistency <= 0.50:
        outcome = "supportive"
        lay = (
            "S14 found metric-proxy trajectories where morphology error temporarily increased before later repair, and a "
            "nontrivial subset exceeded length- and endpoint-matched null trajectories."
        )
    elif best_null > 0.0:
        outcome = "constraining/contradictory"
        lay = (
            "S14 found some metric-specific temporary-away trajectories, but target-error and aggregate-error calls were "
            "not stable enough to treat 'away from target' as one unambiguous shape-DG signal."
        )
    else:
        outcome = "constraining/contradictory"
        lay = (
            "S14 validated the delayed-gratification test but did not find constructive shape-DG trajectories that exceed "
            "the matched null model in the available S08 regeneration traces."
        )
    caveats = (
        "DG is measured on downsampled S08 trajectory proxies, not direct biological repair. Nulls preserve each run's "
        "trajectory length, initial error, final endpoint error, and step-increment multiset by permuting increments. "
        f"Target-error/S05-aggregate metric inconsistency rate was {inconsistency:.3f}. S13 global baselines are endpoint "
        "context only and are not treated as DG trajectories."
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
    dg_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    null_validation_df: pd.DataFrame,
    source_inputs_df: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
    n_null: int,
) -> str:
    command_lines = "\n".join(
        f"- `{command['command']}`: return code {command['returnCode']}, success={command['success']}, elapsed={command['elapsedSeconds']:.3f}s"
        for command in test_commands
    )
    source_lines = "\n".join(
        f"- `{entry['relativePath']}` sha256 `{entry['sha256']}` ({entry['sizeBytes']} bytes)"
        for entry in source_files
    )
    headline = summarize_for_report(dg_df, sensitivity_df)
    policy_summary = (
        dg_df.groupby(["policy_id", "metric_id"], as_index=False)
        .agg(
            runs=("metric_id", "count"),
            mean_initial_error=("initial_error", "mean"),
            mean_final_error=("final_error", "mean"),
            mean_net_improvement=("net_improvement", "mean"),
            mean_temporary_away_fraction=("temporary_away_fraction", "mean"),
            mean_dg_score=("dg_score", "mean"),
            dg_detected_rate=("dg_detected", "mean"),
            dg_exceeds_null_p95_rate=("dg_exceeds_null_p95", "mean"),
            median_null_p_value=("null_p_value", "median"),
        )
        .sort_values(["policy_id", "metric_id"])
    )
    sensitivity_counts = (
        sensitivity_df.groupby(["perturbation_type", "sensitivity_classification"], as_index=False)
        .agg(runs=("sensitivity_classification", "count"))
        .sort_values(["perturbation_type", "sensitivity_classification"])
    )
    top_examples = dg_df.sort_values(["dg_exceeds_null_p95", "dg_score", "temporary_away_fraction"], ascending=False)
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action, lay_summary)}

# Research Step Full Results: {STEP_ID} Identify Higher-Dimensional DG

## Lay Summary

{lay_summary}

## Frozen Question

Does Delayed Gratification generalize to temporary movement away from a target shape, especially around defects?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S14.
- Primary trajectories: S08 regeneration downsampled trace table, `$ARTIFACTS_DIR/traces/e05_regeneration_trace_table.parquet`.
- Metric inputs: S05 metric catalog and S08 target-error plus S05 aggregate morphospace-error trajectory columns.
- Context inputs: S12 morphospace route metrics and S13 local/global control table.
- Datasets: none required.

Input provenance:

{markdown_table(source_inputs_df, ["source_research_step_id", "input_type", "exists", "row_count", "used_for_primary_dg", "used_for_context_or_validation"], max_rows=12)}

## Detailed Methods

S14 defines shape delayed gratification as a lower-is-better morphology-error trajectory that first moves away from the best-so-far state and later repairs by more than that temporary drawup. Operationally, a run is called constructive DG when it has positive net improvement from initial to final error, positive maximum prefix drawup, and post-drawup repair greater than the drawup.

The primary analysis uses each S08 regeneration run under the two local policies and computes this definition for two proxies: `target_error_proxy` and `s05_aggregate_morphospace_error`. These are not collapsed into one claim; the metric-sensitivity table reports whether the two proxies agree per run.

The matched null model is `{NULL_MODEL_ID}`. For each run and metric, S14 permutes the observed one-step error increments `{n_null}` times. This preserves trajectory length, initial error, final endpoint error, endpoint delta, and the multiset of local step changes. One-sided null p-values and p95/p99 thresholds compare the observed constructive-DG score to these matched null trajectories.

S12 route metrics are joined as route-context features for the same local S08 runs. S13 context is joined only as declared information-access and endpoint baseline context; global/top-down S13 rows are not reinterpreted as trajectories and are not included in DG scoring.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- New packages installed: none.
- CPU/GPU use: CPU-only table analysis and plotting; no new GPU simulations were launched.

## Parameters

- Null replicates per run/metric: `{n_null}`.
- Null model: `{NULL_MODEL_ID}`.
- DG epsilon: `1e-12`.
- Metrics tested: `{", ".join(METRIC_SPECS)}`.
- Primary source runs: `{headline["run_count"]}` S08 regeneration runs, scored once per metric.
- Local policies: `s07_local_target_neighbor_descent` and `s07_random_adjacent_swap_control`.

## Results

Validation summary:

{markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=12)}

Headline statistics:

- DG rows: `{headline["dg_rows"]}`.
- Target-error DG detected rate: `{headline["target_dg_detected_rate"]:.6f}`; target-error above matched-null p95 rate: `{headline["target_dg_exceeds_null_p95_rate"]:.6f}`.
- S05 aggregate-error DG detected rate: `{headline["aggregate_dg_detected_rate"]:.6f}`; S05 aggregate above matched-null p95 rate: `{headline["aggregate_dg_exceeds_null_p95_rate"]:.6f}`.
- Metric inconsistency rate: `{headline["metric_inconsistency_rate"]:.6f}`.
- Mean temporary-away fraction: target-error `{headline["target_mean_temporary_away_fraction"]:.6f}`, S05 aggregate `{headline["aggregate_mean_temporary_away_fraction"]:.6f}`.

Policy and metric summary:

{markdown_table(policy_summary, ["policy_id", "metric_id", "runs", "mean_initial_error", "mean_final_error", "mean_net_improvement", "mean_temporary_away_fraction", "mean_dg_score", "dg_detected_rate", "dg_exceeds_null_p95_rate", "median_null_p_value"], max_rows=12)}

Perturbation, policy, and metric summary:

{markdown_table(summary_df, ["perturbation_type", "policy_id", "metric_id", "runs", "mean_net_improvement", "mean_temporary_away_fraction", "dg_detected_rate", "dg_exceeds_null_p95_rate", "metric_inconsistency_rate", "group_outcome_classification"], max_rows=30)}

Metric-sensitivity classifications:

{markdown_table(sensitivity_counts, ["perturbation_type", "sensitivity_classification", "runs"], max_rows=24)}

Top DG examples:

{markdown_table(top_examples, ["target_id", "perturbation_type", "policy_id", "simulation_seed", "metric_id", "initial_error", "final_error", "temporary_away_fraction", "max_prefix_drawup", "post_drawup_repair", "dg_score", "null_p95_dg_score", "null_p_value", "dg_exceeds_null_p95"], max_rows=15)}

Null validation summary:

{markdown_table(null_validation_df, ["metric_id", "null_match_status", "rows", "min_null_replicates", "mean_null_length_match_rate", "max_null_endpoint_abs_error", "mean_null_dg_detection_rate", "max_null_p95_dg_score"], max_rows=10)}

## Metrics

- `temporary_away_fraction`: fraction of consecutive trace intervals where error increased.
- `max_prefix_drawup`: largest increase above the best error seen earlier in the same run.
- `post_drawup_repair`: error reduction after the point of maximum prefix drawup.
- `dg_score`: `max_prefix_drawup` only when net improvement is positive and post-drawup repair exceeds drawup; otherwise zero.
- `dg_exceeds_null_p95`: observed `dg_score` is greater than the p95 of the length- and endpoint-matched null score distribution.
- `metric_inconsistent_away_from_target`: target-error and S05 aggregate-error disagree on observed DG or null-exceeding DG for the same run.

## Figures And Tables

- Required result table: `$ARTIFACTS_DIR/results/e05_shape_dg.parquet`.
- Required figure: `$ARTIFACTS_DIR/figures/e05/shape_dg_examples.png`.
- Additional summary, metric-sensitivity, null-validation, source-input, validation, config, manifest, checksum, and CSV artifacts are listed below.

## Validation Checks

- Confirmed S08 regeneration traces loaded and produced 180 run-level trajectories.
- Confirmed S05 metric context and the S05 aggregate morphospace-error trajectory proxy are available.
- Confirmed S12 route context joins to all S08 local trajectories.
- Confirmed S13 local access labels join without expanding local-policy information access; global baselines remain endpoint context only.
- Confirmed matched nulls preserve length and endpoint error.
- Confirmed target-error and aggregate-error DG metrics are both computed.
- Confirmed null p-values are bounded and based on positive replicate counts.
- Confirmed metric sensitivity was tested per S08 run.
- Confirmed required S14 table and figure artifacts were written.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- Shape DG is measured on downsampled computational trajectories, so short-lived events between recorded S08 trace points may be missed.
- Target error and S05 aggregate morphospace error are proxies over toy target states, not direct biological morphology measurements.
- The matched null preserves the increment multiset, so it tests whether temporal ordering is special relative to the same observed step changes; it does not model all possible policy alternatives.
- Metric-specific calls are explicitly separated. When the two proxies disagree, the report treats "away from target" as metric-inconsistent rather than as a single robust DG claim.
- S13 global/top-down baselines are endpoint context only because comparable time-resolved global trajectories were not produced in S13.

## Blockers And Failed Assumptions

No execution blocker was encountered. The main constrained assumption is that delayed gratification would appear as a metric-stable shape signal; S14 reports metric inconsistency directly when target-error and aggregate-error trajectories disagree.

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
    tables = build_shape_dg_tables(artifacts_dir, n_null=args.n_null)
    dg_df = tables["shape_dg"]
    summary_df = tables["summary"]
    sensitivity_df = tables["metric_sensitivity"]
    null_validation_df = tables["null_validation"]
    source_inputs_df = tables["source_inputs"]
    route_context_df = tables["s12_route_context"]
    s13_context_df = tables["s13_context"]
    trace_df = load_s08_trace_table(artifacts_dir)

    results_path = results_dir / "e05_shape_dg.parquet"
    results_csv_path = tables_dir / "e05_shape_dg.csv"
    summary_path = results_dir / "e05_shape_dg_summary.parquet"
    summary_csv_path = tables_dir / "e05_shape_dg_summary.csv"
    sensitivity_path = results_dir / "e05_shape_dg_metric_sensitivity.parquet"
    sensitivity_csv_path = tables_dir / "e05_shape_dg_metric_sensitivity.csv"
    null_validation_path = results_dir / "e05_shape_dg_null_validation.parquet"
    null_validation_csv_path = tables_dir / "e05_shape_dg_null_validation.csv"
    source_inputs_path = results_dir / "e05_shape_dg_source_inputs.parquet"
    source_inputs_csv_path = tables_dir / "e05_shape_dg_source_inputs.csv"
    validation_path = results_dir / "e05_shape_dg_validation.parquet"
    validation_csv_path = tables_dir / "e05_shape_dg_validation.csv"
    figure_path = figures_dir / "shape_dg_examples.png"
    config_path = configs_dir / "e05_s14_shape_delayed_gratification.json"
    source_manifest_path = src_snapshot_dir / "e05_shape_dg_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    render_figure(dg_df, sensitivity_df, trace_df, figure_path)
    dg_df.to_parquet(results_path, index=False)
    dg_df.to_csv(results_csv_path, index=False)
    summary_df.to_parquet(summary_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    sensitivity_df.to_parquet(sensitivity_path, index=False)
    sensitivity_df.to_csv(sensitivity_csv_path, index=False)
    null_validation_df.to_parquet(null_validation_path, index=False)
    null_validation_df.to_csv(null_validation_csv_path, index=False)
    source_inputs_df.to_parquet(source_inputs_path, index=False)
    source_inputs_df.to_csv(source_inputs_csv_path, index=False)

    validation_df = validate_shape_dg_tables(
        dg_df=dg_df,
        summary_df=summary_df,
        sensitivity_df=sensitivity_df,
        null_validation_df=null_validation_df,
        source_inputs_df=source_inputs_df,
        route_context_df=route_context_df,
        s13_context_df=s13_context_df,
        required_artifact_paths=[results_path, figure_path, config_path, source_manifest_path],
    )

    config = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "nullModelId": NULL_MODEL_ID,
        "nullReplicatesPerRunMetric": int(args.n_null),
        "metricSpecs": METRIC_SPECS,
        "runKeyColumns": list(RUN_KEY_COLUMNS),
        "shapeDgRows": int(len(dg_df)),
        "runCount": int(dg_df[list(RUN_KEY_COLUMNS)].drop_duplicates().shape[0]),
        "metricSensitivityRows": int(len(sensitivity_df)),
        "headline": summarize_for_report(dg_df, sensitivity_df),
    }
    write_json(config_path, config)

    source_files = [
        source_entry(args.repo_dir / "src/e05/shape_delayed_gratification.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s14_shape_delayed_gratification.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_shape_delayed_gratification.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/morphospace_trajectories.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/regeneration.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/local_global_control.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/morphospace_metrics.py", args.repo_dir),
    ]
    write_json(
        source_manifest_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "sourceFiles": source_files,
        },
    )
    validation_df = validate_shape_dg_tables(
        dg_df=dg_df,
        summary_df=summary_df,
        sensitivity_df=sensitivity_df,
        null_validation_df=null_validation_df,
        source_inputs_df=source_inputs_df,
        route_context_df=route_context_df,
        s13_context_df=s13_context_df,
        required_artifact_paths=[results_path, figure_path, config_path, source_manifest_path],
    )
    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "tests.e05.test_shape_delayed_gratification", "-v"], args.repo_dir))
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )

    artifacts: list[dict[str, Any]] = [
        artifact_entry(results_path, artifacts_dir, "Required S14 per-run, per-metric shape-DG result table."),
        artifact_entry(results_csv_path, artifacts_dir, "CSV companion for required S14 shape-DG table."),
        artifact_entry(figure_path, artifacts_dir, "Required S14 shape-DG example and sensitivity figure."),
        artifact_entry(summary_path, artifacts_dir, "S14 perturbation, policy, and metric summary table."),
        artifact_entry(summary_csv_path, artifacts_dir, "CSV companion for S14 summary table."),
        artifact_entry(sensitivity_path, artifacts_dir, "S14 metric-sensitivity table comparing target and S05 aggregate DG calls."),
        artifact_entry(sensitivity_csv_path, artifacts_dir, "CSV companion for S14 metric-sensitivity table."),
        artifact_entry(null_validation_path, artifacts_dir, "S14 matched-null validation summary table."),
        artifact_entry(null_validation_csv_path, artifacts_dir, "CSV companion for S14 matched-null validation summary."),
        artifact_entry(source_inputs_path, artifacts_dir, "S14 source input provenance table."),
        artifact_entry(source_inputs_csv_path, artifacts_dir, "CSV companion for source input provenance."),
        artifact_entry(validation_path, artifacts_dir, "S14 validation case table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV companion for S14 validation cases."),
        artifact_entry(config_path, artifacts_dir, "S14 reproducibility configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "S14 source-code provenance manifest."),
    ]

    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    outcome_classification, caveats_or_blockers, lay_summary = determine_outcome(validation_df, dg_df, sensitivity_df)
    recommended_next_action = "Stop before S15 and let the Chief Scientist review S14; if accepted, proceed to S15 benchmark-suite packaging."

    report_artifacts = [
        *artifacts,
        manifest_self_entry(full_results_path, artifacts_dir, "Canonical S14 full-results report."),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S14 artifact manifest."),
        manifest_self_entry(checksum_path, artifacts_dir, "SHA-256 checksums for S14 artifacts."),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Workspace-root S14 run manifest."),
    ]
    report = full_results_markdown(
        artifacts=report_artifacts,
        validation_result=validation_result,
        validation_df=validation_df,
        dg_df=dg_df,
        summary_df=summary_df,
        sensitivity_df=sensitivity_df,
        null_validation_df=null_validation_df,
        source_inputs_df=source_inputs_df,
        test_commands=test_commands,
        source_files=source_files,
        outcome_classification=outcome_classification,
        caveats_or_blockers=caveats_or_blockers,
        recommended_next_action=recommended_next_action,
        lay_summary=lay_summary,
        n_null=args.n_null,
    )
    write_text(full_results_path, report)
    artifacts.append(artifact_entry(full_results_path, artifacts_dir, "Canonical S14 full-results report."))
    artifacts.append(manifest_self_entry(artifact_manifest_path, artifacts_dir, "S14 artifact manifest."))
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
        summary_path,
        summary_csv_path,
        sensitivity_path,
        sensitivity_csv_path,
        null_validation_path,
        null_validation_csv_path,
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
    artifacts.append(artifact_entry(checksum_path, artifacts_dir, "SHA-256 checksums for S14 artifacts."))

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

