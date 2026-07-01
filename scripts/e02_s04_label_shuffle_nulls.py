#!/usr/bin/env python3
"""Run E02 S04 trajectory-preserving Algotype label-shuffle nulls.

This step uses E01 same-goal chimera trajectories, keeps each recorded
position/value trajectory row fixed, and recomputes Aggregation nulls after
randomly permuting the Algotype label-position code with label counts preserved.
It intentionally stops before S05 dummy-Algotype controls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.e02_s02_scheduler_comparison import artifact_entry, markdown_table, run_command, sha256_file, write_json


STEP_ID = "S04"
STEP_NUMBER = 4
EXPERIMENT_ID = "E02"
TRACE_PATH_DEFAULT = Path("/previous-artifacts/E01/traces/e01_same_goal_chimeras.parquet")
SELECTED_CONDITION_IDS = ("E01C043", "E01C044", "E01C045", "E01C046", "E01C047")
REQUIRED_TRACE_COLUMNS = (
    "condition_id",
    "algotype_mix",
    "repeat_index",
    "swap_step",
    "aggregation_left_neighbor_percent",
    "algotype_positions_code",
    "is_final",
    "initial_array_sha256",
    "final_array_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--trace-path", type=Path, default=TRACE_PATH_DEFAULT)
    parser.add_argument("--null-samples", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=2026070104)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def aggregation_left_neighbor_percent_from_code(code: str) -> float:
    if not code:
        return 0.0
    same_left = sum(1 for idx in range(1, len(code)) if code[idx] == code[idx - 1])
    return 100.0 * same_left / len(code)


def code_counts(code: str) -> dict[str, int]:
    return dict(sorted(Counter(code).items()))


def composition_key_from_counts(counts: dict[str, int]) -> tuple[int, ...]:
    return tuple(sorted(map(int, counts.values()), reverse=True))


def simulate_row_null_distribution(count_values: Sequence[int], n_samples: int, seed: int) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    base_parts = []
    for idx, count in enumerate(count_values):
        base_parts.extend([idx] * int(count))
    base = np.array(base_parts, dtype=np.int16)
    if len(base) == 0:
        raise ValueError("composition cannot be empty.")
    rng = np.random.default_rng(seed)
    matrix = np.tile(base, (int(n_samples), 1))
    matrix = rng.permuted(matrix, axis=1)
    same = (matrix[:, 1:] == matrix[:, :-1]).sum(axis=1)
    return same.astype(np.float64) * (100.0 / len(base))


def empirical_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def empirical_upper_tail(values: np.ndarray, observed: float) -> float:
    return float(np.mean(values >= observed))


def max_distribution_stats(row_null_values: np.ndarray, row_count: int, observed_peak: float) -> dict[str, float]:
    values, counts = np.unique(row_null_values, return_counts=True)
    probs = counts.astype(np.float64) / float(len(row_null_values))
    cdf = np.cumsum(probs)
    previous_cdf = np.concatenate([[0.0], cdf[:-1]])
    row_count = max(int(row_count), 1)
    max_probs = np.power(cdf, row_count) - np.power(previous_cdf, row_count)
    max_probs = max_probs / max_probs.sum()
    max_cdf = np.cumsum(max_probs)
    f_less = float(np.sum(probs[values < observed_peak]))
    p_upper = 1.0 - (f_less**row_count)
    return {
        "null_peak_mean": float(np.sum(values * max_probs)),
        "null_peak_q025": float(values[np.searchsorted(max_cdf, 0.025, side="left")]),
        "null_peak_q50": float(values[np.searchsorted(max_cdf, 0.50, side="left")]),
        "null_peak_q975": float(values[np.searchsorted(max_cdf, 0.975, side="left")]),
        "peak_empirical_p_upper": float(p_upper),
    }


def normal_upper_tail(observed: float, mean: float, sd: float) -> float:
    if not np.isfinite(sd) or sd <= 0:
        return 1.0 if observed <= mean else 0.0
    z = (observed - mean) / sd
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def normal_quantiles(mean: float, sd: float) -> tuple[float, float]:
    if not np.isfinite(sd) or sd <= 0:
        return float(mean), float(mean)
    z975 = 1.959963984540054
    return float(mean - z975 * sd), float(mean + z975 * sd)


def validate_trace_columns(trace_path: Path) -> tuple[bool, list[str], list[str]]:
    if not trace_path.exists():
        return False, list(REQUIRED_TRACE_COLUMNS), []
    try:
        columns = list(pq.read_schema(trace_path).names)
    except Exception:
        df_head = pd.read_parquet(trace_path).head(0)
        columns = list(df_head.columns)
    missing = [col for col in REQUIRED_TRACE_COLUMNS if col not in columns]
    return not missing, missing, columns


def load_trace(trace_path: Path) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            [
                *REQUIRED_TRACE_COLUMNS,
                "research_step_id",
                "experiment_id",
                "mode",
                "algorithm",
                "baseline_source",
                "matched_group_id",
                "value_bank_id",
                "algotype_assignment_bank_id",
                "configured_algotype_counts_json",
                "initial_algotype_counts_json",
                "final_algotype_counts_json",
                "sortedness_percent",
                "monotonicity_error_count",
                "aggregation_right_neighbor_legacy_percent",
                "stop_reason",
                "max_guard_hit",
                "final_values_json",
            ]
        )
    )
    available = list(pq.read_schema(trace_path).names)
    use_columns = [col for col in columns if col in available]
    df = pd.read_parquet(trace_path, columns=use_columns)
    df = df[df["condition_id"].isin(SELECTED_CONDITION_IDS)].copy()
    if df.empty:
        raise RuntimeError("S04 trace input has no selected same-goal chimera conditions.")
    df["swap_step"] = pd.to_numeric(df["swap_step"], errors="coerce").fillna(0).astype(np.int64)
    return df.sort_values(["condition_id", "repeat_index", "swap_step"]).reset_index(drop=True)


def build_row_nulls(trace: pd.DataFrame, null_samples: int, seed: int) -> dict[tuple[int, ...], np.ndarray]:
    keys: set[tuple[int, ...]] = set()
    for code in trace.groupby(["condition_id", "repeat_index"], sort=False)["algotype_positions_code"].first():
        keys.add(composition_key_from_counts(code_counts(str(code))))
    row_nulls: dict[tuple[int, ...], np.ndarray] = {}
    for idx, key in enumerate(sorted(keys, reverse=True), start=1):
        row_nulls[key] = simulate_row_null_distribution(key, null_samples, seed + idx * 1009)
    return row_nulls


def analyze_trace(
    trace: pd.DataFrame,
    row_nulls: dict[tuple[int, ...], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    recomputed = trace["algotype_positions_code"].map(lambda code: aggregation_left_neighbor_percent_from_code(str(code)))
    max_abs_recompute_error = float((recomputed - trace["aggregation_left_neighbor_percent"]).abs().max())
    code_lengths = trace["algotype_positions_code"].map(lambda code: len(str(code)))
    code_lengths_all_100 = bool((code_lengths == 100).all())
    run_rows: list[dict[str, Any]] = []
    curve_accum: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(lambda: {"sum": 0.0, "n": 0, "composition_key": None})

    group_cols = ["condition_id", "algotype_mix", "repeat_index"]
    for (condition_id, mix, repeat_index), group in trace.groupby(group_cols, sort=True):
        group = group.sort_values("swap_step")
        first = group.iloc[0]
        final_rows = group[group["is_final"].fillna(False).astype(bool)]
        final = final_rows.iloc[-1] if len(final_rows) else group.iloc[-1]
        first_code = str(first["algotype_positions_code"])
        counts = code_counts(first_code)
        composition_key = composition_key_from_counts(counts)
        row_null = row_nulls[composition_key]
        row_count = int(len(group))
        observed_peak = float(group["aggregation_left_neighbor_percent"].max())
        observed_final = float(final["aggregation_left_neighbor_percent"])
        observed_curve_mean = float(group["aggregation_left_neighbor_percent"].mean())
        final_step = int(max(group["swap_step"].max(), 1))

        peak_stats = max_distribution_stats(row_null, row_count, observed_peak)
        row_mean = float(np.mean(row_null))
        row_sd = float(np.std(row_null, ddof=1))
        final_p = empirical_upper_tail(row_null, observed_final)
        curve_mean_sd = row_sd / math.sqrt(max(row_count, 1))
        curve_q025, curve_q975 = normal_quantiles(row_mean, curve_mean_sd)
        curve_p = normal_upper_tail(observed_curve_mean, row_mean, curve_mean_sd)
        run_rows.append(
            {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "source_experiment_id": "E01",
                "condition_id": condition_id,
                "algotype_mix": mix,
                "repeat_index": int(repeat_index),
                "trace_row_count": row_count,
                "final_swap_step": final_step,
                "initial_array_sha256": first.get("initial_array_sha256"),
                "final_array_sha256": final.get("final_array_sha256"),
                "configured_algotype_counts_json": first.get("configured_algotype_counts_json"),
                "initial_algotype_counts_json": first.get("initial_algotype_counts_json"),
                "final_algotype_counts_json": final.get("final_algotype_counts_json"),
                "composition_key_json": json.dumps(list(composition_key), separators=(",", ":")),
                "source_trace_path": str(TRACE_PATH_DEFAULT),
                "null_model": "independent_per_timepoint_label_permutation_preserve_counts",
                "observed_peak_aggregation_left_neighbor_percent": observed_peak,
                "observed_final_aggregation_left_neighbor_percent": observed_final,
                "observed_curve_mean_aggregation_left_neighbor_percent": observed_curve_mean,
                "null_row_mean_aggregation_left_neighbor_percent": row_mean,
                "null_row_sd_aggregation_left_neighbor_percent": row_sd,
                "null_final_q025": empirical_quantile(row_null, 0.025),
                "null_final_q50": empirical_quantile(row_null, 0.50),
                "null_final_q975": empirical_quantile(row_null, 0.975),
                "final_empirical_p_upper": final_p,
                "curve_mean_normal_q025": curve_q025,
                "curve_mean_normal_q975": curve_q975,
                "curve_mean_normal_p_upper": curve_p,
                **peak_stats,
                "peak_minus_null_peak_mean": observed_peak - peak_stats["null_peak_mean"],
                "final_minus_null_row_mean": observed_final - row_mean,
                "curve_mean_minus_null_row_mean": observed_curve_mean - row_mean,
            }
        )

        progress_bins = np.rint(100.0 * group["swap_step"].to_numpy(dtype=np.float64) / final_step).astype(int)
        progress_bins = np.clip(progress_bins, 0, 100)
        values = group["aggregation_left_neighbor_percent"].to_numpy(dtype=np.float64)
        for progress_bin in np.unique(progress_bins):
            mask = progress_bins == progress_bin
            key = (str(condition_id), str(mix), int(progress_bin))
            curve_accum[key]["sum"] += float(values[mask].sum())
            curve_accum[key]["n"] += int(mask.sum())
            curve_accum[key]["composition_key"] = composition_key

    curve_rows: list[dict[str, Any]] = []
    for (condition_id, mix, progress_bin), acc in sorted(curve_accum.items()):
        row_null = row_nulls[acc["composition_key"]]
        row_mean = float(np.mean(row_null))
        row_sd = float(np.std(row_null, ddof=1))
        n = int(acc["n"])
        observed_mean = float(acc["sum"] / n) if n else np.nan
        mean_sd = row_sd / math.sqrt(max(n, 1))
        q025, q975 = normal_quantiles(row_mean, mean_sd)
        curve_rows.append(
            {
                "condition_id": condition_id,
                "algotype_mix": mix,
                "progress_percent": int(progress_bin),
                "trace_rows_in_bin": n,
                "observed_mean_aggregation_left_neighbor_percent": observed_mean,
                "null_mean_aggregation_left_neighbor_percent": row_mean,
                "null_mean_q025": q025,
                "null_mean_q975": q975,
                "normal_p_upper": normal_upper_tail(observed_mean, row_mean, mean_sd),
                "composition_key_json": json.dumps(list(acc["composition_key"]), separators=(",", ":")),
            }
        )

    validation_metrics = {
        "maxAbsAggregationRecomputeError": max_abs_recompute_error,
        "codeLengthsAll100": code_lengths_all_100,
        "traceRowsAnalyzed": int(len(trace)),
        "runCount": int(len(run_rows)),
        "curveRows": int(len(curve_rows)),
    }
    return pd.DataFrame(run_rows), pd.DataFrame(curve_rows), validation_metrics


def summarize_results(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition_id, mix), group in run_df.groupby(["condition_id", "algotype_mix"], sort=True):
        rows.append(
            {
                "condition_id": condition_id,
                "algotype_mix": mix,
                "n_runs": int(len(group)),
                "mean_observed_peak": float(group["observed_peak_aggregation_left_neighbor_percent"].mean()),
                "mean_null_peak_mean": float(group["null_peak_mean"].mean()),
                "mean_peak_minus_null": float(group["peak_minus_null_peak_mean"].mean()),
                "median_peak_p_upper": float(group["peak_empirical_p_upper"].median()),
                "runs_peak_p_le_0_05": int((group["peak_empirical_p_upper"] <= 0.05).sum()),
                "mean_observed_final": float(group["observed_final_aggregation_left_neighbor_percent"].mean()),
                "mean_null_final": float(group["null_row_mean_aggregation_left_neighbor_percent"].mean()),
                "mean_final_minus_null": float(group["final_minus_null_row_mean"].mean()),
                "median_final_p_upper": float(group["final_empirical_p_upper"].median()),
                "mean_curve_mean_minus_null": float(group["curve_mean_minus_null_row_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_figure(run_df: pd.DataFrame, curve_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    mixes = list(run_df[["condition_id", "algotype_mix"]].drop_duplicates().itertuples(index=False, name=None))
    labels = [mix.replace("same_goal_", "").replace("same_algorithm_", "") for _, mix in mixes]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    peak_means = run_df.groupby("algotype_mix")["observed_peak_aggregation_left_neighbor_percent"].mean().reindex([m for _, m in mixes])
    peak_null = run_df.groupby("algotype_mix")["null_peak_mean"].mean().reindex([m for _, m in mixes])
    x = np.arange(len(labels))
    ax = axes[0, 0]
    ax.bar(x - 0.18, peak_means, width=0.36, label="Observed peak", color="#4c78a8")
    ax.bar(x + 0.18, peak_null, width=0.36, label="Null peak mean", color="#f58518")
    ax.set_title("Run peak Aggregation")
    ax.set_ylabel("Left-neighbor %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    final_means = run_df.groupby("algotype_mix")["observed_final_aggregation_left_neighbor_percent"].mean().reindex([m for _, m in mixes])
    final_null = run_df.groupby("algotype_mix")["null_row_mean_aggregation_left_neighbor_percent"].mean().reindex([m for _, m in mixes])
    ax.bar(x - 0.18, final_means, width=0.36, label="Observed final", color="#54a24b")
    ax.bar(x + 0.18, final_null, width=0.36, label="Null row mean", color="#e45756")
    ax.set_title("Final Aggregation")
    ax.set_ylabel("Left-neighbor %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for _, mix in mixes:
        sub = curve_df[curve_df["algotype_mix"] == mix].sort_values("progress_percent")
        ax.plot(sub["progress_percent"], sub["observed_mean_aggregation_left_neighbor_percent"], label=mix.replace("same_goal_", "").replace("same_algorithm_", ""))
    first_mix = mixes[0][1]
    ref = curve_df[curve_df["algotype_mix"] == first_mix].sort_values("progress_percent")
    ax.fill_between(ref["progress_percent"], ref["null_mean_q025"], ref["null_mean_q975"], color="#9ecae9", alpha=0.25, label="Null 95% band")
    ax.set_title("Observed curves versus label-shuffle null")
    ax.set_xlabel("Run progress %")
    ax.set_ylabel("Mean left-neighbor %")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    summary = summarize_results(run_df)
    ax.barh(
        [mix.replace("same_goal_", "").replace("same_algorithm_", "") for mix in summary["algotype_mix"]],
        -np.log10(summary["median_peak_p_upper"].clip(lower=1e-12)),
        color="#b279a2",
    )
    ax.axvline(-math.log10(0.05), color="#555555", linewidth=1, linestyle="--")
    ax.set_title("Median peak-null upper-tail evidence")
    ax.set_xlabel("-log10 median p")

    fig.suptitle("E02 S04 trajectory-preserving Algotype label-shuffle nulls", fontsize=14)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def validate_results(
    trace: pd.DataFrame,
    run_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    validation_metrics: dict[str, Any],
    figure_path: Path,
    expected_artifacts: Sequence[Path],
    null_samples: int,
) -> dict[str, Any]:
    observed_conditions = sorted(run_df["condition_id"].unique()) if len(run_df) else []
    condition_run_counts = run_df.groupby("condition_id").size().to_dict() if len(run_df) else {}
    # Label-count preservation is checked by verifying every row in a run has
    # the same code-count dictionary as that run's initial row.
    label_count_mismatches = 0
    for _, group in trace.groupby(["condition_id", "repeat_index"], sort=False):
        initial_counts = code_counts(str(group.iloc[0]["algotype_positions_code"]))
        for code in group["algotype_positions_code"]:
            if code_counts(str(code)) != initial_counts:
                label_count_mismatches += 1
                break
    label_counts_preserved = label_count_mismatches == 0
    figure_ok = figure_path.exists() and figure_path.stat().st_size > 1000
    prevalidation_artifact_names = {
        "e02_label_shuffle_aggregation_nulls.parquet",
        "aggregation_label_shuffle_nulls.png",
        "e02_label_shuffle_curve_null_summary.csv",
        "e02_label_shuffle_condition_summary.csv",
    }
    artifacts_present = all(
        path.exists() and path.stat().st_size > 0
        for path in expected_artifacts
        if path.name in prevalidation_artifact_names
    )
    passed = all(
        [
            len(run_df) == 500,
            observed_conditions == sorted(SELECTED_CONDITION_IDS),
            all(int(v) == 100 for v in condition_run_counts.values()),
            validation_metrics["maxAbsAggregationRecomputeError"] <= 1e-9,
            validation_metrics["codeLengthsAll100"],
            label_counts_preserved,
            len(curve_df) > 0,
            null_samples > 0,
            figure_ok,
            artifacts_present,
        ]
    )
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(passed),
        "status": "completed" if passed else "failed_validation",
        "validationPassed": bool(passed),
        "validationResult": "passed" if passed else "failed",
        "artifactsWritten": [str(path) for path in expected_artifacts],
        "caveatsOrBlockers": [
            "E01 same-goal chimera trace provides per-step Algotype position codes and Aggregation metrics but not full per-step value arrays.",
            "Null labels are independently permuted per recorded timepoint while preserving label counts and the recorded trajectory rows.",
        ],
        "recommendedNextAction": "Proceed to S05 behavior-preserving dummy Algotypes after Chief Scientist instruction; do not start S05 inside S04.",
        "traceRowsAnalyzed": int(validation_metrics["traceRowsAnalyzed"]),
        "runRowsWritten": int(len(run_df)),
        "curveRowsWritten": int(len(curve_df)),
        "observedConditions": observed_conditions,
        "conditionRunCounts": {str(k): int(v) for k, v in condition_run_counts.items()},
        "nullSamplesPerComposition": int(null_samples),
        "maxAbsAggregationRecomputeError": float(validation_metrics["maxAbsAggregationRecomputeError"]),
        "codeLengthsAll100": bool(validation_metrics["codeLengthsAll100"]),
        "labelCountsPreservedAcrossTraceRows": bool(label_counts_preserved),
        "labelCountMismatchRuns": int(label_count_mismatches),
        "figureExistsAndNonempty": bool(figure_ok),
        "artifactsPresent": bool(artifacts_present),
    }


def blocked_outputs(
    *,
    generated_at: str,
    artifacts_dir: Path,
    repo_dir: Path,
    trace_path: Path,
    missing_columns: list[str],
    available_columns: list[str],
) -> int:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e02"
    logs_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    for directory in (step_dir, results_dir, figures_dir, logs_dir, src_snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / "e02_label_shuffle_aggregation_nulls.parquet"
    figure_path = figures_dir / "aggregation_label_shuffle_nulls.png"
    validation_path = step_dir / "s04_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s04_label_shuffle_nulls_manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    log_path = logs_dir / "e02_s04_label_shuffle_nulls.log"
    blocker = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": False,
        "status": "blocked_missing_trace_columns",
        "validationPassed": False,
        "validationResult": "blocked",
        "artifactsWritten": [str(report_path), str(validation_path), str(manifest_path), str(src_manifest_path), str(log_path)],
        "caveatsOrBlockers": [f"Missing required trace columns: {missing_columns}", f"Trace path inspected: {trace_path}"],
        "recommendedNextAction": "Provide trajectory-level chimeric records with condition_id, repeat_index, swap_step, algotype_positions_code, and Aggregation columns before retrying S04.",
        "missingColumns": missing_columns,
        "availableColumns": available_columns,
    }
    write_json(validation_path, blocker)
    log_path.write_text(json.dumps(blocker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    src_manifest = {
        "schema": "eidosoma.src_snapshot.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "gitCommit": git_commit,
        "gitStatusShort": git_status,
        "sourceFiles": [],
        "validation": blocker,
    }
    write_json(src_manifest_path, src_manifest)
    report = f"""# E02 S04 Trajectory-Preserving Label-Shuffle Nulls

## Top Summary

- Research step ID: S04
- Completion status: Blocked on {generated_at}
- Artifacts written: `{report_path}`, `{validation_path}`, `{manifest_path}`, `{src_manifest_path}`, `{log_path}`
- Validation result: blocked: required trajectory-level columns were unavailable
- Outcome classification: constraining/contradictory
- Caveats or blockers: Missing required trace columns `{missing_columns}` in `{trace_path}`.
- Lay summary: S04 could not compute label-shuffle nulls because the required chimeric trajectory records were not available in the expected schema.
- Recommended next action: Provide trajectory-level chimeric records with condition, repeat, step, label-position code, and Aggregation columns before retrying S04; do not start S05.

## Methods

Inspected `{trace_path}` for required columns: `{list(REQUIRED_TRACE_COLUMNS)}`.

## Results

No null table or figure was generated because the input contract failed.

## Provenance

- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Generated at UTC: `{generated_at}`
"""
    report_path.write_text(report, encoding="utf-8")
    artifact_stub = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "artifacts": [
            artifact_entry(report_path, artifacts_dir, "S04 blocked full-results Markdown handoff report."),
            artifact_entry(validation_path, artifacts_dir, "S04 blocked validation evidence."),
            artifact_entry(log_path, artifacts_dir, "S04 blocked execution log."),
            artifact_entry(src_manifest_path, artifacts_dir, "S04 blocked source snapshot manifest."),
        ],
    }
    write_json(manifest_path, artifact_stub)
    return 2


def build_report(
    *,
    generated_at: str,
    artifacts_dir: Path,
    result_path: Path,
    curve_path: Path,
    summary_path: Path,
    validation_path: Path,
    log_path: Path,
    figure_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    validation: dict[str, Any],
    summary: pd.DataFrame,
    unit_result: dict[str, Any] | None,
    git_commit: str,
    git_status: str,
    null_samples: int,
    elapsed_seconds: float,
) -> str:
    validation_result = (
        f"passed: {validation['runRowsWritten']} run null rows, {validation['traceRowsAnalyzed']} trace rows, "
        f"label counts preserved={validation['labelCountsPreservedAcrossTraceRows']}, "
        f"aggregation recompute max error={validation['maxAbsAggregationRecomputeError']}, "
        f"unit tests={'passed' if unit_result and unit_result['success'] else 'not run'}"
    )
    peak_supported = int((summary["median_peak_p_upper"] <= 0.05).sum())
    outcome = "Supportive" if peak_supported else "Null"
    caveats = [
        "E01 same-goal chimera trace materializes Algotype position codes and Aggregation metrics but not full per-step value arrays.",
        "S04 uses independent per-timepoint label permutations preserving label counts, not a fixed cell-identity relabeling across an entire run.",
        "Peak p-values account for the number of recorded timepoints in a run under the independent-row null.",
    ]
    compact_summary = summary[
        [
            "condition_id",
            "algotype_mix",
            "n_runs",
            "mean_observed_peak",
            "mean_null_peak_mean",
            "mean_peak_minus_null",
            "median_peak_p_upper",
            "runs_peak_p_le_0_05",
            "mean_observed_final",
            "mean_null_final",
            "mean_final_minus_null",
        ]
    ]
    lay_summary = (
        "Observed chimeric Aggregation peaks exceeded trajectory-preserving label-shuffle null expectations for at least one condition, "
        "so label clustering is not fully explained by random relabeling on the recorded trajectories."
        if outcome == "Supportive"
        else "Observed chimeric Aggregation did not exceed trajectory-preserving label-shuffle null expectations in this S04 test."
    )
    return f"""# E02 S04 Trajectory-Preserving Label-Shuffle Nulls

## Top Summary

- Research step ID: S04
- Completion status: Completed on {generated_at}
- Artifacts written: `{artifacts_dir / 'research_steps/S04/research_step_full_results.md'}`, `{result_path}`, `{figure_path}`, `{validation_path}`, `{curve_path}`, `{summary_path}`, `{log_path}`, `{manifest_path}`, `{src_manifest_path}`
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers: {'; '.join(caveats)}
- Lay summary: {lay_summary}
- Recommended next action: Proceed to S05 behavior-preserving dummy Algotypes after Chief Scientist instruction; do not start S05 inside S04.

## Chief Handoff

S04 completed the trajectory-preserving Algotype label-shuffle null test using the E01 same-goal chimera trace. The null preserves each recorded trajectory row and Algotype label counts, then recomputes Aggregation after random label-position permutations.

## Frozen Question

Does observed Aggregation exceed a null distribution that preserves positions and values but destroys any real relation between behavior and Algotype labels?

## Inputs

- Repository commit at S04 run time: `{git_commit}`
- E01 same-goal chimera trace: `/previous-artifacts/E01/traces/e01_same_goal_chimeras.parquet`
- Selected conditions: `{', '.join(SELECTED_CONDITION_IDS)}`
- Null samples per composition: `{null_samples}`

## Methods

For each recorded trace row, S04 used the 100-character `algotype_positions_code` as the label-position trajectory state and verified that recomputed left-neighbor Aggregation matched the recorded `aggregation_left_neighbor_percent`. For each run, S04 generated empirical row-level label-shuffle null distributions that preserve label counts. Run peak nulls use the empirical row distribution and the run's recorded timepoint count to account for the peak-over-trajectory multiple opportunity effect. Final nulls use the row-level empirical distribution. Curve mean nulls use a normal approximation from the row-level null mean and variance over the number of rows in each progress bin.

## Commands

- Unit tests: `{unit_result['command'] if unit_result else 'not run'}`
- S04 production run: `{sys.executable} scripts/e02_s04_label_shuffle_nulls.py --repo-dir /workspace/cell-research --artifacts-dir {artifacts_dir} --null-samples {null_samples}`

## Dependencies And Parameters

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- New dependencies installed: none
- Elapsed production time: `{elapsed_seconds:.2f}` seconds

## Results

### Null Summary

{markdown_table(compact_summary, max_rows=10)}

The full run-level null table is in `{result_path}`. The curve-level null summary is `{curve_path}`. The label-shuffle figure is `{figure_path}`.

## Validation

Validation required all 500 selected E01 same-goal chimera runs, all five selected condition IDs, preserved label counts across all trace rows within each run, 100-position label codes, exact recomputation of recorded left-neighbor Aggregation from `algotype_positions_code`, nonempty curve summaries, positive null sample count, a nonempty figure, and present artifacts. The validation result was `{validation['validationPassed']}`.

Validation details were written to `{validation_path}`.

## Artifacts

- Run-level null table: `{result_path}`
- Curve-level null summary: `{curve_path}`
- Condition summary: `{summary_path}`
- Figure: `{figure_path}`
- Validation JSON: `{validation_path}`
- Log: `{log_path}`
- Artifact manifest: `{manifest_path}`
- Source snapshot manifest: `{src_manifest_path}`

## Provenance

- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Generated at UTC: `{generated_at}`
- Output checksums are recorded in `{manifest_path}`.

## Caveats, Blockers, And Limitations

- Full per-step value arrays are not materialized in the E01 same-goal chimera trace. S04 preserves the recorded trajectory rows and recomputes only label-based Aggregation nulls.
- Independent per-timepoint label shuffles are a metric null, not a behavioral rerun. They destroy label-position clustering at each recorded row while keeping row timing, condition, repeat, and trace-level progress fixed.
- Curve mean p-values use a normal approximation. Peak and final summaries use empirical row-level null distributions.

## Recommended Next Action

Proceed to S05 behavior-preserving dummy Algotypes after explicit Chief Scientist instruction. S05 should test whether distinct labels executing identical code can produce above-null Aggregation in an actual rerun rather than a post hoc label shuffle.
"""


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e02"
    logs_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    for directory in (step_dir, results_dir, figures_dir, logs_dir, src_snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now()
    log_lines = [f"E02 S04 started {generated_at}", f"repo_dir={repo_dir}", f"artifacts_dir={artifacts_dir}"]
    ok, missing, available = validate_trace_columns(args.trace_path)
    if not ok:
        return blocked_outputs(
            generated_at=generated_at,
            artifacts_dir=artifacts_dir,
            repo_dir=repo_dir,
            trace_path=args.trace_path,
            missing_columns=missing,
            available_columns=available,
        )

    unit_result: dict[str, Any] | None = None
    if args.run_unit_tests:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_dir)
        unit_result = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_*.py"], repo_dir, env)
        log_lines.extend(["UNIT TESTS:", json.dumps(unit_result, indent=2, sort_keys=True)])
        if not unit_result["success"]:
            log_path = logs_dir / "e02_s04_label_shuffle_nulls.log"
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return int(unit_result["returnCode"] or 1)

    trace = load_trace(args.trace_path)
    row_nulls = build_row_nulls(trace, args.null_samples, args.random_seed)
    run_df, curve_df, validation_metrics = analyze_trace(trace, row_nulls)
    summary_df = summarize_results(run_df)

    result_path = results_dir / "e02_label_shuffle_aggregation_nulls.parquet"
    curve_path = step_dir / "e02_label_shuffle_curve_null_summary.csv"
    summary_path = step_dir / "e02_label_shuffle_condition_summary.csv"
    figure_path = figures_dir / "aggregation_label_shuffle_nulls.png"
    validation_path = step_dir / "s04_validation.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s04_label_shuffle_nulls_manifest.json"
    log_path = logs_dir / "e02_s04_label_shuffle_nulls.log"
    report_path = step_dir / "research_step_full_results.md"

    run_df.to_parquet(result_path, index=False)
    curve_df.to_csv(curve_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    make_figure(run_df, curve_df, figure_path)
    expected_artifacts = [
        report_path,
        result_path,
        figure_path,
        validation_path,
        curve_path,
        summary_path,
        log_path,
        manifest_path,
        src_manifest_path,
    ]
    validation = validate_results(trace, run_df, curve_df, validation_metrics, figure_path, expected_artifacts, args.null_samples)
    write_json(validation_path, validation)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    elapsed_seconds = time.monotonic() - start
    log_lines.extend(
        [
            f"trace_rows={len(trace)} run_rows={len(run_df)} curve_rows={len(curve_df)} null_samples={args.null_samples}",
            f"validation={json.dumps(validation, sort_keys=True)}",
            f"elapsed_seconds={elapsed_seconds:.3f}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    code_files = [
        repo_dir / "scripts/e02_s04_label_shuffle_nulls.py",
        repo_dir / "tests/e02/test_label_shuffle_nulls.py",
    ]
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
            if path.exists()
        ],
        "parameters": {
            "tracePath": str(args.trace_path),
            "nullSamples": int(args.null_samples),
            "randomSeed": int(args.random_seed),
            "selectedConditionIds": list(SELECTED_CONDITION_IDS),
        },
        "validation": validation,
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "newDependenciesInstalled": [],
        },
    }
    write_json(src_manifest_path, src_manifest)

    artifact_stub = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "artifacts": [],
    }
    write_json(manifest_path, artifact_stub)
    report_text = build_report(
        generated_at=generated_at,
        artifacts_dir=artifacts_dir,
        result_path=result_path,
        curve_path=curve_path,
        summary_path=summary_path,
        validation_path=validation_path,
        log_path=log_path,
        figure_path=figure_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        validation=validation,
        summary=summary_df,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        null_samples=args.null_samples,
        elapsed_seconds=elapsed_seconds,
    )
    report_path.write_text(report_text, encoding="utf-8")
    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S04 full-results Markdown handoff report."),
        artifact_entry(result_path, artifacts_dir, "Run-level label-shuffle Aggregation null table."),
        artifact_entry(curve_path, artifacts_dir, "Curve-level label-shuffle null summary."),
        artifact_entry(summary_path, artifacts_dir, "Condition-level label-shuffle null summary."),
        artifact_entry(figure_path, artifacts_dir, "Aggregation label-shuffle null figure."),
        artifact_entry(validation_path, artifacts_dir, "S04 validation evidence."),
        artifact_entry(log_path, artifacts_dir, "S04 execution log."),
        artifact_entry(src_manifest_path, artifacts_dir, "S04 source snapshot manifest."),
    ]
    write_json(manifest_path, artifact_stub | {"artifacts": artifacts})
    if not validation["validationPassed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
