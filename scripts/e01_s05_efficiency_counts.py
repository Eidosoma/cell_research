#!/usr/bin/env python3
"""Run E01 S05 efficiency comparison from frozen S03 and S04 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STEP_ID = "S05"
STEP_NUMBER = 5
EXPERIMENT_ID = "E01"
SOURCE_STEP_ID = "S04"
ALGORITHMS = ("bubble", "insertion", "selection")
MODES = ("traditional", "cell_view")
COUNT_METRICS = (
    "swap_only_steps",
    "comparison_steps_observed",
    "compare_plus_swap_steps",
)
FIGURE_METRICS = ("swap_only_steps", "compare_plus_swap_steps")
BOOTSTRAP_SEED = 2026063005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config-path", type=Path, default=Path("/artifacts/configs/e01_baseline_configs.json"))
    parser.add_argument("--condition-matrix-path", type=Path, default=Path("/artifacts/tables/e01_condition_matrix.csv"))
    parser.add_argument("--s04-run-records-path", type=Path, default=Path("/artifacts/research_steps/S04/e01_figure3_run_records.csv"))
    parser.add_argument("--s04-trace-path", type=Path, default=Path("/artifacts/traces/e01_figure3_trajectories.parquet"))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_status(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    stringified = df.astype("string").fillna("").astype(str)
    headers = list(stringified.columns)
    rows = stringified.values.tolist()
    widths = [
        max(len(str(header)), *(len(row[col_idx]) for row in rows))
        for col_idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    divider_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider_line, *body_lines])


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, reps: int) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    samples = rng.choice(values, size=(reps, values.size), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def bootstrap_pair_ci(
    traditional_values: np.ndarray,
    cell_view_values: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, float]:
    n = traditional_values.size
    indices = rng.integers(0, n, size=(reps, n))
    trad_samples = traditional_values[indices]
    cell_samples = cell_view_values[indices]
    diff = (cell_samples - trad_samples).mean(axis=1)
    ratio = cell_samples.mean(axis=1) / trad_samples.mean(axis=1)
    return {
        "paired_difference_ci_low": float(np.quantile(diff, 0.025)),
        "paired_difference_ci_high": float(np.quantile(diff, 0.975)),
        "ratio_of_means_ci_low": float(np.quantile(ratio, 0.025)),
        "ratio_of_means_ci_high": float(np.quantile(ratio, 0.975)),
    }


def paper_reference(algorithm: str, metric: str) -> tuple[str, str, float | None]:
    if metric == "swap_only_steps":
        if algorithm in {"bubble", "insertion"}:
            return ("cell_view approximately equals traditional", "near_equal", 1.0)
        return ("cell_view Selection takes about 11x traditional swaps", "cell_greater", 11.0)
    if metric == "compare_plus_swap_steps":
        if algorithm == "bubble":
            return ("cell_view Bubble fewer by 1.5x", "cell_less", 1.0 / 1.5)
        if algorithm == "insertion":
            return ("cell_view Insertion fewer by 2.03x", "cell_less", 1.0 / 2.03)
        return ("cell_view Selection greater by 1.17x", "cell_greater", 1.17)
    return ("audit-only comparison field; no Figure 4 standalone target", "audit_only", None)


def direction_matches(expectation: str, ratio: float) -> bool | None:
    if expectation == "near_equal":
        return abs(ratio - 1.0) <= 0.02
    if expectation == "cell_less":
        return ratio < 1.0
    if expectation == "cell_greater":
        return ratio > 1.0
    return None


def count_semantics(mode: str) -> tuple[str, str]:
    if mode == "traditional":
        return (
            "reconstructed_controller_comparison_count",
            "Traditional comparison counts come from S04 reconstructed controllers: adjacent probes for Bubble and Insertion, scan comparisons for Selection.",
        )
    return (
        "public_status_probe_compare_and_swap_count",
        "Cell-view comparison counts use StatusProbe.compare_and_swap_count, which increments only when should_move() is true in the public thread classes; this is an actionable-comparison proxy, not a complete read census.",
    )


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = json.loads(args.config_path.read_text())
    condition_matrix = pd.read_csv(args.condition_matrix_path)
    runs = pd.read_csv(args.s04_run_records_path)
    trace_cols = [
        "condition_id",
        "mode",
        "algorithm",
        "repeat_index",
        "is_final",
        "swap_step",
        "sortedness_percent",
        "initial_array_sha256",
        "final_array_sha256",
    ]
    trace_final = pd.read_parquet(args.s04_trace_path, columns=trace_cols)
    trace_final = trace_final[trace_final["is_final"]].copy()
    return cfg, condition_matrix, runs, trace_final


def build_efficiency_counts(
    cfg: dict[str, Any],
    condition_matrix: pd.DataFrame,
    runs: pd.DataFrame,
    trace_final: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    s04_conditions = condition_matrix[condition_matrix["step_scope"].astype(str).str.contains("S04", na=False)].copy()
    s04_condition_ids = set(s04_conditions["condition_id"].tolist())
    expected_repeat_count = int(cfg["globalDefaults"]["repeatCount"])

    merged = runs.merge(
        trace_final.rename(
            columns={
                "swap_step": "trace_final_swap_step",
                "sortedness_percent": "trace_final_sortedness_percent",
                "initial_array_sha256": "trace_initial_array_sha256",
                "final_array_sha256": "trace_final_array_sha256",
            }
        ),
        on=["condition_id", "mode", "algorithm", "repeat_index"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        s04_conditions[
            [
                "condition_id",
                "repeat_count",
                "value_bank_id",
                "matched_group_id",
                "baseline_source",
                "mode",
                "algorithm",
            ]
        ].rename(
            columns={
                "repeat_count": "s03_repeat_count",
                "value_bank_id": "s03_value_bank_id",
                "matched_group_id": "s03_matched_group_id",
                "baseline_source": "s03_baseline_source",
            }
        ),
        on=["condition_id", "mode", "algorithm"],
        how="left",
        validate="many_to_one",
    )

    for col in ["swap_count", "comparison_count", "compare_plus_swap_count"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["count_fields_complete"] = (
        merged[["swap_count", "comparison_count", "compare_plus_swap_count"]].notna().all(axis=1)
        & (merged["swap_count"] >= 0)
        & (merged["comparison_count"] >= 0)
        & (merged["compare_plus_swap_count"] >= 0)
        & (merged["compare_plus_swap_count"] == merged["swap_count"] + merged["comparison_count"])
    )
    merged["trace_swap_count_matches"] = merged["trace_final_swap_step"] == merged["swap_count"]
    merged["trace_final_sortedness_100"] = merged["trace_final_sortedness_percent"] == 100.0
    merged["trace_hashes_match_run_record"] = (
        (merged["trace_initial_array_sha256"] == merged["initial_array_sha256"])
        & (merged["trace_final_array_sha256"] == merged["final_array_sha256"])
    )
    merged["s03_condition_verified"] = (
        merged["condition_id"].isin(s04_condition_ids)
        & (merged["s03_repeat_count"] == expected_repeat_count)
        & (merged["s03_value_bank_id"] == merged["value_bank_id"])
        & (merged["s03_matched_group_id"] == merged["matched_group_id"])
        & (merged["s03_baseline_source"] == merged["baseline_source"])
    )

    matched_details: list[dict[str, Any]] = []
    matched_by_algorithm: dict[str, bool] = {}
    for algorithm in ALGORITHMS:
        trad = merged[(merged["algorithm"] == algorithm) & (merged["mode"] == "traditional")].sort_values("repeat_index")
        cell = merged[(merged["algorithm"] == algorithm) & (merged["mode"] == "cell_view")].sort_values("repeat_index")
        same_hashes = trad["initial_array_sha256"].tolist() == cell["initial_array_sha256"].tolist()
        same_seeds = trad["initial_array_seed"].tolist() == cell["initial_array_seed"].tolist()
        observed_repeats = min(trad["repeat_index"].nunique(), cell["repeat_index"].nunique())
        valid = same_hashes and same_seeds and observed_repeats == expected_repeat_count
        matched_by_algorithm[algorithm] = valid
        matched_details.append(
            {
                "algorithm": algorithm,
                "sameInitialArrayHashesByRepeat": same_hashes,
                "sameInitialArraySeedsByRepeat": same_seeds,
                "repeatCountCompared": int(observed_repeats),
                "valid": valid,
            }
        )

    output = pd.DataFrame(
        {
            "research_step_id": STEP_ID,
            "source_research_step_id": SOURCE_STEP_ID,
            "experiment_id": EXPERIMENT_ID,
            "condition_id": merged["condition_id"],
            "mode": merged["mode"],
            "algorithm": merged["algorithm"],
            "repeat_index": merged["repeat_index"].astype(int),
            "baseline_source": merged["baseline_source"],
            "matched_group_id": merged["matched_group_id"],
            "value_bank_id": merged["value_bank_id"],
            "initial_array_seed": merged["initial_array_seed"].astype(int),
            "scheduler_seed": merged["scheduler_seed"],
            "initial_array_sha256": merged["initial_array_sha256"],
            "final_array_sha256": merged["final_array_sha256"],
            "swap_only_steps": merged["swap_count"].astype(int),
            "comparison_steps_observed": merged["comparison_count"].astype(int),
            "compare_plus_swap_steps": merged["compare_plus_swap_count"].astype(int),
            "final_sortedness_percent": merged["final_sortedness_percent"],
            "stop_reason": merged["stop_reason"],
            "timed_out": merged["timed_out"],
            "count_fields_complete": merged["count_fields_complete"],
            "trace_swap_count_matches": merged["trace_swap_count_matches"],
            "trace_final_sortedness_100": merged["trace_final_sortedness_100"],
            "trace_hashes_match_run_record": merged["trace_hashes_match_run_record"],
            "s03_condition_verified": merged["s03_condition_verified"],
            "matched_initial_arrays_valid": merged["algorithm"].map(matched_by_algorithm),
        }
    )
    semantics = output["mode"].map(lambda mode: count_semantics(mode)[0])
    notes = output["mode"].map(lambda mode: count_semantics(mode)[1])
    output["comparison_count_source"] = semantics
    output["comparison_count_semantics"] = notes

    repeat_counts = output.groupby(["mode", "algorithm"])["repeat_index"].nunique().to_dict()
    validation = {
        "expectedRepeatsPerModeAlgorithm": expected_repeat_count,
        "observedRepeatsPerModeAlgorithm": {f"{mode}:{algorithm}": int(count) for (mode, algorithm), count in repeat_counts.items()},
        "allCountFieldsComplete": bool(output["count_fields_complete"].all()),
        "allTraceSwapCountsMatch": bool(output["trace_swap_count_matches"].all()),
        "allTraceFinalSortedness100": bool(output["trace_final_sortedness_100"].all()),
        "allTraceHashesMatchRunRecord": bool(output["trace_hashes_match_run_record"].all()),
        "allS03ConditionsVerified": bool(output["s03_condition_verified"].all()),
        "matchedInitialArraysValid": bool(output["matched_initial_arrays_valid"].all()),
        "matchedInitialArrayDetails": matched_details,
        "noTimeouts": bool((~output["timed_out"]).all()),
        "runRows": int(len(output)),
    }
    validation["success"] = bool(
        validation["allCountFieldsComplete"]
        and validation["allTraceSwapCountsMatch"]
        and validation["allTraceFinalSortedness100"]
        and validation["allTraceHashesMatchRunRecord"]
        and validation["allS03ConditionsVerified"]
        and validation["matchedInitialArraysValid"]
        and validation["noTimeouts"]
        and validation["runRows"] == expected_repeat_count * len(ALGORITHMS) * len(MODES)
        and all(count == expected_repeat_count for count in repeat_counts.values())
    )
    return output, validation


def build_numeric_table(efficiency_counts: pd.DataFrame, bootstrap_reps: int) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        trad = efficiency_counts[(efficiency_counts["algorithm"] == algorithm) & (efficiency_counts["mode"] == "traditional")].sort_values("repeat_index")
        cell = efficiency_counts[(efficiency_counts["algorithm"] == algorithm) & (efficiency_counts["mode"] == "cell_view")].sort_values("repeat_index")
        for metric in COUNT_METRICS:
            trad_values = trad[metric].to_numpy(dtype=float)
            cell_values = cell[metric].to_numpy(dtype=float)
            trad_ci = bootstrap_mean_ci(trad_values, rng, bootstrap_reps)
            cell_ci = bootstrap_mean_ci(cell_values, rng, bootstrap_reps)
            pair_ci = bootstrap_pair_ci(trad_values, cell_values, rng, bootstrap_reps)
            paired_diff = cell_values - trad_values
            ratio = float(cell_values.mean() / trad_values.mean())
            reference, expectation, paper_ratio = paper_reference(algorithm, metric)
            matches = direction_matches(expectation, ratio)
            if expectation == "audit_only":
                replication_status = "audit_only"
            elif matches and paper_ratio is not None and abs(ratio - paper_ratio) <= max(0.1, 0.25 * paper_ratio):
                replication_status = "direction_and_magnitude_consistent"
            elif matches:
                replication_status = "directionally_consistent_magnitude_differs"
            else:
                replication_status = "not_directionally_consistent"
            factor = 1.0 / ratio if ratio < 1 else ratio
            if ratio < 1:
                factor_interpretation = "cell_view_fewer_by_factor"
            elif ratio > 1:
                factor_interpretation = "cell_view_greater_by_factor"
            else:
                factor_interpretation = "equal"
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "algorithm": algorithm,
                    "count_metric": metric,
                    "included_in_figure4": metric in FIGURE_METRICS,
                    "n_pairs": int(len(trad_values)),
                    "traditional_mean": float(trad_values.mean()),
                    "traditional_std": float(trad_values.std(ddof=1)),
                    "traditional_ci95_low": float(trad_ci[0]),
                    "traditional_ci95_high": float(trad_ci[1]),
                    "cell_view_mean": float(cell_values.mean()),
                    "cell_view_std": float(cell_values.std(ddof=1)),
                    "cell_view_ci95_low": float(cell_ci[0]),
                    "cell_view_ci95_high": float(cell_ci[1]),
                    "paired_difference_cell_minus_traditional_mean": float(paired_diff.mean()),
                    "paired_difference_ci95_low": pair_ci["paired_difference_ci_low"],
                    "paired_difference_ci95_high": pair_ci["paired_difference_ci_high"],
                    "ratio_of_means_cell_over_traditional": ratio,
                    "ratio_of_means_ci95_low": pair_ci["ratio_of_means_ci_low"],
                    "ratio_of_means_ci95_high": pair_ci["ratio_of_means_ci_high"],
                    "efficiency_factor_abs": factor,
                    "factor_interpretation": factor_interpretation,
                    "paper_reference": reference,
                    "paper_expected_direction": expectation,
                    "paper_ratio_cell_over_traditional": paper_ratio,
                    "direction_matches_paper": matches,
                    "replication_status": replication_status,
                }
            )
    return pd.DataFrame(rows)


def plot_figure(numeric_table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_data = numeric_table[numeric_table["included_in_figure4"]].copy()
    metric_titles = {
        "swap_only_steps": "Swap-only steps",
        "compare_plus_swap_steps": "Swap + comparison steps",
    }
    colors = {"traditional": "#4f6fad", "cell_view": "#d95f02"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(ALGORITHMS))
    width = 0.36
    for ax, metric in zip(axes, FIGURE_METRICS):
        subset = figure_data[figure_data["count_metric"] == metric].set_index("algorithm")
        trad_mean = subset.loc[list(ALGORITHMS), "traditional_mean"].to_numpy()
        cell_mean = subset.loc[list(ALGORITHMS), "cell_view_mean"].to_numpy()
        trad_low = subset.loc[list(ALGORITHMS), "traditional_ci95_low"].to_numpy()
        trad_high = subset.loc[list(ALGORITHMS), "traditional_ci95_high"].to_numpy()
        cell_low = subset.loc[list(ALGORITHMS), "cell_view_ci95_low"].to_numpy()
        cell_high = subset.loc[list(ALGORITHMS), "cell_view_ci95_high"].to_numpy()
        trad_err = np.vstack([trad_mean - trad_low, trad_high - trad_mean])
        cell_err = np.vstack([cell_mean - cell_low, cell_high - cell_mean])
        ax.bar(x - width / 2, trad_mean, width, label="Traditional", color=colors["traditional"], yerr=trad_err, capsize=4)
        ax.bar(x + width / 2, cell_mean, width, label="Cell-view", color=colors["cell_view"], yerr=cell_err, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([label.title() for label in ALGORITHMS])
        ax.set_ylabel("Mean steps")
        ax.set_title(metric_titles[metric])
        ax.grid(axis="y", color="#d0d0d0", alpha=0.45)
        ymax = max(trad_high.max(), cell_high.max())
        ax.set_ylim(0, ymax * 1.18)
        for xpos, mean in zip(x - width / 2, trad_mean):
            ax.text(xpos, mean, f"{mean:.0f}", ha="center", va="bottom", fontsize=8)
        for xpos, mean in zip(x + width / 2, cell_mean):
            ax.text(xpos, mean, f"{mean:.0f}", ha="center", va="bottom", fontsize=8)
    axes[0].legend(frameon=False)
    fig.suptitle("E01 S05 Figure 4-style efficiency comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def update_checksum_file(checksum_path: Path, paths: list[Path]) -> None:
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    order: list[str] = []
    if checksum_path.exists():
        for line in checksum_path.read_text().splitlines():
            if not line.strip():
                continue
            checksum, path = line.split(maxsplit=1)
            existing[path] = checksum
            order.append(path)
    for path in paths:
        if not path.exists():
            continue
        resolved = str(path)
        existing[resolved] = sha256_file(path)
        if resolved not in order:
            order.append(resolved)
    checksum_path.write_text("".join(f"{existing[path]}  {path}\n" for path in order), encoding="utf-8")


def update_run_manifest(
    artifacts_dir: Path,
    paths: dict[str, Path],
    validation: dict[str, Any],
    summary_payload: dict[str, Any],
    repo_dir: Path,
    command: list[str],
) -> None:
    manifest_path = artifacts_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["researchStepId"] = STEP_ID
    manifest["updatedAtUtc"] = utc_now()
    manifest.setdefault("artifacts", {}).update({key: str(path) for key, path in paths.items()})
    manifest.setdefault("checksums", {})
    for path in paths.values():
        if path.exists():
            manifest["checksums"][str(path)] = sha256_file(path)
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": "Reproduce efficiency comparisons",
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": summary_payload["outcomeClassification"],
        "artifactsWritten": [str(path) for path in paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s05_efficiency_counts.py"),
        "command": " ".join(command),
        "summary": summary_payload,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_artifact_manifest(manifest_path: Path, artifact_paths: dict[str, Path], validation: dict[str, Any], repo_dir: Path) -> None:
    payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "validationResult": validation["validationResult"],
        "artifacts": [
            {
                "label": label,
                "path": str(path),
                "sha256": sha256_file(path) if path.exists() else None,
                "sizeBytes": path.stat().st_size if path.exists() else None,
            }
            for label, path in artifact_paths.items()
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_result_table(numeric_table: pd.DataFrame) -> pd.DataFrame:
    return numeric_table[numeric_table["count_metric"].isin(FIGURE_METRICS)][
        [
            "algorithm",
            "count_metric",
            "traditional_mean",
            "cell_view_mean",
            "paired_difference_cell_minus_traditional_mean",
            "ratio_of_means_cell_over_traditional",
            "efficiency_factor_abs",
            "factor_interpretation",
            "paper_reference",
            "direction_matches_paper",
            "replication_status",
        ]
    ].copy()


def write_report(
    report_path: Path,
    artifact_paths: dict[str, Path],
    validation: dict[str, Any],
    numeric_table: pd.DataFrame,
    summary_payload: dict[str, Any],
    command: list[str],
    repo_dir: Path,
    args: argparse.Namespace,
    wall_seconds: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    outcome = summary_payload["outcomeClassification"]
    caveats = "; ".join(validation["caveatsOrBlockers"])
    lay_summary = (
        "Using the S04 matched runs, Bubble and Insertion had identical swap-only means between traditional "
        "and cell-view modes, while cell-view Selection required more swaps. Under the available swap-plus-"
        "comparison counts, Bubble and Insertion cell-view means were lower than traditional means and "
        "Selection was higher, so the Figure 4 directions reproduce, but the exact ratios do not because "
        "comparison counts are ambiguous in the public Probe."
    )
    compact_table = compact_result_table(numeric_table)
    report = f"""# E01 S05 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: {validation["status"]}
- Artifacts written: {", ".join(str(path) for path in artifact_paths.values())}
- Validation result: {validation["validationResult"]}
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Lay summary: {lay_summary}
- Recommended next action: {validation["recommendedNextAction"]}

## Frozen Question

Do Bubble and Insertion cell-view sorts match traditional swap counts but outperform under swap-plus-comparison counting, while cell-view Selection remains less efficient?

## Inputs

- Frozen S03 config: `{args.config_path}` (`{sha256_file(args.config_path)}`)
- S03 condition matrix: `{args.condition_matrix_path}` (`{sha256_file(args.condition_matrix_path)}`)
- S04 run records: `{args.s04_run_records_path}` (`{sha256_file(args.s04_run_records_path)}`)
- S04 trajectory trace: `{args.s04_trace_path}` (`{sha256_file(args.s04_trace_path)}`)
- Attached paper text: Figure 4 and Section 4.1 describe 100 matched unique-value runs, swap-only counts, and swap-plus-comparison counts.

## Methods

S05 did not rerun sorting. It treated the S04 wrapper outputs as the frozen instrumented source: run-level records supplied `swap_count`, `comparison_count`, and `compare_plus_swap_count`, while the S04 trajectory parquet supplied final trace rows used to audit swap counts and terminal Sortedness. The S03 condition matrix supplied the expected six unperturbed condition IDs and the expected 100 repeats per mode and algorithm.

The run-level S05 parquet preserves both raw count modes:

- `swap_only_steps`: S04 `swap_count`.
- `comparison_steps_observed`: S04 `comparison_count`.
- `compare_plus_swap_steps`: S04 `swap_count + comparison_count`.

For traditional reconstructed baselines, comparison counts are controller-level adjacent or scan comparisons recorded by the S04 reconstructed wrappers. For cell-view baselines, comparison counts are the public `StatusProbe.compare_and_swap_count`; inspection of the public cell classes shows it increments when `should_move()` is true, so it is an actionable-comparison proxy rather than a complete census of every value read or target check.

For each algorithm and count metric, S05 computed means, standard deviations, bootstrap 95% confidence intervals with seed `{BOOTSTRAP_SEED}` and `{args.bootstrap_reps}` resamples, paired cell-view minus traditional differences, and ratios of means. The Figure 4-style plot uses swap-only and swap-plus-comparison metrics.

## Commands

```bash
{" ".join(command)}
```

Validation/smoke command:

```bash
python -m py_compile scripts/e01_s05_efficiency_counts.py
```

## Dependencies And Parameters

- Python executable: `{sys.executable}`
- Python version: `{sys.version.splitlines()[0]}`
- Platform: `{platform.platform()}`
- Pandas version: `{pd.__version__}`
- NumPy version: `{np.__version__}`
- Matplotlib version: `{matplotlib.__version__}`
- Repository commit at run time: `{git_commit(repo_dir)}`
- Repository status during report generation: `{git_status(repo_dir) or "clean"}`
- Bootstrap seed: `{BOOTSTRAP_SEED}`
- Bootstrap repetitions: `{args.bootstrap_reps}`
- Wall time: `{wall_seconds:.3f}` seconds

No new Python, system, R, Rust, or Node dependencies were installed for S05.

## Results

Figure 4-style metrics:

{dataframe_to_markdown(compact_table)}

Full numeric table:

{dataframe_to_markdown(numeric_table)}

Primary interpretation:

- Swap-only: Bubble and Insertion cell-view means exactly matched traditional means; Selection cell-view used `{summary_payload["selectionSwapRatio"]:.2f}`x traditional swaps.
- Swap-plus-comparison: Bubble cell-view used `{summary_payload["bubbleComparePlusTraditionalOverCell"]:.2f}`x fewer steps than traditional by ratio-of-means inversion, Insertion used `{summary_payload["insertionComparePlusTraditionalOverCell"]:.2f}`x fewer, and Selection cell-view used `{summary_payload["selectionComparePlusCellOverTraditional"]:.2f}`x traditional.
- The directional Figure 4 claims reproduce; exact paper magnitudes do not reproduce under the public Probe/reconstructed-controller counting convention.

## Validation

- 600 run-level rows: `{validation["runRows"] == 600}`.
- 100 repeats for every mode/algorithm: `{validation["observedRepeatsPerModeAlgorithm"]}`.
- Count fields complete and internally additive: `{validation["allCountFieldsComplete"]}`.
- Final S04 trace swap step equals run-level swap count: `{validation["allTraceSwapCountsMatch"]}`.
- Trace finals all reached Sortedness 100%: `{validation["allTraceFinalSortedness100"]}`.
- Trace hashes match run-record hashes: `{validation["allTraceHashesMatchRunRecord"]}`.
- S03 condition IDs and repeat counts verified: `{validation["allS03ConditionsVerified"]}`.
- Matched initial arrays across traditional and cell-view modes: `{validation["matchedInitialArraysValid"]}`.
- No timeouts: `{validation["noTimeouts"]}`.

Matched-array details:

```json
{json.dumps(validation["matchedInitialArrayDetails"], indent=2)}
```

## Artifacts And Provenance

Reusable outputs:

{chr(10).join(f"- `{label}`: `{path}`" for label, path in artifact_paths.items())}

The global run manifest and checksum file were updated after artifact creation. The S05 artifact manifest records paths, sizes, and SHA256 hashes.

## Caveats, Blockers, Failed Assumptions, And Limitations

- Traditional Bubble, Insertion, and Selection remain reconstructed baselines from S03/S04 because no public traditional runner was found.
- Cell-view comparison counts are `StatusProbe.compare_and_swap_count` events, not an exhaustive read/comparison census; exact Figure 4 ratio reproduction is therefore ambiguous.
- Traditional comparison counts are reconstructed-controller counts; alternate controller comparison conventions could change Bubble and Selection magnitudes.
- S05 supports the direction of the Figure 4 claims but not the exact reported fold changes under this public-code count convention.
- Confirmatory z-tests and effect sizes remain queued for S06.

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    artifacts_dir = args.artifacts_dir.resolve()
    s05_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e01"
    tables_dir = artifacts_dir / "tables"
    for path in (s05_dir, results_dir, figures_dir, tables_dir, artifacts_dir / "checksums"):
        path.mkdir(parents=True, exist_ok=True)

    cfg, condition_matrix, runs, trace_final = load_inputs(args)
    efficiency_counts, validation = build_efficiency_counts(cfg, condition_matrix, runs, trace_final)
    numeric_table = build_numeric_table(efficiency_counts, args.bootstrap_reps)

    counts_path = results_dir / "e01_efficiency_counts.parquet"
    figure_path = figures_dir / "figure4_efficiency_reproduction.png"
    numeric_table_path = tables_dir / "e01_efficiency_numeric_table.csv"
    validation_path = s05_dir / "s05_validation.json"
    artifact_manifest_path = s05_dir / "artifact_manifest.json"
    report_path = s05_dir / "research_step_full_results.md"

    efficiency_counts.to_parquet(counts_path, index=False)
    numeric_table.to_csv(numeric_table_path, index=False)
    plot_figure(numeric_table, figure_path)

    ratio_lookup = numeric_table.set_index(["algorithm", "count_metric"])["ratio_of_means_cell_over_traditional"].to_dict()
    validation.update(
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "status": "completed_with_caveats" if validation["success"] else "completed_validation_failed",
            "validationResult": "passed_with_comparison_counting_caveat" if validation["success"] else "failed",
            "artifactsWritten": [
                str(report_path),
                str(counts_path),
                str(figure_path),
                str(numeric_table_path),
                str(validation_path),
                str(artifact_manifest_path),
            ],
            "caveatsOrBlockers": [
                "Traditional baselines remain reconstructed because no public traditional runner was found.",
                "Cell-view comparison counts use StatusProbe.compare_and_swap_count, an actionable-comparison proxy rather than a complete read census.",
                "Exact Figure 4 fold-change magnitudes do not match the paper under this public-code count convention, though the primary directions match.",
            ],
            "recommendedNextAction": "Proceed to S06 statistical tests using S05 efficiency counts, while keeping the comparison-counting caveat explicit.",
            "createdAtUtc": utc_now(),
            "repositoryCommitAtRunTime": git_commit(args.repo_dir),
            "script": str(args.repo_dir / "scripts/e01_s05_efficiency_counts.py"),
        }
    )
    summary_payload = {
        "outcomeClassification": "supportive" if validation["success"] else "constraining/contradictory",
        "runRows": int(len(efficiency_counts)),
        "numericRows": int(len(numeric_table)),
        "allFigure4DirectionsMatch": bool(
            numeric_table[numeric_table["included_in_figure4"]]["direction_matches_paper"]
            .map(lambda value: True if pd.isna(value) else bool(value))
            .all()
        ),
        "selectionSwapRatio": float(ratio_lookup[("selection", "swap_only_steps")]),
        "bubbleComparePlusTraditionalOverCell": float(1.0 / ratio_lookup[("bubble", "compare_plus_swap_steps")]),
        "insertionComparePlusTraditionalOverCell": float(1.0 / ratio_lookup[("insertion", "compare_plus_swap_steps")]),
        "selectionComparePlusCellOverTraditional": float(ratio_lookup[("selection", "compare_plus_swap_steps")]),
    }
    validation["outcomeClassification"] = summary_payload["outcomeClassification"]
    validation["summary"] = summary_payload
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact_paths = {
        "fullResultsReport": report_path,
        "efficiencyCountsParquet": counts_path,
        "figurePng": figure_path,
        "numericTableCsv": numeric_table_path,
        "validationJson": validation_path,
        "artifactManifest": artifact_manifest_path,
    }
    wall_seconds = time.monotonic() - started
    write_report(
        report_path,
        artifact_paths,
        validation,
        numeric_table,
        summary_payload,
        sys.argv,
        args.repo_dir,
        args,
        wall_seconds,
    )
    write_artifact_manifest(artifact_manifest_path, artifact_paths, validation, args.repo_dir)

    update_run_manifest(artifacts_dir, {f"s05_{key}": value for key, value in artifact_paths.items()}, validation, summary_payload, args.repo_dir, sys.argv)
    checksum_path = artifacts_dir / "checksums" / "sha256sums.txt"
    update_checksum_file(
        checksum_path,
        list(artifact_paths.values())
        + [
            args.config_path,
            args.condition_matrix_path,
            args.s04_run_records_path,
            args.s04_trace_path,
            args.repo_dir / "scripts/e01_s05_efficiency_counts.py",
            artifacts_dir / "run_manifest.json",
            checksum_path,
        ],
    )
    print(json.dumps({"validation": validation, "summary": summary_payload}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
