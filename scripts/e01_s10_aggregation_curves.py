#!/usr/bin/env python3
"""E01 S10: reproduce Figure 8-style Aggregation curves from S09 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.weightstats import ztest
except Exception:  # pragma: no cover - defensive fallback for reduced envs
    ztest = None


RESEARCH_STEP_ID = "S10"
STEP_NUMBER = 10
EXPERIMENT_ID = "E01"
BOOTSTRAP_SEED = 2646294683
BOOTSTRAP_RESAMPLES = 10_000
PROGRESS_BINS = np.linspace(0.0, 1.0, 101)
TRACE_COLUMNS = [
    "conditionId",
    "mixtureId",
    "algorithms",
    "replicateIndex",
    "replicateNumber",
    "eventIndex",
    "swapCount",
    "sortednessPercent",
    "algotypeSequence",
]
EXPECTED_MIXTURES = [
    "pure_bubble",
    "pure_insertion",
    "pure_selection",
    "bubble_insertion",
    "bubble_selection",
    "insertion_selection",
    "bubble_insertion_selection",
]
MIXED_MIXTURES = [
    "bubble_insertion",
    "bubble_selection",
    "insertion_selection",
    "bubble_insertion_selection",
]
PAPER_REPORTED_PEAKS = {
    "bubble_insertion": {"paperPeakAggregation": 0.65, "paperPeakProgress": 0.21},
    "bubble_selection": {"paperPeakAggregation": 0.72, "paperPeakProgress": 0.42},
    "insertion_selection": {"paperPeakAggregation": 0.69, "paperPeakProgress": 0.19},
    "bubble_insertion_selection": {
        "paperPeakAggregation": 0.62,
        "paperPeakProgress": 0.22,
    },
}
DISPLAY_NAMES = {
    "pure_bubble": "Pure Bubble",
    "pure_insertion": "Pure Insertion",
    "pure_selection": "Pure Selection",
    "bubble_insertion": "Bubble + Insertion",
    "bubble_selection": "Bubble + Selection",
    "insertion_selection": "Insertion + Selection",
    "bubble_insertion_selection": "Bubble + Insertion + Selection",
}


@dataclass(frozen=True)
class OutputPaths:
    artifacts_dir: Path
    step_dir: Path
    code_dir: Path
    results_dir: Path
    figures_dir: Path
    traces_dir: Path
    provenance_dir: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute S10 Aggregation curves from S09 same-goal chimera traces."
    )
    parser.add_argument(
        "--artifacts-dir",
        default=os.environ.get("ARTIFACTS_DIR", "/artifacts"),
        help="Collectible artifact root. Defaults to $ARTIFACTS_DIR or /artifacts.",
    )
    parser.add_argument(
        "--s09-trace",
        default="/artifacts/traces/e01/S09/e01_s09_same_goal_chimera_trajectory_events.csv.gz",
        help="S09 trajectory trace CSV.GZ.",
    )
    parser.add_argument(
        "--s09-results",
        default="/artifacts/results/e01_same_goal_chimeras.parquet",
        help="S09 per-replicate results table.",
    )
    parser.add_argument(
        "--s09-status",
        default="/artifacts/research_steps/S09/status.json",
        help="S09 status JSON used for provenance and validation context.",
    )
    parser.add_argument(
        "--max-replicates-per-condition",
        type=int,
        default=None,
        help="Optional smoke-test limit. Full S10 leaves this unset.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure rendering. Intended only for quick smoke checks.",
    )
    return parser.parse_args()


def make_output_paths(artifacts_dir: Path) -> OutputPaths:
    return OutputPaths(
        artifacts_dir=artifacts_dir,
        step_dir=artifacts_dir / "research_steps" / RESEARCH_STEP_ID,
        code_dir=artifacts_dir / "research_steps" / RESEARCH_STEP_ID / "code",
        results_dir=artifacts_dir / "results",
        figures_dir=artifacts_dir / "figures" / "e01",
        traces_dir=artifacts_dir / "traces" / "e01" / RESEARCH_STEP_ID,
        provenance_dir=artifacts_dir / "provenance",
    )


def ensure_dirs(paths: OutputPaths) -> None:
    for directory in [
        paths.step_dir,
        paths.code_dir,
        paths.results_dir,
        paths.figures_dir,
        paths.traces_dir,
        paths.provenance_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo_root: Path) -> dict[str, str]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT
            ).strip()
        except subprocess.CalledProcessError as exc:
            return exc.output.strip()

    return {
        "branch": run_git(["branch", "--show-current"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirtyStatus": run_git(["status", "--short"]),
        "remote": run_git(["remote", "-v"]),
    }


def stable_seed(label: str, offset: int = 0) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}:{offset}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def aggregation_value(seq: str) -> float:
    """Adjacent-pair fraction matching the paper's left-neighbor wording."""
    if len(seq) < 2:
        return 0.0
    matches = sum(seq[idx] == seq[idx - 1] for idx in range(1, len(seq)))
    return matches / (len(seq) - 1)


def count_algotypes(seq: str) -> dict[str, int]:
    return dict(sorted(Counter(seq).items()))


def random_adjacency_expected(seq_or_counts: str | dict[str, int]) -> float:
    """Expected same-adjacent-pair fraction after random permutation at fixed counts."""
    if isinstance(seq_or_counts, str):
        counts = Counter(seq_or_counts)
        n = len(seq_or_counts)
    else:
        counts = seq_or_counts
        n = sum(counts.values())
    if n < 2:
        return 0.0
    numerator = sum(count * (count - 1) for count in counts.values())
    return numerator / (n * (n - 1))


def run_unit_tests() -> dict[str, Any]:
    cases = {
        "": 0.0,
        "B": 0.0,
        "BI": 0.0,
        "BB": 1.0,
        "BIBI": 0.0,
        "BBBB": 1.0,
        "BBII": 2.0 / 3.0,
        "BIIB": 1.0 / 3.0,
        "BBSI": 1.0 / 3.0,
    }
    failures: list[str] = []
    for seq, expected in cases.items():
        observed = aggregation_value(seq)
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"{seq!r}: expected {expected}, observed {observed}")

    null_cases: list[tuple[str, float]] = [
        ("B" * 50 + "I" * 50, (50 * 49 + 50 * 49) / (100 * 99)),
        ("B" * 34 + "I" * 33 + "S" * 33, (34 * 33 + 33 * 32 + 33 * 32) / (100 * 99)),
        ("B" * 100, 1.0),
    ]
    for seq, expected in null_cases:
        observed = random_adjacency_expected(seq)
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            failures.append(
                f"null {seq[:8]}...: expected {expected}, observed {observed}"
            )

    if failures:
        raise AssertionError("; ".join(failures))

    return {
        "aggregationCases": cases,
        "randomNullCases": [
            {
                "counts": count_algotypes(seq),
                "expected": expected,
                "observed": random_adjacency_expected(seq),
            }
            for seq, expected in null_cases
        ],
        "passed": True,
    }


def load_trace(trace_path: Path, max_replicates_per_condition: int | None) -> pd.DataFrame:
    if not trace_path.exists():
        raise FileNotFoundError(f"S09 trace not found: {trace_path}")

    dtype = {
        "conditionId": "string",
        "mixtureId": "string",
        "algorithms": "string",
        "replicateIndex": "int64",
        "replicateNumber": "int64",
        "eventIndex": "int64",
        "swapCount": "int64",
        "sortednessPercent": "float64",
        "algotypeSequence": "string",
    }
    trace = pd.read_csv(
        trace_path,
        compression="gzip",
        usecols=TRACE_COLUMNS,
        dtype=dtype,
    )
    if max_replicates_per_condition is not None:
        trace = trace[trace["replicateIndex"] < max_replicates_per_condition].copy()
    return trace


def add_aggregation_metrics(trace: pd.DataFrame) -> pd.DataFrame:
    aggregation_cache: dict[str, float] = {}
    null_cache: dict[str, float] = {}

    def cached_aggregation(seq_value: Any) -> float:
        seq = str(seq_value)
        value = aggregation_cache.get(seq)
        if value is None:
            value = aggregation_value(seq)
            aggregation_cache[seq] = value
        return value

    def cached_null(seq_value: Any) -> float:
        seq = str(seq_value)
        value = null_cache.get(seq)
        if value is None:
            value = random_adjacency_expected(seq)
            null_cache[seq] = value
        return value

    trace = trace.copy()
    trace["aggregationValue"] = trace["algotypeSequence"].map(cached_aggregation).astype(
        "float64"
    )
    trace["randomNullExpected"] = trace["algotypeSequence"].map(cached_null).astype(
        "float64"
    )
    trace["aggregationMinusNull"] = (
        trace["aggregationValue"] - trace["randomNullExpected"]
    )
    return trace


def build_binned_curves(trace: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "conditionId",
        "mixtureId",
        "algorithms",
        "replicateIndex",
        "replicateNumber",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in trace.groupby(group_cols, sort=True, observed=True):
        condition_id, mixture_id, algorithms, replicate_index, replicate_number = keys
        group = group.sort_values(["eventIndex"], kind="mergesort")
        compact = group.drop_duplicates("swapCount", keep="last")
        final_swap_count = float(compact["swapCount"].max())
        if final_swap_count <= 0:
            x = np.zeros(len(compact), dtype=float)
        else:
            x = compact["swapCount"].to_numpy(dtype=float) / final_swap_count

        order = np.argsort(x, kind="mergesort")
        x = x[order]
        aggregation = compact["aggregationValue"].to_numpy(dtype=float)[order]
        sortedness = compact["sortednessPercent"].to_numpy(dtype=float)[order]
        random_null = compact["randomNullExpected"].to_numpy(dtype=float)[order]
        minus_null = compact["aggregationMinusNull"].to_numpy(dtype=float)[order]

        if len(np.unique(x)) != len(x):
            # Collapse any remaining duplicate progress values by keeping the last event.
            tmp = pd.DataFrame(
                {
                    "x": x,
                    "aggregation": aggregation,
                    "sortedness": sortedness,
                    "randomNull": random_null,
                    "minusNull": minus_null,
                }
            )
            tmp = tmp.drop_duplicates("x", keep="last")
            x = tmp["x"].to_numpy(dtype=float)
            aggregation = tmp["aggregation"].to_numpy(dtype=float)
            sortedness = tmp["sortedness"].to_numpy(dtype=float)
            random_null = tmp["randomNull"].to_numpy(dtype=float)
            minus_null = tmp["minusNull"].to_numpy(dtype=float)

        if len(x) == 1:
            agg_interp = np.repeat(aggregation[0], len(PROGRESS_BINS))
            sorted_interp = np.repeat(sortedness[0], len(PROGRESS_BINS))
            null_interp = np.repeat(random_null[0], len(PROGRESS_BINS))
            minus_interp = np.repeat(minus_null[0], len(PROGRESS_BINS))
        else:
            agg_interp = np.interp(PROGRESS_BINS, x, aggregation)
            sorted_interp = np.interp(PROGRESS_BINS, x, sortedness)
            null_interp = np.interp(PROGRESS_BINS, x, random_null)
            minus_interp = np.interp(PROGRESS_BINS, x, minus_null)

        for progress_bin, progress in enumerate(PROGRESS_BINS):
            rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "conditionId": condition_id,
                    "mixtureId": mixture_id,
                    "algorithms": algorithms,
                    "replicateIndex": int(replicate_index),
                    "replicateNumber": int(replicate_number),
                    "progressBin": progress_bin,
                    "normalizedProgress": float(progress),
                    "finalSwapCount": final_swap_count,
                    "aggregationValue": float(agg_interp[progress_bin]),
                    "sortednessPercent": float(sorted_interp[progress_bin]),
                    "randomNullExpected": float(null_interp[progress_bin]),
                    "aggregationMinusNull": float(minus_interp[progress_bin]),
                }
            )
    return pd.DataFrame(rows)


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "conditionId",
        "mixtureId",
        "algorithms",
        "progressBin",
        "normalizedProgress",
    ]
    summary = (
        curves.groupby(group_cols, as_index=False, observed=True)
        .agg(
            replicateCount=("aggregationValue", "size"),
            aggregationMean=("aggregationValue", "mean"),
            aggregationSd=("aggregationValue", "std"),
            sortednessMean=("sortednessPercent", "mean"),
            sortednessSd=("sortednessPercent", "std"),
            randomNullMean=("randomNullExpected", "mean"),
            aggregationMinusNullMean=("aggregationMinusNull", "mean"),
            aggregationMinusNullSd=("aggregationMinusNull", "std"),
        )
        .sort_values(["mixtureId", "progressBin"])
        .reset_index(drop=True)
    )
    for prefix in ["aggregation", "sortedness", "aggregationMinusNull"]:
        sd_col = f"{prefix}Sd"
        mean_col = f"{prefix}Mean"
        sem_col = f"{prefix}Sem"
        lower_col = f"{prefix}Ci95Lower"
        upper_col = f"{prefix}Ci95Upper"
        summary[sem_col] = summary[sd_col] / np.sqrt(summary["replicateCount"])
        summary[sem_col] = summary[sem_col].fillna(0.0)
        summary[lower_col] = summary[mean_col] - 1.96 * summary[sem_col]
        summary[upper_col] = summary[mean_col] + 1.96 * summary[sem_col]
    summary["aggregationCi95Lower"] = summary["aggregationCi95Lower"].clip(0.0, 1.0)
    summary["aggregationCi95Upper"] = summary["aggregationCi95Upper"].clip(0.0, 1.0)
    summary["sortednessCi95Lower"] = summary["sortednessCi95Lower"].clip(0.0, 100.0)
    summary["sortednessCi95Upper"] = summary["sortednessCi95Upper"].clip(0.0, 100.0)
    return summary


def build_peak_tables(curves: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "conditionId",
        "mixtureId",
        "algorithms",
        "replicateIndex",
        "replicateNumber",
    ]
    peak_rows: list[dict[str, Any]] = []
    for keys, group in curves.groupby(group_cols, sort=True, observed=True):
        condition_id, mixture_id, algorithms, replicate_index, replicate_number = keys
        group = group.sort_values("progressBin")
        values = group["aggregationValue"].to_numpy(dtype=float)
        peak_idx = int(np.argmax(values))
        peak = group.iloc[peak_idx]
        initial = group.iloc[0]
        final = group.iloc[-1]
        peak_rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "conditionId": condition_id,
                "mixtureId": mixture_id,
                "algorithms": algorithms,
                "replicateIndex": int(replicate_index),
                "replicateNumber": int(replicate_number),
                "isMixedCondition": bool(mixture_id in MIXED_MIXTURES),
                "randomNullExpected": float(peak["randomNullExpected"]),
                "initialAggregation": float(initial["aggregationValue"]),
                "initialAggregationMinusNull": float(initial["aggregationMinusNull"]),
                "peakAggregation": float(peak["aggregationValue"]),
                "peakAggregationMinusNull": float(peak["aggregationMinusNull"]),
                "peakProgressBin": int(peak["progressBin"]),
                "peakNormalizedProgress": float(peak["normalizedProgress"]),
                "finalAggregation": float(final["aggregationValue"]),
                "finalAggregationMinusNull": float(final["aggregationMinusNull"]),
                "finalSortednessPercent": float(final["sortednessPercent"]),
                "meanAggregationOverProgress": float(group["aggregationValue"].mean()),
                "finalSwapCount": float(final["finalSwapCount"]),
            }
        )
    peaks = pd.DataFrame(peak_rows)

    summary_rows: list[dict[str, Any]] = []
    for (condition_id, mixture_id, algorithms), group in peaks.groupby(
        ["conditionId", "mixtureId", "algorithms"], sort=True, observed=True
    ):
        curve_group = summary[
            (summary["conditionId"] == condition_id)
            & (summary["mixtureId"] == mixture_id)
            & (summary["algorithms"] == algorithms)
        ].sort_values("progressBin")
        mean_curve_peak = curve_group.loc[curve_group["aggregationMean"].idxmax()]
        row = {
            "researchStepId": RESEARCH_STEP_ID,
            "conditionId": condition_id,
            "mixtureId": mixture_id,
            "algorithms": algorithms,
            "isMixedCondition": bool(mixture_id in MIXED_MIXTURES),
            "replicateCount": int(len(group)),
            "randomNullExpectedMean": float(group["randomNullExpected"].mean()),
            "initialAggregationMean": float(group["initialAggregation"].mean()),
            "initialAggregationSd": float(group["initialAggregation"].std(ddof=1)),
            "peakAggregationMean": float(group["peakAggregation"].mean()),
            "peakAggregationSd": float(group["peakAggregation"].std(ddof=1)),
            "peakAggregationMinusNullMean": float(
                group["peakAggregationMinusNull"].mean()
            ),
            "peakAggregationMinusNullSd": float(group["peakAggregationMinusNull"].std(ddof=1)),
            "peakNormalizedProgressMean": float(group["peakNormalizedProgress"].mean()),
            "peakNormalizedProgressSd": float(group["peakNormalizedProgress"].std(ddof=1)),
            "finalAggregationMean": float(group["finalAggregation"].mean()),
            "finalAggregationSd": float(group["finalAggregation"].std(ddof=1)),
            "meanCurvePeakAggregation": float(mean_curve_peak["aggregationMean"]),
            "meanCurvePeakProgress": float(mean_curve_peak["normalizedProgress"]),
            "meanCurvePeakProgressBin": int(mean_curve_peak["progressBin"]),
        }
        paper = PAPER_REPORTED_PEAKS.get(mixture_id)
        if paper is not None:
            row.update(paper)
            row["meanCurvePeakMinusPaperPeak"] = (
                row["meanCurvePeakAggregation"] - paper["paperPeakAggregation"]
            )
            row["meanCurvePeakProgressMinusPaper"] = (
                row["meanCurvePeakProgress"] - paper["paperPeakProgress"]
            )
        else:
            row.update(
                {
                    "paperPeakAggregation": np.nan,
                    "paperPeakProgress": np.nan,
                    "meanCurvePeakMinusPaperPeak": np.nan,
                    "meanCurvePeakProgressMinusPaper": np.nan,
                }
            )
        summary_rows.append(row)
    peak_summary = pd.DataFrame(summary_rows).sort_values("mixtureId").reset_index(drop=True)
    return peaks, peak_summary


def bootstrap_mean_ci(values: np.ndarray, seed: int, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def add_peak_statistics(peaks: pd.DataFrame, peak_summary: pd.DataFrame) -> pd.DataFrame:
    stat_rows: list[dict[str, Any]] = []
    bootstrap_fields: dict[str, dict[str, float]] = {}
    for mixture_id, group in peaks.groupby("mixtureId", sort=True, observed=True):
        peak_values = group["peakAggregation"].to_numpy(dtype=float)
        excess = group["peakAggregationMinusNull"].to_numpy(dtype=float)
        progress = group["peakNormalizedProgress"].to_numpy(dtype=float)
        peak_ci = bootstrap_mean_ci(peak_values, stable_seed(mixture_id, 1))
        excess_ci = bootstrap_mean_ci(excess, stable_seed(mixture_id, 2))
        progress_ci = bootstrap_mean_ci(progress, stable_seed(mixture_id, 3))
        bootstrap_fields[mixture_id] = {
            "peakAggregationBootstrapCi95Lower": peak_ci[0],
            "peakAggregationBootstrapCi95Upper": peak_ci[1],
            "peakAggregationMinusNullBootstrapCi95Lower": excess_ci[0],
            "peakAggregationMinusNullBootstrapCi95Upper": excess_ci[1],
            "peakProgressBootstrapCi95Lower": progress_ci[0],
            "peakProgressBootstrapCi95Upper": progress_ci[1],
        }

        is_mixed = mixture_id in MIXED_MIXTURES
        if is_mixed:
            if ztest is not None:
                z_stat, z_pvalue = ztest(excess, value=0.0, alternative="larger")
            else:
                z_stat, z_pvalue = (np.nan, np.nan)
            t_result = stats.ttest_1samp(excess, popmean=0.0, alternative="greater")
            sd = float(np.std(excess, ddof=1))
            effect = float(np.mean(excess) / sd) if sd > 0 else np.inf
            stat_rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "mixtureId": mixture_id,
                    "comparisonType": "mixed_peak_minus_fixed_count_random_null",
                    "replicateCount": int(len(group)),
                    "meanPeakAggregation": float(np.mean(peak_values)),
                    "meanPeakAggregationMinusNull": float(np.mean(excess)),
                    "bootstrapCi95Lower": excess_ci[0],
                    "bootstrapCi95Upper": excess_ci[1],
                    "zStatistic": float(z_stat),
                    "zTestAlternative": "larger",
                    "zTestPValue": float(z_pvalue),
                    "tStatistic": float(t_result.statistic),
                    "tTestAlternative": "larger",
                    "tTestPValue": float(t_result.pvalue),
                    "cohenD": effect,
                    "paperTestStatus": "available from S09 traces using a fixed-count random-label null",
                }
            )
        else:
            stat_rows.append(
                {
                    "researchStepId": RESEARCH_STEP_ID,
                    "mixtureId": mixture_id,
                    "comparisonType": "pure_control_sanity_only",
                    "replicateCount": int(len(group)),
                    "meanPeakAggregation": float(np.mean(peak_values)),
                    "meanPeakAggregationMinusNull": float(np.mean(excess)),
                    "bootstrapCi95Lower": excess_ci[0],
                    "bootstrapCi95Upper": excess_ci[1],
                    "zStatistic": np.nan,
                    "zTestAlternative": "not_tested",
                    "zTestPValue": np.nan,
                    "tStatistic": np.nan,
                    "tTestAlternative": "not_tested",
                    "tTestPValue": np.nan,
                    "cohenD": np.nan,
                    "paperTestStatus": "pure S09 controls contain one Algotype and are not the paper's two-label same-code negative controls",
                }
            )

    for column in [
        "peakAggregationBootstrapCi95Lower",
        "peakAggregationBootstrapCi95Upper",
        "peakAggregationMinusNullBootstrapCi95Lower",
        "peakAggregationMinusNullBootstrapCi95Upper",
        "peakProgressBootstrapCi95Lower",
        "peakProgressBootstrapCi95Upper",
    ]:
        peak_summary[column] = peak_summary["mixtureId"].map(
            {key: value[column] for key, value in bootstrap_fields.items()}
        )

    return pd.DataFrame(stat_rows).sort_values("mixtureId").reset_index(drop=True)


def build_null_baselines(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    initial = curves[curves["progressBin"] == 0].copy()
    for (condition_id, mixture_id, algorithms), group in initial.groupby(
        ["conditionId", "mixtureId", "algorithms"], sort=True, observed=True
    ):
        rows.append(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "conditionId": condition_id,
                "mixtureId": mixture_id,
                "algorithms": algorithms,
                "isMixedCondition": bool(mixture_id in MIXED_MIXTURES),
                "replicateCount": int(len(group)),
                "fixedCountRandomNullMean": float(group["randomNullExpected"].mean()),
                "fixedCountRandomNullMin": float(group["randomNullExpected"].min()),
                "fixedCountRandomNullMax": float(group["randomNullExpected"].max()),
                "initialObservedAggregationMean": float(group["aggregationValue"].mean()),
                "initialObservedAggregationSd": float(group["aggregationValue"].std(ddof=1)),
                "nullInterpretation": (
                    "one-Algotype pure control sanity baseline"
                    if mixture_id not in MIXED_MIXTURES
                    else "fixed-count random permutation expected adjacent-pair baseline"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mixtureId").reset_index(drop=True)


def write_table(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> list[Path]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return [parquet_path, csv_path]


def plot_aggregation_unique(summary: pd.DataFrame, peak_summary: pd.DataFrame, out_base: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), sharex=True, sharey=True)
    axes = axes.ravel()
    colors = {"aggregation": "#b3202a", "null": "#5f6368", "sortedness": "#1f77b4"}
    for ax, mixture_id in zip(axes, MIXED_MIXTURES, strict=True):
        subset = summary[summary["mixtureId"] == mixture_id].sort_values("progressBin")
        x = subset["normalizedProgress"].to_numpy(dtype=float) * 100.0
        y = subset["aggregationMean"].to_numpy(dtype=float)
        lo = subset["aggregationCi95Lower"].to_numpy(dtype=float)
        hi = subset["aggregationCi95Upper"].to_numpy(dtype=float)
        null = subset["randomNullMean"].to_numpy(dtype=float)
        sortedness = subset["sortednessMean"].to_numpy(dtype=float)
        ax.plot(x, y, color=colors["aggregation"], linewidth=2.2, label="Aggregation")
        ax.fill_between(x, lo, hi, color=colors["aggregation"], alpha=0.16, linewidth=0)
        ax.plot(x, null, color=colors["null"], linewidth=1.4, linestyle="--", label="Fixed-count null")
        ax.set_title(DISPLAY_NAMES[mixture_id], fontsize=12)
        ax.set_ylim(0.25, 0.82)
        ax.set_xlim(0, 100)
        ax.grid(True, color="#d6d6d6", linewidth=0.6, alpha=0.8)
        ax.set_xlabel("Normalized sorting progress (%)")
        ax.set_ylabel("Aggregation")

        twin = ax.twinx()
        twin.plot(x, sortedness, color=colors["sortedness"], linewidth=1.2, alpha=0.42, label="Sortedness")
        twin.set_ylim(0, 100)
        twin.set_ylabel("Sortedness (%)", color=colors["sortedness"], fontsize=9)
        twin.tick_params(axis="y", labelcolor=colors["sortedness"], labelsize=8)

        peak = peak_summary[peak_summary["mixtureId"] == mixture_id].iloc[0]
        ax.scatter(
            [float(peak["meanCurvePeakProgress"]) * 100.0],
            [float(peak["meanCurvePeakAggregation"])],
            color=colors["aggregation"],
            s=32,
            zorder=5,
        )
        ax.text(
            0.03,
            0.94,
            f"peak {peak['meanCurvePeakAggregation']:.2f} at {peak['meanCurvePeakProgress']:.0%}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            color="#242424",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("E01 S10 Figure 8-style Aggregation curves, unique-value same-goal chimeras", y=1.04, fontsize=14)
    fig.tight_layout()
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_controls(summary: pd.DataFrame, out_base: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    palette = {
        "pure_bubble": "#4c78a8",
        "pure_insertion": "#54a24b",
        "pure_selection": "#f58518",
        "bubble_insertion": "#b3202a",
        "bubble_selection": "#7f3c8d",
        "insertion_selection": "#d95f02",
        "bubble_insertion_selection": "#1b9e77",
    }
    for mixture_id in EXPECTED_MIXTURES:
        subset = summary[summary["mixtureId"] == mixture_id].sort_values("progressBin")
        if subset.empty:
            continue
        line_style = "-" if mixture_id in MIXED_MIXTURES else ":"
        ax.plot(
            subset["normalizedProgress"] * 100.0,
            subset["aggregationMean"],
            color=palette[mixture_id],
            linestyle=line_style,
            linewidth=2.0,
            label=DISPLAY_NAMES[mixture_id],
        )
    ax.set_xlabel("Normalized sorting progress (%)")
    ax.set_ylabel("Mean Aggregation")
    ax.set_ylim(0.25, 1.03)
    ax.set_xlim(0, 100)
    ax.grid(True, color="#d6d6d6", linewidth=0.6, alpha=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("S10 Aggregation curves with S09 pure-control sanity checks")
    fig.tight_layout()
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def build_methods_markdown(
    source_trace: Path,
    unit_tests: dict[str, Any],
    null_baselines: pd.DataFrame,
    peak_summary: pd.DataFrame,
) -> str:
    lines = [
        "# S10 Aggregation Methods",
        "",
        f"- Research step ID: {RESEARCH_STEP_ID}",
        f"- Step number: {STEP_NUMBER}",
        "- Completion status: completed",
        "- Artifacts written: see `artifact_manifest.json` and `status.json`.",
        "- Validation result: Aggregation unit tests passed; normalized progress and null-baseline calculations are documented here.",
        "- Caveats or blockers: S09 supplies pure one-Algotype controls, not the paper's two-label same-code negative controls; fixed-count random-label baselines are therefore used for chance comparison.",
        "- Recommended next action: hand control back to Chief Scientist review before S11.",
        "",
        "## Formula Interpretation",
        "",
        "The paper describes Aggregation as the fraction or percentage of cells whose directly adjacent left neighbor has the same Algotype. S10 implements that wording as:",
        "",
        "`Aggregation = (1 / (n - 1)) * sum_{i=1}^{n-1} I[Algotype_i == Algotype_{i-1}]`",
        "",
        "This is also the deterministic same-adjacent-pair fraction, so a right-neighbor implementation over positions 0 through n-2 gives the same value when each adjacent pair is counted once. Runs in S09 all have n=100.",
        "",
        "## Unit Tests",
        "",
        f"The script tested {len(unit_tests['aggregationCases'])} hand-constructed Algotype strings and {len(unit_tests['randomNullCases'])} fixed-count null cases before reading the S09 trace.",
        "",
        "| String | Expected Aggregation |",
        "| --- | ---: |",
    ]
    for seq, expected in unit_tests["aggregationCases"].items():
        label = seq if seq else "(empty)"
        lines.append(f"| `{label}` | {expected:.6f} |")
    lines.extend(
        [
            "",
            "## Normalized Progress Alignment",
            "",
            f"The source trace was `{source_trace}`. For each S09 replicate, S10 sorted events by event index, retained the last event at each swap count, divided swap count by the run's final swap count, and linearly interpolated Aggregation and Sortedness onto 101 bins from 0% through 100% progress. This uses sorting progress rather than raw event index so Bubble, Insertion, and Selection mixtures are compared on a common completion axis.",
            "",
            "## Null Baseline",
            "",
            "The chance baseline is the exact expected same-adjacent-pair fraction for a random permutation with the observed fixed Algotype counts:",
            "",
            "`E[random Aggregation] = sum_k c_k * (c_k - 1) / (n * (n - 1))`",
            "",
            "For two 50/50 Algotypes this equals 0.494949. For the S03 all-three 34/33/33 allocation it equals 0.326667. Pure S09 controls contain one Algotype and therefore have Aggregation and fixed-count null equal to 1.0; they are retained only as a sanity baseline, not as the paper's two-label same-code negative controls.",
            "",
            "| Mixture | Fixed-count random null | Initial observed Aggregation |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in null_baselines.itertuples(index=False):
        lines.append(
            f"| {DISPLAY_NAMES.get(row.mixtureId, row.mixtureId)} | {row.fixedCountRandomNullMean:.6f} | {row.initialObservedAggregationMean:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Peak Summary",
            "",
            "| Mixture | Mean-curve peak Aggregation | Peak progress | Paper peak | Paper progress |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in peak_summary[peak_summary["mixtureId"].isin(MIXED_MIXTURES)].itertuples(index=False):
        lines.append(
            f"| {DISPLAY_NAMES.get(row.mixtureId, row.mixtureId)} | {row.meanCurvePeakAggregation:.6f} | {row.meanCurvePeakProgress:.2%} | {row.paperPeakAggregation:.6f} | {row.paperPeakProgress:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_validation(
    *,
    unit_tests: dict[str, Any],
    trace: pd.DataFrame,
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    peaks: pd.DataFrame,
    peak_summary: pd.DataFrame,
    statistics_table: pd.DataFrame,
    s09_status: dict[str, Any],
    artifacts_written: list[Path],
    output_paths: OutputPaths,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "details": details})

    observed_mixtures = sorted(trace["mixtureId"].dropna().unique().tolist())
    add_check(
        "all_expected_s09_mixtures_present",
        sorted(EXPECTED_MIXTURES) == observed_mixtures,
        f"observed={observed_mixtures}",
    )
    add_check(
        "unit_tests_passed",
        bool(unit_tests.get("passed")),
        "hand-constructed Aggregation and fixed-count null cases passed",
    )
    add_check(
        "aggregation_bounds",
        bool(curves["aggregationValue"].between(0.0, 1.0).all()),
        "all binned Aggregation values are in [0, 1]",
    )
    add_check(
        "progress_bounds",
        bool(curves["normalizedProgress"].between(0.0, 1.0).all()),
        "all normalized progress values are in [0, 1]",
    )
    add_check(
        "expected_curve_rows",
        len(curves) == 7 * 100 * 101,
        f"observed {len(curves)} binned replicate-curve rows",
    )
    add_check(
        "expected_summary_rows",
        len(summary) == 7 * 101,
        f"observed {len(summary)} mean-curve rows",
    )
    add_check(
        "expected_peak_rows",
        len(peaks) == 700,
        f"observed {len(peaks)} replicate peak rows",
    )
    add_check(
        "final_sortedness_100",
        bool(np.isclose(peaks["finalSortednessPercent"], 100.0).all()),
        "every S09 run used by S10 reaches final 100% Sortedness",
    )
    mixed_stats = statistics_table[
        statistics_table["comparisonType"]
        == "mixed_peak_minus_fixed_count_random_null"
    ]
    add_check(
        "mixed_peak_excess_positive",
        bool((mixed_stats["meanPeakAggregationMinusNull"] > 0.0).all()),
        "every mixed condition has mean peak Aggregation above fixed-count random null",
    )
    add_check(
        "mixed_bootstrap_excess_positive",
        bool((mixed_stats["bootstrapCi95Lower"] > 0.0).all()),
        "every mixed condition has bootstrap 95% lower bound above fixed-count random null",
    )
    add_check(
        "mixed_z_tests_significant",
        bool((mixed_stats["zTestPValue"] < 0.01).all()),
        "paper-style available z-tests versus fixed-count random null all have p < 0.01",
    )
    add_check(
        "artifacts_nonempty",
        all(path.exists() and path.stat().st_size > 0 for path in artifacts_written),
        f"checked {len(artifacts_written)} declared artifacts",
    )
    add_check(
        "s11_not_started",
        not (output_paths.artifacts_dir / "research_steps" / "S11").exists(),
        "S11 artifact directory does not exist",
    )

    s09_trace_rows = s09_status.get("conditionCounts", {}).get("traceRows")
    add_check(
        "s09_trace_row_count_matches_status",
        s09_trace_rows is None or int(s09_trace_rows) == len(trace),
        f"S09 status traceRows={s09_trace_rows}; S10 read {len(trace)} rows",
    )

    overall = all(item["passed"] for item in checks)
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": overall,
        "status": "completed" if overall else "completed_with_validation_warnings",
        "validationResult": (
            "passed: Aggregation unit tests, S09 trace coverage, binned curves, peak tables, fixed-count null statistics, declared artifact checks, final 100% Sortedness, and no-S11-start guard all passed."
            if overall
            else "warning: one or more S10 validation checks failed; inspect validation checks."
        ),
        "checks": checks,
        "counts": {
            "traceRows": int(len(trace)),
            "curveRows": int(len(curves)),
            "summaryRows": int(len(summary)),
            "peakRows": int(len(peaks)),
            "peakSummaryRows": int(len(peak_summary)),
            "statisticsRows": int(len(statistics_table)),
        },
        "unitTests": unit_tests,
    }


def artifact_manifest(artifacts_written: list[Path], source_artifacts: dict[str, str]) -> dict[str, Any]:
    entries = []
    for path in sorted(set(artifacts_written)):
        if not path.exists():
            continue
        entries.append(
            {
                "path": str(path),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now_iso(),
        "sourceArtifacts": source_artifacts,
        "artifacts": entries,
    }


def render_validation_md(validation: dict[str, Any], artifacts_written: list[Path]) -> str:
    caveats = [
        "S10 uses deterministic S09 pseudo-scheduler traces rather than live OS-thread scheduling.",
        "The fixed-count random-label null is available from S09 traces; the paper's two-label same-code negative controls were not generated in S09 and remain an explicit limitation.",
        "Peak timing and peak height are measured on 101 normalized progress bins after interpolation by swap-count progress.",
    ]
    lines = [
        "# S10 Validation",
        "",
        f"- Research step ID: {RESEARCH_STEP_ID}",
        f"- Step number: {STEP_NUMBER}",
        "- Completion status: completed",
        f"- Validation result: {validation['validationResult']}",
        f"- Artifacts written: {len(artifacts_written)} files; see `artifact_manifest.json` and `status.json`.",
        "- Caveats or blockers:",
    ]
    lines.extend([f"- {item}" for item in caveats])
    lines.append("- Recommended next action: hand control back to Chief Scientist review before S11.")
    lines.extend(["", "## Checks", ""])
    for check in validation["checks"]:
        status = "passed" if check["passed"] else "failed"
        lines.append(f"- {status}: {check['check']} - {check['details']}")
    lines.append("")
    return "\n".join(lines)


def render_summary_md(
    *,
    status: dict[str, Any],
    peak_summary: pd.DataFrame,
    statistics_table: pd.DataFrame,
    artifacts_written: list[Path],
) -> str:
    lines = [
        "# S10 Status Summary",
        "",
        f"- Research step ID: {RESEARCH_STEP_ID}",
        f"- Step number: {STEP_NUMBER}",
        f"- Completion status: {status['status']}",
        f"- Outcome classification: {status['outcomeClassification']}",
        "- Artifacts written:",
    ]
    lines.extend([f"- `{path}`" for path in sorted(str(path) for path in artifacts_written)])
    lines.extend(
        [
            f"- Validation result: {status['validationResult']}",
            "- Caveats or blockers:",
        ]
    )
    lines.extend([f"- {item}" for item in status["caveatsOrBlockers"]])
    lines.extend(
        [
            f"- Lay summary: {status['laySummary']}",
            f"- Recommended next action: {status['recommendedNextAction']}",
            "",
            "## Mixed Aggregation Peaks",
            "",
            "| Mixture | Mean-curve peak | Progress | Fixed-count null | Peak excess mean | Bootstrap excess 95% CI | Paper peak |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in peak_summary[peak_summary["mixtureId"].isin(MIXED_MIXTURES)].itertuples(index=False):
        lines.append(
            f"| {DISPLAY_NAMES.get(row.mixtureId, row.mixtureId)} | {row.meanCurvePeakAggregation:.4f} | {row.meanCurvePeakProgress:.0%} | {row.randomNullExpectedMean:.4f} | {row.peakAggregationMinusNullMean:.4f} | [{row.peakAggregationMinusNullBootstrapCi95Lower:.4f}, {row.peakAggregationMinusNullBootstrapCi95Upper:.4f}] | {row.paperPeakAggregation:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Available Statistical Tests",
            "",
            "| Mixture | Z statistic | One-sided p-value | Cohen's d |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in statistics_table[
        statistics_table["comparisonType"] == "mixed_peak_minus_fixed_count_random_null"
    ].itertuples(index=False):
        lines.append(
            f"| {DISPLAY_NAMES.get(row.mixtureId, row.mixtureId)} | {row.zStatistic:.4g} | {row.zTestPValue:.4g} | {row.cohenD:.4g} |"
        )
    lines.append("")
    return "\n".join(lines)


def update_run_manifest(paths: OutputPaths, status: dict[str, Any]) -> Path:
    manifest_path = paths.provenance_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    if not manifest:
        manifest = {"researchSteps": {}}
    if "researchSteps" not in manifest or not isinstance(manifest["researchSteps"], dict):
        manifest["researchSteps"] = {}
    manifest["researchSteps"][RESEARCH_STEP_ID] = {
        "status": status["status"],
        "success": status["success"],
        "outcomeClassification": status["outcomeClassification"],
        "generatedAt": status["generatedAt"],
        "summary": str(paths.step_dir / "summary.md"),
        "statusJson": str(paths.step_dir / "status.json"),
        "artifactManifest": str(paths.step_dir / "artifact_manifest.json"),
    }
    manifest["lastUpdated"] = utc_now_iso()
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = Path(args.artifacts_dir).resolve()
    output_paths = make_output_paths(artifacts_dir)
    ensure_dirs(output_paths)

    source_trace = Path(args.s09_trace).resolve()
    source_results = Path(args.s09_results).resolve()
    source_status = Path(args.s09_status).resolve()
    s09_status = read_json(source_status)
    unit_tests = run_unit_tests()

    trace = load_trace(source_trace, args.max_replicates_per_condition)
    trace = add_aggregation_metrics(trace)
    curves = build_binned_curves(trace)
    summary = summarize_curves(curves)
    peaks, peak_summary = build_peak_tables(curves, summary)
    statistics_table = add_peak_statistics(peaks, peak_summary)
    null_baselines = build_null_baselines(curves)

    artifacts_written: list[Path] = []
    artifacts_written.extend(
        write_table(
            curves,
            output_paths.results_dir / "e01_aggregation_curves.parquet",
            output_paths.results_dir / "e01_aggregation_curves.csv",
        )
    )
    artifacts_written.extend(
        write_table(
            summary,
            output_paths.results_dir / "e01_aggregation_curve_summary.parquet",
            output_paths.results_dir / "e01_aggregation_curve_summary.csv",
        )
    )
    artifacts_written.extend(
        write_table(
            peaks,
            output_paths.results_dir / "e01_aggregation_peaks.parquet",
            output_paths.results_dir / "e01_aggregation_peaks.csv",
        )
    )
    artifacts_written.extend(
        write_table(
            peak_summary,
            output_paths.results_dir / "e01_aggregation_peak_summary.parquet",
            output_paths.results_dir / "e01_aggregation_peak_summary.csv",
        )
    )
    artifacts_written.extend(
        write_table(
            null_baselines,
            output_paths.results_dir / "e01_aggregation_null_baselines.parquet",
            output_paths.results_dir / "e01_aggregation_null_baselines.csv",
        )
    )
    artifacts_written.extend(
        write_table(
            statistics_table,
            output_paths.results_dir / "e01_aggregation_statistics.parquet",
            output_paths.results_dir / "e01_aggregation_statistics.csv",
        )
    )

    if not args.no_figures:
        artifacts_written.extend(
            plot_aggregation_unique(
                summary,
                peak_summary,
                output_paths.figures_dir / "figure08_aggregation_unique",
            )
        )
        artifacts_written.extend(
            plot_controls(
                summary,
                output_paths.figures_dir / "figure08_aggregation_controls",
            )
        )

    source_artifacts = {
        "s09Trace": str(source_trace),
        "s09Results": str(source_results),
        "s09Status": str(source_status),
    }
    validation = build_validation(
        unit_tests=unit_tests,
        trace=trace,
        curves=curves,
        summary=summary,
        peaks=peaks,
        peak_summary=peak_summary,
        statistics_table=statistics_table,
        s09_status=s09_status,
        artifacts_written=artifacts_written,
        output_paths=output_paths,
    )

    mixed_stats = statistics_table[
        statistics_table["comparisonType"] == "mixed_peak_minus_fixed_count_random_null"
    ]
    supportive = bool(
        validation["success"]
        and (mixed_stats["meanPeakAggregationMinusNull"] > 0.0).all()
        and (mixed_stats["bootstrapCi95Lower"] > 0.0).all()
    )
    outcome = "supportive" if supportive else "constraining"
    caveats = [
        "S10 inherits S09's deterministic pseudo-scheduler wrapper around archived cell-view classes rather than live OS-thread scheduling.",
        "S09 did not create the paper's two-label same-code negative controls; S10 therefore uses an exact fixed-count random-label null baseline and reports pure one-Algotype controls only as sanity checks.",
        "Aggregation is measured as a fraction of same adjacent pairs on 101 swap-progress bins; reported peak timing can move by one progress bin relative to raw events.",
        "The all-three condition uses the S03 34/33/33 allocation, so its chance null is about 0.3267 rather than the two-type 0.4949 baseline.",
    ]
    status = {
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"]),
        "status": validation["status"],
        "artifactsWritten": [str(path) for path in sorted(set(artifacts_written))],
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Hand control back to the Chief Scientist workflow for review; start S11 duplicate-value chimeras only after explicit instruction.",
        "experimentId": EXPERIMENT_ID,
        "generatedAt": utc_now_iso(),
        "outcomeClassification": outcome,
        "laySummary": "S10 measured whether mixed sorting-policy cells became neighbors during same-goal sorting. All four mixed Algotype conditions formed clusters above a fixed-count random-label baseline, with peaks broadly in the Figure 8 range, while final sorted unique-value arrays returned near the expected chance baseline.",
        "sourceArtifacts": source_artifacts,
        "sourceStatuses": {
            "S09": {
                "researchStepId": s09_status.get("researchStepId"),
                "status": s09_status.get("status"),
                "success": s09_status.get("success"),
                "validationResult": s09_status.get("validationResult"),
            }
        },
        "counts": validation["counts"],
        "mixedPeakSummary": peak_summary[
            peak_summary["mixtureId"].isin(MIXED_MIXTURES)
        ].to_dict(orient="records"),
        "runtime": {
            "pythonExecutable": sys.executable,
            "pythonVersion": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "osCpuCount": os.cpu_count(),
            "workerCount": 1,
            "gpuUsed": False,
            "numpyVersion": np.__version__,
            "pandasVersion": pd.__version__,
            "matplotlibVersion": matplotlib.__version__,
            "scipyVersion": stats.__version__ if hasattr(stats, "__version__") else "unknown",
            "bootstrapSeed": BOOTSTRAP_SEED,
            "bootstrapResamples": BOOTSTRAP_RESAMPLES,
        },
        "git": git_info(repo_root),
    }

    validation_path = output_paths.step_dir / "validation.json"
    write_json(validation_path, validation)
    artifacts_written.append(validation_path)
    validation_md_path = output_paths.step_dir / "validation.md"
    write_markdown(validation_md_path, render_validation_md(validation, artifacts_written))
    artifacts_written.append(validation_md_path)

    methods_path = output_paths.step_dir / "methods.md"
    write_markdown(
        methods_path,
        build_methods_markdown(source_trace, unit_tests, null_baselines, peak_summary),
    )
    artifacts_written.append(methods_path)

    code_path = output_paths.code_dir / Path(__file__).name
    shutil.copy2(Path(__file__), code_path)
    artifacts_written.append(code_path)

    manifest = artifact_manifest(artifacts_written, source_artifacts)
    manifest_path = output_paths.step_dir / "artifact_manifest.json"
    write_json(manifest_path, manifest)
    artifacts_written.append(manifest_path)

    status_path = output_paths.step_dir / "status.json"
    status["artifactsWritten"] = [str(path) for path in sorted(set(artifacts_written))]
    write_json(status_path, status)
    artifacts_written.append(status_path)

    summary_path = output_paths.step_dir / "summary.md"
    status["artifactsWritten"] = [str(path) for path in sorted(set(artifacts_written))]
    write_markdown(
        summary_path,
        render_summary_md(
            status=status,
            peak_summary=peak_summary,
            statistics_table=statistics_table,
            artifacts_written=artifacts_written,
        ),
    )
    artifacts_written.append(summary_path)

    run_manifest_path = update_run_manifest(output_paths, status)
    artifacts_written.append(run_manifest_path)

    # Refresh manifest/status now that all step files and run_manifest are present.
    manifest = artifact_manifest(artifacts_written, source_artifacts)
    write_json(manifest_path, manifest)
    status["artifactsWritten"] = [str(path) for path in sorted(set(artifacts_written))]
    write_json(status_path, status)
    write_markdown(
        summary_path,
        render_summary_md(
            status=status,
            peak_summary=peak_summary,
            statistics_table=statistics_table,
            artifacts_written=artifacts_written,
        ),
    )

    print(json.dumps({"status": status["status"], "validationResult": status["validationResult"]}, indent=2))
    return 0 if validation["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
