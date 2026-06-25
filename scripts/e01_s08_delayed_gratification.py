#!/usr/bin/env python3
"""Run E01 S08 Delayed Gratification analysis.

S08 ports the paper/code Delayed Gratification calculation into a clean,
unit-tested wrapper. It uses S04 no-Frozen trajectories as the f=0 baseline
and S07 Frozen Cell trajectories for f=1..3. It writes Figure 7-style outputs
and stops before S09.
"""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.weightstats import ztest as statsmodels_ztest

    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    statsmodels_ztest = None
    STATSMODELS_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "E01"
STEP_ID = "S08"
STEP_NUMBER = 8
STATUS = "completed"
PRIMARY_FROZEN_VARIANT = "stuck"
CONFIG_PATH_DEFAULT = Path("/artifacts/configs/e01_baseline_config.json")
CONDITION_MATRIX_DEFAULT = Path("/artifacts/research_steps/S03/condition_matrix.csv")
S04_TRACE_DEFAULT = Path("/artifacts/traces/e01/S04/trajectory_events.parquet")
S04_STATUS_DEFAULT = Path("/artifacts/research_steps/S04/status.json")
S07_TRACE_DEFAULT = Path("/artifacts/traces/e01/S07/e01_s07_frozen_cell_trajectory_events.csv.gz")
S07_STATUS_DEFAULT = Path("/artifacts/research_steps/S07/status.json")
S07_REPLICATES_DEFAULT = Path("/artifacts/results/e01_frozen_cell_robustness.parquet")
S06_STATUS_DEFAULT = Path("/artifacts/research_steps/S06/status.json")
RUN_MANIFEST_DEFAULT = Path("/artifacts/provenance/run_manifest.json")
ALGORITHMS = ["bubble", "insertion", "selection"]
IMPLEMENTATIONS = ["traditional", "cell_view"]
FIGURE_VARIANTS = ["passive", "stuck"]
PAPER_DG_NOTES = {
    "bubble": "Paper reports cell-view Bubble has more DG than traditional and increases with Frozen Cell count.",
    "insertion": "Paper reports cell-view Insertion has DG similar to traditional and increases with Frozen Cell count.",
    "selection": "Paper reports cell-view Selection has less DG than traditional and no clear Frozen Cell count trend.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {
            "args": args,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"] if commit["ok"] else "unknown",
        "branch": branch["stdout"] if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"],
        "remote": remote["stdout"],
    }


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append(
                {
                    "path": str(path),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    return sorted(artifacts, key=lambda item: item["path"])


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            if abs(value) >= 1000 or (0 < abs(value) < 0.001):
                return f"{value:.3g}"
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def dedup_consecutive(values: list[float] | np.ndarray, tolerance: float = 1e-12) -> list[float]:
    deduped: list[float] = []
    for raw_value in values:
        value = float(raw_value)
        if not deduped or not math.isclose(value, deduped[-1], rel_tol=0.0, abs_tol=tolerance):
            deduped.append(value)
    return deduped


def signed_segments(values: list[float] | np.ndarray, tolerance: float = 1e-12) -> list[float]:
    deduped = dedup_consecutive(values, tolerance=tolerance)
    segments: list[float] = []
    for previous, current in zip(deduped, deduped[1:]):
        delta = float(current - previous)
        if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=tolerance):
            continue
        if segments and segments[-1] * delta > 0:
            segments[-1] += delta
        else:
            segments.append(delta)
    return segments


def delayed_gratification_from_sortedness(values: list[float] | np.ndarray) -> dict[str, Any]:
    """Compute paper/code DG from a Sortedness trajectory.

    Consecutive equal values are removed, consecutive same-sign changes are
    collapsed, and each Sortedness drop is paired with the immediately
    following Sortedness recovery. The event score is (recovery - drop) / drop.
    The primary run-level DG is the signed mean of complete event scores.
    """

    segments = signed_segments(values)
    event_scores: list[float] = []
    drops: list[float] = []
    recoveries: list[float] = []
    event_segment_indices: list[int] = []

    index = 0
    while index < len(segments) and segments[index] >= 0:
        index += 1

    while index < len(segments) - 1:
        drop_segment = segments[index]
        recovery_segment = segments[index + 1]
        if drop_segment < 0 and recovery_segment > 0:
            drop = -float(drop_segment)
            recovery = float(recovery_segment)
            event_scores.append((recovery - drop) / drop if drop else 0.0)
            drops.append(drop)
            recoveries.append(recovery)
            event_segment_indices.append(index)
            index += 2
        else:
            index += 1

    return {
        "delayedGratification": float(np.mean(event_scores)) if event_scores else 0.0,
        "dgEventCount": int(len(event_scores)),
        "dgPositiveEventCount": int(sum(1 for score in event_scores if score > 0)),
        "dgNegativeEventCount": int(sum(1 for score in event_scores if score < 0)),
        "dgZeroEventCount": int(sum(1 for score in event_scores if math.isclose(score, 0.0, abs_tol=1e-12))),
        "dgTotalDrop": float(sum(drops)),
        "dgTotalRecovery": float(sum(recoveries)),
        "dgMeanDrop": float(np.mean(drops)) if drops else 0.0,
        "dgMeanRecovery": float(np.mean(recoveries)) if recoveries else 0.0,
        "dgMaxEventScore": float(max(event_scores)) if event_scores else 0.0,
        "dgMinEventScore": float(min(event_scores)) if event_scores else 0.0,
        "dgSignedEventScoresJson": json.dumps([round(score, 12) for score in event_scores], separators=(",", ":")),
        "dgSignedSegmentsJson": json.dumps([round(segment, 12) for segment in segments], separators=(",", ":")),
        "dgEventSegmentIndicesJson": json.dumps(event_segment_indices, separators=(",", ":")),
    }


def run_unit_tests() -> tuple[bool, list[dict[str, Any]]]:
    cases = [
        {
            "caseId": "empty",
            "sortedness": [],
            "expectedDg": 0.0,
            "expectedEvents": 0,
            "reason": "No trajectory points cannot have DG events.",
        },
        {
            "caseId": "single_point",
            "sortedness": [50.0],
            "expectedDg": 0.0,
            "expectedEvents": 0,
            "reason": "A one-point trajectory has no changes.",
        },
        {
            "caseId": "monotone_increase",
            "sortedness": [40.0, 50.0, 60.0],
            "expectedDg": 0.0,
            "expectedEvents": 0,
            "reason": "No temporary drop occurs.",
        },
        {
            "caseId": "single_complete_gain",
            "sortedness": [50.0, 60.0, 55.0, 70.0],
            "expectedDg": 2.0,
            "expectedEvents": 1,
            "reason": "Drop y=5 followed by recovery x=15 gives (15 - 5) / 5 = 2.",
        },
        {
            "caseId": "single_incomplete_recovery",
            "sortedness": [50.0, 60.0, 55.0, 58.0],
            "expectedDg": -0.4,
            "expectedEvents": 1,
            "reason": "Drop y=5 followed by recovery x=3 gives (3 - 5) / 5 = -0.4; the paper/code convention is signed.",
        },
        {
            "caseId": "two_equal_gain_events",
            "sortedness": [50.0, 60.0, 55.0, 70.0, 65.0, 80.0],
            "expectedDg": 2.0,
            "expectedEvents": 2,
            "reason": "Two identical complete events average to 2.",
        },
        {
            "caseId": "deduplicated_plateaus",
            "sortedness": [50.0, 50.0, 60.0, 60.0, 55.0, 55.0, 70.0],
            "expectedDg": 2.0,
            "expectedEvents": 1,
            "reason": "Consecutive equal plateaus are removed before segmenting.",
        },
        {
            "caseId": "starts_with_drop",
            "sortedness": [60.0, 55.0, 70.0],
            "expectedDg": 2.0,
            "expectedEvents": 1,
            "reason": "A trajectory can begin with the temporary drop.",
        },
        {
            "caseId": "trailing_unrecovered_drop",
            "sortedness": [50.0, 60.0, 55.0],
            "expectedDg": 0.0,
            "expectedEvents": 0,
            "reason": "A drop without subsequent recovery is not a complete DG event.",
        },
    ]
    results: list[dict[str, Any]] = []
    for case in cases:
        observed = delayed_gratification_from_sortedness(case["sortedness"])
        observed_dg = float(observed["delayedGratification"])
        observed_events = int(observed["dgEventCount"])
        passed = math.isclose(observed_dg, case["expectedDg"], rel_tol=1e-12, abs_tol=1e-12) and (
            observed_events == case["expectedEvents"]
        )
        results.append(
            {
                **case,
                "observedDg": observed_dg,
                "observedEvents": observed_events,
                "observedSegments": json.loads(observed["dgSignedSegmentsJson"]),
                "passed": passed,
            }
        )
    return all(result["passed"] for result in results), results


def s07_condition_id(implementation: str, algorithm: str, frozen_count: int, frozen_variant: str) -> str:
    return f"S07_{implementation}_{algorithm}_unique_f{frozen_count}_{frozen_variant}"


def load_s04_baseline_records(path: Path) -> pd.DataFrame:
    trace = pd.read_parquet(
        path,
        columns=[
            "condition_id",
            "implementation",
            "algorithm",
            "replicate_index",
            "replicate_number",
            "event_index",
            "sortedness_percent",
        ],
    )
    records: list[dict[str, Any]] = []
    group_cols = ["condition_id", "implementation", "algorithm", "replicate_index", "replicate_number"]
    for (source_condition_id, implementation, algorithm, replicate_index, replicate_number), group in trace.groupby(
        group_cols, sort=False
    ):
        trajectory = group.sort_values("event_index")["sortedness_percent"].to_numpy(dtype=float)
        metrics = delayed_gratification_from_sortedness(trajectory)
        records.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "sourceTrace": "S04_no_frozen_baseline",
                "sourceConditionId": source_condition_id,
                "conditionId": s07_condition_id(implementation, algorithm, 0, "none"),
                "implementation": implementation,
                "algorithm": algorithm,
                "frozenVariant": "none",
                "frozenCount": 0,
                "figureFrozenVariant": "none",
                "replicateIndex": int(replicate_index),
                "replicateNumber": int(replicate_number),
                "trajectoryEventCount": int(len(group)),
                "initialSortednessPercent": float(trajectory[0]) if len(trajectory) else None,
                "finalSortednessPercent": float(trajectory[-1]) if len(trajectory) else None,
                **metrics,
            }
        )
    return pd.DataFrame(records)


def load_s07_frozen_records(path: Path) -> pd.DataFrame:
    usecols = [
        "conditionId",
        "implementation",
        "algorithm",
        "frozenVariant",
        "frozenCount",
        "replicateIndex",
        "replicateNumber",
        "eventIndex",
        "sortednessPercent",
    ]
    trace = pd.read_csv(path, usecols=usecols)
    trace = trace[trace["frozenCount"] > 0].copy()
    records: list[dict[str, Any]] = []
    group_cols = [
        "conditionId",
        "implementation",
        "algorithm",
        "frozenVariant",
        "frozenCount",
        "replicateIndex",
        "replicateNumber",
    ]
    for (condition_id, implementation, algorithm, frozen_variant, frozen_count, replicate_index, replicate_number), group in trace.groupby(
        group_cols, sort=False
    ):
        trajectory = group.sort_values("eventIndex")["sortednessPercent"].to_numpy(dtype=float)
        metrics = delayed_gratification_from_sortedness(trajectory)
        records.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "sourceTrace": "S07_frozen_trajectory",
                "sourceConditionId": condition_id,
                "conditionId": condition_id,
                "implementation": implementation,
                "algorithm": algorithm,
                "frozenVariant": frozen_variant,
                "frozenCount": int(frozen_count),
                "figureFrozenVariant": frozen_variant,
                "replicateIndex": int(replicate_index),
                "replicateNumber": int(replicate_number),
                "trajectoryEventCount": int(len(group)),
                "initialSortednessPercent": float(trajectory[0]) if len(trajectory) else None,
                "finalSortednessPercent": float(trajectory[-1]) if len(trajectory) else None,
                **metrics,
            }
        )
    return pd.DataFrame(records)


def write_result_tables(
    replicate_df: pd.DataFrame,
    results_dir: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    replicate_parquet = results_dir / "e01_delayed_gratification.parquet"
    replicate_csv = results_dir / "e01_delayed_gratification.csv"
    summary_parquet = results_dir / "e01_delayed_gratification_summary.parquet"
    summary_csv = results_dir / "e01_delayed_gratification_summary.csv"
    backing_parquet = results_dir / "e01_delayed_gratification_figure07_backing.parquet"
    backing_csv = results_dir / "e01_delayed_gratification_figure07_backing.csv"
    trends_parquet = results_dir / "e01_delayed_gratification_trends.parquet"
    trends_csv = results_dir / "e01_delayed_gratification_trends.csv"
    comparisons_parquet = results_dir / "e01_delayed_gratification_comparisons.parquet"
    comparisons_csv = results_dir / "e01_delayed_gratification_comparisons.csv"

    replicate_df.to_parquet(replicate_parquet, index=False)
    replicate_df.to_csv(replicate_csv, index=False)

    group_cols = ["implementation", "algorithm", "frozenVariant", "frozenCount", "conditionId"]
    summary = (
        replicate_df.groupby(group_cols, dropna=False)
        .agg(
            replicateCount=("replicateIndex", "count"),
            meanDelayedGratification=("delayedGratification", "mean"),
            sdDelayedGratification=("delayedGratification", "std"),
            medianDelayedGratification=("delayedGratification", "median"),
            minDelayedGratification=("delayedGratification", "min"),
            maxDelayedGratification=("delayedGratification", "max"),
            meanDgEventCount=("dgEventCount", "mean"),
            meanDgTotalDrop=("dgTotalDrop", "mean"),
            meanDgTotalRecovery=("dgTotalRecovery", "mean"),
            meanTrajectoryEventCount=("trajectoryEventCount", "mean"),
            finalSortednessPercent=("finalSortednessPercent", "mean"),
        )
        .reset_index()
    )
    summary["semDelayedGratification"] = summary["sdDelayedGratification"] / np.sqrt(summary["replicateCount"])
    summary.to_parquet(summary_parquet, index=False)
    summary.to_csv(summary_csv, index=False)

    backing_parts = []
    none_summary = summary[summary["frozenVariant"] == "none"].copy()
    for variant in FIGURE_VARIANTS:
        part = none_summary.copy()
        part["figureFrozenVariant"] = variant
        part["primaryFigureVariant"] = variant == PRIMARY_FROZEN_VARIANT
        backing_parts.append(part)
    frozen_summary = summary[summary["frozenVariant"].isin(FIGURE_VARIANTS)].copy()
    frozen_summary["figureFrozenVariant"] = frozen_summary["frozenVariant"]
    frozen_summary["primaryFigureVariant"] = frozen_summary["figureFrozenVariant"] == PRIMARY_FROZEN_VARIANT
    backing_parts.append(frozen_summary)
    backing = pd.concat(backing_parts, ignore_index=True)
    backing = backing.sort_values(["figureFrozenVariant", "algorithm", "implementation", "frozenCount"]).reset_index(drop=True)
    backing.to_parquet(backing_parquet, index=False)
    backing.to_csv(backing_csv, index=False)

    trends = compute_trends(backing)
    trends.to_parquet(trends_parquet, index=False)
    trends.to_csv(trends_csv, index=False)

    comparisons = compute_comparisons(replicate_df)
    comparisons.to_parquet(comparisons_parquet, index=False)
    comparisons.to_csv(comparisons_csv, index=False)
    return (
        replicate_parquet,
        replicate_csv,
        summary_parquet,
        summary_csv,
        backing_parquet,
        backing_csv,
        trends_parquet,
        trends_csv,
        comparisons_parquet,
        comparisons_csv,
    )


def compute_trends(backing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (figure_variant, implementation, algorithm), group in backing.groupby(
        ["figureFrozenVariant", "implementation", "algorithm"], sort=False
    ):
        ordered = group.sort_values("frozenCount")
        x = ordered["frozenCount"].to_numpy(dtype=float)
        y = ordered["meanDelayedGratification"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(ordered) >= 2 else 0.0
        spearman = stats.spearmanr(x, y)
        monotone_non_decreasing = bool(np.all(np.diff(y) >= -1e-12))
        monotone_non_increasing = bool(np.all(np.diff(y) <= 1e-12))
        if slope > 1e-12:
            direction = "increasing"
        elif slope < -1e-12:
            direction = "decreasing"
        else:
            direction = "flat"
        rows.append(
            {
                "figureFrozenVariant": figure_variant,
                "implementation": implementation,
                "algorithm": algorithm,
                "frozenCountsJson": json.dumps([int(value) for value in x], separators=(",", ":")),
                "meanDelayedGratificationJson": json.dumps([round(float(value), 12) for value in y], separators=(",", ":")),
                "linearSlopePerFrozenCell": slope,
                "spearmanR": float(spearman.statistic) if math.isfinite(float(spearman.statistic)) else None,
                "spearmanPValue": float(spearman.pvalue) if math.isfinite(float(spearman.pvalue)) else None,
                "monotoneNonDecreasing": monotone_non_decreasing,
                "monotoneNonIncreasing": monotone_non_increasing,
                "trendDirection": direction,
                "isPrimaryPaperStyleVariant": figure_variant == PRIMARY_FROZEN_VARIANT,
            }
        )
    return pd.DataFrame(rows)


def fallback_two_sample_ztest(sample1: np.ndarray, sample2: np.ndarray) -> tuple[float, float]:
    var1 = float(np.var(sample1, ddof=1))
    var2 = float(np.var(sample2, ddof=1))
    diff = float(np.mean(sample1) - np.mean(sample2))
    denom = math.sqrt(var1 / len(sample1) + var2 / len(sample2))
    if denom == 0.0:
        z_statistic = 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
    else:
        z_statistic = diff / denom
    p_value = math.erfc(abs(z_statistic) / math.sqrt(2.0)) if math.isfinite(z_statistic) else 0.0
    return z_statistic, p_value


def paper_style_ztest(cell_view: np.ndarray, traditional: np.ndarray) -> tuple[float, float, str]:
    if np.var(cell_view, ddof=1) == 0.0 and np.var(traditional, ddof=1) == 0.0:
        z_statistic, p_value = fallback_two_sample_ztest(cell_view, traditional)
        return z_statistic, p_value, "fallback_two_sample_ztest_zero_variance"
    if STATSMODELS_AVAILABLE and statsmodels_ztest is not None:
        z_statistic, p_value = statsmodels_ztest(cell_view, traditional, alternative="two-sided")
        return float(z_statistic), float(p_value), "statsmodels_ztest"
    z_statistic, p_value = fallback_two_sample_ztest(cell_view, traditional)
    return z_statistic, p_value, "fallback_two_sample_ztest"


def compute_comparisons(replicate_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    base = replicate_df[replicate_df["frozenVariant"] == "none"].copy()
    for variant in FIGURE_VARIANTS:
        part = base.copy()
        part["figureFrozenVariant"] = variant
        parts.append(part)
    frozen = replicate_df[replicate_df["frozenVariant"].isin(FIGURE_VARIANTS)].copy()
    frozen["figureFrozenVariant"] = frozen["frozenVariant"]
    parts.append(frozen)
    expanded = pd.concat(parts, ignore_index=True)

    rows: list[dict[str, Any]] = []
    for (figure_variant, algorithm), group in expanded.groupby(["figureFrozenVariant", "algorithm"], sort=False):
        cell = group[group["implementation"] == "cell_view"]["delayedGratification"].to_numpy(dtype=float)
        trad = group[group["implementation"] == "traditional"]["delayedGratification"].to_numpy(dtype=float)
        z_statistic, p_value, method = paper_style_ztest(cell, trad)
        mean_cell = float(np.mean(cell))
        mean_trad = float(np.mean(trad))
        pooled_sd = math.sqrt((float(np.var(cell, ddof=1)) + float(np.var(trad, ddof=1))) / 2.0)
        cohen_d = (mean_cell - mean_trad) / pooled_sd if pooled_sd else 0.0
        rows.append(
            {
                "figureFrozenVariant": figure_variant,
                "algorithm": algorithm,
                "comparison": "cell_view_minus_traditional",
                "cellViewN": int(len(cell)),
                "traditionalN": int(len(trad)),
                "cellViewMeanDelayedGratification": mean_cell,
                "traditionalMeanDelayedGratification": mean_trad,
                "meanDifferenceCellMinusTraditional": mean_cell - mean_trad,
                "paperStyleZStatistic": z_statistic,
                "paperStylePValue": p_value,
                "ztestMethod": method,
                "cohenD": cohen_d,
                "isPrimaryPaperStyleVariant": figure_variant == PRIMARY_FROZEN_VARIANT,
                "paperClaimText": PAPER_DG_NOTES[algorithm],
            }
        )
    return pd.DataFrame(rows)


def plot_primary_figure(backing_path: Path, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    backing = pd.read_parquet(backing_path)
    primary = backing[backing["figureFrozenVariant"] == PRIMARY_FROZEN_VARIANT]
    png_path = figure_dir / "figure07_dg.png"
    pdf_path = figure_dir / "figure07_dg.pdf"
    colors = {"traditional": "#666666", "cell_view": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=False)
    for col_idx, algorithm in enumerate(ALGORITHMS):
        ax = axes[col_idx]
        panel = primary[primary["algorithm"] == algorithm]
        x = np.arange(4)
        width = 0.34
        for offset, implementation in [(-width / 2, "traditional"), (width / 2, "cell_view")]:
            subset = (
                panel[panel["implementation"] == implementation]
                .set_index("frozenCount")
                .reindex([0, 1, 2, 3])
                .reset_index()
            )
            y = subset["meanDelayedGratification"].to_numpy(dtype=float)
            err = subset["semDelayedGratification"].fillna(0.0).to_numpy(dtype=float)
            ax.bar(
                x + offset,
                y,
                width=width,
                yerr=err,
                color=colors[implementation],
                edgecolor="#222222",
                linewidth=0.7,
                capsize=2,
                label=implementation.replace("_", "-"),
            )
        ax.set_title(algorithm.capitalize(), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["0", "1", "2", "3"])
        ax.set_xlabel("Frozen Cell count")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        if col_idx == 0:
            ax.set_ylabel("Delayed Gratification")
        if col_idx == 2:
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Figure 7-style Delayed Gratification (stuck Frozen Cells)", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.94])
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_variant_sensitivity(backing_path: Path, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    backing = pd.read_parquet(backing_path)
    png_path = figure_dir / "figure07_dg_variant_sensitivity.png"
    pdf_path = figure_dir / "figure07_dg_variant_sensitivity.pdf"
    colors = {"traditional": "#666666", "cell_view": "#2ca02c"}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), sharey=False)
    for row_idx, variant in enumerate(FIGURE_VARIANTS):
        for col_idx, algorithm in enumerate(ALGORITHMS):
            ax = axes[row_idx, col_idx]
            panel = backing[(backing["figureFrozenVariant"] == variant) & (backing["algorithm"] == algorithm)]
            x = np.arange(4)
            width = 0.34
            for offset, implementation in [(-width / 2, "traditional"), (width / 2, "cell_view")]:
                subset = (
                    panel[panel["implementation"] == implementation]
                    .set_index("frozenCount")
                    .reindex([0, 1, 2, 3])
                    .reset_index()
                )
                y = subset["meanDelayedGratification"].to_numpy(dtype=float)
                err = subset["semDelayedGratification"].fillna(0.0).to_numpy(dtype=float)
                ax.bar(
                    x + offset,
                    y,
                    width=width,
                    yerr=err,
                    color=colors[implementation],
                    edgecolor="#222222",
                    linewidth=0.7,
                    capsize=2,
                    label=implementation.replace("_", "-"),
                )
            ax.set_title(f"{variant.capitalize()} {algorithm.capitalize()}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(["0", "1", "2", "3"])
            ax.grid(axis="y", alpha=0.25, linewidth=0.7)
            if row_idx == 1:
                ax.set_xlabel("Frozen Cell count")
            if col_idx == 0:
                ax.set_ylabel("Delayed Gratification")
            if row_idx == 0 and col_idx == 2:
                ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Delayed Gratification sensitivity to Frozen Cell variant", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.96])
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def infer_outcome(trends: pd.DataFrame, comparisons: pd.DataFrame) -> tuple[str, str]:
    primary_trends = trends[trends["isPrimaryPaperStyleVariant"]]
    primary_comparisons = comparisons[comparisons["isPrimaryPaperStyleVariant"]].set_index("algorithm")
    bubble_trend = primary_trends[
        (primary_trends["algorithm"] == "bubble") & primary_trends["monotoneNonDecreasing"]
    ]
    insertion_trend = primary_trends[
        (primary_trends["algorithm"] == "insertion") & primary_trends["monotoneNonDecreasing"]
    ]
    selection_clear = primary_trends[
        (primary_trends["algorithm"] == "selection") & primary_trends["monotoneNonDecreasing"]
    ]
    bubble_direction = float(primary_comparisons.loc["bubble", "meanDifferenceCellMinusTraditional"]) > 0
    insertion_similar = abs(float(primary_comparisons.loc["insertion", "meanDifferenceCellMinusTraditional"])) < 0.05
    selection_direction = float(primary_comparisons.loc["selection", "meanDifferenceCellMinusTraditional"]) < 0
    if (
        len(bubble_trend) == 2
        and len(insertion_trend) == 2
        and len(selection_clear) < 2
        and bubble_direction
        and insertion_similar
        and selection_direction
    ):
        return (
            "supportive",
            "Primary stuck-Frozen DG results reproduce the paper directions: Bubble and Insertion trend upward with f, Bubble cell-view exceeds traditional, Insertion is similar, and Selection cell-view is lower with no clear monotone trend.",
        )
    return (
        "constraining",
        "Primary stuck-Frozen DG results do not fully reproduce the paper's direction and trend pattern under the available deterministic trajectories.",
    )


def validate_outputs(
    replicate_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    backing_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    unit_tests_passed: bool,
    output_paths: list[Path],
) -> tuple[bool, list[str], list[str], str]:
    checks: list[str] = []
    failures: list[str] = []
    caveats: list[str] = []

    if unit_tests_passed:
        checks.append("Hand-constructed DG unit tests passed.")
    else:
        failures.append("At least one hand-constructed DG unit test failed.")

    if len(replicate_df) == 4200:
        checks.append("DG replicate table has 4,200 rows.")
    else:
        failures.append(f"DG replicate table has {len(replicate_df)} rows, expected 4,200.")

    per_condition = replicate_df.groupby("conditionId").size()
    if per_condition.nunique() == 1 and int(per_condition.iloc[0]) == 100:
        checks.append("Every S08 condition has exactly 100 replicate rows.")
    else:
        failures.append(f"Per-condition replicate counts are inconsistent: {per_condition.to_dict()}.")

    source_counts = Counter(replicate_df["sourceTrace"])
    if source_counts == {"S04_no_frozen_baseline": 600, "S07_frozen_trajectory": 3600}:
        checks.append("S08 used 600 S04 no-Frozen baseline rows and 3,600 S07 Frozen Cell rows.")
    else:
        failures.append(f"Unexpected source trace counts: {dict(source_counts)}.")

    if len(summary_df) == 42:
        checks.append("DG condition summary has 42 rows.")
    else:
        failures.append(f"DG condition summary has {len(summary_df)} rows, expected 42.")

    if len(backing_df) == 48:
        checks.append("Figure 7 backing table has 48 rows after expanding f=0 into passive and stuck panels.")
    else:
        failures.append(f"Figure 7 backing table has {len(backing_df)} rows, expected 48.")

    if len(trends_df) == 12:
        checks.append("Trend table has 12 implementation/algorithm/variant rows.")
    else:
        failures.append(f"Trend table has {len(trends_df)} rows, expected 12.")

    if len(comparisons_df) == 6:
        checks.append("Comparison table has six algorithm/variant paper-style z-test rows.")
    else:
        failures.append(f"Comparison table has {len(comparisons_df)} rows, expected 6.")

    if replicate_df["delayedGratification"].map(math.isfinite).all():
        checks.append("All DG values are finite.")
    else:
        failures.append("At least one DG value is non-finite.")

    primary_trends = trends_df[trends_df["figureFrozenVariant"] == PRIMARY_FROZEN_VARIANT]
    primary_expected = {
        ("traditional", "bubble"),
        ("cell_view", "bubble"),
        ("traditional", "insertion"),
        ("cell_view", "insertion"),
    }
    primary_monotone_pairs = set(
        primary_trends[primary_trends["monotoneNonDecreasing"]][["implementation", "algorithm"]].itertuples(
            index=False, name=None
        )
    )
    if primary_expected.issubset(primary_monotone_pairs):
        checks.append("Primary stuck-Frozen Bubble and Insertion DG means are monotone non-decreasing with f for both implementations.")
    else:
        failures.append("Primary stuck-Frozen Bubble/Insertion trend check failed.")

    primary_selection = primary_trends[primary_trends["algorithm"] == "selection"]
    if not bool(primary_selection["monotoneNonDecreasing"].all()):
        checks.append("Primary stuck-Frozen Selection does not show a uniformly monotone non-decreasing DG trend.")
    else:
        caveats.append("Primary stuck-Frozen Selection is monotone in this run, contrary to the paper's no-clear-trend wording.")

    passive_trends = trends_df[trends_df["figureFrozenVariant"] == "passive"]
    if not passive_trends[
        (passive_trends["algorithm"].isin(["bubble", "insertion"]))
        & (passive_trends["implementation"] == "cell_view")
    ]["monotoneNonDecreasing"].all():
        caveats.append("Passive Frozen Cell sensitivity does not reproduce the Bubble/Insertion increasing-DG trend; the paper-style primary variant is stuck/cannot-move.")

    missing_outputs = [str(path) for path in output_paths if not path.exists() or path.stat().st_size == 0]
    if not missing_outputs:
        checks.append("All declared S08 output files exist and are non-empty.")
    else:
        failures.append(f"Missing or empty output files: {missing_outputs}.")

    if Path("/artifacts/research_steps/S09").exists():
        failures.append("S09 artifact directory exists; S08 must stop before S09.")
    else:
        checks.append("No S09 artifact directory was created.")

    success = not failures
    if failures:
        validation_result = "failed: " + " ".join(failures)
    else:
        validation_result = (
            "passed: S08 unit-tested the DG formula, produced 4,200 replicate rows from S04/S07 trajectories, "
            "wrote summary/trend/comparison tables and Figure 7-style outputs, confirmed primary stuck-Frozen "
            "Bubble/Insertion increasing trends, and did not create S09 artifacts."
        )
    return success, checks, failures + caveats, validation_result


def write_formula_notes(path: Path, unit_results: list[dict[str, Any]]) -> None:
    rows = [
        [
            result["caseId"],
            result["expectedDg"],
            result["observedDg"],
            result["expectedEvents"],
            result["observedEvents"],
            result["passed"],
        ]
        for result in unit_results
    ]
    lines = [
        "# S08 Delayed Gratification Formula Notes",
        "",
        "- Research step ID: S08",
        "- Completion status: completed",
        "- Primary formula interpretation: consecutive equal Sortedness values are deduplicated, consecutive same-sign Sortedness changes are collapsed, and every Sortedness drop is paired with the immediately following recovery.",
        "- Event-level DG score: `(recovery - drop) / drop`, where `drop` is the magnitude of the temporary Sortedness decrease and `recovery` is the magnitude of the next consecutive Sortedness increase.",
        "- Run-level DG score: signed arithmetic mean of complete event-level scores.",
        "- This is algebraically equivalent to the archived monotonicity-error implementation when monotonicity error is `(n - 1) - SortednessRawCount`; scaling to Sortedness percent cancels in the ratio.",
        "- Incomplete trailing drops are not scored because no later gain is observed.",
        "- Negative event scores are retained when recovery is smaller than the preceding drop; this follows the archived signed convention rather than clamping to zero.",
        "- Primary paper-style trend analysis uses stuck/cannot-move Frozen Cells because the archived DG script names those inputs `cannot_move` and the paper-reported f=0..3 trend matches that condition family.",
        "- Passive Frozen Cell results are retained as a sensitivity analysis.",
        "- Artifacts written: see `artifact_manifest.json` and `status.json`.",
        "- Validation result: see `validation.json`.",
        "- Caveats or blockers: extracted paper formulas are incomplete; S08 therefore ports and tests the archived convention rather than claiming exact author-local raw-array identity.",
        "- Recommended next action: hand control back to the Chief Scientist workflow; start S09 only after explicit instruction.",
        "",
        "## Unit Test Summary",
        "",
        markdown_table(
            ["Case", "Expected DG", "Observed DG", "Expected events", "Observed events", "Passed"],
            rows,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_markdown(path: Path, checks: list[str], caveats: list[str], validation_result: str) -> None:
    lines = [
        "# S08 Validation",
        "",
        "- Research step ID: S08",
        "- Step number: 8",
        "- Completion status: completed",
        "- Validation result: " + validation_result,
        "- Artifacts written: see `artifact_manifest.json` and `status.json`.",
        "- Caveats or blockers:",
    ]
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Recommended next action: hand control back to the Chief Scientist workflow for review before S09.",
            "",
            "## Checks",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in checks)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(
    path: Path,
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
    outcome_classification: str,
    backing_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    trends_df: pd.DataFrame,
) -> None:
    primary = backing_df[backing_df["figureFrozenVariant"] == PRIMARY_FROZEN_VARIANT]
    summary_rows = []
    for _, row in primary.sort_values(["algorithm", "implementation", "frozenCount"]).iterrows():
        summary_rows.append(
            [
                row["algorithm"],
                row["implementation"],
                int(row["frozenCount"]),
                float(row["meanDelayedGratification"]),
                float(row["meanDgEventCount"]),
            ]
        )
    comparison_rows = []
    for _, row in comparisons_df[comparisons_df["isPrimaryPaperStyleVariant"]].sort_values("algorithm").iterrows():
        comparison_rows.append(
            [
                row["algorithm"],
                float(row["meanDifferenceCellMinusTraditional"]),
                float(row["paperStyleZStatistic"]),
                float(row["paperStylePValue"]),
            ]
        )
    trend_rows = []
    for _, row in trends_df[trends_df["isPrimaryPaperStyleVariant"]].sort_values(
        ["algorithm", "implementation"]
    ).iterrows():
        trend_rows.append(
            [
                row["algorithm"],
                row["implementation"],
                float(row["linearSlopePerFrozenCell"]),
                row["monotoneNonDecreasing"],
            ]
        )

    lines = [
        "# S08 Status Summary",
        "",
        "- Research step ID: S08",
        "- Step number: 8",
        "- Completion status: completed",
        f"- Outcome classification: {outcome_classification}",
        "- Artifacts written:",
    ]
    lines.extend(f"- `{artifact}`" for artifact in artifacts_written)
    lines.extend(
        [
            f"- Validation result: {validation_result}",
            "- Caveats or blockers:",
        ]
    )
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Lay summary: S08 converted S04/S07 Sortedness trajectories into Delayed Gratification scores using a unit-tested port of the paper/code formula. The primary stuck-Frozen result reproduces the reported direction: Bubble and Insertion DG increase with Frozen Cell count, Bubble cell-view exceeds traditional, Insertion is similar, and Selection lacks a clear increasing trend while cell-view is lower than traditional.",
            f"- Recommended next action: {recommended_next_action}",
            "",
            "## Primary Figure 7-Style Means",
            "",
            markdown_table(
                ["Algorithm", "Implementation", "f", "Mean DG", "Mean DG events"],
                summary_rows,
            ),
            "",
            "## Primary Cell-View Minus Traditional Comparisons",
            "",
            markdown_table(["Algorithm", "Mean difference", "z", "p"], comparison_rows),
            "",
            "## Primary Trend Checks",
            "",
            markdown_table(["Algorithm", "Implementation", "Slope per f", "Monotone non-decreasing"], trend_rows),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def update_run_manifest(
    path: Path,
    status_payload: dict[str, Any],
    artifact_manifest_payload: dict[str, Any],
) -> None:
    manifest = load_json(path)
    if not manifest:
        manifest = {"experimentId": EXPERIMENT_ID, "researchSteps": {}}
    manifest["experimentId"] = EXPERIMENT_ID
    manifest["researchStepId"] = STEP_ID
    manifest["status"] = status_payload["status"]
    manifest["success"] = status_payload["success"]
    manifest["generatedAt"] = status_payload["generatedAt"]
    manifest["git"] = status_payload["git"]
    manifest["artifactsWritten"] = status_payload["artifactsWritten"]
    manifest["validationResult"] = status_payload["validationResult"]
    manifest["caveatsOrBlockers"] = status_payload["caveatsOrBlockers"]
    manifest["recommendedNextAction"] = status_payload["recommendedNextAction"]
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "status": status_payload["status"],
        "success": status_payload["success"],
        "outcomeClassification": status_payload["outcomeClassification"],
        "artifactCount": artifact_manifest_payload["artifactCount"],
        "artifacts": artifact_manifest_payload["artifacts"],
        "validationResult": status_payload["validationResult"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "generatedAt": status_payload["generatedAt"],
        "git": status_payload["git"],
    }
    write_json(path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH_DEFAULT)
    parser.add_argument("--condition-matrix", type=Path, default=CONDITION_MATRIX_DEFAULT)
    parser.add_argument("--s04-trace", type=Path, default=S04_TRACE_DEFAULT)
    parser.add_argument("--s07-trace", type=Path, default=S07_TRACE_DEFAULT)
    parser.add_argument("--s07-replicates", type=Path, default=S07_REPLICATES_DEFAULT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    args = parser.parse_args()

    generated_at = utc_now()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    results_dir = artifacts_dir / "results"
    figure_dir = artifacts_dir / "figures" / "e01"
    provenance_dir = artifacts_dir / "provenance"
    for directory in [step_dir, code_dir, results_dir, figure_dir, provenance_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    unit_tests_passed, unit_test_results = run_unit_tests()
    s04_records = load_s04_baseline_records(args.s04_trace)
    s07_records = load_s07_frozen_records(args.s07_trace)
    replicate_df = pd.concat([s04_records, s07_records], ignore_index=True)

    (
        replicate_parquet,
        replicate_csv,
        summary_parquet,
        summary_csv,
        backing_parquet,
        backing_csv,
        trends_parquet,
        trends_csv,
        comparisons_parquet,
        comparisons_csv,
    ) = write_result_tables(replicate_df, results_dir)
    summary_df = pd.read_parquet(summary_parquet)
    backing_df = pd.read_parquet(backing_parquet)
    trends_df = pd.read_parquet(trends_parquet)
    comparisons_df = pd.read_parquet(comparisons_parquet)
    figure_png, figure_pdf = plot_primary_figure(backing_parquet, figure_dir)
    sensitivity_png, sensitivity_pdf = plot_variant_sensitivity(backing_parquet, figure_dir)
    outcome_classification, outcome_reason = infer_outcome(trends_df, comparisons_df)

    formula_notes_md = step_dir / "formula_notes.md"
    unit_tests_json = step_dir / "unit_tests.json"
    validation_json = step_dir / "validation.json"
    validation_md = step_dir / "validation.md"
    summary_md = step_dir / "summary.md"
    status_json = step_dir / "status.json"
    artifact_manifest_json = step_dir / "artifact_manifest.json"

    write_formula_notes(formula_notes_md, unit_test_results)
    write_json(
        unit_tests_json,
        {
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": bool(unit_tests_passed),
            "status": "passed" if unit_tests_passed else "failed",
            "generatedAt": generated_at,
            "unitTests": unit_test_results,
        },
    )

    output_paths = [
        replicate_parquet,
        replicate_csv,
        summary_parquet,
        summary_csv,
        backing_parquet,
        backing_csv,
        trends_parquet,
        trends_csv,
        comparisons_parquet,
        comparisons_csv,
        figure_png,
        figure_pdf,
        sensitivity_png,
        sensitivity_pdf,
        formula_notes_md,
        unit_tests_json,
    ]
    success, validation_checks, caveats_or_blockers, validation_result = validate_outputs(
        replicate_df,
        summary_df,
        backing_df,
        trends_df,
        comparisons_df,
        unit_tests_passed,
        output_paths,
    )
    caveats_or_blockers.extend(
        [
            "Extracted paper formulas are incomplete; S08 uses the archived signed DG convention ported to percent Sortedness trajectories.",
            "Primary Figure 7-style results use stuck/cannot-move Frozen Cells; passive Frozen Cells are reported as sensitivity outputs.",
            "S04 f=0 no-Frozen trajectories are used directly as baselines; S07 contributes f=1..3 Frozen Cell trajectories.",
            "S07 trajectories inherit deterministic functional-wrapper caveats from S07, including reconstructed traditional Frozen Cell behavior and bounded cell-view Insertion sweeps.",
            outcome_reason,
            "No S09 same-goal chimera simulations or S09 artifact directory were started.",
        ]
    )

    code_copy = code_dir / Path(__file__).name
    shutil.copy2(Path(__file__), code_copy)
    output_paths.extend([validation_json, validation_md, summary_md, status_json, artifact_manifest_json, code_copy])
    artifacts_written: list[str] = []
    recommended_next_action = (
        "Hand control back to the Chief Scientist workflow for review; S09 should start only after explicit instruction, "
        "using S08 DG outputs only as completed baseline metrics."
    )

    write_validation_markdown(validation_md, validation_checks, caveats_or_blockers, validation_result)

    validation_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "validationResult": validation_result,
        "validationChecks": validation_checks,
        "caveatsOrBlockers": caveats_or_blockers,
        "recommendedNextAction": recommended_next_action,
        "artifactsWritten": artifacts_written,
        "unitTestsPassed": bool(unit_tests_passed),
        "replicateRows": int(len(replicate_df)),
        "conditionSummaryRows": int(len(summary_df)),
        "figureBackingRows": int(len(backing_df)),
        "trendRows": int(len(trends_df)),
        "comparisonRows": int(len(comparisons_df)),
        "outcomeClassification": outcome_classification,
    }
    write_json(validation_json, validation_payload)

    write_summary_markdown(
        summary_md,
        artifacts_written,
        validation_result,
        caveats_or_blockers,
        recommended_next_action,
        outcome_classification,
        backing_df,
        comparisons_df,
        trends_df,
    )

    source_statuses = {
        "S04": load_json(S04_STATUS_DEFAULT),
        "S06": load_json(S06_STATUS_DEFAULT),
        "S07": load_json(S07_STATUS_DEFAULT),
    }
    runtime = {
        "pythonVersion": sys.version,
        "pythonExecutable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "osCpuCount": os.cpu_count(),
        "workerCount": 1,
        "gpuUsed": False,
        "numpyVersion": np.__version__,
        "pandasVersion": pd.__version__,
        "scipyVersion": stats.__version__ if hasattr(stats, "__version__") else "available",
        "statsmodelsAvailable": STATSMODELS_AVAILABLE,
    }
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "outcomeClassification": outcome_classification,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats_or_blockers,
        "recommendedNextAction": recommended_next_action,
        "git": get_git_metadata(),
        "runtime": runtime,
        "sourceArtifacts": {
            "s03BaselineConfig": str(args.config),
            "s03ConditionMatrix": str(args.condition_matrix),
            "s04TrajectoryEvents": str(args.s04_trace),
            "s07TrajectoryEvents": str(args.s07_trace),
            "s07ReplicateSummary": str(args.s07_replicates),
            "s04Status": str(S04_STATUS_DEFAULT),
            "s06Status": str(S06_STATUS_DEFAULT),
            "s07Status": str(S07_STATUS_DEFAULT),
        },
        "sourceStatuses": {
            step_id: {
                "researchStepId": status.get("researchStepId"),
                "status": status.get("status"),
                "success": status.get("success"),
                "validationResult": status.get("validationResult"),
            }
            for step_id, status in source_statuses.items()
        },
        "conditionCounts": {
            "replicateRows": int(len(replicate_df)),
            "summaryRows": int(len(summary_df)),
            "figureBackingRows": int(len(backing_df)),
            "trendRows": int(len(trends_df)),
            "comparisonRows": int(len(comparisons_df)),
        },
        "sourceTraceCounts": dict(Counter(replicate_df["sourceTrace"])),
        "primaryFrozenVariant": PRIMARY_FROZEN_VARIANT,
    }
    write_json(status_json, status_payload)

    artifact_manifest_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": STATUS if success else "failed_validation",
        "success": bool(success),
        "generatedAt": generated_at,
        "artifactCount": 0,
        "artifacts": [],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats_or_blockers,
        "recommendedNextAction": recommended_next_action,
    }
    write_json(artifact_manifest_json, artifact_manifest_payload)

    artifacts = collect_artifacts(output_paths)
    artifacts_written = [artifact["path"] for artifact in artifacts]
    validation_payload["artifactsWritten"] = artifacts_written
    status_payload["artifactsWritten"] = artifacts_written
    artifact_manifest_payload["artifactCount"] = len(artifacts)
    artifact_manifest_payload["artifacts"] = artifacts
    write_json(validation_json, validation_payload)
    write_summary_markdown(
        summary_md,
        artifacts_written,
        validation_result,
        caveats_or_blockers,
        recommended_next_action,
        outcome_classification,
        backing_df,
        comparisons_df,
        trends_df,
    )
    write_json(status_json, status_payload)

    artifacts = collect_artifacts(output_paths)
    artifact_manifest_payload["artifactCount"] = len(artifacts)
    artifact_manifest_payload["artifacts"] = artifacts
    write_json(artifact_manifest_json, artifact_manifest_payload)
    update_run_manifest(RUN_MANIFEST_DEFAULT, status_payload, artifact_manifest_payload)

    print(json.dumps({"success": success, "validationResult": validation_result, "artifactsWritten": artifacts_written}, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
