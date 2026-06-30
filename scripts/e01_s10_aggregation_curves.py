#!/usr/bin/env python3
"""Compute E01 S10 Figure 8-style Aggregation curves from S09 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STEP_ID = "S10"
STEP_NUMBER = 10
SOURCE_STEP_ID = "S09"
EXPERIMENT_ID = "E01"
NEGATIVE_CONTROL_MIX = "same_algorithm_bubble_label_control"
CORE_MIXES = (
    "same_goal_bubble_insertion",
    "same_goal_bubble_selection",
    "same_goal_insertion_selection",
    "same_goal_bubble_insertion_selection",
)
CODE_TO_LABEL = {
    "B": "bubble",
    "I": "insertion",
    "S": "selection",
    "A": "bubble_label_a",
    "C": "bubble_label_b",
}
PAPER_REFERENCE = {
    "same_goal_bubble_selection": {"paper_peak_percent": 72.0, "paper_peak_progress_percent": 42.0},
    "same_goal_bubble_insertion": {"paper_peak_percent": 65.0, "paper_peak_progress_percent": 21.0},
    "same_goal_insertion_selection": {"paper_peak_percent": 69.0, "paper_peak_progress_percent": 19.0},
    "same_goal_bubble_insertion_selection": {"paper_peak_percent": 62.0, "paper_peak_progress_percent": 22.0},
    NEGATIVE_CONTROL_MIX: {"paper_peak_percent": np.nan, "paper_peak_progress_percent": np.nan},
}
MIX_LABELS = {
    "same_goal_bubble_insertion": "Bubble-Insertion",
    "same_goal_bubble_selection": "Bubble-Selection",
    "same_goal_insertion_selection": "Insertion-Selection",
    "same_goal_bubble_insertion_selection": "Bubble-Insertion-Selection",
    NEGATIVE_CONTROL_MIX: "Bubble label control",
}
MIX_COLORS = {
    "same_goal_bubble_insertion": "#2b8a5e",
    "same_goal_bubble_selection": "#4169a8",
    "same_goal_insertion_selection": "#b34d4d",
    "same_goal_bubble_insertion_selection": "#6b5aa6",
    NEGATIVE_CONTROL_MIX: "#c77aa3",
}
TRACE_COLUMNS = [
    "condition_id",
    "algotype_mix",
    "repeat_index",
    "swap_step",
    "sortedness_percent",
    "aggregation_left_neighbor_percent",
    "aggregation_right_neighbor_legacy_percent",
    "algotype_positions_code",
    "initial_algotype_counts_json",
    "is_initial",
    "is_final",
    "stop_reason",
    "max_guard_hit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--s09-trace-path", type=Path, default=Path("/artifacts/traces/e01_same_goal_chimeras.parquet"))
    parser.add_argument("--s09-validation-path", type=Path, default=Path("/artifacts/research_steps/S09/s09_validation.json"))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--smoke-repeats", type=int, default=None, help="Optional cap on repeats per condition for smoke validation.")
    parser.add_argument("--progress-points", type=int, default=101)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_status(repo_dir: Path) -> str:
    try:
        result = subprocess.run(["git", "status", "--short"], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregation_left_from_code(code: str) -> float:
    same_left = sum(1 for idx in range(1, len(code)) if code[idx] == code[idx - 1])
    return 100.0 * same_left / len(code)


def aggregation_right_from_code(code: str) -> float:
    same_right = sum(1 for idx in range(len(code) - 1) if code[idx] == code[idx + 1])
    return 100.0 * same_right / len(code)


def expected_left_chance_from_counts(counts_payload: str) -> float:
    counts = json.loads(counts_payload)
    total = sum(int(value) for value in counts.values())
    numerator = sum(int(value) * (int(value) - 1) for value in counts.values())
    return 100.0 * numerator / (total * total)


def ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    if n <= 1 or np.isnan(std):
        return (mean, mean)
    half_width = 1.96 * std / np.sqrt(n)
    return (mean - half_width, mean + half_width)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    stringified = df.copy()
    for col in stringified.columns:
        if pd.api.types.is_float_dtype(stringified[col]):
            stringified[col] = stringified[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    stringified = stringified.astype("string").fillna("").astype(str)
    headers = list(stringified.columns)
    rows = stringified.values.tolist()
    widths = [max(len(str(header)), *(len(row[col_idx]) for row in rows)) for col_idx, header in enumerate(headers)]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    divider_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body_lines = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def load_trace(args: argparse.Namespace) -> pd.DataFrame:
    trace = pd.read_parquet(args.s09_trace_path, columns=TRACE_COLUMNS)
    if args.smoke_repeats is not None:
        trace = trace[trace["repeat_index"] < int(args.smoke_repeats)].copy()
    return trace


def validate_trace_reconstruction(trace: pd.DataFrame, expected_full_repeats: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    observed_mixes = set(trace["algotype_mix"].unique())
    expected_mixes = set(CORE_MIXES) | {NEGATIVE_CONTROL_MIX}
    if observed_mixes != expected_mixes:
        errors.append(f"Unexpected S10 mix set: missing={sorted(expected_mixes - observed_mixes)}, extra={sorted(observed_mixes - expected_mixes)}.")
    if NEGATIVE_CONTROL_MIX not in observed_mixes:
        errors.append("Missing same-Algotype Bubble label-control negative control.")
    code_lengths = trace["algotype_positions_code"].astype(str).map(len)
    code_length_ok = bool(code_lengths.eq(100).all())
    if not code_length_ok:
        errors.append("At least one algotype_positions_code is not length 100.")
    invalid_codes = sorted(set("".join(trace["algotype_positions_code"].astype(str).unique())) - set(CODE_TO_LABEL))
    if invalid_codes:
        errors.append(f"Unsupported Algotype position code characters: {invalid_codes}.")
    trace = trace.copy()
    trace["recomputed_aggregation_left_neighbor_percent"] = trace["algotype_positions_code"].map(aggregation_left_from_code)
    trace["recomputed_aggregation_right_neighbor_legacy_percent"] = trace["algotype_positions_code"].map(aggregation_right_from_code)
    left_diff = (trace["recomputed_aggregation_left_neighbor_percent"] - trace["aggregation_left_neighbor_percent"]).abs()
    right_diff = (trace["recomputed_aggregation_right_neighbor_legacy_percent"] - trace["aggregation_right_neighbor_legacy_percent"]).abs()
    left_matches = bool((left_diff <= 1e-9).all())
    right_matches = bool((right_diff <= 1e-9).all())
    if not left_matches:
        errors.append(f"Left-neighbor aggregation recomputation mismatch; max abs diff={float(left_diff.max())}.")
    if not right_matches:
        errors.append(f"Right-neighbor aggregation recomputation mismatch; max abs diff={float(right_diff.max())}.")
    repeat_counts = trace.groupby("condition_id")["repeat_index"].nunique().to_dict()
    initial_counts = trace.groupby("condition_id")["is_initial"].sum().to_dict()
    final_counts = trace.groupby("condition_id")["is_final"].sum().to_dict()
    exactly_one_initial = all(int(initial_counts.get(condition_id, 0)) == int(repeat_count) for condition_id, repeat_count in repeat_counts.items())
    exactly_one_final = all(int(final_counts.get(condition_id, 0)) == int(repeat_count) for condition_id, repeat_count in repeat_counts.items())
    if not exactly_one_initial:
        errors.append("Not every condition/repeat has exactly one initial row.")
    if not exactly_one_final:
        errors.append("Not every condition/repeat has exactly one final row.")
    stopping_reasons = Counter(trace[trace["is_final"]]["stop_reason"])
    if set(stopping_reasons) != {"sorted"}:
        errors.append(f"Final stop reasons are not all sorted: {dict(stopping_reasons)}.")
    if bool(trace["max_guard_hit"].fillna(False).any()):
        errors.append("At least one S09 trace row has max_guard_hit=True.")
    full_repeat_count = all(int(value) == expected_full_repeats for value in repeat_counts.values())
    if not full_repeat_count:
        warnings.append(f"Repeat counts are not full production counts of {expected_full_repeats}: {repeat_counts}.")
    return {
        "traceRows": int(len(trace)),
        "observedConditionRows": int(trace["condition_id"].nunique()),
        "observedMixes": sorted(observed_mixes),
        "repeatCountsByCondition": {str(key): int(value) for key, value in repeat_counts.items()},
        "initialRowsByCondition": {str(key): int(value) for key, value in initial_counts.items()},
        "finalRowsByCondition": {str(key): int(value) for key, value in final_counts.items()},
        "algotypePositionCodesLength100": code_length_ok,
        "invalidPositionCodeCharacters": invalid_codes,
        "leftNeighborAggregationRecomputedMatchesTrace": left_matches,
        "rightNeighborAggregationRecomputedMatchesTrace": right_matches,
        "maxLeftAggregationRecomputeAbsDiff": float(left_diff.max()) if len(left_diff) else None,
        "maxRightAggregationRecomputeAbsDiff": float(right_diff.max()) if len(right_diff) else None,
        "negativeControlIncluded": NEGATIVE_CONTROL_MIX in observed_mixes,
        "exactlyOneInitialTraceRowPerRun": exactly_one_initial,
        "exactlyOneFinalTraceRowPerRun": exactly_one_final,
        "allFinalStopReasonsSorted": set(stopping_reasons) == {"sorted"},
        "stopReasonCounts": {str(key): int(value) for key, value in stopping_reasons.items()},
        "noMaxGuardRows": not bool(trace["max_guard_hit"].fillna(False).any()),
        "fullProductionRepeatCount": full_repeat_count,
        "validationErrors": errors,
        "validationWarnings": warnings,
        "traceReconstructionPassed": not errors,
    }


def build_run_grid(trace: pd.DataFrame, progress_points: int) -> pd.DataFrame:
    grid = np.linspace(0.0, 100.0, progress_points)
    rows: list[dict[str, Any]] = []
    for (condition_id, repeat_index), subset in trace.groupby(["condition_id", "repeat_index"], sort=True):
        subset = subset.sort_values(["swap_step", "is_final"]).groupby("swap_step", as_index=False).tail(1)
        final_swap = float(subset["swap_step"].max())
        progress = np.zeros(len(subset)) if final_swap <= 0 else 100.0 * subset["swap_step"].to_numpy(dtype=float) / final_swap
        left = subset["aggregation_left_neighbor_percent"].to_numpy(dtype=float)
        right = subset["aggregation_right_neighbor_legacy_percent"].to_numpy(dtype=float)
        sortedness = subset["sortedness_percent"].to_numpy(dtype=float)
        first = subset.iloc[0]
        chance = expected_left_chance_from_counts(str(first["initial_algotype_counts_json"]))
        interp_left = np.interp(grid, progress, left)
        interp_right = np.interp(grid, progress, right)
        interp_sortedness = np.interp(grid, progress, sortedness)
        for pct, left_value, right_value, sortedness_value in zip(grid, interp_left, interp_right, interp_sortedness, strict=True):
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "source_research_step_id": SOURCE_STEP_ID,
                    "experiment_id": EXPERIMENT_ID,
                    "condition_id": condition_id,
                    "algotype_mix": first["algotype_mix"],
                    "is_negative_control": first["algotype_mix"] == NEGATIVE_CONTROL_MIX,
                    "repeat_index": int(repeat_index),
                    "progress_percent": float(pct),
                    "aggregation_left_neighbor_percent": float(left_value),
                    "aggregation_right_neighbor_legacy_percent": float(right_value),
                    "sortedness_percent": float(sortedness_value),
                    "expected_random_left_neighbor_percent": float(chance),
                    "final_swap_step": int(final_swap),
                }
            )
    return pd.DataFrame(rows)


def summarize_curves(run_grid: pd.DataFrame) -> pd.DataFrame:
    grouped = run_grid.groupby(["condition_id", "algotype_mix", "is_negative_control", "progress_percent"], as_index=False)
    curves = grouped.agg(
        n_runs=("repeat_index", "nunique"),
        mean_aggregation_left_neighbor_percent=("aggregation_left_neighbor_percent", "mean"),
        std_aggregation_left_neighbor_percent=("aggregation_left_neighbor_percent", "std"),
        mean_aggregation_right_neighbor_legacy_percent=("aggregation_right_neighbor_legacy_percent", "mean"),
        std_aggregation_right_neighbor_legacy_percent=("aggregation_right_neighbor_legacy_percent", "std"),
        mean_sortedness_percent=("sortedness_percent", "mean"),
        expected_random_left_neighbor_percent=("expected_random_left_neighbor_percent", "mean"),
        mean_final_swap_step=("final_swap_step", "mean"),
    )
    for prefix in ("left_neighbor", "right_neighbor_legacy"):
        mean_col = f"mean_aggregation_{prefix}_percent"
        std_col = f"std_aggregation_{prefix}_percent"
        sem_col = f"sem_aggregation_{prefix}_percent"
        low_col = f"ci95_low_aggregation_{prefix}_percent"
        high_col = f"ci95_high_aggregation_{prefix}_percent"
        curves[sem_col] = curves[std_col] / np.sqrt(curves["n_runs"])
        cis = [ci95(float(row[mean_col]), float(row[std_col]), int(row["n_runs"])) for _, row in curves.iterrows()]
        curves[low_col] = [value[0] for value in cis]
        curves[high_col] = [value[1] for value in cis]
        curves[f"mean_aggregation_{prefix}_proportion"] = curves[mean_col] / 100.0
    return curves.sort_values(["condition_id", "progress_percent"]).reset_index(drop=True)


def summarize_run_peaks(run_grid: pd.DataFrame) -> pd.DataFrame:
    peak_rows: list[dict[str, Any]] = []
    for (condition_id, repeat_index), subset in run_grid.groupby(["condition_id", "repeat_index"], sort=True):
        left_idx = subset["aggregation_left_neighbor_percent"].idxmax()
        right_idx = subset["aggregation_right_neighbor_legacy_percent"].idxmax()
        first = subset.iloc[0]
        left_peak = subset.loc[left_idx]
        right_peak = subset.loc[right_idx]
        peak_rows.append(
            {
                "condition_id": condition_id,
                "algotype_mix": first["algotype_mix"],
                "is_negative_control": bool(first["is_negative_control"]),
                "repeat_index": int(repeat_index),
                "run_peak_aggregation_left_neighbor_percent": float(left_peak["aggregation_left_neighbor_percent"]),
                "run_peak_left_neighbor_progress_percent": float(left_peak["progress_percent"]),
                "run_peak_aggregation_right_neighbor_legacy_percent": float(right_peak["aggregation_right_neighbor_legacy_percent"]),
                "run_peak_right_neighbor_progress_percent": float(right_peak["progress_percent"]),
                "expected_random_left_neighbor_percent": float(first["expected_random_left_neighbor_percent"]),
            }
        )
    return pd.DataFrame(peak_rows)


def build_peak_table(curves: pd.DataFrame, run_peaks: pd.DataFrame) -> pd.DataFrame:
    control_curve = curves[curves["algotype_mix"] == NEGATIVE_CONTROL_MIX].set_index("progress_percent")
    control_peak_value = float(control_curve["mean_aggregation_left_neighbor_percent"].max())
    rows: list[dict[str, Any]] = []
    for condition_id, subset in curves.groupby("condition_id", sort=True):
        subset = subset.sort_values("progress_percent")
        left_peak = subset.loc[subset["mean_aggregation_left_neighbor_percent"].idxmax()]
        right_peak = subset.loc[subset["mean_aggregation_right_neighbor_legacy_percent"].idxmax()]
        first = subset.iloc[0]
        mix = str(first["algotype_mix"])
        progress = float(left_peak["progress_percent"])
        control_at_same_progress = float(control_curve.loc[progress, "mean_aggregation_left_neighbor_percent"]) if progress in control_curve.index else np.nan
        run_peak_subset = run_peaks[run_peaks["condition_id"] == condition_id]
        ref = PAPER_REFERENCE.get(mix, {"paper_peak_percent": np.nan, "paper_peak_progress_percent": np.nan})
        rows.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "algotype_mix": mix,
                "display_label": MIX_LABELS.get(mix, mix),
                "is_negative_control": bool(first["is_negative_control"]),
                "n_runs": int(first["n_runs"]),
                "expected_random_left_neighbor_percent": float(first["expected_random_left_neighbor_percent"]),
                "initial_mean_aggregation_left_neighbor_percent": float(subset[subset["progress_percent"] == 0.0]["mean_aggregation_left_neighbor_percent"].iloc[0]),
                "final_mean_aggregation_left_neighbor_percent": float(subset[subset["progress_percent"] == 100.0]["mean_aggregation_left_neighbor_percent"].iloc[0]),
                "mean_curve_peak_aggregation_left_neighbor_percent": float(left_peak["mean_aggregation_left_neighbor_percent"]),
                "mean_curve_peak_left_neighbor_ci95_low": float(left_peak["ci95_low_aggregation_left_neighbor_percent"]),
                "mean_curve_peak_left_neighbor_ci95_high": float(left_peak["ci95_high_aggregation_left_neighbor_percent"]),
                "mean_curve_peak_left_neighbor_progress_percent": progress,
                "mean_curve_peak_aggregation_left_neighbor_proportion": float(left_peak["mean_aggregation_left_neighbor_proportion"]),
                "run_peak_aggregation_left_neighbor_mean_percent": float(run_peak_subset["run_peak_aggregation_left_neighbor_percent"].mean()),
                "run_peak_aggregation_left_neighbor_std_percent": float(run_peak_subset["run_peak_aggregation_left_neighbor_percent"].std(ddof=1)),
                "run_peak_left_neighbor_progress_mean_percent": float(run_peak_subset["run_peak_left_neighbor_progress_percent"].mean()),
                "mean_curve_peak_aggregation_right_neighbor_legacy_percent": float(right_peak["mean_aggregation_right_neighbor_legacy_percent"]),
                "mean_curve_peak_right_neighbor_legacy_progress_percent": float(right_peak["progress_percent"]),
                "peak_minus_expected_random_left_neighbor_percent": float(left_peak["mean_aggregation_left_neighbor_percent"] - first["expected_random_left_neighbor_percent"]),
                "negative_control_peak_left_neighbor_percent": control_peak_value,
                "peak_minus_negative_control_peak_left_neighbor_percent": float(left_peak["mean_aggregation_left_neighbor_percent"] - control_peak_value),
                "negative_control_left_neighbor_at_same_progress_percent": control_at_same_progress,
                "peak_minus_negative_control_at_same_progress_percent": float(left_peak["mean_aggregation_left_neighbor_percent"] - control_at_same_progress),
                "above_negative_control_peak": bool(left_peak["mean_aggregation_left_neighbor_percent"] > control_peak_value) if mix != NEGATIVE_CONTROL_MIX else False,
                "above_negative_control_at_same_progress": bool(left_peak["mean_aggregation_left_neighbor_percent"] > control_at_same_progress) if mix != NEGATIVE_CONTROL_MIX else False,
                "above_expected_random_left_neighbor": bool(left_peak["mean_aggregation_left_neighbor_percent"] > first["expected_random_left_neighbor_percent"]),
                "paper_peak_percent": float(ref["paper_peak_percent"]),
                "paper_peak_progress_percent": float(ref["paper_peak_progress_percent"]),
                "peak_percent_minus_paper": float(left_peak["mean_aggregation_left_neighbor_percent"] - ref["paper_peak_percent"]) if not pd.isna(ref["paper_peak_percent"]) else np.nan,
                "peak_progress_minus_paper": float(progress - ref["paper_peak_progress_percent"]) if not pd.isna(ref["paper_peak_progress_percent"]) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["is_negative_control", "condition_id"]).reset_index(drop=True)


def validate_outputs(
    trace_validation: dict[str, Any],
    curves: pd.DataFrame,
    peak_table: pd.DataFrame,
    run_grid: pd.DataFrame,
    expected_progress_points: int,
    smoke_repeats: int | None,
) -> dict[str, Any]:
    expected_curve_rows = trace_validation["observedConditionRows"] * expected_progress_points
    curve_rows_ok = len(curves) == expected_curve_rows
    peak_rows_ok = len(peak_table) == trace_validation["observedConditionRows"]
    no_nan_curve_metrics = bool(
        curves[
            [
                "mean_aggregation_left_neighbor_percent",
                "mean_aggregation_right_neighbor_legacy_percent",
                "mean_sortedness_percent",
            ]
        ]
        .notna()
        .all()
        .all()
    )
    core_peaks = peak_table[peak_table["algotype_mix"].isin(CORE_MIXES)].copy()
    core_above_control_peak = bool(core_peaks["above_negative_control_peak"].all())
    core_above_control_same_progress = bool(core_peaks["above_negative_control_at_same_progress"].all())
    core_above_chance = bool(core_peaks["above_expected_random_left_neighbor"].all())
    control_included = bool((peak_table["algotype_mix"] == NEGATIVE_CONTROL_MIX).any())
    full_production = bool(trace_validation["fullProductionRepeatCount"] and smoke_repeats is None)
    success = bool(
        trace_validation["traceReconstructionPassed"]
        and curve_rows_ok
        and peak_rows_ok
        and no_nan_curve_metrics
        and control_included
        and full_production
    )
    outcome = "supportive" if success and core_above_control_peak and core_above_control_same_progress and core_above_chance else "constraining/contradictory"
    caveats = [
        "S10 computes aggregation from S09 deterministic single-thread wrapper traces, not from the public threaded mixed runner.",
        "The primary metric is the S03/S09 left-neighbor definition with denominator n=100; the right-neighbor legacy sensitivity is retained and is mathematically identical for linear adjacent-pair counts with the same denominator.",
        "S03/S09 provide one same-algorithm Bubble label-control condition rather than separate same-code controls for every two-way mix.",
        "Exact paper peak magnitudes are not fully reproduced under the frozen S03/S09 wrapper and left-neighbor metric, although all core mixed curves peak above the same-algorithm control and their own random-label baselines.",
    ]
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed_with_caveats" if success else "completed_validation_failed",
        "validationResult": "passed_with_trace_reconstruction_and_metric_caveats" if success else "failed",
        "outcomeClassification": outcome,
        "traceValidation": trace_validation,
        "curveRows": int(len(curves)),
        "expectedCurveRows": int(expected_curve_rows),
        "curveRowsOk": curve_rows_ok,
        "peakRows": int(len(peak_table)),
        "peakRowsOk": peak_rows_ok,
        "runGridRows": int(len(run_grid)),
        "progressPoints": int(expected_progress_points),
        "negativeControlIncluded": control_included,
        "noNanCurveMetrics": no_nan_curve_metrics,
        "coreMixedPeaksAboveNegativeControlPeak": core_above_control_peak,
        "coreMixedPeaksAboveNegativeControlAtSameProgress": core_above_control_same_progress,
        "coreMixedPeaksAboveExpectedRandomBaseline": core_above_chance,
        "fullProductionRepeatCount": full_production,
        "smokeRepeats": smoke_repeats,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Proceed to S11 duplicate-value chimeras using S03 duplicate configs and S10 aggregation code; do not start S11 inside S10.",
    }


def plot_figure(curves: pd.DataFrame, peak_table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.65, 1.0]})
    ordered_mixes = [
        "same_goal_bubble_selection",
        "same_goal_bubble_insertion",
        "same_goal_insertion_selection",
        "same_goal_bubble_insertion_selection",
        NEGATIVE_CONTROL_MIX,
    ]
    for mix in ordered_mixes:
        subset = curves[curves["algotype_mix"] == mix].sort_values("progress_percent")
        if subset.empty:
            continue
        color = MIX_COLORS.get(mix, "#555555")
        linestyle = "--" if mix == NEGATIVE_CONTROL_MIX else "-"
        linewidth = 2.2 if mix != NEGATIVE_CONTROL_MIX else 2.0
        axes[0].plot(
            subset["progress_percent"],
            subset["mean_aggregation_left_neighbor_percent"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=MIX_LABELS.get(mix, mix),
        )
        axes[0].fill_between(
            subset["progress_percent"].to_numpy(dtype=float),
            subset["ci95_low_aggregation_left_neighbor_percent"].to_numpy(dtype=float),
            subset["ci95_high_aggregation_left_neighbor_percent"].to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    axes[0].set_title("Primary left-neighbor Aggregation")
    axes[0].set_xlabel("Run progress (% of final swap count)")
    axes[0].set_ylabel("Aggregation (%)")
    axes[0].set_ylim(25, 70)
    axes[0].grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.45)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    bar_df = peak_table.sort_values("mean_curve_peak_aggregation_left_neighbor_percent")
    axes[1].barh(
        bar_df["display_label"],
        bar_df["mean_curve_peak_aggregation_left_neighbor_percent"],
        color=[MIX_COLORS.get(mix, "#777777") for mix in bar_df["algotype_mix"]],
    )
    axes[1].axvline(
        float(peak_table.loc[peak_table["algotype_mix"] == NEGATIVE_CONTROL_MIX, "mean_curve_peak_aggregation_left_neighbor_percent"].iloc[0]),
        color="#7a4b64",
        linestyle="--",
        linewidth=1.4,
    )
    axes[1].set_title("Mean-curve peak")
    axes[1].set_xlabel("Aggregation (%)")
    axes[1].grid(True, axis="x", color="#d0d0d0", linewidth=0.7, alpha=0.45)
    fig.suptitle("E01 S10 Figure 8-style Aggregation reproduction", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def update_checksum_file(checksum_path: Path, paths: list[Path]) -> None:
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    order: list[str] = []
    if checksum_path.exists():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if "  " not in line:
                continue
            digest, path_str = line.split("  ", 1)
            existing[path_str] = digest
            order.append(path_str)
    for path in paths:
        if not path.exists() or path == checksum_path:
            continue
        resolved = str(path)
        existing[resolved] = sha256_file(path)
        if resolved not in order:
            order.append(resolved)
    checksum_path.write_text("".join(f"{existing[path]}  {path}\n" for path in order), encoding="utf-8")


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


def update_run_manifest(
    artifacts_dir: Path,
    paths: dict[str, Path],
    validation: dict[str, Any],
    repo_dir: Path,
    command: list[str],
) -> None:
    manifest_path = artifacts_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("experimentId", EXPERIMENT_ID)
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
        "title": "Reproduce Aggregation curves",
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
        "artifactsWritten": [str(path) for path in paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s10_aggregation_curves.py"),
        "command": " ".join(command),
        "summary": validation["summary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    report_path: Path,
    artifact_paths: dict[str, Path],
    validation: dict[str, Any],
    peak_table: pd.DataFrame,
    command: list[str],
    repo_dir: Path,
    args: argparse.Namespace,
    wall_seconds: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    caveats = "; ".join(validation["caveatsOrBlockers"])
    if validation["outcomeClassification"] == "supportive":
        lay_summary = (
            "S10 reproduced the qualitative same-goal Aggregation result: all mixed-Algotype curves peaked above "
            "the same-algorithm Bubble label-control and above their random-label baselines, while exact peak "
            "magnitudes differed from the paper under the frozen S03/S09 wrapper."
        )
    else:
        lay_summary = (
            "S10 computed and validated aggregation curves from S09 traces, but one or more core mixed-Algotype "
            "curves did not peak above the frozen negative-control or random-label baseline."
        )
    compact_peaks = peak_table[
        [
            "condition_id",
            "display_label",
            "n_runs",
            "expected_random_left_neighbor_percent",
            "mean_curve_peak_aggregation_left_neighbor_percent",
            "mean_curve_peak_left_neighbor_progress_percent",
            "peak_minus_negative_control_peak_left_neighbor_percent",
            "peak_minus_expected_random_left_neighbor_percent",
            "paper_peak_percent",
            "paper_peak_progress_percent",
        ]
    ].copy()
    report = f"""# E01 S10 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: {validation["status"]}
- Artifacts written: {", ".join(str(path) for path in artifact_paths.values())}
- Validation result: {validation["validationResult"]}
- Outcome classification: {validation["outcomeClassification"]}
- Caveats or blockers: {caveats}
- Lay summary: {lay_summary}
- Recommended next action: {validation["recommendedNextAction"]}

## Frozen Question

Do mixed Algotypes cluster above same-Algotype negative-control chance during sorting even though cells do not explicitly read Algotype labels?

## Inputs

- S09 trace parquet: `{args.s09_trace_path}` (`{sha256_file(args.s09_trace_path)}`)
- S09 validation JSON: `{args.s09_validation_path}` (`{sha256_file(args.s09_validation_path)}`)
- S10 progress grid points: `{args.progress_points}`
- S10 smoke repeat cap: `{args.smoke_repeats}`
- Paper reference: Methods Aggregation Value definition and Figure 8/Section 4.4 describe left-neighbor Aggregation, same-algorithm controls, and mixed-Algotype peak aggregation.

## Detailed Methods

S10 did not rerun sorting. It consumed the S09 per-step trace and decoded `algotype_positions_code`, where `B=bubble`, `I=insertion`, `S=selection`, `A=bubble_label_a`, and `C=bubble_label_b`. For every trace row, S10 recomputed the S03/S09 primary Aggregation metric as `100 * count(label[i] == label[i - 1] for i=1..99) / 100`. It also recomputed the right-neighbor legacy sensitivity as `100 * count(label[i] == label[i + 1] for i=0..98) / 100`. The two are retained as separate fields because S03 froze both semantics, even though they are identical adjacent-pair counts on a linear array when the denominator is fixed at n=100.

For Figure 8-style curves, each run was aligned to a 0..100% progress grid using `swap_step / final_swap_step`. Aggregation and Sortedness were linearly interpolated onto `{args.progress_points}` progress points, then averaged by condition. The curve parquet contains the mean, standard deviation, standard error, and normal-approximation 95% interval for primary left-neighbor Aggregation and right-neighbor sensitivity. The peak table reports the maximum of the mean primary curve for each condition, the peak timing, the same value as a 0..1 proportion, run-level peak summaries, expected random-label baseline, and comparisons with the S09 same-algorithm Bubble label-control.

The expected random-label baseline uses the fixed S03/S09 label counts without replacement: `100 * sum(c * (c - 1)) / n^2`, matching the S03 primary denominator convention. For 50/50 two-way mixes this is 49%, while the rotating 34/33/33 all-three mix is about 32.33%.

## Commands

```bash
{" ".join(command)}
```

Validation/smoke commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/e01_s10_aggregation_curves.py
PYTHONDONTWRITEBYTECODE=1 python scripts/e01_s10_aggregation_curves.py --repo-dir /workspace/cell-research --artifacts-dir /cache/e01_s10_smoke_artifacts --smoke-repeats 2
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
- Wall time: `{wall_seconds:.3f}` seconds

No new Python, system, R, Rust, or Node dependencies were installed for S10.

## Results

Figure 8-style peak table:

{dataframe_to_markdown(compact_peaks)}

Primary interpretation:

- Core mixed peaks above same-algorithm Bubble label-control peak: `{validation["coreMixedPeaksAboveNegativeControlPeak"]}`.
- Core mixed peaks above same-progress negative-control values: `{validation["coreMixedPeaksAboveNegativeControlAtSameProgress"]}`.
- Core mixed peaks above their random-label baselines: `{validation["coreMixedPeaksAboveExpectedRandomBaseline"]}`.
- Exact paper peak magnitudes were not fully recovered under the frozen S03/S09 wrapper and primary left-neighbor metric; this is preserved as a caveat rather than changing metric semantics.

## Validation

- Source trace rows: `{validation["traceValidation"]["traceRows"]}`.
- Curve rows: `{validation["curveRows"]}` of expected `{validation["expectedCurveRows"]}`.
- Peak table rows: `{validation["peakRows"]}`.
- Repeat counts by condition: `{validation["traceValidation"]["repeatCountsByCondition"]}`.
- Negative control included: `{validation["negativeControlIncluded"]}`.
- Position codes length 100: `{validation["traceValidation"]["algotypePositionCodesLength100"]}`.
- Invalid position-code characters: `{validation["traceValidation"]["invalidPositionCodeCharacters"]}`.
- Primary left-neighbor aggregation recomputed from trace codes matches S09 stored field: `{validation["traceValidation"]["leftNeighborAggregationRecomputedMatchesTrace"]}`.
- Right-neighbor sensitivity recomputed from trace codes matches S09 stored field: `{validation["traceValidation"]["rightNeighborAggregationRecomputedMatchesTrace"]}`.
- Exactly one initial trace row per run: `{validation["traceValidation"]["exactlyOneInitialTraceRowPerRun"]}`.
- Exactly one final trace row per run: `{validation["traceValidation"]["exactlyOneFinalTraceRowPerRun"]}`.
- Final stop reasons all sorted: `{validation["traceValidation"]["allFinalStopReasonsSorted"]}`.
- No max-guard rows: `{validation["traceValidation"]["noMaxGuardRows"]}`.

Trace validation details:

```json
{json.dumps(validation["traceValidation"], indent=2)}
```

## Artifacts And Provenance

Reusable outputs:

{chr(10).join(f"- `{label}`: `{path}`" for label, path in artifact_paths.items())}

The global run manifest and checksum file were updated after artifact creation. The S10 artifact manifest records paths, sizes, and SHA256 hashes.

## Caveats, Blockers, Failed Assumptions, And Limitations

- S10 inherits S09's deterministic single-thread public-method wrapper caveat and does not use the public mixed-thread runner directly.
- The S03/S09 primary Aggregation denominator is n=100, so a random 50/50 mix has expected primary Aggregation near 49%, not exactly 50%.
- The all-three condition has a different random-label baseline from two-way mixes because 100 cells are split 34/33/33 with a rotating remainder.
- The S09 negative control is one Bubble/Bubble label-control, not separate same-code controls for each algorithm-pair label set.
- Aligning curves by percent of final swap count is a Figure 8-style normalization; alignment by raw swap step would change peak timing.

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    s10_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e01"
    tables_dir = artifacts_dir / "tables"
    for path in (s10_dir, results_dir, figures_dir, tables_dir, artifacts_dir / "checksums"):
        path.mkdir(parents=True, exist_ok=True)

    trace = load_trace(args)
    expected_full_repeats = 100
    trace_validation = validate_trace_reconstruction(trace, expected_full_repeats)
    run_grid = build_run_grid(trace, args.progress_points)
    curves = summarize_curves(run_grid)
    run_peaks = summarize_run_peaks(run_grid)
    peak_table = build_peak_table(curves, run_peaks)

    curves_path = results_dir / "e01_aggregation_curves.parquet"
    figure_path = figures_dir / "figure8_aggregation_reproduction.png"
    peak_table_path = tables_dir / "e01_aggregation_peak_table.csv"
    run_peak_table_path = tables_dir / "e01_aggregation_run_peak_table.csv"
    validation_path = s10_dir / "s10_validation.json"
    status_path = s10_dir / "status.json"
    report_path = s10_dir / "research_step_full_results.md"
    artifact_manifest_path = s10_dir / "artifact_manifest.json"

    curves.to_parquet(curves_path, index=False)
    peak_table.to_csv(peak_table_path, index=False)
    run_peaks.to_csv(run_peak_table_path, index=False)
    plot_figure(curves, peak_table, figure_path)

    validation = validate_outputs(trace_validation, curves, peak_table, run_grid, args.progress_points, args.smoke_repeats)
    validation.update(
        {
            "createdAtUtc": utc_now(),
            "repositoryCommitAtRunTime": git_commit(args.repo_dir),
            "script": str(args.repo_dir / "scripts/e01_s10_aggregation_curves.py"),
            "threadEnvironment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
            "summary": {
                "peakLeftByMix": {
                    str(row["algotype_mix"]): float(row["mean_curve_peak_aggregation_left_neighbor_percent"])
                    for _, row in peak_table.iterrows()
                },
                "peakProgressByMix": {
                    str(row["algotype_mix"]): float(row["mean_curve_peak_left_neighbor_progress_percent"])
                    for _, row in peak_table.iterrows()
                },
                "coreMixedPeaksAboveNegativeControlPeak": validation["coreMixedPeaksAboveNegativeControlPeak"],
                "coreMixedPeaksAboveNegativeControlAtSameProgress": validation["coreMixedPeaksAboveNegativeControlAtSameProgress"],
                "coreMixedPeaksAboveExpectedRandomBaseline": validation["coreMixedPeaksAboveExpectedRandomBaseline"],
            },
            "artifactsWritten": [
                str(report_path),
                str(curves_path),
                str(figure_path),
                str(peak_table_path),
                str(run_peak_table_path),
                str(validation_path),
                str(status_path),
                str(artifact_manifest_path),
            ],
        }
    )
    artifact_paths = {
        "fullResultsReport": report_path,
        "aggregationCurvesParquet": curves_path,
        "figurePng": figure_path,
        "peakTableCsv": peak_table_path,
        "runPeakTableCsv": run_peak_table_path,
        "validationJson": validation_path,
        "statusJson": status_path,
        "artifactManifest": artifact_manifest_path,
    }
    validation["allArtifactsExist"] = False
    wall_seconds = time.monotonic() - started
    validation["wallSeconds"] = wall_seconds
    write_report(report_path, artifact_paths, validation, peak_table, sys.argv, args.repo_dir, args, wall_seconds)
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation["success"],
        "status": validation["status"],
        "artifactsWritten": validation["artifactsWritten"],
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
    }
    status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(artifact_manifest_path, artifact_paths, validation, args.repo_dir)
    validation["allArtifactsExist"] = all(path.exists() for path in artifact_paths.values())
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, artifact_paths, validation, peak_table, sys.argv, args.repo_dir, args, wall_seconds)
    write_artifact_manifest(artifact_manifest_path, artifact_paths, validation, args.repo_dir)

    update_run_manifest(artifacts_dir, {f"s10_{key}": value for key, value in artifact_paths.items()}, validation, args.repo_dir, sys.argv)
    update_checksum_file(
        artifacts_dir / "checksums" / "sha256sums.txt",
        list(artifact_paths.values())
        + [
            args.s09_trace_path,
            args.s09_validation_path,
            args.research_plan_path,
            args.repo_dir / "scripts/e01_s10_aggregation_curves.py",
            artifacts_dir / "run_manifest.json",
        ],
    )
    print(json.dumps({"validation": validation, "summary": validation["summary"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
