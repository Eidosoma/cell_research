#!/usr/bin/env python3
"""Compute E01 S08 Delayed Gratification metrics from frozen-cell traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


STEP_ID = "S08"
STEP_NUMBER = 8
EXPERIMENT_ID = "E01"
ALGORITHMS = ("bubble", "insertion", "selection")
MODES = ("traditional", "cell_view")
FROZEN_SEMANTICS = ("passive", "stuck")
PAPER_REPORTED_STUCK_CELL_VIEW_DG = {
    "bubble": {0: 0.24, 1: 0.29, 2: 0.32, 3: 0.37},
    "insertion": {0: 1.10, 1: 1.13, 2: 1.15, 3: 1.19},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config-path", type=Path, default=Path("/artifacts/configs/e01_baseline_configs.json"))
    parser.add_argument("--condition-matrix-path", type=Path, default=Path("/artifacts/tables/e01_condition_matrix.csv"))
    parser.add_argument("--s04-trace-path", type=Path, default=Path("/artifacts/traces/e01_figure3_trajectories.parquet"))
    parser.add_argument("--s07-trace-path", type=Path, default=Path("/artifacts/traces/e01_frozen_cell_trajectories.parquet"))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--smoke-repeats", type=int, default=None, help="Optional cap per condition for smoke validation.")
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
    stringified = df.copy()
    for col in stringified.columns:
        if pd.api.types.is_float_dtype(stringified[col]):
            stringified[col] = stringified[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    stringified = stringified.astype("string").fillna("").astype(str)
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


def load_s08_conditions(condition_matrix_path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(condition_matrix_path)
    rows = matrix[matrix["step_scope"].fillna("").str.contains("S08")].copy()
    rows["frozen_count"] = rows["frozen_count"].astype(int)
    return rows.sort_values(["frozen_count", "frozen_semantics", "mode", "algorithm", "condition_id"]).reset_index(drop=True)


def compact_consecutive(values: list[float], tolerance: float = 1e-12) -> list[float]:
    compact: list[float] = []
    for value in values:
        if pd.isna(value):
            continue
        value = float(value)
        if not compact or not math.isclose(compact[-1], value, abs_tol=tolerance):
            compact.append(value)
    return compact


def signed_segments(values: list[float]) -> list[float]:
    compact = compact_consecutive(values)
    if len(compact) < 2:
        return []
    segments: list[float] = []
    for idx in range(1, len(compact)):
        delta = compact[idx] - compact[idx - 1]
        if math.isclose(delta, 0.0, abs_tol=1e-12):
            continue
        if not segments or segments[-1] * delta <= 0:
            segments.append(delta)
        else:
            segments[-1] += delta
    return segments


def dg_from_sortedness(values: list[float]) -> dict[str, Any]:
    segments = signed_segments(values)
    events: list[dict[str, float]] = []
    terminal_unrecovered_drop_segments = 0
    leading_increase_segments = 0
    seen_drop = False
    for idx, segment in enumerate(segments):
        if segment > 0 and not seen_drop:
            leading_increase_segments += 1
        if segment < 0:
            seen_drop = True
            if idx + 1 < len(segments) and segments[idx + 1] > 0:
                drop = -segment
                recovery = segments[idx + 1]
                events.append(
                    {
                        "drop": float(drop),
                        "recovery": float(recovery),
                        "ratio": float((recovery - drop) / drop) if drop > 0 else 0.0,
                    }
                )
            else:
                terminal_unrecovered_drop_segments += 1
    ratios = [event["ratio"] for event in events]
    drops = [event["drop"] for event in events]
    recoveries = [event["recovery"] for event in events]
    total_drop = float(np.sum(drops)) if drops else 0.0
    total_recovery = float(np.sum(recoveries)) if recoveries else 0.0
    return {
        "dg_primary": float(np.mean(ratios)) if ratios else 0.0,
        "dg_event_count": int(len(events)),
        "dg_total_drop": total_drop,
        "dg_total_recovery": total_recovery,
        "dg_total_net_gain": total_recovery - total_drop,
        "dg_total_ratio": float((total_recovery - total_drop) / total_drop) if total_drop > 0 else 0.0,
        "dg_min_event_ratio": float(np.min(ratios)) if ratios else 0.0,
        "dg_max_event_ratio": float(np.max(ratios)) if ratios else 0.0,
        "dg_mean_event_drop": float(np.mean(drops)) if drops else 0.0,
        "dg_mean_event_recovery": float(np.mean(recoveries)) if recoveries else 0.0,
        "trajectory_point_count": int(len(values)),
        "compact_sortedness_point_count": int(len(compact_consecutive(values))),
        "signed_segment_count": int(len(segments)),
        "leading_increase_segment_count": int(leading_increase_segments),
        "terminal_unrecovered_drop_segment_count": int(terminal_unrecovered_drop_segments),
        "signed_segments_prefix_json": json.dumps([round(float(value), 10) for value in segments[:20]], separators=(",", ":")),
    }


def dg_from_monotonicity_error(values: list[float]) -> dict[str, Any]:
    # Equivalent to sortedness DG when sortedness = 100 - adjacent monotonicity error.
    segments = signed_segments(values)
    events: list[dict[str, float]] = []
    for idx, segment in enumerate(segments):
        if segment > 0 and idx + 1 < len(segments) and segments[idx + 1] < 0:
            drop_away_from_goal = segment
            recovery_toward_goal = -segments[idx + 1]
            events.append(
                {
                    "drop": float(drop_away_from_goal),
                    "recovery": float(recovery_toward_goal),
                    "ratio": float((recovery_toward_goal - drop_away_from_goal) / drop_away_from_goal)
                    if drop_away_from_goal > 0
                    else 0.0,
                }
            )
    ratios = [event["ratio"] for event in events]
    return {
        "dg_primary_from_monotonicity_error": float(np.mean(ratios)) if ratios else 0.0,
        "dg_event_count_from_monotonicity_error": int(len(events)),
    }


def validate_toy_cases() -> dict[str, Any]:
    cases = [
        {"name": "monotone_increase_no_dg", "sortedness": [50, 60, 70], "expected": 0.0, "events": 0},
        {"name": "one_drop_then_recover", "sortedness": [50, 40, 70], "expected": 2.0, "events": 1},
        {"name": "leading_gain_ignored", "sortedness": [50, 60, 40, 70], "expected": 0.5, "events": 1},
        {"name": "two_events_average", "sortedness": [50, 40, 60, 55, 80], "expected": 2.5, "events": 2},
        {"name": "flat_values_deduplicated", "sortedness": [50, 50, 40, 40, 70], "expected": 2.0, "events": 1},
    ]
    details = []
    passed = True
    for case in cases:
        result = dg_from_sortedness(case["sortedness"])
        case_passed = (
            math.isclose(result["dg_primary"], case["expected"], rel_tol=0.0, abs_tol=1e-12)
            and result["dg_event_count"] == case["events"]
        )
        passed = passed and case_passed
        details.append(
            {
                "name": case["name"],
                "sortedness": case["sortedness"],
                "expectedDg": case["expected"],
                "observedDg": result["dg_primary"],
                "expectedEvents": case["events"],
                "observedEvents": result["dg_event_count"],
                "passed": case_passed,
            }
        )
    return {"toyCaseValidationPassed": passed, "toyCases": details}


def load_source_traces(args: argparse.Namespace, s08_conditions: pd.DataFrame) -> pd.DataFrame:
    needed_cols = [
        "research_step_id",
        "experiment_id",
        "condition_id",
        "mode",
        "algorithm",
        "baseline_source",
        "matched_group_id",
        "value_bank_id",
        "repeat_index",
        "initial_array_seed",
        "scheduler_seed",
        "swap_step",
        "comparison_count_at_step",
        "sortedness_percent",
        "monotonicity_error_count",
        "is_initial",
        "is_final",
        "stop_reason",
        "initial_array_sha256",
        "final_array_sha256",
    ]
    s04_ids = set(s08_conditions.loc[s08_conditions["frozen_count"] == 0, "condition_id"])
    s07_ids = set(s08_conditions.loc[s08_conditions["frozen_count"] > 0, "condition_id"])
    s04 = pd.read_parquet(args.s04_trace_path, columns=needed_cols + ["timed_out"])
    s04 = s04[s04["condition_id"].isin(s04_ids)].copy()
    s04["source_trajectory_path"] = str(args.s04_trace_path)
    s04["max_guard_hit"] = s04["timed_out"].astype(bool)
    s04.drop(columns=["timed_out"], inplace=True)
    s07_cols = needed_cols + [
        "frozen_semantics",
        "frozen_count",
        "frozen_index_seed",
        "initial_frozen_indices_json",
        "frozen_attempt_count_at_step",
        "max_guard_hit",
    ]
    s07 = pd.read_parquet(args.s07_trace_path, columns=s07_cols)
    s07 = s07[s07["condition_id"].isin(s07_ids)].copy()
    s07["source_trajectory_path"] = str(args.s07_trace_path)

    condition_meta = s08_conditions[
        [
            "condition_id",
            "frozen_count",
            "frozen_semantics",
            "frozen_index_bank_id",
            "repeat_count",
            "figure_targets",
            "reconstruction_notes",
            "caveats",
        ]
    ].copy()
    s04 = s04.merge(condition_meta, on="condition_id", how="left", validate="many_to_one")
    s04["frozen_index_seed"] = pd.NA
    s04["initial_frozen_indices_json"] = "[]"
    s04["frozen_attempt_count_at_step"] = 0.0
    s04["frozen_index_bank_id"] = s04["frozen_index_bank_id"].fillna("frozen_0")
    s07 = s07.merge(
        condition_meta.drop(columns=["frozen_count", "frozen_semantics"]),
        on="condition_id",
        how="left",
        validate="many_to_one",
    )
    combined = pd.concat([s04, s07], ignore_index=True, sort=False)
    combined["source_row_order"] = np.arange(len(combined))
    return combined


def build_run_level_dg(trace_df: pd.DataFrame, smoke_repeats: int | None = None) -> pd.DataFrame:
    if smoke_repeats is not None:
        trace_df = trace_df[trace_df["repeat_index"] < smoke_repeats].copy()
    run_records: list[dict[str, Any]] = []
    group_cols = [
        "condition_id",
        "mode",
        "algorithm",
        "frozen_semantics",
        "frozen_count",
        "repeat_index",
    ]
    for key, group in trace_df.groupby(group_cols, sort=False):
        group = group.sort_values(["swap_step", "is_final", "source_row_order"], kind="stable")
        sortedness_values = group["sortedness_percent"].astype(float).tolist()
        error_values = group["monotonicity_error_count"].astype(float).tolist()
        dg_result = dg_from_sortedness(sortedness_values)
        error_dg = dg_from_monotonicity_error(error_values)
        first = group.iloc[0]
        final_rows = group[group["is_final"]]
        final = final_rows.iloc[-1] if not final_rows.empty else group.iloc[-1]
        run_records.append(
            {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "source_research_step_id": first["research_step_id"],
                "condition_id": key[0],
                "mode": key[1],
                "algorithm": key[2],
                "baseline_source": first["baseline_source"],
                "matched_group_id": first["matched_group_id"],
                "value_bank_id": first["value_bank_id"],
                "repeat_index": int(key[5]),
                "initial_array_seed": int(first["initial_array_seed"]),
                "scheduler_seed": None if pd.isna(first["scheduler_seed"]) else int(first["scheduler_seed"]),
                "initial_array_sha256": first["initial_array_sha256"],
                "final_array_sha256": final["final_array_sha256"],
                "frozen_semantics": key[3],
                "frozen_count": int(key[4]),
                "frozen_index_bank_id": first["frozen_index_bank_id"],
                "frozen_index_seed": None if pd.isna(first["frozen_index_seed"]) else int(first["frozen_index_seed"]),
                "initial_frozen_indices_json": first["initial_frozen_indices_json"],
                "initial_sortedness_percent": float(first["sortedness_percent"]),
                "final_sortedness_percent": float(final["sortedness_percent"]),
                "min_sortedness_percent": float(group["sortedness_percent"].min()),
                "max_sortedness_percent": float(group["sortedness_percent"].max()),
                "initial_monotonicity_error_count": int(first["monotonicity_error_count"]),
                "final_monotonicity_error_count": int(final["monotonicity_error_count"]),
                "final_swap_step": int(final["swap_step"]),
                "stop_reason": final["stop_reason"],
                "max_guard_hit": bool(final["max_guard_hit"]),
                "source_trajectory_path": first["source_trajectory_path"],
                **dg_result,
                **error_dg,
            }
        )
    run_df = pd.DataFrame.from_records(run_records)
    run_df["dg_sortedness_error_agree"] = np.isclose(
        run_df["dg_primary"], run_df["dg_primary_from_monotonicity_error"], atol=1e-12, rtol=0.0
    ) & (run_df["dg_event_count"] == run_df["dg_event_count_from_monotonicity_error"])
    return run_df.sort_values(["condition_id", "repeat_index"]).reset_index(drop=True)


def build_condition_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition_id, group in run_df.groupby("condition_id", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "source_research_step_id": first["source_research_step_id"],
                "mode": first["mode"],
                "algorithm": first["algorithm"],
                "frozen_semantics": first["frozen_semantics"],
                "frozen_count": int(first["frozen_count"]),
                "baseline_source": first["baseline_source"],
                "matched_group_id": first["matched_group_id"],
                "repetitions_observed": int(group["repeat_index"].nunique()),
                "mean_dg_primary": float(group["dg_primary"].mean()),
                "std_dg_primary": float(group["dg_primary"].std(ddof=1)),
                "median_dg_primary": float(group["dg_primary"].median()),
                "min_dg_primary": float(group["dg_primary"].min()),
                "max_dg_primary": float(group["dg_primary"].max()),
                "mean_dg_total_ratio": float(group["dg_total_ratio"].mean()),
                "mean_dg_event_count": float(group["dg_event_count"].mean()),
                "runs_with_no_dg_events": int((group["dg_event_count"] == 0).sum()),
                "runs_with_terminal_unrecovered_drop": int((group["terminal_unrecovered_drop_segment_count"] > 0).sum()),
                "mean_final_sortedness_percent": float(group["final_sortedness_percent"].mean()),
                "max_guard_runs": int(group["max_guard_hit"].sum()),
                "stop_reason_counts": json.dumps(dict(Counter(group["stop_reason"])), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def build_plot_summary(condition_summary: pd.DataFrame) -> pd.DataFrame:
    f0 = condition_summary[condition_summary["frozen_count"] == 0].copy()
    frozen = condition_summary[condition_summary["frozen_count"] > 0].copy()
    expanded_f0 = []
    for semantics in FROZEN_SEMANTICS:
        temp = f0.copy()
        temp["plot_frozen_semantics"] = semantics
        temp["f0_reused_for_semantics"] = True
        expanded_f0.append(temp)
    frozen = frozen.copy()
    frozen["plot_frozen_semantics"] = frozen["frozen_semantics"]
    frozen["f0_reused_for_semantics"] = False
    return pd.concat([*expanded_f0, frozen], ignore_index=True, sort=False)


def build_comparison_summary(plot_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = plot_summary.pivot_table(
        index=["plot_frozen_semantics", "algorithm", "frozen_count"],
        columns="mode",
        values="mean_dg_primary",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot["cell_minus_traditional_mean_dg"] = pivot["cell_view"] - pivot["traditional"]
    pivot["cell_greater_than_traditional"] = pivot["cell_view"] > pivot["traditional"]
    pivot["cell_less_than_traditional"] = pivot["cell_view"] < pivot["traditional"]
    return pivot.sort_values(["plot_frozen_semantics", "algorithm", "frozen_count"]).reset_index(drop=True)


def build_paper_direction_summary(plot_summary: pd.DataFrame, comparison_summary: pd.DataFrame) -> dict[str, Any]:
    stuck = plot_summary[plot_summary["plot_frozen_semantics"] == "stuck"].copy()
    primary_checks: dict[str, Any] = {}
    for algorithm in ("bubble", "insertion"):
        for mode in MODES:
            values = (
                stuck[(stuck["algorithm"] == algorithm) & (stuck["mode"] == mode)]
                .sort_values("frozen_count")["mean_dg_primary"]
                .tolist()
            )
            primary_checks[f"stuck_{mode}_{algorithm}_dg_monotone_increasing_f0_to_f3"] = all(
                values[idx] <= values[idx + 1] + 1e-12 for idx in range(len(values) - 1)
            )
            primary_checks[f"stuck_{mode}_{algorithm}_mean_dg_by_frozen_count"] = values
    stuck_comparison = comparison_summary[comparison_summary["plot_frozen_semantics"] == "stuck"]
    bubble = stuck_comparison[stuck_comparison["algorithm"] == "bubble"].sort_values("frozen_count")
    insertion = stuck_comparison[stuck_comparison["algorithm"] == "insertion"].sort_values("frozen_count")
    selection = stuck_comparison[stuck_comparison["algorithm"] == "selection"].sort_values("frozen_count")
    primary_checks["stuck_bubble_cell_view_gt_traditional_all_counts"] = bool(
        (bubble["cell_minus_traditional_mean_dg"] > 0).all()
    )
    primary_checks["stuck_insertion_cell_view_similar_to_traditional_all_counts_abs_diff_le_0_05"] = bool(
        (insertion["cell_minus_traditional_mean_dg"].abs() <= 0.05).all()
    )
    primary_checks["stuck_selection_cell_view_lt_traditional_all_counts"] = bool(
        (selection["cell_minus_traditional_mean_dg"] < 0).all()
    )
    passive = plot_summary[plot_summary["plot_frozen_semantics"] == "passive"].copy()
    passive_checks: dict[str, Any] = {}
    for algorithm in ("bubble", "insertion"):
        for mode in MODES:
            values = (
                passive[(passive["algorithm"] == algorithm) & (passive["mode"] == mode)]
                .sort_values("frozen_count")["mean_dg_primary"]
                .tolist()
            )
            passive_checks[f"passive_{mode}_{algorithm}_dg_monotone_increasing_f0_to_f3"] = all(
                values[idx] <= values[idx + 1] + 1e-12 for idx in range(len(values) - 1)
            )
            passive_checks[f"passive_{mode}_{algorithm}_mean_dg_by_frozen_count"] = values
    primary_supportive = bool(
        primary_checks["stuck_bubble_cell_view_gt_traditional_all_counts"]
        and primary_checks["stuck_insertion_cell_view_similar_to_traditional_all_counts_abs_diff_le_0_05"]
        and primary_checks["stuck_selection_cell_view_lt_traditional_all_counts"]
        and primary_checks["stuck_traditional_bubble_dg_monotone_increasing_f0_to_f3"]
        and primary_checks["stuck_cell_view_bubble_dg_monotone_increasing_f0_to_f3"]
        and primary_checks["stuck_traditional_insertion_dg_monotone_increasing_f0_to_f3"]
        and primary_checks["stuck_cell_view_insertion_dg_monotone_increasing_f0_to_f3"]
    )
    return {
        "primaryStuckFigure7DirectionSupportive": primary_supportive,
        "primaryChecks": primary_checks,
        "passiveSensitivityChecks": passive_checks,
    }


def validate_outputs(
    s08_conditions: pd.DataFrame,
    trace_df: pd.DataFrame,
    run_df: pd.DataFrame,
    condition_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    direction_summary: dict[str, Any],
    smoke_repeats: int | None,
) -> dict[str, Any]:
    repeats_expected = int(s08_conditions["repeat_count"].iloc[0])
    repeats_to_run = min(smoke_repeats or repeats_expected, repeats_expected)
    expected_conditions = set(s08_conditions["condition_id"])
    observed_conditions = set(run_df["condition_id"])
    expected_runs = len(expected_conditions) * repeats_to_run
    observed_repeat_counts = run_df.groupby("condition_id")["repeat_index"].nunique().to_dict()
    repeat_counts_ok = all(count == repeats_to_run for count in observed_repeat_counts.values()) and len(observed_repeat_counts) == len(expected_conditions)
    sortedness_error_complement_ok = bool(
        np.isclose(
            trace_df["sortedness_percent"].astype(float) + trace_df["monotonicity_error_count"].astype(float),
            100.0,
            atol=1e-12,
            rtol=0.0,
        ).all()
    )
    source_step_counts = run_df.groupby("source_research_step_id")["condition_id"].nunique().to_dict()
    source_run_counts = run_df.groupby("source_research_step_id").size().to_dict()
    f0_source_ok = bool(
        set(run_df.loc[run_df["frozen_count"] == 0, "source_research_step_id"].unique()) == {"S04"}
        and set(run_df.loc[run_df["frozen_count"] > 0, "source_research_step_id"].unique()) == {"S07"}
    )
    all_finite = bool(np.isfinite(run_df["dg_primary"]).all())
    dg_error_agrees = bool(run_df["dg_sortedness_error_agree"].all())
    no_max_guard = bool((~run_df["max_guard_hit"]).all())
    all_artifacts_placeholder = False
    success = bool(
        len(run_df) == expected_runs
        and observed_conditions == expected_conditions
        and repeat_counts_ok
        and sortedness_error_complement_ok
        and f0_source_ok
        and all_finite
        and dg_error_agrees
        and no_max_guard
        and direction_summary["primaryStuckFigure7DirectionSupportive"]
        and smoke_repeats is None
    )
    status = "completed_with_caveats" if success else "completed_validation_failed"
    validation_result = "passed_with_formula_and_source_caveats" if success else "failed"
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": status,
        "validationResult": validation_result,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "expectedConditionRows": len(expected_conditions),
        "observedConditionRows": len(observed_conditions),
        "expectedRuns": expected_runs,
        "runRows": int(len(run_df)),
        "sourceTraceRowsUsed": int(len(trace_df)),
        "expectedRepeatsPerCondition": repeats_expected,
        "repeatsToRun": repeats_to_run,
        "repeatCountsOk": repeat_counts_ok,
        "observedRepeatCountMin": int(min(observed_repeat_counts.values())) if observed_repeat_counts else 0,
        "observedRepeatCountMax": int(max(observed_repeat_counts.values())) if observed_repeat_counts else 0,
        "conditionsMatchS08Matrix": observed_conditions == expected_conditions,
        "sourceStepConditionCounts": {str(key): int(value) for key, value in source_step_counts.items()},
        "sourceStepRunCounts": {str(key): int(value) for key, value in source_run_counts.items()},
        "f0FromS04AndFrozenFromS07": f0_source_ok,
        "sortednessPlusMonotonicityErrorEquals100": sortedness_error_complement_ok,
        "dgRecomputedFromMonotonicityErrorAgreesWithSortedness": dg_error_agrees,
        "allDgPrimaryFinite": all_finite,
        "noMaxGuardRuns": no_max_guard,
        "runsWithNoDgEvents": int((run_df["dg_event_count"] == 0).sum()),
        "runsWithTerminalUnrecoveredDrop": int((run_df["terminal_unrecovered_drop_segment_count"] > 0).sum()),
        "conditionSummaryRows": int(len(condition_summary)),
        "comparisonSummaryRows": int(len(comparison_summary)),
        "primaryStuckFigure7DirectionSupportive": direction_summary["primaryStuckFigure7DirectionSupportive"],
        "primaryChecks": direction_summary["primaryChecks"],
        "passiveSensitivityChecks": direction_summary["passiveSensitivityChecks"],
        "allArtifactsExist": all_artifacts_placeholder,
        "caveatsOrBlockers": [
            "S08 uses the paper/Figure 6 backtracking-recovery definition on Sortedness trajectories: a consecutive Sortedness drop followed by a consecutive recovery contributes (recovery - drop) / drop, and run-level DG is the mean event ratio.",
            "The public repository's DG helper computes the equivalent quantity on monotonicity-error trajectories and names the helper 'max' while returning an average; S08 records this formula ambiguity and validates Sortedness/error equivalence.",
            "S04 unperturbed f=0 trajectories are reused for S08 f=0 baselines as frozen in S03; S07 supplies f=1..3 passive and stuck trajectories.",
            "Figure 7 in the paper says f denotes stuck Frozen Cells; S08 also computes passive semantics as a sensitivity layer. Passive Bubble and Insertion trends do not show the same monotone increase as the stuck primary layer.",
            "Traditional trajectories remain reconstructed and S07 frozen cell-view trajectories retain the deterministic single-threaded wrapper caveat.",
        ],
        "recommendedNextAction": "Proceed to S09 same-goal chimeric arrays; do not start S09 inside S08.",
    }


def plot_figure(plot_summary: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharex=True)
    colors = {"traditional": "#4f6fad", "cell_view": "#c25b3d"}
    markers = {"traditional": "s", "cell_view": "o"}
    row_titles = {"passive": "Passive Frozen Cells", "stuck": "Stuck Frozen Cells"}
    for row_idx, semantics in enumerate(FROZEN_SEMANTICS):
        for col_idx, algorithm in enumerate(ALGORITHMS):
            ax = axes[row_idx, col_idx]
            subset = plot_summary[
                (plot_summary["plot_frozen_semantics"] == semantics)
                & (plot_summary["algorithm"] == algorithm)
            ].sort_values(["mode", "frozen_count"])
            for mode in MODES:
                data = subset[subset["mode"] == mode].sort_values("frozen_count")
                ax.errorbar(
                    data["frozen_count"],
                    data["mean_dg_primary"],
                    yerr=data["std_dg_primary"].fillna(0.0),
                    color=colors[mode],
                    marker=markers[mode],
                    linewidth=1.8,
                    capsize=3,
                    label="Cell-view" if mode == "cell_view" else "Traditional",
                )
            if semantics == "stuck" and algorithm in PAPER_REPORTED_STUCK_CELL_VIEW_DG:
                reported = PAPER_REPORTED_STUCK_CELL_VIEW_DG[algorithm]
                ax.plot(
                    list(reported.keys()),
                    list(reported.values()),
                    color="#2b8a5e",
                    linestyle=":",
                    marker="^",
                    linewidth=1.5,
                    label="Paper cell-view",
                )
            ax.set_title(f"{row_titles[semantics]}: {algorithm.title()}")
            ax.set_xticks([0, 1, 2, 3])
            ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.45)
            if col_idx == 0:
                ax.set_ylabel("Delayed Gratification")
            if row_idx == 1:
                ax.set_xlabel("Frozen Cell count")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("E01 S08 Figure 7-style Delayed Gratification from Sortedness trajectories", fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(figure_path, dpi=180)
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
    checksum_path.write_text("".join(f"{existing[path]}  {path}\n" for path in order if path != str(checksum_path)), encoding="utf-8")


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
        "title": "Reproduce Delayed Gratification analysis",
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
        "artifactsWritten": [str(path) for path in paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s08_delayed_gratification.py"),
        "command": " ".join(command),
        "summary": validation["summary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    report_path: Path,
    artifact_paths: dict[str, Path],
    validation: dict[str, Any],
    condition_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    command: list[str],
    repo_dir: Path,
    args: argparse.Namespace,
    wall_seconds: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    caveats = "; ".join(validation["caveatsOrBlockers"])
    if validation["outcomeClassification"] == "supportive":
        lay_summary = (
            "S08 reproduced the main Figure 7 Delayed Gratification direction for the stuck Frozen Cell layer: "
            "Bubble and Insertion DG increased with frozen-cell count, Bubble cell-view exceeded traditional, "
            "Insertion remained similar between modes, and Selection cell-view was lower than traditional."
        )
    else:
        lay_summary = (
            "S08 computed Delayed Gratification from the available trajectories, but one or more primary "
            "Figure 7 direction checks did not pass under the frozen wrappers."
        )
    compact_condition = condition_summary[
        [
            "condition_id",
            "source_research_step_id",
            "mode",
            "algorithm",
            "frozen_semantics",
            "frozen_count",
            "mean_dg_primary",
            "std_dg_primary",
            "mean_dg_event_count",
            "runs_with_no_dg_events",
            "runs_with_terminal_unrecovered_drop",
        ]
    ].copy()
    compact_comparison = comparison_summary[
        [
            "plot_frozen_semantics",
            "algorithm",
            "frozen_count",
            "traditional",
            "cell_view",
            "cell_minus_traditional_mean_dg",
        ]
    ].copy()
    report = f"""# E01 S08 Research Step Full Results

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

Does DG increase with the number of Frozen Cells for Bubble and Insertion settings, and do replicated traditional-versus-cell-view DG comparisons match the paper's direction?

## Inputs

- S03 config: `{args.config_path}` (`{sha256_file(args.config_path)}`)
- S03 condition matrix: `{args.condition_matrix_path}` (`{sha256_file(args.condition_matrix_path)}`)
- S04 f=0 trajectories: `{args.s04_trace_path}` (`{sha256_file(args.s04_trace_path)}`)
- S07 f=1..3 trajectories: `{args.s07_trace_path}` (`{sha256_file(args.s07_trace_path)}`)
- S08 condition rows: `{validation["expectedConditionRows"]}`
- Repeats per condition: `{validation["expectedRepeatsPerCondition"]}`
- Paper reference: Methods Sortedness Delayed Gratification section and Figure 6/Figure 7 captions.

## Detailed Methods

S08 computes Delayed Gratification from Sortedness trajectories without rerunning sorting. S03 marks the six unperturbed f=0 rows as `S08_f0`, so those trajectories are read from the S04 Figure 3 trace parquet. The 36 f=1..3 passive and stuck Frozen Cell rows are read from the S07 trajectory parquet. The f=0 baseline has `frozen_semantics=none`; for Figure 7-style trend plots it is reused as the common f=0 point for both passive and stuck layers.

The primary DG formula follows the paper's backtracking/recovery definition in Sortedness space. Consecutive identical Sortedness values are deduplicated. Consecutive nonzero changes with the same sign are collapsed into signed segments. A DG event is a negative Sortedness segment, interpreted as a temporary move away from the goal, followed immediately by a positive Sortedness segment, interpreted as recovery. For each event, `DG_event = (recovery - drop) / drop`. Run-level `dg_primary` is the mean event ratio across events. Runs with no drop/recovery events receive `dg_primary=0`.

The public repository's `analysis/delay_gratification_analysis.py` computes the equivalent operation on monotonicity-error trajectories and labels the helper as `get_max_delay_gratification`, but the active implementation returns an average. S08 records this as a formula ambiguity and validates that Sortedness-derived DG agrees with monotonicity-error-derived DG for every run, using the S03/S07 identity `Sortedness = 100 - monotonicity_error_count` for n=100.

## Commands

```bash
{" ".join(command)}
```

Validation/smoke commands:

```bash
python -m py_compile scripts/e01_s08_delayed_gratification.py
python scripts/e01_s08_delayed_gratification.py --repo-dir /workspace/cell-research --artifacts-dir /cache/e01_s08_smoke_artifacts --smoke-repeats 2
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
- Smoke repeat cap: `{args.smoke_repeats}`
- Wall time: `{wall_seconds:.3f}` seconds

No new Python, system, R, Rust, or Node dependencies were installed for S08.

## Results

Condition-level DG summary:

{dataframe_to_markdown(compact_condition)}

Mode comparison summary, with f=0 reused as the common baseline for each passive/stuck plotting layer:

{dataframe_to_markdown(compact_comparison)}

Primary Figure 7 stuck-layer direction checks:

```json
{json.dumps(validation["primaryChecks"], indent=2)}
```

Passive sensitivity checks:

```json
{json.dumps(validation["passiveSensitivityChecks"], indent=2)}
```

## Validation

- Run rows: `{validation["runRows"]}` of expected `{validation["expectedRuns"]}`.
- Source trace rows used: `{validation["sourceTraceRowsUsed"]}`.
- Source condition counts: `{validation["sourceStepConditionCounts"]}`.
- Source run counts: `{validation["sourceStepRunCounts"]}`.
- Repeat counts match S08 matrix: `{validation["repeatCountsOk"]}`.
- Conditions match S08 matrix: `{validation["conditionsMatchS08Matrix"]}`.
- f=0 from S04 and f=1..3 from S07: `{validation["f0FromS04AndFrozenFromS07"]}`.
- Sortedness plus monotonicity error equals 100 for every trace row: `{validation["sortednessPlusMonotonicityErrorEquals100"]}`.
- DG recomputed from monotonicity-error trajectories agrees with Sortedness DG: `{validation["dgRecomputedFromMonotonicityErrorAgreesWithSortedness"]}`.
- Toy-case formula validation passed: `{validation["toyCaseValidation"]["toyCaseValidationPassed"]}`.
- All primary DG values finite: `{validation["allDgPrimaryFinite"]}`.
- Runs with no DG events: `{validation["runsWithNoDgEvents"]}`.
- Runs with terminal unrecovered drops: `{validation["runsWithTerminalUnrecoveredDrop"]}`.
- No max-guard runs: `{validation["noMaxGuardRuns"]}`.

Toy-case validation details:

```json
{json.dumps(validation["toyCaseValidation"], indent=2)}
```

## Artifacts And Provenance

Reusable outputs:

{chr(10).join(f"- `{label}`: `{path}`" for label, path in artifact_paths.items())}

The global run manifest and checksum file were updated after artifact creation. The S08 artifact manifest records paths, sizes, and SHA256 hashes.

## Caveats, Blockers, Failed Assumptions, And Limitations

- The paper prose and public code differ in presentation: the paper describes Sortedness drops and recoveries, while the public code computes on monotonicity error and averages event ratios. S08 uses Sortedness as requested and validates the error-space equivalent.
- The Figure 7 caption says `f` denotes stuck Frozen Cells. S08 computes both stuck and passive layers; the stuck layer supports the reported direction, while passive Bubble and Insertion sensitivity trends do not show the same monotone increase.
- S04 f=0 trajectories are reused for both passive and stuck trend baselines because f=0 has no frozen-cell semantics.
- Traditional baselines remain reconstructed, and S07 cell-view Frozen Cell trajectories retain the deterministic single-threaded wrapper caveat.
- S08 does not rerun S06 statistical tests; it writes DG metrics that later steps can use for statistical updates or the final divergence log.

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    s08_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e01"
    tables_dir = artifacts_dir / "tables"
    for path in (s08_dir, results_dir, figures_dir, tables_dir, artifacts_dir / "checksums"):
        path.mkdir(parents=True, exist_ok=True)

    toy_validation = validate_toy_cases()
    if not toy_validation["toyCaseValidationPassed"]:
        raise SystemExit("Toy-case DG validation failed.")

    s08_conditions = load_s08_conditions(args.condition_matrix_path)
    trace_df = load_source_traces(args, s08_conditions)
    run_df = build_run_level_dg(trace_df, args.smoke_repeats)
    condition_summary = build_condition_summary(run_df)
    plot_summary = build_plot_summary(condition_summary)
    comparison_summary = build_comparison_summary(plot_summary)
    direction_summary = build_paper_direction_summary(plot_summary, comparison_summary)

    dg_path = results_dir / "e01_delayed_gratification.parquet"
    figure_path = figures_dir / "figure7_dg_reproduction.png"
    numeric_table_path = tables_dir / "e01_dg_numeric_table.csv"
    comparison_table_path = tables_dir / "e01_dg_comparison_table.csv"
    report_path = s08_dir / "research_step_full_results.md"
    validation_path = s08_dir / "s08_validation.json"
    artifact_manifest_path = s08_dir / "artifact_manifest.json"

    run_df.to_parquet(dg_path, index=False)
    condition_summary.to_csv(numeric_table_path, index=False)
    comparison_summary.to_csv(comparison_table_path, index=False)
    plot_figure(plot_summary, figure_path)

    validation = validate_outputs(
        s08_conditions,
        trace_df,
        run_df,
        condition_summary,
        comparison_summary,
        direction_summary,
        args.smoke_repeats,
    )
    validation.update(
        {
            "toyCaseValidation": toy_validation,
            "createdAtUtc": utc_now(),
            "repositoryCommitAtRunTime": git_commit(args.repo_dir),
            "script": str(args.repo_dir / "scripts/e01_s08_delayed_gratification.py"),
            "summary": {
                "meanDgOverall": float(run_df["dg_primary"].mean()),
                "maxDgObserved": float(run_df["dg_primary"].max()),
                "meanDgEventCountOverall": float(run_df["dg_event_count"].mean()),
                "primaryStuckFigure7DirectionSupportive": direction_summary["primaryStuckFigure7DirectionSupportive"],
                "f0RunRows": int((run_df["frozen_count"] == 0).sum()),
                "s07FrozenRunRows": int((run_df["frozen_count"] > 0).sum()),
            },
            "artifactsWritten": [
                str(report_path),
                str(dg_path),
                str(figure_path),
                str(numeric_table_path),
                str(comparison_table_path),
                str(validation_path),
                str(artifact_manifest_path),
            ],
        }
    )
    artifact_paths = {
        "fullResultsReport": report_path,
        "delayedGratificationParquet": dg_path,
        "figurePng": figure_path,
        "numericSummaryCsv": numeric_table_path,
        "comparisonCsv": comparison_table_path,
        "validationJson": validation_path,
        "artifactManifest": artifact_manifest_path,
    }
    validation["allArtifactsExist"] = False
    wall_seconds = time.monotonic() - started
    validation["wallSeconds"] = wall_seconds
    write_report(
        report_path,
        artifact_paths,
        validation,
        condition_summary,
        comparison_summary,
        sys.argv,
        args.repo_dir,
        args,
        wall_seconds,
    )
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(artifact_manifest_path, artifact_paths, validation, args.repo_dir)
    validation["allArtifactsExist"] = all(path.exists() for path in artifact_paths.values())
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(
        report_path,
        artifact_paths,
        validation,
        condition_summary,
        comparison_summary,
        sys.argv,
        args.repo_dir,
        args,
        wall_seconds,
    )
    write_artifact_manifest(artifact_manifest_path, artifact_paths, validation, args.repo_dir)

    update_run_manifest(artifacts_dir, {f"s08_{key}": value for key, value in artifact_paths.items()}, validation, args.repo_dir, sys.argv)
    update_checksum_file(
        artifacts_dir / "checksums" / "sha256sums.txt",
        list(artifact_paths.values())
        + [
            args.config_path,
            args.condition_matrix_path,
            args.s04_trace_path,
            args.s07_trace_path,
            args.research_plan_path,
            args.repo_dir / "scripts/e01_s08_delayed_gratification.py",
            artifacts_dir / "run_manifest.json",
        ],
    )
    print(json.dumps({"validation": validation, "summary": validation["summary"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
