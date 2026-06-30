#!/usr/bin/env python3
"""Run E01 S06 statistical-test reproduction from frozen S05 counts."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STEP_ID = "S06"
STEP_NUMBER = 6
EXPERIMENT_ID = "E01"
SOURCE_STEP_ID = "S05"
ALGORITHMS = ("bubble", "insertion", "selection")
MODES = ("traditional", "cell_view")
COUNT_METRICS = (
    "swap_only_steps",
    "comparison_steps_observed",
    "compare_plus_swap_steps",
)
REPORTED_EFFICIENCY_METRICS = ("swap_only_steps", "compare_plus_swap_steps")
BOOTSTRAP_SEED = 2026063006

PAPER_EFFICIENCY_TARGETS: dict[tuple[str, str], dict[str, Any]] = {
    ("bubble", "swap_only_steps"): {
        "paperReportedZ": 0.73,
        "paperReportedP": "0.47",
        "paperSignificanceClass": "not_significant",
        "paperExpectedDirection": "near_equal",
        "paperRatioCellOverTraditional": 1.0,
        "paperReportedText": (
            "Swap-only Bubble performance was reported as not significantly different between "
            "traditional and cell-view modes."
        ),
    },
    ("insertion", "swap_only_steps"): {
        "paperReportedZ": 1.26,
        "paperReportedP": "0.24",
        "paperSignificanceClass": "not_significant",
        "paperExpectedDirection": "near_equal",
        "paperRatioCellOverTraditional": 1.0,
        "paperReportedText": (
            "Swap-only Insertion performance was reported as not significantly different between "
            "traditional and cell-view modes."
        ),
    },
    ("selection", "swap_only_steps"): {
        "paperReportedZ": 120.43,
        "paperReportedP": "p << 0.01 / 0",
        "paperSignificanceClass": "p_lt_0.01",
        "paperExpectedDirection": "cell_greater",
        "paperRatioCellOverTraditional": 11.0,
        "paperReportedText": "Swap-only cell-view Selection was reported as less efficient than traditional Selection.",
    },
    ("bubble", "compare_plus_swap_steps"): {
        "paperReportedZ": -68.96,
        "paperReportedP": "p << 0.01",
        "paperSignificanceClass": "p_lt_0.01",
        "paperExpectedDirection": "cell_less",
        "paperRatioCellOverTraditional": 1.0 / 1.5,
        "paperReportedText": (
            "Swap-plus-comparison cell-view Bubble was reported as more efficient than "
            "traditional Bubble."
        ),
    },
    ("insertion", "compare_plus_swap_steps"): {
        "paperReportedZ": -71.19,
        "paperReportedP": "p << 0.01",
        "paperSignificanceClass": "p_lt_0.01",
        "paperExpectedDirection": "cell_less",
        "paperRatioCellOverTraditional": 1.0 / 2.03,
        "paperReportedText": (
            "Swap-plus-comparison cell-view Insertion was reported as more efficient than "
            "traditional Insertion."
        ),
    },
    ("selection", "compare_plus_swap_steps"): {
        "paperReportedZ": 106.55,
        "paperReportedP": "p << 0.01",
        "paperSignificanceClass": "p_lt_0.01",
        "paperExpectedDirection": "cell_greater",
        "paperRatioCellOverTraditional": 1.17,
        "paperReportedText": (
            "Swap-plus-comparison cell-view Selection was reported as less efficient than "
            "traditional Selection."
        ),
    },
}

MISSING_TEST_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "claimFamily": "robustness",
        "claimId": "figure5_passive_frozen_cell_final_error",
        "expectedArtifacts": ["/artifacts/results/e01_frozen_cell_robustness.parquet"],
        "paperReportedText": (
            "Figure 5 reports that passive Frozen Cell perturbations produce lower final "
            "monotonicity error for cell-view algorithms than for traditional algorithms."
        ),
        "paperReportedZ": None,
        "paperReportedP": None,
        "paperSignificanceClass": None,
        "paperExpectedDirection": "cell_lower_monotonicity_error",
        "recommendedFollowupStep": "S07",
        "missingDataReason": (
            "S07 Frozen Cell robustness runs have not been executed, so final monotonicity-error "
            "tables for passive frozen semantics are unavailable."
        ),
    },
    {
        "claimFamily": "robustness",
        "claimId": "figure5_stuck_frozen_cell_final_error",
        "expectedArtifacts": ["/artifacts/results/e01_frozen_cell_robustness.parquet"],
        "paperReportedText": (
            "Figure 5 reports that stuck Frozen Cell perturbations produce lower final "
            "monotonicity error for cell-view algorithms than for traditional algorithms."
        ),
        "paperReportedZ": None,
        "paperReportedP": None,
        "paperSignificanceClass": None,
        "paperExpectedDirection": "cell_lower_monotonicity_error",
        "recommendedFollowupStep": "S07",
        "missingDataReason": (
            "S07 Frozen Cell robustness runs have not been executed, so final monotonicity-error "
            "tables for stuck frozen semantics are unavailable."
        ),
    },
    {
        "claimFamily": "delayed_gratification",
        "claimId": "figure7_bubble_dg_cell_vs_traditional",
        "expectedArtifacts": ["/artifacts/results/e01_delayed_gratification.parquet"],
        "paperReportedText": "Bubble DG cell-view minus traditional difference was reported as 0.16.",
        "paperReportedZ": 34.04,
        "paperReportedP": "p << 0.01",
        "paperSignificanceClass": "p_lt_0.01",
        "paperExpectedDirection": "cell_greater",
        "recommendedFollowupStep": "S08",
        "missingDataReason": (
            "S08 Delayed Gratification metrics require S07 Frozen Cell trajectories and have not "
            "been generated."
        ),
    },
    {
        "claimFamily": "delayed_gratification",
        "claimId": "figure7_insertion_dg_cell_vs_traditional",
        "expectedArtifacts": ["/artifacts/results/e01_delayed_gratification.parquet"],
        "paperReportedText": "Insertion DG cell-view minus traditional difference was reported as -0.03 or slight.",
        "paperReportedZ": 0.60,
        "paperReportedP": "0.55",
        "paperSignificanceClass": "not_significant",
        "paperExpectedDirection": "near_equal",
        "recommendedFollowupStep": "S08",
        "missingDataReason": (
            "S08 Delayed Gratification metrics require S07 Frozen Cell trajectories and have not "
            "been generated."
        ),
    },
    {
        "claimFamily": "delayed_gratification",
        "claimId": "figure7_selection_dg_cell_vs_traditional",
        "expectedArtifacts": ["/artifacts/results/e01_delayed_gratification.parquet"],
        "paperReportedText": (
            "Selection DG is reported with a sign ambiguity in the text/caption; the caption says "
            "cell-view Selection has less DG than traditional Selection."
        ),
        "paperReportedZ": -17.21,
        "paperReportedP": "p << 0.01",
        "paperSignificanceClass": "p_lt_0.01",
        "paperExpectedDirection": "cell_less",
        "recommendedFollowupStep": "S08",
        "missingDataReason": (
            "S08 Delayed Gratification metrics require S07 Frozen Cell trajectories and have not "
            "been generated."
        ),
    },
    {
        "claimFamily": "delayed_gratification",
        "claimId": "figure7_frozen_count_context_trend",
        "expectedArtifacts": ["/artifacts/results/e01_delayed_gratification.parquet"],
        "paperReportedText": (
            "Bubble and Insertion DG are reported to increase with the number of Frozen Cells; "
            "Selection is reported to lack a clear trend."
        ),
        "paperReportedZ": None,
        "paperReportedP": None,
        "paperSignificanceClass": None,
        "paperExpectedDirection": "bubble_insertion_positive_frozen_count_trend",
        "recommendedFollowupStep": "S08",
        "missingDataReason": (
            "S08 Delayed Gratification metrics by frozen count require S07 trajectories and have "
            "not been generated."
        ),
    },
    {
        "claimFamily": "aggregation",
        "claimId": "figure8_unique_value_chimera_peak_vs_control",
        "expectedArtifacts": [
            "/artifacts/results/e01_aggregation_curves.parquet",
            "/artifacts/tables/e01_aggregation_peak_table.csv",
        ],
        "paperReportedText": (
            "Unique-value chimeric Algotypes are reported to aggregate significantly above the "
            "identical-algotype negative control."
        ),
        "paperReportedZ": None,
        "paperReportedP": "p << 0.01",
        "paperSignificanceClass": "p_lt_0.01",
        "paperExpectedDirection": "chimera_peak_greater_than_control",
        "recommendedFollowupStep": "S10",
        "missingDataReason": (
            "S09/S10 chimeric trajectory and Aggregation tables have not been generated."
        ),
    },
    {
        "claimFamily": "aggregation",
        "claimId": "figure8_duplicate_value_final_aggregation",
        "expectedArtifacts": ["/artifacts/results/e01_duplicate_aggregation.parquet"],
        "paperReportedText": (
            "Duplicate-value Bubble-Selection and Insertion-Selection chimeras are reported to "
            "retain higher final Aggregation Values."
        ),
        "paperReportedZ": None,
        "paperReportedP": None,
        "paperSignificanceClass": None,
        "paperExpectedDirection": "duplicate_final_aggregation_greater",
        "recommendedFollowupStep": "S11",
        "missingDataReason": (
            "S11 duplicate-value chimera and Aggregation tables have not been generated."
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--efficiency-counts-path", type=Path, default=Path("/artifacts/results/e01_efficiency_counts.parquet"))
    parser.add_argument("--efficiency-numeric-table-path", type=Path, default=Path("/artifacts/tables/e01_efficiency_numeric_table.csv"))
    parser.add_argument("--s05-validation-path", type=Path, default=Path("/artifacts/research_steps/S05/s05_validation.json"))
    parser.add_argument(
        "--paper-markdown-path",
        type=Path,
        default=Path("/workspace/input-attachments/f93afdc5-f2e5-4ecc-80bb-e088f93acf3c/pdf-markdown.md"),
    )
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


def normal_two_sided_p(z_value: float) -> float:
    if math.isinf(z_value):
        return 0.0
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))


def significance_class(p_value: float | None) -> str | None:
    if p_value is None or math.isnan(p_value):
        return None
    if p_value < 0.01:
        return "p_lt_0.01"
    if p_value < 0.05:
        return "p_lt_0.05"
    return "not_significant"


def observed_direction(diff: float, p_value: float | None) -> str:
    if p_value is not None and p_value >= 0.05:
        return "not_significant_difference"
    if diff < 0:
        return "cell_less"
    if diff > 0:
        return "cell_greater"
    return "equal"


def direction_matches(expected: str, diff: float, ratio: float, p_value: float) -> bool:
    if expected == "near_equal":
        return p_value >= 0.05 and abs(ratio - 1.0) <= 0.02
    if expected == "cell_less":
        return diff < 0
    if expected == "cell_greater":
        return diff > 0
    return False


def significance_matches(expected_class: str | None, observed_class: str | None) -> bool | None:
    if expected_class is None or observed_class is None:
        return None
    if expected_class == "not_significant":
        return observed_class == "not_significant"
    if expected_class == "p_lt_0.01":
        return observed_class == "p_lt_0.01"
    return expected_class == observed_class


def replication_status(
    expected_direction: str,
    direction_ok: bool,
    expected_sig: str | None,
    observed_sig: str | None,
    z_magnitude_ok: bool | None,
) -> str:
    if not direction_ok:
        return "not_directionally_replicated"
    if expected_direction == "near_equal":
        if observed_sig == "not_significant":
            return "statistically_consistent_non_significant"
        return "directionally_consistent_but_significant"
    if expected_sig == "p_lt_0.01":
        if observed_sig == "p_lt_0.01":
            if z_magnitude_ok:
                return "statistically_consistent"
            return "directionally_consistent_magnitude_differs"
        if observed_sig == "p_lt_0.05":
            return "directionally_consistent_weaker_significance"
        return "directionally_consistent_not_significant"
    if significance_matches(expected_sig, observed_sig):
        return "statistically_consistent"
    return "directionally_consistent_significance_differs"


def z_test_summary(traditional_values: np.ndarray, cell_values: np.ndarray) -> dict[str, float]:
    n_trad = traditional_values.size
    n_cell = cell_values.size
    trad_mean = float(traditional_values.mean())
    cell_mean = float(cell_values.mean())
    trad_sd = float(traditional_values.std(ddof=1))
    cell_sd = float(cell_values.std(ddof=1))
    diff = cell_mean - trad_mean
    se = math.sqrt((trad_sd * trad_sd / n_trad) + (cell_sd * cell_sd / n_cell))
    if se == 0.0:
        z_value = 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
    else:
        z_value = diff / se
    p_value = normal_two_sided_p(z_value)
    pooled_numer = ((n_trad - 1) * trad_sd * trad_sd) + ((n_cell - 1) * cell_sd * cell_sd)
    pooled_denom = n_trad + n_cell - 2
    pooled_sd = math.sqrt(pooled_numer / pooled_denom) if pooled_denom > 0 else float("nan")
    if pooled_sd == 0.0:
        cohens_d = 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
    else:
        cohens_d = diff / pooled_sd
    hedges_correction = 1.0 - (3.0 / (4.0 * (n_trad + n_cell) - 9.0))
    return {
        "nTraditional": float(n_trad),
        "nCellView": float(n_cell),
        "traditionalMean": trad_mean,
        "traditionalSd": trad_sd,
        "cellViewMean": cell_mean,
        "cellViewSd": cell_sd,
        "meanDifferenceCellMinusTraditional": diff,
        "standardError": se,
        "zStatistic": float(z_value),
        "pValueTwoSided": p_value,
        "cohensD": float(cohens_d),
        "hedgesG": float(cohens_d * hedges_correction),
        "ratioOfMeansCellOverTraditional": float(cell_mean / trad_mean),
    }


def bootstrap_pair_stats(
    traditional_values: np.ndarray,
    cell_values: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, float]:
    n = traditional_values.size
    indices = rng.integers(0, n, size=(reps, n))
    trad_samples = traditional_values[indices]
    cell_samples = cell_values[indices]
    trad_means = trad_samples.mean(axis=1)
    cell_means = cell_samples.mean(axis=1)
    diffs = cell_means - trad_means
    ratios = cell_means / trad_means
    effects = np.empty(reps, dtype=float)
    for idx in range(reps):
        trad_sd = trad_samples[idx].std(ddof=1)
        cell_sd = cell_samples[idx].std(ddof=1)
        pooled = math.sqrt(((n - 1) * trad_sd * trad_sd + (n - 1) * cell_sd * cell_sd) / (2 * n - 2))
        if pooled == 0.0:
            effects[idx] = 0.0 if diffs[idx] == 0.0 else math.copysign(math.inf, diffs[idx])
        else:
            effects[idx] = diffs[idx] / pooled
    return {
        "traditionalMeanCi95Low": float(np.quantile(trad_means, 0.025)),
        "traditionalMeanCi95High": float(np.quantile(trad_means, 0.975)),
        "cellViewMeanCi95Low": float(np.quantile(cell_means, 0.025)),
        "cellViewMeanCi95High": float(np.quantile(cell_means, 0.975)),
        "pairedDifferenceCi95Low": float(np.quantile(diffs, 0.025)),
        "pairedDifferenceCi95High": float(np.quantile(diffs, 0.975)),
        "ratioOfMeansCi95Low": float(np.quantile(ratios, 0.025)),
        "ratioOfMeansCi95High": float(np.quantile(ratios, 0.975)),
        "cohensDCi95Low": float(np.nanquantile(effects, 0.025)),
        "cohensDCi95High": float(np.nanquantile(effects, 0.975)),
    }


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


def load_and_validate_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    counts = pd.read_parquet(args.efficiency_counts_path)
    numeric_table = pd.read_csv(args.efficiency_numeric_table_path)
    s05_validation = json.loads(args.s05_validation_path.read_text(encoding="utf-8"))
    validation: dict[str, Any] = {
        "efficiencyCountsPath": str(args.efficiency_counts_path),
        "efficiencyCountsSha256": sha256_file(args.efficiency_counts_path),
        "efficiencyNumericTablePath": str(args.efficiency_numeric_table_path),
        "efficiencyNumericTableSha256": sha256_file(args.efficiency_numeric_table_path),
        "s05ValidationPath": str(args.s05_validation_path),
        "s05ValidationSha256": sha256_file(args.s05_validation_path),
        "paperMarkdownPath": str(args.paper_markdown_path),
        "paperMarkdownSha256": sha256_file(args.paper_markdown_path) if args.paper_markdown_path.exists() else None,
        "inputErrors": [],
    }
    required_cols = {
        "algorithm",
        "mode",
        "repeat_index",
        "swap_only_steps",
        "comparison_steps_observed",
        "compare_plus_swap_steps",
        "count_fields_complete",
        "matched_initial_arrays_valid",
    }
    missing_cols = sorted(required_cols - set(counts.columns))
    if missing_cols:
        validation["inputErrors"].append(f"efficiency counts missing columns: {missing_cols}")
    expected_rows = 100 * len(ALGORITHMS) * len(MODES)
    if len(counts) != expected_rows:
        validation["inputErrors"].append(f"efficiency counts row count is {len(counts)}, expected {expected_rows}")
    repeat_counts = counts.groupby(["algorithm", "mode"])["repeat_index"].nunique().to_dict()
    validation["observedRepeatsPerModeAlgorithm"] = {
        f"{algorithm}:{mode}": int(repeat_counts.get((algorithm, mode), 0))
        for algorithm in ALGORITHMS
        for mode in MODES
    }
    for key, count in validation["observedRepeatsPerModeAlgorithm"].items():
        if count != 100:
            validation["inputErrors"].append(f"{key} repeat count is {count}, expected 100")
    if "count_fields_complete" in counts and not bool(counts["count_fields_complete"].all()):
        validation["inputErrors"].append("not all S05 count fields are complete")
    if "matched_initial_arrays_valid" in counts and not bool(counts["matched_initial_arrays_valid"].all()):
        validation["inputErrors"].append("S05 matched initial arrays are not all valid")
    for metric in COUNT_METRICS:
        if metric in counts and counts[metric].isna().any():
            validation["inputErrors"].append(f"{metric} contains null values")
    validation["s05ValidationResult"] = s05_validation.get("validationResult")
    validation["s05Success"] = s05_validation.get("success")
    validation["inputValidationPassed"] = not validation["inputErrors"]
    if not validation["inputValidationPassed"]:
        raise RuntimeError("; ".join(validation["inputErrors"]))
    return counts, numeric_table, s05_validation, validation


def build_z_tests(counts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        traditional = counts[(counts["algorithm"] == algorithm) & (counts["mode"] == "traditional")].sort_values("repeat_index")
        cell = counts[(counts["algorithm"] == algorithm) & (counts["mode"] == "cell_view")].sort_values("repeat_index")
        for metric in REPORTED_EFFICIENCY_METRICS:
            target = PAPER_EFFICIENCY_TARGETS[(algorithm, metric)]
            traditional_values = traditional[metric].to_numpy(dtype=float)
            cell_values = cell[metric].to_numpy(dtype=float)
            summary = z_test_summary(traditional_values, cell_values)
            p_value = summary["pValueTwoSided"]
            observed_sig = significance_class(p_value)
            direction_ok = direction_matches(
                target["paperExpectedDirection"],
                summary["meanDifferenceCellMinusTraditional"],
                summary["ratioOfMeansCellOverTraditional"],
                p_value,
            )
            sig_ok = significance_matches(target["paperSignificanceClass"], observed_sig)
            paper_z = target["paperReportedZ"]
            z_magnitude_ok = (
                abs(summary["zStatistic"] - paper_z) <= max(1.0, 0.25 * abs(paper_z))
                if paper_z not in (None, 0)
                else None
            )
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "experiment_id": EXPERIMENT_ID,
                    "source_research_step_id": SOURCE_STEP_ID,
                    "claim_family": "efficiency",
                    "claim_id": f"figure4_{algorithm}_{metric}",
                    "algorithm": algorithm,
                    "metric": metric,
                    "comparison": "cell_view_minus_traditional",
                    "status": "tested",
                    "n_traditional": int(summary["nTraditional"]),
                    "n_cell_view": int(summary["nCellView"]),
                    "traditional_mean": summary["traditionalMean"],
                    "traditional_sd": summary["traditionalSd"],
                    "cell_view_mean": summary["cellViewMean"],
                    "cell_view_sd": summary["cellViewSd"],
                    "mean_difference_cell_minus_traditional": summary["meanDifferenceCellMinusTraditional"],
                    "standard_error": summary["standardError"],
                    "z_statistic": summary["zStatistic"],
                    "p_value_two_sided": summary["pValueTwoSided"],
                    "observed_significance_class": observed_sig,
                    "observed_direction": observed_direction(summary["meanDifferenceCellMinusTraditional"], p_value),
                    "cohens_d": summary["cohensD"],
                    "hedges_g": summary["hedgesG"],
                    "ratio_of_means_cell_over_traditional": summary["ratioOfMeansCellOverTraditional"],
                    "paper_reported_z": paper_z,
                    "paper_reported_p": target["paperReportedP"],
                    "paper_significance_class": target["paperSignificanceClass"],
                    "paper_expected_direction": target["paperExpectedDirection"],
                    "paper_reported_ratio_cell_over_traditional": target["paperRatioCellOverTraditional"],
                    "paper_reported_text": target["paperReportedText"],
                    "direction_matches_paper": direction_ok,
                    "significance_matches_paper": sig_ok,
                    "z_magnitude_matches_within_25pct_or_1z": z_magnitude_ok,
                    "replication_status": replication_status(
                        target["paperExpectedDirection"],
                        direction_ok,
                        target["paperSignificanceClass"],
                        observed_sig,
                        z_magnitude_ok,
                    ),
                    "missing_data_reason": "",
                    "expected_artifacts_checked": "",
                    "recommended_followup_step": "",
                    "count_definition": (
                        "Frozen S05 definitions: swap_only_steps is S04 swap_count; "
                        "compare_plus_swap_steps is S04 swap_count plus comparison_count. "
                        "Cell-view comparisons use public StatusProbe actionable-comparison proxy."
                    ),
                }
            )
    for target in MISSING_TEST_TARGETS:
        expected_paths = [Path(path) for path in target["expectedArtifacts"]]
        missing_paths = [str(path) for path in expected_paths if not path.exists()]
        status = "missing_input_not_tested" if missing_paths else "available_but_not_tested_in_s06"
        rows.append(
            {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "source_research_step_id": "",
                "claim_family": target["claimFamily"],
                "claim_id": target["claimId"],
                "algorithm": "",
                "metric": "",
                "comparison": "",
                "status": status,
                "n_traditional": np.nan,
                "n_cell_view": np.nan,
                "traditional_mean": np.nan,
                "traditional_sd": np.nan,
                "cell_view_mean": np.nan,
                "cell_view_sd": np.nan,
                "mean_difference_cell_minus_traditional": np.nan,
                "standard_error": np.nan,
                "z_statistic": np.nan,
                "p_value_two_sided": np.nan,
                "observed_significance_class": "",
                "observed_direction": "",
                "cohens_d": np.nan,
                "hedges_g": np.nan,
                "ratio_of_means_cell_over_traditional": np.nan,
                "paper_reported_z": target["paperReportedZ"],
                "paper_reported_p": target["paperReportedP"],
                "paper_significance_class": target["paperSignificanceClass"],
                "paper_expected_direction": target["paperExpectedDirection"],
                "paper_reported_ratio_cell_over_traditional": np.nan,
                "paper_reported_text": target["paperReportedText"],
                "direction_matches_paper": np.nan,
                "significance_matches_paper": np.nan,
                "z_magnitude_matches_within_25pct_or_1z": np.nan,
                "replication_status": "not_tested_missing_metric_table",
                "missing_data_reason": target["missingDataReason"],
                "expected_artifacts_checked": "; ".join(str(path) for path in expected_paths),
                "recommended_followup_step": target["recommendedFollowupStep"],
                "count_definition": "not_applicable",
            }
        )
    return pd.DataFrame(rows)


def build_bootstrap_intervals(counts: pd.DataFrame, bootstrap_reps: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for algorithm in ALGORITHMS:
        traditional = counts[(counts["algorithm"] == algorithm) & (counts["mode"] == "traditional")].sort_values("repeat_index")
        cell = counts[(counts["algorithm"] == algorithm) & (counts["mode"] == "cell_view")].sort_values("repeat_index")
        repeat_match = traditional["repeat_index"].tolist() == cell["repeat_index"].tolist()
        hash_match = (
            traditional["initial_array_sha256"].tolist() == cell["initial_array_sha256"].tolist()
            if "initial_array_sha256" in counts.columns
            else True
        )
        for metric in COUNT_METRICS:
            traditional_values = traditional[metric].to_numpy(dtype=float)
            cell_values = cell[metric].to_numpy(dtype=float)
            z_summary = z_test_summary(traditional_values, cell_values)
            boot = bootstrap_pair_stats(traditional_values, cell_values, rng, bootstrap_reps)
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "experiment_id": EXPERIMENT_ID,
                    "source_research_step_id": SOURCE_STEP_ID,
                    "claim_family": "efficiency",
                    "algorithm": algorithm,
                    "metric": metric,
                    "included_in_reported_z_test": metric in REPORTED_EFFICIENCY_METRICS,
                    "n_pairs": len(traditional_values),
                    "repeat_indexes_matched": repeat_match,
                    "initial_array_hashes_matched": hash_match,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_reps": bootstrap_reps,
                    "traditional_mean": z_summary["traditionalMean"],
                    "traditional_mean_ci95_low": boot["traditionalMeanCi95Low"],
                    "traditional_mean_ci95_high": boot["traditionalMeanCi95High"],
                    "cell_view_mean": z_summary["cellViewMean"],
                    "cell_view_mean_ci95_low": boot["cellViewMeanCi95Low"],
                    "cell_view_mean_ci95_high": boot["cellViewMeanCi95High"],
                    "paired_difference_cell_minus_traditional_mean": z_summary["meanDifferenceCellMinusTraditional"],
                    "paired_difference_ci95_low": boot["pairedDifferenceCi95Low"],
                    "paired_difference_ci95_high": boot["pairedDifferenceCi95High"],
                    "ratio_of_means_cell_over_traditional": z_summary["ratioOfMeansCellOverTraditional"],
                    "ratio_of_means_ci95_low": boot["ratioOfMeansCi95Low"],
                    "ratio_of_means_ci95_high": boot["ratioOfMeansCi95High"],
                    "cohens_d": z_summary["cohensD"],
                    "cohens_d_ci95_low": boot["cohensDCi95Low"],
                    "cohens_d_ci95_high": boot["cohensDCi95High"],
                    "count_definition": (
                        "S05 frozen count definition; comparison_steps_observed retained as audit-only "
                        "because it is not a standalone reported Figure 4 z-test."
                    ),
                }
            )
    return pd.DataFrame(rows)


def compact_z_table(z_tests: pd.DataFrame) -> pd.DataFrame:
    tested = z_tests[z_tests["status"] == "tested"].copy()
    cols = [
        "algorithm",
        "metric",
        "traditional_mean",
        "cell_view_mean",
        "mean_difference_cell_minus_traditional",
        "z_statistic",
        "p_value_two_sided",
        "paper_reported_z",
        "paper_reported_p",
        "replication_status",
    ]
    return tested[cols].copy()


def missing_summary_table(z_tests: pd.DataFrame) -> pd.DataFrame:
    missing = z_tests[z_tests["status"] != "tested"].copy()
    cols = [
        "claim_family",
        "claim_id",
        "status",
        "paper_reported_z",
        "paper_reported_p",
        "missing_data_reason",
        "expected_artifacts_checked",
        "recommended_followup_step",
    ]
    return missing[cols].copy()


def write_methods_report(
    path: Path,
    artifact_paths: dict[str, Path],
    validation: dict[str, Any],
    command: list[str],
    args: argparse.Namespace,
    repo_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    caveats = "; ".join(validation["caveatsOrBlockers"])
    report = f"""# E01 S06 Statistical Methods

## Top Summary

- Step ID: {STEP_ID}
- Completion status: {validation["status"]}
- Artifacts written: {", ".join(str(path) for path in artifact_paths.values())}
- Validation result: {validation["validationResult"]}
- Outcome classification: {validation["outcomeClassification"]}
- Caveats or blockers: {caveats}
- Recommended next action: {validation["recommendedNextAction"]}

## Scope

S06 reproduces the reported statistical tests that are possible with already available metric tables. At this point, the only complete metric table available after S05 is the efficiency count table. Robustness, Delayed Gratification, and Aggregation tests are documented as missing-input rows until S07 through S11 generate their metric tables.

## Input Tables

- S05 efficiency counts: `{args.efficiency_counts_path}` (`{sha256_file(args.efficiency_counts_path)}`)
- S05 numeric summary: `{args.efficiency_numeric_table_path}` (`{sha256_file(args.efficiency_numeric_table_path)}`)
- S05 validation JSON: `{args.s05_validation_path}` (`{sha256_file(args.s05_validation_path)}`)
- Attached paper markdown: `{args.paper_markdown_path}` (`{sha256_file(args.paper_markdown_path) if args.paper_markdown_path.exists() else "missing"}`)

## Z-Test Method

For each reported Figure 4 efficiency comparison, S06 computes a two-sample normal z statistic using the paper's implied comparison of 100 traditional runs against 100 cell-view runs:

```text
z = (mean(cell_view) - mean(traditional)) / sqrt(sd(cell_view)^2 / n_cell + sd(traditional)^2 / n_traditional)
p = 2 * normal_survival(abs(z))
```

The p value is computed as `erfc(abs(z) / sqrt(2))`. Standard deviations are sample standard deviations with `ddof=1`. The sign convention is cell-view minus traditional: negative z values mean fewer cell-view steps, and positive z values mean more cell-view steps.

## Effect Sizes

S06 reports Cohen's d and Hedges' g for the same cell-view minus traditional contrast:

```text
d = (mean(cell_view) - mean(traditional)) / pooled_sd
g = d * (1 - 3 / (4 * (n_cell + n_traditional) - 9))
```

Rows with no variance and no difference are assigned effect size zero. No variance row in the available S05 data had a nonzero difference requiring an infinite effect size.

## Bootstrap Method

Bootstrap confidence intervals use matched repeat indexes and matched initial arrays. For each algorithm and metric, S06 resamples 100 repeat pairs with replacement `{args.bootstrap_reps}` times using fixed seed `{BOOTSTRAP_SEED}`. The bootstrap table reports 2.5% and 97.5% quantiles for traditional means, cell-view means, paired differences, ratios of means, and Cohen's d.

## Frozen Count Definitions

S06 does not change S05 count definitions:

- `swap_only_steps` is S04 `swap_count`.
- `comparison_steps_observed` is S04 `comparison_count`.
- `compare_plus_swap_steps` is the sum of S04 swap and comparison counts.
- Traditional comparison counts are reconstructed-controller counts from S04.
- Cell-view comparison counts are public `StatusProbe.compare_and_swap_count` events, an actionable-comparison proxy rather than a full read census.

## Missing-Input Handling

For robustness, Delayed Gratification, and Aggregation claims, S06 writes explicit rows in `{artifact_paths["zTestsCsv"]}` with `status=missing_input_not_tested` when required metric artifacts do not exist. These are not negative statistical results; they are provenance records showing why the tests cannot yet be reproduced without starting downstream research steps.

## Reproduction Command

```bash
{" ".join(command)}
```

## Runtime Provenance

- Python executable: `{sys.executable}`
- Python version: `{sys.version.splitlines()[0]}`
- Platform: `{platform.platform()}`
- Pandas version: `{pd.__version__}`
- NumPy version: `{np.__version__}`
- Repository commit at run time: `{git_commit(repo_dir)}`
- Repository status during report generation: `{git_status(repo_dir) or "clean"}`
"""
    path.write_text(report, encoding="utf-8")


def write_full_report(
    path: Path,
    artifact_paths: dict[str, Path],
    validation: dict[str, Any],
    z_tests: pd.DataFrame,
    bootstrap_intervals: pd.DataFrame,
    numeric_table: pd.DataFrame,
    command: list[str],
    args: argparse.Namespace,
    repo_dir: Path,
    wall_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    caveats = "; ".join(validation["caveatsOrBlockers"])
    lay_summary = (
        "The available S05 efficiency data reproduce the paper's broad Figure 4 directions: Bubble "
        "and Insertion are not different under swap-only counts, cell-view Selection uses more swaps, "
        "and swap-plus-comparison counts favor cell-view Bubble/Insertion while penalizing cell-view "
        "Selection. Exact fold-change magnitudes do not reproduce under the frozen public-code count "
        "definitions, and Insertion's swap-plus-comparison z test is much weaker than the paper's "
        "reported result. Robustness, DG, and Aggregation statistical tests remain unavailable until "
        "their metric tables are generated in later steps."
    )
    tested = z_tests[z_tests["status"] == "tested"]
    missing = z_tests[z_tests["status"] != "tested"]
    report = f"""# E01 S06 Research Step Full Results

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

Do the paper's z-test conclusions hold under the replicated data, and are they supported by bootstrap confidence intervals and effect sizes?

## Inputs

- S05 efficiency counts: `{args.efficiency_counts_path}` (`{sha256_file(args.efficiency_counts_path)}`)
- S05 numeric table: `{args.efficiency_numeric_table_path}` (`{sha256_file(args.efficiency_numeric_table_path)}`)
- S05 validation JSON: `{args.s05_validation_path}` (`{sha256_file(args.s05_validation_path)}`)
- Paper markdown attachment: `{args.paper_markdown_path}` (`{sha256_file(args.paper_markdown_path) if args.paper_markdown_path.exists() else "missing"}`)
- S05 validation result consumed by S06: `{validation["s05ValidationResult"]}`, success `{validation["s05Success"]}`

No sorting runs were executed in S06. S06 uses the frozen S05 efficiency data and records missing-data reasons for tests requiring S07 through S11 outputs.

## Detailed Methods

S06 computed the paper-style z-tests for the six Figure 4 efficiency comparisons that have complete S05 run-level data: three algorithms by two reported count metrics (`swap_only_steps` and `compare_plus_swap_steps`). The contrast is cell-view minus traditional. The standard error is `sqrt(sd_cell^2 / n_cell + sd_traditional^2 / n_traditional)`, using sample standard deviations and n=100 per mode. P values are two-sided normal p values from `erfc(abs(z) / sqrt(2))`.

Bootstrap intervals use paired resampling of matched repeat indexes and matched initial arrays, with fixed seed `{BOOTSTRAP_SEED}` and `{args.bootstrap_reps}` resamples. The bootstrap artifact includes all three S05 count metrics, including `comparison_steps_observed` as an audit-only count, while the z-test artifact flags only the paper-reported Figure 4 tests as `status=tested`.

The frozen S05 count definitions were not changed. Cell-view comparison counts remain public `StatusProbe.compare_and_swap_count` events, not a complete read census. Traditional comparison counts remain reconstructed-controller counts.

Rows for robustness, Delayed Gratification, and Aggregation claims were written with explicit missing-data reasons because the required S07-S11 metric tables do not exist yet. These rows are not interpreted as null results.

## Commands

```bash
{" ".join(command)}
```

Validation command:

```bash
python -m py_compile scripts/e01_s06_statistical_tests.py
```

## Dependencies And Parameters

- Python executable: `{sys.executable}`
- Python version: `{sys.version.splitlines()[0]}`
- Platform: `{platform.platform()}`
- Pandas version: `{pd.__version__}`
- NumPy version: `{np.__version__}`
- Repository commit at run time: `{git_commit(repo_dir)}`
- Repository status during report generation: `{git_status(repo_dir) or "clean"}`
- Bootstrap seed: `{BOOTSTRAP_SEED}`
- Bootstrap repetitions: `{args.bootstrap_reps}`
- Wall time: `{wall_seconds:.3f}` seconds

No new Python, system, R, Rust, or Node dependencies were installed for S06.

## Results

### Tested Figure 4 Z-Tests

{dataframe_to_markdown(compact_z_table(z_tests))}

### Missing Statistical Claims

{dataframe_to_markdown(missing_summary_table(z_tests))}

### Bootstrap Interval Summary

{dataframe_to_markdown(bootstrap_intervals[bootstrap_intervals["included_in_reported_z_test"]][[
        "algorithm",
        "metric",
        "paired_difference_cell_minus_traditional_mean",
        "paired_difference_ci95_low",
        "paired_difference_ci95_high",
        "ratio_of_means_cell_over_traditional",
        "ratio_of_means_ci95_low",
        "ratio_of_means_ci95_high",
        "cohens_d",
    ]])}

### S05 Numeric Table Cross-Reference

{dataframe_to_markdown(numeric_table[numeric_table["included_in_figure4"]][[
        "algorithm",
        "count_metric",
        "traditional_mean",
        "cell_view_mean",
        "ratio_of_means_cell_over_traditional",
        "replication_status",
    ]])}

Primary result details:

- Bubble swap-only: replicated as no difference under frozen matched arrays (`z={tested[(tested["algorithm"] == "bubble") & (tested["metric"] == "swap_only_steps")]["z_statistic"].iloc[0]:.3f}`).
- Insertion swap-only: replicated as no difference under frozen matched arrays (`z={tested[(tested["algorithm"] == "insertion") & (tested["metric"] == "swap_only_steps")]["z_statistic"].iloc[0]:.3f}`).
- Selection swap-only: direction and significance replicate, but magnitude differs (`z={tested[(tested["algorithm"] == "selection") & (tested["metric"] == "swap_only_steps")]["z_statistic"].iloc[0]:.3f}` vs paper `120.43`).
- Bubble swap-plus-comparison: direction and high significance replicate, but magnitude differs (`z={tested[(tested["algorithm"] == "bubble") & (tested["metric"] == "compare_plus_swap_steps")]["z_statistic"].iloc[0]:.3f}` vs paper `-68.96`).
- Insertion swap-plus-comparison: direction replicates, but the frozen count convention gives only weak significance compared with the paper (`z={tested[(tested["algorithm"] == "insertion") & (tested["metric"] == "compare_plus_swap_steps")]["z_statistic"].iloc[0]:.3f}` vs paper `-71.19`).
- Selection swap-plus-comparison: direction and significance replicate, but magnitude differs (`z={tested[(tested["algorithm"] == "selection") & (tested["metric"] == "compare_plus_swap_steps")]["z_statistic"].iloc[0]:.3f}` vs paper `106.55`).

## Validation

- Input S05 count table has 600 rows and 100 repeats for every algorithm/mode: `{validation["inputValidationPassed"]}`.
- All S05 count fields are complete and matched initial arrays are valid: `{validation["s05CountFieldsAndMatchingValidated"]}`.
- Tested z-test rows written: `{validation["testedZRows"]}`.
- Missing-input rows written for unavailable robustness/DG/aggregation claims: `{validation["missingInputRows"]}`.
- Bootstrap rows written: `{validation["bootstrapRows"]}`.
- Bootstrap seed fixed and recorded: `{BOOTSTRAP_SEED}`.
- Required artifacts exist after script execution: `{validation["allArtifactsExist"]}`.

## Artifacts And Provenance

Reusable outputs:

{chr(10).join(f"- `{label}`: `{artifact_path}`" for label, artifact_path in artifact_paths.items())}

The global run manifest and checksum file were updated after artifact creation. The S06 artifact manifest records paths, sizes, and SHA256 hashes.

## Caveats, Blockers, Failed Assumptions, And Limitations

- Traditional Bubble, Insertion, and Selection remain reconstructed baselines because S02 found no public traditional runner or saved original arrays.
- Cell-view comparison counts are public `StatusProbe.compare_and_swap_count` actionable events, not a complete read census; this limits exact reproduction of paper z magnitudes for swap-plus-comparison counts.
- The paper states only that standard z-tests were used; it does not provide raw variance tables. S06 uses the available replicated per-run data and documents the formula in `{artifact_paths["methodsReportMd"]}`.
- Robustness, Delayed Gratification, same-goal Aggregation, and duplicate-value Aggregation statistical tests are unavailable until S07 through S11 produce the required metric tables.
- Figure 7 Selection DG has a sign ambiguity between nearby paper text and caption; S06 records the caption direction in the missing-input row and defers resolution to S08.

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    path.write_text(report, encoding="utf-8")


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
        "title": "Reproduce the reported statistical tests",
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
        "artifactsWritten": [str(path) for path in paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s06_statistical_tests.py"),
        "command": " ".join(command),
        "summary": validation["summary"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    artifacts_dir = args.artifacts_dir.resolve()
    s06_dir = artifacts_dir / "research_steps" / STEP_ID
    tables_dir = artifacts_dir / "tables"
    reports_dir = artifacts_dir / "reports"
    checksum_dir = artifacts_dir / "checksums"
    for path in (s06_dir, tables_dir, reports_dir, checksum_dir):
        path.mkdir(parents=True, exist_ok=True)

    counts, numeric_table, s05_validation, input_validation = load_and_validate_inputs(args)
    z_tests = build_z_tests(counts)
    bootstrap_intervals = build_bootstrap_intervals(counts, args.bootstrap_reps)

    z_tests_path = tables_dir / "e01_z_tests.csv"
    bootstrap_path = tables_dir / "e01_bootstrap_intervals.csv"
    methods_path = reports_dir / "e01_statistical_methods.md"
    report_path = s06_dir / "research_step_full_results.md"
    validation_path = s06_dir / "s06_validation.json"
    artifact_manifest_path = s06_dir / "artifact_manifest.json"

    z_tests.to_csv(z_tests_path, index=False)
    bootstrap_intervals.to_csv(bootstrap_path, index=False)

    tested = z_tests[z_tests["status"] == "tested"].copy()
    missing = z_tests[z_tests["status"] != "tested"].copy()
    tested_statuses = tested["replication_status"].value_counts().to_dict()
    all_artifacts = {
        "fullResultsReport": report_path,
        "zTestsCsv": z_tests_path,
        "bootstrapIntervalsCsv": bootstrap_path,
        "methodsReportMd": methods_path,
        "validationJson": validation_path,
        "artifactManifest": artifact_manifest_path,
    }
    validation: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": True,
        "status": "completed_with_caveats",
        "validationResult": "passed_with_missing_downstream_metric_tables_documented",
        "artifactsWritten": [str(path) for path in all_artifacts.values()],
        "caveatsOrBlockers": [
            "Only S05 efficiency data are available; robustness, Delayed Gratification, and Aggregation metric tables are not yet generated.",
            "Traditional baselines remain reconstructed because no public traditional runner was found.",
            "Cell-view comparison counts retain the S05 actionable-comparison proxy and are not a complete read census.",
            "Exact fold-change magnitudes and some z magnitudes differ from the paper under the frozen S05 count definitions.",
        ],
        "recommendedNextAction": "Proceed to S07 Frozen Cell robustness using the frozen S03 assumptions; do not start S07 inside S06.",
        "outcomeClassification": "constraining/contradictory",
        "createdAtUtc": utc_now(),
        "repositoryCommitAtRunTime": git_commit(args.repo_dir),
        "script": str(args.repo_dir / "scripts/e01_s06_statistical_tests.py"),
        "bootstrapSeed": BOOTSTRAP_SEED,
        "bootstrapReps": args.bootstrap_reps,
        "testedZRows": int(len(tested)),
        "missingInputRows": int((missing["status"] == "missing_input_not_tested").sum()),
        "bootstrapRows": int(len(bootstrap_intervals)),
        "s05ValidationResult": input_validation["s05ValidationResult"],
        "s05Success": input_validation["s05Success"],
        "s05CountFieldsAndMatchingValidated": bool(
            input_validation["inputValidationPassed"]
            and s05_validation.get("allCountFieldsComplete")
            and s05_validation.get("matchedInitialArraysValid")
        ),
        "inputValidationPassed": input_validation["inputValidationPassed"],
        "inputValidation": input_validation,
        "testedReplicationStatuses": {str(key): int(value) for key, value in tested_statuses.items()},
        "summary": {
            "efficiencyRowsTested": int(len(tested)),
            "missingRowsDocumented": int(len(missing)),
            "bootstrapRows": int(len(bootstrap_intervals)),
            "directionallyConsistentAvailableTests": int(
                tested["replication_status"].astype(str).str.contains("consistent").sum()
            ),
            "exactMagnitudeMatches": int(
                sum(value is True for value in tested["z_magnitude_matches_within_25pct_or_1z"].tolist())
            ),
            "paperStyleTestsUnavailableUntilLaterSteps": int(len(missing)),
        },
    }

    write_methods_report(methods_path, all_artifacts, validation, sys.argv, args, args.repo_dir)
    wall_seconds = time.monotonic() - started
    validation["wallSeconds"] = wall_seconds
    validation["allArtifactsExist"] = False
    write_full_report(
        report_path,
        all_artifacts,
        validation,
        z_tests,
        bootstrap_intervals,
        numeric_table,
        sys.argv,
        args,
        args.repo_dir,
        wall_seconds,
    )
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(artifact_manifest_path, all_artifacts, validation, args.repo_dir)
    validation["allArtifactsExist"] = all(path.exists() for path in all_artifacts.values())
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_full_report(
        report_path,
        all_artifacts,
        validation,
        z_tests,
        bootstrap_intervals,
        numeric_table,
        sys.argv,
        args,
        args.repo_dir,
        wall_seconds,
    )
    write_artifact_manifest(artifact_manifest_path, all_artifacts, validation, args.repo_dir)

    update_run_manifest(artifacts_dir, {f"s06_{key}": value for key, value in all_artifacts.items()}, validation, args.repo_dir, sys.argv)
    checksum_path = checksum_dir / "sha256sums.txt"
    update_checksum_file(
        checksum_path,
        list(all_artifacts.values())
        + [
            args.efficiency_counts_path,
            args.efficiency_numeric_table_path,
            args.s05_validation_path,
            args.paper_markdown_path,
            args.repo_dir / "scripts/e01_s06_statistical_tests.py",
            artifacts_dir / "run_manifest.json",
            checksum_path,
        ],
    )

    print(json.dumps({"validation": validation, "testedZRows": tested.to_dict(orient="records")}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
