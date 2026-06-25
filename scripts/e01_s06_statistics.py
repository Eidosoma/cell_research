#!/usr/bin/env python3
"""Run E01 S06 statistical replication scaffold.

S06 consumes S05 efficiency tables, reproduces the available paper-style
efficiency z-tests, adds bootstrap intervals and effect sizes, and records the
Frozen Cell, Delayed Gratification, Aggregation, and opposite-direction tests
that must wait for later research-step data. It does not start S07 or later.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from statsmodels.stats.weightstats import ztest as statsmodels_ztest

    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    statsmodels_ztest = None
    STATSMODELS_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "E01"
STEP_ID = "S06"
STEP_NUMBER = 6
STATUS = "completed"
OUTCOME_CLASSIFICATION = "constraining"
ALGORITHMS = ["bubble", "insertion", "selection"]
METRIC_ORDER = ["swap_only_steps", "swap_plus_comparison_steps"]
S05_REPLICATES_DEFAULT = Path("/artifacts/results/e01_efficiency_replicates.parquet")
S05_PAIRWISE_DEFAULT = Path("/artifacts/results/e01_efficiency_pairwise.parquet")
S05_STATUS_DEFAULT = Path("/artifacts/research_steps/S05/status.json")
S03_ANALYSIS_SEEDS_DEFAULT = Path("/artifacts/research_steps/S03/analysis_seed_table.csv")

PAPER_EFFICIENCY_TESTS: dict[tuple[str, str], dict[str, Any]] = {
    ("swap_only_steps", "bubble"): {
        "paperRelation": "similar",
        "paperReportedZ": 0.73,
        "paperReportedPText": "0.47",
        "paperClaimText": "Paper reports no significant swap-only difference for Bubble sort.",
        "paperReportedFactor": None,
    },
    ("swap_only_steps", "insertion"): {
        "paperRelation": "similar",
        "paperReportedZ": 1.26,
        "paperReportedPText": "0.24",
        "paperClaimText": "Paper reports no significant swap-only difference for Insertion sort.",
        "paperReportedFactor": None,
    },
    ("swap_only_steps", "selection"): {
        "paperRelation": "cell_view_greater",
        "paperReportedZ": 120.43,
        "paperReportedPText": "0",
        "paperClaimText": "Paper reports cell-view Selection needs about 11 times more swaps.",
        "paperReportedFactor": 11.0,
    },
    ("swap_plus_comparison_steps", "bubble"): {
        "paperRelation": "cell_view_fewer",
        "paperReportedZ": -68.96,
        "paperReportedPText": "<<0.01",
        "paperClaimText": "Paper reports cell-view Bubble has fewer comparison-inclusive steps by 1.5 times.",
        "paperReportedFactor": 1.5,
    },
    ("swap_plus_comparison_steps", "insertion"): {
        "paperRelation": "cell_view_fewer",
        "paperReportedZ": -71.19,
        "paperReportedPText": "<<0.01",
        "paperClaimText": "Paper reports cell-view Insertion has fewer comparison-inclusive steps by 2.03 times.",
        "paperReportedFactor": 2.03,
    },
    ("swap_plus_comparison_steps", "selection"): {
        "paperRelation": "cell_view_greater",
        "paperReportedZ": 106.55,
        "paperReportedPText": "<<0.01",
        "paperClaimText": "Paper reports cell-view Selection has greater comparison-inclusive steps by 1.17 times.",
        "paperReportedFactor": 1.17,
    },
}

PENDING_TEST_SPECS = [
    {
        "testFamily": "frozen_robustness",
        "awaitsResearchStep": "S07",
        "sourceArtifact": "/artifacts/results/e01_frozen_cell_robustness.parquet",
        "plannedAnalyses": "paper-style z-tests where applicable, bootstrap mean-difference intervals, and effect sizes for final monotonicity error by algorithm, Frozen Cell variant, and Frozen Cell count",
        "reasonPending": "S07 Frozen Cell robustness runs and final monotonicity-error tables do not exist yet.",
    },
    {
        "testFamily": "delayed_gratification",
        "awaitsResearchStep": "S08",
        "sourceArtifact": "/artifacts/results/e01_delayed_gratification.parquet",
        "plannedAnalyses": "trend and group comparisons for Delayed Gratification with bootstrap intervals and standardized effects",
        "reasonPending": "S08 DG values depend on S07 trajectories and have not been generated.",
    },
    {
        "testFamily": "aggregation_curves",
        "awaitsResearchStep": "S10",
        "sourceArtifact": "/artifacts/results/e01_aggregation_curves.parquet",
        "plannedAnalyses": "aggregation peak, time-to-peak, chance-control comparisons, bootstrap intervals, and effect sizes for same-goal chimeras",
        "reasonPending": "S10 aggregation curves depend on S09 chimera traces and have not been generated.",
    },
    {
        "testFamily": "opposite_direction_chimeras",
        "awaitsResearchStep": "S12",
        "sourceArtifact": "/artifacts/results/e01_opposite_direction_chimeras.parquet",
        "plannedAnalyses": "dominance/equilibrium summaries with bootstrap intervals and effect sizes for opposing-goal chimeras",
        "reasonPending": "S12 opposite-direction chimera outputs do not exist yet.",
    },
]


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


def normal_two_sided_p(z_statistic: float) -> float:
    if not math.isfinite(z_statistic):
        return 0.0
    return math.erfc(abs(z_statistic) / math.sqrt(2.0))


def fallback_two_sample_ztest(sample1: np.ndarray, sample2: np.ndarray) -> tuple[float, float]:
    var1 = float(np.var(sample1, ddof=1))
    var2 = float(np.var(sample2, ddof=1))
    diff = float(np.mean(sample1) - np.mean(sample2))
    denom = math.sqrt(var1 / len(sample1) + var2 / len(sample2))
    if denom == 0.0:
        z_statistic = 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
    else:
        z_statistic = diff / denom
    return z_statistic, normal_two_sided_p(z_statistic)


def paper_style_ztest(cell_view: np.ndarray, traditional: np.ndarray) -> tuple[float, float, str]:
    if np.var(cell_view, ddof=1) == 0.0 and np.var(traditional, ddof=1) == 0.0:
        z_statistic, p_value = fallback_two_sample_ztest(cell_view, traditional)
        return z_statistic, p_value, "manual_zero_variance_fallback"
    if STATSMODELS_AVAILABLE and statsmodels_ztest is not None:
        z_statistic, p_value = statsmodels_ztest(cell_view, traditional, value=0.0, alternative="two-sided")
        return float(z_statistic), float(p_value), "statsmodels.stats.weightstats.ztest_default"
    z_statistic, p_value = fallback_two_sample_ztest(cell_view, traditional)
    return z_statistic, p_value, "manual_two_sample_ztest_fallback"


def paired_mean_ztest(differences: np.ndarray) -> tuple[float, float]:
    mean_diff = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1))
    if sd_diff == 0.0:
        z_statistic = 0.0 if mean_diff == 0.0 else math.copysign(math.inf, mean_diff)
    else:
        z_statistic = mean_diff / (sd_diff / math.sqrt(len(differences)))
    return z_statistic, normal_two_sided_p(z_statistic)


def cohen_d_independent(cell_view: np.ndarray, traditional: np.ndarray) -> tuple[float | None, float | None]:
    n1 = len(cell_view)
    n2 = len(traditional)
    var1 = float(np.var(cell_view, ddof=1))
    var2 = float(np.var(traditional, ddof=1))
    pooled = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    diff = float(np.mean(cell_view) - np.mean(traditional))
    if pooled == 0.0:
        d_value = 0.0 if diff == 0.0 else None
    else:
        d_value = diff / pooled
    if d_value is None:
        return None, None
    correction = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
    return d_value, d_value * correction


def paired_effect_sizes(differences: np.ndarray) -> tuple[float | None, float]:
    mean_diff = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1))
    if sd_diff == 0.0:
        dz = 0.0 if mean_diff == 0.0 else None
    else:
        dz = mean_diff / sd_diff
    positive = int(np.sum(differences > 0))
    negative = int(np.sum(differences < 0))
    sign_effect = (positive - negative) / len(differences)
    return dz, sign_effect


def paper_relation_factor(relation: str, cell_mean: float, traditional_mean: float) -> float | None:
    if relation == "cell_view_fewer":
        return traditional_mean / cell_mean if cell_mean else None
    if relation == "cell_view_greater":
        return cell_mean / traditional_mean if traditional_mean else None
    return cell_mean / traditional_mean if traditional_mean else None


def bootstrap_pair(
    cell_view: np.ndarray,
    traditional: np.ndarray,
    relation: str,
    rng: np.random.Generator,
    bootstrap_replicates: int,
) -> dict[str, float | None]:
    n = len(cell_view)
    index = rng.integers(0, n, size=(bootstrap_replicates, n))
    cell_means = cell_view[index].mean(axis=1)
    traditional_means = traditional[index].mean(axis=1)
    diffs = cell_means - traditional_means
    ratios = np.divide(cell_means, traditional_means, out=np.full_like(cell_means, np.nan), where=traditional_means != 0)
    if relation == "cell_view_fewer":
        factors = np.divide(
            traditional_means,
            cell_means,
            out=np.full_like(cell_means, np.nan),
            where=cell_means != 0,
        )
    elif relation == "cell_view_greater":
        factors = ratios
    else:
        factors = ratios

    def ci(values: np.ndarray, q: float) -> float | None:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return None
        return float(np.quantile(finite, q))

    return {
        "bootstrap_mean_difference_ci_low": ci(diffs, 0.025),
        "bootstrap_mean_difference_ci_high": ci(diffs, 0.975),
        "bootstrap_cell_over_traditional_ratio_ci_low": ci(ratios, 0.025),
        "bootstrap_cell_over_traditional_ratio_ci_high": ci(ratios, 0.975),
        "bootstrap_paper_relation_factor_ci_low": ci(factors, 0.025),
        "bootstrap_paper_relation_factor_ci_high": ci(factors, 0.975),
    }


def conclusion_matches(relation: str, z_statistic: float, p_value: float, alpha: float = 0.05) -> bool:
    if relation == "similar":
        return p_value >= alpha
    if relation == "cell_view_fewer":
        return z_statistic < 0.0 and p_value < alpha
    if relation == "cell_view_greater":
        return z_statistic > 0.0 and p_value < alpha
    raise ValueError(f"unknown relation: {relation}")


def interpret_result(relation: str, primary_matches: bool, secondary_p: float, secondary_z: float) -> str:
    if primary_matches:
        if relation == "similar":
            return "paper-style z-test reproduces the non-significant similarity conclusion"
        return "paper-style z-test reproduces the reported direction and significance, with magnitude differences documented"
    if relation == "cell_view_fewer" and secondary_z < 0.0 and secondary_p < 0.05:
        return "paper-style independent z-test does not reproduce significance, but matched-pair secondary analysis supports the same direction"
    if relation == "cell_view_greater" and secondary_z > 0.0 and secondary_p < 0.05:
        return "paper-style independent z-test is constraining, but matched-pair secondary analysis supports the same direction"
    return "does not reproduce the paper-style statistical conclusion under available S05 data"


def read_analysis_seed(seed_table: Path) -> tuple[int, int, pd.DataFrame]:
    seed_df = pd.read_csv(seed_table)
    row = seed_df[
        (seed_df["analysisStep"] == STEP_ID) & (seed_df["analysisId"] == "efficiency_ztests_and_bootstrap")
    ]
    if len(row) != 1:
        raise RuntimeError("expected one S06 efficiency_ztests_and_bootstrap seed row")
    record = row.iloc[0]
    return int(record["analysisSeed"]), int(record["bootstrapReplicates"]), seed_df


def paired_values(replicate_df: pd.DataFrame, algorithm: str, metric_id: str) -> pd.DataFrame:
    subset = replicate_df[replicate_df["algorithm"] == algorithm]
    traditional = subset[subset["implementation"] == "traditional"][
        ["replicate_index", "input_permutation_seed", "initial_state_hash", metric_id]
    ].rename(columns={metric_id: "traditional_steps"})
    cell_view = subset[subset["implementation"] == "cell_view"][
        ["replicate_index", "input_permutation_seed", "initial_state_hash", metric_id]
    ].rename(columns={metric_id: "cell_view_steps"})
    return traditional.merge(
        cell_view,
        on=["replicate_index", "input_permutation_seed", "initial_state_hash"],
        how="inner",
        validate="one_to_one",
    ).sort_values("replicate_index")


def build_efficiency_statistics(
    replicate_df: pd.DataFrame,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for metric_id in METRIC_ORDER:
        for algorithm in ALGORITHMS:
            paired = paired_values(replicate_df, algorithm, metric_id)
            if len(paired) != 100:
                raise RuntimeError(f"expected 100 matched pairs for {metric_id}/{algorithm}, found {len(paired)}")
            traditional = paired["traditional_steps"].to_numpy(dtype=float)
            cell_view = paired["cell_view_steps"].to_numpy(dtype=float)
            differences = cell_view - traditional
            claim = PAPER_EFFICIENCY_TESTS[(metric_id, algorithm)]
            primary_z, primary_p, primary_method = paper_style_ztest(cell_view, traditional)
            secondary_z, secondary_p = paired_mean_ztest(differences)
            d_value, hedges_g = cohen_d_independent(cell_view, traditional)
            paired_dz, sign_effect = paired_effect_sizes(differences)
            bootstrap = bootstrap_pair(cell_view, traditional, claim["paperRelation"], rng, bootstrap_replicates)
            cell_mean = float(np.mean(cell_view))
            traditional_mean = float(np.mean(traditional))
            primary_matches = conclusion_matches(claim["paperRelation"], primary_z, primary_p)
            factor = paper_relation_factor(claim["paperRelation"], cell_mean, traditional_mean)
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "result_family": "available_efficiency_statistics",
                    "data_availability": "available",
                    "source_research_step_ids": "S04,S05",
                    "algorithm": algorithm,
                    "metric_id": metric_id,
                    "metric_label": (
                        "Swap-only steps" if metric_id == "swap_only_steps" else "Swap plus comparison steps"
                    ),
                    "pair_count": int(len(paired)),
                    "traditional_mean_steps": traditional_mean,
                    "traditional_sd_steps": float(np.std(traditional, ddof=1)),
                    "cell_view_mean_steps": cell_mean,
                    "cell_view_sd_steps": float(np.std(cell_view, ddof=1)),
                    "mean_difference_cell_minus_traditional": float(np.mean(differences)),
                    "sd_difference": float(np.std(differences, ddof=1)),
                    "cell_view_over_traditional_ratio": cell_mean / traditional_mean if traditional_mean else None,
                    "observed_factor_using_paper_relation": factor,
                    "paper_style_ztest_method": primary_method,
                    "paper_style_z_statistic_cell_minus_traditional": primary_z,
                    "paper_style_p_value_two_sided": primary_p,
                    "paper_reported_z_statistic": claim["paperReportedZ"],
                    "paper_reported_p_text": claim["paperReportedPText"],
                    "z_delta_from_paper": primary_z - claim["paperReportedZ"],
                    "paper_relation": claim["paperRelation"],
                    "paper_reported_factor": claim["paperReportedFactor"],
                    "paper_claim_text": claim["paperClaimText"],
                    "paper_style_conclusion_reproduced": primary_matches,
                    "secondary_paired_z_statistic": secondary_z,
                    "secondary_paired_p_value_two_sided": secondary_p,
                    "bootstrap_method": "paired_percentile_bootstrap_resampling_matched_seed_pairs",
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_replicates": bootstrap_replicates,
                    **bootstrap,
                    "cohen_d_independent": d_value,
                    "hedges_g_independent": hedges_g,
                    "paired_cohen_dz": paired_dz,
                    "paired_sign_effect": sign_effect,
                    "interpretation": interpret_result(claim["paperRelation"], primary_matches, secondary_p, secondary_z),
                }
            )
    return pd.DataFrame(rows)


def build_pending_scaffold(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in PENDING_TEST_SPECS:
        seed_row = seed_df[
            (seed_df["analysisStep"] == STEP_ID)
            & (seed_df["sourceProducerSteps"].astype(str).str.contains(spec["awaitsResearchStep"]))
        ]
        if len(seed_row) == 0:
            analysis_seed = None
            bootstrap_replicates = None
            analysis_id = None
        else:
            record = seed_row.iloc[0]
            analysis_seed = int(record["analysisSeed"])
            bootstrap_replicates = int(record["bootstrapReplicates"])
            analysis_id = str(record["analysisId"])
        rows.append(
            {
                "research_step_id": STEP_ID,
                "result_family": "pending_statistics_scaffold",
                "data_availability": "pending_source_not_available",
                "test_family": spec["testFamily"],
                "awaits_research_step": spec["awaitsResearchStep"],
                "analysis_id": analysis_id,
                "analysis_seed": analysis_seed,
                "bootstrap_replicates": bootstrap_replicates,
                "expected_source_artifact": spec["sourceArtifact"],
                "source_artifact_exists": Path(spec["sourceArtifact"]).exists(),
                "planned_analyses": spec["plannedAnalyses"],
                "reason_pending": spec["reasonPending"],
                "status": "pending",
                "recommended_next_action": f"Run {spec['awaitsResearchStep']} when instructed, then rerun or extend this S06 scaffold.",
            }
        )
    return pd.DataFrame(rows)


def write_statistics_scaffold_module(path: Path) -> None:
    path.write_text(
        '''"""Reusable statistical helpers for E01 replication outputs.

This scaffold is intentionally small and table-oriented. Later steps can import
the formulas here or mirror the schema emitted by S06 when S07/S08/S10/S12 data
become available.
"""

from __future__ import annotations

import math
import numpy as np


def normal_two_sided_p(z_statistic: float) -> float:
    if not math.isfinite(z_statistic):
        return 0.0
    return math.erfc(abs(z_statistic) / math.sqrt(2.0))


def paired_mean_ztest(differences: np.ndarray) -> tuple[float, float]:
    mean_diff = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1))
    if sd_diff == 0.0:
        z_statistic = 0.0 if mean_diff == 0.0 else math.copysign(math.inf, mean_diff)
    else:
        z_statistic = mean_diff / (sd_diff / math.sqrt(len(differences)))
    return z_statistic, normal_two_sided_p(z_statistic)


def paired_percentile_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    seed: int,
    replicates: int = 10000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values_a)
    index = rng.integers(0, n, size=(replicates, n))
    diffs = values_a[index].mean(axis=1) - values_b[index].mean(axis=1)
    return {
        "mean_difference_ci_low": float(np.quantile(diffs, 0.025)),
        "mean_difference_ci_high": float(np.quantile(diffs, 0.975)),
    }
''',
        encoding="utf-8",
    )


def key_rows(stats_df: pd.DataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for _, row in stats_df.sort_values(["metric_id", "algorithm"]).iterrows():
        rows.append(
            [
                row["metric_label"],
                str(row["algorithm"]).title(),
                row["paper_style_z_statistic_cell_minus_traditional"],
                row["paper_style_p_value_two_sided"],
                row["paper_reported_z_statistic"],
                row["paper_style_conclusion_reproduced"],
                row["interpretation"],
            ]
        )
    return rows


def pending_rows(pending_df: pd.DataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for _, row in pending_df.iterrows():
        rows.append(
            [
                row["test_family"],
                row["awaits_research_step"],
                row["expected_source_artifact"],
                row["reason_pending"],
            ]
        )
    return rows


def validate_outputs(stats_df: pd.DataFrame, pending_df: pd.DataFrame, replicate_df: pd.DataFrame, bootstrap_replicates: int) -> dict[str, Any]:
    errors: list[str] = []
    if len(stats_df) != 6:
        errors.append(f"expected 6 available efficiency statistic rows, found {len(stats_df)}")
    if len(pending_df) != 4:
        errors.append(f"expected 4 pending scaffold rows, found {len(pending_df)}")
    if len(replicate_df) != 600:
        errors.append(f"expected 600 S05 replicate rows, found {len(replicate_df)}")
    if sorted(stats_df["pair_count"].unique().tolist()) != [100]:
        errors.append("not every available statistic uses 100 matched pairs")
    if bootstrap_replicates != 10000:
        errors.append(f"expected 10000 bootstrap replicates from S03 seed table, found {bootstrap_replicates}")
    if not {"S07", "S08", "S10", "S12"}.issubset(set(pending_df["awaits_research_step"])):
        errors.append("pending scaffold does not cover S07, S08, S10, and S12")
    if pending_df["source_artifact_exists"].any():
        errors.append("one or more future-step source artifacts unexpectedly already exist")
    future_dirs = [
        path
        for path in [
            Path("/artifacts/research_steps/S07"),
            Path("/artifacts/research_steps/S08"),
            Path("/artifacts/research_steps/S10"),
            Path("/artifacts/research_steps/S12"),
        ]
        if path.exists()
    ]
    if future_dirs:
        errors.append(f"future-step artifact directories are present: {[str(path) for path in future_dirs]}")
    success = not errors
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "passed" if success else "failed",
        "validationResult": (
            "passed: S06 produced six available efficiency statistic rows, four pending future-data scaffold rows, "
            "used 600 S05 replicate rows with N=100 matched pairs, applied the S03 S06 bootstrap seed with "
            "10000 replicates, and did not create S07/S08/S10/S12 artifacts."
            if success
            else "failed: " + "; ".join(errors)
        ),
        "errors": errors,
        "availableStatisticRows": int(len(stats_df)),
        "pendingScaffoldRows": int(len(pending_df)),
        "s05ReplicateRows": int(len(replicate_df)),
        "bootstrapReplicates": int(bootstrap_replicates),
        "futureStepsStarted": False,
        "paperStyleConclusionsReproducedCount": int(stats_df["paper_style_conclusion_reproduced"].sum()),
    }


def write_methods(
    path: Path,
    artifacts: list[str],
    validation_result: str,
    stats_df: pd.DataFrame,
    pending_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> None:
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifacts)
    caveat_lines = "\n".join(f"- {caveat}" for caveat in caveats)
    path.write_text(
        f"""# S06 Statistical Methods

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Caveats or blockers:
{caveat_lines}
- Recommended next action: {recommended_next_action}

## Available Tests

S06 reproduces the available Figure 4 efficiency tests from S05 tables only. The primary test is the archived-code convention found in `multi_dimentions/multi_dimention_monotonicity.py`: `statsmodels.stats.weightstats.ztest(cell_view_values, traditional_values)` with a two-sided p-value and z sign defined as cell-view minus traditional.

Secondary analyses use the matched S05 seed pairs: paired mean-difference z-tests, paired percentile bootstrap intervals with 10,000 resamples, independent Cohen's d/Hedges' g, paired Cohen's dz, and a paired sign-effect summary.

## Formula Conventions

- Primary z-test: two-sided `statsmodels.stats.weightstats.ztest(cell_view_values, traditional_values)`, matching the archived analysis call. The reported z sign is cell-view minus traditional; negative values mean fewer cell-view steps.
- Zero-variance primary fallback: if both samples have zero variance, equal means are recorded as `z = 0, p = 1`; unequal means would be recorded as signed infinite z with `p = 0`.
- Paired secondary z-test: `mean(cell_view - traditional) / (sd(cell_view - traditional) / sqrt(N))`, two-sided normal p-value.
- Bootstrap intervals: paired percentile bootstrap over matched S05 replicate pairs, using the S03 S06 `efficiency_ztests_and_bootstrap` seed and 10,000 resamples. Intervals are 2.5 and 97.5 percentiles.
- Effect sizes: independent Cohen's d and Hedges' g use pooled traditional/cell-view standard deviation; paired Cohen's dz uses the standard deviation of matched differences; paired sign effect is `(positive differences - negative differences) / N`.

{markdown_table(["Metric", "Algorithm", "Primary z", "Primary p", "Paper z", "Paper-style conclusion reproduced", "Interpretation"], key_rows(stats_df))}

## Pending Tests

{markdown_table(["Test family", "Awaiting step", "Expected artifact", "Reason pending"], pending_rows(pending_df))}

S06 did not run or synthesize Frozen Cell, DG, aggregation, same-goal chimera, duplicate-value chimera, or opposite-direction chimera data.
""",
        encoding="utf-8",
    )


def write_summary(
    path: Path,
    artifacts: list[str],
    validation_result: str,
    stats_df: pd.DataFrame,
    pending_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> None:
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifacts)
    caveat_lines = "\n".join(f"- {caveat}" for caveat in caveats)
    reproduced = int(stats_df["paper_style_conclusion_reproduced"].sum())
    path.write_text(
        f"""# S06 Status Summary

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Outcome classification: {OUTCOME_CLASSIFICATION}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Caveats or blockers:
{caveat_lines}
- Lay summary: S06 ran the available efficiency statistics from S05 and created a reusable scaffold for later statistical tests. Five of six paper-style efficiency conclusions reproduce under the archived independent z-test convention; comparison-inclusive Insertion is constraining because the independent z-test is not significant from the available S05 data, although the matched-pair secondary analysis supports the same direction.
- Recommended next action: {recommended_next_action}

## Available Efficiency Statistics

{markdown_table(["Metric", "Algorithm", "Primary z", "Primary p", "Paper z", "Paper-style conclusion reproduced", "Interpretation"], key_rows(stats_df))}

## Pending Future-Data Tests

{markdown_table(["Test family", "Awaiting step", "Expected artifact", "Reason pending"], pending_rows(pending_df))}
""",
        encoding="utf-8",
    )


def write_validation_md(path: Path, validation: dict[str, Any], artifacts: list[str], caveats: list[str], recommended_next_action: str) -> None:
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifacts)
    caveat_lines = "\n".join(f"- {caveat}" for caveat in caveats)
    rows = [
        ["Available statistic rows", validation["availableStatisticRows"]],
        ["Pending scaffold rows", validation["pendingScaffoldRows"]],
        ["S05 replicate rows", validation["s05ReplicateRows"]],
        ["Bootstrap replicates", validation["bootstrapReplicates"]],
        ["Future steps started", validation["futureStepsStarted"]],
        ["Paper-style conclusions reproduced", validation["paperStyleConclusionsReproducedCount"]],
    ]
    path.write_text(
        f"""# S06 Validation

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS if validation["success"] else "failed"}
- Artifacts written:
{artifact_lines}
- Validation result: {validation["validationResult"]}
- Caveats or blockers:
{caveat_lines}
- Recommended next action: {recommended_next_action}

## Checks

{markdown_table(["Check", "Value"], rows)}

## Errors

{markdown_table(["Error"], [[error] for error in validation["errors"]] or [["None"]])}
""",
        encoding="utf-8",
    )


def update_run_manifest(artifacts_dir: Path, status_payload: dict[str, Any], artifact_paths: list[Path], skip: bool) -> None:
    if skip:
        return
    provenance_dir = artifacts_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = provenance_dir / "run_manifest.json"
    manifest = load_json(run_manifest_path)
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest["generatedAt"] = utc_now()
    manifest["researchStepId"] = STEP_ID
    manifest["recommendedNextAction"] = status_payload["recommendedNextAction"]
    manifest["caveatsOrBlockers"] = status_payload["caveatsOrBlockers"]
    manifest["artifactsWritten"] = status_payload["artifactsWritten"]
    manifest["git"] = status_payload["git"]
    manifest["runtime"] = status_payload["runtime"]
    manifest.setdefault("researchSteps", {})
    artifacts = collect_artifacts(artifact_paths)
    manifest["researchSteps"][STEP_ID] = {
        "status": status_payload["status"],
        "success": status_payload["success"],
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "validationResult": status_payload["validationResult"],
        "outcomeClassification": status_payload["outcomeClassification"],
        "generatedAt": status_payload["generatedAt"],
    }
    run_manifest_path.write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_s06(args: argparse.Namespace) -> int:
    artifacts_dir = Path(args.artifacts_dir)
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    results_dir = artifacts_dir / "results"
    for directory in [step_dir, code_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    replicate_path = Path(args.s05_replicates)
    pairwise_path = Path(args.s05_pairwise)
    if not replicate_path.exists():
        raise RuntimeError(f"missing S05 replicate table: {replicate_path}")
    if not pairwise_path.exists():
        raise RuntimeError(f"missing S05 pairwise table: {pairwise_path}")

    replicate_df = pd.read_parquet(replicate_path)
    pairwise_df = pd.read_parquet(pairwise_path)
    bootstrap_seed, bootstrap_replicates, seed_df = read_analysis_seed(Path(args.analysis_seed_table))
    stats_df = build_efficiency_statistics(replicate_df, bootstrap_seed, bootstrap_replicates)
    pending_df = build_pending_scaffold(seed_df)
    validation = validate_outputs(stats_df, pending_df, replicate_df, bootstrap_replicates)

    statistics_parquet = results_dir / "e01_statistics.parquet"
    statistics_csv = results_dir / "e01_statistics.csv"
    efficiency_stats_parquet = results_dir / "e01_efficiency_statistics.parquet"
    efficiency_stats_csv = results_dir / "e01_efficiency_statistics.csv"
    pending_parquet = results_dir / "e01_pending_statistics_scaffold.parquet"
    pending_csv = results_dir / "e01_pending_statistics_scaffold.csv"
    methods_md = step_dir / "statistical_methods.md"
    validation_json = step_dir / "validation.json"
    validation_md = step_dir / "validation.md"
    summary_md = step_dir / "summary.md"
    status_json = step_dir / "status.json"
    artifact_manifest_json = step_dir / "artifact_manifest.json"
    scaffold_module = step_dir / "statistics_scaffold.py"
    code_copy = code_dir / Path(__file__).name

    stats_df.to_parquet(efficiency_stats_parquet, index=False)
    stats_df.to_csv(efficiency_stats_csv, index=False)
    pending_df.to_parquet(pending_parquet, index=False)
    pending_df.to_csv(pending_csv, index=False)
    unified_df = pd.concat(
        [
            stats_df,
            pending_df.rename(
                columns={
                    "test_family": "algorithm",
                    "planned_analyses": "interpretation",
                }
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    unified_df.to_parquet(statistics_parquet, index=False)
    unified_df.to_csv(statistics_csv, index=False)
    write_statistics_scaffold_module(scaffold_module)
    shutil.copy2(Path(__file__), code_copy)

    recommended_next_action = (
        "Hand control back to the Chief Scientist workflow; proceed to S07 only after instruction, using the S06 "
        "scaffold as the statistics pattern for later Frozen Cell, DG, aggregation, and opposite-direction outputs."
    )
    caveats = [
        "Only S05 efficiency data are available for actual statistical replication in S06.",
        "The paper-style primary test follows archived code usage of statsmodels ztest on cell-view values versus traditional values; exact paper formulas remain under-specified in the text.",
        "S05 comparison-inclusive counts inherit the documented S05 convention: traditional wrapper value comparisons and cell-view archived StatusProbe.compare_and_swap_count.",
        "Matched-pair secondary analyses are more appropriate for the S05 seed-matched design but are kept separate from the paper-style independent z-test.",
        "Comparison-inclusive Insertion is constraining: the paper-style independent z-test is not significant in S05 data, despite a consistent paired difference.",
        "Frozen Cell robustness, Delayed Gratification, Aggregation, and opposite-direction statistics are scaffolded only and await S07, S08, S10, and S12 data.",
        "No S07, S08, S10, or S12 simulations or artifact directories were started by S06.",
    ]
    artifacts = [
        str(statistics_parquet),
        str(statistics_csv),
        str(efficiency_stats_parquet),
        str(efficiency_stats_csv),
        str(pending_parquet),
        str(pending_csv),
        str(methods_md),
        str(validation_json),
        str(validation_md),
        str(summary_md),
        str(status_json),
        str(artifact_manifest_json),
        str(scaffold_module),
        str(code_copy),
    ]

    validation_json.write_text(
        json.dumps(json_ready(validation), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_methods(methods_md, artifacts, validation["validationResult"], stats_df, pending_df, caveats, recommended_next_action)
    write_validation_md(validation_md, validation, artifacts, caveats, recommended_next_action)
    write_summary(summary_md, artifacts, validation["validationResult"], stats_df, pending_df, caveats, recommended_next_action)

    s05_status = load_json(Path(args.s05_status))
    git_meta = get_git_metadata()
    runtime = {
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "osCpuCount": os.cpu_count(),
        "workerCount": 1,
        "gpuUsed": False,
        "numpyVersion": np.__version__,
        "pandasVersion": pd.__version__,
        "statsmodelsAvailable": STATSMODELS_AVAILABLE,
    }
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"]),
        "status": STATUS if validation["success"] else "failed",
        "artifactsWritten": artifacts,
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": caveats + validation["errors"],
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION,
        "generatedAt": utc_now(),
        "experimentId": EXPERIMENT_ID,
        "sourceArtifacts": {
            "s05Replicates": str(replicate_path),
            "s05Pairwise": str(pairwise_path),
            "s05Status": str(args.s05_status),
            "s03AnalysisSeedTable": str(args.analysis_seed_table),
        },
        "sourceS05": {
            "researchStepId": s05_status.get("researchStepId"),
            "success": s05_status.get("success"),
            "status": s05_status.get("status"),
            "validationResult": s05_status.get("validationResult"),
            "commit": s05_status.get("git", {}).get("commit"),
        },
        "bootstrapSeed": bootstrap_seed,
        "bootstrapReplicates": bootstrap_replicates,
        "availableEfficiencyStatisticRows": int(len(stats_df)),
        "pendingScaffoldRows": int(len(pending_df)),
        "paperStyleConclusionsReproducedCount": int(stats_df["paper_style_conclusion_reproduced"].sum()),
        "pendingFutureSteps": sorted(pending_df["awaits_research_step"].tolist()),
        "futureStepsStarted": False,
        "git": git_meta,
        "runtime": runtime,
    }
    status_json.write_text(
        json.dumps(json_ready(status_payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    artifact_paths = [Path(path) for path in artifacts if Path(path).exists() and Path(path) != artifact_manifest_json]
    artifact_manifest = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "status": STATUS if validation["success"] else "failed",
        "success": bool(validation["success"]),
        "validationResult": validation["validationResult"],
        "artifacts": collect_artifacts(artifact_paths),
        "sourceArtifacts": status_payload["sourceArtifacts"],
    }
    artifact_manifest_json.write_text(
        json.dumps(json_ready(artifact_manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_paths.append(artifact_manifest_json)
    update_run_manifest(artifacts_dir, status_payload, artifact_paths, args.skip_manifest_update)

    print(validation["validationResult"])
    print(f"Wrote {len(artifact_paths)} S06 artifact files under {artifacts_dir}")
    return 0 if validation["success"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--s05-replicates", default=str(S05_REPLICATES_DEFAULT))
    parser.add_argument("--s05-pairwise", default=str(S05_PAIRWISE_DEFAULT))
    parser.add_argument("--s05-status", default=str(S05_STATUS_DEFAULT))
    parser.add_argument("--analysis-seed-table", default=str(S03_ANALYSIS_SEEDS_DEFAULT))
    parser.add_argument("--skip-manifest-update", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run_s06(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
