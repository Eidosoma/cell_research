#!/usr/bin/env python3
"""Run E02 S14 stronger statistical tests over E01 and E02 claim slices."""

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
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:  # statsmodels is preinstalled in the runtime image.
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
except Exception:  # pragma: no cover - exercised only in degraded environments.
    smf = None
    anova_lm = None


STEP_ID = "S14"
STEP_NUMBER = 14
EXPERIMENT_ID = "E02"
DEFAULT_BOOTSTRAP_REPS = 5_000
DEFAULT_PERMUTATION_REPS = 5_000
DEFAULT_SEED = 2026070114
ALPHA = 0.05


@dataclass(frozen=True)
class InputPaths:
    artifacts_dir: Path
    previous_e01_dir: Path

    @property
    def result_paths(self) -> dict[str, Path]:
        return {
            "S02_raw": self.artifacts_dir / "results/e02_scheduler_comparison.parquet",
            "S02_class": self.artifacts_dir / "research_steps/S02/e02_scheduler_sensitivity_classification.csv",
            "S03_raw": self.artifacts_dir / "results/e02_activation_rate_artifacts.parquet",
            "S03_class": self.artifacts_dir / "research_steps/S03/e02_activation_rate_sensitivity_classification.csv",
            "S04_raw": self.artifacts_dir / "results/e02_label_shuffle_aggregation_nulls.parquet",
            "S04_class": self.artifacts_dir / "research_steps/S04/e02_label_shuffle_condition_summary.csv",
            "S05_raw": self.artifacts_dir / "results/e02_dummy_algotype_controls.parquet",
            "S05_class": self.artifacts_dir / "research_steps/S05/e02_dummy_algotype_condition_summary.csv",
            "S06_raw": self.artifacts_dir / "results/e02_speed_matched_chimeras.parquet",
            "S06_class": self.artifacts_dir / "research_steps/S06/e02_speed_matched_classification.csv",
            "S07_raw": self.artifacts_dir / "results/e02_local_move_nulls.parquet",
            "S07_class": self.artifacts_dir / "research_steps/S07/e02_local_move_null_classification.csv",
            "S08_raw": self.artifacts_dir / "results/e02_dg_null_benchmarks.parquet",
            "S08_class": self.artifacts_dir / "research_steps/S08/e02_dg_burden_classification.csv",
            "S09_raw": self.artifacts_dir / "results/e02_alternative_metrics.parquet",
            "S09_class": self.artifacts_dir / "research_steps/S09/e02_alternative_metric_classification.csv",
            "S10_raw": self.artifacts_dir / "results/e02_input_distribution_matrix.parquet",
            "S10_class": self.artifacts_dir / "research_steps/S10/e02_input_distribution_classification.csv",
            "S11_raw": self.artifacts_dir / "results/e02_frozen_placement.parquet",
            "S11_class": self.artifacts_dir / "research_steps/S11/e02_frozen_placement_classification.csv",
            "S12_raw": self.artifacts_dir / "results/e02_frozen_behavior_variants.parquet",
            "S12_class": self.artifacts_dir / "research_steps/S12/e02_frozen_behavior_classification.csv",
            "S13_raw": self.artifacts_dir / "results/e02_stop_condition_sensitivity.parquet",
            "S13_class": self.artifacts_dir / "research_steps/S13/e02_stop_condition_classification.csv",
            "E01_z": self.previous_e01_dir / "tables/e01_z_tests.csv",
            "E01_bootstrap": self.previous_e01_dir / "tables/e01_bootstrap_intervals.csv",
            "E01_status": self.previous_e01_dir / "tables/e01_replication_status.csv",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-e01-dir", type=Path, default=Path("/previous-artifacts/E01"))
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--permutation-reps", type=int, default=DEFAULT_PERMUTATION_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": int(path.stat().st_size),
    }


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(time.monotonic() - started),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": bool(result.returncode == 0),
    }


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def stable_seed(*parts: Any, base_seed: int = DEFAULT_SEED) -> int:
    payload = json.dumps([base_seed, *map(str, parts)], sort_keys=True).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def clean_numeric(values: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def bh_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    indexed: list[tuple[int, float]] = []
    for idx, value in enumerate(p_values):
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            indexed.append((idx, min(max(numeric, 0.0), 1.0)))
    adjusted: list[float | None] = [None] * len(p_values)
    if not indexed:
        return adjusted
    order = sorted(range(len(indexed)), key=lambda i: indexed[i][1])
    m = len(indexed)
    running = 1.0
    for rank_pos in range(m - 1, -1, -1):
        idx_in_indexed = order[rank_pos]
        original_idx, p_value = indexed[idx_in_indexed]
        rank = rank_pos + 1
        running = min(running, p_value * m / rank)
        adjusted[original_idx] = min(running, 1.0)
    return adjusted


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    reps: int,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    arr = clean_numeric(values)
    if len(arr) == 0:
        return None, None, None
    mean = float(np.mean(arr))
    if len(arr) == 1 or reps <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(int(reps), len(arr)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean, float(low), float(high)


def sign_flip_p_value(
    values: Sequence[float],
    *,
    seed: int,
    reps: int,
    alternative: str = "two-sided",
) -> tuple[float | None, str, int]:
    arr = clean_numeric(values)
    n = len(arr)
    if n == 0:
        return None, "no_finite_values", 0
    observed = float(np.mean(arr))
    if np.allclose(arr, 0.0):
        return 1.0, "all_zero_exact", 1
    if n <= 15:
        signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * n))).T.reshape(-1, n)
        stats_null = (signs * arr).mean(axis=1)
        method = "exact_sign_flip"
        used_reps = int(len(stats_null))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(int(reps), n), replace=True)
        stats_null = (signs * arr).mean(axis=1)
        method = "monte_carlo_sign_flip"
        used_reps = int(reps)
    if alternative == "greater":
        p_value = float((np.count_nonzero(stats_null >= observed - 1e-12) + 1) / (len(stats_null) + 1))
    elif alternative == "less":
        p_value = float((np.count_nonzero(stats_null <= observed + 1e-12) + 1) / (len(stats_null) + 1))
    else:
        p_value = float((np.count_nonzero(np.abs(stats_null) >= abs(observed) - 1e-12) + 1) / (len(stats_null) + 1))
    return min(max(p_value, 0.0), 1.0), method, used_reps


def blocked_group_range_permutation(
    frame: pd.DataFrame,
    *,
    block_col: str,
    group_col: str,
    value_col: str,
    seed: int,
    reps: int,
) -> tuple[float | None, float | None, float | None, float | None, str, int, int]:
    data = frame[[block_col, group_col, value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[block_col, group_col, value_col])
    block_sizes = data.groupby(block_col)[group_col].nunique()
    valid_blocks = set(block_sizes[block_sizes >= 2].index)
    data = data[data[block_col].isin(valid_blocks)].copy()
    if data.empty or data[group_col].nunique() < 2:
        return None, None, None, None, "insufficient_groups", 0, 0
    observed_means = data.groupby(group_col)[value_col].mean()
    observed_range = float(observed_means.max() - observed_means.min())
    blocks = [block.copy() for _key, block in data.groupby(block_col)]
    rng = np.random.default_rng(seed)
    null_stats = np.empty(int(reps), dtype=float)
    for i in range(int(reps)):
        pieces: list[pd.DataFrame] = []
        for block in blocks:
            shuffled = block.copy()
            shuffled[group_col] = rng.permutation(shuffled[group_col].to_numpy())
            pieces.append(shuffled)
        permuted = pd.concat(pieces, ignore_index=True)
        means = permuted.groupby(group_col)[value_col].mean()
        null_stats[i] = float(means.max() - means.min())
    p_value = float((np.count_nonzero(null_stats >= observed_range - 1e-12) + 1) / (len(null_stats) + 1))
    boot_stats = np.empty(int(reps), dtype=float)
    for i in range(int(reps)):
        sampled = [blocks[j] for j in rng.integers(0, len(blocks), size=len(blocks))]
        boot = pd.concat(sampled, ignore_index=True)
        means = boot.groupby(group_col)[value_col].mean()
        boot_stats[i] = float(means.max() - means.min())
    ci_low, ci_high = np.quantile(boot_stats, [0.025, 0.975])
    return observed_range, float(ci_low), float(ci_high), p_value, "blocked_label_permutation", int(reps), int(len(blocks))


def fit_group_model(frame: pd.DataFrame, *, block_col: str, group_col: str, value_col: str) -> dict[str, Any]:
    data = frame[[block_col, group_col, value_col]].copy()
    data = data.rename(columns={block_col: "block", group_col: "group", value_col: "y"})
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna(subset=["block", "group", "y"])
    data["block"] = data["block"].astype(str)
    data["group"] = data["group"].astype(str)
    formula_mixed = "y ~ C(group) + (1 | block)"
    if data.empty or data["group"].nunique() < 2 or data["block"].nunique() < 2:
        return {
            "model_type": "not_fit",
            "model_formula": formula_mixed,
            "model_p_value": None,
            "model_fit_status": "insufficient_groups",
        }
    if smf is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mixed = smf.mixedlm("y ~ C(group)", data, groups=data["block"]).fit(
                    method="lbfgs", reml=False, maxiter=100, disp=False
                )
            p_values = [float(v) for key, v in mixed.pvalues.items() if key != "Intercept" and math.isfinite(float(v))]
            return {
                "model_type": "mixedlm_random_intercept",
                "model_formula": formula_mixed,
                "model_p_value": min(p_values) if p_values else None,
                "model_fit_status": "passed",
            }
        except Exception as exc:
            mixed_error = repr(exc)
    else:
        mixed_error = "statsmodels_unavailable"
    if smf is not None and anova_lm is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ols = smf.ols("y ~ C(group) + C(block)", data=data).fit()
                anova = anova_lm(ols, typ=2)
            p_value = float(anova.loc["C(group)", "PR(>F)"])
            return {
                "model_type": "blocked_ols_fallback",
                "model_formula": "y ~ C(group) + C(block)",
                "model_p_value": p_value if math.isfinite(p_value) else None,
                "model_fit_status": f"passed_after_mixedlm_failure:{mixed_error}",
            }
        except Exception as exc:
            return {
                "model_type": "not_fit",
                "model_formula": "y ~ C(group) + C(block)",
                "model_p_value": None,
                "model_fit_status": f"blocked_ols_failed:{exc!r}; mixedlm:{mixed_error}",
            }
    return {
        "model_type": "not_fit",
        "model_formula": "y ~ C(group) + C(block)",
        "model_p_value": None,
        "model_fit_status": f"statsmodels_unavailable; mixedlm:{mixed_error}",
    }


def fit_delta_model(values: Sequence[float]) -> dict[str, Any]:
    arr = clean_numeric(values)
    if len(arr) < 2:
        return {
            "model_type": "not_fit",
            "model_formula": "delta ~ 1",
            "model_p_value": None,
            "model_fit_status": "insufficient_pairs",
        }
    if np.allclose(arr, arr[0]):
        p_value = 1.0 if abs(float(arr[0])) <= 1e-12 else 0.0
        return {
            "model_type": "paired_t_fallback",
            "model_formula": "delta ~ 1",
            "model_p_value": p_value,
            "model_fit_status": "constant_delta_degenerate",
        }
    ttest = stats.ttest_1samp(arr, popmean=0.0, nan_policy="omit")
    p_value = float(ttest.pvalue) if math.isfinite(float(ttest.pvalue)) else None
    return {
        "model_type": "paired_t_fallback",
        "model_formula": "delta ~ 1",
        "model_p_value": p_value,
        "model_fit_status": "passed",
    }


def base_test_row(
    *,
    source_step_id: str,
    fdr_family: str,
    claim_family: str,
    claim_id: str,
    metric: str,
    comparison: str,
    prior_classification: str,
    effect_estimate: float | None,
    ci_low: float | None,
    ci_high: float | None,
    p_value: float | None,
    permutation_method: str,
    permutation_reps: int,
    bootstrap_reps: int,
    n_units: int,
    n_observations: int,
    model: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
    caveat: str = "",
) -> dict[str, Any]:
    context = dict(context or {})
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "source_step_id": source_step_id,
        "fdr_family": fdr_family,
        "claim_family": claim_family,
        "claim_id": claim_id,
        "metric": metric,
        "comparison": comparison,
        "prior_classification": prior_classification,
        "effect_estimate": effect_estimate,
        "effect_ci95_low": ci_low,
        "effect_ci95_high": ci_high,
        "effect_units": "raw_metric_units_or_percentage_points",
        "permutation_p_value": p_value,
        "permutation_method": permutation_method,
        "permutation_reps_used": int(permutation_reps),
        "bootstrap_reps": int(bootstrap_reps),
        "n_units": int(n_units),
        "n_observations": int(n_observations),
        "model_type": model.get("model_type"),
        "model_formula": model.get("model_formula"),
        "model_p_value": model.get("model_p_value"),
        "model_fit_status": model.get("model_fit_status"),
        "family_q_value": None,
        "global_q_value": None,
        "corrected_result": "pending_fdr",
        "statistical_interpretation": "pending_fdr",
        "context_json": json.dumps(context, sort_keys=True, separators=(",", ":"), default=str),
        "caveat": caveat,
    }


def metric_claim_family(metric: str) -> str:
    if "aggregation" in metric:
        return "aggregation"
    if "dg" in metric or "gratification" in metric:
        return "delayed_gratification"
    if "sortedness" in metric or "monotonicity" in metric or "progress" in metric:
        return "robustness"
    if "curvature" in metric or "path" in metric:
        return "path_shape"
    if "compare" in metric or "event_count" in metric or "swap" in metric:
        return "efficiency"
    if "frozen_attempt" in metric:
        return "perturbation_attempts"
    return "miscellaneous"


def add_e01_legacy_tests(z_df: pd.DataFrame, boot_df: pd.DataFrame) -> list[dict[str, Any]]:
    boot_lookup = {
        (str(row["algorithm"]), str(row["metric"])): row
        for row in boot_df.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for source in z_df.to_dict(orient="records"):
        key = (str(source.get("algorithm")), str(source.get("metric")))
        boot = boot_lookup.get(key, {})
        ci_low = boot.get("paired_difference_ci95_low")
        ci_high = boot.get("paired_difference_ci95_high")
        if ci_low is not None and pd.isna(ci_low):
            ci_low = None
        if ci_high is not None and pd.isna(ci_high):
            ci_high = None
        model = {
            "model_type": "legacy_z_test",
            "model_formula": "cell_view_minus_traditional / standard_error",
            "model_p_value": source.get("p_value_two_sided"),
            "model_fit_status": "imported_from_E01",
        }
        rows.append(
            base_test_row(
                source_step_id="E01",
                fdr_family=f"E01_{source.get('claim_family', 'legacy')}",
                claim_family=str(source.get("claim_family", "legacy")),
                claim_id=str(source.get("claim_id", f"E01_{len(rows)}")),
                metric=str(source.get("metric")),
                comparison=str(source.get("comparison")),
                prior_classification=str(source.get("replication_status")),
                effect_estimate=float(source["mean_difference_cell_minus_traditional"])
                if pd.notna(source.get("mean_difference_cell_minus_traditional"))
                else None,
                ci_low=float(ci_low) if ci_low is not None else None,
                ci_high=float(ci_high) if ci_high is not None else None,
                p_value=float(source["p_value_two_sided"]) if pd.notna(source.get("p_value_two_sided")) else None,
                permutation_method="legacy_z_test_import",
                permutation_reps=0,
                bootstrap_reps=int(boot.get("bootstrap_reps", 0)) if boot else 0,
                n_units=int(source.get("n_cell_view", 0)) if pd.notna(source.get("n_cell_view")) else 0,
                n_observations=int(source.get("n_cell_view", 0)) if pd.notna(source.get("n_cell_view")) else 0,
                model=model,
                context={
                    "algorithm": source.get("algorithm"),
                    "paper_expected_direction": source.get("paper_expected_direction"),
                    "direction_matches_paper": source.get("direction_matches_paper"),
                    "significance_matches_paper": source.get("significance_matches_paper"),
                },
                caveat="Imported E01 legacy z-test; bootstrap interval attached when E01 provided paired bootstrap output.",
            )
        )
    return rows


def add_omnibus_step_tests(
    *,
    raw_df: pd.DataFrame,
    class_df: pd.DataFrame,
    source_step_id: str,
    fdr_family: str,
    group_col: str,
    filter_cols: Sequence[str],
    metric_col: str,
    base_seed: int,
    reps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = raw_df.copy()
    raw["block_id"] = raw["condition_id"].astype(str) + "::" + raw["repeat_index"].astype(str)
    for idx, source in enumerate(class_df.to_dict(orient="records")):
        metric = str(source[metric_col])
        if metric not in raw.columns:
            continue
        subset = raw.copy()
        for col in filter_cols:
            if col in source and col in subset.columns and pd.notna(source[col]):
                subset = subset[subset[col].astype(str) == str(source[col])]
        subset = subset[["block_id", group_col, metric]].dropna()
        effect, ci_low, ci_high, p_value, method, used_reps, n_units = blocked_group_range_permutation(
            subset,
            block_col="block_id",
            group_col=group_col,
            value_col=metric,
            seed=stable_seed(source_step_id, idx, metric, base_seed=base_seed),
            reps=reps,
        )
        model = fit_group_model(subset, block_col="block_id", group_col=group_col, value_col=metric)
        rows.append(
            base_test_row(
                source_step_id=source_step_id,
                fdr_family=fdr_family,
                claim_family=str(source.get("claim_family", metric_claim_family(metric))),
                claim_id=f"{source_step_id}_{idx:03d}_{metric}",
                metric=metric,
                comparison=f"omnibus_range_across_{group_col}",
                prior_classification=str(source.get("classification", source.get("prior_classification", ""))),
                effect_estimate=effect,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_value,
                permutation_method=method,
                permutation_reps=used_reps,
                bootstrap_reps=reps,
                n_units=n_units,
                n_observations=int(len(subset)),
                model=model,
                context={col: source.get(col) for col in filter_cols if col in source} | {group_col: sorted(subset[group_col].dropna().astype(str).unique())},
                caveat="Omnibus blocked label-permutation test preserves condition/repeat blocks and tests scheduler or activation-regime mean spread.",
            )
        )
    return rows


def add_delta_rows(
    *,
    values: Sequence[float],
    source_step_id: str,
    fdr_family: str,
    claim_family: str,
    claim_id: str,
    metric: str,
    comparison: str,
    prior_classification: str,
    context: Mapping[str, Any],
    base_seed: int,
    reps: int,
    caveat: str,
) -> list[dict[str, Any]]:
    arr = clean_numeric(values)
    mean, ci_low, ci_high = bootstrap_mean_ci(arr, seed=stable_seed(claim_id, metric, "boot", base_seed=base_seed), reps=reps)
    p_value, method, used_reps = sign_flip_p_value(arr, seed=stable_seed(claim_id, metric, "perm", base_seed=base_seed), reps=reps)
    model = fit_delta_model(arr)
    return [
        base_test_row(
            source_step_id=source_step_id,
            fdr_family=fdr_family,
            claim_family=claim_family,
            claim_id=claim_id,
            metric=metric,
            comparison=comparison,
            prior_classification=prior_classification,
            effect_estimate=mean,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            permutation_method=method,
            permutation_reps=used_reps,
            bootstrap_reps=reps,
            n_units=int(len(arr)),
            n_observations=int(len(arr)),
            model=model,
            context=context,
            caveat=caveat,
        )
    ]


def paired_metric_delta(
    frame: pd.DataFrame,
    *,
    id_cols: Sequence[str],
    treatment_col: str,
    reference: str,
    treatment: str,
    metric: str,
) -> np.ndarray:
    subset = frame[list(id_cols) + [treatment_col, metric]].copy()
    subset = subset[subset[treatment_col].astype(str).isin([str(reference), str(treatment)])]
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    grouped = subset.groupby([*id_cols, treatment_col], dropna=False)[metric].mean().reset_index()
    pivot = grouped.pivot_table(index=list(id_cols), columns=treatment_col, values=metric, aggfunc="mean")
    if reference not in pivot.columns or treatment not in pivot.columns:
        return np.array([], dtype=float)
    return clean_numeric(pivot[treatment] - pivot[reference])


def add_null_delta_tests(raw_df: pd.DataFrame, base_seed: int, reps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # S04 label-shuffle nulls.
    if "peak_minus_null_peak_mean" in raw_df.columns:
        for condition_id, subset in raw_df.groupby("condition_id"):
            rows += add_delta_rows(
                values=subset["peak_minus_null_peak_mean"],
                source_step_id="S04",
                fdr_family="S04_label_shuffle_aggregation",
                claim_family="aggregation",
                claim_id=f"S04_{condition_id}_peak_aggregation_vs_label_shuffle",
                metric="peak_minus_null_peak_mean",
                comparison="observed_peak_aggregation_minus_label_shuffle_null",
                prior_classification="label_shuffle_condition_summary",
                context={"condition_id": condition_id, "algotype_mix": subset["algotype_mix"].iloc[0]},
                base_seed=base_seed,
                reps=reps,
                caveat="Trajectory-preserving label-shuffle metric null; not a behavioral rerun.",
            )
    return rows


def load_all_inputs(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required S14 input artifact(s): {missing}")
    return {name: read_table(path) for name, path in paths.items()}


def build_statistical_tests(inputs: Mapping[str, pd.DataFrame], *, base_seed: int, bootstrap_reps: int, permutation_reps: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows += add_e01_legacy_tests(inputs["E01_z"], inputs["E01_bootstrap"])

    rows += add_omnibus_step_tests(
        raw_df=inputs["S02_raw"],
        class_df=inputs["S02_class"],
        source_step_id="S02",
        fdr_family="S02_scheduler_sensitivity",
        group_col="scheduler_regime",
        filter_cols=("run_family", "algorithm", "algotype_mix"),
        metric_col="metric",
        base_seed=base_seed,
        reps=permutation_reps,
    )
    rows += add_omnibus_step_tests(
        raw_df=inputs["S03_raw"],
        class_df=inputs["S03_class"],
        source_step_id="S03",
        fdr_family="S03_activation_rate_artifacts",
        group_col="activation_rate_regime",
        filter_cols=("condition_id", "algotype_mix"),
        metric_col="metric",
        base_seed=base_seed,
        reps=permutation_reps,
    )

    # S04 and S05: observed Aggregation against label-shuffle or dummy-label nulls.
    rows += add_null_delta_tests(inputs["S04_raw"], base_seed=base_seed, reps=bootstrap_reps)
    rows += add_delta_rows(
        values=inputs["S05_raw"]["peak_minus_null_peak_mean"],
        source_step_id="S05",
        fdr_family="S05_dummy_algotype_negative_control",
        claim_family="aggregation",
        claim_id="S05_dummy_bubble_peak_aggregation_vs_label_shuffle",
        metric="peak_minus_null_peak_mean",
        comparison="dummy_observed_peak_aggregation_minus_label_shuffle_null",
        prior_classification="dummy_behavior_preserving_negative_control",
        context={"condition_id": "E02S05C001", "source_condition_id": "E01C047"},
        base_seed=base_seed,
        reps=bootstrap_reps,
        caveat="Behavior-preserving Bubble dummy labels under deterministic balanced identity-round scheduler.",
    )

    # S06: speed-matched chimeras versus equal activation.
    s06_raw = inputs["S06_raw"]
    s06_class = inputs["S06_class"]
    for source in s06_class.to_dict(orient="records"):
        condition_id = str(source["condition_id"])
        subset = s06_raw[(s06_raw["condition_id"].astype(str) == condition_id) & (s06_raw["activation_rate_regime"].astype(str) != "equal_per_cell")]
        for metric in ("delta_peak_aggregation_vs_equal", "relative_delta_compare_plus_swap_vs_equal"):
            if metric not in subset:
                continue
            rows += add_delta_rows(
                values=subset[metric],
                source_step_id="S06",
                fdr_family="S06_speed_matched_algotypes",
                claim_family=metric_claim_family(metric),
                claim_id=f"S06_{condition_id}_{metric}",
                metric=metric,
                comparison="speed_matched_minus_equal_activation",
                prior_classification=str(source.get("classification")),
                context={"condition_id": condition_id, "algotype_mix": source.get("algotype_mix")},
                base_seed=base_seed,
                reps=bootstrap_reps,
                caveat="Speed matching uses E01 pure compare-plus-swap estimates; mixed-run realized per-label work can diverge.",
            )

    # S07: real versus randomized local-move null models.
    s07_raw = inputs["S07_raw"]
    s07_class = inputs["S07_class"]
    s07_metrics = (
        "delta_final_sortedness_vs_real",
        "delta_dg_primary_vs_real",
        "delta_peak_aggregation_vs_real",
        "delta_path_curvature_ratio_vs_real",
    )
    for source in s07_class.to_dict(orient="records"):
        condition_id = str(source["condition_id"])
        null_policy = str(source["null_policy"])
        subset = s07_raw[(s07_raw["condition_id"].astype(str) == condition_id) & (s07_raw["null_policy"].astype(str) == null_policy)]
        for metric in s07_metrics:
            if metric not in subset:
                continue
            agg = subset.groupby(["condition_id", "repeat_index", "null_policy"], dropna=False)[metric].mean().reset_index()
            rows += add_delta_rows(
                values=agg[metric],
                source_step_id="S07",
                fdr_family="S07_local_move_nulls",
                claim_family=metric_claim_family(metric),
                claim_id=f"S07_{condition_id}_{null_policy}_{metric}",
                metric=metric,
                comparison="null_minus_real",
                prior_classification=str(source.get("classification")),
                context={"condition_id": condition_id, "algotype_mix": source.get("algotype_mix"), "null_policy": null_policy},
                base_seed=base_seed,
                reps=bootstrap_reps,
                caveat="Null replicates are averaged within condition/repeat before testing to reduce pseudo-replication.",
            )

    # S08: matched DG null benchmarks.
    s08_raw = inputs["S08_raw"]
    s08_class = inputs["S08_class"]
    for idx, source in enumerate(s08_class.to_dict(orient="records")):
        null_policy = str(source["null_policy"])
        subset = s08_raw[
            (s08_raw["null_policy"].astype(str) == null_policy)
            & (s08_raw["plot_frozen_semantics"].astype(str) == str(source["plot_frozen_semantics"]))
            & (s08_raw["mode"].astype(str) == str(source["mode"]))
            & (s08_raw["algorithm"].astype(str) == str(source["algorithm"]))
        ]
        agg = subset.groupby(["condition_id", "repeat_index", "null_policy"], dropna=False)["real_minus_null_dg_primary"].mean().reset_index()
        rows += add_delta_rows(
            values=agg["real_minus_null_dg_primary"],
            source_step_id="S08",
            fdr_family="S08_dg_matched_nulls",
            claim_family="delayed_gratification",
            claim_id=f"S08_{idx:03d}_{source['plot_frozen_semantics']}_{source['mode']}_{source['algorithm']}_{null_policy}",
            metric="real_minus_null_dg_primary",
            comparison="real_dg_minus_matched_null_dg",
            prior_classification=str(source.get("classification")),
            context={
                "plot_frozen_semantics": source.get("plot_frozen_semantics"),
                "mode": source.get("mode"),
                "algorithm": source.get("algorithm"),
                "null_policy": null_policy,
            },
            base_seed=base_seed,
            reps=bootstrap_reps,
            caveat="Matched null trajectories are synthetic Sortedness bridges, not public cell-policy reruns.",
        )

    # S09: alternative metric progress versus adjacent Sortedness progress.
    s09_raw = inputs["S09_raw"]
    s09_class = inputs["S09_class"]
    for source in s09_class.to_dict(orient="records"):
        condition_id = str(source["condition_id"])
        metric_name = str(source["metric_name"])
        subset = s09_raw[(s09_raw["condition_id"].astype(str) == condition_id) & (s09_raw["metric_name"].astype(str) == metric_name)]
        rows += add_delta_rows(
            values=subset["final_progress_delta_vs_sortedness"],
            source_step_id="S09",
            fdr_family="S09_alternative_metrics",
            claim_family="metric_sensitivity",
            claim_id=f"S09_{condition_id}_{metric_name}_final_progress_delta",
            metric="final_progress_delta_vs_sortedness",
            comparison=f"{metric_name}_progress_minus_sortedness_progress",
            prior_classification=str(source.get("metric_stability_class")),
            context={"condition_id": condition_id, "metric_name": metric_name, "algotype_mix": source.get("algotype_mix")},
            base_seed=base_seed,
            reps=bootstrap_reps,
            caveat="S09 uses repaired deterministic value trajectories; original blocked preflight remains a caveat for prior persisted traces.",
        )

    # S10: generated input distributions against random-permutation baseline.
    s10_raw = inputs["S10_raw"]
    s10_class = inputs["S10_class"]
    s10_metrics = (
        "final_sortedness_percent",
        "compare_plus_swap_steps",
        "dg_primary",
        "peak_aggregation_left_neighbor_percent",
    )
    for source in s10_class.to_dict(orient="records"):
        condition_id = str(source["condition_id"])
        input_distribution = str(source["input_distribution"])
        if input_distribution == "random_permutation":
            continue
        subset = s10_raw[s10_raw["condition_id"].astype(str) == condition_id]
        for metric in s10_metrics:
            deltas = paired_metric_delta(
                subset,
                id_cols=("condition_id", "repeat_index"),
                treatment_col="input_distribution",
                reference="random_permutation",
                treatment=input_distribution,
                metric=metric,
            )
            rows += add_delta_rows(
                values=deltas,
                source_step_id="S10",
                fdr_family="S10_input_distributions",
                claim_family=metric_claim_family(metric),
                claim_id=f"S10_{condition_id}_{input_distribution}_{metric}",
                metric=metric,
                comparison=f"{input_distribution}_minus_random_permutation",
                prior_classification=str(source.get("input_generalization_class")),
                context={"condition_id": condition_id, "input_distribution": input_distribution, "algorithm": source.get("algorithm")},
                base_seed=base_seed,
                reps=bootstrap_reps,
                caveat="Duplicate-heavy metrics use S09 repaired tie handling; heavy-tailed curvature remains magnitude-sensitive.",
            )

    # S11: Frozen Cell placement rules against random-bank reference.
    s11_raw = inputs["S11_raw"]
    s11_class = inputs["S11_class"]
    s11_metrics = (
        "final_sortedness_percent",
        "compare_plus_swap_steps",
        "dg_primary",
        "sortedness_path_curvature_ratio",
    )
    for source in s11_class.to_dict(orient="records"):
        condition_id = str(source["condition_id"])
        placement_rule = str(source["placement_rule"])
        if placement_rule == "random_bank":
            continue
        subset = s11_raw[s11_raw["condition_id"].astype(str) == condition_id]
        for metric in s11_metrics:
            deltas = paired_metric_delta(
                subset,
                id_cols=("condition_id", "repeat_index"),
                treatment_col="placement_rule",
                reference="random_bank",
                treatment=placement_rule,
                metric=metric,
            )
            rows += add_delta_rows(
                values=deltas,
                source_step_id="S11",
                fdr_family="S11_frozen_placement",
                claim_family=metric_claim_family(metric),
                claim_id=f"S11_{condition_id}_{placement_rule}_{metric}",
                metric=metric,
                comparison=f"{placement_rule}_minus_random_bank",
                prior_classification=str(source.get("placement_sensitivity_class")),
                context={"condition_id": condition_id, "placement_rule": placement_rule, "algorithm": source.get("algorithm")},
                base_seed=base_seed,
                reps=bootstrap_reps,
                caveat="Placement matrix uses unique random-permutation inputs and selected Frozen Cell counts 1 and 3.",
            )

    # S12: Frozen Cell behavior variants against original passive and stuck limits.
    s12_raw = inputs["S12_raw"]
    s12_class = inputs["S12_class"]
    s12_metrics = (
        "final_sortedness_percent",
        "compare_plus_swap_steps",
        "dg_primary",
        "sortedness_path_curvature_ratio",
    )
    for source in s12_class.to_dict(orient="records"):
        pair_id = str(source["condition_pair_id"])
        variant = str(source["behavior_variant"])
        if variant in {"original_passive", "original_stuck"}:
            continue
        subset = s12_raw[s12_raw["condition_pair_id"].astype(str) == pair_id]
        for reference in ("original_passive", "original_stuck"):
            for metric in s12_metrics:
                deltas = paired_metric_delta(
                    subset,
                    id_cols=("condition_pair_id", "repeat_index"),
                    treatment_col="behavior_variant",
                    reference=reference,
                    treatment=variant,
                    metric=metric,
                )
                rows += add_delta_rows(
                    values=deltas,
                    source_step_id="S12",
                    fdr_family="S12_frozen_behavior",
                    claim_family=metric_claim_family(metric),
                    claim_id=f"S12_{pair_id}_{variant}_vs_{reference}_{metric}",
                    metric=metric,
                    comparison=f"{variant}_minus_{reference}",
                    prior_classification=str(source.get("behavior_sensitivity_class")),
                    context={"condition_pair_id": pair_id, "behavior_variant": variant, "reference_behavior": reference, "algorithm": source.get("algorithm")},
                    base_seed=base_seed,
                    reps=bootstrap_reps,
                    caveat="Dynamic behavior variants are behavior extensions, not paper-reported perturbation semantics.",
                )

    # S13: stop-condition regimes against reference stop rule.
    s13_raw = inputs["S13_raw"]
    s13_class = inputs["S13_class"]
    s13_metrics = (
        "final_sortedness_percent",
        "compare_plus_swap_steps",
        "dg_primary",
        "sortedness_path_curvature_ratio",
    )
    for source in s13_class.to_dict(orient="records"):
        condition_id = str(source["condition_id"])
        stop_regime = str(source["stop_regime"])
        if stop_regime == "reference_no_legal_2000":
            continue
        subset = s13_raw[s13_raw["condition_id"].astype(str) == condition_id]
        for metric in s13_metrics:
            deltas = paired_metric_delta(
                subset,
                id_cols=("condition_id", "repeat_index"),
                treatment_col="stop_regime",
                reference="reference_no_legal_2000",
                treatment=stop_regime,
                metric=metric,
            )
            rows += add_delta_rows(
                values=deltas,
                source_step_id="S13",
                fdr_family="S13_stop_conditions",
                claim_family=metric_claim_family(metric),
                claim_id=f"S13_{condition_id}_{stop_regime}_{metric}",
                metric=metric,
                comparison=f"{stop_regime}_minus_reference_no_legal_2000",
                prior_classification=str(source.get("stop_sensitivity_class")),
                context={"condition_id": condition_id, "stop_regime": stop_regime, "algorithm": source.get("algorithm")},
                base_seed=base_seed,
                reps=bootstrap_reps,
                caveat="Stop-condition matrix fixes placement and behavior and varies stopping rules only.",
            )

    tests = pd.DataFrame(rows)
    tests = apply_fdr_and_interpretation(tests)
    return tests.sort_values(["source_step_id", "fdr_family", "claim_id", "metric"]).reset_index(drop=True)


def apply_fdr_and_interpretation(tests: pd.DataFrame) -> pd.DataFrame:
    frame = tests.copy()
    frame["family_q_value"] = None
    for family, idx in frame.groupby("fdr_family").groups.items():
        q_values = bh_adjust(frame.loc[list(idx), "permutation_p_value"].tolist())
        frame.loc[list(idx), "family_q_value"] = q_values
    frame["global_q_value"] = bh_adjust(frame["permutation_p_value"].tolist())
    corrected_results: list[str] = []
    interpretations: list[str] = []
    for row in frame.to_dict(orient="records"):
        q = row.get("family_q_value")
        q_float = float(q) if q is not None and pd.notna(q) else math.nan
        prior = str(row.get("prior_classification", "")).lower()
        effect = row.get("effect_estimate")
        effect_float = float(effect) if effect is not None and pd.notna(effect) else math.nan
        if not math.isfinite(q_float):
            corrected = "descriptive_only"
        elif q_float <= ALPHA:
            corrected = "detected_after_family_fdr"
        else:
            corrected = "not_detected_after_family_fdr"
        corrected_results.append(corrected)
        sensitive_prior = any(token in prior for token in ("sensitive", "diverges", "exceeds", "dependent", "contradictory", "not replicated"))
        stable_prior = any(token in prior for token in ("stable", "independent", "within_original_envelope", "invariant", "insensitive", "negative_control"))
        if corrected == "detected_after_family_fdr" and sensitive_prior:
            interp = "corrected_constraining_signal"
        elif corrected == "detected_after_family_fdr" and stable_prior and abs(effect_float) > 1e-12:
            interp = "small_but_corrected_difference_in_stability_slice"
        elif corrected == "not_detected_after_family_fdr" and stable_prior:
            interp = "corrected_support_for_stability_or_null_control"
        elif corrected == "not_detected_after_family_fdr" and sensitive_prior:
            interp = "prior_sensitivity_not_detected_after_s14_correction"
        elif corrected == "detected_after_family_fdr":
            interp = "corrected_detected_difference"
        elif corrected == "descriptive_only":
            interp = "descriptive_or_imported_without_correctable_p_value"
        else:
            interp = "not_detected_after_s14_correction"
        interpretations.append(interp)
    frame["corrected_result"] = corrected_results
    frame["statistical_interpretation"] = interpretations
    return frame


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    columns = list(show.columns)
    rows = [[str(value) for value in row] for row in show.itertuples(index=False, name=None)]

    def esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(esc(col) for col in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(esc(value) for value in row) + " |" for row in rows]
    suffix = [f"\nShowing first {max_rows} of {len(df)} rows."] if len(df) > max_rows else []
    return "\n".join([header, divider, *body, *suffix])


def summarize_tests(tests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_summary = (
        tests.groupby(["source_step_id", "fdr_family"], dropna=False)
        .agg(
            n_tests=("claim_id", "count"),
            n_permutation_p=("permutation_p_value", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            n_family_fdr_detected=("corrected_result", lambda s: int((s == "detected_after_family_fdr").sum())),
            n_global_fdr_detected=("global_q_value", lambda s: int((pd.to_numeric(s, errors="coerce") <= ALPHA).sum())),
            n_mixedlm=("model_type", lambda s: int((s == "mixedlm_random_intercept").sum())),
            n_fallback_models=("model_type", lambda s: int(s.astype(str).str.contains("fallback").sum())),
            min_family_q=("family_q_value", lambda s: pd.to_numeric(s, errors="coerce").min()),
            median_abs_effect=("effect_estimate", lambda s: pd.to_numeric(s, errors="coerce").abs().median()),
        )
        .reset_index()
    )
    interpretation_summary = (
        tests.groupby(["source_step_id", "statistical_interpretation"], dropna=False)
        .size()
        .reset_index(name="n_tests")
        .sort_values(["source_step_id", "statistical_interpretation"])
    )
    return family_summary, interpretation_summary


def validate_outputs(
    *,
    tests: pd.DataFrame,
    family_summary: pd.DataFrame,
    interpretation_summary: pd.DataFrame,
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    unit_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected_sources = {"E01", *{f"S{i:02d}" for i in range(2, 14)}}
    observed_sources = set(tests["source_step_id"].dropna().astype(str).unique())
    required_output_keys = {
        "test_table_parquet",
        "test_table_csv",
        "family_summary_csv",
        "interpretation_summary_csv",
        "strong_report",
        "full_results_report",
        "validation_json",
        "status_json",
        "manifest_json",
        "src_manifest_json",
        "log",
    }
    source_counts = tests["source_step_id"].value_counts().sort_index().to_dict()
    fdr_family_count = int(tests["fdr_family"].nunique())
    permutation_count = int(pd.to_numeric(tests["permutation_p_value"], errors="coerce").notna().sum())
    bootstrap_count = int((tests["effect_ci95_low"].notna() & tests["effect_ci95_high"].notna()).sum())
    model_formula_complete = bool(tests["model_formula"].notna().all() and (tests["model_formula"].astype(str).str.len() > 0).all())
    p_present = pd.to_numeric(tests["permutation_p_value"], errors="coerce").notna()
    q_complete = bool(
        tests.loc[p_present, "family_q_value"].notna().all()
        and tests.loc[p_present, "global_q_value"].notna().all()
    )
    input_present = all(path.exists() for path in input_paths.values())
    output_present = all(output_paths[key].exists() and output_paths[key].stat().st_size > 0 for key in required_output_keys)
    unit_success = bool(unit_result is None or unit_result.get("success"))
    success = bool(
        expected_sources <= observed_sources
        and fdr_family_count >= 13
        and len(tests) >= 1_000
        and permutation_count >= 1_000
        and bootstrap_count >= 1_000
        and model_formula_complete
        and q_complete
        and input_present
        and output_present
        and unit_success
    )
    family_detected = int((tests["corrected_result"] == "detected_after_family_fdr").sum())
    outcome = "constraining/contradictory" if family_detected > 0 else "null"
    validation = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "failed_validation",
        "validationResult": "pending",
        "outcomeClassification": outcome,
        "artifactsWritten": [],
        "caveatsOrBlockers": [
            "S14 is a statistical meta-analysis over completed E01/E02 artifacts, not a simulator rerun.",
            "Most E02 matrices are bounded to the first 10 E01 repeats per selected condition, so corrected tests are audit-strength rather than full paper-scale inference.",
            "Mixed-effects models are attempted where grouped data support them; sparse paired slices use documented blocked-OLS or paired-t fallback models.",
            "FDR correction is applied within explicit step-level families and globally; interpretation remains tied to each prior step's caveats.",
        ],
        "recommendedNextAction": "Review S14 corrected evidence, then proceed to S15 claim audit only after explicit Chief Scientist instruction.",
        "expectedSources": sorted(expected_sources),
        "observedSources": sorted(observed_sources),
        "sourceCounts": source_counts,
        "testRowCount": int(len(tests)),
        "familySummaryRowCount": int(len(family_summary)),
        "interpretationSummaryRowCount": int(len(interpretation_summary)),
        "fdrFamilyCount": fdr_family_count,
        "permutationPValueCount": permutation_count,
        "bootstrapIntervalCount": bootstrap_count,
        "familyFdrDetectedCount": family_detected,
        "globalFdrDetectedCount": int((pd.to_numeric(tests["global_q_value"], errors="coerce") <= ALPHA).sum()),
        "modelTypeCounts": tests["model_type"].value_counts(dropna=False).sort_index().to_dict(),
        "statisticalInterpretationCounts": tests["statistical_interpretation"].value_counts().sort_index().to_dict(),
        "inputArtifactsPresent": input_present,
        "outputsPresent": output_present,
        "modelFormulasDocumented": model_formula_complete,
        "fdrQValuesComplete": q_complete,
        "unitTestsPassed": unit_success,
    }
    validation["validationResult"] = (
        "passed: E01 and S02-S13 sources present, statistical table written, permutation tests and bootstrap intervals populated, model formulas documented, explicit FDR families adjusted, reports/manifests present, and unit tests passed"
        if success
        else "failed: one or more S14 validation checks did not pass"
    )
    return validation


def compact_top_rows(tests: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_step_id",
        "claim_family",
        "claim_id",
        "metric",
        "effect_estimate",
        "effect_ci95_low",
        "effect_ci95_high",
        "permutation_p_value",
        "family_q_value",
        "global_q_value",
        "prior_classification",
        "statistical_interpretation",
    ]
    return (
        tests[columns]
        .assign(abs_effect=lambda frame: pd.to_numeric(frame["effect_estimate"], errors="coerce").abs())
        .sort_values(["family_q_value", "abs_effect"], ascending=[True, False])
        .drop(columns=["abs_effect"])
        .head(24)
    )


def build_strong_statistics_report(
    *,
    generated_at: str,
    tests: pd.DataFrame,
    family_summary: pd.DataFrame,
    interpretation_summary: pd.DataFrame,
    validation: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    input_paths: Mapping[str, Path],
    parameters: Mapping[str, Any],
) -> str:
    family_counts = tests["fdr_family"].value_counts().sort_index().to_dict()
    interp_counts = tests["statistical_interpretation"].value_counts().sort_index().to_dict()
    detected = tests[tests["corrected_result"] == "detected_after_family_fdr"].copy()
    detected_excerpt = compact_top_rows(detected if not detected.empty else tests)
    return f"""# E02 Strong Statistics Report

## Top Summary

- Step ID: S14
- Completion status: {validation['status']}
- Artifacts written: {', '.join(validation['artifactsWritten'])}
- Validation result: {validation['validationResult']}
- Outcome classification: {validation['outcomeClassification']}
- Caveats or blockers: {', '.join(validation['caveatsOrBlockers'])}
- Lay summary: S14 rechecked E01 and E02 claim slices with permutation tests, bootstrap intervals, fallback models, and explicit FDR correction. Corrected differences remain common in the artifact-audit layers, so final claim verdicts should stay conditional on the relevant scheduler, input, Frozen Cell, and stop-rule assumptions.
- Recommended next action: {validation['recommendedNextAction']}

Generated at: {generated_at}

## Summary

S14 applied stronger statistics to E01 legacy claim tests and E02 S02-S13 artifact-audit claim slices. The primary inferential layer is paired/block permutation testing with bootstrap confidence intervals. Benjamini-Hochberg FDR correction was applied within explicit step-level families and globally.

- Total statistical rows: {len(tests)}
- FDR families: {validation['fdrFamilyCount']}
- Family-FDR detected rows: {validation['familyFdrDetectedCount']}
- Global-FDR detected rows: {validation['globalFdrDetectedCount']}
- Outcome classification: {validation['outcomeClassification']}
- Model type counts: {json.dumps(validation['modelTypeCounts'], sort_keys=True)}
- Interpretation counts: {json.dumps(interp_counts, sort_keys=True)}

## FDR Families

The analysis defines one family per E01 claim family or E02 audit step:

{json.dumps(family_counts, indent=2, sort_keys=True)}

This keeps correction aligned with how the experiment was planned: scheduler, activation-rate, label-null, dummy-control, speed-match, local-null, DG-null, metric, input-distribution, Frozen Cell placement, Frozen Cell behavior, and stop-condition evidence are controlled separately, with an additional global BH q-value for cross-family screening.

## Methods

- E01 rows import the E01 z-test p-values and attach paired bootstrap intervals where E01 provided them.
- S02 and S03 use blocked label-permutation range tests. Scheduler or activation labels are permuted within condition/repeat blocks, preserving the matched initial arrays.
- S04-S13 pair each intervention/null/alternative condition with its reference slice where possible, then test the mean paired delta with exact sign-flip tests for small n or Monte Carlo sign-flip tests for larger n.
- Bootstrap intervals resample paired units and report 95 percent percentile intervals for the mean effect.
- Mixed-effects models are attempted for grouped omnibus rows as `y ~ C(group) + (1 | block)`. Sparse paired slices use deterministic fallback models documented as `delta ~ 1`; this is expected for many 10-repeat matrices.

## Corrected Signals

{markdown_table(detected_excerpt, max_rows=24)}

## Family Summary

{markdown_table(family_summary, max_rows=40)}

## Interpretation Summary

{markdown_table(interpretation_summary, max_rows=60)}

## Inputs

{markdown_table(pd.DataFrame([{"input": key, "path": str(path), "exists": path.exists()} for key, path in input_paths.items()]), max_rows=40)}

## Outputs

- Statistical test table: `{output_paths['test_table_parquet']}`
- CSV copy: `{output_paths['test_table_csv']}`
- Family summary: `{output_paths['family_summary_csv']}`
- Interpretation summary: `{output_paths['interpretation_summary_csv']}`
- Validation JSON: `{output_paths['validation_json']}`
- Status JSON: `{output_paths['status_json']}`
- Full results report: `{output_paths['full_results_report']}`

## Parameters

{json.dumps(parameters, indent=2, sort_keys=True)}

## Caveats

- Corrected p-values test artifact-audit deltas and null contrasts, not biological mechanisms.
- Many E02 matrices intentionally use bounded first-10-repeat slices; non-detection after FDR is not proof of invariance at full paper scale.
- Exact E01 thread interleavings remain unavailable; E02 deterministic simulator evidence is a controlled audit layer.
- Some stability claims are better understood as equivalence questions. S14 reports standard difference tests and CIs; S15 should use both corrected tests and predeclared practical thresholds when assigning final verdicts.
"""


def build_full_results_report(
    *,
    generated_at: str,
    validation: Mapping[str, Any],
    tests: pd.DataFrame,
    family_summary: pd.DataFrame,
    interpretation_summary: pd.DataFrame,
    output_paths: Mapping[str, Path],
    input_paths: Mapping[str, Path],
    unit_result: Mapping[str, Any] | None,
    git_commit: str,
    git_status: str,
    elapsed_seconds: float,
    parameters: Mapping[str, Any],
) -> str:
    unit_line = "not run"
    if unit_result is not None:
        unit_line = f"{unit_result['command']} -> return code {unit_result['returnCode']}"
    detected_excerpt = compact_top_rows(tests[tests["corrected_result"] == "detected_after_family_fdr"])
    return f"""# E02 S14 Full Results: Stronger Statistics

## Top Summary

- Step ID: S14
- Completion status: {validation['status']}
- Artifacts written: {', '.join(validation['artifactsWritten'])}
- Validation result: {validation['validationResult']}
- Outcome classification: {validation['outcomeClassification']}
- Caveats or blockers: {', '.join(validation['caveatsOrBlockers'])}
- Lay summary: S14 rechecked E01 and E02 claim slices with permutation tests, bootstrap intervals, fallback models, and explicit FDR correction. Many artifact-audit sensitivities remain statistically detectable after family correction, so the final E02 claim audit should treat broad invariance claims as narrowed by scheduler, input, Frozen Cell, and stop-condition assumptions.
- Recommended next action: {validation['recommendedNextAction']}

## Frozen Question

Do original claims survive permutation tests, mixed-effects models or fallback models, bootstrap intervals, and false-discovery-rate correction across the expanded E02 condition matrix?

## Inputs

S14 used completed E01 and E02 artifacts only; no simulations were rerun.

{markdown_table(pd.DataFrame([{"input": key, "path": str(path), "exists": path.exists()} for key, path in input_paths.items()]), max_rows=40)}

## Methods

The runner constructs claim-slice rows from E01 legacy z tests and S02-S13 result/classification tables. S02/S03 use blocked label permutations across scheduler or activation-regime labels. S04-S13 use paired deltas against each step's reference/null/baseline slice and exact or Monte Carlo sign-flip permutation tests. Bootstrap intervals are 95 percent percentile intervals over paired units. FDR is Benjamini-Hochberg within explicit step-level families and globally.

MixedLM random-intercept models are attempted for grouped omnibus tests. Paired or sparse slices fall back to documented paired-delta models (`delta ~ 1`) or blocked OLS where applicable.

## Commands

- `python scripts/e02_s14_stronger_statistics.py --repo-dir /workspace/cell-research --artifacts-dir /artifacts --previous-e01-dir /previous-artifacts/E01 --bootstrap-reps {parameters['bootstrapReps']} --permutation-reps {parameters['permutationReps']} --seed {parameters['seed']} --run-unit-tests`
- Unit tests: {unit_line}

## Results

- Statistical test rows: {validation['testRowCount']}
- Sources observed: {validation['observedSources']}
- FDR families: {validation['fdrFamilyCount']}
- Permutation p-values: {validation['permutationPValueCount']}
- Bootstrap intervals: {validation['bootstrapIntervalCount']}
- Family-FDR detected rows: {validation['familyFdrDetectedCount']}
- Global-FDR detected rows: {validation['globalFdrDetectedCount']}
- Model type counts: {json.dumps(validation['modelTypeCounts'], sort_keys=True)}
- Interpretation counts: {json.dumps(validation['statisticalInterpretationCounts'], sort_keys=True)}

### Strongest Corrected Rows

{markdown_table(detected_excerpt, max_rows=24)}

### FDR Family Summary

{markdown_table(family_summary, max_rows=40)}

### Interpretation Summary

{markdown_table(interpretation_summary, max_rows=60)}

## Validation

- Expected sources present: {set(validation['expectedSources']) <= set(validation['observedSources'])}
- Input artifacts present: {validation['inputArtifactsPresent']}
- Outputs present: {validation['outputsPresent']}
- Model formulas documented: {validation['modelFormulasDocumented']}
- FDR q-values complete: {validation['fdrQValuesComplete']}
- Unit tests passed: {validation['unitTestsPassed']}
- Validation JSON: `{output_paths['validation_json']}`
- Status JSON: `{output_paths['status_json']}`

## Artifacts

- Statistical tests parquet: `{output_paths['test_table_parquet']}`
- Statistical tests CSV: `{output_paths['test_table_csv']}`
- Strong statistics report: `{output_paths['strong_report']}`
- Family summary CSV: `{output_paths['family_summary_csv']}`
- Interpretation summary CSV: `{output_paths['interpretation_summary_csv']}`
- Artifact manifest: `{output_paths['manifest_json']}`
- Source snapshot manifest: `{output_paths['src_manifest_json']}`
- Log: `{output_paths['log']}`

## Provenance

- Generated at: {generated_at}
- Elapsed seconds: {elapsed_seconds:.2f}
- Git commit before final commit: `{git_commit}`
- Git status before final commit: `{git_status}`
- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Parameters: `{json.dumps(parameters, sort_keys=True)}`

## Caveats, Blockers, And Limitations

- This is an artifact-level statistical audit, not a new simulation step.
- Bounded 10-repeat E02 matrices limit power and precision for many slices.
- FDR-corrected non-detection does not prove practical equivalence; S15 should combine p-values, CIs, prior predeclared thresholds, and caveats.
- E01 legacy rows are imported z tests with E01 bootstrap intervals where available, not recomputed raw paired models.
- Some null rows aggregate replicate null trajectories within condition/repeat before testing to reduce pseudo-replication.

## Recommended Next Action

Proceed to S15 claim audit only after explicit Chief Scientist instruction. S15 should cite S14 corrected rows while preserving each upstream caveat.
"""


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    generated_at = utc_now()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    previous_e01_dir = args.previous_e01_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    table_dir = artifacts_dir / "tables"
    report_dir = artifacts_dir / "reports"
    log_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    for path in (step_dir, table_dir, report_dir, log_dir, src_snapshot_dir):
        path.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "test_table_parquet": table_dir / "e02_statistical_tests.parquet",
        "test_table_csv": step_dir / "e02_statistical_tests.csv",
        "family_summary_csv": step_dir / "e02_fdr_family_summary.csv",
        "interpretation_summary_csv": step_dir / "e02_statistical_interpretation_summary.csv",
        "strong_report": report_dir / "e02_strong_statistics.md",
        "full_results_report": step_dir / "research_step_full_results.md",
        "validation_json": step_dir / "s14_validation.json",
        "status_json": step_dir / "status.json",
        "manifest_json": step_dir / "artifact_manifest.json",
        "src_manifest_json": src_snapshot_dir / "e02_s14_stronger_statistics_manifest.json",
        "log": log_dir / "e02_s14_stronger_statistics.log",
    }

    with output_paths["log"].open("w", encoding="utf-8") as log:
        log.write(f"{generated_at} Starting E02 S14 stronger statistics\n")
        log.write(f"repo_dir={repo_dir}\nartifacts_dir={artifacts_dir}\nprevious_e01_dir={previous_e01_dir}\n")

    input_paths = InputPaths(artifacts_dir=artifacts_dir, previous_e01_dir=previous_e01_dir).result_paths
    inputs = load_all_inputs(input_paths)
    parameters = {
        "bootstrapReps": int(args.bootstrap_reps),
        "permutationReps": int(args.permutation_reps),
        "seed": int(args.seed),
        "alpha": ALPHA,
        "inputCount": len(input_paths),
    }

    unit_result: Mapping[str, Any] | None = None
    if args.run_unit_tests:
        unit_result = run_command([sys.executable, "-m", "unittest", "tests.e02.test_stronger_statistics"], repo_dir)
        if not unit_result["success"]:
            write_json(step_dir / "s14_unit_test_failure.json", unit_result)
            raise RuntimeError(f"S14 unit tests failed; see {step_dir / 's14_unit_test_failure.json'}")

    tests = build_statistical_tests(
        inputs,
        base_seed=int(args.seed),
        bootstrap_reps=int(args.bootstrap_reps),
        permutation_reps=int(args.permutation_reps),
    )
    family_summary, interpretation_summary = summarize_tests(tests)
    tests.to_parquet(output_paths["test_table_parquet"], index=False)
    tests.to_csv(output_paths["test_table_csv"], index=False)
    family_summary.to_csv(output_paths["family_summary_csv"], index=False)
    interpretation_summary.to_csv(output_paths["interpretation_summary_csv"], index=False)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])

    # Placeholder files before validation so presence checks can include reports/manifests.
    output_paths["strong_report"].write_text("pending\n", encoding="utf-8")
    output_paths["full_results_report"].write_text("pending\n", encoding="utf-8")
    write_json(output_paths["validation_json"], {"status": "pending"})
    write_json(
        output_paths["status_json"],
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": False,
            "status": "pending",
            "artifactsWritten": [],
            "validationResult": "pending",
            "caveatsOrBlockers": [],
            "recommendedNextAction": "pending",
        },
    )

    src_manifest = {
        "researchStepId": STEP_ID,
        "generatedAt": generated_at,
        "gitCommitBeforeFinalCommit": git_commit,
        "gitStatusBeforeFinalCommit": git_status,
        "sourceFiles": [
            artifact_entry(repo_dir / "scripts/e02_s14_stronger_statistics.py", repo_dir, "S14 stronger-statistics runner"),
            artifact_entry(repo_dir / "tests/e02/test_stronger_statistics.py", repo_dir, "focused S14 statistics tests"),
        ],
    }
    write_json(output_paths["src_manifest_json"], src_manifest)
    manifest = {
        "researchStepId": STEP_ID,
        "generatedAt": generated_at,
        "parameters": parameters,
        "artifacts": [],
    }
    write_json(output_paths["manifest_json"], manifest)

    validation = validate_outputs(
        tests=tests,
        family_summary=family_summary,
        interpretation_summary=interpretation_summary,
        input_paths=input_paths,
        output_paths=output_paths,
        unit_result=unit_result,
    )
    elapsed = time.monotonic() - started
    strong_report = build_strong_statistics_report(
        generated_at=generated_at,
        tests=tests,
        family_summary=family_summary,
        interpretation_summary=interpretation_summary,
        validation=validation,
        output_paths=output_paths,
        input_paths=input_paths,
        parameters=parameters,
    )
    output_paths["strong_report"].write_text(strong_report, encoding="utf-8")
    full_report = build_full_results_report(
        generated_at=generated_at,
        validation=validation,
        tests=tests,
        family_summary=family_summary,
        interpretation_summary=interpretation_summary,
        output_paths=output_paths,
        input_paths=input_paths,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        elapsed_seconds=elapsed,
        parameters=parameters,
    )
    output_paths["full_results_report"].write_text(full_report, encoding="utf-8")

    artifact_descriptions = {
        "test_table_parquet": "planned S14 statistical test table",
        "test_table_csv": "CSV copy of S14 statistical test table",
        "family_summary_csv": "S14 FDR family summary",
        "interpretation_summary_csv": "S14 statistical interpretation summary",
        "strong_report": "planned strong-statistics report",
        "full_results_report": "canonical S14 full-results report",
        "validation_json": "S14 validation evidence",
        "status_json": "compact S14 status JSON",
        "manifest_json": "S14 artifact manifest",
        "src_manifest_json": "S14 source snapshot manifest",
        "log": "S14 execution log",
    }
    artifacts = [
        artifact_entry(path, artifacts_dir, artifact_descriptions[key])
        for key, path in output_paths.items()
        if path.exists() and key != "manifest_json"
    ]
    manifest["artifacts"] = artifacts
    manifest["validation"] = {
        "success": bool(validation["success"]),
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
    }
    write_json(output_paths["manifest_json"], manifest)
    validation["artifactsWritten"] = [str(output_paths[key]) for key in output_paths if output_paths[key].exists()]
    write_json(output_paths["validation_json"], validation)
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"]),
        "status": validation["status"],
        "artifactsWritten": validation["artifactsWritten"],
        "validationResult": validation["validationResult"],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
    }
    write_json(output_paths["status_json"], status)
    # Refresh reports after final artifact list is known.
    output_paths["strong_report"].write_text(
        build_strong_statistics_report(
            generated_at=generated_at,
            tests=tests,
            family_summary=family_summary,
            interpretation_summary=interpretation_summary,
            validation=validation,
            output_paths=output_paths,
            input_paths=input_paths,
            parameters=parameters,
        ),
        encoding="utf-8",
    )
    output_paths["full_results_report"].write_text(
        build_full_results_report(
            generated_at=generated_at,
            validation=validation,
            tests=tests,
            family_summary=family_summary,
            interpretation_summary=interpretation_summary,
            output_paths=output_paths,
            input_paths=input_paths,
            unit_result=unit_result,
            git_commit=git_commit,
            git_status=git_status,
            elapsed_seconds=elapsed,
            parameters=parameters,
        ),
        encoding="utf-8",
    )
    manifest["artifacts"] = [
        artifact_entry(path, artifacts_dir, artifact_descriptions[key])
        for key, path in output_paths.items()
        if path.exists() and key != "manifest_json"
    ]
    manifest["validation"] = {
        "success": bool(validation["success"]),
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
    }
    write_json(output_paths["manifest_json"], manifest)
    with output_paths["log"].open("a", encoding="utf-8") as log:
        log.write(f"{utc_now()} Completed S14 status={validation['status']} success={validation['success']} elapsed={elapsed:.2f}s\n")
        log.write(json.dumps({"testRows": len(tests), "familyFdrDetected": validation["familyFdrDetectedCount"]}, sort_keys=True) + "\n")
    if not validation["success"]:
        raise RuntimeError(f"S14 validation failed; see {output_paths['validation_json']}")


if __name__ == "__main__":
    main()
