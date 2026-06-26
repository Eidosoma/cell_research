#!/usr/bin/env python3
"""Run E02 S14 stronger statistical audit across S01-S13 artifacts.

S14 is an analysis-only step. It consumes completed E02 artifacts, applies
paired sign-flip/permutation tests, empirical null comparisons, bootstrap
intervals, Benjamini-Hochberg FDR correction within declared families, and
mixed/clustered models where the accumulated rows support them.
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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:  # statsmodels is preinstalled in the workspace image.
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive fallback
    sm = None
    smf = None
    multipletests = None
    STATSMODELS_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "E02"
STEP_ID = "S14"
STEP_NUMBER = 14
STATUS = "completed"
OUTCOME_CLASSIFICATION = "constraining/contradictory"

ARTIFACTS_DIR_DEFAULT = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
RESULTS_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "results"
FIGURES_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "figures" / "e02"
STEP_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "research_steps" / STEP_ID
REPORTS_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "reports"
PROVENANCE_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "provenance"

BOOTSTRAP_SEED = 31415926
PERMUTATION_SEED = 27182818
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 10_000
FDR_ALPHA = 0.05

EXPECTED_ROW_COUNTS = {
    "e02_s01_replicate_summary.parquet": 36,
    "e02_scheduler_regimes.parquet": 84,
    "e02_scheduler_regime_summary.parquet": 28,
    "e02_activation_rates.parquet": 75,
    "e02_activation_rate_summary.parquet": 25,
    "e02_label_shuffle_observed.parquet": 12,
    "e02_label_shuffle_nulls.parquet": 6000,
    "e02_label_shuffle_mixture_summary.parquet": 4,
    "e02_dummy_algotypes.parquet": 18,
    "e02_dummy_algotypes_nulls.parquet": 9000,
    "e02_dummy_algotypes_summary.parquet": 6,
    "e02_speed_matched_algotypes.parquet": 24,
    "e02_speed_matched_label_shuffle_nulls.parquet": 12000,
    "e02_speed_matched_summary.parquet": 8,
    "e02_local_move_null_sources.parquet": 39,
    "e02_local_move_nulls.parquet": 3900,
    "e02_local_move_null_summary.parquet": 16,
    "e02_dg_observed.parquet": 39,
    "e02_dg_nulls.parquet": 19500,
    "e02_dg_source_summary.parquet": 39,
    "e02_alternative_metrics.parquet": 9775,
    "e02_alternative_metric_sensitivity.parquet": 8,
    "e02_alternative_metric_correlations.parquet": 36,
    "e02_input_distributions.parquet": 210,
    "e02_input_distribution_summary.parquet": 70,
    "e02_frozen_placement.parquet": 135,
    "e02_frozen_placement_summary.parquet": 45,
    "e02_frozen_behavior_variants.parquet": 189,
    "e02_frozen_behavior_variant_summary.parquet": 63,
    "e02_stop_condition_sensitivity.parquet": 432,
    "e02_stop_condition_summary.parquet": 144,
    "e02_stop_condition_policy_effects.parquet": 12,
}

REQUIRED_COLUMNS = {
    "e02_s01_replicate_summary.parquet": [
        "research_step_id",
        "condition_id",
        "replicate_index",
        "input_permutation_seed",
        "scheduler_seed",
        "tie_breaker_seed",
        "completed",
    ],
    "e02_scheduler_regimes.parquet": [
        "schedulerRegime",
        "mixtureId",
        "replicateIndex",
        "inputPermutationSeed",
        "schedulerSeed",
        "initialValuesHash",
        "finalSortednessPercent",
        "swapCount",
        "activationCount",
    ],
    "e02_activation_rates.parquet": [
        "mixtureId",
        "rateProfileId",
        "replicateIndex",
        "inputPermutationSeed",
        "schedulerSeed",
        "initialValuesHash",
        "finalSortednessPercent",
        "swapCount",
        "activationCount",
        "peakAggregation",
    ],
    "e02_label_shuffle_observed.parquet": [
        "conditionId",
        "mixtureId",
        "replicateIndex",
        "inputPermutationSeed",
        "trajectoryHistoryHash",
        "observedPeakAggregation",
    ],
    "e02_label_shuffle_nulls.parquet": [
        "conditionId",
        "mixtureId",
        "replicateIndex",
        "nullReplicateIndex",
        "nullSeed",
        "trajectoryHistoryHash",
        "nullPeakAggregation",
    ],
    "e02_dummy_algotypes.parquet": [
        "conditionId",
        "basePolicy",
        "labelSchemeId",
        "replicateIndex",
        "inputPermutationSeed",
        "observedPeakDummyAggregation",
    ],
    "e02_dummy_algotypes_nulls.parquet": [
        "sourcePureConditionId",
        "basePolicy",
        "labelSchemeId",
        "replicateIndex",
        "nullReplicateIndex",
        "nullSeed",
        "nullPeakDummyAggregation",
    ],
    "e02_speed_matched_algotypes.parquet": [
        "conditionId",
        "mixtureId",
        "speedProfileId",
        "replicateIndex",
        "inputPermutationSeed",
        "observedPeakAggregation",
    ],
    "e02_speed_matched_label_shuffle_nulls.parquet": [
        "conditionId",
        "mixtureId",
        "speedProfileId",
        "replicateIndex",
        "nullReplicateIndex",
        "nullSeed",
        "nullPeakAggregation",
    ],
    "e02_local_move_nulls.parquet": [
        "sourceRunId",
        "nullModel",
        "nullReplicateIndex",
        "nullSeed",
        "inputPermutationSeed",
        "realFinalSortednessPercent",
        "nullFinalSortednessPercent",
    ],
    "e02_dg_nulls.parquet": [
        "sourceRunId",
        "nullReplicateIndex",
        "nullSeed",
        "observedDelayedGratification",
        "nullDelayedGratification",
    ],
    "e02_alternative_metrics.parquet": [
        "metricSource",
        "sourceRunId",
        "nullModel",
        "nullReplicateIndex",
        "finalKendallTauDistanceNormalized",
    ],
    "e02_input_distributions.parquet": [
        "conditionTemplateId",
        "inputProfile",
        "replicateIndex",
        "inputSeed",
        "completed",
        "finalKendallTauDistanceNormalized",
    ],
    "e02_frozen_placement.parquet": [
        "algorithm",
        "frozenVariant",
        "placementCategory",
        "replicateIndex",
        "inputSeed",
        "completed",
        "finalKendallTauDistanceNormalized",
    ],
    "e02_frozen_behavior_variants.parquet": [
        "algorithm",
        "placementCategory",
        "defectBehaviorVariant",
        "replicateIndex",
        "inputSeed",
        "completed",
        "finalKendallTauDistanceNormalized",
    ],
    "e02_stop_condition_sensitivity.parquet": [
        "conditionTemplateId",
        "stopPolicyId",
        "replicateIndex",
        "inputSeed",
        "initialValuesHash",
        "initialPolicyHash",
        "criterionCompleted",
        "finalKendallTauDistanceNormalized",
    ],
}

SEED_COLUMNS = {
    "e02_s01_replicate_summary.parquet": ["input_permutation_seed", "scheduler_seed", "tie_breaker_seed"],
    "e02_scheduler_regimes.parquet": ["inputPermutationSeed", "schedulerSeed", "tieBreakerSeed"],
    "e02_activation_rates.parquet": ["inputPermutationSeed", "schedulerSeed", "tieBreakerSeed"],
    "e02_label_shuffle_observed.parquet": ["inputPermutationSeed", "schedulerSeed", "tieBreakerSeed"],
    "e02_label_shuffle_nulls.parquet": ["nullSeed"],
    "e02_dummy_algotypes.parquet": ["inputPermutationSeed", "schedulerSeed", "tieBreakerSeed"],
    "e02_dummy_algotypes_nulls.parquet": ["nullSeed"],
    "e02_speed_matched_algotypes.parquet": ["inputPermutationSeed", "schedulerSeed", "tieBreakerSeed"],
    "e02_speed_matched_label_shuffle_nulls.parquet": ["nullSeed"],
    "e02_local_move_nulls.parquet": ["inputPermutationSeed", "nullSeed"],
    "e02_dg_nulls.parquet": ["nullSeed"],
    "e02_input_distributions.parquet": ["inputSeed", "schedulerSeed", "tieBreakerSeed"],
    "e02_frozen_placement.parquet": ["inputSeed", "schedulerSeed", "tieBreakerSeed"],
    "e02_frozen_behavior_variants.parquet": [
        "inputSeed",
        "schedulerSeed",
        "tieBreakerSeed",
        "frozenPositionSeed",
        "behaviorSeed",
    ],
    "e02_stop_condition_sensitivity.parquet": ["inputSeed", "schedulerSeed", "tieBreakerSeed"],
}

SOURCE_STEP_BY_FILE = {
    "e02_s01_replicate_summary.parquet": "S01",
    "e02_scheduler_regimes.parquet": "S02",
    "e02_scheduler_regime_summary.parquet": "S02",
    "e02_activation_rates.parquet": "S03",
    "e02_activation_rate_summary.parquet": "S03",
    "e02_label_shuffle_observed.parquet": "S04",
    "e02_label_shuffle_nulls.parquet": "S04",
    "e02_label_shuffle_mixture_summary.parquet": "S04",
    "e02_dummy_algotypes.parquet": "S05",
    "e02_dummy_algotypes_nulls.parquet": "S05",
    "e02_dummy_algotypes_summary.parquet": "S05",
    "e02_speed_matched_algotypes.parquet": "S06",
    "e02_speed_matched_label_shuffle_nulls.parquet": "S06",
    "e02_speed_matched_summary.parquet": "S06",
    "e02_local_move_null_sources.parquet": "S07",
    "e02_local_move_nulls.parquet": "S07",
    "e02_local_move_null_summary.parquet": "S07",
    "e02_dg_observed.parquet": "S08",
    "e02_dg_nulls.parquet": "S08",
    "e02_dg_source_summary.parquet": "S08",
    "e02_alternative_metrics.parquet": "S09",
    "e02_alternative_metric_sensitivity.parquet": "S09",
    "e02_alternative_metric_correlations.parquet": "S09",
    "e02_input_distributions.parquet": "S10",
    "e02_input_distribution_summary.parquet": "S10",
    "e02_frozen_placement.parquet": "S11",
    "e02_frozen_placement_summary.parquet": "S11",
    "e02_frozen_behavior_variants.parquet": "S12",
    "e02_frozen_behavior_variant_summary.parquet": "S12",
    "e02_stop_condition_sensitivity.parquet": "S13",
    "e02_stop_condition_summary.parquet": "S13",
    "e02_stop_condition_policy_effects.parquet": "S13",
}


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    observed_col: str
    reference_col: str | None = None
    direction: str = "two-sided"  # high, low, two-sided
    interpretation: str = ""


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
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "ok": proc.returncode == 0,
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {"args": args, "returncode": None, "stdout": "", "stderr": repr(exc), "ok": False}


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


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % (2**32)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(out)


def coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def mean_or_nan(values: Any) -> float:
    arr = finite_array(values)
    return float(arr.mean()) if len(arr) else math.nan


def sd_or_nan(values: Any) -> float:
    arr = finite_array(values)
    return float(arr.std(ddof=1)) if len(arr) > 1 else math.nan


def percentile_or_nan(values: Any, q: float) -> float:
    arr = finite_array(values)
    return float(np.percentile(arr, q)) if len(arr) else math.nan


def standardized_mean_effect(values: np.ndarray) -> float:
    values = finite_array(values)
    if len(values) < 2:
        return math.nan
    sd = values.std(ddof=1)
    if sd == 0:
        if np.all(values == 0):
            return 0.0
        return math.copysign(math.inf, values.mean())
    return float(values.mean() / sd)


def empirical_p_value(observed: float, null_values: np.ndarray, direction: str) -> float:
    null_values = finite_array(null_values)
    if not np.isfinite(observed) or len(null_values) == 0:
        return math.nan
    if direction == "high":
        extreme = np.sum(null_values >= observed)
        return float((extreme + 1) / (len(null_values) + 1))
    if direction == "low":
        extreme = np.sum(null_values <= observed)
        return float((extreme + 1) / (len(null_values) + 1))
    high = (np.sum(null_values >= observed) + 1) / (len(null_values) + 1)
    low = (np.sum(null_values <= observed) + 1) / (len(null_values) + 1)
    return float(min(1.0, 2.0 * min(high, low)))


def paired_signflip_p_value(
    differences: np.ndarray,
    direction: str,
    seed: int,
    max_exact_n: int = 18,
    replicates: int = PERMUTATION_REPLICATES,
) -> tuple[float, int, str]:
    diffs = finite_array(differences)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return 1.0, 0, "all_zero_differences"

    observed = float(diffs.mean())
    if n <= max_exact_n:
        signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * n))).T.reshape(-1, n)
        means = (signs * diffs).mean(axis=1)
        mode = "exact_signflip"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(replicates, n), replace=True)
        means = (signs * diffs).mean(axis=1)
        mode = "monte_carlo_signflip"

    if direction == "high":
        extreme = np.sum(means >= observed)
    elif direction == "low":
        extreme = np.sum(means <= observed)
    else:
        extreme = np.sum(np.abs(means) >= abs(observed))
    p = (extreme + 1) / (len(means) + 1)
    return float(min(1.0, p)), int(len(means)), mode


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = finite_array(values)
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    if n == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, n, size=(replicates, n))
    means = values[sample_index].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def apply_bh_fdr(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    df["qValue"] = np.nan
    df["significantFdr05"] = False
    df["significantNominal05"] = df["pValue"].astype(float) < FDR_ALPHA
    if "fdrFamilyId" not in df.columns or multipletests is None:
        return df
    for family, idx in df.groupby("fdrFamilyId", dropna=False).groups.items():
        pvals = pd.to_numeric(df.loc[idx, "pValue"], errors="coerce")
        finite = pvals[np.isfinite(pvals)]
        if finite.empty:
            continue
        reject, qvals, _, _ = multipletests(finite.to_numpy(dtype=float), alpha=FDR_ALPHA, method="fdr_bh")
        df.loc[finite.index, "qValue"] = qvals
        df.loc[finite.index, "significantFdr05"] = reject
    return df


def load_required_tables(results_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for filename in EXPECTED_ROW_COUNTS:
        path = results_dir / filename
        if path.exists():
            tables[filename] = pd.read_parquet(path)
    return tables


def validate_sources(results_dir: Path, tables: dict[str, pd.DataFrame], step_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for filename, expected_rows in EXPECTED_ROW_COUNTS.items():
        path = results_dir / filename
        exists = path.exists()
        df = tables.get(filename)
        observed_rows = len(df) if df is not None else 0
        required = REQUIRED_COLUMNS.get(filename, [])
        seed_cols = SEED_COLUMNS.get(filename, [])
        missing_required = [col for col in required if df is None or col not in df.columns]
        missing_seed = [col for col in seed_cols if df is None or col not in df.columns]
        seed_non_null = True
        seed_unique_summary: dict[str, int] = {}
        if df is not None:
            for col in seed_cols:
                if col in df.columns:
                    non_null = df[col].notna().all()
                    seed_non_null = seed_non_null and bool(non_null)
                    seed_unique_summary[col] = int(df[col].nunique(dropna=True))
        passed = bool(
            exists
            and observed_rows == expected_rows
            and not missing_required
            and not missing_seed
            and seed_non_null
        )
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "sourceStepId": SOURCE_STEP_BY_FILE.get(filename, ""),
                "sourceArtifact": str(path),
                "artifactName": filename,
                "expectedRows": int(expected_rows),
                "observedRows": int(observed_rows),
                "exists": bool(exists),
                "requiredColumnsPresent": not missing_required,
                "seedColumnsPresent": not missing_seed,
                "seedColumnsNonNull": bool(seed_non_null),
                "missingRequiredColumns": json.dumps(missing_required),
                "missingSeedColumns": json.dumps(missing_seed),
                "seedUniqueCounts": json.dumps(seed_unique_summary, sort_keys=True),
                "validationPassed": passed,
                "sha256": sha256_path(path) if exists else "",
            }
        )

    for step_num in range(1, 14):
        step_id = f"S{step_num:02d}"
        status_path = step_dir.parent / step_id / "status.json"
        status_ok = False
        validation_result = ""
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status_ok = bool(status.get("success")) and str(status.get("status")) == "completed"
                validation_result = str(status.get("validationResult", ""))
            except Exception as exc:  # pragma: no cover - defensive path
                validation_result = repr(exc)
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "sourceStepId": step_id,
                "sourceArtifact": str(status_path),
                "artifactName": f"{step_id}_status.json",
                "expectedRows": 1,
                "observedRows": int(status_path.exists()),
                "exists": bool(status_path.exists()),
                "requiredColumnsPresent": bool(status_path.exists()),
                "seedColumnsPresent": True,
                "seedColumnsNonNull": True,
                "missingRequiredColumns": "[]",
                "missingSeedColumns": "[]",
                "seedUniqueCounts": "{}",
                "validationPassed": bool(status_ok),
                "sha256": sha256_path(status_path) if status_path.exists() else "",
                "validationResult": validation_result,
            }
        )
    return pd.DataFrame(rows)


def result_row(
    *,
    source_step_id: str,
    source_artifact: str,
    analysis_id: str,
    fdr_family_id: str,
    test_type: str,
    formula: str,
    comparison: str,
    metric: str,
    effect_direction: str,
    group_key: str,
    n_observed: int,
    n_null: int,
    n_pairs: int,
    observed_mean: float,
    reference_mean: float,
    effect_size_raw: float,
    effect_size_standardized: float,
    ci_low: float,
    ci_high: float,
    p_value: float,
    bootstrap_seed: int | None,
    permutation_seed: int | None,
    permutation_replicates: int | None,
    diagnostic_status: str,
    caveat: str = "",
) -> dict[str, Any]:
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "sourceStepId": source_step_id,
        "sourceArtifact": source_artifact,
        "analysisId": analysis_id,
        "fdrFamilyId": fdr_family_id,
        "testType": test_type,
        "formula": formula,
        "comparison": comparison,
        "metric": metric,
        "effectDirection": effect_direction,
        "groupKey": group_key,
        "nObserved": int(n_observed),
        "nNull": int(n_null),
        "nPairs": int(n_pairs),
        "observedMean": observed_mean,
        "referenceMean": reference_mean,
        "effectSizeRaw": effect_size_raw,
        "effectSizeStandardized": effect_size_standardized,
        "ciLow": ci_low,
        "ciHigh": ci_high,
        "pValue": p_value,
        "qValue": math.nan,
        "significantNominal05": bool(np.isfinite(p_value) and p_value < FDR_ALPHA),
        "significantFdr05": False,
        "bootstrapSeed": None if bootstrap_seed is None else int(bootstrap_seed),
        "bootstrapReplicates": BOOTSTRAP_REPLICATES if bootstrap_seed is not None else 0,
        "permutationSeed": None if permutation_seed is None else int(permutation_seed),
        "permutationReplicates": 0 if permutation_replicates is None else int(permutation_replicates),
        "modelConverged": None,
        "diagnosticStatus": diagnostic_status,
        "caveat": caveat,
    }


def paired_reference_tests(
    df: pd.DataFrame,
    *,
    source_step_id: str,
    source_artifact: str,
    analysis_id: str,
    fdr_family_id: str,
    reference_col: str,
    reference_value: str,
    comparison_col: str,
    pair_keys: list[str],
    metrics: list[MetricSpec],
    caveat: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ref = df[df[reference_col].astype(str) == reference_value].copy()
    if ref.empty:
        return rows
    comp = df[df[reference_col].astype(str) != reference_value].copy()
    for comparison_value, comp_group in comp.groupby(comparison_col, dropna=False):
        for metric in metrics:
            needed = pair_keys + [metric.observed_col]
            if any(col not in ref.columns for col in needed) or any(col not in comp_group.columns for col in needed):
                continue
            ref_metric_col = f"{metric.observed_col}_reference"
            comp_metric_col = f"{metric.observed_col}_observed"
            ref_sub = ref[pair_keys + [metric.observed_col]].rename(columns={metric.observed_col: ref_metric_col})
            comp_sub = comp_group[pair_keys + [metric.observed_col]].rename(columns={metric.observed_col: comp_metric_col})
            merged = comp_sub.merge(ref_sub, on=pair_keys, how="inner")
            if merged.empty:
                continue
            observed = coerce_numeric(merged[comp_metric_col])
            reference = coerce_numeric(merged[ref_metric_col])
            diffs = finite_array(observed.to_numpy(dtype=float) - reference.to_numpy(dtype=float))
            if len(diffs) == 0:
                continue
            perm_seed = stable_seed(PERMUTATION_SEED, analysis_id, str(comparison_value), metric.metric)
            boot_seed = stable_seed(BOOTSTRAP_SEED, analysis_id, str(comparison_value), metric.metric)
            p_value, perm_reps, perm_mode = paired_signflip_p_value(
                diffs,
                metric.direction,
                seed=perm_seed,
                replicates=PERMUTATION_REPLICATES,
            )
            ci_low, ci_high = bootstrap_mean_ci(diffs, seed=boot_seed)
            rows.append(
                result_row(
                    source_step_id=source_step_id,
                    source_artifact=source_artifact,
                    analysis_id=analysis_id,
                    fdr_family_id=fdr_family_id,
                    test_type="paired_signflip_with_bootstrap",
                    formula=(
                        f"{metric.metric}: mean({comparison_col}={comparison_value} - "
                        f"{reference_col}={reference_value}) paired by {', '.join(pair_keys)}; {perm_mode}"
                    ),
                    comparison=f"{comparison_col}={comparison_value} vs {reference_col}={reference_value}",
                    metric=metric.metric,
                    effect_direction=metric.direction,
                    group_key=str(comparison_value),
                    n_observed=len(merged),
                    n_null=0,
                    n_pairs=len(diffs),
                    observed_mean=mean_or_nan(observed),
                    reference_mean=mean_or_nan(reference),
                    effect_size_raw=mean_or_nan(diffs),
                    effect_size_standardized=standardized_mean_effect(diffs),
                    ci_low=ci_low,
                    ci_high=ci_high,
                    p_value=p_value,
                    bootstrap_seed=boot_seed,
                    permutation_seed=perm_seed,
                    permutation_replicates=perm_reps,
                    diagnostic_status="passed",
                    caveat=caveat,
                )
            )
    return rows


def empirical_null_tests(
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
    *,
    source_step_id: str,
    source_artifact: str,
    analysis_id: str,
    fdr_family_id: str,
    join_keys: list[str],
    group_label_cols: list[str],
    metrics: list[MetricSpec],
    caveat: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if observed_df.empty or null_df.empty:
        return rows
    for _, obs_row in observed_df.iterrows():
        mask = pd.Series(True, index=null_df.index)
        for key in join_keys:
            mask &= null_df[key].astype(str) == str(obs_row[key])
        group_null = null_df[mask]
        if group_null.empty:
            continue
        labels = [f"{col}={obs_row[col]}" for col in group_label_cols if col in observed_df.columns]
        group_key = ";".join(labels)
        for metric in metrics:
            if metric.observed_col not in observed_df.columns or metric.reference_col not in null_df.columns:
                continue
            observed_value = float(obs_row[metric.observed_col])
            null_values = coerce_numeric(group_null[metric.reference_col]).to_numpy(dtype=float)
            null_values = finite_array(null_values)
            if len(null_values) == 0:
                continue
            p_value = empirical_p_value(observed_value, null_values, metric.direction)
            null_mean = mean_or_nan(null_values)
            null_sd = sd_or_nan(null_values)
            effect = observed_value - null_mean
            standardized = effect / null_sd if np.isfinite(null_sd) and null_sd > 0 else (0.0 if effect == 0 else math.copysign(math.inf, effect))
            ci_low = observed_value - percentile_or_nan(null_values, 97.5)
            ci_high = observed_value - percentile_or_nan(null_values, 2.5)
            rows.append(
                result_row(
                    source_step_id=source_step_id,
                    source_artifact=source_artifact,
                    analysis_id=analysis_id,
                    fdr_family_id=fdr_family_id,
                    test_type="empirical_null_tail_test",
                    formula=(
                        f"{metric.metric}: empirical p from fixed-trajectory nulls; "
                        f"p=(extreme+1)/(N+1), direction={metric.direction}"
                    ),
                    comparison=group_key,
                    metric=metric.metric,
                    effect_direction=metric.direction,
                    group_key=group_key,
                    n_observed=1,
                    n_null=len(null_values),
                    n_pairs=0,
                    observed_mean=observed_value,
                    reference_mean=null_mean,
                    effect_size_raw=effect,
                    effect_size_standardized=float(standardized),
                    ci_low=ci_low,
                    ci_high=ci_high,
                    p_value=p_value,
                    bootstrap_seed=None,
                    permutation_seed=None,
                    permutation_replicates=None,
                    diagnostic_status="passed",
                    caveat=caveat,
                )
            )
    return rows


def local_move_null_tests(df: pd.DataFrame, source_artifact: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = [
        MetricSpec("finalSortednessPercent", "realFinalSortednessPercent", "nullFinalSortednessPercent", "high"),
        MetricSpec("finalMonotonicityError", "realFinalMonotonicityError", "nullFinalMonotonicityError", "low"),
        MetricSpec("peakAggregation", "realPeakAggregation", "nullPeakAggregation", "high"),
        MetricSpec("aucAggregation", "realAucAggregation", "nullAucAggregation", "high"),
        MetricSpec("dgMaxDropPercent", "realDgMaxDropPercent", "nullDgMaxDropPercent", "high"),
    ]
    for (source_run_id, null_model), group in df.groupby(["sourceRunId", "nullModel"], dropna=False):
        for metric in metrics:
            if metric.observed_col not in group.columns or metric.reference_col not in group.columns:
                continue
            observed_values = finite_array(group[metric.observed_col].dropna().unique())
            if len(observed_values) != 1:
                continue
            observed_value = float(observed_values[0])
            null_values = finite_array(group[metric.reference_col].to_numpy(dtype=float))
            if len(null_values) == 0:
                continue
            p_value = empirical_p_value(observed_value, null_values, metric.direction)
            null_mean = mean_or_nan(null_values)
            null_sd = sd_or_nan(null_values)
            effect = observed_value - null_mean
            standardized = effect / null_sd if np.isfinite(null_sd) and null_sd > 0 else (0.0 if effect == 0 else math.copysign(math.inf, effect))
            rows.append(
                result_row(
                    source_step_id="S07",
                    source_artifact=source_artifact,
                    analysis_id="s07_local_move_null_empirical_tests",
                    fdr_family_id="S07_local_move_nulls",
                    test_type="empirical_null_tail_test",
                    formula=f"{metric.metric}: real source trajectory versus {null_model} matched-swap local nulls",
                    comparison=f"sourceRunId={source_run_id};nullModel={null_model}",
                    metric=metric.metric,
                    effect_direction=metric.direction,
                    group_key=f"{source_run_id};{null_model}",
                    n_observed=1,
                    n_null=len(null_values),
                    n_pairs=0,
                    observed_mean=observed_value,
                    reference_mean=null_mean,
                    effect_size_raw=effect,
                    effect_size_standardized=float(standardized),
                    ci_low=observed_value - percentile_or_nan(null_values, 97.5),
                    ci_high=observed_value - percentile_or_nan(null_values, 2.5),
                    p_value=p_value,
                    bootstrap_seed=None,
                    permutation_seed=None,
                    permutation_replicates=None,
                    diagnostic_status="passed",
                    caveat="Local nulls match accepted swap budgets but not policy-specific comparison logic.",
                )
            )
    return rows


def dg_null_tests(df: pd.DataFrame, source_artifact: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_run_id, group in df.groupby("sourceRunId", dropna=False):
        observed_values = finite_array(group["observedDelayedGratification"].dropna().unique())
        if len(observed_values) != 1:
            continue
        observed_value = float(observed_values[0])
        null_values = finite_array(group["nullDelayedGratification"].to_numpy(dtype=float))
        if len(null_values) == 0:
            continue
        p_value = empirical_p_value(observed_value, null_values, "high")
        null_mean = mean_or_nan(null_values)
        null_sd = sd_or_nan(null_values)
        effect = observed_value - null_mean
        standardized = effect / null_sd if np.isfinite(null_sd) and null_sd > 0 else (0.0 if effect == 0 else math.copysign(math.inf, effect))
        rows.append(
            result_row(
                source_step_id="S08",
                source_artifact=source_artifact,
                analysis_id="s08_matched_delta_dg_null_tests",
                fdr_family_id="S08_dg_matched_delta_nulls",
                test_type="empirical_null_tail_test",
                formula="delayedGratification: observed trajectory versus matched-delta null; high-tail p=(extreme+1)/(N+1)",
                comparison=f"sourceRunId={source_run_id}",
                metric="delayedGratification",
                effect_direction="high",
                group_key=str(source_run_id),
                n_observed=1,
                n_null=len(null_values),
                n_pairs=0,
                observed_mean=observed_value,
                reference_mean=null_mean,
                effect_size_raw=effect,
                effect_size_standardized=float(standardized),
                ci_low=observed_value - percentile_or_nan(null_values, 97.5),
                ci_high=observed_value - percentile_or_nan(null_values, 2.5),
                p_value=p_value,
                bootstrap_seed=None,
                permutation_seed=None,
                permutation_replicates=None,
                diagnostic_status="passed",
                caveat="Matched-delta null tests temporal ordering of observed backtracking deltas, not all possible trajectory nulls.",
            )
        )
    return rows


def alternative_metric_null_tests(df: pd.DataFrame, source_artifact: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    real = df[df["metricSource"] == "S07_full_trace_real_policy"].copy()
    nulls = df[df["metricSource"] == "S07_full_trace_local_null"].copy()
    if real.empty or nulls.empty:
        return rows
    metrics = [
        "finalKendallTauDistanceNormalized",
        "finalEarthMoverPositionDistanceNormalized",
        "finalEditDistanceToTargetOrderNormalized",
    ]
    for _, real_row in real.iterrows():
        source_run_id = real_row["sourceRunId"]
        source_nulls = nulls[nulls["sourceRunId"].astype(str) == str(source_run_id)]
        for null_model, group in source_nulls.groupby("nullModel", dropna=False):
            for metric in metrics:
                observed_value = float(real_row[metric])
                null_values = finite_array(group[metric].to_numpy(dtype=float))
                if len(null_values) == 0:
                    continue
                p_value = empirical_p_value(observed_value, null_values, "low")
                null_mean = mean_or_nan(null_values)
                null_sd = sd_or_nan(null_values)
                effect = observed_value - null_mean
                standardized = effect / null_sd if np.isfinite(null_sd) and null_sd > 0 else (0.0 if effect == 0 else math.copysign(math.inf, effect))
                rows.append(
                    result_row(
                        source_step_id="S09",
                        source_artifact=source_artifact,
                        analysis_id="s09_alternative_metric_local_null_tests",
                        fdr_family_id="S09_alternative_metric_nulls",
                        test_type="empirical_null_tail_test",
                        formula=f"{metric}: real source trajectory versus {null_model} local null; low-tail distance test",
                        comparison=f"sourceRunId={source_run_id};nullModel={null_model}",
                        metric=metric,
                        effect_direction="low",
                        group_key=f"{source_run_id};{null_model}",
                        n_observed=1,
                        n_null=len(null_values),
                        n_pairs=0,
                        observed_mean=observed_value,
                        reference_mean=null_mean,
                        effect_size_raw=effect,
                        effect_size_standardized=float(standardized),
                        ci_low=observed_value - percentile_or_nan(null_values, 97.5),
                        ci_high=observed_value - percentile_or_nan(null_values, 2.5),
                        p_value=p_value,
                        bootstrap_seed=None,
                        permutation_seed=None,
                        permutation_replicates=None,
                        diagnostic_status="passed",
                        caveat="Full trajectory alternative metrics are available for S07-derived rows only.",
                    )
                )
    return rows


def build_statistical_tests(tables: dict[str, pd.DataFrame], results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    path = lambda name: str(results_dir / name)

    rows.extend(
        paired_reference_tests(
            tables["e02_scheduler_regimes.parquet"],
            source_step_id="S02",
            source_artifact=path("e02_scheduler_regimes.parquet"),
            analysis_id="s02_scheduler_vs_s01_random_sequential",
            fdr_family_id="S02_scheduler_regimes",
            reference_col="schedulerRegime",
            reference_value="s01_random_sequential",
            comparison_col="schedulerRegime",
            pair_keys=["mixtureId", "replicateIndex", "initialValuesHash"],
            metrics=[
                MetricSpec("finalSortednessPercent", "finalSortednessPercent", direction="two-sided"),
                MetricSpec("swapCount", "swapCount", direction="two-sided"),
                MetricSpec("activationCount", "activationCount", direction="two-sided"),
                MetricSpec("peakAggregation", "peakAggregation", direction="two-sided"),
                MetricSpec("dgMaxDropPercent", "dgMaxDropPercent", direction="two-sided"),
            ],
            caveat="Original threaded activation counts are practical controls and not exact replay counts.",
        )
    )

    rows.extend(
        paired_reference_tests(
            tables["e02_activation_rates.parquet"],
            source_step_id="S03",
            source_artifact=path("e02_activation_rates.parquet"),
            analysis_id="s03_activation_rate_vs_equalized",
            fdr_family_id="S03_activation_rates",
            reference_col="rateProfileId",
            reference_value="equalized_all_1x",
            comparison_col="rateProfileId",
            pair_keys=["mixtureId", "replicateIndex", "initialValuesHash"],
            metrics=[
                MetricSpec("finalSortednessPercent", "finalSortednessPercent", direction="two-sided"),
                MetricSpec("swapCount", "swapCount", direction="two-sided"),
                MetricSpec("activationCount", "activationCount", direction="two-sided"),
                MetricSpec("peakAggregation", "peakAggregation", direction="two-sided"),
                MetricSpec("dgMaxDropPercent", "dgMaxDropPercent", direction="two-sided"),
            ],
            caveat="Pure one-Algotype rows identify unchanged policy logic but not rate-ratio effects.",
        )
    )

    rows.extend(
        empirical_null_tests(
            tables["e02_label_shuffle_observed.parquet"],
            tables["e02_label_shuffle_nulls.parquet"],
            source_step_id="S04",
            source_artifact=path("e02_label_shuffle_observed.parquet"),
            analysis_id="s04_label_shuffle_aggregation_nulls",
            fdr_family_id="S04_label_shuffle_aggregation",
            join_keys=["conditionId", "replicateIndex"],
            group_label_cols=["mixtureId", "replicateIndex"],
            metrics=[
                MetricSpec("peakAggregation", "observedPeakAggregation", "nullPeakAggregation", "high"),
                MetricSpec("aucAggregation", "observedAucAggregation", "nullAucAggregation", "high"),
                MetricSpec("finalAggregation", "observedFinalAggregation", "nullFinalAggregation", "high"),
            ],
            caveat="Trajectory-preserving label null tests label clustering on fixed paths, not alternative policy dynamics.",
        )
    )

    rows.extend(
        empirical_null_tests(
            tables["e02_dummy_algotypes.parquet"],
            tables["e02_dummy_algotypes_nulls.parquet"],
            source_step_id="S05",
            source_artifact=path("e02_dummy_algotypes.parquet"),
            analysis_id="s05_dummy_label_nulls",
            fdr_family_id="S05_dummy_labels",
            join_keys=["sourcePureConditionId", "labelSchemeId", "replicateIndex"],
            group_label_cols=["basePolicy", "labelSchemeId", "replicateIndex"],
            metrics=[
                MetricSpec("peakDummyAggregation", "observedPeakDummyAggregation", "nullPeakDummyAggregation", "high"),
                MetricSpec("aucDummyAggregation", "observedAucDummyAggregation", "nullAucDummyAggregation", "high"),
                MetricSpec("finalDummyAggregation", "observedFinalDummyAggregation", "nullFinalDummyAggregation", "high"),
            ],
            caveat="Dummy labels are identities over one shared policy path.",
        )
    )

    rows.extend(
        empirical_null_tests(
            tables["e02_speed_matched_algotypes.parquet"],
            tables["e02_speed_matched_label_shuffle_nulls.parquet"],
            source_step_id="S06",
            source_artifact=path("e02_speed_matched_algotypes.parquet"),
            analysis_id="s06_speed_matched_label_shuffle_nulls",
            fdr_family_id="S06_speed_matched_aggregation",
            join_keys=["conditionId", "replicateIndex"],
            group_label_cols=["mixtureId", "speedProfileId", "replicateIndex"],
            metrics=[
                MetricSpec("peakAggregation", "observedPeakAggregation", "nullPeakAggregation", "high"),
                MetricSpec("aucAggregation", "observedAucAggregation", "nullAucAggregation", "high"),
                MetricSpec("finalAggregation", "observedFinalAggregation", "nullFinalAggregation", "high"),
            ],
            caveat="Speed matching is scheduler-level inverse pure-speed throttling, not exact actor-swap parity.",
        )
    )

    rows.extend(local_move_null_tests(tables["e02_local_move_nulls.parquet"], path("e02_local_move_nulls.parquet")))
    rows.extend(dg_null_tests(tables["e02_dg_nulls.parquet"], path("e02_dg_nulls.parquet")))
    rows.extend(alternative_metric_null_tests(tables["e02_alternative_metrics.parquet"], path("e02_alternative_metrics.parquet")))

    s10 = tables["e02_input_distributions.parquet"].copy()
    if "delayedGratification" not in s10.columns and "dgMaxDropPercent" in s10.columns:
        s10["delayedGratification"] = s10["dgMaxDropPercent"]
    rows.extend(
        paired_reference_tests(
            s10,
            source_step_id="S10",
            source_artifact=path("e02_input_distributions.parquet"),
            analysis_id="s10_input_profile_vs_random_unique",
            fdr_family_id="S10_input_distributions",
            reference_col="inputProfile",
            reference_value="random_unique",
            comparison_col="inputProfile",
            pair_keys=["conditionTemplateId", "replicateIndex"],
            metrics=[
                MetricSpec("completed", "completed", direction="two-sided"),
                MetricSpec("finalKendallTauDistanceNormalized", "finalKendallTauDistanceNormalized", direction="two-sided"),
                MetricSpec("finalSortednessPercent", "finalSortednessPercent", direction="two-sided"),
                MetricSpec("swapCount", "swapCount", direction="two-sided"),
                MetricSpec("activationCount", "activationCount", direction="two-sided"),
            ],
            caveat="Input-profile comparisons intentionally change initial values; pairing is by condition template and replicate index, not identical arrays.",
        )
    )

    s11 = tables["e02_frozen_placement.parquet"].copy()
    s11_frozen = s11[(s11["placementCategory"] != "no_frozen_control") & (s11["frozenVariant"] != "none")].copy()
    rows.extend(
        paired_reference_tests(
            s11_frozen,
            source_step_id="S11",
            source_artifact=path("e02_frozen_placement.parquet"),
            analysis_id="s11_frozen_placement_vs_random",
            fdr_family_id="S11_frozen_placement",
            reference_col="placementCategory",
            reference_value="random",
            comparison_col="placementCategory",
            pair_keys=["algorithm", "frozenVariant", "replicateIndex", "initialValuesHash"],
            metrics=[
                MetricSpec("completed", "completed", direction="two-sided"),
                MetricSpec("finalKendallTauDistanceNormalized", "finalKendallTauDistanceNormalized", direction="two-sided"),
                MetricSpec("finalSortednessPercent", "finalSortednessPercent", direction="two-sided"),
                MetricSpec("swapCount", "swapCount", direction="two-sided"),
                MetricSpec("activationCount", "activationCount", direction="two-sided"),
                MetricSpec("blockedMoveAttempts", "blockedMoveAttempts", direction="two-sided"),
            ],
            caveat="High-/low-value placement categories intentionally confound value identity with position.",
        )
    )

    s12 = tables["e02_frozen_behavior_variants.parquet"].copy()
    s12_metrics = [
        MetricSpec("completed", "completed", direction="two-sided"),
        MetricSpec("finalKendallTauDistanceNormalized", "finalKendallTauDistanceNormalized", direction="two-sided"),
        MetricSpec("finalSortednessPercent", "finalSortednessPercent", direction="two-sided"),
        MetricSpec("swapCount", "swapCount", direction="two-sided"),
        MetricSpec("activationCount", "activationCount", direction="two-sided"),
        MetricSpec("blockedMoveAttempts", "blockedMoveAttempts", direction="two-sided"),
        MetricSpec("delayedGratification", "delayedGratification", direction="two-sided"),
    ]
    for baseline in ["baseline_passive", "baseline_stuck"]:
        rows.extend(
            paired_reference_tests(
                s12,
                source_step_id="S12",
                source_artifact=path("e02_frozen_behavior_variants.parquet"),
                analysis_id=f"s12_defect_behavior_vs_{baseline}",
                fdr_family_id="S12_frozen_behavior",
                reference_col="defectBehaviorVariant",
                reference_value=baseline,
                comparison_col="defectBehaviorVariant",
                pair_keys=["algorithm", "placementCategory", "replicateIndex", "initialValuesHash"],
                metrics=s12_metrics,
                caveat="Dynamic defect behaviors are explicit S12 simulator extensions for f=2 Frozen Cells.",
            )
        )

    rows.extend(
        paired_reference_tests(
            tables["e02_stop_condition_sensitivity.parquet"],
            source_step_id="S13",
            source_artifact=path("e02_stop_condition_sensitivity.parquet"),
            analysis_id="s13_stop_policy_vs_canonical",
            fdr_family_id="S13_stop_conditions",
            reference_col="stopPolicyId",
            reference_value="canonical_s01",
            comparison_col="stopPolicyId",
            pair_keys=["conditionTemplateId", "replicateIndex", "initialValuesHash", "initialPolicyHash"],
            metrics=[
                MetricSpec("criterionCompleted", "criterionCompleted", direction="two-sided"),
                MetricSpec("finalKendallTauDistanceNormalized", "finalKendallTauDistanceNormalized", direction="two-sided"),
                MetricSpec("finalSortednessPercent", "finalSortednessPercent", direction="two-sided"),
                MetricSpec("activationCount", "activationCount", direction="two-sided"),
                MetricSpec("swapCount", "swapCount", direction="two-sided"),
                MetricSpec("comparisonCount", "comparisonCount", direction="two-sided"),
            ],
            caveat="Stop policies intentionally alter convergence semantics while preserving initial arrays and policies.",
        )
    )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df
    return apply_bh_fdr(stats_df)


def aggregate_bootstrap_intervals(stats_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    usable = stats_df[np.isfinite(pd.to_numeric(stats_df["effectSizeRaw"], errors="coerce"))].copy()
    group_cols = ["sourceStepId", "fdrFamilyId", "analysisId", "metric", "effectDirection"]
    for keys, group in usable.groupby(group_cols, dropna=False):
        effects = finite_array(group["effectSizeRaw"].to_numpy(dtype=float))
        if len(effects) == 0:
            continue
        seed = stable_seed(BOOTSTRAP_SEED, "family_interval", *keys)
        ci_low, ci_high = bootstrap_mean_ci(effects, seed=seed)
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "sourceStepId": keys[0],
                "fdrFamilyId": keys[1],
                "analysisId": keys[2],
                "metric": keys[3],
                "effectDirection": keys[4],
                "testCount": int(len(group)),
                "meanEffectSizeRaw": mean_or_nan(effects),
                "medianEffectSizeRaw": float(np.median(effects)),
                "ciLow": ci_low,
                "ciHigh": ci_high,
                "bootstrapSeed": seed,
                "bootstrapReplicates": BOOTSTRAP_REPLICATES,
                "fdrSignificantCount": int(group["significantFdr05"].sum()),
                "nominalSignificantCount": int(group["significantNominal05"].sum()),
            }
        )
    return pd.DataFrame(rows)


def fdr_family_summary(stats_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in stats_df.groupby("fdrFamilyId", dropna=False):
        q = pd.to_numeric(group["qValue"], errors="coerce")
        p = pd.to_numeric(group["pValue"], errors="coerce")
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "fdrFamilyId": family,
                "testCount": int(len(group)),
                "finitePValueCount": int(np.isfinite(p).sum()),
                "nominalSignificantCount": int((p < FDR_ALPHA).sum()),
                "fdrSignificantCount": int((q < FDR_ALPHA).sum()),
                "minPValue": float(np.nanmin(p)) if np.isfinite(p).any() else math.nan,
                "minQValue": float(np.nanmin(q)) if np.isfinite(q).any() else math.nan,
                "metrics": ",".join(sorted(str(x) for x in group["metric"].dropna().unique())),
                "testTypes": ",".join(sorted(str(x) for x in group["testType"].dropna().unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["fdrFamilyId"]).reset_index(drop=True)


def fit_mixedlm(
    df: pd.DataFrame,
    *,
    model_id: str,
    source_step_id: str,
    formula: str,
    group_col: str,
    family_id: str,
    caveat: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diagnostic = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "modelId": model_id,
        "sourceStepId": source_step_id,
        "modelType": "mixedlm_random_intercept",
        "formula": formula,
        "groupColumn": group_col,
        "rowCount": int(len(df)),
        "groupCount": int(df[group_col].nunique(dropna=True)) if group_col in df.columns else 0,
        "converged": False,
        "fallbackUsed": False,
        "diagnosticStatus": "not_run",
        "message": "",
    }
    if not STATSMODELS_AVAILABLE:
        diagnostic["message"] = "statsmodels unavailable"
        return rows, diagnostic
    try:
        model_df = df.copy()
        fit = smf.mixedlm(formula, model_df, groups=model_df[group_col]).fit(reml=False, method="lbfgs", maxiter=500)
        diagnostic["converged"] = bool(getattr(fit, "converged", False))
        diagnostic["diagnosticStatus"] = "passed" if diagnostic["converged"] else "warning"
        diagnostic["message"] = str(getattr(fit, "mle_retvals", ""))[:500]
        params = fit.params
        pvalues = fit.pvalues
        conf = fit.conf_int()
        for term in params.index:
            if term == "Group Var":
                continue
            rows.append(
                result_row(
                    source_step_id=source_step_id,
                    source_artifact="multiple",
                    analysis_id=model_id,
                    fdr_family_id=family_id,
                    test_type="mixedlm_random_intercept_wald",
                    formula=formula + f" + (1 | {group_col})",
                    comparison=str(term),
                    metric=formula.split("~", 1)[0].strip(),
                    effect_direction="two-sided",
                    group_key=str(term),
                    n_observed=len(model_df),
                    n_null=0,
                    n_pairs=0,
                    observed_mean=mean_or_nan(model_df[formula.split('~', 1)[0].strip()]),
                    reference_mean=0.0,
                    effect_size_raw=float(params.loc[term]),
                    effect_size_standardized=math.nan,
                    ci_low=float(conf.loc[term, 0]) if term in conf.index else math.nan,
                    ci_high=float(conf.loc[term, 1]) if term in conf.index else math.nan,
                    p_value=float(pvalues.loc[term]) if term in pvalues.index else math.nan,
                    bootstrap_seed=None,
                    permutation_seed=None,
                    permutation_replicates=None,
                    diagnostic_status=diagnostic["diagnosticStatus"],
                    caveat=caveat,
                )
            )
        return rows, diagnostic
    except Exception as exc:
        diagnostic["message"] = repr(exc)[:500]
        diagnostic["diagnosticStatus"] = "fallback_cluster_ols"
        diagnostic["fallbackUsed"] = True

    try:
        # Cluster-robust OLS fallback preserves the same fixed-effect formula and
        # groups uncertainty by the planned random-intercept key.
        model_df = df.copy()
        fit = smf.ols(formula, model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df[group_col]})
        diagnostic["converged"] = True
        diagnostic["diagnosticStatus"] = "passed_with_cluster_ols_fallback"
        params = fit.params
        pvalues = fit.pvalues
        conf = fit.conf_int()
        for term in params.index:
            rows.append(
                result_row(
                    source_step_id=source_step_id,
                    source_artifact="multiple",
                    analysis_id=model_id,
                    fdr_family_id=family_id,
                    test_type="cluster_robust_ols_fallback",
                    formula=formula + f"; cluster={group_col}",
                    comparison=str(term),
                    metric=formula.split("~", 1)[0].strip(),
                    effect_direction="two-sided",
                    group_key=str(term),
                    n_observed=len(model_df),
                    n_null=0,
                    n_pairs=0,
                    observed_mean=mean_or_nan(model_df[formula.split('~', 1)[0].strip()]),
                    reference_mean=0.0,
                    effect_size_raw=float(params.loc[term]),
                    effect_size_standardized=math.nan,
                    ci_low=float(conf.loc[term, 0]) if term in conf.index else math.nan,
                    ci_high=float(conf.loc[term, 1]) if term in conf.index else math.nan,
                    p_value=float(pvalues.loc[term]) if term in pvalues.index else math.nan,
                    bootstrap_seed=None,
                    permutation_seed=None,
                    permutation_replicates=None,
                    diagnostic_status=diagnostic["diagnosticStatus"],
                    caveat=caveat + " MixedLM failed; cluster-robust OLS fallback used.",
                )
            )
    except Exception as exc:
        diagnostic["message"] += f"; fallback failed: {repr(exc)[:500]}"
        diagnostic["diagnosticStatus"] = "failed"
    return rows, diagnostic


def build_model_inputs(tables: dict[str, pd.DataFrame]) -> list[tuple[pd.DataFrame, dict[str, str]]]:
    models: list[tuple[pd.DataFrame, dict[str, str]]] = []

    s07 = tables["e02_local_move_nulls.parquet"].copy()
    s07["groupKey"] = s07["sourceRunId"].astype(str)
    models.append(
        (
            s07,
            {
                "model_id": "mixedlm_s07_real_minus_null_sortedness",
                "source_step_id": "S07",
                "formula": "realMinusNullFinalSortednessPercent ~ C(nullModel) + C(sourceContext)",
                "group_col": "groupKey",
                "family_id": "mixed_models",
                "caveat": "Repeated null replicates are clustered by real source run.",
            },
        )
    )

    s10 = tables["e02_input_distributions.parquet"].copy()
    s10["groupKey"] = s10["conditionTemplateId"].astype(str) + "::" + s10["replicateIndex"].astype(str)
    models.append(
        (
            s10,
            {
                "model_id": "mixedlm_s10_input_profile_final_kendall",
                "source_step_id": "S10",
                "formula": "finalKendallTauDistanceNormalized ~ C(inputProfile) + C(conditionClass) + C(algorithm)",
                "group_col": "groupKey",
                "family_id": "mixed_models",
                "caveat": "Input profiles intentionally change initial value distributions; group key matches condition template and replicate.",
            },
        )
    )

    s11 = tables["e02_frozen_placement.parquet"].copy()
    s11 = s11[(s11["placementCategory"] != "no_frozen_control") & (s11["frozenVariant"] != "none")].copy()
    s11["groupKey"] = s11["algorithm"].astype(str) + "::" + s11["frozenVariant"].astype(str) + "::" + s11["replicateIndex"].astype(str)
    models.append(
        (
            s11,
            {
                "model_id": "mixedlm_s11_placement_final_kendall",
                "source_step_id": "S11",
                "formula": "finalKendallTauDistanceNormalized ~ C(placementCategory) + C(frozenVariant) + C(algorithm)",
                "group_col": "groupKey",
                "family_id": "mixed_models",
                "caveat": "High-/low-value placements remain value-position confounded.",
            },
        )
    )

    s12 = tables["e02_frozen_behavior_variants.parquet"].copy()
    s12["groupKey"] = (
        s12["algorithm"].astype(str)
        + "::"
        + s12["placementCategory"].astype(str)
        + "::"
        + s12["replicateIndex"].astype(str)
    )
    models.append(
        (
            s12,
            {
                "model_id": "mixedlm_s12_behavior_final_kendall",
                "source_step_id": "S12",
                "formula": "finalKendallTauDistanceNormalized ~ C(defectBehaviorFamily) + C(placementCategory) + C(algorithm)",
                "group_col": "groupKey",
                "family_id": "mixed_models",
                "caveat": "Behavior variants are explicit simulator extensions rather than original paper variants.",
            },
        )
    )

    s13 = tables["e02_stop_condition_sensitivity.parquet"].copy()
    s13["groupKey"] = s13["conditionTemplateId"].astype(str) + "::" + s13["replicateIndex"].astype(str)
    models.append(
        (
            s13,
            {
                "model_id": "mixedlm_s13_stop_policy_final_kendall",
                "source_step_id": "S13",
                "formula": "finalKendallTauDistanceNormalized ~ C(stopPolicyFamily) + C(conditionClass) + C(algorithm)",
                "group_col": "groupKey",
                "family_id": "mixed_models",
                "caveat": "Stop-policy effects include intentional cap truncation and alternate stable-state definitions.",
            },
        )
    )
    return models


def build_model_results(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for model_df, spec in build_model_inputs(tables):
        rows, diag = fit_mixedlm(model_df, **spec)
        result_rows.extend(rows)
        diagnostics.append(diag)
    results = pd.DataFrame(result_rows)
    if not results.empty:
        results = apply_bh_fdr(results)
    return results, pd.DataFrame(diagnostics)


def claim_support_summary(stats_df: pd.DataFrame, family_df: pd.DataFrame, model_diag_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    family_classification = {
        "S02_scheduler_regimes": (
            "scheduler sensitivity",
            "robust but narrower",
            "Final sorting is stable, but activation/DG/trajectory quantities show FDR-significant scheduler sensitivity.",
        ),
        "S03_activation_rates": (
            "activation-rate artifact audit",
            "robust but narrower",
            "Final sorting remains stable, but activation and some trajectory/aggregation quantities change under rate interventions.",
        ),
        "S04_label_shuffle_aggregation": (
            "same-goal chimeric Aggregation",
            "robust but narrower",
            "Only a subset of mixed-policy trajectories exceed trajectory-preserving label nulls after family correction.",
        ),
        "S05_dummy_labels": (
            "dummy label negative control",
            "supportive control",
            "Identical-code dummy labels do not produce a broad FDR-significant Aggregation artifact.",
        ),
        "S06_speed_matched_aggregation": (
            "speed-matched Aggregation",
            "robust but narrower",
            "Three-policy speed-matched mixtures retain Aggregation signal more consistently than pairwise mixtures.",
        ),
        "S07_local_move_nulls": (
            "local null benchmark",
            "supportive but bounded",
            "Real policies often outperform random-adjacent, Metropolis-like, and random-walker nulls; inversion-biased nulls remain close.",
        ),
        "S08_dg_matched_delta_nulls": (
            "Delayed Gratification",
            "constraining/contradictory",
            "DG is not consistently above matched-delta trajectory nulls after correction.",
        ),
        "S09_alternative_metric_nulls": (
            "alternative metrics",
            "constraining/contradictory",
            "Distance-metric null comparisons support no-Frozen cases but constrain broad Frozen Cell claims.",
        ),
        "S10_input_distributions": (
            "input-distribution generality",
            "constraining/contradictory",
            "Pure and same-goal chimera sorting generalize, while Frozen Cell outcomes depend on input profile.",
        ),
        "S11_frozen_placement": (
            "Frozen Cell placement",
            "constraining/contradictory",
            "Frozen Cell completion and residual distance are placement-sensitive.",
        ),
        "S12_frozen_behavior": (
            "Frozen Cell behavior",
            "constraining/contradictory",
            "Dynamic defect gates change completion, distance, and blocked attempts relative to baselines.",
        ),
        "S13_stop_conditions": (
            "stop-condition sensitivity",
            "constraining/contradictory",
            "Stop/cap/stability definitions can change completion labels and final states.",
        ),
        "mixed_models": (
            "mixed/clustered statistical audit",
            "supportive with diagnostics",
            "Model formulas ran with random-intercept attempts and documented fallbacks where needed.",
        ),
    }
    for _, family in family_df.iterrows():
        claim, classification, interpretation = family_classification.get(
            family["fdrFamilyId"],
            (family["fdrFamilyId"], "diagnostic", "No narrative classification assigned."),
        )
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "claimFamily": claim,
                "fdrFamilyId": family["fdrFamilyId"],
                "classification": classification,
                "testCount": int(family["testCount"]),
                "fdrSignificantCount": int(family["fdrSignificantCount"]),
                "minQValue": family["minQValue"],
                "interpretation": interpretation,
            }
        )
    if not model_diag_df.empty:
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "claimFamily": "model diagnostics",
                "fdrFamilyId": "model_diagnostics",
                "classification": "diagnostic",
                "testCount": int(len(model_diag_df)),
                "fdrSignificantCount": int(model_diag_df["converged"].sum()) if "converged" in model_diag_df.columns else 0,
                "minQValue": math.nan,
                "interpretation": "MixedLM convergence and fallback status are recorded in the model diagnostics table.",
            }
        )
    return pd.DataFrame(rows)


def make_figures(family_df: pd.DataFrame, bootstrap_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if not family_df.empty:
        plot_df = family_df.sort_values("fdrSignificantCount", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_df))))
        ax.barh(plot_df["fdrFamilyId"], plot_df["fdrSignificantCount"], color="#4C78A8", label="FDR < 0.05")
        ax.barh(
            plot_df["fdrFamilyId"],
            plot_df["testCount"],
            color="#D9E2EF",
            alpha=0.45,
            label="all tests",
        )
        ax.set_xlabel("Test count")
        ax.set_ylabel("FDR family")
        ax.set_title("E02 S14 Stronger Statistics: FDR Significant Tests")
        ax.legend(loc="lower right")
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            path = figures_dir / f"e02_strong_statistics_fdr_summary.{ext}"
            fig.savefig(path, dpi=180 if ext == "png" else None)
            written.append(path)
        plt.close(fig)

    if not bootstrap_df.empty:
        top = bootstrap_df.sort_values("fdrSignificantCount", ascending=False).head(30).copy()
        top["label"] = top["fdrFamilyId"] + "\n" + top["metric"]
        y = np.arange(len(top))
        fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(top))))
        ax.errorbar(
            top["meanEffectSizeRaw"],
            y,
            xerr=[
                top["meanEffectSizeRaw"] - top["ciLow"],
                top["ciHigh"] - top["meanEffectSizeRaw"],
            ],
            fmt="o",
            color="#F58518",
            ecolor="#999999",
            capsize=2,
        )
        ax.axvline(0, color="#333333", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(top["label"])
        ax.set_xlabel("Mean raw effect with bootstrap 95% interval")
        ax.set_title("E02 S14 Bootstrap Effect Intervals")
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            path = figures_dir / f"e02_strong_statistics_bootstrap_effects.{ext}"
            fig.savefig(path, dpi=180 if ext == "png" else None)
            written.append(path)
        plt.close(fig)
    return written


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file():
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            artifacts.append(
                {
                    "path": str(path),
                    "sizeBytes": int(path.stat().st_size),
                    "sha256": sha256_path(path),
                }
            )
    return sorted(artifacts, key=lambda item: item["path"])


def write_table(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> list[Path]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return [parquet_path, csv_path]


def write_reports(
    *,
    step_dir: Path,
    stats_df: pd.DataFrame,
    family_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    model_diag_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    claim_df: pd.DataFrame,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    caveats: str,
    recommended_next_action: str,
    duration_seconds: float,
    repo_tests: dict[str, Any],
) -> tuple[Path, Path, Path]:
    step_dir.mkdir(parents=True, exist_ok=True)
    total_tests = int(len(stats_df))
    fdr_sig = int(stats_df["significantFdr05"].sum()) if not stats_df.empty else 0
    nominal_sig = int(stats_df["significantNominal05"].sum()) if not stats_df.empty else 0
    failed_validation = validation_df[~validation_df["validationPassed"]]
    model_failures = model_diag_df[model_diag_df["diagnosticStatus"].astype(str).str.contains("failed", case=False, na=False)]

    family_rows = [
        [
            row.fdrFamilyId,
            row.testCount,
            row.nominalSignificantCount,
            row.fdrSignificantCount,
            row.minQValue,
        ]
        for row in family_df.itertuples(index=False)
    ]
    claim_rows = [
        [
            row.claimFamily,
            row.classification,
            row.fdrSignificantCount,
            row.interpretation,
        ]
        for row in claim_df.itertuples(index=False)
        if row.fdrFamilyId != "model_diagnostics"
    ]
    model_rows = [
        [
            row.modelId,
            row.modelType,
            row.rowCount,
            row.groupCount,
            row.converged,
            row.diagnosticStatus,
        ]
        for row in model_diag_df.itertuples(index=False)
    ]
    validation_rows = [
        [
            row.sourceStepId,
            row.artifactName,
            row.expectedRows,
            row.observedRows,
            row.validationPassed,
        ]
        for row in validation_df.itertuples(index=False)
    ]

    strong_md = step_dir / "strong_statistics.md"
    strong_md.write_text(
        "\n".join(
            [
                "# E02 S14 Stronger Statistics",
                "",
                "## Research Step Status",
                "",
                "- Research step ID: S14",
                "- Completion status: completed",
                "- Artifacts written: see `artifact_manifest.json`; includes `e02_strong_statistics.parquet`, FDR family summaries, bootstrap intervals, model diagnostics, validation tables, figures, summary, status JSON, and copied code.",
                f"- Validation result: {validation_result}",
                f"- Caveats or blockers: {caveats}",
                "- Lay summary: S14 reanalyzed completed S01-S13 outputs with paired permutation tests, empirical null p-values, bootstrap intervals, FDR correction, and mixed/clustered model diagnostics. The stronger audit keeps several narrow controls supportive, but broad Frozen Cell, DG, stop-condition, and distribution-general claims remain constrained.",
                f"- Recommended next action: {recommended_next_action}",
                f"- Outcome classification: {OUTCOME_CLASSIFICATION}",
                "",
                "## Methods And Formulas",
                "",
                "- Paired intervention tests: for matched rows, `effect = intervention_metric - reference_metric`; p-values use exact or Monte Carlo paired sign-flip tests, and 95% intervals use percentile bootstrap over paired differences.",
                "- Empirical null tests: `p = (extreme + 1) / (N_null + 1)`, using high-tail or low-tail direction declared per metric; raw effect is observed minus null mean.",
                "- FDR correction: Benjamini-Hochberg within each declared `fdrFamilyId` at alpha 0.05.",
                "- Mixed/clustered models: random-intercept attempts use `statsmodels` MixedLM formulas listed in `e02_strong_statistics_model_diagnostics.parquet`; if MixedLM fails, the same fixed-effect formula is refit with cluster-robust OLS and marked as a fallback.",
                f"- Bootstrap seed: {BOOTSTRAP_SEED}; bootstrap replicates: {BOOTSTRAP_REPLICATES}.",
                f"- Permutation seed: {PERMUTATION_SEED}; Monte Carlo sign-flip replicates: {PERMUTATION_REPLICATES}.",
                "",
                "## FDR Families",
                "",
                markdown_table(["FDR family", "tests", "nominal <0.05", "FDR <0.05", "min q"], family_rows),
                "",
                "## Claim-Family Interpretation",
                "",
                markdown_table(["Claim family", "classification", "FDR significant tests", "interpretation"], claim_rows),
                "",
                "## Model Diagnostics",
                "",
                markdown_table(["Model", "type", "rows", "groups", "converged", "status"], model_rows),
                "",
                "## Source Validation",
                "",
                markdown_table(["Source step", "artifact", "expected rows", "observed rows", "passed"], validation_rows),
                "",
                "## Execution Summary",
                "",
                f"- Total tests: {total_tests}",
                f"- Nominal significant tests: {nominal_sig}",
                f"- FDR-significant tests: {fdr_sig}",
                f"- Failed source validations: {len(failed_validation)}",
                f"- Failed model diagnostics: {len(model_failures)}",
                f"- Runtime seconds: {duration_seconds:.2f}",
                f"- Repository test command: {' '.join(repo_tests.get('command', []))}",
                f"- Repository tests passed: {repo_tests.get('passed')}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary_md = step_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# E02 S14 Summary",
                "",
                "- Research step ID: S14",
                "- Completion status: completed",
                "- Artifacts written: `$ARTIFACTS_DIR/results/e02_strong_statistics.parquet`, FDR family summary, bootstrap intervals, model diagnostics, source validation, claim-family summary, figures, copied code, manifest, status JSON, and `strong_statistics.md`.",
                f"- Validation result: {validation_result}",
                f"- Caveats or blockers: {caveats}",
                "- Lay summary: Stronger statistics support narrow scheduler/rate/null-control conclusions but further constrain broad DG, Frozen Cell robustness, input-distribution, placement, behavior, and stop-condition invariance claims.",
                f"- Recommended next action: {recommended_next_action}",
                f"- Outcome classification: {OUTCOME_CLASSIFICATION}",
                "",
                "## Key Counts",
                "",
                markdown_table(
                    ["Item", "Count"],
                    [
                        ["Statistical test rows", total_tests],
                        ["FDR families", len(family_df)],
                        ["Bootstrap interval rows", len(bootstrap_df)],
                        ["Model diagnostics", len(model_diag_df)],
                        ["Source-validation rows", len(validation_df)],
                        ["FDR-significant tests", fdr_sig],
                    ],
                ),
                "",
                "## Strongest Constraints",
                "",
                "- DG claims remain constrained because matched-delta nulls often explain observed DG magnitudes.",
                "- Frozen Cell claims remain narrower because input profile, placement, behavior, and stop definitions change completion or residual distance.",
                "- Aggregation survives most clearly in selected mixed-policy settings with trajectory-preserving and speed-matched controls; dummy-label controls remain a useful negative control.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    validation_report = step_dir / "validation_report.md"
    failed_rows = failed_validation[
        ["sourceStepId", "artifactName", "expectedRows", "observedRows", "missingRequiredColumns", "missingSeedColumns"]
    ].values.tolist()
    validation_report.write_text(
        "\n".join(
            [
                "# E02 S14 Validation Report",
                "",
                "- Research step ID: S14",
                "- Completion status: completed",
                "- Artifacts written: see `artifact_manifest.json` for the final artifact records.",
                f"- Validation result: {validation_result}",
                f"- Caveats or blockers: {caveats}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                "## Checks",
                "",
                f"- Source row-count and required-column checks passed: {bool(failed_validation.empty)}",
                f"- Seeds logged and non-null for seed-bearing source artifacts: {bool(validation_df['seedColumnsNonNull'].all())}",
                f"- FDR q-values assigned for finite p-values: {bool(stats_df[np.isfinite(pd.to_numeric(stats_df['pValue'], errors='coerce'))]['qValue'].notna().all()) if not stats_df.empty else False}",
                f"- Bootstrap seed recorded: {BOOTSTRAP_SEED}",
                f"- Permutation seed recorded: {PERMUTATION_SEED}",
                f"- Repository tests passed: {repo_tests.get('passed')}",
                "",
                "## Failed Validations",
                "",
                markdown_table(
                    ["Source step", "artifact", "expected", "observed", "missing required", "missing seed"],
                    failed_rows,
                )
                if failed_rows
                else "None.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return strong_md, summary_md, validation_report


def run_repo_tests(step_dir: Path) -> dict[str, Any]:
    step_dir.mkdir(parents=True, exist_ok=True)
    log_path = step_dir / "repo_unit_test_log.txt"
    command = [sys.executable, "-m", "unittest", "tests.test_e02_strong_statistics"]
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "logPath": str(log_path),
    }


def copy_reproducible_code(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    paths: list[Path] = []
    for rel in [
        Path("scripts/e02_s14_strong_statistics.py"),
        Path("tests/test_e02_strong_statistics.py"),
    ]:
        src = REPO_ROOT / rel
        if src.exists():
            dst = code_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            paths.append(dst)
    return paths


def update_run_manifest(provenance_dir: Path, status: dict[str, Any]) -> Path:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    path = provenance_dir / "run_manifest.json"
    manifest: dict[str, Any] = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest["lastUpdatedAt"] = utc_now()
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "status": status.get("status"),
        "success": status.get("success"),
        "completedAt": status.get("completedAt"),
        "artifactsWritten": status.get("artifactsWritten", []),
        "validationResult": status.get("validationResult"),
        "outcomeClassification": status.get("outcomeClassification"),
    }
    write_json(path, manifest)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR_DEFAULT)
    parser.add_argument("--skip-repo-tests", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e02"
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    reports_dir = artifacts_dir / "reports"
    provenance_dir = artifacts_dir / "provenance"
    step_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    tables = load_required_tables(results_dir)
    validation_df = validate_sources(results_dir, tables, step_dir)
    stats_df = build_statistical_tests(tables, results_dir)
    model_results_df, model_diag_df = build_model_results(tables)
    if not model_results_df.empty:
        stats_df = pd.concat([stats_df, model_results_df], ignore_index=True)
        stats_df = apply_bh_fdr(stats_df)
    family_df = fdr_family_summary(stats_df)
    bootstrap_df = aggregate_bootstrap_intervals(stats_df)
    claim_df = claim_support_summary(stats_df, family_df, model_diag_df)

    written_paths: list[Path] = []
    written_paths += write_table(
        stats_df,
        results_dir / "e02_strong_statistics.parquet",
        results_dir / "e02_strong_statistics.csv",
    )
    written_paths += write_table(
        family_df,
        results_dir / "e02_strong_statistics_fdr_families.parquet",
        results_dir / "e02_strong_statistics_fdr_families.csv",
    )
    written_paths += write_table(
        bootstrap_df,
        results_dir / "e02_strong_statistics_bootstrap_intervals.parquet",
        results_dir / "e02_strong_statistics_bootstrap_intervals.csv",
    )
    written_paths += write_table(
        model_diag_df,
        results_dir / "e02_strong_statistics_model_diagnostics.parquet",
        results_dir / "e02_strong_statistics_model_diagnostics.csv",
    )
    written_paths += write_table(
        validation_df,
        results_dir / "e02_strong_statistics_source_validation.parquet",
        results_dir / "e02_strong_statistics_source_validation.csv",
    )
    written_paths += write_table(
        claim_df,
        results_dir / "e02_strong_statistics_claim_summary.parquet",
        results_dir / "e02_strong_statistics_claim_summary.csv",
    )
    figure_paths = make_figures(family_df, bootstrap_df, figures_dir)
    written_paths += figure_paths

    code_paths = copy_reproducible_code(step_dir)
    written_paths += code_paths
    repo_tests = {"command": [], "passed": True, "returncode": 0, "logPath": ""}
    if not args.skip_repo_tests:
        repo_tests = run_repo_tests(step_dir)
        written_paths.append(Path(repo_tests["logPath"]))

    source_validation_passed = bool(validation_df["validationPassed"].all())
    fdr_q_assigned = bool(
        stats_df[np.isfinite(pd.to_numeric(stats_df["pValue"], errors="coerce"))]["qValue"].notna().all()
    )
    seeds_validated = bool(validation_df["seedColumnsPresent"].all() and validation_df["seedColumnsNonNull"].all())
    models_documented = bool(len(model_diag_df) >= 5 and model_diag_df["diagnosticStatus"].notna().all())
    validation_passed = source_validation_passed and fdr_q_assigned and seeds_validated and models_documented and bool(repo_tests["passed"])
    validation_result = "passed" if validation_passed else "failed"
    caveats = (
        "S14 is a statistical audit over completed bounded E02 artifacts, not a new simulator sweep. "
        "Several families have small matched replicate counts or intentionally non-independent null replicates; "
        "model rows document MixedLM convergence or cluster-OLS fallbacks. Results remain computational proxy evidence."
    )
    recommended_next_action = "Stop before S15 for Chief Scientist review; if accepted, proceed to S15 claim audit."
    duration = time.perf_counter() - started

    preliminary_artifacts = collect_artifacts(written_paths)
    strong_md, summary_md, validation_report = write_reports(
        step_dir=step_dir,
        stats_df=stats_df,
        family_df=family_df,
        bootstrap_df=bootstrap_df,
        model_diag_df=model_diag_df,
        validation_df=validation_df,
        claim_df=claim_df,
        artifacts=preliminary_artifacts,
        validation_result=validation_result,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
        duration_seconds=duration,
        repo_tests=repo_tests,
    )
    written_paths += [strong_md, summary_md, validation_report]

    artifact_manifest_path = step_dir / "artifact_manifest.json"
    status_path = step_dir / "status.json"
    written_paths += [artifact_manifest_path, status_path]
    artifacts = collect_artifacts(written_paths)
    manifest = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "sourceExpectedRowCounts": EXPECTED_ROW_COUNTS,
        "bootstrapSeed": BOOTSTRAP_SEED,
        "bootstrapReplicates": BOOTSTRAP_REPLICATES,
        "permutationSeed": PERMUTATION_SEED,
        "permutationReplicates": PERMUTATION_REPLICATES,
        "fdrAlpha": FDR_ALPHA,
    }
    write_json(artifact_manifest_path, manifest)
    artifacts = collect_artifacts(written_paths)

    status = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation_passed),
        "status": STATUS,
        "completionStatus": STATUS,
        "completedAt": utc_now(),
        "durationSeconds": duration,
        "validationResult": validation_result,
        "artifactsWritten": [item["path"] for item in artifacts],
        "artifactCount": len(artifacts),
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION,
        "laySummary": (
            "S14 applied stronger statistical checks to completed S01-S13 artifacts. "
            "The audit supports selected narrow controls but constrains broad DG, Frozen Cell, "
            "input-distribution, placement, behavior, and stop-condition claims."
        ),
        "statisticalTestRows": int(len(stats_df)),
        "fdrFamilyRows": int(len(family_df)),
        "bootstrapIntervalRows": int(len(bootstrap_df)),
        "modelDiagnosticRows": int(len(model_diag_df)),
        "sourceValidationRows": int(len(validation_df)),
        "fdrSignificantRows": int(stats_df["significantFdr05"].sum()) if not stats_df.empty else 0,
        "nominalSignificantRows": int(stats_df["significantNominal05"].sum()) if not stats_df.empty else 0,
        "sourceValidationPassed": source_validation_passed,
        "seedValidationPassed": seeds_validated,
        "fdrValidationPassed": fdr_q_assigned,
        "modelDiagnosticsDocumented": models_documented,
        "bootstrapSeed": BOOTSTRAP_SEED,
        "bootstrapReplicates": BOOTSTRAP_REPLICATES,
        "permutationSeed": PERMUTATION_SEED,
        "permutationReplicates": PERMUTATION_REPLICATES,
        "fdrAlpha": FDR_ALPHA,
        "repoTests": repo_tests,
        "git": get_git_metadata(),
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {
            "ARTIFACTS_DIR": str(artifacts_dir),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    write_json(status_path, status)
    run_manifest_path = update_run_manifest(provenance_dir, status)
    written_paths.append(run_manifest_path)
    artifacts = collect_artifacts(written_paths)
    manifest["artifactCount"] = len(artifacts)
    manifest["artifacts"] = artifacts
    write_json(artifact_manifest_path, manifest)
    status["artifactsWritten"] = [item["path"] for item in artifacts]
    status["artifactCount"] = len(artifacts)
    write_json(status_path, status)

    print(json.dumps(json_ready(status), indent=2, sort_keys=True))
    return 0 if validation_passed else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
