"""Quality-diversity helpers for E03 S08."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .competence import canonical_json


QD_SEARCH_VERSION = "e03_s08_quality_diversity.v1"
DEFAULT_DESCRIPTOR_COLUMNS = (
    "completionRate",
    "meanSortednessScore",
    "meanEnergyScore",
    "duplicateCompletionRate",
    "robustnessMeanFilled",
    "aggregationMeanFilled",
    "complexityNormalized",
    "missingBurdenNormalized",
)
NICHING_COLUMNS = (
    "successBin",
    "efficiencyBin",
    "robustnessBin",
    "aggregationBin",
    "complexityBin",
    "supportBin",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _cell_id(row: Mapping[str, Any]) -> str:
    payload = {key: row.get(key) for key in NICHING_COLUMNS}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"qd_cell_{digest}"


def _bin_success(value: float) -> str:
    if value <= 0.0:
        return "none"
    if value < 0.5:
        return "partial"
    if value < 1.0:
        return "high"
    return "perfect"


def _bin_efficiency(value: float) -> str:
    if value >= 0.75:
        return "efficient"
    if value >= 0.45:
        return "moderate"
    return "costly"


def _bin_optional(value: float, count: int, *, high: float, mid: float, missing_label: str) -> str:
    if int(count) <= 0:
        return missing_label
    if value >= high:
        return "high"
    if value >= mid:
        return "medium"
    return "low"


def _bin_complexity(value: float) -> str:
    if value <= 4:
        return "simple"
    if value <= 8:
        return "moderate"
    return "complex"


def _bin_support(missing_reasons: str, missing_count: int) -> str:
    reasons = set(json.loads(missing_reasons or "[]"))
    if "policy_outside_s06_jax_subset" in reasons:
        return "cpu_or_future_kernel"
    if missing_count <= 2:
        return "broad_s07_coverage"
    return "jax_subset_with_deferred_tasks"


def aggregate_s07_policy_metrics(
    competence_df: pd.DataFrame,
    run_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate S07 run-level vectors into one row per base DSL policy."""

    corpus_ids = set(corpus_df["policyId"].astype(str))
    comp = competence_df[competence_df["policyId"].astype(str).isin(corpus_ids)].copy()
    comp["policyId"] = comp["policyId"].astype(str)
    all_comp = competence_df.copy()
    all_comp["policyId"] = all_comp["policyId"].astype(str)
    chimera_comp = all_comp[
        all_comp["policyId"].str.startswith("chimera_") & all_comp["policyId"].str.endswith("_null_alt")
    ].copy()
    if not chimera_comp.empty:
        chimera_comp["basePolicyId"] = (
            chimera_comp["policyId"].str.replace(r"^chimera_", "", regex=True).str.replace(r"_null_alt$", "", regex=True)
        )
    if "policyId" in run_df.columns:
        run = run_df[run_df["policyId"].astype(str).isin(corpus_ids)].copy()
        run["policyId"] = run["policyId"].astype(str)
    else:
        run = pd.DataFrame(columns=["policyId", "stopReason", "backend", "runtimeSeconds"])

    numeric_columns = [
        "completionSuccess",
        "finalSortednessScore",
        "finalMonotonicityScore",
        "energyScore",
        "swapEfficiencyScore",
        "comparisonEfficiencyScore",
        "activationEfficiencyScore",
        "robustnessScore",
        "aggregationFinal",
        "failureOscillationScore",
    ]
    for column in numeric_columns:
        if column in comp.columns:
            comp[column] = pd.to_numeric(comp[column], errors="coerce")

    grouped = comp.groupby("policyId", dropna=False)
    summary = grouped.agg(
        evaluatedRunCount=("vectorId", "count"),
        evaluatedTaskCount=("taskId", pd.Series.nunique),
        evaluatedTaskPanelCount=("taskPanel", pd.Series.nunique),
        completionRate=("completionSuccess", "mean"),
        meanSortednessScore=("finalSortednessScore", "mean"),
        meanMonotonicityScore=("finalMonotonicityScore", "mean"),
        meanEnergyScore=("energyScore", "mean"),
        meanSwapEfficiencyScore=("swapEfficiencyScore", "mean"),
        meanComparisonEfficiencyScore=("comparisonEfficiencyScore", "mean"),
        meanActivationEfficiencyScore=("activationEfficiencyScore", "mean"),
        failureOscillationScoreMean=("failureOscillationScore", "mean"),
    ).reset_index()

    panel_metrics: list[dict[str, Any]] = []
    for policy_id, group in comp.groupby("policyId", dropna=False):
        duplicate = group[group["taskFamily"].eq("sorting_duplicate_values")]
        frozen = group[group["taskFamily"].eq("frozen")]
        chimera = (
            chimera_comp[chimera_comp["basePolicyId"].eq(policy_id)]
            if not chimera_comp.empty
            else group[group["taskFamily"].eq("chimera")]
        )
        panel_metrics.append(
            {
                "policyId": policy_id,
                "duplicateRunCount": int(len(duplicate)),
                "duplicateCompletionRate": float(duplicate["completionSuccess"].mean()) if len(duplicate) else np.nan,
                "duplicateSortednessMean": float(duplicate["finalSortednessScore"].mean()) if len(duplicate) else np.nan,
                "robustnessRunCount": int(frozen["robustnessScore"].notna().sum()) if len(frozen) else 0,
                "robustnessMean": float(frozen["robustnessScore"].mean()) if frozen["robustnessScore"].notna().any() else np.nan,
                "chimeraRunCount": int(len(chimera)),
                "aggregationMean": float(chimera["aggregationFinal"].mean()) if chimera["aggregationFinal"].notna().any() else np.nan,
            }
        )
    summary = summary.merge(pd.DataFrame(panel_metrics), on="policyId", how="left")

    runtime_summary = (
        run.groupby("policyId", dropna=False)
        .agg(
            stopReasonsJson=("stopReason", lambda values: canonical_json(sorted(set(map(str, values))))),
            backendJson=("backend", lambda values: canonical_json(sorted(set(map(str, values))))),
            meanRuntimeSeconds=("runtimeSeconds", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(runtime_summary, on="policyId", how="left")

    missing = missing_df.copy()
    if missing.empty:
        missing_summary = pd.DataFrame(columns=["policyId", "missingPolicyTaskCount", "missingReasonCount", "missingReasonsJson"])
    else:
        missing["policyId"] = missing["policyId"].astype(str)
        missing_summary = (
            missing.groupby("policyId", dropna=False)
            .agg(
                missingPolicyTaskCount=("taskId", "count"),
                missingReasonCount=("missingReason", pd.Series.nunique),
                missingReasonsJson=("missingReason", lambda values: canonical_json(sorted(set(map(str, values))))),
            )
            .reset_index()
        )
    summary = summary.merge(missing_summary, on="policyId", how="left")
    summary["missingPolicyTaskCount"] = pd.to_numeric(summary["missingPolicyTaskCount"], errors="coerce").fillna(0).astype(int)
    summary["missingReasonCount"] = pd.to_numeric(summary["missingReasonCount"], errors="coerce").fillna(0).astype(int)
    summary["missingReasonsJson"] = summary["missingReasonsJson"].fillna("[]")

    metadata_columns = [
        "policyId",
        "family",
        "generationMethod",
        "lineageId",
        "lineageDepth",
        "parentPolicyIdsJson",
        "mutationOperatorsJson",
        "recombinationParentsJson",
        "structureHash",
        "generationIndex",
        "description",
        "tagsJson",
        "observationRequirementsJson",
        "dslProgramJson",
        "dslRelativePath",
        "ruleCount",
        "predicateCount",
        "updateCount",
        "stateKeyCount",
        "stochasticActionCount",
        "targetUpdateCount",
        "signalCount",
        "complexityScore",
        "actionCountsJson",
    ]
    metadata = corpus_df[[column for column in metadata_columns if column in corpus_df.columns]].copy()
    metadata["policyId"] = metadata["policyId"].astype(str)
    summary = metadata.merge(summary, on="policyId", how="inner")

    for column in [
        "completionRate",
        "meanSortednessScore",
        "meanMonotonicityScore",
        "meanEnergyScore",
        "duplicateCompletionRate",
        "duplicateSortednessMean",
        "robustnessMean",
        "aggregationMean",
        "failureOscillationScoreMean",
    ]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")

    max_complexity = max(1.0, float(pd.to_numeric(summary["complexityScore"], errors="coerce").max()))
    max_missing = max(1.0, float(summary["missingPolicyTaskCount"].max()))
    summary["robustnessMeanFilled"] = summary["robustnessMean"].fillna(0.5)
    summary["aggregationMeanFilled"] = summary["aggregationMean"].fillna(0.5)
    summary["duplicateCompletionRate"] = summary["duplicateCompletionRate"].fillna(summary["completionRate"])
    summary["duplicateSortednessMean"] = summary["duplicateSortednessMean"].fillna(summary["meanSortednessScore"])
    summary["complexityNormalized"] = (pd.to_numeric(summary["complexityScore"], errors="coerce") / max_complexity).clip(0.0, 1.0)
    summary["missingBurdenNormalized"] = (summary["missingPolicyTaskCount"] / max_missing).clip(0.0, 1.0)
    summary["coverageScore"] = (summary["evaluatedTaskPanelCount"] / max(1.0, float(summary["evaluatedTaskPanelCount"].max()))).clip(0.0, 1.0)

    summary["qualityScore"] = (
        0.30 * summary["meanSortednessScore"].fillna(0.0)
        + 0.20 * summary["completionRate"].fillna(0.0)
        + 0.15 * summary["meanEnergyScore"].fillna(0.0)
        + 0.10 * summary["duplicateSortednessMean"].fillna(0.0)
        + 0.10 * summary["robustnessMeanFilled"].fillna(0.0)
        + 0.05 * summary["aggregationMeanFilled"].fillna(0.0)
        + 0.05 * summary["coverageScore"].fillna(0.0)
        + 0.05 * (1.0 - summary["complexityNormalized"].fillna(1.0))
    )
    summary["qualityScore"] = summary["qualityScore"].clip(0.0, 1.0)
    return assign_descriptor_bins(add_novelty_scores(summary))


def add_novelty_scores(
    policy_summary: pd.DataFrame,
    descriptor_columns: Sequence[str] = DEFAULT_DESCRIPTOR_COLUMNS,
    *,
    k: int = 5,
) -> pd.DataFrame:
    df = policy_summary.copy()
    available = [column for column in descriptor_columns if column in df.columns]
    if df.empty or not available:
        df["noveltyScore"] = []
        return df
    values = df[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(df) == 1:
        df["noveltyScore"] = 0.0
        return df
    distances = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.nan)
    neighbor_count = max(1, min(int(k), len(df) - 1))
    sorted_distances = np.sort(distances, axis=1)
    df["noveltyScore"] = np.nanmean(sorted_distances[:, :neighbor_count], axis=1)
    max_novelty = max(float(df["noveltyScore"].max()), 1e-12)
    df["noveltyScore"] = (df["noveltyScore"] / max_novelty).clip(0.0, 1.0)
    return df


def assign_descriptor_bins(policy_summary: pd.DataFrame) -> pd.DataFrame:
    df = policy_summary.copy()
    df["successBin"] = df["completionRate"].fillna(0.0).map(_bin_success)
    df["efficiencyBin"] = df["meanEnergyScore"].fillna(0.0).map(_bin_efficiency)
    df["robustnessBin"] = [
        _bin_optional(score, count, high=0.75, mid=0.4, missing_label="not_measured")
        for score, count in zip(df["robustnessMeanFilled"], df["robustnessRunCount"].fillna(0).astype(int))
    ]
    df["aggregationBin"] = [
        _bin_optional(score, count, high=0.75, mid=0.4, missing_label="not_measured")
        for score, count in zip(df["aggregationMeanFilled"], df["chimeraRunCount"].fillna(0).astype(int))
    ]
    df["complexityBin"] = df["complexityScore"].fillna(0).map(_bin_complexity)
    df["supportBin"] = [
        _bin_support(reasons, count)
        for reasons, count in zip(df["missingReasonsJson"], df["missingPolicyTaskCount"].fillna(0).astype(int))
    ]
    df["descriptorCellId"] = [_cell_id(row) for row in df.to_dict("records")]
    df["descriptorCellLabel"] = df[list(NICHING_COLUMNS)].astype(str).agg("|".join, axis=1)
    return df


def build_map_elites_archive(
    policy_summary: pd.DataFrame,
    *,
    max_elites: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one elite per descriptor cell and return selected bounded elites."""

    if policy_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    sort_columns = [
        "descriptorCellId",
        "qualityScore",
        "noveltyScore",
        "coverageScore",
        "meanSortednessScore",
        "policyId",
    ]
    ascending = [True, False, False, False, False, True]
    ordered = policy_summary.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    occupancy = ordered.groupby("descriptorCellId", dropna=False).size().rename("cellOccupancy").reset_index()
    archive = ordered.groupby("descriptorCellId", dropna=False).head(1).copy()
    archive = archive.merge(occupancy, on="descriptorCellId", how="left")
    archive = archive.sort_values(
        ["qualityScore", "noveltyScore", "coverageScore", "descriptorCellId", "policyId"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    archive["archiveRank"] = np.arange(1, len(archive) + 1)
    elites = archive.head(int(max_elites)).copy()
    elites["eliteRank"] = np.arange(1, len(elites) + 1)
    elites["qdSearchVersion"] = QD_SEARCH_VERSION
    archive["qdSearchVersion"] = QD_SEARCH_VERSION
    return archive, elites


def run_qd_smoke_check(policy_summary: pd.DataFrame, *, sample_size: int = 16, max_elites: int = 5) -> pd.DataFrame:
    """Run a tiny deterministic archive construction before the full S08 search."""

    sample = policy_summary.sort_values(["policyId"], kind="mergesort").head(int(sample_size))
    archive, elites = build_map_elites_archive(sample, max_elites=max_elites)
    checks = [
        {
            "checkId": "smoke_policy_sample_nonempty",
            "success": len(sample) > 0,
            "detail": f"{len(sample)} policies included in smoke sample",
        },
        {
            "checkId": "smoke_archive_nonempty",
            "success": len(archive) > 0,
            "detail": f"{len(archive)} archive cells produced",
        },
        {
            "checkId": "smoke_elites_bounded",
            "success": 0 < len(elites) <= int(max_elites),
            "detail": f"{len(elites)} smoke elites with max_elites={max_elites}",
        },
        {
            "checkId": "smoke_descriptor_cells_unique",
            "success": archive["descriptorCellId"].is_unique if len(archive) else False,
            "detail": "one smoke elite per descriptor cell",
        },
    ]
    return pd.DataFrame(checks)


def qd_config_dict(*, max_elites: int, smoke_sample_size: int, smoke_max_elites: int) -> dict[str, Any]:
    return {
        "qdSearchVersion": QD_SEARCH_VERSION,
        "archiveMethod": "deterministic_map_elites_over_s07_policy_summaries",
        "descriptorColumns": list(DEFAULT_DESCRIPTOR_COLUMNS),
        "nichingColumns": list(NICHING_COLUMNS),
        "qualityScore": {
            "meanSortednessScore": 0.30,
            "completionRate": 0.20,
            "meanEnergyScore": 0.15,
            "duplicateSortednessMean": 0.10,
            "robustnessMeanFilled": 0.10,
            "aggregationMeanFilled": 0.05,
            "coverageScore": 0.05,
            "oneMinusComplexityNormalized": 0.05,
        },
        "noveltyScore": {
            "method": "mean_distance_to_5_nearest_policies_in_descriptor_space",
            "tieBreakUse": "secondary after qualityScore within archive ranking",
        },
        "maxElites": int(max_elites),
        "smokeSampleSize": int(smoke_sample_size),
        "smokeMaxElites": int(smoke_max_elites),
        "candidateBoundary": "base S05 DSL policies with at least one S07 competence vector; S07 chimera pair IDs are excluded from elite DSL output",
    }


def validate_qd_archive(policy_summary: pd.DataFrame, archive: pd.DataFrame, elites: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "checkId": "policy_summary_nonempty",
            "success": len(policy_summary) > 0,
            "detail": f"{len(policy_summary)} S07-evaluated base policies summarized",
        },
        {
            "checkId": "archive_cells_unique",
            "success": len(archive) > 0 and archive["descriptorCellId"].is_unique,
            "detail": f"{len(archive)} archive cells",
        },
        {
            "checkId": "elites_nonempty_bounded",
            "success": 0 < len(elites) <= 32,
            "detail": f"{len(elites)} selected elites",
        },
        {
            "checkId": "elite_policy_ids_unique",
            "success": len(elites) > 0 and elites["policyId"].is_unique,
            "detail": "selected elite policy IDs are unique",
        },
        {
            "checkId": "quality_scores_bounded",
            "success": bool(policy_summary["qualityScore"].between(0.0, 1.0).all()),
            "detail": "all quality scores are in [0,1]",
        },
        {
            "checkId": "novelty_scores_bounded",
            "success": bool(policy_summary["noveltyScore"].between(0.0, 1.0).all()),
            "detail": "all novelty scores are in [0,1]",
        },
    ]
    return pd.DataFrame(checks)
