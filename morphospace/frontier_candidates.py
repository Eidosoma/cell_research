"""Frontier-candidate helpers for E03 S14.

S14 curates a small, auditable set of novel policies from the S08-S13 evidence
layers, then validates the selected policies on independent seeds and larger
arrays.  The helpers here keep ranking, diversity filters, holdout summaries,
and validation checks deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .classic_comparison import S13_POLICY_METRICS
from .competence import canonical_json


FRONTIER_CANDIDATE_VERSION = "e03_s14_frontier_candidates.v1"
FRONTIER_OBJECTIVE_TAGS = (
    "classic_dominance",
    "pareto",
    "robustness",
    "delayed_gratification",
    "chimeric_compatibility",
    "energy_efficiency",
    "transfer",
    "qd_elite",
    "phase_boundary",
    "empirical_class_novelty",
    "scientifically_interesting",
    "broad_balanced",
)

S14_VECTOR_METRICS = (
    "completionSuccess",
    "finalSortednessScore",
    "energyScore",
    "robustnessScore",
    "delayedGratificationScore",
    "aggregationAucScore",
    "compatibilityScore",
    "transferScore",
    "oscillationScore",
)


@dataclass(frozen=True)
class FrontierValidationTask:
    task_id: str
    task_family: str
    task_panel: str
    input_profile: str
    initial_values: tuple[int, ...]
    scheduler_seed_base: int
    tie_seed_base: int
    frozen_variant: str = "none"
    frozen_positions: tuple[int, ...] = ()
    chimera_with_null: bool = False
    candidate_positions: tuple[int, ...] = ()
    max_activations: int = 1800
    max_swaps: int = 1800
    max_comparisons: int = 7200

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "taskFamily": self.task_family,
            "taskPanel": self.task_panel,
            "inputProfile": self.input_profile,
            "initialValues": list(self.initial_values),
            "schedulerSeedBase": int(self.scheduler_seed_base),
            "tieSeedBase": int(self.tie_seed_base),
            "frozenVariant": self.frozen_variant,
            "frozenPositions": list(self.frozen_positions),
            "chimeraWithNull": bool(self.chimera_with_null),
            "candidatePositions": list(self.candidate_positions),
            "maxActivations": int(self.max_activations),
            "maxSwaps": int(self.max_swaps),
            "maxComparisons": int(self.max_comparisons),
        }


def _numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[column].map(lambda value: bool(value) if pd.notna(value) else False)


def _safe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return default


def _rank01(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(0.0, index=series.index, dtype=float)
    ranks = numeric.rank(method="average", na_option="bottom", pct=True)
    if not higher_is_better:
        ranks = 1.0 - ranks + (1.0 / max(1, len(series)))
    return ranks.fillna(0.0).clip(0.0, 1.0)


def _mean_present(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.concat([pd.to_numeric(df[column], errors="coerce") for column in available], axis=1).mean(axis=1, skipna=True)


def summarize_pairwise_frontier_evidence(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate S13 classic/discovered pairwise comparisons by discovered policy."""

    if pairwise_df.empty:
        return pd.DataFrame(columns=["policyId"])
    df = pairwise_df.copy()
    df["policyId"] = df["discoveredPolicyId"].astype(str)
    for column in [
        "discoveredDominatesClassicBroad",
        "classicDominatesDiscoveredBroad",
        "discoveredDominatesClassicSameSeed",
        "classicDominatesDiscoveredSameSeed",
    ]:
        df[column] = _bool_series(df, column)
    numeric_cols = [
        "sameSeedNetAdvantageScore",
        "completionScoreDelta",
        "sortednessScoreDelta",
        "energyScoreDelta",
        "robustnessScoreDelta",
        "delayedGratificationScoreDelta",
        "aggregationScoreDelta",
        "oscillationStabilityScoreDelta",
        "completionSuccessDeltaMean",
        "finalSortednessScoreDeltaMean",
        "energyScoreDeltaMean",
        "robustnessScoreDeltaMean",
        "delayedGratificationScoreDeltaMean",
        "aggregationAucScoreDeltaMean",
        "oscillationScoreDeltaMean",
        "normalizedFeatureDistance",
        "pcaSeed0Distance",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    grouped = df.groupby("policyId", dropna=False)
    out = grouped.agg(
        s13ComparedClassicPolicyCount=("classicPolicyId", pd.Series.nunique),
        s13ComparedClassicFamilyCount=("classicFamily", pd.Series.nunique),
        s13DominatesBroadCount=("discoveredDominatesClassicBroad", "sum"),
        s13ClassicDominatesBroadCount=("classicDominatesDiscoveredBroad", "sum"),
        s13DominatesSameSeedCount=("discoveredDominatesClassicSameSeed", "sum"),
        s13ClassicDominatesSameSeedCount=("classicDominatesDiscoveredSameSeed", "sum"),
    ).reset_index()
    for column in numeric_cols:
        if column in df.columns:
            summary = grouped[column].agg(["mean", "max", "min"]).reset_index()
            summary.columns = [
                "policyId" if name == "policyId" else f"s13_{column}{name.capitalize()}"
                for name in summary.columns
            ]
            out = out.merge(summary, on="policyId", how="left")
    if "classicFamily" in df.columns and "discoveredDominatesClassicSameSeed" in df.columns:
        pivot = (
            df.pivot_table(index="policyId", columns="classicFamily", values="discoveredDominatesClassicSameSeed", aggfunc="sum")
            .add_prefix("s13DominatesSameSeedFamily_")
            .reset_index()
        )
        out = out.merge(pivot, on="policyId", how="left")
    return out


def summarize_phase_boundary_roles(phase_boundaries_df: pd.DataFrame) -> pd.DataFrame:
    """Count how often a policy appears on S09 boundary sides."""

    if phase_boundaries_df.empty:
        return pd.DataFrame(columns=["policyId"])
    rows: list[dict[str, Any]] = []
    transition_cols = [
        column
        for column in [
            "sortingSuccessTransition",
            "sortednessTransition",
            "delayedGratificationTransition",
            "aggregationTransition",
            "oscillationTransition",
            "robustnessTransition",
        ]
        if column in phase_boundaries_df.columns
    ]
    for side, policy_col in [("left", "leftPolicyId"), ("right", "rightPolicyId")]:
        if policy_col not in phase_boundaries_df.columns:
            continue
        for _, row in phase_boundaries_df.iterrows():
            policy_id = str(row.get(policy_col, ""))
            if not policy_id or policy_id == "nan":
                continue
            transition_kinds = [column for column in transition_cols if bool(row.get(column, False))]
            rows.append(
                {
                    "policyId": policy_id,
                    "boundarySide": side,
                    "boundaryId": str(row.get("boundaryId", "")),
                    "boundaryScore": float(row.get("boundaryScore", np.nan))
                    if pd.notna(row.get("boundaryScore", np.nan))
                    else np.nan,
                    "transitionKindCount": len(transition_kinds),
                    "transitionKindsJson": canonical_json(transition_kinds),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["policyId"])
    df = pd.DataFrame(rows)
    grouped = df.groupby("policyId", dropna=False)
    out = grouped.agg(
        s09BoundaryAppearanceCount=("boundaryId", "size"),
        s09BoundaryUniqueCount=("boundaryId", pd.Series.nunique),
        s09BoundaryScoreMean=("boundaryScore", "mean"),
        s09BoundaryScoreMax=("boundaryScore", "max"),
        s09TransitionKindCountMean=("transitionKindCount", "mean"),
        s09TransitionKindCountMax=("transitionKindCount", "max"),
    ).reset_index()
    out["s09BoundaryIdsJson"] = grouped["boundaryId"].apply(lambda values: canonical_json(sorted(set(map(str, values))))).values
    out["s09TransitionKindsJson"] = grouped["transitionKindsJson"].apply(
        lambda values: canonical_json(sorted({kind for value in values for kind in _safe_json(value, [])}))
    ).values
    return out


def summarize_s12_ablation_sources(feature_ablations_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize S12 source-policy ablation sensitivity as curation context."""

    if feature_ablations_df.empty or "sourcePolicyId" not in feature_ablations_df.columns:
        return pd.DataFrame(columns=["policyId"])
    df = feature_ablations_df.copy()
    df["policyId"] = df["sourcePolicyId"].astype(str)
    delta_cols = [
        "completionSuccessDelta",
        "finalSortednessScoreDelta",
        "energyScoreDelta",
        "robustnessScoreDelta",
        "delayedGratificationScoreDelta",
        "aggregationAucScoreDelta",
        "oscillationScoreDelta",
    ]
    original_cols = [column.removesuffix("Delta") + "Original" for column in delta_cols]
    for column in delta_cols + original_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    grouped = df.groupby("policyId", dropna=False)
    out = grouped.agg(
        s12AblationPairCount=("ablationPolicyId", "size"),
        s12AblationTypeCount=("ablationType", pd.Series.nunique),
        s12AblationTaskFamilyCount=("taskFamily", pd.Series.nunique),
    ).reset_index()
    available_deltas = [column for column in delta_cols if column in df.columns]
    available_originals = [column for column in original_cols if column in df.columns]
    if available_deltas:
        matrix = df[available_deltas]
        df["_s12MeanAblatedMinusOriginal"] = matrix.mean(axis=1, skipna=True)
        out = out.merge(grouped["_s12MeanAblatedMinusOriginal"].mean().reset_index(), on="policyId", how="left")
        out = out.rename(columns={"_s12MeanAblatedMinusOriginal": "s12MeanAblatedMinusOriginalScore"})
        out["s12FeatureSensitivityScore"] = (-pd.to_numeric(out["s12MeanAblatedMinusOriginalScore"], errors="coerce")).clip(lower=0.0)
    else:
        out["s12MeanAblatedMinusOriginalScore"] = np.nan
        out["s12FeatureSensitivityScore"] = np.nan
    if available_originals:
        df["_s12MeanOriginalScore"] = df[available_originals].mean(axis=1, skipna=True)
        out = out.merge(grouped["_s12MeanOriginalScore"].mean().reset_index(), on="policyId", how="left")
        out = out.rename(columns={"_s12MeanOriginalScore": "s12MeanOriginalScore"})
    else:
        out["s12MeanOriginalScore"] = np.nan
    return out


def _objective_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    if bool(row.get("isParetoOptimal", False)) or float(row.get("paretoEvidenceScore", 0.0)) >= 0.86:
        tags.append("pareto")
    if float(row.get("classicDominanceEvidenceScore", 0.0)) >= 0.82 or float(row.get("s13DominatesSameSeedCount", 0.0)) >= 3:
        tags.append("classic_dominance")
    if float(row.get("robustnessEvidenceScore", 0.0)) >= 0.82:
        tags.append("robustness")
    if float(row.get("dgEvidenceScore", 0.0)) >= 0.82:
        tags.append("delayed_gratification")
    if float(row.get("chimeraEvidenceScore", 0.0)) >= 0.82:
        tags.append("chimeric_compatibility")
    if float(row.get("energyEvidenceScore", 0.0)) >= 0.84:
        tags.append("energy_efficiency")
    if float(row.get("transferEvidenceScore", 0.0)) >= 0.82:
        tags.append("transfer")
    if bool(row.get("isS08Elite", False)) or float(row.get("qdEvidenceScore", 0.0)) >= 0.82:
        tags.append("qd_elite")
    if bool(row.get("isPhaseBoundaryPolicy", False)) or float(row.get("phaseBoundaryEvidenceScore", 0.0)) >= 0.82:
        tags.append("phase_boundary")
    if float(row.get("classNoveltyScore", 0.0)) >= 0.86:
        tags.append("empirical_class_novelty")
    if bool(row.get("isPathologicalPolicy", False)) or float(row.get("s12FeatureSensitivityEvidenceScore", 0.0)) >= 0.90:
        tags.append("scientifically_interesting")
    if not tags:
        tags.append("broad_balanced")
    return sorted(set(tags), key=lambda tag: FRONTIER_OBJECTIVE_TAGS.index(tag) if tag in FRONTIER_OBJECTIVE_TAGS else 99)


def build_frontier_candidate_pool(
    *,
    policy_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    qd_elites_df: pd.DataFrame,
    phase_boundaries_df: pd.DataFrame,
    class_assignments_df: pd.DataFrame,
    feature_ablations_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the S14 candidate pool from S08-S13 evidence."""

    if policy_df.empty:
        return pd.DataFrame()
    base = policy_df.copy()
    base["policyId"] = base["policyId"].astype(str)
    base["comparisonRole"] = base.get("comparisonRole", "").astype(str)
    base = base[
        base["comparisonRole"].eq("discovered")
        & ~_bool_series(base, "isClassicPolicy")
        & ~_bool_series(base, "isNullPolicy")
    ].copy()

    corpus_cols = [
        "policyId",
        "structureHash",
        "dslProgramJson",
        "dslRelativePath",
        "parentPolicyIdsJson",
        "mutationOperatorsJson",
        "recombinationParentsJson",
        "tagsJson",
        "observationRequirementsJson",
        "actionCountsJson",
    ]
    corpus = corpus_df[[column for column in corpus_cols if column in corpus_df.columns]].copy()
    corpus["policyId"] = corpus["policyId"].astype(str)
    base = base.merge(corpus, on="policyId", how="inner", validate="one_to_one", suffixes=("", "_corpus"))

    pair_summary = summarize_pairwise_frontier_evidence(pairwise_df)
    base = base.merge(pair_summary, on="policyId", how="left")

    pareto_cols = [
        "policyId",
        "paretoEligible",
        "paretoFrontRank",
        "isParetoOptimal",
        "s13CompositeScore",
    ] + [column for column in S13_POLICY_METRICS if column in pareto_df.columns]
    if not pareto_df.empty:
        pareto = pareto_df[[column for column in pareto_cols if column in pareto_df.columns]].copy()
        pareto["policyId"] = pareto["policyId"].astype(str)
        base = base.merge(pareto, on="policyId", how="left", suffixes=("", "_pareto"))
        if "s13CompositeScore_pareto" in base.columns:
            base["s13CompositeScore"] = base["s13CompositeScore"].combine_first(base["s13CompositeScore_pareto"])
            base = base.drop(columns=["s13CompositeScore_pareto"])
        for metric in S13_POLICY_METRICS:
            duplicate = f"{metric}_pareto"
            if duplicate in base.columns:
                base[metric] = base[metric].combine_first(base[duplicate])
                base = base.drop(columns=[duplicate])

    if not qd_elites_df.empty:
        qd_cols = [
            "policyId",
            "eliteRank",
            "archiveRank",
            "descriptorCellId",
            "descriptorCellLabel",
            "qualityScore",
            "noveltyScore",
            "coverageScore",
            "completionRate",
            "meanSortednessScore",
            "meanEnergyScore",
            "duplicateCompletionRate",
            "robustnessMean",
            "aggregationMean",
        ]
        qd = qd_elites_df[[column for column in qd_cols if column in qd_elites_df.columns]].copy()
        qd["policyId"] = qd["policyId"].astype(str)
        qd = qd.rename(
            columns={
                "qualityScore": "s08QdQualityScore",
                "noveltyScore": "s08QdNoveltyScore",
                "coverageScore": "s08QdCoverageScore",
                "eliteRank": "s08EliteRank",
                "archiveRank": "s08ArchiveRank",
                "descriptorCellId": "s08DescriptorCellId",
                "descriptorCellLabel": "s08DescriptorCellLabel",
            }
        )
        base = base.merge(qd, on="policyId", how="left")

    boundary_summary = summarize_phase_boundary_roles(phase_boundaries_df)
    base = base.merge(boundary_summary, on="policyId", how="left")

    if not class_assignments_df.empty:
        class_cols = [
            "policyId",
            "universalityClassId",
            "className",
            "classLabel",
            "classInterpretation",
            "empiricalCaveat",
            "distanceToCentroid",
        ]
        class_meta = class_assignments_df[[column for column in class_cols if column in class_assignments_df.columns]].copy()
        class_meta["policyId"] = class_meta["policyId"].astype(str)
        base = base.merge(class_meta, on="policyId", how="left", suffixes=("", "_s11"))
        for column in class_cols:
            duplicate = f"{column}_s11"
            if duplicate in base.columns:
                base[column] = base[column].combine_first(base[duplicate])
                base = base.drop(columns=[duplicate])

    if feature_ablations_df is not None and not feature_ablations_df.empty:
        s12_summary = summarize_s12_ablation_sources(feature_ablations_df)
        base = base.merge(s12_summary, on="policyId", how="left")

    base["frontierCandidateVersion"] = FRONTIER_CANDIDATE_VERSION
    base["paretoFrontRank"] = _numeric(base, "paretoFrontRank").fillna(9999).astype(int)
    base["isParetoOptimal"] = _bool_series(base, "isParetoOptimal")
    base["isS08Elite"] = _bool_series(base, "isS08Elite")
    base["isPhaseBoundaryPolicy"] = _bool_series(base, "isPhaseBoundaryPolicy")
    base["isPathologicalPolicy"] = _bool_series(base, "isPathologicalPolicy")
    for column in [
        "s13DominatesBroadCount",
        "s13ClassicDominatesBroadCount",
        "s13DominatesSameSeedCount",
        "s13ClassicDominatesSameSeedCount",
        "phaseBoundaryInvolvementCount",
        "s09BoundaryAppearanceCount",
        "s09BoundaryUniqueCount",
        "s12AblationPairCount",
        "s12FeatureSensitivityScore",
    ]:
        base[column] = _numeric(base, column, default=0.0).fillna(0.0)

    base["paretoEvidenceScore"] = np.where(base["isParetoOptimal"], 1.0, 1.0 / base["paretoFrontRank"].clip(lower=1))
    broad_dominance = (
        base["s13DominatesSameSeedCount"] * 1.0
        + base["s13DominatesBroadCount"] * 0.55
        - base["s13ClassicDominatesSameSeedCount"] * 0.35
        + _numeric(base, "s13_sameSeedNetAdvantageScoreMean", default=0.0).fillna(0.0) * 8.0
    )
    base["classicDominanceEvidenceScore"] = _rank01(broad_dominance)
    base["robustnessEvidenceScore"] = _rank01(
        _mean_present(base, ["robustnessScore", "s13RobustnessScoreMean", "s13_robustnessScoreDeltaMeanMean", "s08Elite_robustnessMean", "robustnessMean"])
    )
    base["dgEvidenceScore"] = _rank01(
        _mean_present(
            base,
            ["delayedGratificationScore", "s13DelayedGratificationScoreMean", "s13_delayedGratificationScoreDeltaMeanMean"],
        )
    )
    base["chimeraEvidenceScore"] = _rank01(
        _mean_present(
            base,
            [
                "aggregationScore",
                "s13_completionRateTaskFamily_chimera",
                "s13_aggregationScoreDeltaMean",
                "s13_aggregationAucScoreDeltaMeanMean",
                "aggregationMean",
            ],
        )
    )
    base["energyEvidenceScore"] = _rank01(
        _mean_present(base, ["energyScore", "s13_energyScoreMean", "s13_energyScoreDeltaMeanMean", "s13_energyScoreDeltaMean"])
    )
    base["transferEvidenceScore"] = _rank01(
        _mean_present(base, ["s13_completionRateTaskFamily_transfer", "transferScore", "s13TransferScoreMean", "completionScore"])
    )
    base["qdEvidenceScore"] = _rank01(_mean_present(base, ["s08QdQualityScore", "s08Elite_qualityScore", "s08QdNoveltyScore"]))
    base["phaseBoundaryEvidenceScore"] = _rank01(
        base["phaseBoundaryInvolvementCount"]
        + base["s09BoundaryAppearanceCount"]
        + _numeric(base, "s09BoundaryScoreMax", default=0.0).fillna(0.0)
    )
    base["classNoveltyScore"] = _rank01(_numeric(base, "distanceToCentroid", default=np.nan))
    base["s12FeatureSensitivityEvidenceScore"] = _rank01(base["s12FeatureSensitivityScore"])
    base["complexitySimplicityScore"] = _rank01(_numeric(base, "complexityScore", default=np.nan), higher_is_better=False)
    base["broadCompetenceEvidenceScore"] = _rank01(_numeric(base, "s13CompositeScore", default=np.nan))
    base["frontierCompositeScore"] = (
        0.18 * base["classicDominanceEvidenceScore"]
        + 0.15 * base["broadCompetenceEvidenceScore"]
        + 0.12 * base["paretoEvidenceScore"]
        + 0.10 * base["robustnessEvidenceScore"]
        + 0.10 * base["dgEvidenceScore"]
        + 0.10 * base["chimeraEvidenceScore"]
        + 0.09 * base["energyEvidenceScore"]
        + 0.08 * base["transferEvidenceScore"]
        + 0.03 * base["qdEvidenceScore"]
        + 0.03 * base["phaseBoundaryEvidenceScore"]
        + 0.01 * base["classNoveltyScore"]
        + 0.01 * base["complexitySimplicityScore"]
    )
    base["frontierObjectiveTags"] = base.apply(_objective_tags, axis=1)
    base["frontierObjectiveTagsJson"] = base["frontierObjectiveTags"].map(canonical_json)
    base["frontierEvidenceJson"] = base.apply(
        lambda row: canonical_json(
            {
                "frontierCompositeScore": float(row["frontierCompositeScore"]) if pd.notna(row["frontierCompositeScore"]) else None,
                "s13CompositeScore": float(row["s13CompositeScore"]) if pd.notna(row.get("s13CompositeScore")) else None,
                "paretoFrontRank": int(row["paretoFrontRank"]) if pd.notna(row.get("paretoFrontRank")) else None,
                "sameSeedDominanceCount": int(row.get("s13DominatesSameSeedCount", 0)),
                "sameSeedNetAdvantageMean": float(row.get("s13_sameSeedNetAdvantageScoreMean", np.nan))
                if pd.notna(row.get("s13_sameSeedNetAdvantageScoreMean", np.nan))
                else None,
                "className": row.get("className"),
                "phaseBoundaryInvolvementCount": float(row.get("phaseBoundaryInvolvementCount", 0.0)),
            }
        ),
        axis=1,
    )
    return base.sort_values(
        ["frontierCompositeScore", "classicDominanceEvidenceScore", "s13CompositeScore", "policyId"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _select_best_for_tag(
    pool: pd.DataFrame,
    tag: str,
    selected_ids: set[str],
    selected_hashes: set[str],
    class_counts: dict[str, int],
    *,
    max_per_class: int,
) -> pd.Series | None:
    tagged = pool[pool["frontierObjectiveTags"].map(lambda tags: tag in tags if isinstance(tags, list) else tag in _safe_json(tags, []))]
    for _, row in tagged.sort_values(["frontierCompositeScore", f"{_tag_score_column(tag)}", "policyId"], ascending=[False, False, True], kind="mergesort").iterrows():
        policy_id = str(row["policyId"])
        structure_hash = str(row.get("structureHash", policy_id))
        class_name = str(row.get("className", "unknown"))
        if policy_id in selected_ids or structure_hash in selected_hashes:
            continue
        if class_counts.get(class_name, 0) >= max_per_class:
            continue
        return row
    return None


def _tag_score_column(tag: str) -> str:
    return {
        "classic_dominance": "classicDominanceEvidenceScore",
        "pareto": "paretoEvidenceScore",
        "robustness": "robustnessEvidenceScore",
        "delayed_gratification": "dgEvidenceScore",
        "chimeric_compatibility": "chimeraEvidenceScore",
        "energy_efficiency": "energyEvidenceScore",
        "transfer": "transferEvidenceScore",
        "qd_elite": "qdEvidenceScore",
        "phase_boundary": "phaseBoundaryEvidenceScore",
        "empirical_class_novelty": "classNoveltyScore",
        "scientifically_interesting": "s12FeatureSensitivityEvidenceScore",
        "broad_balanced": "broadCompetenceEvidenceScore",
    }.get(tag, "frontierCompositeScore")


def select_frontier_candidates(
    pool_df: pd.DataFrame,
    *,
    min_count: int = 10,
    max_count: int = 20,
    target_count: int = 16,
    max_per_class: int = 5,
) -> pd.DataFrame:
    """Select a diverse 10-20 policy frontier set from a candidate pool."""

    if pool_df.empty:
        return pd.DataFrame()
    target_count = max(min_count, min(max_count, int(target_count)))
    pool = pool_df.copy()
    if "frontierObjectiveTags" not in pool.columns:
        pool["frontierObjectiveTags"] = pool["frontierObjectiveTagsJson"].map(lambda value: _safe_json(value, ["broad_balanced"]))

    selected_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_hashes: set[str] = set()
    class_counts: dict[str, int] = {}

    def add_row(row: pd.Series, reason: str) -> None:
        policy_id = str(row["policyId"])
        structure_hash = str(row.get("structureHash", policy_id))
        class_name = str(row.get("className", "unknown"))
        selected_ids.add(policy_id)
        selected_hashes.add(structure_hash)
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        payload = row.to_dict()
        payload["frontierSelectionStage"] = reason
        payload["duplicateOrNearDuplicateJustification"] = "unique_structure_hash_in_s14_selection"
        selected_rows.append(payload)

    for tag in FRONTIER_OBJECTIVE_TAGS:
        if len(selected_rows) >= target_count:
            break
        best = _select_best_for_tag(pool, tag, selected_ids, selected_hashes, class_counts, max_per_class=max_per_class)
        if best is not None:
            add_row(best, f"top_{tag}")

    if len(selected_rows) < target_count:
        for _, row in pool.sort_values(["frontierCompositeScore", "policyId"], ascending=[False, True], kind="mergesort").iterrows():
            if len(selected_rows) >= target_count:
                break
            policy_id = str(row["policyId"])
            structure_hash = str(row.get("structureHash", policy_id))
            class_name = str(row.get("className", "unknown"))
            if policy_id in selected_ids or structure_hash in selected_hashes:
                continue
            if class_counts.get(class_name, 0) >= max_per_class and len(selected_rows) >= min_count:
                continue
            add_row(row, "composite_fill")

    if len(selected_rows) < min_count:
        for _, row in pool.sort_values(["frontierCompositeScore", "policyId"], ascending=[False, True], kind="mergesort").iterrows():
            if len(selected_rows) >= min_count:
                break
            policy_id = str(row["policyId"])
            structure_hash = str(row.get("structureHash", policy_id))
            if policy_id in selected_ids or structure_hash in selected_hashes:
                continue
            add_row(row, "minimum_count_fill_relaxed_class_cap")

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        return selected
    selected = selected.sort_values(["frontierCompositeScore", "policyId"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    selected["frontierRank"] = np.arange(1, len(selected) + 1, dtype=int)
    selected["selectedForS14"] = True

    def rationale(row: pd.Series) -> str:
        tags = row.get("frontierObjectiveTags", [])
        if not isinstance(tags, list):
            tags = _safe_json(tags, [])
        score_fields = {
            tag: float(row.get(_tag_score_column(tag), np.nan)) if pd.notna(row.get(_tag_score_column(tag), np.nan)) else None
            for tag in tags[:6]
        }
        return canonical_json(
            {
                "selectionStage": row.get("frontierSelectionStage"),
                "objectiveTags": tags,
                "objectiveScores": score_fields,
                "frontierCompositeScore": float(row.get("frontierCompositeScore", np.nan))
                if pd.notna(row.get("frontierCompositeScore", np.nan))
                else None,
                "duplicateCheck": row.get("duplicateOrNearDuplicateJustification"),
            }
        )

    selected["selectionRationaleJson"] = selected.apply(rationale, axis=1)
    selected["frontierObjectiveTagsJson"] = selected["frontierObjectiveTags"].map(canonical_json)
    return selected.reset_index(drop=True)


def summarize_frontier_validation(vector_df: pd.DataFrame, run_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate S14 holdout competence vectors by frontier policy."""

    if vector_df.empty:
        return pd.DataFrame(columns=["policyId"])
    vectors = vector_df.copy()
    vectors["policyId"] = vectors["policyId"].astype(str)
    for column in S14_VECTOR_METRICS:
        if column in vectors.columns:
            vectors[column] = pd.to_numeric(vectors[column], errors="coerce")
    grouped = vectors.groupby("policyId", dropna=False)
    out = grouped.agg(
        holdoutVectorRows=("policyId", "size"),
        holdoutTaskCount=("taskId", pd.Series.nunique),
        holdoutTaskFamilyCount=("taskFamily", pd.Series.nunique),
        holdoutReplicateCount=("replicateIndex", pd.Series.nunique),
    ).reset_index()
    for metric in S14_VECTOR_METRICS:
        if metric in vectors.columns:
            summary = grouped[metric].agg(["mean", "min", "max"]).reset_index()
            summary.columns = ["policyId" if name == "policyId" else f"holdout_{metric}{name.capitalize()}" for name in summary.columns]
            out = out.merge(summary, on="policyId", how="left")
    if "taskFamily" in vectors.columns and "completionSuccess" in vectors.columns:
        pivot = (
            vectors.pivot_table(index="policyId", columns="taskFamily", values="completionSuccess", aggfunc="mean")
            .add_prefix("holdout_completionRateTaskFamily_")
            .reset_index()
        )
        out = out.merge(pivot, on="policyId", how="left")
    if not run_df.empty:
        runs = run_df.copy()
        runs["policyId"] = runs["policyId"].astype(str)
        for column in ["swapCount", "comparisonCount", "activationCount", "runtimeSeconds", "completedNumeric"]:
            if column in runs.columns:
                runs[column] = pd.to_numeric(runs[column], errors="coerce")
        run_summary = runs.groupby("policyId", dropna=False).agg(
            holdoutRunRows=("policyId", "size"),
            holdoutCompletedRunCount=("completedNumeric", "sum"),
            holdoutMeanSwapCount=("swapCount", "mean"),
            holdoutMeanComparisonCount=("comparisonCount", "mean"),
            holdoutMeanActivationCount=("activationCount", "mean"),
            holdoutMeanRuntimeSeconds=("runtimeSeconds", "mean"),
            holdoutStopReasons=("stopReason", lambda values: canonical_json(dict(pd.Series(values).astype(str).value_counts().sort_index()))),
        ).reset_index()
        out = out.merge(run_summary, on="policyId", how="left")
    candidate_cols = [
        "policyId",
        "frontierRank",
        "frontierCompositeScore",
        "frontierObjectiveTagsJson",
        "className",
        "classLabel",
        "primaryRole",
        "generationMethod",
        "lineageId",
        "structureHash",
        "selectionRationaleJson",
    ]
    meta = candidate_df[[column for column in candidate_cols if column in candidate_df.columns]].copy()
    meta["policyId"] = meta["policyId"].astype(str)
    out = out.merge(meta, on="policyId", how="left")
    score_columns = [f"holdout_{metric}Mean" for metric in S14_VECTOR_METRICS if f"holdout_{metric}Mean" in out.columns]
    out["holdoutCompositeScore"] = out[score_columns].mean(axis=1, skipna=True) if score_columns else np.nan
    return out.sort_values(["frontierRank", "policyId"], kind="mergesort").reset_index(drop=True)


def validate_frontier_outputs(
    *,
    candidate_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    validation_run_df: pd.DataFrame,
    validation_vector_df: pd.DataFrame,
    validation_summary_df: pd.DataFrame,
    dsl_index_df: pd.DataFrame,
    upstream_statuses: Mapping[str, Mapping[str, Any]],
    repo_test_payload: Mapping[str, Any],
    report_exists: bool,
    figure_paths: Sequence[str],
    s15_dir_exists: bool,
    expected_run_rows: int,
    minimum_objective_tag_count: int = 6,
    minimum_class_count: int = 3,
) -> pd.DataFrame:
    """Return S14 validation checks as a table."""

    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append({"checkId": check_id, "success": bool(success), "detail": str(detail)})

    for step in ["S08", "S09", "S10", "S11", "S12", "S13"]:
        status = upstream_statuses.get(step, {})
        add(f"upstream_{step.lower()}_success", bool(status.get("success")), f"{step} status={status.get('status')}")

    candidate_count = int(len(candidate_df))
    add("candidate_count_10_to_20", 10 <= candidate_count <= 20, f"candidateCount={candidate_count}")
    add(
        "candidate_pool_nonempty",
        not pool_df.empty and len(pool_df) >= candidate_count,
        f"poolRows={len(pool_df)} candidateRows={candidate_count}",
    )
    add(
        "novel_nonclassic_nonnul",
        bool((~_bool_series(candidate_df, "isClassicPolicy")).all() and (~_bool_series(candidate_df, "isNullPolicy")).all()),
        "selected candidates exclude classic and null policies",
    )
    add(
        "unique_policy_ids",
        candidate_df.get("policyId", pd.Series(dtype=str)).astype(str).is_unique,
        f"uniquePolicyIds={candidate_df.get('policyId', pd.Series(dtype=str)).astype(str).nunique()}",
    )
    add(
        "no_exact_duplicate_structure_hash",
        candidate_df.get("structureHash", pd.Series(dtype=str)).astype(str).is_unique,
        f"uniqueStructureHashes={candidate_df.get('structureHash', pd.Series(dtype=str)).astype(str).nunique()}",
    )
    objective_tags = sorted({tag for value in candidate_df.get("frontierObjectiveTagsJson", pd.Series(dtype=str)) for tag in _safe_json(value, [])})
    add(
        "objective_tag_diversity",
        len(objective_tags) >= minimum_objective_tag_count,
        f"objectiveTags={objective_tags}",
    )
    class_count = candidate_df.get("className", pd.Series(dtype=str)).astype(str).nunique()
    add("class_diversity", class_count >= minimum_class_count, f"classCount={class_count}")
    add(
        "dsl_files_roundtrip",
        not dsl_index_df.empty and bool(dsl_index_df.get("roundtripSuccess", pd.Series(False)).all()),
        f"dslFiles={len(dsl_index_df)}",
    )
    add(
        "holdout_run_rows_complete",
        len(validation_run_df) == expected_run_rows,
        f"rows={len(validation_run_df)} expected={expected_run_rows}",
    )
    add(
        "holdout_vector_rows_match",
        len(validation_vector_df) == len(validation_run_df) and len(validation_vector_df) == expected_run_rows,
        f"vectors={len(validation_vector_df)} runs={len(validation_run_df)}",
    )
    add(
        "holdout_value_counts_conserved",
        not validation_run_df.empty and bool(validation_run_df.get("valueCountsConserved", pd.Series(False)).all()),
        "all S14 CPU validation rows preserve input value multisets",
    )
    add(
        "independent_seed_range",
        not validation_run_df.empty and int(pd.to_numeric(validation_run_df["schedulerSeed"], errors="coerce").min()) >= 81000,
        f"minSchedulerSeed={pd.to_numeric(validation_run_df.get('schedulerSeed', pd.Series(dtype=int)), errors='coerce').min() if not validation_run_df.empty else 'NA'}",
    )
    add(
        "larger_array_tasks_present",
        not validation_run_df.empty and validation_run_df["initialValuesJson"].map(lambda value: len(_safe_json(value, []))).max() > 6,
        "holdout panel includes arrays larger than S13 n=6 maximum",
    )
    add(
        "task_family_coverage",
        {"sorting", "sorting_duplicate_values", "frozen", "transfer", "chimera"}.issubset(
            set(validation_run_df.get("taskFamily", pd.Series(dtype=str)).astype(str))
        ),
        f"taskFamilies={sorted(set(validation_run_df.get('taskFamily', pd.Series(dtype=str)).astype(str)))}",
    )
    add("validation_summary_nonempty", len(validation_summary_df) == candidate_count, f"summaryRows={len(validation_summary_df)}")
    add("frontier_report_exists", bool(report_exists), f"reportExists={report_exists}")
    add("figure_outputs", len([path for path in figure_paths if path]) >= 2, f"figures={len([path for path in figure_paths if path])}")
    add("repo_unit_tests", bool(repo_test_payload.get("success")), f"returnCode={repo_test_payload.get('returnCode')} command={repo_test_payload.get('command')}")
    add("no_s15_artifacts", not s15_dir_exists, f"s15_dir_exists={s15_dir_exists}")
    return pd.DataFrame(rows)
