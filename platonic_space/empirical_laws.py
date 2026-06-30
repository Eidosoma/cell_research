from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .world_schema import compact_json


EMPIRICAL_LAW_SCHEMA_VERSION = "e07_s14_empirical_laws.v1"
EMPIRICAL_LAW_MODEL_VERSION = "e07_s14_bounded_synthesis.v1"
EMPIRICAL_LAW_CLAIM_BOUNDARY = (
    "Bounded empirical computational regularities over S01-S13 simulation artifacts and selected upstream "
    "E04/E06 summary artifacts only. These laws are candidate simulator-scope rules, not universal biological "
    "laws, causal proof, clinical advice, cognition, agency, sentience, or evidence about living systems."
)

REQUIRED_STEPS = tuple(f"S{index:02d}" for index in range(1, 14))


@dataclass(frozen=True)
class EmpiricalLawSynthesis:
    anchor_metrics: Mapping[str, Any]
    law_catalog: pd.DataFrame
    evidence_links: pd.DataFrame
    counterexamples: pd.DataFrame
    uncertainty_register: pd.DataFrame
    falsification_register: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _num(value: Any, digits: int = 6) -> float | None:
    out = _finite(value)
    return None if not math.isfinite(out) else round(out, digits)


def _parse_first_int(text: Any, default: int | None = None) -> int | None:
    match = re.search(r"\d+", str(text or ""))
    return int(match.group(0)) if match else default


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def dataframe_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
            out[column] = out[column].map(
                lambda value: compact_json(sorted(value) if isinstance(value, set) else value)
                if isinstance(value, (dict, list, tuple, set))
                else value
            )
        if out[column].dtype == "object":
            non_null = out[column].dropna()
            observed_types = {type(value) for value in non_null}
            if len(observed_types) > 1:
                out[column] = out[column].map(lambda value: None if value is None or pd.isna(value) else str(value))
    return out


def load_step_statuses(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        step: _read_json(artifacts_dir / "research_steps" / step / "status.json")
        for step in REQUIRED_STEPS
    }


def build_anchor_metrics(
    artifacts_dir: Path,
    previous_artifacts_dir: Path = Path("/previous-artifacts"),
) -> dict[str, Any]:
    """Build compact numerical anchors used by the S14 law catalog."""

    step_dir = artifacts_dir / "research_steps"
    statuses = load_step_statuses(artifacts_dir)

    s05_status = statuses.get("S05", {})
    s05_baseline = _read_parquet(step_dir / "S05" / "baseline_comparison.parquet")
    s05_test = s05_baseline[s05_baseline.get("partition", pd.Series(dtype=str)).astype(str).eq("test")].copy()
    s05_counts = {
        baseline: {
            "beats": int(group["modelBeatsBaseline"].astype(bool).sum()),
            "comparisons": int(len(group)),
        }
        for baseline, group in s05_test.groupby("baselineName")
    }

    s06_retrieval = _read_parquet(step_dir / "S06" / "retrieval_metrics.parquet")
    s06_baselines = _read_parquet(step_dir / "S06" / "baseline_comparison.parquet")
    s06_stability = _read_parquet(step_dir / "S06" / "stability_metrics.parquet")
    s06_k5 = s06_retrieval[
        s06_retrieval["validationTask"].astype(str).eq("heldout_behavior_profile_retrieval")
        & s06_retrieval["modelName"].astype(str).eq("weighted_behavior_pca")
        & s06_retrieval["k"].astype(int).eq(5)
    ]
    s06_k5_baseline = s06_baselines[s06_baselines["k"].astype(int).eq(5)].copy()

    s07_retrieval = _read_parquet(step_dir / "S07" / "retrieval_metrics.parquet")
    s07_baselines = _read_parquet(step_dir / "S07" / "baseline_comparison.parquet")
    s07_stability = _read_parquet(step_dir / "S07" / "stability_metrics.parquet")
    s07_k3 = s07_retrieval[
        s07_retrieval["validationTask"].astype(str).eq("heldout_goal_behavior_profile_retrieval")
        & s07_retrieval["modelName"].astype(str).eq("weighted_goal_behavior_pca")
        & s07_retrieval["k"].astype(int).eq(3)
    ]
    s07_k3_baseline = s07_baselines[s07_baselines["k"].astype(int).eq(3)].copy()

    s08_distances = _read_parquet(step_dir / "S08" / "platonic_distances.parquet")
    s08_validation = _read_parquet(step_dir / "S08" / "distance_validation.parquet")
    s08_sanity = _read_parquet(step_dir / "S08" / "nearest_neighbor_sanity.parquet")

    s09_comparison = _read_parquet(step_dir / "S09" / "model_comparison.parquet")
    s09_targets = int(len(s09_comparison))
    s09_invariant_beats_global = int((pd.to_numeric(s09_comparison.get("invariantMinusGlobalR2", pd.Series(dtype=float)), errors="coerce") > 0).sum())
    s09_combined_beats_confound = int((pd.to_numeric(s09_comparison.get("combinedMinusConfoundR2", pd.Series(dtype=float)), errors="coerce") > 0).sum())

    s10_stability = _read_parquet(step_dir / "S10" / "class_stability.parquet")
    s10_summary = _read_parquet(step_dir / "S10" / "class_summary.parquet")
    s10_class_count = int(len(s10_summary))

    s11_status = statuses.get("S11", {})
    s11_perf = s11_status.get("performanceSummary", {}) if isinstance(s11_status.get("performanceSummary"), Mapping) else {}
    s11_target_summary = _read_parquet(step_dir / "S11" / "target_performance_summary.parquet")

    s12_profile = _read_parquet(step_dir / "S12" / "profile_validation_summary.parquet")
    s12_nearest = _read_parquet(step_dir / "S12" / "nearest_existing_policy_comparison.parquet")

    s13_summary = _read_parquet(step_dir / "S13" / "s08_distance_transfer_prediction.parquet")
    s13_corr = _read_parquet(step_dir / "S13" / "s08_distance_correlation.parquet")
    s13_candidate = s13_summary[
        s13_summary.get("mappingKind", pd.Series(dtype=str)).astype(str).eq("s12_designed_policy_transfer")
        & s13_summary.get("transferTargetRole", pd.Series(dtype=str)).astype(str).eq("transfer_holdout")
    ].copy()
    s13_composite_retention = s13_corr[
        s13_corr.get("predictor", pd.Series(dtype=str)).astype(str).eq("s08CompositeTransferDistance")
        & s13_corr.get("outcome", pd.Series(dtype=str)).astype(str).eq("competenceRetentionRatio")
    ]

    e04_memory = _read_parquet(previous_artifacts_dir / "E04" / "research_steps" / "S09" / "memory_ablation_summary.parquet")
    e04_comm = _read_parquet(previous_artifacts_dir / "E04" / "research_steps" / "S10" / "communication_ablation_summary.parquet")
    e04_field = _read_parquet(previous_artifacts_dir / "E04" / "research_steps" / "S12" / "field_predictor_summary.parquet")
    e04_overfit = _read_parquet(previous_artifacts_dir / "E04" / "research_steps" / "S13" / "overfitting_group_summary.parquet")
    e04_central = _read_parquet(previous_artifacts_dir / "E04" / "research_steps" / "S14" / "centralized_group_summary.parquet")
    e04_status_s15 = _read_json(previous_artifacts_dir / "E04" / "research_steps" / "S15" / "status.json")

    e06_compat = _read_parquet(previous_artifacts_dir / "E06" / "research_steps" / "S05" / "compatibility_outcome_summary.parquet")
    e06_dominance = _read_parquet(previous_artifacts_dir / "E06" / "research_steps" / "S06" / "dominance_pair_summary.parquet")
    e06_intervention = _read_parquet(previous_artifacts_dir / "E06" / "research_steps" / "S14" / "heldout_intervention_validation.parquet")

    e04_memory_cell = e04_memory[e04_memory.get("ablationAxis", pd.Series(dtype=str)).astype(str).eq("cell_memory_capacity")]
    e04_memory_signal = e04_memory[e04_memory.get("ablationAxis", pd.Series(dtype=str)).astype(str).eq("signal_field_memory_capacity")]
    e04_comm_task = (
        e04_comm.groupby("taskFamily").agg(
            meanScoreDelta=("meanScoreDelta", "mean"),
            improvedScoreFraction=("improvedScoreFraction", "mean"),
            meanFalseAlarmDelta=("meanFalseAlarmDelta", "mean"),
            meanEnergyDelta=("meanEnergyDelta", "mean"),
        )
        if not e04_comm.empty and "taskFamily" in e04_comm.columns
        else pd.DataFrame()
    )
    e06_context_dependent = int((e06_dominance.get("contextStableWinner", pd.Series(dtype=bool)).astype(bool) == False).sum()) if not e06_dominance.empty else 0
    e06_stable = int((e06_dominance.get("contextStableWinner", pd.Series(dtype=bool)).astype(bool) == True).sum()) if not e06_dominance.empty else 0

    return {
        "schemaVersion": EMPIRICAL_LAW_SCHEMA_VERSION,
        "statusSuccessCount": int(sum(bool(status.get("success")) for status in statuses.values())),
        "statusCount": int(len(statuses)),
        "s01WorldRecords": _parse_first_int(statuses.get("S01", {}).get("validationResult"), 0),
        "s02PolicyRecords": _parse_first_int(statuses.get("S02", {}).get("validationResult"), 0),
        "s03GoalRecords": _parse_first_int(statuses.get("S03", {}).get("validationResult"), 0),
        "s04BehaviorRecords": _parse_first_int(statuses.get("S04", {}).get("validationResult"), 0),
        "s05": {
            "testBeatsGlobalMean": s05_counts.get("global_mean", {}).get("beats"),
            "testGlobalComparisons": s05_counts.get("global_mean", {}).get("comparisons"),
            "testBeatsSourceGroupMean": s05_counts.get("source_group_mean", {}).get("beats"),
            "testSourceGroupComparisons": s05_counts.get("source_group_mean", {}).get("comparisons"),
            "medianTestR2": _num(s05_status.get("performanceSummary", {}).get("medianTestR2") if isinstance(s05_status.get("performanceSummary"), Mapping) else None),
        },
        "s06": {
            "policyCount": _parse_first_int(statuses.get("S06", {}).get("validationResult"), 0),
            "heldoutK5MeanCosine": _num(s06_k5["meanCosineToHeldoutProfile"].iloc[0]) if not s06_k5.empty else None,
            "k5BaselineCosines": {
                str(row["baselineName"]): _num(row["baselineMeanCosine"])
                for _, row in s06_k5_baseline.iterrows()
            },
            "splitHalfDistanceSpearman": _num(s06_stability["pairwiseDistanceSpearman"].iloc[0]) if not s06_stability.empty else None,
        },
        "s07": {
            "goalCount": _parse_first_int(statuses.get("S07", {}).get("validationResult"), 0),
            "heldoutK3MeanCosine": _num(s07_k3["meanCosineToHeldoutProfile"].iloc[0]) if not s07_k3.empty else None,
            "k3BaselineCosines": {
                str(row["baselineName"]): _num(row["baselineMeanCosine"])
                for _, row in s07_k3_baseline.iterrows()
            },
            "splitHalfDistanceSpearman": _num(s07_stability["pairwiseDistanceSpearman"].iloc[0]) if not s07_stability.empty else None,
        },
        "s08": {
            "pairwiseDistanceRows": int(len(s08_distances)),
            "validationChecks": int(len(s08_validation)),
            "policyNearestNeighborK10PrecisionSource": _num(
                s08_sanity[
                    s08_sanity.get("entityType", pd.Series(dtype=str)).astype(str).eq("policy")
                    & s08_sanity.get("label", pd.Series(dtype=str)).astype(str).eq("sourceExperimentId")
                    & s08_sanity.get("k", pd.Series(dtype=int)).astype(int).eq(10)
                ]["precisionAtK"].iloc[0]
            )
            if not s08_sanity.empty
            else None,
        },
        "s09": {
            "eligibleTargets": s09_targets,
            "invariantBeatsGlobalCount": s09_invariant_beats_global,
            "combinedBeatsConfoundCount": s09_combined_beats_confound,
            "worstInvariantMinusGlobalR2": _num(s09_comparison["invariantMinusGlobalR2"].min()) if not s09_comparison.empty else None,
        },
        "s10": {
            "classCount": s10_class_count,
            "sameClassFractionAt10": _num(
                s10_stability[s10_stability["metricName"].astype(str).eq("same_class_fraction_at_10")]["value"].iloc[0]
            )
            if not s10_stability.empty
            else None,
            "sameClassRandomBaselineAt10": _num(
                s10_stability[s10_stability["metricName"].astype(str).eq("same_class_fraction_at_10")]["baselineValue"].iloc[0]
            )
            if not s10_stability.empty
            else None,
            "subsampleAri": _num(
                s10_stability[s10_stability["metricName"].astype(str).eq("adjusted_rand_index_vs_full_labels")]["value"].iloc[0]
            )
            if not s10_stability.empty
            else None,
            "seedAri": _num(
                s10_stability[s10_stability["metricName"].astype(str).eq("adjusted_rand_index_vs_selected_seed")]["value"].iloc[0]
            )
            if not s10_stability.empty
            else None,
            "s08AverageLinkageAri": _num(
                s10_stability[s10_stability["metricName"].astype(str).eq("adjusted_rand_index_vs_s08_average_linkage")]["value"].iloc[0]
            )
            if not s10_stability.empty
            else None,
        },
        "s11": {
            "candidateCount": int(s11_perf.get("candidateCount", len(s11_target_summary))),
            "withinS05P95Fraction": _num(s11_perf.get("overallWithinS05P95Fraction")),
            "sourceTableVerifiedFraction": _num(s11_perf.get("overallSourceTableVerifiedFraction")),
            "targetFamiliesBeatingGlobalMeanCount": int(s11_perf.get("targetFamiliesBeatingGlobalMeanCount", 0)),
            "targetFamilyCount": int(s11_perf.get("targetFamilyCount", len(s11_target_summary))),
        },
        "s12": {
            "profileCount": int(len(s12_profile)),
            "selectedDesignCount": int(pd.to_numeric(s12_profile.get("selectedDesignCount", pd.Series(dtype=float)), errors="coerce").sum()) if not s12_profile.empty else 0,
            "validatedDesignCount": int(pd.to_numeric(s12_profile.get("validatedDesignCount", pd.Series(dtype=float)), errors="coerce").sum()) if not s12_profile.empty else 0,
            "nearestExactBehaviorMatches": int(s12_nearest.get("exactBehaviorMatchOnPanel", pd.Series(dtype=bool)).astype(bool).sum()) if not s12_nearest.empty else 0,
            "nearestNearBubbleCountAt002": int((pd.to_numeric(s12_nearest.get("behaviorDistanceOnHeldoutPanel", pd.Series(dtype=float)), errors="coerce") <= 0.02).sum()) if not s12_nearest.empty else 0,
            "nearestMedianDistance": _num(pd.to_numeric(s12_nearest.get("behaviorDistanceOnHeldoutPanel", pd.Series(dtype=float)), errors="coerce").median()) if not s12_nearest.empty else None,
        },
        "s13": {
            "candidateTargetSummaryRows": int(len(s13_candidate)),
            "medianS12HeldoutRetention": _num(s13_candidate.get("competenceRetentionRatio", pd.Series(dtype=float)).median()) if not s13_candidate.empty else None,
            "beatsRandomOrNoTransferFraction": _num(s13_candidate.get("beatsRandomOrNoTransferControl", pd.Series(dtype=bool)).astype(bool).mean()) if not s13_candidate.empty else None,
            "beatsBubbleLikeBaselineFraction": _num(s13_candidate.get("beatsBubbleLikeBaseline", pd.Series(dtype=bool)).astype(bool).mean()) if not s13_candidate.empty else None,
            "s12CandidateFailureModes": sorted(set(s13_candidate.get("failureMode", pd.Series(dtype=str)).astype(str))) if not s13_candidate.empty else [],
            "compositeDistanceRetentionSpearman": _num(s13_composite_retention["spearmanR"].iloc[0]) if not s13_composite_retention.empty else None,
            "compositeDistanceRetentionP": _num(s13_composite_retention["spearmanP"].iloc[0]) if not s13_composite_retention.empty else None,
        },
        "e04": {
            "cellMemoryMeanScoreDelta": _num(e04_memory_cell.get("meanScoreDelta", pd.Series(dtype=float)).mean()) if not e04_memory_cell.empty else None,
            "signalFieldMemoryMeanScoreDelta": _num(e04_memory_signal.get("meanScoreDelta", pd.Series(dtype=float)).mean()) if not e04_memory_signal.empty else None,
            "communicationRepairableMeanScoreDelta": _num(e04_comm_task.loc["repairable", "meanScoreDelta"]) if "repairable" in e04_comm_task.index else None,
            "communicationFatigueMeanScoreDelta": _num(e04_comm_task.loc["fatigue", "meanScoreDelta"]) if "fatigue" in e04_comm_task.index else None,
            "communicationHomeostasisMeanScoreDelta": _num(e04_comm_task.loc["homeostasis", "meanScoreDelta"]) if "homeostasis" in e04_comm_task.index else None,
            "fieldPredictorMinObservedMinusNullAuc": _num(e04_field.get("observedMinusBestNullAuc", pd.Series(dtype=float)).min()) if not e04_field.empty else None,
            "enhancedOverfitRelativeDrop": _num(e04_overfit[e04_overfit.get("policyGroup", pd.Series(dtype=str)).astype(str).eq("enhanced")]["meanRelativeDrop"].iloc[0]) if not e04_overfit.empty and (e04_overfit.get("policyGroup", pd.Series(dtype=str)).astype(str).eq("enhanced")).any() else None,
            "baselineOverfitRelativeDrop": _num(e04_overfit[e04_overfit.get("policyGroup", pd.Series(dtype=str)).astype(str).eq("baseline")]["meanRelativeDrop"].iloc[0]) if not e04_overfit.empty and (e04_overfit.get("policyGroup", pd.Series(dtype=str)).astype(str).eq("baseline")).any() else None,
            "centralizedGapBaseline": _num(e04_central[e04_central.get("localPolicyGroup", pd.Series(dtype=str)).astype(str).eq("baseline")]["meanGlobalMinusLocalScore"].mean()) if not e04_central.empty else None,
            "centralizedGapEnhanced": _num(e04_central[e04_central.get("localPolicyGroup", pd.Series(dtype=str)).astype(str).eq("enhanced")]["meanGlobalMinusLocalScore"].mean()) if not e04_central.empty else None,
            "s15Outcome": e04_status_s15.get("outcomeClassification"),
        },
        "e06": {
            "compatibilityOutcomeClasses": int(e06_compat.get("compatibilityOutcomeClass", pd.Series(dtype=str)).nunique()) if not e06_compat.empty else 0,
            "contextDependentDominancePairs": e06_context_dependent,
            "stableDominancePairs": e06_stable,
            "heldoutInterventionRows": int(len(e06_intervention)),
            "recommendedInterventionCount": int(e06_intervention.get("recommendedForS15PlaybookCandidate", pd.Series(dtype=bool)).astype(bool).sum()) if not e06_intervention.empty else 0,
        },
    }


def _evidence(
    law_id: str,
    role: str,
    path: str,
    metric: str,
    value: Any,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": EMPIRICAL_LAW_SCHEMA_VERSION,
        "lawId": law_id,
        "evidenceLinkId": f"{law_id}::{role}::{metric}",
        "evidenceRole": role,
        "artifactPath": path,
        "metricName": metric,
        "metricValue": value,
        "interpretation": interpretation,
    }


def _register_rows(law_id: str, values: Sequence[str], kind: str) -> list[dict[str, Any]]:
    return [
        {
            "schemaVersion": EMPIRICAL_LAW_SCHEMA_VERSION,
            "lawId": law_id,
            f"{kind}Id": f"{law_id}::{kind}_{index:02d}",
            "description": value,
        }
        for index, value in enumerate(values, start=1)
    ]


def build_empirical_law_synthesis(
    artifacts_dir: Path,
    previous_artifacts_dir: Path = Path("/previous-artifacts"),
) -> EmpiricalLawSynthesis:
    metrics = build_anchor_metrics(artifacts_dir, previous_artifacts_dir)
    evidence: list[dict[str, Any]] = []
    counters: list[dict[str, Any]] = []
    uncertainties: list[dict[str, Any]] = []
    falsifications: list[dict[str, Any]] = []
    laws: list[dict[str, Any]] = []

    def add_law(
        *,
        law_id: str,
        title: str,
        statement: str,
        law_family: str,
        support_level: str,
        outcome_classification: str,
        scope: str,
        excluded_scope: str,
        s13_role: str,
        recommended_use: str,
        evidence_rows: Sequence[dict[str, Any]],
        counterexample_text: Sequence[str],
        uncertainty_text: Sequence[str],
        falsification_text: Sequence[str],
    ) -> None:
        evidence.extend(evidence_rows)
        counters.extend(_register_rows(law_id, counterexample_text, "counterexample"))
        uncertainties.extend(_register_rows(law_id, uncertainty_text, "uncertainty"))
        falsifications.extend(_register_rows(law_id, falsification_text, "falsification"))
        laws.append(
            {
                "schemaVersion": EMPIRICAL_LAW_SCHEMA_VERSION,
                "lawId": law_id,
                "title": title,
                "lawFamily": law_family,
                "supportLevel": support_level,
                "outcomeClassification": outcome_classification,
                "lawStatement": statement,
                "scope": scope,
                "excludedScope": excluded_scope,
                "s13ConstraintRole": s13_role,
                "recommendedUse": recommended_use,
                "evidenceLinkIdsJson": [row["evidenceLinkId"] for row in evidence_rows],
                "counterexampleCount": len(counterexample_text),
                "uncertaintyCount": len(uncertainty_text),
                "falsificationConditionCount": len(falsification_text),
                "claimBoundary": EMPIRICAL_LAW_CLAIM_BOUNDARY,
            }
        )

    s06 = metrics["s06"]
    s07 = metrics["s07"]
    s10 = metrics["s10"]
    add_law(
        law_id="LAW01_behavior_embeddings_are_local_maps",
        title="Behavior embeddings are useful local maps",
        law_family="representation",
        support_level="supported_with_scope_limits",
        outcome_classification="supportive",
        statement=(
            "Within behavior-covered S04 entities, S05-weighted policy and goal embeddings retrieve held-out behavior "
            "profiles better than metadata or random baselines, so they are useful local maps for atlas and neighbor tasks."
        ),
        scope="S04-observed policies/goals with enough behavior rows to enter S06/S07 embeddings.",
        excluded_scope="S02 catalog-only policies, unobserved S03 goals, causal mechanism discovery, and direct biological interpretation.",
        s13_role="S13 does not invalidate local-neighbor use; it shows these distances should not be elevated to a simple transfer-success law.",
        recommended_use="Use embeddings for nearest-neighbor browsing, candidate stratification, and local similarity diagnostics.",
        evidence_rows=[
            _evidence(
                "LAW01_behavior_embeddings_are_local_maps",
                "policy_embedding_retrieval",
                "/artifacts/research_steps/S06/retrieval_metrics.parquet",
                "s06_heldout_k5_mean_cosine",
                s06["heldoutK5MeanCosine"],
                f"S06 policy behavior PCA k=5 held-out behavior cosine was {s06['heldoutK5MeanCosine']}.",
            ),
            _evidence(
                "LAW01_behavior_embeddings_are_local_maps",
                "policy_embedding_stability",
                "/artifacts/research_steps/S06/stability_metrics.parquet",
                "s06_split_half_distance_spearman",
                s06["splitHalfDistanceSpearman"],
                f"S06 split-half policy distance Spearman was {s06['splitHalfDistanceSpearman']}.",
            ),
            _evidence(
                "LAW01_behavior_embeddings_are_local_maps",
                "goal_embedding_retrieval",
                "/artifacts/research_steps/S07/retrieval_metrics.parquet",
                "s07_heldout_k3_mean_cosine",
                s07["heldoutK3MeanCosine"],
                f"S07 goal embedding k=3 held-out behavior cosine was {s07['heldoutK3MeanCosine']}.",
            ),
            _evidence(
                "LAW01_behavior_embeddings_are_local_maps",
                "taxonomy_neighbor_coherence",
                "/artifacts/research_steps/S10/class_stability.parquet",
                "s10_same_class_fraction_at_10",
                s10["sameClassFractionAt10"],
                f"S10 class labels align with S08 nearest-neighbor structure at fraction {s10['sameClassFractionAt10']}.",
            ),
        ],
        counterexample_text=[
            "S09 found no robust cross-world invariant predictor from the tested feature groups.",
            f"Direct S08 average-linkage agreement with S10 taxonomy was low (ARI {s10['s08AverageLinkageAri']}), so raw S08 trees are not the class definition.",
        ],
        uncertainty_text=[
            "Embeddings exclude catalog-only policies and sparse/unobserved goals.",
            "Behavior vectors inherit upstream normalization, missingness, and source-table heterogeneity.",
        ],
        falsification_text=[
            "Prospective S15 or follow-up held-out behavior retrieval fails to beat metadata and random baselines at k=3/k=5.",
            "Split-half policy or goal distance stability drops below 0.5 Spearman on a similarly sized rerun.",
        ],
    )

    s05 = metrics["s05"]
    s11 = metrics["s11"]
    add_law(
        law_id="LAW02_surrogates_screen_but_do_not_certify",
        title="Surrogates screen candidates but do not certify them",
        law_family="prediction",
        support_level="supported_with_direct_validation_requirement",
        outcome_classification="supportive",
        statement=(
            "The sparse S05 surrogate is useful for screening candidates inside observed support, but S11 shows it must be paired "
            "with replay/direct validation and explicit error-limit caveats."
        ),
        scope="S04-derived behavior targets with S05 held-out split coverage and S11 replay rows.",
        excluded_scope="Sparse high-scale targets, prospective biological validation, and unobserved worlds without calibration.",
        s13_role="S13 inherits this rule: transfer laws must be directly replayed rather than accepted from surrogate or distance predictions.",
        recommended_use="Use surrogate predictions as triage inputs, then require direct simulation/replay before claiming a law or design success.",
        evidence_rows=[
            _evidence(
                "LAW02_surrogates_screen_but_do_not_certify",
                "test_baseline_comparison",
                "/artifacts/research_steps/S05/baseline_comparison.parquet",
                "s05_test_global_mean_wins",
                f"{s05['testBeatsGlobalMean']}/{s05['testGlobalComparisons']}",
                "S05 sparse ridge beat the global-mean baseline on most test comparisons.",
            ),
            _evidence(
                "LAW02_surrogates_screen_but_do_not_certify",
                "test_baseline_comparison",
                "/artifacts/research_steps/S05/baseline_comparison.parquet",
                "s05_test_source_group_wins",
                f"{s05['testBeatsSourceGroupMean']}/{s05['testSourceGroupComparisons']}",
                "S05 sparse ridge beat source-group mean less consistently, showing source confounding matters.",
            ),
            _evidence(
                "LAW02_surrogates_screen_but_do_not_certify",
                "direct_replay_validation",
                "/artifacts/research_steps/S11/status.json",
                "s11_within_s05_p95_fraction",
                s11["withinS05P95Fraction"],
                "S11 replay validation mostly fell inside S05 P95 error limits.",
            ),
            _evidence(
                "LAW02_surrogates_screen_but_do_not_certify",
                "source_table_verification",
                "/artifacts/research_steps/S11/status.json",
                "s11_source_table_verified_fraction",
                s11["sourceTableVerifiedFraction"],
                "S11 source-table replay verification was complete for selected candidates.",
            ),
        ],
        counterexample_text=[
            "S05 beat source-group mean on fewer test comparisons than global mean, so source metadata remains a strong confound.",
            "S12 explicitly avoided sparse or high-scale S11 targets such as repair_success, compatibility, and high_error_reduction as primary design objectives.",
        ],
        uncertainty_text=[
            "S11 validation replays already observed S04 held-out rows rather than new prospective simulator conditions.",
            "Some target families have single observed records or wide S05 error intervals.",
        ],
        falsification_text=[
            "A new replay panel falls below 0.85 within-S05-P95 coverage on supported target families.",
            "A candidate selected only by S05 repeatedly fails direct simulation despite favorable prediction intervals.",
        ],
    )

    s09 = metrics["s09"]
    add_law(
        law_id="LAW03_no_single_invariant_feature_set",
        title="No tested invariant feature set is globally predictive",
        law_family="constraint",
        support_level="constraining",
        outcome_classification="constraining/contradictory",
        statement=(
            "Across S09 cross-world holdouts, the tested invariant feature groups did not outperform global or confound-only "
            "baselines, so empirical laws must be scoped to mechanisms, worlds, and measurements rather than stated as universal predictors."
        ),
        scope="The 13 S09-eligible behavior targets and 45 tested invariant features over the S04 corpus.",
        excluded_scope="Untested feature families, future richer causal simulators, and targets underpowered in S09.",
        s13_role="S13 strengthens this constraint by contradicting a simple S08-distance transfer predictor.",
        recommended_use="Treat broad invariant claims as hypotheses requiring separate holdouts, controls, and counterexamples.",
        evidence_rows=[
            _evidence(
                "LAW03_no_single_invariant_feature_set",
                "cross_world_holdout",
                "/artifacts/research_steps/S09/model_comparison.parquet",
                "s09_invariant_beats_global_count",
                f"{s09['invariantBeatsGlobalCount']}/{s09['eligibleTargets']}",
                "Invariant-only ridge beat global mean on none of the eligible targets.",
            ),
            _evidence(
                "LAW03_no_single_invariant_feature_set",
                "cross_world_holdout",
                "/artifacts/research_steps/S09/model_comparison.parquet",
                "s09_combined_beats_confound_count",
                f"{s09['combinedBeatsConfoundCount']}/{s09['eligibleTargets']}",
                "Combined invariant+confound ridge beat confound-only ridge on none of the eligible targets.",
            ),
            _evidence(
                "LAW03_no_single_invariant_feature_set",
                "transfer_diagnostic",
                "/artifacts/research_steps/S13/s08_distance_correlation.parquet",
                "s13_s08_composite_retention_spearman",
                metrics["s13"]["compositeDistanceRetentionSpearman"],
                "S13 S08 composite transfer distance associated with retention in the opposite direction from a simple distance law.",
            ),
        ],
        counterexample_text=[
            "S06/S07 embeddings and S10 classes are useful for neighbor/retrieval tasks even though they are not invariant laws.",
            "Some individual E04/E06 mechanism panels show task-specific regularities, but these do not generalize to a single S09 invariant feature set.",
        ],
        uncertainty_text=[
            "S09 target eligibility excluded rare repair, compatibility, and fitness_or_score targets with sparse world coverage.",
            "Feature groups are hand-engineered summaries and may miss untested causal mechanisms.",
        ],
        falsification_text=[
            "A future cross-world holdout with the same target eligibility finds invariant features beat both global and confound baselines on a majority of targets.",
            "An independently specified invariant retains predictive power after controlling source experiment, substrate, coverage, and target missingness.",
        ],
    )

    e04 = metrics["e04"]
    add_law(
        law_id="LAW04_feedback_beats_passive_memory_only_in_scope",
        title="Feedback signals matter more than passive memory counters, with costs",
        law_family="memory_feedback",
        support_level="supported_with_tradeoff",
        outcome_classification="supportive",
        statement=(
            "In the E04 repair/homeostasis proxy panels, passive cell-memory capacity alone showed near-zero or negative deltas, "
            "whereas signal-field/communication feedback produced repair/fatigue gains but also homeostasis, false-alarm, and energy costs."
        ),
        scope="E04 small CPU-reference memory, communication, field-predictor, repair, fatigue, and homeostasis panels integrated as upstream evidence.",
        excluded_scope="A universal memory-depth threshold, biological bioelectric fields, and substrates not replayed with the same feedback policy.",
        s13_role="S13 could not execute exact homeostatic birth/death transfer, so feedback laws remain upstream-scoped and transfer-caveated.",
        recommended_use="State memory/feedback laws as mechanism packages with task-specific costs, not as 'more memory is always better'.",
        evidence_rows=[
            _evidence(
                "LAW04_feedback_beats_passive_memory_only_in_scope",
                "memory_ablation",
                "/previous-artifacts/E04/research_steps/S09/memory_ablation_summary.parquet",
                "e04_cell_memory_mean_score_delta",
                e04["cellMemoryMeanScoreDelta"],
                "Cell-local memory capacity alone had near-zero/slightly negative mean score delta.",
            ),
            _evidence(
                "LAW04_feedback_beats_passive_memory_only_in_scope",
                "memory_ablation",
                "/previous-artifacts/E04/research_steps/S09/memory_ablation_summary.parquet",
                "e04_signal_field_memory_mean_score_delta",
                e04["signalFieldMemoryMeanScoreDelta"],
                "Signal-field memory capacity had a positive mean score delta in the same ablation summary.",
            ),
            _evidence(
                "LAW04_feedback_beats_passive_memory_only_in_scope",
                "communication_ablation",
                "/previous-artifacts/E04/research_steps/S10/communication_ablation_summary.parquet",
                "e04_communication_repairable_mean_score_delta",
                e04["communicationRepairableMeanScoreDelta"],
                "Communication variants improved repairable-task score on average.",
            ),
            _evidence(
                "LAW04_feedback_beats_passive_memory_only_in_scope",
                "communication_ablation",
                "/previous-artifacts/E04/research_steps/S10/communication_ablation_summary.parquet",
                "e04_communication_homeostasis_mean_score_delta",
                e04["communicationHomeostasisMeanScoreDelta"],
                "The same communication panel had strong negative homeostasis score deltas.",
            ),
            _evidence(
                "LAW04_feedback_beats_passive_memory_only_in_scope",
                "field_predictor",
                "/previous-artifacts/E04/research_steps/S12/field_predictor_summary.parquet",
                "e04_min_observed_minus_best_null_auc",
                e04["fieldPredictorMinObservedMinusNullAuc"],
                "Aggregate field predictors beat null controls for future failure/recovery/repair proxies.",
            ),
        ],
        counterexample_text=[
            "E04 S15 concluded the evidence does not isolate a unique smallest mechanism package.",
            "Communication ablations include randomized/noisy controls that can trigger repair-like behavior and false alarms.",
            f"Enhanced local policies still lagged the centralized controller by mean score {e04['centralizedGapEnhanced']} in E04 S14.",
        ],
        uncertainty_text=[
            "E04 panels are small CPU-reference simulations, not broad mechanism rankings.",
            "Feedback signals are computational fields and should not be interpreted as biological tissue fields or bioelectric measurements.",
        ],
        falsification_text=[
            "A broader repair/homeostasis panel shows passive memory-only wrappers consistently beat signal/feedback variants without higher costs.",
            "Signal feedback benefits disappear when paired no-signal and randomized-control baselines are matched for activation and energy.",
        ],
    )

    e06 = metrics["e06"]
    add_law(
        law_id="LAW05_chimeric_outcomes_are_context_dependent",
        title="Chimeric aggregation and dominance are context dependent",
        law_family="aggregation_governance",
        support_level="supported_with_context_limits",
        outcome_classification="supportive",
        statement=(
            "E06 chimeric outcomes vary with goal compatibility, ratios, arrangements, and history; aggregation or dominance labels are "
            "contextual simulator outcomes rather than fixed algotype essences."
        ),
        scope="E06 one-dimensional fixed-size chimeric panels and E07 S11/S12 aggregation-profile validation.",
        excluded_scope="Living chimeras, open-ended population growth, and exact multi-lineage transfer in S13.",
        s13_role="S13 only transferred chimera aggregation as local affinity sorting and explicitly marked exact chimera-growth mappings unsupported.",
        recommended_use="Use chimeric laws as context-stratified compatibility/dominance summaries with paired controls.",
        evidence_rows=[
            _evidence(
                "LAW05_chimeric_outcomes_are_context_dependent",
                "compatibility_outcomes",
                "/previous-artifacts/E06/research_steps/S05/compatibility_outcome_summary.parquet",
                "e06_compatibility_outcome_class_count",
                e06["compatibilityOutcomeClasses"],
                "E06 compatibility summaries contain multiple outcome classes across goal modes.",
            ),
            _evidence(
                "LAW05_chimeric_outcomes_are_context_dependent",
                "dominance_pairs",
                "/previous-artifacts/E06/research_steps/S06/dominance_pair_summary.parquet",
                "e06_context_dependent_dominance_pairs",
                e06["contextDependentDominancePairs"],
                "Most E06 dominance-pair summaries are context-dependent rather than stable-winner cases.",
            ),
            _evidence(
                "LAW05_chimeric_outcomes_are_context_dependent",
                "inverse_design_aggregation",
                "/artifacts/research_steps/S12/profile_validation_summary.parquet",
                "s12_aggregation_profile_validated_designs",
                "3/3",
                "S12 validated three aggregation-profile designs in the E03 candidate/null simulator panel.",
            ),
        ],
        counterexample_text=[
            f"E06 still found {e06['stableDominancePairs']} stable-dominance pair summaries, so context dependence is not universal for every pair.",
            "S13 did not execute exact chimera growth/dominance semantics, only affinity-sorting analogues.",
        ],
        uncertainty_text=[
            "E06 conditions are fixed-size one-dimensional local-policy arrays with computational compatibility and dominance proxies.",
            "S12 aggregation validation is panel-local and nearest-existing comparisons show some designs are close to Bubble-like references.",
        ],
        falsification_text=[
            "A broader E06-style panel shows stable winners independent of ratio, arrangement, history, and value profile for most policy pairs.",
            "Exact chimera-growth transfer replays contradict the affinity-only caveat and preserve aggregation success without context stratification.",
        ],
    )

    add_law(
        law_id="LAW06_universality_classes_are_empirical_neighborhoods",
        title="Universality classes are empirical neighborhoods",
        law_family="taxonomy",
        support_level="supported_as_taxonomy_not_proof",
        outcome_classification="supportive",
        statement=(
            "S10 classes provide stable, substrate-spanning behavioral neighborhoods for the embedded policy set, but they are not mathematical "
            "universality classes or causal invariants."
        ),
        scope="The 511 S06 behavior-embedded policies clustered with S07/S08/S09 diagnostics and upstream labels.",
        excluded_scope="Unembedded S02 catalog-only policies, direct causal laws, and raw S08 distance-tree taxonomy.",
        s13_role="S13 uses classes and distances only as coverage/context aids; transfer success must still be replayed.",
        recommended_use="Use classes as atlas bins, stratification variables, and exemplar summaries with caveats.",
        evidence_rows=[
            _evidence(
                "LAW06_universality_classes_are_empirical_neighborhoods",
                "class_count",
                "/artifacts/research_steps/S10/class_summary.parquet",
                "s10_class_count",
                s10["classCount"],
                "S10 selected eight empirical policy classes.",
            ),
            _evidence(
                "LAW06_universality_classes_are_empirical_neighborhoods",
                "class_stability",
                "/artifacts/research_steps/S10/class_stability.parquet",
                "s10_subsample_ari",
                s10["subsampleAri"],
                "S10 subsample reclustering had high ARI against full labels.",
            ),
            _evidence(
                "LAW06_universality_classes_are_empirical_neighborhoods",
                "seed_stability",
                "/artifacts/research_steps/S10/class_stability.parquet",
                "s10_seed_ari",
                s10["seedAri"],
                "S10 clustering was stable across random initializations.",
            ),
            _evidence(
                "LAW06_universality_classes_are_empirical_neighborhoods",
                "raw_distance_tree_counterweight",
                "/artifacts/research_steps/S10/class_stability.parquet",
                "s10_s08_average_linkage_ari",
                s10["s08AverageLinkageAri"],
                "Direct S08 average-linkage agreement was low, supporting the neighborhood-not-tree caveat.",
            ),
        ],
        counterexample_text=[
            "S09 found no robust cross-world invariant predictor to justify classes as universal laws.",
            "S10 labels have low direct S08 average-linkage agreement because raw distance trees are dominated by broad/outlier structure.",
        ],
        uncertainty_text=[
            "Class interpretation inherits sparse coverage and upstream label ambiguity.",
            "New S12 policies are not native members of S10 because they were created after S10.",
        ],
        falsification_text=[
            "Subsample or seed ARI falls near random under a rerun with the same embedded policy set.",
            "Class exemplars fail to summarize nearest-neighbor behavior or S15 atlas users cannot resolve artifact-linked evidence.",
        ],
    )

    s12 = metrics["s12"]
    add_law(
        law_id="LAW07_inverse_design_matches_profiles_but_novelty_is_limited",
        title="Inverse design can match profiles, but novelty is limited",
        law_family="inverse_design",
        support_level="supported_with_novelty_constraint",
        outcome_classification="supportive",
        statement=(
            "S12 direct inverse design can produce DSL policies that match supported competence profiles on held-out simulator panels, but many "
            "successful designs are near or identical to Bubble-like local inversion references."
        ),
        scope="E03-style local adjacent-swap DSL policies and the three S12 supported target profiles.",
        excluded_scope="Biological design, sparse/high-scale S11 objectives, and novelty claims beyond the held-out panel.",
        s13_role="S13 strengthened the novelty caveat because S12 transfers rarely beat the Bubble-like scalar-rank baseline.",
        recommended_use="Use inverse-designed policies as executable profile candidates and compare every claim against nearest existing baselines.",
        evidence_rows=[
            _evidence(
                "LAW07_inverse_design_matches_profiles_but_novelty_is_limited",
                "profile_validation",
                "/artifacts/research_steps/S12/profile_validation_summary.parquet",
                "s12_validated_design_count",
                f"{s12['validatedDesignCount']}/{s12['selectedDesignCount']}",
                "All selected S12 designs validated against their profile thresholds.",
            ),
            _evidence(
                "LAW07_inverse_design_matches_profiles_but_novelty_is_limited",
                "nearest_existing",
                "/artifacts/research_steps/S12/nearest_existing_policy_comparison.parquet",
                "s12_exact_nearest_behavior_matches",
                f"{s12['nearestExactBehaviorMatches']}/9",
                "Several S12 designs exactly matched a Bubble-like reference on the held-out panel.",
            ),
            _evidence(
                "LAW07_inverse_design_matches_profiles_but_novelty_is_limited",
                "nearest_existing",
                "/artifacts/research_steps/S12/nearest_existing_policy_comparison.parquet",
                "s12_near_bubble_count_distance_le_0_02",
                f"{s12['nearestNearBubbleCountAt002']}/9",
                "Most S12 designs were very close to Bubble-like references by held-out behavior distance <= 0.02.",
            ),
            _evidence(
                "LAW07_inverse_design_matches_profiles_but_novelty_is_limited",
                "transfer_constraint",
                "/artifacts/research_steps/S13/s08_distance_transfer_prediction.parquet",
                "s13_beats_bubble_like_baseline_fraction",
                metrics["s13"]["beatsBubbleLikeBaselineFraction"],
                "S13 transfer candidates beat the Bubble-like scalar-rank baseline in a minority of transfer summaries.",
            ),
        ],
        counterexample_text=[
            "S12 designs that validate a profile may still be behaviorally identical or near-identical to existing references.",
            "S13 transfer performance was often stalled and did not generally outperform the Bubble-like baseline.",
        ],
        uncertainty_text=[
            "Nearest-existing comparisons are panel-local and do not prove global behavioral identity.",
            "Sparse/high-scale S11 objectives were explicitly deferred rather than solved.",
        ],
        falsification_text=[
            "Expanded held-out panels show S12 designs no longer validate target profiles.",
            "A future inverse-design run produces non-Bubble-near policies that consistently outperform nearest baselines across transfer substrates.",
        ],
    )

    s13 = metrics["s13"]
    add_law(
        law_id="LAW08_transfer_requires_mechanism_mapping_not_raw_distance",
        title="Transfer needs mechanism-aware mappings, not raw distance",
        law_family="transfer",
        support_level="constraining",
        outcome_classification="constraining/contradictory",
        statement=(
            "S13 directly constrains a simple Platonic-distance transfer law: transfer retained partial competence only under explicit executable "
            "local-swap mappings, rarely beat the Bubble-like baseline, and S08 composite distance correlated with retention in the unexpected direction."
        ),
        scope="S13 E05 row/grid/graph local-swap analogues of S12 designed DSL policies and included controls.",
        excluded_scope="Exact homeostatic birth/death, exact chimera growth/dominance, graph targets without S08 world/goal distance components, and living substrates.",
        s13_role="S13 is the primary constraining evidence against a simple S08-distance transfer-success law.",
        recommended_use="Treat S08 distance as a diagnostic covariate; require documented observation/action/goal mappings plus direct replay and controls for transfer claims.",
        evidence_rows=[
            _evidence(
                "LAW08_transfer_requires_mechanism_mapping_not_raw_distance",
                "direct_transfer",
                "/artifacts/research_steps/S13/s08_distance_transfer_prediction.parquet",
                "s13_median_s12_heldout_retention",
                s13["medianS12HeldoutRetention"],
                "S12 transfer candidates retained partial but low median competence across held-out non-row targets.",
            ),
            _evidence(
                "LAW08_transfer_requires_mechanism_mapping_not_raw_distance",
                "direct_transfer_controls",
                "/artifacts/research_steps/S13/s08_distance_transfer_prediction.parquet",
                "s13_beats_random_or_no_transfer_fraction",
                s13["beatsRandomOrNoTransferFraction"],
                "S12 transfers beat random/no-transfer controls in fewer than half of candidate target summaries.",
            ),
            _evidence(
                "LAW08_transfer_requires_mechanism_mapping_not_raw_distance",
                "direct_transfer_controls",
                "/artifacts/research_steps/S13/s08_distance_transfer_prediction.parquet",
                "s13_beats_bubble_like_baseline_fraction",
                s13["beatsBubbleLikeBaselineFraction"],
                "S12 transfers beat the Bubble-like scalar-rank baseline in a minority of candidate target summaries.",
            ),
            _evidence(
                "LAW08_transfer_requires_mechanism_mapping_not_raw_distance",
                "distance_prediction",
                "/artifacts/research_steps/S13/s08_distance_correlation.parquet",
                "s13_s08_composite_retention_spearman",
                s13["compositeDistanceRetentionSpearman"],
                "S08 composite distance correlated positively with retention, opposite a simple 'closer transfers better' law.",
            ),
        ],
        counterexample_text=[
            "Some S12 transfers beat random/no-transfer controls and retained partial competence, so transfer is not absent.",
            "Graph-target S08 world/goal distances are missing, so composite-distance diagnostics are incomplete for that substrate.",
            "Exact homeostatic and chimera-growth mappings were documented but unsupported in the conservative S13 recovery substrate.",
        ],
        uncertainty_text=[
            "S12 policies were executed as E05 analogues rather than native DSL on all substrates.",
            "S13 used three held-out seeds and a bounded row/grid/graph panel, not every E05/E06 substrate.",
        ],
        falsification_text=[
            "A prospective transfer panel with documented mappings shows S08 composite distance negatively predicts transfer gap after controls.",
            "S12 or later designed policies beat both random/no-transfer controls and nearest Bubble-like baselines across most held-out substrates.",
        ],
    )

    return EmpiricalLawSynthesis(
        anchor_metrics=metrics,
        law_catalog=pd.DataFrame(laws),
        evidence_links=pd.DataFrame(evidence),
        counterexamples=pd.DataFrame(counters),
        uncertainty_register=pd.DataFrame(uncertainties),
        falsification_register=pd.DataFrame(falsifications),
    )


def validation_checks(
    synthesis: EmpiricalLawSynthesis,
    *,
    artifacts_dir: Path,
    previous_artifacts_dir: Path = Path("/previous-artifacts"),
) -> pd.DataFrame:
    laws = synthesis.law_catalog
    evidence = synthesis.evidence_links
    counters = synthesis.counterexamples
    uncertainty = synthesis.uncertainty_register
    falsification = synthesis.falsification_register
    statuses = load_step_statuses(artifacts_dir)
    all_statuses_success = all(bool(status.get("success")) for status in statuses.values())

    def paths_exist() -> bool:
        for path in evidence.get("artifactPath", pd.Series(dtype=str)).astype(str):
            if not Path(path).exists():
                return False
        return True

    required_law_ids = {
        "LAW01_behavior_embeddings_are_local_maps",
        "LAW02_surrogates_screen_but_do_not_certify",
        "LAW03_no_single_invariant_feature_set",
        "LAW04_feedback_beats_passive_memory_only_in_scope",
        "LAW05_chimeric_outcomes_are_context_dependent",
        "LAW06_universality_classes_are_empirical_neighborhoods",
        "LAW07_inverse_design_matches_profiles_but_novelty_is_limited",
        "LAW08_transfer_requires_mechanism_mapping_not_raw_distance",
    }
    law_ids = set(laws.get("lawId", pd.Series(dtype=str)).astype(str))
    per_law_evidence = evidence.groupby("lawId").size() if not evidence.empty else pd.Series(dtype=int)
    per_law_counter = counters.groupby("lawId").size() if not counters.empty else pd.Series(dtype=int)
    per_law_uncertainty = uncertainty.groupby("lawId").size() if not uncertainty.empty else pd.Series(dtype=int)
    per_law_falsification = falsification.groupby("lawId").size() if not falsification.empty else pd.Series(dtype=int)
    transfer = laws[laws.get("lawId", pd.Series(dtype=str)).astype(str).eq("LAW08_transfer_requires_mechanism_mapping_not_raw_distance")]
    s13_transfer_evidence = evidence[
        evidence.get("lawId", pd.Series(dtype=str)).astype(str).eq("LAW08_transfer_requires_mechanism_mapping_not_raw_distance")
        & evidence.get("artifactPath", pd.Series(dtype=str)).astype(str).str.contains("/S13/", regex=False)
    ]

    rows = [
        {
            "checkId": "upstream_statuses_successful",
            "severity": "error",
            "success": bool(all_statuses_success),
            "observed": f"{sum(bool(status.get('success')) for status in statuses.values())}/{len(statuses)}",
            "expected": "all required S01-S13 statuses successful",
        },
        {
            "checkId": "required_law_ids_present",
            "severity": "error",
            "success": required_law_ids.issubset(law_ids),
            "observed": ",".join(sorted(law_ids)),
            "expected": ",".join(sorted(required_law_ids)),
        },
        {
            "checkId": "each_law_has_evidence",
            "severity": "error",
            "success": all(int(per_law_evidence.get(law_id, 0)) >= 2 for law_id in law_ids),
            "observed": compact_json(per_law_evidence.to_dict()),
            "expected": "at least two evidence links per law",
        },
        {
            "checkId": "evidence_paths_exist",
            "severity": "error",
            "success": paths_exist(),
            "observed": str(len(evidence)),
            "expected": "all evidence artifact paths resolve",
        },
        {
            "checkId": "each_law_has_counterexamples",
            "severity": "error",
            "success": all(int(per_law_counter.get(law_id, 0)) >= 1 for law_id in law_ids),
            "observed": compact_json(per_law_counter.to_dict()),
            "expected": "at least one counterexample per law",
        },
        {
            "checkId": "each_law_has_uncertainty",
            "severity": "error",
            "success": all(int(per_law_uncertainty.get(law_id, 0)) >= 1 for law_id in law_ids),
            "observed": compact_json(per_law_uncertainty.to_dict()),
            "expected": "at least one uncertainty caveat per law",
        },
        {
            "checkId": "each_law_has_falsification_conditions",
            "severity": "error",
            "success": all(int(per_law_falsification.get(law_id, 0)) >= 1 for law_id in law_ids),
            "observed": compact_json(per_law_falsification.to_dict()),
            "expected": "at least one falsification condition per law",
        },
        {
            "checkId": "s13_transfer_constraint_explicit",
            "severity": "error",
            "success": bool(
                not transfer.empty
                and "constraining" in str(transfer.iloc[0].get("supportLevel", ""))
                and "simple" in str(transfer.iloc[0].get("lawStatement", "")).lower()
                and not s13_transfer_evidence.empty
            ),
            "observed": f"transferRows={len(transfer)}; s13Evidence={len(s13_transfer_evidence)}",
            "expected": "S13 constraining evidence against simple S08-distance transfer law",
        },
        {
            "checkId": "claim_boundaries_present",
            "severity": "error",
            "success": bool(laws.get("claimBoundary", pd.Series(dtype=str)).astype(str).str.len().gt(40).all()),
            "observed": str(int(laws.get("claimBoundary", pd.Series(dtype=str)).astype(str).str.len().gt(40).sum())),
            "expected": "claim boundary text on every law",
        },
        {
            "checkId": "no_unbounded_law_language",
            "severity": "warning",
            "success": not laws["lawStatement"].astype(str).str.contains("universal biological|proves biology|causal proof", case=False, regex=True).any(),
            "observed": "bounded language scan complete",
            "expected": "no unbounded biological or causal-proof wording",
        },
    ]
    return pd.DataFrame(rows)


def law_outcome_classification(checks: pd.DataFrame) -> str:
    hard_failures = checks[(checks["severity"].eq("error")) & (~checks["success"])]
    return "supportive" if hard_failures.empty else "constraining/contradictory"
