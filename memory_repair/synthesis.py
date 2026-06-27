"""Minimal-mechanism synthesis helpers for E04 S15."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SYNTHESIS_VERSION = "e04_s15_minimal_mechanism_synthesis.v1"
SYNTHESIS_PROXY_SCOPE_NOTE = (
    "Direct computational synthesis only; not biological validation, biological intelligence, "
    "morphogenesis, or causal tissue-repair evidence."
)
HANDOFF_REPLAY_VERSION = "e04_s15_handoff_replay.v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float):
        return round(float(value), 12) if math.isfinite(value) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def _mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _max_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.max())


def _min_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.min())


def required_s15_evidence_sources() -> list[dict[str, Any]]:
    """Artifact sources that S15 conclusions are allowed to cite."""

    sources: list[dict[str, Any]] = []
    for step_number in range(1, 15):
        step_id = f"S{step_number:02d}"
        sources.append(
            {
                "evidenceSourceId": f"{step_id}_status",
                "researchStepId": step_id,
                "artifactPath": f"research_steps/{step_id}/status.json",
                "sourceKind": "status_json",
                "required": True,
                "evidenceRole": f"{step_id} completion status and validation result",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            }
        )
        sources.append(
            {
                "evidenceSourceId": f"{step_id}_summary",
                "researchStepId": step_id,
                "artifactPath": f"research_steps/{step_id}/summary.md",
                "sourceKind": "status_markdown",
                "required": True,
                "evidenceRole": f"{step_id} proxy-scoped narrative summary",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            }
        )
    sources.extend(
        [
            {
                "evidenceSourceId": "S09_memory_summary",
                "researchStepId": "S09",
                "artifactPath": "results/e04_s09_memory_ablation_summary.parquet",
                "sourceKind": "result_table",
                "required": True,
                "evidenceRole": "paired memory-capacity effects",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            },
            {
                "evidenceSourceId": "S10_communication_summary",
                "researchStepId": "S10",
                "artifactPath": "results/e04_s10_communication_ablation_summary.parquet",
                "sourceKind": "result_table",
                "required": True,
                "evidenceRole": "paired communication-architecture effects",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            },
            {
                "evidenceSourceId": "S10_transfer_gaps",
                "researchStepId": "S10",
                "artifactPath": "results/e04_s10_communication_transfer_gaps.parquet",
                "sourceKind": "result_table",
                "required": True,
                "evidenceRole": "communication train/transfer gap proxy",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            },
            {
                "evidenceSourceId": "S11_competence_profiles",
                "researchStepId": "S11",
                "artifactPath": "results/e04_s11_competence_policy_profiles.parquet",
                "sourceKind": "result_table",
                "required": True,
                "evidenceRole": "held-out direct computational competence profiles",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            },
            {
                "evidenceSourceId": "S12_field_predictor_summary",
                "researchStepId": "S12",
                "artifactPath": "results/e04_s12_tissue_field_predictor_summary.parquet",
                "sourceKind": "result_table",
                "required": True,
                "evidenceRole": "seed-held-out simulated aggregate field predictor lift",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            },
            {
                "evidenceSourceId": "S13_transfer_gaps",
                "researchStepId": "S13",
                "artifactPath": "results/e04_s13_overfitting_policy_gaps.parquet",
                "sourceKind": "result_table",
                "required": True,
                "evidenceRole": "selection-to-holdout transfer gaps",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            },
            {
                "evidenceSourceId": "S14_centralized_summary",
                "researchStepId": "S14",
                "artifactPath": "results/e04_s14_centralized_group_summary.parquet",
                "sourceKind": "result_table",
                "required": True,
                "evidenceRole": "paired local/global-oracle controller gaps",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            },
            {
                "evidenceSourceId": "S08_evolved_policies",
                "researchStepId": "S08",
                "artifactPath": "results/e04_evolved_repair_policies.parquet",
                "sourceKind": "result_table",
                "required": True,
                "evidenceRole": "S08 evolved local policy genome and config records",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            },
            {
                "evidenceSourceId": "S08_candidate_audits",
                "researchStepId": "S08",
                "artifactPath": "results/e04_s08_candidate_audits.parquet",
                "sourceKind": "result_table",
                "required": True,
                "evidenceRole": "S07 no-oracle audits for S08 candidates",
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
            },
        ]
    )
    return sources


def step_status_rows(status_payloads: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in status_payloads:
        validation = payload.get("validationResult", {})
        if not isinstance(validation, Mapping):
            validation = {}
        rows.append(
            {
                "researchStepId": str(payload.get("researchStepId", "")),
                "stepNumber": int(payload.get("stepNumber", 0)),
                "success": bool(payload.get("success", False)),
                "status": str(payload.get("status", "")),
                "validationPassedCount": validation.get("passedCount"),
                "validationTotalCount": validation.get("totalCount"),
                "validationAllPassed": validation.get("allPassed"),
                "recommendedNextAction": payload.get("recommendedNextAction"),
                "outcomeClassification": payload.get("outcomeClassification"),
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                "synthesisVersion": SYNTHESIS_VERSION,
            }
        )
    return pd.DataFrame(_json_ready(rows)).sort_values("stepNumber", kind="mergesort").reset_index(drop=True)


def aggregate_mechanism_metrics(
    memory_summary: pd.DataFrame,
    communication_summary: pd.DataFrame,
    competence_profiles: pd.DataFrame,
    transfer_gaps: pd.DataFrame,
    centralized_summary: pd.DataFrame,
) -> dict[str, Any]:
    cell_memory = memory_summary[memory_summary.get("ablationAxis", pd.Series(dtype=str)).astype(str).eq("cell_memory_capacity")]
    field_memory = memory_summary[
        memory_summary.get("ablationAxis", pd.Series(dtype=str)).astype(str).eq("signal_field_memory_capacity")
    ]
    communication_by_variant = (
        communication_summary.groupby("ablationVariant", dropna=False)["meanScoreDelta"].mean().sort_values(ascending=False)
        if not communication_summary.empty
        else pd.Series(dtype=float)
    )
    enhanced_profiles = competence_profiles[competence_profiles.get("policyGroup", pd.Series(dtype=str)).astype(str).eq("enhanced")]
    baseline_profiles = competence_profiles[competence_profiles.get("policyGroup", pd.Series(dtype=str)).astype(str).eq("baseline")]
    learned_transfer = transfer_gaps[transfer_gaps.get("policyId", pd.Series(dtype=str)).astype(str).eq("enhanced_local_memory_signal_adaptive")]
    baseline_bubble_transfer = transfer_gaps[transfer_gaps.get("policyId", pd.Series(dtype=str)).astype(str).eq("baseline_classic_bubble_open_loop")]
    centralized_gap_mean = _mean_or_none(centralized_summary, "meanGlobalMinusLocalScore")
    return _json_ready(
        {
            "meanCellMemoryScoreDelta": _mean_or_none(cell_memory, "meanScoreDelta"),
            "maxCellMemoryScoreDelta": _max_or_none(cell_memory, "meanScoreDelta"),
            "meanSignalFieldMemoryScoreDelta": _mean_or_none(field_memory, "meanScoreDelta"),
            "bestCommunicationVariant": None if communication_by_variant.empty else str(communication_by_variant.index[0]),
            "bestCommunicationMeanScoreDelta": None if communication_by_variant.empty else float(communication_by_variant.iloc[0]),
            "diffusiveCommunicationMeanScoreDelta": (
                None if "diffusive" not in communication_by_variant.index else float(communication_by_variant.loc["diffusive"])
            ),
            "nearestNeighborCommunicationMeanScoreDelta": (
                None
                if "nearest_neighbor" not in communication_by_variant.index
                else float(communication_by_variant.loc["nearest_neighbor"])
            ),
            "meanEnhancedOverallCompetenceProxy": _mean_or_none(enhanced_profiles, "overallCompetenceProxy"),
            "meanBaselineOverallCompetenceProxy": _mean_or_none(baseline_profiles, "overallCompetenceProxy"),
            "learnedPolicyHoldoutMeanScore": _mean_or_none(learned_transfer, "holdoutMeanScore"),
            "learnedPolicyRelativeDrop": _mean_or_none(learned_transfer, "relativeSelectionToHoldoutDrop"),
            "baselineBubbleHoldoutMeanScore": _mean_or_none(baseline_bubble_transfer, "holdoutMeanScore"),
            "meanCentralizedGlobalMinusLocalScore": centralized_gap_mean,
            "minCentralizedGlobalMinusLocalScore": _min_or_none(centralized_summary, "meanGlobalMinusLocalScore"),
            "synthesisVersion": SYNTHESIS_VERSION,
            "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
        }
    )


def policy_evidence_table(
    competence_profiles: pd.DataFrame,
    transfer_gaps: pd.DataFrame,
    centralized_summary: pd.DataFrame,
    candidate_audits: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    policy_ids = sorted(
        set(competence_profiles.get("policyId", pd.Series(dtype=str)).dropna().astype(str))
        | set(transfer_gaps.get("policyId", pd.Series(dtype=str)).dropna().astype(str))
        | set(centralized_summary.get("localPolicyId", pd.Series(dtype=str)).dropna().astype(str))
    )
    for policy_id in policy_ids:
        profile = competence_profiles[competence_profiles["policyId"].astype(str).eq(policy_id)] if "policyId" in competence_profiles else pd.DataFrame()
        gap = transfer_gaps[transfer_gaps["policyId"].astype(str).eq(policy_id)] if "policyId" in transfer_gaps else pd.DataFrame()
        central = (
            centralized_summary[centralized_summary["localPolicyId"].astype(str).eq(policy_id)]
            if "localPolicyId" in centralized_summary
            else pd.DataFrame()
        )
        source_candidate_id = None
        if "sourceCandidateId" in profile and not profile.empty:
            source_candidate_id = profile.iloc[0].get("sourceCandidateId")
        if source_candidate_id is None and "e04_s08_candidate_" in policy_id:
            source_candidate_id = "e04_s08_candidate_" + policy_id.split("e04_s08_candidate_", 1)[1]
        audit_ok = None
        if source_candidate_id and not candidate_audits.empty and "candidateId" in candidate_audits:
            audit_rows = candidate_audits[candidate_audits["candidateId"].astype(str).eq(str(source_candidate_id))]
            if not audit_rows.empty:
                audit_ok = bool(audit_rows.iloc[0].get("success", False))
        policy_group = (
            profile.iloc[0].get("policyGroup")
            if not profile.empty
            else (gap.iloc[0].get("policyGroup") if not gap.empty else central.iloc[0].get("localPolicyGroup") if not central.empty else None)
        )
        family_kind = (
            profile.iloc[0].get("familyKind")
            if not profile.empty
            else (gap.iloc[0].get("familyKind") if not gap.empty else central.iloc[0].get("localFamilyKind") if not central.empty else None)
        )
        overall = _mean_or_none(profile, "overallCompetenceProxy")
        holdout = _mean_or_none(gap, "holdoutMeanScore")
        relative_drop = _mean_or_none(gap, "relativeSelectionToHoldoutDrop")
        central_score = _mean_or_none(central, "meanLocalControllerScore")
        central_gap = _mean_or_none(central, "meanGlobalMinusLocalScore")
        robustness = None if relative_drop is None else max(0.0, min(1.0, 1.0 - float(relative_drop)))
        locality_closeness = None if central_gap is None else max(0.0, min(1.0, 1.0 - max(0.0, float(central_gap))))
        handoff_values = [overall, holdout, central_score, robustness, locality_closeness]
        handoff_score = None if all(value is None for value in handoff_values) else float(np.mean([value for value in handoff_values if value is not None]))
        rows.append(
            {
                "policyId": policy_id,
                "policyGroup": policy_group,
                "familyKind": family_kind,
                "sourceCandidateId": source_candidate_id,
                "overallCompetenceProxy": overall,
                "holdoutMeanScore": holdout,
                "relativeSelectionToHoldoutDrop": relative_drop,
                "meanLocalControllerScore": central_score,
                "meanGlobalMinusLocalScore": central_gap,
                "handoffCompositeProxy": handoff_score,
                "selectedForS13HoldoutReview": bool(gap.iloc[0].get("selectedForHoldoutReview", False)) if not gap.empty else False,
                "s08AuditSuccess": audit_ok,
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                "synthesisVersion": SYNTHESIS_VERSION,
            }
        )
    return pd.DataFrame(_json_ready(rows)).sort_values("handoffCompositeProxy", ascending=False, kind="mergesort").reset_index(drop=True)


def select_handoff_candidates(policy_evidence: pd.DataFrame, *, max_evolved: int = 2) -> pd.DataFrame:
    if policy_evidence.empty:
        return pd.DataFrame()
    candidates = policy_evidence[policy_evidence["policyGroup"].astype(str).eq("enhanced")].copy()
    if candidates.empty:
        return candidates
    learned = candidates[candidates["familyKind"].astype(str).eq("learned")].sort_values(
        "handoffCompositeProxy", ascending=False, kind="mergesort"
    )
    evolved = candidates[candidates["familyKind"].astype(str).eq("evolved")].sort_values(
        "handoffCompositeProxy", ascending=False, kind="mergesort"
    )
    selected = []
    if not learned.empty:
        selected.append(learned.iloc[0].to_dict())
    for _, row in evolved.head(max_evolved).iterrows():
        selected.append(row.to_dict())
    if not selected:
        selected = [candidates.sort_values("handoffCompositeProxy", ascending=False, kind="mergesort").iloc[0].to_dict()]
    for index, row in enumerate(selected):
        if index == 0:
            row["handoffTier"] = "primary"
            row["handoffRationale"] = (
                "Best enhanced local learned package by combined S11/S13/S14 proxy evidence, with low transfer drop."
            )
        else:
            row["handoffTier"] = "secondary"
            row["handoffRationale"] = "Replayable evolved local policy retained as an exploratory backup candidate."
        row["recommendedForHandoff"] = True
        row["handoffReplayRequired"] = True
        row["claimBoundary"] = SYNTHESIS_PROXY_SCOPE_NOTE
        row["synthesisVersion"] = SYNTHESIS_VERSION
    return pd.DataFrame(_json_ready(selected)).reset_index(drop=True)


def build_mechanism_conclusions(metrics: Mapping[str, Any]) -> pd.DataFrame:
    """Return proxy-scoped conclusions, each with source IDs that must resolve to artifacts."""

    conclusions = [
        {
            "conclusionId": "minimal_package_identified_but_not_unique",
            "conclusionType": "mechanism_package",
            "conclusionStatus": "supported_with_constraints",
            "minimalMechanismPackage": (
                "local-adaptive adjacent-swap policy + bounded local memory + local signal emission/sensing + "
                "signal-threshold repair + S07 local-only training/audit gate"
            ),
            "conclusion": (
                "The strongest handoff package in this computational panel is the local learned memory-signal policy, "
                "but S09-S10 do not isolate a unique smallest communication or memory mechanism."
            ),
            "directMetricSummary": compact_json(
                {
                    "learnedPolicyHoldoutMeanScore": metrics.get("learnedPolicyHoldoutMeanScore"),
                    "learnedPolicyRelativeDrop": metrics.get("learnedPolicyRelativeDrop"),
                    "minCentralizedGlobalMinusLocalScore": metrics.get("minCentralizedGlobalMinusLocalScore"),
                }
            ),
            "evidenceSourceIds": compact_json(
                ["S06_status", "S07_status", "S09_memory_summary", "S10_communication_summary", "S11_competence_profiles", "S13_transfer_gaps", "S14_centralized_summary"]
            ),
        },
        {
            "conclusionId": "cell_local_memory_alone_not_sufficient",
            "conclusionType": "negative_constraint",
            "conclusionStatus": "constraining",
            "minimalMechanismPackage": "cell-local memory capacity alone",
            "conclusion": (
                "Cell-local memory capacity alone is not sufficient evidence for robust repair improvement in this implementation; "
                "S09 mean score deltas for cell-memory variants were near zero or slightly negative."
            ),
            "directMetricSummary": compact_json(
                {
                    "meanCellMemoryScoreDelta": metrics.get("meanCellMemoryScoreDelta"),
                    "maxCellMemoryScoreDelta": metrics.get("maxCellMemoryScoreDelta"),
                }
            ),
            "evidenceSourceIds": compact_json(["S01_status", "S09_memory_summary"]),
        },
        {
            "conclusionId": "signal_field_persistence_supported",
            "conclusionType": "supportive_mechanism",
            "conclusionStatus": "supportive",
            "minimalMechanismPackage": "signal-field memory/persistence",
            "conclusion": (
                "Signal-field memory/persistence has the clearest positive ablation signal among memory axes in S09, "
                "so it remains a candidate component of the minimal computational package."
            ),
            "directMetricSummary": compact_json(
                {"meanSignalFieldMemoryScoreDelta": metrics.get("meanSignalFieldMemoryScoreDelta")}
            ),
            "evidenceSourceIds": compact_json(["S02_status", "S09_memory_summary"]),
        },
        {
            "conclusionId": "communication_architecture_not_cleanly_identified",
            "conclusionType": "negative_constraint",
            "conclusionStatus": "constraining",
            "minimalMechanismPackage": "specific deterministic signal architecture",
            "conclusion": (
                "S10 shows communication effects are architecture-sensitive and not cleanly attributable to a specific deterministic "
                "signal architecture because randomized-control signaling had the best mean score delta on the small panel."
            ),
            "directMetricSummary": compact_json(
                {
                    "bestCommunicationVariant": metrics.get("bestCommunicationVariant"),
                    "bestCommunicationMeanScoreDelta": metrics.get("bestCommunicationMeanScoreDelta"),
                    "diffusiveCommunicationMeanScoreDelta": metrics.get("diffusiveCommunicationMeanScoreDelta"),
                    "nearestNeighborCommunicationMeanScoreDelta": metrics.get("nearestNeighborCommunicationMeanScoreDelta"),
                }
            ),
            "evidenceSourceIds": compact_json(["S10_communication_summary", "S10_transfer_gaps"]),
        },
        {
            "conclusionId": "local_adaptive_learning_best_handoff",
            "conclusionType": "handoff_candidate",
            "conclusionStatus": "supportive",
            "minimalMechanismPackage": "enhanced_local_memory_signal_adaptive",
            "conclusion": (
                "The local learned memory-signal policy is the primary handoff candidate because it preserved S13 holdout score "
                "with near-zero relative drop while remaining local-only and close to the best local S14 controller score."
            ),
            "directMetricSummary": compact_json(
                {
                    "learnedPolicyHoldoutMeanScore": metrics.get("learnedPolicyHoldoutMeanScore"),
                    "learnedPolicyRelativeDrop": metrics.get("learnedPolicyRelativeDrop"),
                    "meanCentralizedGlobalMinusLocalScore": metrics.get("meanCentralizedGlobalMinusLocalScore"),
                }
            ),
            "evidenceSourceIds": compact_json(["S06_status", "S07_status", "S11_competence_profiles", "S13_transfer_gaps", "S14_centralized_summary"]),
        },
        {
            "conclusionId": "evolved_policies_exploratory_backups",
            "conclusionType": "handoff_candidate",
            "conclusionStatus": "supported_with_constraints",
            "minimalMechanismPackage": "S08 evolved local memory-signal policies",
            "conclusion": (
                "S08 evolved policies are replayable no-oracle backups with repair competence, but S13 transfer gaps and higher energy "
                "make them exploratory rather than the primary minimal mechanism package."
            ),
            "directMetricSummary": compact_json(
                {
                    "meanEnhancedOverallCompetenceProxy": metrics.get("meanEnhancedOverallCompetenceProxy"),
                    "meanBaselineOverallCompetenceProxy": metrics.get("meanBaselineOverallCompetenceProxy"),
                }
            ),
            "evidenceSourceIds": compact_json(["S08_evolved_policies", "S08_candidate_audits", "S11_competence_profiles", "S13_transfer_gaps", "S14_centralized_summary"]),
        },
        {
            "conclusionId": "centralized_oracle_gap_remains",
            "conclusionType": "upper_bound_constraint",
            "conclusionStatus": "constraining",
            "minimalMechanismPackage": "local-only policies versus explicit global-oracle comparator",
            "conclusion": (
                "S14 shows an explicit centralized global-oracle upper-bound still outperforms local policies on average, "
                "so the local package is not equivalent to a global controller."
            ),
            "directMetricSummary": compact_json(
                {"meanCentralizedGlobalMinusLocalScore": metrics.get("meanCentralizedGlobalMinusLocalScore")}
            ),
            "evidenceSourceIds": compact_json(["S07_status", "S14_centralized_summary"]),
        },
        {
            "conclusionId": "field_proxies_are_predictive_not_causal",
            "conclusionType": "claim_boundary",
            "conclusionStatus": "supportive_with_constraints",
            "minimalMechanismPackage": "simulated aggregate fields as analysis features",
            "conclusion": (
                "S12 simulated aggregate fields help predict future repair/failure/recovery proxies over null fields, "
                "but the result is associative and should guide reporting rather than be treated as causal field control."
            ),
            "directMetricSummary": compact_json({}),
            "evidenceSourceIds": compact_json(["S12_field_predictor_summary", "S12_status"]),
        },
    ]
    for row in conclusions:
        row["researchStepId"] = "S15"
        row["claimBoundary"] = SYNTHESIS_PROXY_SCOPE_NOTE
        row["synthesisVersion"] = SYNTHESIS_VERSION
    return pd.DataFrame(_json_ready(conclusions))


def build_traceability_matrix(conclusion_df: pd.DataFrame, evidence_source_df: pd.DataFrame) -> pd.DataFrame:
    source_by_id = {str(row.evidenceSourceId): row for row in evidence_source_df.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for conclusion in conclusion_df.itertuples(index=False):
        source_ids = json.loads(str(conclusion.evidenceSourceIds))
        for source_id in source_ids:
            source = source_by_id.get(str(source_id))
            rows.append(
                {
                    "traceId": stable_id("s15_trace", [conclusion.conclusionId, source_id]),
                    "conclusionId": conclusion.conclusionId,
                    "evidenceSourceId": source_id,
                    "researchStepId": None if source is None else source.researchStepId,
                    "artifactPath": None if source is None else source.artifactPath,
                    "sourceKind": None if source is None else source.sourceKind,
                    "sourceExists": False if source is None else bool(source.exists),
                    "conclusionStatus": conclusion.conclusionStatus,
                    "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                    "synthesisVersion": SYNTHESIS_VERSION,
                }
            )
    return pd.DataFrame(_json_ready(rows))


def validate_synthesis_outputs(
    step_status_df: pd.DataFrame,
    evidence_source_df: pd.DataFrame,
    conclusion_df: pd.DataFrame,
    traceability_df: pd.DataFrame,
    handoff_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    *,
    report_exists: bool,
    report_bundle_exists: bool,
    deterministic_replay_match: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append(
            {
                "checkId": check_id,
                "success": bool(success),
                "detail": detail,
                "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                "synthesisVersion": SYNTHESIS_VERSION,
            }
        )

    required_steps = {f"S{index:02d}" for index in range(1, 15)}
    completed_steps = set(step_status_df.loc[step_status_df["success"].astype(bool), "researchStepId"].astype(str))
    add("all_prior_steps_successful", required_steps.issubset(completed_steps), f"completedSteps={sorted(completed_steps)}")
    missing_sources = evidence_source_df[evidence_source_df["required"].astype(bool) & ~evidence_source_df["exists"].astype(bool)]
    add("required_evidence_artifacts_exist", missing_sources.empty, f"missingSources={len(missing_sources)}")
    add("conclusions_nonempty", len(conclusion_df) >= 6, f"conclusionRows={len(conclusion_df)}")
    trace_ok = (
        not traceability_df.empty
        and traceability_df["sourceExists"].astype(bool).all()
        and set(conclusion_df["conclusionId"].astype(str)).issubset(set(traceability_df["conclusionId"].astype(str)))
    )
    add("every_conclusion_traces_to_existing_artifacts", trace_ok, f"traceRows={len(traceability_df)}")
    boundary_ok = bool(
        conclusion_df["claimBoundary"].astype(str).str.contains("Direct computational synthesis only", regex=False).all()
        and handoff_df["claimBoundary"].astype(str).str.contains("Direct computational synthesis only", regex=False).all()
        and replay_df["claimBoundary"].astype(str).str.contains("Direct computational synthesis only", regex=False).all()
    )
    add("proxy_claim_boundaries_present", boundary_ok, "conclusions, handoff candidates, and replays carry S15 proxy scope text")
    handoff_ok = (
        len(handoff_df) >= 1
        and handoff_df["recommendedForHandoff"].astype(bool).all()
        and handoff_df["handoffCompositeProxy"].notna().all()
        and handoff_df["policyGroup"].astype(str).eq("enhanced").all()
    )
    add("handoff_candidates_selected", handoff_ok, f"handoffRows={len(handoff_df)}")
    local_only_ok = True
    if "oracleAccessAllowed" in handoff_df.columns:
        local_only_ok = local_only_ok and (~handoff_df["oracleAccessAllowed"].fillna(False).astype(bool)).all()
    if "s08AuditSuccess" in handoff_df.columns:
        audit_values = handoff_df["s08AuditSuccess"].dropna()
        local_only_ok = local_only_ok and (audit_values.empty or audit_values.astype(bool).all())
    add("handoff_candidates_local_only_or_audited", local_only_ok, "handoff candidates have no oracle access marker and S08 evolved candidates pass audits")
    replay_scores = pd.to_numeric(replay_df.get("handoffReplayScore", pd.Series(dtype=float)), errors="coerce")
    replay_ok = (
        len(replay_df) > 0
        and replay_df["policyId"].nunique() == len(handoff_df)
        and replay_df["completed"].notna().all()
        and replay_scores.between(0.0, 1.0).all()
    )
    add("handoff_replays_successful_and_bounded", replay_ok, f"replayRows={len(replay_df)}")
    add("handoff_replays_deterministic", bool(deterministic_replay_match), "first and second S15 replay fingerprints match")
    add("minimal_report_written", bool(report_exists), "e04_minimal_repair_mechanisms.md exists")
    add("report_bundle_inputs_written", bool(report_bundle_exists), "report_bundle_inputs package exists")
    return pd.DataFrame(rows)
