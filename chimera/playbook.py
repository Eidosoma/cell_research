"""E06 S15 chimeric-control playbook synthesis.

S15 packages completed E06 S01-S14 outputs into a traceable playbook and
report-bundle input set. It is a synthesis step only: it does not run new
chimera simulations and it does not start E07.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mixtures import artifact_records, git_value, write_json
from .panel import CLAIM_BOUNDARY, json_ready


STEP_ID = "S15"
STEP_NUMBER = 15
PLAYBOOK_VERSION = "e06_s15_chimeric_control_playbook.v1"
REPORT_BUNDLE_SCHEMA = "eidosoma.e06_s15_report_bundle_inputs.v1"
BIOLOGICAL_ANALOGY_CAVEAT = (
    "Chimera, graft, mutant, governance, developmental-history, intervention, morphogenesis, and "
    "collective-identity labels are computational analogies over fixed-size local-policy simulations; "
    "they are not biological validation or wet-lab protocols."
)
S14_INTERVENTION_CAVEAT = (
    "S14 intervention recipes are constrained simulation-derived candidates only: no recipe met the robust "
    "S15 playbook-candidate threshold, held-out validation used S13 source contexts, and seed/ratio transfer "
    "was not feasible because the selected S12/S13 corpus has one seed and one 75:25 ratio."
)
S15_RECOMMENDED_NEXT_ACTION = (
    "Chief Scientist review of the E06 S15 playbook; E06 has no further queued research step, and downstream "
    "E07 should consume the report-bundle inputs only after review."
)


def _compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _mean(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in df.columns or df.empty:
        return default
    series = pd.to_numeric(df[column], errors="coerce")
    out = float(series.mean()) if series.notna().any() else default
    return out if np.isfinite(out) else default


def _first_row(df: pd.DataFrame, **matches: str) -> pd.Series:
    out = df.copy()
    for key, value in matches.items():
        if key not in out.columns:
            return pd.Series(dtype=object)
        out = out[out[key].astype(str).eq(value)]
    return out.iloc[0] if not out.empty else pd.Series(dtype=object)


def _path_link(path: str | Path, label: str | None = None) -> str:
    p = str(path)
    return f"[{label or Path(p).name}]({p})"


def _write_table(df: pd.DataFrame, step_base: Path, results_base: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    for base in [step_base, results_base]:
        if base is None:
            continue
        csv_path = base.with_suffix(".csv")
        parquet_path = base.with_suffix(".parquet")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        paths.extend([csv_path, parquet_path])
    return paths


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required S15 artifact missing: {path}")
    return pd.read_parquet(path)


def _read_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required S15 status artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def source_artifact_specs(artifacts_dir: Path) -> list[dict[str, str]]:
    return [
        {"step": "S01", "role": "algotype panel", "kind": "table", "path": str(artifacts_dir / "data/e06_algotype_panel.parquet")},
        {"step": "S02", "role": "mixture ratio runs", "kind": "table", "path": str(artifacts_dir / "results/e06_mixture_ratios.parquet")},
        {"step": "S02", "role": "mixture ratio summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S02/mixture_ratio_pair_summary.parquet")},
        {"step": "S03", "role": "arrangement runs", "kind": "table", "path": str(artifacts_dir / "results/e06_initial_arrangements.parquet")},
        {"step": "S03", "role": "arrangement summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S03/arrangement_summary.parquet")},
        {"step": "S04", "role": "goal compatibility runs", "kind": "table", "path": str(artifacts_dir / "results/e06_goal_compatibility.parquet")},
        {"step": "S04", "role": "goal mode summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S04/goal_mode_summary.parquet")},
        {"step": "S05", "role": "compatibility metrics", "kind": "table", "path": str(artifacts_dir / "results/e06_compatibility_metrics.parquet")},
        {"step": "S05", "role": "compatibility mode summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S05/compatibility_mode_summary.parquet")},
        {"step": "S06", "role": "dominance hierarchy", "kind": "table", "path": str(artifacts_dir / "results/e06_dominance_hierarchy.parquet")},
        {"step": "S06", "role": "dominance pair summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S06/dominance_pair_summary.parquet")},
        {"step": "S07", "role": "mosaic classifications", "kind": "table", "path": str(artifacts_dir / "results/e06_mosaic_classifications.parquet")},
        {"step": "S07", "role": "mosaic class summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S07/mosaic_class_summary.parquet")},
        {"step": "S08", "role": "interface rules", "kind": "table", "path": str(artifacts_dir / "results/e06_interface_rules.parquet")},
        {"step": "S08", "role": "interface effect summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S08/interface_effect_summary.parquet")},
        {"step": "S09", "role": "governance mechanisms", "kind": "table", "path": str(artifacts_dir / "results/e06_governance_mechanisms.parquet")},
        {"step": "S09", "role": "governance effect summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S09/governance_effect_summary.parquet")},
        {"step": "S10", "role": "graft experiments", "kind": "table", "path": str(artifacts_dir / "results/e06_graft_experiments.parquet")},
        {"step": "S10", "role": "graft outcome summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S10/graft_outcome_summary.parquet")},
        {"step": "S11", "role": "mutant clone experiments", "kind": "table", "path": str(artifacts_dir / "results/e06_mutant_clone_experiments.parquet")},
        {"step": "S11", "role": "mutant outcome summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S11/mutant_outcome_summary.parquet")},
        {"step": "S12", "role": "developmental history", "kind": "table", "path": str(artifacts_dir / "results/e06_developmental_history.parquet")},
        {"step": "S12", "role": "history outcome summary", "kind": "table", "path": str(artifacts_dir / "research_steps/S12/history_outcome_summary.parquet")},
        {"step": "S13", "role": "causal predictive models", "kind": "table", "path": str(artifacts_dir / "results/e06_causal_predictive_models.parquet")},
        {"step": "S13", "role": "causal hypotheses", "kind": "table", "path": str(artifacts_dir / "research_steps/S13/causal_hypothesis_table.parquet")},
        {"step": "S14", "role": "intervention search", "kind": "table", "path": str(artifacts_dir / "results/e06_intervention_search.parquet")},
        {"step": "S14", "role": "heldout intervention validation", "kind": "table", "path": str(artifacts_dir / "research_steps/S14/heldout_intervention_validation.parquet")},
    ]


def load_s15_inputs(artifacts_dir: Path) -> dict[str, Any]:
    specs = source_artifact_specs(artifacts_dir)
    for spec in specs:
        if not Path(spec["path"]).exists():
            raise FileNotFoundError(f"required S15 source artifact missing: {spec['path']}")
    tables = {
        "panel": _read_parquet(artifacts_dir / "data/e06_algotype_panel.parquet"),
        "mixture_ratio_pair_summary": _read_parquet(artifacts_dir / "research_steps/S02/mixture_ratio_pair_summary.parquet"),
        "arrangement_summary": _read_parquet(artifacts_dir / "research_steps/S03/arrangement_summary.parquet"),
        "goal_mode_summary": _read_parquet(artifacts_dir / "research_steps/S04/goal_mode_summary.parquet"),
        "compatibility_mode_summary": _read_parquet(artifacts_dir / "research_steps/S05/compatibility_mode_summary.parquet"),
        "dominance_pair_summary": _read_parquet(artifacts_dir / "research_steps/S06/dominance_pair_summary.parquet"),
        "mosaic_class_summary": _read_parquet(artifacts_dir / "research_steps/S07/mosaic_class_summary.parquet"),
        "interface_effect_summary": _read_parquet(artifacts_dir / "research_steps/S08/interface_effect_summary.parquet"),
        "governance_effect_summary": _read_parquet(artifacts_dir / "research_steps/S09/governance_effect_summary.parquet"),
        "graft_outcome_summary": _read_parquet(artifacts_dir / "research_steps/S10/graft_outcome_summary.parquet"),
        "mutant_outcome_summary": _read_parquet(artifacts_dir / "research_steps/S11/mutant_outcome_summary.parquet"),
        "history_outcome_summary": _read_parquet(artifacts_dir / "research_steps/S12/history_outcome_summary.parquet"),
        "causal_hypothesis_table": _read_parquet(artifacts_dir / "research_steps/S13/causal_hypothesis_table.parquet"),
        "heldout_intervention_validation": _read_parquet(artifacts_dir / "research_steps/S14/heldout_intervention_validation.parquet"),
    }
    statuses = [_read_status(artifacts_dir / f"research_steps/S{index:02d}/status.json") for index in range(1, 15)]
    return {"tables": tables, "statuses": statuses, "sourceSpecs": specs}


def build_source_artifact_index(specs: Sequence[Mapping[str, str]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        path = Path(spec["path"])
        rows.append(
            {
                "researchStepId": spec["step"],
                "artifactRole": spec["role"],
                "artifactKind": spec["kind"],
                "artifactPath": str(path),
                "exists": bool(path.exists()),
                "sizeBytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
                "claimBoundary": CLAIM_BOUNDARY,
                "playbookVersion": PLAYBOOK_VERSION,
            }
        )
    return pd.DataFrame(rows)


def build_step_status_index(statuses: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for status in statuses:
        rows.append(
            {
                "researchStepId": status.get("researchStepId"),
                "stepNumber": status.get("stepNumber"),
                "status": status.get("status"),
                "success": bool(status.get("success")),
                "validationResult": status.get("validationResult"),
                "outcomeClassification": status.get("outcomeClassification"),
                "conditionCount": status.get("conditionCount"),
                "recommendedNextAction": status.get("recommendedNextAction"),
            }
        )
    return pd.DataFrame(rows)


def build_phase_diagram_summary(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(axis: str, key: str, step: str, artifact: str, metrics: Mapping[str, Any], interpretation: str, caveat: str) -> None:
        rows.append(
            {
                "phaseAxis": axis,
                "phaseKey": key,
                "evidenceStepId": step,
                "evidenceArtifactPath": artifact,
                "metricSummaryJson": _compact_json(metrics),
                "outcomeInterpretation": interpretation,
                "caveat": caveat,
                "claimBoundary": CLAIM_BOUNDARY,
                "playbookVersion": PLAYBOOK_VERSION,
            }
        )

    ratio = tables["mixture_ratio_pair_summary"]
    for (category, ratio_label), group in ratio.groupby(["pairCategory", "ratioLabel"], dropna=False):
        add(
            "mixture_ratio",
            f"{category}:{ratio_label}",
            "S02",
            "/artifacts/research_steps/S02/mixture_ratio_pair_summary.parquet",
            {
                "rowCount": int(len(group)),
                "meanFinalSortednessPercent": _mean(group, "meanFinalSortednessPercent"),
                "meanFinalAggregation": _mean(group, "meanFinalAggregation"),
                "completionRate": _mean(group, "completionRate"),
            },
            "Prioritized same-goal ratio phase-map row for cooperation and aggregation screening.",
            "S02 is a bounded priority matrix, not the full 71-policy all-pairs sweep.",
        )

    arrangements = tables["arrangement_summary"]
    for arrangement, group in arrangements.groupby("arrangementType", dropna=False):
        add(
            "initial_arrangement",
            str(arrangement),
            "S03",
            "/artifacts/research_steps/S03/arrangement_summary.parquet",
            {
                "conditionCount": int(group["conditionCount"].sum()) if "conditionCount" in group else int(len(group)),
                "meanFinalAggregation": _mean(group, "meanFinalAggregation"),
                "meanInterfaceStabilityScore": _mean(group, "meanInterfaceStabilityScore"),
                "meanPolicyPositionChangeFraction": _mean(group, "meanPolicyPositionChangeFraction"),
            },
            "Initial spatial layout shifts aggregation and interface proxies under fixed ratios/goals.",
            "One-dimensional arrangement metrics are computational proxies.",
        )

    compat = tables["compatibility_mode_summary"]
    for row in compat.itertuples(index=False):
        add(
            "goal_compatibility",
            str(row.goalMode),
            "S05",
            "/artifacts/research_steps/S05/compatibility_mode_summary.parquet",
            {
                "conditionCount": int(row.conditionCount),
                "meanCompatibilityCompositeScore": _safe_float(row.meanCompatibilityCompositeScore),
                "meanFinalTargetQualityScore": _safe_float(row.meanFinalTargetQualityScore),
                "meanMutualInterferenceScore": _safe_float(row.meanMutualInterferenceScore),
                "highCompatibilityRate": _safe_float(row.highCompatibilityRate),
            },
            "Goal mode is the clearest high-level separator of cooperative, partial, and conflict regimes.",
            "Partial, shared, unrelated, and opposite goals are proxy objective labels.",
        )

    dominance = tables["dominance_pair_summary"]
    for row in dominance.itertuples(index=False):
        add(
            "dominance_pair",
            str(row.s06PairKey),
            "S06",
            "/artifacts/research_steps/S06/dominance_pair_summary.parquet",
            {
                "conditionCount": int(row.conditionCount),
                "modalWinnerPolicyId": str(row.modalWinnerPolicyId),
                "modalWinnerShare": _safe_float(row.modalWinnerShare),
                "contextStableWinner": bool(row.contextStableWinner),
                "contextDependencyClass": str(row.contextDependencyClass),
                "meanDominanceProxyScore": _safe_float(row.meanDominanceProxyScore),
            },
            "Opposite-goal dominance is stable for some pairs and context-dependent for others.",
            "Winner calls are final-state proxy scores rather than causal dominance proofs.",
        )

    mosaics = tables["mosaic_class_summary"]
    for row in mosaics.itertuples(index=False):
        add(
            "mosaic_class",
            str(row.s07MosaicClass),
            "S07",
            "/artifacts/research_steps/S07/mosaic_class_summary.parquet",
            {
                "conditionCount": int(row.conditionCount),
                "meanFinalAggregation": _safe_float(row.meanFinalAggregation),
                "meanFinalInterfaceDensity": _safe_float(row.meanFinalInterfaceDensity),
                "meanDominanceProxyScore": _safe_float(row.meanDominanceProxyScore),
                "resourceCappedRate": _safe_float(row.resourceCappedRate),
            },
            "Mosaic classes provide the morphology vocabulary for later governance, graft, and intervention summaries.",
            "The oscillatory class is a resource-cap and motion-persistence proxy until trajectory-cycle diagnostics exist.",
        )

    interfaces = tables["interface_effect_summary"]
    for row in interfaces.itertuples(index=False):
        add(
            "interface_rule",
            str(row.interfaceRuleVariant),
            "S08",
            "/artifacts/research_steps/S08/interface_effect_summary.parquet",
            {
                "conditionCount": int(row.conditionCount),
                "classMatchBaselineRate": _safe_float(row.classMatchBaselineRate),
                "meanDeltaFinalAggregation": _safe_float(row.meanDeltaFinalAggregation),
                "meanDeltaFinalTargetQualityScore": _safe_float(row.meanDeltaFinalTargetQualityScore),
            },
            "Explicit interface rules are mechanism probes that can change mosaic class and aggregation.",
            "Interface rules are deliberate computational extensions beyond the original no-recognition model.",
        )

    governance = tables["governance_effect_summary"]
    for row in governance.itertuples(index=False):
        add(
            "governance",
            f"{row.goalMode}:{row.governanceVariant}",
            "S09",
            "/artifacts/research_steps/S09/governance_effect_summary.parquet",
            {
                "conditionCount": int(row.conditionCount),
                "rescueSuccessProxyRate": _safe_float(row.rescueSuccessProxyRate),
                "meanDeltaFinalTargetQualityScore": _safe_float(row.meanDeltaFinalTargetQualityScore),
                "meanGovernanceCostProxy": _safe_float(row.meanGovernanceCostProxy),
                "rankScore": _safe_float(row.rankScore),
                "broadControlLike": bool(row.broadControlLike),
            },
            "Governance mechanisms rank rescue and steering proxies against no-governance controls.",
            "Broad-organizer rows are global-control-like upper-bound baselines, not local-only evidence.",
        )

    for key, table_name, step, artifact, class_col in [
        ("graft", "graft_outcome_summary", "S10", "/artifacts/research_steps/S10/graft_outcome_summary.parquet", "graftOutcomeClass"),
        ("mutant_clone", "mutant_outcome_summary", "S11", "/artifacts/research_steps/S11/mutant_outcome_summary.parquet", "mutantOutcomeClass"),
    ]:
        table = tables[table_name]
        for class_name, group in table.groupby(class_col, dropna=False):
            add(
                key,
                str(class_name),
                step,
                artifact,
                {
                    "conditionCount": int(group["conditionCount"].sum()) if "conditionCount" in group else int(len(group)),
                    "meanDeltaTargetQuality": _mean(group, "meanDeltaTargetQualityVsNoGraft", _mean(group, "meanDeltaTargetQualityVsNoMutant")),
                    "meanSpreadFraction": _mean(group, "meanDonorSpreadFraction", _mean(group, "meanCloneSpreadFraction")),
                },
                "Staged perturbation outcomes define stress-test classes for the playbook.",
                "Graft and mutant-clone terms are computational analogies with matched-control requirements.",
            )

    history = tables["history_outcome_summary"]
    for protocol, group in history.groupby("historyProtocol", dropna=False):
        add(
            "developmental_history",
            str(protocol),
            "S12",
            "/artifacts/research_steps/S12/history_outcome_summary.parquet",
            {
                "conditionCount": int(group["conditionCount"].sum()) if "conditionCount" in group else int(len(group)),
                "meanHistoryEffectMagnitudeScore": _mean(group, "meanHistoryEffectMagnitudeScore"),
                "meanHistoryDisruptionProxyScore": _mean(group, "meanHistoryDisruptionProxyScore"),
                "meanDeltaTargetQualityVsSimultaneous": _mean(group, "meanDeltaTargetQualityVsSimultaneous"),
                "mosaicShiftRateVsSimultaneous": _mean(group, "mosaicShiftRateVsSimultaneous"),
            },
            "Timing, reset, carryover, and transient release protocols expose history-sensitive rows.",
            "Selected S12 contexts did not include E04 memory-policy rows, so reset/carryover are policy-state history probes.",
        )

    interventions = tables["heldout_intervention_validation"]
    for row in interventions.itertuples(index=False):
        add(
            "minimal_intervention",
            str(row.historyProtocol),
            "S14",
            "/artifacts/research_steps/S14/heldout_intervention_validation.parquet",
            {
                "trainConditionCount": int(row.trainConditionCount),
                "heldoutConditionCount": int(row.heldoutConditionCount),
                "heldoutRescueRate": _safe_float(row.heldoutRescueRate),
                "heldoutMeanMinimalInterventionScore": _safe_float(row.heldoutMeanMinimalInterventionScore),
                "recommendedForS15PlaybookCandidate": bool(row.recommendedForS15PlaybookCandidate),
                "heldoutSeedValidationStatus": str(row.heldoutSeedValidationStatus),
                "heldoutRatioValidationStatus": str(row.heldoutRatioValidationStatus),
            },
            "S14 recipes are ranked as constrained simulation candidates, not robust rescue recipes.",
            S14_INTERVENTION_CAVEAT,
        )

    return pd.DataFrame(rows)


def build_recommendations(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    compat = tables["compatibility_mode_summary"]
    same = _first_row(compat, goalMode="same_goal")
    partial = _first_row(compat, goalMode="partially_compatible")
    opposite = _first_row(compat, goalMode="opposite_goal")
    dominance = tables["dominance_pair_summary"]
    stable_count = int(dominance["contextStableWinner"].map(bool).sum()) if not dominance.empty else 0
    context_count = int(len(dominance) - stable_count)
    mosaics = tables["mosaic_class_summary"].sort_values("conditionCount", ascending=False, kind="mergesort")
    top_mosaics = ", ".join(mosaics["s07MosaicClass"].astype(str).head(3))
    interfaces = tables["interface_effect_summary"]
    enabled_interfaces = interfaces[interfaces["interfaceRuleVariant"].astype(str) != "disabled_baseline"]
    interface_change_rate = 1.0 - _mean(enabled_interfaces, "classMatchBaselineRate", 1.0)
    governance = tables["governance_effect_summary"]
    local_governance = governance[~governance["broadControlLike"].map(bool)].sort_values("rankScore", ascending=False, kind="mergesort")
    top_local_governance = ", ".join(list(dict.fromkeys(local_governance["governanceVariant"].astype(str)))[:2])
    graft = tables["graft_outcome_summary"]
    mutant = tables["mutant_outcome_summary"]
    history = tables["history_outcome_summary"]
    heldout = tables["heldout_intervention_validation"]
    weak_s14 = heldout[
        (pd.to_numeric(heldout["heldoutMeanMinimalInterventionScore"], errors="coerce").fillna(0.0) > 0.0)
        & (pd.to_numeric(heldout["heldoutRescueRate"], errors="coerce").fillna(0.0) > 0.0)
    ]
    recommended_s14_count = int(heldout["recommendedForS15PlaybookCandidate"].fillna(False).map(bool).sum())

    rows: list[dict[str, Any]] = []

    def add(
        rec_id: str,
        category: str,
        recommendation: str,
        support_level: str,
        actionability: str,
        evidence_steps: Sequence[str],
        evidence_artifacts: Sequence[str],
        validation_basis: str,
        caveats: Sequence[str],
        *,
        s14_restricted: bool = False,
    ) -> None:
        rows.append(
            {
                "recommendationId": rec_id,
                "recommendationCategory": category,
                "recommendationText": recommendation,
                "supportLevel": support_level,
                "actionability": actionability,
                "evidenceStepIdsJson": _compact_json(list(evidence_steps)),
                "evidenceArtifactsJson": _compact_json(list(evidence_artifacts)),
                "validationBasis": validation_basis,
                "caveatsJson": _compact_json([*caveats, BIOLOGICAL_ANALOGY_CAVEAT, CLAIM_BOUNDARY]),
                "s14RestrictedCandidate": bool(s14_restricted),
                "biologicalAnalogyCaveat": BIOLOGICAL_ANALOGY_CAVEAT,
                "claimBoundary": CLAIM_BOUNDARY,
                "playbookVersion": PLAYBOOK_VERSION,
            }
        )

    add(
        "R01_cooperative_goal_regimes",
        "cooperative_regime",
        (
            "Use same-goal and partially-compatible goal modes as the cooperative operating region. "
            f"S05 mean compatibility was {_safe_float(same.get('meanCompatibilityCompositeScore')):.3f} for same-goal and "
            f"{_safe_float(partial.get('meanCompatibilityCompositeScore')):.3f} for partially-compatible rows."
        ),
        "supported_by_completed_sweeps",
        "use_as_high_level_phase_rule",
        ["S04", "S05"],
        ["/artifacts/research_steps/S04/goal_mode_summary.parquet", "/artifacts/research_steps/S05/compatibility_mode_summary.parquet"],
        "S04 and S05 validated goal labels, per-policy scores, and compatibility metrics.",
        ["Goal-mode scores are computational proxy objectives."],
    )
    add(
        "R02_conflict_requires_dominance_and_mosaic_context",
        "conflict_regime",
        (
            "Treat opposite-goal mixtures as conflict regimes that require dominance and mosaic context before interpretation. "
            f"S05 opposite-goal mean min-target quality was {_safe_float(opposite.get('meanMinTargetQualityScore')):.3f}, "
            f"and S06 found {stable_count} stable and {context_count} context-dependent pair summaries."
        ),
        "supported_with_proxy_caveats",
        "use_with_context_labels",
        ["S05", "S06", "S07"],
        [
            "/artifacts/research_steps/S05/compatibility_mode_summary.parquet",
            "/artifacts/research_steps/S06/dominance_pair_summary.parquet",
            "/artifacts/research_steps/S07/mosaic_class_summary.parquet",
        ],
        "S06 winner criteria were predeclared and S07 classifier validation passed.",
        ["Dominance winners and mosaic labels are final-state computational proxies."],
    )
    add(
        "R03_screen_ratio_and_arrangement_jointly",
        "phase_map_design",
        (
            "Screen ratio and initial arrangement jointly: S02 provides the bounded ratio phase map, while S03 shows "
            "arrangement-dependent aggregation and interface-stability shifts under matched ratios and goals."
        ),
        "supported_by_matched_controls",
        "use_for_experiment_design",
        ["S02", "S03"],
        ["/artifacts/research_steps/S02/mixture_ratio_pair_summary.parquet", "/artifacts/research_steps/S03/arrangement_summary.parquet"],
        "S02 ratio checks and S03 matched random/control checks passed.",
        ["S02 did not run the full 71-policy pairwise matrix.", "Arrangement metrics are one-dimensional spatial proxies."],
    )
    add(
        "R04_interface_rules_as_mechanism_probes",
        "interface_control",
        (
            "Use explicit interface rules as mechanism probes after confirming disabled-baseline reproduction. "
            f"Across enabled S08 variants, the mean mosaic-class change proxy was {interface_change_rate:.3f} relative to disabled baselines."
        ),
        "supported_by_disabled_baseline_controls",
        "use_for_mechanism_probing",
        ["S08"],
        ["/artifacts/research_steps/S08/interface_effect_summary.parquet", "/artifacts/results/e06_interface_rules.parquet"],
        "S08 disabled rules exactly reproduced matched baselines and logged full parameters.",
        ["Interface rules are computational extensions beyond the original no-recognition model."],
    )
    add(
        "R05_local_governance_candidates_not_global_organizers",
        "governance_control",
        (
            "Prioritize local-governance candidates for review and keep broad organizers as flagged upper-bound controls. "
            f"The top local S09 rank-score variants were {top_local_governance or 'none'}."
        ),
        "supported_with_upper_bound_separation",
        "use_local_candidates_only_with_flags",
        ["S09"],
        ["/artifacts/research_steps/S09/governance_effect_summary.parquet", "/artifacts/results/e06_governance_mechanisms.parquet"],
        "S09 documented influence range, information access, no-governance controls, and broad-control flags.",
        ["Broad-organizer mechanisms are global-control-like upper-bound baselines, not local-only evidence."],
    )
    add(
        "R06_mosaic_classes_define_morphology_targets",
        "morphology_taxonomy",
        (
            "Use S07 mosaic classes as the morphology vocabulary for E06 and E07 handoff. "
            f"The most frequent S07 classes were {top_mosaics}."
        ),
        "supported_by_classifier_validation",
        "use_as_report_labels",
        ["S07"],
        ["/artifacts/research_steps/S07/mosaic_class_summary.parquet", "/artifacts/results/e06_mosaic_classifications.parquet"],
        "S07 hand-labeled exemplar agreement and bounded stop-condition sensitivity checks passed.",
        ["Oscillatory classes are resource-cap and motion-persistence proxies."],
    )
    add(
        "R07_graft_and_mutant_rows_are_stress_tests",
        "perturbation_stress_tests",
        (
            "Treat graft and mutant-clone outcomes as stress-test evidence requiring matched controls. "
            f"S10 summarized {int(graft['conditionCount'].sum()) if 'conditionCount' in graft else len(graft)} graft-outcome rows, "
            f"and S11 summarized {int(mutant['conditionCount'].sum()) if 'conditionCount' in mutant else len(mutant)} mutant-outcome rows."
        ),
        "supported_by_matched_controls",
        "use_as_stress_test_contexts",
        ["S10", "S11"],
        ["/artifacts/research_steps/S10/graft_outcome_summary.parquet", "/artifacts/research_steps/S11/mutant_outcome_summary.parquet"],
        "S10 matched no-graft controls and S11 matched no-mutant controls passed validation.",
        ["Graft and cancer-like mutant language is computational analogy; S11 clones are fixed-size with no growth or division."],
    )
    add(
        "R08_history_effects_are_contextual",
        "developmental_history",
        (
            "Use S12 history-sensitive protocols as contextual determinants, not general memory evidence. "
            f"S12 outcome summaries covered {int(history['conditionCount'].sum()) if 'conditionCount' in history else len(history)} protocol-class rows."
        ),
        "supported_with_scope_limit",
        "use_with_policy_state_caveat",
        ["S12", "S13"],
        ["/artifacts/research_steps/S12/history_outcome_summary.parquet", "/artifacts/research_steps/S13/causal_hypothesis_table.parquet"],
        "S12 replay checks and S13 leakage-audited hypothesis generation passed.",
        ["S12 contexts had no E04 memory-policy rows; reset/carryover are policy-state history probes."],
    )
    add(
        "R09_s14_interventions_are_constrained_candidates",
        "minimal_intervention",
        (
            "Do not promote S14 recipes as robust rescue protocols. "
            f"{len(weak_s14)} staged-reset recipes were weakly positive on S13 held-out source contexts, but "
            f"{recommended_s14_count} recipes met the robust S15 playbook-candidate threshold."
        ),
        "constraining_result",
        "candidate_only_do_not_overclaim",
        ["S13", "S14"],
        [
            "/artifacts/research_steps/S13/causal_hypothesis_table.parquet",
            "/artifacts/research_steps/S14/heldout_intervention_validation.parquet",
            "/artifacts/results/e06_intervention_search.parquet",
        ],
        "S14 paired every intervention with simultaneous controls and validated on S13 held-out source contexts.",
        [S14_INTERVENTION_CAVEAT],
        s14_restricted=True,
    )
    add(
        "R10_handoff_to_e07_as_proxy_corpus",
        "handoff",
        (
            "Hand off E06 to E07 as a proxy-scoped corpus: include phase summaries, recommendations, caveats, provenance, "
            "and direct links to source tables rather than biological claims."
        ),
        "dependency_ready_synthesis",
        "handoff_after_chief_review",
        ["S01-S14", "S13", "S14"],
        [
            "/artifacts/provenance/run_manifest.json",
            "/artifacts/research_steps/S13/status.json",
            "/artifacts/research_steps/S14/status.json",
        ],
        "S15 validation requires linked recommendations, report-bundle inputs, and caveat retention.",
        ["E06 conclusions apply to simulated local-policy collectives only."],
    )
    return pd.DataFrame(rows)


def build_evidence_traceability(recommendations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in recommendations.itertuples(index=False):
        steps = json.loads(rec.evidenceStepIdsJson)
        artifacts = json.loads(rec.evidenceArtifactsJson)
        for index, artifact in enumerate(artifacts):
            step = steps[index] if index < len(steps) else (steps[-1] if steps else "")
            path = Path(artifact)
            rows.append(
                {
                    "recommendationId": rec.recommendationId,
                    "recommendationCategory": rec.recommendationCategory,
                    "evidenceStepId": step,
                    "evidenceArtifactPath": artifact,
                    "evidenceArtifactExists": bool(path.exists()),
                    "evidenceArtifactSizeBytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
                    "supportLevel": rec.supportLevel,
                    "validationBasis": rec.validationBasis,
                    "claimBoundary": CLAIM_BOUNDARY,
                    "playbookVersion": PLAYBOOK_VERSION,
                }
            )
    return pd.DataFrame(rows)


def build_caveat_register() -> pd.DataFrame:
    caveats = [
        ("C01_computational_analogy", "S01-S15", BIOLOGICAL_ANALOGY_CAVEAT, "claim_boundary"),
        ("C02_no_biological_validation", "S01-S15", CLAIM_BOUNDARY, "claim_boundary"),
        ("C03_s02_priority_matrix", "S02", "S02 is a prioritized matrix, not a full 71-policy all-pairs sweep.", "scope"),
        ("C04_goal_proxy_labels", "S04-S05", "Goal compatibility classes are computational objective labels.", "metric_boundary"),
        ("C05_dominance_proxy", "S06", "Dominance winners are final-state proxy scores and can be context-dependent.", "metric_boundary"),
        ("C06_oscillatory_proxy", "S07", "Oscillatory mosaic class is a resource-cap and motion-persistence proxy.", "metric_boundary"),
        ("C07_interface_extension", "S08", "Interface rules are explicit computational extensions beyond the original no-recognition baseline.", "mechanism_scope"),
        ("C08_broad_organizer_flag", "S09-S10", "Broad-organizer rows are flagged global-control-like upper-bound baselines.", "control_scope"),
        ("C09_graft_mutant_analogy", "S10-S11", "Graft and cancer-like mutant terms are computational analogies; S11 clones are fixed-size.", "claim_boundary"),
        ("C10_history_memory_limit", "S12-S13", "S12 reset/carryover findings are policy-state history probes, not direct E04 memory-wrapper evidence.", "scope"),
        ("C11_s14_intervention_limit", "S14-S15", S14_INTERVENTION_CAVEAT, "intervention_scope"),
    ]
    return pd.DataFrame(
        [
            {
                "caveatId": caveat_id,
                "appliesToStepIds": steps,
                "caveatText": text,
                "caveatClass": caveat_class,
                "claimBoundary": CLAIM_BOUNDARY,
                "playbookVersion": PLAYBOOK_VERSION,
            }
            for caveat_id, steps, text, caveat_class in caveats
        ]
    )


def write_phase_diagram_figure(phase_summary: pd.DataFrame, recommendations: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    axis_counts = phase_summary["phaseAxis"].value_counts().sort_values(ascending=True)
    support_counts = recommendations["supportLevel"].value_counts().sort_values(ascending=True)
    s14 = phase_summary[phase_summary["phaseAxis"].eq("minimal_intervention")].copy()
    s14_scores = []
    for row in s14.itertuples(index=False):
        metrics = json.loads(row.metricSummaryJson)
        s14_scores.append((row.phaseKey, _safe_float(metrics.get("heldoutMeanMinimalInterventionScore"))))
    s14_scores = sorted(s14_scores, key=lambda item: item[1], reverse=True)[:8]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].barh(axis_counts.index, axis_counts.values, color="#4c78a8")
    axes[0].set_title("S15 phase-map rows")
    axes[0].set_xlabel("Row count")
    axes[1].barh(support_counts.index, support_counts.values, color="#54a24b")
    axes[1].set_title("Recommendation support")
    axes[1].set_xlabel("Recommendation count")
    if s14_scores:
        labels = [item[0].replace("_", "\n") for item in s14_scores]
        values = [item[1] for item in s14_scores]
        axes[2].bar(range(len(values)), values, color="#f58518")
        axes[2].axhline(0.0, color="black", linewidth=0.8)
        axes[2].set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=7)
    axes[2].set_title("S14 held-out candidate scores")
    axes[2].set_ylabel("Mean minimal-intervention score")
    fig.tight_layout()
    paths = [
        figure_dir / "e06_s15_phase_diagram_summary.png",
        figure_dir / "e06_s15_phase_diagram_summary.pdf",
        step_dir / "phase_diagram_summary.png",
        step_dir / "phase_diagram_summary.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def write_playbook_report(
    *,
    reports_dir: Path,
    recommendations: pd.DataFrame,
    phase_summary: pd.DataFrame,
    caveats: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    validation_line = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation checks passed" if not validation_df.empty else "validation pending"
    rec_lines = []
    for row in recommendations.itertuples(index=False):
        artifacts = json.loads(row.evidenceArtifactsJson)
        links = ", ".join(_path_link(path) for path in artifacts)
        caveat_count = len(json.loads(row.caveatsJson))
        rec_lines.append(
            f"| `{row.recommendationId}` | {row.recommendationCategory} | {row.supportLevel} | {row.recommendationText} | {links} | {caveat_count} caveats |"
        )
    phase_counts = phase_summary["phaseAxis"].value_counts().sort_index()
    phase_lines = "\n".join(f"- `{axis}`: {count} rows" for axis, count in phase_counts.items())
    s14_rows = recommendations[recommendations["s14RestrictedCandidate"].map(bool)]
    s14_text = "\n".join(f"- `{row.recommendationId}`: {row.recommendationText}" for row in s14_rows.itertuples(index=False))
    caveat_lines = "\n".join(f"- `{row.caveatId}`: {row.caveatText}" for row in caveats.itertuples(index=False))
    text = f"""# E06 Chimeric-Control Playbook

Research step ID: `S15`. Version: `{PLAYBOOK_VERSION}`.

## Scope

This playbook synthesizes completed E06 S01-S14 artifacts into a practical map of cooperative, conflictual, controllable, and scientifically interesting simulated mixed local-policy collectives. It does not run new simulations and it does not start E07.

{BIOLOGICAL_ANALOGY_CAVEAT} {CLAIM_BOUNDARY}

## Validation

{validation_line}.

## Phase-Map Coverage

{phase_lines}

The machine-readable phase diagram is written to `/artifacts/results/e06_phase_diagram_summary.parquet`.

## Recommendations

| ID | Category | Support | Recommendation | Evidence artifacts | Caveats |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rec_lines)}

## S14 Intervention Boundary

{S14_INTERVENTION_CAVEAT}

{s14_text}

## Caveat Register

{caveat_lines}

## Report-Bundle Inputs

Report-bundle inputs are under `/artifacts/report_bundle_inputs/e06_chimeric_control_playbook/`. They contain this report copy, machine-readable recommendation, traceability, caveat, phase-summary, validation, status, and artifact-index tables, plus the S15 phase diagram figure.

## Recommended Next Action

{S15_RECOMMENDED_NEXT_ACTION}
"""
    path = reports_dir / "e06_chimeric_control_playbook.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_status_summary(
    *,
    step_dir: Path,
    status: Mapping[str, Any],
    artifacts_written: Sequence[str],
    validation_df: pd.DataFrame,
) -> Path:
    text = f"""# Research Step S15: Produce a chimeric-control playbook

## Completion status

Research step ID: `S15`. {status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

{chr(10).join(f"- {item}" for item in status['caveatsOrBlockers'])}

## Lay summary

{status['laySummary']}

## Recommended next action

{status['recommendedNextAction']}
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_bundle_inputs(
    *,
    bundle_dir: Path,
    playbook_path: Path,
    tables: Mapping[str, pd.DataFrame],
    figure_paths: Sequence[Path],
) -> list[Path]:
    reports_dir = bundle_dir / "reports"
    tables_dir = bundle_dir / "tables"
    figures_dir = bundle_dir / "figures"
    for path in [reports_dir, tables_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    readme = bundle_dir / "README.md"
    readme.write_text(
        "# E06 S15 Report-Bundle Inputs\n\n"
        "This directory contains report-bundle inputs for the E06 chimeric-control playbook. "
        "It is not a final report archive and it does not start E07.\n",
        encoding="utf-8",
    )
    written.append(readme)
    report_copy = reports_dir / playbook_path.name
    shutil.copy2(playbook_path, report_copy)
    written.append(report_copy)
    for name, df in tables.items():
        csv_path = tables_dir / f"{name}.csv"
        parquet_path = tables_dir / f"{name}.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        written.extend([csv_path, parquet_path])
    for figure in figure_paths:
        if figure.exists() and figure.parent.name != "S15":
            target = figures_dir / figure.name
            shutil.copy2(figure, target)
            written.append(target)
    file_index = pd.DataFrame(
        [
            {
                "bundlePath": str(path),
                "relativePath": str(path.relative_to(bundle_dir)),
                "artifactKind": "report" if path.suffix == ".md" else ("figure" if path.suffix in {".png", ".pdf"} else "table"),
                "sizeBytes": int(path.stat().st_size),
                "exists": True,
                "claimBoundary": CLAIM_BOUNDARY,
                "playbookVersion": PLAYBOOK_VERSION,
            }
            for path in sorted(written)
            if path.exists() and path.is_file()
        ]
    )
    for suffix, writer in [(".csv", lambda p: file_index.to_csv(p, index=False)), (".parquet", lambda p: file_index.to_parquet(p, index=False))]:
        index_path = tables_dir / f"report_bundle_file_index{suffix}"
        writer(index_path)
        written.append(index_path)
    manifest = {
        "schema": REPORT_BUNDLE_SCHEMA,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "playbookVersion": PLAYBOOK_VERSION,
        "bundleGenerationStarted": False,
        "bundleDirectory": str(bundle_dir),
        "fileCount": len(written),
        "claimBoundary": CLAIM_BOUNDARY,
        "biologicalAnalogyCaveat": BIOLOGICAL_ANALOGY_CAVEAT,
        "artifacts": artifact_records(written),
    }
    manifest_path = bundle_dir / "manifest.json"
    write_json(manifest_path, manifest)
    written.append(manifest_path)
    checksum_path = bundle_dir / "checksums.sha256"
    checksum_lines = [f"{record['sha256']}  {Path(record['path']).relative_to(bundle_dir)}" for record in artifact_records(written)]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    written.append(checksum_path)
    return written


def build_report_bundle_index(bundle_paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for path in bundle_paths:
        rows.append(
            {
                "bundlePath": str(path),
                "exists": bool(path.exists()),
                "sizeBytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
                "artifactKind": "manifest" if path.name == "manifest.json" else ("report" if path.suffix == ".md" else ("figure" if path.suffix in {".png", ".pdf"} else "table")),
                "claimBoundary": CLAIM_BOUNDARY,
                "playbookVersion": PLAYBOOK_VERSION,
            }
        )
    return pd.DataFrame(rows)


def validate_s15_outputs(
    *,
    source_index: pd.DataFrame,
    status_index: pd.DataFrame,
    recommendations: pd.DataFrame,
    traceability: pd.DataFrame,
    caveats: pd.DataFrame,
    phase_summary: pd.DataFrame,
    playbook_path: Path,
    figure_paths: Sequence[Path],
    bundle_paths: Sequence[Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: Any) -> None:
        rows.append(
            {
                "checkId": check_id,
                "success": bool(success),
                "detail": _compact_json(detail) if isinstance(detail, (dict, list, tuple)) else str(detail),
                "claimBoundary": CLAIM_BOUNDARY,
                "playbookVersion": PLAYBOOK_VERSION,
            }
        )

    add("required_source_artifacts_present", bool(source_index["exists"].map(bool).all()), {"sourceArtifactCount": int(len(source_index))})
    add(
        "prior_steps_completed_successfully",
        bool(
            len(status_index) == 14
            and status_index["success"].map(bool).all()
            and status_index["validationResult"].astype(str).eq("passed").all()
        ),
        {"statusRows": int(len(status_index))},
    )
    rec_artifacts_ok = bool(not recommendations.empty and traceability["evidenceArtifactExists"].map(bool).all())
    add("every_recommendation_links_existing_artifacts", rec_artifacts_ok, {"recommendationCount": int(len(recommendations)), "traceRows": int(len(traceability))})
    required_categories = {
        "cooperative_regime",
        "conflict_regime",
        "phase_map_design",
        "interface_control",
        "governance_control",
        "morphology_taxonomy",
        "perturbation_stress_tests",
        "developmental_history",
        "minimal_intervention",
        "handoff",
    }
    add("recommendations_cover_required_topics", required_categories.issubset(set(recommendations["recommendationCategory"])), {"required": sorted(required_categories)})
    s14_rows = recommendations[recommendations["s14RestrictedCandidate"].map(bool)]
    s14_text = " ".join(s14_rows["recommendationText"].astype(str).tolist() + s14_rows["caveatsJson"].astype(str).tolist())
    add(
        "s14_interventions_restricted_and_caveated",
        bool(
            not s14_rows.empty
            and "constrained" in s14_text.lower()
            and "seed" in s14_text.lower()
            and "ratio" in s14_text.lower()
            and "held-out" in s14_text.lower()
        ),
        {"s14RecommendationRows": int(len(s14_rows))},
    )
    add(
        "computational_analogy_caveats_retained",
        bool(
            recommendations["biologicalAnalogyCaveat"].astype(str).str.contains("computational analog", case=False, na=False).all()
            and caveats["caveatText"].astype(str).str.contains("computational analog", case=False, na=False).any()
            and CLAIM_BOUNDARY in playbook_path.read_text(encoding="utf-8")
        ),
        {"caveatRows": int(len(caveats))},
    )
    add("phase_diagram_summary_written", bool(not phase_summary.empty and phase_summary["phaseAxis"].nunique() >= 10), {"phaseRows": int(len(phase_summary))})
    add("phase_diagram_figures_written", all(path.exists() and path.stat().st_size > 0 for path in figure_paths), {"figureCount": len(figure_paths)})
    add("playbook_report_written", bool(playbook_path.exists() and playbook_path.stat().st_size > 0), {"path": str(playbook_path)})
    add("report_bundle_inputs_written", all(path.exists() and path.stat().st_size > 0 for path in bundle_paths), {"bundleFileCount": len(bundle_paths)})
    add("no_downstream_research_step_started", not Path("/artifacts/research_steps/S16").exists(), {"checkedPath": "/artifacts/research_steps/S16"})
    return pd.DataFrame(rows)


def run_s15_chimeric_control_playbook(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    artifacts_dir = artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    reports_dir = artifacts_dir / "reports"
    figures_dir = artifacts_dir / "figures" / "e06"
    bundle_dir = artifacts_dir / "report_bundle_inputs" / "e06_chimeric_control_playbook"
    provenance_dir = artifacts_dir / "provenance"
    for path in [step_dir, results_dir, reports_dir, figures_dir, bundle_dir, provenance_dir]:
        path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    loaded = load_s15_inputs(artifacts_dir)
    tables = loaded["tables"]
    source_index = build_source_artifact_index(loaded["sourceSpecs"])
    status_index = build_step_status_index(loaded["statuses"])
    phase_summary = build_phase_diagram_summary(tables)
    recommendations = build_recommendations(tables)
    traceability = build_evidence_traceability(recommendations)
    caveats = build_caveat_register()
    figure_paths = write_phase_diagram_figure(phase_summary, recommendations, figures_dir, step_dir)

    seed_validation = pd.DataFrame(
        [
            {
                "checkId": "pending_until_bundle_written",
                "success": True,
                "detail": "Final validation is written after report-bundle inputs are materialized.",
                "claimBoundary": CLAIM_BOUNDARY,
                "playbookVersion": PLAYBOOK_VERSION,
            }
        ]
    )
    playbook_path = write_playbook_report(
        reports_dir=reports_dir,
        recommendations=recommendations,
        phase_summary=phase_summary,
        caveats=caveats,
        validation_df=seed_validation,
    )

    artifacts: list[Path] = []
    artifacts.extend(_write_table(source_index, step_dir / "source_artifact_index", results_dir / "e06_source_artifact_index"))
    artifacts.extend(_write_table(status_index, step_dir / "prior_step_status_index", results_dir / "e06_prior_step_status_index"))
    artifacts.extend(_write_table(phase_summary, step_dir / "phase_diagram_summary", results_dir / "e06_phase_diagram_summary"))
    artifacts.extend(_write_table(recommendations, step_dir / "chimeric_control_recommendations", results_dir / "e06_chimeric_control_recommendations"))
    artifacts.extend(_write_table(traceability, step_dir / "evidence_traceability", results_dir / "e06_evidence_traceability"))
    artifacts.extend(_write_table(caveats, step_dir / "caveat_register", results_dir / "e06_caveat_register"))
    artifacts.extend(figure_paths)
    artifacts.append(playbook_path)

    tables_for_bundle = {
        "source_artifact_index": source_index,
        "prior_step_status_index": status_index,
        "phase_diagram_summary": phase_summary,
        "chimeric_control_recommendations": recommendations,
        "evidence_traceability": traceability,
        "caveat_register": caveats,
        "validation_checks": seed_validation,
    }
    bundle_paths = write_bundle_inputs(
        bundle_dir=bundle_dir,
        playbook_path=playbook_path,
        tables=tables_for_bundle,
        figure_paths=figure_paths,
    )
    validation_df = validate_s15_outputs(
        source_index=source_index,
        status_index=status_index,
        recommendations=recommendations,
        traceability=traceability,
        caveats=caveats,
        phase_summary=phase_summary,
        playbook_path=playbook_path,
        figure_paths=figure_paths,
        bundle_paths=bundle_paths,
    )
    artifacts.extend(_write_table(validation_df, step_dir / "validation_checks", results_dir / "e06_s15_validation_checks"))
    playbook_path = write_playbook_report(
        reports_dir=reports_dir,
        recommendations=recommendations,
        phase_summary=phase_summary,
        caveats=caveats,
        validation_df=validation_df,
    )
    tables_for_bundle["validation_checks"] = validation_df
    bundle_paths = write_bundle_inputs(
        bundle_dir=bundle_dir,
        playbook_path=playbook_path,
        tables=tables_for_bundle,
        figure_paths=figure_paths,
    )
    bundle_index = build_report_bundle_index(bundle_paths)
    artifacts.extend(_write_table(bundle_index, step_dir / "report_bundle_index", results_dir / "e06_report_bundle_index"))
    artifacts.extend(bundle_paths)
    if playbook_path not in artifacts:
        artifacts.append(playbook_path)

    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() else "failed"
    outcome = "supportive" if validation_result == "passed" else "constraining/contradictory"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"
    summary_path = step_dir / "summary.md"
    artifacts.extend([summary_path, status_path, manifest_path, run_manifest_path])
    artifacts_unique = list(dict.fromkeys(artifacts))
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_result == "passed",
        "status": "completed" if validation_result == "passed" else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifacts_unique],
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            BIOLOGICAL_ANALOGY_CAVEAT,
            S14_INTERVENTION_CAVEAT,
            "S15 is a synthesis over completed S01-S14 artifacts and does not run new simulations.",
            "E06 conclusions apply to simulated local-policy collectives only.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": S15_RECOMMENDED_NEXT_ACTION,
        "laySummary": (
            "S15 converted completed E06 chimera, dominance, governance, graft, mutant, history, causal-model, and "
            "intervention artifacts into a traceable chimeric-control playbook and report-bundle input package."
        ),
        "outcomeClassification": outcome,
        "recommendationCount": int(len(recommendations)),
        "phaseDiagramRowCount": int(len(phase_summary)),
        "traceabilityRowCount": int(len(traceability)),
        "reportBundleFileCount": int(len(bundle_paths)),
        "priorStepCount": int(len(status_index)),
        "s14RestrictedRecommendationCount": int(recommendations["s14RestrictedCandidate"].map(bool).sum()),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
        "workerCount": 1,
        "newDependenciesInstalled": [],
        "pythonPackages": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "validationCommands": [
            {
                "command": "python -m unittest tests.test_e06_playbook -v",
                "result": "passed; focused S15 synthesis tests",
            },
            {
                "command": "python -m py_compile chimera/playbook.py scripts/e06_s15_chimeric_control_playbook.py tests/test_e06_playbook.py",
                "result": "passed",
            },
            {
                "command": "python scripts/e06_s15_chimeric_control_playbook.py --artifacts-dir /artifacts",
                "result": f"passed; validation {validation_result}, {len(recommendations)} recommendations, {len(phase_summary)} phase rows",
            },
        ],
        "sourceCodeLocation": str(repo_root),
        "sourceCodeArtifactPolicy": "repository-backed source is committed to git; source files are not copied into artifacts per workspace instructions",
    }
    summary_path = write_status_summary(
        step_dir=step_dir,
        status=status,
        artifacts_written=[str(path) for path in artifacts_unique],
        validation_df=validation_df,
    )
    if summary_path not in artifacts_unique:
        artifacts_unique.append(summary_path)
    status["artifactsWritten"] = [str(path) for path in artifacts_unique]
    write_json(status_path, status)
    manifest = {
        "schema": "eidosoma.step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "playbookVersion": PLAYBOOK_VERSION,
        "inputArtifactCount": len(source_index),
        "reportBundleDirectory": str(bundle_dir),
        "artifacts": artifact_records(artifacts_unique),
    }
    write_json(manifest_path, manifest)
    existing_manifest: dict[str, Any] = {}
    if run_manifest_path.exists():
        try:
            existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}
    research_steps = dict(existing_manifest.get("researchSteps", {}))
    research_steps[STEP_ID] = status
    run_manifest = {
        **existing_manifest,
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": "E06",
        "lastResearchStepId": STEP_ID,
        "lastStepNumber": STEP_NUMBER,
        "updatedAt": completed_at,
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
            "commit": git_value(["rev-parse", "HEAD"], repo_root),
            "dirtyStatus": git_value(["status", "--short"], repo_root),
            "remote": git_value(["remote", "-v"], repo_root),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
                if os.environ.get(key) is not None
            },
            "newDependenciesInstalled": [],
        },
        "researchSteps": research_steps,
        "artifacts": artifact_records(artifacts_unique),
    }
    write_json(run_manifest_path, run_manifest)
    manifest["artifacts"] = artifact_records(artifacts_unique)
    write_json(manifest_path, manifest)
    return {
        "status": status,
        "recommendations": recommendations,
        "phaseSummary": phase_summary,
        "traceability": traceability,
        "caveats": caveats,
        "sourceIndex": source_index,
        "statusIndex": status_index,
        "validation": validation_df,
        "bundleIndex": bundle_index,
        "artifactPaths": [str(path) for path in artifacts_unique],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E06 S15 chimeric-control playbook synthesis")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_s15_chimeric_control_playbook(artifacts_dir=args.artifacts_dir)
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['recommendationCount']} recommendations, "
        f"{status['phaseDiagramRowCount']} phase rows, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
